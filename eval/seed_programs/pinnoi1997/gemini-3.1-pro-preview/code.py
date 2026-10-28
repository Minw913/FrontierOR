import gurobipy as gp
from gurobipy import GRB
import argparse
import json
import math

def solve(instance_path, solution_path, time_limit, log_path):
    # 1. Load instance data
    with open(instance_path, 'r') as f:
        data = json.load(f)
        
    num_tasks = data["num_tasks"]
    cycle_time = data["cycle_time"]
    num_stations = data["num_stations"]
    task_times = data["task_processing_times"]
    arcs = data["precedence_arcs"]
    ES = data["earliest_station"]
    LS = data["latest_station"]
    total_processing_time = data["total_processing_time"]
    
    # 2. Setup Logger
    logger = None
    if log_path:
        try:
            from solution_logger import SolutionLogger
            logger = SolutionLogger(log_path, sense="minimize")
        except ImportError:
            pass

    # 3. Setup Gurobi Model
    m = gp.Model("AssemblyLineBalancing")
    
    # Apply execution constraints
    m.setParam("TimeLimit", time_limit)
    m.setParam("Threads", 1)
    # MIPFocus 1 aims to find high-quality feasible solutions quickly
    m.setParam("MIPFocus", 1)  

    # Define Callback for logger
    cb = None
    if logger:
        def callback(model, where):
            if where == GRB.Callback.MIPSOL:
                obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                logger.log(obj)
        cb = callback

    # 4. Variables Definition
    tasks = list(task_times.keys())
    stations = list(range(1, num_stations + 1))
    
    x = {}
    for i in tasks:
        esi = int(ES[i])
        lsi = int(LS[i])
        # Only instantiate variables for feasible assignments bounded by ES and LS
        for j in range(esi, lsi + 1):
            x[i, j] = m.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
            
    # Worst-case station idle time
    Z = m.addVar(vtype=GRB.INTEGER, name="Z", lb=0, ub=cycle_time)
    
    m.setObjective(Z, GRB.MINIMIZE)

    # 5. Constraints Setup
    
    # Exact one station per task
    for i in tasks:
        esi = int(ES[i])
        lsi = int(LS[i])
        m.addConstr(gp.quicksum(x[i, j] for j in range(esi, lsi + 1)) == 1, name=f"assign_{i}")

    # Cycle time limits & defining Maximum Idle Time
    for j in stations:
        workload = gp.quicksum(task_times[i] * x[i, j] for i in tasks if (i, j) in x)
        m.addConstr(workload <= cycle_time, name=f"ct_{j}")
        # Z is at least the idle time at every single station
        m.addConstr(Z >= cycle_time - workload, name=f"idle_{j}")

    # Precedence Relationships
    for u, v in arcs:
        u_str = str(u)
        v_str = str(v)
        
        if u_str not in ES or v_str not in ES:
            continue
            
        esi_u = int(ES[u_str])
        lsi_u = int(LS[u_str])
        esi_v = int(ES[v_str])
        lsi_v = int(LS[v_str])
        
        # Enforce that cumulative assignments of successor `v` across feasible station bounds 
        # do not exceed cumulative assignments of predecessor `u`
        for s in range(esi_v, lsi_u):
            lhs = gp.quicksum(x[v_str, j] for j in range(esi_v, s + 1) if (v_str, j) in x)
            rhs = gp.quicksum(x[u_str, j] for j in range(esi_u, s + 1) if (u_str, j) in x)
            m.addConstr(lhs <= rhs, name=f"prec_{u_str}_{v_str}_{s}")

    # Tight Lower Bound constraint (The maximum idle time is strictly bounded by average idle time)
    total_slack = num_stations * cycle_time - total_processing_time
    m.addConstr(Z >= math.ceil(total_slack / num_stations), name="Z_lb_tight")

    # 6. Optimize Model
    if cb:
        m.optimize(cb)
    else:
        m.optimize()

    # 7. Collect and Export Solution
    if m.SolCount > 0:
        obj_val = float(m.ObjVal)
        assignments = {}
        for i in tasks:
            esi = int(ES[i])
            lsi = int(LS[i])
            for j in range(esi, lsi + 1):
                if x[i, j].X > 0.5:
                    assignments[i] = j
                    break
                    
        solution = {
            "objective_value": obj_val,
            "task_assignments": assignments
        }
    else:
        # Fallback if no solution was found within the time limit
        solution = {
            "objective_value": float(cycle_time),
            "task_assignments": {}
        }

    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Solver for Assembly Line Balancing Problem")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to instance JSON file")
    parser.add_argument("--solution_path", type=str, required=True, help="Path where solution will be written")
    parser.add_argument("--time_limit", type=int, required=True, help="Maximum execution time limit in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Path to track improving solutions log")
    
    args = parser.parse_args()
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)
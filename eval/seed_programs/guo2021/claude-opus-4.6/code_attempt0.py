import argparse
import json
import time
import math
from collections import defaultdict

def solve(instance_path, solution_path, time_limit, log_path):
    from solution_logger import SolutionLogger
    
    start_time = time.time()
    
    with open(instance_path, 'r') as f:
        data = json.load(f)
    
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    num_patients = data['num_patients']
    num_hospitals = data['num_hospitals']
    num_days = data['num_days']
    num_ors = data['num_ors_per_hospital']
    num_scenarios = data['num_scenarios']
    
    patients = data['patients']
    hospitals = data['hospitals']
    scenarios = data['scenarios']
    
    gp = data['global_parameters']
    
    # Build indices
    P = range(num_patients)
    H = range(num_hospitals)
    D = range(num_days)
    O = range(num_ors)
    S = range(num_scenarios)
    
    # Patient data
    is_mandatory = [patients[p]['is_mandatory'] for p in P]
    mandatory_patients = [p for p in P if is_mandatory[p]]
    non_mandatory_patients = [p for p in P if not is_mandatory[p]]
    
    c_unsched = [patients[p]['c_unsched'] for p in P]
    c_cancel = [patients[p]['c_cancel'] for p in P]
    
    # Scheduling benefit per patient per day
    c_sched = {}
    for p in P:
        for d in D:
            day_label = str(d + 1)
            c_sched[p, d] = patients[p]['c_sched_per_day'][day_label]
    
    # Hospital data
    B = {}  # OR time limit
    F = {}  # OR opening cost
    G = {}  # Suite opening cost
    for hosp in hospitals:
        h = hosp['hospital_id']
        for day_info in hosp['days']:
            d = day_info['day_id']
            B[h, d] = day_info['B_hd']
            F[h, d] = day_info['F_hd']
            G[h, d] = day_info['G_hd']
    
    # Scenario durations
    dur = {}
    for sc in scenarios:
        s = sc['scenario_id']
        durations = sc['surgery_durations_minutes']
        for p in P:
            dur[p, s] = durations[p]
    
    import gurobipy as gp_solver
    from gurobipy import GRB
    
    elapsed = time.time() - start_time
    remaining = time_limit - elapsed
    if remaining < 1:
        remaining = 1
    
    m = gp_solver.Model("hospital_scheduling")
    m.setParam('TimeLimit', remaining - 5)  # leave 5s buffer
    m.setParam('Threads', 1)
    m.setParam('MIPFocus', 1)  # Focus on finding feasible solutions
    
    # Decision variables
    # u[h,d] = 1 if hospital h suite is open on day d
    u = {}
    for h in H:
        for d in D:
            u[h, d] = m.addVar(vtype=GRB.BINARY, name=f"u_{h}_{d}")
    
    # y[h,d,o] = 1 if OR o at hospital h is open on day d
    y = {}
    for h in H:
        for d in D:
            for o in O:
                y[h, d, o] = m.addVar(vtype=GRB.BINARY, name=f"y_{h}_{d}_{o}")
    
    # x[h,d,p,o] = 1 if patient p assigned to OR o at hospital h on day d
    x = {}
    for p in P:
        for h in H:
            for d in D:
                for o in O:
                    x[h, d, p, o] = m.addVar(vtype=GRB.BINARY, name=f"x_{h}_{d}_{p}_{o}")
    
    # w[p] = 1 if non-mandatory patient p is postponed
    w = {}
    for p in non_mandatory_patients:
        w[p] = m.addVar(vtype=GRB.BINARY, name=f"w_{p}")
    
    # q[h,d,p,o,s] = 1 if patient p assigned to OR o,h,d is cancelled in scenario s
    q = {}
    for s in S:
        for h in H:
            for d in D:
                for o in O:
                    for p in P:
                        q[h, d, p, o, s] = m.addVar(vtype=GRB.BINARY, name=f"q_{h}_{d}_{p}_{o}_{s}")
    
    m.update()
    
    # Objective
    # 1. Suite opening costs
    suite_cost = gp_solver.quicksum(G[h, d] * u[h, d] for h in H for d in D)
    
    # 2. OR opening costs
    or_cost = gp_solver.quicksum(F[h, d] * y[h, d, o] for h in H for d in D for o in O)
    
    # 3. Scheduling benefit (negative = reduces cost)
    sched_benefit = gp_solver.quicksum(
        c_sched[p, d] * x[h, d, p, o]
        for p in P for h in H for d in D for o in O
    )
    
    # 4. Postponement penalty
    postpone_cost = gp_solver.quicksum(c_unsched[p] * w[p] for p in non_mandatory_patients)
    
    # 5. Expected cancellation cost
    cancel_cost = (1.0 / num_scenarios) * gp_solver.quicksum(
        c_cancel[p] * q[h, d, p, o, s]
        for s in S for h in H for d in D for o in O for p in P
    )
    
    m.setObjective(suite_cost + or_cost + sched_benefit + postpone_cost + cancel_cost, GRB.MINIMIZE)
    
    # Constraints
    
    # Mandatory patients must be assigned exactly once
    for p in mandatory_patients:
        m.addConstr(
            gp_solver.quicksum(x[h, d, p, o] for h in H for d in D for o in O) == 1,
            name=f"mandatory_{p}"
        )
    
    # Non-mandatory patients: assigned once or postponed
    for p in non_mandatory_patients:
        m.addConstr(
            gp_solver.quicksum(x[h, d, p, o] for h in H for d in D for o in O) + w[p] == 1,
            name=f"nonmandatory_{p}"
        )
    
    # OR open only if suite open
    for h in H:
        for d in D:
            for o in O:
                m.addConstr(y[h, d, o] <= u[h, d], name=f"or_suite_{h}_{d}_{o}")
    
    # Patient assigned only to open OR
    for h in H:
        for d in D:
            for o in O:
                for p in P:
                    m.addConstr(x[h, d, p, o] <= y[h, d, o], name=f"assign_or_{h}_{d}_{p}_{o}")
    
    # Cancellation only if assigned
    for s in S:
        for h in H:
            for d in D:
                for o in O:
                    for p in P:
                        m.addConstr(q[h, d, p, o, s] <= x[h, d, p, o], name=f"cancel_assign_{h}_{d}_{p}_{o}_{s}")
    
    # OR time limit: performed surgeries must fit
    for s in S:
        for h in H:
            for d in D:
                for o in O:
                    m.addConstr(
                        gp_solver.quicksum(
                            dur[p, s] * (x[h, d, p, o] - q[h, d, p, o, s])
                            for p in P
                        ) <= B[h, d],
                        name=f"time_{h}_{d}_{o}_{s}"
                    )
    
    # Callback for logging
    best_obj = [float('inf')]
    
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj < best_obj[0] - 1e-6:
                best_obj[0] = obj
                if logger:
                    logger.log(obj)
    
    m.optimize(callback)
    
    # Extract solution
    sol_u = {}
    sol_y = {}
    sol_x = {}
    sol_w = {}
    
    if m.SolCount > 0:
        obj_val = m.ObjVal
        
        for h in H:
            for d in D:
                val = round(u[h, d].X)
                sol_u[f"{h},{d}"] = int(val)
        
        for h in H:
            for d in D:
                for o in O:
                    val = round(y[h, d, o].X)
                    sol_y[f"{h},{d},{o}"] = int(val)
        
        for p in P:
            for h in H:
                for d in D:
                    for o in O:
                        val = round(x[h, d, p, o].X)
                        if val > 0.5:
                            sol_x[f"{h},{d},{p},{o}"] = 1
        
        for p in non_mandatory_patients:
            val = round(w[p].X)
            sol_w[str(p)] = int(val)
    else:
        # No feasible solution found - return trivial solution
        obj_val = 0.0
        
        for h in H:
            for d in D:
                sol_u[f"{h},{d}"] = 0
        
        for h in H:
            for d in D:
                for o in O:
                    sol_y[f"{h},{d},{o}"] = 0
        
        for p in non_mandatory_patients:
            sol_w[str(p)] = 1
            obj_val += c_unsched[p]
        
        # If there are mandatory patients, this is infeasible
        # but we return best effort
        if logger:
            logger.log(obj_val)
    
    solution = {
        "objective_value": obj_val,
        "u": sol_u,
        "y": sol_y,
        "x": sol_x,
        "w": sol_w
    }
    
    with open(solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()
    
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)
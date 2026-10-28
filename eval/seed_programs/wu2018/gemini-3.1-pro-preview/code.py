import argparse
import json
import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None

def solve():
    parser = argparse.ArgumentParser(description="Solve Capacitated Lot-Sizing Problem with Parallel Machines.")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to the JSON instance file.")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to output the solution JSON.")
    parser.add_argument("--time_limit", type=int, required=True, help="Time limit for the solver in seconds.")
    parser.add_argument("--log_path", type=str, default="", help="Path to the log file for intermediate solutions.")
    
    args = parser.parse_args()

    # Load instance data
    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    logger = None
    if args.log_path and SolutionLogger is not None:
        logger = SolutionLogger(args.log_path, sense="minimize")

    num_items = data["num_items"]
    num_periods = data["num_periods"]
    num_machines = data["num_machines"]
    
    # Infer indexing offset from dictionary keys to accommodate both 0-indexed and 1-indexed inputs seamlessly.
    # By default, we assume 1-indexed. We update to 0 if a `0` index is explicitly encountered.
    min_item = 1
    min_mach = 1
    for k in data.get("production_time", {}).keys():
        parts = k.split('_')
        # Expecting format typically like "item_{item}_machine_{machine}"
        if len(parts) >= 4 and parts[0] == "item" and parts[2] == "machine":
            try:
                if int(parts[1]) == 0:
                    min_item = 0
                if int(parts[3]) == 0:
                    min_mach = 0
            except ValueError:
                pass

    if "initial_inventory" in data:
        init_inv = data["initial_inventory"]
    else:
        init_inv = [0.0] * num_items

    # Initialize Gurobi environment and model
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    model = gp.Model("LotSizing", env=env)
    
    # Set parameters: single core requirement & time limit
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', args.time_limit)

    X = {}
    Y = {}
    I = {}

    # Define Variables
    for i in range(num_items):
        for t in range(num_periods):
            I[i, t] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"I_{i}_{t}")
            for m_cap in data["machine_capabilities"][i]:
                m = m_cap - 1  # Standard 0-offset internal indexing for machine
                X[i, m, t] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"X_{i}_{m}_{t}")
                Y[i, m, t] = model.addVar(vtype=GRB.BINARY, name=f"Y_{i}_{m}_{t}")

    # 1. Inventory Balance Constraint
    for i in range(num_items):
        for t in range(num_periods):
            inflow = gp.quicksum(X[i, m_cap - 1, t] for m_cap in data["machine_capabilities"][i])
            prev_inv = init_inv[i] if t == 0 else I[i, t-1]
            model.addConstr(I[i, t] == prev_inv + inflow - data["demand"][i][t], name=f"inv_bal_{i}_{t}")

    # 2. Logic Forcing Setup and Max Production Bound (Big-M restriction)
    for i in range(num_items):
        i_str = i + min_item
        for m_cap in data["machine_capabilities"][i]:
            m = m_cap - 1
            m_str = m + min_mach
            s_t = data["setup_time"][f"item_{i_str}_machine_{m_str}"]
            p_t = data["production_time"][f"item_{i_str}_machine_{m_str}"]
            
            for t in range(num_periods):
                cap = data["capacity"][m][t]
                
                # Check if even a setup is possible within available capacity
                if cap < s_t:
                    model.addConstr(Y[i, m, t] == 0, name=f"no_setup_{i}_{m}_{t}")
                    model.addConstr(X[i, m, t] == 0, name=f"no_prod_{i}_{m}_{t}")
                else:
                    # Enforce the strict upper bounding physical constraint specified by problem description
                    future_demand = sum(data["demand"][i][tau] for tau in range(t, num_periods))
                    max_prod_cap = (cap - s_t) / p_t if p_t > 0 else float('inf')
                    valid_big_m = min(future_demand, max_prod_cap)
                    model.addConstr(X[i, m, t] <= valid_big_m * Y[i, m, t], name=f"setup_forcing_{i}_{m}_{t}")

    # 3. Machine Capacity Constraint
    for m in range(num_machines):
        for t in range(num_periods):
            cap_used = []
            for i in range(num_items):
                i_str = i + min_item
                if (m + 1) in data["machine_capabilities"][i]:
                    m_str = m + min_mach
                    s_t = data["setup_time"][f"item_{i_str}_machine_{m_str}"]
                    p_t = data["production_time"][f"item_{i_str}_machine_{m_str}"]
                    cap_used.append(p_t * X[i, m, t] + s_t * Y[i, m, t])
            
            if cap_used:
                model.addConstr(gp.quicksum(cap_used) <= data["capacity"][m][t], name=f"cap_sum_{m}_{t}")

    # 4. Maximum Active Facilities (Max Machines per Item per Period) Limit
    for i in range(num_items):
        max_m_limit = data["max_machines_per_item_per_period"][i]
        for t in range(num_periods):
            num_setups = gp.quicksum(Y[i, m_cap - 1, t] for m_cap in data["machine_capabilities"][i])
            model.addConstr(num_setups <= max_m_limit, name=f"max_m_{i}_{t}")

    # 5. Objective Function: Minimize overall costs across inventory, productions, and setups
    obj_expr = gp.LinExpr()
    for i in range(num_items):
        i_str = i + min_item
        for t in range(num_periods):
            obj_expr.add(data["holding_cost"][i] * I[i, t])
            for m_cap in data["machine_capabilities"][i]:
                m = m_cap - 1
                m_str = m + min_mach
                s_c = data["setup_cost"][f"item_{i_str}_machine_{m_str}"]
                p_c = data["production_cost"][f"item_{i_str}_machine_{m_str}"]
                obj_expr.add(s_c * Y[i, m, t])
                obj_expr.add(p_c * X[i, m, t])
                
    model.setObjective(obj_expr, GRB.MINIMIZE)

    # Callback routine to log intermediary feasible solutions improving the incumbent
    best_obj = float('inf')
    def optimize_callback(model, where):
        nonlocal best_obj
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj < best_obj - 1e-5:
                best_obj = obj
                if logger:
                    logger.log(obj)

    # Perform solving procedure
    model.optimize(optimize_callback)

    # Process and evaluate JSON outputs strictly following target schema
    if model.SolCount > 0:
        production = {}
        inventory = {}
        setups = {}
        
        for i in range(num_items):
            i_str = i + min_item
            for t in range(num_periods):
                t_str = t + min_item
                
                # Register inventory outcome
                i_val = I[i, t].X
                inventory[f"I_item{i_str}_period{t_str}"] = max(0.0, float(i_val))
                
                for m_cap in data["machine_capabilities"][i]:
                    m = m_cap - 1
                    m_str = m + min_mach
                    
                    x_val = X[i, m, t].X
                    y_val = Y[i, m, t].X
                    
                    # Register production quantity and setup decision 
                    production[f"X_item{i_str}_machine{m_str}_period{t_str}"] = max(0.0, float(x_val))
                    setups[f"Y_item{i_str}_machine{m_str}_period{t_str}"] = 1 if y_val > 0.5 else 0
        
        final_solution = {
            "objective_value": float(model.ObjVal),
            "production": production,
            "inventory": inventory,
            "setups": setups
        }
        
    else:
        # Fallback safeguard in case no solution is retrieved within parameters
        final_solution = {
            "objective_value": 0.0,
            "production": {},
            "inventory": {},
            "setups": {}
        }
        
    with open(args.solution_path, 'w') as f:
        json.dump(final_solution, f, indent=4)

if __name__ == "__main__":
    solve()
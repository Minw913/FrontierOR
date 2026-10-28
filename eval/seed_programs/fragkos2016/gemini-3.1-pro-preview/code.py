import argparse
import json
import gurobipy as gp

try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None

def solve():
    parser = argparse.ArgumentParser(description="Solve Capacitated Lot Sizing Problem.")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to instance JSON file.")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to solution JSON file.")
    parser.add_argument("--time_limit", type=int, required=True, help="Time limit in seconds.")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log intermediate solutions.")
    args = parser.parse_args()

    # Initialize SolutionLogger
    logger = None
    if args.log_path and SolutionLogger is not None:
        logger = SolutionLogger(args.log_path, sense="minimize")

    # Read instance data
    with open(args.instance_path, "r") as f:
        data = json.load(f)
    
    n_items = data["n_items"]
    n_periods = data["n_periods"]
    capacity = data["capacity"]
    initial_inventory = data["initial_inventory"]
    demands = data["demands"]
    setup_costs = data["setup_costs"]
    holding_costs = data["holding_costs"]
    variable_costs = data["variable_costs"]
    setup_times = data["setup_times"]
    variable_times = data["variable_times"]
    big_M = data["big_M"]

    # Initialize model
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 1)
    env.start()
    
    model = gp.Model("CLSP", env=env)
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", args.time_limit)
    
    # Variables
    x_vars = {}
    y_vars = {}
    s_vars = {}
    
    for i in range(n_items):
        for t in range(n_periods + 1):
            if t == 0:
                s_vars[i, t] = model.addVar(lb=initial_inventory[i], ub=initial_inventory[i], name=f"s_{i}_{t}")
            elif t == n_periods:
                s_vars[i, t] = model.addVar(lb=0, ub=0, name=f"s_{i}_{t}")
            else:
                # Inventory at the beginning of period t
                s_vars[i, t] = model.addVar(lb=0, name=f"s_{i}_{t}")

    for i in range(n_items):
        for t in range(n_periods):
            x_vars[i, t] = model.addVar(lb=0, name=f"x_{i}_{t}")
            y_vars[i, t] = model.addVar(vtype=gp.GRB.BINARY, name=f"y_{i}_{t}")
            
    # Objective
    # Holding cost is calculated based on inventory carried from one period to the next (end of period t / beginning of period t+1)
    obj = gp.quicksum(
        setup_costs[i][t] * y_vars[i, t] +
        variable_costs[i][t] * x_vars[i, t] +
        holding_costs[i][t] * s_vars[i, t+1]
        for i in range(n_items) for t in range(n_periods)
    )
    model.setObjective(obj, gp.GRB.MINIMIZE)
    
    # Constraints
    
    # Flow balance / Inventory balance constraints
    for i in range(n_items):
        for t in range(n_periods):
            model.addConstr(
                s_vars[i, t] + x_vars[i, t] - s_vars[i, t+1] == demands[i][t], 
                name=f"flow_{i}_{t}"
            )
            
    # Capacity constraints
    for t in range(n_periods):
        model.addConstr(
            gp.quicksum(setup_times[i][t] * y_vars[i, t] + variable_times[i][t] * x_vars[i, t] for i in range(n_items)) <= capacity[t],
            name=f"cap_{t}"
        )
        
    # Setup constraints
    for i in range(n_items):
        for t in range(n_periods):
            model.addConstr(
                x_vars[i, t] <= big_M[i][t] * y_vars[i, t], 
                name=f"setup_{i}_{t}"
            )
            
    # Callback to log new best solutions
    def cb(mdl, where):
        if where == gp.GRB.Callback.MIPSOL:
            obj_val = mdl.cbGet(gp.GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj_val)

    # Optimize the model
    model.optimize(cb)
    
    # Write solution to JSON
    if model.SolCount > 0:
        objective_value = model.ObjVal
        
        x_out = []
        y_out = []
        s_out = []
        
        for i in range(n_items):
            x_row = []
            y_row = []
            s_row = []
            for t in range(n_periods + 1):
                s_row.append(max(0.0, float(s_vars[i, t].X)))
                if t < n_periods:
                    x_row.append(max(0.0, float(x_vars[i, t].X)))
                    y_row.append(int(round(y_vars[i, t].X)))
            
            x_out.append(x_row)
            y_out.append(y_row)
            s_out.append(s_row)
            
        solution = {
            "objective_value": objective_value,
            "x": x_out,
            "s": s_out,
            "y": y_out
        }
        
        with open(args.solution_path, "w") as f:
            json.dump(solution, f, indent=2)

if __name__ == "__main__":
    solve()
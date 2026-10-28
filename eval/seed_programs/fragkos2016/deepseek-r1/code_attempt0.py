import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    # Read instance data
    with open(args.instance_path, 'r') as f:
        data = json.load(f)
    
    n_items = data['n_items']
    n_periods = data['n_periods']
    capacity = data['capacity']
    initial_inventory = data['initial_inventory']
    demands = data['demands']
    setup_costs = data['setup_costs']
    holding_costs = data['holding_costs']
    variable_costs = data['variable_costs']
    setup_times = data['setup_times']
    variable_times = data['variable_times']
    big_M = data['big_M']

    # Initialize solution logger
    logger = SolutionLogger(args.log_path, "minimize") if args.log_path else None

    # Create model
    model = gp.Model("LotSizing")
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', args.time_limit)

    # Create variables
    x = {}
    s = {}
    y = {}
    for i in range(n_items):
        for t in range(n_periods):
            x[i, t] = model.addVar(lb=0, ub=big_M[i][t], vtype=GRB.CONTINUOUS, name=f'x_{i}_{t}')
            y[i, t] = model.addVar(vtype=GRB.BINARY, name=f'y_{i}_{t}')
        for t in range(n_periods + 1):
            s[i, t] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f's_{i}_{t}')

    # Set initial inventory
    for i in range(n_items):
        model.addConstr(s[i, 0] == initial_inventory[i], f'init_inv_{i}')

    # Flow conservation constraints
    for i in range(n_items):
        for t in range(n_periods):
            model.addConstr(
                s[i, t] + x[i, t] == demands[i][t] + s[i, t+1],
                f'flow_{i}_{t}'
            )

    # Setup constraints
    for i in range(n_items):
        for t in range(n_periods):
            model.addConstr(
                x[i, t] <= big_M[i][t] * y[i, t],
                f'setup_{i}_{t}'
            )

    # Capacity constraints
    for t in range(n_periods):
        time_used = gp.LinExpr()
        for i in range(n_items):
            time_used += setup_times[i][t] * y[i, t] + variable_times[i][t] * x[i, t]
        model.addConstr(time_used <= capacity[t], f'cap_{t}')

    # Ending inventory must be zero
    for i in range(n_items):
        model.addConstr(s[i, n_periods] == 0, f'end_inv_{i}')

    # Objective function
    total_cost = gp.LinExpr()
    for i in range(n_items):
        for t in range(n_periods):
            total_cost += setup_costs[i][t] * y[i, t]
            total_cost += variable_costs[i][t] * x[i, t]
            total_cost += holding_costs[i][t] * s[i, t+1]
    model.setObjective(total_cost, GRB.MINIMIZE)

    # Callback for logging
    if logger:
        def callback(model, where):
            if where == GRB.Callback.MIPSOL:
                obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                logger.log(obj_val)
        model.optimize(callback)
    else:
        model.optimize()

    # Prepare solution
    sol_x = [[0.0] * n_periods for _ in range(n_items)]
    sol_s = [[0.0] * (n_periods + 1) for _ in range(n_items)]
    sol_y = [[0] * n_periods for _ in range(n_items)]
    obj_val = 1e20  # Default for infeasible

    if model.SolCount > 0:
        obj_val = model.ObjVal
        for i in range(n_items):
            for t in range(n_periods):
                sol_x[i][t] = x[i, t].X
                sol_y[i][t] = int(round(y[i, t].X))
            for t in range(n_periods + 1):
                sol_s[i][t] = s[i, t].X
    else:
        # Fallback solution: initial inventory only
        for i in range(n_items):
            sol_s[i][0] = initial_inventory[i]
            inv = initial_inventory[i]
            for t in range(n_periods):
                next_inv = max(0.0, inv - demands[i][t])
                sol_s[i][t+1] = next_inv
                inv = next_inv

    # Write solution
    solution = {
        "objective_value": obj_val,
        "x": sol_x,
        "s": sol_s,
        "y": sol_y
    }
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == "__main__":
    main()
import argparse
import json
import time
from solution_logger import SolutionLogger
import gurobipy as gp
from gurobipy import GRB


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

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

    model = gp.Model("CLSP")
    model.Params.TimeLimit = args.time_limit
    model.Params.Threads = 1
    model.Params.MIPGap = 0.0
    model.Params.LogToConsole = 0

    x = model.addVars(n_items, n_periods, lb=0, vtype=GRB.CONTINUOUS, name="x")
    y = model.addVars(n_items, n_periods, vtype=GRB.BINARY, name="y")
    s = model.addVars(n_items, n_periods + 1, lb=0, vtype=GRB.CONTINUOUS, name="s")

    # Initial inventory
    for i in range(n_items):
        model.addConstr(s[i, 0] == initial_inventory[i])

    # Inventory balance
    for i in range(n_items):
        for t in range(n_periods):
            model.addConstr(
                s[i, t] + x[i, t] == demands[i][t] + s[i, t + 1]
            )

    # Final inventory must be zero
    for i in range(n_items):
        model.addConstr(s[i, n_periods] == 0)

    # Setup forcing
    for i in range(n_items):
        for t in range(n_periods):
            model.addConstr(x[i, t] <= big_M[i][t] * y[i, t])

    # Capacity constraints
    for t in range(n_periods):
        model.addConstr(
            gp.quicksum(setup_times[i][t] * y[i, t] for i in range(n_items)) +
            gp.quicksum(variable_times[i][t] * x[i, t] for i in range(n_items)) <= capacity[t]
        )

    # Objective
    setup_cost_expr = gp.quicksum(setup_costs[i][t] * y[i, t] for i in range(n_items) for t in range(n_periods))
    prod_cost_expr = gp.quicksum(variable_costs[i][t] * x[i, t] for i in range(n_items) for t in range(n_periods))
    hold_cost_expr = gp.quicksum(holding_costs[i][t] * s[i, t + 1] for i in range(n_items) for t in range(n_periods))
    model.setObjective(setup_cost_expr + prod_cost_expr + hold_cost_expr, GRB.MINIMIZE)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if logger:
                logger.log(float(obj))

    model.optimize(callback)

    if model.SolCount == 0:
        raise RuntimeError("No feasible solution found within time limit.")

    # Extract solution
    x_sol = [[float(x[i, t].X) for t in range(n_periods)] for i in range(n_items)]
    y_sol = [[int(y[i, t].X) for t in range(n_periods)] for i in range(n_items)]
    s_sol = [[float(s[i, t].X) for t in range(n_periods + 1)] for i in range(n_items)]
    obj_val = float(model.ObjVal)

    solution = {
        "objective_value": obj_val,
        "x": x_sol,
        "s": s_sol,
        "y": y_sol
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()
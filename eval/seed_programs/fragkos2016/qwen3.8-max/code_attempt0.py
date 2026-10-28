import json
import argparse
import sys
from gurobipy import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Read instance
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

    # Build model
    model = Model("CLSP")
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('OutputFlag', 0)

    # Decision variables
    x = {}  # production quantity
    y = {}  # setup binary
    s = {}  # inventory

    for i in range(n_items):
        for t in range(n_periods):
            x[i, t] = model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"x_{i}_{t}")
            y[i, t] = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name=f"y_{i}_{t}")
        for t in range(n_periods + 1):
            s[i, t] = model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"s_{i}_{t}")

    # Fix initial inventory
    for i in range(n_items):
        s[i, 0].setAttr('LB', initial_inventory[i])
        s[i, 0].setAttr('UB', initial_inventory[i])

    # Terminal inventory must be zero
    for i in range(n_items):
        s[i, n_periods].setAttr('UB', 0.0)

    # Objective: minimize setup + production + holding costs
    obj = LinExpr()
    for i in range(n_items):
        for t in range(n_periods):
            obj += setup_costs[i][t] * y[i, t]
            obj += variable_costs[i][t] * x[i, t]
            obj += holding_costs[i][t] * s[i, t + 1]
    model.setObjective(obj, GRB.MINIMIZE)

    # Constraints
    for i in range(n_items):
        for t in range(n_periods):
            # Inventory balance: s[i,t] + x[i,t] = demand[i,t] + s[i,t+1]
            model.addConstr(
                s[i, t] + x[i, t] == demands[i][t] + s[i, t + 1],
                name=f"balance_{i}_{t}"
            )
            # Setup forcing: production only if setup
            model.addConstr(
                x[i, t] <= big_M[i][t] * y[i, t],
                name=f"setup_{i}_{t}"
            )

    # Capacity constraints
    for t in range(n_periods):
        cap_expr = LinExpr()
        for i in range(n_items):
            cap_expr += setup_times[i][t] * y[i, t]
            cap_expr += variable_times[i][t] * x[i, t]
        model.addConstr(cap_expr <= capacity[t], name=f"cap_{t}")

    # Store variable lists for callback
    x_vars = [x[i, t] for i in range(n_items) for t in range(n_periods)]
    y_vars = [y[i, t] for i in range(n_items) for t in range(n_periods)]
    s_vars = [s[i, t] for i in range(n_items) for t in range(n_periods + 1)]

    best_obj = [float('inf')]

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj_val < best_obj[0]:
                best_obj[0] = obj_val
                if logger:
                    x_vals = model.cbGetSolution(x_vars)
                    y_vals = model.cbGetSolution(y_vars)
                    s_vals = model.cbGetSolution(s_vars)

                    x_sol = [[x_vals[i * n_periods + t] for t in range(n_periods)] for i in range(n_items)]
                    y_sol = [[int(round(y_vals[i * n_periods + t])) for t in range(n_periods)] for i in range(n_items)]
                    s_sol = [[s_vals[i * (n_periods + 1) + t] for t in range(n_periods + 1)] for i in range(n_items)]

                    solution = {
                        "objective_value": obj_val,
                        "x": x_sol,
                        "s": s_sol,
                        "y": y_sol
                    }
                    logger.log_solution(obj_val, solution)

    model.optimize(callback)

    # Extract final solution
    if model.SolCount > 0:
        obj_val = model.ObjVal

        x_sol = [[x[i, t].X for t in range(n_periods)] for i in range(n_items)]
        y_sol = [[int(round(y[i, t].X)) for t in range(n_periods)] for i in range(n_items)]
        s_sol = [[s[i, t].X for t in range(n_periods + 1)] for i in range(n_items)]

        solution = {
            "objective_value": obj_val,
            "x": x_sol,
            "s": s_sol,
            "y": y_sol
        }

        # Log final solution if it's better than what callback logged
        if logger and obj_val < best_obj[0]:
            logger.log_solution(obj_val, solution)

        with open(args.solution_path, 'w') as f:
            json.dump(solution, f)
    else:
        # No feasible solution found
        solution = {
            "objective_value": 1e20,
            "x": [[0.0] * n_periods for _ in range(n_items)],
            "s": [[0.0] * (n_periods + 1) for _ in range(n_items)],
            "y": [[0] * n_periods for _ in range(n_items)]
        }
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f)


if __name__ == "__main__":
    main()
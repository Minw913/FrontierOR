import argparse
import json
import math
import os
from typing import Any, Dict, List

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def read_instance(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def build_and_solve(instance: Dict[str, Any], time_limit: int, logger: SolutionLogger = None):
    n_items = int(instance["n_items"])
    n_periods = int(instance["n_periods"])

    capacity = instance["capacity"]
    initial_inventory = instance["initial_inventory"]
    demands = instance["demands"]
    setup_costs = instance["setup_costs"]
    holding_costs = instance["holding_costs"]
    variable_costs = instance["variable_costs"]
    setup_times = instance["setup_times"]
    variable_times = instance["variable_times"]
    big_M = instance["big_M"]

    model = gp.Model("multi_item_capacitated_lot_sizing")
    model.Params.TimeLimit = int(time_limit)
    model.Params.Threads = 1
    model.Params.MIPFocus = 1

    # Decision variables
    x = model.addVars(
        n_items, n_periods,
        lb=0.0,
        ub={(i, t): float(big_M[i][t]) for i in range(n_items) for t in range(n_periods)},
        vtype=GRB.CONTINUOUS,
        name="x"
    )
    y = model.addVars(n_items, n_periods, vtype=GRB.BINARY, name="y")
    s = model.addVars(n_items, n_periods + 1, lb=0.0, vtype=GRB.CONTINUOUS, name="s")

    # Initial inventory
    for i in range(n_items):
        model.addConstr(s[i, 0] == float(initial_inventory[i]), name=f"init_inv_{i}")

    # Inventory balance + terminal inventory
    for i in range(n_items):
        for t in range(n_periods):
            model.addConstr(
                s[i, t] + x[i, t] == float(demands[i][t]) + s[i, t + 1],
                name=f"balance_{i}_{t}"
            )
        model.addConstr(s[i, n_periods] == 0.0, name=f"terminal_inv_{i}")

    # Production only if setup
    for i in range(n_items):
        for t in range(n_periods):
            model.addConstr(
                x[i, t] <= float(big_M[i][t]) * y[i, t],
                name=f"link_{i}_{t}"
            )

    # Capacity constraints
    for t in range(n_periods):
        model.addConstr(
            gp.quicksum(float(setup_times[i][t]) * y[i, t] + float(variable_times[i][t]) * x[i, t]
                        for i in range(n_items)) <= float(capacity[t]),
            name=f"capacity_{t}"
        )

    # Objective: setup + variable production + holding inventory
    # holding_costs[i][t] is charged on inventory carried from period t to t+1 => s[i, t+1]
    setup_term = gp.quicksum(float(setup_costs[i][t]) * y[i, t] for i in range(n_items) for t in range(n_periods))
    var_term = gp.quicksum(float(variable_costs[i][t]) * x[i, t] for i in range(n_items) for t in range(n_periods))
    hold_term = gp.quicksum(float(holding_costs[i][t]) * s[i, t + 1] for i in range(n_items) for t in range(n_periods))

    model.setObjective(setup_term + var_term + hold_term, GRB.MINIMIZE)

    # Incumbent logging callback
    model._logger = logger
    model._best_obj = math.inf

    def cb(m, where):
        if where == GRB.Callback.MIPSOL:
            obj = m.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj + 1e-9 < m._best_obj:
                m._best_obj = obj
                if m._logger:
                    m._logger.log(float(obj))

    model.optimize(cb)

    # Extract solution
    if model.SolCount > 0:
        objective_value = float(model.ObjVal)
        x_sol = [[float(x[i, t].X) for t in range(n_periods)] for i in range(n_items)]
        s_sol = [[float(s[i, t].X) for t in range(n_periods + 1)] for i in range(n_items)]
        y_sol = [[int(round(y[i, t].X)) for t in range(n_periods)] for i in range(n_items)]
    else:
        # Fallback if no feasible solution found
        objective_value = float("inf")
        x_sol = [[0.0 for _ in range(n_periods)] for _ in range(n_items)]
        y_sol = [[0 for _ in range(n_periods)] for _ in range(n_items)]
        s_sol = []
        for i in range(n_items):
            row = [0.0 for _ in range(n_periods + 1)]
            row[0] = float(initial_inventory[i])
            s_sol.append(row)

    return {
        "objective_value": objective_value,
        "x": x_sol,
        "s": s_sol,
        "y": y_sol,
    }


def main():
    parser = argparse.ArgumentParser(description="Solve multi-item capacitated lot-sizing problem with Gurobi.")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to input instance JSON file.")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to output solution JSON file.")
    parser.add_argument("--time_limit", type=int, required=True, help="Maximum runtime in seconds.")
    parser.add_argument("--log_path", type=str, default=None, help="Optional JSONL path for incumbent logging.")
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    instance = read_instance(args.instance_path)
    solution = build_and_solve(instance, args.time_limit, logger=logger)

    os.makedirs(os.path.dirname(os.path.abspath(args.solution_path)), exist_ok=True)
    with open(args.solution_path, "w") as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()
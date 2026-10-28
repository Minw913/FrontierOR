import argparse
import json
import sys
import os
from gurobipy import Model, GRB, quicksum


def build_and_solve(instance, time_limit, log_path):
    n = instance['num_items']
    capacity = instance['capacity']
    profits = instance['items']['profits']
    weights = instance['items']['weights']
    edges = instance['conflict_graph']['edges']

    # Trivial feasible fallback: pick the best single item (all weights <= capacity).
    best_single = max(range(n), key=lambda i: profits[i]) if n > 0 else -1

    from solution_logger import SolutionLogger
    logger = SolutionLogger(log_path, sense="maximize") if log_path else None

    # If trivially small instance, solve by enumeration fallback handled by Gurobi anyway.
    model = Model("kcg")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = float(time_limit)
    model.Params.OutputFlag = 0

    x = model.addVars(n, vtype=GRB.BINARY, name="x")
    model.setObjective(quicksum(profits[i] * x[i] for i in range(n)), GRB.MAXIMIZE)
    model.addConstr(quicksum(weights[i] * x[i] for i in range(n)) <= capacity, "cap")

    for idx, e in enumerate(edges):
        i, j = e[0], e[1]
        if i == j:
            continue
        model.addConstr(x[i] + x[j] <= 1, f"conf{idx}")

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            sol_vals = model.cbGetSolution(x)
            selected = [i for i in range(n) if sol_vals[i] > 0.5]
            sol = {"objective_value": float(obj), "selected_items": selected}
            if logger:
                logger.log_solution(float(obj), sol)

    model.optimize(callback)

    selected = []
    obj_val = 0.0
    if model.SolCount > 0:
        for i in range(n):
            if x[i].X > 0.5:
                selected.append(i)
        obj_val = model.ObjVal
    else:
        # Fallback to single best item.
        selected = [best_single] if best_single >= 0 else []
        obj_val = float(profits[best_single]) if best_single >= 0 else 0.0
        if logger:
            logger.log_solution(float(obj_val),
                                 {"objective_value": float(obj_val), "selected_items": selected})

    return {"objective_value": float(obj_val), "selected_items": selected}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    solution = build_and_solve(instance, args.time_limit, args.log_path)

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == '__main__':
    main()
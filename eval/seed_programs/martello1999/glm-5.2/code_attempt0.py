import argparse
import json
import sys

import gurobipy as gp
from gurobipy import GRB


def solve():
    parser = argparse.ArgumentParser(description="0/1 Knapsack solver")
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    # Initialize logger
    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    # Read instance
    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    n = instance["n"]
    capacity = instance["capacity"]
    items = instance["items"]

    ids = [item["id"] for item in items]
    profits = [item["profit"] for item in items]
    weights = [item["weight"] for item in items]

    # ------------------------------------------------------------------
    # Greedy heuristic for warm start and immediate feasible solution
    # ------------------------------------------------------------------
    order = sorted(range(n), key=lambda i: profits[i] / weights[i], reverse=True)
    greedy_sol = [0] * n
    greedy_weight = 0
    greedy_profit = 0
    for i in order:
        if greedy_weight + weights[i] <= capacity:
            greedy_sol[i] = 1
            greedy_weight += weights[i]
            greedy_profit += profits[i]

    if logger:
        sol_dict = {
            "objective_value": int(greedy_profit),
            "selected_items": [ids[i] for i in range(n) if greedy_sol[i] == 1],
        }
        logger.log_solution(int(greedy_profit), sol_dict)

    # ------------------------------------------------------------------
    # Gurobi model
    # ------------------------------------------------------------------
    model = gp.Model("knapsack")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = float(args.time_limit)
    model.Params.OutputFlag = 0

    x = model.addVars(n, vtype=GRB.BINARY, name="x")
    model.setObjective(gp.quicksum(profits[i] * x[i] for i in range(n)), GRB.MAXIMIZE)
    model.addConstr(gp.quicksum(weights[i] * x[i] for i in range(n)) <= capacity, "capacity")

    # Warm start
    for i in range(n):
        x[i].Start = greedy_sol[i]

    # Callback to log incumbents
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            sol = model.cbGet(GRB.Callback.MIPSOL_SOL)
            selected = [ids[i] for i in range(n) if sol[i] > 0.5]
            sol_dict = {
                "objective_value": int(round(obj)),
                "selected_items": selected,
            }
            if logger:
                logger.log_solution(int(round(obj)), sol_dict)

    model.optimize(callback)

    # ------------------------------------------------------------------
    # Extract best solution
    # ------------------------------------------------------------------
    if model.SolCount > 0:
        obj_val = int(round(model.ObjVal))
        selected = [ids[i] for i in range(n) if x[i].X > 0.5]
    else:
        obj_val = int(greedy_profit)
        selected = [ids[i] for i in range(n) if greedy_sol[i] == 1]

    solution = {
        "objective_value": obj_val,
        "selected_items": selected,
    }

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    solve()
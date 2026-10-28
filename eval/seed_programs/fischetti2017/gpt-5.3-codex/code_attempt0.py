import argparse
import json
import math
import time
from typing import List, Tuple, Optional

import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def build_solution_dict(objective_value: float, y_sol: List[int], x_sol: List[List[float]]) -> dict:
    return {
        "objective_value": float(objective_value),
        "variant": "quadratic",
        "y": [int(v) for v in y_sol],
        "x": [[float(v) for v in row] for row in x_sol],
    }


def compute_x_from_open_set(
    open_mask: List[int], alloc_costs: List[List[float]]
) -> Tuple[List[List[float]], float]:
    """Closed-form optimal x for fixed open set in qUFL.
    Returns (x, allocation_cost_part).
    """
    n = len(open_mask)
    m = len(alloc_costs[0]) if n > 0 else 0
    x = [[0.0] * m for _ in range(n)]
    open_idx = [i for i in range(n) if open_mask[i] == 1]

    if not open_idx:
        return x, float("inf")

    alloc_obj = 0.0
    for j in range(m):
        denom = 0.0
        for i in open_idx:
            cij = alloc_costs[i][j]
            denom += 1.0 / cij
        if denom <= 0.0:
            return x, float("inf")
        alloc_obj += 1.0 / denom
        inv_denom = 1.0 / denom
        for i in open_idx:
            x[i][j] = (1.0 / alloc_costs[i][j]) * inv_denom

    return x, alloc_obj


def greedy_initial_solution(
    opening_costs: List[float], alloc_costs: List[List[float]], max_additions: int = 25
) -> Tuple[List[int], List[List[float]], float]:
    """Fast greedy heuristic: best single facility, then add improving facilities."""
    n = len(opening_costs)
    m = len(alloc_costs[0]) if n > 0 else 0

    if n == 0:
        return [], [], 0.0 if m == 0 else 1e100

    # Best single-open facility
    best_i = min(range(n), key=lambda i: opening_costs[i] + sum(alloc_costs[i][j] for j in range(m)))
    open_mask = [0] * n
    open_mask[best_i] = 1

    inv_costs = [[1.0 / alloc_costs[i][j] for j in range(m)] for i in range(n)]
    denom = [inv_costs[best_i][j] for j in range(m)]
    current_obj = opening_costs[best_i] + sum(1.0 / d for d in denom)

    # Greedy additions
    remaining = set(range(n))
    remaining.remove(best_i)

    additions = 0
    while remaining and additions < max_additions:
        best_gain = 0.0
        best_k = None

        for k in remaining:
            alloc_gain = 0.0
            row_inv = inv_costs[k]
            for j in range(m):
                d0 = denom[j]
                d1 = d0 + row_inv[j]
                alloc_gain += (1.0 / d0) - (1.0 / d1)  # reduction in allocation cost
            net_gain = alloc_gain - opening_costs[k]
            if net_gain > best_gain + 1e-12:
                best_gain = net_gain
                best_k = k

        if best_k is None:
            break

        open_mask[best_k] = 1
        row_inv = inv_costs[best_k]
        for j in range(m):
            denom[j] += row_inv[j]
        current_obj -= best_gain
        remaining.remove(best_k)
        additions += 1

    x, alloc_obj = compute_x_from_open_set(open_mask, alloc_costs)
    total_obj = sum(opening_costs[i] * open_mask[i] for i in range(n)) + alloc_obj
    return open_mask, x, total_obj


class IncumbentLoggerCallback:
    def __init__(self, logger, y_vars, x_vars, n, m):
        self.logger = logger
        self.y_vars = y_vars
        self.x_vars = x_vars
        self.n = n
        self.m = m
        self.best_obj = float("inf")

    def __call__(self, model, where):
        if where == GRB.Callback.MIPSOL and self.logger is not None:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj + 1e-9 < self.best_obj:
                self.best_obj = obj
                y_vals = [model.cbGetSolution(self.y_vars[i]) for i in range(self.n)]
                x_vals = [[model.cbGetSolution(self.x_vars[i, j]) for j in range(self.m)] for i in range(self.n)]
                y_sol = [1 if v >= 0.5 else 0 for v in y_vals]
                sol = build_solution_dict(obj, y_sol, x_vals)
                self.logger.log_solution(obj, sol)


def solve(instance: dict, time_limit: int, logger: Optional[SolutionLogger]) -> dict:
    n = int(instance["num_facilities"])
    m = int(instance["num_customers"])
    opening_costs = [float(v) for v in instance["opening_costs"]]
    alloc_costs = [[float(v) for v in row] for row in instance["allocation_costs"]]

    # Edge cases
    if n == 0 and m == 0:
        return build_solution_dict(0.0, [], [])
    if n == 0 and m > 0:
        # Infeasible by definition; return schema-valid placeholder.
        return build_solution_dict(1e100, [], [])

    # Heuristic warm start
    y0, x0, obj0 = greedy_initial_solution(opening_costs, alloc_costs, max_additions=min(25, n))
    if logger is not None:
        logger.log_solution(obj0, build_solution_dict(obj0, y0, x0))

    model = gp.Model("qUFL")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = int(time_limit)

    y = model.addVars(n, vtype=GRB.BINARY, name="y")
    x = model.addVars(n, m, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="x")

    # Demand coverage
    for j in range(m):
        model.addConstr(gp.quicksum(x[i, j] for i in range(n)) == 1.0, name=f"cover_{j}")

    # Linking x <= y
    for i in range(n):
        yi = y[i]
        for j in range(m):
            model.addConstr(x[i, j] <= yi, name=f"link_{i}_{j}")

    # Objective: opening + separable convex quadratic allocation
    obj = gp.quicksum(opening_costs[i] * y[i] for i in range(n))
    obj += gp.quicksum(alloc_costs[i][j] * x[i, j] * x[i, j] for i in range(n) for j in range(m))
    model.setObjective(obj, GRB.MINIMIZE)

    # Warm start
    for i in range(n):
        y[i].Start = float(y0[i])
    for i in range(n):
        for j in range(m):
            x[i, j].Start = float(x0[i][j])

    cb = IncumbentLoggerCallback(logger, y, x, n, m) if logger is not None else None
    if cb is not None:
        model.optimize(cb)
    else:
        model.optimize()

    # Extract best available solution
    if model.SolCount > 0:
        y_sol = [1 if y[i].X >= 0.5 else 0 for i in range(n)]
        x_sol = [[max(0.0, min(1.0, x[i, j].X)) for j in range(m)] for i in range(n)]

        # Recompute objective for output consistency
        alloc_obj = 0.0
        for i in range(n):
            row = x_sol[i]
            crow = alloc_costs[i]
            for j in range(m):
                alloc_obj += crow[j] * row[j] * row[j]
        open_obj = sum(opening_costs[i] * y_sol[i] for i in range(n))
        final_obj = open_obj + alloc_obj
        sol = build_solution_dict(final_obj, y_sol, x_sol)

        if logger is not None and final_obj + 1e-9 < getattr(cb, "best_obj", float("inf")):
            logger.log_solution(final_obj, sol)
        return sol

    # Fallback to heuristic if no incumbent from solver
    return build_solution_dict(obj0, y0, x0)


def main():
    parser = argparse.ArgumentParser(description="Solve quadratic uncapacitated facility location (qUFL).")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to input instance JSON.")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to output solution JSON.")
    parser.add_argument("--time_limit", type=int, required=True, help="Max runtime in seconds.")
    parser.add_argument("--log_path", type=str, default=None, help="Optional JSONL path for convergence logging.")
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    sol = solve(instance, args.time_limit, logger)

    with open(args.solution_path, "w") as f:
        json.dump(sol, f)


if __name__ == "__main__":
    main()
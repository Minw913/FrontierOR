import argparse
import json
import math
import random
import time
from typing import List, Tuple

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def compute_objective(
    assignment: List[int],
    m: int,
    values: List[int],
    pairwise_values: List[List[int]],
) -> float:
    bins = [[] for _ in range(m)]
    total = 0.0

    for i, k in enumerate(assignment):
        if k != -1:
            total += values[i]
            bins[k].append(i)

    for k in range(m):
        items = bins[k]
        for a_idx in range(len(items)):
            i = items[a_idx]
            row = pairwise_values[i]
            for b_idx in range(a_idx + 1, len(items)):
                j = items[b_idx]
                total += row[j]

    return float(total)


def greedy_construct(
    order: List[int],
    n: int,
    m: int,
    capacity: int,
    weights: List[int],
    values: List[int],
    pairwise_values: List[List[int]],
) -> List[int]:
    assignment = [-1] * n
    bin_weights = [0] * m
    bin_items = [[] for _ in range(m)]

    for i in order:
        best_k = -1
        best_delta = 0.0  # only add if non-negative immediate gain

        for k in range(m):
            if bin_weights[k] + weights[i] > capacity:
                continue
            delta = values[i]
            for j in bin_items[k]:
                delta += pairwise_values[i][j]
            if delta > best_delta + 1e-12:
                best_delta = delta
                best_k = k

        if best_k != -1:
            assignment[i] = best_k
            bin_weights[best_k] += weights[i]
            bin_items[best_k].append(i)

    return assignment


def local_improve(
    assignment: List[int],
    n: int,
    m: int,
    capacity: int,
    weights: List[int],
    values: List[int],
    pairwise_values: List[List[int]],
    time_deadline: float,
) -> List[int]:
    # Rebuild bin state
    bin_weights = [0] * m
    bin_items = [[] for _ in range(m)]
    for i, k in enumerate(assignment):
        if k != -1:
            bin_weights[k] += weights[i]
            bin_items[k].append(i)

    improved = True
    while improved and time.time() < time_deadline:
        improved = False

        # Try insert unassigned items
        for i in range(n):
            if time.time() >= time_deadline:
                break
            if assignment[i] != -1:
                continue

            best_k = -1
            best_delta = 0.0
            for k in range(m):
                if bin_weights[k] + weights[i] > capacity:
                    continue
                delta = values[i]
                for j in bin_items[k]:
                    delta += pairwise_values[i][j]
                if delta > best_delta + 1e-12:
                    best_delta = delta
                    best_k = k

            if best_k != -1:
                assignment[i] = best_k
                bin_weights[best_k] += weights[i]
                bin_items[best_k].append(i)
                improved = True

        # Try relocate assigned items
        for i in range(n):
            if time.time() >= time_deadline:
                break
            a = assignment[i]
            if a == -1:
                continue

            # removal contribution
            remove_delta = -values[i]
            for j in bin_items[a]:
                if j != i:
                    remove_delta -= pairwise_values[i][j]

            best_k = a
            best_move_delta = 0.0
            for b in range(m):
                if b == a:
                    continue
                if bin_weights[b] + weights[i] > capacity:
                    continue
                add_delta = values[i]
                for j in bin_items[b]:
                    add_delta += pairwise_values[i][j]
                move_delta = remove_delta + add_delta
                if move_delta > best_move_delta + 1e-12:
                    best_move_delta = move_delta
                    best_k = b

            if best_k != a:
                bin_items[a].remove(i)
                bin_weights[a] -= weights[i]
                assignment[i] = best_k
                bin_items[best_k].append(i)
                bin_weights[best_k] += weights[i]
                improved = True

    return assignment


class IncumbentCallback:
    def __init__(self, logger, initial_best=-float("inf")):
        self.logger = logger
        self.best = initial_best

    def __call__(self, model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj > self.best + 1e-6:
                self.best = obj
                if self.logger:
                    self.logger.log(float(obj))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, required=False, default=None)
    args = parser.parse_args()

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    n = int(inst["n"])
    m = int(inst["m"])
    capacity = int(inst["capacity"])
    weights = list(map(int, inst["weights"]))
    values = list(map(int, inst["values"]))
    pairwise_values = inst["pairwise_values"]

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    t0 = time.time()
    deadline = t0 + max(1, args.time_limit)

    # -------- Heuristic warm start --------
    rng = random.Random(42)
    base_score = []
    for i in range(n):
        pos_pair_sum = 0
        row = pairwise_values[i]
        for j in range(n):
            if j != i and row[j] > 0:
                pos_pair_sum += row[j]
        # light normalization by m to avoid huge bias
        score = values[i] + pos_pair_sum / max(1, m)
        base_score.append((score, i))

    base_order = [i for _, i in sorted(base_score, reverse=True)]

    best_heur_assign = [-1] * n
    best_heur_obj = 0.0  # empty solution objective

    heur_time_budget = min(2.0, max(0.5, 0.15 * args.time_limit))
    heur_deadline = min(deadline - 0.1, t0 + heur_time_budget)

    # deterministic pass
    assign = greedy_construct(base_order, n, m, capacity, weights, values, pairwise_values)
    assign = local_improve(assign, n, m, capacity, weights, values, pairwise_values, heur_deadline)
    obj = compute_objective(assign, m, values, pairwise_values)
    if obj > best_heur_obj + 1e-9:
        best_heur_obj = obj
        best_heur_assign = assign[:]
        if logger:
            logger.log(float(best_heur_obj))

    # few randomized passes while time allows
    while time.time() < heur_deadline:
        perturbed = base_order[:]
        # shuffle chunks lightly
        for _ in range(3):
            a = rng.randrange(n) if n > 0 else 0
            b = rng.randrange(n) if n > 0 else 0
            if a < n and b < n:
                perturbed[a], perturbed[b] = perturbed[b], perturbed[a]

        assign = greedy_construct(perturbed, n, m, capacity, weights, values, pairwise_values)
        assign = local_improve(assign, n, m, capacity, weights, values, pairwise_values, heur_deadline)
        obj = compute_objective(assign, m, values, pairwise_values)
        if obj > best_heur_obj + 1e-9:
            best_heur_obj = obj
            best_heur_assign = assign[:]
            if logger:
                logger.log(float(best_heur_obj))

    # -------- Build MIQP model --------
    model = gp.Model("multi_knapsack_quadratic")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = max(0.0, deadline - time.time())
    model.Params.NonConvex = 2
    model.Params.MIPFocus = 1

    x = model.addVars(n, m, vtype=GRB.BINARY, name="x")

    # Each item in at most one knapsack
    for i in range(n):
        model.addConstr(gp.quicksum(x[i, k] for k in range(m)) <= 1, name=f"assign_{i}")

    # Capacity constraints
    for k in range(m):
        model.addConstr(
            gp.quicksum(weights[i] * x[i, k] for i in range(n)) <= capacity,
            name=f"cap_{k}",
        )

    # Objective: linear individual values + pairwise same-knapsack values
    obj_expr = gp.QuadExpr()

    for i in range(n):
        vi = values[i]
        if vi != 0:
            for k in range(m):
                obj_expr += vi * x[i, k]

    for k in range(m):
        for i in range(n - 1):
            xi = x[i, k]
            row = pairwise_values[i]
            for j in range(i + 1, n):
                pij = row[j]
                if pij != 0:
                    obj_expr += pij * xi * x[j, k]

    model.setObjective(obj_expr, GRB.MAXIMIZE)

    # MIP start from heuristic
    for i in range(n):
        ai = best_heur_assign[i]
        for k in range(m):
            x[i, k].Start = 1.0 if ai == k else 0.0

    cb = IncumbentCallback(logger, initial_best=best_heur_obj)

    # If no time remains, skip optimize
    if model.Params.TimeLimit > 1e-6:
        model.optimize(cb)
    else:
        model.update()

    # -------- Extract best solution --------
    best_assignment = best_heur_assign[:]
    best_obj = float(best_heur_obj)

    if model.SolCount > 0:
        mip_assign = [-1] * n
        for i in range(n):
            assigned_k = -1
            for k in range(m):
                if x[i, k].X > 0.5:
                    assigned_k = k
                    break
            mip_assign[i] = assigned_k

        mip_obj = compute_objective(mip_assign, m, values, pairwise_values)
        if mip_obj > best_obj + 1e-7:
            best_obj = mip_obj
            best_assignment = mip_assign

    # Build output assignment dict with string keys for packed items only
    out_assignment = {}
    for i, k in enumerate(best_assignment):
        if k != -1:
            out_assignment[str(i)] = int(k)

    solution = {
        "objective_value": float(best_obj),
        "assignment": out_assignment,
    }

    with open(args.solution_path, "w") as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()
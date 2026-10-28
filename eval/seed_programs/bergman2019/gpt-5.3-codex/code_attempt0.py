import argparse
import json
import time
from typing import List, Tuple

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def read_instance(path: str):
    with open(path, "r") as f:
        data = json.load(f)
    n = int(data["n"])
    m = int(data["m"])
    profits = list(map(int, data["profits"]))
    pairwise = data["pairwise_profits"]
    weights = list(map(int, data["weights"]))
    capacities = list(map(int, data["capacities"]))

    # Basic sanity checks
    if len(profits) != n:
        raise ValueError("profits length mismatch with n")
    if len(weights) != n:
        raise ValueError("weights length mismatch with n")
    if len(capacities) != m:
        raise ValueError("capacities length mismatch with m")
    if len(pairwise) != n or any(len(row) != n for row in pairwise):
        raise ValueError("pairwise_profits shape mismatch with n x n")

    return n, m, profits, pairwise, weights, capacities


def compute_objective(
    assignment_of_item: List[int],
    profits: List[int],
    pairwise: List[List[int]],
    m: int,
) -> float:
    items_in_k = [[] for _ in range(m)]
    obj = 0
    for i, k in enumerate(assignment_of_item):
        if k >= 0:
            obj += profits[i]
            items_in_k[k].append(i)
    for k in range(m):
        items = items_in_k[k]
        L = len(items)
        for a in range(L):
            ia = items[a]
            row = pairwise[ia]
            for b in range(a + 1, L):
                ib = items[b]
                obj += row[ib]
    return float(obj)


def greedy_heuristic(
    n: int,
    m: int,
    profits: List[int],
    pairwise: List[List[int]],
    weights: List[int],
    capacities: List[int],
    time_budget: float = 0.5,
) -> Tuple[List[int], float]:
    start_t = time.time()

    # Initial ordering score: item profit + half of positive pair potential
    pos_pair_sum = [0] * n
    for i in range(n):
        s = 0
        row = pairwise[i]
        for j in range(n):
            if i != j and row[j] > 0:
                s += row[j]
        pos_pair_sum[i] = s

    order = list(range(n))
    order.sort(key=lambda i: (profits[i] + 0.5 * pos_pair_sum[i], -weights[i]), reverse=True)

    assign = [-1] * n
    rem = capacities[:]
    items_in_k = [[] for _ in range(m)]

    # Constructive phase
    for i in order:
        wi = weights[i]
        best_k = -1
        best_gain = 0.0  # only take nonnegative-improving moves
        for k in range(m):
            if wi <= rem[k]:
                gain = profits[i]
                for j in items_in_k[k]:
                    gain += pairwise[i][j]
                if gain > best_gain:
                    best_gain = gain
                    best_k = k
        if best_k >= 0:
            assign[i] = best_k
            rem[best_k] -= wi
            items_in_k[best_k].append(i)

    # Local improvement loop (add/move/remove improving)
    improved = True
    while improved and (time.time() - start_t) < time_budget:
        improved = False

        # 1) Best add for unassigned items
        best_delta = 1e-9
        best_move = None  # ("add", i, k)
        for i in range(n):
            if assign[i] != -1:
                continue
            wi = weights[i]
            for k in range(m):
                if wi <= rem[k]:
                    delta = profits[i]
                    for j in items_in_k[k]:
                        delta += pairwise[i][j]
                    if delta > best_delta:
                        best_delta = delta
                        best_move = ("add", i, k)

        # 2) Best move between knapsacks
        for i in range(n):
            k_from = assign[i]
            if k_from == -1:
                continue
            wi = weights[i]
            # contribution with current knapsack (excluding itself)
            contrib_from = 0
            for j in items_in_k[k_from]:
                if j != i:
                    contrib_from += pairwise[i][j]
            for k_to in range(m):
                if k_to == k_from:
                    continue
                if wi > rem[k_to]:
                    continue
                contrib_to = 0
                for j in items_in_k[k_to]:
                    contrib_to += pairwise[i][j]
                delta = contrib_to - contrib_from
                if delta > best_delta:
                    best_delta = delta
                    best_move = ("move", i, k_to)

        # 3) Best remove
        for i in range(n):
            k = assign[i]
            if k == -1:
                continue
            delta = -profits[i]
            for j in items_in_k[k]:
                if j != i:
                    delta -= pairwise[i][j]
            if delta > best_delta:
                best_delta = delta
                best_move = ("remove", i, -1)

        if best_move is not None:
            typ, i, k_new = best_move
            k_old = assign[i]
            if typ == "add":
                assign[i] = k_new
                rem[k_new] -= weights[i]
                items_in_k[k_new].append(i)
            elif typ == "move":
                items_in_k[k_old].remove(i)
                rem[k_old] += weights[i]
                assign[i] = k_new
                rem[k_new] -= weights[i]
                items_in_k[k_new].append(i)
            elif typ == "remove":
                items_in_k[k_old].remove(i)
                rem[k_old] += weights[i]
                assign[i] = -1
            improved = True

    obj = compute_objective(assign, profits, pairwise, m)
    return assign, obj


def build_and_solve(
    n: int,
    m: int,
    profits: List[int],
    pairwise: List[List[int]],
    weights: List[int],
    capacities: List[int],
    time_limit: int,
    logger: SolutionLogger = None,
):
    # Heuristic warm start
    heur_budget = min(0.5, max(0.05, 0.05 * time_limit))
    warm_assign, warm_obj = greedy_heuristic(
        n, m, profits, pairwise, weights, capacities, time_budget=heur_budget
    )

    if logger:
        logger.log(warm_obj)

    model = gp.Model("multiple_quadratic_knapsack")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = max(0, int(time_limit))
    model.Params.Threads = 1
    model.Params.NonConvex = 2

    x = model.addVars(n, m, vtype=GRB.BINARY, name="x")

    # Each item assigned to at most one knapsack
    for i in range(n):
        model.addConstr(gp.quicksum(x[i, k] for k in range(m)) <= 1, name=f"assign_{i}")

    # Capacity constraints
    for k in range(m):
        model.addConstr(
            gp.quicksum(weights[i] * x[i, k] for i in range(n)) <= capacities[k],
            name=f"cap_{k}",
        )

    # Objective: linear + pairwise within same knapsack
    obj = gp.LinExpr()
    for i in range(n):
        pi = profits[i]
        if pi != 0:
            for k in range(m):
                obj += pi * x[i, k]

    quad = gp.QuadExpr()
    for i in range(n):
        row = pairwise[i]
        for j in range(i + 1, n):
            q = row[j]
            if q == 0:
                continue
            for k in range(m):
                quad += q * x[i, k] * x[j, k]

    model.setObjective(obj + quad, GRB.MAXIMIZE)

    # Warm start
    for i in range(n):
        k = warm_assign[i]
        if k >= 0:
            x[i, k].Start = 1.0

    best_logged = {"val": warm_obj}

    def cb(model_cb, where):
        if where == GRB.Callback.MIPSOL and logger is not None:
            try:
                val = model_cb.cbGet(GRB.Callback.MIPSOL_OBJ)
                if val > best_logged["val"] + 1e-9:
                    best_logged["val"] = float(val)
                    logger.log(float(val))
            except gp.GurobiError:
                pass

    model.optimize(cb)

    # Extract best available solution
    final_assign = warm_assign[:]
    if model.SolCount > 0:
        extracted = [-1] * n
        for i in range(n):
            chosen = -1
            for k in range(m):
                v = x[i, k].X
                if v > 0.5:
                    chosen = k
                    break
            extracted[i] = chosen
        final_assign = extracted

    final_obj = compute_objective(final_assign, profits, pairwise, m)
    return final_assign, final_obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    n, m, profits, pairwise, weights, capacities = read_instance(args.instance_path)

    assignment_of_item, objective_value = build_and_solve(
        n=n,
        m=m,
        profits=profits,
        pairwise=pairwise,
        weights=weights,
        capacities=capacities,
        time_limit=args.time_limit,
        logger=logger,
    )

    assignment_pairs = [[i, k] for i, k in enumerate(assignment_of_item) if k >= 0]

    solution = {
        "objective_value": float(objective_value),
        "assignment": assignment_pairs,
    }

    with open(args.solution_path, "w") as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()
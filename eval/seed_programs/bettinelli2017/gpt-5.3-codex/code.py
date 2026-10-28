import argparse
import json
import time
import random
from typing import List, Tuple, Set

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def detect_index_base(edges: List[Tuple[int, int]], n: int) -> int:
    """Detect whether edges are 0-based or 1-based. Defaults to 0-based."""
    if not edges:
        return 0
    mn = min(min(u, v) for u, v in edges)
    mx = max(max(u, v) for u, v in edges)

    if 0 <= mn and mx <= n - 1:
        return 0
    if 1 <= mn and mx <= n:
        return 1
    # Fallback: if ambiguous/invalid, assume 0-based
    return 0


def build_adjacency(n: int, edges: List[Tuple[int, int]]) -> List[Set[int]]:
    adj = [set() for _ in range(n)]
    for u, v in edges:
        if u == v:
            adj[u].add(v)
        else:
            adj[u].add(v)
            adj[v].add(u)
    return adj


def greedy_construct(
    order: List[int],
    weights: List[int],
    profits: List[int],
    capacity: int,
    adj: List[Set[int]],
) -> Tuple[List[int], int]:
    n = len(weights)
    selected = [0] * n
    current_weight = 0
    current_profit = 0
    blocked = [False] * n  # blocked by selected neighbors

    for i in order:
        if blocked[i]:
            continue
        wi = weights[i]
        if current_weight + wi > capacity:
            continue
        # If self-conflict exists, can't select
        if i in adj[i]:
            continue

        selected[i] = 1
        current_weight += wi
        current_profit += profits[i]
        for nb in adj[i]:
            blocked[nb] = True

    chosen = [i for i, v in enumerate(selected) if v == 1]
    return chosen, current_profit


def improve_by_single_add(
    chosen_set: Set[int],
    weights: List[int],
    profits: List[int],
    capacity: int,
    adj: List[Set[int]],
) -> Set[int]:
    """Simple 1-opt add improvement: try adding any feasible non-conflicting item."""
    n = len(weights)
    cur_w = sum(weights[i] for i in chosen_set)
    changed = True
    while changed:
        changed = False
        best_i = -1
        best_gain = 0
        for i in range(n):
            if i in chosen_set:
                continue
            if cur_w + weights[i] > capacity:
                continue
            if any((j in adj[i]) for j in chosen_set):
                continue
            if profits[i] > best_gain:
                best_gain = profits[i]
                best_i = i
        if best_i != -1:
            chosen_set.add(best_i)
            cur_w += weights[best_i]
            changed = True
    return chosen_set


def main():
    parser = argparse.ArgumentParser(description="Knapsack with conflict graph solver (Gurobi).")
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None
    best_logged = float("-inf")

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    n = int(inst["num_items"])
    capacity = int(inst["capacity"])
    profits = list(map(int, inst["items"]["profits"]))
    weights = list(map(int, inst["items"]["weights"]))

    raw_edges = [tuple(e) for e in inst["conflict_graph"]["edges"]]
    base = detect_index_base(raw_edges, n)
    if base == 1:
        edges = [(u - 1, v - 1) for (u, v) in raw_edges]
    else:
        edges = raw_edges

    # Keep only valid edges (robustness)
    clean_edges = []
    for (u, v) in edges:
        if 0 <= u < n and 0 <= v < n:
            clean_edges.append((u, v))
    edges = clean_edges

    adj = build_adjacency(n, edges)
    degrees = [len(adj[i]) for i in range(n)]

    # --- Fast heuristic warm-starts ---
    heuristic_best_items = []
    heuristic_best_obj = 0

    def try_update(items, obj):
        nonlocal heuristic_best_items, heuristic_best_obj, best_logged
        if obj > heuristic_best_obj:
            heuristic_best_obj = obj
            heuristic_best_items = items[:]
        if logger and obj > best_logged + 1e-9:
            logger.log(float(obj))
            best_logged = float(obj)

    indices = list(range(n))

    # Deterministic orderings
    orders = []
    orders.append(sorted(indices, key=lambda i: profits[i] / max(1, weights[i]), reverse=True))
    orders.append(sorted(indices, key=lambda i: profits[i], reverse=True))
    orders.append(sorted(indices, key=lambda i: profits[i] / (weights[i] + 1 + 0.2 * degrees[i]), reverse=True))
    orders.append(sorted(indices, key=lambda i: (profits[i], -weights[i]), reverse=True))

    # Randomized perturbations (bounded effort)
    rng = random.Random(0)
    for _ in range(8):
        alpha = rng.uniform(0.0, 0.6)
        beta = rng.uniform(0.8, 1.2)
        order = sorted(
            indices,
            key=lambda i: (profits[i] ** beta) / (max(1, weights[i]) * (1.0 + alpha * degrees[i])),
            reverse=True,
        )
        orders.append(order)

    for order in orders:
        if time.time() - start_time >= max(0.0, args.time_limit - 0.2):
            break
        items, obj = greedy_construct(order, weights, profits, capacity, adj)
        improved_set = improve_by_single_add(set(items), weights, profits, capacity, adj)
        items2 = sorted(improved_set)
        obj2 = sum(profits[i] for i in items2)
        try_update(items2, obj2)

    # --- MIP model ---
    model = gp.Model("knapsack_conflict")
    model.Params.TimeLimit = max(0, args.time_limit - (time.time() - start_time))
    model.Params.Threads = 1
    model.Params.OutputFlag = 0
    model.Params.MIPFocus = 1

    x = model.addVars(n, vtype=GRB.BINARY, name="x")

    model.setObjective(gp.quicksum(profits[i] * x[i] for i in range(n)), GRB.MAXIMIZE)
    model.addConstr(gp.quicksum(weights[i] * x[i] for i in range(n)) <= capacity, name="capacity")

    for (u, v) in edges:
        if u == v:
            model.addConstr(x[u] == 0, name=f"self_{u}")
        else:
            model.addConstr(x[u] + x[v] <= 1, name=f"conf_{u}_{v}")

    # Warm start
    if heuristic_best_items:
        chosen_set = set(heuristic_best_items)
        for i in range(n):
            x[i].Start = 1.0 if i in chosen_set else 0.0

    model._logger = logger
    model._best_logged = best_logged

    def cb(m, where):
        if where == GRB.Callback.MIPSOL and m._logger is not None:
            obj = m.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj > m._best_logged + 1e-9:
                m._logger.log(float(obj))
                m._best_logged = float(obj)

    model.optimize(cb)

    # Gather best solution found
    selected_internal = []
    objective_value = 0.0

    if model.SolCount > 0:
        selected_internal = [i for i in range(n) if x[i].X > 0.5]
        objective_value = float(sum(profits[i] for i in selected_internal))
    else:
        # Fallback to heuristic (always feasible as empty set at minimum)
        selected_internal = heuristic_best_items[:]
        objective_value = float(sum(profits[i] for i in selected_internal))

    # Convert back to original indexing convention
    if base == 1:
        selected_items = [i + 1 for i in sorted(selected_internal)]
    else:
        selected_items = sorted(selected_internal)

    output = {
        "objective_value": float(objective_value),
        "selected_items": selected_items,
    }

    with open(args.solution_path, "w") as f:
        json.dump(output, f)


if __name__ == "__main__":
    main()
import argparse
import json
import random
import time
from typing import List, Tuple, Dict, Set

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def build_graph(num_nodes: int, edges: List[List[int]]) -> Tuple[List[Set[int]], List[Tuple[int, int]]]:
    adj = [set() for _ in range(num_nodes)]
    clean_edges = []
    seen = set()
    for e in edges:
        u, v = int(e[0]), int(e[1])
        if u == v:
            continue
        if u > v:
            u, v = v, u
        if (u, v) in seen:
            continue
        seen.add((u, v))
        adj[u].add(v)
        adj[v].add(u)
        clean_edges.append((u, v))
    return adj, clean_edges


def greedy_coloring_from_order(order: List[int], adj: List[Set[int]]) -> Tuple[List[int], int]:
    n = len(adj)
    color = [-1] * n
    for v in order:
        used = set()
        for nb in adj[v]:
            c = color[nb]
            if c != -1:
                used.add(c)
        c = 0
        while c in used:
            c += 1
        color[v] = c
    k = max(color) + 1 if n > 0 else 0
    return color, k


def dsatur_coloring(adj: List[Set[int]]) -> Tuple[List[int], int]:
    n = len(adj)
    if n == 0:
        return [], 0

    degrees = [len(adj[v]) for v in range(n)]
    color = [-1] * n
    neighbor_colors = [set() for _ in range(n)]
    uncolored = set(range(n))

    # First vertex: highest degree
    v0 = max(range(n), key=lambda v: degrees[v])
    color[v0] = 0
    uncolored.remove(v0)
    for nb in adj[v0]:
        neighbor_colors[nb].add(0)

    while uncolored:
        v = max(uncolored, key=lambda u: (len(neighbor_colors[u]), degrees[u], -u))
        used = neighbor_colors[v]
        c = 0
        while c in used:
            c += 1
        color[v] = c
        uncolored.remove(v)
        for nb in adj[v]:
            if color[nb] == -1:
                neighbor_colors[nb].add(c)

    k = max(color) + 1
    return color, k


def compact_coloring(coloring: List[int]) -> Tuple[List[int], int]:
    mapping = {}
    nxt = 0
    new_col = []
    for c in coloring:
        if c not in mapping:
            mapping[c] = nxt
            nxt += 1
        new_col.append(mapping[c])
    return new_col, nxt


def greedy_clique_heuristic(adj: List[Set[int]], max_starts: int = 30) -> List[int]:
    n = len(adj)
    if n == 0:
        return []
    degrees = [len(adj[v]) for v in range(n)]
    starts = sorted(range(n), key=lambda v: degrees[v], reverse=True)[:max_starts]
    best = []

    for s in starts:
        clique = [s]
        cand = set(adj[s])
        while cand:
            v = max(cand, key=lambda u: (len(adj[u] & cand), degrees[u]))
            clique.append(v)
            cand &= adj[v]
        if len(clique) > len(best):
            best = clique

    return best


def run_heuristics(
    adj: List[Set[int]],
    time_limit: int,
    start_time: float,
    logger: SolutionLogger = None
) -> Tuple[List[int], int]:
    n = len(adj)
    if n == 0:
        return [], 0

    degrees = [len(adj[v]) for v in range(n)]
    best_coloring = None
    best_k = 10**9

    def try_solution(sol: List[int], k: int):
        nonlocal best_coloring, best_k
        if k < best_k:
            best_coloring, best_k = sol, k
            if logger:
                logger.log(float(k))

    # Deterministic heuristics
    sol, k = dsatur_coloring(adj)
    sol, k = compact_coloring(sol)
    try_solution(sol, k)

    order = sorted(range(n), key=lambda v: degrees[v], reverse=True)  # Welsh-Powell
    sol, k = greedy_coloring_from_order(order, adj)
    sol, k = compact_coloring(sol)
    try_solution(sol, k)

    # Randomized greedy (limited time)
    heur_budget = min(5.0, max(0.5, 0.25 * time_limit))
    deadline = min(start_time + heur_budget, start_time + max(0.0, time_limit - 0.2))

    rnd = random.Random(0)
    while time.time() < deadline:
        order = list(range(n))
        order.sort(key=lambda v: (-degrees[v], rnd.random()))
        sol, k = greedy_coloring_from_order(order, adj)
        if k <= best_k:
            sol, k = compact_coloring(sol)
            try_solution(sol, k)

    return best_coloring, best_k


def solve_with_mip(
    adj: List[Set[int]],
    edges: List[Tuple[int, int]],
    initial_coloring: List[int],
    initial_k: int,
    time_limit_remaining: float,
    logger: SolutionLogger = None,
) -> Tuple[List[int], int]:
    n = len(adj)
    if n == 0:
        return [], 0
    if time_limit_remaining <= 0:
        return initial_coloring, initial_k

    K = initial_k  # palette upper bound from heuristic

    model = gp.Model("graph_coloring")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = max(0.01, time_limit_remaining)
    model.Params.MIPFocus = 1

    x = model.addVars(n, K, vtype=GRB.BINARY, name="x")
    y = model.addVars(K, vtype=GRB.BINARY, name="y")

    # Each vertex gets exactly one color
    for v in range(n):
        model.addConstr(gp.quicksum(x[v, c] for c in range(K)) == 1)

    # Adjacent vertices cannot share a color
    for (u, v) in edges:
        for c in range(K):
            model.addConstr(x[u, c] + x[v, c] <= 1)

    # Activate color usage
    for v in range(n):
        for c in range(K):
            model.addConstr(x[v, c] <= y[c])

    # Symmetry: colors used as prefix
    for c in range(K - 1):
        model.addConstr(y[c] >= y[c + 1])

    # Additional symmetry: vertex v cannot use color > v
    for v in range(n):
        max_c = min(K, v + 1)
        for c in range(max_c, K):
            model.addConstr(x[v, c] == 0)

    # Clique fixing (symmetry breaking + stronger LB)
    clique = greedy_clique_heuristic(adj, max_starts=30)
    for i, v in enumerate(clique):
        if i < K:
            model.addConstr(x[v, i] == 1)

    # Warm start from heuristic
    for v in range(n):
        cv = initial_coloring[v]
        for c in range(K):
            x[v, c].Start = 1.0 if c == cv else 0.0
    for c in range(K):
        y[c].Start = 1.0 if c < initial_k else 0.0

    model.setObjective(gp.quicksum(y[c] for c in range(K)), GRB.MINIMIZE)

    best_logged = {"obj": float(initial_k)}

    def callback(m, where):
        if where == GRB.Callback.MIPSOL and logger is not None:
            obj = m.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj + 1e-6 < best_logged["obj"]:
                best_logged["obj"] = obj
                logger.log(float(obj))

    model.optimize(callback)

    if model.SolCount == 0:
        return initial_coloring, initial_k

    coloring = [-1] * n
    for v in range(n):
        chosen = None
        for c in range(K):
            if x[v, c].X > 0.5:
                chosen = c
                break
        if chosen is None:
            # fallback (shouldn't happen)
            chosen = initial_coloring[v]
        coloring[v] = chosen

    coloring, k = compact_coloring(coloring)
    return coloring, k


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    t0 = time.time()

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    n = int(inst["num_nodes"])
    edges_input = inst["edges"]

    adj, edges = build_graph(n, edges_input)

    # Quick trivial cases
    if n == 0:
        solution = {"objective_value": 0.0, "coloring": {}}
        with open(args.solution_path, "w") as f:
            json.dump(solution, f)
        return
    if len(edges) == 0:
        coloring = {str(v): 0 for v in range(n)}
        solution = {"objective_value": 1.0, "coloring": coloring}
        if logger:
            logger.log(1.0)
        with open(args.solution_path, "w") as f:
            json.dump(solution, f)
        return

    # Heuristic phase
    best_coloring, best_k = run_heuristics(adj, args.time_limit, t0, logger)

    # MIP improvement phase
    elapsed = time.time() - t0
    remaining = args.time_limit - elapsed
    final_coloring, final_k = solve_with_mip(
        adj, edges, best_coloring, best_k, remaining, logger
    )

    out_coloring = {str(v): int(final_coloring[v]) for v in range(n)}
    solution = {
        "objective_value": float(final_k),
        "coloring": out_coloring
    }

    with open(args.solution_path, "w") as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()
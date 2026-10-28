import argparse
import json
import time
import random
from collections import defaultdict

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def build_adjacency(n, edges, adjacency_list=None):
    if adjacency_list is not None and len(adjacency_list) == n:
        return [set(neigh) for neigh in adjacency_list]
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def dsatur_coloring(adj, randomized=False, rng=None):
    n = len(adj)
    if n == 0:
        return [], 0

    degrees = [len(adj[v]) for v in range(n)]
    colors = [-1] * n
    neighbor_colors = [set() for _ in range(n)]
    uncolored = set(range(n))
    max_color = -1

    while uncolored:
        best_sat = -1
        best_deg = -1
        candidates = []

        for v in uncolored:
            sat = len(neighbor_colors[v])
            deg = degrees[v]
            if sat > best_sat or (sat == best_sat and deg > best_deg):
                best_sat = sat
                best_deg = deg
                candidates = [v]
            elif sat == best_sat and deg == best_deg:
                candidates.append(v)

        if randomized and rng is not None and len(candidates) > 1:
            v = rng.choice(candidates)
        else:
            v = candidates[0]

        forbidden = neighbor_colors[v]
        if randomized and rng is not None:
            feasible = [c for c in range(max_color + 2) if c not in forbidden]
            rcl = feasible[: min(3, len(feasible))]
            if len(rcl) == 1 or rng.random() < 0.7:
                chosen = rcl[0]
            else:
                chosen = rng.choice(rcl)
        else:
            chosen = 0
            while chosen in forbidden:
                chosen += 1

        colors[v] = chosen
        if chosen > max_color:
            max_color = chosen
        uncolored.remove(v)

        for u in adj[v]:
            if colors[u] == -1:
                neighbor_colors[u].add(chosen)

    return colors, max_color + 1


def normalize_coloring(colors):
    if not colors:
        return [], 0
    used = sorted(set(colors))
    remap = {c: i for i, c in enumerate(used)}
    norm = [remap[c] for c in colors]
    return norm, len(used)


def reduce_coloring(colors, adj):
    n = len(colors)
    if n == 0:
        return colors, 0

    colors = colors[:]  # work on copy
    while True:
        colors, k = normalize_coloring(colors)
        if k <= 1:
            return colors, k

        target = k - 1
        target_vertices = [v for v in range(n) if colors[v] == target]
        changed = False

        for v in target_vertices:
            for c in range(target):
                ok = True
                for u in adj[v]:
                    if colors[u] == c:
                        ok = False
                        break
                if ok:
                    colors[v] = c
                    changed = True
                    break

        # If we couldn't move any vertex from highest color and it's still non-empty, stop
        if not changed and any(colors[v] == target for v in target_vertices):
            break

    return normalize_coloring(colors)


def is_proper_coloring(colors, adj):
    for v in range(len(adj)):
        cv = colors[v]
        for u in adj[v]:
            if u > v and colors[u] == cv:
                return False
    return True


def solve_with_mip(n, edges, initial_colors, time_limit, logger=None):
    initial_colors, ub = normalize_coloring(initial_colors)
    if n == 0:
        return [], 0

    model = gp.Model("graph_coloring")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = max(0.0, time_limit)

    C = range(ub)
    x = model.addVars(n, ub, vtype=GRB.BINARY, name="x")
    y = model.addVars(ub, vtype=GRB.BINARY, name="y")

    # Every vertex gets exactly one color
    for v in range(n):
        model.addConstr(gp.quicksum(x[v, c] for c in C) == 1)

    # Link x and y
    for v in range(n):
        for c in C:
            model.addConstr(x[v, c] <= y[c])

    # Adjacent vertices cannot share same color
    for (u, v) in edges:
        for c in C:
            model.addConstr(x[u, c] + x[v, c] <= 1)

    # Symmetry breaking: use lower indexed colors first
    for c in range(ub - 1):
        model.addConstr(y[c] >= y[c + 1])

    # Strong symmetry break
    model.addConstr(x[0, 0] == 1)

    model.setObjective(gp.quicksum(y[c] for c in C), GRB.MINIMIZE)

    # Warm start from heuristic
    used = set(initial_colors)
    for v, c in enumerate(initial_colors):
        x[v, c].Start = 1.0
    for c in used:
        y[c].Start = 1.0

    best_obj_seen = [ub]

    def cb(m, where):
        if where == GRB.Callback.MIPSOL:
            obj = m.cbGet(GRB.Callback.MIPSOL_OBJ)
            val = int(round(obj))
            if val < best_obj_seen[0]:
                best_obj_seen[0] = val
                if logger:
                    logger.log(val)

    model.optimize(cb)

    if model.SolCount == 0:
        return initial_colors, ub

    sol = [-1] * n
    for v in range(n):
        chosen = None
        for c in C:
            if x[v, c].X > 0.5:
                chosen = c
                break
        if chosen is None:
            chosen = 0
        sol[v] = chosen

    sol, k = normalize_coloring(sol)
    return sol, k


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    graph = data["graph"]
    n = int(graph["n"])
    edges = [tuple(e) for e in graph["edges"]]
    adjacency_list = graph.get("adjacency_list", None)

    adj = build_adjacency(n, edges, adjacency_list=adjacency_list)

    if n == 0:
        result = {
            "objective_value": 0,
            "coloring": {},
            "color_classes": {}
        }
        with open(args.solution_path, "w") as f:
            json.dump(result, f, indent=2)
        return

    start = time.perf_counter()
    rng = random.Random(0)

    # Initial deterministic DSATUR
    best_colors, best_k = dsatur_coloring(adj, randomized=False, rng=rng)
    best_colors, best_k = reduce_coloring(best_colors, adj)

    if logger:
        logger.log(best_k)

    # Multi-start randomized DSATUR (time-bounded)
    heuristic_budget = min(10.0, max(0.5, 0.25 * args.time_limit))
    while time.perf_counter() - start < heuristic_budget:
        cands, ck = dsatur_coloring(adj, randomized=True, rng=rng)
        cands, ck = reduce_coloring(cands, adj)
        if ck < best_k and is_proper_coloring(cands, adj):
            best_colors, best_k = cands, ck
            if logger:
                logger.log(best_k)

    # Try MIP polishing/exact if model size is reasonable
    elapsed = time.perf_counter() - start
    remaining = args.time_limit - elapsed

    # Limit model size for robustness
    mip_size = len(edges) * max(1, best_k)
    if remaining > 0.5 and mip_size <= 2_000_000:
        mip_colors, mip_k = solve_with_mip(
            n=n,
            edges=edges,
            initial_colors=best_colors,
            time_limit=remaining,
            logger=logger
        )
        if mip_k < best_k and is_proper_coloring(mip_colors, adj):
            best_colors, best_k = mip_colors, mip_k
            if logger:
                logger.log(best_k)

    # Final normalization / formatting
    best_colors, best_k = normalize_coloring(best_colors)

    coloring = {str(v): int(best_colors[v]) for v in range(n)}
    classes = defaultdict(list)
    for v, c in enumerate(best_colors):
        classes[str(c)].append(v)

    color_classes = {k: v for k, v in classes.items()}

    result = {
        "objective_value": int(best_k),
        "coloring": coloring,
        "color_classes": color_classes
    }

    with open(args.solution_path, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
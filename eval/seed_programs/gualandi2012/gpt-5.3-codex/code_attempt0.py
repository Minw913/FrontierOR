import argparse
import json
import time
import random
from collections import defaultdict

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def load_instance(path):
    with open(path, "r") as f:
        data = json.load(f)
    g = data["graph"]
    n = int(g["num_vertices"])
    m = int(g["num_edges"])
    edges = [tuple(map(int, e)) for e in g["edges"]]
    # Trust schema, but keep consistent with declared counts when possible
    if m != len(edges):
        m = len(edges)
    return n, m, edges


def build_adjacency(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        if u == v:
            continue
        adj[u].add(v)
        adj[v].add(u)
    return adj


def dsatur_coloring(adj, randomize=False, rng=None):
    n = len(adj)
    if n == 0:
        return {}, 0

    degrees = [len(adj[v]) for v in range(n)]
    uncolored = set(range(n))
    color_of = {}
    neighbor_colors = [set() for _ in range(n)]

    while uncolored:
        # Choose vertex with max saturation, tie by degree, optional random tie-break
        max_sat = -1
        max_deg = -1
        candidates = []
        for v in uncolored:
            sat = len(neighbor_colors[v])
            deg = degrees[v]
            if sat > max_sat or (sat == max_sat and deg > max_deg):
                max_sat = sat
                max_deg = deg
                candidates = [v]
            elif sat == max_sat and deg == max_deg:
                candidates.append(v)

        if randomize and rng is not None and len(candidates) > 1:
            v = rng.choice(candidates)
        else:
            v = min(candidates)

        used = neighbor_colors[v]
        c = 0
        while c in used:
            c += 1

        color_of[v] = c
        uncolored.remove(v)

        for nb in adj[v]:
            if nb in uncolored:
                neighbor_colors[nb].add(c)

    k = max(color_of.values()) + 1 if color_of else 0
    return color_of, k


def greedy_clique_lower_bound(adj):
    """Simple greedy maximal clique heuristic -> lower bound on chromatic number."""
    n = len(adj)
    if n == 0:
        return 0
    order = sorted(range(n), key=lambda v: len(adj[v]), reverse=True)
    clique = []
    for v in order:
        if all((u in adj[v]) for u in clique):
            clique.append(v)
    return max(1, len(clique)) if n > 0 else 0


def normalize_coloring(coloring):
    if not coloring:
        return {}, 0
    used = sorted(set(coloring.values()))
    remap = {c: i for i, c in enumerate(used)}
    new_coloring = {v: remap[c] for v, c in coloring.items()}
    return new_coloring, len(used)


def build_output(coloring, objective_value):
    color_classes = defaultdict(list)
    for v, c in coloring.items():
        color_classes[str(c)].append(v)
    for c in color_classes:
        color_classes[c].sort()

    out = {
        "objective_value": int(objective_value),
        "coloring": {str(v): int(c) for v, c in sorted(coloring.items())},
        "color_classes": {k: color_classes[k] for k in sorted(color_classes, key=lambda x: int(x))},
    }
    return out


def solve_with_mip(n, edges, initial_coloring, initial_ub, lb, time_limit, logger=None):
    if n == 0:
        return {}, 0

    K = initial_ub  # only need up to current best colors
    if K <= 0:
        return initial_coloring, initial_ub

    model = gp.Model("graph_coloring")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = max(0.0, float(time_limit))
    model.Params.MIPFocus = 1  # emphasize finding improved incumbents
    model.Params.Heuristics = 0.2

    x = model.addVars(n, K, vtype=GRB.BINARY, name="x")
    y = model.addVars(K, vtype=GRB.BINARY, name="y")

    # Each vertex gets exactly one color
    for v in range(n):
        model.addConstr(gp.quicksum(x[v, c] for c in range(K)) == 1, name=f"assign_{v}")

    # Adjacent vertices cannot share a color
    for (u, v) in edges:
        if u == v:
            continue
        for c in range(K):
            model.addConstr(x[u, c] + x[v, c] <= 1, name=f"edge_{u}_{v}_{c}")

    # Link x and y
    for v in range(n):
        for c in range(K):
            model.addConstr(x[v, c] <= y[c], name=f"link_{v}_{c}")

    # Symmetry breaking: enforce ordered usage of colors
    for c in range(K - 1):
        model.addConstr(y[c] >= y[c + 1], name=f"sym_{c}")

    # Lower bound (helpful cut)
    if lb > 0:
        model.addConstr(gp.quicksum(y[c] for c in range(K)) >= lb, name="lb_colors")

    # Fix a high-degree vertex to color 0 for extra symmetry breaking
    # (safe as color names are interchangeable)
    # pick vertex 0 by default (works even if not highest degree)
    model.addConstr(x[0, 0] == 1, name="fix_v0_color0")

    model.setObjective(gp.quicksum(y[c] for c in range(K)), GRB.MINIMIZE)

    # Warm start from initial coloring
    for v in range(n):
        c0 = initial_coloring.get(v, 0)
        c0 = min(c0, K - 1)
        for c in range(K):
            x[v, c].Start = 1.0 if c == c0 else 0.0
    for c in range(K):
        y[c].Start = 1.0 if c < initial_ub else 0.0

    # Stop early if reaches known lower bound
    if lb > 0:
        model.Params.BestObjStop = float(lb)

    model._logger = logger
    model._best_logged = float(initial_ub)

    def callback(cb_model, where):
        if where == GRB.Callback.MIPSOL:
            obj = cb_model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj + 1e-9 < cb_model._best_logged:
                cb_model._best_logged = obj
                if cb_model._logger:
                    cb_model._logger.log(int(round(obj)))

    model.optimize(callback)

    best_coloring = dict(initial_coloring)
    best_obj = initial_ub

    if model.SolCount > 0:
        sol_coloring = {}
        for v in range(n):
            assigned = None
            for c in range(K):
                if x[v, c].X > 0.5:
                    assigned = c
                    break
            if assigned is None:
                assigned = 0
            sol_coloring[v] = assigned

        sol_coloring, sol_obj = normalize_coloring(sol_coloring)
        if sol_obj < best_obj:
            best_coloring = sol_coloring
            best_obj = sol_obj
            if logger:
                logger.log(int(best_obj))
        else:
            # still return possibly equivalent normalized incumbent
            best_coloring = sol_coloring
            best_obj = sol_obj

    return best_coloring, best_obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, required=False, default=None)
    args = parser.parse_args()

    start_time = time.time()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    n, m, edges = load_instance(args.instance_path)

    # Trivial case
    if n == 0:
        solution = {
            "objective_value": 0,
            "coloring": {},
            "color_classes": {}
        }
        with open(args.solution_path, "w") as f:
            json.dump(solution, f, indent=2)
        return

    adj = build_adjacency(n, edges)

    # Initial deterministic DSATUR
    best_coloring, best_obj = dsatur_coloring(adj, randomize=False, rng=None)
    best_coloring, best_obj = normalize_coloring(best_coloring)
    if logger:
        logger.log(int(best_obj))

    # Short randomized DSATUR improvement phase
    rng = random.Random(42)
    heuristic_budget = min(2.0, max(0.0, 0.15 * args.time_limit))
    while time.time() - start_time < heuristic_budget:
        cand_coloring, cand_obj = dsatur_coloring(adj, randomize=True, rng=rng)
        cand_coloring, cand_obj = normalize_coloring(cand_coloring)
        if cand_obj < best_obj:
            best_coloring, best_obj = cand_coloring, cand_obj
            if logger:
                logger.log(int(best_obj))

    # Lower bound heuristic
    lb = greedy_clique_lower_bound(adj)
    lb = min(lb, best_obj)

    # If already at LB, skip MIP
    elapsed = time.time() - start_time
    remaining = args.time_limit - elapsed

    if remaining > 0.2 and best_obj > lb:
        mip_coloring, mip_obj = solve_with_mip(
            n=n,
            edges=edges,
            initial_coloring=best_coloring,
            initial_ub=best_obj,
            lb=lb,
            time_limit=remaining,
            logger=logger
        )
        if mip_obj < best_obj:
            best_coloring, best_obj = mip_coloring, mip_obj
        else:
            # keep normalized MIP solution if same objective
            best_coloring, best_obj = mip_coloring, mip_obj

    best_coloring, best_obj = normalize_coloring(best_coloring)
    solution = build_output(best_coloring, best_obj)

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()
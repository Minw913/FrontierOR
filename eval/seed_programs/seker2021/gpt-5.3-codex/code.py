import argparse
import json
import random
import time
from typing import Dict, List, Tuple, Optional

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def dsatur_coloring(selected_vertices: List[int], adj: List[set]) -> Tuple[int, Dict[int, int]]:
    """Greedy DSATUR coloring on the induced subgraph of selected vertices."""
    selected_set = set(selected_vertices)
    induced_adj = {v: set() for v in selected_vertices}
    for v in selected_vertices:
        induced_adj[v] = adj[v].intersection(selected_set)

    colors = {}
    neighbor_colors = {v: set() for v in selected_vertices}
    uncolored = set(selected_vertices)

    while uncolored:
        # Pick max saturation, break ties by degree
        v = max(uncolored, key=lambda x: (len(neighbor_colors[x]), len(induced_adj[x])))
        used = neighbor_colors[v]
        c = 0
        while c in used:
            c += 1
        colors[v] = c
        uncolored.remove(v)

        for nb in induced_adj[v]:
            if nb in uncolored:
                neighbor_colors[nb].add(c)

    num_colors = 0 if not colors else (max(colors.values()) + 1)
    return num_colors, colors


def build_initial_selection(clusters: List[List[int]], cross_degree: List[int]) -> Dict[int, int]:
    """Pick low cross-degree vertex in each cluster."""
    sel = {}
    for c, verts in enumerate(clusters):
        best_v = min(verts, key=lambda v: cross_degree[v])
        sel[c] = best_v
    return sel


def evaluate_selection(selection: Dict[int, int], adj: List[set]) -> Tuple[int, Dict[int, int]]:
    verts = [selection[c] for c in sorted(selection.keys())]
    return dsatur_coloring(verts, adj)


def heuristic_search(
    clusters: List[List[int]],
    adj: List[set],
    cross_degree: List[int],
    time_budget: float,
    seed: int = 0,
):
    random.seed(seed)
    start = time.perf_counter()

    best_sel = build_initial_selection(clusters, cross_degree)
    best_obj, best_col = evaluate_selection(best_sel, adj)

    while time.perf_counter() - start < time_budget:
        sel = {}
        for c, verts in enumerate(clusters):
            sel[c] = random.choice(verts)
        obj, col = evaluate_selection(sel, adj)
        if obj < best_obj:
            best_obj, best_sel, best_col = obj, sel, col

    return best_obj, best_sel, best_col


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    t0 = time.perf_counter()

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    n = data["graph"]["num_vertices"]
    edges = data["graph"]["edges"]
    clusters = data["partition"]["clusters"]
    num_clusters = data["partition"]["num_clusters"]

    # Build cluster map
    cluster_of = [-1] * n
    for c, verts in enumerate(clusters):
        for v in verts:
            cluster_of[v] = c

    # Adjacency and filtered cross-cluster edges
    adj = [set() for _ in range(n)]
    cross_edges = []
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        if cluster_of[u] != cluster_of[v]:
            cross_edges.append((u, v))

    # Cross-degree for heuristic
    cross_degree = [0] * n
    for v in range(n):
        cv = cluster_of[v]
        cnt = 0
        for nb in adj[v]:
            if cluster_of[nb] != cv:
                cnt += 1
        cross_degree[v] = cnt

    # Heuristic phase
    heuristic_budget = min(max(0.2, 0.15 * args.time_limit), 5.0)
    h_obj, h_sel, h_col = heuristic_search(clusters, adj, cross_degree, heuristic_budget, seed=0)

    best_logged = float("inf")
    if logger and h_obj < best_logged:
        logger.log(float(h_obj))
        best_logged = float(h_obj)

    # Use heuristic UB as max number of color slots
    K = max(1, int(h_obj))
    if K > num_clusters:
        K = num_clusters

    # Build MIP
    model = gp.Model("cluster_select_color")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    model.Params.MIPFocus = 1

    elapsed = time.perf_counter() - t0
    remaining = max(1.0, args.time_limit - elapsed)
    model.Params.TimeLimit = remaining

    x = model.addVars(n, K, vtype=GRB.BINARY, name="x")  # x[v,k] = vertex v selected with color k
    w = model.addVars(K, vtype=GRB.BINARY, name="w")     # color usage

    # One selected (and colored) vertex per cluster
    for c, verts in enumerate(clusters):
        model.addConstr(gp.quicksum(x[v, k] for v in verts for k in range(K)) == 1, name=f"cluster_{c}")

    # Edge conflicts: adjacent selected vertices cannot share a color
    model.addConstrs(
        (x[u, k] + x[v, k] <= 1 for (u, v) in cross_edges for k in range(K)),
        name="conflict"
    )

    # Link color usage
    model.addConstrs((x[v, k] <= w[k] for v in range(n) for k in range(K)), name="use_link")

    # Symmetry-breaking on colors
    for k in range(K - 1):
        model.addConstr(w[k] >= w[k + 1], name=f"sym_{k}")

    model.setObjective(gp.quicksum(w[k] for k in range(K)), GRB.MINIMIZE)

    # MIP start from heuristic
    # Compress heuristic colors to [0..h_obj-1], then use if <= K
    if h_col:
        old_to_new = {}
        next_c = 0
        comp_col = {}
        for v, c in h_col.items():
            if c not in old_to_new:
                old_to_new[c] = next_c
                next_c += 1
            comp_col[v] = old_to_new[c]

        if next_c <= K:
            for c in range(num_clusters):
                v = h_sel[c]
                k = comp_col[v]
                x[v, k].Start = 1.0
            for k in range(next_c):
                w[k].Start = 1.0

    model._logger = logger
    model._best_logged = best_logged

    def cb(m, where):
        if where == GRB.Callback.MIPSOL and m._logger is not None:
            obj = m.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj + 1e-6 < m._best_logged:
                m._best_logged = obj
                m._logger.log(float(obj))

    model.optimize(cb)

    selected_vertices = {}
    coloring = {}
    objective_value = None

    if model.SolCount > 0:
        # Extract best MIP solution
        used_colors = set()
        for c, verts in enumerate(clusters):
            chosen_v = None
            chosen_k = None
            for v in verts:
                for k in range(K):
                    if x[v, k].X > 0.5:
                        chosen_v = v
                        chosen_k = k
                        break
                if chosen_v is not None:
                    break

            # Robust fallback if numerical weirdness
            if chosen_v is None:
                best_val = -1.0
                for v in verts:
                    for k in range(K):
                        val = x[v, k].X
                        if val > best_val:
                            best_val = val
                            chosen_v = v
                            chosen_k = k
                if chosen_k is None:
                    chosen_k = 0

            selected_vertices[str(c)] = int(chosen_v)
            coloring[str(chosen_v)] = int(chosen_k + 1)  # 1-based color ids
            used_colors.add(chosen_k)

        objective_value = float(len(used_colors))
    else:
        # Fallback to heuristic solution
        used = set()
        for c in range(num_clusters):
            v = h_sel[c]
            selected_vertices[str(c)] = int(v)
            col = h_col[v] + 1
            coloring[str(v)] = int(col)
            used.add(col)
        objective_value = float(len(used))

    sol = {
        "objective_value": objective_value,
        "selected_vertices": selected_vertices,
        "coloring": coloring
    }

    with open(args.solution_path, "w") as f:
        json.dump(sol, f, indent=2)


if __name__ == "__main__":
    main()
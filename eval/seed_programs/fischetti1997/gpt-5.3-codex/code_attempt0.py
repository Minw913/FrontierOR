import argparse
import json
import math
import time
from typing import List, Dict, Tuple, Optional

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def build_distance_matrix(n: int, upper_vals: List[int]) -> List[List[int]]:
    d = [[0] * n for _ in range(n)]
    idx = 0
    for i in range(n - 1):
        row = d[i]
        for j in range(i + 1, n):
            val = int(upper_vals[idx])
            idx += 1
            row[j] = val
            d[j][i] = val
    return d


def cycle_cost(tour: List[int], dist: List[List[int]]) -> float:
    m = len(tour)
    if m <= 1:
        return 0.0
    if m == 2:
        return float(2 * dist[tour[0]][tour[1]])
    return float(sum(dist[tour[i]][tour[(i + 1) % m]] for i in range(m)))


def nearest_neighbor_cluster_order(cluster_min_dist: List[List[float]]) -> List[int]:
    m = len(cluster_min_dist)
    if m <= 1:
        return [0] if m == 1 else []
    unvisited = set(range(1, m))
    order = [0]
    while unvisited:
        last = order[-1]
        nxt = min(unvisited, key=lambda c: cluster_min_dist[last][c])
        order.append(nxt)
        unvisited.remove(nxt)
    return order


def two_opt_cluster_order(order: List[int], cluster_min_dist: List[List[float]], max_iter: int = 50) -> List[int]:
    m = len(order)
    if m < 4:
        return order[:]

    def edge(a, b):
        return cluster_min_dist[a][b]

    improved = True
    it = 0
    best = order[:]
    while improved and it < max_iter:
        improved = False
        it += 1
        for i in range(m - 1):
            a = best[i]
            b = best[(i + 1) % m]
            for k in range(i + 2, m if i > 0 else m - 1):
                c = best[k]
                d = best[(k + 1) % m]
                old = edge(a, b) + edge(c, d)
                new = edge(a, c) + edge(b, d)
                if new + 1e-9 < old:
                    best[i + 1 : k + 1] = reversed(best[i + 1 : k + 1])
                    improved = True
        if not improved:
            break
    return best


def best_nodes_for_fixed_cluster_order(
    cluster_order: List[int],
    cluster_nodes: List[List[int]],
    dist: List[List[int]],
) -> List[int]:
    m = len(cluster_order)
    if m == 0:
        return []
    if m == 1:
        return [cluster_nodes[cluster_order[0]][0]]

    first_cluster_nodes = cluster_nodes[cluster_order[0]]
    best_total = math.inf
    best_path = None

    for s in first_cluster_nodes:
        dp_prev = {s: 0.0}
        parents = [None] * m  # parents[k][v] = u in previous layer

        feasible = True
        for layer in range(1, m):
            curr_nodes = cluster_nodes[cluster_order[layer]]
            prev_nodes = list(dp_prev.keys())
            if not prev_nodes:
                feasible = False
                break
            pmap = {}
            dp_curr = {}
            for v in curr_nodes:
                best_val = math.inf
                best_u = None
                for u in prev_nodes:
                    val = dp_prev[u] + dist[u][v]
                    if val < best_val:
                        best_val = val
                        best_u = u
                dp_curr[v] = best_val
                pmap[v] = best_u
            parents[layer] = pmap
            dp_prev = dp_curr

        if not feasible:
            continue

        best_last = None
        best_cycle_val = math.inf
        for v, val in dp_prev.items():
            tot = val + dist[v][s]
            if tot < best_cycle_val:
                best_cycle_val = tot
                best_last = v

        if best_cycle_val < best_total:
            path = [None] * m
            path[-1] = best_last
            for layer in range(m - 1, 0, -1):
                path[layer - 1] = parents[layer][path[layer]]
            best_total = best_cycle_val
            best_path = path

    return best_path if best_path is not None else [cluster_nodes[c][0] for c in cluster_order]


def build_cluster_min_dist(
    n: int,
    m: int,
    node_cluster_pos: List[int],
    dist: List[List[int]],
) -> List[List[float]]:
    inf = float("inf")
    cmd = [[inf] * m for _ in range(m)]
    for c in range(m):
        cmd[c][c] = 0.0
    for i in range(n - 1):
        ci = node_cluster_pos[i]
        di = dist[i]
        for j in range(i + 1, n):
            cj = node_cluster_pos[j]
            if ci == cj:
                continue
            v = di[j]
            if v < cmd[ci][cj]:
                cmd[ci][cj] = float(v)
                cmd[cj][ci] = float(v)
    # Fallback for any missing pair (should not happen in valid instances)
    for i in range(m):
        for j in range(m):
            if i != j and math.isinf(cmd[i][j]):
                cmd[i][j] = 1e9
    return cmd


def extract_tour_from_edges(selected: List[int], edges: List[Tuple[int, int]]) -> Optional[List[int]]:
    m = len(selected)
    if m == 0:
        return []
    if m == 1:
        return [selected[0]]
    if m == 2:
        return selected[:]

    adj = {i: [] for i in selected}
    for i, j in edges:
        if i in adj and j in adj:
            adj[i].append(j)
            adj[j].append(i)

    # Validate degree 2
    for i in selected:
        if len(adj[i]) != 2:
            return None

    start = selected[0]
    tour = []
    prev = None
    curr = start
    for _ in range(m):
        tour.append(curr)
        nxts = adj[curr]
        nxt = nxts[0] if nxts[0] != prev else nxts[1]
        prev, curr = curr, nxt

    if curr != start:
        return None
    if len(set(tour)) != m:
        return None
    return tour


def solve_instance(data: dict, time_limit: int, logger: Optional[SolutionLogger]):
    n = int(data["problem_size"]["n"])
    m = int(data["problem_size"]["m"])

    nodes = data["nodes"]
    clusters = data["clusters"]
    upper_vals = data["distance_matrix_upper_triangular"]["values"]

    # Node index mapping
    node_ids = [int(nd["id"]) for nd in nodes]
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    # Cluster mapping by cluster list order -> position [0..m-1]
    cluster_ids = [int(c["id"]) for c in clusters]
    cluster_id_to_pos = {cid: p for p, cid in enumerate(cluster_ids)}

    # cluster_nodes as index lists
    cluster_nodes: List[List[int]] = []
    for c in clusters:
        indices = [id_to_idx[int(nid)] for nid in c["node_ids"]]
        cluster_nodes.append(indices)

    # Node -> cluster position
    node_cluster_pos = [None] * n
    for nd in nodes:
        i = id_to_idx[int(nd["id"])]
        node_cluster_pos[i] = cluster_id_to_pos[int(nd["cluster"])]

    dist = build_distance_matrix(n, upper_vals)

    # Handle trivial cases
    if m == 0:
        return 0.0, [], []
    if m == 1:
        idx = cluster_nodes[0][0]
        nid = node_ids[idx]
        return 0.0, [nid], [nid]

    t0 = time.time()

    # Heuristic initial solution
    cluster_min_dist = build_cluster_min_dist(n, m, node_cluster_pos, dist)
    cluster_order = nearest_neighbor_cluster_order(cluster_min_dist)
    cluster_order = two_opt_cluster_order(cluster_order, cluster_min_dist, max_iter=30)
    heuristic_tour_idx = best_nodes_for_fixed_cluster_order(cluster_order, cluster_nodes, dist)
    heuristic_obj = cycle_cost(heuristic_tour_idx, dist)

    best_tour_idx = heuristic_tour_idx[:]
    best_obj = heuristic_obj

    if logger is not None:
        logger.log(float(best_obj))

    # If no time left, return heuristic
    elapsed = time.time() - t0
    remaining = max(0.0, time_limit - elapsed)
    if remaining < 1.0:
        tour_ids = [node_ids[i] for i in best_tour_idx]
        return float(best_obj), tour_ids, tour_ids[:]

    # Build MIP model
    model = gp.Model("gtsp")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = remaining
    model.Params.LazyConstraints = 1
    model.Params.MIPFocus = 1

    y = model.addVars(n, vtype=GRB.BINARY, name="y")

    edge_keys = []
    for i in range(n - 1):
        ci = node_cluster_pos[i]
        for j in range(i + 1, n):
            if ci != node_cluster_pos[j]:
                edge_keys.append((i, j))
    x = model.addVars(edge_keys, vtype=GRB.BINARY, name="x")

    # Cluster selection
    for cpos in range(m):
        model.addConstr(gp.quicksum(y[i] for i in cluster_nodes[cpos]) == 1, name=f"cluster_{cpos}")

    # Degree constraints
    incident = [[] for _ in range(n)]
    for i, j in edge_keys:
        incident[i].append((i, j))
        incident[j].append((i, j))

    for i in range(n):
        model.addConstr(gp.quicksum(x[e] for e in incident[i]) == 2 * y[i], name=f"deg_{i}")

    # Objective
    model.setObjective(gp.quicksum(dist[i][j] * x[i, j] for i, j in edge_keys), GRB.MINIMIZE)

    # MIP start from heuristic
    sel_set = set(heuristic_tour_idx)
    for i in range(n):
        y[i].Start = 1.0 if i in sel_set else 0.0
    edge_start = set()
    if len(heuristic_tour_idx) >= 2:
        for k in range(len(heuristic_tour_idx)):
            a = heuristic_tour_idx[k]
            b = heuristic_tour_idx[(k + 1) % len(heuristic_tour_idx)]
            if a > b:
                a, b = b, a
            edge_start.add((a, b))
    for e in edge_keys:
        x[e].Start = 1.0 if e in edge_start else 0.0

    # Callback state
    model._x = x
    model._y = y
    model._n = n
    model._m = m
    model._logger = logger
    model._best_logged = float(best_obj) if logger is not None else float("inf")

    def callback_cb(cb_model, where):
        if where == GRB.Callback.MIPSOL:
            # incumbent logging
            if cb_model._logger is not None:
                obj = cb_model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if obj + 1e-9 < cb_model._best_logged:
                    cb_model._logger.log(float(obj))
                    cb_model._best_logged = float(obj)

            # subtour elimination on selected nodes
            selected = [i for i in range(cb_model._n) if cb_model.cbGetSolution(cb_model._y[i]) > 0.5]
            if len(selected) <= 1:
                return

            sel_set_local = set(selected)
            adj = {i: [] for i in selected}
            for (i, j), var in cb_model._x.items():
                if i in sel_set_local and j in sel_set_local:
                    if cb_model.cbGetSolution(var) > 0.5:
                        adj[i].append(j)
                        adj[j].append(i)

            # Find connected components
            unvisited = set(selected)
            comps = []
            while unvisited:
                s = next(iter(unvisited))
                stack = [s]
                comp = []
                unvisited.remove(s)
                while stack:
                    u = stack.pop()
                    comp.append(u)
                    for v in adj[u]:
                        if v in unvisited:
                            unvisited.remove(v)
                            stack.append(v)
                comps.append(comp)

            if len(comps) > 1:
                for comp in comps:
                    if len(comp) == len(selected):
                        continue
                    comp_set = set(comp)
                    lhs_terms = []
                    comp_list = sorted(comp)
                    L = len(comp_list)
                    for a_idx in range(L - 1):
                        a = comp_list[a_idx]
                        for b_idx in range(a_idx + 1, L):
                            b = comp_list[b_idx]
                            key = (a, b) if a < b else (b, a)
                            if key in cb_model._x:
                                lhs_terms.append(cb_model._x[key])
                    if lhs_terms:
                        cb_model.cbLazy(gp.quicksum(lhs_terms) <= gp.quicksum(cb_model._y[i] for i in comp_set) - 1)

    model.optimize(callback_cb)

    if model.SolCount > 0:
        selected = [i for i in range(n) if y[i].X > 0.5]
        used_edges = [(i, j) for (i, j) in edge_keys if x[i, j].X > 0.5]
        tour_idx = extract_tour_from_edges(selected, used_edges)

        if tour_idx is None or len(tour_idx) != m:
            # fallback to heuristic if extraction fails
            tour_idx = best_tour_idx
            obj_val = cycle_cost(tour_idx, dist)
        else:
            obj_val = float(model.ObjVal)
            # keep best between model obj and heuristic (safety)
            if obj_val > best_obj + 1e-6:
                tour_idx = best_tour_idx
                obj_val = float(best_obj)
    else:
        tour_idx = best_tour_idx
        obj_val = float(best_obj)

    tour_ids = [node_ids[i] for i in tour_idx]
    visited_ids = tour_ids[:]
    return float(obj_val), tour_ids, visited_ids


def main():
    parser = argparse.ArgumentParser(description="Solve clustered TSP / GTSP instance.")
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    obj, tour, visited = solve_instance(data, args.time_limit, logger)

    out = {
        "objective_value": float(obj),
        "tour": [int(v) for v in tour],
        "visited_nodes": [int(v) for v in visited],
    }

    with open(args.solution_path, "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()
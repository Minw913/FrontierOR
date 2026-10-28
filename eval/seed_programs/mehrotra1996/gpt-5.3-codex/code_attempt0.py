import argparse
import json
import time
import random
from typing import Dict, List, Set, Tuple

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def build_graph(data: dict):
    g = data["graph"]
    nodes = list(g["nodes"])
    node_set = set(nodes)
    edges_in = g["edges"]

    # Build adjacency with deduplicated undirected edges
    adj: Dict[int, Set[int]] = {v: set() for v in nodes}
    edge_set = set()
    for e in edges_in:
        if len(e) != 2:
            continue
        u, v = e
        if u == v:
            continue
        if u not in node_set or v not in node_set:
            continue
        a, b = (u, v) if u < v else (v, u)
        if (a, b) in edge_set:
            continue
        edge_set.add((a, b))
        adj[a].add(b)
        adj[b].add(a)

    edges = list(edge_set)
    return nodes, edges, adj


def dsatur_coloring(
    nodes: List[int],
    adj: Dict[int, Set[int]],
    rng: random.Random = None
) -> Tuple[Dict[int, int], int]:
    if not nodes:
        return {}, 0

    degree = {v: len(adj[v]) for v in nodes}
    uncolored = set(nodes)
    colors: Dict[int, int] = {}
    neighbor_used = {v: set() for v in nodes}

    while uncolored:
        # Choose node with max saturation, then max degree, then deterministic/random tie break
        max_sat = -1
        max_deg = -1
        candidates = []

        for v in uncolored:
            sat = len(neighbor_used[v])
            deg = degree[v]
            if sat > max_sat or (sat == max_sat and deg > max_deg):
                max_sat = sat
                max_deg = deg
                candidates = [v]
            elif sat == max_sat and deg == max_deg:
                candidates.append(v)

        if rng is None:
            v = min(candidates)
        else:
            v = rng.choice(candidates)

        forbidden = neighbor_used[v]
        c = 0
        while c in forbidden:
            c += 1

        colors[v] = c
        uncolored.remove(v)

        for u in adj[v]:
            if u in uncolored:
                neighbor_used[u].add(c)

    num_colors = (max(colors.values()) + 1) if colors else 0
    return colors, num_colors


def greedy_maximal_clique(nodes: List[int], adj: Dict[int, Set[int]]) -> List[int]:
    # Simple heuristic maximal clique (lower bound on chromatic number)
    order = sorted(nodes, key=lambda v: len(adj[v]), reverse=True)
    clique = []
    for v in order:
        ok = True
        for u in clique:
            if u not in adj[v]:
                ok = False
                break
        if ok:
            clique.append(v)
    return clique


def normalize_solution(colors: Dict[int, int]):
    if not colors:
        return {}, [], 0
    used = sorted(set(colors.values()))
    remap = {c: i for i, c in enumerate(used)}
    k = len(used)

    vertex_colors = {}
    color_classes = [[] for _ in range(k)]
    for v, c in colors.items():
        nc = remap[c]
        vertex_colors[str(v)] = nc
        color_classes[nc].append(v)

    for cls in color_classes:
        cls.sort()

    return vertex_colors, color_classes, k


def solve(instance_path: str, solution_path: str, time_limit: int, log_path: str = None):
    start = time.time()
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None

    with open(instance_path, "r") as f:
        data = json.load(f)

    nodes, edges, adj = build_graph(data)
    n = len(nodes)

    if n == 0:
        sol = {"objective_value": 0, "vertex_colors": {}, "color_classes": []}
        with open(solution_path, "w") as f:
            json.dump(sol, f, indent=2)
        return

    # Initial heuristic
    best_colors, best_obj = dsatur_coloring(nodes, adj, rng=None)
    if logger:
        logger.log(best_obj)

    # Short randomized multi-start for improved upper bound
    heuristic_budget = min(2.0, max(0.2, 0.15 * time_limit))
    it = 0
    while time.time() - start < heuristic_budget and time.time() - start < time_limit:
        rng = random.Random(7919 + it * 104729)
        cand_colors, cand_obj = dsatur_coloring(nodes, adj, rng=rng)
        if cand_obj < best_obj:
            best_obj = cand_obj
            best_colors = cand_colors
            if logger:
                logger.log(best_obj)
            if best_obj <= 1:
                break
        it += 1

    # Lower bound from clique heuristic
    clique = greedy_maximal_clique(nodes, adj)
    lb = len(clique)

    # If already tight or no time left, return heuristic
    elapsed = time.time() - start
    remaining = time_limit - elapsed
    if remaining <= 0.5 or best_obj <= lb:
        vertex_colors, color_classes, obj = normalize_solution(best_colors)
        sol = {
            "objective_value": int(obj),
            "vertex_colors": vertex_colors,
            "color_classes": color_classes,
        }
        with open(solution_path, "w") as f:
            json.dump(sol, f, indent=2)
        return

    # Build MIP
    node_to_idx = {v: i for i, v in enumerate(nodes)}
    idx_to_node = {i: v for i, v in enumerate(nodes)}
    edges_idx = [(node_to_idx[u], node_to_idx[v]) for (u, v) in edges]

    UB = best_obj
    K = list(range(UB))

    # Basic sanity check to avoid oversized model under tiny remaining time
    est_constraints = n + n * UB + len(edges_idx) * UB
    if remaining < 1.0 or (est_constraints > 5_000_000 and remaining < 20):
        vertex_colors, color_classes, obj = normalize_solution(best_colors)
        sol = {
            "objective_value": int(obj),
            "vertex_colors": vertex_colors,
            "color_classes": color_classes,
        }
        with open(solution_path, "w") as f:
            json.dump(sol, f, indent=2)
        return

    model = gp.Model("graph_coloring")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = max(0.1, remaining)
    model.Params.MIPFocus = 1

    x = model.addVars(n, UB, vtype=GRB.BINARY, name="x")
    y = model.addVars(UB, vtype=GRB.BINARY, name="y")

    # Every node exactly one color
    model.addConstrs((gp.quicksum(x[i, c] for c in K) == 1 for i in range(n)), name="assign")

    # Link usage
    model.addConstrs((x[i, c] <= y[c] for i in range(n) for c in K), name="link")

    # Adjacent nodes different color
    model.addConstrs((x[i, c] + x[j, c] <= 1 for (i, j) in edges_idx for c in K), name="edge")

    # Symmetry breaking on color usage
    model.addConstrs((y[c] >= y[c + 1] for c in range(UB - 1)), name="sym_y")

    # Fix highest-degree node to color 0
    anchor = max(nodes, key=lambda v: len(adj[v]))
    model.addConstr(x[node_to_idx[anchor], 0] == 1, name="anchor")

    # Optional clique-based symmetry breaking if clique fits UB
    if len(clique) <= UB:
        for t, v in enumerate(clique):
            model.addConstr(x[node_to_idx[v], t] == 1, name=f"clqfix_{t}")

    model.setObjective(gp.quicksum(y[c] for c in K), GRB.MINIMIZE)

    # Warm start from heuristic
    for i, v in idx_to_node.items():
        c0 = best_colors[v]
        for c in K:
            x[i, c].Start = 1.0 if c == c0 else 0.0
    used_best = set(best_colors.values())
    for c in K:
        y[c].Start = 1.0 if c in used_best else 0.0

    best_logged = [float(best_obj)]

    def cb(m, where):
        if where == GRB.Callback.MIPSOL and logger:
            obj = m.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj + 1e-6 < best_logged[0]:
                best_logged[0] = obj
                logger.log(int(round(obj)))

    model.optimize(cb if logger else None)

    # Extract model solution if available
    if model.SolCount > 0:
        mip_colors = {}
        for i in range(n):
            chosen = None
            for c in K:
                if x[i, c].X > 0.5:
                    chosen = c
                    break
            if chosen is None:
                chosen = max(K, key=lambda cc: x[i, cc].X)
            mip_colors[idx_to_node[i]] = int(chosen)

        mip_obj = len(set(mip_colors.values()))
        if mip_obj < best_obj:
            best_obj = mip_obj
            best_colors = mip_colors
            if logger and mip_obj + 1e-6 < best_logged[0]:
                logger.log(int(mip_obj))
        else:
            # Even if equal, mip may have cleaner structure; keep if not worse
            if mip_obj <= best_obj:
                best_colors = mip_colors

    vertex_colors, color_classes, obj = normalize_solution(best_colors)
    sol = {
        "objective_value": int(obj),
        "vertex_colors": vertex_colors,
        "color_classes": color_classes,
    }

    with open(solution_path, "w") as f:
        json.dump(sol, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Graph Coloring Solver")
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    solve(
        instance_path=args.instance_path,
        solution_path=args.solution_path,
        time_limit=args.time_limit,
        log_path=args.log_path,
    )


if __name__ == "__main__":
    main()
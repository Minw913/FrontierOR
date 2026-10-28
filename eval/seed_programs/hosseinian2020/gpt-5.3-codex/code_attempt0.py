import argparse
import json
import random
import time
from typing import List, Dict, Set, Tuple

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def compute_clique_weight(clique_idx: List[int], w_adj: List[Dict[int, float]]) -> float:
    s = set(clique_idx)
    total = 0.0
    for i in clique_idx:
        for j, w in w_adj[i].items():
            if j in s and i < j:
                total += w
    return total


def greedy_weighted_clique(
    n: int,
    adj: List[Set[int]],
    w_adj: List[Dict[int, float]],
    seeds: List[int],
    deadline: float,
) -> Tuple[List[int], float]:
    best_clique = []
    best_obj = 0.0

    for seed in seeds:
        if time.time() >= deadline:
            break

        clique = [seed]
        candidates = set(adj[seed])

        while candidates and time.time() < deadline:
            # pick candidate with largest incremental edge-weight gain to current clique
            best_v = None
            best_gain = -1.0
            for v in candidates:
                gain = 0.0
                wv = w_adj[v]
                for u in clique:
                    gain += wv.get(u, 0.0)
                if gain > best_gain:
                    best_gain = gain
                    best_v = v

            if best_v is None:
                break

            clique.append(best_v)
            candidates.intersection_update(adj[best_v])
            candidates.discard(best_v)

        obj = compute_clique_weight(clique, w_adj)
        if obj > best_obj:
            best_obj = obj
            best_clique = clique[:]

    return best_clique, best_obj


def main():
    parser = argparse.ArgumentParser(description="Maximum edge-weight clique solver")
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    global_start = time.time()
    deadline = global_start + max(0, args.time_limit)

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None
    best_logged_obj = {"value": -float("inf")}

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    vertices = inst["vertices"]
    n = inst["num_vertices"]
    edges = inst["edges"]

    # Map vertex labels to internal indices [0..n-1]
    idx_of = {v: i for i, v in enumerate(vertices)}

    # Build adjacency and weighted adjacency
    adj = [set() for _ in range(n)]
    w_adj = [dict() for _ in range(n)]

    for e in edges:
        u_lbl = e["i"]
        v_lbl = e["j"]
        w = float(e["weight"])
        if u_lbl not in idx_of or v_lbl not in idx_of:
            continue
        u = idx_of[u_lbl]
        v = idx_of[v_lbl]
        if u == v:
            continue
        if v in w_adj[u]:
            # Robustness: aggregate duplicate edges if present
            w_adj[u][v] += w
            w_adj[v][u] += w
        else:
            w_adj[u][v] = w
            w_adj[v][u] = w
            adj[u].add(v)
            adj[v].add(u)

    # Heuristic phase (short)
    weighted_degree = [sum(w_adj[i].values()) for i in range(n)]
    order = sorted(range(n), key=lambda i: weighted_degree[i], reverse=True)

    heuristic_budget = min(5.0, max(0.0, 0.2 * args.time_limit))
    heuristic_deadline = min(deadline, time.time() + heuristic_budget)

    seeds = order[: min(n, 25)]
    if n > 0:
        random.seed(0)
        extra = random.sample(range(n), k=min(n, 15))
        seeds.extend(extra)

    # remove duplicates while preserving order
    seen = set()
    uniq_seeds = []
    for s in seeds:
        if s not in seen:
            seen.add(s)
            uniq_seeds.append(s)

    heuristic_clique, heuristic_obj = greedy_weighted_clique(
        n, adj, w_adj, uniq_seeds, heuristic_deadline
    )

    if heuristic_obj > best_logged_obj["value"] + 1e-9:
        best_logged_obj["value"] = heuristic_obj
        if logger:
            logger.log(float(heuristic_obj))

    # If no time left, return heuristic
    if time.time() >= deadline or args.time_limit <= 0:
        clique_labels = sorted(vertices[i] for i in heuristic_clique)
        out = {"objective_value": float(heuristic_obj), "clique": clique_labels}
        with open(args.solution_path, "w") as f:
            json.dump(out, f)
        return

    # Build MIP/MIQP model
    model = gp.Model("max_weight_clique")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    model.Params.NonConvex = 2

    remaining = max(0.0, deadline - time.time())
    model.Params.TimeLimit = remaining

    x = model.addVars(n, vtype=GRB.BINARY, name="x")

    # Non-edge constraints: vertices without an edge cannot both be selected
    model.addConstrs(
        (x[i] + x[j] <= 1 for i in range(n) for j in range(i + 1, n) if j not in adj[i]),
        name="nonedge",
    )

    # Quadratic objective: sum of weights over selected edges
    obj = gp.QuadExpr()
    for i in range(n):
        for j, w in w_adj[i].items():
            if i < j:
                obj += w * x[i] * x[j]
    model.setObjective(obj, GRB.MAXIMIZE)

    # Provide heuristic start
    start_set = set(heuristic_clique)
    for i in range(n):
        x[i].Start = 1.0 if i in start_set else 0.0

    def mip_callback(m, where):
        if where == GRB.Callback.MIPSOL:
            val = m.cbGet(GRB.Callback.MIPSOL_OBJ)
            if val > best_logged_obj["value"] + 1e-9:
                best_logged_obj["value"] = val
                if logger:
                    logger.log(float(val))

    model.optimize(mip_callback)

    # Extract best found solution
    best_clique_idx = heuristic_clique[:]
    best_obj = heuristic_obj

    if model.SolCount > 0:
        xv = model.getAttr("X", x)
        clique_idx = [i for i in range(n) if xv[i] > 0.5]
        obj_val = compute_clique_weight(clique_idx, w_adj)
        if obj_val >= best_obj - 1e-9:
            best_obj = obj_val
            best_clique_idx = clique_idx

    clique_labels = sorted(vertices[i] for i in best_clique_idx)
    result = {
        "objective_value": float(best_obj),
        "clique": clique_labels,
    }

    with open(args.solution_path, "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
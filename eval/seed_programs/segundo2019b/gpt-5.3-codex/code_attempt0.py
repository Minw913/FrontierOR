import argparse
import json
import time
import random
from typing import List, Set

import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def parse_args():
    parser = argparse.ArgumentParser(description="Maximum Weighted Clique Solver")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to input instance JSON")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to output solution JSON")
    parser.add_argument("--time_limit", type=int, required=True, help="Time limit in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Optional JSONL log path")
    return parser.parse_args()


def is_clique(vertices: List[int], adj: List[Set[int]]) -> bool:
    k = len(vertices)
    for i in range(k):
        u = vertices[i]
        nu = adj[u]
        for j in range(i + 1, k):
            if vertices[j] not in nu:
                return False
    return True


def greedy_repair(vertices: List[int], weights: List[int], adj: List[Set[int]]) -> List[int]:
    # Keep a high-weight clique subset if input is not a clique
    order = sorted(vertices, key=lambda v: weights[v], reverse=True)
    clique = []
    for v in order:
        ok = True
        for u in clique:
            if v not in adj[u]:
                ok = False
                break
        if ok:
            clique.append(v)
    return clique


def greedy_construct(order: List[int], adj: List[Set[int]]) -> List[int]:
    clique = []
    for v in order:
        ok = True
        for u in clique:
            if v not in adj[u]:
                ok = False
                break
        if ok:
            clique.append(v)
    return clique


def main():
    args = parse_args()
    start_time = time.time()

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    graph = data["graph"]
    n = graph["num_vertices"]
    vertices_ids = graph["vertices"]
    edges = graph["edges"]
    vertex_weights_raw = data["vertex_weights"]

    # Map vertex IDs to internal indices [0..n-1]
    id_to_idx = {vid: i for i, vid in enumerate(vertices_ids)}

    # Robust weight alignment
    if (
        len(vertex_weights_raw) == n
        and all(isinstance(v, int) for v in vertices_ids)
        and set(vertices_ids) == set(range(n))
    ):
        weights = [vertex_weights_raw[vid] for vid in vertices_ids]
    else:
        weights = list(vertex_weights_raw)

    # Build adjacency sets
    adj = [set() for _ in range(n)]
    for e in edges:
        if len(e) != 2:
            continue
        u_id, v_id = e
        if u_id not in id_to_idx or v_id not in id_to_idx:
            continue
        u = id_to_idx[u_id]
        v = id_to_idx[v_id]
        if u == v:
            continue
        adj[u].add(v)
        adj[v].add(u)

    # Best solution state
    best_obj = 0.0
    best_set: List[int] = []

    def try_update(sol: List[int]):
        nonlocal best_obj, best_set
        if not sol:
            return
        if not is_clique(sol, adj):
            sol = greedy_repair(sol, weights, adj)
        obj = float(sum(weights[v] for v in sol))
        if obj > best_obj + 1e-9:
            best_obj = obj
            best_set = list(sol)
            if logger:
                logger.log(best_obj)

    # ---- Fast heuristic phase ----
    elapsed = time.time() - start_time
    remaining_total = max(0.0, args.time_limit - elapsed)
    heuristic_budget = min(2.0, 0.15 * args.time_limit, remaining_total * 0.5)

    h_start = time.time()
    degree = [len(adj[i]) for i in range(n)]

    # Deterministic orders
    order1 = sorted(range(n), key=lambda i: (weights[i], degree[i]), reverse=True)
    try_update(greedy_construct(order1, adj))

    order2 = sorted(range(n), key=lambda i: (weights[i] * (degree[i] + 1)), reverse=True)
    try_update(greedy_construct(order2, adj))

    order3 = sorted(range(n), key=lambda i: (weights[i], - (n - 1 - degree[i])), reverse=True)
    try_update(greedy_construct(order3, adj))

    # Randomized multi-start
    rng = random.Random(0)
    it = 0
    while time.time() - h_start < heuristic_budget:
        it += 1
        # Mild random perturbation of score
        scored = []
        for i in range(n):
            score = weights[i] + 0.20 * degree[i] + rng.random() * (1.0 + (it % 5))
            scored.append((score, i))
        scored.sort(reverse=True)
        order = [i for _, i in scored]
        try_update(greedy_construct(order, adj))

    # ---- Exact phase with Gurobi + lazy constraints ----
    elapsed = time.time() - start_time
    remaining = args.time_limit - elapsed

    if remaining > 0.01:
        model = gp.Model("max_weight_clique")
        model.Params.OutputFlag = 0
        model.Params.Threads = 1
        model.Params.TimeLimit = remaining
        model.Params.LazyConstraints = 1
        model.Params.MIPFocus = 1

        x = model.addVars(n, vtype=GRB.BINARY, name="x")
        model.setObjective(gp.quicksum(weights[i] * x[i] for i in range(n)), GRB.MAXIMIZE)

        # Warm start from heuristic
        if best_set:
            in_best = set(best_set)
            for i in range(n):
                x[i].Start = 1.0 if i in in_best else 0.0

        state = {"best_obj": best_obj, "best_set": list(best_set)}

        def callback(m, where):
            if where == GRB.Callback.MIPSOL:
                vals = m.cbGetSolution(x)
                selected = [i for i in range(n) if vals[i] > 0.5]

                # Separate violated pair constraints x_u + x_v <= 1 for non-edges
                violations_added = 0
                max_cuts = 2000
                k = len(selected)
                for a in range(k):
                    u = selected[a]
                    nu = adj[u]
                    for b in range(a + 1, k):
                        v = selected[b]
                        if v not in nu:
                            m.cbLazy(x[u] + x[v] <= 1)
                            violations_added += 1
                            if violations_added >= max_cuts:
                                break
                    if violations_added >= max_cuts:
                        break

                # If no violation, this is a valid clique incumbent
                if violations_added == 0:
                    obj = float(m.cbGet(GRB.Callback.MIPSOL_OBJ))
                    if obj > state["best_obj"] + 1e-9:
                        state["best_obj"] = obj
                        state["best_set"] = selected
                        if logger:
                            logger.log(obj)

        model.optimize(callback)

        # Pull best from callback state
        if state["best_obj"] > best_obj + 1e-9:
            best_obj = state["best_obj"]
            best_set = state["best_set"]

        # Also check final solver solution if present
        if model.SolCount > 0:
            sol = [i for i in range(n) if x[i].X > 0.5]
            if not is_clique(sol, adj):
                sol = greedy_repair(sol, weights, adj)
            obj = float(sum(weights[i] for i in sol))
            if obj > best_obj + 1e-9:
                best_obj = obj
                best_set = sol
                if logger:
                    logger.log(best_obj)

    # Ensure feasibility (empty set always feasible; with positive weights we usually have non-empty)
    if best_set and not is_clique(best_set, adj):
        best_set = greedy_repair(best_set, weights, adj)
        best_obj = float(sum(weights[i] for i in best_set))

    clique_vertices = [vertices_ids[i] for i in best_set]
    clique_vertices.sort()

    solution = {
        "objective_value": float(best_obj),
        "clique_vertices": clique_vertices,
    }

    with open(args.solution_path, "w") as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()
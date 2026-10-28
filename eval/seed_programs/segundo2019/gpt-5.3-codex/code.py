import argparse
import json
import time
import random
from typing import Dict, List, Set, Tuple

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def parse_args():
    parser = argparse.ArgumentParser(description="Solve Maximum Edge-Weight Clique Problem")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to input instance JSON")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to output solution JSON")
    parser.add_argument("--time_limit", type=int, required=True, help="Time limit in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Optional JSONL path for convergence logging")
    return parser.parse_args()


def build_graph(instance: dict):
    g = instance["graph"]
    vertices = g["vertices"]
    n = g["num_vertices"]

    id_to_idx = {vid: i for i, vid in enumerate(vertices)}
    idx_to_id = list(vertices)

    # Normalize edges, guard against accidental duplicates by keeping max weight
    edge_w: Dict[Tuple[int, int], float] = {}
    for e in g["edges"]:
        u_id, v_id, w = e
        if u_id not in id_to_idx or v_id not in id_to_idx:
            continue
        i, j = id_to_idx[u_id], id_to_idx[v_id]
        if i == j:
            continue
        if i > j:
            i, j = j, i
        w = float(w)
        if (i, j) in edge_w:
            edge_w[(i, j)] = max(edge_w[(i, j)], w)
        else:
            edge_w[(i, j)] = w

    edge_tuples = [(i, j, w) for (i, j), w in edge_w.items()]

    adj: List[Set[int]] = [set() for _ in range(n)]
    weighted_degree = [0.0] * n
    for i, j, w in edge_tuples:
        adj[i].add(j)
        adj[j].add(i)
        weighted_degree[i] += w
        weighted_degree[j] += w

    return n, idx_to_id, edge_w, edge_tuples, adj, weighted_degree


def greedy_clique_from_seed(seed: int, adj: List[Set[int]], edge_w: Dict[Tuple[int, int], float]):
    clique = [seed]
    clique_set = {seed}
    candidates = set(adj[seed])
    obj = 0.0

    while candidates:
        best_v = None
        best_gain = -1.0
        for v in candidates:
            gain = 0.0
            for u in clique:
                a, b = (u, v) if u < v else (v, u)
                gain += edge_w[(a, b)]
            if gain > best_gain:
                best_gain = gain
                best_v = v

        if best_v is None:
            break

        clique.append(best_v)
        clique_set.add(best_v)
        obj += best_gain
        candidates.intersection_update(adj[best_v])

    return clique_set, obj


def heuristic_clique(
    n: int,
    adj: List[Set[int]],
    edge_w: Dict[Tuple[int, int], float],
    weighted_degree: List[float],
    time_budget: float,
):
    if n == 0:
        return set(), 0.0

    start = time.perf_counter()
    rng = random.Random(0)

    ordered = sorted(range(n), key=lambda i: weighted_degree[i], reverse=True)
    seeds = ordered[: min(n, 60)]

    remaining_vertices = ordered[min(n, 60):]
    if remaining_vertices:
        extra = rng.sample(remaining_vertices, min(len(remaining_vertices), 40))
        seeds.extend(extra)

    # Ensure at least one try
    if not seeds:
        seeds = [0]

    best_clique = set()
    best_obj = 0.0

    for s in seeds:
        if time.perf_counter() - start > time_budget:
            break
        c, obj = greedy_clique_from_seed(s, adj, edge_w)
        if obj > best_obj + 1e-9:
            best_clique, best_obj = c, obj

    # Fallback: single vertex if nothing else
    if not best_clique and n > 0:
        best_clique = {ordered[0] if ordered else 0}
        best_obj = 0.0

    return best_clique, best_obj


def repair_to_clique(selected: Set[int], adj: List[Set[int]], edge_w: Dict[Tuple[int, int], float]) -> Set[int]:
    sel = set(selected)
    if len(sel) <= 1:
        return sel

    def contrib(v, current):
        total = 0.0
        for u in current:
            if u == v:
                continue
            a, b = (u, v) if u < v else (v, u)
            w = edge_w.get((a, b))
            if w is not None:
                total += w
        return total

    while True:
        lst = sorted(sel)
        violation = None
        for i in range(len(lst)):
            u = lst[i]
            au = adj[u]
            for j in range(i + 1, len(lst)):
                v = lst[j]
                if v not in au:
                    violation = (u, v)
                    break
            if violation:
                break

        if not violation:
            break

        u, v = violation
        cu = contrib(u, sel)
        cv = contrib(v, sel)
        if cu <= cv:
            sel.remove(u)
        else:
            sel.remove(v)

        if len(sel) <= 1:
            break

    return sel


def extract_solution(
    selected_idx: Set[int],
    idx_to_id: List[int],
    edge_tuples: List[Tuple[int, int, float]],
):
    selected_set = set(selected_idx)
    clique_vertices = sorted(idx_to_id[i] for i in selected_set)

    clique_edges = []
    obj = 0.0
    for i, j, w in edge_tuples:
        if i in selected_set and j in selected_set:
            u_id, v_id = idx_to_id[i], idx_to_id[j]
            if u_id < v_id:
                clique_edges.append([u_id, v_id])
            else:
                clique_edges.append([v_id, u_id])
            obj += w

    clique_edges.sort()
    return float(obj), clique_vertices, clique_edges


def main():
    args = parse_args()
    t0 = time.perf_counter()

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    n, idx_to_id, edge_w, edge_tuples, adj, weighted_degree = build_graph(instance)

    # Trivial empty graph case
    if n == 0:
        out = {
            "objective_value": 0.0,
            "clique_vertices": [],
            "clique_edges": []
        }
        with open(args.solution_path, "w") as f:
            json.dump(out, f)
        return

    # Heuristic warm start
    total_limit = max(0, int(args.time_limit))
    heuristic_budget = min(5.0, max(0.2, 0.15 * total_limit)) if total_limit > 0 else 0.0
    best_heur_clique, best_heur_obj = heuristic_clique(
        n, adj, edge_w, weighted_degree, heuristic_budget
    )

    best_logged_obj = float("-inf")
    if logger:
        logger.log(float(best_heur_obj))
        best_logged_obj = float(best_heur_obj)

    elapsed = time.perf_counter() - t0
    remaining = total_limit - elapsed

    final_selected = set(best_heur_clique)

    # If no time remains, return heuristic solution
    if remaining > 0.05:
        model = gp.Model("mewcp")
        model.Params.OutputFlag = 0
        model.Params.Threads = 1
        model.Params.TimeLimit = max(0.01, remaining)
        model.Params.MIPFocus = 1

        x = model.addVars(n, vtype=GRB.BINARY, name="x")
        y = model.addVars(len(edge_tuples), lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="y")

        model.setObjective(
            gp.quicksum(edge_tuples[eid][2] * y[eid] for eid in range(len(edge_tuples))),
            GRB.MAXIMIZE
        )

        for eid, (i, j, _) in enumerate(edge_tuples):
            model.addConstr(y[eid] <= x[i])
            model.addConstr(y[eid] <= x[j])
            model.addConstr(y[eid] >= x[i] + x[j] - 1)

        # Decide whether to add all non-edge constraints explicitly or lazily
        total_pairs = n * (n - 1) // 2
        non_edges = total_pairs - len(edge_tuples)
        use_lazy = non_edges > 300000

        if not use_lazy:
            for i in range(n):
                ai = adj[i]
                for j in range(i + 1, n):
                    if j not in ai:
                        model.addConstr(x[i] + x[j] <= 1)
        else:
            model.Params.LazyConstraints = 1

        # Warm start
        for i in range(n):
            x[i].Start = 1.0 if i in best_heur_clique else 0.0

        model._x = x
        model._x_list = [x[i] for i in range(n)]
        model._adj = adj
        model._lazy_mode = use_lazy
        model._logger = logger
        model._best_logged = best_logged_obj

        def callback(m, where):
            if where == GRB.Callback.MIPSOL:
                if m._lazy_mode:
                    vals = m.cbGetSolution(m._x_list)
                    selected = [i for i, v in enumerate(vals) if v > 0.5]
                    violations = 0
                    max_cuts = 200
                    for a in range(len(selected)):
                        u = selected[a]
                        au = m._adj[u]
                        for b in range(a + 1, len(selected)):
                            v = selected[b]
                            if v not in au:
                                m.cbLazy(m._x[u] + m._x[v] <= 1)
                                violations += 1
                                if violations >= max_cuts:
                                    break
                        if violations >= max_cuts:
                            break
                    if violations > 0:
                        return

                if m._logger is not None:
                    obj = m.cbGet(GRB.Callback.MIPSOL_OBJ)
                    if obj > m._best_logged + 1e-9:
                        m._best_logged = obj
                        m._logger.log(float(obj))

        model.optimize(callback)

        if model.SolCount > 0:
            selected = {i for i in range(n) if x[i].X > 0.5}
            final_selected = selected

    # Safety repair to clique feasibility
    final_selected = repair_to_clique(final_selected, adj, edge_w)

    objective_value, clique_vertices, clique_edges = extract_solution(final_selected, idx_to_id, edge_tuples)

    out = {
        "objective_value": float(objective_value),
        "clique_vertices": clique_vertices,
        "clique_edges": clique_edges
    }

    with open(args.solution_path, "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()
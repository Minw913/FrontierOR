import argparse
import json
import os
import random
import time
from itertools import combinations

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


EPS = 1e-9


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    start_time = time.monotonic()
    deadline = start_time + max(0, args.time_limit)

    with open(args.instance_path, "r", encoding="utf-8") as f:
        instance = json.load(f)

    graph = instance["graph"]
    vertex_ids = [int(v) for v in graph["vertices"]]
    n = len(vertex_ids)
    id_to_index = {v: i for i, v in enumerate(vertex_ids)}

    # Canonicalize the simple undirected edge set.
    edge_weight = {}
    for raw_edge in graph["edges"]:
        u_id, v_id, weight = raw_edge
        u_id = int(u_id)
        v_id = int(v_id)
        if u_id == v_id:
            continue
        if u_id not in id_to_index or v_id not in id_to_index:
            continue
        u = id_to_index[u_id]
        v = id_to_index[v_id]
        if u > v:
            u, v = v, u
        edge_weight[(u, v)] = float(weight)

    adjacency = [set() for _ in range(n)]
    adjacency_weight = [dict() for _ in range(n)]
    weighted_degree = [0.0] * n

    for (u, v), w in edge_weight.items():
        adjacency[u].add(v)
        adjacency[v].add(u)
        adjacency_weight[u][v] = w
        adjacency_weight[v][u] = w
        weighted_degree[u] += w
        weighted_degree[v] += w

    degree = [len(adjacency[v]) for v in range(n)]

    def make_solution(vertices):
        selected = set(vertices)
        sorted_ids = sorted(vertex_ids[i] for i in selected)
        selected_indices = [id_to_index[v_id] for v_id in sorted_ids]

        clique_edges = []
        objective = 0.0
        for p in range(len(selected_indices)):
            i = selected_indices[p]
            for q in range(p + 1, len(selected_indices)):
                j = selected_indices[q]
                key = (i, j) if i < j else (j, i)
                w = edge_weight.get(key)
                if w is None:
                    return None
                objective += w
                a, b = vertex_ids[i], vertex_ids[j]
                if a > b:
                    a, b = b, a
                clique_edges.append([a, b])

        clique_edges.sort()
        return {
            "objective_value": float(objective),
            "clique_vertices": sorted_ids,
            "clique_edges": clique_edges,
        }

    best_solution = make_solution([])
    best_value = 0.0
    best_indices = set()

    if logger:
        logger.log_solution(best_value, best_solution)

    def consider(vertices):
        nonlocal best_solution, best_value, best_indices
        solution = make_solution(vertices)
        if solution is None:
            return False

        value = float(solution["objective_value"])
        tolerance = EPS * max(1.0, abs(best_value))
        if value > best_value + tolerance:
            best_value = value
            best_indices = set(vertices)
            best_solution = solution
            if logger:
                logger.log_solution(best_value, best_solution)
            return True
        return False

    def greedy_scan(order):
        clique = []
        for v in order:
            if all(v in adjacency[u] for u in clique):
                clique.append(v)
        return clique

    def greedy_from_seed(seed):
        clique = [seed]
        candidates = set(adjacency[seed])

        while candidates and time.monotonic() < heuristic_deadline:
            best_candidate = None
            best_key = None

            for v in candidates:
                gain = 0.0
                for u in clique:
                    gain += adjacency_weight[v][u]
                key = (gain, degree[v], weighted_degree[v], -v)
                if best_key is None or key > best_key:
                    best_key = key
                    best_candidate = v

            clique.append(best_candidate)
            candidates.intersection_update(adjacency[best_candidate])

        return clique

    # Spend a small, bounded fraction of the runtime obtaining a strong MIP start.
    total_limit = max(0.0, float(args.time_limit))
    heuristic_budget = min(4.0, max(0.02, 0.08 * total_limit))
    heuristic_deadline = min(deadline, time.monotonic() + heuristic_budget)

    nonisolated = [v for v in range(n) if degree[v] > 0]

    if nonisolated and time.monotonic() < heuristic_deadline:
        orders = [
            sorted(nonisolated, key=lambda v: (degree[v], weighted_degree[v]), reverse=True),
            sorted(nonisolated, key=lambda v: (weighted_degree[v], degree[v]), reverse=True),
        ]

        rng = random.Random(0)
        for _ in range(6):
            orders.append(
                sorted(
                    nonisolated,
                    key=lambda v: (
                        weighted_degree[v] * (0.75 + 0.5 * rng.random()),
                        degree[v],
                    ),
                    reverse=True,
                )
            )

        for order in orders:
            if time.monotonic() >= heuristic_deadline:
                break
            consider(greedy_scan(order))

        seed_order = sorted(
            nonisolated,
            key=lambda v: (weighted_degree[v], degree[v]),
            reverse=True,
        )
        seed_count = min(len(seed_order), max(20, min(120, int(5000 / max(1, n)) + 30)))

        for seed in seed_order[:seed_count]:
            if time.monotonic() >= heuristic_deadline:
                break
            consider(greedy_from_seed(seed))

    # Simple improving one-vertex exchanges followed by greedy augmentation.
    if best_indices and time.monotonic() < heuristic_deadline:
        current = set(best_indices)

        for _ in range(12):
            if time.monotonic() >= heuristic_deadline:
                break

            # Add any currently compatible vertices.
            while time.monotonic() < heuristic_deadline:
                compatible = [
                    v for v in nonisolated
                    if v not in current and all(v in adjacency[u] for u in current)
                ]
                if not compatible:
                    break

                add_v = max(
                    compatible,
                    key=lambda v: (
                        sum(adjacency_weight[v][u] for u in current),
                        degree[v],
                        weighted_degree[v],
                    ),
                )
                current.add(add_v)
                consider(current)

            best_exchange = None
            best_delta = 0.0

            for v in nonisolated:
                if v in current:
                    continue

                blockers = [u for u in current if v not in adjacency[u]]
                if len(blockers) != 1:
                    continue

                removed = blockers[0]
                rest = current - {removed}
                added_weight = sum(adjacency_weight[v][u] for u in rest)
                removed_weight = sum(adjacency_weight[removed][u] for u in rest)
                delta = added_weight - removed_weight

                if delta > best_delta + EPS:
                    best_delta = delta
                    best_exchange = (removed, v)

            if best_exchange is None:
                break

            removed, added = best_exchange
            current.remove(removed)
            current.add(added)
            consider(current)

    # Find connected components. A positive-weight clique is contained in one
    # nontrivial connected component.
    components = []
    unseen = set(nonisolated)

    while unseen:
        root = unseen.pop()
        component = [root]
        stack = [root]

        while stack:
            u = stack.pop()
            for v in adjacency[u]:
                if v in unseen:
                    unseen.remove(v)
                    component.append(v)
                    stack.append(v)

        components.append(component)

    component_of = {}
    component_upper_bound = []
    for c, component in enumerate(components):
        for v in component:
            component_of[v] = c
        comp_set = set(component)
        upper = sum(
            w for (u, v), w in edge_weight.items()
            if u in comp_set and v in comp_set
        )
        component_upper_bound.append(upper)

    # Components whose sum of all edge weights cannot beat the incumbent may
    # safely be omitted.
    eligible_components = []
    for c, component in enumerate(components):
        tolerance = EPS * max(1.0, abs(best_value))
        if component_upper_bound[c] > best_value + tolerance:
            eligible_components.append(c)

    remaining = deadline - time.monotonic()

    if eligible_components and remaining > 0.02:
        active_vertices = []
        active_set = set()
        for c in eligible_components:
            active_vertices.extend(components[c])
            active_set.update(components[c])

        active_edges = [
            (u, v, w)
            for (u, v), w in edge_weight.items()
            if u in active_set and v in active_set
        ]

        try:
            model = gp.Model("maximum_edge_weight_clique")
            model.Params.OutputFlag = 0
            model.Params.Seed = 0
            model.Params.MIPGap = 1e-4
            model.Params.NumericFocus = 0
            model.Params.Threads = 1
            model.Params.TimeLimit = max(0.01, deadline - time.monotonic())

            x = {
                v: model.addVar(vtype=GRB.BINARY, name=f"x_{v}")
                for v in active_vertices
            }
            y = [
                model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"y_{e}")
                for e in range(len(active_edges))
            ]

            model.update()

            # Link each earned edge weight to both endpoint selections.
            for e, (u, v, _) in enumerate(active_edges):
                model.addLConstr(y[e] <= x[u])
                model.addLConstr(y[e] <= x[v])

            # Pairwise non-edge constraints enforce the clique property.
            for c in eligible_components:
                component = components[c]
                for p in range(len(component)):
                    u = component[p]
                    neighbors_u = adjacency[u]
                    for q in range(p + 1, len(component)):
                        v = component[q]
                        if v not in neighbors_u:
                            model.addLConstr(x[u] + x[v] <= 1.0)

            # Prevent the model from independently selecting a clique in each
            # disconnected component.
            if len(eligible_components) > 1:
                z = {
                    c: model.addVar(vtype=GRB.BINARY, name=f"component_{c}")
                    for c in eligible_components
                }
                model.update()
                model.addLConstr(gp.quicksum(z[c] for c in eligible_components) <= 1.0)
                for c in eligible_components:
                    for v in components[c]:
                        model.addLConstr(x[v] <= z[c])

            model.setObjective(
                gp.quicksum(w * y[e] for e, (_, _, w) in enumerate(active_edges)),
                GRB.MAXIMIZE,
            )

            # Supply the heuristic incumbent as a MIP start when its component
            # remains in the model.
            for v in active_vertices:
                x[v].Start = 1.0 if v in best_indices else 0.0
            for e, (u, v, _) in enumerate(active_edges):
                y[e].Start = 1.0 if u in best_indices and v in best_indices else 0.0

            model.update()

            def incumbent_callback(cb_model, where):
                if where != GRB.Callback.MIPSOL:
                    return
                selected = {
                    v for v in active_vertices
                    if cb_model.cbGetSolution(x[v]) > 0.5
                }
                consider(selected)

            model.optimize(incumbent_callback)

            if model.SolCount > 0:
                selected = {
                    v for v in active_vertices
                    if x[v].X > 0.5
                }
                consider(selected)

        except gp.GurobiError:
            # The heuristic incumbent remains a valid output if optimization
            # cannot be started or is interrupted by the environment.
            pass

    output_directory = os.path.dirname(os.path.abspath(args.solution_path))
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)

    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(best_solution, f, separators=(",", ":"))


if __name__ == "__main__":
    main()
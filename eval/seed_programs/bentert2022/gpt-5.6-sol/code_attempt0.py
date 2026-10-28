import argparse
import heapq
import json
import math
import os
import time

from solution_logger import SolutionLogger


EPS = 1e-9


class DSU:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        a = self.find(a)
        b = self.find(b)
        if a == b:
            return False
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1
        return True


def compute_solution(n, ids, weights, edges):
    vertex_costs = [0.0] * n
    selected_edges = []

    for i, j in edges:
        w = weights[i][j]
        if w > vertex_costs[i]:
            vertex_costs[i] = w
        if w > vertex_costs[j]:
            vertex_costs[j] = w
        selected_edges.append([ids[i], ids[j]])

    objective = float(sum(vertex_costs))
    return {
        "objective_value": objective,
        "selected_edges": selected_edges,
        "vertex_costs": {
            str(ids[i]): float(vertex_costs[i]) for i in range(n)
        },
    }


def is_spanning_tree(n, edges):
    if n == 1:
        return len(edges) == 0
    if len(edges) != n - 1:
        return False

    dsu = DSU(n)
    for i, j in edges:
        if not dsu.union(i, j):
            return False

    root = dsu.find(0)
    return all(dsu.find(i) == root for i in range(n))


def prim_mst(weights):
    n = len(weights)
    if n <= 1:
        return []

    in_tree = [False] * n
    key = [math.inf] * n
    parent = [-1] * n
    key[0] = 0.0
    result = []

    for _ in range(n):
        u = -1
        best = math.inf
        for i in range(n):
            if not in_tree[i] and key[i] < best:
                best = key[i]
                u = i

        if u < 0:
            break

        in_tree[u] = True
        if parent[u] >= 0:
            result.append((min(u, parent[u]), max(u, parent[u])))

        row = weights[u]
        for v in range(n):
            if not in_tree[v] and row[v] < key[v]:
                key[v] = row[v]
                parent[v] = u

    return result


def incremental_power_tree(weights, lower_bounds, root, deadline):
    n = len(weights)
    if n <= 1:
        return []

    connected = [False] * n
    connected[root] = True
    powers = list(lower_bounds)
    edges = []

    for step in range(n - 1):
        best_delta = math.inf
        best_weight = math.inf
        best_edge = None

        for u in range(n):
            if not connected[u]:
                continue
            row = weights[u]
            pu = powers[u]
            for v in range(n):
                if connected[v]:
                    continue
                w = row[v]
                delta = max(0.0, w - pu) + max(0.0, w - powers[v])
                if (delta < best_delta - EPS or
                        (abs(delta - best_delta) <= EPS and w < best_weight)):
                    best_delta = delta
                    best_weight = w
                    best_edge = (u, v)

            if (u & 15) == 0 and time.monotonic() >= deadline:
                return None

        if best_edge is None:
            return None

        u, v = best_edge
        connected[v] = True
        powers[u] = max(powers[u], weights[u][v])
        powers[v] = max(powers[v], weights[u][v])
        edges.append((min(u, v), max(u, v)))

    return edges


def leaf_relocation_search(n, ids, weights, initial_edges, deadline, on_candidate):
    if n <= 2:
        return initial_edges

    edges = set((min(i, j), max(i, j)) for i, j in initial_edges)
    max_moves = min(200, 2 * n)

    for _ in range(max_moves):
        if time.monotonic() >= deadline:
            break

        adjacency = [set() for _ in range(n)]
        for i, j in edges:
            adjacency[i].add(j)
            adjacency[j].add(i)

        costs = [0.0] * n
        for i in range(n):
            if adjacency[i]:
                costs[i] = max(weights[i][j] for j in adjacency[i])

        best_delta = -EPS
        best_move = None

        leaves = [i for i in range(n) if len(adjacency[i]) == 1]
        for leaf_index, leaf in enumerate(leaves):
            parent = next(iter(adjacency[leaf]))

            remaining_parent_cost = 0.0
            for q in adjacency[parent]:
                if q != leaf:
                    remaining_parent_cost = max(
                        remaining_parent_cost, weights[parent][q]
                    )

            old_affected_base = costs[leaf] + costs[parent]

            for v in range(n):
                if v == leaf or v == parent:
                    continue

                w = weights[leaf][v]
                new_leaf_cost = w
                new_v_cost = max(costs[v], w)

                old_affected = old_affected_base + costs[v]
                new_affected = (
                    new_leaf_cost + remaining_parent_cost + new_v_cost
                )
                delta = new_affected - old_affected

                if delta < best_delta:
                    best_delta = delta
                    best_move = (leaf, parent, v)

            if (leaf_index & 15) == 0 and time.monotonic() >= deadline:
                break

        if best_move is None:
            break

        leaf, parent, v = best_move
        edges.remove((min(leaf, parent), max(leaf, parent)))
        edges.add((min(leaf, v), max(leaf, v)))

        candidate = sorted(edges)
        if not is_spanning_tree(n, candidate):
            # Defensive rollback; this should never occur for a leaf move.
            edges.remove((min(leaf, v), max(leaf, v)))
            edges.add((min(leaf, parent), max(leaf, parent)))
            break

        on_candidate(candidate)

    return sorted(edges)


def make_candidate_edges(weights, incumbent_edges, exact_threshold=300, k=30):
    n = len(weights)
    if n <= exact_threshold:
        return [(i, j) for i in range(n) for j in range(i + 1, n)]

    candidates = set((min(i, j), max(i, j)) for i, j in incumbent_edges)
    nearest_count = min(k, n - 1)

    for i in range(n):
        nearest = heapq.nsmallest(
            nearest_count,
            (j for j in range(n) if j != i),
            key=lambda j: weights[i][j],
        )
        for j in nearest:
            candidates.add((min(i, j), max(i, j)))

    return sorted(candidates)


def solve_with_gurobi(
    n,
    ids,
    weights,
    lower_bounds,
    candidate_edges,
    incumbent_edges,
    deadline,
    update_incumbent,
):
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception:
        return

    remaining = deadline - time.monotonic()
    if remaining <= 0.25:
        return

    try:
        model = gp.Model("minimum_power_connectivity")
        model.Params.OutputFlag = 0
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1
        model.Params.TimeLimit = max(0.01, remaining)
        model.Params.LazyConstraints = 1
        model.Params.PreCrush = 1

        x = {}
        for i, j in candidate_edges:
            x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

        max_weight = max(
            (weights[i][j] for i, j in candidate_edges), default=0.0
        )
        p = [
            model.addVar(
                lb=lower_bounds[i],
                ub=max_weight,
                vtype=GRB.CONTINUOUS,
                name=f"p_{i}",
            )
            for i in range(n)
        ]

        model.setObjective(gp.quicksum(p), GRB.MINIMIZE)

        if n > 1:
            model.addConstr(
                gp.quicksum(x[e] for e in candidate_edges) == n - 1,
                name="tree_edge_count",
            )

        incident = [[] for _ in range(n)]
        for i, j in candidate_edges:
            var = x[i, j]
            incident[i].append(var)
            incident[j].append(var)
            w = weights[i][j]

            # Stronger than p_i >= w*x when p_i already has a positive lower bound.
            model.addConstr(
                p[i] >= lower_bounds[i] + (w - lower_bounds[i]) * var
            )
            model.addConstr(
                p[j] >= lower_bounds[j] + (w - lower_bounds[j]) * var
            )

        if n > 1:
            for i in range(n):
                model.addConstr(gp.quicksum(incident[i]) >= 1)

        incumbent_set = set(
            (min(i, j), max(i, j)) for i, j in incumbent_edges
        )
        incumbent_solution = compute_solution(
            n, ids, weights, incumbent_edges
        )
        incumbent_costs = [
            incumbent_solution["vertex_costs"][str(ids[i])] for i in range(n)
        ]

        for edge, var in x.items():
            var.Start = 1.0 if edge in incumbent_set else 0.0
        for i in range(n):
            p[i].Start = incumbent_costs[i]

        model._x = x
        model._edge_list = candidate_edges
        model._n = n
        model._weights = weights
        model._ids = ids
        model._update_incumbent = update_incumbent

        def callback(cb_model, where):
            if where != GRB.Callback.MIPSOL:
                return

            try:
                values = cb_model.cbGetSolution(cb_model._x)
                chosen = [
                    edge for edge in cb_model._edge_list
                    if values[edge] > 0.5
                ]

                dsu = DSU(cb_model._n)
                for i, j in chosen:
                    dsu.union(i, j)

                groups = {}
                for i in range(cb_model._n):
                    root = dsu.find(i)
                    groups.setdefault(root, []).append(i)

                if len(groups) > 1:
                    component_of = [dsu.find(i) for i in range(cb_model._n)]
                    crossing_by_component = {
                        root: [] for root in groups
                    }

                    for i, j in cb_model._edge_list:
                        ri = component_of[i]
                        rj = component_of[j]
                        if ri != rj:
                            crossing_by_component[ri].append(cb_model._x[i, j])
                            crossing_by_component[rj].append(cb_model._x[i, j])

                    # One side of every proper component must have a selected
                    # crossing edge.
                    for root, terms in crossing_by_component.items():
                        if terms:
                            cb_model.cbLazy(gp.quicksum(terms) >= 1)
                    return

                if len(chosen) == cb_model._n - 1:
                    cb_model._update_incumbent(chosen)
            except Exception:
                # A logging or callback issue should not discard the best
                # feasible solution already available.
                return

        model.optimize(callback)

        if model.SolCount > 0:
            chosen = [
                edge for edge, var in x.items() if var.X > 0.5
            ]
            if is_spanning_tree(n, chosen):
                update_incumbent(chosen)

    except Exception:
        # The heuristic incumbent remains valid if model construction, license
        # availability, or optimization fails.
        return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    start_time = time.monotonic()
    deadline = start_time + max(0, args.time_limit)

    logger = (
        SolutionLogger(args.log_path, sense="minimize")
        if args.log_path
        else None
    )

    with open(args.instance_path, "r", encoding="utf-8") as f:
        instance = json.load(f)

    vertices = instance["vertices"]
    n = int(instance["n"])

    if len(vertices) != n:
        raise ValueError("The number of vertices does not match instance['n'].")

    ids = [int(v["id"]) for v in vertices]
    if len(set(ids)) != n:
        raise ValueError("Vertex identifiers must be unique.")

    coords = [(float(v["x"]), float(v["y"])) for v in vertices]

    weights = [[0.0] * n for _ in range(n)]
    for i in range(n):
        xi, yi = coords[i]
        for j in range(i + 1, n):
            xj, yj = coords[j]
            dx = xi - xj
            dy = yi - yj
            w = dx * dx + dy * dy
            weights[i][j] = w
            weights[j][i] = w

    if n <= 1:
        solution = compute_solution(n, ids, weights, [])
        if logger:
            logger.log_solution(solution["objective_value"], solution)
        parent = os.path.dirname(os.path.abspath(args.solution_path))
        os.makedirs(parent, exist_ok=True)
        with open(args.solution_path, "w", encoding="utf-8") as f:
            json.dump(solution, f, indent=2)
        return

    lower_bounds = [
        min(weights[i][j] for j in range(n) if j != i)
        for i in range(n)
    ]

    best_edges = prim_mst(weights)
    best_solution = compute_solution(n, ids, weights, best_edges)
    best_value = best_solution["objective_value"]

    if logger:
        logger.log_solution(best_value, best_solution)

    def update_incumbent(edges):
        nonlocal best_edges, best_solution, best_value

        normalized = sorted(
            set((min(i, j), max(i, j)) for i, j in edges)
        )
        if not is_spanning_tree(n, normalized):
            return False

        solution = compute_solution(n, ids, weights, normalized)
        value = solution["objective_value"]

        tolerance = 1e-8 * max(1.0, abs(best_value))
        if value < best_value - tolerance:
            best_edges = normalized
            best_solution = solution
            best_value = value
            if logger:
                logger.log_solution(value, solution)
            return True
        return False

    # The power-aware construction is cubic, so use it only on moderate
    # instances and while sufficient time remains.
    if n <= 140 and time.monotonic() < deadline:
        root_candidates = [0]

        # Add geometrically diverse roots.
        if n > 1:
            root_candidates.append(max(range(n), key=lambda i: weights[0][i]))
            centroid_x = sum(x for x, _ in coords) / n
            centroid_y = sum(y for _, y in coords) / n
            root_candidates.append(
                min(
                    range(n),
                    key=lambda i: (
                        (coords[i][0] - centroid_x) ** 2
                        + (coords[i][1] - centroid_y) ** 2
                    ),
                )
            )
            root_candidates.append(
                max(
                    range(n),
                    key=lambda i: (
                        (coords[i][0] - centroid_x) ** 2
                        + (coords[i][1] - centroid_y) ** 2
                    ),
                )
            )

        seen_roots = set()
        for root in root_candidates:
            if root in seen_roots or time.monotonic() >= deadline:
                continue
            seen_roots.add(root)
            tree = incremental_power_tree(
                weights, lower_bounds, root, deadline
            )
            if tree is not None:
                update_incumbent(tree)

    if time.monotonic() < deadline:
        leaf_relocation_search(
            n,
            ids,
            weights,
            best_edges,
            deadline,
            update_incumbent,
        )

    if time.monotonic() < deadline:
        candidate_edges = make_candidate_edges(weights, best_edges)
        solve_with_gurobi(
            n,
            ids,
            weights,
            lower_bounds,
            candidate_edges,
            best_edges,
            deadline,
            update_incumbent,
        )

    parent = os.path.dirname(os.path.abspath(args.solution_path))
    os.makedirs(parent, exist_ok=True)
    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(best_solution, f, indent=2)


if __name__ == "__main__":
    main()
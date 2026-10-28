import argparse
import json
import math
import random
import time
from pathlib import Path

from solution_logger import SolutionLogger


class DistanceAccessor:
    def __init__(self, n, values):
        self.n = n
        self.values = values

    def __call__(self, i, j):
        if i == j:
            return 0
        if i > j:
            i, j = j, i
        # Number of entries before row i in the flattened upper triangle.
        index = i * self.n - (i * (i + 1)) // 2 + (j - i - 1)
        return self.values[index]


class IncumbentManager:
    def __init__(self, node_ids, distance, logger):
        self.node_ids = node_ids
        self.distance = distance
        self.logger = logger
        self.best_cost = float("inf")
        self.best_tour = None

    def tour_cost(self, tour):
        if not tour or len(tour) == 1:
            return 0
        return sum(
            self.distance(tour[i], tour[(i + 1) % len(tour)])
            for i in range(len(tour))
        )

    def solution_dict(self, tour, cost=None):
        if cost is None:
            cost = self.tour_cost(tour)
        output_tour = [self.node_ids[i] for i in tour]
        return {
            "objective_value": float(cost),
            "tour": output_tour,
            "visited_nodes": list(output_tour),
        }

    def update(self, tour, cost=None):
        if tour is None:
            return False
        if cost is None:
            cost = self.tour_cost(tour)
        if cost + 1e-9 >= self.best_cost:
            return False

        self.best_cost = float(cost)
        self.best_tour = list(tour)
        solution = self.solution_dict(self.best_tour, self.best_cost)
        if self.logger:
            self.logger.log_solution(self.best_cost, solution)
        return True


def optimize_representatives(cluster_order, clusters, distance, deadline):
    """Find the optimal representative from every cluster for a fixed cycle order."""
    m = len(cluster_order)
    if m == 0:
        return [], 0
    if m == 1:
        return [clusters[cluster_order[0]][0]], 0

    if m == 2:
        a = clusters[cluster_order[0]]
        b = clusters[cluster_order[1]]
        best = None
        best_cost = float("inf")
        for u in a:
            for v in b:
                cost = 2 * distance(u, v)
                if cost < best_cost:
                    best_cost = cost
                    best = [u, v]
        return best, best_cost

    # Rotating the cycle so that its smallest cluster comes first reduces the
    # number of fixed-start dynamic programs.
    smallest_pos = min(
        range(m), key=lambda p: len(clusters[cluster_order[p]])
    )
    order = cluster_order[smallest_pos:] + cluster_order[:smallest_pos]
    layers = [clusters[c] for c in order]

    best_cost = float("inf")
    best_path = None

    for start_node in layers[0]:
        if time.monotonic() >= deadline:
            break

        previous_nodes = [start_node]
        previous_cost = [0]
        parents = []

        complete = True
        for layer_index in range(1, m):
            if time.monotonic() >= deadline:
                complete = False
                break

            current_nodes = layers[layer_index]
            current_cost = [float("inf")] * len(current_nodes)
            current_parent = [-1] * len(current_nodes)

            for j, v in enumerate(current_nodes):
                best_value = float("inf")
                best_predecessor = -1
                for i, u in enumerate(previous_nodes):
                    value = previous_cost[i] + distance(u, v)
                    if value < best_value:
                        best_value = value
                        best_predecessor = i
                current_cost[j] = best_value
                current_parent[j] = best_predecessor

            parents.append(current_parent)
            previous_nodes = current_nodes
            previous_cost = current_cost

        if not complete:
            break

        end_index = -1
        value = float("inf")
        for j, v in enumerate(previous_nodes):
            candidate = previous_cost[j] + distance(v, start_node)
            if candidate < value:
                value = candidate
                end_index = j

        if value < best_cost:
            path = [None] * m
            path[-1] = previous_nodes[end_index]
            index = end_index
            for layer_index in range(m - 1, 0, -1):
                index = parents[layer_index - 1][index]
                path[layer_index - 1] = layers[layer_index - 1][index]

            best_cost = value
            best_path = path

    return best_path, best_cost


def centroid_orders(clusters, coordinates):
    m = len(clusters)
    centroids = []
    for nodes in clusters:
        sx = sum(coordinates[v][0] for v in nodes)
        sy = sum(coordinates[v][1] for v in nodes)
        centroids.append((sx / len(nodes), sy / len(nodes)))

    if m <= 1:
        return [list(range(m))]

    global_x = sum(p[0] for p in centroids) / m
    global_y = sum(p[1] for p in centroids) / m

    angle_order = sorted(
        range(m),
        key=lambda c: math.atan2(
            centroids[c][1] - global_y, centroids[c][0] - global_x
        ),
    )

    orders = [list(range(m)), angle_order, list(reversed(angle_order))]

    starts = {
        0,
        min(range(m), key=lambda c: centroids[c][0]),
        max(range(m), key=lambda c: centroids[c][0]),
        min(range(m), key=lambda c: centroids[c][1]),
        max(range(m), key=lambda c: centroids[c][1]),
    }

    for start in starts:
        remaining = set(range(m))
        remaining.remove(start)
        order = [start]
        while remaining:
            last = order[-1]
            lx, ly = centroids[last]
            nxt = min(
                remaining,
                key=lambda c: (
                    (centroids[c][0] - lx) ** 2
                    + (centroids[c][1] - ly) ** 2,
                    c,
                ),
            )
            order.append(nxt)
            remaining.remove(nxt)
        orders.append(order)

    unique = []
    seen = set()
    for order in orders:
        key = tuple(order)
        if key not in seen:
            seen.add(key)
            unique.append(order)
    return unique


def improve_order_2opt(order, selected_tour, clusters, distance,
                       manager, deadline):
    """2-opt on cluster order, reoptimizing representatives after each move."""
    m = len(order)
    if m < 4:
        return order, selected_tour

    current_order = list(order)
    current_tour = list(selected_tour)

    while time.monotonic() < deadline:
        best_delta = 0
        best_move = None

        for i in range(m):
            if time.monotonic() >= deadline:
                break
            a_pos = i
            b_pos = (i + 1) % m
            if b_pos == 0:
                continue

            for k in range(i + 2, m):
                d_pos = (k + 1) % m
                if i == 0 and k == m - 1:
                    continue

                a = current_tour[a_pos]
                b = current_tour[b_pos]
                c = current_tour[k]
                d = current_tour[d_pos]

                delta = (
                    distance(a, c)
                    + distance(b, d)
                    - distance(a, b)
                    - distance(c, d)
                )
                if delta < best_delta:
                    best_delta = delta
                    best_move = (i + 1, k)

        if best_move is None:
            break

        left, right = best_move
        candidate_order = (
            current_order[:left]
            + list(reversed(current_order[left:right + 1]))
            + current_order[right + 1:]
        )
        candidate_tour, candidate_cost = optimize_representatives(
            candidate_order, clusters, distance, deadline
        )
        if candidate_tour is None:
            break

        # The held-representative 2-opt move is improving, so exact
        # representative optimization cannot make it worse.
        current_order = candidate_order
        current_tour = candidate_tour
        manager.update(candidate_tour, candidate_cost)

    return current_order, current_tour


def run_heuristics(clusters, coordinates, distance, manager,
                   deadline, rng):
    m = len(clusters)
    base_order = list(range(m))
    base_tour = [clusters[c][0] for c in base_order]
    manager.update(base_tour)

    orders = centroid_orders(clusters, coordinates)
    best_order = base_order

    for order in orders:
        if time.monotonic() >= deadline:
            break
        tour, cost = optimize_representatives(
            order, clusters, distance, deadline
        )
        if tour is None:
            continue
        if manager.update(tour, cost):
            best_order = list(order)
        improved_order, improved_tour = improve_order_2opt(
            order, tour, clusters, distance, manager, deadline
        )
        if manager.best_tour == improved_tour:
            best_order = improved_order

    # Randomized perturbations around the best known cluster order.
    while m >= 4 and time.monotonic() < deadline:
        order = list(best_order)
        perturbations = 1 if m < 15 else 2
        for _ in range(perturbations):
            i, j = sorted(rng.sample(range(m), 2))
            if i == j:
                continue
            if rng.random() < 0.5:
                order[i:j + 1] = reversed(order[i:j + 1])
            else:
                value = order.pop(j)
                order.insert(i, value)

        tour, cost = optimize_representatives(
            order, clusters, distance, deadline
        )
        if tour is None:
            break

        was_improved = manager.update(tour, cost)
        order, tour = improve_order_2opt(
            order, tour, clusters, distance, manager, deadline
        )
        if was_improved or manager.best_tour == tour:
            best_order = order


def connected_tour(selected, active_edges):
    if len(selected) == 1:
        return list(selected)
    adjacency = {v: [] for v in selected}
    for u, v in active_edges:
        if u in adjacency and v in adjacency:
            adjacency[u].append(v)
            adjacency[v].append(u)

    if any(len(adjacency[v]) != 2 for v in selected):
        return None

    start = selected[0]
    tour = [start]
    previous = None
    current = start
    while True:
        neighbors = adjacency[current]
        nxt = neighbors[0] if neighbors[0] != previous else neighbors[1]
        if nxt == start:
            break
        if nxt in tour:
            return None
        tour.append(nxt)
        previous, current = current, nxt

    return tour if len(tour) == len(selected) else None


def solve_with_gurobi(clusters, node_cluster, distance, manager, deadline):
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception:
        return

    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return

    n = len(node_cluster)
    m = len(clusters)
    edges = []
    costs = []
    incident = [[] for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            if node_cluster[i] == node_cluster[j]:
                continue
            idx = len(edges)
            edges.append((i, j))
            costs.append(distance(i, j))
            incident[i].append(idx)
            incident[j].append(idx)

    try:
        model = gp.Model("GTSP")
        model.Params.OutputFlag = 0
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1
        model.Params.LazyConstraints = 1
        model.Params.TimeLimit = max(0.01, deadline - time.monotonic())

        y = model.addVars(n, vtype=GRB.BINARY, name="y")
        x = model.addVars(
            len(edges), vtype=GRB.BINARY, obj=costs, name="x"
        )

        for nodes in clusters:
            model.addConstr(gp.quicksum(y[v] for v in nodes) == 1)

        for v in range(n):
            model.addConstr(
                gp.quicksum(x[e] for e in incident[v]) == 2 * y[v]
            )

        if manager.best_tour is not None:
            selected_set = set(manager.best_tour)
            for v in range(n):
                y[v].Start = 1.0 if v in selected_set else 0.0

            start_edges = set()
            tour = manager.best_tour
            for i in range(len(tour)):
                a = tour[i]
                b = tour[(i + 1) % len(tour)]
                start_edges.add((min(a, b), max(a, b)))
            for e, edge in enumerate(edges):
                x[e].Start = 1.0 if edge in start_edges else 0.0

        model._x = x
        model._y = y
        model._edges = edges
        model._incident = incident
        model._manager = manager

        def callback(cb_model, where):
            if where != GRB.Callback.MIPSOL:
                return
            try:
                selected = [
                    v for v in range(n)
                    if cb_model.cbGetSolution(cb_model._y[v]) > 0.5
                ]
                active_indices = [
                    e for e in range(len(edges))
                    if cb_model.cbGetSolution(cb_model._x[e]) > 0.5
                ]
                active_edges = [edges[e] for e in active_indices]

                adjacency = {v: [] for v in selected}
                for u, v in active_edges:
                    if u in adjacency and v in adjacency:
                        adjacency[u].append(v)
                        adjacency[v].append(u)

                unseen = set(selected)
                components = []
                while unseen:
                    root = next(iter(unseen))
                    stack = [root]
                    unseen.remove(root)
                    component = set()
                    while stack:
                        v = stack.pop()
                        component.add(v)
                        for w in adjacency.get(v, []):
                            if w in unseen:
                                unseen.remove(w)
                                stack.append(w)
                    components.append(component)

                if len(components) == 1:
                    tour = connected_tour(selected, active_edges)
                    if tour is not None:
                        cb_model._manager.update(tour)
                    return

                # For every incumbent subtour, force at least two edges to
                # leave its exact node set whenever one of its nodes is used.
                for component in components:
                    if not component or len(component) == len(selected):
                        continue
                    crossing = []
                    for u in component:
                        for e in incident[u]:
                            a, b = edges[e]
                            other = b if a == u else a
                            if other not in component:
                                crossing.append(e)
                    if crossing:
                        anchor = next(iter(component))
                        cb_model.cbLazy(
                            gp.quicksum(cb_model._x[e] for e in crossing)
                            >= 2 * cb_model._y[anchor]
                        )
            except Exception:
                # Never risk losing the best heuristic solution because of a
                # callback-side logging or extraction issue.
                return

        model.optimize(callback)

        if model.SolCount > 0:
            selected = [v for v in range(n) if y[v].X > 0.5]
            active_edges = [
                edges[e] for e in range(len(edges)) if x[e].X > 0.5
            ]
            tour = connected_tour(selected, active_edges)
            if tour is not None:
                manager.update(tour)

    except Exception:
        # The heuristic incumbent remains valid if model construction, license
        # acquisition, or optimization fails.
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
        if args.log_path else None
    )

    with open(args.instance_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    node_records = data["nodes"]
    n = len(node_records)
    node_ids = [record["id"] for record in node_records]
    id_to_index = {node_id: i for i, node_id in enumerate(node_ids)}
    coordinates = [(record["x"], record["y"]) for record in node_records]

    cluster_records = data["clusters"]
    clusters = [
        [id_to_index[node_id] for node_id in record["node_ids"]]
        for record in cluster_records
    ]
    if any(len(nodes) == 0 for nodes in clusters):
        raise ValueError("Every cluster must contain at least one node")

    node_cluster = [-1] * n
    for c, nodes in enumerate(clusters):
        for v in nodes:
            node_cluster[v] = c
    if any(c < 0 for c in node_cluster):
        raise ValueError("Clusters do not cover all nodes")

    values = data["distance_matrix_upper_triangular"]["values"]
    expected = n * (n - 1) // 2
    if len(values) != expected:
        raise ValueError(
            f"Distance array has length {len(values)}, expected {expected}"
        )

    distance = DistanceAccessor(n, values)
    manager = IncumbentManager(node_ids, distance, logger)

    m = len(clusters)
    if m == 0:
        manager.update([], 0)
    elif m == 1:
        manager.update([clusters[0][0]], 0)
    elif m == 2:
        best_pair = None
        best_cost = float("inf")
        for u in clusters[0]:
            for v in clusters[1]:
                cost = 2 * distance(u, v)
                if cost < best_cost:
                    best_cost = cost
                    best_pair = [u, v]
        manager.update(best_pair, best_cost)
    else:
        cross_edges = (
            n * (n - 1) // 2
            - sum(len(nodes) * (len(nodes) - 1) // 2 for nodes in clusters)
        )
        use_mip = cross_edges <= 60000 and m <= 120

        if use_mip:
            heuristic_budget = min(
                max(0.1, args.time_limit * 0.20),
                5.0,
            )
            heuristic_deadline = min(deadline, start_time + heuristic_budget)
        else:
            heuristic_deadline = deadline

        rng = random.Random(0)
        run_heuristics(
            clusters, coordinates, distance, manager,
            heuristic_deadline, rng
        )

        if use_mip and time.monotonic() < deadline:
            solve_with_gurobi(
                clusters, node_cluster, distance, manager, deadline
            )

    # A valid fallback is always available for a nonempty valid instance.
    if manager.best_tour is None:
        fallback = [nodes[0] for nodes in clusters]
        manager.update(fallback)

    solution = manager.solution_dict(
        manager.best_tour, manager.best_cost
    )
    output_path = Path(args.solution_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()
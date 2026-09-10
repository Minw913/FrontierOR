import argparse
import json
import math
import os
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def route_cost(route, distances):
    """Cost of a cyclic route whose depot appears once at route[0]."""
    if len(route) <= 1:
        return 0
    return sum(
        distances[route[i]][route[(i + 1) % len(route)]]
        for i in range(len(route))
    )


def route_to_solution(route, scores):
    if len(route) <= 1:
        edges = []
    else:
        edges = [
            [int(route[i]), int(route[(i + 1) % len(route)])]
            for i in range(len(route))
        ]

    return {
        "objective_value": float(sum(scores[v] for v in route)),
        "visited_vertices": [int(v) for v in route],
        "tour_edges": edges,
    }


def improve_route_2opt(route, distances, deadline):
    """Reduce route length without changing its visited vertices."""
    if len(route) < 4:
        return route

    route = list(route)
    while time.monotonic() < deadline:
        best_delta = 0
        best_move = None
        m = len(route)

        for i in range(1, m - 1):
            if time.monotonic() >= deadline:
                break
            a = route[i - 1]
            first = route[i]

            for j in range(i + 1, m):
                # Reversing every non-depot vertex only reverses orientation.
                if i == 1 and j == m - 1:
                    continue

                last = route[j]
                b = route[(j + 1) % m]
                delta = (
                    distances[a][last]
                    + distances[first][b]
                    - distances[a][first]
                    - distances[last][b]
                )
                if delta < best_delta:
                    best_delta = delta
                    best_move = (i, j)

        if best_move is None:
            break

        i, j = best_move
        route[i:j + 1] = reversed(route[i:j + 1])

    return route


def best_feasible_insertion(route, candidates, current_cost, budget,
                            distances, scores, mode, deadline):
    best = None
    best_key = None
    m = len(route)

    for idx, v in enumerate(candidates):
        if idx % 32 == 0 and time.monotonic() >= deadline:
            break

        best_delta = None
        best_pos = None
        for pos in range(m):
            a = route[pos]
            b = route[(pos + 1) % m]
            delta = distances[a][v] + distances[v][b] - distances[a][b]
            if current_cost + delta <= budget:
                if best_delta is None or delta < best_delta:
                    best_delta = delta
                    best_pos = pos + 1

        if best_delta is None:
            continue

        denominator = max(0, best_delta) + 1.0
        if mode == "score":
            metric = float(scores[v])
        elif mode == "sqrt":
            metric = float(scores[v]) / math.sqrt(denominator)
        else:
            metric = float(scores[v]) / denominator

        key = (metric, scores[v], -best_delta, -v)
        if best_key is None or key > best_key:
            best_key = key
            best = (v, best_pos, best_delta)

    return best


def run_heuristics(depot, distances, scores, budget, deadline, logger,
                   incumbent_solution, incumbent_value):
    n = len(scores)
    non_depot = [v for v in range(n) if v != depot]

    def accept(route):
        nonlocal incumbent_solution, incumbent_value
        solution = route_to_solution(route, scores)
        value = solution["objective_value"]
        if value > incumbent_value + 1e-9:
            incumbent_value = value
            incumbent_solution = solution
            if logger:
                logger.log_solution(value, solution)

    # A simple undirected cycle needs the depot and at least two other vertices.
    if len(non_depot) < 2:
        return incumbent_solution, incumbent_value

    for mode in ("ratio", "score", "sqrt"):
        if time.monotonic() >= deadline:
            break

        best_pair = None
        best_pair_key = None

        for ii, i in enumerate(non_depot):
            if ii % 16 == 0 and time.monotonic() >= deadline:
                break
            for j in non_depot[ii + 1:]:
                cost = distances[depot][i] + distances[i][j] + distances[j][depot]
                if cost > budget:
                    continue

                pair_score = scores[i] + scores[j]
                if mode == "score":
                    metric = float(pair_score)
                elif mode == "sqrt":
                    metric = float(pair_score) / math.sqrt(cost + 1.0)
                else:
                    metric = float(pair_score) / (cost + 1.0)

                key = (metric, pair_score, -cost, -i, -j)
                if best_pair_key is None or key > best_pair_key:
                    best_pair_key = key
                    best_pair = (i, j, cost)

        if best_pair is None:
            continue

        i, j, current_cost = best_pair
        route = [depot, i, j]
        remaining = set(non_depot)
        remaining.discard(i)
        remaining.discard(j)
        accept(route)

        while remaining and time.monotonic() < deadline:
            insertion = best_feasible_insertion(
                route, remaining, current_cost, budget,
                distances, scores, mode, deadline
            )
            if insertion is None:
                break

            v, pos, delta = insertion
            route.insert(pos, v)
            remaining.remove(v)
            current_cost += delta
            accept(route)

        route = improve_route_2opt(route, distances, deadline)
        current_cost = route_cost(route, distances)

        # A shorter 2-opt route may make additional insertions possible.
        while remaining and time.monotonic() < deadline:
            insertion = best_feasible_insertion(
                route, remaining, current_cost, budget,
                distances, scores, "ratio", deadline
            )
            if insertion is None:
                break

            v, pos, delta = insertion
            route.insert(pos, v)
            remaining.remove(v)
            current_cost += delta
            accept(route)

    return incumbent_solution, incumbent_value


def connected_components(selected, selected_edges):
    adjacency = {v: [] for v in selected}
    for i, j in selected_edges:
        if i in adjacency and j in adjacency:
            adjacency[i].append(j)
            adjacency[j].append(i)

    unseen = set(selected)
    components = []
    while unseen:
        root = next(iter(unseen))
        stack = [root]
        unseen.remove(root)
        component = []

        while stack:
            v = stack.pop()
            component.append(v)
            for w in adjacency[v]:
                if w in unseen:
                    unseen.remove(w)
                    stack.append(w)

        components.append(component)

    return components


def validate_mip_solution(selected, edges, depot, distances, budget):
    if depot not in selected or len(selected) < 3:
        return False

    degree = {v: 0 for v in selected}
    total_cost = 0

    for i, j in edges:
        if i not in degree or j not in degree:
            return False
        degree[i] += 1
        degree[j] += 1
        total_cost += distances[i][j]

    if total_cost > budget + 1e-6:
        return False
    if any(degree[v] != 2 for v in selected):
        return False

    return len(connected_components(selected, edges)) == 1


def solve_with_gurobi(distances, scores, depot, budget, deadline,
                      logger, incumbent_solution, incumbent_value):
    n = len(scores)
    if time.monotonic() >= deadline:
        return incumbent_solution, incumbent_value

    model = gp.Model("orienteering")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.LazyConstraints = 1

    y = model.addVars(n, vtype=GRB.BINARY, name="visit")
    edge_keys = [(i, j) for i in range(n) for j in range(i + 1, n)]
    x = model.addVars(edge_keys, vtype=GRB.BINARY, name="edge")

    incident = [[] for _ in range(n)]
    for i, j in edge_keys:
        incident[i].append(x[i, j])
        incident[j].append(x[i, j])

    model.addConstr(y[depot] == 1, name="visit_depot")
    for v in range(n):
        model.addConstr(
            gp.quicksum(incident[v]) == 2 * y[v],
            name=f"degree_{v}"
        )

    model.addConstr(
        gp.quicksum(distances[i][j] * x[i, j] for i, j in edge_keys)
        <= budget,
        name="distance_budget"
    )

    model.setObjective(
        gp.quicksum(scores[v] * y[v] for v in range(n)),
        GRB.MAXIMIZE
    )

    # Supply a feasible heuristic cycle as a MIP start when available.
    warm_vertices = set(incumbent_solution["visited_vertices"])
    warm_edges = {
        (min(e[0], e[1]), max(e[0], e[1]))
        for e in incumbent_solution["tour_edges"]
        if e[0] != e[1]
    }
    if len(warm_vertices) >= 3 and len(warm_edges) == len(warm_vertices):
        for v in range(n):
            y[v].Start = 1.0 if v in warm_vertices else 0.0
        for key in edge_keys:
            x[key].Start = 1.0 if key in warm_edges else 0.0

    model._y = y
    model._x = x
    model._edge_keys = edge_keys
    model._depot = depot
    model._scores = scores
    model._logger = logger
    model._best_logged = incumbent_value

    def callback(cb_model, where):
        if where != GRB.Callback.MIPSOL:
            return

        try:
            y_values = cb_model.cbGetSolution(
                [cb_model._y[v] for v in range(n)]
            )
            x_values = cb_model.cbGetSolution(
                [cb_model._x[key] for key in cb_model._edge_keys]
            )

            selected = {
                v for v, value in enumerate(y_values) if value > 0.5
            }
            selected_edges = [
                key for key, value in zip(cb_model._edge_keys, x_values)
                if value > 0.5
            ]

            components = connected_components(selected, selected_edges)
            disconnected = len(components) > 1

            if disconnected:
                # For each component not containing the depot, impose a
                # generalized subtour elimination inequality.
                for component in components:
                    if cb_model._depot in component:
                        continue

                    component = sorted(component)
                    root = component[0]
                    internal_edges = [
                        cb_model._x[i, j]
                        for p, i in enumerate(component)
                        for j in component[p + 1:]
                    ]
                    cb_model.cbLazy(
                        gp.quicksum(internal_edges)
                        <= gp.quicksum(cb_model._y[v] for v in component)
                        - cb_model._y[root]
                    )
                return

            if cb_model._depot not in selected or len(selected) < 3:
                return

            value = float(sum(cb_model._scores[v] for v in selected))
            if value > cb_model._best_logged + 1e-9:
                solution = {
                    "objective_value": value,
                    "visited_vertices": sorted(int(v) for v in selected),
                    "tour_edges": [[int(i), int(j)] for i, j in selected_edges],
                }
                cb_model._best_logged = value
                if cb_model._logger:
                    cb_model._logger.log_solution(value, solution)

        except gp.GurobiError:
            # Do not interrupt optimization because of nonessential logging.
            return

    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return incumbent_solution, incumbent_value

    model.Params.TimeLimit = max(0.01, remaining)
    model.optimize(callback)

    if model.SolCount > 0:
        selected = {v for v in range(n) if y[v].X > 0.5}
        selected_edges = [key for key in edge_keys if x[key].X > 0.5]

        if validate_mip_solution(
            selected, selected_edges, depot, distances, budget
        ):
            value = float(sum(scores[v] for v in selected))
            solution = {
                "objective_value": value,
                "visited_vertices": sorted(int(v) for v in selected),
                "tour_edges": [
                    [int(i), int(j)] for i, j in selected_edges
                ],
            }

            if value > incumbent_value + 1e-9:
                incumbent_value = value
                incumbent_solution = solution
                if logger and value > model._best_logged + 1e-9:
                    logger.log_solution(value, solution)

    return incumbent_solution, incumbent_value


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
        SolutionLogger(args.log_path, sense="maximize")
        if args.log_path else None
    )

    with open(args.instance_path, "r", encoding="utf-8") as f:
        instance = json.load(f)

    n = int(instance["num_nodes"])
    budget = int(instance["distance_limitation_d0"])
    depot = int(instance["depot_vertex"])
    scores = [int(v) for v in instance["scores"]]
    distances = instance["distance_matrix"]

    if len(scores) != n or len(distances) != n:
        raise ValueError("Instance dimensions do not match num_nodes")

    # Trivial fallback for instances having no feasible nontrivial cycle.
    incumbent_solution = {
        "objective_value": float(scores[depot]),
        "visited_vertices": [depot],
        "tour_edges": [],
    }
    incumbent_value = incumbent_solution["objective_value"]

    if logger:
        logger.log_solution(incumbent_value, incumbent_solution)

    # Keep heuristic effort bounded so most time remains for exact MIP search.
    total_limit = max(0.0, float(args.time_limit))
    heuristic_allowance = min(3.0, max(0.0, 0.10 * total_limit))
    heuristic_deadline = min(deadline, time.monotonic() + heuristic_allowance)

    incumbent_solution, incumbent_value = run_heuristics(
        depot,
        distances,
        scores,
        budget,
        heuristic_deadline,
        logger,
        incumbent_solution,
        incumbent_value,
    )

    if time.monotonic() < deadline:
        try:
            incumbent_solution, incumbent_value = solve_with_gurobi(
                distances,
                scores,
                depot,
                budget,
                deadline,
                logger,
                incumbent_solution,
                incumbent_value,
            )
        except gp.GurobiError:
            # Preserve and emit the best heuristic solution if Gurobi cannot run.
            pass

    output_dir = os.path.dirname(os.path.abspath(args.solution_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(incumbent_solution, f, separators=(",", ":"))


if __name__ == "__main__":
    main()
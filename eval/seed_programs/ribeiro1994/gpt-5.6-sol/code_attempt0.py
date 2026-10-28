import argparse
import json
import math
import os
import sys
import time
from collections import deque

import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    return parser.parse_args()


def euclidean(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def hopcroft_karp(adjacency, n):
    """Maximum-cardinality matching in the trip-predecessor/successor graph."""
    sys.setrecursionlimit(max(10000, 4 * n + 100))

    pair_left = [-1] * n
    pair_right = [-1] * n
    distance = [0] * n
    infinity = n + 1

    def bfs():
        queue = deque()
        found_augmenting = False

        for u in range(n):
            if pair_left[u] == -1:
                distance[u] = 0
                queue.append(u)
            else:
                distance[u] = infinity

        while queue:
            u = queue.popleft()
            for v in adjacency[u]:
                matched_u = pair_right[v]
                if matched_u == -1:
                    found_augmenting = True
                elif distance[matched_u] == infinity:
                    distance[matched_u] = distance[u] + 1
                    queue.append(matched_u)

        return found_augmenting

    def dfs(u):
        for v in adjacency[u]:
            matched_u = pair_right[v]
            if matched_u == -1 or (
                distance[matched_u] == distance[u] + 1 and dfs(matched_u)
            ):
                pair_left[u] = v
                pair_right[v] = u
                return True

        distance[u] = infinity
        return False

    while bfs():
        for u in range(n):
            if pair_left[u] == -1:
                dfs(u)

    return pair_left, pair_right


def routes_from_matching(pair_left, pair_right, n):
    routes = []
    visited = [False] * n

    for first in range(n):
        if pair_right[first] != -1 or visited[first]:
            continue

        route = []
        current = first
        while current != -1 and not visited[current]:
            visited[current] = True
            route.append(current)
            current = pair_left[current]
        routes.append(route)

    # Defensive handling for malformed/cyclic compatibility data.
    for first in range(n):
        if visited[first]:
            continue

        route = []
        current = first
        while current != -1 and not visited[current]:
            visited[current] = True
            route.append(current)
            current = pair_left[current]
        if route:
            routes.append(route)

    return routes


def assign_routes_to_depots(routes, capacities, start_cost, end_cost):
    """
    Capacity-feasible greedy depot assignment. Routes with a high penalty for
    losing their preferred depot are assigned first.
    """
    m = len(capacities)
    remaining = list(capacities)

    route_options = []
    for route_index, route in enumerate(routes):
        first = route[0]
        last = route[-1]
        options = sorted(
            (
                start_cost[d][first] + end_cost[d][last],
                d,
            )
            for d in range(m)
            if capacities[d] > 0
        )

        if not options:
            return None

        regret = (
            options[1][0] - options[0][0]
            if len(options) >= 2
            else 10**18
        )
        route_options.append((regret, route_index, options))

    route_options.sort(key=lambda item: (-item[0], item[1]))
    assignments = [-1] * len(routes)

    for _, route_index, options in route_options:
        chosen = -1
        for _, depot in options:
            if remaining[depot] > 0:
                chosen = depot
                break

        if chosen == -1:
            return None

        assignments[route_index] = chosen
        remaining[chosen] -= 1

    return assignments


def make_solution_from_routes(
    indexed_routes,
    trip_ids,
    start_cost,
    end_cost,
    transition_cost,
):
    output_routes = []
    objective = 0

    for depot, route in indexed_routes:
        if not route:
            continue

        objective += start_cost[depot][route[0]]
        objective += end_cost[depot][route[-1]]

        for i, j in zip(route, route[1:]):
            objective += transition_cost[(i, j)]

        output_routes.append(
            {
                "depot": depot + 1,
                "trips": [trip_ids[i] for i in route],
            }
        )

    output_routes.sort(
        key=lambda r: (
            r["depot"],
            r["trips"][0] if r["trips"] else -1,
        )
    )

    return {
        "objective_value": float(objective),
        "routes": output_routes,
        "num_vehicles": len(output_routes),
    }


def write_solution(path, solution):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)

    temporary_path = path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as file:
        json.dump(solution, file, indent=2)
    os.replace(temporary_path, path)


def main():
    args = parse_args()
    start_wall_time = time.monotonic()

    logger = (
        SolutionLogger(args.log_path, sense="minimize")
        if args.log_path
        else None
    )

    with open(args.instance_path, "r", encoding="utf-8") as file:
        instance = json.load(file)

    trips = instance["trips"]
    relief_points = instance["relief_points"]
    depots = instance["depots"]
    capacities = [int(x) for x in instance["vehicle_capacities"]]

    n = len(trips)
    m = len(depots)

    trip_ids = [int(trip["trip_id"]) for trip in trips]
    trip_index = {trip_id: i for i, trip_id in enumerate(trip_ids)}

    start_point = [
        int(trip["start_relief_point_index"]) for trip in trips
    ]
    end_point = [
        int(trip["end_relief_point_index"]) for trip in trips
    ]
    start_time = [float(trip["start_time"]) for trip in trips]
    end_time = [float(trip["end_time"]) for trip in trips]

    # Depot pull-out and pull-in costs, with formula-based fallbacks.
    start_cost = [[0] * n for _ in range(m)]
    end_cost = [[0] * n for _ in range(m)]

    for d in range(m):
        for i in range(n):
            distance_out = euclidean(
                depots[d], relief_points[start_point[i]]
            )
            distance_in = euclidean(
                relief_points[end_point[i]], depots[d]
            )
            start_cost[d][i] = math.floor(10.0 * distance_out) + 5000
            end_cost[d][i] = math.floor(10.0 * distance_in) + 5000

    for item in instance.get("depot_to_trip_costs", []):
        d = int(item["depot"]) - 1
        trip_id = int(item["trip"])
        if 0 <= d < m and trip_id in trip_index:
            start_cost[d][trip_index[trip_id]] = int(item["cost"])

    for item in instance.get("trip_to_depot_costs", []):
        d = int(item["depot"]) - 1
        trip_id = int(item["trip"])
        if 0 <= d < m and trip_id in trip_index:
            end_cost[d][trip_index[trip_id]] = int(item["cost"])

    # Deduplicate compatibility arcs, retaining the cheapest supplied arc.
    transition_cost = {}
    for item in instance.get("compatible_arcs", []):
        from_id = int(item["from_trip"])
        to_id = int(item["to_trip"])

        if from_id not in trip_index or to_id not in trip_index:
            continue

        i = trip_index[from_id]
        j = trip_index[to_id]
        if i == j:
            continue

        travel = float(item.get("travel_time", 0.0))
        if end_time[i] + travel > start_time[j] + 1e-7:
            continue

        # Break possible zero-duration ties consistently so the selected
        # compatibility graph remains acyclic.
        order_i = (start_time[i], end_time[i], trip_ids[i])
        order_j = (start_time[j], end_time[j], trip_ids[j])
        if order_i >= order_j:
            continue

        cost = int(item["cost"])
        key = (i, j)
        if key not in transition_cost or cost < transition_cost[key]:
            transition_cost[key] = cost

    arcs = sorted(
        transition_cost.keys(),
        key=lambda arc: (
            start_time[arc[0]],
            start_time[arc[1]],
            trip_ids[arc[0]],
            trip_ids[arc[1]],
        ),
    )
    arc_index = {arc: a for a, arc in enumerate(arcs)}

    predecessor_arcs = [[] for _ in range(n)]
    successor_arcs = [[] for _ in range(n)]
    adjacency = [[] for _ in range(n)]

    for a, (i, j) in enumerate(arcs):
        successor_arcs[i].append(a)
        predecessor_arcs[j].append(a)
        adjacency[i].append(j)

    for i in range(n):
        adjacency[i].sort(
            key=lambda j: (
                transition_cost[(i, j)],
                start_time[j],
                trip_ids[j],
            )
        )

    # Obtain a fast feasible path cover for use as the initial incumbent.
    pair_left, pair_right = hopcroft_karp(adjacency, n)
    initial_routes = routes_from_matching(pair_left, pair_right, n)

    initial_solution = None
    initial_assignments = None

    if len(initial_routes) <= sum(capacities):
        initial_assignments = assign_routes_to_depots(
            initial_routes,
            capacities,
            start_cost,
            end_cost,
        )

        if initial_assignments is not None:
            indexed = [
                (initial_assignments[r], route)
                for r, route in enumerate(initial_routes)
            ]
            initial_solution = make_solution_from_routes(
                indexed,
                trip_ids,
                start_cost,
                end_cost,
                transition_cost,
            )
            if logger:
                logger.log_solution(
                    initial_solution["objective_value"],
                    initial_solution,
                )

    # Multi-commodity path-cover formulation. Each depot is one commodity.
    model = gp.Model("multi_depot_vehicle_scheduling")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1

    elapsed = time.monotonic() - start_wall_time
    remaining_time = max(0.01, float(args.time_limit) - elapsed)
    model.Params.TimeLimit = remaining_time

    start_vars = model.addVars(
        m, n, vtype=GRB.BINARY, name="start"
    )
    end_vars = model.addVars(
        m, n, vtype=GRB.BINARY, name="end"
    )

    transition_keys = [
        (d, a) for d in range(m) for a in range(len(arcs))
    ]
    transition_vars = model.addVars(
        transition_keys, vtype=GRB.BINARY, name="transition"
    )

    objective = gp.LinExpr()

    for d in range(m):
        for i in range(n):
            objective += start_cost[d][i] * start_vars[d, i]
            objective += end_cost[d][i] * end_vars[d, i]

    for d in range(m):
        for a, arc in enumerate(arcs):
            objective += transition_cost[arc] * transition_vars[d, a]

    model.setObjective(objective, GRB.MINIMIZE)

    # Every trip has exactly one incoming pull-out or trip transition.
    for i in range(n):
        incoming = gp.LinExpr()
        for d in range(m):
            incoming += start_vars[d, i]
            for a in predecessor_arcs[i]:
                incoming += transition_vars[d, a]
        model.addConstr(incoming == 1, name=f"cover_{i}")

    # Depot-commodity conservation ensures each route returns to its origin.
    for d in range(m):
        for i in range(n):
            incoming = start_vars[d, i] + gp.quicksum(
                transition_vars[d, a] for a in predecessor_arcs[i]
            )
            outgoing = end_vars[d, i] + gp.quicksum(
                transition_vars[d, a] for a in successor_arcs[i]
            )
            model.addConstr(
                incoming == outgoing,
                name=f"flow_{d}_{i}",
            )

    for d in range(m):
        model.addConstr(
            gp.quicksum(start_vars[d, i] for i in range(n))
            <= capacities[d],
            name=f"capacity_{d}",
        )

    # Supply the path-cover heuristic as a complete MIP start.
    if initial_solution is not None:
        for route_number, route in enumerate(initial_routes):
            d = initial_assignments[route_number]
            start_vars[d, route[0]].Start = 1.0
            end_vars[d, route[-1]].Start = 1.0

            for i, j in zip(route, route[1:]):
                a = arc_index[(i, j)]
                transition_vars[d, a].Start = 1.0

    start_keys = [(d, i) for d in range(m) for i in range(n)]
    end_keys = [(d, i) for d in range(m) for i in range(n)]
    transition_key_list = transition_keys

    start_var_list = [start_vars[key] for key in start_keys]
    end_var_list = [end_vars[key] for key in end_keys]
    transition_var_list = [
        transition_vars[key] for key in transition_key_list
    ]

    def build_from_values(start_values, end_values, transition_values):
        selected_starts = [[] for _ in range(m)]
        selected_ends = [set() for _ in range(m)]
        successors = [dict() for _ in range(m)]

        for value, (d, i) in zip(start_values, start_keys):
            if value > 0.5:
                selected_starts[d].append(i)

        for value, (d, i) in zip(end_values, end_keys):
            if value > 0.5:
                selected_ends[d].add(i)

        for value, (d, a) in zip(
            transition_values, transition_key_list
        ):
            if value > 0.5:
                i, j = arcs[a]
                successors[d][i] = j

        indexed_routes = []
        covered = set()

        for d in range(m):
            selected_starts[d].sort(
                key=lambda i: (start_time[i], trip_ids[i])
            )
            for first in selected_starts[d]:
                route = []
                local_seen = set()
                current = first

                while current is not None and current not in local_seen:
                    local_seen.add(current)
                    covered.add(current)
                    route.append(current)
                    current = successors[d].get(current)

                if route:
                    indexed_routes.append((d, route))

        if len(covered) != n:
            return None

        return make_solution_from_routes(
            indexed_routes,
            trip_ids,
            start_cost,
            end_cost,
            transition_cost,
        )

    best_logged = [
        initial_solution["objective_value"]
        if initial_solution is not None
        else math.inf
    ]

    def callback(cb_model, where):
        if where != GRB.Callback.MIPSOL:
            return

        try:
            incumbent_obj = cb_model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if incumbent_obj >= best_logged[0] - 1e-6:
                return

            start_values = cb_model.cbGetSolution(start_var_list)
            end_values = cb_model.cbGetSolution(end_var_list)
            transition_values = cb_model.cbGetSolution(
                transition_var_list
            )

            solution = build_from_values(
                start_values,
                end_values,
                transition_values,
            )
            if solution is None:
                return

            exact_obj = solution["objective_value"]
            if exact_obj < best_logged[0] - 1e-6:
                best_logged[0] = exact_obj
                if logger:
                    logger.log_solution(exact_obj, solution)
        except Exception:
            # Logging must never interrupt optimization.
            return

    model.optimize(callback)

    final_solution = initial_solution

    if model.SolCount > 0:
        model_solution = build_from_values(
            [var.X for var in start_var_list],
            [var.X for var in end_var_list],
            [var.X for var in transition_var_list],
        )

        if model_solution is not None and (
            final_solution is None
            or model_solution["objective_value"]
            < final_solution["objective_value"] - 1e-6
        ):
            final_solution = model_solution
            if (
                logger
                and final_solution["objective_value"]
                < best_logged[0] - 1e-6
            ):
                best_logged[0] = final_solution["objective_value"]
                logger.log_solution(
                    final_solution["objective_value"],
                    final_solution,
                )

    if final_solution is None:
        raise RuntimeError(
            "No feasible solution was found. The instance may be infeasible "
            "under the depot vehicle capacities."
        )

    write_solution(args.solution_path, final_solution)


if __name__ == "__main__":
    main()
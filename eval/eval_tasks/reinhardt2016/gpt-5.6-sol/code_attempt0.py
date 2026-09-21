import argparse
import json
import math
import os
import random
import time
from collections import defaultdict

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


TOL = 1e-6


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    return parser.parse_args()


def route_schedule(route, nodes, dist):
    """Return (feasible, customer visit times, route demand)."""
    depot = nodes[0]
    current_time = float(depot["time_window_start"])
    previous = 0
    visits = {}
    total_demand = 0

    for customer in route:
        current_time += float(nodes[previous]["service_time"]) + float(dist[previous][customer])
        current_time = max(current_time, float(nodes[customer]["time_window_start"]))
        if current_time > float(nodes[customer]["time_window_end"]) + TOL:
            return False, {}, total_demand

        visits[customer] = current_time
        total_demand += int(nodes[customer]["demand"])
        previous = customer

    if route:
        return_time = (
            current_time
            + float(nodes[previous]["service_time"])
            + float(dist[previous][0])
        )
    else:
        return_time = float(depot["time_window_start"])

    if return_time > float(depot["time_window_end"]) + TOL:
        return False, {}, total_demand

    return True, visits, total_demand


def route_edges(route):
    seq = [0] + list(route) + [0]
    return [(seq[i], seq[i + 1]) for i in range(len(seq) - 1)]


def required_edge_sets(routes, edge_to_set):
    result = set()
    for route in routes:
        for i, j in route_edges(route):
            edge_set = edge_to_set.get((min(i, j), max(i, j)))
            if edge_set is not None:
                result.add(edge_set)
    return result


def routes_objective(routes, dist, edge_to_set, access_cost):
    travel = 0.0
    for route in routes:
        for i, j in route_edges(route):
            travel += float(dist[i][j])

    used = required_edge_sets(routes, edge_to_set)
    fixed = sum(float(access_cost[s]) for s in used)
    return travel + fixed


def build_solution(routes, nodes, dist, edge_to_set, access_cost):
    used_sets = sorted(required_edge_sets(routes, edge_to_set))
    output_routes = []

    for vehicle, route in enumerate(routes):
        feasible, visits, _ = route_schedule(route, nodes, dist)
        if not feasible:
            return None

        # The formulation assigns a valid time-window value even to nodes not
        # visited by this route. Visited customer values are then overwritten.
        visit_times = {
            str(i): float(nodes[i]["time_window_start"]) for i in range(len(nodes))
        }
        visit_times["0"] = float(nodes[0]["time_window_start"])
        for i, value in visits.items():
            visit_times[str(i)] = float(value)

        sequence = [0] + list(route) + [0]
        arcs = [[sequence[p], sequence[p + 1]] for p in range(len(sequence) - 1)]

        output_routes.append(
            {
                "vehicle": vehicle,
                "sequence": sequence,
                "arcs": arcs,
                "visit_times": visit_times,
            }
        )

    objective = routes_objective(routes, dist, edge_to_set, access_cost)
    return {
        "objective_value": float(objective),
        "routes": output_routes,
        "edge_sets_used": used_sets,
    }


def greedy_initial_solutions(
    n,
    max_vehicles,
    capacity,
    nodes,
    dist,
    edge_to_set,
    access_cost,
    deadline,
):
    rng = random.Random(0)
    customers = list(range(1, n + 1))

    deterministic_orders = [
        sorted(customers, key=lambda i: (nodes[i]["time_window_end"], nodes[i]["time_window_start"])),
        sorted(customers, key=lambda i: (nodes[i]["time_window_start"], nodes[i]["time_window_end"])),
        sorted(customers, key=lambda i: (-nodes[i]["demand"], nodes[i]["time_window_end"])),
        sorted(
            customers,
            key=lambda i: math.atan2(
                nodes[i]["y"] - nodes[0]["y"], nodes[i]["x"] - nodes[0]["x"]
            ),
        ),
        sorted(customers, key=lambda i: dist[0][i], reverse=True),
    ]

    best_routes = None
    best_objective = math.inf
    attempt = 0

    while time.monotonic() < deadline:
        if attempt < len(deterministic_orders):
            order = deterministic_orders[attempt]
        else:
            order = customers[:]
            rng.shuffle(order)

        attempt += 1
        routes = []
        failed = False

        for customer in order:
            if time.monotonic() >= deadline:
                failed = True
                break

            best_candidate = None
            best_candidate_value = math.inf

            for r_index, route in enumerate(routes):
                current_demand = sum(nodes[i]["demand"] for i in route)
                if current_demand + nodes[customer]["demand"] > capacity:
                    continue

                for position in range(len(route) + 1):
                    candidate_route = route[:position] + [customer] + route[position:]
                    feasible, _, demand = route_schedule(candidate_route, nodes, dist)
                    if not feasible or demand > capacity:
                        continue

                    candidate_routes = [r[:] for r in routes]
                    candidate_routes[r_index] = candidate_route
                    value = routes_objective(
                        candidate_routes, dist, edge_to_set, access_cost
                    )
                    if value < best_candidate_value - TOL:
                        best_candidate_value = value
                        best_candidate = candidate_routes

            if len(routes) < max_vehicles:
                candidate_route = [customer]
                feasible, _, demand = route_schedule(candidate_route, nodes, dist)
                if feasible and demand <= capacity:
                    candidate_routes = [r[:] for r in routes] + [candidate_route]
                    value = routes_objective(
                        candidate_routes, dist, edge_to_set, access_cost
                    )
                    if value < best_candidate_value - TOL:
                        best_candidate_value = value
                        best_candidate = candidate_routes

            if best_candidate is None:
                failed = True
                break

            routes = best_candidate

        if not failed and sum(len(r) for r in routes) == n:
            objective = routes_objective(routes, dist, edge_to_set, access_cost)
            if objective < best_objective - TOL:
                best_objective = objective
                best_routes = [r[:] for r in routes]

        # A modest number of randomized attempts is sufficient; preserve most
        # of the time budget for the exact MIP search.
        if attempt >= max(8, min(30, n + 5)):
            break

    return best_routes


def extract_routes_from_selected(selected_arcs, n, max_vehicles):
    outgoing = defaultdict(list)
    incoming = defaultdict(list)

    for i, j in selected_arcs:
        outgoing[i].append(j)
        incoming[j].append(i)

    for customer in range(1, n + 1):
        if len(outgoing[customer]) != 1 or len(incoming[customer]) != 1:
            return None

    starts = list(outgoing[0])
    if len(starts) > max_vehicles:
        return None

    routes = []
    visited = set()

    for start in starts:
        route = []
        current = start
        local_seen = set()

        while current != 0:
            if current in local_seen or current in visited:
                return None
            if current < 1 or current > n:
                return None

            local_seen.add(current)
            visited.add(current)
            route.append(current)

            if len(outgoing[current]) != 1:
                return None
            current = outgoing[current][0]

            if len(route) > n:
                return None

        routes.append(route)

    if visited != set(range(1, n + 1)):
        return None

    return routes


def main():
    args = parse_args()
    start_time = time.monotonic()
    absolute_deadline = start_time + max(0, args.time_limit)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    n = int(data["num_customers"])
    max_vehicles = min(int(data["num_vehicles"]), n)
    capacity = int(data["vehicle_capacity"])
    dist = data["distance_matrix"]

    nodes = [None] * (n + 1)
    for node in data["nodes"]:
        nodes[int(node["id"])] = node

    edge_to_set = {}
    access_cost = {}
    edge_set_ids = []

    for edge_set in data["edge_sets"]:
        sid = int(edge_set["id"])
        edge_set_ids.append(sid)
        access_cost[sid] = float(edge_set["access_cost"])
        for edge in edge_set["edges"]:
            i, j = int(edge[0]), int(edge[1])
            edge_to_set[(min(i, j), max(i, j))] = sid

    best_solution = None
    best_objective = math.inf

    # Reserve most of the runtime for Gurobi.
    heuristic_allowance = min(
        3.0,
        max(0.0, 0.10 * args.time_limit),
        max(0.0, absolute_deadline - time.monotonic()),
    )
    heuristic_deadline = time.monotonic() + heuristic_allowance

    if heuristic_allowance > 0.0:
        initial_routes = greedy_initial_solutions(
            n,
            max_vehicles,
            capacity,
            nodes,
            dist,
            edge_to_set,
            access_cost,
            heuristic_deadline,
        )
    else:
        initial_routes = None

    if initial_routes is not None:
        solution = build_solution(
            initial_routes, nodes, dist, edge_to_set, access_cost
        )
        if solution is not None:
            best_solution = solution
            best_objective = float(solution["objective_value"])
            if logger:
                logger.log_solution(best_objective, solution)

    model = gp.Model("fixed_charge_cvrptw")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.OutputFlag = 0

    customers = range(1, n + 1)
    depot_start = float(nodes[0]["time_window_start"])
    depot_end = float(nodes[0]["time_window_end"])
    depot_service = float(nodes[0]["service_time"])

    allowed_arcs = []
    for i in range(n + 1):
        for j in range(n + 1):
            if i == j:
                continue

            feasible = True

            if i == 0 and j != 0:
                earliest_arrival = depot_start + depot_service + float(dist[i][j])
                if earliest_arrival > float(nodes[j]["time_window_end"]) + TOL:
                    feasible = False

            elif i != 0 and j == 0:
                earliest_return = (
                    float(nodes[i]["time_window_start"])
                    + float(nodes[i]["service_time"])
                    + float(dist[i][0])
                )
                if earliest_return > depot_end + TOL:
                    feasible = False

            elif i != 0 and j != 0:
                earliest_arrival = (
                    float(nodes[i]["time_window_start"])
                    + float(nodes[i]["service_time"])
                    + float(dist[i][j])
                )
                if earliest_arrival > float(nodes[j]["time_window_end"]) + TOL:
                    feasible = False
                if int(nodes[i]["demand"]) + int(nodes[j]["demand"]) > capacity:
                    feasible = False

            if feasible:
                allowed_arcs.append((i, j))

    x = model.addVars(allowed_arcs, vtype=GRB.BINARY, name="x")
    y = model.addVars(edge_set_ids, vtype=GRB.BINARY, name="edge_set")

    visit_time = {}
    load = {}
    order = {}

    for i in customers:
        visit_time[i] = model.addVar(
            lb=float(nodes[i]["time_window_start"]),
            ub=float(nodes[i]["time_window_end"]),
            vtype=GRB.CONTINUOUS,
            name=f"time_{i}",
        )
        load[i] = model.addVar(
            lb=float(nodes[i]["demand"]),
            ub=float(capacity),
            vtype=GRB.CONTINUOUS,
            name=f"load_{i}",
        )
        order[i] = model.addVar(
            lb=1.0,
            ub=float(n),
            vtype=GRB.CONTINUOUS,
            name=f"order_{i}",
        )

    model.update()

    outgoing_arcs = defaultdict(list)
    incoming_arcs = defaultdict(list)
    for i, j in allowed_arcs:
        outgoing_arcs[i].append((i, j))
        incoming_arcs[j].append((i, j))

    for i in customers:
        model.addConstr(
            gp.quicksum(x[a] for a in incoming_arcs[i]) == 1,
            name=f"visit_in_{i}",
        )
        model.addConstr(
            gp.quicksum(x[a] for a in outgoing_arcs[i]) == 1,
            name=f"visit_out_{i}",
        )

    model.addConstr(
        gp.quicksum(x[a] for a in outgoing_arcs[0]) <= max_vehicles,
        name="fleet_limit",
    )
    model.addConstr(
        gp.quicksum(x[a] for a in outgoing_arcs[0])
        == gp.quicksum(x[a] for a in incoming_arcs[0]),
        name="depot_balance",
    )

    for i, j in allowed_arcs:
        if i != 0 and j != 0:
            model.addConstr(
                load[j]
                >= load[i]
                + float(nodes[j]["demand"])
                - float(capacity) * (1 - x[i, j]),
                name=f"capacity_{i}_{j}",
            )

            model.addConstr(
                order[j] >= order[i] + 1.0 - float(n) * (1 - x[i, j]),
                name=f"order_link_{i}_{j}",
            )

            big_m = max(
                0.0,
                float(nodes[i]["time_window_end"])
                + float(nodes[i]["service_time"])
                + float(dist[i][j])
                - float(nodes[j]["time_window_start"]),
            )
            model.addConstr(
                visit_time[j]
                >= visit_time[i]
                + float(nodes[i]["service_time"])
                + float(dist[i][j])
                - big_m * (1 - x[i, j]),
                name=f"time_{i}_{j}",
            )

        elif i == 0 and j != 0:
            big_m = max(
                0.0,
                depot_start
                + depot_service
                + float(dist[0][j])
                - float(nodes[j]["time_window_start"]),
            )
            model.addConstr(
                visit_time[j]
                >= depot_start
                + depot_service
                + float(dist[0][j])
                - big_m * (1 - x[0, j]),
                name=f"time_depot_{j}",
            )

        elif i != 0 and j == 0:
            big_m = max(
                0.0,
                float(nodes[i]["time_window_end"])
                + float(nodes[i]["service_time"])
                + float(dist[i][0])
                - depot_end,
            )
            model.addConstr(
                visit_time[i]
                + float(nodes[i]["service_time"])
                + float(dist[i][0])
                <= depot_end + big_m * (1 - x[i, 0]),
                name=f"return_{i}",
            )

        sid = edge_to_set.get((min(i, j), max(i, j)))
        if sid is not None:
            model.addConstr(x[i, j] <= y[sid], name=f"access_{sid}_{i}_{j}")

    travel_objective = gp.quicksum(
        float(dist[i][j]) * x[i, j] for i, j in allowed_arcs
    )
    fixed_objective = gp.quicksum(access_cost[sid] * y[sid] for sid in edge_set_ids)
    model.setObjective(travel_objective + fixed_objective, GRB.MINIMIZE)

    # Supply the constructive solution as a MIP start.
    if initial_routes is not None:
        selected_start = set()
        used_start_sets = required_edge_sets(initial_routes, edge_to_set)

        for route in initial_routes:
            selected_start.update(route_edges(route))

        for arc in allowed_arcs:
            x[arc].Start = 1.0 if arc in selected_start else 0.0

        for sid in edge_set_ids:
            y[sid].Start = 1.0 if sid in used_start_sets else 0.0

        for i in customers:
            visit_time[i].Start = float(nodes[i]["time_window_start"])
            load[i].Start = float(nodes[i]["demand"])
            order[i].Start = 1.0

        for route in initial_routes:
            feasible, visits, _ = route_schedule(route, nodes, dist)
            if feasible:
                cumulative = 0
                for position, customer in enumerate(route, start=1):
                    cumulative += int(nodes[customer]["demand"])
                    visit_time[customer].Start = visits[customer]
                    load[customer].Start = cumulative
                    order[customer].Start = position

    remaining = absolute_deadline - time.monotonic()
    if remaining > 0.0:
        model.Params.TimeLimit = max(0.01, remaining)

        x_items = list(x.items())

        def incumbent_callback(cb_model, where):
            nonlocal best_solution, best_objective

            if where != GRB.Callback.MIPSOL:
                return

            try:
                values = cb_model.cbGetSolution([var for _, var in x_items])
                selected = [
                    arc for (arc, _), value in zip(x_items, values) if value > 0.5
                ]
                routes = extract_routes_from_selected(selected, n, max_vehicles)
                if routes is None:
                    return

                solution = build_solution(
                    routes, nodes, dist, edge_to_set, access_cost
                )
                if solution is None:
                    return

                objective = float(solution["objective_value"])
                if objective < best_objective - 1e-7:
                    best_objective = objective
                    best_solution = solution
                    if logger:
                        logger.log_solution(objective, solution)
            except (gp.GurobiError, ValueError, KeyError, TypeError):
                return

        model.optimize(incumbent_callback)

        if model.SolCount > 0:
            selected = [arc for arc in allowed_arcs if x[arc].X > 0.5]
            routes = extract_routes_from_selected(selected, n, max_vehicles)
            if routes is not None:
                solution = build_solution(
                    routes, nodes, dist, edge_to_set, access_cost
                )
                if solution is not None:
                    objective = float(solution["objective_value"])
                    if objective < best_objective - 1e-7:
                        best_objective = objective
                        best_solution = solution
                        if logger:
                            logger.log_solution(objective, solution)

    if best_solution is None:
        # Last constructive attempt, primarily for exceptionally short limits.
        fallback_deadline = time.monotonic() + 1.0
        fallback_routes = greedy_initial_solutions(
            n,
            max_vehicles,
            capacity,
            nodes,
            dist,
            edge_to_set,
            access_cost,
            fallback_deadline,
        )
        if fallback_routes is not None:
            best_solution = build_solution(
                fallback_routes, nodes, dist, edge_to_set, access_cost
            )

    if best_solution is None:
        raise RuntimeError("No feasible solution was found for the supplied instance.")

    output_directory = os.path.dirname(os.path.abspath(args.solution_path))
    os.makedirs(output_directory, exist_ok=True)
    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(best_solution, f, indent=2, sort_keys=False)


if __name__ == "__main__":
    main()
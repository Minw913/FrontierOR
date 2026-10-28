import argparse
import json
import math
import random
import time

import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


EPS = 1e-7


def euclidean(a, b):
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


def route_cost(route, distance):
    return sum(distance[route[i], route[i + 1]] for i in range(len(route) - 1))


def schedule_route(route, nodes, distance, pickup_by_user, dropoff_by_user,
                   maximum_ride_time, maximum_route_duration, capacity):
    """Find a feasible schedule for a fixed route using difference constraints."""
    load = 0
    for node_id in route:
        load += int(nodes[node_id]["load"])
        if load < 0 or load > capacity:
            return None

    position = {node_id: idx for idx, node_id in enumerate(route)}
    for user_id, pickup in pickup_by_user.items():
        dropoff = dropoff_by_user[user_id]
        if pickup in position or dropoff in position:
            if pickup not in position or dropoff not in position:
                return None
            if position[pickup] >= position[dropoff]:
                return None

    m = len(route)
    times = [float(nodes[node_id]["earliest_time"]) for node_id in route]
    latest = [float(nodes[node_id]["latest_time"]) for node_id in route]

    # Constraints have the form t[v] >= t[u] + weight.
    edges = []
    for idx in range(m - 1):
        i, j = route[idx], route[idx + 1]
        weight = float(nodes[i]["service_duration"]) + distance[i, j]
        edges.append((idx, idx + 1, weight))

    for user_id, pickup in pickup_by_user.items():
        dropoff = dropoff_by_user[user_id]
        if pickup in position:
            p = position[pickup]
            d = position[dropoff]
            # t_drop - (t_pickup + service_pickup) <= maximum ride time
            edges.append((
                d,
                p,
                -float(nodes[pickup]["service_duration"]) - maximum_ride_time
            ))

    # t_end - t_start <= maximum route duration.
    edges.append((m - 1, 0, -maximum_route_duration))

    # Longest-path relaxation. A positive cycle means infeasibility.
    for iteration in range(m + 1):
        changed = False
        for u, v, weight in edges:
            candidate = times[u] + weight
            if candidate > times[v] + EPS:
                times[v] = candidate
                if times[v] > latest[v] + 1e-6:
                    return None
                changed = True
        if not changed:
            break
        if iteration == m:
            return None

    for idx in range(m):
        if times[idx] < float(nodes[route[idx]]["earliest_time"]) - 1e-6:
            return None
        if times[idx] > latest[idx] + 1e-6:
            return None

    if times[-1] - times[0] > maximum_route_duration + 1e-6:
        return None

    for user_id, pickup in pickup_by_user.items():
        dropoff = dropoff_by_user[user_id]
        if pickup not in position:
            continue
        ride = (
            times[position[dropoff]]
            - times[position[pickup]]
            - float(nodes[pickup]["service_duration"])
        )
        direct = distance[pickup, dropoff]
        if ride < direct - 1e-5 or ride > maximum_ride_time + 1e-5:
            return None

    return times


def build_solution_from_routes(routes, schedules, nodes, distance,
                               pickup_by_user, dropoff_by_user,
                               start_depot, end_depot):
    objective = sum(route_cost(route, distance) for route in routes)
    service_times = {}
    ride_times = {}

    node_times = {}
    for vehicle, route in enumerate(routes):
        times = schedules[vehicle]
        service_times[f"depot_start_{vehicle}"] = float(times[0])
        service_times[f"depot_end_{vehicle}"] = float(times[-1])
        for idx, node_id in enumerate(route[1:-1], start=1):
            node_times[node_id] = float(times[idx])
            service_times[str(node_id)] = float(times[idx])

    for user_id, pickup in pickup_by_user.items():
        dropoff = dropoff_by_user[user_id]
        ride_times[str(user_id)] = float(
            node_times[dropoff]
            - node_times[pickup]
            - float(nodes[pickup]["service_duration"])
        )

    return {
        "objective_value": float(objective),
        "routes": {
            str(vehicle): [int(node_id) for node_id in route]
            for vehicle, route in enumerate(routes)
        },
        "service_times": service_times,
        "ride_times": ride_times,
    }


def greedy_insertion(instance, nodes, distance, pickup_by_user, dropoff_by_user,
                     deadline, logger):
    num_vehicles = int(instance["num_vehicles"])
    capacity = int(instance["vehicle_capacity"])
    maximum_ride_time = float(instance["maximum_ride_time"])
    maximum_route_duration = float(instance["maximum_route_duration"])
    end_depot = int(instance["num_nodes"]) - 1

    user_ids = list(pickup_by_user)
    best_solution = None
    best_routes = None
    best_schedules = None
    best_objective = float("inf")

    def priority(user_id):
        p = pickup_by_user[user_id]
        d = dropoff_by_user[user_id]
        return (
            min(float(nodes[p]["latest_time"]), float(nodes[d]["latest_time"])),
            float(nodes[p]["earliest_time"]),
        )

    base_order = sorted(user_ids, key=priority)
    rng = random.Random(0)
    attempt = 0

    while time.monotonic() < deadline:
        if attempt == 0:
            order = base_order[:]
        elif attempt == 1:
            order = sorted(
                user_ids,
                key=lambda u: float(nodes[pickup_by_user[u]]["latest_time"])
            )
        elif attempt == 2:
            order = sorted(
                user_ids,
                key=lambda u: float(nodes[dropoff_by_user[u]]["latest_time"])
            )
        else:
            order = base_order[:]
            rng.shuffle(order)

        routes = [[0, end_depot] for _ in range(num_vehicles)]
        schedules = []
        valid_empty = True
        for route in routes:
            schedule = schedule_route(
                route, nodes, distance, pickup_by_user, dropoff_by_user,
                maximum_ride_time, maximum_route_duration, capacity
            )
            if schedule is None:
                valid_empty = False
                break
            schedules.append(schedule)
        if not valid_empty:
            return None, None, None

        complete = True
        for user_id in order:
            if time.monotonic() >= deadline:
                complete = False
                break

            pickup = pickup_by_user[user_id]
            dropoff = dropoff_by_user[user_id]
            chosen = None

            for vehicle in range(num_vehicles):
                route = routes[vehicle]
                old_cost = route_cost(route, distance)
                length = len(route)

                for pickup_pos in range(1, length):
                    for dropoff_pos in range(pickup_pos + 1, length + 1):
                        candidate = route[:]
                        candidate.insert(pickup_pos, pickup)
                        candidate.insert(dropoff_pos, dropoff)

                        candidate_schedule = schedule_route(
                            candidate, nodes, distance,
                            pickup_by_user, dropoff_by_user,
                            maximum_ride_time, maximum_route_duration, capacity
                        )
                        if candidate_schedule is None:
                            continue

                        increase = route_cost(candidate, distance) - old_cost
                        value = (increase, route_cost(candidate, distance))
                        if chosen is None or value < chosen[0]:
                            chosen = (
                                value, vehicle, candidate, candidate_schedule
                            )

                    if time.monotonic() >= deadline:
                        break
                if time.monotonic() >= deadline:
                    break

            if chosen is None:
                complete = False
                break

            _, vehicle, candidate, candidate_schedule = chosen
            routes[vehicle] = candidate
            schedules[vehicle] = candidate_schedule

        if complete:
            objective = sum(route_cost(route, distance) for route in routes)
            if objective < best_objective - 1e-7:
                solution = build_solution_from_routes(
                    routes, schedules, nodes, distance,
                    pickup_by_user, dropoff_by_user, 0, end_depot
                )
                best_objective = objective
                best_solution = solution
                best_routes = [route[:] for route in routes]
                best_schedules = [schedule[:] for schedule in schedules]
                if logger:
                    logger.log_solution(objective, solution)

        attempt += 1
        if attempt >= 20:
            break

    return best_solution, best_routes, best_schedules


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    start_clock = time.monotonic()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as file:
        instance = json.load(file)

    num_vehicles = int(instance["num_vehicles"])
    num_nodes = int(instance["num_nodes"])
    capacity = int(instance["vehicle_capacity"])
    maximum_ride_time = float(instance["maximum_ride_time"])
    maximum_route_duration = float(instance["maximum_route_duration"])

    node_list = instance["nodes"]
    nodes = {int(node["node_id"]): node for node in node_list}
    node_ids = sorted(nodes)
    start_depot = next(
        node_id for node_id in node_ids
        if nodes[node_id]["node_type"] == "origin_depot"
    )
    end_depot = next(
        node_id for node_id in node_ids
        if nodes[node_id]["node_type"] == "destination_depot"
    )
    service_nodes = [
        node_id for node_id in node_ids
        if nodes[node_id]["node_type"] in ("pickup", "dropoff")
    ]
    pickups = [
        node_id for node_id in service_nodes
        if nodes[node_id]["node_type"] == "pickup"
    ]

    pickup_by_user = {}
    dropoff_by_user = {}
    for node_id in service_nodes:
        user_id = int(nodes[node_id]["user_id"])
        if nodes[node_id]["node_type"] == "pickup":
            pickup_by_user[user_id] = node_id
        else:
            dropoff_by_user[user_id] = node_id

    distance = {}
    for i in node_ids:
        for j in node_ids:
            distance[i, j] = euclidean(nodes[i], nodes[j])

    total_deadline = start_clock + max(0, args.time_limit)
    heuristic_budget = min(
        2.0,
        max(0.05, 0.12 * max(1, args.time_limit))
    )
    heuristic_deadline = min(total_deadline, time.monotonic() + heuristic_budget)

    best_solution, warm_routes, warm_schedules = greedy_insertion(
        instance, nodes, distance, pickup_by_user, dropoff_by_user,
        heuristic_deadline, logger
    )
    best_objective = (
        float(best_solution["objective_value"])
        if best_solution is not None else float("inf")
    )

    model = gp.Model("dial_a_ride")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1

    # Remove arcs that are structurally or temporally impossible.
    arcs = []
    for i in node_ids:
        if i == end_depot:
            continue
        for j in node_ids:
            if j == start_depot or i == j:
                continue
            if i == start_depot and nodes[j]["node_type"] == "dropoff":
                continue
            if j == end_depot and nodes[i]["node_type"] == "pickup":
                continue
            if (
                nodes[i]["node_type"] == "dropoff"
                and nodes[j]["node_type"] == "pickup"
                and int(nodes[i].get("user_id", -1))
                == int(nodes[j].get("user_id", -2))
            ):
                continue

            earliest_arrival = (
                float(nodes[i]["earliest_time"])
                + float(nodes[i]["service_duration"])
                + distance[i, j]
            )
            if earliest_arrival > float(nodes[j]["latest_time"]) + 1e-7:
                continue
            arcs.append((i, j))

    outgoing = {i: [] for i in node_ids}
    incoming = {i: [] for i in node_ids}
    for i, j in arcs:
        outgoing[i].append(j)
        incoming[j].append(i)

    x = {}
    for k in range(num_vehicles):
        for i, j in arcs:
            x[k, i, j] = model.addVar(
                vtype=GRB.BINARY, name=f"x_{k}_{i}_{j}"
            )

    y = {}
    for k in range(num_vehicles):
        for i in service_nodes:
            y[k, i] = model.addVar(vtype=GRB.BINARY, name=f"y_{k}_{i}")

    service_time = {}
    load_after = {}
    order = {}
    for k in range(num_vehicles):
        for i in node_ids:
            service_time[k, i] = model.addVar(
                lb=float(nodes[i]["earliest_time"]),
                ub=float(nodes[i]["latest_time"]),
                vtype=GRB.CONTINUOUS,
                name=f"t_{k}_{i}",
            )

            if i in (start_depot, end_depot):
                lower_load = upper_load = 0.0
            else:
                node_load = int(nodes[i]["load"])
                lower_load = max(0, node_load)
                upper_load = min(capacity, capacity + node_load)

            load_after[k, i] = model.addVar(
                lb=lower_load,
                ub=upper_load,
                vtype=GRB.CONTINUOUS,
                name=f"q_{k}_{i}",
            )

        for i in service_nodes:
            order[k, i] = model.addVar(
                lb=0.0, ub=float(len(service_nodes)),
                vtype=GRB.CONTINUOUS,
                name=f"ord_{k}_{i}",
            )

    model.update()

    for k in range(num_vehicles):
        model.addConstr(
            gp.quicksum(x[k, start_depot, j] for j in outgoing[start_depot]) == 1
        )
        model.addConstr(
            gp.quicksum(x[k, i, end_depot] for i in incoming[end_depot]) == 1
        )

        for i in service_nodes:
            model.addConstr(
                gp.quicksum(x[k, h, i] for h in incoming[i]) == y[k, i]
            )
            model.addConstr(
                gp.quicksum(x[k, i, j] for j in outgoing[i]) == y[k, i]
            )

        model.addConstr(
            service_time[k, end_depot] - service_time[k, start_depot]
            <= maximum_route_duration
        )

    for pickup in pickups:
        user_id = int(nodes[pickup]["user_id"])
        dropoff = dropoff_by_user[user_id]
        model.addConstr(
            gp.quicksum(y[k, pickup] for k in range(num_vehicles)) == 1
        )
        for k in range(num_vehicles):
            model.addConstr(y[k, pickup] == y[k, dropoff])

            pickup_service = float(nodes[pickup]["service_duration"])
            direct = distance[pickup, dropoff]

            model.addGenConstrIndicator(
                y[k, pickup], True,
                service_time[k, dropoff] - service_time[k, pickup]
                >= pickup_service + direct
            )
            model.addGenConstrIndicator(
                y[k, pickup], True,
                service_time[k, dropoff] - service_time[k, pickup]
                <= pickup_service + maximum_ride_time
            )

    for k in range(num_vehicles):
        for i, j in arcs:
            travel_and_service = (
                float(nodes[i]["service_duration"]) + distance[i, j]
            )
            model.addGenConstrIndicator(
                x[k, i, j], True,
                service_time[k, j] - service_time[k, i] >= travel_and_service
            )
            model.addGenConstrIndicator(
                x[k, i, j], True,
                load_after[k, j] - load_after[k, i]
                == int(nodes[j]["load"])
            )

            if i in service_nodes and j in service_nodes:
                model.addGenConstrIndicator(
                    x[k, i, j], True,
                    order[k, j] - order[k, i] >= 1
                )

    # Symmetry breaking: unused vehicles are placed after used vehicles.
    direct_arc = (start_depot, end_depot)
    if direct_arc in arcs:
        for k in range(num_vehicles - 1):
            model.addConstr(
                x[k, start_depot, end_depot]
                <= x[k + 1, start_depot, end_depot]
            )

    model.setObjective(
        gp.quicksum(
            distance[i, j] * x[k, i, j]
            for k in range(num_vehicles)
            for i, j in arcs
        ),
        GRB.MINIMIZE
    )

    if warm_routes is not None:
        warm_arc_sets = []
        for route in warm_routes:
            warm_arc_sets.append(set(zip(route[:-1], route[1:])))

        for k in range(num_vehicles):
            visited = set(warm_routes[k][1:-1])
            time_map = {
                node_id: warm_schedules[k][idx]
                for idx, node_id in enumerate(warm_routes[k])
            }

            for i, j in arcs:
                x[k, i, j].Start = (
                    1.0 if (i, j) in warm_arc_sets[k] else 0.0
                )
            for i in service_nodes:
                y[k, i].Start = 1.0 if i in visited else 0.0
            for i in node_ids:
                if i in time_map:
                    service_time[k, i].Start = float(time_map[i])

            occupancy = 0
            for node_id in warm_routes[k]:
                occupancy += int(nodes[node_id]["load"])
                load_after[k, node_id].Start = float(occupancy)

    x_items = list(x.items())

    def extract_solution(value_getter):
        selected = {}
        for key, variable in x_items:
            if value_getter(variable) > 0.5:
                selected[key] = True

        routes = []
        schedules = []
        for k in range(num_vehicles):
            route = [start_depot]
            current = start_depot
            seen = {start_depot}

            while current != end_depot:
                next_node = None
                for j in outgoing[current]:
                    if selected.get((k, current, j), False):
                        next_node = j
                        break
                if next_node is None:
                    raise RuntimeError("Could not reconstruct route")
                route.append(next_node)
                if next_node != end_depot and next_node in seen:
                    raise RuntimeError("Cycle encountered during reconstruction")
                seen.add(next_node)
                current = next_node

            routes.append(route)
            schedules.append([
                float(value_getter(service_time[k, node_id]))
                for node_id in route
            ])

        return build_solution_from_routes(
            routes, schedules, nodes, distance,
            pickup_by_user, dropoff_by_user, start_depot, end_depot
        )

    callback_best = [best_objective]

    def callback(cb_model, where):
        if where != GRB.Callback.MIPSOL:
            return
        try:
            incumbent_obj = cb_model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if incumbent_obj >= callback_best[0] - 1e-7:
                return
            solution = extract_solution(cb_model.cbGetSolution)
            objective = float(solution["objective_value"])
            if objective < callback_best[0] - 1e-7:
                callback_best[0] = objective
                if logger:
                    logger.log_solution(objective, solution)
        except Exception:
            # Logging must not interrupt optimization.
            pass

    remaining = max(0.01, total_deadline - time.monotonic())
    model.Params.TimeLimit = remaining

    try:
        model.optimize(callback)
    except gp.GurobiError:
        if best_solution is None:
            raise

    if model.SolCount > 0:
        mip_solution = extract_solution(lambda variable: variable.X)
        mip_objective = float(mip_solution["objective_value"])
        if mip_objective < best_objective - 1e-7 or best_solution is None:
            best_solution = mip_solution
            best_objective = mip_objective
            if logger and mip_objective < callback_best[0] - 1e-7:
                logger.log_solution(mip_objective, mip_solution)

    if best_solution is None:
        raise RuntimeError("No feasible solution was found within the time limit")

    with open(args.solution_path, "w", encoding="utf-8") as file:
        json.dump(best_solution, file, indent=2)


if __name__ == "__main__":
    main()
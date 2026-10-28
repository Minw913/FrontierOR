import argparse
import json
import math
import random
import time

from solution_logger import SolutionLogger


EPS = 1e-7


def euclidean(node_a, node_b):
    return math.hypot(
        float(node_a["x"]) - float(node_b["x"]),
        float(node_a["y"]) - float(node_b["y"]),
    )


def route_cost(route, distances):
    return sum(
        distances[route[index], route[index + 1]]
        for index in range(len(route) - 1)
    )


def compute_schedule(
    route,
    nodes,
    distances,
    pickup_by_user,
    dropoff_by_user,
    maximum_ride_time,
    maximum_route_duration,
    capacity,
):
    """
    Finds the earliest feasible schedule for a fixed route.

    All timing constraints are represented as difference constraints of the
    form time[v] >= time[u] + weight and solved by longest-path relaxation.
    """
    if len(route) < 2:
        return None

    # Check load feasibility and calculate the exact occupancy.
    occupancy = 0
    for node_id in route:
        occupancy += int(nodes[node_id]["load"])
        node_load = int(nodes[node_id]["load"])
        lower_bound = max(0, node_load)
        upper_bound = min(capacity, capacity + node_load)

        if occupancy < lower_bound or occupancy > upper_bound:
            return None

    if occupancy != 0:
        return None

    positions = {node_id: index for index, node_id in enumerate(route)}

    # Check pickup/drop-off pairing and precedence.
    for user_id, pickup in pickup_by_user.items():
        dropoff = dropoff_by_user[user_id]
        pickup_present = pickup in positions
        dropoff_present = dropoff in positions

        if pickup_present != dropoff_present:
            return None
        if pickup_present and positions[pickup] >= positions[dropoff]:
            return None

    route_length = len(route)
    times = [
        float(nodes[node_id]["earliest_time"])
        for node_id in route
    ]
    latest = [
        float(nodes[node_id]["latest_time"])
        for node_id in route
    ]

    # Each tuple represents t[v] >= t[u] + weight.
    constraints = []

    # Route travel and service constraints.
    for index in range(route_length - 1):
        current_node = route[index]
        next_node = route[index + 1]
        constraints.append(
            (
                index,
                index + 1,
                float(nodes[current_node]["service_duration"])
                + distances[current_node, next_node],
            )
        )

    # Maximum ride-time constraints:
    # t_drop - t_pickup - service_pickup <= maximum_ride_time.
    for user_id, pickup in pickup_by_user.items():
        if pickup not in positions:
            continue

        dropoff = dropoff_by_user[user_id]
        pickup_position = positions[pickup]
        dropoff_position = positions[dropoff]

        constraints.append(
            (
                dropoff_position,
                pickup_position,
                -float(nodes[pickup]["service_duration"])
                - maximum_ride_time,
            )
        )

    # Maximum route duration:
    # t_end - t_start <= maximum_route_duration.
    constraints.append(
        (route_length - 1, 0, -maximum_route_duration)
    )

    # Bellman-Ford-style longest-path relaxation.
    for iteration in range(route_length):
        changed = False

        for source, target, weight in constraints:
            candidate = times[source] + weight
            if candidate > times[target] + EPS:
                times[target] = candidate
                changed = True

                if times[target] > latest[target] + 1e-6:
                    return None

        if not changed:
            break

        if iteration == route_length - 1:
            # A positive cycle means the difference constraints are infeasible.
            return None

    for index, node_id in enumerate(route):
        earliest = float(nodes[node_id]["earliest_time"])
        latest_time = float(nodes[node_id]["latest_time"])

        if times[index] < earliest - 1e-6:
            return None
        if times[index] > latest_time + 1e-6:
            return None

    if times[-1] - times[0] > maximum_route_duration + 1e-6:
        return None

    # Final ride-time validation.
    for user_id, pickup in pickup_by_user.items():
        if pickup not in positions:
            continue

        dropoff = dropoff_by_user[user_id]
        ride_time = (
            times[positions[dropoff]]
            - times[positions[pickup]]
            - float(nodes[pickup]["service_duration"])
        )

        direct_time = distances[pickup, dropoff]
        if ride_time < direct_time - 1e-5:
            return None
        if ride_time > maximum_ride_time + 1e-5:
            return None

    return times


def make_solution(
    routes,
    schedules,
    nodes,
    distances,
    pickup_by_user,
    dropoff_by_user,
):
    objective_value = sum(
        route_cost(route, distances) for route in routes
    )

    service_times = {}
    node_service_times = {}

    for vehicle, route in enumerate(routes):
        schedule = schedules[vehicle]

        service_times[f"depot_start_{vehicle}"] = float(schedule[0])
        service_times[f"depot_end_{vehicle}"] = float(schedule[-1])

        for index in range(1, len(route) - 1):
            node_id = route[index]
            service_time = float(schedule[index])
            service_times[str(node_id)] = service_time
            node_service_times[node_id] = service_time

    ride_times = {}
    for user_id, pickup in pickup_by_user.items():
        dropoff = dropoff_by_user[user_id]
        ride_times[str(user_id)] = float(
            node_service_times[dropoff]
            - node_service_times[pickup]
            - float(nodes[pickup]["service_duration"])
        )

    return {
        "objective_value": float(objective_value),
        "routes": {
            str(vehicle): [int(node_id) for node_id in route]
            for vehicle, route in enumerate(routes)
        },
        "service_times": service_times,
        "ride_times": ride_times,
    }


def enumerate_request_insertions(
    routes,
    user_id,
    nodes,
    distances,
    pickup_by_user,
    dropoff_by_user,
    maximum_ride_time,
    maximum_route_duration,
    capacity,
    deadline,
):
    pickup = pickup_by_user[user_id]
    dropoff = dropoff_by_user[user_id]
    candidates = []

    for vehicle, route in enumerate(routes):
        if time.monotonic() >= deadline:
            break

        old_cost = route_cost(route, distances)
        route_size = len(route)

        # First rank insertion positions by their inexpensive routing delta.
        position_candidates = []
        for pickup_position in range(1, route_size):
            for dropoff_position in range(
                pickup_position + 1, route_size + 1
            ):
                candidate_route = route[:]
                candidate_route.insert(pickup_position, pickup)
                candidate_route.insert(dropoff_position, dropoff)

                delta = (
                    route_cost(candidate_route, distances) - old_cost
                )
                position_candidates.append(
                    (delta, candidate_route)
                )

        position_candidates.sort(key=lambda item: item[0])

        for delta, candidate_route in position_candidates:
            if time.monotonic() >= deadline:
                break

            schedule = compute_schedule(
                candidate_route,
                nodes,
                distances,
                pickup_by_user,
                dropoff_by_user,
                maximum_ride_time,
                maximum_route_duration,
                capacity,
            )

            if schedule is not None:
                candidates.append(
                    (delta, vehicle, candidate_route, schedule)
                )

    candidates.sort(key=lambda item: item[0])
    return candidates


def construct_solution(
    user_order,
    num_vehicles,
    start_depot,
    end_depot,
    nodes,
    distances,
    pickup_by_user,
    dropoff_by_user,
    maximum_ride_time,
    maximum_route_duration,
    capacity,
    deadline,
    rng,
    randomized,
):
    routes = [
        [start_depot, end_depot]
        for _ in range(num_vehicles)
    ]
    schedules = []

    for route in routes:
        schedule = compute_schedule(
            route,
            nodes,
            distances,
            pickup_by_user,
            dropoff_by_user,
            maximum_ride_time,
            maximum_route_duration,
            capacity,
        )
        if schedule is None:
            return None, None
        schedules.append(schedule)

    for user_id in user_order:
        if time.monotonic() >= deadline:
            return None, None

        candidates = enumerate_request_insertions(
            routes,
            user_id,
            nodes,
            distances,
            pickup_by_user,
            dropoff_by_user,
            maximum_ride_time,
            maximum_route_duration,
            capacity,
            deadline,
        )

        if not candidates:
            return None, None

        if randomized and len(candidates) > 1:
            # Restricted candidate list gives diverse but still inexpensive
            # constructions.
            best_delta = candidates[0][0]
            threshold = best_delta + max(1.0, abs(best_delta) * 0.25)
            restricted = [
                candidate
                for candidate in candidates[:8]
                if candidate[0] <= threshold + EPS
            ]
            selected = rng.choice(restricted)
        else:
            selected = candidates[0]

        _, vehicle, candidate_route, candidate_schedule = selected
        routes[vehicle] = candidate_route
        schedules[vehicle] = candidate_schedule

    return routes, schedules


def improve_by_request_relocation(
    routes,
    schedules,
    user_ids,
    nodes,
    distances,
    pickup_by_user,
    dropoff_by_user,
    maximum_ride_time,
    maximum_route_duration,
    capacity,
    deadline,
    rng,
):
    routes = [route[:] for route in routes]
    schedules = [schedule[:] for schedule in schedules]
    current_cost = sum(route_cost(route, distances) for route in routes)

    improved = True
    while improved and time.monotonic() < deadline:
        improved = False
        ordered_users = user_ids[:]
        rng.shuffle(ordered_users)

        for user_id in ordered_users:
            if time.monotonic() >= deadline:
                break

            pickup = pickup_by_user[user_id]
            dropoff = dropoff_by_user[user_id]

            source_vehicle = None
            for vehicle, route in enumerate(routes):
                if pickup in route:
                    source_vehicle = vehicle
                    break

            if source_vehicle is None:
                continue

            old_source_route = routes[source_vehicle]
            reduced_source_route = [
                node_id
                for node_id in old_source_route
                if node_id != pickup and node_id != dropoff
            ]

            reduced_source_schedule = compute_schedule(
                reduced_source_route,
                nodes,
                distances,
                pickup_by_user,
                dropoff_by_user,
                maximum_ride_time,
                maximum_route_duration,
                capacity,
            )
            if reduced_source_schedule is None:
                continue

            temporary_routes = [route[:] for route in routes]
            temporary_routes[source_vehicle] = reduced_source_route

            removed_cost = (
                current_cost
                - route_cost(old_source_route, distances)
                + route_cost(reduced_source_route, distances)
            )

            best_move = None
            best_total_cost = current_cost - 1e-7

            candidates = enumerate_request_insertions(
                temporary_routes,
                user_id,
                nodes,
                distances,
                pickup_by_user,
                dropoff_by_user,
                maximum_ride_time,
                maximum_route_duration,
                capacity,
                deadline,
            )

            for delta, target_vehicle, candidate_route, candidate_schedule in candidates:
                total_cost = removed_cost + delta
                if total_cost < best_total_cost:
                    best_total_cost = total_cost
                    best_move = (
                        target_vehicle,
                        candidate_route,
                        candidate_schedule,
                        reduced_source_schedule,
                    )

            if best_move is None:
                continue

            (
                target_vehicle,
                candidate_route,
                candidate_schedule,
                reduced_source_schedule,
            ) = best_move

            routes[source_vehicle] = reduced_source_route
            schedules[source_vehicle] = reduced_source_schedule

            routes[target_vehicle] = candidate_route
            schedules[target_vehicle] = candidate_schedule

            current_cost = best_total_cost
            improved = True

    return routes, schedules


def improve_single_node_positions(
    routes,
    schedules,
    nodes,
    distances,
    pickup_by_user,
    dropoff_by_user,
    maximum_ride_time,
    maximum_route_duration,
    capacity,
    deadline,
):
    routes = [route[:] for route in routes]
    schedules = [schedule[:] for schedule in schedules]

    improved = True
    while improved and time.monotonic() < deadline:
        improved = False

        for vehicle in range(len(routes)):
            if time.monotonic() >= deadline:
                break

            route = routes[vehicle]
            current_route_cost = route_cost(route, distances)
            best_route = route
            best_schedule = schedules[vehicle]
            best_cost = current_route_cost

            for old_position in range(1, len(route) - 1):
                if time.monotonic() >= deadline:
                    break

                node_id = route[old_position]
                reduced_route = (
                    route[:old_position] + route[old_position + 1:]
                )

                for new_position in range(1, len(reduced_route)):
                    if time.monotonic() >= deadline:
                        break

                    candidate_route = reduced_route[:]
                    candidate_route.insert(new_position, node_id)

                    if candidate_route == route:
                        continue

                    candidate_cost = route_cost(
                        candidate_route, distances
                    )
                    if candidate_cost >= best_cost - 1e-7:
                        continue

                    candidate_schedule = compute_schedule(
                        candidate_route,
                        nodes,
                        distances,
                        pickup_by_user,
                        dropoff_by_user,
                        maximum_ride_time,
                        maximum_route_duration,
                        capacity,
                    )

                    if candidate_schedule is not None:
                        best_cost = candidate_cost
                        best_route = candidate_route
                        best_schedule = candidate_schedule

            if best_cost < current_route_cost - 1e-7:
                routes[vehicle] = best_route
                schedules[vehicle] = best_schedule
                improved = True

    return routes, schedules


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    logger = (
        SolutionLogger(args.log_path, sense="minimize")
        if args.log_path
        else None
    )

    start_time = time.monotonic()
    deadline = start_time + max(0.01, float(args.time_limit))

    with open(args.instance_path, "r", encoding="utf-8") as input_file:
        instance = json.load(input_file)

    num_vehicles = int(instance["num_vehicles"])
    capacity = int(instance["vehicle_capacity"])
    maximum_ride_time = float(instance["maximum_ride_time"])
    maximum_route_duration = float(instance["maximum_route_duration"])

    nodes = {
        int(node["node_id"]): node
        for node in instance["nodes"]
    }
    node_ids = sorted(nodes)

    start_depot = next(
        node_id
        for node_id in node_ids
        if nodes[node_id]["node_type"] == "origin_depot"
    )
    end_depot = next(
        node_id
        for node_id in node_ids
        if nodes[node_id]["node_type"] == "destination_depot"
    )

    pickup_by_user = {}
    dropoff_by_user = {}

    for node_id in node_ids:
        node_type = nodes[node_id]["node_type"]
        if node_type not in ("pickup", "dropoff"):
            continue

        user_id = int(nodes[node_id]["user_id"])
        if node_type == "pickup":
            pickup_by_user[user_id] = node_id
        else:
            dropoff_by_user[user_id] = node_id

    user_ids = sorted(pickup_by_user)

    distances = {}
    for first_node in node_ids:
        for second_node in node_ids:
            distances[first_node, second_node] = euclidean(
                nodes[first_node], nodes[second_node]
            )

    rng = random.Random(0)

    def user_priority(user_id):
        pickup = pickup_by_user[user_id]
        dropoff = dropoff_by_user[user_id]
        return (
            min(
                float(nodes[pickup]["latest_time"]),
                float(nodes[dropoff]["latest_time"]),
            ),
            max(
                float(nodes[pickup]["earliest_time"]),
                float(nodes[dropoff]["earliest_time"]),
            ),
            distances[pickup, dropoff],
        )

    deterministic_orders = [
        sorted(user_ids, key=user_priority),
        sorted(
            user_ids,
            key=lambda user: float(
                nodes[pickup_by_user[user]]["latest_time"]
            ),
        ),
        sorted(
            user_ids,
            key=lambda user: float(
                nodes[dropoff_by_user[user]]["latest_time"]
            ),
        ),
        sorted(
            user_ids,
            key=lambda user: -distances[
                pickup_by_user[user], dropoff_by_user[user]
            ],
        ),
    ]

    best_solution = None
    best_routes = None
    best_schedules = None
    best_objective = float("inf")

    total_budget = max(0.01, float(args.time_limit))
    construction_deadline = min(
        deadline,
        start_time + max(0.01, total_budget * 0.65),
    )

    attempt = 0
    max_attempts = max(4, min(40, 4 + len(user_ids)))

    while (
        attempt < max_attempts
        and time.monotonic() < construction_deadline
    ):
        if attempt < len(deterministic_orders):
            order = deterministic_orders[attempt][:]
            randomized = attempt > 0
        else:
            order = deterministic_orders[0][:]
            rng.shuffle(order)
            randomized = True

        routes, schedules = construct_solution(
            order,
            num_vehicles,
            start_depot,
            end_depot,
            nodes,
            distances,
            pickup_by_user,
            dropoff_by_user,
            maximum_ride_time,
            maximum_route_duration,
            capacity,
            construction_deadline,
            rng,
            randomized,
        )

        if routes is not None:
            solution = make_solution(
                routes,
                schedules,
                nodes,
                distances,
                pickup_by_user,
                dropoff_by_user,
            )
            objective = float(solution["objective_value"])

            if objective < best_objective - 1e-7:
                best_objective = objective
                best_solution = solution
                best_routes = [route[:] for route in routes]
                best_schedules = [
                    schedule[:] for schedule in schedules
                ]

                if logger:
                    logger.log_solution(objective, solution)

        attempt += 1

    # If the initial construction budget was too short, use all remaining time
    # to continue looking for the first feasible solution.
    while best_solution is None and time.monotonic() < deadline:
        order = deterministic_orders[0][:]
        if attempt > 0:
            rng.shuffle(order)

        routes, schedules = construct_solution(
            order,
            num_vehicles,
            start_depot,
            end_depot,
            nodes,
            distances,
            pickup_by_user,
            dropoff_by_user,
            maximum_ride_time,
            maximum_route_duration,
            capacity,
            deadline,
            rng,
            attempt > 0,
        )

        if routes is not None:
            best_routes = routes
            best_schedules = schedules
            best_solution = make_solution(
                routes,
                schedules,
                nodes,
                distances,
                pickup_by_user,
                dropoff_by_user,
            )
            best_objective = float(best_solution["objective_value"])

            if logger:
                logger.log_solution(best_objective, best_solution)

        attempt += 1

    if best_solution is None:
        raise RuntimeError(
            "No feasible solution was found within the time limit."
        )

    if time.monotonic() < deadline:
        improved_routes, improved_schedules = improve_by_request_relocation(
            best_routes,
            best_schedules,
            user_ids,
            nodes,
            distances,
            pickup_by_user,
            dropoff_by_user,
            maximum_ride_time,
            maximum_route_duration,
            capacity,
            deadline,
            rng,
        )

        improved_solution = make_solution(
            improved_routes,
            improved_schedules,
            nodes,
            distances,
            pickup_by_user,
            dropoff_by_user,
        )
        improved_objective = float(
            improved_solution["objective_value"]
        )

        if improved_objective < best_objective - 1e-7:
            best_objective = improved_objective
            best_solution = improved_solution
            best_routes = improved_routes
            best_schedules = improved_schedules

            if logger:
                logger.log_solution(
                    improved_objective, improved_solution
                )

    if time.monotonic() < deadline:
        improved_routes, improved_schedules = improve_single_node_positions(
            best_routes,
            best_schedules,
            nodes,
            distances,
            pickup_by_user,
            dropoff_by_user,
            maximum_ride_time,
            maximum_route_duration,
            capacity,
            deadline,
        )

        improved_solution = make_solution(
            improved_routes,
            improved_schedules,
            nodes,
            distances,
            pickup_by_user,
            dropoff_by_user,
        )
        improved_objective = float(
            improved_solution["objective_value"]
        )

        if improved_objective < best_objective - 1e-7:
            best_solution = improved_solution
            best_objective = improved_objective

            if logger:
                logger.log_solution(
                    improved_objective, improved_solution
                )

    with open(args.solution_path, "w", encoding="utf-8") as output_file:
        json.dump(best_solution, output_file, indent=2)


if __name__ == "__main__":
    main()
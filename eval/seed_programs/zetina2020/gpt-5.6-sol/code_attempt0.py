import argparse
import json
import math
import os
import random
import time

from solution_logger import SolutionLogger


def tour_cost(tour, distances):
    n = len(tour)
    if n == 0:
        return 0.0
    return float(sum(distances[tour[i]][tour[(i + 1) % n]] for i in range(n)))


def decision_variables(tour, n):
    position = {city: step for step, city in enumerate(tour)}
    return {
        f"x_{city}_{step}": int(position[city] == step)
        for city in range(n)
        for step in range(n)
    }


def rotate_to_zero(tour):
    if not tour:
        return tour
    index = tour.index(0)
    return tour[index:] + tour[:index]


def solve_tsp(instance, deadline, logger):
    parameters = instance["parameters"]
    data = instance["data"]

    n = int(parameters["n_cities"])
    if n <= 0:
        raise ValueError("n_cities must be positive")

    raw_distances = data.get("distance_matrix")
    if raw_distances is None:
        coordinates = data["coordinates"]
        raw_distances = [
            [
                math.hypot(
                    coordinates[i][0] - coordinates[j][0],
                    coordinates[i][1] - coordinates[j][1],
                )
                for j in range(n)
            ]
            for i in range(n)
        ]

    if len(raw_distances) != n or any(len(row) != n for row in raw_distances):
        raise ValueError("distance_matrix must have shape [n_cities][n_cities]")

    distances = [[float(value) for value in row] for row in raw_distances]

    best_tour = list(range(n))
    best_cost = tour_cost(best_tour, distances)

    def log_incumbent():
        if logger:
            logger.log_solution(
                float(best_cost), decision_variables(best_tour, n)
            )

    def consider(candidate_tour, candidate_cost=None):
        nonlocal best_tour, best_cost
        if candidate_cost is None:
            candidate_cost = tour_cost(candidate_tour, distances)
        candidate_cost = float(candidate_cost)

        if candidate_cost < best_cost - 1e-12:
            best_tour = list(candidate_tour)
            best_cost = candidate_cost
            log_incumbent()
            return True
        return False

    # A feasible solution is available immediately, even for a zero time limit.
    log_incumbent()

    if n == 1 or time.monotonic() >= deadline:
        return best_cost, best_tour

    # Deterministic nearest-neighbor tours from several starting cities.
    if n <= 64:
        starts = list(range(n))
    else:
        starts = [0]
        starts.extend(
            sorted(
                {
                    int(round(i * (n - 1) / 31))
                    for i in range(32)
                }
            )
        )

    for start_city in starts:
        if time.monotonic() >= deadline:
            break

        unvisited = set(range(n))
        unvisited.remove(start_city)
        route = [start_city]

        while unvisited:
            current = route[-1]
            next_city = min(
                unvisited,
                key=lambda city: (distances[current][city], city),
            )
            route.append(next_city)
            unvisited.remove(next_city)

        consider(rotate_to_zero(route))

    def two_opt(route):
        """Best-improvement 2-opt for the symmetric distance matrix."""
        route = list(route)
        route_value = tour_cost(route, distances)

        while time.monotonic() < deadline:
            best_delta = -1e-12
            best_move = None
            checks = 0

            for i in range(1, n - 1):
                a = route[i - 1]
                b = route[i]

                for k in range(i + 1, n):
                    c = route[k]
                    d = route[(k + 1) % n]
                    delta = (
                        distances[a][c]
                        + distances[b][d]
                        - distances[a][b]
                        - distances[c][d]
                    )

                    if delta < best_delta:
                        best_delta = delta
                        best_move = (i, k)

                    checks += 1
                    if checks % 256 == 0 and time.monotonic() >= deadline:
                        return route, route_value

            if best_move is None:
                break

            i, k = best_move
            route[i : k + 1] = reversed(route[i : k + 1])
            route_value = tour_cost(route, distances)
            consider(route, route_value)

        return route, route_value

    if time.monotonic() < deadline:
        two_opt(best_tour)

    # Exact depth-first branch-and-bound is very effective for the benchmark's
    # four-city instances. Fixing city 0 removes rotationally equivalent tours.
    if n <= 13 and time.monotonic() < deadline:
        neighbor_order = [
            sorted(
                (city for city in range(n) if city != source),
                key=lambda city, source=source: (
                    distances[source][city],
                    city,
                ),
            )
            for source in range(n)
        ]

        full_remaining = ((1 << n) - 1) & ~1
        path = [0]
        stopped = False
        visited_nodes = 0

        def lower_bound(last_city, remaining_mask, current_cost):
            unvisited = []
            mask = remaining_mask
            while mask:
                low_bit = mask & -mask
                unvisited.append(low_bit.bit_length() - 1)
                mask ^= low_bit

            if not unvisited:
                return current_cost + distances[last_city][0]

            bound = current_cost + min(
                distances[last_city][city] for city in unvisited
            )

            # Every unvisited city must eventually have an outgoing edge either
            # to another unvisited city or back to city 0.
            for city in unvisited:
                minimum_outgoing = distances[city][0]
                for other in unvisited:
                    if other != city and distances[city][other] < minimum_outgoing:
                        minimum_outgoing = distances[city][other]
                bound += minimum_outgoing

            return bound

        def dfs(last_city, remaining_mask, current_cost):
            nonlocal stopped, visited_nodes

            visited_nodes += 1
            if visited_nodes % 64 == 0 and time.monotonic() >= deadline:
                stopped = True
                return
            if stopped:
                return

            if remaining_mask == 0:
                consider(path, current_cost + distances[last_city][0])
                return

            if lower_bound(last_city, remaining_mask, current_cost) >= best_cost:
                return

            for next_city in neighbor_order[last_city]:
                bit = 1 << next_city
                if not (remaining_mask & bit):
                    continue

                new_cost = current_cost + distances[last_city][next_city]
                if new_cost >= best_cost:
                    continue

                path.append(next_city)
                dfs(next_city, remaining_mask ^ bit, new_cost)
                path.pop()

                if stopped:
                    return

        dfs(0, full_remaining, 0.0)

    # For unexpectedly large schema-valid instances, use deterministic
    # randomized greedy restarts for the remaining time.
    elif time.monotonic() < deadline:
        rng = random.Random(0)

        while time.monotonic() < deadline:
            route = [0]
            unvisited = set(range(1, n))

            while unvisited:
                if time.monotonic() >= deadline:
                    break
                current = route[-1]
                nearest = sorted(
                    unvisited,
                    key=lambda city: (distances[current][city], city),
                )
                rcl_size = min(3, len(nearest))
                next_city = nearest[rng.randrange(rcl_size)]
                route.append(next_city)
                unvisited.remove(next_city)

            if unvisited:
                break

            consider(route)
            two_opt(route)

    return float(best_cost), best_tour


def main():
    program_start = time.monotonic()

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

    deadline = program_start + max(0, args.time_limit)

    with open(args.instance_path, "r", encoding="utf-8") as input_file:
        instance = json.load(input_file)

    objective_value, tour = solve_tsp(instance, deadline, logger)
    n = int(instance["parameters"]["n_cities"])

    output = {
        "objective_value": float(objective_value),
        "solution": decision_variables(tour, n),
        "tour": [int(city) for city in tour],
    }

    output_directory = os.path.dirname(os.path.abspath(args.solution_path))
    os.makedirs(output_directory, exist_ok=True)

    with open(args.solution_path, "w", encoding="utf-8") as output_file:
        json.dump(output, output_file, separators=(",", ":"))


if __name__ == "__main__":
    main()
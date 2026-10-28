import argparse
import json
import math
import os
import random
import time
from functools import lru_cache

from solution_logger import SolutionLogger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    start_time = time.monotonic()
    deadline = start_time + max(0, args.time_limit) - 0.03
    rng = random.Random(0)

    with open(args.instance_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    nodes = data["nodes"]
    node_by_id = {int(node["id"]): node for node in nodes}

    origins = [int(n["id"]) for n in nodes if n["type"] == "depot_origin"]
    destinations = [int(n["id"]) for n in nodes if n["type"] == "depot_destination"]

    if origins:
        origin = origins[0]
    else:
        origin = min(node_by_id)

    if destinations:
        destination = destinations[0]
    else:
        destination = origin

    num_vehicles = int(data["num_vehicles"])
    capacity = float(data["vehicle_capacity"])

    request_data = data["requests"]
    num_requests = len(request_data)

    pickups = []
    deliveries = []
    request_loads = []
    request_ids = []

    for req in request_data:
        pickups.append(int(req["pickup_node"]))
        deliveries.append(int(req["delivery_node"]))
        request_loads.append(float(req["load"]))
        request_ids.append(int(req["request_id"]))

    request_of_node = {}
    for r in range(num_requests):
        request_of_node[pickups[r]] = r
        request_of_node[deliveries[r]] = r

    coordinates = {}
    demands = {}
    service_times = {}
    windows = {}

    for node_id, node in node_by_id.items():
        coordinates[node_id] = (float(node["x"]), float(node["y"]))
        demands[node_id] = float(node.get("demand", 0))
        service_times[node_id] = float(node.get("service_time", 0))
        tw = node.get("time_window", [0, float(data.get("planning_horizon", 10**12))])
        windows[node_id] = (float(tw[0]), float(tw[1]))

    all_ids = list(node_by_id)
    index_of = {node_id: i for i, node_id in enumerate(all_ids)}
    n_nodes = len(all_ids)

    distance = [[0.0] * n_nodes for _ in range(n_nodes)]
    for i, ni in enumerate(all_ids):
        xi, yi = coordinates[ni]
        for j in range(i + 1, n_nodes):
            nj = all_ids[j]
            xj, yj = coordinates[nj]
            d = math.hypot(xi - xj, yi - yj)
            distance[i][j] = d
            distance[j][i] = d

    def dist(a, b):
        return distance[index_of[a]][index_of[b]]

    @lru_cache(maxsize=350000)
    def evaluate_route(route_tuple):
        """Return route distance if feasible, otherwise None.

        route_tuple contains service nodes only; depots are implicit.
        """
        current = origin
        current_time = windows[origin][0]
        current_load = 0.0
        total_distance = 0.0
        picked = set()

        for nxt in route_tuple:
            d = dist(current, nxt)
            total_distance += d
            arrival = current_time + service_times[current] + d
            early, late = windows[nxt]
            current_time = max(arrival, early)
            if current_time > late + 1e-8:
                return None

            r = request_of_node.get(nxt)
            if r is not None:
                if nxt == pickups[r]:
                    if r in picked:
                        return None
                    picked.add(r)
                else:
                    if r not in picked:
                        return None

            current_load += demands[nxt]
            if current_load < -1e-8 or current_load > capacity + 1e-8:
                return None
            current = nxt

        d = dist(current, destination)
        total_distance += d
        arrival = current_time + service_times[current] + d
        early, late = windows[destination]
        return_time = max(arrival, early)

        if return_time > late + 1e-8:
            return None
        if abs(current_load) > 1e-8:
            return None
        return total_distance

    empty_cost = evaluate_route(tuple())
    if empty_cost is None:
        empty_cost = dist(origin, destination)

    def total_cost(routes):
        value = 0.0
        for route in routes:
            c = evaluate_route(tuple(route))
            if c is None:
                return None
            value += c
        return value

    def make_solution(routes, objective=None):
        if objective is None:
            objective = total_cost(routes)
        return {
            "objective_value": float(objective),
            "routes": [
                {
                    "vehicle": k,
                    "nodes": [origin] + list(routes[k]) + [destination],
                }
                for k in range(num_vehicles)
            ],
        }

    best_routes = None
    best_cost = float("inf")

    def record(routes, cost=None):
        nonlocal best_routes, best_cost
        if cost is None:
            cost = total_cost(routes)
        if cost is None:
            return False
        if cost + 1e-9 < best_cost:
            best_cost = cost
            best_routes = [list(r) for r in routes]
            solution = make_solution(best_routes, best_cost)
            if logger:
                logger.log_solution(best_cost, solution)
            return True
        return False

    def insertion_summary(routes, req, allow_time_abort=True):
        """Find the cheapest and second-cheapest feasible insertion."""
        pnode = pickups[req]
        dnode = deliveries[req]
        best = None
        best_delta = float("inf")
        second_delta = float("inf")
        option_count = 0
        seen_empty = False
        checks = 0

        for k, route in enumerate(routes):
            if not route:
                if seen_empty:
                    continue
                seen_empty = True

            old_cost = evaluate_route(tuple(route))
            if old_cost is None:
                continue

            length = len(route)
            for pi in range(length + 1):
                with_pickup = route[:pi] + [pnode] + route[pi:]
                for di in range(pi + 1, length + 2):
                    checks += 1
                    if allow_time_abort and (checks & 255) == 0 and time.monotonic() >= deadline:
                        return None

                    candidate = with_pickup[:di] + [dnode] + with_pickup[di:]
                    new_cost = evaluate_route(tuple(candidate))
                    if new_cost is None:
                        continue

                    option_count += 1
                    delta = new_cost - old_cost
                    if delta + 1e-12 < best_delta:
                        second_delta = best_delta
                        best_delta = delta
                        best = (k, candidate, delta)
                    elif delta < second_delta:
                        second_delta = delta

        if best is None:
            return None

        if not math.isfinite(second_delta):
            regret = 1e12
        else:
            regret = max(0.0, second_delta - best_delta)

        return best, regret, option_count

    def repair(routes, unassigned, randomized=False):
        routes = [list(r) for r in routes]
        remaining = set(unassigned)

        while remaining:
            if time.monotonic() >= deadline:
                return None

            choices = []
            for req in remaining:
                info = insertion_summary(routes, req)
                if info is None:
                    continue
                best_insert, regret, count = info
                latest_pickup = windows[pickups[req]][1]
                # Priority: uniquely constrained requests, regret, few options,
                # and then early pickup deadlines.
                key = (
                    1 if count == 1 else 0,
                    regret,
                    -count,
                    -latest_pickup,
                )
                choices.append((key, req, best_insert))

            if not choices:
                return None

            choices.sort(key=lambda x: x[0], reverse=True)
            if randomized and len(choices) > 1:
                top = min(3, len(choices))
                weights = [5.0, 2.0, 1.0][:top]
                chosen = rng.choices(choices[:top], weights=weights, k=1)[0]
            else:
                chosen = choices[0]

            _, req, insertion = chosen
            vehicle, new_route, _ = insertion
            routes[vehicle] = new_route
            remaining.remove(req)

        return routes

    # Basic impossibility checks before construction.
    individually_possible = True
    for r in range(num_requests):
        if request_loads[r] > capacity + 1e-8:
            individually_possible = False
            break
        direct = (pickups[r], deliveries[r])
        if evaluate_route(direct) is None:
            individually_possible = False
            break

    # Construct initial solutions with randomized regret insertion.
    if individually_possible:
        construction_attempt = 0
        first_solution_time = None
        while time.monotonic() < deadline:
            construction_attempt += 1
            initial_routes = [[] for _ in range(num_vehicles)]
            built = repair(
                initial_routes,
                range(num_requests),
                randomized=(construction_attempt > 1),
            )
            if built is not None:
                cost = total_cost(built)
                if cost is not None:
                    record(built, cost)
                    if first_solution_time is None:
                        first_solution_time = time.monotonic()

            if best_routes is not None:
                # Reserve most of the runtime for improvement.
                elapsed_after_first = time.monotonic() - first_solution_time
                construction_allowance = min(
                    2.0,
                    max(0.1, 0.10 * max(1, args.time_limit)),
                )
                if construction_attempt >= 3 or elapsed_after_first >= construction_allowance:
                    break

    def requests_in_route(route):
        return [request_of_node[n] for n in route if n in request_of_node and n == pickups[request_of_node[n]]]

    def remove_requests(routes, removed):
        removed = set(removed)
        result = []
        for route in routes:
            result.append([
                node for node in route
                if request_of_node.get(node) not in removed
            ])
        return result

    def request_marginal(routes, req):
        for route in routes:
            if pickups[req] in route:
                old_cost = evaluate_route(tuple(route))
                shortened = [
                    n for n in route
                    if n != pickups[req] and n != deliveries[req]
                ]
                new_cost = evaluate_route(tuple(shortened))
                if old_cost is None or new_cost is None:
                    return 0.0
                return old_cost - new_cost
        return 0.0

    def select_removed(routes, q, operator):
        assigned = list(range(num_requests))
        q = min(q, len(assigned))

        if operator == 0:
            return rng.sample(assigned, q)

        if operator == 1:
            scored = [(request_marginal(routes, r), r) for r in assigned]
            scored.sort(reverse=True)
            pool_size = min(len(scored), max(q, 3 * q))
            pool = scored[:pool_size]
            chosen = []
            while pool and len(chosen) < q:
                # Biased toward large-cost requests, with some randomness.
                idx = int((rng.random() ** 2.5) * len(pool))
                chosen.append(pool.pop(idx)[1])
            return chosen

        if operator == 2:
            seed = rng.choice(assigned)
            px, py = coordinates[pickups[seed]]
            dx, dy = coordinates[deliveries[seed]]
            pt = windows[pickups[seed]][0]
            dt = windows[deliveries[seed]][0]

            related = []
            for r in assigned:
                qx, qy = coordinates[pickups[r]]
                ex, ey = coordinates[deliveries[r]]
                spatial = math.hypot(px - qx, py - qy) + math.hypot(dx - ex, dy - ey)
                temporal = abs(pt - windows[pickups[r]][0]) + abs(dt - windows[deliveries[r]][0])
                related.append((spatial + 0.05 * temporal, r))
            related.sort()
            pool = related[:min(len(related), max(q, 4 * q))]
            chosen = [seed]
            pool = [x for x in pool if x[1] != seed]
            while pool and len(chosen) < q:
                idx = int((rng.random() ** 2.0) * len(pool))
                chosen.append(pool.pop(idx)[1])
            return chosen

        # Remove most or all requests from a randomly selected nonempty route.
        nonempty = [r for r in routes if r]
        if not nonempty:
            return rng.sample(assigned, q)
        chosen_route = rng.choice(nonempty)
        route_requests = requests_in_route(chosen_route)
        rng.shuffle(route_requests)
        chosen = route_requests[:q]
        remaining = [r for r in assigned if r not in set(chosen)]
        if len(chosen) < q:
            chosen.extend(rng.sample(remaining, q - len(chosen)))
        return chosen

    # Adaptive large-neighborhood search.
    if best_routes is not None and num_requests > 0 and time.monotonic() < deadline:
        current_routes = [list(r) for r in best_routes]
        current_cost = best_cost

        initial_temperature = max(1e-6, 0.03 * max(1.0, current_cost))
        iteration = 0
        stagnation = 0

        while time.monotonic() < deadline:
            iteration += 1
            stagnation += 1

            if num_requests <= 4:
                q = 1
            else:
                upper = max(2, min(num_requests, int(math.sqrt(num_requests)) + 2))
                if stagnation > 150:
                    upper = min(num_requests, max(upper, num_requests // 3))
                q = rng.randint(1, upper)

            operator = rng.randrange(4)
            removed = select_removed(current_routes, q, operator)
            partial = remove_requests(current_routes, removed)
            candidate = repair(partial, removed, randomized=(rng.random() < 0.20))

            if candidate is None:
                continue

            candidate_cost = total_cost(candidate)
            if candidate_cost is None:
                continue

            progress = min(
                1.0,
                max(0.0, (time.monotonic() - start_time) / max(1e-9, args.time_limit)),
            )
            temperature = initial_temperature * (1.0 - progress) ** 2 + 1e-9
            delta = candidate_cost - current_cost

            accept = delta <= 1e-10
            if not accept and delta / temperature < 50.0:
                accept = rng.random() < math.exp(-delta / temperature)

            if accept:
                current_routes = candidate
                current_cost = candidate_cost

            if record(candidate, candidate_cost):
                current_routes = [list(r) for r in candidate]
                current_cost = candidate_cost
                stagnation = 0

            # Periodically restart from the incumbent.
            if stagnation > 400:
                current_routes = [list(r) for r in best_routes]
                current_cost = best_cost
                stagnation = 0

    # Last-resort direct assignment when enough vehicles exist.
    if best_routes is None and num_requests <= num_vehicles:
        routes = [[] for _ in range(num_vehicles)]
        feasible = True
        for r in range(num_requests):
            routes[r] = [pickups[r], deliveries[r]]
            if evaluate_route(tuple(routes[r])) is None:
                feasible = False
                break
        if feasible:
            record(routes)

    if best_routes is None:
        # No feasible solution was found. This fallback preserves the required
        # schema; for feasible benchmark instances the construction above is
        # expected to find an incumbent.
        best_routes = [[] for _ in range(num_vehicles)]
        best_cost = sum(empty_cost for _ in range(num_vehicles))

    final_solution = make_solution(best_routes, best_cost)

    output_dir = os.path.dirname(os.path.abspath(args.solution_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(final_solution, f, separators=(",", ":"))


if __name__ == "__main__":
    main()
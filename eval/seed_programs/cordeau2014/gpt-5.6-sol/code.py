import argparse
import bisect
import json
import math
import os
import random
import time
from array import array

from solution_logger import SolutionLogger


INF = float("inf")
EPS = 1e-10


class TimeDependentNetwork:
    def __init__(self, data):
        params = data["parameters"]
        arcs = data["arcs"]

        self.H = int(params["H"])
        self.boundaries = [float(x) for x in params["period_boundaries"]]
        self.horizon = float(params["time_horizon_T"])
        self.distances = arcs["distances"]

        speeds = arcs.get("speeds_v_ijh")
        if speeds is not None:
            self.speeds = speeds
        else:
            max_speeds = arcs["max_speeds_u_ij"]
            delta = arcs["delta_ijh"]
            congestion = params["b_h"]
            n = len(self.distances)
            self.speeds = [
                [
                    [
                        float(delta[i][j][h])
                        * float(congestion[h])
                        * float(max_speeds[i][j])
                        for h in range(self.H)
                    ]
                    for j in range(n)
                ]
                for i in range(n)
            ]

        self.max_speeds = arcs.get("max_speeds_u_ij")
        if self.max_speeds is None:
            n = len(self.distances)
            self.max_speeds = [
                [
                    max((float(v) for v in self.speeds[i][j]), default=0.0)
                    for j in range(n)
                ]
                for i in range(n)
            ]

    def arrival(self, i, j, departure):
        distance = float(self.distances[i][j])
        if distance <= EPS:
            return departure
        if not math.isfinite(departure) or departure >= self.horizon - EPS:
            return INF

        h = bisect.bisect_right(self.boundaries, departure) - 1
        if h < 0:
            h = 0
        if h >= self.H:
            return INF

        current = departure
        remaining = distance

        while h < self.H:
            period_end = float(self.boundaries[h + 1])
            if current < float(self.boundaries[h]):
                current = float(self.boundaries[h])

            available_time = max(0.0, period_end - current)
            speed = float(self.speeds[i][j][h])

            if speed > EPS:
                capacity = speed * available_time
                if remaining <= capacity + 1e-9:
                    arrival = current + remaining / speed
                    if arrival <= self.horizon + 1e-8:
                        return min(arrival, self.horizon)
                    return INF
                remaining -= capacity

            current = period_end
            h += 1

        return INF

    def evaluate_permutation(self, depot, permutation):
        current_time = 0.0
        previous = depot
        for vertex in permutation:
            current_time = self.arrival(previous, vertex, current_time)
            if not math.isfinite(current_time):
                return INF
            previous = vertex

        current_time = self.arrival(previous, depot, current_time)
        if current_time > self.horizon + 1e-7:
            return INF
        return current_time

    def proxy_cost(self, i, j):
        distance = float(self.distances[i][j])
        if distance <= EPS:
            return 0.0
        speed = float(self.max_speeds[i][j])
        if speed <= EPS:
            speed = max((float(x) for x in self.speeds[i][j]), default=0.0)
        return distance / speed if speed > EPS else 1e100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    start_time = time.monotonic()
    deadline = start_time + max(0.0, float(args.time_limit)) - 0.03

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    network = TimeDependentNetwork(data)
    total_vertices = int(data["parameters"]["total_vertices"])
    depot = int(data["depot"]["index"])
    customers = [v for v in range(total_vertices) if v != depot]
    m = len(customers)
    rng = random.Random(0)

    best_perm = list(customers)
    best_value = INF

    def make_solution(value, permutation):
        return {
            "objective_value": float(value),
            "tour": [depot] + list(permutation) + [depot],
        }

    def register(permutation, known_value=None):
        nonlocal best_perm, best_value
        value = (
            network.evaluate_permutation(depot, permutation)
            if known_value is None
            else known_value
        )
        if math.isfinite(value) and value + 1e-9 < best_value:
            best_value = float(value)
            best_perm = list(permutation)
            solution = make_solution(best_value, best_perm)
            if logger:
                logger.log_solution(best_value, solution)
            return True
        return False

    # Always establish a structurally valid initial tour.
    register(customers)

    if m == 0:
        register([], 0.0)

    def greedy_tour(randomized=False):
        remaining = set(customers)
        permutation = []
        current = depot
        current_time = 0.0

        while remaining:
            scored = []
            for vertex in remaining:
                arrival = network.arrival(current, vertex, current_time)
                proxy = network.proxy_cost(current, vertex)
                scored.append((arrival, proxy, vertex))
            scored.sort()

            if randomized and len(scored) > 1:
                rcl_size = min(len(scored), max(2, min(7, 1 + len(scored) // 12)))
                rank = int((rng.random() ** 2.0) * rcl_size)
                _, _, chosen = scored[rank]
            else:
                _, _, chosen = scored[0]

            new_time = network.arrival(current, chosen, current_time)
            permutation.append(chosen)
            remaining.remove(chosen)
            current = chosen
            current_time = new_time

        return permutation

    if m > 0 and time.monotonic() < deadline:
        register(greedy_tour(False))

    # Static cheapest insertion gives a useful tour with a different structure.
    def insertion_tour():
        if not customers:
            return []

        first = max(
            customers,
            key=lambda v: network.proxy_cost(depot, v)
            + network.proxy_cost(v, depot),
        )
        route = [first]
        unvisited = set(customers)
        unvisited.remove(first)

        while unvisited and time.monotonic() < deadline:
            best_choice = None
            best_delta = INF
            k = len(route)

            for vertex in unvisited:
                for pos in range(k + 1):
                    left = depot if pos == 0 else route[pos - 1]
                    right = depot if pos == k else route[pos]
                    delta = (
                        network.proxy_cost(left, vertex)
                        + network.proxy_cost(vertex, right)
                        - network.proxy_cost(left, right)
                    )
                    if delta < best_delta:
                        best_delta = delta
                        best_choice = (vertex, pos)

            if best_choice is None:
                route.extend(sorted(unvisited))
                break

            vertex, pos = best_choice
            route.insert(pos, vertex)
            unvisited.remove(vertex)

        if unvisited:
            route.extend(sorted(unvisited))
        return route

    if 1 < m <= 180 and time.monotonic() < deadline:
        register(insertion_tour())

    # Exact label-setting Held-Karp dynamic programming for small instances.
    # Earliest arrival is a sufficient state because piecewise nonnegative
    # speeds induce a FIFO travel-time function.
    def exact_held_karp():
        if m == 0:
            return [], 0.0
        if m > 16:
            return None

        state_count = 1 << m
        size = state_count * m

        try:
            dp = array("d", [INF]) * size
            parent = array("b", [-2]) * size
        except (MemoryError, OverflowError):
            return None

        for j, vertex in enumerate(customers):
            arr = network.arrival(depot, vertex, 0.0)
            idx = (1 << j) * m + j
            dp[idx] = arr
            parent[idx] = -1

        check_counter = 0
        for mask in range(1, state_count):
            if time.monotonic() >= deadline:
                return None

            present = mask
            while present:
                bit_j = present & -present
                j = bit_j.bit_length() - 1
                present ^= bit_j

                base = dp[mask * m + j]
                if not math.isfinite(base):
                    continue

                remaining = (state_count - 1) ^ mask
                while remaining:
                    bit_k = remaining & -remaining
                    k = bit_k.bit_length() - 1
                    remaining ^= bit_k

                    cand = network.arrival(customers[j], customers[k], base)
                    new_mask = mask | bit_k
                    idx = new_mask * m + k
                    if cand + 1e-12 < dp[idx]:
                        dp[idx] = cand
                        parent[idx] = j

                    check_counter += 1
                    if (check_counter & 8191) == 0 and time.monotonic() >= deadline:
                        return None

        full = state_count - 1
        optimum = INF
        last = -1
        for j in range(m):
            base = dp[full * m + j]
            if not math.isfinite(base):
                continue
            finish = network.arrival(customers[j], depot, base)
            if finish < optimum:
                optimum = finish
                last = j

        if last < 0 or not math.isfinite(optimum):
            return None

        reverse_order = []
        mask = full
        j = last
        while j >= 0:
            reverse_order.append(customers[j])
            idx = mask * m + j
            previous = int(parent[idx])
            mask ^= 1 << j
            j = previous

        reverse_order.reverse()
        return reverse_order, optimum

    remaining_time = deadline - time.monotonic()
    if m <= 16 and remaining_time > 0.15:
        exact_result = exact_held_karp()
        if exact_result is not None:
            exact_perm, exact_value = exact_result
            register(exact_perm, exact_value)
            best_perm = exact_perm
            best_value = exact_value

    # For larger instances, produce several randomized greedy starts.
    initial_pool = []
    if best_perm:
        initial_pool.append(list(best_perm))

    random_starts = 0
    max_random_starts = 3 if m > 150 else 8
    while (
        m > 1
        and random_starts < max_random_starts
        and time.monotonic() < deadline
    ):
        candidate = greedy_tour(True)
        value = network.evaluate_permutation(depot, candidate)
        if math.isfinite(value):
            initial_pool.append(candidate)
            register(candidate, value)
        random_starts += 1

    # Candidate lists based on static free-flow arc costs.
    candidate_lists = {}
    if m > 1 and time.monotonic() < deadline:
        candidate_count = min(14, m - 1)
        for x in customers:
            others = [y for y in customers if y != x]
            others.sort(
                key=lambda y: (
                    network.proxy_cost(x, y) + network.proxy_cost(y, x),
                    y,
                )
            )
            candidate_lists[x] = others[:candidate_count]

    def local_descent(start_perm):
        current = list(start_perm)
        current_value = network.evaluate_permutation(depot, current)
        if not math.isfinite(current_value):
            return current, current_value

        register(current, current_value)
        n = len(current)

        while n > 1 and time.monotonic() < deadline:
            positions = {vertex: i for i, vertex in enumerate(current)}
            improved = False

            for i in range(n):
                if time.monotonic() >= deadline:
                    break
                x = current[i]

                targets = candidate_lists.get(x, [])
                for y in targets:
                    if time.monotonic() >= deadline:
                        break
                    j = positions[y]
                    if i == j:
                        continue

                    # Swap.
                    trial = list(current)
                    trial[i], trial[j] = trial[j], trial[i]
                    value = network.evaluate_permutation(depot, trial)
                    if value + 1e-9 < current_value:
                        current, current_value = trial, value
                        register(current, current_value)
                        improved = True
                        break

                    # Reverse a segment (directed 2-opt analogue).
                    a, b = sorted((i, j))
                    if b - a >= 2:
                        trial = (
                            current[:a]
                            + list(reversed(current[a : b + 1]))
                            + current[b + 1 :]
                        )
                        value = network.evaluate_permutation(depot, trial)
                        if value + 1e-9 < current_value:
                            current, current_value = trial, value
                            register(current, current_value)
                            improved = True
                            break

                    # Relocate x immediately before y.
                    trial = list(current)
                    moved = trial.pop(i)
                    target_pos = trial.index(y)
                    trial.insert(target_pos, moved)
                    if trial != current:
                        value = network.evaluate_permutation(depot, trial)
                        if value + 1e-9 < current_value:
                            current, current_value = trial, value
                            register(current, current_value)
                            improved = True
                            break

                    # Relocate x immediately after y.
                    trial = list(current)
                    moved = trial.pop(i)
                    target_pos = trial.index(y) + 1
                    trial.insert(target_pos, moved)
                    if trial != current:
                        value = network.evaluate_permutation(depot, trial)
                        if value + 1e-9 < current_value:
                            current, current_value = trial, value
                            register(current, current_value)
                            improved = True
                            break

                if improved:
                    break

            if not improved:
                break

        return current, current_value

    # Avoid spending time heuristically if exact DP completed.
    exact_completed = (
        m <= 16
        and math.isfinite(best_value)
        and time.monotonic() < deadline
        and exact_result is not None
    ) if m <= 16 else False

    if not exact_completed:
        # Improve a few diverse starts.
        seen = set()
        for start in initial_pool:
            if time.monotonic() >= deadline:
                break
            key = tuple(start)
            if key in seen:
                continue
            seen.add(key)
            local_descent(start)

        # Iterated local search with small reproducible perturbations.
        while m > 2 and time.monotonic() < deadline:
            perturbed = list(best_perm)
            kick_count = 2 if m < 80 else 3

            for _ in range(kick_count):
                move_type = rng.randrange(3)
                i = rng.randrange(m)
                j = rng.randrange(m)
                if i == j:
                    continue

                if move_type == 0:
                    perturbed[i], perturbed[j] = perturbed[j], perturbed[i]
                elif move_type == 1:
                    item = perturbed.pop(i)
                    perturbed.insert(j, item)
                else:
                    a, b = sorted((i, j))
                    perturbed[a : b + 1] = reversed(perturbed[a : b + 1])

            local_descent(perturbed)

    # A feasible tour is expected by the instance specification. If no tour
    # completed within the horizon, still emit a finite, schema-valid result.
    if not math.isfinite(best_value):
        best_perm = list(customers)
        best_value = 1e100

    final_solution = make_solution(best_value, best_perm)

    output_dir = os.path.dirname(os.path.abspath(args.solution_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(final_solution, f, separators=(",", ":"), allow_nan=False)


if __name__ == "__main__":
    main()
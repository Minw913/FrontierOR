import argparse
import copy
import json
import math
import os
import random
import time

from solution_logger import SolutionLogger


EPS = 1e-9


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    return parser.parse_args()


class Solver:
    def __init__(self, data, time_limit, logger):
        self.data = data
        self.n = int(data["n"])
        self.d = int(data["d"])
        self.t = int(data["t"])
        self.m = int(data["m"])
        self.capacity = float(data["vehicle_capacity"])
        self.duration_limit = float(data["route_duration_limit"])
        self.logger = logger

        self.start_time = time.monotonic()
        # Keep a small amount of time for serialization.
        self.deadline = self.start_time + max(0.01, float(time_limit) - 0.05)
        self.rng = random.Random(0)

        depots_by_id = {int(x["id"]): x for x in data["depots"]}
        customers_by_id = {int(x["id"]): x for x in data["customers"]}

        self.depots = [depots_by_id[i] for i in range(self.d)]
        self.customers = [customers_by_id[i] for i in range(self.n)]

        self.demands = [float(c["demand"]) for c in self.customers]
        self.services = [float(c["service_duration"]) for c in self.customers]
        self.patterns = []
        for c in self.customers:
            seen = set()
            pats = []
            for pattern in c["pattern_list"]:
                p = tuple(sorted(int(x) for x in pattern))
                if p not in seen:
                    seen.add(p)
                    pats.append(p)
            self.patterns.append(pats)

        self.distance = self._read_distance_matrix(data)
        positive = [
            self.distance[i][j]
            for i in range(self.n + self.d)
            for j in range(self.n + self.d)
            if i != j and self.distance[i][j] > EPS
        ]
        self.distance_scale = sum(positive) / len(positive) if positive else 1.0

        self.best_state = None
        self.best_assignments = None
        self.best_objective = float("inf")

    def out_of_time(self):
        return time.monotonic() >= self.deadline

    def node_of_customer(self, customer):
        return self.d + customer

    def _read_distance_matrix(self, data):
        matrix = None
        for key in (
            "travel_costs",
            "travel_times",
            "travel_cost_matrix",
            "travel_time_matrix",
            "distance_matrix",
            "cost_matrix",
        ):
            candidate = data.get(key)
            if isinstance(candidate, list):
                matrix = candidate
                break

        size = self.d + self.n
        if matrix is not None:
            if len(matrix) == size and all(
                isinstance(row, list) and len(row) == size for row in matrix
            ):
                return [[float(x) for x in row] for row in matrix]

            if len(matrix) == size * size and all(
                isinstance(x, (int, float)) for x in matrix
            ):
                return [
                    [float(matrix[i * size + j]) for j in range(size)]
                    for i in range(size)
                ]

        # The schema supplies coordinates, so Euclidean travel is the fallback.
        locations = self.depots + self.customers
        result = [[0.0] * size for _ in range(size)]
        for i in range(size):
            xi, yi = float(locations[i]["x"]), float(locations[i]["y"])
            for j in range(size):
                xj, yj = float(locations[j]["x"]), float(locations[j]["y"])
                result[i][j] = math.hypot(xi - xj, yi - yj)
        return result

    def empty_state(self):
        return [[[] for _ in range(self.t)] for _ in range(self.d)]

    def route_cost(self, depot, sequence):
        if not sequence:
            return 0.0
        total = self.distance[depot][self.node_of_customer(sequence[0])]
        for a, b in zip(sequence, sequence[1:]):
            total += self.distance[self.node_of_customer(a)][
                self.node_of_customer(b)
            ]
        total += self.distance[self.node_of_customer(sequence[-1])][depot]
        return total

    def route_duration(self, depot, sequence):
        return self.route_cost(depot, sequence) + sum(
            self.services[c] for c in sequence
        )

    def route_load(self, sequence):
        return sum(self.demands[c] for c in sequence)

    def make_route(self, depot, sequence):
        sequence = list(sequence)
        return {
            "customers": sequence,
            "load": self.route_load(sequence),
            "duration": self.route_duration(depot, sequence),
            "cost": self.route_cost(depot, sequence),
        }

    def state_objective(self, state):
        return sum(
            route["cost"]
            for depot_periods in state
            for routes in depot_periods
            for route in routes
        )

    def feasible_route(self, route):
        return (
            route["load"] <= self.capacity + EPS
            and route["duration"] <= self.duration_limit + EPS
        )

    def best_insertion(self, state, depot, period, customer, randomized=False):
        routes = state[depot][period - 1]
        best = None

        for r_idx, route in enumerate(routes):
            if route["load"] + self.demands[customer] > self.capacity + EPS:
                continue
            old_cost = route["cost"]
            seq = route["customers"]
            for pos in range(len(seq) + 1):
                new_seq = seq[:pos] + [customer] + seq[pos:]
                new_route = self.make_route(depot, new_seq)
                if not self.feasible_route(new_route):
                    continue

                delta = new_route["cost"] - old_cost
                slack = max(0.0, self.duration_limit - new_route["duration"])
                score = delta + 1e-6 * slack
                if randomized:
                    score += self.rng.random() * max(1.0, self.distance_scale) * 0.04

                candidate = (score, delta, r_idx, pos, new_route)
                if best is None or candidate[0] < best[0] - EPS:
                    best = candidate

        if len(routes) < self.m:
            new_route = self.make_route(depot, [customer])
            if self.feasible_route(new_route):
                delta = new_route["cost"]
                # A small route-opening penalty helps preserve the limited fleet.
                score = delta + 0.02 * self.distance_scale
                if randomized:
                    score += self.rng.random() * max(1.0, self.distance_scale) * 0.04
                candidate = (score, delta, len(routes), 0, new_route)
                if best is None or candidate[0] < best[0] - EPS:
                    best = candidate

        return best

    def apply_insertion(self, state, depot, period, insertion):
        _, _, route_index, _, new_route = insertion
        routes = state[depot][period - 1]
        if route_index == len(routes):
            routes.append(new_route)
        else:
            routes[route_index] = new_route

    def evaluate_assignment_option(
        self, state, customer, depot, pattern, randomized=False
    ):
        insertions = []
        score = 0.0
        delta = 0.0
        for period in pattern:
            if period < 1 or period > self.t:
                return None
            insertion = self.best_insertion(
                state, depot, period, customer, randomized=randomized
            )
            if insertion is None:
                return None
            insertions.append((period, insertion))
            score += insertion[0]
            delta += insertion[1]
        return score, delta, insertions

    def customer_difficulty(self, customer):
        min_singleton = float("inf")
        for depot in range(self.d):
            singleton = self.route_duration(depot, [customer])
            min_singleton = min(min_singleton, singleton)

        option_count = max(
            1, self.d * max(1, len(self.patterns[customer]))
        )
        frequency = max((len(p) for p in self.patterns[customer]), default=0)
        return (
            3.0 * self.demands[customer] / max(1.0, self.capacity)
            + 2.0 * min_singleton / max(1.0, self.duration_limit)
            + 0.3 * frequency
            + 1.0 / option_count
        )

    def construct(self, attempt):
        state = self.empty_state()
        assignments = {}

        customers = list(range(self.n))
        if attempt == 0:
            customers.sort(key=self.customer_difficulty, reverse=True)
        else:
            customers.sort(
                key=lambda c: self.customer_difficulty(c)
                * (0.75 + 0.5 * self.rng.random()),
                reverse=True,
            )

        randomized = attempt > 0

        for customer in customers:
            options = []
            for depot in range(self.d):
                for pattern in self.patterns[customer]:
                    option = self.evaluate_assignment_option(
                        state, customer, depot, pattern, randomized
                    )
                    if option is not None:
                        score, delta, insertions = option
                        options.append(
                            (score, delta, depot, tuple(pattern), insertions)
                        )

            if not options:
                return None, None

            options.sort(key=lambda x: x[0])
            if randomized and len(options) > 1:
                # Mostly greedy, with controlled diversification.
                restricted = min(len(options), 3)
                weights = [1.0 / (i + 1) ** 2 for i in range(restricted)]
                choice = self.rng.choices(
                    options[:restricted], weights=weights, k=1
                )[0]
            else:
                choice = options[0]

            _, _, depot, pattern, insertions = choice
            for period, insertion in insertions:
                self.apply_insertion(state, depot, period, insertion)
            assignments[customer] = {
                "depot": depot,
                "pattern": list(pattern),
            }

        return state, assignments

    def improve_two_opt(self, state):
        for depot in range(self.d):
            for period in range(self.t):
                for route_index in range(len(state[depot][period])):
                    if self.out_of_time():
                        return
                    route = state[depot][period][route_index]
                    seq = route["customers"]
                    if len(seq) < 3:
                        continue

                    improved = True
                    while improved and not self.out_of_time():
                        improved = False
                        current_cost = route["cost"]
                        best_route = route
                        for i in range(len(seq) - 1):
                            for j in range(i + 1, len(seq)):
                                candidate_seq = (
                                    seq[:i]
                                    + list(reversed(seq[i : j + 1]))
                                    + seq[j + 1 :]
                                )
                                candidate = self.make_route(depot, candidate_seq)
                                if (
                                    candidate["duration"]
                                    <= self.duration_limit + EPS
                                    and candidate["cost"]
                                    < best_route["cost"] - EPS
                                ):
                                    best_route = candidate
                            if self.out_of_time():
                                break
                        if best_route["cost"] < current_cost - EPS:
                            route = best_route
                            seq = route["customers"]
                            state[depot][period][route_index] = route
                            improved = True

    def improve_cross_route_relocations(self, state):
        for depot in range(self.d):
            for period in range(self.t):
                routes = state[depot][period]
                improved = True
                while improved and not self.out_of_time():
                    improved = False
                    best_move = None
                    best_delta = -EPS

                    for source_index, source in enumerate(routes):
                        for customer_pos, customer in enumerate(source["customers"]):
                            source_seq = (
                                source["customers"][:customer_pos]
                                + source["customers"][customer_pos + 1 :]
                            )
                            new_source = (
                                self.make_route(depot, source_seq)
                                if source_seq
                                else None
                            )

                            for target_index, target in enumerate(routes):
                                if source_index == target_index:
                                    continue
                                if (
                                    target["load"] + self.demands[customer]
                                    > self.capacity + EPS
                                ):
                                    continue
                                for target_pos in range(
                                    len(target["customers"]) + 1
                                ):
                                    target_seq = (
                                        target["customers"][:target_pos]
                                        + [customer]
                                        + target["customers"][target_pos:]
                                    )
                                    new_target = self.make_route(depot, target_seq)
                                    if not self.feasible_route(new_target):
                                        continue

                                    old_cost = source["cost"] + target["cost"]
                                    new_cost = new_target["cost"] + (
                                        new_source["cost"]
                                        if new_source is not None
                                        else 0.0
                                    )
                                    delta = new_cost - old_cost
                                    if delta < best_delta:
                                        best_delta = delta
                                        best_move = (
                                            source_index,
                                            target_index,
                                            new_source,
                                            new_target,
                                        )

                    if best_move is not None:
                        source_index, target_index, new_source, new_target = best_move
                        routes[target_index] = new_target
                        if new_source is None:
                            del routes[source_index]
                        else:
                            routes[source_index] = new_source
                        improved = True

    def remove_customer(self, state, customer, assignment):
        depot = assignment["depot"]
        for period in assignment["pattern"]:
            routes = state[depot][period - 1]
            new_routes = []
            for route in routes:
                if customer in route["customers"]:
                    seq = [c for c in route["customers"] if c != customer]
                    if seq:
                        new_routes.append(self.make_route(depot, seq))
                else:
                    new_routes.append(route)
            state[depot][period - 1] = new_routes

    def improve_assignments(self, state, assignments):
        current_obj = self.state_objective(state)
        order = list(range(self.n))
        self.rng.shuffle(order)

        for customer in order:
            if self.out_of_time():
                break

            base_state = copy.deepcopy(state)
            self.remove_customer(base_state, customer, assignments[customer])
            base_obj = self.state_objective(base_state)

            best_candidate = None
            best_obj = current_obj

            for depot in range(self.d):
                for pattern in self.patterns[customer]:
                    if self.out_of_time():
                        break
                    option = self.evaluate_assignment_option(
                        base_state, customer, depot, pattern, randomized=False
                    )
                    if option is None:
                        continue
                    _, delta, insertions = option
                    candidate_obj = base_obj + delta
                    if candidate_obj < best_obj - EPS:
                        best_obj = candidate_obj
                        best_candidate = (depot, tuple(pattern), insertions)
                if self.out_of_time():
                    break

            if best_candidate is not None:
                depot, pattern, insertions = best_candidate
                state = base_state
                for period, insertion in insertions:
                    self.apply_insertion(state, depot, period, insertion)
                assignments[customer] = {
                    "depot": depot,
                    "pattern": list(pattern),
                }
                current_obj = self.state_objective(state)
                self.consider_incumbent(state, assignments)

        return state, assignments

    def solution_dict(self, state, assignments):
        routes_output = []
        for depot in range(self.d):
            for period in range(self.t):
                for vehicle, route in enumerate(state[depot][period]):
                    if route["customers"]:
                        routes_output.append(
                            {
                                "depot": depot,
                                "period": period + 1,
                                "vehicle": vehicle,
                                "customers": list(route["customers"]),
                            }
                        )

        assignment_output = {
            str(customer): {
                "depot": int(assignments[customer]["depot"]),
                "pattern": [int(x) for x in assignments[customer]["pattern"]],
            }
            for customer in range(self.n)
        }

        return {
            "objective_value": float(self.state_objective(state)),
            "routes": routes_output,
            "assignments": assignment_output,
        }

    def consider_incumbent(self, state, assignments):
        objective = self.state_objective(state)
        if objective < self.best_objective - EPS:
            self.best_objective = objective
            self.best_state = copy.deepcopy(state)
            self.best_assignments = copy.deepcopy(assignments)
            solution = self.solution_dict(self.best_state, self.best_assignments)
            if self.logger:
                self.logger.log_solution(objective, solution)
            return True
        return False

    def validate_basic_feasibility(self, state, assignments):
        if len(assignments) != self.n:
            return False

        visits = [[0] * (self.t + 1) for _ in range(self.n)]
        route_depot = [[None] * (self.t + 1) for _ in range(self.n)]

        for depot in range(self.d):
            for period in range(self.t):
                routes = state[depot][period]
                if len(routes) > self.m:
                    return False
                for route in routes:
                    if len(route["customers"]) != len(set(route["customers"])):
                        return False
                    rebuilt = self.make_route(depot, route["customers"])
                    if not self.feasible_route(rebuilt):
                        return False
                    for c in route["customers"]:
                        visits[c][period + 1] += 1
                        route_depot[c][period + 1] = depot

        for c in range(self.n):
            assigned_depot = assignments[c]["depot"]
            pattern = set(assignments[c]["pattern"])
            for period in range(1, self.t + 1):
                required = period in pattern
                if visits[c][period] != (1 if required else 0):
                    return False
                if required and route_depot[c][period] != assigned_depot:
                    return False
        return True

    def solve(self):
        attempt = 0
        # Always make at least one construction attempt, even for tiny limits.
        while attempt == 0 or not self.out_of_time():
            state, assignments = self.construct(attempt)
            attempt += 1

            if state is None:
                if attempt >= 500 and self.best_state is None:
                    break
                continue

            if not self.validate_basic_feasibility(state, assignments):
                continue

            raw_objective = self.state_objective(state)
            promising = (
                self.best_state is None
                or raw_objective < self.best_objective * 1.10
            )

            if promising and not self.out_of_time():
                self.improve_two_opt(state)
                self.improve_cross_route_relocations(state)

            self.consider_incumbent(state, assignments)

            elapsed_fraction = (
                (time.monotonic() - self.start_time)
                / max(0.01, self.deadline - self.start_time)
            )
            if elapsed_fraction > 0.68 and self.best_state is not None:
                break

        if self.best_state is None:
            raise RuntimeError(
                "No feasible solution was found. The instance may be infeasible "
                "or the time limit may be too small."
            )

        state = copy.deepcopy(self.best_state)
        assignments = copy.deepcopy(self.best_assignments)

        if not self.out_of_time():
            state, assignments = self.improve_assignments(state, assignments)
        if not self.out_of_time():
            self.improve_two_opt(state)
        if not self.out_of_time():
            self.improve_cross_route_relocations(state)

        self.consider_incumbent(state, assignments)
        return self.solution_dict(self.best_state, self.best_assignments)


def main():
    args = parse_args()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    solver = Solver(data, args.time_limit, logger)
    solution = solver.solve()

    output_directory = os.path.dirname(os.path.abspath(args.solution_path))
    os.makedirs(output_directory, exist_ok=True)

    temporary_path = args.solution_path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as f:
        json.dump(solution, f, separators=(",", ":"), allow_nan=False)
    os.replace(temporary_path, args.solution_path)


if __name__ == "__main__":
    main()
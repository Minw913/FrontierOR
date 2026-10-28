import argparse
import copy
import json
import math
import os
import random
import time

from solution_logger import SolutionLogger


EPS = 1e-9


def load_instance(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_distance_matrix(instance):
    depots = sorted(instance["depots"], key=lambda x: x["id"])
    customers = sorted(instance["customers"], key=lambda x: x["id"])
    d = instance["d"]
    n = instance["n"]
    size = d + n

    matrix = None
    for key in (
        "travel_costs",
        "travel_times",
        "distance_matrix",
        "cost_matrix",
        "travel_cost_matrix",
        "travel_time_matrix",
    ):
        value = instance.get(key)
        if isinstance(value, list) and len(value) == size:
            matrix = value
            break

    if matrix is not None:
        return [[float(matrix[i][j]) for j in range(size)] for i in range(size)]

    coordinates = [(float(x["x"]), float(x["y"])) for x in depots]
    coordinates.extend((float(x["x"]), float(x["y"])) for x in customers)

    result = [[0.0] * size for _ in range(size)]
    for i in range(size):
        xi, yi = coordinates[i]
        for j in range(size):
            xj, yj = coordinates[j]
            result[i][j] = math.hypot(xi - xj, yi - yj)
    return result


class Solver:
    def __init__(self, instance, logger, deadline):
        self.instance = instance
        self.logger = logger
        self.deadline = deadline
        self.rng = random.Random(0)

        self.n = int(instance["n"])
        self.d = int(instance["d"])
        self.t = int(instance["t"])
        self.m = int(instance["m"])
        self.capacity = float(instance["vehicle_capacity"])
        self.duration_limit = float(instance["route_duration_limit"])

        self.depots = sorted(instance["depots"], key=lambda x: x["id"])
        raw_customers = sorted(instance["customers"], key=lambda x: x["id"])
        self.customers = [None] * self.n
        for customer in raw_customers:
            self.customers[int(customer["id"])] = customer

        self.distance = build_distance_matrix(instance)
        self.demands = [float(c["demand"]) for c in self.customers]
        self.service = [float(c["service_duration"]) for c in self.customers]

        self.patterns = []
        for customer in self.customers:
            unique = []
            seen = set()
            for pattern in customer["pattern_list"]:
                p = tuple(sorted(set(int(day) - 1 for day in pattern)))
                if p not in seen:
                    seen.add(p)
                    unique.append(p)
            self.patterns.append(unique)

        self.best_state = None
        self.best_assignments = None
        self.best_objective = math.inf

    def timed_out(self, reserve=0.0):
        return time.monotonic() + reserve >= self.deadline

    def node_customer(self, customer):
        return self.d + customer

    def arc(self, depot, a, b):
        ia = depot if a is None else self.node_customer(a)
        ib = depot if b is None else self.node_customer(b)
        return self.distance[ia][ib]

    def route_metrics(self, depot, sequence):
        if not sequence:
            return 0.0, 0.0, 0.0

        travel = self.arc(depot, None, sequence[0])
        for a, b in zip(sequence, sequence[1:]):
            travel += self.arc(depot, a, b)
        travel += self.arc(depot, sequence[-1], None)

        load = sum(self.demands[i] for i in sequence)
        duration = travel + sum(self.service[i] for i in sequence)
        return load, duration, travel

    def make_route(self, depot, sequence):
        load, duration, travel = self.route_metrics(depot, sequence)
        return {
            "customers": list(sequence),
            "load": load,
            "duration": duration,
            "travel": travel,
        }

    def empty_state(self):
        return [[[] for _ in range(self.t)] for _ in range(self.d)]

    def objective(self, state):
        return sum(
            route["travel"]
            for depot_days in state
            for routes in depot_days
            for route in routes
        )

    def best_insertion(self, state, depot, period, customer):
        routes = state[depot][period]
        best = None

        for route_index, route in enumerate(routes):
            if route["load"] + self.demands[customer] > self.capacity + EPS:
                continue
            sequence = route["customers"]
            for position in range(len(sequence) + 1):
                candidate = sequence[:position] + [customer] + sequence[position:]
                load, duration, travel = self.route_metrics(depot, candidate)
                if duration <= self.duration_limit + EPS:
                    delta = travel - route["travel"]
                    item = (delta, route_index, position, load, duration, travel)
                    if best is None or item[0] < best[0] - EPS:
                        best = item

        if len(routes) < self.m:
            load, duration, travel = self.route_metrics(depot, [customer])
            if load <= self.capacity + EPS and duration <= self.duration_limit + EPS:
                item = (travel, len(routes), 0, load, duration, travel)
                if best is None or item[0] < best[0] - EPS:
                    best = item

        return best

    def evaluate_joint_insertion(self, state, depot, pattern, customer):
        decisions = []
        total_delta = 0.0
        for period in pattern:
            if period < 0 or period >= self.t:
                return None
            insertion = self.best_insertion(state, depot, period, customer)
            if insertion is None:
                return None
            decisions.append((period, insertion))
            total_delta += insertion[0]
        return total_delta, decisions

    def apply_joint_insertion(self, state, depot, customer, evaluated):
        _, decisions = evaluated
        for period, insertion in decisions:
            _, route_index, position, _, _, _ = insertion
            routes = state[depot][period]
            if route_index == len(routes):
                routes.append(self.make_route(depot, [customer]))
            else:
                sequence = routes[route_index]["customers"]
                sequence = sequence[:position] + [customer] + sequence[position:]
                routes[route_index] = self.make_route(depot, sequence)

    def customer_options(self, customer):
        return [
            (depot, pattern)
            for depot in range(self.d)
            for pattern in self.patterns[customer]
        ]

    def construct(self, attempt):
        state = self.empty_state()
        assignments = {}

        order = list(range(self.n))

        def difficulty(i):
            option_count = max(1, self.d * len(self.patterns[i]))
            solo_best = math.inf
            for depot in range(self.d):
                _, duration, _ = self.route_metrics(depot, [i])
                solo_best = min(solo_best, duration)
            hardness = (
                self.demands[i] / max(1.0, self.capacity)
                + solo_best / max(1.0, self.duration_limit)
                + len(self.patterns[i][0]) if self.patterns[i] else 0
            )
            noise = self.rng.random() * (0.35 if attempt else 0.0)
            return option_count + noise, -hardness

        order.sort(key=difficulty)

        for customer in order:
            candidates = []
            for depot, pattern in self.customer_options(customer):
                evaluated = self.evaluate_joint_insertion(
                    state, depot, pattern, customer
                )
                if evaluated is None:
                    continue

                delta = evaluated[0]
                depot_node = depot
                customer_node = self.node_customer(customer)
                geographic = self.distance[depot_node][customer_node]
                noise = 1.0 + self.rng.uniform(-0.08, 0.08) if attempt else 1.0
                score = delta * noise + 1e-7 * geographic
                candidates.append((score, delta, depot, pattern, evaluated))

            if not candidates:
                return None, None

            candidates.sort(key=lambda x: x[0])
            if attempt and len(candidates) > 1 and self.rng.random() < 0.22:
                chosen = candidates[self.rng.randrange(min(3, len(candidates)))]
            else:
                chosen = candidates[0]

            _, _, depot, pattern, evaluated = chosen
            self.apply_joint_insertion(state, depot, customer, evaluated)
            assignments[customer] = (depot, tuple(pattern))

        return state, assignments

    def solution_dict(self, state, assignments):
        routes_output = []
        for depot in range(self.d):
            for period in range(self.t):
                for vehicle_index, route in enumerate(state[depot][period]):
                    if route["customers"]:
                        routes_output.append(
                            {
                                "depot": int(depot),
                                "period": int(period + 1),
                                "vehicle": int(vehicle_index + 1),
                                "customers": [
                                    int(i) for i in route["customers"]
                                ],
                            }
                        )

        assignments_output = {}
        for customer in range(self.n):
            depot, pattern = assignments[customer]
            assignments_output[str(customer)] = {
                "depot": int(depot),
                "pattern": [int(day + 1) for day in pattern],
            }

        return {
            "objective_value": float(self.objective(state)),
            "routes": routes_output,
            "assignments": assignments_output,
        }

    def record_if_better(self, state, assignments):
        value = self.objective(state)
        if value < self.best_objective - 1e-7:
            self.best_objective = value
            self.best_state = copy.deepcopy(state)
            self.best_assignments = copy.deepcopy(assignments)
            if self.logger:
                solution = self.solution_dict(state, assignments)
                self.logger.log_solution(value, solution)
            return True
        return False

    def optimize_route_orders(self, state, assignments):
        improved_any = False

        for depot in range(self.d):
            for period in range(self.t):
                routes = state[depot][period]
                for route_index in range(len(routes)):
                    changed = True
                    while changed and not self.timed_out(0.05):
                        changed = False
                        route = routes[route_index]
                        sequence = route["customers"]
                        length = len(sequence)

                        for left in range(length - 1):
                            if changed:
                                break
                            for right in range(left + 1, length):
                                candidate = (
                                    sequence[:left]
                                    + list(reversed(sequence[left:right + 1]))
                                    + sequence[right + 1:]
                                )
                                new_route = self.make_route(depot, candidate)
                                if (
                                    new_route["duration"]
                                    <= self.duration_limit + EPS
                                    and new_route["travel"]
                                    < route["travel"] - 1e-7
                                ):
                                    routes[route_index] = new_route
                                    route = new_route
                                    sequence = candidate
                                    length = len(sequence)
                                    changed = True
                                    improved_any = True
                                    self.record_if_better(state, assignments)
                                    break

        return improved_any

    def optimize_period_relocations(self, state, assignments):
        improved_any = False

        for depot in range(self.d):
            for period in range(self.t):
                changed = True
                while changed and not self.timed_out(0.08):
                    changed = False
                    routes = state[depot][period]

                    for source_index in range(len(routes)):
                        if changed:
                            break
                        source = routes[source_index]

                        for source_position, customer in enumerate(
                            list(source["customers"])
                        ):
                            if changed:
                                break

                            source_sequence = (
                                source["customers"][:source_position]
                                + source["customers"][source_position + 1:]
                            )
                            reduced_source = self.make_route(
                                depot, source_sequence
                            )
                            old_total = source["travel"]

                            for target_index in range(len(routes)):
                                if target_index == source_index:
                                    base_sequence = source_sequence
                                    base_travel = reduced_source["travel"]
                                else:
                                    target = routes[target_index]
                                    if (
                                        target["load"] + self.demands[customer]
                                        > self.capacity + EPS
                                    ):
                                        continue
                                    base_sequence = target["customers"]
                                    base_travel = target["travel"]
                                    old_total = (
                                        source["travel"] + target["travel"]
                                    )

                                for position in range(len(base_sequence) + 1):
                                    candidate_sequence = (
                                        base_sequence[:position]
                                        + [customer]
                                        + base_sequence[position:]
                                    )
                                    candidate_route = self.make_route(
                                        depot, candidate_sequence
                                    )
                                    if (
                                        candidate_route["load"]
                                        > self.capacity + EPS
                                        or candidate_route["duration"]
                                        > self.duration_limit + EPS
                                    ):
                                        continue

                                    if target_index == source_index:
                                        new_total = candidate_route["travel"]
                                    else:
                                        new_total = (
                                            reduced_source["travel"]
                                            + candidate_route["travel"]
                                        )

                                    if new_total < old_total - 1e-7:
                                        if target_index == source_index:
                                            routes[source_index] = candidate_route
                                        else:
                                            routes[source_index] = reduced_source
                                            routes[target_index] = candidate_route
                                            routes[:] = [
                                                r for r in routes if r["customers"]
                                            ]
                                        changed = True
                                        improved_any = True
                                        self.record_if_better(state, assignments)
                                        break
                                if changed:
                                    break

        return improved_any

    def remove_customer_visits(self, state, assignments, customer):
        depot, pattern = assignments[customer]
        for period in pattern:
            routes = state[depot][period]
            for route_index, route in enumerate(routes):
                if customer in route["customers"]:
                    sequence = [x for x in route["customers"] if x != customer]
                    if sequence:
                        routes[route_index] = self.make_route(depot, sequence)
                    else:
                        del routes[route_index]
                    break

    def optimize_joint_assignments(self, state, assignments):
        improved_any = False
        customers = list(range(self.n))
        self.rng.shuffle(customers)

        for customer in customers:
            if self.timed_out(0.1):
                break

            old_objective = self.objective(state)
            base_state = copy.deepcopy(state)
            self.remove_customer_visits(base_state, assignments, customer)

            best_candidate_state = None
            best_assignment = None
            best_value = old_objective

            options = self.customer_options(customer)
            self.rng.shuffle(options)

            for depot, pattern in options:
                evaluated = self.evaluate_joint_insertion(
                    base_state, depot, pattern, customer
                )
                if evaluated is None:
                    continue

                candidate_state = copy.deepcopy(base_state)
                self.apply_joint_insertion(
                    candidate_state, depot, customer, evaluated
                )
                value = self.objective(candidate_state)
                if value < best_value - 1e-7:
                    best_value = value
                    best_candidate_state = candidate_state
                    best_assignment = (depot, tuple(pattern))

            if best_candidate_state is not None:
                state[:] = best_candidate_state
                assignments[customer] = best_assignment
                improved_any = True
                self.record_if_better(state, assignments)

        return improved_any

    def solve_with_mip_fallback(self):
        try:
            import gurobipy as gp
            from gurobipy import GRB
        except Exception:
            return False

        remaining = self.deadline - time.monotonic()
        if remaining <= 0.5:
            return False

        arc_count = self.d * self.m * self.t * (
            self.n * (self.n - 1) + 2 * self.n
        )
        if arc_count > 350000:
            return False

        model = gp.Model("periodic_multi_depot_vrp")
        model.Params.OutputFlag = 0
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1
        model.Params.TimeLimit = max(0.1, remaining - 0.1)

        y = {}
        for i in range(self.n):
            for depot in range(self.d):
                for pattern_index in range(len(self.patterns[i])):
                    y[i, depot, pattern_index] = model.addVar(
                        vtype=GRB.BINARY,
                        name=f"y_{i}_{depot}_{pattern_index}",
                    )

        x = {}
        nodes = [-1] + list(range(self.n))
        for depot in range(self.d):
            for vehicle in range(self.m):
                for period in range(self.t):
                    for a in nodes:
                        for b in nodes:
                            if a == b:
                                continue
                            if a == -1 and b == -1:
                                continue
                            x[depot, vehicle, period, a, b] = model.addVar(
                                vtype=GRB.BINARY
                            )

        u = {}
        for depot in range(self.d):
            for vehicle in range(self.m):
                for period in range(self.t):
                    for i in range(self.n):
                        u[depot, vehicle, period, i] = model.addVar(
                            lb=0.0, ub=float(self.n), vtype=GRB.CONTINUOUS
                        )

        model.update()

        for i in range(self.n):
            model.addConstr(
                gp.quicksum(
                    y[i, depot, p]
                    for depot in range(self.d)
                    for p in range(len(self.patterns[i]))
                )
                == 1
            )

        for depot in range(self.d):
            for period in range(self.t):
                for i in range(self.n):
                    required = gp.quicksum(
                        y[i, depot, p]
                        for p, pattern in enumerate(self.patterns[i])
                        if period in pattern
                    )
                    outgoing = gp.quicksum(
                        x[depot, vehicle, period, i, b]
                        for vehicle in range(self.m)
                        for b in nodes
                        if b != i
                    )
                    model.addConstr(outgoing == required)

                for vehicle in range(self.m):
                    departure = gp.quicksum(
                        x[depot, vehicle, period, -1, j]
                        for j in range(self.n)
                    )
                    returning = gp.quicksum(
                        x[depot, vehicle, period, i, -1]
                        for i in range(self.n)
                    )
                    model.addConstr(departure == returning)
                    model.addConstr(departure <= 1)

                    for i in range(self.n):
                        incoming = gp.quicksum(
                            x[depot, vehicle, period, a, i]
                            for a in nodes
                            if a != i
                        )
                        outgoing = gp.quicksum(
                            x[depot, vehicle, period, i, b]
                            for b in nodes
                            if b != i
                        )
                        model.addConstr(incoming == outgoing)

                    model.addConstr(
                        gp.quicksum(
                            self.demands[i]
                            * gp.quicksum(
                                x[depot, vehicle, period, i, b]
                                for b in nodes
                                if b != i
                            )
                            for i in range(self.n)
                        )
                        <= self.capacity
                    )

                    duration_expr = gp.LinExpr()
                    for a in nodes:
                        for b in nodes:
                            if a == b or (a == -1 and b == -1):
                                continue
                            travel = self.arc(
                                depot,
                                None if a == -1 else a,
                                None if b == -1 else b,
                            )
                            service = 0.0 if a == -1 else self.service[a]
                            duration_expr += (travel + service) * x[
                                depot, vehicle, period, a, b
                            ]
                    model.addConstr(duration_expr <= self.duration_limit)

                    for i in range(self.n):
                        for j in range(self.n):
                            if i == j:
                                continue
                            model.addConstr(
                                u[depot, vehicle, period, j]
                                >= u[depot, vehicle, period, i]
                                + 1
                                - self.n
                                * (
                                    1
                                    - x[
                                        depot,
                                        vehicle,
                                        period,
                                        i,
                                        j,
                                    ]
                                )
                            )

        objective = gp.LinExpr()
        for key, variable in x.items():
            depot, _, _, a, b = key
            objective += self.arc(
                depot,
                None if a == -1 else a,
                None if b == -1 else b,
            ) * variable
        model.setObjective(objective, GRB.MINIMIZE)
        model.optimize()

        if model.SolCount <= 0:
            return False

        state = self.empty_state()
        assignments = {}

        for i in range(self.n):
            chosen = None
            for depot in range(self.d):
                for p, pattern in enumerate(self.patterns[i]):
                    if y[i, depot, p].X > 0.5:
                        chosen = (depot, tuple(pattern))
                        break
                if chosen:
                    break
            if chosen is None:
                return False
            assignments[i] = chosen

        for depot in range(self.d):
            for vehicle in range(self.m):
                for period in range(self.t):
                    first = None
                    for j in range(self.n):
                        if x[depot, vehicle, period, -1, j].X > 0.5:
                            first = j
                            break
                    if first is None:
                        continue

                    sequence = []
                    current = first
                    seen = set()
                    while current is not None and current not in seen:
                        seen.add(current)
                        sequence.append(current)
                        next_customer = None
                        for j in range(self.n):
                            if (
                                j != current
                                and x[
                                    depot,
                                    vehicle,
                                    period,
                                    current,
                                    j,
                                ].X
                                > 0.5
                            ):
                                next_customer = j
                                break
                        current = next_customer

                    if sequence:
                        state[depot][period].append(
                            self.make_route(depot, sequence)
                        )

        self.record_if_better(state, assignments)
        return True

    def solve(self):
        construction_end = min(
            self.deadline - 0.2,
            time.monotonic()
            + max(0.25, 0.30 * (self.deadline - time.monotonic())),
        )

        attempt = 0
        max_attempts = max(12, min(250, 5 * self.n + 20))
        while (
            attempt < max_attempts
            and time.monotonic() < construction_end
        ):
            state, assignments = self.construct(attempt)
            if state is not None:
                self.record_if_better(state, assignments)
            attempt += 1

        if self.best_state is None:
            if not self.solve_with_mip_fallback():
                while not self.timed_out(0.15):
                    state, assignments = self.construct(attempt)
                    attempt += 1
                    if state is not None:
                        self.record_if_better(state, assignments)
                        break

        if self.best_state is None:
            raise RuntimeError(
                "No feasible solution was found within the time limit."
            )

        state = copy.deepcopy(self.best_state)
        assignments = copy.deepcopy(self.best_assignments)

        while not self.timed_out(0.15):
            improved = False
            improved |= self.optimize_route_orders(state, assignments)
            if self.timed_out(0.15):
                break
            improved |= self.optimize_period_relocations(state, assignments)
            if self.timed_out(0.15):
                break
            improved |= self.optimize_joint_assignments(state, assignments)

            if not improved:
                break

        return self.solution_dict(
            self.best_state,
            self.best_assignments,
        )


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

    start = time.monotonic()
    deadline = start + max(0.1, float(args.time_limit))

    instance = load_instance(args.instance_path)
    solver = Solver(instance, logger, deadline)
    solution = solver.solve()

    output_directory = os.path.dirname(os.path.abspath(args.solution_path))
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)

    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()
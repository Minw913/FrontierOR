import argparse
import copy
import json
import math
import os
import random
import time

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception:
    gp = None
    GRB = None

from solution_logger import SolutionLogger


EPS = 1e-8


class Solver:
    def __init__(self, instance, time_limit, logger):
        self.inst = instance
        self.start_time = time.monotonic()
        self.deadline = self.start_time + max(0.05, float(time_limit))
        self.logger = logger
        self.rng = random.Random(0)

        self.n = int(instance["num_customers"])
        self.num_days = int(instance["num_days"])
        self.num_vehicles = int(instance["num_vehicles"])
        self.capacity = float(instance["vehicle_capacity"])
        self.max_duration = float(instance["max_route_duration"])

        self.depot = instance["depot"]
        self.customers = instance["customers"]
        self.customer_ids = [c["id"] for c in self.customers]

        all_visit_days = []
        for customer in self.customers:
            for schedule in customer["schedules"]:
                for part in schedule.get("parts", []):
                    all_visit_days.append(int(part["visit_day"]))

        self.zero_based_days = any(day == 0 for day in all_visit_days)
        if self.zero_based_days:
            self.days = list(range(self.num_days))
        else:
            self.days = list(range(1, self.num_days + 1))

        self.day_set = set(self.days)

        coords = [(float(self.depot["x"]), float(self.depot["y"]))]
        coords.extend((float(c["x"]), float(c["y"])) for c in self.customers)

        self.dist = [[0.0] * (self.n + 1) for _ in range(self.n + 1)]
        for i in range(self.n + 1):
            xi, yi = coords[i]
            for j in range(i + 1, self.n + 1):
                xj, yj = coords[j]
                value = math.hypot(xi - xj, yi - yj)
                self.dist[i][j] = value
                self.dist[j][i] = value

        self.schedule_data = []
        for i, customer in enumerate(self.customers):
            entries = []
            daily_demand = int(customer["demand_per_day"])

            for schedule_index, schedule in enumerate(customer["schedules"]):
                parts_by_day = {}
                valid = True

                for raw_part in schedule.get("parts", []):
                    day = int(raw_part["visit_day"])
                    days_covered = int(raw_part["days_covered"])
                    demand = int(raw_part.get(
                        "demand_per_visit", daily_demand * days_covered
                    ))

                    if day not in self.day_set or demand > self.capacity + EPS:
                        valid = False
                    if day in parts_by_day:
                        valid = False

                    parts_by_day[day] = {
                        "visit_day": day,
                        "days_covered": days_covered,
                        "demand_per_visit": demand,
                    }

                listed_days = [int(d) for d in schedule.get("days", [])]
                frequency = int(schedule.get("visit_frequency", len(parts_by_day)))

                if frequency != len(parts_by_day):
                    valid = False
                if sorted(listed_days) != sorted(parts_by_day.keys()):
                    valid = False

                entries.append({
                    "index": schedule_index,
                    "days": listed_days,
                    "frequency": frequency,
                    "parts": parts_by_day,
                    "valid": valid,
                })

            if not any(entry["valid"] for entry in entries):
                raise RuntimeError(
                    f"Customer {customer['id']} has no individually feasible schedule"
                )
            self.schedule_data.append(entries)

        self.best_assignment = None
        self.best_routes = None
        self.best_objective = float("inf")

    def time_left(self):
        return self.deadline - time.monotonic()

    def route_cost(self, nodes):
        if not nodes:
            return 0.0
        total = self.dist[0][nodes[0] + 1]
        for a, b in zip(nodes, nodes[1:]):
            total += self.dist[a + 1][b + 1]
        total += self.dist[nodes[-1] + 1][0]
        return total

    def forward_route(self, nodes, departure):
        current = float(departure)

        for customer_index in nodes:
            current += self.dist[0 if current == departure and customer_index == nodes[0]
                                 else 0][0] if False else 0.0

        current = float(departure)
        previous_location = 0
        service_starts = []

        for customer_index in nodes:
            customer = self.customers[customer_index]
            location = customer_index + 1
            current += self.dist[previous_location][location]

            earliest = float(customer["time_window"][0])
            latest = float(customer["time_window"][1])
            start = max(current, earliest)
            if start > latest + EPS:
                return None

            service_starts.append(start)
            current = start + float(customer["service_time"])
            previous_location = location

        current += self.dist[previous_location][0]
        return current, service_starts

    def route_timing_feasible(self, nodes):
        if not nodes:
            return True

        depot_open = float(self.depot["time_window"][0])
        depot_close = float(self.depot["time_window"][1])

        earliest_result = self.forward_route(nodes, depot_open)
        if earliest_result is None or earliest_result[0] > depot_close + EPS:
            return False

        lo = depot_open
        hi = depot_close

        for _ in range(45):
            mid = (lo + hi) * 0.5
            result = self.forward_route(nodes, mid)
            if result is not None and result[0] <= depot_close + EPS:
                lo = mid
            else:
                hi = mid

        result = self.forward_route(nodes, lo)
        if result is None or result[0] > depot_close + 1e-6:
            return False

        duration = result[0] - lo
        return duration <= self.max_duration + EPS

    def make_route(self, nodes, daily_parts):
        demand = sum(
            float(daily_parts[i]["demand_per_visit"]) for i in nodes
        )
        if demand > self.capacity + EPS:
            return None
        if not self.route_timing_feasible(nodes):
            return None
        return {
            "nodes": list(nodes),
            "cost": self.route_cost(nodes),
            "demand": demand,
        }

    def build_day_routes(self, daily_parts, randomized=False):
        if not daily_parts:
            return []

        total_demand = sum(float(p["demand_per_visit"])
                           for p in daily_parts.values())
        if total_demand > self.num_vehicles * self.capacity + EPS:
            return None

        unassigned = set(daily_parts.keys())
        routes = []

        while unassigned:
            if self.time_left() <= 0.02:
                return None

            item_choices = []

            for customer_index in unassigned:
                options = []

                for route_index, route in enumerate(routes):
                    if (route["demand"] +
                            float(daily_parts[customer_index]["demand_per_visit"])
                            > self.capacity + EPS):
                        continue

                    old_cost = route["cost"]
                    for position in range(len(route["nodes"]) + 1):
                        nodes = (
                            route["nodes"][:position]
                            + [customer_index]
                            + route["nodes"][position:]
                        )
                        candidate = self.make_route(nodes, daily_parts)
                        if candidate is not None:
                            options.append((
                                candidate["cost"] - old_cost,
                                route_index,
                                candidate
                            ))

                if len(routes) < self.num_vehicles:
                    candidate = self.make_route([customer_index], daily_parts)
                    if candidate is not None:
                        options.append((
                            candidate["cost"],
                            len(routes),
                            candidate
                        ))

                if not options:
                    return None

                options.sort(key=lambda x: x[0])
                best_delta = options[0][0]
                second_delta = options[1][0] if len(options) > 1 else best_delta + 1e6
                regret = second_delta - best_delta

                demand_ratio = (
                    float(daily_parts[customer_index]["demand_per_visit"])
                    / max(1.0, self.capacity)
                )
                radial = self.dist[0][customer_index + 1]
                priority = regret + 0.02 * radial + 0.5 * demand_ratio

                if randomized:
                    priority *= self.rng.uniform(0.92, 1.08)

                item_choices.append((priority, customer_index, options))

            item_choices.sort(key=lambda x: x[0], reverse=True)

            if randomized and len(item_choices) > 1:
                chosen = item_choices[
                    self.rng.randrange(min(3, len(item_choices)))
                ]
            else:
                chosen = item_choices[0]

            _, customer_index, options = chosen
            option_position = 0
            if randomized and len(options) > 1 and self.rng.random() < 0.16:
                option_position = self.rng.randrange(min(2, len(options)))

            _, route_index, candidate = options[option_position]
            if route_index == len(routes):
                routes.append(candidate)
            else:
                routes[route_index] = candidate

            unassigned.remove(customer_index)

        return self.improve_day_routes(routes, daily_parts)

    def improve_day_routes(self, routes, daily_parts):
        routes = copy.deepcopy(routes)

        # Intra-route 2-opt.
        changed = True
        passes = 0
        while changed and passes < 5 and self.time_left() > 0.03:
            changed = False
            passes += 1

            for r_idx, route in enumerate(routes):
                nodes = route["nodes"]
                for left in range(len(nodes) - 1):
                    if changed:
                        break
                    for right in range(left + 1, len(nodes)):
                        candidate_nodes = (
                            nodes[:left]
                            + list(reversed(nodes[left:right + 1]))
                            + nodes[right + 1:]
                        )
                        candidate = self.make_route(candidate_nodes, daily_parts)
                        if (candidate is not None and
                                candidate["cost"] < route["cost"] - 1e-8):
                            routes[r_idx] = candidate
                            changed = True
                            break

        # First-improvement inter-route relocations.
        moves = 0
        changed = True
        while changed and moves < 50 and self.time_left() > 0.04:
            changed = False

            for source_index in range(len(routes)):
                if changed:
                    break
                source = routes[source_index]

                for node_position, node in enumerate(source["nodes"]):
                    if changed:
                        break

                    reduced_nodes = (
                        source["nodes"][:node_position]
                        + source["nodes"][node_position + 1:]
                    )
                    reduced = (
                        self.make_route(reduced_nodes, daily_parts)
                        if reduced_nodes else None
                    )

                    for target_index in range(len(routes)):
                        if target_index == source_index:
                            continue
                        target = routes[target_index]

                        for insert_position in range(len(target["nodes"]) + 1):
                            enlarged_nodes = (
                                target["nodes"][:insert_position]
                                + [node]
                                + target["nodes"][insert_position:]
                            )
                            enlarged = self.make_route(enlarged_nodes, daily_parts)
                            if enlarged is None:
                                continue

                            old_cost = source["cost"] + target["cost"]
                            new_cost = enlarged["cost"]
                            if reduced is not None:
                                new_cost += reduced["cost"]

                            if new_cost < old_cost - 1e-8:
                                if reduced is None:
                                    new_routes = []
                                    for idx, existing in enumerate(routes):
                                        if idx == source_index:
                                            continue
                                        if idx == target_index:
                                            new_routes.append(enlarged)
                                        else:
                                            new_routes.append(existing)
                                    routes = new_routes
                                else:
                                    routes[source_index] = reduced
                                    routes[target_index] = enlarged
                                changed = True
                                moves += 1
                                break
                        if changed:
                            break

        return routes

    def assignment_parts(self, assignment):
        by_day = {day: {} for day in self.days}
        for customer_index, schedule_index in enumerate(assignment):
            schedule = self.schedule_data[customer_index][schedule_index]
            for day, part in schedule["parts"].items():
                by_day[day][customer_index] = part
        return by_day

    def assignment_necessary_feasible(self, assignment):
        loads = {day: 0.0 for day in self.days}
        visits = {day: 0 for day in self.days}

        for customer_index, schedule_index in enumerate(assignment):
            schedule = self.schedule_data[customer_index][schedule_index]
            if not schedule["valid"]:
                return False
            for day, part in schedule["parts"].items():
                demand = float(part["demand_per_visit"])
                if demand > self.capacity + EPS:
                    return False
                loads[day] += demand
                visits[day] += 1

        for day in self.days:
            if loads[day] > self.num_vehicles * self.capacity + EPS:
                return False
            if visits[day] > 0 and self.num_vehicles <= 0:
                return False
        return True

    def solve_assignment_mip(self):
        if gp is None or self.time_left() < 0.2:
            return []

        budget = min(20.0, max(0.1, 0.14 * self.time_left()))
        if budget <= 0.05:
            return []

        try:
            model = gp.Model()
            model.Params.OutputFlag = 0
            model.Params.Seed = 0
            model.Params.MIPGap = 1e-4
            model.Params.NumericFocus = 0
            model.Params.Threads = 1
            model.Params.TimeLimit = budget
            model.Params.PoolSolutions = 20
            model.Params.PoolSearchMode = 1

            x = {}
            for i, schedules in enumerate(self.schedule_data):
                for s, schedule in enumerate(schedules):
                    x[i, s] = model.addVar(
                        vtype=GRB.BINARY, name=f"x_{i}_{s}"
                    )
                    if not schedule["valid"]:
                        x[i, s].UB = 0.0

            for i, schedules in enumerate(self.schedule_data):
                model.addConstr(
                    gp.quicksum(x[i, s] for s in range(len(schedules))) == 1
                )

            load_vars = {}
            count_vars = {}
            for day in self.days:
                load_vars[day] = model.addVar(lb=0.0, name=f"load_{day}")
                count_vars[day] = model.addVar(lb=0.0, name=f"count_{day}")

                load_expr = gp.LinExpr()
                count_expr = gp.LinExpr()

                for i, schedules in enumerate(self.schedule_data):
                    for s, schedule in enumerate(schedules):
                        if day in schedule["parts"]:
                            demand = float(
                                schedule["parts"][day]["demand_per_visit"]
                            )
                            load_expr += demand * x[i, s]
                            count_expr += x[i, s]

                model.addConstr(load_vars[day] == load_expr)
                model.addConstr(count_vars[day] == count_expr)
                model.addConstr(
                    load_vars[day] <= self.num_vehicles * self.capacity
                )

            cap_scale = max(1.0, self.num_vehicles * self.capacity)
            count_scale = max(1.0, float(self.n))
            objective = gp.QuadExpr()

            for day in self.days:
                objective += (
                    load_vars[day] * load_vars[day] / (cap_scale * cap_scale)
                )
                objective += (
                    0.08 * count_vars[day] * count_vars[day]
                    / (count_scale * count_scale)
                )

            # Small deterministic tie-breaker.
            for i, schedules in enumerate(self.schedule_data):
                for s in range(len(schedules)):
                    objective += 1e-7 * (s + 1) * x[i, s]

            model.setObjective(objective, GRB.MINIMIZE)
            model.optimize()

            if model.SolCount <= 0:
                return []

            assignments = []
            seen = set()
            pool_count = min(model.SolCount, 20)

            for solution_number in range(pool_count):
                model.Params.SolutionNumber = solution_number
                assignment = []
                for i, schedules in enumerate(self.schedule_data):
                    selected = max(
                        range(len(schedules)),
                        key=lambda s: x[i, s].Xn
                    )
                    assignment.append(selected)

                key = tuple(assignment)
                if key not in seen and self.assignment_necessary_feasible(assignment):
                    seen.add(key)
                    assignments.append(assignment)

            return assignments

        except Exception:
            return []

    def greedy_assignment(self):
        assignment = [-1] * self.n
        loads = {day: 0.0 for day in self.days}
        counts = {day: 0 for day in self.days}

        order = list(range(self.n))
        self.rng.shuffle(order)
        order.sort(
            key=lambda i: max(
                sum(float(p["demand_per_visit"])
                    for p in schedule["parts"].values())
                for schedule in self.schedule_data[i]
                if schedule["valid"]
            ),
            reverse=True
        )

        for customer_index in order:
            candidates = []
            schedules = self.schedule_data[customer_index]

            for schedule_index, schedule in enumerate(schedules):
                if not schedule["valid"]:
                    continue

                valid = True
                score = 0.0

                for day in self.days:
                    extra = 0.0
                    extra_count = 0
                    if day in schedule["parts"]:
                        extra = float(
                            schedule["parts"][day]["demand_per_visit"]
                        )
                        extra_count = 1

                    new_load = loads[day] + extra
                    if new_load > self.num_vehicles * self.capacity + EPS:
                        valid = False
                        break

                    utilization = new_load / max(
                        1.0, self.num_vehicles * self.capacity
                    )
                    count_utilization = (
                        counts[day] + extra_count
                    ) / max(1.0, self.n)
                    score += utilization * utilization
                    score += 0.06 * count_utilization * count_utilization

                if valid:
                    score *= self.rng.uniform(0.94, 1.06)
                    candidates.append((score, schedule_index))

            if not candidates:
                return None

            candidates.sort()
            pick = 0
            if len(candidates) > 1 and self.rng.random() < 0.12:
                pick = 1

            schedule_index = candidates[pick][1]
            assignment[customer_index] = schedule_index

            for day, part in self.schedule_data[customer_index][
                    schedule_index]["parts"].items():
                loads[day] += float(part["demand_per_visit"])
                counts[day] += 1

        return assignment

    def evaluate_assignment(self, assignment, tries_per_day=2):
        if not self.assignment_necessary_feasible(assignment):
            return None

        parts_by_day = self.assignment_parts(assignment)
        routes_by_day = {}
        objective = 0.0

        day_order = sorted(
            self.days,
            key=lambda d: (
                sum(float(p["demand_per_visit"])
                    for p in parts_by_day[d].values()),
                len(parts_by_day[d])
            ),
            reverse=True
        )

        for day in day_order:
            if self.time_left() <= 0.02:
                return None

            best_routes = None
            best_cost = float("inf")
            attempts = max(1, tries_per_day)

            for attempt in range(attempts):
                routes = self.build_day_routes(
                    parts_by_day[day],
                    randomized=(attempt > 0)
                )
                if routes is None:
                    continue

                cost = sum(route["cost"] for route in routes)
                if cost < best_cost - 1e-9:
                    best_routes = routes
                    best_cost = cost

            if best_routes is None:
                return None

            routes_by_day[day] = best_routes
            objective += best_cost

        return objective, routes_by_day

    def evaluate_changed_days(self, old_assignment, old_routes,
                              new_assignment, changed_customers):
        old_parts = self.assignment_parts(old_assignment)
        new_parts = self.assignment_parts(new_assignment)

        changed_days = set()
        for customer_index in changed_customers:
            old_schedule = self.schedule_data[customer_index][
                old_assignment[customer_index]
            ]
            new_schedule = self.schedule_data[customer_index][
                new_assignment[customer_index]
            ]
            changed_days.update(old_schedule["parts"].keys())
            changed_days.update(new_schedule["parts"].keys())

        routes_by_day = copy.deepcopy(old_routes)

        for day in changed_days:
            if self.time_left() <= 0.02:
                return None

            best_routes = None
            best_cost = float("inf")
            attempts = 2

            for attempt in range(attempts):
                routes = self.build_day_routes(
                    new_parts[day],
                    randomized=(attempt > 0)
                )
                if routes is None:
                    continue
                cost = sum(route["cost"] for route in routes)
                if cost < best_cost - 1e-9:
                    best_routes = routes
                    best_cost = cost

            if best_routes is None:
                return None
            routes_by_day[day] = best_routes

        objective = sum(
            route["cost"]
            for day in self.days
            for route in routes_by_day.get(day, [])
        )
        return objective, routes_by_day

    def solution_dict(self, assignment, routes_by_day):
        parts_by_day = self.assignment_parts(assignment)

        output_routes = {}
        objective = 0.0

        for day in self.days:
            day_routes = []
            for route in routes_by_day.get(day, []):
                route_parts = {}
                for customer_index in route["nodes"]:
                    customer_id = self.customer_ids[customer_index]
                    part = parts_by_day[day][customer_index]
                    route_parts[str(customer_id)] = {
                        "visit_day": int(part["visit_day"]),
                        "days_covered": int(part["days_covered"]),
                        "demand_per_visit": int(part["demand_per_visit"]),
                    }

                cost = float(self.route_cost(route["nodes"]))
                demand = sum(
                    int(parts_by_day[day][i]["demand_per_visit"])
                    for i in route["nodes"]
                )
                objective += cost

                day_routes.append({
                    "customers": [
                        self.customer_ids[i] for i in route["nodes"]
                    ],
                    "cost": cost,
                    "demand": demand,
                    "parts": route_parts,
                })

            output_routes[str(day)] = day_routes

        output_schedules = {}
        for customer_index, schedule_index in enumerate(assignment):
            customer_id = self.customer_ids[customer_index]
            schedule = self.schedule_data[customer_index][schedule_index]
            output_schedules[str(customer_id)] = {
                "schedule_index": int(schedule_index),
                "days": [int(day) for day in schedule["days"]],
                "visit_frequency": int(schedule["frequency"]),
            }

        return {
            "objective_value": float(objective),
            "routes": output_routes,
            "schedules": output_schedules,
        }

    def register_incumbent(self, assignment, routes_by_day, objective=None):
        solution = self.solution_dict(assignment, routes_by_day)
        exact_objective = float(solution["objective_value"])

        if exact_objective < self.best_objective - 1e-8:
            self.best_objective = exact_objective
            self.best_assignment = list(assignment)
            self.best_routes = copy.deepcopy(routes_by_day)

            if self.logger:
                self.logger.log_solution(exact_objective, solution)
            return True
        return False

    def solve(self):
        candidates = self.solve_assignment_mip()
        seen = {tuple(a) for a in candidates}

        # Add diversified balanced assignments.
        target_candidates = 40
        while (len(candidates) < target_candidates and
               self.time_left() > 0.15):
            assignment = self.greedy_assignment()
            if assignment is None:
                continue
            key = tuple(assignment)
            if key not in seen:
                seen.add(key)
                candidates.append(assignment)

            if len(seen) > 200:
                break

        # Try promising schedule selections.
        for candidate_index, assignment in enumerate(candidates):
            if self.time_left() <= 0.06:
                break

            tries = 3 if candidate_index < 4 else 1
            result = self.evaluate_assignment(assignment, tries_per_day=tries)
            if result is None:
                continue

            objective, routes = result
            self.register_incumbent(assignment, routes, objective)

        # Continue generating assignments if no routing-feasible incumbent exists.
        attempts = 0
        while self.best_assignment is None and self.time_left() > 0.08:
            attempts += 1
            assignment = self.greedy_assignment()
            if assignment is None:
                continue

            result = self.evaluate_assignment(assignment, tries_per_day=2)
            if result is not None:
                objective, routes = result
                self.register_incumbent(assignment, routes, objective)

            if attempts > 1000:
                break

        if self.best_assignment is None:
            raise RuntimeError(
                "No feasible solution was found within the time limit"
            )

        # Schedule-pattern neighborhood search.
        current_assignment = list(self.best_assignment)
        current_routes = copy.deepcopy(self.best_routes)
        current_objective = self.best_objective
        iteration = 0

        mutable_customers = [
            i for i, schedules in enumerate(self.schedule_data)
            if sum(1 for s in schedules if s["valid"]) > 1
        ]

        while mutable_customers and self.time_left() > 0.08:
            iteration += 1
            new_assignment = list(current_assignment)

            mutation_count = 1
            if self.rng.random() < 0.18:
                mutation_count = min(2, len(mutable_customers))
            if self.rng.random() < 0.04:
                mutation_count = min(3, len(mutable_customers))

            changed = self.rng.sample(mutable_customers, mutation_count)

            actual_changes = []
            for customer_index in changed:
                alternatives = [
                    s for s, schedule in enumerate(
                        self.schedule_data[customer_index]
                    )
                    if schedule["valid"]
                    and s != new_assignment[customer_index]
                ]
                if alternatives:
                    new_assignment[customer_index] = self.rng.choice(alternatives)
                    actual_changes.append(customer_index)

            if not actual_changes:
                continue
            if not self.assignment_necessary_feasible(new_assignment):
                continue

            result = self.evaluate_changed_days(
                current_assignment,
                current_routes,
                new_assignment,
                actual_changes
            )
            if result is None:
                continue

            new_objective, new_routes = result
            elapsed_fraction = min(
                1.0,
                (time.monotonic() - self.start_time)
                / max(0.001, self.deadline - self.start_time)
            )
            temperature = max(
                1e-6,
                0.01 * max(1.0, current_objective)
                * (1.0 - elapsed_fraction)
            )

            accept = new_objective < current_objective - 1e-8
            if not accept:
                delta = new_objective - current_objective
                accept = self.rng.random() < math.exp(
                    -delta / max(temperature, 1e-9)
                )

            if accept:
                current_assignment = new_assignment
                current_routes = new_routes
                current_objective = new_objective

            if new_objective < self.best_objective - 1e-8:
                self.register_incumbent(
                    new_assignment, new_routes, new_objective
                )

            # Periodically restart from the best-known solution.
            if iteration % 30 == 0:
                current_assignment = list(self.best_assignment)
                current_routes = copy.deepcopy(self.best_routes)
                current_objective = self.best_objective

        return self.solution_dict(self.best_assignment, self.best_routes)


def write_json_atomic(path, data):
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)

    temporary_path = path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, allow_nan=False)
    os.replace(temporary_path, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    logger = (
        SolutionLogger(args.log_path, sense="minimize")
        if args.log_path else None
    )

    with open(args.instance_path, "r", encoding="utf-8") as handle:
        instance = json.load(handle)

    solver = Solver(instance, args.time_limit, logger)
    solution = solver.solve()
    write_json_atomic(args.solution_path, solution)


if __name__ == "__main__":
    main()
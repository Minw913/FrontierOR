import argparse
import copy
import json
import math
import os
import random
import time

from solution_logger import SolutionLogger


EPS = 1e-9


class Solver:
    def __init__(self, data, time_limit, logger):
        self.data = data
        self.logger = logger
        self.start_time = time.monotonic()
        self.deadline = self.start_time + max(0, time_limit)

        self.n = int(data["num_customers"])
        self.capacity = float(data["vehicle_capacity"])
        self.a = float(data["cost_function"]["a"])
        self.b = float(data["cost_function"]["b"])
        self.dist = data["distance_matrix"]

        self.depot = data["depot"]
        self.depot_id = int(self.depot["id"])
        self.virtual_depot = self.n + 1

        depot_tw = self.depot["time_window"]
        horizon = float(data.get("scheduling_horizon", depot_tw[1]))
        self.depot_earliest = float(depot_tw[0])
        self.depot_latest = min(float(depot_tw[1]), horizon)
        self.depot_service = float(self.depot.get("service_time", 0))

        self.customers = {}
        for c in data["customers"]:
            cid = int(c["id"])
            self.customers[cid] = {
                "id": cid,
                "x": float(c["x"]),
                "y": float(c["y"]),
                "demand": float(c["demand"]),
                "earliest": float(c["time_window"][0]),
                "latest": float(c["time_window"][1]),
                "service": float(c["service_time"]),
            }

        self.customer_ids = sorted(self.customers)
        self.best_routes = None
        self.best_objective = float("inf")
        self.rng = random.Random(0)

    def out_of_time(self, margin=0.0):
        return time.monotonic() >= self.deadline - margin

    def node_for_matrix(self, node):
        return self.depot_id if node == self.virtual_depot else node

    def distance(self, u, v):
        return float(self.dist[self.node_for_matrix(u)][self.node_for_matrix(v)])

    def route_load(self, route):
        return sum(route["q"].values())

    def route_cost(self, route):
        seq = route["seq"]
        q = route["q"]
        load = sum(q.values())
        cost = 0.0
        previous = self.depot_id

        for customer in seq:
            d = self.distance(previous, customer)
            cost += d * (self.a * load + self.b)
            load -= q[customer]
            if abs(load) < 1e-8:
                load = 0.0
            previous = customer

        d = self.distance(previous, self.depot_id)
        cost += d * (self.a * max(0.0, load) + self.b)
        return cost

    def total_cost(self, routes):
        return sum(self.route_cost(r) for r in routes)

    def sequence_feasible(self, seq):
        current_time = self.depot_earliest + self.depot_service
        previous = self.depot_id

        for customer in seq:
            c = self.customers[customer]
            arrival = current_time + self.distance(previous, customer)
            start = max(arrival, c["earliest"])
            if start > c["latest"] + 1e-8:
                return False
            current_time = start + c["service"]
            previous = customer

        return_time = current_time + self.distance(previous, self.depot_id)
        return return_time <= self.depot_latest + 1e-8

    def route_valid(self, route):
        if len(route["seq"]) != len(set(route["seq"])):
            return False
        if self.route_load(route) > self.capacity + 1e-7:
            return False
        if any(route["q"].get(j, 0.0) <= EPS for j in route["seq"]):
            return False
        return self.sequence_feasible(route["seq"])

    def best_insertion(self, route, customer, quantity):
        if customer in route["q"] or quantity <= EPS:
            return None

        old_cost = self.route_cost(route)
        best = None

        for pos in range(len(route["seq"]) + 1):
            seq = route["seq"][:pos] + [customer] + route["seq"][pos:]
            if not self.sequence_feasible(seq):
                continue
            q = dict(route["q"])
            q[customer] = quantity
            candidate = {"seq": seq, "q": q}
            cost = self.route_cost(candidate)
            if best is None or cost < best[0] - 1e-10:
                best = (cost, cost - old_cost, candidate)
        return best

    def singleton_routes(self):
        routes = []
        for customer in self.customer_ids:
            remaining = self.customers[customer]["demand"]
            if not self.sequence_feasible([customer]):
                raise RuntimeError(
                    f"Customer {customer} cannot be visited within its time window."
                )
            while remaining > EPS:
                quantity = min(self.capacity, remaining)
                routes.append({
                    "seq": [customer],
                    "q": {customer: quantity},
                })
                remaining -= quantity
        return routes

    def construct_from_order(self, order):
        routes = []

        for customer in order:
            remaining = self.customers[customer]["demand"]

            while remaining > EPS:
                if self.out_of_time(0.02):
                    return None

                best_choice = None

                for idx, route in enumerate(routes):
                    spare = self.capacity - self.route_load(route)
                    if spare <= EPS or customer in route["q"]:
                        continue

                    quantity = min(remaining, spare)
                    insertion = self.best_insertion(route, customer, quantity)
                    if insertion is None:
                        continue

                    new_cost, delta, candidate = insertion
                    score = delta / max(quantity, EPS)
                    choice = (score, delta, idx, quantity, candidate)
                    if best_choice is None or choice[:2] < best_choice[:2]:
                        best_choice = choice

                new_quantity = min(self.capacity, remaining)
                singleton = {
                    "seq": [customer],
                    "q": {customer: new_quantity},
                }
                singleton_cost = self.route_cost(singleton)
                singleton_score = singleton_cost / max(new_quantity, EPS)

                if best_choice is None or singleton_score < best_choice[0] - 1e-10:
                    routes.append(singleton)
                    remaining -= new_quantity
                else:
                    _, _, idx, quantity, candidate = best_choice
                    routes[idx] = candidate
                    remaining -= quantity

        return routes

    def solution_dict(self, routes, objective=None):
        output_routes = []

        for vehicle, route in enumerate(routes):
            seq = route["seq"]
            q = route["q"]
            load = sum(q.values())
            edges = []
            previous = self.depot_id

            for customer in seq:
                edges.append({
                    "from": int(previous),
                    "to": int(customer),
                    "load_weight": float(max(0.0, load)),
                })
                load -= q[customer]
                if abs(load) < 1e-8:
                    load = 0.0
                previous = customer

            edges.append({
                "from": int(previous),
                "to": int(self.virtual_depot),
                "load_weight": float(max(0.0, load)),
            })

            deliveries = {
                str(customer): float(q[customer])
                for customer in seq
            }

            output_routes.append({
                "vehicle": vehicle,
                "route": [int(self.depot_id)] + [int(x) for x in seq]
                         + [int(self.virtual_depot)],
                "deliveries": deliveries,
                "edges": edges,
            })

        if objective is None:
            objective = self.total_cost(routes)

        return {
            "objective_value": float(objective),
            "routes": output_routes,
        }

    def register(self, routes):
        objective = self.total_cost(routes)
        tolerance = 1e-8 * max(1.0, abs(self.best_objective)) \
            if math.isfinite(self.best_objective) else 0.0

        if objective < self.best_objective - tolerance:
            self.best_objective = objective
            self.best_routes = copy.deepcopy(routes)
            if self.logger:
                solution = self.solution_dict(self.best_routes, self.best_objective)
                self.logger.log_solution(self.best_objective, solution)
            return True
        return False

    def improve_route_ordering(self):
        improved_any = False

        while not self.out_of_time(0.02):
            found = False

            for ridx, route in enumerate(self.best_routes):
                if self.out_of_time(0.02):
                    return improved_any
                if len(route["seq"]) < 2:
                    continue

                old_cost = self.route_cost(route)
                seq = route["seq"]

                # Single-customer relocation within the route.
                for old_pos in range(len(seq)):
                    customer = seq[old_pos]
                    reduced = seq[:old_pos] + seq[old_pos + 1:]

                    for new_pos in range(len(reduced) + 1):
                        candidate_seq = reduced[:new_pos] + [customer] + reduced[new_pos:]
                        if candidate_seq == seq:
                            continue
                        if not self.sequence_feasible(candidate_seq):
                            continue

                        candidate = {
                            "seq": candidate_seq,
                            "q": dict(route["q"]),
                        }
                        candidate_cost = self.route_cost(candidate)

                        if candidate_cost < old_cost - 1e-8:
                            self.best_routes[ridx] = candidate
                            self.register(self.best_routes)
                            found = True
                            improved_any = True
                            break
                    if found:
                        break

                if found:
                    break

                # 2-opt reversals.
                length = len(seq)
                for left in range(length - 1):
                    if found:
                        break
                    for right in range(left + 1, length):
                        candidate_seq = (
                            seq[:left] + list(reversed(seq[left:right + 1]))
                            + seq[right + 1:]
                        )
                        if not self.sequence_feasible(candidate_seq):
                            continue
                        candidate = {
                            "seq": candidate_seq,
                            "q": dict(route["q"]),
                        }
                        candidate_cost = self.route_cost(candidate)
                        if candidate_cost < old_cost - 1e-8:
                            self.best_routes[ridx] = candidate
                            self.register(self.best_routes)
                            found = True
                            improved_any = True
                            break

                if found:
                    break

            if not found:
                break

        return improved_any

    @staticmethod
    def unique_concat(first, second):
        result = []
        seen = set()
        for x in first + second:
            if x not in seen:
                seen.add(x)
                result.append(x)
        return result

    def best_combination(self, route_a, route_b):
        if self.route_load(route_a) + self.route_load(route_b) > self.capacity + 1e-8:
            return None

        combined_q = dict(route_a["q"])
        for customer, quantity in route_b["q"].items():
            combined_q[customer] = combined_q.get(customer, 0.0) + quantity

        sequences = [
            self.unique_concat(route_a["seq"], route_b["seq"]),
            self.unique_concat(route_b["seq"], route_a["seq"]),
            self.unique_concat(route_a["seq"], list(reversed(route_b["seq"]))),
            self.unique_concat(route_b["seq"], list(reversed(route_a["seq"]))),
            self.unique_concat(list(reversed(route_a["seq"])), route_b["seq"]),
            self.unique_concat(list(reversed(route_b["seq"])), route_a["seq"]),
        ]

        best = None
        seen_sequences = set()
        for seq in sequences:
            key = tuple(seq)
            if key in seen_sequences:
                continue
            seen_sequences.add(key)
            if not self.sequence_feasible(seq):
                continue
            candidate = {"seq": seq, "q": dict(combined_q)}
            cost = self.route_cost(candidate)
            if best is None or cost < best[0]:
                best = (cost, candidate)

        # Greedily insert the nodes unique to one route into the other.
        for base, added in ((route_a, route_b), (route_b, route_a)):
            candidate = {
                "seq": list(base["seq"]),
                "q": dict(base["q"]),
            }

            feasible = True
            for customer in added["seq"]:
                quantity = added["q"][customer]
                if customer in candidate["q"]:
                    candidate["q"][customer] += quantity
                    continue

                insertion = self.best_insertion(candidate, customer, quantity)
                if insertion is None:
                    feasible = False
                    break
                candidate = insertion[2]

            if feasible and self.sequence_feasible(candidate["seq"]):
                cost = self.route_cost(candidate)
                if best is None or cost < best[0]:
                    best = (cost, candidate)

        return best

    def improve_merges(self):
        improved_any = False

        while not self.out_of_time(0.03):
            found = False
            route_count = len(self.best_routes)

            candidate_pairs = []
            for i in range(route_count):
                load_i = self.route_load(self.best_routes[i])
                if load_i >= self.capacity - EPS:
                    continue
                for j in range(i + 1, route_count):
                    if load_i + self.route_load(self.best_routes[j]) <= self.capacity + 1e-8:
                        candidate_pairs.append((i, j))

            self.rng.shuffle(candidate_pairs)

            for i, j in candidate_pairs:
                if self.out_of_time(0.03):
                    return improved_any

                route_a = self.best_routes[i]
                route_b = self.best_routes[j]
                old_cost = self.route_cost(route_a) + self.route_cost(route_b)
                combination = self.best_combination(route_a, route_b)

                if combination is None:
                    continue

                new_cost, candidate = combination
                if new_cost < old_cost - 1e-8:
                    new_routes = list(self.best_routes)
                    new_routes[i] = candidate
                    del new_routes[j]
                    self.best_routes = new_routes
                    self.register(self.best_routes)
                    improved_any = True
                    found = True
                    break

            if not found:
                break

        return improved_any

    def transfer_candidate(self, source, destination, customer):
        source_quantity = source["q"][customer]
        spare = self.capacity - self.route_load(destination)
        if spare <= EPS:
            return None

        transfer = min(source_quantity, spare)
        if transfer <= EPS:
            return None

        new_source_q = dict(source["q"])
        new_source_q[customer] -= transfer
        if new_source_q[customer] <= 1e-8:
            del new_source_q[customer]
            new_source_seq = [x for x in source["seq"] if x != customer]
        else:
            new_source_seq = list(source["seq"])

        new_source = {
            "seq": new_source_seq,
            "q": new_source_q,
        }

        if customer in destination["q"]:
            new_destination = {
                "seq": list(destination["seq"]),
                "q": dict(destination["q"]),
            }
            new_destination["q"][customer] += transfer
        else:
            insertion = self.best_insertion(destination, customer, transfer)
            if insertion is None:
                return None
            new_destination = insertion[2]

        if new_source["seq"] and not self.sequence_feasible(new_source["seq"]):
            return None

        old_cost = self.route_cost(source) + self.route_cost(destination)
        new_cost = self.route_cost(new_destination)
        if new_source["seq"]:
            new_cost += self.route_cost(new_source)

        return old_cost, new_cost, new_source, new_destination

    def improve_transfers(self):
        improved_any = False

        while not self.out_of_time(0.03):
            found = False
            route_count = len(self.best_routes)
            source_indices = list(range(route_count))
            self.rng.shuffle(source_indices)

            nonfull = [
                i for i, route in enumerate(self.best_routes)
                if self.route_load(route) < self.capacity - EPS
            ]
            self.rng.shuffle(nonfull)

            attempts = 0
            max_attempts = max(1000, min(10000, route_count * 40))

            for source_idx in source_indices:
                if found or self.out_of_time(0.03):
                    break
                source = self.best_routes[source_idx]
                customers = list(source["seq"])
                self.rng.shuffle(customers)

                for customer in customers:
                    if found:
                        break

                    destinations = nonfull
                    if len(destinations) > 30:
                        destinations = destinations[:30]

                    for destination_idx in destinations:
                        attempts += 1
                        if attempts > max_attempts or self.out_of_time(0.03):
                            break
                        if destination_idx == source_idx:
                            continue

                        destination = self.best_routes[destination_idx]
                        candidate = self.transfer_candidate(
                            source, destination, customer
                        )
                        if candidate is None:
                            continue

                        old_cost, new_cost, new_source, new_destination = candidate
                        if new_cost >= old_cost - 1e-8:
                            continue

                        new_routes = list(self.best_routes)
                        new_routes[destination_idx] = new_destination

                        if new_source["seq"]:
                            new_routes[source_idx] = new_source
                        else:
                            del new_routes[source_idx]

                        self.best_routes = new_routes
                        self.register(self.best_routes)
                        improved_any = True
                        found = True
                        break

                    if attempts > max_attempts:
                        break

                if attempts > max_attempts:
                    break

            if not found:
                break

        return improved_any

    def make_orders(self):
        depot_x = float(self.depot["x"])
        depot_y = float(self.depot["y"])

        def angle(customer):
            c = self.customers[customer]
            return math.atan2(c["y"] - depot_y, c["x"] - depot_x)

        def radial_distance(customer):
            c = self.customers[customer]
            return math.hypot(c["x"] - depot_x, c["y"] - depot_y)

        orders = [
            sorted(self.customer_ids,
                   key=lambda j: (self.customers[j]["latest"],
                                  self.customers[j]["earliest"], j)),
            sorted(self.customer_ids,
                   key=lambda j: (self.customers[j]["earliest"],
                                  self.customers[j]["latest"], j)),
            sorted(self.customer_ids, key=lambda j: (angle(j), radial_distance(j))),
            sorted(self.customer_ids, key=lambda j: (-angle(j), radial_distance(j))),
            sorted(self.customer_ids, key=lambda j: (radial_distance(j), angle(j))),
            sorted(self.customer_ids,
                   key=lambda j: (self.customers[j]["latest"], angle(j))),
        ]

        base = list(self.customer_ids)
        for _ in range(4):
            random_order = list(base)
            self.rng.shuffle(random_order)
            orders.append(random_order)

        unique_orders = []
        seen = set()
        for order in orders:
            key = tuple(order)
            if key not in seen:
                seen.add(key)
                unique_orders.append(order)
        return unique_orders

    def solve(self):
        # Guaranteed fallback under the usual assumption that every customer is
        # individually reachable within its time window.
        initial = self.singleton_routes()
        self.register(initial)

        construction_deadline = (
            self.start_time + 0.45 * max(0.0, self.deadline - self.start_time)
        )

        for order in self.make_orders():
            if time.monotonic() >= construction_deadline or self.out_of_time(0.03):
                break
            candidate = self.construct_from_order(order)
            if candidate is not None:
                self.register(candidate)

        # Alternate complementary local searches until no progress or timeout.
        while not self.out_of_time(0.05):
            changed = False
            changed |= self.improve_route_ordering()
            if self.out_of_time(0.05):
                break
            changed |= self.improve_merges()
            if self.out_of_time(0.05):
                break
            changed |= self.improve_transfers()
            if not changed:
                break

        return self.solution_dict(self.best_routes, self.best_objective)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    solver = Solver(data, args.time_limit, logger)
    solution = solver.solve()

    output_directory = os.path.dirname(os.path.abspath(args.solution_path))
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)

    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(solution, f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
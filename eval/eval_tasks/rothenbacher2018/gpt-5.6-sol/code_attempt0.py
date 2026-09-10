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


EPS = 1e-8


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
        self.time_limit = max(1, time_limit)
        self.logger = logger
        self.start_time = time.monotonic()
        self.random = random.Random(0)

        self.params = data["parameters"]
        self.horizon = self.params.get("planning_horizon", [0.0, float("inf")])
        self.horizon_start = float(self.horizon[0])
        self.horizon_end = float(self.horizon[1])
        self.max_duration = float(
            self.params.get(
                "max_route_duration",
                self.horizon_end - self.horizon_start,
            )
        )

        self.coupling_time = float(self.params.get("coupling_time", 0.0))
        self.decoupling_time = float(self.params.get("decoupling_time", 0.0))
        self.transfer_time = float(self.params.get("transfer_time", 0.0))
        self.coupling_cost = float(self.params.get("coupling_cost", 0.0))
        self.decoupling_cost = float(self.params.get("decoupling_cost", 0.0))
        self.transfer_cost = float(self.params.get("transfer_cost", 0.0))

        self.distance = data["distance_matrix"]
        self.travel_time = data["travel_time_matrix"]
        self.depot = int(data["depot"]["id"])

        self.truck_customers = {
            int(c["id"]): c for c in data.get("truck_customers", [])
        }
        self.trailer_customers = {
            int(c["id"]): c for c in data.get("trailer_customers", [])
        }
        self.customers = {}
        self.customers.update(self.truck_customers)
        self.customers.update(self.trailer_customers)
        self.customer_ids = sorted(self.customers)

        self.transshipment_ids = [
            int(x["id"]) for x in data.get("transshipment_locations", [])
        ]
        self.parking_ids = sorted(
            set(self.transshipment_ids) | set(self.trailer_customers)
        )

        fleet = data["fleet"]
        self.truck_types = {
            int(t["id"]): t for t in fleet.get("truck_types", [])
        }
        self.trailer_types = {
            int(t["id"]): t for t in fleet.get("trailer_types", [])
        }
        self.classes = {
            int(v["class_id"]): v for v in fleet.get("vehicle_classes", [])
        }

        self.candidates = []
        self.candidate_by_key = {}

    def elapsed(self):
        return time.monotonic() - self.start_time

    def remaining(self):
        return max(0.0, self.time_limit - self.elapsed())

    def supply(self, customer):
        return int(self.customers[customer].get("supply", 0))

    def service_time(self, customer):
        return float(
            self.customers[customer].get(
                "service_time", self.params.get("service_time", 0.0)
            )
        )

    def time_window(self, customer):
        record = self.customers[customer]
        tw = record.get("time_window")
        if tw is None:
            tw = record.get("time_windows")
        if tw is None:
            earliest = record.get(
                "earliest_time",
                record.get("ready_time", self.horizon_start),
            )
            latest = record.get(
                "latest_time",
                record.get("due_time", self.horizon_end),
            )
            return float(earliest), float(latest)
        if isinstance(tw, dict):
            earliest = tw.get("earliest", tw.get("start", self.horizon_start))
            latest = tw.get("latest", tw.get("end", self.horizon_end))
            return float(earliest), float(latest)
        return float(tw[0]), float(tw[1])

    def class_data(self, class_id):
        v = self.classes[class_id]
        truck_type = int(v["truck_type_id"])
        trailer_type = int(v.get("trailer_type_id", 0))
        truck_capacity = int(
            v.get("truck_capacity", self.truck_types[truck_type]["capacity"])
        )
        trailer_capacity = int(v.get("trailer_capacity", 0))
        return {
            "class_id": class_id,
            "truck_type": truck_type,
            "trailer_type": trailer_type,
            "truck_capacity": truck_capacity,
            "trailer_capacity": trailer_capacity,
            "total_capacity": int(
                v.get("total_capacity", truck_capacity + trailer_capacity)
            ),
            "fixed_cost": float(
                v.get(
                    "truck_fixed_cost",
                    self.truck_types[truck_type].get("fixed_cost", 0.0),
                )
            ),
            "truck_factor": float(
                v.get(
                    "truck_cost_factor",
                    self.truck_types[truck_type].get("cost_factor", 0.0),
                )
            ),
            "trailer_factor": float(
                v.get(
                    "trailer_towing_cost_multiplier",
                    self.trailer_types.get(trailer_type, {}).get(
                        "towing_cost_multiplier", 0.0
                    ),
                )
            ),
            "class_available": int(v.get("num_available", 10**9)),
        }

    def simulate(self, class_id, events):
        """
        Event actions:
          start, service, decouple, transfer, couple, end
        Each event is a dictionary with keys physical, action, and optionally customer.
        """
        cdata = self.class_data(class_id)
        has_trailer = cdata["trailer_type"] != 0
        attached = has_trailer
        truck_load = 0
        trailer_load = 0
        t = self.horizon_start
        cost = cdata["fixed_cost"]
        arrivals = []
        served = []
        seen = set()

        previous = None
        for event in events:
            physical = int(event["physical"])
            action = event["action"]

            if previous is not None:
                travel_t = float(self.travel_time[previous][physical])
                dist = float(self.distance[previous][physical])
                t += travel_t
                factor = cdata["truck_factor"]
                if attached:
                    factor += cdata["trailer_factor"]
                cost += dist * factor

            arrival = t
            arrivals.append(arrival)

            if action == "service":
                customer = int(event.get("customer", physical))
                if customer in seen:
                    return None
                if customer not in self.customers:
                    return None
                if customer in self.truck_customers and attached:
                    return None

                earliest, latest = self.time_window(customer)
                if t < earliest:
                    t = earliest
                if t > latest + EPS:
                    return None

                quantity = self.supply(customer)
                if attached:
                    put_trailer = min(
                        quantity, cdata["trailer_capacity"] - trailer_load
                    )
                    trailer_load += put_trailer
                    truck_load += quantity - put_trailer
                else:
                    truck_load += quantity

                if truck_load > cdata["truck_capacity"] + EPS:
                    return None
                if trailer_load > cdata["trailer_capacity"] + EPS:
                    return None

                t += self.service_time(customer)
                served.append(customer)
                seen.add(customer)

            elif action == "decouple":
                if not attached or not has_trailer:
                    return None
                attached = False
                t += self.decoupling_time
                cost += self.decoupling_cost

            elif action == "transfer":
                if attached or not has_trailer:
                    return None
                amount = min(
                    truck_load, cdata["trailer_capacity"] - trailer_load
                )
                if amount <= EPS:
                    return None
                truck_load -= amount
                trailer_load += amount
                t += self.transfer_time
                cost += self.transfer_cost

            elif action == "couple":
                if attached or not has_trailer:
                    return None
                attached = True
                t += self.coupling_time
                cost += self.coupling_cost

            elif action in ("start", "end"):
                pass
            else:
                return None

            if t > self.horizon_end + EPS:
                return None
            if t - self.horizon_start > self.max_duration + EPS:
                return None
            previous = physical

        if not events:
            return None
        if events[0]["physical"] != self.depot:
            return None
        if events[-1]["physical"] != self.depot:
            return None
        if has_trailer and not attached:
            return None
        if truck_load > cdata["truck_capacity"] + EPS:
            return None
        if trailer_load > cdata["trailer_capacity"] + EPS:
            return None

        return {
            "class_id": class_id,
            "events": events,
            "customers": tuple(sorted(served)),
            "cost": float(cost),
            "arrivals": arrivals,
            "finish_time": t,
        }

    def build_solo(self, class_id, order):
        if not order:
            return None
        events = [{"physical": self.depot, "action": "start"}]
        for customer in order:
            events.append(
                {
                    "physical": int(customer),
                    "action": "service",
                    "customer": int(customer),
                }
            )
        events.append({"physical": self.depot, "action": "end"})
        return self.simulate(class_id, events)

    def build_attached(self, class_id, order):
        if not order:
            return None
        if any(c not in self.trailer_customers for c in order):
            return None
        events = [{"physical": self.depot, "action": "start"}]
        for customer in order:
            events.append(
                {
                    "physical": int(customer),
                    "action": "service",
                    "customer": int(customer),
                }
            )
        events.append({"physical": self.depot, "action": "end"})
        return self.simulate(class_id, events)

    def build_parked(self, class_id, parking, detached_order, serve_parking):
        cdata = self.class_data(class_id)
        if cdata["trailer_type"] == 0 or not detached_order:
            return None
        if parking not in self.parking_ids:
            return None
        if serve_parking and parking not in self.trailer_customers:
            return None
        if parking in detached_order:
            return None
        if any(self.supply(c) > cdata["truck_capacity"] for c in detached_order):
            return None

        events = [{"physical": self.depot, "action": "start"}]

        truck_load = 0
        trailer_load = 0
        if serve_parking:
            events.append(
                {
                    "physical": parking,
                    "action": "service",
                    "customer": parking,
                }
            )
            q = self.supply(parking)
            to_trailer = min(q, cdata["trailer_capacity"])
            trailer_load += to_trailer
            truck_load += q - to_trailer

        events.append({"physical": parking, "action": "decouple"})

        for pos, customer in enumerate(detached_order):
            quantity = self.supply(customer)
            if truck_load + quantity > cdata["truck_capacity"]:
                movable = min(
                    truck_load, cdata["trailer_capacity"] - trailer_load
                )
                if movable <= 0:
                    return None
                events.append({"physical": parking, "action": "transfer"})
                truck_load -= movable
                trailer_load += movable

            if truck_load + quantity > cdata["truck_capacity"]:
                return None

            events.append(
                {
                    "physical": int(customer),
                    "action": "service",
                    "customer": int(customer),
                }
            )
            truck_load += quantity

        movable = min(truck_load, cdata["trailer_capacity"] - trailer_load)
        if movable > 0:
            events.append({"physical": parking, "action": "transfer"})
            truck_load -= movable
            trailer_load += movable

        events.append({"physical": parking, "action": "couple"})
        events.append({"physical": self.depot, "action": "end"})
        return self.simulate(class_id, events)

    def add_candidate(self, route):
        if route is None or not route["customers"]:
            return
        key = (route["class_id"], route["customers"])
        old_index = self.candidate_by_key.get(key)
        if old_index is None:
            self.candidate_by_key[key] = len(self.candidates)
            self.candidates.append(route)
        elif route["cost"] + EPS < self.candidates[old_index]["cost"]:
            self.candidates[old_index] = route

    def randomized_order(self, allowed, first, origin):
        remaining = set(allowed)
        order = []
        current = origin
        if first is not None and first in remaining:
            order.append(first)
            remaining.remove(first)
            current = first

        while remaining:
            ranked = sorted(
                remaining,
                key=lambda c: (
                    self.distance[current][c],
                    self.distance[c][self.depot],
                    c,
                ),
            )
            width = min(5, len(ranked))
            chosen = ranked[self.random.randrange(width)]
            order.append(chosen)
            remaining.remove(chosen)
            current = chosen
        return order

    def generate_prefixes(self, class_id, mode, allowed, generation_deadline):
        if not allowed:
            return
        allowed = list(allowed)
        cdata = self.class_data(class_id)
        starts = list(allowed)
        self.random.shuffle(starts)
        max_starts = min(len(starts), 80)

        for first in starts[:max_starts]:
            if time.monotonic() >= generation_deadline:
                return
            repetitions = 3 if len(allowed) <= 100 else 2
            for _ in range(repetitions):
                order = self.randomized_order(allowed, first, self.depot)
                prefix = []
                total = 0
                for customer in order:
                    total += self.supply(customer)
                    if total > cdata["total_capacity"]:
                        continue
                    prefix.append(customer)
                    if mode == "solo":
                        route = self.build_solo(class_id, prefix)
                    else:
                        route = self.build_attached(class_id, prefix)
                    self.add_candidate(route)
                    if time.monotonic() >= generation_deadline:
                        return

        # Deterministic angular sweep routes add useful alternatives.
        depot_record = self.data["depot"]
        dx = float(depot_record.get("x", 0.0))
        dy = float(depot_record.get("y", 0.0))
        angular = sorted(
            allowed,
            key=lambda c: math.atan2(
                float(self.customers[c].get("y", 0.0)) - dy,
                float(self.customers[c].get("x", 0.0)) - dx,
            ),
        )
        if angular:
            step = max(1, len(angular) // 20)
            for shift in range(0, len(angular), step):
                rotated = angular[shift:] + angular[:shift]
                prefix = []
                total = 0
                for customer in rotated:
                    q = self.supply(customer)
                    if total + q > cdata["total_capacity"]:
                        break
                    prefix.append(customer)
                    total += q
                    route = (
                        self.build_solo(class_id, prefix)
                        if mode == "solo"
                        else self.build_attached(class_id, prefix)
                    )
                    self.add_candidate(route)
                    if time.monotonic() >= generation_deadline:
                        return

    def generate_parked(self, class_id, generation_deadline):
        if not self.parking_ids:
            return
        cdata = self.class_data(class_id)
        feasible_detached = [
            c
            for c in self.customer_ids
            if self.supply(c) <= cdata["truck_capacity"]
        ]
        if not feasible_detached:
            return

        # Ensure singleton accessibility candidates.
        for customer in feasible_detached:
            nearest_parks = sorted(
                self.parking_ids,
                key=lambda p: (
                    self.distance[self.depot][p]
                    + self.distance[p][customer]
                    + self.distance[customer][p]
                    + self.distance[p][self.depot],
                    p,
                ),
            )[: min(4, len(self.parking_ids))]
            for parking in nearest_parks:
                self.add_candidate(
                    self.build_parked(
                        class_id, parking, [customer], serve_parking=False
                    )
                )
                if (
                    parking in self.trailer_customers
                    and parking != customer
                ):
                    self.add_candidate(
                        self.build_parked(
                            class_id, parking, [customer], serve_parking=True
                        )
                    )
                if time.monotonic() >= generation_deadline:
                    return

        starts = list(feasible_detached)
        self.random.shuffle(starts)
        max_starts = min(len(starts), 70)

        for first in starts[:max_starts]:
            if time.monotonic() >= generation_deadline:
                return
            parks = sorted(
                self.parking_ids,
                key=lambda p: (
                    self.distance[self.depot][p]
                    + self.distance[p][first]
                    + self.distance[first][p],
                    p,
                ),
            )[: min(3, len(self.parking_ids))]

            for parking in parks:
                allowed = [c for c in feasible_detached if c != parking]
                if first not in allowed:
                    continue

                for repetition in range(3):
                    order = self.randomized_order(allowed, first, parking)
                    prefix = []
                    supply_sum = 0
                    serve_options = [False]
                    if parking in self.trailer_customers:
                        serve_options.append(True)

                    for customer in order:
                        q = self.supply(customer)
                        if supply_sum + q > cdata["total_capacity"]:
                            continue
                        prefix.append(customer)
                        supply_sum += q

                        for serve_parking in serve_options:
                            total = supply_sum
                            if serve_parking:
                                total += self.supply(parking)
                            if total > cdata["total_capacity"]:
                                continue
                            route = self.build_parked(
                                class_id,
                                parking,
                                prefix,
                                serve_parking,
                            )
                            self.add_candidate(route)

                        if time.monotonic() >= generation_deadline:
                            return

    def generate_candidates(self):
        generation_budget = max(
            0.1,
            min(self.time_limit * 0.45, self.time_limit - 0.5),
        )
        deadline = self.start_time + generation_budget

        for class_id in sorted(self.classes):
            if time.monotonic() >= deadline:
                break
            cdata = self.class_data(class_id)
            if cdata["class_available"] <= 0:
                continue

            if cdata["trailer_type"] == 0:
                self.generate_prefixes(
                    class_id,
                    "solo",
                    self.customer_ids,
                    deadline,
                )
            else:
                self.generate_prefixes(
                    class_id,
                    "attached",
                    list(self.trailer_customers),
                    deadline,
                )
                self.generate_parked(class_id, deadline)

        # A final mandatory singleton pass is important for set-partition feasibility.
        for class_id in sorted(self.classes):
            cdata = self.class_data(class_id)
            if cdata["class_available"] <= 0:
                continue
            if cdata["trailer_type"] == 0:
                for customer in self.customer_ids:
                    self.add_candidate(self.build_solo(class_id, [customer]))
            else:
                for customer in self.trailer_customers:
                    self.add_candidate(
                        self.build_attached(class_id, [customer])
                    )
                if self.parking_ids:
                    for customer in self.customer_ids:
                        if self.supply(customer) > cdata["truck_capacity"]:
                            continue
                        parking = min(
                            self.parking_ids,
                            key=lambda p: (
                                self.distance[self.depot][p]
                                + self.distance[p][customer]
                                + self.distance[customer][p]
                                + self.distance[p][self.depot],
                                p,
                            ),
                        )
                        if parking != customer:
                            self.add_candidate(
                                self.build_parked(
                                    class_id,
                                    parking,
                                    [customer],
                                    False,
                                )
                            )

    def solution_from_indices(self, selected_indices, objective=None):
        selected = sorted(
            selected_indices,
            key=lambda i: (
                self.candidates[i]["class_id"],
                self.candidates[i]["customers"],
                i,
            ),
        )
        class_counts = defaultdict(int)
        routes_output = []

        for route_number, index in enumerate(selected):
            route = self.candidates[index]
            class_id = route["class_id"]
            vehicle_index = class_counts[class_id]
            class_counts[class_id] += 1

            physical_sequence = [
                int(event["physical"]) for event in route["events"]
            ]
            stop_ids = []
            arrival_times = {}
            # Negative identifiers are distinct stop-occurrence identifiers.
            base = (route_number + 1) * 1000000
            for position, arrival in enumerate(route["arrivals"]):
                stop_id = -(base + position + 1)
                stop_ids.append(stop_id)
                arrival_times[str(stop_id)] = float(arrival)

            decouple_locations = [
                int(e["physical"])
                for e in route["events"]
                if e["action"] == "decouple"
            ]
            couple_locations = [
                int(e["physical"])
                for e in route["events"]
                if e["action"] == "couple"
            ]

            routes_output.append(
                {
                    "vehicle_class": int(class_id),
                    "vehicle_index": int(vehicle_index),
                    "customers_served": [
                        int(c) for c in route["customers"]
                    ],
                    "route_sequence": stop_ids,
                    "route_sequence_physical": physical_sequence,
                    "decouple_locations": decouple_locations,
                    "couple_locations": couple_locations,
                    "arrival_times": arrival_times,
                }
            )

        if objective is None:
            objective = sum(self.candidates[i]["cost"] for i in selected)

        return {
            "objective_value": float(objective),
            "routes": routes_output,
        }

    def solve_master(self):
        if not self.customer_ids:
            solution = {"objective_value": 0.0, "routes": []}
            if self.logger:
                self.logger.log_solution(0.0, solution)
            return solution

        covering = defaultdict(list)
        by_class = defaultdict(list)
        by_truck_type = defaultdict(list)
        by_trailer_type = defaultdict(list)

        for i, route in enumerate(self.candidates):
            cdata = self.class_data(route["class_id"])
            by_class[route["class_id"]].append(i)
            by_truck_type[cdata["truck_type"]].append(i)
            if cdata["trailer_type"] != 0:
                by_trailer_type[cdata["trailer_type"]].append(i)
            for customer in route["customers"]:
                covering[customer].append(i)

        if any(not covering[c] for c in self.customer_ids):
            return None

        model = gp.Model("truck_trailer_routing")
        model.Params.OutputFlag = 0
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1
        model.Params.TimeLimit = max(0.05, self.remaining())

        x = model.addVars(
            len(self.candidates),
            vtype=GRB.BINARY,
            obj=[route["cost"] for route in self.candidates],
            name="route",
        )
        model.ModelSense = GRB.MINIMIZE

        for customer in self.customer_ids:
            model.addConstr(
                gp.quicksum(x[i] for i in covering[customer]) == 1,
                name=f"serve_{customer}",
            )

        for class_id, indices in by_class.items():
            availability = self.class_data(class_id)["class_available"]
            model.addConstr(
                gp.quicksum(x[i] for i in indices) <= availability,
                name=f"class_availability_{class_id}",
            )

        for truck_type, indices in by_truck_type.items():
            availability = int(
                self.truck_types[truck_type].get("num_available", 10**9)
            )
            model.addConstr(
                gp.quicksum(x[i] for i in indices) <= availability,
                name=f"truck_availability_{truck_type}",
            )

        for trailer_type, indices in by_trailer_type.items():
            availability = int(
                self.trailer_types[trailer_type].get(
                    "num_available", 10**9
                )
            )
            model.addConstr(
                gp.quicksum(x[i] for i in indices) <= availability,
                name=f"trailer_availability_{trailer_type}",
            )

        last_logged = [float("inf")]

        def callback(cb_model, where):
            if where != GRB.Callback.MIPSOL:
                return
            try:
                objective = cb_model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if objective >= last_logged[0] - 1e-7:
                    return
                values = cb_model.cbGetSolution(x)
                chosen = [
                    i for i in range(len(self.candidates))
                    if values[i] > 0.5
                ]
                solution = self.solution_from_indices(chosen, objective)
                last_logged[0] = objective
                if self.logger:
                    self.logger.log_solution(objective, solution)
            except Exception:
                # Logging must never interrupt optimization.
                pass

        model.optimize(callback)

        if model.SolCount <= 0:
            return None

        selected = [
            i for i in range(len(self.candidates)) if x[i].X > 0.5
        ]
        solution = self.solution_from_indices(selected, model.ObjVal)

        if self.logger and model.ObjVal < last_logged[0] - 1e-7:
            self.logger.log_solution(model.ObjVal, solution)
        return solution

    def solve(self):
        self.generate_candidates()
        return self.solve_master()


def main():
    args = parse_args()
    logger = (
        SolutionLogger(args.log_path, sense="minimize")
        if args.log_path
        else None
    )

    with open(args.instance_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    solver = Solver(data, args.time_limit, logger)
    solution = solver.solve()

    if solution is None:
        # This normally indicates that the supplied instance is infeasible or
        # that no feasible route survived the stated constraints.
        solution = {
            "objective_value": 0.0,
            "routes": [],
        }

    output_dir = os.path.dirname(os.path.abspath(args.solution_path))
    os.makedirs(output_dir, exist_ok=True)
    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(solution, f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
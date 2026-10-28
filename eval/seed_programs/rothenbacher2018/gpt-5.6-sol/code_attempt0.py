import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


EPS = 1e-9


@dataclass
class RouteCandidate:
    vehicle_class: int
    truck_type_id: int
    trailer_type_id: int
    customers: tuple
    route_sequence: tuple
    route_sequence_physical: tuple
    decouple_locations: tuple
    couple_locations: tuple
    arrival_times: dict
    cost: float


class Solver:
    def __init__(self, data, time_limit, logger):
        self.data = data
        self.time_limit = max(1, int(time_limit))
        self.logger = logger
        self.start_time = time.monotonic()
        self.rng = random.Random(0)

        self.params = data.get("parameters", {})
        self.fleet = data["fleet"]
        self.classes = {
            int(v["class_id"]): v for v in self.fleet.get("vehicle_classes", [])
        }
        self.truck_types = {
            int(v["id"]): v for v in self.fleet.get("truck_types", [])
        }
        self.trailer_types = {
            int(v["id"]): v for v in self.fleet.get("trailer_types", [])
        }

        self.depot = int(data["depot"]["id"])
        self.truck_customers = {
            int(c["id"]): c for c in data.get("truck_customers", [])
        }
        self.trailer_customers = {
            int(c["id"]): c for c in data.get("trailer_customers", [])
        }
        self.customers = dict(self.truck_customers)
        self.customers.update(self.trailer_customers)
        self.customer_ids = sorted(self.customers)

        self.transshipment_ids = [
            int(x["id"]) for x in data.get("transshipment_locations", [])
        ]

        # Parking locations are trailer customers followed by transshipment
        # locations, in their published order, with duplicates removed.
        published_parking = [
            int(c["id"]) for c in data.get("trailer_customers", [])
        ] + self.transshipment_ids
        seen = set()
        self.parking_ids = []
        for p in published_parking:
            if p not in seen:
                seen.add(p)
                self.parking_ids.append(p)

        self.distance = data["distance_matrix"]
        self.travel_time = data["travel_time_matrix"]
        self.n_orig = len(self.distance)
        self.parking_index = {p: i for i, p in enumerate(self.parking_ids)}

        horizon = self.params.get("planning_horizon", [0.0, float("inf")])
        self.horizon_start = float(horizon[0])
        self.horizon_end = float(horizon[1])
        self.max_route_duration = float(
            self.params.get(
                "max_route_duration",
                self.horizon_end - self.horizon_start
                if math.isfinite(self.horizon_end)
                else float("inf"),
            )
        )

        self.coupling_time = float(self.params.get("coupling_time", 0.0))
        self.decoupling_time = float(self.params.get("decoupling_time", 0.0))
        self.transfer_time = float(self.params.get("transfer_time", 0.0))
        self.coupling_cost = float(self.params.get("coupling_cost", 0.0))
        self.decoupling_cost = float(self.params.get("decoupling_cost", 0.0))
        self.transfer_cost = float(self.params.get("transfer_cost", 0.0))

        self.candidates = []
        self.candidate_keys = set()
        self.max_candidates = 12000

    def elapsed(self):
        return time.monotonic() - self.start_time

    def remaining(self):
        return max(0.0, self.time_limit - self.elapsed())

    def decouple_copy(self, parking):
        return self.n_orig + self.parking_index[parking]

    def couple_copy(self, parking):
        return self.n_orig + len(self.parking_ids) + self.parking_index[parking]

    def service_time(self, customer):
        record = self.customers[customer]
        return float(record.get("service_time", self.params.get("service_time", 0.0)))

    def customer_window(self, customer):
        c = self.customers[customer]
        if "time_window" in c:
            tw = c["time_window"]
            return float(tw[0]), float(tw[1])
        if "earliest" in c or "latest" in c:
            return (
                float(c.get("earliest", self.horizon_start)),
                float(c.get("latest", self.horizon_end)),
            )
        if "ready_time" in c or "due_time" in c:
            return (
                float(c.get("ready_time", self.horizon_start)),
                float(c.get("due_time", self.horizon_end)),
            )
        return self.horizon_start, self.horizon_end

    def class_parameters(self, class_id):
        vc = self.classes[class_id]
        truck_type_id = int(vc["truck_type_id"])
        trailer_type_id = int(vc.get("trailer_type_id", 0))
        return {
            "truck_type_id": truck_type_id,
            "trailer_type_id": trailer_type_id,
            "truck_capacity": float(vc.get(
                "truck_capacity", self.truck_types[truck_type_id]["capacity"]
            )),
            "trailer_capacity": float(vc.get("trailer_capacity", 0.0)),
            "truck_fixed_cost": float(vc.get(
                "truck_fixed_cost", self.truck_types[truck_type_id]["fixed_cost"]
            )),
            "truck_cost_factor": float(vc.get(
                "truck_cost_factor", self.truck_types[truck_type_id]["cost_factor"]
            )),
            "towing_multiplier": float(
                vc.get("trailer_towing_cost_multiplier", 0.0)
            ),
        }

    def evaluate_events(self, class_id, events):
        """
        Events are tuples (kind, physical_id, stop_id), where kind is one of:
        depot, customer, decouple, couple.
        """
        cp = self.class_parameters(class_id)
        has_trailer = cp["trailer_type_id"] != 0
        attached = has_trailer
        truck_load = 0.0
        trailer_load = 0.0

        t = self.horizon_start
        cost = cp["truck_fixed_cost"]
        arrival_times = {}
        customers = []
        decouple_locations = []
        couple_locations = []
        sequence = []
        physical_sequence = []

        prev_physical = None

        for pos, (kind, physical, stop_id) in enumerate(events):
            physical = int(physical)
            stop_id = int(stop_id)

            if prev_physical is not None:
                travel = float(self.travel_time[prev_physical][physical])
                dist = float(self.distance[prev_physical][physical])
                t += travel
                multiplier = cp["truck_cost_factor"]
                if attached:
                    multiplier += cp["towing_multiplier"]
                cost += multiplier * dist

            arrival_times[str(stop_id)] = float(t)
            sequence.append(stop_id)
            physical_sequence.append(physical)

            if kind == "customer":
                if physical not in self.customers:
                    return None
                if physical in customers:
                    return None
                if physical in self.truck_customers and attached:
                    return None

                earliest, latest = self.customer_window(physical)
                start_service = max(t, earliest)
                if start_service > latest + EPS:
                    return None
                t = start_service + self.service_time(physical)

                supply = float(self.customers[physical]["supply"])
                if attached:
                    trailer_room = cp["trailer_capacity"] - trailer_load
                    moved = min(supply, max(0.0, trailer_room))
                    trailer_load += moved
                    truck_load += supply - moved
                else:
                    truck_load += supply

                if truck_load > cp["truck_capacity"] + EPS:
                    return None
                if trailer_load > cp["trailer_capacity"] + EPS:
                    return None
                customers.append(physical)

            elif kind == "decouple":
                if not has_trailer or not attached:
                    return None
                t += self.decoupling_time
                cost += self.decoupling_cost
                attached = False
                decouple_locations.append(physical)

            elif kind == "couple":
                if not has_trailer or attached:
                    return None
                t += self.coupling_time
                cost += self.coupling_cost

                transferable = min(
                    truck_load, max(0.0, cp["trailer_capacity"] - trailer_load)
                )
                if transferable > EPS:
                    trailer_load += transferable
                    truck_load -= transferable
                    t += self.transfer_time
                    cost += self.transfer_cost
                attached = True
                couple_locations.append(physical)

            elif kind == "depot":
                if pos not in (0, len(events) - 1):
                    return None
            else:
                return None

            prev_physical = physical

        if events[0][1] != self.depot or events[-1][1] != self.depot:
            return None
        if has_trailer and not attached:
            return None
        if t > self.horizon_end + EPS:
            return None
        if t - self.horizon_start > self.max_route_duration + EPS:
            return None
        if not customers:
            return None

        return RouteCandidate(
            vehicle_class=int(class_id),
            truck_type_id=cp["truck_type_id"],
            trailer_type_id=cp["trailer_type_id"],
            customers=tuple(customers),
            route_sequence=tuple(sequence),
            route_sequence_physical=tuple(physical_sequence),
            decouple_locations=tuple(decouple_locations),
            couple_locations=tuple(couple_locations),
            arrival_times=arrival_times,
            cost=float(cost),
        )

    def add_candidate(self, candidate):
        if candidate is None or len(self.candidates) >= self.max_candidates:
            return False
        key = (
            candidate.vehicle_class,
            candidate.customers,
            candidate.route_sequence,
        )
        if key in self.candidate_keys:
            return False
        self.candidate_keys.add(key)
        self.candidates.append(candidate)
        return True

    def direct_events(self, sequence):
        events = [("depot", self.depot, self.depot)]
        events.extend(("customer", c, c) for c in sequence)
        events.append(("depot", self.depot, self.depot))
        return events

    def parking_events(self, parking, detached_sequence, attached_suffix):
        events = [("depot", self.depot, self.depot)]
        if parking in self.trailer_customers:
            events.append(("customer", parking, parking))

        events.append(("decouple", parking, self.decouple_copy(parking)))
        events.extend(("customer", c, c) for c in detached_sequence)
        events.append(("couple", parking, self.couple_copy(parking)))
        events.extend(("customer", c, c) for c in attached_suffix)
        events.append(("depot", self.depot, self.depot))
        return events

    def nearest_order(self, start, nodes, noise=0.0):
        remaining = set(nodes)
        order = []
        current = start
        while remaining:
            ranked = []
            for node in remaining:
                score = float(self.distance[current][node])
                if noise > 0:
                    score *= 1.0 + noise * self.rng.random()
                ranked.append((score, node))
            _, chosen = min(ranked)
            order.append(chosen)
            remaining.remove(chosen)
            current = chosen
        return order

    def generate_direct_candidates_for_class(self, class_id, generation_deadline):
        cp = self.class_parameters(class_id)
        has_trailer = cp["trailer_type_id"] != 0
        eligible = (
            list(self.trailer_customers)
            if has_trailer
            else list(self.customer_ids)
        )

        for c in eligible:
            self.add_candidate(
                self.evaluate_events(class_id, self.direct_events([c]))
            )

        if not eligible:
            return

        seeds = list(eligible)
        self.rng.shuffle(seeds)
        trials = min(max(8, len(seeds)), 80)

        for trial in range(trials):
            if time.monotonic() >= generation_deadline:
                return
            seed = seeds[trial % len(seeds)]
            unvisited = set(eligible)
            sequence = [seed]
            unvisited.remove(seed)

            candidate = self.evaluate_events(
                class_id, self.direct_events(sequence)
            )
            if candidate is None:
                continue
            self.add_candidate(candidate)

            while unvisited and len(self.candidates) < self.max_candidates:
                current = sequence[-1]
                ranked = []
                for c in unvisited:
                    score = (
                        float(self.distance[current][c])
                        + 0.35 * float(self.distance[c][self.depot])
                    )
                    score *= 1.0 + 0.35 * self.rng.random()
                    ranked.append((score, c))
                ranked.sort()

                chosen = None
                chosen_candidate = None
                for _, c in ranked[: min(16, len(ranked))]:
                    trial_sequence = sequence + [c]
                    tested = self.evaluate_events(
                        class_id, self.direct_events(trial_sequence)
                    )
                    if tested is not None:
                        chosen = c
                        chosen_candidate = tested
                        break

                if chosen is None:
                    break
                sequence.append(chosen)
                unvisited.remove(chosen)
                self.add_candidate(chosen_candidate)

            # Create reverse and depot-angle variants.
            if len(sequence) > 1:
                reversed_candidate = self.evaluate_events(
                    class_id, self.direct_events(list(reversed(sequence)))
                )
                self.add_candidate(reversed_candidate)

    def generate_parking_candidates_for_class(
        self, class_id, generation_deadline
    ):
        cp = self.class_parameters(class_id)
        if cp["trailer_type_id"] == 0 or not self.parking_ids:
            return
        if not self.truck_customers:
            return

        truck_nodes = list(self.truck_customers)
        seeds = list(truck_nodes)
        self.rng.shuffle(seeds)
        seeds = seeds[: min(len(seeds), 60)]

        for seed in seeds:
            if time.monotonic() >= generation_deadline:
                return

            parking_rank = sorted(
                self.parking_ids,
                key=lambda p: (
                    float(self.distance[self.depot][p])
                    + float(self.distance[p][seed])
                ),
            )
            parking_rank = parking_rank[: min(8, len(parking_rank))]

            for parking in parking_rank:
                if time.monotonic() >= generation_deadline:
                    return

                forbidden = {parking} if parking in self.customers else set()
                detached = [seed]
                if seed in forbidden:
                    continue

                base = self.evaluate_events(
                    class_id, self.parking_events(parking, detached, [])
                )
                if base is None:
                    continue
                self.add_candidate(base)

                remaining_truck = set(truck_nodes) - set(detached) - forbidden
                while remaining_truck:
                    current = detached[-1]
                    ranked = sorted(
                        remaining_truck,
                        key=lambda c: (
                            float(self.distance[current][c])
                            + 0.25 * float(self.distance[c][parking])
                        ) * (1.0 + 0.25 * self.rng.random()),
                    )
                    chosen = None
                    selected_candidate = None
                    for c in ranked[: min(14, len(ranked))]:
                        tested = self.evaluate_events(
                            class_id,
                            self.parking_events(parking, detached + [c], []),
                        )
                        if tested is not None:
                            chosen = c
                            selected_candidate = tested
                            break
                    if chosen is None:
                        break
                    detached.append(chosen)
                    remaining_truck.remove(chosen)
                    self.add_candidate(selected_candidate)

                # Add trailer customers after recoupling.
                served = set(detached)
                if parking in self.trailer_customers:
                    served.add(parking)
                suffix = []
                remaining_trailer = set(self.trailer_customers) - served
                current = parking
                while remaining_trailer:
                    ranked = sorted(
                        remaining_trailer,
                        key=lambda c: (
                            float(self.distance[current][c])
                            + 0.30 * float(self.distance[c][self.depot])
                        ) * (1.0 + 0.25 * self.rng.random()),
                    )
                    chosen = None
                    selected_candidate = None
                    for c in ranked[: min(14, len(ranked))]:
                        tested = self.evaluate_events(
                            class_id,
                            self.parking_events(
                                parking, detached, suffix + [c]
                            ),
                        )
                        if tested is not None:
                            chosen = c
                            selected_candidate = tested
                            break
                    if chosen is None:
                        break
                    suffix.append(chosen)
                    remaining_trailer.remove(chosen)
                    current = chosen
                    self.add_candidate(selected_candidate)

                # Also retain shorter detached routes with useful suffixes.
                if len(detached) > 1:
                    short_detached = detached[: max(1, len(detached) // 2)]
                    short_suffix = []
                    served_short = set(short_detached)
                    if parking in self.trailer_customers:
                        served_short.add(parking)
                    available = list(set(self.trailer_customers) - served_short)
                    for c in self.nearest_order(parking, available, 0.15):
                        tested = self.evaluate_events(
                            class_id,
                            self.parking_events(
                                parking, short_detached, short_suffix + [c]
                            ),
                        )
                        if tested is None:
                            continue
                        short_suffix.append(c)
                        self.add_candidate(tested)

    def generate_candidates(self):
        # Reserve most of the runtime for the set-partitioning MIP.
        generation_budget = min(
            max(0.75, 0.30 * self.time_limit),
            max(0.75, self.time_limit - 1.0),
            20.0,
        )
        deadline = self.start_time + generation_budget

        class_ids = sorted(self.classes)
        for class_id in class_ids:
            if time.monotonic() >= deadline:
                break
            self.generate_direct_candidates_for_class(class_id, deadline)

        for class_id in class_ids:
            if time.monotonic() >= deadline:
                break
            self.generate_parking_candidates_for_class(class_id, deadline)

    def route_to_output(self, route, vehicle_index):
        return {
            "vehicle_class": int(route.vehicle_class),
            "vehicle_index": int(vehicle_index),
            "customers_served": [int(c) for c in route.customers],
            "route_sequence": [int(x) for x in route.route_sequence],
            "route_sequence_physical": [
                int(x) for x in route.route_sequence_physical
            ],
            "decouple_locations": [
                int(x) for x in route.decouple_locations
            ],
            "couple_locations": [int(x) for x in route.couple_locations],
            "arrival_times": {
                str(k): float(v) for k, v in route.arrival_times.items()
            },
        }

    def make_solution(self, selected_indices):
        selected = [self.candidates[i] for i in selected_indices]
        selected.sort(
            key=lambda r: (
                r.vehicle_class,
                r.customers,
                r.route_sequence,
            )
        )
        class_count = {}
        routes = []
        objective = 0.0
        for route in selected:
            idx = class_count.get(route.vehicle_class, 0)
            class_count[route.vehicle_class] = idx + 1
            routes.append(self.route_to_output(route, idx))
            objective += route.cost
        return {
            "objective_value": float(objective),
            "routes": routes,
            "expanded_node_convention": (
                "Let n_orig be the number of physical locations and P be the "
                "instance parking-location IDs in their published list order. "
                "Physical nodes retain IDs 0..n_orig-1. The decouple copy of "
                "P[i] has ID n_orig+i, and the couple copy of P[i] has ID "
                "n_orig+len(P)+i. Both copies map back to physical location P[i]."
            ),
        }

    def greedy_fallback(self):
        uncovered = set(self.customer_ids)
        selected = []
        class_use = {}
        truck_use = {}
        trailer_use = {}

        while uncovered:
            best = None
            for i, route in enumerate(self.candidates):
                newly = uncovered.intersection(route.customers)
                if not newly:
                    continue
                if any(c not in uncovered for c in route.customers):
                    continue

                vc = self.classes[route.vehicle_class]
                class_limit = int(vc.get("num_available", 10**9))
                if class_use.get(route.vehicle_class, 0) >= class_limit:
                    continue

                truck_limit = int(
                    self.truck_types[route.truck_type_id]["num_available"]
                )
                if truck_use.get(route.truck_type_id, 0) >= truck_limit:
                    continue

                if route.trailer_type_id:
                    trailer_limit = int(
                        self.trailer_types[route.trailer_type_id]["num_available"]
                    )
                    if trailer_use.get(route.trailer_type_id, 0) >= trailer_limit:
                        continue

                score = route.cost / len(newly)
                if best is None or score < best[0]:
                    best = (score, i, route)

            if best is None:
                return None

            _, i, route = best
            selected.append(i)
            uncovered.difference_update(route.customers)
            class_use[route.vehicle_class] = (
                class_use.get(route.vehicle_class, 0) + 1
            )
            truck_use[route.truck_type_id] = (
                truck_use.get(route.truck_type_id, 0) + 1
            )
            if route.trailer_type_id:
                trailer_use[route.trailer_type_id] = (
                    trailer_use.get(route.trailer_type_id, 0) + 1
                )

        return selected

    def solve_master(self):
        if not self.customer_ids:
            return self.make_solution([])
        if not self.candidates:
            return None

        model = gp.Model("truck_trailer_route_selection")
        model.Params.OutputFlag = 0
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1
        model.Params.TimeLimit = max(0.1, self.remaining())

        x = model.addVars(
            len(self.candidates),
            vtype=GRB.BINARY,
            obj=[r.cost for r in self.candidates],
            name="route",
        )

        routes_by_customer = {c: [] for c in self.customer_ids}
        routes_by_class = {}
        routes_by_truck = {}
        routes_by_trailer = {}

        for i, route in enumerate(self.candidates):
            for c in route.customers:
                routes_by_customer[c].append(i)
            routes_by_class.setdefault(route.vehicle_class, []).append(i)
            routes_by_truck.setdefault(route.truck_type_id, []).append(i)
            if route.trailer_type_id:
                routes_by_trailer.setdefault(route.trailer_type_id, []).append(i)

        for c in self.customer_ids:
            idxs = routes_by_customer[c]
            if not idxs:
                return None
            model.addConstr(gp.quicksum(x[i] for i in idxs) == 1, name=f"serve_{c}")

        for class_id, idxs in routes_by_class.items():
            limit = int(self.classes[class_id].get("num_available", 10**9))
            model.addConstr(
                gp.quicksum(x[i] for i in idxs) <= limit,
                name=f"class_availability_{class_id}",
            )

        for truck_type_id, idxs in routes_by_truck.items():
            limit = int(self.truck_types[truck_type_id]["num_available"])
            model.addConstr(
                gp.quicksum(x[i] for i in idxs) <= limit,
                name=f"truck_availability_{truck_type_id}",
            )

        for trailer_type_id, idxs in routes_by_trailer.items():
            limit = int(self.trailer_types[trailer_type_id]["num_available"])
            model.addConstr(
                gp.quicksum(x[i] for i in idxs) <= limit,
                name=f"trailer_availability_{trailer_type_id}",
            )

        model.ModelSense = GRB.MINIMIZE

        fallback = self.greedy_fallback()
        if fallback is not None:
            fallback_set = set(fallback)
            for i in range(len(self.candidates)):
                x[i].Start = 1.0 if i in fallback_set else 0.0
            fallback_solution = self.make_solution(fallback)
            if self.logger:
                self.logger.log_solution(
                    fallback_solution["objective_value"], fallback_solution
                )

        best_logged = [float("inf")]

        def callback(cb_model, where):
            if where != GRB.Callback.MIPSOL:
                return
            try:
                obj = float(cb_model.cbGet(GRB.Callback.MIPSOL_OBJ))
                if obj >= best_logged[0] - 1e-7:
                    return
                values = cb_model.cbGetSolution([x[i] for i in range(len(self.candidates))])
                chosen = [i for i, value in enumerate(values) if value > 0.5]
                solution = self.make_solution(chosen)
                best_logged[0] = solution["objective_value"]
                if self.logger:
                    self.logger.log_solution(
                        solution["objective_value"], solution
                    )
            except Exception:
                pass

        model.optimize(callback)

        if model.SolCount <= 0:
            if fallback is not None:
                return self.make_solution(fallback)
            return None

        selected = [
            i for i in range(len(self.candidates)) if x[i].X > 0.5
        ]
        return self.make_solution(selected)

    def solve(self):
        self.generate_candidates()
        solution = self.solve_master()
        if solution is None:
            raise RuntimeError(
                "No feasible solution was found with the generated route pool."
            )
        return solution


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

    with open(args.instance_path, "r", encoding="utf-8") as f:
        instance = json.load(f)

    solver = Solver(instance, args.time_limit, logger)
    solution = solver.solve()

    output_path = Path(args.solution_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(solution, f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
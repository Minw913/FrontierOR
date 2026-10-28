import argparse
import json
import math
import os
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


EPS = 1e-7


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    return parser.parse_args()


def truncated_euclidean(x1, y1, x2, y2):
    value = math.hypot(x1 - x2, y1 - y2)
    return math.floor(value * 10.0 + 1e-9) / 10.0


class VRPTWSolver:
    def __init__(self, data, logger, deadline):
        self.data = data
        self.logger = logger
        self.deadline = deadline

        self.n = int(data["num_customers"])
        self.nodes = list(range(self.n + 1))
        self.customers = list(range(1, self.n + 1))
        self.capacity = float(data["vehicle_capacity"])

        depot = data["depot"]
        customer_records = {int(c["id"]): c for c in data["customers"]}

        self.records = {0: depot}
        self.records.update(customer_records)

        self.demand = {
            i: float(self.records[i].get("demand", 0)) for i in self.nodes
        }
        self.ready = {
            i: float(self.records[i].get("ready_time", 0)) for i in self.nodes
        }
        self.due = {
            i: float(self.records[i].get("due_date", 0)) for i in self.nodes
        }
        self.service = {
            i: float(self.records[i].get("service_time", 0)) for i in self.nodes
        }

        horizon = float(data.get("scheduling_horizon", self.due[0]))
        self.route_end = min(self.due[0], horizon)
        self.depot_start = self.ready[0]

        self.base_distance = [[0.0] * (self.n + 1) for _ in self.nodes]
        for i in self.nodes:
            ri = self.records[i]
            for j in self.nodes:
                if i == j:
                    continue
                rj = self.records[j]
                self.base_distance[i][j] = truncated_euclidean(
                    float(ri["x"]),
                    float(ri["y"]),
                    float(rj["x"]),
                    float(rj["y"]),
                )

        supplied = data["distance_matrix"]
        includes_service = bool(
            data.get("travel_time_includes_service_time", False)
        )

        self.travel = [[0.0] * (self.n + 1) for _ in self.nodes]
        for i in self.nodes:
            for j in self.nodes:
                if i == j:
                    continue

                raw = float(supplied[i][j])
                if i != 0 and not includes_service:
                    raw += self.service[i]

                # The ready-time adjustment applies to outbound travel from
                # customers, as specified in the problem statement.
                if i != 0:
                    raw = max(raw, self.ready[j] - self.due[i])

                self.travel[i][j] = raw

        self.arcs = self._build_feasible_arcs()
        self.arc_set = set(self.arcs)

    def _build_feasible_arcs(self):
        arcs = []

        for j in self.customers:
            if self.demand[j] > self.capacity + EPS:
                continue
            earliest_j = max(
                self.ready[j],
                self.depot_start + self.travel[0][j],
            )
            if (
                earliest_j <= self.due[j] + EPS
                and earliest_j + self.travel[j][0] <= self.route_end + EPS
            ):
                arcs.append((0, j))

        for i in self.customers:
            if self.demand[i] > self.capacity + EPS:
                continue
            if self.ready[i] + self.travel[i][0] <= self.route_end + EPS:
                arcs.append((i, 0))

        for i in self.customers:
            for j in self.customers:
                if i == j:
                    continue
                if self.demand[i] + self.demand[j] > self.capacity + EPS:
                    continue
                if self.ready[i] + self.travel[i][j] > self.due[j] + EPS:
                    continue
                if self.ready[j] + self.travel[j][0] > self.route_end + EPS:
                    continue
                arcs.append((i, j))

        return arcs

    def evaluate_route(self, route):
        """Return earliest feasible departure times and cumulative loads."""
        current_time = self.depot_start
        current_load = 0.0
        times = {}
        loads = {}
        previous = 0

        for customer in route:
            if (previous, customer) not in self.arc_set:
                return None

            current_time = max(
                self.ready[customer],
                current_time + self.travel[previous][customer],
            )
            if current_time > self.due[customer] + 1e-6:
                return None

            current_load += self.demand[customer]
            if current_load > self.capacity + 1e-6:
                return None

            times[customer] = current_time
            loads[customer] = current_load
            previous = customer

        if not route:
            return None
        if (previous, 0) not in self.arc_set:
            return None
        if current_time + self.travel[previous][0] > self.route_end + 1e-6:
            return None

        return times, loads

    def route_distance(self, route):
        previous = 0
        total = 0.0
        for customer in route:
            total += self.base_distance[previous][customer]
            previous = customer
        total += self.base_distance[previous][0]
        return total

    def make_solution(self, routes):
        departure_times = {}
        loads = {}
        output_routes = []
        total_distance = 0.0

        for route in routes:
            evaluation = self.evaluate_route(route)
            if evaluation is None:
                return None
            route_times, route_loads = evaluation

            output_routes.append([0] + list(route) + [0])
            total_distance += self.route_distance(route)

            for customer in route:
                departure_times[str(customer)] = round(
                    float(route_times[customer]), 6
                )
                load_value = route_loads[customer]
                if abs(load_value - round(load_value)) <= 1e-8:
                    loads[str(customer)] = int(round(load_value))
                else:
                    loads[str(customer)] = round(float(load_value), 6)

        if len(departure_times) != self.n:
            return None

        return {
            "objective_value": float(len(routes)),
            "num_vehicles": int(len(routes)),
            "routes": output_routes,
            "total_distance": round(float(total_distance), 6),
            "departure_times": departure_times,
            "loads": loads,
        }

    def log_solution(self, solution):
        if self.logger is not None and solution is not None:
            try:
                self.logger.log_solution(
                    solution["objective_value"], solution
                )
            except Exception:
                pass

    def initial_singleton_solution(self):
        routes = []
        for i in self.customers:
            if self.evaluate_route([i]) is None:
                raise RuntimeError(
                    f"Customer {i} cannot be served even by a singleton route"
                )
            routes.append([i])
        return routes

    def merge_heuristic(self, routes, heuristic_deadline):
        """
        Clarke-Wright-style route merging. Since vehicle count is the primary
        objective, even a distance-increasing merge is accepted when feasible.
        """
        routes = [list(r) for r in routes]

        while len(routes) > 1 and time.monotonic() < heuristic_deadline:
            best = None
            best_delta = float("inf")

            route_distances = [self.route_distance(r) for r in routes]

            for a in range(len(routes)):
                if time.monotonic() >= heuristic_deadline:
                    break
                for b in range(a + 1, len(routes)):
                    ra = routes[a]
                    rb = routes[b]

                    orientations_a = [ra]
                    reversed_a = list(reversed(ra))
                    if reversed_a != ra and self.evaluate_route(reversed_a) is not None:
                        orientations_a.append(reversed_a)

                    orientations_b = [rb]
                    reversed_b = list(reversed(rb))
                    if reversed_b != rb and self.evaluate_route(reversed_b) is not None:
                        orientations_b.append(reversed_b)

                    for oa in orientations_a:
                        for ob in orientations_b:
                            for candidate in (oa + ob, ob + oa):
                                if self.evaluate_route(candidate) is None:
                                    continue
                                delta = (
                                    self.route_distance(candidate)
                                    - route_distances[a]
                                    - route_distances[b]
                                )
                                if delta < best_delta - EPS:
                                    best_delta = delta
                                    best = (a, b, candidate)

            if best is None:
                break

            a, b, merged = best
            new_routes = []
            for idx, route in enumerate(routes):
                if idx not in (a, b):
                    new_routes.append(route)
            new_routes.append(merged)
            routes = new_routes

            self.log_solution(self.make_solution(routes))

        return routes

    def routes_from_selected_arcs(self, selected):
        successors = {}
        starts = []

        for i, j in selected:
            if i == 0:
                starts.append(j)
            else:
                if i in successors:
                    return None
                successors[i] = j

        routes = []
        visited = set()

        for start in starts:
            route = []
            current = start

            while current != 0:
                if current in visited or current not in self.customers:
                    return None
                visited.add(current)
                route.append(current)
                if current not in successors:
                    return None
                current = successors[current]

            routes.append(route)

        if visited != set(self.customers):
            return None

        return routes

    def solve_mip(self, incumbent_routes):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0.05:
            return incumbent_routes

        model = gp.Model("pickup_vrptw")
        model.Params.OutputFlag = 0
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1
        model.Params.TimeLimit = max(0.01, remaining)

        x = model.addVars(self.arcs, vtype=GRB.BINARY, name="x")
        t = model.addVars(
            self.customers,
            lb={i: self.ready[i] for i in self.customers},
            ub={i: self.due[i] for i in self.customers},
            vtype=GRB.CONTINUOUS,
            name="time",
        )
        load = model.addVars(
            self.customers,
            lb={i: self.demand[i] for i in self.customers},
            ub=self.capacity,
            vtype=GRB.CONTINUOUS,
            name="load",
        )

        outgoing = {i: [] for i in self.nodes}
        incoming = {i: [] for i in self.nodes}
        for i, j in self.arcs:
            outgoing[i].append((i, j))
            incoming[j].append((i, j))

        for i in self.customers:
            model.addConstr(
                gp.quicksum(x[a] for a in outgoing[i]) == 1,
                name=f"out_{i}",
            )
            model.addConstr(
                gp.quicksum(x[a] for a in incoming[i]) == 1,
                name=f"in_{i}",
            )

        vehicle_count = gp.quicksum(x[a] for a in outgoing[0])
        model.addConstr(
            vehicle_count == gp.quicksum(x[a] for a in incoming[0]),
            name="depot_balance",
        )

        total_demand = sum(self.demand[i] for i in self.customers)
        capacity_lb = int(math.ceil(total_demand / self.capacity - 1e-10))
        model.addConstr(vehicle_count >= capacity_lb, name="vehicle_lb")
        model.addConstr(
            vehicle_count <= len(incumbent_routes), name="incumbent_ub"
        )

        for i, j in self.arcs:
            if i != 0 and j != 0:
                time_m = max(
                    0.0,
                    self.due[i] + self.travel[i][j] - self.ready[j],
                )
                model.addConstr(
                    t[j]
                    >= t[i]
                    + self.travel[i][j]
                    - time_m * (1.0 - x[i, j]),
                    name=f"time_{i}_{j}",
                )

                load_m = self.capacity
                model.addConstr(
                    load[j]
                    >= load[i]
                    + self.demand[j]
                    - load_m * (1.0 - x[i, j]),
                    name=f"load_{i}_{j}",
                )

            elif i == 0:
                start_arrival = self.depot_start + self.travel[0][j]
                time_m = max(0.0, start_arrival - self.ready[j])
                model.addConstr(
                    t[j] >= start_arrival - time_m * (1.0 - x[0, j]),
                    name=f"start_{j}",
                )
                model.addConstr(
                    load[j] <= self.demand[j]
                    + self.capacity * (1.0 - x[0, j]),
                    name=f"start_load_{j}",
                )

            else:  # j == 0
                return_m = max(
                    0.0,
                    self.due[i] + self.travel[i][0] - self.route_end,
                )
                model.addConstr(
                    t[i] + self.travel[i][0]
                    <= self.route_end + return_m * (1.0 - x[i, 0]),
                    name=f"return_{i}",
                )

        max_distance = max(
            (
                self.base_distance[i][j]
                for i in self.nodes
                for j in self.nodes
                if i != j
            ),
            default=0.0,
        )
        # Any solution has at most 2*n used arcs. This coefficient makes one
        # fewer vehicle more valuable than any possible distance difference.
        vehicle_weight = 2.0 * max(1, self.n) * max_distance + 1.0

        distance_expr = gp.quicksum(
            self.base_distance[i][j] * x[i, j] for i, j in self.arcs
        )
        model.setObjective(
            vehicle_weight * vehicle_count + distance_expr,
            GRB.MINIMIZE,
        )

        # Warm start from the heuristic solution.
        selected_start_arcs = set()
        warm_times = {}
        warm_loads = {}

        for route in incumbent_routes:
            evaluation = self.evaluate_route(route)
            if evaluation is None:
                continue
            route_times, route_loads = evaluation
            warm_times.update(route_times)
            warm_loads.update(route_loads)

            previous = 0
            for customer in route:
                selected_start_arcs.add((previous, customer))
                previous = customer
            selected_start_arcs.add((previous, 0))

        for arc in self.arcs:
            x[arc].Start = 1.0 if arc in selected_start_arcs else 0.0
        for i in self.customers:
            if i in warm_times:
                t[i].Start = warm_times[i]
            if i in warm_loads:
                load[i].Start = warm_loads[i]

        best_routes_holder = {
            "routes": [list(r) for r in incumbent_routes],
            "scalar": (
                vehicle_weight * len(incumbent_routes)
                + sum(self.route_distance(r) for r in incumbent_routes)
            ),
        }

        def callback(cb_model, where):
            if where != GRB.Callback.MIPSOL:
                return
            try:
                objective = cb_model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if objective >= best_routes_holder["scalar"] - 1e-7:
                    return

                values = cb_model.cbGetSolution(x)
                selected = [
                    arc for arc in self.arcs if values[arc] > 0.5
                ]
                routes = self.routes_from_selected_arcs(selected)
                if routes is None:
                    return

                solution = self.make_solution(routes)
                if solution is None:
                    return

                best_routes_holder["scalar"] = objective
                best_routes_holder["routes"] = routes
                self.log_solution(solution)
            except Exception:
                return

        try:
            model.optimize(callback)
        except gp.GurobiError:
            return best_routes_holder["routes"]

        if model.SolCount > 0:
            try:
                selected = [
                    arc for arc in self.arcs if x[arc].X > 0.5
                ]
                routes = self.routes_from_selected_arcs(selected)
                if routes is not None and self.make_solution(routes) is not None:
                    candidate_score = (
                        vehicle_weight * len(routes)
                        + sum(self.route_distance(r) for r in routes)
                    )
                    if candidate_score < best_routes_holder["scalar"] + 1e-6:
                        best_routes_holder["routes"] = routes
            except (gp.GurobiError, AttributeError):
                pass

        return best_routes_holder["routes"]

    def solve(self, time_limit):
        routes = self.initial_singleton_solution()
        initial_solution = self.make_solution(routes)
        self.log_solution(initial_solution)

        now = time.monotonic()
        heuristic_budget = min(3.0, max(0.05, 0.12 * time_limit))
        heuristic_deadline = min(self.deadline, now + heuristic_budget)
        routes = self.merge_heuristic(routes, heuristic_deadline)

        routes = self.solve_mip(routes)
        solution = self.make_solution(routes)
        if solution is None:
            solution = initial_solution
        return solution


def write_json(path, value):
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2)


def main():
    args = parse_args()
    logger = (
        SolutionLogger(args.log_path, sense="minimize")
        if args.log_path
        else None
    )

    start = time.monotonic()
    # Reserve a small amount of time for extracting and writing the solution.
    usable_time = max(0.01, float(args.time_limit) - 0.10)
    deadline = start + usable_time

    with open(args.instance_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    solver = VRPTWSolver(data, logger, deadline)
    solution = solver.solve(max(0.01, float(args.time_limit)))
    write_json(args.solution_path, solution)


if __name__ == "__main__":
    main()
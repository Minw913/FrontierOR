import argparse
import heapq
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


class Solver:
    def __init__(self, instance, logger, deadline):
        self.data = instance
        self.logger = logger
        self.deadline = deadline

        self.n = int(instance["num_customers"])
        self.m = 2 * self.n
        self.capacity = int(instance["vehicle_capacity"])
        self.dist = instance["distance_matrix"]

        customers_by_id = {int(c["id"]): c for c in instance["customers"]}

        self.delivery = [0] * (self.n + 1)
        self.pickup = [0] * (self.n + 1)
        for i in range(1, self.n + 1):
            customer = customers_by_id[i]
            self.delivery[i] = int(customer["delivery_demand"])
            self.pickup[i] = int(customer["pickup_demand"])

        self.node_delivery = [0] * (self.m + 1)
        self.node_pickup = [0] * (self.m + 1)
        for i in range(1, self.n + 1):
            self.node_delivery[i] = self.delivery[i]
            self.node_pickup[self.n + i] = self.pickup[i]

        self.best_routes = None
        self.best_objective = math.inf

    def location(self, expanded_node):
        if expanded_node == 0:
            return 0
        if expanded_node <= self.n:
            return expanded_node
        return expanded_node - self.n

    def distance(self, a, b):
        return int(self.dist[self.location(a)][self.location(b)])

    def route_cost(self, route):
        if not route:
            return 0
        total = self.distance(0, route[0])
        for a, b in zip(route, route[1:]):
            total += self.distance(a, b)
        total += self.distance(route[-1], 0)
        return total

    def total_cost(self, routes):
        return sum(self.route_cost(r) for r in routes if r)

    def route_totals(self, route):
        delivery_total = sum(self.node_delivery[v] for v in route)
        pickup_total = sum(self.node_pickup[v] for v in route)
        return delivery_total, pickup_total

    def route_feasible(self, route):
        if not route:
            return True

        delivery_total, pickup_total = self.route_totals(route)
        if delivery_total > self.capacity or pickup_total > self.capacity:
            return False

        load = delivery_total
        if load < 0 or load > self.capacity:
            return False

        for node in route:
            load -= self.node_delivery[node]
            load += self.node_pickup[node]
            if load < 0 or load > self.capacity:
                return False

        return True

    def make_solution(self, routes, objective=None):
        clean_routes = [list(r) for r in routes if r]
        if objective is None:
            objective = self.total_cost(clean_routes)

        output_routes = []
        detailed_routes = []

        for route in clean_routes:
            full = [0] + route + [0]
            output_routes.append(full)

            details = [{
                "node_id": 0,
                "role": "depot",
                "original_customer_id": 0,
                "quantity": 0
            }]

            for node in route:
                if node <= self.n:
                    details.append({
                        "node_id": int(node),
                        "role": "linehaul",
                        "original_customer_id": int(node),
                        "quantity": int(self.delivery[node])
                    })
                else:
                    customer = node - self.n
                    details.append({
                        "node_id": int(node),
                        "role": "backhaul",
                        "original_customer_id": int(customer),
                        "quantity": int(self.pickup[customer])
                    })

            details.append({
                "node_id": 0,
                "role": "depot",
                "original_customer_id": 0,
                "quantity": 0
            })
            detailed_routes.append(details)

        return {
            "objective_value": float(objective),
            "routes": output_routes,
            "routes_detailed": detailed_routes
        }

    def record(self, routes, force=False):
        routes = [list(r) for r in routes if r]
        objective = self.total_cost(routes)

        if force or objective < self.best_objective - EPS:
            if objective < self.best_objective - EPS or self.best_routes is None:
                self.best_objective = float(objective)
                self.best_routes = routes
                solution = self.make_solution(routes, objective)
                if self.logger:
                    self.logger.log_solution(float(objective), solution)
                return True
        return False

    def merge_candidate(self, route_a, route_b):
        da, pa = self.route_totals(route_a)
        db, pb = self.route_totals(route_b)
        if da + db > self.capacity or pa + pb > self.capacity:
            return None

        candidates = []
        orientations_a = (route_a, list(reversed(route_a)))
        orientations_b = (route_b, list(reversed(route_b)))

        for a in orientations_a:
            for b in orientations_b:
                candidates.append(list(a) + list(b))
                candidates.append(list(b) + list(a))

        combined = list(route_a) + list(route_b)
        linehauls = [v for v in combined if v <= self.n]
        backhauls = [v for v in combined if v > self.n]
        candidates.append(linehauls + backhauls)

        # Nearest-neighbor ordering with all deliveries before pickups.
        nn_route = []
        current = 0
        remaining = set(linehauls)
        while remaining:
            nxt = min(remaining, key=lambda v: (self.distance(current, v), v))
            nn_route.append(nxt)
            remaining.remove(nxt)
            current = nxt
        remaining = set(backhauls)
        while remaining:
            nxt = min(remaining, key=lambda v: (self.distance(current, v), v))
            nn_route.append(nxt)
            remaining.remove(nxt)
            current = nxt
        candidates.append(nn_route)

        best = None
        best_cost = math.inf
        seen = set()
        for candidate in candidates:
            key = tuple(candidate)
            if key in seen:
                continue
            seen.add(key)
            if self.route_feasible(candidate):
                cost = self.route_cost(candidate)
                if cost < best_cost:
                    best_cost = cost
                    best = candidate

        return best

    def savings_construction(self):
        routes = {}
        active = set()
        next_id = 0

        for i in range(1, self.n + 1):
            # Deliver first and pick up second at the same physical location.
            route = [i, self.n + i]
            routes[next_id] = route
            active.add(next_id)
            next_id += 1

        self.record(list(routes.values()), force=True)

        heap = []
        ids = list(active)
        for pos, a in enumerate(ids):
            if time.monotonic() >= self.deadline:
                break
            for b in ids[pos + 1:]:
                merged = self.merge_candidate(routes[a], routes[b])
                if merged is None:
                    continue
                delta = (
                    self.route_cost(merged)
                    - self.route_cost(routes[a])
                    - self.route_cost(routes[b])
                )
                if delta < -EPS:
                    heapq.heappush(heap, (delta, a, b, merged))

        while heap and time.monotonic() < self.deadline:
            delta, a, b, merged = heapq.heappop(heap)
            if a not in active or b not in active:
                continue

            # Recompute because stored routes are immutable, but this also
            # protects against accidental stale candidates.
            candidate = self.merge_candidate(routes[a], routes[b])
            if candidate is None:
                continue
            actual_delta = (
                self.route_cost(candidate)
                - self.route_cost(routes[a])
                - self.route_cost(routes[b])
            )
            if actual_delta >= -EPS:
                continue

            active.remove(a)
            active.remove(b)
            new_id = next_id
            next_id += 1
            routes[new_id] = candidate
            active.add(new_id)

            current_routes = [routes[rid] for rid in active]
            self.record(current_routes)

            for other in list(active):
                if other == new_id:
                    continue
                new_candidate = self.merge_candidate(
                    routes[new_id], routes[other]
                )
                if new_candidate is None:
                    continue
                new_delta = (
                    self.route_cost(new_candidate)
                    - self.route_cost(routes[new_id])
                    - self.route_cost(routes[other])
                )
                if new_delta < -EPS:
                    aa, bb = sorted((new_id, other))
                    heapq.heappush(
                        heap, (new_delta, aa, bb, new_candidate)
                    )

        return [routes[rid] for rid in active]

    def improve_two_opt(self, routes, local_deadline):
        improved_any = False

        while time.monotonic() < local_deadline:
            best_delta = -EPS
            best_move = None

            for r_idx, route in enumerate(routes):
                length = len(route)
                if length < 3:
                    continue

                for i in range(length - 1):
                    prev_node = 0 if i == 0 else route[i - 1]
                    first = route[i]
                    for j in range(i + 1, length):
                        last = route[j]
                        next_node = 0 if j + 1 == length else route[j + 1]

                        delta = (
                            self.distance(prev_node, last)
                            + self.distance(first, next_node)
                            - self.distance(prev_node, first)
                            - self.distance(last, next_node)
                        )
                        if delta >= best_delta:
                            continue

                        candidate = (
                            route[:i]
                            + list(reversed(route[i:j + 1]))
                            + route[j + 1:]
                        )
                        if self.route_feasible(candidate):
                            best_delta = delta
                            best_move = (r_idx, candidate)

                    if time.monotonic() >= local_deadline:
                        break
                if time.monotonic() >= local_deadline:
                    break

            if best_move is None:
                break

            r_idx, candidate = best_move
            routes[r_idx] = candidate
            improved_any = True
            self.record(routes)

        return improved_any

    def improve_relocate(self, routes, local_deadline):
        improved_any = False

        while time.monotonic() < local_deadline:
            best_delta = -EPS
            best_move = None

            totals = [self.route_totals(r) for r in routes]

            for source_idx, source in enumerate(routes):
                if not source:
                    continue

                for source_pos, node in enumerate(source):
                    source_candidate = (
                        source[:source_pos] + source[source_pos + 1:]
                    )
                    if not self.route_feasible(source_candidate):
                        continue

                    source_prev = 0 if source_pos == 0 else source[source_pos - 1]
                    source_next = (
                        0 if source_pos + 1 == len(source)
                        else source[source_pos + 1]
                    )
                    removal_delta = (
                        self.distance(source_prev, source_next)
                        - self.distance(source_prev, node)
                        - self.distance(node, source_next)
                    )

                    for target_idx, target in enumerate(routes):
                        if target_idx == source_idx:
                            continue

                        td, tp = totals[target_idx]
                        if td + self.node_delivery[node] > self.capacity:
                            continue
                        if tp + self.node_pickup[node] > self.capacity:
                            continue

                        for insert_pos in range(len(target) + 1):
                            target_prev = (
                                0 if insert_pos == 0 else target[insert_pos - 1]
                            )
                            target_next = (
                                0 if insert_pos == len(target)
                                else target[insert_pos]
                            )
                            insertion_delta = (
                                self.distance(target_prev, node)
                                + self.distance(node, target_next)
                                - self.distance(target_prev, target_next)
                            )
                            delta = removal_delta + insertion_delta
                            if delta >= best_delta:
                                continue

                            target_candidate = (
                                target[:insert_pos]
                                + [node]
                                + target[insert_pos:]
                            )
                            if self.route_feasible(target_candidate):
                                best_delta = delta
                                best_move = (
                                    source_idx,
                                    target_idx,
                                    source_candidate,
                                    target_candidate
                                )

                        if time.monotonic() >= local_deadline:
                            break
                    if time.monotonic() >= local_deadline:
                        break
                if time.monotonic() >= local_deadline:
                    break

            if best_move is None:
                break

            source_idx, target_idx, source_new, target_new = best_move
            routes[source_idx] = source_new
            routes[target_idx] = target_new
            routes[:] = [r for r in routes if r]
            improved_any = True
            self.record(routes)

        return improved_any

    def improve_merges(self, routes, local_deadline):
        while time.monotonic() < local_deadline:
            best_delta = -EPS
            best_pair = None
            best_route = None

            for i in range(len(routes)):
                for j in range(i + 1, len(routes)):
                    candidate = self.merge_candidate(routes[i], routes[j])
                    if candidate is None:
                        continue
                    delta = (
                        self.route_cost(candidate)
                        - self.route_cost(routes[i])
                        - self.route_cost(routes[j])
                    )
                    if delta < best_delta:
                        best_delta = delta
                        best_pair = (i, j)
                        best_route = candidate

                if time.monotonic() >= local_deadline:
                    break

            if best_pair is None:
                break

            i, j = best_pair
            routes[i] = best_route
            del routes[j]
            self.record(routes)

    def local_search(self, routes, local_deadline):
        routes = [list(r) for r in routes if r]

        while time.monotonic() < local_deadline:
            before = self.total_cost(routes)
            self.improve_two_opt(routes, local_deadline)
            self.improve_relocate(routes, local_deadline)
            self.improve_merges(routes, local_deadline)
            after = self.total_cost(routes)
            if after >= before - EPS:
                break

        return routes

    def extract_mip_routes(self, x_values, z_values, vehicle_count):
        routes = []
        for v in range(vehicle_count):
            if z_values.get(v, 0.0) < 0.5:
                continue

            route = []
            current = 0
            seen = set()

            for _ in range(self.m + 1):
                next_node = None
                for j in range(0, self.m + 1):
                    if j == current:
                        continue
                    if x_values.get((v, current, j), 0.0) > 0.5:
                        next_node = j
                        break

                if next_node is None:
                    return None
                if next_node == 0:
                    break
                if next_node in seen:
                    return None

                seen.add(next_node)
                route.append(next_node)
                current = next_node
            else:
                return None

            if route:
                routes.append(route)

        covered = sorted(node for route in routes for node in route)
        if covered != list(range(1, self.m + 1)):
            return None
        if any(not self.route_feasible(route) for route in routes):
            return None
        return routes

    def run_mip(self):
        remaining = self.deadline - time.monotonic()
        if remaining <= 1.0 or self.best_routes is None:
            return

        warm_routes = [list(r) for r in self.best_routes]
        base_k = len(warm_routes)
        vehicle_count = min(self.n, base_k + (1 if base_k < self.n else 0))

        estimated_arc_vars = vehicle_count * self.m * (self.m + 1)
        if self.m > 120 or estimated_arc_vars > 350000:
            return

        model = gp.Model("mixed_delivery_pickup_vrp")
        model.Params.OutputFlag = 0
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1
        model.Params.TimeLimit = max(0.1, self.deadline - time.monotonic())

        vehicles = range(vehicle_count)
        nodes = range(1, self.m + 1)
        all_nodes = range(0, self.m + 1)

        z = model.addVars(vehicles, vtype=GRB.BINARY, name="z")
        y = model.addVars(vehicles, nodes, vtype=GRB.BINARY, name="y")

        arc_keys = [
            (v, i, j)
            for v in vehicles
            for i in all_nodes
            for j in all_nodes
            if i != j
        ]
        x = model.addVars(arc_keys, vtype=GRB.BINARY, name="x")
        load = model.addVars(
            vehicles, nodes, lb=0.0, ub=float(self.capacity),
            vtype=GRB.CONTINUOUS, name="load"
        )
        order = model.addVars(
            vehicles, nodes, lb=0.0, ub=float(self.m),
            vtype=GRB.CONTINUOUS, name="order"
        )

        model.setObjective(
            gp.quicksum(
                self.distance(i, j) * x[v, i, j]
                for v, i, j in arc_keys
            ),
            GRB.MINIMIZE
        )

        for node in nodes:
            model.addConstr(
                gp.quicksum(y[v, node] for v in vehicles) == 1
            )

        for v in vehicles:
            model.addConstr(
                gp.quicksum(x[v, 0, j] for j in nodes) == z[v]
            )
            model.addConstr(
                gp.quicksum(x[v, i, 0] for i in nodes) == z[v]
            )

            delivery_sum = gp.quicksum(
                self.node_delivery[node] * y[v, node] for node in nodes
            )
            pickup_sum = gp.quicksum(
                self.node_pickup[node] * y[v, node] for node in nodes
            )
            model.addConstr(delivery_sum <= self.capacity)
            model.addConstr(pickup_sum <= self.capacity)

            for node in nodes:
                model.addConstr(
                    gp.quicksum(
                        x[v, i, node] for i in all_nodes if i != node
                    ) == y[v, node]
                )
                model.addConstr(
                    gp.quicksum(
                        x[v, node, j] for j in all_nodes if j != node
                    ) == y[v, node]
                )
                model.addConstr(order[v, node] >= y[v, node])
                model.addConstr(order[v, node] <= self.m * y[v, node])
                model.addConstr(load[v, node] <= self.capacity * y[v, node])

            big_m = max(1.0, 3.0 * self.capacity)

            for j in nodes:
                rhs = (
                    delivery_sum
                    - self.node_delivery[j]
                    + self.node_pickup[j]
                )
                model.addConstr(
                    load[v, j] - rhs <= big_m * (1 - x[v, 0, j])
                )
                model.addConstr(
                    rhs - load[v, j] <= big_m * (1 - x[v, 0, j])
                )

            for i in nodes:
                for j in nodes:
                    if i == j:
                        continue
                    change = -self.node_delivery[j] + self.node_pickup[j]
                    model.addConstr(
                        load[v, j] - load[v, i] - change
                        <= big_m * (1 - x[v, i, j])
                    )
                    model.addConstr(
                        load[v, i] + change - load[v, j]
                        <= big_m * (1 - x[v, i, j])
                    )
                    model.addConstr(
                        order[v, j] >= order[v, i] + 1
                        - self.m * (1 - x[v, i, j])
                    )

        for v in range(vehicle_count - 1):
            model.addConstr(z[v] >= z[v + 1])

        # MIP start from the heuristic solution.
        for v in vehicles:
            z[v].Start = 1.0 if v < len(warm_routes) else 0.0
            for node in nodes:
                y[v, node].Start = (
                    1.0 if v < len(warm_routes)
                    and node in warm_routes[v] else 0.0
                )

        for key in arc_keys:
            x[key].Start = 0.0

        for v, route in enumerate(warm_routes):
            if v >= vehicle_count:
                break
            sequence = [0] + route + [0]
            for a, b in zip(sequence, sequence[1:]):
                x[v, a, b].Start = 1.0

            initial_load = sum(self.node_delivery[node] for node in route)
            current_load = initial_load
            for position, node in enumerate(route, start=1):
                current_load -= self.node_delivery[node]
                current_load += self.node_pickup[node]
                load[v, node].Start = current_load
                order[v, node].Start = position

        solver = self

        def callback(cb_model, where):
            if where != GRB.Callback.MIPSOL:
                return
            try:
                cb_obj = cb_model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if cb_obj >= solver.best_objective - EPS:
                    return

                x_vals_raw = cb_model.cbGetSolution(x)
                z_vals_raw = cb_model.cbGetSolution(z)
                x_vals = {key: float(value) for key, value in x_vals_raw.items()}
                z_vals = {key: float(value) for key, value in z_vals_raw.items()}

                candidate_routes = solver.extract_mip_routes(
                    x_vals, z_vals, vehicle_count
                )
                if candidate_routes is not None:
                    solver.record(candidate_routes)
            except Exception:
                # Logging must never interrupt optimization.
                pass

        try:
            model.optimize(callback)
        except gp.GurobiError:
            return

        if model.SolCount > 0:
            x_vals = {key: float(x[key].X) for key in arc_keys}
            z_vals = {v: float(z[v].X) for v in vehicles}
            candidate_routes = self.extract_mip_routes(
                x_vals, z_vals, vehicle_count
            )
            if candidate_routes is not None:
                self.record(candidate_routes)

    def solve(self, start_time, time_limit):
        initial_routes = self.savings_construction()

        remaining = max(0.0, self.deadline - time.monotonic())
        if remaining > 0.2:
            estimated_k = len(initial_routes)
            estimated_vars = estimated_k * self.m * (self.m + 1)
            mip_likely = self.m <= 120 and estimated_vars <= 350000

            if mip_likely:
                local_budget = min(
                    remaining * 0.30,
                    5.0,
                    max(0.1, remaining - 1.0)
                )
            else:
                local_budget = max(0.0, remaining - 0.1)

            local_deadline = min(
                self.deadline,
                time.monotonic() + local_budget
            )
            improved_routes = self.local_search(
                initial_routes, local_deadline
            )
            self.record(improved_routes)

        if time.monotonic() < self.deadline - 0.2:
            self.run_mip()

        return self.make_solution(
            self.best_routes,
            self.best_objective
        )


def main():
    args = parse_args()
    logger = SolutionLogger(
        args.log_path, sense="minimize"
    ) if args.log_path else None

    start_time = time.monotonic()
    deadline = start_time + max(0, args.time_limit)

    with open(args.instance_path, "r", encoding="utf-8") as f:
        instance = json.load(f)

    solver = Solver(instance, logger, deadline)
    solution = solver.solve(start_time, args.time_limit)

    solution_dir = os.path.dirname(os.path.abspath(args.solution_path))
    if solution_dir:
        os.makedirs(solution_dir, exist_ok=True)

    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()
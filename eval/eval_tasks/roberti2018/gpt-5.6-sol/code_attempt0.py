import argparse
import json
import math
import os
import random
import time

from solution_logger import SolutionLogger

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception:
    gp = None
    GRB = None


EPS = 1e-9


def euclidean(a, b):
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


class Solver:
    def __init__(self, data, time_limit, logger):
        self.data = data
        self.start_time = time.monotonic()
        self.deadline = self.start_time + max(0.05, time_limit)
        self.time_limit = time_limit
        self.logger = logger
        self.rng = random.Random(0)

        self.Q1 = int(data["Q1"])
        self.Q2 = int(data["Q2"])
        self.T1 = int(data["T1"])
        self.T2 = int(data["T2"])

        self.depot = data["depot"]
        self.depot_id = self.depot["id"]

        self.satellites = list(data["satellites"])
        self.customers = list(data["customers"])
        self.groups_input = list(data["groups"])

        self.sat_ids = [s["id"] for s in self.satellites]
        self.customer_ids = [c["id"] for c in self.customers]
        self.sat_by_id = {s["id"]: s for s in self.satellites}
        self.customer_by_id = {c["id"]: c for c in self.customers}

        self.group_ids = [g["group_id"] for g in self.groups_input]
        self.group_customers = {
            g["group_id"]: list(g["customer_ids"]) for g in self.groups_input
        }

        # Be tolerant of group lists whose ordering differs from customer records.
        for c in self.customers:
            self.group_customers.setdefault(c["group"], [])
            if c["id"] not in self.group_customers[c["group"]]:
                self.group_customers[c["group"]].append(c["id"])
        self.group_ids = list(dict.fromkeys(
            self.group_ids + [c["group"] for c in self.customers]
        ))

        self.group_demand = {
            g: sum(int(self.customer_by_id[c]["demand"])
                   for c in self.group_customers[g])
            for g in self.group_ids
        }

        self.nodes = {self.depot_id: self.depot}
        self.nodes.update(self.sat_by_id)
        self.nodes.update(self.customer_by_id)

        ids = list(self.nodes)
        self.dist = {
            i: {j: euclidean(self.nodes[i], self.nodes[j]) for j in ids}
            for i in ids
        }

        self.total_demand = sum(int(c["demand"]) for c in self.customers)
        self.best_solution = None
        self.best_objective = float("inf")

    def remaining(self):
        return self.deadline - time.monotonic()

    def out_of_time(self, reserve=0.0):
        return self.remaining() <= reserve

    def route_cost(self, route):
        return sum(self.dist[route[i]][route[i + 1]]
                   for i in range(len(route) - 1))

    def improve_tour(self, center, customers, intensive=True):
        """Nearest-neighbor tour followed by bounded 2-opt."""
        if not customers:
            return [center, center]
        unvisited = set(customers)
        order = []
        cur = center
        while unvisited:
            nxt = min(unvisited, key=lambda x: self.dist[cur][x])
            order.append(nxt)
            unvisited.remove(nxt)
            cur = nxt

        route = [center] + order + [center]
        if len(order) <= 2:
            return route

        max_passes = 20 if intensive else 5
        for _ in range(max_passes):
            if self.out_of_time(0.03):
                break
            best_delta = -EPS
            best_pair = None
            n = len(route)
            for i in range(1, n - 2):
                a, b = route[i - 1], route[i]
                for k in range(i + 1, n - 1):
                    c, d = route[k], route[k + 1]
                    delta = (self.dist[a][c] + self.dist[b][d]
                             - self.dist[a][b] - self.dist[c][d])
                    if delta < best_delta:
                        best_delta = delta
                        best_pair = (i, k)
            if best_pair is None:
                break
            i, k = best_pair
            route[i:k + 1] = reversed(route[i:k + 1])
        return route

    def best_fit_decreasing(self, customer_ids):
        bins = []
        loads = []
        ordered = sorted(
            customer_ids,
            key=lambda c: (-int(self.customer_by_id[c]["demand"]), c)
        )
        for cid in ordered:
            demand = int(self.customer_by_id[cid]["demand"])
            if demand > self.Q2:
                return None
            choices = [
                (self.Q2 - loads[k] - demand, k)
                for k in range(len(bins))
                if loads[k] + demand <= self.Q2
            ]
            if choices:
                _, k = min(choices)
                bins[k].append(cid)
                loads[k] += demand
            else:
                bins.append([cid])
                loads.append(demand)
        return bins

    def sweep_partitions(self, sid, customer_ids):
        if not customer_ids:
            return [[]]
        sat = self.sat_by_id[sid]
        ordered = sorted(
            customer_ids,
            key=lambda cid: math.atan2(
                self.customer_by_id[cid]["y"] - sat["y"],
                self.customer_by_id[cid]["x"] - sat["x"]
            )
        )
        n = len(ordered)
        offsets = sorted(set([0, n // 4, n // 2, (3 * n) // 4]))
        results = []
        for reverse in (False, True):
            base = list(reversed(ordered)) if reverse else ordered
            for off in offsets:
                seq = base[off:] + base[:off]
                bins, cur, load = [], [], 0
                feasible = True
                for cid in seq:
                    d = int(self.customer_by_id[cid]["demand"])
                    if d > self.Q2:
                        feasible = False
                        break
                    if cur and load + d > self.Q2:
                        bins.append(cur)
                        cur, load = [], 0
                    cur.append(cid)
                    load += d
                if cur:
                    bins.append(cur)
                if feasible:
                    results.append(bins)
        return results

    def clarke_wright_partition(self, sid, customer_ids):
        if any(int(self.customer_by_id[c]["demand"]) > self.Q2
               for c in customer_ids):
            return None
        routes = {c: [c] for c in customer_ids}
        load = {c: int(self.customer_by_id[c]["demand"]) for c in customer_ids}
        owner = {c: c for c in customer_ids}

        savings = []
        ids = list(customer_ids)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                saving = self.dist[sid][a] + self.dist[sid][b] - self.dist[a][b]
                savings.append((saving, a, b))
        savings.sort(reverse=True)

        for _, a, b in savings:
            ra, rb = owner[a], owner[b]
            if ra == rb or load[ra] + load[rb] > self.Q2:
                continue
            A, B = routes[ra], routes[rb]
            if a not in (A[0], A[-1]) or b not in (B[0], B[-1]):
                continue
            if A[-1] != a:
                A = list(reversed(A))
            if B[0] != b:
                B = list(reversed(B))
            merged = A + B
            routes[ra] = merged
            load[ra] += load[rb]
            for c in B:
                owner[c] = ra
            del routes[rb]
            del load[rb]

        return list(routes.values())

    def satellite_route_options(self, sid, customer_ids, intensive=False):
        if not customer_ids:
            return [(0, 0.0, [])]

        raw = []
        bfd = self.best_fit_decreasing(customer_ids)
        if bfd is not None:
            raw.append(bfd)
        cw = self.clarke_wright_partition(sid, customer_ids)
        if cw is not None:
            raw.append(cw)
        raw.extend(self.sweep_partitions(sid, customer_ids))

        max_vehicles = min(
            int(self.sat_by_id[sid]["max_vehicles"]),
            self.T2,
            len(customer_ids)
        )
        seen = set()
        options = []

        for partition in raw:
            if not partition or len(partition) > max_vehicles:
                continue
            if any(sum(int(self.customer_by_id[c]["demand"]) for c in part)
                   > self.Q2 for part in partition):
                continue
            canonical = tuple(sorted(tuple(sorted(part)) for part in partition))
            if canonical in seen:
                continue
            seen.add(canonical)

            routes = [
                self.improve_tour(sid, part, intensive=intensive)
                for part in partition
            ]
            cost = sum(self.route_cost(r) for r in routes)
            options.append((len(routes), cost, routes))

        # Keep the cheapest option for each vehicle count.
        by_count = {}
        for option in options:
            k = option[0]
            if k not in by_count or option[1] < by_count[k][1] - EPS:
                by_count[k] = option
        return list(by_count.values())

    def construct_second_echelon(self, assignment, intensive=False,
                                 route_bins_override=None):
        sat_customers = {sid: [] for sid in self.sat_ids}
        for g, sid in assignment.items():
            if sid not in sat_customers:
                return None
            sat_customers[sid].extend(self.group_customers[g])

        # Throughput and assignment checks.
        for sid, customers in sat_customers.items():
            demand = sum(int(self.customer_by_id[c]["demand"]) for c in customers)
            if demand > int(self.sat_by_id[sid]["capacity"]):
                return None

        if route_bins_override is not None:
            all_routes = []
            total_cost = 0.0
            used = 0
            for sid in self.sat_ids:
                bins = route_bins_override.get(sid, [])
                expected = sorted(sat_customers[sid])
                actual = sorted(c for part in bins for c in part)
                if expected != actual:
                    return None
                if len(bins) > int(self.sat_by_id[sid]["max_vehicles"]):
                    return None
                for part in bins:
                    if sum(int(self.customer_by_id[c]["demand"]) for c in part) > self.Q2:
                        return None
                    route = self.improve_tour(sid, part, intensive=intensive)
                    all_routes.append({"satellite": sid, "route": route})
                    total_cost += self.route_cost(route)
                    used += 1
            if used > self.T2:
                return None
            return all_routes, total_cost

        options_by_sat = {}
        for sid in self.sat_ids:
            customers = sat_customers[sid]
            if not customers:
                options_by_sat[sid] = [(0, 0.0, [])]
            else:
                opts = self.satellite_route_options(
                    sid, customers, intensive=intensive
                )
                if not opts:
                    return None
                options_by_sat[sid] = opts

        # Multiple-choice knapsack DP over total second-echelon vehicles.
        dp = {0: (0.0, [])}
        for sid in self.sat_ids:
            nxt = {}
            for used, (cost, selections) in dp.items():
                for option in options_by_sat[sid]:
                    new_used = used + option[0]
                    if new_used > self.T2:
                        continue
                    new_cost = cost + option[1]
                    if (new_used not in nxt or
                            new_cost < nxt[new_used][0] - EPS):
                        nxt[new_used] = (new_cost, selections + [(sid, option)])
            dp = nxt
            if not dp:
                return None

        _, (second_cost, selections) = min(
            dp.items(), key=lambda item: item[1][0]
        )
        all_routes = []
        for sid, option in selections:
            for route in option[2]:
                all_routes.append({"satellite": sid, "route": route})
        return all_routes, second_cost

    def construct_first_echelon(self, satellite_loads, intensive=False):
        active = [sid for sid in self.sat_ids if satellite_loads.get(sid, 0) > 0]
        if not active:
            return [], 0.0
        if sum(satellite_loads[s] for s in active) > self.T1 * self.Q1:
            return None

        active.sort(key=lambda sid: math.atan2(
            self.sat_by_id[sid]["y"] - self.depot["y"],
            self.sat_by_id[sid]["x"] - self.depot["x"]
        ))

        m = len(active)
        if m <= 40:
            offsets = range(m)
        else:
            offsets = sorted(set(int(i * m / 20) % m for i in range(20)))

        best = None
        for reverse in (False, True):
            base = list(reversed(active)) if reverse else active
            for off in offsets:
                seq = base[off:] + base[:off]
                vehicle_deliveries = []
                current = {}
                remaining_capacity = self.Q1

                for sid in seq:
                    quantity = int(satellite_loads[sid])
                    while quantity > 0:
                        if remaining_capacity == 0:
                            vehicle_deliveries.append(current)
                            current = {}
                            remaining_capacity = self.Q1
                        amount = min(quantity, remaining_capacity)
                        current[sid] = current.get(sid, 0) + amount
                        quantity -= amount
                        remaining_capacity -= amount
                if current:
                    vehicle_deliveries.append(current)

                if len(vehicle_deliveries) > self.T1:
                    continue

                routes = []
                cost = 0.0
                for v, deliveries in enumerate(vehicle_deliveries):
                    tour = self.improve_tour(
                        self.depot_id, list(deliveries), intensive=intensive
                    )
                    routes.append({
                        "vehicle": v,
                        "route": tour,
                        "deliveries": {
                            str(sid): qty for sid, qty in deliveries.items()
                        }
                    })
                    cost += self.route_cost(tour)

                if best is None or cost < best[1] - EPS:
                    best = (routes, cost)
        return best

    def build_solution(self, assignment, intensive=False,
                       route_bins_override=None):
        if set(assignment) != set(self.group_ids):
            return None

        satellite_loads = {sid: 0 for sid in self.sat_ids}
        for g, sid in assignment.items():
            if sid not in satellite_loads:
                return None
            satellite_loads[sid] += self.group_demand[g]

        for sid in self.sat_ids:
            if satellite_loads[sid] > int(self.sat_by_id[sid]["capacity"]):
                return None

        second = self.construct_second_echelon(
            assignment,
            intensive=intensive,
            route_bins_override=route_bins_override
        )
        if second is None:
            return None
        second_routes, second_cost = second

        first = self.construct_first_echelon(
            satellite_loads, intensive=intensive
        )
        if first is None:
            return None
        first_routes, first_cost = first

        handling_cost = sum(
            satellite_loads[sid] *
            float(self.sat_by_id[sid]["handling_cost"])
            for sid in self.sat_ids
        )
        objective = first_cost + handling_cost + second_cost

        return {
            "objective_value": float(objective),
            "first_echelon_routes": first_routes,
            "second_echelon_routes": second_routes,
            "group_assignments": {
                str(g): assignment[g] for g in self.group_ids
            }
        }

    def accept_solution(self, solution):
        if solution is None:
            return False
        objective = float(solution["objective_value"])
        if objective < self.best_objective - 1e-7:
            self.best_objective = objective
            self.best_solution = solution
            if self.logger:
                self.logger.log_solution(objective, solution)
            return True
        return False

    def greedy_assignments(self, attempts=100):
        """Generate feasible-looking assignments without relying on Gurobi."""
        results = []
        for attempt in range(attempts):
            if self.out_of_time(0.1):
                break
            loads = {sid: 0 for sid in self.sat_ids}
            assignment = {}

            groups = list(self.group_ids)
            if attempt == 0:
                groups.sort(key=lambda g: -self.group_demand[g])
            else:
                groups.sort(
                    key=lambda g: (
                        -self.group_demand[g]
                        + self.rng.random() *
                        max(1, self.total_demand) * 0.15
                    )
                )

            feasible = True
            for g in groups:
                gd = self.group_demand[g]
                candidates = []
                for sid in self.sat_ids:
                    sat = self.sat_by_id[sid]
                    new_load = loads[sid] + gd
                    vehicle_capacity = min(
                        int(sat["capacity"]),
                        int(sat["max_vehicles"]) * self.Q2
                    )
                    if new_load > vehicle_capacity:
                        continue

                    radial = sum(
                        int(self.customer_by_id[c]["demand"]) *
                        self.dist[sid][c]
                        for c in self.group_customers[g]
                    )
                    handling = gd * float(sat["handling_cost"])
                    opening = (2.0 * self.dist[self.depot_id][sid]
                               if loads[sid] == 0 else 0.0)
                    pressure = (
                        new_load / max(1, vehicle_capacity)
                    ) ** 3 * max(1.0, self.total_demand)
                    noise = self.rng.random() * 0.05 * max(1.0, radial)
                    candidates.append(
                        (2.0 * radial + handling + opening + pressure + noise,
                         sid)
                    )

                if not candidates:
                    feasible = False
                    break
                _, sid = min(candidates)
                assignment[g] = sid
                loads[sid] += gd

            if feasible:
                key = tuple(assignment[g] for g in self.group_ids)
                if all(tuple(a[g] for g in self.group_ids) != key
                       for a in results):
                    results.append(assignment)
        return results

    def solve_assignment_mip(self, time_budget, start_assignment=None):
        if gp is None or time_budget <= 0.2 or self.out_of_time(0.2):
            return None

        try:
            model = gp.Model("grouped_two_echelon_assignment")
            model.Params.OutputFlag = 0
            model.Params.Seed = 0
            model.Params.MIPGap = 1e-4
            model.Params.NumericFocus = 0
            model.Params.Threads = 1
            model.Params.TimeLimit = max(0.1, min(time_budget, self.remaining() - 0.05))

            slot_count = {}
            for sid in self.sat_ids:
                sat = self.sat_by_id[sid]
                slot_count[sid] = min(
                    int(sat["max_vehicles"]),
                    self.T2,
                    max(1, len(self.customers))
                )

            x = model.addVars(
                [(g, sid) for g in self.group_ids for sid in self.sat_ids],
                vtype=GRB.BINARY, name="x"
            )
            u = model.addVars(
                [(sid, k) for sid in self.sat_ids
                 for k in range(slot_count[sid])],
                vtype=GRB.BINARY, name="u"
            )
            y = model.addVars(
                [(c, sid, k)
                 for c in self.customer_ids
                 for sid in self.sat_ids
                 for k in range(slot_count[sid])],
                vtype=GRB.BINARY, name="y"
            )
            opened = model.addVars(self.sat_ids, vtype=GRB.BINARY, name="open")

            for g in self.group_ids:
                model.addConstr(gp.quicksum(x[g, sid] for sid in self.sat_ids) == 1)

            customer_group = {
                c["id"]: c["group"] for c in self.customers
            }
            for cid in self.customer_ids:
                g = customer_group[cid]
                for sid in self.sat_ids:
                    model.addConstr(
                        gp.quicksum(
                            y[cid, sid, k] for k in range(slot_count[sid])
                        ) == x[g, sid]
                    )

            for sid in self.sat_ids:
                sat = self.sat_by_id[sid]
                model.addConstr(
                    gp.quicksum(
                        self.group_demand[g] * x[g, sid]
                        for g in self.group_ids
                    ) <= int(sat["capacity"])
                )
                for k in range(slot_count[sid]):
                    model.addConstr(
                        gp.quicksum(
                            int(self.customer_by_id[c]["demand"]) *
                            y[c, sid, k]
                            for c in self.customer_ids
                        ) <= self.Q2 * u[sid, k]
                    )
                    if k + 1 < slot_count[sid]:
                        model.addConstr(u[sid, k] >= u[sid, k + 1])
                for g in self.group_ids:
                    model.addConstr(x[g, sid] <= opened[sid])

            model.addConstr(
                gp.quicksum(u[sid, k]
                            for sid in self.sat_ids
                            for k in range(slot_count[sid])) <= self.T2
            )

            objective = gp.LinExpr()
            for g in self.group_ids:
                gd = self.group_demand[g]
                for sid in self.sat_ids:
                    assignment_cost = (
                        gd * float(self.sat_by_id[sid]["handling_cost"])
                        + 2.0 * sum(
                            int(self.customer_by_id[c]["demand"]) *
                            self.dist[sid][c]
                            for c in self.group_customers[g]
                        )
                    )
                    objective += assignment_cost * x[g, sid]
            for sid in self.sat_ids:
                objective += 2.0 * self.dist[self.depot_id][sid] * opened[sid]
                for k in range(slot_count[sid]):
                    objective += 0.001 * u[sid, k]
            model.setObjective(objective, GRB.MINIMIZE)

            if start_assignment:
                for g in self.group_ids:
                    for sid in self.sat_ids:
                        x[g, sid].Start = 1.0 if start_assignment[g] == sid else 0.0

            model.optimize()
            if model.SolCount <= 0:
                return None

            assignment = {}
            for g in self.group_ids:
                assignment[g] = max(
                    self.sat_ids, key=lambda sid: x[g, sid].X
                )

            bins = {sid: [] for sid in self.sat_ids}
            for sid in self.sat_ids:
                for k in range(slot_count[sid]):
                    customers = [
                        c for c in self.customer_ids
                        if y[c, sid, k].X > 0.5
                    ]
                    if customers:
                        bins[sid].append(customers)

            return assignment, bins
        except Exception:
            return None

    def local_search(self, initial_assignment):
        current_assignment = dict(initial_assignment)
        current_solution = self.build_solution(current_assignment, intensive=False)
        if current_solution is None:
            return
        current_obj = current_solution["objective_value"]
        self.accept_solution(current_solution)

        group_order = list(self.group_ids)
        while not self.out_of_time(0.12):
            improved = False
            self.rng.shuffle(group_order)

            for g in group_order:
                if self.out_of_time(0.12):
                    break
                old_sid = current_assignment[g]
                candidate_sats = sorted(
                    (sid for sid in self.sat_ids if sid != old_sid),
                    key=lambda sid: sum(
                        self.dist[sid][c] for c in self.group_customers[g]
                    )
                )
                for sid in candidate_sats:
                    if self.out_of_time(0.12):
                        break
                    trial = dict(current_assignment)
                    trial[g] = sid
                    solution = self.build_solution(trial, intensive=False)
                    if (solution is not None and
                            solution["objective_value"] < current_obj - 1e-7):
                        current_assignment = trial
                        current_solution = solution
                        current_obj = solution["objective_value"]
                        self.accept_solution(solution)
                        improved = True
                        break
                if improved:
                    break

            if improved:
                continue

            # Pair swaps can escape capacity-induced local optima.
            pairs = []
            for i in range(len(self.group_ids)):
                for j in range(i + 1, len(self.group_ids)):
                    g1, g2 = self.group_ids[i], self.group_ids[j]
                    if current_assignment[g1] != current_assignment[g2]:
                        pairs.append((g1, g2))
            self.rng.shuffle(pairs)
            for g1, g2 in pairs[:min(100, len(pairs))]:
                if self.out_of_time(0.12):
                    break
                trial = dict(current_assignment)
                trial[g1], trial[g2] = trial[g2], trial[g1]
                solution = self.build_solution(trial, intensive=False)
                if (solution is not None and
                        solution["objective_value"] < current_obj - 1e-7):
                    current_assignment = trial
                    current_solution = solution
                    current_obj = solution["objective_value"]
                    self.accept_solution(solution)
                    improved = True
                    break

            if not improved:
                break

    def solve(self):
        if any(int(c["demand"]) > self.Q2 for c in self.customers):
            raise RuntimeError("Instance is infeasible: a customer demand exceeds Q2.")
        if self.total_demand > self.T1 * self.Q1:
            raise RuntimeError("Instance is infeasible: insufficient first-echelon capacity.")
        if self.total_demand > self.T2 * self.Q2:
            raise RuntimeError("Instance is infeasible: insufficient second-echelon capacity.")
        if sum(int(s["capacity"]) for s in self.satellites) < self.total_demand:
            raise RuntimeError("Instance is infeasible: insufficient satellite throughput.")

        greedy = self.greedy_assignments(
            attempts=max(20, min(200, 10 * max(1, len(self.group_ids))))
        )
        best_assignment = None

        for assignment in greedy:
            if self.out_of_time(0.15):
                break
            solution = self.build_solution(assignment, intensive=False)
            if solution is not None:
                if self.accept_solution(solution):
                    best_assignment = dict(assignment)
                elif best_assignment is None:
                    best_assignment = dict(assignment)

        # Use an exact assignment-and-bin-packing MIP for a bounded portion
        # of the available runtime.
        if not self.out_of_time(0.3):
            fraction = 0.35 if self.best_solution is not None else 0.70
            mip_budget = min(30.0, max(0.2, self.remaining() * fraction))
            mip_result = self.solve_assignment_mip(
                mip_budget, start_assignment=best_assignment
            )
            if mip_result is not None:
                assignment, bins = mip_result
                exact_bin_solution = self.build_solution(
                    assignment,
                    intensive=False,
                    route_bins_override=bins
                )
                self.accept_solution(exact_bin_solution)

                general_solution = self.build_solution(
                    assignment, intensive=False
                )
                self.accept_solution(general_solution)
                if (general_solution is not None and
                        (best_assignment is None or
                         general_solution["objective_value"]
                         <= self.best_objective + 1e-7)):
                    best_assignment = dict(assignment)

        if best_assignment is None and self.best_solution is not None:
            best_assignment = {
                self._parse_key(g): sid
                for g, sid in self.best_solution["group_assignments"].items()
            }

        if best_assignment is not None and not self.out_of_time(0.15):
            self.local_search(best_assignment)

        # Final intensive tour polishing for the best assignment.
        if self.best_solution is not None and not self.out_of_time(0.08):
            assignment = {
                self._parse_key(g): sid
                for g, sid in self.best_solution["group_assignments"].items()
            }
            polished = self.build_solution(assignment, intensive=True)
            self.accept_solution(polished)

        if self.best_solution is None:
            raise RuntimeError(
                "No feasible solution was found within the time limit."
            )
        return self.best_solution

    def _parse_key(self, key):
        for g in self.group_ids:
            if str(g) == str(key):
                return g
        return key


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

    output_dir = os.path.dirname(os.path.abspath(args.solution_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()
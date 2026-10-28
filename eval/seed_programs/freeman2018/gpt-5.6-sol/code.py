import argparse
import json
import math
import os
import random
import time
import heapq

import numpy as np

from solution_logger import SolutionLogger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    return parser.parse_args()


def read_instance(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class TourSolver:
    def __init__(self, data, deadline, logger):
        self.data = data
        self.deadline = deadline
        self.logger = logger
        self.rng = random.Random(0)

        self.sites = data["sites"]
        self.customers = data["customers"]
        self.params = data["parameters"]

        self.n = len(self.sites)
        self.m = len(self.customers)
        self.T = int(self.params.get("T", data.get("num_periods", 0)))
        self.delta = int(self.params["delta"])
        self.epsilon = int(self.params["epsilon"])
        self.L = float(self.params["L"])
        self.travel_cost = float(self.params["c"])
        self.q = float(self.params["q"])

        self.site_ids = [int(s["id"]) for s in self.sites]
        self.site_index = {sid: i for i, sid in enumerate(self.site_ids)}

        self.capacity = np.asarray(
            [float(s["capacity"]) for s in self.sites], dtype=float
        )
        self.revenue = np.asarray(
            [float(s["revenue_per_customer"]) for s in self.sites], dtype=float
        )
        self.fixed_cost = np.asarray(
            [float(s["fixed_cost"]) for s in self.sites], dtype=float
        )
        self.max_events = np.asarray(
            [int(s["max_events"]) for s in self.sites], dtype=np.int32
        )
        self.population = np.asarray(
            [float(c["population"]) for c in self.customers], dtype=float
        )

        self.dist = self._build_site_distance_matrix()
        self.start_dist = self._build_start_distances()
        self.reachable = [
            np.flatnonzero(self.dist[j] <= self.L + 1e-9).tolist()
            for j in range(self.n)
        ]

        self.attraction, self.self_attraction = self._build_attractions()
        self.solo_revenue = self._build_solo_revenues()

        self.best_objective = -float("inf")
        self.best_route = None
        self.best_events = None
        self.best_attendance = None

    def time_left(self):
        return self.deadline - time.monotonic()

    def _lookup_distance(self, mapping, a, b, fallback):
        candidates = (
            f"{a}_{b}",
            f"{int(a)}_{int(b)}",
            f"{str(a)}_{str(b)}",
        )
        for key in candidates:
            if key in mapping:
                return float(mapping[key])
        return float(fallback)

    def _build_site_distance_matrix(self):
        mapping = self.data.get("site_to_site_distances", {})
        result = np.zeros((self.n, self.n), dtype=float)
        for a, site_a in enumerate(self.sites):
            for b, site_b in enumerate(self.sites):
                fallback = math.hypot(
                    float(site_a["x"]) - float(site_b["x"]),
                    float(site_a["y"]) - float(site_b["y"]),
                )
                result[a, b] = self._lookup_distance(
                    mapping, self.site_ids[a], self.site_ids[b], fallback
                )
        return result

    def _build_start_distances(self):
        mapping = self.data.get("site_to_site_distances", {})
        result = np.zeros(self.n, dtype=float)
        for j, sid in enumerate(self.site_ids):
            result[j] = self._lookup_distance(mapping, 0, sid, 0.0)
        return result

    def _build_attractions(self):
        avg_sites = max(float(self.params["avg_businesses_sites"]), 1e-12)
        avg_customers = max(
            float(self.params["avg_businesses_customers"]), 1e-12
        )

        site_strength = np.asarray(
            [float(s["num_businesses"]) / avg_sites for s in self.sites],
            dtype=float,
        )
        self_attraction = np.asarray(
            [
                max(float(c["num_businesses"]) / avg_customers, 1e-12)
                for c in self.customers
            ],
            dtype=float,
        )

        customer_site_dist = np.empty((self.m, self.n), dtype=float)
        mapping = self.data.get("customer_to_site_distances", {})

        for i, customer in enumerate(self.customers):
            cid = int(customer["id"])
            for j, site in enumerate(self.sites):
                sid = int(site["id"])
                fallback = math.hypot(
                    float(customer["x"]) - float(site["x"]),
                    float(customer["y"]) - float(site["y"]),
                )
                d = self._lookup_distance(mapping, cid, sid, fallback)
                customer_site_dist[i, j] = max(d, 1e-9)

        base = site_strength[np.newaxis, :] / np.power(
            customer_site_dist, self.q
        )

        weekly = list(self.params["weekly_attraction_cycle"])
        if not weekly:
            weekly = [1.0] * 7

        attraction = np.empty((self.T, self.m, self.n), dtype=float)
        for t in range(self.T):
            factor = max(0.0, float(weekly[t % len(weekly)]))
            attraction[t] = base * factor

        return attraction, self_attraction

    def _build_solo_revenues(self):
        result = np.zeros((self.T, self.n), dtype=float)
        for t in range(self.T):
            numerators = self.attraction[t]
            denominators = self.self_attraction[:, None] + numerators
            shares = np.divide(
                numerators,
                denominators,
                out=np.zeros_like(numerators),
                where=denominators > 0,
            )
            demand = np.sum(self.population[:, None] * shares, axis=0)
            attendance = np.minimum(self.capacity, demand)
            result[t] = attendance * self.revenue
        return result

    def valid_rest_bits(self, bits):
        if len(bits) < self.delta + 2:
            return True
        return sum(bits[-(self.delta + 2):]) <= self.delta

    def full_rest_feasible(self, events):
        window = self.delta + 2
        if window > self.T:
            return True
        running = sum(events[:window])
        if running > self.delta:
            return False
        for t in range(window, self.T):
            running += events[t] - events[t - window]
            if running > self.delta:
                return False
        return True

    def attendance_for_event(self, period, event_sites):
        site = event_sites[period]
        if site < 0:
            return 0.0

        denominator = self.self_attraction.copy()
        lo = max(0, period - self.epsilon)
        hi = min(len(event_sites) - 1, period + self.epsilon)
        for u in range(lo, hi + 1):
            other_site = event_sites[u]
            if other_site >= 0:
                denominator += self.attraction[u, :, other_site]

        numerator = self.attraction[period, :, site]
        share = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 0,
        )
        demand = float(np.dot(self.population, share))
        return min(float(self.capacity[site]), demand)

    def evaluate(self, route, events, return_attendance=False):
        event_sites = [
            route[t] if events[t] else -1
            for t in range(self.T)
        ]

        denominators = np.repeat(
            self.self_attraction[:, None], self.T, axis=1
        )
        for u, site in enumerate(event_sites):
            if site < 0:
                continue
            lo = max(0, u - self.epsilon)
            hi = min(self.T - 1, u + self.epsilon)
            denominators[:, lo:hi + 1] += self.attraction[u, :, site][:, None]

        attendance = np.zeros((self.n, self.T), dtype=float)
        total_revenue = 0.0
        fixed = 0.0

        for t, site in enumerate(event_sites):
            if site < 0:
                continue
            numerator = self.attraction[t, :, site]
            share = np.divide(
                numerator,
                denominators[:, t],
                out=np.zeros_like(numerator),
                where=denominators[:, t] > 0,
            )
            demand = float(np.dot(self.population, share))
            value = min(float(self.capacity[site]), demand)
            attendance[site, t] = value
            total_revenue += float(self.revenue[site]) * value
            fixed += float(self.fixed_cost[site])

        travel = float(self.start_dist[route[0]]) if self.T > 0 else 0.0
        for t in range(1, self.T):
            travel += float(self.dist[route[t - 1], route[t]])

        objective = total_revenue - fixed - self.travel_cost * travel
        if return_attendance:
            return objective, attendance
        return objective

    def route_feasible(self, route):
        if not route or len(route) != self.T:
            return False
        if self.start_dist[route[0]] > self.L + 1e-9:
            return False
        for t in range(1, self.T):
            if self.dist[route[t - 1], route[t]] > self.L + 1e-9:
                return False
        return True

    def schedule_feasible(self, route, events):
        if not self.route_feasible(route):
            return False
        if not self.full_rest_feasible(events):
            return False
        counts = np.zeros(self.n, dtype=np.int32)
        for t in range(self.T):
            if events[t]:
                counts[route[t]] += 1
        return bool(np.all(counts <= self.max_events))

    def make_solution(self, route, events, objective=None, attendance=None):
        if objective is None or attendance is None:
            objective, attendance = self.evaluate(
                route, events, return_attendance=True
            )

        tour = {
            str(t + 1): int(self.site_ids[route[t]])
            for t in range(self.T)
        }
        event_dict = {
            str(t + 1): int(bool(events[t]))
            for t in range(self.T)
        }
        event_sites = {
            str(t + 1): int(self.site_ids[route[t]])
            for t in range(self.T)
            if events[t]
        }
        attendance_dict = {}
        for j, sid in enumerate(self.site_ids):
            for t in range(self.T):
                attendance_dict[f"{sid}_{t + 1}"] = float(attendance[j, t])

        return {
            "objective_value": float(objective),
            "tour": tour,
            "events": event_dict,
            "event_sites": event_sites,
            "attendance": attendance_dict,
        }

    def update_incumbent(self, route, events):
        objective, attendance = self.evaluate(
            route, events, return_attendance=True
        )
        if objective > self.best_objective + 1e-8:
            self.best_objective = objective
            self.best_route = list(route)
            self.best_events = list(events)
            self.best_attendance = attendance
            if self.logger:
                solution = self.make_solution(
                    self.best_route,
                    self.best_events,
                    self.best_objective,
                    self.best_attendance,
                )
                self.logger.log_solution(self.best_objective, solution)
            return True
        return False

    def initialize_incumbent(self):
        if self.n == 0 or self.T == 0:
            self.best_objective = 0.0
            self.best_route = []
            self.best_events = []
            self.best_attendance = np.zeros((self.n, self.T), dtype=float)
            return

        feasible_start = [
            j for j in range(self.n)
            if self.start_dist[j] <= self.L + 1e-9
        ]
        start = feasible_start[0] if feasible_start else 0
        route = [start] * self.T
        events = [0] * self.T
        self.update_incumbent(route, events)

    def state_pending_score(self, state):
        route, event_sites, counts, bits, finalized, cost, pending_solo = state
        current_t = len(route) - 1
        lo = max(0, current_t - self.epsilon + 1)
        pending_revenue = 0.0
        for s in range(lo, current_t + 1):
            if event_sites[s] >= 0:
                attendance = self.attendance_for_event(s, event_sites)
                pending_revenue += (
                    attendance * self.revenue[event_sites[s]]
                )
        return finalized - cost + pending_revenue

    def beam_search(self, beam_deadline):
        if self.n == 0 or self.T == 0:
            return

        beam_width = max(30, min(450, 14000 // max(2 * self.n, 1)))
        state0 = (
            tuple(),                       # route
            tuple(),                       # event site or -1
            tuple([0] * self.n),           # event counts
            tuple(),                       # event bits
            0.0,                           # finalized revenue
            0.0,                           # accumulated costs
            0.0,                           # solo proxy for unresolved events
        )
        beam = [state0]

        for t in range(self.T):
            if time.monotonic() >= beam_deadline:
                return

            rough_candidates = []
            keep_rough = max(beam_width * 4, beam_width)

            for state in beam:
                route, event_sites, counts, bits, finalized, cost, pending = state

                if t == 0:
                    destinations = [
                        j for j in range(self.n)
                        if self.start_dist[j] <= self.L + 1e-9
                    ]
                else:
                    destinations = self.reachable[route[-1]]

                for destination in destinations:
                    move_distance = (
                        self.start_dist[destination]
                        if t == 0
                        else self.dist[route[-1], destination]
                    )
                    move_cost = self.travel_cost * float(move_distance)

                    for hold_event in (0, 1):
                        if hold_event and counts[destination] >= self.max_events[destination]:
                            continue

                        new_bits = bits + (hold_event,)
                        if not self.valid_rest_bits(new_bits):
                            continue

                        new_route = route + (destination,)
                        new_event_sites = event_sites + (
                            destination if hold_event else -1,
                        )

                        new_counts_list = list(counts)
                        new_cost = cost + move_cost
                        new_pending = pending
                        if hold_event:
                            new_counts_list[destination] += 1
                            new_cost += float(self.fixed_cost[destination])
                            new_pending += float(
                                self.solo_revenue[t, destination]
                            )

                        new_finalized = finalized
                        final_period = t - self.epsilon
                        if final_period >= 0:
                            final_site = new_event_sites[final_period]
                            if final_site >= 0:
                                att = self.attendance_for_event(
                                    final_period, new_event_sites
                                )
                                new_finalized += (
                                    att * self.revenue[final_site]
                                )
                                new_pending -= float(
                                    self.solo_revenue[final_period, final_site]
                                )

                        new_state = (
                            new_route,
                            new_event_sites,
                            tuple(new_counts_list),
                            new_bits,
                            new_finalized,
                            new_cost,
                            new_pending,
                        )
                        cheap_score = new_finalized - new_cost + new_pending

                        item = (cheap_score, id(new_state), new_state)
                        if len(rough_candidates) < keep_rough:
                            heapq.heappush(rough_candidates, item)
                        elif cheap_score > rough_candidates[0][0]:
                            heapq.heapreplace(rough_candidates, item)

                if time.monotonic() >= beam_deadline:
                    return

            candidates = [item[2] for item in rough_candidates]
            scored = []
            for state in candidates:
                score = self.state_pending_score(state)
                scored.append((score, state))

            scored.sort(key=lambda x: x[0], reverse=True)
            beam = [state for _, state in scored[:beam_width]]
            if not beam:
                return

        final_scored = []
        for state in beam:
            route, event_sites, counts, bits, _, _, _ = state
            events = [1 if s >= 0 else 0 for s in event_sites]
            objective = self.evaluate(route, events)
            final_scored.append((objective, route, events))

        final_scored.sort(key=lambda x: x[0])
        for objective, route, events in final_scored:
            if objective > self.best_objective + 1e-8:
                self.update_incumbent(route, events)

    def counts_for(self, route, events):
        counts = np.zeros(self.n, dtype=np.int32)
        for t in range(self.T):
            if events[t]:
                counts[route[t]] += 1
        return counts

    def deterministic_event_improvement(self):
        improved = True
        while improved and self.time_left() > 0.05:
            improved = False
            route = list(self.best_route)
            events = list(self.best_events)
            current_obj = self.best_objective
            best_move = None
            best_value = current_obj

            counts = self.counts_for(route, events)
            for t in range(self.T):
                if self.time_left() <= 0.05:
                    break
                candidate = list(events)
                if candidate[t]:
                    candidate[t] = 0
                else:
                    site = route[t]
                    if counts[site] >= self.max_events[site]:
                        continue
                    candidate[t] = 1

                if not self.full_rest_feasible(candidate):
                    continue

                value = self.evaluate(route, candidate)
                if value > best_value + 1e-8:
                    best_value = value
                    best_move = candidate

            if best_move is not None:
                self.update_incumbent(route, best_move)
                improved = True

    def random_neighbor(self, route, events):
        new_route = list(route)
        new_events = list(events)
        move_type = self.rng.random()

        if move_type < 0.50:
            t = self.rng.randrange(self.T)
            new_events[t] = 1 - new_events[t]

        elif move_type < 0.80:
            t = self.rng.randrange(self.T)
            candidates = []
            for j in range(self.n):
                if t == 0:
                    prev_ok = self.start_dist[j] <= self.L + 1e-9
                else:
                    prev_ok = self.dist[new_route[t - 1], j] <= self.L + 1e-9

                next_ok = (
                    t == self.T - 1
                    or self.dist[j, new_route[t + 1]] <= self.L + 1e-9
                )
                if prev_ok and next_ok:
                    candidates.append(j)
            if not candidates:
                return None
            new_route[t] = self.rng.choice(candidates)

        else:
            a = self.rng.randrange(self.T)
            max_length = min(self.T - a, max(1, self.epsilon + 2))
            length = self.rng.randint(1, max_length)
            b = a + length - 1

            candidates = []
            for j in range(self.n):
                if a == 0:
                    left_ok = self.start_dist[j] <= self.L + 1e-9
                else:
                    left_ok = self.dist[new_route[a - 1], j] <= self.L + 1e-9
                right_ok = (
                    b == self.T - 1
                    or self.dist[j, new_route[b + 1]] <= self.L + 1e-9
                )
                if left_ok and right_ok:
                    candidates.append(j)
            if not candidates:
                return None

            site = self.rng.choice(candidates)
            for t in range(a, b + 1):
                new_route[t] = site

        if not self.schedule_feasible(new_route, new_events):
            return None
        return new_route, new_events

    def local_search(self):
        if self.T == 0 or self.n == 0:
            return

        self.deterministic_event_improvement()

        current_route = list(self.best_route)
        current_events = list(self.best_events)
        current_obj = self.evaluate(current_route, current_events)

        start = time.monotonic()
        initial_temperature = max(1.0, abs(current_obj) * 0.01 + 10.0)

        while self.time_left() > 0.03:
            candidate = self.random_neighbor(current_route, current_events)
            if candidate is None:
                continue

            route, events = candidate
            value = self.evaluate(route, events)

            elapsed = time.monotonic() - start
            total = max(elapsed + self.time_left(), 1e-9)
            fraction = min(1.0, elapsed / total)
            temperature = max(
                1e-6, initial_temperature * (1.0 - fraction) ** 2
            )

            difference = value - current_obj
            accept = difference >= 0.0
            if not accept:
                exponent = max(-700.0, difference / temperature)
                accept = self.rng.random() < math.exp(exponent)

            if accept:
                current_route = route
                current_events = events
                current_obj = value

            if value > self.best_objective + 1e-8:
                self.update_incumbent(route, events)

                if self.rng.random() < 0.25:
                    current_route = list(self.best_route)
                    current_events = list(self.best_events)
                    current_obj = self.best_objective

    def solve(self):
        self.initialize_incumbent()
        if self.n == 0 or self.T == 0 or self.time_left() <= 0.03:
            return

        remaining = self.time_left()
        beam_budget = max(0.0, remaining * 0.62)
        beam_deadline = min(
            self.deadline - 0.03,
            time.monotonic() + beam_budget,
        )
        self.beam_search(beam_deadline)

        if self.time_left() > 0.05:
            self.local_search()

    def final_solution(self):
        if self.best_route is None:
            return {
                "objective_value": 0.0,
                "tour": {},
                "events": {},
                "event_sites": {},
                "attendance": {},
            }

        objective, attendance = self.evaluate(
            self.best_route, self.best_events, return_attendance=True
        )
        return self.make_solution(
            self.best_route,
            self.best_events,
            objective,
            attendance,
        )


def main():
    args = parse_args()
    logger = (
        SolutionLogger(args.log_path, sense="maximize")
        if args.log_path
        else None
    )

    start = time.monotonic()
    deadline = start + max(0, args.time_limit)

    data = read_instance(args.instance_path)
    solver = TourSolver(data, deadline, logger)
    solver.solve()
    solution = solver.final_solution()

    output_dir = os.path.dirname(os.path.abspath(args.solution_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(solution, f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
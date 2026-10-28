import argparse
import itertools
import json
import math
import os
import random
import time
from collections import defaultdict
from functools import lru_cache

from solution_logger import SolutionLogger


EPS = 1e-9


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    return parser.parse_args()


def weighted_choice(rng, realizations):
    x = rng.random()
    acc = 0.0
    for r in realizations:
        acc += float(r["probability"])
        if x <= acc + 1e-15:
            return r
    return realizations[-1]


def normalize_distribution(dist):
    total = sum(p for _, _, p in dist)
    if total <= 0:
        return dist
    return [(h, w, p / total) for h, w, p in dist]


class Solver:
    def __init__(self, data, deadline, logger):
        self.data = data
        self.deadline = deadline
        self.logger = logger

        vehicle = data["vehicle"]
        self.H = int(vehicle["H"])
        self.W = int(vehicle["W"])
        self.Q = int(vehicle["Q"])
        self.K = int(vehicle["K"])
        self.capacity_area = self.H * self.W

        self.cf = float(data["parameters"]["recourse_cost_cf"])
        self.distance = data["distance_matrix"]
        self.depot = int(data["depot"]["id"])
        self.customers = [int(c["id"]) for c in data["customers"]]
        self.customer_data = {int(c["id"]): c for c in data["customers"]}
        self.n = len(self.customers)

        self.rng = random.Random(0)

        self.expected_area = {}
        self.expected_weight = {}
        self.minimum_area = {}
        self.minimum_weight = {}
        self.has_stochastic = {}

        self.customer_distributions = {}
        self.sample_states = {}

        self.route_eval_cache = {}
        self.shelf_cache = {}

        self.best_objective = float("inf")
        self.best_routes = None

        self._preprocess_customers()

    def out_of_time(self, reserve=0.03):
        return time.monotonic() >= self.deadline - reserve

    def _preprocess_customers(self):
        for cid in self.customers:
            customer = self.customer_data[cid]
            exp_area = 0.0
            exp_weight = 0.0
            min_area = 0
            min_weight = 0
            stochastic = False

            for item in customer["items"]:
                realizations = item["realizations"]
                stochastic = stochastic or bool(item.get("is_stochastic", False))
                exp_area += sum(
                    float(r["probability"]) * int(r["height"]) * int(r["width"])
                    for r in realizations
                )
                exp_weight += sum(
                    float(r["probability"]) * int(r["weight"])
                    for r in realizations
                )
                min_area += min(int(r["height"]) * int(r["width"]) for r in realizations)
                min_weight += min(int(r["weight"]) for r in realizations)

            self.expected_area[cid] = exp_area
            self.expected_weight[cid] = exp_weight
            self.minimum_area[cid] = min_area
            self.minimum_weight[cid] = min_weight
            self.has_stochastic[cid] = stochastic

            dist = self._build_customer_distribution(customer)
            self.customer_distributions[cid] = dist
            self.sample_states[cid] = self._make_customer_samples(cid, dist, 384)

    def _minimum_shelf_height(self, dimensions):
        key = tuple(sorted((int(h), int(w)) for h, w in dimensions))
        if key in self.shelf_cache:
            return self.shelf_cache[key]

        if not key:
            return 0
        if any(w > self.W or h > self.H for h, w in key):
            return self.H + 1

        items = sorted(key, key=lambda z: (-z[0], -z[1]))

        # Exact shelf partitioning for small customer orders.
        if len(items) <= 9:
            @lru_cache(maxsize=None)
            def dfs(index, shelves):
                if index == len(items):
                    return sum(max_height for _, max_height in shelves)

                h, w = items[index]
                best = self.H + sum(x[0] for x in items[index:]) + 1
                seen = set()

                for j, (used_width, max_height) in enumerate(shelves):
                    signature = (used_width, max_height)
                    if signature in seen:
                        continue
                    seen.add(signature)
                    if used_width + w <= self.W:
                        new_shelves = list(shelves)
                        new_shelves[j] = (used_width + w, max(max_height, h))
                        new_shelves.sort()
                        best = min(best, dfs(index + 1, tuple(new_shelves)))

                new_shelves = list(shelves)
                new_shelves.append((w, h))
                new_shelves.sort()
                best = min(best, dfs(index + 1, tuple(new_shelves)))
                return best

            result = dfs(0, tuple())
        else:
            def first_fit(order):
                shelves = []
                for h, w in order:
                    best_j = None
                    best_increment = None
                    for j, (used, height) in enumerate(shelves):
                        if used + w <= self.W:
                            inc = max(height, h) - height
                            if best_increment is None or inc < best_increment:
                                best_increment = inc
                                best_j = j
                    if best_j is None:
                        shelves.append([w, h])
                    else:
                        shelves[best_j][0] += w
                        shelves[best_j][1] = max(shelves[best_j][1], h)
                return sum(height for _, height in shelves)

            orders = [
                sorted(items, key=lambda z: (-z[0], -z[1])),
                sorted(items, key=lambda z: (-z[1], -z[0])),
                sorted(items, key=lambda z: (-(z[0] * z[1]), -z[0])),
            ]
            result = min(first_fit(order) for order in orders)

        self.shelf_cache[key] = result
        return result

    def _build_customer_distribution(self, customer):
        items = customer["items"]
        product_size = 1
        for item in items:
            product_size *= max(1, len(item["realizations"]))

        compressed = defaultdict(float)

        if product_size <= 5000 and not self.out_of_time(0.15):
            realization_lists = [item["realizations"] for item in items]
            for combination in itertools.product(*realization_lists):
                dimensions = []
                weight = 0
                probability = 1.0
                for r in combination:
                    dimensions.append((int(r["height"]), int(r["width"])))
                    weight += int(r["weight"])
                    probability *= float(r["probability"])
                shelf_height = self._minimum_shelf_height(dimensions)
                compressed[(shelf_height, weight)] += probability
        else:
            rng = random.Random(104729 * int(customer["id"]) + 17)
            sample_count = 768
            for _ in range(sample_count):
                dimensions = []
                weight = 0
                for item in items:
                    r = weighted_choice(rng, item["realizations"])
                    dimensions.append((int(r["height"]), int(r["width"])))
                    weight += int(r["weight"])
                shelf_height = self._minimum_shelf_height(dimensions)
                compressed[(shelf_height, weight)] += 1.0 / sample_count

        return normalize_distribution(
            [(h, w, p) for (h, w), p in compressed.items() if p > 0]
        )

    def _make_customer_samples(self, cid, dist, count):
        rng = random.Random(1000003 + 7919 * cid)
        cumulative = []
        acc = 0.0
        for h, w, p in dist:
            acc += p
            cumulative.append((acc, h, w))

        result = []
        # Independent stratified samples for each customer.
        offset = rng.random()
        order = list(range(count))
        rng.shuffle(order)
        for rank in order:
            x = ((rank + offset) / count) % 1.0
            chosen = cumulative[-1]
            for entry in cumulative:
                if x <= entry[0] + 1e-15:
                    chosen = entry
                    break
            result.append((chosen[1], chosen[2]))
        return result

    def route_travel(self, route):
        if not route:
            return 0
        total = self.distance[self.depot][route[0]]
        for a, b in zip(route, route[1:]):
            total += self.distance[a][b]
        total += self.distance[route[-1]][self.depot]
        return total

    def route_expected_capacity_feasible(self, route):
        area = sum(self.expected_area[c] for c in route)
        weight = sum(self.expected_weight[c] for c in route)
        return area <= self.capacity_area + 1e-7 and weight <= self.Q + 1e-7

    def _max_served(self, states):
        # Two-dimensional cardinality knapsack. A state is
        # (shelf height needed by the customer, customer weight).
        dp = {(0, 0): 0}

        for height, weight in states:
            if height > self.H or weight > self.Q:
                continue

            additions = {}
            for (used_h, used_w), count in dp.items():
                nh = used_h + height
                nw = used_w + weight
                if nh <= self.H and nw <= self.Q:
                    key = (nh, nw)
                    value = count + 1
                    if value > additions.get(key, -1):
                        additions[key] = value

            for key, value in additions.items():
                if value > dp.get(key, -1):
                    dp[key] = value

            # Remove simple dominated resource states with the same height.
            if len(dp) > 25000:
                by_height = defaultdict(list)
                for (h, w), value in dp.items():
                    by_height[h].append((w, value))
                reduced = {}
                for h, entries in by_height.items():
                    entries.sort()
                    best_value = -1
                    for w, value in entries:
                        if value > best_value:
                            reduced[(h, w)] = value
                            best_value = value
                dp = reduced

        return max(dp.values()) if dp else 0

    def evaluate_route_set(self, route):
        key = frozenset(route)
        cached = self.route_eval_cache.get(key)
        if cached is not None:
            return cached

        route = list(key)
        m = len(route)
        if m == 0:
            result = (0.0, False)
            self.route_eval_cache[key] = result
            return result

        if not self.route_expected_capacity_feasible(route):
            result = (float("inf"), False)
            self.route_eval_cache[key] = result
            return result

        distributions = [self.customer_distributions[c] for c in route]
        combinations = 1
        for dist in distributions:
            combinations *= len(dist)
            if combinations > 12000:
                break

        expected_served = 0.0
        full_probability = 0.0

        if combinations <= 12000 and not self.out_of_time(0.15):
            for combination in itertools.product(*distributions):
                probability = 1.0
                states = []
                total_h = 0
                total_w = 0
                for h, w, p in combination:
                    probability *= p
                    states.append((h, w))
                    total_h += h
                    total_w += w

                if probability <= 0:
                    continue
                served = self._max_served(states)
                expected_served += probability * served
                if total_h <= self.H and total_w <= self.Q:
                    full_probability += probability
        else:
            samples = 256
            if self.out_of_time(0.08):
                samples = 64

            full_count = 0
            served_sum = 0.0
            for s in range(samples):
                states = []
                total_h = 0
                total_w = 0
                for c in route:
                    sample_array = self.sample_states[c]
                    index = (s * 131 + c * 17) % len(sample_array)
                    h, w = sample_array[index]
                    states.append((h, w))
                    total_h += h
                    total_w += w

                served_sum += self._max_served(states)
                if total_h <= self.H and total_w <= self.Q:
                    full_count += 1

            expected_served = served_sum / samples
            full_probability = full_count / samples

        all_deterministic = not any(self.has_stochastic[c] for c in route)

        if all_deterministic:
            allowed = full_probability > EPS
        elif full_probability > EPS:
            allowed = True
        else:
            # A necessary optimistic test is retained for rare realizations that
            # may not appear in the sample used above.
            allowed = (
                sum(self.minimum_area[c] for c in route) <= self.capacity_area
                and sum(self.minimum_weight[c] for c in route) <= self.Q
            )

        penalty = self.cf * max(0.0, m - expected_served)
        result = (penalty, allowed)
        self.route_eval_cache[key] = result
        return result

    def total_objective(self, routes):
        travel = sum(self.route_travel(r) for r in routes)
        penalty = 0.0
        for route in routes:
            p, allowed = self.evaluate_route_set(route)
            if not allowed or not math.isfinite(p):
                return float("inf")
            penalty += p
        return float(travel + penalty)

    def make_solution(self, routes, objective=None):
        if objective is None:
            objective = self.total_objective(routes)
        return {
            "objective_value": float(objective),
            "routes": [[int(c) for c in route] for route in routes],
        }

    def register(self, routes):
        if len(routes) != self.K or any(not r for r in routes):
            return
        objective = self.total_objective(routes)
        if objective + 1e-8 < self.best_objective:
            self.best_objective = objective
            self.best_routes = [list(r) for r in routes]
            if self.logger:
                solution = self.make_solution(self.best_routes, objective)
                self.logger.log_solution(objective, solution)

    def best_orientation_merge(self, a, b):
        candidates = [
            a + b,
            a + list(reversed(b)),
            list(reversed(a)) + b,
            list(reversed(a)) + list(reversed(b)),
        ]
        return min(candidates, key=self.route_travel)

    def savings_construction(self, randomized=False):
        routes = [[c] for c in self.customers]

        while len(routes) > self.K and not self.out_of_time(0.15):
            rough = []
            for i in range(len(routes)):
                for j in range(i + 1, len(routes)):
                    union = routes[i] + routes[j]
                    if not self.route_expected_capacity_feasible(union):
                        continue
                    merged = self.best_orientation_merge(routes[i], routes[j])
                    travel_delta = (
                        self.route_travel(merged)
                        - self.route_travel(routes[i])
                        - self.route_travel(routes[j])
                    )
                    rough.append((travel_delta, i, j, merged))

            if not rough:
                return None

            rough.sort(key=lambda z: z[0])
            if randomized:
                pool_limit = min(len(rough), max(30, 4 * len(routes)))
                candidates = rough[:pool_limit]
                self.rng.shuffle(candidates)
                candidates.sort(
                    key=lambda z: z[0] + self.rng.random() * max(1.0, abs(z[0]) * 0.15)
                )
            else:
                candidates = rough

            best = None
            best_delta = float("inf")
            eval_limit = min(len(candidates), max(50, 5 * len(routes)))

            for travel_delta, i, j, merged in candidates[:eval_limit]:
                penalty_new, allowed = self.evaluate_route_set(merged)
                if not allowed:
                    continue
                penalty_i, _ = self.evaluate_route_set(routes[i])
                penalty_j, _ = self.evaluate_route_set(routes[j])
                delta = travel_delta + penalty_new - penalty_i - penalty_j
                if delta < best_delta:
                    best_delta = delta
                    best = (i, j, merged)

            if best is None:
                # Search the remaining capacity-feasible pairs before giving up.
                for travel_delta, i, j, merged in candidates[eval_limit:]:
                    if self.out_of_time(0.12):
                        break
                    penalty_new, allowed = self.evaluate_route_set(merged)
                    if not allowed:
                        continue
                    penalty_i, _ = self.evaluate_route_set(routes[i])
                    penalty_j, _ = self.evaluate_route_set(routes[j])
                    delta = travel_delta + penalty_new - penalty_i - penalty_j
                    if delta < best_delta:
                        best_delta = delta
                        best = (i, j, merged)

            if best is None:
                return None

            i, j, merged = best
            new_routes = []
            for index, route in enumerate(routes):
                if index not in (i, j):
                    new_routes.append(route)
            new_routes.append(merged)
            routes = new_routes

        return routes if len(routes) == self.K else None

    def assignment_fallback(self):
        if self.out_of_time(0.4):
            return self.greedy_assignment()

        try:
            import gurobipy as gp
            from gurobipy import GRB

            model = gp.Model()
            model.Params.OutputFlag = 0
            model.Params.Seed = 0
            model.Params.MIPGap = 1e-4
            model.Params.NumericFocus = 0
            model.Params.Threads = 1
            model.Params.TimeLimit = max(
                0.1, min(8.0, self.deadline - time.monotonic() - 0.15)
            )

            x = {}
            for c in self.customers:
                for k in range(self.K):
                    x[c, k] = model.addVar(vtype=GRB.BINARY)

            for c in self.customers:
                model.addConstr(gp.quicksum(x[c, k] for k in range(self.K)) == 1)

            for k in range(self.K):
                model.addConstr(gp.quicksum(x[c, k] for c in self.customers) >= 1)
                model.addConstr(
                    gp.quicksum(self.expected_area[c] * x[c, k] for c in self.customers)
                    <= self.capacity_area + 1e-7
                )
                model.addConstr(
                    gp.quicksum(self.expected_weight[c] * x[c, k] for c in self.customers)
                    <= self.Q + 1e-7
                )

            objective = gp.quicksum(
                (
                    2.0 * self.distance[self.depot][c]
                    + 1e-4 * ((c * 31 + k * 17) % 101)
                )
                * x[c, k]
                for c in self.customers
                for k in range(self.K)
            )
            model.setObjective(objective, GRB.MINIMIZE)
            model.optimize()

            if model.SolCount <= 0:
                return self.greedy_assignment()

            bins = [[] for _ in range(self.K)]
            for c in self.customers:
                for k in range(self.K):
                    if x[c, k].X > 0.5:
                        bins[k].append(c)
                        break

            return [self.nearest_insertion(route) for route in bins]
        except Exception:
            return self.greedy_assignment()

    def greedy_assignment(self):
        customers = sorted(
            self.customers,
            key=lambda c: max(
                self.expected_area[c] / max(1.0, self.capacity_area),
                self.expected_weight[c] / max(1.0, self.Q),
            ),
            reverse=True,
        )

        bins = [[] for _ in range(self.K)]
        area = [0.0] * self.K
        weight = [0.0] * self.K

        for k in range(min(self.K, len(customers))):
            c = customers[k]
            bins[k].append(c)
            area[k] += self.expected_area[c]
            weight[k] += self.expected_weight[c]

        for c in customers[self.K:]:
            feasible = []
            for k in range(self.K):
                if (
                    area[k] + self.expected_area[c] <= self.capacity_area + 1e-7
                    and weight[k] + self.expected_weight[c] <= self.Q + 1e-7
                ):
                    score = (
                        area[k] / max(1.0, self.capacity_area)
                        + weight[k] / max(1.0, self.Q)
                    )
                    feasible.append((score, k))
            if not feasible:
                return None
            _, k = min(feasible)
            bins[k].append(c)
            area[k] += self.expected_area[c]
            weight[k] += self.expected_weight[c]

        return [self.nearest_insertion(route) for route in bins]

    def nearest_insertion(self, customers):
        if not customers:
            return []

        remaining = set(customers)
        first = min(remaining, key=lambda c: self.distance[self.depot][c])
        route = [first]
        remaining.remove(first)

        while remaining:
            best = None
            for c in remaining:
                for pos in range(len(route) + 1):
                    prev_node = self.depot if pos == 0 else route[pos - 1]
                    next_node = self.depot if pos == len(route) else route[pos]
                    delta = (
                        self.distance[prev_node][c]
                        + self.distance[c][next_node]
                        - self.distance[prev_node][next_node]
                    )
                    if best is None or delta < best[0]:
                        best = (delta, c, pos)
            _, c, pos = best
            route.insert(pos, c)
            remaining.remove(c)

        return route

    def improve_route_order(self, routes):
        improved_any = True
        while improved_any and not self.out_of_time(0.08):
            improved_any = False
            for r_index, route in enumerate(routes):
                if len(route) < 3:
                    continue
                current_cost = self.route_travel(route)
                best_route = route
                best_cost = current_cost
                for i in range(len(route) - 1):
                    for j in range(i + 1, len(route)):
                        candidate = route[:i] + list(reversed(route[i:j + 1])) + route[j + 1:]
                        cost = self.route_travel(candidate)
                        if cost < best_cost:
                            best_cost = cost
                            best_route = candidate
                if best_cost < current_cost:
                    routes[r_index] = best_route
                    improved_any = True
                    self.register(routes)
        return routes

    def local_search(self, routes):
        routes = [list(r) for r in routes]
        routes = self.improve_route_order(routes)
        self.register(routes)

        no_improvement = 0
        iteration = 0

        while not self.out_of_time(0.12) and no_improvement < 600:
            iteration += 1
            move_type = self.rng.randrange(2)
            candidate_routes = [list(r) for r in routes]

            if move_type == 0:
                non_singletons = [i for i, r in enumerate(routes) if len(r) > 1]
                if not non_singletons or self.K < 2:
                    no_improvement += 1
                    continue
                a = self.rng.choice(non_singletons)
                b = self.rng.randrange(self.K - 1)
                if b >= a:
                    b += 1

                pos = self.rng.randrange(len(candidate_routes[a]))
                customer = candidate_routes[a].pop(pos)
                insert_pos = self.rng.randrange(len(candidate_routes[b]) + 1)
                candidate_routes[b].insert(insert_pos, customer)

                if not self.route_expected_capacity_feasible(candidate_routes[b]):
                    no_improvement += 1
                    continue
            else:
                if self.K < 2:
                    no_improvement += 1
                    continue
                a, b = self.rng.sample(range(self.K), 2)
                ia = self.rng.randrange(len(candidate_routes[a]))
                ib = self.rng.randrange(len(candidate_routes[b]))
                candidate_routes[a][ia], candidate_routes[b][ib] = (
                    candidate_routes[b][ib],
                    candidate_routes[a][ia],
                )
                if (
                    not self.route_expected_capacity_feasible(candidate_routes[a])
                    or not self.route_expected_capacity_feasible(candidate_routes[b])
                ):
                    no_improvement += 1
                    continue

            affected = {a, b}
            allowed = True
            for k in affected:
                _, ok = self.evaluate_route_set(candidate_routes[k])
                if not ok:
                    allowed = False
                    break
            if not allowed:
                no_improvement += 1
                continue

            for k in affected:
                candidate_routes[k] = self.nearest_insertion(candidate_routes[k])
                candidate_routes[k] = self._quick_two_opt(candidate_routes[k])

            current_obj = self.total_objective(routes)
            candidate_obj = self.total_objective(candidate_routes)

            if candidate_obj + 1e-8 < current_obj:
                routes = candidate_routes
                no_improvement = 0
                self.register(routes)
            else:
                no_improvement += 1

        return routes

    def _quick_two_opt(self, route):
        if len(route) < 3:
            return route
        route = list(route)
        improved = True
        while improved and not self.out_of_time(0.05):
            improved = False
            for i in range(len(route) - 1):
                prev_node = self.depot if i == 0 else route[i - 1]
                for j in range(i + 1, len(route)):
                    next_node = self.depot if j + 1 == len(route) else route[j + 1]
                    old = (
                        self.distance[prev_node][route[i]]
                        + self.distance[route[j]][next_node]
                    )
                    new = (
                        self.distance[prev_node][route[j]]
                        + self.distance[route[i]][next_node]
                    )
                    if new < old:
                        route[i:j + 1] = reversed(route[i:j + 1])
                        improved = True
                        break
                if improved:
                    break
        return route

    def solve(self):
        if self.n < self.K:
            # Such an instance cannot use all vehicles with nonempty routes.
            # Retain schema compliance while making the best possible partition.
            routes = [[c] for c in self.customers] + [[] for _ in range(self.K - self.n)]
            self.best_routes = routes
            self.best_objective = self.total_objective([r for r in routes if r])
            return

        initial = self.savings_construction(randomized=False)

        if initial is None and not self.out_of_time(0.5):
            for _ in range(3):
                candidate = self.savings_construction(randomized=True)
                if candidate is not None:
                    initial = candidate
                    break
                if self.out_of_time(0.5):
                    break

        if initial is None:
            initial = self.assignment_fallback()

        if initial is None:
            # Last-resort round-robin partition.
            initial = [[] for _ in range(self.K)]
            for index, c in enumerate(self.customers):
                initial[index % self.K].append(c)
            initial = [self.nearest_insertion(r) for r in initial]

        if len(initial) == self.K and all(initial):
            self.register(initial)
            if not self.out_of_time(0.2):
                self.local_search(initial)

        if self.best_routes is None:
            self.best_routes = initial
            self.best_objective = self.total_objective(initial)


def main():
    args = parse_args()
    start = time.monotonic()
    deadline = start + max(0, args.time_limit)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    solver = Solver(data, deadline, logger)
    solver.solve()

    solution = solver.make_solution(solver.best_routes, solver.best_objective)

    output_dir = os.path.dirname(os.path.abspath(args.solution_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(solution, f, separators=(",", ":"))


if __name__ == "__main__":
    main()
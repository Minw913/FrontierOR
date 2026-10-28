import json
import argparse
import time
import random
import math
from copy import deepcopy
from solution_logger import SolutionLogger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    return parser.parse_args()


def load_instance(path):
    with open(path, 'r') as f:
        return json.load(f)


class VRPBSolver:
    def __init__(self, instance, time_limit, logger=None):
        self.instance = instance
        self.time_limit = time_limit
        self.start_time = time.time()
        self.logger = logger

        self.num_lh = instance['num_linehaul_customers']
        self.num_bh = instance['num_backhaul_customers']
        self.num_vehicles = instance['num_vehicles']
        self.capacity = instance['vehicle_capacity']
        self.dist = instance['distance_matrix']
        self.depot = instance['depot']['index']

        self.lh_customers = instance['linehaul_customers']
        self.bh_customers = instance['backhaul_customers']

        self.lh_indices = [c['index'] for c in self.lh_customers]
        self.bh_indices = [c['index'] for c in self.bh_customers]
        self.lh_demands = {c['index']: c['demand'] for c in self.lh_customers}
        self.bh_demands = {c['index']: c['demand'] for c in self.bh_customers}

        self.best_routes = None
        self.best_cost = float('inf')

    def time_remaining(self):
        return self.time_limit - (time.time() - self.start_time)

    def route_cost(self, lh_seq, bh_seq):
        if not lh_seq:
            return float('inf')
        cost = self.dist[self.depot][lh_seq[0]]
        for i in range(len(lh_seq) - 1):
            cost += self.dist[lh_seq[i]][lh_seq[i + 1]]
        if bh_seq:
            cost += self.dist[lh_seq[-1]][bh_seq[0]]
            for i in range(len(bh_seq) - 1):
                cost += self.dist[bh_seq[i]][bh_seq[i + 1]]
            cost += self.dist[bh_seq[-1]][self.depot]
        else:
            cost += self.dist[lh_seq[-1]][self.depot]
        return cost

    def total_cost(self, routes):
        return sum(self.route_cost(lh, bh) for lh, bh in routes)

    def lh_demand(self, lh_seq):
        return sum(self.lh_demands[c] for c in lh_seq)

    def bh_demand(self, bh_seq):
        return sum(self.bh_demands[c] for c in bh_seq)

    def is_feasible(self, routes):
        if len(routes) != self.num_vehicles:
            return False
        visited_lh = set()
        visited_bh = set()
        for lh_seq, bh_seq in routes:
            if not lh_seq:
                return False
            if self.lh_demand(lh_seq) > self.capacity:
                return False
            if self.bh_demand(bh_seq) > self.capacity:
                return False
            for c in lh_seq:
                if c in visited_lh:
                    return False
                visited_lh.add(c)
            for c in bh_seq:
                if c in visited_bh:
                    return False
                visited_bh.add(c)
        return visited_lh == set(self.lh_indices) and visited_bh == set(self.bh_indices)

    def update_best(self, routes):
        cost = self.total_cost(routes)
        if cost < self.best_cost:
            self.best_cost = cost
            self.best_routes = deepcopy(routes)
            if self.logger:
                solution_dict = {
                    "objective_value": cost,
                    "routes": self.format_solution(routes)
                }
                self.logger.log_solution(cost, solution_dict)
        return cost

    def format_solution(self, routes):
        output_routes = []
        for lh_seq, bh_seq in routes:
            route = [self.depot] + lh_seq + bh_seq + [self.depot]
            output_routes.append(route)
        return output_routes

    # ==================== CONSTRUCTION ====================

    def construct_savings(self):
        """Clarke-Wright savings for linehaul, then insert backhaul"""
        if not self.lh_indices:
            return None

        # Initialize: each linehaul customer in its own route
        routes = [[c] for c in self.lh_indices]
        route_of = {c: i for i, c in enumerate(self.lh_indices)}
        active = list(range(len(self.lh_indices)))

        # Compute savings
        savings = []
        for i in range(len(self.lh_indices)):
            for j in range(i + 1, len(self.lh_indices)):
                ci, cj = self.lh_indices[i], self.lh_indices[j]
                s = (self.dist[self.depot][ci] + self.dist[cj][self.depot]
                     - self.dist[ci][cj])
                savings.append((s, ci, cj))
        savings.sort(reverse=True)

        for s, ci, cj in savings:
            if len(active) <= self.num_vehicles:
                break

            ri = route_of[ci]
            rj = route_of[cj]
            if ri == rj:
                continue
            if ri not in active or rj not in active:
                continue

            route_i = routes[ri]
            route_j = routes[rj]

            if self.lh_demand(route_i) + self.lh_demand(route_j) > self.capacity:
                continue

            # Try merging
            merged = None
            if route_i[-1] == ci and route_j[0] == cj:
                merged = route_i + route_j
            elif route_i[-1] == ci and route_j[-1] == cj:
                merged = route_i + route_j[::-1]
            elif route_i[0] == ci and route_j[0] == cj:
                merged = route_i[::-1] + route_j
            elif route_i[0] == ci and route_j[-1] == cj:
                merged = route_j + route_i

            if merged is None:
                continue

            routes[ri] = merged
            active.remove(rj)
            for c in merged:
                route_of[c] = ri

        result = [routes[i] for i in active]

        # If too many routes, force merge
        while len(result) > self.num_vehicles:
            best_pair = None
            best_incr = float('inf')
            for i in range(len(result)):
                for j in range(i + 1, len(result)):
                    if self.lh_demand(result[i]) + self.lh_demand(result[j]) > self.capacity:
                        continue
                    for ri in [result[i], result[i][::-1]]:
                        for rj in [result[j], result[j][::-1]]:
                            merged = ri + rj
                            incr = self.route_cost(merged, []) - self.route_cost(result[i], []) - self.route_cost(result[j], [])
                            if incr < best_incr:
                                best_incr = incr
                                best_pair = (i, j, merged)
            if best_pair is None:
                break
            i, j, merged = best_pair
            result[i] = merged
            result.pop(j)

        # Convert to (lh, []) format and insert backhauls
        route_tuples = [(r, []) for r in result]
        route_tuples = self.insert_backhauls(route_tuples)
        return route_tuples

    def construct_nn(self, seed=0):
        """Nearest neighbor construction"""
        rng = random.Random(seed)
        unvisited_lh = list(self.lh_indices)
        rng.shuffle(unvisited_lh)
        unvisited_bh = list(self.bh_indices)
        rng.shuffle(unvisited_bh)

        routes = []
        lh_ptr = 0

        for v in range(self.num_vehicles):
            if lh_ptr >= len(unvisited_lh):
                break
            lh_seq = []
            remaining = self.capacity
            current = self.depot

            while lh_ptr < len(unvisited_lh):
                # Find nearest feasible
                best_c = None
                best_d = float('inf')
                for idx in range(lh_ptr, len(unvisited_lh)):
                    c = unvisited_lh[idx]
                    if self.lh_demands[c] <= remaining:
                        d = self.dist[current][c]
                        if d < best_d:
                            best_d = d
                            best_c = c
                            best_idx = idx

                if best_c is None:
                    break

                lh_seq.append(best_c)
                remaining -= self.lh_demands[best_c]
                current = best_c
                # Remove from unvisited by swapping
                unvisited_lh[best_idx] = unvisited_lh[lh_ptr]
                unvisited_lh[lh_ptr] = best_c
                lh_ptr += 1

            if lh_seq:
                routes.append((lh_seq, []))

        # Distribute remaining linehaul customers
        while lh_ptr < len(unvisited_lh):
            c = unvisited_lh[lh_ptr]
            best_r = -1
            best_remaining = -1
            for r_idx, (lh_seq, bh_seq) in enumerate(routes):
                rem = self.capacity - self.lh_demand(lh_seq)
                if rem >= self.lh_demands[c] and rem > best_remaining:
                    best_remaining = rem
                    best_r = r_idx
            if best_r >= 0:
                routes[best_r] = (routes[best_r][0] + [c], routes[best_r][1])
            else:
                routes[0] = (routes[0][0] + [c], routes[0][1])
            lh_ptr += 1

        # Insert backhauls
        routes = self.insert_backhauls(routes)
        return routes

    def insert_backhauls(self, routes):
        """Insert backhaul customers using cheapest insertion"""
        bh_sorted = sorted(self.bh_indices, key=lambda c: self.bh_demands[c], reverse=True)

        for bh in bh_sorted:
            best_r = -1
            best_pos = -1
            best_incr = float('inf')

            for r_idx, (lh_seq, bh_seq) in enumerate(routes):
                if self.bh_demand(bh_seq) + self.bh_demands[bh] > self.capacity:
                    continue
                for pos in range(len(bh_seq) + 1):
                    new_bh = bh_seq[:pos] + [bh] + bh_seq[pos:]
                    incr = self.route_cost(lh_seq, new_bh) - self.route_cost(lh_seq, bh_seq)
                    if incr < best_incr:
                        best_incr = incr
                        best_r = r_idx
                        best_pos = pos

            if best_r >= 0:
                lh_seq, bh_seq = routes[best_r]
                routes[best_r] = (lh_seq, bh_seq[:best_pos] + [bh] + bh_seq[best_pos:])
            else:
                # Force insert into route with most remaining backhaul capacity
                best_r = 0
                best_rem = -1
                for r_idx, (lh_seq, bh_seq) in enumerate(routes):
                    rem = self.capacity - self.bh_demand(bh_seq)
                    if rem > best_rem:
                        best_rem = rem
                        best_r = r_idx
                lh_seq, bh_seq = routes[best_r]
                routes[best_r] = (lh_seq, bh_seq + [bh])

        return routes

    def construct_giant_tour_split(self, seed=0):
        """Build a giant tour and split into routes"""
        rng = random.Random(seed)
        lh_order = list(self.lh_indices)
        rng.shuffle(lh_order)

        # Split into num_vehicles groups respecting capacity
        groups = []
        current_group = []
        current_demand = 0

        for c in lh_order:
            if current_demand + self.lh_demands[c] > self.capacity and current_group:
                groups.append(current_group)
                current_group = [c]
                current_demand = self.lh_demands[c]
            else:
                current_group.append(c)
                current_demand += self.lh_demands[c]

        if current_group:
            groups.append(current_group)

        # If we have fewer groups than vehicles, split largest groups
        while len(groups) < self.num_vehicles and any(len(g) > 1 for g in groups):
            # Split the largest group
            largest_idx = max(range(len(groups)), key=lambda i: len(groups[i]))
            g = groups[largest_idx]
            if len(g) > 1:
                mid = len(g) // 2
                groups[largest_idx] = g[:mid]
                groups.append(g[mid:])

        # If we have more groups than vehicles, merge smallest
        while len(groups) > self.num_vehicles:
            # Merge two smallest that fit
            merged = False
            groups.sort(key=lambda g: self.lh_demand(g))
            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    if self.lh_demand(groups[i]) + self.lh_demand(groups[j]) <= self.capacity:
                        groups[i] = groups[i] + groups[j]
                        groups.pop(j)
                        merged = True
                        break
                if merged:
                    break
            if not merged:
                groups[0] = groups[0] + groups[-1]
                groups.pop()

        routes = [(g, []) for g in groups]
        routes = self.insert_backhauls(routes)
        return routes

    # ==================== LOCAL SEARCH ====================

    def two_opt_lh(self, routes):
        """2-opt on linehaul segments"""
        improved = True
        while improved:
            improved = False
            for r_idx in range(len(routes)):
                lh_seq, bh_seq = routes[r_idx]
                n = len(lh_seq)
                if n < 3:
                    continue
                best_improvement = 0
                best_i, best_j = -1, -1
                for i in range(n - 1):
                    for j in range(i + 2, n):
                        # Cost of reversing segment [i+1..j]
                        # Old edges: (lh[i], lh[i+1]) and (lh[j], lh[j+1] or next)
                        # New edges: (lh[i], lh[j]) and (lh[i+1], lh[j+1] or next)
                        if j == n - 1:
                            # Next after j is first backhaul or depot
                            if bh_seq:
                                next_node = bh_seq[0]
                            else:
                                next_node = self.depot
                            old_cost = self.dist[lh_seq[i]][lh_seq[i + 1]] + self.dist[lh_seq[j]][next_node]
                            new_cost = self.dist[lh_seq[i]][lh_seq[j]] + self.dist[lh_seq[i + 1]][next_node]
                        else:
                            old_cost = self.dist[lh_seq[i]][lh_seq[i + 1]] + self.dist[lh_seq[j]][lh_seq[j + 1]]
                            new_cost = self.dist[lh_seq[i]][lh_seq[j]] + self.dist[lh_seq[i + 1]][lh_seq[j + 1]]

                        improvement = old_cost - new_cost
                        if improvement > best_improvement + 1e-10:
                            best_improvement = improvement
                            best_i, best_j = i, j

                if best_i >= 0:
                    new_lh = lh_seq[:best_i + 1] + lh_seq[best_i + 1:best_j + 1][::-1] + lh_seq[best_j + 1:]
                    routes[r_idx] = (new_lh, bh_seq)
                    improved = True
        return routes

    def two_opt_bh(self, routes):
        """2-opt on backhaul segments"""
        improved = True
        while improved:
            improved = False
            for r_idx in range(len(routes)):
                lh_seq, bh_seq = routes[r_idx]
                n = len(bh_seq)
                if n < 3:
                    continue
                best_improvement = 0
                best_i, best_j = -1, -1
                for i in range(n - 1):
                    for j in range(i + 2, n):
                        if i == 0:
                            prev_node = lh_seq[-1]
                        else:
                            prev_node = bh_seq[i - 1]

                        if j == n - 1:
                            next_node = self.depot
                        else:
                            next_node = bh_seq[j + 1]

                        old_cost = self.dist[prev_node][bh_seq[i]] + self.dist[bh_seq[j]][next_node]
                        new_cost = self.dist[prev_node][bh_seq[j]] + self.dist[bh_seq[i]][next_node]

                        improvement = old_cost - new_cost
                        if improvement > best_improvement + 1e-10:
                            best_improvement = improvement
                            best_i, best_j = i, j

                if best_i >= 0:
                    new_bh = bh_seq[:best_i] + bh_seq[best_i:best_j + 1][::-1] + bh_seq[best_j + 1:]
                    routes[r_idx] = (lh_seq, new_bh)
                    improved = True
        return routes

    def or_opt(self, routes):
        """Or-opt: relocate sequences of 1, 2, 3 customers"""
        improved = True
        while improved:
            improved = False

            # Relocate linehaul customers
            for seg_len in [1, 2, 3]:
                for r1 in range(len(routes)):
                    if self.time_remaining() < 0.5:
                        return routes
                    lh1, bh1 = routes[r1]
                    if len(lh1) <= seg_len:
                        continue
                    for pos in range(len(lh1) - seg_len + 1):
                        segment = lh1[pos:pos + seg_len]
                        new_lh1 = lh1[:pos] + lh1[pos + seg_len:]
                        if not new_lh1:
                            continue

                        old_cost1 = self.route_cost(lh1, bh1)
                        new_cost1 = self.route_cost(new_lh1, bh1)
                        saving1 = old_cost1 - new_cost1

                        # Try inserting into same route
                        for pos2 in range(len(new_lh1) + 1):
                            new_lh_test = new_lh1[:pos2] + segment + new_lh1[pos2:]
                            cost_test = self.route_cost(new_lh_test, bh1)
                            if cost_test < old_cost1 - 1e-10:
                                routes[r1] = (new_lh_test, bh1)
                                improved = True
                                break
                        if improved:
                            break

                        # Try inserting into other routes
                        for r2 in range(len(routes)):
                            if r2 == r1:
                                continue
                            lh2, bh2 = routes[r2]
                            if self.lh_demand(lh2) + self.lh_demand(segment) > self.capacity:
                                continue
                            old_cost2 = self.route_cost(lh2, bh2)
                            best_pos2 = -1
                            best_cost2 = float('inf')
                            for pos2 in range(len(lh2) + 1):
                                new_lh2 = lh2[:pos2] + segment + lh2[pos2:]
                                c2 = self.route_cost(new_lh2, bh2)
                                if c2 < best_cost2:
                                    best_cost2 = c2
                                    best_pos2 = pos2

                            if saving1 + (old_cost2 - best_cost2) > 1e-10:
                                routes[r1] = (new_lh1, bh1)
                                routes[r2] = (lh2[:best_pos2] + segment + lh2[best_pos2:], bh2)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break

            if improved:
                continue

            # Relocate backhaul customers
            for seg_len in [1, 2, 3]:
                for r1 in range(len(routes)):
                    if self.time_remaining() < 0.5:
                        return routes
                    lh1, bh1 = routes[r1]
                    if len(bh1) <= seg_len:
                        continue
                    for pos in range(len(bh1) - seg_len + 1):
                        segment = bh1[pos:pos + seg_len]
                        new_bh1 = bh1[:pos] + bh1[pos + seg_len:]

                        old_cost1 = self.route_cost(lh1, bh1)
                        new_cost1 = self.route_cost(lh1, new_bh1)
                        saving1 = old_cost1 - new_cost1

                        # Same route
                        for pos2 in range(len(new_bh1) + 1):
                            new_bh_test = new_bh1[:pos2] + segment + new_bh1[pos2:]
                            cost_test = self.route_cost(lh1, new_bh_test)
                            if cost_test < old_cost1 - 1e-10:
                                routes[r1] = (lh1, new_bh_test)
                                improved = True
                                break
                        if improved:
                            break

                        # Other routes
                        for r2 in range(len(routes)):
                            if r2 == r1:
                                continue
                            lh2, bh2 = routes[r2]
                            if self.bh_demand(bh2) + self.bh_demand(segment) > self.capacity:
                                continue
                            old_cost2 = self.route_cost(lh2, bh2)
                            best_pos2 = -1
                            best_cost2 = float('inf')
                            for pos2 in range(len(bh2) + 1):
                                new_bh2 = bh2[:pos2] + segment + bh2[pos2:]
                                c2 = self.route_cost(lh2, new_bh2)
                                if c2 < best_cost2:
                                    best_cost2 = c2
                                    best_pos2 = pos2

                            if saving1 + (old_cost2 - best_cost2) > 1e-10:
                                routes[r1] = (lh1, new_bh1)
                                routes[r2] = (lh2, bh2[:best_pos2] + segment + bh2[best_pos2:])
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break

        return routes

    def cross_route_swap(self, routes):
        """Swap single customers between routes"""
        improved = True
        while improved:
            improved = False
            if self.time_remaining() < 0.5:
                break

            # Swap linehaul customers between routes
            for r1 in range(len(routes)):
                lh1, bh1 = routes[r1]
                for p1 in range(len(lh1)):
                    c1 = lh1[p1]
                    for r2 in range(r1 + 1, len(routes)):
                        lh2, bh2 = routes[r2]
                        for p2 in range(len(lh2)):
                            c2 = lh2[p2]
                            # Check capacity
                            new_demand1 = self.lh_demand(lh1) - self.lh_demands[c1] + self.lh_demands[c2]
                            new_demand2 = self.lh_demand(lh2) - self.lh_demands[c2] + self.lh_demands[c1]
                            if new_demand1 > self.capacity or new_demand2 > self.capacity:
                                continue

                            new_lh1 = lh1[:p1] + [c2] + lh1[p1 + 1:]
                            new_lh2 = lh2[:p2] + [c1] + lh2[p2 + 1:]

                            old_cost = self.route_cost(lh1, bh1) + self.route_cost(lh2, bh2)
                            new_cost = self.route_cost(new_lh1, bh1) + self.route_cost(new_lh2, bh2)

                            if new_cost < old_cost - 1e-10:
                                routes[r1] = (new_lh1, bh1)
                                routes[r2] = (new_lh2, bh2)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break

            if improved:
                continue

            # Swap backhaul customers between routes
            for r1 in range(len(routes)):
                lh1, bh1 = routes[r1]
                for p1 in range(len(bh1)):
                    c1 = bh1[p1]
                    for r2 in range(r1 + 1, len(routes)):
                        lh2, bh2 = routes[r2]
                        for p2 in range(len(bh2)):
                            c2 = bh2[p2]
                            new_demand1 = self.bh_demand(bh1) - self.bh_demands[c1] + self.bh_demands[c2]
                            new_demand2 = self.bh_demand(bh2) - self.bh_demands[c2] + self.bh_demands[c1]
                            if new_demand1 > self.capacity or new_demand2 > self.capacity:
                                continue

                            new_bh1 = bh1[:p1] + [c2] + bh1[p1 + 1:]
                            new_bh2 = bh2[:p2] + [c1] + bh2[p2 + 1:]

                            old_cost = self.route_cost(lh1, bh1) + self.route_cost(lh2, bh2)
                            new_cost = self.route_cost(lh1, new_bh1) + self.route_cost(lh2, new_bh2)

                            if new_cost < old_cost - 1e-10:
                                routes[r1] = (lh1, new_bh1)
                                routes[r2] = (lh2, new_bh2)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break

        return routes

    def local_search(self, routes):
        """Apply all local search operators"""
        max_passes = 50
        for _ in range(max_passes):
            if self.time_remaining() < 1.0:
                break
            old_cost = self.total_cost(routes)

            routes = self.two_opt_lh(routes)
            routes = self.two_opt_bh(routes)
            routes = self.or_opt(routes)
            routes = self.cross_route_swap(routes)

            new_cost = self.total_cost(routes)
            if abs(new_cost - old_cost) < 1e-10:
                break

        return routes

    # ==================== METAHEURISTIC ====================

    def perturb(self, routes, rng):
        """Perturbation for ILS"""
        routes = deepcopy(routes)
        num_moves = rng.randint(1, 3)

        for _ in range(num_moves):
            r_idx = rng.randint(0, len(routes) - 1)
            lh_seq, bh_seq = routes[r_idx]

            choice = rng.random()

            if choice < 0.4 and len(lh_seq) > 1:
                # Relocate a linehaul customer
                pos = rng.randint(0, len(lh_seq) - 1)
                customer = lh_seq[pos]
                new_lh = lh_seq[:pos] + lh_seq[pos + 1:]

                other_routes = [i for i in range(len(routes)) if i != r_idx and routes[i][0]]
                if other_routes:
                    target = rng.choice(other_routes)
                    tlh, tbh = routes[target]
                    if self.lh_demand(tlh) + self.lh_demands[customer] <= self.capacity:
                        insert_pos = rng.randint(0, len(tlh))
                        routes[target] = (tlh[:insert_pos] + [customer] + tlh[insert_pos:], tbh)
                        routes[r_idx] = (new_lh, bh_seq)
                    else:
                        insert_pos = rng.randint(0, len(new_lh))
                        routes[r_idx] = (new_lh[:insert_pos] + [customer] + new_lh[insert_pos:], bh_seq)
                else:
                    insert_pos = rng.randint(0, len(new_lh))
                    routes[r_idx] = (new_lh[:insert_pos] + [customer] + new_lh[insert_pos:], bh_seq)

            elif choice < 0.7 and len(bh_seq) > 0:
                # Relocate a backhaul customer
                pos = rng.randint(0, len(bh_seq) - 1)
                customer = bh_seq[pos]
                new_bh = bh_seq[:pos] + bh_seq[pos + 1:]

                other_routes = [i for i in range(len(routes)) if i != r_idx]
                if other_routes:
                    target = rng.choice(other_routes)
                    tlh, tbh = routes[target]
                    if self.bh_demand(tbh) + self.bh_demands[customer] <= self.capacity:
                        insert_pos = rng.randint(0, len(tbh))
                        routes[target] = (tlh, tbh[:insert_pos] + [customer] + tbh[insert_pos:])
                        routes[r_idx] = (lh_seq, new_bh)
                    else:
                        insert_pos = rng.randint(0, len(new_bh))
                        routes[r_idx] = (lh_seq, new_bh[:insert_pos] + [customer] + new_bh[insert_pos:])
                else:
                    insert_pos = rng.randint(0, len(new_bh))
                    routes[r_idx] = (lh_seq, new_bh[:insert_pos] + [customer] + new_bh[insert_pos:])

            elif len(lh_seq) > 2:
                # Reverse a random segment in linehaul
                i = rng.randint(0, len(lh_seq) - 2)
                j = rng.randint(i + 1, len(lh_seq) - 1)
                new_lh = lh_seq[:i] + lh_seq[i:j + 1][::-1] + lh_seq[j + 1:]
                routes[r_idx] = (new_lh, bh_seq)

            elif len(bh_seq) > 2:
                # Reverse a random segment in backhaul
                i = rng.randint(0, len(bh_seq) - 2)
                j = rng.randint(i + 1, len(bh_seq) - 1)
                new_bh = bh_seq[:i] + bh_seq[i:j + 1][::-1] + bh_seq[j + 1:]
                routes[r_idx] = (lh_seq, new_bh)

        return routes

    def iterated_local_search(self, routes):
        """ILS with adaptive acceptance"""
        rng = random.Random(42)
        current = deepcopy(routes)
        current_cost = self.total_cost(current)
        best = deepcopy(current)
        best_cost = current_cost

        iteration = 0
        no_improve_count = 0

        while self.time_remaining() > 2.0:
            iteration += 1

            # Perturb from best or current
            if no_improve_count > 20:
                base = deepcopy(current)
            else:
                base = deepcopy(best)

            perturbed = self.perturb(base, rng)
            perturbed = self.local_search(perturbed)
            perturbed_cost = self.total_cost(perturbed)

            if perturbed_cost < best_cost - 1e-10:
                best = deepcopy(perturbed)
                best_cost = perturbed_cost
                no_improve_count = 0
                self.update_best(best)
            else:
                no_improve_count += 1

            # Acceptance
            if perturbed_cost < current_cost:
                current = perturbed
                current_cost = perturbed_cost
            else:
                # Simulated annealing acceptance
                temp = max(0.01, 50.0 * (1.0 - iteration / 500.0))
                delta = perturbed_cost - current_cost
                if delta > 0 and rng.random() < math.exp(-delta / temp):
                    current = perturbed
                    current_cost = perturbed_cost

        return best

    # ==================== MAIN SOLVE ====================

    def solve(self):
        rng = random.Random(42)

        # Try multiple construction methods
        candidates = []

        # Savings-based
        if self.time_remaining() > 1:
            routes = self.construct_savings()
            if routes and self.is_feasible(routes):
                routes = self.local_search(routes)
                cost = self.update_best(routes)
                candidates.append((cost, routes))

        # Nearest neighbor with different seeds
        for seed in range(5):
            if self.time_remaining() < 2:
                break
            routes = self.construct_nn(seed=seed)
            if routes and self.is_feasible(routes):
                routes = self.local_search(routes)
                cost = self.update_best(routes)
                candidates.append((cost, routes))

        # Giant tour split
        for seed in range(3):
            if self.time_remaining() < 2:
                break
            routes = self.construct_giant_tour_split(seed=seed)
            if routes and self.is_feasible(routes):
                routes = self.local_search(routes)
                cost = self.update_best(routes)
                candidates.append((cost, routes))

        if not candidates:
            # Fallback: simple construction
            routes = self.construct_nn(seed=0)
            if routes:
                self.update_best(routes)
                candidates.append((self.total_cost(routes), routes))

        # Pick best candidate for ILS
        if candidates:
            candidates.sort(key=lambda x: x[0])
            best_initial = candidates[0][1]
        else:
            best_initial = self.best_routes

        # ILS
        if self.time_remaining() > 3 and best_initial:
            self.iterated_local_search(best_initial)

        # Final: format output
        if self.best_routes is None:
            # Emergency fallback
            self.best_routes = self.construct_nn(seed=0)
            self.best_cost = self.total_cost(self.best_routes) if self.best_routes else 0

        output_routes = self.format_solution(self.best_routes)

        return {
            "objective_value": self.best_cost,
            "routes": output_routes
        }


def main():
    args = parse_args()
    instance = load_instance(args.instance_path)
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    solver = VRPBSolver(instance, args.time_limit, logger)
    solution = solver.solve()

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == '__main__':
    main()
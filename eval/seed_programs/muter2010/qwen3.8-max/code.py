import json
import argparse
import math
import time
import random
import copy
from solution_logger import SolutionLogger


def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def load_instance(instance_path):
    with open(instance_path, 'r') as f:
        data = json.load(f)
    return data


class VRPTWSolver:
    def __init__(self, instance, time_limit, logger=None):
        self.instance = instance
        self.time_limit = time_limit
        self.logger = logger
        self.start_time = time.time()

        self.num_customers = instance['num_customers']
        self.depot = instance['depot']
        self.vehicle_capacity = instance['vehicle_capacity']
        self.customers = instance['customers']

        # Node 0 = depot, nodes 1..num_customers = customers
        self.nodes = [self.depot] + self.customers
        self.n = self.num_customers + 1

        # Precompute distance matrix
        self.dist = [[0.0] * self.n for _ in range(self.n)]
        for i in range(self.n):
            for j in range(i + 1, self.n):
                d = euclidean_distance(
                    self.nodes[i]['x'], self.nodes[i]['y'],
                    self.nodes[j]['x'], self.nodes[j]['y']
                )
                self.dist[i][j] = d
                self.dist[j][i] = d

        # Demand, time windows, service times
        self.demand = [self.nodes[i]['demand'] for i in range(self.n)]
        self.ready = [self.nodes[i]['ready_time'] for i in range(self.n)]
        self.due = [self.nodes[i]['due_date'] for i in range(self.n)]
        self.service = [self.nodes[i]['service_time'] for i in range(self.n)]

        self.best_routes = None
        self.best_obj = float('inf')

    def time_remaining(self):
        return self.time_limit - (time.time() - self.start_time)

    def is_time_up(self):
        return self.time_remaining() < 0.5

    def compute_route_distance(self, route):
        if not route:
            return 0.0
        total = self.dist[0][route[0]]
        for i in range(len(route) - 1):
            total += self.dist[route[i]][route[i + 1]]
        total += self.dist[route[-1]][0]
        return total

    def compute_route_demand(self, route):
        return sum(self.demand[c] for c in route)

    def is_route_feasible(self, route):
        if not route:
            return True
        if self.compute_route_demand(route) > self.vehicle_capacity:
            return False
        current_time = self.ready[0]
        prev = 0
        for c in route:
            arrival = current_time + self.dist[prev][c]
            if arrival > self.due[c] + 1e-9:
                return False
            current_time = max(arrival, self.ready[c]) + self.service[c]
            prev = c
        arrival_depot = current_time + self.dist[prev][0]
        if arrival_depot > self.due[0] + 1e-9:
            return False
        return True

    def check_insertion_feasible(self, route, pos, customer):
        """Check if inserting customer at position pos in route is feasible."""
        new_route = route[:pos] + [customer] + route[pos:]
        return self.is_route_feasible(new_route)

    def total_distance(self, routes):
        return sum(self.compute_route_distance(r) for r in routes)

    def construct_initial_solution(self):
        """Construct initial solution using Solomon's I1 insertion heuristic."""
        unserved = set(range(1, self.n))
        routes = []

        while unserved:
            if self.is_time_up():
                break
            # Seed: customer with earliest due date
            seed = min(unserved, key=lambda c: (self.due[c], -self.dist[0][c]))
            unserved.remove(seed)
            route = [seed]

            # Iteratively insert customers
            while unserved:
                if self.is_time_up():
                    break
                best_cost = float('inf')
                best_pos = -1
                best_cust = None

                for c in unserved:
                    if self.demand[c] + self.compute_route_demand(route) > self.vehicle_capacity:
                        continue
                    for pos in range(len(route) + 1):
                        new_route = route[:pos] + [c] + route[pos:]
                        if self.is_route_feasible(new_route):
                            cost = self.compute_route_distance(new_route) - self.compute_route_distance(route)
                            if cost < best_cost:
                                best_cost = cost
                                best_pos = pos
                                best_cust = c

                if best_cust is not None:
                    route.insert(best_pos, best_cust)
                    unserved.remove(best_cust)
                else:
                    break

            routes.append(route)

        return routes

    def intra_route_2opt(self, route):
        """Apply 2-opt improvement within a route."""
        if len(route) <= 2:
            return route
        improved = True
        while improved and not self.is_time_up():
            improved = False
            for i in range(len(route) - 1):
                for j in range(i + 1, len(route)):
                    new_route = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                    if self.is_route_feasible(new_route):
                        new_dist = self.compute_route_distance(new_route)
                        old_dist = self.compute_route_distance(route)
                        if new_dist < old_dist - 1e-10:
                            route = new_route
                            improved = True
                            break
                if improved:
                    break
        return route

    def or_opt(self, route):
        """Or-opt: move segments of 1, 2, or 3 consecutive customers."""
        improved = True
        while improved and not self.is_time_up():
            improved = False
            for seg_len in [1, 2, 3]:
                if len(route) <= seg_len:
                    continue
                for i in range(len(route) - seg_len + 1):
                    segment = route[i:i + seg_len]
                    remaining = route[:i] + route[i + seg_len:]
                    for j in range(len(remaining) + 1):
                        if j == i:
                            continue
                        new_route = remaining[:j] + segment + remaining[j:]
                        if new_route == route:
                            continue
                        if self.is_route_feasible(new_route):
                            if self.compute_route_distance(new_route) < self.compute_route_distance(route) - 1e-10:
                                route = new_route
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
        return route

    def relocate(self, routes):
        """Move a customer from one route to another."""
        improved = True
        while improved and not self.is_time_up():
            improved = False
            for i in range(len(routes)):
                if self.is_time_up():
                    break
                for ci in range(len(routes[i])):
                    customer = routes[i][ci]
                    for j in range(len(routes)):
                        if i == j:
                            continue
                        if self.compute_route_demand(routes[j]) + self.demand[customer] > self.vehicle_capacity:
                            continue
                        for pos in range(len(routes[j]) + 1):
                            new_route_j = routes[j][:pos] + [customer] + routes[j][pos:]
                            if self.is_route_feasible(new_route_j):
                                new_route_i = routes[i][:ci] + routes[i][ci + 1:]
                                old_dist = self.compute_route_distance(routes[i]) + self.compute_route_distance(routes[j])
                                new_dist = self.compute_route_distance(new_route_i) + self.compute_route_distance(new_route_j)
                                if new_dist < old_dist - 1e-10:
                                    routes[i] = new_route_i
                                    routes[j] = new_route_j
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        routes = [r for r in routes if r]
        return routes

    def swap(self, routes):
        """Swap customers between routes."""
        improved = True
        while improved and not self.is_time_up():
            improved = False
            for i in range(len(routes)):
                if self.is_time_up():
                    break
                for j in range(i + 1, len(routes)):
                    for ci in range(len(routes[i])):
                        for cj in range(len(routes[j])):
                            ci_node = routes[i][ci]
                            cj_node = routes[j][cj]
                            new_demand_i = self.compute_route_demand(routes[i]) - self.demand[ci_node] + self.demand[cj_node]
                            new_demand_j = self.compute_route_demand(routes[j]) - self.demand[cj_node] + self.demand[ci_node]
                            if new_demand_i > self.vehicle_capacity or new_demand_j > self.vehicle_capacity:
                                continue
                            new_route_i = routes[i][:ci] + [cj_node] + routes[i][ci + 1:]
                            new_route_j = routes[j][:cj] + [ci_node] + routes[j][cj + 1:]
                            if self.is_route_feasible(new_route_i) and self.is_route_feasible(new_route_j):
                                old_dist = self.compute_route_distance(routes[i]) + self.compute_route_distance(routes[j])
                                new_dist = self.compute_route_distance(new_route_i) + self.compute_route_distance(new_route_j)
                                if new_dist < old_dist - 1e-10:
                                    routes[i] = new_route_i
                                    routes[j] = new_route_j
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        return routes

    def cross_exchange(self, routes):
        """2-opt* between routes: exchange tails after positions ci and cj."""
        improved = True
        while improved and not self.is_time_up():
            improved = False
            for i in range(len(routes)):
                if self.is_time_up():
                    break
                for j in range(i + 1, len(routes)):
                    for ci in range(len(routes[i])):
                        for cj in range(len(routes[j])):
                            new_route_i = routes[i][:ci + 1] + routes[j][cj + 1:]
                            new_route_j = routes[j][:cj + 1] + routes[i][ci + 1:]
                            if self.compute_route_demand(new_route_i) <= self.vehicle_capacity and \
                               self.compute_route_demand(new_route_j) <= self.vehicle_capacity:
                                if self.is_route_feasible(new_route_i) and self.is_route_feasible(new_route_j):
                                    old_dist = self.compute_route_distance(routes[i]) + self.compute_route_distance(routes[j])
                                    new_dist = self.compute_route_distance(new_route_i) + self.compute_route_distance(new_route_j)
                                    if new_dist < old_dist - 1e-10:
                                        routes[i] = new_route_i
                                        routes[j] = new_route_j
                                        improved = True
                                        break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        routes = [r for r in routes if r]
        return routes

    def local_search(self, routes):
        """Apply local search operators."""
        for i in range(len(routes)):
            if self.is_time_up():
                break
            routes[i] = self.intra_route_2opt(routes[i])
            if not self.is_time_up():
                routes[i] = self.or_opt(routes[i])

        if not self.is_time_up():
            routes = self.relocate(routes)
        if not self.is_time_up():
            routes = self.swap(routes)
        if not self.is_time_up():
            routes = self.cross_exchange(routes)

        return routes

    def lns_destroy_random(self, routes, num_remove):
        all_customers = []
        for r in routes:
            all_customers.extend(r)
        if not all_customers:
            return routes, []
        removed = random.sample(all_customers, min(num_remove, len(all_customers)))
        removed_set = set(removed)
        new_routes = []
        for r in routes:
            new_r = [c for c in r if c not in removed_set]
            if new_r:
                new_routes.append(new_r)
        return new_routes, removed

    def lns_destroy_related(self, routes, num_remove):
        all_customers = []
        for r in routes:
            all_customers.extend(r)
        if not all_customers:
            return routes, []

        seed = random.choice(all_customers)
        removed = [seed]
        removed_set = {seed}

        while len(removed) < num_remove and len(removed_set) < len(all_customers):
            last = removed[-1]
            candidates = [(self.dist[last][c], c) for c in all_customers if c not in removed_set]
            if not candidates:
                break
            candidates.sort()
            idx = min(int(random.random() ** 2 * len(candidates)), len(candidates) - 1)
            _, next_c = candidates[idx]
            removed.append(next_c)
            removed_set.add(next_c)

        new_routes = []
        for r in routes:
            new_r = [c for c in r if c not in removed_set]
            if new_r:
                new_routes.append(new_r)
        return new_routes, removed

    def lns_destroy_worst(self, routes, num_remove):
        """Remove customers whose removal saves the most distance."""
        all_customers = []
        for r in routes:
            all_customers.extend(r)
        if not all_customers:
            return routes, []

        # Compute cost of each customer in its route
        costs = {}
        for r in routes:
            for idx, c in enumerate(r):
                prev_node = 0 if idx == 0 else r[idx - 1]
                next_node = 0 if idx == len(r) - 1 else r[idx + 1]
                cost_c = self.dist[prev_node][c] + self.dist[c][next_node] - self.dist[prev_node][next_node]
                costs[c] = cost_c

        sorted_customers = sorted(all_customers, key=lambda c: costs.get(c, 0), reverse=True)
        removed = sorted_customers[:min(num_remove, len(sorted_customers))]
        removed_set = set(removed)

        new_routes = []
        for r in routes:
            new_r = [c for c in r if c not in removed_set]
            if new_r:
                new_routes.append(new_r)
        return new_routes, removed

    def lns_repair(self, routes, removed):
        """Repair using regret-2 insertion."""
        unserved = list(removed)

        while unserved:
            if self.is_time_up():
                # Force insert remaining into new routes
                for c in unserved:
                    routes.append([c])
                unserved = []
                break

            best_regret = -float('inf')
            best_customer = None
            best_pos = None
            best_route_idx = None
            best_insert_cost = float('inf')

            for c in unserved:
                insertion_costs = []
                for ri, r in enumerate(routes):
                    if self.compute_route_demand(r) + self.demand[c] > self.vehicle_capacity:
                        continue
                    for pos in range(len(r) + 1):
                        new_r = r[:pos] + [c] + r[pos:]
                        if self.is_route_feasible(new_r):
                            cost = self.compute_route_distance(new_r) - self.compute_route_distance(r)
                            insertion_costs.append((cost, ri, pos))

                if not insertion_costs:
                    routes.append([c])
                    unserved.remove(c)
                    best_customer = None
                    break

                insertion_costs.sort()
                c_best = insertion_costs[0][0]
                if len(insertion_costs) > 1:
                    regret = insertion_costs[1][0] - c_best
                else:
                    regret = c_best

                if regret > best_regret or (regret == best_regret and c_best < best_insert_cost):
                    best_regret = regret
                    best_customer = c
                    best_pos = insertion_costs[0][2]
                    best_route_idx = insertion_costs[0][1]
                    best_insert_cost = c_best
            else:
                if best_customer is not None:
                    routes[best_route_idx] = routes[best_route_idx][:best_pos] + [best_customer] + routes[best_route_idx][best_pos:]
                    unserved.remove(best_customer)

        return routes

    def lns(self, routes, max_iterations=1000):
        """Large Neighborhood Search."""
        best_routes = [r[:] for r in routes]
        best_obj = self.total_distance(routes)

        destroy_ops = [self.lns_destroy_random, self.lns_destroy_related, self.lns_destroy_worst]
        destroy_weights = [1.0, 1.0, 1.0]
        destroy_scores = [0.0, 0.0, 0.0]
        destroy_counts = [0, 0, 0]

        total_customers = sum(len(r) for r in routes)
        temperature = best_obj * 0.05

        for it in range(max_iterations):
            if self.is_time_up():
                break

            # Select destroy operator
            total_w = sum(destroy_weights)
            r_val = random.random() * total_w
            cumulative = 0
            op_idx = 0
            for k in range(len(destroy_weights)):
                cumulative += destroy_weights[k]
                if r_val <= cumulative:
                    op_idx = k
                    break

            current_routes = [r[:] for r in best_routes]

            # Destroy
            num_remove = max(1, int(random.uniform(0.1, 0.4) * total_customers))
            destroyed, removed = destroy_ops[op_idx](current_routes, num_remove)
            destroy_counts[op_idx] += 1

            # Repair
            repaired = self.lns_repair(destroyed, removed)

            # Quick local search
            if not self.is_time_up() and random.random() < 0.3:
                for i in range(len(repaired)):
                    if self.is_time_up():
                        break
                    repaired[i] = self.or_opt(repaired[i])

            new_obj = self.total_distance(repaired)

            if new_obj < best_obj - 1e-10:
                best_obj = new_obj
                best_routes = [r[:] for r in repaired]
                destroy_scores[op_idx] += 2

                solution_dict = self.format_solution(best_routes, best_obj)
                if self.logger:
                    self.logger.log_solution(best_obj, solution_dict)
            elif new_obj < best_obj + temperature:
                # Accept with probability (simulated annealing)
                delta = new_obj - best_obj
                if temperature > 0 and random.random() < math.exp(-delta / max(temperature, 1e-10)):
                    best_routes = [r[:] for r in repaired]
                    best_obj = new_obj
                    destroy_scores[op_idx] += 1

            # Update temperature
            temperature *= 0.9995

            # Update weights periodically
            if (it + 1) % 50 == 0:
                for k in range(len(destroy_weights)):
                    if destroy_counts[k] > 0:
                        destroy_weights[k] = max(0.1, destroy_scores[k] / destroy_counts[k])
                    destroy_scores[k] = 0
                    destroy_counts[k] = 0

        return best_routes, best_obj

    def format_solution(self, routes, obj=None):
        if obj is None:
            obj = self.total_distance(routes)

        solution = {
            "objective_value": obj,
            "routes": []
        }

        for route in routes:
            customer_ids = list(route)
            path = [0] + list(route) + [0]
            solution["routes"].append({
                "customer_ids": customer_ids,
                "path": path
            })

        return solution

    def solve(self):
        random.seed(42)

        # Construct initial solution
        routes = self.construct_initial_solution()

        # Apply local search
        if not self.is_time_up():
            routes = self.local_search(routes)

        obj = self.total_distance(routes)
        self.best_routes = routes
        self.best_obj = obj

        solution_dict = self.format_solution(routes, obj)
        if self.logger:
            self.logger.log_solution(obj, solution_dict)

        # Apply LNS
        if not self.is_time_up():
            routes, obj = self.lns(routes, max_iterations=2000)
            self.best_routes = routes
            self.best_obj = obj

        # Final local search pass
        if not self.is_time_up():
            routes = self.local_search(routes)
            obj = self.total_distance(routes)
            if obj < self.best_obj:
                self.best_routes = routes
                self.best_obj = obj
                solution_dict = self.format_solution(routes, obj)
                if self.logger:
                    self.logger.log_solution(obj, solution_dict)

        return self.best_routes, self.best_obj


def main():
    parser = argparse.ArgumentParser(description='VRPTW Solver')
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    instance = load_instance(args.instance_path)

    solver = VRPTWSolver(instance, args.time_limit, logger)
    routes, obj = solver.solve()

    solution = solver.format_solution(routes, obj)

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

    print(f"Objective value: {obj:.4f}")
    print(f"Number of routes: {len(routes)}")


if __name__ == '__main__':
    main()
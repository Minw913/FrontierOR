import json
import argparse
import math
import time
import random
import copy
from solution_logger import SolutionLogger


def euclidean_dist(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


class VRPTWSolver:
    def __init__(self, instance, time_limit):
        self.time_limit = time_limit
        self.start_time = time.time()

        self.n_nodes = instance['problem_parameters']['n_nodes']
        self.n_customers = instance['problem_parameters']['n_customers']
        self.capacity = instance['problem_parameters']['vehicle_capacity']
        self.horizon = instance['problem_parameters']['planning_horizon']

        self.nodes = instance['nodes']

        # Identify depot and customers
        self.depot_source = None
        self.depot_sink = None
        self.customers = []
        for node in self.nodes:
            if node['type'] == 'depot_source':
                self.depot_source = node['id']
            elif node['type'] == 'depot_sink':
                self.depot_sink = node['id']
            else:
                self.customers.append(node['id'])

        # Build distance/time matrix
        self.dist = [[0.0] * self.n_nodes for _ in range(self.n_nodes)]
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if i != j:
                    self.dist[i][j] = euclidean_dist(
                        self.nodes[i]['x'], self.nodes[i]['y'],
                        self.nodes[j]['x'], self.nodes[j]['y']
                    )

        # Extract node attributes
        self.demand = [self.nodes[i]['demand'] for i in range(self.n_nodes)]
        self.service_time = [self.nodes[i]['service_time'] for i in range(self.n_nodes)]
        self.tw_open = [self.nodes[i]['time_window_open'] for i in range(self.n_nodes)]
        self.tw_close = [self.nodes[i]['time_window_close'] for i in range(self.n_nodes)]

        self.best_routes = None
        self.best_cost = float('inf')

    def time_remaining(self):
        return self.time_limit - (time.time() - self.start_time)

    def check_route_feasibility(self, route):
        """Check if a route (including depot_source and depot_sink) is feasible."""
        if len(route) < 2:
            return True

        # Check capacity
        total_demand = 0
        for node in route:
            if node != self.depot_source and node != self.depot_sink:
                total_demand += self.demand[node]
        if total_demand > self.capacity:
            return False

        # Check time windows
        current_time = self.tw_open[self.depot_source]
        for i in range(1, len(route)):
            prev = route[i - 1]
            curr = route[i]
            arrival = current_time + self.service_time[prev] + self.dist[prev][curr]
            start_service = max(arrival, self.tw_open[curr])
            if start_service > self.tw_close[curr] + 1e-6:
                return False
            current_time = start_service

        # Check return to depot within horizon
        last_node = route[-2] if len(route) >= 2 else self.depot_source
        if route[-1] == self.depot_sink:
            return_time = current_time + self.service_time[last_node] + self.dist[last_node][self.depot_sink]
            if return_time > self.horizon + 1e-6:
                return False

        return True

    def route_cost(self, route):
        cost = 0.0
        for i in range(len(route) - 1):
            cost += self.dist[route[i]][route[i + 1]]
        return cost

    def total_cost(self, routes):
        return sum(self.route_cost(r) for r in routes)

    def get_route_customers(self, route):
        return [n for n in route if n != self.depot_source and n != self.depot_sink]

    def construct_initial_solution(self):
        """Solomon's I1 insertion heuristic with regret."""
        unassigned = list(self.customers)
        routes = []

        while unassigned:
            if self.time_remaining() < 1:
                break

            # Start new route with farthest unassigned customer from depot
            best_c = max(unassigned, key=lambda c: self.dist[self.depot_source][c])
            route = [self.depot_source, best_c, self.depot_sink]
            unassigned.remove(best_c)

            # Insert remaining customers
            while unassigned:
                best_customer = None
                best_pos = None
                best_cost_increase = float('inf')

                for c in unassigned:
                    for pos in range(1, len(route)):
                        # Quick capacity check
                        current_demand = sum(self.demand[n] for n in route
                                           if n != self.depot_source and n != self.depot_sink)
                        if current_demand + self.demand[c] > self.capacity:
                            break

                        new_route = route[:pos] + [c] + route[pos:]
                        if self.check_route_feasibility(new_route):
                            cost_increase = (self.dist[route[pos - 1]][c] +
                                           self.dist[c][route[pos]] -
                                           self.dist[route[pos - 1]][route[pos]])
                            if cost_increase < best_cost_increase:
                                best_cost_increase = cost_increase
                                best_customer = c
                                best_pos = pos

                if best_customer is not None:
                    route = route[:best_pos] + [best_customer] + route[best_pos:]
                    unassigned.remove(best_customer)
                else:
                    break

            routes.append(route)

        # If time ran out, create individual routes for remaining
        for c in unassigned:
            routes.append([self.depot_source, c, self.depot_sink])

        return routes

    def local_search_intra(self, routes, max_passes=3):
        """Intra-route 2-opt improvement."""
        for _ in range(max_passes):
            if self.time_remaining() < 0.5:
                break
            improved = False
            for r_idx in range(len(routes)):
                route = routes[r_idx]
                n = len(route)
                for i in range(1, n - 2):
                    for j in range(i + 1, n - 1):
                        new_route = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                        if self.check_route_feasibility(new_route):
                            if self.route_cost(new_route) < self.route_cost(route) - 1e-9:
                                routes[r_idx] = new_route
                                route = new_route
                                improved = True
            if not improved:
                break
        return routes

    def local_search_inter(self, routes, max_passes=3):
        """Inter-route relocate and swap."""
        for _ in range(max_passes):
            if self.time_remaining() < 0.5:
                break
            improved = False

            # Relocate
            for r1 in range(len(routes)):
                if self.time_remaining() < 0.5:
                    break
                i = 1
                while i < len(routes[r1]) - 1:
                    customer = routes[r1][i]
                    new_r1 = routes[r1][:i] + routes[r1][i + 1:]
                    if len(new_r1) < 2:
                        i += 1
                        continue

                    old_cost = self.route_cost(routes[r1])
                    best_saving = 0
                    best_r2 = -1
                    best_pos = -1

                    for r2 in range(len(routes)):
                        if r2 == r1:
                            continue
                        # Quick capacity check
                        demand_r2 = sum(self.demand[n] for n in routes[r2]
                                       if n != self.depot_source and n != self.depot_sink)
                        if demand_r2 + self.demand[customer] > self.capacity:
                            continue

                        for pos in range(1, len(routes[r2])):
                            new_r2 = routes[r2][:pos] + [customer] + routes[r2][pos:]
                            if self.check_route_feasibility(new_r2):
                                new_cost = self.route_cost(new_r1) + self.route_cost(new_r2)
                                saving = old_cost + self.route_cost(routes[r2]) - new_cost
                                if saving > best_saving + 1e-9:
                                    best_saving = saving
                                    best_r2 = r2
                                    best_pos = pos

                    if best_r2 >= 0:
                        routes[r1] = new_r1
                        routes[best_r2] = (routes[best_r2][:best_pos] +
                                          [customer] + routes[best_r2][best_pos:])
                        improved = True
                        if len(routes[r1]) < 2:
                            routes.pop(r1)
                            break
                    else:
                        i += 1

            # Swap between routes
            for r1 in range(len(routes)):
                if self.time_remaining() < 0.5:
                    break
                for i in range(1, len(routes[r1]) - 1):
                    for r2 in range(r1 + 1, len(routes)):
                        for j in range(1, len(routes[r2]) - 1):
                            c1 = routes[r1][i]
                            c2 = routes[r2][j]

                            new_r1 = routes[r1][:i] + [c2] + routes[r1][i + 1:]
                            new_r2 = routes[r2][:j] + [c1] + routes[r2][j + 1:]

                            if (self.check_route_feasibility(new_r1) and
                                    self.check_route_feasibility(new_r2)):
                                old_cost = self.route_cost(routes[r1]) + self.route_cost(routes[r2])
                                new_cost = self.route_cost(new_r1) + self.route_cost(new_r2)
                                if new_cost < old_cost - 1e-9:
                                    routes[r1] = new_r1
                                    routes[r2] = new_r2
                                    improved = True

            if not improved:
                break

        return routes

    def destroy_random(self, routes, n_remove):
        """Random removal destroy operator."""
        all_customers = []
        for route in routes:
            for node in route:
                if node != self.depot_source and node != self.depot_sink:
                    all_customers.append(node)

        n_remove = min(n_remove, len(all_customers))
        removed = random.sample(all_customers, n_remove)
        removed_set = set(removed)

        new_routes = []
        for route in routes:
            new_route = [n for n in route if n not in removed_set]
            if len(new_route) >= 2:
                new_routes.append(new_route)

        return new_routes, removed

    def destroy_worst(self, routes, n_remove):
        """Worst removal - remove customers with highest removal cost."""
        all_customers = []
        customer_costs = {}

        for route in routes:
            for idx in range(1, len(route) - 1):
                c = route[idx]
                prev_node = route[idx - 1]
                next_node = route[idx + 1]
                cost_contribution = (self.dist[prev_node][c] + self.dist[c][next_node] -
                                    self.dist[prev_node][next_node])
                all_customers.append(c)
                customer_costs[c] = cost_contribution

        n_remove = min(n_remove, len(all_customers))

        # Sort by cost contribution descending, add randomness
        sorted_customers = sorted(all_customers, key=lambda c: customer_costs.get(c, 0), reverse=True)

        removed = []
        while len(removed) < n_remove and sorted_customers:
            # Pick from top with some randomness
            idx = int(len(sorted_customers) * (random.random() ** 2))
            removed.append(sorted_customers[idx])
            sorted_customers.pop(idx)

        removed_set = set(removed)
        new_routes = []
        for route in routes:
            new_route = [n for n in route if n not in removed_set]
            if len(new_route) >= 2:
                new_routes.append(new_route)

        return new_routes, removed

    def destroy_related(self, routes, n_remove):
        """Related removal - remove a cluster of related customers."""
        all_customers = []
        for route in routes:
            for node in route:
                if node != self.depot_source and node != self.depot_sink:
                    all_customers.append(node)

        n_remove = min(n_remove, len(all_customers))
        if n_remove == 0:
            return routes, []

        # Pick a random seed customer
        seed = random.choice(all_customers)
        removed = [seed]

        # Find most related customers
        while len(removed) < n_remove:
            last = removed[-1]
            candidates = [c for c in all_customers if c not in removed]
            if not candidates:
                break
            # Sort by distance to last removed
            candidates.sort(key=lambda c: self.dist[last][c])
            # Pick with randomness
            idx = int(len(candidates) * (random.random() ** 2))
            removed.append(candidates[idx])

        removed_set = set(removed)
        new_routes = []
        for route in routes:
            new_route = [n for n in route if n not in removed_set]
            if len(new_route) >= 2:
                new_routes.append(new_route)

        return new_routes, removed

    def repair_greedy(self, routes, unassigned):
        """Greedy insertion repair."""
        unassigned = list(unassigned)
        random.shuffle(unassigned)

        while unassigned:
            best_customer = None
            best_cost = float('inf')
            best_route_idx = -1
            best_pos = -1

            for c in unassigned:
                for r_idx, route in enumerate(routes):
                    # Quick capacity check
                    demand = sum(self.demand[n] for n in route
                               if n != self.depot_source and n != self.depot_sink)
                    if demand + self.demand[c] > self.capacity:
                        continue

                    for pos in range(1, len(route)):
                        new_route = route[:pos] + [c] + route[pos:]
                        if self.check_route_feasibility(new_route):
                            cost = self.route_cost(new_route) - self.route_cost(route)
                            if cost < best_cost:
                                best_cost = cost
                                best_customer = c
                                best_route_idx = r_idx
                                best_pos = pos

            if best_customer is None:
                # Create new route for first unassigned
                c = unassigned[0]
                routes.append([self.depot_source, c, self.depot_sink])
                unassigned.remove(c)
            else:
                routes[best_route_idx] = (routes[best_route_idx][:best_pos] +
                                         [best_customer] +
                                         routes[best_route_idx][best_pos:])
                unassigned.remove(best_customer)

        return routes

    def repair_regret(self, routes, unassigned, k=2):
        """Regret-k insertion repair."""
        unassigned = list(unassigned)

        while unassigned:
            best_customer = None
            best_regret = -float('inf')
            best_route_idx = -1
            best_pos = -1
            best_insert_cost = float('inf')

            for c in unassigned:
                # Find k best insertion positions
                insertion_options = []
                for r_idx, route in enumerate(routes):
                    demand = sum(self.demand[n] for n in route
                               if n != self.depot_source and n != self.depot_sink)
                    if demand + self.demand[c] > self.capacity:
                        continue

                    for pos in range(1, len(route)):
                        new_route = route[:pos] + [c] + route[pos:]
                        if self.check_route_feasibility(new_route):
                            cost = self.route_cost(new_route) - self.route_cost(route)
                            insertion_options.append((cost, r_idx, pos))

                # Also consider new route
                new_route = [self.depot_source, c, self.depot_sink]
                if self.check_route_feasibility(new_route):
                    cost = self.route_cost(new_route)
                    insertion_options.append((cost, -1, -1))

                if not insertion_options:
                    continue

                insertion_options.sort(key=lambda x: x[0])

                # Regret = difference between best and k-th best
                if len(insertion_options) >= k:
                    regret = sum(insertion_options[i][0] - insertion_options[0][0]
                               for i in range(1, k))
                else:
                    regret = float('inf')  # Must insert here

                if regret > best_regret or (regret == best_regret and
                                           insertion_options[0][0] < best_insert_cost):
                    best_regret = regret
                    best_customer = c
                    best_route_idx = insertion_options[0][1]
                    best_pos = insertion_options[0][2]
                    best_insert_cost = insertion_options[0][0]

            if best_customer is None:
                # Force create route for first unassigned
                c = unassigned[0]
                routes.append([self.depot_source, c, self.depot_sink])
                unassigned.remove(c)
            else:
                if best_route_idx == -1:
                    routes.append([self.depot_source, best_customer, self.depot_sink])
                else:
                    routes[best_route_idx] = (routes[best_route_idx][:best_pos] +
                                             [best_customer] +
                                             routes[best_route_idx][best_pos:])
                unassigned.remove(best_customer)

        return routes

    def solve(self, logger=None):
        random.seed(42)

        # Phase 1: Construction
        routes = self.construct_initial_solution()
        cost = self.total_cost(routes)

        self.best_routes = [r[:] for r in routes]
        self.best_cost = cost

        if logger:
            logger.log_solution(cost, {"objective_value": cost, "routes": routes})

        # Phase 2: Initial local search
        routes = self.local_search_intra(routes, max_passes=5)
        routes = self.local_search_inter(routes, max_passes=3)
        cost = self.total_cost(routes)

        if cost < self.best_cost:
            self.best_cost = cost
            self.best_routes = [r[:] for r in routes]
            if logger:
                logger.log_solution(cost, {"objective_value": cost, "routes": routes})

        # Phase 3: LNS
        n_customers = len(self.customers)
        current_routes = [r[:] for r in self.best_routes]
        current_cost = self.best_cost
        temperature = 0.05 * self.best_cost if self.best_cost > 0 else 1.0
        cooling_rate = 0.9995

        destroy_operators = [self.destroy_random, self.destroy_worst, self.destroy_related]
        repair_operators = [self.repair_greedy, self.repair_regret]

        iteration = 0
        no_improve_count = 0

        while self.time_remaining() > 1:
            iteration += 1

            # Adaptive destroy size
            if no_improve_count > 50:
                n_remove = max(1, min(int(0.4 * n_customers), 15))
                temperature = max(temperature * 2, 0.01 * self.best_cost)
                no_improve_count = 0
            else:
                n_remove = max(1, min(int(0.2 * n_customers) + random.randint(0, 3), 12))

            # Select operators
            destroy_op = random.choice(destroy_operators)
            repair_op = random.choice(repair_operators)

            # Destroy
            destroyed_routes, removed = destroy_op([r[:] for r in current_routes], n_remove)

            # Repair
            candidate_routes = repair_op(destroyed_routes, removed)

            # Quick local search
            candidate_routes = self.local_search_intra(candidate_routes, max_passes=2)

            candidate_cost = self.total_cost(candidate_routes)

            if candidate_cost < current_cost - 1e-9:
                current_routes = candidate_routes
                current_cost = candidate_cost
                no_improve_count = 0

                if current_cost < self.best_cost:
                    self.best_cost = current_cost
                    self.best_routes = [r[:] for r in current_routes]
                    if logger:
                        logger.log_solution(self.best_cost,
                                          {"objective_value": self.best_cost,
                                           "routes": self.best_routes})
            else:
                # Simulated annealing acceptance
                delta = candidate_cost - current_cost
                if temperature > 0 and delta > 0:
                    if random.random() < math.exp(-delta / temperature):
                        current_routes = candidate_routes
                        current_cost = candidate_cost
                no_improve_count += 1

            temperature *= cooling_rate

        return self.best_routes, self.best_cost


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    solver = VRPTWSolver(instance, args.time_limit)
    routes, cost = solver.solve(logger)

    solution = {
        "objective_value": cost,
        "routes": routes
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == '__main__':
    main()
import json
import argparse
import math
import random
import time
import copy
from solution_logger import SolutionLogger


def euclidean(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


class VRPTW:
    def __init__(self, instance):
        self.depot = instance['depot']
        self.customers = instance['customers']
        self.num_customers = instance['num_customers']
        self.capacity = instance['vehicle_capacity']

        nodes = [self.depot] + self.customers
        n = len(nodes)
        self.dist = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    self.dist[i][j] = euclidean(
                        (nodes[i]['x'], nodes[i]['y']),
                        (nodes[j]['x'], nodes[j]['y'])
                    )

        self.demand = [0] + [c['demand'] for c in self.customers]
        self.ready = [self.depot['ready_time']] + [c['ready_time'] for c in self.customers]
        self.due = [self.depot['due_date']] + [c['due_date'] for c in self.customers]
        self.service = [0] + [c['service_time'] for c in self.customers]
        self.depot_ready = self.depot['ready_time']
        self.depot_due = self.depot['due_date']
        self.N = self.num_customers

    def route_distance(self, route):
        if not route:
            return 0.0
        d = self.dist[0][route[0]]
        for i in range(len(route) - 1):
            d += self.dist[route[i]][route[i + 1]]
        d += self.dist[route[-1]][0]
        return d

    def is_feasible_route(self, route):
        if not route:
            return True
        load = 0
        t = self.depot_ready
        prev = 0
        for c in route:
            load += self.demand[c]
            if load > self.capacity:
                return False
            arrival = t + self.dist[prev][c]
            if arrival > self.due[c]:
                return False
            start = max(arrival, self.ready[c])
            t = start + self.service[c]
            prev = c
        arrival_depot = t + self.dist[prev][0]
        if arrival_depot > self.depot_due:
            return False
        return True

    def total_distance(self, routes):
        return sum(self.route_distance(r) for r in routes)

    def initial_solution(self):
        unserved = list(range(1, self.N + 1))
        routes = []
        while unserved:
            route = []
            current_time = self.depot_ready
            current_load = 0
            prev = 0
            while True:
                best_c = None
                best_dist = float('inf')
                best_arrival = None
                for c in unserved:
                    travel = self.dist[prev][c]
                    arrival = current_time + travel
                    if arrival > self.due[c]:
                        continue
                    if current_load + self.demand[c] > self.capacity:
                        continue
                    start = max(arrival, self.ready[c])
                    depart = start + self.service[c]
                    return_depot = depart + self.dist[c][0]
                    if return_depot > self.depot_due:
                        continue
                    if travel < best_dist:
                        best_dist = travel
                        best_c = c
                        best_arrival = arrival
                if best_c is None:
                    break
                route.append(best_c)
                unserved.remove(best_c)
                current_time = max(best_arrival, self.ready[best_c]) + self.service[best_c]
                current_load += self.demand[best_c]
                prev = best_c
            if route:
                routes.append(route)
            else:
                for c in unserved:
                    routes.append([c])
                break
        return routes

    def try_relocate(self, routes):
        n_routes = len(routes)
        if n_routes == 0:
            return False
        ri = random.randrange(n_routes)
        if not routes[ri]:
            return False
        ci = random.randrange(len(routes[ri]))
        c = routes[ri][ci]
        route_without = routes[ri][:ci] + routes[ri][ci + 1:]

        rj = random.randrange(n_routes + 1)
        if rj == n_routes:
            if self.is_feasible_route([c]):
                old_total = self.total_distance(routes)
                new_total = old_total - self.route_distance(routes[ri]) + self.route_distance(route_without) + self.route_distance([c])
                if new_total < old_total - 1e-6:
                    routes.append([c])
                    routes[ri] = route_without
                    if not routes[ri]:
                        routes.pop(ri)
                    return True
        else:
            if rj == ri:
                if len(routes[ri]) <= 1:
                    return False
                pos = random.randrange(len(route_without) + 1)
                if pos == ci:
                    return False
                new_route = route_without[:pos] + [c] + route_without[pos:]
                if self.is_feasible_route(new_route):
                    old_dist = self.route_distance(routes[ri])
                    new_dist = self.route_distance(new_route)
                    if new_dist < old_dist - 1e-6:
                        routes[ri] = new_route
                        return True
            else:
                pos = random.randrange(len(routes[rj]) + 1)
                new_route = routes[rj][:pos] + [c] + routes[rj][pos:]
                if self.is_feasible_route(route_without) and self.is_feasible_route(new_route):
                    old_dist = self.route_distance(routes[ri]) + self.route_distance(routes[rj])
                    new_dist = self.route_distance(route_without) + self.route_distance(new_route)
                    if new_dist < old_dist - 1e-6:
                        routes[ri] = route_without
                        routes[rj] = new_route
                        if not routes[ri]:
                            routes.pop(ri)
                        return True
        return False

    def try_swap(self, routes):
        n_routes = len(routes)
        if n_routes == 0:
            return False
        ri = random.randrange(n_routes)
        if not routes[ri]:
            return False
        ci = random.randrange(len(routes[ri]))
        rj = random.randrange(n_routes)
        if rj == ri:
            if len(routes[ri]) < 2:
                return False
            cj = random.randrange(len(routes[ri]))
            if cj == ci:
                return False
            route = routes[ri]
            c1 = route[ci]
            c2 = route[cj]
            new_route = route[:]
            new_route[ci] = c2
            new_route[cj] = c1
            if self.is_feasible_route(new_route):
                old_dist = self.route_distance(route)
                new_dist = self.route_distance(new_route)
                if new_dist < old_dist - 1e-6:
                    routes[ri] = new_route
                    return True
        else:
            if not routes[rj]:
                return False
            cj = random.randrange(len(routes[rj]))
            c1 = routes[ri][ci]
            c2 = routes[rj][cj]
            new_ri = routes[ri][:ci] + [c2] + routes[ri][ci + 1:]
            new_rj = routes[rj][:cj] + [c1] + routes[rj][cj + 1:]
            if self.is_feasible_route(new_ri) and self.is_feasible_route(new_rj):
                old_dist = self.route_distance(routes[ri]) + self.route_distance(routes[rj])
                new_dist = self.route_distance(new_ri) + self.route_distance(new_rj)
                if new_dist < old_dist - 1e-6:
                    routes[ri] = new_ri
                    routes[rj] = new_rj
                    return True
        return False

    def try_2opt(self, routes):
        n_routes = len(routes)
        if n_routes == 0:
            return False
        ri = random.randrange(n_routes)
        route = routes[ri]
        n = len(route)
        if n < 2:
            return False
        i = random.randrange(n - 1)
        j = random.randrange(i + 1, n)
        new_route = route[:i + 1] + route[i + 1:j + 1][::-1] + route[j + 1:]
        if self.is_feasible_route(new_route):
            old_dist = self.route_distance(route)
            new_dist = self.route_distance(new_route)
            if new_dist < old_dist - 1e-6:
                routes[ri] = new_route
                return True
        return False

    def try_oropt(self, routes):
        n_routes = len(routes)
        if n_routes == 0:
            return False
        ri = random.randrange(n_routes)
        route = routes[ri]
        n = len(route)
        if n < 2:
            return False
        seg_len = random.choice([1, 2, 3])
        if seg_len > n:
            return False
        start = random.randrange(n - seg_len + 1)
        segment = route[start:start + seg_len]
        route_without = route[:start] + route[start + seg_len:]

        rj = random.randrange(n_routes + 1)
        if rj == n_routes:
            if self.is_feasible_route(segment):
                old_total = self.total_distance(routes)
                new_total = old_total - self.route_distance(route) + self.route_distance(route_without) + self.route_distance(segment)
                if new_total < old_total - 1e-6:
                    routes.append(segment)
                    routes[ri] = route_without
                    if not routes[ri]:
                        routes.pop(ri)
                    return True
        else:
            if rj == ri:
                if len(route_without) == 0:
                    return False
                pos = random.randrange(len(route_without) + 1)
                if pos == start:
                    return False
                if pos == start + seg_len:
                    return False
                new_route = route_without[:pos] + segment + route_without[pos:]
                if self.is_feasible_route(new_route):
                    old_dist = self.route_distance(route)
                    new_dist = self.route_distance(new_route)
                    if new_dist < old_dist - 1e-6:
                        routes[ri] = new_route
                        return True
            else:
                pos = random.randrange(len(routes[rj]) + 1)
                new_route = routes[rj][:pos] + segment + routes[rj][pos:]
                if self.is_feasible_route(route_without) and self.is_feasible_route(new_route):
                    old_dist = self.route_distance(route) + self.route_distance(routes[rj])
                    new_dist = self.route_distance(route_without) + self.route_distance(new_route)
                    if new_dist < old_dist - 1e-6:
                        routes[ri] = route_without
                        routes[rj] = new_route
                        if not routes[ri]:
                            routes.pop(ri)
                        return True
        return False

    def local_search(self, routes, deadline, max_stagnation=3000):
        stagnation = 0
        while time.time() < deadline:
            move = random.choice(['relocate', 'swap', '2opt', 'oropt'])
            if move == 'relocate':
                improved = self.try_relocate(routes)
            elif move == 'swap':
                improved = self.try_swap(routes)
            elif move == '2opt':
                improved = self.try_2opt(routes)
            else:
                improved = self.try_oropt(routes)
            if improved:
                stagnation = 0
            else:
                stagnation += 1
                if stagnation >= max_stagnation:
                    break
        return routes

    def perturb(self, routes, strength=0.1):
        num_remove = max(1, int(strength * self.N))
        num_remove = min(num_remove, self.N)
        to_remove = random.sample(range(1, self.N + 1), num_remove)
        new_routes = []
        for route in routes:
            new_route = [c for c in route if c not in to_remove]
            if new_route:
                new_routes.append(new_route)
        random.shuffle(to_remove)
        for c in to_remove:
            best_delta = float('inf')
            best_insert = None
            for ri in range(len(new_routes)):
                for pos in range(len(new_routes[ri]) + 1):
                    new_route = new_routes[ri][:pos] + [c] + new_routes[ri][pos:]
                    if self.is_feasible_route(new_route):
                        delta = self.route_distance(new_route) - self.route_distance(new_routes[ri])
                        if delta < best_delta:
                            best_delta = delta
                            best_insert = (ri, pos)
            if self.is_feasible_route([c]):
                delta = self.route_distance([c])
                if delta < best_delta:
                    best_delta = delta
                    best_insert = (-1, 0)
            if best_insert is None:
                new_routes.append([c])
            else:
                if best_insert[0] == -1:
                    new_routes.append([c])
                else:
                    ri, pos = best_insert
                    new_routes[ri] = new_routes[ri][:pos] + [c] + new_routes[ri][pos:]
        return new_routes

    def solve(self, time_limit, logger=None):
        start = time.time()
        deadline = start + time_limit

        routes = self.initial_solution()
        best_routes = copy.deepcopy(routes)
        best_dist = self.total_distance(best_routes)
        if logger:
            logger.log_solution(best_dist, self.solution_dict(best_routes, best_dist))

        routes = self.local_search(routes, deadline)
        dist = self.total_distance(routes)
        if dist < best_dist:
            best_routes = copy.deepcopy(routes)
            best_dist = dist
            if logger:
                logger.log_solution(best_dist, self.solution_dict(best_routes, best_dist))

        while time.time() < deadline:
            new_routes = copy.deepcopy(routes)
            new_routes = self.perturb(new_routes)
            new_routes = self.local_search(new_routes, deadline)
            dist = self.total_distance(new_routes)
            if dist < best_dist:
                best_routes = copy.deepcopy(new_routes)
                best_dist = dist
                if logger:
                    logger.log_solution(best_dist, self.solution_dict(best_routes, best_dist))
            if dist < self.total_distance(routes) * 1.05:
                routes = new_routes

        return best_routes, best_dist

    def solution_dict(self, routes, dist):
        sol_routes = []
        depot_id = self.depot['id']
        for route in routes:
            if not route:
                continue
            customer_ids = [self.customers[c - 1]['id'] for c in route]
            path = [depot_id] + customer_ids + [depot_id]
            sol_routes.append({
                "customer_ids": customer_ids,
                "path": path
            })
        return {
            "objective_value": dist,
            "routes": sol_routes
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    random.seed(0)

    with open(args.instance_path) as f:
        instance = json.load(f)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    solver = VRPTW(instance)
    routes, dist = solver.solve(args.time_limit, logger)
    solution = solver.solution_dict(routes, dist)

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == '__main__':
    main()
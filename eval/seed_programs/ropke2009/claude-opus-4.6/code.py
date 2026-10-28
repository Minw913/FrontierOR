import argparse
import json
import math
import time
import random
import sys
from solution_logger import SolutionLogger


def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


class PDPSolver:
    def __init__(self, instance, time_limit, logger=None):
        self.instance = instance
        self.time_limit = time_limit
        self.logger = logger
        self.start_time = time.time()
        
        self.num_requests = instance['num_requests']
        self.num_vehicles = instance['num_vehicles']
        self.vehicle_capacity = instance['vehicle_capacity']
        self.nodes = instance['nodes']
        self.requests = instance['requests']
        self.num_nodes = instance['num_nodes']
        
        # Build node index map
        self.node_map = {}
        for node in self.nodes:
            self.node_map[node['id']] = node
        
        # Find depot origin and destination
        self.depot_origin = None
        self.depot_destination = None
        for node in self.nodes:
            if node['type'] == 'depot_origin':
                self.depot_origin = node['id']
            elif node['type'] == 'depot_destination':
                self.depot_destination = node['id']
        
        # Build pickup/delivery pairs
        self.pickup_nodes = []
        self.delivery_nodes = []
        self.pickup_to_request = {}
        self.delivery_to_request = {}
        self.request_pickup = {}
        self.request_delivery = {}
        
        for req in self.requests:
            pid = req['pickup_node']
            did = req['delivery_node']
            rid = req['request_id']
            self.pickup_nodes.append(pid)
            self.delivery_nodes.append(did)
            self.pickup_to_request[pid] = rid
            self.delivery_to_request[did] = rid
            self.request_pickup[rid] = pid
            self.request_delivery[rid] = did
        
        # Build request index mapping (request_id -> index in self.requests list)
        self.req_id_to_idx = {}
        for i, req in enumerate(self.requests):
            self.req_id_to_idx[req['request_id']] = i
        
        # Distance and travel time matrices
        self.node_ids = sorted(self.node_map.keys())
        self.id_to_idx = {nid: idx for idx, nid in enumerate(self.node_ids)}
        
        n = len(self.node_ids)
        self.dist = [[0.0] * n for _ in range(n)]
        self.tt = [[0.0] * n for _ in range(n)]
        
        for i in range(n):
            ni = self.node_map[self.node_ids[i]]
            for j in range(n):
                if i != j:
                    nj = self.node_map[self.node_ids[j]]
                    d = euclidean_distance(ni['x'], ni['y'], nj['x'], nj['y'])
                    self.dist[i][j] = d
                    self.tt[i][j] = d + ni['service_time']
        
        self.best_solution = None
        self.best_cost = float('inf')
    
    def time_remaining(self):
        return self.time_limit - (time.time() - self.start_time)
    
    def get_dist(self, nid1, nid2):
        return self.dist[self.id_to_idx[nid1]][self.id_to_idx[nid2]]
    
    def get_tt(self, nid1, nid2):
        return self.tt[self.id_to_idx[nid1]][self.id_to_idx[nid2]]
    
    def evaluate_route(self, route):
        """Evaluate a single route. Returns (cost, feasible)."""
        if len(route) == 0:
            return 0.0, True
        
        cost = 0.0
        current_time = self.node_map[self.depot_origin]['time_window'][0]
        current_load = 0
        prev = self.depot_origin
        
        for node_id in route:
            node = self.node_map[node_id]
            travel_time = self.get_tt(prev, node_id)
            travel_dist = self.get_dist(prev, node_id)
            cost += travel_dist
            
            arrival_time = current_time + travel_time
            tw = node['time_window']
            
            if arrival_time < tw[0]:
                current_time = tw[0]
            elif arrival_time > tw[1]:
                return float('inf'), False
            else:
                current_time = arrival_time
            
            current_load += node['demand']
            if current_load > self.vehicle_capacity or current_load < 0:
                return float('inf'), False
            
            prev = node_id
        
        # Return to depot
        travel_time = self.get_tt(prev, self.depot_destination)
        travel_dist = self.get_dist(prev, self.depot_destination)
        cost += travel_dist
        
        arrival_time = current_time + travel_time
        depot_tw = self.node_map[self.depot_destination]['time_window']
        if arrival_time > depot_tw[1]:
            return float('inf'), False
        
        return cost, True
    
    def evaluate_solution(self, routes):
        """Evaluate total cost of a solution."""
        total_cost = 0.0
        for route in routes:
            cost, feasible = self.evaluate_route(route)
            if not feasible:
                return float('inf'), False
            total_cost += cost
        return total_cost, True
    
    def check_route_feasibility(self, route):
        """Check if a route is feasible (time windows, capacity, precedence)."""
        if len(route) == 0:
            return True
        
        # Check precedence: pickup before delivery
        visited = set()
        for node_id in route:
            node = self.node_map[node_id]
            if node['type'] == 'delivery':
                pickup_node = node['pair']
                if pickup_node not in visited:
                    return False
            visited.add(node_id)
        
        _, feasible = self.evaluate_route(route)
        return feasible
    
    def find_best_insertion(self, route, pid, did):
        """Find best insertion positions for a pickup-delivery pair in a route."""
        n = len(route)
        best_cost = float('inf')
        best_route = None
        
        for pi in range(n + 1):
            for di in range(pi + 1, n + 2):
                new_route = route[:pi] + [pid] + route[pi:di-1] + [did] + route[di-1:]
                cost, feasible = self.evaluate_route(new_route)
                if feasible and cost < best_cost:
                    # Also check precedence
                    prec_ok = True
                    visited = set()
                    for nid in new_route:
                        nd = self.node_map[nid]
                        if nd['type'] == 'delivery':
                            if nd['pair'] not in visited:
                                prec_ok = False
                                break
                        visited.add(nid)
                    if prec_ok:
                        best_cost = cost
                        best_route = list(new_route)
        
        return best_route, best_cost
    
    def remove_request_from_route(self, route, req_id):
        """Remove a request (pickup and delivery) from a route by request_id."""
        pid = self.request_pickup[req_id]
        did = self.request_delivery[req_id]
        new_route = [n for n in route if n != pid and n != did]
        return new_route
    
    def get_requests_in_route(self, route):
        """Get list of request ids served by a route."""
        reqs = []
        for node_id in route:
            if node_id in self.pickup_to_request:
                reqs.append(self.pickup_to_request[node_id])
        return reqs
    
    def greedy_insertion(self):
        """Build initial solution using greedy insertion."""
        routes = [[] for _ in range(self.num_vehicles)]
        unassigned = [req['request_id'] for req in self.requests]
        random.shuffle(unassigned)
        
        for req_id in unassigned:
            pid = self.request_pickup[req_id]
            did = self.request_delivery[req_id]
            
            best_vehicle = -1
            best_cost_increase = float('inf')
            best_new_route = None
            
            for v in range(self.num_vehicles):
                route = routes[v]
                old_cost, _ = self.evaluate_route(route)
                new_route, new_cost = self.find_best_insertion(route, pid, did)
                
                if new_route is not None:
                    cost_increase = new_cost - old_cost
                    if cost_increase < best_cost_increase:
                        best_cost_increase = cost_increase
                        best_vehicle = v
                        best_new_route = new_route
            
            if best_vehicle >= 0:
                routes[best_vehicle] = best_new_route
            else:
                # Force insert into least loaded vehicle
                min_len = float('inf')
                best_v = 0
                for v in range(self.num_vehicles):
                    if len(routes[v]) < min_len:
                        min_len = len(routes[v])
                        best_v = v
                routes[best_v].append(pid)
                routes[best_v].append(did)
        
        return routes
    
    def regret_insertion(self):
        """Build initial solution using regret-2 insertion."""
        routes = [[] for _ in range(self.num_vehicles)]
        route_costs = [0.0] * self.num_vehicles
        unassigned = [req['request_id'] for req in self.requests]
        
        while unassigned:
            if self.time_remaining() < 1:
                for req_id in list(unassigned):
                    pid = self.request_pickup[req_id]
                    did = self.request_delivery[req_id]
                    best_v = 0
                    best_cost = float('inf')
                    best_route = None
                    for v in range(self.num_vehicles):
                        new_route, new_cost = self.find_best_insertion(routes[v], pid, did)
                        if new_route is not None and new_cost < best_cost:
                            best_cost = new_cost
                            best_v = v
                            best_route = new_route
                    if best_route is not None:
                        routes[best_v] = best_route
                        route_costs[best_v] = best_cost
                    else:
                        routes[0].append(pid)
                        routes[0].append(did)
                        c, _ = self.evaluate_route(routes[0])
                        route_costs[0] = c
                unassigned = []
                break
            
            best_regret = -float('inf')
            best_req = None
            best_vehicle_for_req = None
            best_route_for_req = None
            best_cost_for_req = None
            
            for req_id in unassigned:
                pid = self.request_pickup[req_id]
                did = self.request_delivery[req_id]
                
                insertion_costs = []
                
                for v in range(self.num_vehicles):
                    route = routes[v]
                    new_route, new_cost = self.find_best_insertion(route, pid, did)
                    
                    if new_route is not None:
                        increase = new_cost - route_costs[v]
                        insertion_costs.append((increase, v, new_route, new_cost))
                
                if len(insertion_costs) == 0:
                    continue
                
                insertion_costs.sort(key=lambda x: x[0])
                
                if len(insertion_costs) >= 2:
                    regret = insertion_costs[1][0] - insertion_costs[0][0]
                else:
                    regret = 1e9
                
                if regret > best_regret:
                    best_regret = regret
                    best_req = req_id
                    best_vehicle_for_req = insertion_costs[0][1]
                    best_route_for_req = insertion_costs[0][2]
                    best_cost_for_req = insertion_costs[0][3]
            
            if best_req is not None:
                v = best_vehicle_for_req
                routes[v] = best_route_for_req
                route_costs[v] = best_cost_for_req
                unassigned.remove(best_req)
            else:
                for req_id in list(unassigned):
                    pid = self.request_pickup[req_id]
                    did = self.request_delivery[req_id]
                    min_len = float('inf')
                    best_v = 0
                    for v in range(self.num_vehicles):
                        if len(routes[v]) < min_len:
                            min_len = len(routes[v])
                            best_v = v
                    routes[best_v].append(pid)
                    routes[best_v].append(did)
                    c, _ = self.evaluate_route(routes[best_v])
                    route_costs[best_v] = c
                    unassigned.remove(req_id)
        
        return routes
    
    def intra_route_relocate(self, routes):
        """Try relocating a request within the same route."""
        improved = True
        while improved:
            improved = False
            if self.time_remaining() < 1:
                return routes
            for v in range(len(routes)):
                route = routes[v]
                if len(route) <= 2:
                    continue
                reqs = self.get_requests_in_route(route)
                current_cost, _ = self.evaluate_route(route)
                
                found = False
                for req_id in reqs:
                    pid = self.request_pickup[req_id]
                    did = self.request_delivery[req_id]
                    
                    reduced_route = self.remove_request_from_route(route, req_id)
                    new_route, new_cost = self.find_best_insertion(reduced_route, pid, did)
                    
                    if new_route is not None and new_cost < current_cost - 1e-6:
                        routes[v] = new_route
                        improved = True
                        found = True
                        break
                
                if found:
                    break
                
                if self.time_remaining() < 1:
                    return routes
        
        return routes
    
    def inter_route_relocate(self, routes):
        """Try moving a request from one route to another."""
        improved = True
        while improved:
            improved = False
            if self.time_remaining() < 1:
                return routes
            
            found = False
            for v1 in range(len(routes)):
                if found:
                    break
                if len(routes[v1]) == 0:
                    continue
                reqs = self.get_requests_in_route(routes[v1])
                current_v1_cost, _ = self.evaluate_route(routes[v1])
                
                for req_id in reqs:
                    if found:
                        break
                    pid = self.request_pickup[req_id]
                    did = self.request_delivery[req_id]
                    
                    reduced_route = self.remove_request_from_route(routes[v1], req_id)
                    reduced_cost, red_feasible = self.evaluate_route(reduced_route)
                    if not red_feasible:
                        reduced_cost = 0.0
                    
                    for v2 in range(len(routes)):
                        if v2 == v1:
                            continue
                        
                        current_v2_cost, _ = self.evaluate_route(routes[v2])
                        new_route, new_cost = self.find_best_insertion(routes[v2], pid, did)
                        
                        if new_route is not None:
                            old_total = current_v1_cost + current_v2_cost
                            new_total = reduced_cost + new_cost
                            
                            if new_total < old_total - 1e-6:
                                routes[v1] = reduced_route
                                routes[v2] = new_route
                                improved = True
                                found = True
                                break
                
                if self.time_remaining() < 1:
                    return routes
        
        return routes
    
    def swap_within_route(self, routes):
        """Swap two nodes within a route (maintaining feasibility)."""
        improved = True
        while improved:
            improved = False
            if self.time_remaining() < 1:
                return routes
            
            found = False
            for v in range(len(routes)):
                if found:
                    break
                route = routes[v]
                n = len(route)
                if n <= 2:
                    continue
                current_cost, _ = self.evaluate_route(route)
                
                for i in range(n):
                    if found:
                        break
                    for j in range(i + 1, n):
                        new_route = list(route)
                        new_route[i], new_route[j] = new_route[j], new_route[i]
                        
                        if not self.check_route_feasibility(new_route):
                            continue
                        
                        cost, feasible = self.evaluate_route(new_route)
                        if feasible and cost < current_cost - 1e-6:
                            routes[v] = new_route
                            improved = True
                            found = True
                            break
                
                if self.time_remaining() < 1:
                    return routes
        
        return routes
    
    def swap_between_routes(self, routes):
        """Swap a request between two routes."""
        improved = True
        while improved:
            improved = False
            if self.time_remaining() < 1:
                return routes
            
            found = False
            for v1 in range(len(routes)):
                if found:
                    break
                if len(routes[v1]) == 0:
                    continue
                reqs1 = self.get_requests_in_route(routes[v1])
                
                for v2 in range(v1 + 1, len(routes)):
                    if found:
                        break
                    if len(routes[v2]) == 0:
                        continue
                    reqs2 = self.get_requests_in_route(routes[v2])
                    
                    c1, _ = self.evaluate_route(routes[v1])
                    c2, _ = self.evaluate_route(routes[v2])
                    current_cost = c1 + c2
                    
                    for r1 in reqs1:
                        if found:
                            break
                        for r2 in reqs2:
                            route1_without = self.remove_request_from_route(routes[v1], r1)
                            route2_without = self.remove_request_from_route(routes[v2], r2)
                            
                            p2 = self.request_pickup[r2]
                            d2 = self.request_delivery[r2]
                            new_route1, cost1 = self.find_best_insertion(route1_without, p2, d2)
                            
                            p1 = self.request_pickup[r1]
                            d1 = self.request_delivery[r1]
                            new_route2, cost2 = self.find_best_insertion(route2_without, p1, d1)
                            
                            if new_route1 is not None and new_route2 is not None:
                                new_cost = cost1 + cost2
                                if new_cost < current_cost - 1e-6:
                                    routes[v1] = new_route1
                                    routes[v2] = new_route2
                                    improved = True
                                    found = True
                                    break
                    
                    if self.time_remaining() < 1:
                        return routes
        
        return routes
    
    def local_search(self, routes):
        """Apply local search operators."""
        improved = True
        while improved and self.time_remaining() > 2:
            improved = False
            old_cost, _ = self.evaluate_solution(routes)
            
            if self.time_remaining() < 2:
                break
            routes = self.intra_route_relocate(routes)
            
            if self.time_remaining() < 2:
                break
            routes = self.inter_route_relocate(routes)
            
            if self.time_remaining() < 2:
                break
            routes = self.swap_within_route(routes)
            
            if self.time_remaining() < 2:
                break
            routes = self.swap_between_routes(routes)
            
            new_cost, feasible = self.evaluate_solution(routes)
            if feasible and new_cost < old_cost - 1e-6:
                improved = True
        
        return routes
    
    def perturbation(self, routes, strength=2):
        """Random perturbation: remove and reinsert some requests."""
        routes = [list(r) for r in routes]
        
        all_reqs = []
        req_to_vehicle = {}
        for v in range(len(routes)):
            for req_id in self.get_requests_in_route(routes[v]):
                all_reqs.append(req_id)
                req_to_vehicle[req_id] = v
        
        if len(all_reqs) == 0:
            return routes
        
        num_remove = min(strength, len(all_reqs))
        removed = random.sample(all_reqs, num_remove)
        
        for req_id in removed:
            v = req_to_vehicle[req_id]
            routes[v] = self.remove_request_from_route(routes[v], req_id)
        
        random.shuffle(removed)
        for req_id in removed:
            pid = self.request_pickup[req_id]
            did = self.request_delivery[req_id]
            
            best_v = -1
            best_cost_increase = float('inf')
            best_new_route = None
            
            for v in range(len(routes)):
                old_cost, _ = self.evaluate_route(routes[v])
                new_route, new_cost = self.find_best_insertion(routes[v], pid, did)
                if new_route is not None:
                    increase = new_cost - old_cost
                    if increase < best_cost_increase:
                        best_cost_increase = increase
                        best_v = v
                        best_new_route = new_route
            
            if best_v >= 0:
                routes[best_v] = best_new_route
            else:
                routes[0].append(pid)
                routes[0].append(did)
        
        return routes
    
    def ruin_and_recreate(self, routes, num_remove=None):
        """Shaw removal + regret insertion."""
        routes = [list(r) for r in routes]
        
        all_reqs = []
        req_to_vehicle = {}
        for v in range(len(routes)):
            for req_id in self.get_requests_in_route(routes[v]):
                all_reqs.append(req_id)
                req_to_vehicle[req_id] = v
        
        if len(all_reqs) == 0:
            return routes
        
        if num_remove is None:
            num_remove = max(1, min(len(all_reqs), int(0.2 * len(all_reqs)) + 1))
        num_remove = min(num_remove, len(all_reqs))
        
        removed = random.sample(all_reqs, num_remove)
        
        for req_id in removed:
            v = req_to_vehicle[req_id]
            routes[v] = self.remove_request_from_route(routes[v], req_id)
        
        # Regret-2 insertion
        unassigned = list(removed)
        while unassigned:
            if self.time_remaining() < 1:
                for req_id in list(unassigned):
                    pid = self.request_pickup[req_id]
                    did = self.request_delivery[req_id]
                    best_v = -1
                    best_cost = float('inf')
                    best_route = None
                    for v in range(len(routes)):
                        new_route, new_cost = self.find_best_insertion(routes[v], pid, did)
                        if new_route is not None and new_cost < best_cost:
                            best_cost = new_cost
                            best_v = v
                            best_route = new_route
                    if best_v >= 0:
                        routes[best_v] = best_route
                    else:
                        routes[0].append(pid)
                        routes[0].append(did)
                break
            
            best_regret = -float('inf')
            best_req = None
            best_vehicle_for_req = None
            best_route_for_req = None
            
            for req_id in unassigned:
                pid = self.request_pickup[req_id]
                did = self.request_delivery[req_id]
                
                insertion_costs = []
                for v in range(len(routes)):
                    old_cost, _ = self.evaluate_route(routes[v])
                    new_route, new_cost = self.find_best_insertion(routes[v], pid, did)
                    if new_route is not None:
                        insertion_costs.append((new_cost - old_cost, v, new_route))
                
                if len(insertion_costs) == 0:
                    continue
                
                insertion_costs.sort(key=lambda x: x[0])
                
                if len(insertion_costs) >= 2:
                    regret = insertion_costs[1][0] - insertion_costs[0][0]
                else:
                    regret = 1e9
                
                if regret > best_regret:
                    best_regret = regret
                    best_req = req_id
                    best_vehicle_for_req = insertion_costs[0][1]
                    best_route_for_req = insertion_costs[0][2]
            
            if best_req is not None:
                routes[best_vehicle_for_req] = best_route_for_req
                unassigned.remove(best_req)
            else:
                for req_id in list(unassigned):
                    pid = self.request_pickup[req_id]
                    did = self.request_delivery[req_id]
                    routes[0].append(pid)
                    routes[0].append(did)
                    unassigned.remove(req_id)
        
        return routes
    
    def update_best(self, routes):
        """Check if routes improve best solution and update accordingly."""
        total_cost, feasible = self.evaluate_solution(routes)
        if feasible and total_cost < self.best_cost:
            self.best_solution = [list(r) for r in routes]
            self.best_cost = total_cost
            if self.logger:
                self.logger.log(self.best_cost)
            return True
        return False
    
    def solve(self):
        """Main solving loop."""
        # Build initial solution with regret insertion
        routes = self.regret_insertion()
        total_cost, feasible = self.evaluate_solution(routes)
        
        if feasible:
            self.best_solution = [list(r) for r in routes]
            self.best_cost = total_cost
            if self.logger:
                self.logger.log(self.best_cost)
        
        # Local search on initial solution
        if self.time_remaining() > 2:
            routes = self.local_search(routes)
            self.update_best(routes)
        
        # Try greedy insertion as alternative
        if self.time_remaining() > 5:
            routes2 = self.greedy_insertion()
            cost2, feas2 = self.evaluate_solution(routes2)
            if feas2:
                if self.time_remaining() > 3:
                    routes2 = self.local_search(routes2)
                cost2, feas2 = self.evaluate_solution(routes2)
                if feas2 and cost2 < self.best_cost:
                    self.best_solution = [list(r) for r in routes2]
                    self.best_cost = cost2
                    routes = routes2
                    if self.logger:
                        self.logger.log(self.best_cost)
        
        # ALNS-like loop
        iteration = 0
        temperature = max(self.best_cost * 0.05, 1.0) if self.best_cost < float('inf') else 100
        cooling_rate = 0.9995
        
        if self.best_solution is not None:
            current_routes = [list(r) for r in self.best_solution]
            current_cost = self.best_cost
        else:
            current_routes = routes
            current_cost, _ = self.evaluate_solution(routes)
        
        while self.time_remaining() > 2:
            iteration += 1
            
            # Choose perturbation strategy
            strategy = random.random()
            if strategy < 0.5:
                strength = random.randint(1, max(1, min(self.num_requests, 4)))
                new_routes = self.perturbation(current_routes, strength)
            else:
                num_remove = random.randint(1, max(1, min(self.num_requests, 5)))
                new_routes = self.ruin_and_recreate(current_routes, num_remove)
            
            if self.time_remaining() < 2:
                break
            
            # Quick local search
            new_routes = self.intra_route_relocate(new_routes)
            if self.time_remaining() > 3:
                new_routes = self.inter_route_relocate(new_routes)
            
            new_cost, new_feasible = self.evaluate_solution(new_routes)
            
            if new_feasible:
                # Simulated annealing acceptance
                accept = False
                if new_cost < current_cost:
                    accept = True
                elif temperature > 0:
                    delta = new_cost - current_cost
                    prob = math.exp(-delta / max(temperature, 1e-10))
                    if random.random() < prob:
                        accept = True
                
                if accept:
                    current_routes = [list(r) for r in new_routes]
                    current_cost = new_cost
                
                if new_cost < self.best_cost:
                    self.best_solution = [list(r) for r in new_routes]
                    self.best_cost = new_cost
                    if self.logger:
                        self.logger.log(self.best_cost)
            
            temperature *= cooling_rate
            
            # Restart from best periodically
            if iteration % 100 == 0 and self.best_solution is not None:
                current_routes = [list(r) for r in self.best_solution]
                current_cost = self.best_cost
                temperature = max(self.best_cost * 0.02, 0.1)
        
        # Final local search on best solution
        if self.time_remaining() > 1 and self.best_solution is not None:
            final_routes = self.local_search([list(r) for r in self.best_solution])
            self.update_best(final_routes)
        
        return self.best_solution, self.best_cost
    
    def format_solution(self):
        """Format solution for output."""
        routes_output = []
        
        if self.best_solution is None:
            for v in range(self.num_vehicles):
                routes_output.append({
                    "vehicle": v,
                    "nodes": [self.depot_origin, self.depot_destination]
                })
            return {
                "objective_value": 0.0,
                "routes": routes_output
            }
        
        used_vehicles = set()
        for v in range(len(self.best_solution)):
            if len(self.best_solution[v]) > 0:
                used_vehicles.add(v)
        
        for v in range(self.num_vehicles):
            if v < len(self.best_solution) and len(self.best_solution[v]) > 0:
                route_nodes = [self.depot_origin] + self.best_solution[v] + [self.depot_destination]
            else:
                route_nodes = [self.depot_origin, self.depot_destination]
            routes_output.append({
                "vehicle": v,
                "nodes": route_nodes
            })
        
        return {
            "objective_value": round(self.best_cost, 10),
            "routes": routes_output
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    solver = PDPSolver(instance, args.time_limit, logger)
    solver.solve()
    
    solution = solver.format_solution()
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == '__main__':
    main()
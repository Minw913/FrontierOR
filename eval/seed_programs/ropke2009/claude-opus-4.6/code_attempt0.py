import argparse
import json
import math
import time
import random
import sys
from solution_logger import SolutionLogger


def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def compute_distance_matrix(nodes):
    n = len(nodes)
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                dist[i][j] = euclidean_distance(nodes[i]['x'], nodes[i]['y'],
                                                  nodes[j]['x'], nodes[j]['y'])
    return dist


def compute_travel_time_matrix(nodes, dist):
    """Travel time from i to j = dist[i][j] + service_time[i]"""
    n = len(nodes)
    tt = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                tt[i][j] = dist[i][j] + nodes[i]['service_time']
    return tt


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
        
        # Distance and travel time matrices
        # Map node ids to indices
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
    
    def greedy_insertion(self):
        """Build initial solution using greedy insertion."""
        routes = [[] for _ in range(self.num_vehicles)]
        unassigned = list(range(self.num_requests))
        random.shuffle(unassigned)
        
        for req_idx in unassigned:
            req = self.requests[req_idx]
            pid = req['pickup_node']
            did = req['delivery_node']
            
            best_vehicle = -1
            best_cost_increase = float('inf')
            best_positions = None
            
            for v in range(self.num_vehicles):
                route = routes[v]
                n = len(route)
                
                # Try all valid insertion positions
                for pi in range(n + 1):
                    for di in range(pi + 1, n + 2):
                        new_route = route[:pi] + [pid] + route[pi:di-1] + [did] + route[di-1:]
                        cost, feasible = self.evaluate_route(new_route)
                        if feasible:
                            old_cost, _ = self.evaluate_route(route)
                            cost_increase = cost - old_cost
                            if cost_increase < best_cost_increase:
                                best_cost_increase = cost_increase
                                best_vehicle = v
                                best_positions = new_route
            
            if best_vehicle >= 0:
                routes[best_vehicle] = best_positions
            else:
                # Force insert into least loaded vehicle
                # Find vehicle with fewest nodes
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
        unassigned = list(range(self.num_requests))
        
        while unassigned:
            if self.time_remaining() < 1:
                # Fall back: insert remaining greedily
                for req_idx in unassigned:
                    req = self.requests[req_idx]
                    pid = req['pickup_node']
                    did = req['delivery_node']
                    best_v = 0
                    best_cost = float('inf')
                    best_route = None
                    for v in range(self.num_vehicles):
                        route = routes[v]
                        new_route = route + [pid, did]
                        cost, feasible = self.evaluate_route(new_route)
                        if feasible and cost < best_cost:
                            best_cost = cost
                            best_v = v
                            best_route = new_route
                    if best_route is not None:
                        routes[best_v] = best_route
                        route_costs[best_v] = best_cost
                    else:
                        routes[0].append(pid)
                        routes[0].append(did)
                break
            
            best_regret = -float('inf')
            best_req = None
            best_vehicle_for_req = None
            best_route_for_req = None
            best_cost_for_req = None
            
            for req_idx in unassigned:
                req = self.requests[req_idx]
                pid = req['pickup_node']
                did = req['delivery_node']
                
                # Find best and second-best insertion for this request
                insertion_costs = []
                
                for v in range(self.num_vehicles):
                    route = routes[v]
                    n = len(route)
                    local_best_cost = float('inf')
                    local_best_route = None
                    
                    for pi in range(n + 1):
                        for di in range(pi + 1, n + 2):
                            new_route = route[:pi] + [pid] + route[pi:di-1] + [did] + route[di-1:]
                            cost, feasible = self.evaluate_route(new_route)
                            if feasible:
                                increase = cost - route_costs[v]
                                if increase < local_best_cost:
                                    local_best_cost = increase
                                    local_best_route = new_route
                    
                    if local_best_route is not None:
                        insertion_costs.append((local_best_cost, v, local_best_route))
                
                if len(insertion_costs) == 0:
                    continue
                
                insertion_costs.sort(key=lambda x: x[0])
                
                if len(insertion_costs) >= 2:
                    regret = insertion_costs[1][0] - insertion_costs[0][0]
                else:
                    regret = 1e9  # Only one feasible insertion, prioritize it
                
                if regret > best_regret:
                    best_regret = regret
                    best_req = req_idx
                    best_vehicle_for_req = insertion_costs[0][1]
                    best_route_for_req = insertion_costs[0][2]
                    best_cost_for_req = insertion_costs[0][0]
            
            if best_req is not None:
                v = best_vehicle_for_req
                routes[v] = best_route_for_req
                route_costs[v] += best_cost_for_req
                unassigned.remove(best_req)
            else:
                # Force insert
                for req_idx in list(unassigned):
                    req = self.requests[req_idx]
                    pid = req['pickup_node']
                    did = req['delivery_node']
                    best_v = 0
                    min_len = float('inf')
                    for v in range(self.num_vehicles):
                        if len(routes[v]) < min_len:
                            min_len = len(routes[v])
                            best_v = v
                    routes[best_v].append(pid)
                    routes[best_v].append(did)
                    cost, _ = self.evaluate_route(routes[best_v])
                    route_costs[best_v] = cost
                    unassigned.remove(req_idx)
        
        return routes
    
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
                    best_cost = cost
                    best_route = list(new_route)
        
        return best_route, best_cost
    
    def remove_request_from_route(self, route, req_idx):
        """Remove a request (pickup and delivery) from a route."""
        req = self.requests[req_idx]
        pid = req['pickup_node']
        did = req['delivery_node']
        new_route = [n for n in route if n != pid and n != did]
        return new_route
    
    def get_requests_in_route(self, route):
        """Get list of request indices served by a route."""
        reqs = []
        for node_id in route:
            if node_id in self.pickup_to_request:
                reqs.append(self.pickup_to_request[node_id])
        return reqs
    
    def intra_route_relocate(self, routes):
        """Try relocating a request within the same route."""
        improved = True
        while improved:
            improved = False
            for v in range(len(routes)):
                route = routes[v]
                if len(route) <= 2:
                    continue
                reqs = self.get_requests_in_route(route)
                current_cost, _ = self.evaluate_route(route)
                
                for req_idx in reqs:
                    req = self.requests[req_idx]
                    pid = req['pickup_node']
                    did = req['delivery_node']
                    
                    reduced_route = self.remove_request_from_route(route, req_idx)
                    new_route, new_cost = self.find_best_insertion(reduced_route, pid, did)
                    
                    if new_route is not None and new_cost < current_cost - 1e-6:
                        routes[v] = new_route
                        current_cost = new_cost
                        improved = True
                        break
                
                if self.time_remaining() < 1:
                    return routes
        
        return routes
    
    def inter_route_relocate(self, routes):
        """Try moving a request from one route to another."""
        improved = True
        while improved:
            improved = False
            for v1 in range(len(routes)):
                if len(routes[v1]) == 0:
                    continue
                reqs = self.get_requests_in_route(routes[v1])
                
                for req_idx in reqs:
                    req = self.requests[req_idx]
                    pid = req['pickup_node']
                    did = req['delivery_node']
                    
                    reduced_route = self.remove_request_from_route(routes[v1], req_idx)
                    reduced_cost, red_feasible = self.evaluate_route(reduced_route)
                    if not red_feasible:
                        reduced_cost = 0.0
                    
                    current_v1_cost, _ = self.evaluate_route(routes[v1])
                    
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
                                break
                    
                    if improved:
                        break
                
                if improved:
                    break
                
                if self.time_remaining() < 1:
                    return routes
        
        return routes
    
    def or_opt_within_route(self, routes):
        """Or-opt: move a segment of 1, 2, or 3 consecutive nodes within a route."""
        improved = True
        while improved:
            improved = False
            for v in range(len(routes)):
                route = routes[v]
                n = len(route)
                if n <= 2:
                    continue
                current_cost, _ = self.evaluate_route(route)
                
                for seg_len in [1, 2, 3]:
                    if improved:
                        break
                    for i in range(n - seg_len + 1):
                        if improved:
                            break
                        segment = route[i:i + seg_len]
                        remaining = route[:i] + route[i + seg_len:]
                        
                        for j in range(len(remaining) + 1):
                            if j == i:
                                continue
                            new_route = remaining[:j] + segment + remaining[j:]
                            cost, feasible = self.evaluate_route(new_route)
                            if feasible and cost < current_cost - 1e-6:
                                # Check precedence
                                if self.check_route_feasibility(new_route):
                                    routes[v] = new_route
                                    current_cost = cost
                                    improved = True
                                    break
                
                if self.time_remaining() < 1:
                    return routes
        
        return routes
    
    def swap_within_route(self, routes):
        """Swap two nodes within a route (maintaining feasibility)."""
        improved = True
        while improved:
            improved = False
            for v in range(len(routes)):
                route = routes[v]
                n = len(route)
                if n <= 2:
                    continue
                current_cost, _ = self.evaluate_route(route)
                
                for i in range(n):
                    if improved:
                        break
                    for j in range(i + 1, n):
                        new_route = list(route)
                        new_route[i], new_route[j] = new_route[j], new_route[i]
                        cost, feasible = self.evaluate_route(new_route)
                        if feasible and cost < current_cost - 1e-6:
                            if self.check_route_feasibility(new_route):
                                routes[v] = new_route
                                improved = True
                                break
                
                if self.time_remaining() < 1:
                    return routes
        
        return routes
    
    def swap_between_routes(self, routes):
        """Swap a request between two routes."""
        improved = True
        while improved:
            improved = False
            for v1 in range(len(routes)):
                if improved:
                    break
                if len(routes[v1]) == 0:
                    continue
                reqs1 = self.get_requests_in_route(routes[v1])
                
                for v2 in range(v1 + 1, len(routes)):
                    if improved:
                        break
                    if len(routes[v2]) == 0:
                        continue
                    reqs2 = self.get_requests_in_route(routes[v2])
                    
                    current_cost = 0
                    c1, _ = self.evaluate_route(routes[v1])
                    c2, _ = self.evaluate_route(routes[v2])
                    current_cost = c1 + c2
                    
                    for r1 in reqs1:
                        if improved:
                            break
                        for r2 in reqs2:
                            req1 = self.requests[r1]
                            req2 = self.requests[r2]
                            
                            # Remove r1 from v1, r2 from v2
                            route1_without = self.remove_request_from_route(routes[v1], r1)
                            route2_without = self.remove_request_from_route(routes[v2], r2)
                            
                            # Insert r2 into v1, r1 into v2
                            new_route1, cost1 = self.find_best_insertion(
                                route1_without, req2['pickup_node'], req2['delivery_node'])
                            new_route2, cost2 = self.find_best_insertion(
                                route2_without, req1['pickup_node'], req1['delivery_node'])
                            
                            if new_route1 is not None and new_route2 is not None:
                                new_cost = cost1 + cost2
                                if new_cost < current_cost - 1e-6:
                                    routes[v1] = new_route1
                                    routes[v2] = new_route2
                                    improved = True
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
            if self.time_remaining() < 2:
                break
            
            new_cost, feasible = self.evaluate_solution(routes)
            if feasible and new_cost < old_cost - 1e-6:
                improved = True
        
        return routes
    
    def perturbation(self, routes, strength=2):
        """Random perturbation: remove and reinsert some requests."""
        routes = [list(r) for r in routes]
        
        # Collect all requests and their current vehicles
        all_reqs = []
        req_to_vehicle = {}
        for v in range(len(routes)):
            for req_idx in self.get_requests_in_route(routes[v]):
                all_reqs.append(req_idx)
                req_to_vehicle[req_idx] = v
        
        if len(all_reqs) == 0:
            return routes
        
        # Remove random requests
        num_remove = min(strength, len(all_reqs))
        removed = random.sample(all_reqs, num_remove)
        
        for req_idx in removed:
            v = req_to_vehicle[req_idx]
            routes[v] = self.remove_request_from_route(routes[v], req_idx)
        
        # Reinsert in random order
        random.shuffle(removed)
        for req_idx in removed:
            req = self.requests[req_idx]
            pid = req['pickup_node']
            did = req['delivery_node']
            
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
                # Force insert into first available
                routes[0].append(pid)
                routes[0].append(did)
        
        return routes
    
    def ruin_and_recreate(self, routes, num_remove=None):
        """Shaw removal + regret insertion."""
        routes = [list(r) for r in routes]
        
        all_reqs = []
        req_to_vehicle = {}
        for v in range(len(routes)):
            for req_idx in self.get_requests_in_route(routes[v]):
                all_reqs.append(req_idx)
                req_to_vehicle[req_idx] = v
        
        if len(all_reqs) == 0:
            return routes
        
        if num_remove is None:
            num_remove = max(1, min(len(all_reqs), int(0.2 * len(all_reqs)) + 1))
        num_remove = min(num_remove, len(all_reqs))
        
        # Random removal
        removed = random.sample(all_reqs, num_remove)
        
        for req_idx in removed:
            v = req_to_vehicle[req_idx]
            routes[v] = self.remove_request_from_route(routes[v], req_idx)
        
        # Regret-2 insertion
        unassigned = list(removed)
        while unassigned:
            if self.time_remaining() < 1:
                # Fast insert remaining
                for req_idx in unassigned:
                    req = self.requests[req_idx]
                    pid = req['pickup_node']
                    did = req['delivery_node']
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
            
            for req_idx in unassigned:
                req = self.requests[req_idx]
                pid = req['pickup_node']
                did = req['delivery_node']
                
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
                    best_req = req_idx
                    best_vehicle_for_req = insertion_costs[0][1]
                    best_route_for_req = insertion_costs[0][2]
            
            if best_req is not None:
                routes[best_vehicle_for_req] = best_route_for_req
                unassigned.remove(best_req)
            else:
                for req_idx in list(unassigned):
                    req = self.requests[req_idx]
                    routes[0].append(req['pickup_node'])
                    routes[0].append(req['delivery_node'])
                    unassigned.remove(req_idx)
        
        return routes
    
    def solve(self):
        """Main solving loop."""
        # Build initial solution
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
            total_cost, feasible = self.evaluate_solution(routes)
            if feasible and total_cost < self.best_cost:
                self.best_solution = [list(r) for r in routes]
                self.best_cost = total_cost
                if self.logger:
                    self.logger.log(self.best_cost)
        
        # Try greedy insertion as alternative
        if self.time_remaining() > 5:
            routes2 = self.greedy_insertion()
            cost2, feas2 = self.evaluate_solution(routes2)
            if feas2:
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
        temperature = self.best_cost * 0.05 if self.best_cost > 0 else 100
        cooling_rate = 0.9995
        
        current_routes = [list(r) for r in self.best_solution] if self.best_solution else routes
        current_cost = self.best_cost
        
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
                if new_cost < current_cost:
                    current_routes = new_routes
                    current_cost = new_cost
                elif temperature > 0:
                    delta = new_cost - current_cost
                    prob = math.exp(-delta / max(temperature, 1e-10))
                    if random.random() < prob:
                        current_routes = new_routes
                        current_cost = new_cost
                
                if new_cost < self.best_cost:
                    self.best_solution = [list(r) for r in new_routes]
                    self.best_cost = new_cost
                    if self.logger:
                        self.logger.log(self.best_cost)
            
            temperature *= cooling_rate
            
            # Restart from best periodically
            if iteration % 100 == 0:
                current_routes = [list(r) for r in self.best_solution]
                current_cost = self.best_cost
                temperature = self.best_cost * 0.02
        
        # Final local search on best solution
        if self.time_remaining() > 1 and self.best_solution:
            final_routes = self.local_search([list(r) for r in self.best_solution])
            final_cost, final_feasible = self.evaluate_solution(final_routes)
            if final_feasible and final_cost < self.best_cost:
                self.best_solution = final_routes
                self.best_cost = final_cost
                if self.logger:
                    self.logger.log(self.best_cost)
        
        return self.best_solution, self.best_cost
    
    def format_solution(self):
        """Format solution for output."""
        routes_output = []
        
        if self.best_solution is None:
            # Return empty routes
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
                route_nodes = [self.depot_origin] + self.best_solution[v] + [self.depot_destination]
                routes_output.append({
                    "vehicle": v,
                    "nodes": route_nodes
                })
        
        # Add empty routes for unused vehicles
        vehicle_id = 0
        for v in range(self.num_vehicles):
            if v not in used_vehicles:
                routes_output.append({
                    "vehicle": v,
                    "nodes": [self.depot_origin, self.depot_destination]
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
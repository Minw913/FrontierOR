import math
import json
import time
import argparse
from solution_logger import SolutionLogger

class PDPTWSolver:
    def __init__(self, instance, time_limit, logger=None):
        self.instance = instance
        self.time_limit = time_limit
        self.logger = logger
        self.start_time = time.time()
        
        self.num_vehicles = instance['num_vehicles']
        self.vehicle_capacity = instance['vehicle_capacity']
        self.nodes_data = instance['nodes']
        self.requests_data = instance['requests']
        
        self.coords = {}
        self.demands = {}
        self.service_times = {}
        self.time_windows = {}
        self.node_ids = []
        self.depot_origin_id = None
        self.depot_destination_id = None
        
        for node in self.nodes_data:
            node_id = node['id']
            self.node_ids.append(node_id)
            self.coords[node_id] = (node['x'], node['y'])
            self.demands[node_id] = node['demand']
            self.service_times[node_id] = node['service_time']
            self.time_windows[node_id] = node['time_window']
            if node['type'] == 'depot_origin':
                self.depot_origin_id = node_id
            elif node['type'] == 'depot_destination':
                self.depot_destination_id = node_id
                
        if self.depot_origin_id is None or self.depot_destination_id is None:
            raise ValueError("Depot origin or destination not found.")
        
        self.requests = []
        for req in self.requests_data:
            self.requests.append((req['pickup_node'], req['delivery_node'], req['load']))
        
        self.precompute_matrices()
        
    def precompute_matrices(self):
        self.distance_matrix = {}
        self.travel_time_matrix = {}
        for i in self.node_ids:
            for j in self.node_ids:
                x1, y1 = self.coords[i]
                x2, y2 = self.coords[j]
                d = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                self.distance_matrix[(i, j)] = d
                self.travel_time_matrix[(i, j)] = d + self.service_times[i]
                
    def check_route_feasibility(self, route):
        if len(route) == 0:
            return True, 0, 0
        
        current_time = self.time_windows[route[0]][0]
        load = 0
        total_cost = 0
        n = len(route)
        
        for i in range(n):
            node = route[i]
            if current_time < self.time_windows[node][0]:
                current_time = self.time_windows[node][0]
            elif current_time > self.time_windows[node][1]:
                return False, None, None
                
            load += self.demands[node]
            if load < 0 or load > self.vehicle_capacity:
                return False, None, None
                
            if i < n-1:
                next_node = route[i+1]
                total_cost += self.distance_matrix[(node, next_node)]
                current_time += self.travel_time_matrix[(node, next_node)]
                
        return True, total_cost, current_time

    def initial_solution(self):
        routes = []
        route_costs = []
        for _ in range(self.num_vehicles):
            route = [self.depot_origin_id, self.depot_destination_id]
            routes.append(route)
            feasible, cost, _ = self.check_route_feasibility(route)
            if not feasible:
                cost = 0
            route_costs.append(cost)
            
        requests_sorted = sorted(self.requests, key=lambda r: self.time_windows[r[0]][0])
        
        for r in requests_sorted:
            p, d, load = r
            best_delta = float('inf')
            best_route_idx = None
            best_new_route = None
            best_new_cost = None
            
            if time.time() - self.start_time > self.time_limit:
                break
                
            for k in range(len(routes)):
                current_route = routes[k]
                current_cost = route_costs[k]
                L = len(current_route)
                for i in range(L - 1):
                    for j in range(i+1, L):
                        new_route = current_route[:i+1] + [p] + current_route[i+1:j] + [d] + current_route[j:]
                        feasible, new_cost, _ = self.check_route_feasibility(new_route)
                        if feasible:
                            delta = new_cost - current_cost
                            if delta < best_delta:
                                best_delta = delta
                                best_route_idx = k
                                best_new_route = new_route
                                best_new_cost = new_cost
                                
            if best_route_idx is None:
                for k in range(len(routes)):
                    if len(routes[k]) == 2:
                        new_route = [self.depot_origin_id, p, d, self.depot_destination_id]
                        feasible, new_cost, _ = self.check_route_feasibility(new_route)
                        if feasible:
                            best_route_idx = k
                            best_new_route = new_route
                            best_new_cost = new_cost
                            best_delta = new_cost - route_costs[k]
                            break
                if best_route_idx is None:
                    for k in range(len(routes)):
                        if len(routes[k]) == 2:
                            new_route = [self.depot_origin_id, p, d, self.depot_destination_id]
                            _, new_cost, _ = self.check_route_feasibility(new_route)
                            best_route_idx = k
                            best_new_route = new_route
                            best_new_cost = new_cost
                            break
                    if best_route_idx is None:
                        best_route_idx = 0
                        new_route = [self.depot_origin_id, p, d, self.depot_destination_id]
                        _, new_cost, _ = self.check_route_feasibility(new_route)
                        best_new_route = new_route
                        best_new_cost = new_cost
                        
            routes[best_route_idx] = best_new_route
            route_costs[best_route_idx] = best_new_cost
            
        return routes, route_costs
        
    def local_search(self, routes, route_costs):
        improved = True
        while improved:
            if time.time() - self.start_time > self.time_limit:
                break
                
            improved = False
            best_delta = 0
            best_move = None
            
            for r in self.requests:
                p, d, load = r
                from_idx = None
                for k in range(len(routes)):
                    if p in routes[k] and d in routes[k]:
                        from_idx = k
                        break
                if from_idx is None:
                    continue
                    
                current_route = routes[from_idx]
                current_cost = route_costs[from_idx]
                
                try:
                    idx_p = current_route.index(p)
                    idx_d = current_route.index(d)
                except:
                    continue
                    
                if idx_p > idx_d:
                    continue
                    
                new_route1 = current_route[:]
                del new_route1[idx_d]
                del new_route1[idx_p]
                
                feasible1, cost1, _ = self.check_route_feasibility(new_route1)
                if not feasible1:
                    continue
                    
                for to_idx in range(len(routes)):
                    if to_idx == from_idx:
                        candidate_route = new_route1
                        candidate_route_cost = cost1
                    else:
                        candidate_route = routes[to_idx]
                        candidate_route_cost = route_costs[to_idx]
                        
                    best_insertion_route = None
                    best_insertion_cost = float('inf')
                    L = len(candidate_route)
                    for i in range(L-1):
                        for j in range(i+1, L):
                            new_route2 = candidate_route[:i+1] + [p] + candidate_route[i+1:j] + [d] + candidate_route[j:]
                            feasible2, cost2, _ = self.check_route_feasibility(new_route2)
                            if feasible2 and cost2 < best_insertion_cost:
                                best_insertion_route = new_route2
                                best_insertion_cost = cost2
                                
                    if best_insertion_route is None:
                        continue
                        
                    if to_idx == from_idx:
                        delta = best_insertion_cost - current_cost
                    else:
                        old_cost_total = current_cost + candidate_route_cost
                        new_cost_total = cost1 + best_insertion_cost
                        delta = new_cost_total - old_cost_total
                        
                    if delta < best_delta - 1e-5:
                        best_delta = delta
                        best_move = (r, from_idx, to_idx, new_route1, best_insertion_route, cost1, best_insertion_cost)
                        
            if best_move is not None and best_delta < -1e-5:
                r, from_idx, to_idx, new_route1, new_route2, cost1, cost2 = best_move
                if from_idx == to_idx:
                    routes[from_idx] = new_route2
                    route_costs[from_idx] = cost2
                else:
                    routes[from_idx] = new_route1
                    route_costs[from_idx] = cost1
                    routes[to_idx] = new_route2
                    route_costs[to_idx] = cost2
                improved = True
                total_cost = sum(route_costs)
                if self.logger:
                    self.logger.log(total_cost)
                    
        return routes, route_costs
        
    def solve(self):
        routes, route_costs = self.initial_solution()
        total_cost = sum(route_costs)
        if self.logger:
            self.logger.log(total_cost)
            
        routes, route_costs = self.local_search(routes, route_costs)
        total_cost = sum(route_costs)
        return routes, total_cost

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
        
    solver = PDPTWSolver(instance, args.time_limit, logger)
    routes, total_cost = solver.solve()
    
    solution = {
        "objective_value": total_cost,
        "routes": []
    }
    for k, route in enumerate(routes):
        solution['routes'].append({
            "vehicle": k,
            "nodes": route
        })
        
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == "__main__":
    main()
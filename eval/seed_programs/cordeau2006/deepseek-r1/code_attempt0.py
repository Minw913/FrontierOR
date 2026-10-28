import argparse
import json
import time
import math
from solution_logger import SolutionLogger

class DARPSolver:
    def __init__(self, instance_data):
        self.num_vehicles = instance_data['num_vehicles']
        self.num_users = instance_data['num_users']
        self.vehicle_capacity = instance_data['vehicle_capacity']
        self.maximum_ride_time = instance_data['maximum_ride_time']
        self.maximum_route_duration = instance_data['maximum_route_duration']
        self.nodes = instance_data['nodes']
        self.node_dict = {node['node_id']: node for node in self.nodes}
        self.dist_matrix = self.compute_distance_matrix()
        self.requests = self.build_requests()
        self.dest_depot_id = 2 * self.num_users + 1
        self.logger = None
        self.start_time = 0

    def compute_distance_matrix(self):
        n = len(self.nodes)
        dist = [[0.0] * n for _ in range(n)]
        for i in range(n):
            node_i = self.node_dict[i]
            for j in range(n):
                node_j = self.node_dict[j]
                dx = node_i['x'] - node_j['x']
                dy = node_i['y'] - node_j['y']
                dist[i][j] = math.sqrt(dx*dx + dy*dy)
        return dist

    def build_requests(self):
        requests = []
        for node in self.nodes:
            if node['node_type'] == 'pickup':
                req = {
                    'user_id': node['user_id'],
                    'pickup': node['node_id'],
                    'dropoff': node['paired_node'],
                    'pickup_earliest': node['earliest_time'],
                    'pickup_latest': node['latest_time'],
                    'dropoff_earliest': self.node_dict[node['paired_node']]['earliest_time'],
                    'dropoff_latest': self.node_dict[node['paired_node']]['latest_time'],
                    'service_duration_pickup': node['service_duration'],
                    'service_duration_dropoff': self.node_dict[node['paired_node']]['service_duration'],
                    'pickup_load': node['load'],
                    'dropoff_load': self.node_dict[node['paired_node']]['load'],
                    'direct_time': self.dist_matrix[node['node_id']][node['paired_node']]
                }
                requests.append(req)
        return requests

    def compute_route_cost(self, route):
        cost = 0.0
        for i in range(len(route)-1):
            cost += self.dist_matrix[route[i]][route[i+1]]
        return cost

    def simulate_route(self, route, vehicle_index=None):
        times = [0.0] * len(route)
        loads = [0] * len(route)
        
        node0 = self.node_dict[route[0]]
        times[0] = max(0.0, node0['earliest_time'])
        if times[0] > node0['latest_time']:
            return None, None, False, float('inf')
        loads[0] = 0 + node0['load']
        
        for i in range(1, len(route)):
            prev_node_id = route[i-1]
            curr_node_id = route[i]
            prev_node = self.node_dict[prev_node_id]
            curr_node = self.node_dict[curr_node_id]
            
            travel_time = self.dist_matrix[prev_node_id][curr_node_id]
            departure_prev = times[i-1] + prev_node['service_duration']
            arrival_curr = departure_prev + travel_time
            start_curr = max(arrival_curr, curr_node['earliest_time'])
            if start_curr > curr_node['latest_time']:
                return None, None, False, float('inf')
            times[i] = start_curr
            new_load = loads[i-1] + curr_node['load']
            if new_load < 0 or new_load > self.vehicle_capacity:
                return None, None, False, float('inf')
            loads[i] = new_load
        
        total_cost = self.compute_route_cost(route)
        return times, loads, True, total_cost

    def initial_solution(self):
        dest_depot_id = self.dest_depot_id
        vehicles = []
        for i in range(self.num_vehicles):
            node0 = self.node_dict[0]
            node_dest = self.node_dict[dest_depot_id]
            time0 = node0['earliest_time']
            time_dest = time0 + self.dist_matrix[0][dest_depot_id]
            if time_dest > node_dest['latest_time']:
                time0 = max(node0['earliest_time'], node_dest['latest_time'] - self.dist_matrix[0][dest_depot_id])
                time_dest = time0 + self.dist_matrix[0][dest_depot_id]
                if time0 > node0['latest_time']:
                    time0 = node0['latest_time']
                    time_dest = time0 + self.dist_matrix[0][dest_depot_id]
            route = [0, dest_depot_id]
            times = [time0, time_dest]
            loads = [0, 0]
            vehicles.append({
                'route': route,
                'times': times,
                'loads': loads
            })
        
        sorted_requests = sorted(self.requests, key=lambda r: r['pickup_earliest'])
        unassigned = []
        
        for req in sorted_requests:
            user_id = req['user_id']
            pickup = req['pickup']
            dropoff = req['dropoff']
            best_cost_increase = float('inf')
            best_vehicle = None
            best_route = None
            best_times = None
            best_loads = None
            found = False
            
            for v_idx, vehicle in enumerate(vehicles):
                route = vehicle['route']
                for k in range(len(route)-1):
                    for l in range(k+1, len(route)):
                        new_route = route[:k+1] + [pickup] + route[k+1:l] + [dropoff] + route[l:]
                        times_new, loads_new, feasible, new_cost = self.simulate_route(new_route)
                        if not feasible:
                            continue
                        
                        pos_pickup = k+1
                        pos_dropoff = k+1 + 1 + (l - (k+1))
                        time_pickup = times_new[pos_pickup]
                        time_dropoff = times_new[pos_dropoff]
                        ride_time = time_dropoff - (time_pickup + req['service_duration_pickup'])
                        if ride_time < req['direct_time'] or ride_time > self.maximum_ride_time:
                            continue
                        route_duration = times_new[-1] - times_new[0]
                        if route_duration > self.maximum_route_duration:
                            continue
                            
                        old_cost = self.compute_route_cost(route)
                        cost_increase = new_cost - old_cost
                        if cost_increase < best_cost_increase:
                            best_cost_increase = cost_increase
                            best_vehicle = v_idx
                            best_route = new_route
                            best_times = times_new
                            best_loads = loads_new
                            found = True
            
            if found:
                vehicles[best_vehicle]['route'] = best_route
                vehicles[best_vehicle]['times'] = best_times
                vehicles[best_vehicle]['loads'] = best_loads
                if self.logger:
                    total_cost = self.compute_total_cost(vehicles)
                    self.logger.log(total_cost)
            else:
                for v_idx, vehicle in enumerate(vehicles):
                    if len(vehicle['route']) == 2:
                        new_route = [0, pickup, dropoff, dest_depot_id]
                        times_new, loads_new, feasible, new_cost = self.simulate_route(new_route)
                        if not feasible:
                            continue
                        pos_pickup = 1
                        pos_dropoff = 2
                        time_pickup = times_new[1]
                        time_dropoff = times_new[2]
                        ride_time = time_dropoff - (time_pickup + req['service_duration_pickup'])
                        if ride_time < req['direct_time'] or ride_time > self.maximum_ride_time:
                            continue
                        route_duration = times_new[-1] - times_new[0]
                        if route_duration > self.maximum_route_duration:
                            continue
                        vehicles[v_idx]['route'] = new_route
                        vehicles[v_idx]['times'] = times_new
                        vehicles[v_idx]['loads'] = loads_new
                        found = True
                        if self.logger:
                            total_cost = self.compute_total_cost(vehicles)
                            self.logger.log(total_cost)
                        break
                if not found:
                    unassigned.append(req)
        
        return vehicles, unassigned

    def compute_total_cost(self, vehicles):
        total_cost = 0.0
        for vehicle in vehicles:
            total_cost += self.compute_route_cost(vehicle['route'])
        return total_cost

    def do_local_search(self, vehicles, time_limit):
        start_time = time.time()
        total_cost = self.compute_total_cost(vehicles)
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            for req in self.requests:
                if time.time() - start_time >= time_limit:
                    break
                user_id = req['user_id']
                pickup = req['pickup']
                dropoff = req['dropoff']
                source_vehicle_idx = None
                source_route = None
                for v_idx, vehicle in enumerate(vehicles):
                    if pickup in vehicle['route'] and dropoff in vehicle['route']:
                        source_vehicle_idx = v_idx
                        source_route = vehicle['route']
                        break
                if source_vehicle_idx is None:
                    continue
                
                pos_pickup = source_route.index(pickup)
                pos_dropoff = source_route.index(dropoff)
                if pos_pickup > pos_dropoff:
                    continue
                new_route_source = source_route[:pos_pickup] + source_route[pos_pickup+1:pos_dropoff] + source_route[pos_dropoff+1:]
                times_source, loads_source, feasible_source, cost_source = self.simulate_route(new_route_source)
                if not feasible_source:
                    continue
                
                best_net_change = 0
                best_target_vehicle = None
                best_route_target = None
                best_times_target = None
                best_loads_target = None
                found = False
                
                for target_vehicle_idx in range(len(vehicles)):
                    if time.time() - start_time >= time_limit:
                        break
                    if target_vehicle_idx == source_vehicle_idx:
                        current_route = new_route_source
                    else:
                        current_route = vehicles[target_vehicle_idx]['route']
                    
                    for k in range(len(current_route)-1):
                        for l in range(k+1, len(current_route)):
                            new_route_target = current_route[:k+1] + [pickup] + current_route[k+1:l] + [dropoff] + current_route[l:]
                            times_target, loads_target, feasible_target, cost_target = self.simulate_route(new_route_target)
                            if not feasible_target:
                                continue
                            
                            pos_p = k+1
                            pos_d = k+1 + 1 + (l - (k+1))
                            time_p = times_target[pos_p]
                            time_d = times_target[pos_d]
                            ride_time = time_d - (time_p + req['service_duration_pickup'])
                            if ride_time < req['direct_time'] or ride_time > self.maximum_ride_time:
                                continue
                            route_duration = times_target[-1] - times_target[0]
                            if route_duration > self.maximum_route_duration:
                                continue
                            
                            if target_vehicle_idx == source_vehicle_idx:
                                old_cost_source = self.compute_route_cost(source_route)
                                net_change = cost_target - old_cost_source
                            else:
                                old_cost_source = self.compute_route_cost(source_route)
                                old_cost_target = self.compute_route_cost(vehicles[target_vehicle_idx]['route'])
                                net_change = (cost_source + cost_target) - (old_cost_source + old_cost_target)
                            
                            if net_change < best_net_change:
                                best_net_change = net_change
                                best_target_vehicle = target_vehicle_idx
                                best_route_target = new_route_target
                                best_times_target = times_target
                                best_loads_target = loads_target
                                found = True
                
                if found and best_net_change < 0:
                    if best_target_vehicle == source_vehicle_idx:
                        vehicles[source_vehicle_idx]['route'] = best_route_target
                        vehicles[source_vehicle_idx]['times'] = best_times_target
                        vehicles[source_vehicle_idx]['loads'] = best_loads_target
                    else:
                        vehicles[source_vehicle_idx]['route'] = new_route_source
                        vehicles[source_vehicle_idx]['times'] = times_source
                        vehicles[source_vehicle_idx]['loads'] = loads_source
                        vehicles[best_target_vehicle]['route'] = best_route_target
                        vehicles[best_target_vehicle]['times'] = best_times_target
                        vehicles[best_target_vehicle]['loads'] = best_loads_target
                    
                    total_cost += best_net_change
                    improved = True
                    if self.logger:
                        self.logger.log(total_cost)
        return vehicles

    def format_solution(self, vehicles):
        objective_value = self.compute_total_cost(vehicles)
        routes = {}
        service_times = {}
        ride_times = {}
        
        for idx, vehicle in enumerate(vehicles):
            route = vehicle['route']
            times = vehicle['times']
            routes[f'vehicle_{idx}'] = route
            service_times[f'depot_start_{idx}'] = times[0]
            service_times[f'depot_end_{idx}'] = times[-1]
            for i in range(1, len(route)-1):
                node_id = route[i]
                service_times[node_id] = times[i]
        
        for req in self.requests:
            user_id = req['user_id']
            pickup = req['pickup']
            dropoff = req['dropoff']
            for vehicle in vehicles:
                route = vehicle['route']
                if pickup in route and dropoff in route:
                    times = vehicle['times']
                    pos_p = route.index(pickup)
                    pos_d = route.index(dropoff)
                    if pos_p < pos_d:
                        pickup_time = times[pos_p]
                        dropoff_time = times[pos_d]
                        ride_time = dropoff_time - (pickup_time + req['service_duration_pickup'])
                        ride_times[user_id] = ride_time
                    break
        
        return {
            "objective_value": objective_value,
            "routes": routes,
            "service_times": service_times,
            "ride_times": ride_times
        }

    def solve(self, time_limit, log_path=None):
        self.start_time = time.time()
        if log_path:
            self.logger = SolutionLogger(log_path, sense="minimize")
        else:
            self.logger = None
        
        vehicles, unassigned = self.initial_solution()
        time_used = time.time() - self.start_time
        time_left = max(0, time_limit - time_used)
        
        if time_left > 0:
            vehicles = self.do_local_search(vehicles, time_left)
        
        solution = self.format_solution(vehicles)
        return solution

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)
    
    solver = DARPSolver(instance_data)
    solution = solver.solve(args.time_limit, args.log_path)
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == '__main__':
    main()
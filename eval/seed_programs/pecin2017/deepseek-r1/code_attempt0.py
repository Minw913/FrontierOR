import json
import math
import time
import argparse
from solution_logger import SolutionLogger

def compute_distance_matrix(nodes):
    n = len(nodes)
    dist_matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            dx = nodes[i]['x'] - nodes[j]['x']
            dy = nodes[i]['y'] - nodes[j]['y']
            dist_matrix[i][j] = math.sqrt(dx*dx + dy*dy)
    return dist_matrix

def compute_feasible_arc(nodes, vehicle_capacity, dist_matrix):
    n = len(nodes)
    feasible_arc = [[False] * n for _ in range(n)]
    depot_source_idx = None
    depot_sink_idx = None
    for i in range(n):
        if nodes[i]['type'] == 'depot_source':
            depot_source_idx = i
        elif nodes[i]['type'] == 'depot_sink':
            depot_sink_idx = i
    if depot_source_idx is None or depot_sink_idx is None:
        raise ValueError("Depot source or sink not found")
    
    for i in range(n):
        if nodes[i]['type'] == 'depot_sink':
            continue
        for j in range(n):
            if i == j:
                continue
            if nodes[j]['type'] == 'depot_source':
                continue
            if i == depot_source_idx and j == depot_sink_idx:
                continue
            condition1 = True
            if nodes[j]['type'] == 'depot_sink':
                pass
            elif i == depot_source_idx:
                if nodes[j]['demand'] > vehicle_capacity:
                    condition1 = False
            else:
                if nodes[i]['demand'] + nodes[j]['demand'] > vehicle_capacity:
                    condition1 = False
            if not condition1:
                continue
            condition2 = True
            if i == depot_source_idx:
                travel_time = dist_matrix[i][j]
                arrival_j = travel_time
                if arrival_j > nodes[j]['time_window_close']:
                    condition2 = False
            else:
                start_i_earliest = nodes[i]['time_window_open']
                travel_time = dist_matrix[i][j]
                arrival_j = start_i_earliest + nodes[i]['service_time'] + travel_time
                if arrival_j > nodes[j]['time_window_close']:
                    condition2 = False
            feasible_arc[i][j] = condition1 and condition2
    return feasible_arc

def is_route_feasible(route, nodes, dist_matrix, vehicle_capacity, horizon):
    total_demand = 0
    for node_idx in route:
        if nodes[node_idx]['type'] == 'customer':
            total_demand += nodes[node_idx]['demand']
    if total_demand > vehicle_capacity:
        return False
    
    prev_leave = 0.0
    for i in range(1, len(route)):
        from_node = route[i-1]
        to_node = route[i]
        travel_time = dist_matrix[from_node][to_node]
        arrival = prev_leave + travel_time
        if i == len(route) - 1:
            if arrival > horizon:
                return False
        else:
            to_node_data = nodes[to_node]
            start_time = max(arrival, to_node_data['time_window_open'])
            if start_time > to_node_data['time_window_close']:
                return False
            prev_leave = start_time + to_node_data['service_time']
    return True

def compute_route_cost(route, dist_matrix):
    cost = 0.0
    for i in range(1, len(route)):
        cost += dist_matrix[route[i-1]][route[i]]
    return cost

def insertion_heuristic(nodes, dist_matrix, feasible_arc, vehicle_capacity, horizon, depot_source_idx, depot_sink_idx, customer_indices, time_limit, start_time, logger=None):
    routes = []
    total_cost = 0.0
    unserved = set(customer_indices)
    
    while unserved and time.time() - start_time < time_limit:
        best_increase = float('inf')
        best_customer = None
        best_route_idx = None
        best_pos = None
        
        for customer in unserved:
            new_route = [depot_source_idx, customer, depot_sink_idx]
            if is_route_feasible(new_route, nodes, dist_matrix, vehicle_capacity, horizon):
                cost = dist_matrix[depot_source_idx][customer] + dist_matrix[customer][depot_sink_idx]
                if cost < best_increase:
                    best_increase = cost
                    best_customer = customer
                    best_route_idx = None
                    best_pos = None
        
        for route_idx, route in enumerate(routes):
            for j in range(1, len(route)):
                if not feasible_arc[route[j-1]][customer] or not feasible_arc[customer][route[j]]:
                    continue
                new_route = route[:j] + [customer] + route[j:]
                if not is_route_feasible(new_route, nodes, dist_matrix, vehicle_capacity, horizon):
                    continue
                old_cost = compute_route_cost(route, dist_matrix)
                new_cost = compute_route_cost(new_route, dist_matrix)
                increase = new_cost - old_cost
                if increase < best_increase:
                    best_increase = increase
                    best_customer = customer
                    best_route_idx = route_idx
                    best_pos = j
        
        if best_increase == float('inf'):
            break
        
        if best_route_idx is None:
            new_route = [depot_source_idx, best_customer, depot_sink_idx]
            routes.append(new_route)
            total_cost += best_increase
            unserved.remove(best_customer)
            if logger:
                logger.log(total_cost)
        else:
            route = routes[best_route_idx]
            new_route = route[:best_pos] + [best_customer] + route[best_pos:]
            routes[best_route_idx] = new_route
            total_cost += best_increase
            unserved.remove(best_customer)
            if logger:
                logger.log(total_cost)
    
    return routes, total_cost, unserved

def do_relocate_pass(routes, total_cost, nodes, dist_matrix, feasible_arc, vehicle_capacity, horizon, depot_source_idx, depot_sink_idx, time_limit, start_time, logger=None):
    n_routes = len(routes)
    improved = False
    for customer in [idx for idx in range(len(nodes)) if nodes[idx]['type'] == 'customer']:
        if time.time() - start_time >= time_limit:
            break
        for from_route_idx in range(n_routes):
            if customer not in routes[from_route_idx]:
                continue
            from_route = routes[from_route_idx]
            pos = from_route.index(customer)
            new_from_route = from_route[:pos] + from_route[pos+1:]
            if not is_route_feasible(new_from_route, nodes, dist_matrix, vehicle_capacity, horizon):
                continue
            old_from_cost = compute_route_cost(from_route, dist_matrix)
            new_from_cost = compute_route_cost(new_from_route, dist_matrix)
            savings_from = old_from_cost - new_from_cost
            
            best_increase = float('inf')
            best_insertion = None
            
            new_route_candidate = [depot_source_idx, customer, depot_sink_idx]
            if is_route_feasible(new_route_candidate, nodes, dist_matrix, vehicle_capacity, horizon):
                cost_new_route = dist_matrix[depot_source_idx][customer] + dist_matrix[customer][depot_sink_idx]
                total_change = cost_new_route - savings_from
                if total_change < best_increase:
                    best_increase = total_change
                    best_insertion = ('new', None, None)
            
            for to_route_idx in range(n_routes):
                to_route = routes[to_route_idx]
                for j in range(1, len(to_route)):
                    if j-1 < len(to_route) and not feasible_arc[to_route[j-1]][customer]:
                        continue
                    if j < len(to_route) and not feasible_arc[customer][to_route[j]]:
                        continue
                    new_to_route = to_route[:j] + [customer] + to_route[j:]
                    if not is_route_feasible(new_to_route, nodes, dist_matrix, vehicle_capacity, horizon):
                        continue
                    old_to_cost = compute_route_cost(to_route, dist_matrix)
                    new_to_cost = compute_route_cost(new_to_route, dist_matrix)
                    increase_to = new_to_cost - old_to_cost
                    total_change = increase_to - savings_from
                    if total_change < best_increase:
                        best_increase = total_change
                        best_insertion = ('existing', to_route_idx, j)
            
            if best_increase < -1e-5:
                routes[from_route_idx] = new_from_route
                if best_insertion[0] == 'new':
                    routes.append(new_route_candidate)
                    total_cost += best_increase
                    improved = True
                else:
                    to_route_idx = best_insertion[1]
                    j = best_insertion[2]
                    to_route = routes[to_route_idx]
                    new_to_route = to_route[:j] + [customer] + to_route[j:]
                    routes[to_route_idx] = new_to_route
                    total_cost += best_increase
                    improved = True
                if logger:
                    logger.log(total_cost)
                return routes, total_cost, True
    return routes, total_cost, False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    start_time = time.time()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    with open(args.instance_path) as f:
        instance = json.load(f)
    
    params = instance['problem_parameters']
    n_nodes = params['n_nodes']
    n_customers = params['n_customers']
    vehicle_capacity = params['vehicle_capacity']
    planning_horizon = params['planning_horizon']
    
    nodes = instance['nodes']
    id_to_index = {node['id']: idx for idx, node in enumerate(nodes)}
    index_to_id = [node['id'] for node in nodes]
    
    depot_source_idx = None
    depot_sink_idx = None
    customer_indices = []
    for idx, node in enumerate(nodes):
        if node['type'] == 'depot_source':
            depot_source_idx = idx
        elif node['type'] == 'depot_sink':
            depot_sink_idx = idx
        elif node['type'] == 'customer':
            customer_indices.append(idx)
    
    if depot_source_idx is None or depot_sink_idx is None:
        raise ValueError("Depot source or sink not found")
    
    dist_matrix = compute_distance_matrix(nodes)
    feasible_arc = compute_feasible_arc(nodes, vehicle_capacity, dist_matrix)
    
    routes, total_cost, unserved = insertion_heuristic(
        nodes, dist_matrix, feasible_arc, vehicle_capacity, planning_horizon,
        depot_source_idx, depot_sink_idx, customer_indices, args.time_limit,
        start_time, logger
    )
    
    if unserved:
        pass
    
    improved = True
    while improved and time.time() - start_time < args.time_limit:
        routes, total_cost, improved = do_relocate_pass(
            routes, total_cost, nodes, dist_matrix, feasible_arc,
            vehicle_capacity, planning_horizon, depot_source_idx,
            depot_sink_idx, args.time_limit, start_time, logger
        )
    
    solution_routes = []
    for route in routes:
        solution_routes.append([index_to_id[i] for i in route])
    
    solution = {
        "objective_value": total_cost,
        "routes": solution_routes
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == '__main__':
    main()
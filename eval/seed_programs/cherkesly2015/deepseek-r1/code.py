import json
import time
import argparse
from solution_logger import SolutionLogger

def build_route_from_pickup_sequence(pickup_sequence, n_requests, depot_dest_id):
    delivery_sequence = [r + n_requests for r in reversed(pickup_sequence)]
    return [0] + pickup_sequence + delivery_sequence + [depot_dest_id]

def check_route(route, node_info, arc_time_dict, capacity):
    current_time = max(0, node_info[route[0]]['tw_early'])
    current_load = 0
    
    for i in range(1, len(route)):
        from_node = route[i-1]
        to_node = route[i]
        
        if (from_node, to_node) not in arc_time_dict:
            return False, float('inf')
        travel_time = arc_time_dict[(from_node, to_node)]
        
        current_time += node_info[from_node]['service_time'] + travel_time
        if current_time < node_info[to_node]['tw_early']:
            current_time = node_info[to_node]['tw_early']
        if current_time > node_info[to_node]['tw_late']:
            return False, float('inf')
            
        current_load += node_info[to_node]['load']
        if current_load < 0 or current_load > capacity:
            return False, float('inf')
            
    return True, current_time

def compute_route_cost(route, arc_cost_dict):
    total_cost = 0
    for i in range(len(route)-1):
        from_node = route[i]
        to_node = route[i+1]
        total_cost += arc_cost_dict.get((from_node, to_node), float('inf'))
    return total_cost

def main(instance_path, solution_path, time_limit, log_path=None):
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    with open(instance_path, 'r') as f:
        data = json.load(f)
    
    n_requests = data['n_requests']
    depot_dest_id = 2 * n_requests + 1
    vehicle_capacity = data['vehicle_capacity']
    
    node_info = {}
    for node in data['nodes']:
        node_info[node['node_id']] = node
    
    arc_cost_dict = {}
    arc_time_dict = {}
    for arc in data['arcs']:
        arc_cost_dict[(arc['from'], arc['to'])] = arc['cost']
        arc_time_dict[(arc['from'], arc['to'])] = arc['travel_time']
    
    routes = []
    unassigned_requests = set(range(1, n_requests+1))
    start_time = time.time()
    
    while unassigned_requests and time.time() - start_time < time_limit:
        best_cost_increase = float('inf')
        best_route_index = None
        best_request = None
        best_position = None
        
        for idx, pickup_sequence in enumerate(routes):
            for req in unassigned_requests:
                for j in range(0, len(pickup_sequence)+1):
                    new_pickup_sequence = pickup_sequence[:j] + [req] + pickup_sequence[j:]
                    new_route = build_route_from_pickup_sequence(new_pickup_sequence, n_requests, depot_dest_id)
                    feasible, _ = check_route(new_route, node_info, arc_time_dict, vehicle_capacity)
                    if not feasible:
                        continue
                    new_cost = compute_route_cost(new_route, arc_cost_dict)
                    old_route = build_route_from_pickup_sequence(pickup_sequence, n_requests, depot_dest_id)
                    old_cost = compute_route_cost(old_route, arc_cost_dict)
                    cost_increase = new_cost - old_cost
                    if cost_increase < best_cost_increase:
                        best_cost_increase = cost_increase
                        best_route_index = idx
                        best_request = req
                        best_position = j
        
        new_vehicle_cost = None
        best_new_vehicle_request = None
        for req in unassigned_requests:
            new_route = [0, req, req + n_requests, depot_dest_id]
            feasible, _ = check_route(new_route, node_info, arc_time_dict, vehicle_capacity)
            if feasible:
                cost = compute_route_cost(new_route, arc_cost_dict)
                if cost < (new_vehicle_cost if new_vehicle_cost is not None else float('inf')):
                    new_vehicle_cost = cost
                    best_new_vehicle_request = req
        
        if best_route_index is not None and best_new_vehicle_request is not None:
            if best_cost_increase < new_vehicle_cost:
                best_new_vehicle_request = None
            else:
                best_route_index = None
        elif best_route_index is None and best_new_vehicle_request is None:
            break
        
        if best_route_index is not None:
            pickup_sequence = routes[best_route_index]
            new_pickup_sequence = pickup_sequence[:best_position] + [best_request] + pickup_sequence[best_position:]
            routes[best_route_index] = new_pickup_sequence
            unassigned_requests.remove(best_request)
        elif best_new_vehicle_request is not None:
            routes.append([best_new_vehicle_request])
            unassigned_requests.remove(best_new_vehicle_request)
    
    for req in unassigned_requests:
        routes.append([req])
    
    total_cost = 0
    solution_routes = []
    for vehicle_id, pickup_sequence in enumerate(routes):
        full_route = build_route_from_pickup_sequence(pickup_sequence, n_requests, depot_dest_id)
        current_time = max(0, node_info[0]['tw_early'])
        current_load = 0
        route_details = []
        
        for node_id in full_route:
            if node_id != 0:
                prev_node = full_route[full_route.index(node_id)-1]
                travel_time = arc_time_dict.get((prev_node, node_id), 0)
                current_time += node_info[prev_node]['service_time'] + travel_time
                if current_time < node_info[node_id]['tw_early']:
                    current_time = node_info[node_id]['tw_early']
            current_load += node_info[node_id]['load']
            route_details.append({
                "node_id": node_id,
                "arrival_time": current_time,
                "load_after": current_load
            })
        
        total_cost += compute_route_cost(full_route, arc_cost_dict)
        solution_routes.append({
            "vehicle_id": vehicle_id,
            "route_sequence": full_route,
            "route_details": route_details
        })
    
    solution = {
        "objective_value": total_cost,
        "routes": solution_routes
    }
    
    if logger:
        logger.log_solution(total_cost, solution)
    
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()
    main(args.instance_path, args.solution_path, args.time_limit, args.log_path)
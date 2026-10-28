import json
import time
import sys
import os

def check_route(route, node_info, arc_time_dict, arc_cost_dict, capacity):
    current_time = max(0, node_info[route[0]]['tw_early'])
    current_load = 0
    total_cost = 0
    
    for i in range(1, len(route)):
        from_node = route[i-1]
        to_node = route[i]
        
        if (from_node, to_node) not in arc_time_dict:
            return False, None
        travel_time = arc_time_dict[(from_node, to_node)]
        cost_segment = arc_cost_dict[(from_node, to_node)]
        total_cost += cost_segment
        
        current_time += node_info[from_node]['service_time'] + travel_time
        if current_time < node_info[to_node]['tw_early']:
            current_time = node_info[to_node]['tw_early']
        if current_time > node_info[to_node]['tw_late']:
            return False, None
            
        current_load += node_info[to_node]['load']
        if current_load < 0 or current_load > capacity:
            return False, None
            
    return True, total_cost

def compute_route_cost(route, arc_cost_dict):
    total_cost = 0
    for i in range(len(route)-1):
        from_node = route[i]
        to_node = route[i+1]
        if (from_node, to_node) in arc_cost_dict:
            total_cost += arc_cost_dict[(from_node, to_node)]
    return total_cost

def main(instance_path, solution_path):
    with open(instance_path, 'r') as f:
        data = json.load(f)
    
    n_requests = data['n_requests']
    depot_dest_id = 2 * n_requests + 1
    vehicle_capacity = data['vehicle_capacity']
    time_limit = 300  # 5 minutes in seconds
    
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
        
        for idx, (pickup_list, delivery_list) in enumerate(routes):
            for req in unassigned_requests:
                for j in range(0, len(pickup_list)+1):
                    new_pickup_list = pickup_list[:j] + [req] + pickup_list[j:]
                    new_delivery_list = delivery_list[:len(pickup_list)-j] + [req + n_requests] + delivery_list[len(pickup_list)-j:]
                    new_route = [0] + new_pickup_list + new_delivery_list + [depot_dest_id]
                    feasible, cost = check_route(new_route, node_info, arc_time_dict, arc_cost_dict, vehicle_capacity)
                    if not feasible:
                        continue
                    old_route = [0] + pickup_list + delivery_list + [depot_dest_id]
                    old_cost = compute_route_cost(old_route, arc_cost_dict)
                    cost_increase = cost - old_cost
                    if cost_increase < best_cost_increase:
                        best_cost_increase = cost_increase
                        best_route_index = idx
                        best_request = req
                        best_position = j
        
        for req in unassigned_requests:
            new_route = [0, req, req + n_requests, depot_dest_id]
            feasible, cost = check_route(new_route, node_info, arc_time_dict, arc_cost_dict, vehicle_capacity)
            if feasible and cost < best_cost_increase:
                best_cost_increase = cost
                best_route_index = None
                best_request = req
                best_position = 0
        
        if best_request is None:
            break
        
        if best_route_index is not None:
            pickup_list, delivery_list = routes[best_route_index]
            new_pickup_list = pickup_list[:best_position] + [best_request] + pickup_list[best_position:]
            new_delivery_list = delivery_list[:len(pickup_list)-best_position] + [best_request + n_requests] + delivery_list[len(pickup_list)-best_position:]
            routes[best_route_index] = (new_pickup_list, new_delivery_list)
        else:
            routes.append(([best_request], [best_request + n_requests]))
        unassigned_requests.remove(best_request)
    
    for req in unassigned_requests:
        routes.append(([req], [req + n_requests]))
    
    total_cost = 0
    solution_routes = []
    for idx, (pickup_list, delivery_list) in enumerate(routes):
        full_route = [0] + pickup_list + delivery_list + [depot_dest_id]
        current_time = max(0, node_info[0]['tw_early'])
        current_load = 0
        route_details = []
        
        for i, node_id in enumerate(full_route):
            if i > 0:
                from_node = full_route[i-1]
                to_node = node_id
                travel_time = arc_time_dict[(from_node, to_node)]
                current_time += node_info[from_node]['service_time'] + travel_time
                if current_time < node_info[to_node]['tw_early']:
                    current_time = node_info[to_node]['tw_early']
            current_load += node_info[node_id]['load']
            route_details.append({
                "node_id": node_id,
                "arrival_time": current_time,
                "load_after": current_load
            })
        
        total_cost += compute_route_cost(full_route, arc_cost_dict)
        solution_routes.append({
            "vehicle_id": idx,
            "route_sequence": full_route,
            "route_details": route_details
        })
    
    solution = {
        "objective_value": total_cost,
        "routes": solution_routes
    }
    
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    args = parser.parse_args()
    main(args.instance_path, args.solution_path)
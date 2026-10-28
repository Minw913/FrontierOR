import argparse
import json
import math
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description="Solve the Fleet Routing Problem.")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to the JSON instance file.")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to write the solution JSON file.")
    parser.add_argument("--time_limit", type=int, required=True, help="Maximum runtime in seconds.")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log intermediate solutions.")
    args = parser.parse_args()

    try:
        instance = json.load(open(args.instance_path, 'r'))
    except FileNotFoundError:
        print(f"Error: Instance file not found at path {args.instance_path}")
        return
    
    depot = instance['depot']
    truck_customers = instance['truck_customers']
    trailer_customers = instance['trailer_customers']
    transshipment_locations = instance['transshipment_locations']
    distance_matrix = instance['distance_matrix']
    travel_time_matrix = instance['travel_time_matrix']
    parameters = instance['parameters']
    fleet_data = instance['fleet']
    vehicle_classes = fleet_data['vehicle_classes']

    combined_locations = [depot] + truck_customers + trailer_customers + transshipment_locations
    n_orig = len(combined_locations)
    id_to_index = {loc['id']: idx for idx, loc in enumerate(combined_locations)}
    depot_id = depot['id']
    depot_idx = id_to_index[depot_id]
    
    planning_horizon = parameters['planning_horizon']
    max_route_duration = min(parameters['max_route_duration'], 
                             planning_horizon[1] - planning_horizon[0])

    service_time = parameters.get('service_time', 0.0)
    
    all_customers = truck_customers + trailer_customers
    solution = {
        "objective_value": 0.0,
        "routes": []
    }
    
    total_cost = 0
    routes = []
    class_usage = {vclass['class_id']: 0 for vclass in vehicle_classes}
    
    for cust in all_customers:
        cust_id = cust['id']
        if cust in truck_customers:
            cust_type = 'truck'
        else:
            cust_type = 'trailer'
        
        cust_idx = id_to_index[cust_id]
        dist_depot_to_cust = distance_matrix[depot_idx][cust_idx]
        dist_cust_to_depot = distance_matrix[cust_idx][depot_idx]
        travel_time_depot_to_cust = travel_time_matrix[depot_idx][cust_idx]
        travel_time_cust_to_depot = travel_time_matrix[cust_idx][depot_idx]
        travel_time_total = travel_time_depot_to_cust + travel_time_cust_to_depot
        svc_time = cust.get('service_time', service_time)
        total_route_time = travel_time_total + svc_time

        if total_route_time > max_route_duration:
            raise RuntimeError(f"Customer {cust_id} cannot be served within the given constraints. Route time: {total_route_time}, max allowed: {max_route_duration}")
        
        best_cost = float('inf')
        best_class = None
        demand = cust['supply']
        capacity_needed = demand
        
        for vclass in vehicle_classes:
            class_id = vclass['class_id']
            if vclass['num_available'] <= class_usage.get(class_id, 0):
                continue
            total_cap = vclass['total_capacity']
            if total_cap < capacity_needed:
                continue
            fixed_cost = vclass['truck_fixed_cost']
            if cust_type == 'truck':
                if vclass['trailer_type_id'] != 0:  # Only solo for truck_customers
                    continue
            truck_cf = vclass['truck_cost_factor']
            trailer_mult = vclass.get('trailer_towing_cost_multiplier', 0.0) if vclass['trailer_type_id'] != 0 else 0.0
            var_cost_per_dist = truck_cf + trailer_mult
            route_dist = dist_depot_to_cust + dist_cust_to_depot
            var_cost = var_cost_per_dist * route_dist
            cost = fixed_cost + var_cost
            
            if cost < best_cost:
                best_cost = cost
                best_class = vclass
                
        if best_class is None:
            available_classes = [v['class_id'] for v in vehicle_classes]
            raise RuntimeError(f"Customer {cust_id} could not be assigned; Demand: {demand}; Available classes: {available_classes}")
        
        class_id = best_class['class_id']
        class_usage[class_id] = class_usage.get(class_id, 0) + 1
        total_cost += best_cost
        sol_route = {
            "vehicle_class": class_id,
            "vehicle_index": class_usage[class_id],
            "customers_served": [cust_id],
            "route_sequence": [
                depot_id,
                cust_id,
                depot_id
            ],
            "route_sequence_physical": [
                depot_id,
                cust_id,
                depot_id
            ],
            "decouple_locations": [],
            "couple_locations": [],
            "arrival_times": {
                str(depot_id): planning_horizon[0],
                str(cust_id): planning_horizon[0] + travel_time_depot_to_cust,
                str(depot_id) + "_back": planning_horizon[0] + travel_time_depot_to_cust + svc_time + travel_time_cust_to_depot
            }
        }
        routes.append(sol_route)

    solution['routes'] = routes
    solution['objective_value'] = total_cost

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=4)

if __name__ == "__main__":
    main()
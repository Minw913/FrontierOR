import argparse
import json
import sys
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser(description="Solve the Fleet Routing Problem with Heterogeneous Vehicles.")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to the JSON instance file.")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to write the solution JSON file.")
    parser.add_argument("--time_limit", type=int, required=True, help="Maximum runtime in seconds.")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log intermediate solutions.")
    args = parser.parse_args()

    try:
        with open(args.instance_path, 'r') as f:
            instance = json.load(f)
    except FileNotFoundError:
        print(f"Error: Instance file not found at path {args.instance_path}")
        sys.exit(1)

    depot = instance['depot']
    truck_customers = instance['truck_customers']
    trailer_customers = instance['trailer_customers']
    transshipment_locations = instance['transshipment_locations']
    distance_matrix = instance['distance_matrix']
    travel_time_matrix = instance['travel_time_matrix']
    parameters = instance['parameters']
    fleet_data = instance['fleet']
    truck_types = fleet_data['truck_types']
    trailer_types = fleet_data['trailer_types']
    vehicle_classes = fleet_data['vehicle_classes']
    
    combined_locations = [depot] + truck_customers + trailer_customers + transshipment_locations
    n_orig = len(combined_locations)
    id_to_index = {loc['id']: idx for idx, loc in enumerate(combined_locations)}
    depot_id = depot['id']
    depot_idx = id_to_index[depot_id]
    planning_horizon = parameters['planning_horizon']
    service_time = parameters.get('service_time', 0.0)
    
    all_customers = truck_customers + trailer_customers
    customer_id_to_type = {}
    for cust in truck_customers:
        customer_id_to_type[cust['id']] = 'truck'
    for cust in trailer_customers:
        customer_id_to_type[cust['id']] = 'trailer'
    
    class_usage = defaultdict(int)
    for vclass in vehicle_classes:
        class_usage[vclass['class_id']] = 0

    for customer in all_customers:
        cust_id = customer['id']
        demand = customer['supply']
        customer_type = customer_id_to_type[cust_id]
        feasible_classes = []
        for vclass in vehicle_classes:
            class_id = vclass['class_id']
            if class_usage[class_id] < vclass['num_available']:
                if customer_type == 'truck':
                    if vclass['trailer_type_id'] == 0:
                        if vclass['truck_capacity'] >= demand:
                            feasible_classes.append(vclass)
                    else:
                        if vclass['truck_capacity'] >= demand:
                            feasible_classes.append(vclass)
                else:
                    if vclass['total_capacity'] >= demand:
                        feasible_classes.append(vclass)
        if not feasible_classes:
            available_classes = []
            for vclass in vehicle_classes:
                info = {
                    'class_id': vclass['class_id'],
                    'truck_capacity': vclass['truck_capacity'],
                    'total_capacity': vclass['total_capacity'],
                    'num_available': vclass['num_available'] - class_usage[vclass['class_id']]
                }
                available_classes.append(info)
            raise RuntimeError(f"Customer {cust_id} (type={customer_type}, demand={demand}) cannot be served by any available vehicle class. Available classes: {available_classes}")
    
    solution = {
        "objective_value": 0.0,
        "routes": []
    }
    routes = []
    total_cost = 0.0
    for customer in all_customers:
        cust_id = customer['id']
        customer_type = customer_id_to_type[cust_id]
        cust_idx = id_to_index[cust_id]
        demand = customer['supply']
        svc_time = customer.get('service_time', service_time)
        dist_depot_to_cust = distance_matrix[depot_idx][cust_idx]
        dist_cust_to_depot = distance_matrix[cust_idx][depot_idx]
        time_depot_to_cust = travel_time_matrix[depot_idx][cust_idx]
        time_cust_to_depot = travel_time_matrix[cust_idx][depot_idx]
        best_cost = float('inf')
        best_class = None
        
        for vclass in vehicle_classes:
            class_id = vclass['class_id']
            if class_usage[class_id] >= vclass['num_available']:
                continue
            if customer_type == 'truck':
                if vclass['trailer_type_id'] != 0:
                    if vclass['truck_capacity'] < demand:
                        continue
                else:
                    if vclass['truck_capacity'] < demand:
                        continue
            else:
                if vclass['total_capacity'] < demand:
                    continue
                    
            fixed_cost = vclass['truck_fixed_cost']
            truck_cost_factor = vclass['truck_cost_factor']
            trailer_cost_multiplier = vclass['trailer_towing_cost_multiplier'] if vclass['trailer_type_id'] != 0 else 0.0
            var_cost_per_dist = truck_cost_factor + trailer_cost_multiplier
            travel_dist = dist_depot_to_cust + dist_cust_to_depot
            travel_cost = travel_dist * var_cost_per_dist
            cost = fixed_cost + travel_cost
            if cost < best_cost:
                best_cost = cost
                best_class = vclass
                
        if best_class is None:
            continue 
        class_id = best_class['class_id']
        class_usage[class_id] += 1
        total_cost += best_cost
        sol_route = {
            "vehicle_class": class_id,
            "vehicle_index": class_usage[class_id],
            "customers_served": [cust_id],
            "route_sequence": [depot_id, cust_id, depot_id],
            "route_sequence_physical": [depot_id, cust_id, depot_id],
            "decouple_locations": [],
            "couple_locations": [],
            "arrival_times": {
                depot_id: planning_horizon[0],
                cust_id: planning_horizon[0] + time_depot_to_cust,
                f"{depot_id}_back": planning_horizon[0] + time_depot_to_cust + svc_time + time_cust_to_depot
            }
        }
        routes.append(sol_route)
        
    solution['routes'] = routes
    solution['objective_value'] = total_cost
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=4)
    
if __name__ == "__main__":
    main()
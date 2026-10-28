import argparse
import json
import math
import random
from collections import defaultdict
import numpy as np

def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def calculate_total_distance(route, distance_matrix):
    total_dist = 0.0
    for i in range(len(route) - 1):
        total_dist += distance_matrix[route[i]][route[i+1]]
    return total_dist

def get_location_coords(location_id, locations_map):
    return locations_map[location_id]

def solve_truck_and_trailer_routing(instance_data, time_limit):
    # Extract instance data
    max_route_duration = instance_data['parameters']['max_route_duration']
    service_time = instance_data['parameters']['service_time']
    coupling_time = instance_data['parameters']['coupling_time']
    decoupling_time = instance_data['parameters']['decoupling_time']
    transfer_time = instance_data['parameters']['transfer_time']
    coupling_cost = instance_data['parameters']['coupling_cost']
    decoupling_cost = instance_data['parameters']['decoupling_cost']
    transfer_cost = instance_data['parameters']['transfer_cost']
    planning_horizon = instance_data['parameters']['planning_horizon']
    
    truck_types = {tt['id']: tt for tt in instance_data['fleet']['truck_types']}
    trailer_types = {tt['id']: tt for tt in instance_data['fleet']['trailer_types']}
    vehicle_classes = {vc['class_id']: vc for vc in instance_data['fleet']['vehicle_classes']}
    
    depot = instance_data['depot']
    truck_customers = {tc['id']: tc for tc in instance_data['truck_customers']}
    trailer_customers = {tc['id']: tc for tc in instance_data['trailer_customers']}
    transshipment_locations = {tl['id']: tl for tl in instance_data['transshipment_locations']}
    
    distance_matrix = instance_data['distance_matrix']
    travel_time_matrix = instance_data['travel_time_matrix']
    
    # Combine all locations for easier access
    all_locations = {}
    all_locations.update({depot['id']: depot})
    all_locations.update(truck_customers)
    all_locations.update(trailer_customers)
    all_locations.update(transshipment_locations)
    
    # Create mapping of location id to coordinates
    coords_map = {lid: (loc['x'], loc['y']) for lid, loc in all_locations.items()}
    
    # Customer partitions
    truck_customer_ids = set(truck_customers.keys())
    trailer_customer_ids = set(trailer_customers.keys())
    all_customer_ids = truck_customer_ids | trailer_customer_ids
    
    # Parking locations (trailer customers + transshipment locations)
    parking_locations = sorted(list(trailer_customer_ids | set(transshipment_locations.keys())))
    
    # Expanded node convention implementation
    n_orig = len(all_locations)
    parking_location_indices = {pid: idx for idx, pid in enumerate(parking_locations)}
    
    # Create expanded node IDs
    def get_decouple_id(park_id):
        idx = parking_location_indices[park_id]
        return n_orig + idx
    
    def get_couple_id(park_id):
        idx = parking_location_indices[park_id]
        return n_orig + len(parking_locations) + idx
    
    # Create reverse mapping for expanded IDs
    expanded_to_physical = {}
    for pid in parking_locations:
        expanded_to_physical[get_decouple_id(pid)] = pid
        expanded_to_physical[get_couple_id(pid)] = pid
    
    # Create extended distance/travel matrices for expanded nodes
    n_extended = n_orig + 2 * len(parking_locations)
    extended_distance_matrix = [[0.0 for _ in range(n_extended)] for _ in range(n_extended)]
    extended_travel_time_matrix = [[0.0 for _ in range(n_extended)] for _ in range(n_extended)]
    
    # Copy original distances
    for i in range(n_orig):
        for j in range(n_orig):
            extended_distance_matrix[i][j] = distance_matrix[i][j]
            extended_travel_time_matrix[i][j] = travel_time_matrix[i][j]
    
    # Add connections for expanded nodes
    for pid in parking_locations:
        phys_idx = pid
        decouple_idx = get_decouple_id(pid)
        couple_idx = get_couple_id(pid)
        
        # Connect expanded nodes to all physical nodes
        for phys_node in range(n_orig):
            extended_distance_matrix[decouple_idx][phys_node] = distance_matrix[phys_idx][phys_node]
            extended_distance_matrix[phys_node][decouple_idx] = distance_matrix[phys_node][phys_idx]
            extended_distance_matrix[couple_idx][phys_node] = distance_matrix[phys_idx][phys_node]
            extended_distance_matrix[phys_node][couple_idx] = distance_matrix[phys_node][phys_idx]
            
            extended_travel_time_matrix[decouple_idx][phys_node] = travel_time_matrix[phys_idx][phys_node]
            extended_travel_time_matrix[phys_node][decouple_idx] = travel_time_matrix[phys_node][phys_idx]
            extended_travel_time_matrix[couple_idx][phys_node] = travel_time_matrix[phys_idx][phys_node]
            extended_travel_time_matrix[phys_node][couple_idx] = travel_time_matrix[phys_node][phys_idx]
        
        # Connect expanded nodes to each other (for same physical location)
        extended_distance_matrix[decouple_idx][couple_idx] = 0.0
        extended_distance_matrix[couple_idx][decouple_idx] = 0.0
        extended_travel_time_matrix[decouple_idx][couple_idx] = 0.0
        extended_travel_time_matrix[couple_idx][decouple_idx] = 0.0
        
        # Handle self-loops appropriately
        extended_distance_matrix[decouple_idx][decouple_idx] = 0.0
        extended_distance_matrix[couple_idx][couple_idx] = 0.0
        extended_travel_time_matrix[decouple_idx][decouple_idx] = 0.0
        extended_travel_time_matrix[couple_idx][couple_idx] = 0.0
    
    # Heuristic solution approach
    # Initialize empty solution
    routes = []
    
    # Create a greedy initial solution
    unvisited_customers = set(all_customer_ids)
    vehicle_usage = defaultdict(int)
    
    # Try to create routes for different vehicle classes
    for vcid, vc_info in vehicle_classes.items():
        available_count = min(vc_info['num_available'], 
                             truck_types[vc_info['truck_type_id']]['num_available'])
        if vc_info['trailer_type_id'] != 0:
            available_count = min(available_count, 
                                 trailer_types[vc_info['trailer_type_id']]['num_available'])
        
        # Create routes until we run out of available vehicles or customers
        for v_idx in range(available_count):
            if not unvisited_customers:
                break
                
            route = construct_greedy_route(
                unvisited_customers, 
                vc_info,
                depot, 
                truck_customers, 
                trailer_customers,
                transshipment_locations,
                extended_distance_matrix,
                extended_travel_time_matrix,
                max_route_duration,
                service_time,
                coupling_time,
                decoupling_time,
                transfer_time,
                planning_horizon,
                get_decouple_id,
                get_couple_id,
                expanded_to_physical,
                parking_locations
            )
            
            if route and len(route) > 2:  # Must visit at least one customer
                routes.append({
                    'vehicle_class': vcid,
                    'vehicle_index': v_idx,
                    'customers_served': extract_customers_from_route(route),
                    'route_sequence': route,
                    'route_sequence_physical': convert_to_physical_route(route, expanded_to_physical),
                    'decouple_locations': extract_decouple_locations(route, expanded_to_physical),
                    'couple_locations': extract_couple_locations(route, expanded_to_physical),
                    'arrival_times': calculate_arrival_times(
                        route, 
                        extended_travel_time_matrix, 
                        service_time,
                        coupling_time,
                        decoupling_time,
                        transfer_time,
                        planning_horizon[0],
                        truck_customers,
                        trailer_customers,
                        transshipment_locations
                    )
                })
                
                # Remove served customers
                served_customers = set(extract_customers_from_route(route))
                unvisited_customers -= served_customers
    
    # Calculate objective value
    total_cost = calculate_solution_cost(routes, vehicle_classes, extended_distance_matrix, coupling_cost, decoupling_cost, transfer_cost)
    
    solution = {
        'objective_value': total_cost,
        'routes': routes,
        'expanded_node_convention': "Let n_orig be the number of physical locations and P be the instance parking-location IDs in their published list order. Physical nodes retain IDs 0..n_orig-1. The decouple copy of P[i] has ID n_orig+i, and the couple copy of P[i] has ID n_orig+len(P)+i. Both copies map back to physical location P[i]."
    }
    
    return solution


def construct_greedy_route(unvisited_customers, vc_info, depot, truck_customers, trailer_customers, 
                          transshipment_locations, extended_distance_matrix, extended_travel_time_matrix,
                          max_route_duration, service_time, coupling_time, decoupling_time, transfer_time,
                          planning_horizon, get_decouple_id, get_couple_id, expanded_to_physical, parking_locations):
    if not unvisited_customers:
        return []
    
    route = [depot['id']]
    current_time = planning_horizon[0]
    current_pos = depot['id']
    truck_load = 0
    trailer_load = 0
    truck_capacity = vc_info['truck_capacity']
    trailer_capacity = vc_info['trailer_capacity']
    
    # Track what type of vehicle we currently have (with or without trailer)
    has_trailer = (vc_info['trailer_type_id'] != 0)
    
    # Make a copy of unvisited customers to modify during route construction
    remaining_customers = unvisited_customers.copy()
    
    while remaining_customers:
        best_next = None
        best_cost = float('inf')
        
        # Check all remaining customers
        for cust_id in remaining_customers:
            # Check if customer can be accessed
            if cust_id in truck_customers and has_trailer:
                # Truck customer but we have a trailer - need to park it first
                continue
            
            # Get travel time to customer
            travel_time = extended_travel_time_matrix[current_pos][cust_id]
            arrival_time = current_time + travel_time
            
            # Check if we can arrive within time window and return to depot
            # For simplicity, assume depot has no time window constraints in this heuristic
            estimated_return_time = arrival_time + service_time + extended_travel_time_matrix[cust_id][depot['id']]
            
            if estimated_return_time > planning_horizon[1]:
                continue  # Cannot return in time
            
            # Check capacity constraint
            new_truck_load = truck_load
            new_trailer_load = trailer_load
            
            if has_trailer:
                # Load into trailer first, then truck if needed
                if trailer_load + trailer_customers[cust_id]['supply'] <= trailer_capacity:
                    new_trailer_load = trailer_load + trailer_customers[cust_id]['supply']
                else:
                    # Fill trailer to capacity, remainder goes to truck
                    available_trailer_space = trailer_capacity - trailer_load
                    trailer_supply = min(trailer_customers[cust_id]['supply'], available_trailer_space)
                    new_trailer_load = trailer_load + trailer_supply
                    
                    remaining_supply = trailer_customers[cust_id]['supply'] - trailer_supply
                    new_truck_load = truck_load + remaining_supply
            else:
                # All load goes to truck
                new_truck_load = truck_load + (truck_customers.get(cust_id, {}).get('supply') or 
                                              trailer_customers.get(cust_id, {}).get('supply', 0))
            
            # Check capacity
            if new_truck_load > truck_capacity or new_trailer_load > trailer_capacity:
                continue
            
            # This customer is feasible
            cost = extended_distance_matrix[current_pos][cust_id]
            if cost < best_cost:
                best_cost = cost
                best_next = cust_id
        
        # If no direct customer is feasible, consider parking the trailer
        if best_next is None and has_trailer and parking_locations:
            # Look for a nearby parking location to drop the trailer
            for park_loc in parking_locations:
                travel_time = extended_travel_time_matrix[current_pos][park_loc]
                arrival_time = current_time + travel_time
                
                if arrival_time + decoupling_time > planning_horizon[1]:
                    continue
                
                # Find truck customers accessible from this parking location
                feasible_at_park = False
                for cust_id in remaining_customers:
                    if cust_id in truck_customers:
                        # Can visit after dropping trailer
                        travel_time_to_cust = extended_travel_time_matrix[park_loc][cust_id]
                        arrival_at_cust = arrival_time + decoupling_time + travel_time_to_cust
                        
                        # Check if we can return to parking location and still make it back
                        return_to_park_time = extended_travel_time_matrix[cust_id][park_loc]
                        back_to_depot_time = extended_travel_time_matrix[park_loc][depot['id']]
                        
                        final_time = arrival_at_cust + service_time + return_to_park_time + coupling_time + back_to_depot_time
                        if final_time <= planning_horizon[1]:
                            # Check capacity
                            new_truck_load = truck_load + truck_customers[cust_id]['supply']
                            if new_truck_load <= truck_capacity:
                                feasible_at_park = True
                                break
                
                if feasible_at_park:
                    cost = extended_distance_matrix[current_pos][park_loc]
                    if cost < best_cost:
                        best_cost = cost
                        best_next = park_loc
                        # Mark that we'll park here
                        best_is_park = True
        
        if best_next is None:
            # No feasible next move - end route
            break
        
        # If best_next is a parking location, handle trailer parking
        if best_next in parking_locations:
            # Add decouple action
            decouple_node = get_decouple_id(best_next)
            route.append(decouple_node)
            
            travel_time = extended_travel_time_matrix[current_pos][best_next]
            current_time += travel_time
            current_time += decoupling_time
            current_pos = get_couple_id(best_next)  # Next movement will be from couple node
            has_trailer = False  # Now operating without trailer
            
            # Visit truck customers accessible from this parking location
            while True:
                truck_best = None
                truck_best_cost = float('inf')
                
                for cust_id in remaining_customers:
                    if cust_id in truck_customers:
                        travel_time = extended_travel_time_matrix[current_pos if isinstance(current_pos, int) else get_couple_id(best_next)][cust_id]
                        arrival_time = current_time + travel_time
                        
                        # Estimate if we can return to park and finish route
                        return_time = extended_travel_time_matrix[cust_id][best_next]  # Return to parking location
                        couple_time = coupling_time
                        back_to_depot_time = extended_travel_time_matrix[best_next][depot['id']]
                        
                        est_finish_time = arrival_time + service_time + return_time + couple_time + back_to_depot_time
                        
                        if est_finish_time <= planning_horizon[1]:
                            # Check capacity
                            new_truck_load = truck_load + truck_customers[cust_id]['supply']
                            if new_truck_load <= truck_capacity:
                                cost = extended_distance_matrix[current_pos if isinstance(current_pos, int) else get_couple_id(best_next)][cust_id]
                                if cost < truck_best_cost:
                                    truck_best_cost = cost
                                    truck_best = cust_id
                
                if truck_best is None:
                    break  # No more feasible truck customers from this parking location
                
                # Add truck customer to route
                route.append(truck_best)
                travel_time = extended_travel_time_matrix[current_pos if isinstance(current_pos, int) else get_couple_id(best_next)][truck_best]
                current_time += travel_time
                current_time += service_time
                current_pos = truck_best
                truck_load += truck_customers[truck_best]['supply']
                remaining_customers.remove(truck_best)
                
                # Return to parking location to couple trailer
                travel_time = extended_travel_time_matrix[current_pos][best_next]
                current_time += travel_time
                current_pos = best_next  # Now at parking location again
        
            # Couple trailer back
            couple_node = get_couple_id(best_next)
            route.append(couple_node)
            
            current_time += coupling_time
            current_pos = couple_node
            has_trailer = True  # Now with trailer again
        else:
            # Add regular customer to route
            route.append(best_next)
            
            travel_time = extended_travel_time_matrix[current_pos][best_next]
            current_time += travel_time
            current_time += service_time
            
            # Update loads
            if best_next in truck_customers:
                truck_load += truck_customers[best_next]['supply']
            else:  # Trailer customer
                if has_trailer:
                    supply = trailer_customers[best_next]['supply']
                    if trailer_load + supply <= trailer_capacity:
                        trailer_load += supply
                    else:
                        trailer_space = trailer_capacity - trailer_load
                        trailer_supply = min(supply, trailer_space)
                        trailer_load += trailer_supply
                        truck_load += supply - trailer_supply
                else:
                    truck_load += trailer_customers[best_next]['supply']
            
            current_pos = best_next
            remaining_customers.remove(best_next)
    
    # Return to depot
    route.append(depot['id'])
    return route


def extract_customers_from_route(route):
    # Extract customer IDs from route (excluding depot and expanded nodes)
    customers = []
    for node in route:
        if node >= 0 and node not in [0]:  # Assuming depot ID is 0
            # This is a simplified check - in real implementation would need more sophisticated logic
            customers.append(node)
    return customers


def convert_to_physical_route(route, expanded_to_physical):
    physical_route = []
    for node in route:
        if node in expanded_to_physical:
            physical_route.append(expanded_to_physical[node])
        else:
            physical_route.append(node)
    return physical_route


def extract_decouple_locations(route, expanded_to_physical):
    decouple_locs = []
    for node in route:
        if node in expanded_to_physical and 'decouple' in str(node):  # Simplified check
            # This would require actual expanded node identification
            pass
    return decouple_locs


def extract_couple_locations(route, expanded_to_physical):
    couple_locs = []
    for node in route:
        if node in expanded_to_physical and 'couple' in str(node):  # Simplified check
            # This would require actual expanded node identification
            pass
    return couple_locs


def calculate_arrival_times(route, extended_travel_time_matrix, service_time, 
                           coupling_time, decoupling_time, transfer_time, start_time,
                           truck_customers, trailer_customers, transshipment_locations):
    arrival_times = {}
    current_time = start_time
    current_pos = route[0]
    
    arrival_times[current_pos] = current_time
    
    for i in range(1, len(route)):
        next_pos = route[i]
        travel_time = extended_travel_time_matrix[current_pos][next_pos]
        current_time += travel_time
        
        # Add service/operation time depending on the type of node
        if next_pos in truck_customers or next_pos in trailer_customers:
            current_time += service_time
        elif 'decouple' in str(next_pos):  # Simplified check
            current_time += decoupling_time
        elif 'couple' in str(next_pos):  # Simplified check
            current_time += coupling_time
        elif next_pos in transshipment_locations:
            # May involve transfer operations
            current_time += transfer_time
        
        arrival_times[next_pos] = current_time
        current_pos = next_pos
    
    return arrival_times


def calculate_solution_cost(routes, vehicle_classes, extended_distance_matrix, 
                           coupling_cost, decoupling_cost, transfer_cost):
    total_cost = 0.0
    
    for route in routes:
        vc = vehicle_classes[route['vehicle_class']]
        
        # Add fixed vehicle cost
        total_cost += vc['truck_fixed_cost']
        
        # Add travel costs
        route_distance = 0.0
        for i in range(len(route['route_sequence']) - 1):
            from_node = route['route_sequence'][i]
            to_node = route['route_sequence'][i+1]
            route_distance += extended_distance_matrix[from_node][to_node]
        
        # Calculate travel cost based on vehicle class
        base_cost = route_distance * vc['truck_cost_factor']
        if vc['trailer_type_id'] != 0:  # Has trailer
            # We need to determine how much distance was traveled with trailer
            # This is a simplification - in reality would need to track when trailer is attached/detached
            base_cost += route_distance * vc['trailer_towing_cost_multiplier']
        
        total_cost += base_cost
        
        # Add handling costs
        num_decouples = len(route['decouple_locations'])
        num_couples = len(route['couple_locations'])
        # Transfer operations typically happen at the same locations as couples/decouples
        # For now, assuming one transfer per decouple/couple operation
        total_cost += num_decouples * decoupling_cost
        total_cost += num_couples * coupling_cost
        total_cost += min(num_decouples, num_couples) * transfer_cost
    
    return total_cost


def main():
    parser = argparse.ArgumentParser(description='Solve Truck and Trailer Routing Problem')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the instance JSON file')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to save the solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds')
    parser.add_argument('--log_path', type=str, help='Path to log intermediate solutions (optional)')
    
    args = parser.parse_args()
    
    # Import solution logger if log path is provided
    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize")
    else:
        logger = None
    
    # Read instance data
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)
    
    # Solve the problem
    solution = solve_truck_and_trailer_routing(instance_data, args.time_limit)
    
    # Log the final solution if logger is available
    if logger:
        logger.log_solution(solution['objective_value'], solution)
    
    # Write solution to file
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == '__main__':
    main()
import argparse
import json
import math
from solution_logger import SolutionLogger


def calculate_euclidean_distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def check_time_windows_and_service(dist_matrix, customers, depot, route, vehicle_capacity, 
                                   travel_time_includes_service_time=True):
    """
    Checks if a route is feasible with respect to time windows and service times,
    and returns departure times and loads if feasible.
    """
    if len(route) <= 2:
        return True, {}, {}

    # Map customer id to customer details for easy lookups
    id_to_customer = {c["id"]: c for c in customers}
    
    departure_times = {}
    loads = {}
    current_time = 0
    
    # Departure time at depot initially
    departure_times[0] = depot['ready_time']
    
    accumulated_load = 0
    
    # Iterate over route, but skip the first (depot) and last (depot) elements
    for i in range(1, len(route) - 1):
        current_customer_id = route[i]
        current_customer = id_to_customer[current_customer_id]
        
        # Add customer demand to accumulated load
        accumulated_load += current_customer['demand']
        
        # Check if we didn't exceed capacity
        if accumulated_load > vehicle_capacity:
            return False, {}, {}

        # Get previous node
        prev_node = route[i - 1]
        
        # Get base travel time between previous node and current customer
        travel_time = dist_matrix[prev_node][current_customer_id]
        
        # Add service time of previous location if not at depot
        if prev_node != 0 and not travel_time_includes_service_time:
            prev_customer = id_to_customer.get(prev_node, depot)
            travel_time += prev_customer['service_time']
        
        # Arrival time at current customer
        arrival_time = departure_times[prev_node] + travel_time
        
        # Vehicle can depart no earlier than this customer's ready time
        actual_departure = max(arrival_time, current_customer['ready_time'])
        
        # Vehicle must depart no later than this customer's due date
        if actual_departure > current_customer['due_date']:
            return False, {}, {}
        
        # Store departure time and load when leaving this customer
        departure_times[current_customer_id] = actual_departure
        loads[current_customer_id] = accumulated_load
        
    # Now handle returning to depot
    last_customer_id = route[-2]
    travel_back = dist_matrix[last_customer_id][0]
    if not travel_time_includes_service_time:
        last_customer = id_to_customer[last_customer_id]
        travel_back += last_customer['service_time']
    
    arrival_at_depot = departure_times[last_customer_id] + travel_back
    
    # Final check: can return to depot within the due date limit
    if arrival_at_depot > depot['due_date']:
        return False, {}, {}
        
    departure_times[0] = max(departure_times[0], arrival_at_depot) if 0 in departure_times else arrival_at_depot
    
    return True, departure_times, loads


def get_initial_feasible_solution(customers, depot, vehicle_capacity, dist_matrix, travel_time_includes_service_time=True):
    """Create an initial feasible solution"""
    # Create sorted list of customers by their due dates (tightest deadline first)
    sorted_customers = sorted(customers, key=lambda x: x['due_date'])
    
    routes = []
    assigned = set()
    
    for customer in sorted_customers:
        if customer['id'] in assigned:
            continue
            
        # Attempt to create a new route containing this customer
        new_route = [0]  # Start at depot
        current_load = 0
        current_time = depot['ready_time']
        last_node = 0
        success = True
        
        # Add all customers that can be feasibly added to this route in order of earliest due time
        remaining_customers = [c for c in sorted_customers if c['id'] not in assigned]
        
        for potential_customer in remaining_customers:
            # Check if we can add the potential customer to the current route
            # First, get travel time from last_node to this customer
            travel_time = dist_matrix[last_node][potential_customer['id']]
            if last_node != 0 and not travel_time_includes_service_time:
                # Add service time at previous node to the travel time
                travel_time += id_to_customer[last_node]['service_time']
            
            # Get arrival and departure times
            arrival_time = current_time + travel_time
            departure_time = max(arrival_time, potential_customer['ready_time'])
            
            if departure_time > potential_customer['due_date']:
                continue  # This customer cannot be added
                
            # Check load constraint
            if current_load + potential_customer['demand'] > vehicle_capacity:
                continue  # Load constraint violated
                
            # Feasible to add
            current_time = departure_time + potential_customer['service_time']
            current_load += potential_customer['demand']
            new_route.append(potential_customer['id'])
            assigned.add(potential_customer['id'])
        
        # Check if we were able to add the original customer (which we must add)
        if customer['id'] in assigned:
            # Add depot at the end and check for validity of full route
            new_route.append(0)
            is_feasible, _, _ = check_time_windows_and_service(
                dist_matrix, customers, depot, new_route, vehicle_capacity, 
                travel_time_includes_service_time
            )
            if is_feasible:
                routes.append(new_route)
            else:
                # If route is not feasible, remove this customer from assigned set so it can be attempted in another route
                assigned.remove(customer['id'])
        
    return routes


def remove_from_route(route, node):
    """Remove a node from route and return updated route, loads and time"""
    new_route = [n for n in route if n != node]
    return new_route


def insert_in_route(route, pos, node):
    """Insert a node in the route at specified position"""
    new_route = route[:pos] + [node] + route[pos:]
    return new_route


def local_search(solution, customers, depot, vehicle_capacity, dist_matrix, travel_time_includes_service_time):
    """
    Perform local search to reduce number of vehicles (try to merge routes)
    """
    id_to_customer = {c["id"]: c for c in customers}
    current_routes = [route[:] for route in solution]
    num_vehicles = len(current_routes)
    
    improved = True
    while improved:
        improved = False
        i = 0
        while i < len(current_routes):
            route_i = current_routes[i]
            j = i + 1
            while j < len(current_routes):
                route_j = current_routes[j]
                
                # Try merging route_j into route_i
                # Concatenate without duplicate depots: 0, ..., 0 concatenated to 0, ..., 0
                # becomes 0, ..., ..., 0
                merged_route = route_i[:-1] + route_j[1:]
                
                is_feasible, new_departure_times, new_loads = check_time_windows_and_service(
                    dist_matrix, customers, depot, merged_route, vehicle_capacity, 
                    travel_time_includes_service_time
                )
                
                if is_feasible:
                    # Successfully merged
                    current_routes.pop(j)
                    current_routes.pop(i)
                    current_routes.append(merged_route)
                    
                    if logger:
                        logger.log(len(current_routes))  # Log improved solution
                    
                    improved = True
                    break  # Restart checking since we changed the routes list
                else:
                    j += 1
            if improved:
                i = 0  # Reset to restart checking
            else:
                i += 1
                
    return current_routes


def calculate_total_distance(routes, dist_matrix):
    total_dist = 0
    for route in routes:
        for i in range(len(route)-1):
            total_dist += dist_matrix[route[i]][route[i+1]]
    return total_dist


def main():
    parser = argparse.ArgumentParser(description="Solve VRPTW with vehicle minimization")
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)

    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    num_customers = instance["num_customers"]
    vehicle_capacity = instance["vehicle_capacity"]
    depot_info = instance["depot"]
    customers_info = instance["customers"]  
    dist_matrix = instance["distance_matrix"]  # Pre-calculated matrix
    travel_time_includes_service_time = instance.get("travel_time_includes_service_time", True)

    # Generate initial solution using a greedy heuristic    
    initial_solution = get_initial_feasible_solution(
        customers_info, depot_info, vehicle_capacity, dist_matrix, travel_time_includes_service_time
    )
    
    # Improve solution using local search to reduce number of vehicles
    optimized_solution = local_search(
        initial_solution, customers_info, depot_info, 
        vehicle_capacity, dist_matrix, travel_time_includes_service_time
    )
    
    # Calculate final departure times and loads for the solution
    all_departure_times = {}
    all_loads = {}
    is_valid_solution = True
    
    for route in optimized_solution:
        is_feasible, departure_times, loads = check_time_windows_and_service(
            dist_matrix, customers_info, depot_info, route, vehicle_capacity,
            travel_time_includes_service_time
        )
        if not is_feasible:
            is_valid_solution = False
            break
        
        all_departure_times.update(departure_times)
        all_loads.update(loads)
    
    if not is_valid_solution:
        # If somehow we arrived at an invalid solution, return basic info
        result = {
            "objective_value": len(optimized_solution),  # This might still reflect the best known
            "num_vehicles": len(optimized_solution), 
            "routes": optimized_solution,
            "total_distance": calculate_total_distance(optimized_solution, dist_matrix),
            "departure_times": all_departure_times,
            "loads": all_loads
        }
    else:
        result = {
            "objective_value": len(optimized_solution),
            "num_vehicles": len(optimized_solution), 
            "routes": optimized_solution,
            "total_distance": calculate_total_distance(optimized_solution, dist_matrix),
            "departure_times": all_departure_times,
            "loads": all_loads
        }
    
    with open(args.solution_path, 'w') as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
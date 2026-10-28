import argparse
import json
import math
from solution_logger import SolutionLogger


def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2)


def build_distance_matrix(instance):
    """Build a distance matrix for all locations (depots + customers)."""
    all_locations = instance['depots'] + instance['customers']
    n = len(all_locations)
    distances = [[0.0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            if i != j:
                distances[i][j] = calculate_distance(all_locations[i], all_locations[j])
    
    return distances


def solve_cluster_first_route_second(instance, time_limit):
    """
    Solve the Multi-Depot Periodic Vehicle Routing Problem using 
    cluster-first, route-second heuristic.
    """
    # Build distance matrix
    distances = build_distance_matrix(instance)
    
    # Prepare data structures
    n_customers = instance['n']
    n_depots = instance['d']
    n_periods = instance['t']
    n_vehicles_per_depot = instance['m']
    
    # Assign customers to depots using nearest neighbor
    assignments = {}
    for c_idx in range(n_customers):
        customer = instance['customers'][c_idx]
        min_dist = float('inf')
        closest_depot = 0
        
        for d_idx in range(n_depots):
            depot = instance['depots'][d_idx]
            dist = calculate_distance(customer, depot)
            if dist < min_dist:
                min_dist = dist
                closest_depot = d_idx
        
        # Select first available pattern for this customer
        selected_pattern = instance['customers'][c_idx]['pattern_list'][0]
        
        assignments[c_idx] = {
            'depot': closest_depot,
            'pattern': selected_pattern
        }
    
    # Group customers by depot and period
    depot_period_customers = {}
    for c_idx in range(n_customers):
        depot_id = assignments[c_idx]['depot']
        for period in assignments[c_idx]['pattern']:
            key = (depot_id, period)
            if key not in depot_period_customers:
                depot_period_customers[key] = []
            depot_period_customers[key].append(c_idx)
    
    # Solve VRP for each depot and period separately
    routes = []
    
    for (depot_id, period), customer_list in depot_period_customers.items():
        # Get depot index in the combined list (depots come first)
        depot_idx = depot_id
        
        # Create sub-problem for this depot and period
        sub_customers = [instance['customers'][c] for c in customer_list]
        
        # Apply Clarke-Wright savings heuristic for VRP
        current_routes = initialize_routes(depot_idx, customer_list, distances)
        current_routes = savings_algorithm(
            depot_idx, customer_list, instance, distances, current_routes
        )
        
        # Add to main routes list
        for v_idx, route in enumerate(current_routes):
            if route:  # Only add non-empty routes
                routes.append({
                    'depot': depot_id,
                    'period': period,
                    'vehicle': v_idx + 1,
                    'customers': route
                })
    
    # Calculate total cost
    total_cost = calculate_total_cost(routes, distances, instance['depots'])
    
    # Construct solution
    solution = {
        'objective_value': total_cost,
        'routes': routes,
        'assignments': assignments
    }
    
    return solution


def initialize_routes(depot_idx, customer_list, distances):
    """Initialize routes with one customer per route."""
    routes = []
    for c in customer_list:
        routes.append([c])
    return routes


def savings_algorithm(depot_idx, customer_list, instance, distances, initial_routes):
    """Apply the Clarke-Wright savings algorithm."""
    # Calculate savings
    savings = []
    n = len(customer_list)
    
    for i in range(n):
        for j in range(i + 1, n):
            c1 = customer_list[i]
            c2 = customer_list[j]
            
            # Savings = dist(0,c1) + dist(0,c2) - dist(c1,c2)
            saving = distances[depot_idx][c1] + distances[depot_idx][c2] - distances[c1][c2]
            savings.append((saving, i, j))
    
    # Sort savings in descending order
    savings.sort(reverse=True)
    
    # Initialize routes: each customer in its own route
    routes = {i: [customer_list[i]] for i in range(len(customer_list))}
    route_belongs_to = {customer_list[i]: i for i in range(len(customer_list))}  # customer -> route_id
    
    # Process savings
    for saving, i, j in savings:
        c1 = customer_list[i]
        c2 = customer_list[j]
        
        route1 = route_belongs_to[c1]
        route2 = route_belongs_to[c2]
        
        # Skip if already in same route
        if route1 == route2:
            continue
            
        # Check if merging violates constraints
        merged_route = merge_routes(routes[route1], routes[route2], c1, c2, depot_idx, distances, instance)
        
        if merged_route and is_feasible(merged_route, instance):
            # Perform the merge
            new_route_id = min(route1, route2)
            old_route_id = max(route1, route2)
            
            routes[new_route_id] = merged_route
            
            # Update route ownership for customers in old route
            for c in routes[old_route_id]:
                route_belongs_to[c] = new_route_id
                
            del routes[old_route_id]
    
    # Return all non-empty routes
    result = []
    for r in routes.values():
        if r:
            result.append(r)
    return result


def merge_routes(route1, route2, c1, c2, depot_idx, distances, instance):
    """Try to merge two routes by connecting the ends that include c1 and c2."""
    # Find positions of c1 and c2 in their respective routes
    pos1 = route1.index(c1)
    pos2 = route2.index(c2)
    
    # Possible ways to connect:
    # 1. End of route1 to start of route2
    # 2. Start of route1 to start of route2 (reversed)
    # 3. End of route1 to end of route2 (one reversed)
    # 4. Start of route1 to end of route2 (reversed)
    
    options = []
    
    # Option 1: route1 + route2
    option1 = route1 + route2
    if is_feasible(option1, instance):
        options.append(option1)
    
    # Option 2: reversed route1 + route2
    option2 = list(reversed(route1)) + route2
    if is_feasible(option2, instance):
        options.append(option2)
        
    # Option 3: route1 + reversed route2
    option3 = route1 + list(reversed(route2))
    if is_feasible(option3, instance):
        options.append(option3)
        
    # Option 4: reversed route2 + reversed route1 (equivalent to reversed(route1 + route2))
    option4 = list(reversed(route2)) + route1
    if is_feasible(option4, instance):
        options.append(option4)
    
    # Return the best option based on minimal additional distance
    if not options:
        return None
    
    best_option = min(options, key=lambda r: calculate_route_cost(r, depot_idx, distances))
    return best_option


def is_feasible(route, instance):
    """Check if a route is feasible regarding capacity and duration."""
    if not route:
        return True
    
    total_demand = sum(instance['customers'][c]['demand'] for c in route)
    if total_demand > instance['vehicle_capacity']:
        return False
    
    # Calculate route duration
    total_duration = 0
    prev_loc = 0  # Assuming depot is at index 0 initially, we'll fix this later
    
    # This function needs the actual depot index
    # We'll implement feasibility check differently
    depot_idx = 0  # Placeholder - this will be passed correctly
    total_duration = calculate_route_duration(route, depot_idx, instance)
    
    if total_duration > instance['route_duration_limit']:
        return False
    
    return True


def calculate_route_duration(route, depot_idx, instance, distances):
    """Calculate the total duration of a route."""
    if not route:
        return 0
    
    total_duration = 0
    current_pos = depot_idx
    
    for customer_idx in route:
        # Add travel time to next customer
        total_duration += distances[current_pos][customer_idx]
        # Add service time at customer
        total_duration += instance['customers'][customer_idx]['service_duration']
        current_pos = customer_idx
    
    # Add travel time back to depot
    total_duration += distances[current_pos][depot_idx]
    
    return total_duration


def calculate_route_cost(route, depot_idx, distances):
    """Calculate the total cost of a route."""
    if not route:
        return 0
    
    total_cost = 0
    current_pos = depot_idx
    
    for customer_idx in route:
        total_cost += distances[current_pos][customer_idx]
        current_pos = customer_idx
    
    # Add cost to return to depot
    total_cost += distances[current_pos][depot_idx]
    
    return total_cost


def calculate_total_cost(routes, distances, depots):
    """Calculate the total cost of all routes."""
    total_cost = 0
    
    for route_info in routes:
        depot_id = route_info['depot']
        route = route_info['customers']
        
        # Get depot index in the full distance matrix
        depot_idx = depot_id  # Since depots are listed first
        
        current_pos = depot_idx
        for customer_idx in route:
            # Adjust customer index since depots come first in the distance matrix
            adjusted_customer_idx = depots[depot_id]['id'] + len(depots)  # This is wrong logic
            # Actually, customer_idx refers to position in customer list, so need to adjust
            actual_customer_idx = len(depots) + customer_idx
            total_cost += distances[current_pos][actual_customer_idx]
            current_pos = actual_customer_idx
        
        # Return to depot
        total_cost += distances[current_pos][depot_idx]
    
    return total_cost


def improve_solution_with_local_search(instance, initial_solution, time_limit):
    """Apply local search improvements to the initial solution."""
    # For now, just return the initial solution
    # More sophisticated improvement could be added here
    return initial_solution


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    # Initialize logger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Read instance
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    # Solve the problem
    solution = solve_cluster_first_route_second(instance, args.time_limit)
    
    # Improve solution with local search
    solution = improve_solution_with_local_search(instance, solution, args.time_limit)

    # Log the final solution if logger is available
    if logger:
        logger.log_solution(solution['objective_value'], solution)

    # Write solution
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == '__main__':
    main()
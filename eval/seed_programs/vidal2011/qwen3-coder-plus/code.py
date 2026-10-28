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
        current_routes = initialize_routes(depot_idx, customer_list, distances, instance)
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
    total_cost = calculate_total_cost(routes, distances, instance)
    
    # Construct solution
    solution = {
        'objective_value': total_cost,
        'routes': routes,
        'assignments': assignments
    }
    
    return solution


def initialize_routes(depot_idx, customer_list, distances, instance):
    """Initialize routes with one customer per route."""
    routes = []
    for c in customer_list:
        # Check if a single customer route is feasible
        route = [c]
        if is_route_feasible(route, depot_idx, instance, distances):
            routes.append([c])
        else:
            # Handle infeasible single customer routes (shouldn't happen with proper validation)
            routes.append([c])  # Just add it anyway for now
    return routes


def is_route_feasible(route, depot_idx, instance, distances):
    """Check if a route is feasible regarding capacity and duration."""
    if not route:
        return True
    
    # Calculate total demand
    total_demand = sum(instance['customers'][c]['demand'] for c in route)
    if total_demand > instance['vehicle_capacity']:
        return False
    
    # Calculate route duration
    total_duration = 0
    current_pos = depot_idx
    
    for customer_idx in route:
        # Add travel time to next customer (adjusting for position in distance matrix)
        total_duration += distances[current_pos][len(instance['depots']) + customer_idx]
        # Add service time at customer
        total_duration += instance['customers'][customer_idx]['service_duration']
        current_pos = len(instance['depots']) + customer_idx
    
    # Add travel time back to depot
    total_duration += distances[current_pos][depot_idx]
    
    if total_duration > instance['route_duration_limit']:
        return False
    
    return True


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
            # Adjust for position in distance matrix
            adj_c1 = len(instance['depots']) + c1
            adj_c2 = len(instance['depots']) + c2
            saving = distances[depot_idx][adj_c1] + distances[depot_idx][adj_c2] - distances[adj_c1][adj_c2]
            savings.append((saving, i, j))
    
    # Sort savings in descending order
    savings.sort(reverse=True, key=lambda x: x[0])
    
    # Initialize routes: each customer in its own route
    routes = {}  # route_id -> [customer_indices]
    route_belongs_to = {}  # customer_idx -> route_id
    
    for idx, c in enumerate(customer_list):
        route_id = idx
        routes[route_id] = [c]
        route_belongs_to[c] = route_id
    
    # Process savings
    for saving, i, j in savings:
        c1 = customer_list[i]
        c2 = customer_list[j]
        
        route1 = route_belongs_to[c1]
        route2 = route_belongs_to[c2]
        
        # Skip if already in same route
        if route1 == route2:
            continue
            
        # Try merging the routes
        route1_customers = routes[route1]
        route2_customers = routes[route2]
        
        # Try all combinations of orientations
        candidates = [
            route1_customers + route2_customers,  # route1 + route2
            list(reversed(route1_customers)) + route2_customers,  # reversed(route1) + route2
            route1_customers + list(reversed(route2_customers)),  # route1 + reversed(route2)
            list(reversed(route2_customers)) + route1_customers,  # reversed(route2) + route1
        ]
        
        best_candidate = None
        for candidate in candidates:
            if is_route_feasible(candidate, depot_idx, instance, distances):
                best_candidate = candidate
                break  # Take first feasible one
        
        if best_candidate is not None:
            # Perform the merge
            new_route_id = min(route1, route2)
            old_route_id = max(route1, route2)
            
            routes[new_route_id] = best_candidate
            
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


def calculate_total_cost(routes, distances, instance):
    """Calculate the total cost of all routes."""
    total_cost = 0
    
    for route_info in routes:
        depot_id = route_info['depot']
        route = route_info['customers']
        
        # Get depot index in the full distance matrix
        depot_idx = depot_id  # Since depots are listed first in the distance matrix
        
        current_pos = depot_idx
        for customer_idx in route:
            # Adjust customer index since customers come after depots in the distance matrix
            actual_customer_idx = len(instance['depots']) + customer_idx
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
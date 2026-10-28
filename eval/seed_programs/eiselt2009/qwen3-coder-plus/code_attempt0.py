import argparse
import json
import math
import random
import itertools
from solution_logger import SolutionLogger


def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def compute_time_fare_and_utility(
    origin,
    dest,
    hub1,
    hub2,
    alpha,
    gamma_param,
    beta,
    lambda_param,
    cost_scalar,
    attractiveness_value,
    node_coords,
    waiting_times,
    distance_matrix,
    distance_to_time_conv=1.0,  # Assuming unit conversion factor for time/distance
):
    # Distances for the segments
    d_orig_hub1 = distance_matrix[origin][hub1]
    d_hub1_hub2 = distance_matrix[hub1][hub2]
    d_hub2_dest = distance_matrix[hub2][dest]

    # Time calculation
    time_orig_hub1 = d_orig_hub1 * distance_to_time_conv
    time_hub1_hub2 = d_hub1_hub2 * distance_to_time_conv
    time_hub2_dest = d_hub2_dest * distance_to_time_conv
    total_time = (time_orig_hub1 + time_hub1_hub2 + time_hub2_dest +
                  waiting_times[origin] + waiting_times[hub1] + waiting_times[hub2])

    # Fare calculation
    fare_orig_hub1 = d_orig_hub1 * cost_scalar
    fare_hub1_hub2 = alpha * d_hub1_hub2 * cost_scalar  # discounted interhub fare
    fare_hub2_dest = d_hub2_dest * cost_scalar
    total_fare = fare_orig_hub1 + fare_hub1_hub2 + fare_hub2_dest

    # Utility calculation
    denominator = (gamma_param * (total_time ** beta)) + ((1 - gamma_param) * (total_fare ** lambda_param))
    
    if denominator == 0:
        utility = float('inf')  # This should ideally be handled to avoid infinite values
    else:
        utility = attractiveness_value / denominator

    return total_time, total_fare, utility


def solve_airline_hub_location(instance_path, time_limit):
    with open(instance_path, 'r') as f:
        data = json.load(f)

    n_nodes = data['n_nodes']
    p = data['p']
    r = data['r']
    nodes = data['nodes']
    demand_matrix = data['demand_matrix']
    distance_matrix = data['distance_matrix']
    competitor_hub_locations = data['competitor_hub_locations']

    params = data['parameters']
    alpha = params['alpha']
    gamma_param = params['gamma']
    beta = params['beta']
    lambda_param = params['lambda']
    cost_scalar = params['cost_scalar']
    attractiveness_2hub = params['attractiveness_2hub']
    attractiveness_1hub = params['attractiveness_1hub']

    waiting_times = [node['waiting_time'] for node in nodes]
    node_coords = [(node['x'], node['y']) for node in nodes]

    # Precompute all relevant utility values for existing competitors
    # We'll store these in a dictionary for quick lookups
    comp_utilities_cache = {}

    def add_comp_utilities(src, dst):
        # Add single-hub routes for all competitor hubs
        for h in competitor_hub_locations:
            _, _, utility = compute_time_fare_and_utility(
                src, dst, h, h, alpha, gamma_param, beta, lambda_param,
                cost_scalar, attractiveness_1hub, node_coords, waiting_times,
                distance_matrix)
            comp_utilities_cache[(src, dst, h, h)] = utility

        # Add all two-hub routes from each competitor hub to every other competitor hub
        for h1_idx in range(len(competitor_hub_locations)):
            for h2_idx in range(h1_idx, len(competitor_hub_locations)):
                h1 = competitor_hub_locations[h1_idx]
                h2 = competitor_hub_locations[h2_idx]
                
                # To handle symmetric pairs and pick the optimal route order
                _, _, ut_first_order = compute_time_fare_and_utility(
                    src, dst, h1, h2, alpha, gamma_param, beta, lambda_param,
                    cost_scalar, attractiveness_2hub, node_coords, waiting_times,
                    distance_matrix)
                    
                if h1 != h2:
                    _, _, ut_second_order = compute_time_fare_and_utility(
                        src, dst, h2, h1, alpha, gamma_param, beta, lambda_param,
                        cost_scalar, attractiveness_2hub, node_coords, waiting_times,
                        distance_matrix)
                        
                    # Choose the max utility path for the competitor
                    utility = max(ut_first_order, ut_second_order)
                else:
                    utility = ut_first_order
                    
                comp_utilities_cache[(src, dst, h1, h2)] = utility
                comp_utilities_cache[(src, dst, h2, h1)] = utility

    # Pre-populate the competitor utility cache for all O-D pairs
    for i in range(n_nodes):
        for j in range(n_nodes):
            add_comp_utilities(i, j)

    # Objective function evaluation helper
    def evaluate_solution(candidate_hubs):
        total_passengers = 0.0
        
        candidate_hub_set = set(candidate_hubs)
        
        # Precompute utilities for all routes of the entrant airline
        entrant_single_utilities = {}
        entrant_double_utilities = {}
        
        for i in range(n_nodes):
            for j in range(n_nodes):
                total_demand = demand_matrix[i][j]
                if total_demand == 0:
                    continue
                
                # Collect utilities from competitors and entrant for this O-D pair
                total_route_utilities = []
                
                # Add competitor utilities
                for h1 in competitor_hub_locations:
                    for h2 in competitor_hub_locations:
                        # Only consider the upper triangular for symmetrical caching to avoid duplication
                        if h1 <= h2:
                            key = (i, j, h1, h2)
                            total_route_utilities.append(comp_utilities_cache[key])
                
                # Add entrant airline utilities
                # Single hub routes
                for h in candidate_hubs:
                    _, _, utility = compute_time_fare_and_utility(
                        i, j, h, h, alpha, gamma_param, beta, lambda_param,
                        cost_scalar, attractiveness_1hub, node_coords, waiting_times,
                        distance_matrix)
                    entrant_single_utilities[(i, j, h, h)] = utility
                    total_route_utilities.append(utility)
                
                # Two-hub routes
                for k in candidate_hubs:
                    for l in candidate_hubs:
                        if k != l:
                            # Calculate utility for i -> k -> l -> j and i -> l -> k -> j
                            _, _, u0 = compute_time_fare_and_utility(
                                i, j, k, l, alpha, gamma_param, beta, lambda_param,
                                cost_scalar, attractiveness_2hub, node_coords, waiting_times,
                                distance_matrix)

                            _, _, u1 = compute_time_fare_and_utility(
                                i, j, l, k, alpha, gamma_param, beta, lambda_param,
                                cost_scalar, attractiveness_2hub, node_coords, waiting_times,
                                distance_matrix)

                            max_utility = max(u0, u1)  # Take the best one between k->l and l->k orders
                            entrant_double_utilities[(i, j, k, l)] = max_utility
                            total_route_utilities.append(max_utility)

                if sum(total_route_utilities) > 0:
                    total_utility_sum = sum(total_route_utilities)
                    # Sum up only the utilities from routes the entrant offers
                    entrant_utils = []
                    for h in candidate_hubs:
                        entrant_utils.append(entrant_single_utilities.get((i, j, h, h), 0))
                    
                    for k in candidate_hubs:
                        for l in candidate_hubs:
                            if k != l:
                                # Ensure we don't double-count by taking both directions in our dictionary
                                # For consistency, add utilities using only the canonical ordering (k,l) with k<l
                                # But for simplicity, use precomputed cached value
                                max_utility = max(
                                    entrant_double_utilities.get((i, j, k, l), 0),
                                    entrant_double_utilities.get((i, j, l, k), 0)
                                )
                                entrant_utils.append(max_utility)

                    total_entrant_utility = sum(entrant_utils)
                    fraction_captured = total_entrant_utility / total_utility_sum
                    total_passengers += total_demand * fraction_captured

        return total_passengers

    # For problems of reasonable size, try all combinations
    # Otherwise use a greedy/local search approach
    best_objective = float('-inf')
    best_hubs = []

    # Try a greedy initialization
    hubs_so_far = []
    remaining_nodes = list(range(n_nodes))

    while len(hubs_so_far) < p:
        best_gain = float('-inf')
        best_next_hub = -1

        for node_candidate in remaining_nodes:
            temp_hubs = hubs_so_far + [node_candidate]
            obj_val = evaluate_solution(temp_hubs)
            gain = obj_val - best_objective if len(hubs_so_far) == 0 else obj_val - evaluate_solution(hubs_so_far)
            
            if gain > best_gain:
                best_gain = gain
                best_next_hub = node_candidate

        if best_next_hub != -1:
            hubs_so_far.append(best_next_hub)
            remaining_nodes.remove(best_next_hub)
            current_obj = evaluate_solution(hubs_so_far)
            if current_obj > best_objective:
                best_objective = current_obj
                best_hubs = list(hubs_so_far)

    # Now perform local search improvement (swap-based hill climbing)
    def get_neighbors(selected_hubs):
        neighbors = []
        non_hub_nodes = [n for n in range(n_nodes) if n not in selected_hubs]
        for hub_idx in range(len(selected_hubs)):
            for replace_node in non_hub_nodes:
                neighbor = list(selected_hubs)
                neighbor[hub_idx] = replace_node
                neighbors.append(neighbor)
        return neighbors

    # Local search loop
    max_iterations_no_improvement = 10  # To avoid getting stuck in long search loops
    consecutive_non_improving_steps = 0
    
    while consecutive_non_improving_steps < max_iterations_no_improvement:
        improved = False
        neighbors = get_neighbors(best_hubs)
        
        for neighbor_hubs in neighbors:
            obj_val = evaluate_solution(neighbor_hubs)
            if obj_val > best_objective:
                best_objective = obj_val
                best_hubs = neighbor_hubs
                improved = True
                consecutive_non_improving_steps = 0
                break  # Restart from new solution to avoid redundant calculations

        if not improved:
            consecutive_non_improving_steps += 1
            
    return best_objective, best_hubs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True, help="Path to the input instance JSON file")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to save the solution JSON file")
    parser.add_argument("--time_limit", type=int, required=True, help="Maximum runtime in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log file for intermediate solutions")

    args = parser.parse_args()

    # Initialize logger
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    try:
        # Solve the problem within the time limit
        objective_value, hub_locations = solve_airline_hub_location(args.instance_path, args.time_limit)

        # Log the best solution found
        if logger:
            logger.log(objective_value)

        # Write output
        solution = {
            "objective_value": objective_value,
            "hub_locations": hub_locations
        }

        with open(args.solution_path, 'w') as f:
            json.dump(solution, f)

    except Exception as e:
        print(f"Error occurred: {e}")
        raise


if __name__ == "__main__":
    main()
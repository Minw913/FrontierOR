import argparse
import json
import math
import time
import random
from solution_logger import SolutionLogger


def calculate_cost_matrix(node_coordinates):
    """Calculate cost matrix from coordinates using Euclidean distance."""
    n = len(node_coordinates)
    cost_matrix = [[0 for _ in range(n)] for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            if i != j:
                dx = node_coordinates[i][0] - node_coordinates[j][0]
                dy = node_coordinates[i][1] - node_coordinates[j][1]
                dist = math.sqrt(dx * dx + dy * dy)
                cost_matrix[i][j] = round(dist)
            else:
                cost_matrix[i][j] = 0
    
    return cost_matrix


def greedy_initial_solution(n, p, cost_matrix):
    """Create initial solution using greedy method"""
    # Start with random facilities
    facilities = list(range(n))
    random.shuffle(facilities)
    current_medians = set(facilities[:p])
    
    # Assign each customer to closest median
    assignments = [0] * n
    for i in range(n):
        min_cost = float('inf')
        closest_median = -1
        for j in range(n):
            if j in current_medians and cost_matrix[i][j] < min_cost:
                min_cost = cost_matrix[i][j]
                closest_median = j
        assignments[i] = closest_median
    
    # Calculate objective value
    total_cost = sum(cost_matrix[i][assignments[i]] for i in range(n))
    
    return list(current_medians), assignments, total_cost


def local_search_improvement(n, p, cost_matrix, max_iterations_without_improvement=100):
    """
    Local search improvement for the p-median problem.
    This explores swapping one non-facility for an existing facility.
    """
    # Start from a greedy initial solution
    current_medians, current_assignments, current_objective = greedy_initial_solution(n, p, cost_matrix)
    best_medians, best_assignments, best_objective = current_medians[:], current_assignments[:], current_objective
    
    iterations_since_improvement = 0
    iteration_limit = max_iterations_without_improvement
    
    while iterations_since_improvement < iteration_limit:
        improved = False
        
        # Try moving each non-facility to be a facility by replacing another
        for non_facility_id in range(n):
            if non_facility_id in current_medians:
                continue
                
            for original_facility_id in current_medians:
                # Try swapping original_facility_id with non_facility_id
                temp_medians = set(current_medians)
                temp_medians.remove(original_facility_id)
                temp_medians.add(non_facility_id)
                
                # Reassign all nodes to their closest temporary facility
                temp_assignments = [0] * n
                for i in range(n):
                    min_cost = float('inf')
                    closest_median = -1
                    for f in temp_medians:
                        if cost_matrix[i][f] < min_cost:
                            min_cost = cost_matrix[i][f]
                            closest_median = f
                    temp_assignments[i] = closest_median
                
                # Compute cost of this temporary configuration
                temp_obj = sum(cost_matrix[i][temp_assignments[i]] for i in range(n))
                
                # If this swap improves our solution, make the change
                if temp_obj < current_objective:
                    current_medians = list(temp_medians)
                    current_assignments = temp_assignments[:]
                    current_objective = temp_obj
                    
                    if current_objective < best_objective:
                        best_medians, best_assignments, best_objective = \
                            current_medians[:], current_assignments[:], current_objective
                        improved = True
                        iterations_since_improvement = 0
                        
                    break
            
            if current_objective < best_objective:
                break
                
        if not improved:
            iterations_since_improvement += 1
    
    return best_medians, best_assignments, best_objective


def solve_p_median_heuristic(n, p, cost_matrix, time_limit):
    """
    Solves the p-median problem heuristic with time limits.
    """
    start_time = time.time()
    
    best_medians, best_assignments, best_objective = local_search_improvement(n, p, cost_matrix)
    
    # Time-aware solution search
    remaining_time = time_limit - (time.time() - start_time)
    
    # Try multiple random restarts with remaining time
    iterations_since_best_improvement = 0
    while remaining_time > 1:  # At least 1 second remaining for meaningful computation
        try:
            medians, assignments, objective = local_search_improvement(n, p, cost_matrix)
            
            if objective < best_objective:
                best_objective = objective
                best_medians = medians[:]
                best_assignments = assignments[:]
                iterations_since_best_improvement = 0
            else:
                iterations_since_best_improvement += 1
                
            remaining_time = time_limit - (time.time() - start_time)
            
            # Early stopping if no improvements after several restarts
            if iterations_since_best_improvement > 5:
                break
        
        except:
            break  # In case of any errors just break
            
    return best_medians, best_assignments, best_objective


def solve_p_median(instance_path, solution_path, time_limit, log_path, seed=None):
    """
    Main function to solve the p-median problem.
    """
    if seed is not None:
        random.seed(seed)
    else:
        random.seed(42)  # Default seed
    
    # Load instance data
    with open(instance_path, 'r') as f:
        instance_data = json.load(f)
    
    n = instance_data['n']
    p = instance_data['p']
    node_coordinates = instance_data.get('node_coordinates', [])
    
    # Use provided cost matrix if available, otherwise compute from coordinates
    cost_matrix = instance_data.get('cost_matrix')
    if cost_matrix is None or len(cost_matrix) == 0:
        if node_coordinates:
            cost_matrix = calculate_cost_matrix(node_coordinates)
        else:
            raise ValueError("Neither cost_matrix nor node_coordinates provided in instance data")
    
    # Initialize logger
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    # Solve problem
    start_time = time.time()
    opened_facilities, assignments, objective_value = solve_p_median_heuristic(
        n, p, cost_matrix, time_limit
    )
    
    # Update assignments array to match final facilities
    final_assignments = []
    for i in range(n):
        min_cost = float('inf')
        closest_median = -1
        for j in opened_facilities:
            if cost_matrix[i][j] < min_cost:
                min_cost = cost_matrix[i][j]
                closest_median = j
        final_assignments.append({
            "customer": i,
            "facility": closest_median,
            "cost": cost_matrix[i][closest_median]
        })
    
    # Create solution dict
    solution = {
        "objective_value": objective_value,
        "opened_facilities": sorted(opened_facilities),
        "assignments": final_assignments
    }
    
    # Log the solution
    if logger:
        logger.log(objective_value)
    
    # Write solution to file
    with open(solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, required=False)
    
    args = parser.parse_args()
    
    solve_p_median(args.instance_path, args.solution_path, args.time_limit, args.log_path)
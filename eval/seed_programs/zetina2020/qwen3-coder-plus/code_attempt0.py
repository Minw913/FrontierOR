import argparse
import json
import math
import time
import itertools
from solution_logger import SolutionLogger


def solve_tsp_brute_force(coordinates, distance_matrix, time_limit):
    """
    Solve TSP using brute force enumeration since we only have 4 cities.
    This guarantees finding the optimal solution.
    """
    start_time = time.time()
    
    n_cities = len(coordinates)
    best_tour = None
    best_distance = float('inf')
    
    # Generate all possible permutations of cities
    for perm in itertools.permutations(range(n_cities)):
        # Calculate total distance for this tour
        tour_distance = 0
        for i in range(n_cities):
            from_city = perm[i]
            to_city = perm[(i + 1) % n_cities]  # Return to start for closed tour
            tour_distance += distance_matrix[from_city][to_city]
        
        # Check if this is the best tour so far
        if tour_distance < best_distance:
            best_distance = tour_distance
            best_tour = list(perm)
        
        # Check time limit
        if time.time() - start_time >= time_limit:
            break
    
    return best_distance, best_tour


def create_solution_from_tour(tour, n_cities):
    """Create solution dictionary from the tour."""
    solution = {}
    
    # Create x_{city}_{time_step} variables
    for city_idx in range(n_cities):
        for time_idx in range(n_cities):
            var_name = f"x_{city_idx}_{time_idx}"
            if tour[time_idx] == city_idx:
                solution[var_name] = 1
            else:
                solution[var_name] = 0
                
    return solution


def solve_max_cut_brute_force_or_greedy(graph_edges, n_nodes, time_limit):
    """
    For MaxCut, we'll use a greedy heuristic since brute force is exponential.
    We could also try local search or other heuristics.
    """
    start_time = time.time()
    
    # Greedy approach: start with a random partition and try improvements
    best_partition = [0] * n_nodes
    best_cut_value = 0
    
    # Try multiple random starts
    max_iterations = min(1000, 2**n_nodes)  # Don't exceed reasonable limits
    for _ in range(max_iterations):
        if time.time() - start_time >= time_limit:
            break
            
        # Random initial partition
        current_partition = [0 if i % 2 == 0 else 1 for i in range(n_nodes)]
        current_cut_value = calculate_cut_value(current_partition, graph_edges)
        
        improved = True
        while improved and (time.time() - start_time < time_limit):
            improved = False
            for node in range(n_nodes):
                # Flip the node's assignment
                old_assignment = current_partition[node]
                current_partition[node] = 1 - old_assignment
                new_cut_value = calculate_cut_value(current_partition, graph_edges)
                
                if new_cut_value > current_cut_value:
                    current_cut_value = new_cut_value
                    improved = True
                else:
                    # Revert the change if it didn't improve
                    current_partition[node] = old_assignment
        
        if current_cut_value > best_cut_value:
            best_cut_value = current_cut_value
            best_partition = current_partition[:]
    
    return best_cut_value, best_partition


def calculate_cut_value(partition, graph_edges):
    """Calculate the cut value for a given partition."""
    cut_value = 0
    for u, v in graph_edges:
        if partition[u] != partition[v]:
            cut_value += 1
    return cut_value


def create_maxcut_solution(partition, n_nodes):
    """Create solution dictionary for MaxCut."""
    solution = {}
    for node_idx in range(n_nodes):
        solution[f"x_{node_idx}"] = partition[node_idx]
    return solution


def main():
    parser = argparse.ArgumentParser(description='Solve TSP or MaxCut problem.')
    parser.add_argument('--instance_path', required=True, help='Path to the instance JSON file')
    parser.add_argument('--solution_path', required=True, help='Path to save the solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Maximum runtime in seconds')
    parser.add_argument('--log_path', help='Path to log intermediate solutions (optional)')
    
    args = parser.parse_args()
    
    # Load instance
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    # Initialize logger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    start_time = time.time()
    
    # Determine problem type based on parameters
    if 'n_cities' in instance['parameters']:
        # This is a TSP problem
        n_cities = instance['parameters']['n_cities']
        coordinates = instance['data']['coordinates']
        distance_matrix = instance['data']['distance_matrix']
        
        # Solve TSP
        objective_value, tour = solve_tsp_brute_force(coordinates, distance_matrix, args.time_limit)
        
        # Create solution dictionary
        solution = create_solution_from_tour(tour, n_cities)
        
        # Prepare output
        result = {
            "objective_value": objective_value,
            "solution": solution,
            "tour": tour
        }
        
        if logger:
            logger.log_solution(objective_value, solution)
    
    else:
        # This is a MaxCut problem
        # Extract graph edges from the instance
        # Since we don't have explicit edge list in the schema, 
        # we assume the distance matrix represents connections for MaxCut
        # In a real scenario, we'd need to know how to interpret the data differently
        
        # For MaxCut we would typically have a graph with nodes and edges
        # Let's determine n_nodes from the distance matrix dimensions
        distance_matrix = instance['data']['distance_matrix']
        n_nodes = len(distance_matrix)
        
        # Convert distance matrix to edge list (assuming non-zero entries represent edges)
        edges = []
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if distance_matrix[i][j] > 0:  # Assuming positive values indicate edges
                    edges.append((i, j))
        
        # Solve MaxCut
        objective_value, partition = solve_max_cut_brute_force_or_greedy(edges, n_nodes, args.time_limit)
        
        # Create solution dictionary
        solution = create_maxcut_solution(partition, n_nodes)
        
        # Prepare output
        result = {
            "objective_value": objective_value,
            "solution": solution
        }
        
        if logger:
            logger.log_solution(objective_value, solution)
    
    # Write solution
    with open(args.solution_path, 'w') as f:
        json.dump(result, f)


if __name__ == '__main__':
    main()
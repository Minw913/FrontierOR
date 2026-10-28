import argparse
import json
import math
import sys
import time
from collections import defaultdict
import itertools
import random
from solution_logger import SolutionLogger


def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        data = json.load(f)
    
    # Reconstruct full pairwise distances from upper triangular matrix
    n = data["problem_size"]["n"]
    dist_matrix = [[0] * n for _ in range(n)]
    
    # Fill the upper triangular part
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            dist = data["distance_matrix_upper_triangular"]["values"][idx]
            dist_matrix[i][j] = dist
            dist_matrix[j][i] = dist  # Symmetric
            idx += 1
    
    return {
        "n": n,
        "m": data["problem_size"]["m"],
        "nodes": data["nodes"],
        "clusters": data["clusters"],
        "dist_matrix": dist_matrix
    }


def get_distance(dist_matrix, node_id1, node_id2):
    return dist_matrix[node_id1][node_id2]


def solve_tsp_from_node_sequence(nodes_sequence, dist_matrix):
    """Given a sequence of node IDs, calculate the TSP path cost"""
    if not nodes_sequence:
        return 0, []
    
    total_cost = 0
    n = len(nodes_sequence)
    
    for i in range(n):
        from_node = nodes_sequence[i]
        to_node = nodes_sequence[(i + 1) % n]  # Wrap around to form cycle
        total_cost += dist_matrix[from_node][to_node]
    
    return total_cost, nodes_sequence


def tsp_heuristic(distances, nodes):
    """Simple greedy heuristic for TSP"""
    if not nodes:
        return [], 0
   
    start_node = nodes[0]
    unvisited = set(nodes[1:])
    current = start_node
    path = [current]
    total_cost = 0
    
    while unvisited:
        next_node = min(unvisited, 
                       key=lambda x: distances[current][x])
        total_cost += distances[current][next_node]
        current = next_node
        path.append(current)
        unvisited.remove(current)
    
    # Return to the start node
    total_cost += distances[current][start_node]
    
    return path, total_cost


def local_search_improve(initial_tour, dist_matrix, initial_cost):
    """Local search improvement using 2-opt exchanges"""
    tour = list(initial_tour)
    cost = initial_cost
    improved = True
    
    while improved:
        improved = False
        n = len(tour)
        
        for i in range(n):
            for j in range(i + 2, n):
                # Reverse path between i and j
                new_tour = tour[:]
                new_tour[i:j+1] = reversed(new_tour[i:j+1])
                
                # Calculate new cost
                new_cost = 0
                for k in range(len(new_tour)):
                    from_node = new_tour[k]
                    to_node = new_tour[(k + 1) % len(new_tour)]
                    new_cost += dist_matrix[from_node][to_node]
                
                if new_cost < cost:
                    tour = new_tour
                    cost = new_cost
                    improved = True
                    break
            if improved:
                break
                
    return tour, cost


def solve_clustered_tsp(instance_data, time_limit):
    start_time = time.time()
    
    n = instance_data["n"]
    m = instance_data["m"]
    nodes = instance_data["nodes"]
    clusters = instance_data["clusters"]
    dist_matrix = instance_data["dist_matrix"]
    
    # Group nodes by cluster
    cluster_to_nodes = {}
    for cluster_info in clusters:
        cluster_id = cluster_info["id"]
        node_ids = cluster_info["node_ids"]
        cluster_to_nodes[cluster_id] = node_ids
    
    # Find the best representatives for a given assignment
    def evaluate_representatives(rep_list):
        tour, base_cost = tsp_heuristic(dist_matrix, rep_list)
        improved_tour, improved_cost = local_search_improve(tour, dist_matrix, base_cost)
        return improved_tour, improved_cost
    
    best_overall_cost = float('inf')
    best_overall_tour = []
    best_visited_nodes = []
    
    # Random sampling and clustering-specific greedy improvement
    iterations = 800  # Adjust based on time limit
    iteration_time_budget = time_limit / iterations

    for iteration in range(iterations):
        if time.time() - start_time > time_limit * 0.95:  # 95% of time limit to be safe
            break
            
        # Randomly select one node from each cluster
        selected_nodes = []
        for cluster_id in sorted(cluster_to_nodes.keys()):
            node_id = random.choice(cluster_to_nodes[cluster_id])
            selected_nodes.append(node_id)
        
        # Improve with local search over selected nodes
        current_selected_nodes = list(selected_nodes)
        local_improved = True
        
        # Small local search over representative selection
        for rep_swap_iteration in range(max(3, 50 // m)):  # Adapt to problem size
            if time.time() - start_time > time_limit * 0.95:
                break
                
            # Try swapping one representative at random
            cluster_idx_to_change = random.randint(0, len(current_selected_nodes) - 1)
            old_representative = current_selected_nodes[cluster_idx_to_change]
            cluster_of_change = None
            
            # Find which cluster this representative belongs to
            for cluster_info in clusters:
                if old_representative in cluster_info["node_ids"]:
                    cluster_of_change = cluster_info["id"]
                    break
                    
            # Choose new representative from same cluster
            if cluster_of_change is not None:
                possible_new_reps = [nid for nid in cluster_to_nodes[cluster_of_change] 
                                     if nid != old_representative]
                
                if possible_new_reps:
                    new_rep = random.choice(possible_new_reps)
                    new_selected_nodes = list(current_selected_nodes)
                    new_selected_nodes[cluster_idx_to_change] = new_rep
                    
                    # Evaluate with new representatives
                    tmp_tour, tmp_cost = evaluate_representatives(new_selected_nodes)
                    
                    if tmp_cost < best_overall_cost:
                        best_overall_cost = tmp_cost
                        best_overall_tour = tmp_tour
                        best_visited_nodes = new_selected_nodes
                        
                        yield best_overall_tour, best_overall_cost, best_visited_nodes
        
        # Now apply TSP improvement on current selection
        current_tour, current_cost = evaluate_representatives(current_selected_nodes)
        
        if current_cost < best_overall_cost:
            best_overall_cost = current_cost
            best_overall_tour = current_tour
            best_visited_nodes = current_selected_nodes
            
            yield best_overall_tour, best_overall_cost, best_visited_nodes

    # Perform more focused optimization if there's still time
    if best_overall_tour:
        # Multiple random starts for TSP given best cluster representatives
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time > 1:
            # Do additional local searches with TSP improvements
            for _ in range(min(100, max(1, int(remaining_time * 5)))):
                if time.time() - start_time > time_limit * 0.95:
                    break
                trial_tour, trial_cost = local_search_improve(best_overall_tour, dist_matrix, best_overall_cost)
                
                if trial_cost < best_overall_cost:
                    best_overall_cost = trial_cost
                    best_overall_tour = trial_tour
                    
                    yield best_overall_tour, best_overall_cost, best_visited_nodes


def main():
    parser = argparse.ArgumentParser(description='Solve Clustered Traveling Salesman Problem')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to instance JSON file')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to output solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds')
    parser.add_argument('--log_path', type=str, help='Path to log file (optional)')
    args = parser.parse_args()

    # Initialize logger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Read the instance
    instance_data = read_instance(args.instance_path)
    
    best_tour = None
    best_cost = float('inf')
    best_visited = []

    # Solve with periodic updates and logging
    generator = solve_clustered_tsp(instance_data, args.time_limit)
    try:
        while True:
            tour, cost, visited_nodes = next(generator)
            if cost < best_cost:
                best_cost = cost
                best_tour = tour
                best_visited = visited_nodes
                
                if logger:
                    logger.log(cost)
    except StopIteration:
        pass

    # Prepare and save solution
    solution = {
        "objective_value": float(best_cost),
        "tour": best_tour,
        "visited_nodes": best_visited
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == '__main__':
    main()
import argparse
import json
import time
from itertools import combinations
from solution_logger import SolutionLogger


def solve_max_clique(instance_data, time_limit, logger=None):
    start_time = time.time()
    
    num_vertices = instance_data['graph']['num_vertices']
    adjacency_list = {int(k): v for k, v in instance_data['graph']['adjacency_list'].items()}

    # Convert adjacency lists to sets for faster lookup
    adj_sets = {v: set(neighbors) for v, neighbors in adjacency_list.items()}
    
    # Start with empty clique and try to build larger ones
    best_clique = []
    
    def is_clique(vertices):
        """Check if a set of vertices forms a clique"""
        for v1, v2 in combinations(vertices, 2):
            if v2 not in adj_sets[v1]:
                return False
        return True
    
    # Try expanding partial cliques
    # We start with single vertices and try to extend them
    vertices_sorted_by_degree = sorted(range(num_vertices), key=lambda x: len(adj_sets.get(x, [])), reverse=True)

    def branch_and_bound(current_clique, candidates):
        nonlocal best_clique
        
        current_time = time.time()
        if current_time - start_time >= time_limit:
            raise TimeoutError()
        
        # Update best solution if current is better
        if len(current_clique) > len(best_clique):
            best_clique = current_clique[:]
            if logger:
                logger.log(len(best_clique))
        
        # Prune if remaining candidates can't improve the solution
        if len(current_clique) + len(candidates) <= len(best_clique):
            return
            
        # Order candidates by degree to prioritize high-degree vertices first
        ordered_candidates = sorted(candidates, key=lambda x: len(adj_sets.get(x, [])), reverse=True)
        
        for v in ordered_candidates:
            # Check if adding v maintains clique property
            valid_extension = True
            for cv in current_clique:
                if v not in adj_sets[cv]:
                    valid_extension = False
                    break
            
            if valid_extension:
                current_clique.append(v)
                
                # New candidates are those in the intersection of current clique's neighborhood
                new_candidates = set(candidates)
                for cv in current_clique:
                    new_candidates &= adj_sets[cv]
                
                # Only consider candidates with higher indices to avoid duplicates
                new_candidates = [c for c in new_candidates if c > v]
                
                try:
                    branch_and_bound(current_clique, new_candidates)
                except TimeoutError:
                    return
                
                current_clique.pop()
    
    # Try growing cliques starting from different vertices
    for i in range(num_vertices):
        current_time = time.time()
        if current_time - start_time >= time_limit:
            break
            
        try:
            # Start with single vertex
            initial_clique = [i]
            candidates = [j for j in adj_sets[i] if j > i]  # Only higher index vertices to prevent redundancy
            
            # Try growing from this vertex
            branch_and_bound(initial_clique, candidates)
        except TimeoutError:
            break

    return best_clique


def main():
    parser = argparse.ArgumentParser(description='Solve the Maximum Clique Problem')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the input JSON instance file')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to output the solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds')
    parser.add_argument('--log_path', type=str, default=None, help='Path to the log file for intermediate solutions')

    args = parser.parse_args()

    # Load the instance data
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)

    # Initialize logger
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    # Solve the maximum clique problem
    best_clique = solve_max_clique(instance_data, args.time_limit, logger)

    # Prepare the solution in required format
    solution = {
        "objective_value": len(best_clique),
        "clique_vertices": best_clique
    }

    # Write the solution to output file
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == '__main__':
    main()
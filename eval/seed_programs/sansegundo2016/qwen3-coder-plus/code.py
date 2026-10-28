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
    adj_sets = {}
    for v in range(num_vertices):
        adj_sets[v] = set(adjacency_list.get(str(v), []))
    
    # Start with empty clique and try to build larger ones
    best_clique = []
    
    def is_clique(vertices):
        """Check if a set of vertices forms a clique"""
        for v1, v2 in combinations(vertices, 2):
            if v2 not in adj_sets.get(v1, set()):
                return False
        return True
    
    # Recursive function to find max clique using branch and bound
    def branch_and_bound(current_clique, candidates):
        nonlocal best_clique
        
        current_time = time.time()
        if current_time - start_time >= time_limit:
            return
        
        # Update best solution if current is better
        if len(current_clique) > len(best_clique):
            best_clique = current_clique[:]
            if logger:
                logger.log(len(best_clique))
        
        # Prune if remaining candidates can't improve the solution
        if len(current_clique) + len(candidates) <= len(best_clique):
            return
            
        # Order candidates by connectivity to reduce search space
        candidates = [v for v in candidates if v not in current_clique]
        
        # Heuristic: process candidates with higher connectivity first
        candidates = sorted(candidates, 
                            key=lambda x: len(adj_sets.get(x, set()) & set(current_clique)), 
                            reverse=True)
        
        for v in candidates:
            # Check if all vertices in current clique have v as neighbor
            valid = all(v in adj_sets[cv] for cv in current_clique)
            
            if valid and len(current_clique) + 1 > len(best_clique):
                current_clique.append(v)
                
                # Candidates for next iteration are neighbors of v that are also in current candidates
                new_candidates = []
                for c in candidates:
                    if c != v and v in adj_sets.get(c, set()):
                        new_candidates.append(c)
                
                branch_and_bound(current_clique, new_candidates)
                
                current_clique.pop()
    
    # Try starting from every vertex to ensure we don't miss better solutions
    visited = [False] * num_vertices 
    for start_vertex in range(num_vertices):
        if visited[start_vertex]:
            continue
        
        current_clique = [start_vertex]
        # Get candidate vertices that are connected to start_vertex
        potential_candidates = list(adj_sets.get(start_vertex, set()))
        
        try:
            branch_and_bound(current_clique, potential_candidates)
        except Exception:
            # In case we run into recursion issue or timeout errors
            pass
        
        # Mark vertices as visited so we can skip potentially redundant searches
        # (This is more for symmetry but we'll process all anyway)
    
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
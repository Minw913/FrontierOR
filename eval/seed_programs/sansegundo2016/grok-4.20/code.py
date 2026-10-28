import json
import argparse
import time
import sys
from solution_logger import SolutionLogger

def find_max_clique(n, edges, time_limit, logger):
    # Simple greedy + local search heuristic (no external solvers)
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Greedy with random restarts
    best_clique = []
    best_size = 0
    start_time = time.time()
    
    # Initial greedy by degree
    degrees = sorted(range(n), key=lambda v: len(adj[v]), reverse=True)
    clique = []
    candidates = set(range(n))
    for v in degrees:
        if v not in candidates:
            continue
        if all(u in adj[v] for u in clique):
            clique.append(v)
            candidates &= adj[v]
    if len(clique) > best_size:
        best_size = len(clique)
        best_clique = clique[:]
        if logger:
            logger.log(best_size)
    
    # Local search with multiple restarts
    max_restarts = 30
    for r in range(max_restarts):
        if time.time() - start_time > time_limit * 0.95:
            break
        
        # Randomized greedy
        clique = []
        candidates = set(range(n))
        order = list(range(n))
        # Bias toward high degree but randomize
        order.sort(key=lambda v: len(adj[v]) + (hash(v + r) % 5), reverse=True)
        
        for v in order:
            if time.time() - start_time > time_limit * 0.95:
                break
            if v not in candidates:
                continue
            if all(u in adj[v] for u in clique):
                clique.append(v)
                candidates.intersection_update(adj[v])
        
        if len(clique) > best_size:
            best_size = len(clique)
            best_clique = clique[:]
            if logger:
                logger.log(best_size)
        
        # Local improvement: try to add vertices and remove conflicts
        improved = True
        while improved and (time.time() - start_time < time_limit * 0.95):
            improved = False
            # Try to expand
            for v in sorted(candidates, key=lambda x: -len(adj[x])):
                if all(u in adj[v] for u in clique):
                    clique.append(v)
                    candidates.intersection_update(adj[v])
                    best_size = len(clique)
                    best_clique = clique[:]
                    if logger:
                        logger.log(best_size)
                    improved = True
                    break
            
            if not improved and len(clique) > 3:
                # Try swapping: remove one, add two
                for i in range(len(clique)):
                    removed = clique[i]
                    temp_clique = clique[:i] + clique[i+1:]
                    new_candidates = set(range(n))
                    for u in temp_clique:
                        new_candidates &= adj[u]
                    added = []
                    for v in sorted(new_candidates, key=lambda x: -len(adj[x])):
                        if all(u in adj[v] for u in temp_clique):
                            added.append(v)
                            temp_clique.append(v)
                            new_candidates &= adj[v]
                            if len(added) >= 2:
                                break
                    if len(added) >= 2:
                        clique = temp_clique
                        if len(clique) > best_size:
                            best_size = len(clique)
                            best_clique = clique[:]
                            if logger:
                                logger.log(best_size)
                        improved = True
                        break
                    if time.time() - start_time > time_limit * 0.9:
                        break
                if improved:
                    continue
    
    return best_size, sorted(best_clique)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    # Load instance
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    graph = instance['graph']
    n = graph['num_vertices']
    adjacency = graph['adjacency_list']
    
    # Build edge set (undirected, no duplicates)
    edges = set()
    for u_str, neighbors in adjacency.items():
        u = int(u_str)
        for v_str in neighbors:
            v = int(v_str)
            if u < v:
                edges.add((u, v))
    
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None
    
    # Solve with heuristic
    objective, clique = find_max_clique(n, edges, args.time_limit, logger)
    
    # Write solution
    solution = {
        "objective_value": int(objective),
        "clique_vertices": clique
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()
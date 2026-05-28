import json
import argparse
import time
import sys
from solution_logger import SolutionLogger

def bron_kerbosch(R, P, X, adj, best):
    if not P and not X:
        if len(R) > len(best[0]):
            best[0] = R[:]
            return True
        return False
    improved = False
    for v in list(P):
        if len(R) + len(P) <= len(best[0]):
            break
        Nv = adj[v]
        new_R = R + [v]
        new_P = [u for u in P if u in Nv]
        new_X = [u for u in X if u in Nv]
        if bron_kerbosch(new_R, new_P, new_X, adj, best):
            improved = True
        P.remove(v)
        X.append(v)
    return improved

def find_max_clique(n, adj, time_limit):
    best = [[]]
    start_time = time.time()
    
    # Simple degeneracy ordering for pivot selection
    degrees = [len(adj[i]) for i in range(n)]
    order = sorted(range(n), key=lambda x: degrees[x])
    
    # Run BK with basic pivoting
    P = list(order)
    improved = bron_kerbosch([], P, [], adj, best)
    
    # If time allows, try restarting from different seeds for larger graphs
    elapsed = time.time() - start_time
    if elapsed < time_limit * 0.6:
        for seed in range(min(5, n)):
            if time.time() - start_time > time_limit:
                break
            R = [seed]
            P = [u for u in range(n) if u != seed and u in adj[seed]]
            X = []
            bron_kerbosch(R, P, X, adj, best)
    
    return len(best[0]), sorted(best[0])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    n = instance['n']
    edges = instance.get('edges', [])
    
    # Build adjacency lists
    adj = [set() for _ in range(n)]
    for u, v in edges:
        if u != v:
            adj[u].add(v)
            adj[v].add(u)
    
    # Convert to lists for BK
    adj_list = [list(s) for s in adj]
    
    # Run exact branch-and-bound clique finder with time awareness
    clique_size, clique_vertices = find_max_clique(n, adj_list, args.time_limit)
    
    if logger and clique_size > 0:
        logger.log(clique_size)
    
    solution = {
        "objective_value": clique_size,
        "clique_vertices": clique_vertices
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()
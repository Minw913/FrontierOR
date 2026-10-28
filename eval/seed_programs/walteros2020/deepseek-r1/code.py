import json
import argparse
import time
import sys
from solution_logger import SolutionLogger

sys.setrecursionlimit(100000)

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
    edges = instance['edges']
    omega = instance.get('omega', 0)
    
    adj = [set() for _ in range(n)]
    for edge in edges:
        u, v = edge
        adj[u].add(v)
        adj[v].add(u)
    
    best_solution = set()
    best_size = 0
    
    def greedy_clique_heuristic():
        clique = []
        candidates = set(range(n))
        while candidates:
            v = max(candidates, key=lambda x: len(adj[x] & candidates))
            clique.append(v)
            candidates = candidates & adj[v]
        return clique
    
    greedy_clique = greedy_clique_heuristic()
    greedy_size = len(greedy_clique)
    
    if omega > greedy_size:
        found_clique = None
        vertices_ordered = sorted(range(n), key=lambda v: len(adj[v]), reverse=True)
        for v in vertices_ordered:
            clique = [v]
            candidates = adj[v].copy()
            while len(clique) < omega and candidates:
                u = max(candidates, key=lambda x: len(adj[x] & candidates))
                clique.append(u)
                candidates = candidates & adj[u]
            if len(clique) == omega:
                found_clique = clique
                break
        if found_clique is not None:
            best_solution = set(found_clique)
            best_size = omega
        else:
            best_solution = set(greedy_clique)
            best_size = greedy_size
    else:
        best_solution = set(greedy_clique)
        best_size = greedy_size
    
    if logger:
        logger.log(best_size)
    
    start_time = time.time()
    
    if best_size == n:
        solution = {
            "objective_value": best_size,
            "clique_vertices": sorted(list(best_solution))
        }
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f)
        return
    
    def backtrack(R, P, X):
        nonlocal best_solution, best_size
        current_time = time.time()
        if current_time - start_time > args.time_limit:
            return
        
        if not P and not X:
            if len(R) > best_size:
                best_solution = set(R)
                best_size = len(R)
                if logger:
                    logger.log(best_size)
            return
        
        max_deg_in_P = 0
        deg_in_P = {}
        for v in P:
            deg = len(adj[v] & P)
            deg_in_P[v] = deg
            if deg > max_deg_in_P:
                max_deg_in_P = deg
        
        if len(R) + max_deg_in_P + 1 <= best_size:
            return
        
        best_pivot = None
        best_count = -1
        for u in P | X:
            count = len(adj[u] & P)
            if count > best_count:
                best_count = count
                best_pivot = u
        
        if best_pivot is None:
            return
        
        branch_set = P - adj[best_pivot]
        branch_ordered = sorted(branch_set, key=lambda v: deg_in_P.get(v, 0), reverse=True)
        
        for v in branch_ordered:
            new_P = P & adj[v]
            new_X = X & adj[v]
            backtrack(R | {v}, new_P, new_X)
            P = P - {v}
            X = X | {v}
    
    backtrack(set(), set(range(n)), set())
    
    solution = {
        "objective_value": best_size,
        "clique_vertices": sorted(list(best_solution))
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == '__main__':
    main()
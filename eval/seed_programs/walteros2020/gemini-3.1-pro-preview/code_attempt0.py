import argparse
import json
import time
import random
import sys
import numpy as np
import scipy.sparse as sp
import gurobipy as gp

from solution_logger import SolutionLogger

def find_heuristic_clique(n, adj_set, adjList, time_limit, logger):
    start_time = time.time()
    
    deg = [len(adj_set[i]) for i in range(n)]
    vertices = list(range(n))
    vertices.sort(key=lambda x: deg[x], reverse=True)
    
    best_clq = []
    best_size = 0
    
    # Pass 1: Deterministic greedy from top vertices
    for start_node in vertices[:min(200, n)]:
        clq = [start_node]
        candidates = set(adj_set[start_node])
        while candidates:
            # Pick candidate with most connections into current candidates
            best_c = max(candidates, key=lambda c: len(candidates.intersection(adj_set[c])))
            clq.append(best_c)
            candidates.intersection_update(adj_set[best_c])
            
        if len(clq) > best_size:
            best_size = len(clq)
            best_clq = clq
            if logger:
                logger.log(best_size)
                
    # Pass 2: Randomized greedy to explore more diverse cliques
    if n > 0:
        pool_size = max(1, n // 5)
        while time.time() - start_time < time_limit:
            start_node = random.choice(vertices[:pool_size])
            clq = [start_node]
            candidates = set(adj_set[start_node])
            while candidates:
                if random.random() < 0.2:
                    best_c = random.choice(tuple(candidates))
                else:
                    sample = random.sample(tuple(candidates), min(10, len(candidates)))
                    best_c = max(sample, key=lambda c: len(candidates.intersection(adj_set[c])))
                clq.append(best_c)
                candidates.intersection_update(adj_set[best_c])
                
            if len(clq) > best_size:
                best_size = len(clq)
                best_clq = clq
                if logger:
                    logger.log(best_size)
                    
    return best_clq

def advanced_prune(n, adjList, K):
    if K <= 1:
        return list(range(n))
        
    active = [True] * n
    deg = [len(adjList[i]) for i in range(n)]
    
    queue = [i for i in range(n) if deg[i] < K - 1]
    
    def run_queue():
        while queue:
            u = queue.pop()
            if not active[u]:
                continue
            active[u] = False
            for v in adjList[u]:
                if active[v]:
                    deg[v] -= 1
                    if deg[v] == K - 2:
                        queue.append(v)
                        
    run_queue()
    
    changed = True
    while changed:
        changed = False
        valid_nodes = [i for i in range(n) if active[i]]
        # Process from lowest degree to highest for maximal early pruning
        valid_nodes.sort(key=lambda x: deg[x])
        
        for v in valid_nodes:
            if not active[v]: 
                continue
            
            if deg[v] < K - 1:
                queue.append(v)
                run_queue()
                changed = True
                continue
                
            active_neighbors = [u for u in adjList[v] if active[u]]
            # Sorthing by available active degree aids quick hitting the max_color bound
            active_neighbors.sort(key=lambda x: deg[x], reverse=True)
            
            color_assignment = {}
            max_color = 0
            for u in active_neighbors:
                used = {color_assignment[w] for w in adjList[u] if w in color_assignment}
                c = 1
                while c in used:
                    c += 1
                color_assignment[u] = c
                if c > max_color:
                    max_color = c
                    # If we use at least K colors for active_neighbors, bound exceeds K. Break early!
                    if max_color >= K:
                        break
                        
            bound = max_color + 1
            if bound <= K:
                queue.append(v)
                run_queue()
                changed = True
                
    return [i for i in range(n) if active[i]]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    proc_start_time = time.time()
    
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None
    
    with open(args.instance_path, 'r') as f:
        data = json.load(f)
        
    n = data['n']
    edges = data['edges']
    
    adjList = [[] for _ in range(n)]
    for u, v in edges:
        adjList[u].append(v)
        adjList[v].append(u)
        
    adj_set = [set(neighbors) for neighbors in adjList]
    
    # 1. Randomized greedy heuristic bound
    heuristic_time = min(2.0, args.time_limit * 0.05)
    best_clq = find_heuristic_clique(n, adj_set, adjList, heuristic_time, logger)
    best_solution = best_clq
    K = len(best_clq)
    
    # 2. Deep Core & Graph Coloring Pruning
    valid_nodes = advanced_prune(n, adjList, K)
    
    # 3. Formulate Reduced Problem in Gurobi
    time_left = args.time_limit - (time.time() - proc_start_time)
    if time_left > 0 and len(valid_nodes) > 0:
        n_prime = len(valid_nodes)
        node_to_idx = {v: i for i, v in enumerate(valid_nodes)}
        idx_to_node = {i: v for i, v in enumerate(valid_nodes)}
        
        # Build logical adjacency matrix to flag entirely missing edges efficiently
        adj_matrix = np.ones((n_prime, n_prime), dtype=bool)
        np.fill_diagonal(adj_matrix, False)
        
        for u in valid_nodes:
            u_idx = node_to_idx[u]
            for v in adjList[u]:
                if v in node_to_idx:
                    v_idx = node_to_idx[v]
                    adj_matrix[u_idx, v_idx] = False
                    
        u_idx, v_idx = np.nonzero(adj_matrix)
        mask = u_idx < v_idx
        u_idx = u_idx[mask]
        v_idx = v_idx[mask]
        num_non_edges = len(u_idx)
        
        A = None
        if num_non_edges > 0:
            row_idx = np.repeat(np.arange(num_non_edges), 2)
            col_idx = np.empty(2 * num_non_edges, dtype=np.int32)
            col_idx[0::2] = u_idx
            col_idx[1::2] = v_idx
            data_arr = np.ones(2 * num_non_edges, dtype=np.float64)
            A = sp.csr_matrix((data_arr, (row_idx, col_idx)), shape=(num_non_edges, n_prime))
            
        model = gp.Model("max_clique")
        model.Params.OutputFlag = 0
        model.Params.Threads = 1
        model.Params.TimeLimit = max(1.0, args.time_limit - (time.time() - proc_start_time))
        model.Params.Cutoff = K + 0.5  # Instantly limits IP to solutions strictly better than best_clq
        model.Params.CliqueCuts = 2
        model.Params.Presolve = 2
        
        x = model.addMVar(n_prime, vtype=gp.GRB.BINARY, name="x")
        model.setObjective(x.sum(), gp.GRB.MAXIMIZE)
        
        if A is not None:
            b = np.ones(num_non_edges, dtype=np.float64)
            model.addConstr(A @ x <= b, name="non_edge")
            
        def gurobi_callback(m, where):
            if where == gp.GRB.Callback.MIPSOL:
                obj = m.cbGet(gp.GRB.Callback.MIPSOL_OBJ)
                if logger:
                    logger.log(int(obj + 0.5))
                    
        model.optimize(gurobi_callback)
        
        if model.SolCount > 0:
            x_val = x.X
            selected_indices = np.where(x_val > 0.5)[0]
            if len(selected_indices) > K:
                best_solution = [int(idx_to_node[i]) for i in selected_indices]
                
    output_data = {
        "objective_value": len(best_solution),
        "clique_vertices": [int(v) for v in best_solution]
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(output_data, f, indent=4)

if __name__ == "__main__":
    main()
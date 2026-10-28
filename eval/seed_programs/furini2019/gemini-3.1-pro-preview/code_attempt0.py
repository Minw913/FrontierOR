import argparse
import json
import time
import sys
import random
import numpy as np
import scipy.sparse as sp
import gurobipy as gp
from gurobipy import GRB

def main():
    start_total = time.time()
    
    parser = argparse.ArgumentParser(description="Solve Maximum Clique Problem")
    parser.add_argument('--instance_path', type=str, required=True, help="Path to instance JSON")
    parser.add_argument('--solution_path', type=str, required=True, help="Path to output solution JSON")
    parser.add_argument('--time_limit', type=int, required=True, help="Maximum runtime in seconds")
    parser.add_argument('--log_path', type=str, default=None, help="Path to log intermediate solutions")
    args = parser.parse_args()

    logger = None
    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="maximize")

    # Load instance
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    num_vertices = instance['num_vertices']
    num_edges = instance['num_edges']
    adjacency_list = instance.get('adjacency_list', [])

    if num_vertices == 0:
        result = {"objective_value": 0, "clique_vertices": []}
        with open(args.solution_path, 'w') as f:
            json.dump(result, f, indent=2)
        if logger:
            logger.log(0)
        return

    # Convert adjacency list to sets for fast checks
    adj_set = [set(neighbors) for neighbors in adjacency_list]
    degrees = [len(s) for s in adj_set]
    vertices = list(range(num_vertices))
    
    # Pre-calculated sorting order for deterministic greedy
    vertices_sorted = sorted(vertices, key=lambda v: degrees[v], reverse=True)

    # Ultra-fast greedy heuristic
    def greedy_clique(vertices_order):
        clique = []
        allowed = set(range(num_vertices))
        for v in vertices_order:
            if v in allowed:
                clique.append(v)
                allowed.intersection_update(adj_set[v])
                if not allowed:
                    break
        return clique

    # Initial deterministic run
    best_clique = greedy_clique(vertices_sorted)
    best_obj = len(best_clique)
    if logger:
        logger.log(best_obj)

    # Fast initial heuristic phase (up to 1.5 seconds)
    start_heur = time.time()
    iters = 0
    while time.time() - start_heur < 1.5 and iters < 2000:
        iters += 1
        if iters % 2 == 0:
            # Vary initial vertex selection based on randomly perturbed degrees
            perturbed = {v: degrees[v] * random.uniform(0.5, 1.5) for v in range(num_vertices)}
            random_order = sorted(vertices, key=lambda v: perturbed[v], reverse=True)
        else:
            # Completely random orders
            random_order = list(vertices)
            random.shuffle(random_order)
            
        c = greedy_clique(random_order)
        if len(c) > best_obj:
            best_clique = c
            best_obj = len(c)
            if logger:
                logger.log(best_obj)

    M_expected = num_vertices * (num_vertices - 1) // 2 - num_edges
    MAX_M = 25_000_000
    time_left = args.time_limit - (time.time() - start_total)

    # Use exact MIP, unless it's a huge sparse topology exceeding threshold or too little time left
    if M_expected > MAX_M or time_left <= 2.0:
        # Failsafe: Run extended heuristic to leverage maximum allowable optimization time
        end_time = start_total + args.time_limit - 1.0
        while time.time() < end_time:
            iters += 1
            if iters % 10 < 5:
                perturbed = {v: degrees[v] * random.uniform(0.2, 1.8) for v in range(num_vertices)}
                random_order = sorted(vertices, key=lambda v: perturbed[v], reverse=True)
            else:
                random_order = list(vertices)
                random.shuffle(random_order)
                
            c = greedy_clique(random_order)
            if len(c) > best_obj:
                best_clique = c
                best_obj = len(c)
                if logger:
                    logger.log(best_obj)
        
        final_obj = best_obj
        clique_vertices = best_clique

    else:
        # Build memory-efficient Gurobi Matrix constraint execution
        model = gp.Model("MaxClique")
        model.Params.TimeLimit = max(1.0, args.time_limit - (time.time() - start_total))
        model.Params.Threads = 1  # Ensures compliance with single-CPU restriction
        model.Params.OutputFlag = 0  # Prevents polluting terminal output buffer
        
        x = model.addMVar(num_vertices, vtype=GRB.BINARY, name="x")
        
        if num_vertices <= 25000:
            adj_matrix = np.zeros((num_vertices, num_vertices), dtype=bool)
            for i in range(num_vertices):
                if adjacency_list[i]:
                    adj_matrix[i, adjacency_list[i]] = True
                    
            np.logical_not(adj_matrix, out=adj_matrix)
            row, col = np.where(np.triu(adj_matrix, k=1))
            M = len(row)
            del adj_matrix
        else:
            row_list = []
            col_list = []
            for i in range(num_vertices):
                neighbors = adj_set[i]
                for j in range(i + 1, num_vertices):
                    if j not in neighbors:
                        row_list.append(i)
                        col_list.append(j)
            row = np.array(row_list, dtype=np.int32)
            col = np.array(col_list, dtype=np.int32)
            M = len(row)

        if M > 0:
            data = np.ones(2 * M, dtype=np.float64)
            rows = np.repeat(np.arange(M), 2)
            cols = np.empty(2 * M, dtype=np.int32)
            cols[0::2] = row
            cols[1::2] = col
            A = sp.csr_matrix((data, (rows, cols)), shape=(M, num_vertices))
            rhs = np.ones(M)
            model.addConstr(A @ x <= rhs, name="indep_set")

        model.setObjective(x.sum(), GRB.MAXIMIZE)

        # Inject robust warm-start solution
        mip_start = np.zeros(num_vertices)
        for v in best_clique:
            mip_start[v] = 1
        x.Start = mip_start

        # Environment proxy attachments for callback access limits
        model._logger = logger
        model._best_obj = best_obj

        def cb(mdl, where):
            if where == GRB.Callback.MIPSOL:
                try:
                    obj = mdl.cbGet(GRB.Callback.MIPSOL_OBJ)
                    obj_int = int(round(obj))
                    if obj_int > mdl._best_obj:
                        mdl._best_obj = obj_int
                        if mdl._logger:
                            mdl._logger.log(obj_int)
                except Exception:
                    pass

        # Trigger model solve process
        model.optimize(cb)

        # Final extraction
        if model.SolCount > 0:
            sol = x.X
            gurobi_clique = [i for i in range(num_vertices) if sol[i] > 0.5]
            if len(gurobi_clique) > best_obj:
                clique_vertices = gurobi_clique
                final_obj = len(gurobi_clique)
            else:
                clique_vertices = best_clique
                final_obj = best_obj
        else:
            clique_vertices = best_clique
            final_obj = best_obj

    # Conclude with perfectly conformed JSON document
    result = {
        "objective_value": final_obj,
        "clique_vertices": clique_vertices
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(result, f, indent=2)

if __name__ == '__main__':
    main()
import json
import argparse
import time
import itertools
import sys
import os
import random
import numpy as np

def solve_tsp(data, params, time_limit, logger):
    """Solve TSP by enumerating all permutations (feasible for small n)."""
    n = params['n_cities']
    dist_matrix = data['distance_matrix']
    
    best_cost = float('inf')
    best_tour = None
    
    # For n=4, enumerate all (n-1)! permutations fixing city 0 as start
    cities = list(range(1, n))
    
    for perm in itertools.permutations(cities):
        tour = [0] + list(perm)
        cost = 0.0
        for i in range(n):
            cost += dist_matrix[tour[i]][tour[(i + 1) % n]]
        
        if cost < best_cost:
            best_cost = cost
            best_tour = tour[:]
            
            # Build solution dict
            solution = {}
            for city in range(n):
                for t in range(n):
                    key = f"x_{city}_{t}"
                    solution[key] = 1 if best_tour[t] == city else 0
            
            if logger:
                logger.log_solution(best_cost, {
                    "objective_value": best_cost,
                    "solution": solution,
                    "tour": best_tour
                })
    
    # Build final solution
    solution = {}
    for city in range(n):
        for t in range(n):
            key = f"x_{city}_{t}"
            solution[key] = 1 if best_tour[t] == city else 0
    
    return {
        "objective_value": best_cost,
        "solution": solution,
        "tour": best_tour
    }


def local_search_maxcut(n_nodes, edges, initial_assignment=None, max_iter=1000):
    """Greedy local search for MaxCut: repeatedly flip the node that gives max improvement."""
    # Build adjacency list with weights
    adj = [[] for _ in range(n_nodes)]
    for u, v, w in edges:
        adj[u].append((v, w))
        adj[v].append((u, w))
    
    if initial_assignment is None:
        assignment = [random.randint(0, 1) for _ in range(n_nodes)]
    else:
        assignment = initial_assignment[:]
    
    def compute_cut_value():
        val = 0
        for u, v, w in edges:
            if assignment[u] != assignment[v]:
                val += w
        return val
    
    current_value = compute_cut_value()
    
    for iteration in range(max_iter):
        best_gain = 0
        best_node = -1
        
        for node in range(n_nodes):
            gain = 0
            for neighbor, w in adj[node]:
                if assignment[node] != assignment[neighbor]:
                    gain -= w  # currently cut, will become uncut
                else:
                    gain += w  # currently uncut, will become cut
            
            if gain > best_gain:
                best_gain = gain
                best_node = node
        
        if best_node == -1 or best_gain <= 0:
            break
        
        assignment[best_node] = 1 - assignment[best_node]
        current_value += best_gain
    
    return assignment, current_value


def solve_maxcut(data, params, time_limit, logger):
    """Solve MaxCut using Gurobi MIQP with local search for initial solution."""
    import gurobipy as gp
    from gurobipy import GRB
    
    n_nodes = params.get('n_nodes', params.get('num_nodes', None))
    edges_raw = data.get('edges', [])
    
    # Parse edges: could be list of [u, v] or [u, v, w]
    edges = []
    for e in edges_raw:
        if len(e) == 2:
            edges.append((e[0], e[1], 1))
        else:
            edges.append((e[0], e[1], e[2]))
    
    if n_nodes is None:
        # Infer from edges
        n_nodes = max(max(u, v) for u, v, w in edges) + 1
    
    start_time = time.time()
    
    # Run multiple local search restarts to get good initial solution
    best_assignment = None
    best_value = -1
    
    random.seed(0)
    np.random.seed(0)
    
    remaining = time_limit - (time.time() - start_time)
    n_restarts = max(1, min(50, int(remaining / 0.1)))
    
    for restart in range(n_restarts):
        if time.time() - start_time > time_limit * 0.3:
            break
        assignment, value = local_search_maxcut(n_nodes, edges, max_iter=500)
        if value > best_value:
            best_value = value
            best_assignment = assignment[:]
    
    # Log the local search solution
    if best_assignment is not None and logger:
        solution = {}
        for i in range(n_nodes):
            solution[f"x_{i}"] = best_assignment[i]
        logger.log_solution(best_value, {
            "objective_value": best_value,
            "solution": solution
        })
    
    # Now try Gurobi to improve
    remaining_time = time_limit - (time.time() - start_time)
    
    if remaining_time > 1:
        try:
            model = gp.Model("maxcut")
            model.setParam('OutputFlag', 0)
            model.setParam('Seed', 0)
            model.setParam('MIPGap', 1e-4)
            model.setParam('NumericFocus', 0)
            model.setParam('Threads', 1)
            model.setParam('TimeLimit', max(1, remaining_time - 1))
            
            # Binary variables
            x = model.addVars(n_nodes, vtype=GRB.BINARY, name="x")
            
            # Objective: maximize sum of w_ij * (x_i + x_j - 2*x_i*x_j)
            # Gurobi can handle quadratic objectives directly
            obj = gp.QuadExpr()
            for u, v, w in edges:
                obj += w * (x[u] + x[v] - 2 * x[u] * x[v])
            
            model.setObjective(obj, GRB.MAXIMIZE)
            
            # Set initial solution from local search
            if best_assignment is not None:
                for i in range(n_nodes):
                    x[i].Start = best_assignment[i]
            
            model.optimize()
            
            if model.SolCount > 0:
                gurobi_value = model.ObjVal
                if gurobi_value > best_value:
                    best_value = gurobi_value
                    best_assignment = [int(round(x[i].X)) for i in range(n_nodes)]
                    
                    if logger:
                        solution = {}
                        for i in range(n_nodes):
                            solution[f"x_{i}"] = best_assignment[i]
                        logger.log_solution(best_value, {
                            "objective_value": best_value,
                            "solution": solution
                        })
        except Exception as e:
            # If Gurobi fails, fall back to local search result
            pass
    
    # Build final solution
    solution = {}
    for i in range(n_nodes):
        solution[f"x_{i}"] = best_assignment[i]
    
    return {
        "objective_value": best_value,
        "solution": solution
    }


def main():
    parser = argparse.ArgumentParser(description="Solve MaxCut or TSP optimization problem")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to instance JSON file")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to write solution JSON file")
    parser.add_argument("--time_limit", type=int, default=60, help="Maximum runtime in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Path to JSONL log file")
    args = parser.parse_args()
    
    # Read instance
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    # Detect problem type
    params = instance.get('parameters', {})
    data = instance.get('data', {})
    
    # Determine if TSP or MaxCut
    is_tsp = 'n_cities' in params and 'coordinates' in data
    
    if is_tsp:
        logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
        result = solve_tsp(data, params, args.time_limit, logger)
    else:
        logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None
        result = solve_maxcut(data, params, args.time_limit, logger)
    
    # Write solution
    with open(args.solution_path, 'w') as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    from solution_logger import SolutionLogger
    main()
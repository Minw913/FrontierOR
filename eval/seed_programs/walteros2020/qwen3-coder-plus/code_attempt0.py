import argparse
import json
import sys
from collections import defaultdict
from itertools import combinations
import time
from solution_logger import SolutionLogger


def solve_max_clique_brute_force(vertices, edges, n, time_limit):
    """Finds the maximum clique using optimized backtracking."""
    start_time = time.time()
    
    # Create a set for fast edge lookups
    edge_set = set()
    adj_list = defaultdict(set)
    for u, v in edges:
        edge_set.add((min(u, v), max(u, v)))
        adj_list[u].add(v)
        adj_list[v].add(u)
    
    def has_edge(u, v):
        return (min(u, v), max(u, v)) in edge_set
    
    # Check if adding vertex 'v' to current_clique would keep it a clique
    def is_safe_to_add(v, current_clique):
        for u in current_clique:
            if not has_edge(u, v):
                return False
        return True
    
    global_best_clique = []
    
    def branch_and_bound(clique, remaining, lower_bound_clique):
        nonlocal global_best_clique
        
        # Pruning: If remaining elements + current clique size cannot exceed best solution
        if len(clique) + len(remaining) <= len(global_best_clique):
            return
        
        # If we have already beaten the best-known size, update immediately
        if len(clique) > len(global_best_clique):
            global_best_clique = list(clique)
        
        # Early termination based on time limit
        elapsed = time.time() - start_time
        if elapsed >= time_limit:
            return
        
        while remaining:
            # Add current candidate to the clique
            candidate = remaining.pop()
            
            # Recurse by adding candidate if safe
            if is_safe_to_add(candidate, clique):
                new_clique = clique + [candidate]
                
                # Calculate potential next candidates - neighbors of the current element that are in remaining
                new_remaining = [v for v in remaining if v in adj_list[candidate]]
                
                branch_and_bound(new_clique, new_remaining, lower_bound_clique)
            
            elapsed = time.time() - start_time
            if elapsed >= time_limit:
                return
                
            # Continue with the same clique, without adding current candidate
            if len(clique) + len(remaining) <= len(global_best_clique):
                break
    
    # Try starting from different vertices
    for start_vertex in vertices:
        elapsed = time.time() - start_time
        if elapsed >= time_limit:
            break
            
        # Start building cliques from this vertex
        initial_clique = [start_vertex]
        # Only consider the neighbors for addition
        initial_remaining = [v for v in adj_list[start_vertex] if v > start_vertex]
        
        branch_and_bound(initial_clique, initial_remaining, global_best_clique)
    
    return global_best_clique


class MaxCliqueMipsol:
    def __init__(self, edges, n, time_limit):
        self.n = n
        self.edges = edges
        self.time_limit = time_limit
        
        # Build adjacency matrix
        self.adjacent = [[False] * n for _ in range(n)]
        for u, v in edges:
            self.adjacent[u][v] = True
            self.adjacent[v][u] = True
    
    def solve_with_gurobi(self):
        import gurobipy as gp
        from gurobipy import GRB
        
        # Create model
        m = gp.Model("MaxClique")
        
        # Set time limit
        m.Params.TimeLimit = self.time_limit
        m.Params.LogToConsole = 0  # Suppress Gurobi log
        
        # Decision variables: x[i] = 1 if node i is in the clique
        x = m.addVars(self.n, vtype=GRB.BINARY, name="x")
        
        # Objective: maximize sum of selected nodes
        m.setObjective(gp.quicksum(x[i] for i in range(self.n)), GRB.MAXIMIZE)
        
        # For each non-edge (i,j), at most one of the endpoints can be in the clique
        non_edges = []
        for i in range(self.n):
            for j in range(i+1, self.n):
                if not self.adjacent[i][j]:
                    non_edges.append((i, j))
        
        # Add constraints for each non-edge
        for i, j in non_edges:
            m.addConstr(x[i] + x[j] <= 1, name=f"noedge_{i}_{j}")
        
        # Optimize the model
        m.optimize()

        if m.status == GRB.INFEASIBLE:
            print("Model is infeasible", file=sys.stderr)
            return [], 0

        # Get the solution
        obj_val = int(m.ObjVal) if (m.SolCount > 0 and m.Status != GRB.TIME_LIMIT) else 0
        clique = []
        
        if m.SolCount > 0:
            for i in range(self.n):
                if x[i].X > 0.5:  # if binary variable is 1
                    clique.append(i)
            return clique, obj_val
        
        return clique, obj_val


def main():
    parser = argparse.ArgumentParser(description='Solve Maximum Clique Problem')
    parser.add_argument('--instance_path', required=True, help='Path to instance JSON file')
    parser.add_argument('--solution_path', required=True, help='Path to output solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds')
    parser.add_argument('--log_path', help='Optional path for logging intermediate solutions')
    args = parser.parse_args()

    # Initialize the logger
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None
    
    # Load instance
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)

    n = instance_data["n"]
    m = instance_data["m"]
    edges = instance_data["edges"]

    timeout = args.time_limit
    
    # Use different strategies depending on graph properties
    # For smaller graphs: brute_force / MIP approach
    # For large sparse graphs: use the MIP formulation
    
    if n <= 25 :  # Small enough for brute force attempt initially
        clique = solve_max_clique_brute_force(list(range(n)), edges, n, timeout)
        best_clique = clique
        best_size = len(clique)
    else:
        # Use Gurobi for better scalability
        mip_solver = MaxCliqueMipsol(edges, n, args.time_limit)
        try:
            best_clique, best_size = mip_solver.solve_with_gurobi()
        except ImportError:
            # Fallback if Gurobi not available
            clique = solve_max_clique_brute_force(list(range(n)), edges, n, timeout)
            best_clique = clique
            best_size = len(clique)

    if logger:
        if best_size > 0:
            logger.log(best_size)

    # Prepare solution
    solution = {
        "objective_value": best_size,
        "clique_vertices": sorted(best_clique)
    }

    # Write solution
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == '__main__':
    main()
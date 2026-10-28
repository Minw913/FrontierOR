import sys
import time
import json
import argparse
from solution_logger import SolutionLogger

# Increase recursion limit to handle deep recursion
sys.setrecursionlimit(100000)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    # Read the instance JSON file
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    vertices = instance['vertices']
    adjacency_list_dict = instance['adjacency_list']
    greedy_clique = instance['greedy_clique']
    
    # Build the graph as a dictionary of sets for efficient operations
    graph = {}
    for v_str, neighbors in adjacency_list_dict.items():
        v = int(v_str)
        graph[v] = set(neighbors)
    
    # Initialize best_clique with the provided greedy solution
    best_clique = set(greedy_clique)
    best_size = len(best_clique)
    
    # Handle empty graph case
    if not vertices:
        with open(args.solution_path, 'w') as f:
            json.dump({"objective_value": 0, "clique_vertices": []}, f)
        return
    
    # Log the initial solution (greedy clique)
    if logger:
        logger.log(best_size)
    
    class Solver:
        def __init__(self, graph, vertices, time_limit, logger):
            self.graph = graph
            self.vertices = vertices
            self.time_limit = time_limit
            self.start_time = time.time()
            self.logger = logger
            self.best_clique = best_clique
            self.best_size = best_size
        
        def backtrack(self, R, P):
            # Check time limit
            current_time = time.time()
            if current_time - self.start_time > self.time_limit:
                return
            
            # If no candidates left, check if current clique is the best
            if not P:
                if len(R) > self.best_size:
                    self.best_clique = set(R)
                    self.best_size = len(R)
                    if self.logger:
                        self.logger.log(self.best_size)
                return
            
            # Compute degrees of vertices in P relative to the current candidate set
            deg_in_P = {}
            for v in P:
                deg_in_P[v] = len(self.graph[v] & P)
            
            # Sort candidates by degree descending to prioritize high-degree vertices
            sorted_P = sorted(P, key=lambda v: deg_in_P[v], reverse=True)
            
            # Greedy coloring of the candidate set for bounding
            color = {}
            for v in sorted_P:
                used_colors = set()
                for u in self.graph[v]:
                    if u in P and u in color:
                        used_colors.add(color[u])
                c = 1
                while c in used_colors:
                    c += 1
                color[v] = c
            
            max_color = max(color.values()) if color else 0
            bound = len(R) + max_color
            if bound <= self.best_size:
                return
            
            # Explore each candidate vertex
            for v in sorted_P:
                if v not in P:
                    continue
                
                # Include vertex v in the clique
                new_R = R | {v}
                new_P = P & self.graph[v]
                self.backtrack(new_R, new_P)
                
                # Exclude vertex v from further consideration in this branch
                P.discard(v)
                
                # Prune if remaining candidates cannot improve the solution
                if len(R) + len(P) <= self.best_size:
                    break
        
        def solve(self):
            R0 = set()
            P0 = set(self.vertices)
            self.backtrack(R0, P0)
    
    # Solve the problem
    solver = Solver(graph, vertices, args.time_limit, logger)
    solver.solve()
    
    # Prepare and write the solution
    clique_vertices = sorted(solver.best_clique)
    solution = {
        "objective_value": solver.best_size,
        "clique_vertices": clique_vertices
    }
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == '__main__':
    main()
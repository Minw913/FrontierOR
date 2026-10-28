import argparse
import json
import sys
from collections import defaultdict
import gurobipy as gp
from gurobipy import GRB
import time
from itertools import combinations

def parse_args():
    parser = argparse.ArgumentParser(description="Graph Coloring Solver")
    parser.add_argument("--instance_path", required=True, help="Path to the input instance JSON file")
    parser.add_argument("--solution_path", required=True, help="Path to the output solution JSON file")
    parser.add_argument("--time_limit", required=True, type=int, help="Time limit in seconds")
    parser.add_argument("--log_path", type=str, help="Path to the output log JSONL file")
    return parser.parse_args()

class GraphColoringSolver:
    def __init__(self, instance_data, time_limit, logger):
        self.graph = instance_data["graph"]
        self.bounds = instance_data.get("bounds", {})
        self.time_limit = time_limit
        self.logger = logger
        
        # Parse instance data
        self.num_vertices = self.graph["num_vertices"]
        self.edges = [(e[0], e[1]) for e in self.graph["edges"]]
        
        # Get initial bounds
        self.lower_bound = self.bounds.get("clique_lower_bound", 1)
        self.upper_bound = self.bounds.get("greedy_upper_bound", self.num_vertices)
        
        # Build adjacency list for faster access
        self.adj_list = defaultdict(list)
        for u, v in self.edges:
            self.adj_list[u].append(v)
            self.adj_list[v].append(u)

    def greedy_coloring(self):
        """Compute an upper bound using a greedy algorithm"""
        # Sort vertices by degree in descending order
        degrees = {v: len(neighborhood) for v, neighborhood in self.adj_list.items()}
        sorted_vertices = sorted(range(self.num_vertices), key=lambda x: degrees[x], reverse=True)
        
        colors = {}
        color_classes = defaultdict(list)
        
        for vertex in sorted_vertices:
            used_colors = {colors[neighbor] for neighbor in self.adj_list[vertex] if neighbor in colors}
            available_color = 0
            while available_color in used_colors:
                available_color += 1
            colors[vertex] = available_color
            color_classes[available_color].append(vertex)
        
        return max(colors.values()) + 1, list(color_classes.values()), colors

    def initial_solution_from_bounds(self):
        # Check if we have a valid starting solution in the bounds
        if 'greedy_upper_bound' in self.bounds:
            # Since we will compute greedy solution anyway, let's do it here
            _, color_classes, vertex_colors = self.greedy_coloring()
            return self.upper_bound, color_classes, vertex_colors
        else:
            # Compute a greedy upper bound if not provided
            k, color_classes, vertex_colors = self.greedy_coloring()
            return k, color_classes, vertex_colors

    def solve_with_gurobi(self, max_colors):
        """Solve the graph coloring problem using Gurobi for a fixed number of colors."""
        try:
            model = gp.Model("GraphColoring")
            model.setParam('TimeLimit', self.time_limit)
            model.setParam('OutputFlag', 0)

            # Variables: x[i][j] = 1 if vertex i gets color j, y[j] = 1 if color j is used
            x = {}  # x[vertex][color]
            y = {}  # y[color]

            for v in range(self.num_vertices):
                for c in range(max_colors):
                    x[(v, c)] = model.addVar(vtype=GRB.BINARY, name=f"x_{v}_{c}")

            for c in range(max_colors):
                y[c] = model.addVar(vtype=GRB.BINARY, name=f"y_{c}")

            # Constraint: Each vertex gets exactly one color
            for v in range(self.num_vertices):
                model.addConstr(gp.quicksum(x[(v, c)] for c in range(max_colors)) == 1)

            # Constraint: Adjacent vertices get different colors
            for u, v in self.edges:
                for c in range(max_colors):
                    model.addConstr(x[(u, c)] + x[(v, c)] <= 1)

            # Constraint: If any vertex uses color c, then y[c] = 1
            for c in range(max_colors):
                for v in range(self.num_vertices):
                    model.addConstr(x[(v, c)] <= y[c])

            # Objective: Minimize the number of colors used
            model.setObjective(gp.quicksum(y[c] for c in range(max_colors)), GRB.MINIMIZE)

            # Solve
            start_time = time.time()
            model.optimize()
            time_spent = time.time() - start_time

            # Extract solution if feasible
            if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
                if model.SolCount > 0:  # There's at least one solution found
                    
                    color_assignment = {}
                    
                    # Get color assignments for vertices
                    for v in range(self.num_vertices):
                        for c in range(max_colors):
                            if x[(v, c)].X > 0.5:  # If vertex v gets color c
                                color_assignment[v] = c
                                break
                    
                    # Group vertices by color to form color classes
                    color_to_vertices = defaultdict(list)
                    for vertex, color in color_assignment.items():
                        color_to_vertices[color].append(vertex)
                    
                    color_classes = [sorted(color_to_vertices[c]) for c in range(max_colors) 
                                    if any(x[(v, c)].X > 0.5 for v in range(self.num_vertices))]
                    
                    actual_num_colors = len(color_classes)
                    
                    return True, actual_num_colors, color_classes, color_assignment

            status_msg = f"Model status: {model.status}"
            if model.status == GRB.INFEASIBLE:
                return False, max_colors, [], {}
            elif model.status in [GRB.SUBOPTIMAL, GRB.TIME_LIMIT]:
                # If solution exists despite time limit, return it
                if model.SolCount > 0:
                    color_assignment = {}
                    for v in range(self.num_vertices):
                        for c in range(max_colors):
                            if x[(v, c)].X > 0.5:
                                color_assignment[v] = c
                                break
                
                    color_to_vertices = defaultdict(list)
                    for vertex, color in color_assignment.items():
                        color_to_vertices[color].append(vertex)
                    
                    color_classes = [sorted(color_to_vertices[c]) for c in range(max_colors) 
                                    if any(x[(v, c)].X > 0.5 for v in range(self.num_vertices))]
                    
                    actual_num_colors = len(color_classes)
                    return True, actual_num_colors, color_classes, color_assignment
                else:
                    return False, max_colors, [], {}

        except Exception as e:
            print(f"Error occurred during Gurobi optimization: {e}")
            return False, max_colors, [], {}

    def solve(self):
        # First, try a greedy solution to have an initial upper bound
        current_k, current_color_classes, current_vertex_colors = self.initial_solution_from_bounds()
        
        if self.logger:
            self.logger.log(current_k)
        
        best_solution = (current_k, current_color_classes, current_vertex_colors)
        
        # Perform binary search between lower and upper bounds
        low = max(self.lower_bound, 1)
        high = min(self.upper_bound, current_k)  # Update high with greedy result
        
        # If lower bound is equal to our greedy solution, we already have the optimal solution
        if low >= current_k:
            return best_solution
        
        while low < high:
            mid = (low + high) // 2
            
            is_feasible, num_colors, color_classes, vertex_colors = self.solve_with_gurobi(mid)
            
            if is_feasible and time.time() > initial_start_time + self.time_limit:
                # Time exceeded - return the best we currently have (best_solution)
                return best_solution
                
            if is_feasible:
                # Update our best known solution
                current_k = num_colors
                current_color_classes = color_classes
                current_vertex_colors = vertex_colors
                best_solution = (current_k, current_color_classes, current_vertex_colors)
                
                # Log improvement
                if self.logger:
                    self.logger.log(current_k)
                
                # This was feasible, try fewer colors
                high = mid
            else:
                # Not feasible, must use at least mid+1 colors
                low = mid + 1
                
            remaining_time = initial_start_time + self.time_limit - time.time()
            if remaining_time <= 0:
                break

        return best_solution


def main():
    global initial_start_time
    args = parse_args()
    
    # Import solution_logger after processing args to make sure it exists
    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize")
    else:
        logger = None
    
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)
        
    initial_start_time = time.time()
    solver = GraphColoringSolver(instance_data, args.time_limit, logger)
    objective_value, color_classes, vertex_colors = solver.solve()
    
    solution = {
        "objective_value": objective_value,
        "selected_sets": color_classes,
        "coloring": {str(k): v for k, v in vertex_colors.items()}
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == "__main__":
    main()
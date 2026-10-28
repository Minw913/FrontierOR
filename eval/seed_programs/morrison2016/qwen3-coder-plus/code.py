import argparse
import json
import sys
from collections import defaultdict
import time
from itertools import combinations
import os

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
        """Compute a coloring using a greedy algorithm with DSatur heuristic"""
        colors = [-1] * self.num_vertices  # -1 indicates uncolored
        degree_saturation = [0] * self.num_vertices  # How many different colors are adjacent
        active_colors = [set() for _ in range(self.num_vertices)]  # Colors adjacent to each vertex
        available_colors = [set(range(self.num_vertices)) for _ in range(self.num_vertices)]  # Track available colors per vertex
        
        # Initially, degree saturation is just degree for unassigned vertices
        for v in range(self.num_vertices):
            degree_saturation[v] = len(self.adj_list[v])
        
        uncolored = set(range(self.num_vertices))
        color_count = 0
        
        while uncolored:
            # Find uncolored vertex with highest saturation
            # If ties, pick the one with highest degree among uncolored neighbors
            
            max_sat = max(degree_saturation[v] for v in uncolored)
            candidates = [v for v in uncolored if degree_saturation[v] == max_sat]
            
            if len(candidates) > 1:
                # Tie-break by original degree
                max_adj = max(len(self.adj_list[v]) for v in candidates)
                v = next(v for v in candidates if len(self.adj_list[v]) == max_adj)
            else:
                v = candidates[0]
            
            # Find smallest available color for this vertex
            color = min(available_colors[v])
            
            colors[v] = color
            color_count = max(color_count, color + 1)
            
            # Update affected vertices
            for neighbor in self.adj_list[v]:
                active_colors[neighbor].add(color)
                degree_saturation[neighbor] = len(active_colors[neighbor])
                if neighbor in uncolored:
                    available_colors[neighbor].discard(color)
            
            uncolored.remove(v)
        
        # Group vertices by color
        color_classes = [[] for _ in range(color_count)]
        for vertex, color in enumerate(colors):
            color_classes[color].append(vertex)
        
        for class_list in color_classes:
            class_list.sort()
        
        # Create coloring dict
        vertex_colors = {i: colors[i] for i in range(self.num_vertices)}
        
        return len(color_classes), color_classes, vertex_colors

    def initial_solution_from_bounds(self):
        # Compute greedy solution regardless of what's in bounds
        k, color_classes, vertex_colors = self.greedy_coloring()
        return k, color_classes, vertex_colors

    def solve_with_local_search(self, init_k, init_color_classes, init_vertex_colors):
        """
        Improve solution using local search
        """
        current_colors = init_vertex_colors.copy()
        current_color_classes = [cls.copy() for cls in init_color_classes]
        current_objective = len(current_color_classes)
        
        start_time = time.time()
        
        improved = True
        while improved and (time.time() - start_time < self.time_limit * 0.9):  # Leave small buffer for finalization
            improved = False
            
            max_color = max(current_colors.values())
            for test_color in range(max_color + 1):
                # Identify vertices with this color
                vertices_with_test_color = [v for v, color in current_colors.items() if color == test_color]
                
                # Try to eliminate this color if it has at least 2 vertices
                if len(vertices_with_test_color) < 1:
                    continue
                    
                # Remove one vertex at a time and try to recolor it
                for v_to_reassign in vertices_with_test_color[:1]:  # Just first vertex
                    # Get available colors for this vertex
                    neighbor_colors = {current_colors[n] for n in self.adj_list[v_to_reassign]}
                    avbl_colors = []
                    for c in range(current_objective - 1):  # Up to one less than current number of colors
                        if c not in neighbor_colors:
                            avbl_colors.append(c)
                    
                    if avbl_colors:
                        new_color = min(avbl_colors)
                        
                        # Perform reassignment
                        old_color = current_colors[v_to_reassign]
                        
                        # Update data structures
                        current_colors[v_to_reassign] = new_color
                        
                        # Update color classes
                        current_color_classes[old_color].remove(v_to_reassign)
                        current_color_classes[new_color].append(v_to_reassign)
                        current_color_classes[new_color].sort()
                        
                        # If old color's class is empty, remove it
                        if len(current_color_classes[old_color]) == 0:
                            # Reindex all higher colors down one
                            old_color_count = current_objective
                            new_color_classes = []
                            new_colors_map = {}
                            
                            color_idx = 0
                            for c_idx in range(old_color_count):
                                if c_idx != old_color:
                                    new_colors_map[c_idx] = color_idx
                                    new_class = [x for x in current_color_classes[c_idx]]
                                    new_color_classes.append(new_class)
                                    # Update colors of the vertices we're adding
                                    for vert in new_class:
                                        current_colors[vert] = color_idx
                                    color_idx += 1
                            
                            current_color_classes = new_color_classes
                            current_objective -= 1
                            
                        improved = True
                        
                        # Log improvement
                        if self.logger:
                            self.logger.log(current_objective)
                        
                        # Break and continue loop to see the impact of this change
                        break
                
                if improved:
                    break
    
        return current_objective, current_color_classes, current_colors


    def solve(self):
        # Start with greedy solution
        current_k, current_color_classes, current_vertex_colors = self.initial_solution_from_bounds()
        
        if self.logger:
            self.logger.log(current_k)
        
        best_solution = (current_k, current_color_classes, current_vertex_colors)
        
        # Then try to refine with local search
        if self.time_limit > 0.1:  # Only if we have some time
            refined_obj, refined_color_classes, refined_colors = self.solve_with_local_search(
                current_k, current_color_classes, current_vertex_colors
            )
            
            if refined_obj < current_k:
                current_k = refined_obj
                current_color_classes = refined_color_classes
                current_vertex_colors = refined_colors
                if self.logger:
                    self.logger.log(current_k)
                best_solution = (refined_obj, refined_color_classes, refined_colors)
        
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
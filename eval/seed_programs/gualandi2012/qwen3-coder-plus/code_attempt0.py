import argparse
import json
import sys
from collections import defaultdict
import random

def solve_graph_coloring(instance_data, time_limit):
    """
    Solve the graph coloring problem with the goal of minimizing the number of colors used.
    
    Args:
        instance_data: Dictionary containing the graph information
        time_limit: Maximum allowed computation time in seconds
    
    Returns:
        tuple: (objective_value, coloring, color_classes) for the found solution
    """
    num_vertices = instance_data['graph']['num_vertices']
    edges = instance_data['graph']['edges']
    
    # Build adjacency list
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)
        
    max_degree = max(len(graph[v]) for v in range(num_vertices)) if num_vertices > 0 else 0
    
    def greedy_coloring(order=None):
        """Apply greedy coloring algorithm with the given order."""
        if order is None:
            order = list(range(num_vertices))
        
        colors = {}
        for v in order:
            used_neighbors_colors = {colors[nbr] for nbr in graph[v] if nbr in colors}
            available_color = 0
            while available_color in used_neighbors_colors:
                available_color += 1
            colors[v] = available_color
        
        return colors
    
    # Start with greedy coloring using different orders as baselines
    initial_orders = [
        list(range(num_vertices)),  # natural order
        list(reversed(range(num_vertices))),  # reverse order
        sorted(range(num_vertices), key=lambda x: len(graph[x]), reverse=True),  # decreasing degree order
        sorted(range(num_vertices), key=lambda x: len(graph[x])),  # increasing degree order
    ]
    
    best_colors = None
    best_color_classes = defaultdict(list)
    
    for order in initial_orders:
        current_colors = greedy_coloring(order)
        current_num_colors = len(set(current_colors.values()))
        
        if best_colors is None or current_num_colors < len(set(best_colors.values())):
            best_colors = current_colors
            
    if not best_colors:
        # Handle case where there are no vertices
        return 0, {}, {}
    
    best_color_classes = defaultdict(list)
    for vertex, color in best_colors.items():
        best_color_classes[color].append(vertex)
    
    best_objective = len(set(best_colors.values()))
    
    # Improve with DSatur heuristic
    def dsatur_coloring():
        """Apply the DSatur heuristic for graph coloring."""
        # Initialize uncolored vertices
        uncolored = set(range(num_vertices))
        colors = {}
        
        # Select first vertex with highest degree
        degrees = [len(graph[v]) for v in range(num_vertices)]
        start_vertex = max(range(num_vertices), key=lambda x: degrees[x])
        
        colors[start_vertex] = 0
        uncolored.remove(start_vertex)
        
        while uncolored:
            # Calculate saturation degree (number of different adjacent colored neighbors)
            saturation_degrees = {}
            for v in uncolored:
                adjacent_colors = set()
                for neighbor in graph[v]:
                    if neighbor in colors:
                        adjacent_colors.add(colors[neighbor])
                saturation_degrees[v] = len(adjacent_colors)
            
            # Find maximum saturation degree
            max_saturation = max(saturation_degrees.values()) if saturation_degrees else 0
            
            # Among vertices with max saturation, pick the one with max degree in uncolored subgraph
            candidates = [v for v, sat_deg in saturation_degrees.items() if sat_deg == max_saturation]
            
            # Find vertex with most uncolored neighbors among candidates
            max_uncolored_neighbors = -1
            selected_vertex = candidates[0]
            for v in candidates:
                deg_to_uncolored = sum(1 for nbr in graph[v] if nbr in uncolored)
                if deg_to_uncolored > max_uncolored_neighbors:
                    max_uncolored_neighbors = deg_to_uncolored
                    selected_vertex = v
            
            # Assign the lowest numbered color not used by adjacent colored vertices
            used_neighbors_colors = {colors[nbr] for nbr in graph[selected_vertex] if nbr in colors}
            new_color = 0
            while new_color in used_neighbors_colors:
                new_color += 1
            colors[selected_vertex] = new_color
            
            # Remove vertex from uncolored
            uncolored.remove(selected_vertex)
        
        return colors
    
    dsatur_colors = dsatur_coloring()
    dsatur_objective = len(set(dsatur_colors.values()))
    
    if dsatur_objective < best_objective:
        best_colors = dsatur_colors
        best_objective = dsatur_objective
    
    # Create color classes
    best_color_classes = defaultdict(list)
    for vertex, color in best_colors.items():
        best_color_classes[color].append(vertex)
        
    # Additional improvement: try local search improvements
    improved = True
    iterations = 500  # limit iterations to stay within time limits
    
    for iteration in range(iterations):
        if best_objective <= max_degree + 1:
            break
            
        improved = False
        current_colors = best_colors.copy()
        current_color_classes = best_color_classes.copy()
        original_objective = best_objective
        
        # Local search: try re-coloring vertices 
        for _ in range(min(20, num_vertices)):  # limit local search steps per iteration
            # Pick a random vertex
            v = random.randint(0, num_vertices - 1)
            old_color = current_colors.get(v, -1)
            
            # Find available colors for this vertex
            used_neighbor_colors = {current_colors[n] for n in graph[v] if n in current_colors and n != v}
            available_colors = []
            for c in range(original_objective):  # check existing colors
                if c != old_color and c not in used_neighbor_colors:
                    available_colors.append(c)
            
            # Try using a color from color class reduction perspective
            if available_colors:
                new_color = min(available_colors)
                
                # Update color classes
                old_color_class = current_color_classes[old_color]
                old_color_class.remove(v)
                
                new_color_class = current_color_classes.setdefault(new_color, [])
                new_color_class.append(v)
                
                current_colors[v] = new_color
                
                # Count number of colors that have at least one vertex
                actual_colors_count = len([c for c, vertices in current_color_classes.items() if vertices])
                
                if actual_colors_count < original_objective:
                    best_colors = current_colors.copy()
                    best_color_classes = {c: vs[:] for c, vs in current_color_classes.items()}
                    best_objective = actual_colors_count
                    improved = True
        
        if not improved:
            break
            
        best_color_classes = defaultdict(list)
        for vertex, color in best_colors.items():
            best_color_classes[color].append(vertex)
        
        best_objective = len([c for c, vertices in best_color_classes.items() if vertices])

    # Ensure color classes do not have empty lists for better output consistency
    pruned_color_classes = {c: vs for c, vs in best_color_classes.items() if vs}
    
    return best_objective, best_colors, pruned_color_classes


def main():
    parser = argparse.ArgumentParser(description='Solve the Graph Coloring Problem')
    parser.add_argument('--instance_path', required=True, help='Path to the input JSON instance')
    parser.add_argument('--solution_path', required=True, help='Path to the output JSON solution')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds')
    parser.add_argument('--log_path', required=False, help='Path to the log file for intermediate solutions')
    
    args = parser.parse_args()
    
    # Read the instance from JSON file
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)
    
    # Loggers for intermediate results
    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    # Call the solver function
    objective_value, coloring, color_classes = solve_graph_coloring(instance_data, args.time_limit)
    
    # Log any intermediate solution (here we just log the final one)
    if logger:
        # Since we don't track intermediate solutions in this basic heuristic approach,
        # we could improve the method to log improvements during local search
        pass  # In a real implementation, you'd want to log progress in the actual solver function
    
    # Prepare the solution dictionary
    solution = {
        "objective_value": objective_value,
        "coloring": coloring,
        "color_classes": color_classes
    }
    
    # Write the solution to the output file
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()
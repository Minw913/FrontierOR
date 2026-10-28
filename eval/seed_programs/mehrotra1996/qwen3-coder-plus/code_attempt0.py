import argparse
import json
import sys
from collections import defaultdict, deque
import time

def parse_args():
    parser = argparse.ArgumentParser(description='Graph Coloring Problem Solver')
    parser.add_argument('--instance_path', type=str, required=True,
                        help='Path to the JSON file containing the problem instance.')
    parser.add_argument('--solution_path', type=str, required=True,
                        help='Path where the final solution JSON file must be written.')
    parser.add_argument('--time_limit', type=int, required=True,
                        help='Maximum runtime in seconds.')
    parser.add_argument('--log_path', type=str, default=None,
                        help='Path to a JSONL file for logging intermediate solutions (optional).')
    return parser.parse_args()

def build_adjacency_list(nodes, edges):
    adj_list = {node: set() for node in nodes}
    for u, v in edges:
        if u != v:  # Handle self-loops if present
            adj_list[u].add(v)
            adj_list[v].add(u)
    return adj_list

def greedy_coloring(nodes, edges, adj_list, reverse=False):
    """Implements greedy graph coloring in order determined by reverse parameter."""
    ordering = sorted(nodes, reverse=reverse)
    
    color_assignment = {}
    max_color_index = -1
    
    for node in ordering:
        # Find colors used by neighbors
        used_colors = {color_assignment[neighbor] 
                       for neighbor in adj_list[node] 
                       if neighbor in color_assignment}
        
        # Assign the lowest available color index
        available_color = 0
        while available_color in used_colors:
            available_color += 1
        
        color_assignment[node] = available_color
        max_color_index = max(max_color_index, available_color)
    
    return color_assignment, max_color_index + 1

def get_degree_ordering(nodes, adj_list):
    """Get node ordering by degree (descending)"""
    degrees = [(len(adj_list[node]), node) for node in nodes]
    degrees.sort(reverse=True)
    return [node for degree, node in degrees]

def dsatur_coloring(nodes, adj_list):
    """
    DSATUR (Degree Saturation) graph coloring heuristic.
    Selects the uncolored vertex with:
    1. Highest degree of already colored neighbors (saturation degree)
    2. Then highest degree among uncolored
    """
    color_assignment = {}
    available = set(nodes)
    
    # Color first node (highest degree) with color 0
    ordered_nodes = get_degree_ordering(nodes, adj_list)
    first_node = ordered_nodes[0]  
    color_assignment[first_node] = 0
    available.remove(first_node)
    
    while available:
        # Calculate saturation degree for each uncolored node
        # Saturation degree: number of distinct colors in neighborhood
        saturation_degrees = {}
        for node in available:
            neighbor_colors = {color_assignment[neighbor] 
                              for neighbor in adj_list[node] 
                              if neighbor in color_assignment}
            saturation_degrees[node] = len(neighbor_colors)
        
        # Select node with highest saturation degree
        # In case of ties, select one with highest uncolored degree
        max_saturation = max(saturation_degrees.values())
        candidates = [node for node, sat in saturation_degrees.items() 
                     if sat == max_saturation]
        
        # Among these, pick the one with highest degree in uncolored graph
        candidate_degrees = [(len([n for n in adj_list[node] if n in available]), node) 
                           for node in candidates]
        candidate_degrees.sort(reverse=True)
        selected_node = candidate_degrees[0][1]
        
        # Find the lowest available color
        used_colors = {color_assignment[neighbor] 
                      for neighbor in adj_list[selected_node] 
                      if neighbor in color_assignment}
        
        next_color = 0
        while next_color in used_colors:
            next_color += 1
        
        color_assignment[selected_node] = next_color
        available.remove(selected_node)
    
    max_color = max(color_assignment.values()) if color_assignment else 0
    return color_assignment, max_color + 1

def initial_solution_greedy(nodes, edges, adj_list):
    """Generate initial solutions using multiple greedy heuristics."""
    best_coloring = None
    best_num_colors = float('inf')
    
    # Standard greedy
    greedy_assignment, num_colors = greedy_coloring(nodes, edges, adj_list)
    if num_colors < best_num_colors:
        best_num_colors = num_colors
        best_coloring = greedy_assignment
    
    # Reverse greedy
    reverse_assignment, num_colors = greedy_coloring(nodes, edges, adj_list, reverse=True)
    if num_colors < best_num_colors:
        best_num_colors = num_colors
        best_coloring = reverse_assignment
    
    # Degree-based greedy
    degree_order = get_degree_ordering(nodes, adj_list)
    degree_assignment = {}
    max_color_idx = -1
    
    for node in degree_order:
        used_colors = {degree_assignment[neighbor] 
                      for neighbor in adj_list[node] 
                      if neighbor in degree_assignment}
        
        available_color = 0
        while available_color in used_colors:
            available_color += 1
        
        degree_assignment[node] = available_color
        max_color_idx = max(max_color_idx, available_color)
    
    if (max_color_idx + 1) < best_num_colors:
        best_num_colors = max_color_idx + 1
        best_coloring = degree_assignment
    
    # DSATUR coloring
    dsatur_assignment, num_colors = dsatur_coloring(nodes, adj_list)
    if num_colors < best_num_colors:
        best_num_colors = num_colors
        best_coloring = dsatur_assignment
    
    return best_coloring, best_num_colors

def local_search_improvement(nodes, adj_list, color_assignment, num_colors, initial_best_value, timeout_time):
    """Try to improve existing solution through local search."""
    current_assignment = color_assignment.copy()
    current_num_colors = num_colors
    max_iter = 500  # Limit local search iterations
    
    for iteration in range(max_iter):
        if time.time() > timeout_time:
            break
            
        # Early termination if we're at best-known value
        if current_num_colors <= initial_best_value and initial_best_value > 0:
            break
            
        # Identify redundant colors (colors that could potentially be reclaimed)
        color_counts = defaultdict(int)
        for node, color in current_assignment.items():
            color_counts[color] += 1

        modified = False
        remaining_colors = list(set(current_assignment.values()))
                
        # Try to remove one color by re-coloring nodes with that color
        for try_remove_color in sorted(remaining_colors, reverse=True):  # Start from highest colors
            if time.time() > timeout_time:
                break
                
            if color_counts[try_remove_color] != 1:
                continue  # Only worth trying if just one node uses this color
            
            nodes_of_target_color = [n for n, c in current_assignment.items() 
                                     if c == try_remove_color]
            
            # For this one node, see if we can recolor it with available lower color
            node_to_recolor = nodes_of_target_color[0]
            used_neighbor_colors = {current_assignment[nbor] for nbor in adj_list[node_to_recolor]}
            
            # Find minimum unused color for this node
            for new_color in range(try_remove_color):  # Look for smaller colors first
                if new_color not in used_neighbor_colors:
                    # Can recolor this node to eliminate the higher color
                    current_assignment[node_to_recolor] = new_color
                    # Update number of colors
                    new_max_color = max(current_assignment.values())
                    current_num_colors = new_max_color + 1
                    
                    updated_color_counts = defaultdict(int)
                    for color in current_assignment.values():
                        updated_color_counts[color] += 1
                    current_num_colors = len(updated_color_counts)
                    
                    modified = True
                    break
            
            if modified:
                break
        
        if not modified:
            # Try reordering and recoloring less constrained nodes with smaller colors
            # For now, use DSATUR-like re-coloring with different priorities
            old_vals = list(current_assignment.values())
            ordered_nodes = get_degree_ordering(nodes, adj_list)
            
            # Temporarily unassign one color class and try reassigning
            for color_cl in sorted(set(old_vals)):
                nodes_with_cl = [n for n, c in current_assignment.items() if c == color_cl]
                
                # Unassign these temporarily
                temp_assign = current_assignment.copy()
                for n in nodes_with_cl:
                    del temp_assign[n]
                    
                # Reassign with preference for lower colors
                for n in nodes_with_cl:
                    color_opts = [c for c in range(max(temp_assign.values())+2) if c < color_cl] + \
                                 [c for c in range(max(temp_assign.values())+2) if c >= color_cl]
                    
                    used_neighbors = {temp_assign[adj] for adj in adj_list[n] if adj in temp_assign}
                    for c_opt in color_opts:
                        if c_opt not in used_neighbors:
                            temp_assign[n] = c_opt
                            break
                            
                if len(set(temp_assign.values())) < len(set(current_assignment.values())):
                    current_assignment = temp_assign
                    current_num_colors = len(set(current_assignment.values()))
                    modified = True
                    break
        
        if not modified:
            break
    
    # Count actual number of distinct colors used
    num_colors_used = len(set(current_assignment.values()))
    return current_assignment, num_colors_used

def construct_solution_output(color_assignment, graph):
    """Construct solution according to schema."""
    nodes = graph['nodes']
    
    # Get unique colors used
    used_colors = sorted(list(set(color_assignment.values())))
    color_map = {color: idx for idx, color in enumerate(used_colors)}
    
    # Remap to contiguous 0..k-1 coloring
    vertex_colors = {str(node): color_map[color_assignment[node]] for node in nodes}
    
    # Create color classes
    color_classes_dict = defaultdict(list)
    for node, color in vertex_colors.items():
        color_classes_dict[color].append(int(node))
    
    # Sort color classes and ensure correct order
    color_classes = [sorted(color_classes_dict[i]) for i in range(len(used_colors))]
    
    objective_value = len(used_colors)
    
    return {
        "objective_value": objective_value,
        "vertex_colors": vertex_colors,
        "color_classes": color_classes
    }

def solve_graph_coloring(instance_file, time_limit):
    with open(instance_file, 'r') as f:
        graph_data = json.load(f)['graph']

    nodes = graph_data['nodes']
    edges = graph_data['edges']
    n = len(nodes)
    
    # Build adjacency list
    adj_list = build_adjacency_list(nodes, edges)

    start_time = time.time()
    timeout_time = start_time + time_limit - 0.1  # Leave small buffer

    # Generate initial solution using greedy methods
    best_coloring, best_num_colors = initial_solution_greedy(nodes, edges, adj_list)
    
    # Attempt improvement via local search if time permits
    if time.time() < timeout_time:
        improved_coloring, improved_num_colors = local_search_improvement(
            nodes, adj_list, best_coloring, best_num_colors, best_num_colors, timeout_time
        )
        
        if improved_num_colors < best_num_colors:
            best_coloring = improved_coloring
            best_num_colors = improved_num_colors

    # If graph has nodes but no edges, all can have same color
    if graph_data['num_edges'] == 0:
        for node in nodes:
            best_coloring[node] = 0
        best_num_colors = 1

    return best_coloring, best_num_colors, graph_data

def main():
    args = parse_args()
    
    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    start_time = time.time()

    color_assignment, num_colors, graph = solve_graph_coloring(args.instance_path, args.time_limit)
    
    if logger:
        logger.log(num_colors)
    
    solution = construct_solution_output(color_assignment, {'nodes': graph['nodes']})
    solution["objective_value"] = num_colors
    
    # Verify solution (basic check of adjacent vertices having different colors)
    edges = graph['edges']
    adj_list = build_adjacency_list(graph['nodes'], edges)
    for u, v in edges:
        if solution['vertex_colors'][str(u)] == solution['vertex_colors'][str(v)]:
            print(f"ERROR: Adjacent vertices {u} and {v} have same color.", file=sys.stderr)
            sys.exit(1)
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == '__main__':
    main()
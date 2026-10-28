import argparse
import json
import sys
from collections import defaultdict
import gurobipy as gp
from gurobipy import GRB
import heapq


def compute_greedy_upper_bound(num_nodes, edges):
    """
    Computes an upper bound using greedy graph coloring algorithm (Welsh-Powell).
    """
    # Build adjacency list to determine degrees
    adj_list = defaultdict(list)
    for u, v in edges:
        adj_list[u].append(v)
        adj_list[v].append(u)
    
    # Compute degree of each node
    degree = {i: len(adj_list[i]) for i in range(num_nodes)}
    
    # Sort nodes by degree in descending order
    sorted_nodes = sorted(range(num_nodes), key=lambda x: degree[x], reverse=True)
    
    # Initialize color assignments
    color_assignment = {}
    
    # Assign colors greedily
    for node in sorted_nodes:
        used_colors = set()
        for neighbor in adj_list[node]:
            if neighbor in color_assignment:
                used_colors.add(color_assignment[neighbor])
        
        # Assign smallest available color 
        color = 0
        while color in used_colors:
            color += 1
        color_assignment[node] = color
    
    max_color = max(color_assignment.values()) if color_assignment else -1
    return max_color + 1  # Number of colors used


def solve_graph_coloring_greedy_instance(num_nodes, edges, time_limit=60):
    """
    Solves the graph coloring problem with a greedy approach.
    This serves as initial solution and fallback when optimization needs more guidance.
    """
    # Build adjacency list
    adj_list = defaultdict(list)
    for u, v in edges:
        adj_list[u].append(v)
        adj_list[v].append(u)

    # Apply Welsh-Powell algorithm (greedy)
    degree = {i: len(adj_list[i]) for i in range(num_nodes)}
    sorted_nodes = sorted(range(num_nodes), key=lambda x: degree[x], reverse=True)

    color_assignment = {}
    colors = []  # List to keep track of colors in order they're added
    
    for node in sorted_nodes:
        used_colors = set()
        for neighbor in adj_list[node]:
            if neighbor in color_assignment:
                used_colors.add(color_assignment[neighbor])

        # Find smallest available color
        color = 0
        while color in used_colors:
            color += 1
        
        color_assignment[node] = color
        if color >= len(colors):
            colors.append(color)

    # Optimize the coloring: Try to reassign colors in canonical form (0, 1, ..., k)
    color_map = {}  # Maps actual colors to compressed color numbers
    reordered_colors = {}
    next_new_color = 0
    for node, color in color_assignment.items():
        if color not in color_map:
            color_map[color] = next_new_color
            next_new_color += 1
        reordered_colors[node] = color_map[color]

    return len(set(reordered_colors.values())), reordered_colors


def solve_graph_coloring_mip(num_nodes, edges, time_limit=60, logger=None):
    """
    Solve graph coloring problem using Mixed Integer Programming
    """
    # Compute upper bound using greedy method first
    max_colors = compute_greedy_upper_bound(num_nodes, edges)
    # Limit to reasonable number to prevent memory issues
    max_colors = min(max_colors, num_nodes)

    # Create model
    model = gp.Model("graph_coloring")
    model.setParam("Threads", 1)
    model.setParam('TimeLimit', time_limit)
    model.setParam('IntegralityFocus', 1)  # Improve integrality tolerance
    model.setParam('MIPFocus', 1)  # Focus on finding good feasible solutions fast
    
    # Model variables:
    # x[i,c] = 1 if node i is assigned color c
    # y[c] = 1 if color c is used
    x = {}
    for i in range(num_nodes):
        for c in range(max_colors):
            x[i, c] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{c}")

    y = {}
    for c in range(max_colors):
        y[c] = model.addVar(vtype=GRB.BINARY, name=f"y_{c}")
    
    # Constraints
    # Each node gets exactly one color
    for i in range(num_nodes):
        model.addConstr(gp.quicksum(x[i, c] for c in range(max_colors)) == 1, f"one_color_{i}")
    
    # Adjacent nodes get different colors
    for u, v in edges:
        for c in range(max_colors):
            model.addConstr(x[u, c] + x[v, c] <= 1, f"no_adj_same_color_{u}_{v}_{c}")
    
    # If color c is assigned to any node, then y[c] must be 1
    for c in range(max_colors):
        for i in range(num_nodes):
            model.addConstr(x[i, c] <= y[c], f"use_if_color_{i}_{c}")
            
    model.setObjective(gp.quicksum(y[c] for c in range(max_colors)), GRB.MINIMIZE)
    
    # Set callback for logging intermediate solutions
    def mycallback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj)
                
    # Optimize the model
    model.optimize(mycallback)
    
    # Extract the solution
    if model.SolCount > 0:
        # Process solution to ensure canonical representation of colors
        selected_colors = []
        for c in range(max_colors):
            if y[c].X > 0.5:
                selected_colors.append(c)
        
        # Map selected colors back to consecutive integers 0..ncolors-1
        color_map = {selected_colors[i]: i for i in range(len(selected_colors))}
        
        final_coloring = {}
        for i in range(num_nodes):
            for c in range(max_colors):
                if x[i, c].X > 0.5:
                    mapped_color = color_map[c]
                    final_coloring[i] = mapped_color
                    break
        
        obj_val = len(selected_colors)
        return obj_val, final_coloring
                
    else:
        # If no solution found from MIP due to timeout, try with greedy solution
        opt_val, colors = solve_graph_coloring_greedy_instance(num_nodes, edges, time_limit)
        return opt_val, colors
    

def parse_args():
    parser = argparse.ArgumentParser(description='Solve Graph Coloring Problem')
    parser.add_argument('--instance_path', type=str, required=True,
                        help='Path to the JSON file containing the problem instance.')
    parser.add_argument('--solution_path', type=str, required=True,
                        help='Path where the final solution JSON file must be written.')
    parser.add_argument('--time_limit', type=int, default=300,
                        help='Maximum runtime in seconds.')
    parser.add_argument('--log_path', type=str, default=None,
                        help='Path to a JSONL file for logging intermediate solutions (optional).')
    
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    
    # Load instance
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)
    
    # Initialize logger if path provided
    logger = None
    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize")

    # Extract data
    num_nodes = instance_data['num_nodes']
    num_edges = instance_data['num_edges']
    edges = instance_data['edges']

    # Solve the problem
    objective_value, coloring = solve_graph_coloring_mip(
        num_nodes, 
        edges, 
        time_limit=args.time_limit,
        logger=logger
    )

    # Prepare and save solution
    solution_data = {
        "objective_value": objective_value,
        "coloring": {str(node): color for node, color in coloring.items()}
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution_data, f)


if __name__ == "__main__":
    main()
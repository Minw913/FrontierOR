import argparse
import json
import sys
from collections import defaultdict
import time
try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    gp = None
    GRB = None

def create_graph_coloring_model(n, edges):
    # Create model
    model = gp.Model("GraphColoring")
    model.setParam('OutputFlag', 0)  # Disable console logging
    
    # The upper bound on colors needed is n (each vertex gets a unique color)
    max_colors = n
    
    # Binary variables x[i][c] = 1 if vertex i gets color c
    x = {}
    for i in range(n):
        x[i] = {}
        for c in range(max_colors):
            x[i][c] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{c}")

    # Binary variables y[c] = 1 if color c is used
    y = {}
    for c in range(max_colors):
        y[c] = model.addVar(vtype=GRB.BINARY, name=f"y_{c}")

    # Constraint: Each vertex must be assigned exactly one color
    for i in range(n):
        model.addConstr(gp.quicksum(x[i][c] for c in range(max_colors)) == 1)

    # Constraint: Adjacent vertices get different colors
    for u, v in edges:
        for c in range(max_colors):
            model.addConstr(x[u][c] + x[v][c] <= 1)

    # Constraint: If vertex i uses color c, then y[c] must be 1
    for i in range(n):
        for c in range(max_colors):
            model.addConstr(x[i][c] <= y[c])

    # Objective: Minimize the number of colors used
    model.setObjective(gp.quicksum(y[c] for c in range(max_colors)), GRB.MINIMIZE)
    
    return model, x, y, max_colors


def solve_graph_coloring_mip(n, edges, time_limit):
    model, x, y, max_colors = create_graph_coloring_model(n, edges)
    model.setParam('TimeLimit', time_limit)
    model.optimize()
    
    # Check if solution exists
    if model.status == GRB.INFEASIBLE:
        print("Model is infeasible")
        return None, float('inf')
    elif model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        # Extract solution
        coloring = {}
        color_classes = defaultdict(list)
        
        for i in range(n):
            for c in range(max_colors):
                if x[i][c].X > 0.5:  # If vertex i is assigned to color c
                    coloring[i] = c
                    color_classes[c].append(i)
                    break
        
        # Return best objective value found
        obj_val = int(model.objVal)
        
        # Convert color_classes to regular dictionary
        result_color_classes = {}
        for color, vertices in color_classes.items():
            if len(vertices) > 0:  # Only include colors that are actually used
                result_color_classes[color] = sorted(vertices)
        
        return coloring, obj_val, result_color_classes
    else:
        return None, float('inf')


def greedy_graph_coloring(n, adj_list):
    """
    A greedy coloring algorithm to get a feasible initial solution
    """
    coloring = [-1] * n  # Initialize all vertices with no color
    result = {}

    # Order vertices by descending degree (larger first)
    degrees = [(len(adj_list[i]), i) for i in range(n)]
    degrees.sort(reverse=True)
    ordered_vertices = [v for _, v in degrees]

    for vertex in ordered_vertices:
        # Find all colors used by neighbors
        used_colors = set()
        for neighbor in adj_list[vertex]:
            if coloring[neighbor] != -1:
                used_colors.add(coloring[neighbor])

        # Assign the smallest numbered color not used by any neighbor
        color = 0
        while color in used_colors:
            color += 1
        coloring[vertex] = color
        result[vertex] = color

    # Group vertices by color
    color_classes = defaultdict(list)
    for vertex, color in result.items():
        color_classes[color].append(vertex)

    # Convert to dict of lists
    color_classes_dict = {color: sorted(vertices) for color, vertices in color_classes.items()}

    return result, len(color_classes), color_classes_dict


def main():
    parser = argparse.ArgumentParser(description="Solve Graph Coloring Problem")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to the input instance JSON")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to save the output solution JSON")
    parser.add_argument("--time_limit", type=int, required=True, help="Maximum runtime in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log intermediate solutions (optional)")
    args = parser.parse_args()

    # Load instance
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)["graph"]
    
    n = instance_data["n"]
    edges = instance_data["edges"]
    adj_list = instance_data["adjacency_list"]
    
    # Initialize logger
    logger = None
    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize") 

    start_time = time.time()
    
    # Find initial solution with greedy coloring
    coloring, num_colors, color_classes = greedy_graph_coloring(n, adj_list)
    current_best = num_colors
    
    if logger:
        logger.log(current_best)
    
    remaining_time = args.time_limit - (time.time() - start_time)
    if remaining_time <= 0:
        # No more time left, return best solution found
        solution_out = {
            "objective_value": current_best,
            "coloring": {},
            "color_classes": {}
        }
        for vertex, color in coloring.items():
            solution_out["coloring"][str(vertex)] = color
        for color, vertices in color_classes.items():
            solution_out["color_classes"][str(color)] = vertices
            
        with open(args.solution_path, 'w') as f:
            json.dump(solution_out, f)
        return
    
    if gp is not None:
        # Use Gurobi if available to try to improve the solution
        try:
            mip_coloring, mip_obj, mip_color_classes = solve_graph_coloring_mip(
                n, edges, remaining_time
            )
            
            # Check if MIP solution is better than greedy solution
            if mip_obj < current_best and mip_coloring is not None:
                current_best = mip_obj
                coloring = mip_coloring
                color_classes = mip_color_classes
                
                if logger:
                    logger.log(current_best)
        
        except Exception as e:
            # If there's any issue with MIP, continue with greedy solution
            print(f"MIP solving failed: {e}")
    else:
        print("Gurobi not available, continuing with greedy solution...")

    # Final solution
    solution_out = {
        "objective_value": current_best,
        "coloring": {},
        "color_classes": {}
    }
    
    for vertex, color in coloring.items():
        solution_out["coloring"][str(vertex)] = color
    for color, vertices in color_classes.items():
        solution_out["color_classes"][str(color)] = vertices

    # Write solution
    with open(args.solution_path, 'w') as f:
        json.dump(solution_out, f)


if __name__ == "__main__":
    main()
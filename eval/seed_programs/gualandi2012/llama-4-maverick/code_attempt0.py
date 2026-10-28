import json
import argparse
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    return instance

def solve(instance, time_limit, log_path):
    num_vertices = instance['graph']['num_vertices']
    edges = instance['graph']['edges']

    # Initialize the logger
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None

    # Create a Gurobi model
    model = gp.Model("graph_coloring")
    model.Params.TimeLimit = time_limit

    # Variables: x[v, c] = 1 if vertex v is colored with color c
    max_colors = num_vertices
    x = model.addVars(num_vertices, max_colors, vtype=GRB.BINARY, name='x')

    # Variables: y[c] = 1 if color c is used
    y = model.addVars(max_colors, vtype=GRB.BINARY, name='y')

    # Objective: minimize the number of colors used
    model.setObjective(gp.quicksum(y[c] for c in range(max_colors)), GRB.MINIMIZE)

    # Constraints: each vertex is assigned exactly one color
    for v in range(num_vertices):
        model.addConstr(gp.quicksum(x[v, c] for c in range(max_colors)) == 1, name=f'vertex_coloring_{v}')

    # Constraints: adjacent vertices have different colors
    for u, v in edges:
        for c in range(max_colors):
            model.addConstr(x[u, c] + x[v, c] <= 1, name=f'edge_constraint_{u}_{v}_{c}')

    # Constraints: y[c] = 1 if any vertex is colored with c
    for c in range(max_colors):
        model.addConstr(y[c] >= gp.quicksum(x[v, c] for v in range(num_vertices)) / num_vertices, name=f'color_usage_{c}')
        model.addConstr(y[c] <= gp.quicksum(x[v, c] for v in range(num_vertices)), name=f'color_usage_upper_{c}')

    # Symmetry breaking: try to use lower colors first
    for c in range(max_colors - 1):
        model.addConstr(y[c] >= y[c + 1], name=f'symmetry_breaking_{c}')

    # Optimize the model
    model.optimize()

    # Log intermediate solutions
    if logger:
        for obj_value in model.getAttr('ObjVal'):
            logger.log(obj_value)

    # Extract the solution
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        coloring = {}
        color_classes = {}
        num_colors_used = 0
        for v in range(num_vertices):
            for c in range(max_colors):
                if x[v, c].x > 0.5:
                    coloring[f'{v}'] = c
                    if c not in color_classes:
                        color_classes[c] = []
                    color_classes[c].append(v)
                    num_colors_used = max(num_colors_used, c + 1)
        solution = {
            "objective_value": num_colors_used,
            "coloring": coloring,
            "color_classes": {f'{c}': [f'{v}' for v in color_classes[c]] for c in color_classes}
        }
    else:
        solution = None

    return solution

def main():
    parser = argparse.ArgumentParser(description='Graph Coloring Solver')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the instance JSON file')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to the solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds')
    parser.add_argument('--log_path', type=str, help='Path to the log JSONL file')
    args = parser.parse_args()

    instance = read_instance(args.instance_path)
    solution = solve(instance, args.time_limit, args.log_path)

    if solution:
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=4)
    else:
        print("No solution found.")

if __name__ == "__main__":
    main()
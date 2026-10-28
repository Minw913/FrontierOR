import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    return parser.parse_args()

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    return instance['graph']

def solve_graph_coloring(graph, time_limit, logger):
    n = graph['n']
    edges = graph['edges']
    adjacency_list = graph['adjacency_list']

    model = gp.Model()
    model.Params.TimeLimit = time_limit
    model.Params.LogToConsole = False

    # Variables: x[v, c] = 1 if vertex v is assigned color c, 0 otherwise
    max_colors = n
    x = model.addVars(n, max_colors, vtype=GRB.BINARY, name='x')

    # Variables: y[c] = 1 if color c is used, 0 otherwise
    y = model.addVars(max_colors, vtype=GRB.BINARY, name='y')

    # Objective: Minimize the number of colors used
    model.setObjective(gp.quicksum(y[c] for c in range(max_colors)), GRB.MINIMIZE)

    # Constraints: Each vertex is assigned exactly one color
    model.addConstrs((gp.quicksum(x[v, c] for c in range(max_colors)) == 1 for v in range(n)), name='assign_one_color')

    # Constraints: Adjacent vertices cannot have the same color
    for u, v in edges:
        u_idx, v_idx = u, v
        model.addConstrs((x[u_idx, c] + x[v_idx, c] <= 1 for c in range(max_colors)), name='adjacent_diff_colors')

    # Constraints: y[c] = 1 if any vertex is assigned color c
    for c in range(max_colors):
        model.addConstr(y[c] >= gp.quicksum(x[v, c] for v in range(n)), name='color_usage')

    # Symmetry breaking: Use colors in order
    for c in range(max_colors - 1):
        model.addConstr(y[c] >= y[c + 1], name='color_order')

    model.optimize()
    objective_value = model.ObjVal

    if logger and model.SolCount > 0:
        logger.log(objective_value)

    coloring = {}
    color_classes = {}
    if model.SolCount > 0:
        for v in range(n):
            for c in range(max_colors):
                if x[v, c].X > 0.5:
                    coloring[str(v)] = c
                    if str(c) not in color_classes:
                        color_classes[str(c)] = []
                    color_classes[str(c)].append(v)
        objective_value = len(set(coloring.values()))

    solution = {
        'objective_value': int(objective_value),
        'coloring': coloring,
        'color_classes': color_classes
    }
    return solution

def main():
    args = parse_args()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    graph = read_instance(args.instance_path)
    solution = solve_graph_coloring(graph, args.time_limit, logger)
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=4)

if __name__ == '__main__':
    main()
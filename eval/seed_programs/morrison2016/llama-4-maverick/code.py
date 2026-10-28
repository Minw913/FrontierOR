import argparse
import json
import gurobipy as gp
from gurobipy import GRB
import time
from solution_logger import SolutionLogger
def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        instance_data = json.load(f)
    return instance_data

def solve_graph_coloring(instance_data, time_limit, logger):
    num_vertices = instance_data['graph']['num_vertices']
    edges = instance_data['graph']['edges']
    clique_lower_bound = instance_data['bounds']['clique_lower_bound']
    greedy_upper_bound = instance_data['bounds']['greedy_upper_bound']

    model = gp.Model("graph_coloring")
    model.setParam("Seed", 0)
    model.setParam("MIPGap", 1e-4)
    model.setParam("NumericFocus", 0)
    model.setParam("Threads", 1)
    model.Params.TimeLimit = time_limit
    model.Params.LogToConsole = 0

    # Variables: x[v, c] = 1 if vertex v is colored with color c
    x = model.addVars(num_vertices, greedy_upper_bound, vtype=GRB.BINARY, name='x')

    # Variables: y[c] = 1 if color c is used
    y = model.addVars(greedy_upper_bound, vtype=GRB.BINARY, name='y')

    # Objective: minimize the number of colors used
    model.setObjective(gp.quicksum(y[c] for c in range(greedy_upper_bound)), GRB.MINIMIZE)

    # Constraints: each vertex is assigned exactly one color
    model.addConstrs((gp.quicksum(x[v, c] for c in range(greedy_upper_bound)) == 1 for v in range(num_vertices)), name='assign_color')

    # Constraints: adjacent vertices have different colors
    model.addConstrs((x[v, c] + x[u, c] <= 1 for (v, u) in edges for c in range(greedy_upper_bound)), name='adjacent_diff_color')

    # Constraints: y[c] = 1 if any vertex is colored with c
    model.addConstrs((x[v, c] <= y[c] for v in range(num_vertices) for c in range(greedy_upper_bound)), name='color_used')

    # Symmetry breaking: use colors in order
    model.addConstrs((y[c] >= y[c+1] for c in range(greedy_upper_bound-1)), name='symmetry_breaking')

    model.optimize()

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        objective_value = int(model.getObjective().getValue())
        coloring = {}
        selected_sets = [[] for _ in range(objective_value)]
        for v in range(num_vertices):
            for c in range(objective_value):
                if x[v, c].x > 0.5:
                    coloring[str(v)] = c
                    selected_sets[c].append(v)

        solution = {
            "objective_value": objective_value,
            "selected_sets": [sorted(vertices) for vertices in selected_sets],
            "coloring": coloring
        }

        if logger:
            logger.log(objective_value)

        return solution
    else:
        raise Exception("Failed to solve the problem")

def main():
    parser = argparse.ArgumentParser(description='Graph Coloring Solver')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the instance JSON file')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to write the solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds')
    parser.add_argument('--log_path', type=str, help='Path to log intermediate solutions')
    args = parser.parse_args()

    instance_data = read_instance(args.instance_path)
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    solution = solve_graph_coloring(instance_data, args.time_limit, logger)

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=4)

if __name__ == "__main__":
    main()
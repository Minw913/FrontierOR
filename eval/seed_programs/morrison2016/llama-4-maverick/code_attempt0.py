import argparse
import json
from gurobipy import Model, GRB, quicksum
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

    model = Model("Graph Coloring")
    model.Params.TimeLimit = time_limit

    # Variables: x[v, c] = 1 if vertex v is colored with color c
    x = model.addVars(num_vertices, greedy_upper_bound, vtype=GRB.BINARY, name='x')

    # Variables: y[c] = 1 if color c is used
    y = model.addVars(greedy_upper_bound, vtype=GRB.BINARY, name='y')

    # Objective: Minimize the number of colors used
    model.setObjective(quicksum(y[c] for c in range(greedy_upper_bound)), GRB.MINIMIZE)

    # Constraints: Each vertex is assigned exactly one color
    model.addConstrs((quicksum(x[v, c] for c in range(greedy_upper_bound)) == 1 for v in range(num_vertices)), name='assign_color')

    # Constraints: Vertices connected by an edge must have different colors
    for u, v in edges:
        model.addConstrs((x[u, c] + x[v, c] <= 1 for c in range(greedy_upper_bound)), name='edge_constraint')

    # Constraints: y[c] = 1 if any vertex is colored with c
    for c in range(greedy_upper_bound):
        model.addConstr(quicksum(x[v, c] for v in range(num_vertices)) <= num_vertices * y[c], name='color_usage')

    # Symmetry breaking: Use colors in order
    for c in range(greedy_upper_bound - 1):
        model.addConstr(y[c] >= y[c + 1], name='symmetry_breaking')

    model.optimize()

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        objective_value = int(model.getObjective().getValue())
        coloring = {}
        selected_sets = [[] for _ in range(objective_value)]
        for v in range(num_vertices):
            for c in range(greedy_upper_bound):
                if x[v, c].x > 0.5:
                    coloring[str(v)] = c
                    selected_sets[c].append(v)
                    break
        selected_sets = [sorted(vertices) for vertices in selected_sets[:objective_value]]

        solution = {
            "objective_value": objective_value,
            "selected_sets": selected_sets,
            "coloring": coloring
        }

        if logger:
            logger.log(objective_value)

        return solution
    else:
        raise Exception("No feasible solution found")

def main():
    parser = argparse.ArgumentParser(description='Graph Coloring Solver')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the instance JSON file')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to write the solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds')
    parser.add_argument('--log_path', type=str, help='Path to log intermediate solutions')
    args = parser.parse_args()

    instance_data = read_instance(args.instance_path)
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    start_time = time.time()
    solution = solve_graph_coloring(instance_data, args.time_limit, logger)
    end_time = time.time()

    print(f"Time taken: {end_time - start_time} seconds")

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=4)

if __name__ == "__main__":
    main()
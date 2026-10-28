import json
import argparse
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as file:
        instance_data = json.load(file)
    return instance_data

def solve(instance_data, time_limit, logger):
    num_nodes = instance_data['num_nodes']
    num_edges = instance_data['num_edges']
    edges = instance_data['edges']

    # Calculate the maximum degree of the graph to determine the upper bound for the number of colors needed
    max_degree = max(sum(1 for edge in edges if node in edge) for node in range(num_nodes))

    model = gp.Model('graph_coloring')
    model.setParam("Threads", 1)
    model.Params.TimeLimit = time_limit
    model.Params.LogToConsole = 0  # Disable Gurobi console output

    # Decision variables: x[n, c] = 1 if node n is colored with color c, 0 otherwise
    x = model.addVars(num_nodes, max_degree + 1, vtype=GRB.BINARY, name='x')

    # Decision variables: y[c] = 1 if color c is used, 0 otherwise
    y = model.addVars(max_degree + 1, vtype=GRB.BINARY, name='y')

    # Objective: minimize the total number of colors used
    model.setObjective(gp.quicksum(y[c] for c in range(max_degree + 1)), GRB.MINIMIZE)

    # Constraints: each node must be assigned exactly one color
    model.addConstrs((gp.quicksum(x[n, c] for c in range(max_degree + 1)) == 1 for n in range(num_nodes)), name='assign_one_color')

    # Constraints: adjacent nodes cannot have the same color
    model.addConstrs((x[n1, c] + x[n2, c] <= 1 for (n1, n2) in edges for c in range(max_degree + 1)), name='adjacent_nodes_different_colors')

    # Constraints: if a color c is used by any node, then y[c] must be 1
    model.addConstrs((x[n, c] <= y[c] for n in range(num_nodes) for c in range(max_degree + 1)), name='color_usage')

    model.optimize()

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        coloring = {str(n): int(c) for (n, c) in [(n, c) for n in range(num_nodes) for c in range(max_degree + 1) if x[n, c].X > 0.5]}
        objective_value = model.getObjective().getValue()
        if logger:
            logger.log(objective_value)
        return coloring, objective_value
    else:
        raise Exception(f"Gurobi model failed to find a feasible solution. Status code: {model.status}")

def write_solution(solution_path, coloring, objective_value):
    solution = {
        "objective_value": float(objective_value),
        "coloring": coloring
    }
    with open(solution_path, 'w') as file:
        json.dump(solution, file)

def main():
    parser = argparse.ArgumentParser(description='Graph Coloring Solver')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the instance JSON file')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to write the solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds')
    parser.add_argument('--log_path', type=str, help='Path to log intermediate solutions (optional)')
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    instance_data = read_instance(args.instance_path)
    coloring, objective_value = solve(instance_data, args.time_limit, logger)
    write_solution(args.solution_path, coloring, objective_value)

if __name__ == '__main__':
    main()
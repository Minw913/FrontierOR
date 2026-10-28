import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def parse_args():
    parser = argparse.ArgumentParser(description='Maximum Clique Solver')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the instance JSON file')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to write the solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Maximum runtime in seconds')
    parser.add_argument('--log_path', type=str, help='Path to log intermediate solutions (optional)')
    return parser.parse_args()

def read_instance(instance_path):
    with open(instance_path, 'r') as file:
        return json.load(file)

def solve_max_clique(instance_data, time_limit, logger):
    num_vertices = instance_data['num_vertices']
    edges = instance_data['edges']
    adjacency_list = instance_data['adjacency_list']

    # Create a Gurobi model
    model = gp.Model('MaxClique')
    model.Params.LogToConsole = 0  # Disable console logging

    # Create binary variables for vertex selection
    x = model.addVars(num_vertices, vtype=GRB.BINARY, name='x')

    # Objective: Maximize the number of selected vertices
    model.setObjective(gp.quicksum(x[v] for v in range(num_vertices)), GRB.MAXIMIZE)

    # Constraints: For every pair of non-adjacent vertices, at most one can be selected
    for edge in edges:
        u, v = edge
        model.addConstr(x[u] + x[v] <= 1, f'c_{u}_{v}')

    # Consider adding constraints for all non-adjacent pairs for completeness
    for u in range(num_vertices):
        for v in range(u + 1, num_vertices):
            if str(v) not in adjacency_list[str(u)] and str(u) not in adjacency_list[str(v)]:
                model.addConstr(x[u] + x[v] <= 1, f'c_{u}_{v}')

    # Set time limit
    model.Params.TimeLimit = time_limit

    # Optimize the model
    model.optimize()

    # Extract the solution
    objective_value = model.getObjective().getValue()
    clique_vertices = [v for v in range(num_vertices) if x[v].x > 0.5]

    # Log intermediate solutions if logger is provided
    if logger:
        for i in range(model.SolCount):
            model.Params.SolutionNumber = i
            obj_val = model.getObjective().getValue()
            logger.log(obj_val)

    return objective_value, clique_vertices

def write_solution(solution_path, objective_value, clique_vertices):
    solution = {
        'objective_value': objective_value,
        'clique_vertices': clique_vertices
    }
    with open(solution_path, 'w') as file:
        json.dump(solution, file)

def main():
    args = parse_args()
    instance_data = read_instance(args.instance_path)
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None
    objective_value, clique_vertices = solve_max_clique(instance_data, args.time_limit, logger)
    write_solution(args.solution_path, objective_value, clique_vertices)

if __name__ == '__main__':
    main()
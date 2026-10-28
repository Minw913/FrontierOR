import json
import argparse
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as file:
        instance = json.load(file)
    return instance

def solve(instance, time_limit, log_path):
    num_vertices = instance['num_vertices']
    adjacency_list = instance['adjacency_list']

    model = gp.Model()
    model.Params.TimeLimit = time_limit
    model.Params.Threads = 1  # Ensure single-threaded execution

    # Create variables
    x = model.addVars(num_vertices, vtype=GRB.BINARY, name='x')

    # Objective: Maximize the number of selected vertices
    model.setObjective(gp.quicksum(x), GRB.MAXIMIZE)

    # Constraints: For each pair of vertices not connected by an edge, at most one can be selected
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            if j not in adjacency_list[i]:
                model.addConstr(x[i] + x[j] <= 1, f'non_adjacent_{i}_{j}')

    # Optimize the model
    model.optimize()

    # Extract the solution
    if model.SolCount > 0:
        objective_value = int(model.getObjective().getValue())
        clique_vertices = [i for i in range(num_vertices) if x[i].X > 0.5]
    else:
        objective_value = 0
        clique_vertices = []

    # Logging intermediate solutions
    logger = SolutionLogger(log_path, sense="maximize") if log_path else None
    if logger and model.SolCount > 0:
        logger.log(objective_value)

    return objective_value, clique_vertices

def write_solution(solution_path, objective_value, clique_vertices):
    solution = {
        'objective_value': objective_value,
        'clique_vertices': clique_vertices
    }
    with open(solution_path, 'w') as file:
        json.dump(solution, file)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', required=False)
    args = parser.parse_args()

    instance = read_instance(args.instance_path)
    objective_value, clique_vertices = solve(instance, args.time_limit, args.log_path)
    write_solution(args.solution_path, objective_value, clique_vertices)

if __name__ == '__main__':
    main()
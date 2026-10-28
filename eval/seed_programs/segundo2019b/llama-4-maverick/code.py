import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        instance_data = json.load(f)
    return instance_data

def solve_max_weight_clique(instance_data, time_limit, log_path):
    graph = instance_data['graph']
    num_vertices = graph['num_vertices']
    edges = graph['edges']
    vertex_weights = instance_data['vertex_weights']

    model = gp.Model('max_weight_clique')
    model.setParam("Threads", 1)
    model.Params.LogToConsole = 0  # Silence Gurobi output
    
    # Create variables
    x = model.addVars(num_vertices, vtype=GRB.BINARY, name='x')
    
    # Objective function: maximize the sum of weights of selected vertices
    model.setObjective(gp.quicksum(vertex_weights[i] * x[i] for i in range(num_vertices)), GRB.MAXIMIZE)
    
    # Constraints: for every pair of vertices not connected by an edge, at most one can be selected
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            if [i, j] not in edges and [j, i] not in edges:
                model.addConstr(x[i] + x[j] <= 1, f'edge_constraint_{i}_{j}')
    
    # Initialize logger for intermediate solutions
    logger = SolutionLogger(log_path, sense="maximize") if log_path else None
    
    # Optimize the model within the given time limit
    model.Params.TimeLimit = time_limit
    model.optimize()
    
    # Extract solution
    if model.SolCount > 0:
        objective_value = model.getObjective().getValue()
        clique_vertices = [i for i in range(num_vertices) if x[i].x > 0.5]
    else:
        objective_value = 0
        clique_vertices = []
    
    # Log the final solution if logger is available
    if logger:
        logger.log(objective_value)
    
    return objective_value, clique_vertices

def write_solution(solution_path, objective_value, clique_vertices):
    solution = {
        'objective_value': objective_value,
        'clique_vertices': clique_vertices
    }
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=4)

def main():
    parser = argparse.ArgumentParser(description='Solve Max Weight Clique Problem')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the instance JSON file')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to write the solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds')
    parser.add_argument('--log_path', type=str, help='Path to log intermediate solutions (optional)')
    args = parser.parse_args()
    
    instance_data = read_instance(args.instance_path)
    objective_value, clique_vertices = solve_max_weight_clique(instance_data, args.time_limit, args.log_path)
    write_solution(args.solution_path, objective_value, clique_vertices)

if __name__ == '__main__':
    main()
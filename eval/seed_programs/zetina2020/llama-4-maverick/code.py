import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
import time

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, required=False)
    return parser.parse_args()

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        instance_data = json.load(f)
    return instance_data

def solve_tsp(instance_data, time_limit, logger):
    n_cities = instance_data['parameters']['n_cities']
    distance_matrix = instance_data['data']['distance_matrix']

    model = gp.Model()
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = time_limit

    x = model.addVars(n_cities, n_cities, vtype=GRB.BINARY, name='x')
    model.setObjective(gp.quicksum(distance_matrix[i][j] * x[i, j] for i in range(n_cities) for j in range(n_cities)), GRB.MINIMIZE)

    for i in range(n_cities):
        model.addConstr(gp.quicksum(x[i, j] for j in range(n_cities)) == 1)
        model.addConstr(gp.quicksum(x[j, i] for j in range(n_cities)) == 1)

    model.optimize()

    if model.SolCount > 0:
        solution_dict = {f'x_{i}_{j}': x[i, j].X for i in range(n_cities) for j in range(n_cities)}
        tour = [0] * n_cities
        current_city = 0
        for j in range(n_cities):
            for i in range(n_cities):
                if x[current_city, i].X > 0.5:
                    tour[j] = current_city
                    current_city = i
                    break
        objective_value = model.ObjVal + distance_matrix[tour[-1]][tour[0]]
        solution_dict = {f'x_{city}_{time_step}': 1 if tour[time_step] == city else 0 for city in range(n_cities) for time_step in range(n_cities)}
        if logger:
            logger.log_solution(objective_value, solution_dict)
        return objective_value, solution_dict, tour
    else:
        return None, None, None

def solve_maxcut(instance_data, time_limit, logger):
    n_nodes = len(instance_data['data']['edges'])
    edges = instance_data['data']['edges']

    model = gp.Model()
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = time_limit

    x = model.addVars(n_nodes, vtype=GRB.BINARY, name='x')
    model.setObjective(gp.quicksum(1 - x[i] - x[j] + 2*x[i]*x[j] for i, j in edges), GRB.MAXIMIZE)

    model.optimize()

    if model.SolCount > 0:
        solution_dict = {f'x_{i}': x[i].X for i in range(n_nodes)}
        objective_value = model.ObjVal
        if logger:
            logger.log_solution(objective_value, solution_dict)
        return objective_value, solution_dict
    else:
        return None, None

def main():
    args = parse_args()
    instance_data = read_instance(args.instance_path)

    logger = SolutionLogger(args.log_path, sense="minimize" if instance_data['problem_type'] == 'TSP' else "maximize") if args.log_path else None

    start_time = time.time()
    if instance_data['problem_type'] == 'TSP':
        objective_value, solution_dict, tour = solve_tsp(instance_data, args.time_limit, logger)
        if objective_value is not None:
            output = {
                'objective_value': objective_value,
                'solution': solution_dict,
                'tour': tour
            }
        else:
            output = None
    else:
        objective_value, solution_dict = solve_maxcut(instance_data, args.time_limit, logger)
        if objective_value is not None:
            output = {
                'objective_value': objective_value,
                'solution': solution_dict
            }
        else:
            output = None

    if output is not None:
        with open(args.solution_path, 'w') as f:
            json.dump(output, f)

if __name__ == '__main__':
    main()
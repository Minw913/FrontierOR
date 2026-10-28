import json
import argparse
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as file:
        instance = json.load(file)
    return instance

def solve_instance(instance, time_limit, logger):
    n = instance['n']
    capacity = instance['capacity']
    weights = instance['weights']
    profit_matrix = instance['profit_matrix']

    model = gp.Model()
    model.Params.TimeLimit = time_limit
    model.Params.LogToConsole = False

    # Decision variables
    x = model.addVars(n, vtype=GRB.BINARY, name='x')

    # Objective function
    objective = gp.quicksum(profit_matrix[i][i] * x[i] for i in range(n))
    for i in range(n):
        for j in range(i + 1, n):
            objective += profit_matrix[i][j] * x[i] * x[j]
    model.setObjective(objective, GRB.MAXIMIZE)

    # Capacity constraint
    model.addConstr(gp.quicksum(weights[i] * x[i] for i in range(n)) <= capacity)

    # Optimize the model
    model.optimize()

    # Log intermediate solutions
    if logger:
        for i in range(model.SolCount):
            model.Params.SolutionNumber = i
            objective_value = model.PoolObjVal
            logger.log(objective_value)

    # Extract the best solution
    if model.SolCount > 0:
        solution = [int(x[i].X) for i in range(n)]
        objective_value = model.ObjVal
    else:
        solution = [0] * n
        objective_value = 0.0

    return objective_value, solution

def write_solution(solution_path, objective_value, solution):
    solution_data = {
        'objective_value': objective_value,
        'solution': solution
    }
    with open(solution_path, 'w') as file:
        json.dump(solution_data, file)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, required=False)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    instance = read_instance(args.instance_path)
    objective_value, solution = solve_instance(instance, args.time_limit, logger)
    write_solution(args.solution_path, objective_value, solution)

if __name__ == '__main__':
    main()
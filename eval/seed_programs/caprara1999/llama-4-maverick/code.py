import json
import argparse
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as file:
        instance = json.load(file)
    return instance

def solve(instance, time_limit, logger):
    n = instance['n']
    capacity = instance['capacity']
    weights = instance['weights']
    profit_matrix = instance['profit_matrix']

    model = gp.Model()
    model.setParam("Threads", 1)
    model.Params.LogToConsole = 0  # Suppress Gurobi console output
    model.Params.TimeLimit = time_limit  # Set time limit

    # Decision variables: whether each item is selected
    x = model.addVars(n, vtype=GRB.BINARY, name='x')

    # Objective function: maximize total profit
    objective = gp.quicksum(profit_matrix[i][i] * x[i] for i in range(n))
    for i in range(n):
        for j in range(i+1, n):
            objective += profit_matrix[i][j] * x[i] * x[j]
    model.setObjective(objective, GRB.MAXIMIZE)

    # Capacity constraint
    model.addConstr(gp.quicksum(weights[i] * x[i] for i in range(n)) <= capacity, name='capacity')

    # Optimize the model
    model.optimize()

    # Extract solution
    objective_value = model.ObjVal
    selected_items = [int(x[i].X) for i in range(n)]

    # Log intermediate solutions
    if logger:
        for sol_idx in range(model.SolCount):
            model.setParam(GRB.Param.SolutionNumber, sol_idx)
            logger.log(model.getAttr('PoolObjVal'))

    return objective_value, selected_items

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, required=False)
    args = parser.parse_args()

    instance = read_instance(args.instance_path)
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    objective_value, selected_items = solve(instance, args.time_limit, logger)

    # Prepare and write solution
    solution = {
        'objective_value': objective_value,
        'selected_items': selected_items
    }
    with open(args.solution_path, 'w') as file:
        json.dump(solution, file)

if __name__ == '__main__':
    main()
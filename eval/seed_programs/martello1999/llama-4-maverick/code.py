import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    return instance

def solve_knapsack(instance, time_limit, log_path):
    n = instance['n']
    capacity = instance['capacity']
    items = instance['items']

    # Create a Gurobi model
    model = gp.Model('Knapsack')
    model.setParam("Threads", 1)
    model.Params.TimeLimit = time_limit
    model.Params.LogToConsole = 0  # Disable console logging

    # Create variables
    x = model.addVars([item['id'] for item in items], vtype=GRB.BINARY, name='x')

    # Objective function: Maximize total profit
    model.setObjective(gp.quicksum(item['profit'] * x[item['id']] for item in items), GRB.MAXIMIZE)

    # Constraint: Total weight not to exceed capacity
    model.addConstr(gp.quicksum(item['weight'] * x[item['id']] for item in items) <= capacity, 'Capacity')

    # Initialize logger
    logger = SolutionLogger(log_path, sense="maximize") if log_path else None

    # Optimize the model
    model.optimize(lambda model, where: callback(model, where, logger))

    # Extract solution
    objective_value = model.getObjective().getValue()
    selected_items = [item_id for item_id, var in x.items() if var.x > 0.5]

    return objective_value, selected_items

def callback(model, where, logger):
    if where == GRB.Callback.MIPSOL:
        objective_value = model.cbGet(GRB.Callback.MIPSOL_OBJ)
        if logger:
            logger.log(objective_value)

def write_solution(solution_path, objective_value, selected_items):
    solution = {
        'objective_value': int(objective_value),
        'selected_items': selected_items
    }
    with open(solution_path, 'w') as f:
        json.dump(solution, f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, required=False)
    args = parser.parse_args()

    instance = read_instance(args.instance_path)
    objective_value, selected_items = solve_knapsack(instance, args.time_limit, args.log_path)
    write_solution(args.solution_path, objective_value, selected_items)

if __name__ == '__main__':
    main()
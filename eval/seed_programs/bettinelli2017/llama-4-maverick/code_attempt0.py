import json
import argparse
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    return instance

def solve(instance, time_limit, log_path):
    num_items = instance['num_items']
    capacity = instance['capacity']
    profits = instance['items']['profits']
    weights = instance['items']['weights']
    edges = instance['conflict_graph']['edges']

    # Create a Gurobi model
    model = gp.Model('knapsack_with_conflicts')
    model.Params.LogToConsole = 0  # Silence Gurobi output

    # Create variables: x[i] = 1 if item i is selected, 0 otherwise
    x = model.addVars(num_items, vtype=GRB.BINARY, name='x')

    # Objective: Maximize total profit
    model.setObjective(gp.quicksum(profits[i] * x[i] for i in range(num_items)), GRB.MAXIMIZE)

    # Constraint: Total weight not exceeding capacity
    model.addConstr(gp.quicksum(weights[i] * x[i] for i in range(num_items)) <= capacity, 'capacity_constraint')

    # Constraints: At most one item from each conflicting pair
    for edge in edges:
        i, j = edge
        model.addConstr(x[i] + x[j] <= 1, f'conflict_{i}_{j}')

    # Set time limit
    model.Params.TimeLimit = time_limit

    # Initialize logger
    logger = SolutionLogger(log_path, sense='maximize') if log_path else None

    # Optimize the model
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            objective_value = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(objective_value)

    model.optimize(callback)

    # Extract solution
    if model.SolCount > 0:
        objective_value = model.ObjVal
        selected_items = [i for i in range(num_items) if x[i].x > 0.5]
    else:
        objective_value = 0.0
        selected_items = []

    return objective_value, selected_items

def write_solution(solution_path, objective_value, selected_items):
    solution = {
        'objective_value': objective_value,
        'selected_items': selected_items
    }
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=4)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str)
    args = parser.parse_args()

    instance = read_instance(args.instance_path)
    objective_value, selected_items = solve(instance, args.time_limit, args.log_path)
    write_solution(args.solution_path, objective_value, selected_items)

if __name__ == '__main__':
    main()
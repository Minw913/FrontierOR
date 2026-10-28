import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    return instance

def solve(instance, time_limit, log_path=None):
    n = instance['n']
    m = instance['m']
    profits = instance['profits']
    pairwise_profits = instance['pairwise_profits']
    weights = instance['weights']
    capacities = instance['capacities']

    model = gp.Model('KnapsackProblem')
    model.setParam("Threads", 1)
    model.Params.LogToConsole = 0  # Silence Gurobi output

    # Variables
    x = model.addVars(n, m, vtype=GRB.BINARY, name='x')  # x[i, k] = 1 if item i is assigned to knapsack k
    y = model.addVars(n, vtype=GRB.BINARY, name='y')  # y[i] = 1 if item i is assigned to any knapsack

    # Objective: Maximize total profit
    objective = gp.quicksum(profits[i] * y[i] for i in range(n)) + \
                 gp.quicksum(0.5 * pairwise_profits[i][j] * x[i, k] * x[j, k] for i in range(n) for j in range(i+1, n) for k in range(m))
    model.setObjective(objective, GRB.MAXIMIZE)

    # Constraints
    # Each item is assigned to at most one knapsack
    model.addConstrs((gp.quicksum(x[i, k] for k in range(m)) <= 1 for i in range(n)), name='assign_at_most_one')
    
    # Link y[i] to x[i, k]
    model.addConstrs((y[i] == gp.quicksum(x[i, k] for k in range(m)) for i in range(n)), name='link_y_to_x')

    # Knapsack capacity constraints
    model.addConstrs((gp.quicksum(weights[i] * x[i, k] for i in range(n)) <= capacities[k] for k in range(m)), name='knapsack_capacity')

    # Set time limit
    model.Params.TimeLimit = time_limit

    # Initialize logger
    logger = SolutionLogger(log_path, sense="maximize") if log_path else None

    # Optimize
    def callback(model, where):
        if where == GRB.Callback.MIP:
            obj = model.cbGet(GRB.Callback.MIP_OBJBST)
            if logger:
                logger.log(obj)

    model.optimize(callback)

    # Extract solution
    assignment = []
    objective_value = model.getObjective().getValue()
    for i in range(n):
        for k in range(m):
            if x[i, k].X > 0.5:
                assignment.append([i, k])

    return objective_value, assignment

def write_solution(solution_path, objective_value, assignment):
    solution = {
        'objective_value': objective_value,
        'assignment': assignment
    }
    with open(solution_path, 'w') as f:
        json.dump(solution, f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    instance = read_instance(args.instance_path)
    objective_value, assignment = solve(instance, args.time_limit, args.log_path)
    write_solution(args.solution_path, objective_value, assignment)

if __name__ == '__main__':
    main()
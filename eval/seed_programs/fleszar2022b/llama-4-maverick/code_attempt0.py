import json
import argparse
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    return instance

def solve_instance(instance, time_limit, log_path):
    n = instance['n']
    m = instance['m']
    capacity = instance['capacity']
    weights = instance['weights']
    values = instance['values']
    pairwise_values = instance['pairwise_values']

    model = Model("Knapsack Problem")
    model.Params.TimeLimit = time_limit
    model.Params.LogToConsole = False  # Disable console logging

    # Variables
    x = model.addVars(n, m, vtype=GRB.BINARY, name="x")  # 1 if item i is in knapsack j
    y = model.addVars(n, vtype=GRB.BINARY, name="y")  # 1 if item i is packed

    # Objective function
    model.setObjective(
        quicksum(values[i] * x[i, j] for i in range(n) for j in range(m)) + 
        quicksum(0.5 * pairwise_values[i][k] * x[i, j] * x[k, j] for i in range(n) for k in range(i+1, n) for j in range(m)),
        GRB.MAXIMIZE
    )

    # Constraints
    for i in range(n):
        model.addConstr(quicksum(x[i, j] for j in range(m)) <= y[i])  # Item i is in at most one knapsack
    for j in range(m):
        model.addConstr(quicksum(weights[i] * x[i, j] for i in range(n)) <= capacity)  # Knapsack capacity constraint

    # Solve the model
    logger = SolutionLogger(log_path, sense="maximize") if log_path else None

    def callback(model, where):
        if where == GRB.Callback.MIP:
            objbst = model.cbGet(GRB.Callback.MIP_OBJBST)
            if logger:
                logger.log(objbst)

    model.optimize(callback)

    # Extract solution
    assignment = {}
    objective_value = model.getObjective().getValue() if model.SolCount > 0 else 0
    if model.SolCount > 0:
        for i in range(n):
            for j in range(m):
                if x[i, j].X > 0.5:
                    assignment[str(i)] = j

    return objective_value, assignment

def main():
    parser = argparse.ArgumentParser(description='Solve Knapsack Problem')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the instance JSON file')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to the solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds')
    parser.add_argument('--log_path', type=str, help='Path to the log JSONL file')
    args = parser.parse_args()

    instance = read_instance(args.instance_path)
    objective_value, assignment = solve_instance(instance, args.time_limit, args.log_path)

    solution = {
        "objective_value": objective_value,
        "assignment": assignment
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=4)

if __name__ == "__main__":
    main()
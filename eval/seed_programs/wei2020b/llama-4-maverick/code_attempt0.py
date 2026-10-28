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
    W = instance['W']
    V = instance['V']
    items = instance['items']

    model = gp.Model()
    model.Params.TimeLimit = time_limit
    model.Params.LogToConsole = False  # Disable Gurobi logging to console

    # Variables
    x = model.addVars(n, n, vtype=GRB.BINARY, name='x')
    y = model.addVars(n, vtype=GRB.BINARY, name='y')

    # Constraints
    model.addConstrs((gp.quicksum(x[i, j] for j in range(n)) == 1 for i in range(n)), name='assign_item')
    model.addConstrs((gp.quicksum(items[i]['w'] * x[i, j] for i in range(n)) <= W * y[j] for j in range(n)), name='weight_limit')
    model.addConstrs((gp.quicksum(items[i]['v'] * x[i, j] for i in range(n)) <= V * y[j] for j in range(n)), name='volume_limit')
    model.addConstrs((x[i, j] <= y[j] for i in range(n) for j in range(n)), name='item_in_used_bin')

    # Symmetry breaking constraint to improve performance
    model.addConstrs((y[j] <= gp.quicksum(y[k] for k in range(j)) for j in range(1, n)), name='symmetry_breaking')

    # Objective
    model.setObjective(gp.quicksum(y[j] for j in range(n)), GRB.MINIMIZE)

    # Optimize
    model.optimize()
    objective_value = model.getObjective().getValue()

    if logger and model.SolCount > 0:
        logger.log(int(objective_value))

    # Extract solution
    bins = []
    if model.SolCount > 0:
        for j in range(n):
            if y[j].x > 0.5:
                bin_items = [i for i in range(n) if x[i, j].x > 0.5]
                total_weight = sum(items[i]['w'] for i in bin_items)
                total_volume = sum(items[i]['v'] for i in bin_items)
                bins.append({
                    'items': bin_items,
                    'total_weight': total_weight,
                    'total_volume': total_volume
                })

    solution = {
        'objective_value': int(objective_value),
        'bins': bins
    }

    return solution

def write_solution(solution_path, solution):
    with open(solution_path, 'w') as file:
        json.dump(solution, file, indent=4)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, required=False)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    instance = read_instance(args.instance_path)
    solution = solve(instance, args.time_limit, logger)
    write_solution(args.solution_path, solution)

    if logger:
        logger.close()

if __name__ == "__main__":
    main()
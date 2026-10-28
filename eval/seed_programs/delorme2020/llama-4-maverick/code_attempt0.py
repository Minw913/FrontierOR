import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    return instance

def solve(instance, time_limit, logger):
    n = instance['n']
    m = instance['m']
    c = instance['c']
    weights = instance['weights']
    demands = instance['demands']

    model = gp.Model()
    model.Params.LogToConsole = 0  # Suppress Gurobi logging
    model.Params.TimeLimit = time_limit

    # Variables: x_ij = number of items of type i packed into bin j
    x = model.addVars(m, n, vtype=GRB.INTEGER, name='x')
    # Variables: y_j = 1 if bin j is used, 0 otherwise
    y = model.addVars(n, vtype=GRB.BINARY, name='y')

    # Objective: minimize the number of bins used
    model.setObjective(gp.quicksum(y[j] for j in range(n)), GRB.MINIMIZE)

    # Constraints: sum of widths of items in each bin does not exceed capacity
    for j in range(n):
        model.addConstr(gp.quicksum(weights[i] * x[i, j] for i in range(m)) <= c * y[j], name=f'capacity_{j}')

    # Constraints: demand for each item type is met
    for i in range(m):
        model.addConstr(gp.quicksum(x[i, j] for j in range(n)) == demands[i], name=f'demand_{i}')

    # Symmetry breaking: items are packed into bins in order
    for j in range(n - 1):
        model.addConstr(y[j] >= y[j + 1], name=f'symmetry_{j}')

    model.optimize()

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        objective_value = model.getObjective().getValue()
        if logger:
            logger.log(objective_value)

        bin_assignments = []
        x_values = model.getAttr('x', x)
        y_values = model.getAttr('x', y)
        used_bins = [j for j in range(n) if y_values[j] > 0.5]
        for j in used_bins:
            bin_content = []
            for i in range(m):
                num_items = int(x_values[i, j])
                bin_content.extend([weights[i]] * num_items)
            bin_assignments.append(bin_content)

        solution = {
            'objective_value': objective_value,
            'num_bins': round(objective_value),
            'bin_assignments': bin_assignments
        }
        return solution
    else:
        raise Exception(f'Gurobi model status: {model.status}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense='minimize') if args.log_path else None

    instance = read_instance(args.instance_path)
    solution = solve(instance, args.time_limit, logger)
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=4)

if __name__ == '__main__':
    main()
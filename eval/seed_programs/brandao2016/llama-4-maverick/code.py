import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    return parser.parse_args()

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        data = json.load(f)
    return data

def solve(instance_data, time_limit, logger):
    params = instance_data['parameters']
    n = params['n']
    m = params['m']
    W = params['W']
    items = instance_data['items']

    model = gp.Model()
    model.setParam("Threads", 1)
    model.Params.LogToConsole = 0
    model.Params.TimeLimit = time_limit

    # Variables
    num_bins = model.addVar(vtype=GRB.INTEGER, name='num_bins')
    x = model.addVars(m, num_bins.upper_bound, vtype=GRB.INTEGER, name='x')
    y = model.addVars(num_bins.upper_bound, vtype=GRB.BINARY, name='y')
    z = model.addVars(m, num_bins.upper_bound, vtype=GRB.BINARY, name='z')

    # Constraints
    for i in range(m):
        model.addConstr(gp.quicksum(x[i, j] for j in range(num_bins.upper_bound)) >= items[i]['demand'])
    for j in range(num_bins.upper_bound):
        model.addConstr(gp.quicksum(items[i]['weight'] * x[i, j] for i in range(m)) <= W * y[j])
        for i in range(m):
            model.addConstr(x[i, j] <= n * z[i, j])
            model.addConstr(z[i, j] <= y[j])
    for j in range(num_bins.upper_bound - 1):
        model.addConstr(y[j] >= y[j + 1])

    # Objective
    model.setObjective(num_bins, GRB.MINIMIZE)

    # Linking num_bins to y variables
    model.addConstr(num_bins == gp.quicksum(y[j] for j in range(num_bins.upper_bound)))

    # Initial upper bound for num_bins
    num_bins.upper_bound = n

    model.optimize()

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        objective_value = model.getObjective().getValue()
        num_bins_used = int(objective_value)
        bins = []
        for j in range(num_bins_used):
            bin_items = []
            for i in range(m):
                if x[i, j].x > 0.5:
                    bin_items.extend([items[i]['item_type']] * int(x[i, j].x))
            bins.append(bin_items)
        return objective_value, num_bins_used, bins
    else:
        raise Exception('No solution found')

def main():
    args = parse_args()
    instance_data = read_instance(args.instance_path)
    logger = SolutionLogger(args.log_path, sense='minimize') if args.log_path else None

    # Solve the problem
    try:
        objective_value, num_bins, bins = solve(instance_data, args.time_limit, logger)
        solution = {
            'objective_value': objective_value,
            'num_bins': num_bins,
            'bins': bins
        }
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=4)
    except Exception as e:
        print(f'An error occurred: {e}')

if __name__ == '__main__':
    main()
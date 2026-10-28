import json
import argparse
from gurobipy import Model, GRB, quicksum
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
        return json.load(f)

def solve(instance):
    n = instance['n']
    c = instance['c']
    weights = instance['weights']

    model = Model('BinPacking')
    model.Params.LogToConsole = 0  # Disable Gurobi console output

    # Variables
    x = model.addVars(n, n, vtype=GRB.BINARY, name='x')  # x[i, k] = 1 if item i is in bin k
    y = model.addVars(n, vtype=GRB.BINARY, name='y')  # y[k] = 1 if bin k is used

    # Objective: Minimize the number of bins used
    model.setObjective(quicksum(y[k] for k in range(n)), GRB.MINIMIZE)

    # Constraints
    # Each item is assigned to exactly one bin
    model.addConstrs((quicksum(x[i, k] for k in range(n)) == 1 for i in range(n)), name='assign_item')

    # The total weight of items in a bin does not exceed its capacity
    model.addConstrs((quicksum(weights[i] * x[i, k] for i in range(n)) <= c * y[k] for k in range(n)), name='bin_capacity')

    # Incompatible items are not in the same bin
    if 'incompatibilities' in instance:
        incompatibilities = instance['incompatibilities']
        model.addConstrs((x[i, k] + x[j, k] <= 1 for (i, j) in incompatibilities for k in range(n)), name='incompatibility')

    # Symmetry breaking: y[k] >= y[k+1]
    model.addConstrs((y[k] >= y[k+1] for k in range(n-1)), name='symmetry_breaking')

    # Linking x and y: x[i, k] <= y[k]
    model.addConstrs((x[i, k] <= y[k] for i in range(n) for k in range(n)), name='linking')

    model.update()
    model.optimize()

    # Extract solution
    bins = []
    for k in range(n):
        bin_items = [i for i in range(n) if x[i, k].X > 0.5]
        if bin_items:
            bins.append(bin_items)

    return len(bins), bins

def main():
    args = parse_args()
    instance = read_instance(args.instance_path)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    model = Model('BinPacking')
    model.Params.LogToConsole = 0  # Disable Gurobi console output
    model.Params.TimeLimit = args.time_limit

    n = instance['n']
    c = instance['c']
    weights = instance['weights']

    # Variables
    x = model.addVars(n, n, vtype=GRB.BINARY, name='x')  # x[i, k] = 1 if item i is in bin k
    y = model.addVars(n, vtype=GRB.BINARY, name='y')  # y[k] = 1 if bin k is used

    # Objective: Minimize the number of bins used
    model.setObjective(quicksum(y[k] for k in range(n)), GRB.MINIMIZE)

    # Constraints
    # Each item is assigned to exactly one bin
    model.addConstrs((quicksum(x[i, k] for k in range(n)) == 1 for i in range(n)), name='assign_item')

    # The total weight of items in a bin does not exceed its capacity
    model.addConstrs((quicksum(weights[i] * x[i, k] for i in range(n)) <= c * y[k] for k in range(n)), name='bin_capacity')

    # Incompatible items are not in the same bin
    if 'incompatibilities' in instance:
        incompatibilities = instance['incompatibilities']
        model.addConstrs((x[i, k] + x[j, k] <= 1 for (i, j) in incompatibilities for k in range(n)), name='incompatibility')

    # Symmetry breaking: y[k] >= y[k+1]
    model.addConstrs((y[k] >= y[k+1] for k in range(n-1)), name='symmetry_breaking')

    # Linking x and y: x[i, k] <= y[k]
    model.addConstrs((x[i, k] <= y[k] for i in range(n) for k in range(n)), name='linking')

    model.update()

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_value = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj_value)

    model.optimize(callback)

    # Extract solution
    x = model.getAttr('X', model.getVars()[:n*n])
    x = {i: x[i] for i in range(n*n)}
    x = {(i, k): x[i*n + k] for i in range(n) for k in range(n)}

    bins = []
    for k in range(n):
        bin_items = [i for i in range(n) if x[i, k] > 0.5]
        if bin_items:
            bins.append(bin_items)

    solution = {
        "objective_value": model.objVal,
        "bins": bins
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=4)

if __name__ == '__main__':
    main()
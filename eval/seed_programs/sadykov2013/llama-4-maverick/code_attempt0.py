import argparse
import json
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
    num_items = instance['num_items']
    bin_capacity = instance['bin_capacity']
    items = instance['items']
    conflict_edges = instance['conflict_edges']

    # Create a Gurobi model
    model = Model('BinPackingWithConflicts')
    model.modelSense = GRB.MINIMIZE

    # Variables
    x = {}  # x[i, k] = 1 if item i is assigned to bin k
    y = {}  # y[k] = 1 if bin k is used
    for item in items:
        i = item['id']
        for k in range(num_items):  # Assuming at most num_items bins are needed
            x[i, k] = model.addVar(vtype=GRB.BINARY, name=f'x_{i}_{k}')
    for k in range(num_items):
        y[k] = model.addVar(vtype=GRB.BINARY, name=f'y_{k}')

    # Objective: Minimize the number of bins used
    model.setObjective(quicksum(y[k] for k in range(num_items)), GRB.MINIMIZE)

    # Constraints
    for item in items:
        i = item['id']
        model.addConstr(quicksum(x[i, k] for k in range(num_items)) >= 1, name=f'assign_{i}')

    for k in range(num_items):
        model.addConstr(quicksum(items[i]['weight'] * x[i, k] for i in range(num_items)) <= bin_capacity * y[k], name=f'capacity_{k}')

    for edge in conflict_edges:
        i, j = edge
        for k in range(num_items):
            model.addConstr(x[i, k] + x[j, k] <= 1, name=f'conflict_{i}_{j}_{k}')

    # Symmetry breaking: try to use bins in order
    for k in range(1, num_items):
        model.addConstr(y[k-1] >= y[k], name=f'symmetry_{k}')

    # Optimize
    model.Params.TimeLimit = args.time_limit
    model.optimize()

    # Extract solution
    solution = {'bins': {}}
    if model.SolCount > 0:
        bins = {}
        for k in range(num_items):
            bin_items = [i for i in range(num_items) if x[i, k].X > 0.5]
            if bin_items:
                total_weight = sum(items[i]['weight'] for i in bin_items)
                bins[k] = {'items': bin_items, 'total_weight': total_weight}
        solution['objective_value'] = len(bins)
        solution['bins'] = {str(k): v for k, v in bins.items()}
    else:
        solution['objective_value'] = 0
        solution['bins'] = {}

    return solution

def main():
    global args
    args = parse_args()
    instance = read_instance(args.instance_path)
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Create a Gurobi model and solve
    model = Model('BinPackingWithConflicts')
    model.modelSense = GRB.MINIMIZE

    num_items = instance['num_items']
    bin_capacity = instance['bin_capacity']
    items = instance['items']
    conflict_edges = instance['conflict_edges']

    # Variables
    x = {}  # x[i, k] = 1 if item i is assigned to bin k
    y = {}  # y[k] = 1 if bin k is used
    for item in items:
        i = item['id']
        for k in range(num_items):  
            x[i, k] = model.addVar(vtype=GRB.BINARY, name=f'x_{i}_{k}')
    for k in range(num_items):
        y[k] = model.addVar(vtype=GRB.BINARY, name=f'y_{k}')

    # Objective: Minimize the number of bins used
    model.setObjective(quicksum(y[k] for k in range(num_items)), GRB.MINIMIZE)

    # Constraints
    for item in items:
        i = item['id']
        model.addConstr(quicksum(x[i, k] for k in range(num_items)) >= 1, name=f'assign_{i}')

    for k in range(num_items):
        model.addConstr(quicksum(items[i]['weight'] * x[i, k] for i in range(num_items)) <= bin_capacity * y[k], name=f'capacity_{k}')

    for edge in conflict_edges:
        i, j = edge
        for k in range(num_items):
            model.addConstr(x[i, k] + x[j, k] <= 1, name=f'conflict_{i}_{j}_{k}')

    # Symmetry breaking: try to use bins in order
    for k in range(1, num_items):
        model.addConstr(y[k-1] >= y[k], name=f'symmetry_{k}')

    model.Params.TimeLimit = args.time_limit

    def log_callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj)

    model.optimize(log_callback)

    # Extract solution
    solution = {'bins': {}}
    if model.SolCount > 0:
        bins = {}
        for k in range(num_items):
            bin_items = [i for i in range(num_items) if x[i, k].X > 0.5]
            if bin_items:
                total_weight = sum(items[i]['weight'] for i in bin_items)
                bins[k] = {'items': bin_items, 'total_weight': total_weight}
        solution['objective_value'] = len(bins)
        solution['bins'] = {str(k): v for k, v in bins.items()}
    else:
        solution['objective_value'] = 0
        solution['bins'] = {}

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=4)

if __name__ == '__main__':
    main()
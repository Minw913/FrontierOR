import argparse
import json
from gurobipy import Model, GRB, quicksum

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    return instance

def solve(instance, time_limit):
    bin_capacity = instance['bin_capacity']
    items = instance['items']
    num_items = instance['num_items']

    model = Model()
    model.Params.TimeLimit = time_limit
    model.Params.LogToConsole = 0  # Disable Gurobi console output

    # Create variables
    num_bins = num_items  # Upper bound on the number of bins needed
    y = model.addVars(num_bins, vtype=GRB.BINARY, name='y')
    x = model.addVars(num_items, num_bins, vtype=GRB.BINARY, name='x')

    # Objective: minimize the number of bins used
    model.setObjective(quicksum(y[i] for i in range(num_bins)), GRB.MINIMIZE)

    # Constraints
    for j in range(num_items):
        model.addConstr(quicksum(x[j, i] for i in range(num_bins)) == 1, name=f'assign_{j}')

    for i in range(num_bins):
        model.addConstr(quicksum(items[j] * x[j, i] for j in range(num_items)) <= bin_capacity * y[i], name=f'capacity_{i}')

    # Symmetry breaking: try to fill bins in order
    for i in range(num_bins - 1):
        model.addConstr(y[i] >= y[i + 1], name=f'symmetry_{i}')

    model.optimize()

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        objective_value = model.getObjective().getValue()
        bin_assignments = []
        for i in range(int(objective_value)):
            bin_items = [items[j] for j in range(num_items) if x[j, i].x > 0.5]
            total_size = sum(bin_items)
            bin_assignments.append({'items': bin_items, 'total_size': total_size})
        return objective_value, bin_assignments
    else:
        raise Exception(f'Gurobi model failed to find a solution within the time limit. Status: {model.status}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, required=False)
    args = parser.parse_args()

    instance = read_instance(args.instance_path)
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    model = Model()
    model.Params.TimeLimit = args.time_limit
    model.Params.LogToConsole = 0  

    bin_capacity = instance['bin_capacity']
    items = instance['items']
    num_items = instance['num_items']

    num_bins = num_items  
    y = model.addVars(num_bins, vtype=GRB.BINARY, name='y')
    x = model.addVars(num_items, num_bins, vtype=GRB.BINARY, name='x')

    model.setObjective(quicksum(y[i] for i in range(num_bins)), GRB.MINIMIZE)

    for j in range(num_items):
        model.addConstr(quicksum(x[j, i] for i in range(num_bins)) == 1, name=f'assign_{j}')

    for i in range(num_bins):
        model.addConstr(quicksum(items[j] * x[j, i] for j in range(num_items)) <= bin_capacity * y[i], name=f'capacity_{i}')

    for i in range(num_bins - 1):
        model.addConstr(y[i] >= y[i + 1], name=f'symmetry_{i}')

    def log_callback(model, where):
        if where == GRB.Callback.MIP:
            incumbent = model.cbGet(GRB.Callback.MIP_OBJBST)
            if model._logger and incumbent < model._best_incumbent:
                model._best_incumbent = incumbent
                model._logger.log(incumbent)

    model._logger = logger
    model._best_incumbent = float('inf')
    model.optimize(log_callback)

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        objective_value = model.getObjective().getValue()
        bin_assignments = []
        for i in range(int(objective_value)):
            bin_items = [items[j] for j in range(num_items) if x[j, i].x > 0.5]
            total_size = sum(bin_items)
            bin_assignments.append({'items': bin_items, 'total_size': total_size})
        solution = {
            'objective_value': int(objective_value),
            'bin_assignments': bin_assignments,
            'num_bins': int(objective_value)
        }
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=4)
    else:
        raise Exception(f'Gurobi model failed to find a solution within the time limit. Status: {model.status}')

if __name__ == '__main__':
    from solution_logger import SolutionLogger
    main()
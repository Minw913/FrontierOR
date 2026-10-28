import argparse
import json
from gurobipy import Model, GRB, quicksum
from collections import Counter

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    return instance

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, required=False)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    instance = read_instance(args.instance_path)
    bin_capacity = instance['bin_capacity']
    items = instance['items']
    item_counts = Counter(items)

    # First-Fit Decreasing (FFD) heuristic to get an initial solution
    bins = []
    for size, count in item_counts.most_common():
        for _ in range(count):
            added = False
            for bin in bins:
                if sum(bin) + size <= bin_capacity:
                    bin.append(size)
                    added = True
                    break
            if not added:
                bins.append([size])

    model = Model()
    model.setParam("Threads", 1)
    model.Params.TimeLimit = args.time_limit
    model.Params.LogToConsole = 0  
    model.Params.MIPFocus = 1  # Focus on finding feasible solutions

    num_bins = len(bins)
    y = model.addVars(num_bins, vtype=GRB.BINARY, name='y')
    x = model.addVars(len(items), num_bins, vtype=GRB.BINARY, name='x')

    model.setObjective(quicksum(y[i] for i in range(num_bins)), GRB.MINIMIZE)

    for j, item in enumerate(items):
        model.addConstr(quicksum(x[j, i] for i in range(num_bins)) == 1, name=f'assign_{j}')

    for i in range(num_bins):
        model.addConstr(quicksum(items[j] * x[j, i] for j in range(len(items))) <= bin_capacity * y[i], name=f'capacity_{i}')

    # Initialize variables based on the FFD heuristic solution
    for i, bin_items in enumerate(bins):
        for item in bin_items:
            j = items.index(item)
            x[j, i].Start = 1
        y[i].Start = 1

    def log_callback(model, where):
        if where == GRB.Callback.MIP:
            incumbent = model.cbGet(GRB.Callback.MIP_OBJBST)
            if model._logger and incumbent < model._best_incumbent:
                model._best_incumbent = incumbent
                model._logger.log(int(incumbent)) 

    model._logger = logger
    model._best_incumbent = float('inf')
    model.optimize(log_callback)

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        objective_value = round(model.getObjective().getValue()) 
        bin_assignments = []
        used_bins = []
        for i in range(num_bins):
            bin_items = [items[j] for j in range(len(items)) if x[j, i].x > 0.5]
            if bin_items:
                total_size = sum(bin_items)
                bin_assignments.append({'items': bin_items, 'total_size': total_size})
                used_bins.append(i)
        solution = {
            'objective_value': objective_value,
            'bin_assignments': bin_assignments,
            'num_bins': objective_value
        }
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=4)
    else:
        raise Exception(f'Gurobi model failed to find a solution within the time limit. Status: {model.status}')

if __name__ == '__main__':
    main()
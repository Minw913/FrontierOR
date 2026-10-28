import argparse
import json
from gurobipy import Model, GRB, quicksum
import sys
import os
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    # Read the instance data
    with open(args.instance_path, 'r') as f:
        data = json.load(f)
    
    W = data['parameters']['W']
    items = data['items']

    # Compute upper bounds for each item and possible T
    sum_ceil = 0
    sum_total_weight = 0
    item_demands = []
    item_weights = []
    u_list = []
    for item in items:
        weight = item['weight']
        demand = item['demand']
        u = W // weight
        sum_total_weight += weight * demand
        item_demands.append(demand)
        item_weights.append(weight)
        u_list.append(u)
        if demand > 0:
            sum_ceil += (demand + u - 1) // u  # Ceiling division
    
    sum_total_ceil = (sum_total_weight + W - 1) // W
    T = max(sum_ceil, sum_total_ceil)
    if T == 0:
        # All items have zero demand
        solution = {
            "objective_value": 0,
            "num_bins": 0,
            "bins": []
        }
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f)
        return

    # Set up the model
    model = Model()
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('Threads', 1)

    # Variables
    y = model.addVars(T, vtype=GRB.BINARY, name='y')
    x = model.addVars(len(items), T, vtype=GRB.INTEGER, name='x')

    # Set upper bounds for x variables
    for i in range(len(items)):
        u = u_list[i]
        for b in range(T):
            x[i, b].ub = u

    # Capacity constraints
    for b in range(T):
        model.addConstr(
            quicksum(x[i, b] * item_weights[i] for i in range(len(items))) <= W * y[b],
            name=f'cap_{b}'
        )

    # Demand constraints
    for i in range(len(items)):
        if item_demands[i] > 0:
            model.addConstr(
                quicksum(x[i, b] for b in range(T)) >= item_demands[i],
                name=f'demand_{i}'
            )

    # Objective: minimize the number of bins used
    model.setObjective(quicksum(y[b] for b in range(T)), GRB.MINIMIZE)

    # Initialize logger
    logger = None
    if args.log_path is not None:
        logger = SolutionLogger(args.log_path, sense='minimize')

    # Callback function for logging
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            current_obj = int(model.cbGet(GRB.Callback.MIPSOL_OBJ))
            if logger:
                logger.log(current_obj)

    # Solve the model
    model.optimize(callback if args.log_path else None)

    # Process the solution
    bins = []
    if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SOLUTION_LIMIT):
        active_bins = [b for b in range(T) if y[b].X > 0.5]
        num_bins = len(active_bins)
        for b in active_bins:
            items_in_bin = []
            for i, item in enumerate(items):
                count = int(round(x[i, b].X))
                if count > 0:
                    items_in_bin.extend([item['item_type']] * count)
            bins.append(items_in_bin)

        solution = {
            "objective_value": int(num_bins),
            "num_bins": num_bins,
            "bins": bins
        }
    else:
        # No feasible solution found, return empty bins
        solution = {
            "objective_value": 0,
            "num_bins": 0,
            "bins": []
        }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == '__main__':
    main()
import json
import argparse
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        instance_data = json.load(f)
    return instance_data

def solve(instance_data, time_limit, log_path):
    N = instance_data['N']
    knapsack_capacity = instance_data['knapsack_capacity']
    families = instance_data['families']

    model = Model('LogisticsKnapsack')
    model.setParam("Threads", 1)
    model.Params.TimeLimit = time_limit
    model.Params.LogToConsole = False  # Disable Gurobi logging to console

    # Variables
    family_vars = [model.addVar(vtype=GRB.BINARY, name=f'family_{i}') for i in range(N)]
    item_vars = [[model.addVar(vtype=GRB.BINARY, name=f'item_{i}_{j}') for j in range(families[i]['n_i'])] for i in range(N)]

    # Objective function
    total_profit = quicksum(families[i]['items'][j]['profit'] * item_vars[i][j] for i in range(N) for j in range(families[i]['n_i']))
    total_setup_cost = quicksum(families[i]['setup_cost'] * family_vars[i] for i in range(N))
    model.setObjective(total_profit - total_setup_cost, GRB.MAXIMIZE)

    # Constraints
    for i in range(N):
        for j in range(families[i]['n_i']):
            model.addConstr(item_vars[i][j] <= family_vars[i], name=f'item_family_link_{i}_{j}')

    total_weight = quicksum(families[i]['items'][j]['weight'] * item_vars[i][j] for i in range(N) for j in range(families[i]['n_i']))
    total_setup_capacity = quicksum(families[i]['setup_capacity'] * family_vars[i] for i in range(N))
    model.addConstr(total_weight + total_setup_capacity <= knapsack_capacity, name='knapsack_capacity')

    # Solve
    logger = SolutionLogger(log_path, sense="maximize") if log_path else None

    def callback(model, where):
        if where == GRB.Callback.MIP:
            obj = model.cbGet(GRB.Callback.MIP_OBJBST)
            if logger and obj > -GRB.INFINITY:
                logger.log(obj)

    model.optimize(callback)

    # Extract solution
    objective_value = model.getObjective().getValue() if model.SolCount > 0 else float('-inf')
    families_selected = [i for i in range(N) if family_vars[i].x > 0.5]
    items_selected = [{'family': i, 'item': j} for i in range(N) for j in range(families[i]['n_i']) if item_vars[i][j].x > 0.5]

    # Log final solution
    if logger:
        logger.log(objective_value)

    return objective_value, families_selected, items_selected

def main():
    parser = argparse.ArgumentParser(description='Logistics Knapsack Problem Solver')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the instance JSON file')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to the output solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds')
    parser.add_argument('--log_path', type=str, help='Path to the log JSONL file')
    args = parser.parse_args()

    instance_data = read_instance(args.instance_path)
    objective_value, families_selected, items_selected = solve(instance_data, args.time_limit, args.log_path)

    solution = {
        'objective_value': objective_value,
        'families_selected': families_selected,
        'items_selected': items_selected
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=4)

if __name__ == '__main__':
    main()
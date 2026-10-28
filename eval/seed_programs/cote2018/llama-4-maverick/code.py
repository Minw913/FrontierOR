import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    return instance

def solve_instance(instance, time_limit, logger):
    bin_capacity = instance['bin_capacity']
    items = instance['items']

    model = gp.Model()
    model.Params.LogToConsole = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = time_limit

    # Create variables
    y = model.addVar(vtype=GRB.INTEGER, name='y')
    x = {}
    for item in items:
        type_id = item['type_id']
        for i in range(instance['total_num_items']):
            x[type_id, i] = model.addVar(vtype=GRB.BINARY, name=f'x_{type_id}_{i}')

    z = {}
    for i in range(instance['total_num_items']):
        z[i] = model.addVar(vtype=GRB.BINARY, name=f'z_{i}')

    # Objective function
    model.setObjective(y, GRB.MINIMIZE)

    # Constraints
    for i in range(instance['total_num_items']):
        model.addConstr(gp.quicksum(x[type_id, i] for type_id, _ in [(item['type_id'], i) for item in items if i < item['demand']]) <= bin_capacity * z[i], name=f'capacity_{i}')

    for item in items:
        type_id = item['type_id']
        demand = item['demand']
        model.addConstr(gp.quicksum(x[type_id, i] for i in range(demand)) == demand, name=f'demand_{type_id}')

    model.addConstr(y == gp.quicksum(z[i] for i in range(instance['total_num_items'])), name='objective_link')

    # Optimize
    model.optimize()
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        objective_value = model.getVarByName('y').X
        solution_dict = {v.VarName: v.X for v in model.getVars() if v.X != 0}
        if logger:
            logger.log_solution(objective_value, solution_dict)
        return objective_value, solution_dict
    else:
        raise Exception(f"Model status: {model.status}")

def write_solution(solution_path, objective_value, solution_dict):
    solution = {
        "objective_value": int(objective_value),
        "model_variables": {k: v for k, v in solution_dict.items() if not k.startswith('x_') or v > 0.5}
    }
    with open(solution_path, 'w') as f:
        json.dump(solution, f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    instance = read_instance(args.instance_path)
    objective_value, solution_dict = solve_instance(instance, args.time_limit, logger)
    write_solution(args.solution_path, objective_value, solution_dict)

if __name__ == "__main__":
    main()
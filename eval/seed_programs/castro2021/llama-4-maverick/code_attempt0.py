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
        return json.load(f)

def build_model(instance):
    n = instance['n']
    m = instance['m']
    supplies = instance['supplies']
    demands = instance['demands']
    linear_costs = instance['linear_costs']
    quadratic_costs = instance['quadratic_costs']
    arc_capacity = instance['arc_capacity']
    cost_type = instance['cost_type']

    model = gp.Model()
    model.Params.LogToConsole = 0  # Disable Gurobi console output

    # Flow variables
    flows = model.addVars(n, m, lb=0, ub=arc_capacity, name='x')

    # Objective function
    if cost_type == 'linear_integer' or cost_type == 'linear_fractional':
        model.setObjective(gp.quicksum(linear_costs[i][j] * flows[i, j] for i in range(n) for j in range(m)), GRB.MINIMIZE)
    elif cost_type == 'quadratic':
        model.setObjective(gp.quicksum(linear_costs[i][j] * flows[i, j] + quadratic_costs[i][j] * flows[i, j] * flows[i, j] for i in range(n) for j in range(m)), GRB.MINIMIZE)
    else:
        raise ValueError(f"Unsupported cost type: {cost_type}")

    # Demand constraints
    for j in range(m):
        model.addConstr(gp.quicksum(flows[i, j] for i in range(n)) == demands[j], name=f'demand_{j}')

    # Supply constraints
    for i in range(n):
        model.addConstr(gp.quicksum(flows[i, j] for j in range(m)) <= supplies[i], name=f'supply_{i}')

    return model, flows

def solve_model(model, time_limit, logger):
    model.Params.TimeLimit = time_limit
    model.optimize(lambda model, where: callback(model, where, logger))
    return model

def callback(model, where, logger):
    if where == GRB.Callback.MIPSOL:
        objective_value = model.cbGet(GRB.Callback.MIPSOL_OBJ)
        if logger:
            logger.log(objective_value)

def extract_solution(model, flows, n, m):
    objective_value = model.ObjVal
    flow_values = {f'x_{i}_{j}': flows[i, j].X for i in range(n) for j in range(m)}
    return {'objective_value': objective_value, 'flows': flow_values}

def main():
    args = parse_args()
    instance = read_instance(args.instance_path)
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    model, flows = build_model(instance)
    model = solve_model(model, args.time_limit, logger)
    solution = extract_solution(model, flows, instance['n'], instance['m'])

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=4)

if __name__ == '__main__':
    main()
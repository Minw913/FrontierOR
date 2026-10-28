import json
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import argparse
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    return instance

def solve_instance(instance, time_limit, logger):
    N = instance['N']
    num_scenarios = instance['second_stage']['demand_distribution']['num_scenarios']
    scenarios = instance['second_stage']['demand_distribution']['scenarios']
    probabilities = instance['second_stage']['demand_distribution']['probabilities']
    holding_costs = instance['second_stage']['holding_costs']
    penalty_costs = instance['second_stage']['penalty_costs']
    transshipment_costs_mean = instance['second_stage']['transshipment_costs_mean']
    random_cost_links = instance['second_stage']['random_cost_links']
    random_cost_std_fraction = instance['second_stage']['random_cost_std_fraction']
    variable_lower_bounds = instance['first_stage']['variable_lower_bounds']

    model = gp.Model('RetailDistribution')
    model.setParam("Threads", 1)
    model.Params.TimeLimit = time_limit

    # First-stage variables: order-up-to stock levels
    s = model.addVars(N, lb=variable_lower_bounds, name='s')

    objective = gp.LinExpr()

    # Second-stage variables for each scenario
    for scenario_idx in range(num_scenarios):
        scenario_prefix = f'scenario_{scenario_idx}_'
        e = model.addVars(N, lb=0, name=scenario_prefix + 'e')  # Ending inventory
        f = model.addVars(N, lb=0, name=scenario_prefix + 'f')  # Stock used locally
        q = model.addVars(N, lb=0, name=scenario_prefix + 'q')  # Replenishment quantity
        r = model.addVars(N, lb=0, name=scenario_prefix + 'r')  # Backlog
        t = model.addVars(N, N, lb=0, name=scenario_prefix + 't')  # Transshipment quantity

        # Balance constraints for this scenario
        for i in range(N):
            model.addConstr(s[i] == e[i] + gp.quicksum(t[i, j] for j in range(N)) + f[i], name=f'{scenario_prefix}balance_stock_{i}')
            model.addConstr(scenarios[scenario_idx][i] == f[i] + gp.quicksum(t[j, i] for j in range(N)) + r[i], name=f'{scenario_prefix}balance_demand_{i}')
            model.addConstr(e[i] + q[i] == s[i], name=f'{scenario_prefix}inventory_replenishment_{i}')

        # System-wide balance constraint for this scenario
        model.addConstr(gp.quicksum(r[i] + q[i] for i in range(N)) == gp.quicksum(scenarios[scenario_idx][i] for i in range(N)), name=f'{scenario_prefix}system_balance')

        # Calculate scenario cost
        scenario_cost = gp.quicksum(holding_costs[i] * e[i] + penalty_costs[i] * r[i] for i in range(N)) 
        transshipment_cost = gp.quicksum(transshipment_costs_mean[i][j] * t[i, j] for i in range(N) for j in range(N))
        scenario_total_cost = scenario_cost + transshipment_cost
        objective += probabilities[scenario_idx] * scenario_total_cost

    # Objective: Minimize expected total cost across scenarios
    model.setObjective(objective, GRB.MINIMIZE)

    model.optimize()
    if logger and model.SolCount > 0:
        logger.log(model.ObjVal)

    # Prepare solution output
    if model.SolCount > 0:
        solution = {
            'objective_value': model.ObjVal,
            's': [s[i].X for i in range(N)],
            'scenarios': []
        }

        for scenario_idx in range(num_scenarios):
            scenario_prefix = f'scenario_{scenario_idx}_'
            scenario_solution = {
                'e': [model.getVarByName(scenario_prefix + f'e[{i}]').X for i in range(N)],
                'f': [model.getVarByName(scenario_prefix + f'f[{i}]').X for i in range(N)],
                'q': [model.getVarByName(scenario_prefix + f'q[{i}]').X for i in range(N)],
                'r': [model.getVarByName(scenario_prefix + f'r[{i}]').X for i in range(N)],
                't': [[model.getVarByName(scenario_prefix + f't[{i},{j}]').X for j in range(N)] for i in range(N)]
            }
            solution['scenarios'].append(scenario_solution)
    else:
        solution = {
            'objective_value': float('inf'),
            's': [0.0] * N,
            'scenarios': []
        }
        for _ in range(num_scenarios):
            scenario_solution = {
                'e': [0.0] * N,
                'f': [0.0] * N,
                'q': [0.0] * N,
                'r': [0.0] * N,
                't': [[0.0] * N for _ in range(N)]
            }
            solution['scenarios'].append(scenario_solution)

    return solution

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense='minimize') if args.log_path else None

    instance = read_instance(args.instance_path)
    solution = solve_instance(instance, args.time_limit, logger)

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=4)

if __name__ == '__main__':
    main()
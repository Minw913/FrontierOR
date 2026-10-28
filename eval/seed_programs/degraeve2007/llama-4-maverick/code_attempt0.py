import json
import argparse
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    return instance

def solve_instance(instance, time_limit):
    num_products = instance['dimensions']['num_products']
    num_periods = instance['dimensions']['num_periods']
    demand = instance['parameters']['demand']
    holding_cost = instance['parameters']['holding_cost']
    setup_cost = instance['parameters']['setup_cost']
    variable_production_cost = instance['parameters']['variable_production_cost']
    variable_production_time = instance['parameters']['variable_production_time']
    setup_time = instance['parameters']['setup_time']
    capacity = instance['parameters']['capacity']
    initial_inventory_cost = instance['parameters']['initial_inventory_cost']

    model = Model()
    model.Params.TimeLimit = time_limit
    model.Params.LogToConsole = 0  # Disable Gurobi console output

    # Variables
    production = [[model.addVar(vtype=GRB.CONTINUOUS, name=f'production_{i}_{t}') for t in range(num_periods)] for i in range(num_products)]
    setup = [[model.addVar(vtype=GRB.BINARY, name=f'setup_{i}_{t}') for t in range(num_periods)] for i in range(num_products)]
    inventory = [[model.addVar(vtype=GRB.CONTINUOUS, name=f'inventory_{i}_{t}') for t in range(num_periods)] for i in range(num_products)]
    initial_inventory = [model.addVar(vtype=GRB.CONTINUOUS, name=f'initial_inventory_{i}') for i in range(num_products)]

    # Objective
    model.setObjective(
        quicksum(initial_inventory_cost[i] * initial_inventory[i] for i in range(num_products)) +
        quicksum(setup_cost[i][t] * setup[i][t] for i in range(num_products) for t in range(num_periods)) +
        quicksum(variable_production_cost[i][t] * production[i][t] for i in range(num_products) for t in range(num_periods)) +
        quicksum(holding_cost[i][t] * inventory[i][t] for i in range(num_products) for t in range(num_periods)),
        GRB.MINIMIZE
    )

    # Constraints
    for i in range(num_products):
        model.addConstr(initial_inventory[i] + production[i][0] == demand[i][0] + inventory[i][0], f'balance_period_0_product_{i}')
        for t in range(1, num_periods):
            model.addConstr(inventory[i][t-1] + production[i][t] == demand[i][t] + inventory[i][t], f'balance_period_{t}_product_{i}')

    for i in range(num_products):
        for t in range(num_periods):
            model.addConstr(production[i][t] <= demand[i][t] * setup[i][t], f'production_setup_link_{i}_{t}')
            cumulative_demand = sum(demand[i][k] for k in range(t, num_periods))
            model.addConstr(production[i][t] <= (capacity[t] - setup_time[i][t] * setup[i][t]) / variable_production_time[i][t], f'production_capacity_{i}_{t}')
            model.addConstr(production[i][t] <= cumulative_demand, f'production_demand_{i}_{t}')

    for t in range(num_periods):
        model.addConstr(quicksum(setup_time[i][t] * setup[i][t] + variable_production_time[i][t] * production[i][t] for i in range(num_products)) <= capacity[t], f'capacity_constraint_{t}')

    model.optimize()
    return model, production, setup, inventory, initial_inventory

def extract_solution(model, production, setup, inventory, initial_inventory):
    num_products = len(initial_inventory)
    num_periods = len(production[0])
    solution = {
        'objective_value': model.ObjVal if model.status == GRB.OPTIMAL else float('inf'),
        'variables': {
            'production': [[production[i][t].X for t in range(num_periods)] for i in range(num_products)],
            'setup': [[int(setup[i][t].X) for t in range(num_periods)] for i in range(num_products)],
            'inventory': [[inventory[i][t].X for t in range(num_periods)] for i in range(num_products)],
            'initial_inventory': [initial_inventory[i].X for i in range(num_products)]
        }
    }
    return solution

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path')
    args = parser.parse_args()

    instance = read_instance(args.instance_path)
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    model, production, setup, inventory, initial_inventory = solve_instance(instance, args.time_limit)

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        solution = extract_solution(model, production, setup, inventory, initial_inventory)
        if logger:
            logger.log(solution['objective_value'])
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f)
    else:
        print(f"Model status: {model.status}")

if __name__ == "__main__":
    main()
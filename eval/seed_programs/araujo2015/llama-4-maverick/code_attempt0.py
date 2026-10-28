import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    return instance

def solve_instance(instance, time_limit, log_path):
    num_items = instance['num_items']
    num_periods = instance['num_periods']
    demand = instance['demand']
    setup_cost = instance['setup_cost']
    holding_cost = instance['holding_cost']
    variable_cost = instance['variable_cost']
    setup_time = instance['setup_time']
    variable_time = instance['variable_time']
    capacity = instance['capacity']
    initial_inventory_cost = instance['initial_inventory_cost']

    model = gp.Model()
    model.Params.TimeLimit = time_limit
    model.Params.LogToConsole = False

    # Variables
    production = [[model.addVar(vtype=GRB.CONTINUOUS, name=f'production_{i}_{t}') for t in range(num_periods)] for i in range(num_items)]
    setup = [[model.addVar(vtype=GRB.BINARY, name=f'setup_{i}_{t}') for t in range(num_periods)] for i in range(num_items)]
    inventory = [[model.addVar(vtype=GRB.CONTINUOUS, name=f'inventory_{i}_{t}') for t in range(num_periods+1)] for i in range(num_items)]

    # Objective
    model.setObjective(
        gp.quicksum(initial_inventory_cost[i] * inventory[i][0] for i in range(num_items)) +
        gp.quicksum(setup_cost[i] * setup[i][t] for i in range(num_items) for t in range(num_periods)) +
        gp.quicksum(variable_cost[i] * production[i][t] for i in range(num_items) for t in range(num_periods)) +
        gp.quicksum(holding_cost[i] * inventory[i][t] for i in range(num_items) for t in range(1, num_periods+1)),
        GRB.MINIMIZE
    )

    # Constraints
    for i in range(num_items):
        for t in range(num_periods):
            model.addConstr(inventory[i][t] + production[i][t] == demand[i][t] + inventory[i][t+1], name=f'inventory_balance_{i}_{t}')
            model.addConstr(production[i][t] <= (capacity[t] - setup_time[i]) / variable_time[i] * setup[i][t], name=f'production_capacity_{i}_{t}')
            model.addConstr(production[i][t] <= sum(demand[i][k] for k in range(t, num_periods)) * setup[i][t], name=f'production_demand_{i}_{t}')
        model.addConstr(inventory[i][num_periods] == 0, name=f'final_inventory_{i}')

    for t in range(num_periods):
        model.addConstr(gp.quicksum(setup_time[i] * setup[i][t] + variable_time[i] * production[i][t] for i in range(num_items)) <= capacity[t], name=f'capacity_{t}')

    logger = SolutionLogger(log_path, sense="minimize") if log_path else None

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            objective_value = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(objective_value)

    model.optimize(callback=callback)

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        objective_value = model.ObjVal
        production_sol = [[production[i][t].X for t in range(num_periods)] for i in range(num_items)]
        setup_sol = [[int(setup[i][t].X) for t in range(num_periods)] for i in range(num_items)]
        inventory_sol = [[inventory[i][t].X for t in range(num_periods+1)] for i in range(num_items)]
        return objective_value, production_sol, setup_sol, inventory_sol
    else:
        raise Exception(f"Model status: {model.status}")

def write_solution(solution_path, objective_value, production, setup, inventory):
    solution = {
        "objective_value": objective_value,
        "production": production,
        "setup": setup,
        "inventory": inventory
    }
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=4)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str)
    args = parser.parse_args()

    instance = read_instance(args.instance_path)
    objective_value, production, setup, inventory = solve_instance(instance, args.time_limit, args.log_path)
    write_solution(args.solution_path, objective_value, production, setup, inventory)

if __name__ == "__main__":
    main()
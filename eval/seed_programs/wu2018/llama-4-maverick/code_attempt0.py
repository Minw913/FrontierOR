import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    return instance

def solve_instance(instance, time_limit):
    num_items = instance['num_items']
    num_periods = instance['num_periods']
    num_machines = instance['num_machines']
    max_machines_per_item = instance['max_machines_per_item_per_period']
    machine_capabilities = instance['machine_capabilities']
    demand = instance['demand']
    production_time = instance['production_time']
    holding_cost = instance['holding_cost']
    setup_cost = instance['setup_cost']
    setup_time = instance['setup_time']
    production_cost = instance['production_cost']
    capacity = instance['capacity']

    model = gp.Model("production_planning")
    model.Params.LogToConsole = 0  # Disable console logging

    # Decision Variables
    production = model.addVars(num_items, num_machines, num_periods, lb=0, vtype=GRB.CONTINUOUS, name="X")
    inventory = model.addVars(num_items, num_periods, lb=0, vtype=GRB.CONTINUOUS, name="I")
    setups = model.addVars(num_items, num_machines, num_periods, vtype=GRB.BINARY, name="Y")

    # Objective Function
    model.setObjective(
        gp.quicksum(setup_cost[f'item_{i}_machine_{m}'] * setups[i, m, t] for i in range(num_items) for m in range(num_machines) for t in range(num_periods)) +
        gp.quicksum(production_cost[f'item_{i}_machine_{m}'] * production[i, m, t] for i in range(num_items) for m in range(num_machines) for t in range(num_periods)) +
        gp.quicksum(holding_cost[i] * inventory[i, t] for i in range(num_items) for t in range(num_periods)),
        GRB.MINIMIZE
    )

    # Constraints
    for i in range(num_items):
        for t in range(num_periods):
            model.addConstr(
                gp.quicksum(production[i, m, t] for m in range(num_machines) if m + 1 in machine_capabilities[i]) + 
                (inventory[i, t-1] if t > 0 else instance.get('initial_inventory', [0]*num_items)[i]) - 
                inventory[i, t] == demand[i][t],
                name=f"demand_balance_item{i}_period{t}"
            )

    for m in range(num_machines):
        for t in range(num_periods):
            model.addConstr(
                gp.quicksum(production_time[f'item_{i}_machine_{m}'] * production[i, m, t] for i in range(num_items) if m + 1 in machine_capabilities[i]) +
                gp.quicksum(setup_time[f'item_{i}_machine_{m}'] * setups[i, m, t] for i in range(num_items) if m + 1 in machine_capabilities[i]) <= capacity[m][t],
                name=f"machine_capacity_machine{m}_period{t}"
            )

    for i in range(num_items):
        for m in range(num_machines):
            for t in range(num_periods):
                if m + 1 in machine_capabilities[i]:
                    model.addConstr(production[i, m, t] <= demand[i][t:].sum() * setups[i, m, t], name=f"production_setup_link_item{i}_machine{m}_period{t}")
                    model.addConstr(production[i, m, t] <= (capacity[m][t] - setup_time[f'item_{i}_machine_{m}']) / production_time[f'item_{i}_machine_{m}'] * setups[i, m, t], name=f"production_capacity_link_item{i}_machine{m}_period{t}")
                else:
                    model.addConstr(production[i, m, t] == 0, name=f"infeasible_production_item{i}_machine{m}_period{t}")
                    model.addConstr(setups[i, m, t] == 0, name=f"infeasible_setup_item{i}_machine{m}_period{t}")

    for i in range(num_items):
        for t in range(num_periods):
            model.addConstr(gp.quicksum(setups[i, m, t] for m in range(num_machines) if m + 1 in machine_capabilities[i]) <= max_machines_per_item[i], name=f"max_machines_item{i}_period{t}")

    model.Params.TimeLimit = time_limit
    model.optimize()

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        objective_value = model.ObjVal
        production_sol = {(i, m, t): production[i, m, t].X for i in range(num_items) for m in range(num_machines) for t in range(num_periods)}
        inventory_sol = {(i, t): inventory[i, t].X for i in range(num_items) for t in range(num_periods)}
        setups_sol = {(i, m, t): setups[i, m, t].X for i in range(num_items) for m in range(num_machines) for t in range(num_periods)}
        return objective_value, production_sol, inventory_sol, setups_sol
    else:
        raise Exception(f"Gurobi model status: {model.status}")

def format_solution(objective_value, production, inventory, setups, instance):
    num_items = instance['num_items']
    num_machines = instance['num_machines']
    num_periods = instance['num_periods']
    machine_capabilities = instance['machine_capabilities']

    solution = {
        "objective_value": objective_value,
        "production": {f"X_item{i}_machine{m}_period{t}": production.get((i, m, t), 0) for i in range(num_items) for m in range(num_machines) for t in range(num_periods) if m + 1 in machine_capabilities[i]},
        "inventory": {f"I_item{i}_period{t}": inventory.get((i, t), 0) for i in range(num_items) for t in range(num_periods)},
        "setups": {f"Y_item{i}_machine{m}_period{t}": setups.get((i, m, t), 0) for i in range(num_items) for m in range(num_machines) for t in range(num_periods) if m + 1 in machine_capabilities[i]}
    }
    return solution

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path")
    args = parser.parse_args()

    instance = read_instance(args.instance_path)
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    try:
        objective_value, production, inventory, setups = solve_instance(instance, args.time_limit)
        if logger:
            logger.log(objective_value)
        solution = format_solution(objective_value, production, inventory, setups, instance)
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
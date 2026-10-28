import json
import argparse
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as file:
        instance = json.load(file)
    return instance

def solve_instance(instance, time_limit, log_path):
    num_periods = instance['num_periods']
    num_warehouses = instance['num_warehouses']
    num_retailers = instance['num_retailers']
    num_scenarios = instance['num_scenarios']
    retailer_to_warehouse = instance['retailer_to_warehouse']
    scenario_probabilities = instance['scenario_probabilities']
    demands = instance['demands']
    holding_cost_plant = instance['holding_cost_plant']
    holding_cost_warehouse = instance['holding_cost_warehouse']
    holding_costs_retailers = instance['holding_costs_retailers']
    setup_costs_plant = instance['setup_costs_plant']
    setup_costs_warehouses = instance['setup_costs_warehouses']
    setup_costs_retailers = instance['setup_costs_retailers']
    initial_setups_imposed = instance['initial_setups_imposed']

    model = Model("DistributionNetwork")
    model.modelSense = GRB.MINIMIZE

    # Setup Variables
    y_plant = model.addVars(num_periods, vtype=GRB.BINARY, name='y_plant')
    y_warehouses = model.addVars(num_warehouses, num_periods, vtype=GRB.BINARY, name='y_warehouses')
    y_retailers = model.addVars(num_retailers, num_periods, vtype=GRB.BINARY, name='y_retailers')

    # Production and Inventory Variables
    production_plant = model.addVars(num_periods, num_scenarios, vtype=GRB.CONTINUOUS, name='production_plant')
    inventory_plant = model.addVars(num_periods, num_scenarios, vtype=GRB.CONTINUOUS, name='inventory_plant')
    delivery_warehouse = model.addVars(num_warehouses, num_periods, num_scenarios, vtype=GRB.CONTINUOUS, name='delivery_warehouse')
    inventory_warehouse = model.addVars(num_warehouses, num_periods, num_scenarios, vtype=GRB.CONTINUOUS, name='inventory_warehouse')
    delivery_retailer = model.addVars(num_retailers, num_periods, num_scenarios, vtype=GRB.CONTINUOUS, name='delivery_retailer')
    inventory_retailer = model.addVars(num_retailers, num_periods, num_scenarios, vtype=GRB.CONTINUOUS, name='inventory_retailer')

    # Tracking Variables for Demands
    production_plant_demand = model.addVars(num_retailers, num_periods, num_periods, num_scenarios, vtype=GRB.CONTINUOUS, name='production_plant_demand')
    inventory_plant_demand = model.addVars(num_retailers, num_periods, num_periods, num_scenarios, vtype=GRB.CONTINUOUS, name='inventory_plant_demand')
    delivery_warehouse_demand = model.addVars(num_retailers, num_periods, num_periods, num_scenarios, vtype=GRB.CONTINUOUS, name='delivery_warehouse_demand')
    inventory_warehouse_demand = model.addVars(num_retailers, num_periods, num_periods, num_scenarios, vtype=GRB.CONTINUOUS, name='inventory_warehouse_demand')
    delivery_retailer_demand = model.addVars(num_retailers, num_periods, num_periods, num_scenarios, vtype=GRB.CONTINUOUS, name='delivery_retailer_demand')
    inventory_retailer_demand = model.addVars(num_retailers, num_periods, num_periods, num_scenarios, vtype=GRB.CONTINUOUS, name='inventory_retailer_demand')

    # Objective Function
    setup_cost = quicksum(setup_costs_plant[t] * y_plant[t] for t in range(num_periods))
    setup_cost += quicksum(setup_costs_warehouses[w][t] * y_warehouses[w, t] for w in range(num_warehouses) for t in range(num_periods))
    setup_cost += quicksum(setup_costs_retailers[r][t] * y_retailers[r, t] for r in range(num_retailers) for t in range(num_periods))

    holding_cost = quicksum(scenario_probabilities[s] * (
        holding_cost_plant * inventory_plant_demand[r, d, t, s] +
        holding_cost_warehouse * inventory_warehouse_demand[r, d, t, s] +
        holding_costs_retailers[r] * inventory_retailer_demand[r, d, t, s]
    ) for r in range(num_retailers) for d in range(num_periods) for t in range(d + 1) for s in range(num_scenarios))

    model.setObjective(setup_cost + holding_cost)

    # Constraints
    for s in range(num_scenarios):
        for t in range(num_periods):
            model.addConstr(production_plant[t, s] == quicksum(production_plant_demand[r, d, t, s] for r in range(num_retailers) for d in range(num_periods) if t <= d))
            model.addConstr(inventory_plant[t, s] == quicksum(inventory_plant_demand[r, d, t, s] for r in range(num_retailers) for d in range(num_periods) if t <= d))
            for w in range(num_warehouses):
                model.addConstr(delivery_warehouse[w, t, s] == quicksum(delivery_warehouse_demand[r, d, t, s] for r in range(num_retailers) if retailer_to_warehouse[r] == w for d in range(num_periods) if t <= d))
                model.addConstr(inventory_warehouse[w, t, s] == quicksum(inventory_warehouse_demand[r, d, t, s] for r in range(num_retailers) if retailer_to_warehouse[r] == w for d in range(num_periods) if t <= d))
            for r in range(num_retailers):
                model.addConstr(delivery_retailer[r, t, s] == quicksum(delivery_retailer_demand[r, d, t, s] for d in range(num_periods) if t <= d))
                model.addConstr(inventory_retailer[r, t, s] == quicksum(inventory_retailer_demand[r, d, t, s] for d in range(num_periods) if t <= d))

    for r in range(num_retailers):
        for d in range(num_periods):
            for t in range(d + 1):
                for s in range(num_scenarios):
                    # Plant level
                    if t == 0:
                        model.addConstr(inventory_plant_demand[r, d, t, s] == production_plant_demand[r, d, t, s] - delivery_warehouse_demand[r, d, t, s])
                    else:
                        model.addConstr(inventory_plant_demand[r, d, t, s] == inventory_plant_demand[r, d, t - 1, s] + production_plant_demand[r, d, t, s] - delivery_warehouse_demand[r, d, t, s])

                    # Warehouse level
                    w = retailer_to_warehouse[r]
                    if t == 0:
                        model.addConstr(inventory_warehouse_demand[r, d, t, s] == delivery_warehouse_demand[r, d, t, s] - delivery_retailer_demand[r, d, t, s])
                    else:
                        model.addConstr(inventory_warehouse_demand[r, d, t, s] == inventory_warehouse_demand[r, d, t - 1, s] + delivery_warehouse_demand[r, d, t, s] - delivery_retailer_demand[r, d, t, s])

                    # Retailer level
                    if t == 0:
                        model.addConstr(inventory_retailer_demand[r, d, t, s] == delivery_retailer_demand[r, d, t, s] if t == d else 0)
                    elif t < d:
                        model.addConstr(inventory_retailer_demand[r, d, t, s] == inventory_retailer_demand[r, d, t - 1, s] + delivery_retailer_demand[r, d, t, s])
                    else:
                        model.addConstr(inventory_retailer_demand[r, d, t, s] == inventory_retailer_demand[r, d, t - 1, s] + delivery_retailer_demand[r, d, t, s] - demands[s][r][d] if t == d else inventory_retailer_demand[r, d, t - 1, s])

                    # Linking production and setup
                    model.addConstr(production_plant_demand[r, d, t, s] <= demands[s][r][d] * y_plant[t])
                    model.addConstr(delivery_warehouse_demand[r, d, t, s] <= demands[s][r][d] * y_warehouses[retailer_to_warehouse[r], t])
                    model.addConstr(delivery_retailer_demand[r, d, t, s] <= demands[s][r][d] * y_retailers[r, t])

    if initial_setups_imposed:
        model.addConstr(y_plant[0] == 1)
        for w in range(num_warehouses):
            model.addConstr(y_warehouses[w, 0] == 1)
        for r in range(num_retailers):
            model.addConstr(y_retailers[r, 0] == 1)

    logger = SolutionLogger(log_path, sense="minimize") if log_path else None

    model.optimize()

    if logger:
        logger.log(model.ObjVal)

    model.setParam('TimeLimit', time_limit)
    solution = {
        'objective_value': model.ObjVal,
        'setup_variables': {},
        'production_plant': {},
        'delivery_warehouse': {},
        'delivery_retailer': {},
        'inventory_plant': {},
        'inventory_warehouse': {},
        'inventory_retailer': {},
    }

    for t in range(num_periods):
        solution['setup_variables'][f'y_0_{t}'] = y_plant[t].X
        for w in range(num_warehouses):
            solution['setup_variables'][f'y_{w+1}_{t}'] = y_warehouses[w, t].X
        for r in range(num_retailers):
            solution['setup_variables'][f'y_{num_warehouses+r+1}_{t}'] = y_retailers[r, t].X

    for s in range(num_scenarios):
        for t in range(num_periods):
            solution['production_plant'][f'{t}_{s}'] = production_plant[t, s].X
            solution['inventory_plant'][f'{t}_{s}'] = inventory_plant[t, s].X
            for w in range(num_warehouses):
                solution['delivery_warehouse'][f'{w}_{t}_{s}'] = delivery_warehouse[w, t, s].X
                solution['inventory_warehouse'][f'{w}_{t}_{s}'] = inventory_warehouse[w, t, s].X
            for r in range(num_retailers):
                solution['delivery_retailer'][f'{r}_{t}_{s}'] = delivery_retailer[r, t, s].X
                solution['inventory_retailer'][f'{r}_{t}_{s}'] = inventory_retailer[r, t, s].X

    return solution

def main():
    parser = argparse.ArgumentParser(description='Solve a distribution network optimization problem.')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the instance JSON file.')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to the solution JSON file.')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds.')
    parser.add_argument('--log_path', type=str, help='Path to the log JSONL file.')
    args = parser.parse_args()

    instance = read_instance(args.instance_path)
    solution = solve_instance(instance, args.time_limit, args.log_path)

    with open(args.solution_path, 'w') as file:
        json.dump(solution, file, indent=4)

if __name__ == "__main__":
    main()
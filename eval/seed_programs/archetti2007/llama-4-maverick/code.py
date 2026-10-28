import argparse
import json
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        return json.load(f)

def solve(instance):
    num_retailers = instance['num_retailers']
    time_horizon = instance['time_horizon']
    supplier_id = instance['supplier']['id']
    retailer_ids = [retailer['id'] for retailer in instance['retailers']]
    locations = [instance['supplier']] + instance['retailers']
    distance_matrix = instance['distance_matrix']
    vehicle_capacity = instance['vehicle_capacity']

    try:
        model = gp.Model("IRP")
        model.setParam("Threads", 1)
        model.Params.LogToConsole = 0

        # Variables
        deliveries = model.addVars(time_horizon, num_retailers, vtype=GRB.BINARY, name='deliveries')
        quantities = model.addVars(time_horizon, num_retailers, vtype=GRB.CONTINUOUS, name='quantities')
        routes = model.addVars(time_horizon, num_retailers+1, num_retailers+1, vtype=GRB.INTEGER, name='routes', lb=0, ub=2)
        supplier_inventory = model.addVars(time_horizon+1, vtype=GRB.CONTINUOUS, name='supplier_inventory')
        retailer_inventory = model.addVars(time_horizon+1, num_retailers, vtype=GRB.CONTINUOUS, name='retailer_inventory')

        # Objective
        obj = gp.quicksum(distance_matrix[i][j] * routes[t, i, j] for t in range(time_horizon) for i in range(num_retailers+1) for j in range(num_retailers+1))
        obj += instance['supplier']['inventory_cost'] * gp.quicksum(supplier_inventory[t] for t in range(time_horizon+1))
        for r in range(num_retailers):
            obj += instance['retailers'][r]['inventory_cost'] * gp.quicksum(retailer_inventory[t, r] for t in range(time_horizon+1))
        model.setObjective(obj, GRB.MINIMIZE)

        # Constraints
        # Supplier inventory
        model.addConstr(supplier_inventory[0] == instance['supplier']['starting_inventory'], name='supplier_starting_inventory')
        for t in range(time_horizon):
            model.addConstr(supplier_inventory[t+1] == supplier_inventory[t] + instance['supplier']['production_rate'] - gp.quicksum(quantities[t, r] for r in range(num_retailers)), name=f'supplier_inventory_update_{t}')
            model.addConstr(supplier_inventory[t] >= gp.quicksum(quantities[t, r] for r in range(num_retailers)), name=f'supplier_inventory_constraint_{t}')

        # Retailer inventory
        for r in range(num_retailers):
            model.addConstr(retailer_inventory[0, r] == instance['retailers'][r]['starting_inventory'], name=f'retailer_starting_inventory_{r}')
            for t in range(time_horizon):
                model.addConstr(retailer_inventory[t+1, r] == retailer_inventory[t, r] + quantities[t, r] - instance['retailers'][r]['demand'], name=f'retailer_inventory_update_{t}_{r}')
                model.addConstr(quantities[t, r] <= deliveries[t, r] * instance['retailers'][r]['max_inventory'], name=f'delivery_quantity_bound_{t}_{r}')
                model.addConstr(quantities[t, r] == deliveries[t, r] * (instance['retailers'][r]['max_inventory'] - retailer_inventory[t, r]), name=f'delivery_quantity_exact_{t}_{r}')
            model.addConstr(retailer_inventory[time_horizon, r] >= 0, name=f'retailer_inventory_non_negative_{r}')

        # Vehicle capacity
        for t in range(time_horizon):
            model.addConstr(gp.quicksum(quantities[t, r] for r in range(num_retailers)) <= vehicle_capacity, name=f'vehicle_capacity_{t}')

        # Routing constraints
        for t in range(time_horizon):
            model.addConstr(gp.quicksum(routes[t, 0, r] for r in range(1, num_retailers+1)) == gp.quicksum(routes[t, r, 0] for r in range(1, num_retailers+1)), name=f'routing_supplier_balance_{t}')
            for r in range(1, num_retailers+1):
                model.addConstr(routes[t, 0, r] + routes[t, r, 0] == 2 * deliveries[t, r-1], name=f'routing_retailer_visit_{t}_{r}')
                model.addConstr(gp.quicksum(routes[t, r, j] for j in range(num_retailers+1)) == gp.quicksum(routes[t, j, r] for j in range(num_retailers+1)), name=f'routing_retailer_balance_{t}_{r}')
            for subset in get_subsets(num_retailers+1):
                if len(subset) > 1 and 0 in subset:
                    subset_retailers = [r-1 for r in subset if r != 0]
                    model.addConstr(gp.quicksum(routes[t, i, j] for i in subset for j in subset if i != j) <= 2 * (len(subset_retailers) - gp.quicksum(deliveries[t, r] for r in subset_retailers if r in subset_retailers) + 1), name=f'subtour_elimination_{t}_{subset}')

        model.optimize()

        if model.status == GRB.OPTIMAL:
            return extract_solution(instance, model)
        else:
            return None

    except gp.GurobiError as e:
        print(f'Gurobi error: {e}')
        return None

def get_subsets(n):
    subsets = []
    for i in range(1, 2**n):
        subset = [j for j in range(n) if (i & (1 << j))]
        subsets.append(subset)
    return subsets

def extract_solution(instance, model):
    num_retailers = instance['num_retailers']
    time_horizon = instance['time_horizon']
    solution = {
        'objective_value': model.ObjVal,
        'deliveries': {},
        'routes': {},
        'supplier_inventory': {},
        'retailer_inventory': {}
    }

    for t in range(time_horizon):
        solution['deliveries'][str(t)] = {}
        for r in range(num_retailers):
            if model.getVarByName(f'deliveries[{t},{r}]').X > 0.5:
                solution['deliveries'][str(t)][str(r)] = model.getVarByName(f'quantities[{t},{r}]').X

    for t in range(time_horizon):
        solution['routes'][str(t)] = []
        for i in range(num_retailers+1):
            for j in range(i+1, num_retailers+1):
                count = model.getVarByName(f'routes[{t},{i},{j}]').X
                if count > 0:
                    solution['routes'][str(t)].append([i, j, int(count)])
                count = model.getVarByName(f'routes[{t},{j},{i}]').X
                if count > 0:
                    solution['routes'][str(t)].append([j, i, int(count)])

    for t in range(time_horizon+1):
        solution['supplier_inventory'][str(t)] = model.getVarByName(f'supplier_inventory[{t}]').X

    for r in range(num_retailers):
        solution['retailer_inventory'][str(r)] = {}
        for t in range(time_horizon+1):
            solution['retailer_inventory'][str(r)][str(t)] = model.getVarByName(f'retailer_inventory[{t},{r}]').X

    return solution

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    instance = read_instance(args.instance_path)

    model = gp.Model("IRP")
    model.setParam("Threads", 1)
    model.Params.TimeLimit = args.time_limit
    model.Params.LogToConsole = 0

    num_retailers = instance['num_retailers']
    time_horizon = instance['time_horizon']
    supplier_id = instance['supplier']['id']
    retailer_ids = [retailer['id'] for retailer in instance['retailers']]
    locations = [instance['supplier']] + instance['retailers']
    distance_matrix = instance['distance_matrix']
    vehicle_capacity = instance['vehicle_capacity']

    # Variables
    deliveries = model.addVars(time_horizon, num_retailers, vtype=GRB.BINARY, name='deliveries')
    quantities = model.addVars(time_horizon, num_retailers, vtype=GRB.CONTINUOUS, name='quantities')
    routes = model.addVars(time_horizon, num_retailers+1, num_retailers+1, vtype=GRB.INTEGER, name='routes', lb=0, ub=2)
    supplier_inventory = model.addVars(time_horizon+1, vtype=GRB.CONTINUOUS, name='supplier_inventory')
    retailer_inventory = model.addVars(time_horizon+1, num_retailers, vtype=GRB.CONTINUOUS, name='retailer_inventory')

    # Objective
    obj = gp.quicksum(distance_matrix[i][j] * routes[t, i, j] for t in range(time_horizon) for i in range(num_retailers+1) for j in range(num_retailers+1))
    obj += instance['supplier']['inventory_cost'] * gp.quicksum(supplier_inventory[t] for t in range(time_horizon+1))
    for r in range(num_retailers):
        obj += instance['retailers'][r]['inventory_cost'] * gp.quicksum(retailer_inventory[t, r] for t in range(time_horizon+1))
    model.setObjective(obj, GRB.MINIMIZE)

    # Constraints
    # Supplier inventory
    model.addConstr(supplier_inventory[0] == instance['supplier']['starting_inventory'], name='supplier_starting_inventory')
    for t in range(time_horizon):
        model.addConstr(supplier_inventory[t+1] == supplier_inventory[t] + instance['supplier']['production_rate'] - gp.quicksum(quantities[t, r] for r in range(num_retailers)), name=f'supplier_inventory_update_{t}')
        model.addConstr(supplier_inventory[t] >= gp.quicksum(quantities[t, r] for r in range(num_retailers)), name=f'supplier_inventory_constraint_{t}')

    # Retailer inventory
    for r in range(num_retailers):
        model.addConstr(retailer_inventory[0, r] == instance['retailers'][r]['starting_inventory'], name=f'retailer_starting_inventory_{r}')
        for t in range(time_horizon):
            model.addConstr(retailer_inventory[t+1, r] == retailer_inventory[t, r] + quantities[t, r] - instance['retailers'][r]['demand'], name=f'retailer_inventory_update_{t}_{r}')
            model.addConstr(quantities[t, r] <= deliveries[t, r] * instance['retailers'][r]['max_inventory'], name=f'delivery_quantity_bound_{t}_{r}')
            model.addConstr(quantities[t, r] == deliveries[t, r] * (instance['retailers'][r]['max_inventory'] - retailer_inventory[t, r]), name=f'delivery_quantity_exact_{t}_{r}')
        model.addConstr(retailer_inventory[time_horizon, r] >= 0, name=f'retailer_inventory_non_negative_{r}')

    # Vehicle capacity
    for t in range(time_horizon):
        model.addConstr(gp.quicksum(quantities[t, r] for r in range(num_retailers)) <= vehicle_capacity, name=f'vehicle_capacity_{t}')

    # Routing constraints
    for t in range(time_horizon):
        model.addConstr(gp.quicksum(routes[t, 0, r] for r in range(1, num_retailers+1)) == gp.quicksum(routes[t, r, 0] for r in range(1, num_retailers+1)), name=f'routing_supplier_balance_{t}')
        for r in range(1, num_retailers+1):
            model.addConstr(routes[t, 0, r] + routes[t, r, 0] == 2 * deliveries[t, r-1], name=f'routing_retailer_visit_{t}_{r}')
            model.addConstr(gp.quicksum(routes[t, r, j] for j in range(num_retailers+1)) == gp.quicksum(routes[t, j, r] for j in range(num_retailers+1)), name=f'routing_retailer_balance_{t}_{r}')
        for subset in get_subsets(num_retailers+1):
            if len(subset) > 1 and 0 in subset:
                subset_retailers = [r-1 for r in subset if r != 0]
                model.addConstr(gp.quicksum(routes[t, i, j] for i in subset for j in subset if i != j) <= 2 * (len(subset_retailers) - gp.quicksum(deliveries[t, r] for r in subset_retailers) + 1), name=f'subtour_elimination_{t}_{subset}')

    def log_callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj)

    model.optimize(log_callback)

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        solution = extract_solution(instance, model)
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()
import argparse
import json
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    return instance

def solve_instance(instance, time_limit):
    num_customers = instance['num_customers']
    num_periods = instance['num_periods']
    num_vehicles = instance['num_vehicles']
    vehicle_capacity = instance['vehicle_capacity']
    depot = instance['depot']
    customers = instance['customers']
    distance_matrix = instance['distance_matrix']

    model = gp.Model('IRP')
    model.Params.TimeLimit = time_limit
    model.Params.LogToConsole = 0

    # Variables
    I_s = model.addVars(num_periods + 1, vtype=GRB.CONTINUOUS, name='I_s')  # Supplier inventory
    I_c = model.addVars(range(num_customers), num_periods + 1, vtype=GRB.CONTINUOUS, name='I_c')  # Customer inventory
    q = model.addVars(range(num_customers), num_periods, vtype=GRB.CONTINUOUS, name='q')  # Quantity delivered to customers
    x = model.addVars(range(num_vehicles), range(num_customers + 1), range(num_customers + 1), num_periods, vtype=GRB.BINARY, name='x')  # Vehicle routing
    y = model.addVars(range(num_vehicles), range(num_customers + 1), num_periods, vtype=GRB.BINARY, name='y')  # Vehicle visiting customers

    # Objective function
    travel_cost = gp.quicksum(distance_matrix[i][j] * x[v, i, j, t] for v in range(num_vehicles) for i in range(num_customers + 1) for j in range(num_customers + 1) for t in range(num_periods))
    holding_cost_supplier = gp.quicksum(depot['holding_cost'] * I_s[t] for t in range(1, num_periods + 1))
    holding_cost_customers = gp.quicksum(customers[c]['holding_cost'] * I_c[c, t] for c in range(num_customers) for t in range(1, num_periods + 1))
    model.setObjective(travel_cost + holding_cost_supplier + holding_cost_customers, GRB.MINIMIZE)

    # Constraints
    # Inventory balance at supplier
    model.addConstrs((I_s[t] == I_s[t - 1] + depot['production_per_period'] - gp.quicksum(q[c, t] for c in range(num_customers)) for t in range(1, num_periods + 1)), name='supplier_inventory_balance')
    # Inventory balance at customers
    model.addConstrs((I_c[c, t] == I_c[c, t - 1] + q[c, t] - customers[c]['demand_per_period'] for c in range(num_customers) for t in range(1, num_periods + 1)), name='customer_inventory_balance')
    # Inventory capacity at supplier
    model.addConstrs((I_s[t] <= depot['max_inventory'] for t in range(num_periods + 1)), name='supplier_inventory_capacity')
    model.addConstrs((I_s[t] >= 0 for t in range(num_periods + 1)), name='supplier_inventory_non_negative')
    # Inventory capacity at customers
    model.addConstrs((I_c[c, t] <= customers[c]['max_inventory'] for c in range(num_customers) for t in range(num_periods + 1)), name='customer_inventory_capacity')
    model.addConstrs((I_c[c, t] >= customers[c]['min_inventory'] for c in range(num_customers) for t in range(num_periods + 1)), name='customer_inventory_min')
    # Vehicle capacity
    model.addConstrs((gp.quicksum(q[c, t] for c in range(num_customers) if y[v, c + 1, t] == 1) <= vehicle_capacity for v in range(num_vehicles) for t in range(num_periods)), name='vehicle_capacity')
    # Routing constraints
    model.addConstrs((gp.quicksum(x[v, i, j, t] for j in range(num_customers + 1)) == gp.quicksum(x[v, j, i, t] for j in range(num_customers + 1)) for v in range(num_vehicles) for i in range(num_customers + 1) for t in range(num_periods)), name='routing_flow_conservation')
    model.addConstrs((gp.quicksum(x[v, 0, j, t] for j in range(num_customers + 1)) == 1 for v in range(num_vehicles) for t in range(num_periods)), name='routing_depot_departure')
    model.addConstrs((gp.quicksum(x[v, i, 0, t] for i in range(num_customers + 1)) == 1 for v in range(num_vehicles) for t in range(num_periods)), name='routing_depot_return')
    model.addConstrs((y[v, i + 1, t] <= gp.quicksum(x[v, j, i + 1, t] for j in range(num_customers + 1)) for v in range(num_vehicles) for i in range(num_customers) for t in range(num_periods)), name='routing_visit')
    model.addConstrs((gp.quicksum(x[v, i + 1, j, t] for v in range(num_vehicles) for j in range(num_customers + 1)) <= 1 for i in range(num_customers) for t in range(num_periods)), name='customer_visit_at_most_once')
    model.addConstrs((q[c, t] <= customers[c]['max_inventory'] - I_c[c, t - 1] for c in range(num_customers) for t in range(num_periods)), name='delivery_within_customer_capacity')
    model.addConstrs((q[c, t] <= gp.quicksum(vehicle_capacity * y[v, c + 1, t] for v in range(num_vehicles)) for c in range(num_customers) for t in range(num_periods)), name='delivery_by_assigned_vehicle')

    # Subtour elimination
    for v in range(num_vehicles):
        for t in range(num_periods):
            model.addConstr(gp.quicksum(x[v, i, j, t] for i in range(num_customers + 1) for j in range(num_customers + 1) if i != j) >= gp.quicksum(y[v, i + 1, t] for i in range(num_customers)), name=f'subtour_elimination_{v}_{t}')

    # Initial conditions
    model.addConstr(I_s[0] == depot['initial_inventory'], name='initial_supplier_inventory')
    for c in range(num_customers):
        model.addConstr(I_c[c, 0] == customers[c]['initial_inventory'], name=f'initial_customer_inventory_{c}')

    model.optimize()

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        solution = construct_solution(instance, model, I_s, I_c, q, x, y)
        return solution
    else:
        return None

def construct_solution(instance, model, I_s, I_c, q, x, y):
    num_customers = instance['num_customers']
    num_periods = instance['num_periods']
    num_vehicles = instance['num_vehicles']
    customers = instance['customers']

    solution = {'periods': {}}
    objective_value = model.ObjVal

    for t in range(num_periods):
        period = t + 1
        solution['periods'][str(period)] = {'routes': [], 'deliveries': {}, 'inventories': {}}
        routes = []
        for v in range(num_vehicles):
            route = []
            current = 0
            route.append(current)
            next_node = None
            for j in range(num_customers + 1):
                if x[v, 0, j, t].x > 0.5:
                    next_node = j
                    break
            while next_node != 0:
                route.append(next_node)
                for j in range(num_customers + 1):
                    if x[v, next_node, j, t].x > 0.5:
                        next_node = j
                        break
            if len(route) > 2:
                routes.append(route)
        solution['periods'][str(period)]['routes'] = routes

        for c in range(num_customers):
            if q[c, t].x > 0:
                solution['periods'][str(period)]['deliveries'][str(c + 1)] = q[c, t].x

        solution['periods'][str(period)]['inventories']['depot'] = I_s[t + 1].x
        for c in range(num_customers):
            solution['periods'][str(period)]['inventories'][str(c + 1)] = I_c[c, t + 1].x

    solution = {'objective_value': objective_value, 'solution_details': solution}

    return solution

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, required=False)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    instance = read_instance(args.instance_path)
    solution = solve_instance(instance, args.time_limit)

    if solution is not None:
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
        if logger:
            logger.log(solution['objective_value'])
    else:
        print("No solution found.")

if __name__ == '__main__':
    main()
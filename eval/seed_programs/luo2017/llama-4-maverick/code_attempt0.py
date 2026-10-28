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
    vehicle_capacity = instance['vehicle_capacity']
    depot_id = instance['depot']['id']
    customers = instance['customers']
    distance_matrix = np.array(instance['distance_matrix'])
    cost_a = instance['cost_function']['a']
    cost_b = instance['cost_function']['b']

    model = gp.Model('VRPSD')
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = time_limit

    # Sets
    Locations = list(range(num_customers + 1))
    Customers = list(range(1, num_customers + 1))
    Vehicles = model.addVars(num_customers + 1, vtype=GRB.INTEGER, name='Vehicles').getAttr('X')
    max_vehicles = max(Vehicles)
    Vehicles = list(range(max_vehicles))

    # Variables
    x = model.addVars(Vehicles, Locations, Locations, vtype=GRB.BINARY, name='x')
    l = model.addVars(Vehicles, Locations, Locations, vtype=GRB.CONTINUOUS, name='l', lb=0)
    d = model.addVars(Vehicles, Customers, vtype=GRB.CONTINUOUS, name='d', lb=0)
    t = model.addVars(Vehicles, Locations, vtype=GRB.CONTINUOUS, name='t')

    # Objective
    model.setObjective(gp.quicksum(
        distance_matrix[i, j] * (cost_a * l[v, i, j] + cost_b) * x[v, i, j]
        for v in Vehicles for i in Locations for j in Locations
    ), GRB.MINIMIZE)

    # Constraints
    model.addConstrs((gp.quicksum(x[v, depot_id, j] for j in Locations if j != depot_id) <= 1 for v in Vehicles), name='vehicle_depot_departure')
    model.addConstrs((gp.quicksum(x[v, i, depot_id] for i in Locations if i != depot_id) <= 1 for v in Vehicles), name='vehicle_depot_arrival')
    model.addConstrs((gp.quicksum(x[v, i, j] for v in Vehicles for j in Locations if j != i) == gp.quicksum(x[v, j, i] for v in Vehicles for j in Locations if j != i) for i in Customers), name='customer_visit_balance')
    model.addConstrs((d[v, i] <= vehicle_capacity * x[v, depot_id, i] for v in Vehicles for i in Customers), name='delivery_capacity')
    model.addConstrs((gp.quicksum(d[v, i] for v in Vehicles) == customers[i-1]['demand'] for i in Customers), name='demand_satisfaction')
    model.addConstrs((l[v, i, j] <= vehicle_capacity * x[v, i, j] for v in Vehicles for i in Locations for j in Locations), name='load_capacity')
    model.addConstrs((l[v, i, j] >= 0 for v in Vehicles for i in Locations for j in Locations), name='non_negative_load')
    model.addConstrs((gp.quicksum(l[v, j, i] for j in Locations if j != i) - gp.quicksum(l[v, i, j] for j in Locations if j != i) == d[v, i] for v in Vehicles for i in Customers), name='load_balance')
    model.addConstrs((t[v, j] >= t[v, i] + instance['depot']['service_time'] + distance_matrix[i, j] - (1 - x[v, i, j]) * instance['scheduling_horizon'] for v in Vehicles for i in Locations for j in Locations if i != j), name='time_consistency')
    model.addConstrs((t[v, i] >= instance['depot']['time_window'][0] for v in Vehicles for i in [depot_id]), name='depot_earliest_departure')
    model.addConstrs((t[v, i] <= instance['depot']['time_window'][1] for v in Vehicles for i in [depot_id]), name='depot_latest_arrival')
    model.addConstrs((t[v, i] >= customers[i-1]['time_window'][0] for v in Vehicles for i in Customers), name='customer_earliest_service')
    model.addConstrs((t[v, i] <= customers[i-1]['time_window'][1] for v in Vehicles for i in Customers), name='customer_latest_service')

    model.optimize()

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        objective_value = model.ObjVal
        solution = {'objective_value': objective_value, 'routes': []}

        for v in Vehicles:
            if gp.quicksum(x[v, depot_id, j].X for j in Locations if j != depot_id) > 0.5:
                route = [depot_id]
                deliveries = {}
                edges = []
                current_location = depot_id
                while True:
                    next_location = None
                    for j in Locations:
                        if x[v, current_location, j].X > 0.5:
                            next_location = j
                            break
                    if next_location == depot_id:
                        route.append(num_customers + 1)
                        edges.append({'from': current_location, 'to': next_location, 'load_weight': l[v, current_location, next_location].X})
                        break
                    else:
                        route.append(next_location)
                        if next_location in Customers:
                            deliveries[str(next_location)] = d[v, next_location].X
                        edges.append({'from': current_location, 'to': next_location, 'load_weight': l[v, current_location, next_location].X})
                        current_location = next_location
                solution['routes'].append({'vehicle': v, 'route': route, 'deliveries': deliveries, 'edges': edges})

        return objective_value, solution
    else:
        return None, None

def main():
    parser = argparse.ArgumentParser(description='Solve VRPSD instance')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the instance JSON file')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to write the solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds')
    parser.add_argument('--log_path', type=str, help='Path to log intermediate solutions (optional)')
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense='minimize') if args.log_path else None

    instance = read_instance(args.instance_path)
    objective_value, solution = solve_instance(instance, args.time_limit)

    if objective_value is not None and solution is not None:
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
        if logger:
            logger.log_solution(objective_value, solution)

if __name__ == '__main__':
    main()
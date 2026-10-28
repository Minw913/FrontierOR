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

def solve_instance(instance, time_limit, logger):
    num_customers = instance['num_customers']
    num_vehicles = instance['num_vehicles']
    vehicle_capacity = instance['vehicle_capacity']
    depot_start_id = instance['depot']['id']
    depot_end_id = instance['depot_end']['id']
    customers = instance['customers']
    distance_matrix = np.array(instance['distance_matrix'])
    rho = instance['inconvenience_cost_function']['rho']

    model = gp.Model('VRPTW')
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = time_limit

    # Variables
    x = model.addVars(num_vehicles, num_customers + 2, num_customers + 2, vtype=GRB.BINARY, name='x')
    s = model.addVars(num_vehicles, num_customers + 2, vtype=GRB.CONTINUOUS, name='s')
    y = model.addVars(num_customers + 1, vtype=GRB.CONTINUOUS, name='y')

    # Objective
    model.setObjective(
        gp.quicksum(distance_matrix[i, j] * x[v, i, j] for v in range(num_vehicles) for i in range(num_customers + 2) for j in range(num_customers + 2)) +
        gp.quicksum(rho * (y[c] - customers[c-1]['preferred_time']) ** 2 for c in range(1, num_customers+1)),
        GRB.MINIMIZE
    )

    # Constraints
    model.addConstrs((gp.quicksum(x[v, i, c] for v in range(num_vehicles) for i in range(num_customers + 2)) == 1 for c in range(1, num_customers + 1)), name='visit_once')
    model.addConstrs((gp.quicksum(x[v, 0, j] for j in range(1, num_customers + 2)) <= 1 for v in range(num_vehicles)), name='leave_depot')
    model.addConstrs((gp.quicksum(x[v, i, 0] for i in range(num_customers + 2)) == 0 for v in range(num_vehicles)), name='return_to_depot_not_allowed')
    model.addConstrs((gp.quicksum(x[v, i, num_customers + 1] for i in range(num_customers + 1)) <= 1 for v in range(num_vehicles)), name='return_to_end_depot')
    model.addConstrs((gp.quicksum(x[v, num_customers + 1, i] for i in range(num_customers + 2)) == 0 for v in range(num_vehicles)), name='leave_end_depot_not_allowed')
    model.addConstrs((gp.quicksum(x[v, i, c] for i in range(num_customers + 2)) - gp.quicksum(x[v, c, j] for j in range(num_customers + 2)) == 0 for v in range(num_vehicles) for c in range(1, num_customers + 1)), name='flow_conservation')
    model.addConstrs((s[v, i] + distance_matrix[i, j] + (customers[j - 1]['service_time'] if j > 0 and j <= num_customers else 0) <= s[v, j] + (1 - x[v, i, j]) * (instance['planning_horizon'] + distance_matrix[i, j] + (customers[j - 1]['service_time'] if j > 0 and j <= num_customers else 0)) for v in range(num_vehicles) for i in range(num_customers + 2) for j in range(1, num_customers + 2)), name='time_consistency')
    model.addConstrs((s[v, 0] >= instance['depot']['time_window'][0] for v in range(num_vehicles)), name='depot_start_time_lower_bound')
    model.addConstrs((s[v, 0] <= instance['depot']['time_window'][1] for v in range(num_vehicles)), name='depot_start_time_upper_bound')
    model.addConstrs((s[v, num_customers + 1] >= instance['depot_end']['time_window'][0] for v in range(num_vehicles)), name='depot_end_time_lower_bound')
    model.addConstrs((s[v, num_customers + 1] <= instance['depot_end']['time_window'][1] for v in range(num_vehicles)), name='depot_end_time_upper_bound')
    model.addConstrs((s[v, c] >= customers[c - 1]['time_window'][0] for v in range(num_vehicles) for c in range(1, num_customers + 1)), name='customer_time_window_lower_bound')
    model.addConstrs((s[v, c] <= customers[c - 1]['time_window'][1] for v in range(num_vehicles) for c in range(1, num_customers + 1)), name='customer_time_window_upper_bound')
    model.addConstrs((y[c] == gp.quicksum(s[v, c] for v in range(num_vehicles)) for c in range(1, num_customers + 1)), name='service_time')
    model.addConstrs((gp.quicksum(customers[c - 1]['demand'] * gp.quicksum(x[v, i, c] for i in range(num_customers + 2)) for c in range(1, num_customers + 1)) <= vehicle_capacity for v in range(num_vehicles)), name='capacity_constraint')

    model.optimize()

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        objective_value = model.ObjVal
        routes = []
        schedule = {}
        for v in range(num_vehicles):
            route = []
            for i in range(num_customers + 2):
                for j in range(num_customers + 2):
                    if x[v, i, j].x > 0.5:
                        if i not in route and i != num_customers + 1:
                            route.append(i)
                        if j not in route and j != num_customers + 1 and j != 0:
                            route.append(j)
            if len(route) > 0:
                route.insert(0, 0)
                route.append(num_customers + 1)
                routes.append(route)
                for stop in route:
                    if stop != 0 and stop != num_customers + 1:
                        schedule[stop] = s[v, stop].x
        solution = {
            'objective_value': objective_value,
            'routes': routes,
            'schedule': {f'{customer_id}': time for customer_id, time in schedule.items()}
        }
        if logger:
            logger.log_solution(objective_value, solution)
        return solution
    else:
        return None

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

    if solution is not None:
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=4)

if __name__ == '__main__':
    main()
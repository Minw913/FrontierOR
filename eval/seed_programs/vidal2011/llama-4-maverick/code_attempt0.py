import argparse
import json
import numpy as np
from solution_logger import SolutionLogger
import gurobipy as gp
from gurobipy import GRB

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        return json.load(f)

def calculate_distance(loc1, loc2):
    return np.sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)

def solve(instance, time_limit, logger):
    n = instance['n']
    d = instance['d']
    t = instance['t']
    m = instance['m']
    vehicle_capacity = instance['vehicle_capacity']
    route_duration_limit = instance['route_duration_limit']
    depots = instance['depots']
    customers = instance['customers']

    depot_coords = [(depot['x'], depot['y']) for depot in depots]
    customer_coords = [(customer['x'], customer['y']) for customer in customers]
    all_coords = depot_coords + customer_coords

    distances = np.zeros((d + n, d + n))
    for i in range(d + n):
        for j in range(d + n):
            distances[i, j] = calculate_distance(all_coords[i], all_coords[j])

    model = gp.Model('VRP')
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = time_limit

    # Assignment variables
    assign_vars = model.addVars(n, d, t, vtype=GRB.BINARY, name='assign')

    # Routing variables
    route_vars = model.addVars(d, t, m, n + d, n + d, vtype=GRB.BINARY, name='route')

    # Objective function
    model.setObjective(gp.quicksum(distances[i + d, j + d] * route_vars[k, p, v, i + d, j + d] for k in range(d) for p in range(t) for v in range(m) for i in range(n) for j in range(n) if i != j), GRB.MINIMIZE)

    # Constraints
    for i in range(n):
        customer = customers[i]
        for pattern_idx, pattern in enumerate(customer['pattern_list']):
            for depot in range(d):
                for period in range(t):
                    if period + 1 not in pattern:
                        model.addConstr(assign_vars[i, depot, period] == 0)

        # Each customer is assigned to exactly one depot and pattern
        model.addConstr(gp.quicksum(assign_vars[i, depot, period] for depot in range(d) for period in range(t)) == customer['frequency'])

        # Consistency across periods for the same customer
        for depot in range(d):
            count = gp.quicksum(assign_vars[i, depot, period] for period in range(t))
            model.addConstr(count <= customer['frequency'] * assign_vars.sum(i, depot, '*'))

    for k in range(d):
        for p in range(t):
            # Vehicle capacity constraint
            for v in range(m):
                model.addConstr(gp.quicksum(customers[i]['demand'] * route_vars[k, p, v, i + d, j + d] for i in range(n) for j in range(n + d) if i != j) <= vehicle_capacity)

            # Route duration constraint
            for v in range(m):
                route_duration = gp.quicksum((distances[i + d, j + d] + customers[j]['service_duration'] if j < n else 0) * route_vars[k, p, v, i + d, j + d] for i in range(n + d) for j in range(n + d) if i != j)
                model.addConstr(route_duration <= route_duration_limit)

            # Routing constraints
            model.addConstr(route_vars.sum(k, p, v, '*', k + n + d) <= 1)  # Depart from depot
            model.addConstr(route_vars.sum(k, p, v, k + n + d, '*') <= 1)  # Return to depot
            for i in range(n):
                model.addConstr(route_vars.sum(k, p, v, '*', i + d) == route_vars.sum(k, p, v, i + d, '*'))  # Flow conservation
                model.addConstr(route_vars.sum(k, p, v, '*', i + d) <= 1)  # Visit customer at most once

            # Linking assignment and routing variables
            for i in range(n):
                model.addConstr(route_vars.sum(k, p, v, '*', i + d) <= assign_vars.sum(i, k, p))

    model.optimize()

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        objective_value = model.ObjVal
        solution = {
            'objective_value': objective_value,
            'routes': [],
            'assignments': {}
        }

        for k in range(d):
            for p in range(t):
                for v in range(m):
                    route = []
                    i = k + n + d
                    while True:
                        for j in range(n + d):
                            if route_vars[k, p, v, i, j].x > 0.5:
                                if j < n + d:
                                    if j >= d:
                                        route.append(j - d)
                                    i = j
                                break
                        if i == k + n + d:
                            break

                    if route:
                        solution['routes'].append({
                            'depot': k,
                            'period': p + 1,
                            'vehicle': v + 1,
                            'customers': route
                        })

        for i in range(n):
            for k in range(d):
                for p in range(t):
                    if assign_vars[i, k, p].x > 0.5:
                        if str(i) not in solution['assignments']:
                            solution['assignments'][str(i)] = {'depot': k, 'pattern': []}
                        solution['assignments'][str(i)]['pattern'].append(p + 1)

        return objective_value, solution
    else:
        return None, None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, required=False)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense='minimize') if args.log_path else None

    instance = read_instance(args.instance_path)
    objective_value, solution = solve(instance, args.time_limit, logger)

    if objective_value is not None and solution is not None:
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)

        if logger:
            logger.log_solution(objective_value, solution)

if __name__ == '__main__':
    main()
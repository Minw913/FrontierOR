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
    assign_vars = model.addVars(n, d, vtype=GRB.BINARY, name='assign')
    visit_vars = model.addVars(n, t, vtype=GRB.BINARY, name='visit')

    # Routing variables
    route_vars = model.addVars(d, t, m, d + n, d + n, vtype=GRB.BINARY, name='route')

    # Objective function
    model.setObjective(gp.quicksum(distances[i, j] * route_vars[k, p, v, i, j] for k in range(d) for p in range(t) for v in range(m) for i in range(d + n) for j in range(d + n) if i != j), GRB.MINIMIZE)

    # Constraints
    for i in range(n):
        customer = customers[i]
        # Each customer is assigned to exactly one depot
        model.addConstr(gp.quicksum(assign_vars[i, k] for k in range(d)) == 1)

        # Visit pattern constraints
        for p in range(t):
            model.addConstr(visit_vars[i, p] <= gp.quicksum(assign_vars[i, k] for k in range(d)))
        for pattern in customer['pattern_list']:
            model.addConstr(gp.quicksum(visit_vars[i, p] for p in range(t) if p + 1 in pattern) >= 1)
        model.addConstr(gp.quicksum(visit_vars[i, p] for p in range(t)) == customer['frequency'])

        # Linking assignment and visit variables
        for k in range(d):
            for p in range(t):
                model.addConstr(visit_vars[i, p] <= assign_vars[i, k])

    for k in range(d):
        for p in range(t):
            # Vehicle capacity constraint
            for v in range(m):
                model.addConstr(gp.quicksum(customers[i]['demand'] * route_vars[k, p, v, i + d, j] for i in range(n) for j in range(d + n) if i + d != j) <= vehicle_capacity)

            # Route duration constraint
            for v in range(m):
                route_duration = gp.quicksum((distances[i, j] + (customers[j - d]['service_duration'] if j >= d else 0)) * route_vars[k, p, v, i, j] for i in range(d + n) for j in range(d + n) if i != j)
                model.addConstr(route_duration <= route_duration_limit)

            # Routing constraints
            for v in range(m):
                model.addConstr(route_vars.sum(k, p, v, k, '*') == 1)  # Depart from depot
                model.addConstr(route_vars.sum(k, p, v, '*', k) == 1)  # Return to depot
                for i in range(d + n):
                    if i == k:
                        continue
                    model.addConstr(route_vars.sum(k, p, v, '*', i) == route_vars.sum(k, p, v, i, '*'))  # Flow conservation

            # Linking visit and routing variables
            for i in range(n):
                for v in range(m):
                    model.addConstr(route_vars.sum(k, p, v, '*', i + d) <= visit_vars[i, p])
                    model.addConstr(route_vars.sum(k, p, v, '*', i + d) <= assign_vars[i, k])

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
                    i = k
                    while True:
                        for j in range(d + n):
                            if route_vars[k, p, v, i, j].x > 0.5:
                                if j >= d:
                                    route.append(j - d)
                                i = j
                                break
                        if i == k:
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
                if assign_vars[i, k].x > 0.5:
                    solution['assignments'][str(i)] = {'depot': k, 'pattern': [p + 1 for p in range(t) if visit_vars[i, p].x > 0.5]}

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
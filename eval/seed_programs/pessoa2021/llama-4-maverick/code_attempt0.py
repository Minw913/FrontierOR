import json
import argparse
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger
import time

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    return instance

def solve(instance, time_limit, logger):
    num_customers = instance['num_customers']
    num_vehicles = instance['num_vehicles']
    vehicle_capacity = instance['vehicle_capacity']
    Gamma = instance['uncertainty_set']['Gamma']
    customers = instance['customers']
    distance_matrix = instance['distance_matrix']

    model = Model('RobustCVRP')
    model.Params.TimeLimit = time_limit
    model.Params.LogToConsole = 0

    # Variables
    x = model.addVars(num_customers + 1, num_customers + 1, num_vehicles, vtype=GRB.BINARY)
    y = model.addVars(num_customers, num_vehicles, vtype=GRB.BINARY)
    z = model.addVars(num_customers, vtype=GRB.BINARY)
    w = model.addVars(num_vehicles, vtype=GRB.CONTINUOUS)

    # Objective
    model.setObjective(quicksum(distance_matrix[i][j] * x[i, j, k] for i in range(num_customers + 1) for j in range(num_customers + 1) for k in range(num_vehicles)))

    # Constraints
    for i in range(1, num_customers + 1):
        model.addConstr(quicksum(y[i - 1, k] for k in range(num_vehicles)) == 1)

    for k in range(num_vehicles):
        model.addConstr(quicksum(y[i - 1, k] for i in range(1, num_customers + 1)) <= 1)

    for k in range(num_vehicles):
        model.addConstr(quicksum(x[0, j, k] for j in range(1, num_customers + 1)) == quicksum(x[i, 0, k] for i in range(1, num_customers + 1)))
        model.addConstr(quicksum(x[0, j, k] for j in range(1, num_customers + 1)) <= 1)

    for i in range(num_customers + 1):
        for k in range(num_vehicles):
            model.addConstr(quicksum(x[i, j, k] for j in range(num_customers + 1)) == quicksum(x[j, i, k] for j in range(num_customers + 1)))

    for i in range(1, num_customers + 1):
        for k in range(num_vehicles):
            model.addConstr(x[i, 0, k] + quicksum(x[j, i, k] for j in range(num_customers + 1) if j != i) == y[i - 1, k])

    for i in range(1, num_customers + 1):
        for k in range(num_vehicles):
            model.addConstr(x[0, i, k] + quicksum(x[j, i, k] for j in range(num_customers + 1) if j != i) == quicksum(x[i, j, k] for j in range(num_customers + 1)))

    for k in range(num_vehicles):
        model.addConstr(quicksum(customers[i - 1]['nominal_demand'] * y[i - 1, k] for i in range(1, num_customers + 1)) + 
                        quicksum(customers[i - 1]['demand_deviation'] * z[i - 1] for i in range(1, num_customers + 1)) <= vehicle_capacity)

    for k in range(num_vehicles):
        model.addConstr(w[k] >= quicksum(customers[i - 1]['demand_deviation'] * y[i - 1, k] for i in range(1, num_customers + 1)) - Gamma)

    for i in range(1, num_customers + 1):
        model.addConstr(z[i - 1] >= y[i - 1, k] - (1 - y[i - 1, k]) * 10000)
        model.addConstr(z[i - 1] <= y[i - 1, k])

    model.addConstr(quicksum(y[i - 1, k] for i in range(1, num_customers + 1) for k in range(num_vehicles)) == num_customers)
    model.addConstr(quicksum(quicksum(x[0, j, k] for j in range(1, num_customers + 1)) for k in range(num_vehicles)) == num_vehicles)

    # Symmetry breaking
    for k in range(num_vehicles - 1):
        model.addConstr(quicksum(x[0, j, k] for j in range(1, num_customers + 1)) >= quicksum(x[0, j, k + 1] for j in range(1, num_customers + 1)))

    model.optimize()

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        objective_value = model.getObjective().getValue()
        if logger:
            logger.log(objective_value)

        routes = []
        for k in range(num_vehicles):
            route = []
            customers_on_route = []
            for i in range(1, num_customers + 1):
                if y[i - 1, k].x > 0.5:
                    customers_on_route.append(i)
            current_node = 0
            while customers_on_route:
                next_node = min(customers_on_route, key=lambda i: distance_matrix[current_node][i])
                route.append(next_node)
                customers_on_route.remove(next_node)
                current_node = next_node
            cost = distance_matrix[0][route[0]] if route else 0
            for i in range(len(route) - 1):
                cost += distance_matrix[route[i]][route[i + 1]]
            if route:
                cost += distance_matrix[route[-1]][0]
                worst_case_demand = sum(customers[i - 1]['nominal_demand'] for i in route) + min(Gamma, len(route)) * max(customers[i - 1]['demand_deviation'] for i in route)
            else:
                worst_case_demand = 0
            routes.append({
                'customers': [i for i in route],
                'cost': cost,
                'worst_case_demand': worst_case_demand
            })

        return {
            'objective_value': objective_value,
            'routes': routes
        }
    else:
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str)
    args = parser.parse_args()

    if args.log_path:
        logger = SolutionLogger(args.log_path, sense="minimize")
    else:
        logger = None

    instance = read_instance(args.instance_path)
    start_time = time.time()
    solution = solve(instance, args.time_limit, logger)
    end_time = time.time()

    if solution is not None:
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
    else:
        print("No solution found.")

if __name__ == '__main__':
    main()
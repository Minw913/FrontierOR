import argparse
import json
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger

def read_instance_data(instance_path):
    with open(instance_path, 'r') as file:
        instance_data = json.load(file)
    return instance_data

def solve_milp(instance_data, time_limit, logger):
    num_linehaul = instance_data['num_linehaul_customers']
    num_backhaul = instance_data['num_backhaul_customers']
    num_total_customers = instance_data['num_total_customers']
    num_vehicles = instance_data['num_vehicles']
    vehicle_capacity = instance_data['vehicle_capacity']
    distance_matrix = instance_data['distance_matrix']
    depot_index = instance_data['depot']['index']
    linehaul_customers = instance_data['linehaul_customers']
    backhaul_customers = instance_data['backhaul_customers']

    model = Model('VRP')
    model.setParam("Threads", 1)
    model.Params.LogToConsole = 0  
    model.Params.TimeLimit = time_limit

    x = model.addVars(num_vehicles, num_total_customers + 1, num_total_customers + 1, vtype=GRB.BINARY, name='x')
    y = model.addVars(num_vehicles, num_linehaul, vtype=GRB.BINARY, name='y')
    z = model.addVars(num_vehicles, num_backhaul, vtype=GRB.BINARY, name='z')
    u = model.addVars(num_vehicles, num_linehaul + 1, vtype=GRB.INTEGER, name='u')

    model.setObjective(quicksum(distance_matrix[i][j] * x[k, i, j] for k in range(num_vehicles) for i in range(num_total_customers + 1) for j in range(num_total_customers + 1)), GRB.MINIMIZE)

    model.addConstrs((quicksum(y[k, i] for k in range(num_vehicles)) == 1 for i in range(num_linehaul)), name='linehaul_visited_once')
    model.addConstrs((quicksum(z[k, i] for k in range(num_vehicles)) == 1 for i in range(num_backhaul)), name='backhaul_visited_once')

    for k in range(num_vehicles):
        model.addConstr(quicksum(x[k, depot_index, j] for j in range(num_total_customers + 1)) == 1, name='vehicle_leaves_depot')
        model.addConstr(quicksum(x[k, i, depot_index] for i in range(num_total_customers + 1)) == 1, name='vehicle_returns_depot')

    for k in range(num_vehicles):
        for i in range(num_linehaul):
            model.addConstr(quicksum(x[k, j, linehaul_customers[i]['index']] for j in range(num_total_customers + 1)) == y[k, i], name='linehaul_inflow')
            model.addConstr(quicksum(x[k, linehaul_customers[i]['index'], j] for j in range(num_total_customers + 1)) == y[k, i], name='linehaul_outflow')

    for k in range(num_vehicles):
        for i in range(num_backhaul):
            model.addConstr(quicksum(x[k, j, backhaul_customers[i]['index']] for j in range(num_total_customers + 1)) == z[k, i], name='backhaul_inflow')
            model.addConstr(quicksum(x[k, backhaul_customers[i]['index'], j] for j in range(num_total_customers + 1)) == z[k, i], name='backhaul_outflow')

    for k in range(num_vehicles):
        model.addConstr(quicksum(linehaul_customers[i]['demand'] * y[k, i] for i in range(num_linehaul)) <= vehicle_capacity, name='linehaul_capacity')
        model.addConstr(quicksum(backhaul_customers[i]['demand'] * z[k, i] for i in range(num_backhaul)) <= vehicle_capacity, name='backhaul_capacity')

    for k in range(num_vehicles):
        for i in range(num_linehaul):
            for j in range(num_linehaul):
                if i != j:
                    model.addConstr(u[k, i + 1] - u[k, j + 1] + num_linehaul * x[k, linehaul_customers[i]['index'], linehaul_customers[j]['index']] <= num_linehaul - 1, name='linehaul_mtz')

    for k in range(num_vehicles):
        model.addConstr(u[k, 0] == 0, name='linehaul_mtz_start')
        for i in range(num_linehaul):
            model.addConstr(u[k, i + 1] >= 1, name='linehaul_mtz_nonzero')
            model.addConstr(u[k, i + 1] <= num_linehaul, name='linehaul_mtz_bound')

    model.optimize()

    if logger:
        for i in range(model.SolCount):
            model.Params.SolutionNumber = i
            objective_value = model.PoolObjVal
            logger.log(objective_value)

    routes = []
    if model.SolCount > 0:
        for k in range(num_vehicles):
            route = [depot_index]
            linehaul_route = []
            backhaul_route = []
            for i in range(num_linehaul):
                if y[k, i].x > 0.5:
                    linehaul_route.append(linehaul_customers[i]['index'])
            for i in range(num_backhaul):
                if z[k, i].x > 0.5:
                    backhaul_route.append(backhaul_customers[i]['index'])

            # Simple construction of route
            current = depot_index
            for customer in linehaul_route:
                route.append(customer)
                current = customer
            for customer in backhaul_route:
                route.append(customer)
                current = customer
            route.append(depot_index)

            if len(route) > 2:
                routes.append(route)

    objective_value = model.ObjVal if model.SolCount > 0 else float('inf')
    return objective_value, routes

def main():
    parser = argparse.ArgumentParser(description='Solve VRP')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the instance JSON file')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to the output solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds')
    parser.add_argument('--log_path', type=str, help='Path to the log file for intermediate solutions')
    args = parser.parse_args()

    instance_data = read_instance_data(args.instance_path)
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    objective_value, routes = solve_milp(instance_data, args.time_limit, logger)

    solution = {
        'objective_value': objective_value,
        'routes': routes
    }

    with open(args.solution_path, 'w') as file:
        json.dump(solution, file, indent=4)

if __name__ == '__main__':
    main()
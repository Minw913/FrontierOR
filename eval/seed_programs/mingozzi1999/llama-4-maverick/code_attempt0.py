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
    num_customers = instance_data['num_total_customers']
    num_vehicles = instance_data['num_vehicles']
    vehicle_capacity = instance_data['vehicle_capacity']
    distance_matrix = instance_data['distance_matrix']
    depot_index = instance_data['depot']['index']
    linehaul_customers = instance_data['linehaul_customers']
    backhaul_customers = instance_data['backhaul_customers']

    model = Model('VRP')
    model.Params.LogToConsole = 0  # Disable Gurobi console output
    model.Params.TimeLimit = time_limit

    # Decision Variables
    x = model.addVars(num_vehicles, num_customers + 1, num_customers + 1, vtype=GRB.BINARY, name='x')
    y = model.addVars(num_vehicles, num_linehaul_customers, vtype=GRB.BINARY, name='y')
    z = model.addVars(num_vehicles, num_backhaul_customers, vtype=GRB.BINARY, name='z')
    u = model.addVars(num_vehicles, num_linehaul_customers + 1, vtype=GRB.INTEGER, name='u')  # Subtour elimination for linehaul
    v = model.addVars(num_vehicles, num_backhaul_customers + 1, vtype=GRB.INTEGER, name='v')  # Subtour elimination for backhaul

    # Objective Function
    model.setObjective(quicksum(distance_matrix[i][j] * x[k, i, j] for k in range(num_vehicles) for i in range(num_customers + 1) for j in range(num_customers + 1)), GRB.MINIMIZE)

    # Constraints
    # 1. Each linehaul customer is visited exactly once
    model.addConstrs((quicksum(y[k, i] for k in range(num_vehicles)) == 1 for i in range(num_linehaul_customers)), name='linehaul_visited_once')

    # 2. Each backhaul customer is visited exactly once
    model.addConstrs((quicksum(z[k, i] for k in range(num_vehicles)) == 1 for i in range(num_backhaul_customers)), name='backhaul_visited_once')

    # 3. Flow conservation for linehaul customers
    for k in range(num_vehicles):
        for i in range(num_linehaul_customers):
            model.addConstr(quicksum(x[k, depot_index, linehaul_customers[i]['index']] + x[k, linehaul_customers[j]['index'], linehaul_customers[i]['index']] for j in range(num_linehaul_customers)) == y[k, i])

    # 4. Flow conservation for backhaul customers
    for k in range(num_vehicles):
        for i in range(num_backhaul_customers):
            model.addConstr(quicksum(x[k, backhaul_customers[i]['index'], depot_index] + x[k, backhaul_customers[i]['index'], backhaul_customers[j]['index']] for j in range(num_backhaul_customers)) + quicksum(x[k, linehaul_customers[j]['index'], backhaul_customers[i]['index']] for j in range(num_linehaul_customers) if y[k, j] == 1) == z[k, i])

    # 5. Linehaul before backhaul
    for k in range(num_vehicles):
        for i in range(num_linehaul_customers):
            for j in range(num_backhaul_customers):
                model.addConstr(x[k, linehaul_customers[i]['index'], backhaul_customers[j]['index']] <= y[k, i])

    # 6. Vehicle capacity constraints for linehaul and backhaul
    for k in range(num_vehicles):
        model.addConstr(quicksum(linehaul_customers[i]['demand'] * y[k, i] for i in range(num_linehaul_customers)) <= vehicle_capacity)
        model.addConstr(quicksum(backhaul_customers[i]['demand'] * z[k, i] for i in range(num_backhaul_customers)) <= vehicle_capacity)

    # 7. Subtour elimination for linehaul customers
    for k in range(num_vehicles):
        for i in range(num_linehaul_customers):
            for j in range(num_linehaul_customers):
                if i != j:
                    model.addConstr(u[k, i] - u[k, j] + num_linehaul_customers * x[k, linehaul_customers[i]['index'], linehaul_customers[j]['index']] <= num_linehaul_customers - 1)

    # 8. Subtour elimination for backhaul customers
    for k in range(num_vehicles):
        for i in range(num_backhaul_customers):
            for j in range(num_backhaul_customers):
                if i != j:
                    model.addConstr(v[k, i] - v[k, j] + num_backhaul_customers * x[k, backhaul_customers[i]['index'], backhaul_customers[j]['index']] <= num_backhaul_customers - 1)

    # 9. Each route starts and ends at the depot
    for k in range(num_vehicles):
        model.addConstr(quicksum(x[k, depot_index, j] for j in range(num_customers + 1)) == 1)
        model.addConstr(quicksum(x[k, i, depot_index] for i in range(num_customers + 1)) == 1)

    # Optimize the model
    model.optimize()

    # Log intermediate solutions
    if logger:
        for i in range(model.SolCount):
            model.Params.SolutionNumber = i
            objective_value = model.PoolObjVal
            logger.log(objective_value)

    # Extract solution
    routes = []
    if model.SolCount > 0:
        for k in range(num_vehicles):
            route = [depot_index]
            linehaul_route = []
            backhaul_route = []
            for i in range(num_linehaul_customers):
                if y[k, i].x > 0.5:
                    linehaul_route.append(linehaul_customers[i]['index'])
            for i in range(num_backhaul_customers):
                if z[k, i].x > 0.5:
                    backhaul_route.append(backhaul_customers[i]['index'])
            # Reconstruct the route respecting linehaul before backhaul order
            current = depot_index
            while linehaul_route:
                next_customer = None
                min_cost = float('inf')
                for customer in linehaul_route:
                    if x[k, current, customer].x > 0.5:
                        next_customer = customer
                        min_cost = distance_matrix[current][customer]
                        break
                    elif distance_matrix[current][customer] < min_cost:
                        min_cost = distance_matrix[current][customer]
                        next_customer = customer
                route.append(next_customer)
                linehaul_route.remove(next_customer)
                current = next_customer
            if backhaul_route:
                next_customer = None
                min_cost = float('inf')
                for customer in backhaul_route:
                    if distance_matrix[current][customer] < min_cost:
                        min_cost = distance_matrix[current][customer]
                        next_customer = customer
                route.append(next_customer)
                backhaul_route.remove(next_customer)
                current = next_customer
                while backhaul_route:
                    next_customer = None
                    min_cost = float('inf')
                    for customer in backhaul_route:
                        if x[k, current, customer].x > 0.5:
                            next_customer = customer
                            min_cost = distance_matrix[current][customer]
                            break
                        elif distance_matrix[current][customer] < min_cost:
                            min_cost = distance_matrix[current][customer]
                            next_customer = customer
                    route.append(next_customer)
                    backhaul_route.remove(next_customer)
                    current = next_customer
            route.append(depot_index)
            if len(route) > 2:  # Only add non-empty routes
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
import argparse
import json
import math
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as file:
        instance = json.load(file)
    return instance

def solve_vrptw(instance, time_limit):
    num_customers = instance['num_customers']
    vehicle_capacity = instance['vehicle_capacity']
    depot = instance['depot']
    customers = instance['customers']
    distance_matrix = instance['distance_matrix']
    scheduling_horizon = instance['scheduling_horizon']

    # Adjust distance matrix if necessary
    if not instance['travel_time_includes_service_time']:
        for i in range(num_customers + 1):
            for j in range(num_customers + 1):
                if i > 0:
                    distance_matrix[i][j] += customers[i - 1]['service_time']

    # Create model
    model = Model('VRPTW')
    model.setParam("Threads", 1)
    model.Params.TimeLimit = time_limit
    model.Params.LogToConsole = 0

    # Variables
    x = model.addVars(num_customers + 1, num_customers + 1, vtype=GRB.BINARY, name='x')
    departure_times = model.addVars(num_customers + 1, name='departure_times')
    loads = model.addVars(num_customers + 1, name='loads')
    num_vehicles = model.addVar(vtype=GRB.INTEGER, name='num_vehicles')

    # Objective: minimize number of vehicles
    model.setObjective(num_vehicles, GRB.MINIMIZE)

    # Constraints
    # Flow conservation
    model.addConstrs((quicksum(x[i, j] for j in range(num_customers + 1)) == 1 for i in range(1, num_customers + 1)), name='visit_once')
    model.addConstrs((quicksum(x[i, j] for i in range(num_customers + 1)) == 1 for j in range(1, num_customers + 1)), name='visited_once')
    model.addConstrs((quicksum(x[0, j] for j in range(1, num_customers + 1)) == num_vehicles), name='depot_out')
    model.addConstrs((quicksum(x[i, 0] for i in range(1, num_customers + 1)) == num_vehicles), name='depot_in')

    # Time window constraints
    model.addConstrs((departure_times[i] >= customers[i - 1]['ready_time'] for i in range(1, num_customers + 1)), name='ready_time')
    model.addConstrs((departure_times[i] <= customers[i - 1]['due_date'] for i in range(1, num_customers + 1)), name='due_date')
    model.addConstr(departure_times[0] == depot['ready_time'], name='depot_ready_time')

    # Time consistency
    model.addConstrs((departure_times[j] >= departure_times[i] + distance_matrix[i][j] - (1 - x[i, j]) * scheduling_horizon for i in range(num_customers + 1) for j in range(1, num_customers + 1)), name='time_consistency')

    # Capacity constraints
    model.addConstrs((loads[i] >= customers[i - 1]['demand'] for i in range(1, num_customers + 1)), name='demand_met')
    model.addConstr(loads[0] == 0, name='depot_load')
    model.addConstrs((loads[i] <= vehicle_capacity for i in range(num_customers + 1)), name='capacity')
    model.addConstrs((loads[j] >= loads[i] + customers[j - 1]['demand'] - (1 - x[i, j]) * vehicle_capacity for i in range(num_customers + 1) for j in range(1, num_customers + 1)), name='load_consistency')

    # Fix num_vehicles to be within reasonable bounds
    max_vehicles = num_customers
    model.addConstr(num_vehicles <= max_vehicles, name='max_vehicles')
    model.addConstr(num_vehicles >= 0, name='min_vehicles')
    model.addConstrs((x[i, j] == 0 for i in range(num_customers + 1) for j in range(num_customers + 1) if distance_matrix[i][j] > customers[i-1]['due_date'] - customers[j-1]['ready_time'] if i > 0 and j > 0), name='infeasible_arcs')

    # Optimize
    model.optimize()

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        # Extract solution
        solution = extract_solution(model, instance)
        return solution
    else:
        raise Exception(f"Model status {model.status} is not handled.")

def extract_solution(model, instance):
    num_customers = instance['num_customers']
    depot_id = instance['depot']['id']
    customers = instance['customers']

    # Extract routes
    x = model.getAttr('X', model.getVars()[: (num_customers + 1) * (num_customers + 1)])
    x = { (i, j): x[i * (num_customers + 1) + j] for i in range(num_customers + 1) for j in range(num_customers + 1) }
    routes = []
    visited = set()
    for _ in range(round(model.getVarByName('num_vehicles').X)):
        route = [depot_id]
        current = depot_id
        while True:
            next_customer = None
            for j in range(1, num_customers + 1):
                if x[current, j] > 0.5 and j not in visited:
                    next_customer = j
                    visited.add(j)
                    break
            if next_customer is None:
                break
            route.append(next_customer)
            current = next_customer
        route.append(depot_id)
        routes.append(route)

    # Extract other solution components
    departure_times = { var.VarName.split('[')[1].split(']')[0]: var.X for var in model.getVars() if 'departure_times' in var.VarName }
    loads = { var.VarName.split('[')[1].split(']')[0]: var.X for var in model.getVars() if 'loads' in var.VarName }

    # Convert to required output format
    solution = {
        'objective_value': model.getVarByName('num_vehicles').X,
        'num_vehicles': round(model.getVarByName('num_vehicles').X),
        'routes': routes,
        'total_distance': sum(x[i, j] * instance['distance_matrix'][i][j] for i in range(num_customers + 1) for j in range(num_customers + 1)),
        'departure_times': { str(int(key)): value for key, value in departure_times.items() if key != '0' },
        'loads': { str(int(key)): value for key, value in loads.items() if key != '0' }
    }

    return solution

def main():
    parser = argparse.ArgumentParser(description='Solve VRPTW instance.')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the instance JSON file.')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to write the solution JSON file.')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds.')
    parser.add_argument('--log_path', type=str, help='Path to log intermediate solutions (optional).')
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    instance = read_instance(args.instance_path)
    try:
        solution = solve_vrptw(instance, args.time_limit)
        with open(args.solution_path, 'w') as file:
            json.dump(solution, file, indent=4)
        if logger:
            logger.log(solution['objective_value'])
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
import argparse
import json
import math
from typing import List, Dict
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def calculate_distance(x1: int, y1: int, x2: int, y2: int) -> float:
    """Calculate Euclidean distance between two points."""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def read_instance(instance_path: str) -> Dict:
    """Read instance data from JSON file."""
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    return instance

def solve(instance: Dict, time_limit: int, logger: SolutionLogger = None) -> Dict:
    """Solve the VRPTW instance using Gurobi."""
    num_customers = instance['num_customers']
    depot = instance['depot']
    customers = instance['customers']
    vehicle_capacity = instance['vehicle_capacity']
    num_vehicles = instance['num_vehicles']

    # Sets and parameters
    N = num_customers + 1  # Including depot
    V = range(N)
    C = range(1, N)  # Customer indices
    K = range(num_vehicles)  # Vehicle indices
    depot_x, depot_y = depot['x'], depot['y']
    locations = [(depot_x, depot_y)] + [(c['x'], c['y']) for c in customers]
    demands = [0] + [c['demand'] for c in customers]
    ready_times = [depot['ready_time']] + [c['ready_time'] for c in customers]
    due_dates = [depot['due_date']] + [c['due_date'] for c in customers]
    service_times = [0] + [c['service_time'] for c in customers]

    # Model setup
    model = gp.Model('VRPTW')
    model.Params.LogToConsole = 0  # Suppress Gurobi console output

    # Variables
    x = model.addVars([(i, j, k) for i in V for j in V for k in K], vtype=GRB.BINARY, name='x')
    s = model.addVars([(i, k) for i in V for k in K], vtype=GRB.CONTINUOUS, name='s')
    model.update()

    # Objective: Minimize total travel distance
    obj = gp.quicksum(calculate_distance(locations[i][0], locations[i][1], locations[j][0], locations[j][1]) * x[i, j, k] for i in V for j in V for k in K)
    model.setObjective(obj, GRB.MINIMIZE)

    # Constraints
    # Flow conservation
    model.addConstrs((gp.quicksum(x[i, j, k] for j in V) == gp.quicksum(x[j, i, k] for j in V) for i in V for k in K), name='flow_conservation')
    # Customer visit
    model.addConstrs((gp.quicksum(x[i, j, k] for i in V for k in K) >= 1 for j in C), name='visit_customer')
    # Depot start and end
    model.addConstrs((gp.quicksum(x[0, j, k] for j in V) == 1 for k in K), name='depart_depot')
    model.addConstrs((gp.quicksum(x[i, 0, k] for i in V) == 1 for k in K), name='return_depot')
    # Capacity constraint
    model.addConstrs((gp.quicksum(demands[j] * x[i, j, k] for i in V for j in V) <= vehicle_capacity for k in K), name='vehicle_capacity')
    # Time window constraints
    model.addConstrs((s[i, k] + service_times[i] + calculate_distance(locations[i][0], locations[i][1], locations[j][0], locations[j][1]) <= s[j, k] + (1 - x[i, j, k]) * (due_dates[i] + calculate_distance(locations[i][0], locations[i][1], locations[j][0], locations[j][1]) - ready_times[j]) for i in V for j in C for k in K), name='time_window')
    model.addConstrs((ready_times[i] <= s[i, k] for i in V for k in K), name='ready_time')
    model.addConstrs((s[i, k] <= due_dates[i] for i in V for k in K), name='due_date')

    # Time limit
    model.Params.TimeLimit = time_limit

    # Solve
    model.optimize()
    if logger:
        logger.log(model.ObjVal)

    # Extract solution
    solution = {'routes': []}
    for k in K:
        route = []
        path = [0]
        current_node = 0
        while True:
            next_node = None
            for j in V:
                if x[current_node, j, k].x > 0.5:
                    next_node = j
                    break
            if next_node is None or next_node == 0:
                break
            path.append(next_node)
            route.append(customers[next_node - 1]['id'])
            current_node = next_node
        path.append(0)
        if len(route) > 0:
            solution['routes'].append({'customer_ids': route, 'path': path})

    solution['objective_value'] = model.ObjVal
    return solution

def main():
    parser = argparse.ArgumentParser(description='Solve VRPTW instance.')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to instance JSON file.')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to output solution JSON file.')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds.')
    parser.add_argument('--log_path', type=str, help='Path to log intermediate solutions (optional).')
    args = parser.parse_args()

    instance = read_instance(args.instance_path)
    logger = SolutionLogger(args.log_path, sense='minimize') if args.log_path else None

    solution = solve(instance, args.time_limit, logger)
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=4)

if __name__ == '__main__':
    main()
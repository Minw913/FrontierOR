import argparse
import json
import math
from gurobipy import Model, GRB, quicksum
import numpy as np
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as file:
        instance = json.load(file)
    return instance

def solve(instance, time_limit, logger):
    num_customers = instance['num_customers']
    vehicle_capacity = instance['vehicle_capacity']
    delivery_demands = [customer['delivery_demand'] for customer in instance['customers']]
    pickup_demands = [customer['pickup_demand'] for customer in instance['customers']]
    distance_matrix = instance['distance_matrix']

    model = Model("VRPB")
    model.Params.LogToConsole = 0
    model.Params.TimeLimit = time_limit

    # Create variables
    x = model.addVars(2*num_customers+1, 2*num_customers+1, vtype=GRB.BINARY, name='x')
    y = model.addVars(2*num_customers+1, vtype=GRB.CONTINUOUS, name='y')
    z = model.addVars(2*num_customers+1, vtype=GRB.CONTINUOUS, name='z')

    # Objective function
    model.setObjective(quicksum(x[i, j] * distance_matrix[i][j] for i in range(2*num_customers+1) for j in range(2*num_customers+1)), GRB.MINIMIZE)

    # Constraints
    for i in range(1, num_customers+1):
        model.addConstr(quicksum(x[0, j] for j in range(1, num_customers+1)) >= 1)  # At least one vehicle starts from depot
        model.addConstr(quicksum(x[i, 0] for i in range(num_customers+1, 2*num_customers+1)) >= 1)  # At least one vehicle ends at depot
        model.addConstr(quicksum(x[i, j] for j in range(2*num_customers+1)) == quicksum(x[j, i] for j in range(2*num_customers+1)))  # Flow conservation for delivery and pickup nodes
        model.addConstr(quicksum(x[i, j] for i in range(2*num_customers+1)) == 1)  # Each delivery and pickup request is served exactly once
        model.addConstr(quicksum(x[j, i] for j in range(2*num_customers+1)) == 1)  # Each delivery and pickup request is served exactly once

    for i in range(2*num_customers+1):
        for j in range(2*num_customers+1):
            if i != j:
                model.addConstr(y[i] + delivery_demands[i-1] - y[j] <= (1 - x[i, j]) * vehicle_capacity)  # Delivery load constraint
                model.addConstr(z[i] - pickup_demands[i-num_customers-1] - z[j] <= (1 - x[i, j]) * vehicle_capacity)  # Pickup load constraint
                if i <= num_customers and j <= num_customers:  # No direct delivery-delivery edge
                    model.addConstr(x[i, j] == 0)
                if i > num_customers and j > num_customers and i != j:  # No direct pickup-pickup edge except for depot
                    model.addConstr(x[i, j] == 0)

    model.addConstr(y[0] == 0)  # Initial delivery load at depot is 0
    model.addConstr(z[0] == 0)  # Initial pickup load at depot is 0

    for i in range(1, num_customers+1):
        model.addConstr(y[i] >= delivery_demands[i-1])  # Delivery load after visiting a delivery node
        model.addConstr(z[i] == 0)  # Pickup load at a delivery node is 0
        model.addConstr(y[i+num_customers] == 0)  # Delivery load at a pickup node is 0
        model.addConstr(z[i+num_customers] >= pickup_demands[i-1])  # Pickup load after visiting a pickup node

    model.optimize()

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        objective_value = model.ObjVal
        routes = []
        routes_detailed = []

        # Reconstruct routes from the solution
        for vehicle in range(num_customers):
            route = [0]
            route_detailed = [{'id': 0, 'role': 'depot', 'demand': 0}]
            current_node = 0
            while True:
                next_node = None
                for j in range(2*num_customers+1):
                    if x[current_node, j].x > 0.5:
                        next_node = j
                        break
                if next_node is None or next_node == 0:
                    break
                demand = delivery_demands[next_node-1] if next_node <= num_customers else pickup_demands[next_node-num_customers-1]
                role = 'linehaul' if next_node <= num_customers else 'backhaul'
                customer_id = next_node if next_node <= num_customers else next_node - num_customers
                route.append(next_node)
                route_detailed.append({'id': customer_id, 'role': role, 'demand': demand})
                current_node = next_node
            if len(route) > 2:  # Ensure route is not empty
                route.append(0)
                route_detailed.append({'id': 0, 'role': 'depot', 'demand': 0})
                routes.append(route)
                routes_detailed.append(route_detailed)

        solution = {
            'objective_value': objective_value,
            'routes': routes,
            'routes_detailed': routes_detailed
        }

        if logger:
            logger.log(objective_value)

        return solution
    else:
        raise Exception("No solution found")

def main():
    parser = argparse.ArgumentParser(description='Solve VRPB')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the instance JSON file')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to the solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds')
    parser.add_argument('--log_path', type=str, help='Path to the log JSONL file')
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    instance = read_instance(args.instance_path)
    solution = solve(instance, args.time_limit, logger)

    with open(args.solution_path, 'w') as file:
        json.dump(solution, file, indent=2)

if __name__ == "__main__":
    main()
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

def solve(instance, time_limit, logger):
    num_customers = instance['num_customers']
    num_nodes = instance['num_nodes']
    truck_travel_times = instance['truck_travel_times']
    drone_travel_times = instance['drone_travel_times']

    model = gp.Model('TruckDroneRouting')
    model.Params.TimeLimit = time_limit

    # Decision Variables
    customer_types = model.addVars(num_customers, vtype=GRB.BINARY, name='customer_type')
    truck_arcs = model.addVars(num_nodes, num_nodes, vtype=GRB.BINARY, name='truck_arc')
    drone_arcs = model.addVars(num_nodes, num_nodes, vtype=GRB.BINARY, name='drone_arc')
    arrival_times = model.addVars(num_nodes, vtype=GRB.CONTINUOUS, name='arrival_time')

    # Objective: Minimize the maximum arrival time at the depot destination
    model.setObjective(arrival_times[num_nodes-1], GRB.MINIMIZE)

    # Constraints
    # 1. Truck and drone depart from and return to the depot exactly once
    model.addConstr(truck_arcs.sum(0, '*') == 1)
    model.addConstr(truck_arcs.sum('*', num_nodes-1) == 1)
    model.addConstr(drone_arcs.sum(0, '*') == 1)
    model.addConstr(drone_arcs.sum('*', num_nodes-1) == 1)

    # 2. Flow conservation for truck and drone at each customer
    for i in range(1, num_customers+1):
        model.addConstr(truck_arcs.sum(i, '*') == truck_arcs.sum('*', i))
        model.addConstr(drone_arcs.sum(i, '*') == drone_arcs.sum('*', i))

    # 3. Customer service type constraints
    for i in range(1, num_customers+1):
        model.addConstr(truck_arcs.sum(i, '*') == customer_types[i-1])
        model.addConstr(drone_arcs.sum(i, '*') + customer_types[i-1] >= 1)

    # 4. Timing constraints for truck and drone
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                model.addConstr(arrival_times[j] >= arrival_times[i] + truck_travel_times[i][j] - (1 - truck_arcs[i, j]) * 1e6)
                model.addConstr(arrival_times[j] >= arrival_times[i] + drone_travel_times[i][j] - (1 - drone_arcs[i, j]) * 1e6)

    # 5. Drone cannot serve a single customer and return directly to the depot
    for i in range(1, num_customers+1):
        model.addConstr(drone_arcs[0, i] + drone_arcs[i, num_nodes-1] <= 1)

    # 6. Drone arcs between customers are limited by the number of combined customers
    for i in range(1, num_customers+1):
        for j in range(i+1, num_customers+1):
            model.addConstr(drone_arcs[i, j] + drone_arcs[j, i] <= customer_types[i-1] + customer_types[j-1])

    model.optimize()
    if logger:
        logger.log(model.ObjVal)

    # Extract solution
    solution = {}
    solution['objective_value'] = model.ObjVal

    truck_route = [0]
    current_node = 0
    while True:
        next_node = None
        for j in range(num_nodes):
            if truck_arcs[current_node, j].x > 0.5:
                next_node = j
                break
        if next_node is None:
            raise Exception("Truck route is not valid")
        truck_route.append(next_node)
        current_node = next_node
        if current_node == num_nodes - 1:
            break
    solution['truck_route'] = truck_route

    drone_route = [0]
    current_node = 0
    while True:
        next_node = None
        for j in range(num_nodes):
            if drone_arcs[current_node, j].x > 0.5:
                next_node = j
                break
        if next_node is None:
            raise Exception("Drone route is not valid")
        drone_route.append(next_node)
        current_node = next_node
        if current_node == num_nodes - 1:
            break
    solution['drone_route'] = drone_route

    customer_types_dict = {}
    for i in range(num_customers):
        if customer_types[i].x > 0.5:
            customer_types_dict[str(i+1)] = 'combined'
        else:
            if drone_arcs.sum(i+1, '*').getValue() > 0.5:
                customer_types_dict[str(i+1)] = 'drone'
            else:
                customer_types_dict[str(i+1)] = 'truck'
    solution['customer_types'] = customer_types_dict

    arrival_times_dict = {str(i): arrival_times[i].x for i in range(num_nodes)}
    solution['arrival_times'] = arrival_times_dict

    return solution

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str)
    args = parser.parse_args()

    instance = read_instance(args.instance_path)
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    solution = solve(instance, args.time_limit, logger)

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=4)

if __name__ == '__main__':
    main()
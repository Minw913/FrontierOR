import argparse
import json
from gurobipy import Model, GRB, quicksum
import numpy as np
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        return json.load(f)

def solve_instance(instance):
    num_customers = instance['num_customers']
    vehicle_capacity = instance['vehicle_capacity']
    num_vehicles_upper_bound = instance['num_vehicles_upper_bound']
    depot = instance['depot']
    customers = instance['customers']
    distance_matrix = np.array(instance['distance_matrix'])
    travel_time_equals_distance = instance['travel_time_equals_distance']

    model = Model('VRPTW')

    # Variables
    x = model.addVars(num_vehicles_upper_bound, num_customers+2, num_customers+2, vtype=GRB.BINARY, name='x')
    y = model.addVars(num_vehicles_upper_bound, num_customers+1, vtype=GRB.CONTINUOUS, name='y', lb=0)
    z = model.addVars(num_customers+1, vtype=GRB.CONTINUOUS, name='z', lb=0)
    u = model.addVars(num_vehicles_upper_bound, num_customers+2, vtype=GRB.CONTINUOUS, name='u', lb=0)

    # Objective
    model.setObjective(quicksum(distance_matrix[i, j] * x[k, i, j] for k in range(num_vehicles_upper_bound) for i in range(num_customers+2) for j in range(num_customers+2)), GRB.MINIMIZE)

    # Constraints
    for k in range(num_vehicles_upper_bound):
        model.addConstr(quicksum(x[k, 0, j] for j in range(1, num_customers+2)) == 1)
        model.addConstr(quicksum(x[k, i, num_customers+1] for i in range(num_customers+1)) == 1)
        for i in range(1, num_customers+1):
            model.addConstr(quicksum(x[k, i, j] for j in range(num_customers+2)) == quicksum(x[k, j, i] for j in range(num_customers+2)))
            model.addConstr(y[k, i] <= vehicle_capacity * quicksum(x[k, j, i] for j in range(num_customers+2)))
            model.addConstr(y[k, i] <= customers[i-1]['demand'])
        model.addConstr(quicksum(y[k, i] for i in range(1, num_customers+1)) <= vehicle_capacity)

    for i in range(1, num_customers+1):
        model.addConstr(quicksum(y[k, i] for k in range(num_vehicles_upper_bound)) >= customers[i-1]['demand'])
        model.addConstr(z[i] >= depot['time_window'][0] + distance_matrix[0, i])
        for k in range(num_vehicles_upper_bound):
            model.addConstr(u[k, i] >= u[k, 0] + distance_matrix[0, i] - depot['time_window'][0]*(1-x[k, 0, i]))
            for j in range(1, num_customers+2):
                if i != j:
                    model.addConstr(u[k, j] >= u[k, i] + distance_matrix[i, j] - (depot['time_window'][1] + distance_matrix[i, j])*(1-x[k, i, j]))

    for k in range(num_vehicles_upper_bound):
        for i in range(num_customers+2):
            for j in range(num_customers+2):
                if i != j:
                    model.addConstr(u[k, i] + distance_matrix[i, j] <= u[k, j] + (depot['time_window'][1] + distance_matrix[i, j])*(1-x[k, i, j]))

    model.update()
    model.optimize()

    return model

def extract_solution(instance, model):
    num_customers = instance['num_customers']
    num_vehicles_upper_bound = instance['num_vehicles_upper_bound']
    customers = instance['customers']
    distance_matrix = np.array(instance['distance_matrix'])

    x = model.getAttr('x', model.getVars()[:num_vehicles_upper_bound*(num_customers+2)*(num_customers+2)])
    x = np.array(x).reshape((num_vehicles_upper_bound, num_customers+2, num_customers+2))
    y = model.getAttr('x', model.getVars()[num_vehicles_upper_bound*(num_customers+2)*(num_customers+2):num_vehicles_upper_bound*(num_customers+2)*(num_customers+2) + num_vehicles_upper_bound*(num_customers+1)])

    solution = {'routes': [], 'deliveries': {}}
    num_vehicles_used = 0

    for k in range(num_vehicles_upper_bound):
        route = [0]
        deliveries = {}
        while True:
            next_node = np.argmax(x[k, route[-1], :])
            if next_node == num_customers+1:
                route.append(num_customers+1)
                break
            else:
                route.append(next_node)
        if len(route) > 2:
            num_vehicles_used += 1
            solution['routes'].append({'vehicle': k, 'route': route, 'deliveries': {}})
            for i in range(1, len(route)-1):
                customer_id = route[i]
                deliveries[customer_id] = y[k*num_customers+customer_id-1].x
            solution['routes'][-1]['deliveries'] = deliveries

    for customer in customers:
        customer_id = customer['id']
        total_delivered = sum([route['deliveries'].get(customer_id, 0) for route in solution['routes']])
        solution['deliveries'][customer_id] = {'demand': customer['demand'], 'total_delivered': total_delivered}

    objective_value = model.getObjective().getValue()
    solution['objective_value'] = objective_value
    solution['num_vehicles_used'] = num_vehicles_used

    return solution

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str)
    args = parser.parse_args()

    instance = read_instance(args.instance_path)
    model = Model('VRPTW')
    model.setParam('TimeLimit', args.time_limit)

    logger = SolutionLogger(args.log_path, sense='minimize') if args.log_path else None

    num_customers = instance['num_customers']
    vehicle_capacity = instance['vehicle_capacity']
    num_vehicles_upper_bound = instance['num_vehicles_upper_bound']
    depot = instance['depot']
    customers = instance['customers']
    distance_matrix = np.array(instance['distance_matrix'])
    travel_time_equals_distance = instance['travel_time_equals_distance']

    model = Model('VRPTW')

    x = model.addVars(num_vehicles_upper_bound, num_customers+2, num_customers+2, vtype=GRB.BINARY, name='x')
    y = model.addVars(num_vehicles_upper_bound, num_customers+1, vtype=GRB.CONTINUOUS, name='y', lb=0)
    z = model.addVars(num_customers+1, vtype=GRB.CONTINUOUS, name='z', lb=0)
    u = model.addVars(num_vehicles_upper_bound, num_customers+2, vtype=GRB.CONTINUOUS, name='u', lb=0)

    model.setObjective(quicksum(distance_matrix[i, j] * x[k, i, j] for k in range(num_vehicles_upper_bound) for i in range(num_customers+2) for j in range(num_customers+2)), GRB.MINIMIZE)

    for k in range(num_vehicles_upper_bound):
        model.addConstr(quicksum(x[k, 0, j] for j in range(1, num_customers+2)) == 1)
        model.addConstr(quicksum(x[k, i, num_customers+1] for i in range(num_customers+1)) == 1)
        for i in range(1, num_customers+1):
            model.addConstr(quicksum(x[k, i, j] for j in range(num_customers+2)) == quicksum(x[k, j, i] for j in range(num_customers+2)))
            model.addConstr(y[k, i] <= vehicle_capacity * quicksum(x[k, j, i] for j in range(num_customers+2)))
            model.addConstr(y[k, i] <= customers[i-1]['demand'])
        model.addConstr(quicksum(y[k, i] for i in range(1, num_customers+1)) <= vehicle_capacity)

    for i in range(1, num_customers+1):
        model.addConstr(quicksum(y[k, i] for k in range(num_vehicles_upper_bound)) >= customers[i-1]['demand'])
        model.addConstr(z[i] >= depot['time_window'][0] + distance_matrix[0, i])
        for k in range(num_vehicles_upper_bound):
            model.addConstr(u[k, i] >= u[k, 0] + distance_matrix[0, i] - depot['time_window'][0]*(1-x[k, 0, i]))
            for j in range(1, num_customers+2):
                if i != j:
                    model.addConstr(u[k, j] >= u[k, i] + distance_matrix[i, j] - (depot['time_window'][1] + distance_matrix[i, j])*(1-x[k, i, j]))

    for k in range(num_vehicles_upper_bound):
        for i in range(num_customers+2):
            for j in range(num_customers+2):
                if i != j:
                    model.addConstr(u[k, i] + distance_matrix[i, j] <= u[k, j] + (depot['time_window'][1] + distance_matrix[i, j])*(1-x[k, i, j]))

    def log_callback(model, where):
        if where == GRB.Callback.MIPSOL:
            if logger:
                logger.log(model.getAttr('ObjVal'))

    model.optimize(log_callback)

    solution = extract_solution(instance, model)

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=4)

if __name__ == '__main__':
    main()
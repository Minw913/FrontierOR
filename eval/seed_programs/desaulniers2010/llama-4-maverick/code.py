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

    model = Model('VRPTW')
    model.setParam("Threads", 1)
    model.modelSense = GRB.MINIMIZE

    # Variables
    x = model.addVars(num_vehicles_upper_bound, num_customers+2, num_customers+2, vtype=GRB.BINARY, name='x')
    y = model.addVars(num_vehicles_upper_bound, num_customers+1, vtype=GRB.CONTINUOUS, name='y', lb=0)
    u = model.addVars(num_vehicles_upper_bound, num_customers+2, vtype=GRB.CONTINUOUS, name='u', lb=0)

    # Objective
    model.setObjective(quicksum(distance_matrix[i, j] * x[k, i, j] for k in range(num_vehicles_upper_bound) for i in range(num_customers+2) for j in range(num_customers+2)))

    # Constraints
    for k in range(num_vehicles_upper_bound):
        model.addConstr(quicksum(x[k, 0, j] for j in range(1, num_customers+2)) == quicksum(x[k, j, num_customers+1] for j in range(num_customers+1)))
        for i in range(1, num_customers+1):
            model.addConstr(quicksum(x[k, i, j] for j in range(num_customers+2)) == quicksum(x[k, j, i] for j in range(num_customers+2)))
            model.addConstr(y[k, i] <= vehicle_capacity * quicksum(x[k, j, i] for j in range(num_customers+2)))
            model.addConstr(y[k, i] <= customers[i-1]['demand'])
        model.addConstr(quicksum(y[k, i] for i in range(1, num_customers+1)) <= vehicle_capacity)

    for i in range(1, num_customers+1):
        model.addConstr(quicksum(y[k, i] for k in range(num_vehicles_upper_bound)) >= customers[i-1]['demand'])

    for k in range(num_vehicles_upper_bound):
        for i in range(num_customers+2):
            for j in range(num_customers+2):
                if i != j:
                    model.addConstr(u[k, j] >= u[k, i] + (distance_matrix[i, j] + (customers[i-1]['service_time'] if i > 0 and i <= num_customers else 0)) * x[k, i, j] - 1e6*(1-x[k, i, j]))
        for i in range(1, num_customers+1):
            model.addConstr(u[k, i] >= customers[i-1]['time_window'][0])
            model.addConstr(u[k, i] <= customers[i-1]['time_window'][1])
        model.addConstr(u[k, 0] == depot['time_window'][0])
        model.addConstr(u[k, num_customers+1] <= depot['time_window'][1])

    model.update()
    return model

def extract_solution(instance, model):
    num_customers = instance['num_customers']
    num_vehicles_upper_bound = instance['num_vehicles_upper_bound']
    customers = instance['customers']

    vars = model.getVars()
    x = [var.x for var in vars if var.varName.startswith('x')]
    x = np.array(x).reshape((num_vehicles_upper_bound, num_customers+2, num_customers+2))
    y = [var.x for var in vars if var.varName.startswith('y')]
    y = np.array(y).reshape((num_vehicles_upper_bound, num_customers+1))
    u = [var.x for var in vars if var.varName.startswith('u')]
    u = np.array(u).reshape((num_vehicles_upper_bound, num_customers+2))

    solution = {'routes': [], 'deliveries': {}}
    num_vehicles_used = 0

    for k in range(num_vehicles_upper_bound):
        route = []
        deliveries = {}
        node = 0
        visited = False
        while True:
            next_node = np.argmax(x[k, node, :])
            route.append(node)
            if next_node == num_customers + 1:
                route.append(next_node)
                break
            else:
                if next_node != 0:
                    visited = True
                    deliveries[str(next_node)] = y[k, next_node]
                node = next_node
        if visited:
            num_vehicles_used += 1
            solution['routes'].append({'vehicle': k, 'route': route, 'deliveries': deliveries})

    for customer in customers:
        customer_id = str(customer['id'])
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

    model = solve_instance(instance)
    model.setParam('TimeLimit', args.time_limit)

    logger = SolutionLogger(args.log_path, sense='minimize') if args.log_path else None

    def log_callback(model, where):
        if where == GRB.Callback.MIPSOL:
            if logger:
                logger.log(model.getAttr('ObjVal'))

    model.optimize(log_callback)

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        solution = extract_solution(instance, model)
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=4)
    else:
        print("No solution found")

if __name__ == '__main__':
    main()
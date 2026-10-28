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
    num_stacks = instance['parameters']['num_stacks_s']
    stack_height = instance['parameters']['stack_height_l']
    num_requests = instance['parameters']['num_requests_n']
    num_pickup_vertices = instance['parameters']['num_pickup_vertices']
    num_delivery_vertices = instance['parameters']['num_delivery_vertices']
    pickup_distance_matrix = np.array(instance['pickup_region']['distance_matrix'])
    delivery_distance_matrix = np.array(instance['delivery_region']['distance_matrix'])
    pickup_depot_index = instance['pickup_region']['depot_index']
    delivery_depot_index = instance['delivery_region']['depot_index']
    requests = instance['requests']

    model = gp.Model()
    model.setParam("Threads", 1)
    model.Params.TimeLimit = time_limit
    model.Params.LogToConsole = 0

    # Variables for pickup and delivery tours
    pickup_x = model.addVars(num_pickup_vertices, num_pickup_vertices, vtype=GRB.BINARY)
    delivery_x = model.addVars(num_delivery_vertices, num_delivery_vertices, vtype=GRB.BINARY)

    # Stack assignment variables
    z = model.addVars(num_requests, num_stacks, vtype=GRB.BINARY)

    # Objective function
    pickup_cost = gp.quicksum(pickup_distance_matrix[i, j] * pickup_x[i, j] for i in range(num_pickup_vertices) for j in range(num_pickup_vertices))
    delivery_cost = gp.quicksum(delivery_distance_matrix[i, j] * delivery_x[i, j] for i in range(num_delivery_vertices) for j in range(num_delivery_vertices))
    model.setObjective(pickup_cost + delivery_cost, GRB.MINIMIZE)

    # Constraints for pickup tour
    for i in range(num_pickup_vertices):
        model.addConstr(gp.quicksum(pickup_x[i, j] for j in range(num_pickup_vertices)) == 1)
        model.addConstr(gp.quicksum(pickup_x[j, i] for j in range(num_pickup_vertices)) == 1)
    model.addConstrs((gp.quicksum(pickup_x[i, j] for i in range(num_pickup_vertices) if i != j) == 1 for j in range(1, num_pickup_vertices)), name='pickup_out')
    model.addConstrs((gp.quicksum(pickup_x[i, j] for j in range(num_pickup_vertices) if i != j) == 1 for i in range(1, num_pickup_vertices)), name='pickup_in')

    # Constraints for delivery tour
    for i in range(num_delivery_vertices):
        model.addConstr(gp.quicksum(delivery_x[i, j] for j in range(num_delivery_vertices)) == 1)
        model.addConstr(gp.quicksum(delivery_x[j, i] for j in range(num_delivery_vertices)) == 1)
    model.addConstrs((gp.quicksum(delivery_x[i, j] for i in range(num_delivery_vertices) if i != j) == 1 for j in range(1, num_delivery_vertices)), name='delivery_out')
    model.addConstrs((gp.quicksum(delivery_x[i, j] for j in range(num_delivery_vertices) if i != j) == 1 for i in range(1, num_delivery_vertices)), name='delivery_in')

    # Subtour elimination for pickup tour
    pickup_u = model.addVars(num_pickup_vertices, vtype=GRB.INTEGER)
    for i in range(1, num_pickup_vertices):
        for j in range(1, num_pickup_vertices):
            if i != j:
                model.addConstr(pickup_u[i] - pickup_u[j] + num_pickup_vertices * pickup_x[i, j] <= num_pickup_vertices - 1)

    # Subtour elimination for delivery tour
    delivery_u = model.addVars(num_delivery_vertices, vtype=GRB.INTEGER)
    for i in range(1, num_delivery_vertices):
        for j in range(1, num_delivery_vertices):
            if i != j:
                model.addConstr(delivery_u[i] - delivery_u[j] + num_delivery_vertices * delivery_x[i, j] <= num_delivery_vertices - 1)

    # Stack assignment constraints
    for k in range(num_requests):
        model.addConstr(gp.quicksum(z[k, s] for s in range(num_stacks)) == 1)
    for s in range(num_stacks):
        model.addConstr(gp.quicksum(z[k, s] for k in range(num_requests)) <= stack_height)

    # LIFO constraints
    pickup_sequence = model.addVars(num_requests, vtype=GRB.INTEGER)
    delivery_sequence = model.addVars(num_requests, vtype=GRB.INTEGER)
    for k in range(num_requests):
        pickup_vertex = requests[k]['pickup_vertex']
        model.addConstr(pickup_sequence[k] == gp.quicksum(i * pickup_x[pickup_depot_index, i] for i in range(num_pickup_vertices) if i != pickup_depot_index) + gp.quicksum(i * pickup_x[i, pickup_vertex] for i in range(num_pickup_vertices)))
        delivery_vertex = requests[k]['delivery_vertex']
        model.addConstr(delivery_sequence[k] == gp.quicksum(i * delivery_x[delivery_depot_index, i] for i in range(num_delivery_vertices) if i != delivery_depot_index) + gp.quicksum(i * delivery_x[i, delivery_vertex] for i in range(num_delivery_vertices)))

    for s in range(num_stacks):
        for k1 in range(num_requests):
            for k2 in range(num_requests):
                if k1 != k2:
                    model.addConstr((z[k1, s] + z[k2, s] + (pickup_sequence[k1] >= pickup_sequence[k2]) + (delivery_sequence[k2] >= delivery_sequence[k1]) <= 3))

    model.optimize()
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        if model.SolCount > 0:
            pickup_tour = []
            start = pickup_depot_index
            pickup_tour.append(start)
            for _ in range(num_pickup_vertices - 1):
                for j in range(num_pickup_vertices):
                    if pickup_x[start, j].x > 0.5:
                        pickup_tour.append(j)
                        start = j
                        break
            pickup_tour.append(pickup_depot_index)

            delivery_tour = []
            start = delivery_depot_index
            delivery_tour.append(start)
            for _ in range(num_delivery_vertices - 1):
                for j in range(num_delivery_vertices):
                    if delivery_x[start, j].x > 0.5:
                        delivery_tour.append(j)
                        start = j
                        break
            delivery_tour.append(delivery_depot_index)

            objective_value = model.ObjVal

            if logger:
                logger.log(objective_value)

            return {
                'objective_value': objective_value,
                'pickup_tour': pickup_tour,
                'delivery_tour': delivery_tour
            }
        else:
            raise Exception("No solution found")
    else:
        raise Exception(f"Gurobi model status: {model.status}")

def write_solution(solution_path, solution):
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=4)

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
    write_solution(args.solution_path, solution)

if __name__ == "__main__":
    main()
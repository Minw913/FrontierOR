import argparse
import json
import numpy as np
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    return instance

def compute_travel_time(instance, departure_time, i, j):
    distance = instance['arcs']['distances'][i][j]
    max_speed = instance['arcs']['max_speeds_u_ij'][i][j]
    delta = instance['arcs']['delta_ijh'][i][j]
    b_h = instance['parameters']['b_h']
    period_boundaries = instance['parameters']['period_boundaries']
    H = instance['parameters']['H']

    time = departure_time
    remaining_distance = distance
    for h in range(H):
        if time >= period_boundaries[h+1]:
            continue
        speed = max_speed * delta[h] * b_h[h]
        travel_time_in_period = min((period_boundaries[h+1] - time), remaining_distance / speed)
        remaining_distance -= travel_time_in_period * speed
        time += travel_time_in_period
        if remaining_distance <= 0:
            break
    return time - departure_time

def build_model(instance):
    n_customers = instance['parameters']['n_customers']
    total_vertices = instance['parameters']['total_vertices']
    depot_index = instance['depot']['index']

    model = Model('TDTSP')
    model.modelSense = GRB.MINIMIZE

    x = model.addVars(total_vertices, total_vertices, vtype=GRB.BINARY, name='x')
    model.addConstrs((quicksum(x[i, j] for j in range(total_vertices)) == 1 for i in range(total_vertices)), name='outgoing')
    model.addConstrs((quicksum(x[j, i] for j in range(total_vertices)) == 1 for i in range(total_vertices)), name='incoming')

    # Subtour elimination
    u = model.addVars(total_vertices, vtype=GRB.INTEGER, name='u')
    model.addConstrs((u[i] - u[j] + total_vertices * x[i, j] <= total_vertices - 1 for i in range(1, total_vertices) for j in range(1, total_vertices) if i != j), name='subtour')

    # Objective
    travel_time = model.addVars(total_vertices, total_vertices, name='travel_time')
    for i in range(total_vertices):
        for j in range(total_vertices):
            if i != j:
                model.addConstr(travel_time[i, j] == compute_travel_time(instance, 0, i, j), name=f'travel_time_{i}_{j}')

    departure_time = model.addVars(total_vertices, name='departure_time')
    model.addConstr(departure_time[depot_index] == 0)
    for i in range(total_vertices):
        for j in range(total_vertices):
            if i != j:
                model.addConstr(departure_time[j] >= departure_time[i] + travel_time[i, j] * x[i, j] - GRB.BIGM * (1 - x[i, j]))

    model.setObjective(departure_time[depot_index] + quicksum(travel_time[i, j] * x[i, j] for i in range(total_vertices) for j in range(total_vertices) if i != j))

    return model, x

def solve_instance(instance_path, time_limit, log_path=None):
    instance = read_instance(instance_path)
    model, x = build_model(instance)
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None

    model.Params.TimeLimit = time_limit
    model.Params.LogToConsole = 0  # Optional: disable Gurobi console output
    model.optimize()

    if model.SolCount > 0:
        solution = {
            'objective_value': model.ObjVal,
            'tour': get_tour(x, instance['depot']['index'])
        }
        if logger:
            logger.log(model.ObjVal)
    else:
        solution = None

    return solution

def get_tour(x, depot_index):
    total_vertices = x.shape[0]
    tour = [depot_index]
    current_vertex = depot_index
    for _ in range(total_vertices - 1):
        for j in range(total_vertices):
            if x[current_vertex, j].X > 0.5:
                tour.append(j)
                current_vertex = j
                break
    tour.append(depot_index)
    return tour

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True, type=str)
    parser.add_argument('--solution_path', required=True, type=str)
    parser.add_argument('--time_limit', required=True, type=int)
    parser.add_argument('--log_path', type=str)
    args = parser.parse_args()

    solution = solve_instance(args.instance_path, args.time_limit, args.log_path)
    if solution:
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f)
    else:
        print("No solution found.")

if __name__ == '__main__':
    main()
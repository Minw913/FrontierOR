import argparse
import json
import numpy as np
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger
import time
from itertools import combinations
def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    return instance

def calculate_distance_matrix(nodes):
    n_nodes = len(nodes)
    distance_matrix = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            distance_matrix[i, j] = np.sqrt((nodes[i]['x'] - nodes[j]['x']) ** 2 + (nodes[i]['y'] - nodes[j]['y']) ** 2)
    return distance_matrix

def calculate_travel_time(distance_matrix, nodes):
    n_nodes = len(nodes)
    travel_time_matrix = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            travel_time_matrix[i, j] = distance_matrix[i, j] + nodes[i]['waiting_time'] + nodes[j]['waiting_time']
    return travel_time_matrix

def calculate_fare(distance_matrix, cost_scalar, alpha):
    n_nodes = distance_matrix.shape[0]
    fare_matrix = cost_scalar * distance_matrix
    discounted_fare_matrix = alpha * fare_matrix
    return fare_matrix, discounted_fare_matrix

def calculate_route_travel_time(travel_time_matrix, i, j, k, l):
    return travel_time_matrix[i, k] + travel_time_matrix[k, l] + travel_time_matrix[l, j]

def calculate_route_fare(fare_matrix, discounted_fare_matrix, i, j, k, l, alpha):
    if k == l:
        return fare_matrix[i, k] + fare_matrix[k, j]
    else:
        return fare_matrix[i, k] + discounted_fare_matrix[k, l] + fare_matrix[l, j]

def calculate_utility(attractiveness, travel_time, fare, gamma, beta, lambda_):
    denominator = gamma * (travel_time ** beta) + (1 - gamma) * (fare ** lambda_)
    if denominator > 0:
        return attractiveness / denominator
    else:
        return 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, required=False)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    instance = read_instance(args.instance_path)
    n_nodes = instance['n_nodes']
    p = instance['p']
    nodes = instance['nodes']
    demand_matrix = np.array(instance['demand_matrix'])
    distance_matrix = np.array(instance['distance_matrix'])
    competitor_hub_locations = instance['competitor_hub_locations']
    parameters = instance['parameters']
    alpha = parameters['alpha']
    gamma = parameters['gamma']
    beta = parameters['beta']
    lambda_ = parameters['lambda']
    cost_scalar = parameters['cost_scalar']
    attractiveness_2hub = parameters['attractiveness_2hub']
    attractiveness_1hub = parameters['attractiveness_1hub']

    travel_time_matrix = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            travel_time_matrix[i, j] = distance_matrix[i, j] + nodes[i]['waiting_time'] + nodes[j]['waiting_time']

    fare_matrix = cost_scalar * distance_matrix
    discounted_fare_matrix = alpha * fare_matrix

    competitor_utilities = {}
    for i in range(n_nodes):
        for j in range(n_nodes):
            if demand_matrix[i, j] > 0:
                total_utility = 0
                for k, l in combinations(competitor_hub_locations, 2):
                    travel_time1 = travel_time_matrix[i, k] + travel_time_matrix[k, l] + travel_time_matrix[l, j]
                    fare1 = fare_matrix[i, k] + discounted_fare_matrix[k, l] + fare_matrix[l, j]
                    utility1 = calculate_utility(attractiveness_2hub, travel_time1, fare1, gamma, beta, lambda_)

                    travel_time2 = travel_time_matrix[i, l] + travel_time_matrix[l, k] + travel_time_matrix[k, j]
                    fare2 = fare_matrix[i, l] + discounted_fare_matrix[l, k] + fare_matrix[k, j]
                    utility2 = calculate_utility(attractiveness_2hub, travel_time2, fare2, gamma, beta, lambda_)
                    total_utility += max(utility1, utility2)

                for k in competitor_hub_locations:
                    travel_time = travel_time_matrix[i, k] + travel_time_matrix[k, k] + travel_time_matrix[k, j]
                    fare = fare_matrix[i, k] + fare_matrix[k, j]
                    utility = calculate_utility(attractiveness_1hub, travel_time, fare, gamma, beta, lambda_)
                    total_utility += utility
                competitor_utilities[(i, j)] = total_utility

    model = Model('Hub Location Problem')
    model.setParam("Seed", 0)
    model.setParam("MIPGap", 1e-4)
    model.setParam("NumericFocus", 0)
    model.setParam("Threads", 1)
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('LogToConsole', 0)

    hub_vars = model.addVars(n_nodes, vtype=GRB.BINARY, name='hub')
    model.addConstr(quicksum(hub_vars[i] for i in range(n_nodes)) == p, name='p_hubs')

    passenger_capture = model.addVars([(i, j) for i in range(n_nodes) for j in range(n_nodes) if demand_matrix[i, j] > 0], vtype=GRB.CONTINUOUS, name='passenger_capture')

    for i, j in [(i, j) for i in range(n_nodes) for j in range(n_nodes) if demand_matrix[i, j] > 0]:
        total_entrant_utility = 0
        for k in range(n_nodes):
            for l in range(n_nodes):
                if k == l:
                    travel_time = travel_time_matrix[i, k] + travel_time_matrix[k, l] + travel_time_matrix[l, j]
                    fare = fare_matrix[i, k] + fare_matrix[k, j]
                    utility = calculate_utility(attractiveness_1hub, travel_time, fare, gamma, beta, lambda_)
                    total_entrant_utility += utility * hub_vars[k]
                elif k < l:
                    travel_time1 = travel_time_matrix[i, k] + travel_time_matrix[k, l] + travel_time_matrix[l, j]
                    fare1 = fare_matrix[i, k] + discounted_fare_matrix[k, l] + fare_matrix[l, j]
                    utility1 = calculate_utility(attractiveness_2hub, travel_time1, fare1, gamma, beta, lambda_)

                    travel_time2 = travel_time_matrix[i, l] + travel_time_matrix[l, k] + travel_time_matrix[k, j]
                    fare2 = fare_matrix[i, l] + discounted_fare_matrix[l, k] + fare_matrix[k, j]
                    utility2 = calculate_utility(attractiveness_2hub, travel_time2, fare2, gamma, beta, lambda_)
                    total_entrant_utility += max(utility1, utility2) * hub_vars[k] * hub_vars[l]

        # Linearize the nonlinear constraint
        model.addConstr(passenger_capture[i, j] <= demand_matrix[i, j], name=f'passenger_capture_upper_{i}_{j}')
        aux_var = model.addVar(name=f'aux_{i}_{j}')
        model.addConstr(aux_var == total_entrant_utility, name=f'aux_constr_{i}_{j}')
        model.addConstr(passenger_capture[i, j] * (competitor_utilities[(i, j)] + aux_var) == demand_matrix[i, j] * aux_var, name=f'passenger_capture_{i}_{j}')

    model.setObjective(quicksum(passenger_capture[i, j] for i, j in [(i, j) for i in range(n_nodes) for j in range(n_nodes) if demand_matrix[i, j] > 0]), GRB.MAXIMIZE)

    model.optimize()

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        objective_value = model.getObjective().getValue()
        hub_locations = [i for i in range(n_nodes) if hub_vars[i].X > 0.5]
        solution = {'objective_value': objective_value, 'hub_locations': hub_locations}
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f)
        if logger:
            logger.log(objective_value)

if __name__ == '__main__':
    main()
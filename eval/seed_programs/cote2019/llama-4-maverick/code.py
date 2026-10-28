import argparse
import json
from gurobipy import Model, GRB, quicksum
import numpy as np
from solution_logger import SolutionLogger
import time
import itertools

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    return instance

def compute_expected_area_and_weight(customer):
    expected_area = 0
    expected_weight = 0
    for item in customer['items']:
        for realization in item['realizations']:
            area = realization['height'] * realization['width']
            weight = realization['weight']
            expected_area += area * realization['probability']
            expected_weight += weight * realization['probability']
    return expected_area, expected_weight

def generate_scenarios(instance):
    scenarios = {}
    for customer in instance['customers']:
        if customer['is_stochastic']:
            customer_scenarios = list(itertools.product(*(range(item['num_realizations']) for item in customer['items'])))
            scenarios[customer['id']] = customer_scenarios
    return scenarios

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, required=False)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    instance = read_instance(args.instance_path)

    n_customers = instance['parameters']['n_customers']
    K = instance['vehicle']['K']
    H = instance['vehicle']['H']
    W = instance['vehicle']['W']
    Q = instance['vehicle']['Q']
    distance_matrix = instance['distance_matrix']
    recourse_cost_cf = instance['parameters']['recourse_cost_cf']

    customers = instance['customers']
    depot_id = instance['depot']['id']

    model = Model("VRP")
    model.setParam("Threads", 1)
    model.Params.LogToConsole = 0
    model.Params.TimeLimit = args.time_limit

    # Variables
    x = model.addVars([(i, j) for i in range(n_customers + 1) for j in range(n_customers + 1)], vtype=GRB.BINARY, name='x')
    y = model.addVars([i for i in range(1, n_customers + 1)], vtype=GRB.INTEGER, name='y')

    # Objective
    travel_cost = quicksum(distance_matrix[i][j] * x[i, j] for i in range(n_customers + 1) for j in range(n_customers + 1))
    model.setObjective(travel_cost, GRB.MINIMIZE)

    # Constraints
    model.addConstrs((quicksum(x[i, j] for j in range(n_customers + 1)) == 1 for i in range(1, n_customers + 1)), name='customer_visited_once')
    model.addConstrs((quicksum(x[i, j] for i in range(n_customers + 1)) == 1 for j in range(1, n_customers + 1)), name='customer_visited_once_in')
    model.addConstr(quicksum(x[0, j] for j in range(1, n_customers + 1)) == K, name='K_vehicles_out')
    model.addConstr(quicksum(x[i, 0] for i in range(1, n_customers + 1)) == K, name='K_vehicles_in')

    model.addConstrs((y[i] - y[j] + (n_customers + 1) * x[i, j] <= n_customers for i in range(1, n_customers + 1) for j in range(1, n_customers + 1) if i != j), name='mtz')

    # Capacity and subtour elimination constraints
    for S in range(1, 2**n_customers):
        subset = [i + 1 for i in range(n_customers) if (S & (1 << i))]
        if len(subset) > 0:
            total_expected_area = sum(compute_expected_area_and_weight(customers[i - 1])[0] for i in subset)
            total_expected_weight = sum(compute_expected_area_and_weight(customers[i - 1])[1] for i in subset)
            capacity_constraint_area = np.ceil(total_expected_area / (H * W))
            capacity_constraint_weight = np.ceil(total_expected_weight / Q)
            model.addConstr(quicksum(x[i, j] for i in subset for j in subset) <= len(subset) - max(capacity_constraint_area, capacity_constraint_weight), name=f'subtour_elimination_{S}')

    model.optimize()

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        objective_value = model.getObjective().getValue()
        if logger:
            logger.log(objective_value)

        routes = []
        for k in range(K):
            route = []
            current_node = 0
            next_node = None
            while True:
                for j in range(n_customers + 1):
                    if x[current_node, j].x > 0.5:
                        next_node = j
                        break
                if next_node == 0:
                    break
                route.append(next_node)
                current_node = next_node
            routes.append(route)

        solution = {
            "objective_value": objective_value,
            "routes": routes
        }

        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=4)

if __name__ == "__main__":
    main()
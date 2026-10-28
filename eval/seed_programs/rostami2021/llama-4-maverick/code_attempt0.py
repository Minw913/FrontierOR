import argparse
import json
import math
import numpy as np
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as file:
        instance = json.load(file)
    return instance

def write_solution(solution_path, solution):
    with open(solution_path, 'w') as file:
        json.dump(solution, file, indent=4)

def compute_distances(node_coordinates):
    n = len(node_coordinates)
    distances = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                dist = math.sqrt((node_coordinates[i][0] - node_coordinates[j][0])**2 + 
                                 (node_coordinates[i][1] - node_coordinates[j][1])**2)
                distances[i][j] = dist
    return distances

def solve_hub_location(instance, time_limit, log_path=None):
    n = instance['n']
    num_scenarios = instance['num_scenarios']
    chi = instance['cost_parameters']['chi']
    alpha = instance['cost_parameters']['alpha']
    delta = instance['cost_parameters']['delta']
    p_hubs = instance['p_hubs']
    fixed_costs = instance['fixed_costs']
    hub_capacities = instance.get('hub_capacities', None)
    scenarios = instance['scenarios']
    distances = instance.get('distances', None)
    if distances is None:
        distances = compute_distances(instance['node_coordinates'])

    logger = SolutionLogger(log_path, sense="minimize") if log_path else None

    model = Model("Hub Location Problem")
    model.Params.TimeLimit = time_limit
    model.Params.LogToConsole = 0

    # Decision Variables
    z = model.addVars(n, vtype=GRB.BINARY, name="z")  # Hub selection
    x = {}  # Allocation variables per scenario
    for s in range(num_scenarios):
        x[s] = model.addVars(n, n, vtype=GRB.BINARY, name=f"x_{s}")

    # Objective Function
    obj = quicksum(fixed_costs[i] * z[i] for i in range(n))
    for s, scenario in enumerate(scenarios):
        probability = scenario['probability']
        demands = scenario['demands']
        outgoing_flows = [sum(demands[i][j] for j in range(n)) for i in range(n)]
        incoming_flows = [sum(demands[j][i] for j in range(n)) for i in range(n)]

        # Collection and Distribution Cost
        collection_distribution_cost = quicksum(
            (outgoing_flows[i] * chi + incoming_flows[i] * delta) * distances[i][k] * x[s][i, k]
            for i in range(n) for k in range(n) if i != k
        )
        collection_distribution_cost += quicksum(
            (outgoing_flows[k] * chi + incoming_flows[k] * delta) * 0 * z[k]
            for k in range(n)
        )

        # Inter-hub Transfer Cost
        inter_hub_transfer_cost = quicksum(
            demands[i][j] * alpha * (quicksum(distances[k][l] * x[s][i, k] * x[s][j, l] for k in range(n) for l in range(n)))
            for i in range(n) for j in range(n)
        )

        obj += probability * (collection_distribution_cost + inter_hub_transfer_cost)

    model.setObjective(obj, GRB.MINIMIZE)

    # Constraints
    for s in range(num_scenarios):
        for i in range(n):
            model.addConstr(quicksum(x[s][i, k] for k in range(n) if k != i) + z[i] == 1, name=f"assign_{s}_{i}")

    for s in range(num_scenarios):
        for i in range(n):
            for k in range(n):
                if i != k:
                    model.addConstr(x[s][i, k] <= z[k], name=f"hub_open_{s}_{i}_{k}")

    if p_hubs is not None:
        model.addConstr(quicksum(z[i] for i in range(n)) == p_hubs, name="p_hubs")

    if hub_capacities is not None:
        for s, scenario in enumerate(scenarios):
            demands = scenario['demands']
            outgoing_flows = [sum(demands[i][j] for j in range(n)) for i in range(n)]
            for k in range(n):
                model.addConstr(
                    quicksum(outgoing_flows[i] * x[s][i, k] for i in range(n) if i != k) + outgoing_flows[k] * z[k]
                    <= hub_capacities[k] * z[k], name=f"capacity_{s}_{k}"
                )

    model.addConstr(quicksum(z[i] for i in range(n)) >= 1, name="at_least_one_hub")

    model.optimize()

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        objective_value = model.ObjVal
        hubs = [i for i in range(n) if z[i].X > 0.5]
        allocations = {}
        x_values = {}
        for s in range(num_scenarios):
            allocations[str(s)] = {i: k for i in range(n) for k in range(n) if (x[s][i, k].X > 0.5 or (i == k and z[k].X > 0.5))}
            x_values[str(s)] = {(i, k): x[s][i, k].X for i in range(n) for k in range(n) if i != k}

        z_values = [z[i].X for i in range(n)]

        solution = {
            "objective_value": objective_value,
            "hubs": hubs,
            "allocations": allocations,
            "z_values": z_values,
            "x_values": x_values
        }

        if logger:
            logger.log(objective_value)

        return solution
    else:
        raise Exception("No solution found")

def main():
    parser = argparse.ArgumentParser(description='Solve Hub Location Problem')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the instance JSON file')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to write the solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds')
    parser.add_argument('--log_path', type=str, help='Path to log intermediate solutions (optional)')
    args = parser.parse_args()

    instance = read_instance(args.instance_path)
    solution = solve_hub_location(instance, args.time_limit, args.log_path)
    write_solution(args.solution_path, solution)

if __name__ == "__main__":
    main()
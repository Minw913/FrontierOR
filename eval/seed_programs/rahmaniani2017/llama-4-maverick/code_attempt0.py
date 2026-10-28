import argparse
import json
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from solution_logger import SolutionLogger

def calculate_distances(nodes):
    num_nodes = len(nodes)
    distances = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            dist = np.sqrt((nodes[i]['x'] - nodes[j]['x'])**2 + (nodes[i]['y'] - nodes[j]['y'])**2)
            distances[i, j] = distances[j, i] = dist
    return distances

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, required=False)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    num_nodes = instance['num_nodes']
    num_commodities = instance['num_commodities']
    tau = instance['parameters']['tau']
    chi = instance['parameters']['chi']
    delta = instance['parameters']['delta']
    TC = instance['parameters']['TC']
    nodes = instance['nodes']
    distances = np.array(instance['distances'])
    commodities = instance['commodities']

    # Calculate distances if not provided
    # distances = calculate_distances(nodes)

    model = gp.Model()
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('OutputFlag', 0)  # Silence Gurobi output

    # Variables
    hub_open = model.addVars(num_nodes, vtype=GRB.BINARY, name='hub_open')
    assignment = model.addVars(num_commodities, num_nodes, num_nodes, vtype=GRB.BINARY, name='assignment')

    # Objective
    total_cost = gp.quicksum(nodes[k]['fixed_cost'] * hub_open[k] for k in range(num_nodes))
    for c in range(num_commodities):
        commodity = commodities[c]
        origin, destination, flow = commodity['origin'], commodity['destination'], commodity['flow']
        for i in range(num_nodes):
            for j in range(i, num_nodes):
                if i == j:
                    cost = flow * (distances[origin, i] + distances[i, destination])
                else:
                    cost = flow * (distances[origin, i] + tau * distances[i, j] + distances[j, destination])
                total_cost += assignment[c, i, j] * cost
    model.setObjective(total_cost, GRB.MINIMIZE)

    # Constraints
    for c in range(num_commodities):
        commodity = commodities[c]
        origin, destination = commodity['origin'], commodity['destination']
        model.addConstr(gp.quicksum(assignment[c, i, j] for i in range(num_nodes) for j in range(i, num_nodes)) == 1)
        for i in range(num_nodes):
            for j in range(i, num_nodes):
                model.addConstr(assignment[c, i, j] <= hub_open[i])
                model.addConstr(assignment[c, i, j] <= hub_open[j])

    # Special case for commodities with the same origin and destination
    for c in range(num_commodities):
        commodity = commodities[c]
        if commodity['origin'] == commodity['destination']:
            model.addConstr(gp.quicksum(assignment[c, i, i] for i in range(num_nodes)) == 1)

    model.optimize()

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        objective_value = model.ObjVal
        open_hubs = [k for k in range(num_nodes) if hub_open[k].X > 0.5]
        assignments = []
        for c in range(num_commodities):
            commodity = commodities[c]
            origin, destination = commodity['origin'], commodity['destination']
            min_cost = float('inf')
            best_hubs = None
            for i in range(num_nodes):
                for j in range(i, num_nodes):
                    if assignment[c, i, j].X > 0.5:
                        if i == j:
                            cost = commodity['flow'] * (distances[origin, i] + distances[i, destination])
                            hub_edge = [i]
                        else:
                            cost = commodity['flow'] * (distances[origin, i] + tau * distances[i, j] + distances[j, destination])
                            hub_edge = [i, j]
                        assignments.append({
                            'commodity': c,
                            'origin': origin,
                            'destination': destination,
                            'hub_edge': hub_edge,
                            'cost': cost
                        })
        solution = {
            'objective_value': objective_value,
            'open_hubs': open_hubs,
            'assignments': assignments
        }
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=4)
        if logger:
            logger.log(objective_value)
    else:
        print("No solution found.")

if __name__ == "__main__":
    main()
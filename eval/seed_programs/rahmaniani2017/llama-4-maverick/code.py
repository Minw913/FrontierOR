import argparse
import json
import math
from gurobipy import Model, GRB, quicksum

class SolutionLogger:
    def __init__(self, log_path, sense):
        self.log_path = log_path
        self.sense = sense
        self.best_objective = float('inf') if sense == 'minimize' else -float('inf')

    def log(self, objective_value):
        if (self.sense == 'minimize' and objective_value < self.best_objective) or (self.sense == 'maximize' and objective_value > self.best_objective):
            self.best_objective = objective_value
            with open(self.log_path, 'a') as f:
                f.write(json.dumps({'objective_value': objective_value}) + '\n')

def calculate_distances(nodes, TC):
    num_nodes = len(nodes)
    distances = [[0 for _ in range(num_nodes)] for _ in range(num_nodes)]
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            dist = math.sqrt((nodes[i]['x'] - nodes[j]['x'])**2 + (nodes[i]['y'] - nodes[j]['y'])**2) * TC
            distances[i][j] = distances[j][i] = dist
    return distances

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, required=False)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, 'minimize') if args.log_path else None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    num_nodes = instance['num_nodes']
    num_commodities = instance['num_commodities']
    tau = instance['parameters']['tau']
    chi = instance['parameters']['chi']
    delta = instance['parameters']['delta']
    TC = instance['parameters']['TC']
    nodes = instance['nodes']
    distances = instance['distances'] if 'distances' in instance else calculate_distances(nodes, TC)
    commodities = instance['commodities']

    model = Model("Hub Location Problem")
    model.Params.TimeLimit = args.time_limit
    model.Params.Threads = 1  # Ensure single-threaded execution

    # Variables
    y = {i: model.addVar(vtype=GRB.BINARY, name=f"y_{i}") for i in range(num_nodes)}
    z = {(k, i, j): model.addVar(vtype=GRB.BINARY, name=f"z_{k}_{i}_{j}") for k in range(num_commodities) for i in range(num_nodes) for j in range(i, num_nodes)}

    # Objective
    model.setObjective(
        quicksum(node['fixed_cost'] * y[node['id']] for node in nodes) +
        quicksum(commodities[k]['flow'] * (
            chi * distances[commodities[k]['origin']][i] +
            tau * distances[i][j] * (0 if i == j else 1) +
            delta * distances[j][commodities[k]['destination']]
        ) * z[(k, i, j)] for k in range(num_commodities) for i in range(num_nodes) for j in range(i, num_nodes)),
        GRB.MINIMIZE
    )

    # Constraints
    for k in range(num_commodities):
        model.addConstr(quicksum(z[(k, i, j)] for i in range(num_nodes) for j in range(i, num_nodes)) == 1, name=f"single_routing_{k}")
    for k in range(num_commodities):
        for i in range(num_nodes):
            for j in range(i, num_nodes):
                model.addConstr(z[(k, i, j)] <= y[i], name=f"hub_open_{k}_{i}_{j}_1")
                model.addConstr(z[(k, i, j)] <= y[j], name=f"hub_open_{k}_{i}_{j}_2")

    model.optimize()

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        open_hubs = [i for i in range(num_nodes) if y[i].x > 0.5]
        assignments = []
        for k in range(num_commodities):
            origin = commodities[k]['origin']
            destination = commodities[k]['destination']
            min_cost = float('inf')
            best_hubs = None
            for i in open_hubs:
                for j in open_hubs:
                    if i <= j and z[(k, i, j)].x > 0.5:
                        if i == j:
                            cost = commodities[k]['flow'] * (chi * distances[origin][i] + delta * distances[i][destination])
                        else:
                            cost = commodities[k]['flow'] * (chi * distances[origin][i] + tau * distances[i][j] + delta * distances[j][destination])
                        min_cost = cost
                        best_hubs = [i, j] if i != j else [i]
            assignments.append({
                'commodity': k,
                'origin': origin,
                'destination': destination,
                'hub_edge': best_hubs,
                'cost': min_cost
            })

        solution = {
            'objective_value': model.getObjective().getValue(),
            'open_hubs': open_hubs,
            'assignments': assignments
        }

        if logger:
            logger.log(model.getObjective().getValue())

        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=4)

if __name__ == "__main__":
    main()
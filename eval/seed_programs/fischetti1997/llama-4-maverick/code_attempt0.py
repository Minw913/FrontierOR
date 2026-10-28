import argparse
import json
import math
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    return instance

def compute_distance_matrix(instance):
    n = instance['problem_size']['n']
    distance_matrix = [[0] * n for _ in range(n)]
    values = instance['distance_matrix_upper_triangular']['values']
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            distance_matrix[i][j] = distance_matrix[j][i] = values[idx]
            idx += 1
    return distance_matrix

def solve_gtsp(instance, time_limit):
    n = instance['problem_size']['n']
    m = instance['problem_size']['m']
    nodes = instance['nodes']
    clusters = instance['clusters']
    distance_matrix = compute_distance_matrix(instance)

    model = gp.Model()
    model.Params.TimeLimit = time_limit
    model.Params.LogToConsole = 0

    # Variables
    x = model.addVars(n, n, vtype=GRB.BINARY, name='x')
    y = model.addVars(n, vtype=GRB.BINARY, name='y')
    u = model.addVars(m, vtype=GRB.INTEGER, name='u')

    # Objective
    model.setObjective(gp.quicksum(distance_matrix[i][j] * x[i, j] for i in range(n) for j in range(n)), GRB.MINIMIZE)

    # Constraints
    for cluster in clusters:
        model.addConstr(gp.quicksum(y[node_id] for node_id in cluster['node_ids']) == 1)

    for i in range(n):
        model.addConstr(gp.quicksum(x[i, j] for j in range(n) if j != i) == y[i])
        model.addConstr(gp.quicksum(x[j, i] for j in range(n) if j != i) == y[i])

    for i in range(m):
        for j in range(m):
            if i != j:
                for node_id_i in clusters[i]['node_ids']:
                    for node_id_j in clusters[j]['node_ids']:
                        model.addConstr(u[i] - u[j] + m * x[node_id_i, node_id_j] <= m - 1)

    for i in range(m):
        model.addConstr(u[i] >= 0)
        model.addConstr(u[i] <= m - 1)

    model.optimize()

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        tour = []
        visited_nodes = []
        for node in nodes:
            if y[node['id']].x > 0.5:
                visited_nodes.append(node['id'])
        current_node = visited_nodes[0]
        tour.append(current_node)
        visited = set([current_node])
        while len(tour) < m:
            for j in range(n):
                if x[current_node, j].x > 0.5 and j not in visited:
                    tour.append(j)
                    visited.add(j)
                    current_node = j
                    break
        objective_value = model.getObjective().getValue()
        return objective_value, tour, visited_nodes
    else:
        return None, None, None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, required=False)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    instance = read_instance(args.instance_path)
    def callback(model, where):
        if where == GRB.Callback.MIP:
            objective_value = model.cbGet(GRB.Callback.MIP_OBJBST)
            if logger:
                logger.log(objective_value)

    objective_value, tour, visited_nodes = solve_gtsp(instance, args.time_limit)

    if objective_value is not None:
        solution = {
            'objective_value': objective_value,
            'tour': tour,
            'visited_nodes': visited_nodes
        }
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=4)

if __name__ == "__main__":
    main()
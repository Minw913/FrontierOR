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

def solve(instance, time_limit, log_path):
    num_nodes = instance['num_nodes']
    distance_limitation_d0 = instance['distance_limitation_d0']
    depot_vertex = instance['depot_vertex']
    scores = instance['scores']
    distance_matrix = instance['distance_matrix']

    logger = SolutionLogger(log_path, sense="maximize") if log_path else None

    model = gp.Model()
    model.setParam('TimeLimit', time_limit)

    # Variables
    x = model.addVars(num_nodes, num_nodes, vtype=GRB.BINARY, name='x')
    y = model.addVars(num_nodes, vtype=GRB.BINARY, name='y')
    u = model.addVars(num_nodes, vtype=GRB.CONTINUOUS, name='u')

    # Objective
    model.setObjective(gp.quicksum(scores[i] * y[i] for i in range(num_nodes)), GRB.MAXIMIZE)

    # Constraints
    model.addConstr(gp.quicksum(distance_matrix[i][j] * x[i, j] for i in range(num_nodes) for j in range(num_nodes)) <= distance_limitation_d0)
    model.addConstr(y[depot_vertex] == 1)
    for i in range(num_nodes):
        model.addConstr(gp.quicksum(x[i, j] for j in range(num_nodes)) == y[i])
        model.addConstr(gp.quicksum(x[j, i] for j in range(num_nodes)) == y[i])
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                model.addConstr(u[i] - u[j] + num_nodes * x[i, j] <= num_nodes - 1)
    for i in range(num_nodes):
        model.addConstr(u[i] >= 0)
        model.addConstr(u[i] <= num_nodes - 1)

    # Optimize
    model.optimize()
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        objective_value = model.getObjective().getValue()
        visited_vertices = [i for i in range(num_nodes) if y[i].x > 0.5]
        tour_edges = [(i, j) for i in range(num_nodes) for j in range(num_nodes) if x[i, j].x > 0.5]

        # Logging
        if logger:
            logger.log(objective_value)

        return {
            'objective_value': objective_value,
            'visited_vertices': visited_vertices,
            'tour_edges': tour_edges
        }
    else:
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, required=False)
    args = parser.parse_args()

    instance = read_instance(args.instance_path)
    solution = solve(instance, args.time_limit, args.log_path)
    if solution is not None:
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f)
    else:
        print("No solution found.")

if __name__ == "__main__":
    main()
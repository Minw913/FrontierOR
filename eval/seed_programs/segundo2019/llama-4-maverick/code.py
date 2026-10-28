import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, required=False)
    return parser.parse_args()

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    return instance

def solve_mewcp(instance, time_limit, logger):
    graph = instance['graph']
    num_vertices = graph['num_vertices']
    vertices = graph['vertices']
    edges = graph['edges']

    model = gp.Model('MEWCP')
    model.setParam("Threads", 1)
    model.Params.LogToConsole = 0  # Disable console logging

    # Create variables
    x = model.addVars(vertices, vtype=GRB.BINARY, name='x')
    y = model.addVars([(u, v) for u, v, _ in edges], vtype=GRB.BINARY, name='y')

    # Objective function
    model.setObjective(gp.quicksum(weight * y[u, v] for u, v, weight in edges), GRB.MAXIMIZE)

    # Constraints
    for u, v, _ in edges:
        model.addConstr(y[u, v] <= x[u])
        model.addConstr(y[u, v] <= x[v])
        model.addConstr(x[u] + x[v] <= 1 + y[u, v])

    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            if not any((u == vertices[i] and v == vertices[j]) or (u == vertices[j] and v == vertices[i]) for u, v, _ in edges):
                model.addConstr(x[vertices[i]] + x[vertices[j]] <= 1)

    # Set time limit
    model.Params.TimeLimit = time_limit

    # Optimize
    model.optimize(lambda model, where: callback(model, where, logger))

    # Extract solution
    objective_value = model.getObjective().getValue()
    clique_vertices = sorted([v for v in vertices if x[v].X > 0.5])
    clique_edges = sorted([[u, v] if u < v else [v, u] for u, v, _ in edges if y[u, v].X > 0.5])

    return objective_value, clique_vertices, clique_edges

def callback(model, where, logger):
    if where == GRB.Callback.MIPSOL:
        objective_value = model.cbGet(GRB.Callback.MIPSOL_OBJ)
        if logger:
            logger.log(objective_value)

def write_solution(solution_path, objective_value, clique_vertices, clique_edges):
    solution = {
        'objective_value': objective_value,
        'clique_vertices': clique_vertices,
        'clique_edges': clique_edges
    }
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=4)

def main():
    args = parse_args()
    instance = read_instance(args.instance_path)
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None
    objective_value, clique_vertices, clique_edges = solve_mewcp(instance, args.time_limit, logger)
    write_solution(args.solution_path, objective_value, clique_vertices, clique_edges)

if __name__ == '__main__':
    main()
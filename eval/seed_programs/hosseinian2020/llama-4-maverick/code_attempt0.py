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
    parser.add_argument('--log_path', type=str, default=None)
    return parser.parse_args()

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    return instance

def solve(instance, time_limit, logger):
    num_vertices = instance['num_vertices']
    vertices = instance['vertices']
    edges = instance['edges']

    model = gp.Model('max_clique')
    model.setParam('TimeLimit', time_limit)
    model.setParam('LogToConsole', 0)

    # Create variables
    vertex_vars = model.addVars(vertices, vtype=GRB.BINARY, name='vertex')
    edge_vars = model.addVars([(edge['i'], edge['j']) for edge in edges], vtype=GRB.BINARY, name='edge')

    # Set objective function
    model.setObjective(gp.quicksum(edge['weight'] * edge_vars[edge['i'], edge['j']] for edge in edges), GRB.MAXIMIZE)

    # Add constraints
    for edge in edges:
        i, j = edge['i'], edge['j']
        model.addConstr(edge_vars[i, j] <= vertex_vars[i])
        model.addConstr(edge_vars[i, j] <= vertex_vars[j])

    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            if not any((edge['i'] == vertices[i] and edge['j'] == vertices[j]) or (edge['i'] == vertices[j] and edge['j'] == vertices[i]) for edge in edges):
                model.addConstr(vertex_vars[vertices[i]] + vertex_vars[vertices[j]] <= 1)

    model.optimize()

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        objective_value = model.getObjective().getValue()
        clique = [int(v.VarName.split('_')[1]) for v in model.getVars() if v.VarName.startswith('vertex') and v.x > 0.5]
        if logger:
            logger.log(objective_value)
        return objective_value, clique
    else:
        raise Exception(f'Gurobi model status: {model.status}')

def write_solution(solution_path, objective_value, clique):
    solution = {'objective_value': objective_value, 'clique': clique}
    with open(solution_path, 'w') as f:
        json.dump(solution, f)

def main():
    args = parse_args()
    instance = read_instance(args.instance_path)
    logger = SolutionLogger(args.log_path, sense='maximize') if args.log_path else None
    objective_value, clique = solve(instance, args.time_limit, logger)
    write_solution(args.solution_path, objective_value, clique)

if __name__ == '__main__':
    main()
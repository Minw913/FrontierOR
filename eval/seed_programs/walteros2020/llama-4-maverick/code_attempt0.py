import json
import argparse
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    return instance

def solve_max_clique(instance, time_limit, log_path):
    n = instance['n']
    edges = instance['edges']
    edge_set = set((min(u, v), max(u, v)) for u, v in edges)

    model = gp.Model()
    model.Params.TimeLimit = time_limit
    model.Params.LogToConsole = False

    x = model.addVars(n, vtype=GRB.BINARY, name='x')
    model.setObjective(gp.quicksum(x[i] for i in range(n)), GRB.MAXIMIZE)

    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) not in edge_set:
                model.addConstr(x[i] + x[j] <= 1, f'non_edge_{i}_{j}')

    logger = SolutionLogger(log_path, sense="maximize") if log_path else None

    def log_solution(model, where):
        if where == GRB.Callback.MIP:
            obj_value = model.cbGet(GRB.Callback.MIP_OBJBST)
            if logger:
                logger.log(obj_value)

    model.optimize(callback=log_solution)

    if logger:
        logger.log(model.objVal)

    clique_vertices = [i for i in range(n) if x[i].x > 0.5]
    objective_value = len(clique_vertices)

    return objective_value, clique_vertices

def write_solution(solution_path, objective_value, clique_vertices):
    solution = {
        'objective_value': objective_value,
        'clique_vertices': clique_vertices
    }
    with open(solution_path, 'w') as f:
        json.dump(solution, f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str)
    args = parser.parse_args()

    instance = read_instance(args.instance_path)
    objective_value, clique_vertices = solve_max_clique(instance, args.time_limit, args.log_path)
    write_solution(args.solution_path, objective_value, clique_vertices)

if __name__ == '__main__':
    main()
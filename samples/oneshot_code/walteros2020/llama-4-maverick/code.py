import json
import argparse
from gurobipy import Model, GRB
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    return instance

def solve_max_clique(instance, time_limit, log_path):
    n = instance['n']
    edges = instance['edges']

    model = Model("max_clique")
    model.Params.TimeLimit = time_limit
    model.Params.LogToConsole = 0

    x = model.addVars(n, vtype=GRB.BINARY, name="x")
    model.setObjective(sum(x[i] for i in range(n)), GRB.MAXIMIZE)

    for i in range(n):
        for j in range(i+1, n):
            if [i, j] not in edges and [j, i] not in edges:
                model.addConstr(x[i] + x[j] <= 1, f"non_edge_{i}_{j}")

    logger = SolutionLogger(log_path, sense="maximize") if log_path else None

    model.optimize(lambda model, where: callback(model, where, logger))

    objective_value = int(model.getObjective().getValue())
    clique_vertices = [i for i in range(n) if x[i].x > 0.5]

    return objective_value, sorted(clique_vertices)

def callback(model, where, logger):
    if where == GRB.Callback.MIPSOL:
        objective_value = model.cbGet(GRB.Callback.MIPSOL_OBJ)
        if logger:
            logger.log(objective_value)

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
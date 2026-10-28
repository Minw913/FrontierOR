import json
import argparse
from solution_logger import SolutionLogger
import gurobipy as gp
from gurobipy import GRB

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    return instance

def solve_instance(instance, time_limit, logger):
    num_items = instance['num_items']
    num_objectives = instance['num_objectives']
    weights = instance['weights']
    capacity = instance['capacity']
    objective_coefficients = instance['objective_coefficients']

    model = gp.Model()
    model.Params.TimeLimit = time_limit
    model.Params.LogToConsole = 0
    
    # Variables
    x = model.addVars(num_items, vtype=GRB.BINARY, name='x')
    objvars = model.addVars(num_objectives, lb=0, name='objvars')
    
    # Constraints
    model.addConstr(gp.quicksum(weights[i] * x[i] for i in range(num_items)) <= capacity, name='capacity')
    for k in range(num_objectives):
        model.addConstr(objvars[k] == gp.quicksum(objective_coefficients[k][i] * x[i] for i in range(num_items)), name=f'obj{k}')

    # Set objective
    model.setObjective(gp.quicksum(objvars[k] for k in range(num_objectives)), GRB.MINIMIZE)

    # Optimize
    model.optimize()

    non_dominated_objective_vectors = []
    solutions = []

    if model.SolCount > 0:
        for sol in range(model.SolCount):
            model.Params.SolutionNumber = sol
            objvals = [model.getVarByName(f'objvars[{k}]').Xn for k in range(num_objectives)]
            is_dominated = False
            for existing_objvals in non_dominated_objective_vectors:
                if all(objvals[k] >= existing_objvals[k] for k in range(num_objectives)) and any(objvals[k] > existing_objvals[k] for k in range(num_objectives)):
                    is_dominated = True
                    break
                elif all(objvals[k] <= existing_objvals[k] for k in range(num_objectives)) and any(objvals[k] < existing_objvals[k] for k in range(num_objectives)):
                    non_dominated_objective_vectors.remove(existing_objvals)
                    solutions.remove(solutions[non_dominated_objective_vectors.index(existing_objvals)])
            if not is_dominated:
                non_dominated_objective_vectors.append(objvals)
                solution = [int(model.getVarByName(f'x[{i}]').Xn) for i in range(num_items)]
                solutions.append(solution)

    objective_value = len(non_dominated_objective_vectors)

    if logger:
        for sol in range(model.SolCount):
            model.Params.SolutionNumber = sol
            objvals = sum([model.getVarByName(f'objvars[{k}]').Xn for k in range(num_objectives)])
            logger.log(objvals)

    return objective_value, non_dominated_objective_vectors, solutions

def write_solution(solution_path, objective_value, non_dominated_objective_vectors, solutions):
    solution = {
        'objective_value': objective_value,
        'non_dominated_objective_vectors': non_dominated_objective_vectors,
        'solutions': solutions
    }
    with open(solution_path, 'w') as f:
        json.dump(solution, f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, required=False)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    instance = read_instance(args.instance_path)
    objective_value, non_dominated_objective_vectors, solutions = solve_instance(instance, args.time_limit, logger)
    write_solution(args.solution_path, objective_value, non_dominated_objective_vectors, solutions)

if __name__ == '__main__':
    main()
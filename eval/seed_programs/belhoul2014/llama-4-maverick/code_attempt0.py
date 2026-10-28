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

def solve_instance(instance, time_limit, logger):
    n = instance['n']
    p = instance['p']
    cost_matrices = instance['cost_matrices']
    reference_point = instance['reference_point']
    search_direction_lambda = instance['search_direction_lambda']

    model = gp.Model()
    model.Params.TimeLimit = time_limit
    model.Params.LogToConsole = 0

    # Assignment variables: x[i, j] = 1 if task i is assigned to agent j, 0 otherwise
    x = model.addVars(n, n, vtype=GRB.BINARY, name='x')

    # Auxiliary variable to represent the worst-case weighted deviation
    worst_case_deviation = model.addVar(lb=-GRB.INFINITY, name='worst_case_deviation')

    # Objective: minimize the worst-case weighted deviation
    model.setObjective(worst_case_deviation, GRB.MINIMIZE)

    # Constraints to ensure that worst_case_deviation is at least as large as the weighted deviation for each criterion
    for k in range(p):
        criterion_deviation = gp.quicksum(cost_matrices[k][i][j] * x[i, j] for i in range(n) for j in range(n)) - reference_point[k]
        weighted_deviation = search_direction_lambda[k] * criterion_deviation
        model.addConstr(worst_case_deviation >= weighted_deviation, name=f'deviation_constr_{k}')

    # Constraints to ensure one-to-one assignment
    for i in range(n):
        model.addConstr(gp.quicksum(x[i, j] for j in range(n)) == 1, name=f'task_constr_{i}')
    for j in range(n):
        model.addConstr(gp.quicksum(x[i, j] for i in range(n)) == 1, name=f'agent_constr_{j}')

    model.optimize()

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        objective_value = worst_case_deviation.X
        assignment = [int(gp.quicksum(j * x[i, j] for j in range(n)).getValue()) for i in range(n)]
        assignment_matrix = [[int(x[i, j].X) for j in range(n)] for i in range(n)]

        if logger:
            logger.log(objective_value)

        return {
            'objective_value': objective_value,
            'assignment': assignment,
            'assignment_matrix': assignment_matrix
        }
    else:
        raise RuntimeError(f'Gurobi model failed to find a feasible solution. Status: {model.status}')

def write_solution(solution_path, solution):
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=4)

def main():
    args = parse_args()
    instance = read_instance(args.instance_path)
    logger = SolutionLogger(args.log_path, sense='minimize') if args.log_path else None
    try:
        solution = solve_instance(instance, args.time_limit, logger)
        write_solution(args.solution_path, solution)
    except Exception as e:
        print(f'An error occurred: {e}')
    finally:
        if logger:
            logger.close()

if __name__ == '__main__':
    main()
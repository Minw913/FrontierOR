import argparse
import json
from gurobipy import GRB, Model, quicksum
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        data = json.load(f)
    
    n = data['n']
    p = data['p']
    cost_matrices = data['cost_matrices']
    reference_point = data['reference_point']
    lambdas = data['search_direction_lambda']

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    model = Model()
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', args.time_limit)

    x = [[model.addVar(vtype=GRB.BINARY) for j in range(n)] for i in range(n)]
    z = model.addVar(lb=-GRB.INFINITY, name='z')

    for i in range(n):
        model.addConstr(quicksum(x[i][j] for j in range(n)) == 1)
    for j in range(n):
        model.addConstr(quicksum(x[i][j] for i in range(n)) == 1)

    for k in range(p):
        total_cost = quicksum(cost_matrices[k][i][j] * x[i][j] for i in range(n) for j in range(n))
        model.addConstr(z >= lambdas[k] * (total_cost - reference_point[k]))

    model.setObjective(z, GRB.MINIMIZE)

    def callback(model_cb, where):
        if where == GRB.Callback.MIPSOL:
            x_vals = [[model_cb.cbGetSolution(x[i][j]) for j in range(n)] for i in range(n)]
            sums = [0.0] * p
            for i in range(n):
                for j in range(n):
                    if x_vals[i][j] > 0.5:
                        for k in range(p):
                            sums[k] += cost_matrices[k][i][j]
            obj = max(lambdas[k] * (sums[k] - reference_point[k]) for k in range(p))
            if logger:
                logger.log(obj)

    model.optimize(callback)

    assignment = [0] * n
    assignment_matrix = [[0]*n for _ in range(n)]
    if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        for i in range(n):
            for j in range(n):
                if x[i][j].X > 0.5:
                    assignment[i] = j
                    assignment_matrix[i][j] = 1

        sums = [0] * p
        for i in range(n):
            for j in range(n):
                if assignment_matrix[i][j]:
                    for k in range(p):
                        sums[k] += cost_matrices[k][i][j]
        objective_value = max(lambdas[k] * (sums[k] - reference_point[k]) for k in range(p))
    else:
        raise Exception("No feasible solution found.")

    solution = {
        "objective_value": float(objective_value),
        "assignment": assignment,
        "assignment_matrix": assignment_matrix
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, separators=(',', ':'))
    
    if logger:
        logger.close()

if __name__ == '__main__':
    main()
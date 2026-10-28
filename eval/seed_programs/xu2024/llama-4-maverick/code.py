import json
import argparse
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    return instance

def solve_instance(instance, time_limit, log_path):
    n = instance['n']
    k = instance['k']
    b = instance['b']
    r = instance['r']
    Q = instance['Q']

    logger = SolutionLogger(log_path, sense="minimize") if log_path else None

    model = gp.Model()
    model.setParam("Threads", 1)
    model.setParam('TimeLimit', time_limit)
    model.setParam('OutputFlag', 0)  # Silence Gurobi output

    x = model.addVars(n, lb=-GRB.INFINITY, name='x')
    z = model.addVars(n, vtype=GRB.BINARY, name='z')

    model.setObjective(gp.quicksum(Q[i][j] * x[i] * x[j] for i in range(n) for j in range(n)) / 2)

    model.addConstr(gp.quicksum(x[i] for i in range(n)) == 1, name='budget')
    model.addConstr(gp.quicksum(r[i] * x[i] for i in range(n)) >= b, name='return')
    model.addConstr(gp.quicksum(z[i] for i in range(n)) <= k, name='cardinality')

    for i in range(n):
        model.addConstr(x[i] <= z[i] * 1e6, name=f'linking_ub_{i}')
        model.addConstr(x[i] >= -z[i] * 1e6, name=f'linking_lb_{i}')

    model.optimize()

    if model.status == GRB.OPTIMAL:
        objective_value = model.getObjective().getValue()
        x_sol = [x[i].X for i in range(n)]
    else:
        objective_value = model.getObjective().getValue() if model.SolCount > 0 else float('inf')
        x_sol = [x[i].X for i in range(n)] if model.SolCount > 0 else [0.0] * n

    if logger and model.SolCount > 0:
        logger.log(objective_value)

    return objective_value, x_sol

def write_solution(solution_path, objective_value, x_sol):
    solution = {
        'objective_value': objective_value,
        'x': x_sol
    }
    with open(solution_path, 'w') as f:
        json.dump(solution, f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str)
    args = parser.parse_args()

    instance = read_instance(args.instance_path)
    objective_value, x_sol = solve_instance(instance, args.time_limit, args.log_path)
    write_solution(args.solution_path, objective_value, x_sol)

if __name__ == '__main__':
    main()
import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    K = float(instance['problem_parameters']['knapsack_capacity'])
    Gamma = float(instance['problem_parameters']['uncertainty_budget'])
    items = instance['items']
    n = len(items)

    w = [float(item['weight']) for item in items]
    p = [float(item['nominal_profit']) for item in items]
    d = [float(item['max_degradation']) for item in items]
    c = [float(item['outsource_penalty']) for item in items]
    r = [float(item['repair_capacity']) for item in items]

    model = gp.Model('robust_knapsack')
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', max(1, args.time_limit))
    model.setParam('OutputFlag', 0)

    x = model.addVars(n, vtype=GRB.BINARY, name='x')
    pi = model.addVars(n, vtype=GRB.CONTINUOUS, lb=0, ub=1, name='pi')
    sigma = model.addVars(n, vtype=GRB.CONTINUOUS, lb=0, ub=1, name='sigma')
    tau = model.addVars(n, vtype=GRB.CONTINUOUS, lb=0, name='tau')
    rho = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name='rho')

    obj = gp.quicksum(x[i] * (c[i] - p[i]) for i in range(n)) \
          - gp.quicksum(c[i] * pi[i] for i in range(n)) \
          + gp.quicksum(tau[i] for i in range(n)) \
          + Gamma * rho
    model.setObjective(obj, GRB.MINIMIZE)

    model.addConstr(gp.quicksum(w[i] * pi[i] + r[i] * sigma[i] for i in range(n)) <= K, 'capacity')

    for i in range(n):
        model.addConstr(pi[i] <= x[i], f'pi_leq_x_{i}')
        model.addConstr(pi[i] >= sigma[i], f'pi_geq_sigma_{i}')
        model.addConstr(d[i] * (sigma[i] - pi[i]) + tau[i] + rho >= 0, f'deg_{i}')

    def callback(model_cb, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model_cb.cbGet(GRB.Callback.MIPSOL_OBJ)
            x_vals = model_cb.cbGetSolution(x)
            solution_dict = {str(i): int(round(x_vals[i])) for i in range(n)}
            if logger:
                logger.log_solution(obj_val, solution_dict)

    model.optimize(callback)

    if model.SolCount > 0:
        obj_val = model.ObjVal
        x_vals = [int(round(x[i].X)) for i in range(n)]
    else:
        obj_val = 0.0
        x_vals = [0] * n

    solution_dict = {str(i): x_vals[i] for i in range(n)}

    if logger:
        logger.log_solution(float(obj_val), solution_dict)

    solution = {
        'objective_value': float(obj_val),
        'x': solution_dict
    }
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == '__main__':
    main()
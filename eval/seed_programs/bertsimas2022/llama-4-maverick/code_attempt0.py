import json
import argparse
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        return json.load(f)

def solve(instance):
    n = instance['n']
    k = instance['k']
    r = instance['r']
    gamma = instance['gamma']
    kappa = instance['kappa']
    mu = instance['mu']
    factor_loadings = instance['factor_loadings']
    idiosyncratic_variance = instance['idiosyncratic_variance']
    constraints = instance['constraints']
    has_min_return_constraint = constraints['has_min_return_constraint']
    r_bar = constraints['r_bar']
    has_min_investment_constraint = constraints['has_min_investment_constraint']
    l_min_investment = constraints['l_min_investment']
    u_max_investment = constraints['u_max_investment']

    model = gp.Model()
    model.Params.LogToConsole = 0
    
    x = model.addVars(n, lb=0, ub=1, name='x')
    z = model.addVars(n, vtype=GRB.BINARY, name='z')
    
    model.addConstr(x.sum() == 1)
    model.addConstr(z.sum() <= k)
    
    for i in range(n):
        model.addConstr(x[i] <= z[i])
        if has_min_investment_constraint:
            model.addConstr(x[i] >= l_min_investment[i] * z[i])
        if u_max_investment is not None:
            model.addConstr(x[i] <= u_max_investment[i] * z[i])
    
    if has_min_return_constraint:
        model.addConstr(gp.quicksum(mu[i] * x[i] for i in range(n)) >= r_bar)
    
    objective = 0
    for i in range(n):
        for j in range(i, n):
            if i == j:
                cov_ij = sum(factor_loadings[i][f] * factor_loadings[j][f] for f in range(r)) + idiosyncratic_variance[i]
            else:
                cov_ij = sum(factor_loadings[i][f] * factor_loadings[j][f] for f in range(r))
            objective += x[i] * cov_ij * x[j]
    objective *= 0.5
    
    concentration_penalty = gp.quicksum(x[i] * x[i] for i in range(n)) / (2 * gamma)
    objective += concentration_penalty
    
    if kappa == 1:
        objective -= gp.quicksum(mu[i] * x[i] for i in range(n))
    
    model.setObjective(objective, GRB.MINIMIZE)
    
    return model, x

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, required=False)
    args = parser.parse_args()
    
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    instance = read_instance(args.instance_path)
    model, x = solve(instance)
    
    model.Params.TimeLimit = args.time_limit
    model.optimize()
    
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        if logger and model.SolCount > 0:
            logger.log(model.ObjVal)
        solution = {
            'objective_value': model.ObjVal if model.SolCount > 0 else float('inf'),
            'x': [x[i].X for i in range(instance['n'])] if model.SolCount > 0 else [0.0] * instance['n']
        }
    else:
        solution = {
            'objective_value': float('inf'),
            'x': [0.0] * instance['n']
        }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == '__main__':
    main()
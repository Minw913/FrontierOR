import argparse
import json
from gurobipy import Model, GRB, quicksum
import numpy as np
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        return json.load(f)

def solve_one_stage_IP_mu(instance, K, model, logger):
    N = instance['num_scenarios_N']
    n = instance['problem_dimension_n']
    scenarios = np.array(instance['scenarios'])
    
    model.setParam('OutputFlag', 0)
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    
    # Variables
    t = model.addVar(lb=0.0, ub=1.0, name='t')
    x = model.addVars(N, vtype=GRB.BINARY, name='x')
    mu = model.addVars(N, N, vtype=GRB.BINARY, name='mu')
    lambda_weights = model.addVars(N, N, vtype=GRB.CONTINUOUS, lb=0.0, name='lambda_weights')
    
    # Constraints
    model.addConstr(quicksum(x[i] for i in range(N)) == K, name='num_scenarios')
    for i in range(N):
        model.addConstr(quicksum(mu[i, j] for j in range(N)) == 1, name=f'mu_sum_{i}')
        for j in range(N):
            model.addConstr(mu[i, j] <= x[j], name=f'mu_ub_{i}_{j}')
    for j in range(N):
        model.addConstr(quicksum(lambda_weights[j, i] for i in range(N)) == x[j], name=f'lambda_sum_{j}')
        for i in range(N):
            model.addConstr(lambda_weights[j, i] <= mu[i, j], name=f'lambda_ub_{j}_{i}')
    for i in range(N):
        for d in range(n):
            model.addConstr(t * quicksum(scenarios[j, d] * mu[i, j] for j in range(N)) <= scenarios[i, d], name=f't_lb_{i}_{d}')
    
    # Objective
    model.setObjective(t, GRB.MAXIMIZE)
    
    model.optimize()
    
    if model.status == GRB.OPTIMAL:
        assignment = {i: j for i in range(N) for j in range(N) if mu[i, j].X > 0.5}
        lambda_weights_dict = {j: {i: lambda_weights[j, i].X for i in range(N) if lambda_weights[j, i].X > 0} for j in range(N) if x[j].X > 0.5}
        return {
            't': t.X,
            'approximation_ratio': 1 / t.X if t.X > 0 else float('inf'),
            'assignment': assignment,
            'lambda_weights': lambda_weights_dict,
            'solver_status': model.status,
            'best_bound': model.ObjBound,
            'mip_gap': model.MIPGap
        }
    else:
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, required=False)
    args = parser.parse_args()
    
    instance = read_instance(args.instance_path)
    N = instance['num_scenarios_N']
    K_values = instance['K_values']
    
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None
    
    one_stage_IP_mu_results = {}
    model = Model('one_stage_IP_mu')
    model.setParam('TimeLimit', args.time_limit)
    
    best_objective_value = -float('inf')
    solution_dict = {}
    
    for K in K_values:
        result = solve_one_stage_IP_mu(instance, K, model, logger)
        if result is not None:
            one_stage_IP_mu_results[str(K)] = result
            if result['t'] > best_objective_value:
                best_objective_value = result['t']
                solution_dict = {'one_stage_IP_mu': one_stage_IP_mu_results, 'objective_value': best_objective_value}
                if logger:
                    logger.log_solution(best_objective_value, solution_dict)
    
    output = {
        'objective_value': best_objective_value,
        'one_stage_IP_mu': one_stage_IP_mu_results,
        'two_stage_IP': {}  # As per the schema, this should be included even if not implemented
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(output, f, indent=4)

if __name__ == '__main__':
    main()
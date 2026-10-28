import json
import argparse
import time
import numpy as np
from gurobipy import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None
    
    start_time = time.time()
    
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    N = instance['num_scenarios_N']
    n = instance['problem_dimension_n']
    K_values = instance['K_values']
    scenarios = np.array(instance['scenarios'], dtype=float)
    
    # For each K, solve one-stage and two-stage
    solution = {
        'objective_value': 0.0,
        'one_stage_IP_mu': {},
        'two_stage_IP': {}
    }
    
    best_overall_t = 0.0
    
    for K in K_values:
        if time.time() - start_time > args.time_limit - 1:
            break
        
        # Solve two-stage IP
        two_stage_result = solve_two_stage(scenarios, N, n, K, args.time_limit - (time.time() - start_time))
        solution['two_stage_IP'][str(K)] = two_stage_result
        
        # Solve one-stage IP-mu
        one_stage_result = solve_one_stage(scenarios, N, n, K, args.time_limit - (time.time() - start_time))
        solution['one_stage_IP_mu'][str(K)] = one_stage_result
        
        # Update best objective
        best_t = max(two_stage_result['t'], one_stage_result['t'])
        if best_t > best_overall_t:
            best_overall_t = best_t
            solution['objective_value'] = best_overall_t
            if logger:
                logger.log_solution(best_overall_t, solution)
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

def solve_two_stage(scenarios, N, n, K, time_limit):
    # Bisection on t
    lo, hi = 0.0, 10.0  # reasonable range
    best_t = 0.0
    best_assignment = None
    best_selected = None
    solver_status = 0
    best_bound = 0.0
    mip_gap = 1.0
    
    for iteration in range(30):  # bisection iterations
        if time.time() - start_time > time_limit:
            break
        
        t_mid = (lo + hi) / 2.0
        
        # Check feasibility for t_mid
        feasible, assignment, selected, status, bound = check_two_stage_feasibility(
            scenarios, N, n, K, t_mid, time_limit
        )
        
        if feasible:
            best_t = t_mid
            best_assignment = assignment
            best_selected = selected
            lo = t_mid
        else:
            hi = t_mid
    
    result = {
        't': best_t,
        'approximation_ratio': 1.0 / best_t if best_t > 0 else float('inf'),
        'selected_scenarios': best_selected if best_selected else list(range(K))
    }
    
    return result
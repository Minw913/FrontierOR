import argparse
import json
import time
import numpy as np
from itertools import permutations
from scipy.optimize import linear_sum_assignment
import random

def solve(instance_path, solution_path, time_limit, log_path):
    from solution_logger import SolutionLogger
    
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    
    n = instance['n']
    A = np.array(instance['A'], dtype=float)
    B = np.array(instance['B'], dtype=float)
    C = np.array(instance['C'], dtype=float)
    
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    start_time = time.time()
    
    best_assignment = None
    best_obj = float('inf')
    
    idx = np.arange(n)
    
    def compute_objective(perm):
        p = np.array(perm, dtype=int)
        sum_a = float(A[idx, p].sum())
        sum_b = float(B[idx, p].sum())
        sum_c = float(C[idx, p].sum())
        return sum_a * sum_b + sum_c
    
    def update_best(perm, obj=None):
        nonlocal best_assignment, best_obj
        perm = list(int(x) for x in perm)
        if obj is None:
            obj = compute_objective(perm)
        if obj < best_obj:
            best_obj = obj
            best_assignment = perm[:]
            if logger:
                logger.log(best_obj)
        return obj
    
    if n == 1:
        update_best([0])
    elif n <= 8:
        for perm in permutations(range(n)):
            if time.time() - start_time > time_limit - 0.5:
                break
            update_best(list(perm))
    else:
        # 1. Solve LAP on C alone
        row_ind, col_ind = linear_sum_assignment(C)
        update_best(col_ind.tolist())
        
        # 2. Solve LAP on A+B+C (heuristic)
        row_ind, col_ind = linear_sum_assignment(A + B + C)
        update_best(col_ind.tolist())
        
        # 3. Parametric approach
        row_ind, col_ind = linear_sum_assignment(B)
        min_sum_b = float(B[row_ind, col_ind].sum())
        row_ind, col_ind = linear_sum_assignment(-B)
        max_sum_b = float(B[row_ind, col_ind].sum())
        
        row_ind, col_ind = linear_sum_assignment(A)
        min_sum_a = float(A[row_ind, col_ind].sum())
        row_ind, col_ind = linear_sum_assignment(-A)
        max_sum_a = float(A[row_ind, col_ind].sum())
        
        num_steps = min(200, max(50, int((time_limit - 1) * 10)))
        
        # Parametric on beta (multiplier for A, representing sum_b)
        margin_b = max(abs(min_sum_b), abs(max_sum_b), 10) * 0.5
        betas = np.linspace(min_sum_b - margin_b, max_sum_b + margin_b, num_steps)
        for beta in betas:
            if time.time() - start_time > time_limit * 0.3:
                break
            cost_matrix = beta * A + C
            _, col_ind = linear_sum_assignment(cost_matrix)
            update_best(col_ind.tolist())
        
        # Parametric on alpha (multiplier for B, representing sum_a)
        margin_a = max(abs(min_sum_a), abs(max_sum_a), 10) * 0.5
        alphas = np.linspace(min_sum_a - margin_a, max_sum_a + margin_a, num_steps)
        for alpha in alphas:
            if time.time() - start_time > time_limit * 0.5:
                break
            cost_matrix = alpha * B + C
            _, col_ind = linear_sum_assignment(cost_matrix)
            update_best(col_ind.tolist())
        
        # Iterative approach
        perm = np.array(best_assignment, dtype=int)
        for iteration in range(50):
            if time.time() - start_time > time_limit * 0.6:
                break
            sum_a = float(A[idx, perm].sum())
            sum_b = float(B[idx, perm].sum())
            
            cost_matrix = sum_b * A + C
            _, col_ind = linear_sum_assignment(cost_matrix)
            update_best(col_ind.tolist())
            
            new_sum_a = float(A[idx, col_ind].sum())
            cost_matrix = new_sum_a * B + C
            _, col_ind2 = linear_sum_assignment(cost_matrix)
            update_best(col_ind2.tolist())
            
            new_perm = np.array(best_assignment, dtype=int)
            if np.array_equal(new_perm, perm):
                break
            perm = new_perm
        
        # Fine-grained parametric search
        perm = np.array(best_assignment, dtype=int)
        sum_a_best = float(A[idx, perm].sum())
        sum_b_best = float(B[idx, perm].sum())
        
        fine_range_b = np.linspace(sum_b_best - max(abs(sum_b_best)*0.3, 10),
                                    sum_b_best + max(abs(sum_b_best)*0.3, 10), 100)
        for beta in fine_range_b:
            if time.time() - start_time > time_limit * 0.7:
                break
            cost_matrix = beta * A + C
            _, col_ind = linear_sum_assignment(cost_matrix)
            update_best(col_ind.tolist())
        
        fine_range_a = np.linspace(sum_a_best - max(abs(sum_a_best)*0.3, 10),
                                    sum_a_best + max(abs(sum_a_best)*0.3, 10), 100)
        for alpha in fine_range_a:
            if time.time() - start_time > time_limit * 0.75:
                break
            cost_matrix = alpha * B + C
            _, col_ind = linear_sum_assignment(cost_matrix)
            update_best(col_ind.tolist())
        
        # Local search: 2-opt swaps
        def local_search_2opt(perm_in, deadline):
            perm_ls = list(int(x) for x in perm_in)
            cur_obj = compute_objective(perm_ls)
            improved = True
            while improved:
                improved = False
                for i in range(n):
                    if time.time() > deadline:
                        return perm_ls
                    for j in range(i+1, n):
                        perm_ls[i], perm_ls[j] = perm_ls[j], perm_ls[i]
                        new_obj = compute_objective(perm_ls)
                        if new_obj < cur_obj:
                            cur_obj = new_obj
                            update_best(perm_ls, new_obj)
                            improved = True
                        else:
                            perm_ls[i], perm_ls[j] = perm_ls[j], perm_ls[i]
            return perm_ls
        
        deadline = start_time + time_limit * 0.85
        local_search_2opt(best_assignment, deadline)
        
        # Random restarts with local search
        deadline = start_time + time_limit - 0.5
        while time.time() < deadline:
            perm = list(int(x) for x in best_assignment)
            num_swaps = random.randint(2, max(2, n // 3))
            for _ in range(num_swaps):
                i, j = random.sample(range(n), 2)
                perm[i], perm[j] = perm[j], perm[i]
            
            if random.random() < 0.3:
                beta = random.uniform(min_sum_b, max_sum_b)
                cost_matrix = beta * A + C
                _, col_ind = linear_sum_assignment(cost_matrix)
                perm = col_ind.tolist()
                update_best(perm)
            elif random.random() < 0.5:
                alpha = random.uniform(min_sum_a, max_sum_a)
                cost_matrix = alpha * B + C
                _, col_ind = linear_sum_assignment(cost_matrix)
                perm = col_ind.tolist()
                update_best(perm)
            
            perm = local_search_2opt(perm, min(time.time() + 1.0, deadline))
    
    solution = {
        "objective_value": float(best_obj),
        "assignment": best_assignment
    }
    
    with open(solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)
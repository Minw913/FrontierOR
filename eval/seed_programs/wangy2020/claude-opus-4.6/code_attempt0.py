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
    A = np.array(instance['A'])
    B = np.array(instance['B'])
    C = np.array(instance['C'])
    
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    start_time = time.time()
    
    best_assignment = None
    best_obj = float('inf')
    
    def compute_objective(perm):
        perm = np.array(perm)
        idx = np.arange(n)
        sum_a = A[idx, perm].sum()
        sum_b = B[idx, perm].sum()
        sum_c = C[idx, perm].sum()
        return sum_a * sum_b + sum_c
    
    def update_best(perm, obj=None):
        nonlocal best_assignment, best_obj
        if obj is None:
            obj = compute_objective(perm)
        if obj < best_obj:
            best_obj = obj
            best_assignment = list(perm)
            if logger:
                logger.log(best_obj)
        return obj
    
    # For very small n, enumerate all permutations
    if n <= 8:
        for perm in permutations(range(n)):
            if time.time() - start_time > time_limit - 0.5:
                break
            update_best(perm)
    else:
        # Strategy: Use multiple approaches
        # 1. Solve LAP on C alone
        row_ind, col_ind = linear_sum_assignment(C)
        perm_c = col_ind.tolist()
        update_best(perm_c)
        
        # 2. Solve LAP on A+B+C (heuristic)
        row_ind, col_ind = linear_sum_assignment(A + B + C)
        update_best(col_ind.tolist())
        
        # 3. Try to minimize sum_a * sum_b + sum_c using parametric approach
        # For a fixed lambda, minimize (A + lambda*B + ... ) type problems
        # The objective is f(perm) = sum_a(perm) * sum_b(perm) + sum_c(perm)
        # We can linearize: for fixed sum_b = beta, minimize beta * sum_a + sum_c
        # i.e., minimize sum_i (beta * A[i,perm[i]] + C[i,perm[i]])
        # Then check if the resulting sum_b matches beta.
        
        # Try parametric approach with different beta values
        # Collect candidate beta values
        # First, get range of possible sum_b
        row_ind, col_ind = linear_sum_assignment(B)
        min_sum_b = B[row_ind, col_ind].sum()
        row_ind, col_ind = linear_sum_assignment(-B)
        max_sum_b = B[row_ind, col_ind].sum()
        
        # Similarly for sum_a
        row_ind, col_ind = linear_sum_assignment(A)
        min_sum_a = A[row_ind, col_ind].sum()
        row_ind, col_ind = linear_sum_assignment(-A)
        max_sum_a = A[row_ind, col_ind].sum()
        
        # Try parametric: fix multiplier for A (treating it as if sum_b = beta)
        # minimize beta * sum_a + sum_c => LAP on (beta * A + C)
        # Also: fix multiplier for B (treating sum_a = alpha)
        # minimize alpha * sum_b + sum_c => LAP on (alpha * B + C)
        
        num_steps = min(200, max(50, int((time_limit - 1) * 10)))
        
        # Parametric on beta (multiplier for A, representing sum_b)
        betas = np.linspace(min_sum_b - abs(min_sum_b)*0.5, max_sum_b + abs(max_sum_b)*0.5, num_steps)
        # Also include some specific values
        for beta in betas:
            if time.time() - start_time > time_limit * 0.3:
                break
            cost_matrix = beta * A + C
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            update_best(col_ind.tolist())
        
        # Parametric on alpha (multiplier for B, representing sum_a)
        alphas = np.linspace(min_sum_a - abs(min_sum_a)*0.5, max_sum_a + abs(max_sum_a)*0.5, num_steps)
        for alpha in alphas:
            if time.time() - start_time > time_limit * 0.5:
                break
            cost_matrix = alpha * B + C
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            update_best(col_ind.tolist())
        
        # Iterative approach: alternate fixing sum_a and sum_b
        # Start from best known solution
        perm = np.array(best_assignment)
        idx = np.arange(n)
        for iteration in range(50):
            if time.time() - start_time > time_limit * 0.6:
                break
            sum_a = A[idx, perm].sum()
            sum_b = B[idx, perm].sum()
            
            # Fix sum_b, solve LAP on sum_b * A + C
            cost_matrix = sum_b * A + C
            _, col_ind = linear_sum_assignment(cost_matrix)
            obj1 = update_best(col_ind.tolist())
            
            # Fix sum_a from new solution
            new_sum_a = A[idx, col_ind].sum()
            cost_matrix = new_sum_a * B + C
            _, col_ind2 = linear_sum_assignment(cost_matrix)
            obj2 = update_best(col_ind2.tolist())
            
            new_perm = np.array(best_assignment)
            if np.array_equal(new_perm, perm):
                break
            perm = new_perm
        
        # More fine-grained parametric search around best found multipliers
        perm = np.array(best_assignment)
        sum_a_best = A[idx, perm].sum()
        sum_b_best = B[idx, perm].sum()
        
        # Fine search around sum_b_best
        fine_range_b = np.linspace(sum_b_best - abs(sum_b_best)*0.3 - 10, 
                                    sum_b_best + abs(sum_b_best)*0.3 + 10, 100)
        for beta in fine_range_b:
            if time.time() - start_time > time_limit * 0.7:
                break
            cost_matrix = beta * A + C
            _, col_ind = linear_sum_assignment(cost_matrix)
            update_best(col_ind.tolist())
        
        fine_range_a = np.linspace(sum_a_best - abs(sum_a_best)*0.3 - 10,
                                    sum_a_best + abs(sum_a_best)*0.3 + 10, 100)
        for alpha in fine_range_a:
            if time.time() - start_time > time_limit * 0.75:
                break
            cost_matrix = alpha * B + C
            _, col_ind = linear_sum_assignment(cost_matrix)
            update_best(col_ind.tolist())
        
        # Local search: 2-opt swaps
        def local_search_2opt(perm, deadline):
            perm = list(perm)
            improved = True
            while improved:
                improved = False
                for i in range(n):
                    if time.time() > deadline:
                        return perm
                    for j in range(i+1, n):
                        # Try swapping positions of entity i and entity j
                        new_perm = perm[:]
                        new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
                        new_obj = compute_objective(new_perm)
                        if new_obj < compute_objective(perm):
                            perm = new_perm
                            update_best(perm, new_obj)
                            improved = True
            return perm
        
        deadline = start_time + time_limit * 0.85
        perm = local_search_2opt(best_assignment, deadline)
        
        # Random restarts with local search
        deadline = start_time + time_limit - 0.5
        while time.time() < deadline:
            # Random perturbation of best
            perm = list(best_assignment)
            num_swaps = random.randint(2, max(2, n // 3))
            for _ in range(num_swaps):
                i, j = random.sample(range(n), 2)
                perm[i], perm[j] = perm[j], perm[i]
            
            # Also try random parametric
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
    
    # Write solution
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
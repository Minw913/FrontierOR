import argparse
import json
import time
import sys
import os

# Suppress D-Bus warnings in containerized environments
os.environ["DBUS_SESSION_BUS_ADDRESS"] = "/dev/null"

import numpy as np


def solve(instance_path, solution_path, time_limit, log_path):
    from solution_logger import SolutionLogger
    
    with open(instance_path, 'r') as f:
        data = json.load(f)
    
    start_time = time.time()
    
    n = data['num_facilities']
    m = data['num_customers']
    opening_costs = np.array(data['opening_costs'], dtype=np.float64)
    alloc_costs = np.array(data['allocation_costs'], dtype=np.float64)  # shape (n, m)
    
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    # For qUFL with quadratic costs c_ij * x_ij^2, given open set S,
    # optimal x_ij = (1/c_ij) / sum_{k in S} (1/c_kj)
    # and the allocation cost for customer j = 1 / sum_{k in S} (1/c_kj)
    
    inv_costs = 1.0 / alloc_costs  # shape (n, m)
    
    def compute_allocation(open_list):
        x = np.zeros((n, m), dtype=np.float64)
        if len(open_list) == 0:
            return x
        open_arr = np.array(open_list)
        inv_sum = inv_costs[open_arr].sum(axis=0)
        for i in open_arr:
            x[i, :] = inv_costs[i, :] / inv_sum
        return x
    
    def make_solution(open_set):
        open_list = sorted(open_set)
        y = [0] * n
        for i in open_list:
            y[i] = 1
        x = compute_allocation(open_list)
        actual_obj = float(sum(opening_costs[i] for i in open_list) + (alloc_costs * x * x).sum())
        sol = {
            "objective_value": actual_obj,
            "variant": "quadratic",
            "y": y,
            "x": x.tolist()
        }
        return sol, actual_obj
    
    best_obj = float('inf')
    best_open = set()
    
    # Evaluate each single facility
    single_objs = []
    for i in range(n):
        obj_i = opening_costs[i] + float(alloc_costs[i].sum())
        single_objs.append((obj_i, i))
        if obj_i < best_obj:
            best_obj = obj_i
            best_open = {i}
    
    sol, actual_obj = make_solution(best_open)
    best_obj = actual_obj
    if logger:
        logger.log_solution(best_obj, sol)
    
    # Greedy add construction
    single_objs.sort()
    
    current_open = set()
    current_inv_sum = np.zeros(m, dtype=np.float64)
    current_fixed_cost = 0.0
    
    remaining = set(range(n))
    
    while remaining and time.time() - start_time < time_limit * 0.10:
        best_marginal = float('inf')
        best_fac = -1
        
        for i in remaining:
            new_inv_sum = current_inv_sum + inv_costs[i]
            new_alloc_cost = float((1.0 / new_inv_sum).sum())
            new_fixed_cost = current_fixed_cost + opening_costs[i]
            total = new_fixed_cost + new_alloc_cost
            if total < best_marginal:
                best_marginal = total
                best_fac = i
        
        if len(current_open) > 0:
            current_total = current_fixed_cost + float((1.0 / current_inv_sum).sum())
            if best_marginal >= current_total:
                break
        
        current_open.add(best_fac)
        remaining.remove(best_fac)
        current_inv_sum += inv_costs[best_fac]
        current_fixed_cost += opening_costs[best_fac]
        total = current_fixed_cost + float((1.0 / current_inv_sum).sum())
        
        if total < best_obj:
            best_obj = total
            best_open = set(current_open)
            sol, actual_obj = make_solution(best_open)
            best_obj = actual_obj
            if logger:
                logger.log_solution(best_obj, sol)
    
    def local_search(open_set, deadline):
        improved = True
        current = set(open_set)
        
        if not current:
            return current, float('inf')
        
        cur_inv_sum = inv_costs[list(current)].sum(axis=0)
        cur_alloc = float((1.0 / cur_inv_sum).sum())
        cur_fixed = float(opening_costs[list(current)].sum())
        current_obj = cur_fixed + cur_alloc
        
        while improved and time.time() < deadline:
            improved = False
            
            # Try DROP
            if len(current) > 1:
                best_drop = None
                best_drop_obj = current_obj
                for i in list(current):
                    if time.time() >= deadline:
                        break
                    new_inv_sum = cur_inv_sum - inv_costs[i]
                    if np.any(new_inv_sum <= 1e-15):
                        continue
                    new_alloc = float((1.0 / new_inv_sum).sum())
                    new_fixed = cur_fixed - opening_costs[i]
                    new_obj = new_fixed + new_alloc
                    if new_obj < best_drop_obj - 1e-10:
                        best_drop_obj = new_obj
                        best_drop = i
                
                if best_drop is not None:
                    current.remove(best_drop)
                    cur_inv_sum = cur_inv_sum - inv_costs[best_drop]
                    cur_alloc = float((1.0 / cur_inv_sum).sum())
                    cur_fixed -= opening_costs[best_drop]
                    current_obj = best_drop_obj
                    improved = True
                    continue
            
            # Try ADD
            closed = set(range(n)) - current
            best_add = None
            best_add_obj = current_obj
            for i in closed:
                if time.time() >= deadline:
                    break
                new_inv_sum = cur_inv_sum + inv_costs[i]
                new_alloc = float((1.0 / new_inv_sum).sum())
                new_fixed = cur_fixed + opening_costs[i]
                new_obj = new_fixed + new_alloc
                if new_obj < best_add_obj - 1e-10:
                    best_add_obj = new_obj
                    best_add = i
            
            if best_add is not None:
                current.add(best_add)
                cur_inv_sum = cur_inv_sum + inv_costs[best_add]
                cur_alloc = float((1.0 / cur_inv_sum).sum())
                cur_fixed += opening_costs[best_add]
                current_obj = best_add_obj
                improved = True
                continue
            
            # Try SWAP
            best_swap = None
            best_swap_obj = current_obj
            open_list = list(current)
            closed_list = list(set(range(n)) - current)
            
            for i_out in open_list:
                if time.time() >= deadline:
                    break
                base_inv_sum = cur_inv_sum - inv_costs[i_out]
                base_fixed = cur_fixed - opening_costs[i_out]
                
                for i_in in closed_list:
                    if time.time() >= deadline:
                        break
                    new_inv_sum = base_inv_sum + inv_costs[i_in]
                    if np.any(new_inv_sum <= 1e-15):
                        continue
                    new_alloc = float((1.0 / new_inv_sum).sum())
                    new_fixed = base_fixed + opening_costs[i_in]
                    new_obj = new_fixed + new_alloc
                    if new_obj < best_swap_obj - 1e-10:
                        best_swap_obj = new_obj
                        best_swap = (i_out, i_in)
            
            if best_swap is not None:
                i_out, i_in = best_swap
                current.remove(i_out)
                current.add(i_in)
                cur_inv_sum = cur_inv_sum - inv_costs[i_out] + inv_costs[i_in]
                cur_alloc = float((1.0 / cur_inv_sum).sum())
                cur_fixed = cur_fixed - opening_costs[i_out] + opening_costs[i_in]
                current_obj = best_swap_obj
                improved = True
                continue
        
        return current, current_obj
    
    deadline = start_time + time_limit * 0.85
    
    ls_open, ls_obj = local_search(best_open, deadline)
    if ls_obj < best_obj:
        best_obj = ls_obj
        best_open = ls_open
        sol, actual_obj = make_solution(best_open)
        best_obj = actual_obj
        if logger:
            logger.log_solution(best_obj, sol)
    
    for obj_i, i in single_objs[:min(10, n)]:
        if time.time() >= deadline:
            break
        ls_open_i, ls_obj_i = local_search({i}, deadline)
        if ls_obj_i < best_obj:
            best_obj = ls_obj_i
            best_open = ls_open_i
            sol, actual_obj = make_solution(best_open)
            best_obj = actual_obj
            if logger:
                logger.log_solution(best_obj, sol)
    
    rng = np.random.RandomState(42)
    while time.time() < deadline:
        k = max(1, rng.randint(1, max(2, n // 2 + 1)))
        rand_open = set(rng.choice(n, size=k, replace=False).tolist())
        ls_open_r, ls_obj_r = local_search(rand_open, deadline)
        if ls_obj_r < best_obj:
            best_obj = ls_obj_r
            best_open = ls_open_r
            sol, actual_obj = make_solution(best_open)
            best_obj = actual_obj
            if logger:
                logger.log_solution(best_obj, sol)
    
    final_sol, final_obj = make_solution(best_open)
    
    if logger:
        logger.log_solution(final_obj, final_sol)
    
    with open(solution_path, 'w') as f:
        json.dump(final_sol, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()
    
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)
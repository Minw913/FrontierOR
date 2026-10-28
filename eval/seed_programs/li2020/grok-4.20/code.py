import json
import argparse
import time
import math
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    global_params = data['global_parameters']
    instance_params = data['instance_parameters']
    interd_params = data['interdiction_parameters']
    customers = data['customers']
    level1 = data['level1_facilities']
    level2 = data['level2_facilities']
    distances = data['distances']

    theta = global_params['theta']
    sigma = global_params['sigma']
    alpha = global_params['alpha']
    beta = global_params['beta']
    gamma = global_params['gamma']
    alpha_p = global_params['alpha_prime']
    beta_p = global_params['beta_prime']
    gamma_p = global_params['gamma_prime']

    N = instance_params['num_customers']
    M1 = instance_params['num_level1_facilities']
    M2 = instance_params['num_level2_facilities']
    K = instance_params['num_interdiction_levels_K']
    B = instance_params['total_interdiction_budget']

    # Precompute costs without numpy
    c1 = [[0.0 for _ in range(M1)] for _ in range(N)]
    for i in range(N):
        for j in range(M1):
            c1[i][j] = alpha * distances['customer_to_level1_facility'][i][j]

    c2 = [[0.0 for _ in range(M2)] for _ in range(N)]
    for i in range(N):
        for k_idx in range(M2):
            c2[i][k_idx] = beta * distances['customer_to_level2_facility'][i][k_idx]

    c_ref = [[0.0 for _ in range(M2)] for _ in range(M1)]
    for j in range(M1):
        for k_idx in range(M2):
            c_ref[j][k_idx] = gamma * distances['level1_to_level2_facility'][j][k_idx]

    d1 = [interd_params['capacity_reduction_level1_d1'].get(str(k), 0.0) for k in range(K)]
    d2 = [interd_params['capacity_reduction_level2_d2'].get(str(k), 0.0) for k in range(K)]
    cost1 = [interd_params['interdiction_cost_level1_h1'].get(str(k), 0.0) for k in range(K)]
    cost2 = [interd_params['interdiction_cost_level2_h2'].get(str(k), 0.0) for k in range(K)]

    cap1 = level1['capacity_type1']
    cap2_type1 = level2['capacity_type1']
    cap2_type2 = level2['capacity_type2']
    dem = customers['demands']
    dem1 = [theta * dem[i] for i in range(N)]
    dem2 = [(1.0 - theta) * dem[i] for i in range(N)]

    start_time = time.time()
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    # Calculate facility importance scores without numpy
    importance1 = [0.0] * M1
    importance2 = [0.0] * M2
    
    for j in range(M1):
        cap_importance = cap1[j]
        avg_dist = 0.0
        for i in range(N):
            avg_dist += distances['customer_to_level1_facility'][i][j]
        avg_dist /= N
        importance1[j] = cap_importance / (avg_dist + 1.0)
    
    for m in range(M2):
        cap_importance = cap2_type1[m] + cap2_type2[m]
        avg_dist = 0.0
        for i in range(N):
            avg_dist += distances['customer_to_level2_facility'][i][m]
        avg_dist /= N
        importance2[m] = cap_importance / (avg_dist + 1.0)
    
    # Sort by importance (descending) - simple insertion sort
    sorted_level1 = list(range(M1))
    for i in range(1, M1):
        key = sorted_level1[i]
        key_val = importance1[key]
        j = i - 1
        while j >= 0 and importance1[sorted_level1[j]] < key_val:
            sorted_level1[j + 1] = sorted_level1[j]
            j -= 1
        sorted_level1[j + 1] = key
    
    sorted_level2 = list(range(M2))
    for i in range(1, M2):
        key = sorted_level2[i]
        key_val = importance2[key]
        j = i - 1
        while j >= 0 and importance2[sorted_level2[j]] < key_val:
            sorted_level2[j + 1] = sorted_level2[j]
            j -= 1
        sorted_level2[j + 1] = key
    
    remaining_budget = B
    interdiction_level1 = [0] * M1
    interdiction_level2 = [0] * M2
    
    # Assign to level 2 first (higher impact)
    for idx in range(M2):
        m = sorted_level2[idx]
        for k in range(K-1, 0, -1):
            if cost2[k] <= remaining_budget + 1e-6:
                interdiction_level2[m] = k
                remaining_budget -= cost2[k]
                break
    
    for idx in range(M1):
        j = sorted_level1[idx]
        for k in range(K-1, 0, -1):
            if cost1[k] <= remaining_budget + 1e-6:
                interdiction_level1[j] = k
                remaining_budget -= cost1[k]
                break
    
    # Build pattern
    pattern = {}
    for j in range(M1):
        pattern[f"S1_{j}"] = interdiction_level1[j]
    for m in range(M2):
        pattern[f"S2_{m}"] = interdiction_level2[m]
    
    # Evaluate defender cost
    obj_value = evaluate_defender(pattern, data, global_params, instance_params, distances,
                                 d1, d2, c1, c2, c_ref, dem1, dem2, cap1, cap2_type1, cap2_type2, 
                                 N, M1, M2, K, alpha_p, beta_p, gamma_p, sigma)
    
    if logger:
        logger.log(float(obj_value))
    
    # Limited local search if time allows
    elapsed = time.time() - start_time
    if elapsed < args.time_limit * 0.6:
        improved = True
        iteration = 0
        max_iter = 3
        while improved and iteration < max_iter and (time.time() - start_time) < args.time_limit * 0.85:
            iteration += 1
            improved = False
            
            for fac_type in [0, 1]:  # 0 = S1, 1 = S2
                if fac_type == 0:
                    facilities_count = M1
                    costs = cost1
                    levels = interdiction_level1
                    prefix = "S1"
                else:
                    facilities_count = M2
                    costs = cost2
                    levels = interdiction_level2
                    prefix = "S2"
                
                for i in range(facilities_count):
                    current_level = levels[i]
                    if current_level > 0:
                        saved_budget = costs[current_level]
                        for other_i in range(facilities_count):
                            if other_i == i:
                                continue
                            other_level = levels[other_i]
                            if other_level < K - 1:
                                new_cost = costs[other_level + 1]
                                net_cost = new_cost - saved_budget
                                if net_cost <= remaining_budget + 1e-6:
                                    # Create test pattern
                                    test_pattern = pattern.copy()
                                    test_pattern[f"{prefix}_{i}"] = 0
                                    test_pattern[f"{prefix}_{other_i}"] = other_level + 1
                                    
                                    test_obj = evaluate_defender(test_pattern, data, global_params, 
                                        instance_params, distances, d1, d2, c1, c2, c_ref, dem1, dem2,
                                        cap1, cap2_type1, cap2_type2, N, M1, M2, K, alpha_p, beta_p, 
                                        gamma_p, sigma)
                                    
                                    if test_obj > obj_value + 1e-4:
                                        obj_value = test_obj
                                        pattern = test_pattern
                                        levels[i] = 0
                                        levels[other_i] = other_level + 1
                                        remaining_budget += saved_budget - new_cost
                                        improved = True
                                        if logger:
                                            logger.log(float(obj_value))
                                        break
                        if improved:
                            break
                if improved:
                    break
    
    solution = {
        "objective_value": float(obj_value),
        "interdiction_pattern": pattern
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


def evaluate_defender(pattern, data, global_params, instance_params, distances,
                     d1, d2, c1, c2, c_ref, dem1, dem2, cap1, cap2_type1, cap2_type2,
                     N, M1, M2, K, alpha_p, beta_p, gamma_p, sigma):
    """Evaluate defender's optimal response using greedy allocation"""
    rem_cap1 = [0.0] * M1
    for j in range(M1):
        k = pattern.get(f"S1_{j}", 0)
        rem_cap1[j] = cap1[j] * (1.0 - d1[k])
    
    rem_cap2_t1 = [0.0] * M2
    rem_cap2_t2 = [0.0] * M2
    for m in range(M2):
        k = pattern.get(f"S2_{m}", 0)
        red = d2[k]
        rem_cap2_t1[m] = cap2_type1[m] * (1.0 - red)
        rem_cap2_t2[m] = cap2_type2[m] * (1.0 - red)
    
    total_cost = 0.0
    remaining_dem1 = [dem1[i] for i in range(N)]
    remaining_dem2 = [dem2[i] for i in range(N)]
    
    # Type I to level 1 (cheapest usually)
    for i in range(N):
        for j in range(M1):
            if remaining_dem1[i] > 1e-8 and rem_cap1[j] > 1e-8 and c1[i][j] < alpha_p:
                can_assign = min(remaining_dem1[i], rem_cap1[j])
                assigned = can_assign
                total_cost += c1[i][j] * assigned
                remaining_dem1[i] -= assigned
                rem_cap1[j] -= assigned
    
    # Remaining type I to level 2 or outsourcing
    for i in range(N):
        if remaining_dem1[i] > 1e-8:
            best_cost = alpha_p
            best_k = -1
            for k in range(M2):
                if rem_cap2_t1[k] > 1e-8 and c2[i][k] < best_cost:
                    best_cost = c2[i][k]
                    best_k = k
            if best_k >= 0:
                can_assign = min(remaining_dem1[i], rem_cap2_t1[best_k])
                total_cost += best_cost * can_assign
                remaining_dem1[i] -= can_assign
                rem_cap2_t1[best_k] -= can_assign
            if remaining_dem1[i] > 1e-8:
                total_cost += alpha_p * remaining_dem1[i]
                total_cost += sigma * gamma_p * remaining_dem1[i]
    
    # Type II demand
    for i in range(N):
        if remaining_dem2[i] > 1e-8:
            best_cost = beta_p
            best_k = -1
            for k_idx in range(M2):
                if rem_cap2_t2[k_idx] > 1e-8 and c2[i][k_idx] < best_cost:
                    best_cost = c2[i][k_idx]
                    best_k = k_idx
            if best_k >= 0:
                can_assign = min(remaining_dem2[i], rem_cap2_t2[best_k])
                total_cost += best_cost * can_assign
                remaining_dem2[i] -= can_assign
                rem_cap2_t2[best_k] -= can_assign
            if remaining_dem2[i] > 1e-8:
                total_cost += beta_p * remaining_dem2[i]
    
    # Referral from level 1
    ref_from_l1 = [0.0] * M1
    for j in range(M1):
        served = 0.0
        for i in range(N):
            # Approximate served by level1 - we use a simplified remaining cap approach
            served += max(0.0, cap1[j] * (1.0 - d1[pattern.get(f"S1_{j}", 0)]) - rem_cap1[j])
        ref_from_l1[j] = sigma * served
    
    for j in range(M1):
        if ref_from_l1[j] > 1e-8:
            best_cost = gamma_p
            best_k = -1
            for k in range(M2):
                if rem_cap2_t2[k] > 1e-8 and c_ref[j][k] < best_cost:
                    best_cost = c_ref[j][k]
                    best_k = k
            if best_k >= 0:
                can_assign = min(ref_from_l1[j], rem_cap2_t2[best_k])
                total_cost += best_cost * can_assign
                ref_from_l1[j] -= can_assign
                rem_cap2_t2[best_k] -= can_assign
            if ref_from_l1[j] > 1e-8:
                total_cost += gamma_p * ref_from_l1[j]
    
    return total_cost


if __name__ == "__main__":
    main()
import argparse
import json
import time
import itertools
import numpy as np
from solution_logger import SolutionLogger


def solve(instance_path, solution_path, time_limit, log_path=None):
    start_time = time.time()
    
    with open(instance_path, 'r') as f:
        data = json.load(f)
    
    logger = SolutionLogger(log_path, sense="maximize") if log_path else None
    
    n = data['n_nodes']
    p = data['p']
    r = data['r']
    nodes = data['nodes']
    demand = np.array(data['demand_matrix'], dtype=np.float64)
    dist = np.array(data['distance_matrix'], dtype=np.float64)
    comp_hubs = data['competitor_hub_locations']
    params = data['parameters']
    
    alpha = params['alpha']
    gamma = params['gamma']
    beta = params['beta']
    lam = params['lambda']
    cost_scalar = params['cost_scalar']
    attr_2hub = params['attractiveness_2hub']
    attr_1hub = params['attractiveness_1hub']
    
    waiting = np.array([nodes[i]['waiting_time'] for i in range(n)], dtype=np.float64)
    
    # Precompute travel times and fares for legs
    # travel_time from a to b = dist[a][b] (proportional to distance)
    # fare from a to b = cost_scalar * dist[a][b]
    
    # For a route i -> k -> l -> j:
    # total_time = dist[i][k] + dist[k][l] + dist[l][j] + waiting[i] + waiting[k] + waiting[l]
    # total_fare = cost_scalar * (dist[i][k] + alpha * dist[k][l] + dist[l][j])
    
    # Precompute competitor utilities for each OD pair
    # comp_utilities[i][j] = sum of utilities over all competitor hub pairs
    
    def compute_utility(attractiveness, total_time, total_fare):
        denom = gamma * (total_time ** beta) + (1.0 - gamma) * (total_fare ** lam)
        if denom <= 0:
            return 0.0
        return attractiveness / denom
    
    # Precompute all competitor route utilities per OD pair
    # For competitor, hub pairs are from comp_hubs
    # For each OD (i,j), for each hub pair (k,l) where k,l in comp_hubs:
    #   if k == l: single hub route, attractiveness = attr_1hub
    #   if k != l: two distinct hubs, only one ordering considered (the better one)
    
    # Vectorized precomputation of competitor utilities
    comp_hub_set = comp_hubs
    n_comp = len(comp_hub_set)
    
    # Precompute competitor utility sum for each OD pair
    comp_utility_sum = np.zeros((n, n), dtype=np.float64)
    
    for ki_idx in range(n_comp):
        k = comp_hub_set[ki_idx]
        # Single hub routes (k == l)
        # total_time = dist[i][k] + 0 + dist[k][j] + waiting[i] + waiting[k] + waiting[k]
        # total_fare = cost_scalar * (dist[i][k] + 0 + dist[k][j])
        for i in range(n):
            for j in range(n):
                if demand[i][j] <= 0:
                    continue
                tt = dist[i][k] + dist[k][j] + waiting[i] + waiting[k] + waiting[k]
                tf = cost_scalar * (dist[i][k] + dist[k][j])
                if tt <= 0 or tf <= 0:
                    continue
                u = attr_1hub / (gamma * (tt ** beta) + (1.0 - gamma) * (tf ** lam))
                comp_utility_sum[i][j] += u
        
        # Two hub routes (k, l) where l > k to avoid double counting
        for li_idx in range(ki_idx + 1, n_comp):
            l = comp_hub_set[li_idx]
            for i in range(n):
                for j in range(n):
                    if demand[i][j] <= 0:
                        continue
                    # Route i->k->l->j
                    tt1 = dist[i][k] + dist[k][l] + dist[l][j] + waiting[i] + waiting[k] + waiting[l]
                    tf1 = cost_scalar * (dist[i][k] + alpha * dist[k][l] + dist[l][j])
                    # Route i->l->k->j
                    tt2 = dist[i][l] + dist[l][k] + dist[k][j] + waiting[i] + waiting[l] + waiting[k]
                    tf2 = cost_scalar * (dist[i][l] + alpha * dist[l][k] + dist[k][j])
                    
                    if tt1 > 0 and tf1 > 0:
                        u1 = attr_2hub / (gamma * (tt1 ** beta) + (1.0 - gamma) * (tf1 ** lam))
                    else:
                        u1 = 0.0
                    if tt2 > 0 and tf2 > 0:
                        u2 = attr_2hub / (gamma * (tt2 ** beta) + (1.0 - gamma) * (tf2 ** lam))
                    else:
                        u2 = 0.0
                    
                    comp_utility_sum[i][j] += max(u1, u2)
    
    # Now we need to select p hubs to maximize captured demand
    # For each candidate hub set S, entrant utility for OD (i,j) is sum of utilities
    # over all hub pairs (k,l) in S x S (with the ordering rule for distinct pairs)
    
    # Precompute entrant utility for each OD pair and each possible hub pair
    # Hub pair (k, k): single hub
    # Hub pair {k, l} with k < l: best of two orderings
    
    # For efficiency, precompute utility contribution of each hub pair for each OD
    # single_hub_util[k][i][j] = utility of single hub k for OD (i,j)
    # two_hub_util[k][l][i][j] = max utility of hub pair {k,l} for OD (i,j), k < l
    
    # Find OD pairs with positive demand
    od_pairs = []
    for i in range(n):
        for j in range(n):
            if demand[i][j] > 0:
                od_pairs.append((i, j))
    
    n_od = len(od_pairs)
    od_i = np.array([p[0] for p in od_pairs])
    od_j = np.array([p[1] for p in od_pairs])
    od_demand = demand[od_i, od_j]
    od_comp_util = comp_utility_sum[od_i, od_j]
    
    # Precompute single hub utilities for each hub k and each OD pair
    # Shape: [n, n_od]
    single_hub_utils = np.zeros((n, n_od), dtype=np.float64)
    
    for k in range(n):
        tt = dist[od_i, k] + dist[k, od_j] + waiting[od_i] + waiting[k] + waiting[k]
        tf = cost_scalar * (dist[od_i, k] + dist[k, od_j])
        valid = (tt > 0) & (tf > 0)
        denom = np.where(valid, gamma * np.power(tt, beta, where=valid) + (1.0 - gamma) * np.power(tf, lam, where=valid), 1.0)
        single_hub_utils[k] = np.where(valid, attr_1hub / denom, 0.0)
    
    # Precompute two-hub utilities for each pair (k, l), k < l, and each OD pair
    # Store as dictionary or sparse structure
    # For n up to ~25, n*(n-1)/2 pairs, each with n_od values
    
    two_hub_utils = {}  # (k, l) -> array of shape [n_od]
    
    for k in range(n):
        for l in range(k + 1, n):
            # Route i->k->l->j
            tt1 = dist[od_i, k] + dist[k, l] + dist[l, od_j] + waiting[od_i] + waiting[k] + waiting[l]
            tf1 = cost_scalar * (dist[od_i, k] + alpha * dist[k, l] + dist[l, od_j])
            valid1 = (tt1 > 0) & (tf1 > 0)
            denom1 = np.where(valid1, gamma * np.power(tt1, beta, where=valid1) + (1.0 - gamma) * np.power(tf1, lam, where=valid1), 1.0)
            u1 = np.where(valid1, attr_2hub / denom1, 0.0)
            
            # Route i->l->k->j
            tt2 = dist[od_i, l] + dist[l, k] + dist[k, od_j] + waiting[od_i] + waiting[l] + waiting[k]
            tf2 = cost_scalar * (dist[od_i, l] + alpha * dist[l, k] + dist[k, od_j])
            valid2 = (tt2 > 0) & (tf2 > 0)
            denom2 = np.where(valid2, gamma * np.power(tt2, beta, where=valid2) + (1.0 - gamma) * np.power(tf2, lam, where=valid2), 1.0)
            u2 = np.where(valid2, attr_2hub / denom2, 0.0)
            
            two_hub_utils[(k, l)] = np.maximum(u1, u2)
    
    # Now, for a given hub set S, the entrant utility for each OD is:
    # sum_{k in S} single_hub_utils[k] + sum_{k<l in S} two_hub_utils[(k,l)]
    # And captured demand = sum over OD of demand[i][j] * entrant_util / (entrant_util + comp_util)
    
    def evaluate_hub_set(hubs):
        hubs_sorted = sorted(hubs)
        # Sum single hub utilities
        entrant_util = np.zeros(n_od, dtype=np.float64)
        for k in hubs_sorted:
            entrant_util += single_hub_utils[k]
        # Sum two-hub utilities
        for idx1 in range(len(hubs_sorted)):
            for idx2 in range(idx1 + 1, len(hubs_sorted)):
                k, l = hubs_sorted[idx1], hubs_sorted[idx2]
                entrant_util += two_hub_utils[(k, l)]
        
        total_util = entrant_util + od_comp_util
        # Avoid division by zero
        valid = total_util > 0
        capture = np.where(valid, od_demand * entrant_util / total_util, 0.0)
        return np.sum(capture)
    
    # Strategy: If n is small enough, enumerate all combinations
    # C(n, p) combinations
    from math import comb
    total_combos = comb(n, p)
    
    best_obj = -1.0
    best_hubs = None
    
    elapsed = time.time() - start_time
    remaining = time_limit - elapsed
    
    if total_combos <= 500000 and remaining > 5:
        # Enumerate all
        count = 0
        for combo in itertools.combinations(range(n), p):
            if time.time() - start_time > time_limit - 1:
                break
            obj = evaluate_hub_set(combo)
            if obj > best_obj:
                best_obj = obj
                best_hubs = list(combo)
                if logger:
                    logger.log(best_obj)
            count += 1
    else:
        # Use greedy + local search approach
        # Greedy construction
        # Start with the hub that has the highest individual contribution
        individual_scores = []
        for k in range(n):
            score = evaluate_hub_set([k])
            individual_scores.append((score, k))
        individual_scores.sort(reverse=True)
        
        # Greedy: add hubs one by one
        selected = []
        remaining_nodes = set(range(n))
        
        for _ in range(p):
            best_add_obj = -1
            best_add_node = -1
            for node in remaining_nodes:
                candidate = selected + [node]
                obj = evaluate_hub_set(candidate)
                if obj > best_add_obj:
                    best_add_obj = obj
                    best_add_node = node
            selected.append(best_add_node)
            remaining_nodes.remove(best_add_node)
        
        best_obj = evaluate_hub_set(selected)
        best_hubs = list(selected)
        if logger:
            logger.log(best_obj)
        
        # Also try top-p by individual score
        top_p = [x[1] for x in individual_scores[:p]]
        obj_top = evaluate_hub_set(top_p)
        if obj_top > best_obj:
            best_obj = obj_top
            best_hubs = top_p
            if logger:
                logger.log(best_obj)
        
        # Local search: swap one hub in, one hub out
        improved = True
        while improved and (time.time() - start_time < time_limit - 2):
            improved = False
            current_hubs = set(best_hubs)
            non_hubs = set(range(n)) - current_hubs
            
            for hub_out in list(current_hubs):
                if time.time() - start_time > time_limit - 2:
                    break
                for hub_in in non_hubs:
                    candidate = (current_hubs - {hub_out}) | {hub_in}
                    obj = evaluate_hub_set(candidate)
                    if obj > best_obj:
                        best_obj = obj
                        best_hubs = sorted(candidate)
                        improved = True
                        if logger:
                            logger.log(best_obj)
                        break
                if improved:
                    break
        
        # Try multiple random restarts if time permits
        import random
        random.seed(42)
        while time.time() - start_time < time_limit - 3:
            # Random initial solution
            rand_hubs = random.sample(range(n), p)
            obj = evaluate_hub_set(rand_hubs)
            
            # Local search from this random start
            local_improved = True
            while local_improved and (time.time() - start_time < time_limit - 2):
                local_improved = False
                current = set(rand_hubs)
                non_h = set(range(n)) - current
                
                best_swap_obj = obj
                best_swap = None
                
                for h_out in list(current):
                    if time.time() - start_time > time_limit - 2:
                        break
                    for h_in in non_h:
                        cand = (current - {h_out}) | {h_in}
                        o = evaluate_hub_set(cand)
                        if o > best_swap_obj:
                            best_swap_obj = o
                            best_swap = (h_out, h_in)
                
                if best_swap is not None:
                    h_out, h_in = best_swap
                    current = (current - {h_out}) | {h_in}
                    rand_hubs = sorted(current)
                    obj = best_swap_obj
                    local_improved = True
            
            if obj > best_obj:
                best_obj = obj
                best_hubs = sorted(rand_hubs) if isinstance(rand_hubs, list) else sorted(rand_hubs)
                if logger:
                    logger.log(best_obj)
    
    solution = {
        "objective_value": float(best_obj),
        "hub_locations": list(best_hubs)
    }
    
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)
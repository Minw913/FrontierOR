import json
import random
import time
import math
import argparse
from solution_logger import SolutionLogger

def compute_U1(i, j, k, distance_matrix, waiting_times, params):
    gamma = params['gamma']
    beta_val = params['beta']
    lambda_val = params['lambda']
    cost_scalar = params['cost_scalar']
    attractiveness_1hub = params['attractiveness_1hub']
    d_ik = distance_matrix[i][k]
    d_kj = distance_matrix[k][j]
    total_time = d_ik + d_kj + waiting_times[i] + 2 * waiting_times[k]
    total_fare = cost_scalar * (d_ik + d_kj)
    denom = gamma * (total_time ** beta_val) + (1 - gamma) * (total_fare ** lambda_val)
    if denom <= 1e-10:
        return 0.0
    utility = attractiveness_1hub / denom
    return utility

def compute_U2(i, j, k, l, distance_matrix, waiting_times, params):
    gamma = params['gamma']
    beta_val = params['beta']
    lambda_val = params['lambda']
    cost_scalar = params['cost_scalar']
    alpha = params['alpha']
    attractiveness_2hub = params['attractiveness_2hub']
    
    # Route1: i->k->l->j
    time1 = distance_matrix[i][k] + distance_matrix[k][l] + distance_matrix[l][j] + waiting_times[i] + waiting_times[k] + waiting_times[l]
    fare1 = cost_scalar * (distance_matrix[i][k] + alpha * distance_matrix[k][l] + distance_matrix[l][j])
    denom1 = gamma * (time1 ** beta_val) + (1 - gamma) * (fare1 ** lambda_val)
    if denom1 <= 1e-10:
        util1 = 0.0
    else:
        util1 = attractiveness_2hub / denom1

    # Route2: i->l->k->j
    time2 = distance_matrix[i][l] + distance_matrix[l][k] + distance_matrix[k][j] + waiting_times[i] + waiting_times[l] + waiting_times[k]
    fare2 = cost_scalar * (distance_matrix[i][l] + alpha * distance_matrix[l][k] + distance_matrix[k][j])
    denom2 = gamma * (time2 ** beta_val) + (1 - gamma) * (fare2 ** lambda_val)
    if denom2 <= 1e-10:
        util2 = 0.0
    else:
        util2 = attractiveness_2hub / denom2

    return max(util1, util2)

def compute_full_objective(H, demand, V, distance_matrix, waiting_times, params, n_nodes):
    U_entrant = [[0.0] * n_nodes for _ in range(n_nodes)]
    for i in range(n_nodes):
        for j in range(n_nodes):
            for k in H:
                U_entrant[i][j] += compute_U1(i, j, k, distance_matrix, waiting_times, params)
            for idx1 in range(len(H)):
                k = H[idx1]
                for idx2 in range(idx1 + 1, len(H)):
                    l = H[idx2]
                    U_entrant[i][j] += compute_U2(i, j, k, l, distance_matrix, waiting_times, params)
    obj = 0.0
    for i in range(n_nodes):
        for j in range(n_nodes):
            total_utility = U_entrant[i][j] + V[i][j]
            if total_utility <= 1e-10:
                term = 0.0
            else:
                term = demand[i][j] * (U_entrant[i][j] / total_utility)
            obj += term
    return obj, U_entrant

def compute_S1_for_hub(h, H, U1, n_nodes, distance_matrix, waiting_times, params):
    S1_h = [[0.0] * n_nodes for _ in range(n_nodes)]
    for i in range(n_nodes):
        for j in range(n_nodes):
            val = U1[i][j][h]
            for k in H:
                if k == h:
                    continue
                val += compute_U2(i, j, h, k, distance_matrix, waiting_times, params)
            S1_h[i][j] = val
    return S1_h

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    n_nodes = instance['n_nodes']
    p = instance['p']
    nodes = instance['nodes']
    demand = instance['demand_matrix']
    distance_matrix = instance['distance_matrix']
    competitor_hubs = instance['competitor_hub_locations']
    params = instance['parameters']

    gamma = params['gamma']
    beta_val = params['beta']
    lambda_val = params['lambda']
    cost_scalar = params['cost_scalar']
    alpha = params['alpha']
    attractiveness_2hub = params['attractiveness_2hub']
    attractiveness_1hub = params['attractiveness_1hub']

    waiting_times = [node['waiting_time'] for node in nodes]

    V = [[0.0] * n_nodes for _ in range(n_nodes)]
    for i in range(n_nodes):
        for j in range(n_nodes):
            for k in competitor_hubs:
                d_ik = distance_matrix[i][k]
                d_kj = distance_matrix[k][j]
                total_time = d_ik + d_kj + waiting_times[i] + 2 * waiting_times[k]
                total_fare = cost_scalar * (d_ik + d_kj)
                denom = gamma * (total_time ** beta_val) + (1 - gamma) * (total_fare ** lambda_val)
                if denom <= 1e-10:
                    utility = 0.0
                else:
                    utility = attractiveness_1hub / denom
                V[i][j] += utility
            for idx1 in range(len(competitor_hubs)):
                k = competitor_hubs[idx1]
                for idx2 in range(idx1 + 1, len(competitor_hubs)):
                    l = competitor_hubs[idx2]
                    time1 = distance_matrix[i][k] + distance_matrix[k][l] + distance_matrix[l][j] + waiting_times[i] + waiting_times[k] + waiting_times[l]
                    fare1 = cost_scalar * (distance_matrix[i][k] + alpha * distance_matrix[k][l] + distance_matrix[l][j])
                    denom1 = gamma * (time1 ** beta_val) + (1 - gamma) * (fare1 ** lambda_val)
                    util1 = attractiveness_2hub / denom1 if denom1 > 1e-10 else 0.0
                    time2 = distance_matrix[i][l] + distance_matrix[l][k] + distance_matrix[k][j] + waiting_times[i] + waiting_times[l] + waiting_times[k]
                    fare2 = cost_scalar * (distance_matrix[i][l] + alpha * distance_matrix[l][k] + distance_matrix[k][j])
                    denom2 = gamma * (time2 ** beta_val) + (1 - gamma) * (fare2 ** lambda_val)
                    util2 = attractiveness_2hub / denom2 if denom2 > 1e-10 else 0.0
                    V[i][j] += max(util1, util2)

    U1 = [[[0.0] * n_nodes for _ in range(n_nodes)] for __ in range(n_nodes)]
    for i in range(n_nodes):
        for j in range(n_nodes):
            for k in range(n_nodes):
                d_ik = distance_matrix[i][k]
                d_kj = distance_matrix[k][j]
                total_time = d_ik + d_kj + waiting_times[i] + 2 * waiting_times[k]
                total_fare = cost_scalar * (d_ik + d_kj)
                denom = gamma * (total_time ** beta_val) + (1 - gamma) * (total_fare ** lambda_val)
                if denom <= 1e-10:
                    utility = 0.0
                else:
                    utility = attractiveness_1hub / denom
                U1[i][j][k] = utility

    all_nodes = list(range(n_nodes))
    current_set = set(random.sample(all_nodes, p))
    current_obj, current_U_entrant = compute_full_objective(list(current_set), demand, V, distance_matrix, waiting_times, params, n_nodes)
    best_solution = list(current_set)
    best_objective = current_obj
    if logger:
        logger.log(best_objective)

    S1 = {}
    for h in current_set:
        S1_h = compute_S1_for_hub(h, current_set, U1, n_nodes, distance_matrix, waiting_times, params)
        S1[h] = S1_h

    start_time = time.time()

    if time.time() - start_time >= args.time_limit:
        with open(args.solution_path, 'w') as f:
            json.dump({"objective_value": best_objective, "hub_locations": best_solution}, f)
        return

    improved = True
    while improved and (time.time() - start_time < args.time_limit):
        improved = False
        candidate_swaps = []
        for h in current_set:
            for a in all_nodes:
                if a not in current_set:
                    candidate_swaps.append((h, a))
        random.shuffle(candidate_swaps)
        
        for swap in candidate_swaps:
            if time.time() - start_time >= args.time_limit:
                break
            h, a = swap
            new_U_entrant = [[0.0] * n_nodes for _ in range(n_nodes)]
            new_obj = 0.0
            for i in range(n_nodes):
                if time.time() - start_time >= args.time_limit:
                    break
                for j in range(n_nodes):
                    T_ij = U1[i][j][a]
                    for k in current_set:
                        if k == h:
                            continue
                        T_ij += compute_U2(i, j, a, k, distance_matrix, waiting_times, params)
                    new_U_entrant_ij_val = current_U_entrant[i][j] - S1[h][i][j] + T_ij
                    new_U_entrant[i][j] = new_U_entrant_ij_val
                    total_utility_ij = new_U_entrant_ij_val + V[i][j]
                    if total_utility_ij <= 1e-10:
                        term = 0.0
                    else:
                        term = demand[i][j] * (new_U_entrant_ij_val / total_utility_ij)
                    new_obj += term
            if new_obj > current_obj:
                current_set.discard(h)
                current_set.add(a)
                current_U_entrant = new_U_entrant
                current_obj = new_obj
                if current_obj > best_objective:
                    best_objective = current_obj
                    best_solution = list(current_set)
                    if logger:
                        logger.log(best_objective)
                del S1[h]
                for x in current_set:
                    if x == a:
                        continue
                    for i in range(n_nodes):
                        for j in range(n_nodes):
                            S1[x][i][j] = S1[x][i][j] - compute_U2(i, j, x, h, distance_matrix, waiting_times, params) + compute_U2(i, j, x, a, distance_matrix, waiting_times, params)
                S1_a = compute_S1_for_hub(a, current_set, U1, n_nodes, distance_matrix, waiting_times, params)
                S1[a] = S1_a
                improved = True
                break
        else:
            continue
        break

    with open(args.solution_path, 'w') as f:
        json.dump({"objective_value": best_objective, "hub_locations": best_solution}, f)

if __name__ == "__main__":
    main()
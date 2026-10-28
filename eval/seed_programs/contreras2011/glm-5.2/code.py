import argparse
import json
import time
import math
import random
from solution_logger import SolutionLogger


def compute_objective(n, flow, dist, outgoing, incoming, setup, chi, alpha, delta, assignment, hubs):
    cost = sum(setup[k] for k in hubs)
    for i in range(n):
        k = assignment[i]
        cost += chi * outgoing[i] * dist[i][k]
        cost += delta * incoming[i] * dist[k][i]
    for i in range(n):
        for j in range(n):
            cost += alpha * flow[i][j] * dist[assignment[i]][assignment[j]]
    return float(cost)


def solve_sa(n, flow, dist, outgoing, incoming, setup, cap, chi, alpha, delta, D, time_limit, logger):
    start_time = time.time()
    
    is_hub = [False] * n
    assignment = [-1] * n
    hub_load = [0.0] * n
    
    candidates = sorted(range(n), key=lambda k: setup[k] + sum(dist[j][k] * (outgoing[j] + incoming[j]) for j in range(n)) * (chi + delta) / (2 * n))
    
    total_cap = 0.0
    for k in candidates:
        is_hub[k] = True
        total_cap += cap[k]
        if total_cap >= D:
            break
            
    hubs = [k for k in range(n) if is_hub[k]]
    if not hubs:
        is_hub[candidates[0]] = True
        hubs = [candidates[0]]
    
    for k in hubs:
        assignment[k] = k
        hub_load[k] += outgoing[k]
        
    for i in range(n):
        if assignment[i] != -1:
            continue
        best_k = -1
        best_cost = float('inf')
        for k in hubs:
            if hub_load[k] + outgoing[i] <= cap[k] + 1e-6:
                cost = chi * outgoing[i] * dist[i][k] + delta * incoming[i] * dist[k][i]
                if cost < best_cost:
                    best_cost = cost
                    best_k = k
        if best_k == -1:
            best_k = min(hubs, key=lambda k: dist[i][k])
        assignment[i] = best_k
        hub_load[best_k] += outgoing[i]
        
    obj = compute_objective(n, flow, dist, outgoing, incoming, setup, chi, alpha, delta, assignment, hubs)
    best_obj = obj
    best_assignment = list(assignment)
    best_hubs = list(hubs)
    
    if logger:
        logger.log_solution(best_obj, {
            'objective_value': best_obj,
            'hubs': best_hubs,
            'assignment': best_assignment
        })

    rng = random.Random(0)
    iterations = 0
    
    def get_delta_reassign(i, k_new):
        k_old = assignment[i]
        if k_old == k_new:
            return 0.0, True
        if hub_load[k_new] + outgoing[i] > cap[k_new] + 1e-6:
            return float('inf'), False
        
        d = chi * outgoing[i] * (dist[i][k_new] - dist[i][k_old])
        d += delta * incoming[i] * (dist[k_new][i] - dist[k_old][i])
        
        for j in range(n):
            if j == i: continue
            m_j = assignment[j]
            diff = dist[k_new][m_j] - dist[k_old][m_j]
            if diff != 0:
                d += alpha * (flow[i][j] + flow[j][i]) * diff
            
        d += alpha * flow[i][i] * (-2.0 * dist[k_old][k_new])
        return d, True

    def get_delta_open_and_reassign(i, k_new):
        if is_hub[k_new]:
            return get_delta_reassign(i, k_new)
        if outgoing[i] > cap[k_new] + 1e-6:
            return float('inf'), False
        
        d = setup[k_new]
        k_old = assignment[i]
        d += chi * outgoing[i] * (dist[i][k_new] - dist[i][k_old])
        d += delta * incoming[i] * (dist[k_new][i] - dist[k_old][i])
        
        for j in range(n):
            if j == i: continue
            m_j = assignment[j]
            diff = dist[k_new][m_j] - dist[k_old][m_j]
            if diff != 0:
                d += alpha * (flow[i][j] + flow[j][i]) * diff
            
        d += alpha * flow[i][i] * (-2.0 * dist[k_old][k_new])
        return d, True

    def try_close_hub(k_old):
        if not is_hub[k_old]:
            return None
        
        nodes_to_move = [i for i in range(n) if assignment[i] == k_old]
        is_hub[k_old] = False
        if not any(is_hub):
            is_hub[k_old] = True
            return None
            
        temp_load = [0.0] * n
        for h in range(n):
            if is_hub[h]:
                temp_load[h] = hub_load[h]
        
        total_delta = -setup[k_old]
        moves = []
        temp_assignment = list(assignment)
        
        for i in nodes_to_move:
            best_k = -1
            best_d = float('inf')
            for h in range(n):
                if is_hub[h]:
                    k_old_i = temp_assignment[i]
                    if temp_load[h] + outgoing[i] > cap[h] + 1e-6:
                        continue
                    d = chi * outgoing[i] * (dist[i][h] - dist[i][k_old_i])
                    d += delta * incoming[i] * (dist[h][i] - dist[k_old_i][i])
                    for j in range(n):
                        if j == i: continue
                        m_j = temp_assignment[j]
                        diff = dist[h][m_j] - dist[k_old_i][m_j]
                        if diff != 0:
                            d += alpha * (flow[i][j] + flow[j][i]) * diff
                    d += alpha * flow[i][i] * (-2.0 * dist[k_old_i][h])
                    
                    if d < best_d:
                        best_d = d
                        best_k = h
            if best_k == -1:
                is_hub[k_old] = True
                return None
            moves.append((i, best_k, best_d))
            temp_assignment[i] = best_k
            temp_load[best_k] += outgoing[i]
            
        sum_d = sum(d for _, _, d in moves)
        elapsed = time.time() - start_time
        Temp = max(0.1, 1000.0 * (1 - elapsed / time_limit))
        if total_delta + sum_d < 0 or rng.random() < math.exp(-(total_delta + sum_d) / Temp):
            total_delta += sum_d
            for i, h, d in moves:
                hub_load[assignment[i]] -= outgoing[i]
                assignment[i] = h
                hub_load[h] += outgoing[i]
            return float(total_delta)
        else:
            is_hub[k_old] = True
            return None

    while time.time() - start_time < time_limit:
        iterations += 1
        elapsed = time.time() - start_time
        Temp = max(0.1, 1000.0 * (1 - elapsed / time_limit))
        
        move_type = rng.random()
        if move_type < 0.8:
            i = rng.randint(0, n - 1)
            hubs = [k for k in range(n) if is_hub[k]]
            if not hubs:
                continue
            k = rng.choice(hubs)
            
            d, feasible = get_delta_reassign(i, k)
            if not feasible:
                continue
            
            if d < 0 or rng.random() < math.exp(-d / Temp):
                k_old = assignment[i]
                hub_load[k_old] -= outgoing[i]
                assignment[i] = k
                hub_load[k] += outgoing[i]
                obj += d
                
                if obj < best_obj - 1e-6:
                    best_obj = obj
                    best_assignment = list(assignment)
                    best_hubs = [k for k in range(n) if is_hub[k]]
                    if logger:
                        logger.log_solution(best_obj, {
                            'objective_value': best_obj,
                            'hubs': best_hubs,
                            'assignment': best_assignment
                        })
        elif move_type < 0.95:
            closed_hubs = [k for k in range(n) if not is_hub[k]]
            if not closed_hubs:
                continue
            k = rng.choice(closed_hubs)
            i = rng.randint(0, n - 1)
            
            d, feasible = get_delta_open_and_reassign(i, k)
            if not feasible:
                continue
            
            if d < 0 or rng.random() < math.exp(-d / Temp):
                k_old = assignment[i]
                is_hub[k] = True
                hub_load[k_old] -= outgoing[i]
                assignment[i] = k
                hub_load[k] += outgoing[i]
                obj += d
                
                if obj < best_obj - 1e-6:
                    best_obj = obj
                    best_assignment = list(assignment)
                    best_hubs = [k for k in range(n) if is_hub[k]]
                    if logger:
                        logger.log_solution(best_obj, {
                            'objective_value': best_obj,
                            'hubs': best_hubs,
                            'assignment': best_assignment
                        })
        else:
            hubs = [k for k in range(n) if is_hub[k]]
            if not hubs:
                continue
            k = rng.choice(hubs)
            
            total_delta = try_close_hub(k)
            if total_delta is not None:
                obj += total_delta
                if obj < best_obj - 1e-6:
                    best_obj = obj
                    best_assignment = list(assignment)
                    best_hubs = [k for k in range(n) if is_hub[k]]
                    if logger:
                        logger.log_solution(best_obj, {
                            'objective_value': best_obj,
                            'hubs': best_hubs,
                            'assignment': best_assignment
                        })
                        
    final_obj = compute_objective(n, flow, dist, outgoing, incoming, setup, chi, alpha, delta, best_assignment, best_hubs)
    return {
        'objective_value': final_obj,
        'hubs': best_hubs,
        'assignment': best_assignment
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    with open(args.instance_path) as f:
        inst = json.load(f)

    n = inst['n']
    chi = inst['cost_parameters']['collection_cost_chi']
    alpha = inst['cost_parameters']['transfer_cost_alpha']
    delta = inst['cost_parameters']['distribution_cost_delta']
    D = inst['total_flow_D']
    flow = inst['flow_matrix']
    dist = inst['distance_matrix']
    outgoing = inst['outgoing_flow']
    setup = inst['setup_costs']
    cap = inst['capacities']
    
    incoming = [0.0] * n
    for i in range(n):
        for j in range(n):
            incoming[i] += flow[j][i]

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    solution = solve_sa(n, flow, dist, outgoing, incoming, setup, cap, chi, alpha, delta, D, args.time_limit, logger)

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == '__main__':
    main()
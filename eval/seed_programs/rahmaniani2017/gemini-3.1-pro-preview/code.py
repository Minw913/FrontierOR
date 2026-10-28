import argparse
import json
import time
import random

def evaluate_X(X, unique_dests, dists, tau, U_list, fixed_costs, N):
    cost = sum(fixed_costs[h] for h in X)
    
    M = {}
    for i in X:
        row_i = dists[i]
        M_i = [0.0] * N
        for d in unique_dests:
            min_val = float('inf')
            for m in X:
                v = tau * row_i[m] + dists[m][d]
                if v < min_val: min_val = v
            M_i[d] = min_val
        M[i] = M_i

    for o, d, w in U_list:
        row_o = dists[o]
        min_val = float('inf')
        for i in X:
            v = row_o[i] + M[i][d]
            if v < min_val: min_val = v
        cost += w * min_val
        
    return cost

def main():
    parser = argparse.ArgumentParser(description="Uncapacitated Hub Location Problem")
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default="")
    args = parser.parse_args()

    start_time = time.time()
    random.seed(42)

    logger = None
    if args.log_path:
        try:
            from solution_logger import SolutionLogger
            logger = SolutionLogger(args.log_path, sense="minimize")
        except ImportError:
            pass

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    N = instance['num_nodes']
    tau = instance['parameters']['tau']
    distances = instance['distances']

    fixed_costs = [0.0] * N
    for n in instance['nodes']:
        fixed_costs[n['id']] = n['fixed_cost']

    U = {}
    for c in instance['commodities']:
        o = c['origin']
        d = c['destination']
        w = c['flow']
        u = (o, d)
        if u not in U:
            U[u] = 0.0
        U[u] += w
        
    U_list = [(o, d, w) for (o, d), w in U.items()]
    unique_dests = tuple(set(d for o, d, w in U_list))

    best_X = [0]
    best_cost = float('inf')
    for i in range(N):
        c = evaluate_X([i], unique_dests, distances, tau, U_list, fixed_costs, N)
        if c < best_cost:
            best_cost = c
            best_X = [i]

    if logger:
        logger.log(best_cost)

    TLM = args.time_limit - 1.5

    def local_search(cur_X_in, cur_cost_in):
        lx = list(cur_X_in)
        lc = cur_cost_in
        
        while time.time() - start_time < TLM:
            improved = False
            ops = ['add', 'drop', 'swap']
            random.shuffle(ops)
            
            for op in ops:
                if time.time() - start_time > TLM: return lx, lc
                
                if op == 'add':
                    nodes_idx = list(range(N))
                    random.shuffle(nodes_idx)
                    for i in nodes_idx:
                        if i not in lx:
                            cand = lx + [i]
                            c = evaluate_X(cand, unique_dests, distances, tau, U_list, fixed_costs, N)
                            if c < lc - 1e-6:
                                lc = c
                                lx = cand
                                improved = True
                                if logger: logger.log(lc)
                                break
                                
                elif op == 'drop' and len(lx) > 1:
                    cur_X_iter = list(lx)
                    random.shuffle(cur_X_iter)
                    for i in cur_X_iter:
                        if len(lx) == 1: break
                        cand = [h for h in lx if h != i]
                        c = evaluate_X(cand, unique_dests, distances, tau, U_list, fixed_costs, N)
                        if c < lc - 1e-6:
                            lc = c
                            lx = cand
                            improved = True
                            if logger: logger.log(lc)
                            break
                            
                elif op == 'swap':
                    cur_X_iter = list(lx)
                    random.shuffle(cur_X_iter)
                    for i in cur_X_iter:
                        if improved: break
                        nodes_idx = list(range(N))
                        random.shuffle(nodes_idx)
                        for j in nodes_idx:
                            if j not in lx:
                                cand = [h for h in lx if h != i] + [j]
                                c = evaluate_X(cand, unique_dests, distances, tau, U_list, fixed_costs, N)
                                if c < lc - 1e-6:
                                    lc = c
                                    lx = cand
                                    improved = True
                                    if logger: logger.log(lc)
                                    break
                                
                if improved:
                    break 
                    
            if not improved:
                break
                
        return lx, lc

    current_X, current_cost = local_search(best_X, best_cost)
    best_global_X = list(current_X)
    best_global_cost = current_cost

    while time.time() - start_time < TLM:
        perturb_X = list(best_global_X)
        k = random.randint(1, min(3, max(1, len(perturb_X))))
        for _ in range(k):
            op = random.choice(['add', 'drop', 'swap'])
            if op == 'add' and len(perturb_X) < N:
                pool = [x for x in range(N) if x not in perturb_X]
                if pool:
                    perturb_X.append(random.choice(pool))
            elif op == 'drop' and len(perturb_X) > 1:
                perturb_X.remove(random.choice(perturb_X))
            elif op == 'swap' and 0 < len(perturb_X) < N:
                drop_c = random.choice(perturb_X)
                pool_add = [x for x in range(N) if x not in perturb_X]
                if pool_add:
                    add_c = random.choice(pool_add)
                    perturb_X.remove(drop_c)
                    perturb_X.append(add_c)
                
        perturb_X = list(set(perturb_X))
        if not perturb_X:
            perturb_X = [best_global_X[0] if best_global_X else 0]
            
        perturb_cost = evaluate_X(perturb_X, unique_dests, distances, tau, U_list, fixed_costs, N)
        current_X, current_cost = local_search(perturb_X, perturb_cost)
        
        if current_cost < best_global_cost - 1e-5:
            best_global_cost = current_cost
            best_global_X = list(current_X)
            if logger:
                logger.log(best_global_cost)

    M_final = {}
    j_star = {}
    for i in best_global_X:
        M_final[i] = {}
        j_star[i] = {}
        row_i_dists = distances[i]
        for dest in unique_dests:
            best_val = float('inf')
            best_m = None
            for m in best_global_X:
                v = tau * row_i_dists[m] + distances[m][dest]
                if v < best_val:
                    best_val = v
                    best_m = m
            M_final[i][dest] = best_val
            j_star[i][dest] = best_m
            
    assignments = []
    total_assigned_cost = 0.0
    
    for idx, c in enumerate(instance['commodities']):
        o = c['origin']
        d = c['destination']
        w = c['flow']
        
        best_val = float('inf')
        best_i = None
        for i in best_global_X:
            v = distances[o][i] + M_final[i][d]
            if v < best_val:
                best_val = v
                best_i = i
                
        best_j = j_star[best_i][d]
        min_cost = best_val * w
        total_assigned_cost += min_cost
        
        hub_edge = [best_i] if best_i == best_j else [best_i, best_j]
        
        assignments.append({
            "commodity": idx,
            "origin": o,
            "destination": d,
            "hub_edge": hub_edge,
            "cost": min_cost
        })

    final_objective_value = sum(fixed_costs[h] for h in best_global_X) + total_assigned_cost

    solution = {
        "objective_value": final_objective_value,
        "open_hubs": sorted(best_global_X),
        "assignments": assignments
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()
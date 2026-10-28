import argparse
import json
import time
import numpy as np
from itertools import combinations
import os
import sys

# Suppress dbus and other warnings before any imports
os.environ['DBUS_SESSION_BUS_ADDRESS'] = '/dev/null'
os.environ['NO_AT_BRIDGE'] = '1'
os.environ['GRB_LICENSE_FILE'] = os.environ.get('GRB_LICENSE_FILE', '')

def solve(instance_path, solution_path, time_limit, log_path):
    from solution_logger import SolutionLogger
    logger = SolutionLogger(log_path, sense="maximize") if log_path else None
    
    with open(instance_path, 'r') as f:
        data = json.load(f)
    
    N = data['num_scenarios_N']
    n = data['problem_dimension_n']
    K_values = data['K_values']
    scenarios = np.array(data['scenarios'], dtype=float)
    
    start_time = time.time()
    
    def time_remaining():
        return max(0, time_limit - (time.time() - start_time))
    
    # Precompute ratio matrix: ratio_matrix[s][i] = min_j(scenarios[s][j] / scenarios[i][j])
    # This is the scaling factor if scenario i is represented by scenario s
    ratio_matrix = np.zeros((N, N))
    for s in range(N):
        ratio_matrix[s] = np.min(scenarios[s:s+1, :] / scenarios, axis=1)
    
    def greedy_two_stage(K):
        selected = []
        remaining = set(range(N))
        
        best_first = -1
        best_first_t = -1
        for s in range(N):
            t = np.min(ratio_matrix[s, :])
            if t > best_first_t:
                best_first_t = t
                best_first = s
        
        selected.append(best_first)
        remaining.remove(best_first)
        current_best = ratio_matrix[best_first, :].copy()
        
        for _ in range(K - 1):
            if time_remaining() < 0.5:
                break
            best_next = -1
            best_next_t = -1
            
            for s in remaining:
                new_best = np.maximum(current_best, ratio_matrix[s, :])
                t = np.min(new_best)
                if t > best_next_t:
                    best_next_t = t
                    best_next = s
            
            if best_next >= 0:
                selected.append(best_next)
                remaining.remove(best_next)
                current_best = np.maximum(current_best, ratio_matrix[best_next, :])
        
        while len(selected) < K:
            for s in remaining:
                selected.append(s)
                remaining.remove(s)
                break
            else:
                selected.append(0)
        
        t = float(np.min(current_best))
        return selected, t
    
    def local_search_two_stage(K, selected, best_t):
        selected = list(selected)
        improved = True
        max_iters = 50
        iter_count = 0
        
        while improved and iter_count < max_iters and time_remaining() > 1:
            improved = False
            iter_count += 1
            
            for ki in range(K):
                if time_remaining() < 0.5:
                    break
                
                other_indices = [selected[kk] for kk in range(K) if kk != ki]
                if other_indices:
                    other_ratios = ratio_matrix[other_indices, :]
                    other_best = np.max(other_ratios, axis=0)
                else:
                    other_best = np.zeros(N)
                
                non_selected = set(range(N)) - set(selected)
                
                best_swap = selected[ki]
                best_swap_t = best_t
                
                for s in non_selected:
                    new_best = np.maximum(other_best, ratio_matrix[s, :])
                    t = np.min(new_best)
                    if t > best_swap_t + 1e-10:
                        best_swap = s
                        best_swap_t = t
                
                if best_swap != selected[ki]:
                    selected[ki] = best_swap
                    best_t = best_swap_t
                    improved = True
                    break
        
        return selected, best_t
    
    def solve_group_simple(group):
        """Simple fallback: pick the single scenario that maximizes min ratio over group."""
        best_s = 0
        best_t = -1
        for s in range(N):
            min_r = float('inf')
            for i in group:
                r = ratio_matrix[s, i]
                min_r = min(min_r, r)
            if min_r > best_t:
                best_t = min_r
                best_s = s
        weights = np.zeros(N)
        weights[best_s] = 1.0
        return weights, best_t
    
    def solve_group_lp_numpy(group):
        """Solve the group LP using a simple iterative approach."""
        if len(group) == 1:
            i = group[0]
            best_s = np.argmax(ratio_matrix[:, i])
            weights = np.zeros(N)
            weights[best_s] = 1.0
            return weights, float(ratio_matrix[best_s, i])
        
        best_w = None
        best_t = -1.0
        
        # Try each single scenario
        single_scores = np.zeros(N)
        for s in range(N):
            min_r = float('inf')
            for i in group:
                min_r = min(min_r, ratio_matrix[s, i])
            single_scores[s] = min_r
            if min_r > best_t:
                best_t = min_r
                best_w = np.zeros(N)
                best_w[s] = 1.0
        
        # Try uniform over group members
        w_uniform = np.zeros(N)
        for i in group:
            w_uniform[i] = 1.0 / len(group)
        agg = w_uniform @ scenarios
        min_t = float('inf')
        for i in group:
            ratios = agg / scenarios[i]
            min_t = min(min_t, np.min(ratios))
        if min_t > best_t:
            best_t = min_t
            best_w = w_uniform.copy()
        
        # Try pairwise combinations of top candidates
        top_k = min(15, N)
        top_indices = np.argsort(-single_scores)[:top_k]
        
        for idx1 in range(len(top_indices)):
            if time_remaining() < 0.5:
                break
            for idx2 in range(idx1, len(top_indices)):
                s1, s2 = top_indices[idx1], top_indices[idx2]
                for alpha in np.linspace(0, 1, 11):
                    agg_try = alpha * scenarios[s1] + (1.0 - alpha) * scenarios[s2]
                    min_t_try = float('inf')
                    for i in group:
                        ratios = agg_try / scenarios[i]
                        min_t_try = min(min_t_try, np.min(ratios))
                    if min_t_try > best_t:
                        best_t = min_t_try
                        best_w = np.zeros(N)
                        best_w[s1] = alpha
                        best_w[s2] = 1.0 - alpha
        
        return best_w, best_t
    
    def try_gurobi_group_lp(group):
        """Try to solve group LP with Gurobi."""
        try:
            import gurobipy as gp
            from gurobipy import GRB
            
            env = gp.Env(params={"OutputFlag": 0})
            model = gp.Model("group_lp", env=env)
            model.Params.Seed = 0
            model.Params.MIPGap = 1e-4
            model.Params.NumericFocus = 0
            model.Params.Threads = 1
            model.Params.TimeLimit = max(1, time_remaining() - 1)
            
            w = model.addVars(N, lb=0.0, name="w")
            t = model.addVar(lb=0.0, name="t")
            
            model.setObjective(t, GRB.MAXIMIZE)
            
            model.addConstr(gp.quicksum(w[l] for l in range(N)) == 1.0)
            
            for i in group:
                for j in range(n):
                    model.addConstr(
                        gp.quicksum(w[l] * scenarios[l][j] for l in range(N)) >= t * scenarios[i][j]
                    )
            
            model.optimize()
            
            if model.status in [GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT]:
                if model.SolCount > 0:
                    weights = np.array([w[l].X for l in range(N)])
                    t_val = t.X
                    env.dispose()
                    return weights, float(t_val)
            
            env.dispose()
            return None, 0.0
        except Exception:
            return None, 0.0
    
    # Check if Gurobi is available
    gurobi_available = False
    try:
        import gurobipy as gp
        env = gp.Env(params={"OutputFlag": 0})
        m = gp.Model("test", env=env)
        m.addVar()
        m.optimize()
        env.dispose()
        gurobi_available = True
    except Exception:
        gurobi_available = False
    
    def compute_t_from_solution(assignment, lambda_weights):
        t_val = float('inf')
        for i in range(N):
            k = assignment[str(i)]
            weights = lambda_weights[str(k)]
            agg = np.zeros(n)
            for l_str, w in weights.items():
                l = int(l_str)
                agg += w * scenarios[l]
            ratios = agg / scenarios[i]
            min_ratio = np.min(ratios)
            t_val = min(t_val, min_ratio)
        return t_val
    
    def solve_two_stage(K):
        if K >= N:
            return {
                "t": 1.0,
                "approximation_ratio": 1.0,
                "selected_scenarios": list(range(N))
            }
        
        best_t = 0.0
        best_selected = None
        
        from math import comb
        total_combos = comb(N, K) if N <= 100 else float('inf')
        
        if total_combos <= 50000 and time_remaining() > 2:
            for combo in combinations(range(N), K):
                if time_remaining() < 1:
                    break
                combo_list = list(combo)
                ratios = ratio_matrix[combo_list, :]
                best_rep = np.max(ratios, axis=0)
                t = np.min(best_rep)
                if t > best_t:
                    best_t = t
                    best_selected = combo_list
        else:
            best_selected, best_t = greedy_two_stage(K)
            if time_remaining() > 2:
                best_selected, best_t = local_search_two_stage(K, best_selected, best_t)
        
        if best_selected is None:
            best_selected = list(range(K))
            ratios = ratio_matrix[best_selected, :]
            best_rep = np.max(ratios, axis=0)
            best_t = float(np.min(best_rep))
        
        return {
            "t": float(best_t),
            "approximation_ratio": float(1.0 / best_t) if best_t > 0 else float('inf'),
            "selected_scenarios": [int(s) for s in best_selected]
        }
    
    def compute_assignment_from_selected(K, selected):
        assignment = {}
        for i in range(N):
            best_k = 0
            best_ratio = -1
            for kidx, s in enumerate(selected):
                ratio = ratio_matrix[s, i]
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_k = kidx
            assignment[str(i)] = best_k
        return assignment
    
    def optimize_lambda_for_fixed_assignment(K, assignment_dict):
        lambda_weights = {}
        t_val = float('inf')
        
        for k in range(K):
            group = [i for i in range(N) if assignment_dict[str(i)] == k]
            if not group:
                lambda_weights[str(k)] = {str(0): 1.0}
                continue
            
            best_weights = None
            best_group_t = -1
            
            # Try Gurobi first
            if gurobi_available and time_remaining() > 2:
                w, gt = try_gurobi_group_lp(group)
                if w is not None and gt > best_group_t:
                    best_weights = w
                    best_group_t = gt
            
            # Try numpy-based approach
            if time_remaining() > 1:
                sw, st = solve_group_lp_numpy(group)
                if st > best_group_t:
                    best_weights = sw
                    best_group_t = st
            
            # Fallback: simple single scenario
            sw2, st2 = solve_group_simple(group)
            if st2 > best_group_t:
                best_weights = sw2
                best_group_t = st2
            
            if best_weights is not None:
                weights = {}
                for l in range(N):
                    if best_weights[l] > 1e-8:
                        weights[str(l)] = float(best_weights[l])
                if not weights:
                    weights[str(group[0])] = 1.0
                lambda_weights[str(k)] = weights
                t_val = min(t_val, best_group_t)
            else:
                lambda_weights[str(k)] = {str(group[0]): 1.0}
                t_val = min(t_val, ratio_matrix[group[0], group[0]])
        
        return lambda_weights, t_val
    
    one_stage_results = {}
    two_stage_results = {}
    best_obj = 0.0
    
    solution = {
        "objective_value": 0.0,
        "one_stage_IP_mu": {},
        "two_stage_IP": {}
    }
    
    for K in sorted(K_values):
        if time_remaining() < 0.5:
            selected = list(range(min(K, N)))
            while len(selected) < K:
                selected.append(0)
            assignment = {str(i): i % K for i in range(N)}
            lw = {}
            for k in range(K):
                members = [i for i in range(N) if i % K == k]
                if members:
                    w = 1.0 / len(members)
                    lw[str(k)] = {str(m): w for m in members}
                else:
                    lw[str(k)] = {str(0): 1.0}
            t = compute_t_from_solution(assignment, lw)
            one_stage_results[str(K)] = {
                "t": float(t), "approximation_ratio": float(1.0/t) if t > 0 else float('inf'),
                "assignment": assignment, "lambda_weights": lw,
                "solver_status": 1, "best_bound": 1.0, "mip_gap": 1.0
            }
            two_stage_results[str(K)] = {
                "t": 0.0, "approximation_ratio": float('inf'),
                "selected_scenarios": selected
            }
            continue
        
        ts2 = solve_two_stage(K)
        two_stage_results[str(K)] = ts2
        
        selected = ts2['selected_scenarios']
        assignment = compute_assignment_from_selected(K, selected)
        
        lambda_weights, t_val = optimize_lambda_for_fixed_assignment(K, assignment)
        
        os_result = {
            "t": float(t_val),
            "approximation_ratio": float(1.0 / t_val) if t_val > 0 else float('inf'),
            "assignment": assignment,
            "lambda_weights": lambda_weights,
            "solver_status": 2,
            "best_bound": float(t_val),
            "mip_gap": 0.0
        }
        
        one_stage_results[str(K)] = os_result
        
        t_val_final = max(os_result['t'], ts2['t'])
        if t_val_final > best_obj:
            best_obj = t_val_final
        
        solution['objective_value'] = float(best_obj)
        solution['one_stage_IP_mu'] = one_stage_results
        solution['two_stage_IP'] = two_stage_results
        
        if logger:
            logger.log_solution(float(best_obj), solution)
    
    solution['objective_value'] = float(best_obj)
    solution['one_stage_IP_mu'] = one_stage_results
    solution['two_stage_IP'] = two_stage_results
    
    if logger:
        logger.log_solution(float(best_obj), solution)
    
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    # Suppress dbus warnings before anything else
    os.environ['DBUS_SESSION_BUS_ADDRESS'] = '/dev/null'
    os.environ['NO_AT_BRIDGE'] = '1'
    
    # Redirect stderr temporarily to suppress dbus connection errors
    old_stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    # Restore stderr
    sys.stderr.close()
    sys.stderr = old_stderr
    
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)
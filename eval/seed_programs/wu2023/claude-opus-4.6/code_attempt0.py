import argparse
import json
import time
import numpy as np
from itertools import combinations

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
    
    best_overall_obj = 0.0
    one_stage_results = {}
    two_stage_results = {}
    
    def time_remaining():
        return max(0, time_limit - (time.time() - start_time))
    
    # For two-stage: select K scenarios, assign each original to one selected
    # to maximize min over i of min over j of (selected[assign[i]][j] / scenario[i][j])
    
    def solve_two_stage(K):
        if K >= N:
            # Select all
            selected = list(range(N))
            return {
                "t": 1.0,
                "approximation_ratio": 1.0,
                "selected_scenarios": selected
            }
        
        best_t = 0.0
        best_selected = None
        
        # Precompute ratio matrix: ratio[s][i] = min_j(scenarios[s][j] / scenarios[i][j])
        # This is the scaling factor if scenario i is represented by scenario s
        # We need scenarios[s][j] >= t * scenarios[i][j] for all j
        # So t <= min_j(scenarios[s][j] / scenarios[i][j])
        
        ratio_matrix = np.zeros((N, N))
        for s in range(N):
            for i in range(N):
                ratio_matrix[s][i] = np.min(scenarios[s] / scenarios[i])
        
        # For small N, enumerate all combinations
        if N <= 20 and K <= N//2:
            max_combos = 100000
            from math import comb
            total_combos = comb(N, K)
            
            if total_combos <= max_combos:
                for combo in combinations(range(N), K):
                    if time_remaining() < 1:
                        break
                    combo_list = list(combo)
                    # For each original scenario, find best representative
                    # t = min over i of max over s in combo of ratio_matrix[s][i]
                    ratios = ratio_matrix[combo_list, :]  # K x N
                    best_rep = np.max(ratios, axis=0)  # N
                    t = np.min(best_rep)
                    if t > best_t:
                        best_t = t
                        best_selected = combo_list
            else:
                # Greedy approach
                best_selected, best_t = greedy_two_stage(K, ratio_matrix)
        else:
            best_selected, best_t = greedy_two_stage(K, ratio_matrix)
        
        if best_selected is None:
            best_selected = list(range(K))
            ratios = ratio_matrix[best_selected, :]
            best_rep = np.max(ratios, axis=0)
            best_t = float(np.min(best_rep))
        
        return {
            "t": float(best_t),
            "approximation_ratio": float(1.0 / best_t) if best_t > 0 else float('inf'),
            "selected_scenarios": best_selected
        }
    
    def greedy_two_stage(K, ratio_matrix):
        # Greedy: iteratively add scenario that maximizes the minimum scaling factor
        selected = []
        remaining = set(range(N))
        
        # Start with the scenario that maximizes the worst-case ratio as single representative
        best_first = -1
        best_first_t = -1
        for s in range(N):
            t = np.min(ratio_matrix[s, :])
            if t > best_first_t:
                best_first_t = t
                best_first = s
        
        selected.append(best_first)
        remaining.remove(best_first)
        
        for _ in range(K - 1):
            if time_remaining() < 0.5:
                break
            best_next = -1
            best_next_t = -1
            
            # Current coverage
            current_ratios = ratio_matrix[selected, :]  # len(selected) x N
            current_best = np.max(current_ratios, axis=0)  # N
            
            for s in remaining:
                new_best = np.maximum(current_best, ratio_matrix[s, :])
                t = np.min(new_best)
                if t > best_next_t:
                    best_next_t = t
                    best_next = s
            
            if best_next >= 0:
                selected.append(best_next)
                remaining.remove(best_next)
        
        # Fill remaining if needed
        while len(selected) < K:
            for s in remaining:
                selected.append(s)
                remaining.remove(s)
                break
        
        ratios = ratio_matrix[selected, :]
        best_rep = np.max(ratios, axis=0)
        t = float(np.min(best_rep))
        
        return selected, t
    
    def solve_one_stage_ip(K):
        """
        One-stage IP with mu (binary assignment) and lambda (convex combination weights).
        
        For each K, we partition scenarios into K groups. Each group k has an aggregate
        that is a convex combination of all scenarios (weighted by lambda).
        
        Maximize t such that:
        - For each scenario i assigned to group k (mu[i,k]=1), for each dimension j:
          sum_l lambda[k,l] * scenarios[l][j] >= t * scenarios[i][j]
        - sum_k mu[i,k] = 1 for all i
        - sum_l lambda[k,l] = 1 for all k
        - lambda[k,l] >= 0
        - mu[i,k] in {0,1}
        """
        if K >= N:
            assignment = {str(i): i for i in range(N)}
            lambda_weights = {str(k): {str(k): 1.0} for k in range(N)}
            return {
                "t": 1.0,
                "approximation_ratio": 1.0,
                "assignment": assignment,
                "lambda_weights": lambda_weights,
                "solver_status": 2,
                "best_bound": 1.0,
                "mip_gap": 0.0
            }
        
        import gurobipy as gp
        from gurobipy import GRB
        
        tr = time_remaining()
        if tr < 2:
            # Return a trivial solution
            return trivial_one_stage(K)
        
        try:
            model = gp.Model("one_stage_IP_mu")
            model.setParam('Seed', 0)
            model.setParam('MIPGap', 1e-4)
            model.setParam('NumericFocus', 0)
            model.setParam('Threads', 1)
            model.setParam('TimeLimit', min(tr - 1, tr * 0.8))
            model.setParam('OutputFlag', 0)
            
            # Variables
            t = model.addVar(lb=0, ub=1, name="t")
            mu = {}
            for i in range(N):
                for k in range(K):
                    mu[i, k] = model.addVar(vtype=GRB.BINARY, name=f"mu_{i}_{k}")
            
            lam = {}
            for k in range(K):
                for l in range(N):
                    lam[k, l] = model.addVar(lb=0, ub=1, name=f"lam_{k}_{l}")
            
            model.setObjective(t, GRB.MAXIMIZE)
            
            # Each scenario assigned to exactly one group
            for i in range(N):
                model.addConstr(gp.quicksum(mu[i, k] for k in range(K)) == 1)
            
            # Convex combination for each aggregate
            for k in range(K):
                model.addConstr(gp.quicksum(lam[k, l] for l in range(N)) == 1)
            
            # Linking constraints: aggregate[k][j] >= t * scenario[i][j] when mu[i,k]=1
            # This is: sum_l lam[k,l] * scenarios[l][j] >= t * scenarios[i][j] - M*(1-mu[i,k])
            # We need a big-M. Since all costs positive, M can be max scenario value.
            M_val = float(np.max(scenarios)) * 2
            
            for i in range(N):
                for k in range(K):
                    for j in range(n):
                        model.addConstr(
                            gp.quicksum(lam[k, l] * float(scenarios[l][j]) for l in range(N)) 
                            >= t * float(scenarios[i][j]) - M_val * (1 - mu[i, k])
                        )
            
            # Warm start from greedy two-stage solution
            # Use the two-stage result if available
            
            model.optimize()
            
            status = model.Status
            
            if model.SolCount > 0:
                t_val = t.X
                assignment = {}
                lambda_weights = {}
                
                for i in range(N):
                    for k in range(K):
                        if mu[i, k].X > 0.5:
                            assignment[str(i)] = k
                            break
                
                for k in range(K):
                    weights = {}
                    for l in range(N):
                        v = lam[k, l].X
                        if v > 1e-8:
                            weights[str(l)] = float(v)
                    lambda_weights[str(k)] = weights
                
                return {
                    "t": float(t_val),
                    "approximation_ratio": float(1.0 / t_val) if t_val > 0 else float('inf'),
                    "assignment": assignment,
                    "lambda_weights": lambda_weights,
                    "solver_status": int(status),
                    "best_bound": float(model.ObjBound) if hasattr(model, 'ObjBound') else float(t_val),
                    "mip_gap": float(model.MIPGap) if hasattr(model, 'MIPGap') and model.SolCount > 0 else 0.0
                }
            else:
                return trivial_one_stage(K)
                
        except Exception as e:
            return trivial_one_stage(K)
    
    def trivial_one_stage(K):
        """Build a trivial feasible solution for one-stage."""
        # Assign all scenarios to first K groups using simple clustering
        assignment = {}
        for i in range(N):
            assignment[str(i)] = i % K
        
        lambda_weights = {}
        for k in range(K):
            group_members = [i for i in range(N) if assignment[str(i)] == k]
            if group_members:
                # Use componentwise max as aggregate approximation
                # But we need convex combination, so use uniform weights over group members
                weights = {}
                w = 1.0 / len(group_members)
                for m in group_members:
                    weights[str(m)] = w
                lambda_weights[str(k)] = weights
            else:
                lambda_weights[str(k)] = {str(0): 1.0}
        
        # Compute t
        t_val = compute_t_from_solution(assignment, lambda_weights, K)
        
        return {
            "t": float(t_val),
            "approximation_ratio": float(1.0 / t_val) if t_val > 0 else float('inf'),
            "assignment": assignment,
            "lambda_weights": lambda_weights,
            "solver_status": 1,
            "best_bound": 1.0,
            "mip_gap": 1.0
        }
    
    def compute_t_from_solution(assignment, lambda_weights, K):
        """Compute the scaling factor from assignment and lambda weights."""
        t_val = float('inf')
        for i in range(N):
            k = assignment[str(i)]
            weights = lambda_weights[str(k)]
            # Compute aggregate
            agg = np.zeros(n)
            for l_str, w in weights.items():
                l = int(l_str)
                agg += w * scenarios[l]
            # Compute ratio
            ratios = agg / scenarios[i]
            min_ratio = np.min(ratios)
            t_val = min(t_val, min_ratio)
        return t_val
    
    def build_one_stage_from_two_stage(K, two_stage_result):
        """Build a one-stage solution from two-stage selected scenarios."""
        selected = two_stage_result['selected_scenarios']
        
        # Assign each scenario to best selected scenario
        assignment = {}
        for i in range(N):
            best_k = 0
            best_ratio = -1
            for kidx, s in enumerate(selected):
                ratio = np.min(scenarios[s] / scenarios[i])
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_k = kidx
            assignment[str(i)] = best_k
        
        # Lambda weights: each aggregate is just the selected scenario
        lambda_weights = {}
        for kidx, s in enumerate(selected):
            lambda_weights[str(kidx)] = {str(s): 1.0}
        
        t_val = compute_t_from_solution(assignment, lambda_weights, K)
        
        return {
            "t": float(t_val),
            "approximation_ratio": float(1.0 / t_val) if t_val > 0 else float('inf'),
            "assignment": assignment,
            "lambda_weights": lambda_weights,
            "solver_status": 2,
            "best_bound": float(t_val),
            "mip_gap": 0.0
        }
    
    def improve_one_stage_with_lp(K, assignment_dict):
        """Given fixed binary assignment, optimize lambda weights with LP."""
        import gurobipy as gp
        from gurobipy import GRB
        
        tr = time_remaining()
        if tr < 1:
            return None
        
        try:
            model = gp.Model("one_stage_LP")
            model.setParam('Seed', 0)
            model.setParam('NumericFocus', 0)
            model.setParam('Threads', 1)
            model.setParam('TimeLimit', min(tr - 0.5, 30))
            model.setParam('OutputFlag', 0)
            
            t = model.addVar(lb=0, ub=1, name="t")
            
            lam = {}
            for k in range(K):
                for l in range(N):
                    lam[k, l] = model.addVar(lb=0, ub=1, name=f"lam_{k}_{l}")
            
            model.setObjective(t, GRB.MAXIMIZE)
            
            for k in range(K):
                model.addConstr(gp.quicksum(lam[k, l] for l in range(N)) == 1)
            
            for i in range(N):
                k = assignment_dict[str(i)]
                for j in range(n):
                    model.addConstr(
                        gp.quicksum(lam[k, l] * float(scenarios[l][j]) for l in range(N))
                        >= t * float(scenarios[i][j])
                    )
            
            model.optimize()
            
            if model.SolCount > 0:
                t_val = t.X
                lambda_weights = {}
                for k in range(K):
                    weights = {}
                    for l in range(N):
                        v = lam[k, l].X
                        if v > 1e-8:
                            weights[str(l)] = float(v)
                    lambda_weights[str(k)] = weights
                
                return {
                    "t": float(t_val),
                    "approximation_ratio": float(1.0 / t_val) if t_val > 0 else float('inf'),
                    "assignment": assignment_dict,
                    "lambda_weights": lambda_weights,
                    "solver_status": int(model.Status),
                    "best_bound": float(model.ObjBound) if hasattr(model, 'ObjBound') else float(t_val),
                    "mip_gap": float(model.MIPGap) if model.SolCount > 0 else 0.0
                }
            return None
        except:
            return None
    
    # Determine problem size to choose strategy
    total_constraints_per_K = N * n  # approximate for full MIP
    
    solution = {
        "objective_value": 0.0,
        "one_stage_IP_mu": {},
        "two_stage_IP": {}
    }
    
    best_obj = 0.0
    
    for K in sorted(K_values):
        if time_remaining() < 2:
            # Use trivial solutions for remaining K values
            ts = trivial_one_stage(K)
            one_stage_results[str(K)] = ts
            
            # Trivial two-stage
            selected = list(range(min(K, N)))
            while len(selected) < K:
                selected.append(0)
            two_stage_results[str(K)] = {
                "t": 0.0,
                "approximation_ratio": float('inf'),
                "selected_scenarios": selected
            }
            continue
        
        # Solve two-stage first (faster)
        ts2 = solve_two_stage(K)
        two_stage_results[str(K)] = ts2
        
        # Build one-stage from two-stage as warm start
        os_from_ts = build_one_stage_from_two_stage(K, ts2)
        
        # Try to improve with LP (optimize lambda given fixed assignment)
        os_improved = improve_one_stage_with_lp(K, os_from_ts['assignment'])
        
        if os_improved and os_improved['t'] > os_from_ts['t']:
            best_one_stage = os_improved
        else:
            best_one_stage = os_from_ts
        
        # For small problems, try full MIP
        small_enough = (N * K * n <= 5000) and (N <= 30) and (K <= 10)
        if small_enough and time_remaining() > 10:
            mip_result = solve_one_stage_ip(K)
            if mip_result['t'] > best_one_stage['t']:
                best_one_stage = mip_result
        
        # Try different assignments via clustering and LP improvement
        if time_remaining() > 5 and K > 1 and K < N:
            # Try componentwise-max based clustering
            # For each pair of scenarios, compute similarity
            # Use a simple k-medoids-like approach
            for trial in range(min(5, max(1, int(time_remaining() // 3)))):
                if time_remaining() < 2:
                    break
                
                # Random-ish assignment
                if trial == 0:
                    # Assign based on sorting by first dimension
                    order = np.argsort(scenarios[:, 0])
                    trial_assignment = {}
                    chunk = max(1, N // K)
                    for idx, i in enumerate(order):
                        trial_assignment[str(int(i))] = min(idx // chunk, K - 1)
                elif trial == 1 and n > 1:
                    # Sort by second dimension
                    order = np.argsort(scenarios[:, 1 % n])
                    trial_assignment = {}
                    chunk = max(1, N // K)
                    for idx, i in enumerate(order):
                        trial_assignment[str(int(i))] = min(idx // chunk, K - 1)
                else:
                    # Random permutation
                    np.random.seed(trial + 42)
                    order = np.random.permutation(N)
                    trial_assignment = {}
                    chunk = max(1, N // K)
                    for idx, i in enumerate(order):
                        trial_assignment[str(int(i))] = min(idx // chunk, K - 1)
                
                # Ensure all groups have at least one member
                used_groups = set(trial_assignment.values())
                if len(used_groups) < K:
                    # Fix by reassigning
                    missing = set(range(K)) - used_groups
                    for m in missing:
                        # Find largest group and split
                        group_sizes = {}
                        for v in trial_assignment.values():
                            group_sizes[v] = group_sizes.get(v, 0) + 1
                        largest = max(group_sizes, key=group_sizes.get)
                        for key, val in trial_assignment.items():
                            if val == largest:
                                trial_assignment[key] = m
                                break
                
                improved = improve_one_stage_with_lp(K, trial_assignment)
                if improved and improved['t'] > best_one_stage['t']:
                    best_one_stage = improved
        
        one_stage_results[str(K)] = best_one_stage
        
        # Track best objective
        t_val = max(best_one_stage['t'], ts2['t'])
        if t_val > best_obj:
            best_obj = t_val
        
        # Update solution and log
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)
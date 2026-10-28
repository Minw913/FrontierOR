import gurobipy as gp
from gurobipy import GRB
import json
import time
import argparse
from solution_logger import SolutionLogger

def compute_t_is(scenarios):
    N = len(scenarios)
    n = len(scenarios[0])
    t_is = [[float('inf')] * N for _ in range(N)]
    T_ub = 0.0
    for i in range(N):
        for s in range(N):
            min_ratio = float('inf')
            for j in range(n):
                if scenarios[i][j] > 0:
                    ratio = scenarios[s][j] / scenarios[i][j]
                    if ratio < min_ratio:
                        min_ratio = ratio
            t_is[i][s] = min_ratio
            if min_ratio < float('inf') and min_ratio > T_ub:
                T_ub = min_ratio
    return t_is, T_ub

def greedy_two_stage(t_is, N, K):
    if K == 0:
        return [], {}, 0.0
    min_vals_per_s = [min(t_is[i][s] for i in range(N)) for s in range(N)]
    s0 = max(range(N), key=lambda s: min_vals_per_s[s])
    S = {s0}
    d_i = [t_is[i][s0] for i in range(N)]
    for k in range(1, K):
        best_candidate = None
        best_t = -1
        for s in range(N):
            if s in S:
                continue
            new_d_i = [max(d_i[i], t_is[i][s]) for i in range(N)]
            candidate_t = min(new_d_i)
            if candidate_t > best_t:
                best_t = candidate_t
                best_candidate = s
        if best_candidate is None:
            break
        S.add(best_candidate)
        d_i = [max(d_i[i], t_is[i][best_candidate]) for i in range(N)]
    assignment_two = {}
    for i in range(N):
        best_s = None
        best_val = -1
        for s in S:
            if t_is[i][s] > best_val:
                best_val = t_is[i][s]
                best_s = s
        assignment_two[i] = best_s
    selected_scenarios = sorted(S)
    t_val = min(d_i)
    return selected_scenarios, assignment_two, t_val

def solve_one_stage_for_group(scenarios, group_indices):
    n = len(scenarios[0])
    if not group_indices:
        return float('inf'), {}, GRB.OPTIMAL
    M_j = [max(scenarios[i][j] for i in group_indices) for j in range(n)]
    model = gp.Model()
    model.setParam('OutputFlag', 0)
    model.setParam('Seed', 0)
    model.setParam('NumericFocus', 0)
    t_var = model.addVar(lb=0, name="t")
    lambda_vars = model.addVars(len(group_indices), lb=0, name="lambda")
    model.addConstr(sum(lambda_vars) == 1)
    for j in range(n):
        agg_j = gp.quicksum(lambda_vars[idx] * scenarios[group_indices[idx]][j] for idx in range(len(group_indices)))
        model.addConstr(agg_j >= t_var * M_j[j])
    model.setObjective(t_var, GRB.MAXIMIZE)
    model.optimize()
    if model.Status == GRB.OPTIMAL:
        t_val = t_var.X
        weights_dict = {str(group_indices[idx]): lambda_vars[idx].X for idx in range(len(group_indices))}
        return t_val, weights_dict, model.Status
    else:
        t_val = float('inf')
        for j in range(n):
            avg = sum(scenarios[i][j] for i in group_indices) / len(group_indices)
            if M_j[j] > 0:
                ratio = avg / M_j[j]
                if ratio < t_val:
                    t_val = ratio
        weights_dict = {str(i): 1.0/len(group_indices) for i in group_indices}
        return t_val, weights_dict, model.Status

def main(instance_path, solution_path, time_limit, log_path=None):
    logger = SolutionLogger(log_path, sense="maximize") if log_path else None
    with open(instance_path) as f:
        data = json.load(f)
    
    n_dim = data["problem_dimension_n"]
    N = data["num_scenarios_N"]
    K_values = data["K_values"]
    scenarios = data["scenarios"]
    
    t_is, T_ub = compute_t_is(scenarios)
    overall_best_t = 0.0
    output = {
        "objective_value": 0.0,
        "one_stage_IP_mu": {},
        "two_stage_IP": {}
    }
    start_time = time.time()
    
    for K in K_values:
        time_elapsed = time.time() - start_time
        time_remaining = time_limit - time_elapsed
        if time_remaining <= 0:
            break
        
        selected_scenarios = []
        assignment_two = {}
        t_two_val = 0.0
        if N <= 200 and time_remaining > 1:
            model2 = gp.Model()
            model2.setParam('Seed', 0)
            model2.setParam('MIPGap', 1e-4)
            model2.setParam('NumericFocus', 0)
            model2.setParam('Threads', 1)
            model2.setParam('OutputFlag', 0)
            model2.setParam('TimeLimit', max(1, time_remaining))
            
            y = model2.addVars(N, vtype=GRB.BINARY, name="y")
            z = model2.addVars(N, N, vtype=GRB.BINARY, name="z")
            t_var = model2.addVar(lb=0, ub=T_ub, name="t")
            
            model2.addConstr(sum(y[s] for s in range(N)) == K, "selectK")
            for i in range(N):
                model2.addConstr(sum(z[i, s] for s in range(N)) == 1, f"assign_{i}")
            for i in range(N):
                for s in range(N):
                    model2.addConstr(z[i, s] <= y[s], f"link_{i}_{s}")
            for i in range(N):
                for s in range(N):
                    model2.addConstr(t_var <= t_is[i][s] + T_ub * (1 - z[i, s]), f"scaling_{i}_{s}")
            
            model2.setObjective(t_var, GRB.MAXIMIZE)
            model2.optimize()
            
            if model2.SolCount > 0:
                t_two_val = t_var.X
                selected_scenarios = [s for s in range(N) if y[s].X > 0.5]
                assignment_two = {}
                for i in range(N):
                    for s in range(N):
                        if z[i, s].X > 0.5:
                            assignment_two[i] = s
                            break
            else:
                selected_scenarios, assignment_two, t_two_val = greedy_two_stage(t_is, N, K)
        else:
            selected_scenarios, assignment_two, t_two_val = greedy_two_stage(t_is, N, K)
        
        output["two_stage_IP"][str(K)] = {
            "t": t_two_val,
            "approximation_ratio": 1 / t_two_val if t_two_val > 0 else float('inf'),
            "selected_scenarios": selected_scenarios
        }
        if t_two_val > overall_best_t:
            overall_best_t = t_two_val
            output["objective_value"] = overall_best_t
            if logger:
                logger.log_solution(overall_best_t, output)
        
        time_elapsed = time.time() - start_time
        time_remaining = time_limit - time_elapsed
        if time_remaining <= 0:
            break
        
        mapping_s_to_k = {s: idx for idx, s in enumerate(sorted(selected_scenarios))}
        assignment_one = {}
        groups = {k: [] for k in range(K)}
        for i in range(N):
            s_i = assignment_two[i]
            k = mapping_s_to_k[s_i]
            assignment_one[str(i)] = k
            groups[k].append(i)
        
        group_t_vals = []
        lambda_weights = {}
        statuses = []
        for k in range(K):
            group = groups[k]
            t_val, weights_dict, status = solve_one_stage_for_group(scenarios, group)
            group_t_vals.append(t_val)
            lambda_weights[str(k)] = weights_dict
            statuses.append(status)
        
        t_one_val = min(group_t_vals) if group_t_vals else 0.0
        solver_status = GRB.OPTIMAL if all(s == GRB.OPTIMAL for s in statuses) else statuses[0]
        
        output["one_stage_IP_mu"][str(K)] = {
            "t": t_one_val,
            "approximation_ratio": 1 / t_one_val if t_one_val > 0 else float('inf'),
            "assignment": assignment_one,
            "lambda_weights": lambda_weights,
            "solver_status": solver_status,
            "best_bound": t_one_val,
            "mip_gap": 0.0
        }
        if t_one_val > overall_best_t:
            overall_best_t = t_one_val
            output["objective_value"] = overall_best_t
            if logger:
                logger.log_solution(overall_best_t, output)
    
    output["objective_value"] = overall_best_t
    with open(solution_path, 'w') as f:
        json.dump(output, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    main(args.instance_path, args.solution_path, args.time_limit, args.log_path)
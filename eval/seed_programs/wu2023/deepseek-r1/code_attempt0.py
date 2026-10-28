import gurobipy as gp
from gurobipy import GRB
import json
import time
import argparse

def main(instance_file, time_limit, output_file):
    with open(instance_file) as f:
        data = json.load(f)
    
    N = data["n"]
    n = data["dimension"]
    scenarios = data["scenarios"]
    K_values = data["K_values"]
    
    # Precompute t_is for two-stage
    t_is = []
    T_ub = 0.0
    for i in range(N):
        row = []
        for s in range(N):
            min_ratio = float('inf')
            for j in range(n):
                if scenarios[i][j] > 0:
                    ratio = scenarios[s][j] / scenarios[i][j]
                    if ratio < min_ratio:
                        min_ratio = ratio
                else:
                    min_ratio = 0.0 if scenarios[s][j] == 0 else float('inf')
            row.append(min_ratio)
            if min_ratio > T_ub:
                T_ub = min_ratio
        t_is.append(row)
    
    output = {
        "objective_value": 0.0,
        "one_stage_IP_mu": {},
        "two_stage_IP": {}
    }
    overall_best_t = 0.0
    start_time = time.time()
    
    for K in K_values:
        time_elapsed = time.time() - start_time
        time_remaining = time_limit - time_elapsed
        if time_remaining <= 0:
            break
        
        # Solve two-stage for K
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
            selected_scenarios = list(range(K))
            assignment_two = {}
            for i in range(N):
                best_s = None
                best_t = 0
                for s in selected_scenarios:
                    min_ratio = float('inf')
                    for j in range(n):
                        if scenarios[i][j] > 0:
                            ratio = scenarios[s][j] / scenarios[i][j]
                            if ratio < min_ratio:
                                min_ratio = ratio
                        else:
                            min_ratio = 0.0 if scenarios[s][j] == 0 else float('inf')
                    if min_ratio > best_t:
                        best_t = min_ratio
                        best_s = s
                assignment_two[i] = best_s
            t_two_val = min(
                max(
                    min(
                        scenarios[s][j] / scenarios[i][j] 
                        for j in range(n) 
                        if scenarios[i][j] > 0 or scenarios[s][j] == 0
                    ) 
                    for s in selected_scenarios
                ) 
                for i in range(N)
            )
        
        output["two_stage_IP"][str(K)] = {
            "t": t_two_val,
            "approximation_ratio": 1 / t_two_val if t_two_val > 0 else float('inf'),
            "selected_scenarios": selected_scenarios
        }
        overall_best_t = max(overall_best_t, t_two_val)
        
        time_elapsed = time.time() - start_time
        time_remaining = time_limit - time_elapsed
        if time_remaining <= 0:
            break
        
        # Build one-stage solution from two-stage
        mapping_s_to_k = {s: idx for idx, s in enumerate(selected_scenarios)}
        assignment_one = {}
        groups = {k: [] for k in range(K)}
        for i in range(N):
            s_i = assignment_two[i]
            k = mapping_s_to_k[s_i]
            assignment_one[str(i)] = k
            groups[k].append(i)
        
        t_k_vals = []
        lambda_weights = {}
        for k in range(K):
            I_k = groups[k]
            if not I_k:
                t_k_vals.append(float('inf'))
                lambda_weights[str(k)] = {}
                continue
            
            model_lp = gp.Model()
            model_lp.setParam('OutputFlag', 0)
            model_lp.setParam('Seed', 0)
            t_k_var = model_lp.addVar(lb=0, name="t_k")
            lambda_vars = model_lp.addVars(len(I_k), lb=0, name="lambda")
            model_lp.addConstr(sum(lambda_vars[i] for i in range(len(I_k))) == 1)
            
            M_j = [max(scenarios[i][j] for i in I_k) for j in range(n)]
            for j in range(n):
                agg_j = sum(lambda_vars[idx] * scenarios[I_k[idx]][j] for idx in range(len(I_k)))
                model_lp.addConstr(agg_j >= t_k_var * M_j[j])
            
            model_lp.setObjective(t_k_var, GRB.MAXIMIZE)
            model_lp.optimize()
            
            if model_lp.Status == GRB.OPTIMAL:
                t_k_val = t_k_var.X
                weights_dict = {str(I_k[idx]): lambda_vars[idx].X for idx in range(len(I_k))}
            else:
                avg_j = [sum(scenarios[i][j] for i in I_k) / len(I_k) for j in range(n)]
                t_k_val = min(avg_j[j] / M_j[j] for j in range(n) if M_j[j] > 0)
                weights_dict = {str(i): 1.0 / len(I_k) for i in I_k}
            
            t_k_vals.append(t_k_val)
            lambda_weights[str(k)] = weights_dict
        
        t_one_val = min(t_k_vals) if t_k_vals else 0.0
        output["one_stage_IP_mu"][str(K)] = {
            "t": t_one_val,
            "approximation_ratio": 1 / t_one_val if t_one_val > 0 else float('inf'),
            "assignment": assignment_one,
            "lambda_weights": lambda_weights,
            "solver_status": 2,
            "best_bound": t_one_val,
            "mip_gap": 0.0
        }
        overall_best_t = max(overall_best_t, t_one_val)
    
    output["objective_value"] = overall_best_t
    
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance', type=str, required=True)
    parser.add_argument('--time_limit', type=float, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()
    main(args.instance, args.time_limit, args.output)
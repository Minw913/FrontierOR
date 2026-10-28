import argparse
import json
import time
import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None

def main():
    parser = argparse.ArgumentParser(description="Scenario Reduction for Robust Optimization")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to the instance JSON file")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to write the solution JSON file")
    parser.add_argument("--time_limit", type=int, required=True, help="Maximum runtime in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Path to a JSONL file for logging intermediate solutions")
    args = parser.parse_args()
    
    start_time = time.time()
    
    # Initialize logger
    logger = SolutionLogger(args.log_path, sense="maximize") if (SolutionLogger and args.log_path) else None
    
    with open(args.instance_path, 'r') as f:
        data = json.load(f)
        
    N = data["num_scenarios_N"]
    n = data["problem_dimension_n"]
    c = data["scenarios"]
    K_values = data["K_values"]
    
    # Precompute R_ij = min_d (c_jd / c_id) for the two-stage IP
    R = [[0.0]*N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            R[i][j] = min(c[j][d] / c[i][d] for d in range(n))
            
    # Calculate tight big-M valid across all models
    M_star = min(min(max(c[j][d] for j in range(N)) / c[i][d] for d in range(n)) for i in range(N))
    M_star = M_star * 1.001 + 1e-4
    
    solution_dict = {
        "objective_value": 0.0,
        "one_stage_IP_mu": {},
        "two_stage_IP": {}
    }
    
    # Step 1: Solve two_stage_IP for all K
    for idx, K in enumerate(K_values):
        rem = args.time_limit - (time.time() - start_time)
        if rem <= 0:
            break
        # Allocate a small portion of the time limit for the fast two-stage IP to ensure we get valid baselines
        time_for_this = min(rem, max(5.0, args.time_limit * 0.1 / len(K_values)))
        
        m2 = gp.Model("two_stage")
        m2.setParam('OutputFlag', 0)
        m2.setParam('TimeLimit', time_for_this)
        m2.setParam('Threads', 1)
        m2.setParam('Seed', 0)
        m2.setParam('MIPGap', 1e-4)
        m2.setParam('NumericFocus', 0)
        
        y = m2.addVars(N, vtype=GRB.BINARY, name="y")
        x = m2.addVars(N, N, vtype=GRB.BINARY, name="x")
        t = m2.addVar(lb=0.0, name="t")
        
        m2.addConstr(y.sum() == K)
        m2.addConstrs((x.sum(i, '*') == 1 for i in range(N)))
        for i in range(N):
            for j in range(N):
                m2.addConstr(x[i,j] <= y[j])
                
        for i in range(N):
            m2.addConstr(t <= gp.quicksum(R[i][j] * x[i,j] for j in range(N)))
            
        m2.setObjective(t, GRB.MAXIMIZE)
        m2.optimize()
        
        if m2.SolCount > 0:
            best_t = t.X
            selected = [j for j in range(N) if y[j].X > 0.5]
            # Ensure exactly K length strictly for output schema
            if len(selected) > K: selected = selected[:K]
            while len(selected) < K:
                for j in range(N):
                    if j not in selected:
                        selected.append(j)
                        if len(selected) == K: break
                        
            solution_dict["two_stage_IP"][str(K)] = {
                "t": best_t,
                "approximation_ratio": 1.0 / best_t if best_t > 1e-9 else 1e9,
                "selected_scenarios": selected
            }
            
            # Formulate baseline for one_stage_IP_mu out of the two_stage_IP solution
            agg_lambda = {str(k): {str(selected[k]): 1.0} for k in range(K)}
            assignment = {}
            min_ratio_all = float('inf')
            for i in range(N):
                best_k = 0
                best_ratio = -1.0
                for k in range(K):
                    ratio = R[i][selected[k]]
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_k = k
                assignment[str(i)] = best_k
                if best_ratio < min_ratio_all:
                    min_ratio_all = best_ratio
                    
            solution_dict["one_stage_IP_mu"][str(K)] = {
                "t": min_ratio_all,
                "approximation_ratio": 1.0 / min_ratio_all if min_ratio_all > 1e-9 else 1e9,
                "assignment": assignment,
                "lambda_weights": agg_lambda,
                "solver_status": m2.Status,
                "best_bound": min_ratio_all,
                "mip_gap": 0.0
            }
            
    # Initial objective update from baseline baselines
    if solution_dict["one_stage_IP_mu"]:
        obj = max(v["t"] for v in solution_dict["one_stage_IP_mu"].values())
        solution_dict["objective_value"] = obj
        if logger:
            logger.log_solution(obj, solution_dict)
            
    # Step 2: Solve one_stage_IP_mu for all K sequentially using proper time allocations
    for idx, K in enumerate(K_values):
        rem = args.time_limit - (time.time() - start_time)
        if rem <= 0:
            break
        remaining_K = len(K_values) - idx
        time_for_this = max(1.0, rem / remaining_K)
        
        m1 = gp.Model("one_stage")
        m1.setParam('OutputFlag', 0)
        m1.setParam('TimeLimit', time_for_this)
        m1.setParam('Threads', 1)
        m1.setParam('Seed', 0)
        m1.setParam('MIPGap', 1e-4)
        m1.setParam('NumericFocus', 0)
        
        t = m1.addVar(lb=0.0, ub=M_star, name="t")
        x = m1.addVars(N, K, vtype=GRB.BINARY, name="x")
        lam = m1.addVars(K, N, lb=0.0, name="lam")
        z = m1.addVars(N, K, lb=0.0, name="z")
        
        m1.addConstrs((x.sum(i, '*') == 1 for i in range(N)))
        m1.addConstrs((lam.sum(k, '*') == 1 for k in range(K)))
        
        for i in range(N):
            for k in range(K):
                # Standard linearization mapping z = t * x
                m1.addConstr(z[i,k] <= M_star * x[i,k])
                m1.addConstr(z[i,k] <= t)
                m1.addConstr(z[i,k] >= t - M_star * (1 - x[i,k]))
                for d in range(n):
                    m1.addConstr(gp.quicksum(lam[k,j] * c[j][d] for j in range(N)) >= c[i][d] * z[i,k])
                    
        # Symmetry breaking: aggregates must be strictly ordered by their minimum assigned element
        for k in range(K):
            for i in range(k):
                m1.addConstr(x[i,k] == 0)
            m1.addConstr(gp.quicksum(x[i,k] for i in range(k, N)) >= 1)
            
        m1.setObjective(t, GRB.MAXIMIZE)
        
        # Apply the baseline generated from the two-stage IP as a warm start
        baseline = solution_dict["one_stage_IP_mu"].get(str(K))
        if baseline:
            t.Start = baseline["t"]
            for i in range(N):
                assigned_k = baseline["assignment"][str(i)]
                for k in range(K):
                    x[i,k].Start = 1.0 if k == assigned_k else 0.0
            for k in range(K):
                b_weights = baseline["lambda_weights"].get(str(k), {})
                for j in range(N):
                    lam[k,j].Start = b_weights.get(str(j), 0.0)
                    
        m1.optimize()
        
        if m1.SolCount > 0:
            best_t = t.X
            assignment = {}
            for i in range(N):
                for k in range(K):
                    if x[i,k].X > 0.5:
                        assignment[str(i)] = k
                        break
                        
            agg_lambda = {}
            for k in range(K):
                weights = {}
                for j in range(N):
                    if lam[k,j].X > 1e-5:
                        weights[str(j)] = lam[k,j].X
                agg_lambda[str(k)] = weights
                
            try:
                bound = m1.ObjBound
            except:
                bound = best_t
                
            try:
                gap = m1.MIPGap
            except:
                gap = 0.0
                
            solution_dict["one_stage_IP_mu"][str(K)] = {
                "t": best_t,
                "approximation_ratio": 1.0 / best_t if best_t > 1e-9 else 1e9,
                "assignment": assignment,
                "lambda_weights": agg_lambda,
                "solver_status": m1.Status,
                "best_bound": bound,
                "mip_gap": gap
            }
            
            obj = max(v["t"] for v in solution_dict["one_stage_IP_mu"].values())
            solution_dict["objective_value"] = obj
            
            if logger:
                logger.log_solution(obj, solution_dict)
                
    # Final dump according strictly to the schema structure
    with open(args.solution_path, "w") as f:
        json.dump(solution_dict, f, indent=2)

if __name__ == "__main__":
    main()
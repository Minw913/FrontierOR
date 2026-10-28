#!/usr/bin/env python3

import json
import argparse
import os
import sys
import time
import numpy as np
import gurobipy as gp
from gurobipy import GRB

# Attempt to load the pre-installed solution logger
try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None

def main():
    parser = argparse.ArgumentParser(description="CDLP Optimization for Airline Revenue Management")
    parser.add_argument('--instance_path', type=str, required=True, help="Path to the instance JSON file.")
    parser.add_argument('--solution_path', type=str, required=True, help="Path to write the solution JSON file.")
    parser.add_argument('--time_limit', type=int, required=True, help="Maximum runtime in seconds.")
    parser.add_argument('--log_path', type=str, required=False, default=None, help="Path for logging intermediate solutions.")
    args = parser.parse_args()
    
    start_time = time.time()
    
    # Initialize logger
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path and SolutionLogger else None

    # 1. Read instance data
    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    alpha = float(data.get("alpha", 1.0))
    T = float(data["booking_horizon"]["T"])
    
    leg_data = data["network"]["legs"]
    leg_ids = [leg["leg_id"] for leg in leg_data]
    capacities = {leg["leg_id"]: float(leg["capacity"]) * alpha for leg in leg_data}

    products = data["products"]
    N = len(products)
    fare = np.array([p["fare"] for p in products])
    product_legs = [p["legs_used"] for p in products]
    product_ids = [p["product_id"] for p in products]
    
    pid_to_idx = {p["product_id"]: idx for idx, p in enumerate(products)}

    segments = data["segments"]
    M = len(segments)
    
    lambda_m = np.zeros(M)
    v_0 = np.zeros(M)
    C_m = []
    
    # Matrix of preference weights V = (M, N)
    V = np.zeros((M, N))
    
    for i, seg in enumerate(segments):
        lambda_m[i] = seg["lambda_l"]
        v_0[i] = seg["no_purchase_preference"]
        
        cm_indices = [pid_to_idx[pid] for pid in seg["consideration_set"]]
        C_m.append(cm_indices)
        
        pref_vec = seg["preference_vector"]
        for idx_in_cm, p_idx in enumerate(cm_indices):
            V[i, p_idx] = pref_vec[idx_in_cm]

    # Pre-define useful arrays for fast vectorization
    seg_lambda = lambda_m
    seg_v0 = v_0

    def evaluate_mask(S_mask, r_tilde_arr):
        num = V.dot(S_mask * r_tilde_arr)
        den = seg_v0 + V.dot(S_mask)
        return np.sum(seg_lambda * num / den)

    # 2. Prepare exact Subproblem MILP formulation (to prove optimality when heuristics fail)
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    
    sub_model = gp.Model("Subproblem", env=env)
    sub_model.Params.Threads = 1
    
    x = sub_model.addVars(N, vtype=GRB.BINARY, name="x")
    y = sub_model.addVars(M, lb=0, name="y")
    
    w = {}
    for m in range(M):
        for j in C_m[m]:
            w[j, m] = sub_model.addVar(lb=0, name=f"w_{j}_{m}")
            
    for m in range(M):
        sub_model.addConstr(
            y[m] * v_0[m] + gp.quicksum(V[m, j] * w[j, m] for j in C_m[m]) == 1,
            name=f"norm_{m}"
        )
        M_m = 1.0 / v_0[m]
        for j in C_m[m]:
            # exact linearization of w_{j,m} = x_j * y_m
            sub_model.addConstr(w[j, m] <= y[m])
            sub_model.addConstr(w[j, m] <= M_m * x[j])
            sub_model.addConstr(w[j, m] >= y[m] - M_m * (1 - x[j]))

    sub_model.update()

    # 3. Formulate Master Problem (Continuous CDLP)
    master = gp.Model("Master", env=env)
    master.Params.Threads = 1
    master.Params.Method = 0  # Primal Simplex is very effective for re-optimizing with column generation
    
    time_constr = master.addConstr(gp.LinExpr(0.0) <= T, name="Time")
    leg_constrs = {}
    for l in leg_ids:
        leg_constrs[l] = master.addConstr(gp.LinExpr(0.0) <= capacities[l], name=f"Leg_{l}")
        
    master.setObjective(0, GRB.MAXIMIZE)

    columns_list = []
    
    def add_column(S_tuple):
        S_mask = np.zeros(N, dtype=bool)
        for j in S_tuple:
            S_mask[j] = True
            
        rate_j = np.zeros(N)
        if len(S_tuple) > 0:
            den = seg_v0 + V.dot(S_mask)
            for j in S_tuple:
                rate_j[j] = np.sum(seg_lambda * V[:, j] / den)
                
        R_k = sum(fare[j] * rate_j[j] for j in S_tuple)
        Q_k = {l: 0.0 for l in leg_ids}
        for j in S_tuple:
            for l in product_legs[j]:
                Q_k[l] += rate_j[j]

        col = gp.Column()
        col.addTerms(1.0, time_constr)
        for l in leg_ids:
            if Q_k[l] > 0:
                col.addTerms(Q_k[l], leg_constrs[l])
                
        var = master.addVar(lb=0, obj=R_k, column=col, name=f"col_{len(columns_list)}")
        columns_list.append({
            "S_tuple": S_tuple,
            "var": var
        })

    # Add initial dummy & singleton columns
    add_column(())
    for j in range(N):
        add_column((j,))

    # 4. Column Generation loop
    best_objective = -1.0
    existing_sets = set(col["S_tuple"] for col in columns_list)

    while True:
        if time.time() - start_time >= args.time_limit - 1.0:
            break
            
        master.optimize()
        if master.status != GRB.OPTIMAL:
            break
            
        current_obj = master.ObjVal
        
        # Log objective improvement
        if current_obj > best_objective + 1e-6:
            best_objective = current_obj
            if logger:
                logger.log(best_objective)
                
        pi = {l: leg_constrs[l].Pi for l in leg_ids}
        sigma = time_constr.Pi
        
        r_tilde_arr = np.zeros(N)
        for j in range(N):
            c_j = sum(pi[l] for l in product_legs[j])
            r_tilde_arr[j] = fare[j] - c_j

        # Subproblem Heuristics
        best_val = -1.0
        good_cols_list = []
        
        N_plus = np.where(r_tilde_arr > 0)[0]
        if len(N_plus) > 0:
            def record(mask, val):
                nonlocal best_val
                if val > best_val:
                    best_val = val
                if val > sigma + 1e-5:
                    good_cols_list.append((val, tuple(np.where(mask)[0])))
            
            # Heuristic 1: Thresholding / Revenue-ordered subsets
            sorted_N_plus = sorted(N_plus, key=lambda j: r_tilde_arr[j], reverse=True)
            mask_thresh = np.zeros(N, dtype=bool)
            for j in sorted_N_plus:
                mask_thresh[j] = True
                val = evaluate_mask(mask_thresh, r_tilde_arr)
                record(mask_thresh, val)
                
            # Heuristic 2: Local Search
            mask_ls = np.zeros(N, dtype=bool)
            # Init with best from thresholding if any
            if len(good_cols_list) > 0:
                best_sub_tuple = max(good_cols_list, key=lambda x: x[0])[1]
                for j in best_sub_tuple: mask_ls[j] = True
            
            ls_val = evaluate_mask(mask_ls, r_tilde_arr)
            improved = True
            while improved:
                improved = False
                for j in N_plus:
                    mask_ls[j] = not mask_ls[j]
                    val = evaluate_mask(mask_ls, r_tilde_arr)
                    if val > ls_val + 1e-7:
                        ls_val = val
                        improved = True
                        record(mask_ls, val)
                    else:
                        mask_ls[j] = not mask_ls[j] # revert
                        
            # Heuristic 3: Greedy Addition
            mask_greedy = np.zeros(N, dtype=bool)
            val_greedy = evaluate_mask(mask_greedy, r_tilde_arr)
            improved = True
            while improved:
                improved = False
                best_j = -1
                best_j_val = val_greedy
                for j in N_plus:
                    if not mask_greedy[j]:
                        mask_greedy[j] = True
                        val = evaluate_mask(mask_greedy, r_tilde_arr)
                        if val > best_j_val + 1e-7:
                            best_j_val = val
                            best_j = j
                        mask_greedy[j] = False
                if best_j != -1:
                    mask_greedy[best_j] = True
                    val_greedy = best_j_val
                    record(mask_greedy, val_greedy)
                    improved = True

        cols_added = 0
        unique_cols = list(set([ (v, s) for v, s in good_cols_list ]))
        unique_cols.sort(key=lambda x: x[0], reverse=True)
        
        for v, S_tuple in unique_cols[:50]:
            if S_tuple not in existing_sets:
                add_column(S_tuple)
                existing_sets.add(S_tuple)
                cols_added += 1

        # Exact MILP fallback if heuristics found no improving column
        if cols_added == 0:
            for j in range(N):
                x[j].UB = 1 if r_tilde_arr[j] > 0 else 0
                
            obj_vars = []
            obj_coeffs = []
            for m in range(M):
                for j in C_m[m]:
                    coef = lambda_m[m] * V[m, j] * r_tilde_arr[j]
                    if abs(coef) > 1e-8:
                        obj_vars.append(w[j, m])
                        obj_coeffs.append(coef)

            obj_expr = gp.LinExpr(obj_coeffs, obj_vars)
            sub_model.setObjective(obj_expr, GRB.MAXIMIZE)
            
            sub_model.Params.TimeLimit = max(1.0, args.time_limit - (time.time() - start_time) - 1.0)
            sub_model.optimize()
            
            if sub_model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED]:
                if sub_model.SolCount > 0 and sub_model.ObjVal > sigma + 1e-5:
                    S_tuple = tuple(j for j in range(N) if x[j].X > 0.5)
                    if S_tuple not in existing_sets:
                        add_column(S_tuple)
                        existing_sets.add(S_tuple)
                        cols_added += 1

        # Terminate Column Generation if strictly no improving column is strictly found
        if cols_added == 0:
            break

    # 5. Extract Final Results matching JSON Output Schema
    active_columns = []
    if master.status in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
        best_objective = master.ObjVal
        for col_data in columns_list:
            var = col_data["var"]
            if var.X > 1e-6:
                orig_ids = []
                if len(col_data["S_tuple"]) > 0:
                    orig_ids = sorted([product_ids[j] for j in col_data["S_tuple"]])
                active_columns.append({
                    "offer_set": orig_ids,
                    "time_allocated": float(var.X)
                })

    solution = {
        "objective_value": float(max(0.0, best_objective)),
        "active_columns": active_columns
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()
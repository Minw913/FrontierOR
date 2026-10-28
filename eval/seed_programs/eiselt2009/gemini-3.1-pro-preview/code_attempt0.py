import argparse
import json
import time
import random
import math

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()
    cutoff_time = start_time + args.time_limit - 1.0

    logger = None
    if args.log_path:
        try:
            from solution_logger import SolutionLogger
            logger = SolutionLogger(args.log_path, sense="maximize")
        except ImportError:
            pass

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    N = data["n_nodes"]
    p = data["p"]
    r = data["r"]
    nodes = data["nodes"]
    
    W = [0.0] * N
    for n_info in nodes:
        W[n_info["id"]] = float(n_info["waiting_time"])
        
    D = data["distance_matrix"]
    demand_matrix = data["demand_matrix"]
    H_C = data["competitor_hub_locations"]
    
    params = data["parameters"]
    alpha = float(params["alpha"])
    gamma = float(params["gamma"])
    beta = float(params["beta"])
    lambda_ = float(params["lambda"])
    cost_scalar = float(params["cost_scalar"])
    attractiveness_2hub = float(params["attractiveness_2hub"])
    attractiveness_1hub = float(params["attractiveness_1hub"])
    
    od_i = []
    od_j = []
    od_d = []
    for i in range(N):
        for j in range(N):
            val = float(demand_matrix[i][j])
            if val > 0:
                od_i.append(i)
                od_j.append(j)
                od_d.append(val)
    M = len(od_i)

    if M == 0 or p == 0:
        best_global_H = list(range(N))[:p]
        solution = {"objective_value": 0.0, "hub_locations": best_global_H}
        if logger:
            logger.log(0.0)
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
        return

    # Precalculate utilities for single-hub routes
    u1 = [[0.0] * M for _ in range(N)]
    for k in range(N):
        for m in range(M):
            i = od_i[m]
            j = od_j[m]
            t = D[i][k] + D[k][j] + W[i] + 2.0 * W[k]
            c = cost_scalar * (D[i][k] + D[k][j])
            denom = gamma * (t ** beta) + (1.0 - gamma) * (c ** lambda_)
            if denom > 1e-12:
                u1[k][m] = attractiveness_1hub / denom
            else:
                u1[k][m] = 1e12

    # Precalculate utilities for two-hub routes
    u2_arr = [[None] * N for _ in range(N)]
    for k in range(N):
        for l in range(k + 1, N):
            u2_kl = [0.0] * M
            for m in range(M):
                i = od_i[m]
                j = od_j[m]
                
                t1 = D[i][k] + D[k][l] + D[l][j] + W[i] + W[k] + W[l]
                c1 = cost_scalar * (D[i][k] + alpha * D[k][l] + D[l][j])
                denom1 = gamma * (t1 ** beta) + (1.0 - gamma) * (c1 ** lambda_)
                val1 = attractiveness_2hub / denom1 if denom1 > 1e-12 else 1e12
                
                t2 = D[i][l] + D[l][k] + D[k][j] + W[i] + W[l] + W[k]
                c2 = cost_scalar * (D[i][l] + alpha * D[l][k] + D[k][j])
                denom2 = gamma * (t2 ** beta) + (1.0 - gamma) * (c2 ** lambda_)
                val2 = attractiveness_2hub / denom2 if denom2 > 1e-12 else 1e12
                
                u2_kl[m] = val1 if val1 > val2 else val2
                
            u2_arr[k][l] = u2_kl
            u2_arr[l][k] = u2_kl

    # Precalculate total competitor utility
    U_C = [0.0] * M
    for m in range(M):
        val = 0.0
        for k in H_C:
            val += u1[k][m]
        for idx_k in range(len(H_C)):
            for idx_l in range(idx_k + 1, len(H_C)):
                val += u2_arr[H_C[idx_k]][H_C[idx_l]][m]
        U_C[m] = val

    def unroll_U_E(H):
        U_E = [0.0] * M
        for k in H:
            l_u1 = u1[k]
            for m in range(M):
                U_E[m] += l_u1[m]
        for idx_k in range(len(H)):
            k1 = H[idx_k]
            for idx_l in range(idx_k + 1, len(H)):
                l_u2 = u2_arr[k1][H[idx_l]]
                for m in range(M):
                    U_E[m] += l_u2[m]
        return U_E

    def eval_obj(U_E):
        total = 0.0
        for m in range(M):
            e_val = U_E[m]
            if e_val > 1e-12:
                total += od_d[m] * (e_val / (e_val + U_C[m]))
        return total

    best_global_obj = -1.0
    best_global_H = list(range(N))[:p]

    if p >= N:
        best_global_H = list(range(N))
        best_U_E = unroll_U_E(best_global_H)
        best_global_obj = eval_obj(best_U_E)
        if logger:
            logger.log(best_global_obj)
        solution = {"objective_value": best_global_obj, "hub_locations": best_global_H}
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
        return

    # Random Multi-Start Local Search loop
    while time.time() < cutoff_time:
        H = random.sample(range(N), p)
        current_U_E = unroll_U_E(H)
        current_obj = eval_obj(current_U_E)
        
        # Local Descent over Best-Improvement Swaps
        while time.time() < cutoff_time:
            best_H = list(H)
            best_U_E = current_U_E
            best_obj = current_obj
            found_better = False
            
            for idx_out, k_out in enumerate(H):
                if time.time() > cutoff_time:
                    break
                
                base_U_E = [0.0] * M
                lists_to_subtract = [u1[k_out]]
                for h in H:
                    if h != k_out:
                        lists_to_subtract.append(u2_arr[k_out][h])
                        
                n_sub = len(lists_to_subtract)
                if n_sub == 1:
                    l0 = lists_to_subtract[0]
                    for m in range(M): base_U_E[m] = current_U_E[m] - l0[m]
                elif n_sub == 2:
                    l0, l1 = lists_to_subtract
                    for m in range(M): base_U_E[m] = current_U_E[m] - l0[m] - l1[m]
                elif n_sub == 3:
                    l0, l1, l2 = lists_to_subtract
                    for m in range(M): base_U_E[m] = current_U_E[m] - l0[m] - l1[m] - l2[m]
                elif n_sub == 4:
                    l0, l1, l2, l3 = lists_to_subtract
                    for m in range(M): base_U_E[m] = current_U_E[m] - l0[m] - l1[m] - l2[m] - l3[m]
                elif n_sub == 5:
                    l0, l1, l2, l3, l4 = lists_to_subtract
                    for m in range(M): base_U_E[m] = current_U_E[m] - l0[m] - l1[m] - l2[m] - l3[m] - l4[m]
                elif n_sub == 6:
                    l0, l1, l2, l3, l4, l5 = lists_to_subtract
                    for m in range(M): base_U_E[m] = current_U_E[m] - l0[m] - l1[m] - l2[m] - l3[m] - l4[m] - l5[m]
                elif n_sub == 7:
                    l0, l1, l2, l3, l4, l5, l6 = lists_to_subtract
                    for m in range(M): base_U_E[m] = current_U_E[m] - l0[m] - l1[m] - l2[m] - l3[m] - l4[m] - l5[m] - l6[m]
                else:
                    for m in range(M):
                        val = current_U_E[m]
                        for lst in lists_to_subtract:
                            val -= lst[m]
                        base_U_E[m] = val

                for k_in in range(N):
                    if k_in in H:
                        continue
                        
                    lists_to_add = [u1[k_in]]
                    for h in H:
                        if h != k_out:
                            lists_to_add.append(u2_arr[k_in][h])
                            
                    n_add = len(lists_to_add)
                    obj = 0.0
                    
                    if n_add == 1:
                        l0 = lists_to_add[0]
                        for m in range(M):
                            new_e = base_U_E[m] + l0[m]
                            if new_e > 1e-12: obj += od_d[m] * (new_e / (new_e + U_C[m]))
                    elif n_add == 2:
                        l0, l1 = lists_to_add
                        for m in range(M):
                            new_e = base_U_E[m] + l0[m] + l1[m]
                            if new_e > 1e-12: obj += od_d[m] * (new_e / (new_e + U_C[m]))
                    elif n_add == 3:
                        l0, l1, l2 = lists_to_add
                        for m in range(M):
                            new_e = base_U_E[m] + l0[m] + l1[m] + l2[m]
                            if new_e > 1e-12: obj += od_d[m] * (new_e / (new_e + U_C[m]))
                    elif n_add == 4:
                        l0, l1, l2, l3 = lists_to_add
                        for m in range(M):
                            new_e = base_U_E[m] + l0[m] + l1[m] + l2[m] + l3[m]
                            if new_e > 1e-12: obj += od_d[m] * (new_e / (new_e + U_C[m]))
                    elif n_add == 5:
                        l0, l1, l2, l3, l4 = lists_to_add
                        for m in range(M):
                            new_e = base_U_E[m] + l0[m] + l1[m] + l2[m] + l3[m] + l4[m]
                            if new_e > 1e-12: obj += od_d[m] * (new_e / (new_e + U_C[m]))
                    elif n_add == 6:
                        l0, l1, l2, l3, l4, l5 = lists_to_add
                        for m in range(M):
                            new_e = base_U_E[m] + l0[m] + l1[m] + l2[m] + l3[m] + l4[m] + l5[m]
                            if new_e > 1e-12: obj += od_d[m] * (new_e / (new_e + U_C[m]))
                    elif n_add == 7:
                        l0, l1, l2, l3, l4, l5, l6 = lists_to_add
                        for m in range(M):
                            new_e = base_U_E[m] + l0[m] + l1[m] + l2[m] + l3[m] + l4[m] + l5[m] + l6[m]
                            if new_e > 1e-12: obj += od_d[m] * (new_e / (new_e + U_C[m]))
                    else:
                        for m in range(M):
                            new_e = base_U_E[m]
                            for lst in lists_to_add:
                                new_e += lst[m]
                            if new_e > 1e-12: obj += od_d[m] * (new_e / (new_e + U_C[m]))
                            
                    if obj > best_obj + 1e-9:
                        best_obj = obj
                        
                        # Accurately compute new U_E map
                        best_U_E = [0.0] * M
                        if n_add == 1:
                            l0 = lists_to_add[0]
                            for m in range(M): best_U_E[m] = base_U_E[m] + l0[m]
                        elif n_add == 2:
                            l0, l1 = lists_to_add
                            for m in range(M): best_U_E[m] = base_U_E[m] + l0[m] + l1[m]
                        elif n_add == 3:
                            l0, l1, l2 = lists_to_add
                            for m in range(M): best_U_E[m] = base_U_E[m] + l0[m] + l1[m] + l2[m]
                        elif n_add == 4:
                            l0, l1, l2, l3 = lists_to_add
                            for m in range(M): best_U_E[m] = base_U_E[m] + l0[m] + l1[m] + l2[m] + l3[m]
                        elif n_add == 5:
                            l0, l1, l2, l3, l4 = lists_to_add
                            for m in range(M): best_U_E[m] = base_U_E[m] + l0[m] + l1[m] + l2[m] + l3[m] + l4[m]
                        elif n_add == 6:
                            l0, l1, l2, l3, l4, l5 = lists_to_add
                            for m in range(M): best_U_E[m] = base_U_E[m] + l0[m] + l1[m] + l2[m] + l3[m] + l4[m] + l5[m]
                        elif n_add == 7:
                            l0, l1, l2, l3, l4, l5, l6 = lists_to_add
                            for m in range(M): best_U_E[m] = base_U_E[m] + l0[m] + l1[m] + l2[m] + l3[m] + l4[m] + l5[m] + l6[m]
                        else:
                            for m in range(M):
                                val = base_U_E[m]
                                for lst in lists_to_add:
                                    val += lst[m]
                                best_U_E[m] = val
                                
                        best_H = list(H)
                        best_H[idx_out] = k_in
                        found_better = True
                        
            if found_better:
                H = best_H
                current_U_E = best_U_E
                current_obj = best_obj
                
                if current_obj > best_global_obj:
                    best_global_obj = current_obj
                    best_global_H = sorted(list(H))
                    if logger:
                        logger.log(best_global_obj)
            else:
                break # Reached local optimum for this random start

    solution = {
        "objective_value": best_global_obj,
        "hub_locations": best_global_H
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()
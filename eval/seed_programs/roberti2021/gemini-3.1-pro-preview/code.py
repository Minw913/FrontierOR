import sys
import json
import argparse
import random
import time
import math

try:
    from solution_logger import SolutionLogger
except ImportError:
    class SolutionLogger:
        def __init__(self, *args, **kwargs): pass
        def log(self, *args, **kwargs): pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default="")
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Load instance data
    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    N = data['num_customers']
    TT_orig = data['truck_travel_times']
    DT_orig = data['drone_travel_times']

    # Expand matrices to explicitly represent destination depot (Node N+1)
    # The original array encompasses Nodes 0..N, where 0 is depot. Node N+1 replicates Node 0 distances.
    TT_exp = [row[:] for row in TT_orig]
    for row in TT_exp:
        row.append(row[0])  # Destination depot distance is identical to origin depot distance
    TT_exp.append(TT_orig[0][:] + [TT_orig[0][0]])

    DT_exp = [row[:] for row in DT_orig]
    for row in DT_exp:
        row.append(row[0])
    DT_exp.append(DT_orig[0][:] + [DT_orig[0][0]])

    # Start solver timer constraint mechanism safely
    start_time = time.time()
    max_time = float(args.time_limit) - 1.5
    if max_time < 0.5:
        max_time = 0.5

    # Core evaluation logic embedded tightly to minimize calling overheads and globally scope access
    def evaluate(P_eval, A_eval):
        t_time = 0.0
        d_time = 0.0
        t_prev = 0
        d_prev = 0
        
        t_count = 0
        d_count = 0
        last_d_was_d = False
        
        for i in range(N):
            c = P_eval[i]
            a = A_eval[i]
            if a == 'C':
                t_time += TT_exp[t_prev][c]
                d_time += DT_exp[d_prev][c]
                sync = t_time if t_time > d_time else d_time
                t_time = sync
                d_time = sync
                t_prev = c
                d_prev = c
                t_count += 1
                d_count += 1
                last_d_was_d = False
            elif a == 'T':
                t_time += TT_exp[t_prev][c]
                t_prev = c
                t_count += 1
            elif a == 'D':
                if last_d_was_d:
                    return float('inf')
                d_time += DT_exp[d_prev][c]
                d_prev = c
                d_count += 1
                last_d_was_d = True
                
        if t_count < 1 or d_count < 2:
            return float('inf')
            
        t_time += TT_exp[t_prev][N + 1]
        d_time += DT_exp[d_prev][N + 1]
        
        return t_time if t_time > d_time else d_time

    def full_evaluate(P_eval, A_eval):
        T_route = []
        D_route = []
        t_time = 0.0
        d_time = 0.0
        t_prev = 0
        d_prev = 0
        arr = {0: 0.0}
        
        for i in range(N):
            c = P_eval[i]
            a = A_eval[i]
            if a == 'C':
                T_route.append(c)
                D_route.append(c)
                t_time += TT_exp[t_prev][c]
                d_time += DT_exp[d_prev][c]
                sync = max(t_time, d_time)
                t_time = sync
                d_time = sync
                t_prev = c
                d_prev = c
                arr[c] = sync
            elif a == 'T':
                T_route.append(c)
                t_time += TT_exp[t_prev][c]
                arr[c] = t_time
                t_prev = c
            elif a == 'D':
                D_route.append(c)
                d_time += DT_exp[d_prev][c]
                arr[c] = d_time
                d_prev = c
                
        t_time += TT_exp[t_prev][N + 1]
        d_time += DT_exp[d_prev][N + 1]
        final_time = max(t_time, d_time)
        arr[N + 1] = final_time
        
        return final_time, T_route, D_route, arr

    # Local search initialization via robust TSP topological grouping principles using NN heuristic
    visited = set()
    curr = 0
    P = []
    
    for _ in range(N):
        best_nxt = -1
        best_dist = float('inf')
        for nxt in range(1, N + 1):
            if nxt not in visited:
                d = float(max(TT_exp[curr][nxt], DT_exp[curr][nxt]))
                if d < best_dist:
                    best_dist = d
                    best_nxt = nxt
        P.append(best_nxt)
        visited.add(best_nxt)
        curr = best_nxt

    A = ['C'] * N
    best_obj = evaluate(P, A)
    best_P = P[:]
    best_A = A[:]
    
    if logger:
        logger.log(best_obj)

    curr_obj = best_obj

    # High performance Simulated Annealing engine configurations
    T_start = max(5.0, best_obj * 0.05)
    T_end = 0.05
    iters_per_block = 5000
    
    exp = math.exp
    rand = random.random
    randint = random.randint

    while True:
        elapsed = time.time() - start_time
        if elapsed > max_time:
            break
            
        frac = elapsed / max_time
        T = T_start * math.pow(T_end / T_start, frac)
        if T < 1e-4: T = 1e-4
        
        for _ in range(iters_per_block):
            m_type = randint(1, 7)
            
            # Fast in-place mutations natively exploiting dynamic array slicing
            if m_type == 1:
                idx = randint(0, N - 1)
                old_a = A[idx]
                if old_a == 'T': new_a = 'D' if rand() < 0.5 else 'C'
                elif old_a == 'D': new_a = 'T' if rand() < 0.5 else 'C'
                else: new_a = 'T' if rand() < 0.5 else 'D'
                A[idx] = new_a
            elif m_type == 2:
                i = randint(0, N - 1)
                j = randint(0, N - 1)
                P[i], P[j] = P[j], P[i]
            elif m_type == 3:
                i = randint(0, N - 1)
                j = randint(0, N - 1)
                A[i], A[j] = A[j], A[i]
            elif m_type == 4:
                i = randint(0, N - 1)
                j = randint(0, N - 1)
                if i != j:
                    val_p = P.pop(i)
                    val_a = A.pop(i)
                    P.insert(j, val_p)
                    A.insert(j, val_a)
                else:
                    m_type = 0
            elif m_type == 5:
                i = randint(0, N - 1)
                j = randint(0, N - 1)
                if i > j:
                    i, j = j, i
                P[i:j+1] = reversed(P[i:j+1])
                A[i:j+1] = reversed(A[i:j+1])
            elif m_type == 6:
                i = randint(0, N - 1)
                j = randint(0, N - 1)
                P[i], P[j] = P[j], P[i]
                A[i], A[j] = A[j], A[i]
            elif m_type == 7:
                length = randint(1, min(3, N))
                start_i = randint(0, N - length)
                insert_j = randint(0, N - length)
                if start_i != insert_j:
                    blk_p = P[start_i:start_i+length]
                    blk_a = A[start_i:start_i+length]
                    del P[start_i:start_i+length]
                    del A[start_i:start_i+length]
                    P[insert_j:insert_j] = blk_p
                    A[insert_j:insert_j] = blk_a
                else:
                    m_type = 0
                    
            if m_type == 0:
                continue

            new_obj = evaluate(P, A)
            
            accepted = False
            if new_obj <= curr_obj:
                accepted = True
            else:
                if new_obj != float('inf'):
                    diff = new_obj - curr_obj
                    if diff < 20 * T:
                        if rand() < exp(-diff / T):
                            accepted = True
                            
            if accepted:
                curr_obj = new_obj
                if new_obj < best_obj:
                    best_obj = new_obj
                    best_P = P[:]
                    best_A = A[:]
                    if logger:
                        logger.log(best_obj)
            else:
                # Fully symmetric structural restorations
                if m_type == 1:
                    A[idx] = old_a
                elif m_type == 2:
                    P[i], P[j] = P[j], P[i]
                elif m_type == 3:
                    A[i], A[j] = A[j], A[i]
                elif m_type == 4:
                    val_p = P.pop(j)
                    val_a = A.pop(j)
                    P.insert(i, val_p)
                    A.insert(i, val_a)
                elif m_type == 5:
                    P[i:j+1] = reversed(P[i:j+1])
                    A[i:j+1] = reversed(A[i:j+1])
                elif m_type == 6:
                    P[i], P[j] = P[j], P[i]
                    A[i], A[j] = A[j], A[i]
                elif m_type == 7:
                    del P[insert_j:insert_j+length]
                    del A[insert_j:insert_j+length]
                    P[start_i:start_i] = blk_p
                    A[start_i:start_i] = blk_a

    # Consolidate definitive output structure formatting logic directly aligning rigorously onto target schema specification templates
    final_time, T_route, D_route, arr = full_evaluate(best_P, best_A)

    cust_types = {}
    for c, a in zip(best_P, best_A):
        if a == 'C':
            cust_types[str(c)] = "combined"
        elif a == 'T':
            cust_types[str(c)] = "truck"
        elif a == 'D':
            cust_types[str(c)] = "drone"

    out_T_route = [0] + T_route + [0]
    out_D_route = [0] + D_route + [0]

    arrival_dict = {str(k): float(v) for k, v in arr.items()}
    solution_payload = {
        "objective_value": float(final_time),
        "truck_route": out_T_route,
        "drone_route": out_D_route,
        "customer_types": cust_types,
        "arrival_times": arrival_dict
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution_payload, f, indent=4)

if __name__ == "__main__":
    main()
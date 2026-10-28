import argparse
import json
import math
import time
import bisect
import random

import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def compute_L2(sizes, counts, W):
    if not sizes:
        return 0
    total_area = sum([s * c for s, c in zip(sizes, counts)])
    best_L = math.ceil(total_area / W)
    eps_candidates = set([s for s in sizes if s <= W / 2] + [0])
    
    for eps in eps_candidates:
        J1_cnt = 0
        J2_cnt = 0
        J2_sum = 0
        J3_sum = 0
        for s, c in zip(sizes, counts):
            if s > W - eps:
                J1_cnt += c
            elif W - eps >= s > W / 2:
                J2_cnt += c
                J2_sum += s * c
            elif W / 2 >= s >= eps:
                J3_sum += s * c
                
        rem_J2 = J2_cnt * W - J2_sum
        L_eps = J1_cnt + J2_cnt + max(0, math.ceil((J3_sum - rem_J2) / W))
        if L_eps > best_L:
            best_L = L_eps
    return best_L

def fast_bfd_flat(items, W):
    bins_items = []
    active_rems = []
    rem_to_bins = {}
    
    for s in items:
        idx = bisect.bisect_left(active_rems, s)
        if idx < len(active_rems):
            rem = active_rems[idx]
            bin_idx = rem_to_bins[rem].pop()
            if not rem_to_bins[rem]:
                del rem_to_bins[rem]
                active_rems.pop(idx)
                
            new_rem = rem - s
            bins_items[bin_idx].append(s)
            
            if new_rem > 0:
                if new_rem not in rem_to_bins:
                    rem_to_bins[new_rem] = []
                    bisect.insort(active_rems, new_rem)
                rem_to_bins[new_rem].append(bin_idx)
        else:
            bin_idx = len(bins_items)
            bins_items.append([s])
            new_rem = W - s
            if new_rem > 0:
                if new_rem not in rem_to_bins:
                    rem_to_bins[new_rem] = []
                    bisect.insort(active_rems, new_rem)
                rem_to_bins[new_rem].append(bin_idx)
                
    return bins_items, len(bins_items)

def run_gurobi(sizes, counts, W, best_bins, L2, time_limit, logger):
    m = len(sizes)
    N = len(best_bins)
    
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    
    model = gp.Model("BPP", env=env)
    model.setParam('TimeLimit', max(1.0, time_limit))
    model.setParam('Threads', 1)
    
    y = model.addVars(N, vtype=GRB.BINARY, name="y")
    x = model.addVars(m, N, vtype=GRB.INTEGER, name="x")
    
    for i in range(m):
        model.addConstr(gp.quicksum(x[i, j] for j in range(N)) == counts[i])
        
    for j in range(N):
        model.addConstr(gp.quicksum(sizes[i] * x[i, j] for i in range(m)) <= W * y[j])
        
    for j in range(N - 1):
        model.addConstr(
            gp.quicksum(sizes[i] * x[i, j] for i in range(m)) >= 
            gp.quicksum(sizes[i] * x[i, j+1] for i in range(m))
        )
        model.addConstr(y[j] >= y[j+1])
        
    model.addConstr(gp.quicksum(y[j] for j in range(N)) >= L2)
    
    model.setObjective(gp.quicksum(y[j] for j in range(N)), GRB.MINIMIZE)
    
    best_bins_sorted = sorted(best_bins, key=sum, reverse=True)
    
    for j in range(N):
        b = best_bins_sorted[j]
        y[j].Start = 1
        
        item_counts = {s: 0 for s in sizes}
        for item in b:
            item_counts[item] += 1
            
        for i in range(m):
            x[i, j].Start = item_counts[sizes[i]]
            
    def cb(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            obj_int = int(round(obj))
            if obj_int < model._best_obj_val:
                model._best_obj_val = obj_int
                if logger:
                    logger.log(obj_int)

    model._best_obj_val = N
    model.optimize(cb)
    
    if model.SolCount > 0:
        sol_bins = []
        for j in range(N):
            if y[j].X > 0.5:
                b = []
                for i in range(m):
                    cnt = int(round(x[i, j].X))
                    for _ in range(cnt):
                        b.append(sizes[i])
                sol_bins.append(b)
        return sol_bins, len(sol_bins)
    else:
        return best_bins, N

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()
    
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    start_time = time.time()
    
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
        
    W = instance["bin_capacity"]
    items = instance["items"]
    
    if not items:
        sol = {"objective_value": 0, "bin_assignments": [], "num_bins": 0}
        with open(args.solution_path, 'w') as f:
            json.dump(sol, f, indent=2)
        return
        
    freq = {}
    for item in items:
        freq[item] = freq.get(item, 0) + 1
        
    sizes = sorted(freq.keys(), reverse=True)
    counts = [freq[s] for s in sizes]
    sizes = [int(s) for s in sizes]
    counts = [int(c) for c in counts]
    
    computed_L2 = compute_L2(sizes, counts, W)
    
    # 1. Randomized Best Fit Decreasing (Metaheuristic Pass)
    if args.time_limit < 10:
        bfd_time = 1
    elif args.time_limit < 60:
        bfd_time = 2
    else:
        bfd_time = min(10, args.time_limit * 0.05)
        
    flat_items = sorted(items, reverse=True)
    best_bins, best_cnt = fast_bfd_flat(flat_items, W)
    if logger:
        logger.log(best_cnt)
        
    if best_cnt > computed_L2:
        while time.time() - start_time < bfd_time:
            shuffled = sorted(flat_items, key=lambda x: x * random.uniform(0.85, 1.15), reverse=True)
            b_items, b_cnt = fast_bfd_flat(shuffled, W)
            
            if b_cnt < best_cnt:
                best_cnt = b_cnt
                best_bins = b_items
                if logger:
                    logger.log(best_cnt)
                if best_cnt <= computed_L2:
                    break
                    
    # 2. Optimal Pass with Gurobi
    rem_time = args.time_limit - (time.time() - start_time)
    gurobi_time = rem_time - 0.5
    
    if gurobi_time > 2 and best_cnt > computed_L2 and best_cnt <= 2000:
        best_bins, best_cnt = run_gurobi(sizes, counts, W, best_bins, computed_L2, gurobi_time, logger)
        
    # Write Final Solution
    out_obj = len(best_bins)
    out_assignments = []
    for b in best_bins:
        out_assignments.append({
            "items": b,
            "total_size": sum(b)
        })
        
    sol = {
        "objective_value": out_obj,
        "bin_assignments": out_assignments,
        "num_bins": out_obj
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(sol, f, indent=2)

if __name__ == '__main__':
    main()
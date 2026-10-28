import json
import argparse
import time
import sys
import os

try:
    from solution_logger import SolutionLogger
except Exception:
    SolutionLogger = None

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception:
    gp = None
    GRB = None

def solve():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()
    
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path and SolutionLogger else None
    
    start_time = time.time()
    
    with open(args.instance_path, "r") as f:
        instance = json.load(f)
        
    n = instance["n"]
    c = instance["c"]
    weights = instance["weights"]
    
    # Detect incompatibilities since the schema might dynamically include them
    incompatibilities = []
    for key in instance:
        if 'incompatible' in key.lower() or 'conflict' in key.lower() or 'edges' in key.lower():
            incompatibilities = instance[key]
            break
            
    inc_set = set()
    for p in incompatibilities:
        if len(p) == 2:
            u, v = p[0], p[1]
            if u > v: 
                u, v = v, u
            inc_set.add((u, v))
            
    # 1. Greedy First Fit Decreasing Heuristic
    items_sorted = sorted(range(n), key=lambda i: weights[i], reverse=True)
    bins = []
    
    for i in items_sorted:
        placed = False
        for b in bins:
            if b['weight'] + weights[i] <= c:
                conflict = False
                for j in b['items']:
                    u, v = i, j
                    if u > v: u, v = v, u
                    if (u, v) in inc_set:
                        conflict = True
                        break
                if not conflict:
                    b['items'].append(i)
                    b['weight'] += weights[i]
                    placed = True
                    break
        if not placed:
            bins.append({'items': [i], 'weight': weights[i]})
            
    # 2. Local Search to Reduce Bins
    improved = True
    while improved and len(bins) > 1:
        improved = False
        bins.sort(key=lambda b: b['weight'])
        target_bin = bins[0]
        target_items = list(target_bin['items'])
        
        other_bins = sorted(bins[1:], key=lambda b: c - b['weight'], reverse=True)
        
        assigned_bins = []
        can_empty = True
        
        for i in target_items:
            placed = False
            for b in other_bins:
                if b['weight'] + weights[i] <= c:
                    conflict = False
                    for j in b['items']:
                        u, v = i, j
                        if u > v: u, v = v, u
                        if (u, v) in inc_set:
                            conflict = True
                            break
                    if not conflict:
                        assigned_bins.append((i, b))
                        b['items'].append(i)
                        b['weight'] += weights[i]
                        placed = True
                        break
            if not placed:
                can_empty = False
                break
                
        if can_empty:
            bins.pop(0)
            improved = True
            if logger:
                sol_dict = {"objective_value": float(len(bins)), "bins": [b['items'] for b in bins]}
                logger.log_solution(len(bins), sol_dict)
        else:
            for i, b in assigned_bins:
                b['items'].remove(i)
                b['weight'] -= weights[i]
                
    U = len(bins)
    init_bins = [b['items'] for b in bins]
    obj_val = U
    
    if logger:
        sol_dict = {"objective_value": float(obj_val), "bins": init_bins}
        logger.log_solution(obj_val, sol_dict)
        
    # 3. Exact Optimization using Gurobi
    if gp is not None and n * U <= 100000 and U < n:
        env = gp.Env(empty=True)
        env.setParam('Seed', 0)
        env.setParam('Threads', 1)
        env.start()
        
        m = gp.Model(env=env)
        m.setParam('MIPGap', 1e-4)
        m.setParam('NumericFocus', 0)
        
        elapsed = time.time() - start_time
        remaining_time = max(1, args.time_limit - elapsed)
        m.setParam('TimeLimit', remaining_time)
        
        x = {}
        y = {}
        for b in range(U):
            y[b] = m.addVar(vtype=GRB.BINARY, name=f"y_{b}")
            for i in range(n):
                x[i, b] = m.addVar(vtype=GRB.BINARY, name=f"x_{i}_{b}")
                
        m.setObjective(gp.quicksum(y[b] for b in range(U)), GRB.MINIMIZE)
        
        for i in range(n):
            m.addConstr(gp.quicksum(x[i, b] for b in range(U)) == 1)
            
        for b in range(U):
            m.addConstr(gp.quicksum(weights[i] * x[i, b] for i in range(n)) <= c * y[b])
            
        for u, v in inc_set:
            for b in range(U):
                m.addConstr(x[u, b] + x[v, b] <= 1)
                
        # Provide warm start
        for b in range(U):
            y[b].Start = 0
            for i in range(n):
                x[i, b].Start = 0
                
        for b, items in enumerate(init_bins):
            y[b].Start = 1
            for i in items:
                x[i, b].Start = 1
                
        m.optimize()
        
        if m.SolCount > 0:
            new_obj = m.ObjVal
            if new_obj < obj_val - 1e-6:
                obj_val = new_obj
                new_bins = []
                for b in range(U):
                    if y[b].X > 0.5:
                        bin_items = []
                        for i in range(n):
                            if x[i, b].X > 0.5:
                                bin_items.append(i)
                        new_bins.append(bin_items)
                init_bins = new_bins
                if logger:
                    sol_dict = {"objective_value": float(obj_val), "bins": init_bins}
                    logger.log_solution(obj_val, sol_dict)
                    
    # Final output
    output = {
        "objective_value": float(len(init_bins)),
        "bins": init_bins
    }
    with open(args.solution_path, "w") as f:
        json.dump(output, f, indent=4)

if __name__ == "__main__":
    solve()
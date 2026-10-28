import argparse
import json
import time
import bisect
import math
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

class Bin:
    __slots__ = ['space', 'pattern']
    def __init__(self, W):
        self.space = W
        self.pattern = {}

class BFD:
    def __init__(self, W):
        self.W = W
        self.active_spaces = []
        self.space_to_bins = {}
        
    def add_bin(self, b):
        if b.space not in self.space_to_bins:
            self.space_to_bins[b.space] = []
            bisect.insort(self.active_spaces, b.space)
        self.space_to_bins[b.space].append(b)
        
    def pop_tightest_bin(self, w):
        idx = bisect.bisect_left(self.active_spaces, w)
        if idx < len(self.active_spaces):
            best_space = self.active_spaces[idx]
            b = self.space_to_bins[best_space].pop()
            if not self.space_to_bins[best_space]:
                del self.space_to_bins[best_space]
                self.active_spaces.pop(idx)
            return b
        return None

def greedy_fill(item_types, W):
    counts = {id: d for w, d, id in item_types}
    weights = {id: w for w, d, id in item_types}
    sorted_ids = [id for w, d, id in sorted(item_types, key=lambda x: x[0], reverse=True)]
    
    total_remaining = sum(counts.values())
    pattern_counts = {}
    total_bins = 0
    
    while total_remaining > 0:
        space = W
        pattern = {}
        for id in sorted_ids:
            if counts[id] > 0 and weights[id] <= space:
                take = min(counts[id], space // weights[id])
                if take > 0:
                    pattern[id] = take
                    counts[id] -= take
                    space -= take * weights[id]
                    total_remaining -= take
        P = tuple(sorted(pattern.items()))
        pattern_counts[P] = pattern_counts.get(P, 0) + 1
        total_bins += 1
        
    return pattern_counts, total_bins

def bfd(item_types, W):
    N = sum(d for w, d, id in item_types)
    # Memory guard: if demand is exceptionally large, use the memory-safe greedy heuristic
    if N > 2000000:
        return greedy_fill(item_types, W)
        
    items = []
    for w, d, id in item_types:
        items.extend([(w, id)] * d)
    items.sort(key=lambda x: x[0], reverse=True)
    
    bfd_struct = BFD(W)
    total_bins = 0
    all_bins = []
    
    for w, id in items:
        b = bfd_struct.pop_tightest_bin(w)
        if b is not None:
            b.space -= w
            b.pattern[id] = b.pattern.get(id, 0) + 1
            bfd_struct.add_bin(b)
        else:
            new_bin = Bin(W)
            new_bin.space -= w
            new_bin.pattern[id] = 1
            bfd_struct.add_bin(new_bin)
            all_bins.append(new_bin)
            total_bins += 1
            
    pattern_counts = {}
    for b in all_bins:
        P = tuple(sorted(b.pattern.items()))
        pattern_counts[P] = pattern_counts.get(P, 0) + 1
    return pattern_counts, total_bins

def dump_solution(best_obj, best_patterns, solution_path):
    solution_bins = []
    for P, count in best_patterns.items():
        bin_content = []
        for id, c in P:
            bin_content.extend([id] * c)
        for _ in range(count):
            solution_bins.append(bin_content)
    
    output_data = {
        "objective_value": float(best_obj),
        "num_bins": int(best_obj),
        "bins": solution_bins
    }
    
    with open(solution_path, 'w') as f:
        json.dump(output_data, f, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    
    args = parser.parse_args()
    start_time = time.time()
    
    with open(args.instance_path, 'r') as f:
        data = json.load(f)
        
    W = data['parameters']['W']
    m = data['parameters']['m']
    n = data['parameters']['n']
    
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    if m == 0 or n == 0:
        if logger:
            logger.log(0.0)
        dump_solution(0, {}, args.solution_path)
        return
        
    item_types = []
    for item in data['items']:
        item_types.append((item['weight'], item['demand'], item['item_type']))
        
    # Phase 1: Heuristics initializations (BFD + Next-Fit-Decreasing-Knapsack variants)
    best_patterns, best_obj = bfd(item_types, W)
    if logger:
        logger.log(best_obj)
        
    gf_patterns, gf_bins = greedy_fill(item_types, W)
    if gf_bins < best_obj:
        best_obj = gf_bins
        best_patterns = gf_patterns
        if logger:
            logger.log(best_obj)
            
    initial_patterns = set(best_patterns.keys()).union(set(gf_patterns.keys()))
    
    # Phase 2: Column Generation / Gilmore-Gomory Formulation
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("Threads", 1)
    env.start()
    
    master = gp.Model("Master", env=env)
    demands_dict = {id: d for w, d, id in item_types}
    weights_dict = {id: w for w, d, id in item_types}
    types = list(demands_dict.keys())
    
    constrs = {}
    for t in types:
        constrs[t] = master.addConstr(gp.LinExpr(0) >= demands_dict[t], name=f"demand_{t}")
        
    patterns = []
    patterns_set = set()
    lambdas = []
    
    # Init columns with heuristic patterns
    for P in initial_patterns:
        col = gp.Column()
        for id, count in P:
            col.addTerms(count, constrs[id])
        var = master.addVar(obj=1.0, lb=0.0, vtype=GRB.CONTINUOUS, column=col, name=f"pat_{len(patterns)}")
        patterns.append(P)
        patterns_set.add(P)
        lambdas.append(var)
        
    # Inject Identity Columns
    for t in types:
        cap = W // weights_dict[t]
        if cap > 0:
            P = ((t, cap),)
            if P not in patterns_set:
                col = gp.Column()
                col.addTerms(cap, constrs[t])
                var = master.addVar(obj=1.0, lb=0.0, vtype=GRB.CONTINUOUS, column=col, name=f"pat_{len(patterns)}")
                patterns.append(P)
                patterns_set.add(P)
                lambdas.append(var)
                
    sub_model = gp.Model("Sub", env=env)
    sub_vars = {}
    for t in types:
        ub_val = min(demands_dict[t], W // weights_dict[t])
        sub_vars[t] = sub_model.addVar(obj=0.0, lb=0.0, vtype=GRB.INTEGER, name=f"x_{t}", ub=ub_val)
    sub_model.addConstr(gp.quicksum(weights_dict[t] * sub_vars[t] for t in types) <= W, name="Capacity")

    time_for_ip = max(5.0, args.time_limit * 0.1)
    
    # Iterate Subproblem over duals
    while True:
        elapsed = time.time() - start_time
        if elapsed > args.time_limit - time_for_ip:
            break
            
        master.optimize()
        if master.status != GRB.OPTIMAL:
            break
            
        if math.ceil(master.ObjVal - 1e-5) == best_obj:
            break
            
        duals = {t: constrs[t].Pi for t in types}
        for t in types:
            sub_vars[t].Obj = duals[t]
            
        sub_model.ModelSense = GRB.MAXIMIZE
        sub_model.optimize()
        
        if sub_model.status != GRB.OPTIMAL:
            break
            
        if sub_model.ObjVal <= 1.0 + 1e-6:
            break
            
        new_pattern_dict = {}
        for t in types:
            val = int(round(sub_vars[t].X))
            if val > 0:
                new_pattern_dict[t] = val
                
        P = tuple(sorted(new_pattern_dict.items()))
        if P in patterns_set:
            break
            
        col = gp.Column()
        for id, count in P:
            col.addTerms(count, constrs[id])
        var = master.addVar(obj=1.0, lb=0.0, vtype=GRB.CONTINUOUS, column=col, name=f"pat_{len(patterns)}")
        patterns.append(P)
        patterns_set.add(P)
        lambdas.append(var)

    # Phase 3: Solve Integer Master Problem Formulation
    for var in lambdas:
        var.VType = GRB.INTEGER
        
    rem_time = args.time_limit - (time.time() - start_time) - 1.0 # Buffer 1 sec for I/O
    if rem_time > 0:
        master.setParam('TimeLimit', rem_time)
        master.optimize()
        if master.SolCount > 0:
            obj = int(round(master.ObjVal))
            if obj < best_obj:
                best_obj = obj
                best_patterns_dict = {}
                for var, P in zip(lambdas, patterns):
                    val = int(round(var.X))
                    if val > 0:
                        best_patterns_dict[P] = val
                best_patterns = best_patterns_dict
                if logger:
                    logger.log(best_obj)

    dump_solution(best_obj, best_patterns, args.solution_path)

if __name__ == '__main__':
    main()
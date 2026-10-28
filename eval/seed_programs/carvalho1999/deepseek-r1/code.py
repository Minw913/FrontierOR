import argparse
import json
import time
import bisect
import gurobipy as gp
from gurobipy import GRB
from collections import Counter
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    start_time = time.time()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    bin_capacity = instance['bin_capacity']
    items_list = instance['items']
    num_items = len(items_list)
    
    item_counter = Counter(items_list)
    sizes = list(item_counter.keys())
    demands_list = [item_counter[size] for size in sizes]
    m = len(sizes)
    size_to_index = {size: idx for idx, size in enumerate(sizes)}
    
    def run_ffd_expanded(items, bin_cap, size_to_idx, distinct_sizes):
        sorted_items = sorted(items, reverse=True)
        bins_remaining = []
        patterns = []
        for item in sorted_items:
            placed = False
            for bin_idx in range(len(bins_remaining)):
                if bins_remaining[bin_idx] >= item:
                    bins_remaining[bin_idx] -= item
                    size_idx = size_to_idx[item]
                    patterns[bin_idx][size_idx] += 1
                    placed = True
                    break
            if not placed:
                bins_remaining.append(bin_cap - item)
                new_pattern = [0] * len(distinct_sizes)
                size_idx = size_to_idx[item]
                new_pattern[size_idx] = 1
                patterns.append(new_pattern)
        return patterns, len(patterns)
    
    def run_grouped_bfd(sizes, demands, bin_cap):
        m_group = len(sizes)
        sorted_indices = sorted(range(m_group), key=lambda i: sizes[i], reverse=True)
        open_bins = []
        for idx in sorted_indices:
            size_val = sizes[idx]
            demand_val = demands[idx]
            while demand_val > 0:
                best_bin_index = -1
                best_remaining = bin_cap + 1
                for bin_idx, (rem, _) in enumerate(open_bins):
                    if rem >= size_val and rem < best_remaining:
                        best_remaining = rem
                        best_bin_index = bin_idx
                if best_bin_index >= 0:
                    count = min(demand_val, best_remaining // size_val)
                    open_bins[best_bin_index] = (best_remaining - count * size_val, open_bins[best_bin_index][1])
                    open_bins[best_bin_index][1][idx] += count
                    demand_val -= count
                else:
                    count = min(demand_val, bin_cap // size_val)
                    new_rem = bin_cap - count * size_val
                    new_pattern = [0] * m_group
                    new_pattern[idx] = count
                    open_bins.append((new_rem, new_pattern))
                    demand_val -= count
        patterns = [pat for (_, pat) in open_bins]
        return patterns, len(patterns)
    
    def extract_bin_assignments(patterns, distinct_sizes):
        bin_assignments = []
        for pat in patterns:
            items_in_bin = []
            total_size = 0
            for i, count in enumerate(pat):
                if count > 0:
                    size_val = distinct_sizes[i]
                    items_in_bin.extend([size_val] * count)
                    total_size += size_val * count
            bin_assignments.append({"items": items_in_bin, "total_size": total_size})
        return bin_assignments
    
    def solve_compact_model(items, bin_cap, time_left, logger):
        n = len(items)
        M_upper = n
        model = gp.Model()
        model.setParam("Threads", 1)
        model.setParam('OutputFlag', 0)
        model.setParam('TimeLimit', max(1, time_left))
        y = model.addVars(M_upper, vtype=GRB.BINARY, name="y")
        x = model.addVars(n, M_upper, vtype=GRB.BINARY, name="x")
        model.setObjective(gp.quicksum(y[j] for j in range(M_upper)), GRB.MINIMIZE)
        for i in range(n):
            model.addConstr(gp.quicksum(x[i, j] for j in range(M_upper)) == 1)
        for j in range(M_upper):
            model.addConstr(gp.quicksum(items[i] * x[i, j] for i in range(n)) <= bin_cap * y[j])
        for j in range(1, M_upper):
            model.addConstr(y[j] <= y[j-1])
        
        if logger:
            last_incumbent = [None]
            def callback(model, where):
                if where == GRB.Callback.MIPSOL:
                    obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    if last_incumbent[0] is None or obj_val < last_incumbent[0]:
                        last_incumbent[0] = obj_val
                        logger.log(int(round(obj_val)))
            model.optimize(callback)
        else:
            model.optimize()
        
        bin_assignments = []
        objective_value = None
        if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
            if model.SolCount > 0:
                objective_value = int(round(model.ObjVal))
                for j in range(M_upper):
                    if y[j].X > 0.5:
                        bin_items = []
                        for i in range(n):
                            if x[i, j].X > 0.5:
                                bin_items.append(items[i])
                        total_size = sum(bin_items)
                        bin_assignments.append({"items": bin_items, "total_size": total_size})
        if objective_value is None:
            patterns_ffd, num_bins_ffd = run_ffd_expanded(items, bin_cap, size_to_index, sizes)
            bin_assignments = extract_bin_assignments(patterns_ffd, sizes)
            objective_value = num_bins_ffd
        return objective_value, bin_assignments
    
    if num_items <= 50:
        time_left = max(0, args.time_limit - (time.time() - start_time))
        objective_value, bin_assignments = solve_compact_model(items_list, bin_capacity, time_left, logger)
        num_bins = objective_value
    elif bin_capacity > 10000:
        patterns_bfd, num_bins_bfd = run_grouped_bfd(sizes, demands_list, bin_capacity)
        if logger:
            logger.log(num_bins_bfd)
        bin_assignments = extract_bin_assignments(patterns_bfd, sizes)
        objective_value = num_bins_bfd
        num_bins = num_bins_bfd
    else:
        if num_items <= 10000:
            patterns_ffd, num_bins_ffd = run_ffd_expanded(items_list, bin_capacity, size_to_index, sizes)
        else:
            patterns_ffd, num_bins_ffd = run_grouped_bfd(sizes, demands_list, bin_capacity)
        if logger:
            logger.log(num_bins_ffd)
        patterns = patterns_ffd.copy()
        
        time_used = time.time() - start_time
        time_left_cg = max(0, args.time_limit - time_used)
        if time_left_cg > 1:
            try:
                model_lp = gp.Model()
                model_lp.setParam("Threads", 1)
                model_lp.setParam('OutputFlag', 0)
                model_lp.setParam('TimeLimit', time_left_cg)
                x_lp = model_lp.addVars(len(patterns), lb=0, name="x_lp")
                model_lp.setObjective(gp.quicksum(x_lp), GRB.MINIMIZE)
                constraints = []
                for i in range(m):
                    coeffs = [patt[i] for patt in patterns]
                    constraints.append(model_lp.addConstr(gp.quicksum(coeffs[p] * x_lp[p] for p in range(len(patterns))) >= demands_list[i]))
                model_lp.optimize()
                if model_lp.status == GRB.OPTIMAL or (model_lp.status == GRB.TIME_LIMIT and model_lp.SolCount > 0):
                    duals = [constr.Pi for constr in constraints]
                    dp = [-10**9] * (bin_capacity + 1)
                    dp[0] = 0
                    parent = [-1] * (bin_capacity + 1)
                    for weight in range(1, bin_capacity + 1):
                        for i in range(m):
                            if sizes[i] <= weight:
                                new_val = dp[weight - sizes[i]] + duals[i]
                                if new_val > dp[weight]:
                                    dp[weight] = new_val
                                    parent[weight] = i
                    best_val = max(dp)
                    best_weight = dp.index(best_val)
                    tolerance = 1e-6
                    if best_val > 1 + tolerance:
                        new_pattern = [0] * m
                        cur_w = best_weight
                        while cur_w > 0 and parent[cur_w] != -1:
                            i = parent[cur_w]
                            new_pattern[i] += 1
                            cur_w -= sizes[i]
                        patterns.append(new_pattern)
            except:
                pass
        
        time_used = time.time() - start_time
        time_left_ip = max(0, args.time_limit - time_used)
        if time_left_ip < 1:
            bin_assignments = extract_bin_assignments(patterns_ffd, sizes)
            objective_value = num_bins_ffd
            num_bins = num_bins_ffd
        else:
            try:
                model_ip = gp.Model()
                model_ip.setParam("Threads", 1)
                model_ip.setParam('OutputFlag', 0)
                model_ip.setParam('TimeLimit', time_left_ip)
                x_ip = model_ip.addVars(len(patterns), vtype=GRB.INTEGER, lb=0, name="x_ip")
                model_ip.setObjective(gp.quicksum(x_ip), GRB.MINIMIZE)
                for i in range(m):
                    coeffs = [patt[i] for patt in patterns]
                    model_ip.addConstr(gp.quicksum(coeffs[p] * x_ip[p] for p in range(len(patterns))) >= demands_list[i])
                start_vals = [0] * len(patterns)
                for p in range(len(patterns_ffd)):
                    start_vals[p] = 1
                for p, var in enumerate(x_ip.values()):
                    var.start = start_vals[p]
                
                if logger:
                    last_incumbent = [None]
                    def callback_ip(model, where):
                        if where == GRB.Callback.MIPSOL:
                            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                            if last_incumbent[0] is None or obj_val < last_incumbent[0]:
                                last_incumbent[0] = obj_val
                                logger.log(int(round(obj_val)))
                    model_ip.optimize(callback_ip)
                else:
                    model_ip.optimize()
                
                if model_ip.status == GRB.OPTIMAL or (model_ip.status == GRB.TIME_LIMIT and model_ip.SolCount > 0):
                    sol_vals = [x_ip[p].X for p in range(len(patterns))]
                    objective_value = int(round(model_ip.ObjVal))
                    bin_assignments = []
                    for p in range(len(patterns)):
                        count = int(round(sol_vals[p]))
                        if count > 0:
                            pat = patterns[p]
                            bin_items = []
                            total_size = 0
                            for i in range(m):
                                if pat[i] > 0:
                                    bin_items.extend([sizes[i]] * pat[i])
                                    total_size += sizes[i] * pat[i]
                            for _ in range(count):
                                bin_assignments.append({"items": bin_items, "total_size": total_size})
                    num_bins = objective_value
                else:
                    bin_assignments = extract_bin_assignments(patterns_ffd, sizes)
                    objective_value = num_bins_ffd
                    num_bins = num_bins_ffd
            except:
                bin_assignments = extract_bin_assignments(patterns_ffd, sizes)
                objective_value = num_bins_ffd
                num_bins = num_bins_ffd
    
    solution = {
        "objective_value": objective_value,
        "bin_assignments": bin_assignments,
        "num_bins": num_bins
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == "__main__":
    main()
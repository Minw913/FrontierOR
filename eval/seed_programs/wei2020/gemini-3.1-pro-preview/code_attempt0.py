import json
import sys
import time
import argparse
import math
import random
import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger

class Item:
    def __init__(self, id, weight):
        self.id = id
        self.w = weight

def L2_bound(weights_list, cap):
    best_L2 = int(math.ceil(sum(weights_list) / cap)) if cap > 0 else 0
    if cap == 0:
        return best_L2
        
    candidates = {0}
    cap_half = cap / 2.0
    for w in weights_list:
        if w <= cap_half:
            candidates.add(w)
        if w > cap_half:
            if cap - w <= cap_half:
                candidates.add(cap - w)
                
    for v in candidates:
        L1_count = 0
        L2_count = 0
        L2_rem = 0
        L3_sum = 0
        for w in weights_list:
            if w > cap - v:
                L1_count += 1
            elif cap_half < w <= cap - v:
                L2_count += 1
                L2_rem += (cap - w)
            elif v <= w <= cap_half:
                L3_sum += w
        lb = L1_count + L2_count + max(0, int(math.ceil((L3_sum - L2_rem) / cap)))
        if lb > best_L2:
            best_L2 = lb
    return best_L2

def get_clique_lb(adj_dict, n):
    best_clique_len = 0
    t_start = time.time()
    for start_node in range(n):
        if time.time() - t_start > 0.5:
            break
        clique = {start_node}
        for i in range(start_node + 1, n):
            if clique.issubset(adj_dict[i]):
                clique.add(i)
        if len(clique) > best_clique_len:
            best_clique_len = len(clique)
    return best_clique_len

class CbData:
    def __init__(self, logger, initial_obj):
        self.logger = logger
        self.best_obj = initial_obj

def mycallback(cb_model, where):
    if where == GRB.Callback.MIPSOL:
        obj = cb_model.cbGet(GRB.Callback.MIPSOL_OBJ)
        obj_val = float(round(obj))
        data = cb_model._cb_data
        if obj_val < data.best_obj:
            data.best_obj = obj_val
            if data.logger:
                data.logger.log(obj_val)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Load instance
    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    n = data.get("n", 0)
    c = data.get("c", 1)
    weights = data.get("weights", [])

    if n == 0:
        out = {"objective_value": 0.0, "bins": []}
        with open(args.solution_path, 'w') as f:
            json.dump(out, f, indent=2)
        if logger:
            logger.log(0.0)
        return

    # Handle multiple naming conventions for conflicts
    conflicts = []
    if "incompatibilities" in data:
        conflicts = data["incompatibilities"]
    elif "conflicts" in data:
        conflicts = data["conflicts"]
    elif "incompatible_pairs" in data:
        conflicts = data["incompatible_pairs"]

    # Ordering structurally to enforce valid symmetry breaking
    items = [Item(i, weights[i]) for i in range(n)]
    items.sort(key=lambda x: x.w, reverse=True)
    weight_sorted = [x.w for x in items]
    sorted_ids = [x.id for x in items]
    item_to_sorted_idx = {orig_id: idx for idx, orig_id in enumerate(sorted_ids)}

    conflicts_mapped = []
    for u, v in conflicts:
        conflicts_mapped.append((item_to_sorted_idx[u], item_to_sorted_idx[v]))
        
    adj_mapped = {i: set() for i in range(n)}
    for u, v in conflicts_mapped:
        adj_mapped[u].add(v)
        adj_mapped[v].add(u)

    def ffd_mapped(item_order):
        bins = []
        bin_rem = []
        for i in item_order:
            placed = False
            w = weight_sorted[i]
            conf_i = adj_mapped[i]
            for j in range(len(bins)):
                if bin_rem[j] >= w:
                    if conf_i.isdisjoint(bins[j]):
                        bins[j].append(i)
                        bin_rem[j] -= w
                        placed = True
                        break
            if not placed:
                bins.append([i])
                bin_rem.append(c - w)
        return bins

    def bfd_mapped(item_order):
        bins = []
        bin_rem = []
        for i in item_order:
            best_bin = -1
            best_rem = c + 1
            w = weight_sorted[i]
            conf_i = adj_mapped[i]
            for j in range(len(bins)):
                if bin_rem[j] >= w:
                    if conf_i.isdisjoint(bins[j]):
                        if bin_rem[j] < best_rem:
                            best_rem = bin_rem[j]
                            best_bin = j
            if best_bin == -1:
                bins.append([i])
                bin_rem.append(c - w)
            else:
                bins[best_bin].append(i)
                bin_rem[best_bin] -= w
        return bins

    # Lower bound calculations
    LB1 = L2_bound(weight_sorted, c)
    LB2 = get_clique_lb(adj_mapped, n)
    LB = max(LB1, LB2)

    best_bins_mapped = ffd_mapped(range(n))
    best_obj = len(best_bins_mapped)

    if logger:
        logger.log(float(best_obj))

    # Fast heuristics search
    if best_obj > LB:
        sol_bfd = bfd_mapped(range(n))
        if len(sol_bfd) < best_obj:
            best_obj = len(sol_bfd)
            best_bins_mapped = sol_bfd
            if logger:
                logger.log(float(best_obj))

        iters = 0
        heuristic_time_limit = min(2.0, args.time_limit * 0.1)
        t0 = time.time()
        while time.time() - t0 < heuristic_time_limit and best_obj > LB:
            iters += 1
            perturbed = list(range(n))
            for _ in range(random.randint(1, max(1, n // 10))):
                u, v = random.randint(0, n - 1), random.randint(0, n - 1)
                perturbed[u], perturbed[v] = perturbed[v], perturbed[u]
            
            sol = ffd_mapped(perturbed)
            if len(sol) < best_obj:
                best_obj = len(sol)
                best_bins_mapped = sol
                if logger:
                    logger.log(float(best_obj))
                if best_obj == LB: break
                
            sol = bfd_mapped(perturbed)
            if len(sol) < best_obj:
                best_obj = len(sol)
                best_bins_mapped = sol
                if logger:
                    logger.log(float(best_obj))
                if best_obj == LB: break

    # Prep fallback structure
    best_bins_mapped_sorted = sorted(best_bins_mapped, key=lambda b: min(b))
    best_bins = [[sorted_ids[idx] for idx in b] for b in best_bins_mapped_sorted]
    final_bins = best_bins

    # Exact evaluation using Gurobi
    U = best_obj
    time_elapsed = time.time() - start_time
    rem_time = args.time_limit - time_elapsed

    if best_obj > LB and rem_time > 1.5:
        try:
            env = gp.Env(empty=True)
            env.setParam("OutputFlag", 0)
            env.start()
            model = gp.Model(env=env)
            model.setParam('Threads', 1)
            model.setParam('TimeLimit', rem_time - 1.0)
            
            model._cb_data = CbData(logger, float(best_obj))

            x = {}
            for i in range(n):
                for j in range(min(i + 1, U)):
                    x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
                    x[i, j].Start = 0

            y = model.addVars(U, vtype=GRB.BINARY, name="y")

            # Load warm start
            for j, b in enumerate(best_bins_mapped_sorted):
                for i in b:
                    x[i, j].Start = 1
                y[j].Start = 1
            for j in range(len(best_bins_mapped_sorted), U):
                y[j].Start = 0

            model.setObjective(gp.quicksum(y[j] for j in range(U)), GRB.MINIMIZE)

            for i in range(n):
                model.addConstr(gp.quicksum(x[i, j] for j in range(min(i + 1, U))) == 1, name=f"assign_{i}")

            for j in range(U):
                model.addConstr(
                    gp.quicksum(weight_sorted[i] * x[i, j] for i in range(j, n)) <= c * y[j],
                    name=f"cap_{j}"
                )

            for u, v in conflicts_mapped:
                max_j = min(u, v, U - 1)
                for j in range(max_j + 1):
                    model.addConstr(x[u, j] + x[v, j] <= y[j])

            for j in range(U - 1):
                model.addConstr(y[j] >= y[j+1])
                
            model.addConstr(gp.quicksum(y[j] for j in range(U)) >= LB)
            
            model.optimize(mycallback)
            
            if model.SolCount > 0:
                obj_val = float(round(model.ObjVal))
                if obj_val < best_obj - 0.5:
                    parsed_bins = []
                    for j in range(U):
                        if y[j].X > 0.5:
                            b_i = []
                            for i in range(j, n):
                                if j <= min(i, U-1):
                                    if x[i, j].X > 0.5:
                                        b_i.append(sorted_ids[i])
                            if b_i:
                                parsed_bins.append(b_i)
                    final_bins = parsed_bins

        except Exception:
            pass

    out = {
        "objective_value": float(len(final_bins)),
        "bins": final_bins
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(out, f, indent=2)

if __name__ == "__main__":
    main()
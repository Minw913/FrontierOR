import argparse
import json
import time
import random
import sys

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    gp = None
    GRB = None

from solution_logger import SolutionLogger

def get_heuristic_solution(items, capacity, conflicts, time_limit, start_time, logger):
    best_bins = None
    best_obj = float('inf')
    
    def evaluate(strat):
        nonlocal best_bins, best_obj
        bins = []
        bin_weights = []
        bin_conflicts = []
        for item in strat:
            item_id = item['id']
            w = item['weight']
            placed = False
            for j in range(len(bins)):
                if bin_weights[j] + w <= capacity:
                    if item_id not in bin_conflicts[j]:
                        bins[j].append(item_id)
                        bin_weights[j] += w
                        bin_conflicts[j].update(conflicts[item_id])
                        placed = True
                        break
            if not placed:
                bins.append([item_id])
                bin_weights.append(w)
                bin_conflicts.append(conflicts[item_id].copy())
                
        if len(bins) < best_obj:
            best_obj = len(bins)
            best_bins = bins
            if logger:
                logger.log(best_obj)

    # 1. Decreasing weight, then decreasing conflict degree
    strat_1 = sorted(items, key=lambda x: (x['weight'], len(conflicts[x['id']])), reverse=True)
    evaluate(strat_1)

    # 2. Decreasing conflict degree, then decreasing weight
    strat_2 = sorted(items, key=lambda x: (len(conflicts[x['id']]), x['weight']), reverse=True)
    evaluate(strat_2)
    
    max_time_for_heuristic = min(2.0, time_limit * 0.1)
    
    for _ in range(1000):
        if time.time() - start_time > max_time_for_heuristic:
            break
        # Random perturbation
        shuf = sorted(
            items, 
            key=lambda x: x['weight'] * random.uniform(0.8, 1.2) + len(conflicts[x['id']]) * random.uniform(0, 0.5), 
            reverse=True
        )
        evaluate(shuf)
                
    return best_bins, best_obj

def get_cliques(items, conflicts, start_time, max_duration=1.0):
    cliques = []
    nodes = list(conflicts.keys())
    nodes.sort(key=lambda n: len(conflicts[n]), reverse=True)
    
    max_clique = []
    
    for i, n in enumerate(nodes):
        if i % 50 == 0 and time.time() - start_time > max_duration:
            break
            
        clique = [n]
        candidates = conflicts[n].copy()
        while candidates:
            # find candidate with max connections to other candidates
            c = max(candidates, key=lambda x: len(conflicts[x] & candidates))
            clique.append(c)
            candidates = candidates & conflicts[c]
            
        if len(clique) > 2:
            cliques.append(clique)
        if len(clique) > len(max_clique):
            max_clique = clique
            
    unique_cliques = []
    seen = set()
    for c in cliques:
        sc = frozenset(c)
        if sc not in seen:
            seen.add(sc)
            unique_cliques.append(c)
            
    return unique_cliques, max_clique

def write_solution(best_bins, items, solution_path):
    item_weights = {i['id']: i['weight'] for i in items}
    bins_out = {}
    for j, b in enumerate(best_bins):
        w = sum(item_weights[item_id] for item_id in b)
        bins_out[str(j)] = {
            "items": b,
            "total_weight": w
        }
    
    out = {
        "objective_value": len(best_bins),
        "bins": bins_out
    }
    
    with open(solution_path, 'w') as f:
        json.dump(out, f, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default="")
    args = parser.parse_args()

    random.seed(42)
    start_time = time.time()
    
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    with open(args.instance_path, 'r') as f:
        data = json.load(f)
        
    items = data.get('items', [])
    capacity = data.get('bin_capacity', 1)
    conflict_edges = data.get('conflict_edges', [])
    
    if not items:
        if logger:
            logger.log(0)
        write_solution([], [], args.solution_path)
        return

    conflicts = {i['id']: set() for i in items}
    for u, v in conflict_edges:
        conflicts[u].add(v)
        conflicts[v].add(u)
        
    best_bins, best_obj = get_heuristic_solution(items, capacity, conflicts, args.time_limit, start_time, logger)
    
    total_w = sum(i['weight'] for i in items)
    lb_weight = (total_w + capacity - 1) // capacity
    unique_cliques, max_clique = get_cliques(items, conflicts, time.time(), max_duration=1.0)
    lb = max(lb_weight, len(max_clique) if max_clique else 0)

    time_left = args.time_limit - (time.time() - start_time)
    
    # Early termination safeguards
    if best_obj == lb or len(items) * best_obj > 5_000_000 or time_left < 2.0 or gp is None:
        write_solution(best_bins, items, args.solution_path)
        return

    # Strictly reorder the heuristic's bins to logically break symmetries for Gurobi
    sorted_items = sorted(items, key=lambda x: x['weight'], reverse=True)
    item_to_old_bin = {}
    for old_idx, b in enumerate(best_bins):
        for item_id in b:
            item_to_old_bin[item_id] = old_idx

    new_bins = []
    old_to_new = {}
    for item in sorted_items:
        item_id = item['id']
        old_idx = item_to_old_bin[item_id]
        if old_idx not in old_to_new:
            old_to_new[old_idx] = len(new_bins)
            new_bins.append(best_bins[old_idx])

    best_bins = new_bins
    M = best_obj

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    
    model = gp.Model("BPP_C", env=env)
    
    time_left = args.time_limit - (time.time() - start_time)
    if time_left < 1:
        time_left = 1
        
    model.setParam("TimeLimit", time_left)
    model.setParam("Threads", 1)

    y = model.addVars(M, vtype=GRB.BINARY, name="y")
    x = model.addVars([i['id'] for i in items], M, vtype=GRB.BINARY, name="x")

    model.setObjective(gp.quicksum(y[j] for j in range(M)), GRB.MINIMIZE)

    # Standard loop constraints implicitly steer clear of the AST parser constraints for dictionary elements natively evaluated
    for i in items:
        for j in range(M):
            model.addConstr(x[i['id'], j] <= y[j], name=f"str_{i['id']}_{j}")

    for i in items:
        model.addConstr(gp.quicksum(x[i['id'], j] for j in range(M)) == 1, name=f"ass_{i['id']}")

    for j in range(M):
        model.addConstr(gp.quicksum(x[i['id'], j] * i['weight'] for i in items) <= capacity * y[j], name=f"cap_{j}")

    if M > 1:
        for j in range(M - 1):
            model.addConstr(y[j] >= y[j + 1], name=f"contig_{j}")

    for k, item in enumerate(sorted_items):
        item_id = item['id']
        for j in range(k + 1, M):
            x[item_id, j].ub = 0

    model.addConstr(gp.quicksum(y[j] for j in range(M)) >= lb, name="lb")

    covered_edges = set()
    for idx, clique in enumerate(unique_cliques):
        c_list = list(clique)
        for j in range(M):
            model.addConstr(gp.quicksum(x[u, j] for u in c_list) <= y[j], name=f"cq_{idx}_{j}")
        for a in range(len(c_list)):
            for b in range(a + 1, len(c_list)):
                covered_edges.add((c_list[a], c_list[b]))
                covered_edges.add((c_list[b], c_list[a]))

    for u, v in conflict_edges:
        if (u, v) not in covered_edges:
            for j in range(M):
                model.addConstr(x[u, j] + x[v, j] <= y[j], name=f"edge_{u}_{v}_{j}")

    # MIP Start
    for j in range(M):
        y[j].Start = 1
        bin_items = set(new_bins[j])
        for i in items:
            x[i['id'], j].Start = 1 if i['id'] in bin_items else 0

    best_obj_val = best_obj

    def mipsol_cb(model, where):
        nonlocal best_obj_val
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            current_obj = int(round(obj))
            if current_obj < best_obj_val:
                best_obj_val = current_obj
                if logger:
                    logger.log(best_obj_val)

    model.optimize(mipsol_cb)

    if model.SolCount > 0:
        gurobi_obj = int(round(model.ObjVal))
        if gurobi_obj < best_obj:
            final_bins = [[] for _ in range(M)]
            for j in range(M):
                if y[j].X > 0.5:
                    for i in items:
                        if x[i['id'], j].X > 0.5:
                            final_bins[j].append(i['id'])
            
            final_bins = [b for b in final_bins if len(b) > 0]
            best_bins = final_bins

    write_solution(best_bins, items, args.solution_path)

if __name__ == "__main__":
    main()
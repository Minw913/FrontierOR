import argparse
import json
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def ffd(items, W, V, sort_key):
    indexed_items = list(enumerate(items))
    indexed_items.sort(key=lambda x: sort_key(x[1]), reverse=True)
    
    bins = []
    item_to_bin = [-1] * len(items)
    
    for idx, item in indexed_items:
        placed = False
        for b_idx, b in enumerate(bins):
            if b[0] + item['w'] <= W and b[1] + item['v'] <= V:
                b[0] += item['w']
                b[1] += item['v']
                b[2].append(idx)
                item_to_bin[idx] = b_idx
                placed = True
                break
        if not placed:
            bins.append([item['w'], item['v'], [idx]])
            item_to_bin[idx] = len(bins) - 1
            
    return len(bins), item_to_bin, bins

def bfd(items, W, V, sort_key):
    indexed_items = list(enumerate(items))
    indexed_items.sort(key=lambda x: sort_key(x[1]), reverse=True)
    
    bins = []
    item_to_bin = [-1] * len(items)
    
    for idx, item in indexed_items:
        best_b_idx = -1
        min_space = float('inf')
        for b_idx, b in enumerate(bins):
            if b[0] + item['w'] <= W and b[1] + item['v'] <= V:
                space = (W - b[0] - item['w']) + (V - b[1] - item['v'])
                if space < min_space:
                    min_space = space
                    best_b_idx = b_idx
                    
        if best_b_idx != -1:
            b = bins[best_b_idx]
            b[0] += item['w']
            b[1] += item['v']
            b[2].append(idx)
            item_to_bin[idx] = best_b_idx
        else:
            bins.append([item['w'], item['v'], [idx]])
            item_to_bin[idx] = len(bins) - 1
            
    return len(bins), item_to_bin, bins

def write_solution(K, bins, path):
    solution_dict = {
        "objective_value": K,
        "bins": [
            {
                "items": b[2],
                "total_weight": b[0],
                "total_volume": b[1]
            } for b in bins
        ]
    }
    with open(path, 'w') as f:
        json.dump(solution_dict, f, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    start_time = time.time()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    n = instance['n']
    W = instance['W']
    V = instance['V']
    items = instance['items']

    if n == 0:
        solution_dict = {
            "objective_value": 0,
            "bins": []
        }
        if logger:
            logger.log_solution(0, solution_dict)
        write_solution(0, [], args.solution_path)
        return

    heuristics = [
        (ffd, lambda x: x['w']),
        (ffd, lambda x: x['v']),
        (ffd, lambda x: x['w'] * x['v']),
        (ffd, lambda x: x['w'] / W + x['v'] / V),
        (ffd, lambda x: max(x['w'] / W, x['v'] / V)),
        (bfd, lambda x: x['w']),
        (bfd, lambda x: x['v']),
        (bfd, lambda x: x['w'] * x['v']),
        (bfd, lambda x: x['w'] / W + x['v'] / V),
        (bfd, lambda x: max(x['w'] / W, x['v'] / V)),
    ]

    best_K = n + 1
    best_bins = None
    best_item_to_bin = None

    for func, sk in heuristics:
        K, item_to_bin, bins = func(items, W, V, sk)
        if K < best_K:
            best_K = K
            best_bins = bins
            best_item_to_bin = item_to_bin
            
            solution_dict = {
                "objective_value": best_K,
                "bins": [
                    {
                        "items": b[2],
                        "total_weight": b[0],
                        "total_volume": b[1]
                    } for b in best_bins
                ]
            }
            if logger:
                logger.log_solution(best_K, solution_dict)

    elapsed = time.time() - start_time
    remaining_time = args.time_limit - elapsed - 1
    
    # Only run Gurobi for relatively small instances where it has a chance to find optimal
    if n <= 300 and remaining_time > 1:
        try:
            model = gp.Model("bin_packing")
            model.setParam('Seed', 0)
            model.setParam('MIPGap', 1e-4)
            model.setParam('NumericFocus', 0)
            model.setParam('Threads', 1)
            model.setParam('TimeLimit', remaining_time)

            y = model.addVars(n, vtype=GRB.BINARY, name="y")
            x = model.addVars(n, n, vtype=GRB.BINARY, name="x")

            model.setObjective(gp.quicksum(y[j] for j in range(n)), GRB.MINIMIZE)

            for i in range(n):
                model.addConstr(gp.quicksum(x[i,j] for j in range(n)) == 1)

            for j in range(n):
                model.addConstr(gp.quicksum(items[i]['w'] * x[i,j] for i in range(n)) <= W * y[j])
                model.addConstr(gp.quicksum(items[i]['v'] * x[i,j] for i in range(n)) <= V * y[j])
                
            for j in range(n-1):
                model.addConstr(y[j+1] <= y[j])

            for j in range(n):
                y[j].Start = 1 if j < best_K else 0
                for i in range(n):
                    x[i,j].Start = 1 if best_item_to_bin[i] == j else 0

            model.optimize()

            if model.SolCount > 0 and model.ObjVal < best_K:
                obj_val = int(round(model.ObjVal))
                bins = []
                for j in range(obj_val):
                    items_in_bin = []
                    total_w = 0
                    total_v = 0
                    for i in range(n):
                        if x[i,j].X > 0.5:
                            items_in_bin.append(i)
                            total_w += items[i]['w']
                            total_v += items[i]['v']
                    bins.append((total_w, total_v, items_in_bin))
                
                best_K = obj_val
                best_bins = bins
                
                solution_dict = {
                    "objective_value": best_K,
                    "bins": [
                        {
                            "items": b[2],
                            "total_weight": b[0],
                            "total_volume": b[1]
                        } for b in best_bins
                    ]
                }
                if logger:
                    logger.log_solution(best_K, solution_dict)
        except Exception as e:
            pass

    write_solution(best_K, best_bins, args.solution_path)

if __name__ == '__main__':
    main()
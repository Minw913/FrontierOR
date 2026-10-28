import argparse
import json
import time
import random
import math

def pack_items_fast(w_list, v_list, order, W, V, mode_id):
    # mode_id: 0 = First Fit, 1 = Best Fit Sum, 2 = Best Fit Max, 3 = Best Fit L2
    bins_rem_w = []
    bins_rem_v = []
    bins_items = []
    
    for idx in order:
        w = w_list[idx]
        v = v_list[idx]
        
        best_b_idx = -1
        if mode_id == 0:
            for b_idx in range(len(bins_rem_w)):
                if bins_rem_w[b_idx] >= w and bins_rem_v[b_idx] >= v:
                    best_b_idx = b_idx
                    break
                    
        elif mode_id == 1:
            best_score = float('inf')
            for b_idx in range(len(bins_rem_w)):
                if bins_rem_w[b_idx] >= w and bins_rem_v[b_idx] >= v:
                    score = bins_rem_w[b_idx] + bins_rem_v[b_idx]
                    if score < best_score:
                        best_score = score
                        best_b_idx = b_idx
                        
        elif mode_id == 2:
            best_score = float('inf')
            for b_idx in range(len(bins_rem_w)):
                if bins_rem_w[b_idx] >= w and bins_rem_v[b_idx] >= v:
                    score = bins_rem_w[b_idx] if bins_rem_w[b_idx] > bins_rem_v[b_idx] else bins_rem_v[b_idx]
                    if score < best_score:
                        best_score = score
                        best_b_idx = b_idx
                        
        elif mode_id == 3:
            best_score = float('inf')
            for b_idx in range(len(bins_rem_w)):
                if bins_rem_w[b_idx] >= w and bins_rem_v[b_idx] >= v:
                    rw = bins_rem_w[b_idx] - w
                    rv = bins_rem_v[b_idx] - v
                    score = rw * rw + rv * rv
                    if score < best_score:
                        best_score = score
                        best_b_idx = b_idx

        if best_b_idx != -1:
            bins_rem_w[best_b_idx] -= w
            bins_rem_v[best_b_idx] -= v
            bins_items[best_b_idx].append(idx)
        else:
            bins_rem_w.append(W - w)
            bins_rem_v.append(V - v)
            bins_items.append([idx])
            
    return bins_items

def build_solution(args, n, W, V, items):
    w_list = [items[i]['w'] for i in range(n)]
    v_list = [items[i]['v'] for i in range(n)]
    
    w_norm = [w / W for w in w_list]
    v_norm = [v / V for v in v_list]
    
    w_sum = sum(w_list)
    v_sum = sum(v_list)
    
    # Absolute minimum number of bins required based on total sums
    lower_bound = max((w_sum + W - 1) // W, (v_sum + V - 1) // V)

    sol_logger = None
    if args.log_path:
        try:
            from solution_logger import SolutionLogger
            sol_logger = SolutionLogger(args.log_path, sense="minimize")
        except ImportError:
            pass
            
    start_time = time.time()
    
    min_bins = float('inf')
    best_bins = []
    current_best_order = []
    
    # 1. Deterministic Heuristics
    orders = []
    
    # Linear combinations
    for i in range(21):
        alpha = i / 20.0
        scores = [alpha * w_norm[k] + (1 - alpha) * v_norm[k] for k in range(n)]
        orders.append(sorted(range(n), key=lambda k: (scores[k], w_norm[k]), reverse=True))
        
    # Chebyshev distance (Max)
    scores = [max(w_norm[k], v_norm[k]) for k in range(n)]
    orders.append(sorted(range(n), key=lambda k: (scores[k], w_norm[k]), reverse=True))

    # Euclidean distance
    scores = [(w_norm[k]**2 + v_norm[k]**2)**0.5 for k in range(n)]
    orders.append(sorted(range(n), key=lambda k: (scores[k], w_norm[k]), reverse=True))

    for order in orders:
        if time.time() - start_time > args.time_limit - 1.0:
            break
        for mode in (0, 1, 2, 3):
            if time.time() - start_time > args.time_limit - 1.0:
                break
            bins = pack_items_fast(w_list, v_list, order, W, V, mode)
            if len(bins) < min_bins:
                min_bins = len(bins)
                best_bins = bins
                current_best_order = order
                if sol_logger:
                    sol_logger.log(min_bins)
                if min_bins <= lower_bound:
                    break
        if min_bins <= lower_bound:
            break
                    
    # 2. Randomized Intensive Local Search Mode
    evals = 0
    modes = [0, 1, 2, 3]
    
    while min_bins > lower_bound:
        evals += 1
        if evals % 20 == 0:
            if time.time() - start_time > args.time_limit - 0.5:
                break
                
        r = random.random()
        if r < 0.2:
            # Completely randomized weights (diversification)
            alpha = random.random()
            beta = random.random()
            scores = [alpha * w_norm[k] + beta * v_norm[k] + random.uniform(0, 0.05) for k in range(n)]
            new_order = sorted(range(n), key=lambda k: scores[k], reverse=True)
            
        elif r < 0.7 and len(best_bins) > 0:
            # Falkenauer-inspired Grouping Mutation (Intensification)
            bin_fullness = []
            for b_items in best_bins:
                bw = sum(w_list[i] for i in b_items)
                bv = sum(v_list[i] for i in b_items)
                bin_fullness.append(bw/W + bv/V)
            
            # Sort full bins first to prioritize retaining tight fits
            sorted_bins_by_fullness = sorted(range(len(best_bins)), key=lambda x: bin_fullness[x], reverse=True)
            
            new_order = []
            bins_to_process = sorted_bins_by_fullness[:]
            
            if random.random() < 0.5 and len(bins_to_process) > 2:
                # push one mostly-full bin to the end to destroy/recombine it
                idx_to_move = random.randint(0, len(bins_to_process) // 2)
                b_moved = bins_to_process.pop(idx_to_move)
                bins_to_process.append(b_moved)
                
            for b_idx in bins_to_process:
                b_items = best_bins[b_idx][:]
                # Mildly shuffle contents within identical bins
                if random.random() < 0.3:
                    random.shuffle(b_items)
                new_order.extend(b_items)
                
            # Random bubbling to avoid local optima sinks
            for _ in range(max(1, n // 20)):
                i1 = random.randint(0, n - 1)
                i2 = random.randint(max(0, i1 - 5), min(n - 1, i1 + 5))
                if i1 != i2:
                    new_order[i1], new_order[i2] = new_order[i2], new_order[i1]
        else:
            # Minor random swap mutations on the previous best order
            new_order = current_best_order[:]
            swaps = random.randint(1, max(1, n // 10))
            for _ in range(swaps):
                i1 = random.randint(0, n - 1)
                i2 = random.randint(0, n - 1)
                new_order[i1], new_order[i2] = new_order[i2], new_order[i1]
                
        mode = random.choice(modes)
        bins = pack_items_fast(w_list, v_list, new_order, W, V, mode)
        
        if len(bins) < min_bins:
            min_bins = len(bins)
            best_bins = bins
            current_best_order = new_order
            if sol_logger:
                sol_logger.log(min_bins)
            if min_bins <= lower_bound:
                break
        elif len(bins) == min_bins:
            # Accept plateau with small probability for random drifting
            if random.random() < 0.1:
                current_best_order = new_order

    # Transform to the mandated output schema target
    output_bins = []
    for b_items in best_bins:
        tw = sum(w_list[i] for i in b_items)
        tv = sum(v_list[i] for i in b_items)
        output_bins.append({
            "items": b_items,
            "total_weight": int(tw),
            "total_volume": int(tv)
        })
        
    return {
        "objective_value": int(min_bins),
        "bins": output_bins
    }

def main():
    parser = argparse.ArgumentParser(description="2D Bin Packing Problem Solver")
    parser.add_argument("--instance_path", required=True, help="Path to the instance JSON file")
    parser.add_argument("--solution_path", required=True, help="Path to write the solution JSON file")
    parser.add_argument("--time_limit", type=int, required=True, help="Maximum execution time in seconds")
    parser.add_argument("--log_path", required=False, help="Path for JSONL incumbent solution logger")
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    n = data.get('n', 0)
    W = data.get('W', 1)
    V = data.get('V', 1)
    items = data.get('items', [])

    if n <= 0:
        output = {
            "objective_value": 0,
            "bins": []
        }
    else:
        output = build_solution(args, n, W, V, items)

    with open(args.solution_path, 'w') as f:
        json.dump(output, f, indent=2)

if __name__ == "__main__":
    main()
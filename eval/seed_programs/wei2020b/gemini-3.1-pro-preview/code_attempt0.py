import argparse
import json
import time

import gurobipy as gp
from gurobipy import GRB

class LoggerData:
    def __init__(self, logger, best_obj):
        self.logger = logger
        self.best_obj = best_obj

def solve_heuristic(items, W, V, sort_key, mode='first'):
    sorted_items = sorted(enumerate(items), key=lambda x: sort_key(x[1]), reverse=True)
    bins = []
    for orig_idx, item in sorted_items:
        best_b_idx = -1
        best_score = float('inf')
        for b_idx, b in enumerate(bins):
            if b['W'] >= item['w'] and b['V'] >= item['v']:
                if mode == 'first':
                    best_b_idx = b_idx
                    break
                elif mode == 'best_sum':
                    score = (b['W'] - item['w']) / W + (b['V'] - item['v']) / V
                    if score < best_score:
                        best_score = score
                        best_b_idx = b_idx
                elif mode == 'best_max':
                    score = max((b['W'] - item['w']) / W, (b['V'] - item['v']) / V)
                    if score < best_score:
                        best_score = score
                        best_b_idx = b_idx
        if best_b_idx != -1:
            bins[best_b_idx]['W'] -= item['w']
            bins[best_b_idx]['V'] -= item['v']
            bins[best_b_idx]['items'].append(orig_idx)
        else:
            bins.append({
                'W': W - item['w'],
                'V': V - item['v'],
                'items': [orig_idx]
            })
    return bins

def callback(model, where):
    if where == GRB.Callback.MIPSOL:
        try:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            obj_int = int(round(obj))
            cb_data = model._cb_data
            if obj_int < cb_data.best_obj:
                cb_data.best_obj = obj_int
                if cb_data.logger:
                    cb_data.logger.log(obj_int)
        except Exception:
            pass

def main():
    parser = argparse.ArgumentParser(description="2D Bin Packing Problem Solver")
    parser.add_argument("--instance_path", required=True, help="Path to the instance JSON file")
    parser.add_argument("--solution_path", required=True, help="Path to write the solution JSON file")
    parser.add_argument("--time_limit", type=int, required=True, help="Maximum execution time in seconds")
    parser.add_argument("--log_path", required=False, help="Path for JSONL incumbent solution logger")
    args = parser.parse_args()

    start_time = time.time()

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    n = data.get('n', 0)
    W = data.get('W', 1)
    V = data.get('V', 1)
    items = data.get('items', [])

    # Edge Case: empty collection
    if n == 0:
        output = {
            "objective_value": 0,
            "bins": []
        }
        with open(args.solution_path, 'w') as f:
            json.dump(output, f, indent=2)
        return

    # Adaptive scale-down of initial heuristics for extreme problem sizes
    sort_keys = [
        lambda item: item['w'] / W + item['v'] / V,
        lambda item: (item['w'] / W, item['v'] / V),
        lambda item: (item['v'] / V, item['w'] / W),
        lambda item: (item['w'] / W) ** 2 + (item['v'] / V) ** 2,
        lambda item: max(item['w'] / W, item['v'] / V)
    ]
    modes = ['first', 'best_sum', 'best_max']

    if n > 5000:
        sort_keys = sort_keys[:3]
        modes = ['first', 'best_sum']
    if n > 10000:
        sort_keys = sort_keys[:1]
        modes = ['first']

    best_bins = None
    min_bin_count = float('inf')

    # Execute FFD and Best-Fit multi-criteria variations
    for sk in sort_keys:
        for m in modes:
            b = solve_heuristic(items, W, V, sk, mode=m)
            if len(b) < min_bin_count:
                min_bin_count = len(b)
                best_bins = b

    sol_logger = None
    if args.log_path:
        try:
            from solution_logger import SolutionLogger
            sol_logger = SolutionLogger(args.log_path, sense="minimize")
        except ImportError:
            pass

    if sol_logger:
        sol_logger.log(min_bin_count)

    elapsed = time.time() - start_time
    rem_time = args.time_limit - elapsed

    use_gurobi_sol = False
    
    # Try an asymmetric MIP modeling bounded by processing time leftovers
    if rem_time > 0:
        try:
            # Reorder items descending loosely by volume and size sizes
            O = sorted(range(n), key=lambda idx: (items[idx]['w'] / W + items[idx]['v'] / V, items[idx]['w']), reverse=True)
            inverse_O = {orig: i for i, orig in enumerate(O)}

            w_prime = [items[orig]['w'] for orig in O]
            v_prime = [items[orig]['v'] for orig in O]

            # Constructing Partial warm-start seeds maps
            x_start = {}
            for b in best_bins:
                mapped_items = [inverse_O[orig_idx] for orig_idx in b['items']]
                seed = min(mapped_items)
                for mi in mapped_items:
                    x_start[(mi, seed)] = 1

            # Bounding valid search scopes for asymmetric indexing to avoid memory issues on huge models
            max_vars = 1_500_000
            if n * (n + 1) / 2 <= max_vars:
                window = n
            else:
                window = max(10, int(max_vars / max(1, n)))

            allowed_pairs = set()
            for (i, j) in x_start:
                allowed_pairs.add((i, j))

            for i in range(n):
                allowed_pairs.add((i, i))
                start_j = max(0, i - window)
                for j in range(start_j, i):
                    if w_prime[i] + w_prime[j] <= W and v_prime[i] + v_prime[j] <= V:
                        allowed_pairs.add((i, j))

            allowed_pairs_list = list(allowed_pairs)
            j_for_i = {i: [] for i in range(n)}
            i_for_j = {j: [] for j in range(n)}
            for (i, j) in allowed_pairs_list:
                j_for_i[i].append(j)
                i_for_j[j].append(i)

            env = gp.Env(empty=True)
            env.setParam('OutputFlag', 0)
            env.start()
            
            model = gp.Model("2d_bin_packing_asymmetric", env=env)
            model.setParam('Threads', 1)
            model.setParam('TimeLimit', max(1.0, args.time_limit - (time.time() - start_time)))

            cb_data = LoggerData(sol_logger, min_bin_count)
            model._cb_data = cb_data

            x = model.addVars(allowed_pairs_list, vtype=GRB.BINARY, name="x")
            for k in allowed_pairs_list:
                x[k].Start = x_start.get(k, 0)

            model.setObjective(gp.quicksum(x[j, j] for j in range(n)), GRB.MINIMIZE)

            # Assign properties guarantees: exactly 1 bin per item
            model.addConstrs(
                (gp.quicksum(x[i, j] for j in j_for_i[i]) == 1 for i in range(n)),
                name="pack"
            )

            # Combined total metrics and linear-relaxation enforcement 
            for j in range(n):
                j_items = [i for i in i_for_j[j] if i != j]
                if j_items:
                    model.addConstr(
                        gp.quicksum(w_prime[i] * x[i, j] for i in j_items) <= (W - w_prime[j]) * x[j, j],
                        name=f"weight_{j}"
                    )
                    model.addConstr(
                        gp.quicksum(v_prime[i] * x[i, j] for i in j_items) <= (V - v_prime[j]) * x[j, j],
                        name=f"vol_{j}"
                    )
                    for i in j_items:
                        model.addConstr(x[i, j] <= x[j, j], name=f"cl_{i}_{j}")
            
            model.optimize(callback)

            # Verification checkpoint before adopting exact solver configurations
            if model.SolCount > 0:
                gurobi_obj = int(round(model.ObjVal))
                if gurobi_obj <= min_bin_count:
                    temp_bins = []
                    for j in range(n):
                        if x[j, j].X > 0.5:
                            bin_items = []
                            tot_w = 0
                            tot_v = 0
                            for i in i_for_j[j]:
                                if x[i, j].X > 0.5:
                                    bin_items.append(O[i])
                                    tot_w += w_prime[i]
                                    tot_v += v_prime[i]
                            temp_bins.append({
                                "items": bin_items,
                                "total_weight": int(tot_w),
                                "total_volume": int(tot_v)
                            })
                    
                    packed_counts = {i: 0 for i in range(n)}
                    for b in temp_bins:
                        for item in b["items"]:
                            packed_counts[item] += 1
                    
                    if all(c == 1 for c in packed_counts.values()):
                        use_gurobi_sol = True
                        ans_bins = temp_bins
                        final_obj = len(ans_bins)
                        
        except Exception:
            pass

    # Restore stable best found from logic blocks if needed
    if not use_gurobi_sol:
        final_obj = min_bin_count
        ans_bins = []
        for b in best_bins:
            tot_w = sum(items[i]['w'] for i in b['items'])
            tot_v = sum(items[i]['v'] for i in b['items'])
            ans_bins.append({
                "items": b['items'],
                "total_weight": int(tot_w),
                "total_volume": int(tot_v)
            })

    output = {
        "objective_value": int(final_obj),
        "bins": ans_bins
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(output, f, indent=2)

if __name__ == "__main__":
    main()
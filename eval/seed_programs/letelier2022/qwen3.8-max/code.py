import json
import argparse
import time
import math
from collections import defaultdict
import sys


def compute_time_windows(items, arcs, source_id, sink_id, T):
    """Compute earliest and latest feasible time for each real item using longest paths."""
    item_ids = [item['item_id'] for item in items]
    NEG_INF = float('-inf')
    
    # Longest path from source (earliest times)
    earliest = {item_id: NEG_INF for item_id in item_ids}
    earliest[source_id] = 0
    
    for _ in range(len(item_ids) + 1):
        updated = False
        for arc in arcs:
            u, v, lag = arc['from'], arc['to'], arc['lag']
            if earliest[u] > NEG_INF and earliest[u] + lag > earliest[v]:
                earliest[v] = earliest[u] + lag
                updated = True
        if not updated:
            break
    
    # Longest path to sink (for latest times)
    dist_to_sink = {item_id: NEG_INF for item_id in item_ids}
    dist_to_sink[sink_id] = 0
    
    for _ in range(len(item_ids) + 1):
        updated = False
        for arc in arcs:
            u, v, lag = arc['from'], arc['to'], arc['lag']
            if dist_to_sink[v] > NEG_INF and dist_to_sink[v] + lag > dist_to_sink[u]:
                dist_to_sink[u] = dist_to_sink[v] + lag
                updated = True
        if not updated:
            break
    
    windows = {}
    for item in items:
        if item['type'] == 'treatment':
            eid = item['item_id']
            e = max(1, int(earliest[eid])) if earliest[eid] > NEG_INF else 1
            l = min(T, T + 1 - int(dist_to_sink[eid])) if dist_to_sink[eid] > NEG_INF else T
            windows[eid] = (max(1, e), min(T, l))
    
    return windows


def first_fit_decreasing(items_list, W):
    """Pack items into bins using First Fit Decreasing."""
    if not items_list:
        return []
    sorted_items = sorted(items_list, key=lambda x: x['weight'], reverse=True)
    bins = []
    bin_weights = []
    
    for item in sorted_items:
        placed = False
        for b_idx in range(len(bins)):
            if bin_weights[b_idx] + item['weight'] <= W:
                bins[b_idx].append(item)
                bin_weights[b_idx] += item['weight']
                placed = True
                break
        if not placed:
            bins.append([item])
            bin_weights.append(item['weight'])
    
    return bins


def exact_bin_packing(items_list, W):
    """Exact bin packing using branch and bound for small instances."""
    if not items_list:
        return []
    
    n = len(items_list)
    if n == 0:
        return []
    
    weights = [item['weight'] for item in items_list]
    total_weight = sum(weights)
    lower_bound = math.ceil(total_weight / W)
    
    indices = sorted(range(n), key=lambda i: weights[i], reverse=True)
    sorted_weights = [weights[i] for i in indices]
    
    best_num_bins = [n + 1]
    best_solution = [None]
    
    def backtrack(idx, bin_weights, bin_contents):
        if len(bin_weights) >= best_num_bins[0]:
            return
        if idx == n:
            if len(bin_weights) < best_num_bins[0]:
                best_num_bins[0] = len(bin_weights)
                best_solution[0] = [list(bc) for bc in bin_contents]
            return
        
        w = sorted_weights[idx]
        item = items_list[indices[idx]]
        
        tried_weights = set()
        for b in range(len(bin_weights)):
            if bin_weights[b] + w <= W and bin_weights[b] not in tried_weights:
                tried_weights.add(bin_weights[b])
                bin_weights[b] += w
                bin_contents[b].append(item)
                backtrack(idx + 1, bin_weights, bin_contents)
                bin_contents[b].pop()
                bin_weights[b] -= w
        
        if len(bin_weights) < best_num_bins[0] - 1:
            bin_weights.append(w)
            bin_contents.append([item])
            backtrack(idx + 1, bin_weights, bin_contents)
            bin_contents.pop()
            bin_weights.pop()
    
    backtrack(0, [], [])
    
    if best_solution[0] is not None:
        return best_solution[0]
    return first_fit_decreasing(items_list, W)


def pack_period(items_in_period, W):
    """Pack items for a single period, choosing algorithm based on size."""
    if not items_in_period:
        return []
    if len(items_in_period) <= 15:
        return exact_bin_packing(items_in_period, W)
    else:
        return first_fit_decreasing(items_in_period, W)


def solve_aggregate_mip(real_items, windows, arcs, T, W, L_val, B_max, time_limit, source_id, sink_id):
    """Solve aggregate MIP to determine period assignments."""
    import gurobipy as gp
    from gurobipy import GRB
    
    model = gp.Model("scheduling")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = max(1, time_limit)
    model.Params.OutputFlag = 0
    
    x = {}
    for item in real_items:
        i = item['item_id']
        e, l = windows[i]
        for t in range(e, l + 1):
            x[i, t] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{t}")
    
    n = {}
    for t in range(1, T + 1):
        ub = int(L_val) if L_val != float('inf') else B_max
        n[t] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=ub, name=f"n_{t}")
    
    model.update()
    
    for item in real_items:
        i = item['item_id']
        e, l = windows[i]
        model.addConstr(
            gp.quicksum(x[i, t] for t in range(e, l + 1)) == 1,
            name=f"assign_{i}"
        )
    
    real_item_ids = set(item['item_id'] for item in real_items)
    for arc in arcs:
        u, v, lag = arc['from'], arc['to'], arc['lag']
        if u in real_item_ids and v in real_item_ids:
            e_u, l_u = windows[u]
            e_v, l_v = windows[v]
            model.addConstr(
                gp.quicksum(t * x[v, t] for t in range(e_v, l_v + 1)) -
                gp.quicksum(t * x[u, t] for t in range(e_u, l_u + 1)) >= lag,
                name=f"prec_{u}_{v}"
            )
        elif u == source_id and v in real_item_ids:
            e_v, l_v = windows[v]
            model.addConstr(
                gp.quicksum(t * x[v, t] for t in range(e_v, l_v + 1)) >= lag,
                name=f"prec_src_{v}"
            )
        elif u in real_item_ids and v == sink_id:
            e_u, l_u = windows[u]
            model.addConstr(
                gp.quicksum(t * x[u, t] for t in range(e_u, l_u + 1)) <= T + 1 - lag,
                name=f"prec_{u}_sink"
            )
    
    for t in range(1, T + 1):
        items_in_t = [(item['item_id'], item['weight']) for item in real_items
                     if windows[item['item_id']][0] <= t <= windows[item['item_id']][1]]
        if items_in_t:
            model.addConstr(
                gp.quicksum(w * x[i, t] for i, w in items_in_t) <= W * n[t],
                name=f"weight_{t}"
            )
    
    for t in range(1, T + 1):
        items_in_t = [item['item_id'] for item in real_items
                     if windows[item['item_id']][0] <= t <= windows[item['item_id']][1]]
        for i in items_in_t:
            model.addConstr(n[t] >= x[i, t], name=f"link_{i}_{t}")
    
    model.setObjective(gp.quicksum(n[t] for t in range(1, T + 1)), GRB.MINIMIZE)
    model.optimize()
    
    if model.Status == GRB.INFEASIBLE or model.SolCount == 0:
        return None
    
    period_assignment = {}
    for item in real_items:
        i = item['item_id']
        e, l = windows[i]
        for t in range(e, l + 1):
            if x[i, t].X > 0.5:
                period_assignment[i] = t
                break
    
    return period_assignment


def greedy_period_assignment(real_items, windows, arcs, T, W, L_val):
    """Greedy heuristic respecting precedence constraints."""
    real_item_ids = set(item['item_id'] for item in real_items)
    period_assignment = {}
    
    for item in real_items:
        i = item['item_id']
        e, l = windows[i]
        period_assignment[i] = e
    
    for _ in range(len(real_items) + 1):
        changed = False
        for arc in arcs:
            u, v, lag = arc['from'], arc['to'], arc['lag']
            if u in real_item_ids and v in real_item_ids:
                if u in period_assignment and v in period_assignment:
                    if period_assignment[v] < period_assignment[u] + lag:
                        new_t = period_assignment[u] + lag
                        e_v, l_v = windows[v]
                        if new_t <= l_v:
                            period_assignment[v] = new_t
                            changed = True
        if not changed:
            break
    
    return period_assignment


def build_solution(real_items, period_assignment, T, W):
    """Build complete solution with bin assignments."""
    period_items = defaultdict(list)
    for item in real_items:
        i = item['item_id']
        if i in period_assignment:
            period_items[period_assignment[i]].append(item)
    
    assignments = []
    active_bin_periods = []
    
    for t in range(1, T + 1):
        items_in_t = period_items.get(t, [])
        if not items_in_t:
            continue
        
        bins = pack_period(items_in_t, W)
        
        for b_idx, bin_items in enumerate(bins):
            bin_id = b_idx + 1
            total_weight = sum(item['weight'] for item in bin_items)
            item_ids = [item['item_id'] for item in bin_items]
            
            active_bin_periods.append({
                "bin": bin_id,
                "period": t,
                "items": item_ids,
                "total_weight": total_weight
            })
            
            for item in bin_items:
                assignments.append({
                    "item_id": item['item_id'],
                    "bin": bin_id,
                    "period": t,
                    "weight": item['weight']
                })
    
    return {
        "objective_value": float(len(active_bin_periods)),
        "assignments": assignments,
        "active_bin_periods": active_bin_periods,
        "num_active_bin_periods": len(active_bin_periods)
    }


def local_search(real_items, period_assignment, windows, arcs, T, W, L_val, time_limit, start_time):
    """Improve solution using local search with efficient objective evaluation."""
    real_item_ids = set(item['item_id'] for item in real_items)
    item_by_id = {item['item_id']: item for item in real_items}
    
    # Build precedence adjacency
    succ = defaultdict(list)
    pred = defaultdict(list)
    for arc in arcs:
        u, v, lag = arc['from'], arc['to'], arc['lag']
        if u in real_item_ids and v in real_item_ids:
            succ[u].append((v, lag))
            pred[v].append((u, lag))
    
    # Group items by period
    period_items = defaultdict(list)
    for item in real_items:
        i = item['item_id']
        if i in period_assignment:
            period_items[period_assignment[i]].append(item)
    
    # Compute current bins per period
    bins_per_period = {}
    for t in range(1, T + 1):
        items_t = period_items.get(t, [])
        bins_per_period[t] = len(pack_period(items_t, W)) if items_t else 0
    
    current_obj = sum(bins_per_period.values())
    best_obj = current_obj
    best_assignment = dict(period_assignment)
    
    improved = True
    iteration = 0
    while improved and (time.time() - start_time) < time_limit - 1:
        improved = False
        iteration += 1
        
        for item in real_items:
            if (time.time() - start_time) > time_limit - 1:
                break
            
            i = item['item_id']
            current_t = period_assignment[i]
            e, l = windows[i]
            
            for new_t in range(e, l + 1):
                if new_t == current_t:
                    continue
                
                # Check precedence
                feasible = True
                for (v, lag) in succ[i]:
                    if v in period_assignment and period_assignment[v] < new_t + lag:
                        feasible = False
                        break
                if not feasible:
                    continue
                for (u, lag) in pred[i]:
                    if u in period_assignment and new_t < period_assignment[u] + lag:
                        feasible = False
                        break
                if not feasible:
                    continue
                
                # Compute change in objective
                # Remove item from current_t
                items_ct = [it for it in period_items[current_t] if it['item_id'] != i]
                new_bins_ct = len(pack_period(items_ct, W)) if items_ct else 0
                
                # Add item to new_t
                items_nt = period_items[new_t] + [item]
                new_bins_nt = len(pack_period(items_nt, W))
                
                delta = (new_bins_ct - bins_per_period[current_t]) + (new_bins_nt - bins_per_period[new_t])
                
                if delta < 0:
                    # Accept move
                    period_assignment[i] = new_t
                    period_items[current_t] = items_ct
                    period_items[new_t] = items_nt
                    bins_per_period[current_t] = new_bins_ct
                    bins_per_period[new_t] = new_bins_nt
                    current_obj += delta
                    
                    if current_obj < best_obj:
                        best_obj = current_obj
                        best_assignment = dict(period_assignment)
                        improved = True
                    break  # Move to next item
        
        if iteration > 100:
            break
    
    return best_assignment, best_obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    start_time = time.time()
    
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    params = instance['parameters']
    T = params['T']
    W = params['W']
    L = params['L']
    L_val = float('inf') if L == 'infinity' else int(L)
    
    items = instance['items']
    arcs = instance['arcs']
    source_id = instance['source_item_id']
    sink_id = instance['sink_item_id']
    
    real_items = [item for item in items if item['type'] == 'treatment']
    num_real = len(real_items)
    
    if num_real == 0:
        solution = {
            "objective_value": 0.0,
            "assignments": [],
            "active_bin_periods": [],
            "num_active_bin_periods": 0
        }
        if logger:
            logger.log_solution(0.0, solution)
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
        return
    
    # Compute time windows
    windows = compute_time_windows(items, arcs, source_id, sink_id, T)
    
    # Determine B_max
    total_weight = sum(item['weight'] for item in real_items)
    B_max = math.ceil(total_weight / W) if W > 0 else num_real
    
    # Try MIP
    period_assignment = None
    remaining = args.time_limit - (time.time() - start_time)
    
    if remaining > 3:
        try:
            period_assignment = solve_aggregate_mip(
                real_items, windows, arcs, T, W, L_val, B_max,
                max(1, remaining - 2), source_id, sink_id
            )
        except Exception as e:
            print(f"MIP failed: {e}", file=sys.stderr)
            period_assignment = None
    
    if period_assignment is None:
        period_assignment = greedy_period_assignment(real_items, windows, arcs, T, W, L_val)
    
    # Build initial solution
    solution = build_solution(real_items, period_assignment, T, W)
    
    if logger:
        logger.log_solution(solution['objective_value'], solution)
    
    # Local search improvement
    remaining = args.time_limit - (time.time() - start_time)
    if remaining > 2:
        improved_assignment, improved_obj = local_search(
            real_items, period_assignment, windows, arcs, T, W, L_val,
            args.time_limit, start_time
        )
        
        if improved_obj < solution['objective_value']:
            solution = build_solution(real_items, improved_assignment, T, W)
            if logger:
                logger.log_solution(solution['objective_value'], solution)
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()
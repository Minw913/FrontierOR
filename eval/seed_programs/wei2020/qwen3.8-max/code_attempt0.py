import json
import argparse
import time
import sys
import random
from copy import deepcopy

def read_instance(path):
    with open(path, 'r') as f:
        data = json.load(f)
    n = data['n']
    c = data['c']
    weights = data['weights']
    conflicts = data.get('conflicts', [])
    return n, c, weights, conflicts

def build_conflict_set(conflicts):
    conflict_adj = {}
    for pair in conflicts:
        a, b = pair[0], pair[1]
        if a not in conflict_adj:
            conflict_adj[a] = set()
        if b not in conflict_adj:
            conflict_adj[b] = set()
        conflict_adj[a].add(b)
        conflict_adj[b].add(a)
    return conflict_adj

def ffd(n, c, weights, conflict_adj):
    """First Fit Decreasing heuristic"""
    order = sorted(range(n), key=lambda i: weights[i], reverse=True)
    bins = []
    bin_weights = []
    
    for item in order:
        placed = False
        for b in range(len(bins)):
            if bin_weights[b] + weights[item] > c:
                continue
            conflict_found = False
            for other in bins[b]:
                if other in conflict_adj.get(item, set()):
                    conflict_found = True
                    break
            if not conflict_found:
                bins[b].append(item)
                bin_weights[b] += weights[item]
                placed = True
                break
        if not placed:
            bins.append([item])
            bin_weights.append(weights[item])
    
    return bins

def bfd(n, c, weights, conflict_adj):
    """Best Fit Decreasing heuristic"""
    order = sorted(range(n), key=lambda i: weights[i], reverse=True)
    bins = []
    bin_weights = []
    
    for item in order:
        best_bin = -1
        best_remaining = float('inf')
        
        for b in range(len(bins)):
            if bin_weights[b] + weights[item] > c:
                continue
            conflict_found = False
            for other in bins[b]:
                if other in conflict_adj.get(item, set()):
                    conflict_found = True
                    break
            if not conflict_found:
                remaining = c - bin_weights[b] - weights[item]
                if remaining < best_remaining:
                    best_remaining = remaining
                    best_bin = b
        
        if best_bin >= 0:
            bins[best_bin].append(item)
            bin_weights[best_bin] += weights[item]
        else:
            bins.append([item])
            bin_weights.append(weights[item])
    
    return bins

def worst_fit_decreasing(n, c, weights, conflict_adj):
    """Worst Fit Decreasing heuristic"""
    order = sorted(range(n), key=lambda i: weights[i], reverse=True)
    bins = []
    bin_weights = []
    
    for item in order:
        best_bin = -1
        best_remaining = -1
        
        for b in range(len(bins)):
            if bin_weights[b] + weights[item] > c:
                continue
            conflict_found = False
            for other in bins[b]:
                if other in conflict_adj.get(item, set()):
                    conflict_found = True
                    break
            if not conflict_found:
                remaining = c - bin_weights[b] - weights[item]
                if remaining > best_remaining:
                    best_remaining = remaining
                    best_bin = b
        
        if best_bin >= 0:
            bins[best_bin].append(item)
            bin_weights[best_bin] += weights[item]
        else:
            bins.append([item])
            bin_weights.append(weights[item])
    
    return bins

def local_search(bins, n, c, weights, conflict_adj, time_limit, start_time):
    """Local search to reduce number of bins"""
    
    def get_bin_weight(b):
        return sum(weights[i] for i in bins[b])
    
    def can_place_in_bin(item, bin_idx, exclude_items=None):
        if exclude_items is None:
            exclude_items = set()
        current_weight = sum(weights[i] for i in bins[bin_idx] if i not in exclude_items)
        if current_weight + weights[item] > c:
            return False
        for other in bins[bin_idx]:
            if other not in exclude_items and other in conflict_adj.get(item, set()):
                return False
        return True
    
    improved = True
    while improved and time.time() - start_time < time_limit - 0.5:
        improved = False
        
        # Strategy 1: Try to empty bins (from smallest to largest)
        bin_sizes = [(len(bins[b]), b) for b in range(len(bins))]
        bin_sizes.sort()
        
        for _, b in bin_sizes:
            if time.time() - start_time > time_limit - 0.5:
                break
            if len(bins[b]) == 0:
                continue
            
            items_to_move = list(bins[b])
            moves = []
            success = True
            
            # Sort items to move by weight descending (heavier items are harder to place)
            items_to_move.sort(key=lambda x: weights[x], reverse=True)
            
            for item in items_to_move:
                placed = False
                # Try to find a bin for this item
                candidates = []
                for b2 in range(len(bins)):
                    if b2 == b:
                        continue
                    # Compute effective weight considering planned moves to b2
                    extra_weight = sum(weights[m_item] for m_item, m_bin in moves if m_bin == b2)
                    current_weight = get_bin_weight(b2) + extra_weight
                    if current_weight + weights[item] > c:
                        continue
                    # Check conflicts
                    conflict_found = False
                    for other in bins[b2]:
                        if other in conflict_adj.get(item, set()):
                            conflict_found = True
                            break
                    if conflict_found:
                        continue
                    for m_item, m_bin in moves:
                        if m_bin == b2 and m_item in conflict_adj.get(item, set()):
                            conflict_found = True
                            break
                    if conflict_found:
                        continue
                    remaining = c - current_weight - weights[item]
                    candidates.append((remaining, b2))
                
                if candidates:
                    # Best fit among candidates
                    candidates.sort()
                    moves.append((item, candidates[0][1]))
                    placed = True
                
                if not placed:
                    success = False
                    break
            
            if success and len(moves) > 0:
                for item, target_bin in moves:
                    bins[b].remove(item)
                    bins[target_bin].append(item)
                bins.pop(b)
                improved = True
                break
        
        if improved:
            continue
        
        # Strategy 2: Move single items to enable future bin emptying
        # Try moving items from bins with few items to consolidate
        for b in range(len(bins)):
            if time.time() - start_time > time_limit - 0.5:
                break
            if len(bins[b]) <= 1:
                continue
            for item in list(bins[b]):
                for b2 in range(len(bins)):
                    if b2 == b:
                        continue
                    if can_place_in_bin(item, b2):
                        # Check if this helps (moves item to a bin that's more full)
                        if get_bin_weight(b2) + weights[item] <= c:
                            bins[b].remove(item)
                            bins[b2].append(item)
                            if len(bins[b]) == 0:
                                bins.pop(b)
                                improved = True
                            break
                if improved:
                    break
            if improved:
                break
    
    return bins

def local_search_v2(bins, n, c, weights, conflict_adj, time_limit, start_time):
    """More aggressive local search with swaps"""
    
    def get_bin_weight(bin_items):
        return sum(weights[i] for i in bin_items)
    
    improved = True
    iterations = 0
    max_iterations = 1000
    
    while improved and iterations < max_iterations and time.time() - start_time < time_limit - 0.5:
        improved = False
        iterations += 1
        
        # Try to empty the bin with fewest items
        non_empty = [(len(bins[b]), b) for b in range(len(bins)) if len(bins[b]) > 0]
        non_empty.sort()
        
        for _, source_bin in non_empty:
            if time.time() - start_time > time_limit - 0.5:
                break
            
            items_in_source = list(bins[source_bin])
            if len(items_in_source) == 0:
                continue
            
            # Try to redistribute all items from source_bin
            # Use a greedy approach with backtracking for small bins
            if len(items_in_source) <= 5:
                result = try_empty_bin(bins, source_bin, n, c, weights, conflict_adj)
                if result is not None:
                    bins = result
                    improved = True
                    break
    
    return bins

def try_empty_bin(bins, source_bin, n, c, weights, conflict_adj):
    """Try to empty a specific bin by moving its items elsewhere"""
    items = list(bins[source_bin])
    num_other_bins = len(bins) - 1
    
    if num_other_bins == 0:
        return None
    
    # For small number of items, try all combinations
    if len(items) > 6:
        # Greedy approach
        return greedy_empty_bin(bins, source_bin, n, c, weights, conflict_adj)
    
    # DFS/backtracking
    other_bins = [b for b in range(len(bins)) if b != source_bin]
    
    def backtrack(idx, assignments):
        if idx == len(items):
            return assignments[:]
        
        item = items[idx]
        for b in other_bins:
            # Check if item can go in bin b
            current_items = bins[b] + [assignments[j] for j in range(idx) if assignments[j][1] == b]
            current_items_in_b = [x for x, target in assignments[:idx] if target == b]
            
            weight = sum(weights[i] for i in bins[b]) + sum(weights[i] for i in current_items_in_b)
            if weight + weights[item] > c:
                continue
            
            conflict_found = False
            for other in bins[b]:
                if other in conflict_adj.get(item, set()):
                    conflict_found = True
                    break
            if conflict_found:
                continue
            for prev_item in current_items_in_b:
                if prev_item in conflict_adj.get(item, set()):
                    conflict_found = True
                    break
            if conflict_found:
                continue
            
            assignments.append((item, b))
            result = backtrack(idx + 1, assignments)
            if result is not None:
                return result
            assignments.pop()
        
        return None
    
    result = backtrack(0, [])
    if result is not None:
        new_bins = [list(b) for b in bins]
        for item, target in result:
            new_bins[source_bin].remove(item)
            new_bins[target].append(item)
        new_bins.pop(source_bin)
        return new_bins
    
    return None

def greedy_empty_bin(bins, source_bin, n, c, weights, conflict_adj):
    """Greedily try to empty a bin"""
    items = sorted(bins[source_bin], key=lambda x: weights[x], reverse=True)
    other_bins = [b for b in range(len(bins)) if b != source_bin]
    
    moves = []
    for item in items:
        placed = False
        best_b = -1
        best_remaining = float('inf')
        
        for b in other_bins:
            extra = sum(weights[m[0]] for m in moves if m[1] == b)
            current_w = sum(weights[i] for i in bins[b]) + extra
            if current_w + weights[item] > c:
                continue
            
            conflict_found = False
            for other in bins[b]:
                if other in conflict_adj.get(item, set()):
                    conflict_found = True
                    break
            if conflict_found:
                continue
            for m_item, m_bin in moves:
                if m_bin == b and m_item in conflict_adj.get(item, set()):
                    conflict_found = True
                    break
            if conflict_found:
                continue
            
            remaining = c - current_w - weights[item]
            if remaining < best_remaining:
                best_remaining = remaining
                best_b = b
        
        if best_b >= 0:
            moves.append((item, best_b))
            placed = True
        
        if not placed:
            return None
    
    new_bins = [list(b) for b in bins]
    for item, target in moves:
        new_bins[source_bin].remove(item)
        new_bins[target].append(item)
    new_bins.pop(source_bin)
    return new_bins

def solve_mip(n, c, weights, conflicts, conflict_adj, upper_bound, time_limit, start_time):
    """Solve using Gurobi MIP"""
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError:
        return None
    
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time <= 1:
        return None
    
    num_bins = upper_bound
    
    # If problem is too large for MIP, skip
    if n * num_bins > 10000:
        return None
    
    model = gp.Model("bin_packing")
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', max(1, remaining_time - 1))
    model.setParam('OutputFlag', 0)
    
    # Variables
    x = {}
    for i in range(n):
        for j in range(num_bins):
            x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
    
    y = {}
    for j in range(num_bins):
        y[j] = model.addVar(vtype=GRB.BINARY, name=f"y_{j}")
    
    model.update()
    
    # Objective
    model.setObjective(gp.quicksum(y[j] for j in range(num_bins)), GRB.MINIMIZE)
    
    # Each item assigned to exactly one bin
    for i in range(n):
        model.addConstr(gp.quicksum(x[i, j] for j in range(num_bins)) == 1, name=f"assign_{i}")
    
    # Capacity constraints
    for j in range(num_bins):
        model.addConstr(gp.quicksum(weights[i] * x[i, j] for i in range(n)) <= c * y[j], name=f"cap_{j}")
    
    # Conflict constraints
    for pair in conflicts:
        i, k = pair[0], pair[1]
        for j in range(num_bins):
            model.addConstr(x[i, j] + x[k, j] <= 1, name=f"conf_{i}_{k}_{j}")
    
    # Symmetry breaking
    for j in range(num_bins - 1):
        model.addConstr(y[j] >= y[j + 1], name=f"sym_{j}")
    
    # Additional symmetry breaking: item 0 goes to bin 0
    if n > 0:
        model.addConstr(x[0, 0] == 1, name="fix_item0")
    
    model.optimize()
    
    if model.SolCount > 0:
        bins = [[] for _ in range(num_bins)]
        for i in range(n):
            for j in range(num_bins):
                if x[i, j].X > 0.5:
                    bins[j].append(i)
                    break
        bins = [b for b in bins if len(b) > 0]
        return bins
    
    return None

def compute_lower_bound(n, c, weights):
    """Compute a lower bound on the number of bins needed"""
    total_weight = sum(weights)
    lb = (total_weight + c - 1) // c
    
    # L2 bound (Martello and Toth)
    # For each item size class
    max_w = max(weights) if weights else 0
    for alpha in range(1, c // 2 + 1):
        large_items = [w for w in weights if w > c - alpha]
        small_items = [w for w in weights if w <= alpha]
        if len(large_items) > 0:
            residual_capacity = len(large_items) * c - sum(large_items)
            small_total = sum(small_items)
            if small_total > residual_capacity:
                extra = (small_total - residual_capacity + c - 1) // c
                lb = max(lb, len(large_items) + extra)
    
    return lb

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
    
    n, c, weights, conflicts = read_instance(args.instance_path)
    conflict_adj = build_conflict_set(conflicts)
    
    # Compute lower bound
    lb = compute_lower_bound(n, c, weights)
    
    # Generate initial solutions from multiple heuristics
    candidates = []
    
    bins_ffd = ffd(n, c, weights, conflict_adj)
    candidates.append(bins_ffd)
    
    bins_bfd = bfd(n, c, weights, conflict_adj)
    candidates.append(bins_bfd)
    
    bins_wfd = worst_fit_decreasing(n, c, weights, conflict_adj)
    candidates.append(bins_wfd)
    
    # Pick the best initial solution
    best_bins = min(candidates, key=lambda b: len(b))
    best_obj = len(best_bins)
    
    # Log initial solution
    solution_dict = {
        "objective_value": float(best_obj),
        "bins": [list(b) for b in best_bins]
    }
    if logger:
        logger.log_solution(best_obj, solution_dict)
    
    # If already at lower bound, we're optimal
    if best_obj == lb:
        with open(args.solution_path, 'w') as f:
            json.dump(solution_dict, f)
        return
    
    # Local search improvement
    if time.time() - start_time < args.time_limit - 2:
        improved_bins = local_search(deepcopy(best_bins), n, c, weights, conflict_adj, args.time_limit, start_time)
        if len(improved_bins) < best_obj:
            best_bins = improved_bins
            best_obj = len(best_bins)
            solution_dict = {
                "objective_value": float(best_obj),
                "bins": [list(b) for b in best_bins]
            }
            if logger:
                logger.log_solution(best_obj, solution_dict)
    
    # More aggressive local search
    if best_obj > lb and time.time() - start_time < args.time_limit - 2:
        improved_bins = local_search_v2(deepcopy(best_bins), n, c, weights, conflict_adj, args.time_limit, start_time)
        if len(improved_bins) < best_obj:
            best_bins = improved_bins
            best_obj = len(best_bins)
            solution_dict = {
                "objective_value": float(best_obj),
                "bins": [list(b) for b in best_bins]
            }
            if logger:
                logger.log_solution(best_obj, solution_dict)
    
    # Try MIP if instance is small enough and we have time
    remaining = args.time_limit - (time.time() - start_time)
    if best_obj > lb and remaining > 3:
        # Determine if MIP is feasible
        if n * best_obj <= 10000:
            mip_bins = solve_mip(n, c, weights, conflicts, conflict_adj, best_obj, args.time_limit, start_time)
            if mip_bins is not None and len(mip_bins) < best_obj:
                best_bins = mip_bins
                best_obj = len(best_bins)
                solution_dict = {
                    "objective_value": float(best_obj),
                    "bins": [list(b) for b in best_bins]
                }
                if logger:
                    logger.log_solution(best_obj, solution_dict)
    
    # Write final solution
    solution_dict = {
        "objective_value": float(best_obj),
        "bins": [list(b) for b in best_bins]
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution_dict, f)

if __name__ == '__main__':
    main()
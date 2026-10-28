import argparse
import json
import time
import sys


def format_solution(bins, weights, idx_to_id=None):
    """Convert internal bin representation to output format."""
    solution = {
        "objective_value": len(bins),
        "bins": {}
    }
    for idx, bin_items in enumerate(bins):
        if idx_to_id:
            item_ids = sorted([idx_to_id[i] for i in bin_items])
        else:
            item_ids = sorted(bin_items)
        solution["bins"][str(idx)] = {
            "items": item_ids,
            "total_weight": sum(weights[i] for i in bin_items)
        }
    return solution


def first_fit_decreasing(num_items, bin_capacity, weights, conflict_adj, order=None):
    """First Fit Decreasing heuristic."""
    if order is None:
        order = sorted(range(num_items), key=lambda i: (-weights[i], -len(conflict_adj[i])))
    
    bins = []
    bin_weights = []
    
    for item in order:
        placed = False
        for b in range(len(bins)):
            if bin_weights[b] + weights[item] > bin_capacity:
                continue
            if any(other in conflict_adj[item] for other in bins[b]):
                continue
            bins[b].append(item)
            bin_weights[b] += weights[item]
            placed = True
            break
        if not placed:
            bins.append([item])
            bin_weights.append(weights[item])
    
    return bins


def best_fit_decreasing(num_items, bin_capacity, weights, conflict_adj, order=None):
    """Best Fit Decreasing heuristic."""
    if order is None:
        order = sorted(range(num_items), key=lambda i: (-weights[i], -len(conflict_adj[i])))
    
    bins = []
    bin_weights = []
    
    for item in order:
        best_bin = -1
        best_remaining = float('inf')
        
        for b in range(len(bins)):
            if bin_weights[b] + weights[item] > bin_capacity:
                continue
            if any(other in conflict_adj[item] for other in bins[b]):
                continue
            remaining = bin_capacity - bin_weights[b] - weights[item]
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


def worst_fit_decreasing(num_items, bin_capacity, weights, conflict_adj, order=None):
    """Worst Fit Decreasing heuristic."""
    if order is None:
        order = sorted(range(num_items), key=lambda i: (-weights[i], -len(conflict_adj[i])))
    
    bins = []
    bin_weights = []
    
    for item in order:
        best_bin = -1
        best_remaining = -1
        
        for b in range(len(bins)):
            if bin_weights[b] + weights[item] > bin_capacity:
                continue
            if any(other in conflict_adj[item] for other in bins[b]):
                continue
            remaining = bin_capacity - bin_weights[b] - weights[item]
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


def local_search_bin_elimination(bins, num_items, bin_capacity, weights, conflict_adj, time_limit):
    """Try to reduce the number of bins by eliminating one bin at a time."""
    start = time.time()
    
    bin_sets = [set(b) for b in bins]
    
    improved = True
    while improved and (time.time() - start) < time_limit:
        improved = False
        
        # Get active bins sorted by weight (try to eliminate lightest first)
        active_bins = []
        for b in range(len(bin_sets)):
            if bin_sets[b]:
                w = sum(weights[i] for i in bin_sets[b])
                active_bins.append((w, b))
        active_bins.sort()
        
        for _, b in active_bins:
            if (time.time() - start) >= time_limit:
                break
            if not bin_sets[b]:
                continue
            
            # Try to move all items from bin b to other bins
            items_to_move = sorted(bin_sets[b], key=lambda i: -weights[i])
            
            # Create temporary state
            temp_bins = [set(bs) for bs in bin_sets]
            temp_bins[b] = set()
            temp_weights = [sum(weights[i] for i in temp_bins[j]) for j in range(len(temp_bins))]
            
            success = True
            for item in items_to_move:
                placed = False
                best_target = -1
                best_remaining = float('inf')
                
                for other_b in range(len(temp_bins)):
                    if other_b == b:
                        continue
                    if not temp_bins[other_b]:
                        continue  # Don't place items in empty bins
                    if temp_weights[other_b] + weights[item] > bin_capacity:
                        continue
                    if any(other in conflict_adj[item] for other in temp_bins[other_b]):
                        continue
                    remaining = bin_capacity - temp_weights[other_b] - weights[item]
                    if remaining < best_remaining:
                        best_remaining = remaining
                        best_target = other_b
                
                if best_target >= 0:
                    temp_bins[best_target].add(item)
                    temp_weights[best_target] += weights[item]
                    placed = True
                
                if not placed:
                    success = False
                    break
            
            if success:
                bin_sets = temp_bins
                improved = True
                break
    
    # Remove empty bins
    result = [sorted(list(bs)) for bs in bin_sets if bs]
    return result


def local_search_pairwise(bins, num_items, bin_capacity, weights, conflict_adj, time_limit):
    """Try pairwise moves to enable bin elimination."""
    start = time.time()
    
    bin_sets = [set(b) for b in bins]
    
    improved = True
    while improved and (time.time() - start) < time_limit:
        improved = False
        
        # Try moving single items between bins to enable future elimination
        for b1 in range(len(bin_sets)):
            if (time.time() - start) >= time_limit:
                break
            if not bin_sets[b1]:
                continue
            
            for item in list(bin_sets[b1]):
                if (time.time() - start) >= time_limit:
                    break
                
                for b2 in range(len(bin_sets)):
                    if b2 == b1:
                        continue
                    if not bin_sets[b2]:
                        continue
                    
                    # Check if item can move to b2
                    w_b2 = sum(weights[i] for i in bin_sets[b2])
                    if w_b2 + weights[item] > bin_capacity:
                        continue
                    if any(other in conflict_adj[item] for other in bin_sets[b2]):
                        continue
                    
                    # Check if moving this item would help (b1 becomes empty or lighter)
                    remaining_in_b1 = len(bin_sets[b1]) - 1
                    
                    if remaining_in_b1 == 0:
                        # This would empty bin b1 - always good
                        bin_sets[b1].remove(item)
                        bin_sets[b2].add(item)
                        improved = True
                        break
                    elif remaining_in_b1 <= 2:
                        # Might enable future elimination
                        bin_sets[b1].remove(item)
                        bin_sets[b2].add(item)
                        improved = True
                        break
                
                if improved:
                    break
            if improved:
                break
    
    result = [sorted(list(bs)) for bs in bin_sets if bs]
    return result


def solve_mip(num_items, bin_capacity, weights, conflict_edges, conflict_adj, heuristic_bins, time_limit, logger):
    """Solve using Gurobi MIP."""
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError:
        return None
    
    upper_bound = len(heuristic_bins)
    total_weight = sum(weights)
    
    if bin_capacity > 0:
        lower_bound = max(1, -(-total_weight // bin_capacity))
    else:
        lower_bound = num_items
    
    if upper_bound <= lower_bound:
        return None  # Already at lower bound, optimal
    
    # Limit MIP size to keep it tractable
    if num_items * upper_bound > 80000:
        return None
    
    model = gp.Model("bppc")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = time_limit
    model.Params.OutputFlag = 0
    
    # Variables
    x = {}
    for i in range(num_items):
        for j in range(upper_bound):
            x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
    
    y = {}
    for j in range(upper_bound):
        y[j] = model.addVar(vtype=GRB.BINARY, name=f"y_{j}")
    
    model.setObjective(gp.quicksum(y[j] for j in range(upper_bound)), GRB.MINIMIZE)
    
    # Assignment constraints
    for i in range(num_items):
        model.addConstr(gp.quicksum(x[i, j] for j in range(upper_bound)) == 1)
    
    # Capacity constraints
    for j in range(upper_bound):
        model.addConstr(
            gp.quicksum(weights[i] * x[i, j] for i in range(num_items)) <= bin_capacity * y[j]
        )
    
    # Conflict constraints
    for a, b in conflict_edges:
        for j in range(upper_bound):
            model.addConstr(x[a, j] + x[b, j] <= 1)
    
    # Symmetry breaking
    for j in range(upper_bound - 1):
        model.addConstr(y[j] >= y[j + 1])
    
    # Warm start from heuristic
    for i in range(num_items):
        for j in range(upper_bound):
            x[i, j].Start = 0.0
    for j in range(upper_bound):
        y[j].Start = 0.0
    
    for j, bin_items in enumerate(heuristic_bins):
        y[j].Start = 1.0
        for i in bin_items:
            x[i, j].Start = 1.0
    
    model.optimize()
    
    if model.SolCount > 0:
        bins = []
        for j in range(upper_bound):
            if y[j].X > 0.5:
                bin_items = [i for i in range(num_items) if x[i, j].X > 0.5]
                if bin_items:
                    bins.append(bin_items)
        
        if bins:
            solution = format_solution(bins, weights)
            if logger:
                logger.log_solution(solution['objective_value'], solution)
            return solution
    
    return None


def main():
    parser = argparse.ArgumentParser(description="Bin Packing with Conflicts")
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    start_time = time.time()
    
    # Read instance
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    num_items = instance['num_items']
    bin_capacity = instance['bin_capacity']
    items = instance['items']
    conflict_edges_raw = instance['conflict_edges']
    
    # Handle edge case
    if num_items == 0:
        solution = {"objective_value": 0, "bins": {}}
        if logger:
            logger.log_solution(0, solution)
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f)
        return
    
    # Extract weights and build ID mapping
    item_ids = [item['id'] for item in items]
    id_to_idx = {item_id: idx for idx, item_id in enumerate(item_ids)}
    idx_to_id = {idx: item_id for idx, item_id in enumerate(item_ids)}
    weights = [item['weight'] for item in items]
    
    # Map conflict edges to internal indices
    conflict_edges = []
    for edge in conflict_edges_raw:
        a, b = edge[0], edge[1]
        if a in id_to_idx and b in id_to_idx:
            conflict_edges.append((id_to_idx[a], id_to_idx[b]))
    
    # Build conflict adjacency
    conflict_adj = [set() for _ in range(num_items)]
    for a, b in conflict_edges:
        conflict_adj[a].add(b)
        conflict_adj[b].add(a)
    
    best_bins = None
    best_obj = float('inf')
    final_solution = None
    
    # Phase 1: Heuristics with different orderings and strategies
    orderings = [
        sorted(range(num_items), key=lambda i: (-weights[i], -len(conflict_adj[i]))),
        sorted(range(num_items), key=lambda i: (-len(conflict_adj[i]), -weights[i])),
        sorted(range(num_items), key=lambda i: (-weights[i],)),
        sorted(range(num_items), key=lambda i: (weights[i], -len(conflict_adj[i]))),
    ]
    
    heuristic_funcs = [first_fit_decreasing, best_fit_decreasing, worst_fit_decreasing]
    
    for order in orderings:
        for heuristic in heuristic_funcs:
            elapsed = time.time() - start_time
            if elapsed >= args.time_limit * 0.2:
                break
            bins = heuristic(num_items, bin_capacity, weights, conflict_adj, order)
            obj = len(bins)
            if obj < best_obj:
                best_obj = obj
                best_bins = bins
                final_solution = format_solution(best_bins, weights, idx_to_id)
                if logger:
                    logger.log_solution(best_obj, final_solution)
    
    # Phase 2: Local search - bin elimination
    remaining = args.time_limit - (time.time() - start_time)
    if remaining > 1 and best_bins is not None:
        ls_time = min(remaining * 0.35, remaining - 3)
        if ls_time > 0.5:
            improved_bins = local_search_bin_elimination(
                best_bins, num_items, bin_capacity, weights, conflict_adj, ls_time
            )
            obj = len(improved_bins)
            if obj < best_obj:
                best_obj = obj
                best_bins = improved_bins
                final_solution = format_solution(best_bins, weights, idx_to_id)
                if logger:
                    logger.log_solution(best_obj, final_solution)
    
    # Phase 2b: Pairwise local search
    remaining = args.time_limit - (time.time() - start_time)
    if remaining > 2 and best_bins is not None:
        ls_time = min(remaining * 0.2, remaining - 3)
        if ls_time > 0.5:
            improved_bins = local_search_pairwise(
                best_bins, num_items, bin_capacity, weights, conflict_adj, ls_time
            )
            obj = len(improved_bins)
            if obj < best_obj:
                best_obj = obj
                best_bins = improved_bins
                final_solution = format_solution(best_bins, weights, idx_to_id)
                if logger:
                    logger.log_solution(best_obj, final_solution)
    
    # Phase 3: MIP for small/medium instances
    remaining = args.time_limit - (time.time() - start_time)
    if remaining > 3 and best_bins is not None:
        mip_solution = solve_mip(
            num_items, bin_capacity, weights, conflict_edges, conflict_adj,
            best_bins, remaining - 1, logger
        )
        if mip_solution is not None and mip_solution['objective_value'] < best_obj:
            best_obj = mip_solution['objective_value']
            # Need to convert back to use idx_to_id
            # MIP already returns item indices, format_solution handles conversion
            final_solution = mip_solution
            # Re-format with proper IDs
            # Extract bins from mip_solution
            mip_bins = []
            for bin_key in sorted(mip_solution['bins'].keys(), key=lambda x: int(x)):
                mip_bins.append(mip_solution['bins'][bin_key]['items'])
            # The items in mip_solution are already internal indices from MIP
            # We need to re-format with idx_to_id
            final_solution = format_solution(mip_bins, weights, idx_to_id)
            if logger:
                logger.log_solution(best_obj, final_solution)
    
    # Write final solution
    if final_solution is None:
        final_solution = format_solution(best_bins, weights, idx_to_id)
    
    with open(args.solution_path, 'w') as f:
        json.dump(final_solution, f, indent=2)


if __name__ == "__main__":
    main()
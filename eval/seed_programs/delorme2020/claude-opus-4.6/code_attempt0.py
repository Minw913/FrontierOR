import argparse
import json
import time
from solution_logger import SolutionLogger


def solve_bin_packing(instance_path, solution_path, time_limit, log_path=None):
    start_time = time.time()
    
    # Read instance
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    
    n = instance['n']
    m = instance['m']
    c = instance['c']
    weights = instance['weights']
    demands = instance['demands']
    
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    # Expand items: create a list of all individual item weights
    items = []
    for i in range(m):
        items.extend([weights[i]] * demands[i])
    
    assert len(items) == n
    
    # Sort items in decreasing order (First Fit Decreasing is a good heuristic)
    items.sort(reverse=True)
    
    # --- First Fit Decreasing (FFD) as initial solution ---
    def first_fit_decreasing(items, capacity):
        bins = []
        bin_remaining = []
        for item in items:
            placed = False
            for j in range(len(bins)):
                if bin_remaining[j] >= item:
                    bins[j].append(item)
                    bin_remaining[j] -= item
                    placed = True
                    break
            if not placed:
                bins.append([item])
                bin_remaining.append(capacity - item)
        return bins
    
    best_bins = first_fit_decreasing(items, c)
    best_obj = len(best_bins)
    
    if logger:
        logger.log(best_obj)
    
    # --- Best Fit Decreasing (BFD) ---
    def best_fit_decreasing(items, capacity):
        bins = []
        bin_remaining = []
        for item in items:
            best_idx = -1
            best_remaining = capacity + 1
            for j in range(len(bins)):
                if bin_remaining[j] >= item and bin_remaining[j] < best_remaining:
                    best_remaining = bin_remaining[j]
                    best_idx = j
            if best_idx >= 0:
                bins[best_idx].append(item)
                bin_remaining[best_idx] -= item
            else:
                bins.append([item])
                bin_remaining.append(capacity - item)
        return bins
    
    bfd_bins = best_fit_decreasing(items, c)
    if len(bfd_bins) < best_obj:
        best_bins = bfd_bins
        best_obj = len(best_bins)
        if logger:
            logger.log(best_obj)
    
    # Compute lower bound
    total_weight = sum(items)
    lower_bound = (total_weight + c - 1) // c  # ceiling division
    
    if best_obj <= lower_bound:
        # Already optimal
        solution = {
            "objective_value": float(best_obj),
            "num_bins": best_obj,
            "bin_assignments": best_bins
        }
        with open(solution_path, 'w') as f:
            json.dump(solution, f)
        return
    
    # --- Try Column Generation / MIP approach for better solutions ---
    # For medium-to-large instances, we use a column generation approach
    # For small instances, we can try direct MIP
    
    elapsed = time.time() - start_time
    remaining_time = time_limit - elapsed
    
    if remaining_time > 2 and n <= 500:
        try:
            import gurobipy as gp
            from gurobipy import GRB
            
            # Column generation approach for bin packing
            # We work with item types (weights[i], demands[i])
            
            # Generate initial patterns using FFD-like approach
            # Each pattern is a list of length m, where pattern[i] = number of copies of item type i
            
            def generate_initial_patterns():
                """Generate one pattern per item type (single item patterns) plus FFD patterns."""
                patterns = []
                # Single item patterns
                for i in range(m):
                    if weights[i] <= c:
                        pat = [0] * m
                        pat[i] = c // weights[i]  # max copies that fit
                        patterns.append(pat)
                
                # Also generate patterns from FFD solution
                for bin_items in best_bins:
                    pat = [0] * m
                    for w in bin_items:
                        for i in range(m):
                            if weights[i] == w and pat[i] < demands[i]:
                                # This is a heuristic mapping
                                pass
                    # Better: count occurrences
                    item_counts = {}
                    for w in bin_items:
                        item_counts[w] = item_counts.get(w, 0) + 1
                    
                    pat = [0] * m
                    remaining_counts = dict(item_counts)
                    for i in range(m):
                        if weights[i] in remaining_counts and remaining_counts[weights[i]] > 0:
                            take = min(remaining_counts[weights[i]], demands[i])
                            pat[i] = take
                            remaining_counts[weights[i]] -= take
                    patterns.append(pat)
                
                # Remove duplicates
                unique = []
                seen = set()
                for p in patterns:
                    key = tuple(p)
                    if key not in seen and any(x > 0 for x in p):
                        seen.add(key)
                        unique.append(p)
                return unique
            
            patterns = generate_initial_patterns()
            
            # Column generation loop
            max_cg_iterations = 1000
            
            for cg_iter in range(max_cg_iterations):
                elapsed = time.time() - start_time
                if elapsed > time_limit - 2:
                    break
                
                # Solve restricted master problem (LP relaxation)
                master = gp.Model("master")
                master.setParam('OutputFlag', 0)
                master.setParam('TimeLimit', max(1, time_limit - elapsed - 1))
                
                num_patterns = len(patterns)
                x = master.addVars(num_patterns, lb=0, vtype=GRB.CONTINUOUS, name="x")
                
                # Demand constraints
                demand_constrs = []
                for i in range(m):
                    constr = master.addConstr(
                        gp.quicksum(patterns[j][i] * x[j] for j in range(num_patterns)) >= demands[i],
                        name=f"demand_{i}"
                    )
                    demand_constrs.append(constr)
                
                master.setObjective(gp.quicksum(x[j] for j in range(num_patterns)), GRB.MINIMIZE)
                master.optimize()
                
                if master.status != GRB.OPTIMAL:
                    break
                
                # Get dual values
                duals = [demand_constrs[i].Pi for i in range(m)]
                
                # Solve pricing problem (knapsack)
                pricing = gp.Model("pricing")
                pricing.setParam('OutputFlag', 0)
                pricing.setParam('TimeLimit', max(1, time_limit - (time.time() - start_time) - 1))
                
                y = pricing.addVars(m, lb=0, vtype=GRB.INTEGER, name="y")
                for i in range(m):
                    y[i].ub = min(demands[i], c // weights[i])
                
                pricing.addConstr(gp.quicksum(weights[i] * y[i] for i in range(m)) <= c)
                pricing.setObjective(gp.quicksum(duals[i] * y[i] for i in range(m)), GRB.MAXIMIZE)
                pricing.optimize()
                
                if pricing.status != GRB.OPTIMAL:
                    break
                
                reduced_cost = 1.0 - pricing.ObjVal
                
                if reduced_cost >= -1e-6:
                    # No improving column found
                    break
                
                # Add new pattern
                new_pattern = [int(round(y[i].X)) for i in range(m)]
                # Verify feasibility
                if sum(new_pattern[i] * weights[i] for i in range(m)) <= c:
                    key = tuple(new_pattern)
                    if key not in set(tuple(p) for p in patterns):
                        patterns.append(new_pattern)
                    else:
                        break
                else:
                    break
                
                master.dispose()
                pricing.dispose()
            
            # Now solve the integer master problem
            elapsed = time.time() - start_time
            remaining = time_limit - elapsed
            
            if remaining > 1:
                master_ip = gp.Model("master_ip")
                master_ip.setParam('OutputFlag', 0)
                master_ip.setParam('TimeLimit', max(1, remaining - 0.5))
                master_ip.setParam('Threads', 1)
                
                # Set cutoff based on best known solution
                master_ip.setParam('Cutoff', best_obj - 0.5)
                
                num_patterns = len(patterns)
                x = master_ip.addVars(num_patterns, lb=0, vtype=GRB.INTEGER, name="x")
                
                for i in range(m):
                    master_ip.addConstr(
                        gp.quicksum(patterns[j][i] * x[j] for j in range(num_patterns)) >= demands[i],
                        name=f"demand_{i}"
                    )
                
                master_ip.setObjective(gp.quicksum(x[j] for j in range(num_patterns)), GRB.MINIMIZE)
                master_ip.optimize()
                
                if master_ip.status in [GRB.OPTIMAL, GRB.SUBOPTIMAL] or (master_ip.status == GRB.TIME_LIMIT and master_ip.SolCount > 0):
                    ip_obj = int(round(master_ip.ObjVal))
                    
                    if ip_obj < best_obj:
                        # Reconstruct bins
                        new_bins = []
                        for j in range(num_patterns):
                            count = int(round(x[j].X))
                            for _ in range(count):
                                bin_content = []
                                for i in range(m):
                                    bin_content.extend([weights[i]] * patterns[j][i])
                                if bin_content:
                                    new_bins.append(bin_content)
                        
                        # Verify and fix: we might have excess items due to >= constraints
                        # Count actual items packed
                        packed_counts = [0] * m
                        for bin_content in new_bins:
                            for w in bin_content:
                                for i in range(m):
                                    if weights[i] == w:
                                        packed_counts[i] += 1
                                        break
                        
                        # More careful counting with duplicate weights
                        # Rebuild with exact tracking
                        packed_counts = [0] * m
                        for bin_content in new_bins:
                            wc = {}
                            for w in bin_content:
                                wc[w] = wc.get(w, 0) + 1
                            # Assign to item types greedily
                            for i in range(m):
                                w = weights[i]
                                if w in wc and wc[w] > 0:
                                    take = min(wc[w], demands[i] - packed_counts[i])
                                    if take < 0:
                                        take = 0
                                    packed_counts[i] += take
                                    wc[w] -= take
                        
                        # The IP solution with >= constraints is valid but may pack extra
                        # We need to reconstruct properly
                        # Let's just use the pattern multiplicities directly
                        
                        pattern_usage = []
                        for j in range(num_patterns):
                            count = int(round(x[j].X))
                            if count > 0:
                                pattern_usage.append((j, count))
                        
                        # Now we need to assign exact demands
                        # Reduce pattern usage if we have excess
                        remaining_demand = list(demands)
                        final_bins = []
                        
                        for j, count in pattern_usage:
                            pat = patterns[j]
                            for _ in range(count):
                                actual_pat = [0] * m
                                for i in range(m):
                                    actual_pat[i] = min(pat[i], remaining_demand[i])
                                    remaining_demand[i] -= actual_pat[i]
                                
                                bin_content = []
                                for i in range(m):
                                    bin_content.extend([weights[i]] * actual_pat[i])
                                if bin_content:
                                    final_bins.append(bin_content)
                        
                        # Check if all demands met
                        if all(d == 0 for d in remaining_demand):
                            if len(final_bins) < best_obj:
                                best_bins = final_bins
                                best_obj = len(best_bins)
                                if logger:
                                    logger.log(best_obj)
                        else:
                            # Some items remaining, add them with FFD
                            leftover_items = []
                            for i in range(m):
                                leftover_items.extend([weights[i]] * remaining_demand[i])
                            leftover_items.sort(reverse=True)
                            
                            # Try to fit into existing bins first
                            bin_remaining = [c - sum(b) for b in final_bins]
                            for item in leftover_items:
                                placed = False
                                for idx in range(len(final_bins)):
                                    if bin_remaining[idx] >= item:
                                        final_bins[idx].append(item)
                                        bin_remaining[idx] -= item
                                        placed = True
                                        break
                                if not placed:
                                    final_bins.append([item])
                                    bin_remaining.append(c - item)
                            
                            if len(final_bins) < best_obj:
                                best_bins = final_bins
                                best_obj = len(best_bins)
                                if logger:
                                    logger.log(best_obj)
                
                master_ip.dispose()
        
        except Exception as e:
            # If Gurobi fails, stick with heuristic solution
            pass
    
    # --- Local search improvement ---
    elapsed = time.time() - start_time
    remaining = time_limit - elapsed
    
    if remaining > 1 and best_obj > lower_bound:
        # Try to improve by swapping items between bins and eliminating bins
        improved = True
        while improved and (time.time() - start_time) < time_limit - 0.5:
            improved = False
            
            # Try to empty the least-filled bin
            bin_loads = [(sum(b), idx) for idx, b in enumerate(best_bins)]
            bin_loads.sort()
            
            for load, target_idx in bin_loads:
                if (time.time() - start_time) >= time_limit - 0.5:
                    break
                    
                target_bin = best_bins[target_idx]
                if not target_bin:
                    continue
                
                # Try to redistribute items from target_bin to other bins
                items_to_place = list(target_bin)
                items_to_place.sort(reverse=True)
                
                # Compute remaining capacity of other bins
                other_bins = [(c - sum(best_bins[j]), j) for j in range(len(best_bins)) if j != target_idx]
                other_bins.sort(reverse=True)
                
                # Greedy placement
                placement = {}  # item_index -> bin_index
                temp_remaining = {j: rem for rem, j in other_bins}
                all_placed = True
                
                for item_idx, item in enumerate(items_to_place):
                    placed = False
                    for rem, j in other_bins:
                        if temp_remaining[j] >= item:
                            temp_remaining[j] -= item
                            placement[item_idx] = j
                            placed = True
                            break
                    if not placed:
                        all_placed = False
                        break
                
                if all_placed:
                    # Apply the redistribution
                    for item_idx, j in placement.items():
                        best_bins[j].append(items_to_place[item_idx])
                    best_bins[target_idx] = []
                    improved = True
            
            # Remove empty bins
            best_bins = [b for b in best_bins if b]
            
            if len(best_bins) < best_obj:
                best_obj = len(best_bins)
                if logger:
                    logger.log(best_obj)
                improved = True
            else:
                improved = False
    
    # Output solution
    solution = {
        "objective_value": float(best_obj),
        "num_bins": best_obj,
        "bin_assignments": best_bins
    }
    
    with open(solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    solve_bin_packing(args.instance_path, args.solution_path, args.time_limit, args.log_path)
import argparse
import json
import time
import sys
from solution_logger import SolutionLogger


def solve_bin_packing(instance, time_limit, logger=None):
    start_time = time.time()
    
    bin_capacity = instance["bin_capacity"]
    items = instance["items"]
    num_items = instance["num_items"]
    
    # Sort items in decreasing order (First Fit Decreasing heuristic)
    indexed_items = sorted(enumerate(items), key=lambda x: -x[1])
    sorted_sizes = [s for _, s in indexed_items]
    
    # --- Phase 1: First Fit Decreasing (FFD) heuristic ---
    bins = []  # each bin is a list of item sizes
    bin_remaining = []  # remaining capacity of each bin
    
    for size in sorted_sizes:
        placed = False
        # Find first bin that fits
        for i in range(len(bins)):
            if bin_remaining[i] >= size:
                bins[i].append(size)
                bin_remaining[i] -= size
                placed = True
                break
        if not placed:
            bins.append([size])
            bin_remaining.append(bin_capacity - size)
    
    best_num_bins = len(bins)
    best_bins = [b[:] for b in bins]
    
    if logger:
        logger.log(best_num_bins)
    
    elapsed = time.time() - start_time
    if elapsed > time_limit * 0.1:
        return best_num_bins, best_bins
    
    # --- Phase 1.5: Best Fit Decreasing (BFD) heuristic ---
    bins_bfd = []
    bin_remaining_bfd = []
    
    for size in sorted_sizes:
        best_idx = -1
        best_rem = bin_capacity + 1
        for i in range(len(bins_bfd)):
            if bin_remaining_bfd[i] >= size and bin_remaining_bfd[i] - size < best_rem:
                best_rem = bin_remaining_bfd[i] - size
                best_idx = i
        if best_idx >= 0:
            bins_bfd[best_idx].append(size)
            bin_remaining_bfd[best_idx] -= size
        else:
            bins_bfd.append([size])
            bin_remaining_bfd.append(bin_capacity - size)
    
    if len(bins_bfd) < best_num_bins:
        best_num_bins = len(bins_bfd)
        best_bins = [b[:] for b in bins_bfd]
        if logger:
            logger.log(best_num_bins)
    
    # --- Phase 2: Try to improve with column generation / Gurobi MIP ---
    # Compute lower bound: ceil(sum of sizes / capacity)
    total_size = sum(items)
    lower_bound = (total_size + bin_capacity - 1) // bin_capacity
    
    if best_num_bins <= lower_bound:
        return best_num_bins, best_bins
    
    elapsed = time.time() - start_time
    remaining_time = time_limit - elapsed
    
    if remaining_time < 2 or num_items > 5000:
        return best_num_bins, best_bins
    
    # For moderate instances, try MIP formulation
    # Use arc-flow or direct assignment formulation
    
    if num_items <= 300:
        try:
            import gurobipy as gp
            from gurobipy import GRB
            
            n = num_items
            max_bins = best_num_bins  # We know FFD solution uses this many
            
            elapsed = time.time() - start_time
            remaining_time = time_limit - elapsed - 1
            if remaining_time < 2:
                return best_num_bins, best_bins
            
            model = gp.Model("binpacking")
            model.setParam("OutputFlag", 0)
            model.setParam("TimeLimit", remaining_time)
            model.setParam("Threads", 1)
            model.setParam("MIPFocus", 1)  # Focus on finding feasible solutions
            
            # Use sorted items for symmetry breaking
            item_sizes = sorted_sizes
            
            # Variables: x[i,j] = 1 if item i is in bin j
            # y[j] = 1 if bin j is used
            x = {}
            y = {}
            
            for j in range(max_bins):
                y[j] = model.addVar(vtype=GRB.BINARY, name=f"y_{j}")
                for i in range(n):
                    x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
            
            model.update()
            
            # Objective: minimize number of bins used
            model.setObjective(gp.quicksum(y[j] for j in range(max_bins)), GRB.MINIMIZE)
            
            # Each item must be assigned to exactly one bin
            for i in range(n):
                model.addConstr(gp.quicksum(x[i, j] for j in range(max_bins)) == 1)
            
            # Bin capacity constraints
            for j in range(max_bins):
                model.addConstr(gp.quicksum(item_sizes[i] * x[i, j] for i in range(n)) <= bin_capacity * y[j])
            
            # Symmetry breaking: bins are ordered (y[j] >= y[j+1])
            for j in range(max_bins - 1):
                model.addConstr(y[j] >= y[j + 1])
            
            # Symmetry breaking for items of same size
            for i in range(n - 1):
                if item_sizes[i] == item_sizes[i + 1]:
                    for j in range(1, max_bins):
                        model.addConstr(x[i, j] <= gp.quicksum(x[i + 1, k] for k in range(j + 1)))
            
            # Provide FFD solution as warm start
            # Map sorted items to bins
            for j in range(max_bins):
                y[j].Start = 1 if j < len(best_bins) else 0
            
            # Build warm start
            item_idx = 0
            for j, bin_items in enumerate(best_bins):
                for size in bin_items:
                    # Find the corresponding index in sorted order
                    # This is tricky; let's just not provide a warm start for x
                    pass
            
            model.optimize()
            
            if model.SolCount > 0:
                sol_bins = {}
                for j in range(max_bins):
                    if y[j].X > 0.5:
                        sol_bins[j] = []
                        for i in range(n):
                            if x[i, j].X > 0.5:
                                sol_bins[j].append(item_sizes[i])
                
                new_num_bins = len(sol_bins)
                if new_num_bins < best_num_bins:
                    best_num_bins = new_num_bins
                    best_bins = list(sol_bins.values())
                    if logger:
                        logger.log(best_num_bins)
        except Exception:
            pass
    elif num_items <= 1500:
        # For medium instances, try a column-generation-inspired approach
        # Group items by size and use a pattern-based MIP
        from collections import Counter
        size_counts = Counter(items)
        distinct_sizes = sorted(size_counts.keys(), reverse=True)
        demands = [size_counts[s] for s in distinct_sizes]
        m = len(distinct_sizes)
        
        if m <= 50:
            # Generate feasible patterns
            patterns = []
            
            def generate_patterns(idx, remaining_cap, current_pattern):
                patterns.append(tuple(current_pattern))
                for i in range(idx, m):
                    if distinct_sizes[i] <= remaining_cap:
                        max_fit = min(demands[i], remaining_cap // distinct_sizes[i])
                        for count in range(1, max_fit + 1):
                            current_pattern[i] = count
                            generate_patterns(i + 1, remaining_cap - count * distinct_sizes[i], current_pattern)
                        current_pattern[i] = 0
            
            # Limit pattern generation
            elapsed = time.time() - start_time
            remaining_time = time_limit - elapsed - 2
            
            if remaining_time > 3 and m <= 30:
                patterns = []
                
                def gen_patterns(idx, rem, pat, max_patterns=500000):
                    if len(patterns) >= max_patterns:
                        return
                    if sum(pat) > 0:
                        patterns.append(tuple(pat))
                    for i in range(idx, m):
                        if distinct_sizes[i] <= rem:
                            max_fit = min(demands[i], rem // distinct_sizes[i])
                            for c in range(1, max_fit + 1):
                                if len(patterns) >= max_patterns:
                                    return
                                pat[i] = c
                                gen_patterns(i + 1, rem - c * distinct_sizes[i], pat, max_patterns)
                            pat[i] = 0
                
                pat = [0] * m
                gen_patterns(0, bin_capacity, pat)
                
                if len(patterns) > 0 and len(patterns) <= 500000:
                    try:
                        import gurobipy as gp
                        from gurobipy import GRB
                        
                        elapsed = time.time() - start_time
                        remaining_time = time_limit - elapsed - 1
                        if remaining_time < 2:
                            return best_num_bins, best_bins
                        
                        model = gp.Model("cutting_stock")
                        model.setParam("OutputFlag", 0)
                        model.setParam("TimeLimit", remaining_time)
                        model.setParam("Threads", 1)
                        model.setParam("MIPFocus", 1)
                        
                        P = len(patterns)
                        z = model.addVars(P, vtype=GRB.INTEGER, lb=0, name="z")
                        
                        model.setObjective(gp.quicksum(z[p] for p in range(P)), GRB.MINIMIZE)
                        
                        for i in range(m):
                            model.addConstr(
                                gp.quicksum(patterns[p][i] * z[p] for p in range(P) if patterns[p][i] > 0) >= demands[i]
                            )
                        
                        model.optimize()
                        
                        if model.SolCount > 0:
                            new_bins = []
                            for p in range(P):
                                count = int(round(z[p].X))
                                if count > 0:
                                    bin_content = []
                                    for i in range(m):
                                        bin_content.extend([distinct_sizes[i]] * patterns[p][i])
                                    for _ in range(count):
                                        new_bins.append(bin_content[:])
                            
                            new_num_bins = len(new_bins)
                            if new_num_bins < best_num_bins:
                                best_num_bins = new_num_bins
                                best_bins = new_bins
                                if logger:
                                    logger.log(best_num_bins)
                    except Exception:
                        pass
    
    return best_num_bins, best_bins


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, default=300)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()
    
    with open(args.instance_path, "r") as f:
        instance = json.load(f)
    
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    best_num_bins, best_bins = solve_bin_packing(instance, args.time_limit, logger)
    
    bin_capacity = instance["bin_capacity"]
    
    # Validate and format output
    bin_assignments = []
    for b in best_bins:
        total = sum(b)
        assert total <= bin_capacity, f"Bin overflow: {total} > {bin_capacity}"
        bin_assignments.append({
            "items": b,
            "total_size": total
        })
    
    solution = {
        "objective_value": best_num_bins,
        "num_bins": best_num_bins,
        "bin_assignments": bin_assignments
    }
    
    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()
import argparse
import json
import time
from collections import defaultdict

def solve_bin_packing(instance_path, solution_path, time_limit, log_path):
    from solution_logger import SolutionLogger
    
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    start_time = time.time()
    
    W = instance['parameters']['W']
    m = instance['parameters']['m']
    items = instance['items']
    
    # Build item list: (weight, item_type, demand)
    item_types = []
    for item in items:
        item_types.append((item['weight'], item['item_type'], item['demand']))
    
    # Expand items into individual copies, sorted by weight descending (for FFD)
    all_items = []
    for weight, itype, demand in item_types:
        for _ in range(demand):
            all_items.append((weight, itype))
    
    all_items.sort(key=lambda x: -x[0])
    
    total_weight = sum(w for w, _ in all_items)
    lower_bound = (total_weight + W - 1) // W
    
    if not all_items:
        solution = {
            "objective_value": 0.0,
            "num_bins": 0,
            "bins": []
        }
        with open(solution_path, 'w') as f:
            json.dump(solution, f)
        return
    
    # First Fit Decreasing heuristic
    def first_fit_decreasing(items_list):
        bins = []  # list of (remaining_capacity, [item_types])
        for weight, itype in items_list:
            placed = False
            for i in range(len(bins)):
                if bins[i][0] >= weight:
                    bins[i] = (bins[i][0] - weight, bins[i][1] + [itype])
                    placed = True
                    break
            if not placed:
                bins.append((W - weight, [itype]))
        return bins
    
    # Best Fit Decreasing heuristic
    def best_fit_decreasing(items_list):
        bins = []  # list of (remaining_capacity, [item_types])
        for weight, itype in items_list:
            best_idx = -1
            best_remaining = W + 1
            for i in range(len(bins)):
                if bins[i][0] >= weight and bins[i][0] - weight < best_remaining:
                    best_remaining = bins[i][0] - weight
                    best_idx = i
            if best_idx >= 0:
                bins[best_idx] = (bins[best_idx][0] - weight, bins[best_idx][1] + [itype])
            else:
                bins.append((W - weight, [itype]))
        return bins
    
    ffd_bins = first_fit_decreasing(all_items)
    best_bins = ffd_bins
    best_obj = len(ffd_bins)
    
    if logger:
        logger.log(best_obj)
    
    bfd_bins = best_fit_decreasing(all_items)
    if len(bfd_bins) < best_obj:
        best_bins = bfd_bins
        best_obj = len(bfd_bins)
        if logger:
            logger.log(best_obj)
    
    if best_obj <= lower_bound:
        # Already optimal
        result_bins = [b[1] for b in best_bins]
        solution = {
            "objective_value": float(best_obj),
            "num_bins": best_obj,
            "bins": result_bins
        }
        with open(solution_path, 'w') as f:
            json.dump(solution, f)
        return
    
    # Try to improve with column generation / MIP approach using Gurobi
    # For moderate instances, use a direct MIP formulation
    # For large instances, stick with heuristic + local search
    
    n_items = len(all_items)
    elapsed = time.time() - start_time
    remaining_time = time_limit - elapsed
    
    # Try MIP if instance is not too large
    if n_items <= 2000 and remaining_time > 2:
        try:
            import gurobipy as gp
            from gurobipy import GRB
            
            # Use column generation approach:
            # Generate feasible packing patterns, then solve set cover
            
            # For practical purposes with moderate instances, use direct assignment MIP
            # x[i,j] = 1 if item i is in bin j
            # But number of bins is variable... use upper bound from heuristic
            
            UB = best_obj
            
            # Group identical items by type for more compact formulation
            # type_info: {itype: (weight, demand)}
            type_info = {}
            for weight, itype, demand in item_types:
                type_info[itype] = (weight, demand)
            
            # If m is small enough, use type-based formulation
            # x[t,j] = number of copies of type t in bin j
            # sum_t w_t * x[t,j] <= W for all j
            # sum_j x[t,j] >= demand_t for all t
            # y[j] = 1 if bin j is used
            # minimize sum_j y[j]
            
            if m <= 200 and UB <= 500:
                model = gp.Model("binpacking")
                model.setParam('OutputFlag', 0)
                model.setParam('TimeLimit', max(1, remaining_time - 1))
                model.setParam('Threads', 1)
                model.setParam('MIPGap', 1e-4)
                
                # Variables
                y = {}
                x = {}
                
                for j in range(UB):
                    y[j] = model.addVar(vtype=GRB.BINARY, name=f"y_{j}")
                    for t_idx, (weight, itype, demand) in enumerate(item_types):
                        x[t_idx, j] = model.addVar(vtype=GRB.INTEGER, lb=0, 
                                                     ub=min(demand, W // weight),
                                                     name=f"x_{t_idx}_{j}")
                
                model.update()
                
                # Objective
                model.setObjective(gp.quicksum(y[j] for j in range(UB)), GRB.MINIMIZE)
                
                # Capacity constraints
                for j in range(UB):
                    model.addConstr(
                        gp.quicksum(item_types[t_idx][0] * x[t_idx, j] 
                                   for t_idx in range(m)) <= W * y[j],
                        name=f"cap_{j}"
                    )
                
                # Demand constraints
                for t_idx in range(m):
                    model.addConstr(
                        gp.quicksum(x[t_idx, j] for j in range(UB)) >= item_types[t_idx][2],
                        name=f"dem_{t_idx}"
                    )
                
                # Symmetry breaking: order bins by usage
                for j in range(1, UB):
                    model.addConstr(y[j] <= y[j-1])
                
                # Set initial solution from heuristic
                # Map best_bins to type counts per bin
                for j in range(UB):
                    if j < len(best_bins):
                        y[j].Start = 1.0
                        type_counts = defaultdict(int)
                        for itype in best_bins[j][1]:
                            type_counts[itype] += 1
                        for t_idx in range(m):
                            itype = item_types[t_idx][1]
                            x[t_idx, j].Start = float(type_counts.get(itype, 0))
                    else:
                        y[j].Start = 0.0
                        for t_idx in range(m):
                            x[t_idx, j].Start = 0.0
                
                # Callback to log improvements
                def callback(model, where):
                    if where == GRB.Callback.MIPSOL:
                        obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                        obj_int = int(round(obj))
                        nonlocal best_obj, best_bins_mip
                        if obj_int < best_obj:
                            best_obj = obj_int
                            # Extract solution
                            bins_sol = []
                            for j in range(UB):
                                yv = model.cbGetSolution(y[j])
                                if yv > 0.5:
                                    bin_items = []
                                    for t_idx in range(m):
                                        cnt = int(round(model.cbGetSolution(x[t_idx, j])))
                                        for _ in range(cnt):
                                            bin_items.append(item_types[t_idx][1])
                                    if bin_items:
                                        bins_sol.append(bin_items)
                            best_bins_mip = bins_sol
                            if logger:
                                logger.log(best_obj)
                
                best_bins_mip = None
                
                model.optimize(callback)
                
                if model.Status in [GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT]:
                    if model.SolCount > 0 and best_bins_mip is not None:
                        best_bins_final = best_bins_mip
                    elif model.SolCount > 0:
                        # Extract from model
                        bins_sol = []
                        for j in range(UB):
                            if y[j].X > 0.5:
                                bin_items = []
                                for t_idx in range(m):
                                    cnt = int(round(x[t_idx, j].X))
                                    for _ in range(cnt):
                                        bin_items.append(item_types[t_idx][1])
                                if bin_items:
                                    bins_sol.append(bin_items)
                        obj_val = len(bins_sol)
                        if obj_val < best_obj:
                            best_obj = obj_val
                            if logger:
                                logger.log(best_obj)
                        best_bins_final = bins_sol
                    else:
                        best_bins_final = [b[1] for b in best_bins]
                else:
                    best_bins_final = [b[1] for b in best_bins]
                
                solution = {
                    "objective_value": float(best_obj),
                    "num_bins": best_obj,
                    "bins": best_bins_final
                }
                with open(solution_path, 'w') as f:
                    json.dump(solution, f)
                return
            
            elif remaining_time > 5:
                # Use column generation approach
                # First solve LP relaxation with column generation, then solve restricted MIP
                
                # Generate initial patterns from FFD solution
                patterns = []  # Each pattern is a dict {t_idx: count}
                for cap, bin_items in best_bins:
                    pattern = defaultdict(int)
                    for itype in bin_items:
                        for t_idx in range(m):
                            if item_types[t_idx][1] == itype:
                                pattern[t_idx] += 1
                                break
                    patterns.append(dict(pattern))
                
                # Also add single-item patterns
                for t_idx in range(m):
                    weight = item_types[t_idx][0]
                    max_copies = W // weight
                    pattern = {t_idx: max_copies}
                    patterns.append(pattern)
                
                # Remove duplicate patterns
                unique_patterns = []
                seen = set()
                for p in patterns:
                    key = tuple(sorted(p.items()))
                    if key not in seen:
                        seen.add(key)
                        unique_patterns.append(p)
                patterns = unique_patterns
                
                # Column generation loop
                max_cg_iters = 1000
                cg_time_limit = min(remaining_time * 0.5, 60)
                
                for cg_iter in range(max_cg_iters):
                    if time.time() - start_time > time_limit - 2:
                        break
                    
                    # Solve master LP
                    master = gp.Model("master")
                    master.setParam('OutputFlag', 0)
                    master.setParam('Threads', 1)
                    master.setParam('TimeLimit', max(1, time_limit - (time.time() - start_time) - 2))
                    
                    lam = {}
                    for p_idx in range(len(patterns)):
                        lam[p_idx] = master.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"lam_{p_idx}")
                    
                    master.update()
                    
                    master.setObjective(gp.quicksum(lam[p_idx] for p_idx in range(len(patterns))), GRB.MINIMIZE)
                    
                    demand_constrs = {}
                    for t_idx in range(m):
                        demand_constrs[t_idx] = master.addConstr(
                            gp.quicksum(patterns[p_idx].get(t_idx, 0) * lam[p_idx] 
                                       for p_idx in range(len(patterns))) >= item_types[t_idx][2],
                            name=f"dem_{t_idx}"
                        )
                    
                    master.optimize()
                    
                    if master.Status != GRB.OPTIMAL:
                        break
                    
                    # Get duals
                    duals = {}
                    for t_idx in range(m):
                        duals[t_idx] = demand_constrs[t_idx].Pi
                    
                    # Solve pricing problem (knapsack)
                    pricing = gp.Model("pricing")
                    pricing.setParam('OutputFlag', 0)
                    pricing.setParam('Threads', 1)
                    pricing.setParam('TimeLimit', max(1, min(10, time_limit - (time.time() - start_time) - 2)))
                    
                    z = {}
                    for t_idx in range(m):
                        weight = item_types[t_idx][0]
                        ub = min(item_types[t_idx][2], W // weight)
                        z[t_idx] = pricing.addVar(vtype=GRB.INTEGER, lb=0, ub=ub, name=f"z_{t_idx}")
                    
                    pricing.update()
                    
                    pricing.setObjective(gp.quicksum(duals[t_idx] * z[t_idx] for t_idx in range(m)), GRB.MAXIMIZE)
                    
                    pricing.addConstr(
                        gp.quicksum(item_types[t_idx][0] * z[t_idx] for t_idx in range(m)) <= W,
                        name="capacity"
                    )
                    
                    pricing.optimize()
                    
                    if pricing.Status != GRB.OPTIMAL:
                        break
                    
                    reduced_cost = 1.0 - pricing.ObjVal
                    
                    if reduced_cost >= -1e-6:
                        break  # No improving column
                    
                    # Add new pattern
                    new_pattern = {}
                    for t_idx in range(m):
                        val = int(round(z[t_idx].X))
                        if val > 0:
                            new_pattern[t_idx] = val
                    
                    key = tuple(sorted(new_pattern.items()))
                    if key not in seen:
                        seen.add(key)
                        patterns.append(new_pattern)
                    else:
                        break  # Duplicate pattern, stop
                
                # Now solve restricted MIP with all generated patterns
                remaining = time_limit - (time.time() - start_time) - 1
                if remaining > 1:
                    mip = gp.Model("mip_master")
                    mip.setParam('OutputFlag', 0)
                    mip.setParam('Threads', 1)
                    mip.setParam('TimeLimit', max(1, remaining))
                    
                    lam_int = {}
                    for p_idx in range(len(patterns)):
                        # Upper bound on how many times we use this pattern
                        max_use = max(
                            (item_types[t_idx][2] // patterns[p_idx][t_idx] if t_idx in patterns[p_idx] and patterns[p_idx][t_idx] > 0 else 0)
                            for t_idx in range(m)
                        ) if patterns[p_idx] else 0
                        max_use = max(max_use, UB)
                        lam_int[p_idx] = mip.addVar(vtype=GRB.INTEGER, lb=0, ub=UB, name=f"lam_{p_idx}")
                    
                    mip.update()
                    
                    mip.setObjective(gp.quicksum(lam_int[p_idx] for p_idx in range(len(patterns))), GRB.MINIMIZE)
                    
                    for t_idx in range(m):
                        mip.addConstr(
                            gp.quicksum(patterns[p_idx].get(t_idx, 0) * lam_int[p_idx] 
                                       for p_idx in range(len(patterns))) >= item_types[t_idx][2],
                            name=f"dem_{t_idx}"
                        )
                    
                    # Set cutoff
                    mip.setParam('Cutoff', best_obj - 0.5)
                    
                    best_bins_cg = None
                    
                    def callback_cg(model, where):
                        nonlocal best_obj, best_bins_cg
                        if where == GRB.Callback.MIPSOL:
                            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                            obj_int = int(round(obj))
                            if obj_int < best_obj:
                                best_obj = obj_int
                                # Extract solution
                                bins_sol = []
                                for p_idx in range(len(patterns)):
                                    cnt = int(round(model.cbGetSolution(lam_int[p_idx])))
                                    for _ in range(cnt):
                                        bin_items = []
                                        for t_idx, c in patterns[p_idx].items():
                                            for __ in range(c):
                                                bin_items.append(item_types[t_idx][1])
                                        if bin_items:
                                            bins_sol.append(bin_items)
                                best_bins_cg = bins_sol
                                if logger:
                                    logger.log(best_obj)
                    
                    mip.optimize(callback_cg)
                    
                    if mip.SolCount > 0 and best_bins_cg is not None:
                        solution = {
                            "objective_value": float(best_obj),
                            "num_bins": best_obj,
                            "bins": best_bins_cg
                        }
                        with open(solution_path, 'w') as f:
                            json.dump(solution, f)
                        return
                    elif mip.SolCount > 0:
                        bins_sol = []
                        for p_idx in range(len(patterns)):
                            cnt = int(round(lam_int[p_idx].X))
                            for _ in range(cnt):
                                bin_items = []
                                for t_idx, c in patterns[p_idx].items():
                                    for __ in range(c):
                                        bin_items.append(item_types[t_idx][1])
                                if bin_items:
                                    bins_sol.append(bin_items)
                        obj_val = len(bins_sol)
                        if obj_val < best_obj:
                            best_obj = obj_val
                            if logger:
                                logger.log(best_obj)
                            solution = {
                                "objective_value": float(best_obj),
                                "num_bins": best_obj,
                                "bins": bins_sol
                            }
                            with open(solution_path, 'w') as f:
                                json.dump(solution, f)
                            return
                
        except Exception as e:
            # Fall back to heuristic solution
            pass
    
    # Output heuristic solution
    result_bins = [b[1] for b in best_bins]
    solution = {
        "objective_value": float(best_obj),
        "num_bins": best_obj,
        "bins": result_bins
    }
    with open(solution_path, 'w') as f:
        json.dump(solution, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    solve_bin_packing(args.instance_path, args.solution_path, args.time_limit, args.log_path)


if __name__ == '__main__':
    main()
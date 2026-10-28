import argparse
import json
import time
import sys
from collections import defaultdict

def solve():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    num_items = instance['num_items']
    bin_capacity = instance['bin_capacity']
    items = instance['items']
    conflict_edges = instance['conflict_edges']

    start_time = time.time()

    # Build data structures
    item_weight = {}
    for item in items:
        item_weight[item['id']] = item['weight']
    
    item_ids = [item['id'] for item in items]
    
    # Build conflict adjacency
    conflicts = defaultdict(set)
    for edge in conflict_edges:
        a, b = edge[0], edge[1]
        conflicts[a].add(b)
        conflicts[b].add(a)

    def is_feasible_addition(bin_items, bin_weight, item_id):
        """Check if item can be added to bin."""
        if bin_weight + item_weight[item_id] > bin_capacity:
            return False
        for existing in bin_items:
            if existing in conflicts[item_id]:
                return False
        return True

    # Greedy heuristic: First Fit Decreasing with conflict awareness
    # Sort items by weight descending, then by conflict degree descending
    sorted_items = sorted(item_ids, key=lambda x: (item_weight[x], len(conflicts[x])), reverse=True)
    
    bins = []  # Each bin is (set of items, total weight)
    
    for item_id in sorted_items:
        placed = False
        best_bin = -1
        best_remaining = bin_capacity + 1  # We want to minimize remaining space (best fit)
        
        for i, (bin_items, bin_wt) in enumerate(bins):
            if is_feasible_addition(bin_items, bin_wt, item_id):
                remaining = bin_capacity - bin_wt - item_weight[item_id]
                if remaining < best_remaining:
                    best_remaining = remaining
                    best_bin = i
        
        if best_bin >= 0:
            bins[best_bin][0].add(item_id)
            bins[best_bin] = (bins[best_bin][0], bins[best_bin][1] + item_weight[item_id])
        else:
            bins.append((set([item_id]), item_weight[item_id]))
    
    best_solution = [set(b[0]) for b in bins]
    best_obj = len(best_solution)
    
    if logger:
        logger.log(best_obj)

    # Try to improve with MIP if instance is not too large
    elapsed = time.time() - start_time
    remaining_time = args.time_limit - elapsed - 1
    
    if remaining_time > 2 and num_items <= 500:
        try:
            import gurobipy as gp
            from gurobipy import GRB
            
            # Upper bound from heuristic
            UB = best_obj
            # Lower bound: max of ceil(total_weight/capacity), chromatic number lower bound (max clique)
            total_weight = sum(item_weight[i] for i in item_ids)
            LB_weight = (total_weight + bin_capacity - 1) // bin_capacity
            
            # Simple clique-based lower bound from conflicts
            max_clique_size = 1
            if conflict_edges:
                # Greedy coloring to get lower bound
                # Actually, for LB we need clique, greedy gives UB on chromatic number
                # Use degree-based heuristic for clique
                max_clique_size = 1
                for item_id in item_ids:
                    if len(conflicts[item_id]) + 1 > max_clique_size:
                        # Check neighborhood for clique
                        neighbors = sorted(conflicts[item_id], key=lambda x: len(conflicts[x]), reverse=True)
                        clique = [item_id]
                        for n in neighbors:
                            if all(n in conflicts[c] for c in clique):
                                clique.append(n)
                        if len(clique) > max_clique_size:
                            max_clique_size = len(clique)
            
            LB = max(LB_weight, max_clique_size)
            
            num_bins = UB  # Use heuristic solution as upper bound on bins
            
            model = gp.Model("binpacking")
            model.setParam('TimeLimit', max(1, remaining_time))
            model.setParam('Threads', 1)
            model.setParam('OutputFlag', 0)
            
            # Variables
            # x[i,j] = 1 if item i is in bin j
            # y[j] = 1 if bin j is used
            
            x = {}
            y = {}
            
            for j in range(num_bins):
                y[j] = model.addVar(vtype=GRB.BINARY, name=f"y_{j}")
            
            for i in item_ids:
                for j in range(num_bins):
                    x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
            
            model.update()
            
            # Objective: minimize bins used
            model.setObjective(gp.quicksum(y[j] for j in range(num_bins)), GRB.MINIMIZE)
            
            # Each item assigned to exactly one bin
            for i in item_ids:
                model.addConstr(gp.quicksum(x[i, j] for j in range(num_bins)) == 1)
            
            # Capacity constraints
            for j in range(num_bins):
                model.addConstr(
                    gp.quicksum(item_weight[i] * x[i, j] for i in item_ids) <= bin_capacity * y[j]
                )
            
            # Conflict constraints
            for edge in conflict_edges:
                a, b = edge[0], edge[1]
                for j in range(num_bins):
                    model.addConstr(x[a, j] + x[b, j] <= 1)
            
            # Symmetry breaking: bins used in order
            for j in range(num_bins - 1):
                model.addConstr(y[j] >= y[j + 1])
            
            # Provide initial solution from heuristic
            for j in range(num_bins):
                if j < len(best_solution):
                    y[j].Start = 1.0
                    for i in item_ids:
                        if i in best_solution[j]:
                            x[i, j].Start = 1.0
                        else:
                            x[i, j].Start = 0.0
                else:
                    y[j].Start = 0.0
                    for i in item_ids:
                        x[i, j].Start = 0.0
            
            # Add lower bound constraint
            model.addConstr(gp.quicksum(y[j] for j in range(num_bins)) >= LB)
            
            def callback(model, where):
                if where == GRB.Callback.MIPSOL:
                    obj = int(round(model.cbGet(GRB.Callback.MIPSOL_OBJBND)))
                    obj_val = int(round(model.cbGet(GRB.Callback.MIPSOL_OBJ)))
                    nonlocal best_obj, best_solution
                    if obj_val < best_obj:
                        # Extract solution
                        new_bins = []
                        for j in range(num_bins):
                            yval = model.cbGetSolution(y[j])
                            if yval > 0.5:
                                bin_items = set()
                                for i in item_ids:
                                    if model.cbGetSolution(x[i, j]) > 0.5:
                                        bin_items.add(i)
                                if bin_items:
                                    new_bins.append(bin_items)
                        best_solution = new_bins
                        best_obj = obj_val
                        if logger:
                            logger.log(best_obj)
            
            model.optimize(callback)
            
            if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED]:
                if model.SolCount > 0:
                    obj_val = int(round(model.ObjVal))
                    if obj_val < best_obj:
                        new_bins = []
                        for j in range(num_bins):
                            if y[j].X > 0.5:
                                bin_items = set()
                                for i in item_ids:
                                    if x[i, j].X > 0.5:
                                        bin_items.add(i)
                                if bin_items:
                                    new_bins.append(bin_items)
                        best_solution = new_bins
                        best_obj = obj_val
                        if logger:
                            logger.log(best_obj)
        except Exception as e:
            pass  # Fall back to heuristic solution
    elif remaining_time > 2:
        # For larger instances, try local search improvements
        # Try to reduce bins by merging
        def try_merge_bins(solution):
            improved = True
            while improved and time.time() - start_time < args.time_limit - 1:
                improved = False
                n_bins = len(solution)
                # Try to empty smallest bins by redistributing items
                bin_sizes = [(sum(item_weight[i] for i in b), idx) for idx, b in enumerate(solution)]
                bin_sizes.sort()
                
                for bw, bidx in bin_sizes:
                    if time.time() - start_time > args.time_limit - 1:
                        break
                    # Try to move all items from this bin to other bins
                    items_to_move = list(solution[bidx])
                    moved_all = True
                    moves = []  # (item, target_bin)
                    
                    remaining_items = list(items_to_move)
                    # Sort by weight descending for better packing
                    remaining_items.sort(key=lambda x: item_weight[x], reverse=True)
                    
                    temp_bins = [set(b) for b in solution]
                    temp_weights = [sum(item_weight[i] for i in b) for b in temp_bins]
                    
                    for item in remaining_items:
                        placed = False
                        # Remove from current bin temporarily
                        temp_bins[bidx].discard(item)
                        temp_weights[bidx] -= item_weight[item]
                        
                        best_target = -1
                        best_fit = bin_capacity + 1
                        
                        for tidx in range(len(temp_bins)):
                            if tidx == bidx:
                                continue
                            if temp_weights[tidx] + item_weight[item] <= bin_capacity:
                                # Check conflicts
                                conflict_free = True
                                for existing in temp_bins[tidx]:
                                    if existing in conflicts[item]:
                                        conflict_free = False
                                        break
                                if conflict_free:
                                    remaining = bin_capacity - temp_weights[tidx] - item_weight[item]
                                    if remaining < best_fit:
                                        best_fit = remaining
                                        best_target = tidx
                        
                        if best_target >= 0:
                            temp_bins[best_target].add(item)
                            temp_weights[best_target] += item_weight[item]
                            moves.append((item, best_target))
                        else:
                            # Can't move this item, restore
                            temp_bins[bidx].add(item)
                            temp_weights[bidx] += item_weight[item]
                            moved_all = False
                            break
                    
                    if moved_all and len(temp_bins[bidx]) == 0:
                        # Successfully emptied this bin
                        new_solution = [b for idx, b in enumerate(temp_bins) if idx != bidx and len(b) > 0]
                        solution = new_solution
                        improved = True
                        break
            
            return solution
        
        best_solution = try_merge_bins(best_solution)
        if len(best_solution) < best_obj:
            best_obj = len(best_solution)
            if logger:
                logger.log(best_obj)
        
        # Swap-based local search
        def local_search(solution, time_limit_ls):
            nonlocal best_obj
            improved = True
            while improved and time.time() - start_time < time_limit_ls:
                improved = False
                # Try swapping items between bins to free up space for merging
                n = len(solution)
                for i in range(n):
                    if time.time() - start_time > time_limit_ls:
                        break
                    for j in range(i + 1, n):
                        if time.time() - start_time > time_limit_ls:
                            break
                        # Try moving one item from i to j and one from j to i
                        for item_i in list(solution[i]):
                            for item_j in list(solution[j]):
                                if time.time() - start_time > time_limit_ls:
                                    break
                                # Check if swap is feasible
                                wi = item_weight[item_i]
                                wj = item_weight[item_j]
                                
                                weight_i = sum(item_weight[x] for x in solution[i])
                                weight_j = sum(item_weight[x] for x in solution[j])
                                
                                new_weight_i = weight_i - wi + wj
                                new_weight_j = weight_j - wj + wi
                                
                                if new_weight_i > bin_capacity or new_weight_j > bin_capacity:
                                    continue
                                
                                # Check conflicts for item_j in bin i (without item_i)
                                ok_i = True
                                for x in solution[i]:
                                    if x != item_i and x in conflicts[item_j]:
                                        ok_i = False
                                        break
                                if not ok_i:
                                    continue
                                
                                ok_j = True
                                for x in solution[j]:
                                    if x != item_j and x in conflicts[item_i]:
                                        ok_j = False
                                        break
                                if not ok_j:
                                    continue
                                
                                # Perform swap
                                solution[i].discard(item_i)
                                solution[i].add(item_j)
                                solution[j].discard(item_j)
                                solution[j].add(item_i)
                
                # Try merging again
                new_solution = try_merge_bins(solution)
                if len(new_solution) < len(solution):
                    solution = new_solution
                    improved = True
                    if len(solution) < best_obj:
                        best_obj = len(solution)
                        if logger:
                            logger.log(best_obj)
            
            return solution
        
        best_solution = local_search(best_solution, args.time_limit + start_time - 1)
        best_obj = len(best_solution)

    # Format output
    output_bins = {}
    for idx, bin_items in enumerate(best_solution):
        bin_id = str(idx)
        items_list = sorted(list(bin_items))
        total_w = sum(item_weight[i] for i in bin_items)
        output_bins[bin_id] = {
            "items": items_list,
            "total_weight": total_w
        }
    
    solution = {
        "objective_value": best_obj,
        "bins": output_bins
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == '__main__':
    solve()
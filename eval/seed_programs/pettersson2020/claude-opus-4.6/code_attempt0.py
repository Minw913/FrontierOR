import argparse
import json
import time
import itertools
from copy import deepcopy

def solve(instance_path, solution_path, time_limit, log_path=None):
    from solution_logger import SolutionLogger
    
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    
    num_items = instance['num_items']
    num_objectives = instance['num_objectives']
    weights = instance['weights']
    capacity = instance['capacity']
    obj_coeffs = instance['objective_coefficients']
    
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    start_time = time.time()
    deadline = start_time + time_limit - 1  # Leave 1 second buffer
    
    # We need to find all non-dominated objective vectors achievable by feasible knapsack selections.
    # This is a multi-objective 0/1 knapsack problem.
    
    # For small instances, we can enumerate. For larger ones, we use dynamic programming
    # on weight with Pareto-optimal objective vector tracking.
    
    # Strategy: DP over items. State = weight used so far. For each weight, maintain
    # a set of non-dominated objective vectors (with corresponding solutions).
    
    # dp[w] = list of (obj_vector, selection) that are non-dominated at weight w
    
    # But we need to be careful about efficiency. With many items and large capacity,
    # this can blow up. Let's see the typical instance size.
    
    # Since weights are 60-100 and capacity ~ half sum of weights, for n items,
    # capacity ~ n*80/2 = 40n. For n=20, capacity~800.
    
    # The key challenge is the number of Pareto-optimal vectors at each weight state.
    
    def is_dominated(vec, pareto_set):
        """Check if vec is dominated by any vector in pareto_set."""
        for other in pareto_set:
            if all(other[j] <= vec[j] for j in range(num_objectives)) and \
               any(other[j] < vec[j] for j in range(num_objectives)):
                return True
        return False
    
    def add_to_pareto(vec, sol, pareto_list):
        """Add (vec, sol) to pareto_list, removing dominated entries. Returns True if added."""
        # Check if vec is dominated
        new_list = []
        dominated = False
        for (ov, os) in pareto_list:
            # Check if ov dominates vec
            if not dominated:
                if all(ov[j] <= vec[j] for j in range(num_objectives)) and \
                   any(ov[j] < vec[j] for j in range(num_objectives)):
                    dominated = True
                    # vec is dominated, keep all existing
                    return pareto_list, False
            # Check if vec dominates ov
            if all(vec[j] <= ov[j] for j in range(num_objectives)) and \
               any(vec[j] < ov[j] for j in range(num_objectives)):
                continue  # ov is dominated by vec, remove it
            # Check if identical
            if all(vec[j] == ov[j] for j in range(num_objectives)):
                # Same objective vector, no need to add
                return pareto_list, False
            new_list.append((ov, os))
        
        new_list.append((tuple(vec), sol))
        return new_list, True
    
    def merge_pareto(list1, list2):
        """Merge two pareto lists."""
        result = list(list1)
        for (vec, sol) in list2:
            result, _ = add_to_pareto(vec, sol, result)
        return result
    
    # For efficiency, let's try a different approach for small num_items:
    # Direct enumeration if num_items <= 20 or so
    
    if num_items <= 23:
        # Direct enumeration with pruning
        best_pareto = []  # list of (obj_vector_tuple, selection_list)
        
        # Use branch and bound / enumeration
        # We'll enumerate all 2^n subsets but with pruning
        
        def enumerate_subsets():
            nonlocal best_pareto
            # Iterate through all subsets using gray code or simple iteration
            count = 0
            total = 1 << num_items
            
            # Precompute
            current_weight = 0
            current_obj = [0] * num_objectives
            selection = [0] * num_items
            
            # Simple enumeration
            for mask in range(total):
                if time.time() > deadline:
                    break
                
                # Compute weight and objectives
                w = 0
                obj = [0] * num_objectives
                sel = [0] * num_items
                feasible = True
                for i in range(num_items):
                    if mask & (1 << i):
                        w += weights[i]
                        if w > capacity:
                            feasible = False
                            break
                        sel[i] = 1
                        for k in range(num_objectives):
                            obj[k] += obj_coeffs[k][i]
                
                if not feasible:
                    continue
                
                obj_tuple = tuple(obj)
                best_pareto, added = add_to_pareto(obj_tuple, sel, best_pareto)
            
        enumerate_subsets()
        
        # Output
        result = {
            "objective_value": len(best_pareto),
            "non_dominated_objective_vectors": [list(ov) for (ov, _) in best_pareto],
            "solutions": [s for (_, s) in best_pareto]
        }
        
        if logger:
            logger.log(len(best_pareto))
        
        with open(solution_path, 'w') as f:
            json.dump(result, f)
        return
    
    # For larger instances, use DP approach
    # dp[w] = list of (obj_vector_tuple, selection_as_frozenset_of_indices)
    # To save memory, store selections as frozensets of included item indices
    
    # Actually, let's use a different DP formulation:
    # Process items one by one. Maintain a global Pareto front per weight.
    # dp is a dict: weight -> list of (obj_tuple, item_set)
    
    # Initialize with empty selection
    dp = {}
    dp[0] = [(tuple([0]*num_objectives), frozenset())]
    
    items_processed = 0
    
    for i in range(num_items):
        if time.time() > deadline:
            break
        
        items_processed += 1
        wi = weights[i]
        ci = [obj_coeffs[k][i] for k in range(num_objectives)]
        
        new_dp = {}
        
        # First copy existing dp (not taking item i)
        for w, pareto_list in dp.items():
            if w not in new_dp:
                new_dp[w] = list(pareto_list)
            else:
                new_dp[w] = merge_pareto(new_dp[w], pareto_list)
        
        # Now consider taking item i
        for w, pareto_list in dp.items():
            new_w = w + wi
            if new_w > capacity:
                continue
            
            new_entries = []
            for (ov, items) in pareto_list:
                new_ov = tuple(ov[k] + ci[k] for k in range(num_objectives))
                new_items = items | {i}
                new_entries.append((new_ov, new_items))
            
            if new_w not in new_dp:
                new_dp[new_w] = new_entries
            else:
                new_dp[new_w] = merge_pareto(new_dp[new_w], new_entries)
        
        dp = new_dp
        
        # Periodically check time
        if time.time() > deadline:
            break
    
    # Collect all Pareto-optimal vectors across all weights
    global_pareto = []
    for w, pareto_list in dp.items():
        global_pareto = merge_pareto(global_pareto, pareto_list)
    
    # Convert to output format
    solutions = []
    obj_vectors = []
    for (ov, item_set) in global_pareto:
        obj_vectors.append(list(ov))
        sel = [0] * num_items
        for idx in item_set:
            sel[idx] = 1
        solutions.append(sel)
    
    if logger:
        logger.log(len(global_pareto))
    
    result = {
        "objective_value": len(global_pareto),
        "non_dominated_objective_vectors": obj_vectors,
        "solutions": solutions
    }
    
    with open(solution_path, 'w') as f:
        json.dump(result, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)
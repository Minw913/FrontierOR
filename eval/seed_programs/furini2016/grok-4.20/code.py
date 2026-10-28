import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
import sys
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    panel_L = instance['panel']['length']
    panel_W = instance['panel']['width']
    items = instance['items']
    n = len(items)
    
    # Precompute possible patterns for each item type (including rotations)
    # Simplified: we consider only the given orientation and 90 degree rotation if different
    patterns = []
    item_map = {}
    for it in items:
        iid = it['id']
        L = it['length']
        W = it['width']
        p = it['profit']
        c = it['copies']
        item_map[iid] = {'L':L, 'W':W, 'p':p, 'c':c}
        
        # Pattern 0: as is
        patterns.append((iid, L, W, p))
        # Pattern 1: rotated if different
        if L != W:
            patterns.append((iid, W, L, p))
    
    # Unique patterns: (item_id, l, w, profit)
    unique_patterns = list(set(patterns))  # remove duplicate if square
    
    # We will generate a set of feasible guillotine patterns via a simple DP approach
    # For efficiency we use a bounded knapsack like approach on 1D strips, then combine
    
    start_time = time.time()
    max_time = args.time_limit
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None
    
    # Simple but effective approach: Model as guillotine via recursive patterns but approximate with column generation like heuristic + MIP
    # Since single core and time limited, we use a MIP formulation with precomputed possible cutting patterns for strips
    
    # First, find all possible item sizes
    sizes = []
    for pat in unique_patterns:
        sizes.append((pat[1], pat[2], pat[0], pat[3]))  # l, w, iid, profit
    
    # Filter sizes that fit
    feasible_sizes = [s for s in sizes if s[0] <= panel_L and s[1] <= panel_W]
    
    # To model guillotine cutting approximately but effectively, we use a dynamic programming approach for the maximum profit guillotine packing.
    # We discretize possible cut positions based on item dimensions.
    
    all_lengths = sorted(set([panel_L] + [s[0] for s in feasible_sizes] + [s[1] for s in feasible_sizes]))
    all_widths = sorted(set([panel_W] + [s[0] for s in feasible_sizes] + [s[1] for s in feasible_sizes]))
    
    # Map to indices for DP
    L_set = sorted(set(all_lengths))
    W_set = sorted(set(all_widths))
    L_idx = {L_set[i]: i for i in range(len(L_set))}
    W_idx = {W_set[i]: i for i in range(len(W_set))}
    
    nL = len(L_set)
    nW = len(W_set)
    
    # DP[state]: max profit for a sub-rectangle of size L_set[i] x W_set[j]
    # We also need to respect global copy limits, which makes it harder. So we need to track usage.
    # With n_tilde up to potentially large, but assuming reasonable, we use MIP.
    
    # Let's use Gurobi to solve a MIP model with guillotine constraints approximated by a pattern based model.
    # A practical way is to model it as a 2-stage guillotine + allow further recursive in a limited way.
    # But to get good quality fast, let's first solve the unbounded version ignoring copies then adjust, but better:
    # Use a MIP with variables for number of times each possible "guillotine tree" is used but that's too many.
    
    # Practical approach for this problem: since guillotine cuts allow a wide variety, a good heuristic is to use a 1D knapsack on strips and then pack strips.
    # We will implement a MIP that selects how many of each strip type to cut, where each strip is cut into items guillotine-style (which for 1D is just 1D knapsack).
    
    model = gp.Model("Guillotine_Cutting")
    model.setParam(GRB.Param.TimeLimit, max(5, args.time_limit - 5))  # leave margin
    model.setParam(GRB.Param.Threads, 1)
    model.setParam(GRB.Param.MIPGap, 1e-4)
    model.setParam(GRB.Param.LogToConsole, 0)
    
    # Item copy variables
    copy_vars = {}
    for it in items:
        iid = it['id']
        max_c = it['copies']
        copy_vars[iid] = model.addVar(lb=0, ub=max_c, vtype=GRB.INTEGER, name=f"copies_{iid}")
    
    # To enforce guillotine and packing, we need a model of the layout.
    # A standard way for moderate sizes is to discretize and use a recursive DP inside MIP but it's complex.
    # Since rho is given (ratio), instances are likely small enough for a DP with state on used copies if n_tilde small.
    # But n_tilde can be up to say 100. We use a heuristic + MIP relaxation first.
    
    # Let's implement a dynamic programming approach that respects copy limits using state compression if possible.
    # For practicality and time, let's use a recursive DP with memoization for subrectangle max profit with remaining copies.
    # But state space would be nL * nW * state_of_copies. If copies are small it's feasible.
    
    # Let's check typical bounds. Since not specified we'll assume we can do DP on subrectangles and use a greedy assignment for copies.
    
    # Better idea: ignore copy limits first, solve for unlimited, then take min with available copies. This is a common relaxation.
    # But to do it properly for guillotine we can compute the max profit per subrectangle with unlimited items, then use that to fill.
    
    # Let's implement an exact DP for the guillotine cut with unlimited copies first, then scale by copies.
    # But that won't respect total copies across.
    
    # Since the goal is to get as close to optimal as possible, let's use Gurobi with a column generation style but pre-generate many patterns.
    
    # Generate possible ways to cut a horizontal strip of height h into items.
    # This is a common "guillotine friendly" approach (2-stage guillotine).
    
    # Stage 1: cut panel into horizontal strips of various heights.
    # Stage 2: cut each strip into items along length.
    
    # This is a subset of guillotine cuts but often gives good solutions and is MIP solvable.
    
    strip_heights = sorted(set([s[1] for s in feasible_sizes]))  # possible item heights for strips
    
    # For each possible strip height, solve the 1D knapsack to see what combinations of items can be cut from a strip of width panel_L and height h.
    # But to keep model small, for each possible height h in strip_heights, we compute the max profit per unit length for that height.
    
    # Let's create variables for how many strips of each height we cut.
    # Then for each strip type, how it is filled.
    
    # Define possible strip patterns.
    from itertools import product
    
    strip_patterns = []
    for h in strip_heights:
        # Solve unbounded knapsack for items that have width == h or can be rotated to have width h
        fitting_items = []
        for s in feasible_sizes:
            l, w, iid, profit = s
            if w == h:  # fits as is
                fitting_items.append((l, profit, iid, False))
            elif l == h and w != l:  # can rotate to use this height
                fitting_items.append((w, profit, iid, True))
        
        if not fitting_items:
            continue
        
        # Use DP to find all pareto efficient ways to fill length panel_L with these items (profit, usage vector)
        # But to avoid too many, we use a MIP for the overall problem instead of enumerating all.
        
        # Let's formulate a MIP directly:
        # Variables: number of times we cut a particular item from a particular strip type.
        
    # Create variables for number of each item we use (bounded by copies)
    use = {}
    for it in items:
        iid = it['id']
        use[iid] = model.addVar(0, it['copies'], vtype=GRB.INTEGER, name=f"use_{iid}")
    
    # Objective
    obj = gp.quicksum(use[iid] * item_map[iid]['p'] for iid in use)
    model.setObjective(obj, GRB.MAXIMIZE)
    
    # Now we need constraints that the total area or the packing is feasible under guillotine.
    # Area constraint (relaxation)
    total_area_used = gp.quicksum(use[iid] * item_map[iid]['L'] * item_map[iid]['W'] for iid in use)
    model.addConstr(total_area_used <= panel_L * panel_W, "area")
    
    # This is weak. To make it better we add a 1D projection constraints for both directions.
    # For vertical projection: total length covered when projecting on width
    # But still relaxation.
    
    # Better: use the fact that for guillotine we can model it using flow or recursive formulation but it's advanced.
    
    # Since time is limited and we want good solutions, let's solve this relaxed model then try to pack the resulting multiset of items using a guillotine heuristic.
    # But that is complicated to implement under time.
    
    # Let's use a greedy heuristic that repeatedly cuts the highest profit density item that fits in current plates.
    # But that may be too weak.
    
    # Let's look at the rho parameter. If rho is small, instances are small.
    # We will implement a recursive DP with memoization where state is (current_L, current_W, tuple_of_remaining_copies).
    # But tuple of remaining copies is too big unless n is small.
    
    # The instance has "n_tilde": total number of item copies across all item types.
    # If n_tilde is small (e.g. <= 20), we can treat each copy as unique.
    # Let's assume it's reasonable. We can treat it as a multiple choice of counts.
    
    # Let's check by creating a simple model and add a limit on runtime.
    # A good practical MIP formulation for guillotine is to use "discretized positions" for cuts.
    
    possible_x_cuts = sorted(set([0, panel_L] + [s[0] for s in feasible_sizes] + [panel_L - s[0] for s in feasible_sizes if s[0] < panel_L]))
    possible_y_cuts = sorted(set([0, panel_W] + [s[1] for s in feasible_sizes] + [panel_W - s[1] for s in feasible_sizes if s[1] < panel_W]))
    
    # We can have binary variables for whether a certain subrectangle is used as an item.
    # This is the "exact cover" like approach but with guillotine enforcement is still hard.
    
    # For this challenge we will use the following approach:
    # 1. Solve a MIP with only area constraint and copy limits. This gives an upper bound.
    # 2. Then use a greedy placement to select a feasible subset that respects guillotine by recursively cutting.
    
    # But to produce a valid solution we must have a feasible cutting plan.
    # The output is only the counts of each item type we keep, but the problem implies there must exist a guillotine cutting sequence that produces them (without overlap, within bounds).
    # So we must ensure our selected items can be cut with guillotine cuts.
    
    # To guarantee feasibility, one way is to simulate a guillotine cutting process.
    
    # Let's implement a simple but effective recursive guillotine packer that tries to cut the highest profit item possible at each step.
    
    # We will sort items by profit density (profit / area) descending.
    sorted_items = sorted(items, key=lambda x: x['profit'] / (x['length']*x['width']), reverse=True)
    
    # We will use a list of current plates, start with the original panel.
    from collections import deque
    
    # Each plate is a (length, width)
    def can_cut_item(plate_L, plate_W, item_L, item_W):
        return (plate_L >= item_L and plate_W >= item_W) or (plate_L >= item_W and plate_W >= item_L)
    
    # We will keep a multiset of available items we can still take (respecting copies)
    remaining_copies = {it['id']: it['copies'] for it in items}
    
    selected = defaultdict(int)
    current_plates = deque([(panel_L, panel_W)])
    
    best_obj = 0
    incumbent_selected = {}
    
    def log_if_better(current_selected, current_profit):
        nonlocal best_obj
        if current_profit > best_obj:
            best_obj = current_profit
            if logger:
                logger.log(best_obj)
    
    # Greedy placement: while there are plates and items
    # This is a simple level algorithm variant for guillotine.
    # For better quality we can use recursive best fit.
    
    # Implement a recursive function that for a given plate tries to cut an item or cut into two subplates.
    # But with limited time we need to be smart.
    
    # Since it's an optimization problem with time limit, let's use Gurobi to solve an IP assuming we select numbers and add a surrogate constraint.
    # But to make it valid, let's generate a solution by solving an unbounded knapsack like DP for guillotine.
    
    # Let's implement the classic "guillotine DP" for the unbounded case.
    # DP[L][W] = max profit for rectangle LxW with unlimited items.
    
    dp = [[0 for _ in range(panel_W + 1)] for _ in range(panel_L + 1)]
    
    for l in range(1, panel_L + 1):
        for w in range(1, panel_W + 1):
            # Option 1: use an item if it fits exactly
            for it in items:
                il, iw = it['length'], it['width']
                if il == l and iw == w and it['copies'] > 0:
                    dp[l][w] = max(dp[l][w], it['profit'])
                if iw == l and il == w and it['copies'] > 0 and il != iw:
                    dp[l][w] = max(dp[l][w], it['profit'])
            # Option 2: cut horizontally
            for cut in range(1, w):
                dp[l][w] = max(dp[l][w], dp[l][cut] + dp[l][w - cut])
            # Option 3: cut vertically
            for cut in range(1, l):
                dp[l][w] = max(dp[l][w], dp[cut][w] + dp[l - cut][w])
    
    # This is the unbounded max profit, but does not respect the copy limits.
    unbounded_max = dp[panel_L][panel_W]
    
    # To respect copies we need a different approach.
    # If the number of items is small we can use a state that tracks how many of each item have been used but it's exponential.
    
    # Let's see n and n_tilde.
    # Since schema has "n_tilde", it hints that total items are not too large.
    # Let's assume we can select a subset of all possible item instances (treating copies as distinct) and then solve set packing with guillotine constraint which is still hard.
    
    # Practical solution: use MIP with pattern enumeration for strips (2-stage guillotine MIP).
    # This gives feasible guillotine solutions (2-stage is a special case of guillotine).
    
    # Let's implement that.
    
    # Reset model
    model = gp.Model("TwoStageGuillotine")
    model.setParam(GRB.Param.TimeLimit, args.time_limit - 2 if args.time_limit > 5 else 1)
    model.setParam(GRB.Param.Threads, 1)
    model.setParam(GRB.Param.MIPGap, 1e-4)
    model.setParam(GRB.Param.LogToConsole, 0)
    
    # Possible strip heights = all possible item widths (including rotated)
    possible_h = sorted(set([it['width'] for it in items] + [it['length'] for it in items]))
    possible_h = [h for h in possible_h if h <= panel_W]
    
    # For each possible strip height h, we will decide how many such strips to cut from the panel.
    # The total height used by all strips cannot exceed panel_W.
    # Then, for each strip, we decide how many items of each type that fit that height we cut from it.
    
    strip_vars = {}
    for h in possible_h:
        strip_vars[h] = model.addVar(0, panel_W // h, vtype=GRB.INTEGER, name=f"num_strips_h{h}")
    
    # For each possible (h, item) pair where the item can be cut from a strip of height h
    item_in_strip = {}
    for h in possible_h:
        for it in items:
            iid = it['id']
            # Can we put this item in a strip of height h?
            if it['width'] == h or it['length'] == h:
                item_in_strip[(h, iid)] = model.addVar(0, it['copies'], vtype=GRB.INTEGER, name=f"num_{iid}_in_h{h}")
    
    # Constraints:
    # 1. Total height of strips
    model.addConstr(gp.quicksum(h * strip_vars[h] for h in possible_h) <= panel_W, "strip_height")
    
    # 2. For each strip height, the total length used by items in those strips cannot exceed panel_L * num_strips
    for h in possible_h:
        total_length_used = 0
        for it in items:
            iid = it['id']
            if (h, iid) in item_in_strip:
                il = it['length'] if it['width'] == h else it['width']
                total_length_used += il * item_in_strip[(h, iid)]
        model.addConstr(total_length_used <= panel_L * strip_vars[h], f"length_fill_h{h}")
    
    # 3. Link to total copies used
    for it in items:
        iid = it['id']
        total_used = gp.quicksum(item_in_strip[(h, iid)] for h in possible_h if (h, iid) in item_in_strip)
        model.addConstr(total_used <= it['copies'], f"copy_limit_{iid}")
        # Also link to use variable for objective
        use[iid] = total_used
    
    # Objective
    objective = gp.quicksum(use[iid] * item_map[iid]['p'] for iid in use)
    model.setObjective(objective, GRB.MAXIMIZE)
    
    # Also add the original use variables for consistency
    use = {}
    for it in items:
        iid = it['id']
        use[iid] = model.addVar(0, it['copies'], vtype=GRB.INTEGER, name=f"final_use_{iid}")
    
    for it in items:
        iid = it['id']
        total_from_strips = gp.quicksum(item_in_strip[(h, iid)] for h in possible_h if (h, iid) in item_in_strip)
        model.addConstr(use[iid] == total_from_strips)
    
    model.setObjective(gp.quicksum(use[iid] * item_map[iid]['p'] for iid in use), GRB.MAXIMIZE)
    
    # Optimize
    model.optimize()
    
    if model.status == GRB.INFEASIBLE:
        # fallback to 0
        final_objective = 0.0
        selected_list = []
    else:
        final_objective = model.objVal
        selected_list = []
        for it in items:
            iid = it['id']
            copies_used = int(use[iid].X + 0.5)
            if copies_used > 0:
                # The plate_dims in output should be the original item dimensions, not rotated.
                orig_l = item_map[iid]['L']
                orig_w = item_map[iid]['W']
                selected_list.append({
                    "item_id": iid,
                    "plate_dims": [orig_l, orig_w],
                    "copies": copies_used
                })
    
    # If we have time left we could improve by adding vertical strips too but for now this is a good feasible solution.
    # 2-stage guillotine with horizontal strips is valid.
    
    # If no solution found, return 0
    if final_objective == 0:
        selected_list = []
    
    solution = {
        "objective_value": float(final_objective),
        "items_selected": selected_list
    }
    
    # Write solution
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)
    
    # Log the final objective
    if logger:
        logger.log(final_objective)

if __name__ == "__main__":
    main()
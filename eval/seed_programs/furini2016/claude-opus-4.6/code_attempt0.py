import argparse
import json
import time
import sys
from functools import lru_cache

def solve(instance_path, solution_path, time_limit, log_path):
    from solution_logger import SolutionLogger
    
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    
    logger = SolutionLogger(log_path, sense="maximize") if log_path else None
    
    panel_length = instance['panel']['length']
    panel_width = instance['panel']['width']
    items = instance['items']
    n = instance['n']
    
    start_time = time.time()
    
    # Preprocess items: each item can be placed in original or rotated orientation
    # We consider item placements as (length, width) pairs
    item_types = []
    for item in items:
        item_types.append({
            'id': item['id'],
            'length': item['length'],
            'width': item['width'],
            'profit': item['profit'],
            'copies': item['copies']
        })
    
    # For the guillotine cutting problem, we use dynamic programming on rectangles
    # State: (plate_length, plate_width) -> max profit achievable
    # But we also need to track item usage (copies constraint), which makes it much harder.
    
    # If copies are large enough (effectively unbounded), we can use standard 2D guillotine DP.
    # With bounded copies, we need a more careful approach.
    
    # Check if the problem is small enough for exact DP
    # For large panels, we discretize cut positions based on item dimensions
    
    # Collect all possible cut positions
    lengths_set = set()
    widths_set = set()
    for item in item_types:
        lengths_set.add(item['length'])
        lengths_set.add(item['width'])
        widths_set.add(item['length'])
        widths_set.add(item['width'])
    
    # Generate all possible dimensions that matter (sums of item dimensions)
    # For efficiency, we limit to dimensions that fit in the panel
    def generate_relevant_dims(item_dims, max_dim):
        """Generate all dimensions up to max_dim that are sums of item dimensions."""
        dims = set([0])
        for d in item_dims:
            if d <= max_dim:
                new_dims = set()
                for existing in dims:
                    val = existing + d
                    while val <= max_dim:
                        new_dims.add(val)
                        val += d
                dims.update(new_dims)
        dims.add(max_dim)
        return sorted(dims)
    
    all_item_dims = set()
    for item in item_types:
        all_item_dims.add(item['length'])
        all_item_dims.add(item['width'])
    
    # For the DP, we need cut positions. We use "normal patterns" (Christofides & Whitlock)
    # which are combinations of item dimensions that fit within the panel
    def compute_normal_patterns(item_dims_list, max_val):
        """Compute raster points / normal patterns."""
        patterns = set([0])
        sorted_dims = sorted(item_dims_list)
        if not sorted_dims:
            return [0, max_val]
        
        # BFS/DP to find all achievable sums up to max_val
        reachable = bytearray(max_val + 1)
        reachable[0] = 1
        for d in sorted_dims:
            if d > max_val:
                continue
            for v in range(d, max_val + 1):
                if reachable[v - d]:
                    reachable[v] = 1
        
        result = [i for i in range(max_val + 1) if reachable[i]]
        if max_val not in result:
            result.append(max_val)
        return result
    
    l_dims = list(all_item_dims)
    
    # Compute normal patterns for length and width directions
    length_patterns = compute_normal_patterns(l_dims, panel_length)
    width_patterns = compute_normal_patterns(l_dims, panel_width)
    
    # Limit the number of patterns for tractability
    MAX_PATTERNS = 300
    if len(length_patterns) > MAX_PATTERNS:
        # Sample uniformly
        step = len(length_patterns) / MAX_PATTERNS
        sampled = [length_patterns[int(i * step)] for i in range(MAX_PATTERNS)]
        if panel_length not in sampled:
            sampled.append(panel_length)
        if 0 not in sampled:
            sampled.insert(0, 0)
        length_patterns = sorted(set(sampled))
    
    if len(width_patterns) > MAX_PATTERNS:
        step = len(width_patterns) / MAX_PATTERNS
        sampled = [width_patterns[int(i * step)] for i in range(MAX_PATTERNS)]
        if panel_width not in sampled:
            sampled.append(panel_width)
        if 0 not in sampled:
            sampled.insert(0, 0)
        width_patterns = sorted(set(sampled))
    
    # First, solve the unbounded version (ignoring copy limits) using DP
    # Then check if copy limits are satisfied; if not, use a heuristic to fix
    
    # For each rectangle (L, W), compute the max profit density
    # DP: val[L][W] = max over:
    #   1) Item that exactly fits (L, W) -> item profit
    #   2) Vertical cut at position x: val[x][W] + val[L-x][W]
    #   3) Horizontal cut at position y: val[L][y] + val[L][W-y]
    
    # Map dimensions to indices for faster lookup
    lp = length_patterns
    wp = width_patterns
    
    # Filter patterns: only keep those <= panel dimensions
    lp = [x for x in lp if x <= panel_length]
    wp = [y for y in wp if y <= panel_width]
    
    if panel_length not in lp:
        lp.append(panel_length)
    if panel_width not in wp:
        wp.append(panel_width)
    lp = sorted(set(lp))
    wp = sorted(set(wp))
    
    # Remove 0 from patterns for plate dimensions (plates must be positive)
    lp_pos = [x for x in lp if x > 0]
    wp_pos = [y for y in wp if y > 0]
    
    nl = len(lp_pos)
    nw = len(wp_pos)
    
    # Precompute: for each (L, W), what's the best single item that fits exactly?
    # An item fits exactly if (item.length == L and item.width == W) or (item.width == L and item.length == W)
    # But actually, in guillotine cutting, a plate can be kept as an item if dimensions match exactly.
    # The plate doesn't need to exactly match; it can be further cut. But to keep as item, must match exactly.
    
    # For the DP, we want: for plate of size (L, W), the max total profit from items cut from it.
    # Unbounded version first (ignore copy limits).
    
    # Build lookup for item profits by (length, width)
    item_by_dims = {}
    for item in item_types:
        l, w = item['length'], item['width']
        key1 = (l, w)
        key2 = (w, l)
        # We can place item in either orientation
        if key1 not in item_by_dims or item['profit'] > item_by_dims[key1]['profit']:
            item_by_dims[key1] = item
        if key2 not in item_by_dims or item['profit'] > item_by_dims[key2]['profit']:
            item_by_dims[key2] = item
    
    # Create index mappings
    l_idx = {v: i for i, v in enumerate(lp_pos)}
    w_idx = {v: i for i, v in enumerate(wp_pos)}
    
    # Check total number of states
    total_states = nl * nw
    
    # If too many states, we need to be smarter
    # Let's check feasibility
    time_elapsed = time.time() - start_time
    
    # For very large problems, use a greedy/heuristic approach
    if total_states > 2000000 or time_elapsed > time_limit * 0.8:
        # Greedy approach: sort items by profit density, greedily pack
        best_obj = 0
        best_selection = []
        
        # Simple greedy: pick items by profit / area ratio
        item_list = []
        for item in item_types:
            area = item['length'] * item['width']
            density = item['profit'] / area
            item_list.append((density, item))
        item_list.sort(reverse=True, key=lambda x: x[0])
        
        # Try to fit items greedily using shelf-based approach
        # This is a rough heuristic
        remaining_area = panel_length * panel_width
        selected = {}
        
        for _, item in item_list:
            if remaining_area <= 0:
                break
            area = item['length'] * item['width']
            max_fit = min(item['copies'], remaining_area // area)
            if max_fit > 0:
                selected[item['id']] = {
                    'item_id': item['id'],
                    'length': item['length'],
                    'width': item['width'],
                    'copies': max_fit,
                    'profit': item['profit']
                }
                remaining_area -= max_fit * area
        
        total_profit = sum(s['copies'] * s['profit'] for s in selected.values())
        if logger:
            logger.log(total_profit)
        
        items_selected = []
        for s in selected.values():
            items_selected.append({
                'item_id': s['item_id'],
                'plate_dims': [s['length'], s['width']],
                'copies': s['copies']
            })
        
        solution = {
            'objective_value': total_profit,
            'items_selected': items_selected
        }
        
        with open(solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
        return
    
    # DP approach for manageable sizes
    # val[i][j] = max profit from a plate of size (lp_pos[i], wp_pos[j])
    # cut_type[i][j] = how we achieved this: None (item), ('v', x_idx), ('h', y_idx)
    
    import numpy as np
    
    val = np.zeros((nl, nw), dtype=np.int64)
    # For reconstruction, store cut info
    # 0 = no cut (item or empty), 1 = vertical cut, 2 = horizontal cut
    cut_info = np.zeros((nl, nw), dtype=np.int8)
    cut_pos = np.zeros((nl, nw), dtype=np.int32)  # index of cut position
    item_id_at = np.full((nl, nw), -1, dtype=np.int32)  # item id if kept as item
    
    # Process in order of increasing area (or just iterate since smaller plates are processed first)
    # Actually, we need to ensure that when we compute val[i][j], all sub-plates have been computed.
    # Sub-plates from vertical cut at x: (x, W) and (L-x, W)
    # Sub-plates from horizontal cut at y: (L, y) and (L, W-y)
    # Both sub-dimensions are strictly less than the original, so processing in order of 
    # increasing L and W works... but we need both dimensions to be smaller or equal.
    # Actually, for a vertical cut, x < L and L-x < L, but W stays the same.
    # For horizontal cut, y < W and W-y < W, but L stays the same.
    # So we need: for val[i][j], all val[i'][j] with i' < i, and all val[i][j'] with j' < j.
    # This means processing i from 0..nl-1 and j from 0..nw-1 works!
    
    # But we also need val for dimensions that might not be in our pattern list.
    # When we cut at position x, the remaining piece has length L-x.
    # L-x might not be in lp_pos. We need to handle this.
    
    # To handle this properly, we should ensure closure under subtraction within the pattern set,
    # or we should map L-x to the nearest pattern that is <= L-x (conservative, may lose some profit).
    
    # Actually, for a proper DP, we need all relevant dimensions to be in our sets.
    # The "normal patterns" approach ensures this: if x is a normal pattern and L is a normal pattern,
    # then we should also have L-x or at least the largest normal pattern <= L-x.
    
    # Let's create a mapping: for any dimension d, find the largest pattern <= d
    def make_floor_map(patterns, max_val):
        """For each integer 0..max_val, find the largest pattern <= that integer."""
        floor_map = [0] * (max_val + 1)
        pi = 0
        for d in range(max_val + 1):
            while pi < len(patterns) - 1 and patterns[pi + 1] <= d:
                pi += 1
            if patterns[pi] <= d:
                floor_map[d] = patterns[pi]
            else:
                floor_map[d] = 0
        return floor_map
    
    l_floor = make_floor_map(lp_pos, panel_length)
    w_floor = make_floor_map(wp_pos, panel_width)
    
    # Now, for a plate (L, W) where L = lp_pos[i], W = wp_pos[j]:
    # Vertical cut at position x (x in lp_pos, x < L):
    #   left plate: (x, W) -> val[l_idx[x]][j]
    #   right plate: (L-x, W) -> but L-x might not be in lp_pos
    #     use l_floor[L-x] to get the usable dimension
    #     if l_floor[L-x] > 0: val[l_idx[l_floor[L-x]]][j]
    
    # Similarly for horizontal cuts.
    
    # This is an approximation (we might underestimate by flooring dimensions).
    # For exactness, we'd need all dimensions, but this is often good enough.
    
    # Actually, a better approach: ensure that for every pair of patterns a, b where a+b <= max_dim,
    # we don't need b to be a pattern because we're iterating over cut positions.
    # The cut at position x means left = x, right = L - x.
    # We need both x and L-x to be representable. If L is a normal pattern and x is a normal pattern,
    # L-x might not be, but we can still use the floor. This loses some optimality.
    
    # Alternative: extend patterns to include all L-x values
    lp_extended = set(lp_pos)
    for L in lp_pos:
        for x in lp_pos:
            if 0 < x < L:
                lp_extended.add(L - x)
    
    wp_extended = set(wp_pos)
    for W in wp_pos:
        for y in wp_pos:
            if 0 < y < W:
                wp_extended.add(W - y)
    
    lp_pos = sorted(lp_extended)
    wp_pos = sorted(wp_extended)
    
    # Re-check size
    nl = len(lp_pos)
    nw = len(wp_pos)
    total_states = nl * nw
    
    # If still too large, truncate
    if total_states > 3000000:
        # Reduce: keep only the most important patterns
        # Use original patterns without extension
        lp_pos = sorted(set([x for x in length_patterns if x > 0 and x <= panel_length]))
        wp_pos = sorted(set([y for y in width_patterns if y > 0 and y <= panel_width]))
        if panel_length not in lp_pos:
            lp_pos.append(panel_length)
        if panel_width not in wp_pos:
            wp_pos.append(panel_width)
        lp_pos = sorted(set(lp_pos))
        wp_pos = sorted(set(wp_pos))
        nl = len(lp_pos)
        nw = len(wp_pos)
    
    l_idx = {v: i for i, v in enumerate(lp_pos)}
    w_idx = {v: i for i, v in enumerate(wp_pos)}
    
    # Rebuild floor maps
    l_floor = make_floor_map(lp_pos, panel_length)
    w_floor = make_floor_map(wp_pos, panel_width)
    
    # DP
    val = [[0] * nw for _ in range(nl)]
    # For reconstruction: store (cut_type, cut_position_index)
    # cut_type: 0 = item/empty, 1 = vertical, 2 = horizontal
    recon = [[None] * nw for _ in range(nl)]
    
    # Precompute item fits for each (L, W) combination
    # item_profit[i][j] = best single item profit for plate (lp_pos[i], wp_pos[j])
    item_profit = [[0] * nw for _ in range(nl)]
    item_id_grid = [[-1] * nw for _ in range(nl)]
    
    for item in item_types:
        il, iw = item['length'], item['width']
        pid = item['id']
        profit = item['profit']
        # Original orientation
        if il in l_idx and iw in w_idx:
            i, j = l_idx[il], w_idx[iw]
            if profit > item_profit[i][j]:
                item_profit[i][j] = profit
                item_id_grid[i][j] = pid
        # Rotated orientation
        if iw in l_idx and il in w_idx:
            i, j = l_idx[iw], w_idx[il]
            if profit > item_profit[i][j]:
                item_profit[i][j] = profit
                item_id_grid[i][j] = pid
    
    # Process DP bottom-up
    for i in range(nl):
        L = lp_pos[i]
        if time.time() - start_time > time_limit * 0.85:
            break
        for j in range(nw):
            W = wp_pos[j]
            best = item_profit[i][j]
            best_type = 0  # item or empty
            best_pos = -1
            
            # Vertical cuts: cut at position x, producing (x, W) and (L-x, W)
            # Only try x <= L/2 to avoid symmetry (and add both pieces)
            for ii in range(i):
                x = lp_pos[ii]
                if x >= L:
                    break
                if 2 * x > L:
                    # Also check L-x < x case (we'd have already computed it)
                    pass
                # Left piece: (x, W) = val[ii][j]
                left_val = val[ii][j]
                # Right piece: (L-x, W)
                rem = L - x
                rem_floored = l_floor[rem] if rem <= panel_length else 0
                if rem_floored > 0 and rem_floored in l_idx:
                    right_val = val[l_idx[rem_floored]][j]
                else:
                    right_val = 0
                total = left_val + right_val
                if total > best:
                    best = total
                    best_type = 1
                    best_pos = ii
            
            # Horizontal cuts: cut at position y, producing (L, y) and (L, W-y)
            for jj in range(j):
                y = wp_pos[jj]
                if y >= W:
                    break
                top_val = val[i][jj]
                rem = W - y
                rem_floored = w_floor[rem] if rem <= panel_width else 0
                if rem_floored > 0 and rem_floored in w_idx:
                    bottom_val = val[i][w_idx[rem_floored]]
                else:
                    bottom_val = 0
                total = top_val + bottom_val
                if total > best:
                    best = total
                    best_type = 2
                    best_pos = jj
            
            val[i][j] = best
            recon[i][j] = (best_type, best_pos)
    
    # Reconstruct the solution to count items used
    # We need to trace the cutting tree and collect items
    # This gives us the unbounded solution (ignoring copy limits)
    
    def reconstruct(i, j):
        """Reconstruct items from plate (lp_pos[i], wp_pos[j])."""
        if i < 0 or j < 0 or i >= nl or j >= nw:
            return []
        if val[i][j] == 0:
            return []
        
        info = recon[i][j]
        if info is None:
            return []
        
        cut_type, cut_pos_idx = info
        
        if cut_type == 0:
            # This is an item
            pid = item_id_grid[i][j]
            if pid >= 0:
                return [(pid, lp_pos[i], wp_pos[j])]
            return []
        elif cut_type == 1:
            # Vertical cut
            L = lp_pos[i]
            x = lp_pos[cut_pos_idx]
            rem = L - x
            rem_floored = l_floor[rem] if rem <= panel_length else 0
            left_items = reconstruct(cut_pos_idx, j)
            if rem_floored > 0 and rem_floored in l_idx:
                right_items = reconstruct(l_idx[rem_floored], j)
            else:
                right_items = []
            return left_items + right_items
        elif cut_type == 2:
            # Horizontal cut
            W = wp_pos[j]
            y = wp_pos[cut_pos_idx]
            rem = W - y
            rem_floored = w_floor[rem] if rem <= panel_width else 0
            top_items = reconstruct(i, cut_pos_idx)
            if rem_floored > 0 and rem_floored in w_idx:
                bottom_items = reconstruct(i, w_idx[rem_floored])
            else:
                bottom_items = []
            return top_items + bottom_items
    
    # Find the panel in our patterns
    panel_l_idx = l_idx.get(panel_length)
    panel_w_idx = w_idx.get(panel_width)
    
    if panel_l_idx is None:
        # Find closest
        best_l = 0
        for x in lp_pos:
            if x <= panel_length:
                best_l = x
        panel_l_idx = l_idx.get(best_l)
    
    if panel_w_idx is None:
        best_w = 0
        for y in wp_pos:
            if y <= panel_width:
                best_w = y
        panel_w_idx = w_idx.get(best_w)
    
    if panel_l_idx is not None and panel_w_idx is not None:
        # Increase recursion limit for deep trees
        sys.setrecursionlimit(100000)
        raw_items = reconstruct(panel_l_idx, panel_w_idx)
    else:
        raw_items = []
    
    # Count items and enforce copy limits
    item_counts = {}
    for pid, l, w in raw_items:
        if pid not in item_counts:
            item_counts[pid] = {'count': 0, 'length': l, 'width': w}
        item_counts[pid]['count'] += 1
    
    # Build item lookup
    item_lookup = {item['id']: item for item in item_types}
    
    # Enforce copy limits
    total_profit = 0
    items_selected = []
    for pid, info in item_counts.items():
        item = item_lookup[pid]
        actual_copies = min(info['count'], item['copies'])
        if actual_copies > 0:
            total_profit += actual_copies * item['profit']
            items_selected.append({
                'item_id': pid,
                'plate_dims': [info['length'], info['width']],
                'copies': actual_copies
            })
    
    if logger:
        logger.log(total_profit)
    
    # Try also: greedy by profit density (sometimes better when copy limits are tight)
    greedy_items = []
    greedy_profit = 0
    remaining_area = panel_length * panel_width
    
    item_list = []
    for item in item_types:
        area = item['length'] * item['width']
        if area <= panel_length * panel_width:
            density = item['profit'] / area
            item_list.append((density, item))
    item_list.sort(reverse=True, key=lambda x: (x[0], x[1]['profit']))
    
    used_copies = {}
    for _, item in item_list:
        if remaining_area <= 0:
            break
        area = item['length'] * item['width']
        # Check if item fits in panel (either orientation)
        fits = False
        chosen_l, chosen_w = item['length'], item['width']
        if (item['length'] <= panel_length and item['width'] <= panel_width):
            fits = True
        elif (item['width'] <= panel_length and item['length'] <= panel_width):
            fits = True
            chosen_l, chosen_w = item['width'], item['length']
        
        if not fits:
            continue
        
        max_by_area = remaining_area // area
        max_copies = min(item['copies'], max_by_area)
        if max_copies > 0:
            used_copies[item['id']] = max_copies
            greedy_profit += max_copies * item['profit']
            remaining_area -= max_copies * area
            greedy_items.append({
                'item_id': item['id'],
                'plate_dims': [chosen_l, chosen_w],
                'copies': max_copies
            })
    
    if greedy_profit > total_profit:
        total_profit = greedy_profit
        items_selected = greedy_items
        if logger:
            logger.log(total_profit)
    
    # Also try: for each single item type, how many can we fit?
    for item in item_types:
        for l, w in [(item['length'], item['width']), (item['width'], item['length'])]:
            if l <= panel_length and w <= panel_width:
                # How many can fit with guillotine cuts?
                # Simple: (panel_length // l) * (panel_width // w)
                nx = panel_length // l
                ny = panel_width // w
                copies = min(nx * ny, item['copies'])
                profit = copies * item['profit']
                if profit > total_profit:
                    total_profit = profit
                    items_selected = [{
                        'item_id': item['id'],
                        'plate_dims': [l, w],
                        'copies': copies
                    }]
                    if logger:
                        logger.log(total_profit)
    
    # Try pairwise combinations
    if time.time() - start_time < time_limit * 0.9 and len(item_types) <= 50:
        for i_idx, item1 in enumerate(item_types):
            if time.time() - start_time > time_limit * 0.9:
                break
            for j_idx, item2 in enumerate(item_types):
                if j_idx <= i_idx:
                    continue
                # Try horizontal then vertical split
                for l1, w1 in [(item1['length'], item1['width']), (item1['width'], item1['length'])]:
                    for l2, w2 in [(item2['length'], item2['width']), (item2['width'], item2['length'])]:
                        if l1 <= panel_length and w1 <= panel_width and l2 <= panel_length and w2 <= panel_width:
                            # Horizontal split: top strip of height w1, bottom strip of height panel_width - w1
                            if w1 + w2 <= panel_width:
                                n1 = min(item1['copies'], (panel_length // l1) * (panel_width // w1))
                                # Give item1 a horizontal strip of width w1
                                strip1_nx = panel_length // l1
                                strip1_ny = 1  # one row in the strip
                                # How many strips of w1 can we fit?
                                strips1 = panel_width // w1
                                remaining_w = panel_width - strips1 * w1
                                
                                c1 = min(item1['copies'], strip1_nx * strips1)
                                # Remaining width for item2
                                if remaining_w >= w2:
                                    c2 = min(item2['copies'], (panel_length // l2) * (remaining_w // w2))
                                else:
                                    c2 = 0
                                
                                p = c1 * item1['profit'] + c2 * item2['profit']
                                if p > total_profit:
                                    total_profit = p
                                    items_selected = []
                                    if c1 > 0:
                                        items_selected.append({
                                            'item_id': item1['id'],
                                            'plate_dims': [l1, w1],
                                            'copies': c1
                                        })
                                    if c2 > 0:
                                        items_selected.append({
                                            'item_id': item2['id'],
                                            'plate_dims': [l2, w2],
                                            'copies': c2
                                        })
                                    if logger:
                                        logger.log(total_profit)
                            
                            # Vertical split
                            if l1 + l2 <= panel_length:
                                strips1 = panel_length // l1
                                remaining_l = panel_length - strips1 * l1
                                c1 = min(item1['copies'], strips1 * (panel_width // w1))
                                if remaining_l >= l2:
                                    c2 = min(item2['copies'], (remaining_l // l2) * (panel_width // w2))
                                else:
                                    c2 = 0
                                
                                p = c1 * item1['profit'] + c2 * item2['profit']
                                if p > total_profit:
                                    total_profit = p
                                    items_selected = []
                                    if c1 > 0:
                                        items_selected.append({
                                            'item_id': item1['id'],
                                            'plate_dims': [l1, w1],
                                            'copies': c1
                                        })
                                    if c2 > 0:
                                        items_selected.append({
                                            'item_id': item2['id'],
                                            'plate_dims': [l2, w2],
                                            'copies': c2
                                        })
                                    if logger:
                                        logger.log(total_profit)
    
    solution = {
        'objective_value': total_profit,
        'items_selected': items_selected
    }
    
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)


if __name__ == '__main__':
    main()
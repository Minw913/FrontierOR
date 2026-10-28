import argparse
import json
import math
import time
import random
import itertools
from copy import deepcopy

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
    vehicle = instance['vehicle']
    depot = instance['depot']
    customers = instance['customers']
    dist_matrix = instance['distance_matrix']

    n = params['n_customers']
    H = vehicle['H']
    W = vehicle['W']
    Q = vehicle['Q']
    K = vehicle['K']
    recourse_cost = params['recourse_cost_cf']

    # Precompute customer expected area and expected weight
    cust_exp_area = {}
    cust_exp_weight = {}
    cust_items = {}

    for c in customers:
        cid = c['id']
        total_exp_area = 0.0
        total_exp_weight = 0.0
        items_list = []
        for item in c['items']:
            item_exp_area = 0.0
            item_exp_weight = 0.0
            for r in item['realizations']:
                prob = r['probability']
                item_exp_area += prob * r['height'] * r['width']
                item_exp_weight += prob * r['weight']
            total_exp_area += item_exp_area
            total_exp_weight += item_exp_weight
            items_list.append(item)
        cust_exp_area[cid] = total_exp_area
        cust_exp_weight[cid] = total_exp_weight
        cust_items[cid] = items_list

    customer_ids = [c['id'] for c in customers]
    cust_is_stochastic = {c['id']: c['is_stochastic'] for c in customers}

    def route_travel_cost(route):
        """Compute travel cost for a route (list of customer ids)."""
        if not route:
            return 0
        depot_id = depot['id']
        cost = dist_matrix[depot_id][route[0]]
        for i in range(len(route) - 1):
            cost += dist_matrix[route[i]][route[i+1]]
        cost += dist_matrix[route[-1]][depot_id]
        return cost

    def get_scenarios_for_route(route):
        """Generate all scenarios for items on a route.
        Each scenario is a list of (customer_id, [(h, w, weight), ...]) tuples.
        Returns list of (probability, scenario) pairs.
        """
        # Collect all stochastic items on this route
        stochastic_items_info = []  # (customer_id, item_index_in_customer, item)
        for cid in route:
            for idx, item in enumerate(cust_items[cid]):
                if item['is_stochastic']:
                    stochastic_items_info.append((cid, idx, item))

        if not stochastic_items_info:
            # All deterministic - single scenario
            scenario = {}
            for cid in route:
                items_dims = []
                for item in cust_items[cid]:
                    r = item['realizations'][0]
                    items_dims.append((r['height'], r['width'], r['weight']))
                scenario[cid] = items_dims
            return [(1.0, scenario)]

        # For efficiency, limit scenario enumeration
        # Each stochastic item has num_realizations possibilities
        realization_lists = []
        for cid, idx, item in stochastic_items_info:
            realization_lists.append(item['realizations'])

        total_scenarios = 1
        for rl in realization_lists:
            total_scenarios *= len(rl)

        # If too many scenarios, sample
        MAX_SCENARIOS = 200
        
        if total_scenarios <= MAX_SCENARIOS:
            scenarios = []
            for combo in itertools.product(*realization_lists):
                prob = 1.0
                realization_map = {}
                for i, (cid, idx, item) in enumerate(stochastic_items_info):
                    r = combo[i]
                    prob *= r['probability']
                    if cid not in realization_map:
                        realization_map[cid] = {}
                    realization_map[cid][idx] = (r['height'], r['width'], r['weight'])

                scenario = {}
                for cid in route:
                    items_dims = []
                    for idx, item in enumerate(cust_items[cid]):
                        if item['is_stochastic'] and cid in realization_map and idx in realization_map[cid]:
                            items_dims.append(realization_map[cid][idx])
                        else:
                            r = item['realizations'][0]
                            items_dims.append((r['height'], r['width'], r['weight']))
                    scenario[cid] = items_dims
                scenarios.append((prob, scenario))
            return scenarios
        else:
            # Sample scenarios
            scenarios = []
            total_prob_sampled = 0.0
            for _ in range(MAX_SCENARIOS):
                prob = 1.0
                realization_map = {}
                for i, (cid, idx, item) in enumerate(stochastic_items_info):
                    realizations = item['realizations']
                    probs = [r['probability'] for r in realizations]
                    chosen_idx = random.choices(range(len(realizations)), weights=probs, k=1)[0]
                    r = realizations[chosen_idx]
                    prob *= r['probability']
                    if cid not in realization_map:
                        realization_map[cid] = {}
                    realization_map[cid][idx] = (r['height'], r['width'], r['weight'])

                scenario = {}
                for cid in route:
                    items_dims = []
                    for idx, item in enumerate(cust_items[cid]):
                        if item['is_stochastic'] and cid in realization_map and idx in realization_map[cid]:
                            items_dims.append(realization_map[cid][idx])
                        else:
                            r = item['realizations'][0]
                            items_dims.append((r['height'], r['width'], r['weight']))
                    scenario[cid] = items_dims
                scenarios.append((1.0 / MAX_SCENARIOS, scenario))
            return scenarios

    def can_pack_2d_items(items_list, H, W):
        """Check if items (list of (h, w)) can be packed in H x W bin.
        Uses a simple bottom-left heuristic with different orderings.
        items_list: list of (height, width) tuples.
        """
        if not items_list:
            return True

        total_area = sum(h * w for h, w in items_list)
        if total_area > H * W:
            return False

        # Try bottom-left-fill with decreasing area ordering
        def try_pack(items):
            # Each placed item: (x, y, w, h) where (x,y) is bottom-left
            placements = []
            # Available spaces: list of (x, y, max_w, max_h) - simplified
            # Use skyline approach
            # skyline[x] = y means the height at position x
            skyline = [0] * W

            for ih, iw in items:
                placed = False
                best_y = H + 1
                best_x = -1

                # Try each x position
                for x in range(W - iw + 1):
                    # Find max height in [x, x+iw)
                    max_y = max(skyline[x:x+iw])
                    if max_y + ih <= H and max_y < best_y:
                        best_y = max_y
                        best_x = x
                        placed = True

                if not placed:
                    return False

                # Place item
                for xx in range(best_x, best_x + iw):
                    skyline[xx] = best_y + ih
                placements.append((best_x, best_y, iw, ih))

            return True

        # Try different orderings
        items_by_area = sorted(items_list, key=lambda x: x[0]*x[1], reverse=True)
        if try_pack(items_by_area):
            return True

        items_by_height = sorted(items_list, key=lambda x: x[0], reverse=True)
        if try_pack(items_by_height):
            return True

        items_by_width = sorted(items_list, key=lambda x: x[1], reverse=True)
        if try_pack(items_by_width):
            return True

        return False

    def pack_with_unloading_constraint(route, scenario):
        """
        Try to pack items for a route with the LIFO unloading constraint.
        Customers are visited in order route[0], route[1], ..., route[-1].
        Items for route[-1] (last customer) must be at the rear (top),
        items for route[0] (first customer) can be at the bottom.
        
        Actually, the constraint is: at each stop, current customer's items 
        must be unloadable from the rear without moving items of yet-unserved customers.
        
        This means items must be packed in reverse order of delivery:
        - Items for last customer are packed first (closest to rear/top)
        - Items for first customer are packed last (deepest)
        
        Wait, actually items for the first customer delivered should be most accessible.
        Since we pull from the rear (top), the first customer's items should be at the top.
        
        Let me reconsider: items are pulled from the top. At the first stop, we need to 
        access customer route[0]'s items. So those must be on top (packed last in a 
        bottom-up packing). At the second stop, route[1]'s items, etc.
        
        So packing order (bottom to top): route[-1], route[-2], ..., route[0]
        Each customer's items form a "layer" and must be above the items of later customers.
        
        Simplified approach: pack as sequential layers from bottom to top.
        """
        # We need to pack items layer by layer
        # Bottom layer: last customer's items
        # Top layer: first customer's items
        
        # For the LIFO constraint, a simple approach:
        # Divide the vehicle height into strips, one per customer
        # Each customer's items must fit within their allocated strip width W and allocated height
        
        # More sophisticated: use a 2D packing where items of customer i 
        # are all above items of customer j for j visited after i
        
        # Simple feasibility check: sequential strip packing
        all_items = []
        for cid in reversed(route):  # bottom to top
            items = scenario[cid]
            all_items.append((cid, items))
        
        current_height = 0
        for cid, items in all_items:
            if not items:
                continue
            # Pack these items in a strip of width W, find minimum height needed
            min_h = pack_strip(items, W)
            if min_h is None or current_height + min_h > H:
                return False
            current_height += min_h
        
        return True

    def pack_strip(items, W):
        """Pack items (list of (h, w, weight)) into a strip of width W.
        Returns minimum height needed, or None if impossible.
        Uses bottom-left skyline heuristic.
        """
        if not items:
            return 0
        
        dims = [(h, w) for h, w, _ in items]
        
        # Check if any item exceeds W
        for h, w in dims:
            if w > W:
                return None
        
        # Skyline packing
        skyline = [0] * W
        
        # Sort by height descending for better packing
        sorted_dims = sorted(dims, key=lambda x: x[0], reverse=True)
        
        for ih, iw in sorted_dims:
            best_y = float('inf')
            best_x = -1
            
            for x in range(W - iw + 1):
                max_y = max(skyline[x:x+iw])
                if max_y < best_y:
                    best_y = max_y
                    best_x = x
            
            if best_x == -1:
                return None
            
            for xx in range(best_x, best_x + iw):
                skyline[xx] = best_y + ih
        
        return max(skyline)

    def pack_strip_best(items, W):
        """Try multiple orderings for strip packing."""
        if not items:
            return 0
        
        dims = [(h, w) for h, w, _ in items]
        
        for h, w in dims:
            if w > W:
                return None
        
        best_height = None
        
        orderings = [
            sorted(dims, key=lambda x: x[0], reverse=True),
            sorted(dims, key=lambda x: x[1], reverse=True),
            sorted(dims, key=lambda x: x[0]*x[1], reverse=True),
            sorted(dims, key=lambda x: max(x[0], x[1]), reverse=True),
        ]
        
        for ordered in orderings:
            skyline = [0] * W
            feasible = True
            
            for ih, iw in ordered:
                best_y = float('inf')
                best_x = -1
                
                for x in range(W - iw + 1):
                    max_y = max(skyline[x:x+iw])
                    if max_y < best_y:
                        best_y = max_y
                        best_x = x
                
                if best_x == -1:
                    feasible = False
                    break
                
                for xx in range(best_x, best_x + iw):
                    skyline[xx] = best_y + ih
            
            if feasible:
                h = max(skyline)
                if best_height is None or h < best_height:
                    best_height = h
        
        return best_height

    def compute_route_penalty(route):
        """Compute expected penalty cost for a route."""
        if not route:
            return 0.0
        
        # Check if route has any stochastic customers
        has_stochastic = any(cust_is_stochastic[cid] for cid in route)
        
        if not has_stochastic:
            # Deterministic route - check if all items can be packed
            scenario = {}
            total_weight = 0
            for cid in route:
                items_dims = []
                for item in cust_items[cid]:
                    r = item['realizations'][0]
                    items_dims.append((r['height'], r['width'], r['weight']))
                    total_weight += r['weight']
                scenario[cid] = items_dims
            
            if total_weight > Q:
                # Weight exceeded - need to determine how many unserved
                # For simplicity, assume greedy: serve as many as possible
                return compute_penalty_for_scenario(route, scenario)
            
            if try_pack_with_unloading(route, scenario):
                return 0.0
            else:
                return compute_penalty_for_scenario(route, scenario)
        
        # Stochastic route
        scenarios = get_scenarios_for_route(route)
        expected_penalty = 0.0
        for prob, scenario in scenarios:
            penalty = compute_penalty_for_scenario(route, scenario)
            expected_penalty += prob * penalty
        
        return expected_penalty

    def compute_penalty_for_scenario(route, scenario):
        """For a given scenario, compute penalty (number of unserved * recourse_cost).
        Try to pack all items; if fails, try removing customers one by one.
        """
        # First check weight
        total_weight = sum(sum(w for _, _, w in scenario[cid]) for cid in route)
        
        if total_weight <= Q and try_pack_with_unloading(route, scenario):
            return 0.0
        
        # Need to find minimum unserved customers
        # Greedy: try removing customers that free up the most space
        # For efficiency, try a simple approach: binary search on how many to remove
        
        best_unserved = len(route)
        
        # Try removing customers one at a time (greedy by largest expected area)
        remaining = list(route)
        unserved = 0
        
        # Sort customers by area (largest first for removal)
        cust_area_in_scenario = {cid: sum(h*w for h, w, _ in scenario[cid]) for cid in route}
        cust_weight_in_scenario = {cid: sum(wt for _, _, wt in scenario[cid]) for cid in route}
        
        # Try removing largest area customers first
        removal_order = sorted(route, key=lambda c: cust_area_in_scenario[c], reverse=True)
        
        removed = set()
        for cid_to_remove in removal_order:
            current_route = [c for c in route if c not in removed]
            total_w = sum(cust_weight_in_scenario[c] for c in current_route)
            
            if total_w <= Q and try_pack_with_unloading(current_route, scenario):
                return len(removed) * recourse_cost
            
            removed.add(cid_to_remove)
        
        return len(route) * recourse_cost

    def try_pack_with_unloading(route, scenario):
        """Try to pack items for route with LIFO unloading constraint."""
        if not route:
            return True
        
        # Check total weight
        total_weight = sum(sum(w for _, _, w in scenario[cid]) for cid in route)
        if total_weight > Q:
            return False
        
        # Check total area
        total_area = sum(sum(h*w for h, w, _ in scenario[cid]) for cid in route)
        if total_area > H * W:
            return False
        
        # Sequential strip packing (LIFO): pack from bottom to top
        # Bottom: last customer, Top: first customer
        current_height = 0
        for cid in reversed(route):
            items = scenario[cid]
            if not items:
                continue
            min_h = pack_strip_best(items, W)
            if min_h is None:
                return False
            current_height += min_h
        
        if current_height <= H:
            return True
        
        # Try the other dimension orientation: pack strips along width
        current_width = 0
        for cid in reversed(route):
            items = scenario[cid]
            if not items:
                continue
            # Pack in strip of height H
            min_w = pack_strip_best([(w, h, wt) for h, w, wt in items], H)
            if min_w is None:
                return False
            current_width += min_w
        
        if current_width <= W:
            return True
        
        # Try more sophisticated 2D packing with layers
        # For now, also try a combined approach where we use the full 2D space
        # but maintain the ordering constraint
        return try_pack_2d_layered(route, scenario)

    def try_pack_2d_layered(route, scenario):
        """More sophisticated 2D packing with LIFO layers."""
        if not route:
            return True
        
        # Use a skyline-based approach where each customer's items
        # are placed in a band, but we allow some interleaving in the width dimension
        # as long as the height ordering is maintained
        
        # For simplicity, try all permutations of items within each customer's set
        # and use a global skyline
        
        # Actually, let's try a different strip orientation
        # The "rear" could be along either dimension
        
        # Approach: treat height as the loading direction
        # Items for customer i must have their bottom y-coordinate >= 
        # top y-coordinate of all items for customer j where j is visited after i
        
        # This is equivalent to: allocate height bands per customer in reverse visit order
        
        # We already tried simple sequential. Let's try allowing reordering within strips
        # but that's what pack_strip_best does.
        
        # For small routes, try exact packing
        if len(route) <= 3:
            all_items = []
            for cid in route:
                for h, w, wt in scenario[cid]:
                    all_items.append((h, w, cid))
            
            # Check if items fit in H x W ignoring ordering
            total_area = sum(h*w for h, w, _ in all_items)
            if total_area > H * W:
                return False
            
            # Could try more orderings but for now return False
        
        return False

    def compute_route_cost(route):
        """Compute total cost (travel + penalty) for a route."""
        travel = route_travel_cost(route)
        penalty = compute_route_penalty(route)
        return travel + penalty

    def compute_solution_cost(routes):
        """Compute total cost of a solution."""
        total = 0.0
        for route in routes:
            total += compute_route_cost(route)
        return total

    def compute_solution_travel_cost(routes):
        total = 0
        for route in routes:
            total += route_travel_cost(route)
        return total

    # ---- Nearest neighbor construction heuristic ----
    def nearest_neighbor_construction():
        """Build initial routes using nearest neighbor, respecting capacity."""
        depot_id = depot['id']
        unvisited = set(customer_ids)
        routes = [[] for _ in range(K)]

        # Compute capacity per vehicle
        area_cap = H * W
        weight_cap = Q

        for k in range(K):
            if not unvisited:
                break
            current = depot_id
            route_area = 0.0
            route_weight = 0.0

            while unvisited:
                # Find nearest unvisited customer that fits
                best_cust = None
                best_dist = float('inf')
                for cid in unvisited:
                    if route_area + cust_exp_area[cid] <= area_cap and \
                       route_weight + cust_exp_weight[cid] <= weight_cap:
                        d = dist_matrix[current][cid]
                        if d < best_dist:
                            best_dist = d
                            best_cust = cid

                if best_cust is None:
                    break

                routes[k].append(best_cust)
                route_area += cust_exp_area[best_cust]
                route_weight += cust_exp_weight[best_cust]
                unvisited.remove(best_cust)
                current = best_cust

        # Assign remaining unvisited to routes with space
        remaining = list(unvisited)
        for cid in remaining:
            best_route = -1
            best_cost_increase = float('inf')
            for k in range(K):
                ra = sum(cust_exp_area[c] for c in routes[k])
                rw = sum(cust_exp_weight[c] for c in routes[k])
                if ra + cust_exp_area[cid] <= area_cap and \
                   rw + cust_exp_weight[cid] <= weight_cap:
                    # Find best insertion position
                    cost_inc = insertion_cost(routes[k], cid)
                    if cost_inc < best_cost_increase:
                        best_cost_increase = cost_inc
                        best_route = k

            if best_route == -1:
                # Force into least loaded route
                min_load = float('inf')
                for k in range(K):
                    load = sum(cust_exp_area[c] for c in routes[k])
                    if load < min_load:
                        min_load = load
                        best_route = k

            # Insert at best position
            best_pos = find_best_insertion_pos(routes[best_route], cid)
            routes[best_route].insert(best_pos, cid)
            unvisited.discard(cid)

        return routes

    def insertion_cost(route, cid):
        """Find minimum insertion cost for adding cid to route."""
        depot_id = depot['id']
        if not route:
            return dist_matrix[depot_id][cid] + dist_matrix[cid][depot_id]

        best = float('inf')
        # Insert at beginning
        cost = dist_matrix[depot_id][cid] + dist_matrix[cid][route[0]] - dist_matrix[depot_id][route[0]]
        best = min(best, cost)
        # Insert in middle
        for i in range(len(route) - 1):
            cost = dist_matrix[route[i]][cid] + dist_matrix[cid][route[i+1]] - dist_matrix[route[i]][route[i+1]]
            best = min(best, cost)
        # Insert at end
        cost = dist_matrix[route[-1]][cid] + dist_matrix[cid][depot_id] - dist_matrix[route[-1]][depot_id]
        best = min(best, cost)

        return best

    def find_best_insertion_pos(route, cid):
        """Find best position to insert cid in route."""
        depot_id = depot['id']
        if not route:
            return 0

        best_cost = float('inf')
        best_pos = 0

        for pos in range(len(route) + 1):
            if pos == 0:
                prev = depot_id
            else:
                prev = route[pos - 1]
            if pos == len(route):
                nxt = depot_id
            else:
                nxt = route[pos]
            cost = dist_matrix[prev][cid] + dist_matrix[cid][nxt] - dist_matrix[prev][nxt]
            if cost < best_cost:
                best_cost = cost
                best_pos = pos

        return best_pos

    # ---- Savings-based construction ----
    def savings_construction():
        """Clarke-Wright savings algorithm."""
        depot_id = depot['id']
        # Start with each customer in its own route
        routes = {cid: [cid] for cid in customer_ids}
        route_of = {cid: cid for cid in customer_ids}  # maps customer to route key
        route_area = {cid: cust_exp_area[cid] for cid in customer_ids}
        route_weight = {cid: cust_exp_weight[cid] for cid in customer_ids}

        # Compute savings
        savings = []
        for i in customer_ids:
            for j in customer_ids:
                if i < j:
                    s = dist_matrix[depot_id][i] + dist_matrix[depot_id][j] - dist_matrix[i][j]
                    savings.append((s, i, j))

        savings.sort(reverse=True)

        area_cap = H * W
        weight_cap = Q

        for s, i, j in savings:
            ri = route_of[i]
            rj = route_of[j]
            if ri == rj:
                continue
            if ri not in routes or rj not in routes:
                continue

            # Check capacity
            new_area = route_area[ri] + route_area[rj]
            new_weight = route_weight[ri] + route_weight[rj]
            if new_area > area_cap or new_weight > weight_cap:
                continue

            # Check if i is at the end and j is at the start (or vice versa)
            route_i = routes[ri]
            route_j = routes[rj]

            merged = None
            if route_i[-1] == i and route_j[0] == j:
                merged = route_i + route_j
            elif route_j[-1] == j and route_i[0] == i:
                merged = route_j + route_i
            elif route_i[0] == i and route_j[0] == j:
                merged = list(reversed(route_i)) + route_j
            elif route_i[-1] == i and route_j[-1] == j:
                merged = route_i + list(reversed(route_j))

            if merged is None:
                continue

            # Limit route count: we need at least K routes
            num_routes = len(routes)
            if num_routes <= K:
                continue

            # Merge
            new_key = ri
            routes[new_key] = merged
            route_area[new_key] = new_area
            route_weight[new_key] = new_weight

            for c in merged:
                route_of[c] = new_key

            if rj in routes:
                del routes[rj]
                del route_area[rj]
                del route_weight[rj]

        # Convert to list of routes
        result = list(routes.values())

        # If we have more than K routes, merge the smallest ones
        while len(result) > K:
            result.sort(key=len)
            # Merge two smallest
            r1 = result.pop(0)
            r2 = result.pop(0)
            result.append(r1 + r2)

        # If we have fewer than K routes, split the largest ones
        while len(result) < K:
            result.sort(key=len, reverse=True)
            largest = result.pop(0)
            if len(largest) > 1:
                mid = len(largest) // 2
                result.append(largest[:mid])
                result.append(largest[mid:])
            else:
                result.append(largest)
                result.append([])

        return result

    # ---- Local search improvements ----
    def two_opt_route(route):
        """Apply 2-opt within a single route."""
        if len(route) <= 2:
            return route
        depot_id = depot['id']
        improved = True
        route = list(route)
        while improved:
            improved = False
            for i in range(len(route) - 1):
                for j in range(i + 1, len(route)):
                    # Current edges: (prev_i, route[i]), (route[j], next_j)
                    prev_i = depot_id if i == 0 else route[i-1]
                    next_j = depot_id if j == len(route) - 1 else route[j+1]
                    
                    old_cost = dist_matrix[prev_i][route[i]] + dist_matrix[route[j]][next_j]
                    new_cost = dist_matrix[prev_i][route[j]] + dist_matrix[route[i]][next_j]
                    
                    if new_cost < old_cost:
                        route[i:j+1] = reversed(route[i:j+1])
                        improved = True
        return route

    def or_opt_route(route):
        """Apply or-opt (move segments of 1-2-3 customers) within a route."""
        if len(route) <= 2:
            return route
        depot_id = depot['id']
        improved = True
        route = list(route)
        while improved:
            improved = False
            for seg_len in [1, 2, 3]:
                for i in range(len(route) - seg_len + 1):
                    segment = route[i:i+seg_len]
                    remaining = route[:i] + route[i+seg_len:]
                    
                    # Cost of removing segment
                    prev_i = depot_id if i == 0 else route[i-1]
                    after_seg = depot_id if i + seg_len >= len(route) else route[i+seg_len]
                    
                    removal_savings = (dist_matrix[prev_i][route[i]] + 
                                      dist_matrix[route[i+seg_len-1]][after_seg] -
                                      dist_matrix[prev_i][after_seg])
                    
                    # Try reinserting at each position in remaining
                    for j in range(len(remaining) + 1):
                        if j == i:
                            continue
                        prev_j = depot_id if j == 0 else remaining[j-1]
                        next_j = depot_id if j >= len(remaining) else remaining[j]
                        
                        insertion_cost_val = (dist_matrix[prev_j][segment[0]] + 
                                            dist_matrix[segment[-1]][next_j] -
                                            dist_matrix[prev_j][next_j])
                        
                        if insertion_cost_val < removal_savings - 1e-10:
                            new_route = remaining[:j] + segment + remaining[j:]
                            route = new_route
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        return route

    def relocate_between_routes(routes):
        """Try moving a customer from one route to another."""
        depot_id = depot['id']
        area_cap = H * W
        weight_cap = Q
        improved = True
        
        while improved:
            improved = False
            for k1 in range(len(routes)):
                for k2 in range(len(routes)):
                    if k1 == k2:
                        continue
                    for i in range(len(routes[k1])):
                        cid = routes[k1][i]
                        
                        # Check if cid fits in route k2
                        new_area = sum(cust_exp_area[c] for c in routes[k2]) + cust_exp_area[cid]
                        new_weight = sum(cust_exp_weight[c] for c in routes[k2]) + cust_exp_weight[cid]
                        
                        if new_area > area_cap or new_weight > weight_cap:
                            continue
                        
                        # Cost of removing cid from k1
                        prev = depot_id if i == 0 else routes[k1][i-1]
                        nxt = depot_id if i == len(routes[k1]) - 1 else routes[k1][i+1]
                        removal_saving = dist_matrix[prev][cid] + dist_matrix[cid][nxt] - dist_matrix[prev][nxt]
                        
                        # Best insertion cost in k2
                        best_ins = float('inf')
                        best_pos = 0
                        for j in range(len(routes[k2]) + 1):
                            p = depot_id if j == 0 else routes[k2][j-1]
                            n = depot_id if j >= len(routes[k2]) else routes[k2][j]
                            ins = dist_matrix[p][cid] + dist_matrix[cid][n] - dist_matrix[p][n]
                            if ins < best_ins:
                                best_ins = ins
                                best_pos = j
                        
                        if best_ins < removal_saving - 1e-10:
                            routes[k1].pop(i)
                            routes[k2].insert(best_pos, cid)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        return routes

    def swap_between_routes(routes):
        """Try swapping a customer between two routes."""
        depot_id = depot['id']
        area_cap = H * W
        weight_cap = Q
        improved = True
        
        while improved:
            improved = False
            for k1 in range(len(routes)):
                for k2 in range(k1 + 1, len(routes)):
                    for i in range(len(routes[k1])):
                        for j in range(len(routes[k2])):
                            c1 = routes[k1][i]
                            c2 = routes[k2][j]
                            
                            # Check capacity after swap
                            area1 = sum(cust_exp_area[c] for c in routes[k1]) - cust_exp_area[c1] + cust_exp_area[c2]
                            area2 = sum(cust_exp_area[c] for c in routes[k2]) - cust_exp_area[c2] + cust_exp_area[c1]
                            weight1 = sum(cust_exp_weight[c] for c in routes[k1]) - cust_exp_weight[c1] + cust_exp_weight[c2]
                            weight2 = sum(cust_exp_weight[c] for c in routes[k2]) - cust_exp_weight[c2] + cust_exp_weight[c1]
                            
                            if area1 > area_cap or area2 > area_cap or weight1 > weight_cap or weight2 > weight_cap:
                                continue
                            
                            # Cost change for route k1
                            prev1 = depot_id if i == 0 else routes[k1][i-1]
                            nxt1 = depot_id if i == len(routes[k1]) - 1 else routes[k1][i+1]
                            old1 = dist_matrix[prev1][c1] + dist_matrix[c1][nxt1]
                            new1 = dist_matrix[prev1][c2] + dist_matrix[c2][nxt1]
                            
                            prev2 = depot_id if j == 0 else routes[k2][j-1]
                            nxt2 = depot_id if j == len(routes[k2]) - 1 else routes[k2][j+1]
                            old2 = dist_matrix[prev2][c2] + dist_matrix[c2][nxt2]
                            new2 = dist_matrix[prev2][c1] + dist_matrix[c1][nxt2]
                            
                            if (new1 + new2) < (old1 + old2) - 1e-10:
                                routes[k1][i] = c2
                                routes[k2][j] = c1
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        return routes

    def local_search(routes, time_limit_ls):
        """Apply local search to improve routes."""
        # Intra-route optimization
        for k in range(len(routes)):
            if time.time() - start_time > time_limit_ls:
                break
            routes[k] = two_opt_route(routes[k])
            routes[k] = or_opt_route(routes[k])
        
        # Inter-route optimization
        if time.time() - start_time < time_limit_ls:
            routes = relocate_between_routes(routes)
        if time.time() - start_time < time_limit_ls:
            routes = swap_between_routes(routes)
        
        # Another round of intra-route
        for k in range(len(routes)):
            if time.time() - start_time > time_limit_ls:
                break
            routes[k] = two_opt_route(routes[k])
        
        return routes

    def perturb_solution(routes):
        """Perturbation for iterated local search."""
        routes = deepcopy(routes)
        
        # Random double-bridge or random relocations
        perturbation_type = random.random()
        
        if perturbation_type < 0.5 and len(routes) > 1:
            # Random relocate: move a random customer to a random route
            non_empty = [k for k in range(len(routes)) if len(routes[k]) > 0]
            if len(non_empty) >= 1:
                k1 = random.choice(non_empty)
                k2 = random.randint(0, len(routes) - 1)
                if k1 != k2 and len(routes[k1]) > 0:
                    i = random.randint(0, len(routes[k1]) - 1)
                    cid = routes[k1].pop(i)
                    pos = random.randint(0, len(routes[k2]))
                    routes[k2].insert(pos, cid)
        else:
            # Random swap within a route
            non_empty = [k for k in range(len(routes)) if len(routes[k]) >= 4]
            if non_empty:
                k = random.choice(non_empty)
                route = routes[k]
                n_r = len(route)
                # Double bridge
                if n_r >= 4:
                    cuts = sorted(random.sample(range(1, n_r), 3))
                    seg1 = route[:cuts[0]]
                    seg2 = route[cuts[0]:cuts[1]]
                    seg3 = route[cuts[1]:cuts[2]]
                    seg4 = route[cuts[2]:]
                    routes[k] = seg1 + seg3 + seg2 + seg4
            else:
                # Random swap between routes
                if len(routes) >= 2:
                    non_empty = [k for k in range(len(routes)) if len(routes[k]) > 0]
                    if len(non_empty) >= 2:
                        k1, k2 = random.sample(non_empty, 2)
                        i1 = random.randint(0, len(routes[k1]) - 1)
                        i2 = random.randint(0, len(routes[k2]) - 1)
                        routes[k1][i1], routes[k2][i2] = routes[k2][i2], routes[k1][i1]
        
        return routes

    # ---- Main solution procedure ----
    
    # Build initial solutions
    routes_nn = nearest_neighbor_construction()
    routes_savings = savings_construction()
    
    # Quick local search on both
    time_for_init = min(start_time + args.time_limit * 0.3, start_time + args.time_limit - 2)
    
    routes_nn = local_search(routes_nn, time_for_init)
    routes_savings = local_search(routes_savings, time_for_init)
    
    # Evaluate (travel cost only for speed, then add penalty for best)
    cost_nn = compute_solution_travel_cost(routes_nn)
    cost_savings = compute_solution_travel_cost(routes_savings)
    
    if cost_nn <= cost_savings:
        best_routes = routes_nn
        best_cost = cost_nn
    else:
        best_routes = routes_savings
        best_cost = cost_savings
    
    # Now compute with penalties
    # For speed, first use travel-only cost for local search, 
    # then evaluate penalties for the final solution
    
    # Compute full cost including penalties
    def compute_full_cost(routes):
        travel = compute_solution_travel_cost(routes)
        penalty = 0.0
        for route in routes:
            if route:
                penalty += compute_route_penalty(route)
        return travel + penalty
    
    # For small instances, compute full cost
    full_cost = compute_full_cost(best_routes)
    best_full_cost = full_cost
    
    if logger:
        logger.log(best_full_cost)
    
    # Iterated local search
    current_routes = deepcopy(best_routes)
    current_cost = best_full_cost
    
    iteration = 0
    while time.time() - start_time < args.time_limit - 2:
        iteration += 1
        
        # Perturb
        new_routes = perturb_solution(current_routes)
        
        # Local search
        time_for_ls = min(time.time() + (args.time_limit - (time.time() - start_time)) * 0.3, 
                         start_time + args.time_limit - 1)
        new_routes = local_search(new_routes, time_for_ls)
        
        # Evaluate
        new_travel = compute_solution_travel_cost(new_routes)
        
        # Only compute full cost if travel cost is competitive
        if new_travel < best_full_cost * 1.1:
            new_cost = compute_full_cost(new_routes)
        else:
            new_cost = new_travel  # Approximate
        
        # Accept if better
        if new_cost < current_cost:
            current_routes = new_routes
            current_cost = new_cost
            
            if new_cost < best_full_cost:
                best_routes = deepcopy(new_routes)
                best_full_cost = new_cost
                if logger:
                    logger.log(best_full_cost)
        else:
            # Accept with some probability (simulated annealing)
            temp = max(0.01, 1.0 - (time.time() - start_time) / args.time_limit)
            delta = new_cost - current_cost
            if delta > 0 and random.random() < math.exp(-delta / (temp * max(1, current_cost * 0.01))):
                current_routes = new_routes
                current_cost = new_cost
        
        if time.time() - start_time > args.time_limit - 2:
            break

    # Final output
    objective_value = best_full_cost
    
    solution = {
        "objective_value": objective_value,
        "routes": best_routes
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == '__main__':
    main()
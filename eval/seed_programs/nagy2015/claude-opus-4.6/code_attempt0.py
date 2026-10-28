import argparse
import json
import math
import time
import random
import copy
from solution_logger import SolutionLogger


def solve(instance_path, solution_path, time_limit, log_path):
    start_time = time.time()
    
    # Initialize logger
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    # Read instance
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    
    n = instance['num_customers']
    C = instance['vehicle_capacity']
    depot = instance['depot']
    customers = instance['customers']
    dist = instance['distance_matrix']
    
    # Customer data (1-indexed)
    delivery = [0] * (n + 1)
    pickup = [0] * (n + 1)
    for c in customers:
        delivery[c['id']] = c['delivery_demand']
        pickup[c['id']] = c['pickup_demand']
    
    # In this problem, each customer has a delivery demand and a pickup demand.
    # These can be served on separate visits (split delivery/pickup).
    # We use an expanded node representation:
    # - Linehaul node i (1..n): serves delivery demand of customer i
    # - Backhaul node n+i (n+1..2n): serves pickup demand of customer i
    # 
    # A route visits some linehaul nodes first, then some backhaul nodes.
    # The vehicle leaves depot with delivery goods, delivers them (linehaul),
    # then picks up goods (backhaul), and returns to depot.
    #
    # Capacity constraint: at any point, load <= C.
    # When leaving depot: load = sum of deliveries on route
    # After delivering to linehaul node i: load decreases by delivery[i]
    # After picking up from backhaul node j: load increases by pickup[j-n]
    #
    # Key insight: linehaul nodes must come before backhaul nodes in a route
    # (this is the standard VRPB structure, but here it's more flexible since
    # delivery and pickup for same customer can be split).
    
    # Actually, re-reading the problem: "At any point during a vehicle's trip 
    # the total cargo on board (delivery goods still to be dropped off plus 
    # pickup goods already collected) must not exceed the vehicle capacity."
    # This means we need to track the load carefully. The optimal ordering
    # would typically be to deliver first, then pick up (linehaul-backhaul).
    # But actually, interleaving could sometimes help.
    
    # For simplicity and efficiency, let's use the standard approach:
    # Each route has a linehaul segment (deliveries) followed by a backhaul segment (pickups).
    # This is a common structure for VRPB problems.
    
    # Distance function for expanded nodes
    def node_dist(a, b):
        # Map expanded node to original customer (0 = depot)
        if a == 0:
            ca = 0
        elif a <= n:
            ca = a
        else:
            ca = a - n
        if b == 0:
            cb = 0
        elif b <= n:
            cb = b
        else:
            cb = b - n
        return dist[ca][cb]
    
    def route_distance(route):
        """Calculate total distance of a route (list of expanded nodes, without depot)."""
        if not route:
            return 0
        d = dist[0][get_customer(route[0])]
        for i in range(len(route) - 1):
            d += dist[get_customer(route[i])][get_customer(route[i+1])]
        d += dist[get_customer(route[-1])][0]
        return d
    
    def get_customer(node):
        if node == 0:
            return 0
        elif node <= n:
            return node
        else:
            return node - n
    
    def get_demand(node):
        """Return (delivery_amount, pickup_amount) for a node."""
        if node == 0:
            return 0, 0
        elif node <= n:
            return delivery[node], 0
        else:
            return 0, pickup[node - n]
    
    def check_route_feasibility(route):
        """Check if route is feasible w.r.t. capacity.
        Returns True if feasible."""
        # Calculate initial load (sum of all deliveries in route)
        initial_load = sum(delivery[node] for node in route if node <= n)
        if initial_load > C:
            return False
        load = initial_load
        for node in route:
            if node <= n:
                load -= delivery[node]
            else:
                load += pickup[node - n]
                if load > C:
                    return False
        return True
    
    def check_route_feasibility_lh_bh(linehaul, backhaul):
        """Check feasibility for a route with linehaul then backhaul structure."""
        total_delivery = sum(delivery[node] for node in linehaul)
        total_pickup = sum(pickup[node - n] for node in backhaul)
        if total_delivery > C:
            return False
        if total_pickup > C:
            return False
        # After all deliveries, load = 0, then pickups accumulate
        # But we need to check intermediate: load starts at total_delivery,
        # decreases as we deliver. Then increases as we pick up.
        # Max load during delivery phase = total_delivery (at start) <= C (checked)
        # Max load during pickup phase = total_pickup (at end) <= C (checked)
        # At transition point: load = 0 + first pickup <= C
        # Actually, if we do all linehaul first then all backhaul, the load is:
        # - Start: total_delivery
        # - After each linehaul delivery: decreases
        # - After all linehaul: 0
        # - After each backhaul pickup: increases
        # - After all backhaul: total_pickup
        # So we just need total_delivery <= C and total_pickup <= C
        return True
    
    # Build initial solution using nearest-neighbor heuristic
    # We'll build routes with linehaul-first, backhaul-second structure
    
    # First, handle nodes with zero demand - they don't need to be visited
    linehaul_nodes = [i for i in range(1, n+1) if delivery[i] > 0]
    backhaul_nodes = [n+i for i in range(1, n+1) if pickup[i] > 0]
    
    # For customers with both delivery=0 and pickup=0, they still need to be served
    # Actually, re-reading: "serve every customer's delivery demand and every customer's pickup demand"
    # If demand is 0, arguably no visit needed. But to be safe, let's include all.
    # Actually the schema says delivery_demand and pickup_demand are quantities, if 0, no service needed.
    # Let's include all customers to be safe.
    linehaul_nodes = list(range(1, n+1))
    backhaul_nodes = [n+i for i in range(1, n+1)]
    
    # Filter out zero-demand nodes
    linehaul_nodes = [i for i in range(1, n+1) if delivery[i] > 0]
    backhaul_nodes = [n+i for i in range(1, n+1) if pickup[i] > 0]
    
    # For customers with delivery=0, we still might need a linehaul visit if the problem requires it.
    # But the problem says "serve every customer's delivery demand" - if demand is 0, it's trivially served.
    # Similarly for pickup. Let's only include non-zero demands.
    
    # However, we must check: does the solution schema require visiting every customer?
    # Looking at the schema: routes contain expanded node IDs for served requests.
    # If delivery_demand = 0, we don't need linehaul node. If pickup_demand = 0, we don't need backhaul node.
    
    def build_initial_solution():
        """Build initial solution using savings algorithm."""
        routes = []
        
        # Start with individual routes for each linehaul and backhaul node
        # Then merge using Clarke-Wright savings
        
        # Approach: Build linehaul routes and backhaul routes separately, then try to merge
        
        # Build linehaul routes using savings algorithm
        lh_routes = build_savings_routes(linehaul_nodes, lambda node: delivery[node])
        
        # Build backhaul routes using savings algorithm
        bh_routes = build_savings_routes(backhaul_nodes, lambda node: pickup[node - n])
        
        # Now try to combine linehaul and backhaul routes
        # Each combined route: linehaul_segment + backhaul_segment
        # Feasibility: sum(deliveries) <= C and sum(pickups) <= C
        
        # For combining, we use a greedy approach: for each linehaul route,
        # find the best backhaul route to append
        combined_routes = combine_lh_bh_routes(lh_routes, bh_routes)
        
        return combined_routes
    
    def build_savings_routes(nodes, demand_func):
        """Build routes using Clarke-Wright savings algorithm."""
        if not nodes:
            return []
        
        # Start with individual routes
        routes = [[node] for node in nodes]
        
        # Calculate savings
        savings = []
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                ni, nj = nodes[i], nodes[j]
                ci, cj = get_customer(ni), get_customer(nj)
                s = dist[0][ci] + dist[0][cj] - dist[ci][cj]
                savings.append((s, ni, nj))
        
        savings.sort(reverse=True)
        
        # Track which route each node belongs to
        node_route = {}
        for idx, route in enumerate(routes):
            for node in route:
                node_route[node] = idx
        
        # Track route endpoints
        route_start = {idx: route[0] for idx, route in enumerate(routes)}
        route_end = {idx: route[-1] for idx, route in enumerate(routes)}
        
        for s, ni, nj in savings:
            if s <= 0:
                break
            ri = node_route[ni]
            rj = node_route[nj]
            if ri == rj:
                continue
            
            # Check if ni is at end of ri and nj is at start of rj (or vice versa)
            can_merge = False
            merge_order = None
            
            if route_end[ri] == ni and route_start[rj] == nj:
                can_merge = True
                merge_order = (ri, rj)
            elif route_end[rj] == nj and route_start[ri] == ni:
                can_merge = True
                merge_order = (rj, ri)
            elif route_end[ri] == ni and route_end[rj] == nj:
                # Reverse rj
                routes[rj] = routes[rj][::-1]
                route_start[rj] = routes[rj][0]
                route_end[rj] = routes[rj][-1]
                can_merge = True
                merge_order = (ri, rj)
            elif route_start[ri] == ni and route_start[rj] == nj:
                # Reverse ri
                routes[ri] = routes[ri][::-1]
                route_start[ri] = routes[ri][0]
                route_end[ri] = routes[ri][-1]
                can_merge = True
                merge_order = (ri, rj)
            
            if not can_merge:
                continue
            
            r1, r2 = merge_order
            # Check capacity
            total_demand = sum(demand_func(node) for node in routes[r1]) + sum(demand_func(node) for node in routes[r2])
            if total_demand > C:
                continue
            
            # Merge r2 into r1
            new_route = routes[r1] + routes[r2]
            routes[r1] = new_route
            routes[r2] = []
            
            for node in new_route:
                node_route[node] = r1
            route_start[r1] = new_route[0]
            route_end[r1] = new_route[-1]
        
        return [r for r in routes if r]
    
    def combine_lh_bh_routes(lh_routes, bh_routes):
        """Combine linehaul and backhaul routes to minimize total distance."""
        combined = []
        
        used_bh = [False] * len(bh_routes)
        
        # For each linehaul route, find best backhaul route to append
        # Savings of combining: dist[last_lh][0] + dist[0][first_bh] - dist[last_lh][first_bh]
        
        # Pre-compute route demands
        lh_demands = [sum(delivery[node] for node in r) for r in lh_routes]
        bh_demands = [sum(pickup[node-n] for node in r) for r in bh_routes]
        
        # Create savings for combining
        combine_savings = []
        for i, lhr in enumerate(lh_routes):
            last_lh_cust = get_customer(lhr[-1])
            for j, bhr in enumerate(bh_routes):
                first_bh_cust = get_customer(bhr[0])
                # Saving from combining vs separate trips
                saving = dist[last_lh_cust][0] + dist[0][first_bh_cust] - dist[last_lh_cust][first_bh_cust]
                combine_savings.append((saving, i, j))
        
        combine_savings.sort(reverse=True)
        
        used_lh = [False] * len(lh_routes)
        
        for saving, i, j in combine_savings:
            if saving <= 0:
                break
            if used_lh[i] or used_bh[j]:
                continue
            # Check capacity: both lh and bh must independently fit
            # (since after delivering all, load is 0, then pickup starts)
            if lh_demands[i] <= C and bh_demands[j] <= C:
                combined.append(lh_routes[i] + bh_routes[j])
                used_lh[i] = True
                used_bh[j] = True
        
        # Add remaining linehaul routes
        for i, r in enumerate(lh_routes):
            if not used_lh[i]:
                combined.append(r)
        
        # Add remaining backhaul routes
        for j, r in enumerate(bh_routes):
            if not used_bh[j]:
                combined.append(r)
        
        return combined
    
    def total_distance(routes):
        total = 0
        for route in routes:
            if not route:
                continue
            total += dist[0][get_customer(route[0])]
            for i in range(len(route) - 1):
                total += dist[get_customer(route[i])][get_customer(route[i+1])]
            total += dist[get_customer(route[-1])][0]
        return total
    
    def route_load_check(route):
        """Full load check for a mixed route."""
        # Determine initial load = sum of all delivery demands in route
        initial_load = 0
        for node in route:
            if node <= n:
                initial_load += delivery[node]
        
        if initial_load > C:
            return False
        
        load = initial_load
        for node in route:
            if node <= n:
                load -= delivery[node]
            else:
                load += pickup[node - n]
            if load > C:
                return False
        return True
    
    def is_linehaul(node):
        return 1 <= node <= n
    
    def is_backhaul(node):
        return node > n
    
    # Build initial solution
    routes = build_initial_solution()
    best_obj = total_distance(routes)
    best_routes = [r[:] for r in routes]
    
    if logger:
        logger.log(best_obj)
    
    # Local search improvement
    def two_opt_route(route):
        """Apply 2-opt within a single route."""
        improved = True
        while improved:
            improved = False
            for i in range(len(route) - 1):
                for j in range(i + 2, len(route)):
                    # Try reversing route[i+1:j+1]
                    new_route = route[:i+1] + route[i+1:j+1][::-1] + route[j+1:]
                    
                    # Check if all linehaul before backhaul
                    # Actually, we maintain LH-before-BH structure
                    valid = True
                    seen_backhaul = False
                    for node in new_route:
                        if is_backhaul(node):
                            seen_backhaul = True
                        elif is_linehaul(node) and seen_backhaul:
                            valid = False
                            break
                    
                    if not valid:
                        continue
                    
                    # Calculate distance change
                    ci = get_customer(route[i]) if i >= 0 else 0
                    ci1 = get_customer(route[i+1])
                    cj = get_customer(route[j])
                    cj1 = get_customer(route[j+1]) if j+1 < len(route) else 0
                    
                    # Use proper endpoints
                    if i == 0:
                        prev_i = 0
                    else:
                        prev_i = get_customer(route[i])
                    
                    # Old: ... route[i] -> route[i+1] ... route[j] -> route[j+1] ...
                    # New: ... route[i] -> route[j] ... route[i+1] -> route[j+1] ...
                    
                    old_dist = dist[ci][ci1]
                    new_dist_val = dist[ci][cj]
                    if j + 1 < len(route):
                        old_dist += dist[cj][get_customer(route[j+1])]
                        new_dist_val += dist[ci1][get_customer(route[j+1])]
                    else:
                        old_dist += dist[cj][0]
                        new_dist_val += dist[ci1][0]
                    
                    if i == 0:
                        pass  # ci is already route[i]
                    
                    if new_dist_val < old_dist:
                        route[:] = new_route
                        improved = True
                        break
                if improved:
                    break
    
    def or_opt_within_route(route):
        """Try moving segments of 1, 2, or 3 nodes within a route."""
        improved = True
        while improved:
            improved = False
            for seg_len in [1, 2, 3]:
                for i in range(len(route) - seg_len + 1):
                    segment = route[i:i+seg_len]
                    rest = route[:i] + route[i+seg_len:]
                    
                    for j in range(len(rest) + 1):
                        new_route = rest[:j] + segment + rest[j:]
                        
                        # Check LH before BH
                        valid = True
                        seen_bh = False
                        for node in new_route:
                            if is_backhaul(node):
                                seen_bh = True
                            elif is_linehaul(node) and seen_bh:
                                valid = False
                                break
                        if not valid:
                            continue
                        
                        new_d = route_distance(new_route)
                        old_d = route_distance(route)
                        if new_d < old_d - 0.5:
                            route[:] = new_route
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
    
    def fast_two_opt_route(route):
        """Apply 2-opt within a single route using distance delta."""
        n_r = len(route)
        if n_r < 2:
            return
        
        # Build customer sequence with depot at start and end
        custs = [0] + [get_customer(node) for node in route] + [0]
        
        improved = True
        while improved:
            improved = False
            for i in range(n_r):
                for j in range(i + 2, n_r + 1):
                    # Reverse segment from i+1 to j in custs (which is route[i] to route[j-1])
                    # In route indices: reverse route[i:j]
                    
                    # Check LH-BH ordering
                    # The reversed segment: route[i:j] reversed
                    # Check that no linehaul comes after any backhaul in the new route
                    new_route = route[:i] + route[i:j][::-1] + route[j:]
                    
                    valid = True
                    seen_bh = False
                    for node in new_route:
                        if is_backhaul(node):
                            seen_bh = True
                        elif is_linehaul(node) and seen_bh:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    # Delta calculation
                    ci = custs[i]  # node before reversed segment
                    ci1 = custs[i+1]  # first node of segment
                    cj = custs[j]  # last node of segment
                    cj1 = custs[j+1]  # node after segment
                    
                    delta = (dist[ci][cj] + dist[ci1][cj1]) - (dist[ci][ci1] + dist[cj][cj1])
                    
                    if delta < -0.5:
                        route[:] = new_route
                        custs = [0] + [get_customer(node) for node in route] + [0]
                        improved = True
                        break
                if improved:
                    break
    
    def relocate_between_routes(routes):
        """Try relocating a node from one route to another."""
        improved = True
        while improved:
            improved = False
            for r1_idx in range(len(routes)):
                if not routes[r1_idx]:
                    continue
                for pos1 in range(len(routes[r1_idx])):
                    node = routes[r1_idx][pos1]
                    node_cust = get_customer(node)
                    node_is_lh = is_linehaul(node)
                    node_demand = delivery[node] if node_is_lh else pickup[node - n]
                    
                    # Cost of removing node from r1
                    r1 = routes[r1_idx]
                    prev_c = 0 if pos1 == 0 else get_customer(r1[pos1-1])
                    next_c = 0 if pos1 == len(r1) - 1 else get_customer(r1[pos1+1])
                    
                    remove_saving = dist[prev_c][node_cust] + dist[node_cust][next_c] - dist[prev_c][next_c]
                    
                    best_insert_cost = float('inf')
                    best_r2_idx = -1
                    best_pos2 = -1
                    
                    for r2_idx in range(len(routes)):
                        if r2_idx == r1_idx:
                            continue
                        r2 = routes[r2_idx]
                        
                        # Check capacity
                        if node_is_lh:
                            r2_delivery = sum(delivery[nd] for nd in r2 if nd <= n)
                            if r2_delivery + node_demand > C:
                                continue
                        else:
                            r2_pickup = sum(pickup[nd-n] for nd in r2 if nd > n)
                            if r2_pickup + node_demand > C:
                                continue
                        
                        # Find best insertion position
                        for pos2 in range(len(r2) + 1):
                            # Check LH-BH ordering
                            if node_is_lh:
                                # Must insert before any backhaul
                                if pos2 > 0 and is_backhaul(r2[pos2-1]):
                                    continue
                            else:
                                # Must insert after all linehaul
                                if pos2 < len(r2) and is_linehaul(r2[pos2]):
                                    continue
                            
                            prev2_c = 0 if pos2 == 0 else get_customer(r2[pos2-1])
                            next2_c = 0 if pos2 >= len(r2) else get_customer(r2[pos2])
                            
                            insert_cost = dist[prev2_c][node_cust] + dist[node_cust][next2_c] - dist[prev2_c][next2_c]
                            
                            if insert_cost < best_insert_cost:
                                best_insert_cost = insert_cost
                                best_r2_idx = r2_idx
                                best_pos2 = pos2
                    
                    if best_insert_cost < remove_saving - 0.5:
                        # Perform the move
                        routes[r1_idx] = r1[:pos1] + r1[pos1+1:]
                        routes[best_r2_idx].insert(best_pos2, node)
                        improved = True
                        break
                if improved:
                    break
    
    def swap_between_routes(routes):
        """Try swapping nodes between routes."""
        improved = True
        while improved:
            improved = False
            for r1_idx in range(len(routes)):
                for r2_idx in range(r1_idx + 1, len(routes)):
                    if not routes[r1_idx] or not routes[r2_idx]:
                        continue
                    for pos1 in range(len(routes[r1_idx])):
                        for pos2 in range(len(routes[r2_idx])):
                            n1 = routes[r1_idx][pos1]
                            n2 = routes[r2_idx][pos2]
                            
                            # Both must be same type (lh or bh) for simple swap
                            if is_linehaul(n1) != is_linehaul(n2):
                                continue
                            
                            # Check capacity
                            r1 = routes[r1_idx]
                            r2 = routes[r2_idx]
                            
                            if is_linehaul(n1):
                                r1_del = sum(delivery[nd] for nd in r1 if nd <= n) - delivery[n1] + delivery[n2]
                                r2_del = sum(delivery[nd] for nd in r2 if nd <= n) - delivery[n2] + delivery[n1]
                                if r1_del > C or r2_del > C:
                                    continue
                            else:
                                r1_pk = sum(pickup[nd-n] for nd in r1 if nd > n) - pickup[n1-n] + pickup[n2-n]
                                r2_pk = sum(pickup[nd-n] for nd in r2 if nd > n) - pickup[n2-n] + pickup[n1-n]
                                if r1_pk > C or r2_pk > C:
                                    continue
                            
                            # Cost calculation
                            c1 = get_customer(n1)
                            c2 = get_customer(n2)
                            
                            prev1 = 0 if pos1 == 0 else get_customer(r1[pos1-1])
                            next1 = 0 if pos1 == len(r1)-1 else get_customer(r1[pos1+1])
                            prev2 = 0 if pos2 == 0 else get_customer(r2[pos2-1])
                            next2 = 0 if pos2 == len(r2)-1 else get_customer(r2[pos2+1])
                            
                            old_cost = (dist[prev1][c1] + dist[c1][next1] + 
                                       dist[prev2][c2] + dist[c2][next2])
                            new_cost = (dist[prev1][c2] + dist[c2][next1] + 
                                       dist[prev2][c1] + dist[c1][next2])
                            
                            if new_cost < old_cost - 0.5:
                                routes[r1_idx][pos1] = n2
                                routes[r2_idx][pos2] = n1
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
    
    # Apply local search
    def local_search(routes):
        """Apply various local search operators."""
        improved = True
        iteration = 0
        while improved and time.time() - start_time < time_limit * 0.9:
            improved = False
            iteration += 1
            
            # Intra-route 2-opt
            for route in routes:
                old_d = route_distance(route)
                fast_two_opt_route(route)
                if route_distance(route) < old_d - 0.5:
                    improved = True
            
            if time.time() - start_time > time_limit * 0.9:
                break
            
            # Inter-route relocate
            old_total = total_distance(routes)
            relocate_between_routes(routes)
            if total_distance(routes) < old_total - 0.5:
                improved = True
            
            if time.time() - start_time > time_limit * 0.9:
                break
            
            # Inter-route swap
            old_total = total_distance(routes)
            swap_between_routes(routes)
            if total_distance(routes) < old_total - 0.5:
                improved = True
            
            # Remove empty routes
            routes[:] = [r for r in routes if r]
    
    local_search(routes)
    
    obj = total_distance(routes)
    if obj < best_obj:
        best_obj = obj
        best_routes = [r[:] for r in routes]
        if logger:
            logger.log(best_obj)
    
    # Perturbation + local search (iterated local search)
    def perturb(routes):
        """Random perturbation: randomly relocate a few nodes."""
        routes = [r[:] for r in routes if r]
        if not routes:
            return routes
        
        num_moves = max(1, min(5, sum(len(r) for r in routes) // 10))
        
        for _ in range(num_moves):
            if not routes:
                break
            # Pick random non-empty route and random node
            non_empty = [i for i in range(len(routes)) if routes[i]]
            if not non_empty:
                break
            r1_idx = random.choice(non_empty)
            if not routes[r1_idx]:
                continue
            pos1 = random.randint(0, len(routes[r1_idx]) - 1)
            node = routes[r1_idx][pos1]
            
            # Remove from current route
            routes[r1_idx] = routes[r1_idx][:pos1] + routes[r1_idx][pos1+1:]
            
            # Pick target route (could be same or different)
            r2_idx = random.randint(0, len(routes) - 1)
            r2 = routes[r2_idx]
            
            # Find valid insertion positions
            valid_positions = []
            for pos2 in range(len(r2) + 1):
                if is_linehaul(node):
                    if pos2 > 0 and r2[pos2-1] > n:
                        continue
                    # Check capacity
                    r2_del = sum(delivery[nd] for nd in r2 if nd <= n)
                    if r2_del + delivery[node] > C:
                        continue
                else:
                    if pos2 < len(r2) and r2[pos2] <= n:
                        continue
                    r2_pk = sum(pickup[nd-n] for nd in r2 if nd > n)
                    if r2_pk + pickup[node-n] > C:
                        continue
                valid_positions.append(pos2)
            
            if valid_positions:
                pos2 = random.choice(valid_positions)
                routes[r2_idx].insert(pos2, node)
            else:
                # Create new route
                routes.append([node])
        
        routes = [r for r in routes if r]
        return routes
    
    # ILS loop
    current_routes = [r[:] for r in best_routes]
    current_obj = best_obj
    
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() - start_time < time_limit * 0.85:
        # Perturb
        new_routes = perturb(current_routes)
        
        # Local search
        local_search(new_routes)
        
        new_obj = total_distance(new_routes)
        
        if new_obj < best_obj:
            best_obj = new_obj
            best_routes = [r[:] for r in new_routes]
            if logger:
                logger.log(best_obj)
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # Accept with some probability (simulated annealing-like)
        if new_obj < current_obj or random.random() < 0.3:
            current_routes = new_routes
            current_obj = new_obj
        
        if no_improve_count > max_no_improve:
            # Restart from best
            current_routes = [r[:] for r in best_routes]
            current_obj = best_obj
            no_improve_count = 0
    
    # Try additional improvement: merge short routes
    def try_merge_routes(routes):
        improved = True
        while improved:
            improved = False
            for i in range(len(routes)):
                if not routes[i]:
                    continue
                for j in range(i+1, len(routes)):
                    if not routes[j]:
                        continue
                    
                    # Try appending j to i
                    # Check: all linehaul of i+j, then all backhaul of i+j
                    lh_i = [nd for nd in routes[i] if nd <= n]
                    bh_i = [nd for nd in routes[i] if nd > n]
                    lh_j = [nd for nd in routes[j] if nd <= n]
                    bh_j = [nd for nd in routes[j] if nd > n]
                    
                    total_del = sum(delivery[nd] for nd in lh_i + lh_j)
                    total_pk = sum(pickup[nd-n] for nd in bh_i + bh_j)
                    
                    if total_del > C or total_pk > C:
                        continue
                    
                    # Try different orderings of the merged route
                    candidates = []
                    
                    # Option 1: route_i_lh + route_j_lh + route_i_bh + route_j_bh
                    cand = lh_i + lh_j + bh_i + bh_j
                    candidates.append(cand)
                    
                    # Option 2: route_j_lh + route_i_lh + route_j_bh + route_i_bh
                    cand = lh_j + lh_i + bh_j + bh_i
                    candidates.append(cand)
                    
                    # Option 3: route_i_lh + route_j_lh + route_j_bh + route_i_bh
                    cand = lh_i + lh_j + bh_j + bh_i
                    candidates.append(cand)
                    
                    # Option 4: route_j_lh + route_i_lh + route_i_bh + route_j_bh
                    cand = lh_j + lh_i + bh_i + bh_j
                    candidates.append(cand)
                    
                    old_cost = route_distance(routes[i]) + route_distance(routes[j])
                    
                    best_cand = None
                    best_cand_cost = old_cost
                    
                    for cand in candidates:
                        if not cand:
                            continue
                        cand_cost = route_distance(cand)
                        if cand_cost < best_cand_cost - 0.5:
                            best_cand = cand
                            best_cand_cost = cand_cost
                    
                    if best_cand is not None:
                        routes[i] = best_cand
                        routes[j] = []
                        improved = True
                        break
                if improved:
                    break
            routes[:] = [r for r in routes if r]
    
    try_merge_routes(best_routes)
    
    # Final local search
    local_search(best_routes)
    
    final_obj = total_distance(best_routes)
    if final_obj < best_obj:
        best_obj = final_obj
        if logger:
            logger.log(best_obj)
    
    # Remove empty routes
    best_routes = [r for r in best_routes if r]
    
    # Format output
    output_routes = []
    output_routes_detailed = []
    
    for route in best_routes:
        # Add depot at start and end
        expanded_route = [0] + route + [0]
        output_routes.append(expanded_route)
        
        detailed = []
        # Depot start
        detailed.append({
            "node_id": 0,
            "role": "depot",
            "original_customer": 0,
            "quantity": 0
        })
        
        for node in route:
            if node <= n:
                detailed.append({
                    "node_id": node,
                    "role": "linehaul",
                    "original_customer": node,
                    "quantity": delivery[node]
                })
            else:
                orig = node - n
                detailed.append({
                    "node_id": node,
                    "role": "backhaul",
                    "original_customer": orig,
                    "quantity": pickup[orig]
                })
        
        # Depot end
        detailed.append({
            "node_id": 0,
            "role": "depot",
            "original_customer": 0,
            "quantity": 0
        })
        
        output_routes_detailed.append(detailed)
    
    solution = {
        "objective_value": best_obj,
        "routes": output_routes,
        "routes_detailed": output_routes_detailed
    }
    
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)
import argparse
import json
import math
import time
import random
import copy

def solve(instance_path, solution_path, time_limit, log_path=None):
    from solution_logger import SolutionLogger
    
    with open(instance_path, 'r') as f:
        data = json.load(f)
    
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    start_time = time.time()
    
    num_customers = data['num_customers']
    num_vehicles = data['num_vehicles']
    vehicle_capacity = data['vehicle_capacity']
    depot = data['depot']
    depot_end = data['depot_end']
    customers = data['customers']
    dist_matrix = data['distance_matrix']
    rho = data['inconvenience_cost_function']['rho']
    
    # Index mapping: depot=0, customers 1..num_customers, depot_end=num_customers+1
    n = num_customers
    depot_id = 0
    depot_end_id = n + 1
    
    # Customer data indexed by customer id (1-based)
    cust_demand = {}
    cust_tw = {}
    cust_service_time = {}
    cust_preferred = {}
    
    for c in customers:
        cid = c['id']
        cust_demand[cid] = c['demand']
        cust_tw[cid] = (c['time_window'][0], c['time_window'][1])
        cust_service_time[cid] = c['service_time']
        cust_preferred[cid] = c['preferred_time']
    
    depot_tw = (depot['time_window'][0], depot['time_window'][1])
    depot_end_tw = (depot_end['time_window'][0], depot_end['time_window'][1])
    
    def travel_time(i, j):
        return dist_matrix[i][j]
    
    def routing_cost(i, j):
        return dist_matrix[i][j]
    
    def inconvenience_cost(cid, service_start):
        return rho * (service_start - cust_preferred[cid]) ** 2
    
    def get_tw(node):
        if node == depot_id:
            return depot_tw
        elif node == depot_end_id:
            return depot_end_tw
        else:
            return cust_tw[node]
    
    def get_service_time(node):
        if node == depot_id or node == depot_end_id:
            return 0
        return cust_service_time[node]
    
    def optimize_schedule(route):
        """Given a route (list of customer ids, not including depot), compute optimal schedule."""
        # Route: list of customer ids visited in order
        # We need to determine service start times
        # Forward pass: earliest possible times
        if not route:
            return {}, 0, 0
        
        full_route = [depot_id] + route + [depot_end_id]
        earliest = [0.0] * len(full_route)
        earliest[0] = depot_tw[0]
        
        for i in range(1, len(full_route)):
            prev = full_route[i-1]
            curr = full_route[i]
            arr = earliest[i-1] + get_service_time(prev) + travel_time(prev, curr)
            tw = get_tw(curr)
            earliest[i] = max(arr, tw[0])
        
        # Check feasibility
        for i in range(len(full_route)):
            tw = get_tw(full_route[i])
            if earliest[i] > tw[1] + 1e-6:
                return None, float('inf'), float('inf')
        
        # Check depot end
        if earliest[-1] > depot_end_tw[1] + 1e-6:
            return None, float('inf'), float('inf')
        
        # Now optimize: we want to minimize sum of rho*(t_i - preferred_i)^2 for customers
        # subject to: t_i >= earliest[i] for each position
        #             t_i >= t_{i-1} + service_time(prev) + travel_time(prev, curr) for consecutive
        #             t_i <= tw[1] for each position
        # This is a quadratic optimization with chain constraints
        
        # Use forward-backward approach
        # First compute latest times (backward pass)
        latest = [0.0] * len(full_route)
        latest[-1] = depot_end_tw[1]
        
        for i in range(len(full_route) - 2, -1, -1):
            curr = full_route[i]
            nxt = full_route[i+1]
            tw = get_tw(curr)
            dep = latest[i+1] - travel_time(curr, nxt) - get_service_time(curr)
            latest[i] = min(dep, tw[1])
        
        # Check feasibility
        for i in range(len(full_route)):
            if earliest[i] > latest[i] + 1e-6:
                return None, float('inf'), float('inf')
        
        # For customers (indices 1 to len(route)), set service time to preferred if possible,
        # otherwise clamp to [earliest, latest]
        schedule = {}
        times = [0.0] * len(full_route)
        times[0] = earliest[0]
        
        for i in range(1, len(full_route)):
            prev = full_route[i-1]
            curr = full_route[i]
            arr = times[i-1] + get_service_time(prev) + travel_time(prev, curr)
            tw = get_tw(curr)
            lo = max(arr, tw[0])
            hi = min(latest[i], tw[1])
            
            if curr != depot_end_id and curr != depot_id:
                # Try preferred time
                pref = cust_preferred[curr]
                times[i] = max(lo, min(hi, pref))
            else:
                times[i] = lo
        
        # Now do a more sophisticated optimization: iterative refinement
        # Actually, let's do a proper quadratic chain optimization
        # We'll use a simple approach: set each to preferred, then do forward feasibility fix
        
        # Better approach: set desired times, then fix forward, then fix backward
        desired = [0.0] * len(full_route)
        desired[0] = earliest[0]
        for i in range(1, len(full_route) - 1):
            curr = full_route[i]
            desired[i] = cust_preferred[curr]
        desired[-1] = earliest[-1]  # don't care about depot end timing
        
        # Forward pass: ensure feasibility
        times2 = [0.0] * len(full_route)
        times2[0] = max(desired[0], earliest[0])
        for i in range(1, len(full_route)):
            prev = full_route[i-1]
            curr = full_route[i]
            arr = times2[i-1] + get_service_time(prev) + travel_time(prev, curr)
            tw = get_tw(curr)
            lo = max(arr, tw[0])
            hi = tw[1]
            if curr != depot_end_id and curr != depot_id:
                pref = cust_preferred[curr]
                times2[i] = max(lo, min(hi, pref))
            else:
                times2[i] = lo
        
        # Compute costs
        total_inc = 0
        for i in range(1, len(full_route) - 1):
            cid = full_route[i]
            schedule[cid] = times2[i]
            total_inc += inconvenience_cost(cid, times2[i])
        
        total_route_cost = 0
        for i in range(len(full_route) - 1):
            total_route_cost += routing_cost(full_route[i], full_route[i+1])
        
        return schedule, total_route_cost, total_inc
    
    def evaluate_route(route):
        """Evaluate a single route. Returns (schedule, routing_cost, inconvenience_cost) or None if infeasible."""
        if not route:
            return {}, 0, 0
        
        # Check capacity
        total_demand = sum(cust_demand[c] for c in route)
        if total_demand > vehicle_capacity:
            return None, float('inf'), float('inf')
        
        return optimize_schedule(route)
    
    def is_route_feasible(route):
        """Check if a route is feasible (capacity and time windows)."""
        if not route:
            return True
        total_demand = sum(cust_demand[c] for c in route)
        if total_demand > vehicle_capacity:
            return False
        sched, rc, ic = optimize_schedule(route)
        return sched is not None
    
    def compute_solution_cost(routes):
        """Compute total cost of a solution."""
        total_cost = 0
        all_schedule = {}
        for route in routes:
            if not route:
                continue
            sched, rc, ic = evaluate_route(route)
            if sched is None:
                return float('inf'), {}
            total_cost += rc + ic
            all_schedule.update(sched)
        return total_cost, all_schedule
    
    def route_demand(route):
        return sum(cust_demand[c] for c in route)
    
    # ============ Construction Heuristic ============
    # Solomon's I1 insertion heuristic with modifications for inconvenience
    
    def cheapest_insertion_cost(route, cust, pos):
        """Cost of inserting customer cust at position pos in route."""
        new_route = route[:pos] + [cust] + route[pos:]
        
        # Check capacity
        if route_demand(new_route) > vehicle_capacity:
            return float('inf')
        
        sched, rc, ic = optimize_schedule(new_route)
        if sched is None:
            return float('inf')
        return rc + ic
    
    def build_initial_solution():
        """Build initial solution using cheapest insertion."""
        unassigned = list(range(1, n + 1))
        random.shuffle(unassigned)
        routes = []
        
        while unassigned and len(routes) < num_vehicles:
            # Start a new route with the farthest unassigned customer from depot
            # or the one with earliest deadline
            best_seed = None
            best_val = -1
            for c in unassigned:
                val = travel_time(depot_id, c)
                if val > best_val:
                    best_val = val
                    best_seed = c
            
            if best_seed is None:
                break
            
            route = [best_seed]
            unassigned.remove(best_seed)
            
            improved = True
            while improved and unassigned:
                improved = False
                best_cust = None
                best_pos = None
                best_cost_increase = float('inf')
                
                # Current route cost
                sched, rc, ic = evaluate_route(route)
                if sched is None:
                    break
                current_cost = rc + ic
                
                for c in unassigned:
                    if route_demand(route) + cust_demand[c] > vehicle_capacity:
                        continue
                    for pos in range(len(route) + 1):
                        new_route = route[:pos] + [c] + route[pos:]
                        sched2, rc2, ic2 = optimize_schedule(new_route)
                        if sched2 is None:
                            continue
                        cost_increase = (rc2 + ic2) - current_cost
                        if cost_increase < best_cost_increase:
                            best_cost_increase = cost_increase
                            best_cust = c
                            best_pos = pos
                            improved = True
                
                if best_cust is not None and best_cost_increase < float('inf'):
                    route = route[:best_pos] + [best_cust] + route[best_pos:]
                    unassigned.remove(best_cust)
                    improved = True
                else:
                    improved = False
            
            routes.append(route)
        
        # If any unassigned customers remain, try to insert them
        for c in list(unassigned):
            best_route_idx = None
            best_pos = None
            best_cost = float('inf')
            
            for ri, route in enumerate(routes):
                if route_demand(route) + cust_demand[c] > vehicle_capacity:
                    continue
                sched, rc, ic = evaluate_route(route)
                if sched is None:
                    continue
                current_cost = rc + ic
                
                for pos in range(len(route) + 1):
                    new_route = route[:pos] + [c] + route[pos:]
                    sched2, rc2, ic2 = optimize_schedule(new_route)
                    if sched2 is None:
                        continue
                    cost_increase = (rc2 + ic2) - current_cost
                    if cost_increase < best_cost:
                        best_cost = cost_increase
                        best_route_idx = ri
                        best_pos = pos
            
            if best_route_idx is not None:
                routes[best_route_idx] = routes[best_route_idx][:best_pos] + [c] + routes[best_route_idx][best_pos:]
                unassigned.remove(c)
            elif len(routes) < num_vehicles:
                routes.append([c])
                unassigned.remove(c)
        
        return routes, unassigned
    
    def build_initial_solution_fast():
        """Faster construction using nearest neighbor with time window awareness."""
        unassigned = set(range(1, n + 1))
        routes = []
        
        while unassigned and len(routes) < num_vehicles:
            route = []
            current_node = depot_id
            current_time = depot_tw[0]
            current_load = 0
            
            while unassigned:
                best_cust = None
                best_score = float('inf')
                
                for c in unassigned:
                    if current_load + cust_demand[c] > vehicle_capacity:
                        continue
                    arr = current_time + (get_service_time(current_node) if current_node != depot_id else 0) + travel_time(current_node, c)
                    tw = cust_tw[c]
                    if arr > tw[1] + 1e-6:
                        continue
                    service_start = max(arr, tw[0])
                    # Check if we can return to depot after
                    depart = service_start + cust_service_time[c]
                    arr_depot = depart + travel_time(c, depot_end_id)
                    if arr_depot > depot_end_tw[1] + 1e-6:
                        continue
                    
                    # Score: combination of routing cost and inconvenience
                    route_add = travel_time(current_node, c)
                    inc = inconvenience_cost(c, service_start)
                    score = route_add + inc
                    
                    if score < best_score:
                        best_score = score
                        best_cust = c
                
                if best_cust is None:
                    break
                
                c = best_cust
                arr = current_time + (get_service_time(current_node) if current_node != depot_id else 0) + travel_time(current_node, c)
                service_start = max(arr, cust_tw[c][0])
                
                route.append(c)
                unassigned.remove(c)
                current_node = c
                current_time = service_start
                current_load += cust_demand[c]
            
            if route:
                routes.append(route)
        
        return routes, list(unassigned)
    
    # Build initial solution
    routes, unassigned = build_initial_solution_fast()
    
    # If there are unassigned customers, try harder
    if unassigned:
        # Try inserting remaining customers
        for c in list(unassigned):
            best_route_idx = None
            best_pos = None
            best_cost = float('inf')
            
            for ri, route in enumerate(routes):
                if route_demand(route) + cust_demand[c] > vehicle_capacity:
                    continue
                sched, rc, ic = evaluate_route(route)
                if sched is None:
                    continue
                current_cost = rc + ic
                
                for pos in range(len(route) + 1):
                    new_route = route[:pos] + [c] + route[pos:]
                    sched2, rc2, ic2 = optimize_schedule(new_route)
                    if sched2 is None:
                        continue
                    cost_increase = (rc2 + ic2) - current_cost
                    if cost_increase < best_cost:
                        best_cost = cost_increase
                        best_route_idx = ri
                        best_pos = pos
            
            if best_route_idx is not None:
                routes[best_route_idx] = routes[best_route_idx][:best_pos] + [c] + routes[best_route_idx][best_pos:]
                unassigned.remove(c)
            elif len(routes) < num_vehicles:
                routes.append([c])
                unassigned.remove(c)
    
    if unassigned:
        # Last resort: create new routes for remaining customers
        for c in unassigned:
            if len(routes) < num_vehicles:
                routes.append([c])
            else:
                # Force insert into route with least demand
                routes_by_demand = sorted(range(len(routes)), key=lambda i: route_demand(routes[i]))
                for ri in routes_by_demand:
                    for pos in range(len(routes[ri]) + 1):
                        new_route = routes[ri][:pos] + [c] + routes[ri][pos:]
                        sched, rc, ic = optimize_schedule(new_route)
                        if sched is not None:
                            routes[ri] = new_route
                            break
                    else:
                        continue
                    break
    
    # Compute initial cost
    best_cost, best_schedule = compute_solution_cost(routes)
    best_routes = [r[:] for r in routes]
    
    def make_solution_dict(routes, schedule):
        sol_routes = []
        for route in routes:
            if route:
                sol_routes.append([depot_id] + route + [depot_end_id])
        sol_schedule = {str(k): v for k, v in schedule.items()}
        return {
            "objective_value": None,
            "routes": sol_routes,
            "schedule": sol_schedule
        }
    
    if best_cost < float('inf'):
        sol = make_solution_dict(best_routes, best_schedule)
        sol["objective_value"] = best_cost
        if logger:
            logger.log_solution(best_cost, sol)
    
    # Cache route costs
    def route_cost(route):
        if not route:
            return 0, {}, 0, 0
        sched, rc, ic = evaluate_route(route)
        if sched is None:
            return float('inf'), None, float('inf'), float('inf')
        return rc + ic, sched, rc, ic
    
    # Precompute route costs
    route_costs = []
    route_schedules = []
    for route in routes:
        c, s, _, _ = route_cost(route)
        route_costs.append(c)
        route_schedules.append(s if s else {})
    
    # ============ Local Search ============
    
    def try_relocate(routes, route_costs, route_schedules):
        """Try relocating a customer from one route to another."""
        best_improvement = 0
        best_move = None
        
        for ri in range(len(routes)):
            if not routes[ri]:
                continue
            for ci in range(len(routes[ri])):
                cust = routes[ri][ci]
                # Remove customer from route ri
                new_route_ri = routes[ri][:ci] + routes[ri][ci+1:]
                cost_ri_new, sched_ri_new, _, _ = route_cost(new_route_ri)
                
                if cost_ri_new == float('inf') and new_route_ri:
                    continue
                
                saving_ri = route_costs[ri] - cost_ri_new
                
                for rj in range(len(routes)):
                    if ri == rj:
                        continue
                    if route_demand(routes[rj]) + cust_demand[cust] > vehicle_capacity:
                        continue
                    
                    for pos in range(len(routes[rj]) + 1):
                        new_route_rj = routes[rj][:pos] + [cust] + routes[rj][pos:]
                        cost_rj_new, sched_rj_new, _, _ = route_cost(new_route_rj)
                        
                        if cost_rj_new == float('inf'):
                            continue
                        
                        increase_rj = cost_rj_new - route_costs[rj]
                        improvement = saving_ri - increase_rj
                        
                        if improvement > best_improvement + 1e-6:
                            best_improvement = improvement
                            best_move = (ri, ci, rj, pos, new_route_ri, new_route_rj, 
                                        cost_ri_new, cost_rj_new, sched_ri_new, sched_rj_new)
        
        return best_move, best_improvement
    
    def try_swap(routes, route_costs, route_schedules):
        """Try swapping two customers between different routes."""
        best_improvement = 0
        best_move = None
        
        for ri in range(len(routes)):
            if not routes[ri]:
                continue
            for ci in range(len(routes[ri])):
                c1 = routes[ri][ci]
                for rj in range(ri + 1, len(routes)):
                    if not routes[rj]:
                        continue
                    for cj in range(len(routes[rj])):
                        c2 = routes[rj][cj]
                        
                        # Check capacity
                        new_demand_ri = route_demand(routes[ri]) - cust_demand[c1] + cust_demand[c2]
                        new_demand_rj = route_demand(routes[rj]) - cust_demand[c2] + cust_demand[c1]
                        if new_demand_ri > vehicle_capacity or new_demand_rj > vehicle_capacity:
                            continue
                        
                        new_route_ri = routes[ri][:ci] + [c2] + routes[ri][ci+1:]
                        new_route_rj = routes[rj][:cj] + [c1] + routes[rj][cj+1:]
                        
                        cost_ri_new, sched_ri_new, _, _ = route_cost(new_route_ri)
                        if cost_ri_new == float('inf'):
                            continue
                        cost_rj_new, sched_rj_new, _, _ = route_cost(new_route_rj)
                        if cost_rj_new == float('inf'):
                            continue
                        
                        improvement = (route_costs[ri] + route_costs[rj]) - (cost_ri_new + cost_rj_new)
                        
                        if improvement > best_improvement + 1e-6:
                            best_improvement = improvement
                            best_move = (ri, ci, rj, cj, new_route_ri, new_route_rj,
                                        cost_ri_new, cost_rj_new, sched_ri_new, sched_rj_new)
        
        return best_move, best_improvement
    
    def try_2opt_within(routes, route_costs, route_schedules):
        """Try 2-opt within each route."""
        best_improvement = 0
        best_move = None
        
        for ri in range(len(routes)):
            route = routes[ri]
            if len(route) < 2:
                continue
            for i in range(len(route) - 1):
                for j in range(i + 1, len(route)):
                    # Reverse segment from i to j
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    cost_new, sched_new, _, _ = route_cost(new_route)
                    if cost_new == float('inf'):
                        continue
                    improvement = route_costs[ri] - cost_new
                    if improvement > best_improvement + 1e-6:
                        best_improvement = improvement
                        best_move = (ri, new_route, cost_new, sched_new)
        
        return best_move, best_improvement
    
    def try_or_opt(routes, route_costs, route_schedules):
        """Try Or-opt: move a segment of 1, 2, or 3 consecutive customers within or between routes."""
        best_improvement = 0
        best_move = None
        
        for ri in range(len(routes)):
            route = routes[ri]
            if not route:
                continue
            for seg_len in range(1, min(4, len(route) + 1)):
                for start in range(len(route) - seg_len + 1):
                    segment = route[start:start + seg_len]
                    remaining = route[:start] + route[start + seg_len:]
                    seg_demand = sum(cust_demand[c] for c in segment)
                    
                    # Try reinserting within same route
                    cost_remaining, sched_remaining, _, _ = route_cost(remaining)
                    if cost_remaining == float('inf') and remaining:
                        continue
                    
                    for pos in range(len(remaining) + 1):
                        if pos == start:
                            continue  # Same position
                        new_route = remaining[:pos] + segment + remaining[pos:]
                        cost_new, sched_new, _, _ = route_cost(new_route)
                        if cost_new == float('inf'):
                            continue
                        improvement = route_costs[ri] - cost_new
                        if improvement > best_improvement + 1e-6:
                            best_improvement = improvement
                            best_move = ('intra', ri, new_route, cost_new, sched_new)
                    
                    # Try reinserting into other routes
                    for rj in range(len(routes)):
                        if ri == rj:
                            continue
                        if route_demand(routes[rj]) + seg_demand > vehicle_capacity:
                            continue
                        for pos in range(len(routes[rj]) + 1):
                            new_route_rj = routes[rj][:pos] + segment + routes[rj][pos:]
                            cost_rj_new, sched_rj_new, _, _ = route_cost(new_route_rj)
                            if cost_rj_new == float('inf'):
                                continue
                            improvement = (route_costs[ri] + route_costs[rj]) - (cost_remaining + cost_rj_new)
                            if improvement > best_improvement + 1e-6:
                                best_improvement = improvement
                                best_move = ('inter', ri, rj, remaining, new_route_rj,
                                            cost_remaining, cost_rj_new, sched_remaining, sched_rj_new)
        
        return best_move, best_improvement
    
    def try_intra_relocate(routes, route_costs, route_schedules):
        """Try relocating a customer within the same route."""
        best_improvement = 0
        best_move = None
        
        for ri in range(len(routes)):
            route = routes[ri]
            if len(route) < 2:
                continue
            for ci in range(len(route)):
                cust = route[ci]
                remaining = route[:ci] + route[ci+1:]
                for pos in range(len(remaining) + 1):
                    if pos == ci:
                        continue
                    new_route = remaining[:pos] + [cust] + remaining[pos:]
                    cost_new, sched_new, _, _ = route_cost(new_route)
                    if cost_new == float('inf'):
                        continue
                    improvement = route_costs[ri] - cost_new
                    if improvement > best_improvement + 1e-6:
                        best_improvement = improvement
                        best_move = (ri, new_route, cost_new, sched_new)
        
        return best_move, best_improvement
    
    # Main local search loop
    iteration = 0
    no_improve_count = 0
    
    while time.time() - start_time < time_limit - 1:
        iteration += 1
        improved = False
        
        # Try relocate (inter-route)
        if time.time() - start_time < time_limit - 1:
            move, imp = try_relocate(routes, route_costs, route_schedules)
            if move and imp > 1e-6:
                ri, ci, rj, pos, new_ri, new_rj, cost_ri, cost_rj, sched_ri, sched_rj = move
                routes[ri] = new_ri
                routes[rj] = new_rj
                route_costs[ri] = cost_ri
                route_costs[rj] = cost_rj
                route_schedules[ri] = sched_ri if sched_ri else {}
                route_schedules[rj] = sched_rj if sched_rj else {}
                improved = True
        
        # Try swap
        if time.time() - start_time < time_limit - 1:
            move, imp = try_swap(routes, route_costs, route_schedules)
            if move and imp > 1e-6:
                ri, ci, rj, cj, new_ri, new_rj, cost_ri, cost_rj, sched_ri, sched_rj = move
                routes[ri] = new_ri
                routes[rj] = new_rj
                route_costs[ri] = cost_ri
                route_costs[rj] = cost_rj
                route_schedules[ri] = sched_ri if sched_ri else {}
                route_schedules[rj] = sched_rj if sched_rj else {}
                improved = True
        
        # Try 2-opt
        if time.time() - start_time < time_limit - 1:
            move, imp = try_2opt_within(routes, route_costs, route_schedules)
            if move and imp > 1e-6:
                ri, new_route, cost_new, sched_new = move
                routes[ri] = new_route
                route_costs[ri] = cost_new
                route_schedules[ri] = sched_new if sched_new else {}
                improved = True
        
        # Try intra-relocate
        if time.time() - start_time < time_limit - 1:
            move, imp = try_intra_relocate(routes, route_costs, route_schedules)
            if move and imp > 1e-6:
                ri, new_route, cost_new, sched_new = move
                routes[ri] = new_route
                route_costs[ri] = cost_new
                route_schedules[ri] = sched_new if sched_new else {}
                improved = True
        
        # Try or-opt
        if time.time() - start_time < time_limit - 1:
            move, imp = try_or_opt(routes, route_costs, route_schedules)
            if move and imp > 1e-6:
                if move[0] == 'intra':
                    _, ri, new_route, cost_new, sched_new = move
                    routes[ri] = new_route
                    route_costs[ri] = cost_new
                    route_schedules[ri] = sched_new if sched_new else {}
                else:
                    _, ri, rj, new_ri, new_rj, cost_ri, cost_rj, sched_ri, sched_rj = move
                    routes[ri] = new_ri
                    routes[rj] = new_rj
                    route_costs[ri] = cost_ri
                    route_costs[rj] = cost_rj
                    route_schedules[ri] = sched_ri if sched_ri else {}
                    route_schedules[rj] = sched_rj if sched_rj else {}
                improved = True
        
        if improved:
            no_improve_count = 0
            total_cost = sum(route_costs[i] for i in range(len(routes)))
            all_sched = {}
            for s in route_schedules:
                if s:
                    all_sched.update(s)
            
            if total_cost < best_cost - 1e-6:
                best_cost = total_cost
                best_routes = [r[:] for r in routes]
                best_schedule = dict(all_sched)
                
                sol = make_solution_dict(best_routes, best_schedule)
                sol["objective_value"] = best_cost
                if logger:
                    logger.log_solution(best_cost, sol)
        else:
            no_improve_count += 1
            if no_improve_count >= 3:
                # Try perturbation
                if len(routes) >= 2 and time.time() - start_time < time_limit - 2:
                    # Random perturbation: move a random customer to a random position in another route
                    non_empty = [i for i in range(len(routes)) if routes[i]]
                    if len(non_empty) >= 2:
                        ri = random.choice(non_empty)
                        if routes[ri]:
                            ci = random.randint(0, len(routes[ri]) - 1)
                            cust = routes[ri][ci]
                            
                            candidates = [i for i in non_empty if i != ri and 
                                        route_demand(routes[i]) + cust_demand[cust] <= vehicle_capacity]
                            if candidates:
                                rj = random.choice(candidates)
                                pos = random.randint(0, len(routes[rj]))
                                
                                new_ri = routes[ri][:ci] + routes[ri][ci+1:]
                                new_rj = routes[rj][:pos] + [cust] + routes[rj][pos:]
                                
                                c_ri, s_ri, _, _ = route_cost(new_ri)
                                c_rj, s_rj, _, _ = route_cost(new_rj)
                                
                                if c_ri < float('inf') and c_rj < float('inf'):
                                    routes[ri] = new_ri
                                    routes[rj] = new_rj
                                    route_costs[ri] = c_ri
                                    route_costs[rj] = c_rj
                                    route_schedules[ri] = s_ri if s_ri else {}
                                    route_schedules[rj] = s_rj if s_rj else {}
                                    no_improve_count = 0
                                    continue
                
                if no_improve_count >= 10:
                    break
    
    # Now try to improve schedules more carefully using quadratic optimization
    # For each route, try to optimize the schedule more carefully
    def optimize_schedule_advanced(route):
        """More careful schedule optimization using iterative approach."""
        if not route:
            return {}, 0, 0
        
        full_route = [depot_id] + route + [depot_end_id]
        n_stops = len(full_route)
        
        # Compute earliest and latest times
        earliest = [0.0] * n_stops
        earliest[0] = depot_tw[0]
        for i in range(1, n_stops):
            prev = full_route[i-1]
            curr = full_route[i]
            arr = earliest[i-1] + get_service_time(prev) + travel_time(prev, curr)
            tw = get_tw(curr)
            earliest[i] = max(arr, tw[0])
        
        latest = [0.0] * n_stops
        latest[-1] = depot_end_tw[1]
        for i in range(n_stops - 2, -1, -1):
            curr = full_route[i]
            nxt = full_route[i+1]
            tw = get_tw(curr)
            dep = latest[i+1] - travel_time(curr, nxt) - get_service_time(curr)
            latest[i] = min(dep, tw[1])
        
        # Check feasibility
        for i in range(n_stops):
            if earliest[i] > latest[i] + 1e-6:
                return None, float('inf'), float('inf')
        
        # Iterative optimization: for each customer position, set to preferred time
        # while maintaining feasibility with chain constraints
        times = list(earliest)
        
        # Multiple passes
        for _ in range(20):
            changed = False
            # Forward pass: try to set each to preferred, maintaining forward constraints
            for i in range(1, n_stops - 1):
                curr = full_route[i]
                prev = full_route[i-1]
                lo = max(times[i-1] + get_service_time(prev) + travel_time(prev, curr), get_tw(curr)[0])
                hi = min(latest[i], get_tw(curr)[1])
                pref = cust_preferred[curr]
                new_t = max(lo, min(hi, pref))
                if abs(new_t - times[i]) > 1e-9:
                    times[i] = new_t
                    changed = True
            
            # Backward pass: try to adjust each to preferred, maintaining backward constraints
            for i in range(n_stops - 2, 0, -1):
                curr = full_route[i]
                nxt = full_route[i+1]
                prev = full_route[i-1]
                lo = max(times[i-1] + get_service_time(prev) + travel_time(prev, curr), get_tw(curr)[0])
                hi_back = times[i+1] - travel_time(curr, nxt) - get_service_time(curr)
                hi = min(hi_back, get_tw(curr)[1], latest[i])
                if hi < lo - 1e-6:
                    return None, float('inf'), float('inf')
                pref = cust_preferred[curr]
                new_t = max(lo, min(hi, pref))
                if abs(new_t - times[i]) > 1e-9:
                    times[i] = new_t
                    changed = True
            
            if not changed:
                break
        
        schedule = {}
        total_inc = 0
        for i in range(1, n_stops - 1):
            cid = full_route[i]
            schedule[cid] = times[i]
            total_inc += inconvenience_cost(cid, times[i])
        
        total_route_cost = 0
        for i in range(n_stops - 1):
            total_route_cost += routing_cost(full_route[i], full_route[i+1])
        
        return schedule, total_route_cost, total_inc
    
    # Re-optimize schedules for best routes
    final_schedule = {}
    final_cost = 0
    for route in best_routes:
        if not route:
            continue
        sched, rc, ic = optimize_schedule_advanced(route)
        if sched is None:
            sched, rc, ic = optimize_schedule(route)
        if sched:
            final_schedule.update(sched)
        final_cost += rc + ic
    
    if final_cost < best_cost:
        best_cost = final_cost
        best_schedule = final_schedule
    
    # Build output
    sol_routes = []
    for route in best_routes:
        if route:
            sol_routes.append([depot_id] + route + [depot_end_id])
    
    sol_schedule = {str(k): v for k, v in best_schedule.items()}
    
    solution = {
        "objective_value": best_cost,
        "routes": sol_routes,
        "schedule": sol_schedule
    }
    
    if logger:
        logger.log_solution(best_cost, solution)
    
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()
    
    random.seed(42)
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)
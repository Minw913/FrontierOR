import argparse
import json
import math
import time
import random
import itertools
from copy import deepcopy

def euclidean_distance(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def solve_instance(instance, time_limit, logger=None):
    start_time = time.time()
    
    depot = instance['depot']
    satellites = instance['satellites']
    customers = instance['customers']
    groups = instance['groups']
    
    Q1 = instance['Q1']
    Q2 = instance['Q2']
    T1 = instance['T1']
    T2 = instance['T2']
    num_satellites = instance['num_satellites']
    num_customers = instance['num_customers']
    num_groups = instance['num_groups']
    
    # Build coordinate lookup
    coords = {}
    coords[depot['id']] = (depot['x'], depot['y'])
    for s in satellites:
        coords[s['id']] = (s['x'], s['y'])
    for c in customers:
        coords[c['id']] = (c['x'], c['y'])
    
    # Distance cache
    dist_cache = {}
    all_ids = list(coords.keys())
    for i in all_ids:
        for j in all_ids:
            if i != j:
                dist_cache[(i, j)] = euclidean_distance(coords[i], coords[j])
    
    def dist(i, j):
        if i == j:
            return 0.0
        return dist_cache[(i, j)]
    
    # Index mappings
    sat_by_id = {s['id']: s for s in satellites}
    cust_by_id = {c['id']: c for c in customers}
    group_by_id = {g['group_id']: g for g in groups}
    
    # Group demands
    group_demand = {}
    for g in groups:
        group_demand[g['group_id']] = sum(cust_by_id[cid]['demand'] for cid in g['customer_ids'])
    
    # Customer to group mapping
    cust_to_group = {}
    for c in customers:
        cust_to_group[c['id']] = c['group']
    
    # Greedy nearest-neighbor TSP for a set of nodes starting and ending at a depot
    def nn_tsp(depot_id, node_ids):
        if not node_ids:
            return [depot_id, depot_id], 0.0
        unvisited = list(node_ids)
        route = [depot_id]
        total_dist = 0.0
        current = depot_id
        while unvisited:
            nearest = min(unvisited, key=lambda n: dist(current, n))
            total_dist += dist(current, nearest)
            route.append(nearest)
            current = nearest
            unvisited.remove(nearest)
        total_dist += dist(current, depot_id)
        route.append(depot_id)
        return route, total_dist
    
    # 2-opt improvement for a route
    def two_opt(route, max_iter=100):
        n = len(route)
        if n <= 4:
            return route, sum(dist(route[i], route[i+1]) for i in range(n-1))
        improved = True
        iterations = 0
        while improved and iterations < max_iter:
            improved = False
            iterations += 1
            for i in range(1, n - 2):
                for j in range(i + 1, n - 1):
                    d1 = dist(route[i-1], route[i]) + dist(route[j], route[j+1])
                    d2 = dist(route[i-1], route[j]) + dist(route[i], route[j+1])
                    if d2 < d1 - 1e-10:
                        route[i:j+1] = reversed(route[i:j+1])
                        improved = True
        total = sum(dist(route[i], route[i+1]) for i in range(n-1))
        return route, total
    
    def build_second_echelon_routes(sat_id, customer_ids):
        """Build VRP routes for customers assigned to a satellite using Clarke-Wright savings."""
        if not customer_ids:
            return [], 0.0
        
        custs = list(customer_ids)
        demands = {cid: cust_by_id[cid]['demand'] for cid in custs}
        
        # Start with individual routes
        routes = []
        for cid in custs:
            routes.append([cid])
        
        # Compute savings
        savings = []
        for i in range(len(custs)):
            for j in range(i+1, len(custs)):
                ci, cj = custs[i], custs[j]
                s = dist(sat_id, ci) + dist(sat_id, cj) - dist(ci, cj)
                savings.append((s, ci, cj))
        savings.sort(reverse=True)
        
        # Route lookup
        cust_route = {}
        for idx, r in enumerate(routes):
            for c in r:
                cust_route[c] = idx
        
        for s, ci, cj in savings:
            ri = cust_route[ci]
            rj = cust_route[cj]
            if ri == rj:
                continue
            route_i = routes[ri]
            route_j = routes[rj]
            if route_i is None or route_j is None:
                continue
            
            # Check capacity
            demand_i = sum(demands[c] for c in route_i)
            demand_j = sum(demands[c] for c in route_j)
            if demand_i + demand_j > Q2:
                continue
            
            # Check if ci and cj are at the ends of their routes
            can_merge = False
            new_route = None
            if route_i[-1] == ci and route_j[0] == cj:
                new_route = route_i + route_j
                can_merge = True
            elif route_i[0] == ci and route_j[-1] == cj:
                new_route = route_j + route_i
                can_merge = True
            elif route_i[-1] == ci and route_j[-1] == cj:
                new_route = route_i + list(reversed(route_j))
                can_merge = True
            elif route_i[0] == ci and route_j[0] == cj:
                new_route = list(reversed(route_i)) + route_j
                can_merge = True
            
            if can_merge:
                routes[ri] = new_route
                routes[rj] = None
                for c in new_route:
                    cust_route[c] = ri
        
        final_routes = [r for r in routes if r is not None]
        total_cost = 0.0
        result_routes = []
        for r in final_routes:
            full_route = [sat_id] + r + [sat_id]
            full_route, rc = two_opt(full_route)
            total_cost += rc
            result_routes.append(full_route)
        
        return result_routes, total_cost
    
    def evaluate_assignment(assignment):
        """Evaluate a group-to-satellite assignment. Returns (cost, details) or (inf, None) if infeasible."""
        # assignment: dict group_id -> sat_id
        
        # Collect customers per satellite
        sat_customers = {s['id']: [] for s in satellites}
        for g_id, s_id in assignment.items():
            for cid in group_by_id[g_id]['customer_ids']:
                sat_customers[s_id].append(cid)
        
        # Check satellite capacity constraints
        for s in satellites:
            sid = s['id']
            total_demand = sum(cust_by_id[cid]['demand'] for cid in sat_customers[sid])
            if total_demand > s['capacity']:
                return float('inf'), None
        
        # Build second echelon routes
        all_se_routes = []
        total_se_cost = 0.0
        total_handling_cost = 0.0
        sat_demand = {}
        
        for s in satellites:
            sid = s['id']
            custs = sat_customers[sid]
            if not custs:
                sat_demand[sid] = 0
                continue
            
            total_demand = sum(cust_by_id[cid]['demand'] for cid in custs)
            sat_demand[sid] = total_demand
            
            # Check max vehicles
            # Minimum number of vehicles needed
            min_vehicles = math.ceil(total_demand / Q2)
            if min_vehicles > s['max_vehicles']:
                return float('inf'), None
            
            routes, cost = build_second_echelon_routes(sid, custs)
            
            if len(routes) > s['max_vehicles']:
                return float('inf'), None
            
            all_se_routes.extend([(sid, r) for r in routes])
            total_se_cost += cost
            total_handling_cost += total_demand * s['handling_cost']
        
        # Check total vehicles
        if len(all_se_routes) > T2:
            return float('inf'), None
        
        # Build first echelon routes
        active_sats = [sid for sid in sat_demand if sat_demand[sid] > 0]
        
        if not active_sats:
            total_cost = 0.0
            return total_cost, {
                'first_echelon_routes': [],
                'second_echelon_routes': all_se_routes,
                'total_se_cost': total_se_cost,
                'total_handling_cost': total_handling_cost,
                'sat_demand': sat_demand
            }
        
        # Simple first-echelon routing: bin-pack satellites into vehicle tours
        fe_routes, fe_cost = build_first_echelon_routes(active_sats, sat_demand)
        
        if fe_routes is None:
            return float('inf'), None
        
        total_cost = fe_cost + total_handling_cost + total_se_cost
        
        return total_cost, {
            'first_echelon_routes': fe_routes,
            'second_echelon_routes': all_se_routes,
            'total_se_cost': total_se_cost,
            'total_handling_cost': total_handling_cost,
            'total_fe_cost': fe_cost,
            'sat_demand': sat_demand
        }
    
    def build_first_echelon_routes(active_sats, sat_demand):
        """Build first-echelon routes using bin-packing + TSP."""
        # Each vehicle can carry Q1 total
        # Each satellite has a throughput capacity - but we already checked that
        # We need to deliver sat_demand[sid] to each active satellite
        
        # For simplicity, we can have multiple vehicles visit the same satellite
        # But each vehicle visits each satellite at most once
        
        # Simple approach: assign satellites to vehicles greedily
        # First, check if we even need to split deliveries
        
        # Create delivery tasks: (satellite_id, amount)
        tasks = []
        for sid in active_sats:
            remaining = sat_demand[sid]
            sat_cap = sat_by_id[sid]['capacity']
            # The per-vehicle delivery to a satellite is limited by satellite throughput
            # Actually, throughput is total, not per-vehicle. Let me re-read.
            # "the quantity delivered by any single vehicle to any single satellite cannot exceed that satellite's throughput capacity"
            # Actually re-reading: "Each satellite has a throughput capacity limiting the total quantity of freight it can handle across all first-echelon deliveries."
            # And: "the quantity delivered by any single vehicle to any single satellite cannot exceed that satellite's throughput capacity"
            # These seem contradictory. Let me re-read more carefully.
            
            # "the quantity delivered by any single vehicle to any single satellite cannot exceed that satellite's throughput capacity"
            # So per-vehicle-per-satellite limit is the satellite's throughput capacity.
            # "Each satellite has a throughput capacity limiting the total quantity of freight it can handle across all first-echelon deliveries."
            # This means total across all vehicles is also limited by throughput.
            
            # So the total freight through a satellite <= satellite capacity (throughput)
            # AND each individual delivery <= satellite capacity
            # The first constraint is binding since total >= each individual.
            
            # We already checked total demand <= capacity above.
            # For splitting: we might need multiple vehicles if demand > Q1 per vehicle
            # But each vehicle can deliver at most min(Q1, sat_cap) to this satellite
            
            max_per_delivery = min(Q1, sat_cap)
            while remaining > 0:
                delivery = min(remaining, max_per_delivery)
                tasks.append((sid, delivery))
                remaining -= delivery
        
        # Now bin-pack tasks into vehicles (each vehicle has capacity Q1)
        # Use first-fit decreasing
        tasks.sort(key=lambda t: t[1], reverse=True)
        
        vehicles = []  # Each vehicle: list of (sat_id, amount)
        vehicle_loads = []
        
        for sid, amount in tasks:
            placed = False
            for vi in range(len(vehicles)):
                if vehicle_loads[vi] + amount <= Q1:
                    # Check that this vehicle doesn't already visit this satellite
                    if any(s == sid for s, a in vehicles[vi]):
                        continue
                    vehicles[vi].append((sid, amount))
                    vehicle_loads[vi] += amount
                    placed = True
                    break
            if not placed:
                if len(vehicles) >= T1:
                    return None, float('inf')
                vehicles.append([(sid, amount)])
                vehicle_loads.append(amount)
        
        if len(vehicles) > T1:
            return None, float('inf')
        
        # Build TSP route for each vehicle
        fe_routes = []
        total_fe_cost = 0.0
        
        for vi, deliveries in enumerate(vehicles):
            sat_ids = [sid for sid, amt in deliveries]
            route, cost = nn_tsp(depot['id'], sat_ids)
            route, cost = two_opt(route)
            delivery_dict = {}
            for sid, amt in deliveries:
                delivery_dict[sid] = delivery_dict.get(sid, 0) + amt
            fe_routes.append({
                'vehicle': vi,
                'route': route,
                'deliveries': delivery_dict
            })
            total_fe_cost += cost
        
        return fe_routes, total_fe_cost
    
    # Strategy: try different group-to-satellite assignments
    # For small instances, try all combinations; for larger ones, use heuristics
    
    sat_ids = [s['id'] for s in satellites]
    group_ids = [g['group_id'] for g in groups]
    
    best_cost = float('inf')
    best_solution = None
    best_assignment = None
    
    def check_time():
        return time.time() - start_time < time_limit - 1  # Leave 1 second buffer
    
    # For each group, compute the "best" satellites (closest to group centroid)
    group_centroids = {}
    for g in groups:
        cx = sum(cust_by_id[cid]['x'] for cid in g['customer_ids']) / len(g['customer_ids'])
        cy = sum(cust_by_id[cid]['y'] for cid in g['customer_ids']) / len(g['customer_ids'])
        group_centroids[g['group_id']] = (cx, cy)
    
    # Sort satellites for each group by distance to centroid + handling cost consideration
    group_sat_preferences = {}
    for g_id in group_ids:
        cx, cy = group_centroids[g_id]
        gd = group_demand[g_id]
        prefs = []
        for s in satellites:
            d = math.sqrt((s['x'] - cx)**2 + (s['y'] - cy)**2)
            # Also consider handling cost
            handling = s['handling_cost'] * gd
            # Also consider distance from depot to satellite for first echelon
            depot_dist = dist(depot['id'], s['id'])
            # Weighted score
            score = d * 2 + handling + depot_dist * 0.5
            if gd <= s['capacity']:
                prefs.append((score, s['id']))
        prefs.sort()
        group_sat_preferences[g_id] = [sid for _, sid in prefs]
    
    # Determine search strategy based on problem size
    total_combinations = 1
    for g_id in group_ids:
        total_combinations *= len(group_sat_preferences[g_id])
    
    def format_solution(assignment, details):
        fe_routes = []
        if details['first_echelon_routes']:
            if isinstance(details['first_echelon_routes'][0], dict):
                fe_routes = details['first_echelon_routes']
            else:
                # Should not happen with current code
                pass
        
        se_routes = []
        for sat_id, route in details['second_echelon_routes']:
            se_routes.append({
                'satellite': sat_id,
                'route': route
            })
        
        group_assignments = {str(g_id): s_id for g_id, s_id in assignment.items()}
        
        return {
            'objective_value': details.get('total_cost', best_cost),
            'first_echelon_routes': fe_routes,
            'second_echelon_routes': se_routes,
            'group_assignments': group_assignments
        }
    
    if total_combinations <= 50000 and num_groups <= 15:
        # Enumerate all feasible assignments
        from itertools import product
        
        candidate_lists = [group_sat_preferences[g_id] for g_id in group_ids]
        
        for combo in product(*candidate_lists):
            if not check_time():
                break
            
            assignment = {group_ids[i]: combo[i] for i in range(len(group_ids))}
            
            # Quick feasibility check: satellite capacity
            sat_load = {}
            feasible = True
            for g_id, s_id in assignment.items():
                sat_load[s_id] = sat_load.get(s_id, 0) + group_demand[g_id]
            for s_id, load in sat_load.items():
                if load > sat_by_id[s_id]['capacity']:
                    feasible = False
                    break
            if not feasible:
                continue
            
            cost, details = evaluate_assignment(assignment)
            if cost < best_cost:
                best_cost = cost
                best_assignment = assignment
                best_solution = details
                if logger:
                    logger.log(best_cost)
    else:
        # Use heuristic search
        # Start with greedy assignment (each group to nearest feasible satellite)
        
        def greedy_assignment():
            assignment = {}
            sat_load = {s['id']: 0 for s in satellites}
            sat_vehicles_used = {s['id']: 0 for s in satellites}
            
            # Sort groups by demand (largest first)
            sorted_groups = sorted(group_ids, key=lambda g: group_demand[g], reverse=True)
            
            for g_id in sorted_groups:
                gd = group_demand[g_id]
                best_sid = None
                best_score = float('inf')
                
                for sid in group_sat_preferences[g_id]:
                    s = sat_by_id[sid]
                    if sat_load[sid] + gd > s['capacity']:
                        continue
                    # Estimate vehicles needed
                    num_custs = len(group_by_id[g_id]['customer_ids'])
                    min_veh = math.ceil(gd / Q2)
                    if sat_vehicles_used[sid] + min_veh > s['max_vehicles']:
                        continue
                    
                    cx, cy = group_centroids[g_id]
                    d = math.sqrt((s['x'] - cx)**2 + (s['y'] - cy)**2)
                    handling = s['handling_cost'] * gd
                    score = d * 2 + handling
                    
                    if score < best_score:
                        best_score = score
                        best_sid = sid
                
                if best_sid is None:
                    # Infeasible, try any satellite
                    for sid in sat_ids:
                        s = sat_by_id[sid]
                        if sat_load[sid] + gd <= s['capacity']:
                            best_sid = sid
                            break
                
                if best_sid is None:
                    return None  # Infeasible
                
                assignment[g_id] = best_sid
                sat_load[best_sid] += gd
                min_veh = math.ceil(gd / Q2)
                sat_vehicles_used[best_sid] += min_veh
            
            return assignment
        
        assignment = greedy_assignment()
        if assignment is not None:
            cost, details = evaluate_assignment(assignment)
            if cost < best_cost:
                best_cost = cost
                best_assignment = assignment
                best_solution = details
                if logger:
                    logger.log(best_cost)
        
        # Local search: try moving groups to different satellites
        if best_assignment is not None:
            current_assignment = dict(best_assignment)
            current_cost = best_cost
            
            no_improve_count = 0
            max_no_improve = max(50, num_groups * len(sat_ids) * 2)
            
            while check_time() and no_improve_count < max_no_improve:
                # Random perturbation: move a random group to a random satellite
                g_id = random.choice(group_ids)
                old_sid = current_assignment[g_id]
                
                # Try all satellites for this group
                improved = False
                candidates = list(sat_ids)
                random.shuffle(candidates)
                
                for new_sid in candidates:
                    if new_sid == old_sid:
                        continue
                    
                    new_assignment = dict(current_assignment)
                    new_assignment[g_id] = new_sid
                    
                    # Quick feasibility
                    sat_load = {}
                    for gid, sid in new_assignment.items():
                        sat_load[sid] = sat_load.get(sid, 0) + group_demand[gid]
                    feasible = True
                    for sid, load in sat_load.items():
                        if load > sat_by_id[sid]['capacity']:
                            feasible = False
                            break
                    if not feasible:
                        continue
                    
                    new_cost, new_details = evaluate_assignment(new_assignment)
                    if new_cost < current_cost:
                        current_cost = new_cost
                        current_assignment = new_assignment
                        improved = True
                        
                        if new_cost < best_cost:
                            best_cost = new_cost
                            best_assignment = dict(new_assignment)
                            best_solution = new_details
                            if logger:
                                logger.log(best_cost)
                        break
                
                if not improved:
                    no_improve_count += 1
                else:
                    no_improve_count = 0
            
            # Try swap moves: swap satellite assignments of two groups
            no_improve_count = 0
            max_no_improve_swap = max(30, num_groups * 3)
            
            while check_time() and no_improve_count < max_no_improve_swap:
                g1, g2 = random.sample(group_ids, 2)
                if current_assignment[g1] == current_assignment[g2]:
                    no_improve_count += 1
                    continue
                
                new_assignment = dict(current_assignment)
                new_assignment[g1], new_assignment[g2] = new_assignment[g2], new_assignment[g1]
                
                # Quick feasibility
                sat_load = {}
                for gid, sid in new_assignment.items():
                    sat_load[sid] = sat_load.get(sid, 0) + group_demand[gid]
                feasible = True
                for sid, load in sat_load.items():
                    if load > sat_by_id[sid]['capacity']:
                        feasible = False
                        break
                if not feasible:
                    no_improve_count += 1
                    continue
                
                new_cost, new_details = evaluate_assignment(new_assignment)
                if new_cost < current_cost:
                    current_cost = new_cost
                    current_assignment = new_assignment
                    no_improve_count = 0
                    
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_assignment = dict(new_assignment)
                        best_solution = new_details
                        if logger:
                            logger.log(best_cost)
                else:
                    no_improve_count += 1
        
        # Multi-start with random assignments
        num_restarts = 0
        while check_time():
            num_restarts += 1
            # Random assignment respecting capacity
            assignment = {}
            sat_load = {s['id']: 0 for s in satellites}
            feasible = True
            
            sorted_groups = sorted(group_ids, key=lambda g: group_demand[g], reverse=True)
            
            for g_id in sorted_groups:
                gd = group_demand[g_id]
                candidates = []
                for sid in sat_ids:
                    if sat_load[sid] + gd <= sat_by_id[sid]['capacity']:
                        candidates.append(sid)
                
                if not candidates:
                    feasible = False
                    break
                
                # Weighted random choice based on preferences
                prefs = group_sat_preferences[g_id]
                # Filter by feasible candidates
                feas_prefs = [sid for sid in prefs if sid in candidates]
                if not feas_prefs:
                    feas_prefs = candidates
                
                # Pick with some randomness favoring better options
                if len(feas_prefs) > 1:
                    weights = [1.0 / (i + 1) for i in range(len(feas_prefs))]
                    total_w = sum(weights)
                    weights = [w / total_w for w in weights]
                    r = random.random()
                    cumsum = 0
                    chosen = feas_prefs[-1]
                    for i, w in enumerate(weights):
                        cumsum += w
                        if r <= cumsum:
                            chosen = feas_prefs[i]
                            break
                else:
                    chosen = feas_prefs[0]
                
                assignment[g_id] = chosen
                sat_load[chosen] += gd
            
            if not feasible:
                continue
            
            cost, details = evaluate_assignment(assignment)
            
            if cost < best_cost:
                best_cost = cost
                best_assignment = dict(assignment)
                best_solution = details
                if logger:
                    logger.log(best_cost)
            
            # Local search on this assignment
            current_assignment = dict(assignment)
            current_cost = cost
            
            no_improve = 0
            while check_time() and no_improve < num_groups * len(sat_ids):
                g_id = random.choice(group_ids)
                old_sid = current_assignment[g_id]
                
                candidates = list(sat_ids)
                random.shuffle(candidates)
                
                improved = False
                for new_sid in candidates:
                    if new_sid == old_sid:
                        continue
                    
                    new_assignment = dict(current_assignment)
                    new_assignment[g_id] = new_sid
                    
                    sat_load = {}
                    for gid, sid in new_assignment.items():
                        sat_load[sid] = sat_load.get(sid, 0) + group_demand[gid]
                    feas = True
                    for sid, load in sat_load.items():
                        if load > sat_by_id[sid]['capacity']:
                            feas = False
                            break
                    if not feas:
                        continue
                    
                    new_cost, new_details = evaluate_assignment(new_assignment)
                    if new_cost < current_cost:
                        current_cost = new_cost
                        current_assignment = new_assignment
                        improved = True
                        
                        if new_cost < best_cost:
                            best_cost = new_cost
                            best_assignment = dict(new_assignment)
                            best_solution = new_details
                            if logger:
                                logger.log(best_cost)
                        break
                
                if not improved:
                    no_improve += 1
                else:
                    no_improve = 0
    
    # Refine second echelon routes with more aggressive optimization
    if best_assignment is not None and best_solution is not None and check_time():
        # Re-optimize second echelon routes with or-opt and relocate moves
        sat_customers = {s['id']: [] for s in satellites}
        for g_id, s_id in best_assignment.items():
            for cid in group_by_id[g_id]['customer_ids']:
                sat_customers[s_id].append(cid)
        
        improved_se_routes = []
        total_se_cost = 0.0
        
        for s in satellites:
            sid = s['id']
            custs = sat_customers[sid]
            if not custs:
                continue
            
            routes, cost = build_second_echelon_routes(sid, custs)
            
            # Additional improvement: or-opt (move single customer between routes or within route)
            for attempt in range(5):
                if not check_time():
                    break
                improved = False
                for ri in range(len(routes)):
                    if not check_time():
                        break
                    route_i = routes[ri]
                    # route_i: [sat_id, c1, c2, ..., sat_id]
                    for pos_i in range(1, len(route_i) - 1):
                        cid = route_i[pos_i]
                        demand_cid = cust_by_id[cid]['demand']
                        
                        # Cost of removing cid from route_i
                        cost_remove = (dist(route_i[pos_i-1], route_i[pos_i+1]) 
                                      - dist(route_i[pos_i-1], cid) 
                                      - dist(cid, route_i[pos_i+1]))
                        
                        # Try inserting into other routes
                        best_insert_saving = 0
                        best_insert_route = -1
                        best_insert_pos = -1
                        
                        for rj in range(len(routes)):
                            if rj == ri:
                                # Try relocating within same route
                                for pos_j in range(1, len(route_i) - 1):
                                    if pos_j == pos_i or pos_j == pos_i - 1 or pos_j == pos_i + 1:
                                        continue
                                    # Would need to rebuild route, skip for now
                                continue
                            
                            route_j = routes[rj]
                            # Check capacity
                            route_j_demand = sum(cust_by_id[c]['demand'] for c in route_j[1:-1])
                            if route_j_demand + demand_cid > Q2:
                                continue
                            
                            for pos_j in range(1, len(route_j)):
                                cost_insert = (dist(route_j[pos_j-1], cid) + dist(cid, route_j[pos_j])
                                              - dist(route_j[pos_j-1], route_j[pos_j]))
                                total_saving = -(cost_remove + cost_insert)
                                if total_saving > best_insert_saving + 1e-10:
                                    best_insert_saving = total_saving
                                    best_insert_route = rj
                                    best_insert_pos = pos_j
                        
                        if best_insert_route >= 0:
                            # Perform the move
                            routes[ri] = route_i[:pos_i] + route_i[pos_i+1:]
                            rj = best_insert_route
                            routes[rj] = routes[rj][:best_insert_pos] + [cid] + routes[rj][best_insert_pos:]
                            
                            # Remove empty routes
                            if len(routes[ri]) == 2:  # Only sat_id ... sat_id
                                routes.pop(ri)
                                if best_insert_route > ri:
                                    best_insert_route -= 1
                            
                            improved = True
                            break
                    if improved:
                        break
                
                if not improved:
                    break
            
            # Final 2-opt on each route
            for i in range(len(routes)):
                routes[i], _ = two_opt(routes[i])
            
            for route in routes:
                rc = sum(dist(route[j], route[j+1]) for j in range(len(route)-1))
                total_se_cost += rc
                improved_se_routes.append((sid, route))
        
        # Recalculate total cost
        total_handling = sum(
            sum(cust_by_id[cid]['demand'] for cid in sat_customers[s['id']]) * s['handling_cost']
            for s in satellites
        )
        
        sat_demand_new = {}
        for s in satellites:
            sat_demand_new[s['id']] = sum(cust_by_id[cid]['demand'] for cid in sat_customers[s['id']])
        
        active_sats = [sid for sid in sat_demand_new if sat_demand_new[sid] > 0]
        fe_routes, fe_cost = build_first_echelon_routes(active_sats, sat_demand_new)
        
        if fe_routes is not None:
            new_total = fe_cost + total_handling + total_se_cost
            if new_total < best_cost:
                best_cost = new_total
                best_solution = {
                    'first_echelon_routes': fe_routes,
                    'second_echelon_routes': improved_se_routes,
                    'total_se_cost': total_se_cost,
                    'total_handling_cost': total_handling,
                    'total_fe_cost': fe_cost,
                    'sat_demand': sat_demand_new
                }
                if logger:
                    logger.log(best_cost)
    
    # Format output
    if best_assignment is None or best_solution is None:
        # Return empty solution
        return {
            'objective_value': 0.0,
            'first_echelon_routes': [],
            'second_echelon_routes': [],
            'group_assignments': {}
        }
    
    fe_routes_out = []
    if best_solution['first_echelon_routes']:
        for fr in best_solution['first_echelon_routes']:
            deliveries_str = {str(k): v for k, v in fr['deliveries'].items()}
            fe_routes_out.append({
                'vehicle': fr['vehicle'],
                'route': fr['route'],
                'deliveries': deliveries_str
            })
    
    se_routes_out = []
    for item in best_solution['second_echelon_routes']:
        if isinstance(item, tuple):
            sat_id, route = item
        else:
            sat_id = item['satellite']
            route = item['route']
        se_routes_out.append({
            'satellite': sat_id,
            'route': route
        })
    
    group_assignments_out = {str(g_id): s_id for g_id, s_id in best_assignment.items()}
    
    return {
        'objective_value': best_cost,
        'first_echelon_routes': fe_routes_out,
        'second_echelon_routes': se_routes_out,
        'group_assignments': group_assignments_out
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, default=300)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    solution = solve_instance(instance, args.time_limit, logger)
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == '__main__':
    main()
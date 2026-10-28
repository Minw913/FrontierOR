import argparse
import json
import math
import time
import random
import copy
from solution_logger import SolutionLogger

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    return parser.parse_args()

def load_instance(path):
    with open(path, 'r') as f:
        return json.load(f)

def compute_distance_matrix(depots, customers):
    """Compute distance matrix for all locations (depots first, then customers)."""
    locations = [(dep['x'], dep['y']) for dep in depots] + [(c['x'], c['y']) for c in customers]
    n = len(locations)
    dist = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            dx = locations[i][0] - locations[j][0]
            dy = locations[i][1] - locations[j][1]
            d = math.sqrt(dx*dx + dy*dy)
            dist[i][j] = d
            dist[j][i] = d
    return dist

def savings_algorithm(depot_idx, customer_indices, dist, demands, service_durations, 
                       vehicle_capacity, route_duration_limit, num_vehicles):
    """Clarke-Wright savings algorithm for CVRP."""
    if not customer_indices:
        return []
    
    # Compute savings
    savings = []
    for i_idx, ci in enumerate(customer_indices):
        for j_idx in range(i_idx+1, len(customer_indices)):
            cj = customer_indices[j_idx]
            s = dist[depot_idx][ci] + dist[depot_idx][cj] - dist[ci][cj]
            savings.append((s, ci, cj))
    
    savings.sort(key=lambda x: -x[0])
    
    # Initialize: each customer in its own route
    routes = {}
    route_of = {}
    route_id = 0
    for ci in customer_indices:
        routes[route_id] = {
            'customers': [ci],
            'demand': demands[ci],
            'duration': dist[depot_idx][ci] + service_durations[ci] + dist[ci][depot_idx]
        }
        route_of[ci] = route_id
        route_id += 1
    
    # Track which customers are at the start or end of their route
    def get_route_ends(rid):
        custs = routes[rid]['customers']
        return custs[0], custs[-1]
    
    for s, ci, cj in savings:
        if s <= 0:
            break
        ri = route_of.get(ci)
        rj = route_of.get(cj)
        if ri is None or rj is None or ri == rj:
            continue
        if ri not in routes or rj not in routes:
            continue
        
        # Check if ci is at end of ri and cj is at start of rj (or vice versa)
        custs_i = routes[ri]['customers']
        custs_j = routes[rj]['customers']
        
        merged = None
        # Try: ri...ci + cj...rj
        if custs_i[-1] == ci and custs_j[0] == cj:
            merged = custs_i + custs_j
        elif custs_i[0] == ci and custs_j[-1] == cj:
            merged = custs_j + custs_i
        elif custs_i[-1] == ci and custs_j[-1] == cj:
            merged = custs_i + list(reversed(custs_j))
        elif custs_i[0] == ci and custs_j[0] == cj:
            merged = list(reversed(custs_i)) + custs_j
        
        if merged is None:
            continue
        
        # Check capacity
        new_demand = routes[ri]['demand'] + routes[rj]['demand']
        if new_demand > vehicle_capacity:
            continue
        
        # Check duration
        new_duration = dist[depot_idx][merged[0]] + service_durations[merged[0]]
        for k in range(1, len(merged)):
            new_duration += dist[merged[k-1]][merged[k]] + service_durations[merged[k]]
        new_duration += dist[merged[-1]][depot_idx]
        
        if new_duration > route_duration_limit:
            continue
        
        # Merge
        new_rid = route_id
        route_id += 1
        routes[new_rid] = {
            'customers': merged,
            'demand': new_demand,
            'duration': new_duration
        }
        del routes[ri]
        del routes[rj]
        for c in merged:
            route_of[c] = new_rid
    
    result = [r['customers'] for r in routes.values()]
    
    if len(result) > num_vehicles:
        # Sort routes by demand, merge smallest
        result.sort(key=lambda r: sum(demands[c] for c in r))
        while len(result) > num_vehicles:
            # Try to merge the two smallest routes
            merged = False
            for i in range(len(result)):
                for j in range(i+1, len(result)):
                    combined = result[i] + result[j]
                    d = sum(demands[c] for c in combined)
                    if d > vehicle_capacity:
                        continue
                    dur = dist[depot_idx][combined[0]] + service_durations[combined[0]]
                    for k in range(1, len(combined)):
                        dur += dist[combined[k-1]][combined[k]] + service_durations[combined[k]]
                    dur += dist[combined[-1]][depot_idx]
                    if dur > route_duration_limit:
                        continue
                    result[i] = combined
                    result.pop(j)
                    merged = True
                    break
                if merged:
                    break
            if not merged:
                break
    
    return result

def route_cost(depot_idx, route, dist):
    if not route:
        return 0.0
    cost = dist[depot_idx][route[0]]
    for i in range(1, len(route)):
        cost += dist[route[i-1]][route[i]]
    cost += dist[route[-1]][depot_idx]
    return cost

def route_duration(depot_idx, route, dist, service_durations):
    if not route:
        return 0.0
    dur = dist[depot_idx][route[0]] + service_durations[route[0]]
    for i in range(1, len(route)):
        dur += dist[route[i-1]][route[i]] + service_durations[route[i]]
    dur += dist[route[-1]][depot_idx]
    return dur

def route_demand(route, demands):
    return sum(demands[c] for c in route)

def two_opt(depot_idx, route, dist, service_durations, route_duration_limit):
    """Improve a single route using 2-opt."""
    if len(route) < 2:
        return route
    improved = True
    best = list(route)
    while improved:
        improved = False
        best_cost = route_cost(depot_idx, best, dist)
        for i in range(len(best) - 1):
            for j in range(i + 1, len(best)):
                new_route = best[:i] + list(reversed(best[i:j+1])) + best[j+1:]
                dur = route_duration(depot_idx, new_route, dist, service_durations)
                if dur > route_duration_limit:
                    continue
                new_cost = route_cost(depot_idx, new_route, dist)
                if new_cost < best_cost - 1e-10:
                    best = new_route
                    best_cost = new_cost
                    improved = True
                    break
            if improved:
                break
    return best

def or_opt(depot_idx, route, dist, service_durations, route_duration_limit):
    """Or-opt: move segments of 1, 2, or 3 customers."""
    if len(route) < 2:
        return route
    improved = True
    best = list(route)
    while improved:
        improved = False
        best_cost = route_cost(depot_idx, best, dist)
        for seg_len in [1, 2, 3]:
            for i in range(len(best) - seg_len + 1):
                segment = best[i:i+seg_len]
                remaining = best[:i] + best[i+seg_len:]
                for j in range(len(remaining) + 1):
                    new_route = remaining[:j] + segment + remaining[j:]
                    if new_route == best:
                        continue
                    dur = route_duration(depot_idx, new_route, dist, service_durations)
                    if dur > route_duration_limit:
                        continue
                    new_cost = route_cost(depot_idx, new_route, dist)
                    if new_cost < best_cost - 1e-10:
                        best = new_route
                        best_cost = new_cost
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    return best

def improve_route(depot_idx, route, dist, service_durations, route_duration_limit):
    route = two_opt(depot_idx, route, dist, service_durations, route_duration_limit)
    route = or_opt(depot_idx, route, dist, service_durations, route_duration_limit)
    return route

def inter_route_relocate(depot_idx, routes, dist, demands, service_durations,
                          vehicle_capacity, route_duration_limit, time_left):
    """Try relocating a customer from one route to another."""
    start = time.time()
    improved = True
    while improved:
        if time.time() - start > time_left:
            break
        improved = False
        best_saving = 0
        best_move = None
        for r1_idx in range(len(routes)):
            for pos1, cust in enumerate(routes[r1_idx]):
                # Cost of removing cust from r1
                r1 = routes[r1_idx]
                if len(r1) == 1:
                    remove_cost = dist[depot_idx][cust] + dist[cust][depot_idx]
                else:
                    prev1 = depot_idx if pos1 == 0 else r1[pos1-1]
                    next1 = depot_idx if pos1 == len(r1)-1 else r1[pos1+1]
                    old_cost = dist[prev1][cust] + dist[cust][next1]
                    new_cost = dist[prev1][next1]
                    remove_saving = old_cost - new_cost
                
                for r2_idx in range(len(routes)):
                    if r1_idx == r2_idx:
                        continue
                    r2 = routes[r2_idx]
                    # Check capacity
                    if route_demand(r2, demands) + demands[cust] > vehicle_capacity:
                        continue
                    
                    # Find best insertion position
                    for pos2 in range(len(r2) + 1):
                        prev2 = depot_idx if pos2 == 0 else r2[pos2-1]
                        next2 = depot_idx if pos2 == len(r2) else r2[pos2]
                        insert_cost = dist[prev2][cust] + dist[cust][next2] - dist[prev2][next2]
                        
                        if len(r1) == 1:
                            saving = remove_cost - insert_cost  # removing whole route saves its cost
                            saving = (dist[depot_idx][cust] + dist[cust][depot_idx]) - insert_cost
                        else:
                            saving = remove_saving - insert_cost
                        
                        if saving > best_saving + 1e-10:
                            # Check duration feasibility
                            new_r2 = r2[:pos2] + [cust] + r2[pos2:]
                            dur = route_duration(depot_idx, new_r2, dist, service_durations)
                            if dur > route_duration_limit:
                                continue
                            best_saving = saving
                            best_move = (r1_idx, pos1, r2_idx, pos2)
        
        if best_move:
            r1_idx, pos1, r2_idx, pos2 = best_move
            cust = routes[r1_idx][pos1]
            routes[r1_idx] = routes[r1_idx][:pos1] + routes[r1_idx][pos1+1:]
            # Adjust pos2 if needed (shouldn't matter since different routes)
            routes[r2_idx] = routes[r2_idx][:pos2] + [cust] + routes[r2_idx][pos2:]
            # Remove empty routes
            routes = [r for r in routes if len(r) > 0]
            improved = True
    
    return routes

def inter_route_swap(depot_idx, routes, dist, demands, service_durations,
                      vehicle_capacity, route_duration_limit, time_left):
    """Try swapping customers between routes."""
    start = time.time()
    improved = True
    while improved:
        if time.time() - start > time_left:
            break
        improved = False
        best_saving = 0
        best_move = None
        for r1_idx in range(len(routes)):
            for r2_idx in range(r1_idx+1, len(routes)):
                r1 = routes[r1_idx]
                r2 = routes[r2_idx]
                for pos1 in range(len(r1)):
                    for pos2 in range(len(r2)):
                        c1 = r1[pos1]
                        c2 = r2[pos2]
                        # Check capacity
                        new_d1 = route_demand(r1, demands) - demands[c1] + demands[c2]
                        new_d2 = route_demand(r2, demands) - demands[c2] + demands[c1]
                        if new_d1 > vehicle_capacity or new_d2 > vehicle_capacity:
                            continue
                        
                        # Compute cost change
                        prev1 = depot_idx if pos1 == 0 else r1[pos1-1]
                        next1 = depot_idx if pos1 == len(r1)-1 else r1[pos1+1]
                        prev2 = depot_idx if pos2 == 0 else r2[pos2-1]
                        next2 = depot_idx if pos2 == len(r2)-1 else r2[pos2+1]
                        
                        old1 = dist[prev1][c1] + dist[c1][next1]
                        new1 = dist[prev1][c2] + dist[c2][next1]
                        old2 = dist[prev2][c2] + dist[c2][next2]
                        new2 = dist[prev2][c1] + dist[c1][next2]
                        
                        saving = (old1 + old2) - (new1 + new2)
                        if saving > best_saving + 1e-10:
                            # Check duration feasibility
                            new_r1 = r1[:pos1] + [c2] + r1[pos1+1:]
                            new_r2 = r2[:pos2] + [c1] + r2[pos2+1:]
                            dur1 = route_duration(depot_idx, new_r1, dist, service_durations)
                            dur2 = route_duration(depot_idx, new_r2, dist, service_durations)
                            if dur1 > route_duration_limit or dur2 > route_duration_limit:
                                continue
                            best_saving = saving
                            best_move = (r1_idx, pos1, r2_idx, pos2)
        
        if best_move:
            r1_idx, pos1, r2_idx, pos2 = best_move
            c1 = routes[r1_idx][pos1]
            c2 = routes[r2_idx][pos2]
            routes[r1_idx][pos1] = c2
            routes[r2_idx][pos2] = c1
            improved = True
    
    return routes

def solve(instance, time_limit, logger):
    start_time = time.time()
    random.seed(0)
    
    n = instance['n']
    d = instance['d']
    t = instance['t']
    m = instance['m']
    vehicle_capacity = instance['vehicle_capacity']
    route_duration_limit = instance['route_duration_limit']
    depots = instance['depots']
    customers = instance['customers']
    
    # Compute distance matrix
    # Indices: 0..d-1 are depots, d..d+n-1 are customers
    dist = compute_distance_matrix(depots, customers)
    
    # Customer index in dist matrix
    def cust_dist_idx(cid):
        return d + cid
    
    # Build demands and service_durations arrays indexed by dist matrix index
    demands_arr = [0] * (d + n)
    service_arr = [0] * (d + n)
    for c in customers:
        idx = cust_dist_idx(c['id'])
        demands_arr[idx] = c['demand']
        service_arr[idx] = c['service_duration']
    
    # Phase 1: Assign customers to depots and patterns
    # Use nearest depot heuristic, picking first feasible pattern
    assignments = {}  # customer_id -> {'depot': depot_id, 'pattern': pattern}
    
    def assign_nearest_depot():
        """Assign each customer to nearest depot with first feasible pattern."""
        for c in customers:
            cid = c['id']
            cidx = cust_dist_idx(cid)
            # Find nearest depot
            best_dep = None
            best_dist_val = float('inf')
            for dep in depots:
                did = dep['id']
                d_val = dist[did][cidx]
                if d_val < best_dist_val:
                    best_dist_val = d_val
                    best_dep = did
            # Pick first pattern
            pattern = c['pattern_list'][0]
            assignments[cid] = {'depot': best_dep, 'pattern': pattern}
    
    assign_nearest_depot()
    
    # Phase 2: Build routes for each depot-period pair
    def build_all_routes():
        all_routes = {}  # (depot_id, period) -> list of routes
        total_cost = 0.0
        
        for dep in depots:
            did = dep['id']
            for period in range(1, t + 1):
                # Get customers assigned to this depot on this period
                cust_list = []
                for cid, asg in assignments.items():
                    if asg['depot'] == did and period in asg['pattern']:
                        cust_list.append(cust_dist_idx(cid))
                
                if not cust_list:
                    all_routes[(did, period)] = []
                    continue
                
                # Build routes using savings algorithm
                routes = savings_algorithm(
                    did, cust_list, dist, demands_arr, service_arr,
                    vehicle_capacity, route_duration_limit, m
                )
                
                all_routes[(did, period)] = routes
                for r in routes:
                    total_cost += route_cost(did, r, dist)
        
        return all_routes, total_cost
    
    all_routes, total_cost = build_all_routes()
    
    # Improve routes with local search
    def improve_all_routes(all_routes, time_budget):
        per_route_budget = 0.5  # seconds per route pair for inter-route ops
        total_cost = 0.0
        improved_routes = {}
        
        for (did, period), routes in all_routes.items():
            if time.time() - start_time > time_limit - 2:
                improved_routes[(did, period)] = routes
                for r in routes:
                    total_cost += route_cost(did, r, dist)
                continue
            
            # Intra-route improvement
            new_routes = []
            for r in routes:
                r = improve_route(did, r, dist, service_arr, route_duration_limit)
                new_routes.append(r)
            
            # Inter-route improvement
            if len(new_routes) > 1:
                time_left = min(per_route_budget, time_limit - (time.time() - start_time) - 2)
                if time_left > 0:
                    new_routes = inter_route_relocate(
                        did, new_routes, dist, demands_arr, service_arr,
                        vehicle_capacity, route_duration_limit, time_left
                    )
                
                time_left = min(per_route_budget, time_limit - (time.time() - start_time) - 2)
                if time_left > 0:
                    new_routes = inter_route_swap(
                        did, new_routes, dist, demands_arr, service_arr,
                        vehicle_capacity, route_duration_limit, time_left
                    )
                
                # Re-do intra-route after inter-route changes
                final_routes = []
                for r in new_routes:
                    if r:
                        r = improve_route(did, r, dist, service_arr, route_duration_limit)
                        final_routes.append(r)
                new_routes = final_routes
            
            improved_routes[(did, period)] = new_routes
            for r in new_routes:
                total_cost += route_cost(did, r, dist)
        
        return improved_routes, total_cost
    
    time_budget = (time_limit - (time.time() - start_time)) * 0.4
    all_routes, total_cost = improve_all_routes(all_routes, time_budget)
    
    def build_solution(all_routes, total_cost):
        routes_output = []
        vehicle_counter = {}  # (depot, period) -> next vehicle id
        
        for (did, period), routes in all_routes.items():
            for r_idx, route in enumerate(routes):
                if not route:
                    continue
                veh = r_idx + 1
                # Convert dist matrix indices back to customer ids
                cust_ids = [c - d for c in route]
                routes_output.append({
                    'depot': did,
                    'period': period,
                    'vehicle': veh,
                    'customers': cust_ids
                })
        
        assignments_output = {}
        for cid, asg in assignments.items():
            assignments_output[str(cid)] = {
                'depot': asg['depot'],
                'pattern': asg['pattern']
            }
        
        return {
            'objective_value': total_cost,
            'routes': routes_output,
            'assignments': assignments_output
        }
    
    solution = build_solution(all_routes, total_cost)
    best_cost = total_cost
    best_solution = solution
    
    if logger:
        logger.log_solution(best_cost, best_solution)
    
    # Phase 3: Try different depot-pattern assignments
    # For each customer, try reassigning to different depots/patterns
    remaining_time = time_limit - (time.time() - start_time)
    
    if remaining_time > 5 and n <= 200:
        # Try reassigning customers to better depots
        customer_list = list(range(n))
        random.shuffle(customer_list)
        
        iteration = 0
        max_no_improve = max(n * 3, 50)
        no_improve_count = 0
        
        while time.time() - start_time < time_limit - 5 and no_improve_count < max_no_improve:
            iteration += 1
            # Pick a random customer
            cid = customer_list[iteration % n]
            c = customers[cid]
            current_depot = assignments[cid]['depot']
            current_pattern = assignments[cid]['pattern']
            
            # Try each depot and pattern
            best_local_cost = best_cost
            best_local_depot = current_depot
            best_local_pattern = current_pattern
            found_better = False
            
            candidates = []
            for dep in depots:
                for pat in c['pattern_list']:
                    if dep['id'] == current_depot and pat == current_pattern:
                        continue
                    candidates.append((dep['id'], pat))
            
            if not candidates:
                no_improve_count += 1
                continue
            
            # Only try a subset if too many
            if len(candidates) > 10:
                random.shuffle(candidates)
                candidates = candidates[:10]
            
            for new_depot, new_pattern in candidates:
                if time.time() - start_time > time_limit - 3:
                    break
                
                # Temporarily reassign
                old_asg = assignments[cid].copy()
                assignments[cid] = {'depot': new_depot, 'pattern': new_pattern}
                
                # Recompute affected routes
                affected_depot_periods = set()
                for p in current_pattern:
                    affected_depot_periods.add((current_depot, p))
                for p in new_pattern:
                    affected_depot_periods.add((new_depot, p))
                
                # Compute cost change
                old_affected_cost = 0
                for dp in affected_depot_periods:
                    if dp in all_routes:
                        for r in all_routes[dp]:
                            old_affected_cost += route_cost(dp[0], r, dist)
                
                # Rebuild affected routes
                new_affected_routes = {}
                new_affected_cost = 0
                feasible = True
                
                for dp in affected_depot_periods:
                    did_a, period_a = dp
                    cust_list = []
                    for cid2, asg in assignments.items():
                        if asg['depot'] == did_a and period_a in asg['pattern']:
                            cust_list.append(cust_dist_idx(cid2))
                    
                    if not cust_list:
                        new_affected_routes[dp] = []
                        continue
                    
                    routes = savings_algorithm(
                        did_a, cust_list, dist, demands_arr, service_arr,
                        vehicle_capacity, route_duration_limit, m
                    )
                    
                    if len(routes) > m:
                        feasible = False
                        break
                    
                    # Quick improve
                    improved_r = []
                    for r in routes:
                        r = two_opt(did_a, r, dist, service_arr, route_duration_limit)
                        improved_r.append(r)
                    
                    new_affected_routes[dp] = improved_r
                    for r in improved_r:
                        new_affected_cost += route_cost(dp[0], r, dist)
                
                if not feasible:
                    assignments[cid] = old_asg
                    continue
                
                new_total = best_cost - old_affected_cost + new_affected_cost
                
                if new_total < best_cost - 1e-6:
                    # Accept
                    for dp, routes in new_affected_routes.items():
                        all_routes[dp] = routes
                    best_cost = new_total
                    found_better = True
                    best_local_depot = new_depot
                    best_local_pattern = new_pattern
                    
                    # Build and log solution
                    solution = build_solution(all_routes, best_cost)
                    best_solution = solution
                    if logger:
                        logger.log_solution(best_cost, best_solution)
                    break
                else:
                    assignments[cid] = old_asg
            
            if found_better:
                no_improve_count = 0
            else:
                no_improve_count += 1
    
    # Final improvement pass on all routes
    remaining = time_limit - (time.time() - start_time)
    if remaining > 3:
        total_cost = 0.0
        for (did, period), routes in all_routes.items():
            if time.time() - start_time > time_limit - 1:
                for r in routes:
                    total_cost += route_cost(did, r, dist)
                continue
            
            new_routes = []
            for r in routes:
                r = improve_route(did, r, dist, service_arr, route_duration_limit)
                new_routes.append(r)
            
            if len(new_routes) > 1:
                time_left = min(1.0, time_limit - (time.time() - start_time) - 1)
                if time_left > 0:
                    new_routes = inter_route_relocate(
                        did, new_routes, dist, demands_arr, service_arr,
                        vehicle_capacity, route_duration_limit, time_left
                    )
                    final_routes = []
                    for r in new_routes:
                        if r:
                            r = improve_route(did, r, dist, service_arr, route_duration_limit)
                            final_routes.append(r)
                    new_routes = final_routes
            
            all_routes[(did, period)] = new_routes
            for r in new_routes:
                total_cost += route_cost(did, r, dist)
        
        best_cost = total_cost
        best_solution = build_solution(all_routes, best_cost)
        if logger:
            logger.log_solution(best_cost, best_solution)
    
    return best_solution

def main():
    args = parse_args()
    instance = load_instance(args.instance_path)
    
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    solution = solve(instance, args.time_limit, logger)
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == '__main__':
    main()
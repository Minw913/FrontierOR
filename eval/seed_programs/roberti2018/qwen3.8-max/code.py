import json
import argparse
import math
import time
import sys
from collections import defaultdict
from solution_logger import SolutionLogger


def euclidean_dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def compute_distance_matrix(nodes):
    """Compute pairwise Euclidean distances between all nodes."""
    n = len(nodes)
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = euclidean_dist(nodes[i], nodes[j])
            dist[i][j] = d
            dist[j][i] = d
    return dist


def clarke_wright_savings(depot_idx, customer_indices, dist, capacity, max_routes=None):
    """
    Clarke-Wright savings algorithm for CVRP.
    Returns list of routes, each route is a list of node indices (without depot).
    """
    if not customer_indices:
        return []
    
    # Initial: each customer on its own route
    routes = [[c] for c in customer_indices]
    route_demand = {i: 0 for i in range(len(routes))}
    
    # Compute demands (we'll pass them externally)
    # For now, assume each customer has demand accessible
    
    # Compute savings
    savings = []
    for i in range(len(customer_indices)):
        for j in range(i + 1, len(customer_indices)):
            ci = customer_indices[i]
            cj = customer_indices[j]
            s = dist[depot_idx][ci] + dist[depot_idx][cj] - dist[ci][cj]
            savings.append((s, ci, cj))
    
    savings.sort(key=lambda x: -x[0])
    
    # Track which route each customer belongs to
    cust_route = {}
    for idx, c in enumerate(customer_indices):
        cust_route[c] = idx
    
    route_list = [[c] for c in customer_indices]
    route_active = [True] * len(customer_indices)
    
    # We need demand info - pass as parameter
    return route_list


def solve_vrp_savings(depot_idx, customers, dist, demands, capacity, max_routes=None):
    """
    Solve CVRP using Clarke-Wright savings algorithm.
    customers: list of node indices
    demands: dict node_idx -> demand
    Returns list of routes (each route is list of node indices without depot).
    """
    if not customers:
        return []
    
    # Initial: each customer on its own route
    routes = [[c] for c in customers]
    route_demand = [demands.get(c, 0) for c in customers]
    
    # Map customer to route index
    cust_to_route = {c: i for i, c in enumerate(customers)}
    active = [True] * len(customers)
    
    # Compute savings
    savings_list = []
    for i in range(len(customers)):
        for j in range(i + 1, len(customers)):
            ci = customers[i]
            cj = customers[j]
            s = dist[depot_idx][ci] + dist[depot_idx][cj] - dist[ci][cj]
            if s > 0:
                savings_list.append((s, ci, cj))
    
    savings_list.sort(key=lambda x: -x[0])
    
    for s, ci, cj in savings_list:
        ri = cust_to_route[ci]
        rj = cust_to_route[cj]
        
        if ri == rj:
            continue
        if not active[ri] or not active[rj]:
            continue
        
        # Check if ci is at end of its route and cj is at start of its route (or vice versa)
        route_i = routes[ri]
        route_j = routes[rj]
        
        # Check capacity
        if route_demand[ri] + route_demand[rj] > capacity:
            continue
        
        # Check max routes constraint
        if max_routes is not None:
            current_routes = sum(1 for a in active if a)
            if current_routes - 1 < 1:
                continue
        
        # Try merging: ci at end of route_i, cj at start of route_j
        merged = None
        if route_i[-1] == ci and route_j[0] == cj:
            merged = route_i + route_j
        elif route_j[-1] == cj and route_i[0] == ci:
            merged = route_j + route_i
        elif route_i[-1] == ci and route_j[-1] == cj:
            merged = route_i + route_j[::-1]
        elif route_i[0] == ci and route_j[0] == cj:
            merged = route_i[::-1] + route_j
        else:
            continue
        
        # Perform merge
        routes[ri] = merged
        route_demand[ri] = route_demand[ri] + route_demand[rj]
        active[rj] = False
        
        # Update cust_to_route
        for c in merged:
            cust_to_route[c] = ri
    
    result = [routes[i] for i in range(len(routes)) if active[i]]
    return result


def two_opt_route(route, depot_idx, dist):
    """Apply 2-opt improvement to a single route."""
    if len(route) <= 2:
        return route
    
    improved = True
    best_route = route[:]
    
    while improved:
        improved = False
        for i in range(len(best_route) - 1):
            for j in range(i + 1, len(best_route)):
                # Try reversing segment between i and j
                new_route = best_route[:i] + best_route[i:j+1][::-1] + best_route[j+1:]
                
                # Compute cost difference
                # Original edges: (prev_i, best_route[i]) and (best_route[j], next_j)
                # New edges: (prev_i, best_route[j]) and (best_route[i], next_j)
                prev_i = depot_idx if i == 0 else best_route[i-1]
                next_j = depot_idx if j == len(best_route)-1 else best_route[j+1]
                
                old_cost = dist[prev_i][best_route[i]] + dist[best_route[j]][next_j]
                new_cost = dist[prev_i][best_route[j]] + dist[best_route[i]][next_j]
                
                if new_cost < old_cost - 1e-10:
                    best_route = new_route
                    improved = True
                    break
            if improved:
                break
    
    return best_route


def or_opt_routes(routes, depot_idx, dist, demands, capacity):
    """Apply or-opt improvement: move segments of 1-3 customers between/within routes."""
    improved = True
    while improved:
        improved = False
        for seg_len in [1, 2, 3]:
            for r1 in range(len(routes)):
                for i in range(len(routes[r1]) - seg_len + 1):
                    segment = routes[r1][i:i+seg_len]
                    seg_demand = sum(demands.get(c, 0) for c in segment)
                    
                    # Remove segment from r1
                    new_r1 = routes[r1][:i] + routes[r1][i+seg_len:]
                    
                    # Try inserting into r2
                    for r2 in range(len(routes)):
                        if r2 == r1:
                            target_route = new_r1
                        else:
                            target_route = routes[r2]
                        
                        # Check capacity
                        target_demand = sum(demands.get(c, 0) for c in target_route)
                        if target_demand + seg_demand > capacity:
                            continue
                        
                        # Try all insertion positions
                        best_pos = -1
                        best_saving = 0
                        
                        for pos in range(len(target_route) + 1):
                            # Cost of inserting segment at position pos
                            prev_node = depot_idx if pos == 0 else target_route[pos-1]
                            next_node = depot_idx if pos == len(target_route) else target_route[pos]
                            
                            # Cost without segment in r1
                            if i == 0:
                                prev_r1 = depot_idx
                            else:
                                prev_r1 = routes[r1][i-1]
                            if i + seg_len >= len(routes[r1]):
                                next_r1 = depot_idx
                            else:
                                next_r1 = routes[r1][i+seg_len]
                            
                            old_cost_r1 = dist[prev_r1][segment[0]] + dist[segment[-1]][next_r1]
                            new_cost_r1 = dist[prev_r1][next_r1] if prev_r1 != next_r1 or True else 0
                            # Actually: removing segment saves dist[prev_r1][segment[0]] + dist[segment[-1]][next_r1]
                            # and adds dist[prev_r1][next_r1]
                            saving_r1 = dist[prev_r1][segment[0]] + dist[segment[-1]][next_r1] - dist[prev_r1][next_r1]
                            
                            # Cost of inserting segment in r2
                            old_cost_r2 = dist[prev_node][next_node]
                            new_cost_r2 = dist[prev_node][segment[0]] + dist[segment[-1]][next_node]
                            for k in range(len(segment) - 1):
                                new_cost_r2 += dist[segment[k]][segment[k+1]]
                            
                            saving_r2 = old_cost_r2 - new_cost_r2
                            
                            total_saving = saving_r1 + saving_r2
                            if total_saving > best_saving + 1e-10:
                                best_saving = total_saving
                                best_pos = pos
                                best_r2 = r2
                        
                        if best_pos >= 0 and best_saving > 1e-10:
                            # Apply the move
                            if best_r2 == r1:
                                new_route = new_r1[:best_pos] + segment + new_r1[best_pos:]
                                routes[r1] = new_route
                            else:
                                routes[r1] = new_r1
                                routes[best_r2] = routes[best_r2][:best_pos] + segment + routes[best_r2][best_pos:]
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    
    # Remove empty routes
    routes = [r for r in routes if len(r) > 0]
    return routes


def route_cost(route, depot_idx, dist):
    """Compute total distance of a route starting and ending at depot."""
    if not route:
        return 0.0
    cost = dist[depot_idx][route[0]]
    for i in range(len(route) - 1):
        cost += dist[route[i]][route[i+1]]
    cost += dist[route[-1]][depot_idx]
    return cost


def solve_second_echelon(sat_idx, customers, dist, demands, Q2, max_vehicles):
    """
    Solve second echelon VRP for a given satellite.
    Returns list of routes (each is list of customer node indices).
    """
    if not customers:
        return []
    
    routes = solve_vrp_savings(sat_idx, customers, dist, demands, Q2, max_vehicles)
    
    # Apply 2-opt to each route
    for i in range(len(routes)):
        routes[i] = two_opt_route(routes[i], sat_idx, dist)
    
    # Apply or-opt
    routes = or_opt_routes(routes, sat_idx, dist, demands, Q2)
    
    # Apply 2-opt again after or-opt
    for i in range(len(routes)):
        routes[i] = two_opt_route(routes[i], sat_idx, dist)
    
    return routes


def solve_first_echelon(depot_idx, used_satellites, sat_demands, dist, Q1, T1):
    """
    Solve first echelon CVRP.
    used_satellites: list of satellite node indices that need service
    sat_demands: dict sat_idx -> demand
    Returns list of routes (each is list of satellite node indices).
    """
    if not used_satellites:
        return []
    
    routes = solve_vrp_savings(depot_idx, used_satellites, dist, sat_demands, Q1, T1)
    
    # Apply 2-opt
    for i in range(len(routes)):
        routes[i] = two_opt_route(routes[i], depot_idx, dist)
    
    return routes


def compute_objective(first_routes, second_routes_by_sat, group_assignment, 
                      dist, depot_idx, sat_handling_cost, sat_demands_total):
    """Compute total objective value."""
    # First echelon travel cost
    fe_cost = 0.0
    for route in first_routes:
        fe_cost += route_cost(route, depot_idx, dist)
    
    # Handling cost
    handling_cost = 0.0
    for sat_idx, total_demand in sat_demands_total.items():
        if total_demand > 0:
            handling_cost += sat_handling_cost[sat_idx] * total_demand
    
    # Second echelon travel cost
    se_cost = 0.0
    for sat_idx, routes in second_routes_by_sat.items():
        for route in routes:
            se_cost += route_cost(route, sat_idx, dist)
    
    return fe_cost + handling_cost + se_cost


def build_solution_dict(instance, first_routes, second_routes_by_sat, group_assignment,
                        node_ids, depot_id, sat_ids, cust_ids, total_cost):
    """Build the solution dictionary in the required output format."""
    
    # First echelon routes
    fe_routes_out = []
    for v_idx, route in enumerate(first_routes):
        route_ids = [depot_id] + [node_ids[n] for n in route] + [depot_id]
        deliveries = {}
        for sat_node in route:
            sat_id = node_ids[sat_node]
            # Delivery amount: we need to determine how much each vehicle delivers
            # For simplicity, we'll compute this based on the total demand at each satellite
            # and distribute among vehicles visiting that satellite
            deliveries[str(sat_id)] = 0  # Will be filled below
        fe_routes_out.append({
            "vehicle": v_idx + 1,
            "route": route_ids,
            "deliveries": deliveries
        })
    
    # Compute deliveries: for each satellite, total demand must be delivered
    # Distribute among first-echelon vehicles visiting that satellite
    sat_total_demand = defaultdict(float)
    for sat_idx, demand in sat_demands_total_global.items():
        sat_total_demand[sat_idx] = demand
    
    # Track which vehicles visit which satellites
    sat_visitors = defaultdict(list)
    for v_idx, route in enumerate(first_routes):
        for sat_node in route:
            sat_visitors[sat_node].append(v_idx)
    
    # Assign deliveries
    for v_idx, route in enumerate(first_routes):
        for sat_node in route:
            sat_id = node_ids[sat_node]
            visitors = sat_visitors[sat_node]
            # Simple: distribute evenly (or give all to first visitor)
            # Better: give proportional share
            share = sat_total_demand[sat_node] / len(visitors) if visitors else 0
            fe_routes_out[v_idx]["deliveries"][str(sat_id)] = round(share, 6)
    
    # Second echelon routes
    se_routes_out = []
    for sat_idx, routes in second_routes_by_sat.items():
        sat_id = node_ids[sat_idx]
        for route in routes:
            route_ids = [sat_id] + [node_ids[n] for n in route] + [sat_id]
            se_routes_out.append({
                "satellite": sat_id,
                "route": route_ids
            })
    
    # Group assignments
    ga_out = {}
    for g_id, sat_idx in group_assignment.items():
        ga_out[str(g_id)] = node_ids[sat_idx]
    
    return {
        "objective_value": total_cost,
        "first_echelon_routes": fe_routes_out,
        "second_echelon_routes": se_routes_out,
        "group_assignments": ga_out
    }


# Global variable for delivery computation
sat_demands_total_global = {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    start_time = time.time()
    
    # Parse instance
    num_satellites = instance['num_satellites']
    num_customers = instance['num_customers']
    num_groups = instance['num_groups']
    Q1 = instance['Q1']
    Q2 = instance['Q2']
    T1 = instance['T1']
    T2 = instance['T2']
    
    depot_info = instance['depot']
    depot_id = depot_info['id']
    depot_x, depot_y = depot_info['x'], depot_info['y']
    
    satellites_info = instance['satellites']
    customers_info = instance['customers']
    groups_info = instance['groups']
    
    # Build node list: index 0 = depot, 1..num_satellites = satellites, then customers
    # Actually, let's use a mapping approach
    nodes = []  # (x, y)
    node_ids = []  # original id for each internal index
    
    # Index 0: depot
    nodes.append((depot_x, depot_y))
    node_ids.append(depot_id)
    depot_idx = 0
    
    # Satellites: indices 1 to num_satellites
    sat_indices = []
    sat_capacity = {}
    sat_max_vehicles = {}
    sat_handling = {}
    sat_id_to_idx = {}
    
    for i, s in enumerate(satellites_info):
        idx = len(nodes)
        nodes.append((s['x'], s['y']))
        node_ids.append(s['id'])
        sat_indices.append(idx)
        sat_capacity[idx] = s['capacity']
        sat_max_vehicles[idx] = s['max_vehicles']
        sat_handling[idx] = s['handling_cost']
        sat_id_to_idx[s['id']] = idx
    
    # Customers
    cust_indices = []
    cust_demand = {}
    cust_group = {}
    cust_id_to_idx = {}
    
    for c in customers_info:
        idx = len(nodes)
        nodes.append((c['x'], c['y']))
        node_ids.append(c['id'])
        cust_indices.append(idx)
        cust_demand[idx] = c['demand']
        cust_group[idx] = c['group']
        cust_id_to_idx[c['id']] = idx
    
    # Groups
    group_customers = {}  # group_id -> list of customer internal indices
    for g in groups_info:
        g_id = g['group_id']
        group_customers[g_id] = [cust_id_to_idx[cid] for cid in g['customer_ids']]
    
    group_ids = list(group_customers.keys())
    
    # Compute distance matrix
    dist = compute_distance_matrix(nodes)
    
    # Compute group demands
    group_demand = {}
    for g_id, custs in group_customers.items():
        group_demand[g_id] = sum(cust_demand[c] for c in custs)
    
    # --- Initial greedy assignment ---
    # Assign each group to nearest feasible satellite
    
    # Sort groups by demand (descending) for greedy assignment
    sorted_groups = sorted(group_ids, key=lambda g: -group_demand[g])
    
    group_assignment = {}  # group_id -> sat_idx
    sat_load = {s: 0 for s in sat_indices}  # current load at each satellite
    sat_vehicle_count = {s: 0 for s in sat_indices}
    
    for g_id in sorted_groups:
        g_demand = group_demand[g_id]
        g_custs = group_customers[g_id]
        
        # Compute centroid of group
        cx = sum(nodes[c][0] for c in g_custs) / len(g_custs)
        cy = sum(nodes[c][1] for c in g_custs) / len(g_custs)
        
        best_sat = None
        best_dist = float('inf')
        
        for s in sat_indices:
            # Check capacity
            if sat_load[s] + g_demand > sat_capacity[s]:
                continue
            # Check if satellite can handle with Q2 vehicles
            # Rough estimate of vehicles needed
            est_vehicles = math.ceil((sat_load[s] + g_demand) / Q2) if Q2 > 0 else float('inf')
            if est_vehicles > sat_max_vehicles[s]:
                continue
            
            d = euclidean_dist((cx, cy), nodes[s])
            if d < best_dist:
                best_dist = d
                best_sat = s
        
        if best_sat is None:
            # Fallback: assign to satellite with most remaining capacity
            best_sat = max(sat_indices, key=lambda s: sat_capacity[s] - sat_load[s])
        
        group_assignment[g_id] = best_sat
        sat_load[best_sat] += g_demand
    
    # --- Solve second echelon for each satellite ---
    def get_sat_customers(ga):
        """Get customers assigned to each satellite based on group assignment."""
        sat_custs = defaultdict(list)
        for g_id, s in ga.items():
            for c in group_customers[g_id]:
                sat_custs[s].append(c)
        return sat_custs
    
    def solve_all_second_echelon(ga):
        """Solve second echelon for all satellites given group assignment."""
        sat_custs = get_sat_customers(ga)
        second_routes = {}
        total_se_vehicles = 0
        
        for s in sat_indices:
            custs = sat_custs.get(s, [])
            if not custs:
                second_routes[s] = []
                continue
            routes = solve_second_echelon(s, custs, dist, cust_demand, Q2, sat_max_vehicles[s])
            second_routes[s] = routes
            total_se_vehicles += len(routes)
        
        return second_routes, total_se_vehicles
    
    def solve_all_first_echelon(ga):
        """Solve first echelon given group assignment."""
        # Determine satellite demands
        sat_custs = get_sat_customers(ga)
        sat_demands = {}
        used_sats = []
        for s in sat_indices:
            demand = sum(cust_demand[c] for c in sat_custs.get(s, []))
            if demand > 0:
                sat_demands[s] = demand
                used_sats.append(s)
        
        if not used_sats:
            return [], sat_demands
        
        routes = solve_first_echelon(depot_idx, used_sats, sat_demands, dist, Q1, T1)
        return routes, sat_demands
    
    def compute_total_cost(ga):
        """Compute total cost for a given group assignment."""
        second_routes, total_se_vehicles = solve_all_second_echelon(ga)
        first_routes, sat_demands = solve_all_first_echelon(ga)
        
        # First echelon cost
        fe_cost = sum(route_cost(r, depot_idx, dist) for r in first_routes)
        
        # Handling cost
        h_cost = sum(sat_handling[s] * sat_demands.get(s, 0) for s in sat_indices)
        
        # Second echelon cost
        se_cost = 0.0
        for s in sat_indices:
            for r in second_routes.get(s, []):
                se_cost += route_cost(r, s, dist)
        
        return fe_cost + h_cost + se_cost, first_routes, second_routes, sat_demands
    
    # Get initial solution
    total_cost, first_routes, second_routes, sat_demands = compute_total_cost(group_assignment)
    
    # Build and log initial solution
    global sat_demands_total_global
    sat_demands_total_global = sat_demands
    
    solution_dict = build_solution_dict(
        instance, first_routes, second_routes, group_assignment,
        node_ids, depot_id, [node_ids[s] for s in sat_indices],
        [node_ids[c] for c in cust_indices], total_cost
    )
    
    if logger:
        logger.log_solution(total_cost, solution_dict)
    
    best_cost = total_cost
    best_ga = dict(group_assignment)
    best_first_routes = first_routes
    best_second_routes = second_routes
    best_sat_demands = sat_demands
    
    # --- Local search: try moving groups to different satellites ---
    time_elapsed = time.time() - start_time
    remaining = args.time_limit - time_elapsed
    
    iteration = 0
    while remaining > 1.0:
        improved = False
        
        for g_id in group_ids:
            if time.time() - start_time > args.time_limit - 2:
                break
            
            current_sat = group_assignment[g_id]
            
            for s in sat_indices:
                if s == current_sat:
                    continue
                
                # Check feasibility
                g_demand_val = group_demand[g_id]
                
                # Temporarily move group
                old_sat = group_assignment[g_id]
                group_assignment[g_id] = s
                
                # Quick feasibility check
                sat_custs_new = get_sat_customers(group_assignment)
                new_demand_s = sum(cust_demand[c] for c in sat_custs_new.get(s, []))
                
                if new_demand_s > sat_capacity[s]:
                    group_assignment[g_id] = old_sat
                    continue
                
                # Check vehicle count estimate
                est_veh = math.ceil(new_demand_s / Q2) if Q2 > 0 else float('inf')
                if est_veh > sat_max_vehicles[s]:
                    group_assignment[g_id] = old_sat
                    continue
                
                # Compute new cost
                new_cost, new_fr, new_sr, new_sd = compute_total_cost(group_assignment)
                
                # Check total vehicle constraints
                total_se_veh = sum(len(new_sr.get(s2, [])) for s2 in sat_indices)
                if total_se_veh > T2 or len(new_fr) > T1:
                    group_assignment[g_id] = old_sat
                    continue
                
                if new_cost < best_cost - 1e-10:
                    best_cost = new_cost
                    best_ga = dict(group_assignment)
                    best_first_routes = new_fr
                    best_second_routes = new_sr
                    best_sat_demands = new_sd
                    improved = True
                    
                    sat_demands_total_global = new_sd
                    solution_dict = build_solution_dict(
                        instance, new_fr, new_sr, group_assignment,
                        node_ids, depot_id, [node_ids[s2] for s2 in sat_indices],
                        [node_ids[c] for c in cust_indices], new_cost
                    )
                    if logger:
                        logger.log_solution(new_cost, solution_dict)
                    break
                else:
                    group_assignment[g_id] = old_sat
            
            if time.time() - start_time > args.time_limit - 2:
                break
        
        if not improved:
            # Try swap: exchange two groups between satellites
            if time.time() - start_time > args.time_limit - 5:
                break
            
            swap_improved = False
            for i in range(len(group_ids)):
                for j in range(i + 1, len(group_ids)):
                    if time.time() - start_time > args.time_limit - 2:
                        break
                    
                    g1, g2 = group_ids[i], group_ids[j]
                    s1, s2 = group_assignment[g1], group_assignment[g2]
                    
                    if s1 == s2:
                        continue
                    
                    # Swap
                    group_assignment[g1] = s2
                    group_assignment[g2] = s1
                    
                    # Quick feasibility
                    sat_custs_new = get_sat_customers(group_assignment)
                    d_s1 = sum(cust_demand[c] for c in sat_custs_new.get(s1, []))
                    d_s2 = sum(cust_demand[c] for c in sat_custs_new.get(s2, []))
                    
                    if d_s1 > sat_capacity[s1] or d_s2 > sat_capacity[s2]:
                        group_assignment[g1] = s1
                        group_assignment[g2] = s2
                        continue
                    
                    new_cost, new_fr, new_sr, new_sd = compute_total_cost(group_assignment)
                    
                    total_se_veh = sum(len(new_sr.get(s2, [])) for s2 in sat_indices)
                    if total_se_veh > T2 or len(new_fr) > T1:
                        group_assignment[g1] = s1
                        group_assignment[g2] = s2
                        continue
                    
                    if new_cost < best_cost - 1e-10:
                        best_cost = new_cost
                        best_ga = dict(group_assignment)
                        best_first_routes = new_fr
                        best_second_routes = new_sr
                        best_sat_demands = new_sd
                        swap_improved = True
                        
                        sat_demands_total_global = new_sd
                        solution_dict = build_solution_dict(
                            instance, new_fr, new_sr, group_assignment,
                            node_ids, depot_id, [node_ids[s3] for s3 in sat_indices],
                            [node_ids[c] for c in cust_indices], new_cost
                        )
                        if logger:
                            logger.log_solution(new_cost, solution_dict)
                        break
                    else:
                        group_assignment[g1] = s1
                        group_assignment[g2] = s2
                
                if swap_improved:
                    break
                if time.time() - start_time > args.time_limit - 2:
                    break
            
            if not swap_improved:
                break
        
        iteration += 1
        remaining = args.time_limit - (time.time() - start_time)
    
    # --- Final: re-optimize routes with best assignment ---
    group_assignment = best_ga
    total_cost, first_routes, second_routes, sat_demands = compute_total_cost(group_assignment)
    
    # Additional local search on second echelon routes
    for s in sat_indices:
        if s in second_routes and second_routes[s]:
            routes = second_routes[s]
            # Try or-opt between routes
            routes = or_opt_routes(routes, s, dist, cust_demand, Q2)
            for i in range(len(routes)):
                routes[i] = two_opt_route(routes[i], s, dist)
            second_routes[s] = routes
    
    # Recompute cost
    fe_cost = sum(route_cost(r, depot_idx, dist) for r in first_routes)
    h_cost = sum(sat_handling[s] * sat_demands.get(s, 0) for s in sat_indices)
    se_cost = 0.0
    for s in sat_indices:
        for r in second_routes.get(s, []):
            se_cost += route_cost(r, s, dist)
    total_cost = fe_cost + h_cost + se_cost
    
    if total_cost < best_cost:
        best_cost = total_cost
    
    # Build final solution
    sat_demands_total_global = sat_demands
    
    # Compute deliveries for first echelon
    # Each satellite's total demand must be delivered by first echelon vehicles visiting it
    sat_visitors = defaultdict(list)
    for v_idx, route in enumerate(first_routes):
        for sat_node in route:
            sat_visitors[sat_node].append(v_idx)
    
    # Distribute deliveries
    delivery_amounts = {}  # (vehicle_idx, sat_node) -> amount
    for s in sat_indices:
        if s in sat_demands and sat_demands[s] > 0:
            visitors = sat_visitors.get(s, [])
            if visitors:
                # Try to respect Q1 capacity per vehicle
                remaining_demand = sat_demands[s]
                for v_idx in visitors:
                    # How much capacity does this vehicle have left?
                    current_load = sum(
                        delivery_amounts.get((v_idx, s2), 0) 
                        for s2 in first_routes[v_idx]
                    )
                    available = Q1 - current_load
                    deliver = min(remaining_demand, available, sat_capacity[s])
                    delivery_amounts[(v_idx, s)] = deliver
                    remaining_demand -= deliver
                    if remaining_demand <= 0:
                        break
                # If still remaining, distribute to first visitor
                if remaining_demand > 0 and visitors:
                    delivery_amounts[(visitors[0], s)] = delivery_amounts.get((visitors[0], s), 0) + remaining_demand
    
    # Build output
    fe_routes_out = []
    for v_idx, route in enumerate(first_routes):
        route_ids = [depot_id] + [node_ids[n] for n in route] + [depot_id]
        deliveries = {}
        for sat_node in route:
            sat_id = node_ids[sat_node]
            amt = delivery_amounts.get((v_idx, sat_node), 0)
            if amt > 0:
                deliveries[str(sat_id)] = round(amt, 6)
        fe_routes_out.append({
            "vehicle": v_idx + 1,
            "route": route_ids,
            "deliveries": deliveries
        })
    
    se_routes_out = []
    for s in sat_indices:
        sat_id = node_ids[s]
        for route in second_routes.get(s, []):
            route_ids = [sat_id] + [node_ids[n] for n in route] + [sat_id]
            se_routes_out.append({
                "satellite": sat_id,
                "route": route_ids
            })
    
    ga_out = {}
    for g_id, s in group_assignment.items():
        ga_out[str(g_id)] = node_ids[s]
    
    solution = {
        "objective_value": best_cost,
        "first_echelon_routes": fe_routes_out,
        "second_echelon_routes": se_routes_out,
        "group_assignments": ga_out
    }
    
    if logger:
        logger.log_solution(best_cost, solution)
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)
    
    print(f"Solution written with objective value: {best_cost:.4f}")


if __name__ == '__main__':
    main()
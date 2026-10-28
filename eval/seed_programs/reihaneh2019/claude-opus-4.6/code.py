import argparse
import json
import math
import time
import random
from itertools import combinations

def solve(instance_path, solution_path, time_limit, log_path):
    from solution_logger import SolutionLogger
    
    with open(instance_path, 'r') as f:
        data = json.load(f)
    
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    start_time = time.time()
    
    num_sites = data['num_delivery_sites']
    num_customers = data['num_customers']
    vehicle_capacity = data['vehicle_capacity']
    num_vehicles = data['num_vehicles']
    depot = data['depot']
    sites = data['delivery_sites']
    customers = data['customers']
    routing_cost = data['routing_cost_matrix']
    assignment_cost = data['assignment_cost_matrix']
    
    # Site IDs (1-indexed in routing cost matrix)
    site_ids = [s['id'] for s in sites]
    customer_ids = [c['id'] for c in customers]
    customer_demands = [c['demand'] for c in customers]
    
    # Mapping from site id to index in routing_cost_matrix (1-based)
    site_id_to_rc_idx = {}
    for i, s in enumerate(sites):
        site_id_to_rc_idx[s['id']] = i + 1  # routing cost matrix: 0=depot, 1..num_sites=sites
    
    # Mapping from site id to index in assignment_cost_matrix (0-based)
    site_id_to_ac_idx = {}
    for i, s in enumerate(sites):
        site_id_to_ac_idx[s['id']] = i
    
    def get_routing_cost(node_a, node_b):
        """Get routing cost between two routing network nodes (0=depot, 1..n=sites)"""
        return routing_cost[node_a][node_b]
    
    def get_assignment_cost(cust_idx, site_idx):
        """Get assignment cost for customer index to site index (both 0-based)"""
        return assignment_cost[cust_idx][site_idx]
    
    total_demand = sum(customer_demands)
    
    # For each customer, find best sites sorted by assignment cost
    cust_best_sites = []
    for ci in range(num_customers):
        site_costs = [(get_assignment_cost(ci, si), si) for si in range(num_sites)]
        site_costs.sort()
        cust_best_sites.append(site_costs)
    
    def evaluate_solution(site_assignments, routes):
        """
        site_assignments: dict customer_idx -> site_idx (0-based)
        routes: list of lists of site_rc_idx (1-based routing cost matrix indices)
        Returns total cost or None if infeasible.
        """
        total_cost = 0.0
        
        # Assignment cost
        for ci, si in site_assignments.items():
            total_cost += get_assignment_cost(ci, si)
        
        # Routing cost
        for route in routes:
            if not route:
                continue
            # Route: depot -> s1 -> s2 -> ... -> depot
            prev = 0
            for node in route:
                total_cost += get_routing_cost(prev, node)
                prev = node
            total_cost += get_routing_cost(prev, 0)
        
        return total_cost
    
    def route_cost(route):
        """Calculate routing cost for a single route (list of rc_indices)"""
        if not route:
            return 0.0
        cost = get_routing_cost(0, route[0])
        for i in range(len(route) - 1):
            cost += get_routing_cost(route[i], route[i+1])
        cost += get_routing_cost(route[-1], 0)
        return cost
    
    def tsp_nearest_neighbor(nodes):
        """Solve TSP for nodes starting/ending at depot (node 0) using nearest neighbor"""
        if not nodes:
            return []
        if len(nodes) == 1:
            return list(nodes)
        
        unvisited = set(nodes)
        route = []
        current = 0
        while unvisited:
            best_next = min(unvisited, key=lambda n: get_routing_cost(current, n))
            route.append(best_next)
            unvisited.remove(best_next)
            current = best_next
        return route
    
    def tsp_2opt(route):
        """Improve route using 2-opt"""
        if len(route) <= 2:
            return route
        
        improved = True
        best_route = list(route)
        while improved:
            improved = False
            for i in range(len(best_route) - 1):
                for j in range(i + 1, len(best_route)):
                    # Current edges: (prev_i, route[i]), (route[j], next_j)
                    if i == 0:
                        prev_i = 0
                    else:
                        prev_i = best_route[i-1]
                    
                    if j == len(best_route) - 1:
                        next_j = 0
                    else:
                        next_j = best_route[j+1]
                    
                    old_cost = get_routing_cost(prev_i, best_route[i]) + get_routing_cost(best_route[j], next_j)
                    new_cost = get_routing_cost(prev_i, best_route[j]) + get_routing_cost(best_route[i], next_j)
                    
                    if new_cost < old_cost - 1e-10:
                        best_route[i:j+1] = reversed(best_route[i:j+1])
                        improved = True
            
        return best_route
    
    def build_routes_from_clusters(clusters):
        """
        clusters: list of (list of site_rc_indices, total_demand)
        Returns list of optimized routes
        """
        routes = []
        for site_list, _ in clusters:
            if not site_list:
                continue
            route = tsp_nearest_neighbor(site_list)
            route = tsp_2opt(route)
            routes.append(route)
        return routes
    
    # Greedy constructive heuristic
    def greedy_construct():
        """
        Assign each customer to nearest site, then cluster sites into vehicle routes.
        """
        # Assign each customer to nearest site
        assignments = {}  # cust_idx -> site_idx
        for ci in range(num_customers):
            best_si = cust_best_sites[ci][0][1]
            assignments[ci] = best_si
        
        # Determine which sites are active and their demands
        site_demand = {}  # site_rc_idx -> total demand
        site_customers = {}  # site_rc_idx -> list of customer indices
        for ci, si in assignments.items():
            rc_idx = si + 1  # routing cost matrix index
            if rc_idx not in site_demand:
                site_demand[rc_idx] = 0
                site_customers[rc_idx] = []
            site_demand[rc_idx] += customer_demands[ci]
            site_customers[rc_idx].append(ci)
        
        active_sites = list(site_demand.keys())
        
        # Cluster sites using a savings-based approach (Clarke-Wright)
        clusters = cluster_sites_cw(active_sites, site_demand, vehicle_capacity, num_vehicles)
        
        routes = build_routes_from_clusters(clusters)
        
        return assignments, routes, site_customers
    
    def cluster_sites_cw(active_sites, site_demand, capacity, max_vehicles):
        """Clarke-Wright savings algorithm to cluster sites into routes"""
        # Start with each site in its own route
        site_to_cluster = {}
        clusters = {}
        for i, s in enumerate(active_sites):
            clusters[i] = {'sites': [s], 'demand': site_demand[s]}
            site_to_cluster[s] = i
        
        # Calculate savings
        savings = []
        for i in range(len(active_sites)):
            for j in range(i+1, len(active_sites)):
                s_i = active_sites[i]
                s_j = active_sites[j]
                saving = get_routing_cost(s_i, 0) + get_routing_cost(0, s_j) - get_routing_cost(s_i, s_j)
                savings.append((saving, s_i, s_j))
        
        savings.sort(reverse=True)
        
        for saving, s_i, s_j in savings:
            c_i = site_to_cluster[s_i]
            c_j = site_to_cluster[s_j]
            if c_i == c_j:
                continue
            if c_i not in clusters or c_j not in clusters:
                continue
            
            new_demand = clusters[c_i]['demand'] + clusters[c_j]['demand']
            if new_demand > capacity:
                continue
            
            # Merge clusters
            clusters[c_i]['sites'].extend(clusters[c_j]['sites'])
            clusters[c_i]['demand'] = new_demand
            for s in clusters[c_j]['sites']:
                site_to_cluster[s] = c_i
            del clusters[c_j]
        
        result = []
        for cid, cdata in clusters.items():
            result.append((cdata['sites'], cdata['demand']))
        
        return result
    
    # Initial solution
    assignments, routes, site_customers_map = greedy_construct()
    
    # Build complete solution info
    def compute_objective(assignments, routes):
        total = 0.0
        for ci, si in assignments.items():
            total += get_assignment_cost(ci, si)
        for route in routes:
            total += route_cost(route)
        return total
    
    best_obj = compute_objective(assignments, routes)
    best_assignments = dict(assignments)
    best_routes = [list(r) for r in routes]
    
    if logger:
        logger.log(best_obj)
    
    # Build reverse mapping: site_rc_idx -> route_idx
    def build_site_route_map(routes):
        m = {}
        for ri, route in enumerate(routes):
            for s in route:
                m[s] = ri
        return m
    
    # Build site -> customers map
    def build_site_cust_map(assignments):
        m = {}
        for ci, si in assignments.items():
            rc_idx = si + 1
            if rc_idx not in m:
                m[rc_idx] = []
            m[rc_idx].append(ci)
        return m
    
    def build_route_demand(routes, assignments):
        """Calculate demand for each route"""
        site_cust = build_site_cust_map(assignments)
        demands = []
        for route in routes:
            d = 0
            for s in route:
                if s in site_cust:
                    for ci in site_cust[s]:
                        d += customer_demands[ci]
            demands.append(d)
        return demands
    
    # Local search improvement
    def local_search(assignments, routes, best_obj, time_limit_abs):
        improved = True
        current_obj = best_obj
        
        while improved and time.time() < time_limit_abs:
            improved = False
            
            site_cust = build_site_cust_map(assignments)
            site_route = build_site_route_map(routes)
            route_demands = build_route_demand(routes, assignments)
            
            # Try reassigning customers to different sites
            customer_order = list(range(num_customers))
            random.shuffle(customer_order)
            
            for ci in customer_order:
                if time.time() >= time_limit_abs:
                    break
                
                current_si = assignments[ci]
                current_rc = current_si + 1
                current_ri = site_route.get(current_rc)
                current_assign_cost = get_assignment_cost(ci, current_si)
                
                best_delta = 0
                best_new_si = None
                best_action = None  # ('reassign', new_si) or ('reassign_new_route', new_si)
                
                for cost, new_si in cust_best_sites[ci][:min(10, num_sites)]:
                    if new_si == current_si:
                        continue
                    
                    new_rc = new_si + 1
                    new_assign_cost = cost
                    assign_delta = new_assign_cost - current_assign_cost
                    
                    if new_rc in site_route:
                        # Site already active on some route
                        new_ri = site_route[new_rc]
                        
                        # Check capacity
                        added_demand = customer_demands[ci]
                        if new_ri != current_ri:
                            if route_demands[new_ri] + added_demand > vehicle_capacity:
                                continue
                        
                        delta = assign_delta
                        
                        # Check if old site becomes empty
                        old_site_custs = [c for c in site_cust.get(current_rc, []) if c != ci]
                        if not old_site_custs:
                            # Remove site from its route
                            old_route = routes[current_ri]
                            if len(old_route) == 1:
                                # Route becomes empty
                                delta -= route_cost(old_route)
                            else:
                                new_route = [s for s in old_route if s != current_rc]
                                delta += route_cost(new_route) - route_cost(old_route)
                        
                        if delta < best_delta - 1e-10:
                            best_delta = delta
                            best_new_si = new_si
                            best_action = 'reassign_existing'
                    
                    else:
                        # Site not active, need to add to some route
                        # Try adding to each route
                        for ri, route in enumerate(routes):
                            if time.time() >= time_limit_abs:
                                break
                            
                            added_demand = customer_demands[ci]
                            if ri != current_ri:
                                if route_demands[ri] + added_demand > vehicle_capacity:
                                    continue
                            else:
                                # Same route, demand doesn't change if old site still has customers
                                old_site_custs = [c for c in site_cust.get(current_rc, []) if c != ci]
                                if old_site_custs:
                                    if route_demands[ri] + added_demand > vehicle_capacity:
                                        continue
                            
                            # Calculate cost of adding new_rc to route ri
                            # Find best insertion position
                            ext_route = route[:]
                            best_insert_cost = float('inf')
                            old_rc = route_cost(ext_route)
                            
                            for pos in range(len(ext_route) + 1):
                                trial = ext_route[:pos] + [new_rc] + ext_route[pos:]
                                ic = route_cost(trial)
                                if ic < best_insert_cost:
                                    best_insert_cost = ic
                            
                            delta = assign_delta + (best_insert_cost - old_rc)
                            
                            # Check if old site becomes empty
                            old_site_custs = [c for c in site_cust.get(current_rc, []) if c != ci]
                            if not old_site_custs and current_ri is not None:
                                old_route = routes[current_ri]
                                if ri == current_ri:
                                    # Both on same route - need to handle together
                                    new_route_sites = [s for s in old_route if s != current_rc] + [new_rc]
                                    # Optimize this route
                                    trial_route = tsp_nearest_neighbor(new_route_sites)
                                    trial_route = tsp_2opt(trial_route)
                                    delta = assign_delta + route_cost(trial_route) - route_cost(old_route)
                                else:
                                    if len(old_route) == 1:
                                        delta -= route_cost(old_route)
                                    else:
                                        rem_route = [s for s in old_route if s != current_rc]
                                        delta += route_cost(rem_route) - route_cost(old_route)
                            
                            if delta < best_delta - 1e-10:
                                best_delta = delta
                                best_new_si = new_si
                                best_action = ('reassign_new', ri)
                
                if best_action is not None and best_delta < -1e-10:
                    # Apply the move
                    old_si = assignments[ci]
                    old_rc = old_si + 1
                    old_ri = site_route.get(old_rc)
                    
                    assignments[ci] = best_new_si
                    new_rc = best_new_si + 1
                    
                    # Update site_cust
                    if old_rc in site_cust:
                        site_cust[old_rc] = [c for c in site_cust[old_rc] if c != ci]
                    if new_rc not in site_cust:
                        site_cust[new_rc] = []
                    site_cust[new_rc].append(ci)
                    
                    old_site_empty = not site_cust.get(old_rc, [])
                    
                    if best_action == 'reassign_existing':
                        new_ri = site_route[new_rc]
                        
                        if old_site_empty and old_ri is not None:
                            old_route = routes[old_ri]
                            if len(old_route) == 1:
                                routes[old_ri] = []
                            else:
                                new_route = [s for s in old_route if s != old_rc]
                                new_route = tsp_2opt(new_route)
                                routes[old_ri] = new_route
                            if old_rc in site_route:
                                del site_route[old_rc]
                        
                        route_demands = build_route_demand(routes, assignments)
                        
                    elif isinstance(best_action, tuple) and best_action[0] == 'reassign_new':
                        target_ri = best_action[1]
                        
                        if old_site_empty and old_ri is not None and old_ri == target_ri:
                            # Same route: remove old, add new
                            new_sites = [s for s in routes[old_ri] if s != old_rc] + [new_rc]
                            new_route = tsp_nearest_neighbor(new_sites)
                            new_route = tsp_2opt(new_route)
                            routes[target_ri] = new_route
                            if old_rc in site_route:
                                del site_route[old_rc]
                            site_route[new_rc] = target_ri
                        else:
                            # Add new_rc to target route
                            ext_route = routes[target_ri][:]
                            best_insert_cost = float('inf')
                            best_insert_route = ext_route + [new_rc]
                            for pos in range(len(ext_route) + 1):
                                trial = ext_route[:pos] + [new_rc] + ext_route[pos:]
                                ic = route_cost(trial)
                                if ic < best_insert_cost:
                                    best_insert_cost = ic
                                    best_insert_route = trial
                            routes[target_ri] = best_insert_route
                            site_route[new_rc] = target_ri
                            
                            # Remove old site if empty
                            if old_site_empty and old_ri is not None:
                                old_route = routes[old_ri]
                                if len(old_route) <= 1:
                                    routes[old_ri] = []
                                else:
                                    new_route = [s for s in old_route if s != old_rc]
                                    new_route = tsp_2opt(new_route)
                                    routes[old_ri] = new_route
                                if old_rc in site_route:
                                    del site_route[old_rc]
                        
                        route_demands = build_route_demand(routes, assignments)
                    
                    current_obj = compute_objective(assignments, routes)
                    improved = True
                    
                    if current_obj < best_obj - 1e-10:
                        pass  # Will update below
            
            # Try inter-route moves: move a site from one route to another
            if time.time() < time_limit_abs:
                site_route = build_site_route_map(routes)
                route_demands = build_route_demand(routes, assignments)
                
                for ri in range(len(routes)):
                    if time.time() >= time_limit_abs:
                        break
                    if not routes[ri]:
                        continue
                    for si_pos, s in enumerate(routes[ri]):
                        if time.time() >= time_limit_abs:
                            break
                        s_demand = sum(customer_demands[ci] for ci in site_cust.get(s, []))
                        
                        # Try moving s to another route
                        old_route = routes[ri]
                        if len(old_route) == 1:
                            old_without = []
                            cost_delta_remove = -route_cost(old_route)
                        else:
                            old_without = [x for x in old_route if x != s]
                            cost_delta_remove = route_cost(old_without) - route_cost(old_route)
                        
                        best_move_delta = 0
                        best_target_ri = None
                        best_target_route = None
                        
                        for rj in range(len(routes)):
                            if rj == ri:
                                continue
                            if route_demands[rj] + s_demand > vehicle_capacity:
                                continue
                            
                            target_route = routes[rj]
                            if not target_route:
                                # New route with just this site
                                cost_add = route_cost([s])
                            else:
                                # Best insertion
                                best_ic = float('inf')
                                best_tr = None
                                for pos in range(len(target_route) + 1):
                                    trial = target_route[:pos] + [s] + target_route[pos:]
                                    ic = route_cost(trial)
                                    if ic < best_ic:
                                        best_ic = ic
                                        best_tr = trial
                                cost_add = best_ic - route_cost(target_route)
                            
                            delta = cost_delta_remove + cost_add
                            if delta < best_move_delta - 1e-10:
                                best_move_delta = delta
                                best_target_ri = rj
                                if not target_route:
                                    best_target_route = [s]
                                else:
                                    best_target_route = best_tr
                        
                        if best_target_ri is not None and best_move_delta < -1e-10:
                            routes[ri] = old_without
                            routes[best_target_ri] = best_target_route
                            site_route = build_site_route_map(routes)
                            route_demands = build_route_demand(routes, assignments)
                            current_obj = compute_objective(assignments, routes)
                            improved = True
                            break
                    else:
                        continue
                    break
            
            # Apply 2-opt to all routes
            for ri in range(len(routes)):
                if time.time() >= time_limit_abs:
                    break
                if routes[ri]:
                    old_cost = route_cost(routes[ri])
                    new_route = tsp_2opt(routes[ri])
                    new_cost = route_cost(new_route)
                    if new_cost < old_cost - 1e-10:
                        routes[ri] = new_route
                        improved = True
            
            current_obj = compute_objective(assignments, routes)
        
        return assignments, routes, current_obj
    
    time_limit_abs = start_time + time_limit - 1  # Leave 1 second buffer
    
    assignments, routes, current_obj = local_search(assignments, routes, best_obj, time_limit_abs)
    
    if current_obj < best_obj - 1e-10:
        best_obj = current_obj
        best_assignments = dict(assignments)
        best_routes = [list(r) for r in routes]
        if logger:
            logger.log(best_obj)
    
    # Try multiple random restarts
    num_restarts = 0
    while time.time() < time_limit_abs - 2:
        num_restarts += 1
        
        # Randomized construction: assign each customer to one of top-k sites randomly
        trial_assignments = {}
        for ci in range(num_customers):
            k = min(3, len(cust_best_sites[ci]))
            weights = []
            choices = []
            for cost, si in cust_best_sites[ci][:k]:
                choices.append(si)
                weights.append(1.0 / (cost + 1e-6))
            total_w = sum(weights)
            r = random.random() * total_w
            cum = 0
            chosen = choices[0]
            for idx, w in enumerate(weights):
                cum += w
                if r <= cum:
                    chosen = choices[idx]
                    break
            trial_assignments[ci] = chosen
        
        # Build site demand map
        site_demand = {}
        for ci, si in trial_assignments.items():
            rc_idx = si + 1
            if rc_idx not in site_demand:
                site_demand[rc_idx] = 0
            site_demand[rc_idx] += customer_demands[ci]
        
        active_sites = list(site_demand.keys())
        clusters = cluster_sites_cw(active_sites, site_demand, vehicle_capacity, num_vehicles)
        trial_routes = build_routes_from_clusters(clusters)
        
        trial_obj = compute_objective(trial_assignments, trial_routes)
        
        if trial_obj < best_obj:
            best_obj = trial_obj
            best_assignments = dict(trial_assignments)
            best_routes = [list(r) for r in trial_routes]
            if logger:
                logger.log(best_obj)
        
        # Local search on this solution
        remaining = time_limit_abs - time.time()
        if remaining < 2:
            break
        ls_time = min(remaining - 1, max(2, remaining / 3))
        
        trial_assignments, trial_routes, trial_obj = local_search(
            trial_assignments, trial_routes, trial_obj, time.time() + ls_time
        )
        
        if trial_obj < best_obj - 1e-10:
            best_obj = trial_obj
            best_assignments = dict(trial_assignments)
            best_routes = [list(r) for r in trial_routes]
            if logger:
                logger.log(best_obj)
    
    # Try Gurobi for small instances if time remains
    remaining_time = time_limit_abs - time.time()
    if remaining_time > 5 and num_sites <= 30 and num_customers <= 50:
        try:
            import gurobipy as gp
            from gurobipy import GRB
            
            # Simplified MIP formulation
            active_routes = [r for r in best_routes if r]
            
            # Use Gurobi to optimize customer-to-site assignments given fixed routes
            m = gp.Model("assignment_opt")
            m.setParam('TimeLimit', max(1, remaining_time - 2))
            m.setParam('OutputFlag', 0)
            m.setParam('Threads', 1)
            
            # Variables: x[ci][si] = 1 if customer ci assigned to site si
            x = {}
            for ci in range(num_customers):
                for si in range(num_sites):
                    rc_idx = si + 1
                    # Only allow assignment to active sites
                    site_active = any(rc_idx in r for r in active_routes)
                    if site_active:
                        x[ci, si] = m.addVar(vtype=GRB.BINARY, obj=get_assignment_cost(ci, si))
            
            # Each customer assigned to exactly one site
            for ci in range(num_customers):
                m.addConstr(gp.quicksum(x[ci, si] for si in range(num_sites) if (ci, si) in x) == 1)
            
            # Capacity constraints per route
            site_to_route_idx = {}
            for ri, route in enumerate(active_routes):
                for s in route:
                    site_to_route_idx[s] = ri
            
            for ri, route in enumerate(active_routes):
                m.addConstr(
                    gp.quicksum(
                        customer_demands[ci] * x[ci, s-1]
                        for s in route
                        for ci in range(num_customers)
                        if (ci, s-1) in x
                    ) <= vehicle_capacity
                )
            
            m.optimize()
            
            if m.status in [GRB.OPTIMAL, GRB.SUBOPTIMAL] and m.SolCount > 0:
                new_assignments = {}
                for ci in range(num_customers):
                    for si in range(num_sites):
                        if (ci, si) in x and x[ci, si].X > 0.5:
                            new_assignments[ci] = si
                            break
                
                if len(new_assignments) == num_customers:
                    new_obj = compute_objective(new_assignments, active_routes)
                    if new_obj < best_obj - 1e-10:
                        best_obj = new_obj
                        best_assignments = new_assignments
                        best_routes = active_routes
                        if logger:
                            logger.log(best_obj)
        except Exception:
            pass
    
    # Format output
    output_routes = {}
    output_customer_tours = {}
    output_assignments = {}
    
    # Build site -> route mapping
    site_to_vehicle = {}
    vehicle_id = 0
    active_routes = [r for r in best_routes if r]
    
    for route in active_routes:
        arcs = []
        sequence = [0]
        prev = 0
        for s in route:
            arcs.append([prev, s])
            sequence.append(s)
            prev = s
        arcs.append([prev, 0])
        sequence.append(0)
        
        output_routes[str(vehicle_id)] = {
            'arcs': arcs,
            'sequence': sequence
        }
        
        for s in route:
            site_to_vehicle[s] = vehicle_id
        
        vehicle_id += 1
    
    for ci, si in best_assignments.items():
        cust_id = customer_ids[ci]
        site_id = site_ids[si]
        rc_idx = si + 1
        vid = site_to_vehicle.get(rc_idx, 0)
        
        output_assignments[str(cust_id)] = site_id
        output_customer_tours[str(cust_id)] = {
            'vehicle': vid,
            'site': site_id
        }
    
    solution = {
        'objective_value': best_obj,
        'routes': output_routes,
        'assignments': output_assignments,
        'customer_tours': output_customer_tours
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
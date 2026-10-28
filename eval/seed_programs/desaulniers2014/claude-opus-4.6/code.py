import argparse
import json
import math
import time
import random
from copy import deepcopy

def solve_irp(instance_path, solution_path, time_limit, log_path):
    from solution_logger import SolutionLogger
    
    with open(instance_path, 'r') as f:
        data = json.load(f)
    
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    start_time = time.time()
    
    num_customers = data['num_customers']
    num_periods = data['num_periods']
    num_vehicles = data['num_vehicles']
    vehicle_capacity = data['vehicle_capacity']
    depot = data['depot']
    customers = data['customers']
    dist = data['distance_matrix']
    
    cust_demand = [0] * (num_customers + 1)
    cust_init_inv = [0] * (num_customers + 1)
    cust_max_inv = [0] * (num_customers + 1)
    cust_min_inv = [0] * (num_customers + 1)
    cust_hold = [0.0] * (num_customers + 1)
    
    for c in customers:
        cid = c['id']
        cust_demand[cid] = c['demand_per_period']
        cust_init_inv[cid] = c['initial_inventory']
        cust_max_inv[cid] = c['max_inventory']
        cust_min_inv[cid] = c.get('min_inventory', 0)
        cust_hold[cid] = c['holding_cost']
    
    depot_init_inv = depot['initial_inventory']
    depot_max_inv = depot['max_inventory']
    depot_prod = depot['production_per_period']
    depot_hold = depot['holding_cost']
    
    customer_ids = [c['id'] for c in customers]
    
    def time_left():
        return time_limit - (time.time() - start_time)
    
    def route_cost(route):
        cost = 0
        for i in range(len(route) - 1):
            cost += dist[route[i]][route[i+1]]
        return cost
    
    def evaluate_solution(sol_periods):
        total_cost = 0.0
        depot_inv = depot_init_inv
        cust_inv = {}
        for c in customers:
            cust_inv[c['id']] = c['initial_inventory']
        
        for t in range(num_periods):
            period_data = sol_periods[t]
            routes = period_data['routes']
            deliveries = period_data['deliveries']
            
            for route in routes:
                for i in range(len(route) - 1):
                    total_cost += dist[route[i]][route[i+1]]
            
            depot_inv += depot_prod
            total_delivered = sum(deliveries.values())
            depot_inv -= total_delivered
            
            if depot_inv < -0.5 or depot_inv > depot_max_inv + 0.5:
                return float('inf')
            
            total_cost += depot_hold * depot_inv
            
            for c in customers:
                cid = c['id']
                cust_inv[cid] += deliveries.get(cid, 0)
                cust_inv[cid] -= cust_demand[cid]
                
                if cust_inv[cid] < cust_min_inv[cid] - 0.5 or cust_inv[cid] > cust_max_inv[cid] + 0.5:
                    return float('inf')
                
                total_cost += cust_hold[cid] * cust_inv[cid]
        
        return total_cost
    
    def two_opt(route):
        if len(route) <= 4:
            return list(route)
        improved = True
        best_route = list(route)
        while improved:
            improved = False
            for i in range(1, len(best_route) - 2):
                for j in range(i + 1, len(best_route) - 1):
                    new_route = best_route[:i] + best_route[i:j+1][::-1] + best_route[j+1:]
                    if route_cost(new_route) < route_cost(best_route) - 0.01:
                        best_route = new_route
                        improved = True
        return best_route
    
    def or_opt_route(route):
        if len(route) <= 3:
            return list(route)
        best_route = list(route)
        best_cost = route_cost(best_route)
        improved = True
        while improved:
            improved = False
            for i in range(1, len(best_route) - 1):
                node = best_route[i]
                temp = best_route[:i] + best_route[i+1:]
                for j in range(1, len(temp)):
                    if j == i:
                        continue
                    new_route = temp[:j] + [node] + temp[j:]
                    nc = route_cost(new_route)
                    if nc < best_cost - 0.01:
                        best_route = new_route
                        best_cost = nc
                        improved = True
                        break
                if improved:
                    break
        return best_route
    
    def improve_all_routes(sol_periods):
        for t in range(num_periods):
            new_routes = []
            for route in sol_periods[t]['routes']:
                r = two_opt(route)
                r = or_opt_route(r)
                new_routes.append(r)
            sol_periods[t]['routes'] = new_routes
    
    def build_routes_savings(customers_to_visit, deliveries_dict):
        if not customers_to_visit:
            return []
        
        # Start with individual routes for each customer
        active_routes = []
        for c in customers_to_visit:
            active_routes.append([0, c, 0])
        
        # Compute savings
        savings = []
        custs = list(customers_to_visit)
        for i in range(len(custs)):
            for j in range(i + 1, len(custs)):
                ci = custs[i]
                cj = custs[j]
                s = dist[0][ci] + dist[0][cj] - dist[ci][cj]
                if s > 0:
                    savings.append((s, ci, cj))
        
        savings.sort(reverse=True)
        
        def find_route(node):
            for idx, r in enumerate(active_routes):
                if node in r[1:-1]:
                    return idx
            return -1
        
        def route_load(r):
            return sum(deliveries_dict.get(n, 0) for n in r if n != 0)
        
        for s, ci, cj in savings:
            ri = find_route(ci)
            rj = find_route(cj)
            if ri == -1 or rj == -1 or ri == rj:
                continue
            
            route_i = active_routes[ri]
            route_j = active_routes[rj]
            
            if route_load(route_i) + route_load(route_j) > vehicle_capacity:
                continue
            
            # Check if ci and cj are at the ends of their routes (adjacent to depot)
            can_merge = False
            new_route = None
            
            if route_i[-2] == ci and route_j[1] == cj:
                new_route = route_i[:-1] + route_j[1:]
                can_merge = True
            elif route_j[-2] == cj and route_i[1] == ci:
                new_route = route_j[:-1] + route_i[1:]
                can_merge = True
            elif route_i[1] == ci and route_j[1] == cj:
                new_route = list(reversed(route_i[:-1])) + route_j[1:]
                if new_route[0] != 0:
                    continue
                can_merge = True
            elif route_i[-2] == ci and route_j[-2] == cj:
                new_route = route_i[:-1] + list(reversed(route_j[1:]))
                if new_route[-1] != 0:
                    continue
                can_merge = True
            
            if can_merge and new_route:
                # Remove both, add merged
                indices = sorted([ri, rj], reverse=True)
                for idx in indices:
                    active_routes.pop(idx)
                active_routes.append(new_route)
        
        # Limit to num_vehicles by merging smallest routes
        while len(active_routes) > num_vehicles:
            active_routes.sort(key=lambda r: route_load(r))
            small = active_routes.pop(0)
            small_custs = [n for n in small if n != 0]
            small_load = sum(deliveries_dict.get(n, 0) for n in small_custs)
            merged = False
            for ri in range(len(active_routes)):
                r = active_routes[ri]
                rl = route_load(r)
                if rl + small_load <= vehicle_capacity:
                    for cc in small_custs:
                        best_pos = 1
                        best_cv = float('inf')
                        for pos in range(1, len(r)):
                            cv = dist[r[pos-1]][cc] + dist[cc][r[pos]] - dist[r[pos-1]][r[pos]]
                            if cv < best_cv:
                                best_cv = cv
                                best_pos = pos
                        r.insert(best_pos, cc)
                    merged = True
                    break
            if not merged:
                # Can't merge, put it back
                active_routes.insert(0, small)
                break
        
        return active_routes
    
    def build_routes_nn(customers_to_visit, deliveries_dict):
        if not customers_to_visit:
            return []
        
        routes = []
        unvisited = list(customers_to_visit)
        
        while unvisited and len(routes) < num_vehicles:
            route = [0]
            load = 0
            current = 0
            
            while True:
                best_cust = None
                best_dist = float('inf')
                for c in unvisited:
                    d = deliveries_dict.get(c, 0)
                    if load + d <= vehicle_capacity:
                        if dist[current][c] < best_dist:
                            best_dist = dist[current][c]
                            best_cust = c
                
                if best_cust is None:
                    break
                
                route.append(best_cust)
                load += deliveries_dict.get(best_cust, 0)
                unvisited.remove(best_cust)
                current = best_cust
            
            route.append(0)
            if len(route) > 2:
                routes.append(route)
        
        # If still unvisited, try to fit into existing routes
        for c in list(unvisited):
            d = deliveries_dict.get(c, 0)
            best_route_idx = -1
            best_pos = -1
            best_cost_increase = float('inf')
            
            for ri, r in enumerate(routes):
                rl = sum(deliveries_dict.get(n, 0) for n in r if n != 0)
                if rl + d <= vehicle_capacity:
                    for pos in range(1, len(r)):
                        cv = dist[r[pos-1]][c] + dist[c][r[pos]] - dist[r[pos-1]][r[pos]]
                        if cv < best_cost_increase:
                            best_cost_increase = cv
                            best_route_idx = ri
                            best_pos = pos
            
            if best_route_idx >= 0:
                routes[best_route_idx].insert(best_pos, c)
                unvisited.remove(c)
            elif len(routes) < num_vehicles:
                routes.append([0, c, 0])
                unvisited.remove(c)
        
        return routes
    
    def compute_urgency(cid, period, inv):
        remaining = inv
        for t in range(period, num_periods):
            remaining -= cust_demand[cid]
            if remaining < cust_min_inv[cid]:
                return t - period
        return num_periods - period
    
    def build_solution_with_params(deliver_threshold, fill_ratio, use_savings=False, seed_val=42):
        """Build a solution with parameterized strategy."""
        rng = random.Random(seed_val)
        
        sol_periods = []
        depot_inv = depot_init_inv
        cust_inv = {}
        for c in customers:
            cust_inv[c['id']] = c['initial_inventory']
        
        feasible = True
        
        for t in range(num_periods):
            depot_inv += depot_prod
            
            customer_info = []
            for c in customers:
                cid = c['id']
                
                urgency = compute_urgency(cid, t, cust_inv[cid])
                min_del = max(0, cust_min_inv[cid] + cust_demand[cid] - cust_inv[cid])
                max_del = max(0, cust_max_inv[cid] - cust_inv[cid] + cust_demand[cid])
                
                customer_info.append({
                    'id': cid,
                    'min_del': min_del,
                    'max_del': max_del,
                    'urgency': urgency,
                    'inv': cust_inv[cid]
                })
            
            customer_info.sort(key=lambda x: (x['urgency'], -x['min_del']))
            
            deliveries = {}
            customers_to_visit = []
            available_supply = depot_inv
            
            # Must-deliver customers
            for ci in customer_info:
                cid = ci['id']
                if ci['min_del'] > 0:
                    target_inv = cust_max_inv[cid] * fill_ratio
                    target_del = max(ci['min_del'], int(target_inv - ci['inv'] + cust_demand[cid]))
                    target_del = min(target_del, ci['max_del'])
                    del_qty = min(target_del, available_supply)
                    if del_qty > 0:
                        deliveries[cid] = del_qty
                        customers_to_visit.append(cid)
                        available_supply -= del_qty
            
            # Soon-to-need customers
            for ci in customer_info:
                cid = ci['id']
                if cid in deliveries:
                    continue
                if available_supply <= 0:
                    break
                if ci['urgency'] <= deliver_threshold and ci['max_del'] > 0:
                    target_inv = cust_max_inv[cid] * fill_ratio
                    target_del = max(1, int(target_inv - ci['inv'] + cust_demand[cid]))
                    target_del = min(target_del, ci['max_del'])
                    del_qty = min(target_del, available_supply)
                    if del_qty > 0:
                        deliveries[cid] = del_qty
                        customers_to_visit.append(cid)
                        available_supply -= del_qty
            
            # Depot overflow prevention
            future_depot = depot_inv - sum(deliveries.values())
            if t < num_periods - 1 and future_depot + depot_prod > depot_max_inv:
                overflow = future_depot + depot_prod - depot_max_inv
                for ci in customer_info:
                    if overflow <= 0:
                        break
                    cid = ci['id']
                    if cid in deliveries:
                        current_del = deliveries[cid]
                        max_increase = ci['max_del'] - current_del
                        if max_increase > 0:
                            increase = min(max_increase, overflow)
                            deliveries[cid] += increase
                            overflow -= increase
                    else:
                        if ci['max_del'] > 0:
                            del_qty = min(ci['max_del'], overflow)
                            if del_qty > 0:
                                deliveries[cid] = del_qty
                                customers_to_visit.append(cid)
                                overflow -= del_qty
            
            # Ensure total delivery doesn't exceed depot inventory
            total_del = sum(deliveries.values())
            if total_del > depot_inv:
                # Scale down non-mandatory deliveries
                excess = total_del - depot_inv
                for ci in reversed(customer_info):
                    if excess <= 0:
                        break
                    cid = ci['id']
                    if cid not in deliveries:
                        continue
                    can_reduce = deliveries[cid] - ci['min_del']
                    if can_reduce > 0:
                        reduction = min(can_reduce, excess)
                        deliveries[cid] -= reduction
                        excess -= reduction
                        if deliveries[cid] == 0:
                            del deliveries[cid]
                            if cid in customers_to_visit:
                                customers_to_visit.remove(cid)
            
            # Build routes
            if customers_to_visit:
                if use_savings:
                    routes = build_routes_savings(customers_to_visit, deliveries)
                else:
                    routes = build_routes_nn(customers_to_visit, deliveries)
                routes = [two_opt(r) for r in routes]
            else:
                routes = []
            
            # Update inventories
            depot_inv -= sum(deliveries.values())
            if depot_inv < 0 or depot_inv > depot_max_inv:
                feasible = False
            
            inventories = {'depot': depot_inv}
            for c in customers:
                cid = c['id']
                cust_inv[cid] += deliveries.get(cid, 0)
                cust_inv[cid] -= cust_demand[cid]
                if cust_inv[cid] < cust_min_inv[cid] or cust_inv[cid] > cust_max_inv[cid]:
                    feasible = False
                inventories[cid] = cust_inv[cid]
            
            sol_periods.append({
                'routes': routes,
                'deliveries': deliveries,
                'inventories': inventories
            })
        
        if not feasible:
            return None
        return sol_periods
    
    # Try multiple parameter combinations
    best_sol = None
    best_obj = float('inf')
    
    param_combos = []
    for threshold in [0, 1, 2, 3]:
        for fill in [0.3, 0.5, 0.7, 0.9, 1.0]:
            for use_sav in [False, True]:
                param_combos.append((threshold, fill, use_sav))
    
    for threshold, fill, use_sav in param_combos:
        if time_left() < 2:
            break
        try:
            sol = build_solution_with_params(threshold, fill, use_sav)
            if sol is not None:
                obj = evaluate_solution(sol)
                if obj < best_obj:
                    best_sol = deepcopy(sol)
                    best_obj = obj
                    if logger:
                        logger.log(best_obj)
        except Exception:
            pass
    
    # Try random seeds
    for seed_val in range(200):
        if time_left() < 2:
            break
        threshold = random.Random(seed_val).randint(0, 4)
        fill = random.Random(seed_val + 1000).uniform(0.2, 1.0)
        use_sav = random.Random(seed_val + 2000).choice([True, False])
        try:
            sol = build_solution_with_params(threshold, fill, use_sav, seed_val)
            if sol is not None:
                obj = evaluate_solution(sol)
                if obj < best_obj:
                    best_sol = deepcopy(sol)
                    best_obj = obj
                    if logger:
                        logger.log(best_obj)
        except Exception:
            pass
    
    if best_sol is None:
        # Fallback: try very aggressive delivery
        try:
            sol = build_solution_with_params(num_periods, 1.0, True)
            if sol is not None:
                best_sol = sol
                best_obj = evaluate_solution(sol)
                if logger:
                    logger.log(best_obj)
        except Exception:
            pass
    
    # Local search improvements
    def try_reduce_deliveries(sol_periods):
        """Try reducing delivery quantities to save holding costs."""
        improved_any = False
        
        depot_inv_arr = []
        cust_inv_arr = []
        d_inv = depot_init_inv
        c_inv = {}
        for c in customers:
            c_inv[c['id']] = c['initial_inventory']
        
        for t in range(num_periods):
            d_inv += depot_prod
            d_inv -= sum(sol_periods[t]['deliveries'].values())
            depot_inv_arr.append(d_inv)
            ci_copy = {}
            for c in customers:
                cid = c['id']
                c_inv[cid] += sol_periods[t]['deliveries'].get(cid, 0)
                c_inv[cid] -= cust_demand[cid]
                ci_copy[cid] = c_inv[cid]
            cust_inv_arr.append(dict(ci_copy))
        
        for t in range(num_periods):
            if time_left() < 1:
                break
            for cid in list(sol_periods[t]['deliveries'].keys()):
                qty = sol_periods[t]['deliveries'][cid]
                if qty <= 0:
                    continue
                
                # Find next delivery period
                next_del = num_periods
                for tt in range(t + 1, num_periods):
                    if cid in sol_periods[tt]['deliveries'] and sol_periods[tt]['deliveries'][cid] > 0:
                        next_del = tt
                        break
                
                # Max reduction based on customer inventory slack
                slack = float('inf')
                for tt in range(t, next_del):
                    if tt < len(cust_inv_arr):
                        slack = min(slack, cust_inv_arr[tt][cid] - cust_min_inv[cid])
                
                # Max reduction based on depot capacity
                depot_cap_slack = float('inf')
                for tt in range(t, num_periods):
                    depot_cap_slack = min(depot_cap_slack, depot_max_inv - depot_inv_arr[tt])
                
                reduce = min(int(slack), int(depot_cap_slack), qty)
                if reduce <= 0:
                    continue
                
                # Calculate cost impact
                periods_at_cust = next_del - t
                cust_save = cust_hold[cid] * reduce * periods_at_cust
                depot_extra = depot_hold * reduce * (num_periods - t)
                
                if cust_save > depot_extra:
                    sol_periods[t]['deliveries'][cid] -= reduce
                    if sol_periods[t]['deliveries'][cid] == 0:
                        del sol_periods[t]['deliveries'][cid]
                        for ri, route in enumerate(sol_periods[t]['routes']):
                            if cid in route:
                                route.remove(cid)
                                if len(route) <= 2:
                                    sol_periods[t]['routes'].pop(ri)
                                break
                    
                    for tt in range(t, num_periods):
                        depot_inv_arr[tt] += reduce
                    for tt in range(t, next_del):
                        cust_inv_arr[tt][cid] -= reduce
                    
                    improved_any = True
        
        return improved_any
    
    def try_increase_deliveries(sol_periods):
        """Try increasing deliveries to push inventory from depot to cheaper customer storage."""
        improved_any = False
        
        depot_inv_arr = []
        cust_inv_arr = []
        d_inv = depot_init_inv
        c_inv = {}
        for c in customers:
            c_inv[c['id']] = c['initial_inventory']
        
        for t in range(num_periods):
            d_inv += depot_prod
            d_inv -= sum(sol_periods[t]['deliveries'].values())
            depot_inv_arr.append(d_inv)
            ci_copy = {}
            for c in customers:
                cid = c['id']
                c_inv[cid] += sol_periods[t]['deliveries'].get(cid, 0)
                c_inv[cid] -= cust_demand[cid]
                ci_copy[cid] = c_inv[cid]
            cust_inv_arr.append(dict(ci_copy))
        
        for t in range(num_periods):
            if time_left() < 1:
                break
            for cid_data in customers:
                cid = cid_data['id']
                if cid not in sol_periods[t]['deliveries']:
                    continue
                if depot_hold <= cust_hold[cid]:
                    continue
                
                next_del = num_periods
                for tt in range(t + 1, num_periods):
                    if cid in sol_periods[tt]['deliveries'] and sol_periods[tt]['deliveries'][cid] > 0:
                        next_del = tt
                        break
                
                cust_slack = float('inf')
                for tt in range(t, next_del):
                    cust_slack = min(cust_slack, cust_max_inv[cid] - cust_inv_arr[tt][cid])
                
                depot_slack = float('inf')
                for tt in range(t, num_periods):
                    depot_slack = min(depot_slack, depot_inv_arr[tt])
                
                # Vehicle capacity check
                route_idx = -1
                for ri, route in enumerate(sol_periods[t]['routes']):
                    if cid in route:
                        route_idx = ri
                        break
                
                if route_idx >= 0:
                    rl = sum(sol_periods[t]['deliveries'].get(n, 0) for n in sol_periods[t]['routes'][route_idx] if n != 0)
                    veh_slack = vehicle_capacity - rl
                else:
                    veh_slack = 0
                
                increase = min(int(cust_slack), int(depot_slack), veh_slack)
                if increase <= 0:
                    continue
                
                periods_at_cust = next_del - t
                cust_extra = cust_hold[cid] * increase * periods_at_cust
                depot_save = depot_hold * increase * (num_periods - t)
                
                if depot_save > cust_extra:
                    sol_periods[t]['deliveries'][cid] += increase
                    for tt in range(t, num_periods):
                        depot_inv_arr[tt] -= increase
                    for tt in range(t, next_del):
                        cust_inv_arr[tt][cid] += increase
                    improved_any = True
        
        return improved_any
    
    def recompute_inventories(sol_periods):
        d_inv = depot_init_inv
        c_inv = {}
        for c in customers:
            c_inv[c['id']] = c['initial_inventory']
        for t in range(num_periods):
            d_inv += depot_prod
            d_inv -= sum(sol_periods[t]['deliveries'].values())
            sol_periods[t]['inventories']['depot'] = d_inv
            for c in customers:
                cid = c['id']
                c_inv[cid] += sol_periods[t]['deliveries'].get(cid, 0)
                c_inv[cid] -= cust_demand[cid]
                sol_periods[t]['inventories'][cid] = c_inv[cid]
    
    def try_inter_route_moves(sol_periods):
        improved_any = False
        for t in range(num_periods):
            if time_left() < 1:
                break
            routes = sol_periods[t]['routes']
            deliveries = sol_periods[t]['deliveries']
            if len(routes) < 2:
                continue
            
            improved = True
            while improved:
                improved = False
                if time_left() < 0.5:
                    break
                for ri in range(len(routes)):
                    found = False
                    for ci_idx in range(1, len(routes[ri]) - 1):
                        cid = routes[ri][ci_idx]
                        prev_node = routes[ri][ci_idx - 1]
                        next_node = routes[ri][ci_idx + 1]
                        removal_saving = dist[prev_node][cid] + dist[cid][next_node] - dist[prev_node][next_node]
                        
                        best_net = -0.01
                        best_rj = -1
                        best_pos = -1
                        
                        for rj in range(len(routes)):
                            if rj == ri:
                                continue
                            rj_load = sum(deliveries.get(n, 0) for n in routes[rj] if n != 0)
                            if rj_load + deliveries.get(cid, 0) > vehicle_capacity:
                                continue
                            for pos in range(1, len(routes[rj])):
                                ins_cost = dist[routes[rj][pos-1]][cid] + dist[cid][routes[rj][pos]] - dist[routes[rj][pos-1]][routes[rj][pos]]
                                net = ins_cost - removal_saving
                                if net < best_net:
                                    best_net = net
                                    best_rj = rj
                                    best_pos = pos
                        
                        if best_rj >= 0:
                            routes[ri].pop(ci_idx)
                            # Adjust best_pos if needed (if rj > ri and route changed)
                            routes[best_rj].insert(best_pos, cid)
                            
                            if len(routes[ri]) <= 2:
                                routes.pop(ri)
                                # Adjust indices
                            
                            improved = True
                            improved_any = True
                            found = True
                            break
                    if found:
                        break
        
        return improved_any
    
    # Apply improvements
    if best_sol is not None and best_obj < float('inf'):
        for iteration in range(10):
            if time_left() < 2:
                break
            
            changed = False
            
            if try_reduce_deliveries(best_sol):
                changed = True
            if try_increase_deliveries(best_sol):
                changed = True
            
            recompute_inventories(best_sol)
            improve_all_routes(best_sol)
            
            if try_inter_route_moves(best_sol):
                changed = True
            
            new_obj = evaluate_solution(best_sol)
            if new_obj < best_obj:
                best_obj = new_obj
                if logger:
                    logger.log(best_obj)
            
            if not changed:
                break
    
    # Perturbation-based search
    def perturb_and_improve(sol_periods, rng):
        new_sol = deepcopy(sol_periods)
        
        action = rng.randint(0, 3)
        
        try:
            if action == 0 and num_periods >= 2:
                # Move delivery between periods
                t = rng.randint(0, num_periods - 2)
                cids = list(new_sol[t]['deliveries'].keys())
                if cids:
                    cid = rng.choice(cids)
                    qty = new_sol[t]['deliveries'][cid]
                    if qty <= 0:
                        return None
                    portion = rng.randint(1, max(1, qty))
                    
                    new_sol[t]['deliveries'][cid] -= portion
                    if new_sol[t]['deliveries'][cid] <= 0:
                        del new_sol[t]['deliveries'][cid]
                        for ri, route in enumerate(new_sol[t]['routes']):
                            if cid in route:
                                route.remove(cid)
                                if len(route) <= 2:
                                    new_sol[t]['routes'].pop(ri)
                                break
                    
                    t2 = t + 1
                    if cid in new_sol[t2]['deliveries']:
                        new_sol[t2]['deliveries'][cid] += portion
                    else:
                        new_sol[t2]['deliveries'][cid] = portion
                        added = False
                        for ri, route in enumerate(new_sol[t2]['routes']):
                            rl = sum(new_sol[t2]['deliveries'].get(n, 0) for n in route if n != 0)
                            if rl <= vehicle_capacity and len(route) >= 2:
                                best_pos = 1
                                best_cv = float('inf')
                                for pos in range(1, len(route)):
                                    cv = dist[route[pos-1]][cid] + dist[cid][route[pos]] - dist[route[pos-1]][route[pos]]
                                    if cv < best_cv:
                                        best_cv = cv
                                        best_pos = pos
                                route.insert(best_pos, cid)
                                added = True
                                break
                        if not added:
                            if len(new_sol[t2]['routes']) < num_vehicles:
                                new_sol[t2]['routes'].append([0, cid, 0])
                            else:
                                return None
            
            elif action == 1:
                # Change delivery quantity
                t = rng.randint(0, num_periods - 1)
                cids = list(new_sol[t]['deliveries'].keys())
                if cids:
                    cid = rng.choice(cids)
                    current_qty = new_sol[t]['deliveries'][cid]
                    delta = rng.randint(-max(5, current_qty // 2), max(5, current_qty // 2))
                    new_qty = max(1, current_qty + delta)
                    new_qty = min(new_qty, cust_max_inv[cid] + cust_demand[cid])
                    new_sol[t]['deliveries'][cid] = new_qty
            
            elif action == 2:
                # Rebuild routes for a random period
                t = rng.randint(0, num_periods - 1)
                custs = list(new_sol[t]['deliveries'].keys())
                if custs:
                    if rng.random() < 0.5:
                        new_routes = build_routes_savings(custs, new_sol[t]['deliveries'])
                    else:
                        new_routes = build_routes_nn(custs, new_sol[t]['deliveries'])
                    new_routes = [two_opt(r) for r in new_routes]
                    new_sol[t]['routes'] = new_routes
            
            elif action == 3 and num_periods >= 2:
                # Move delivery earlier
                t = rng.randint(1, num_periods - 1)
                cids = list(new_sol[t]['deliveries'].keys())
                if cids:
                    cid = rng.choice(cids)
                    qty = new_sol[t]['deliveries'][cid]
                    if qty <= 0:
                        return None
                    portion = rng.randint(1, max(1, qty))
                    
                    new_sol[t]['deliveries'][cid] -= portion
                    if new_sol[t]['deliveries'][cid] <= 0:
                        del new_sol[t]['deliveries'][cid]
                        for ri, route in enumerate(new_sol[t]['routes']):
                            if cid in route:
                                route.remove(cid)
                                if len(route) <= 2:
                                    new_sol[t]['routes'].pop(ri)
                                break
                    
                    t2 = t - 1
                    if cid in new_sol[t2]['deliveries']:
                        new_sol[t2]['deliveries'][cid] += portion
                    else:
                        new_sol[t2]['deliveries'][cid] = portion
                        added = False
                        for ri, route in enumerate(new_sol[t2]['routes']):
                            rl = sum(new_sol[t2]['deliveries'].get(n, 0) for n in route if n != 0)
                            if rl <= vehicle_capacity and len(route) >= 2:
                                best_pos = 1
                                best_cv = float('inf')
                                for pos in range(1, len(route)):
                                    cv = dist[route[pos-1]][cid] + dist[cid][route[pos]] - dist[route[pos-1]][route[pos]]
                                    if cv < best_cv:
                                        best_cv = cv
                                        best_pos = pos
                                route.insert(best_pos, cid)
                                added = True
                                break
                        if not added:
                            if len(new_sol[t2]['routes']) < num_vehicles:
                                new_sol[t2]['routes'].append([0, cid, 0])
                            else:
                                return None
            
            recompute_inventories(new_sol)
            obj = evaluate_solution(new_sol)
            if obj < float('inf'):
                improve_all_routes(new_sol)
                obj = evaluate_solution(new_sol)
                if obj < float('inf'):
                    return new_sol, obj
        except Exception:
            pass
        
        return None
    
    if best_sol is not None and best_obj < float('inf'):
        rng = random.Random(12345)
        no_improve_count = 0
        while time_left() > 2 and no_improve_count < 1000:
            result = perturb_and_improve(best_sol, rng)
            if result is not None:
                new_sol, new_obj = result
                if new_obj < best_obj:
                    best_sol = new_sol
                    best_obj = new_obj
                    no_improve_count = 0
                    if logger:
                        logger.log(best_obj)
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
    
    # Format output
    def format_solution(sol_periods, obj_val):
        periods_dict = {}
        for t in range(num_periods):
            period_data = sol_periods[t]
            deliveries_formatted = {}
            for cid, qty in period_data['deliveries'].items():
                deliveries_formatted[str(cid)] = qty
            
            inventories_formatted = {}
            for key, val in period_data['inventories'].items():
                if key == 'depot':
                    inventories_formatted['depot'] = float(val)
                else:
                    inventories_formatted[str(key)] = float(val)
            
            periods_dict[str(t + 1)] = {
                'routes': period_data['routes'],
                'deliveries': deliveries_formatted,
                'inventories': inventories_formatted
            }
        
        return {
            'objective_value': obj_val,
            'solution_details': {
                'periods': periods_dict
            }
        }
    
    if best_sol is not None and best_obj < float('inf'):
        recompute_inventories(best_sol)
        best_obj = evaluate_solution(best_sol)
        result = format_solution(best_sol, best_obj)
    else:
        # Output empty/dummy solution
        periods_dict = {}
        for t in range(num_periods):
            periods_dict[str(t + 1)] = {
                'routes': [],
                'deliveries': {},
                'inventories': {'depot': 0.0}
            }
        result = {
            'objective_value': 0.0,
            'solution_details': {
                'periods': periods_dict
            }
        }
    
    with open(solution_path, 'w') as f:
        json.dump(result, f, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    solve_irp(args.instance_path, args.solution_path, args.time_limit, args.log_path)
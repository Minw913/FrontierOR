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
    
    # Customer indexing: customers are 1-indexed in the data
    # depot is 0
    
    # Precompute customer data
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
    
    def time_left():
        return time_limit - (time.time() - start_time)
    
    # Greedy heuristic: For each period, determine which customers need delivery
    # and how much, then build routes using nearest neighbor
    
    def compute_min_delivery(cid, period, inv):
        """Minimum delivery needed for customer cid in given period with current inventory inv."""
        # Must have inv + delivery - demand >= min_inv
        needed = max(0, cust_min_inv[cid] + cust_demand[cid] - inv)
        return needed
    
    def compute_max_delivery(cid, period, inv):
        """Maximum delivery possible for customer cid."""
        # inv + delivery - demand <= max_inv
        # delivery <= max_inv - inv + demand
        return max(0, cust_max_inv[cid] - inv + cust_demand[cid])
    
    def compute_urgency(cid, period, inv):
        """How many periods until stockout without delivery."""
        remaining = inv
        for t in range(period, num_periods):
            remaining -= cust_demand[cid]
            if remaining < cust_min_inv[cid]:
                return t - period
        return num_periods - period
    
    def build_routes_nn(customers_to_visit, deliveries_dict):
        """Build routes using nearest neighbor heuristic."""
        routes = []
        unvisited = list(customers_to_visit)
        
        while unvisited:
            route = [0]
            load = 0
            current = 0
            remaining = list(unvisited)
            
            while remaining:
                # Find nearest feasible customer
                best_cust = None
                best_dist = float('inf')
                for c in remaining:
                    d = deliveries_dict[c]
                    if load + d <= vehicle_capacity:
                        if dist[current][c] < best_dist:
                            best_dist = dist[current][c]
                            best_cust = c
                
                if best_cust is None:
                    break
                
                route.append(best_cust)
                load += deliveries_dict[best_cust]
                remaining.remove(best_cust)
                unvisited.remove(best_cust)
                current = best_cust
            
            route.append(0)
            routes.append(route)
            
            if len(routes) >= num_vehicles:
                # Force remaining into existing routes if possible, otherwise create overflow
                # For now, just warn and continue
                if unvisited:
                    # Try to fit remaining customers into existing routes
                    still_unvisited = []
                    for c in unvisited:
                        fitted = False
                        for ri, r in enumerate(routes):
                            route_load = sum(deliveries_dict.get(node, 0) for node in r if node != 0)
                            if route_load + deliveries_dict[c] <= vehicle_capacity:
                                # Insert at best position
                                best_pos = 1
                                best_cost = float('inf')
                                for pos in range(1, len(r)):
                                    cost = dist[r[pos-1]][c] + dist[c][r[pos]] - dist[r[pos-1]][r[pos]]
                                    if cost < best_cost:
                                        best_cost = cost
                                        best_pos = pos
                                r.insert(best_pos, c)
                                fitted = True
                                break
                        if not fitted:
                            still_unvisited.append(c)
                    
                    for c in still_unvisited:
                        unvisited.remove(c) if c in unvisited else None
                    
                    if still_unvisited:
                        # Create additional routes anyway (might violate vehicle constraint)
                        for c in still_unvisited:
                            routes.append([0, c, 0])
                break
        
        return routes
    
    def evaluate_solution(sol_periods):
        """Evaluate total cost of a solution."""
        total_cost = 0.0
        depot_inv = depot_init_inv
        cust_inv = [0] * (num_customers + 1)
        for c in customers:
            cust_inv[c['id']] = c['initial_inventory']
        
        for t in range(num_periods):
            period_data = sol_periods[t]
            routes = period_data['routes']
            deliveries = period_data['deliveries']
            
            # Travel cost
            for route in routes:
                for i in range(len(route) - 1):
                    total_cost += dist[route[i]][route[i+1]]
            
            # Depot inventory
            depot_inv += depot_prod
            total_delivered = sum(deliveries.values())
            depot_inv -= total_delivered
            
            if depot_inv < 0 or depot_inv > depot_max_inv:
                return float('inf')  # Infeasible
            
            total_cost += depot_hold * depot_inv
            
            # Customer inventories
            for c in customers:
                cid = c['id']
                cust_inv[cid] += deliveries.get(cid, 0)
                cust_inv[cid] -= cust_demand[cid]
                
                if cust_inv[cid] < cust_min_inv[cid] or cust_inv[cid] > cust_max_inv[cid]:
                    return float('inf')  # Infeasible
                
                total_cost += cust_hold[cid] * cust_inv[cid]
        
        return total_cost
    
    def greedy_solution():
        """Build a greedy solution."""
        sol_periods = []
        depot_inv = depot_init_inv
        cust_inv = {}
        for c in customers:
            cust_inv[c['id']] = c['initial_inventory']
        
        for t in range(num_periods):
            depot_inv += depot_prod
            
            # Determine urgency and needed deliveries
            customer_needs = []
            for c in customers:
                cid = c['id']
                urgency = compute_urgency(cid, t, cust_inv[cid])
                min_del = compute_min_delivery(cid, t, cust_inv[cid])
                max_del = compute_max_delivery(cid, t, cust_inv[cid])
                
                if urgency <= 1 or min_del > 0:
                    # Must deliver this period
                    customer_needs.append((cid, min_del, max_del, 0, urgency))  # priority 0 = must
                elif urgency <= 2:
                    customer_needs.append((cid, min_del, max_del, 1, urgency))  # priority 1 = should
                else:
                    # Check if delivering now saves cost (holding cost comparison)
                    # Simple heuristic: deliver if holding cost at depot is higher
                    if depot_hold > cust_hold[cid] and max_del > 0:
                        customer_needs.append((cid, min_del, max_del, 2, urgency))
            
            # Sort by priority then urgency
            customer_needs.sort(key=lambda x: (x[3], x[4]))
            
            deliveries = {}
            customers_to_visit = []
            available_supply = depot_inv
            
            for cid, min_del, max_del, priority, urgency in customer_needs:
                if available_supply <= 0:
                    break
                
                if priority == 0:
                    # Must deliver
                    del_qty = min(max_del, available_supply, vehicle_capacity)
                    if del_qty < min_del:
                        # Try to deliver minimum
                        del_qty = min(min_del, available_supply)
                    if del_qty > 0:
                        deliveries[cid] = del_qty
                        customers_to_visit.append(cid)
                        available_supply -= del_qty
                elif priority <= 2:
                    # Try to deliver a good amount
                    # Deliver enough for a few periods
                    target = min(max_del, cust_demand[cid] * min(3, num_periods - t))
                    target = min(target, cust_max_inv[cid] - cust_inv[cid] + cust_demand[cid])
                    del_qty = min(target, available_supply, vehicle_capacity)
                    if del_qty > 0:
                        deliveries[cid] = del_qty
                        customers_to_visit.append(cid)
                        available_supply -= del_qty
            
            # Ensure depot inventory doesn't go negative
            total_del = sum(deliveries.values())
            if total_del > depot_inv:
                # Scale down deliveries
                scale = depot_inv / total_del
                for cid in deliveries:
                    deliveries[cid] = int(deliveries[cid] * scale)
                # Recalculate
                total_del = sum(deliveries.values())
            
            # Ensure minimum deliveries for urgent customers
            # (already handled above approximately)
            
            # Build routes
            routes = build_routes_nn(customers_to_visit, deliveries) if customers_to_visit else []
            
            # Limit routes to num_vehicles
            while len(routes) > num_vehicles:
                # Merge smallest routes
                routes.sort(key=lambda r: len(r))
                small = routes.pop(0)
                # Try to merge into another route
                merged = False
                small_custs = [n for n in small if n != 0]
                small_load = sum(deliveries.get(n, 0) for n in small_custs)
                for ri in range(len(routes)):
                    r = routes[ri]
                    r_load = sum(deliveries.get(n, 0) for n in r if n != 0)
                    if r_load + small_load <= vehicle_capacity:
                        # Insert customers
                        for c in small_custs:
                            best_pos = 1
                            best_cost = float('inf')
                            for pos in range(1, len(r)):
                                cost = dist[r[pos-1]][c] + dist[c][r[pos]] - dist[r[pos-1]][r[pos]]
                                if cost < best_cost:
                                    best_cost = cost
                                    best_pos = pos
                            r.insert(best_pos, c)
                        merged = True
                        break
                if not merged:
                    # Drop some customers from the smallest route
                    for c in small_custs:
                        if deliveries[c] == 0 or compute_urgency(c, t, cust_inv[c]) > 1:
                            del deliveries[c]
                            customers_to_visit.remove(c) if c in customers_to_visit else None
                    if small_custs:
                        # If all are urgent, we have a problem - just keep extra route
                        routes.insert(0, small)
                        break
            
            # Update inventories
            depot_inv -= sum(deliveries.values())
            
            inventories = {'depot': depot_inv}
            for c in customers:
                cid = c['id']
                cust_inv[cid] += deliveries.get(cid, 0)
                cust_inv[cid] -= cust_demand[cid]
                inventories[cid] = cust_inv[cid]
            
            sol_periods.append({
                'routes': routes,
                'deliveries': deliveries,
                'inventories': inventories
            })
        
        return sol_periods
    
    def format_solution(sol_periods, obj_val):
        """Format solution according to schema."""
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
    
    # Phase 1: Build initial greedy solution
    def build_careful_greedy():
        """Build a more careful greedy solution using look-ahead."""
        sol_periods = []
        depot_inv = depot_init_inv
        cust_inv = {}
        for c in customers:
            cust_inv[c['id']] = c['initial_inventory']
        
        for t in range(num_periods):
            depot_inv += depot_prod
            
            # For each customer, compute: min delivery needed now, max possible, 
            # and "optimal" delivery quantity considering holding costs
            customer_info = []
            for c in customers:
                cid = c['id']
                
                # Check how many periods customer can survive without delivery
                inv_temp = cust_inv[cid]
                periods_ok = 0
                for tt in range(t, num_periods):
                    inv_temp -= cust_demand[cid]
                    if inv_temp < cust_min_inv[cid]:
                        break
                    periods_ok += 1
                
                min_del = max(0, cust_min_inv[cid] + cust_demand[cid] - cust_inv[cid])
                max_del = max(0, cust_max_inv[cid] - cust_inv[cid] + cust_demand[cid])
                
                customer_info.append({
                    'id': cid,
                    'min_del': min_del,
                    'max_del': max_del,
                    'periods_ok': periods_ok,
                    'inv': cust_inv[cid]
                })
            
            # Sort: most urgent first
            customer_info.sort(key=lambda x: (x['periods_ok'], -x['min_del']))
            
            deliveries = {}
            customers_to_visit = []
            available_supply = depot_inv
            
            # First pass: handle mandatory deliveries
            for ci in customer_info:
                cid = ci['id']
                if ci['min_del'] > 0:
                    del_qty = min(ci['max_del'], available_supply)
                    if del_qty >= ci['min_del']:
                        # Deliver up to fill or a reasonable amount
                        # Consider: deliver enough for multiple periods to reduce future trips
                        target = ci['min_del']
                        # Try to deliver more if holding cost is favorable
                        remaining_demand = cust_demand[cid] * (num_periods - t)
                        optimal_extra = 0
                        if depot_hold >= cust_hold[cid]:
                            # Cheaper to hold at customer
                            optimal_extra = ci['max_del'] - ci['min_del']
                        
                        del_qty = min(ci['min_del'] + optimal_extra, ci['max_del'], available_supply)
                        deliveries[cid] = del_qty
                        customers_to_visit.append(cid)
                        available_supply -= del_qty
                    else:
                        # Deliver what we can
                        if del_qty > 0:
                            deliveries[cid] = del_qty
                            customers_to_visit.append(cid)
                            available_supply -= del_qty
            
            # Second pass: proactive deliveries for soon-to-be-urgent customers
            for ci in customer_info:
                cid = ci['id']
                if cid in deliveries:
                    continue
                if available_supply <= 0:
                    break
                
                if ci['periods_ok'] <= 1:
                    # Will need delivery next period - consider delivering now
                    target = min(ci['max_del'], available_supply)
                    if target > 0:
                        deliveries[cid] = target
                        customers_to_visit.append(cid)
                        available_supply -= target
            
            # Third pass: if depot is getting full, push inventory to customers
            if depot_inv > depot_max_inv * 0.8:
                for ci in customer_info:
                    cid = ci['id']
                    if cid in deliveries:
                        continue
                    if available_supply <= 0:
                        break
                    if ci['max_del'] > 0 and depot_hold >= cust_hold[cid]:
                        target = min(ci['max_del'], available_supply)
                        if target > 0:
                            deliveries[cid] = target
                            customers_to_visit.append(cid)
                            available_supply -= target
            
            # Check depot capacity constraint: depot_inv - total_del >= 0
            total_del = sum(deliveries.values())
            if total_del > depot_inv:
                # Need to reduce deliveries - reduce non-urgent ones first
                excess = total_del - depot_inv
                # Remove deliveries in reverse priority (least urgent first)
                for ci in reversed(customer_info):
                    if excess <= 0:
                        break
                    cid = ci['id']
                    if cid not in deliveries:
                        continue
                    if ci['min_del'] == 0:
                        reduction = min(deliveries[cid], excess)
                        deliveries[cid] -= reduction
                        excess -= reduction
                        if deliveries[cid] == 0:
                            del deliveries[cid]
                            customers_to_visit.remove(cid)
                
                # If still excess, reduce amounts for non-critical
                if excess > 0:
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
                                customers_to_visit.remove(cid)
            
            # Also check depot doesn't go below 0 after future production
            # Check depot max inventory constraint for future periods
            future_depot_inv = depot_inv - sum(deliveries.values())
            if future_depot_inv + depot_prod > depot_max_inv and t < num_periods - 1:
                # Need to deliver more to avoid overflow next period
                overflow = future_depot_inv + depot_prod - depot_max_inv
                for ci in customer_info:
                    if overflow <= 0:
                        break
                    cid = ci['id']
                    if cid in deliveries:
                        # Can we increase?
                        current_del = deliveries[cid]
                        max_increase = ci['max_del'] - current_del
                        if max_increase > 0:
                            increase = min(max_increase, overflow)
                            deliveries[cid] += increase
                            overflow -= increase
                    else:
                        if ci['max_del'] > 0:
                            del_qty = min(ci['max_del'], overflow)
                            deliveries[cid] = del_qty
                            customers_to_visit.append(cid)
                            overflow -= del_qty
            
            # Build routes
            if customers_to_visit:
                routes = build_routes_nn(customers_to_visit, deliveries)
            else:
                routes = []
            
            # Ensure we don't exceed vehicle limit
            while len(routes) > num_vehicles:
                # Try merging
                routes.sort(key=lambda r: sum(deliveries.get(n, 0) for n in r if n != 0))
                small = routes.pop(0)
                small_custs = [n for n in small if n != 0]
                small_load = sum(deliveries.get(n, 0) for n in small_custs)
                merged = False
                for ri in range(len(routes)):
                    r = routes[ri]
                    r_load = sum(deliveries.get(n, 0) for n in r if n != 0)
                    if r_load + small_load <= vehicle_capacity:
                        for cc in small_custs:
                            best_pos = 1
                            best_cost = float('inf')
                            for pos in range(1, len(r)):
                                cost = dist[r[pos-1]][cc] + dist[cc][r[pos]] - dist[r[pos-1]][r[pos]]
                                if cost < best_cost:
                                    best_cost = cost
                                    best_pos = pos
                            r.insert(best_pos, cc)
                        merged = True
                        break
                if not merged:
                    # Drop least urgent customers from the small route
                    dropped = False
                    for cc in sorted(small_custs, key=lambda c: -compute_urgency(c, t, cust_inv[c])):
                        ci_info = next(x for x in customer_info if x['id'] == cc)
                        if ci_info['min_del'] == 0:
                            del deliveries[cc]
                            if cc in customers_to_visit:
                                customers_to_visit.remove(cc)
                            small_custs.remove(cc)
                            dropped = True
                            break
                    if not dropped:
                        # Can't drop, keep the route
                        routes.insert(0, small)
                        break
                    else:
                        if small_custs:
                            # Re-add remaining
                            new_route = [0] + small_custs + [0]
                            routes.append(new_route)
            
            # Update inventories
            total_del = sum(deliveries.values())
            depot_inv -= total_del
            
            inventories = {'depot': depot_inv}
            for c in customers:
                cid = c['id']
                cust_inv[cid] += deliveries.get(cid, 0)
                cust_inv[cid] -= cust_demand[cid]
                inventories[cid] = cust_inv[cid]
            
            sol_periods.append({
                'routes': routes,
                'deliveries': deliveries,
                'inventories': inventories
            })
        
        return sol_periods
    
    # Try to build a feasible solution
    sol = build_careful_greedy()
    obj = evaluate_solution(sol)
    
    if obj == float('inf'):
        # Try alternative: deliver maximum to everyone as much as possible
        sol = greedy_solution()
        obj = evaluate_solution(sol)
    
    if obj < float('inf'):
        best_sol = deepcopy(sol)
        best_obj = obj
        if logger:
            logger.log(best_obj)
    else:
        best_sol = sol
        best_obj = obj
    
    # Phase 2: Improvement via local search
    def route_cost(route):
        cost = 0
        for i in range(len(route) - 1):
            cost += dist[route[i]][route[i+1]]
        return cost
    
    def two_opt(route):
        """Apply 2-opt improvement to a route."""
        if len(route) <= 4:  # [0, a, b, 0] minimum for 2-opt
            return route
        improved = True
        best_route = list(route)
        while improved:
            improved = False
            for i in range(1, len(best_route) - 2):
                for j in range(i + 1, len(best_route) - 1):
                    # Reverse segment [i, j]
                    new_route = best_route[:i] + best_route[i:j+1][::-1] + best_route[j+1:]
                    if route_cost(new_route) < route_cost(best_route):
                        best_route = new_route
                        improved = True
        return best_route
    
    def improve_routes(sol_periods):
        """Apply 2-opt to all routes."""
        improved = False
        for t in range(num_periods):
            new_routes = []
            for route in sol_periods[t]['routes']:
                new_route = two_opt(route)
                if route_cost(new_route) < route_cost(route):
                    improved = True
                new_routes.append(new_route)
            sol_periods[t]['routes'] = new_routes
        return improved
    
    def or_opt(route):
        """Apply or-opt (relocate single customer) improvement."""
        if len(route) <= 3:
            return route
        best_route = list(route)
        best_cost = route_cost(best_route)
        improved = True
        while improved:
            improved = False
            for i in range(1, len(best_route) - 1):
                node = best_route[i]
                # Remove node
                temp = best_route[:i] + best_route[i+1:]
                # Try reinserting at every position
                for j in range(1, len(temp)):
                    new_route = temp[:j] + [node] + temp[j:]
                    nc = route_cost(new_route)
                    if nc < best_cost - 1e-10:
                        best_route = new_route
                        best_cost = nc
                        improved = True
                        break
                if improved:
                    break
        return best_route
    
    def improve_routes_or_opt(sol_periods):
        improved = False
        for t in range(num_periods):
            new_routes = []
            for route in sol_periods[t]['routes']:
                new_route = or_opt(route)
                if route_cost(new_route) < route_cost(route):
                    improved = True
                new_routes.append(new_route)
            sol_periods[t]['routes'] = new_routes
        return improved
    
    # Apply route improvements
    if best_obj < float('inf') and time_left() > 1:
        improve_routes(best_sol)
        improve_routes_or_opt(best_sol)
        new_obj = evaluate_solution(best_sol)
        if new_obj < best_obj:
            best_obj = new_obj
            if logger:
                logger.log(best_obj)
    
    # Phase 3: Try different delivery strategies
    def build_solution_with_strategy(strategy='fill', seed_val=None):
        """Build solution with different delivery strategies."""
        if seed_val is not None:
            rng = random.Random(seed_val)
        else:
            rng = random.Random(42)
        
        sol_periods = []
        depot_inv = depot_init_inv
        cust_inv = {}
        for c in customers:
            cust_inv[c['id']] = c['initial_inventory']
        
        for t in range(num_periods):
            depot_inv += depot_prod
            
            customer_info = []
            for c in customers:
                cid = c['id']
                
                inv_temp = cust_inv[cid]
                periods_ok = 0
                for tt in range(t, num_periods):
                    inv_temp -= cust_demand[cid]
                    if inv_temp < cust_min_inv[cid]:
                        break
                    periods_ok += 1
                
                min_del = max(0, cust_min_inv[cid] + cust_demand[cid] - cust_inv[cid])
                max_del = max(0, cust_max_inv[cid] - cust_inv[cid] + cust_demand[cid])
                
                customer_info.append({
                    'id': cid,
                    'min_del': min_del,
                    'max_del': max_del,
                    'periods_ok': periods_ok,
                    'inv': cust_inv[cid],
                    'demand': cust_demand[cid],
                    'hold_cost': cust_hold[cid]
                })
            
            customer_info.sort(key=lambda x: (x['periods_ok'], -x['min_del']))
            
            deliveries = {}
            customers_to_visit = []
            available_supply = depot_inv
            
            # Determine delivery threshold
            if strategy == 'fill':
                threshold = 0  # deliver when needed, fill to max
            elif strategy == 'lot_for_lot':
                threshold = 1  # deliver just what's needed
            elif strategy == 'balanced':
                threshold = 2  # moderate fill
            elif strategy == 'random':
                threshold = rng.randint(0, 3)
            else:
                threshold = 1
            
            for ci in customer_info:
                cid = ci['id']
                if available_supply <= 0:
                    break
                
                needs_delivery = False
                if ci['min_del'] > 0:
                    needs_delivery = True
                elif ci['periods_ok'] <= threshold:
                    needs_delivery = True
                elif depot_inv > depot_max_inv * 0.9 and ci['max_del'] > 0:
                    needs_delivery = True
                
                # Check if depot will overflow next period
                future_depot = depot_inv - sum(deliveries.values())
                if t < num_periods - 1 and future_depot + depot_prod > depot_max_inv:
                    if ci['max_del'] > 0:
                        needs_delivery = True
                
                if needs_delivery and ci['max_del'] > 0:
                    if strategy == 'fill':
                        del_qty = min(ci['max_del'], available_supply)
                    elif strategy == 'lot_for_lot':
                        # Deliver for ~2 periods
                        target = ci['demand'] * 2 if ci['min_del'] == 0 else ci['min_del'] + ci['demand']
                        del_qty = min(target, ci['max_del'], available_supply)
                        del_qty = max(del_qty, ci['min_del'])
                    elif strategy == 'balanced':
                        # Deliver to reach about 60% capacity
                        target_inv = int(cust_max_inv[cid] * 0.6)
                        needed = target_inv - ci['inv'] + ci['demand']
                        del_qty = min(max(needed, ci['min_del']), ci['max_del'], available_supply)
                    elif strategy == 'random':
                        min_q = ci['min_del']
                        max_q = min(ci['max_del'], available_supply)
                        if max_q >= min_q:
                            del_qty = rng.randint(min_q, max_q)
                        else:
                            del_qty = min_q
                    else:
                        del_qty = min(ci['max_del'], available_supply)
                    
                    del_qty = min(del_qty, available_supply)
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
                            deliveries[cid] = del_qty
                            customers_to_visit.append(cid)
                            overflow -= del_qty
            
            # Build routes
            if customers_to_visit:
                routes = build_routes_nn(customers_to_visit, deliveries)
            else:
                routes = []
            
            # Limit to num_vehicles
            if len(routes) > num_vehicles:
                # Try merging
                while len(routes) > num_vehicles:
                    routes.sort(key=lambda r: sum(deliveries.get(n, 0) for n in r if n != 0))
                    small = routes.pop(0)
                    small_custs = [n for n in small if n != 0]
                    small_load = sum(deliveries.get(n, 0) for n in small_custs)
                    merged = False
                    for ri in range(len(routes)):
                        r = routes[ri]
                        r_load = sum(deliveries.get(n, 0) for n in r if n != 0)
                        if r_load + small_load <= vehicle_capacity:
                            for cc in small_custs:
                                best_pos = 1
                                best_cost_val = float('inf')
                                for pos in range(1, len(r)):
                                    cost_val = dist[r[pos-1]][cc] + dist[cc][r[pos]] - dist[r[pos-1]][r[pos]]
                                    if cost_val < best_cost_val:
                                        best_cost_val = cost_val
                                        best_pos = pos
                                r.insert(best_pos, cc)
                            merged = True
                            break
                    if not merged:
                        # Drop non-urgent customers
                        dropped_any = False
                        for cc in sorted(small_custs, key=lambda c: -compute_urgency(c, t, cust_inv[c])):
                            ci_info = next((x for x in customer_info if x['id'] == cc), None)
                            if ci_info and ci_info['min_del'] == 0:
                                del deliveries[cc]
                                small_custs.remove(cc)
                                if cc in customers_to_visit:
                                    customers_to_visit.remove(cc)
                                dropped_any = True
                                break
                        if not dropped_any:
                            routes.insert(0, small)
                            break
                        elif small_custs:
                            new_route = [0] + small_custs + [0]
                            routes.append(new_route)
            
            # Apply 2-opt to routes
            routes = [two_opt(r) for r in routes]
            
            # Update inventories
            depot_inv -= sum(deliveries.values())
            
            inventories = {'depot': depot_inv}
            for c in customers:
                cid = c['id']
                cust_inv[cid] += deliveries.get(cid, 0)
                cust_inv[cid] -= cust_demand[cid]
                inventories[cid] = cust_inv[cid]
            
            sol_periods.append({
                'routes': routes,
                'deliveries': deliveries,
                'inventories': inventories
            })
        
        return sol_periods
    
    # Try different strategies
    for strategy in ['fill', 'lot_for_lot', 'balanced']:
        if time_left() < 2:
            break
        try:
            sol = build_solution_with_strategy(strategy)
            obj = evaluate_solution(sol)
            if obj < best_obj:
                best_sol = deepcopy(sol)
                best_obj = obj
                if logger:
                    logger.log(best_obj)
        except:
            pass
    
    # Try random strategies
    for seed_val in range(100):
        if time_left() < 2:
            break
        try:
            sol = build_solution_with_strategy('random', seed_val=seed_val)
            obj = evaluate_solution(sol)
            if obj < best_obj:
                best_sol = deepcopy(sol)
                best_obj = obj
                if logger:
                    logger.log(best_obj)
        except:
            pass
    
    # Phase 4: Local search on delivery quantities
    def try_adjust_deliveries(sol_periods):
        """Try adjusting delivery quantities to reduce holding costs."""
        # Recompute inventories to verify
        depot_inv = depot_init_inv
        cust_inv = {}
        for c in customers:
            cust_inv[c['id']] = c['initial_inventory']
        
        improved = False
        
        for t in range(num_periods):
            depot_inv += depot_prod
            deliveries = sol_periods[t]['deliveries']
            depot_inv -= sum(deliveries.values())
            
            for c in customers:
                cid = c['id']
                cust_inv[cid] += deliveries.get(cid, 0)
                cust_inv[cid] -= cust_demand[cid]
        
        return improved
    
    # Phase 5: Inter-route and inter-period customer moves
    def try_move_customer_between_periods(sol_periods):
        """Try moving a customer's delivery from one period to another."""
        nonlocal best_obj, best_sol
        
        # Recompute full state
        depot_invs = []
        cust_invs = []
        depot_inv = depot_init_inv
        ci = {}
        for c in customers:
            ci[c['id']] = c['initial_inventory']
        
        for t in range(num_periods):
            depot_inv += depot_prod
            depot_inv -= sum(sol_periods[t]['deliveries'].values())
            depot_invs.append(depot_inv)
            
            ci_copy = {}
            for c in customers:
                cid = c['id']
                ci[cid] += sol_periods[t]['deliveries'].get(cid, 0)
                ci[cid] -= cust_demand[cid]
                ci_copy[cid] = ci[cid]
            cust_invs.append(ci_copy)
        
        # Try removing a customer from a period (postpone delivery)
        improved = False
        for t in range(num_periods - 1):
            if time_left() < 1:
                return improved
            for cid in list(sol_periods[t]['deliveries'].keys()):
                qty = sol_periods[t]['deliveries'][cid]
                
                # Check if we can move this delivery to period t+1
                # Check customer feasibility: would customer survive period t without delivery?
                if t == 0:
                    prev_inv = next(c['initial_inventory'] for c in customers if c['id'] == cid)
                else:
                    prev_inv = cust_invs[t-1][cid]
                
                # Without delivery in period t:
                new_inv_t = prev_inv - cust_demand[cid]
                if new_inv_t < cust_min_inv[cid]:
                    continue  # Can't remove
                
                # With delivery in period t+1 instead:
                new_inv_t1 = new_inv_t + qty - cust_demand[cid]
                if new_inv_t1 > cust_max_inv[cid] or new_inv_t1 < cust_min_inv[cid]:
                    continue
                
                # Check depot: removing delivery from t means more inventory at depot in t
                # Adding to t+1 means less inventory
                new_depot_t = depot_invs[t] + qty
                if new_depot_t > depot_max_inv:
                    continue
                new_depot_t1 = depot_invs[t+1] - qty
                if new_depot_t1 < 0:
                    continue
                
                # Check vehicle capacity in period t+1
                total_t1 = sum(sol_periods[t+1]['deliveries'].values()) + qty
                # Rough check - more precise would check route capacity
                
                # Calculate cost change
                # Old holding cost contribution from this customer
                old_hold = cust_hold[cid] * cust_invs[t][cid] + cust_hold[cid] * cust_invs[t+1][cid]
                new_hold = cust_hold[cid] * new_inv_t + cust_hold[cid] * new_inv_t1
                
                # Depot holding cost change
                old_depot_hold = depot_hold * depot_invs[t] + depot_hold * depot_invs[t+1]
                new_depot_hold = depot_hold * new_depot_t + depot_hold * new_depot_t1
                
                hold_savings = (old_hold + old_depot_hold) - (new_hold + new_depot_hold)
                
                # Route cost change: rough estimate
                # Removing customer from route in t saves ~2*dist[0][cid]
                # Adding to route in t+1 costs ~2*dist[0][cid]
                # So net route cost change is roughly 0, but could be different
                
                # Only do this if holding savings are positive
                if hold_savings > 0:
                    # Try the move
                    new_sol = deepcopy(sol_periods)
                    del new_sol[t]['deliveries'][cid]
                    
                    # Remove customer from route in t
                    for ri, route in enumerate(new_sol[t]['routes']):
                        if cid in route:
                            idx = route.index(cid)
                            route.pop(idx)
                            if len(route) == 2 and route[0] == 0 and route[1] == 0:
                                new_sol[t]['routes'].pop(ri)
                            break
                    
                    # Add delivery in t+1
                    if cid in new_sol[t+1]['deliveries']:
                        new_sol[t+1]['deliveries'][cid] += qty
                    else:
                        new_sol[t+1]['deliveries'][cid] = qty
                        # Add to a route or create new one
                        added = False
                        for ri, route in enumerate(new_sol[t+1]['routes']):
                            r_load = sum(new_sol[t+1]['deliveries'].get(n, 0) for n in route if n != 0)
                            if r_load <= vehicle_capacity:
                                # Insert at best position
                                best_pos = 1
                                best_cost_val = float('inf')
                                for pos in range(1, len(route)):
                                    cv = dist[route[pos-1]][cid] + dist[cid][route[pos]] - dist[route[pos-1]][route[pos]]
                                    if cv < best_cost_val:
                                        best_cost_val = cv
                                        best_pos = pos
                                route.insert(best_pos, cid)
                                added = True
                                break
                        if not added:
                            if len(new_sol[t+1]['routes']) < num_vehicles:
                                new_sol[t+1]['routes'].append([0, cid, 0])
                            else:
                                continue  # Can't add route
                    
                    # Recompute inventories
                    d_inv = depot_init_inv
                    c_inv = {}
                    for c in customers:
                        c_inv[c['id']] = c['initial_inventory']
                    
                    valid = True
                    for tt in range(num_periods):
                        d_inv += depot_prod
                        d_inv -= sum(new_sol[tt]['deliveries'].values())
                        if d_inv < 0 or d_inv > depot_max_inv:
                            valid = False
                            break
                        new_sol[tt]['inventories']['depot'] = d_inv
                        
                        for c in customers:
                            ccid = c['id']
                            c_inv[ccid] += new_sol[tt]['deliveries'].get(ccid, 0)
                            c_inv[ccid] -= cust_demand[ccid]
                            if c_inv[ccid] < cust_min_inv[ccid] or c_inv[ccid] > cust_max_inv[ccid]:
                                valid = False
                                break
                            new_sol[tt]['inventories'][ccid] = c_inv[ccid]
                        if not valid:
                            break
                    
                    if valid:
                        new_obj = evaluate_solution(new_sol)
                        if new_obj < best_obj:
                            best_obj = new_obj
                            best_sol = new_sol
                            sol_periods = new_sol
                            improved = True
                            if logger:
                                logger.log(best_obj)
                            # Recompute depot_invs and cust_invs
                            depot_invs = []
                            cust_invs_new = []
                            d_inv2 = depot_init_inv
                            c_inv2 = {}
                            for c in customers:
                                c_inv2[c['id']] = c['initial_inventory']
                            for tt in range(num_periods):
                                d_inv2 += depot_prod
                                d_inv2 -= sum(sol_periods[tt]['deliveries'].values())
                                depot_invs.append(d_inv2)
                                ci_copy = {}
                                for c in customers:
                                    ccid = c['id']
                                    c_inv2[ccid] += sol_periods[tt]['deliveries'].get(ccid, 0)
                                    c_inv2[ccid] -= cust_demand[ccid]
                                    ci_copy[ccid] = c_inv2[ccid]
                                cust_invs_new.append(ci_copy)
                            cust_invs = cust_invs_new
        
        return improved
    
    # Apply inter-period moves
    if best_obj < float('inf') and time_left() > 2:
        try_move_customer_between_periods(best_sol)
    
    # Apply final route improvements
    if best_obj < float('inf') and time_left() > 1:
        improve_routes(best_sol)
        improve_routes_or_opt(best_sol)
        new_obj = evaluate_solution(best_sol)
        if new_obj < best_obj:
            best_obj = new_obj
            if logger:
                logger.log(best_obj)
    
    # Phase 6: Try inter-route customer moves within same period
    def try_inter_route_moves(sol_periods):
        """Try moving customers between routes in the same period."""
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
                for ri in range(len(routes)):
                    if time_left() < 0.5:
                        break
                    for ci_idx in range(1, len(routes[ri]) - 1):
                        cid = routes[ri][ci_idx]
                        # Cost of removing cid from route ri
                        prev_node = routes[ri][ci_idx - 1]
                        next_node = routes[ri][ci_idx + 1]
                        removal_saving = dist[prev_node][cid] + dist[cid][next_node] - dist[prev_node][next_node]
                        
                        # Try inserting into another route
                        best_insertion_cost = 0  # Must improve
                        best_rj = -1
                        best_pos = -1
                        
                        for rj in range(len(routes)):
                            if rj == ri:
                                continue
                            # Check capacity
                            rj_load = sum(deliveries.get(n, 0) for n in routes[rj] if n != 0)
                            if rj_load + deliveries.get(cid, 0) > vehicle_capacity:
                                continue
                            
                            for pos in range(1, len(routes[rj])):
                                insertion_cost = dist[routes[rj][pos-1]][cid] + dist[cid][routes[rj][pos]] - dist[routes[rj][pos-1]][routes[rj][pos]]
                                net_cost = insertion_cost - removal_saving
                                if net_cost < best_insertion_cost:
                                    best_insertion_cost = net_cost
                                    best_rj = rj
                                    best_pos = pos
                        
                        if best_rj >= 0:
                            # Perform the move
                            routes[ri].pop(ci_idx)
                            routes[best_rj].insert(best_pos, cid)
                            
                            # Remove empty routes
                            if len(routes[ri]) == 2:  # [0, 0]
                                routes.pop(ri)
                            
                            improved = True
                            improved_any = True
                            break
                    if improved:
                        break
        
        return improved_any
    
    if best_obj < float('inf') and time_left() > 2:
        if try_inter_route_moves(best_sol):
            new_obj = evaluate_solution(best_sol)
            if new_obj < best_obj:
                best_obj = new_obj
                if logger:
                    logger.log(best_obj)
    
    # Final route improvement
    if best_obj < float('inf') and time_left() > 1:
        improve_routes(best_sol)
        improve_routes_or_opt(best_sol)
        new_obj = evaluate_solution(best_sol)
        if new_obj < best_obj:
            best_obj = new_obj
            if logger:
                logger.log(best_obj)
    
    # Phase 7: Try reducing delivery quantities (to reduce holding costs)
    def try_reduce_deliveries(sol_periods):
        """Try reducing delivery quantities to customers to save holding costs."""
        improved_any = False
        
        for iteration in range(5):
            if time_left() < 1:
                break
            
            # Recompute inventories
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
                cust_inv_arr.append(ci_copy)
            
            improved = False
            for t in range(num_periods):
                if time_left() < 0.5:
                    break
                for cid in list(sol_periods[t]['deliveries'].keys()):
                    qty = sol_periods[t]['deliveries'][cid]
                    if qty <= 0:
                        continue
                    
                    # How much can we reduce?
                    # Check customer inventory from t onwards
                    min_reduce = 0
                    max_reduce = qty
                    
                    # Customer's inventory at end of period t with reduction r is:
                    # cust_inv_arr[t][cid] - r
                    # Must remain >= min_inv for all subsequent periods until next delivery
                    
                    # Find next delivery period for this customer after t
                    next_del_period = num_periods
                    for tt in range(t + 1, num_periods):
                        if cid in sol_periods[tt]['deliveries'] and sol_periods[tt]['deliveries'][cid] > 0:
                            next_del_period = tt
                            break
                    
                    # Check how much we can reduce
                    slack = float('inf')
                    for tt in range(t, next_del_period):
                        slack = min(slack, cust_inv_arr[tt][cid] - cust_min_inv[cid])
                    
                    reduce = min(int(slack), qty)
                    
                    # Also check depot: reducing delivery by r increases depot inv for periods t onwards
                    for tt in range(t, num_periods):
                        depot_slack = depot_max_inv - depot_inv_arr[tt]
                        reduce = min(reduce, depot_slack)
                    
                    if reduce <= 0:
                        continue
                    
                    # Cost savings: holding cost reduction at customer, increase at depot
                    # For periods t to next_del_period-1, customer holds r less
                    cust_save = cust_hold[cid] * reduce * (next_del_period - t)
                    depot_cost = depot_hold * reduce * (num_periods - t)
                    
                    if cust_save > depot_cost:
                        # Apply reduction
                        sol_periods[t]['deliveries'][cid] -= reduce
                        if sol_periods[t]['deliveries'][cid] == 0:
                            del sol_periods[t]['deliveries'][cid]
                            # Remove from route
                            for ri, route in enumerate(sol_periods[t]['routes']):
                                if cid in route:
                                    idx = route.index(cid)
                                    route.pop(idx)
                                    if len(route) == 2:
                                        sol_periods[t]['routes'].pop(ri)
                                    break
                        
                        # Update inventory arrays
                        for tt in range(t, num_periods):
                            depot_inv_arr[tt] += reduce
                        for tt in range(t, next_del_period):
                            cust_inv_arr[tt][cid] -= reduce
                        
                        improved = True
                        improved_any = True
            
            if not improved:
                break
        
        return improved_any
    
    if best_obj < float('inf') and time_left() > 2:
        if try_reduce_deliveries(best_sol):
            # Recompute inventories in sol
            d_inv = depot_init_inv
            c_inv = {}
            for c in customers:
                c_inv[c['id']] = c['initial_inventory']
            for t in range(num_periods):
                d_inv += depot_prod
                d_inv -= sum(best_sol[t]['deliveries'].values())
                best_sol[t]['inventories']['depot'] = d_inv
                for c in customers:
                    cid = c['id']
                    c_inv[cid] += best_sol[t]['deliveries'].get(cid, 0)
                    c_inv[cid] -= cust_demand[cid]
                    best_sol[t]['inventories'][cid] = c_inv[cid]
            
            improve_routes(best_sol)
            improve_routes_or_opt(best_sol)
            new_obj = evaluate_solution(best_sol)
            if new_obj < best_obj:
                best_obj = new_obj
                if logger:
                    logger.log(best_obj)
    
    # Phase 8: Try increasing deliveries (to push inventory to cheaper locations)
    def try_increase_deliveries(sol_periods):
        """Try increasing delivery quantities when depot holding cost > customer holding cost."""
        improved_any = False
        
        # Recompute inventories
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
            cust_inv_arr.append(ci_copy)
        
        for t in range(num_periods):
            if time_left() < 1:
                break
            for cid_data in customers:
                cid = cid_data['id']
                if cid not in sol_periods[t]['deliveries']:
                    continue
                
                if depot_hold <= cust_hold[cid]:
                    continue  # No benefit
                
                # How much can we increase?
                # Find next delivery period
                next_del_period = num_periods
                for tt in range(t + 1, num_periods):
                    if cid in sol_periods[tt]['deliveries'] and sol_periods[tt]['deliveries'][cid] > 0:
                        next_del_period = tt
                        break
                
                # Customer capacity slack from t to next_del_period-1
                cust_slack = float('inf')
                for tt in range(t, next_del_period):
                    cust_slack = min(cust_slack, cust_max_inv[cid] - cust_inv_arr[tt][cid])
                
                # Depot supply slack from t onwards (depot must stay >= 0)
                depot_slack = float('inf')
                for tt in range(t, num_periods):
                    depot_slack = min(depot_slack, depot_inv_arr[tt])
                
                # Vehicle capacity slack
                route_containing = None
                for ri, route in enumerate(sol_periods[t]['routes']):
                    if cid in route:
                        route_load = sum(sol_periods[t]['deliveries'].get(n, 0) for n in route if n != 0)
                        vehicle_slack = vehicle_capacity - route_load
                        route_containing = ri
                        break
                else:
                    vehicle_slack = 0
                
                increase = min(int(cust_slack), int(depot_slack), vehicle_slack)
                if increase <= 0:
                    continue
                
                # Cost: holding at customer increases, holding at depot decreases
                periods_at_cust = next_del_period - t
                cust_extra_cost = cust_hold[cid] * increase * periods_at_cust
                periods_at_depot = num_periods - t
                depot_savings = depot_hold * increase * periods_at_depot
                
                if depot_savings > cust_extra_cost:
                    sol_periods[t]['deliveries'][cid] += increase
                    for tt in range(t, num_periods):
                        depot_inv_arr[tt] -= increase
                    for tt in range(t, next_del_period):
                        cust_inv_arr[tt][cid] += increase
                    improved_any = True
        
        return improved_any
    
    if best_obj < float('inf') and time_left() > 2:
        if try_increase_deliveries(best_sol):
            d_inv = depot_init_inv
            c_inv = {}
            for c in customers:
                c_inv[c['id']] = c['initial_inventory']
            for t in range(num_periods):
                d_inv += depot_prod
                d_inv -= sum(best_sol[t]['deliveries'].values())
                best_sol[t]['inventories']['depot'] = d_inv
                for c in customers:
                    cid = c['id']
                    c_inv[cid] += best_sol[t]['deliveries'].get(cid, 0)
                    c_inv[cid] -= cust_demand[cid]
                    best_sol[t]['inventories'][cid] = c_inv[cid]
            
            new_obj = evaluate_solution(best_sol)
            if new_obj < best_obj:
                best_obj = new_obj
                if logger:
                    logger.log(best_obj)
    
    # Phase 9: Larger neighborhood search - rebuild routes from scratch using savings algorithm
    def savings_algorithm(customers_to_visit, deliveries_dict):
        """Build routes using Clarke-Wright savings algorithm."""
        if not customers_to_visit:
            return []
        
        # Start with individual routes
        routes = {c: [0, c, 0] for c in customers_to_visit}
        route_of = {c: c for c in customers_to_visit}
        loads = {c: deliveries_dict[c] for c in customers_to_visit}
        
        # Compute savings
        savings = []
        for i in range(len(customers_to_visit)):
            for j in range(i + 1, len(customers_to_visit)):
                ci = customers_to_visit[i]
                cj = customers_to_visit[j]
                s = dist[0][ci] + dist[0][cj] - dist[ci][cj]
                if s > 0:
                    savings.append((s, ci, cj))
        
        savings.sort(reverse=True)
        
        for s, ci, cj in savings:
            ri = route_of[ci]
            rj = route_of[cj]
            if ri == rj:
                continue
            
            route_i = routes[ri]
            route_j = routes[rj]
            
            # Check capacity
            if loads[ri] + loads[rj] > vehicle_capacity:
                continue
            
            # Check if ci is at end of route_i and cj is at start of route_j (or vice versa)
            # ci should be adjacent to depot at end, cj adjacent to depot at start
            if route_i[-2] == ci and route_j[1] == cj:
                # Merge: route_i[:-1] + route_j[1:]
                new_route = route_i[:-1] + route_j[1:]
            elif route_j[-2] == cj and route_i[1] == ci:
                new_route = route_j[:-1] + route_i[1:]
            elif route_i[1] == ci and route_j[1] == cj:
                new_route = list(reversed(route_i[:-1])) + route_j[1:]
                # Ensure starts with 0
                if new_route[0] != 0:
                    new_route = [0] + new_route
            elif route_i[-2] == ci and route_j[-2] == cj:
                new_route = route_i[:-1] + list(reversed(route_j[1:]))
                if new_route[-1] != 0:
                    new_route.append(0)
            else:
                continue
            
            # Update
            new_load = loads[ri] + loads[rj]
            routes[ri] = new_route
            loads[ri] = new_load
            del routes[rj]
            del loads[rj]
            for node in new_route:
                if node != 0:
                    route_of[node] = ri
        
        result = list(routes.values())
        
        # Limit to num_vehicles
        while len(result) > num_vehicles:
            result.sort(key=lambda r: sum(deliveries_dict.get(n, 0) for n in r if n != 0))
            small = result.pop(0)
            small_custs = [n for n in small if n != 0]
            small_load = sum(deliveries_dict.get(n, 0) for n in small_custs)
            merged = False
            for ri in range(len(result)):
                r = result[ri]
                r_load = sum(deliveries_dict.get(n, 0) for n in r if n != 0)
                if r_load + small_load <= vehicle_capacity:
                    for cc in small_custs:
                        best_pos = 1
                        best_cost_val = float('inf')
                        for pos in range(1, len(r)):
                            cv = dist[r[pos-1]][cc] + dist[cc][r[pos]] - dist[r[pos-1]][r[pos]]
                            if cv < best_cost_val:
                                best_cost_val = cv
                                best_pos = pos
                        r.insert(best_pos, cc)
                    merged = True
                    break
            if not merged:
                result.insert(0, small)
                break
        
        return result
    
    # Rebuild routes using savings algorithm
    if best_obj < float('inf') and time_left() > 2:
        new_sol = deepcopy(best_sol)
        for t in range(num_periods):
            custs = list(new_sol[t]['deliveries'].keys())
            if custs:
                new_routes = savings_algorithm(custs, new_sol[t]['deliveries'])
                # Apply 2-opt
                new_routes = [two_opt(r) for r in new_routes]
                new_routes = [or_opt(r) for r in new_routes]
                new_sol[t]['routes'] = new_routes
        
        new_obj = evaluate_solution(new_sol)
        if new_obj < best_obj:
            best_sol = new_sol
            best_obj = new_obj
            if logger:
                logger.log(best_obj)
    
    # Phase 10: Perturbation + local search (iterated local search)
    def perturb_solution(sol_periods, rng):
        """Perturb solution by randomly adjusting deliveries."""
        new_sol = deepcopy(sol_periods)
        
        # Randomly pick an action
        action = rng.randint(0, 2)
        
        if action == 0:
            # Move some delivery from one period to another
            t = rng.randint(0, num_periods - 2)
            cids_in_t = list(new_sol[t]['deliveries'].keys())
            if cids_in_t:
                cid = rng.choice(cids_in_t)
                qty = new_sol[t]['deliveries'][cid]
                portion = rng.randint(1, qty)
                
                # Try to move to t+1
                new_sol[t]['deliveries'][cid] -= portion
                if new_sol[t]['deliveries'][cid] == 0:
                    del new_sol[t]['deliveries'][cid]
                    for ri, route in enumerate(new_sol[t]['routes']):
                        if cid in route:
                            route.remove(cid)
                            if len(route) == 2:
                                new_sol[t]['routes'].pop(ri)
                            break
                
                if cid in new_sol[t+1]['deliveries']:
                    new_sol[t+1]['deliveries'][cid] += portion
                else:
                    new_sol[t+1]['deliveries'][cid] = portion
                    # Add to route
                    added = False
                    for ri, route in enumerate(new_sol[t+1]['routes']):
                        r_load = sum(new_sol[t+1]['deliveries'].get(n, 0) for n in route if n != 0)
                        if r_load <= vehicle_capacity:
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
                    if not added and len(new_sol[t+1]['routes']) < num_vehicles:
                        new_sol[t+1]['routes'].append([0, cid, 0])
        
        elif action == 1:
            # Swap two customers between routes in same period
            t = rng.randint(0, num_periods - 1)
            if len(new_sol[t]['routes']) >= 2:
                ri = rng.randint(0, len(new_sol[t]['routes']) - 1)
                rj = rng.randint(0, len(new_sol[t]['routes']) - 1)
                while ri == rj:
                    rj = rng.randint(0, len(new_sol[t]['routes']) - 1)
                
                custs_i = [n for n in new_sol[t]['routes'][ri] if n != 0]
                custs_j = [n for n in new_sol[t]['routes'][rj] if n != 0]
                if custs_i and custs_j:
                    ci = rng.choice(custs_i)
                    cj = rng.choice(custs_j)
                    
                    # Swap
                    idx_i = new_sol[t]['routes'][ri].index(ci)
                    idx_j = new_sol[t]['routes'][rj].index(cj)
                    new_sol[t]['routes'][ri][idx_i] = cj
                    new_sol[t]['routes'][rj][idx_j] = ci
        
        elif action == 2:
            # Randomly change delivery quantity
            t = rng.randint(0, num_periods - 1)
            cids_in_t = list(new_sol[t]['deliveries'].keys())
            if cids_in_t:
                cid = rng.choice(cids_in_t)
                current_qty = new_sol[t]['deliveries'][cid]
                max_del = cust_max_inv[cid]  # rough upper bound
                new_qty = rng.randint(max(1, current_qty - 20), min(max_del, current_qty + 20))
                new_sol[t]['deliveries'][cid] = max(1, new_qty)
        
        return new_sol
    
    def recompute_inventories(sol_periods):
        """Recompute inventories for a solution."""
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
    
    # Iterated local search
    rng = random.Random(42)
    ils_iterations = 0
    while time_left() > 3 and best_obj < float('inf'):
        ils_iterations += 1
        if ils_iterations > 500:
            break
        
        new_sol = perturb_solution(best_sol, rng)
        recompute_inventories(new_sol)
        
        new_obj = evaluate_solution(new_sol)
        if new_obj < float('inf'):
            # Apply local search
            improve_routes(new_sol)
            new_obj = evaluate_solution(new_sol)
            
            if new_obj < best_obj:
                best_sol = new_sol
                best_obj = new_obj
                if logger:
                    logger.log(best_obj)
    
    # Final output
    if best_obj == float('inf'):
        # No feasible solution found - output empty solution
        result = {
            'objective_value': 0,
            'solution_details': {
                'periods': {}
            }
        }
        for t in range(num_periods):
            result['solution_details']['periods'][str(t+1)] = {
                'routes': [],
                'deliveries': {},
                'inventories': {'depot': 0.0}
            }
    else:
        result = format_solution(best_sol, best_obj)
    
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
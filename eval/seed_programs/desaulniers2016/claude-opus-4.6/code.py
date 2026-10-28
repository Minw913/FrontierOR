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
    num_nodes = data['num_nodes']
    dist = data['distance_matrix']
    holding_costs = data['holding_costs']
    demands = data['customer_demands']  # [customer_idx][period]
    production = data['supplier_production']
    inv_cap = data['inventory_capacities']
    init_inv = data['initial_inventories']
    
    def time_remaining():
        return time_limit - (time.time() - start_time)
    
    def route_cost(route):
        """Calculate travel cost for a route [0, c1, c2, ..., 0]"""
        cost = 0
        for i in range(len(route) - 1):
            cost += dist[route[i]][route[i+1]]
        return cost
    
    def nearest_neighbor_route(customers, depot=0):
        """Build a route using nearest neighbor heuristic"""
        if not customers:
            return [depot, depot]
        unvisited = list(customers)
        route = [depot]
        current = depot
        while unvisited:
            nearest = min(unvisited, key=lambda c: dist[current][c])
            route.append(nearest)
            unvisited.remove(nearest)
            current = nearest
        route.append(depot)
        return route
    
    def two_opt(route):
        """Improve route with 2-opt"""
        if len(route) <= 4:  # [0, c1, 0] or smaller
            return route
        improved = True
        best = route[:]
        while improved:
            improved = False
            for i in range(1, len(best) - 2):
                for j in range(i + 1, len(best) - 1):
                    new_route = best[:i] + best[i:j+1][::-1] + best[j+1:]
                    if route_cost(new_route) < route_cost(best):
                        best = new_route
                        improved = True
        return best
    
    def compute_objective(plan, inventories_depot, inventories_customers):
        """Compute total objective value"""
        total_cost = 0.0
        # Travel costs
        for t in range(num_periods):
            for routes_in_period in plan[t]:
                route = routes_in_period['route']
                total_cost += route_cost(route)
        # Holding costs
        for t in range(num_periods):
            total_cost += holding_costs[0] * inventories_depot[t]
            for c in range(num_customers):
                total_cost += holding_costs[c + 1] * inventories_customers[c][t]
        return total_cost
    
    def build_greedy_solution():
        """Build initial feasible solution using a greedy heuristic"""
        # Track inventories
        depot_inv = init_inv[0]
        cust_inv = [init_inv[c + 1] for c in range(num_customers)]
        
        plan = [[] for _ in range(num_periods)]
        inv_depot_history = []
        inv_cust_history = [[0]*num_periods for _ in range(num_customers)]
        
        for t in range(num_periods):
            # Add production
            depot_inv += production[t]
            depot_inv = min(depot_inv, inv_cap[0])
            
            # Determine which customers need delivery and how much
            deliveries_needed = []
            for c in range(num_customers):
                # Check if customer will stock out this period without delivery
                remaining = cust_inv[c] - demands[c][t]
                if remaining < 0:
                    # Must deliver
                    min_delivery = -remaining
                    max_delivery = min(inv_cap[c + 1] - cust_inv[c], depot_inv)
                    if max_delivery < min_delivery:
                        max_delivery = min_delivery  # Force it even if exceeding
                    # Deliver enough to cover future demands too (up to capacity)
                    future_demand = 0
                    for ft in range(t, min(t + 3, num_periods)):
                        future_demand += demands[c][ft]
                    ideal_delivery = min(future_demand - cust_inv[c], inv_cap[c + 1] - cust_inv[c])
                    ideal_delivery = max(ideal_delivery, min_delivery)
                    ideal_delivery = min(ideal_delivery, depot_inv) if depot_inv > 0 else min_delivery
                    deliveries_needed.append((c, ideal_delivery, True))
                else:
                    # Check if it's beneficial to deliver (look ahead)
                    future_inv = remaining
                    needs_delivery_soon = False
                    for ft in range(t + 1, min(t + 3, num_periods)):
                        future_inv -= demands[c][ft]
                        if future_inv < 0:
                            needs_delivery_soon = True
                            break
                    if needs_delivery_soon and depot_inv > 0:
                        future_demand_sum = sum(demands[c][ft] for ft in range(t, min(t + 3, num_periods)))
                        ideal_delivery = min(future_demand_sum - cust_inv[c], inv_cap[c + 1] - cust_inv[c])
                        ideal_delivery = max(ideal_delivery, 0)
                        ideal_delivery = min(ideal_delivery, depot_inv)
                        if ideal_delivery > 0:
                            deliveries_needed.append((c, ideal_delivery, False))
            
            # Sort: mandatory first, then by urgency
            deliveries_needed.sort(key=lambda x: (not x[2], -x[1]))
            
            # Build routes using bin-packing + nearest neighbor
            routes_this_period = []
            assigned = set()
            
            # Group customers into vehicle loads
            remaining_deliveries = [(c, qty) for c, qty, _ in deliveries_needed]
            
            while remaining_deliveries and len(routes_this_period) < num_vehicles:
                current_load = 0
                current_customers = []
                current_deliveries = {}
                still_remaining = []
                
                for c, qty in remaining_deliveries:
                    if c in assigned:
                        continue
                    actual_qty = min(qty, vehicle_capacity - current_load)
                    actual_qty = min(actual_qty, depot_inv)
                    if actual_qty <= 0:
                        still_remaining.append((c, qty))
                        continue
                    if current_load + actual_qty <= vehicle_capacity:
                        current_customers.append(c)
                        current_deliveries[c] = actual_qty
                        current_load += actual_qty
                        assigned.add(c)
                    else:
                        still_remaining.append((c, qty))
                
                if current_customers:
                    # Build route
                    route = nearest_neighbor_route(current_customers)
                    route = two_opt(route)
                    routes_this_period.append({
                        'route': route,
                        'deliveries': current_deliveries
                    })
                    # Update depot inventory
                    total_shipped = sum(current_deliveries.values())
                    depot_inv -= total_shipped
                    # Update customer inventories for delivery
                    for c, qty in current_deliveries.items():
                        cust_inv[c] += qty
                
                remaining_deliveries = still_remaining
                if not current_customers:
                    break
            
            plan[t] = routes_this_period
            
            # Consume demands
            for c in range(num_customers):
                cust_inv[c] -= demands[c][t]
            
            inv_depot_history.append(depot_inv)
            for c in range(num_customers):
                inv_cust_history[c][t] = cust_inv[c]
        
        return plan, inv_depot_history, inv_cust_history
    
    def validate_and_fix_solution(plan, inv_depot_history, inv_cust_history):
        """Recompute inventories from plan to ensure consistency"""
        depot_inv = init_inv[0]
        cust_inv = [init_inv[c + 1] for c in range(num_customers)]
        
        new_inv_depot = []
        new_inv_cust = [[0]*num_periods for _ in range(num_customers)]
        
        feasible = True
        for t in range(num_periods):
            depot_inv += production[t]
            
            total_shipped = 0
            deliveries_to_customers = [0] * num_customers
            for route_info in plan[t]:
                for c_str, qty in route_info['deliveries'].items():
                    c = int(c_str) if isinstance(c_str, str) else c_str
                    deliveries_to_customers[c] += qty
                    total_shipped += qty
            
            depot_inv -= total_shipped
            if depot_inv < 0 or depot_inv > inv_cap[0]:
                feasible = False
            
            for c in range(num_customers):
                cust_inv[c] += deliveries_to_customers[c]
                if cust_inv[c] > inv_cap[c + 1]:
                    feasible = False
                cust_inv[c] -= demands[c][t]
                if cust_inv[c] < 0:
                    feasible = False
                new_inv_cust[c][t] = cust_inv[c]
            
            new_inv_depot.append(depot_inv)
        
        return feasible, new_inv_depot, new_inv_cust
    
    def build_smarter_solution():
        """
        Build a feasible solution using an order-up-to-level (OU) policy.
        For each period, determine which customers must be visited (would stock out otherwise)
        and which customers are beneficial to visit (combine trips to save routing cost).
        """
        depot_inv = init_inv[0]
        cust_inv = [init_inv[c + 1] for c in range(num_customers)]
        
        plan = [[] for _ in range(num_periods)]
        inv_depot_history = []
        inv_cust_history = [[0]*num_periods for _ in range(num_customers)]
        
        for t in range(num_periods):
            depot_inv += production[t]
            depot_inv = min(depot_inv, inv_cap[0])
            
            # Categorize customers
            must_visit = []  # will stock out without delivery
            can_visit = []   # beneficial to visit
            
            for c in range(num_customers):
                end_inv = cust_inv[c] - demands[c][t]
                if end_inv < 0:
                    # Must deliver at least -end_inv
                    min_del = -end_inv
                    # Order up to level: try to fill to capacity or cover future demands
                    future_dem = 0
                    for ft in range(t, min(t + num_periods, num_periods)):
                        future_dem += demands[c][ft]
                        if cust_inv[c] + (inv_cap[c+1] - cust_inv[c]) - future_dem < 0:
                            break
                    max_del = inv_cap[c + 1] - cust_inv[c]
                    # Ideal: fill to cover as many future periods as possible
                    ideal_del = max(min_del, min(max_del, future_dem - cust_inv[c]))
                    ideal_del = max(ideal_del, min_del)
                    must_visit.append((c, min_del, ideal_del))
                else:
                    # Look ahead: when will this customer next need delivery?
                    future_inv = end_inv
                    periods_until_stockout = num_periods - t  # if never stocks out
                    for ft in range(t + 1, num_periods):
                        future_inv -= demands[c][ft]
                        if future_inv < 0:
                            periods_until_stockout = ft - t
                            break
                    
                    if periods_until_stockout <= 2:
                        # Beneficial to deliver now
                        future_dem = sum(demands[c][ft] for ft in range(t, min(t + periods_until_stockout + 1, num_periods)))
                        ideal_del = min(inv_cap[c + 1] - cust_inv[c], future_dem - cust_inv[c])
                        ideal_del = max(ideal_del, 0)
                        if ideal_del > 0:
                            can_visit.append((c, 0, ideal_del, periods_until_stockout))
            
            # Sort can_visit by urgency (sooner stockout first) then by holding cost (lower first to avoid storing expensive items)
            can_visit.sort(key=lambda x: (x[3], holding_costs[x[0] + 1]))
            
            # First allocate depot inventory to must-visit customers
            all_to_deliver = []
            available = depot_inv
            
            for c, min_del, ideal_del in must_visit:
                actual = min(ideal_del, available)
                actual = max(actual, min(min_del, available))  # Try to meet minimum
                if actual > 0:
                    all_to_deliver.append((c, actual))
                    available -= actual
            
            # Then allocate to can-visit customers
            for c, _, ideal_del, _ in can_visit:
                if available <= 0:
                    break
                actual = min(ideal_del, available)
                if actual > 0:
                    all_to_deliver.append((c, actual))
                    available -= actual
            
            # Now build routes using a simple bin-packing + routing approach
            # Sort by delivery quantity descending for better bin packing
            all_to_deliver.sort(key=lambda x: -x[1])
            
            routes_this_period = []
            unassigned = list(all_to_deliver)
            
            while unassigned and len(routes_this_period) < num_vehicles:
                current_load = 0
                current_customers = []
                current_deliveries = {}
                still_unassigned = []
                
                for c, qty in unassigned:
                    if current_load + qty <= vehicle_capacity:
                        current_customers.append(c)
                        current_deliveries[c] = qty
                        current_load += qty
                    else:
                        # Try partial delivery if must-visit
                        remaining_cap = vehicle_capacity - current_load
                        if remaining_cap > 0:
                            # Check if this is a must-visit customer that hasn't been assigned yet
                            is_must = any(mc == c for mc, _, _ in must_visit)
                            if is_must and c not in current_deliveries:
                                # Deliver partial
                                current_customers.append(c)
                                current_deliveries[c] = remaining_cap
                                current_load += remaining_cap
                                leftover = qty - remaining_cap
                                if leftover > 0:
                                    still_unassigned.append((c, leftover))
                                continue
                        still_unassigned.append((c, qty))
                
                if current_customers:
                    route = nearest_neighbor_route(current_customers)
                    route = two_opt(route)
                    routes_this_period.append({
                        'route': route,
                        'deliveries': current_deliveries
                    })
                else:
                    break
                
                unassigned = still_unassigned
            
            # Update inventories
            total_shipped = 0
            for route_info in routes_this_period:
                for c, qty in route_info['deliveries'].items():
                    cust_inv[c] += qty
                    total_shipped += qty
            
            depot_inv -= total_shipped
            
            # Consume demands
            for c in range(num_customers):
                cust_inv[c] -= demands[c][t]
            
            plan[t] = routes_this_period
            inv_depot_history.append(depot_inv)
            for c in range(num_customers):
                inv_cust_history[c][t] = cust_inv[c]
        
        return plan, inv_depot_history, inv_cust_history
    
    def compute_obj_from_plan(plan):
        """Recompute everything from plan"""
        depot_inv = init_inv[0]
        cust_inv = [init_inv[c + 1] for c in range(num_customers)]
        total_cost = 0.0
        
        inv_depot_history = []
        inv_cust_history = [[0]*num_periods for _ in range(num_customers)]
        
        for t in range(num_periods):
            depot_inv += production[t]
            
            for route_info in plan[t]:
                route = route_info['route']
                total_cost += route_cost(route)
                for c, qty in route_info['deliveries'].items():
                    c_int = int(c) if isinstance(c, str) else c
                    cust_inv[c_int] += qty
                    depot_inv -= qty
            
            total_cost += holding_costs[0] * depot_inv
            for c in range(num_customers):
                cust_inv[c] -= demands[c][t]
                total_cost += holding_costs[c + 1] * cust_inv[c]
                inv_cust_history[c][t] = cust_inv[c]
            
            inv_depot_history.append(depot_inv)
        
        return total_cost, inv_depot_history, inv_cust_history
    
    # Try to build a solution with a more sophisticated approach
    # Strategy: use LP to determine optimal delivery quantities, then route
    
    def solve_with_lp_then_route():
        """
        Phase 1: Solve an LP to determine optimal delivery quantities (ignoring routing).
        Phase 2: Build routes for the determined deliveries.
        """
        try:
            import gurobipy as gp
            from gurobipy import GRB
        except ImportError:
            return None
        
        remaining = time_remaining()
        if remaining < 5:
            return None
        
        C = range(num_customers)
        T = range(num_periods)
        
        try:
            model = gp.Model("IRP_delivery")
            model.setParam('OutputFlag', 0)
            model.setParam('TimeLimit', min(remaining * 0.3, 60))
            model.setParam('Threads', 1)
            
            # Decision variables: delivery quantity to customer c in period t
            q = {}
            for c in C:
                for t in T:
                    q[c, t] = model.addVar(lb=0, ub=inv_cap[c + 1], name=f"q_{c}_{t}")
            
            # Binary: whether customer c is visited in period t
            y = {}
            for c in C:
                for t in T:
                    y[c, t] = model.addVar(vtype=GRB.BINARY, name=f"y_{c}_{t}")
            
            # Inventory variables
            I_depot = {}
            I_cust = {}
            
            for t in T:
                I_depot[t] = model.addVar(lb=0, ub=inv_cap[0], name=f"Id_{t}")
            for c in C:
                for t in T:
                    I_cust[c, t] = model.addVar(lb=0, ub=inv_cap[c + 1], name=f"Ic_{c}_{t}")
            
            model.update()
            
            # Depot inventory balance
            for t in T:
                prev_depot = init_inv[0] if t == 0 else I_depot[t - 1]
                model.addConstr(
                    I_depot[t] == prev_depot + production[t] - gp.quicksum(q[c, t] for c in C)
                )
            
            # Customer inventory balance
            for c in C:
                for t in T:
                    prev_cust = init_inv[c + 1] if t == 0 else I_cust[c, t - 1]
                    model.addConstr(
                        I_cust[c, t] == prev_cust + q[c, t] - demands[c][t]
                    )
            
            # Link delivery to visit
            for c in C:
                for t in T:
                    model.addConstr(q[c, t] <= inv_cap[c + 1] * y[c, t])
            
            # Inventory capacity after delivery
            for c in C:
                for t in T:
                    prev_cust = init_inv[c + 1] if t == 0 else I_cust[c, t - 1]
                    # prev_cust + q[c,t] <= inv_cap[c+1]
                    # Already q[c,t] <= inv_cap[c+1] - prev_cust handled implicitly since I_cust has ub
                    # But let's add explicit constraint
                    model.addConstr(q[c, t] <= inv_cap[c + 1] - (prev_cust if t == 0 else I_cust[c, t - 1]))
            
            # Vehicle capacity constraint (approximate: total delivery per period / vehicle_capacity <= num_vehicles)
            for t in T:
                model.addConstr(
                    gp.quicksum(q[c, t] for c in C) <= num_vehicles * vehicle_capacity
                )
                # Number of visits per period <= num_vehicles * max_customers_per_route
                # Rough: each vehicle can visit at most some customers
                # At minimum, each visited customer needs one vehicle
                # Approximate: number of customers visited <= num_vehicles * (some upper bound)
                # Simple bound: sum of y <= num_vehicles * num_customers (trivial, skip)
                # Better: we need at least ceil(total_delivery / vehicle_capacity) vehicles
                # This is hard to linearize exactly, but we can add:
                # sum(y[c,t]) <= num_vehicles * max_per_route where max_per_route is generous
                pass
            
            # Estimate routing cost per customer visit (average distance to depot * 2)
            avg_cost = {}
            for c in C:
                avg_cost[c] = 2 * dist[0][c + 1]  # Simple: cost = 2 * distance to depot
            
            # Objective: holding costs + approximate routing costs
            obj = gp.LinExpr()
            for t in T:
                obj += holding_costs[0] * I_depot[t]
                for c in C:
                    obj += holding_costs[c + 1] * I_cust[c, t]
                    obj += avg_cost[c] * y[c, t]
            
            model.setObjective(obj, GRB.MINIMIZE)
            model.optimize()
            
            if model.status in [GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT]:
                if model.SolCount > 0:
                    deliveries = {}
                    for c in C:
                        for t in T:
                            val = q[c, t].X
                            if val > 0.5:
                                deliveries[(c, t)] = val
                    return deliveries
            
            return None
            
        except Exception as e:
            return None
    
    def build_solution_from_deliveries(deliveries):
        """Given delivery quantities, build routes"""
        depot_inv = init_inv[0]
        cust_inv = [init_inv[c + 1] for c in range(num_customers)]
        
        plan = [[] for _ in range(num_periods)]
        inv_depot_history = []
        inv_cust_history = [[0]*num_periods for _ in range(num_customers)]
        
        for t in range(num_periods):
            depot_inv += production[t]
            
            # Get deliveries for this period
            period_deliveries = []
            for c in range(num_customers):
                if (c, t) in deliveries and deliveries[(c, t)] > 0.5:
                    qty = round(deliveries[(c, t)])
                    # Ensure feasibility
                    max_qty = min(inv_cap[c + 1] - cust_inv[c], depot_inv)
                    qty = min(qty, max(0, max_qty))
                    # Ensure customer doesn't stock out
                    min_needed = max(0, demands[c][t] - cust_inv[c])
                    qty = max(qty, min(min_needed, depot_inv, inv_cap[c + 1] - cust_inv[c]))
                    if qty > 0:
                        period_deliveries.append((c, qty))
            
            # Also check for customers that need delivery but weren't planned
            for c in range(num_customers):
                if any(pc == c for pc, _ in period_deliveries):
                    continue
                if cust_inv[c] - demands[c][t] < 0:
                    min_needed = demands[c][t] - cust_inv[c]
                    max_qty = min(inv_cap[c + 1] - cust_inv[c], depot_inv)
                    qty = min(min_needed, max_qty)
                    if qty > 0:
                        period_deliveries.append((c, qty))
            
            # Build routes using bin packing
            period_deliveries.sort(key=lambda x: -x[1])
            routes_this_period = []
            unassigned = list(period_deliveries)
            
            while unassigned and len(routes_this_period) < num_vehicles:
                current_load = 0
                current_customers = []
                current_deliveries = {}
                still_unassigned = []
                
                for c, qty in unassigned:
                    if current_load + qty <= vehicle_capacity:
                        current_customers.append(c)
                        current_deliveries[c] = qty
                        current_load += qty
                    else:
                        remaining_cap = vehicle_capacity - current_load
                        if remaining_cap > 0 and c not in current_deliveries:
                            # Partial delivery
                            current_customers.append(c)
                            current_deliveries[c] = remaining_cap
                            current_load += remaining_cap
                            leftover = qty - remaining_cap
                            if leftover > 0:
                                still_unassigned.append((c, leftover))
                        else:
                            still_unassigned.append((c, qty))
                
                if current_customers:
                    route = nearest_neighbor_route(current_customers)
                    route = two_opt(route)
                    routes_this_period.append({
                        'route': route,
                        'deliveries': current_deliveries
                    })
                else:
                    break
                
                unassigned = still_unassigned
            
            # Update inventories
            total_shipped = 0
            for route_info in routes_this_period:
                for c, qty in route_info['deliveries'].items():
                    cust_inv[c] += qty
                    total_shipped += qty
            depot_inv -= total_shipped
            
            for c in range(num_customers):
                cust_inv[c] -= demands[c][t]
            
            plan[t] = routes_this_period
            inv_depot_history.append(depot_inv)
            for c in range(num_customers):
                inv_cust_history[c][t] = cust_inv[c]
        
        return plan, inv_depot_history, inv_cust_history
    
    def try_improve_routes(plan):
        """Try to improve routes using local search"""
        improved_plan = deepcopy(plan)
        
        for t in range(num_periods):
            for ri, route_info in enumerate(improved_plan[t]):
                route = route_info['route']
                if len(route) > 4:
                    new_route = two_opt(route)
                    improved_plan[t][ri]['route'] = new_route
        
        return improved_plan
    
    def try_merge_routes(plan):
        """Try to merge routes within the same period"""
        improved_plan = deepcopy(plan)
        
        for t in range(num_periods):
            routes = improved_plan[t]
            if len(routes) < 2:
                continue
            
            merged = True
            while merged:
                merged = False
                for i in range(len(routes)):
                    for j in range(i + 1, len(routes)):
                        # Check if routes can be merged
                        total_load = sum(routes[i]['deliveries'].values()) + sum(routes[j]['deliveries'].values())
                        if total_load <= vehicle_capacity:
                            # Check no duplicate customers
                            custs_i = set(routes[i]['deliveries'].keys())
                            custs_j = set(routes[j]['deliveries'].keys())
                            if not custs_i.intersection(custs_j):
                                # Merge
                                new_deliveries = {}
                                new_deliveries.update(routes[i]['deliveries'])
                                new_deliveries.update(routes[j]['deliveries'])
                                all_custs = list(new_deliveries.keys())
                                new_route = nearest_neighbor_route(all_custs)
                                new_route = two_opt(new_route)
                                
                                old_cost = route_cost(routes[i]['route']) + route_cost(routes[j]['route'])
                                new_cost = route_cost(new_route)
                                
                                if new_cost < old_cost:
                                    routes[i] = {'route': new_route, 'deliveries': new_deliveries}
                                    routes.pop(j)
                                    merged = True
                                    break
                    if merged:
                        break
            
            improved_plan[t] = routes
        
        return improved_plan
    
    def try_relocate_between_routes(plan):
        """Try moving a customer from one route to another in the same period"""
        improved_plan = deepcopy(plan)
        
        for t in range(num_periods):
            routes = improved_plan[t]
            if len(routes) < 2:
                continue
            
            improved = True
            while improved:
                improved = False
                for i in range(len(routes)):
                    for j in range(len(routes)):
                        if i == j:
                            continue
                        route_i = routes[i]
                        route_j = routes[j]
                        
                        # Try moving each customer from route i to route j
                        for c in list(route_i['deliveries'].keys()):
                            qty = route_i['deliveries'][c]
                            load_j = sum(route_j['deliveries'].values())
                            
                            if load_j + qty > vehicle_capacity:
                                continue
                            if c in route_j['deliveries']:
                                continue
                            
                            # Compute current costs
                            old_cost = route_cost(route_i['route']) + route_cost(route_j['route'])
                            
                            # New route i without c
                            new_custs_i = [x for x in route_i['deliveries'].keys() if x != c]
                            new_dels_i = {x: route_i['deliveries'][x] for x in new_custs_i}
                            new_route_i = nearest_neighbor_route(new_custs_i) if new_custs_i else [0, 0]
                            if new_custs_i:
                                new_route_i = two_opt(new_route_i)
                            
                            # New route j with c
                            new_custs_j = list(route_j['deliveries'].keys()) + [c]
                            new_dels_j = dict(route_j['deliveries'])
                            new_dels_j[c] = qty
                            new_route_j = nearest_neighbor_route(new_custs_j)
                            new_route_j = two_opt(new_route_j)
                            
                            new_cost = route_cost(new_route_i) + route_cost(new_route_j)
                            
                            if new_cost < old_cost - 0.5:
                                if new_custs_i:
                                    routes[i] = {'route': new_route_i, 'deliveries': new_dels_i}
                                else:
                                    routes[i] = {'route': [0, 0], 'deliveries': {}}
                                routes[j] = {'route': new_route_j, 'deliveries': new_dels_j}
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
            
            # Remove empty routes
            improved_plan[t] = [r for r in routes if r['deliveries']]
        
        return improved_plan
    
    def or_opt_route(route):
        """Or-opt: move segments of 1, 2, or 3 consecutive customers"""
        if len(route) <= 4:
            return route
        
        best = route[:]
        best_cost = route_cost(best)
        improved = True
        
        while improved:
            improved = False
            for seg_len in [1, 2, 3]:
                for i in range(1, len(best) - seg_len - 1):
                    segment = best[i:i + seg_len]
                    remaining = best[:i] + best[i + seg_len:]
                    
                    for j in range(1, len(remaining)):
                        new_route = remaining[:j] + segment + remaining[j:]
                        new_cost = route_cost(new_route)
                        if new_cost < best_cost - 0.5:
                            best = new_route
                            best_cost = new_cost
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        
        return best
    
    def improve_all_routes(plan):
        """Apply more advanced route improvement"""
        improved_plan = deepcopy(plan)
        for t in range(num_periods):
            for ri, route_info in enumerate(improved_plan[t]):
                route = route_info['route']
                if len(route) > 4:
                    route = two_opt(route)
                    route = or_opt_route(route)
                    improved_plan[t][ri]['route'] = route
        return improved_plan
    
    # Build initial solution
    plan1, inv_d1, inv_c1 = build_smarter_solution()
    feasible1, inv_d1, inv_c1 = validate_and_fix_solution(plan1, inv_d1, inv_c1)
    
    plan2, inv_d2, inv_c2 = build_greedy_solution()
    feasible2, inv_d2, inv_c2 = validate_and_fix_solution(plan2, inv_d2, inv_c2)
    
    best_plan = None
    best_obj = float('inf')
    best_inv_d = None
    best_inv_c = None
    
    if feasible1:
        obj1, inv_d1, inv_c1 = compute_obj_from_plan(plan1)
        if obj1 < best_obj:
            best_plan = plan1
            best_obj = obj1
            best_inv_d = inv_d1
            best_inv_c = inv_c1
    
    if feasible2:
        obj2, inv_d2, inv_c2 = compute_obj_from_plan(plan2)
        if obj2 < best_obj:
            best_plan = plan2
            best_obj = obj2
            best_inv_d = inv_d2
            best_inv_c = inv_c2
    
    if best_plan is not None and logger:
        logger.log(best_obj)
    
    # Try LP-based approach
    if time_remaining() > 10:
        lp_deliveries = solve_with_lp_then_route()
        if lp_deliveries is not None:
            plan3, inv_d3, inv_c3 = build_solution_from_deliveries(lp_deliveries)
            feasible3, inv_d3, inv_c3 = validate_and_fix_solution(plan3, inv_d3, inv_c3)
            if feasible3:
                obj3, inv_d3, inv_c3 = compute_obj_from_plan(plan3)
                if obj3 < best_obj:
                    best_plan = plan3
                    best_obj = obj3
                    best_inv_d = inv_d3
                    best_inv_c = inv_c3
                    if logger:
                        logger.log(best_obj)
    
    # Improve routes
    if best_plan is not None and time_remaining() > 5:
        improved = improve_all_routes(best_plan)
        improved = try_merge_routes(improved)
        improved = try_relocate_between_routes(improved)
        improved = improve_all_routes(improved)
        
        feasible_imp, _, _ = validate_and_fix_solution(improved, best_inv_d, best_inv_c)
        if feasible_imp:
            obj_imp, inv_d_imp, inv_c_imp = compute_obj_from_plan(improved)
            if obj_imp < best_obj:
                best_plan = improved
                best_obj = obj_imp
                best_inv_d = inv_d_imp
                best_inv_c = inv_c_imp
                if logger:
                    logger.log(best_obj)
    
    # Try MIP approach for small instances
    def solve_mip():
        try:
            import gurobipy as gp
            from gurobipy import GRB
        except ImportError:
            return None
        
        remaining = time_remaining()
        if remaining < 10 or num_customers > 25:
            return None
        
        C = range(num_customers)
        T = range(num_periods)
        N = range(num_nodes)  # 0=depot, 1..num_customers = customers
        
        try:
            model = gp.Model("IRP_MIP")
            model.setParam('OutputFlag', 0)
            model.setParam('TimeLimit', remaining - 5)
            model.setParam('Threads', 1)
            model.setParam('MIPGap', 1e-4)
            
            if best_obj < float('inf'):
                model.setParam('Cutoff', best_obj)
            
            V = range(num_vehicles)
            
            # Decision variables
            # x[i,j,v,t] = 1 if vehicle v travels from i to j in period t
            x = {}
            for t in T:
                for v in V:
                    for i in N:
                        for j in N:
                            if i != j:
                                x[i, j, v, t] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}_{v}_{t}")
            
            # q[c,v,t] = quantity delivered to customer c by vehicle v in period t
            q = {}
            for c in C:
                for v in V:
                    for t in T:
                        q[c, v, t] = model.addVar(lb=0, name=f"q_{c}_{v}_{t}")
            
            # y[c,v,t] = 1 if customer c+1 is visited by vehicle v in period t
            y = {}
            for c in C:
                for v in V:
                    for t in T:
                        y[c, v, t] = model.addVar(vtype=GRB.BINARY, name=f"y_{c}_{v}_{t}")
            
            # Inventory
            I_depot = {}
            I_cust = {}
            for t in T:
                I_depot[t] = model.addVar(lb=0, ub=inv_cap[0], name=f"Id_{t}")
            for c in C:
                for t in T:
                    I_cust[c, t] = model.addVar(lb=0, ub=inv_cap[c + 1], name=f"Ic_{c}_{t}")
            
            # MTZ subtour elimination variables
            u = {}
            for c in C:
                for v in V:
                    for t in T:
                        u[c, v, t] = model.addVar(lb=0, ub=num_customers, name=f"u_{c}_{v}_{t}")
            
            model.update()
            
            # Constraints
            
            # Depot inventory balance
            for t in T:
                prev = init_inv[0] if t == 0 else I_depot[t - 1]
                model.addConstr(
                    I_depot[t] == prev + production[t] - gp.quicksum(q[c, v, t] for c in C for v in V)
                )
            
            # Customer inventory balance
            for c in C:
                for t in T:
                    prev = init_inv[c + 1] if t == 0 else I_cust[c, t - 1]
                    model.addConstr(
                        I_cust[c, t] == prev + gp.quicksum(q[c, v, t] for v in V) - demands[c][t]
                    )
            
            # Customer inventory capacity after delivery
            for c in C:
                for t in T:
                    prev = init_inv[c + 1] if t == 0 else I_cust[c, t - 1]
                    model.addConstr(
                        gp.quicksum(q[c, v, t] for v in V) <= inv_cap[c + 1] - prev
                    )
            
            # Link delivery to visit
            for c in C:
                for v in V:
                    for t in T:
                        model.addConstr(q[c, v, t] <= inv_cap[c + 1] * y[c, v, t])
            
            # Vehicle capacity
            for v in V:
                for t in T:
                    model.addConstr(gp.quicksum(q[c, v, t] for c in C) <= vehicle_capacity)
            
            # Each customer visited by at most one vehicle per period
            for c in C:
                for t in T:
                    model.addConstr(gp.quicksum(y[c, v, t] for v in V) <= 1)
            
            # Flow conservation: if customer is visited, exactly one arc in and one arc out
            for c in C:
                node = c + 1
                for v in V:
                    for t in T:
                        model.addConstr(
                            gp.quicksum(x[i, node, v, t] for i in N if i != node and (i, node, v, t) in x) == y[c, v, t]
                        )
                        model.addConstr(
                            gp.quicksum(x[node, j, v, t] for j in N if j != node and (node, j, v, t) in x) == y[c, v, t]
                        )
            
            # Depot: vehicle leaves and returns
            for v in V:
                for t in T:
                    model.addConstr(
                        gp.quicksum(x[0, j, v, t] for j in range(1, num_nodes) if (0, j, v, t) in x) ==
                        gp.quicksum(x[i, 0, v, t] for i in range(1, num_nodes) if (i, 0, v, t) in x)
                    )
                    model.addConstr(
                        gp.quicksum(x[0, j, v, t] for j in range(1, num_nodes) if (0, j, v, t) in x) <= 1
                    )
            
            # MTZ subtour elimination
            for v in V:
                for t in T:
                    for c1 in C:
                        for c2 in C:
                            if c1 != c2:
                                node1 = c1 + 1
                                node2 = c2 + 1
                                if (node1, node2, v, t) in x:
                                    model.addConstr(
                                        u[c1, v, t] - u[c2, v, t] + num_customers * x[node1, node2, v, t] <= num_customers - 1
                                    )
            
            # Symmetry breaking: order vehicles
            for v in range(len(V) - 1):
                for t in T:
                    model.addConstr(
                        gp.quicksum(y[c, v, t] for c in C) >= gp.quicksum(y[c, v + 1, t] for c in C)
                    )
            
            # Objective
            obj = gp.LinExpr()
            for t in T:
                obj += holding_costs[0] * I_depot[t]
                for c in C:
                    obj += holding_costs[c + 1] * I_cust[c, t]
                for v in V:
                    for i in N:
                        for j in N:
                            if i != j and (i, j, v, t) in x:
                                obj += dist[i][j] * x[i, j, v, t]
            
            model.setObjective(obj, GRB.MINIMIZE)
            
            # Warm start from best solution
            if best_plan is not None:
                # Set initial values
                for c in C:
                    for v in V:
                        for t in T:
                            y[c, v, t].Start = 0
                            q[c, v, t].Start = 0
                
                for i in N:
                    for j in N:
                        if i != j:
                            for v in V:
                                for t in T:
                                    if (i, j, v, t) in x:
                                        x[i, j, v, t].Start = 0
                
                for t in T:
                    for vi, route_info in enumerate(best_plan[t]):
                        if vi >= num_vehicles:
                            break
                        route = route_info['route']
                        for c, qty in route_info['deliveries'].items():
                            c_int = int(c) if isinstance(c, str) else c
                            y[c_int, vi, t].Start = 1
                            q[c_int, vi, t].Start = qty
                        
                        for idx in range(len(route) - 1):
                            i_node = route[idx]
                            j_node = route[idx + 1]
                            if (i_node, j_node, vi, t) in x:
                                x[i_node, j_node, vi, t].Start = 1
                
                for t in T:
                    I_depot[t].Start = best_inv_d[t]
                    for c in C:
                        I_cust[c, t].Start = best_inv_c[c][t]
            
            model.optimize()
            
            if model.SolCount > 0:
                # Extract solution
                new_plan = [[] for _ in range(num_periods)]
                
                for t in T:
                    for v in V:
                        visited = []
                        dels = {}
                        for c in C:
                            if y[c, v, t].X > 0.5:
                                visited.append(c + 1)
                                dels[c] = q[c, v, t].X
                        
                        if visited:
                            # Reconstruct route from x variables
                            route = [0]
                            current = 0
                            while True:
                                next_node = None
                                for j in N:
                                    if j != current and (current, j, v, t) in x and x[current, j, v, t].X > 0.5:
                                        next_node = j
                                        break
                                if next_node is None or next_node == 0:
                                    route.append(0)
                                    break
                                route.append(next_node)
                                current = next_node
                            
                            # Verify route contains all visited customers
                            route_custs = set(route[1:-1])
                            if route_custs != set(visited):
                                # Fallback: build route from visited
                                route = nearest_neighbor_route(visited)
                                route = two_opt(route)
                            
                            new_plan[t].append({
                                'route': route,
                                'deliveries': dels
                            })
                
                return new_plan
            
            return None
            
        except Exception as e:
            return None
    
    # Try MIP for small instances
    if num_customers <= 20 and time_remaining() > 15:
        mip_plan = solve_mip()
        if mip_plan is not None:
            feasible_mip, inv_d_mip, inv_c_mip = validate_and_fix_solution(mip_plan, None, None)
            if feasible_mip:
                obj_mip, inv_d_mip, inv_c_mip = compute_obj_from_plan(mip_plan)
                if obj_mip < best_obj:
                    best_plan = mip_plan
                    best_obj = obj_mip
                    best_inv_d = inv_d_mip
                    best_inv_c = inv_c_mip
                    if logger:
                        logger.log(best_obj)
    
    # Iterative improvement: adjust delivery quantities
    def try_adjust_deliveries(plan):
        """Try to adjust delivery quantities to reduce holding costs"""
        nonlocal best_obj, best_plan, best_inv_d, best_inv_c
        
        for iteration in range(50):
            if time_remaining() < 2:
                break
            
            improved = False
            new_plan = deepcopy(plan)
            
            # For each customer, try to reduce deliveries (reduce holding) 
            # or shift deliveries between periods
            for c in range(num_customers):
                # Find all periods where this customer gets delivery
                delivery_periods = []
                for t in range(num_periods):
                    for ri, route_info in enumerate(new_plan[t]):
                        c_key = c
                        if c_key in route_info['deliveries']:
                            delivery_periods.append((t, ri, route_info['deliveries'][c_key]))
                
                if len(delivery_periods) < 2:
                    continue
                
                # Try reducing delivery in one period and increasing in another (later) period
                for idx1 in range(len(delivery_periods)):
                    t1, ri1, qty1 = delivery_periods[idx1]
                    if qty1 <= 1:
                        continue
                    
                    # Try reducing this delivery
                    # Check how much we can reduce
                    temp_plan = deepcopy(new_plan)
                    reduce_amt = min(qty1 - 1, 5)
                    
                    old_qty = temp_plan[t1][ri1]['deliveries'][c]
                    temp_plan[t1][ri1]['deliveries'][c] = old_qty - reduce_amt
                    
                    # Check feasibility
                    feasible_temp, _, _ = validate_and_fix_solution(temp_plan, None, None)
                    if feasible_temp:
                        obj_temp, inv_d_temp, inv_c_temp = compute_obj_from_plan(temp_plan)
                        if obj_temp < best_obj:
                            new_plan = temp_plan
                            best_plan = new_plan
                            best_obj = obj_temp
                            best_inv_d = inv_d_temp
                            best_inv_c = inv_c_temp
                            improved = True
                            if logger:
                                logger.log(best_obj)
                            break
                
                if improved:
                    break
            
            if not improved:
                break
            plan = new_plan
        
        return plan
    
    if best_plan is not None and time_remaining() > 5:
        best_plan = try_adjust_deliveries(best_plan)
    
    # Additional local search: try removing unnecessary visits
    def try_remove_visits(plan):
        """Try removing customer visits that aren't strictly necessary"""
        nonlocal best_obj, best_plan, best_inv_d, best_inv_c
        
        improved = True
        current_plan = deepcopy(plan)
        
        while improved and time_remaining() > 2:
            improved = False
            
            for t in range(num_periods):
                for ri in range(len(current_plan[t])):
                    route_info = current_plan[t][ri]
                    customers_in_route = list(route_info['deliveries'].keys())
                    
                    for c in customers_in_route:
                        if time_remaining() < 1:
                            return current_plan
                        
                        # Try removing this customer from the route
                        test_plan = deepcopy(current_plan)
                        del test_plan[t][ri]['deliveries'][c]
                        
                        # Rebuild route without this customer
                        remaining_custs = [x for x in test_plan[t][ri]['route'][1:-1] if x != c + 1 and x != c]
                        # Figure out node index
                        node = c + 1 if (c + 1) in test_plan[t][ri]['route'] else c
                        remaining_custs = [x for x in test_plan[t][ri]['route'][1:-1] if x != node]
                        
                        if remaining_custs:
                            new_route = nearest_neighbor_route(remaining_custs)
                            new_route = two_opt(new_route)
                            test_plan[t][ri]['route'] = new_route
                        else:
                            test_plan[t].pop(ri)
                        
                        feasible_test, _, _ = validate_and_fix_solution(test_plan, None, None)
                        if feasible_test:
                            obj_test, inv_d_test, inv_c_test = compute_obj_from_plan(test_plan)
                            if obj_test < best_obj:
                                current_plan = test_plan
                                best_plan = current_plan
                                best_obj = obj_test
                                best_inv_d = inv_d_test
                                best_inv_c = inv_c_test
                                improved = True
                                if logger:
                                    logger.log(best_obj)
                                break
                    if improved:
                        break
                if improved:
                    break
        
        return current_plan
    
    if best_plan is not None and time_remaining() > 5:
        best_plan = try_remove_visits(best_plan)
    
    # Final route improvement pass
    if best_plan is not None and time_remaining() > 2:
        improved = improve_all_routes(best_plan)
        improved = try_merge_routes(improved)
        improved = try_relocate_between_routes(improved)
        improved = improve_all_routes(improved)
        
        feasible_imp, _, _ = validate_and_fix_solution(improved, None, None)
        if feasible_imp:
            obj_imp, inv_d_imp, inv_c_imp = compute_obj_from_plan(improved)
            if obj_imp < best_obj:
                best_plan = improved
                best_obj = obj_imp
                best_inv_d = inv_d_imp
                best_inv_c = inv_c_imp
                if logger:
                    logger.log(best_obj)
    
    # If no feasible solution found, try a very basic approach
    if best_plan is None:
        # Deliver everything possible every period
        depot_inv = init_inv[0]
        cust_inv = [init_inv[c + 1] for c in range(num_customers)]
        
        plan = [[] for _ in range(num_periods)]
        inv_depot_history = []
        inv_cust_history = [[0]*num_periods for _ in range(num_customers)]
        
        for t in range(num_periods):
            depot_inv += production[t]
            depot_inv = min(depot_inv, inv_cap[0])
            
            deliveries = []
            for c in range(num_customers):
                need = demands[c][t] - cust_inv[c]
                if need > 0:
                    qty = min(need, depot_inv, inv_cap[c + 1] - cust_inv[c])
                    if qty > 0:
                        deliveries.append((c, qty))
                        depot_inv -= qty
            
            # Simple: one customer per route
            routes = []
            for c, qty in deliveries:
                if len(routes) >= num_vehicles:
                    break
                if qty <= vehicle_capacity:
                    routes.append({
                        'route': [0, c + 1, 0],
                        'deliveries': {c: qty}
                    })
                    cust_inv[c] += qty
            
            for c in range(num_customers):
                cust_inv[c] -= demands[c][t]
            
            plan[t] = routes
            inv_depot_history.append(depot_inv)
            for c in range(num_customers):
                inv_cust_history[c][t] = cust_inv[c]
        
        feasible, inv_depot_history, inv_cust_history = validate_and_fix_solution(plan, inv_depot_history, inv_cust_history)
        if feasible:
            best_plan = plan
            best_obj, best_inv_d, best_inv_c = compute_obj_from_plan(plan)
            if logger:
                logger.log(best_obj)
    
    # Format output
    routes_output = {}
    inventories_output = {}
    
    for t in range(num_periods):
        period_key = str(t)
        routes_output[period_key] = {}
        
        if best_plan is not None:
            for vi, route_info in enumerate(best_plan[t]):
                vehicle_key = str(vi)
                deliveries_dict = {}
                for c, qty in route_info['deliveries'].items():
                    c_int = int(c) if isinstance(c, str) else c
                    deliveries_dict[str(c_int)] = round(qty, 2)
                
                routes_output[period_key][vehicle_key] = {
                    'route': route_info['route'],
                    'deliveries': deliveries_dict
                }
    
    # Inventories
    if best_inv_d is not None:
        for loc in range(num_nodes):
            loc_key = str(loc)
            inventories_output[loc_key] = {}
            for t in range(num_periods):
                if loc == 0:
                    inventories_output[loc_key][str(t)] = round(best_inv_d[t], 2)
                else:
                    inventories_output[loc_key][str(t)] = round(best_inv_c[loc - 1][t], 2)
    
    solution = {
        'objective_value': round(best_obj, 2) if best_obj < float('inf') else 0,
        'routes': routes_output,
        'inventories': inventories_output
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
    
    solve_irp(args.instance_path, args.solution_path, args.time_limit, args.log_path)
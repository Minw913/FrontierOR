import json
import argparse
import math
import time
import copy
from collections import defaultdict

def solve(instance_path, solution_path, time_limit, log_path):
    from solution_logger import SolutionLogger
    
    with open(instance_path, 'r') as f:
        inst = json.load(f)
    
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    start_time = time.time()
    
    gp = inst['global_parameters']
    num_orders = inst['num_orders']
    num_products = inst['num_products']
    depot = inst['depot']
    num_vehicles = inst['num_vehicles']
    num_drivers = inst['num_drivers']
    vehicle_capacity = inst['vehicle_capacity']
    num_workstations = inst['num_workstations']
    num_production_employees = inst['num_production_employees']
    
    period_length = gp['period_length_minutes']
    subperiod_length = gp['subperiod_length_minutes']
    subperiods_per_period = gp['subperiods_per_period']
    setup_length_subperiods = gp['setup_length_subperiods']
    max_shift_route_length_periods = gp['max_shift_route_length_periods']
    max_shift_route_length_minutes = gp['max_shift_route_length_minutes']
    min_paid_time_subperiods = gp['min_paid_time_subperiods']
    min_paid_time_minutes = gp['min_paid_time_minutes']
    driver_salary_per_minute = gp['driver_salary_per_minute']
    employee_salary_per_subperiod = gp['employee_salary_per_subperiod']
    setup_cost = gp['setup_cost']
    vehicle_op_cost_per_minute = gp['vehicle_operating_cost_per_minute']
    preparation_time = gp['preparation_time_minutes']
    service_time_default = gp['service_time_minutes']
    max_trips_per_route = gp['max_trips_per_route']
    horizon_start = gp['horizon_start_minutes']
    horizon_end = gp['horizon_end_minutes']
    num_periods = gp['num_periods']
    num_subperiods = gp['num_subperiods']
    
    products = inst['products']
    orders = inst['orders']
    travel_time_matrix = inst['travel_time_matrix']
    production_windows = inst['production_windows']
    periods = inst['periods']
    subperiods = inst['subperiods']
    
    # Build lookup structures
    order_by_id = {}
    for o in orders:
        order_by_id[o['order_id']] = o
    
    product_by_id = {}
    for p in products:
        product_by_id[p['product_id']] = p
    
    # Order index in travel_time_matrix: depot=0, order i has index i (1-based if order_ids are 1-based)
    # Actually, "indices 1 through num_orders correspond to customer order locations"
    # We need to figure out the mapping. Let's assume order_id maps to index order_id in the matrix
    # (since order_ids typically start from 1)
    
    order_ids = [o['order_id'] for o in orders]
    # Create mapping from order_id to matrix index
    # The problem says "indices 1 through num_orders correspond to customer order locations"
    # So order with order_id i might map to matrix index i, or it could be ordered.
    # Let's assume orders are indexed 1..num_orders in order of appearance
    order_id_to_matrix_idx = {}
    for i, o in enumerate(orders):
        order_id_to_matrix_idx[o['order_id']] = i + 1  # 1-based
    
    def travel(i, j):
        """Travel time from location i to location j (matrix indices)"""
        return travel_time_matrix[i][j]
    
    def order_matrix_idx(oid):
        return order_id_to_matrix_idx[oid]
    
    # Period start times
    period_start = {}
    period_end = {}
    for p in periods:
        period_start[p['period_id']] = p['start_time']
        period_end[p['period_id']] = p['end_time']
    
    subperiod_start = {}
    subperiod_end = {}
    for s in subperiods:
        subperiod_start[s['subperiod_id']] = s['start_time']
        subperiod_end[s['subperiod_id']] = s['end_time']
    
    # For each period, which subperiods belong to it
    period_subperiods = {}
    for pid in range(num_periods):
        ps = period_start[pid]
        pe = period_end[pid]
        period_subperiods[pid] = [s['subperiod_id'] for s in subperiods 
                                   if s['start_time'] >= ps and s['end_time'] <= pe]
    
    # Subperiod to period mapping
    subperiod_to_period = {}
    for pid in range(num_periods):
        for sid in period_subperiods[pid]:
            subperiod_to_period[sid] = pid
    
    def time_remaining():
        return time_limit - (time.time() - start_time)
    
    # ---- ROUTING HEURISTIC ----
    
    # Evaluate feasibility of a trip: given a list of order_ids visited in sequence,
    # starting at depot at the beginning of a period.
    def evaluate_trip(customer_list, start_period_id):
        """Returns (feasible, total_travel_time, arrival_times, return_time) or (False, ...)"""
        if not customer_list:
            return True, 0, [], period_start[start_period_id]
        
        dep_time = period_start[start_period_id]  # trip departs at start of period
        current_time = dep_time + preparation_time  # after preparation
        current_loc = 0  # depot
        total_travel = 0
        arrival_times = []
        total_space = 0
        
        for cid in customer_list:
            o = order_by_id[cid]
            midx = order_matrix_idx(cid)
            tt = travel(current_loc, midx)
            total_travel += tt
            arrival = current_time + tt
            
            # Check time window
            if arrival > o['time_window_end']:
                return False, 0, [], 0
            
            # Wait if early
            arrival = max(arrival, o['time_window_start'])
            arrival_times.append(arrival)
            
            st = o.get('service_time', service_time_default)
            current_time = arrival + st
            current_loc = midx
            total_space += o['space_requirement']
        
        # Check capacity
        if total_space > vehicle_capacity:
            return False, 0, [], 0
        
        # Return to depot
        tt_return = travel(current_loc, 0)
        total_travel += tt_return
        return_time = current_time + tt_return
        
        return True, total_travel, arrival_times, return_time
    
    def compute_trip_travel_time(customer_list):
        """Just compute travel time for a trip"""
        if not customer_list:
            return 0
        total = 0
        prev = 0
        for cid in customer_list:
            midx = order_matrix_idx(cid)
            total += travel(prev, midx)
            prev = midx
        total += travel(prev, 0)
        return total
    
    # Greedy construction: cluster orders by time windows and proximity
    # Use a savings-based or insertion heuristic
    
    unassigned = set(order_ids)
    routes = []  # List of routes, each route is list of trips
    # Each trip: (start_period, [customer_ids])
    
    # Sort orders by time window end (earliest deadline first)
    sorted_orders = sorted(orders, key=lambda o: (o['time_window_end'], o['time_window_start']))
    
    def find_best_period_for_trip(customer_list):
        """Find the best (earliest feasible) period to start a trip"""
        best_period = None
        best_travel = None
        for pid in range(num_periods):
            feas, tt, arrivals, ret = evaluate_trip(customer_list, pid)
            if feas:
                if best_period is None or pid < best_period:
                    best_period = pid
                    best_travel = tt
        return best_period, best_travel
    
    def find_latest_period_for_trip(customer_list):
        """Find the latest feasible period to start a trip (to allow more production time)"""
        best_period = None
        best_travel = None
        for pid in range(num_periods - 1, -1, -1):
            feas, tt, arrivals, ret = evaluate_trip(customer_list, pid)
            if feas:
                best_period = pid
                best_travel = tt
                break
        return best_period, best_travel
    
    # Simple greedy: build trips one at a time
    # For each trip, start with the unassigned order with earliest deadline,
    # then insert nearest compatible orders
    
    def build_solution_greedy():
        unassigned_local = set(order_ids)
        routes_local = []
        
        while unassigned_local and len(routes_local) < min(num_vehicles, num_drivers):
            if time_remaining() < 5:
                break
            
            # Pick seed: unassigned order with earliest deadline
            seed = None
            for o in sorted_orders:
                if o['order_id'] in unassigned_local:
                    seed = o['order_id']
                    break
            if seed is None:
                break
            
            route_trips = []
            
            for trip_idx in range(max_trips_per_route):
                if not unassigned_local:
                    break
                if time_remaining() < 5:
                    break
                
                # For first trip of route, pick seed
                if trip_idx == 0:
                    current_trip = [seed]
                    unassigned_local.discard(seed)
                else:
                    # Pick new seed from unassigned
                    new_seed = None
                    for o in sorted_orders:
                        if o['order_id'] in unassigned_local:
                            new_seed = o['order_id']
                            break
                    if new_seed is None:
                        break
                    current_trip = [new_seed]
                    unassigned_local.discard(new_seed)
                
                # Find a feasible period
                # Determine minimum period: for multi-trip routes, must be > previous trip's period
                min_period = 0
                if trip_idx > 0:
                    min_period = route_trips[-1][0] + 1
                
                # Try to insert more customers
                improved = True
                while improved and unassigned_local:
                    improved = False
                    best_cid = None
                    best_pos = None
                    best_cost = float('inf')
                    best_period = None
                    
                    # Check capacity
                    current_space = sum(order_by_id[c]['space_requirement'] for c in current_trip)
                    
                    candidates = list(unassigned_local)
                    # Limit candidates for speed
                    if len(candidates) > 50:
                        # Sort by distance to current trip centroid
                        cx = sum(order_by_id[c]['x'] for c in current_trip) / len(current_trip)
                        cy = sum(order_by_id[c]['y'] for c in current_trip) / len(current_trip)
                        candidates.sort(key=lambda c: (order_by_id[c]['x']-cx)**2 + (order_by_id[c]['y']-cy)**2)
                        candidates = candidates[:50]
                    
                    for cid in candidates:
                        o = order_by_id[cid]
                        if current_space + o['space_requirement'] > vehicle_capacity:
                            continue
                        
                        # Try inserting at each position
                        for pos in range(len(current_trip) + 1):
                            new_trip = current_trip[:pos] + [cid] + current_trip[pos:]
                            
                            # Find feasible period
                            found = False
                            for pid in range(min_period, num_periods):
                                feas, tt, arrivals, ret = evaluate_trip(new_trip, pid)
                                if feas:
                                    if tt < best_cost:
                                        best_cost = tt
                                        best_cid = cid
                                        best_pos = pos
                                        best_period = pid
                                    found = True
                                    break
                            
                    if best_cid is not None:
                        current_trip.insert(best_pos, best_cid)
                        unassigned_local.discard(best_cid)
                        improved = True
                
                # Find best period for this trip
                best_pid = None
                for pid in range(min_period, num_periods):
                    feas, tt, arrivals, ret = evaluate_trip(current_trip, pid)
                    if feas:
                        best_pid = pid
                        break
                
                if best_pid is None:
                    # Can't schedule this trip, put customers back
                    for c in current_trip:
                        unassigned_local.add(c)
                    break
                
                route_trips.append((best_pid, current_trip))
            
            if route_trips:
                routes_local.append(route_trips)
        
        return routes_local, unassigned_local
    
    routes, remaining = build_solution_greedy()
    
    # If there are remaining orders, try harder
    if remaining:
        # Try to insert remaining orders into existing routes/trips
        for oid in list(remaining):
            if time_remaining() < 3:
                break
            inserted = False
            o = order_by_id[oid]
            
            best_route_idx = None
            best_trip_idx = None
            best_pos = None
            best_extra_cost = float('inf')
            best_new_period = None
            
            for ri, route in enumerate(routes):
                for ti, (sp, clist) in enumerate(route):
                    # Check capacity
                    current_space = sum(order_by_id[c]['space_requirement'] for c in clist)
                    if current_space + o['space_requirement'] > vehicle_capacity:
                        continue
                    
                    min_period = 0
                    max_period = num_periods - 1
                    if ti > 0:
                        min_period = route[ti-1][0] + 1
                    if ti < len(route) - 1:
                        max_period = route[ti+1][0] - 1
                    
                    old_tt = compute_trip_travel_time(clist)
                    
                    for pos in range(len(clist) + 1):
                        new_clist = clist[:pos] + [oid] + clist[pos:]
                        for pid in range(min_period, max_period + 1):
                            feas, tt, arrivals, ret = evaluate_trip(new_clist, pid)
                            if feas:
                                extra = tt - old_tt
                                if extra < best_extra_cost:
                                    best_extra_cost = extra
                                    best_route_idx = ri
                                    best_trip_idx = ti
                                    best_pos = pos
                                    best_new_period = pid
                                break
            
            if best_route_idx is not None:
                sp, clist = routes[best_route_idx][best_trip_idx]
                new_clist = clist[:best_pos] + [oid] + clist[best_pos:]
                routes[best_route_idx][best_trip_idx] = (best_new_period, new_clist)
                remaining.discard(oid)
                inserted = True
            
            if not inserted:
                # Create a new route for this order if possible
                if len(routes) < min(num_vehicles, num_drivers):
                    for pid in range(num_periods):
                        feas, tt, arrivals, ret = evaluate_trip([oid], pid)
                        if feas:
                            routes.append([(pid, [oid])])
                            remaining.discard(oid)
                            break
    
    # If still remaining, try adding new trips to existing routes
    if remaining:
        for oid in list(remaining):
            if time_remaining() < 2:
                break
            o = order_by_id[oid]
            inserted = False
            
            for ri, route in enumerate(routes):
                if len(route) >= max_trips_per_route:
                    continue
                
                # Try adding a new trip after the last one
                last_period = route[-1][0]
                for pid in range(last_period + 1, num_periods):
                    feas, tt, arrivals, ret = evaluate_trip([oid], pid)
                    if feas:
                        route.append((pid, [oid]))
                        remaining.discard(oid)
                        inserted = True
                        break
                if inserted:
                    break
                
                # Try adding before first trip
                first_period = route[0][0]
                for pid in range(first_period - 1, -1, -1):
                    feas, tt, arrivals, ret = evaluate_trip([oid], pid)
                    if feas:
                        route.insert(0, (pid, [oid]))
                        remaining.discard(oid)
                        inserted = True
                        break
                if inserted:
                    break
            
            if not inserted and len(routes) < min(num_vehicles, num_drivers):
                for pid in range(num_periods):
                    feas, tt, arrivals, ret = evaluate_trip([oid], pid)
                    if feas:
                        routes.append([(pid, [oid])])
                        remaining.discard(oid)
                        break
    
    if remaining:
        print(f"WARNING: {len(remaining)} orders could not be assigned!")
        # Force them in somewhere - create single-order routes
        for oid in list(remaining):
            if len(routes) < min(num_vehicles, num_drivers):
                for pid in range(num_periods):
                    feas, tt, arrivals, ret = evaluate_trip([oid], pid)
                    if feas:
                        routes.append([(pid, [oid])])
                        remaining.discard(oid)
                        break
    
    # ---- LOCAL SEARCH on routes ----
    def try_improve_routes(routes, time_budget):
        """Try 2-opt within trips and order relocation between trips"""
        improve_start = time.time()
        improved = True
        
        while improved and (time.time() - improve_start) < time_budget:
            improved = False
            
            # Intra-trip 2-opt
            for ri, route in enumerate(routes):
                for ti, (sp, clist) in enumerate(route):
                    if len(clist) < 3:
                        continue
                    
                    min_period = 0
                    max_period = num_periods - 1
                    if ti > 0:
                        min_period = route[ti-1][0] + 1
                    if ti < len(route) - 1:
                        max_period = route[ti+1][0] - 1
                    
                    old_tt = compute_trip_travel_time(clist)
                    
                    for i in range(len(clist) - 1):
                        for j in range(i + 1, len(clist)):
                            new_clist = clist[:i] + clist[i:j+1][::-1] + clist[j+1:]
                            
                            for pid in range(min_period, max_period + 1):
                                feas, tt, arrivals, ret = evaluate_trip(new_clist, pid)
                                if feas and tt < old_tt - 0.01:
                                    routes[ri][ti] = (pid, new_clist)
                                    clist = new_clist
                                    old_tt = tt
                                    sp = pid
                                    improved = True
                                    break
                        
                        if (time.time() - improve_start) > time_budget:
                            return routes
            
            # Inter-trip relocation: move a customer from one trip to another
            for ri1, route1 in enumerate(routes):
                for ti1, (sp1, clist1) in enumerate(route1):
                    if len(clist1) <= 1:
                        continue
                    
                    for ci, cid in enumerate(clist1):
                        if (time.time() - improve_start) > time_budget:
                            return routes
                        
                        # Try removing cid from this trip and inserting elsewhere
                        new_clist1 = clist1[:ci] + clist1[ci+1:]
                        old_tt1 = compute_trip_travel_time(clist1)
                        
                        min_p1 = 0
                        max_p1 = num_periods - 1
                        if ti1 > 0:
                            min_p1 = route1[ti1-1][0] + 1
                        if ti1 < len(route1) - 1:
                            max_p1 = route1[ti1+1][0] - 1
                        
                        # Check if remaining trip is feasible
                        if new_clist1:
                            feas1 = False
                            new_tt1 = 0
                            new_sp1 = sp1
                            for pid in range(min_p1, max_p1 + 1):
                                f, tt, a, r = evaluate_trip(new_clist1, pid)
                                if f:
                                    feas1 = True
                                    new_tt1 = tt
                                    new_sp1 = pid
                                    break
                            if not feas1:
                                continue
                        else:
                            new_tt1 = 0
                            new_sp1 = sp1
                        
                        saving = old_tt1 - new_tt1
                        
                        # Try inserting into other trips
                        best_insert = None
                        best_insert_cost = float('inf')
                        
                        for ri2, route2 in enumerate(routes):
                            for ti2, (sp2, clist2) in enumerate(route2):
                                if ri1 == ri2 and ti1 == ti2:
                                    continue
                                
                                current_space = sum(order_by_id[c]['space_requirement'] for c in clist2)
                                if current_space + order_by_id[cid]['space_requirement'] > vehicle_capacity:
                                    continue
                                
                                min_p2 = 0
                                max_p2 = num_periods - 1
                                if ti2 > 0:
                                    min_p2 = route2[ti2-1][0] + 1
                                if ti2 < len(route2) - 1:
                                    max_p2 = route2[ti2+1][0] - 1
                                
                                old_tt2 = compute_trip_travel_time(clist2)
                                
                                for pos in range(len(clist2) + 1):
                                    nc2 = clist2[:pos] + [cid] + clist2[pos:]
                                    for pid in range(min_p2, max_p2 + 1):
                                        f, tt, a, r = evaluate_trip(nc2, pid)
                                        if f:
                                            cost_increase = tt - old_tt2
                                            if cost_increase < best_insert_cost:
                                                best_insert_cost = cost_increase
                                                best_insert = (ri2, ti2, pos, pid, nc2)
                                            break
                        
                        if best_insert is not None and best_insert_cost < saving - 0.01:
                            # Apply the move
                            ri2, ti2, pos, pid2, nc2 = best_insert
                            
                            if new_clist1:
                                routes[ri1][ti1] = (new_sp1, new_clist1)
                            else:
                                routes[ri1].pop(ti1)
                                if not routes[ri1]:
                                    routes.pop(ri1)
                                    if ri2 > ri1:
                                        ri2 -= 1
                            
                            routes[ri2][ti2] = (pid2, nc2)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        
        return routes
    
    improvement_time = min(time_remaining() * 0.3, 30)
    if improvement_time > 2:
        routes = try_improve_routes(routes, improvement_time)
    
    # ---- PRODUCTION SCHEDULING ----
    
    # For each order, determine the departure period (the start_period of its trip)
    order_departure_period = {}
    order_departure_subperiod = {}  # last subperiod before departure
    
    for route in routes:
        for sp, clist in route:
            for cid in clist:
                order_departure_period[cid] = sp
                # Departure is at start of period sp
                # Production must be complete by end of previous period's last subperiod
                # Actually, production must be complete by the time the trip departs
                # The trip departs at start of period sp
                # So production must be done by the last subperiod that ends <= period_start[sp]
                dep_time = period_start[sp]
                # Find latest subperiod that ends by dep_time
                latest_sub = -1
                for sid in range(num_subperiods):
                    if subperiod_end[sid] <= dep_time:
                        latest_sub = sid
                order_departure_subperiod[cid] = latest_sub
    
    # Build production schedule
    # For each order and product, we need to produce demand[product] units
    # within the valid subperiods, and cumulative production must meet demand by departure
    
    production_schedule = []  # (order, product, subperiod, quantity)
    
    # Aggregate: for each product and subperiod, track total employees needed
    product_subperiod_demand = defaultdict(float)  # (product_id, subperiod_id) -> total units needed
    
    # First, collect all production tasks
    tasks = []  # (order_id, product_id, demand, valid_subperiods, due_subperiod)
    for o in orders:
        oid = o['order_id']
        if oid not in order_departure_period:
            continue
        demands = o['demands']
        for pid_str, qty in demands.items():
            pid = int(pid_str)
            if qty <= 0:
                continue
            
            pw = production_windows[str(oid)][str(pid)]
            valid_subs = pw['valid_subperiod_ids']
            
            # Due subperiod: latest valid subperiod before departure
            dep_sub = order_departure_subperiod[oid]
            feasible_subs = [s for s in valid_subs if s <= dep_sub]
            
            if not feasible_subs:
                # Use all valid subperiods (might be an issue)
                feasible_subs = valid_subs
            
            tasks.append((oid, pid, qty, feasible_subs))
    
    # Schedule production: for each product, sort tasks by urgency (earliest due)
    # and assign production across subperiods
    
    # Group tasks by product
    tasks_by_product = defaultdict(list)
    for oid, pid, qty, fsubs in tasks:
        tasks_by_product[pid].append((oid, qty, fsubs))
    
    # For each product, schedule greedily
    # employees_assigned[product][subperiod] = number of employees
    employees_assigned = defaultdict(lambda: defaultdict(int))
    production_entries = []
    
    for pid, task_list in tasks_by_product.items():
        prod = product_by_id[pid]
        rate = prod['production_rate_per_subperiod']
        
        # Sort tasks by latest feasible subperiod (most urgent first)
        task_list.sort(key=lambda t: max(t[2]) if t[2] else 0)
        
        # For each task, spread production across available subperiods
        for oid, qty, fsubs in task_list:
            remaining_qty = qty
            fsubs_sorted = sorted(fsubs)
            
            for sid in fsubs_sorted:
                if remaining_qty <= 0:
                    break
                # How much can we produce in this subperiod?
                # We need employees. Each employee produces 'rate' units per subperiod.
                needed_employees = math.ceil(remaining_qty / rate)
                # Produce as much as possible
                can_produce = min(remaining_qty, rate * max(1, needed_employees))
                # For simplicity, assign 1 employee at a time
                produce_qty = min(remaining_qty, rate)
                if produce_qty > 0:
                    production_entries.append({
                        'order': oid,
                        'product': pid,
                        'subperiod': sid,
                        'quantity': produce_qty
                    })
                    employees_assigned[pid][sid] += 1
                    remaining_qty -= produce_qty
            
            # If still remaining, add more employees to earlier subperiods
            if remaining_qty > 0:
                for sid in fsubs_sorted:
                    if remaining_qty <= 0:
                        break
                    additional = math.ceil(remaining_qty / rate)
                    produce_qty = min(remaining_qty, rate * additional)
                    production_entries.append({
                        'order': oid,
                        'product': pid,
                        'subperiod': sid,
                        'quantity': produce_qty
                    })
                    employees_assigned[pid][sid] += additional
                    remaining_qty -= produce_qty
    
    # Now compute staffing certificate
    # employees_by_product_subperiod: aggregate employees per product per subperiod
    agg_employees = defaultdict(lambda: defaultdict(int))
    for entry in production_entries:
        agg_employees[entry['product']][entry['subperiod']] += 0  # placeholder
    
    # Recompute: for each product and subperiod, how many employees needed?
    # Total production needed per product per subperiod
    prod_by_ps = defaultdict(float)  # (product, subperiod) -> total quantity
    for entry in production_entries:
        prod_by_ps[(entry['product'], entry['subperiod'])] += entry['quantity']
    
    emp_by_ps = {}
    for (pid, sid), qty in prod_by_ps.items():
        rate = product_by_id[pid]['production_rate_per_subperiod']
        emp_by_ps[(pid, sid)] = math.ceil(qty / rate)
    
    # Compute setups: a setup is needed when employees start producing a product
    # Setup starts when employees are first assigned or increase
    # For simplicity, track when a product first appears or employees increase
    
    # Get all products and their active subperiods sorted
    product_active_subs = defaultdict(list)
    for (pid, sid) in emp_by_ps:
        if emp_by_ps[(pid, sid)] > 0:
            product_active_subs[pid].append(sid)
    
    for pid in product_active_subs:
        product_active_subs[pid].sort()
    
    setups_by_ps = defaultdict(int)
    for pid, subs in product_active_subs.items():
        prev_emp = 0
        for sid in subs:
            emp = emp_by_ps.get((pid, sid), 0)
            if emp > prev_emp:
                setups_by_ps[(pid, sid)] = emp - prev_emp
            # Check if there's a gap (non-consecutive subperiods)
            idx_in_list = subs.index(sid)
            if idx_in_list > 0 and subs[idx_in_list - 1] != sid - 1:
                # Gap detected, need setup for all employees
                setups_by_ps[(pid, sid)] = emp
                prev_emp = 0
            
            if emp > prev_emp:
                setups_by_ps[(pid, sid)] = emp - prev_emp
            prev_emp = emp
    
    # Recompute setups more carefully
    setups_by_ps = {}
    for pid, subs in product_active_subs.items():
        prev_emp = 0
        prev_sid = -999
        for sid in subs:
            emp = emp_by_ps.get((pid, sid), 0)
            if sid != prev_sid + 1:
                # Discontinuity: all employees need setup
                setups_by_ps[(pid, sid)] = emp
            elif emp > prev_emp:
                setups_by_ps[(pid, sid)] = emp - prev_emp
            else:
                setups_by_ps[(pid, sid)] = 0
            prev_emp = emp
            prev_sid = sid
    
    # Workstations by period: max employees across subperiods in each period
    # Actually: total employees producing + setting up across all products in each subperiod
    # Then max per period
    
    total_emp_by_subperiod = defaultdict(int)
    for (pid, sid), emp in emp_by_ps.items():
        total_emp_by_subperiod[sid] += emp
    # Add setup employees (they occupy workstations during setup)
    for (pid, sid), setups in setups_by_ps.items():
        if setups > 0:
            # Setup occupies employees for setup_length_subperiods
            for ds in range(setup_length_subperiods):
                s = sid + ds
                if s < num_subperiods:
                    total_emp_by_subperiod[s] += 0  # setup employees are counted in emp already
                    # Actually, in the model, setup employees are separate from producing employees
                    # But for simplicity, we count setups as needing employees too
    
    # For the certificate, workstations_by_period = max employees needed in any subperiod of that period
    workstations_by_period = []
    for pid_p in range(num_periods):
        max_emp = 0
        for sid in period_subperiods[pid_p]:
            emp = total_emp_by_subperiod.get(sid, 0)
            max_emp = max(max_emp, emp)
        workstations_by_period.append(min(max_emp, num_workstations))
    
    # Shift starts by period
    # An employee shift starts in a period when an employee first begins working
    # Simple model: all employees needed in a period start their shift in that period
    shift_starts_by_period = []
    prev_ws = 0
    for pid_p in range(num_periods):
        ws = workstations_by_period[pid_p]
        if ws > prev_ws:
            shift_starts_by_period.append(ws - prev_ws)
        else:
            shift_starts_by_period.append(0)
        prev_ws = ws
    
    # ---- COMPUTE OBJECTIVE ----
    
    # Route costs
    total_driver_salary = 0
    total_vehicle_cost = 0
    
    route_outputs = []
    for vi, route in enumerate(routes):
        trips_output = []
        
        # Compute route duration
        earliest_start = min(period_start[sp] for sp, clist in route)
        latest_return = earliest_start  # will update
        total_travel_time = 0
        
        for trip_idx, (sp, clist) in enumerate(route):
            feas, tt, arrivals, ret = evaluate_trip(clist, sp)
            total_travel_time += tt
            if ret > latest_return:
                latest_return = ret
            
            trips_output.append({
                'trip': trip_idx,
                'customers': clist,
                'start_period': sp
            })
        
        route_duration = latest_return - earliest_start
        route_duration = min(route_duration, max_shift_route_length_minutes)
        paid_time = max(route_duration, min_paid_time_minutes)
        
        driver_cost = driver_salary_per_minute * paid_time
        vehicle_cost = vehicle_op_cost_per_minute * total_travel_time
        
        total_driver_salary += driver_cost
        total_vehicle_cost += vehicle_cost
        
        route_outputs.append({
            'vehicle': vi,
            'trips': trips_output
        })
    
    # Production costs
    total_employee_salary = employee_salary_per_subperiod * sum(workstations_by_period) * subperiods_per_period
    # Actually: employee salary = employee_salary_per_subperiod * sum over periods of workstations_by_period * subperiods_per_period?
    # No: "employee salary rate per subperiod times the period-level paid-workstation count"
    # So it's employee_salary_per_subperiod * sum_periods(workstations_by_period[p]) * subperiods_per_period? 
    # Or is it per period? Let me re-read...
    # "Production cost is the employee salary rate per subperiod times the period-level paid-workstation count"
    # This means: for each period, cost = employee_salary_per_subperiod * workstations_by_period[p]
    # Wait, that doesn't make dimensional sense. Let me assume:
    # production_salary_cost = sum over periods of (employee_salary_per_subperiod * subperiods_per_period * workstations_by_period[p])
    # Or: employee_salary_per_subperiod * (sum of workstations across all subperiods)
    
    # Actually re-reading: "employee salary rate per subperiod times the period-level paid-workstation count"
    # I think it means: for each period p, the cost is employee_salary_per_subperiod * workstations_by_period[p] * subperiods_per_period
    # Because workstations_by_period[p] employees are paid for all subperiods in that period
    
    total_employee_salary = 0
    for pid_p in range(num_periods):
        total_employee_salary += employee_salary_per_subperiod * workstations_by_period[pid_p] * subperiods_per_period
    
    total_setup_cost = 0
    for (pid, sid), s in setups_by_ps.items():
        total_setup_cost += setup_cost * s
    
    objective = total_driver_salary + total_vehicle_cost + total_employee_salary + total_setup_cost
    
    # Build employees_by_product_subperiod and setups_by_product_subperiod
    emp_list = []
    for (pid, sid), emp in emp_by_ps.items():
        if emp > 0:
            emp_list.append({
                'product': pid,
                'subperiod': sid,
                'employees': emp
            })
    
    setup_list = []
    for (pid, sid), s in setups_by_ps.items():
        if s > 0:
            setup_list.append({
                'product': pid,
                'subperiod': sid,
                'setups': s
            })
    
    solution = {
        'objective_value': objective,
        'routes': route_outputs,
        'production_schedule': production_entries,
        'production_staffing': {
            'employees_by_product_subperiod': emp_list,
            'setups_by_product_subperiod': setup_list,
            'workstations_by_period': workstations_by_period,
            'shift_starts_by_period': shift_starts_by_period
        }
    }
    
    if logger:
        logger.log_solution(objective, solution)
    
    # ---- IMPROVEMENT PHASE: Try to use later departure periods to reduce production urgency ----
    # Try shifting trips to later periods where feasible, which might allow more relaxed production
    
    if time_remaining() > 5:
        for ri, route in enumerate(routes):
            for ti in range(len(route) - 1, -1, -1):
                sp, clist = route[ti]
                max_period = num_periods - 1
                if ti < len(route) - 1:
                    max_period = route[ti + 1][0] - 1
                
                best_sp = sp
                for pid_try in range(max_period, sp, -1):
                    feas, tt, arrivals, ret = evaluate_trip(clist, pid_try)
                    if feas:
                        best_sp = pid_try
                        break
                
                if best_sp != sp:
                    routes[ri][ti] = (best_sp, clist)
    
    # Recompute everything after potential improvement
    # Recompute order departure info
    for route in routes:
        for sp, clist in route:
            for cid in clist:
                order_departure_period[cid] = sp
                dep_time = period_start[sp]
                latest_sub = -1
                for sid in range(num_subperiods):
                    if subperiod_end[sid] <= dep_time:
                        latest_sub = sid
                order_departure_subperiod[cid] = latest_sub
    
    # Recompute production schedule
    production_entries = []
    prod_by_ps = defaultdict(float)
    
    for pid_prod, task_list in tasks_by_product.items():
        prod = product_by_id[pid_prod]
        rate = prod['production_rate_per_subperiod']
        
        for oid, qty, fsubs_orig in task_list:
            if oid not in order_departure_subperiod:
                continue
            dep_sub = order_departure_subperiod[oid]
            
            # Recompute feasible subperiods based on updated departure
            pw = production_windows[str(oid)][str(pid_prod)]
            valid_subs = pw['valid_subperiod_ids']
            feasible_subs = sorted([s for s in valid_subs if s <= dep_sub])
            
            if not feasible_subs:
                feasible_subs = sorted(valid_subs)
            
            remaining_qty = qty
            for sid in feasible_subs:
                if remaining_qty <= 0:
                    break
                produce_qty = min(remaining_qty, rate)
                production_entries.append({
                    'order': oid,
                    'product': pid_prod,
                    'subperiod': sid,
                    'quantity': produce_qty
                })
                prod_by_ps[(pid_prod, sid)] += produce_qty
                remaining_qty -= produce_qty
            
            # If still remaining, use more employees per subperiod
            if remaining_qty > 0:
                for sid in feasible_subs:
                    if remaining_qty <= 0:
                        break
                    additional_emp = math.ceil(remaining_qty / rate)
                    produce_qty = min(remaining_qty, rate * additional_emp)
                    production_entries.append({
                        'order': oid,
                        'product': pid_prod,
                        'subperiod': sid,
                        'quantity': produce_qty
                    })
                    prod_by_ps[(pid_prod, sid)] += produce_qty
                    remaining_qty -= produce_qty
    
    # Recompute staffing
    emp_by_ps = {}
    for (pid_prod, sid), qty in prod_by_ps.items():
        rate = product_by_id[pid_prod]['production_rate_per_subperiod']
        emp_by_ps[(pid_prod, sid)] = math.ceil(qty / rate)
    
    product_active_subs = defaultdict(list)
    for (pid_prod, sid) in emp_by_ps:
        if emp_by_ps[(pid_prod, sid)] > 0:
            product_active_subs[pid_prod].append(sid)
    for pid_prod in product_active_subs:
        product_active_subs[pid_prod].sort()
    
    setups_by_ps = {}
    for pid_prod, subs in product_active_subs.items():
        prev_emp = 0
        prev_sid = -999
        for sid in subs:
            emp = emp_by_ps.get((pid_prod, sid), 0)
            if sid != prev_sid + 1:
                setups_by_ps[(pid_prod, sid)] = emp
            elif emp > prev_emp:
                setups_by_ps[(pid_prod, sid)] = emp - prev_emp
            else:
                setups_by_ps[(pid_prod, sid)] = 0
            prev_emp = emp
            prev_sid = sid
    
    total_emp_by_subperiod = defaultdict(int)
    for (pid_prod, sid), emp in emp_by_ps.items():
        total_emp_by_subperiod[sid] += emp
    
    workstations_by_period = []
    for pid_p in range(num_periods):
        max_emp = 0
        for sid in period_subperiods[pid_p]:
            emp = total_emp_by_subperiod.get(sid, 0)
            max_emp = max(max_emp, emp)
        workstations_by_period.append(min(max_emp, num_workstations))
    
    shift_starts_by_period = []
    prev_ws = 0
    for pid_p in range(num_periods):
        ws = workstations_by_period[pid_p]
        if ws > prev_ws:
            shift_starts_by_period.append(ws - prev_ws)
        else:
            shift_starts_by_period.append(0)
        prev_ws = ws
    
    # Recompute objective
    total_driver_salary = 0
    total_vehicle_cost = 0
    
    route_outputs = []
    for vi, route in enumerate(routes):
        trips_output = []
        earliest_start = min(period_start[sp] for sp, clist in route)
        latest_return = earliest_start
        total_travel_time = 0
        
        for trip_idx, (sp, clist) in enumerate(route):
            feas, tt, arrivals, ret = evaluate_trip(clist, sp)
            total_travel_time += tt
            if ret > latest_return:
                latest_return = ret
            
            trips_output.append({
                'trip': trip_idx,
                'customers': clist,
                'start_period': sp
            })
        
        route_duration = latest_return - earliest_start
        route_duration = min(route_duration, max_shift_route_length_minutes)
        paid_time = max(route_duration, min_paid_time_minutes)
        
        driver_cost = driver_salary_per_minute * paid_time
        vehicle_cost = vehicle_op_cost_per_minute * total_travel_time
        
        total_driver_salary += driver_cost
        total_vehicle_cost += vehicle_cost
        
        route_outputs.append({
            'vehicle': vi,
            'trips': trips_output
        })
    
    total_employee_salary = 0
    for pid_p in range(num_periods):
        total_employee_salary += employee_salary_per_subperiod * workstations_by_period[pid_p] * subperiods_per_period
    
    total_setup_cost = 0
    for (pid_prod, sid), s in setups_by_ps.items():
        total_setup_cost += setup_cost * s
    
    objective = total_driver_salary + total_vehicle_cost + total_employee_salary + total_setup_cost
    
    emp_list = []
    for (pid_prod, sid), emp in emp_by_ps.items():
        if emp > 0:
            emp_list.append({
                'product': pid_prod,
                'subperiod': sid,
                'employees': emp
            })
    
    setup_list = []
    for (pid_prod, sid), s in setups_by_ps.items():
        if s > 0:
            setup_list.append({
                'product': pid_prod,
                'subperiod': sid,
                'setups': s
            })
    
    solution = {
        'objective_value': objective,
        'routes': route_outputs,
        'production_schedule': production_entries,
        'production_staffing': {
            'employees_by_product_subperiod': emp_list,
            'setups_by_product_subperiod': setup_list,
            'workstations_by_period': workstations_by_period,
            'shift_starts_by_period': shift_starts_by_period
        }
    }
    
    if logger:
        logger.log_solution(objective, solution)
    
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=2)
    
    return solution

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)
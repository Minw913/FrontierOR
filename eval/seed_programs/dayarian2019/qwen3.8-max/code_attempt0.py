import json
import argparse
import time
import math
import sys
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Parse instance
    num_orders = instance['num_orders']
    num_products = instance['num_products']
    gp = instance['global_parameters']
    orders = instance['orders']
    products = instance['products']
    travel_time = instance['travel_time_matrix']
    production_windows = instance['production_windows']
    periods = instance['periods']
    subperiods = instance['subperiods']
    num_periods = gp['num_periods']
    num_subperiods = gp['num_subperiods']
    period_length = gp['period_length_minutes']
    subperiod_length = gp['subperiod_length_minutes']
    prep_time = gp['preparation_time_minutes']
    service_time_default = gp['service_time_minutes']
    vehicle_capacity = instance['vehicle_capacity']
    num_vehicles = instance['num_vehicles']
    num_drivers = instance['num_drivers']
    max_trips = gp['max_trips_per_route']
    max_route_length_min = gp['max_shift_route_length_minutes']
    min_paid_time_min = gp['min_paid_time_minutes']
    driver_salary = gp['driver_salary_per_minute']
    vehicle_cost_rate = gp['vehicle_operating_cost_per_minute']
    emp_salary = gp['employee_salary_per_subperiod']
    setup_cost = gp['setup_cost']
    setup_length_sub = gp['setup_length_subperiods']
    num_workstations = instance['num_workstations']
    num_prod_employees = instance['num_production_employees']
    min_paid_time_sub = gp['min_paid_time_subperiods']
    subperiods_per_period = gp['subperiods_per_period']
    horizon_start = gp['horizon_start_minutes']
    horizon_end = gp['horizon_end_minutes']

    # Product info
    prod_rate = {}
    prod_life = {}
    for p in products:
        prod_rate[p['product_id']] = p['production_rate_per_subperiod']
        prod_life[p['product_id']] = p['life_span_minutes']

    # Order info
    order_map = {}
    for o in orders:
        order_map[o['order_id']] = o

    def get_service_time(oid):
        return order_map[oid].get('service_time', service_time_default)

    def get_space(oid):
        return order_map[oid]['space_requirement']

    def get_tw(oid):
        return order_map[oid]['time_window_start'], order_map[oid]['time_window_end']

    def compute_trip_times(customers, departure_time):
        """Compute arrival/departure times for a trip. Returns (feasible, times, return_time)."""
        current_time = departure_time
        times = []
        prev_loc = 0  # depot
        for cid in customers:
            travel = travel_time[prev_loc][cid]
            arrival = current_time + travel
            tw_s, tw_e = get_tw(cid)
            if arrival > tw_e:
                return False, None, None
            effective_arrival = max(arrival, tw_s)
            svc = get_service_time(cid)
            departure = effective_arrival + svc
            times.append((arrival, effective_arrival, departure))
            current_time = departure
            prev_loc = cid
        # Return to depot
        return_time = current_time + travel_time[prev_loc][0]
        return True, times, return_time

    def find_start_period_for_trip(customers):
        """Find the earliest period where this trip is feasible."""
        for p_idx in range(num_periods):
            dep_time = periods[p_idx]['start_time'] + prep_time
            feasible, _, ret_time = compute_trip_times(customers, dep_time)
            if feasible and ret_time <= horizon_end:
                # Check max route length
                if ret_time - periods[p_idx]['start_time'] <= max_route_length_min:
                    return p_idx
        return -1

    def compute_route_travel_time(customers):
        """Compute total travel time for a sequence of customers (depot -> customers -> depot)."""
        if not customers:
            return 0
        total = travel_time[0][customers[0]]
        for i in range(len(customers) - 1):
            total += travel_time[customers[i]][customers[i + 1]]
        total += travel_time[customers[-1]][0]
        return total

    # ============ ROUTING HEURISTIC ============
    
    def build_initial_routes():
        """Build routes using nearest neighbor with time windows."""
        unassigned = set(range(1, num_orders + 1))
        routes = []  # list of lists of customers
        
        # Sort by time window end (earliest deadline first)
        sorted_orders = sorted(unassigned, key=lambda x: get_tw(x)[1])
        
        while unassigned:
            if time.time() - start_time > args.time_limit * 0.5:
                break
            
            # Start a new route with the earliest deadline unassigned order
            seed = None
            for oid in sorted_orders:
                if oid in unassigned:
                    seed = oid
                    break
            if seed is None:
                break
            
            route = [seed]
            unassigned.remove(seed)
            route_space = get_space(seed)
            
            # Try to add more customers
            improved = True
            while improved:
                improved = False
                best_oid = None
                best_cost = float('inf')
                best_pos = -1
                
                for oid in list(unassigned):
                    if route_space + get_space(oid) > vehicle_capacity:
                        continue
                    # Try inserting at each position
                    for pos in range(len(route) + 1):
                        new_route = route[:pos] + [oid] + route[pos:]
                        # Check feasibility with earliest possible departure
                        # Try all periods
                        feasible = False
                        for p_idx in range(num_periods):
                            dep_time = periods[p_idx]['start_time'] + prep_time
                            feas, _, ret = compute_trip_times(new_route, dep_time)
                            if feas and ret <= horizon_end:
                                feasible = True
                                break
                        if feasible:
                            # Cost = additional travel time
                            cost = compute_route_travel_time(new_route) - compute_route_travel_time(route)
                            if cost < best_cost:
                                best_cost = cost
                                best_oid = oid
                                best_pos = pos
                
                if best_oid is not None:
                    route = route[:best_pos] + [best_oid] + route[best_pos:]
                    unassigned.remove(best_oid)
                    route_space += get_space(best_oid)
                    improved = True
            
            routes.append(route)
        
        # Handle any remaining unassigned orders
        for oid in list(unassigned):
            routes.append([oid])
            unassigned.remove(oid)
        
        return routes

    def assign_periods_to_routes(routes):
        """Assign start periods to trips for each vehicle/route."""
        # Group routes into vehicles (max_trips per vehicle)
        # Simple: assign routes to vehicles, up to max_trips per vehicle
        vehicle_routes = []  # list of (vehicle_id, [trip1_customers, trip2_customers, ...])
        
        route_periods = []
        for route in routes:
            p = find_start_period_for_trip(route)
            route_periods.append(p)
        
        # Sort routes by their period
        indexed_routes = sorted(enumerate(routes), key=lambda x: route_periods[x[0]] if route_periods[x[0]] >= 0 else float('inf'))
        
        vehicles = []
        current_vehicle_trips = []
        current_vehicle = 0
        last_period = -1
        
        for idx, route in indexed_routes:
            p = route_periods[idx]
            if p < 0:
                # Try to find any feasible period
                p = 0
                for p_try in range(num_periods):
                    dep_time = periods[p_try]['start_time'] + prep_time
                    feas, _, ret = compute_trip_times(route, dep_time)
                    if feas and ret <= horizon_end:
                        p = p_try
                        break
            
            if len(current_vehicle_trips) >= max_trips or (current_vehicle_trips and p <= last_period):
                # Need a new vehicle or try to fit
                if current_vehicle_trips and p > last_period and len(current_vehicle_trips) < max_trips:
                    current_vehicle_trips.append((route, p))
                    last_period = p
                else:
                    if current_vehicle_trips:
                        vehicles.append((current_vehicle, current_vehicle_trips))
                        current_vehicle += 1
                    current_vehicle_trips = [(route, p)]
                    last_period = p
            else:
                current_vehicle_trips.append((route, p))
                last_period = p
        
        if current_vehicle_trips:
            vehicles.append((current_vehicle, current_vehicle_trips))
        
        return vehicles

    def try_improve_routes(routes, remaining_time):
        """Try to improve routes with local search."""
        # Simple 2-opt within each route
        for r_idx in range(len(routes)):
            if time.time() - start_time > start_time + remaining_time:
                break
            route = routes[r_idx]
            if len(route) <= 2:
                continue
            
            improved = True
            while improved:
                improved = False
                for i in range(len(route) - 1):
                    for j in range(i + 1, len(route)):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        # Check feasibility
                        feasible = False
                        for p_idx in range(num_periods):
                            dep_time = periods[p_idx]['start_time'] + prep_time
                            feas, _, ret = compute_trip_times(new_route, dep_time)
                            if feas and ret <= horizon_end:
                                feasible = True
                                break
                        if feasible:
                            old_tt = compute_route_travel_time(route)
                            new_tt = compute_route_travel_time(new_route)
                            if new_tt < old_tt - 1e-6:
                                route = new_route
                                improved = True
                                break
                    if improved:
                        break
            routes[r_idx] = route
        return routes

    # ============ PRODUCTION SCHEDULING ============
    
    def solve_production(vehicles, remaining_time):
        """Solve production scheduling given routes using Gurobi."""
        import gurobipy as gp_model
        from gurobipy import GRB
        
        # Determine due time for each order (based on trip departure)
        order_due_time = {}  # order_id -> departure time in minutes
        order_departure_subperiod = {}  # order_id -> latest subperiod for production
        
        for vid, trips in vehicles:
            for trip_customers, start_period in trips:
                dep_time = periods[start_period]['start_time'] + prep_time
                for oid in trip_customers:
                    order_due_time[oid] = dep_time
                    # Find the latest subperiod that ends before departure
                    due_sub = -1
                    for s in range(num_subperiods):
                        if subperiods[s]['end_time'] <= dep_time:
                            due_sub = s
                    order_departure_subperiod[oid] = due_sub
        
        # Collect all (order, product) demands and valid subperiods
        demands = {}  # (order_id, product_id) -> quantity
        valid_subs = {}  # (order_id, product_id) -> list of valid subperiod ids
        
        for o in orders:
            oid = o['order_id']
            for pid_str, qty in o['demands'].items():
                pid = int(pid_str)
                demands[(oid, pid)] = qty
                # Get valid subperiods from production_windows
                pw = production_windows.get(str(oid), {}).get(str(pid), {})
                valid_sub_ids = pw.get('valid_subperiod_ids', [])
                # Filter by due time
                due_sub = order_departure_subperiod.get(oid, num_subperiods - 1)
                filtered = [s for s in valid_sub_ids if s <= due_sub]
                valid_subs[(oid, pid)] = filtered
        
        # Try Gurobi MIP for production
        try:
            model = gp_model.Model("production")
            model.setParam('Seed', 0)
            model.setParam('MIPGap', 1e-4)
            model.setParam('NumericFocus', 0)
            model.setParam('Threads', 1)
            model.setParam('TimeLimit', max(1, remaining_time * 0.6))
            model.setParam('OutputFlag', 0)
            
            # Decision variables
            # q[o, p, s] = quantity produced
            q_vars = {}
            for (oid, pid), subs in valid_subs.items():
                for s in subs:
                    q_vars[(oid, pid, s)] = model.addVar(lb=0, ub=demands[(oid, pid)], 
                                                          name=f"q_{oid}_{pid}_{s}")
            
            # e[p, s] = employees producing product p in subperiod s
            all_products = [p['product_id'] for p in products]
            e_vars = {}
            for pid in all_products:
                for s in range(num_subperiods):
                    e_vars[(pid, s)] = model.addVar(lb=0, ub=num_prod_employees, vtype=GRB.INTEGER,
                                                     name=f"e_{pid}_{s}")
            
            # u[p, s] = setup starts for product p in subperiod s
            u_vars = {}
            for pid in all_products:
                for s in range(num_subperiods):
                    u_vars[(pid, s)] = model.addVar(lb=0, ub=num_prod_employees, vtype=GRB.INTEGER,
                                                     name=f"u_{pid}_{s}")
            
            # w[t] = workstations paid in period t
            w_vars = {}
            for t in range(num_periods):
                w_vars[t] = model.addVar(lb=0, ub=num_workstations, vtype=GRB.INTEGER,
                                          name=f"w_{t}")
            
            # shift_starts[t]
            shift_vars = {}
            for t in range(num_periods):
                shift_vars[t] = model.addVar(lb=0, ub=num_prod_employees, vtype=GRB.INTEGER,
                                              name=f"shift_{t}")
            
            model.update()
            
            # Objective: minimize production cost
            obj = gp_model.LinExpr()
            for t in range(num_periods):
                obj += emp_salary * subperiods_per_period * w_vars[t]
            for pid in all_products:
                for s in range(num_subperiods):
                    obj += setup_cost * u_vars[(pid, s)]
            model.setObjective(obj, GRB.MINIMIZE)
            
            # Constraint: demand satisfaction
            for (oid, pid), qty in demands.items():
                subs = valid_subs[(oid, pid)]
                if subs:
                    model.addConstr(
                        gp_model.quicksum(q_vars.get((oid, pid, s), 0) for s in subs) >= qty,
                        name=f"demand_{oid}_{pid}"
                    )
            
            # Constraint: production capacity
            for pid in all_products:
                rate = prod_rate[pid]
                for s in range(num_subperiods):
                    total_prod = gp_model.LinExpr()
                    for (oid, p) in demands:
                        if p == pid and s in valid_subs.get((oid, p), []):
                            if (oid, pid, s) in q_vars:
                                total_prod += q_vars[(oid, pid, s)]
                    model.addConstr(total_prod <= rate * e_vars[(pid, s)],
                                    name=f"cap_{pid}_{s}")
            
            # Constraint: employee rise requires setups
            for pid in all_products:
                for s in range(num_subperiods):
                    if s == 0:
                        model.addConstr(e_vars[(pid, s)] <= u_vars[(pid, s)],
                                        name=f"rise_{pid}_{s}")
                    else:
                        model.addConstr(e_vars[(pid, s)] <= e_vars[(pid, s-1)] + u_vars[(pid, s)],
                                        name=f"rise_{pid}_{s}")
            
            # Constraint: total employees (producing + in setup) <= limit
            for s in range(num_subperiods):
                total_emp = gp_model.LinExpr()
                for pid in all_products:
                    total_emp += e_vars[(pid, s)]
                    # Setup occupied: setups started in [s - setup_length_sub + 1, s]
                    for s2 in range(max(0, s - setup_length_sub + 1), s + 1):
                        total_emp += u_vars[(pid, s2)]
                model.addConstr(total_emp <= num_prod_employees, name=f"emp_limit_{s}")
                model.addConstr(total_emp <= num_workstations, name=f"ws_limit_{s}")
            
            # Constraint: workstations per period
            for t in range(num_periods):
                sub_start = t * subperiods_per_period
                sub_end = min((t + 1) * subperiods_per_period, num_subperiods)
                for s in range(sub_start, sub_end):
                    total_emp = gp_model.LinExpr()
                    for pid in all_products:
                        total_emp += e_vars[(pid, s)]
                        for s2 in range(max(0, s - setup_length_sub + 1), s + 1):
                            total_emp += u_vars[(pid, s2)]
                    model.addConstr(w_vars[t] >= total_emp, name=f"ws_period_{t}_{s}")
            
            # Constraint: minimum paid time
            # Cumulative constraint
            for t in range(num_periods):
                cum_ws = gp_model.LinExpr()
                cum_shift = gp_model.LinExpr()
                for t2 in range(t + 1):
                    cum_ws += w_vars[t2] * subperiods_per_period
                    cum_shift += shift_vars[t2] * min_paid_time_sub
                model.addConstr(cum_ws >= cum_shift, name=f"min_paid_{t}")
            
            # Constraint: total shifts = total employees used
            # (approximate: total shifts should cover the workforce)
            # shift_starts sum should relate to workforce
            
            model.optimize()
            
            if model.SolCount > 0:
                # Extract solution
                prod_schedule = []
                for (oid, pid), subs in valid_subs.items():
                    for s in subs:
                        if (oid, pid, s) in q_vars:
                            val = q_vars[(oid, pid, s)].X
                            if val > 1e-6:
                                prod_schedule.append({
                                    'order': oid,
                                    'product': pid,
                                    'subperiod': s,
                                    'quantity': round(val, 6)
                                })
                
                emp_by_prod_sub = []
                for pid in all_products:
                    for s in range(num_subperiods):
                        val = int(round(e_vars[(pid, s)].X))
                        if val > 0:
                            emp_by_prod_sub.append({
                                'product': pid,
                                'subperiod': s,
                                'employees': val
                            })
                
                setups_by_prod_sub = []
                for pid in all_products:
                    for s in range(num_subperiods):
                        val = int(round(u_vars[(pid, s)].X))
                        if val > 0:
                            setups_by_prod_sub.append({
                                'product': pid,
                                'subperiod': s,
                                'setups': val
                            })
                
                workstations_by_period = [int(round(w_vars[t].X)) for t in range(num_periods)]
                shift_starts_by_period = [int(round(shift_vars[t].X)) for t in range(num_periods)]
                
                return prod_schedule, emp_by_prod_sub, setups_by_prod_sub, workstations_by_period, shift_starts_by_period
            
        except Exception as ex:
            pass
        
        # Fallback: greedy production scheduling
        return greedy_production(demands, valid_subs, order_departure_subperiod)

    def greedy_production(demands, valid_subs, order_departure_subperiod):
        """Greedy production scheduling fallback."""
        prod_schedule = []
        emp_by_prod_sub = defaultdict(int)
        
        # For each (order, product), produce as late as possible (just-in-time)
        for (oid, pid), qty in demands.items():
            subs = valid_subs.get((oid, pid), [])
            if not subs:
                continue
            remaining = qty
            rate = prod_rate[pid]
            # Produce in latest valid subperiods first
            for s in sorted(subs, reverse=True):
                if remaining <= 0:
                    break
                produce = min(remaining, rate)  # One employee
                prod_schedule.append({
                    'order': oid,
                    'product': pid,
                    'subperiod': s,
                    'quantity': round(produce, 6)
                })
                remaining -= produce
                emp_by_prod_sub[(pid, s)] += 1
        
        # Convert to list format
        emp_list = [{'product': pid, 'subperiod': s, 'employees': cnt} 
                    for (pid, s), cnt in emp_by_prod_sub.items() if cnt > 0]
        
        # Compute setups (simplified: one setup per product per contiguous block)
        setups_list = []
        total_setups = 0
        for pid in [p['product_id'] for p in products]:
            prev_active = False
            for s in range(num_subperiods):
                active = emp_by_prod_sub.get((pid, s), 0) > 0
                if active and not prev_active:
                    setups_list.append({'product': pid, 'subperiod': s, 'setups': emp_by_prod_sub[(pid, s)]})
                    total_setups += emp_by_prod_sub[(pid, s)]
                prev_active = active
        
        # Workstations per period
        workstations_by_period = []
        for t in range(num_periods):
            sub_start = t * subperiods_per_period
            sub_end = min((t + 1) * subperiods_per_period, num_subperiods)
            max_emp = 0
            for s in range(sub_start, sub_end):
                total = sum(emp_by_prod_sub.get((pid, s), 0) for pid in [p['product_id'] for p in products])
                max_emp = max(max_emp, total)
            workstations_by_period.append(min(max_emp, num_workstations))
        
        # Shift starts (simplified)
        shift_starts_by_period = [0] * num_periods
        total_shifts = sum(workstations_by_period)
        if total_shifts > 0:
            # Put all shifts in first period with workstations
            for t in range(num_periods):
                if workstations_by_period[t] > 0:
                    shift_starts_by_period[t] = workstations_by_period[t]
                    break
        
        return prod_schedule, emp_list, setups_list, workstations_by_period, shift_starts_by_period

    def compute_route_cost(vehicles):
        """Compute total routing cost."""
        total_cost = 0.0
        for vid, trips in vehicles:
            if not trips:
                continue
            # Route duration: from earliest trip start to latest return
            earliest_start = float('inf')
            latest_return = 0
            total_travel = 0.0
            
            for trip_customers, start_period in trips:
                dep_time = periods[start_period]['start_time'] + prep_time
                earliest_start = min(earliest_start, periods[start_period]['start_time'])
                
                # Compute trip travel time and return time
                tt = compute_route_travel_time(trip_customers)
                total_travel += tt
                
                # Compute return time
                current_time = dep_time
                prev_loc = 0
                for cid in trip_customers:
                    travel = travel_time[prev_loc][cid]
                    arrival = current_time + travel
                    tw_s, tw_e = get_tw(cid)
                    effective_arrival = max(arrival, tw_s)
                    svc = get_service_time(cid)
                    current_time = effective_arrival + svc
                    prev_loc = cid
                return_time = current_time + travel_time[prev_loc][0]
                latest_return = max(latest_return, return_time)
            
            route_duration = min(latest_return - earliest_start, max_route_length_min)
            paid_time = max(route_duration, min_paid_time_min)
            
            driver_cost = driver_salary * paid_time
            vehicle_op_cost = vehicle_cost_rate * total_travel
            total_cost += driver_cost + vehicle_op_cost
        
        return total_cost

    def compute_production_cost(workstations_by_period, setups_by_prod_sub):
        """Compute total production cost."""
        emp_cost = emp_salary * sum(w * subperiods_per_period for w in workstations_by_period)
        total_setups = sum(s['setups'] for s in setups_by_prod_sub)
        setup_total_cost = setup_cost * total_setups
        return emp_cost + setup_total_cost

    def build_solution_dict(vehicles, prod_schedule, emp_by_prod_sub, setups_by_prod_sub, 
                           workstations_by_period, shift_starts_by_period, objective):
        """Build the solution dictionary."""
        routes_out = []
        for vid, trips in vehicles:
            if not trips:
                continue
            trip_list = []
            for t_idx, (customers, start_period) in enumerate(trips):
                trip_list.append({
                    'trip': t_idx,
                    'customers': customers,
                    'start_period': start_period
                })
            routes_out.append({
                'vehicle': vid,
                'trips': trip_list
            })
        
        solution = {
            'objective_value': objective,
            'routes': routes_out,
            'production_schedule': prod_schedule,
            'production_staffing': {
                'employees_by_product_subperiod': emp_by_prod_sub,
                'setups_by_product_subperiod': setups_by_prod_sub,
                'workstations_by_period': workstations_by_period,
                'shift_starts_by_period': shift_starts_by_period
            }
        }
        return solution

    # ============ MAIN SOLVER ============
    
    elapsed = time.time() - start_time
    remaining = args.time_limit - elapsed
    
    # Build initial routes
    routes = build_initial_routes()
    
    # Try to improve
    elapsed = time.time() - start_time
    remaining = args.time_limit - elapsed
    if remaining > 5:
        routes = try_improve_routes(routes, remaining * 0.3)
    
    # Assign to vehicles
    vehicles = assign_periods_to_routes(routes)
    
    # Solve production
    elapsed = time.time() - start_time
    remaining = max(1, args.time_limit - elapsed)
    prod_schedule, emp_by_prod_sub, setups_by_prod_sub, workstations_by_period, shift_starts_by_period = \
        solve_production(vehicles, remaining)
    
    # Compute cost
    route_cost = compute_route_cost(vehicles)
    prod_cost = compute_production_cost(workstations_by_period, setups_by_prod_sub)
    objective = route_cost + prod_cost
    
    # Build solution
    solution = build_solution_dict(vehicles, prod_schedule, emp_by_prod_sub, setups_by_prod_sub,
                                   workstations_by_period, shift_starts_by_period, objective)
    
    # Log solution
    if logger:
        logger.log_solution(objective, solution)
    
    # Try to improve further if time allows
    elapsed = time.time() - start_time
    if elapsed < args.time_limit * 0.8:
        # Try alternative route constructions
        best_objective = objective
        best_solution = solution
        
        # Try different orderings
        for attempt in range(3):
            if time.time() - start_time > args.time_limit * 0.8:
                break
            
            # Shuffle and rebuild
            import random
            random.seed(attempt + 42)
            shuffled_orders = list(range(1, num_orders + 1))
            random.shuffle(shuffled_orders)
            
            # Rebuild routes with different order
            unassigned = set(shuffled_orders)
            new_routes = []
            while unassigned:
                seed = min(unassigned, key=lambda x: get_tw(x)[1])
                route = [seed]
                unassigned.remove(seed)
                route_space = get_space(seed)
                
                for oid in sorted(unassigned, key=lambda x: get_tw(x)[1]):
                    if route_space + get_space(oid) > vehicle_capacity:
                        continue
                    # Try appending
                    new_route = route + [oid]
                    feasible = False
                    for p_idx in range(num_periods):
                        dep_time = periods[p_idx]['start_time'] + prep_time
                        feas, _, ret = compute_trip_times(new_route, dep_time)
                        if feas and ret <= horizon_end:
                            feasible = True
                            break
                    if feasible:
                        route = new_route
                        unassigned.remove(oid)
                        route_space += get_space(oid)
                
                new_routes.append(route)
            
            for oid in list(unassigned):
                new_routes.append([oid])
                unassigned.remove(oid)
            
            new_vehicles = assign_periods_to_routes(new_routes)
            
            elapsed = time.time() - start_time
            remaining = max(1, args.time_limit - elapsed)
            new_prod = solve_production(new_vehicles, remaining * 0.5)
            new_prod_schedule, new_emp, new_setups, new_ws, new_shifts = new_prod
            
            new_route_cost = compute_route_cost(new_vehicles)
            new_prod_cost = compute_production_cost(new_ws, new_setups)
            new_objective = new_route_cost + new_prod_cost
            
            if new_objective < best_objective:
                best_objective = new_objective
                best_solution = build_solution_dict(new_vehicles, new_prod_schedule, new_emp, 
                                                    new_setups, new_ws, new_shifts, new_objective)
                if logger:
                    logger.log_solution(best_objective, best_solution)
        
        solution = best_solution
        objective = best_objective
    
    # Write solution
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == '__main__':
    main()
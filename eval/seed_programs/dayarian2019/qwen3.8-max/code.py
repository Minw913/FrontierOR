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
    all_product_ids = []
    for p in products:
        prod_rate[p['product_id']] = p['production_rate_per_subperiod']
        prod_life[p['product_id']] = p['life_span_minutes']
        all_product_ids.append(p['product_id'])

    # Order info - use actual order IDs from instance
    order_map = {}
    all_order_ids = []
    for o in orders:
        order_map[o['order_id']] = o
        all_order_ids.append(o['order_id'])

    # Build mapping from order_id to travel matrix index
    # Travel matrix: index 0 = depot, index i = order at position i-1 in orders list
    order_to_matrix_idx = {}
    for i, o in enumerate(orders):
        order_to_matrix_idx[o['order_id']] = i + 1

    def get_service_time(oid):
        return order_map[oid].get('service_time', service_time_default)

    def get_space(oid):
        return order_map[oid]['space_requirement']

    def get_tw(oid):
        return order_map[oid]['time_window_start'], order_map[oid]['time_window_end']

    def tt(from_oid, to_oid):
        """Travel time between two locations. Use 0 for depot."""
        idx_from = 0 if from_oid == 0 else order_to_matrix_idx[from_oid]
        idx_to = 0 if to_oid == 0 else order_to_matrix_idx[to_oid]
        return travel_time[idx_from][idx_to]

    def compute_trip_times(customers, departure_time):
        """Compute arrival/departure times for a trip. Returns (feasible, return_time)."""
        current_time = departure_time
        prev_loc = 0  # depot
        for cid in customers:
            travel = tt(prev_loc, cid)
            arrival = current_time + travel
            tw_s, tw_e = get_tw(cid)
            if arrival > tw_e:
                return False, None
            effective_arrival = max(arrival, tw_s)
            svc = get_service_time(cid)
            departure = effective_arrival + svc
            current_time = departure
            prev_loc = cid
        # Return to depot
        return_time = current_time + tt(prev_loc, 0)
        return True, return_time

    def find_best_start_period(customers):
        """Find the earliest period where this trip is feasible."""
        for p_idx in range(num_periods):
            dep_time = periods[p_idx]['start_time'] + prep_time
            feasible, ret_time = compute_trip_times(customers, dep_time)
            if feasible and ret_time <= horizon_end:
                if ret_time - periods[p_idx]['start_time'] <= max_route_length_min:
                    return p_idx
        return -1

    def compute_route_travel_time(customers):
        """Compute total travel time for depot -> customers -> depot."""
        if not customers:
            return 0
        total = tt(0, customers[0])
        for i in range(len(customers) - 1):
            total += tt(customers[i], customers[i + 1])
        total += tt(customers[-1], 0)
        return total

    # ============ ROUTING HEURISTIC ============

    def build_initial_routes():
        """Build routes using cheapest insertion with time windows."""
        unassigned = set(all_order_ids)
        routes = []

        # Sort by time window end (earliest deadline first)
        sorted_orders = sorted(all_order_ids, key=lambda x: get_tw(x)[1])

        while unassigned:
            if time.time() - start_time > args.time_limit * 0.4:
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

            # Try to add more customers using cheapest insertion
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
                        # Check feasibility with any period
                        feasible = False
                        for p_idx in range(num_periods):
                            dep_time = periods[p_idx]['start_time'] + prep_time
                            feas, ret = compute_trip_times(new_route, dep_time)
                            if feas and ret <= horizon_end:
                                feasible = True
                                break
                        if feasible:
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
        """Assign start periods and vehicles to routes."""
        route_periods = []
        for route in routes:
            p = find_best_start_period(route)
            route_periods.append(p)

        # Sort routes by their period
        indexed_routes = list(range(len(routes)))
        indexed_routes.sort(key=lambda x: route_periods[x] if route_periods[x] >= 0 else float('inf'))

        vehicles = []
        current_vehicle_trips = []
        current_vehicle = 0
        last_period = -1

        for idx in indexed_routes:
            route = routes[idx]
            p = route_periods[idx]
            if p < 0:
                # Try harder to find a feasible period
                for p_try in range(num_periods):
                    dep_time = periods[p_try]['start_time'] + prep_time
                    feas, ret = compute_trip_times(route, dep_time)
                    if feas and ret <= horizon_end:
                        p = p_try
                        break
                if p < 0:
                    p = 0  # fallback

            if current_vehicle_trips and p > last_period and len(current_vehicle_trips) < max_trips:
                current_vehicle_trips.append((route, p))
                last_period = p
            else:
                if current_vehicle_trips:
                    vehicles.append((current_vehicle, current_vehicle_trips))
                    current_vehicle += 1
                current_vehicle_trips = [(route, p)]
                last_period = p

        if current_vehicle_trips:
            vehicles.append((current_vehicle, current_vehicle_trips))

        return vehicles

    def try_improve_routes(routes, time_budget):
        """Try to improve routes with 2-opt."""
        for r_idx in range(len(routes)):
            if time.time() - start_time > time_budget:
                break
            route = routes[r_idx]
            if len(route) <= 2:
                continue

            improved = True
            iterations = 0
            while improved and iterations < 50:
                improved = False
                iterations += 1
                for i in range(len(route) - 1):
                    for j in range(i + 1, len(route)):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        feasible = False
                        for p_idx in range(num_periods):
                            dep_time = periods[p_idx]['start_time'] + prep_time
                            feas, ret = compute_trip_times(new_route, dep_time)
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

    def solve_production_gurobi(vehicles, time_budget):
        """Solve production scheduling given routes using Gurobi."""
        try:
            import gurobipy as gp_model
            from gurobipy import GRB
        except ImportError:
            return None

        # Determine due time for each order (based on trip departure)
        order_departure_subperiod = {}

        for vid, trips in vehicles:
            for trip_customers, start_period in trips:
                dep_time = periods[start_period]['start_time'] + prep_time
                for oid in trip_customers:
                    due_sub = -1
                    for s in range(num_subperiods):
                        if subperiods[s]['end_time'] <= dep_time:
                            due_sub = s
                    order_departure_subperiod[oid] = due_sub

        # Collect demands and valid subperiods
        demands = {}
        valid_subs = {}

        for o in orders:
            oid = o['order_id']
            for pid_str, qty in o['demands'].items():
                pid = int(pid_str)
                demands[(oid, pid)] = qty
                pw = production_windows.get(str(oid), {}).get(str(pid), {})
                valid_sub_ids = pw.get('valid_subperiod_ids', [])
                due_sub = order_departure_subperiod.get(oid, num_subperiods - 1)
                filtered = [s for s in valid_sub_ids if s <= due_sub]
                valid_subs[(oid, pid)] = filtered

        try:
            model = gp_model.Model("production")
            model.setParam('Seed', 0)
            model.setParam('MIPGap', 1e-4)
            model.setParam('NumericFocus', 0)
            model.setParam('Threads', 1)
            model.setParam('TimeLimit', max(1, time_budget))
            model.setParam('OutputFlag', 0)

            # Decision variables
            q_vars = {}
            for (oid, pid), subs in valid_subs.items():
                for s in subs:
                    q_vars[(oid, pid, s)] = model.addVar(lb=0, ub=demands[(oid, pid)],
                                                          name=f"q_{oid}_{pid}_{s}")

            e_vars = {}
            for pid in all_product_ids:
                for s in range(num_subperiods):
                    e_vars[(pid, s)] = model.addVar(lb=0, ub=num_prod_employees, vtype=GRB.INTEGER,
                                                     name=f"e_{pid}_{s}")

            u_vars = {}
            for pid in all_product_ids:
                for s in range(num_subperiods):
                    u_vars[(pid, s)] = model.addVar(lb=0, ub=num_prod_employees, vtype=GRB.INTEGER,
                                                     name=f"u_{pid}_{s}")

            w_vars = {}
            for t in range(num_periods):
                w_vars[t] = model.addVar(lb=0, ub=num_workstations, vtype=GRB.INTEGER,
                                          name=f"w_{t}")

            shift_vars = {}
            for t in range(num_periods):
                shift_vars[t] = model.addVar(lb=0, ub=num_prod_employees, vtype=GRB.INTEGER,
                                              name=f"shift_{t}")

            model.update()

            # Objective
            obj = gp_model.LinExpr()
            for t in range(num_periods):
                obj += emp_salary * subperiods_per_period * w_vars[t]
            for pid in all_product_ids:
                for s in range(num_subperiods):
                    obj += setup_cost * u_vars[(pid, s)]
            model.setObjective(obj, GRB.MINIMIZE)

            # Demand satisfaction
            for (oid, pid), qty in demands.items():
                subs = valid_subs.get((oid, pid), [])
                if subs:
                    expr = gp_model.LinExpr()
                    for s in subs:
                        if (oid, pid, s) in q_vars:
                            expr += q_vars[(oid, pid, s)]
                    model.addConstr(expr >= qty, name=f"demand_{oid}_{pid}")

            # Production capacity
            for pid in all_product_ids:
                rate = prod_rate[pid]
                for s in range(num_subperiods):
                    total_prod = gp_model.LinExpr()
                    for (oid, p) in demands:
                        if p == pid and s in valid_subs.get((oid, p), []):
                            if (oid, pid, s) in q_vars:
                                total_prod += q_vars[(oid, pid, s)]
                    model.addConstr(total_prod <= rate * e_vars[(pid, s)],
                                    name=f"cap_{pid}_{s}")

            # Employee rise requires setups
            for pid in all_product_ids:
                for s in range(num_subperiods):
                    if s == 0:
                        model.addConstr(e_vars[(pid, s)] <= u_vars[(pid, s)],
                                        name=f"rise_{pid}_{s}")
                    else:
                        model.addConstr(e_vars[(pid, s)] <= e_vars[(pid, s-1)] + u_vars[(pid, s)],
                                        name=f"rise_{pid}_{s}")

            # Total employees constraint
            for s in range(num_subperiods):
                total_emp = gp_model.LinExpr()
                for pid in all_product_ids:
                    total_emp += e_vars[(pid, s)]
                    for s2 in range(max(0, s - setup_length_sub + 1), s + 1):
                        total_emp += u_vars[(pid, s2)]
                model.addConstr(total_emp <= num_prod_employees, name=f"emp_limit_{s}")
                model.addConstr(total_emp <= num_workstations, name=f"ws_limit_{s}")

            # Workstations per period
            for t in range(num_periods):
                sub_start = t * subperiods_per_period
                sub_end = min((t + 1) * subperiods_per_period, num_subperiods)
                for s in range(sub_start, sub_end):
                    total_emp = gp_model.LinExpr()
                    for pid in all_product_ids:
                        total_emp += e_vars[(pid, s)]
                        for s2 in range(max(0, s - setup_length_sub + 1), s + 1):
                            total_emp += u_vars[(pid, s2)]
                    model.addConstr(w_vars[t] >= total_emp, name=f"ws_period_{t}_{s}")

            # Minimum paid time (cumulative)
            for t in range(num_periods):
                cum_ws = gp_model.LinExpr()
                cum_shift = gp_model.LinExpr()
                for t2 in range(t + 1):
                    cum_ws += w_vars[t2] * subperiods_per_period
                    cum_shift += shift_vars[t2] * min_paid_time_sub
                model.addConstr(cum_ws >= cum_shift, name=f"min_paid_{t}")

            model.optimize()

            if model.SolCount > 0:
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
                for pid in all_product_ids:
                    for s in range(num_subperiods):
                        val = int(round(e_vars[(pid, s)].X))
                        if val > 0:
                            emp_by_prod_sub.append({
                                'product': pid,
                                'subperiod': s,
                                'employees': val
                            })

                setups_by_prod_sub = []
                for pid in all_product_ids:
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

        return None

    def greedy_production(vehicles):
        """Greedy production scheduling fallback."""
        order_departure_subperiod = {}
        for vid, trips in vehicles:
            for trip_customers, start_period in trips:
                dep_time = periods[start_period]['start_time'] + prep_time
                for oid in trip_customers:
                    due_sub = -1
                    for s in range(num_subperiods):
                        if subperiods[s]['end_time'] <= dep_time:
                            due_sub = s
                    order_departure_subperiod[oid] = due_sub

        demands = {}
        valid_subs = {}
        for o in orders:
            oid = o['order_id']
            for pid_str, qty in o['demands'].items():
                pid = int(pid_str)
                demands[(oid, pid)] = qty
                pw = production_windows.get(str(oid), {}).get(str(pid), {})
                valid_sub_ids = pw.get('valid_subperiod_ids', [])
                due_sub = order_departure_subperiod.get(oid, num_subperiods - 1)
                filtered = [s for s in valid_sub_ids if s <= due_sub]
                valid_subs[(oid, pid)] = filtered

        prod_schedule = []
        emp_count = defaultdict(int)  # (pid, subperiod) -> count

        for (oid, pid), qty in demands.items():
            subs = valid_subs.get((oid, pid), [])
            if not subs:
                continue
            remaining = qty
            rate = prod_rate[pid]
            # Produce in latest valid subperiods first
            for s in sorted(subs, reverse=True):
                if remaining <= 1e-9:
                    break
                produce = min(remaining, rate)
                prod_schedule.append({
                    'order': oid,
                    'product': pid,
                    'subperiod': s,
                    'quantity': round(produce, 6)
                })
                remaining -= produce
                emp_count[(pid, s)] += 1

        emp_list = [{'product': pid, 'subperiod': s, 'employees': cnt}
                    for (pid, s), cnt in sorted(emp_count.items()) if cnt > 0]

        # Compute setups
        setups_list = []
        for pid in all_product_ids:
            prev_active = False
            for s in range(num_subperiods):
                active = emp_count.get((pid, s), 0) > 0
                if active and not prev_active:
                    setups_list.append({'product': pid, 'subperiod': s, 'setups': emp_count[(pid, s)]})
                prev_active = active

        # Workstations per period
        workstations_by_period = []
        for t in range(num_periods):
            sub_start = t * subperiods_per_period
            sub_end = min((t + 1) * subperiods_per_period, num_subperiods)
            max_emp = 0
            for s in range(sub_start, sub_end):
                total = sum(emp_count.get((pid, s), 0) for pid in all_product_ids)
                max_emp = max(max_emp, total)
            workstations_by_period.append(min(max_emp, num_workstations))

        # Shift starts
        shift_starts_by_period = [0] * num_periods
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
            earliest_start = float('inf')
            latest_return = 0
            total_travel = 0.0

            for trip_customers, start_period in trips:
                dep_time = periods[start_period]['start_time'] + prep_time
                earliest_start = min(earliest_start, periods[start_period]['start_time'])

                trip_tt = compute_route_travel_time(trip_customers)
                total_travel += trip_tt

                # Compute return time
                current_time = dep_time
                prev_loc = 0
                for cid in trip_customers:
                    travel = tt(prev_loc, cid)
                    arrival = current_time + travel
                    tw_s, tw_e = get_tw(cid)
                    effective_arrival = max(arrival, tw_s)
                    svc = get_service_time(cid)
                    current_time = effective_arrival + svc
                    prev_loc = cid
                return_time = current_time + tt(prev_loc, 0)
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

    # Build initial routes
    routes = build_initial_routes()

    # Try to improve routes
    elapsed = time.time() - start_time
    if elapsed < args.time_limit * 0.3:
        routes = try_improve_routes(routes, start_time + args.time_limit * 0.4)

    # Assign to vehicles
    vehicles = assign_periods_to_routes(routes)

    # Solve production
    elapsed = time.time() - start_time
    remaining = max(1, args.time_limit - elapsed - 2)

    prod_result = solve_production_gurobi(vehicles, remaining * 0.7)
    if prod_result is not None:
        prod_schedule, emp_by_prod_sub, setups_by_prod_sub, workstations_by_period, shift_starts_by_period = prod_result
    else:
        prod_schedule, emp_by_prod_sub, setups_by_prod_sub, workstations_by_period, shift_starts_by_period = \
            greedy_production(vehicles)

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

    # Try alternative route constructions if time allows
    best_objective = objective
    best_solution = solution

    elapsed = time.time() - start_time
    if elapsed < args.time_limit * 0.7:
        import random
        for attempt in range(5):
            if time.time() - start_time > args.time_limit * 0.7:
                break

            random.seed(attempt + 42)
            shuffled = list(all_order_ids)
            random.shuffle(shuffled)

            unassigned = set(shuffled)
            new_routes = []
            while unassigned:
                # Pick seed with earliest deadline
                seed = min(unassigned, key=lambda x: get_tw(x)[1])
                route = [seed]
                unassigned.remove(seed)
                route_space = get_space(seed)

                # Greedily add customers
                candidates = sorted(unassigned, key=lambda x: get_tw(x)[1])
                for oid in candidates:
                    if oid not in unassigned:
                        continue
                    if route_space + get_space(oid) > vehicle_capacity:
                        continue
                    new_route = route + [oid]
                    feasible = False
                    for p_idx in range(num_periods):
                        dep_time = periods[p_idx]['start_time'] + prep_time
                        feas, ret = compute_trip_times(new_route, dep_time)
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
            rem = max(1, args.time_limit - elapsed - 2)
            new_prod_result = solve_production_gurobi(new_vehicles, rem * 0.5)
            if new_prod_result is not None:
                new_prod_schedule, new_emp, new_setups, new_ws, new_shifts = new_prod_result
            else:
                new_prod_schedule, new_emp, new_setups, new_ws, new_shifts = greedy_production(new_vehicles)

            new_route_cost = compute_route_cost(new_vehicles)
            new_prod_cost = compute_production_cost(new_ws, new_setups)
            new_objective = new_route_cost + new_prod_cost

            if new_objective < best_objective:
                best_objective = new_objective
                best_solution = build_solution_dict(new_vehicles, new_prod_schedule, new_emp,
                                                    new_setups, new_ws, new_shifts, new_objective)
                if logger:
                    logger.log_solution(best_objective, best_solution)

    # Write final solution
    with open(args.solution_path, 'w') as f:
        json.dump(best_solution, f, indent=2)

if __name__ == '__main__':
    main()
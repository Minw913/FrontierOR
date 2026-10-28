import json
import argparse
import math
import sys
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def load_instance(path):
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    inst = load_instance(args.instance_path)
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # ---- extract global parameters ----
    g = inst['global_parameters']
    period_length = g['period_length_minutes']
    subperiods_per_period = g['subperiods_per_period']
    setup_length_subperiods = g['setup_length_subperiods']
    max_route_length_minutes = g['max_shift_route_length_minutes']
    min_paid_time_minutes = g['min_paid_time_minutes']
    driver_salary_per_minute = g['driver_salary_per_minute']
    employee_salary_per_subperiod = g['employee_salary_per_subperiod']
    setup_cost = g['setup_cost']
    vehicle_operating_cost_per_minute = g['vehicle_operating_cost_per_minute']
    prep_time = g['preparation_time_minutes']
    service_time = g['service_time_minutes']
    max_trips = g['max_trips_per_route']

    num_orders = inst['num_orders']
    num_products = inst['num_products']
    num_vehicles = inst['num_vehicles']
    num_drivers = inst['num_drivers']
    vehicle_capacity = inst['vehicle_capacity']
    num_workstations = inst['num_workstations']
    num_production_employees = inst['num_production_employees']

    products = inst['products']
    orders = inst['orders']
    travel_matrix = inst['travel_time_matrix']
    periods = inst['periods']
    subperiods = inst['subperiods']
    production_windows = inst['production_windows']

    # sort periods and subperiods by time
    periods.sort(key=lambda p: p['start_time'])
    subperiods.sort(key=lambda s: s['start_time'])

    # map order_id to travel matrix index (0=depot, 1..num_orders for orders list order)
    order_matrix_idx = {o['order_id']: i + 1 for i, o in enumerate(orders)}
    # map subperiod_id to sorted index
    sub_id_to_idx = {s['subperiod_id']: i for i, s in enumerate(subperiods)}
    # map period_id to sorted index
    per_id_to_idx = {p['period_id']: i for i, p in enumerate(periods)}

    # ---------- routing heuristic ----------
    # feasible periods for each order (necessary condition: earliest arrival <= tw_end)
    feasible_periods = {}
    for o in orders:
        oid = o['order_id']
        mi = order_matrix_idx[oid]
        fp = []
        for p_idx, p in enumerate(periods):
            earliest_arrival = p['start_time'] + prep_time + travel_matrix[0][mi]
            if earliest_arrival <= o['time_window_end']:
                fp.append(p_idx)
        feasible_periods[oid] = fp

    def simulate_route(customers, start_period_idx):
        if not customers:
            return True, 0.0, 0.0
        start_time = periods[start_period_idx]['start_time'] + prep_time
        current_time = start_time
        prev = 0
        total_travel = 0.0
        total_service = 0.0
        for oid in customers:
            mi = order_matrix_idx[oid]
            travel = travel_matrix[prev][mi]
            total_travel += travel
            current_time += travel
            o = next(x for x in orders if x['order_id'] == oid)
            tw_start = o['time_window_start']
            tw_end = o['time_window_end']
            if current_time < tw_start:
                current_time = tw_start
            if current_time > tw_end:
                return False, 0.0, 0.0
            current_time += o['service_time']
            total_service += o['service_time']
            prev = mi
        total_travel += travel_matrix[prev][0]
        return True, total_travel, total_service

    # trips and vehicle assignment
    trips = []
    vehicle_trips = [[] for _ in range(num_vehicles)]

    sorted_orders = sorted(orders, key=lambda o: o['time_window_end'])

    for o in sorted_orders:
        oid = o['order_id']
        space = o['space_requirement']
        best_insert = None
        best_cost_increase = float('inf')

        # try existing trips
        for trip in trips:
            if trip['total_space'] + space > vehicle_capacity:
                continue
            for pos in range(len(trip['customers']) + 1):
                new_customers = trip['customers'][:pos] + [oid] + trip['customers'][pos:]
                feas, new_travel, new_service = simulate_route(new_customers, trip['start_period'])
                if not feas:
                    continue
                cost_increase = new_travel - trip['total_travel']
                if cost_increase < best_cost_increase:
                    best_cost_increase = cost_increase
                    best_insert = ('existing', trip, pos, new_travel, new_service)

        if best_insert and best_insert[0] == 'existing':
            _, trip, pos, new_travel, new_service = best_insert
            trip['customers'].insert(pos, oid)
            trip['total_space'] += space
            trip['total_travel'] = new_travel
            trip['total_service'] = new_service
            continue

        # create a new trip
        created = False
        for p_idx in feasible_periods[oid]:
            for v in range(num_vehicles):
                if len(vehicle_trips[v]) >= max_trips:
                    continue
                if not vehicle_trips[v] or vehicle_trips[v][-1]['start_period'] < p_idx:
                    feas, total_travel, total_service = simulate_route([oid], p_idx)
                    if not feas:
                        continue
                    trip = {
                        'vehicle': v,
                        'trip_idx': len(vehicle_trips[v]),
                        'start_period': p_idx,
                        'customers': [oid],
                        'total_space': space,
                        'total_travel': total_travel,
                        'total_service': total_service
                    }
                    vehicle_trips[v].append(trip)
                    trips.append(trip)
                    created = True
                    break
            if created:
                break

        if not created:
            # fallback: try later feasible periods on any vehicle with free slots
            for p_idx in feasible_periods[oid]:
                for v in range(num_vehicles):
                    if len(vehicle_trips[v]) >= max_trips:
                        continue
                    if vehicle_trips[v] and vehicle_trips[v][-1]['start_period'] >= p_idx:
                        continue
                    feas, total_travel, total_service = simulate_route([oid], p_idx)
                    if feas:
                        trip = {
                            'vehicle': v,
                            'trip_idx': len(vehicle_trips[v]),
                            'start_period': p_idx,
                            'customers': [oid],
                            'total_space': space,
                            'total_travel': total_travel,
                            'total_service': total_service
                        }
                        vehicle_trips[v].append(trip)
                        trips.append(trip)
                        created = True
                        break
                if created:
                    break

        if not created:
            # last resort: force a trip on vehicle 0, earliest feasible period
            p_idx = feasible_periods[oid][0] if feasible_periods[oid] else 0
            v = 0
            mi = order_matrix_idx[oid]
            trip = {
                'vehicle': v,
                'trip_idx': len(vehicle_trips[v]),
                'start_period': p_idx,
                'customers': [oid],
                'total_space': space,
                'total_travel': travel_matrix[0][mi] + travel_matrix[mi][0],
                'total_service': o['service_time']
            }
            vehicle_trips[v].append(trip)
            trips.append(trip)

    # compute routing cost
    routing_cost = 0.0
    routes_output = []
    for v in range(num_vehicles):
        v_trips = vehicle_trips[v]
        if not v_trips:
            continue
        v_trips.sort(key=lambda t: t['start_period'])
        earliest_start = periods[v_trips[0]['start_period']]['start_time']
        latest_return = 0
        total_travel_all = 0.0
        for trip in v_trips:
            start_time = periods[trip['start_period']]['start_time'] + prep_time
            return_time = start_time + trip['total_travel'] + trip['total_service']
            if return_time > latest_return:
                latest_return = return_time
            total_travel_all += trip['total_travel']
        route_duration = min(max_route_length_minutes, latest_return - earliest_start)
        driver_cost = driver_salary_per_minute * max(route_duration, min_paid_time_minutes)
        vehicle_cost = vehicle_operating_cost_per_minute * total_travel_all
        routing_cost += driver_cost + vehicle_cost

        trips_out = []
        for trip in v_trips:
            trips_out.append({
                'trip': trip['trip_idx'],
                'customers': trip['customers'],
                'start_period': trip['start_period']
            })
        routes_output.append({
            'vehicle': v,
            'trips': trips_out
        })

    # departure times for orders
    order_departure = {}
    for trip in trips:
        dep_time = periods[trip['start_period']]['start_time'] + prep_time
        for oid in trip['customers']:
            order_departure[oid] = dep_time

    # ---------- production MIP ----------
    L_setup = setup_length_subperiods
    L_shift = max(1, math.ceil(min_paid_time_minutes / period_length))

    # prepare valid subperiods with dynamic due time
    def build_valid_subperiods(use_dynamic_due=True):
        vs = {}
        for o in orders:
            oid = o['order_id']
            vs[oid] = {}
            for pid_str, qty in o['demands'].items():
                pid = int(pid_str)
                if qty <= 0:
                    continue
                pw = production_windows.get(str(oid), {}).get(str(pid), {})
                if not pw:
                    instance_due = float('inf')
                    valid_subs = list(range(len(subperiods)))
                else:
                    instance_due = pw.get('due_time', float('inf'))
                    valid_subs = [sub_id_to_idx[sid] for sid in pw.get('valid_subperiod_ids', [])]
                dep_time = order_departure.get(oid, instance_due)
                due_time = min(dep_time, instance_due) if use_dynamic_due else instance_due
                filtered = [s for s in valid_subs if subperiods[s]['end_time'] <= due_time]
                if not filtered:
                    filtered = valid_subs
                vs[oid][pid] = filtered
        return vs

    valid_subperiods = build_valid_subperiods(use_dynamic_due=True)

    def solve_production(vs, time_limit):
        model = gp.Model("production")
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1
        model.Params.TimeLimit = max(1, time_limit)
        model.Params.OutputFlag = 0

        q = {}
        e = {}
        setup = {}
        w = {}
        shift_start = {}

        for o in orders:
            oid = o['order_id']
            q[oid] = {}
            for pid, subs in vs[oid].items():
                q[oid][pid] = {}
                for s in subs:
                    q[oid][pid][s] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"q_{oid}_{pid}_{s}")

        for p in products:
            pid = p['product_id']
            e[pid] = {}
            setup[pid] = {}
            for s in range(len(subperiods)):
                e[pid][s] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"e_{pid}_{s}")
                setup[pid][s] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"setup_{pid}_{s}")

        for t in range(len(periods)):
            w[t] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"w_{t}")
            shift_start[t] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"shift_start_{t}")

        model.update()

        # demand
        for o in orders:
            oid = o['order_id']
            for pid, subs in vs[oid].items():
                qty = o['demands'][str(pid)]
                model.addConstr(gp.quicksum(q[oid][pid][s] for s in subs) >= qty, name=f"demand_{oid}_{pid}")

        # capacity
        for p in products:
            pid = p['product_id']
            rate = p['production_rate_per_subperiod']
            for s in range(len(subperiods)):
                lhs = gp.quicksum(q[oid][pid][s] for oid in q if pid in q[oid] and s in q[oid][pid])
                model.addConstr(lhs <= rate * e[pid][s], name=f"cap_{pid}_{s}")

        # setup increase
        for p in products:
            pid = p['product_id']
            for s in range(len(subperiods)):
                prev_e = e[pid][s-1] if s > 0 else 0
                model.addConstr(setup[pid][s] >= e[pid][s] - prev_e, name=f"setup_inc_{pid}_{s}")

        # active employees and workstations
        for s in range(len(subperiods)):
            setup_active = gp.quicksum(
                setup[p['product_id']][sp]
                for p in products
                for sp in range(max(0, s - L_setup + 1), s + 1)
            )
            producing = gp.quicksum(e[p['product_id']][s] for p in products)
            active = producing + setup_active
            model.addConstr(active <= num_workstations, name=f"active_ws_{s}")
            model.addConstr(active <= num_production_employees, name=f"active_emp_{s}")
            t = s // subperiods_per_period
            if t < len(periods):
                model.addConstr(w[t] >= active, name=f"paid_ws_{s}")

        # shift coverage
        for t in range(len(periods)):
            coverage = gp.quicksum(shift_start[tp] for tp in range(max(0, t - L_shift + 1), t + 1))
            model.addConstr(w[t] >= coverage, name=f"shift_cov_{t}")

        # objective
        prod_cost = gp.quicksum(w[t] * employee_salary_per_subperiod * subperiods_per_period for t in range(len(periods)))
        setup_total = gp.quicksum(setup[p['product_id']][s] for p in products for s in range(len(subperiods))) * setup_cost
        model.setObjective(prod_cost + setup_total, GRB.MINIMIZE)

        model.optimize()

        return model, q, e, setup, w, shift_start

    production_cost = 0.0
    production_schedule = []
    employees_by_product_subperiod = []
    setups_by_product_subperiod = []
    workstations_by_period = [0] * len(periods)
    shift_starts_by_period = [0] * len(periods)

    try:
        model, q, e, setup, w, shift_start = solve_production(valid_subperiods, args.time_limit - 5)
        if model.SolCount == 0:
            # fallback: use instance due time only
            valid_subperiods = build_valid_subperiods(use_dynamic_due=False)
            model, q, e, setup, w, shift_start = solve_production(valid_subperiods, args.time_limit - 5)

        if model.SolCount > 0:
            for o in orders:
                oid = o['order_id']
                for pid, subs in valid_subperiods[oid].items():
                    for s in subs:
                        val = q[oid][pid][s].X
                        if val > 1e-6:
                            production_schedule.append({
                                'order': oid,
                                'product': pid,
                                'subperiod': subperiods[s]['subperiod_id'],
                                'quantity': val
                            })
            for p in products:
                pid = p['product_id']
                for s in range(len(subperiods)):
                    val = e[pid][s].X
                    if val > 0.5:
                        employees_by_product_subperiod.append({
                            'product': pid,
                            'subperiod': subperiods[s]['subperiod_id'],
                            'employees': int(round(val))
                        })
                    val = setup[pid][s].X
                    if val > 0.5:
                        setups_by_product_subperiod.append({
                            'product': pid,
                            'subperiod': subperiods[s]['subperiod_id'],
                            'setups': int(round(val))
                        })
            workstations_by_period = [int(round(w[t].X)) for t in range(len(periods))]
            shift_starts_by_period = [int(round(shift_start[t].X)) for t in range(len(periods))]
            production_cost = sum(workstations_by_period) * employee_salary_per_subperiod * subperiods_per_period + \
                             sum(s['setups'] for s in setups_by_product_subperiod) * setup_cost
    except Exception as ex:
        print(f"Production MIP failed: {ex}", file=sys.stderr)

    total_cost = routing_cost + production_cost

    solution = {
        'objective_value': total_cost,
        'routes': routes_output,
        'production_schedule': production_schedule,
        'production_staffing': {
            'employees_by_product_subperiod': employees_by_product_subperiod,
            'setups_by_product_subperiod': setups_by_product_subperiod,
            'workstations_by_period': workstations_by_period,
            'shift_starts_by_period': shift_starts_by_period
        }
    }

    if logger:
        logger.log_solution(total_cost, solution)

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == '__main__':
    main()
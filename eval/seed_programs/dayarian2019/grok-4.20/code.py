import json
import argparse
import time
import sys
from typing import Dict, Any, List
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    # ====================== Data extraction ======================
    n_orders = instance['num_orders']
    n_products = instance['num_products']
    n_periods = instance['global_parameters']['num_periods']
    n_subperiods = instance['global_parameters']['num_subperiods']
    customers = list(range(1, n_orders + 1))
    products = [p['product_id'] for p in instance['products']]
    prod_rate = {p['product_id']: p['production_rate_per_subperiod'] for p in instance['products']}

    g = instance['global_parameters']
    period_len = g['period_length_minutes']
    sub_len = g['subperiod_length_minutes']
    subs_per_period = g['subperiods_per_period']
    max_trips = g['max_trips_per_route']
    driver_rate = g['driver_salary_per_minute']
    employee_rate = g['employee_salary_per_subperiod']
    setup_cost_val = g['setup_cost']
    vehicle_rate = g['vehicle_operating_cost_per_minute']
    prep_time = g['preparation_time_minutes']
    min_paid_min = g['min_paid_time_minutes']
    max_route_min = g['max_shift_route_length_minutes']

    num_vehicles = instance['num_vehicles']
    vehicle_capacity = instance['vehicle_capacity']
    num_workstations = instance['num_workstations']
    num_prod_employees = instance['num_production_employees']

    travel = instance['travel_time_matrix']
    orders_data = {o['order_id']: o for o in instance['orders']}
    prod_windows = instance['production_windows']

    demand = {}
    space_req = {}
    tw_start = {}
    tw_end = {}
    valid_subs = {}
    for oid_str, odata in orders_data.items():
        oid = int(oid_str)
        space_req[oid] = odata['space_requirement']
        tw_start[oid] = odata['time_window_start']
        tw_end[oid] = odata['time_window_end']
        demand[oid] = {int(k): float(v) for k, v in odata['demands'].items()}
        for pid_str, win in prod_windows.get(str(oid), {}).items():
            pid = int(pid_str)
            valid_subs[(oid, pid)] = win.get('valid_subperiod_ids', [])

    # ====================== Model ======================
    model = gp.Model("Catering_Integrated")
    model.setParam(GRB.Param.Seed, 0)
    model.setParam(GRB.Param.MIPGap, 1e-4)
    model.setParam(GRB.Param.NumericFocus, 0)
    model.setParam(GRB.Param.Threads, 1)
    model.setParam(GRB.Param.TimeLimit, max(10, args.time_limit - 20))

    vehicles = list(range(num_vehicles))
    trip_idx = list(range(max_trips))
    locations = [0] + customers

    # ------------------ Delivery variables ------------------
    x = model.addVars(vehicles, trip_idx, locations, locations,
                      vtype=GRB.BINARY, name="x")
    y = model.addVars(customers, vehicles, trip_idx, vtype=GRB.BINARY, name="y")
    start_period_var = model.addVars(vehicles, trip_idx, vtype=GRB.INTEGER,
                                     lb=0, ub=n_periods-1, name="start_period")
    route_used = model.addVars(vehicles, vtype=GRB.BINARY, name="route_used")
    trip_used = model.addVars(vehicles, trip_idx, vtype=GRB.BINARY, name="trip_used")

    # ------------------ Production variables ------------------
    q = {}
    for o in customers:
        for p in products:
            for s in valid_subs.get((o, p), []):
                q[o, p, s] = model.addVar(lb=0.0, ub=demand.get(o, {}).get(p, 0.0) * 2,
                                          vtype=GRB.CONTINUOUS, name=f"q_{o}_{p}_{s}")

    e = model.addVars(products, range(n_subperiods), vtype=GRB.INTEGER, lb=0,
                      ub=num_prod_employees, name="e")
    setup_var = model.addVars(products, range(n_subperiods), vtype=GRB.INTEGER, lb=0,
                              ub=num_workstations, name="setup")
    w = model.addVars(range(n_periods), vtype=GRB.INTEGER, lb=0, ub=num_workstations, name="w")
    shift_start_var = model.addVars(range(n_periods), vtype=GRB.INTEGER, lb=0, name="shift_start")

    route_paid = model.addVars(vehicles, vtype=GRB.CONTINUOUS, lb=0, name="route_paid")
    total_travel = model.addVars(vehicles, vtype=GRB.CONTINUOUS, lb=0, name="total_travel")

    # ====================== CONSTRAINTS ======================

    # Each order served exactly once
    for o in customers:
        model.addConstr(gp.quicksum(y[o, v, k] for v in vehicles for k in trip_idx) == 1,
                        name=f"serve_once_{o}")

    # Flow conservation
    for v in vehicles:
        for k in trip_idx:
            for i in locations:
                out_sum = gp.quicksum(x[v, k, i, j] for j in locations if j != i)
                in_sum = gp.quicksum(x[v, k, j, i] for j in locations if j != i)

                if i == 0:
                    model.addConstr(out_sum == trip_used[v, k], name=f"out_depot_{v}_{k}")
                    model.addConstr(in_sum == trip_used[v, k], name=f"in_depot_{v}_{k}")
                else:
                    model.addConstr(out_sum == y[i, v, k], name=f"out_cust_{i}_{v}_{k}")
                    model.addConstr(in_sum == y[i, v, k], name=f"in_cust_{i}_{v}_{k}")

    # Trip ordering and route usage
    for v in vehicles:
        model.addConstr(route_used[v] <= 1)
        for k in range(1, max_trips):
            model.addConstr(trip_used[v, k] <= trip_used[v, k - 1])
        for k in trip_idx:
            model.addConstr(trip_used[v, k] <= route_used[v])

    # Capacity, time windows, and start period linking
    M = 10000.0
    for v in vehicles:
        for k in trip_idx:
            start_min_expr = start_period_var[v, k] * period_len + prep_time

            model.addConstr(
                gp.quicksum(space_req.get(o, 0) * y[o, v, k] for o in customers)
                <= vehicle_capacity * trip_used[v, k],
                name=f"cap_{v}_{k}")

            for o in customers:
                model.addConstr(
                    start_min_expr <= tw_start.get(o, 0) + M * (1 - y[o, v, k]),
                    name=f"tw_start_{o}_{v}_{k}")
                model.addConstr(
                    start_min_expr + 1440 <= tw_end.get(o, 1440) + M * (1 - y[o, v, k]),
                    name=f"tw_end_{o}_{v}_{k}")

    # Production – demand satisfaction & linking to departure
    for o in customers:
        for p in products:
            d = demand.get(o, {}).get(p, 0.0)
            if d <= 0:
                continue
            valid_s = valid_subs.get((o, p), [])
            if not valid_s:
                continue

            model.addConstr(
                gp.quicksum(q.get((o, p, s), 0) for s in valid_s) >= d,
                name=f"demand_{o}_{p}")

            due_time = float(prod_windows.get(str(o), {}).get(str(p), {}).get('due_time', 0))
            due_sub = int(due_time // sub_len)

            for v in vehicles:
                for k in trip_idx:
                    model.addConstr(
                        gp.quicksum(q.get((o, p, s), 0) for s in valid_s if s <= due_sub)
                        >= d * y[o, v, k],
                        name=f"link_{o}_{p}_{v}_{k}"
                    )

    # Production capacity
    for s in range(n_subperiods):
        model.addConstr(gp.quicksum(e[p, s] for p in products) <= num_prod_employees,
                        name=f"emp_limit_{s}")
        model.addConstr(
            gp.quicksum(e[p, s] + setup_var[p, s] for p in products) <= num_workstations,
            name=f"ws_limit_{s}")

    # Setup constraint
    for p in products:
        for s in range(1, n_subperiods):
            model.addConstr(e[p, s] - e[p, s - 1] <= setup_var[p, s] * num_workstations,
                            name=f"setup_link_{p}_{s}")

    # Aggregate staffing per period
    for t in range(n_periods):
        s_start = t * subs_per_period
        s_end = min(s_start + subs_per_period, n_subperiods)
        total_staff = gp.quicksum(e[p, s] + setup_var[p, s]
                                  for p in products for s in range(s_start, s_end))
        model.addConstr(w[t] >= total_staff / subs_per_period, name=f"workstation_{t}")
        prev = w[t - 1] if t > 0 else 0
        model.addConstr(shift_start_var[t] >= w[t] - prev, name=f"shift_start_{t}")

    # Route cost modeling
    for v in vehicles:
        travel_expr = gp.quicksum(
            travel[i][j] * x[v, k, i, j]
            for k in trip_idx
            for i in locations
            for j in locations if i != j)
        model.addConstr(total_travel[v] == travel_expr, name=f"travel_{v}")

        model.addConstr(
            route_paid[v] >= total_travel[v] + prep_time * gp.quicksum(trip_used[v, k] for k in trip_idx),
            name=f"paid_lower_travel_{v}")
        model.addConstr(route_paid[v] >= min_paid_min, name=f"paid_lower_min_{v}")
        model.addConstr(route_paid[v] <= max_route_min + M * (1 - route_used[v]), name=f"paid_upper_{v}")

    # ====================== Objective ======================
    driver_cost = driver_rate * gp.quicksum(route_paid[v] for v in vehicles)
    vehicle_cost = vehicle_rate * gp.quicksum(total_travel[v] for v in vehicles)
    labor_cost = employee_rate * gp.quicksum(w[t] for t in range(n_periods))
    setup_cost_term = setup_cost_val * gp.quicksum(setup_var[p, s] for p in products for s in range(n_subperiods))

    model.setObjective(driver_cost + vehicle_cost + labor_cost + setup_cost_term, GRB.MINIMIZE)

    # ====================== Callback ======================
    best_obj = float('inf')
    best_solution = None

    def callback(model, where):
        nonlocal best_obj, best_solution
        if where != GRB.Callback.MIPSOL:
            return
        obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
        if obj >= best_obj - 1e-6:
            return
        best_obj = obj

        try:
            routes = []
            for v in vehicles:
                trips = []
                for k in trip_idx:
                    if model.cbGetSolution(trip_used[v, k]) > 0.5:
                        cust_list = [o for o in customers if model.cbGetSolution(y[o, v, k]) > 0.5]
                        if cust_list:
                            sp = int(round(model.cbGetSolution(start_period_var[v, k])))
                            trips.append({
                                "trip": int(k),
                                "customers": sorted([int(c) for c in cust_list]),
                                "start_period": sp
                            })
                if trips:
                    routes.append({
                        "vehicle": int(v),
                        "trips": sorted(trips, key=lambda t: t["trip"])
                    })

            prod_sched = []
            for (o, p, s), var in q.items():
                val = model.cbGetSolution(var)
                if val > 1e-6:
                    prod_sched.append({
                        "order": int(o),
                        "product": int(p),
                        "subperiod": int(s),
                        "quantity": float(round(val, 6))
                    })

            emp_list = []
            setup_list = []
            for p in products:
                for s in range(n_subperiods):
                    e_val = int(round(model.cbGetSolution(e[p, s])))
                    s_val = int(round(model.cbGetSolution(setup_var[p, s])))
                    if e_val > 0:
                        emp_list.append({"product": int(p), "subperiod": int(s), "employees": e_val})
                    if s_val > 0:
                        setup_list.append({"product": int(p), "subperiod": int(s), "setups": s_val})

            ws = [int(round(model.cbGetSolution(w[t]))) for t in range(n_periods)]
            ss = [int(round(model.cbGetSolution(shift_start_var[t]))) for t in range(n_periods)]

            sol_dict = {
                "objective_value": float(obj),
                "routes": routes,
                "production_schedule": prod_sched,
                "production_staffing": {
                    "employees_by_product_subperiod": emp_list,
                    "setups_by_product_subperiod": setup_list,
                    "workstations_by_period": ws,
                    "shift_starts_by_period": ss
                }
            }
            best_solution = sol_dict
            if logger:
                logger.log_solution(obj, sol_dict)
        except Exception:
            if logger:
                logger.log(obj)

    model.optimize(callback)

    # ====================== Final solution ======================
    final_obj = model.objVal if model.SolCount > 0 else 1e9
    if best_solution is None or final_obj < best_solution.get("objective_value", 1e9):
        # Extract from final solution if callback didn't fire
        try:
            routes = []
            for v in vehicles:
                trips = []
                for k in trip_idx:
                    if trip_used[v, k].X > 0.5:
                        cust_list = [o for o in customers if y[o, v, k].X > 0.5]
                        if cust_list:
                            sp = int(round(start_period_var[v, k].X))
                            trips.append({
                                "trip": int(k),
                                "customers": sorted([int(c) for c in cust_list]),
                                "start_period": sp
                            })
                if trips:
                    routes.append({"vehicle": int(v), "trips": sorted(trips, key=lambda t: t["trip"])})

            prod_sched = []
            for (o, p, s), var in q.items():
                val = var.X
                if val > 1e-6:
                    prod_sched.append({
                        "order": int(o),
                        "product": int(p),
                        "subperiod": int(s),
                        "quantity": float(round(val, 6))
                    })

            emp_list = []
            setup_list = []
            for p in products:
                for s in range(n_subperiods):
                    e_val = int(round(e[p, s].X))
                    s_val = int(round(setup_var[p, s].X))
                    if e_val > 0:
                        emp_list.append({"product": int(p), "subperiod": int(s), "employees": e_val})
                    if s_val > 0:
                        setup_list.append({"product": int(p), "subperiod": int(s), "setups": s_val})

            ws = [int(round(w[t].X)) for t in range(n_periods)]
            ss = [int(round(shift_start_var[t].X)) for t in range(n_periods)]

            best_solution = {
                "objective_value": float(final_obj),
                "routes": routes,
                "production_schedule": prod_sched,
                "production_staffing": {
                    "employees_by_product_subperiod": emp_list,
                    "setups_by_product_subperiod": setup_list,
                    "workstations_by_period": ws,
                    "shift_starts_by_period": ss
                }
            }
        except Exception:
            best_solution = {
                "objective_value": float(final_obj),
                "routes": [],
                "production_schedule": [],
                "production_staffing": {
                    "employees_by_product_subperiod": [],
                    "setups_by_product_subperiod": [],
                    "workstations_by_period": [0] * n_periods,
                    "shift_starts_by_period": [0] * n_periods
                }
            }

    with open(args.solution_path, 'w') as f:
        json.dump(best_solution, f, indent=2)


if __name__ == "__main__":
    main()
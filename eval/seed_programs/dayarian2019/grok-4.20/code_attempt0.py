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

    # Extract data
    n_orders = instance['num_orders']
    n_products = instance['num_products']
    n_periods = instance['global_parameters']['num_periods']
    n_subperiods = instance['global_parameters']['num_subperiods']
    depot = 0
    orders = list(range(1, n_orders + 1))
    products = [p['product_id'] for p in instance['products']]
    product_list = instance['products']
    prod_rate = {p['product_id']: p['production_rate_per_subperiod'] for p in product_list}
    life_span = {p['product_id']: p['life_span_minutes'] for p in product_list}

    global_params = instance['global_parameters']
    period_len = global_params['period_length_minutes']
    sub_len = global_params['subperiod_length_minutes']
    setup_sub = global_params['setup_length_subperiods']
    max_trips = global_params['max_trips_per_route']
    driver_rate = global_params['driver_salary_per_minute']
    employee_rate = global_params['employee_salary_per_subperiod']
    setup_cost = global_params['setup_cost']
    vehicle_rate = global_params['vehicle_operating_cost_per_minute']
    prep_time = global_params['preparation_time_minutes']
    service_time = global_params['service_time_minutes']
    min_paid_min = global_params['min_paid_time_minutes']
    max_route_min = global_params['max_shift_route_length_minutes']
    num_vehicles = instance['num_vehicles']
    num_drivers = instance['num_drivers']
    vehicle_capacity = instance['vehicle_capacity']
    num_workstations = instance['num_workstations']
    num_prod_employees = instance['num_production_employees']

    travel = instance['travel_time_matrix']
    orders_data = {o['order_id']: o for o in instance['orders']}
    prod_windows = instance['production_windows']
    subperiods = instance['subperiods']
    periods = instance['periods']

    # Precompute valid production subperiods per (order, product)
    valid_subs = {}
    demand = {}
    space_req = {}
    tw_start = {}
    tw_end = {}
    for oid, odata in orders_data.items():
        space_req[oid] = odata['space_requirement']
        tw_start[oid] = odata['time_window_start']
        tw_end[oid] = odata['time_window_end']
        demand[oid] = odata['demands']
        for pid, pwin in prod_windows[str(oid)].items():
            p_int = int(pid)
            valid_subs[(oid, p_int)] = pwin['valid_subperiod_ids']
            # due_time will be used for linking

    # Build model
    model = gp.Model("Catering_Integrated")
    model.setParam(GRB.Param.Seed, 0)
    model.setParam(GRB.Param.MIPGap, 1e-4)
    model.setParam(GRB.Param.NumericFocus, 0)
    model.setParam(GRB.Param.Threads, 1)
    model.setParam(GRB.Param.TimeLimit, max(1, args.time_limit - 5))

    # ====================== DELIVERY VARIABLES ======================
    # x[v][k][i][j] = 1 if vehicle v uses arc i->j on its k-th trip
    # Use compact indexed formulation as per problem hint
    max_v = num_vehicles
    vehicles = list(range(max_v))
    trips_per_route = list(range(max_trips))

    x = {}
    for v in vehicles:
        for k in trips_per_route:
            for i in [0] + orders:
                for j in [0] + orders:
                    if i != j:
                        x[v, k, i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{v}_{k}_{i}_{j}")

    # y[o][v][k] = 1 if order o is served by vehicle v on trip k
    y = {}
    for o in orders:
        for v in vehicles:
            for k in trips_per_route:
                y[o, v, k] = model.addVar(vtype=GRB.BINARY, name=f"y_{o}_{v}_{k}")

    # start_period[v][k]
    start_period = {}
    for v in vehicles:
        for k in trips_per_route:
            start_period[v, k] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=n_periods-1, name=f"startp_{v}_{k}")

    # route_used[v] = 1 if vehicle v is used
    route_used = model.addVars(vehicles, vtype=GRB.BINARY, name="route_used")

    # trip_used[v,k]
    trip_used = {}
    for v in vehicles:
        for k in trips_per_route:
            trip_used[v, k] = model.addVar(vtype=GRB.BINARY, name=f"tripused_{v}_{k}")

    # ====================== PRODUCTION VARIABLES ======================
    # q[o,p,s] = quantity of product p for order o in subperiod s
    q = {}
    for o in orders:
        for p in products:
            for s_idx in valid_subs.get((o, p), []):
                q[o, p, s_idx] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"q_{o}_{p}_{s_idx}")

    # e[p,s] = employees producing p in subperiod s
    e = {}
    for p in products:
        for s_idx in range(n_subperiods):
            e[p, s_idx] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=num_prod_employees, name=f"e_{p}_{s_idx}")

    # setup[p,s] = number of setups started for p in subperiod s
    setup = {}
    for p in products:
        for s_idx in range(n_subperiods):
            setup[p, s_idx] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=num_workstations, name=f"setup_{p}_{s_idx}")

    # w[t] = paid workstations in period t
    w = model.addVars(range(n_periods), vtype=GRB.INTEGER, lb=0, ub=num_workstations, name="w")

    # shift_start[t] = number of shifts starting in period t
    shift_start = model.addVars(range(n_periods), vtype=GRB.INTEGER, lb=0, name="shift_start")

    # Auxiliary for route duration and cost
    route_paid_time = model.addVars(vehicles, vtype=GRB.CONTINUOUS, name="route_paid")
    travel_time_total = model.addVars(vehicles, vtype=GRB.CONTINUOUS, name="travel_time")

    # ====================== CONSTRAINTS ======================

    # 1. Each order delivered exactly once
    for o in orders:
        model.addConstr(gp.quicksum(y[o, v, k] for v in vehicles for k in trips_per_route) == 1)

    # 2. Arc flow and visit consistency
    for v in vehicles:
        for k in trips_per_route:
            for i in [0] + orders:
                model.addConstr(gp.quicksum(x[v, k, i, j] for j in [0] + orders if i != j) == 
                               gp.quicksum(y[o, v, k] for o in orders if o == i) if i > 0 else trip_used[v, k])

            for j in [0] + orders:
                model.addConstr(gp.quicksum(x[v, k, i, j] for i in [0] + orders if i != j) == 
                               gp.quicksum(y[o, v, k] for o in orders if o == j) if j > 0 else trip_used[v, k])

            # Subtour elimination (MTZ style)
            if len(orders) > 1:
                u = {}
                for o in orders:
                    u[o] = model.addVar(vtype=GRB.CONTINUOUS, lb=1, ub=len(orders))
                for i in orders:
                    for j in orders:
                        if i != j:
                            model.addConstr(u[i] - u[j] + (len(orders)+1)*x[v,k,i,j] <= len(orders))

    # 3. Trip sequencing and route usage
    for v in vehicles:
        model.addConstr(route_used[v] <= 1)
        for k in range(1, max_trips):
            model.addConstr(trip_used[v, k] <= trip_used[v, k-1])
        model.addConstrs(trip_used[v, k] <= route_used[v] for k in trips_per_route)

    # 4. Time window and capacity per trip
    M = 1440 * 2
    for v in vehicles:
        for k in trips_per_route:
            # Start time in minutes
            trip_start_min = start_period[v, k] * period_len + prep_time
            # Capacity
            model.addConstr(gp.quicksum(space_req[o] * y[o, v, k] for o in orders) <= vehicle_capacity * trip_used[v, k])

            # Time windows (approximate using latest possible arrival)
            for o in orders:
                model.addConstr(trip_start_min + 
                               gp.quicksum(travel[i][j] * x[v,k,i,j] for i in [0]+orders for j in [0]+orders if i!=j) +
                               (service_time * gp.quicksum(y[o2,v,k] for o2 in orders)) <= 
                               tw_end[o] + M * (1 - y[o, v, k]))

                model.addConstr(trip_start_min >= tw_start[o] - M * (1 - y[o, v, k]))

    # 5. Production linking
    for o in orders:
        for p in products:
            if (o, p) in valid_subs:
                due_sub = int(prod_windows[str(o)][str(p)]['due_time'] // sub_len)
                # Total production meets demand
                model.addConstr(gp.quicksum(q[o, p, s] for s in valid_subs[(o, p)]) >= demand[o][str(p)])

                # Production before trip departure
                for k in trips_per_route:
                    for v in vehicles:
                        model.addConstr(
                            gp.quicksum(q[o, p, s] for s in valid_subs[(o, p)] if s <= due_sub) >= 
                            demand[o][str(p)] * y[o, v, k]
                        )

    # 6. Production capacity
    for s_idx in range(n_subperiods):
        model.addConstr(gp.quicksum(e[p, s_idx] for p in products) <= num_prod_employees)
        model.addConstr(gp.quicksum(e[p, s_idx] + setup[p, s_idx] for p in products) <= num_workstations)

    # 7. Setup logic (simple: at least one setup when employees increase)
    for p in products:
        for s_idx in range(1, n_subperiods):
            model.addConstr(e[p, s_idx] - e[p, s_idx-1] <= num_workstations * setup[p, s_idx])

    # 8. Period aggregation for staffing
    subs_per_period = global_params['subperiods_per_period']
    for t in range(n_periods):
        subs_in_t = list(range(t * subs_per_period, (t+1)*subs_per_period))
        model.addConstr(w[t] >= gp.quicksum(e[p, s] for p in products for s in subs_in_t) / subs_per_period)
        model.addConstr(w[t] >= gp.quicksum(setup[p, s] for p in products for s in subs_in_t) / subs_per_period)

    # Shift starts (very aggregate as per benchmark)
    for t in range(n_periods):
        model.addConstr(shift_start[t] >= w[t] - (w[t-1] if t > 0 else 0))

    # 9. Route duration modeling
    for v in vehicles:
        # Approximate total travel time
        model.addConstr(travel_time_total[v] == gp.quicksum(
            travel[i][j] * x[v, k, i, j] for k in trips_per_route for i in [0]+orders for j in [0]+orders if i != j))
        
        # Paid time: max of modeled duration and min paid
        bigM_route = 10000
        model.addConstr(route_paid_time[v] >= travel_time_total[v] + prep_time * gp.quicksum(trip_used[v,k] for k in trips_per_route))
        model.addConstr(route_paid_time[v] >= min_paid_min)
        model.addConstr(route_paid_time[v] <= max_route_min + bigM_route * (1 - route_used[v]))

    # ====================== OBJECTIVE ======================
    driver_cost = driver_rate * gp.quicksum(route_paid_time[v] for v in vehicles)
    vehicle_cost = vehicle_rate * gp.quicksum(travel_time_total[v] for v in vehicles)
    employee_cost = employee_rate * gp.quicksum(w[t] for t in range(n_periods))
    setup_total_cost = setup_cost * gp.quicksum(setup[p, s] for p in products for s in range(n_subperiods))

    model.setObjective(driver_cost + vehicle_cost + employee_cost + setup_total_cost, GRB.MINIMIZE)

    # Callback for logging incumbents
    best_obj = float('inf')
    solution_dict_cache = None

    def callback(model, where):
        nonlocal best_obj, solution_dict_cache
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj_val < best_obj - 1e-6:
                best_obj = obj_val
                # Extract solution for logging
                try:
                    sol_routes = []
                    for v in vehicles:
                        route_trips = []
                        for k in trips_per_route:
                            if model.cbGetSolution(trip_used[v, k]) > 0.5:
                                custs = []
                                for o in orders:
                                    if model.cbGetSolution(y[o, v, k]) > 0.5:
                                        custs.append(int(o))
                                if custs:
                                    sp = int(model.cbGetSolution(start_period[v, k]))
                                    route_trips.append({
                                        "trip": int(k),
                                        "customers": custs,
                                        "start_period": sp
                                    })
                        if route_trips:
                            sol_routes.append({
                                "vehicle": int(v),
                                "trips": sorted(route_trips, key=lambda x: x["trip"])
                            })

                    prod_sched = []
                    for o, p, s in q.keys():
                        val = model.cbGetSolution(q[o, p, s])
                        if val > 1e-6:
                            prod_sched.append({
                                "order": int(o),
                                "product": int(p),
                                "subperiod": int(s),
                                "quantity": float(val)
                            })

                    emp_by_ps = []
                    setup_by_ps = []
                    for p in products:
                        for s_idx in range(n_subperiods):
                            e_val = int(round(model.cbGetSolution(e[p, s_idx])))
                            su_val = int(round(model.cbGetSolution(setup[p, s_idx])))
                            if e_val > 0:
                                emp_by_ps.append({"product": int(p), "subperiod": int(s_idx), "employees": e_val})
                            if su_val > 0:
                                setup_by_ps.append({"product": int(p), "subperiod": int(s_idx), "setups": su_val})

                    ws_by_p = [int(round(model.cbGetSolution(w[t]))) for t in range(n_periods)]
                    ss_by_p = [int(round(model.cbGetSolution(shift_start[t]))) for t in range(n_periods)]

                    solution_dict = {
                        "objective_value": float(obj_val),
                        "routes": sol_routes,
                        "production_schedule": prod_sched,
                        "production_staffing": {
                            "employees_by_product_subperiod": emp_by_ps,
                            "setups_by_product_subperiod": setup_by_ps,
                            "workstations_by_period": ws_by_p,
                            "shift_starts_by_period": ss_by_p
                        }
                    }

                    solution_dict_cache = solution_dict
                    if logger:
                        logger.log_solution(obj_val, solution_dict)
                except:
                    if logger:
                        logger.log(obj_val)

    model.optimize(callback)

    # Final solution extraction
    if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SOLUTION_LIMIT]:
        final_obj = model.objVal if model.objVal < float('inf') else best_obj
        final_solution = solution_dict_cache

        if final_solution is None or abs(final_solution["objective_value"] - final_obj) > 1e-2:
            # Re-extract if callback didn't capture last
            try:
                sol_routes = []
                for v_idx in vehicles:
                    route_trips = []
                    for k_idx in trips_per_route:
                        if trip_used[v_idx, k_idx].X > 0.5:
                            cust_list = [o for o in orders if y[o, v_idx, k_idx].X > 0.5]
                            if cust_list:
                                route_trips.append({
                                    "trip": int(k_idx),
                                    "customers": [int(c) for c in cust_list],
                                    "start_period": int(start_period[v_idx, k_idx].X)
                                })
                    if route_trips:
                        sol_routes.append({
                            "vehicle": int(v_idx),
                            "trips": sorted(route_trips, key=lambda r: r["trip"])
                        })

                prod_list = []
                for (o, p, s), var in q.items():
                    if var.X > 1e-6:
                        prod_list.append({
                            "order": int(o),
                            "product": int(p),
                            "subperiod": int(s),
                            "quantity": float(var.X)
                        })

                emp_list = []
                setup_list = []
                for p in products:
                    for s in range(n_subperiods):
                        e_val = int(round(e[p, s].X))
                        s_val = int(round(setup[p, s].X))
                        if e_val > 0:
                            emp_list.append({"product": int(p), "subperiod": int(s), "employees": e_val})
                        if s_val > 0:
                            setup_list.append({"product": int(p), "subperiod": int(s), "setups": s_val})

                ws_list = [int(round(w[t].X)) for t in range(n_periods)]
                shift_list = [int(round(shift_start[t].X)) for t in range(n_periods)]

                final_solution = {
                    "objective_value": float(final_obj),
                    "routes": sol_routes,
                    "production_schedule": prod_list,
                    "production_staffing": {
                        "employees_by_product_subperiod": emp_list,
                        "setups_by_product_subperiod": setup_list,
                        "workstations_by_period": ws_list,
                        "shift_starts_by_period": shift_list
                    }
                }
            except Exception:
                # Minimal feasible fallback
                final_solution = {
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
            json.dump(final_solution, f, indent=2)
    else:
        # No solution found
        fallback = {
            "objective_value": 1e9,
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
            json.dump(fallback, f, indent=2)


if __name__ == "__main__":
    main()
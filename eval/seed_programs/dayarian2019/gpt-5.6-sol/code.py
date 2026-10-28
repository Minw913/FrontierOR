import argparse
import json
import math
import os
import time

import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


EPS = 1e-7


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, allow_nan=False)
    os.replace(tmp_path, path)


def as_number(x):
    x = float(x)
    if abs(x - round(x)) <= 1e-8:
        return int(round(x))
    return x


def build_and_solve(instance, time_limit, logger):
    start_clock = time.monotonic()

    gp_data = instance["global_parameters"]
    orders = instance["orders"]
    products = instance["products"]
    periods = instance["periods"]
    subperiods = instance["subperiods"]
    travel = instance["travel_time_matrix"]
    production_windows = instance.get("production_windows", {})

    n_orders = len(orders)
    n_products = len(products)
    n_periods = len(periods)
    n_subperiods = len(subperiods)

    order_ids = [int(o["order_id"]) for o in orders]
    product_ids = [int(p["product_id"]) for p in products]
    period_ids = [int(p["period_id"]) for p in periods]
    subperiod_ids = [int(s["subperiod_id"]) for s in subperiods]

    order_pos = {oid: i for i, oid in enumerate(order_ids)}
    product_pos = {pid: j for j, pid in enumerate(product_ids)}
    period_pos = {pid: p for p, pid in enumerate(period_ids)}
    subperiod_pos = {sid: s for s, sid in enumerate(subperiod_ids)}

    period_starts = [float(p["start_time"]) for p in periods]
    period_ends = [float(p["end_time"]) for p in periods]
    subperiod_starts = [float(s["start_time"]) for s in subperiods]
    subperiod_ends = [float(s["end_time"]) for s in subperiods]

    horizon_start = float(gp_data["horizon_start_minutes"])
    horizon_end = float(gp_data["horizon_end_minutes"])
    preparation = float(gp_data["preparation_time_minutes"])
    max_route_minutes = float(gp_data["max_shift_route_length_minutes"])
    max_shift_periods = int(gp_data["max_shift_route_length_periods"])
    max_trips = int(gp_data["max_trips_per_route"])
    min_paid_minutes = float(gp_data["min_paid_time_minutes"])
    min_paid_subperiods = int(gp_data["min_paid_time_subperiods"])
    spp = int(gp_data["subperiods_per_period"])
    setup_len = int(gp_data["setup_length_subperiods"])

    driver_rate = float(gp_data["driver_salary_per_minute"])
    employee_rate = float(gp_data["employee_salary_per_subperiod"])
    setup_cost = float(gp_data["setup_cost"])
    vehicle_rate = float(gp_data["vehicle_operating_cost_per_minute"])

    capacity = float(instance["vehicle_capacity"])
    employee_limit = int(instance["num_production_employees"])
    workstation_limit = int(instance["num_workstations"])
    workforce_limit = min(employee_limit, workstation_limit)

    num_pairs = min(int(instance["num_vehicles"]), int(instance["num_drivers"]))
    if n_orders > 0 and num_pairs <= 0:
        raise RuntimeError("No vehicle/driver pair is available.")

    # More than n_orders routes can never be useful.
    k_count = min(num_pairs, max(1, n_orders))
    K = range(k_count)
    H = range(max_trips)
    I = range(n_orders)
    J = range(n_products)
    P = range(n_periods)
    S = range(n_subperiods)

    demands = {}
    demanded_products = {i: [] for i in I}
    for i, order in enumerate(orders):
        raw = order.get("demands", {})
        for key, val in raw.items():
            pid = int(key)
            if pid not in product_pos:
                continue
            qty = float(val)
            if qty > EPS:
                j = product_pos[pid]
                demands[i, j] = qty
                demanded_products[i].append(j)

    prod_rate = [float(p["production_rate_per_subperiod"]) for p in products]

    valid_subperiods = {}
    for (i, j), demand in demands.items():
        oid = str(order_ids[i])
        pid = str(product_ids[j])
        win = production_windows.get(oid, {}).get(pid)
        if win is None:
            # Be permissive for instances where zero-based JSON keys were
            # serialized differently.
            win = production_windows.get(order_ids[i], {}).get(product_ids[j], {})
        ids = win.get("valid_subperiod_ids", []) if win else []
        positions = []
        for sid in ids:
            sid = int(sid)
            if sid in subperiod_pos:
                positions.append(subperiod_pos[sid])
        valid_subperiods[i, j] = sorted(set(positions))
        if demand > EPS and not valid_subperiods[i, j]:
            raise RuntimeError(
                f"Order {order_ids[i]}, product {product_ids[j]} has no valid production subperiod."
            )

    # Map each subperiod to its containing period.
    subperiod_to_period = {}
    for s in S:
        chosen = None
        for p in P:
            if (subperiod_starts[s] >= period_starts[p] - EPS and
                    subperiod_ends[s] <= period_ends[p] + EPS):
                chosen = p
                break
        if chosen is None:
            chosen = min(
                P,
                key=lambda p: abs(subperiod_starts[s] - period_starts[p])
            )
        subperiod_to_period[s] = chosen

    model = gp.Model("integrated_catering")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.MIPFocus = 1
    model.Params.Heuristics = 0.20
    model.Params.Presolve = 2
    model.Params.TimeLimit = max(1.0, float(time_limit) - 0.5)

    big_m_time = (
        max_route_minutes + (horizon_end - horizon_start)
        + max(max(row) for row in travel) + preparation + 120.0
    )
    big_m_period = n_periods + 1

    used = model.addVars(K, vtype=GRB.BINARY, name="route_used")
    active = model.addVars(K, H, vtype=GRB.BINARY, name="trip_active")
    start_bin = model.addVars(K, H, P, vtype=GRB.BINARY, name="trip_start")
    visit = model.addVars(K, H, I, vtype=GRB.BINARY, name="visit")
    visit_period = model.addVars(K, H, I, P, vtype=GRB.BINARY,
                                 name="visit_period")

    # Nodes use 0 for depot and i+1 for customer array position i.
    arc_keys = []
    for k in K:
        for h in H:
            for a in range(n_orders + 1):
                for b in range(n_orders + 1):
                    if a != b:
                        arc_keys.append((k, h, a, b))
    arc = model.addVars(arc_keys, vtype=GRB.BINARY, name="arc")

    arrival = model.addVars(K, H, I, lb=0.0, ub=horizon_end + big_m_time,
                            vtype=GRB.CONTINUOUS, name="arrival")
    trip_return = model.addVars(K, H, lb=0.0,
                                ub=horizon_end + big_m_time,
                                vtype=GRB.CONTINUOUS, name="trip_return")
    route_paid = model.addVars(K, lb=0.0, ub=max_route_minutes,
                               vtype=GRB.CONTINUOUS, name="route_paid")

    # Production variables.
    quantity = {}
    for (i, j), demand in demands.items():
        for s in valid_subperiods[i, j]:
            quantity[i, j, s] = model.addVar(
                lb=0.0, ub=demand, vtype=GRB.CONTINUOUS,
                name=f"q_{i}_{j}_{s}"
            )

    employees = model.addVars(J, S, lb=0, ub=workforce_limit,
                              vtype=GRB.INTEGER, name="prod_employees")
    setups = model.addVars(J, S, lb=0, ub=workforce_limit,
                           vtype=GRB.INTEGER, name="setups")
    paid_workstations = model.addVars(
        P, lb=0, ub=workforce_limit, vtype=GRB.INTEGER, name="paid_workstations"
    )
    shift_starts = model.addVars(
        P, lb=0, ub=workforce_limit, vtype=GRB.INTEGER, name="shift_starts"
    )

    # Routing structure.
    for k in K:
        model.addConstr(active[k, 0] == used[k])
        for h in H:
            model.addConstr(gp.quicksum(start_bin[k, h, p] for p in P)
                            == active[k, h])
            model.addConstr(gp.quicksum(visit[k, h, i] for i in I)
                            >= active[k, h])
            model.addConstr(gp.quicksum(visit[k, h, i] for i in I)
                            <= n_orders * active[k, h])

            if h + 1 < max_trips:
                model.addConstr(active[k, h + 1] <= active[k, h])

            # Depot degree.
            model.addConstr(
                gp.quicksum(arc[k, h, 0, i + 1] for i in I) == active[k, h]
            )
            model.addConstr(
                gp.quicksum(arc[k, h, i + 1, 0] for i in I) == active[k, h]
            )

            # Customer degree.
            for i in I:
                node = i + 1
                model.addConstr(
                    gp.quicksum(
                        arc[k, h, a, node]
                        for a in range(n_orders + 1) if a != node
                    ) == visit[k, h, i]
                )
                model.addConstr(
                    gp.quicksum(
                        arc[k, h, node, b]
                        for b in range(n_orders + 1) if b != node
                    ) == visit[k, h, i]
                )
                model.addConstr(
                    gp.quicksum(visit_period[k, h, i, p] for p in P)
                    == visit[k, h, i]
                )
                for p in P:
                    model.addConstr(visit_period[k, h, i, p]
                                    <= start_bin[k, h, p])

            model.addConstr(
                gp.quicksum(float(orders[i]["space_requirement"]) *
                            visit[k, h, i] for i in I) <= capacity
            )

        if k + 1 < k_count:
            model.addConstr(used[k] >= used[k + 1])

    # Every customer exactly once.
    for i in I:
        model.addConstr(
            gp.quicksum(visit[k, h, i] for k in K for h in H) == 1
        )

    # Eliminate equivalent gaps and require strictly later start periods.
    for k in K:
        for h in range(max_trips - 1):
            lhs = gp.quicksum(p * start_bin[k, h + 1, p] for p in P)
            rhs = gp.quicksum(p * start_bin[k, h, p] for p in P)
            model.addConstr(
                lhs >= rhs + 1 - big_m_period * (1 - active[k, h + 1])
            )

    # Arrival-time and time-window constraints also eliminate subtours.
    for k in K:
        first_start = gp.quicksum(period_starts[p] * start_bin[k, 0, p]
                                  for p in P)
        model.addConstr(route_paid[k] >= min_paid_minutes * used[k])
        model.addConstr(route_paid[k] <= max_route_minutes * used[k])

        for h in H:
            trip_start_expr = gp.quicksum(
                period_starts[p] * start_bin[k, h, p] for p in P
            )

            for i in I:
                tw_start = float(orders[i]["time_window_start"])
                tw_end = float(orders[i]["time_window_end"])
                service = float(orders[i].get(
                    "service_time", gp_data.get("service_time_minutes", 0)
                ))

                model.addConstr(
                    arrival[k, h, i] >= tw_start -
                    big_m_time * (1 - visit[k, h, i])
                )
                model.addConstr(
                    arrival[k, h, i] <= tw_end +
                    big_m_time * (1 - visit[k, h, i])
                )
                model.addConstr(
                    arrival[k, h, i] >= trip_start_expr + preparation
                    + float(travel[0][i + 1])
                    - big_m_time * (1 - arc[k, h, 0, i + 1])
                )
                model.addConstr(
                    trip_return[k, h] >= arrival[k, h, i] + service
                    + float(travel[i + 1][0])
                    - big_m_time * (1 - arc[k, h, i + 1, 0])
                )

                for j in I:
                    if i == j:
                        continue
                    model.addConstr(
                        arrival[k, h, j] >= arrival[k, h, i] + service
                        + float(travel[i + 1][j + 1])
                        - big_m_time * (1 - arc[k, h, i + 1, j + 1])
                    )

            model.addConstr(
                trip_return[k, h] <= first_start + max_route_minutes
                + big_m_time * (1 - active[k, h])
            )
            model.addConstr(
                route_paid[k] >= trip_return[k, h] - first_start
                - big_m_time * (1 - active[k, h])
            )

    # Production quantities and capacities.
    for (i, j), demand in demands.items():
        model.addConstr(
            gp.quicksum(quantity[i, j, s]
                        for s in valid_subperiods[i, j]) == demand
        )

    for j in J:
        for s in S:
            produced_here = [
                quantity[i, jj, ss]
                for (i, jj, ss) in quantity
                if jj == j and ss == s
            ]
            model.addConstr(
                gp.quicksum(produced_here) <= prod_rate[j] * employees[j, s]
            )

            previous = employees[j, s - 1] if s > 0 else 0
            model.addConstr(employees[j, s] - previous <= setups[j, s])

    # Setup employees remain occupied for setup_len subperiods.
    for s in S:
        setup_occupancy = gp.quicksum(
            setups[j, tau]
            for j in J
            for tau in range(max(0, s - setup_len + 1), s + 1)
        )
        production_occupancy = gp.quicksum(employees[j, s] for j in J)
        p = subperiod_to_period[s]
        model.addConstr(production_occupancy + setup_occupancy
                        <= paid_workstations[p])
        model.addConstr(production_occupancy + setup_occupancy
                        <= employee_limit)
        model.addConstr(production_occupancy + setup_occupancy
                        <= workstation_limit)

    # Aggregate paid-workforce contract.
    min_paid_periods = max(1, int(math.ceil(min_paid_subperiods / max(1, spp))))
    for p in P:
        previous = paid_workstations[p - 1] if p > 0 else 0
        model.addConstr(
            paid_workstations[p] - previous <= shift_starts[p]
        )

        oldest = max(0, p - max_shift_periods + 1)
        model.addConstr(
            paid_workstations[p] <=
            gp.quicksum(shift_starts[r] for r in range(oldest, p + 1))
        )

        paid_range = list(range(p, min(n_periods, p + min_paid_periods)))
        if paid_range:
            model.addConstr(
                gp.quicksum(paid_workstations[r] for r in paid_range)
                >= len(paid_range) * shift_starts[p]
            )

    # Production must be completed by the selected trip departure.
    for (i, j), demand in demands.items():
        valid = valid_subperiods[i, j]
        for p in P:
            eligible = [
                s for s in valid
                if subperiod_ends[s] <= period_starts[p] + EPS
            ]
            cumulative = gp.quicksum(quantity[i, j, s] for s in eligible)
            for k in K:
                for h in H:
                    model.addConstr(
                        cumulative >= demand * visit_period[k, h, i, p]
                    )

    route_travel_cost = gp.quicksum(
        vehicle_rate * float(travel[a][b]) * arc[k, h, a, b]
        for k, h, a, b in arc_keys
    )
    driver_cost = gp.quicksum(driver_rate * route_paid[k] for k in K)

    # A paid workstation covers all subperiods in its period.
    production_salary_cost = gp.quicksum(
        employee_rate * spp * paid_workstations[p] for p in P
    )
    production_setup_cost = gp.quicksum(
        setup_cost * setups[j, s] for j in J for s in S
    )

    model.setObjective(
        driver_cost + route_travel_cost
        + production_salary_cost + production_setup_cost,
        GRB.MINIMIZE
    )

    last_logged = [math.inf]

    def incumbent_callback(cb_model, where):
        if where == GRB.Callback.MIPSOL and logger:
            try:
                obj = cb_model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if obj + 1e-7 < last_logged[0]:
                    last_logged[0] = obj
                    logger.log(float(obj))
            except Exception:
                pass

    model.optimize(incumbent_callback)

    if model.SolCount <= 0:
        raise RuntimeError(
            "No feasible solution was found within the time limit. "
            f"Gurobi status: {model.Status}"
        )

    def val(v):
        return float(v.X)

    def intval(v):
        return int(round(float(v.X)))

    # Extract route order from selected arcs.
    routes = []
    exact_driver_cost = 0.0
    exact_vehicle_cost = 0.0

    for k in K:
        if intval(used[k]) == 0:
            continue

        trips_out = []
        route_returns = []
        earliest_start = None

        for h in H:
            if intval(active[k, h]) == 0:
                continue

            p = max(P, key=lambda pp: val(start_bin[k, h, pp]))
            start_time = period_starts[p]
            if earliest_start is None:
                earliest_start = start_time

            sequence = []
            current_node = 0
            seen = set()
            while True:
                next_nodes = [
                    b for b in range(n_orders + 1)
                    if b != current_node
                    and (k, h, current_node, b) in arc
                    and val(arc[k, h, current_node, b]) > 0.5
                ]
                if not next_nodes:
                    break
                nxt = max(next_nodes,
                          key=lambda b: val(arc[k, h, current_node, b]))
                if nxt == 0:
                    break
                if nxt in seen:
                    break
                seen.add(nxt)
                sequence.append(nxt - 1)
                current_node = nxt

            customer_ids = [order_ids[i] for i in sequence]
            trips_out.append({
                "trip": h,
                "customers": customer_ids,
                "start_period": period_ids[p]
            })

            # Recompute exact earliest feasible trip timing.
            current_time = start_time + preparation
            prev_matrix_node = 0
            trip_travel = 0.0
            for i in sequence:
                matrix_node = i + 1
                leg = float(travel[prev_matrix_node][matrix_node])
                trip_travel += leg
                current_time += leg
                current_time = max(
                    current_time, float(orders[i]["time_window_start"])
                )
                current_time += float(orders[i].get(
                    "service_time", gp_data.get("service_time_minutes", 0)
                ))
                prev_matrix_node = matrix_node
            back = float(travel[prev_matrix_node][0])
            trip_travel += back
            current_time += back

            route_returns.append(current_time)
            exact_vehicle_cost += vehicle_rate * trip_travel

        routes.append({"vehicle": k, "trips": trips_out})
        if route_returns:
            duration = max(route_returns) - earliest_start
            exact_driver_cost += driver_rate * max(duration, min_paid_minutes)

    production_schedule = []
    for (i, j, s), variable in sorted(quantity.items()):
        qv = val(variable)
        if qv > 1e-6:
            production_schedule.append({
                "order": order_ids[i],
                "product": product_ids[j],
                "subperiod": subperiod_ids[s],
                "quantity": as_number(qv)
            })

    employee_entries = []
    setup_entries = []
    for j in J:
        for s in S:
            ev = intval(employees[j, s])
            av = intval(setups[j, s])
            if ev > 0:
                employee_entries.append({
                    "product": product_ids[j],
                    "subperiod": subperiod_ids[s],
                    "employees": ev
                })
            if av > 0:
                setup_entries.append({
                    "product": product_ids[j],
                    "subperiod": subperiod_ids[s],
                    "setups": av
                })

    workstation_values = [intval(paid_workstations[p]) for p in P]
    shift_values = [intval(shift_starts[p]) for p in P]

    exact_employee_cost = employee_rate * spp * sum(workstation_values)
    exact_setup_cost = setup_cost * sum(x["setups"] for x in setup_entries)
    objective = (
        exact_driver_cost + exact_vehicle_cost
        + exact_employee_cost + exact_setup_cost
    )

    solution = {
        "objective_value": float(objective),
        "routes": routes,
        "production_schedule": production_schedule,
        "production_staffing": {
            "employees_by_product_subperiod": employee_entries,
            "setups_by_product_subperiod": setup_entries,
            "workstations_by_period": workstation_values,
            "shift_starts_by_period": shift_values
        }
    }

    if logger:
        logger.log_solution(float(objective), solution)

    return solution


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    instance = read_json(args.instance_path)
    solution = build_and_solve(
        instance=instance,
        time_limit=max(1, args.time_limit),
        logger=logger
    )
    write_json(args.solution_path, solution)


if __name__ == "__main__":
    main()
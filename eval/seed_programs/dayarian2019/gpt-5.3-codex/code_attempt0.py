import argparse
import json
import math
import time
from collections import defaultdict

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", type=str, required=True)
    ap.add_argument("--solution_path", type=str, required=True)
    ap.add_argument("--time_limit", type=int, required=True)
    ap.add_argument("--log_path", type=str, default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    t0 = time.time()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    gp_params = {
        "Seed": 0,
        "MIPGap": 1e-4,
        "NumericFocus": 0,
        "Threads": 1,
    }

    # -------------------------
    # Basic data + mappings
    # -------------------------
    gpms = inst["global_parameters"]
    period_len = gpms["period_length_minutes"]
    subperiods_per_period = gpms["subperiods_per_period"]
    prep_time = gpms["preparation_time_minutes"]
    max_trips_per_route = gpms["max_trips_per_route"]
    max_route_minutes = gpms["max_shift_route_length_minutes"]
    max_route_periods = gpms["max_shift_route_length_periods"]
    min_paid_minutes = gpms["min_paid_time_minutes"]
    min_paid_subperiods = gpms["min_paid_time_subperiods"]
    driver_salary = gpms["driver_salary_per_minute"]
    veh_cost = gpms["vehicle_operating_cost_per_minute"]
    emp_salary_sub = gpms["employee_salary_per_subperiod"]
    setup_cost = gpms["setup_cost"]
    setup_len_sub = gpms["setup_length_subperiods"]

    num_vehicles = inst["num_vehicles"]
    num_drivers = inst["num_drivers"]
    K = min(num_vehicles, num_drivers)

    vehicle_capacity = inst["vehicle_capacity"]
    num_prod_emp = inst["num_production_employees"]
    num_workstations = inst["num_workstations"]

    periods = sorted(inst["periods"], key=lambda x: x["start_time"])
    subperiods = sorted(inst["subperiods"], key=lambda x: x["start_time"])

    period_ids = [p["period_id"] for p in periods]
    period_starts = [p["start_time"] for p in periods]
    period_ends = [p["end_time"] for p in periods]
    T = len(periods)

    sub_ids = [s["subperiod_id"] for s in subperiods]
    sub_starts = [s["start_time"] for s in subperiods]
    sub_ends = [s["end_time"] for s in subperiods]
    S = len(subperiods)

    period_id_to_pos = {pid: i for i, pid in enumerate(period_ids)}
    sub_id_to_pos = {sid: i for i, sid in enumerate(sub_ids)}

    # map each subperiod pos to period pos
    sub_to_period = []
    pi = 0
    for si in range(S):
        st = sub_starts[si]
        while pi + 1 < T and period_starts[pi + 1] <= st:
            pi += 1
        sub_to_period.append(pi)

    orders = inst["orders"]
    order_ids = [o["order_id"] for o in orders]
    oid_to_order = {o["order_id"]: o for o in orders}
    oid_to_idx = {oid: i + 1 for i, oid in enumerate(order_ids)}  # travel matrix index 1..n

    products = inst["products"]
    product_ids = [p["product_id"] for p in products]
    prod_rate = {p["product_id"]: float(p["production_rate_per_subperiod"]) for p in products}

    travel = inst["travel_time_matrix"]  # index 0 depot, 1..n orders by list position

    prod_windows = inst["production_windows"]

    # -------------------------
    # Delivery trip helpers
    # -------------------------
    def order_service_time(oid):
        o = oid_to_order[oid]
        return o.get("service_time", gpms.get("service_time_minutes", 0))

    def tw_start(oid):
        return oid_to_order[oid]["time_window_start"]

    def tw_end(oid):
        return oid_to_order[oid]["time_window_end"]

    def space_req(oid):
        return oid_to_order[oid]["space_requirement"]

    def trip_travel_only(seq):
        if not seq:
            return 0.0
        total = travel[0][oid_to_idx[seq[0]]]
        for a, b in zip(seq[:-1], seq[1:]):
            total += travel[oid_to_idx[a]][oid_to_idx[b]]
        total += travel[oid_to_idx[seq[-1]]][0]
        return float(total)

    def simulate_trip(seq, period_pos):
        """Return (feasible, return_time, travel_only) with waiting allowed."""
        t = period_starts[period_pos] + prep_time
        prev_idx = 0
        for oid in seq:
            idx = oid_to_idx[oid]
            t += travel[prev_idx][idx]
            if t < tw_start(oid):
                t = tw_start(oid)
            if t > tw_end(oid) + 1e-9:
                return False, None, None
            t += order_service_time(oid)
            prev_idx = idx
        t += travel[prev_idx][0]
        return True, t, trip_travel_only(seq)

    def feasible_period_positions(seq):
        feas = []
        for pp in range(T):
            ok, ret_t, _ = simulate_trip(seq, pp)
            if not ok:
                continue
            # optional horizon bound via route max duration later; trip itself just must run in horizon roughly
            if ret_t <= gpms["horizon_end_minutes"] + max_route_minutes + 1e-6:
                feas.append(pp)
        return feas

    # -------------------------
    # Build trips by insertion heuristic
    # -------------------------
    class Trip:
        __slots__ = ("customers", "load", "feas_periods", "sim_cache")

        def __init__(self, customers):
            self.customers = customers[:]  # ordered
            self.load = sum(space_req(c) for c in customers)
            self.feas_periods = feasible_period_positions(self.customers)
            self.sim_cache = {}  # period_pos -> (return_time, travel_only)

        def refresh(self):
            self.load = sum(space_req(c) for c in self.customers)
            self.feas_periods = feasible_period_positions(self.customers)
            self.sim_cache = {}

        def get_sim(self, pp):
            if pp not in self.sim_cache:
                ok, rt, tr = simulate_trip(self.customers, pp)
                if not ok:
                    return None
                self.sim_cache[pp] = (rt, tr)
            return self.sim_cache[pp]

    sorted_orders = sorted(order_ids, key=lambda x: (tw_end(x), tw_start(x)))
    trips = []

    for oid in sorted_orders:
        best = None  # (delta_travel, trip_idx, pos, new_feas)
        for ti, tr in enumerate(trips):
            if tr.load + space_req(oid) > vehicle_capacity:
                continue
            old_travel = trip_travel_only(tr.customers)
            m = len(tr.customers)
            for pos in range(m + 1):
                cand = tr.customers[:pos] + [oid] + tr.customers[pos:]
                if sum(space_req(c) for c in cand) > vehicle_capacity:
                    continue
                feas = feasible_period_positions(cand)
                if not feas:
                    continue
                new_travel = trip_travel_only(cand)
                delta = new_travel - old_travel
                score = (delta, len(feas))  # tie-break by flexibility
                if best is None or score < (best[0], best[3]):
                    best = (delta, ti, pos, len(feas), cand)
        if best is None:
            nt = Trip([oid])
            if not nt.feas_periods:
                # hard fallback: still keep as single with all periods (may become infeasible later)
                nt.feas_periods = list(range(T))
            trips.append(nt)
        else:
            _, ti, pos, _, cand = best
            trips[ti].customers = cand
            trips[ti].refresh()

    # Consolidate if too many trips
    max_total_trips = K * max_trips_per_route if K > 0 else len(trips)
    if max_total_trips <= 0:
        max_total_trips = len(trips)

    def best_insert_customer_into_trip(customer, trip_obj):
        if trip_obj.load + space_req(customer) > vehicle_capacity:
            return None
        old_tr = trip_travel_only(trip_obj.customers)
        best_loc = None
        m = len(trip_obj.customers)
        for pos in range(m + 1):
            cand = trip_obj.customers[:pos] + [customer] + trip_obj.customers[pos:]
            if sum(space_req(c) for c in cand) > vehicle_capacity:
                continue
            feas = feasible_period_positions(cand)
            if not feas:
                continue
            d = trip_travel_only(cand) - old_tr
            if best_loc is None or d < best_loc[0]:
                best_loc = (d, pos, cand)
        return best_loc

    while len(trips) > max_total_trips:
        trips.sort(key=lambda tr: len(tr.customers))
        victim = trips[0]
        others = trips[1:]
        moved_all = True
        for c in victim.customers:
            ins_best = None
            for j, tr in enumerate(others):
                cand = best_insert_customer_into_trip(c, tr)
                if cand is None:
                    continue
                if ins_best is None or cand[0] < ins_best[0]:
                    ins_best = (cand[0], j, cand[2])
            if ins_best is None:
                moved_all = False
                break
            _, j, newcust = ins_best
            others[j].customers = newcust
            others[j].refresh()
        if moved_all:
            trips = others
        else:
            break  # cannot reduce further quickly

    # -------------------------
    # Assign trips to routes (choose period per trip during assignment)
    # -------------------------
    # Prepare trip dicts
    trip_objs = trips
    for tr in trip_objs:
        if not tr.feas_periods:
            tr.feas_periods = list(range(T))

    # Sort harder trips first (less flexibility)
    trip_order = sorted(range(len(trip_objs)),
                        key=lambda i: (len(trip_objs[i].feas_periods), tw_end(trip_objs[i].customers[0])))

    routes = []  # each route: dict
    unassigned = []

    for ti in trip_order:
        tr = trip_objs[ti]
        best_choice = None  # (route_index or None for new, period_pos, added_duration, ret_t, travel_only)

        # existing routes
        for ri, r in enumerate(routes):
            if len(r["trips"]) >= max_trips_per_route:
                continue
            for pp in sorted(tr.feas_periods, reverse=True):
                if pp <= r["last_start_pp"]:
                    continue
                sim = tr.get_sim(pp)
                if sim is None:
                    continue
                rt, trv = sim
                new_start_t = min(r["start_time"], period_starts[pp])
                new_end_t = max(r["end_time"], rt)
                dur = new_end_t - new_start_t
                period_span = max(r["last_start_pp"], pp) - min(r["first_start_pp"], pp) + 1
                if dur > max_route_minutes + 1e-9:
                    continue
                if period_span > max_route_periods:
                    continue
                added = dur - (r["end_time"] - r["start_time"])
                cand = (ri, pp, added, rt, trv)
                if best_choice is None or cand[2] < best_choice[2]:
                    best_choice = cand

        # new route
        if len(routes) < K:
            for pp in sorted(tr.feas_periods, reverse=True):
                sim = tr.get_sim(pp)
                if sim is None:
                    continue
                rt, trv = sim
                dur = rt - period_starts[pp]
                if dur > max_route_minutes + 1e-9:
                    continue
                cand = (None, pp, dur, rt, trv)
                if best_choice is None or cand[2] < best_choice[2]:
                    best_choice = cand

        if best_choice is None:
            unassigned.append(ti)
            continue

        ri, pp, _, rt, trv = best_choice
        trip_info = {
            "trip_idx_global": ti,
            "start_pp": pp,
            "start_period_id": period_ids[pp],
            "start_time": period_starts[pp],
            "return_time": rt,
            "travel_only": trv,
            "customers": tr.customers[:],
        }

        if ri is None:
            routes.append({
                "trips": [trip_info],
                "first_start_pp": pp,
                "last_start_pp": pp,
                "start_time": period_starts[pp],
                "end_time": rt,
            })
        else:
            r = routes[ri]
            r["trips"].append(trip_info)
            r["first_start_pp"] = min(r["first_start_pp"], pp)
            r["last_start_pp"] = max(r["last_start_pp"], pp)
            r["start_time"] = min(r["start_time"], period_starts[pp])
            r["end_time"] = max(r["end_time"], rt)
            r["trips"].sort(key=lambda x: x["start_pp"])

    # Fallback assign unassigned (may violate some compact constraints, but keep complete solution)
    for ti in unassigned:
        tr = trip_objs[ti]
        pp = max(tr.feas_periods) if tr.feas_periods else 0
        sim = tr.get_sim(pp)
        if sim is None:
            ok = False
            for ppf in range(T):
                simx = tr.get_sim(ppf)
                if simx is not None:
                    sim = simx
                    pp = ppf
                    ok = True
                    break
            if not ok:
                sim = (period_starts[pp] + prep_time, trip_travel_only(tr.customers))
        rt, trv = sim
        if routes and len(routes[-1]["trips"]) < max_trips_per_route and pp > routes[-1]["last_start_pp"]:
            r = routes[-1]
            r["trips"].append({
                "trip_idx_global": ti,
                "start_pp": pp,
                "start_period_id": period_ids[pp],
                "start_time": period_starts[pp],
                "return_time": rt,
                "travel_only": trv,
                "customers": tr.customers[:],
            })
            r["last_start_pp"] = max(r["last_start_pp"], pp)
            r["end_time"] = max(r["end_time"], rt)
        else:
            routes.append({
                "trips": [{
                    "trip_idx_global": ti,
                    "start_pp": pp,
                    "start_period_id": period_ids[pp],
                    "start_time": period_starts[pp],
                    "return_time": rt,
                    "travel_only": trv,
                    "customers": tr.customers[:],
                }],
                "first_start_pp": pp,
                "last_start_pp": pp,
                "start_time": period_starts[pp],
                "end_time": rt,
            })

    # limit routes to K if exceeded: merge naive by appending where possible
    if K > 0 and len(routes) > K:
        routes = sorted(routes, key=lambda r: len(r["trips"]), reverse=True)
        kept = routes[:K]
        extra = routes[K:]
        for ex in extra:
            for trip in ex["trips"]:
                placed = False
                for r in kept:
                    if len(r["trips"]) >= max_trips_per_route:
                        continue
                    if trip["start_pp"] <= r["last_start_pp"]:
                        continue
                    new_start = min(r["start_time"], trip["start_time"])
                    new_end = max(r["end_time"], trip["return_time"])
                    if new_end - new_start > max_route_minutes + 1e-9:
                        continue
                    r["trips"].append(trip)
                    r["trips"].sort(key=lambda x: x["start_pp"])
                    r["last_start_pp"] = max(r["last_start_pp"], trip["start_pp"])
                    r["first_start_pp"] = min(r["first_start_pp"], trip["start_pp"])
                    r["start_time"] = new_start
                    r["end_time"] = new_end
                    placed = True
                    break
                if not placed:
                    # force into first route with capacity
                    target = min(kept, key=lambda x: len(x["trips"]))
                    target["trips"].append(trip)
                    target["trips"].sort(key=lambda x: x["start_pp"])
                    target["last_start_pp"] = max(target["last_start_pp"], trip["start_pp"])
                    target["first_start_pp"] = min(target["first_start_pp"], trip["start_pp"])
                    target["start_time"] = min(target["start_time"], trip["start_time"])
                    target["end_time"] = max(target["end_time"], trip["return_time"])
        routes = kept

    # Map order -> departure time (trip start)
    order_departure_time = {}
    for r in routes:
        for tr in r["trips"]:
            dep = tr["start_time"]
            for oid in tr["customers"]:
                order_departure_time[oid] = dep

    # ensure every order has a departure
    for oid in order_ids:
        if oid not in order_departure_time:
            # assign latest period start as fallback
            order_departure_time[oid] = period_starts[-1]

    # -------------------------
    # Fixed routing objective
    # -------------------------
    routing_cost = 0.0
    route_output = []
    for rv, r in enumerate(routes):
        start_t = min(ti["start_time"] for ti in r["trips"])
        end_t = max(ti["return_time"] for ti in r["trips"])
        dur = end_t - start_t
        paid = max(dur, min_paid_minutes)
        dcost = driver_salary * paid
        vcost = veh_cost * sum(ti["travel_only"] for ti in r["trips"])
        routing_cost += dcost + vcost

        out_trips = []
        for k, tr in enumerate(sorted(r["trips"], key=lambda x: x["start_pp"]), start=1):
            out_trips.append({
                "trip": k,
                "customers": tr["customers"],
                "start_period": tr["start_period_id"],
            })
        route_output.append({
            "vehicle": rv,
            "trips": out_trips
        })

    # -------------------------
    # Production + staffing MIP
    # -------------------------
    # Gather demanded (order,product)
    demand_op = {}
    for o in orders:
        oid = o["order_id"]
        for pk, v in o.get("demands", {}).items():
            pid = int(pk) if isinstance(pk, str) and pk.lstrip("-").isdigit() else pk
            if float(v) > 1e-9:
                demand_op[(oid, pid)] = float(v)

    # Valid subperiod sets from production_windows
    valid_subs = {}
    for (oid, pid), d in demand_op.items():
        okey = str(oid)
        pkey = str(pid)
        vw = []
        if okey in prod_windows and pkey in prod_windows[okey]:
            vw = prod_windows[okey][pkey].get("valid_subperiod_ids", [])
        vw = [sid for sid in vw if sid in sub_id_to_pos]
        valid_subs[(oid, pid)] = vw

    time_left = max(1, args.time_limit - int(time.time() - t0))
    m = gp.Model("production_staffing")
    for k, v in gp_params.items():
        m.setParam(k, v)
    m.setParam("TimeLimit", time_left)
    m.setParam("OutputFlag", 0)

    # Variables
    q_vars = {}
    # aggregate produced by product/subperiod
    prod_sub_terms = defaultdict(list)

    for (oid, pid), d in demand_op.items():
        vsubs = valid_subs.get((oid, pid), [])
        for sid in vsubs:
            v = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"q_{oid}_{pid}_{sid}")
            q_vars[(oid, pid, sid)] = v
            prod_sub_terms[(pid, sid)].append(v)

    e_vars = {}
    z_vars = {}
    for pid in product_ids:
        for sid in sub_ids:
            e_vars[(pid, sid)] = m.addVar(lb=0, vtype=GRB.INTEGER, name=f"e_{pid}_{sid}")
            z_vars[(pid, sid)] = m.addVar(lb=0, vtype=GRB.INTEGER, name=f"z_{pid}_{sid}")

    W = [m.addVar(lb=0, ub=min(num_workstations, num_prod_emp), vtype=GRB.INTEGER, name=f"W_{t}") for t in range(T)]
    H = [m.addVar(lb=0, ub=num_prod_emp, vtype=GRB.INTEGER, name=f"H_{t}") for t in range(T)]

    m.update()

    # Demand fulfillment
    for (oid, pid), d in demand_op.items():
        vsubs = valid_subs.get((oid, pid), [])
        if vsubs:
            m.addConstr(gp.quicksum(q_vars[(oid, pid, sid)] for sid in vsubs) >= d, name=f"demand_{oid}_{pid}")
        else:
            # force infeasible if no valid production subperiod for positive demand
            m.addConstr(0 >= d, name=f"demand_infeas_{oid}_{pid}")

        dep = order_departure_time.get(oid, period_starts[-1])
        elig = [sid for sid in vsubs if sub_ends[sub_id_to_pos[sid]] <= dep + 1e-9]
        if elig:
            m.addConstr(gp.quicksum(q_vars[(oid, pid, sid)] for sid in elig) >= d, name=f"dep_{oid}_{pid}")
        else:
            m.addConstr(0 >= d, name=f"dep_infeas_{oid}_{pid}")

    # Production-rate capacity
    for pid in product_ids:
        rate = prod_rate.get(pid, 0.0)
        for sid in sub_ids:
            lhs = gp.quicksum(prod_sub_terms.get((pid, sid), []))
            m.addConstr(lhs <= rate * e_vars[(pid, sid)], name=f"cap_{pid}_{sid}")

    # Setup rise constraints
    for pid in product_ids:
        for s_pos, sid in enumerate(sub_ids):
            if s_pos == 0:
                m.addConstr(e_vars[(pid, sid)] <= z_vars[(pid, sid)], name=f"rise0_{pid}_{sid}")
            else:
                sid_prev = sub_ids[s_pos - 1]
                m.addConstr(e_vars[(pid, sid)] - e_vars[(pid, sid_prev)] <= z_vars[(pid, sid)], name=f"rise_{pid}_{sid}")

    # Employee use each subperiod (production + active setups)
    for s_pos, sid in enumerate(sub_ids):
        prod_use = gp.quicksum(e_vars[(pid, sid)] for pid in product_ids)
        setup_active_terms = []
        if setup_len_sub > 0:
            lo = max(0, s_pos - setup_len_sub + 1)
            for pp in range(lo, s_pos + 1):
                sid2 = sub_ids[pp]
                for pid in product_ids:
                    setup_active_terms.append(z_vars[(pid, sid2)])
        setup_use = gp.quicksum(setup_active_terms) if setup_active_terms else 0
        total_use = prod_use + setup_use
        m.addConstr(total_use <= num_prod_emp, name=f"emp_{sid}")
        ppos = sub_to_period[s_pos]
        m.addConstr(total_use <= W[ppos], name=f"paid_{sid}")

    # Shift aggregation constraints
    for t in range(T):
        prev = W[t - 1] if t > 0 else 0
        m.addConstr(W[t] - prev <= H[t], name=f"winc_{t}")

    min_paid_periods = max(1, math.ceil(min_paid_subperiods / max(1, subperiods_per_period)))
    for t in range(T):
        end = min(T - 1, t + min_paid_periods - 1)
        m.addConstr(gp.quicksum(W[u] for u in range(t, end + 1)) >= (end - t + 1) * H[t], name=f"minpaid_{t}")

    prod_obj = emp_salary_sub * subperiods_per_period * gp.quicksum(W[t] for t in range(T)) + \
               setup_cost * gp.quicksum(z_vars.values())

    m.setObjective(prod_obj, GRB.MINIMIZE)

    if logger:
        m._logger = logger
        m._route_cost = routing_cost
        m._best = float("inf")

        def cb(model, where):
            if where == GRB.Callback.MIPSOL:
                obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                tot = model._route_cost + obj
                if tot + 1e-6 < model._best:
                    model._best = tot
                    model._logger.log(tot)
        m.optimize(cb)
    else:
        m.optimize()

    # -------------------------
    # Extract production solution
    # -------------------------
    production_schedule = []
    emp_by_prod_sub = []
    setups_by_prod_sub = []
    W_sol = [0] * T
    H_sol = [0] * T

    have_sol = m.SolCount > 0 and m.Status in {GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED, GRB.SUBOPTIMAL}
    if have_sol:
        for (oid, pid, sid), var in q_vars.items():
            val = var.X
            if val > 1e-8:
                production_schedule.append({
                    "order": oid,
                    "product": pid,
                    "subperiod": sid,
                    "quantity": float(val)
                })
        for (pid, sid), var in e_vars.items():
            val = int(round(var.X))
            if val > 0:
                emp_by_prod_sub.append({
                    "product": pid,
                    "subperiod": sid,
                    "employees": val
                })
        for (pid, sid), var in z_vars.items():
            val = int(round(var.X))
            if val > 0:
                setups_by_prod_sub.append({
                    "product": pid,
                    "subperiod": sid,
                    "setups": val
                })
        for t in range(T):
            W_sol[t] = int(round(W[t].X))
            H_sol[t] = int(round(H[t].X))
    else:
        # Basic fallback certificate
        # Put all demand at latest valid subperiod before departure if possible
        prod_by_ps = defaultdict(float)
        for (oid, pid), d in demand_op.items():
            vsubs = valid_subs.get((oid, pid), [])
            dep = order_departure_time.get(oid, period_starts[-1])
            elig = [sid for sid in vsubs if sub_ends[sub_id_to_pos[sid]] <= dep + 1e-9]
            target = max(elig, key=lambda sid: sub_starts[sub_id_to_pos[sid]]) if elig else (max(vsubs) if vsubs else sub_ids[0])
            production_schedule.append({
                "order": oid, "product": pid, "subperiod": target, "quantity": float(d)
            })
            prod_by_ps[(pid, target)] += d

        for pid in product_ids:
            prev = 0
            for sid in sub_ids:
                qty = prod_by_ps.get((pid, sid), 0.0)
                need = int(math.ceil(qty / max(1e-9, prod_rate.get(pid, 1.0))))
                if need > 0:
                    emp_by_prod_sub.append({"product": pid, "subperiod": sid, "employees": need})
                rise = max(0, need - prev)
                if rise > 0:
                    setups_by_prod_sub.append({"product": pid, "subperiod": sid, "setups": rise})
                prev = need
        # crude W/H
        use_period = [0] * T
        e_lookup = {(d["product"], d["subperiod"]): d["employees"] for d in emp_by_prod_sub}
        z_lookup = {(d["product"], d["subperiod"]): d["setups"] for d in setups_by_prod_sub}
        for s_pos, sid in enumerate(sub_ids):
            prod_use = sum(e_lookup.get((pid, sid), 0) for pid in product_ids)
            setup_use = 0
            lo = max(0, s_pos - setup_len_sub + 1)
            for pp in range(lo, s_pos + 1):
                sid2 = sub_ids[pp]
                setup_use += sum(z_lookup.get((pid, sid2), 0) for pid in product_ids)
            ppos = sub_to_period[s_pos]
            use_period[ppos] = max(use_period[ppos], prod_use + setup_use)
        for t in range(T):
            W_sol[t] = min(num_prod_emp, min(num_workstations, use_period[t]))
        for t in range(T):
            prev = W_sol[t - 1] if t > 0 else 0
            H_sol[t] = max(0, W_sol[t] - prev)

    # objective recompute
    production_cost = emp_salary_sub * subperiods_per_period * sum(W_sol) + \
                      setup_cost * sum(x["setups"] for x in setups_by_prod_sub)
    objective_value = float(routing_cost + production_cost)

    solution = {
        "objective_value": objective_value,
        "routes": route_output,
        "production_schedule": production_schedule,
        "production_staffing": {
            "employees_by_product_subperiod": emp_by_prod_sub,
            "setups_by_product_subperiod": setups_by_prod_sub,
            "workstations_by_period": [int(x) for x in W_sol],
            "shift_starts_by_period": [int(x) for x in H_sol],
        },
    }

    if logger:
        logger.log_solution(objective_value, solution)

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()
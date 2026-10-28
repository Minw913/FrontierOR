import argparse
import json
import math
from collections import defaultdict
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True, type=str)
    parser.add_argument("--solution_path", required=True, type=str)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", required=False, type=str, default=None)
    return parser.parse_args()


def to_int_dict_keys(d):
    return {int(k): v for k, v in d.items()}


def build_arrivals_absolute(flight, T):
    """
    Assumption:
    arrival_profile is anchored at earliest_start_period in absolute time.
    """
    arr = [0] * T
    pre_horizon = 0
    es = int(flight["earliest_start_period"])
    prof = flight["arrival_profile"]
    for k, v in enumerate(prof):
        t = es + k
        if t < 0:
            pre_horizon += v
        elif t < T:
            arr[t] += v
    return arr, pre_horizon


def simulate_option(flight, hs, ds, ws, T, work_rate, dep_rate, dep_offset, transfer_tau):
    """
    Simulate queue/storage dynamics and derive:
      - workload on carousel (bags on belt, per period)
      - storage occupancy per period
    Feasibility checks are conservative and based on end_period.
    """
    end_p = int(flight["end_period"])
    arr_abs, pre_h = build_arrivals_absolute(flight, T)

    # Basic temporal feasibility
    if hs < int(flight["earliest_start_period"]) or hs > int(flight["latest_start_period"]):
        return None
    if ds < hs:
        return None
    if ds > end_p - dep_offset:
        return None
    if hs > end_p:
        return None

    # If arrival profile has positive arrivals after end period, reject (conservative)
    for t in range(max(0, end_p), T):
        if arr_abs[t] > 0:
            return None

    mu = ws * work_rate

    storage = pre_h
    queue = 0
    from_storage = defaultdict(int)

    workload = {}
    storage_profile = {}
    active_periods = []

    # Simulate on horizon up to end period (inclusive logic uses active <= end_p-1)
    for t in range(T):
        arr_t = arr_abs[t]

        # Before handling start, all arrivals accumulate in storage
        if t < hs:
            storage += arr_t
            direct_in = 0
        else:
            direct_in = arr_t

        # Deplete storage during active horizon, starting at ds
        dep = 0
        if hs <= t <= end_p - 1 and t >= ds and storage > 0:
            dep = min(dep_rate, storage)
            storage -= dep
            arr_to_car_t = t + transfer_tau
            from_storage[arr_to_car_t] += dep

        inflow = direct_in + from_storage[t]

        # Active processing period
        if hs <= t <= end_p - 1:
            active_periods.append(t)
            queue += inflow
            workload[t] = queue  # occupancy before processing (conservative)
            processed = min(mu, queue)
            queue -= processed
        else:
            # outside active window, queue should not be processed by decision definition
            # any inflow outside active window indicates infeasibility
            if inflow > 0:
                return None

        storage_profile[t] = storage

    # End feasibility: all bags should be cleared by end of handling
    # queue must be empty and no pending transfer after end_p - 1
    if queue > 0:
        return None
    if storage > 0:
        return None
    for tt, val in from_storage.items():
        if tt > end_p - 1 and val > 0:
            return None

    return {
        "workload": workload,                # dict t->bags
        "storage_profile": storage_profile,  # dict t->bags
        "active_periods": active_periods
    }


def main():
    args = parse_args()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    T = int(inst["planning_horizon"]["num_periods"])
    target_util = float(inst["objective"]["target_utilization"])
    thresholds = list(inst["objective"]["threshold_values"])
    penalties = list(inst["objective"]["penalties"])
    num_thr = len(thresholds)

    dep_rate = int(inst["central_storage"]["depletion_rate_bags_per_period"])
    dep_offset = int(inst["central_storage"]["depletion_offset_periods"])
    storage_cap = int(inst["central_storage"]["capacity_bags"])
    work_rate = int(inst["working_rate_bags_per_period"])
    transfer_tau = int(inst.get("baggage_transfer_time_periods", 0))

    flights = inst["flights"]
    carousels = inst["carousels"]

    # Carousel data maps
    carousel_by_id = {int(c["carousel_id"]): c for c in carousels}
    carousel_ids = [int(c["carousel_id"]) for c in carousels]
    carousel_type_of = {int(c["carousel_id"]): int(c["type_r"]) for c in carousels}
    Kws = {int(c["carousel_id"]): int(c["Kws"]) for c in carousels}
    Kpp = {int(c["carousel_id"]): int(c["Kpp"]) for c in carousels}
    Kcb = {int(c["carousel_id"]): float(c["Kcb"]) for c in carousels}

    # Peak intervals
    peaks = []
    for p in inst["peak_intervals"]:
        pid = int(p["peak_id"])
        s = max(0, int(p["start_period"]))
        e = min(T - 1, int(p["end_period"]))
        if s <= e:
            peaks.append((pid, s, e))

    # Build options
    # Each option = one complete assignment choice for a flight
    options = []  # list of dict
    options_by_flight = defaultdict(list)

    # Sparse coeff collectors
    ws_usage_terms = defaultdict(list)       # (carousel_id, t) -> [(opt_idx, coeff)]
    pp_usage_terms = defaultdict(list)       # (carousel_id, t) -> [(opt_idx, coeff)]
    storage_terms = defaultdict(list)        # t -> [(opt_idx, coeff)]
    workload_terms = defaultdict(list)       # (carousel_id, t) -> [(opt_idx, coeff)]

    for f_idx, fl in enumerate(flights):
        fid = int(fl["flight_id"])
        n_cont = int(fl["num_containers"])
        es = int(fl["earliest_start_period"])
        ls = int(fl["latest_start_period"])
        end_p = int(fl["end_period"])

        bounds_by_type_raw = fl.get("working_station_bounds_by_carousel_type", {})
        bounds_by_type = to_int_dict_keys(bounds_by_type_raw) if bounds_by_type_raw else {}

        f_opts_before = len(options)

        for c in carousels:
            cid = int(c["carousel_id"])
            ctype = int(c["type_r"])

            if ctype in bounds_by_type:
                wmin = int(bounds_by_type[ctype]["Wmin"])
                wmax = int(bounds_by_type[ctype]["Wmax"])
            else:
                wmin = int(fl.get("Wmin", 1))
                wmax = int(fl.get("Wmax", Kws[cid]))

            wmin = max(1, wmin)
            wmax = min(wmax, Kws[cid])
            if wmin > wmax:
                continue

            for ws in range(wmin, wmax + 1):
                for hs in range(es, ls + 1):
                    ds_max = end_p - dep_offset
                    if ds_max < hs:
                        continue

                    # Reduced candidate depletion starts for tractability
                    mid = (hs + ds_max) // 2
                    ds_candidates = sorted(set([hs, mid, ds_max]))

                    for ds in ds_candidates:
                        sim = simulate_option(
                            fl, hs, ds, ws, T, work_rate, dep_rate, dep_offset, transfer_tau
                        )
                        if sim is None:
                            continue

                        opt_idx = len(options)
                        opt = {
                            "flight_index": f_idx,
                            "flight_id": fid,
                            "carousel_id": cid,
                            "carousel_type": ctype,
                            "working_stations": ws,
                            "handling_start_period": hs,
                            "depletion_start_period": ds,
                            "end_period": end_p,
                            "num_containers": n_cont,
                            "active_periods": sim["active_periods"],
                            "workload": sim["workload"],
                            "storage_profile": sim["storage_profile"],
                        }
                        options.append(opt)
                        options_by_flight[f_idx].append(opt_idx)

                        # Capacity terms
                        for t in sim["active_periods"]:
                            ws_usage_terms[(cid, t)].append((opt_idx, ws))
                            pp_usage_terms[(cid, t)].append((opt_idx, n_cont))

                        # Storage terms
                        for t, s_val in sim["storage_profile"].items():
                            if s_val > 0:
                                storage_terms[t].append((opt_idx, s_val))

                        # Workload terms (for utilization)
                        for t, wv in sim["workload"].items():
                            if wv > 0:
                                workload_terms[(cid, t)].append((opt_idx, wv))

        # Fallback option if none generated
        if len(options) == f_opts_before:
            # Construct a simplistic fallback (may be weak, but keeps model buildable)
            c0 = carousels[0]
            cid = int(c0["carousel_id"])
            ctype = int(c0["type_r"])
            ws = max(1, min(int(fl.get("Wmin", 1)), Kws[cid]))
            hs = es
            ds = hs
            opt_idx = len(options)
            opt = {
                "flight_index": f_idx,
                "flight_id": fid,
                "carousel_id": cid,
                "carousel_type": ctype,
                "working_stations": ws,
                "handling_start_period": hs,
                "depletion_start_period": ds,
                "end_period": end_p,
                "num_containers": n_cont,
                "active_periods": list(range(max(0, hs), min(T - 1, end_p - 1) + 1)) if hs <= end_p - 1 else [],
                "workload": {},
                "storage_profile": {t: 0 for t in range(T)},
            }
            options.append(opt)
            options_by_flight[f_idx].append(opt_idx)
            for t in opt["active_periods"]:
                ws_usage_terms[(cid, t)].append((opt_idx, ws))
                pp_usage_terms[(cid, t)].append((opt_idx, n_cont))

    # Build MIP
    m = Model("baggage_outbound")
    m.Params.OutputFlag = 0
    m.Params.TimeLimit = max(1, int(args.time_limit))
    m.Params.Threads = 1

    x = [m.addVar(vtype=GRB.BINARY, name=f"x_{i}") for i in range(len(options))]

    y = {}  # (carousel_id, peak_id, k)
    for cid in carousel_ids:
        for (pid, _, _) in peaks:
            for k in range(num_thr):
                y[(cid, pid, k)] = m.addVar(vtype=GRB.BINARY, name=f"y_{cid}_{pid}_{k}")

    m.update()

    # Each flight exactly one option
    for f_idx in range(len(flights)):
        opts = options_by_flight[f_idx]
        m.addConstr(quicksum(x[i] for i in opts) == 1, name=f"assign_f{f_idx}")

    # Carousel working-station and parking capacities
    for cid in carousel_ids:
        for t in range(T):
            ws_terms = ws_usage_terms.get((cid, t), [])
            if ws_terms:
                m.addConstr(
                    quicksum(coeff * x[i] for i, coeff in ws_terms) <= Kws[cid],
                    name=f"kws_c{cid}_t{t}"
                )
            pp_terms = pp_usage_terms.get((cid, t), [])
            if pp_terms:
                m.addConstr(
                    quicksum(coeff * x[i] for i, coeff in pp_terms) <= Kpp[cid],
                    name=f"kpp_c{cid}_t{t}"
                )

    # Central storage capacity
    for t in range(T):
        st_terms = storage_terms.get(t, [])
        if st_terms:
            m.addConstr(
                quicksum(coeff * x[i] for i, coeff in st_terms) <= storage_cap,
                name=f"storage_t{t}"
            )

    # Threshold linking + one threshold level per (carousel, peak)
    for cid in carousel_ids:
        for (pid, ps, pe) in peaks:
            m.addConstr(
                quicksum(y[(cid, pid, k)] for k in range(num_thr)) <= 1,
                name=f"one_thr_c{cid}_p{pid}"
            )

            thr_expr = quicksum(thresholds[k] * y[(cid, pid, k)] for k in range(num_thr))
            for t in range(ps, pe + 1):
                wl_terms = workload_terms.get((cid, t), [])
                util_expr = quicksum((coeff / Kcb[cid]) * x[i] for i, coeff in wl_terms) if wl_terms else 0.0
                m.addConstr(
                    util_expr - thr_expr <= target_util,
                    name=f"link_c{cid}_p{pid}_t{t}"
                )

    # Objective: penalties for selected threshold violations
    m.setObjective(
        quicksum(
            penalties[k] * y[(cid, pid, k)]
            for cid in carousel_ids
            for (pid, _, _) in peaks
            for k in range(num_thr)
        ),
        GRB.MINIMIZE
    )

    best_inc = [math.inf]

    def cb(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj + 1e-9 < best_inc[0]:
                best_inc[0] = obj
                if logger:
                    logger.log(float(obj))

    m.optimize(cb)

    # Extract solution
    assignments = []
    util_viol = []

    has_sol = m.SolCount > 0

    if has_sol:
        chosen_opt_by_f = {}
        for i, var in enumerate(x):
            if var.X > 0.5:
                f_idx = options[i]["flight_index"]
                chosen_opt_by_f[f_idx] = i

        # Ensure one assignment per flight (fallback to first if needed)
        for f_idx, fl in enumerate(flights):
            if f_idx not in chosen_opt_by_f:
                chosen_opt_by_f[f_idx] = options_by_flight[f_idx][0]

            opt = options[chosen_opt_by_f[f_idx]]
            assignments.append({
                "flight_id": int(opt["flight_id"]),
                "carousel_id": int(opt["carousel_id"]),
                "carousel_type": int(opt["carousel_type"]),
                "working_stations": int(opt["working_stations"]),
                "handling_start_period": int(opt["handling_start_period"]),
                "depletion_start_period": int(opt["depletion_start_period"]),
                "end_period": int(opt["end_period"])
            })

        for cid in carousel_ids:
            for (pid, _, _) in peaks:
                for k in range(num_thr):
                    if y[(cid, pid, k)].X > 0.5:
                        util_viol.append({
                            "carousel_id": int(cid),
                            "peak_interval": int(pid),
                            "threshold_level": int(k),
                            "threshold_value": float(thresholds[k])
                        })

        objective_value = float(m.ObjVal)
    else:
        # Hard fallback: first option per flight
        for f_idx, fl in enumerate(flights):
            i = options_by_flight[f_idx][0]
            opt = options[i]
            assignments.append({
                "flight_id": int(opt["flight_id"]),
                "carousel_id": int(opt["carousel_id"]),
                "carousel_type": int(opt["carousel_type"]),
                "working_stations": int(opt["working_stations"]),
                "handling_start_period": int(opt["handling_start_period"]),
                "depletion_start_period": int(opt["depletion_start_period"]),
                "end_period": int(opt["end_period"])
            })
        util_viol = []
        objective_value = float("inf")

    out = {
        "objective_value": objective_value,
        "assignments": assignments,
        "utilization_violations": util_viol
    }

    with open(args.solution_path, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
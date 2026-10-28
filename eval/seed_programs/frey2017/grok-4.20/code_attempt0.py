import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    # Extract data
    PH = instance['planning_horizon']['num_periods']
    target_util = instance['objective']['target_utilization']
    thresholds = instance['objective']['threshold_values']
    penalties = instance['objective']['penalties']
    num_thresholds = len(thresholds)
    peak_intervals = instance['peak_intervals']
    num_peaks = len(peak_intervals)
    central_capacity = instance['central_storage']['capacity_bags']
    depletion_rate = instance['central_storage']['depletion_rate_bags_per_period']
    depletion_offset = instance['central_storage']['depletion_offset_periods']
    working_rate = instance['working_rate_bags_per_period']
    baggage_transfer_time = instance['baggage_transfer_time_periods']
    flight_end_offset = instance['flight_end_offset_periods']
    flights = instance['flights']
    carousels = instance['carousels']
    num_flights = len(flights)
    num_carousels = len(carousels)
    carousel_types = instance['carousel_types']

    # Precompute for each flight possible (carousel, w, s, d)
    # To keep model size manageable we will use variables x[f,c,w,s,d]
    # But only create where feasible

    model = gp.Model("BaggageHandling")
    model.Params.TimeLimit = args.time_limit
    model.Params.MIPGap = 0.01
    model.Params.Threads = 1
    model.Params.LogToConsole = 0

    # Decision variables
    # x[f, c, w, s, d] = 1 if flight f assigned to carousel c with w stations, handling start s, depletion start d
    x = {}

    flight_possible = [[] for _ in range(num_flights)]

    for f_idx in range(num_flights):
        f = flights[f_idx]
        fid = f['flight_id']
        earliest = f['earliest_start_period']
        latest = f['latest_start_period']
        end_p = f['end_period']
        arrival_profile = f['arrival_profile']
        handling_dur = f['handling_duration_periods']
        num_cont = f['num_containers']
        wmin_global = f['Wmin']
        wmax_global = f['Wmax']
        ws_by_type = f['working_station_bounds_by_carousel_type']

        total_bags = sum(arrival_profile)

        for c_idx in range(num_carousels):
            c = carousels[c_idx]
            cid = c['carousel_id']
            ctype = str(c['type_r'])
            if ctype not in ws_by_type:
                continue
            bounds = ws_by_type[ctype]
            w_min = bounds['Wmin']
            w_max = bounds['Wmax']
            Kpp = c['Kpp']
            Kppws = c['Kppws']
            # Check container feasibility
            if num_cont > Kpp:
                continue

            for w in range(max(wmin_global, w_min), min(wmax_global, w_max) + 1):
                # Compute processing rate
                process_rate = w * working_rate
                # Minimum handling time in periods
                min_handle = (total_bags + process_rate - 1) // process_rate if process_rate > 0 else PH
                # Feasible start times
                for s in range(earliest, latest + 1):
                    actual_handle_end = s + min_handle
                    if actual_handle_end > end_p:
                        continue
                    # Depletion start
                    # Must start depletion so that all stored bags are moved by end - offset
                    latest_depletion_start = end_p - depletion_offset
                    # Depletion must not start before handling
                    earliest_depletion = s
                    # Storage receives all early bags before s, then depletes at depletion_rate
                    # We will compute stored amount later, but for feasibility of (s,d)
                    for d in range(earliest_depletion, latest_depletion_start + 1):
                        # Also need to respect baggage transfer time? The depletion_offset already incorporates need to finish early
                        x_key = (f_idx, c_idx, w, s, d)
                        x[x_key] = model.addVar(vtype=GRB.BINARY, name=f"x_f{fid}_c{cid}_w{w}_s{s}_d{d}")
                        flight_possible[f_idx].append((c_idx, w, s, d))

    model.update()

    # Each flight assigned exactly once
    for f_idx in range(num_flights):
        if not flight_possible[f_idx]:
            # Infeasible instance
            print("No feasible assignment for flight", flights[f_idx]['flight_id'], file=sys.stderr)
            continue
        model.addConstr(gp.quicksum(x[k] for k in flight_possible[f_idx] if k in x) == 1, name=f"assign_f{f_idx}")

    # Carousel working station capacity per period
    for c_idx in range(num_carousels):
        c = carousels[c_idx]
        Kws = c['Kws']
        for t in range(PH):
            active = []
            for f_idx in range(num_flights):
                f = flights[f_idx]
                for tup in flight_possible[f_idx]:
                    cc, w, s, d = tup
                    if cc != c_idx:
                        continue
                    # When is this flight active? From s to end of its handling
                    # We need to know how long it actually runs for given w
                    arrival_profile = f['arrival_profile']
                    total_bags_f = sum(arrival_profile)
                    process_rate = w * working_rate
                    handle_len = (total_bags_f + process_rate - 1) // process_rate if process_rate > 0 else 1
                    e_f = s + handle_len
                    if s <= t < e_f:
                        key = (f_idx, cc, w, s, d)
                        if key in x:
                            active.append((x[key], w))
            if active:
                model.addConstr(gp.quicksum(w_var * coeff for w_var, coeff in active) <= Kws, name=f"ws_c{c_idx}_t{t}")

    # Container parking capacity per carousel per period
    for c_idx in range(num_carousels):
        c = carousels[c_idx]
        Kpp = c['Kpp']
        for t in range(PH):
            active = []
            for f_idx in range(num_flights):
                f = flights[f_idx]
                num_cont = f['num_containers']
                end_p_f = f['end_period']
                for tup in flight_possible[f_idx]:
                    cc, w, s, d = tup
                    if cc != c_idx:
                        continue
                    # Containers parked from s until end_p_f ?
                    # Typically from start of handling until departure, but to be conservative assume from s until end_p_f
                    if s <= t < end_p_f:
                        key = (f_idx, cc, w, s, d)
                        if key in x:
                            active.append(x[key])
            if active:
                model.addConstr(gp.quicksum(active) * num_cont <= Kpp, name=f"cont_c{c_idx}_t{t}")

    # Central storage capacity
    # This is complex: need to compute stored bags per period as function of chosen (s,d)
    # For each flight, for each possible assignment, we need the storage profile
    # We precompute for each (f, s, d) the storage usage in each period
    storage_usage = {}  # (f_idx, s, d, t) -> bags in storage at beginning of t

    for f_idx in range(num_flights):
        f = flights[f_idx]
        arrival_profile = f['arrival_profile']
        handling_dur = len(arrival_profile)
        earliest_arrival_period = f['earliest_start_period'] - handling_dur
        # Bags arriving in relative period k arrive in absolute time earliest_arrival_period + k
        for tup in flight_possible[f_idx]:
            c_idx, w, s, d = tup
            # Compute cumulative arrivals before each time
            # Arrivals are assumed to arrive at the beginning of each period in the handling window
            # The handling window for arrivals is from earliest_start - handling_dur +1 to earliest_start?
            # The instance gives arrival_profile over handling_duration_periods, and earliest/latest start.
            # We assume the arrival profile is relative to the chosen handling start? No, typically the arrival profile is fixed in time.
            # Looking back at problem: "a known baggage arrival profile specifying the number of bags arriving in each period of its handling duration"
            # "handling_duration_periods"
            # It is given per flight in instance as fixed list, and handling_duration_periods.
            # earliest_start_period, latest_start_period, end_period.
            # I believe the arrival_profile[i] corresponds to bags arriving in period = (earliest possible arrival time + i)
            # But to make it consistent, we assume the arrival_profile is fixed in absolute time: the list corresponds to periods from 
            # (earliest_start_period - handling_duration_periods) to (earliest_start_period - 1)
            arrival_start_period = f['earliest_start_period'] - handling_dur
            fixed_arrival = [0] * PH
            for i in range(handling_dur):
                arr_period = arrival_start_period + i
                if 0 <= arr_period < PH:
                    fixed_arrival[arr_period] = arrival_profile[i]

            # Now for chosen s and d:
            # Bags that arrive before s go to storage.
            # Then from d onwards, storage is depleted at depletion_rate per period, but not before s.
            # The problem says: "once handling starts, stored bags are depleted from the storage to the assigned carousel at a constant rate"
            # "The storage depletion cannot begin before the handling start, and must commence early enough (by an offset that depends on the handling start time, typically several periods before the handling end)"
            # So depletion_start d >= s, and d <= end - depletion_offset
            # The depletion is at rate = depletion_rate_bags_per_period until storage is zero.
            # Also there is "baggage_transfer_time_periods" which might be the time for bags to move from storage to carousel.

            # To simplify for model, we compute the storage level for each period t for this choice of (s,d)
            stored = [0] * PH
            current_storage = 0
            depleted = 0
            for t in range(PH):
                # Add arrivals in this period
                current_storage += fixed_arrival[t]
                # If t >= d and handling has started i.e. t >= s, we can deplete
                if t >= max(s, d):
                    can_deplete = min(current_storage, depletion_rate)
                    current_storage -= can_deplete
                    depleted += can_deplete
                stored[t] = current_storage
                # Also after handling end we may still have depletion but we stop at end_period
                if t >= f['end_period']:
                    break

            # Also the problem mentions "The storage capacity may be tightened on a per-period basis by subtracting the baggage known to have arrived before the earliest possible handling start of flights that have not yet begun handling"
            # This is a bit complex. For now, we use the raw storage and rely on central_capacity.

            for t in range(PH):
                if stored[t] > 0:
                    storage_usage[(f_idx, s, d, t)] = stored[t]

    # Add storage constraint
    for t in range(PH):
        terms = []
        for f_idx in range(num_flights):
            f = flights[f_idx]
            for tup in flight_possible[f_idx]:
                c_idx, w, s, d = tup
                key_var = (f_idx, c_idx, w, s, d)
                if key_var not in x:
                    continue
                for tt in range(t + 1):  # if storage at t depends on cumulative
                    usage_key = (f_idx, s, d, t)
                    if usage_key in storage_usage:
                        terms.append(storage_usage[usage_key] * x[key_var])
        if terms:
            model.addConstr(gp.quicksum(terms) <= central_capacity, name=f"storage_t{t}")

    # Workload on carousel and utilization
    # Workload in period t for flight = bags being loaded in that period on carousel.
    # If flight starts at s, then in period t, if t >= s, the number of bags loaded is min(remaining, process_rate)
    # The problem defines: "the flight's workload on a carousel during each period (the number of bags present on the conveyor belt attributable to that flight)"

    # For simplicity and to respect Kcb, we need to model the bags on belt.
    # But this can be very complicated. Since we have working stations constraint and process rate, we assume the workload in period t is the number of bags processed in that period, bounded by Kcb.

    # To make the model tractable, we introduce auxiliary for each carousel and peak, the max utilization level.

    # Utilization = total workload / Kcb
    # Workload = sum over flights of bags processed in that period.

    # So we need per (f, assignment, t) the bags processed on carousel in t.

    workload_contrib = {}  # (f_idx, c_idx, w, s, d, t) -> bags processed

    for f_idx in range(num_flights):
        f = flights[f_idx]
        total_bags_f = f['expected_baggage']
        arrival_profile = f['arrival_profile']
        handling_dur = len(arrival_profile)
        arrival_start = f['earliest_start_period'] - handling_dur
        fixed_arrivals = [0] * PH
        for i, bags in enumerate(arrival_profile):
            p = arrival_start + i
            if 0 <= p < PH:
                fixed_arrivals[p] = bags

        for tup in flight_possible[f_idx]:
            c_idx, w, s, d = tup
            process_rate = w * working_rate
            # Simulate the process
            remaining = total_bags_f
            processed_per_period = [0] * PH
            for t in range(s, PH):
                if remaining <= 0:
                    break
                proc = min(remaining, process_rate)
                processed_per_period[t] = proc
                remaining -= proc
                if t >= f['end_period']:
                    break
            for t in range(PH):
                if processed_per_period[t] > 0:
                    workload_contrib[(f_idx, c_idx, w, s, d, t)] = processed_per_period[t]

    # Now for utilization constraints during peaks
    threshold_vars = {}
    for c_idx in range(num_carousels):
        c = carousels[c_idx]
        Kcb = c['Kcb']
        for p_idx, peak in enumerate(peak_intervals):
            pstart = peak['start_period']
            pend = peak['end_period']
            for thresh_idx in range(num_thresholds):
                tv = model.addVar(vtype=GRB.BINARY, name=f"thresh_c{c_idx}_p{p_idx}_th{thresh_idx}")
                threshold_vars[(c_idx, p_idx, thresh_idx)] = tv

            # At most one threshold per carousel per peak
            model.addConstr(gp.quicksum(threshold_vars[(c_idx, p_idx, k)] for k in range(num_thresholds)) <= 1,
                            name=f"atmostone_c{c_idx}_p{p_idx}")

            for t in range(pstart, pend + 1):
                # Total workload at t on this carousel
                total_work = []
                for f_idx in range(num_flights):
                    for tup in flight_possible[f_idx]:
                        cc, w, s, d = tup
                        if cc != c_idx:
                            continue
                        keyx = (f_idx, cc, w, s, d)
                        if keyx not in x:
                            continue
                        contrib_key = (f_idx, cc, w, s, d, t)
                        if contrib_key in workload_contrib:
                            total_work.append(workload_contrib[contrib_key] * x[keyx])

                util = gp.quicksum(total_work) / Kcb
                # util - sum (threshold_values[k] * y[k]) <= target
                lhs = util
                for k in range(num_thresholds):
                    lhs -= thresholds[k] * threshold_vars[(c_idx, p_idx, k)]
                model.addConstr(lhs <= target_util, name=f"util_c{c_idx}_p{p_idx}_t{t}")

    # Objective
    obj = gp.LinExpr()
    for c_idx in range(num_carousels):
        for p_idx in range(num_peaks):
            for k in range(num_thresholds):
                obj += penalties[k] * threshold_vars[(c_idx, p_idx, k)]

    model.setObjective(obj, GRB.MINIMIZE)

    # Callback for logging incumbents
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            objval = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if logger:
                logger.log(objval)

    # Optimize
    start_time = time.time()
    model.optimize(callback)

    # Extract solution
    if model.status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED):
        # Try to get any feasible
        if model.SolCount == 0:
            print("No feasible solution found", file=sys.stderr)
            # Write empty
            result = {
                "objective_value": 0.0,
                "assignments": [],
                "utilization_violations": []
            }
            with open(args.solution_path, 'w') as f:
                json.dump(result, f, indent=2)
            return

    objective_value = model.objVal if model.objVal is not None else 0.0

    assignments = []
    for key, var in x.items():
        if var.X > 0.5:
            f_idx, c_idx, w, s, d = key
            f = flights[f_idx]
            c = carousels[c_idx]
            assignments.append({
                "flight_id": f['flight_id'],
                "carousel_id": c['carousel_id'],
                "carousel_type": c['type_r'],
                "working_stations": int(w),
                "handling_start_period": int(s),
                "depletion_start_period": int(d),
                "end_period": f['end_period']
            })

    # Extract violations
    utilization_violations = []
    for c_idx in range(num_carousels):
        for p_idx in range(num_peaks):
            for k in range(num_thresholds):
                var = threshold_vars.get((c_idx, p_idx, k))
                if var and var.X > 0.5:
                    utilization_violations.append({
                        "carousel_id": carousels[c_idx]['carousel_id'],
                        "peak_interval": peak_intervals[p_idx]['peak_id'],
                        "threshold_level": k,
                        "threshold_value": thresholds[k]
                    })

    result = {
        "objective_value": float(objective_value),
        "assignments": assignments,
        "utilization_violations": utilization_violations
    }

    with open(args.solution_path, 'w') as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
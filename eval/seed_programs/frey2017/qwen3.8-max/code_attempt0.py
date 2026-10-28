#!/usr/bin/env python3
"""
Baggage Handling System Optimization
Solves the carousel assignment and scheduling problem using MIP (Gurobi) with heuristic fallback.
"""

import json
import argparse
import time
import sys
import numpy as np
from collections import defaultdict


def compute_profiles(arrival, D, W, s_h, s_d, working_rate, depletion_rate, num_periods):
    """Compute workload (bags on carousel belt) and storage profiles for a given option."""
    # Cumulative arrivals at end of each period
    cum_arr = np.zeros(num_periods)
    for i in range(D):
        t = s_h + i
        if 0 <= t < num_periods:
            cum_arr[t] += arrival[i]
    np.cumsum(cum_arr, out=cum_arr)

    # Cumulative depleted from storage at end of each period
    cum_depl = np.zeros(num_periods)
    for t in range(s_d, num_periods):
        periods_depl = t - s_d + 1
        cum_depl[t] = min(cum_arr[t], depletion_rate * periods_depl)

    # Cumulative processed at carousel at end of each period
    cum_proc = np.zeros(num_periods)
    for t in range(s_h, num_periods):
        periods_work = t - s_h + 1
        cum_proc[t] = min(cum_depl[t], W * working_rate * periods_work)

    workload = np.maximum(0, cum_depl - cum_proc)
    storage = np.maximum(0, cum_arr - cum_depl)

    return workload, storage


def enumerate_options(instance, max_options_per_flight=None):
    """Enumerate feasible options for each flight."""
    num_periods = instance['planning_horizon']['num_periods']
    working_rate = instance['working_rate_bags_per_period']
    depletion_rate = instance['central_storage']['depletion_rate_bags_per_period']
    depletion_offset = instance['central_storage']['depletion_offset_periods']

    flights = instance['flights']
    carousels = instance['carousels']

    flight_options = []

    for fi, flight in enumerate(flights):
        D = flight['handling_duration_periods']
        arrival = flight['arrival_profile']
        earliest_start = flight['earliest_start_period']
        latest_start = flight['latest_start_period']
        end_period = flight['end_period']
        num_containers = flight['num_containers']

        options = []

        for ci, carousel in enumerate(carousels):
            cid = carousel['carousel_id']
            ctype = carousel['type_r']
            Kpp = carousel['Kpp']
            Kws = carousel['Kws']
            Kcb = carousel['Kcb']

            type_key = str(ctype)
            ws_bounds = flight.get('working_station_bounds_by_carousel_type', {})
            if type_key in ws_bounds:
                Wmin = ws_bounds[type_key]['Wmin']
                Wmax = ws_bounds[type_key]['Wmax']
            else:
                Wmin = flight['Wmin']
                Wmax = flight['Wmax']

            if num_containers > Kpp:
                continue

            for W in range(Wmin, min(Wmax, Kws) + 1):
                for s_h in range(earliest_start, latest_start + 1):
                    if s_h + D - 1 > end_period:
                        continue

                    s_d_min = s_h
                    s_d_max = end_period - depletion_offset

                    if s_d_min > s_d_max:
                        continue

                    for s_d in range(s_d_min, s_d_max + 1):
                        workload, storage = compute_profiles(
                            arrival, D, W, s_h, s_d,
                            working_rate, depletion_rate, num_periods
                        )

                        option = {
                            'flight_idx': fi,
                            'carousel_idx': ci,
                            'carousel_id': cid,
                            'carousel_type': ctype,
                            'W': W,
                            'handling_start': s_h,
                            'depletion_start': s_d,
                            'end_period': end_period,
                            'num_containers': num_containers,
                            'workload': workload,
                            'storage': storage,
                        }
                        options.append(option)

        # Limit options if needed
        if max_options_per_flight and len(options) > max_options_per_flight:
            # Prioritize options with lower peak workload during typical peak times
            options.sort(key=lambda o: np.max(o['workload']) if len(o['workload']) > 0 else 0)
            options = options[:max_options_per_flight]

        flight_options.append(options)

    return flight_options


def evaluate_objective(assignments, instance):
    """Evaluate the objective value for a given assignment."""
    target_util = instance['objective']['target_utilization']
    thresholds = instance['objective']['threshold_values']
    penalties = instance['objective']['penalties']
    peak_intervals = instance['peak_intervals']
    carousels = instance['carousels']
    num_periods = instance['planning_horizon']['num_periods']
    num_carousels = len(carousels)

    # Compute total workload per carousel per period
    carousel_workload = np.zeros((num_carousels, num_periods))
    for opt in assignments:
        if opt is not None:
            ci = opt['carousel_idx']
            carousel_workload[ci] += opt['workload']

    total_penalty = 0
    violations = []

    for ci, carousel in enumerate(carousels):
        Kcb = carousel['Kcb']
        if Kcb == 0:
            continue
        for pi, peak in enumerate(peak_intervals):
            max_util = 0.0
            for t in range(peak['start_period'], min(peak['end_period'] + 1, num_periods)):
                util = carousel_workload[ci, t] / Kcb
                max_util = max(max_util, util)

            excess = max_util - target_util
            if excess > 1e-9:
                # Find the minimum threshold that covers this excess
                flagged_k = None
                for k in range(len(thresholds)):
                    if excess <= thresholds[k] + 1e-9:
                        flagged_k = k
                        break
                if flagged_k is None:
                    flagged_k = len(thresholds) - 1

                total_penalty += penalties[flagged_k]
                violations.append({
                    'carousel_id': carousel['carousel_id'],
                    'peak_interval': peak['peak_id'],
                    'threshold_level': flagged_k,
                    'threshold_value': thresholds[flagged_k]
                })

    return total_penalty, violations


def greedy_heuristic(instance, flight_options):
    """Greedy heuristic to find a feasible solution."""
    num_periods = instance['planning_horizon']['num_periods']
    target_util = instance['objective']['target_utilization']
    peak_intervals = instance['peak_intervals']
    storage_capacity = instance['central_storage']['capacity_bags']
    carousels = instance['carousels']
    flights = instance['flights']
    num_carousels = len(carousels)

    carousel_workload = np.zeros((num_carousels, num_periods))
    carousel_ws_used = np.zeros((num_carousels, num_periods), dtype=int)
    carousel_pp_used = np.zeros((num_carousels, num_periods), dtype=int)
    storage_used = np.zeros(num_periods)

    assignments = [None] * len(flights)

    # Sort flights by expected baggage (descending)
    flight_order = sorted(range(len(flights)),
                          key=lambda f: flights[f]['expected_baggage'], reverse=True)

    # Precompute peak periods set
    peak_period_set = set()
    for peak in peak_intervals:
        for t in range(peak['start_period'], peak['end_period'] + 1):
            peak_period_set.add(t)

    for fi in flight_order:
        best_option = None
        best_cost = float('inf')

        for oi, opt in enumerate(flight_options[fi]):
            ci = opt['carousel_idx']
            feasible = True

            # Check WS and PP capacity
            for t in range(opt['handling_start'], min(opt['end_period'] + 1, num_periods)):
                if carousel_ws_used[ci, t] + opt['W'] > carousels[ci]['Kws']:
                    feasible = False
                    break
                if carousel_pp_used[ci, t] + opt['num_containers'] > carousels[ci]['Kpp']:
                    feasible = False
                    break
            if not feasible:
                continue

            # Check storage
            for t in range(num_periods):
                if opt['storage'][t] > 0 and storage_used[t] + opt['storage'][t] > storage_capacity:
                    feasible = False
                    break
            if not feasible:
                continue

            # Evaluate cost
            cost = 0.0
            Kcb = carousels[ci]['Kcb']
            for peak in peak_intervals:
                for t in range(peak['start_period'], min(peak['end_period'] + 1, num_periods)):
                    new_wl = carousel_workload[ci, t] + opt['workload'][t]
                    util = new_wl / Kcb if Kcb > 0 else 0
                    if util > target_util:
                        cost += (util - target_util) ** 2

            if cost < best_cost:
                best_cost = cost
                best_option = opt

        if best_option is not None:
            ci = best_option['carousel_idx']
            assignments[fi] = best_option
            for t in range(best_option['handling_start'], min(best_option['end_period'] + 1, num_periods)):
                carousel_ws_used[ci, t] += best_option['W']
                carousel_pp_used[ci, t] += best_option['num_containers']
            for t in range(num_periods):
                carousel_workload[ci, t] += best_option['workload'][t]
                storage_used[t] += best_option['storage'][t]

    return assignments


def local_search(instance, flight_options, assignments, max_iterations=1000, time_limit=None):
    """Local search to improve the heuristic solution."""
    start_time = time.time()
    flights = instance['flights']
    carousels = instance['carousels']
    num_periods = instance['planning_horizon']['num_periods']
    num_flights = len(flights)

    best_obj, _ = evaluate_objective(assignments, instance)
    best_assignments = [a for a in assignments]

    for iteration in range(max_iterations):
        if time_limit and (time.time() - start_time) > time_limit:
            break

        improved = False
        for fi in range(num_flights):
            if assignments[fi] is None:
                continue

            current_opt = assignments[fi]
            current_ci = current_opt['carousel_idx']

            for oi, opt in enumerate(flight_options[fi]):
                if opt is current_opt:
                    continue

                # Try swapping
                assignments[fi] = opt
                new_obj, _ = evaluate_objective(assignments, instance)

                if new_obj < best_obj:
                    best_obj = new_obj
                    best_assignments = [a for a in assignments]
                    improved = True
                    break
                else:
                    assignments[fi] = current_opt

            if improved:
                break

        if not improved:
            break

        assignments = best_assignments[:]

    return best_assignments, best_obj


def solve_mip(instance, flight_options, time_remaining, logger=None):
    """Solve using Gurobi MIP."""
    import gurobipy as gp
    from gurobipy import GRB

    num_periods = instance['planning_horizon']['num_periods']
    target_util = instance['objective']['target_utilization']
    thresholds = instance['objective']['threshold_values']
    penalties = instance['objective']['penalties']
    num_thresholds = len(thresholds)
    peak_intervals = instance['peak_intervals']
    storage_capacity = instance['central_storage']['capacity_bags']

    flights = instance['flights']
    carousels = instance['carousels']
    num_flights = len(flights)
    num_carousels = len(carousels)

    # Flatten options
    all_options = []
    flight_option_indices = []  # flight_option_indices[fi] = list of global indices
    for fi, options in enumerate(flight_options):
        indices = []
        for opt in options:
            indices.append(len(all_options))
            all_options.append(opt)
        flight_option_indices.append(indices)

    total_options = len(all_options)
    if total_options == 0:
        return None

    # Precompute: for each carousel, list of option indices
    carousel_opt_indices = defaultdict(list)
    for idx, opt in enumerate(all_options):
        carousel_opt_indices[opt['carousel_idx']].append(idx)

    # Create model
    model = gp.Model("baggage")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = max(1, time_remaining)
    model.Params.OutputFlag = 0

    # Decision variables
    x = model.addVars(total_options, vtype=GRB.BINARY, name="x")

    # Threshold variables
    y = {}
    for ci in range(num_carousels):
        for pi in range(len(peak_intervals)):
            for k in range(num_thresholds):
                y[ci, pi, k] = model.addVar(vtype=GRB.BINARY, name=f"y_{ci}_{pi}_{k}")

    model.update()

    # Objective
    obj = gp.LinExpr()
    for ci in range(num_carousels):
        for pi in range(len(peak_intervals)):
            for k in range(num_thresholds):
                obj += penalties[k] * y[ci, pi, k]
    model.setObjective(obj, GRB.MINIMIZE)

    # Constraint 1: Exactly one option per flight
    for fi in range(num_flights):
        indices = flight_option_indices[fi]
        if len(indices) == 0:
            return None
        model.addConstr(
            gp.quicksum(x[idx] for idx in indices) == 1,
            name=f"assign_{fi}"
        )

    # Constraint 2: Working station capacity per carousel per period
    for ci in range(num_carousels):
        Kws = carousels[ci]['Kws']
        opt_indices = carousel_opt_indices[ci]
        if not opt_indices:
            continue
        for t in range(num_periods):
            terms = []
            for idx in opt_indices:
                opt = all_options[idx]
                if opt['handling_start'] <= t <= opt['end_period']:
                    terms.append((opt['W'], idx))
            if terms:
                model.addConstr(
                    gp.quicksum(w * x[idx] for w, idx in terms) <= Kws,
                    name=f"ws_{ci}_{t}"
                )

    # Constraint 3: Parking position capacity
    for ci in range(num_carousels):
        Kpp = carousels[ci]['Kpp']
        opt_indices = carousel_opt_indices[ci]
        if not opt_indices:
            continue
        for t in range(num_periods):
            terms = []
            for idx in opt_indices:
                opt = all_options[idx]
                if opt['handling_start'] <= t <= opt['end_period']:
                    terms.append((opt['num_containers'], idx))
            if terms:
                model.addConstr(
                    gp.quicksum(nc * x[idx] for nc, idx in terms) <= Kpp,
                    name=f"pp_{ci}_{t}"
                )

    # Constraint 4: Storage capacity
    for t in range(num_periods):
        terms = []
        for idx, opt in enumerate(all_options):
            if opt['storage'][t] > 0.01:
                terms.append((opt['storage'][t], idx))
        if terms:
            model.addConstr(
                gp.quicksum(s * x[idx] for s, idx in terms) <= storage_capacity,
                name=f"storage_{t}"
            )

    # Constraint 5: Utilization linking during peak intervals
    for ci in range(num_carousels):
        Kcb = carousels[ci]['Kcb']
        if Kcb == 0:
            continue
        opt_indices = carousel_opt_indices[ci]
        if not opt_indices:
            continue
        for pi, peak in enumerate(peak_intervals):
            for t in range(peak['start_period'], min(peak['end_period'] + 1, num_periods)):
                terms = []
                for idx in opt_indices:
                    wl = all_options[idx]['workload'][t]
                    if wl > 0.01:
                        terms.append((wl, idx))
                if terms:
                    lhs = gp.quicksum(wl * x[idx] for wl, idx in terms)
                    rhs = Kcb * target_util + gp.quicksum(
                        Kcb * thresholds[k] * y[ci, pi, k] for k in range(num_thresholds)
                    )
                    model.addConstr(lhs <= rhs, name=f"util_{ci}_{pi}_{t}")

    # Constraint 6: At most one threshold per carousel per peak
    for ci in range(num_carousels):
        for pi in range(len(peak_intervals)):
            model.addConstr(
                gp.quicksum(y[ci, pi, k] for k in range(num_thresholds)) <= 1,
                name=f"one_thresh_{ci}_{pi}"
            )

    # Callback for logging
    def mip_callback(model_cb, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model_cb.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj_val)

    model.optimize(mip_callback)

    if model.SolCount == 0:
        return None

    # Extract solution
    x_vals = model.getAttr("X", x)
    y_vals = model.getAttr("X", y)

    assignments = [None] * num_flights
    for fi in range(num_flights):
        for idx in flight_option_indices[fi]:
            if x_vals[idx] > 0.5:
                assignments[fi] = all_options[idx]
                break

    # Check all flights assigned
    for fi in range(num_flights):
        if assignments[fi] is None:
            return None

    return assignments


def build_solution_dict(assignments, instance):
    """Build the output solution dictionary."""
    flights = instance['flights']
    carousels = instance['carousels']

    obj_val, violations = evaluate_objective(assignments, instance)

    assignment_list = []
    for fi, opt in enumerate(assignments):
        if opt is not None:
            assignment_list.append({
                'flight_id': flights[fi]['flight_id'],
                'carousel_id': opt['carousel_id'],
                'carousel_type': opt['carousel_type'],
                'working_stations': opt['W'],
                'handling_start_period': opt['handling_start'],
                'depletion_start_period': opt['depletion_start'],
                'end_period': opt['end_period']
            })

    solution = {
        'objective_value': float(obj_val),
        'assignments': assignment_list,
        'utilization_violations': violations
    }

    return solution


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    start_time = time.time()
    time_limit = args.time_limit

    # Read instance
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    num_flights = len(instance['flights'])
    num_carousels = len(instance['carousels'])
    num_periods = instance['planning_horizon']['num_periods']

    # Determine max options per flight based on problem size
    estimated_options_per_flight = num_carousels * 5 * 10 * 5  # rough estimate
    total_estimated = num_flights * estimated_options_per_flight

    max_opts = None
    if total_estimated > 100000:
        max_opts = max(50, 100000 // num_flights)

    # Enumerate options
    flight_options = enumerate_options(instance, max_options_per_flight=max_opts)

    # Check if all flights have options
    all_have_options = all(len(opts) > 0 for opts in flight_options)

    elapsed = time.time() - start_time
    remaining = time_limit - elapsed

    solution_dict = None

    # Try MIP if problem is not too large
    total_options = sum(len(opts) for opts in flight_options)

    if all_have_options and total_options <= 80000 and remaining > 10:
        try:
            mip_result = solve_mip(instance, flight_options, remaining - 5, logger)
            if mip_result is not None:
                solution_dict = build_solution_dict(mip_result, instance)
                if logger:
                    logger.log_solution(solution_dict['objective_value'], solution_dict)
        except Exception as e:
            pass

    # Fallback to heuristic
    if solution_dict is None and all_have_options:
        assignments = greedy_heuristic(instance, flight_options)

        # Check if all flights assigned
        if all(a is not None for a in assignments):
            # Try local search if time allows
            elapsed = time.time() - start_time
            remaining = time_limit - elapsed
            if remaining > 5:
                assignments, _ = local_search(
                    instance, flight_options, assignments,
                    max_iterations=200,
                    time_limit=remaining - 2
                )

            solution_dict = build_solution_dict(assignments, instance)
            if logger:
                logger.log_solution(solution_dict['objective_value'], solution_dict)

    # Last resort: assign first available option for each flight
    if solution_dict is None:
        assignments = []
        for fi in range(num_flights):
            if len(flight_options[fi]) > 0:
                assignments.append(flight_options[fi][0])
            else:
                assignments.append(None)

        # Fill unassigned with dummy
        for fi in range(num_flights):
            if assignments[fi] is None:
                # Create a minimal option
                flight = instance['flights'][fi]
                carousel = instance['carousels'][0]
                D = flight['handling_duration_periods']
                s_h = flight['earliest_start_period']
                s_d = s_h
                W = flight['Wmin']
                workload, storage = compute_profiles(
                    flight['arrival_profile'], D, W, s_h, s_d,
                    instance['working_rate_bags_per_period'],
                    instance['central_storage']['depletion_rate_bags_per_period'],
                    num_periods
                )
                assignments[fi] = {
                    'flight_idx': fi,
                    'carousel_idx': 0,
                    'carousel_id': carousel['carousel_id'],
                    'carousel_type': carousel['type_r'],
                    'W': W,
                    'handling_start': s_h,
                    'depletion_start': s_d,
                    'end_period': flight['end_period'],
                    'num_containers': flight['num_containers'],
                    'workload': workload,
                    'storage': storage,
                }

        solution_dict = build_solution_dict(assignments, instance)
        if logger:
            logger.log_solution(solution_dict['objective_value'], solution_dict)

    # Write solution
    with open(args.solution_path, 'w') as f:
        json.dump(solution_dict, f, indent=2)


if __name__ == '__main__':
    main()
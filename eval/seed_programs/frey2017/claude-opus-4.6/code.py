import argparse
import json
import math
import time
import sys
from collections import defaultdict

def solve(instance_path, solution_path, time_limit, log_path):
    from solution_logger import SolutionLogger
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    start_time = time.time()
    
    with open(instance_path, 'r') as f:
        data = json.load(f)
    
    num_periods = data['planning_horizon']['num_periods']
    target_util = data['objective']['target_utilization']
    threshold_values = data['objective']['threshold_values']
    penalties = data['objective']['penalties']
    num_thresholds = len(threshold_values)
    
    peak_intervals = data['peak_intervals']
    storage = data['central_storage']
    storage_cap = storage['capacity_bags']
    depletion_rate = storage['depletion_rate_bags_per_period']
    depletion_offset = storage['depletion_offset_periods']
    
    working_rate = data['working_rate_bags_per_period']
    
    carousels = data['carousels']
    flights = data['flights']
    carousel_types = data['carousel_types']
    
    num_flights = len(flights)
    num_carousels = len(carousels)
    
    # Precompute flight options: for each flight, enumerate (carousel, working_stations, handling_start, depletion_start)
    # and precompute workload and storage profiles
    
    def compute_workload_and_storage(flight, ws, hs, ds):
        """Compute workload on carousel per period and storage per period."""
        arrival_profile = flight['arrival_profile']
        handling_duration = flight['handling_duration_periods']
        end_period = flight['end_period']
        
        # Processing capacity per period
        proc_cap = ws * working_rate
        
        # Bags arrive according to arrival_profile starting at hs
        # Before hs, bags go to storage
        # At hs, handling begins - bags on belt processed at proc_cap rate
        # Storage depletion starts at ds at depletion_rate
        
        # Simulate period by period
        workload = {}  # period -> bags on belt
        storage_profile = {}  # period -> bags in storage
        
        belt_bags = 0.0
        stored_bags = 0.0
        
        earliest = flight['earliest_start_period']
        
        # Bags arrive from earliest_start_period for handling_duration periods
        # But handling starts at hs >= earliest
        
        for t in range(earliest, end_period + 1):
            # Bags arriving this period
            arr_idx = t - earliest
            arriving = 0
            if 0 <= arr_idx < handling_duration:
                arriving = arrival_profile[arr_idx]
            
            if t < hs:
                # Before handling starts, bags go to storage
                stored_bags += arriving
            else:
                # Handling active
                belt_bags += arriving
                
                # Storage depletion
                if t >= ds and stored_bags > 0:
                    depleted = min(stored_bags, depletion_rate)
                    stored_bags -= depleted
                    belt_bags += depleted
                
                # Processing removes bags from belt
                processed = min(belt_bags, proc_cap)
                belt_bags -= processed
            
            if belt_bags > 0:
                workload[t] = belt_bags
            if stored_bags > 0:
                storage_profile[t] = stored_bags
        
        return workload, storage_profile
    
    # Enumerate options for each flight
    flight_options = []  # list of lists of (carousel_id, carousel_type, ws, hs, ds, workload_dict, storage_dict)
    
    for fi, flight in enumerate(flights):
        options = []
        es = flight['earliest_start_period']
        ls = flight['latest_start_period']
        ep = flight['end_period']
        ws_bounds_by_type = flight['working_station_bounds_by_carousel_type']
        
        for ci, car in enumerate(carousels):
            ctype = str(car['type_r'])
            if ctype not in ws_bounds_by_type:
                continue
            bounds = ws_bounds_by_type[ctype]
            wmin = bounds['Wmin']
            wmax = bounds['Wmax']
            
            for ws in range(wmin, wmax + 1):
                for hs in range(es, ls + 1):
                    # Compute depletion start
                    # ds: depletion must start by (ep - depletion_offset) but not before hs
                    ds_latest = ep - depletion_offset
                    # Check if any bags are stored (arrive before hs)
                    # Count stored bags
                    arrival_profile = flight['arrival_profile']
                    handling_duration = flight['handling_duration_periods']
                    stored = 0
                    for t in range(es, min(hs, es + handling_duration)):
                        idx = t - es
                        if 0 <= idx < handling_duration:
                            stored += arrival_profile[idx]
                    
                    if stored == 0:
                        ds = hs  # No storage needed
                    else:
                        # Need enough periods to deplete storage
                        periods_needed = math.ceil(stored / depletion_rate)
                        ds_earliest = max(hs, ds_latest - periods_needed + 1)
                        # But ds must be >= hs
                        ds = max(hs, min(ds_latest, ds_earliest))
                    
                    # Validate
                    if ds < hs:
                        ds = hs
                    
                    workload, storage_prof = compute_workload_and_storage(flight, ws, hs, ds)
                    
                    options.append({
                        'carousel_id': car['carousel_id'],
                        'carousel_idx': ci,
                        'carousel_type': car['type_r'],
                        'ws': ws,
                        'hs': hs,
                        'ds': ds,
                        'workload': workload,
                        'storage': storage_prof,
                        'num_containers': flight['num_containers'],
                        'end_period': ep,
                    })
        
        flight_options.append(options)
    
    # Check if problem is small enough for MIP
    total_options = sum(len(opts) for opts in flight_options)
    
    import gurobipy as gp
    from gurobipy import GRB
    
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time < 5:
        remaining_time = 5
    
    # Build MIP
    model = gp.Model("baggage")
    model.setParam('TimeLimit', remaining_time - 2)
    model.setParam('Threads', 1)
    model.setParam('MIPFocus', 1)  # Focus on finding feasible solutions
    
    # Decision variables: x[f][o] = 1 if flight f uses option o
    x = {}
    for fi in range(num_flights):
        for oi in range(len(flight_options[fi])):
            x[fi, oi] = model.addVar(vtype=GRB.BINARY, name=f"x_{fi}_{oi}")
    
    # Each flight assigned exactly one option
    for fi in range(num_flights):
        model.addConstr(
            gp.quicksum(x[fi, oi] for oi in range(len(flight_options[fi]))) == 1,
            name=f"assign_{fi}"
        )
    
    # Precompute which options affect which carousel in which period
    # For working station constraints
    carousel_period_ws = defaultdict(list)  # (ci, t) -> [(fi, oi, ws)]
    carousel_period_containers = defaultdict(list)  # (ci, t) -> [(fi, oi, num_containers)]
    carousel_period_workload = defaultdict(list)  # (ci, t) -> [(fi, oi, workload)]
    period_storage = defaultdict(list)  # t -> [(fi, oi, storage)]
    
    for fi in range(num_flights):
        for oi, opt in enumerate(flight_options[fi]):
            ci = opt['carousel_idx']
            ws = opt['ws']
            nc = opt['num_containers']
            hs = opt['hs']
            ep = opt['end_period']
            
            # Working stations active from hs to ep
            for t in range(hs, ep + 1):
                carousel_period_ws[ci, t].append((fi, oi, ws))
            
            # Containers parked from hs to ep
            for t in range(hs, ep + 1):
                carousel_period_containers[ci, t].append((fi, oi, nc))
            
            # Workload
            for t, wl in opt['workload'].items():
                carousel_period_workload[ci, t].append((fi, oi, wl))
            
            # Storage
            for t, st in opt['storage'].items():
                period_storage[t].append((fi, oi, st))
    
    # Working station capacity constraints
    for (ci, t), entries in carousel_period_ws.items():
        car = carousels[ci]
        kws = car['Kws']
        model.addConstr(
            gp.quicksum(ws * x[fi, oi] for fi, oi, ws in entries) <= kws,
            name=f"ws_{ci}_{t}"
        )
    
    # Parking position capacity constraints
    for (ci, t), entries in carousel_period_containers.items():
        car = carousels[ci]
        kpp = car['Kpp']
        model.addConstr(
            gp.quicksum(nc * x[fi, oi] for fi, oi, nc in entries) <= kpp,
            name=f"pp_{ci}_{t}"
        )
    
    # Storage capacity constraints
    for t, entries in period_storage.items():
        model.addConstr(
            gp.quicksum(st * x[fi, oi] for fi, oi, st in entries) <= storage_cap,
            name=f"storage_{t}"
        )
    
    # Utilization penalty variables
    # For each carousel and peak interval, threshold exceeded indicators
    # y[ci, pi, k] = 1 if threshold k is exceeded
    y = {}
    for ci in range(num_carousels):
        for pi_data in peak_intervals:
            pi = pi_data['peak_id']
            for k in range(num_thresholds):
                y[ci, pi, k] = model.addVar(vtype=GRB.BINARY, name=f"y_{ci}_{pi}_{k}")
    
    # At most one threshold level flagged per carousel per peak interval
    for ci in range(num_carousels):
        for pi_data in peak_intervals:
            pi = pi_data['peak_id']
            model.addConstr(
                gp.quicksum(y[ci, pi, k] for k in range(num_thresholds)) <= 1,
                name=f"at_most_one_{ci}_{pi}"
            )
    
    # Utilization linking constraints
    # For each carousel, each period in a peak interval:
    # utilization - sum_k(threshold_values[k] * y[ci, pi, k]) <= target_util
    # utilization = total_workload / Kcb
    # => total_workload / Kcb - sum_k(threshold_values[k] * y[ci, pi, k]) <= target_util
    # => total_workload - Kcb * sum_k(threshold_values[k] * y[ci, pi, k]) <= target_util * Kcb
    
    for ci in range(num_carousels):
        car = carousels[ci]
        kcb = car['Kcb']
        for pi_data in peak_intervals:
            pi = pi_data['peak_id']
            sp = pi_data['start_period']
            ep_peak = pi_data['end_period']
            for t in range(sp, ep_peak + 1):
                entries = carousel_period_workload.get((ci, t), [])
                if not entries:
                    continue
                model.addConstr(
                    gp.quicksum(wl * x[fi, oi] for fi, oi, wl in entries) 
                    - kcb * gp.quicksum(threshold_values[k] * y[ci, pi, k] for k in range(num_thresholds))
                    <= target_util * kcb,
                    name=f"util_{ci}_{pi}_{t}"
                )
    
    # Objective: minimize total penalty
    model.setObjective(
        gp.quicksum(
            penalties[k] * y[ci, pi_data['peak_id'], k]
            for ci in range(num_carousels)
            for pi_data in peak_intervals
            for k in range(num_thresholds)
        ),
        GRB.MINIMIZE
    )
    
    # Callback for logging
    best_obj = [float('inf')]
    
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj < best_obj[0]:
                best_obj[0] = obj
                if logger:
                    logger.log(obj)
    
    model.optimize(callback)
    
    # Extract solution
    if model.SolCount == 0:
        # Return trivial solution - assign each flight to first option
        assignments = []
        for fi, flight in enumerate(flights):
            opt = flight_options[fi][0]
            assignments.append({
                "flight_id": flight['flight_id'],
                "carousel_id": opt['carousel_id'],
                "carousel_type": opt['carousel_type'],
                "working_stations": opt['ws'],
                "handling_start_period": opt['hs'],
                "depletion_start_period": opt['ds'],
                "end_period": opt['end_period'],
            })
        solution = {
            "objective_value": 0.0,
            "assignments": assignments,
            "utilization_violations": []
        }
    else:
        assignments = []
        for fi, flight in enumerate(flights):
            for oi in range(len(flight_options[fi])):
                if x[fi, oi].X > 0.5:
                    opt = flight_options[fi][oi]
                    assignments.append({
                        "flight_id": flight['flight_id'],
                        "carousel_id": opt['carousel_id'],
                        "carousel_type": opt['carousel_type'],
                        "working_stations": opt['ws'],
                        "handling_start_period": opt['hs'],
                        "depletion_start_period": opt['ds'],
                        "end_period": opt['end_period'],
                    })
                    break
        
        violations = []
        for ci in range(num_carousels):
            for pi_data in peak_intervals:
                pi = pi_data['peak_id']
                for k in range(num_thresholds):
                    if y[ci, pi, k].X > 0.5:
                        violations.append({
                            "carousel_id": carousels[ci]['carousel_id'],
                            "peak_interval": pi,
                            "threshold_level": k,
                            "threshold_value": threshold_values[k]
                        })
        
        solution = {
            "objective_value": model.ObjVal,
            "assignments": assignments,
            "utilization_violations": violations
        }
    
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)
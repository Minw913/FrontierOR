import argparse
import json
import math
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def main(instance_path, solution_path, time_limit, log_path=None):
    start_time = time.time()
    
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    
    planning_horizon = instance['planning_horizon']['num_periods']
    period_length_minutes = instance['planning_horizon']['period_length_minutes']
    central_storage = instance['central_storage']
    depletion_rate = central_storage['depletion_rate_bags_per_period']
    depletion_offset = central_storage['depletion_offset_periods']
    working_rate = instance['working_rate_bags_per_period']
    quantile_used = instance['quantile_used']
    num_flights = instance['num_flights']
    num_carousels = instance['num_carousels']
    carousel_config = instance['carousel_config']
    carousel_types = instance['carousel_types']
    carousels_list = instance['carousels']
    flights = instance['flights']
    peak_intervals = instance['peak_intervals']
    objective_params = instance['objective']
    target_utilization = objective_params['target_utilization']
    threshold_values = objective_params['threshold_values']
    penalties = objective_params['penalties']
    num_thresholds = len(threshold_values)
    baggage_transfer_time = instance.get('baggage_transfer_time_periods', 0)
    flight_end_offset = instance.get('flight_end_offset_periods', 0)
    
    carousels = {}
    for carousel in carousels_list:
        cid = carousel['carousel_id']
        carousels[cid] = carousel
    
    model = gp.Model('BaggageScheduling')
    model.setParam("Threads", 1)
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', time_limit)
    
    flight_vars = {}
    configs = {}
    storage_contrib = {}
    workload_contrib = {}
    
    for flight in flights:
        fid = flight['flight_id']
        configs_f = []
        base_period = flight['end_period'] - flight['handling_duration_periods'] + 1
        arrival_profile = flight['arrival_profile']
        total_bags = sum(arrival_profile)
        expected_bags = flight['expected_baggage']
        if total_bags != expected_bags:
            pass
        
        for carousel in carousels_list:
            cid = carousel['carousel_id']
            type_r = carousel['type_r']
            carousel_type = carousel_types[str(type_r)]
            Kpp = carousel_type['Kpp']
            Kws = carousel_type['Kws']
            Kppws = carousel_type['Kppws']
            Kcb = carousel_type['Kcb']
            
            if str(type_r) in flight['working_station_bounds_by_carousel_type']:
                bounds = flight['working_station_bounds_by_carousel_type'][str(type_r)]
                Wmin = bounds['Wmin']
                Wmax = bounds['Wmax']
            else:
                Wmin = flight['Wmin']
                Wmax = flight['Wmax']
            
            for w in range(Wmin, Wmax + 1):
                for s in range(flight['earliest_start_period'], flight['latest_start_period'] + 1):
                    for d in range(s, flight['end_period'] - depletion_offset + 1):
                        stored_bags = 0
                        if s > base_period:
                            stored_bags = sum(arrival_profile[0:s - base_period])
                        
                        T_deplete = 0
                        if stored_bags > 0:
                            T_deplete = math.ceil(stored_bags / depletion_rate)
                            if d + T_deplete > flight['end_period'] + 1:
                                continue
                        
                        config = (cid, w, s, d)
                        configs_f.append(config)
        
        if not configs_f:
            config = (carousels_list[0]['carousel_id'], flight['Wmin'], flight['earliest_start_period'], flight['earliest_start_period'])
            configs_f.append(config)
        
        configs[fid] = configs_f
        y_f = model.addVars(len(configs_f), vtype=GRB.BINARY, name=f"flight_{fid}")
        flight_vars[fid] = y_f
        model.addConstr(y_f.sum() == 1, name=f"assign_{fid}")
        
        for idx, config in enumerate(configs_f):
            cid, w, s, d = config
            base_period = flight['end_period'] - flight['handling_duration_periods'] + 1
            stored_bags = sum(arrival_profile[0:s - base_period]) if s > base_period else 0
            T_deplete = math.ceil(stored_bags / depletion_rate) if stored_bags > 0 else 0
            
            storage_vals = [0] * planning_horizon
            workload_vals = [0] * planning_horizon
            delivered = [0] * planning_horizon
            cumulative_processing = 0
            
            for t in range(planning_horizon):
                if t >= base_period and t < base_period + flight['handling_duration_periods']:
                    if t < s:
                        storage_vals[t] = arrival_profile[t - base_period]
                    else:
                        delivered[t] += arrival_profile[t - base_period]
                
                if stored_bags > 0 and t >= d and t < d + T_deplete:
                    if t == d + T_deplete - 1:
                        deplete = stored_bags - depletion_rate * (T_deplete - 1)
                    else:
                        deplete = depletion_rate
                    delivered[t] += deplete
            
            workload = 0
            for t in range(planning_horizon):
                if t < base_period:
                    storage_vals[t] = 0
                else:
                    if t < s:
                        if t == base_period:
                            storage_vals[t] = arrival_profile[0]
                        else:
                            storage_vals[t] = storage_vals[t-1] + (arrival_profile[t - base_period] if t < base_period + flight['handling_duration_periods'] and t < s else 0)
                    else:
                        storage_vals[t] = storage_vals[t-1]
                
                if stored_bags > 0 and t >= d:
                    if t < d + T_deplete:
                        deplete = depletion_rate if t < d + T_deplete - 1 else stored_bags - depletion_rate * (T_deplete - 1)
                        storage_vals[t] = max(0, storage_vals[t] - deplete)
                
                workload += delivered[t]
                processing = min(workload, w * working_rate)
                workload -= processing
                workload_vals[t] = workload
                
            storage_contrib[(fid, config)] = storage_vals
            workload_contrib[(fid, config)] = workload_vals
    
    for t in range(planning_horizon):
        storage_expr = 0
        for fid in flight_vars:
            for idx, config in enumerate(configs[fid]):
                storage_expr += storage_contrib[(fid, config)][t] * flight_vars[fid][idx]
        model.addConstr(storage_expr <= central_storage['capacity_bags'], name=f"storage_t{t}")
    
    for carousel in carousels_list:
        cid = carousel['carousel_id']
        type_r = carousel['type_r']
        carousel_type = carousel_types[str(type_r)]
        Kws = carousel_type['Kws']
        Kpp = carousel_type['Kpp']
        Kcb = carousel_type['Kcb']
        
        for t in range(planning_horizon):
            ws_expr = 0
            pp_expr = 0
            workload_expr = 0
            
            for fid in flight_vars:
                flight = next(f for f in flights if f['flight_id'] == fid)
                for idx, config in enumerate(configs[fid]):
                    if config[0] != cid:
                        continue
                    cid_conf, w, s, d = config
                    if s <= t <= flight['end_period']:
                        ws_expr += w * flight_vars[fid][idx]
                        pp_expr += flight['num_containers'] * flight_vars[fid][idx]
                    workload_expr += workload_contrib[(fid, config)][t] * flight_vars[fid][idx]
            
            model.addConstr(ws_expr <= Kws, name=f"ws_c{cid}_t{t}")
            model.addConstr(pp_expr <= Kpp, name=f"pp_c{cid}_t{t}")
            model.addConstr(workload_expr <= Kcb, name=f"cb_c{cid}_t{t}")
    
    I = model.addVars(num_carousels, len(peak_intervals), num_thresholds, vtype=GRB.BINARY, name="I")
    M = model.addVars(num_carousels, len(peak_intervals), vtype=GRB.CONTINUOUS, name="M")
    
    for c_idx, carousel in enumerate(carousels_list):
        cid = carousel['carousel_id']
        type_r = carousel['type_r']
        carousel_type = carousel_types[str(type_r)]
        Kcb = carousel_type['Kcb']
        
        for p_idx, peak in enumerate(peak_intervals):
            start_p = peak['start_period']
            end_p = peak['end_period']
            peak_id = peak['peak_id']
            model.addConstr(M[c_idx, p_idx] - target_utilization <= gp.quicksum(threshold_values[k] * I[c_idx, p_idx, k] for k in range(num_thresholds)), 
                           name=f"util_constr_c{cid}_p{peak_id}")
            model.addConstr(gp.quicksum(I[c_idx, p_idx, k] for k in range(num_thresholds)) <= 1, 
                           name=f"one_threshold_c{cid}_p{peak_id}")
            
            for t in range(start_p, end_p + 1):
                workload_expr = 0
                for fid in flight_vars:
                    for idx, config in enumerate(configs[fid]):
                        if config[0] != cid:
                            continue
                        workload_expr += workload_contrib[(fid, config)][t] * flight_vars[fid][idx]
                model.addConstr(M[c_idx, p_idx] >= (workload_expr / Kcb), 
                               name=f"max_util_c{cid}_p{peak_id}_t{t}")
    
    obj = gp.quicksum(I[c_idx, p_idx, k] * penalties[k] 
                      for c_idx in range(num_carousels)
                      for p_idx in range(len(peak_intervals))
                      for k in range(num_thresholds))
    model.setObjective(obj, GRB.MINIMIZE)
    
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj_val)
    
    model.optimize(callback)
    
    assignments = []
    utilization_violations = []
    objective_value = 0
    
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        objective_value = model.ObjVal
        for fid in flight_vars:
            flight = next(f for f in flights if f['flight_id'] == fid)
            y_f = flight_vars[fid]
            for idx, config in enumerate(configs[fid]):
                if y_f[idx].X > 0.5:
                    cid, w, s, d = config
                    carousel = carousels[cid]
                    assignment = {
                        'flight_id': fid,
                        'carousel_id': cid,
                        'carousel_type': carousel['type_r'],
                        'working_stations': w,
                        'handling_start_period': s,
                        'depletion_start_period': d,
                        'end_period': flight['end_period']
                    }
                    assignments.append(assignment)
                    break
        
        for c_idx in range(num_carousels):
            for p_idx in range(len(peak_intervals)):
                for k in range(num_thresholds):
                    if I[c_idx, p_idx, k].X > 0.5:
                        carousel_id = carousels_list[c_idx]['carousel_id']
                        peak_id = peak_intervals[p_idx]['peak_id']
                        threshold_val = threshold_values[k]
                        violation = {
                            'carousel_id': carousel_id,
                            'peak_interval': peak_id,
                            'threshold_level': k,
                            'threshold_value': threshold_val
                        }
                        utilization_violations.append(violation)
                        break
    
    solution = {
        'objective_value': objective_value,
        'assignments': assignments,
        'utilization_violations': utilization_violations
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
    main(args.instance_path, args.solution_path, args.time_limit, args.log_path)
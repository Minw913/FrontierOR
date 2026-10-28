import json
import argparse
import math
import random
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger

def get_profiles(flight, w, S_h, S_d, params):
    B = flight['expected_baggage']
    D = flight['handling_duration_periods']
    A = flight['arrival_profile']
    R_d = params['depletion_rate']
    R_w = params['working_rate']
    transfer = params['baggage_transfer_time']
    offset = params['depletion_offset']
    end = flight['end_period']
    T = params['num_periods']
    
    N_d = math.ceil(B / R_d) if R_d > 0 else 0
    N_w = math.ceil(B / (w * R_w)) if w * R_w > 0 else 0
    
    arr = [0] * T
    early_arrival = 0
    for k in range(D):
        t = S_h - D + k
        if 0 <= t < T:
            arr[t] += A[k]
        elif t < 0:
            early_arrival += A[k]
            
    dep = [0] * T
    for k in range(N_d):
        t = S_d + k
        if 0 <= t < T:
            dep[t] += min(R_d, B - R_d * k)
            
    level = early_arrival
    storage = [0] * T
    for t in range(T):
        level += arr[t] - dep[t]
        if level < 0: level = 0
        storage[t] = level
        
    inflow = [0] * T
    for k in range(N_d):
        t = S_d + transfer + k
        if 0 <= t < T:
            inflow[t] += min(R_d, B - R_d * k)
            
    outflow = [0] * T
    for k in range(N_w):
        t = S_h + k
        if 0 <= t < T:
            outflow[t] += min(w * R_w, B - w * R_w * k)
            
    level = 0
    workload = [0] * T
    for t in range(T):
        level += inflow[t] - outflow[t]
        if level < 0: level = 0
        workload[t] = level
        
    ws_usage = [0] * T
    for k in range(N_w):
        t = S_h + k
        if 0 <= t < T:
            ws_usage[t] = w
            
    cont_usage = [0] * T
    for k in range(N_w):
        t = S_h + k
        if 0 <= t < T:
            cont_usage[t] = flight['num_containers']
            
    return storage, workload, ws_usage, cont_usage

def solve():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path) as f:
        instance = json.load(f)

    flights = instance['flights']
    carousels = instance['carousels']
    peak_intervals = instance['peak_intervals']
    central_storage = instance['central_storage']
    
    T = instance['planning_horizon']['num_periods']
    R_d = central_storage['depletion_rate_bags_per_period']
    R_w = instance['working_rate_bags_per_period']
    transfer = instance['baggage_transfer_time_periods']
    offset = instance['central_storage']['depletion_offset_periods']
    
    params = {
        'num_periods': T,
        'depletion_rate': R_d,
        'working_rate': R_w,
        'baggage_transfer_time': transfer,
        'depletion_offset': offset
    }
    
    carousel_type_map = {c['carousel_id']: c['type_r'] for c in carousels}
    carousel_obj_map = {c['carousel_id']: c for c in carousels}
    
    configs = {}
    max_configs_per_flight = 2000
    random.seed(0)
    
    for f_idx, flight in enumerate(flights):
        f_configs = []
        B = flight['expected_baggage']
        N_d = math.ceil(B / R_d) if R_d > 0 else 0
        
        earliest = flight['earliest_start_period']
        latest = min(flight['latest_start_period'], flight['end_period'] - offset - N_d)
        if latest < earliest:
            latest = earliest
            
        step = 1
        if latest - earliest > 20:
            step = (latest - earliest) // 10
            if step == 0: step = 1
            
        for S_h in range(earliest, latest + 1, step):
            S_d_options = [S_h]
            max_S_d = flight['end_period'] - offset - N_d
            if max_S_d > S_h:
                S_d_options.append(max_S_d)
                if max_S_d > S_h + 1:
                    S_d_options.append(S_h + (max_S_d - S_h)//2)
                    
            for c in carousels:
                c_type = c['type_r']
                if str(c_type) in flight['working_station_bounds_by_carousel_type']:
                    bounds = flight['working_station_bounds_by_carousel_type'][str(c_type)]
                    Wmin, Wmax = bounds['Wmin'], bounds['Wmax']
                    for w in range(Wmin, Wmax + 1):
                        for S_d in S_d_options:
                            f_configs.append((c['carousel_id'], w, S_h, S_d))
                            
        if len(f_configs) > max_configs_per_flight:
            f_configs = random.sample(f_configs, max_configs_per_flight)
        configs[f_idx] = f_configs

    model = Model("BaggageHandling")
    model.setParam("Seed", 0)
    model.setParam("MIPGap", 1e-4)
    model.setParam("NumericFocus", 0)
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", args.time_limit)
    
    x = {}
    for f_idx in configs:
        for c_idx in range(len(configs[f_idx])):
            x[f_idx, c_idx] = model.addVar(vtype=GRB.BINARY, name=f"x_{f_idx}_{c_idx}")
            
    target = instance['objective']['target_utilization']
    threshold_values = instance['objective']['threshold_values']
    penalties = instance['objective']['penalties']
    num_thresholds = len(threshold_values)
    
    y = {}
    for c in carousels:
        for p in peak_intervals:
            for k in range(num_thresholds):
                y[c['carousel_id'], p['peak_id'], k] = model.addVar(vtype=GRB.BINARY, name=f"y_{c['carousel_id']}_{p['peak_id']}_{k}")
                
    model.update()
    
    # 1. Each flight gets exactly one config
    for f_idx in configs:
        model.addConstr(quicksum(x[f_idx, c_idx] for c_idx in range(len(configs[f_idx]))) == 1)
        
    # Aggregate usage
    carousel_ws = {c['carousel_id']: [0]*T for c in carousels}
    carousel_cont = {c['carousel_id']: [0]*T for c in carousels}
    carousel_workload = {c['carousel_id']: [0]*T for c in carousels}
    storage_level = [0]*T
    
    for f_idx, f_configs in configs.items():
        flight = flights[f_idx]
        for c_idx, cfg in enumerate(f_configs):
            c_id, w, S_h, S_d = cfg
            storage, workload, ws_usage, cont_usage = get_profiles(flight, w, S_h, S_d, params)
            for t in range(T):
                if ws_usage[t] > 0:
                    carousel_ws[c_id][t] += ws_usage[t] * x[f_idx, c_idx]
                if cont_usage[t] > 0:
                    carousel_cont[c_id][t] += cont_usage[t] * x[f_idx, c_idx]
                if workload[t] > 0:
                    carousel_workload[c_id][t] += workload[t] * x[f_idx, c_idx]
                if storage[t] > 0:
                    storage_level[t] += storage[t] * x[f_idx, c_idx]
                    
    # 2. Carousel capacities
    for c in carousels:
        c_id = c['carousel_id']
        Kws = c['Kws']
        Kpp = c['Kpp']
        for t in range(T):
            if carousel_ws[c_id][t] != 0:
                model.addConstr(carousel_ws[c_id][t] <= Kws)
            if carousel_cont[c_id][t] != 0:
                model.addConstr(carousel_cont[c_id][t] <= Kpp)
                
    # 3. Storage capacity
    cap = central_storage['capacity_bags']
    for t in range(T):
        if storage_level[t] != 0:
            model.addConstr(storage_level[t] <= cap)
            
    # 4. Utilization constraints
    for c in carousels:
        c_id = c['carousel_id']
        Kcb = c['Kcb']
        if Kcb == 0: continue
        for p in peak_intervals:
            for t in range(p['start_period'], p['end_period'] + 1):
                if t >= T: continue
                if carousel_workload[c_id][t] != 0:
                    util = carousel_workload[c_id][t] / Kcb
                    model.addConstr(util - quicksum(threshold_values[k] * y[c_id, p['peak_id'], k] for k in range(num_thresholds)) <= target)
            model.addConstr(quicksum(y[c_id, p['peak_id'], k] for k in range(num_thresholds)) <= 1)
            
    model.setObjective(quicksum(penalties[k] * y[c['carousel_id'], p['peak_id'], k] 
                               for c in carousels 
                               for p in peak_intervals 
                               for k in range(num_thresholds)), GRB.MINIMIZE)
    
    def log_callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj)
                
    if logger:
        model.optimize(log_callback)
    else:
        model.optimize()
        
    if model.SolCount == 0:
        # Fallback if no solution found
        sol = {
            "objective_value": 0.0,
            "assignments": [],
            "utilization_violations": []
        }
    else:
        assignments = []
        for f_idx in configs:
            for c_idx in range(len(configs[f_idx])):
                if x[f_idx, c_idx].X > 0.5:
                    c_id, w, S_h, S_d = configs[f_idx][c_idx]
                    assignments.append({
                        "flight_id": flights[f_idx]['flight_id'],
                        "carousel_id": c_id,
                        "carousel_type": carousel_type_map[c_id],
                        "working_stations": w,
                        "handling_start_period": S_h,
                        "depletion_start_period": S_d,
                        "end_period": flights[f_idx]['end_period']
                    })
                    break
                    
        utilization_violations = []
        for c in carousels:
            c_id = c['carousel_id']
            for p in peak_intervals:
                for k in range(num_thresholds):
                    if y[c_id, p['peak_id'], k].X > 0.5:
                        utilization_violations.append({
                            "carousel_id": c_id,
                            "peak_interval": p['peak_id'],
                            "threshold_level": k,
                            "threshold_value": threshold_values[k]
                        })
                        
        sol = {
            "objective_value": model.ObjVal,
            "assignments": assignments,
            "utilization_violations": utilization_violations
        }
        
    if logger:
        logger.log_solution(sol["objective_value"], sol)
        
    with open(args.solution_path, 'w') as f:
        json.dump(sol, f, indent=2)

if __name__ == "__main__":
    solve()
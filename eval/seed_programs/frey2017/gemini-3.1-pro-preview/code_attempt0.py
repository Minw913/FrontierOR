import json
import argparse
import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None

class OptimizationCallback:
    def __init__(self, logger):
        self.logger = logger
    
    def __call__(self, model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if self.logger:
                self.logger.log(obj)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    # Initialize Logger
    logger = None
    if SolutionLogger is not None and args.log_path:
        logger = SolutionLogger(args.log_path, sense="minimize")

    # Load Instance Data
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    num_periods = instance['planning_horizon']['num_periods']
    flights = instance['flights']
    carousels = instance['carousels']
    peak_intervals = instance['peak_intervals']
    target_util = instance['objective']['target_utilization']
    threshold_vals = instance['objective']['threshold_values']
    penalties = instance['objective']['penalties']
    
    transf_time = instance['baggage_transfer_time_periods']
    w_rate = instance['working_rate_bags_per_period']
    dep_rate = instance['central_storage']['depletion_rate_bags_per_period']
    offset_periods = instance['central_storage']['depletion_offset_periods']
    storage_cap = instance['central_storage']['capacity_bags']

    # Pre-generate Valid Working Configurations for Each Flight
    valid_configs = {}
    for f in flights:
        f_id = f['flight_id']
        valid_configs[f_id] = {}
        w_bounds_dict = f['working_station_bounds_by_carousel_type']
        
        t_start = f['end_period'] - f['handling_duration_periods']
        max_td_base = f['end_period'] - offset_periods
        arrival = f['arrival_profile']
        
        for type_id_str, bounds in w_bounds_dict.items():
            t_type = int(type_id_str)
            w_min = bounds['Wmin']
            w_max = bounds['Wmax']
            
            configs = []
            for w in range(w_min, w_max + 1):
                for t_h in range(f['earliest_start_period'], f['latest_start_period'] + 1):
                    # Compute initially stored bags before handling begins
                    SB = 0
                    for t in range(t_start, t_h):
                        if 0 <= t - t_start < len(arrival):
                            SB += arrival[t - t_start]
                            
                    # Narrow down depletion start options (t_d doesn't matter much if SB == 0)
                    if SB == 0:
                        td_range = [t_h] if t_h <= max_td_base else []
                    else:
                        td_range = range(t_h, max_td_base + 1)

                    for t_d in td_range:
                        deplete_amt = {}
                        S_rem = SB
                        possible = True
                        for t in range(t_d, f['end_period']):
                            if S_rem > 0:
                                amt = min(S_rem, dep_rate)
                                deplete_amt[t] = amt
                                S_rem -= amt
                                if t + transf_time >= f['end_period']:
                                    possible = False
                                    break
                            else:
                                break
                        
                        if S_rem > 0 or not possible:
                            continue
                            
                        W_val = 0
                        W_bags = {}
                        S_bags = {}
                        late_bags = False
                        
                        # Full duration tracking for central storage volume
                        S_vol = 0
                        for t in range(min(0, t_start), num_periods):
                            arr = arrival[t - t_start] if 0 <= t - t_start < len(arrival) else 0
                            if t < t_h:
                                S_vol += arr
                            if t in deplete_amt:
                                S_vol -= deplete_amt[t]
                            if 0 <= t < num_periods:
                                S_bags[t] = max(0, S_vol)
                                
                        # Carousel queue dynamic verification
                        for t in range(min(t_start, t_h), f['end_period'] + 1):
                            arr = arrival[t - t_start] if (t >= t_h and 0 <= t - t_start < len(arrival)) else 0
                            dep_arr = deplete_amt.get(t - transf_time, 0)
                            entry = arr + dep_arr
                            
                            # Standard Workload flow
                            W_val = max(0, W_val + entry - w * w_rate)
                            
                            if t == f['end_period'] and W_val > 0:
                                late_bags = True
                                
                            if 0 <= t < num_periods:
                                W_bags[t] = W_val
                                
                        if not late_bags:
                            configs.append({
                                'w': w, 't_h': t_h, 't_d': t_d, 'W': W_bags, 'S': S_bags
                            })
            
            if configs:
                valid_configs[f_id][t_type] = configs

    # Initialize Gurobi Model
    m = gp.Model("BaggageHandlingOptimization")
    if args.time_limit:
        m.setParam('TimeLimit', args.time_limit)
    m.setParam('Threads', 1)
    # Ensure silent solving output as to adhere gracefully internally
    m.setParam('OutputFlag', 1) 

    # Building mapped structures for expression caching
    storage_exprs = {t: [] for t in range(num_periods)}
    ws_exprs = {c['carousel_id']: {t: [] for t in range(num_periods)} for c in carousels}
    pp_exprs = {c['carousel_id']: {t: [] for t in range(num_periods)} for c in carousels}
    util_exprs = {c['carousel_id']: {t: [] for t in range(num_periods)} for c in carousels}

    # Configuration Binaries Setup
    x = {}
    for f in flights:
        f_id = f['flight_id']
        for c in carousels:
            c_id = c['carousel_id']
            c_type = c['type_r']
            if c_type in valid_configs[f_id]:
                for k, conf in enumerate(valid_configs[f_id][c_type]):
                    v = m.addVar(vtype=GRB.BINARY, name=f"x_{f_id}_{c_id}_{k}")
                    x[f_id, c_id, k] = v
                    
                    for t, S_val in conf['S'].items():
                        if S_val > 0:
                            storage_exprs[t].append((S_val, v))
                    
                    for t in range(conf['t_h'], f['end_period']):
                        if 0 <= t < num_periods:
                            ws_exprs[c_id][t].append((conf['w'], v))
                            pp_exprs[c_id][t].append((f['num_containers'], v))
                            
                    for t, W_val in conf['W'].items():
                        if W_val > 0:
                            util_exprs[c_id][t].append((W_val / c['Kcb'], v))
                            
    # Assignment Constraint: Exactly ONE combination per flight total
    for f in flights:
        f_id = f['flight_id']
        vars_list = [
            x[f_id, c['carousel_id'], k] 
            for c in carousels if c['type_r'] in valid_configs.get(f_id, {})
            for k in range(len(valid_configs[f_id][c['type_r']]))
        ]
        m.addConstr(gp.quicksum(vars_list) == 1, name=f"assign_{f_id}")

    # Central Storage Limits
    for t in range(num_periods):
        if storage_exprs[t]:
            m.addConstr(gp.quicksum(coeff * v for coeff, v in storage_exprs[t]) <= storage_cap, name=f"str_{t}")
            
    # Carousel Capabilities Bounds
    for c in carousels:
        c_id = c['carousel_id']
        for t in range(num_periods):
            if ws_exprs[c_id][t]:
                m.addConstr(gp.quicksum(coeff * v for coeff, v in ws_exprs[c_id][t]) <= c['Kws'], name=f"ws_{c_id}_{t}")
            if pp_exprs[c_id][t]:
                m.addConstr(gp.quicksum(coeff * v for coeff, v in pp_exprs[c_id][t]) <= c['Kpp'], name=f"pp_{c_id}_{t}")

    # Peak Intervals Limits and Utilization Over-usage Modeling
    y = {}
    for c in carousels:
        c_id = c['carousel_id']
        for p in peak_intervals:
            p_id = p['peak_id']
            for l in range(len(threshold_vals)):
                y[c_id, p_id, l] = m.addVar(vtype=GRB.BINARY, name=f"y_{c_id}_{p_id}_{l}")
            
            # Constraint: At most one threshold indexed violation
            m.addConstr(gp.quicksum(y[c_id, p_id, l] for l in range(len(threshold_vals))) <= 1)
            
            for t in range(p['start_period'], p['end_period'] + 1):
                if 0 <= t < num_periods:
                    terms = util_exprs[c_id][t]
                    lhs = gp.quicksum(coeff * v for coeff, v in terms) if terms else 0
                    rhs = target_util + gp.quicksum(threshold_vals[l] * y[c_id, p_id, l] for l in range(len(threshold_vals)))
                    m.addConstr(lhs <= rhs, name=f"util_{c_id}_{p_id}_{t}")

    # Soft Symmetry Breaking Constraints across same-type carousels
    type_groups = {}
    for c in carousels:
        type_groups.setdefault(c['type_r'], []).append(c['carousel_id'])
        
    for t_r, c_ids in type_groups.items():
        if len(c_ids) > 1:
            for i in range(len(c_ids) - 1):
                c1, c2 = c_ids[i], c_ids[i+1]
                sum1 = gp.quicksum(x[f['flight_id'], c1, k] 
                                   for f in flights if t_r in valid_configs[f['flight_id']]
                                   for k in range(len(valid_configs[f['flight_id']][t_r])))
                sum2 = gp.quicksum(x[f['flight_id'], c2, k] 
                                   for f in flights if t_r in valid_configs[f['flight_id']]
                                   for k in range(len(valid_configs[f['flight_id']][t_r])))
                m.addConstr(sum1 >= sum2, name=f"sym_{t_r}_{i}")

    # Set Penalty Minimizing Target Evaluation
    obj = gp.quicksum(penalties[l] * y[c['carousel_id'], p['peak_id'], l]
                      for c in carousels 
                      for p in peak_intervals 
                      for l in range(len(threshold_vals)))
    m.setObjective(obj, GRB.MINIMIZE)

    # Solve Implementation Process
    cb = OptimizationCallback(logger)
    m.optimize(cb)

    # Structuring Output Result Format
    assignments = []
    utilization_violations = []
    
    if m.SolCount > 0:
        obj_value = m.ObjVal
        for f in flights:
            f_id = f['flight_id']
            for c in carousels:
                c_id = c['carousel_id']
                c_type = c['type_r']
                if c_type in valid_configs[f_id]:
                    for k in range(len(valid_configs[f_id][c_type])):
                        if x[f_id, c_id, k].X > 0.5:
                            conf = valid_configs[f_id][c_type][k]
                            assignments.append({
                                "flight_id": f_id,
                                "carousel_id": c_id,
                                "carousel_type": c_type,
                                "working_stations": conf['w'],
                                "handling_start_period": conf['t_h'],
                                "depletion_start_period": conf['t_d'],
                                "end_period": f['end_period']
                            })
                            break
                            
        for c in carousels:
            c_id = c['carousel_id']
            for p in peak_intervals:
                p_id = p['peak_id']
                for l in range(len(threshold_vals)):
                    if y[c_id, p_id, l].X > 0.5:
                        utilization_violations.append({
                            "carousel_id": c_id,
                            "peak_interval": p_id,
                            "threshold_level": l,
                            "threshold_value": threshold_vals[l]
                        })
                        break
    else:
        obj_value = 0.0

    solution = {
        "objective_value": obj_value,
        "assignments": assignments,
        "utilization_violations": utilization_violations
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()
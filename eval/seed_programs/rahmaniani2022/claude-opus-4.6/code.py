import argparse
import json
import time
import sys
from collections import defaultdict
import gurobipy as gp
from gurobipy import GRB

def solve(instance_path, solution_path, time_limit, log_path):
    from solution_logger import SolutionLogger
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None

    start_time = time.time()

    with open(instance_path, 'r') as f:
        data = json.load(f)

    pp = data['problem_parameters']
    num_zones = pp['num_zones']
    num_weeks = pp['num_weeks']
    H = pp['hours_per_subperiod']  # 168
    total_hours = pp['total_hours']
    
    policy = data['policy']
    policy_type = policy['type']
    
    subperiod_weights = data['subperiod_weights']
    generators = data['generators']
    gen_params = data['generator_parameters']
    existing_cap = data['existing_capacity']
    demand = data['demand']
    consumer_segments = data['consumer_segments']
    trans_lines = data['transmission_lines']
    availability = data['availability']
    hydro_inflow = data.get('hydro_inflow', {})
    
    # Compute hourly weights
    hourly_weights = []
    for w in range(num_weeks):
        wt = subperiod_weights[w] / H
        for h in range(H):
            hourly_weights.append(wt)
    
    # Index generators
    gen_by_id = {}
    for g in generators:
        gen_by_id[g['cluster_id']] = g
    
    gen_ids = [g['cluster_id'] for g in generators]
    
    # Get parameters for each generator
    def get_params(g):
        rt = g['resource_type']
        if rt in gen_params:
            return gen_params[rt]
        return {}
    
    # Zone-based indexing
    gens_in_zone = defaultdict(list)
    for g in generators:
        gens_in_zone[g['zone']].append(g['cluster_id'])
    
    # Consumer segments by zone
    segs_in_zone = defaultdict(list)
    for seg in consumer_segments:
        segs_in_zone[seg['zone']].append(seg)
    
    # Lines by zone
    lines_from_zone = defaultdict(list)
    lines_to_zone = defaultdict(list)
    for line in trans_lines:
        lines_from_zone[line['from_zone']].append(line)
        lines_to_zone[line['to_zone']].append(line)
    
    # Subperiod hour ranges
    subperiod_hours = []
    for w in range(num_weeks):
        start_h = w * H
        subperiod_hours.append(list(range(start_h, start_h + H)))
    
    # Map hour to subperiod
    hour_to_subperiod = {}
    for w in range(num_weeks):
        for t in subperiod_hours[w]:
            hour_to_subperiod[t] = w
    
    # Previous hour within subperiod (wrap-around)
    prev_hour = {}
    for w in range(num_weeks):
        hrs = subperiod_hours[w]
        prev_hour[hrs[0]] = hrs[-1]
        for idx in range(1, len(hrs)):
            prev_hour[hrs[idx]] = hrs[idx - 1]
    
    # Build model
    model = gp.Model("capacity_expansion")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = max(time_limit - 30, 10)  # Reserve time for output
    
    # ========== INVESTMENT VARIABLES ==========
    y_P_new = {}  # new investment count
    y_P_ret = {}  # retirement count  
    y_P = {}      # total installed capacity MW
    y_E_new = {}  # storage energy new
    y_E = {}      # total energy capacity
    
    for g in generators:
        i = g['cluster_id']
        rt = g['resource_type']
        params = get_params(g)
        cap_size = g['capacity_size_MW']
        exist_cap = existing_cap.get(str(i), 0.0)
        is_UC = g['is_UC']
        
        if is_UC:
            inv_block = cap_size
        else:
            inv_block = 1.0  # 1 MW increments
        
        # Max new capacity
        max_new_cap = params.get('max_capacity_MW', 0.0)
        max_new_units = int(max_new_cap / inv_block) if inv_block > 0 else 0
        
        y_P_new[i] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=max_new_units, name=f"y_P_new[{i}]")
        
        # Retirement
        if g['can_retire']:
            if is_UC:
                ret_block = cap_size
                max_ret = int(exist_cap / ret_block) if ret_block > 0 and exist_cap > 0 else 0
            else:
                ret_block = 1.0
                max_ret = int(exist_cap / ret_block) if exist_cap > 0 else 0
            y_P_ret[i] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=max_ret, name=f"y_P_ret[{i}]")
        else:
            y_P_ret[i] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=0, name=f"y_P_ret[{i}]")
        
        # Total capacity
        y_P[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"y_P[{i}]")
        model.addConstr(y_P[i] == exist_cap + inv_block * y_P_new[i] - inv_block * y_P_ret[i], name=f"cap_bal_{i}")
        
        # Storage energy
        if g['is_storage']:
            bp = params
            if is_UC:
                e_block = bp.get('energy_capacity_size_MWh', 1.0)
            else:
                e_block = 1.0
            
            max_new_e = bp.get('max_energy_capacity_MWh', 0.0)
            max_new_e_units = int(max_new_e / e_block) if e_block > 0 else 0
            
            # Existing energy capacity: derive from existing power capacity and some default duration
            # For storage, existing energy = existing_cap * some ratio. 
            # Actually, let's check: existing energy capacity isn't explicitly given.
            # The existing capacity is power capacity. Energy capacity for existing storage
            # needs to be determined. Looking at the schema, existing_capacity is power capacity.
            # For existing storage, the energy capacity should be existing_power * some duration.
            # But duration is variable between min and max. We need to handle this.
            # Actually, for existing storage, the energy is also a decision variable.
            # Let me re-read: "The final installed power and energy capacities equal pre-existing capacity plus their explicitly declared increment times net investment."
            # This suggests there's a pre-existing energy capacity too. But the schema only has existing_capacity for power.
            # For storage: existing energy = existing_power * max_duration (or we derive it from cluster size)
            # Actually for battery: existing units * energy_capacity_size_MWh
            exist_power_units = exist_cap / cap_size if cap_size > 0 else 0
            exist_e_cap = exist_power_units * bp.get('energy_capacity_size_MWh', 0.0)
            
            y_E_new[i] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=max_new_e_units, name=f"y_E_new[{i}]")
            y_E[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"y_E[{i}]")
            model.addConstr(y_E[i] == exist_e_cap + e_block * y_E_new[i], name=f"ecap_bal_{i}")
            
            # Duration constraints
            min_dur = bp.get('min_duration_MWh_per_MW', 0.0)
            max_dur = bp.get('max_duration_MWh_per_MW', float('inf'))
            model.addConstr(y_E[i] >= min_dur * y_P[i], name=f"min_dur_{i}")
            if max_dur < float('inf'):
                model.addConstr(y_E[i] <= max_dur * y_P[i], name=f"max_dur_{i}")
    
    # Transmission variables
    y_F_new = {}
    y_F = {}
    for line in trans_lines:
        l = line['line_id']
        exist_f = line['existing_capacity_MW']
        max_f = line['max_capacity_MW']
        max_new = max_f - exist_f
        max_new_int = int(max_new)
        
        y_F_new[l] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=max_new_int, name=f"y_F_new[{l}]")
        y_F[l] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"y_F[{l}]")
        model.addConstr(y_F[l] == exist_f + y_F_new[l], name=f"trans_bal_{l}")
    
    # ========== OPERATIONAL VARIABLES ==========
    x_inj = {}   # generation/injection
    x_wdw = {}   # storage withdrawal (charging)
    x_soc = {}   # state of charge
    x_nse = {}   # non-served energy
    x_lvl = {}   # hydro reservoir level
    x_spill = {} # hydro spillage
    x_flow = {}  # transmission flow
    x_commit = {}
    x_start = {}
    x_shut = {}
    
    # Create operational variables
    for g in generators:
        i = g['cluster_id']
        for t in range(total_hours):
            x_inj[i, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"x_inj[{i},{t}]")
            
            if g['is_storage']:
                x_wdw[i, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"x_wdw[{i},{t}]")
                x_soc[i, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"x_soc[{i},{t}]")
            
            if g['is_hydro']:
                x_lvl[i, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"x_lvl[{i},{t}]")
                x_spill[i, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"x_spill[{i},{t}]")
            
            if g['is_UC']:
                x_commit[i, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"x_commit[{i},{t}]")
                x_start[i, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"x_start[{i},{t}]")
                x_shut[i, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"x_shut[{i},{t}]")
    
    for seg in consumer_segments:
        s = seg['segment_id']
        z = seg['zone']
        for t in range(total_hours):
            x_nse[s, z, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"x_nse[{s},{z},{t}]")
    
    for line in trans_lines:
        l = line['line_id']
        for t in range(total_hours):
            x_flow[l, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name=f"x_flow[{l},{t}]")
    
    # Policy slack variables
    x_RPS = {}
    x_CO2 = {}
    for w in range(num_weeks):
        x_RPS[w] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"x_RPS[{w}]")
        x_CO2[w] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"x_CO2[{w}]")
    
    model.update()
    
    # ========== CONSTRAINTS ==========
    
    # --- UC constraints ---
    for g in generators:
        i = g['cluster_id']
        params = get_params(g)
        cap_size = g['capacity_size_MW']
        
        if g['is_UC']:
            min_out = params.get('min_output_frac', 0.0)
            avail = availability.get(str(i), [1.0] * total_hours)
            ramp_up = params.get('ramp_up_rate_pct_per_hr', 1.0)
            ramp_down = params.get('ramp_down_rate_pct_per_hr', 1.0)
            min_up = params.get('min_up_time_hr', 1)
            min_down = params.get('min_down_time_hr', 1)
            
            for t in range(total_hours):
                tp = prev_hour[t]
                a_t = avail[t]
                
                # Commit/start/shut bounded by capacity
                model.addConstr(x_commit[i, t] * cap_size <= y_P[i], name=f"uc_cap_c_{i}_{t}")
                model.addConstr(x_start[i, t] * cap_size <= y_P[i], name=f"uc_cap_s_{i}_{t}")
                model.addConstr(x_shut[i, t] * cap_size <= y_P[i], name=f"uc_cap_d_{i}_{t}")
                
                # Commitment balance
                model.addConstr(x_commit[i, t] - x_commit[i, tp] == x_start[i, t] - x_shut[i, t],
                                name=f"uc_bal_{i}_{t}")
                
                # Generation bounds
                model.addConstr(x_inj[i, t] >= x_commit[i, t] * cap_size * min_out,
                                name=f"uc_min_{i}_{t}")
                model.addConstr(x_inj[i, t] <= x_commit[i, t] * cap_size * a_t,
                                name=f"uc_max_{i}_{t}")
                
                # Ramping for UC
                # Ramp up: x_inj[t] - x_inj[tp] <= ramp_up * cap_size * (x_commit[tp] - x_start[t]) 
                #   + cap_size * min(a_t, max(min_out, ramp_up)) * x_start[t]
                #   - cap_size * min_out * x_shut[t]
                # But we need x_commit at tp, not excluding x_start at t. Let me re-read.
                # "the increase in output is limited by the ramp-up rate times the unit size times
                #  the number of units that were already committed (excluding those just starting)"
                # Already committed excluding starting = x_commit[tp] (since commit balance wraps)
                # Actually: units already committed = x_commit[t] - x_start[t] 
                # (units that were committed at t but not starting at t = units that were on from before)
                # Wait, re-read: "the number of units that were already committed (excluding those just starting)"
                # This means x_commit[t] - x_start[t].
                
                ramp_up_lim = ramp_up * cap_size * (x_commit[i, t] - x_start[i, t]) \
                    + cap_size * min(a_t, max(min_out, ramp_up)) * x_start[i, t] \
                    - cap_size * min_out * x_shut[i, t]
                
                model.addConstr(x_inj[i, t] - x_inj[i, tp] <= ramp_up_lim,
                                name=f"uc_ramp_up_{i}_{t}")
                
                # Ramp down
                a_tp = avail[tp]
                ramp_down_lim = ramp_down * cap_size * (x_commit[i, tp] - x_start[i, t]) \
                    + cap_size * min(a_tp, max(min_out, ramp_down)) * x_shut[i, t] \
                    - cap_size * min_out * x_start[i, t]
                
                # Wait, let me re-read: "An analogous expression governs ramp-down"
                # ramp down: x_inj[tp] - x_inj[t] <= ramp_down * cap_size * (x_commit[t] - x_shut[... wait
                # Let me think more carefully. The "analogous" for ramp down should mirror the up:
                # x_inj[tp] - x_inj[t] <= ramp_down * cap_size * (x_commit[t] - ... )
                # Actually the standard formulation:
                # Ramp down: x_inj[tp] - x_inj[t] <= ramp_down * cap_size * (x_commit[t] - x_start[t])
                #   + cap_size * min(a_tp, max(min_out, ramp_down)) * x_shut[i,t]  
                #   - cap_size * min_out * x_start[i,t]
                
                ramp_down_lim2 = ramp_down * cap_size * (x_commit[i, t] - x_start[i, t]) \
                    + cap_size * min(a_tp, max(min_out, ramp_down)) * x_shut[i, t] \
                    - cap_size * min_out * x_start[i, t]
                
                model.addConstr(x_inj[i, tp] - x_inj[i, t] <= ramp_down_lim2,
                                name=f"uc_ramp_down_{i}_{t}")
            
            # Min up/down time with wrap-around within subperiod
            for w in range(num_weeks):
                hrs = subperiod_hours[w]
                n_hrs = len(hrs)
                for idx in range(n_hrs):
                    t = hrs[idx]
                    
                    # Min up time: x_commit[t] >= sum of x_start over preceding min_up hours
                    start_sum = gp.LinExpr()
                    for k in range(min_up):
                        tk = hrs[(idx - k) % n_hrs]
                        start_sum += x_start[i, tk]
                    model.addConstr(x_commit[i, t] >= start_sum, name=f"min_up_{i}_{t}")
                    
                    # Min down time: x_commit[t] + sum of x_shut over preceding min_down hours <= max_units
                    shut_sum = gp.LinExpr()
                    for k in range(min_down):
                        tk = hrs[(idx - k) % n_hrs]
                        shut_sum += x_shut[i, tk]
                    # max_units = y_P[i] / cap_size ... but y_P is variable
                    # We need: x_commit[t] + shut_sum <= y_P[i] / cap_size
                    model.addConstr(x_commit[i, t] * cap_size + shut_sum * cap_size <= y_P[i],
                                    name=f"min_down_{i}_{t}")
        
        elif g['is_storage']:
            # Storage operational constraints
            bp = params
            ch_eff = bp.get('charging_efficiency', 1.0)
            dis_eff = bp.get('discharging_efficiency', 1.0)
            self_dis = bp.get('self_discharge_rate_pct', 0.0)
            avail = availability.get(str(i), [1.0] * total_hours)
            
            for t in range(total_hours):
                tp = prev_hour[t]
                a_t = avail[t]
                
                # SOC balance
                model.addConstr(
                    x_soc[i, t] == (1 - self_dis) * x_soc[i, tp] 
                    + ch_eff * x_wdw[i, t] 
                    - x_inj[i, t] / dis_eff,
                    name=f"soc_bal_{i}_{t}")
                
                # SOC <= energy capacity
                model.addConstr(x_soc[i, t] <= y_E[i], name=f"soc_max_{i}_{t}")
                
                # Charging cannot exceed remaining room
                model.addConstr(ch_eff * x_wdw[i, t] <= y_E[i] - x_soc[i, t], name=f"charge_room_{i}_{t}")
                
                # Discharging cannot exceed SOC
                model.addConstr(x_inj[i, t] / dis_eff <= x_soc[i, t], name=f"discharge_soc_{i}_{t}")
                
                # Combined injection + withdrawal <= power capacity
                model.addConstr(x_inj[i, t] + x_wdw[i, t] <= y_P[i], name=f"stor_power_{i}_{t}")
                
                # Withdrawal bounded by availability * power capacity
                model.addConstr(x_wdw[i, t] <= a_t * y_P[i], name=f"stor_wdw_max_{i}_{t}")
                
                # Injection bounded by availability * power capacity (generation side)
                model.addConstr(x_inj[i, t] <= a_t * y_P[i], name=f"stor_inj_max_{i}_{t}")
        
        elif g['is_hydro']:
            hp = params
            dur = hp.get('duration_MWh_per_MW', 0.0)
            min_out = hp.get('min_output_frac', 0.0)
            avail = availability.get(str(i), [1.0] * total_hours)
            inflow = hydro_inflow.get(str(i), [0.0] * total_hours)
            ramp_up = hp.get('ramp_up_rate_pct_per_hr', 1.0)
            ramp_down = hp.get('ramp_down_rate_pct_per_hr', 1.0)
            
            for t in range(total_hours):
                tp = prev_hour[t]
                a_t = avail[t]
                inf_t = inflow[t]
                
                # Reservoir balance
                model.addConstr(
                    x_lvl[i, t] == x_lvl[i, tp] + inf_t * y_P[i] - x_inj[i, t] - x_spill[i, t],
                    name=f"hydro_bal_{i}_{t}")
                
                # Reservoir max
                model.addConstr(x_lvl[i, t] <= dur * y_P[i], name=f"hydro_lvl_max_{i}_{t}")
                
                # Min output
                model.addConstr(x_inj[i, t] + x_spill[i, t] >= min_out * y_P[i], name=f"hydro_min_{i}_{t}")
                
                # Generation max
                model.addConstr(x_inj[i, t] <= a_t * y_P[i], name=f"hydro_gen_max_{i}_{t}")
                
                # Ramping
                model.addConstr(x_inj[i, t] - x_inj[i, tp] <= ramp_up * y_P[i], name=f"hydro_ramp_up_{i}_{t}")
                model.addConstr(x_inj[i, tp] - x_inj[i, t] <= ramp_down * y_P[i], name=f"hydro_ramp_down_{i}_{t}")
        
        else:
            # Non-UC, non-storage, non-hydro
            avail = availability.get(str(i), [1.0] * total_hours)
            min_out = params.get('min_output_frac', 0.0)
            ramp_up = params.get('ramp_up_rate_pct_per_hr', 1.0)
            ramp_down = params.get('ramp_down_rate_pct_per_hr', 1.0)
            
            for t in range(total_hours):
                tp = prev_hour[t]
                a_t = avail[t]
                
                # Gen max
                model.addConstr(x_inj[i, t] <= a_t * y_P[i], name=f"gen_max_{i}_{t}")
                
                # Min output for non-UC non-storage non-hydro with positive min_output
                if min_out > 0:
                    model.addConstr(x_inj[i, t] >= min_out * y_P[i], name=f"gen_min_{i}_{t}")
                
                # Ramping
                if ramp_up < 1.0:
                    model.addConstr(x_inj[i, t] - x_inj[i, tp] <= ramp_up * y_P[i], name=f"ramp_up_{i}_{t}")
                if ramp_down < 1.0:
                    model.addConstr(x_inj[i, tp] - x_inj[i, t] <= ramp_down * y_P[i], name=f"ramp_down_{i}_{t}")
    
    # NSE constraints
    for seg in consumer_segments:
        s = seg['segment_id']
        z = seg['zone']
        max_frac = seg['max_nse_fraction']
        dem = demand[str(z)]
        for t in range(total_hours):
            model.addConstr(x_nse[s, z, t] <= max_frac * dem[t], name=f"nse_max_{s}_{z}_{t}")
    
    # Transmission flow constraints
    for line in trans_lines:
        l = line['line_id']
        for t in range(total_hours):
            model.addConstr(x_flow[l, t] <= y_F[l], name=f"flow_max_{l}_{t}")
            model.addConstr(x_flow[l, t] >= -y_F[l], name=f"flow_min_{l}_{t}")
    
    # Power balance per zone per hour
    zones = set()
    for g in generators:
        zones.add(g['zone'])
    for seg in consumer_segments:
        zones.add(seg['zone'])
    for line in trans_lines:
        zones.add(line['from_zone'])
        zones.add(line['to_zone'])
    
    for z in zones:
        dem = demand.get(str(z), [0.0] * total_hours)
        for t in range(total_hours):
            lhs = gp.LinExpr()
            
            # Generation from local resources
            for i in gens_in_zone[z]:
                lhs += x_inj[i, t]
            
            # Minus storage withdrawals
            for i in gens_in_zone[z]:
                g = gen_by_id[i]
                if g['is_storage']:
                    lhs -= x_wdw[i, t]
            
            # Net transmission imports
            for line in lines_to_zone[z]:
                l = line['line_id']
                lhs += x_flow[l, t]
            for line in lines_from_zone[z]:
                l = line['line_id']
                lhs -= x_flow[l, t]
            
            # NSE
            for seg in segs_in_zone[z]:
                s = seg['segment_id']
                lhs += x_nse[s, z, t]
            
            model.addConstr(lhs == dem[t], name=f"balance_{z}_{t}")
    
    # ========== POLICY CONSTRAINTS ==========
    if policy_type == 'RPS':
        rps_share = policy.get('RPS_share', 0.0)
        for w in range(num_weeks):
            hrs = subperiod_hours[w]
            rps_gen = gp.LinExpr()
            total_dem = 0.0
            for t in hrs:
                hw = hourly_weights[t]
                for g in generators:
                    if g['is_RPS_qualifying']:
                        rps_gen += hw * x_inj[g['cluster_id'], t]
                for z in zones:
                    dem_z = demand.get(str(z), [0.0] * total_hours)
                    total_dem += hw * dem_z[t]
            
            model.addConstr(rps_gen + x_RPS[w] >= rps_share * total_dem,
                            name=f"rps_{w}")
    
    if policy_type == 'CO2':
        co2_cap = policy.get('CO2_cap_tons_per_MWh', 0.05)
        for w in range(num_weeks):
            hrs = subperiod_hours[w]
            co2_emit = gp.LinExpr()
            total_dem = 0.0
            for t in hrs:
                hw = hourly_weights[t]
                for g in generators:
                    i = g['cluster_id']
                    p = get_params(g)
                    ef = p.get('CO2_emission_tons_per_MWh', 0.0)
                    if ef > 0:
                        co2_emit += hw * ef * x_inj[i, t]
                        if g['is_storage']:
                            co2_emit += hw * ef * x_wdw[i, t]
                for z in zones:
                    dem_z = demand.get(str(z), [0.0] * total_hours)
                    total_dem += hw * dem_z[t]
            
            model.addConstr(co2_emit - x_CO2[w] <= co2_cap * total_dem,
                            name=f"co2_{w}")
    
    # ========== OBJECTIVE ==========
    obj = gp.LinExpr()
    
    for g in generators:
        i = g['cluster_id']
        params_g = get_params(g)
        cap_size = g['capacity_size_MW']
        is_UC = g['is_UC']
        inv_block = cap_size if is_UC else 1.0
        
        # Investment cost
        inv_cost = params_g.get('investment_cost_per_MW_yr', 0.0)
        obj += inv_block * y_P_new[i] * inv_cost
        
        # Fixed O&M
        fom = params_g.get('fixed_OM_cost_per_MW_yr', 0.0)
        obj += fom * y_P[i]
        
        # Storage energy investment and FOM
        if g['is_storage']:
            bp = params_g
            e_inv = bp.get('energy_investment_cost_per_MWh_yr', 0.0)
            if is_UC:
                e_block = bp.get('energy_capacity_size_MWh', 1.0)
            else:
                e_block = 1.0
            obj += e_block * y_E_new[i] * e_inv
            
            e_fom = bp.get('energy_fixed_OM_cost_per_MWh_yr', 0.0)
            obj += e_fom * y_E[i]
        
        # Hydro energy investment and FOM
        if g['is_hydro']:
            hp = params_g
            dur = hp.get('duration_MWh_per_MW', 0.0)
            e_inv = hp.get('energy_investment_cost_per_MWh_yr', 0.0)
            obj += e_inv * dur * inv_block * y_P_new[i]
            
            e_fom = hp.get('energy_fixed_OM_cost_per_MWh_yr', 0.0)
            obj += e_fom * dur * y_P[i]
        
        # Variable cost
        vc = params_g.get('variable_cost_per_MWh', 0.0)
        for t in range(total_hours):
            hw = hourly_weights[t]
            obj += hw * vc * x_inj[i, t]
            if g['is_storage']:
                obj += hw * vc * x_wdw[i, t]
        
        # Start-up cost
        if g['is_UC']:
            su_cost = params_g.get('start_up_cost', 0.0)
            if su_cost > 0:
                for t in range(total_hours):
                    hw = hourly_weights[t]
                    obj += hw * su_cost * x_start[i, t]
    
    # Transmission investment cost
    for line in trans_lines:
        l = line['line_id']
        t_inv = line['investment_cost_per_MW_yr']
        obj += t_inv * y_F_new[l]
    
    # NSE cost
    for seg in consumer_segments:
        s = seg['segment_id']
        z = seg['zone']
        nse_cost = seg['cost_per_MWh']
        for t in range(total_hours):
            hw = hourly_weights[t]
            obj += hw * nse_cost * x_nse[s, z, t]
    
    # Policy noncompliance cost
    rps_penalty = policy.get('RPS_noncompliance_cost_per_MWh', 0.0)
    co2_penalty = policy.get('CO2_noncompliance_cost_per_ton', 0.0)
    for w in range(num_weeks):
        obj += rps_penalty * x_RPS[w]
        obj += co2_penalty * x_CO2[w]
    
    model.setObjective(obj, GRB.MINIMIZE)
    
    # Callback for logging
    best_obj = [float('inf')]
    
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj_val < best_obj[0] - 1e-6:
                best_obj[0] = obj_val
                if logger:
                    logger.log(obj_val)
    
    model.optimize(callback)
    
    # ========== EXTRACT SOLUTION ==========
    if model.SolCount == 0:
        print("No feasible solution found", file=sys.stderr)
        # Write empty solution
        solution = {
            "objective_value": float('inf'),
            "model_variables": {},
            "investments": {},
            "storage_energy_investments": {},
            "transmission_investments": {}
        }
        with open(solution_path, 'w') as f:
            json.dump(solution, f)
        return
    
    obj_val = model.ObjVal
    
    # Build model_variables (sparse)
    model_vars = {}
    
    for g in generators:
        i = g['cluster_id']
        val = y_P_new[i].X
        if abs(val) > 1e-9:
            model_vars[f"y_P_new[{i}]"] = val
        val = y_P_ret[i].X
        if abs(val) > 1e-9:
            model_vars[f"y_P_ret[{i}]"] = val
        val = y_P[i].X
        if abs(val) > 1e-9:
            model_vars[f"y_P[{i}]"] = val
        
        if g['is_storage']:
            val = y_E_new[i].X
            if abs(val) > 1e-9:
                model_vars[f"y_E_new[{i}]"] = val
            val = y_E[i].X
            if abs(val) > 1e-9:
                model_vars[f"y_E[{i}]"] = val
    
    for line in trans_lines:
        l = line['line_id']
        val = y_F_new[l].X
        if abs(val) > 1e-9:
            model_vars[f"y_F_new[{l}]"] = val
        val = y_F[l].X
        if abs(val) > 1e-9:
            model_vars[f"y_F[{l}]"] = val
    
    for g in generators:
        i = g['cluster_id']
        for t in range(total_hours):
            val = x_inj[i, t].X
            if abs(val) > 1e-9:
                model_vars[f"x_inj[{i},{t}]"] = val
            if g['is_storage']:
                val = x_wdw[i, t].X
                if abs(val) > 1e-9:
                    model_vars[f"x_wdw[{i},{t}]"] = val
                val = x_soc[i, t].X
                if abs(val) > 1e-9:
                    model_vars[f"x_soc[{i},{t}]"] = val
            if g['is_hydro']:
                val = x_lvl[i, t].X
                if abs(val) > 1e-9:
                    model_vars[f"x_lvl[{i},{t}]"] = val
                val = x_spill[i, t].X
                if abs(val) > 1e-9:
                    model_vars[f"x_spill[{i},{t}]"] = val
            if g['is_UC']:
                val = x_commit[i, t].X
                if abs(val) > 1e-9:
                    model_vars[f"x_commit[{i},{t}]"] = val
                val = x_start[i, t].X
                if abs(val) > 1e-9:
                    model_vars[f"x_start[{i},{t}]"] = val
                val = x_shut[i, t].X
                if abs(val) > 1e-9:
                    model_vars[f"x_shut[{i},{t}]"] = val
    
    for seg in consumer_segments:
        s = seg['segment_id']
        z = seg['zone']
        for t in range(total_hours):
            val = x_nse[s, z, t].X
            if abs(val) > 1e-9:
                model_vars[f"x_nse[{s},{z},{t}]"] = val
    
    for line in trans_lines:
        l = line['line_id']
        for t in range(total_hours):
            val = x_flow[l, t].X
            if abs(val) > 1e-9:
                model_vars[f"x_flow[{l},{t}]"] = val
    
    for w in range(num_weeks):
        val = x_RPS[w].X
        if abs(val) > 1e-9:
            model_vars[f"x_RPS[{w}]"] = val
        val = x_CO2[w].X
        if abs(val) > 1e-9:
            model_vars[f"x_CO2[{w}]"] = val
    
    # Build investments
    investments = {}
    for g in generators:
        i = g['cluster_id']
        cap_size = g['capacity_size_MW']
        is_UC = g['is_UC']
        inv_block = cap_size if is_UC else 1.0
        ret_block = cap_size if is_UC else 1.0
        
        new_units = round(y_P_new[i].X)
        ret_units = round(y_P_ret[i].X)
        new_mw = new_units * inv_block
        ret_mw = ret_units * ret_block
        total_mw = y_P[i].X
        
        investments[str(i)] = {
            "resource_type": g['resource_type'],
            "zone": g['zone'],
            "new_units": float(new_units),
            "new_MW": float(new_mw),
            "retired_units": float(ret_units),
            "retired_MW": float(ret_mw),
            "total_capacity_MW": float(total_mw)
        }
    
    # Storage energy investments
    storage_energy_investments = {}
    for g in generators:
        if g['is_storage']:
            i = g['cluster_id']
            is_UC = g['is_UC']
            bp = get_params(g)
            if is_UC:
                e_block = bp.get('energy_capacity_size_MWh', 1.0)
            else:
                e_block = 1.0
            
            new_e_units = round(y_E_new[i].X)
            new_mwh = new_e_units * e_block
            total_e = y_E[i].X
            
            storage_energy_investments[str(i)] = {
                "new_units": float(new_e_units),
                "new_MWh": float(new_mwh),
                "total_energy_MWh": float(total_e)
            }
    
    # Transmission investments
    transmission_investments = {}
    for line in trans_lines:
        l = line['line_id']
        new_mw = round(y_F_new[l].X)
        total_cap = y_F[l].X
        
        transmission_investments[str(l)] = {
            "new_MW": float(new_mw),
            "total_capacity_MW": float(total_cap)
        }
    
    solution = {
        "objective_value": float(obj_val),
        "model_variables": model_vars,
        "investments": investments,
        "storage_energy_investments": storage_energy_investments,
        "transmission_investments": transmission_investments
    }
    
    if logger:
        logger.log_solution(obj_val, solution)
    
    with open(solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()
    
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)
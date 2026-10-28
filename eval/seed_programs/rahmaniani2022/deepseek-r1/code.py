import argparse
import json
import math
import time
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    prob_params = instance['problem_parameters']
    num_zones = prob_params['num_zones']
    num_weeks = prob_params['num_weeks']
    hours_per_subperiod = prob_params['hours_per_subperiod']
    total_hours = prob_params['total_hours']
    num_gen_clusters = prob_params['num_generator_clusters']
    num_uc_gen = prob_params['num_UC_generators']
    subperiod_weights = instance['subperiod_weights']
    generators = instance['generators']
    gen_params = instance['generator_parameters']
    
    existing_capacity = {int(k): v for k, v in instance['existing_capacity'].items()}
    demand = {int(k): np.array(v, dtype=float) for k, v in instance['demand'].items()}
    availability = {int(k): np.array(v, dtype=float) for k, v in instance['availability'].items()}
    hydro_inflow = instance.get('hydro_inflow', {})
    if hydro_inflow:
        hydro_inflow = {int(k): np.array(v, dtype=float) for k, v in hydro_inflow.items()}
    
    consumer_segments = instance['consumer_segments']
    transmission_lines = instance['transmission_lines']
    policy = instance['policy']

    hourly_weights = []
    for w in range(num_weeks):
        weight_per_hour = subperiod_weights[w] / hours_per_subperiod
        hourly_weights.extend([weight_per_hour] * hours_per_subperiod)
    hourly_weights = np.array(hourly_weights)

    uc_clusters = [g for g in generators if g['is_UC']]
    non_uc_clusters = [g for g in generators if not g['is_UC']]
    storage_clusters = [g for g in generators if g['is_storage']]
    hydro_clusters = [g for g in generators if g['is_hydro']]
    uc_storage_clusters = [g for g in storage_clusters if g['is_UC']]
    non_uc_storage_clusters = [g for g in storage_clusters if not g['is_UC']]
    uc_ids = [g['cluster_id'] for g in uc_clusters]
    storage_ids = [g['cluster_id'] for g in storage_clusters]
    hydro_ids = [g['cluster_id'] for g in hydro_clusters]

    cluster_id_to_idx = {g['cluster_id']: i for i, g in enumerate(generators)}
    zone_to_demand = demand

    model = gp.Model("Electricity_System_Planning")
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', args.time_limit)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Precompute the previous time index for each t within the same week
    prev_t_arr = []
    for w in range(num_weeks):
        start_idx = w * hours_per_subperiod
        for h in range(hours_per_subperiod):
            t = start_idx + h
            if h == 0:
                prev_t_arr.append(start_idx + hours_per_subperiod - 1)
            else:
                prev_t_arr.append(t - 1)

    y_P_new = {}
    y_P_ret = {}
    y_E_new = {}
    y_F_new = {}
    for g in generators:
        cid = g['cluster_id']
        res_type = g['resource_type']
        cap_size = gen_params[res_type]['capacity_size_MW']
        max_cap = gen_params[res_type]['max_capacity_MW']
        existing_mw = existing_capacity[cid]
        if g['is_UC']:
            existing_units = existing_mw / cap_size
            max_new_units = max_cap / cap_size
            y_P_new[cid] = model.addVar(lb=0, ub=max_new_units, vtype=GRB.INTEGER, name=f"y_P_new[{cid}]")
            if g['can_retire']:
                y_P_ret[cid] = model.addVar(lb=0, ub=existing_units, vtype=GRB.INTEGER, name=f"y_P_ret[{cid}]")
            else:
                y_P_ret[cid] = model.addVar(lb=0, ub=0, vtype=GRB.INTEGER, name=f"y_P_ret[{cid}]")
        else:
            max_new_mw = max_cap
            y_P_new[cid] = model.addVar(lb=0, ub=max_new_mw, vtype=GRB.INTEGER, name=f"y_P_new[{cid}]")
            if g['can_retire']:
                y_P_ret[cid] = model.addVar(lb=0, ub=existing_mw, vtype=GRB.INTEGER, name=f"y_P_ret[{cid}]")
            else:
                y_P_ret[cid] = model.addVar(lb=0, ub=0, vtype=GRB.INTEGER, name=f"y_P_ret[{cid}]")
        
        if g['is_storage']:
            en_cap_size = gen_params['battery']['energy_capacity_size_MWh']
            max_en_cap = gen_params['battery']['max_energy_capacity_MWh']
            if g['is_UC']:
                max_en_units = max_en_cap / en_cap_size
                y_E_new[cid] = model.addVar(lb=0, ub=max_en_units, vtype=GRB.INTEGER, name=f"y_E_new[{cid}]")
            else:
                y_E_new[cid] = model.addVar(lb=0, ub=max_en_cap, vtype=GRB.INTEGER, name=f"y_E_new[{cid}]")
    
    for line in transmission_lines:
        lid = line['line_id']
        existing_mw = line['existing_capacity_MW']
        max_mw = line['max_capacity_MW']
        max_new_mw = max(0, max_mw - existing_mw)
        y_F_new[lid] = model.addVar(lb=0, ub=max_new_mw, vtype=GRB.INTEGER, name=f"y_F_new[{lid}]")

    # Define variables
    x_inj = model.addVars([(g['cluster_id'], t) for g in generators for t in range(total_hours)], lb=0, name="x_inj")
    x_wdw = model.addVars([(g['cluster_id'], t) for g in storage_clusters for t in range(total_hours)], lb=0, name="x_wdw")
    x_soc = model.addVars([(g['cluster_id'], t) for g in storage_clusters for t in range(total_hours)], lb=0, name="x_soc")
    x_spill = model.addVars([(g['cluster_id'], t) for g in hydro_clusters for t in range(total_hours)], lb=0, name="x_spill")
    x_lvl = model.addVars([(g['cluster_id'], t) for g in hydro_clusters for t in range(total_hours)], lb=0, name="x_lvl")
    
    x_commit = model.addVars([(g['cluster_id'], t) for g in uc_clusters for t in range(total_hours)], lb=0, name="x_commit")
    x_start = model.addVars([(g['cluster_id'], t) for g in uc_clusters for t in range(total_hours)], lb=0, name="x_start")
    x_shut = model.addVars([(g['cluster_id'], t) for g in uc_clusters for t in range(total_hours)], lb=0, name="x_shut")
    
    x_flow = model.addVars([(line['line_id'], t) for line in transmission_lines for t in range(total_hours)], lb=-GRB.INFINITY, name="x_flow")
    
    x_nse = model.addVars([(seg['segment_id'], seg['zone'], t) for seg in consumer_segments for t in range(total_hours)], lb=0, name="x_nse")
    
    if policy['type'] == 'RPS':
        x_RPS = model.addVars(num_weeks, lb=0, name="x_RPS")
    elif policy['type'] == 'CO2':
        x_CO2 = model.addVars(num_weeks, lb=0, name="x_CO2")

    total_power = {}
    total_energy = {}
    for g in generators:
        cid = g['cluster_id']
        res_type = g['resource_type']
        cap_size = gen_params[res_type]['capacity_size_MW']
        existing_mw = existing_capacity[cid]
        if g['is_UC']:
            new_mw = y_P_new[cid] * cap_size
            ret_mw = y_P_ret[cid] * cap_size
        else:
            new_mw = y_P_new[cid]
            ret_mw = y_P_ret[cid]
        total_power[cid] = existing_mw + new_mw - ret_mw
        
        if g['is_storage']:
            en_cap_size = gen_params['battery']['energy_capacity_size_MWh']
            if g['is_UC']:
                total_energy[cid] = y_E_new[cid] * en_cap_size
            else:
                total_energy[cid] = y_E_new[cid]
    
    total_line_cap = {}
    for line in transmission_lines:
        lid = line['line_id']
        total_line_cap[lid] = line['existing_capacity_MW'] + y_F_new[lid]

    for (lid, t), var in x_flow.items():
        cap = total_line_cap[lid]
        model.addConstr(var >= -cap, name=f"flow_lb_{lid}_{t}")
        model.addConstr(var <= cap, name=f"flow_ub_{lid}_{t}")

    for seg in consumer_segments:
        seg_id = seg['segment_id']
        zone = seg['zone']
        max_frac = seg['max_nse_fraction']
        for t in range(total_hours):
            d = zone_to_demand[zone][t]
            var = x_nse[seg_id, zone, t]
            model.addConstr(var <= max_frac * d, name=f"nse_ub_{seg_id}_{zone}_{t}")

    for z in range(num_zones):
        for t in range(total_hours):
            gen_sum = 0
            for g in generators:
                if g['zone'] != z:
                    continue
                cid = g['cluster_id']
                if g['is_storage']:
                    gen_sum += x_inj[cid, t] - x_wdw[cid, t]
                else:
                    gen_sum += x_inj[cid, t]
            
            imp_sum = 0
            for line in transmission_lines:
                lid = line['line_id']
                if line['from_zone'] == z:
                    imp_sum -= x_flow[lid, t]
                if line['to_zone'] == z:
                    imp_sum += x_flow[lid, t]
            
            nse_sum = 0
            for seg in consumer_segments:
                if seg['zone'] == z:
                    seg_id = seg['segment_id']
                    nse_sum += x_nse[seg_id, z, t]
            
            model.addConstr(gen_sum + imp_sum + nse_sum == zone_to_demand[z][t], name=f"balance_z{z}_t{t}")

    for g in generators:
        cid = g['cluster_id']
        res_type = g['resource_type']
        params = gen_params[res_type]
        avail_arr = availability[cid]
        min_out = params.get('min_output_frac', 0)
        ramp_up = params.get('ramp_up_rate_pct_per_hr', 1) / 100.0
        ramp_down = params.get('ramp_down_rate_pct_per_hr', 1) / 100.0
        cap_size = params['capacity_size_MW']
        total_power_cid = total_power[cid]
        
        if g['is_UC']:
            min_up = params.get('min_up_time_hr', 1)
            min_down = params.get('min_down_time_hr', 1)
            existing_mw = existing_capacity[cid]
            existing_units = existing_mw / cap_size
            total_units = existing_units + y_P_new[cid] - y_P_ret[cid]
            
            # Precompute ramp coefficients for UC
            up_coef_arr = []
            down_coef_arr = []
            for t in range(total_hours):
                prev_t = prev_t_arr[t]
                up_coef = min(avail_arr[t], max(min_out, ramp_up))
                up_coef_arr.append(up_coef)
                down_coef = min(avail_arr[prev_t], max(min_out, ramp_down))
                down_coef_arr.append(down_coef)
            
            for t in range(total_hours):
                prev_t = prev_t_arr[t]
                
                model.addConstr(x_commit[cid, t] == x_commit[cid, prev_t] + x_start[cid, t] - x_shut[cid, t], name=f"uc_balance_{cid}_{t}")
                
                model.addConstr(x_inj[cid, t] >= min_out * cap_size * x_commit[cid, t], name=f"uc_min_out_{cid}_{t}")
                avail_t = avail_arr[t]
                model.addConstr(x_inj[cid, t] <= avail_t * cap_size * x_commit[cid, t], name=f"uc_max_out_{cid}_{t}")
                model.addConstr(x_commit[cid, t] <= total_units, name=f"uc_commit_ub_{cid}_{t}")
                
                w_index = t // hours_per_subperiod
                h_index = t % hours_per_subperiod
                
                if min_up > 1:
                    min_up_eff = min(min_up, hours_per_subperiod)
                    start_local = h_index - min_up_eff + 1
                    if start_local < 0:
                        indices_local = list(range(0, h_index+1)) + list(range(hours_per_subperiod + start_local, hours_per_subperiod))
                    else:
                        indices_local = range(start_local, h_index+1)
                    global_indices = [w_index * hours_per_subperiod + idx for idx in indices_local]
                    start_sum = gp.quicksum(x_start[cid, tt] for tt in global_indices)
                    model.addConstr(x_commit[cid, t] >= start_sum, name=f"uc_min_up_{cid}_{t}")
                
                if min_down > 1:
                    min_down_eff = min(min_down, hours_per_subperiod)
                    start_local = h_index - min_down_eff + 1
                    if start_local < 0:
                        indices_local = list(range(0, h_index+1)) + list(range(hours_per_subperiod + start_local, hours_per_subperiod))
                    else:
                        indices_local = range(start_local, h_index+1)
                    global_indices = [w_index * hours_per_subperiod + idx for idx in indices_local]
                    shut_sum = gp.quicksum(x_shut[cid, tt] for tt in global_indices)
                    model.addConstr(x_commit[cid, t] + shut_sum <= total_units, name=f"uc_min_down_{cid}_{t}")
                
                inj_change = x_inj[cid, t] - x_inj[cid, prev_t]
                prev_commit = x_commit[cid, prev_t]
                
                ramp_up_bound = ramp_up * cap_size * (prev_commit - x_start[cid, t])
                ramp_up_bound += up_coef_arr[t] * cap_size * x_start[cid, t]
                ramp_up_bound -= min_out * cap_size * x_shut[cid, t]
                model.addConstr(inj_change <= ramp_up_bound, name=f"uc_ramp_up_{cid}_{t}")
                
                ramp_down_bound = ramp_down * cap_size * (x_commit[cid, t] - x_shut[cid, t])
                ramp_down_bound += down_coef_arr[t] * cap_size * x_shut[cid, t]
                ramp_down_bound -= min_out * cap_size * x_start[cid, t]
                model.addConstr(-inj_change <= ramp_down_bound, name=f"uc_ramp_down_{cid}_{t}")
        else:
            for t in range(total_hours):
                prev_t = prev_t_arr[t]
                avail_t = avail_arr[t]
                if min_out > 0:
                    model.addConstr(x_inj[cid, t] >= min_out * total_power_cid, name=f"gen_min_out_{cid}_{t}")
                model.addConstr(x_inj[cid, t] <= avail_t * total_power_cid, name=f"gen_max_out_{cid}_{t}")
                
                inj_change = x_inj[cid, t] - x_inj[cid, prev_t]
                model.addConstr(inj_change <= ramp_up * total_power_cid, name=f"gen_ramp_up_{cid}_{t}")
                model.addConstr(-inj_change <= ramp_down * total_power_cid, name=f"gen_ramp_down_{cid}_{t}")

    for g in storage_clusters:
        cid = g['cluster_id']
        res_type = 'battery'
        params = gen_params[res_type]
        avail_arr = availability[cid]
        charge_eff = params['charging_efficiency']
        discharge_eff = params['discharging_efficiency']
        self_discharge = params['self_discharge_rate_pct'] / 100.0
        total_power_cid = total_power[cid]
        total_energy_cid = total_energy[cid]
        
        min_dur = params['min_duration_MWh_per_MW']
        max_dur = params['max_duration_MWh_per_MW']
        model.addConstr(total_energy_cid >= min_dur * total_power_cid, name=f"storage_min_dur_{cid}")
        model.addConstr(total_energy_cid <= max_dur * total_power_cid, name=f"storage_max_dur_{cid}")
        
        for t in range(total_hours):
            prev_t = prev_t_arr[t]
            avail_t = avail_arr[t]
            model.addConstr(x_inj[cid, t] <= avail_t * total_power_cid, name=f"storage_discharge_ub_{cid}_{t}")
            model.addConstr(x_wdw[cid, t] <= avail_t * total_power_cid, name=f"storage_charge_ub_{cid}_{t}")
            model.addConstr(x_inj[cid, t] + x_wdw[cid, t] <= total_power_cid, name=f"storage_power_ub_{cid}_{t}")
            
            soc_change = charge_eff * x_wdw[cid, t] - x_inj[cid, t] / discharge_eff
            model.addConstr(x_soc[cid, t] == (1 - self_discharge) * x_soc[cid, prev_t] + soc_change, name=f"soc_balance_{cid}_{t}")
            model.addConstr(x_soc[cid, t] <= total_energy_cid, name=f"soc_ub_{cid}_{t}")
            
            model.addConstr(charge_eff * x_wdw[cid, t] <= total_energy_cid - (1 - self_discharge) * x_soc[cid, prev_t], name=f"charge_room_{cid}_{t}")
            model.addConstr(x_inj[cid, t] / discharge_eff <= (1 - self_discharge) * x_soc[cid, prev_t], name=f"discharge_avail_{cid}_{t}")

    for g in hydro_clusters:
        cid = g['cluster_id']
        res_type = 'hydro'
        params = gen_params[res_type]
        dur_ratio = params['duration_MWh_per_MW']
        min_out = params.get('min_output_frac', 0)
        total_power_cid = total_power[cid]
        inflow_arr = hydro_inflow[cid] if cid in hydro_inflow else np.zeros(total_hours)
        
        for t in range(total_hours):
            prev_t = prev_t_arr[t]
            inflow_t = inflow_arr[t] * total_power_cid
            model.addConstr(x_inj[cid, t] + x_spill[cid, t] >= min_out * total_power_cid, name=f"hydro_min_out_{cid}_{t}")
            model.addConstr(x_inj[cid, t] <= total_power_cid, name=f"hydro_max_out_{cid}_{t}")
            model.addConstr(x_lvl[cid, t] == x_lvl[cid, prev_t] + inflow_t - x_inj[cid, t] - x_spill[cid, t], name=f"hydro_balance_{cid}_{t}")
            model.addConstr(x_lvl[cid, t] <= dur_ratio * total_power_cid, name=f"hydro_lvl_ub_{cid}_{t}")

    if policy['type'] == 'RPS':
        rps_share = policy['RPS_share']
        rps_qualifying = [g for g in generators if g['is_RPS_qualifying']]
        noncompliance_cost = policy['RPS_noncompliance_cost_per_MWh']
        
        for w in range(num_weeks):
            start_idx = w * hours_per_subperiod
            end_idx = start_idx + hours_per_subperiod
            total_demand_w = 0
            rps_gen = 0
            
            for z in range(num_zones):
                total_demand_w += np.sum(zone_to_demand[z][start_idx:end_idx]) * (subperiod_weights[w] / hours_per_subperiod)
            
            for g in rps_qualifying:
                cid = g['cluster_id']
                for t in range(start_idx, end_idx):
                    rps_gen += x_inj[cid, t] * (subperiod_weights[w] / hours_per_subperiod)
            
            model.addConstr(rps_gen + x_RPS[w] >= rps_share * total_demand_w, name=f"RPS_{w}")
    elif policy['type'] == 'CO2':
        co2_cap = policy['CO2_cap_tons_per_MWh']
        noncompliance_cost = policy['CO2_noncompliance_cost_per_ton']
        total_demand_year = 0
        for z in range(num_zones):
            total_demand_year += np.sum(zone_to_demand[z])
        total_co2_cap = co2_cap * total_demand_year
        
        for w in range(num_weeks):
            start_idx = w * hours_per_subperiod
            end_idx = start_idx + hours_per_subperiod
            emissions = 0
            
            for g in generators:
                cid = g['cluster_id']
                res_type = g['resource_type']
                co2_rate = gen_params[res_type]['CO2_emission_tons_per_MWh']
                weight = subperiod_weights[w] / hours_per_subperiod
                for t in range(start_idx, end_idx):
                    if g['is_storage']:
                        emissions += weight * co2_rate * (x_inj[cid, t] + x_wdw[cid, t])
                    else:
                        emissions += weight * co2_rate * x_inj[cid, t]
            
            model.addConstr(emissions - x_CO2[w] <= total_co2_cap, name=f"CO2_{w}")

    total_cost = 0

    for g in generators:
        cid = g['cluster_id']
        res_type = g['resource_type']
        params = gen_params[res_type]
        inv_cost = params['investment_cost_per_MW_yr']
        fixed_om = params['fixed_OM_cost_per_MW_yr']
        var_cost = params['variable_cost_per_MWh']
        
        if g['is_UC']:
            cap_size = params['capacity_size_MW']
            new_mw = y_P_new[cid] * cap_size
        else:
            new_mw = y_P_new[cid]
        
        total_cost += inv_cost * new_mw
        total_cost += fixed_om * total_power[cid]
        
        if g['is_storage']:
            en_inv_cost = params['energy_investment_cost_per_MWh_yr']
            en_om_cost = params['energy_fixed_OM_cost_per_MWh_yr']
            total_cost += en_inv_cost * total_energy[cid]
            total_cost += en_om_cost * total_energy[cid]
            
            for t in range(total_hours):
                weight_t = hourly_weights[t]
                total_cost += weight_t * var_cost * (x_inj[cid, t] + x_wdw[cid, t])
        elif g['is_hydro']:
            en_inv_cost = params['energy_investment_cost_per_MWh_yr']
            en_om_cost = params['energy_fixed_OM_cost_per_MWh_yr']
            dur_ratio = params['duration_MWh_per_MW']
            total_cost += en_inv_cost * dur_ratio * new_mw
            total_cost += en_om_cost * dur_ratio * total_power[cid]
            
            for t in range(total_hours):
                weight_t = hourly_weights[t]
                total_cost += weight_t * var_cost * x_inj[cid, t]
        else:
            for t in range(total_hours):
                weight_t = hourly_weights[t]
                total_cost += weight_t * var_cost * x_inj[cid, t]
        
        if g['is_UC']:
            start_up_cost = params['start_up_cost']
            cap_size = params['capacity_size_MW']
            for t in range(total_hours):
                weight_t = hourly_weights[t]
                total_cost += weight_t * start_up_cost * cap_size * x_start[cid, t]
    
    for seg in consumer_segments:
        seg_id = seg['segment_id']
        zone = seg['zone']
        cost_per_mwh = seg['cost_per_MWh']
        for t in range(total_hours):
            weight_t = hourly_weights[t]
            total_cost += weight_t * cost_per_mwh * x_nse[seg_id, zone, t]
    
    for line in transmission_lines:
        lid = line['line_id']
        inv_cost = line['investment_cost_per_MW_yr']
        total_cost += inv_cost * y_F_new[lid]
    
    if policy['type'] == 'RPS':
        for w in range(num_weeks):
            total_cost += noncompliance_cost * x_RPS[w]
    elif policy['type'] == 'CO2':
        for w in range(num_weeks):
            total_cost += noncompliance_cost * x_CO2[w]

    model.setObjective(total_cost, GRB.MINIMIZE)

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            if logger:
                obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                logger.log(obj)

    model.optimize(callback if logger else None)

    solution = {
        "objective_value": model.ObjVal,
        "investments": {},
        "storage_energy_investments": {},
        "transmission_investments": {},
        "model_variables": {}
    }

    for g in generators:
        cid = g['cluster_id']
        res_type = g['resource_type']
        cap_size = gen_params[res_type]['capacity_size_MW']
        existing_mw = existing_capacity[cid]
        
        if g['is_UC']:
            new_units = y_P_new[cid].X
            new_mw = new_units * cap_size
            ret_units = y_P_ret[cid].X
            ret_mw = ret_units * cap_size
        else:
            new_units = y_P_new[cid].X
            new_mw = new_units
            ret_units = y_P_ret[cid].X
            ret_mw = ret_units
        
        total_mw = existing_mw + new_mw - ret_mw
        solution["investments"][cid] = {
            "resource_type": res_type,
            "zone": g['zone'],
            "new_units": float(new_units),
            "new_MW": float(new_mw),
            "retired_units": float(ret_units),
            "retired_MW": float(ret_mw),
            "total_capacity_MW": float(total_mw)
        }
        
        if g['is_storage']:
            en_cap_size = gen_params['battery']['energy_capacity_size_MWh']
            if g['is_UC']:
                new_energy_units = y_E_new[cid].X
                new_mwh = new_energy_units * en_cap_size
            else:
                new_energy_units = y_E_new[cid].X
                new_mwh = new_energy_units
            
            total_mwh = new_mwh
            solution["storage_energy_investments"][cid] = {
                "new_units": float(new_energy_units),
                "new_MWh": float(new_mwh),
                "total_energy_MWh": float(total_mwh)
            }
    
    for line in transmission_lines:
        lid = line['line_id']
        new_mw = y_F_new[lid].X
        total_cap = line['existing_capacity_MW'] + new_mw
        solution["transmission_investments"][lid] = {
            "new_MW": float(new_mw),
            "total_capacity_MW": float(total_cap)
        }
    
    # Only include non-zero variables in solution
    epsilon = 1e-6
    def add_var_to_solution(var_name, value):
        if abs(value) > epsilon:
            solution["model_variables"][var_name] = float(value)
    
    for key, var in x_inj.items():
        add_var_to_solution(f"x_inj[{key[0]},{key[1]}]", var.X)
    
    for key, var in x_wdw.items():
        add_var_to_solution(f"x_wdw[{key[0]},{key[1]}]", var.X)
    
    for key, var in x_soc.items():
        add_var_to_solution(f"x_soc[{key[0]},{key[1]}]", var.X)
    
    for key, var in x_spill.items():
        add_var_to_solution(f"x_spill[{key[0]},{key[1]}]", var.X)
    
    for key, var in x_lvl.items():
        add_var_to_solution(f"x_lvl[{key[0]},{key[1]}]", var.X)
    
    for key, var in x_commit.items():
        add_var_to_solution(f"x_commit[{key[0]},{key[1]}]", var.X)
    
    for key, var in x_start.items():
        add_var_to_solution(f"x_start[{key[0]},{key[1]}]", var.X)
    
    for key, var in x_shut.items():
        add_var_to_solution(f"x_shut[{key[0]},{key[1]}]", var.X)
    
    for key, var in x_flow.items():
        add_var_to_solution(f"x_flow[{key[0]},{key[1]}]", var.X)
    
    for key, var in x_nse.items():
        seg_id, zone, t = key
        add_var_to_solution(f"x_nse[{seg_id},{zone},{t}]", var.X)
    
    if policy['type'] == 'RPS':
        for w in range(num_weeks):
            add_var_to_solution(f"x_RPS[{w}]", x_RPS[w].X)
    elif policy['type'] == 'CO2':
        for w in range(num_weeks):
            add_var_to_solution(f"x_CO2[{w}]", x_CO2[w].X)

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()
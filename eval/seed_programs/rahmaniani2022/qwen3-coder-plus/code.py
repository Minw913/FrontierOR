import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
import numpy as np


def solve_instance(instance_path, solution_path, time_limit, log_path):
    # Load instance data
    with open(instance_path, 'r') as f:
        data = json.load(f)
    
    # Initialize logger
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    # Extract problem parameters
    num_zones = data['problem_parameters']['num_zones']
    num_weeks = data['problem_parameters']['num_weeks']
    hours_per_subperiod = data['problem_parameters']['hours_per_subperiod']  # Always 168
    total_hours = data['problem_parameters']['total_hours']
    num_generator_clusters = data['problem_parameters']['num_generator_clusters']
    num_UC_generators = data['problem_parameters']['num_UC_generators']
    
    # Policy parameters
    policy_type = data['policy']['type']
    rps_share = data['policy']['RPS_share']
    co2_cap = data['policy']['CO2_cap_tons_per_MWh']
    rps_penalty = data['policy']['RPS_noncompliance_cost_per_MWh']
    co2_penalty = data['policy']['CO2_noncompliance_cost_per_ton']
    
    subperiod_weights = data['subperiod_weights']
    hourly_weights = []
    for w_idx, weight in enumerate(subperiod_weights):
        for h in range(hours_per_subperiod):
            idx = w_idx * hours_per_subperiod + h
            hourly_weights.append(weight / hours_per_subperiod)
    
    # Generator information
    generators = {gen['cluster_id']: gen for gen in data['generators']}
    gen_ids = list(generators.keys())
    gen_to_idx = {cid: i for i, cid in enumerate(gen_ids)}
    
    # Separate UC and non-UC generators
    uc_gen_ids = [cid for cid, gen in generators.items() if gen['is_UC']]
    non_uc_gen_ids = [cid for cid, gen in generators.items() if not gen['is_UC']]
    
    # Storage and hydro generators
    storage_gen_ids = [cid for cid, gen in generators.items() if gen['is_storage']]
    hydro_gen_ids = [cid for cid, gen in generators.items() if gen['is_hydro']]
    
    # Extract generator parameters
    params = {}
    for gen_type, param_data in data['generator_parameters'].items():
        params[gen_type] = param_data
    
    # Existing capacities
    existing_capacity = {int(k): v for k, v in data['existing_capacity'].items()}
    
    # Demand
    demand_by_zone = {int(k): v for k, v in data['demand'].items()}
    
    # Consumer segments
    consumer_segments = data['consumer_segments']
    
    # Transmission lines
    transmission_lines = data['transmission_lines']
    line_ids = [line['line_id'] for line in transmission_lines]
    line_to_idx = {lid: i for i, lid in enumerate(line_ids)}
    
    # Availability factors
    availability = {int(k): v for k, v in data['availability'].items()}
    
    # Hydro inflow
    hydro_inflow = {int(k): v for k, v in data['hydro_inflow'].items()}
    
    # Create model
    model = gp.Model("ElectricitySystemPlanning")
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', time_limit)
    
    # Decision variables
    # Investment variables
    y_P_new = {}  # New power capacity (in units for UC, in MW for non-UC)
    y_P_ret = {}  # Retired power capacity (in units for UC, in MW for non-UC)
    y_P = {}      # Total power capacity (MW)
    
    for cid in gen_ids:
        gen = generators[cid]
        existing_cap = existing_capacity.get(cid, 0.0)
        
        if gen['is_UC']:
            # For UC, investment is in blocks of capacity_size_MW
            unit_size = gen['capacity_size_MW']
            max_new_units = int(data['generator_parameters'][gen['resource_type']]['max_capacity_MW'] / unit_size)
            max_retire_units = int(existing_cap / unit_size)
            
            y_P_new[cid] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=max_new_units, name=f"y_P_new[{cid}]")
            y_P_ret[cid] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=max_retire_units, name=f"y_P_ret[{cid}]")
            y_P[cid] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"y_P[{cid}]")
        else:
            # For non-UC, investment is in 1 MW increments
            max_new_mw = data['generator_parameters'][gen['resource_type']]['max_capacity_MW']
            max_retire_mw = existing_cap
            
            y_P_new[cid] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=max_new_mw, name=f"y_P_new[{cid}]")
            y_P_ret[cid] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=max_retire_mw, name=f"y_P_ret[{cid}]")
            y_P[cid] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"y_P[{cid}]")
        
        # Add constraint linking total capacity
        if gen['is_UC']:
            unit_size = gen['capacity_size_MW']
            model.addConstr(y_P[cid] == existing_cap + y_P_new[cid] * unit_size - y_P_ret[cid] * unit_size, 
                           name=f"total_cap_{cid}")
        else:
            model.addConstr(y_P[cid] == existing_cap + y_P_new[cid] - y_P_ret[cid], 
                           name=f"total_cap_{cid}")
        
        # Non-retirable constraint
        if not gen['can_retire']:
            model.addConstr(y_P_ret[cid] == 0, name=f"no_retire_{cid}")
    
    # Storage energy investment variables
    y_E_new = {}  # New energy capacity (in units for UC, in MWh for non-UC)
    y_E = {}      # Total energy capacity (MWh)
    
    for cid in storage_gen_ids:
        gen = generators[cid]
        existing_cap = existing_capacity.get(cid, 0.0)
        gen_params = params[gen['resource_type']]
        
        if gen['is_UC']:
            unit_size = gen_params['energy_capacity_size_MWh']
            max_new_units = int(gen_params['max_energy_capacity_MWh'] / unit_size)
            
            y_E_new[cid] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=max_new_units, name=f"y_E_new[{cid}]")
            y_E[cid] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"y_E[{cid}]")
            model.addConstr(y_E[cid] == existing_cap * gen_params['max_duration_MWh_per_MW'] + 
                           y_E_new[cid] * unit_size, name=f"total_energy_cap_{cid}")
        else:
            max_new_mwh = gen_params['max_energy_capacity_MWh']
            y_E_new[cid] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=max_new_mwh, name=f"y_E_new[{cid}]")
            y_E[cid] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"y_E[{cid}]")
            model.addConstr(y_E[cid] == existing_cap * gen_params['max_duration_MWh_per_MW'] + 
                           y_E_new[cid], name=f"total_energy_cap_{cid}")
        
        # Duration constraints
        min_dur = gen_params['min_duration_MWh_per_MW']
        max_dur = gen_params['max_duration_MWh_per_MW']
        model.addConstr(y_E[cid] >= y_P[cid] * min_dur, name=f"min_duration_{cid}")
        model.addConstr(y_E[cid] <= y_P[cid] * max_dur, name=f"max_duration_{cid}")
    
    # Hydro energy capacity is fixed by duration ratio
    for cid in hydro_gen_ids:
        gen = generators[cid]
        existing_cap = existing_capacity.get(cid, 0.0)
        gen_params = params[gen['resource_type']]
        duration_ratio = gen_params['duration_MWh_per_MW']
        
        y_E[cid] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"y_E[{cid}]")
        model.addConstr(y_E[cid] == y_P[cid] * duration_ratio, name=f"hydro_energy_{cid}")
    
    # Transmission investment variables
    y_F_new = {}  # New transmission capacity (in MW)
    y_F = {}      # Total transmission capacity (MW)
    
    for line in transmission_lines:
        lid = line['line_id']
        existing_cap = line['existing_capacity_MW']
        max_total_cap = line['max_capacity_MW']
        
        y_F_new[lid] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=int(max_total_cap - existing_cap), 
                                   name=f"y_F_new[{lid}]")
        y_F[lid] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"y_F[{lid}]")
        model.addConstr(y_F[lid] == existing_cap + y_F_new[lid], name=f"total_trans_cap_{lid}")
    
    # Operational variables
    x_inj = {}  # Generation/injection (for all resources)
    x_wdw = {}  # Storage withdrawal (only for storage)
    x_soc = {}  # State of charge (only for storage)
    
    for cid in gen_ids:
        x_inj[cid] = []
        for t in range(total_hours):
            ub = GRB.INFINITY
            if not generators[cid]['is_storage']:
                ub = y_P[cid] * availability[cid][t]
            var = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=ub, name=f"x_inj[{cid},{t}]")
            x_inj[cid].append(var)
    
    for cid in storage_gen_ids:
        x_wdw[cid] = []
        x_soc[cid] = []
        for t in range(total_hours):
            # Withdrawal upper bound
            wd_ub = y_P[cid] * availability[cid][t]
            wd_var = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=wd_ub, name=f"x_wdw[{cid},{t}]")
            soc_var = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=y_E[cid], name=f"x_soc[{cid},{t}]")
            x_wdw[cid].append(wd_var)
            x_soc[cid].append(soc_var)
    
    # Hydro variables
    x_lvl = {}    # Reservoir level (only for hydro)
    x_spill = {}  # Spillage (only for hydro)
    
    for cid in hydro_gen_ids:
        x_lvl[cid] = []
        x_spill[cid] = []
        for t in range(total_hours):
            lvl_var = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=y_E[cid], name=f"x_lvl[{cid},{t}]")
            spill_var = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"x_spill[{cid},{t}]")
            x_lvl[cid].append(lvl_var)
            x_spill[cid].append(spill_var)
    
    # Non-served energy variables
    x_nse = {}
    for seg in consumer_segments:
        seg_id = seg['segment_id']
        zone = seg['zone']
        x_nse[(seg_id, zone)] = []
        for t in range(total_hours):
            max_nse = demand_by_zone[zone][t] * seg['max_nse_fraction']
            var = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=max_nse, name=f"x_nse[{seg_id},{zone},{t}]")
            x_nse[(seg_id, zone)].append(var)
    
    # Transmission flow variables
    x_flow = {}
    for line in transmission_lines:
        lid = line['line_id']
        from_zone = line['from_zone']
        to_zone = line['to_zone']
        x_flow[lid] = []
        for t in range(total_hours):
            var = model.addVar(vtype=GRB.CONTINUOUS, lb=-y_F[lid], ub=y_F[lid], name=f"x_flow[{lid},{t}]")
            x_flow[lid].append(var)
    
    # Unit commitment variables (for UC resources)
    x_commit = {}
    x_start = {}
    x_shut = {}
    
    for cid in uc_gen_ids:
        gen = generators[cid]
        unit_size = gen['capacity_size_MW']
        max_units = y_P[cid] / unit_size
        
        x_commit[cid] = []
        x_start[cid] = []
        x_shut[cid] = []
        
        for t in range(total_hours):
            commit_var = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=max_units, name=f"x_commit[{cid},{t}]")
            start_var = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"x_start[{cid},{t}]")
            shut_var = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"x_shut[{cid},{t}]")
            x_commit[cid].append(commit_var)
            x_start[cid].append(start_var)
            x_shut[cid].append(shut_var)
    
    # Policy compliance slack variables
    if policy_type == 'RPS':
        x_RPS = []  # RPS noncompliance slack per subperiod
        for w_idx in range(num_weeks):
            var = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"x_RPS[{w_idx}]")
            x_RPS.append(var)
    elif policy_type == 'CO2':
        x_CO2 = []  # CO2 noncompliance slack per subperiod
        for w_idx in range(num_weeks):
            var = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"x_CO2[{w_idx}]")
            x_CO2.append(var)
    
    # Constraints
    
    # Unit commitment constraints
    for cid in uc_gen_ids:
        gen = generators[cid]
        gen_params = params[gen['resource_type']]
        unit_size = gen['capacity_size_MW']
        min_out_frac = gen_params['min_output_frac']
        availability_ts = availability[cid]
        
        for t in range(total_hours):
            # Generation bounds
            model.addConstr(x_inj[cid][t] >= x_commit[cid][t] * unit_size * min_out_frac,
                           name=f"uc_min_gen_{cid}_{t}")
            model.addConstr(x_inj[cid][t] <= x_commit[cid][t] * unit_size * availability_ts[t],
                           name=f"uc_max_gen_{cid}_{t}")
            
            # Commitment balance (with wraparound)
            prev_t = (t - 1) % total_hours
            model.addConstr(x_commit[cid][t] == x_commit[cid][prev_t] + x_start[cid][t] - x_shut[cid][t],
                           name=f"commit_balance_{cid}_{t}")
    
    # Ramping constraints for UC resources
    for cid in uc_gen_ids:
        gen = generators[cid]
        gen_params = params[gen['resource_type']]
        unit_size = gen['capacity_size_MW']
        ramp_up_rate = gen_params['ramp_up_rate_pct_per_hr']
        ramp_down_rate = gen_params['ramp_down_rate_pct_per_hr']
        min_out_frac = gen_params['min_output_frac']
        availability_ts = availability[cid]
        
        for t in range(total_hours):
            prev_t = (t - 1) % total_hours
            
            # Ramping up constraint
            model.addConstr(
                x_inj[cid][t] - x_inj[cid][prev_t] <= 
                x_commit[cid][prev_t] * unit_size * ramp_up_rate +
                x_start[cid][t] * unit_size * min(availability_ts[t], max(min_out_frac, ramp_up_rate)) -
                x_shut[cid][t] * unit_size * min_out_frac,
                name=f"ramp_up_{cid}_{t}"
            )
            
            # Ramping down constraint
            model.addConstr(
                x_inj[cid][prev_t] - x_inj[cid][t] <= 
                x_commit[cid][t] * unit_size * ramp_down_rate +
                x_shut[cid][t] * unit_size * min_out_frac +
                x_start[cid][t] * unit_size * max(0, min_out_frac - ramp_down_rate),
                name=f"ramp_down_{cid}_{t}"
            )
    
    # Ramping constraints for non-UC resources
    for cid in non_uc_gen_ids:
        if generators[cid]['is_storage'] or generators[cid]['is_hydro']:
            continue  # Storage and hydro have different dynamics
        
        gen = generators[cid]
        gen_params = params[gen['resource_type']]
        ramp_up_rate = gen_params['ramp_up_rate_pct_per_hr']
        ramp_down_rate = gen_params['ramp_down_rate_pct_per_hr']
        
        for t in range(total_hours):
            prev_t = (t - 1) % total_hours
            cap = y_P[cid]
            
            model.addConstr(x_inj[cid][t] - x_inj[cid][prev_t] <= cap * ramp_up_rate, 
                           name=f"ramp_up_non_uc_{cid}_{t}")
            model.addConstr(x_inj[cid][prev_t] - x_inj[cid][t] <= cap * ramp_down_rate, 
                           name=f"ramp_down_non_uc_{cid}_{t}")
    
    # Minimum uptime/downtime constraints for UC resources
    for cid in uc_gen_ids:
        gen = generators[cid]
        gen_params = params[gen['resource_type']]
        min_up_time = gen_params['min_up_time_hr']
        min_down_time = gen_params['min_down_time_hr']
        unit_size = gen['capacity_size_MW']
        
        max_units = y_P[cid] / unit_size
        
        for t in range(total_hours):
            # Minimum up time: if we're online at t, we must have been online for min_up_time hours
            for u in range(1, min_up_time):
                prev_t = (t - u) % total_hours
                model.addConstr(x_commit[cid][t] <= x_start[cid][prev_t] + (1 - x_start[cid][prev_t]) * max_units,
                               name=f"min_up_time_{cid}_{t}_{u}")
            
            # Minimum down time: if we're offline at t, we must have been offline for min_down_time hours
            for d in range(1, min_down_time):
                prev_t = (t - d) % total_hours
                model.addConstr(max_units - x_commit[cid][t] <= x_shut[cid][prev_t] + (1 - x_shut[cid][prev_t]) * max_units,
                               name=f"min_down_time_{cid}_{t}_{d}")
    
    # Storage dynamics
    for cid in storage_gen_ids:
        gen_params = params[gen['resource_type']]
        ch_eff = gen_params['charging_efficiency']
        disch_eff = gen_params['discharging_efficiency']
        self_disch_rate = gen_params['self_discharge_rate_pct']
        
        for t in range(total_hours):
            prev_t = (t - 1) % total_hours
            
            # State of charge balance
            model.addConstr(
                x_soc[cid][t] == 
                x_soc[cid][prev_t] * (1 - self_disch_rate) + 
                x_inj[cid][t] * ch_eff - 
                x_wdw[cid][t] / disch_eff,
                name=f"soc_balance_{cid}_{t}"
            )
            
            # Charging + discharging <= power capacity
            model.addConstr(x_inj[cid][t] + x_wdw[cid][t] <= y_P[cid], 
                           name=f"power_limit_{cid}_{t}")
    
    # Hydro dynamics
    for cid in hydro_gen_ids:
        gen_params = params[gen['resource_type']]
        dur_ratio = gen_params['duration_MWh_per_MW']
        min_out_frac = gen_params['min_output_frac']
        inflow_ts = hydro_inflow[cid]
        
        for t in range(total_hours):
            prev_t = (t - 1) % total_hours
            inflow = inflow_ts[t] * y_P[cid]
            
            # Level balance
            model.addConstr(
                x_lvl[cid][t] == x_lvl[cid][prev_t] + inflow - x_inj[cid][t] - x_spill[cid][t],
                name=f"lvl_balance_{cid}_{t}"
            )
            
            # Minimum output constraint
            model.addConstr(x_inj[cid][t] + x_spill[cid][t] >= min_out_frac * y_P[cid],
                           name=f"hydro_min_out_{cid}_{t}")
    
    # Power balance constraints
    for z in range(num_zones):
        for t in range(total_hours):
            total_gen = gp.LinExpr()
            total_withdrawal = gp.LinExpr()
            total_import = gp.LinExpr()
            total_nse = gp.LinExpr()
            
            # Add generation from all resources in this zone
            for cid in gen_ids:
                if generators[cid]['zone'] == z:
                    total_gen += x_inj[cid][t]
            
            # Add withdrawal from all storage in this zone
            for cid in storage_gen_ids:
                if generators[cid]['zone'] == z:
                    total_withdrawal += x_wdw[cid][t]
            
            # Add net transmission flows to this zone
            for line in transmission_lines:
                if line['from_zone'] == z:
                    total_import -= x_flow[line['line_id']][t]
                elif line['to_zone'] == z:
                    total_import += x_flow[line['line_id']][t]
            
            # Add non-served energy for this zone
            for seg in consumer_segments:
                if seg['zone'] == z:
                    total_nse += x_nse[(seg['segment_id'], z)][t]
            
            # Power balance equation
            model.addConstr(
                total_gen - total_withdrawal + total_import + total_nse == demand_by_zone[z][t],
                name=f"power_balance_{z}_{t}"
            )
    
    # Policy constraints
    if policy_type == 'RPS':
        # Calculate total weighted RPS qualifying generation and total demand
        total_rps_gen = gp.LinExpr()
        total_demand = gp.LinExpr()
        
        for w_idx in range(num_weeks):
            week_rps_gen = gp.LinExpr()
            week_demand = gp.LinExpr()
            
            for h in range(hours_per_subperiod):
                t = w_idx * hours_per_subperiod + h
                weight = hourly_weights[t]
                
                # Add RPS qualifying generation
                for cid in gen_ids:
                    if generators[cid]['is_RPS_qualifying']:
                        week_rps_gen += x_inj[cid][t] * weight
                
                # Add total demand
                for z in range(num_zones):
                    week_demand += demand_by_zone[z][t] * weight
            
            # Add to total
            total_rps_gen += week_rps_gen
            total_demand += week_demand
            
            # Subperiod RPS constraint
            model.addConstr(week_rps_gen + x_RPS[w_idx] >= rps_share * week_demand,
                           name=f"rps_subperiod_{w_idx}")
        
        # Overall RPS constraint
        model.addConstr(total_rps_gen + gp.quicksum(x_RPS) >= rps_share * total_demand,
                       name="rps_total")
    
    elif policy_type == 'CO2':
        # Calculate total weighted CO2 emissions
        total_co2_emissions = gp.LinExpr()
        total_demand = gp.LinExpr()
        
        for w_idx in range(num_weeks):
            week_co2 = gp.LinExpr()
            week_demand = gp.LinExpr()
            
            for h in range(hours_per_subperiod):
                t = w_idx * hours_per_subperiod + h
                weight = hourly_weights[t]
                
                # Add CO2 emissions from generation
                for cid in gen_ids:
                    gen_params = params[generators[cid]['resource_type']]
                    co2_factor = gen_params['CO2_emission_tons_per_MWh']
                    week_co2 += x_inj[cid][t] * co2_factor * weight
                
                # Add CO2 emissions from storage withdrawals
                for cid in storage_gen_ids:
                    gen_params = params[generators[cid]['resource_type']]
                    co2_factor = gen_params['CO2_emission_tons_per_MWh']
                    week_co2 += x_wdw[cid][t] * co2_factor * weight
                
                # Add total demand
                for z in range(num_zones):
                    week_demand += demand_by_zone[z][t] * weight
            
            # Subperiod CO2 constraint
            model.addConstr(week_co2 - x_CO2[w_idx] <= co2_cap * week_demand,
                           name=f"co2_subperiod_{w_idx}")
        
        # Overall CO2 constraint
        model.addConstr(total_co2_emissions - gp.quicksum(x_CO2) <= co2_cap * total_demand,
                       name="co2_total")
    
    # Objective function
    # Fixed costs
    fixed_costs = gp.LinExpr()
    
    # Investment costs
    for cid in gen_ids:
        gen = generators[cid]
        gen_params = params[gen['resource_type']]
        inv_cost = gen_params['investment_cost_per_MW_yr']
        
        if gen['is_UC']:
            unit_size = gen['capacity_size_MW']
            fixed_costs.addTerms([inv_cost * unit_size], [y_P_new[cid]])
        else:
            fixed_costs.addTerms([inv_cost], [y_P_new[cid]])
    
    # Storage energy investment costs
    for cid in storage_gen_ids:
        gen = generators[cid]
        gen_params = params[gen['resource_type']]
        inv_cost = gen_params['energy_investment_cost_per_MWh_yr']
        
        if gen['is_UC']:
            unit_size = gen_params['energy_capacity_size_MWh']
            fixed_costs.addTerms([inv_cost * unit_size], [y_E_new[cid]])
        else:
            fixed_costs.addTerms([inv_cost], [y_E_new[cid]])
    
    # Hydro energy investment costs
    for cid in hydro_gen_ids:
        gen = generators[cid]
        gen_params = params[gen['resource_type']]
        inv_cost = gen_params['energy_investment_cost_per_MWh_yr']
        dur_ratio = gen_params['duration_MWh_per_MW']
        
        if gen['is_UC']:
            unit_size = gen['capacity_size_MW']
            fixed_costs.addTerms([inv_cost * dur_ratio * unit_size], [y_P_new[cid]])
        else:
            fixed_costs.addTerms([inv_cost * dur_ratio], [y_P_new[cid]])
    
    # Transmission investment costs
    for line in transmission_lines:
        lid = line['line_id']
        inv_cost = line['investment_cost_per_MW_yr']
        fixed_costs.addTerms([inv_cost], [y_F_new[lid]])
    
    # Fixed O&M costs
    for cid in gen_ids:
        gen = generators[cid]
        gen_params = params[gen['resource_type']]
        om_cost = gen_params['fixed_OM_cost_per_MW_yr']
        fixed_costs.addTerms([om_cost], [y_P[cid]])
    
    # Storage energy fixed O&M costs
    for cid in storage_gen_ids:
        gen = generators[cid]
        gen_params = params[gen['resource_type']]
        om_cost = gen_params['energy_fixed_OM_cost_per_MWh_yr']
        fixed_costs.addTerms([om_cost], [y_E[cid]])
    
    # Hydro energy fixed O&M costs
    for cid in hydro_gen_ids:
        gen = generators[cid]
        gen_params = params[gen['resource_type']]
        om_cost = gen_params['energy_fixed_OM_cost_per_MWh_yr']
        fixed_costs.addTerms([om_cost], [y_E[cid]])
    
    # Variable costs
    var_costs = gp.LinExpr()
    for t in range(total_hours):
        weight = hourly_weights[t]
        
        # Generation variable costs
        for cid in gen_ids:
            gen = generators[cid]
            gen_params = params[gen['resource_type']]
            var_cost = gen_params['variable_cost_per_MWh']
            var_costs.addTerms([var_cost * weight], [x_inj[cid][t]])
        
        # Storage withdrawal variable costs
        for cid in storage_gen_ids:
            gen_params = params[generators[cid]['resource_type']]
            var_cost = gen_params['variable_cost_per_MWh']
            var_costs.addTerms([var_cost * weight], [x_wdw[cid][t]])
        
        # Non-served energy costs
        for seg in consumer_segments:
            seg_id = seg['segment_id']
            zone = seg['zone']
            cost = seg['cost_per_MWh']
            var_costs.addTerms([cost * weight], [x_nse[(seg_id, zone)][t]])
    
    # Start-up costs for UC resources
    startup_costs = gp.LinExpr()
    for cid in uc_gen_ids:
        gen = generators[cid]
        gen_params = params[gen['resource_type']]
        startup_cost = gen_params['start_up_cost']
        
        for t in range(total_hours):
            weight = hourly_weights[t]
            startup_costs.addTerms([startup_cost * weight], [x_start[cid][t]])
    
    # Policy noncompliance costs
    policy_costs = gp.LinExpr()
    if policy_type == 'RPS':
        for w_idx in range(num_weeks):
            policy_costs.addTerms([rps_penalty * subperiod_weights[w_idx]], [x_RPS[w_idx]])
    elif policy_type == 'CO2':
        for w_idx in range(num_weeks):
            policy_costs.addTerms([co2_penalty * subperiod_weights[w_idx]], [x_CO2[w_idx]])
    
    # Total cost
    total_cost = fixed_costs + var_costs + startup_costs + policy_costs
    model.setObjective(total_cost, GRB.MINIMIZE)
    
    # Optimize
    model.optimize()
    
    # Prepare solution dictionary
    solution = {
        "objective_value": model.ObjVal if model.SolCount > 0 else float('inf'),
        "model_variables": {},
        "investments": {},
        "storage_energy_investments": {},
        "transmission_investments": {}
    }
    
    if model.SolCount > 0:
        # Fill investments
        for cid in gen_ids:
            gen = generators[cid]
            solution["investments"][str(cid)] = {
                "resource_type": gen['resource_type'],
                "zone": gen['zone'],
                "new_units": y_P_new[cid].X,
                "new_MW": y_P_new[cid].X * gen['capacity_size_MW'] if gen['is_UC'] else y_P_new[cid].X,
                "retired_units": y_P_ret[cid].X,
                "retired_MW": y_P_ret[cid].X * gen['capacity_size_MW'] if gen['is_UC'] else y_P_ret[cid].X,
                "total_capacity_MW": y_P[cid].X
            }
        
        # Fill storage energy investments
        for cid in storage_gen_ids:
            gen = generators[cid]
            solution["storage_energy_investments"][str(cid)] = {
                "new_units": y_E_new[cid].X,
                "new_MWh": y_E_new[cid].X * params[gen['resource_type']]['energy_capacity_size_MWh'] if gen['is_UC'] else y_E_new[cid].X,
                "total_energy_MWh": y_E[cid].X
            }
        
        # Fill transmission investments
        for line in transmission_lines:
            lid = line['line_id']
            solution["transmission_investments"][str(lid)] = {
                "new_MW": y_F_new[lid].X,
                "total_capacity_MW": y_F[lid].X
            }
        
        # Log solution if logger exists
        if logger:
            logger.log_solution(model.ObjVal, solution)
    else:
        # Even if no solution found, still create empty structure
        for cid in gen_ids:
            gen = generators[cid]
            solution["investments"][str(cid)] = {
                "resource_type": gen['resource_type'],
                "zone": gen['zone'],
                "new_units": 0,
                "new_MW": 0,
                "retired_units": 0,
                "retired_MW": 0,
                "total_capacity_MW": existing_capacity.get(cid, 0.0)
            }
        
        for cid in storage_gen_ids:
            gen = generators[cid]
            solution["storage_energy_investments"][str(cid)] = {
                "new_units": 0,
                "new_MWh": 0,
                "total_energy_MWh": existing_capacity.get(cid, 0.0) * params[gen['resource_type']]['max_duration_MWh_per_MW']
            }
        
        for line in transmission_lines:
            lid = line['line_id']
            solution["transmission_investments"][str(lid)] = {
                "new_MW": 0,
                "total_capacity_MW": line['existing_capacity_MW']
            }
    
    # Write solution
    with open(solution_path, 'w') as f:
        json.dump(solution, f)


def main():
    parser = argparse.ArgumentParser(description='Solve electricity system planning problem')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the instance JSON file')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to save the solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds')
    parser.add_argument('--log_path', type=str, default=None, help='Path to log intermediate solutions')
    
    args = parser.parse_args()
    
    solve_instance(args.instance_path, args.solution_path, args.time_limit, args.log_path)


if __name__ == '__main__':
    main()
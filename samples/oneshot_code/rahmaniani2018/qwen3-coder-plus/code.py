import argparse
import json
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from typing import List, Dict, Union
from itertools import product
from solution_logger import SolutionLogger  # Pre-installed in environment


def parse_args():
    parser = argparse.ArgumentParser(description='Solve the least-cost electricity expansion planning problem')
    parser.add_argument('--instance_path', required=True, help='Path to the input instance JSON file')
    parser.add_argument('--solution_path', required=True, help='Path to write the solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Maximum runtime in seconds')
    parser.add_argument('--log_path', type=str, help='Path to JSONL file for logging intermediate solutions')
    return parser.parse_args()


def create_model(instance_data):
    """
    Create Gurobi model based on instance data.
    This function creates the mathematical optimization model for the generator expansion planning problem.
    """
    pdims = instance_data['problem_dimensions']
    
    # Extract dimensions
    num_zones = pdims['num_zones']
    num_weeks = pdims['num_weeks']
    hours_per_week = pdims['hours_per_week']
    total_timesteps = pdims['total_timesteps']
    num_generators = pdims['num_generators']
    num_UC_generators = pdims['num_UC_generators']
    num_storage = pdims['num_storage_resources']
    num_hydro = pdims['num_hydro_resources']
    num_lines = pdims['num_transmission_lines']
    num_segments = pdims['num_demand_segments']
    
    num_hours = total_timesteps
    
    # Policy settings
    RPS_enabled = instance_data['policy']['RPS_enabled']
    RPS_share = instance_data['policy']['RPS_share']
    CO2_cap_enabled = instance_data['policy']['CO2_cap_enabled']
    CO2_cap_tons_per_MWh = instance_data['policy']['CO2_cap_tons_per_MWh']
    RPS_noncompliance_cost_per_MWh = instance_data['policy']['RPS_noncompliance_cost_per_MWh']
    CO2_noncompliance_cost_per_ton = instance_data['policy']['CO2_noncompliance_cost_per_ton']
    
    # Subperiod settings
    num_subperiods = instance_data['subperiods']['num_subperiods']
    hours_per_subperiod = instance_data['subperiods']['hours_per_subperiod']
    subperiod_weights = instance_data['subperiods']['subperiod_weights']
    circular_indexing = instance_data['subperiods']['circular_indexing']
    
    # Initialize lists/indices
    ZONES = list(range(num_zones))
    HOURS = list(range(num_hours))
    GENERATORS = list(range(num_generators))
    UC_GENS = [i for i, gen in enumerate(instance_data['generators']) if gen['is_UC']]
    STORAGE_GENS = [i for i, gen in enumerate(instance_data['generators']) if gen['is_storage']]
    HYDRO_GENS = [i for i, gen in enumerate(instance_data['generators']) if gen['is_hydro']]
    LINES = list(range(num_lines))
    NSE_SEGS = list(range(num_segments))
    
    # Calculate subperiod mapping
    period_map = []
    for p_idx in range(num_subperiods):
        start_time = p_idx * hours_per_subperiod
        end_time = start_time + hours_per_subperiod
        period_map.append(list(range(start_time, end_time)))
    
    # Generator data mappings
    gen_data = {i: instance_data['generators'][i] for i in GENERATORS}
    line_data = {i: instance_data['transmission_lines'][i] for i in LINES}
    
    # Demand data - convert string keys to integers
    demand_dict = {int(zone): np.array(demand) for zone, demand in instance_data['demand'].items()}
    
    # Create mapping from each zone to generators/loads in that zone
    zone_to_generators = {z: [] for z in ZONES}
    for gen_idx, gen_info in enumerate(instance_data['generators']):
        z = gen_info['zone']
        zone_to_generators[z].append(gen_idx)
    
    # Get availability profiles - note we need this before creating the model variables
    availabilities = {}
    for gen_id_str, profile in instance_data['availability_profiles'].items():
        gen_id = int(gen_id_str)
        if profile['type'] == 'constant':
            profile_values = [profile['value']] * num_hours
        else:
            profile_values = profile['values']
        availabilities[gen_id] = np.array(profile_values)
    
    # Calculate hourly weights
    hourly_weights = []
    for p_idx, subperiod_weight in enumerate(subperiod_weights):
        hourly_weight = subperiod_weight / hours_per_subperiod
        for _ in range(hours_per_subperiod):
            hourly_weights.append(hourly_weight)
    
    # Calculate total weighted demand
    total_weighted_demand = 0.0
    for h_idx in HOURS:
        for z_idx in ZONES:
            total_weighted_demand += hourly_weights[h_idx] * demand_dict[z_idx][h_idx]
    
    # Create model
    m = gp.Model("generation_expansion_planning")
    
    # Investment decision variables
    y_P_new = {}
    y_P_ret = {}
    
    for g_idx in GENERATORS:
        gen = gen_data[g_idx]
        capacity_size = max(gen['capacity_size_MW'], 1e-9)  # Avoid division by zero
        if capacity_size <= 0:
            capacity_size = 1  # Default to 1 if unavailable
        
        # Compute max units allowed
        max_units_build = int(gen['max_capacity_MW'] / capacity_size) if gen['max_capacity_MW'] > 0 else 0
        max_units_retire = int(gen['existing_capacity_MW'] / capacity_size) if gen['existing_capacity_MW'] > 0 else 0
        
        # Ensure we have valid bounds
        max_units_build = max(0, max_units_build)
        max_units_retire = max(0, max_units_retire)
        
        y_P_new[g_idx] = m.addVar(vtype=GRB.INTEGER, name=f"y_P_new_{g_idx}", lb=0, ub=max_units_build)
        y_P_ret[g_idx] = m.addVar(vtype=GRB.INTEGER, name=f"y_P_ret_{g_idx}",
                                  lb=0, ub=max_units_retire if gen['can_retire'] else 0)
    
    # Storage energy investment decision variables
    y_E_new = {}
    y_E_ret = {}
    
    for g_idx in STORAGE_GENS:
        gen = gen_data[g_idx]
        storage_capacity_size = max(gen['storage_capacity_size_MWh'], 1e-9)  # Avoid division by zero
        if storage_capacity_size <= 0:
            storage_capacity_size = 1  # Default to 1 if not available
        
        max_units_build = int(gen['max_storage_capacity_MWh'] / storage_capacity_size) if gen['max_storage_capacity_MWh'] > 0 else 0
        max_units_retire = int(gen['existing_storage_capacity_MWh'] / storage_capacity_size) if gen['existing_storage_capacity_MWh'] > 0 else 0
        
        max_units_build = max(0, max_units_build)
        max_units_retire = max(0, max_units_retire)
        
        # Correct the dictionary keys to use g_idx instead of gen_id
        y_E_new[g_idx] = m.addVar(vtype=GRB.INTEGER, name=f"y_E_new_{g_idx}", lb=0, ub=max_units_build)
        y_E_ret[g_idx] = m.addVar(vtype=GRB.INTEGER, name=f"y_E_ret_{g_idx}", 
                                 lb=0, ub=max_units_retire)
    
    # Transmission expansion decisions
    y_F_new = {}
    
    for l_idx in LINES:
        line = line_data[l_idx]
        max_new_units = line['max_new_capacity_MW']
        y_F_new[l_idx] = m.addVar(vtype=GRB.CONTINUOUS, name=f"y_F_new_{l_idx}", 
                                 lb=0, ub=max_new_units)
    
    # Operational variables
    p_gen = {}  # Generation from resources
    p_withdraw = {}  # Withdrawal (charging for storage)
    p_flow = {}  # Power flows
    p_nse = {}  # Non-served energy
    
    # Variables for unit commitment
    u_commit = {}  # Committed units
    u_start = {}  # Started up units
    u_shutdown = {}  # Shut down units
    
    # State of charge and reservoir levels
    soc = {}  # State of charge for storage
    res_level = {}  # Reservoir level for hydro
    
    # Spillage for hydro
    p_spill = {}
    
    # Policy noncompliance slacks
    slack_rps = m.addVar(name="slack_rps", vtype=GRB.CONTINUOUS, lb=0) if RPS_enabled else None
    slack_co2 = m.addVar(name="slack_co2", vtype=GRB.CONTINUOUS, lb=0) if CO2_cap_enabled else None
    
    # Generate operational variables
    for h_idx in HOURS:
        for g_idx in GENERATORS:
            p_gen[(g_idx, h_idx)] = m.addVar(vtype=GRB.CONTINUOUS, name=f"p_gen_{g_idx}_{h_idx}", lb=0)
        
        for g_idx in STORAGE_GENS:
            p_withdraw[(g_idx, h_idx)] = m.addVar(vtype=GRB.CONTINUOUS, name=f"p_withdraw_{g_idx}_{h_idx}", lb=0)
        
        for l_idx in LINES:
            line = line_data[l_idx]
            p_flow[(l_idx, h_idx)] = m.addVar(vtype=GRB.CONTINUOUS, 
                                            name=f"p_flow_{l_idx}_{h_idx}", 
                                            lb=-(line['existing_capacity_MW'] + y_F_new[l_idx]),
                                            ub=(line['existing_capacity_MW'] + y_F_new[l_idx]))
        
        for z_idx in ZONES:
            for nse_idx in NSE_SEGS:
                p_nse[(z_idx, nse_idx, h_idx)] = m.addVar(vtype=GRB.CONTINUOUS, name=f"p_nse_{z_idx}_{nse_idx}_{h_idx}",
                                                        lb=0,
                                                        ub=instance_data['nse_segments'][nse_idx]['max_frac'] * demand_dict[z_idx][h_idx])
    
    # Unit commitment variables for eligible generators
    # Add these variables after defining their constraints for capacity limits
    for g_idx in UC_GENS:
        unit_capacity_mw = gen_data[g_idx]['capacity_size_MW']
        max_cap_mw = gen_data[g_idx]['existing_capacity_MW'] + y_P_new[g_idx]*unit_capacity_mw  # Max possible capacity
        if unit_capacity_mw > 0:
            max_units_poss = int(max_cap_mw/unit_capacity_mw) + 10 # Add padding to stay safe 
        else:
            max_units_poss = 1000  # Some large default number if capacity per unit unavailable
        
        for h_idx in HOURS:
            u_commit[(g_idx, h_idx)] = m.addVar(vtype=GRB.INTEGER, name=f"u_commit_{g_idx}_{h_idx}", 
                                               lb=0, ub=max_units_poss)
            u_start[(g_idx, h_idx)] = m.addVar(vtype=GRB.INTEGER, name=f"u_start_{g_idx}_{h_idx}", lb=0, ub=max_units_poss)
            u_shutdown[(g_idx, h_idx)] = m.addVar(vtype=GRB.INTEGER, name=f"u_shutdown_{g_idx}_{h_idx}", lb=0, ub=max_units_poss)
    
    # State of charge for storage
    for g_idx in STORAGE_GENS:
        for h_idx in HOURS:
            soc[(g_idx, h_idx)] = m.addVar(vtype=GRB.CONTINUOUS, name=f"soc_{g_idx}_{h_idx}", lb=0)
    
    # Reservoir levels for hydro
    for g_idx in HYDRO_GENS:
        for h_idx in HOURS:
            res_level[(g_idx, h_idx)] = m.addVar(vtype=GRB.CONTINUOUS, name=f"res_level_{g_idx}_{h_idx}", lb=0)
            p_spill[(g_idx, h_idx)] = m.addVar(vtype=GRB.CONTINUOUS, name=f"p_spill_{g_idx}_{h_idx}", lb=0)
    
    # Add constraints
    
    # Define total capacities post investment as expressions
    cap_total = {}
    energy_cap_total = {}

    for g_idx in GENERATORS:
        cap_total[g_idx] = gen_data[g_idx]['existing_capacity_MW'] + (y_P_new[g_idx] - y_P_ret[g_idx]) * gen_data[g_idx]['capacity_size_MW']

    for g_idx in STORAGE_GENS:
        energy_cap_total[g_idx] = (gen_data[g_idx]['existing_storage_capacity_MWh'] + 
                                  (y_E_new[g_idx] - y_E_ret[g_idx]) * gen_data[g_idx]['storage_capacity_size_MWh'])
    for g_idx in HYDRO_GENS:
        duration = gen_data[g_idx]['duration_MWh_per_MW']
        energy_cap_total[g_idx] = (gen_data[g_idx]['existing_capacity_MW'] + 
                                  (y_P_new[g_idx] - y_P_ret[g_idx]) * gen_data[g_idx]['capacity_size_MW']) * duration
    # Create placeholders for other gens
    for g_idx in [g for g in GENERATORS if g not in STORAGE_GENS and g not in HYDRO_GENS]:
        energy_cap_total[g_idx] = 0  # Just an initial placeholder


    # 1. Capacity bounds for storage durations
    for g_idx in STORAGE_GENS:
        gen = gen_data[g_idx]
        min_dur = gen.get('min_duration_MWh_per_MW', 0)
        max_dur = gen.get('max_duration_MWh_per_MW', 1e6)
        
        # Check the bounds before applying them to avoid impossible constraints
        if min_dur > 0:  # Only apply bounds if needed
            min_energy = min_dur * cap_total[g_idx]
            max_energy = max_dur * cap_total[g_idx]
            
            m.addConstr(energy_cap_total[g_idx] >= min_energy, name=f"min_dur_{g_idx}")
            m.addConstr(energy_cap_total[g_idx] <= max_energy, name=f"max_dur_{g_idx}")
    
    # 2. Power balance constraint for each zone and hour
    for z_idx in ZONES:
        for h_idx in HOURS:
            lhs_gen = gp.quicksum(p_gen[(g_idx, h_idx)] for g_idx in zone_to_generators[z_idx])
            lhs_withdraw = gp.quicksum(-p_withdraw[(g_idx, h_idx)] for g_idx in zone_to_generators[z_idx] if g_idx in STORAGE_GENS)
            
            # Subtract flows outgoing from this zone, add flows incoming to this zone  
            flow_expr = gp.LinExpr()
            for l_idx in LINES:
                line = line_data[l_idx]
                if line['from_zone'] == z_idx:  # Flow goes from this zone to another -> subtract
                    flow_expr.add(-p_flow[(l_idx, h_idx)])
                elif line['to_zone'] == z_idx:  # Flow goes to this zone from another -> add
                    flow_expr.add(p_flow[(l_idx, h_idx)])
                    
            nse_sum = gp.quicksum(p_nse[(z_idx, nse_seg, h_idx)] for nse_seg in NSE_SEGS)
            
            demand_z_h = float(demand_dict[z_idx][h_idx])  # Make sure it's a float not an array
            
            # Net balance equation: generation - withdrawal + net inflow + nse = demand
            m.addConstr(lhs_gen + lhs_withdraw + flow_expr + nse_sum == demand_z_h, name=f"power_balance_{z_idx}_{h_idx}")
    
    # 3. Generator output limits and rules
    for g_idx in GENERATORS:
        gen = gen_data[g_idx]
        g_availabilities = availabilities[g_idx]
        
        for h_idx in HOURS:
            avail_factor = float(g_availabilities[h_idx])  # Ensure it's a scalar
            
            if gen['is_UC']:
                # Unit commitment constraints
                m.addConstr(p_gen[(g_idx, h_idx)] >= u_commit[(g_idx, h_idx)] * gen['min_output_frac'] * gen['capacity_size_MW'],
                           name=f"uc_min_out_{g_idx}_{h_idx}")
                
                m.addConstr(p_gen[(g_idx, h_idx)] <= u_commit[(g_idx, h_idx)] * gen['capacity_size_MW'] * avail_factor,
                           name=f"uc_max_out_{g_idx}_{h_idx}")
                            
                # Committed units cannot exceed physical capacity: cap_total = existing + built*MW/unit - ret*MW/unit
                m.addConstr(u_commit[(g_idx, h_idx)] * gen['capacity_size_MW'] <= cap_total[g_idx],
                           name=f"uc_capacity_limit_{g_idx}_{h_idx}")
                           
            elif gen['is_storage']:
                # Storage resource capacity and operational constraints
                m.addConstr(p_withdraw[(g_idx, h_idx)] <= cap_total[g_idx] * avail_factor,
                           name=f"storage_max_withd_{g_idx}_{h_idx}")
                m.addConstr(p_gen[(g_idx, h_idx)] + p_withdraw[(g_idx, h_idx)] <= cap_total[g_idx],
                           name=f"storage_total_limit_{g_idx}_{h_idx}")
                           
                # Charging and state-of-charge relationships
                m.addConstr(soc[(g_idx, h_idx)] <= energy_cap_total[g_idx], 
                           name=f"storage_soc_limit_{g_idx}_{h_idx}")
                m.addConstr(p_gen[(g_idx, h_idx)] <= soc[(g_idx, h_idx)] / gen['discharge_efficiency'],
                           name=f"storage_discharge_limit_{g_idx}_{h_idx}")
                m.addConstr(p_withdraw[(g_idx, h_idx)] * gen['charge_efficiency'] <= 
                                energy_cap_total[g_idx] - soc[(g_idx, h_idx)],
                           name=f"storage_charge_limit_{g_idx}_{h_idx}")
                           
            elif gen['is_hydro']:
                duration = gen.get('duration_MWh_per_MW', 1.0)
                # Hydro generator constraints
                m.addConstr(res_level[(g_idx, h_idx)] <= duration * cap_total[g_idx],  # Corrected
                           name=f"hyd_res_max_{g_idx}_{h_idx}")
                m.addConstr(p_gen[(g_idx, h_idx)] + p_spill[(g_idx, h_idx)] >= 
                            gen['min_output_frac'] * cap_total[g_idx],
                           name=f"hyd_min_gen_{g_idx}_{h_idx}")
            
            else:  # Regular generator (not UC, not storage, not hydro)
                # Availability and output constraints for regular generators (like renewables)
                m.addConstr(p_gen[(g_idx, h_idx)] <= cap_total[g_idx] * avail_factor,
                           name=f"gen_max_out_{g_idx}_{h_idx}")
                           
                # Only apply minimum output constraint if it's non-trivial
                if gen.get('min_output_frac', 0) > 0:
                    m.addConstr(p_gen[(g_idx, h_idx)] >= cap_total[g_idx] * gen['min_output_frac'],
                               name=f"gen_min_out_{g_idx}_{h_idx}")
        
        # Add ramping for non-UC, non-storage, non-hydro generators
        if not gen['is_UC'] and not gen['is_storage'] and not gen['is_hydro']:
            if 'ramp_up_frac_per_hr' in gen and 'ramp_dn_frac_per_hr' in gen:
                if gen['ramp_up_frac_per_hr'] is not None and gen['ramp_dn_frac_per_hr'] is not None:
                    for h_idx in range(0, num_hours - 1):
                        next_h = h_idx + 1
                        # Ramping constraints: diff in gen less than up/down rates * cap
                        m.addConstr(p_gen[(g_idx, next_h)] - p_gen[(g_idx, h_idx)] <= 
                                   cap_total[g_idx] * gen['ramp_up_frac_per_hr'],
                                   name=f"ramp_up_{g_idx}_{h_idx}")
                        m.addConstr(p_gen[(g_idx, h_idx)] - p_gen[(g_idx, next_h)] <= 
                                   cap_total[g_idx] * gen['ramp_dn_frac_per_hr'],
                                   name=f"ramp_dn_{g_idx}_{h_idx}")

                    # Handle circular ramping at periods
                    if circular_indexing:
                        for p_idx in range(num_subperiods):
                            period_start = p_idx * hours_per_subperiod
                            period_end = (p_idx + 1) * hours_per_subperiod - 1
                            
                            h_curr = period_end
                            h_next = period_start  # Wrap around
                            m.addConstr(p_gen[(g_idx, h_next)] - p_gen[(g_idx, h_curr)] <= 
                                       cap_total[g_idx] * gen['ramp_up_frac_per_hr'],
                                       name=f"ramp_up_circ_{g_idx}_{p_idx}")
                            m.addConstr(p_gen[(g_idx, h_curr)] - p_gen[(g_idx, h_next)] <= 
                                       cap_total[g_idx] * gen['ramp_dn_frac_per_hr'],
                                       name=f"ramp_dn_circ_{g_idx}_{p_idx}")

    # State of charge dynamics for storage with circular indexing
    for g_idx in STORAGE_GENS:
        gen = gen_data[g_idx]
        for p_idx in range(num_subperiods):
            period_start = p_idx * hours_per_subperiod
            period_end = (p_idx + 1) * hours_per_subperiod - 1
            
            # Iterate through all hours in the period with circular treatment
            for offset in range(hours_per_subperiod):
                h_idx = period_start + offset
                if offset == 0 and circular_indexing:
                    prev_h = period_end  # Wrap around to last hour of this period
                elif offset == 0 and not circular_indexing:
                    prev_h = h_idx  # Same hour 
                else:
                    prev_h = h_idx - 1
                
                # SOC[t] = SOC[t-1] + charge_eff*withdraw[t] - discharge_eff*gen[t] - self_disc*SOC[t-1]
                # Simplified form: SOC[t] = SOC[t-1]*(1-self_disc) + c_eff*withd - gen/d_eff
                m.addConstr(soc[(g_idx, h_idx)] == 
                           soc[(g_idx, prev_h)]*(1-gen['self_discharge_rate']) + 
                           p_withdraw[(g_idx, h_idx)] * gen['charge_efficiency'] - 
                           p_gen[(g_idx, h_idx)] / gen['discharge_efficiency'],
                           name=f"storage_dynamics_{g_idx}_{h_idx}")
    
    # Reservoir dynamics for hydro with circular indexing
    for g_idx in HYDRO_GENS:
        gen = gen_data[g_idx]
        inflow_profile = availabilities[g_idx]
        duration = gen.get('duration_MWh_per_MW', 1.0)
        for p_idx in range(num_subperiods):
            period_start = p_idx * hours_per_subperiod
            period_end = (p_idx + 1) * hours_per_subperiod - 1
            
            for offset in range(hours_per_subperiod):
                h_idx = period_start + offset
                if offset == 0 and circular_indexing:
                    prev_h = period_end  # Wrap around
                elif offset == 0 and not circular_indexing:
                    prev_h = h_idx
                else:
                    prev_h = h_idx - 1
                
                # Inflow is based on power capacity * availability fraction (which is normalized inflow in hydro context)
                current_inflow_mwh = cap_total[g_idx] * inflow_profile[prev_h]  # Apply to prev_h as per typical dynamic
                
                # Reservoir level updates with: inflow - generation - spillage
                m.addConstr(res_level[(g_idx, h_idx)] == 
                           res_level[(g_idx, prev_h)] + 
                           current_inflow_mwh - 
                           p_gen[(g_idx, h_idx)] - 
                           p_spill[(g_idx, h_idx)],
                           name=f"hyd_dynamics_{g_idx}_{h_idx}")
    
    # Unit commitment logical constraints (transition of committed units)
    for g_idx in UC_GENS:
        for p_idx in range(num_subperiods):
            period_start = p_idx * hours_per_subperiod
            period_end = (p_idx + 1) * hours_per_subperiod - 1
            
            for offset in range(hours_per_subperiod):
                h_idx = period_start + offset
                if offset == 0 and circular_indexing:
                    prev_h = period_end
                elif offset == 0 and not circular_indexing:
                    prev_h = h_idx
                else:
                    prev_h = h_idx - 1
                
                # Transition constraint: change in commitment equals starts minus shutdowns
                m.addConstr(u_commit[(g_idx, h_idx)] == u_commit[(g_idx, prev_h)] + u_start[(g_idx, h_idx)] - u_shutdown[(g_idx, h_idx)],
                           name=f"uc_transition_{g_idx}_{h_idx}")
    
    # Unit commitment ramping constraints (tighter and more detailed)
    for g_idx in UC_GENS:
        gen = gen_data[g_idx]
        cap_size = gen['capacity_size_MW']
        ramp_up_rate = gen['ramp_up_frac_per_hr']
        ramp_dn_rate = gen['ramp_dn_frac_per_hr']
        avail = gen['availability_fraction']
        min_out = gen['min_output_frac']
        
        for h_idx in HOURS:
            # For ramp constraints, we get the previous hour
            if h_idx == 0 and circular_indexing:
                prev_h_idx = num_hours - 1
            elif h_idx == 0 and not circular_indexing:
                prev_h_idx = 0  # Use same for non-circular
            else:
                prev_h_idx = h_idx - 1
            
            # Calculate units that were running last period but not started this period
            # units_running_not_started = u_commit[(g_idx, prev_h_idx)] - u_start[(g_idx, h_idx)]
            # units_just_starting = u_start[(g_idx, h_idx)]
            # units_just_shutting = u_shutdown[(g_idx, h_idx)]  
            
            # Calculate ramps based on different types of units
            
            # Define helper indicators
            m._gen = gen  # Store temporarily in model if needed elsewhere
            started_this_hour = u_start[(g_idx, h_idx)]
            shutdown_this_hour = u_shutdown[(g_idx, h_idx)]
            was_committed_prev = u_commit[(g_idx, prev_h_idx)]
            is_committed_now = u_commit[(g_idx, h_idx)]
            
            # Upper bound on ramp-up: based on the units still running and newly started
            # Increase = increase from continued units + from started units (minus decrease from shutdowns)
            # This is a linearization of the concept from the math - we need tighter constraints:
            
            # Simplified version: p_gen[now] - p_gen[prev] <= 
            #                     ramp_up * cap_size * (committed_not_newly_started) +
            #                     ramp_up * cap_size * (newly_started) - 
            #                     min_gen * cap_size * (newly_shutdown)
            m.addConstr(p_gen[(g_idx, h_idx)] - p_gen[(g_idx, prev_h_idx)] <= 
                       cap_size * ramp_up_rate * (was_committed_prev) +  # Conservative bound
                       cap_size * min(max(avail, ramp_up_rate), min_out) * started_this_hour -
                       cap_size * min_out * shutdown_this_hour,
                       name=f"uc_ramp_up_{g_idx}_{h_idx}")
            
            m.addConstr(p_gen[(g_idx, prev_h_idx)] - p_gen[(g_idx, h_idx)] <= 
                       cap_size * ramp_dn_rate * (is_committed_now) + # Use current committed units
                       cap_size * min(max(avail, ramp_dn_rate), min_out) * shutdown_this_hour -
                       cap_size * min_out * started_this_hour,
                       name=f"uc_ramp_dn_{g_idx}_{h_idx}")

    # Unit commitment minimum up/down time constraints
    # Note: This uses circular indexing but makes sure not to cross period boundaries incorrectly
    for g_idx in UC_GENS:
        gen = gen_data[g_idx]
        min_up_time = gen.get('min_up_time_hr', 0)
        min_down_time = gen.get('min_down_time_hr', 0)
        
        # Minimum up-time constraint: if a unit starts up it stays up for at least min_up_time
        if min_up_time > 0 and circular_indexing:
            for h_idx in HOURS:
                # Only enforce within the same subperiod
                current_period = h_idx // hours_per_subperiod
                # Sum the starts in the last min_up_time hours
                starts_sum = gp.LinExpr(0)
                for back_step in range(min_up_time):
                    check_hour = (h_idx - back_step) % num_hours  # circular indexing
                    # Make sure we're checking within same period as h_idx 
                    if check_hour // hours_per_subperiod == current_period:
                        starts_sum.addTerms(1, u_start[(g_idx, check_hour)])
                    else:
                        break
                # Number of currently running units must be at least the total startups that occurred in previous up_period hours
                m.addConstr(u_commit[(g_idx, h_idx)] >= starts_sum, name=f"min_up_time_{g_idx}_{h_idx}")
                
        # Minimum down-time constraint (optional - more complex to implement fully)
        if min_down_time > 0 and circular_indexing:
            for h_idx in HOURS:
                current_period = h_idx // hours_per_subperiod
                shuts_sum = gp.LinExpr(0)
                for forward_step in range(min_down_time):
                    check_hour = (h_idx + forward_step) % num_hours
                    if check_hour // hours_per_subperiod == current_period:
                        shuts_sum.addTerms(1, u_shutdown[(g_idx, check_hour)])
                    else:
                        break
                # The number of committed units + shutting ones in near future shouldn't exceed total available units
                total_available = (gen_data[g_idx]['existing_capacity_MW'] + 
                                  y_P_new[g_idx]*gen_data[g_idx]['capacity_size_MW']) / gen['capacity_size_MW']
                m.addConstr(u_commit[(g_idx, h_idx)] + shuts_sum <= total_available,
                           name=f"min_down_time_{g_idx}_{h_idx}")

    # Transmission capacity limits
    for l_idx in LINES:
        for h_idx in HOURS:
            m.addConstr(p_flow[(l_idx, h_idx)] <= (line_data[l_idx]['existing_capacity_MW'] + y_F_new[l_idx]),
                       name=f"net_trm_max_{l_idx}_{h_idx}")
            m.addConstr(p_flow[(l_idx, h_idx)] >= -(line_data[l_idx]['existing_capacity_MW'] + y_F_new[l_idx]),
                       name=f"net_trm_min_{l_idx}_{h_idx}")
    
    # Non-served energy constraints
    for z_idx in ZONES:
        for h_idx in HOURS:
            for nse_idx in NSE_SEGS:
                max_frac = instance_data['nse_segments'][nse_idx]['max_frac']
                max_allowed = float(demand_dict[z_idx][h_idx]) * max_frac
                m.addConstr(p_nse[(z_idx, nse_idx, h_idx)] <= max_allowed,
                           name=f"nse_max_{z_idx}_{nse_idx}_{h_idx}")
    
    # Policy constraints
    # Renewable Portfolio Standard
    if RPS_enabled:
        # Calculate eligible generation weighted across time
        rps_eligible_gen = gp.LinExpr(0)
        for g_idx in GENERATORS:
            if gen_data[g_idx]['is_RPS_qualifying']:
                for h_idx in HOURS:
                    rps_eligible_gen.addTerms(hourly_weights[h_idx], p_gen[(g_idx, h_idx)])

        m.addConstr(rps_eligible_gen + slack_rps >= RPS_share * total_weighted_demand, name="rps_constraint")
    
    # CO2 constraint
    if CO2_cap_enabled:
        # Calculate emissions weighted across time
        co2_emissions_expr = gp.LinExpr(0)
        for g_idx in GENERATORS:
            emission_factor = gen_data[g_idx]['co2_tons_per_MWh']
            for h_idx in HOURS:
                co2_emissions_expr.addTerms(hourly_weights[h_idx] * emission_factor, p_gen[(g_idx, h_idx)])
                if g_idx in STORAGE_GENS and gen_data[g_idx].get('var_cost_per_MWh', 0) > 0: # Add to emission if there's a cost for withdraw
                    co2_emissions_expr.addTerms(hourly_weights[h_idx] * emission_factor, p_withdraw[(g_idx, h_idx)])

        # Calculate emission cap
        emission_cap_actual = 0.05 * total_weighted_demand if CO2_cap_tons_per_MWh is None else (
            CO2_cap_tons_per_MWh * total_weighted_demand)
            
        m.addConstr(co2_emissions_expr - slack_co2 <= emission_cap_actual, name="co2_constraint")

    # Create objective function    
    fixed_costs = gp.LinExpr(0)
    var_costs = gp.LinExpr(0)
    nse_costs = gp.LinExpr(0)
    startup_costs = gp.LinExpr(0)
    rps_penalty = 0
    co2_penalty = 0
    
    # Fixed investment and O&M costs for all generators
    for g_idx in GENERATORS:
        gen = gen_data[g_idx]
        
        # Power capacity fixed investment and O&M costs
        # Investment: (units_built * cap_size) * inv_cost_per_MW_yr
        fixed_costs.addTerms(gen['capacity_size_MW'] * gen['inv_cost_per_MW_yr'], y_P_new[g_idx])
        
        # O&M: total power capacity * fom_cost_per_MW_yr
        # Total power capacity after investment
        fixed_costs.addTerms(gen['fom_cost_per_MW_yr'], cap_total[g_idx])

        # Special handling for hydro (has energy investment & O&M based on power capacity and duration)
        if gen['is_hydro']:
            duration = gen.get('duration_MWh_per_MW', 1.0)
            energy_inv_cost = gen.get('inv_cost_energy_per_MWh_yr', 0)
            energy_fom_cost = gen.get('fom_cost_energy_per_MWh_yr', 0)
            
            if energy_inv_cost > 0:
                # Investment cost per MWh * duration(MWh/MW) => Inv per MW 
                fixed_costs.addTerms(gen['capacity_size_MW'] * duration * energy_inv_cost, y_P_new[g_idx])
            if energy_fom_cost > 0:
                # Total energy capacity is (cap_total[g_idx] * duration)
                fixed_costs.addTerms(duration * energy_fom_cost, cap_total[g_idx])
    
    # Energy (storage) investment and O&M for storage resources
    for g_idx in STORAGE_GENS:
        gen = gen_data[g_idx]
        # Investment: units_built * storage_capacity_size * energy_inv_cost
        fixed_costs.addTerms(gen['storage_capacity_size_MWh'] * gen['inv_cost_energy_per_MWh_yr'], y_E_new[g_idx])

        # O&M on total capacity: total_energy capacity
        fixed_costs.addTerms(gen['fom_cost_energy_per_MWh_yr'], energy_cap_total[g_idx])
    
    # Transmission investment
    for l_idx in LINES:
        line = line_data[l_idx]
        fixed_costs.addTerms(line['inv_cost_per_MW_yr'], y_F_new[l_idx])

    # Variable production and withdrawal costs
    for h_idx in HOURS:
        weight = hourly_weights[h_idx]
        # Production costs for all gen units
        for g_idx in GENERATORS:
            var_costs.addTerms(weight * gen_data[g_idx]['var_cost_per_MWh'], p_gen[(g_idx, h_idx)])
        
        # Withdrawal cost for storage units 
        for g_idx in STORAGE_GENS:
            var_costs.addTerms(weight * gen_data[g_idx].get('var_cost_per_MWh', 0), p_withdraw[(g_idx, h_idx)])
    
    # Non-served energy costs
    for z_idx in ZONES:
        for nse_idx in NSE_SEGS:
            cost_segment = instance_data['nse_segments'][nse_idx]['cost_per_MWh']
            for h_idx in HOURS:
                nse_costs.addTerms(hourly_weights[h_idx] * cost_segment, p_nse[(z_idx, nse_idx, h_idx)])
    
    # Startup costs for Unit Commitment units
    for g_idx in UC_GENS:
        cost_st = gen_data[g_idx]['start_cost_per_unit']
        for h_idx in HOURS:
            startup_costs.addTerms(hourly_weights[h_idx] * cost_st, u_start[(g_idx, h_idx)])
    
    # Penalties for violation of policies
    if RPS_enabled and slack_rps is not None:
        rps_penalty = RPS_noncompliance_cost_per_MWh * slack_rps
        
    if CO2_cap_enabled and slack_co2 is not None:
        co2_penalty = CO2_noncompliance_cost_per_ton * slack_co2

    # Combine full objective
    m.setObjective(fixed_costs + var_costs + startup_costs + nse_costs + rps_penalty + co2_penalty, GRB.MINIMIZE)

    # Return the structured data along with model
    return m, {
        'model': m,
        'gen_data': gen_data,
        'line_data': line_data,
        'demand_dict': demand_dict,
        'GENERATORS': GENERATORS,
        'ZONES': ZONES,
        'HOURS': HOURS,
        'LINES': LINES,
        'STORAGE_GENS': STORAGE_GENS,
        'UC_GENS': UC_GENS,
        'HYDRO_GENS': HYDRO_GENS,
        'NSE_SEGS': NSE_SEGS,
        'CAP_TOTAL': cap_total,
        'ENERGY_CAP_TOTAL': energy_cap_total,
        
        # Decision vars
        'y_P_new': y_P_new,
        'y_P_ret': y_P_ret,
        'y_E_new': y_E_new if y_E_new else {},
        'y_E_ret': y_E_ret if y_E_ret else {},
        'y_F_new': y_F_new,
        'p_flow': p_flow,
        'p_gen': p_gen,
        'p_withdraw': p_withdraw if p_withdraw else {},
        'p_nse': p_nse,
        'u_commit': u_commit if u_commit else {},
        'u_start': u_start if u_start else {},
        'u_shutdown': u_shutdown if u_shutdown else {},
        'soc': soc if soc else {},
        'res_level': res_level if res_level else {},
        'p_spill': p_spill if p_spill else {},
        'slack_rps': slack_rps if RPS_enabled else None,
        'slack_co2': slack_co2 if CO2_cap_enabled else None,
        
        'hourly_weights': hourly_weights,
        'RPS_ENABLED': RPS_enabled,
        'CO2_CAP_ENABLED': CO2_cap_enabled,
        'total_weighted_demand': total_weighted_demand
    }


def solve_instance(model_data, time_limit):
    m, d = model_data
    
    # Set time limit and other parameters
    m.setParam(GRB.Param.TimeLimit, time_limit)
    m.setParam(GRB.Param.OutputFlag, 1)
    m.setParam(GRB.Param.FeasibilityTol, 1e-6)
    m.setParam(GRB.Param.OptimalityTol, 1e-6)
    
    # Optimize
    m.optimize()
    
    return m, d


def extract_solution(m, d):
    """
    Extract solution values from optimized model.
    """
    # Build investment decisions
    inv_decisions = []
    for g_idx in d['GENERATORS']:
        gen = d['gen_data'][g_idx]
        
        decision_entry = {
            'generator_id': gen['id'],
            'resource_type': gen['resource_type'],
            'zone': gen['zone'],
            'y_P_new': round(d['y_P_new'][g_idx].X),
            'y_P_ret': round(d['y_P_ret'][g_idx].X),
            # Total is exisiting + (built - retired)*size
            'y_P_total_MW': (gen['existing_capacity_MW'] + 
                             (d['y_P_new'][g_idx].X - d['y_P_ret'][g_idx].X) * 
                             gen['capacity_size_MW'])
        }
        
        if g_idx in d['STORAGE_GENS']:
            decision_entry['y_E_new'] = round(d['y_E_new'][g_idx].X)
            decision_entry['y_E_ret'] = round(d['y_E_ret'][g_idx].X)
            decision_entry['y_E_total_MWh'] = (gen['existing_storage_capacity_MWh'] +
                                              (d['y_E_new'][g_idx].X - d['y_E_ret'][g_idx].X) *
                                              gen['storage_capacity_size_MWh'])
        
        inv_decisions.append(decision_entry)
    
    trans_decisions = []
    for l_idx in d['LINES']:
        line = d['line_data'][l_idx]

        trans_entry = {
            'line_id': line['id'],
            'y_F_new_MW': d['y_F_new'][l_idx].X,
            'total_capacity_MW': (line['existing_capacity_MW'] + d['y_F_new'][l_idx].X)
        }
        
        trans_decisions.append(trans_entry)
    
    # Determine proper objective value depending on status
    objval = m.ObjVal if m.SolCount > 0 else float('inf')
    
    solution = {
        'objective_value': objval,
        'investment_decisions': inv_decisions,
        'transmission_decisions': trans_decisions
    }
    
    return solution


def main():
    args = parse_args()
    
    # Read instance
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)
    
    # Setup logger if path given
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    # Callback for logging intermediate solutions (though we'd need more access to mip callbacks for this)
    # For now we just log at end if improved solution seen
    
    # Create model and data structures
    model_data = create_model(instance_data)
    m, d = model_data
    
    # Solve
    m, d = solve_instance(model_data, args.time_limit)
    
    # Log final best solution if feasible
    if logger and m.status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INT_SOLUTION] and m.SolCount > 0:
        logger.log(m.objVal)
    
    # Extract and save solution
    if m.SolCount > 0:
        solution = extract_solution(m, d)
    else:
        # In case no feasible solution was found within time limit, create empty response
        solution = {
            'objective_value': float('inf'),
            'investment_decisions': [],
            'transmission_decisions': []
        }

    # Write solution
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == '__main__':
    main()
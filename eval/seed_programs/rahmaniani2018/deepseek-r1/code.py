import gurobipy as gp
from gurobipy import GRB
import json
import argparse
import math
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    # Load instance data
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    # Extract dimensions and parameters
    dim = instance['problem_dimensions']
    policy = instance['policy']
    subperiods = instance['subperiods']
    generators = instance['generators']
    transmission_lines = instance['transmission_lines']
    demand = instance['demand']
    nse_segments = instance['nse_segments']
    availability_profiles = instance['availability_profiles']
    
    num_subperiods = subperiods['num_subperiods']
    hours_per_subperiod = subperiods['hours_per_subperiod']
    subperiod_weights = subperiods['subperiod_weights']
    circular_indexing = subperiods['circular_indexing']
    total_timesteps = num_subperiods * hours_per_subperiod
    hourly_weights = [weight / hours_per_subperiod for weight in subperiod_weights for _ in range(hours_per_subperiod)]
    
    # Precompute hourly weights and total weighted demand
    total_weighted_demand = 0.0
    for zone, demand_list in demand.items():
        for t, d_val in enumerate(demand_list):
            total_weighted_demand += hourly_weights[t] * d_val
    
    # Create Gurobi model
    model = gp.Model("Electricity_System_Planning")
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', args.time_limit)
    if args.log_path:
        logger = SolutionLogger(args.log_path, sense="minimize")
    else:
        logger = None
        
    # Preprocess generator data
    generators_by_id = {g['id']: g for g in generators}
    availability_by_generator = {}
    for gen in generators:
        gen_id = gen['id']
        profile_key = str(gen_id)
        if profile_key in availability_profiles:
            profile = availability_profiles[profile_key]
            if profile['type'] == 'constant':
                availability_by_generator[gen_id] = [profile['value']] * total_timesteps
            else:  # timeseries
                availability_by_generator[gen_id] = profile['values']
        else:
            # Default to full availability if no profile provided
            availability_by_generator[gen_id] = [1.0] * total_timesteps

    # Mapping for transmission lines
    lines_by_id = {line['id']: line for line in transmission_lines}
    zone_to_outgoing_lines = {}
    zone_to_incoming_lines = {}
    for line in transmission_lines:
        fz, tz = line['from_zone'], line['to_zone']
        if fz not in zone_to_outgoing_lines:
            zone_to_outgoing_lines[fz] = []
        zone_to_outgoing_lines[fz].append(line['id'])
        if tz not in zone_to_incoming_lines:
            zone_to_incoming_lines[tz] = []
        zone_to_incoming_lines[tz].append(line['id'])
    
    # Group generators by zone
    generators_by_zone = {}
    for gen in generators:
        zone = gen['zone']
        if zone not in generators_by_zone:
            generators_by_zone[zone] = []
        generators_by_zone[zone].append(gen['id'])
    
    # =====================
    # Decision Variables
    # =====================
    # Investment variables
    yPnew = {}  # New power capacity units
    yPret = {}  # Retired power capacity units
    yP_total_MW = {}  # Total power capacity after investment
    yEnew = {}  # New energy capacity units (for storage)
    yEret = {}  # Retired energy capacity units (for storage)
    yE_total_MWh = {}  # Total energy capacity after investment
    yFnew = {}  # New transmission capacity (MW)
    
    for gen in generators:
        i = gen['id']
        cap_size = gen['capacity_size_MW']
        existing_P = gen['existing_capacity_MW']
        max_new_units = (gen['max_capacity_MW'] - existing_P) / cap_size
        
        yPnew[i] = model.addVar(
            vtype=GRB.INTEGER, lb=0, ub=max_new_units,
            name=f"yPnew_{i}"
        )
        
        if gen.get('can_retire', True):
            max_ret_units = existing_P / cap_size
            yPret[i] = model.addVar(
                vtype=GRB.INTEGER, lb=0, ub=max_ret_units,
                name=f"yPret_{i}"
            )
        else:
            yPret[i] = 0  # Can't retire
        
        # Expression for total power capacity
        init_units = existing_P / cap_size
        if isinstance(yPret[i], gp.Var):
            yP_total_MW[i] = cap_size * (init_units - yPret[i] + yPnew[i])
        else:
            yP_total_MW[i] = cap_size * (init_units + yPnew[i])
            
        # Storage-specific variables
        if gen.get('is_storage', False):
            storage_cap_size = gen.get('storage_capacity_size_MWh', 0.0)
            exist_storage = gen.get('existing_storage_capacity_MWh', 0.0)
            max_new_E_units = (gen.get('max_storage_capacity_MWh', 0.0) - exist_storage) / storage_cap_size
            
            yEnew[i] = model.addVar(
                vtype=GRB.INTEGER, lb=0, ub=max_new_E_units,
                name=f"yEnew_{i}"
            )
            
            if gen.get('can_retire', True):
                max_ret_E_units = exist_storage / storage_cap_size
                yEret[i] = model.addVar(
                    vtype=GRB.INTEGER, lb=0, ub=max_ret_E_units,
                    name=f"yEret_{i}"
                )
            else:
                yEret[i] = 0
                
            init_units_E = exist_storage / storage_cap_size
            if isinstance(yEret[i], gp.Var):
                yE_total_MWh[i] = storage_cap_size * (init_units_E - yEret[i] + yEnew[i])
            else:
                yE_total_MWh[i] = storage_cap_size * (init_units_E + yEnew[i])
    
    # Transmission investment
    for line in transmission_lines:
        l = line['id']
        yFnew[l] = model.addVar(
            vtype=GRB.INTEGER, lb=0, ub=line['max_new_capacity_MW'],
            name=f"yFnew_{l}"
        )
    
    # Operational variables
    xinj = {}  # Generator injection (MWh)
    xwdw = {}  # Storage withdrawal (MWh)
    xsoc = {}  # State of charge (MWh)
    xcommit = {}  # Committed units (for UC resources)
    xstart = {}  # Units started (for UC resources)
    xshut = {}  # Units shut down (for UC resources)
    xflow = {}  # Transmission flow (MWh)
    xnse = {}  # Non-served energy (MWh)
    xRPS = {}  # RPS slack (MWh, per week)
    xCO2 = {}  # CO2 slack (tons, per week)
    
    # Policy slack variables
    if policy.get('RPS_enabled', False):
        for w in range(num_subperiods):
            xRPS[w] = model.addVar(name=f"xRPS_{w}")
    
    if policy.get('CO2_cap_enabled', False):
        for w in range(num_subperiods):
            xCO2[w] = model.addVar(name=f"xCO2_{w}")
    
    # Create generator operational variables
    for gen in generators:
        i = gen['id']
        is_storage = gen.get('is_storage', False)
        is_uc = gen.get('is_UC', False)
        for t in range(total_timesteps):
            xinj[i, t] = model.addVar(name=f"xinj_{i}_{t}")
            if is_storage:
                xwdw[i, t] = model.addVar(name=f"xwdw_{i}_{t}")
                xsoc[i, t] = model.addVar(name=f"xsoc_{i}_{t}")
            if is_uc:
                cap_size = gen['capacity_size_MW']
                xcommit[i, t] = model.addVar(
                    vtype=GRB.INTEGER, 
                    name=f"xcommit_{i}_{t}"
                )
                xstart[i, t] = model.addVar(
                    vtype=GRB.INTEGER, 
                    name=f"xstart_{i}_{t}"
                )
                xshut[i, t] = model.addVar(
                    vtype=GRB.INTEGER,
                    name=f"xshut_{i}_{t}"
                )
    
    # Transmission flows
    for line in transmission_lines:
        l = line['id']
        for t in range(total_timesteps):
            xflow[l, t] = model.addVar(
                name=f"xflow_{l}_{t}"
            )
    
    # Non-served energy
    for seg in nse_segments:
        seg_id = seg['segment']
        max_frac = seg.get('max_frac', 1.0)
        for zone, demand_list in demand.items():
            z = int(zone)
            for t in range(total_timesteps):
                max_nse = max_frac * demand_list[t]
                xnse[seg_id, z, t] = model.addVar(
                    lb=0, ub=max_nse,
                    name=f"xnse_{seg_id}_{z}_{t}"
                )
    
    # ===================================
    # Constraints
    # ===================================
    # Power balance for each zone and hour
    for zone, demand_list in demand.items():
        z = int(zone)
        for t, d_val in enumerate(demand_list):
            # Generation in this zone
            gen_sum = gp.quicksum(
                xinj[i, t] 
                for i in generators_by_zone.get(z, [])
            )
            
            # Storage charging in this zone
            storage_wdw = gp.quicksum(
                xwdw[i, t] 
                for i in generators_by_zone.get(z, []) 
                if generators_by_id[i].get('is_storage', False)
            )
            
            # Outgoing flows
            flow_out = gp.quicksum(
                xflow[l, t] 
                for l in zone_to_outgoing_lines.get(z, [])
            )
            
            # Incoming flows
            flow_in = gp.quicksum(
                xflow[l, t] 
                for l in zone_to_incoming_lines.get(z, [])
            )
            
            # Non-served energy
            nse_sum = gp.quicksum(
                xnse[seg['segment'], z, t] 
                for seg in nse_segments
            )
            
            model.addConstr(
                gen_sum - storage_wdw - flow_out + flow_in + nse_sum == d_val,
                f"balance_z{z}_t{t}"
            )
    
    # Generator operational constraints
    for gen in generators:
        i = gen['id']
        is_uc = gen.get('is_UC', False)
        is_storage = gen.get('is_storage', False)
        is_hydro = gen.get('is_hydro', False)
        min_out = gen.get('min_output_frac', 0.0)  # Default to 0 if missing
        cap_size = gen['capacity_size_MW']
        avail = availability_by_generator[i]
        total_P = yP_total_MW[i]
        
        for t in range(total_timesteps):
            # General generation constraints
            model.addConstr(
                xinj[i, t] <= avail[t] * total_P,
                f"cap_upper_{i}_{t}"
            )
            
            # Min output for non-UC generators
            if not is_uc and not is_storage and min_out > 0:
                model.addConstr(
                    xinj[i, t] >= min_out * total_P,
                    f"min_out_{i}_{t}"
                )
            
            # Storage-specific constraints
            if is_storage:
                eff_c = gen.get('charge_efficiency', 1.0)
                eff_d = gen.get('discharge_efficiency', 1.0)
                sd_rate = gen.get('self_discharge_rate', 0.0)
                total_E = yE_total_MWh[i]
                
                # Charge/discharge constraints
                model.addConstr(
                    xwdw[i, t] <= avail[t] * total_P,
                    f"wdw_upper_{i}_{t}"
                )
                model.addConstr(
                    xinj[i, t] + xwdw[i, t] <= total_P,
                    f"power_cap_{i}_{t}"
                )
                model.addConstr(
                    xsoc[i, t] <= total_E,
                    f"soc_upper_{i}_{t}"
                )
                
                # Compute previous timestep considering circular indexing
                if circular_indexing:
                    week = t // hours_per_subperiod
                    t_in_week = t % hours_per_subperiod
                    if t_in_week == 0:
                        t_prev = week * hours_per_subperiod + (hours_per_subperiod - 1)
                    else:
                        t_prev = week * hours_per_subperiod + (t_in_week - 1)
                else:
                    if t == 0:
                        t_prev = None
                    else:
                        t_prev = t - 1
                
                # State of charge evolution
                if t_prev is None:  # First hour, no previous state
                    model.addConstr(
                        xsoc[i, t] == 
                        eff_c * xwdw[i, t] - (xinj[i, t] / eff_d),
                        f"soc_init_{i}_{t}"
                    )
                else:
                    model.addConstr(
                        xsoc[i, t] == 
                        (1 - sd_rate) * xsoc[i, t_prev] + 
                        eff_c * xwdw[i, t] - (xinj[i, t] / eff_d),
                        f"soc_evol_{i}_{t}"
                    )
                
                # Charge/discharge limits based on state of charge
                if t_prev is not None:
                    model.addConstr(
                        eff_c * xwdw[i, t] <= total_E - xsoc[i, t_prev],
                        f"charge_limit_{i}_{t}"
                    )
                    model.addConstr(
                        xinj[i, t] / eff_d <= xsoc[i, t_prev],
                        f"discharge_limit_{i}_{t}"
                    )
                elif not circular_indexing:  # First hour without circular indexing
                    model.addConstr(
                        eff_c * xwdw[i, t] <= total_E,
                        f"charge_limit_init_{i}_{t}"
                    )
                    model.addConstr(
                        xinj[i, t] / eff_d <= 0,
                        f"discharge_limit_init_{i}_{t}"
                    )
            
            # Unit commitment constraints
            if is_uc:
                if isinstance(yPret[i], gp.Var):
                    total_units = (gen['existing_capacity_MW'] / cap_size) + \
                                 yPnew[i] - yPret[i]
                else:
                    total_units = (gen['existing_capacity_MW'] / cap_size) + yPnew[i]
                
                # Constrain UC variables to total units
                model.addConstr(
                    xcommit[i, t] <= total_units,
                    f"commit_ub_{i}_{t}"
                )
                model.addConstr(
                    xstart[i, t] <= total_units,
                    f"start_ub_{i}_{t}"
                )
                model.addConstr(
                    xshut[i, t] <= total_units,
                    f"shut_ub_{i}_{t}"
                )
                
                # Generation bounds based on committed units
                model.addConstr(
                    xinj[i, t] <= avail[t] * cap_size * xcommit[i, t],
                    f"uc_upper_{i}_{t}"
                )
                model.addConstr(
                    xinj[i, t] >= min_out * cap_size * xcommit[i, t],
                    f"uc_lower_{i}_{t}"
                )
                
                # Compute previous timestep considering circular indexing
                if circular_indexing:
                    week = t // hours_per_subperiod
                    t_in_week = t % hours_per_subperiod
                    if t_in_week == 0:
                        t_prev = week * hours_per_subperiod + (hours_per_subperiod - 1)
                    else:
                        t_prev = week * hours_per_subperiod + (t_in_week - 1)
                else:
                    if t == 0:
                        t_prev = None
                    else:
                        t_prev = t - 1
                
                # Commitment change constraints
                if t_prev is None:  # First hour
                    model.addConstr(
                        xcommit[i, t] == xstart[i, t],
                        f"commit_init_{i}_{t}"
                    )
                else:
                    model.addConstr(
                        xcommit[i, t] == xcommit[i, t_prev] + xstart[i, t] - xshut[i, t],
                        f"commit_change_{i}_{t}"
                    )
                
                # Ramp rate constraints
                ramp_up = gen.get('ramp_up_frac_per_hr', 1.0)  # Default to full ramp
                ramp_dn = gen.get('ramp_dn_frac_per_hr', 1.0)
                ramp_limit_new = min(avail[t], max(min_out, ramp_up))
                ramp_limit_shut = min(avail[t], max(min_out, ramp_dn))
                
                if t_prev is not None:
                    # Ramp up constraint
                    model.addConstr(
                        xinj[i, t] - xinj[i, t_prev] <=
                        cap_size * (ramp_up * (xcommit[i, t_prev] - xshut[i, t]) +
                                    ramp_limit_new * xstart[i, t] -
                                    min_out * cap_size * xshut[i, t]),
                        f"ramp_up_{i}_{t}"
                    )
                    
                    # Ramp down constraint
                    model.addConstr(
                        xinj[i, t_prev] - xinj[i, t] <=
                        cap_size * (ramp_dn * (xcommit[i, t_prev] - xstart[i, t]) +
                                    ramp_limit_shut * xshut[i, t] -
                                    min_out * cap_size * xstart[i, t]),
                        f"ramp_dn_{i}_{t}"
                    )
                
                # Min up/down time constraints
                min_up = gen.get('min_up_time_hr', 0)
                min_dn = gen.get('min_down_time_hr', 0)
                
                if min_up > 1:
                    if t >= min_up - 1:
                        model.addConstr(
                            gp.quicksum(xstart[i, k] for k in range(t - min_up + 1, t + 1))
                            <= xcommit[i, t],
                            f"min_up_{i}_{t}"
                        )
                    elif circular_indexing:  # Handle beginning of week
                        week = t // hours_per_subperiod
                        prev_hours = []
                        for offset in range(min_up):
                            hour_idx = t - offset
                            if hour_idx < week * hours_per_subperiod:
                                hour_idx = (week + 1) * hours_per_subperiod - (week * hours_per_subperiod - hour_idx)
                            prev_hours.append(hour_idx)
                        model.addConstr(
                            gp.quicksum(xstart[i, k] for k in prev_hours)
                            <= xcommit[i, t],
                            f"min_up_circular_{i}_{t}"
                        )
                
                if min_dn > 1:
                    if t >= min_dn - 1:
                        model.addConstr(
                            gp.quicksum(xshut[i, k] for k in range(t - min_dn + 1, t + 1))
                            <= total_units - xcommit[i, t],
                            f"min_dn_{i}_{t}"
                        )
                    elif circular_indexing:  # Handle beginning of week
                        week = t // hours_per_subperiod
                        prev_hours = []
                        for offset in range(min_dn):
                            hour_idx = t - offset
                            if hour_idx < week * hours_per_subperiod:
                                hour_idx = (week + 1) * hours_per_subperiod - (week * hours_per_subperiod - hour_idx)
                            prev_hours.append(hour_idx)
                        model.addConstr(
                            gp.quicksum(xshut[i, k] for k in prev_hours)
                            <= total_units - xcommit[i, t],
                            f"min_dn_circular_{i}_{t}"
                        )
    
    # Link storage power and energy capacities
    for gen in generators:
        i = gen['id']
        if gen.get('is_storage', False):
            min_duration = gen.get('min_duration_MWh_per_MW', 0.0)
            max_duration = gen.get('max_duration_MWh_per_MW', float('inf'))
            model.addConstr(
                yE_total_MWh[i] >= min_duration * yP_total_MW[i],
                f"min_duration_{i}"
            )
            model.addConstr(
                yE_total_MWh[i] <= max_duration * yP_total_MW[i],
                f"max_duration_{i}"
            )
    
    # Transmission constraints
    for line in transmission_lines:
        l = line['id']
        total_cap = line['existing_capacity_MW'] + yFnew[l]
        for t in range(total_timesteps):
            model.addConstr(
                xflow[l, t] <= total_cap,
                f"flow_ub_{l}_{t}"
            )
            model.addConstr(
                xflow[l, t] >= -total_cap,
                f"flow_lb_{l}_{t}"
            )
    
    # Policy constraints
    # Renewable Portfolio Standard (RPS)
    if policy.get('RPS_enabled', False):
        qualifying_gens = [g['id'] for g in generators if g.get('is_RPS_qualifying', False)]
        total_RPS_required = policy.get('RPS_share', 0.0) * total_weighted_demand
        for w in range(num_subperiods):
            week_sum = 0
            weight = subperiod_weights[w]
            start_t = w * hours_per_subperiod
            end_t = (w + 1) * hours_per_subperiod
            for t in range(start_t, end_t):
                hr_weight = hourly_weights[t]
                for gen_id in qualifying_gens:
                    week_sum += hr_weight * xinj[gen_id, t]
            model.addConstr(
                week_sum + xRPS[w] >= total_RPS_required * (weight / sum(subperiod_weights)),
                f"RPS_week_{w}"
            )
    
    # CO2 Cap
    if policy.get('CO2_cap_enabled', False):
        total_co2 = 0
        for w in range(num_subperiods):
            weight = subperiod_weights[w]
            start_t = w * hours_per_subperiod
            end_t = (w + 1) * hours_per_subperiod
            for t in range(start_t, end_t):
                hr_weight = hourly_weights[t]
                for gen in generators:
                    i = gen['id']
                    co2_rate = gen.get('co2_tons_per_MWh', 0.0)
                    if gen.get('is_storage', False):
                        total_co2 += hr_weight * co2_rate * (xinj[i, t] + xwdw[i, t])
                    else:
                        total_co2 += hr_weight * co2_rate * xinj[i, t]
        
        model.addConstr(
            total_co2 - gp.quicksum(xCO2[w] for w in range(num_subperiods)) <= 
            0.05 * total_weighted_demand,
            "CO2_cap_total"
        )
    
    # ===================================
    # Objective Function
    # ===================================
    # 1. Fixed costs (investment and fixed O&M)
    fixed_costs = 0
    
    # Generator power investment and FOM
    for gen in generators:
        i = gen['id']
        # Power investment cost
        fixed_costs += gen.get('inv_cost_per_MW_yr', 0.0) * (yPnew[i] * gen['capacity_size_MW'])
        
        # Power FOM cost
        fixed_costs += gen.get('fom_cost_per_MW_yr', 0.0) * yP_total_MW[i]
        
        # Storage energy investment and FOM
        if gen.get('is_storage', False):
            fixed_costs += gen.get('inv_cost_energy_per_MWh_yr', 0.0) * (yEnew[i] * gen.get('storage_capacity_size_MWh', 0.0))
            fixed_costs += gen.get('fom_cost_energy_per_MWh_yr', 0.0) * yE_total_MWh[i]
        
        # Hydropower has additional costs
        if gen.get('is_hydro', False):
            duration_ratio = gen.get('duration_MWh_per_MW', 0.0)
            fixed_costs += gen.get('inv_cost_energy_per_MWh_yr', 0.0) * (duration_ratio * yPnew[i] * gen['capacity_size_MW'])
            fixed_costs += gen.get('fom_cost_energy_per_MWh_yr', 0.0) * (duration_ratio * yP_total_MW[i])
    
    # Transmission investment
    for line in transmission_lines:
        fixed_costs += line.get('inv_cost_per_MW_yr', 0.0) * yFnew[line['id']]
    
    # 2. Variable costs (generation, starts, NSE)
    variable_costs = 0
    
    for t in range(total_timesteps):
        # Generation costs
        for gen in generators:
            i = gen['id']
            var_cost = gen.get('var_cost_per_MWh', 0.0)
            variable_costs += hourly_weights[t] * var_cost * xinj[i, t]
            if gen.get('is_storage', False):
                variable_costs += hourly_weights[t] * var_cost * xwdw[i, t]
            if gen.get('is_UC', False):
                start_cost = gen.get('start_cost_per_unit', 0.0)
                variable_costs += hourly_weights[t] * start_cost * xstart[i, t]
        
        # Non-served energy costs
        for seg in nse_segments:
            seg_id = seg['segment']
            cost_per_mwh = seg.get('cost_per_MWh', 0.0)
            for zone in demand.keys():
                z = int(zone)
                variable_costs += hourly_weights[t] * cost_per_mwh * xnse[seg_id, z, t]
    
    # 3. Policy non-compliance costs
    policy_costs = 0
    if policy.get('RPS_enabled', False):
        policy_costs += policy.get('RPS_noncompliance_cost_per_MWh', 0.0) * gp.quicksum(xRPS[w] for w in range(num_subperiods))
    
    if policy.get('CO2_cap_enabled', False):
        policy_costs += policy.get('CO2_noncompliance_cost_per_ton', 0.0) * gp.quicksum(xCO2[w] for w in range(num_subperiods))
    
    # Total objective
    total_cost = fixed_costs + variable_costs + policy_costs
    model.setObjective(total_cost)
    
    # ===================================
    # Solve and Process Results
    # ===================================
    # Callback function for intermediate solutions
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            if logger:
                obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                sol = {}
                for v in model.getVars():
                    val = model.cbGetSolution(v)
                    if abs(val) > 1e-6:  # Only record non-zero values
                        sol[v.VarName] = val
                logger.log_solution(obj_val, sol)
    
    # Optimize model
    model.optimize(callback)
    
    # Prepare solution dictionary
    solution = {
        "objective_value": None,
        "model_variables": {}
    }
    
    if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL]:
        if model.SolCount > 0:  # Solution found
            solution["objective_value"] = model.ObjVal
            for v in model.getVars():
                if abs(v.X) > 1e-6:  # Only output non-zero variables
                    solution["model_variables"][v.VarName] = v.X
    else:  # No feasible solution found
        solution["objective_value"] = None
    
    # Output solution to file
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()
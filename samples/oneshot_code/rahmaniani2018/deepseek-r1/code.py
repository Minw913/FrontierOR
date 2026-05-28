import argparse
import json
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from solution_logger import SolutionLogger

def main(instance_path, solution_path, time_limit, log_path):
    # Load instance data
    with open(instance_path, 'r') as f:
        data = json.load(f)
    
    dims = data['problem_dimensions']
    policy = data['policy']
    subperiods = data['subperiods']
    generators = data['generators']
    transmission_lines = data['transmission_lines']
    demand_data = data['demand']
    nse_segments = data['nse_segments']
    availability_profiles = data['availability_profiles']
    
    num_zones = dims['num_zones']
    num_weeks = dims['num_weeks']
    hours_per_week = dims['hours_per_week']
    total_timesteps = dims['total_timesteps']
    num_generators = dims['num_generators']
    num_UC_generators = dims['num_UC_generators']
    num_storage_resources = dims['num_storage_resources']
    num_hydro_resources = dims['num_hydro_resources']
    num_transmission_lines = dims['num_transmission_lines']
    num_demand_segments = dims['num_demand_segments']
    
    subperiod_weights = subperiods['subperiod_weights']
    circular_indexing = subperiods['circular_indexing']
    
    # Map generator data by ID for quick access
    gen_data = {g['id']: g for g in generators}
    
    # Precompute hourly weights and total weighted demand
    hourly_weights = []
    for w in range(num_weeks):
        weight_week = subperiod_weights[w]
        for t in range(hours_per_week):
            hourly_weights.append(weight_week / hours_per_week)
    
    total_weighted_demand = 0.0
    for zone_id, demand_list in demand_data.items():
        for h in range(total_timesteps):
            total_weighted_demand += hourly_weights[h] * demand_list[h]
    
    # Precompute availability factors for generators
    avail_factors = {}
    for gen_id, profile in availability_profiles.items():
        gen_id_int = int(gen_id)
        if profile['type'] == 'constant':
            avail_factors[gen_id_int] = [profile['value']] * total_timesteps
        else:
            avail_factors[gen_id_int] = profile['values']
    
    # Initialize Gurobi model
    model = gp.Model("Electricity_System_Planning")
    model.setParam('OutputFlag', 1)
    model.setParam('TimeLimit', time_limit)
    model.setParam('Threads', 1)
    
    # Create solution logger if log_path is provided
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    last_incumbent = None
    
    def callback(model, where):
        nonlocal last_incumbent
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if last_incumbent is None or obj_val < last_incumbent:
                last_incumbent = obj_val
                if logger:
                    logger.log(obj_val)
    
    # Investment variables
    y_P_new = {}  # New power capacity units (integer)
    y_P_ret = {}  # Retired power capacity units (integer)
    y_E_new = {}  # New energy capacity units for storage (integer)
    y_E_ret = {}  # Retired energy capacity units for storage (integer)
    y_F_new = {}  # New transmission capacity (integer MW)
    
    for g in generators:
        gid = g['id']
        # Power capacity variables
        max_new_units = g['max_capacity_MW'] / g['capacity_size_MW'] if g['max_capacity_MW'] > 0 else 0
        max_ret_units = g['existing_capacity_MW'] / g['capacity_size_MW'] if g['can_retire'] else 0
        y_P_new[gid] = model.addVar(lb=0, ub=max_new_units, vtype=GRB.INTEGER, name=f"y_P_new_{gid}")
        y_P_ret[gid] = model.addVar(lb=0, ub=max_ret_units, vtype=GRB.INTEGER, name=f"y_P_ret_{gid}")
        
        # Storage energy capacity variables
        if g['is_storage']:
            max_new_e_units = g['max_storage_capacity_MWh'] / g['storage_capacity_size_MWh'] if g['max_storage_capacity_MWh'] > 0 else 0
            max_ret_e_units = g['existing_storage_capacity_MWh'] / g['storage_capacity_size_MWh'] if g['can_retire'] else 0
            y_E_new[gid] = model.addVar(lb=0, ub=max_new_e_units, vtype=GRB.INTEGER, name=f"y_E_new_{gid}")
            y_E_ret[gid] = model.addVar(lb=0, ub=max_ret_e_units, vtype=GRB.INTEGER, name=f"y_E_ret_{gid}")
    
    for line in transmission_lines:
        lid = line['id']
        y_F_new[lid] = model.addVar(lb=0, ub=line['max_new_capacity_MW'], vtype=GRB.INTEGER, name=f"y_F_new_{lid}")
    
    # Operational variables
    gen = {}  # Generation for each generator and hour
    charge = {}  # Storage charge for storage resources
    soc = {}  # State of charge for storage resources
    spill = {}  # Spillage for hydro resources
    res = {}  # Reservoir level for hydro resources
    flow = {}  # Transmission flow for each line and hour
    nse = {}  # Non-served energy for each zone, segment, and hour
    
    # Unit commitment variables
    u = {}  # Committed units for UC resources
    v_start = {}  # Startup events for UC resources
    v_shut = {}  # Shutdown events for UC resources
    
    # Policy slack variables
    RPS_slack = None
    CO2_slack = None
    if policy['RPS_enabled']:
        RPS_slack = model.addVar(lb=0, name="RPS_slack")
    if policy['CO2_cap_enabled']:
        CO2_slack = model.addVar(lb=0, name="CO2_slack")
    
    # Create variables
    for g in generators:
        gid = g['id']
        gen[gid] = []
        for h in range(total_timesteps):
            gen[gid].append(model.addVar(lb=0, name=f"gen_{gid}_{h}"))
        
        if g['is_storage']:
            charge[gid] = []
            soc[gid] = []
            for h in range(total_timesteps):
                charge[gid].append(model.addVar(lb=0, name=f"charge_{gid}_{h}"))
                soc[gid].append(model.addVar(lb=0, name=f"soc_{gid}_{h}"))
        
        if g['is_hydro']:
            spill[gid] = []
            res[gid] = []
            for h in range(total_timesteps):
                spill[gid].append(model.addVar(lb=0, name=f"spill_{gid}_{h}"))
                res[gid].append(model.addVar(lb=0, name=f"res_{gid}_{h}"))
        
        if g['is_UC']:
            u[gid] = []
            v_start[gid] = []
            v_shut[gid] = []
            for h in range(total_timesteps):
                u[gid].append(model.addVar(vtype=GRB.INTEGER, lb=0, name=f"u_{gid}_{h}"))
                v_start[gid].append(model.addVar(vtype=GRB.INTEGER, lb=0, name=f"v_start_{gid}_{h}"))
                v_shut[gid].append(model.addVar(vtype=GRB.INTEGER, lb=0, name=f"v_shut_{gid}_{h}"))
    
    for line in transmission_lines:
        lid = line['id']
        flow[lid] = []
        for h in range(total_timesteps):
            flow[lid].append(model.addVar(lb=-line['existing_capacity_MW'] - y_F_new[lid].ub, 
                                          ub=line['existing_capacity_MW'] + y_F_new[lid].ub, 
                                          name=f"flow_{lid}_{h}"))
    
    for zone_id in demand_data.keys():
        zone_id_int = int(zone_id)
        nse[zone_id_int] = {}
        for seg in nse_segments:
            seg_id = seg['segment']
            nse[zone_id_int][seg_id] = []
            for h in range(total_timesteps):
                nse[zone_id_int][seg_id].append(model.addVar(lb=0, name=f"nse_{zone_id_int}_{seg_id}_{h}"))
    
    # Total installed capacities
    P_total = {}
    E_total = {}
    F_total = {}
    
    for g in generators:
        gid = g['id']
        P_total[gid] = g['existing_capacity_MW'] + y_P_new[gid] * g['capacity_size_MW'] - y_P_ret[gid] * g['capacity_size_MW']
        if g['is_storage']:
            E_total[gid] = g['existing_storage_capacity_MWh'] + y_E_new[gid] * g['storage_capacity_size_MWh'] - y_E_ret[gid] * g['storage_capacity_size_MWh']
    
    for line in transmission_lines:
        lid = line['id']
        F_total[lid] = line['existing_capacity_MW'] + y_F_new[lid]
    
    # Constraints: Power balance for each zone and hour
    for zone_id, demand_list in demand_data.items():
        zone_id_int = int(zone_id)
        for h in range(total_timesteps):
            # Sum generation from resources in the zone
            gen_sum = gp.quicksum(gen[gid][h] for g in generators if g['zone'] == zone_id_int)
            # Sum storage charge (withdrawal) in the zone
            charge_sum = gp.quicksum(charge[gid][h] for g in generators if g['is_storage'] and g['zone'] == zone_id_int)
            # Net flow: sum(flow into zone) - sum(flow out of zone)
            net_flow = gp.quicksum(flow[lid][h] for lid, line in enumerate(transmission_lines) if line['to_zone'] == zone_id_int)
            net_flow -= gp.quicksum(flow[lid][h] for lid, line in enumerate(transmission_lines) if line['from_zone'] == zone_id_int)
            # Non-served energy
            nse_sum = gp.quicksum(nse[zone_id_int][seg['segment']][h] for seg in nse_segments)
            
            model.addConstr(
                gen_sum - charge_sum + net_flow + nse_sum == demand_list[h],
                name=f"power_balance_zone{zone_id_int}_hour{h}"
            )
    
    # Constraints: Generator capacity and operational limits
    for g in generators:
        gid = g['id']
        cap_size = g['capacity_size_MW']
        min_output_frac = g['min_output_frac']
        is_UC = g['is_UC']
        is_storage = g['is_storage']
        is_hydro = g['is_hydro']
        
        for h in range(total_timesteps):
            avail = avail_factors[gid][h]
            
            if is_UC:
                # UC generation constraints
                model.addConstr(gen[gid][h] >= min_output_frac * cap_size * u[gid][h], 
                               name=f"gen_min_UC_{gid}_{h}")
                model.addConstr(gen[gid][h] <= avail * cap_size * u[gid][h], 
                               name=f"gen_max_UC_{gid}_{h}")
            else:
                # Non-UC generation constraints
                model.addConstr(gen[gid][h] <= avail * P_total[gid], 
                               name=f"gen_max_nonUC_{gid}_{h}")
                if not is_storage and not is_hydro:
                    model.addConstr(gen[gid][h] >= min_output_frac * P_total[gid], 
                                   name=f"gen_min_nonUC_{gid}_{h}")
            
            if is_storage:
                # Storage charge and discharge constraints
                model.addConstr(charge[gid][h] <= avail * P_total[gid], 
                               name=f"charge_max_{gid}_{h}")
                model.addConstr(gen[gid][h] + charge[gid][h] <= P_total[gid], 
                               name=f"power_capacity_{gid}_{h}")
    
    # Constraints: Storage state of charge dynamics
    for g in generators:
        if not g['is_storage']:
            continue
        gid = g['id']
        eff_charge = g['charge_efficiency']
        eff_discharge = g['discharge_efficiency']
        self_discharge = g['self_discharge_rate']
        
        for w in range(num_weeks):
            for t in range(hours_per_week):
                h = w * hours_per_week + t
                prev_h = w * hours_per_week + (t - 1) if t > 0 else w * hours_per_week + (hours_per_week - 1)
                
                if t == 0 and circular_indexing:
                    # Circular linkage: last hour of the week
                    prev_soc = soc[gid][prev_h]
                else:
                    prev_soc = soc[gid][h-1] if h > 0 else 0
                
                # State of charge update
                model.addConstr(
                    soc[gid][h] == (1 - self_discharge) * prev_soc + 
                    eff_charge * charge[gid][h] - 
                    gen[gid][h] / eff_discharge,
                    name=f"soc_balance_{gid}_{h}"
                )
                
                # State of charge bounds
                model.addConstr(soc[gid][h] <= E_total[gid], name=f"soc_max_{gid}_{h}")
                model.addConstr(charge[gid][h] * eff_charge <= E_total[gid] - (1 - self_discharge) * prev_soc, 
                               name=f"charge_room_{gid}_{h}")
                model.addConstr(gen[gid][h] / eff_discharge <= (1 - self_discharge) * prev_soc, 
                               name=f"discharge_available_{gid}_{h}")
    
    # Constraints: Hydropower reservoir dynamics
    for g in generators:
        if not g['is_hydro']:
            continue
        gid = g['id']
        duration_ratio = g['duration_MWh_per_MW']
        inflow_profile = avail_factors[gid]  # Using availability profile as inflow
        
        for w in range(num_weeks):
            for t in range(hours_per_week):
                h = w * hours_per_week + t
                prev_h = w * hours_per_week + (t - 1) if t > 0 else w * hours_per_week + (hours_per_week - 1)
                
                if t == 0 and circular_indexing:
                    prev_res = res[gid][prev_h]
                else:
                    prev_res = res[gid][h-1] if h > 0 else 0
                
                # Reservoir balance
                inflow = inflow_profile[h] * P_total[gid]
                model.addConstr(
                    res[gid][h] == prev_res + inflow - gen[gid][h] - spill[gid][h],
                    name=f"res_balance_{gid}_{h}"
                )
                
                # Reservoir bounds
                model.addConstr(res[gid][h] <= duration_ratio * P_total[gid], 
                               name=f"res_max_{gid}_{h}")
                model.addConstr(gen[gid][h] + spill[gid][h] >= g['min_output_frac'] * P_total[gid], 
                               name=f"hydro_min_output_{gid}_{h}")
    
    # Constraints: Transmission line capacity
    for line in transmission_lines:
        lid = line['id']
        for h in range(total_timesteps):
            model.addConstr(flow[lid][h] <= F_total[lid], name=f"flow_upper_{lid}_{h}")
            model.addConstr(flow[lid][h] >= -F_total[lid], name=f"flow_lower_{lid}_{h}")
    
    # Constraints: Non-served energy segments
    for seg in nse_segments:
        seg_id = seg['segment']
        max_frac = seg['max_frac']
        for zone_id, demand_list in demand_data.items():
            zone_id_int = int(zone_id)
            for h in range(total_timesteps):
                model.addConstr(
                    nse[zone_id_int][seg_id][h] <= max_frac * demand_list[h],
                    name=f"nse_max_{zone_id_int}_{seg_id}_{h}"
                )
    
    # Constraints: Unit commitment
    for g in generators:
        if not g['is_UC']:
            continue
        gid = g['id']
        cap_size = g['capacity_size_MW']
        min_up = g['min_up_time_hr']
        min_down = g['min_down_time_hr']
        ramp_up = g['ramp_up_frac_per_hr']
        ramp_dn = g['ramp_dn_frac_per_hr']
        min_output_frac = g['min_output_frac']
        
        n_units = P_total[gid] / cap_size  # Total units (continuous expression)
        
        for w in range(num_weeks):
            for t in range(hours_per_week):
                h = w * hours_per_week + t
                prev_t = t - 1 if t > 0 else hours_per_week - 1 if circular_indexing else -1
                prev_h = w * hours_per_week + prev_t if prev_t >= 0 else -1
                
                # Commitment change constraint
                if h == 0:
                    if circular_indexing:
                        prev_h = (w+1) * hours_per_week - 1  # Last hour of the week
                        model.addConstr(
                            u[gid][h] - u[gid][prev_h] == v_start[gid][h] - v_shut[gid][h],
                            name=f"commit_change_{gid}_{h}"
                        )
                else:
                    model.addConstr(
                        u[gid][h] - u[gid][h-1] == v_start[gid][h] - v_shut[gid][h],
                        name=f"commit_change_{gid}_{h}"
                    )
                
                # Min up time constraint
                if min_up > 0:
                    sum_start = gp.quicksum(v_start[gid][k] for k in range(max(0, h - min_up + 1), h+1))
                    model.addConstr(u[gid][h] >= sum_start, name=f"min_up_{gid}_{h}")
                
                # Min down time constraint
                if min_down > 0:
                    sum_shut = gp.quicksum(v_shut[gid][k] for k in range(max(0, h - min_down + 1), h+1))
                    model.addConstr(u[gid][h] + sum_shut <= n_units, name=f"min_down_{gid}_{h}")
                
                # Ramp constraints
                if h > 0 or (h == 0 and circular_indexing):
                    prev_gen = gen[gid][h-1] if h > 0 else gen[gid][w * hours_per_week + hours_per_week - 1]
                    delta = gen[gid][h] - prev_gen
                    
                    # Ramp-up constraint
                    ramp_up_bound = ramp_up * cap_size * (u[gid][h-1] - v_start[gid][h]) + \
                                   min(avail_factors[gid][h], max(min_output_frac, ramp_up)) * cap_size * v_start[gid][h] - \
                                   min_output_frac * cap_size * v_shut[gid][h]
                    model.addConstr(delta <= ramp_up_bound, name=f"ramp_up_{gid}_{h}")
                    
                    # Ramp-down constraint
                    ramp_dn_bound = ramp_dn * cap_size * (u[gid][h-1] - v_shut[gid][h]) + \
                                   min(avail_factors[gid][h], max(min_output_frac, ramp_dn)) * cap_size * v_shut[gid][h] - \
                                   min_output_frac * cap_size * v_start[gid][h]
                    model.addConstr(-delta <= ramp_dn_bound, name=f"ramp_dn_{gid}_{h}")
    
    # Constraints: Policy compliance
    if policy['RPS_enabled']:
        qualifying_gen = 0.0
        for g in generators:
            if g['is_RPS_qualifying']:
                gid = g['id']
                for h in range(total_timesteps):
                    qualifying_gen += hourly_weights[h] * gen[gid][h]
        model.addConstr(
            qualifying_gen + RPS_slack >= policy['RPS_share'] * total_weighted_demand,
            name="RPS_constraint"
        )
    
    if policy['CO2_cap_enabled']:
        emissions = 0.0
        for g in generators:
            gid = g['id']
            co2_rate = g['co2_tons_per_MWh']
            for h in range(total_timesteps):
                emissions += hourly_weights[h] * co2_rate * gen[gid][h]
                if g['is_storage']:
                    emissions += hourly_weights[h] * co2_rate * charge[gid][h]
        model.addConstr(
            emissions - CO2_slack <= 0.05 * total_weighted_demand,
            name="CO2_cap_constraint"
        )
    
    # Objective function: Total annual cost
    fixed_cost = 0.0
    # Investment costs
    for g in generators:
        gid = g['id']
        fixed_cost += g['inv_cost_per_MW_yr'] * (y_P_new[gid] * g['capacity_size_MW'])
        fixed_cost += g['fom_cost_per_MW_yr'] * P_total[gid]
        if g['is_storage']:
            fixed_cost += g['inv_cost_energy_per_MWh_yr'] * (y_E_new[gid] * g['storage_capacity_size_MWh'])
            fixed_cost += g['fom_cost_energy_per_MWh_yr'] * E_total[gid]
        if g['is_hydro']:
            # Assume hydro has energy fixed O&M cost if provided
            if 'fom_cost_energy_per_MWh_yr' in g:
                fixed_cost += g['fom_cost_energy_per_MWh_yr'] * (g['duration_MWh_per_MW'] * P_total[gid])
    
    for line in transmission_lines:
        lid = line['id']
        fixed_cost += line['inv_cost_per_MW_yr'] * y_F_new[lid]
    
    var_cost = 0.0
    for g in generators:
        gid = g['id']
        var_cost_g = g['var_cost_per_MWh']
        for h in range(total_timesteps):
            var_cost += hourly_weights[h] * var_cost_g * gen[gid][h]
            if g['is_storage']:
                var_cost += hourly_weights[h] * var_cost_g * charge[gid][h]
    
    nse_cost = 0.0
    for zone_id in demand_data.keys():
        zone_id_int = int(zone_id)
        for seg in nse_segments:
            seg_id = seg['segment']
            cost_per_mwh = seg['cost_per_MWh']
            for h in range(total_timesteps):
                nse_cost += hourly_weights[h] * cost_per_mwh * nse[zone_id_int][seg_id][h]
    
    startup_cost = 0.0
    for g in generators:
        if g['is_UC']:
            gid = g['id']
            cost_per_start = g['start_cost_per_unit']
            for h in range(total_timesteps):
                startup_cost += hourly_weights[h] * cost_per_start * v_start[gid][h]
    
    policy_cost = 0.0
    if policy['RPS_enabled']:
        policy_cost += policy['RPS_noncompliance_cost_per_MWh'] * RPS_slack
    if policy['CO2_cap_enabled']:
        policy_cost += policy['CO2_noncompliance_cost_per_ton'] * CO2_slack
    
    total_cost = fixed_cost + var_cost + nse_cost + startup_cost + policy_cost
    model.setObjective(total_cost, GRB.MINIMIZE)
    
    # Optimize
    model.optimize(callback)
    
    # Extract solution
    solution = {
        "objective_value": model.ObjVal,
        "investment_decisions": [],
        "transmission_decisions": []
    }
    
    for g in generators:
        gid = g['id']
        y_P_new_val = y_P_new[gid].X if hasattr(y_P_new[gid], 'X') else 0
        y_P_ret_val = y_P_ret[gid].X if hasattr(y_P_ret[gid], 'X') else 0
        P_total_val = P_total[gid].getValue() if hasattr(P_total[gid], 'getValue') else \
                      g['existing_capacity_MW'] + y_P_new_val * g['capacity_size_MW'] - y_P_ret_val * g['capacity_size_MW']
        
        decision = {
            "generator_id": gid,
            "resource_type": g['resource_type'],
            "zone": g['zone'],
            "y_P_new": y_P_new_val,
            "y_P_ret": y_P_ret_val,
            "y_P_total_MW": P_total_val
        }
        
        if g['is_storage']:
            y_E_new_val = y_E_new[gid].X if hasattr(y_E_new[gid], 'X') else 0
            y_E_ret_val = y_E_ret[gid].X if hasattr(y_E_ret[gid], 'X') else 0
            E_total_val = E_total[gid].getValue() if hasattr(E_total[gid], 'getValue') else \
                          g['existing_storage_capacity_MWh'] + y_E_new_val * g['storage_capacity_size_MWh'] - y_E_ret_val * g['storage_capacity_size_MWh']
            decision["y_E_new"] = y_E_new_val
            decision["y_E_ret"] = y_E_ret_val
            decision["y_E_total_MWh"] = E_total_val
        
        solution["investment_decisions"].append(decision)
    
    for line in transmission_lines:
        lid = line['id']
        y_F_new_val = y_F_new[lid].X if hasattr(y_F_new[lid], 'X') else 0
        total_cap = line['existing_capacity_MW'] + y_F_new_val
        solution["transmission_decisions"].append({
            "line_id": lid,
            "y_F_new_MW": y_F_new_val,
            "total_capacity_MW": total_cap
        })
    
    # Save solution
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
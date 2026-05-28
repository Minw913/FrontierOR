import argparse
import json
import time
import math
import gurobipy as gp
from gurobipy import GRB

def solve(instance_path, solution_path, time_limit, log_path=None):
    from solution_logger import SolutionLogger
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None

    start_time = time.time()

    with open(instance_path, 'r') as f:
        data = json.load(f)

    dims = data['problem_dimensions']
    num_zones = dims['num_zones']
    num_weeks = dims['num_weeks']
    hours_per_week = dims['hours_per_week']
    total_timesteps = dims['total_timesteps']
    num_generators = dims['num_generators']
    num_UC = dims['num_UC_generators']
    num_storage = dims['num_storage_resources']
    num_hydro = dims['num_hydro_resources']
    num_lines = dims['num_transmission_lines']
    num_demand_segments = dims['num_demand_segments']

    policy = data['policy']
    rps_enabled = policy['RPS_enabled']
    rps_share = policy['RPS_share']
    co2_cap_enabled = policy['CO2_cap_enabled']
    co2_cap_tpmwh = policy.get('CO2_cap_tons_per_MWh')
    rps_penalty = policy['RPS_noncompliance_cost_per_MWh']
    co2_penalty = policy['CO2_noncompliance_cost_per_ton']

    subperiods = data['subperiods']
    num_subperiods = subperiods['num_subperiods']
    hours_per_subperiod = subperiods['hours_per_subperiod']
    sp_weights = subperiods['subperiod_weights']

    generators = data['generators']
    lines = data['transmission_lines']
    demand = data['demand']
    nse_segments = data['nse_segments']
    avail_profiles = data['availability_profiles']

    # Precompute hourly weights
    hour_weights = []
    for sp in range(num_subperiods):
        w = sp_weights[sp] / hours_per_subperiod
        for h in range(hours_per_subperiod):
            hour_weights.append(w)

    # Precompute total weighted demand
    total_weighted_demand = 0.0
    for z in range(1, num_zones + 1):
        zkey = str(z)
        for t in range(total_timesteps):
            total_weighted_demand += hour_weights[t] * demand[zkey][t]

    # CO2 cap
    co2_cap = None
    if co2_cap_enabled and co2_cap_tpmwh is not None:
        co2_cap = co2_cap_tpmwh * total_weighted_demand

    # Availability profiles
    def get_avail(gen_id, t):
        prof = avail_profiles[str(gen_id)]
        if prof['type'] == 'constant':
            return prof['value']
        else:
            return prof['values'][t]

    # Index generators by zone
    zone_gens = {z: [] for z in range(1, num_zones + 1)}
    for i, g in enumerate(generators):
        zone_gens[g['zone']].append(i)

    # Index lines by zone
    zone_lines_from = {z: [] for z in range(1, num_zones + 1)}
    zone_lines_to = {z: [] for z in range(1, num_zones + 1)}
    for j, ln in enumerate(lines):
        zone_lines_from[ln['from_zone']].append(j)
        zone_lines_to[ln['to_zone']].append(j)

    # Storage, hydro, UC indices
    storage_indices = [i for i, g in enumerate(generators) if g['is_storage']]
    hydro_indices = [i for i, g in enumerate(generators) if g['is_hydro']]
    uc_indices = [i for i, g in enumerate(generators) if g['is_UC']]
    rps_indices = [i for i, g in enumerate(generators) if g['is_RPS_qualifying']]

    # Build model
    model = gp.Model("capacity_expansion")
    model.setParam('TimeLimit', max(1, time_limit - int(time.time() - start_time) - 5))
    model.setParam('Threads', 1)
    model.setParam('MIPGap', 0.005)
    model.setParam('MIPFocus', 1)  # Focus on finding feasible solutions

    # ========== INVESTMENT VARIABLES ==========
    y_P_new = {}
    y_P_ret = {}
    y_P_total = {}

    for i, g in enumerate(generators):
        cap_size = g['capacity_size_MW']
        existing = g['existing_capacity_MW']
        max_new_units = int(math.floor((g['max_capacity_MW'] - existing) / cap_size + 0.5)) if cap_size > 0 else 0
        max_new_units = max(0, max_new_units)
        max_ret_units = int(math.floor(existing / cap_size + 0.5)) if cap_size > 0 else 0

        if g['is_UC']:
            y_P_new[i] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=max_new_units, name=f"yPn_{i}")
            if g['can_retire']:
                y_P_ret[i] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=max_ret_units, name=f"yPr_{i}")
            else:
                y_P_ret[i] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=0, name=f"yPr_{i}")
        else:
            y_P_new[i] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=max_new_units, name=f"yPn_{i}")
            if g['can_retire']:
                y_P_ret[i] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=max_ret_units, name=f"yPr_{i}")
            else:
                y_P_ret[i] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=0, name=f"yPr_{i}")

        y_P_total[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"yPt_{i}")

    # Storage energy capacity
    y_E_new = {}
    y_E_ret = {}
    y_E_total = {}
    for i in storage_indices:
        g = generators[i]
        e_size = g['storage_capacity_size_MWh']
        existing_e = g['existing_storage_capacity_MWh']
        max_new_e = int(math.floor((g['max_storage_capacity_MWh'] - existing_e) / e_size + 0.5)) if e_size > 0 else 0
        max_new_e = max(0, max_new_e)
        max_ret_e = int(math.floor(existing_e / e_size + 0.5)) if e_size > 0 else 0

        y_E_new[i] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=max_new_e, name=f"yEn_{i}")
        if g['can_retire']:
            y_E_ret[i] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=max_ret_e, name=f"yEr_{i}")
        else:
            y_E_ret[i] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=0, name=f"yEr_{i}")
        y_E_total[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"yEt_{i}")

    # Transmission
    y_F_new = {}
    y_F_total = {}
    for j, ln in enumerate(lines):
        y_F_new[j] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=int(ln['max_new_capacity_MW']), name=f"yFn_{j}")
        y_F_total[j] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"yFt_{j}")

    # ========== OPERATIONAL VARIABLES ==========
    # Generation
    gen_var = {}
    for i, g in enumerate(generators):
        for t in range(total_timesteps):
            gen_var[i, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"g_{i}_{t}")

    # Storage withdrawal (charging)
    withdraw_var = {}
    for i in storage_indices:
        for t in range(total_timesteps):
            withdraw_var[i, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"w_{i}_{t}")

    # State of charge
    soc_var = {}
    for i in storage_indices:
        for t in range(total_timesteps):
            soc_var[i, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"soc_{i}_{t}")

    # Hydro reservoir and spillage
    res_var = {}
    spill_var = {}
    for i in hydro_indices:
        for t in range(total_timesteps):
            res_var[i, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"res_{i}_{t}")
            spill_var[i, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"sp_{i}_{t}")

    # Transmission flow
    flow_var = {}
    for j in range(num_lines):
        for t in range(total_timesteps):
            flow_var[j, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name=f"f_{j}_{t}")

    # NSE
    nse_var = {}
    for s, seg in enumerate(nse_segments):
        for z in range(1, num_zones + 1):
            for t in range(total_timesteps):
                nse_var[s, z, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"nse_{s}_{z}_{t}")

    # UC variables
    commit_var = {}
    startup_var = {}
    shutdown_var = {}
    for i in uc_indices:
        for t in range(total_timesteps):
            commit_var[i, t] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"u_{i}_{t}")
            startup_var[i, t] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"su_{i}_{t}")
            shutdown_var[i, t] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"sd_{i}_{t}")

    # Policy slack
    rps_slack = {}
    co2_slack = {}
    if rps_enabled:
        for sp in range(num_subperiods):
            rps_slack[sp] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"rps_sl_{sp}")
    if co2_cap_enabled:
        for sp in range(num_subperiods):
            co2_slack[sp] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"co2_sl_{sp}")

    model.update()

    # ========== CAPACITY LINKING CONSTRAINTS ==========
    for i, g in enumerate(generators):
        cap_size = g['capacity_size_MW']
        existing = g['existing_capacity_MW']
        model.addConstr(y_P_total[i] == existing + cap_size * (y_P_new[i] - y_P_ret[i]), name=f"cap_link_{i}")

    for i in storage_indices:
        g = generators[i]
        e_size = g['storage_capacity_size_MWh']
        existing_e = g['existing_storage_capacity_MWh']
        model.addConstr(y_E_total[i] == existing_e + e_size * (y_E_new[i] - y_E_ret[i]), name=f"ecap_link_{i}")
        # Duration constraints
        model.addConstr(y_E_total[i] >= g['min_duration_MWh_per_MW'] * y_P_total[i], name=f"dur_min_{i}")
        model.addConstr(y_E_total[i] <= g['max_duration_MWh_per_MW'] * y_P_total[i], name=f"dur_max_{i}")

    for j, ln in enumerate(lines):
        model.addConstr(y_F_total[j] == ln['existing_capacity_MW'] + y_F_new[j], name=f"tcap_link_{j}")

    # ========== OPERATIONAL CONSTRAINTS ==========
    # Helper for subperiod/hour mapping
    def sp_hours(sp):
        start = sp * hours_per_subperiod
        return list(range(start, start + hours_per_subperiod))

    def prev_hour(t, sp_start):
        if t == sp_start:
            return sp_start + hours_per_subperiod - 1
        return t - 1

    # Power balance
    for z in range(1, num_zones + 1):
        zkey = str(z)
        for t in range(total_timesteps):
            lhs = gp.LinExpr()
            for i in zone_gens[z]:
                lhs.add(gen_var[i, t], 1.0)
                if generators[i]['is_storage']:
                    lhs.add(withdraw_var[i, t], -1.0)
            for j in zone_lines_from[z]:
                lhs.add(flow_var[j, t], -1.0)
            for j in zone_lines_to[z]:
                lhs.add(flow_var[j, t], 1.0)
            for s in range(num_demand_segments):
                lhs.add(nse_var[s, z, t], 1.0)
            model.addConstr(lhs == demand[zkey][t], name=f"bal_{z}_{t}")

    # Generation limits for non-UC resources
    for i, g in enumerate(generators):
        if g['is_UC']:
            continue
        if g['is_storage']:
            continue
        if g['is_hydro']:
            continue
        for t in range(total_timesteps):
            av = get_avail(g['id'], t)
            model.addConstr(gen_var[i, t] <= av * y_P_total[i], name=f"gmax_{i}_{t}")
            # Min output for non-UC, non-storage, non-hydro
            if g['min_output_frac'] > 0:
                model.addConstr(gen_var[i, t] >= g['min_output_frac'] * y_P_total[i], name=f"gmin_{i}_{t}")

    # Storage constraints
    for i in storage_indices:
        g = generators[i]
        for t in range(total_timesteps):
            av = get_avail(g['id'], t)
            # Gen limit
            model.addConstr(gen_var[i, t] <= av * y_P_total[i], name=f"sg_max_{i}_{t}")
            # Withdrawal limit
            model.addConstr(withdraw_var[i, t] <= av * y_P_total[i], name=f"sw_max_{i}_{t}")
            # Combined
            model.addConstr(gen_var[i, t] + withdraw_var[i, t] <= y_P_total[i], name=f"sgw_max_{i}_{t}")
            # SOC limit
            model.addConstr(soc_var[i, t] <= y_E_total[i], name=f"soc_max_{i}_{t}")
            # Charging limit (charge_eff * withdraw <= E_cap - soc)
            model.addConstr(g['charge_efficiency'] * withdraw_var[i, t] <= y_E_total[i] - soc_var[i, t], name=f"chrg_{i}_{t}")
            # Discharging limit (gen / discharge_eff <= soc)
            model.addConstr(gen_var[i, t] <= g['discharge_efficiency'] * soc_var[i, t], name=f"dchrg_{i}_{t}")

    # SOC evolution
    for i in storage_indices:
        g = generators[i]
        for sp in range(num_subperiods):
            hours = sp_hours(sp)
            for t in hours:
                tp = prev_hour(t, hours[0])
                model.addConstr(
                    soc_var[i, t] == soc_var[i, tp] * (1 - g['self_discharge_rate'])
                    + g['charge_efficiency'] * withdraw_var[i, t]
                    - gen_var[i, t] / g['discharge_efficiency'],
                    name=f"soc_ev_{i}_{t}"
                )

    # Hydro constraints
    for i in hydro_indices:
        g = generators[i]
        dur = g['duration_MWh_per_MW']
        # Get inflow profile - check if it exists in availability or in generator data
        # Hydro inflow is typically in availability_profiles or a separate field
        # Based on problem description: "hourly normalized inflow profile as a fraction of power capacity"
        # We'll look for it in the generator data or availability profiles
        
        for t in range(total_timesteps):
            av = get_avail(g['id'], t)
            # Generation upper bound
            model.addConstr(gen_var[i, t] <= av * y_P_total[i], name=f"hg_max_{i}_{t}")
            # Reservoir upper bound
            model.addConstr(res_var[i, t] <= dur * y_P_total[i], name=f"res_max_{i}_{t}")
            # Min output: gen + spill >= min_output_frac * capacity
            if g['min_output_frac'] > 0:
                model.addConstr(gen_var[i, t] + spill_var[i, t] >= g['min_output_frac'] * y_P_total[i], name=f"hmin_{i}_{t}")

        # Reservoir evolution - inflow from availability profile
        # The problem says inflow is a separate profile, but per schema it might be in availability_profiles
        # Actually re-reading: "For hydropower resources the data specifies a fixed duration ratio and an hourly normalized inflow profile"
        # The availability_profiles for hydro serve as the inflow profile
        for sp in range(num_subperiods):
            hours = sp_hours(sp)
            for t in hours:
                tp = prev_hour(t, hours[0])
                inflow_frac = get_avail(g['id'], t)
                model.addConstr(
                    res_var[i, t] == res_var[i, tp]
                    + inflow_frac * y_P_total[i]
                    - gen_var[i, t]
                    - spill_var[i, t],
                    name=f"res_ev_{i}_{t}"
                )

    # Transmission flow limits
    for j in range(num_lines):
        for t in range(total_timesteps):
            model.addConstr(flow_var[j, t] <= y_F_total[j], name=f"fmax_{j}_{t}")
            model.addConstr(flow_var[j, t] >= -y_F_total[j], name=f"fmin_{j}_{t}")

    # NSE limits
    for s, seg in enumerate(nse_segments):
        for z in range(1, num_zones + 1):
            zkey = str(z)
            for t in range(total_timesteps):
                model.addConstr(nse_var[s, z, t] <= seg['max_frac'] * demand[zkey][t], name=f"nse_max_{s}_{z}_{t}")

    # Ramp constraints for non-UC, non-storage, non-hydro resources
    for i, g in enumerate(generators):
        if g['is_UC'] or g['is_storage'] or g['is_hydro']:
            continue
        rup = g.get('ramp_up_frac_per_hr', 1.0)
        rdn = g.get('ramp_dn_frac_per_hr', 1.0)
        if rup >= 1.0 and rdn >= 1.0:
            continue
        for sp in range(num_subperiods):
            hours = sp_hours(sp)
            for t in hours:
                tp = prev_hour(t, hours[0])
                if rup < 1.0:
                    model.addConstr(gen_var[i, t] - gen_var[i, tp] <= rup * y_P_total[i], name=f"rup_{i}_{t}")
                if rdn < 1.0:
                    model.addConstr(gen_var[i, tp] - gen_var[i, t] <= rdn * y_P_total[i], name=f"rdn_{i}_{t}")

    # UC constraints
    for i in uc_indices:
        g = generators[i]
        cap_size = g['capacity_size_MW']
        min_out = g['min_output_frac']
        rup = g.get('ramp_up_frac_per_hr', 1.0)
        rdn = g.get('ramp_dn_frac_per_hr', 1.0)
        min_up = g.get('min_up_time_hr', 1)
        min_dn = g.get('min_down_time_hr', 1)

        for t in range(total_timesteps):
            av = get_avail(g['id'], t)
            # Commit limit
            model.addConstr(commit_var[i, t] * cap_size <= y_P_total[i], name=f"uc_clim_{i}_{t}")
            # Startup/shutdown limits
            model.addConstr(startup_var[i, t] * cap_size <= y_P_total[i], name=f"uc_sulim_{i}_{t}")
            model.addConstr(shutdown_var[i, t] * cap_size <= y_P_total[i], name=f"uc_sdlim_{i}_{t}")
            # Gen bounds
            model.addConstr(gen_var[i, t] >= commit_var[i, t] * min_out * cap_size, name=f"uc_gmin_{i}_{t}")
            model.addConstr(gen_var[i, t] <= commit_var[i, t] * av * cap_size, name=f"uc_gmax_{i}_{t}")

        # Commitment balance
        for sp in range(num_subperiods):
            hours = sp_hours(sp)
            for t in hours:
                tp = prev_hour(t, hours[0])
                model.addConstr(
                    commit_var[i, t] - commit_var[i, tp] == startup_var[i, t] - shutdown_var[i, t],
                    name=f"uc_bal_{i}_{t}"
                )

            # Ramp constraints for UC
            for t in hours:
                tp = prev_hour(t, hours[0])
                av_t = get_avail(g['id'], t)
                av_tp = get_avail(g['id'], tp)

                # Ramp up
                su_coeff = cap_size * min(av_t, max(min_out, rup))
                model.addConstr(
                    gen_var[i, t] - gen_var[i, tp] <=
                    cap_size * rup * (commit_var[i, t] - startup_var[i, t])
                    + su_coeff * startup_var[i, t]
                    - cap_size * min_out * shutdown_var[i, t],
                    name=f"uc_rup_{i}_{t}"
                )

                # Ramp down
                sd_coeff = cap_size * min(av_tp, max(min_out, rdn))
                model.addConstr(
                    gen_var[i, tp] - gen_var[i, t] <=
                    cap_size * rdn * (commit_var[i, t] - startup_var[i, t])
                    + sd_coeff * shutdown_var[i, t]
                    - cap_size * min_out * startup_var[i, t],
                    name=f"uc_rdn_{i}_{t}"
                )

            # Min up time
            for idx_in_sp, t in enumerate(hours):
                lhs = gp.LinExpr()
                for k in range(min_up):
                    past_idx = (idx_in_sp - k) % hours_per_subperiod
                    lhs.add(startup_var[i, hours[past_idx]], 1.0)
                model.addConstr(commit_var[i, t] >= lhs, name=f"uc_mut_{i}_{t}")

            # Min down time
            for idx_in_sp, t in enumerate(hours):
                lhs = gp.LinExpr()
                lhs.add(commit_var[i, t], 1.0)
                for k in range(min_dn):
                    past_idx = (idx_in_sp - k) % hours_per_subperiod
                    lhs.add(shutdown_var[i, hours[past_idx]], 1.0)
                # Total units available = y_P_total / cap_size
                # commit + sum(shutdowns) <= y_P_total / cap_size
                # Linearize: commit * cap_size + sum(shutdowns) * cap_size <= y_P_total
                model.addConstr(lhs * cap_size <= y_P_total[i], name=f"uc_mdt_{i}_{t}")

    # ========== POLICY CONSTRAINTS ==========
    if rps_enabled:
        lhs = gp.LinExpr()
        for i in rps_indices:
            for t in range(total_timesteps):
                lhs.add(gen_var[i, t], hour_weights[t])
        for sp in range(num_subperiods):
            lhs.add(rps_slack[sp], 1.0)
        model.addConstr(lhs >= rps_share * total_weighted_demand, name="rps")

    if co2_cap_enabled and co2_cap is not None:
        lhs = gp.LinExpr()
        for i, g in enumerate(generators):
            if g['co2_tons_per_MWh'] > 0:
                for t in range(total_timesteps):
                    lhs.add(gen_var[i, t], hour_weights[t] * g['co2_tons_per_MWh'])
        # Add storage withdrawal emissions (typically 0 but include per problem description)
        for i in storage_indices:
            g = generators[i]
            if g['co2_tons_per_MWh'] > 0:
                for t in range(total_timesteps):
                    lhs.add(withdraw_var[i, t], hour_weights[t] * g['co2_tons_per_MWh'])
        for sp in range(num_subperiods):
            lhs.add(co2_slack[sp], -1.0)
        model.addConstr(lhs <= co2_cap, name="co2_cap")

    # ========== OBJECTIVE ==========
    obj = gp.LinExpr()

    # Fixed costs
    for i, g in enumerate(generators):
        cap_size = g['capacity_size_MW']
        # Investment cost
        obj.add(y_P_new[i], g['inv_cost_per_MW_yr'] * cap_size)
        # FOM
        obj.add(y_P_total[i], g['fom_cost_per_MW_yr'])

        if g['is_hydro']:
            dur = g['duration_MWh_per_MW']
            # Hydro energy investment cost
            inv_e = g.get('inv_cost_energy_per_MWh_yr', 0)
            if inv_e and inv_e > 0:
                obj.add(y_P_new[i], inv_e * dur * cap_size)
            fom_e = g.get('fom_cost_energy_per_MWh_yr', 0)
            if fom_e and fom_e > 0:
                obj.add(y_P_total[i], fom_e * dur)

    for i in storage_indices:
        g = generators[i]
        e_size = g['storage_capacity_size_MWh']
        obj.add(y_E_new[i], g['inv_cost_energy_per_MWh_yr'] * e_size)
        obj.add(y_E_total[i], g['fom_cost_energy_per_MWh_yr'])

    for j, ln in enumerate(lines):
        obj.add(y_F_new[j], ln['inv_cost_per_MW_yr'])

    # Variable costs
    for i, g in enumerate(generators):
        vc = g['var_cost_per_MWh']
        if vc > 0:
            for t in range(total_timesteps):
                obj.add(gen_var[i, t], hour_weights[t] * vc)
        if g['is_storage'] and vc > 0:
            for t in range(total_timesteps):
                obj.add(withdraw_var[i, t], hour_weights[t] * vc)

    # NSE costs
    for s, seg in enumerate(nse_segments):
        for z in range(1, num_zones + 1):
            for t in range(total_timesteps):
                obj.add(nse_var[s, z, t], hour_weights[t] * seg['cost_per_MWh'])

    # Startup costs
    for i in uc_indices:
        g = generators[i]
        sc = g['start_cost_per_unit']
        if sc > 0:
            for t in range(total_timesteps):
                obj.add(startup_var[i, t], hour_weights[t] * sc)

    # Policy noncompliance
    if rps_enabled:
        for sp in range(num_subperiods):
            obj.add(rps_slack[sp], rps_penalty)

    if co2_cap_enabled:
        for sp in range(num_subperiods):
            obj.add(co2_slack[sp], co2_penalty)

    model.setObjective(obj, GRB.MINIMIZE)

    # Callback for logging
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj_val)

    model.optimize(callback)

    # Extract solution
    if model.SolCount == 0:
        # No solution found - try to return something
        result = {
            "objective_value": float('inf'),
            "investment_decisions": [],
            "transmission_decisions": []
        }
        with open(solution_path, 'w') as f:
            json.dump(result, f, indent=2)
        return

    obj_val = model.ObjVal
    if logger:
        logger.log(obj_val)

    investment_decisions = []
    for i, g in enumerate(generators):
        entry = {
            "generator_id": g['id'],
            "resource_type": g['resource_type'],
            "zone": g['zone'],
            "y_P_new": round(y_P_new[i].X),
            "y_P_ret": round(y_P_ret[i].X),
            "y_P_total_MW": y_P_total[i].X
        }
        if g['is_storage']:
            entry["y_E_new"] = round(y_E_new[i].X)
            entry["y_E_ret"] = round(y_E_ret[i].X)
            entry["y_E_total_MWh"] = y_E_total[i].X
        investment_decisions.append(entry)

    transmission_decisions = []
    for j, ln in enumerate(lines):
        transmission_decisions.append({
            "line_id": ln['id'],
            "y_F_new_MW": round(y_F_new[j].X),
            "total_capacity_MW": y_F_total[j].X
        })

    result = {
        "objective_value": obj_val,
        "investment_decisions": investment_decisions,
        "transmission_decisions": transmission_decisions
    }

    with open(solution_path, 'w') as f:
        json.dump(result, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)
import argparse
import json
import time
import math
import sys
from collections import defaultdict

def solve(instance_path, solution_path, time_limit, log_path):
    from solution_logger import SolutionLogger
    import gurobipy as gp
    from gurobipy import GRB

    start_time = time.time()

    with open(instance_path, 'r') as f:
        data = json.load(f)

    logger = SolutionLogger(log_path, sense="minimize") if log_path else None

    dims = data['problem_dimensions']
    num_zones = dims['num_zones']
    num_weeks = dims['num_weeks']
    hours_per_week = dims['hours_per_week']
    total_timesteps = dims['total_timesteps']
    num_generators = dims['num_generators']
    num_storage = dims['num_storage_resources']
    num_hydro = dims['num_hydro_resources']
    num_lines = dims['num_transmission_lines']
    num_demand_segments = dims['num_demand_segments']

    policy = data['policy']
    subperiods = data['subperiods']
    num_subperiods = subperiods['num_subperiods']
    hours_per_subperiod = subperiods['hours_per_subperiod']
    sub_weights = subperiods['subperiod_weights']

    # Compute hour weights
    hour_weights = []
    for w in range(num_subperiods):
        wt = sub_weights[w] / hours_per_subperiod
        for h in range(hours_per_subperiod):
            hour_weights.append(wt)

    generators = data['generators']
    lines = data['transmission_lines']
    demand = data['demand']
    nse_segments = data['nse_segments']
    avail = data['availability_profiles']

    # Preprocess availability profiles
    def get_avail(gen_id, t):
        prof = avail[str(gen_id)]
        if prof['type'] == 'constant':
            return prof['value']
        else:
            return prof['values'][t]

    # Precompute availability arrays
    avail_arr = {}
    for g in generators:
        gid = g['id']
        prof = avail[str(gid)]
        if prof['type'] == 'constant':
            avail_arr[gid] = [prof['value']] * total_timesteps
        else:
            avail_arr[gid] = prof['values']

    # Precompute hydro inflow profiles
    hydro_inflow = {}
    for g in generators:
        if g['is_hydro']:
            key = f"hydro_inflow_{g['id']}"
            if key in data:
                hydro_inflow[g['id']] = data[key]

    # Build zone-to-generator mapping
    zone_gens = defaultdict(list)
    for g in generators:
        zone_gens[g['zone']].append(g)

    # Build zone-to-line mapping
    zone_lines_from = defaultdict(list)
    zone_lines_to = defaultdict(list)
    for l in lines:
        zone_lines_from[l['from_zone']].append(l)
        zone_lines_to[l['to_zone']].append(l)

    # Subperiod time mapping
    subperiod_hours = []
    for w in range(num_subperiods):
        start_h = w * hours_per_subperiod
        subperiod_hours.append(list(range(start_h, start_h + hours_per_subperiod)))

    # Compute total weighted demand
    total_weighted_demand = 0.0
    for z_str, dem_list in demand.items():
        for t in range(total_timesteps):
            total_weighted_demand += hour_weights[t] * dem_list[t]

    # Create model
    model = gp.Model("capacity_expansion")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = max(1, time_limit - 30)

    # ============ VARIABLES ============

    # Generator investment variables
    yPnew = {}
    yPret = {}
    yP = {}
    for g in generators:
        gid = g['id']
        cap_size = g['capacity_size_MW']
        existing = g['existing_capacity_MW']
        max_cap = g['max_capacity_MW']
        max_new = max_cap - existing
        max_new_units = max(0, math.floor(max_new / cap_size + 0.5)) if cap_size > 0 else 0
        max_ret_units = max(0, math.floor(existing / cap_size + 0.5)) if cap_size > 0 else 0

        if g['is_UC']:
            yPnew[gid] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=max_new_units, name=f"yPnew_{gid}")
            if g['can_retire']:
                yPret[gid] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=max_ret_units, name=f"yPret_{gid}")
            else:
                yPret[gid] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=0, name=f"yPret_{gid}")
        else:
            yPnew[gid] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=max_new_units, name=f"yPnew_{gid}")
            if g['can_retire']:
                yPret[gid] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=max_ret_units, name=f"yPret_{gid}")
            else:
                yPret[gid] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=0, name=f"yPret_{gid}")

        yP[gid] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"yP_{gid}")

    # Storage energy investment variables
    yEnew = {}
    yEret = {}
    yE = {}
    for g in generators:
        if g['is_storage']:
            gid = g['id']
            ecap_size = g['storage_capacity_size_MWh']
            existing_e = g['existing_storage_capacity_MWh']
            max_e = g['max_storage_capacity_MWh']
            max_new_e = max_e - existing_e
            max_new_e_units = max(0, math.floor(max_new_e / ecap_size + 0.5)) if ecap_size > 0 else 0
            max_ret_e_units = max(0, math.floor(existing_e / ecap_size + 0.5)) if ecap_size > 0 else 0

            yEnew[gid] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=max_new_e_units, name=f"yEnew_{gid}")
            if g['can_retire']:
                yEret[gid] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=max_ret_e_units, name=f"yEret_{gid}")
            else:
                yEret[gid] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=0, name=f"yEret_{gid}")
            yE[gid] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"yE_{gid}")

    # Transmission investment variables
    yFnew = {}
    for l in lines:
        lid = l['id']
        max_new_f = l['max_new_capacity_MW']
        yFnew[lid] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=max_new_f, name=f"yFnew_{lid}")

    # Operational variables
    xinj = {}  # generation
    xwdw = {}  # withdrawal (storage)
    xsoc = {}  # state of charge (storage)
    xhydro_level = {}  # hydro reservoir level
    xhydro_spill = {}  # hydro spillage
    xflow = {}  # transmission flow
    xnse = {}  # non-served energy
    xcommit = {}  # unit commitment
    xstart = {}  # startups
    xshut = {}  # shutdowns

    for g in generators:
        gid = g['id']
        for t in range(total_timesteps):
            xinj[gid, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"xinj_{gid}_{t}")

        if g['is_storage']:
            for t in range(total_timesteps):
                xwdw[gid, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"xwdw_{gid}_{t}")
                xsoc[gid, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"xsoc_{gid}_{t}")

        if g['is_hydro']:
            for t in range(total_timesteps):
                xhydro_level[gid, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"xhydro_{gid}_{t}")
                xhydro_spill[gid, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"xspill_{gid}_{t}")

        if g['is_UC']:
            max_units = math.floor((g['existing_capacity_MW'] + g['max_capacity_MW'] - g['existing_capacity_MW']) / g['capacity_size_MW'] + 0.5)
            for t in range(total_timesteps):
                xcommit[gid, t] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=max_units, name=f"xcommit_{gid}_{t}")
                xstart[gid, t] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=max_units, name=f"xstart_{gid}_{t}")
                xshut[gid, t] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=max_units, name=f"xshut_{gid}_{t}")

    for l in lines:
        lid = l['id']
        max_total = l['existing_capacity_MW'] + l['max_new_capacity_MW']
        for t in range(total_timesteps):
            xflow[lid, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=-max_total, ub=max_total, name=f"xflow_{lid}_{t}")

    for seg in nse_segments:
        s = seg['segment']
        for z_str in demand:
            z = int(z_str)
            for t in range(total_timesteps):
                max_nse = seg['max_frac'] * demand[z_str][t]
                xnse[s, z, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=max(0, max_nse), name=f"xnse_{s}_{z}_{t}")

    # Policy slack variables
    xRPS = {}
    xCO2 = {}
    if policy['RPS_enabled']:
        for w in range(num_subperiods):
            xRPS[w] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"xRPS_{w}")
    if policy['CO2_cap_enabled']:
        for w in range(num_subperiods):
            xCO2[w] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"xCO2_{w}")

    model.update()

    # ============ CONSTRAINTS ============

    # Capacity definition constraints
    for g in generators:
        gid = g['id']
        cap_size = g['capacity_size_MW']
        existing = g['existing_capacity_MW']
        model.addConstr(yP[gid] == existing + cap_size * (yPnew[gid] - yPret[gid]), name=f"cap_def_{gid}")

    # Storage energy capacity definition
    for g in generators:
        if g['is_storage']:
            gid = g['id']
            ecap_size = g['storage_capacity_size_MWh']
            existing_e = g['existing_storage_capacity_MWh']
            model.addConstr(yE[gid] == existing_e + ecap_size * (yEnew[gid] - yEret[gid]), name=f"ecap_def_{gid}")

            # Duration constraints
            min_dur = g['min_duration_MWh_per_MW']
            max_dur = g['max_duration_MWh_per_MW']
            model.addConstr(yE[gid] >= min_dur * yP[gid], name=f"min_dur_{gid}")
            model.addConstr(yE[gid] <= max_dur * yP[gid], name=f"max_dur_{gid}")

    # Total line capacity
    line_total_cap = {}
    for l in lines:
        lid = l['id']
        line_total_cap[lid] = l['existing_capacity_MW'] + yFnew[lid]

    # Power balance constraints
    for z_str, dem_list in demand.items():
        z = int(z_str)
        for t in range(total_timesteps):
            lhs = gp.LinExpr()
            # Generation from resources in zone
            for g in zone_gens[z]:
                gid = g['id']
                lhs.add(xinj[gid, t], 1.0)
                if g['is_storage']:
                    lhs.add(xwdw[gid, t], -1.0)
            # Transmission flows
            for l in zone_lines_from[z]:
                lhs.add(xflow[l['id'], t], -1.0)
            for l in zone_lines_to[z]:
                lhs.add(xflow[l['id'], t], 1.0)
            # NSE
            for seg in nse_segments:
                s = seg['segment']
                lhs.add(xnse[s, z, t], 1.0)
            model.addConstr(lhs == dem_list[t], name=f"pbal_{z}_{t}")

    # Generation limits for non-UC resources
    for g in generators:
        gid = g['id']
        if not g['is_UC']:
            for t in range(total_timesteps):
                av = avail_arr[gid][t]
                model.addConstr(xinj[gid, t] <= av * yP[gid], name=f"gen_max_{gid}_{t}")

    # Storage withdrawal limits and combined limits
    for g in generators:
        if g['is_storage']:
            gid = g['id']
            charge_eff = g['charge_efficiency']
            discharge_eff = g['discharge_efficiency']
            self_disc = g['self_discharge_rate']

            for t in range(total_timesteps):
                av = avail_arr[gid][t]
                # Withdrawal limit
                model.addConstr(xwdw[gid, t] <= av * yP[gid], name=f"wdw_max_{gid}_{t}")
                # Combined limit
                model.addConstr(xinj[gid, t] + xwdw[gid, t] <= yP[gid], name=f"comb_{gid}_{t}")
                # SOC limit
                model.addConstr(xsoc[gid, t] <= yE[gid], name=f"soc_max_{gid}_{t}")
                # Charging limit: charge_eff * withdrawal <= energy_cap - soc
                model.addConstr(charge_eff * xwdw[gid, t] <= yE[gid] - xsoc[gid, t], name=f"charge_lim_{gid}_{t}")
                # Discharging limit: generation / discharge_eff <= soc
                model.addConstr(xinj[gid, t] / discharge_eff <= xsoc[gid, t], name=f"disc_lim_{gid}_{t}")

    # Hydro reservoir limits
    for g in generators:
        if g['is_hydro']:
            gid = g['id']
            dur = g['duration_MWh_per_MW']
            for t in range(total_timesteps):
                model.addConstr(xhydro_level[gid, t] <= dur * yP[gid], name=f"hydro_max_{gid}_{t}")

    # Minimum output for non-UC, non-storage, non-hydro resources
    for g in generators:
        gid = g['id']
        if not g['is_UC'] and not g['is_storage'] and not g['is_hydro']:
            min_out = g['min_output_frac']
            if min_out > 0:
                for t in range(total_timesteps):
                    model.addConstr(xinj[gid, t] >= min_out * yP[gid], name=f"gen_min_{gid}_{t}")

    # Minimum output for hydro: generation + spillage >= min_output_frac * capacity
    for g in generators:
        if g['is_hydro']:
            gid = g['id']
            min_out = g['min_output_frac']
            if min_out > 0:
                for t in range(total_timesteps):
                    model.addConstr(xinj[gid, t] + xhydro_spill[gid, t] >= min_out * yP[gid], name=f"hydro_min_{gid}_{t}")

    # Transmission flow limits
    for l in lines:
        lid = l['id']
        for t in range(total_timesteps):
            total_cap_expr = l['existing_capacity_MW'] + yFnew[lid]
            model.addConstr(xflow[lid, t] <= total_cap_expr, name=f"flow_max_{lid}_{t}")
            model.addConstr(xflow[lid, t] >= -total_cap_expr, name=f"flow_min_{lid}_{t}")

    # Storage state of charge dynamics (circular within subperiod)
    for g in generators:
        if g['is_storage']:
            gid = g['id']
            charge_eff = g['charge_efficiency']
            discharge_eff = g['discharge_efficiency']
            self_disc = g['self_discharge_rate']

            for w in range(num_subperiods):
                hours = subperiod_hours[w]
                for h_idx in range(len(hours)):
                    t = hours[h_idx]
                    if h_idx == 0:
                        t_prev = hours[-1]  # circular
                    else:
                        t_prev = hours[h_idx - 1]

                    model.addConstr(
                        xsoc[gid, t] - (1 - self_disc) * xsoc[gid, t_prev] ==
                        charge_eff * xwdw[gid, t] - xinj[gid, t] / discharge_eff,
                        name=f"soc_bal_{gid}_{t}"
                    )

    # Hydro reservoir dynamics
    for g in generators:
        if g['is_hydro']:
            gid = g['id']
            dur = g['duration_MWh_per_MW']

            # Get inflow profile
            inflow_key = f"hydro_inflow_{gid}"
            if inflow_key in data:
                inflow_profile = data[inflow_key]
            else:
                # Default: constant inflow
                inflow_profile = [1.0 / total_timesteps] * total_timesteps

            for w in range(num_subperiods):
                hours = subperiod_hours[w]
                for h_idx in range(len(hours)):
                    t = hours[h_idx]
                    if h_idx == 0:
                        t_prev = hours[-1]
                    else:
                        t_prev = hours[h_idx - 1]

                    inflow_frac = inflow_profile[t] if t < len(inflow_profile) else 0.0
                    model.addConstr(
                        xhydro_level[gid, t] - xhydro_level[gid, t_prev] ==
                        inflow_frac * yP[gid] - xinj[gid, t] - xhydro_spill[gid, t],
                        name=f"hydro_bal_{gid}_{t}"
                    )

    # Ramping constraints for non-UC, non-storage, non-hydro resources
    for g in generators:
        gid = g['id']
        if not g['is_UC'] and not g['is_storage'] and not g['is_hydro']:
            ramp_up = g.get('ramp_up_frac_per_hr', 1.0)
            ramp_dn = g.get('ramp_dn_frac_per_hr', 1.0)
            if ramp_up < 1.0 or ramp_dn < 1.0:
                for w in range(num_subperiods):
                    hours = subperiod_hours[w]
                    for h_idx in range(len(hours)):
                        t = hours[h_idx]
                        if h_idx == 0:
                            t_prev = hours[-1]
                        else:
                            t_prev = hours[h_idx - 1]

                        if ramp_up < 1.0:
                            model.addConstr(
                                xinj[gid, t] - xinj[gid, t_prev] <= ramp_up * yP[gid],
                                name=f"ramp_up_{gid}_{t}"
                            )
                        if ramp_dn < 1.0:
                            model.addConstr(
                                xinj[gid, t_prev] - xinj[gid, t] <= ramp_dn * yP[gid],
                                name=f"ramp_dn_{gid}_{t}"
                            )

    # Ramping for hydro (if ramp rates exist)
    for g in generators:
        if g['is_hydro']:
            gid = g['id']
            ramp_up = g.get('ramp_up_frac_per_hr', 1.0)
            ramp_dn = g.get('ramp_dn_frac_per_hr', 1.0)
            if ramp_up < 1.0 or ramp_dn < 1.0:
                for w in range(num_subperiods):
                    hours = subperiod_hours[w]
                    for h_idx in range(len(hours)):
                        t = hours[h_idx]
                        if h_idx == 0:
                            t_prev = hours[-1]
                        else:
                            t_prev = hours[h_idx - 1]
                        if ramp_up < 1.0:
                            model.addConstr(
                                xinj[gid, t] - xinj[gid, t_prev] <= ramp_up * yP[gid],
                                name=f"hydro_ramp_up_{gid}_{t}"
                            )
                        if ramp_dn < 1.0:
                            model.addConstr(
                                xinj[gid, t_prev] - xinj[gid, t] <= ramp_dn * yP[gid],
                                name=f"hydro_ramp_dn_{gid}_{t}"
                            )

    # Unit commitment constraints
    for g in generators:
        if g['is_UC']:
            gid = g['id']
            cap_size = g['capacity_size_MW']
            min_out = g['min_output_frac']
            ramp_up = g.get('ramp_up_frac_per_hr', 1.0)
            ramp_dn = g.get('ramp_dn_frac_per_hr', 1.0)
            min_up = g['min_up_time_hr']
            min_dn = g['min_down_time_hr']
            start_cost = g['start_cost_per_unit']

            # Committed units <= total units
            for t in range(total_timesteps):
                model.addConstr(xcommit[gid, t] * cap_size <= yP[gid], name=f"commit_max_{gid}_{t}")
                model.addConstr(xstart[gid, t] * cap_size <= yP[gid], name=f"start_max_{gid}_{t}")
                model.addConstr(xshut[gid, t] * cap_size <= yP[gid], name=f"shut_max_{gid}_{t}")

            # Generation bounds
            for t in range(total_timesteps):
                av = avail_arr[gid][t]
                model.addConstr(xinj[gid, t] >= xcommit[gid, t] * min_out * cap_size, name=f"uc_gen_min_{gid}_{t}")
                model.addConstr(xinj[gid, t] <= xcommit[gid, t] * av * cap_size, name=f"uc_gen_max_{gid}_{t}")

            # Commitment balance: commit[t] - commit[t-1] = start[t] - shut[t]
            for w in range(num_subperiods):
                hours = subperiod_hours[w]
                for h_idx in range(len(hours)):
                    t = hours[h_idx]
                    if h_idx == 0:
                        t_prev = hours[-1]
                    else:
                        t_prev = hours[h_idx - 1]

                    model.addConstr(
                        xcommit[gid, t] - xcommit[gid, t_prev] == xstart[gid, t] - xshut[gid, t],
                        name=f"commit_bal_{gid}_{t}"
                    )

            # Ramp limits for UC resources
            for w in range(num_subperiods):
                hours = subperiod_hours[w]
                for h_idx in range(len(hours)):
                    t = hours[h_idx]
                    if h_idx == 0:
                        t_prev = hours[-1]
                    else:
                        t_prev = hours[h_idx - 1]

                    av_t = avail_arr[gid][t]
                    av_prev = avail_arr[gid][t_prev]

                    # Ramp up
                    ramp_up_start_coeff = min(av_t, max(min_out, ramp_up))
                    model.addConstr(
                        xinj[gid, t] - xinj[gid, t_prev] <=
                        cap_size * ramp_up * (xcommit[gid, t] - xstart[gid, t]) +
                        cap_size * ramp_up_start_coeff * xstart[gid, t] -
                        cap_size * min_out * xshut[gid, t],
                        name=f"uc_ramp_up_{gid}_{t}"
                    )

                    # Ramp down
                    ramp_dn_shut_coeff = min(av_prev, max(min_out, ramp_dn))
                    model.addConstr(
                        xinj[gid, t_prev] - xinj[gid, t] <=
                        cap_size * ramp_dn * (xcommit[gid, t] - xstart[gid, t]) +
                        cap_size * ramp_dn_shut_coeff * xshut[gid, t] -
                        cap_size * min_out * xstart[gid, t],
                        name=f"uc_ramp_dn_{gid}_{t}"
                    )

            # Minimum up-time constraints
            if min_up > 1:
                for w in range(num_subperiods):
                    hours = subperiod_hours[w]
                    n_hours = len(hours)
                    for h_idx in range(n_hours):
                        t = hours[h_idx]
                        startup_sum = gp.LinExpr()
                        for k in range(min_up):
                            prev_idx = (h_idx - k) % n_hours
                            startup_sum.add(xstart[gid, hours[prev_idx]], 1.0)
                        model.addConstr(xcommit[gid, t] >= startup_sum, name=f"min_up_{gid}_{t}")

            # Minimum down-time constraints
            if min_dn > 1:
                for w in range(num_subperiods):
                    hours = subperiod_hours[w]
                    n_hours = len(hours)
                    for h_idx in range(n_hours):
                        t = hours[h_idx]
                        shutdown_sum = gp.LinExpr()
                        for k in range(min_dn):
                            prev_idx = (h_idx - k) % n_hours
                            shutdown_sum.add(xshut[gid, hours[prev_idx]], 1.0)
                        # commit[t] + sum(shutdowns) <= total_units
                        model.addConstr(
                            xcommit[gid, t] + shutdown_sum <= yP[gid] / cap_size,
                            name=f"min_dn_{gid}_{t}"
                        )

    # RPS constraint
    if policy['RPS_enabled']:
        rps_share = policy['RPS_share']
        rps_lhs = gp.LinExpr()
        for g in generators:
            if g['is_RPS_qualifying']:
                gid = g['id']
                for t in range(total_timesteps):
                    rps_lhs.add(xinj[gid, t], hour_weights[t])
        for w in range(num_subperiods):
            rps_lhs.add(xRPS[w], 1.0)
        model.addConstr(rps_lhs >= rps_share * total_weighted_demand, name="rps")

    # CO2 cap constraint
    if policy['CO2_cap_enabled']:
        co2_cap = policy['CO2_cap_tons_per_MWh'] * total_weighted_demand
        co2_lhs = gp.LinExpr()
        for g in generators:
            gid = g['id']
            co2_rate = g['co2_tons_per_MWh']
            if co2_rate > 0:
                for t in range(total_timesteps):
                    co2_lhs.add(xinj[gid, t], hour_weights[t] * co2_rate)
                    if g['is_storage']:
                        co2_lhs.add(xwdw[gid, t], hour_weights[t] * co2_rate)
        for w in range(num_subperiods):
            co2_lhs.add(xCO2[w], -1.0)
        model.addConstr(co2_lhs <= co2_cap, name="co2_cap")

    # ============ OBJECTIVE ============
    obj = gp.LinExpr()

    # Fixed costs
    for g in generators:
        gid = g['id']
        cap_size = g['capacity_size_MW']
        inv_cost = g['inv_cost_per_MW_yr']
        fom_cost = g['fom_cost_per_MW_yr']

        # Investment cost
        obj.add(yPnew[gid], inv_cost * cap_size)
        # FOM cost
        obj.add(yP[gid], fom_cost)

        # Hydro energy investment and FOM
        if g['is_hydro']:
            dur = g['duration_MWh_per_MW']
            inv_cost_e = g.get('inv_cost_energy_per_MWh_yr', 0)
            fom_cost_e = g.get('fom_cost_energy_per_MWh_yr', 0)
            if inv_cost_e > 0:
                obj.add(yPnew[gid], inv_cost_e * dur * cap_size)
            if fom_cost_e > 0:
                obj.add(yP[gid], fom_cost_e * dur)

        # Storage energy investment and FOM
        if g['is_storage']:
            inv_cost_e = g['inv_cost_energy_per_MWh_yr']
            fom_cost_e = g['fom_cost_energy_per_MWh_yr']
            ecap_size = g['storage_capacity_size_MWh']
            obj.add(yEnew[gid], inv_cost_e * ecap_size)
            obj.add(yE[gid], fom_cost_e)

    # Transmission investment cost
    for l in lines:
        lid = l['id']
        obj.add(yFnew[lid], l['inv_cost_per_MW_yr'])

    # Variable operating costs
    for g in generators:
        gid = g['id']
        var_cost = g['var_cost_per_MWh']
        if var_cost > 0:
            for t in range(total_timesteps):
                obj.add(xinj[gid, t], hour_weights[t] * var_cost)
                if g['is_storage']:
                    obj.add(xwdw[gid, t], hour_weights[t] * var_cost)

    # NSE costs
    for seg in nse_segments:
        s = seg['segment']
        cost = seg['cost_per_MWh']
        for z_str in demand:
            z = int(z_str)
            for t in range(total_timesteps):
                obj.add(xnse[s, z, t], hour_weights[t] * cost)

    # Startup costs
    for g in generators:
        if g['is_UC']:
            gid = g['id']
            start_cost = g['start_cost_per_unit']
            if start_cost > 0:
                for t in range(total_timesteps):
                    obj.add(xstart[gid, t], hour_weights[t] * start_cost)

    # RPS noncompliance cost
    if policy['RPS_enabled']:
        rps_penalty = policy['RPS_noncompliance_cost_per_MWh']
        for w in range(num_subperiods):
            obj.add(xRPS[w], rps_penalty)

    # CO2 noncompliance cost
    if policy['CO2_cap_enabled']:
        co2_penalty = policy['CO2_noncompliance_cost_per_ton']
        for w in range(num_subperiods):
            obj.add(xCO2[w], co2_penalty)

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

    # ============ EXTRACT SOLUTION ============
    if model.SolCount == 0:
        # No feasible solution found
        solution = {
            "objective_value": float('inf'),
            "model_variables": {},
            "investment_decisions": [],
            "transmission_decisions": []
        }
        with open(solution_path, 'w') as f:
            json.dump(solution, f)
        return

    obj_val = model.ObjVal

    # Build model_variables dict (sparse)
    model_vars = {}

    def add_var(name, val):
        if abs(val) > 1e-9:
            model_vars[name] = val

    for g in generators:
        gid = g['id']
        add_var(f"yPnew_{gid}", yPnew[gid].X)
        add_var(f"yPret_{gid}", yPret[gid].X)
        add_var(f"yP_{gid}", yP[gid].X)

        if g['is_storage']:
            add_var(f"yEnew_{gid}", yEnew[gid].X)
            add_var(f"yEret_{gid}", yEret[gid].X)
            add_var(f"yE_{gid}", yE[gid].X)

        for t in range(total_timesteps):
            add_var(f"xinj_{gid}_{t}", xinj[gid, t].X)
            if g['is_storage']:
                add_var(f"xwdw_{gid}_{t}", xwdw[gid, t].X)
                add_var(f"xsoc_{gid}_{t}", xsoc[gid, t].X)
            if g['is_hydro']:
                add_var(f"xhydro_{gid}_{t}", xhydro_level[gid, t].X)
            if g['is_UC']:
                add_var(f"xcommit_{gid}_{t}", xcommit[gid, t].X)
                add_var(f"xstart_{gid}_{t}", xstart[gid, t].X)
                add_var(f"xshut_{gid}_{t}", xshut[gid, t].X)

    for l in lines:
        lid = l['id']
        add_var(f"yFnew_{lid}", yFnew[lid].X)
        for t in range(total_timesteps):
            add_var(f"xflow_{lid}_{t}", xflow[lid, t].X)

    for seg in nse_segments:
        s = seg['segment']
        for z_str in demand:
            z = int(z_str)
            for t in range(total_timesteps):
                add_var(f"xnse_{s}_{z}_{t}", xnse[s, z, t].X)

    if policy['RPS_enabled']:
        for w in range(num_subperiods):
            add_var(f"xRPS_{w}", xRPS[w].X)
    if policy['CO2_cap_enabled']:
        for w in range(num_subperiods):
            add_var(f"xCO2_{w}", xCO2[w].X)

    # Investment decisions
    inv_decisions = []
    for g in generators:
        gid = g['id']
        entry = {
            "generator_id": gid,
            "resource_type": g['resource_type'],
            "zone": g['zone'],
            "y_P_new": yPnew[gid].X,
            "y_P_ret": yPret[gid].X,
            "y_P_total_MW": yP[gid].X
        }
        if g['is_storage']:
            entry["y_E_new"] = yEnew[gid].X
            entry["y_E_ret"] = yEret[gid].X
            entry["y_E_total_MWh"] = yE[gid].X
        inv_decisions.append(entry)

    # Transmission decisions
    trans_decisions = []
    for l in lines:
        lid = l['id']
        trans_decisions.append({
            "line_id": lid,
            "y_F_new_MW": yFnew[lid].X,
            "total_capacity_MW": l['existing_capacity_MW'] + yFnew[lid].X
        })

    solution = {
        "objective_value": obj_val,
        "model_variables": model_vars,
        "investment_decisions": inv_decisions,
        "transmission_decisions": trans_decisions
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
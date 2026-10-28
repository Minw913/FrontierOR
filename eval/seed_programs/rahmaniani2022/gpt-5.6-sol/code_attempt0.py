import argparse
import json
import math
import os
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


TOL = 1e-7


def clean_number(value):
    value = float(value)
    if abs(value) < TOL:
        return 0.0
    nearest = round(value)
    if abs(value - nearest) < 1e-6:
        return float(nearest)
    return value


def get_param(parameters, resource_type, name, default=0.0):
    return parameters.get(resource_type, {}).get(name, default)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    wall_start = time.monotonic()

    with open(args.instance_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pp = data["problem_parameters"]
    W = int(pp["num_weeks"])
    H = int(pp["hours_per_subperiod"])
    T = int(pp["total_hours"])

    generators = data["generators"]
    params = data["generator_parameters"]
    existing_capacity = {int(k): float(v) for k, v in data["existing_capacity"].items()}
    availability_data = {int(k): v for k, v in data["availability"].items()}
    hydro_inflow_data = {
        int(k): v for k, v in data.get("hydro_inflow", {}).items()
    }

    gen_by_id = {int(g["cluster_id"]): g for g in generators}
    generator_ids = [int(g["cluster_id"]) for g in generators]
    storage_ids = [int(g["cluster_id"]) for g in generators if g["is_storage"]]
    hydro_ids = [int(g["cluster_id"]) for g in generators if g["is_hydro"]]
    uc_ids = [int(g["cluster_id"]) for g in generators if g["is_UC"]]
    non_uc_ids = [i for i in generator_ids if i not in set(uc_ids)]

    lines = data.get("transmission_lines", [])
    line_by_id = {int(line["line_id"]): line for line in lines}
    line_ids = [int(line["line_id"]) for line in lines]

    segments = data.get("consumer_segments", [])
    zones = sorted(
        set(int(z) for z in data["demand"].keys())
        | set(int(g["zone"]) for g in generators)
    )
    demand = {int(k): [float(x) for x in v] for k, v in data["demand"].items()}
    weights = [float(x) for x in data["subperiod_weights"]]

    def week_of(t):
        return t // H

    def hour_weight(t):
        return weights[week_of(t)] / float(H)

    def previous_hour(t):
        start = (t // H) * H
        offset = t - start
        return start + ((offset - 1) % H)

    def preceding_hours_including_current(t, count):
        if count <= 0:
            return []
        count = min(int(count), H)
        start = (t // H) * H
        offset = t - start
        return [start + ((offset - k) % H) for k in range(count)]

    def power_increment(i):
        g = gen_by_id[i]
        return float(g["capacity_size_MW"]) if g["is_UC"] else 1.0

    def energy_increment(i):
        g = gen_by_id[i]
        p = params[g["resource_type"]]
        if g["is_UC"]:
            return float(p.get("energy_capacity_size_MWh", 1.0))
        return 1.0

    def existing_storage_energy(i):
        g = gen_by_id[i]
        p = params[g["resource_type"]]
        power_block = float(g.get("capacity_size_MW", p.get("capacity_size_MW", 1.0)))
        energy_block = float(p.get("energy_capacity_size_MWh", power_block))
        if power_block <= 0:
            return 0.0
        return existing_capacity.get(i, 0.0) * energy_block / power_block

    model = gp.Model("capacity_expansion_dispatch")
    model.Params.OutputFlag = 1
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1

    # Capacity investment variables.
    y_P_new = {}
    y_P_ret = {}
    y_P = {}

    for i in generator_ids:
        g = gen_by_id[i]
        rt = g["resource_type"]
        inc = power_increment(i)
        max_new_mw = float(get_param(params, rt, "max_capacity_MW", 0.0))
        new_ub = max(0, int(math.floor(max_new_mw / inc + 1e-9)))
        ret_ub = max(
            0,
            int(math.floor(existing_capacity.get(i, 0.0) / inc + 1e-9)),
        )
        if not g.get("can_retire", True):
            ret_ub = 0

        y_P_new[i] = model.addVar(
            lb=0, ub=new_ub, vtype=GRB.INTEGER, name=f"y_P_new[{i}]"
        )
        y_P_ret[i] = model.addVar(
            lb=0, ub=ret_ub, vtype=GRB.INTEGER, name=f"y_P_ret[{i}]"
        )
        y_P[i] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"y_P[{i}]")
        model.addConstr(
            y_P[i]
            == existing_capacity.get(i, 0.0)
            + inc * y_P_new[i]
            - inc * y_P_ret[i],
            name=f"power_capacity_balance[{i}]",
        )

    # Enforce resource-type/zone aggregate new-build limits.
    type_zone_groups = {}
    for i in generator_ids:
        g = gen_by_id[i]
        key = (g["resource_type"], int(g["zone"]))
        type_zone_groups.setdefault(key, []).append(i)

    for (rt, zone), ids in type_zone_groups.items():
        max_new = float(get_param(params, rt, "max_capacity_MW", 0.0))
        model.addConstr(
            gp.quicksum(power_increment(i) * y_P_new[i] for i in ids) <= max_new,
            name=f"new_power_limit[{rt},{zone}]",
        )

    # Storage energy capacity.
    y_E_new = {}
    y_E_ret = {}
    y_E = {}

    for i in storage_ids:
        g = gen_by_id[i]
        rt = g["resource_type"]
        p = params[rt]
        inc = energy_increment(i)
        existing_e = existing_storage_energy(i)
        max_new_e = float(p.get("max_energy_capacity_MWh", 0.0))
        new_ub = max(0, int(math.floor(max_new_e / inc + 1e-9)))
        ret_ub = max(0, int(math.floor(existing_e / inc + 1e-9)))
        if not g.get("can_retire", True):
            ret_ub = 0

        y_E_new[i] = model.addVar(
            lb=0, ub=new_ub, vtype=GRB.INTEGER, name=f"y_E_new[{i}]"
        )
        y_E_ret[i] = model.addVar(
            lb=0, ub=ret_ub, vtype=GRB.INTEGER, name=f"y_E_ret[{i}]"
        )
        y_E[i] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"y_E[{i}]")

        model.addConstr(
            y_E[i] == existing_e + inc * y_E_new[i] - inc * y_E_ret[i],
            name=f"energy_capacity_balance[{i}]",
        )

        min_duration = float(p.get("min_duration_MWh_per_MW", 0.0))
        max_duration = float(p.get("max_duration_MWh_per_MW", GRB.INFINITY))
        model.addConstr(
            y_E[i] >= min_duration * y_P[i],
            name=f"storage_min_duration[{i}]",
        )
        if math.isfinite(max_duration):
            model.addConstr(
                y_E[i] <= max_duration * y_P[i],
                name=f"storage_max_duration[{i}]",
            )

    # Aggregate storage energy-build limits by type and zone.
    for (rt, zone), ids in type_zone_groups.items():
        storage_group = [i for i in ids if i in y_E_new]
        if not storage_group:
            continue
        max_new_e = float(params[rt].get("max_energy_capacity_MWh", 0.0))
        model.addConstr(
            gp.quicksum(energy_increment(i) * y_E_new[i] for i in storage_group)
            <= max_new_e,
            name=f"new_energy_limit[{rt},{zone}]",
        )

    # Transmission investment.
    y_F_new = {}
    y_F = {}
    for l in line_ids:
        line = line_by_id[l]
        existing_f = float(line["existing_capacity_MW"])
        maximum_f = float(line["max_capacity_MW"])
        new_ub = max(0, int(math.floor(maximum_f - existing_f + 1e-9)))
        y_F_new[l] = model.addVar(
            lb=0, ub=new_ub, vtype=GRB.INTEGER, name=f"y_F_new[{l}]"
        )
        y_F[l] = model.addVar(lb=0, ub=maximum_f, name=f"y_F[{l}]")
        model.addConstr(
            y_F[l] == existing_f + y_F_new[l],
            name=f"transmission_capacity_balance[{l}]",
        )

    # Operational variables.
    x_inj = {}
    for i in generator_ids:
        for t in range(T):
            x_inj[i, t] = model.addVar(
                lb=0, name=f"x_inj[{i},{t + 1}]"
            )

    x_wdw = {}
    x_soc = {}
    for i in storage_ids:
        for t in range(T):
            x_wdw[i, t] = model.addVar(
                lb=0, name=f"x_wdw[{i},{t + 1}]"
            )
            x_soc[i, t] = model.addVar(
                lb=0, name=f"x_soc[{i},{t + 1}]"
            )

    x_lvl = {}
    x_spill = {}
    for i in hydro_ids:
        for t in range(T):
            x_lvl[i, t] = model.addVar(
                lb=0, name=f"x_lvl[{i},{t + 1}]"
            )
            x_spill[i, t] = model.addVar(
                lb=0, name=f"x_spill[{i},{t + 1}]"
            )

    x_commit = {}
    x_start = {}
    x_shut = {}
    for i in uc_ids:
        for t in range(T):
            x_commit[i, t] = model.addVar(
                lb=0, name=f"x_commit[{i},{t + 1}]"
            )
            x_start[i, t] = model.addVar(
                lb=0, name=f"x_start[{i},{t + 1}]"
            )
            x_shut[i, t] = model.addVar(
                lb=0, name=f"x_shut[{i},{t + 1}]"
            )

    x_flow = {}
    for l in line_ids:
        for t in range(T):
            x_flow[l, t] = model.addVar(
                lb=-GRB.INFINITY, name=f"x_flow[{l},{t + 1}]"
            )

    x_nse = {}
    for seg in segments:
        s = int(seg["segment_id"])
        z = int(seg["zone"])
        for t in range(T):
            x_nse[s, z, t] = model.addVar(
                lb=0, name=f"x_nse[{s},{z},{t + 1}]"
            )

    policy = data["policy"]
    policy_type = str(policy["type"]).upper()
    x_RPS = {}
    x_CO2 = {}
    if policy_type == "RPS":
        for w in range(W):
            x_RPS[w] = model.addVar(lb=0, name=f"x_RPS[{w + 1}]")
    elif policy_type == "CO2":
        for w in range(W):
            x_CO2[w] = model.addVar(lb=0, name=f"x_CO2[{w + 1}]")

    model.update()

    # Dispatch and unit-commitment constraints.
    for i in generator_ids:
        g = gen_by_id[i]
        rt = g["resource_type"]
        p = params[rt]
        min_output = float(p.get("min_output_frac", 0.0))
        ru = float(p.get("ramp_up_rate_pct_per_hr", 1.0))
        rd = float(p.get("ramp_down_rate_pct_per_hr", 1.0))
        unit_size = float(g["capacity_size_MW"])

        for t in range(T):
            avail = float(availability_data[i][t])
            prev = previous_hour(t)

            if g["is_UC"]:
                model.addConstr(
                    unit_size * x_commit[i, t] <= y_P[i],
                    name=f"commit_capacity[{i},{t + 1}]",
                )
                model.addConstr(
                    unit_size * x_start[i, t] <= y_P[i],
                    name=f"start_capacity[{i},{t + 1}]",
                )
                model.addConstr(
                    unit_size * x_shut[i, t] <= y_P[i],
                    name=f"shut_capacity[{i},{t + 1}]",
                )
                model.addConstr(
                    x_inj[i, t] >= unit_size * min_output * x_commit[i, t],
                    name=f"uc_min_generation[{i},{t + 1}]",
                )
                model.addConstr(
                    x_inj[i, t] <= unit_size * avail * x_commit[i, t],
                    name=f"uc_max_generation[{i},{t + 1}]",
                )
                model.addConstr(
                    x_commit[i, t] - x_commit[i, prev]
                    == x_start[i, t] - x_shut[i, t],
                    name=f"commitment_transition[{i},{t + 1}]",
                )

                start_level = min(avail, max(min_output, ru))
                shut_level = min(
                    float(availability_data[i][prev]),
                    max(min_output, rd),
                )

                model.addConstr(
                    x_inj[i, t] - x_inj[i, prev]
                    <= unit_size * ru * (x_commit[i, t] - x_start[i, t])
                    + unit_size * start_level * x_start[i, t]
                    - unit_size * min_output * x_shut[i, t],
                    name=f"uc_ramp_up[{i},{t + 1}]",
                )
                model.addConstr(
                    x_inj[i, prev] - x_inj[i, t]
                    <= unit_size * rd * (x_commit[i, t] - x_start[i, t])
                    + unit_size * shut_level * x_shut[i, t]
                    - unit_size * min_output * x_start[i, t],
                    name=f"uc_ramp_down[{i},{t + 1}]",
                )
            else:
                model.addConstr(
                    x_inj[i, t] <= avail * y_P[i],
                    name=f"nonuc_max_generation[{i},{t + 1}]",
                )
                if not g["is_storage"] and not g["is_hydro"] and min_output > 0:
                    model.addConstr(
                        x_inj[i, t] >= min_output * y_P[i],
                        name=f"nonuc_min_generation[{i},{t + 1}]",
                    )

                model.addConstr(
                    x_inj[i, t] - x_inj[i, prev] <= ru * y_P[i],
                    name=f"nonuc_ramp_up[{i},{t + 1}]",
                )
                model.addConstr(
                    x_inj[i, prev] - x_inj[i, t] <= rd * y_P[i],
                    name=f"nonuc_ramp_down[{i},{t + 1}]",
                )

    # UC minimum up/down constraints.
    for i in uc_ids:
        g = gen_by_id[i]
        p = params[g["resource_type"]]
        up_time = int(p.get("min_up_time_hr", 0))
        down_time = int(p.get("min_down_time_hr", 0))
        unit_size = float(g["capacity_size_MW"])

        for t in range(T):
            if up_time > 0:
                model.addConstr(
                    x_commit[i, t]
                    >= gp.quicksum(
                        x_start[i, tau]
                        for tau in preceding_hours_including_current(t, up_time)
                    ),
                    name=f"minimum_up_time[{i},{t + 1}]",
                )
            if down_time > 0:
                model.addConstr(
                    x_commit[i, t]
                    + gp.quicksum(
                        x_shut[i, tau]
                        for tau in preceding_hours_including_current(t, down_time)
                    )
                    <= y_P[i] / unit_size,
                    name=f"minimum_down_time[{i},{t + 1}]",
                )

    # Storage dynamics and operating limits.
    for i in storage_ids:
        g = gen_by_id[i]
        p = params[g["resource_type"]]
        eta_c = float(p.get("charging_efficiency", 1.0))
        eta_d = float(p.get("discharging_efficiency", 1.0))
        self_discharge = float(p.get("self_discharge_rate_pct", 0.0))

        for t in range(T):
            prev = previous_hour(t)
            avail = float(availability_data[i][t])

            model.addConstr(
                x_soc[i, t]
                == (1.0 - self_discharge) * x_soc[i, prev]
                + eta_c * x_wdw[i, t]
                - x_inj[i, t] / eta_d,
                name=f"storage_balance[{i},{t + 1}]",
            )
            model.addConstr(
                x_soc[i, t] <= y_E[i],
                name=f"storage_energy_limit[{i},{t + 1}]",
            )
            model.addConstr(
                eta_c * x_wdw[i, t] <= y_E[i] - x_soc[i, t],
                name=f"storage_charge_room[{i},{t + 1}]",
            )
            model.addConstr(
                x_inj[i, t] / eta_d <= x_soc[i, t],
                name=f"storage_discharge_energy[{i},{t + 1}]",
            )
            model.addConstr(
                x_inj[i, t] + x_wdw[i, t] <= y_P[i],
                name=f"storage_power_limit[{i},{t + 1}]",
            )
            model.addConstr(
                x_wdw[i, t] <= avail * y_P[i],
                name=f"storage_withdrawal_availability[{i},{t + 1}]",
            )

    # Hydropower reservoir constraints.
    for i in hydro_ids:
        g = gen_by_id[i]
        p = params[g["resource_type"]]
        duration = float(p.get("duration_MWh_per_MW", 0.0))
        min_output = float(p.get("min_output_frac", 0.0))
        inflow = hydro_inflow_data.get(i, [0.0] * T)

        for t in range(T):
            prev = previous_hour(t)
            model.addConstr(
                x_lvl[i, t]
                == x_lvl[i, prev]
                + float(inflow[t]) * y_P[i]
                - x_inj[i, t]
                - x_spill[i, t],
                name=f"hydro_balance[{i},{t + 1}]",
            )
            model.addConstr(
                x_lvl[i, t] <= duration * y_P[i],
                name=f"hydro_reservoir_limit[{i},{t + 1}]",
            )
            model.addConstr(
                x_inj[i, t] + x_spill[i, t] >= min_output * y_P[i],
                name=f"hydro_min_release[{i},{t + 1}]",
            )

    # Transmission limits.
    for l in line_ids:
        for t in range(T):
            model.addConstr(
                x_flow[l, t] <= y_F[l],
                name=f"flow_upper[{l},{t + 1}]",
            )
            model.addConstr(
                x_flow[l, t] >= -y_F[l],
                name=f"flow_lower[{l},{t + 1}]",
            )

    # Unserved-energy limits.
    for seg in segments:
        s = int(seg["segment_id"])
        z = int(seg["zone"])
        fraction = float(seg["max_nse_fraction"])
        for t in range(T):
            model.addConstr(
                x_nse[s, z, t] <= fraction * demand[z][t],
                name=f"nse_limit[{s},{z},{t + 1}]",
            )

    # Nodal power balance.
    generators_by_zone = {z: [] for z in zones}
    storage_by_zone = {z: [] for z in zones}
    segments_by_zone = {z: [] for z in zones}
    incoming_by_zone = {z: [] for z in zones}
    outgoing_by_zone = {z: [] for z in zones}

    for i in generator_ids:
        z = int(gen_by_id[i]["zone"])
        generators_by_zone.setdefault(z, []).append(i)
        if i in storage_ids:
            storage_by_zone.setdefault(z, []).append(i)

    for seg in segments:
        segments_by_zone.setdefault(int(seg["zone"]), []).append(int(seg["segment_id"]))

    for l in line_ids:
        line = line_by_id[l]
        outgoing_by_zone.setdefault(int(line["from_zone"]), []).append(l)
        incoming_by_zone.setdefault(int(line["to_zone"]), []).append(l)

    for z in zones:
        for t in range(T):
            model.addConstr(
                gp.quicksum(x_inj[i, t] for i in generators_by_zone.get(z, []))
                - gp.quicksum(x_wdw[i, t] for i in storage_by_zone.get(z, []))
                + gp.quicksum(x_flow[l, t] for l in incoming_by_zone.get(z, []))
                - gp.quicksum(x_flow[l, t] for l in outgoing_by_zone.get(z, []))
                + gp.quicksum(
                    x_nse[s, z, t] for s in segments_by_zone.get(z, [])
                )
                == demand[z][t],
                name=f"power_balance[{z},{t + 1}]",
            )

    total_weighted_demand = sum(
        hour_weight(t) * demand[z][t] for z in zones for t in range(T)
    )

    if policy_type == "RPS":
        qualifying_generation = gp.quicksum(
            hour_weight(t) * x_inj[i, t]
            for i in generator_ids
            if gen_by_id[i].get("is_RPS_qualifying", False)
            for t in range(T)
        )
        model.addConstr(
            qualifying_generation + gp.quicksum(x_RPS[w] for w in range(W))
            >= float(policy["RPS_share"]) * total_weighted_demand,
            name="RPS_requirement",
        )

    elif policy_type == "CO2":
        emissions = gp.LinExpr()
        for i in generator_ids:
            rt = gen_by_id[i]["resource_type"]
            rate = float(get_param(params, rt, "CO2_emission_tons_per_MWh", 0.0))
            if rate == 0:
                continue
            for t in range(T):
                emissions += hour_weight(t) * rate * x_inj[i, t]
                if i in storage_ids:
                    emissions += hour_weight(t) * rate * x_wdw[i, t]

        cap = float(policy["CO2_cap_tons_per_MWh"]) * total_weighted_demand
        model.addConstr(
            emissions - gp.quicksum(x_CO2[w] for w in range(W)) <= cap,
            name="CO2_cap",
        )

    # Objective.
    objective = gp.LinExpr()

    for i in generator_ids:
        g = gen_by_id[i]
        rt = g["resource_type"]
        p = params[rt]
        inc = power_increment(i)

        objective += (
            inc
            * float(p.get("investment_cost_per_MW_yr", 0.0))
            * y_P_new[i]
        )
        objective += float(p.get("fixed_OM_cost_per_MW_yr", 0.0)) * y_P[i]

        if g["is_hydro"]:
            duration = float(p.get("duration_MWh_per_MW", 0.0))
            objective += (
                duration
                * inc
                * float(p.get("energy_investment_cost_per_MWh_yr", 0.0))
                * y_P_new[i]
            )
            objective += (
                duration
                * float(p.get("energy_fixed_OM_cost_per_MWh_yr", 0.0))
                * y_P[i]
            )

        variable_cost = float(p.get("variable_cost_per_MWh", 0.0))
        for t in range(T):
            objective += hour_weight(t) * variable_cost * x_inj[i, t]
            if i in storage_ids:
                objective += hour_weight(t) * variable_cost * x_wdw[i, t]

        if g["is_UC"]:
            start_cost = float(p.get("start_up_cost", 0.0))
            for t in range(T):
                objective += hour_weight(t) * start_cost * x_start[i, t]

    for i in storage_ids:
        rt = gen_by_id[i]["resource_type"]
        p = params[rt]
        objective += (
            energy_increment(i)
            * float(p.get("energy_investment_cost_per_MWh_yr", 0.0))
            * y_E_new[i]
        )
        objective += (
            float(p.get("energy_fixed_OM_cost_per_MWh_yr", 0.0)) * y_E[i]
        )

    for l in line_ids:
        objective += (
            float(line_by_id[l]["investment_cost_per_MW_yr"]) * y_F_new[l]
        )

    for seg in segments:
        s = int(seg["segment_id"])
        z = int(seg["zone"])
        penalty = float(seg["cost_per_MWh"])
        for t in range(T):
            objective += hour_weight(t) * penalty * x_nse[s, z, t]

    if policy_type == "RPS":
        penalty = float(policy["RPS_noncompliance_cost_per_MWh"])
        objective += penalty * gp.quicksum(x_RPS[w] for w in range(W))
    elif policy_type == "CO2":
        penalty = float(policy["CO2_noncompliance_cost_per_ton"])
        objective += penalty * gp.quicksum(x_CO2[w] for w in range(W))

    model.setObjective(objective, GRB.MINIMIZE)
    model.update()

    all_vars = model.getVars()
    var_names = [v.VarName for v in all_vars]

    def build_solution(objective_value, values):
        value_by_var = {v: float(values[k]) for k, v in enumerate(all_vars)}

        model_variables = {}
        for name, value in zip(var_names, values):
            value = clean_number(value)
            if abs(value) > TOL:
                model_variables[name] = value

        investments = {}
        for i in generator_ids:
            g = gen_by_id[i]
            inc = power_increment(i)
            new_units = clean_number(value_by_var[y_P_new[i]])
            retired_units = clean_number(value_by_var[y_P_ret[i]])
            total_capacity = clean_number(value_by_var[y_P[i]])
            investments[str(i)] = {
                "resource_type": g["resource_type"],
                "zone": int(g["zone"]),
                "new_units": new_units,
                "new_MW": clean_number(inc * new_units),
                "retired_units": retired_units,
                "retired_MW": clean_number(inc * retired_units),
                "total_capacity_MW": total_capacity,
            }

        storage_energy_investments = {}
        for i in storage_ids:
            inc = energy_increment(i)
            new_units = clean_number(value_by_var[y_E_new[i]])
            storage_energy_investments[str(i)] = {
                "new_units": new_units,
                "new_MWh": clean_number(inc * new_units),
                "total_energy_MWh": clean_number(value_by_var[y_E[i]]),
            }

        transmission_investments = {}
        for l in line_ids:
            transmission_investments[str(l)] = {
                "new_MW": clean_number(value_by_var[y_F_new[l]]),
                "total_capacity_MW": clean_number(value_by_var[y_F[l]]),
            }

        return {
            "objective_value": clean_number(objective_value),
            "model_variables": model_variables,
            "investments": investments,
            "storage_energy_investments": storage_energy_investments,
            "transmission_investments": transmission_investments,
        }

    callback_state = {"best": math.inf}

    def incumbent_callback(cb_model, where):
        if where != GRB.Callback.MIPSOL or logger is None:
            return
        try:
            incumbent_obj = float(cb_model.cbGet(GRB.Callback.MIPSOL_OBJ))
            if incumbent_obj >= callback_state["best"] - 1e-6:
                return
            incumbent_values = cb_model.cbGetSolution(all_vars)
            solution = build_solution(incumbent_obj, incumbent_values)
            logger.log_solution(incumbent_obj, solution)
            callback_state["best"] = incumbent_obj
        except Exception:
            # Logging must not interrupt optimization.
            pass

    elapsed = time.monotonic() - wall_start
    remaining = max(0.01, float(args.time_limit) - elapsed)
    model.Params.TimeLimit = remaining

    model.optimize(incumbent_callback)

    if model.SolCount <= 0:
        raise RuntimeError(
            f"No feasible solution was found. Gurobi status: {model.Status}"
        )

    final_values = [v.X for v in all_vars]
    final_solution = build_solution(model.ObjVal, final_values)

    if logger:
        try:
            logger.log_solution(model.ObjVal, final_solution)
        except Exception:
            pass

    solution_dir = os.path.dirname(os.path.abspath(args.solution_path))
    if solution_dir:
        os.makedirs(solution_dir, exist_ok=True)

    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(final_solution, f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
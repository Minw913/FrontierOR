import argparse
import json
from typing import Dict, List, Any

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def get_profile_value(profile_obj: Dict[str, Any], t: int, T: int, default: float = 1.0) -> float:
    if not profile_obj:
        return default
    ptype = profile_obj.get("type", None)
    if ptype == "constant":
        return float(profile_obj.get("value", default))
    vals = profile_obj.get("values", None)
    if isinstance(vals, list) and len(vals) == T:
        return float(vals[t])
    return default


def get_timeseries(source: Any, T: int, default: float = 0.0) -> List[float]:
    if source is None:
        return [default] * T
    if isinstance(source, dict):
        ptype = source.get("type", None)
        if ptype == "constant":
            v = float(source.get("value", default))
            return [v] * T
        vals = source.get("values", None)
        if isinstance(vals, list) and len(vals) == T:
            return [float(x) for x in vals]
    if isinstance(source, list) and len(source) == T:
        return [float(x) for x in source]
    return [default] * T


def get_zone_demand(demand_obj: Dict[str, Any], z: int, T: int) -> List[float]:
    if str(z) in demand_obj:
        return [float(x) for x in demand_obj[str(z)]]
    if z in demand_obj:
        return [float(x) for x in demand_obj[z]]
    return [0.0] * T


def zonal_param(val, z, default):
    if isinstance(val, dict):
        if str(z) in val:
            return float(val[str(z)])
        if z in val:
            return float(val[z])
        return float(default)
    return float(val if val is not None else default)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    dims = inst["problem_dimensions"]
    policy = inst["policy"]
    subp = inst["subperiods"]

    Z = int(dims["num_zones"])
    T = int(dims["total_timesteps"])
    S = int(subp["num_subperiods"])
    H = int(subp["hours_per_subperiod"])
    sub_weights = [float(x) for x in subp["subperiod_weights"]]

    # Time consistency fallback
    if S * H != T:
        # fallback to dimensions
        H = int(dims.get("hours_per_week", H))
        S = int(dims.get("num_weeks", S))
        if S * H != T:
            S = max(1, S)
            H = T // S if T % S == 0 else T

    # Hour weights
    t_weight = []
    for s in range(S):
        w = sub_weights[s] / float(H)
        for _h in range(H):
            t_weight.append(w)
    if len(t_weight) != T:
        # robust fallback
        avg_w = (sum(sub_weights) / max(1, len(sub_weights))) / max(1, H)
        t_weight = [avg_w] * T

    generators = inst["generators"]
    lines = inst["transmission_lines"]
    nse_segments = inst["nse_segments"]
    availability_profiles = inst.get("availability_profiles", {})
    hydro_inflow_profiles = inst.get("hydro_inflow_profiles", {})

    g_ids = [int(g["id"]) for g in generators]
    gdata = {int(g["id"]): g for g in generators}

    uc_ids = [gid for gid in g_ids if bool(gdata[gid].get("is_UC", False))]
    storage_ids = [gid for gid in g_ids if bool(gdata[gid].get("is_storage", False))]
    hydro_ids = [gid for gid in g_ids if bool(gdata[gid].get("is_hydro", False))]

    # By zone
    gens_by_zone = {z: [] for z in range(Z)}
    storage_by_zone = {z: [] for z in range(Z)}
    for gid in g_ids:
        z = int(gdata[gid]["zone"])
        if z not in gens_by_zone:
            gens_by_zone[z] = []
            storage_by_zone[z] = []
        gens_by_zone[z].append(gid)
        if gid in storage_ids:
            storage_by_zone[z].append(gid)

    line_ids = [int(l["id"]) for l in lines]
    ldata = {int(l["id"]): l for l in lines}
    lines_out = {z: [] for z in range(Z)}
    lines_in = {z: [] for z in range(Z)}
    for lid in line_ids:
        fz = int(ldata[lid]["from_zone"])
        tz = int(ldata[lid]["to_zone"])
        if fz not in lines_out:
            lines_out[fz] = []
        if tz not in lines_in:
            lines_in[tz] = []
        lines_out[fz].append(lid)
        lines_in[tz].append(lid)

    demand = {z: get_zone_demand(inst["demand"], z, T) for z in range(Z)}

    # Availability and hydro inflow arrays
    avail = {}
    inflow = {}
    for gid in g_ids:
        prof = availability_profiles.get(str(gid), availability_profiles.get(gid, None))
        avail[gid] = get_timeseries(prof, T, default=1.0)

        g = gdata[gid]
        if gid in hydro_ids:
            src = g.get("inflow_profile", None)
            if src is None:
                src = hydro_inflow_profiles.get(str(gid), hydro_inflow_profiles.get(gid, None))
            inflow[gid] = get_timeseries(src, T, default=0.0)
        else:
            inflow[gid] = [0.0] * T

    m = gp.Model("capacity_expansion_uc")
    m.Params.OutputFlag = 0
    m.Params.TimeLimit = max(1, int(args.time_limit))
    m.Params.Threads = 1
    m.Params.MIPGap = 0.001

    # Investment variables
    yP_new = m.addVars(g_ids, vtype=GRB.INTEGER, lb=0.0, name="yP_new")
    yP_ret = m.addVars(g_ids, vtype=GRB.INTEGER, lb=0.0, name="yP_ret")
    Pcap = m.addVars(g_ids, vtype=GRB.CONTINUOUS, lb=0.0, name="Pcap")

    yE_new = m.addVars(storage_ids, vtype=GRB.INTEGER, lb=0.0, name="yE_new")
    yE_ret = m.addVars(storage_ids, vtype=GRB.INTEGER, lb=0.0, name="yE_ret")
    Ecap = m.addVars(storage_ids, vtype=GRB.CONTINUOUS, lb=0.0, name="Ecap")

    yF_new = m.addVars(line_ids, vtype=GRB.INTEGER, lb=0.0, name="yF_new")
    Fcap = m.addVars(line_ids, vtype=GRB.CONTINUOUS, lb=0.0, name="Fcap")

    # Operations
    gen = m.addVars(g_ids, range(T), vtype=GRB.CONTINUOUS, lb=0.0, name="gen")
    wd = m.addVars(storage_ids, range(T), vtype=GRB.CONTINUOUS, lb=0.0, name="withdraw")
    flow = m.addVars(line_ids, range(T), vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="flow")
    nse = m.addVars(range(len(nse_segments)), range(Z), range(T), vtype=GRB.CONTINUOUS, lb=0.0, name="nse")

    # UC vars
    u = m.addVars(uc_ids, range(T), vtype=GRB.INTEGER, lb=0.0, name="u")
    su = m.addVars(uc_ids, range(T), vtype=GRB.INTEGER, lb=0.0, name="su")
    sd = m.addVars(uc_ids, range(T), vtype=GRB.INTEGER, lb=0.0, name="sd")

    # Storage/Hydro state vars
    soc = m.addVars(storage_ids, range(T), vtype=GRB.CONTINUOUS, lb=0.0, name="soc")
    res = m.addVars(hydro_ids, range(T), vtype=GRB.CONTINUOUS, lb=0.0, name="res")
    spill = m.addVars(hydro_ids, range(T), vtype=GRB.CONTINUOUS, lb=0.0, name="spill")

    # Policy slacks per subperiod
    rps_slack = m.addVars(range(S), vtype=GRB.CONTINUOUS, lb=0.0, name="rps_slack")
    co2_slack = m.addVars(range(S), vtype=GRB.CONTINUOUS, lb=0.0, name="co2_slack")

    # Capacity accounting
    for gid in g_ids:
        g = gdata[gid]
        size = float(g.get("capacity_size_MW", 1.0))
        exist = float(g.get("existing_capacity_MW", 0.0))
        max_total = float(g.get("max_capacity_MW", max(exist, exist + 1e6)))

        m.addConstr(Pcap[gid] == exist + size * (yP_new[gid] - yP_ret[gid]), name=f"Pcap_bal_{gid}")
        m.addConstr(Pcap[gid] <= max_total, name=f"Pcap_max_{gid}")
        m.addConstr(size * yP_ret[gid] <= exist + 1e-9, name=f"Pret_max_{gid}")

        max_new_est = max(0.0, max_total - min(exist, max_total))
        m.addConstr(size * yP_new[gid] <= max_new_est + 1e-9, name=f"Pnew_max_{gid}")

        if not bool(g.get("can_retire", True)):
            m.addConstr(yP_ret[gid] == 0, name=f"no_ret_{gid}")

    for gid in storage_ids:
        g = gdata[gid]
        esize = float(g.get("storage_capacity_size_MWh", 1.0))
        eexist = float(g.get("existing_storage_capacity_MWh", 0.0))
        emax = float(g.get("max_storage_capacity_MWh", max(eexist, eexist + 1e6)))

        m.addConstr(Ecap[gid] == eexist + esize * (yE_new[gid] - yE_ret[gid]), name=f"Ecap_bal_{gid}")
        m.addConstr(Ecap[gid] <= emax, name=f"Ecap_max_{gid}")
        m.addConstr(esize * yE_ret[gid] <= eexist + 1e-9, name=f"Eret_max_{gid}")
        m.addConstr(esize * yE_new[gid] <= max(0.0, emax - min(eexist, emax)) + 1e-9, name=f"Enew_max_{gid}")

        if not bool(g.get("can_retire", True)):
            m.addConstr(yE_ret[gid] == 0, name=f"no_eret_{gid}")

        min_dur = float(g.get("min_duration_MWh_per_MW", 0.0))
        max_dur = float(g.get("max_duration_MWh_per_MW", 1e6))
        m.addConstr(Ecap[gid] >= min_dur * Pcap[gid], name=f"dur_min_{gid}")
        m.addConstr(Ecap[gid] <= max_dur * Pcap[gid], name=f"dur_max_{gid}")

    for lid in line_ids:
        l = ldata[lid]
        exist = float(l.get("existing_capacity_MW", 0.0))
        max_new = float(l.get("max_new_capacity_MW", 0.0))
        m.addConstr(Fcap[lid] == exist + yF_new[lid], name=f"Fcap_bal_{lid}")
        m.addConstr(yF_new[lid] <= max_new, name=f"Fnew_max_{lid}")

    # Operational constraints
    # Generation bounds + storage/hydro specifics
    for gid in g_ids:
        g = gdata[gid]
        min_out = float(g.get("min_output_frac", 0.0))
        is_uc = gid in uc_ids
        is_st = gid in storage_ids
        is_hy = gid in hydro_ids

        for t in range(T):
            a = float(avail[gid][t])

            if is_uc:
                size = float(g.get("capacity_size_MW", 1.0))
                eff_min = min(min_out, a)
                m.addConstr(gen[gid, t] >= u[gid, t] * eff_min * size, name=f"uc_gen_lb_{gid}_{t}")
                m.addConstr(gen[gid, t] <= u[gid, t] * a * size, name=f"uc_gen_ub_{gid}_{t}")
            else:
                m.addConstr(gen[gid, t] <= a * Pcap[gid], name=f"gen_ub_{gid}_{t}")
                if (not is_st) and (not is_hy):
                    eff_min = min(min_out, a)
                    m.addConstr(gen[gid, t] >= eff_min * Pcap[gid], name=f"gen_lb_{gid}_{t}")

            if is_st:
                ce = float(g.get("charge_efficiency", 1.0))
                de = max(1e-6, float(g.get("discharge_efficiency", 1.0)))

                m.addConstr(wd[gid, t] <= a * Pcap[gid], name=f"wd_ub1_{gid}_{t}")
                m.addConstr(gen[gid, t] + wd[gid, t] <= Pcap[gid], name=f"wd_ub2_{gid}_{t}")
                m.addConstr(ce * wd[gid, t] <= Ecap[gid] - soc[gid, t], name=f"soc_room_{gid}_{t}")
                m.addConstr(gen[gid, t] / de <= soc[gid, t], name=f"soc_energy_{gid}_{t}")
                m.addConstr(soc[gid, t] <= Ecap[gid], name=f"soc_cap_{gid}_{t}")

            if is_hy:
                dur = float(g.get("duration_MWh_per_MW", 0.0))
                m.addConstr(res[gid, t] <= dur * Pcap[gid], name=f"res_cap_{gid}_{t}")
                m.addConstr(gen[gid, t] + spill[gid, t] >= min_out * Pcap[gid], name=f"hydro_min_{gid}_{t}")

    # Flow limits
    for lid in line_ids:
        for t in range(T):
            m.addConstr(flow[lid, t] <= Fcap[lid], name=f"flow_ub_{lid}_{t}")
            m.addConstr(-flow[lid, t] <= Fcap[lid], name=f"flow_lb_{lid}_{t}")

    # NSE bounds
    for k, seg in enumerate(nse_segments):
        maxf_raw = seg.get("max_frac", 0.0)
        for z in range(Z):
            maxf = zonal_param(maxf_raw, z, 0.0)
            for t in range(T):
                m.addConstr(nse[k, z, t] <= maxf * demand[z][t], name=f"nse_ub_{k}_{z}_{t}")

    # Power balance
    for z in range(Z):
        gz = gens_by_zone.get(z, [])
        sz = storage_by_zone.get(z, [])
        lo = lines_out.get(z, [])
        li = lines_in.get(z, [])
        for t in range(T):
            m.addConstr(
                gp.quicksum(gen[g, t] for g in gz)
                - gp.quicksum(wd[g, t] for g in sz)
                - gp.quicksum(flow[l, t] for l in lo)
                + gp.quicksum(flow[l, t] for l in li)
                + gp.quicksum(nse[k, z, t] for k in range(len(nse_segments)))
                == demand[z][t],
                name=f"balance_{z}_{t}",
            )

    # Circular indexing helper
    def prev_t(ti: int) -> int:
        s = ti // H
        h = ti % H
        return s * H + ((h - 1) % H)

    def window_indices(ti: int, L: int) -> List[int]:
        s = ti // H
        h = ti % H
        return [s * H + ((h - k) % H) for k in range(max(0, L))]

    # Storage dynamics
    for gid in storage_ids:
        g = gdata[gid]
        ce = float(g.get("charge_efficiency", 1.0))
        de = max(1e-6, float(g.get("discharge_efficiency", 1.0)))
        loss = float(g.get("self_discharge_rate", 0.0))
        for t in range(T):
            p = prev_t(t)
            m.addConstr(
                soc[gid, t] - soc[gid, p]
                == ce * wd[gid, t] - gen[gid, t] / de - loss * soc[gid, p],
                name=f"soc_dyn_{gid}_{t}",
            )

    # Hydro dynamics
    for gid in hydro_ids:
        for t in range(T):
            p = prev_t(t)
            m.addConstr(
                res[gid, t] - res[gid, p] == inflow[gid][t] * Pcap[gid] - gen[gid, t] - spill[gid, t],
                name=f"res_dyn_{gid}_{t}",
            )

    # Non-UC ramping (excluding storage)
    for gid in g_ids:
        if gid in uc_ids or gid in storage_ids:
            continue
        g = gdata[gid]
        ru = float(g.get("ramp_up_frac_per_hr", 1e6))
        rd = float(g.get("ramp_dn_frac_per_hr", 1e6))
        for t in range(T):
            p = prev_t(t)
            m.addConstr(gen[gid, t] - gen[gid, p] <= ru * Pcap[gid], name=f"ramp_up_{gid}_{t}")
            m.addConstr(gen[gid, p] - gen[gid, t] <= rd * Pcap[gid], name=f"ramp_dn_{gid}_{t}")

    # UC constraints
    for gid in uc_ids:
        g = gdata[gid]
        size = float(g.get("capacity_size_MW", 1.0))
        min_out = float(g.get("min_output_frac", 0.0))
        ru = float(g.get("ramp_up_frac_per_hr", 1.0))
        rd = float(g.get("ramp_dn_frac_per_hr", 1.0))
        mut = int(g.get("min_up_time_hr", 0))
        mdt = int(g.get("min_down_time_hr", 0))

        for t in range(T):
            p = prev_t(t)

            m.addConstr(u[gid, t] * size <= Pcap[gid], name=f"u_cap_{gid}_{t}")
            m.addConstr(su[gid, t] * size <= Pcap[gid], name=f"su_cap_{gid}_{t}")
            m.addConstr(sd[gid, t] * size <= Pcap[gid], name=f"sd_cap_{gid}_{t}")

            m.addConstr(u[gid, t] - u[gid, p] == su[gid, t] - sd[gid, t], name=f"uc_trans_{gid}_{t}")

            a = float(avail[gid][t])
            start_coef = min(a, max(min_out, ru))
            stop_coef = min(a, max(min_out, rd))

            m.addConstr(
                gen[gid, t] - gen[gid, p]
                <= size * ru * (u[gid, t] - su[gid, t])
                + size * start_coef * su[gid, t]
                - size * min_out * sd[gid, t],
                name=f"uc_rup_{gid}_{t}",
            )
            m.addConstr(
                gen[gid, p] - gen[gid, t]
                <= size * rd * (u[gid, t] - su[gid, t])
                + size * stop_coef * sd[gid, t]
                - size * min_out * su[gid, t],
                name=f"uc_rdn_{gid}_{t}",
            )

            if mut > 0:
                idxs = window_indices(t, min(mut, H))
                m.addConstr(u[gid, t] >= gp.quicksum(su[gid, tt] for tt in idxs), name=f"mut_{gid}_{t}")
            if mdt > 0:
                idxs = window_indices(t, min(mdt, H))
                m.addConstr(
                    u[gid, t] + gp.quicksum(sd[gid, tt] for tt in idxs) <= Pcap[gid] / size,
                    name=f"mdt_{gid}_{t}",
                )

    # Policy constraints
    total_weighted_demand = sum(t_weight[t] * sum(demand[z][t] for z in range(Z)) for t in range(T))

    if bool(policy.get("RPS_enabled", False)):
        rps_share = float(policy.get("RPS_share", 0.0))
        qual_ids = [gid for gid in g_ids if bool(gdata[gid].get("is_RPS_qualifying", False))]
        lhs = gp.quicksum(t_weight[t] * gp.quicksum(gen[gid, t] for gid in qual_ids) for t in range(T)) + gp.quicksum(
            rps_slack[s] for s in range(S)
        )
        m.addConstr(lhs >= rps_share * total_weighted_demand, name="RPS_req")
    else:
        for s in range(S):
            m.addConstr(rps_slack[s] == 0.0, name=f"rps_off_{s}")

    if bool(policy.get("CO2_cap_enabled", False)):
        cap_factor = policy.get("CO2_cap_tons_per_MWh", None)
        if cap_factor is None:
            cap_factor = 0.05
        cap_factor = float(cap_factor)
        emissions = gp.quicksum(
            t_weight[t]
            * gp.quicksum(
                float(gdata[gid].get("co2_tons_per_MWh", 0.0))
                * (gen[gid, t] + (wd[gid, t] if gid in storage_ids else 0.0))
                for gid in g_ids
            )
            for t in range(T)
        )
        m.addConstr(emissions - gp.quicksum(co2_slack[s] for s in range(S)) <= cap_factor * total_weighted_demand, name="CO2_cap")
    else:
        for s in range(S):
            m.addConstr(co2_slack[s] == 0.0, name=f"co2_off_{s}")

    # Objective
    obj = gp.LinExpr()

    # Fixed investment + fixed O&M (power side)
    for gid in g_ids:
        g = gdata[gid]
        size = float(g.get("capacity_size_MW", 1.0))
        inv = float(g.get("inv_cost_per_MW_yr", 0.0))
        fom = float(g.get("fom_cost_per_MW_yr", 0.0))
        obj += inv * size * yP_new[gid]
        obj += fom * Pcap[gid]

        if gid in hydro_ids:
            dur = float(g.get("duration_MWh_per_MW", 0.0))
            inv_e = float(g.get("inv_cost_energy_per_MWh_yr", 0.0))
            fom_e = float(g.get("fom_cost_energy_per_MWh_yr", 0.0))
            obj += inv_e * dur * size * yP_new[gid]
            obj += fom_e * dur * Pcap[gid]

    # Storage energy side fixed costs
    for gid in storage_ids:
        g = gdata[gid]
        esize = float(g.get("storage_capacity_size_MWh", 1.0))
        inv_e = float(g.get("inv_cost_energy_per_MWh_yr", 0.0))
        fom_e = float(g.get("fom_cost_energy_per_MWh_yr", 0.0))
        obj += inv_e * esize * yE_new[gid]
        obj += fom_e * Ecap[gid]

    # Transmission investment
    for lid in line_ids:
        inv = float(ldata[lid].get("inv_cost_per_MW_yr", 0.0))
        obj += inv * yF_new[lid]

    # Variable generation/withdrawal costs
    for gid in g_ids:
        vc = float(gdata[gid].get("var_cost_per_MWh", 0.0))
        if gid in storage_ids:
            obj += gp.quicksum(t_weight[t] * vc * (gen[gid, t] + wd[gid, t]) for t in range(T))
        else:
            obj += gp.quicksum(t_weight[t] * vc * gen[gid, t] for t in range(T))

    # NSE costs
    for k, seg in enumerate(nse_segments):
        c_raw = seg.get("cost_per_MWh", 0.0)
        for z in range(Z):
            c = zonal_param(c_raw, z, 0.0)
            obj += gp.quicksum(t_weight[t] * c * nse[k, z, t] for t in range(T))

    # Startup costs
    for gid in uc_ids:
        sc = float(gdata[gid].get("start_cost_per_unit", 0.0))
        obj += gp.quicksum(t_weight[t] * sc * su[gid, t] for t in range(T))

    # Policy noncompliance penalties
    rps_pen = float(policy.get("RPS_noncompliance_cost_per_MWh", 0.0))
    co2_pen = float(policy.get("CO2_noncompliance_cost_per_ton", 0.0))
    obj += rps_pen * gp.quicksum(rps_slack[s] for s in range(S))
    obj += co2_pen * gp.quicksum(co2_slack[s] for s in range(S))

    m.setObjective(obj, GRB.MINIMIZE)

    # Incumbent logging callback
    def cb(model, where):
        if where == GRB.Callback.MIPSOL and logger is not None:
            val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if (not hasattr(model, "_best_logged")) or (val < model._best_logged - 1e-9):
                model._best_logged = val
                logger.log(float(val))

    m._best_logged = float("inf")
    m.optimize(cb)

    has_sol = m.SolCount > 0
    if has_sol and logger is not None:
        final_obj = float(m.ObjVal)
        if final_obj < m._best_logged - 1e-9:
            logger.log(final_obj)

    objective_value = float(m.ObjVal) if has_sol else 1e30

    investment_decisions = []
    for gid in sorted(g_ids):
        g = gdata[gid]
        rec = {
            "generator_id": int(gid),
            "resource_type": str(g.get("resource_type", "")),
            "zone": int(g.get("zone", 0)),
            "y_P_new": float(yP_new[gid].X) if has_sol else 0.0,
            "y_P_ret": float(yP_ret[gid].X) if has_sol else 0.0,
            "y_P_total_MW": float(Pcap[gid].X) if has_sol else float(g.get("existing_capacity_MW", 0.0)),
        }
        if gid in storage_ids:
            rec["y_E_new"] = float(yE_new[gid].X) if has_sol else 0.0
            rec["y_E_ret"] = float(yE_ret[gid].X) if has_sol else 0.0
            rec["y_E_total_MWh"] = float(Ecap[gid].X) if has_sol else float(g.get("existing_storage_capacity_MWh", 0.0))
        investment_decisions.append(rec)

    transmission_decisions = []
    for lid in sorted(line_ids):
        l = ldata[lid]
        transmission_decisions.append(
            {
                "line_id": int(lid),
                "y_F_new_MW": float(yF_new[lid].X) if has_sol else 0.0,
                "total_capacity_MW": float(Fcap[lid].X) if has_sol else float(l.get("existing_capacity_MW", 0.0)),
            }
        )

    sol = {
        "objective_value": objective_value,
        "investment_decisions": investment_decisions,
        "transmission_decisions": transmission_decisions,
    }

    with open(args.solution_path, "w") as f:
        json.dump(sol, f, indent=2)


if __name__ == "__main__":
    main()
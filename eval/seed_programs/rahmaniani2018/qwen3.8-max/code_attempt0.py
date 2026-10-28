#!/usr/bin/env python3
import argparse
import json
import math
import time
import sys

import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except Exception:
    SolutionLogger = None


def to_float(x, default=0.0):
    if x is None:
        return default
    try:
        return float(x)
    except Exception:
        return default


def to_int(x, default=0):
    if x is None:
        return default
    try:
        return int(x)
    except Exception:
        return default


def floor_nonneg(x):
    try:
        x = float(x)
    except Exception:
        return 0
    if x <= 0.0:
        return 0
    return int(math.floor(x + 1e-6))


def clean_number(v, integer=False):
    if v is None:
        return 0.0
    try:
        v = float(v)
    except Exception:
        return 0.0
    if abs(v) < 1e-7:
        return 0.0
    if integer or abs(v - round(v)) < 1e-4:
        v = float(round(v))
    else:
        v = float(round(v, 6))
    if abs(v) < 1e-7:
        return 0.0
    return v


def extend_series(vals, T, clamp01=False):
    if vals is None:
        return []
    vals = list(vals)
    if len(vals) == 0:
        return []
    if len(vals) >= T:
        vals = vals[:T]
    else:
        reps = (T + len(vals) - 1) // len(vals)
        vals = (vals * reps)[:T]
    out = []
    for v in vals:
        vf = to_float(v, 0.0)
        if clamp01:
            vf = max(0.0, min(1.0, vf))
        else:
            vf = max(0.0, vf)
        out.append(vf)
    return out


def get_availability(profiles, gid, T):
    prof = profiles.get(str(gid), profiles.get(gid))
    if prof is None:
        return True, 1.0, None

    if prof.get("type", "") == "timeseries" or "values" in prof:
        vals = extend_series(prof.get("values", []), T, clamp01=True)
        if vals:
            return False, None, vals

    val = to_float(prof.get("value", 1.0), 1.0)
    val = max(0.0, min(1.0, val))
    return True, val, None


def get_hydro_inflow(gen, profiles, gid, T):
    # Explicit inflow fields, if present.
    for key, val in gen.items():
        if "inflow" in key.lower():
            if isinstance(val, dict):
                if "values" in val:
                    vals = extend_series(val.get("values", []), T, clamp01=False)
                    if vals:
                        return False, None, vals
                return True, to_float(val.get("value", 0.0), 0.0), None
            if isinstance(val, list):
                vals = extend_series(val, T, clamp01=False)
                if vals:
                    return False, None, vals
            return True, to_float(val, 0.0), None

    # Fallback: if this hydro has an explicit availability profile, use it as inflow.
    prof = profiles.get(str(gid), profiles.get(gid))
    if prof is not None:
        if prof.get("type", "") == "timeseries" or "values" in prof:
            vals = extend_series(prof.get("values", []), T, clamp01=False)
            if vals:
                return False, None, vals
        else:
            return True, to_float(prof.get("value", 0.0), 0.0), None

    return True, 0.0, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    start_time = time.time()

    logger = None
    if args.log_path and SolutionLogger is not None:
        try:
            logger = SolutionLogger(args.log_path, sense="minimize")
        except Exception:
            logger = None

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    dims = inst.get("problem_dimensions", {})
    policy = inst.get("policy", {})
    subperiods = inst.get("subperiods", {})
    generators = inst.get("generators", [])
    transmission_lines = inst.get("transmission_lines", [])
    demand_raw = inst.get("demand", {})
    nse_segments_raw = inst.get("nse_segments", [])
    availability_profiles = inst.get("availability_profiles", {})

    # Time indexing
    W = to_int(subperiods.get("num_subperiods", dims.get("num_weeks", 1)), 1)
    H = to_int(subperiods.get("hours_per_subperiod", dims.get("hours_per_week", 168)), 168)
    W = max(1, W)
    H = max(1, H)
    T = W * H

    weights = subperiods.get("subperiod_weights", [])
    weights = [to_float(w, 1.0) for w in weights]
    while len(weights) < W:
        weights.append(1.0)
    weights = weights[:W]

    hour_weight_t = []
    for w in range(W):
        hw = weights[w] / float(H)
        hour_weight_t.extend([hw] * H)

    prev_idx = [0] * T
    for t in range(T):
        if t % H == 0:
            prev_idx[t] = t + H - 1
        else:
            prev_idx[t] = t - 1

    # Zones and demand
    zone_ids_set = set()
    for k in demand_raw.keys():
        zone_ids_set.add(to_int(k))
    for g in generators:
        zone_ids_set.add(to_int(g.get("zone", 0), 0))
    for l in transmission_lines:
        zone_ids_set.add(to_int(l.get("from_zone", 0), 0))
        zone_ids_set.add(to_int(l.get("to_zone", 0), 0))
    nz = to_int(dims.get("num_zones", 0), 0)
    if nz > 0:
        for z in range(nz):
            zone_ids_set.add(z)
    if not zone_ids_set:
        zone_ids_set.add(0)
    zone_ids = sorted(zone_ids_set)

    demand_by_zone = {}
    total_weighted_demand = 0.0
    for z in zone_ids:
        raw = demand_raw.get(str(z), demand_raw.get(z, []))
        arr = []
        for t in range(T):
            if t < len(raw):
                v = to_float(raw[t], 0.0)
            else:
                v = 0.0
            arr.append(v)
            total_weighted_demand += hour_weight_t[t] * v
        demand_by_zone[z] = arr

    # NSE segments
    nse_segments = []
    for seg in nse_segments_raw:
        s_id = to_int(seg.get("segment", len(nse_segments)), len(nse_segments))
        cost = to_float(seg.get("cost_per_MWh", 0.0), 0.0)
        max_frac = to_float(seg.get("max_frac", 0.0), 0.0)
        if max_frac > 0:
            nse_segments.append({"segment": s_id, "cost_per_MWh": cost, "max_frac": max_frac})

    # Generator preprocessing
    gen_infos = []
    for idx, g in enumerate(generators):
        gid = to_int(g.get("id", idx), idx)
        zone = to_int(g.get("zone", 0), 0)
        category = str(g.get("category", "")).upper()

        is_UC = bool(g.get("is_UC", False))
        is_storage = bool(g.get("is_storage", False))
        is_hydro = bool(g.get("is_hydro", False))
        if not is_storage and category == "STORAGE":
            is_storage = True
        if not is_hydro and category == "HYDRO":
            is_hydro = True

        is_rps_default = category in ("VRE", "HYDRO")
        is_RPS = bool(g.get("is_RPS_qualifying", is_rps_default))
        can_retire = bool(g.get("can_retire", True))

        C = to_float(g.get("capacity_size_MW", 1.0), 1.0)
        if C <= 0:
            C = 1.0

        existing = max(0.0, to_float(g.get("existing_capacity_MW", 0.0), 0.0))
        max_cap = max(0.0, to_float(g.get("max_capacity_MW", 0.0), 0.0))

        inv_cost = to_float(g.get("inv_cost_per_MW_yr", 0.0), 0.0)
        fom_cost = to_float(g.get("fom_cost_per_MW_yr", 0.0), 0.0)
        var_cost = to_float(g.get("var_cost_per_MWh", 0.0), 0.0)
        co2 = to_float(g.get("co2_tons_per_MWh", 0.0), 0.0)
        min_out = to_float(g.get("min_output_frac", 0.0), 0.0)

        ramp_up = g.get("ramp_up_frac_per_hr", None)
        ramp_dn = g.get("ramp_dn_frac_per_hr", None)
        ramp_up = to_float(ramp_up, None) if ramp_up is not None else None
        ramp_dn = to_float(ramp_dn, None) if ramp_dn is not None else None

        min_up = to_int(g.get("min_up_time_hr", 1 if is_UC else 0), 1 if is_UC else 0)
        min_dn = to_int(g.get("min_down_time_hr", 1 if is_UC else 0), 1 if is_UC else 0)
        start_cost = to_float(g.get("start_cost_per_unit", 0.0), 0.0)

        # Storage fields
        Esize = to_float(g.get("storage_capacity_size_MWh", 1.0 if is_storage else 0.0), 0.0)
        if is_storage and Esize <= 0:
            Esize = 1.0
        existing_E = max(0.0, to_float(g.get("existing_storage_capacity_MWh", 0.0), 0.0))
        max_E = max(0.0, to_float(g.get("max_storage_capacity_MWh", 0.0), 0.0))
        eff_c = to_float(g.get("charge_efficiency", 1.0), 1.0)
        eff_d = to_float(g.get("discharge_efficiency", 1.0), 1.0)
        selfd = to_float(g.get("self_discharge_rate", 0.0), 0.0)
        selfd = max(0.0, min(1.0, selfd))
        min_dur = to_float(g.get("min_duration_MWh_per_MW", 0.0), 0.0)
        max_dur_raw = g.get("max_duration_MWh_per_MW", None)
        max_dur = to_float(max_dur_raw, None) if max_dur_raw is not None else None
        inv_cost_E = to_float(g.get("inv_cost_energy_per_MWh_yr", 0.0), 0.0)
        fom_cost_E = to_float(g.get("fom_cost_energy_per_MWh_yr", 0.0), 0.0)

        # Hydro fields
        duration = to_float(g.get("duration_MWh_per_MW", 0.0), 0.0)

        # Availability
        avail_is_const, avail_const, avail_series = get_availability(availability_profiles, gid, T)

        # Hydro inflow
        if is_hydro:
            inflow_is_const, inflow_const, inflow_series = get_hydro_inflow(g, availability_profiles, gid, T)
        else:
            inflow_is_const, inflow_const, inflow_series = True, 0.0, None

        # Investment bounds and objective coefficients.
        # The text describes max_capacity as maximum new capacity, while the schema calls it total capacity.
        # We use a safe hybrid: if max_cap >= existing, also enforce total <= max_cap; otherwise avoid forcing infeasibility.
        new_ub = floor_nonneg(max_cap / C)
        ret_ub = floor_nonneg(existing / C) if can_retire else 0
        enforce_total_cap = (max_cap + 1e-6 >= existing)
        yP_ub = max_cap if enforce_total_cap else (existing + max_cap)

        hydro_energy_inv = inv_cost_E if is_hydro else 0.0
        hydro_energy_fom = fom_cost_E if is_hydro else 0.0
        power_inv_coeff = C * (inv_cost + hydro_energy_inv * duration)
        power_fom_coeff = fom_cost + hydro_energy_fom * duration

        # Storage energy bounds
        if is_storage:
            e_new_ub = floor_nonneg(max_E / Esize)
            e_ret_ub = floor_nonneg(existing_E / Esize) if can_retire else 0
            enforce_total_E = (max_E + 1e-6 >= existing_E)
            yE_ub = max_E if enforce_total_E else (existing_E + max_E)
            energy_inv_coeff = inv_cost_E * Esize
            energy_fom_coeff = fom_cost_E
        else:
            e_new_ub = 0
            e_ret_ub = 0
            enforce_total_E = False
            yE_ub = 0.0
            energy_inv_coeff = 0.0
            energy_fom_coeff = 0.0

        active = (
            existing > 1e-9
            or max_cap > 1e-9
            or (is_storage and (existing_E > 1e-9 or max_E > 1e-9))
        )

        gen_infos.append({
            "idx": idx,
            "id": gid,
            "zone": zone,
            "resource_type": str(g.get("resource_type", "")),
            "category": category,
            "is_UC": is_UC,
            "is_storage": is_storage,
            "is_hydro": is_hydro,
            "is_RPS": is_RPS,
            "can_retire": can_retire,
            "C": C,
            "existing": existing,
            "max_cap": max_cap,
            "new_ub": new_ub,
            "ret_ub": ret_ub,
            "enforce_total_cap": enforce_total_cap,
            "yP_ub": yP_ub,
            "power_inv_coeff": power_inv_coeff,
            "power_fom_coeff": power_fom_coeff,
            "var_cost": var_cost,
            "co2": co2,
            "min_out": min_out,
            "ramp_up": ramp_up,
            "ramp_dn": ramp_dn,
            "min_up": min_up,
            "min_dn": min_dn,
            "start_cost": start_cost,
            "Esize": Esize,
            "existing_E": existing_E,
            "max_E": max_E,
            "e_new_ub": e_new_ub,
            "e_ret_ub": e_ret_ub,
            "enforce_total_E": enforce_total_E,
            "yE_ub": yE_ub,
            "energy_inv_coeff": energy_inv_coeff,
            "energy_fom_coeff": energy_fom_coeff,
            "eff_c": eff_c,
            "eff_d": eff_d,
            "selfd": selfd,
            "min_dur": min_dur,
            "max_dur": max_dur,
            "duration": duration,
            "active": active,
            "avail_is_const": avail_is_const,
            "avail_const": avail_const,
            "avail_series": avail_series,
            "inflow_is_const": inflow_is_const,
            "inflow_const": inflow_const,
            "inflow_series": inflow_series,
        })

    # Transmission preprocessing
    line_infos = []
    for idx, l in enumerate(transmission_lines):
        lid = to_int(l.get("id", idx), idx)
        existing = max(0.0, to_float(l.get("existing_capacity_MW", 0.0), 0.0))
        max_new = max(0.0, to_float(l.get("max_new_capacity_MW", 0.0), 0.0))
        inv_cost = to_float(l.get("inv_cost_per_MW_yr", 0.0), 0.0)
        active = existing > 1e-9 or max_new > 1e-9
        line_infos.append({
            "idx": idx,
            "id": lid,
            "from_zone": to_int(l.get("from_zone", 0), 0),
            "to_zone": to_int(l.get("to_zone", 0), 0),
            "existing": existing,
            "max_new": max_new,
            "new_ub": floor_nonneg(max_new),
            "inv_cost": inv_cost,
            "active": active,
        })

    # -------------------------------------------------------------------------
    # Build Gurobi model
    # -------------------------------------------------------------------------
    model = gp.Model("expansion")
    model.ModelSense = GRB.MINIMIZE
    model.Params.OutputFlag = 0

    yPnew_vars = {}
    yPret_vars = {}
    yP_vars = {}
    yEnew_vars = {}
    yEret_vars = {}
    yE_vars = {}

    # Investment variables
    for info in gen_infos:
        if not info["active"]:
            continue
        idx = info["idx"]
        C = info["C"]

        yPnew = model.addVar(vtype=GRB.INTEGER, lb=0.0, ub=info["new_ub"], obj=info["power_inv_coeff"])
        yPret = model.addVar(vtype=GRB.INTEGER, lb=0.0, ub=info["ret_ub"], obj=0.0)
        yP = model.addVar(lb=0.0, ub=info["yP_ub"], obj=info["power_fom_coeff"])

        model.addConstr(yP == info["existing"] + C * (yPnew - yPret))
        if info["enforce_total_cap"]:
            model.addConstr(yP <= info["max_cap"])
        if not info["can_retire"]:
            model.addConstr(yPret == 0)

        yPnew_vars[idx] = yPnew
        yPret_vars[idx] = yPret
        yP_vars[idx] = yP

        if info["is_storage"]:
            Esize = info["Esize"]
            yEnew = model.addVar(vtype=GRB.INTEGER, lb=0.0, ub=info["e_new_ub"], obj=info["energy_inv_coeff"])
            yEret = model.addVar(vtype=GRB.INTEGER, lb=0.0, ub=info["e_ret_ub"], obj=0.0)
            yE = model.addVar(lb=0.0, ub=info["yE_ub"], obj=info["energy_fom_coeff"])

            model.addConstr(yE == info["existing_E"] + Esize * (yEnew - yEret))
            if info["enforce_total_E"]:
                model.addConstr(yE <= info["max_E"])
            if not info["can_retire"]:
                model.addConstr(yEret == 0)

            if info["min_dur"] > 0:
                model.addConstr(yE >= info["min_dur"] * yP)
            if info["max_dur"] is not None:
                model.addConstr(yE <= info["max_dur"] * yP)

            yEnew_vars[idx] = yEnew
            yEret_vars[idx] = yEret
            yE_vars[idx] = yE

    # Transmission investment and flow variables
    yFnew_vars = {}
    xflow_vars = {}
    for line in line_infos:
        if not line["active"]:
            continue
        lid = line["id"]
        yFnew = model.addVar(vtype=GRB.INTEGER, lb=0.0, ub=line["new_ub"], obj=line["inv_cost"])
        yFnew_vars[lid] = yFnew

        flow = model.addVars(T, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
        xflow_vars[lid] = flow

        if T > 0:
            model.addConstrs((flow[t] <= line["existing"] + yFnew for t in range(T)))
            model.addConstrs((flow[t] >= -line["existing"] - yFnew for t in range(T)))

    # Operation variables
    xinj_vars = {}
    xwdw_vars = {}
    xsoc_vars = {}
    xhydro_vars = {}
    xspill_vars = {}
    xcommit_vars = {}
    xstart_vars = {}
    xshut_vars = {}

    for info in gen_infos:
        if not info["active"]:
            continue
        idx = info["idx"]
        var_cost = info["var_cost"]
        power_ub = info["yP_ub"]

        obj_xinj = 0.0
        if var_cost != 0.0:
            obj_xinj = [hour_weight_t[t] * var_cost for t in range(T)]

        xinj = model.addVars(T, lb=0.0, ub=power_ub, obj=obj_xinj, vtype=GRB.CONTINUOUS)
        xinj_vars[idx] = xinj

        if info["is_storage"]:
            obj_xwdw = 0.0
            if var_cost != 0.0:
                obj_xwdw = [hour_weight_t[t] * var_cost for t in range(T)]
            xwdw = model.addVars(T, lb=0.0, ub=power_ub, obj=obj_xwdw, vtype=GRB.CONTINUOUS)
            xwdw_vars[idx] = xwdw

            xsoc = model.addVars(T, lb=0.0, ub=info["yE_ub"], vtype=GRB.CONTINUOUS)
            xsoc_vars[idx] = xsoc

        if info["is_hydro"]:
            hydro_energy_ub = max(0.0, info["duration"] * info["yP_ub"])
            xhydro = model.addVars(T, lb=0.0, ub=hydro_energy_ub, obj=0.0, vtype=GRB.CONTINUOUS)
            xspill = model.addVars(T, lb=0.0, ub=GRB.INFINITY, obj=0.0, vtype=GRB.CONTINUOUS)
            xhydro_vars[idx] = xhydro
            xspill_vars[idx] = xspill

        if info["is_UC"]:
            units_ub = info["existing"] / info["C"] + float(info["new_ub"])
            if units_ub <= 1e-9 and info["C"] > 0:
                units_ub = info["yP_ub"] / info["C"]

            xcommit = model.addVars(T, lb=0.0, ub=units_ub, obj=0.0, vtype=GRB.CONTINUOUS)

            start_obj = 0.0
            if info["start_cost"] != 0.0:
                start_obj = [hour_weight_t[t] * info["start_cost"] for t in range(T)]

            xstart = model.addVars(T, lb=0.0, ub=GRB.INFINITY, obj=start_obj, vtype=GRB.CONTINUOUS)
            xshut = model.addVars(T, lb=0.0, ub=GRB.INFINITY, obj=0.0, vtype=GRB.CONTINUOUS)

            xcommit_vars[idx] = xcommit
            xstart_vars[idx] = xstart
            xshut_vars[idx] = xshut

    # NSE variables
    xnse_vars = {}
    for z in zone_ids:
        for seg in nse_segments:
            s_id = seg["segment"]
            cost = seg["cost_per_MWh"]
            max_frac = seg["max_frac"]

            obj_nse = 0.0
            if cost != 0.0:
                obj_nse = [hour_weight_t[t] * cost for t in range(T)]

            xnse = model.addVars(T, lb=0.0, ub=GRB.INFINITY, obj=obj_nse, vtype=GRB.CONTINUOUS)
            xnse_vars[(z, s_id)] = xnse

            for t in range(T):
                xnse_ub = max_frac * demand_by_zone[z][t]
                model.addConstr(xnse[t] <= xnse_ub)

    # -------------------------------------------------------------------------
    # Operational constraints
    # -------------------------------------------------------------------------
    for info in gen_infos:
        if not info["active"]:
            continue
        idx = info["idx"]
        if idx not in xinj_vars:
            continue

        xinj = xinj_vars[idx]
        yP = yP_vars.get(idx)
        avail_series = info["avail_series"] if info["avail_series"] is not None else []

        if info["is_UC"]:
            commit = xcommit_vars.get(idx)
            start = xstart_vars.get(idx)
            shut = xshut_vars.get(idx)

            if commit is not None and yP is not None:
                model.addConstrs((info["C"] * commit[t] <= yP for t in range(T)))

            if commit is not None and start is not None and shut is not None:
                model.addConstrs((start[t] >= commit[t] - commit[prev_idx[t]] for t in range(T)))
                model.addConstrs((shut[t] >= commit[prev_idx[t]] - commit[t] for t in range(T)))

                units_ub = info["existing"] / info["C"] + float(info["new_ub"])
                if units_ub <= 1e-9 and info["C"] > 0:
                    units_ub = info["yP_ub"] / info["C"]

                if info["min_up"] > 1:
                    for t in range(T):
                        base = (t // H) * H
                        end = base + H
                        for k in range(1, info["min_up"]):
                            tt = t + k
                            if tt < end:
                                model.addConstr(commit[tt] >= start[t])

                if info["min_dn"] > 1:
                    for t in range(T):
                        base = (t // H) * H
                        end = base + H
                        for k in range(1, info["min_dn"]):
                            tt = t + k
                            if tt < end:
                                model.addConstr(commit[tt] <= units_ub - shut[t])

            if commit is not None:
                for t in range(T):
                    if info["avail_is_const"]:
                        avail = info["avail_const"]
                    elif avail_series:
                        avail = avail_series[t]
                    else:
                        avail = 1.0

                    model.addConstr(xinj[t] <= avail * info["C"] * commit[t])
                    model.addConstr(xinj[t] >= info["min_out"] * info["C"] * commit[t])

        else:
            if yP is not None:
                if info["avail_is_const"]:
                    avail = info["avail_const"]
                    model.addConstrs((xinj[t] <= avail * yP for t in range(T)))
                elif avail_series:
                    model.addConstrs((xinj[t] <= avail_series[t] * yP for t in range(T)))
                else:
                    model.addConstrs((xinj[t] <= yP for t in range(T)))

        if yP is not None:
            if info["ramp_up"] is not None:
                model.addConstrs((xinj[t] - xinj[prev_idx[t]] <= info["ramp_up"] * yP for t in range(T)))
            if info["ramp_dn"] is not None:
                model.addConstrs((xinj[prev_idx[t]] - xinj[t] <= info["ramp_dn"] * yP for t in range(T)))

        if info["is_storage"]:
            xwdw = xwdw_vars.get(idx)
            xsoc = xsoc_vars.get(idx)

            if xwdw is not None and yP is not None:
                if info["avail_is_const"]:
                    avail = info["avail_const"]
                    model.addConstrs((xwdw[t] <= avail * yP for t in range(T)))
                elif avail_series:
                    model.addConstrs((xwdw[t] <= avail_series[t] * yP for t in range(T)))
                else:
                    model.addConstrs((xwdw[t] <= yP for t in range(T)))

            if info["is_UC"]:
                commit = xcommit_vars.get(idx)
                if commit is not None:
                    model.addConstrs((xinj[t] <= info["C"] * commit[t] for t in range(T)))
                    if xwdw is not None:
                        model.addConstrs((xwdw[t] <= info["C"] * commit[t] for t in range(T)))

            if xsoc is not None and xwdw is not None:
                yE = yE_vars.get(idx)
                if yE is not None:
                    model.addConstrs((xsoc[t] <= yE for t in range(T)))

                eff_c = info["eff_c"] if info["eff_c"] > 1e-9 else 1.0
                eff_d = info["eff_d"] if info["eff_d"] > 1e-9 else 1.0
                selfd = info["selfd"]

                model.addConstrs(
                    xsoc[t] == (1.0 - selfd) * xsoc[prev_idx[t]] + eff_c * xinj[t] - xwdw[t] / eff_d
                    for t in range(T)
                )

        if info["is_hydro"]:
            xhydro = xhydro_vars.get(idx)
            xspill = xspill_vars.get(idx)

            if xhydro is not None and xspill is not None:
                if info["duration"] > 0 and yP is not None:
                    model.addConstrs((xhydro[t] <= info["duration"] * yP for t in range(T)))

                if info["inflow_is_const"]:
                    inflow = [info["inflow_const"]] * T
                elif info["inflow_series"]:
                    inflow = info["inflow_series"]
                else:
                    inflow = [0.0] * T

                model.addConstrs(
                    xhydro[t] == xhydro[prev_idx[t]] + inflow[t] - xinj[t] - xspill[t]
                    for t in range(T)
                )

    # Zone power balance
    for z in zone_ids:
        for t in range(T):
            lhs = gp.LinExpr(0.0)

            for info in gen_infos:
                if not info["active"] or info["zone"] != z:
                    continue
                idx = info["idx"]
                if idx not in xinj_vars:
                    continue

                if info["is_storage"]:
                    if idx in xwdw_vars:
                        lhs += xwdw_vars[idx][t]
                    lhs -= xinj_vars[idx][t]
                else:
                    lhs += xinj_vars[idx][t]

            for line in line_infos:
                if not line["active"]:
                    continue
                lid = line["id"]
                if lid not in xflow_vars:
                    continue

                if line["to_zone"] == z:
                    lhs += xflow_vars[lid][t]
                if line["from_zone"] == z:
                    lhs -= xflow_vars[lid][t]

            for seg in nse_segments:
                var = xnse_vars.get((z, seg["segment"]))
                if var is not None:
                    lhs += var[t]

            model.addConstr(lhs == demand_by_zone[z][t])

    # Optional RPS constraint
    rps_target = None
    for key in ("RPS_target", "rps_target", "RPS", "rps"):
        if key in policy:
            val = policy[key]
            if isinstance(val, dict):
                for k2 in ("target", "value", "fraction"):
                    if k2 in val:
                        val = val[k2]
                        break
            rps_target = to_float(val, None)
            if rps_target is not None:
                break

    if rps_target is not None and rps_target > 1e-9:
        rps_expr = gp.LinExpr(0.0)
        for info in gen_infos:
            if not info["active"] or not info["is_RPS"]:
                continue
            idx = info["idx"]
            if idx not in xinj_vars:
                continue

            if info["is_storage"]:
                if idx in xwdw_vars:
                    for t in range(T):
                        rps_expr += hour_weight_t[t] * xwdw_vars[idx][t]
            else:
                for t in range(T):
                    rps_expr += hour_weight_t[t] * xinj_vars[idx][t]

        model.addConstr(rps_expr >= rps_target * total_weighted_demand)

    # Optional CO2 constraint
    co2_cap = None
    for key in ("CO2_cap", "co2_cap", "CO2_limit", "co2_limit", "emissions_cap", "co2_tons_cap"):
        if key in policy:
            val = policy[key]
            if isinstance(val, dict):
                for k2 in ("value", "cap", "limit", "tons"):
                    if k2 in val:
                        val = val[k2]
                        break
            co2_cap = to_float(val, None)
            if co2_cap is not None:
                break

    if co2_cap is not None and co2_cap >= 0.0:
        co2_expr = gp.LinExpr(0.0)
        for info in gen_infos:
            if not info["active"] or info["co2"] <= 0.0:
                continue
            idx = info["idx"]
            if idx not in xinj_vars:
                continue

            if not info["is_storage"]:
                for t in range(T):
                    co2_expr += hour_weight_t[t] * info["co2"] * xinj_vars[idx][t]

        model.addConstr(co2_expr <= co2_cap)

    if args.time_limit is not None and args.time_limit > 0:
        model.Params.TimeLimit = float(args.time_limit)

    model.optimize()

    # -------------------------------------------------------------------------
    # Extract and write solution
    # -------------------------------------------------------------------------
    status = model.Status
    status_map = {
        GRB.OPTIMAL: "optimal",
        GRB.TIME_LIMIT: "time_limit",
        GRB.INFEASIBLE: "infeasible",
        GRB.INF_OR_UNBD: "infeasible_or_unbounded",
        GRB.UNBOUNDED: "unbounded",
        GRB.INTERRUPTED: "interrupted",
    }
    status_text = status_map.get(status, str(status))

    has_solution = False
    try:
        has_solution = model.SolCount > 0
    except Exception:
        has_solution = False

    def safe_x(var):
        try:
            x = var.X
            if not math.isfinite(x):
                return 0.0
            return clean_number(x)
        except Exception:
            return 0.0

    def safe_series(var_dict):
        if not has_solution or var_dict is None:
            return [0.0 for _ in range(T)]
        try:
            out = []
            for t in range(T):
                x = var_dict[t].X
                if not math.isfinite(x):
                    x = 0.0
                out.append(clean_number(x))
            return out
        except Exception:
            return [0.0 for _ in range(T)]

    obj_val = None
    try:
        if has_solution:
            ov = model.ObjVal
            if math.isfinite(ov):
                obj_val = clean_number(ov)
    except Exception:
        obj_val = None

    generation_investment = []
    for info in gen_infos:
        idx = info["idx"]

        new_units = safe_x(yPnew_vars[idx]) if idx in yPnew_vars else 0.0
        retired_units = safe_x(yPret_vars[idx]) if idx in yPret_vars else 0.0
        capacity_MW = safe_x(yP_vars[idx]) if idx in yP_vars else 0.0

        entry = {
            "id": info["id"],
            "zone": info["zone"],
            "category": info["category"],
            "new_units": new_units,
            "retired_units": retired_units,
            "capacity_MW": capacity_MW,
        }

        if info["is_storage"]:
            entry["new_energy_units"] = safe_x(yEnew_vars[idx]) if idx in yEnew_vars else 0.0
            entry["retired_energy_units"] = safe_x(yEret_vars[idx]) if idx in yEret_vars else 0.0
            entry["energy_capacity_MWh"] = safe_x(yE_vars[idx]) if idx in yE_vars else 0.0

        generation_investment.append(entry)

    transmission_investment = []
    for line in line_infos:
        lid = line["id"]
        new_units = safe_x(yFnew_vars[lid]) if lid in yFnew_vars else 0.0
        transmission_investment.append({
            "id": lid,
            "from_zone": line["from_zone"],
            "to_zone": line["to_zone"],
            "new_units": new_units,
            "capacity_MW": line["existing"] + new_units,
        })

    operation = {
        "injection": {},
        "withdrawal": {},
        "state_of_charge": {},
        "hydro_reservoir": {},
        "hydro_spill": {},
        "commitment": {},
        "start": {},
        "shutdown": {},
        "nse": {},
        "flow": {},
    }

    for info in gen_infos:
        idx = info["idx"]
        gid = str(info["id"])

        if idx in xinj_vars:
            operation["injection"][gid] = safe_series(xinj_vars[idx])
        if idx in xwdw_vars:
            operation["withdrawal"][gid] = safe_series(xwdw_vars[idx])
        if idx in xsoc_vars:
            operation["state_of_charge"][gid] = safe_series(xsoc_vars[idx])
        if idx in xhydro_vars:
            operation["hydro_reservoir"][gid] = safe_series(xhydro_vars[idx])
        if idx in xspill_vars:
            operation["hydro_spill"][gid] = safe_series(xspill_vars[idx])
        if idx in xcommit_vars:
            operation["commitment"][gid] = safe_series(xcommit_vars[idx])
        if idx in xstart_vars:
            operation["start"][gid] = safe_series(xstart_vars[idx])
        if idx in xshut_vars:
            operation["shutdown"][gid] = safe_series(xshut_vars[idx])

    for line in line_infos:
        lid = line["id"]
        if lid in xflow_vars:
            operation["flow"][str(lid)] = safe_series(xflow_vars[lid])

    for (z, s_id), var in xnse_vars.items():
        operation["nse"][f"{z}_{s_id}"] = safe_series(var)

    solution = {
        "status": status_text,
        "objective": obj_val,
        "runtime_seconds": clean_number(time.time() - start_time),
        "generation_investment": generation_investment,
        "transmission_investment": transmission_investment,
        "operation": operation,
    }

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)

    if logger is not None:
        try:
            logger.log(obj_val if obj_val is not None else float("inf"), solution)
        except Exception:
            pass


if __name__ == "__main__":
    main()
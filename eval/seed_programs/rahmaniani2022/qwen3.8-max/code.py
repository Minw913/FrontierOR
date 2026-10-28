import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except Exception:
    class SolutionLogger:
        def __init__(self, path, sense="minimize"):
            self.path = path
            self.sense = sense

        def log(self, objective_value):
            self._write({"objective_value": float(objective_value)})

        def log_solution(self, objective_value, solution_dict):
            self._write({"objective_value": float(objective_value), "solution": solution_dict})

        def _write(self, record):
            if not getattr(self, "path", None):
                return
            try:
                with open(self.path, "a") as f:
                    f.write(json.dumps(record) + "\n")
            except Exception:
                pass


def get_num(d, keys, default=0.0):
    if not isinstance(d, dict):
        return default
    if isinstance(keys, str):
        keys = [keys]
    for k in keys:
        if k in d and d[k] is not None:
            try:
                val = float(d[k])
                if np.isfinite(val):
                    return val
            except Exception:
                pass
    return default


def clean_float(v, tol=1e-6):
    try:
        v = float(v)
    except Exception:
        return 0.0
    if not np.isfinite(v):
        return 0.0
    if abs(v) <= tol:
        return 0.0
    r = round(v)
    if abs(v - r) <= 1e-5 and abs(v) < 1e12:
        return float(r)
    return v


def clean_int(v):
    try:
        v = float(v)
    except Exception:
        return 0.0
    if not np.isfinite(v):
        return 0.0
    if abs(v) <= 1e-6:
        return 0.0
    r = round(v)
    if abs(v - r) <= 1e-4:
        return float(r)
    return v


def _parse_time(x):
    try:
        return int(float(x))
    except Exception:
        return None


def scalarize_expr(expr, depth=0):
    """
    Convert matrix-like Gurobi expressions to scalar expressions when possible.
    Never calls gp.LinExpr(expr) on a non-real expression.
    """
    if depth > 4:
        return expr

    if isinstance(expr, (int, float, np.number)):
        return gp.LinExpr(float(expr))

    if isinstance(expr, gp.LinExpr):
        return expr

    if hasattr(expr, "sum"):
        try:
            return scalarize_expr(expr.sum(), depth + 1)
        except Exception:
            pass

    if hasattr(expr, "item"):
        try:
            return scalarize_expr(expr.item(), depth + 1)
        except Exception:
            pass

    return expr


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--instance_path", "--instance-path", dest="instance_path", type=str, default=None)
    parser.add_argument("--solution_path", "--solution-path", dest="solution_path", type=str, default=None)
    parser.add_argument("--log_path", "--log-path", dest="log_path", type=str, default=None)
    parser.add_argument("--time_limit", "--time-limit", dest="time_limit", type=_parse_time, default=None)
    parser.add_argument("positional", nargs="*", default=[])

    args, _ = parser.parse_known_args()

    pos = list(args.positional or [])
    pos_idx = 0

    instance_path = args.instance_path
    solution_path = args.solution_path
    log_path = args.log_path
    time_limit = args.time_limit

    if not instance_path and pos_idx < len(pos):
        instance_path = pos[pos_idx]
        pos_idx += 1

    if not solution_path and pos_idx < len(pos):
        solution_path = pos[pos_idx]
        pos_idx += 1

    while pos_idx < len(pos):
        cand = pos[pos_idx]

        parsed_num = None
        try:
            parsed_num = float(cand)
        except Exception:
            parsed_num = None

        if parsed_num is not None and time_limit is None:
            time_limit = int(parsed_num)
            pos_idx += 1
            continue

        if log_path is None:
            log_path = cand
            pos_idx += 1
            continue

        pos_idx += 1

    if not instance_path:
        instance_path = "instance.json"
    if not solution_path:
        solution_path = "solution.json"
    if time_limit is None:
        time_limit = 60

    time_limit = float(time_limit)
    if time_limit <= 0:
        time_limit = 1.0

    start_time = time.time()

    logger = None
    if log_path:
        try:
            logger = SolutionLogger(log_path, sense="minimize")
        except Exception:
            try:
                logger = SolutionLogger(log_path)
            except Exception:
                logger = None

    with open(instance_path, "r") as f:
        inst = json.load(f)

    pp = inst.get("problem_parameters", {}) or {}
    if not isinstance(pp, dict):
        pp = {}

    raw_gens = inst.get("generators", []) or []
    if isinstance(raw_gens, dict):
        gens = []
        for k, v in raw_gens.items():
            if isinstance(v, dict):
                vv = dict(v)
                vv.setdefault("cluster_id", k)
                gens.append(vv)
            else:
                gens.append(v)
    elif isinstance(raw_gens, list):
        gens = raw_gens
    else:
        try:
            gens = list(raw_gens)
        except Exception:
            gens = []

    demand = inst.get("demand", {}) or {}
    if isinstance(demand, list):
        demand = {"0": demand}
    elif not isinstance(demand, dict):
        demand = {}

    raw_lines = inst.get("transmission_lines", []) or []
    if isinstance(raw_lines, dict):
        lines = []
        for k, v in raw_lines.items():
            if isinstance(v, dict):
                vv = dict(v)
                vv.setdefault("line_id", k)
                lines.append(vv)
            else:
                lines.append(v)
    elif isinstance(raw_lines, list):
        lines = raw_lines
    else:
        try:
            lines = list(raw_lines)
        except Exception:
            lines = []

    raw_segments = inst.get("consumer_segments", []) or []
    if isinstance(raw_segments, dict):
        segments = []
        for k, v in raw_segments.items():
            if isinstance(v, dict):
                vv = dict(v)
                vv.setdefault("segment_id", k)
                segments.append(vv)
            else:
                segments.append(v)
    elif isinstance(raw_segments, list):
        segments = raw_segments
    else:
        try:
            segments = list(raw_segments)
        except Exception:
            segments = []

    gen_params = inst.get("generator_parameters", {}) or {}
    if not isinstance(gen_params, dict):
        gen_params = {}

    existing_cap = inst.get("existing_capacity", {}) or {}
    if not isinstance(existing_cap, dict):
        existing_cap = {}

    avail_dict = inst.get("availability", {}) or {}
    if not isinstance(avail_dict, dict):
        avail_dict = {}

    hydro_dict = inst.get("hydro_inflow", {}) or {}
    if not isinstance(hydro_dict, dict):
        hydro_dict = {}

    policy = inst.get("policy", {}) or {}
    if not isinstance(policy, dict):
        policy = {}

    weights_raw = inst.get("subperiod_weights", []) or []
    if isinstance(weights_raw, dict):
        weights_raw = list(weights_raw.values())
    elif not isinstance(weights_raw, list):
        weights_raw = [weights_raw]

    default_weeks = max(1, len(weights_raw))

    nw_raw = pp.get("num_weeks", None)
    if nw_raw is None:
        num_weeks = default_weeks
    else:
        try:
            num_weeks = int(float(nw_raw))
        except Exception:
            num_weeks = default_weeks
    if num_weeks <= 0:
        num_weeks = default_weeks

    h_raw = pp.get("hours_per_subperiod", 168)
    try:
        H = int(float(h_raw))
    except Exception:
        H = 168
    if H <= 0:
        H = 168

    total_hours = num_weeks * H
    if total_hours <= 0:
        total_hours = H
        num_weeks = 1

    weights = np.ones(num_weeks, dtype=float)
    for w in range(num_weeks):
        if w < len(weights_raw):
            try:
                weights[w] = float(weights_raw[w])
            except Exception:
                weights[w] = 1.0

    weights = np.nan_to_num(weights, nan=1.0, posinf=1.0, neginf=0.0)
    weights = np.maximum(weights, 0.0)
    if float(np.sum(weights)) <= 0.0:
        weights = np.ones(num_weeks, dtype=float)

    hour_weight = np.repeat(weights / float(H), H)

    zone_set = set()

    for g in gens:
        if isinstance(g, dict):
            try:
                zone_set.add(int(g.get("zone", 0)))
            except Exception:
                pass

    for k in demand.keys():
        try:
            zone_set.add(int(k))
        except Exception:
            pass

    for l in lines:
        if isinstance(l, dict):
            try:
                zone_set.add(int(l.get("from_zone", 0)))
                zone_set.add(int(l.get("to_zone", 0)))
            except Exception:
                pass

    for s in segments:
        if isinstance(s, dict):
            try:
                zone_set.add(int(s.get("zone", 0)))
            except Exception:
                pass

    if not zone_set:
        zone_set.add(0)

    zone_orig = sorted(zone_set)
    zone_map = {z: i for i, z in enumerate(zone_orig)}
    Z = len(zone_orig)

    demand_mat = np.zeros((Z, total_hours), dtype=float)
    for k, series in demand.items():
        try:
            z = zone_map.get(int(k))
        except Exception:
            z = None
        if z is None:
            continue

        try:
            arr = np.atleast_1d(np.asarray(series, dtype=float))
        except Exception:
            arr = np.zeros(1, dtype=float)

        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if arr.size == 0:
            continue

        if arr.size == 1:
            demand_mat[z, :] = arr[0]
        else:
            n = min(total_hours, arr.size)
            demand_mat[z, :n] = arr[:n]

    demand_mat = np.nan_to_num(demand_mat, nan=0.0, posinf=0.0, neginf=0.0)
    demand_mat = np.maximum(demand_mat, 0.0)

    N = len(gens)

    cluster_ids = []
    gen_zone_orig = []
    gen_rtype = []
    zone_int_gen = np.zeros(N, dtype=int)

    is_uc = np.zeros(N, dtype=bool)
    is_storage = np.zeros(N, dtype=bool)
    is_hydro = np.zeros(N, dtype=bool)
    is_rps = np.zeros(N, dtype=bool)
    can_retire = np.zeros(N, dtype=bool)

    cap_size = np.ones(N, dtype=float)
    min_frac = np.zeros(N, dtype=float)

    ramp_up = np.zeros(N, dtype=float)
    ramp_down = np.zeros(N, dtype=float)
    min_up = np.zeros(N, dtype=int)
    min_down = np.zeros(N, dtype=int)

    var_cost = np.zeros(N, dtype=float)
    fixedOM = np.zeros(N, dtype=float)
    inv_cost = np.zeros(N, dtype=float)
    startup_cost = np.zeros(N, dtype=float)
    co2 = np.zeros(N, dtype=float)

    hydro_duration = np.zeros(N, dtype=float)
    hydro_energy_inv = np.zeros(N, dtype=float)
    hydro_energy_fixedOM = np.zeros(N, dtype=float)

    eff_c = np.ones(N, dtype=float)
    eff_d = np.ones(N, dtype=float)
    self_rate = np.zeros(N, dtype=float)
    min_dur = np.zeros(N, dtype=float)
    max_dur = np.zeros(N, dtype=float)

    energy_inv_cost = np.zeros(N, dtype=float)
    energy_fixedOM = np.zeros(N, dtype=float)
    energy_block = np.ones(N, dtype=float)

    existing_power = np.zeros(N, dtype=float)
    existing_energy = np.zeros(N, dtype=float)

    group_cap_lists = defaultdict(list)
    group_energy_lists = defaultdict(list)

    for i, g in enumerate(gens):
        if not isinstance(g, dict):
            g = {}

        cid_raw = g.get("cluster_id", g.get("id", i))
        try:
            cid = int(cid_raw)
        except Exception:
            cid = i
        cluster_ids.append(cid)

        rtype = str(g.get("resource_type", ""))
        gen_rtype.append(rtype)

        try:
            zorig = int(g.get("zone", 0))
        except Exception:
            zorig = 0
        gen_zone_orig.append(zorig)
        zone_int_gen[i] = zone_map.get(zorig, 0)

        p = gen_params.get(rtype, {}) if isinstance(gen_params, dict) else {}
        if not isinstance(p, dict):
            p = {}

        is_uc_i = bool(g.get("is_UC", p.get("is_UC", False)))
        is_storage_i = bool(g.get("is_storage", p.get("is_storage", False)))
        is_hydro_i = bool(g.get("is_hydro", p.get("is_hydro", False)))
        is_rps_i = bool(g.get("is_RPS_qualifying", p.get("is_RPS_qualifying", False)))
        can_retire_i = bool(g.get("can_retire", p.get("can_retire", False)))

        rl = rtype.lower()
        if (not is_hydro_i) and ("hydro" in rl):
            is_hydro_i = True
        if (not is_storage_i) and (("battery" in rl) or ("storage" in rl)):
            is_storage_i = True

        is_uc[i] = is_uc_i
        is_storage[i] = is_storage_i
        is_hydro[i] = is_hydro_i
        is_rps[i] = is_rps_i
        can_retire[i] = can_retire_i

        cs = get_num(g, "capacity_size_MW", get_num(p, "capacity_size_MW", 1.0))
        if cs <= 0:
            cs = 1.0
        cap_size[i] = cs

        min_frac[i] = max(0.0, get_num(p, "min_output_frac", 0.0))

        ru = p.get("ramp_up_rate_pct_per_hr", None)
        if ru is not None:
            try:
                ramp_up[i] = max(0.0, float(ru))
            except Exception:
                ramp_up[i] = 0.0

        rd = p.get("ramp_down_rate_pct_per_hr", None)
        if rd is not None:
            try:
                ramp_down[i] = max(0.0, float(rd))
            except Exception:
                ramp_down[i] = 0.0

        min_up[i] = int(max(0.0, get_num(p, ["min_up_time_hr", "minimum_up_time_hr"], 0.0)))
        min_down[i] = int(max(0.0, get_num(p, ["min_down_time_hr", "minimum_down_time_hr"], 0.0)))

        var_cost[i] = get_num(p, "variable_cost_per_MWh", 0.0)
        fixedOM[i] = get_num(p, ["fixed_OM_cost_per_MW_yr", "fixed_OM_cost_per_MW_year"], 0.0)
        inv_cost[i] = get_num(p, ["investment_cost_per_MW_yr", "investment_cost_per_MW_year"], 0.0)
        startup_cost[i] = get_num(p, ["start_up_cost", "startup_cost"], 0.0)
        co2[i] = get_num(p, ["CO2_emission_tons_per_MWh", "CO2_emissions_tons_per_MWh"], 0.0)

        if is_hydro_i:
            hydro_duration[i] = max(0.0, get_num(p, "duration_MWh_per_MW", 0.0))
            hydro_energy_inv[i] = get_num(p, "energy_investment_cost_per_MWh_yr", 0.0)
            hydro_energy_fixedOM[i] = get_num(p, "energy_fixed_OM_cost_per_MWh_yr", 0.0)

        if is_storage_i:
            ec_val = get_num(p, "charging_efficiency", 1.0)
            ed_val = get_num(p, "discharging_efficiency", 1.0)

            if ec_val > 1.0:
                ec_val /= 100.0
            if ed_val > 1.0:
                ed_val /= 100.0

            eff_c[i] = ec_val
            eff_d[i] = ed_val

            sr = get_num(p, "self_discharge_rate_pct", 0.0)
            if sr > 1.0:
                sr /= 100.0
            self_rate[i] = sr

            min_dur_i = max(0.0, get_num(p, "min_duration_MWh_per_MW", 0.0))

            if p.get("max_duration_MWh_per_MW", None) is not None:
                try:
                    max_dur_i = float(p["max_duration_MWh_per_MW"])
                except Exception:
                    max_dur_i = max(min_dur_i, 24.0)
            else:
                max_dur_i = max(min_dur_i, 24.0)

            if not np.isfinite(max_dur_i) or max_dur_i > 1e30:
                max_dur_i = 1e30
            if max_dur_i < min_dur_i:
                max_dur_i = min_dur_i

            min_dur[i] = min_dur_i
            max_dur[i] = max_dur_i

            energy_inv_cost[i] = get_num(p, "energy_investment_cost_per_MWh_yr", 0.0)
            energy_fixedOM[i] = get_num(p, "energy_fixed_OM_cost_per_MWh_yr", 0.0)

            eb = get_num(p, "energy_capacity_size_MWh", 1.0)
            if eb <= 0:
                eb = 1.0
            energy_block[i] = eb

        key = (zorig, rtype)

        mc = p.get("max_capacity_MW", None)
        if mc is not None:
            try:
                val = float(mc)
                if np.isfinite(val):
                    group_cap_lists[key].append(max(0.0, val))
            except Exception:
                pass

        me = p.get("max_energy_capacity_MWh", None)
        if me is not None:
            try:
                val = float(me)
                if np.isfinite(val):
                    group_energy_lists[key].append(max(0.0, val))
            except Exception:
                pass

    cap_size = np.nan_to_num(cap_size, nan=1.0, posinf=1.0, neginf=1.0)
    cap_size = np.maximum(cap_size, 1e-6)

    p_inc = np.where(is_uc, cap_size, 1.0)
    e_inc = np.where(is_uc, energy_block, 1.0)

    min_frac = np.nan_to_num(min_frac, nan=0.0, posinf=1.0, neginf=0.0)
    min_frac = np.clip(min_frac, 0.0, 1.0)

    ramp_up = np.nan_to_num(ramp_up, nan=0.0, posinf=0.0, neginf=0.0)
    ramp_down = np.nan_to_num(ramp_down, nan=0.0, posinf=0.0, neginf=0.0)
    ramp_up = np.maximum(ramp_up, 0.0)
    ramp_down = np.maximum(ramp_down, 0.0)

    var_cost = np.nan_to_num(var_cost, nan=0.0, posinf=0.0, neginf=0.0)
    fixedOM = np.nan_to_num(fixedOM, nan=0.0, posinf=0.0, neginf=0.0)
    inv_cost = np.nan_to_num(inv_cost, nan=0.0, posinf=0.0, neginf=0.0)
    startup_cost = np.nan_to_num(startup_cost, nan=0.0, posinf=0.0, neginf=0.0)
    co2 = np.nan_to_num(co2, nan=0.0, posinf=0.0, neginf=0.0)

    hydro_duration = np.nan_to_num(hydro_duration, nan=0.0, posinf=0.0, neginf=0.0)
    hydro_energy_inv = np.nan_to_num(hydro_energy_inv, nan=0.0, posinf=0.0, neginf=0.0)
    hydro_energy_fixedOM = np.nan_to_num(hydro_energy_fixedOM, nan=0.0, posinf=0.0, neginf=0.0)

    eff_c = np.nan_to_num(eff_c, nan=1.0, posinf=1.0, neginf=0.0)
    eff_d = np.nan_to_num(eff_d, nan=1.0, posinf=1.0, neginf=0.0)
    eff_c = np.clip(eff_c, 0.0, 1.0)
    eff_d = np.clip(eff_d, 0.0, 1.0)

    self_rate = np.nan_to_num(self_rate, nan=0.0, posinf=0.999, neginf=0.0)
    self_rate = np.clip(self_rate, 0.0, 0.999)

    min_dur = np.nan_to_num(min_dur, nan=0.0, posinf=1e30, neginf=0.0)
    max_dur = np.nan_to_num(max_dur, nan=0.0, posinf=1e30, neginf=0.0)
    min_dur = np.maximum(min_dur, 0.0)
    max_dur = np.maximum(max_dur, min_dur)

    energy_inv_cost = np.nan_to_num(energy_inv_cost, nan=0.0, posinf=0.0, neginf=0.0)
    energy_fixedOM = np.nan_to_num(energy_fixedOM, nan=0.0, posinf=0.0, neginf=0.0)
    energy_block = np.nan_to_num(energy_block, nan=1.0, posinf=1.0, neginf=1.0)
    energy_block = np.maximum(energy_block, 1e-6)
    e_inc = np.where(is_uc, energy_block, 1.0)

    if isinstance(existing_cap, dict):
        for i, cid in enumerate(cluster_ids):
            v = existing_cap.get(str(cid), existing_cap.get(cid, 0.0))
            try:
                existing_power[i] = float(v)
            except Exception:
                existing_power[i] = 0.0

    existing_power = np.nan_to_num(existing_power, nan=0.0, posinf=0.0, neginf=0.0)
    existing_power = np.maximum(existing_power, 0.0)

    avail = np.ones((N, total_hours), dtype=float)
    if isinstance(avail_dict, dict):
        for i, cid in enumerate(cluster_ids):
            arr = avail_dict.get(str(cid), avail_dict.get(cid, None))
            if arr is not None:
                try:
                    a = np.atleast_1d(np.asarray(arr, dtype=float))
                except Exception:
                    a = np.ones(1, dtype=float)

                a = np.nan_to_num(a, nan=1.0, posinf=1.0, neginf=0.0)
                if a.size == 0:
                    continue

                if a.size == 1:
                    avail[i, :] = a[0]
                else:
                    n = min(total_hours, a.size)
                    avail[i, :n] = a[:n]

    for i in range(N):
        if np.max(avail[i, :]) > 1.5:
            avail[i, :] /= 100.0

    avail = np.nan_to_num(avail, nan=1.0, posinf=1.0, neginf=0.0)
    avail = np.clip(avail, 0.0, 1.0)

    hydro_inflow = np.zeros((N, total_hours), dtype=float)
    if isinstance(hydro_dict, dict):
        for i, cid in enumerate(cluster_ids):
            if is_hydro[i]:
                arr = hydro_dict.get(str(cid), hydro_dict.get(cid, None))
                if arr is not None:
                    try:
                        a = np.atleast_1d(np.asarray(arr, dtype=float))
                    except Exception:
                        a = np.zeros(1, dtype=float)

                    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
                    if a.size == 0:
                        continue

                    if a.size == 1:
                        hydro_inflow[i, :] = a[0]
                    else:
                        n = min(total_hours, a.size)
                        hydro_inflow[i, :n] = a[:n]

    hydro_inflow = np.nan_to_num(hydro_inflow, nan=0.0, posinf=0.0, neginf=0.0)
    hydro_inflow = np.maximum(hydro_inflow, 0.0)

    for i in range(N):
        if is_storage[i] and existing_power[i] > 0.0:
            if cap_size[i] > 1e-9 and energy_block[i] > 1e-9:
                ratio = energy_block[i] / cap_size[i]
            else:
                ratio = min_dur[i]

            if ratio <= 0.0:
                ratio = min_dur[i]

            ratio = min(max(ratio, min_dur[i]), max_dur[i])
            existing_energy[i] = existing_power[i] * ratio

    existing_energy = np.nan_to_num(existing_energy, nan=0.0, posinf=0.0, neginf=0.0)
    existing_energy = np.maximum(existing_energy, 0.0)

    ret_ub = np.zeros(N, dtype=float)
    for i in range(N):
        if can_retire[i]:
            ret_ub[i] = max(0.0, np.floor(existing_power[i] / max(p_inc[i], 1e-9) + 1e-6))

    total_demand_per_hour = demand_mat.sum(axis=0)
    max_demand = float(np.max(total_demand_per_hour)) if total_hours > 0 else 0.0
    if not np.isfinite(max_demand):
        max_demand = 0.0

    default_cap = max(1000.0, 100.0 * max_demand + float(existing_power.sum()))
    max_int_ub = 10_000_000

    group_max_cap = {k: max(0.0, min(v)) for k, v in group_cap_lists.items() if v}
    group_max_energy = {k: max(0.0, min(v)) for k, v in group_energy_lists.items() if v}

    new_ub = np.zeros(N, dtype=float)
    for i in range(N):
        key = (gen_zone_orig[i], gen_rtype[i])
        ub_mw = group_max_cap.get(key, default_cap)

        if not np.isfinite(ub_mw) or ub_mw < 0:
            ub_mw = default_cap

        ub = int(np.floor(ub_mw / max(p_inc[i], 1e-9) + 1e-6))
        new_ub[i] = float(max(0, min(ub, max_int_ub)))

    storage_indices = [i for i in range(N) if is_storage[i]]
    hydro_indices = [i for i in range(N) if is_hydro[i]]
    uc_indices = [i for i in range(N) if is_uc[i]]

    S = len(storage_indices)
    HY = len(hydro_indices)
    U = len(uc_indices)

    storage_ids = [cluster_ids[i] for i in storage_indices]
    hydro_ids = [cluster_ids[i] for i in hydro_indices]
    uc_ids = [cluster_ids[i] for i in uc_indices]

    existing_energy_storage = np.asarray([existing_energy[i] for i in storage_indices], dtype=float)
    e_inc_storage = np.asarray([e_inc[i] for i in storage_indices], dtype=float)

    MAX_DURATION_CAP = 1_000_000.0

    e_new_ub = np.zeros(S, dtype=float)
    e_ret_ub = np.zeros(S, dtype=float)

    for k, i in enumerate(storage_indices):
        key = (gen_zone_orig[i], gen_rtype[i])

        if key in group_max_energy:
            ub_mwh = group_max_energy[key]
        else:
            dur_for_ub = max(min_dur[i], min(max_dur[i], MAX_DURATION_CAP))
            max_power = existing_power[i] + new_ub[i] * p_inc[i]
            ub_mwh = dur_for_ub * max_power

        if not np.isfinite(ub_mwh) or ub_mwh < 0:
            ub_count = max_int_ub
        else:
            ub_count = int(min(max_int_ub, np.floor(ub_mwh / max(e_inc[i], 1e-9) + 1e-6)))

        e_new_ub[k] = float(max(0, ub_count))

        if can_retire[i]:
            e_ret_ub[k] = max(0.0, np.floor(existing_energy[i] / max(e_inc[i], 1e-9) + 1e-6))

    line_ids = []
    existing_line = []
    max_line = []
    trans_inv = []
    from_int = []
    to_int = []

    for l in lines:
        if not isinstance(l, dict):
            continue

        lid_raw = l.get("line_id", l.get("id", len(line_ids)))
        try:
            lid = int(lid_raw)
        except Exception:
            lid = len(line_ids)
        line_ids.append(lid)

        ex = max(0.0, get_num(l, ["existing_capacity_MW", "capacity_MW"], 0.0))
        mx = get_num(l, "max_capacity_MW", ex)
        if (not np.isfinite(mx)) or mx < ex:
            mx = ex

        existing_line.append(ex)
        max_line.append(mx)
        trans_inv.append(get_num(l, ["investment_cost_per_MW_yr", "investment_cost_per_MW_year"], 0.0))

        try:
            from_int.append(zone_map.get(int(l.get("from_zone", 0)), 0))
        except Exception:
            from_int.append(0)

        try:
            to_int.append(zone_map.get(int(l.get("to_zone", 0)), 0))
        except Exception:
            to_int.append(0)

    L = len(line_ids)

    existing_line = np.asarray(existing_line, dtype=float)
    max_line = np.asarray(max_line, dtype=float)
    trans_inv = np.asarray(trans_inv, dtype=float)

    if L > 0:
        line_ub = np.maximum(0.0, np.floor(max_line - existing_line + 1e-6))
        line_ub = np.nan_to_num(line_ub, nan=0.0, posinf=max_int_ub, neginf=0.0)
        line_ub = np.minimum(line_ub, max_int_ub)
    else:
        line_ub = np.asarray([], dtype=float)

    seg_ids = []
    seg_zone_orig = []
    seg_zone_int = []
    seg_frac = []
    seg_cost = []

    for s in segments:
        if not isinstance(s, dict):
            continue

        sid_raw = s.get("segment_id", s.get("id", len(seg_ids)))
        try:
            sid = int(sid_raw)
        except Exception:
            sid = len(seg_ids)

        try:
            zorig = int(s.get("zone", 0))
        except Exception:
            zorig = 0

        seg_ids.append(sid)
        seg_zone_orig.append(zorig)
        seg_zone_int.append(zone_map.get(zorig, 0))
        seg_frac.append(max(0.0, get_num(s, ["max_nse_fraction", "max_unserved_fraction"], 0.0)))
        seg_cost.append(get_num(s, ["nse_cost_per_MWh", "unserved_energy_cost_per_MWh", "cost_per_MWh"], 0.0))

    G = len(seg_ids)
    seg_frac = np.asarray(seg_frac, dtype=float)
    seg_cost = np.asarray(seg_cost, dtype=float)
    seg_zone_int = np.asarray(seg_zone_int, dtype=int)

    seg_frac = np.nan_to_num(seg_frac, nan=0.0, posinf=0.0, neginf=0.0)
    seg_cost = np.nan_to_num(seg_cost, nan=0.0, posinf=0.0, neginf=0.0)
    seg_frac = np.maximum(seg_frac, 0.0)

    try:
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()
    except Exception:
        env = gp.Env()
        env.setParam("OutputFlag", 0)

    model = gp.Model(env=env)

    flags = {
        "le_coeff": True,
        "ge_coeff": True,
        "le_scalar": True,
        "ge_scalar": True,
        "le_mvar": True,
        "eq_mvar": True,
        "le_mvar_coeff": True,
        "ge_mvar_scalar": True,
        "le_sum_var": True,
        "ge_sum_scalar": True,
        "le_array": True,
    }

    def add_le_coeff_var(x, coeff, var):
        coeff = np.asarray(coeff, dtype=float)
        if flags["le_coeff"]:
            try:
                model.addConstr(x <= var * coeff)
                return
            except Exception:
                flags["le_coeff"] = False
        for t in range(x.shape[0]):
            model.addConstr(x[t] <= coeff[t] * var)

    def add_ge_scalar(x, scale, var):
        if flags["ge_scalar"]:
            try:
                model.addConstr(x >= scale * var)
                return
            except Exception:
                flags["ge_scalar"] = False
        for t in range(x.shape[0]):
            model.addConstr(x[t] >= scale * var)

    def add_le_scalar(x, scale, var):
        if flags["le_scalar"]:
            try:
                model.addConstr(x <= scale * var)
                return
            except Exception:
                flags["le_scalar"] = False
        for t in range(x.shape[0]):
            model.addConstr(x[t] <= scale * var)

    def add_le_var(x, var):
        add_le_scalar(x, 1.0, var)

    def add_le_mvar(lhs, rhs):
        if flags["le_mvar"]:
            try:
                model.addConstr(lhs <= rhs)
                return
            except Exception:
                flags["le_mvar"] = False
        for t in range(lhs.shape[0]):
            model.addConstr(lhs[t] <= rhs[t])

    def add_eq_mvar(lhs, rhs):
        if flags["eq_mvar"]:
            try:
                model.addConstr(lhs == rhs)
                return
            except Exception:
                flags["eq_mvar"] = False
        for t in range(lhs.shape[0]):
            model.addConstr(lhs[t] == rhs[t])

    def add_le_mvar_coeff(x, y, coeff):
        coeff = np.asarray(coeff, dtype=float)
        if flags["le_mvar_coeff"]:
            try:
                model.addConstr(x <= y * coeff)
                return
            except Exception:
                flags["le_mvar_coeff"] = False
        for t in range(x.shape[0]):
            model.addConstr(x[t] <= y[t] * coeff[t])

    def add_ge_mvar_scalar(x, y, scale):
        if flags["ge_mvar_scalar"]:
            try:
                model.addConstr(x >= y * scale)
                return
            except Exception:
                flags["ge_mvar_scalar"] = False
        for t in range(x.shape[0]):
            model.addConstr(x[t] >= y[t] * scale)

    def add_le_sum_var(x, y, var):
        if flags["le_sum_var"]:
            try:
                model.addConstr(x + y <= var)
                return
            except Exception:
                flags["le_sum_var"] = False
        for t in range(x.shape[0]):
            model.addConstr(x[t] + y[t] <= var)

    def add_ge_sum_scalar(x, y, scale, var):
        if flags["ge_sum_scalar"]:
            try:
                model.addConstr(x + y >= scale * var)
                return
            except Exception:
                flags["ge_sum_scalar"] = False
        for t in range(x.shape[0]):
            model.addConstr(x[t] + y[t] >= scale * var)

    if N > 0:
        y_P_new = model.addMVar(N, vtype=GRB.INTEGER, lb=0.0, ub=new_ub, name="y_P_new")
        y_P_ret = model.addMVar(N, vtype=GRB.INTEGER, lb=0.0, ub=ret_ub, name="y_P_ret")
        y_P = model.addMVar(N, vtype=GRB.CONTINUOUS, lb=0.0, name="y_P")

        for i in range(N):
            model.addConstr(
                y_P[i] == existing_power[i] + p_inc[i] * y_P_new[i] - p_inc[i] * y_P_ret[i]
            )

        x_inj = model.addMVar((N, total_hours), lb=0.0, name="x_inj")
    else:
        y_P_new = model.addMVar(1, vtype=GRB.INTEGER, lb=0.0, ub=0.0, name="y_P_new")
        y_P_ret = model.addMVar(1, vtype=GRB.INTEGER, lb=0.0, ub=0.0, name="y_P_ret")
        y_P = model.addMVar(1, vtype=GRB.CONTINUOUS, lb=0.0, ub=0.0, name="y_P")
        x_inj = model.addMVar((1, total_hours), lb=0.0, ub=0.0, name="x_inj")

    if S > 0:
        y_E_new = model.addMVar(S, vtype=GRB.INTEGER, lb=0.0, ub=e_new_ub, name="y_E_new")
        y_E_ret = model.addMVar(S, vtype=GRB.INTEGER, lb=0.0, ub=e_ret_ub, name="y_E_ret")
        y_E = model.addMVar(S, vtype=GRB.CONTINUOUS, lb=0.0, name="y_E")

        for k in range(S):
            model.addConstr(
                y_E[k] == existing_energy_storage[k]
                + e_inc_storage[k] * y_E_new[k]
                - e_inc_storage[k] * y_E_ret[k]
            )

        x_wdw = model.addMVar((S, total_hours), lb=0.0, name="x_wdw")
        x_soc = model.addMVar((S, total_hours), lb=0.0, name="x_soc")
    else:
        y_E_new = None
        y_E_ret = None
        y_E = None
        x_wdw = None
        x_soc = None

    if HY > 0:
        x_lvl = model.addMVar((HY, total_hours), lb=0.0, name="x_lvl")
        x_spill = model.addMVar((HY, total_hours), lb=0.0, name="x_spill")
    else:
        x_lvl = None
        x_spill = None

    if U > 0:
        x_commit = model.addMVar((U, total_hours), lb=0.0, name="x_commit")
        x_start = model.addMVar((U, total_hours), lb=0.0, name="x_start")
        x_shut = model.addMVar((U, total_hours), lb=0.0, name="x_shut")
    else:
        x_commit = None
        x_start = None
        x_shut = None

    if G > 0:
        nse_ub = np.zeros((G, total_hours), dtype=float)
        for k in range(G):
            z = seg_zone_int[k]
            nse_ub[k, :] = seg_frac[k] * demand_mat[z, :]

        x_nse = model.addMVar((G, total_hours), lb=0.0, ub=nse_ub, name="x_nse")
    else:
        x_nse = None

    if L > 0:
        y_F_new = model.addMVar(L, vtype=GRB.INTEGER, lb=0.0, ub=line_ub, name="y_F_new")
        y_F = model.addMVar(L, vtype=GRB.CONTINUOUS, lb=existing_line, ub=max_line, name="y_F")

        for l in range(L):
            model.addConstr(y_F[l] == existing_line[l] + y_F_new[l])

        x_flow = model.addMVar((L, total_hours), lb=-GRB.INFINITY, name="x_flow")
    else:
        y_F_new = None
        y_F = None
        x_flow = None

    zone_gen_idx = [[] for _ in range(Z)]
    for i in range(N):
        zone_gen_idx[zone_int_gen[i]].append(i)

    zone_stor_idx = [[] for _ in range(Z)]
    for k, i in enumerate(storage_indices):
        zone_stor_idx[zone_int_gen[i]].append(k)

    line_to_zone = [[] for _ in range(Z)]
    line_from_zone = [[] for _ in range(Z)]
    for l in range(L):
        line_to_zone[to_int[l]].append(l)
        line_from_zone[from_int[l]].append(l)

    zone_nse_idx = [[] for _ in range(Z)]
    for k in range(G):
        zone_nse_idx[seg_zone_int[k]].append(k)

    for z in range(Z):
        expr = None

        for i in zone_gen_idx[z]:
            term = x_inj[i, :]
            expr = term if expr is None else expr + term

        for k in zone_stor_idx[z]:
            term = x_wdw[k, :]
            expr = -term if expr is None else expr - term

        for l in line_to_zone[z]:
            term = x_flow[l, :]
            expr = term if expr is None else expr + term

        for l in line_from_zone[z]:
            term = x_flow[l, :]
            expr = -term if expr is None else expr - term

        for k in zone_nse_idx[z]:
            term = x_nse[k, :]
            expr = term if expr is None else expr + term

        if expr is None:
            if np.any(demand_mat[z, :] > 1e-9):
                for t in range(total_hours):
                    model.addConstr(0.0 == demand_mat[z, t])
        else:
            try:
                model.addConstr(expr == demand_mat[z, :])
            except Exception:
                try:
                    for t in range(total_hours):
                        model.addConstr(expr[t] == demand_mat[z, t])
                except Exception:
                    for t in range(total_hours):
                        model.addConstr(expr == demand_mat[z, t])

    for i in range(N):
        if not is_uc[i]:
            add_le_coeff_var(x_inj[i, :], avail[i, :], y_P[i])
            if (not is_storage[i]) and (not is_hydro[i]) and min_frac[i] > 1e-9:
                add_ge_scalar(x_inj[i, :], min_frac[i], y_P[i])

    for k, i in enumerate(uc_indices):
        unit = cap_size[i]

        add_le_scalar(x_commit[k, :], 1.0 / unit, y_P[i])
        add_le_scalar(x_start[k, :], 1.0 / unit, y_P[i])
        add_le_scalar(x_shut[k, :], 1.0 / unit, y_P[i])

        add_le_mvar_coeff(x_inj[i, :], x_commit[k, :], unit * avail[i, :])

        if min_frac[i] > 1e-9:
            add_ge_mvar_scalar(x_inj[i, :], x_commit[k, :], unit * min_frac[i])

        add_le_mvar(x_start[k, :], x_commit[k, :])

    for k, i in enumerate(storage_indices):
        add_le_coeff_var(x_wdw[k, :], avail[i, :], y_P[i])
        add_le_coeff_var(x_inj[i, :], avail[i, :], y_P[i])
        add_le_sum_var(x_inj[i, :], x_wdw[k, :], y_P[i])
        add_le_var(x_soc[k, :], y_E[k])

        model.addConstr(y_E[k] >= min_dur[i] * y_P[i])

        max_dur_eff = max(min_dur[i], min(max_dur[i], MAX_DURATION_CAP))
        model.addConstr(y_E[k] <= max_dur_eff * y_P[i])

        ec = eff_c[i]
        ed = eff_d[i]
        sr = self_rate[i]
        inv_ed = 0.0 if ed <= 1e-6 else 1.0 / ed

        if ed <= 1e-6:
            model.addConstr(x_inj[i, :] == 0.0)

        for w in range(num_weeks):
            st = w * H
            en = st + H

            soc = x_soc[k, st:en]
            wdw = x_wdw[k, st:en]
            inj = x_inj[i, st:en]

            if H > 1:
                add_eq_mvar(
                    soc[1:],
                    (1.0 - sr) * soc[:-1] + ec * wdw[1:] - inv_ed * inj[1:]
                )
                model.addConstr(
                    soc[0] == (1.0 - sr) * soc[-1] + ec * wdw[0] - inv_ed * inj[0]
                )

                add_le_var(ec * wdw[1:] + (1.0 - sr) * soc[:-1], y_E[k])
                model.addConstr(ec * wdw[0] + (1.0 - sr) * soc[-1] <= y_E[k])

                if inv_ed > 0.0:
                    add_le_mvar(inv_ed * inj[1:], soc[:-1])
                    model.addConstr(inv_ed * inj[0] <= soc[-1])
                else:
                    model.addConstr(inj[1:] == 0.0)
                    model.addConstr(inj[0] == 0.0)
            else:
                model.addConstr(
                    soc[0] == (1.0 - sr) * soc[0] + ec * wdw[0] - inv_ed * inj[0]
                )
                model.addConstr(ec * wdw[0] + (1.0 - sr) * soc[0] <= y_E[k])

                if inv_ed > 0.0:
                    model.addConstr(inv_ed * inj[0] <= soc[0])
                else:
                    model.addConstr(inj[0] == 0.0)

    for h, i in enumerate(hydro_indices):
        dur = hydro_duration[i]
        add_le_scalar(x_lvl[h, :], dur, y_P[i])

        if min_frac[i] > 1e-9:
            add_ge_sum_scalar(x_inj[i, :], x_spill[h, :], min_frac[i], y_P[i])

        for w in range(num_weeks):
            st = w * H
            for pos in range(H):
                t = st + pos
                prev = st + ((pos - 1) % H)
                model.addConstr(
                    x_lvl[h, t] == x_lvl[h, prev]
                    + hydro_inflow[i, t] * y_P[i]
                    - x_inj[i, t]
                    - x_spill[h, t]
                )

    for i in range(N):
        if is_uc[i]:
            continue

        rup = ramp_up[i] if np.isfinite(ramp_up[i]) else 0.0
        rdn = ramp_down[i] if np.isfinite(ramp_down[i]) else 0.0

        if rup <= 1e-9 and rdn <= 1e-9:
            continue

        for w in range(num_weeks):
            st = w * H
            gen = x_inj[i, st:st + H]

            if H > 1:
                if rup > 1e-9:
                    add_le_scalar(gen[1:] - gen[:-1], rup, y_P[i])
                    model.addConstr(gen[0] - gen[-1] <= rup * y_P[i])

                if rdn > 1e-9:
                    add_le_scalar(gen[:-1] - gen[1:], rdn, y_P[i])
                    model.addConstr(gen[-1] - gen[0] <= rdn * y_P[i])

    for k, i in enumerate(uc_indices):
        unit = cap_size[i]
        minf = min_frac[i]
        rup = ramp_up[i] if np.isfinite(ramp_up[i]) else 0.0
        rdn = ramp_down[i] if np.isfinite(ramp_down[i]) else 0.0
        up = int(min_up[i])
        down = int(min_down[i])

        for w in range(num_weeks):
            st = w * H
            hours = list(range(st, st + H))

            commit = x_commit[k, st:st + H]
            startv = x_start[k, st:st + H]
            shutv = x_shut[k, st:st + H]

            if H > 1:
                add_eq_mvar(commit[1:] - commit[:-1], startv[1:] - shutv[1:])
                model.addConstr(commit[0] - commit[-1] == startv[0] - shutv[0])

                add_le_mvar(shutv[1:], commit[:-1])
                model.addConstr(shutv[0] <= commit[-1])
            else:
                model.addConstr(startv[0] == shutv[0])
                model.addConstr(shutv[0] <= commit[0])

            up_eff = min(up, H) if up > 0 else 0
            if up_eff > 1:
                for pos, t in enumerate(hours):
                    expr = gp.quicksum(
                        x_start[k, hours[(pos - q) % H]]
                        for q in range(up_eff)
                    )
                    model.addConstr(x_commit[k, t] >= expr)

            down_eff = min(down, H) if down > 0 else 0
            if down_eff >= 1:
                units_expr = y_P[i] / unit
                for pos, t in enumerate(hours):
                    expr = gp.quicksum(
                        x_shut[k, hours[(pos - q) % H]]
                        for q in range(down_eff)
                    )
                    model.addConstr(x_commit[k, t] + expr <= units_expr)

            if rup > 1e-9 or rdn > 1e-9:
                for pos, t in enumerate(hours):
                    prev = hours[(pos - 1) % H]
                    continuing = x_commit[k, t] - x_start[k, t]

                    if rup > 1e-9:
                        startup_cap = min(float(avail[i, t]), max(minf, rup))
                        rhs_up = unit * (
                            rup * continuing
                            + startup_cap * x_start[k, t]
                            - minf * x_shut[k, t]
                        )
                        model.addConstr(x_inj[i, t] - x_inj[i, prev] <= rhs_up)

                    if rdn > 1e-9:
                        shutdown_cap = min(float(avail[i, prev]), max(minf, rdn))
                        rhs_down = unit * (
                            rdn * continuing
                            + shutdown_cap * x_shut[k, t]
                            - minf * x_start[k, t]
                        )
                        model.addConstr(x_inj[i, prev] - x_inj[i, t] <= rhs_down)

    if L > 0:
        for l in range(L):
            add_le_var(x_flow[l, :], y_F[l])
            add_le_var(-x_flow[l, :], y_F[l])

    for key, maxcap in group_max_cap.items():
        idxs = [i for i in range(N) if (gen_zone_orig[i], gen_rtype[i]) == key]
        if idxs:
            model.addConstr(gp.quicksum(p_inc[i] * y_P_new[i] for i in idxs) <= maxcap)

    if S > 0:
        for key, maxe in group_max_energy.items():
            idxs = [
                k for k, i in enumerate(storage_indices)
                if (gen_zone_orig[i], gen_rtype[i]) == key
            ]
            if idxs:
                model.addConstr(
                    gp.quicksum(e_inc_storage[k] * y_E_new[k] for k in idxs) <= maxe
                )

    obj = 0.0
    cost_tol = 1e-12

    if N > 0:
        invest_coeff = inv_cost.copy()
        fixed_coeff = fixedOM.copy()

        for i in range(N):
            if is_hydro[i]:
                invest_coeff[i] += hydro_energy_inv[i] * hydro_duration[i]
                fixed_coeff[i] += hydro_energy_fixedOM[i] * hydro_duration[i]

        coeff_new = invest_coeff * p_inc

        if np.any(np.abs(coeff_new) > cost_tol):
            obj = obj + (y_P_new @ coeff_new)

        if np.any(np.abs(fixed_coeff) > cost_tol):
            obj = obj + (y_P @ fixed_coeff)

    if S > 0:
        energy_inv_coeff = np.asarray([energy_inv_cost[i] for i in storage_indices], dtype=float)
        energy_fixed_coeff = np.asarray([energy_fixedOM[i] for i in storage_indices], dtype=float)

        coeff_e_new = energy_inv_coeff * e_inc_storage
        if np.any(np.abs(coeff_e_new) > cost_tol):
            obj = obj + (y_E_new @ coeff_e_new)

        if np.any(np.abs(energy_fixed_coeff) > cost_tol):
            obj = obj + (y_E @ energy_fixed_coeff)

    if L > 0:
        if np.any(np.abs(trans_inv) > cost_tol):
            obj = obj + (y_F_new @ trans_inv)

    for i in range(N):
        c = var_cost[i]
        if abs(c) > cost_tol:
            obj = obj + (x_inj[i, :] @ (c * hour_weight))

    if S > 0:
        for k, i in enumerate(storage_indices):
            c = var_cost[i]
            if abs(c) > cost_tol:
                obj = obj + (x_wdw[k, :] @ (c * hour_weight))

    if U > 0:
        for k, i in enumerate(uc_indices):
            c = startup_cost[i]
            if c > cost_tol:
                obj = obj + (x_start[k, :] @ (c * hour_weight))

    if G > 0:
        for k in range(G):
            c = seg_cost[k]
            if c > cost_tol:
                obj = obj + (x_nse[k, :] @ (c * hour_weight))

    total_weighted_demand = float(np.sum(hour_weight * demand_mat.sum(axis=0)))
    if not np.isfinite(total_weighted_demand):
        total_weighted_demand = 0.0
    total_weighted_demand = max(0.0, total_weighted_demand)

    ptype = str(policy.get("type", "REF")).upper()

    x_RPS = None
    x_CO2 = None

    if "RPS" in ptype:
        share = max(0.0, get_num(policy, ["RPS_share", "rps_share"], 0.0))
        if share > 1.0:
            share /= 100.0

        rps_ub = max(1.0, total_weighted_demand * max(1.0, share) * 10.0 + 1e6)
        x_RPS = model.addMVar(num_weeks, lb=0.0, ub=rps_ub, name="x_RPS")

        rps_expr = 0.0
        for i in range(N):
            if is_rps[i]:
                rps_expr = rps_expr + (x_inj[i, :] @ hour_weight)

        rps_expr = scalarize_expr(rps_expr)
        model.addConstr(rps_expr + x_RPS.sum() >= share * total_weighted_demand)

        penalty = get_num(policy, "RPS_noncompliance_cost_per_MWh", 0.0)
        if abs(penalty) > cost_tol:
            obj = obj + penalty * x_RPS.sum()

    if "CO2" in ptype:
        cap_rate = get_num(policy, "CO2_cap_tons_per_MWh", 0.05)

        max_co2_rate = float(np.max(co2)) if co2.size > 0 else 1.0
        if (not np.isfinite(max_co2_rate)) or max_co2_rate <= 0.0:
            max_co2_rate = 1.0

        co2_ub = max(
            1.0,
            total_weighted_demand * max(1.0, max_co2_rate, cap_rate) * 1000.0 + 1e9
        )

        x_CO2 = model.addMVar(num_weeks, lb=0.0, ub=co2_ub, name="x_CO2")

        co2_expr = 0.0
        for i in range(N):
            c = co2[i]
            if abs(c) > cost_tol:
                co2_expr = co2_expr + (x_inj[i, :] @ (c * hour_weight))

        if S > 0:
            for k, i in enumerate(storage_indices):
                c = co2[i]
                if abs(c) > cost_tol:
                    co2_expr = co2_expr + (x_wdw[k, :] @ (c * hour_weight))

        co2_expr = scalarize_expr(co2_expr)
        model.addConstr(co2_expr - x_CO2.sum() <= cap_rate * total_weighted_demand)

        penalty = get_num(policy, "CO2_noncompliance_cost_per_ton", 0.0)
        if abs(penalty) > cost_tol:
            obj = obj + penalty * x_CO2.sum()

    obj = scalarize_expr(obj)

    def set_objective_safe(m, expr):
        candidates = [expr]

        if hasattr(expr, "sum"):
            try:
                candidates.append(expr.sum())
            except Exception:
                pass

        if hasattr(expr, "item"):
            try:
                candidates.append(expr.item())
            except Exception:
                pass

        if isinstance(expr, (int, float, np.number)):
            candidates.append(gp.LinExpr(float(expr)))

        for cand in candidates:
            try:
                m.setObjective(cand, GRB.MINIMIZE)
                return
            except Exception:
                pass

        try:
            m.setObjective(gp.quicksum(expr), GRB.MINIMIZE)
        except Exception:
            m.setObjective(gp.LinExpr(0.0), GRB.MINIMIZE)

    set_objective_safe(model, obj)

    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.OutputFlag = 0

    elapsed = time.time() - start_time
    remaining = float(time_limit) - elapsed

    has_solution = False

    if remaining > 0.5:
        model.Params.TimeLimit = max(0.1, remaining - 0.5)

        def gurobi_callback(cb_model, where):
            if where == GRB.Callback.MIPSOL:
                try:
                    objval = cb_model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    if logger is not None:
                        logger.log(float(objval))
                except Exception:
                    pass

        try:
            if logger is not None:
                model.optimize(gurobi_callback)
            else:
                model.optimize()
        except Exception:
            try:
                model.optimize()
            except Exception:
                pass

        try:
            has_solution = model.SolCount > 0
        except Exception:
            has_solution = False

    obj_val = 1e100

    yP_new_val = np.zeros(N, dtype=float)
    yP_ret_val = np.zeros(N, dtype=float)

    yE_new_val = np.zeros(S, dtype=float)
    yE_ret_val = np.zeros(S, dtype=float)

    yF_new_val = np.zeros(L, dtype=float)

    x_inj_val = np.zeros((N, total_hours), dtype=float)
    x_wdw_val = np.zeros((S, total_hours), dtype=float)
    x_soc_val = np.zeros((S, total_hours), dtype=float)

    x_lvl_val = np.zeros((HY, total_hours), dtype=float)
    x_spill_val = np.zeros((HY, total_hours), dtype=float)

    x_commit_val = np.zeros((U, total_hours), dtype=float)
    x_start_val = np.zeros((U, total_hours), dtype=float)
    x_shut_val = np.zeros((U, total_hours), dtype=float)

    x_nse_val = np.zeros((G, total_hours), dtype=float)
    x_flow_val = np.zeros((L, total_hours), dtype=float)

    x_RPS_val = np.zeros(num_weeks, dtype=float) if x_RPS is not None else None
    x_CO2_val = np.zeros(num_weeks, dtype=float) if x_CO2 is not None else None

    if has_solution:
        try:
            obj_val = float(model.ObjVal)

            if N > 0:
                yP_new_val = np.asarray(y_P_new.X, dtype=float)
                yP_ret_val = np.asarray(y_P_ret.X, dtype=float)
                x_inj_val = np.asarray(x_inj.X, dtype=float)

            if S > 0:
                yE_new_val = np.asarray(y_E_new.X, dtype=float)
                yE_ret_val = np.asarray(y_E_ret.X, dtype=float)
                x_wdw_val = np.asarray(x_wdw.X, dtype=float)
                x_soc_val = np.asarray(x_soc.X, dtype=float)

            if HY > 0:
                x_lvl_val = np.asarray(x_lvl.X, dtype=float)
                x_spill_val = np.asarray(x_spill.X, dtype=float)

            if U > 0:
                x_commit_val = np.asarray(x_commit.X, dtype=float)
                x_start_val = np.asarray(x_start.X, dtype=float)
                x_shut_val = np.asarray(x_shut.X, dtype=float)

            if G > 0:
                x_nse_val = np.asarray(x_nse.X, dtype=float)

            if L > 0:
                yF_new_val = np.asarray(y_F_new.X, dtype=float)
                x_flow_val = np.asarray(x_flow.X, dtype=float)

            if x_RPS is not None:
                x_RPS_val = np.asarray(x_RPS.X, dtype=float).copy()
            if x_CO2 is not None:
                x_CO2_val = np.asarray(x_CO2.X, dtype=float).copy()

            yP_new_val = np.nan_to_num(yP_new_val, nan=0.0, posinf=0.0, neginf=0.0)
            yP_ret_val = np.nan_to_num(yP_ret_val, nan=0.0, posinf=0.0, neginf=0.0)
            yE_new_val = np.nan_to_num(yE_new_val, nan=0.0, posinf=0.0, neginf=0.0)
            yE_ret_val = np.nan_to_num(yE_ret_val, nan=0.0, posinf=0.0, neginf=0.0)
            yF_new_val = np.nan_to_num(yF_new_val, nan=0.0, posinf=0.0, neginf=0.0)

            x_inj_val = np.nan_to_num(x_inj_val, nan=0.0, posinf=0.0, neginf=0.0)
            x_wdw_val = np.nan_to_num(x_wdw_val, nan=0.0, posinf=0.0, neginf=0.0)
            x_soc_val = np.nan_to_num(x_soc_val, nan=0.0, posinf=0.0, neginf=0.0)
            x_lvl_val = np.nan_to_num(x_lvl_val, nan=0.0, posinf=0.0, neginf=0.0)
            x_spill_val = np.nan_to_num(x_spill_val, nan=0.0, posinf=0.0, neginf=0.0)
            x_commit_val = np.nan_to_num(x_commit_val, nan=0.0, posinf=0.0, neginf=0.0)
            x_start_val = np.nan_to_num(x_start_val, nan=0.0, posinf=0.0, neginf=0.0)
            x_shut_val = np.nan_to_num(x_shut_val, nan=0.0, posinf=0.0, neginf=0.0)
            x_nse_val = np.nan_to_num(x_nse_val, nan=0.0, posinf=0.0, neginf=0.0)
            x_flow_val = np.nan_to_num(x_flow_val, nan=0.0, posinf=0.0, neginf=0.0)

            if x_RPS_val is not None:
                x_RPS_val = np.nan_to_num(x_RPS_val, nan=0.0, posinf=0.0, neginf=0.0)
            if x_CO2_val is not None:
                x_CO2_val = np.nan_to_num(x_CO2_val, nan=0.0, posinf=0.0, neginf=0.0)

        except Exception:
            has_solution = False

    new_units = np.maximum(0.0, np.asarray([clean_int(v) for v in yP_new_val], dtype=float))
    ret_units = np.maximum(0.0, np.asarray([clean_int(v) for v in yP_ret_val], dtype=float))

    if N > 0:
        new_units = np.minimum(new_units, new_ub)
        ret_units = np.minimum(ret_units, ret_ub)

    total_cap = existing_power + p_inc * new_units - p_inc * ret_units
    total_cap = np.nan_to_num(total_cap, nan=0.0, posinf=0.0, neginf=0.0)
    total_cap = np.maximum(total_cap, 0.0)

    if S > 0:
        new_e_units = np.maximum(0.0, np.asarray([clean_int(v) for v in yE_new_val], dtype=float))
        ret_e_units = np.maximum(0.0, np.asarray([clean_int(v) for v in yE_ret_val], dtype=float))

        new_e_units = np.minimum(new_e_units, e_new_ub)
        ret_e_units = np.minimum(ret_e_units, e_ret_ub)

        total_e = existing_energy_storage + e_inc_storage * new_e_units - e_inc_storage * ret_e_units
        total_e = np.nan_to_num(total_e, nan=0.0, posinf=0.0, neginf=0.0)
        total_e = np.maximum(total_e, 0.0)
    else:
        new_e_units = np.asarray([], dtype=float)
        ret_e_units = np.asarray([], dtype=float)
        total_e = np.asarray([], dtype=float)

    if L > 0:
        new_line = np.maximum(0.0, np.asarray([clean_int(v) for v in yF_new_val], dtype=float))
        new_line = np.minimum(new_line, line_ub)

        total_line = existing_line + new_line
        total_line = np.nan_to_num(total_line, nan=0.0, posinf=0.0, neginf=0.0)
        total_line = np.maximum(total_line, 0.0)
    else:
        new_line = np.asarray([], dtype=float)
        total_line = np.asarray([], dtype=float)

    investments = {}
    for i, cid in enumerate(cluster_ids):
        investments[str(cid)] = {
            "resource_type": gen_rtype[i],
            "zone": gen_zone_orig[i],
            "new_units": clean_float(new_units[i]),
            "new_MW": clean_float(p_inc[i] * new_units[i]),
            "retired_units": clean_float(ret_units[i]),
            "retired_MW": clean_float(p_inc[i] * ret_units[i]),
            "total_capacity_MW": clean_float(total_cap[i]),
        }

    storage_energy_investments = {}
    for k, i in enumerate(storage_indices):
        cid = cluster_ids[i]
        storage_energy_investments[str(cid)] = {
            "new_units": clean_float(new_e_units[k]),
            "new_MWh": clean_float(e_inc_storage[k] * new_e_units[k]),
            "total_energy_MWh": clean_float(total_e[k]),
        }

    transmission_investments = {}
    for l, lid in enumerate(line_ids):
        transmission_investments[str(lid)] = {
            "new_MW": clean_float(new_line[l]),
            "total_capacity_MW": clean_float(total_line[l]),
        }

    model_variables = {}
    OP_TOL = 1e-4

    for i, cid in enumerate(cluster_ids):
        if new_units[i] > OP_TOL:
            model_variables[f"y_P_new[{cid}]"] = clean_float(new_units[i])
        if ret_units[i] > OP_TOL:
            model_variables[f"y_P_ret[{cid}]"] = clean_float(ret_units[i])
        if total_cap[i] > OP_TOL:
            model_variables[f"y_P[{cid}]"] = clean_float(total_cap[i])

    for k, i in enumerate(storage_indices):
        cid = cluster_ids[i]
        if new_e_units[k] > OP_TOL:
            model_variables[f"y_E_new[{cid}]"] = clean_float(new_e_units[k])
        if ret_e_units[k] > OP_TOL:
            model_variables[f"y_E_ret[{cid}]"] = clean_float(ret_e_units[k])
        if total_e[k] > OP_TOL:
            model_variables[f"y_E[{cid}]"] = clean_float(total_e[k])

    for l, lid in enumerate(line_ids):
        if new_line[l] > OP_TOL:
            model_variables[f"y_F_new[{lid}]"] = clean_float(new_line[l])
        if total_line[l] > OP_TOL:
            model_variables[f"y_F[{lid}]"] = clean_float(total_line[l])

    def add_2d(vals, row_ids, name):
        if vals is None or vals.size == 0 or len(row_ids) == 0:
            return

        vals = np.asarray(vals, dtype=float)
        if vals.ndim != 2:
            return

        m = min(vals.shape[0], len(row_ids))
        if m <= 0:
            return

        v = vals[:m, :]
        rows, cols = np.nonzero(np.abs(v) > OP_TOL)
        for r, c in zip(rows, cols):
            model_variables[f"{name}[{row_ids[r]},{int(c)}]"] = clean_float(v[r, c])

    add_2d(x_inj_val, cluster_ids, "x_inj")
    add_2d(x_wdw_val, storage_ids, "x_wdw")
    add_2d(x_soc_val, storage_ids, "x_soc")
    add_2d(x_lvl_val, hydro_ids, "x_lvl")
    add_2d(x_spill_val, hydro_ids, "x_spill")
    add_2d(x_commit_val, uc_ids, "x_commit")
    add_2d(x_start_val, uc_ids, "x_start")
    add_2d(x_shut_val, uc_ids, "x_shut")
    add_2d(x_flow_val, line_ids, "x_flow")

    if G > 0 and x_nse_val.size > 0:
        m = min(x_nse_val.shape[0], G)
        v = x_nse_val[:m, :]
        rows, cols = np.nonzero(np.abs(v) > OP_TOL)
        for r, c in zip(rows, cols):
            model_variables[f"x_nse[{seg_ids[r]},{seg_zone_orig[r]},{int(c)}]"] = clean_float(v[r, c])

    if x_RPS_val is not None:
        for w, v in enumerate(x_RPS_val):
            if abs(v) > OP_TOL:
                model_variables[f"x_RPS[{w}]"] = clean_float(v)

    if x_CO2_val is not None:
        for w, v in enumerate(x_CO2_val):
            if abs(v) > OP_TOL:
                model_variables[f"x_CO2[{w}]"] = clean_float(v)

    solution = {
        "objective_value": clean_float(obj_val),
        "model_variables": model_variables,
        "investments": investments,
        "storage_energy_investments": storage_energy_investments,
        "transmission_investments": transmission_investments,
        "model_variable_name_patterns": {
            "y_P_new[i], y_P_ret[i], y_P[i]": "generator capacity variables",
            "y_E_new[i], y_E_ret[i], y_E[i]": "storage-energy variables",
            "y_F_new[l], y_F[l]": "transmission variables",
            "x_inj[i,t], x_wdw[i,t], x_soc[i,t]": "generation/storage operations",
            "x_nse[s,z,t]": "non-served energy",
            "x_lvl[i,t], x_spill[i,t]": "hydro operations",
            "x_flow[l,t]": "transmission flow",
            "x_commit[i,t], x_start[i,t], x_shut[i,t]": "unit commitment",
            "x_RPS[w], x_CO2[w]": "policy slack variables"
        },
    }

    out_dir = os.path.dirname(os.path.abspath(solution_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(solution_path, "w") as f:
        json.dump(solution, f)

    if logger is not None:
        try:
            logger.log_solution(clean_float(obj_val), solution)
        except Exception:
            try:
                logger.log(clean_float(obj_val))
            except Exception:
                pass


if __name__ == "__main__":
    main()
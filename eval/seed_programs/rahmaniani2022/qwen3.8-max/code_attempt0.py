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
            self._write({"objective_value": objective_value})

        def log_solution(self, objective_value, solution_dict):
            self._write({"objective_value": objective_value, "solution": solution_dict})

        def _write(self, record):
            if not self.path:
                return
            try:
                with open(self.path, "a") as f:
                    f.write(json.dumps(record) + "\n")
            except Exception:
                pass


def get_num(d, keys, default=0.0):
    if isinstance(keys, str):
        keys = [keys]
    for k in keys:
        if k in d and d[k] is not None:
            try:
                return float(d[k])
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
    if abs(v) <= 1e-6:
        return 0.0
    r = round(v)
    if abs(v - r) <= 1e-4:
        return float(r)
    return v


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance-path", dest="instance_path", default=None)
    parser.add_argument("--solution-path", dest="solution_path", default=None)
    parser.add_argument("--log-path", dest="log_path", default=None)
    parser.add_argument("--time-limit", dest="time_limit", type=float, default=None)
    parser.add_argument("positional", nargs="*")
    args = parser.parse_args()

    pos = args.positional or []
    instance_path = args.instance_path or (pos[0] if len(pos) > 0 else "instance.json")
    solution_path = args.solution_path or (pos[1] if len(pos) > 1 else "solution.json")
    log_path = args.log_path or (pos[2] if len(pos) > 2 else None)
    time_limit = args.time_limit if args.time_limit is not None else (float(pos[3]) if len(pos) > 3 else 60.0)

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
    gens = inst.get("generators", []) or []
    demand = inst.get("demand", {}) or {}
    lines = inst.get("transmission_lines", []) or []
    segments = inst.get("consumer_segments", []) or []
    gen_params = inst.get("generator_parameters", {}) or {}
    existing_cap = inst.get("existing_capacity", {}) or {}
    avail_dict = inst.get("availability", {}) or {}
    hydro_dict = inst.get("hydro_inflow", {}) or {}
    policy = inst.get("policy", {}) or {}

    num_weeks = int(pp.get("num_weeks", len(inst.get("subperiod_weights", [])) or 1))
    if num_weeks <= 0:
        num_weeks = 1
    H = int(pp.get("hours_per_subperiod", 168))
    if H <= 0:
        H = 168
    total_hours = num_weeks * H

    weights_raw = inst.get("subperiod_weights", [1.0] * num_weeks)
    weights = np.ones(num_weeks, dtype=float)
    for w in range(num_weeks):
        if w < len(weights_raw):
            try:
                weights[w] = float(weights_raw[w])
            except Exception:
                weights[w] = 1.0
    if (not np.all(np.isfinite(weights))) or float(np.sum(weights)) == 0.0:
        weights = np.ones(num_weeks, dtype=float)
    hour_weight = np.repeat(weights / float(H), H)

    zone_set = set()
    for g in gens:
        zone_set.add(int(g.get("zone", 0)))
    for k in demand.keys():
        zone_set.add(int(k))
    for l in lines:
        zone_set.add(int(l.get("from_zone", 0)))
        zone_set.add(int(l.get("to_zone", 0)))
    for s in segments:
        zone_set.add(int(s.get("zone", 0)))
    if not zone_set:
        zone_set.add(0)

    zone_orig = sorted(zone_set)
    zone_map = {z: i for i, z in enumerate(zone_orig)}
    Z = len(zone_orig)

    demand_mat = np.zeros((Z, total_hours), dtype=float)
    for k, series in demand.items():
        z = zone_map.get(int(k))
        if z is None:
            continue
        arr = np.asarray(series, dtype=float)
        n = min(total_hours, len(arr))
        if n > 0:
            demand_mat[z, :n] = arr[:n]

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
    p_inc = np.ones(N, dtype=float)
    min_frac = np.zeros(N, dtype=float)

    ramp_up = np.full(N, np.nan, dtype=float)
    ramp_down = np.full(N, np.nan, dtype=float)
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
    e_inc = np.ones(N, dtype=float)

    existing_power = np.zeros(N, dtype=float)
    existing_energy = np.zeros(N, dtype=float)

    group_cap_lists = defaultdict(list)
    group_energy_lists = defaultdict(list)

    for i, g in enumerate(gens):
        cid = int(g.get("cluster_id", g.get("id", i)))
        cluster_ids.append(cid)

        rtype = str(g.get("resource_type", ""))
        gen_rtype.append(rtype)

        zorig = int(g.get("zone", 0))
        gen_zone_orig.append(zorig)
        zone_int_gen[i] = zone_map.get(zorig, 0)

        p = gen_params.get(rtype, {}) if isinstance(gen_params, dict) else {}

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
        p_inc[i] = cs if is_uc_i else 1.0

        min_frac[i] = get_num(p, "min_output_frac", 0.0)

        ru = p.get("ramp_up_rate_pct_per_hr", None)
        if ru is not None:
            try:
                ramp_up[i] = float(ru)
            except Exception:
                pass
        rd = p.get("ramp_down_rate_pct_per_hr", None)
        if rd is not None:
            try:
                ramp_down[i] = float(rd)
            except Exception:
                pass

        min_up[i] = int(get_num(p, ["min_up_time_hr", "minimum_up_time_hr"], 0.0))
        min_down[i] = int(get_num(p, ["min_down_time_hr", "minimum_down_time_hr"], 0.0))

        var_cost[i] = get_num(p, "variable_cost_per_MWh", 0.0)
        fixedOM[i] = get_num(p, ["fixed_OM_cost_per_MW_yr", "fixed_OM_cost_per_MW_year"], 0.0)
        inv_cost[i] = get_num(p, ["investment_cost_per_MW_yr", "investment_cost_per_MW_year"], 0.0)
        startup_cost[i] = get_num(p, "start_up_cost", 0.0)
        co2[i] = get_num(p, "CO2_emission_tons_per_MWh", 0.0)

        if is_hydro_i:
            hydro_duration[i] = get_num(p, "duration_MWh_per_MW", 0.0)
            hydro_energy_inv[i] = get_num(p, "energy_investment_cost_per_MWh_yr", 0.0)
            hydro_energy_fixedOM[i] = get_num(p, "energy_fixed_OM_cost_per_MWh_yr", 0.0)

        if is_storage_i:
            eff_c[i] = get_num(p, "charging_efficiency", 1.0)
            eff_d[i] = get_num(p, "discharging_efficiency", 1.0)

            sr = get_num(p, "self_discharge_rate_pct", 0.0)
            if sr > 1.0:
                sr /= 100.0
            if sr < 0.0:
                sr = 0.0
            if sr >= 1.0:
                sr = 0.999
            self_rate[i] = sr

            min_dur[i] = get_num(p, "min_duration_MWh_per_MW", 0.0)
            if "max_duration_MWh_per_MW" in p and p["max_duration_MWh_per_MW"] is not None:
                max_dur[i] = float(p["max_duration_MWh_per_MW"])
            else:
                max_dur[i] = max(min_dur[i], 24.0)
            if max_dur[i] < min_dur[i]:
                max_dur[i] = min_dur[i]

            energy_inv_cost[i] = get_num(p, "energy_investment_cost_per_MWh_yr", 0.0)
            energy_fixedOM[i] = get_num(p, "energy_fixed_OM_cost_per_MWh_yr", 0.0)

            eb = get_num(p, "energy_capacity_size_MWh", 1.0)
            if eb <= 0:
                eb = 1.0
            energy_block[i] = eb
            e_inc[i] = eb if is_uc_i else 1.0

        key = (zorig, rtype)
        mc = p.get("max_capacity_MW", None)
        if mc is not None:
            try:
                group_cap_lists[key].append(float(mc))
            except Exception:
                pass
        me = p.get("max_energy_capacity_MWh", None)
        if me is not None:
            try:
                group_energy_lists[key].append(float(me))
            except Exception:
                pass

    for i, cid in enumerate(cluster_ids):
        v = existing_cap.get(str(cid), existing_cap.get(cid, 0.0))
        try:
            existing_power[i] = float(v)
        except Exception:
            existing_power[i] = 0.0

    avail = np.ones((N, total_hours), dtype=float)
    for i, cid in enumerate(cluster_ids):
        arr = avail_dict.get(str(cid), avail_dict.get(cid, None))
        if arr is not None:
            a = np.asarray(arr, dtype=float)
            n = min(total_hours, len(a))
            if n > 0:
                avail[i, :n] = a[:n]
    avail = np.clip(avail, 0.0, 1.0)

    hydro_inflow = np.zeros((N, total_hours), dtype=float)
    for i, cid in enumerate(cluster_ids):
        if is_hydro[i]:
            arr = hydro_dict.get(str(cid), hydro_dict.get(cid, None))
            if arr is not None:
                a = np.asarray(arr, dtype=float)
                n = min(total_hours, len(a))
                if n > 0:
                    hydro_inflow[i, :n] = a[:n]

    for i in range(N):
        if is_storage[i] and existing_power[i] > 0.0:
            if is_uc[i]:
                ratio = energy_block[i] / cap_size[i] if cap_size[i] > 0 else min_dur[i]
                if ratio <= 0:
                    ratio = min_dur[i]
                if max_dur[i] < 1e30:
                    ratio = min(max(ratio, min_dur[i]), max_dur[i])
                else:
                    ratio = max(ratio, min_dur[i])
                existing_energy[i] = existing_power[i] * ratio
            else:
                existing_energy[i] = existing_power[i] * min_dur[i]

    ret_ub = np.zeros(N, dtype=float)
    for i in range(N):
        if can_retire[i]:
            ret_ub[i] = max(0.0, np.floor(existing_power[i] / max(p_inc[i], 1e-9) + 1e-6))

    total_demand_per_hour = demand_mat.sum(axis=0)
    max_demand = float(total_demand_per_hour.max()) if total_hours > 0 else 0.0
    default_cap = max(100.0, 20.0 * max_demand, float(existing_power.sum()))
    max_int_ub = 1_000_000

    group_max_cap = {k: min(v) for k, v in group_cap_lists.items() if v}
    group_max_energy = {k: min(v) for k, v in group_energy_lists.items() if v}

    new_ub = np.zeros(N, dtype=float)
    for i in range(N):
        key = (gen_zone_orig[i], gen_rtype[i])
        ub_mw = group_max_cap.get(key, default_cap)
        if ub_mw < 0:
            ub_mw = 0.0
        ub = int(np.floor(ub_mw / max(p_inc[i], 1e-9) + 1e-6))
        new_ub[i] = float(max(0, min(ub, max_int_ub)))

    storage_indices = [i for i in range(N) if is_storage[i]]
    hydro_indices = [i for i in range(N) if is_hydro[i]]
    uc_indices = [i for i in range(N) if is_uc[i]]

    S = len(storage_indices)
    HY = len(hydro_indices)
    U = len(uc_indices)

    stor_pos = {i: k for k, i in enumerate(storage_indices)}
    hydro_pos = {i: h for h, i in enumerate(hydro_indices)}
    uc_pos = {i: k for k, i in enumerate(uc_indices)}

    e_new_ub = np.zeros(S, dtype=float)
    e_ret_ub = np.zeros(S, dtype=float)
    for k, i in enumerate(storage_indices):
        key = (gen_zone_orig[i], gen_rtype[i])
        if key in group_max_energy:
            ub_mwh = group_max_energy[key]
        else:
            max_power = existing_power[i] + new_ub[i] * p_inc[i]
            ub_mwh = max_dur[i] * max_power if max_dur[i] < 1e30 else max(100.0, default_cap)
        if ub_mwh < 0:
            ub_mwh = 0.0
        e_new_ub[k] = float(max(0, min(int(np.floor(ub_mwh / max(e_inc[i], 1e-9) + 1e-6)), max_int_ub)))
        if can_retire[i]:
            e_ret_ub[k] = max(0.0, np.floor(existing_energy[i] / max(e_inc[i], 1e-9) + 1e-6))

    line_ids = []
    existing_line = []
    max_line = []
    trans_inv = []
    from_int = []
    to_int = []

    for l in lines:
        lid = int(l.get("line_id", l.get("id", len(line_ids))))
        line_ids.append(lid)

        ex = get_num(l, ["existing_capacity_MW", "capacity_MW"], 0.0)
        mx = get_num(l, "max_capacity_MW", ex)
        if mx < ex:
            mx = ex

        existing_line.append(ex)
        max_line.append(mx)
        trans_inv.append(get_num(l, ["investment_cost_per_MW_yr", "investment_cost_per_MW_year"], 0.0))
        from_int.append(zone_map.get(int(l.get("from_zone", 0)), 0))
        to_int.append(zone_map.get(int(l.get("to_zone", 0)), 0))

    L = len(line_ids)
    existing_line = np.asarray(existing_line, dtype=float)
    max_line = np.asarray(max_line, dtype=float)
    trans_inv = np.asarray(trans_inv, dtype=float)
    line_ub = np.maximum(0.0, np.floor(max_line - existing_line + 1e-6))

    seg_ids = []
    seg_zone_orig = []
    seg_zone_int = []
    seg_frac = []
    seg_cost = []

    for s in segments:
        sid = int(s.get("segment_id", s.get("id", len(seg_ids))))
        zorig = int(s.get("zone", 0))
        seg_ids.append(sid)
        seg_zone_orig.append(zorig)
        seg_zone_int.append(zone_map.get(zorig, 0))
        seg_frac.append(get_num(s, ["max_nse_fraction", "max_unserved_fraction"], 0.0))
        seg_cost.append(get_num(s, ["nse_cost_per_MWh", "unserved_energy_cost_per_MWh", "cost_per_MWh"], 0.0))

    G = len(seg_ids)
    seg_frac = np.asarray(seg_frac, dtype=float)
    seg_cost = np.asarray(seg_cost, dtype=float)
    seg_zone_int = np.asarray(seg_zone_int, dtype=int)

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
        "ge_mvar_coeff": True,
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

    def add_ge_coeff_var(x, coeff, var):
        coeff = np.asarray(coeff, dtype=float)
        if flags["ge_coeff"]:
            try:
                model.addConstr(x >= var * coeff)
                return
            except Exception:
                flags["ge_coeff"] = False
        for t in range(x.shape[0]):
            model.addConstr(x[t] >= coeff[t] * var)

    def add_le_scalar(x, scale, var):
        if flags["le_scalar"]:
            try:
                model.addConstr(x <= scale * var)
                return
            except Exception:
                flags["le_scalar"] = False
        for t in range(x.shape[0]):
            model.addConstr(x[t] <= scale * var)

    def add_ge_scalar(x, scale, var):
        if flags["ge_scalar"]:
            try:
                model.addConstr(x >= scale * var)
                return
            except Exception:
                flags["ge_scalar"] = False
        for t in range(x.shape[0]):
            model.addConstr(x[t] >= scale * var)

    def add_le_var(x, var):
        add_le_scalar(x, 1.0, var)

    def add_le_array(x, arr):
        arr = np.asarray(arr, dtype=float)
        if flags["le_array"]:
            try:
                model.addConstr(x <= arr)
                return
            except Exception:
                flags["le_array"] = False
        for t in range(x.shape[0]):
            model.addConstr(x[t] <= arr[t])

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

    def add_ge_mvar_coeff(x, y, coeff):
        coeff = np.asarray(coeff, dtype=float)
        if flags["ge_mvar_coeff"]:
            try:
                model.addConstr(x >= y * coeff)
                return
            except Exception:
                flags["ge_mvar_coeff"] = False
        for t in range(x.shape[0]):
            model.addConstr(x[t] >= y[t] * coeff[t])

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
            model.addConstr(y_P[i] == existing_power[i] + p_inc[i] * y_P_new[i] - p_inc[i] * y_P_ret[i])
        x_inj = model.addMVar((N, total_hours), lb=0.0, name="x_inj")
    else:
        y_P_new = model.addMVar(1, vtype=GRB.INTEGER, lb=0.0, ub=0.0, name="y_P_new")
        y_P_ret = model.addMVar(1, vtype=GRB.INTEGER, lb=0.0, ub=0.0, name="y_P_ret")
        y_P = model.addMVar(1, vtype=GRB.CONTINUOUS, lb=0.0, ub=0.0, name="y_P")
        x_inj = model.addMVar((1, total_hours), lb=0.0, ub=0.0, name="x_inj")

    if S > 0:
        existing_energy_storage = np.asarray([existing_energy[i] for i in storage_indices], dtype=float)
        e_inc_storage = np.asarray([e_inc[i] for i in storage_indices], dtype=float)

        y_E_new = model.addMVar(S, vtype=GRB.INTEGER, lb=0.0, ub=e_new_ub, name="y_E_new")
        y_E_ret = model.addMVar(S, vtype=GRB.INTEGER, lb=0.0, ub=e_ret_ub, name="y_E_ret")
        y_E = model.addMVar(S, vtype=GRB.CONTINUOUS, lb=0.0, name="y_E")

        for k in range(S):
            model.addConstr(
                y_E[k] == existing_energy_storage[k] + e_inc_storage[k] * y_E_new[k] - e_inc_storage[k] * y_E_ret[k]
            )

        x_wdw = model.addMVar((S, total_hours), lb=0.0, name="x_wdw")
        x_soc = model.addMVar((S, total_hours), lb=0.0, name="x_soc")
    else:
        existing_energy_storage = np.asarray([], dtype=float)
        e_inc_storage = np.asarray([], dtype=float)
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
        zone_max_demand = demand_mat.max(axis=1)
        nse_ub = seg_frac * np.maximum(zone_max_demand[seg_zone_int], 0.0)
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

    if N > 0:
        C = np.zeros((Z, N), dtype=float)
        for i in range(N):
            C[zone_int_gen[i], i] = 1.0
        lhs = C @ x_inj
    else:
        lhs = None

    if S > 0:
        D = np.zeros((Z, S), dtype=float)
        for k, i in enumerate(storage_indices):
            D[zone_int_gen[i], k] = 1.0
        term = D @ x_wdw
        lhs = term if lhs is None else lhs - term

    if L > 0:
        A = np.zeros((Z, L), dtype=float)
        for l in range(L):
            A[to_int[l], l] += 1.0
            A[from_int[l], l] -= 1.0
        term = A @ x_flow
        lhs = term if lhs is None else lhs + term

    if G > 0:
        B = np.zeros((Z, G), dtype=float)
        for k in range(G):
            B[seg_zone_int[k], k] = 1.0
        term = B @ x_nse
        lhs = term if lhs is None else lhs + term

    if lhs is not None:
        model.addConstr(lhs == demand_mat)

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
        model.addConstr(y_E[k] <= max_dur[i] * y_P[i])

        ec = eff_c[i]
        ed = eff_d[i]
        sr = self_rate[i]
        inv_ed = 0.0 if ed <= 1e-9 else 1.0 / ed

        if ed <= 1e-9:
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
                model.addConstr(soc[0] == (1.0 - sr) * soc[-1] + ec * wdw[0] - inv_ed * inj[0])

                add_le_var(ec * wdw[1:] + (1.0 - sr) * soc[:-1], y_E[k])
                model.addConstr(ec * wdw[0] + (1.0 - sr) * soc[-1] <= y_E[k])

                if inv_ed > 0.0:
                    add_le_mvar(inv_ed * inj[1:], soc[:-1])
                    model.addConstr(inv_ed * inj[0] <= soc[-1])
                else:
                    model.addConstr(inj[1:] == 0.0)
                    model.addConstr(inj[0] == 0.0)
            else:
                model.addConstr(soc[0] == (1.0 - sr) * soc[0] + ec * wdw[0] - inv_ed * inj[0])
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
                    x_lvl[h, t] == x_lvl[h, prev] + hydro_inflow[i, t] * y_P[i] - x_inj[i, t] - x_spill[h, t]
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
            en = st + H
            gen = x_inj[i, st:en]

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
            en = st + H
            hours = list(range(st, en))

            commit = x_commit[k, st:en]
            startv = x_start[k, st:en]
            shutv = x_shut[k, st:en]

            if H > 1:
                add_eq_mvar(commit[1:] - commit[:-1], startv[1:] - shutv[1:])
                model.addConstr(commit[0] - commit[-1] == startv[0] - shutv[-1])

                add_le_mvar(shutv[1:], commit[:-1])
                model.addConstr(shutv[0] <= commit[-1])
            else:
                model.addConstr(startv[0] == shutv[0])
                model.addConstr(shutv[0] <= commit[0])

            up_eff = min(up, H) if up > 0 else 0
            if up_eff > 1:
                for pos, t in enumerate(hours):
                    expr = gp.quicksum(x_start[k, hours[(pos - q) % H]] for q in range(up_eff))
                    model.addConstr(x_commit[k, t] >= expr)

            down_eff = min(down, H) if down > 0 else 0
            if down_eff >= 1:
                units_expr = y_P[i] / unit
                for pos, t in enumerate(hours):
                    expr = gp.quicksum(x_shut[k, hours[(pos - q) % H]] for q in range(down_eff))
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
                        rhs_down = unit * (
                            rdn * continuing
                            + float(avail[i, prev]) * x_shut[k, t]
                        )
                        model.addConstr(x_inj[i, prev] - x_inj[i, t] <= rhs_down)

    if L > 0:
        for l in range(L):
            add_le_var(x_flow[l, :], y_F[l])
            add_le_var(-x_flow[l, :], y_F[l])

    if G > 0:
        for k in range(G):
            z = seg_zone_int[k]
            bound = seg_frac[k] * np.maximum(demand_mat[z, :], 0.0)
            add_le_array(x_nse[k, :], bound)

    for key, maxcap in group_max_cap.items():
        idxs = [i for i in range(N) if (gen_zone_orig[i], gen_rtype[i]) == key]
        if idxs:
            model.addConstr(gp.quicksum(p_inc[i] * y_P_new[i] for i in idxs) <= maxcap)

    if S > 0:
        for key, maxe in group_max_energy.items():
            idxs = [k for k, i in enumerate(storage_indices) if (gen_zone_orig[i], gen_rtype[i]) == key]
            if idxs:
                model.addConstr(gp.quicksum(e_inc_storage[k] * y_E_new[k] for k in idxs) <= maxe)

    obj = 0

    if N > 0:
        invest_coeff = inv_cost.copy()
        fixed_coeff = fixedOM.copy()

        for i in range(N):
            if is_hydro[i]:
                invest_coeff[i] += hydro_energy_inv[i] * hydro_duration[i]
                fixed_coeff[i] += hydro_energy_fixedOM[i] * hydro_duration[i]

        obj = obj + gp.quicksum(invest_coeff[i] * p_inc[i] * y_P_new[i] for i in range(N))
        obj = obj + gp.quicksum(fixed_coeff[i] * y_P[i] for i in range(N))

    if S > 0:
        energy_inv_coeff = np.asarray([energy_inv_cost[i] for i in storage_indices], dtype=float)
        energy_fixed_coeff = np.asarray([energy_fixedOM[i] for i in storage_indices], dtype=float)
        obj = obj + gp.quicksum(energy_inv_coeff[k] * e_inc_storage[k] * y_E_new[k] for k in range(S))
        obj = obj + gp.quicksum(energy_fixed_coeff[k] * y_E[k] for k in range(S))

    if L > 0:
        obj = obj + gp.quicksum(trans_inv[l] * y_F_new[l] for l in range(L))

    for i in range(N):
        c = var_cost[i]
        if abs(c) > 1e-12:
            obj = obj + (x_inj[i, :] @ (c * hour_weight))

    if S > 0:
        for k, i in enumerate(storage_indices):
            c = var_cost[i]
            if abs(c) > 1e-12:
                obj = obj + (x_wdw[k, :] @ (c * hour_weight))

    if U > 0:
        for k, i in enumerate(uc_indices):
            c = startup_cost[i]
            if c > 1e-12:
                obj = obj + (x_start[k, :] @ (c * hour_weight))

    if G > 0:
        for k in range(G):
            c = seg_cost[k]
            if c > 1e-12:
                obj = obj + (x_nse[k, :] @ (c * hour_weight))

    total_weighted_demand = float(np.sum(hour_weight * demand_mat.sum(axis=0)))
    ptype = str(policy.get("type", "REF")).upper()

    x_RPS = None
    x_CO2 = None

    if "RPS" in ptype:
        x_RPS = model.addMVar(num_weeks, lb=0.0, name="x_RPS")
        rps_expr = 0
        for i in range(N):
            if is_rps[i]:
                rps_expr = rps_expr + (x_inj[i, :] @ hour_weight)

        share = get_num(policy, ["RPS_share", "rps_share"], 0.0)
        model.addConstr(rps_expr + x_RPS.sum() >= share * total_weighted_demand)

        penalty = get_num(policy, "RPS_noncompliance_cost_per_MWh", 0.0)
        if abs(penalty) > 1e-12:
            obj = obj + penalty * x_RPS.sum()

    if "CO2" in ptype:
        x_CO2 = model.addMVar(num_weeks, lb=0.0, name="x_CO2")
        co2_expr = 0

        for i in range(N):
            c = co2[i]
            if abs(c) > 1e-12:
                co2_expr = co2_expr + (x_inj[i, :] @ (c * hour_weight))

        if S > 0:
            for k, i in enumerate(storage_indices):
                c = co2[i]
                if abs(c) > 1e-12:
                    co2_expr = co2_expr + (x_wdw[k, :] @ (c * hour_weight))

        cap_rate = get_num(policy, "CO2_cap_tons_per_MWh", 0.05)
        model.addConstr(co2_expr - x_CO2.sum() <= cap_rate * total_weighted_demand)

        penalty = get_num(policy, "CO2_noncompliance_cost_per_ton", 0.0)
        if abs(penalty) > 1e-12:
            obj = obj + penalty * x_CO2.sum()

    model.setObjective(obj, GRB.MINIMIZE)

    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.OutputFlag = 0

    elapsed = time.time() - start_time
    remaining = max(1.0, float(time_limit) - elapsed - 5.0)
    model.Params.TimeLimit = remaining

    def gurobi_callback(cb_model, where):
        if where == GRB.Callback.MIPSOL:
            try:
                objval = cb_model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if logger is not None:
                    logger.log(objval)
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

    has_solution = False
    try:
        if model.SolCount > 0:
            has_solution = True
    except Exception:
        has_solution = False

    if has_solution:
        try:
            obj_val = float(model.ObjVal)
        except Exception:
            obj_val = 1e100

        yP_new_val = y_P_new.X if N > 0 else np.asarray([], dtype=float)
        yP_ret_val = y_P_ret.X if N > 0 else np.asarray([], dtype=float)

        if S > 0:
            yE_new_val = y_E_new.X
            yE_ret_val = y_E_ret.X
        else:
            yE_new_val = np.asarray([], dtype=float)
            yE_ret_val = np.asarray([], dtype=float)

        if L > 0:
            yF_new_val = y_F_new.X
        else:
            yF_new_val = np.asarray([], dtype=float)

        x_inj_val = x_inj.X
        x_wdw_val = x_wdw.X if S > 0 else np.zeros((0, total_hours), dtype=float)
        x_soc_val = x_soc.X if S > 0 else np.zeros((0, total_hours), dtype=float)
        x_lvl_val = x_lvl.X if HY > 0 else np.zeros((0, total_hours), dtype=float)
        x_spill_val = x_spill.X if HY > 0 else np.zeros((0, total_hours), dtype=float)
        x_commit_val = x_commit.X if U > 0 else np.zeros((0, total_hours), dtype=float)
        x_start_val = x_start.X if U > 0 else np.zeros((0, total_hours), dtype=float)
        x_shut_val = x_shut.X if U > 0 else np.zeros((0, total_hours), dtype=float)
        x_nse_val = x_nse.X if G > 0 else np.zeros((0, total_hours), dtype=float)
        x_flow_val = x_flow.X if L > 0 else np.zeros((0, total_hours), dtype=float)

        if x_RPS is not None:
            x_RPS_val = np.asarray(x_RPS.X, dtype=float).copy()
        else:
            x_RPS_val = None

        if x_CO2 is not None:
            x_CO2_val = np.asarray(x_CO2.X, dtype=float).copy()
        else:
            x_CO2_val = None

    else:
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

    new_units = np.asarray([clean_int(v) for v in yP_new_val], dtype=float)
    ret_units = np.asarray([clean_int(v) for v in yP_ret_val], dtype=float)
    total_cap = existing_power + p_inc * new_units - p_inc * ret_units
    total_cap = np.maximum(total_cap, 0.0)

    if S > 0:
        new_e_units = np.asarray([clean_int(v) for v in yE_new_val], dtype=float)
        ret_e_units = np.asarray([clean_int(v) for v in yE_ret_val], dtype=float)
        total_e = existing_energy_storage + e_inc_storage * new_e_units - e_inc_storage * ret_e_units
        total_e = np.maximum(total_e, 0.0)
    else:
        new_e_units = np.asarray([], dtype=float)
        ret_e_units = np.asarray([], dtype=float)
        total_e = np.asarray([], dtype=float)

    if L > 0:
        new_line = np.asarray([clean_int(v) for v in yF_new_val], dtype=float)
        total_line = existing_line + new_line
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
            "resource_type": gen_rtype[i],
            "zone": gen_zone_orig[i],
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
        if vals is None or vals.size == 0:
            return
        rows, cols = np.nonzero(np.abs(vals) > OP_TOL)
        for r, c in zip(rows, cols):
            model_variables[f"{name}[{row_ids[r]},{int(c)}]"] = clean_float(vals[r, c])

    add_2d(x_inj_val, cluster_ids, "x_inj")

    if S > 0:
        storage_ids = [cluster_ids[i] for i in storage_indices]
        add_2d(x_wdw_val, storage_ids, "x_wdw")
        add_2d(x_soc_val, storage_ids, "x_soc")

    if HY > 0:
        hydro_ids = [cluster_ids[i] for i in hydro_indices]
        add_2d(x_lvl_val, hydro_ids, "x_lvl")
        add_2d(x_spill_val, hydro_ids, "x_spill")

    if U > 0:
        uc_ids = [cluster_ids[i] for i in uc_indices]
        add_2d(x_commit_val, uc_ids, "x_commit")
        add_2d(x_start_val, uc_ids, "x_start")
        add_2d(x_shut_val, uc_ids, "x_shut")

    if L > 0:
        add_2d(x_flow_val, line_ids, "x_flow")

    if G > 0 and x_nse_val.size > 0:
        rows, cols = np.nonzero(np.abs(x_nse_val) > OP_TOL)
        for r, c in zip(rows, cols):
            model_variables[f"x_nse[{seg_ids[r]},{seg_zone_orig[r]},{int(c)}]"] = clean_float(x_nse_val[r, c])

    if x_RPS_val is not None:
        total_slack = float(np.sum(x_RPS_val))
        if abs(total_slack) > OP_TOL:
            x_RPS_val = np.zeros_like(x_RPS_val)
            x_RPS_val[0] = total_slack
        for w, v in enumerate(x_RPS_val):
            if abs(v) > OP_TOL:
                model_variables[f"x_RPS[{w}]"] = clean_float(v)

    if x_CO2_val is not None:
        total_slack = float(np.sum(x_CO2_val))
        if abs(total_slack) > OP_TOL:
            x_CO2_val = np.zeros_like(x_CO2_val)
            x_CO2_val[0] = total_slack
        for w, v in enumerate(x_CO2_val):
            if abs(v) > OP_TOL:
                model_variables[f"x_CO2[{w}]"] = clean_float(v)

    solution = {
        "objective_value": clean_float(obj_val),
        "model_variables": model_variables,
        "investments": investments,
        "storage_energy_investments": storage_energy_investments,
        "transmission_investments": transmission_investments,
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
            pass


if __name__ == "__main__":
    main()
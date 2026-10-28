#!/usr/bin/env python3
"""
Joint long-term capacity expansion and hourly operational dispatch MILP
(GenX-style multi-zone planning model) solved with Gurobi.

Reads an instance JSON (instance_schema.json), builds the MILP described in the
problem statement, solves it under a wall-clock time limit, and writes a
solution JSON following solution_schema.json.
"""

import argparse
import json
import time
from collections import defaultdict

import gurobipy as gp
from gurobipy import GRB

MODEL_VARIABLE_NAME_PATTERNS = {
    "y_P_new[i], y_P_ret[i], y_P[i]": "generator capacity variables",
    "y_E_new[i], y_E_ret[i], y_E[i]": "storage-energy variables",
    "y_F_new[l], y_F[l]": "transmission variables",
    "x_inj[i,t], x_wdw[i,t], x_soc[i,t]": "generation/storage operations",
    "x_nse[s,z,t]": "non-served energy",
    "x_lvl[i,t], x_spill[i,t]": "hydro operations",
    "x_flow[l,t]": "transmission flow",
    "x_commit[i,t], x_start[i,t], x_shut[i,t]": "unit commitment",
    "x_RPS[w], x_CO2[w]": "policy slack variables",
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", required=True)
    ap.add_argument("--solution_path", required=True)
    ap.add_argument("--time_limit", type=int, required=True)
    ap.add_argument("--log_path", default=None)
    return ap.parse_args()


def fval(d, key, default=0.0):
    v = d.get(key, None)
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def int_keyed(d):
    out = {}
    for k, v in (d or {}).items():
        try:
            out[int(k)] = v
        except (TypeError, ValueError):
            out[k] = v
    return out


def hours_back(t, length, H):
    """Indices of the `length` hours ending at t, wrapping within the subperiod."""
    base = (t // H) * H
    off = t - base
    return [base + ((off - k) % H) for k in range(length)]


def main():
    t_start = time.time()
    args = parse_args()

    logger = None
    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize")

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    # ---------------- instance data ----------------
    pp = inst.get("problem_parameters", {}) or {}
    weights = [float(x) for x in inst.get("subperiod_weights", [])]
    W = int(pp.get("num_weeks") or len(weights) or 1)
    H = int(pp.get("hours_per_subperiod") or 168)
    T = int(pp.get("total_hours") or (W * H))
    if not weights:
        weights = [1.0] * W
    wt = [weights[t // H] / float(H) for t in range(T)]
    rngT = range(T)

    gens = inst.get("generators", [])
    gparams = inst.get("generator_parameters", {})
    by_id = {g["cluster_id"]: g for g in gens}
    ids = [g["cluster_id"] for g in gens]

    zone_of = {i: by_id[i]["zone"] for i in ids}
    rtype = {i: by_id[i]["resource_type"] for i in ids}
    is_UC = {i: bool(by_id[i].get("is_UC", False)) for i in ids}
    is_storage = {i: bool(by_id[i].get("is_storage", False)) for i in ids}
    is_hydro = {i: bool(by_id[i].get("is_hydro", False)) for i in ids}
    is_RPS = {i: bool(by_id[i].get("is_RPS_qualifying", False)) for i in ids}
    can_retire = {i: bool(by_id[i].get("can_retire", False)) for i in ids}
    params = {i: (gparams.get(rtype[i], {}) or {}) for i in ids}

    existing = int_keyed(inst.get("existing_capacity", {}))
    existing = {i: float(existing.get(i, 0.0)) for i in ids}

    availability = int_keyed(inst.get("availability", {}))
    avail = {}
    for i in ids:
        arr = availability.get(i, None)
        if arr is None:
            arr = [1.0] * T
        avail[i] = [float(x) for x in arr]
    hydro_in_raw = int_keyed(inst.get("hydro_inflow", {}))

    demand = {z: [float(x) for x in v] for z, v in int_keyed(inst.get("demand", {})).items()}

    segs = inst.get("consumer_segments", [])
    seg_ids = [s["segment_id"] for s in segs]
    seg_zone = {s["segment_id"]: s["zone"] for s in segs}
    seg_frac = {s["segment_id"]: float(s.get("max_nse_fraction", 0.0)) for s in segs}
    seg_cost = {s["segment_id"]: float(s.get("cost_per_MWh", 0.0)) for s in segs}

    lines = inst.get("transmission_lines", [])
    line_ids = [l["line_id"] for l in lines]
    L_from = {l["line_id"]: l["from_zone"] for l in lines}
    L_to = {l["line_id"]: l["to_zone"] for l in lines}
    L_exist = {l["line_id"]: float(l.get("existing_capacity_MW", 0.0)) for l in lines}
    L_max = {l["line_id"]: float(l.get("max_capacity_MW", L_exist[l["line_id"]])) for l in lines}
    L_cost = {l["line_id"]: float(l.get("investment_cost_per_MW_yr", 0.0)) for l in lines}

    zones = sorted(set(demand.keys()) | set(zone_of.values()) | set(seg_zone.values())
                   | set(L_from.values()) | set(L_to.values()))
    for z in zones:
        if z not in demand:
            demand[z] = [0.0] * T

    BIG = 1e12
    block = {}
    for i in ids:
        if is_UC[i]:
            b = fval(by_id[i], "capacity_size_MW", fval(params[i], "capacity_size_MW", 1.0))
            block[i] = b if b > 0 else 1.0
        else:
            block[i] = 1.0

    storage_ids = [i for i in ids if is_storage[i]]
    hydro_ids = [i for i in ids if is_hydro[i]]
    uc_ids = [i for i in ids if is_UC[i] and not is_storage[i] and not is_hydro[i]]
    uc_set = set(uc_ids)
    nonuc_ids = [i for i in ids if i not in uc_set]

    max_new = {}
    for i in ids:
        v = params[i].get("max_capacity_MW", None)
        max_new[i] = BIG if v is None or float(v) < 0 else float(v)
    capmax = {i: existing[i] + max_new[i] for i in ids}

    blockE, max_newE, min_dur, max_dur = {}, {}, {}, {}
    for i in storage_ids:
        blockE[i] = (fval(params[i], "energy_capacity_size_MWh", 1.0) or 1.0) if is_UC[i] else 1.0
        v = params[i].get("max_energy_capacity_MWh", None)
        max_newE[i] = BIG if v is None or float(v) < 0 else float(v)
        min_dur[i] = fval(params[i], "min_duration_MWh_per_MW", 0.0)
        md = params[i].get("max_duration_MWh_per_MW", None)
        max_dur[i] = BIG if md is None else float(md)

    vc = {i: fval(params[i], "variable_cost_per_MWh", 0.0) for i in ids}
    fom = {i: fval(params[i], "fixed_OM_cost_per_MW_yr", 0.0) for i in ids}
    invc = {i: fval(params[i], "investment_cost_per_MW_yr", 0.0) for i in ids}
    EF = {i: fval(params[i], "CO2_emission_tons_per_MWh", 0.0) for i in ids}
    suc = {i: fval(params[i], "start_up_cost", 0.0) for i in uc_ids}
    einv = {i: fval(params[i], "energy_investment_cost_per_MWh_yr", 0.0) for i in storage_ids}
    efom = {i: fval(params[i], "energy_fixed_OM_cost_per_MWh_yr", 0.0) for i in storage_ids}
    hdur = {i: fval(params[i], "duration_MWh_per_MW", 0.0) for i in hydro_ids}
    heinv = {i: fval(params[i], "energy_investment_cost_per_MWh_yr", 0.0) for i in hydro_ids}
    hefom = {i: fval(params[i], "energy_fixed_OM_cost_per_MWh_yr", 0.0) for i in hydro_ids}

    hydro_in = {}
    for i in hydro_ids:
        arr = hydro_in_raw.get(i, [0.0] * T)
        hydro_in[i] = [float(x) for x in arr]

    # ---------------- solution assembly helpers ----------------
    def make_solution(name2val, objval):
        mv = {n: float(v) for n, v in name2val.items() if abs(v) > 1e-6}
        investments = {}
        for i in ids:
            nu = float(int(round(name2val.get("y_P_new[{}]".format(i), 0.0))))
            ru = float(int(round(name2val.get("y_P_ret[{}]".format(i), 0.0))))
            tot = float(name2val.get("y_P[{}]".format(i), existing[i]))
            investments[str(i)] = {
                "resource_type": rtype[i],
                "zone": zone_of[i],
                "new_units": nu,
                "new_MW": nu * block[i],
                "retired_units": ru,
                "retired_MW": ru * block[i],
                "total_capacity_MW": tot,
            }
        sei = {}
        for i in storage_ids:
            nu = float(int(round(name2val.get("y_E_new[{}]".format(i), 0.0))))
            sei[str(i)] = {
                "new_units": nu,
                "new_MWh": nu * blockE[i],
                "total_energy_MWh": float(name2val.get("y_E[{}]".format(i), 0.0)),
            }
        ti = {}
        for l in line_ids:
            nm = float(int(round(name2val.get("y_F_new[{}]".format(l), 0.0))))
            ti[str(l)] = {
                "new_MW": nm,
                "total_capacity_MW": float(name2val.get("y_F[{}]".format(l), L_exist[l])),
            }
        return {
            "objective_value": float(objval),
            "model_variables": mv,
            "investments": investments,
            "storage_energy_investments": sei,
            "transmission_investments": ti,
            "model_variable_name_patterns": MODEL_VARIABLE_NAME_PATTERNS,
        }

    def fallback_solution():
        investments = {}
        for i in ids:
            investments[str(i)] = {
                "resource_type": rtype[i],
                "zone": zone_of[i],
                "new_units": 0.0,
                "new_MW": 0.0,
                "retired_units": 0.0,
                "retired_MW": 0.0,
                "total_capacity_MW": float(existing[i]),
            }
        sei = {str(i): {"new_units": 0.0, "new_MWh": 0.0, "total_energy_MWh": 0.0}
               for i in storage_ids}
        ti = {str(l): {"new_MW": 0.0, "total_capacity_MW": float(L_exist[l])}
              for l in line_ids}
        return {
            "objective_value": 1e18,
            "model_variables": {},
            "investments": investments,
            "storage_energy_investments": sei,
            "transmission_investments": ti,
            "model_variable_name_patterns": MODEL_VARIABLE_NAME_PATTERNS,
        }

    sol = None
    try:
        m = gp.Model("joint_capacity_expansion_dispatch")
        m.Params.Seed = 0
        m.Params.MIPGap = 1e-4
        m.Params.NumericFocus = 0
        m.Params.Threads = 1
        m.Params.LogToConsole = 0

        # ---------------- investment variables ----------------
        yP_new = m.addVars(ids, lb=0.0, ub={i: max_new[i] / block[i] for i in ids},
                           vtype=GRB.INTEGER, name="y_P_new")
        yP_ret = m.addVars(ids, lb=0.0,
                           ub={i: (existing[i] / block[i] if can_retire[i] else 0.0) for i in ids},
                           vtype=GRB.INTEGER, name="y_P_ret")
        yP = m.addVars(ids, lb=0.0, ub={i: capmax[i] for i in ids}, name="y_P")
        m.addConstrs((yP[i] == existing[i] + block[i] * (yP_new[i] - yP_ret[i]) for i in ids),
                     name="c_cap")

        ztype = defaultdict(list)
        for i in ids:
            ztype[(zone_of[i], rtype[i])].append(i)
        for (z, rt), mem in ztype.items():
            m.addConstr(gp.quicksum(block[i] * yP_new[i] for i in mem) <= max_new[mem[0]],
                        name="c_maxnew[{},{}]".format(z, rt))

        st_idx = [(i, t) for i in storage_ids for t in rngT]
        yE_new = m.addVars(storage_ids, lb=0.0,
                           ub={i: max_newE[i] / blockE[i] for i in storage_ids},
                           vtype=GRB.INTEGER, name="y_E_new")
        yE_ret = m.addVars(storage_ids, lb=0.0, ub=0.0, vtype=GRB.INTEGER, name="y_E_ret")
        socmax = {i: min(max_newE[i], max_dur[i] * capmax[i]) for i in storage_ids}
        yE = m.addVars(storage_ids, lb=0.0, ub=socmax, name="y_E")
        if storage_ids:
            m.addConstrs((yE[i] == blockE[i] * (yE_new[i] - yE_ret[i]) for i in storage_ids),
                         name="c_ecap")
            m.addConstrs((yE[i] >= min_dur[i] * yP[i] for i in storage_ids), name="c_emin")
            m.addConstrs((yE[i] <= max_dur[i] * yP[i] for i in storage_ids), name="c_emax")
            zstor = defaultdict(list)
            for i in storage_ids:
                zstor[zone_of[i]].append(i)
            for z, mem in zstor.items():
                m.addConstr(gp.quicksum(blockE[i] * yE_new[i] for i in mem) <= max_newE[mem[0]],
                            name="c_maxnewE[{}]".format(z))

        yF_new = m.addVars(line_ids, lb=0.0,
                           ub={l: max(0.0, L_max[l] - L_exist[l]) for l in line_ids},
                           vtype=GRB.INTEGER, name="y_F_new")
        yF = m.addVars(line_ids, lb=0.0, ub={l: L_max[l] for l in line_ids}, name="y_F")
        m.addConstrs((yF[l] == L_exist[l] + yF_new[l] for l in line_ids), name="c_fcap")

        # ---------------- operational variables ----------------
        prev = [(t - 1) if (t % H) else (t + H - 1) for t in rngT]

        it_idx = [(i, t) for i in ids for t in rngT]
        x_inj = m.addVars(it_idx, lb=0.0,
                          ub={(i, t): avail[i][t] * capmax[i] for (i, t) in it_idx},
                          name="x_inj")

        x_wdw = m.addVars(st_idx, lb=0.0,
                          ub={(i, t): avail[i][t] * capmax[i] for (i, t) in st_idx},
                          name="x_wdw")
        x_soc = m.addVars(st_idx, lb=0.0,
                          ub={(i, t): socmax[i] for (i, t) in st_idx}, name="x_soc")

        hy_idx = [(i, t) for i in hydro_ids for t in rngT]
        x_lvl = m.addVars(hy_idx, lb=0.0,
                          ub={(i, t): hdur[i] * capmax[i] for (i, t) in hy_idx}, name="x_lvl")
        x_spill = m.addVars(hy_idx, lb=0.0, name="x_spill")

        uc_idx = [(i, t) for i in uc_ids for t in rngT]
        u_ub = {(i, t): capmax[i] / block[i] for (i, t) in uc_idx}
        x_commit = m.addVars(uc_idx, lb=0.0, ub=u_ub, name="x_commit")
        x_start = m.addVars(uc_idx, lb=0.0, ub=u_ub, name="x_start")
        x_shut = m.addVars(uc_idx, lb=0.0, ub=u_ub, name="x_shut")

        lt_idx = [(l, t) for l in line_ids for t in rngT]
        x_flow = m.addVars(lt_idx,
                           lb={(l, t): -L_max[l] for (l, t) in lt_idx},
                           ub={(l, t): L_max[l] for (l, t) in lt_idx}, name="x_flow")

        szt_idx = [(s, seg_zone[s], t) for s in seg_ids for t in rngT]
        x_nse = m.addVars(szt_idx, lb=0.0,
                          ub={(s, z, t): seg_frac[s] * demand[z][t] for (s, z, t) in szt_idx},
                          name="x_nse")

        # ---------------- non-UC resource constraints ----------------
        for i in nonuc_ids:
            a = avail[i]
            m.addConstrs((x_inj[i, t] <= a[t] * yP[i] for t in rngT), name="c_av[{}]".format(i))
            if is_storage[i] or is_hydro[i]:
                continue
            p = params[i]
            mo = fval(p, "min_output_frac", 0.0)
            if mo > 0:
                m.addConstrs((x_inj[i, t] >= mo * yP[i] for t in rngT), name="c_mo[{}]".format(i))
            ru = p.get("ramp_up_rate_pct_per_hr", None)
            rd = p.get("ramp_down_rate_pct_per_hr", None)
            if ru is not None and rd is not None:
                ru = float(ru)
                rd = float(rd)
                m.addConstrs((x_inj[i, t] - x_inj[i, prev[t]] <= ru * yP[i] for t in rngT),
                             name="c_ru[{}]".format(i))
                m.addConstrs((x_inj[i, prev[t]] - x_inj[i, t] <= rd * yP[i] for t in rngT),
                             name="c_rd[{}]".format(i))

        # ---------------- unit-commitment constraints ----------------
        for i in uc_ids:
            p = params[i]
            size = block[i]
            mo = fval(p, "min_output_frac", 0.0)
            ru = fval(p, "ramp_up_rate_pct_per_hr", 1.0)
            rd = fval(p, "ramp_down_rate_pct_per_hr", 1.0)
            a = avail[i]
            m.addConstrs((size * x_commit[i, t] <= yP[i] for t in rngT), name="c_cc[{}]".format(i))
            m.addConstrs((size * x_start[i, t] <= yP[i] for t in rngT), name="c_cs[{}]".format(i))
            m.addConstrs((size * x_shut[i, t] <= yP[i] for t in rngT), name="c_ch[{}]".format(i))
            m.addConstrs((x_inj[i, t] <= a[t] * size * x_commit[i, t] for t in rngT),
                         name="c_gx[{}]".format(i))
            m.addConstrs((x_inj[i, t] >= mo * size * x_commit[i, t] for t in rngT),
                         name="c_gn[{}]".format(i))
            m.addConstrs((x_commit[i, t] - x_commit[i, prev[t]] == x_start[i, t] - x_shut[i, t]
                          for t in rngT), name="c_cb[{}]".format(i))
            m.addConstrs((x_inj[i, t] - x_inj[i, prev[t]]
                          <= ru * size * (x_commit[i, t] - x_start[i, t])
                          + size * min(a[t], max(mo, ru)) * x_start[i, t]
                          - size * mo * x_shut[i, t] for t in rngT), name="c_ru[{}]".format(i))
            m.addConstrs((x_inj[i, prev[t]] - x_inj[i, t]
                          <= rd * size * (x_commit[i, t] - x_start[i, t])
                          - size * mo * x_start[i, t]
                          + size * min(a[t], max(mo, rd)) * x_shut[i, t] for t in rngT),
                         name="c_rd[{}]".format(i))
            Lup = int(fval(p, "min_up_time_hr", 0.0))
            Ldn = int(fval(p, "min_down_time_hr", 0.0))
            if Lup > 0:
                m.addConstrs((x_commit[i, t] >= gp.quicksum(x_start[i, tt]
                                                            for tt in hours_back(t, Lup, H))
                              for t in rngT), name="c_mu[{}]".format(i))
            if Ldn > 0:
                m.addConstrs((x_commit[i, t] + gp.quicksum(x_shut[i, tt]
                                                           for tt in hours_back(t, Ldn, H))
                              <= yP[i] / size for t in rngT), name="c_md[{}]".format(i))

        # ---------------- storage constraints ----------------
        for i in storage_ids:
            p = params[i]
            effc = fval(p, "charging_efficiency", 1.0) or 1.0
            effd = fval(p, "discharging_efficiency", 1.0) or 1.0
            sd = fval(p, "self_discharge_rate_pct", 0.0)
            a = avail[i]
            m.addConstrs((x_soc[i, t] == (1.0 - sd) * x_soc[i, prev[t]]
                          + effc * x_wdw[i, t] - x_inj[i, t] / effd for t in rngT),
                         name="c_sb[{}]".format(i))
            m.addConstrs((x_soc[i, t] <= yE[i] for t in rngT), name="c_sc[{}]".format(i))
            m.addConstrs((effc * x_wdw[i, t] <= yE[i] - x_soc[i, prev[t]] for t in rngT),
                         name="c_sr[{}]".format(i))
            m.addConstrs((x_inj[i, t] <= effd * x_soc[i, prev[t]] for t in rngT),
                         name="c_sd[{}]".format(i))
            m.addConstrs((x_wdw[i, t] <= a[t] * yP[i] for t in rngT), name="c_wc[{}]".format(i))
            m.addConstrs((x_inj[i, t] + x_wdw[i, t] <= yP[i] for t in rngT),
                         name="c_pc[{}]".format(i))

        # ---------------- hydro constraints ----------------
        for i in hydro_ids:
            p = params[i]
            mo = fval(p, "min_output_frac", 0.0)
            infl = hydro_in[i]
            m.addConstrs((x_lvl[i, t] == x_lvl[i, prev[t]] + infl[t] * yP[i]
                          - x_inj[i, t] - x_spill[i, t] for t in rngT),
                         name="c_hb[{}]".format(i))
            m.addConstrs((x_lvl[i, t] <= hdur[i] * yP[i] for t in rngT),
                         name="c_hc[{}]".format(i))
            if mo > 0:
                m.addConstrs((x_inj[i, t] + x_spill[i, t] >= mo * yP[i] for t in rngT),
                             name="c_hm[{}]".format(i))

        # ---------------- transmission constraints ----------------
        for l in line_ids:
            m.addConstrs((x_flow[l, t] <= yF[l] for t in rngT), name="c_fu[{}]".format(l))
            m.addConstrs((x_flow[l, t] >= -yF[l] for t in rngT), name="c_fl[{}]".format(l))

        # ---------------- zonal power balance ----------------
        zone_gens = {z: [] for z in zones}
        for i in ids:
            zone_gens[zone_of[i]].append(i)
        lines_to = {z: [] for z in zones}
        lines_fr = {z: [] for z in zones}
        for l in line_ids:
            lines_to[L_to[l]].append(l)
            lines_fr[L_from[l]].append(l)
        segs_z = {z: [] for z in zones}
        for s in seg_ids:
            segs_z[seg_zone[s]].append(s)

        for z in zones:
            gz = zone_gens[z]
            stz = [i for i in gz if is_storage[i]]
            lt_ = lines_to[z]
            lf_ = lines_fr[z]
            sz = segs_z[z]
            dz = demand[z]
            m.addConstrs((gp.quicksum(x_inj[i, t] for i in gz)
                          - gp.quicksum(x_wdw[i, t] for i in stz)
                          + gp.quicksum(x_flow[l, t] for l in lt_)
                          - gp.quicksum(x_flow[l, t] for l in lf_)
                          + gp.quicksum(x_nse[s, z, t] for s in sz)
                          == dz[t] for t in rngT), name="c_bal[{}]".format(z))

        # ---------------- policy constraints ----------------
        policy = inst.get("policy", {}) or {}
        ptype = str(policy.get("type", "REF")).upper()
        total_dem = sum(wt[t] * demand[z][t] for z in zones for t in rngT)

        pen_expr = 0.0
        if ptype == "RPS":
            x_RPS = m.addVars(range(W), lb=0.0, name="x_RPS")
            share = float(policy.get("RPS_share", 0.0))
            qual = gp.quicksum(wt[t] * x_inj[i, t] for i in ids if is_RPS[i] for t in rngT)
            m.addConstr(qual + gp.quicksum(x_RPS[w] for w in range(W)) >= share * total_dem,
                        name="c_rps")
            pen_expr = pen_expr + float(policy.get("RPS_noncompliance_cost_per_MWh", 0.0)) * \
                gp.quicksum(x_RPS[w] for w in range(W))
        elif ptype == "CO2":
            x_CO2 = m.addVars(range(W), lb=0.0, name="x_CO2")
            cap_rate = float(policy.get("CO2_cap_tons_per_MWh", 0.0))
            emis = (gp.quicksum(wt[t] * EF[i] * x_inj[i, t] for i in ids for t in rngT)
                    + gp.quicksum(wt[t] * EF[i] * x_wdw[i, t] for i in storage_ids for t in rngT))
            m.addConstr(emis - gp.quicksum(x_CO2[w] for w in range(W)) <= cap_rate * total_dem,
                        name="c_co2")
            pen_expr = pen_expr + float(policy.get("CO2_noncompliance_cost_per_ton", 0.0)) * \
                gp.quicksum(x_CO2[w] for w in range(W))

        # ---------------- objective ----------------
        fixed = gp.quicksum(invc[i] * block[i] * yP_new[i] + fom[i] * yP[i] for i in ids)
        fixed += gp.quicksum(einv[i] * blockE[i] * yE_new[i] + efom[i] * yE[i]
                             for i in storage_ids)
        fixed += gp.quicksum(heinv[i] * hdur[i] * yP_new[i] + hefom[i] * hdur[i] * yP[i]
                             for i in hydro_ids)
        fixed += gp.quicksum(L_cost[l] * yF_new[l] for l in line_ids)

        varc = gp.quicksum(wt[t] * vc[i] * x_inj[i, t] for i in ids for t in rngT)
        varc += gp.quicksum(wt[t] * vc[i] * x_wdw[i, t] for i in storage_ids for t in rngT)
        startc = gp.quicksum(wt[t] * suc[i] * x_start[i, t] for i in uc_ids for t in rngT)
        nsec = gp.quicksum(wt[t] * seg_cost[s] * x_nse[s, seg_zone[s], t]
                           for s in seg_ids for t in rngT)

        m.setObjective(fixed + varc + startc + nsec + pen_expr, GRB.MINIMIZE)

        # ---------------- solve ----------------
        remaining = float(args.time_limit) - (time.time() - t_start)
        m.Params.TimeLimit = max(1.0, remaining - 1.0)

        varlist = m.getVars()
        varnames = [v.VarName for v in varlist]
        state = {"last": float("inf")}

        def callback(model, where):
            if where != GRB.Callback.MIPSOL or logger is None:
                return
            try:
                objv = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if objv < state["last"] - 1e-6:
                    state["last"] = objv
                    vals = model.cbGetSolution(varlist)
                    logger.log_solution(objv, make_solution(dict(zip(varnames, vals)), objv))
            except Exception:
                pass

        m.optimize(callback)

        if m.SolCount > 0:
            vals = [v.X for v in varlist]
            objv = float(m.ObjVal)
            sol = make_solution(dict(zip(varnames, vals)), objv)
            if logger is not None and objv < state["last"] - 1e-6:
                state["last"] = objv
                logger.log_solution(objv, sol)
    except Exception:
        sol = None

    if sol is None:
        sol = fallback_solution()

    with open(args.solution_path, "w") as f:
        json.dump(sol, f)


if __name__ == "__main__":
    main()
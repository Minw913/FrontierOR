import argparse
import json
import math
import os
import time

import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except Exception:  # pragma: no cover
    SolutionLogger = None


PATTERNS = {
    "yPnew_{i}, yPret_{i}, yP_{i}": "generator investment/retirement/total capacity",
    "yEnew_{i}, yEret_{i}, yE_{i}": "storage-energy investment variables",
    "yFnew_{l}": "transmission investment",
    "xinj_{i}_{t}, xwdw_{i}_{t}, xsoc_{i}_{t}, xhydro_{i}_{t}": "operations",
    "xflow_{l}_{t}": "transmission flow",
    "xnse_{s}_{z}_{t}": "non-served energy",
    "xcommit_{i}_{t}, xstart_{i}_{t}, xshut_{i}_{t}": "unit commitment",
    "xRPS_{w}, xCO2_{w}": "policy slack variables",
}

INFLOW_KEYS = (
    "inflow_profile", "inflows", "inflow", "inflow_frac",
    "inflow_fraction", "hydro_inflow", "inflow_profile_frac",
)


def fnum(x, default=0.0):
    if x is None:
        return default
    return float(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = None
    if args.log_path and SolutionLogger is not None:
        logger = SolutionLogger(args.log_path, sense="minimize")

    t0 = time.time()

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    dims = inst.get("problem_dimensions", {})
    sp = inst.get("subperiods", {})
    W = int(sp.get("num_subperiods", dims.get("num_weeks", 1)))
    H = int(sp.get("hours_per_subperiod", dims.get("hours_per_week", 168)))
    weights = list(sp.get("subperiod_weights", [1.0] * W))
    if len(weights) < W:
        weights = weights + [1.0] * (W - len(weights))
    circular = bool(sp.get("circular_indexing", True))
    T = W * H
    hw = [weights[min(t // H, W - 1)] / float(H) for t in range(T)]

    gens = inst.get("generators", [])
    Gids = [g["id"] for g in gens]
    lines = inst.get("transmission_lines", [])
    Lids = [l["id"] for l in lines]

    demand = {int(z): [float(x) for x in v] for z, v in inst.get("demand", {}).items()}
    zones = sorted(demand.keys())
    nse_segments = inst.get("nse_segments", [])
    seg_ids = [s["segment"] for s in nse_segments]
    avail_profiles = inst.get("availability_profiles", {})

    def avail_for(g):
        ap = avail_profiles.get(str(g["id"]))
        if ap is None:
            ap = avail_profiles.get(g["id"])
        if not isinstance(ap, dict):
            return [1.0] * T
        if ap.get("type") == "timeseries" and ap.get("values") is not None:
            return [float(x) for x in ap["values"]]
        return [fnum(ap.get("value"), 1.0)] * T

    avail = {g["id"]: avail_for(g) for g in gens}

    # Hydro inflow profile (fallback: availability profile).
    inflow = {}
    for g in gens:
        if not g.get("is_hydro"):
            continue
        src = None
        for k in INFLOW_KEYS:
            if g.get(k) is not None:
                src = g[k]
                break
        if src is None:
            ap = avail_profiles.get(str(g["id"]), {})
            if isinstance(ap, dict):
                for k in INFLOW_KEYS:
                    if ap.get(k) is not None:
                        src = ap[k]
                        break
        if src is None:
            src = avail[g["id"]]
        if isinstance(src, (int, float)):
            src = [float(src)] * T
        inflow[g["id"]] = [float(x) for x in src]

    policy = inst.get("policy", {}) or {}
    rps_on = bool(policy.get("RPS_enabled", False))
    co2_on = bool(policy.get("CO2_cap_enabled", False))

    storage_ids = [g["id"] for g in gens if g.get("is_storage")]
    hydro_ids = [g["id"] for g in gens if g.get("is_hydro")]
    uc_ids = [g["id"] for g in gens if g.get("is_UC")]

    # ---------------- Build model ----------------
    m = gp.Model("capacity_expansion")
    m.Params.Seed = 0
    m.Params.MIPGap = 1e-4
    m.Params.NumericFocus = 0
    m.Params.Threads = 1
    m.Params.LogToConsole = 0

    out_vars = []

    # Investment / retirement variables
    yPnew, yPret, yP = {}, {}, {}
    yEnew, yEret, yE = {}, {}, {}
    for g in gens:
        i = g["id"]
        size = fnum(g.get("capacity_size_MW"), 1.0) or 1.0
        exist = fnum(g.get("existing_capacity_MW"), 0.0)
        maxcap = g.get("max_capacity_MW")
        ub_new = GRB.INFINITY if maxcap is None else max(0.0, math.floor(float(maxcap) / size + 1e-6))
        ub_ret = max(0.0, math.floor(exist / size + 1e-6)) if g.get("can_retire", True) else 0.0
        yPnew[i] = m.addVar(lb=0.0, ub=ub_new, vtype=GRB.INTEGER, name=f"yPnew_{i}")
        yPret[i] = m.addVar(lb=0.0, ub=ub_ret, vtype=GRB.INTEGER, name=f"yPret_{i}")
        yP[i] = m.addVar(lb=0.0, name=f"yP_{i}")
        m.addConstr(yP[i] == exist + size * (yPnew[i] - yPret[i]), name=f"c_cap_{i}")
        out_vars += [(f"yPnew_{i}", yPnew[i]), (f"yPret_{i}", yPret[i]), (f"yP_{i}", yP[i])]
        if g.get("is_storage"):
            esz = fnum(g.get("storage_capacity_size_MWh"), 1.0) or 1.0
            eexist = fnum(g.get("existing_storage_capacity_MWh"), 0.0)
            emax = g.get("max_storage_capacity_MWh")
            eub_new = GRB.INFINITY if emax is None else max(0.0, math.floor(float(emax) / esz + 1e-6))
            eub_ret = max(0.0, math.floor(eexist / esz + 1e-6)) if g.get("can_retire", True) else 0.0
            yEnew[i] = m.addVar(lb=0.0, ub=eub_new, vtype=GRB.INTEGER, name=f"yEnew_{i}")
            yEret[i] = m.addVar(lb=0.0, ub=eub_ret, vtype=GRB.INTEGER, name=f"yEret_{i}")
            yE[i] = m.addVar(lb=0.0, name=f"yE_{i}")
            m.addConstr(yE[i] == eexist + esz * (yEnew[i] - yEret[i]), name=f"c_ecap_{i}")
            mind = fnum(g.get("min_duration_MWh_per_MW"), 0.0)
            maxd = g.get("max_duration_MWh_per_MW")
            if mind > 0:
                m.addConstr(yE[i] >= mind * yP[i], name=f"c_durlo_{i}")
            if maxd is not None:
                m.addConstr(yE[i] <= float(maxd) * yP[i], name=f"c_durhi_{i}")
            out_vars += [(f"yEnew_{i}", yEnew[i]), (f"yEret_{i}", yEret[i]), (f"yE_{i}", yE[i])]

    yFnew = {}
    for l in lines:
        lid = l["id"]
        mx = fnum(l.get("max_new_capacity_MW"), 0.0)
        yFnew[lid] = m.addVar(lb=0.0, ub=mx, vtype=GRB.INTEGER, name=f"yFnew_{lid}")
        out_vars.append((f"yFnew_{lid}", yFnew[lid]))

    # Operational variables
    xinj = m.addVars(Gids, range(T), lb=0.0, name="xinj")
    out_vars += [(f"xinj_{i}_{t}", xinj[i, t]) for i in Gids for t in range(T)]

    xwdw = m.addVars(storage_ids, range(T), lb=0.0, name="xwdw")
    xsoc = m.addVars(storage_ids, range(T), lb=0.0, name="xsoc")
    out_vars += [(f"xwdw_{i}_{t}", xwdw[i, t]) for i in storage_ids for t in range(T)]
    out_vars += [(f"xsoc_{i}_{t}", xsoc[i, t]) for i in storage_ids for t in range(T)]

    xhydro = m.addVars(hydro_ids, range(T), lb=0.0, name="xhydro")
    spill = m.addVars(hydro_ids, range(T), lb=0.0, name="xspill")  # internal only
    out_vars += [(f"xhydro_{i}_{t}", xhydro[i, t]) for i in hydro_ids for t in range(T)]

    xflow = m.addVars(Lids, range(T), lb=-GRB.INFINITY, name="xflow")
    out_vars += [(f"xflow_{l}_{t}", xflow[l, t]) for l in Lids for t in range(T)]

    xnse = m.addVars(seg_ids, zones, range(T), lb=0.0, name="xnse")
    out_vars += [(f"xnse_{s}_{z}_{t}", xnse[s, z, t]) for s in seg_ids for z in zones for t in range(T)]

    xcommit, xstart, xshut = {}, {}, {}
    if uc_ids:
        xcommit = m.addVars(uc_ids, range(T), lb=0.0, vtype=GRB.INTEGER, name="xcommit")
        xstart = m.addVars(uc_ids, range(T), lb=0.0, vtype=GRB.INTEGER, name="xstart")
        xshut = m.addVars(uc_ids, range(T), lb=0.0, vtype=GRB.INTEGER, name="xshut")
        out_vars += [(f"xcommit_{i}_{t}", xcommit[i, t]) for i in uc_ids for t in range(T)]
        out_vars += [(f"xstart_{i}_{t}", xstart[i, t]) for i in uc_ids for t in range(T)]
        out_vars += [(f"xshut_{i}_{t}", xshut[i, t]) for i in uc_ids for t in range(T)]

    xRPS, xCO2 = {}, {}
    if rps_on:
        xRPS = m.addVars(range(W), lb=0.0, name="xRPS")
        out_vars += [(f"xRPS_{w}", xRPS[w]) for w in range(W)]
    if co2_on:
        xCO2 = m.addVars(range(W), lb=0.0, name="xCO2")
        out_vars += [(f"xCO2_{w}", xCO2[w]) for w in range(W)]

    # ---------------- Constraints ----------------
    gens_by_zone, stor_by_zone = {}, {}
    for g in gens:
        gens_by_zone.setdefault(g["zone"], []).append(g["id"])
        if g.get("is_storage"):
            stor_by_zone.setdefault(g["zone"], []).append(g["id"])
    lines_from, lines_to = {}, {}
    for l in lines:
        lines_from.setdefault(l["from_zone"], []).append(l["id"])
        lines_to.setdefault(l["to_zone"], []).append(l["id"])

    # Power balance
    for z in zones:
        gz = gens_by_zone.get(z, [])
        sz = stor_by_zone.get(z, [])
        fl = lines_from.get(z, [])
        tl = lines_to.get(z, [])
        dz = demand[z]
        for t in range(T):
            expr = gp.LinExpr()
            if gz:
                expr.addTerms([1.0] * len(gz), [xinj[i, t] for i in gz])
            if sz:
                expr.addTerms([-1.0] * len(sz), [xwdw[i, t] for i in sz])
            if fl:
                expr.addTerms([-1.0] * len(fl), [xflow[l, t] for l in fl])
            if tl:
                expr.addTerms([1.0] * len(tl), [xflow[l, t] for l in tl])
            if seg_ids:
                expr.addTerms([1.0] * len(seg_ids), [xnse[s, z, t] for s in seg_ids])
            m.addConstr(expr == dz[t], name=f"c_bal_{z}_{t}")

    # Availability caps and minimum output
    for g in gens:
        i = g["id"]
        av = avail[i]
        minout = fnum(g.get("min_output_frac"), 0.0)
        if not g.get("is_UC"):
            for t in range(T):
                m.addConstr(xinj[i, t] <= av[t] * yP[i], name=f"c_gcap_{i}_{t}")
            if not g.get("is_storage") and not g.get("is_hydro") and minout > 0:
                for t in range(T):
                    m.addConstr(xinj[i, t] >= minout * yP[i], name=f"c_gmin_{i}_{t}")
        if g.get("is_hydro") and minout > 0:
            for t in range(T):
                m.addConstr(xinj[i, t] + spill[i, t] >= minout * yP[i], name=f"c_hmin_{i}_{t}")

    # Storage
    for g in gens:
        if not g.get("is_storage"):
            continue
        i = g["id"]
        av = avail[i]
        ec = fnum(g.get("charge_efficiency"), 1.0) or 1.0
        ed = fnum(g.get("discharge_efficiency"), 1.0) or 1.0
        sd = fnum(g.get("self_discharge_rate"), 0.0)
        for t in range(T):
            m.addConstr(xwdw[i, t] <= av[t] * yP[i], name=f"c_wcap_{i}_{t}")
            m.addConstr(xinj[i, t] + xwdw[i, t] <= yP[i], name=f"c_simul_{i}_{t}")
            m.addConstr(ec * xwdw[i, t] + xsoc[i, t] <= yE[i], name=f"c_room_{i}_{t}")
            m.addConstr(xinj[i, t] <= ed * xsoc[i, t], name=f"c_socdis_{i}_{t}")
            m.addConstr(xsoc[i, t] <= yE[i], name=f"c_soccap_{i}_{t}")
        for w in range(W):
            base = w * H
            for h in range(H):
                t = base + h
                if h == 0 and not circular:
                    continue
                tp = base + ((h - 1) % H)
                m.addConstr(
                    xsoc[i, t] == (1.0 - sd) * xsoc[i, tp] + ec * xwdw[i, t] - (1.0 / ed) * xinj[i, t],
                    name=f"c_socbal_{i}_{t}",
                )

    # Hydro
    for g in gens:
        if not g.get("is_hydro"):
            continue
        i = g["id"]
        dur = fnum(g.get("duration_MWh_per_MW"), 0.0)
        infl = inflow[i]
        for t in range(T):
            m.addConstr(xhydro[i, t] <= dur * yP[i], name=f"c_hcap_{i}_{t}")
        for w in range(W):
            base = w * H
            for h in range(H):
                t = base + h
                if h == 0 and not circular:
                    continue
                tp = base + ((h - 1) % H)
                m.addConstr(
                    xhydro[i, t] - xhydro[i, tp] == infl[t] * yP[i] - xinj[i, t] - spill[i, t],
                    name=f"c_hbal_{i}_{t}",
                )

    # Ramping for non-UC, non-storage resources
    for g in gens:
        if g.get("is_UC") or g.get("is_storage"):
            continue
        ru = g.get("ramp_up_frac_per_hr")
        rd = g.get("ramp_dn_frac_per_hr")
        if ru is None or rd is None:
            continue
        ru = float(ru)
        rd = float(rd)
        i = g["id"]
        for w in range(W):
            base = w * H
            for h in range(H):
                if h == 0 and not circular:
                    continue
                t = base + h
                tp = base + ((h - 1) % H)
                m.addConstr(xinj[i, t] - xinj[i, tp] <= ru * yP[i], name=f"c_ru_{i}_{t}")
                m.addConstr(xinj[i, tp] - xinj[i, t] <= rd * yP[i], name=f"c_rd_{i}_{t}")

    # Unit commitment
    for g in gens:
        if not g.get("is_UC"):
            continue
        i = g["id"]
        av = avail[i]
        size = fnum(g.get("capacity_size_MW"), 1.0) or 1.0
        minout = fnum(g.get("min_output_frac"), 0.0)
        ru = fnum(g.get("ramp_up_frac_per_hr"), 1.0) or 1.0
        rd = fnum(g.get("ramp_dn_frac_per_hr"), 1.0) or 1.0
        UT = int(g.get("min_up_time_hr") or 1)
        DT = int(g.get("min_down_time_hr") or 1)
        exist = fnum(g.get("existing_capacity_MW"), 0.0)
        total_units = exist / size + yPnew[i] - yPret[i]
        for t in range(T):
            m.addConstr(size * xcommit[i, t] <= yP[i], name=f"c_ucmax_{i}_{t}")
            m.addConstr(size * xstart[i, t] <= yP[i], name=f"c_stmax_{i}_{t}")
            m.addConstr(size * xshut[i, t] <= yP[i], name=f"c_sdmax_{i}_{t}")
            m.addConstr(xinj[i, t] >= minout * size * xcommit[i, t], name=f"c_ucmin_{i}_{t}")
            m.addConstr(xinj[i, t] <= av[t] * size * xcommit[i, t], name=f"c_ucgen_{i}_{t}")
        for w in range(W):
            base = w * H
            for h in range(H):
                t = base + h
                if not (h == 0 and not circular):
                    tp = base + ((h - 1) % H)
                    m.addConstr(
                        xcommit[i, t] - xcommit[i, tp] == xstart[i, t] - xshut[i, t],
                        name=f"c_uctrans_{i}_{t}",
                    )
                    cup = min(av[t], max(minout, ru))
                    cdn = min(av[t], max(minout, rd))
                    m.addConstr(
                        xinj[i, t] - xinj[i, tp]
                        <= size * (ru * (xcommit[i, t] - xstart[i, t])
                                   + cup * xstart[i, t] - minout * xshut[i, t]),
                        name=f"c_ucru_{i}_{t}",
                    )
                    m.addConstr(
                        xinj[i, tp] - xinj[i, t]
                        <= size * (rd * (xcommit[i, t] - xstart[i, t])
                                   + cdn * xshut[i, t] - minout * xstart[i, t]),
                        name=f"c_ucrd_{i}_{t}",
                    )
                if UT >= 1:
                    ks = []
                    for k in range(UT):
                        hk = h - k
                        if hk < 0:
                            if circular:
                                hk %= H
                            else:
                                continue
                        ks.append(base + hk)
                    if ks:
                        m.addConstr(
                            xcommit[i, t] >= gp.quicksum(xstart[i, tt] for tt in ks),
                            name=f"c_minup_{i}_{t}",
                        )
                if DT >= 1:
                    ks = []
                    for k in range(DT):
                        hk = h - k
                        if hk < 0:
                            if circular:
                                hk %= H
                            else:
                                continue
                        ks.append(base + hk)
                    if ks:
                        m.addConstr(
                            xcommit[i, t] + gp.quicksum(xshut[i, tt] for tt in ks) <= total_units,
                            name=f"c_mindn_{i}_{t}",
                        )

    # Transmission flow bounds
    for l in lines:
        lid = l["id"]
        Ftot = fnum(l.get("existing_capacity_MW"), 0.0) + yFnew[lid]
        for t in range(T):
            m.addConstr(xflow[lid, t] <= Ftot, name=f"c_fup_{lid}_{t}")
            m.addConstr(xflow[lid, t] >= -Ftot, name=f"c_flo_{lid}_{t}")

    # NSE bounds
    for seg in nse_segments:
        s = seg["segment"]
        mf = fnum(seg.get("max_frac"), 1.0)
        for z in zones:
            dz = demand[z]
            for t in range(T):
                m.addConstr(xnse[s, z, t] <= mf * dz[t], name=f"c_nse_{s}_{z}_{t}")

    # Policy constraints
    total_dem = 0.0
    for z in zones:
        dz = demand[z]
        for t in range(T):
            total_dem += hw[t] * dz[t]

    if rps_on:
        share = fnum(policy.get("RPS_share"), 0.0)
        rps_ids = [g["id"] for g in gens if g.get("is_RPS_qualifying")]
        expr = gp.LinExpr()
        for i in rps_ids:
            for t in range(T):
                if hw[t] != 0:
                    expr.add(xinj[i, t], hw[t])
        for w in range(W):
            expr.add(xRPS[w], 1.0)
        m.addConstr(expr >= share * total_dem, name="c_rps")

    if co2_on:
        cap_per = policy.get("CO2_cap_tons_per_MWh")
        cap_per = 0.05 if cap_per is None else float(cap_per)
        expr = gp.LinExpr()
        for g in gens:
            i = g["id"]
            ef = fnum(g.get("co2_tons_per_MWh"), 0.0)
            if ef == 0.0:
                continue
            for t in range(T):
                if hw[t] != 0:
                    expr.add(xinj[i, t], hw[t] * ef)
            if g.get("is_storage"):
                for t in range(T):
                    if hw[t] != 0:
                        expr.add(xwdw[i, t], hw[t] * ef)
        for w in range(W):
            expr.add(xCO2[w], -1.0)
        m.addConstr(expr <= cap_per * total_dem, name="c_co2")

    # ---------------- Objective ----------------
    obj = gp.LinExpr()
    for g in gens:
        i = g["id"]
        size = fnum(g.get("capacity_size_MW"), 1.0) or 1.0
        obj.add(yPnew[i], fnum(g.get("inv_cost_per_MW_yr"), 0.0) * size)
        obj.add(yP[i], fnum(g.get("fom_cost_per_MW_yr"), 0.0))
        if g.get("is_hydro"):
            dur = fnum(g.get("duration_MWh_per_MW"), 0.0)
            obj.add(yPnew[i], fnum(g.get("inv_cost_energy_per_MWh_yr"), 0.0) * dur * size)
            obj.add(yP[i], fnum(g.get("fom_cost_energy_per_MWh_yr"), 0.0) * dur)
        if g.get("is_storage"):
            esz = fnum(g.get("storage_capacity_size_MWh"), 1.0) or 1.0
            obj.add(yEnew[i], fnum(g.get("inv_cost_energy_per_MWh_yr"), 0.0) * esz)
            obj.add(yE[i], fnum(g.get("fom_cost_energy_per_MWh_yr"), 0.0))
    for l in lines:
        obj.add(yFnew[l["id"]], fnum(l.get("inv_cost_per_MW_yr"), 0.0))

    for g in gens:
        i = g["id"]
        vc = g.get("var_cost_per_MWh")
        if isinstance(vc, (list, tuple)):
            for t in range(T):
                c = hw[t] * float(vc[t])
                if c != 0:
                    obj.add(xinj[i, t], c)
                if g.get("is_storage"):
                    if c != 0:
                        obj.add(xwdw[i, t], c)
        else:
            vcf = fnum(vc, 0.0)
            if vcf != 0:
                for t in range(T):
                    c = hw[t] * vcf
                    if c != 0:
                        obj.add(xinj[i, t], c)
                if g.get("is_storage"):
                    for t in range(T):
                        c = hw[t] * vcf
                        if c != 0:
                            obj.add(xwdw[i, t], c)

    def seg_cost(seg, z):
        c = seg.get("cost_per_MWh", 0.0)
        if isinstance(c, dict):
            return fnum(c.get(str(z), c.get(z, 0.0)), 0.0)
        return fnum(c, 0.0)

    for seg in nse_segments:
        s = seg["segment"]
        for z in zones:
            c = seg_cost(seg, z)
            if c == 0:
                continue
            for t in range(T):
                if hw[t] != 0:
                    obj.add(xnse[s, z, t], hw[t] * c)

    for g in gens:
        if not g.get("is_UC"):
            continue
        i = g["id"]
        sc = fnum(g.get("start_cost_per_unit"), 0.0)
        if sc == 0:
            continue
        for t in range(T):
            if hw[t] != 0:
                obj.add(xstart[i, t], hw[t] * sc)

    if rps_on:
        c = fnum(policy.get("RPS_noncompliance_cost_per_MWh"), 0.0)
        for w in range(W):
            obj.add(xRPS[w], c)
    if co2_on:
        c = fnum(policy.get("CO2_noncompliance_cost_per_ton"), 0.0)
        for w in range(W):
            obj.add(xCO2[w], c)

    m.setObjective(obj, GRB.MINIMIZE)

    # ---------------- Solution assembly ----------------
    var_names = [nm for nm, _ in out_vars]
    var_list = [v for _, v in out_vars]

    def assemble(objval, vals):
        mv = {}
        valmap = {}
        for nm, v in zip(var_names, vals):
            v = 0.0 if v is None else float(v)
            valmap[nm] = v
            if abs(v) > 1e-9:
                rv = round(v, 6)
                if rv != 0:
                    mv[nm] = rv
        inv = []
        for g in gens:
            i = g["id"]
            entry = {
                "generator_id": i,
                "resource_type": g.get("resource_type"),
                "zone": g.get("zone"),
                "y_P_new": round(valmap.get(f"yPnew_{i}", 0.0), 6),
                "y_P_ret": round(valmap.get(f"yPret_{i}", 0.0), 6),
                "y_P_total_MW": round(valmap.get(f"yP_{i}", fnum(g.get("existing_capacity_MW"), 0.0)), 6),
            }
            if g.get("is_storage"):
                entry["y_E_new"] = round(valmap.get(f"yEnew_{i}", 0.0), 6)
                entry["y_E_ret"] = round(valmap.get(f"yEret_{i}", 0.0), 6)
                entry["y_E_total_MWh"] = round(
                    valmap.get(f"yE_{i}", fnum(g.get("existing_storage_capacity_MWh"), 0.0)), 6
                )
            inv.append(entry)
        trans = []
        for l in lines:
            lid = l["id"]
            yn = valmap.get(f"yFnew_{lid}", 0.0)
            trans.append({
                "line_id": lid,
                "y_F_new_MW": round(yn, 6),
                "total_capacity_MW": round(fnum(l.get("existing_capacity_MW"), 0.0) + yn, 6),
            })
        return {
            "objective_value": objval,
            "model_variables": mv,
            "investment_decisions": inv,
            "transmission_decisions": trans,
            "model_variable_name_patterns": PATTERNS,
        }

    best_logged = {"obj": float("inf")}

    def cb(model, where):
        if where == GRB.Callback.MIPSOL and logger is not None:
            try:
                cur = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if cur < best_logged["obj"] - 1e-6:
                    vals = model.cbGetSolution(var_list)
                    best_logged["obj"] = cur
                    logger.log_solution(cur, assemble(cur, vals))
            except Exception:
                pass

    remaining = max(1, int(args.time_limit - (time.time() - t0)))
    m.Params.TimeLimit = remaining
    m.optimize(cb)

    if m.SolCount > 0:
        objval = float(m.ObjVal)
        vals = [v.X for v in var_list]
        sol = assemble(objval, vals)
        if logger is not None and objval < best_logged["obj"] - 1e-6:
            best_logged["obj"] = objval
            logger.log_solution(objval, sol)
    else:
        # Fallback: zero-activity witness (keeps output schema valid)
        sol = assemble(0.0, [0.0] * len(var_list))
        for e, g in zip(sol["investment_decisions"], gens):
            e["y_P_total_MW"] = fnum(g.get("existing_capacity_MW"), 0.0)
            if g.get("is_storage"):
                e["y_E_total_MWh"] = fnum(g.get("existing_storage_capacity_MWh"), 0.0)

    outdir = os.path.dirname(os.path.abspath(args.solution_path))
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    with open(args.solution_path, "w") as f:
        json.dump(sol, f)


if __name__ == "__main__":
    main()
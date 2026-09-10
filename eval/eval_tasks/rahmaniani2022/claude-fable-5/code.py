import argparse
import json
import math
import time
from types import SimpleNamespace

import gurobipy as gp
from gurobipy import GRB, quicksum


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", required=True)
    ap.add_argument("--solution_path", required=True)
    ap.add_argument("--time_limit", type=int, default=3600)
    ap.add_argument("--log_path", default=None)
    args = ap.parse_args()
    t0 = time.time()

    logger = None
    if args.log_path:
        try:
            from solution_logger import SolutionLogger
            logger = SolutionLogger(args.log_path, sense="minimize")
        except Exception:
            logger = None

    with open(args.instance_path) as f:
        data = json.load(f)

    P = data["problem_parameters"]
    H = int(P.get("hours_per_subperiod", 168))
    T = int(P["total_hours"])
    nw = int(P["num_weeks"])
    weights = data["subperiod_weights"]
    omega = [float(weights[t // H]) / H for t in range(T)]
    prevh = [t - 1 if t % H else t + H - 1 for t in range(T)]

    gens = data["generators"]
    gpar = data["generator_parameters"]
    exist = data.get("existing_capacity", {}) or {}
    demand = data["demand"]
    zones = sorted(int(k) for k in demand.keys())
    segs = data.get("consumer_segments", []) or []
    lines = data.get("transmission_lines", []) or []
    avail = data.get("availability", {}) or {}
    hinf = data.get("hydro_inflow", {}) or {}
    policy = data.get("policy", {}) or {}

    m = gp.Model("cep")
    m.Params.OutputFlag = 0
    m.Params.Seed = 0
    m.Params.MIPGap = 1e-4
    m.Params.NumericFocus = 0
    m.Params.Threads = 1

    INF = GRB.INFINITY
    varlist = []
    varidx = {}

    def reg(v):
        varidx[id(v)] = len(varlist)
        varlist.append(v)
        return v

    def regvars(td):
        for k in range(len(td)):
            reg(td[k])
        return td

    # ------------------ build clusters ------------------
    clusters = []
    for g in gens:
        c = SimpleNamespace()
        c.gid = int(g["cluster_id"])
        c.zone = int(g["zone"])
        c.typ = g["resource_type"]
        c.stor = bool(g.get("is_storage", False))
        c.hyd = bool(g.get("is_hydro", False))
        c.vre = bool(g.get("is_VRE", False))
        c.uc = bool(g.get("is_UC", False)) and not c.stor and not c.hyd
        c.rps = bool(g.get("is_RPS_qualifying", False))
        c.canret = bool(g.get("can_retire", False))
        c.size = float(g.get("capacity_size_MW", 1.0) or 1.0)
        pr = gpar.get(c.typ, {}) or {}
        c.pr = pr
        c.inc = c.size if c.uc else 1.0
        c.ex = float(exist.get(str(c.gid), 0.0))
        av = avail.get(str(c.gid))
        c.av = av if av is not None else [1.0] * T
        c.vc = float(pr.get("variable_cost_per_MWh", 0.0) or 0.0)
        c.fom = float(pr.get("fixed_OM_cost_per_MW_yr", 0.0) or 0.0)
        c.ic = float(pr.get("investment_cost_per_MW_yr", 0.0) or 0.0)
        c.em = float(pr.get("CO2_emission_tons_per_MWh", 0.0) or 0.0)
        c.minf = float(pr.get("min_output_frac", 0.0) or 0.0)
        mc = pr.get("max_capacity_MW", -1.0)
        mc = -1.0 if mc is None else float(mc)
        ubN = INF if mc < 0 else max(0, int(math.floor(mc / c.inc + 1e-6)))
        ubR = int(math.floor(c.ex / c.inc + 1e-6)) if c.canret else 0
        c.vN = reg(m.addVar(vtype=GRB.INTEGER, lb=0.0, ub=ubN))
        c.vR = reg(m.addVar(vtype=GRB.INTEGER, lb=0.0, ub=ubR))
        c.capconst = (ubN == 0 and ubR == 0)
        c.Cap = c.ex + c.inc * (c.vN - c.vR)

        if c.stor:
            psize = float(pr.get("capacity_size_MW", 0.0) or 0.0)
            esize = float(pr.get("energy_capacity_size_MWh", 0.0) or 0.0)
            c.einc = esize if bool(g.get("is_UC", False)) and esize > 0 else 1.0
            c.exE = (c.ex / psize * esize) if psize > 0 else 0.0
            me = pr.get("max_energy_capacity_MWh", -1.0)
            me = -1.0 if me is None else float(me)
            ubNE = INF if me < 0 else max(0, int(math.floor(me / c.einc + 1e-6)))
            ubRE = int(math.floor(c.exE / c.einc + 1e-6)) if c.canret else 0
            c.vNE = reg(m.addVar(vtype=GRB.INTEGER, lb=0.0, ub=ubNE))
            c.vRE = reg(m.addVar(vtype=GRB.INTEGER, lb=0.0, ub=ubRE))
            c.econst = (ubNE == 0 and ubRE == 0)
            c.ECap = c.exE + c.einc * (c.vNE - c.vRE)
            c.effc = float(pr.get("charging_efficiency", 1.0) or 1.0)
            c.effd = float(pr.get("discharging_efficiency", 1.0) or 1.0)
            c.sdr = float(pr.get("self_discharge_rate_pct", 0.0) or 0.0)
            mind = float(pr.get("min_duration_MWh_per_MW", 0.0) or 0.0)
            maxd = float(pr.get("max_duration_MWh_per_MW", 1e9) or 1e9)
            m.addConstr(c.ECap >= mind * c.Cap)
            m.addConstr(c.ECap <= maxd * c.Cap)
        if c.hyd:
            c.dur = float(pr.get("duration_MWh_per_MW", 0.0) or 0.0)
            c.infl = hinf.get(str(c.gid), [0.0] * T)
        clusters.append(c)

    # operational variables
    for c in clusters:
        if c.capconst and not c.uc:
            pub = [c.av[t] * c.ex for t in range(T)]
        else:
            pub = [INF] * T
        c.p = regvars(m.addVars(T, lb=0.0, ub=pub))
        if c.uc:
            cu = (c.ex / c.size) if c.capconst else INF
            c.cm = regvars(m.addVars(T, lb=0.0, ub=cu))
            c.su = regvars(m.addVars(T, lb=0.0, ub=cu))
            c.sd = regvars(m.addVars(T, lb=0.0, ub=cu))
        if c.stor:
            cub = [c.av[t] * c.ex for t in range(T)] if c.capconst else [INF] * T
            c.ch = regvars(m.addVars(T, lb=0.0, ub=cub))
            sub = c.exE if c.econst else INF
            c.soc = regvars(m.addVars(T, lb=0.0, ub=sub))
        if c.hyd:
            lub = c.dur * c.ex if c.capconst else INF
            c.lev = regvars(m.addVars(T, lb=0.0, ub=lub))
            c.sp = regvars(m.addVars(T, lb=0.0))

    # ------------------ operational constraints ------------------
    for c in clusters:
        if c.uc:
            size = c.size
            ru = float(c.pr.get("ramp_up_rate_pct_per_hr", 1.0) or 1.0)
            rd = float(c.pr.get("ramp_down_rate_pct_per_hr", 1.0) or 1.0)
            mu = max(1, min(int(c.pr.get("min_up_time_hr", 1) or 1), H))
            md = max(1, min(int(c.pr.get("min_down_time_hr", 1) or 1), H))
            for t in range(T):
                tp = prevh[t]
                if not c.capconst:
                    m.addConstr(size * c.cm[t] <= c.Cap)
                    m.addConstr(size * c.su[t] <= c.Cap)
                    m.addConstr(size * c.sd[t] <= c.Cap)
                m.addConstr(c.p[t] <= c.av[t] * size * c.cm[t])
                if c.minf > 0:
                    m.addConstr(c.p[t] >= c.minf * size * c.cm[t])
                m.addConstr(c.cm[t] - c.cm[tp] == c.su[t] - c.sd[t])
                kup = min(c.av[t], max(c.minf, ru))
                m.addConstr(c.p[t] - c.p[tp] <= ru * size * (c.cm[t] - c.su[t])
                            + kup * size * c.su[t] - c.minf * size * c.sd[t])
                kdn = min(c.av[t], max(c.minf, rd))
                m.addConstr(c.p[tp] - c.p[t] <= rd * size * (c.cm[t] - c.su[t])
                            - c.minf * size * c.su[t] + kdn * size * c.sd[t])
            for t in range(T):
                w0 = (t // H) * H
                h = t - w0
                m.addConstr(c.cm[t] >= quicksum(c.su[w0 + (h - k) % H] for k in range(mu)))
                m.addConstr(size * c.cm[t]
                            + size * quicksum(c.sd[w0 + (h - k) % H] for k in range(md))
                            <= c.Cap)
        elif c.stor:
            for t in range(T):
                tp = prevh[t]
                m.addConstr(c.soc[t] == (1.0 - c.sdr) * c.soc[tp]
                            + c.effc * c.ch[t] - c.p[t] / c.effd)
                if not c.econst:
                    m.addConstr(c.soc[t] <= c.ECap)
                # room / content constraints (both start-of-hour and end-of-hour readings)
                m.addConstr(c.effc * c.ch[t] + c.soc[tp] <= c.ECap)
                m.addConstr(c.effc * c.ch[t] + c.soc[t] <= c.ECap)
                m.addConstr(c.p[t] <= c.effd * c.soc[tp])
                m.addConstr(c.p[t] <= c.effd * c.soc[t])
                m.addConstr(c.p[t] + c.ch[t] <= c.Cap)
                if not c.capconst:
                    m.addConstr(c.ch[t] <= c.av[t] * c.Cap)
                    m.addConstr(c.p[t] <= c.av[t] * c.Cap)
        elif c.hyd:
            for t in range(T):
                tp = prevh[t]
                m.addConstr(c.lev[t] == c.lev[tp] + float(c.infl[t]) * c.Cap
                            - c.p[t] - c.sp[t])
                if not c.capconst:
                    m.addConstr(c.lev[t] <= c.dur * c.Cap)
                    m.addConstr(c.p[t] <= c.av[t] * c.Cap)
                if c.minf > 0:
                    m.addConstr(c.p[t] + c.sp[t] >= c.minf * c.Cap)
        else:
            if not c.capconst:
                for t in range(T):
                    m.addConstr(c.p[t] <= c.av[t] * c.Cap)
                if c.minf > 0:
                    for t in range(T):
                        m.addConstr(c.p[t] >= c.minf * c.Cap)
            else:
                if c.minf > 0:
                    for t in range(T):
                        c.p[t].LB = c.minf * c.ex
        if not c.uc:
            ru = float(c.pr.get("ramp_up_rate_pct_per_hr", 1.0) or 1.0)
            rd = float(c.pr.get("ramp_down_rate_pct_per_hr", 1.0) or 1.0)
            if ru < 1.0 - 1e-9:
                for t in range(T):
                    m.addConstr(c.p[t] - c.p[prevh[t]] <= ru * c.Cap)
            if rd < 1.0 - 1e-9:
                for t in range(T):
                    m.addConstr(c.p[prevh[t]] - c.p[t] <= rd * c.Cap)

    # ------------------ transmission ------------------
    linfo = []
    for ld in lines:
        ex = float(ld["existing_capacity_MW"])
        mx = ld.get("max_capacity_MW", -1.0)
        mx = -1.0 if mx is None else float(mx)
        ubT = INF if mx < 0 else max(0, int(math.floor(mx - ex + 1e-6)))
        vT = reg(m.addVar(vtype=GRB.INTEGER, lb=0.0, ub=ubT))
        if ubT == 0:
            fl = regvars(m.addVars(T, lb=-ex, ub=ex))
        else:
            fl = regvars(m.addVars(T, lb=-INF, ub=INF))
            for t in range(T):
                m.addConstr(fl[t] - vT <= ex)
                m.addConstr(-fl[t] - vT <= ex)
        linfo.append((ld, vT, fl))

    # ------------------ non-served energy ------------------
    nsev = []
    for s in segs:
        z = int(s["zone"])
        frac = float(s["max_nse_fraction"])
        dz = demand[str(z)]
        ub = [frac * float(dz[t]) for t in range(T)]
        v = regvars(m.addVars(T, lb=0.0, ub=ub))
        nsev.append((s, v))

    # ------------------ zonal balance ------------------
    for z in zones:
        dz = demand[str(z)]
        gz = [c for c in clusters if c.zone == z]
        stz = [c for c in gz if c.stor]
        sz = [v for (s, v) in nsev if int(s["zone"]) == z]
        lz = []
        for (ld, vT, fl) in linfo:
            if int(ld["to_zone"]) == z:
                lz.append((fl, 1.0))
            if int(ld["from_zone"]) == z:
                lz.append((fl, -1.0))
        for t in range(T):
            expr = quicksum(c.p[t] for c in gz)
            if stz:
                expr = expr - quicksum(c.ch[t] for c in stz)
            if lz:
                expr = expr + quicksum(sg * fl[t] for (fl, sg) in lz)
            if sz:
                expr = expr + quicksum(v[t] for v in sz)
            m.addConstr(expr == float(dz[t]))

    # ------------------ policy ------------------
    ptype = str(policy.get("type", "REF")).upper()
    Dtot = sum(omega[t] * sum(float(demand[str(z)][t]) for z in zones) for t in range(T))
    slackR = None
    slackC = None
    obj = gp.LinExpr()
    if ptype == "RPS":
        slackR = regvars(m.addVars(nw, lb=0.0))
        m.addConstr(quicksum(omega[t] * c.p[t] for c in clusters if c.rps for t in range(T))
                    + quicksum(slackR[w] for w in range(nw))
                    >= float(policy.get("RPS_share", 0.0)) * Dtot)
        obj += float(policy.get("RPS_noncompliance_cost_per_MWh", 0.0)) * \
            quicksum(slackR[w] for w in range(nw))
    elif ptype == "CO2":
        slackC = regvars(m.addVars(nw, lb=0.0))
        em = gp.LinExpr()
        for c in clusters:
            if c.em != 0.0:
                em += quicksum(omega[t] * c.em * c.p[t] for t in range(T))
                if c.stor:
                    em += quicksum(omega[t] * c.em * c.ch[t] for t in range(T))
        m.addConstr(em - quicksum(slackC[w] for w in range(nw))
                    <= float(policy.get("CO2_cap_tons_per_MWh", 0.0)) * Dtot)
        obj += float(policy.get("CO2_noncompliance_cost_per_ton", 0.0)) * \
            quicksum(slackC[w] for w in range(nw))

    # ------------------ objective ------------------
    for c in clusters:
        obj += c.ic * c.inc * c.vN
        obj += c.fom * c.Cap
        if c.stor:
            obj += float(c.pr.get("energy_investment_cost_per_MWh_yr", 0.0) or 0.0) * c.einc * c.vNE
            obj += float(c.pr.get("energy_fixed_OM_cost_per_MWh_yr", 0.0) or 0.0) * c.ECap
        if c.hyd:
            obj += float(c.pr.get("energy_investment_cost_per_MWh_yr", 0.0) or 0.0) * c.dur * c.inc * c.vN
            obj += float(c.pr.get("energy_fixed_OM_cost_per_MWh_yr", 0.0) or 0.0) * c.dur * c.Cap
        if c.vc != 0.0:
            obj += quicksum(omega[t] * c.vc * c.p[t] for t in range(T))
            if c.stor:
                obj += quicksum(omega[t] * c.vc * c.ch[t] for t in range(T))
        if c.uc:
            sc = float(c.pr.get("start_up_cost", 0.0) or 0.0)
            if sc != 0.0:
                obj += quicksum(omega[t] * sc * c.su[t] for t in range(T))
    for (s, v) in nsev:
        cst = float(s["cost_per_MWh"])
        if cst != 0.0:
            obj += quicksum(omega[t] * cst * v[t] for t in range(T))
    for (ld, vT, fl) in linfo:
        obj += float(ld.get("investment_cost_per_MW_yr", 0.0) or 0.0) * vT
    m.setObjective(obj, GRB.MINIMIZE)

    # ------------------ solution builder ------------------
    def build_solution(objval, val):
        mv = {}

        def put(name, x):
            if x is not None and abs(x) > 1e-7:
                mv[name] = float(x)

        inv = {}
        sinv = {}
        tinv = {}
        for c in clusters:
            nv = int(round(val(c.vN)))
            rv = int(round(val(c.vR)))
            put("vNEW[%d]" % c.gid, nv)
            put("vRET[%d]" % c.gid, rv)
            inv[str(c.gid)] = {
                "resource_type": c.typ,
                "zone": c.zone,
                "new_units": float(nv),
                "new_MW": float(c.inc * nv),
                "retired_units": float(rv),
                "retired_MW": float(c.inc * rv),
                "total_capacity_MW": float(c.ex + c.inc * (nv - rv)),
            }
            for t in range(T):
                put("vP[%d,%d]" % (c.gid, t), val(c.p[t]))
            if c.uc:
                for t in range(T):
                    put("vCOMMIT[%d,%d]" % (c.gid, t), val(c.cm[t]))
                    put("vSTART[%d,%d]" % (c.gid, t), val(c.su[t]))
                    put("vSHUT[%d,%d]" % (c.gid, t), val(c.sd[t]))
            if c.stor:
                ne = int(round(val(c.vNE)))
                re_ = int(round(val(c.vRE)))
                put("vNEW_E[%d]" % c.gid, ne)
                put("vRET_E[%d]" % c.gid, re_)
                sinv[str(c.gid)] = {
                    "new_units": float(ne),
                    "new_MWh": float(c.einc * ne),
                    "total_energy_MWh": float(c.exE + c.einc * (ne - re_)),
                }
                for t in range(T):
                    put("vCHARGE[%d,%d]" % (c.gid, t), val(c.ch[t]))
                    put("vSOC[%d,%d]" % (c.gid, t), val(c.soc[t]))
            if c.hyd:
                for t in range(T):
                    put("vLEVEL[%d,%d]" % (c.gid, t), val(c.lev[t]))
                    put("vSPILL[%d,%d]" % (c.gid, t), val(c.sp[t]))
        for (ld, vT, fl) in linfo:
            lid = int(ld["line_id"])
            tn = int(round(val(vT)))
            put("vTNEW[%d]" % lid, tn)
            tinv[str(lid)] = {
                "new_MW": float(tn),
                "total_capacity_MW": float(float(ld["existing_capacity_MW"]) + tn),
            }
            for t in range(T):
                put("vFLOW[%d,%d]" % (lid, t), val(fl[t]))
        for (s, v) in nsev:
            sid = int(s["segment_id"])
            z = int(s["zone"])
            for t in range(T):
                put("vNSE[%d,%d,%d]" % (sid, z, t), val(v[t]))
        if slackR is not None:
            for w in range(nw):
                put("vRPS_slack[%d]" % w, val(slackR[w]))
        if slackC is not None:
            for w in range(nw):
                put("vCO2_slack[%d]" % w, val(slackC[w]))
        return {
            "objective_value": float(objval),
            "model_variables": mv,
            "investments": inv,
            "storage_energy_investments": sinv,
            "transmission_investments": tinv,
        }

    best = {"obj": float("inf"), "sol": None}

    def cb(model, where):
        if where == GRB.Callback.MIPSOL:
            try:
                objv = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if objv < best["obj"] - 1e-9:
                    vals = model.cbGetSolution(varlist)
                    sol = build_solution(objv, lambda v: vals[varidx[id(v)]])
                    best["obj"] = objv
                    best["sol"] = sol
                    if logger:
                        logger.log_solution(objv, sol)
            except Exception:
                pass

    remaining = args.time_limit - (time.time() - t0) - 10.0
    m.Params.TimeLimit = max(5.0, remaining)

    try:
        m.optimize(cb)
    except Exception:
        pass

    sol = None
    try:
        if m.SolCount > 0:
            sol = build_solution(m.ObjVal, lambda v: v.X)
            if logger and m.ObjVal < best["obj"] - 1e-9:
                logger.log_solution(m.ObjVal, sol)
    except Exception:
        sol = None
    if sol is None:
        if best["sol"] is not None:
            sol = best["sol"]
        else:
            sol = build_solution(1e18, lambda v: 0.0)

    with open(args.solution_path, "w") as f:
        json.dump(sol, f)


if __name__ == "__main__":
    main()
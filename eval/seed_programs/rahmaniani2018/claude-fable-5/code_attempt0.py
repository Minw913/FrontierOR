import argparse
import json
import time

import gurobipy as gp
from gurobipy import GRB


def gf(d, key, default=0.0):
    """Get field with None treated as default."""
    v = d.get(key)
    return default if v is None else v


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

    sub = data["subperiods"]
    W = int(sub["num_subperiods"])
    H = int(sub["hours_per_subperiod"])
    weights = [float(w) for w in sub["subperiod_weights"]]
    T = W * H
    omega = [weights[t // H] / float(H) for t in range(T)]

    gens = data["generators"]
    lines = data.get("transmission_lines", [])
    demand = {int(z): v for z, v in data["demand"].items()}
    zones = sorted(demand.keys())
    segs = data["nse_segments"]
    policy = data["policy"]
    profs = data.get("availability_profiles", {})

    # availability / inflow profiles
    avail = {}
    for g in gens:
        i = g["id"]
        p = profs.get(str(i), profs.get(i))
        if p is None:
            avail[i] = [1.0] * T
        elif p.get("type") == "constant":
            avail[i] = [float(p["value"])] * T
        else:
            avail[i] = [float(x) for x in p["values"][:T]]

    def vlist(g):
        v = g.get("var_cost_per_MWh", 0.0)
        if isinstance(v, (list, tuple)):
            return [float(x) for x in list(v)[:T]]
        return [float(v or 0.0)] * T

    def prevt(t):
        b = (t // H) * H
        return b + (t - b - 1) % H

    # total weighted demand
    D_tot = 0.0
    for z in zones:
        dz = demand[z]
        for t in range(T):
            D_tot += omega[t] * dz[t]

    m = gp.Model("cep")
    m.Params.OutputFlag = 1
    m.Params.Seed = 0
    m.Params.MIPGap = 1e-4
    m.Params.NumericFocus = 0
    m.Params.Threads = 1

    INF = GRB.INFINITY
    obj_c = []
    obj_v = []

    def addobj(c, v):
        if c:
            obj_c.append(float(c))
            obj_v.append(v)

    # ---------------- investment variables ----------------
    yPnew = {}
    yPret = {}
    yP = {}
    yEnew = {}
    yEret = {}
    yE = {}
    for g in gens:
        i = g["id"]
        size = g.get("capacity_size_MW") or 1.0
        ex = gf(g, "existing_capacity_MW")
        mx = gf(g, "max_capacity_MW")
        can_ret = bool(g.get("can_retire", True))
        yPnew[i] = m.addVar(vtype=GRB.INTEGER, lb=0.0, ub=mx / size,
                            name=f"yPnew_{i}")
        yPret[i] = m.addVar(vtype=GRB.INTEGER, lb=0.0,
                            ub=(ex / size if can_ret else 0.0),
                            name=f"yPret_{i}")
        yP[i] = m.addVar(lb=0.0, name=f"yP_{i}")
        m.addConstr(yP[i] - size * yPnew[i] + size * yPret[i] == ex)
        addobj(gf(g, "inv_cost_per_MW_yr") * size, yPnew[i])
        addobj(gf(g, "fom_cost_per_MW_yr"), yP[i])
        if g.get("is_hydro"):
            dur = gf(g, "duration_MWh_per_MW")
            addobj(gf(g, "inv_cost_energy_per_MWh_yr") * dur * size, yPnew[i])
            addobj(gf(g, "fom_cost_energy_per_MWh_yr") * dur, yP[i])
        if g.get("is_storage"):
            sizeE = g.get("storage_capacity_size_MWh") or 1.0
            exE = gf(g, "existing_storage_capacity_MWh")
            mxE = gf(g, "max_storage_capacity_MWh")
            yEnew[i] = m.addVar(vtype=GRB.INTEGER, lb=0.0, ub=mxE / sizeE,
                                name=f"yEnew_{i}")
            yEret[i] = m.addVar(vtype=GRB.INTEGER, lb=0.0,
                                ub=(exE / sizeE if can_ret else 0.0),
                                name=f"yEret_{i}")
            yE[i] = m.addVar(lb=0.0, name=f"yE_{i}")
            m.addConstr(yE[i] - sizeE * yEnew[i] + sizeE * yEret[i] == exE)
            mind = g.get("min_duration_MWh_per_MW")
            if mind is not None and mind > 0:
                m.addConstr(yE[i] >= mind * yP[i])
            maxd = g.get("max_duration_MWh_per_MW")
            if maxd is not None:
                m.addConstr(yE[i] <= maxd * yP[i])
            addobj(gf(g, "inv_cost_energy_per_MWh_yr") * sizeE, yEnew[i])
            addobj(gf(g, "fom_cost_energy_per_MWh_yr"), yE[i])

    yFnew = {}
    for l in lines:
        lid = l["id"]
        yFnew[lid] = m.addVar(vtype=GRB.INTEGER, lb=0.0,
                              ub=gf(l, "max_new_capacity_MW"),
                              name=f"yFnew_{lid}")
        addobj(gf(l, "inv_cost_per_MW_yr"), yFnew[lid])

    # ---------------- operational variables ----------------
    xinj = {}
    xwdw = {}
    xsoc = {}
    xhyd = {}
    xsp = {}
    xcom = {}
    xst = {}
    xsh = {}
    for g in gens:
        i = g["id"]
        xinj[i] = [m.addVar(lb=0.0, name=f"xinj_{i}_{t}") for t in range(T)]
        vcl = vlist(g)
        for t in range(T):
            addobj(omega[t] * vcl[t], xinj[i][t])
        if g.get("is_storage"):
            xwdw[i] = [m.addVar(lb=0.0, name=f"xwdw_{i}_{t}") for t in range(T)]
            xsoc[i] = [m.addVar(lb=0.0, name=f"xsoc_{i}_{t}") for t in range(T)]
            for t in range(T):
                addobj(omega[t] * vcl[t], xwdw[i][t])
        if g.get("is_hydro"):
            xhyd[i] = [m.addVar(lb=0.0, name=f"xhydro_{i}_{t}") for t in range(T)]
            xsp[i] = [m.addVar(lb=0.0, name=f"SPILL_{i}_{t}") for t in range(T)]
        if g.get("is_UC"):
            size = g.get("capacity_size_MW") or 1.0
            umax = (gf(g, "existing_capacity_MW") + gf(g, "max_capacity_MW")) / size
            xcom[i] = [m.addVar(vtype=GRB.INTEGER, lb=0.0, ub=umax,
                                name=f"xcommit_{i}_{t}") for t in range(T)]
            xst[i] = [m.addVar(vtype=GRB.INTEGER, lb=0.0, ub=umax,
                               name=f"xstart_{i}_{t}") for t in range(T)]
            xsh[i] = [m.addVar(vtype=GRB.INTEGER, lb=0.0, ub=umax,
                               name=f"xshut_{i}_{t}") for t in range(T)]
            sc = gf(g, "start_cost_per_unit")
            for t in range(T):
                addobj(omega[t] * sc, xst[i][t])

    xflow = {}
    for l in lines:
        lid = l["id"]
        xflow[lid] = [m.addVar(lb=-INF, name=f"xflow_{lid}_{t}") for t in range(T)]

    xnse = {}
    seg_ids = []
    for seg in segs:
        s = seg["segment"]
        seg_ids.append(s)
        mf = gf(seg, "max_frac")
        c = seg.get("cost_per_MWh", 0.0)
        for z in zones:
            if isinstance(c, dict):
                cz = c.get(str(z), c.get(z, 0.0))
            else:
                cz = c
            dz = demand[z]
            row = []
            for t in range(T):
                v = m.addVar(lb=0.0, ub=mf * dz[t], name=f"xnse_{s}_{z}_{t}")
                addobj(omega[t] * cz, v)
                row.append(v)
            xnse[(s, z)] = row

    # ---------------- power balance ----------------
    genz = {z: [] for z in zones}
    storz = {z: [] for z in zones}
    for g in gens:
        genz[g["zone"]].append(g["id"])
        if g.get("is_storage"):
            storz[g["zone"]].append(g["id"])
    lout = {z: [] for z in zones}
    lin_ = {z: [] for z in zones}
    for l in lines:
        lout[l["from_zone"]].append(l["id"])
        lin_[l["to_zone"]].append(l["id"])

    for z in zones:
        dz = demand[z]
        gl = genz[z]
        sl = storz[z]
        lo = lout[z]
        li = lin_[z]
        for t in range(T):
            expr = gp.LinExpr()
            for i in gl:
                expr.add(xinj[i][t], 1.0)
            for i in sl:
                expr.add(xwdw[i][t], -1.0)
            for lid in lo:
                expr.add(xflow[lid][t], -1.0)
            for lid in li:
                expr.add(xflow[lid][t], 1.0)
            for s in seg_ids:
                expr.add(xnse[(s, z)][t], 1.0)
            m.addConstr(expr == dz[t])

    # ---------------- transmission limits ----------------
    for l in lines:
        lid = l["id"]
        exl = gf(l, "existing_capacity_MW")
        fv = yFnew[lid]
        for t in range(T):
            m.addConstr(xflow[lid][t] - fv <= exl)
            m.addConstr(-xflow[lid][t] - fv <= exl)

    # ---------------- resource constraints ----------------
    for g in gens:
        i = g["id"]
        a = avail[i]
        is_uc = bool(g.get("is_UC"))
        is_st = bool(g.get("is_storage"))
        is_hy = bool(g.get("is_hydro"))
        size = g.get("capacity_size_MW") or 1.0
        minf = gf(g, "min_output_frac")
        capv = yP[i]

        if not is_uc:
            for t in range(T):
                m.addConstr(xinj[i][t] <= a[t] * capv)

        if is_st:
            ec = gf(g, "charge_efficiency", 1.0)
            ed = max(gf(g, "discharge_efficiency", 1.0), 1e-6)
            sd = gf(g, "self_discharge_rate", 0.0)
            for t in range(T):
                p = prevt(t)
                m.addConstr(xwdw[i][t] <= a[t] * capv)
                m.addConstr(xinj[i][t] + xwdw[i][t] <= capv)
                m.addConstr(ec * xwdw[i][t] + xsoc[i][p] <= yE[i])
                m.addConstr(xinj[i][t] <= ed * xsoc[i][p])
                m.addConstr(xsoc[i][t] <= yE[i])
                m.addConstr(xsoc[i][t] - (1.0 - sd) * xsoc[i][p]
                            - ec * xwdw[i][t] + (1.0 / ed) * xinj[i][t] == 0.0)
        elif is_hy:
            dur = gf(g, "duration_MWh_per_MW")
            rup = gf(g, "ramp_up_frac_per_hr", 1.0)
            rdn = gf(g, "ramp_dn_frac_per_hr", 1.0)
            for t in range(T):
                p = prevt(t)
                m.addConstr(xhyd[i][t] - xhyd[i][p] - a[t] * capv
                            + xinj[i][t] + xsp[i][t] == 0.0)
                m.addConstr(xhyd[i][t] <= dur * capv)
                if minf > 0:
                    m.addConstr(xinj[i][t] + xsp[i][t] >= minf * capv)
                if rup < 1.0:
                    m.addConstr(xinj[i][t] - xinj[i][p] <= rup * capv)
                if rdn < 1.0:
                    m.addConstr(xinj[i][p] - xinj[i][t] <= rdn * capv)
        elif is_uc:
            rup = gf(g, "ramp_up_frac_per_hr", 1.0)
            rdn = gf(g, "ramp_dn_frac_per_hr", 1.0)
            MU = max(1, min(int(gf(g, "min_up_time_hr", 1)), H))
            MD = max(1, min(int(gf(g, "min_down_time_hr", 1)), H))
            for t in range(T):
                p = prevt(t)
                b = (t // H) * H
                m.addConstr(size * xcom[i][t] <= capv)
                m.addConstr(size * xst[i][t] <= capv)
                m.addConstr(size * xsh[i][t] <= capv)
                m.addConstr(xinj[i][t] >= minf * size * xcom[i][t])
                m.addConstr(xinj[i][t] <= a[t] * size * xcom[i][t])
                m.addConstr(xcom[i][t] - xcom[i][p] - xst[i][t] + xsh[i][t] == 0)
                ru_eff = min(a[t], max(minf, rup))
                rd_eff = min(a[t], max(minf, rdn))
                m.addConstr(xinj[i][t] - xinj[i][p]
                            <= size * rup * (xcom[i][t] - xst[i][t])
                            + size * ru_eff * xst[i][t]
                            - size * minf * xsh[i][t])
                m.addConstr(xinj[i][p] - xinj[i][t]
                            <= size * rdn * (xcom[i][t] - xst[i][t])
                            + size * rd_eff * xsh[i][t]
                            - size * minf * xst[i][t])
                m.addConstr(xcom[i][t] >= gp.quicksum(
                    xst[i][b + (t - b - k) % H] for k in range(MU)))
                m.addConstr(size * xcom[i][t] + size * gp.quicksum(
                    xsh[i][b + (t - b - k) % H] for k in range(MD)) <= capv)
        else:
            rup = gf(g, "ramp_up_frac_per_hr", 1.0)
            rdn = gf(g, "ramp_dn_frac_per_hr", 1.0)
            if minf > 0:
                for t in range(T):
                    m.addConstr(xinj[i][t] >= minf * capv)
            if rup < 1.0 or rdn < 1.0:
                for t in range(T):
                    p = prevt(t)
                    if rup < 1.0:
                        m.addConstr(xinj[i][t] - xinj[i][p] <= rup * capv)
                    if rdn < 1.0:
                        m.addConstr(xinj[i][p] - xinj[i][t] <= rdn * capv)

    # ---------------- policies ----------------
    if policy.get("RPS_enabled"):
        xRPS = [m.addVar(lb=0.0, name=f"xRPS_{w}") for w in range(W)]
        rexpr = gp.LinExpr()
        for g in gens:
            if g.get("is_RPS_qualifying"):
                rexpr.addTerms(omega, xinj[g["id"]])
        m.addConstr(rexpr + gp.quicksum(xRPS)
                    >= gf(policy, "RPS_share") * D_tot)
        pen = gf(policy, "RPS_noncompliance_cost_per_MWh")
        for w in range(W):
            addobj(pen, xRPS[w])

    if policy.get("CO2_cap_enabled"):
        xCO2 = [m.addVar(lb=0.0, name=f"xCO2_{w}") for w in range(W)]
        eexpr = gp.LinExpr()
        for g in gens:
            co2 = gf(g, "co2_tons_per_MWh")
            if co2:
                i = g["id"]
                coefs = [omega[t] * co2 for t in range(T)]
                eexpr.addTerms(coefs, xinj[i])
                if g.get("is_storage"):
                    eexpr.addTerms(coefs, xwdw[i])
        capr = policy.get("CO2_cap_tons_per_MWh")
        cap = (capr if capr is not None else 0.05) * D_tot
        m.addConstr(eexpr - gp.quicksum(xCO2) <= cap)
        pen = gf(policy, "CO2_noncompliance_cost_per_ton")
        for w in range(W):
            addobj(pen, xCO2[w])

    m.setObjective(gp.LinExpr(obj_c, obj_v), GRB.MINIMIZE)
    m.update()

    allvars = m.getVars()
    names = m.getAttr("VarName", allvars)
    vtypes = m.getAttr("VType", allvars)

    # ---------------- solution construction ----------------
    def build_solution(valmap, objval):
        mv = {}
        for k in range(len(names)):
            nm = names[k]
            if nm.startswith("SPILL"):
                continue
            v = valmap.get(nm, 0.0)
            if vtypes[k] != GRB.CONTINUOUS:
                v = round(v)
            if abs(v) > 1e-6:
                mv[nm] = float(v)
        inv = []
        for g in gens:
            i = g["id"]
            size = g.get("capacity_size_MW") or 1.0
            nnew = int(round(valmap.get(f"yPnew_{i}", 0.0)))
            nret = int(round(valmap.get(f"yPret_{i}", 0.0)))
            tot = gf(g, "existing_capacity_MW") + size * (nnew - nret)
            for nm, val in ((f"yPnew_{i}", nnew), (f"yPret_{i}", nret),
                            (f"yP_{i}", tot)):
                if abs(val) > 1e-9:
                    mv[nm] = float(val)
                else:
                    mv.pop(nm, None)
            rec = {"generator_id": i,
                   "resource_type": g.get("resource_type", ""),
                   "zone": g["zone"],
                   "y_P_new": float(nnew),
                   "y_P_ret": float(nret),
                   "y_P_total_MW": float(tot)}
            if g.get("is_storage"):
                se = g.get("storage_capacity_size_MWh") or 1.0
                en = int(round(valmap.get(f"yEnew_{i}", 0.0)))
                er = int(round(valmap.get(f"yEret_{i}", 0.0)))
                et = gf(g, "existing_storage_capacity_MWh") + se * (en - er)
                for nm, val in ((f"yEnew_{i}", en), (f"yEret_{i}", er),
                                (f"yE_{i}", et)):
                    if abs(val) > 1e-9:
                        mv[nm] = float(val)
                    else:
                        mv.pop(nm, None)
                rec.update({"y_E_new": float(en), "y_E_ret": float(er),
                            "y_E_total_MWh": float(et)})
            inv.append(rec)
        td = []
        for l in lines:
            lid = l["id"]
            nn = int(round(valmap.get(f"yFnew_{lid}", 0.0)))
            if nn:
                mv[f"yFnew_{lid}"] = float(nn)
            else:
                mv.pop(f"yFnew_{lid}", None)
            td.append({"line_id": lid,
                       "y_F_new_MW": float(nn),
                       "total_capacity_MW": float(gf(l, "existing_capacity_MW") + nn)})
        return {"objective_value": float(objval),
                "model_variables": mv,
                "investment_decisions": inv,
                "transmission_decisions": td}

    m._best = float("inf")
    m._bestsol = None

    def cb(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj < model._best - 1e-6:
                model._best = obj
                try:
                    vals = model.cbGetSolution(allvars)
                    valmap = dict(zip(names, vals))
                    sol = build_solution(valmap, obj)
                    model._bestsol = sol
                    if logger:
                        logger.log_solution(obj, sol)
                except Exception:
                    if logger:
                        try:
                            logger.log(obj)
                        except Exception:
                            pass

    elapsed = time.time() - t0
    m.Params.TimeLimit = max(1.0, args.time_limit - elapsed - 5.0)

    try:
        m.optimize(cb)
    except Exception:
        pass

    sol = None
    try:
        if m.SolCount > 0:
            vals = m.getAttr("X", allvars)
            valmap = dict(zip(names, vals))
            sol = build_solution(valmap, m.ObjVal)
            if logger and m.ObjVal < m._best - 1e-6:
                try:
                    logger.log_solution(m.ObjVal, sol)
                except Exception:
                    pass
    except Exception:
        sol = None
    if sol is None:
        sol = m._bestsol
    if sol is None:
        sol = {"objective_value": 1e20,
               "model_variables": {},
               "investment_decisions": [],
               "transmission_decisions": []}

    with open(args.solution_path, "w") as f:
        json.dump(sol, f)


if __name__ == "__main__":
    main()
import argparse
import json
import math
import time

import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except Exception:
    SolutionLogger = None


def clean_int(v):
    if v is None:
        return 0.0
    if v < 0:
        v = 0.0
    r = round(v)
    return float(r) if abs(v - r) <= 1e-4 else float(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", required=True)
    ap.add_argument("--solution_path", required=True)
    ap.add_argument("--time_limit", type=int, default=3600)
    ap.add_argument("--log_path", default=None)
    args = ap.parse_args()

    t0 = time.time()
    logger = SolutionLogger(args.log_path, sense="minimize") if (args.log_path and SolutionLogger) else None

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    pp = data["problem_parameters"]
    H = int(pp["hours_per_subperiod"])
    T = int(pp["total_hours"])
    weights = data["subperiod_weights"]
    W = len(weights)
    wt = [weights[t // H] / float(H) for t in range(T)]

    gens = data["generators"]
    gparams = data["generator_parameters"]
    exist_cap = {int(k): float(v) for k, v in data["existing_capacity"].items()}
    demand = {int(z): lst for z, lst in data["demand"].items()}
    zones = sorted(demand.keys())
    segments = data["consumer_segments"]
    lines = data["transmission_lines"]
    avail = {int(k): v for k, v in data["availability"].items()}
    hyd_inflow = {int(k): v for k, v in data.get("hydro_inflow", {}).items()}
    pol = data.get("policy", {}) or {}
    ptype = str(pol.get("type", "REF")).upper()

    # ---- cluster info ----
    info = {}
    for g in gens:
        cid = int(g["cluster_id"])
        par = gparams[g["resource_type"]]
        is_storage = bool(g.get("is_storage", False))
        is_hydro = bool(g.get("is_hydro", False))
        is_uc = bool(g.get("is_UC", False)) and not is_storage and not is_hydro
        size = float(g.get("capacity_size_MW", par.get("capacity_size_MW", 1.0)) or 1.0)
        inc_P = size if is_uc else 1.0
        exP = float(exist_cap.get(cid, 0.0))
        d = {
            "gen": g, "par": par, "cid": cid, "zone": int(g["zone"]),
            "is_storage": is_storage, "is_hydro": is_hydro, "is_uc": is_uc,
            "is_rps": bool(g.get("is_RPS_qualifying", False)),
            "can_retire": bool(g.get("can_retire", True)),
            "size": size, "inc_P": inc_P, "exP": exP,
            "mf": float(par.get("min_output_frac", 0.0) or 0.0),
            "ru": float(par.get("ramp_up_rate_pct_per_hr", 1.0) or 1.0),
            "rd": float(par.get("ramp_down_rate_pct_per_hr", 1.0) or 1.0),
            "ut": int(par.get("min_up_time_hr", 0) or 0),
            "dt": int(par.get("min_down_time_hr", 0) or 0),
            "vc": float(par.get("variable_cost_per_MWh", 0.0) or 0.0),
            "fom": float(par.get("fixed_OM_cost_per_MW_yr", 0.0) or 0.0),
            "inv": float(par.get("investment_cost_per_MW_yr", 0.0) or 0.0),
            "suc": float(par.get("start_up_cost", 0.0) or 0.0),
            "emis": float(par.get("CO2_emission_tons_per_MWh", 0.0) or 0.0),
            "maxcap": float(par.get("max_capacity_MW", 1e12) or 1e12),
            "av": avail.get(cid, [1.0] * T),
        }
        if is_storage:
            esize = float(par.get("energy_capacity_size_MWh", 1.0) or 1.0)
            d["inc_E"] = esize if bool(g.get("is_UC", False)) else 1.0
            d["exE"] = (exP / size * esize) if size > 0 else 0.0
            d["effc"] = float(par.get("charging_efficiency", 1.0) or 1.0)
            d["effd"] = float(par.get("discharging_efficiency", 1.0) or 1.0)
            d["sd"] = float(par.get("self_discharge_rate_pct", 0.0) or 0.0)
            d["mindur"] = float(par.get("min_duration_MWh_per_MW", 0.0) or 0.0)
            d["maxdur"] = float(par.get("max_duration_MWh_per_MW", 1e9) or 1e9)
            d["e_inv"] = float(par.get("energy_investment_cost_per_MWh_yr", 0.0) or 0.0)
            d["e_fom"] = float(par.get("energy_fixed_OM_cost_per_MWh_yr", 0.0) or 0.0)
            d["maxE"] = float(par.get("max_energy_capacity_MWh", 1e12) or 1e12)
        if is_hydro:
            d["dur"] = float(par.get("duration_MWh_per_MW", 0.0) or 0.0)
            d["e_inv"] = float(par.get("energy_investment_cost_per_MWh_yr", 0.0) or 0.0)
            d["e_fom"] = float(par.get("energy_fixed_OM_cost_per_MWh_yr", 0.0) or 0.0)
            d["inflow"] = hyd_inflow.get(cid, [0.0] * T)
        info[cid] = d

    cids = sorted(info.keys())
    st_ids = [c for c in cids if info[c]["is_storage"]]
    hy_ids = [c for c in cids if info[c]["is_hydro"]]
    uc_ids = [c for c in cids if info[c]["is_uc"]]

    # ---- model ----
    m = gp.Model("cem")
    m.Params.OutputFlag = 0
    m.Params.Seed = 0
    m.Params.MIPGap = 1e-4
    m.Params.NumericFocus = 0
    m.Params.Threads = 1

    obj_expr = gp.LinExpr()

    def addobj(var, c):
        if c:
            obj_expr.addTerms(c, var)

    # investment variables
    yPn, yPr, yP = {}, {}, {}
    for c in cids:
        d = info[c]
        ub_new = math.floor(d["maxcap"] / d["inc_P"] + 1e-9)
        yPn[c] = m.addVar(vtype=GRB.INTEGER, lb=0, ub=max(0, ub_new), name=f"y_P_new[{c}]")
        ub_ret = math.floor(d["exP"] / d["inc_P"] + 1e-9) if d["can_retire"] else 0
        yPr[c] = m.addVar(vtype=GRB.INTEGER, lb=0, ub=max(0, ub_ret), name=f"y_P_ret[{c}]")
        yP[c] = m.addVar(lb=0, name=f"y_P[{c}]")
        m.addConstr(yP[c] == d["exP"] + d["inc_P"] * (yPn[c] - yPr[c]))
        addobj(yPn[c], d["inc_P"] * d["inv"])
        addobj(yP[c], d["fom"])
        if d["is_hydro"]:
            addobj(yPn[c], d["e_inv"] * d["dur"] * d["inc_P"])
            addobj(yP[c], d["e_fom"] * d["dur"])

    yEn, yEr, yE = {}, {}, {}
    for c in st_ids:
        d = info[c]
        ub_new = math.floor(d["maxE"] / d["inc_E"] + 1e-9)
        yEn[c] = m.addVar(vtype=GRB.INTEGER, lb=0, ub=max(0, ub_new), name=f"y_E_new[{c}]")
        ub_ret = math.floor(d["exE"] / d["inc_E"] + 1e-9) if d["can_retire"] else 0
        yEr[c] = m.addVar(vtype=GRB.INTEGER, lb=0, ub=max(0, ub_ret), name=f"y_E_ret[{c}]")
        yE[c] = m.addVar(lb=0, name=f"y_E[{c}]")
        m.addConstr(yE[c] == d["exE"] + d["inc_E"] * (yEn[c] - yEr[c]))
        m.addConstr(yE[c] >= d["mindur"] * yP[c])
        m.addConstr(yE[c] <= d["maxdur"] * yP[c])
        addobj(yEn[c], d["inc_E"] * d["e_inv"])
        addobj(yE[c], d["e_fom"])

    yFn, yF = {}, {}
    line_ids = []
    for L in lines:
        lid = int(L["line_id"])
        line_ids.append(lid)
        ex = float(L["existing_capacity_MW"])
        mx = float(L["max_capacity_MW"])
        ub = max(0, math.floor(mx - ex + 1e-9))
        yFn[lid] = m.addVar(vtype=GRB.INTEGER, lb=0, ub=ub, name=f"y_F_new[{lid}]")
        yF[lid] = m.addVar(lb=0, name=f"y_F[{lid}]")
        m.addConstr(yF[lid] == ex + yFn[lid])
        addobj(yFn[lid], float(L["investment_cost_per_MW_yr"]))

    # operations variables
    inj = m.addVars([(c, t) for c in cids for t in range(T)], name="x_inj")
    wdw = m.addVars([(c, t) for c in st_ids for t in range(T)], name="x_wdw")
    soc = m.addVars([(c, t) for c in st_ids for t in range(T)], name="x_soc")
    lvl = m.addVars([(c, t) for c in hy_ids for t in range(T)], name="x_lvl")
    spill = m.addVars([(c, t) for c in hy_ids for t in range(T)], name="x_spill")
    commit = m.addVars([(c, t) for c in uc_ids for t in range(T)], name="x_commit")
    start = m.addVars([(c, t) for c in uc_ids for t in range(T)], name="x_start")
    shut = m.addVars([(c, t) for c in uc_ids for t in range(T)], name="x_shut")

    nse_idx, nse_ub = [], []
    for s in segments:
        sid, z, frac = int(s["segment_id"]), int(s["zone"]), float(s["max_nse_fraction"])
        dz = demand[z]
        for t in range(T):
            nse_idx.append((sid, z, t))
            nse_ub.append(max(0.0, frac * dz[t]))
    nse = m.addVars(nse_idx, ub=nse_ub, name="x_nse")
    for s in segments:
        sid, z, cost = int(s["segment_id"]), int(s["zone"]), float(s["cost_per_MWh"])
        for t in range(T):
            addobj(nse[sid, z, t], wt[t] * cost)

    flow = m.addVars([(l, t) for l in line_ids for t in range(T)], lb=-GRB.INFINITY, name="x_flow")
    for l in line_ids:
        for t in range(T):
            m.addConstr(flow[l, t] <= yF[l])
            m.addConstr(flow[l, t] >= -yF[l])

    def prev(t):
        return t - 1 if (t % H) else t + H - 1

    def window(t, K):
        base = (t // H) * H
        off = t - base
        return [base + ((off - s) % H) for s in range(K)]

    # per-cluster operational constraints
    for c in cids:
        d = info[c]
        av = d["av"]
        p = yP[c]
        for t in range(T):
            addobj(inj[c, t], wt[t] * d["vc"])
        if d["is_uc"]:
            sz, mf, ru, rd = d["size"], d["mf"], d["ru"], d["rd"]
            ut, dt = min(d["ut"], H), min(d["dt"], H)
            for t in range(T):
                tm = prev(t)
                m.addConstr(sz * commit[c, t] <= p)
                m.addConstr(sz * start[c, t] <= p)
                m.addConstr(sz * shut[c, t] <= p)
                m.addConstr(inj[c, t] <= sz * av[t] * commit[c, t])
                if mf > 0:
                    m.addConstr(inj[c, t] >= sz * mf * commit[c, t])
                m.addConstr(commit[c, t] - commit[c, tm] == start[c, t] - shut[c, t])
                rue = sz * min(av[t], max(mf, ru))
                rde = sz * min(av[t], max(mf, rd))
                m.addConstr(inj[c, t] - inj[c, tm]
                            <= ru * sz * (commit[c, t] - start[c, t])
                            + rue * start[c, t] - mf * sz * shut[c, t])
                m.addConstr(inj[c, tm] - inj[c, t]
                            <= rd * sz * (commit[c, t] - start[c, t])
                            - mf * sz * start[c, t] + rde * shut[c, t])
                if ut > 0:
                    m.addConstr(commit[c, t] >= gp.quicksum(start[c, h] for h in window(t, ut)))
                if dt > 0:
                    m.addConstr(sz * commit[c, t] + sz * gp.quicksum(shut[c, h] for h in window(t, dt)) <= p)
                addobj(start[c, t], wt[t] * d["suc"])
        elif d["is_storage"]:
            effc, effd, sd = d["effc"], d["effd"], d["sd"]
            e = yE[c]
            for t in range(T):
                tm = prev(t)
                m.addConstr(inj[c, t] + wdw[c, t] <= p)
                m.addConstr(wdw[c, t] <= av[t] * p)
                m.addConstr(inj[c, t] <= av[t] * p)
                m.addConstr(soc[c, t] <= e)
                m.addConstr(soc[c, t] == (1.0 - sd) * soc[c, tm] + effc * wdw[c, t] - inj[c, t] / effd)
                m.addConstr(effc * wdw[c, t] <= e - soc[c, tm])
                m.addConstr(inj[c, t] / effd <= soc[c, tm])
                addobj(wdw[c, t], wt[t] * d["vc"])
        elif d["is_hydro"]:
            dur, mf = d["dur"], d["mf"]
            infl = d["inflow"]
            for t in range(T):
                tm = prev(t)
                m.addConstr(inj[c, t] <= av[t] * p)
                m.addConstr(lvl[c, t] <= dur * p)
                m.addConstr(lvl[c, t] == lvl[c, tm] + infl[t] * p - inj[c, t] - spill[c, t])
                if mf > 0:
                    m.addConstr(inj[c, t] + spill[c, t] >= mf * p)
        else:
            mf, ru, rd = d["mf"], d["ru"], d["rd"]
            for t in range(T):
                m.addConstr(inj[c, t] <= av[t] * p)
                if mf > 0:
                    m.addConstr(inj[c, t] >= mf * p)
                if ru < 1.0:
                    m.addConstr(inj[c, t] - inj[c, prev(t)] <= ru * p)
                if rd < 1.0:
                    m.addConstr(inj[c, prev(t)] - inj[c, t] <= rd * p)

    # zonal balance
    for z in zones:
        gz = [c for c in cids if info[c]["zone"] == z]
        stz = [c for c in st_ids if info[c]["zone"] == z]
        lin = [int(L["line_id"]) for L in lines if int(L["to_zone"]) == z]
        lout = [int(L["line_id"]) for L in lines if int(L["from_zone"]) == z]
        segz = [int(s["segment_id"]) for s in segments if int(s["zone"]) == z]
        dz = demand[z]
        for t in range(T):
            lhs = gp.LinExpr()
            if gz:
                lhs.addTerms([1.0] * len(gz), [inj[c, t] for c in gz])
            if stz:
                lhs.addTerms([-1.0] * len(stz), [wdw[c, t] for c in stz])
            if lin:
                lhs.addTerms([1.0] * len(lin), [flow[l, t] for l in lin])
            if lout:
                lhs.addTerms([-1.0] * len(lout), [flow[l, t] for l in lout])
            if segz:
                lhs.addTerms([1.0] * len(segz), [nse[s, z, t] for s in segz])
            m.addConstr(lhs == dz[t])

    # policies
    TWD = sum(wt[t] * sum(demand[z][t] for z in zones) for t in range(T))
    xRPS, xCO2 = {}, {}
    if ptype == "RPS":
        xRPS = m.addVars(range(W), name="x_RPS")
        pen = float(pol.get("RPS_noncompliance_cost_per_MWh", 0.0) or 0.0)
        share = float(pol.get("RPS_share", 0.0) or 0.0)
        expr = gp.LinExpr()
        for c in cids:
            if info[c]["is_rps"]:
                expr.addTerms(wt, [inj[c, t] for t in range(T)])
        expr.addTerms([1.0] * W, [xRPS[w] for w in range(W)])
        m.addConstr(expr >= share * TWD)
        for w in range(W):
            addobj(xRPS[w], pen)
    if ptype == "CO2":
        xCO2 = m.addVars(range(W), name="x_CO2")
        pen = float(pol.get("CO2_noncompliance_cost_per_ton", 0.0) or 0.0)
        cap = float(pol.get("CO2_cap_tons_per_MWh", 0.0) or 0.0)
        expr = gp.LinExpr()
        for c in cids:
            em = info[c]["emis"]
            if em:
                expr.addTerms([wt[t] * em for t in range(T)], [inj[c, t] for t in range(T)])
                if c in st_ids:
                    expr.addTerms([wt[t] * em for t in range(T)], [wdw[c, t] for t in range(T)])
        expr.addTerms([-1.0] * W, [xCO2[w] for w in range(W)])
        m.addConstr(expr <= cap * TWD)
        for w in range(W):
            addobj(xCO2[w], pen)

    m.setObjective(obj_expr, GRB.MINIMIZE)
    m.update()

    all_vars = m.getVars()
    names = m.getAttr("VarName", all_vars)
    int_names = set()
    for c in cids:
        int_names.add(f"y_P_new[{c}]")
        int_names.add(f"y_P_ret[{c}]")
    for c in st_ids:
        int_names.add(f"y_E_new[{c}]")
        int_names.add(f"y_E_ret[{c}]")
    for l in line_ids:
        int_names.add(f"y_F_new[{l}]")

    def build_solution(nv, objval):
        mv = {}
        for k, v in nv.items():
            if k in int_names:
                v = clean_int(v)
            if abs(v) > 1e-6:
                mv[k] = v
        invs = {}
        for c in cids:
            d = info[c]
            nu = clean_int(nv.get(f"y_P_new[{c}]", 0.0))
            rt = clean_int(nv.get(f"y_P_ret[{c}]", 0.0))
            invs[str(c)] = {
                "resource_type": d["gen"]["resource_type"],
                "zone": d["zone"],
                "new_units": nu,
                "new_MW": d["inc_P"] * nu,
                "retired_units": rt,
                "retired_MW": d["inc_P"] * rt,
                "total_capacity_MW": d["exP"] + d["inc_P"] * (nu - rt),
            }
        se = {}
        for c in st_ids:
            d = info[c]
            nu = clean_int(nv.get(f"y_E_new[{c}]", 0.0))
            rt = clean_int(nv.get(f"y_E_ret[{c}]", 0.0))
            se[str(c)] = {
                "new_units": nu,
                "new_MWh": d["inc_E"] * nu,
                "total_energy_MWh": d["exE"] + d["inc_E"] * (nu - rt),
            }
        tx = {}
        for L in lines:
            lid = int(L["line_id"])
            nu = clean_int(nv.get(f"y_F_new[{lid}]", 0.0))
            tx[str(lid)] = {
                "new_MW": nu,
                "total_capacity_MW": float(L["existing_capacity_MW"]) + nu,
            }
        return {
            "objective_value": float(objval),
            "model_variables": mv,
            "investments": invs,
            "storage_energy_investments": se,
            "transmission_investments": tx,
        }

    best = [float("inf")]

    def cb(mod, where):
        if where == GRB.Callback.MIPSOL:
            o = mod.cbGet(GRB.Callback.MIPSOL_OBJ)
            if o < best[0] - 1e-6:
                best[0] = o
                if logger:
                    try:
                        vals = mod.cbGetSolution(all_vars)
                        nv = dict(zip(names, vals))
                        logger.log_solution(o, build_solution(nv, o))
                    except Exception:
                        try:
                            logger.log(o)
                        except Exception:
                            pass

    elapsed = time.time() - t0
    m.Params.TimeLimit = max(5.0, args.time_limit - elapsed - 10.0)
    try:
        m.optimize(cb)
    except gp.GurobiError:
        m.optimize()

    if m.SolCount > 0:
        vals = m.getAttr("X", all_vars)
        nv = dict(zip(names, vals))
        sol = build_solution(nv, m.ObjVal)
        if logger:
            try:
                logger.log_solution(m.ObjVal, sol)
            except Exception:
                pass
    else:
        nv = {}
        sol = build_solution(nv, 1e18)

    with open(args.solution_path, "w") as f:
        json.dump(sol, f)


if __name__ == "__main__":
    main()
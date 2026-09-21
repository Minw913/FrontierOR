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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--instance_path', required=True)
    ap.add_argument('--solution_path', required=True)
    ap.add_argument('--time_limit', type=int, default=3600)
    ap.add_argument('--log_path', default=None)
    args = ap.parse_args()
    t0 = time.time()

    logger = SolutionLogger(args.log_path, sense="minimize") if (args.log_path and SolutionLogger) else None

    with open(args.instance_path) as f:
        data = json.load(f)

    sub = data['subperiods']
    W = sub['num_subperiods']
    H = sub['hours_per_subperiod']
    weights = sub['subperiod_weights']
    T = W * H

    gens = data['generators']
    lines = data['transmission_lines']
    demand = {int(z): v for z, v in data['demand'].items()}
    zones = sorted(demand.keys())
    segs = data.get('nse_segments', [])
    profs = data['availability_profiles']

    def get_profile(gid):
        p = profs.get(str(gid), profs.get(gid, None))
        if p is None:
            return [1.0] * T
        if p.get('type') == 'constant':
            return [float(p.get('value', 1.0))] * T
        return [float(x) for x in p['values']]

    av = {g['id']: get_profile(g['id']) for g in gens}

    # hour weights: subperiod weight / hours_per_subperiod
    omega = [weights[t // H] / float(H) for t in range(T)]
    TWD = 0.0
    for z in zones:
        dz = demand[z]
        for t in range(T):
            TWD += omega[t] * dz[t]

    policy = data.get('policy', {})
    rps_on = bool(policy.get('RPS_enabled', False))
    co2_on = bool(policy.get('CO2_cap_enabled', False))
    rps_share = float(policy.get('RPS_share') or 0.0)
    rps_pen = float(policy.get('RPS_noncompliance_cost_per_MWh') or 0.0)
    co2_pen = float(policy.get('CO2_noncompliance_cost_per_ton') or 0.0)
    cap_rate = policy.get('CO2_cap_tons_per_MWh')
    if cap_rate is None:
        cap_rate = 0.05
    co2_cap = float(cap_rate) * TWD

    # circular previous-hour index within subperiod
    prv = [0] * T
    for t in range(T):
        w = t // H
        base = w * H
        prv[t] = base + (t - base - 1) % H

    m = gp.Model("capex")
    m.Params.Seed = 0
    m.Params.MIPGap = 1e-4
    m.Params.NumericFocus = 0
    m.Params.Threads = 1
    m.Params.OutputFlag = 1

    G = [g['id'] for g in gens]
    gd = {g['id']: g for g in gens}
    STOR = [g['id'] for g in gens if g.get('is_storage', False)]
    HYDRO = [g['id'] for g in gens if g.get('is_hydro', False)]
    UC = [g['id'] for g in gens if g.get('is_UC', False)]
    RPSQ = [g['id'] for g in gens if g.get('is_RPS_qualifying', False)]
    NONUC = [i for i in G if i not in set(UC)]

    # ---- Investment variables ----
    yPn, yPr = {}, {}
    yEn, yEr = {}, {}
    capX = {}   # power capacity expression
    capConst = {}
    ecapX = {}
    ecapConst = {}
    for g in gens:
        i = g['id']
        size = float(g.get('capacity_size_MW', 1.0) or 1.0)
        ex = float(g.get('existing_capacity_MW', 0.0) or 0.0)
        mx = float(g.get('max_capacity_MW', 0.0) or 0.0)
        ub_new = math.floor(mx / size + 1e-9) if size > 0 else 0
        ub_ret = math.floor(ex / size + 1e-9) if (g.get('can_retire', True) and size > 0) else 0
        yPn[i] = m.addVar(vtype=GRB.INTEGER, lb=0, ub=max(0, ub_new), name=f"y_P_new[{i}]")
        yPr[i] = m.addVar(vtype=GRB.INTEGER, lb=0, ub=max(0, ub_ret), name=f"y_P_ret[{i}]")
        capX[i] = ex + size * yPn[i] - size * yPr[i]
        capConst[i] = (ub_new == 0 and ub_ret == 0)
        if g.get('is_storage', False):
            esize = float(g.get('storage_capacity_size_MWh', 1.0) or 1.0)
            eex = float(g.get('existing_storage_capacity_MWh', 0.0) or 0.0)
            emx = float(g.get('max_storage_capacity_MWh', 0.0) or 0.0)
            eub_new = math.floor(emx / esize + 1e-9) if esize > 0 else 0
            eub_ret = math.floor(eex / esize + 1e-9) if (g.get('can_retire', True) and esize > 0) else 0
            yEn[i] = m.addVar(vtype=GRB.INTEGER, lb=0, ub=max(0, eub_new), name=f"y_E_new[{i}]")
            yEr[i] = m.addVar(vtype=GRB.INTEGER, lb=0, ub=max(0, eub_ret), name=f"y_E_ret[{i}]")
            ecapX[i] = eex + esize * yEn[i] - esize * yEr[i]
            ecapConst[i] = (eub_new == 0 and eub_ret == 0)

    yFn = {}
    lcapX = {}
    for l in lines:
        li = l['id']
        ub = float(l.get('max_new_capacity_MW', 0.0) or 0.0)
        yFn[li] = m.addVar(vtype=GRB.INTEGER, lb=0, ub=max(0.0, ub), name=f"y_F_new[{li}]")
        lcapX[li] = float(l.get('existing_capacity_MW', 0.0) or 0.0) + yFn[li]

    # ---- Operational variables ----
    gen = {}
    for i in G:
        for t in range(T):
            gen[i, t] = m.addVar(lb=0.0, name=f"gen[{i},{t}]")
    wdr, soc = {}, {}
    for i in STOR:
        for t in range(T):
            wdr[i, t] = m.addVar(lb=0.0, name=f"withdraw[{i},{t}]")
            soc[i, t] = m.addVar(lb=0.0, name=f"soc[{i},{t}]")
    lvl, spl = {}, {}
    for i in HYDRO:
        for t in range(T):
            lvl[i, t] = m.addVar(lb=0.0, name=f"level[{i},{t}]")
            spl[i, t] = m.addVar(lb=0.0, name=f"spill[{i},{t}]")
    cmt, stt, sht = {}, {}, {}
    for i in UC:
        g = gd[i]
        size = float(g.get('capacity_size_MW', 1.0) or 1.0)
        maxcap = float(g.get('existing_capacity_MW', 0.0) or 0.0) + float(g.get('max_capacity_MW', 0.0) or 0.0)
        ubU = math.floor(maxcap / size + 1e-9) if size > 0 else 0
        for t in range(T):
            cmt[i, t] = m.addVar(vtype=GRB.INTEGER, lb=0, ub=ubU, name=f"commit[{i},{t}]")
            stt[i, t] = m.addVar(vtype=GRB.INTEGER, lb=0, ub=ubU, name=f"start[{i},{t}]")
            sht[i, t] = m.addVar(vtype=GRB.INTEGER, lb=0, ub=ubU, name=f"shut[{i},{t}]")
    flow = {}
    for l in lines:
        li = l['id']
        fub = float(l.get('existing_capacity_MW', 0.0) or 0.0) + float(l.get('max_new_capacity_MW', 0.0) or 0.0)
        for t in range(T):
            flow[li, t] = m.addVar(lb=-fub, ub=fub, name=f"flow[{li},{t}]")
    nse = {}
    for s in segs:
        sidx = s['segment']
        mf = float(s.get('max_frac', 0.0) or 0.0)
        for z in zones:
            dz = demand[z]
            for t in range(T):
                ub = max(0.0, mf * dz[t])
                nse[sidx, z, t] = m.addVar(lb=0.0, ub=ub, name=f"nse[{sidx},{z},{t}]")
    rps_sl, co2_sl = {}, {}
    if rps_on:
        for w in range(W):
            rps_sl[w] = m.addVar(lb=0.0, name=f"rps_slack[{w}]")
    if co2_on:
        for w in range(W):
            co2_sl[w] = m.addVar(lb=0.0, name=f"co2_slack[{w}]")

    m.update()

    # ---- Constraints ----
    # storage duration coupling
    for i in STOR:
        g = gd[i]
        mind = float(g.get('min_duration_MWh_per_MW', 0.0) or 0.0)
        maxd = float(g.get('max_duration_MWh_per_MW', 1e9) or 1e9)
        m.addConstr(ecapX[i] >= mind * capX[i], name=f"durmin[{i}]")
        m.addConstr(ecapX[i] <= maxd * capX[i], name=f"durmax[{i}]")

    # power balance
    zgen = {z: [] for z in zones}
    zstor = {z: [] for z in zones}
    for g in gens:
        zgen[g['zone']].append(g['id'])
        if g.get('is_storage', False):
            zstor[g['zone']].append(g['id'])
    zout = {z: [] for z in zones}
    zin = {z: [] for z in zones}
    for l in lines:
        zout[l['from_zone']].append(l['id'])
        zin[l['to_zone']].append(l['id'])
    seg_ids = [s['segment'] for s in segs]
    for z in zones:
        dz = demand[z]
        gl, sl_, ol, il = zgen[z], zstor[z], zout[z], zin[z]
        for t in range(T):
            expr = gp.LinExpr()
            for i in gl:
                expr.add(gen[i, t])
            for i in sl_:
                expr.add(wdr[i, t], -1.0)
            for li in ol:
                expr.add(flow[li, t], -1.0)
            for li in il:
                expr.add(flow[li, t], 1.0)
            for sidx in seg_ids:
                expr.add(nse[sidx, z, t])
            m.addConstr(expr == dz[t], name=f"bal[{z},{t}]")

    # non-UC max output; min output; ramps
    for i in NONUC:
        g = gd[i]
        a = av[i]
        is_st = g.get('is_storage', False)
        is_hy = g.get('is_hydro', False)
        cconst = capConst[i]
        cval = float(g.get('existing_capacity_MW', 0.0) or 0.0) if cconst else None
        for t in range(T):
            if cconst:
                gen[i, t].UB = min(gen[i, t].UB, a[t] * cval) if gen[i, t].UB < GRB.INFINITY else a[t] * cval
            else:
                m.addConstr(gen[i, t] <= a[t] * capX[i], name=f"maxout[{i},{t}]")
        minf = float(g.get('min_output_frac', 0.0) or 0.0)
        if (not is_st) and (not is_hy) and minf > 0:
            for t in range(T):
                m.addConstr(gen[i, t] >= minf * capX[i], name=f"minout[{i},{t}]")
        if not is_st:
            ru = float(g.get('ramp_up_frac_per_hr', 1.0) or 1.0)
            rd = float(g.get('ramp_dn_frac_per_hr', 1.0) or 1.0)
            if ru < 1.0 - 1e-12:
                for t in range(T):
                    m.addConstr(gen[i, t] - gen[i, prv[t]] <= ru * capX[i], name=f"rup[{i},{t}]")
            if rd < 1.0 - 1e-12:
                for t in range(T):
                    m.addConstr(gen[i, prv[t]] - gen[i, t] <= rd * capX[i], name=f"rdn[{i},{t}]")

    # storage
    for i in STOR:
        g = gd[i]
        a = av[i]
        ec = float(g.get('charge_efficiency', 1.0) or 1.0)
        ed = float(g.get('discharge_efficiency', 1.0) or 1.0)
        sd = float(g.get('self_discharge_rate', 0.0) or 0.0)
        for t in range(T):
            tp = prv[t]
            m.addConstr(wdr[i, t] <= a[t] * capX[i], name=f"maxchg[{i},{t}]")
            m.addConstr(gen[i, t] + wdr[i, t] <= capX[i], name=f"pcap[{i},{t}]")
            m.addConstr(ec * wdr[i, t] <= ecapX[i] - soc[i, tp], name=f"chgroom[{i},{t}]")
            m.addConstr(gen[i, t] <= ed * soc[i, tp], name=f"disav[{i},{t}]")
            m.addConstr(soc[i, t] <= ecapX[i], name=f"socmax[{i},{t}]")
            m.addConstr(
                soc[i, t] == (1.0 - sd) * soc[i, tp] + ec * wdr[i, t] - (1.0 / ed) * gen[i, t],
                name=f"socbal[{i},{t}]")

    # hydro
    for i in HYDRO:
        g = gd[i]
        a = av[i]
        dur = float(g.get('duration_MWh_per_MW', 0.0) or 0.0)
        minf = float(g.get('min_output_frac', 0.0) or 0.0)
        for t in range(T):
            tp = prv[t]
            m.addConstr(
                lvl[i, t] == lvl[i, tp] + a[t] * capX[i] - gen[i, t] - spl[i, t],
                name=f"resbal[{i},{t}]")
            m.addConstr(lvl[i, t] <= dur * capX[i], name=f"resmax[{i},{t}]")
            if minf > 0:
                m.addConstr(gen[i, t] + spl[i, t] >= minf * capX[i], name=f"hymin[{i},{t}]")

    # unit commitment
    for i in UC:
        g = gd[i]
        a = av[i]
        size = float(g.get('capacity_size_MW', 1.0) or 1.0)
        minf = float(g.get('min_output_frac', 0.0) or 0.0)
        ru = float(g.get('ramp_up_frac_per_hr', 1.0) or 1.0)
        rd = float(g.get('ramp_dn_frac_per_hr', 1.0) or 1.0)
        up = int(g.get('min_up_time_hr', 1) or 1)
        dn = int(g.get('min_down_time_hr', 1) or 1)
        up = max(1, min(up, H))
        dn = max(1, min(dn, H))
        nunits = (float(g.get('existing_capacity_MW', 0.0) or 0.0) / size) + yPn[i] - yPr[i]
        for t in range(T):
            tp = prv[t]
            m.addConstr(size * cmt[i, t] <= capX[i], name=f"cmtcap[{i},{t}]")
            m.addConstr(size * stt[i, t] <= capX[i], name=f"sttcap[{i},{t}]")
            m.addConstr(size * sht[i, t] <= capX[i], name=f"shtcap[{i},{t}]")
            if minf > 0:
                m.addConstr(gen[i, t] >= minf * size * cmt[i, t], name=f"ucmin[{i},{t}]")
            m.addConstr(gen[i, t] <= a[t] * size * cmt[i, t], name=f"ucmax[{i},{t}]")
            m.addConstr(cmt[i, t] - cmt[i, tp] == stt[i, t] - sht[i, t], name=f"cmtbal[{i},{t}]")
            cu = min(a[t], max(minf, ru))
            cd = min(a[t], max(minf, rd))
            m.addConstr(
                gen[i, t] - gen[i, tp]
                <= size * ru * (cmt[i, t] - stt[i, t]) + size * cu * stt[i, t] - size * minf * sht[i, t],
                name=f"ucrup[{i},{t}]")
            m.addConstr(
                gen[i, tp] - gen[i, t]
                <= size * rd * (cmt[i, t] - stt[i, t]) + size * cd * sht[i, t] - size * minf * stt[i, t],
                name=f"ucrdn[{i},{t}]")
            # min up / min down (circular within subperiod)
            idx = t
            sup = gp.LinExpr()
            for _ in range(up):
                sup.add(stt[i, idx])
                idx = prv[idx]
            m.addConstr(cmt[i, t] >= sup, name=f"minup[{i},{t}]")
            idx = t
            sdn = gp.LinExpr()
            for _ in range(dn):
                sdn.add(sht[i, idx])
                idx = prv[idx]
            m.addConstr(cmt[i, t] + sdn <= nunits, name=f"mindn[{i},{t}]")

    # transmission flow limits vs expandable capacity
    for l in lines:
        li = l['id']
        if yFn[li].UB > 1e-9:
            for t in range(T):
                m.addConstr(flow[li, t] <= lcapX[li], name=f"fpos[{li},{t}]")
                m.addConstr(-flow[li, t] <= lcapX[li], name=f"fneg[{li},{t}]")

    # RPS
    if rps_on:
        expr = gp.LinExpr()
        for i in RPSQ:
            for t in range(T):
                expr.add(gen[i, t], omega[t])
        for w in range(W):
            expr.add(rps_sl[w])
        m.addConstr(expr >= rps_share * TWD, name="rps")

    # CO2 cap
    if co2_on:
        expr = gp.LinExpr()
        for g in gens:
            i = g['id']
            e = float(g.get('co2_tons_per_MWh', 0.0) or 0.0)
            if e != 0.0:
                for t in range(T):
                    expr.add(gen[i, t], omega[t] * e)
                if i in set(STOR):
                    for t in range(T):
                        expr.add(wdr[i, t], omega[t] * e)
        for w in range(W):
            expr.add(co2_sl[w], -1.0)
        m.addConstr(expr <= co2_cap, name="co2cap")

    # ---- Objective ----
    obj = gp.LinExpr()
    for g in gens:
        i = g['id']
        size = float(g.get('capacity_size_MW', 1.0) or 1.0)
        inv = float(g.get('inv_cost_per_MW_yr', 0.0) or 0.0)
        fom = float(g.get('fom_cost_per_MW_yr', 0.0) or 0.0)
        obj.add(yPn[i], inv * size)
        obj += fom * capX[i]
        if g.get('is_hydro', False):
            dur = float(g.get('duration_MWh_per_MW', 0.0) or 0.0)
            invE = float(g.get('inv_cost_energy_per_MWh_yr', 0.0) or 0.0)
            fomE = float(g.get('fom_cost_energy_per_MWh_yr', 0.0) or 0.0)
            obj.add(yPn[i], invE * dur * size)
            obj += fomE * dur * capX[i]
        if g.get('is_storage', False):
            esize = float(g.get('storage_capacity_size_MWh', 1.0) or 1.0)
            invE = float(g.get('inv_cost_energy_per_MWh_yr', 0.0) or 0.0)
            fomE = float(g.get('fom_cost_energy_per_MWh_yr', 0.0) or 0.0)
            obj.add(yEn[i], invE * esize)
            obj += fomE * ecapX[i]
        vc = float(g.get('var_cost_per_MWh', 0.0) or 0.0)
        if vc != 0.0:
            for t in range(T):
                obj.add(gen[i, t], omega[t] * vc)
            if g.get('is_storage', False):
                for t in range(T):
                    obj.add(wdr[i, t], omega[t] * vc)
        if g.get('is_UC', False):
            sc = float(g.get('start_cost_per_unit', 0.0) or 0.0)
            if sc != 0.0:
                for t in range(T):
                    obj.add(stt[i, t], omega[t] * sc)
    for l in lines:
        obj.add(yFn[l['id']], float(l.get('inv_cost_per_MW_yr', 0.0) or 0.0))
    for s in segs:
        sidx = s['segment']
        c = float(s.get('cost_per_MWh', 0.0) or 0.0)
        for z in zones:
            for t in range(T):
                obj.add(nse[sidx, z, t], omega[t] * c)
    if rps_on:
        for w in range(W):
            obj.add(rps_sl[w], rps_pen)
    if co2_on:
        for w in range(W):
            obj.add(co2_sl[w], co2_pen)
    m.setObjective(obj, GRB.MINIMIZE)

    m.update()
    allvars = m.getVars()
    names = m.getAttr('VarName', allvars)
    vtypes = m.getAttr('VType', allvars)

    def make_sol(vals, objv):
        fd = {}
        mv = {}
        for nm, vt, v in zip(names, vtypes, vals):
            if vt in ('I', 'B'):
                v = float(round(v))
            fd[nm] = v
            if abs(v) > 1e-7:
                mv[nm] = float(v)
        invest = []
        for g in gens:
            i = g['id']
            size = float(g.get('capacity_size_MW', 1.0) or 1.0)
            yn = round(fd.get(f"y_P_new[{i}]", 0.0))
            yr = round(fd.get(f"y_P_ret[{i}]", 0.0))
            tot = float(g.get('existing_capacity_MW', 0.0) or 0.0) + size * (yn - yr)
            entry = {
                "generator_id": i,
                "resource_type": g.get('resource_type', ''),
                "zone": g.get('zone', 0),
                "y_P_new": float(yn),
                "y_P_ret": float(yr),
                "y_P_total_MW": float(tot),
            }
            if g.get('is_storage', False):
                esize = float(g.get('storage_capacity_size_MWh', 1.0) or 1.0)
                en = round(fd.get(f"y_E_new[{i}]", 0.0))
                er = round(fd.get(f"y_E_ret[{i}]", 0.0))
                etot = float(g.get('existing_storage_capacity_MWh', 0.0) or 0.0) + esize * (en - er)
                entry["y_E_new"] = float(en)
                entry["y_E_ret"] = float(er)
                entry["y_E_total_MWh"] = float(etot)
            invest.append(entry)
        trans = []
        for l in lines:
            li = l['id']
            fn = round(fd.get(f"y_F_new[{li}]", 0.0))
            trans.append({
                "line_id": li,
                "y_F_new_MW": float(fn),
                "total_capacity_MW": float(l.get('existing_capacity_MW', 0.0) or 0.0) + float(fn),
            })
        return {
            "objective_value": float(objv),
            "model_variables": mv,
            "investment_decisions": invest,
            "transmission_decisions": trans,
        }

    best = [float('inf')]

    def cb(model, where):
        if where == GRB.Callback.MIPSOL:
            objv = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if objv < best[0] - 1e-6:
                best[0] = objv
                if logger:
                    try:
                        vals = model.cbGetSolution(allvars)
                        logger.log_solution(objv, make_sol(vals, objv))
                    except Exception:
                        try:
                            logger.log(objv)
                        except Exception:
                            pass

    elapsed = time.time() - t0
    m.Params.TimeLimit = max(1.0, args.time_limit - elapsed - 3.0)

    try:
        m.optimize(cb)
    except gp.GurobiError:
        pass

    if m.SolCount > 0:
        vals = m.getAttr('X', allvars)
        sol = make_sol(vals, m.ObjVal)
        if logger and m.ObjVal < best[0] - 1e-6:
            try:
                logger.log_solution(m.ObjVal, sol)
            except Exception:
                pass
    else:
        # fallback: no feasible solution found -- emit zero-investment skeleton
        invest = []
        for g in gens:
            entry = {
                "generator_id": g['id'],
                "resource_type": g.get('resource_type', ''),
                "zone": g.get('zone', 0),
                "y_P_new": 0.0,
                "y_P_ret": 0.0,
                "y_P_total_MW": float(g.get('existing_capacity_MW', 0.0) or 0.0),
            }
            if g.get('is_storage', False):
                entry["y_E_new"] = 0.0
                entry["y_E_ret"] = 0.0
                entry["y_E_total_MWh"] = float(g.get('existing_storage_capacity_MWh', 0.0) or 0.0)
            invest.append(entry)
        trans = [{
            "line_id": l['id'],
            "y_F_new_MW": 0.0,
            "total_capacity_MW": float(l.get('existing_capacity_MW', 0.0) or 0.0),
        } for l in lines]
        sol = {
            "objective_value": 1e30,
            "model_variables": {},
            "investment_decisions": invest,
            "transmission_decisions": trans,
        }

    with open(args.solution_path, 'w') as f:
        json.dump(sol, f)


if __name__ == '__main__':
    main()
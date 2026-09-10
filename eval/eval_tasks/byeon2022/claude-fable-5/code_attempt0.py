import argparse
import json
import time

import gurobipy as gp
from gurobipy import GRB


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", required=True)
    ap.add_argument("--solution_path", required=True)
    ap.add_argument("--time_limit", type=int, default=600)
    ap.add_argument("--log_path", default=None)
    args = ap.parse_args()
    t_start = time.time()

    logger = None
    if args.log_path:
        try:
            from solution_logger import SolutionLogger
            logger = SolutionLogger(args.log_path, sense="minimize")
        except Exception:
            logger = None

    with open(args.instance_path) as fh:
        inst = json.load(fh)

    T = int(inst["time_periods"])
    beta = float(inst["objective_weight_beta"])
    enet = inst["electricity_network"]
    buses = enet["buses"]
    lines = enet["lines"]
    gens_raw = inst["generators"]["generators"]
    gnet = inst["gas_network"]
    juncs_raw = gnet["junctions"]
    conns = gnet["connections"]
    zones = gnet.get("pricing_zones", [])
    maxP = float(gnet["max_gas_price_mmBtu"])
    minP = float(gnet["min_gas_price_mmBtu"])

    jzone = {}
    for z in zones:
        for j in z["junctions"]:
            jzone[int(j)] = z["id"]

    THRESH_KEYS = ("profitability_threshold", "max_profitable_gas_price",
                   "gas_price_threshold", "profit_threshold", "max_gas_price",
                   "phi_max", "phimax", "threshold")

    GENS = []
    for g in gens_raw:
        gd = {}
        gd["id"] = g["id"]
        gd["bus"] = g["bus"]
        gd["is_gfpp"] = bool(g.get("is_gfpp", False))
        gd["junction"] = g.get("gas_junction")
        gd["pmin"] = float(g["min_power"])
        gd["pmax"] = float(g["max_power"])
        gd["rd"] = float(g["ramp_down"])
        gd["ru"] = float(g["ramp_up"])
        gd["noload"] = float(g["no_load_cost"])
        gd["ut"] = int(g["min_up_time"])
        gd["dt"] = int(g["min_down_time"])
        gd["status"] = int(g["initial_status"])
        gd["p0"] = float(g["initial_gen"])
        gd["act"] = int(g["initial_active_periods"])
        gd["inact"] = int(g["initial_inactive_periods"])
        gd["tiers"] = [(int(round(x[0])), float(x[1]))
                       for x in (g.get("startup_cost_params") or [])]
        bids = sorted(g["bids"], key=lambda b: b["id"])
        frac = g.get("max_gas_price_fraction")
        blist = []
        for b in bids:
            th = None
            for k in THRESH_KEYS:
                if k in b and b[k] is not None:
                    th = float(b[k])
                    break
            if th is None:
                th = (float(frac) if frac is not None else 1.0) * maxP
            blist.append((b["id"], float(b["price"]), float(b["max_amount"]), th))
        gd["bids"] = blist
        hr = g.get("heat_rate_coefficients") or {}
        gd["H2"] = float(hr.get("H_u2") or 0.0)
        gd["H1"] = float(hr.get("H_u1") or 0.0)
        gd["H0"] = float(hr.get("H_u0") or 0.0)
        GENS.append(gd)

    gfpp_at_junction = {}
    for gd in GENS:
        if gd["is_gfpp"] and gd["junction"] is not None:
            gfpp_at_junction.setdefault(int(gd["junction"]), []).append(gd)

    zone_gfpps = {}
    for gd in GENS:
        if gd["is_gfpp"] and gd["junction"] is not None:
            z = jzone.get(int(gd["junction"]))
            if z is not None:
                zone_gfpps.setdefault(z, []).append(gd)

    JUNCS = []
    for j in juncs_raw:
        jd = {"id": j["id"], "src": bool(j.get("is_source", False)),
              "plb": float(j["pressure_lb_squared"]),
              "pub": float(j["pressure_ub_squared"]),
              "shed": float(j["demand_shedding_cost"]),
              "dem": [float(x) for x in j["gas_demand_profile"]]}
        ivs = []
        for s in (j.get("supply_intervals") or []):
            width = max(0.0, float(s["interval_ub"]) - float(s["interval_lb"]))
            ivs.append((s["id"], width, float(s["slope"])))
        jd["ivs"] = ivs
        JUNCS.append(jd)

    def hist_on(gd, tau):
        if tau == 0:
            return float(gd["status"])
        if gd["status"] == 1 and (-tau) < gd["act"]:
            return 1.0
        return 0.0

    # ------------------------------------------------------------------ model
    m = gp.Model("ucgna")
    m.Params.OutputFlag = 0
    m.Params.Seed = 0
    m.Params.MIPGap = 1e-4
    m.Params.NumericFocus = 0
    m.Params.Threads = 1
    m.Params.IntFeasTol = 1e-7

    o = {}; vp = {}; vm = {}; r = {}; p = {}; se = {}; w = {}
    for gd in GENS:
        gid = gd["id"]
        for t in range(1, T + 1):
            o[gid, t] = m.addVar(vtype=GRB.BINARY)
            vp[gid, t] = m.addVar(vtype=GRB.BINARY)
            vm[gid, t] = m.addVar(vtype=GRB.BINARY)
            r[gid, t] = m.addVar(lb=0.0)
            p[gid, t] = m.addVar(lb=-GRB.INFINITY)
            for (bid, price, amt, th) in gd["bids"]:
                se[gid, bid, t] = m.addVar(lb=0.0, ub=amt)
                w[gid, bid, t] = m.addVar(vtype=GRB.BINARY)

    for gd in GENS:
        gid = gd["id"]; st = gd["status"]
        # initial-duration fixing
        if st == 1:
            nfix = max(0, gd["ut"] - gd["act"])
            for t in range(1, min(T, nfix) + 1):
                o[gid, t].LB = 1.0
        else:
            nfix = max(0, gd["dt"] - gd["inact"])
            for t in range(1, min(T, nfix) + 1):
                o[gid, t].UB = 0.0
        for t in range(1, T + 1):
            oprev = o[gid, t - 1] if t >= 2 else float(st)
            m.addConstr(vp[gid, t] - vm[gid, t] == o[gid, t] - oprev)
            m.addConstr(vp[gid, t] + vm[gid, t] <= 1)
            for (L, K) in gd["tiers"]:
                sub = gp.LinExpr()
                const = 0.0
                for n in range(1, L + 1):
                    tt = t - n
                    if tt >= 1:
                        sub += o[gid, tt]
                    else:
                        const += hist_on(gd, tt)
                m.addConstr(r[gid, t] >= K * (o[gid, t] - sub - const))
            ut = max(1, gd["ut"])
            s0 = max(1, t - ut + 1)
            m.addConstr(gp.quicksum(vp[gid, s] for s in range(s0, t + 1)) <= o[gid, t])
        dt_ = gd["dt"]
        if dt_ >= 1:
            for s in range(1, T + 1):
                e = min(T, s + dt_ - 1)
                oprev = o[gid, s - 1] if s >= 2 else float(st)
                m.addConstr(gp.quicksum(vp[gid, u] for u in range(s, e + 1)) <= 1 - oprev)
        # dispatch and bids
        for t in range(1, T + 1):
            m.addConstr(p[gid, t] == gp.quicksum(se[gid, b[0], t] for b in gd["bids"]))
            m.addConstr(p[gid, t] >= gd["pmin"] * o[gid, t])
            m.addConstr(p[gid, t] <= gd["pmax"] * o[gid, t])
            pprev = p[gid, t - 1] if t >= 2 else gd["p0"]
            oprev = o[gid, t - 1] if t >= 2 else float(st)
            m.addConstr(p[gid, t] - pprev <= gd["ru"] * oprev + gd["pmax"] * vp[gid, t])
            m.addConstr(pprev - p[gid, t] <= gd["rd"] * o[gid, t] + gd["pmin"] * vm[gid, t])
            nb = len(gd["bids"])
            for i, (bid, price, amt, th) in enumerate(gd["bids"]):
                m.addConstr(w[gid, bid, t] <= o[gid, t])
                m.addConstr(se[gid, bid, t] <= amt * w[gid, bid, t])
                if i + 1 < nb:
                    nxt = gd["bids"][i + 1][0]
                    m.addConstr(w[gid, nxt, t] <= w[gid, bid, t])
                    m.addConstr(se[gid, bid, t] >= amt * w[gid, nxt, t])

    # electricity network
    theta = {}; f = {}
    for b in buses:
        bid_ = b["id"]
        for t in range(1, T + 1):
            theta[bid_, t] = m.addVar(lb=float(b["voltage_angle_lb"]),
                                      ub=float(b["voltage_angle_ub"]))
    for l in lines:
        lid = l["id"]; fb = l["from_bus"]; tb = l["to_bus"]
        sus = float(l["susceptance"]); lim = float(l["thermal_limit"])
        adl = float(l["angle_diff_limit"])
        for t in range(1, T + 1):
            f[lid, t] = m.addVar(lb=-lim, ub=lim)
            m.addConstr(f[lid, t] == sus * (theta[fb, t] - theta[tb, t]))
            m.addConstr(theta[fb, t] - theta[tb, t] <= adl)
            m.addConstr(theta[fb, t] - theta[tb, t] >= -adl)
    gens_at_bus = {}
    for gd in GENS:
        gens_at_bus.setdefault(gd["bus"], []).append(gd["id"])
    lines_out = {}; lines_in = {}
    for l in lines:
        lines_out.setdefault(l["from_bus"], []).append(l["id"])
        lines_in.setdefault(l["to_bus"], []).append(l["id"])
    for b in buses:
        bid_ = b["id"]
        for t in range(1, T + 1):
            m.addConstr(gp.quicksum(p[g, t] for g in gens_at_bus.get(bid_, []))
                        - float(b["demand_profile"][t - 1])
                        == gp.quicksum(f[l, t] for l in lines_out.get(bid_, []))
                        - gp.quicksum(f[l, t] for l in lines_in.get(bid_, [])))

    # gas network
    sg = {}; sgs = {}; pisq = {}; lg = {}; qg = {}; gam = {}
    for jd in JUNCS:
        jid = jd["id"]
        has_g = jid in gfpp_at_junction
        cap = sum(iv[1] for iv in jd["ivs"]) if jd["src"] else 0.0
        for t in range(1, T + 1):
            sg[jid, t] = m.addVar(lb=0.0, ub=cap if jd["src"] else 0.0)
            pisq[jid, t] = m.addVar(lb=jd["plb"], ub=jd["pub"])
            d = jd["dem"][t - 1]
            lg[jid, t] = m.addVar(lb=min(0.0, d))
            qg[jid, t] = m.addVar(lb=0.0, ub=max(0.0, d))
            m.addConstr(lg[jid, t] + qg[jid, t] == d)
            gam[jid, t] = m.addVar(lb=0.0, ub=GRB.INFINITY if has_g else 0.0)
            if jd["src"]:
                for (iid, wdt, slope) in jd["ivs"]:
                    sgs[jid, iid, t] = m.addVar(lb=0.0, ub=wdt)
                m.addConstr(sg[jid, t] ==
                            gp.quicksum(sgs[jid, iv[0], t] for iv in jd["ivs"]))

    phig = {}
    for c in conns:
        cid = c["id"]; fj = c["from_junction"]; tj = c["to_junction"]
        wf = c.get("weymouth_factor")
        crl = c.get("compression_ratio_lb"); cru = c.get("compression_ratio_ub")
        cvl = c.get("control_ratio_lb"); cvu = c.get("control_ratio_ub")
        for t in range(1, T + 1):
            phig[cid, t] = m.addVar(lb=0.0)
            if wf is not None:
                m.addQConstr(float(wf) * phig[cid, t] * phig[cid, t]
                             <= pisq[fj, t] - pisq[tj, t])
            if crl is not None and cru is not None:
                m.addConstr(pisq[tj, t] >= float(crl) ** 2 * pisq[fj, t])
                m.addConstr(pisq[tj, t] <= float(cru) ** 2 * pisq[fj, t])
            if cvl is not None and cvu is not None:
                m.addConstr(pisq[tj, t] >= float(cvl) ** 2 * pisq[fj, t])
                m.addConstr(pisq[tj, t] <= float(cvu) ** 2 * pisq[fj, t])

    cons_out = {}; cons_in = {}
    for c in conns:
        cons_out.setdefault(c["from_junction"], []).append(c["id"])
        cons_in.setdefault(c["to_junction"], []).append(c["id"])
    for jd in JUNCS:
        jid = jd["id"]
        for t in range(1, T + 1):
            m.addConstr(sg[jid, t] - lg[jid, t] - gam[jid, t]
                        == gp.quicksum(phig[cc, t] for cc in cons_out.get(jid, []))
                        - gp.quicksum(phig[cc, t] for cc in cons_in.get(jid, [])))

    for jid, glist in gfpp_at_junction.items():
        if (jid, 1) not in gam:
            continue
        for t in range(1, T + 1):
            expr = gp.QuadExpr()
            for gd in glist:
                gid = gd["id"]
                expr += (gd["H2"] * p[gid, t] * p[gid, t]
                         + gd["H1"] * p[gid, t] + gd["H0"] * o[gid, t])
            m.addQConstr(expr <= gam[jid, t])

    psi = {}
    for z in zones:
        zid = z["id"]
        for t in range(1, T + 1):
            psi[zid, t] = m.addVar(lb=minP, ub=maxP)
    for gd in GENS:
        if not gd["is_gfpp"] or gd["junction"] is None:
            continue
        z = jzone.get(int(gd["junction"]))
        if z is None:
            continue
        gid = gd["id"]; nb = len(gd["bids"])
        for t in range(1, T + 1):
            expr = gp.LinExpr()
            for i, (bid, price, amt, th) in enumerate(gd["bids"]):
                if i + 1 < nb:
                    nxt = gd["bids"][i + 1][0]
                    expr += th * (w[gid, bid, t] - w[gid, nxt, t])
                else:
                    expr += th * w[gid, bid, t]
            m.addConstr(expr >= psi[z, t] - maxP * (1 - o[gid, t]))

    elec = gp.LinExpr()
    for gd in GENS:
        gid = gd["id"]
        for t in range(1, T + 1):
            elec += gd["noload"] * o[gid, t] + r[gid, t]
            for (bid, price, amt, th) in gd["bids"]:
                elec += price * se[gid, bid, t]
    gascost = gp.LinExpr()
    for jd in JUNCS:
        jid = jd["id"]
        for t in range(1, T + 1):
            gascost += jd["shed"] * qg[jid, t]
            if jd["src"]:
                for (iid, wdt, slope) in jd["ivs"]:
                    gascost += slope * sgs[jid, iid, t]
    m.setObjective(beta * elec + (1.0 - beta) * gascost, GRB.MINIMIZE)

    # ------------------------------------------------- key/var registration
    key_list = []; var_list = []

    def reg(k, v):
        key_list.append(k); var_list.append(v)

    for gd in GENS:
        gid = gd["id"]
        for t in range(1, T + 1):
            reg(f"o_{gid}_{t}", o[gid, t])
            reg(f"vp_{gid}_{t}", vp[gid, t])
            reg(f"vm_{gid}_{t}", vm[gid, t])
            reg(f"r_{gid}_{t}", r[gid, t])
            reg(f"p_{gid}_{t}", p[gid, t])
            for (bid, price, amt, th) in gd["bids"]:
                reg(f"se_{gid}_{bid}_{t}", se[gid, bid, t])
                reg(f"w_{gid}_{bid}_{t}", w[gid, bid, t])
    for l in lines:
        for t in range(1, T + 1):
            reg(f"f_{l['id']}_{t}", f[l["id"], t])
    for b in buses:
        for t in range(1, T + 1):
            reg(f"theta_{b['id']}_{t}", theta[b["id"], t])
    for jd in JUNCS:
        jid = jd["id"]
        for t in range(1, T + 1):
            reg(f"sg_{jid}_{t}", sg[jid, t])
            reg(f"pisq_{jid}_{t}", pisq[jid, t])
            reg(f"lg_{jid}_{t}", lg[jid, t])
            reg(f"qg_{jid}_{t}", qg[jid, t])
            reg(f"gamma_{jid}_{t}", gam[jid, t])
            if jd["src"]:
                for (iid, wdt, slope) in jd["ivs"]:
                    reg(f"sgs_{jid}_{iid}_{t}", sgs[jid, iid, t])
    for c in conns:
        for t in range(1, T + 1):
            reg(f"phig_{c['id']}_{t}", phig[c["id"], t])
    for z in zones:
        for t in range(1, T + 1):
            reg(f"psi_{z['id']}_{t}", psi[z["id"], t])

    variant = inst.get("released_problem_variant", "integrated_primal_ucgna")

    # ------------------------------------------------------------- finalize
    def finalize(raw):
        pv = {}
        el = 0.0
        ovals = {}
        for gd in GENS:
            gid = gd["id"]
            pv[f"o_{gid}_0"] = float(gd["status"])
            pv[f"p_{gid}_0"] = float(gd["p0"])
            prev = float(gd["status"])
            for t in range(1, T + 1):
                ov = 1.0 if raw[f"o_{gid}_{t}"] > 0.5 else 0.0
                pv[f"o_{gid}_{t}"] = ov
                ovals[gid, t] = ov
                pv[f"vp_{gid}_{t}"] = max(0.0, ov - prev)
                pv[f"vm_{gid}_{t}"] = max(0.0, prev - ov)
                prev = ov
            for t in range(1, T + 1):
                rv = 0.0
                for (L, K) in gd["tiers"]:
                    ssum = 0.0
                    for n in range(1, L + 1):
                        tt = t - n
                        ssum += ovals[gid, tt] if tt >= 1 else hist_on(gd, tt)
                    rv = max(rv, K * (ovals[gid, t] - ssum))
                rv = max(rv, 0.0)
                pv[f"r_{gid}_{t}"] = rv
                el += gd["noload"] * ovals[gid, t] + rv
            nb = len(gd["bids"])
            for t in range(1, T + 1):
                pv[f"p_{gid}_{t}"] = float(raw[f"p_{gid}_{t}"])
                wv_list = []
                for i, (bid, price, amt, th) in enumerate(gd["bids"]):
                    wv = 1.0 if raw[f"w_{gid}_{bid}_{t}"] > 0.5 else 0.0
                    if ovals[gid, t] < 0.5:
                        wv = 0.0
                    if i > 0 and wv > wv_list[i - 1]:
                        wv = wv_list[i - 1]
                    wv_list.append(wv)
                    pv[f"w_{gid}_{bid}_{t}"] = wv
                for i, (bid, price, amt, th) in enumerate(gd["bids"]):
                    sev = min(max(0.0, float(raw[f"se_{gid}_{bid}_{t}"])), amt)
                    if sev > amt * wv_list[i]:
                        sev = amt * wv_list[i]
                    if i + 1 < nb and wv_list[i + 1] > 0.5:
                        sev = amt
                    pv[f"se_{gid}_{bid}_{t}"] = sev
                    el += price * sev
                if gd["is_gfpp"]:
                    ph = 0.0
                    for i in range(nb):
                        th = gd["bids"][i][3]
                        if i + 1 < nb:
                            ph += th * (wv_list[i] - wv_list[i + 1])
                        else:
                            ph += th * wv_list[i]
                    pv[f"phimax_{gid}_{t}"] = ph
        for l in lines:
            for t in range(1, T + 1):
                pv[f"f_{l['id']}_{t}"] = float(raw[f"f_{l['id']}_{t}"])
        for b in buses:
            for t in range(1, T + 1):
                pv[f"theta_{b['id']}_{t}"] = float(raw[f"theta_{b['id']}_{t}"])
        gc = 0.0
        for jd in JUNCS:
            jid = jd["id"]
            for t in range(1, T + 1):
                tot = 0.0
                if jd["src"]:
                    for (iid, wdt, slope) in jd["ivs"]:
                        v = min(max(0.0, float(raw[f"sgs_{jid}_{iid}_{t}"])), wdt)
                        pv[f"sgs_{jid}_{iid}_{t}"] = v
                        tot += v
                        gc += slope * v
                pv[f"sg_{jid}_{t}"] = tot
                pv[f"pisq_{jid}_{t}"] = min(max(float(raw[f"pisq_{jid}_{t}"]),
                                                jd["plb"]), jd["pub"])
                pv[f"lg_{jid}_{t}"] = float(raw[f"lg_{jid}_{t}"])
                qv = min(max(0.0, float(raw[f"qg_{jid}_{t}"])),
                         max(0.0, jd["dem"][t - 1]))
                pv[f"qg_{jid}_{t}"] = qv
                gc += jd["shed"] * qv
                pv[f"gamma_{jid}_{t}"] = max(0.0, float(raw[f"gamma_{jid}_{t}"]))
        for c in conns:
            for t in range(1, T + 1):
                pv[f"phig_{c['id']}_{t}"] = max(0.0, float(raw[f"phig_{c['id']}_{t}"]))
        for z in zones:
            zid = z["id"]
            for t in range(1, T + 1):
                val = min(max(float(raw[f"psi_{zid}_{t}"]), minP), maxP)
                cap = None
                for gd in zone_gfpps.get(zid, []):
                    if ovals[gd["id"], t] > 0.5:
                        ph = pv[f"phimax_{gd['id']}_{t}"]
                        cap = ph if cap is None else min(cap, ph)
                if cap is not None and val > cap:
                    val = cap
                pv[f"psi_{zid}_{t}"] = val
        obj = beta * el + (1.0 - beta) * gc
        sol = {"objective_value": obj,
               "released_problem_variant": variant,
               "objective_weight_beta": beta,
               "primary_variables": pv}
        return obj, sol

    # ----------------------------------------------------------------- solve
    best = {"obj": float("inf"), "sol": None}

    def cb(model, where):
        if where == GRB.Callback.MIPSOL:
            try:
                vals = model.cbGetSolution(var_list)
                raw = dict(zip(key_list, vals))
                obj, sol = finalize(raw)
                if obj < best["obj"] - 1e-9:
                    best["obj"] = obj
                    best["sol"] = sol
                    if logger:
                        logger.log_solution(obj, sol)
            except Exception:
                pass

    elapsed = time.time() - t_start
    rem = args.time_limit - elapsed - 8.0
    m.Params.TimeLimit = max(5.0, rem)

    try:
        m.optimize(cb)
    except Exception:
        pass

    sol_final = None
    try:
        if m.SolCount > 0:
            raw = {k: v.X for k, v in zip(key_list, var_list)}
            obj, sol = finalize(raw)
            if obj < best["obj"] - 1e-9:
                best["obj"] = obj
                best["sol"] = sol
                if logger:
                    logger.log_solution(obj, sol)
            sol_final = best["sol"] if best["sol"] is not None else sol
    except Exception:
        pass

    if sol_final is None and best["sol"] is not None:
        sol_final = best["sol"]

    if sol_final is None:
        # emergency fallback: zero witness (may be infeasible, but ensures output)
        raw = {k: 0.0 for k in key_list}
        obj, sol_final = finalize(raw)
        if logger:
            logger.log(obj)

    with open(args.solution_path, "w") as fh:
        json.dump(sol_final, fh)


if __name__ == "__main__":
    main()
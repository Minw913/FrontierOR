import argparse
import json
import time

import gurobipy as gp
from gurobipy import GRB, quicksum

VAR_KEYS = ["alpha", "gamma", "p_gen", "s_res", "f_fwd", "f_bwd", "u_stor", "v_stor", "r_stor"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", required=True)
    ap.add_argument("--solution_path", required=True)
    ap.add_argument("--time_limit", type=int, default=600)
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

    sysd = data["system"]
    gens = sysd.get("generators", []) or []
    lines = sysd.get("transmission_lines", []) or []
    stors = sysd.get("storage", []) or []
    buses = list(sysd.get("buses", []) or [])
    peak = {}
    for ld in sysd.get("loads", []) or []:
        peak[ld["bus"]] = peak.get(ld["bus"], 0.0) + float(ld["annual_peak_load_MW"])
    for b in peak:
        if b not in buses:
            buses.append(b)

    sp = data["scenario_parameters"]
    H = int(sp["hours_per_day"])
    gh = float(sp.get("demand_growth_high_percent") or 0.0)
    gl = float(sp.get("demand_growth_low_percent") or 0.0)
    cv = float(sp.get("cost_variation_percent") or 0.0)
    glb = data["global_parameters"]
    rm = float(glb.get("reserve_margin_requirement") or 0.0)
    srp = float(glb.get("spinning_reserve_percent") or 0.0)
    der = glb.get("derating_factors", {}) or {}

    # ---------- scenario tree ----------
    tnodes = {nd["id"]: nd for nd in data["scenario_tree"]}
    root_id = None
    children_of = {nid: [] for nid in tnodes}
    for nid, nd in tnodes.items():
        if nd.get("parent_id") is None:
            root_id = nid
    have_children_field = False
    for nid, nd in tnodes.items():
        ch = nd.get("children")
        if ch:
            children_of[nid] = list(ch)
            have_children_field = True
    if not have_children_field and len(tnodes) > 1:
        for nid, nd in tnodes.items():
            pid = nd.get("parent_id")
            if pid is not None:
                children_of[pid].append(nid)
        for nid in children_of:
            children_of[nid].sort(key=lambda z: str(z))

    dm = {root_id: 1.0}
    cmm = {root_id: 1.0}
    path = {root_id: [root_id]}
    order = []
    stack = [root_id]
    while stack:
        nid = stack.pop(0)
        order.append(nid)
        ch = children_of.get(nid, [])
        k = len(ch)
        for i, c in enumerate(ch):
            if k == 1:
                g_ = 0.5 * (gh + gl)
                cmu = 1.0
            else:
                g_ = gh if i < k / 2.0 else gl
                cmu = (1.0 + cv) if (i % 2 == 0) else (1.0 - cv)
            dm[c] = dm[nid] * (1.0 + g_)
            cmm[c] = cmm[nid] * cmu
            path[c] = path[nid] + [c]
            stack.append(c)
    prob = {nid: float(tnodes[nid].get("probability", 1.0)) for nid in tnodes}

    # ---------- generator / network prep ----------
    def gkind(g):
        t = str(g.get("type", "")).lower()
        if "wind" in t:
            return "wind"
        if "solar" in t or "pv" in t:
            return "solar"
        return "fossil"

    for g in gens:
        g["_k"] = gkind(g)
        g["_df"] = float(der.get(g["_k"], der.get("fossil", 1.0)))

    GN, LN, SN = len(gens), len(lines), len(stors)
    gens_at, stor_at, lfrom, lto = {}, {}, {}, {}
    for gi, g in enumerate(gens):
        gens_at.setdefault(g["bus"], []).append(gi)
    for si, st_ in enumerate(stors):
        stor_at.setdefault(st_["bus"], []).append(si)
    for li, l in enumerate(lines):
        lfrom.setdefault(l["from_bus"], []).append(li)
        lto.setdefault(l["to_bus"], []).append(li)
        if l["from_bus"] not in buses:
            buses.append(l["from_bus"])
        if l["to_bus"] not in buses:
            buses.append(l["to_bus"])

    cand_g = [g for g in gens if not g["existing"]]
    cand_l = [l for l in lines if not l["existing"]]
    cand_s = [s_ for s_ in stors if not s_["existing"]]
    have_cands = bool(cand_g or cand_l or cand_s)

    days = data.get("representative_days", []) or []
    lls = data.get("lower_level_scenarios", {}) or {}
    day_scens = {}
    for d in days:
        scl = lls.get(str(d["id"]))
        if scl is None:
            scl = lls.get(d["id"], [])
        day_scens[d["id"]] = scl or []

    def build_ctx_data(day, sc):
        lp = day["load_profile"]
        spf = day.get("solar_profile", [0.0] * H)
        dmul = sc["demand_multipliers"]
        smul = sc.get("solar_multipliers", [1.0] * H)
        wpf = sc.get("wind_power_factors", [0.0] * H)
        dem = {}
        for b, pk in peak.items():
            if pk > 0:
                dem[b] = [pk * lp[t] * dmul[t] for t in range(H)]
        cfs = [spf[t] * smul[t] for t in range(H)]
        cfw = [wpf[t] for t in range(H)]
        return {"dem": dem, "cfs": cfs, "cfw": cfw}

    ctx_data = {}
    ctx_list = []
    for n in order:
        for d in days:
            for idx, sc in enumerate(day_scens[d["id"]]):
                sid = sc.get("scenario_id", idx)
                if (d["id"], sid) not in ctx_data:
                    ctx_data[(d["id"], sid)] = build_ctx_data(d, sc)
                ctx_list.append((n, d["id"], sid))

    n_ctx = len(ctx_list)
    per_ctx = H * (4 * GN + 2 * LN + 3 * SN + 1)
    write_buf = max(5.0, min(40.0, 3.0 + n_ctx * per_ctx * 2.5e-7))
    deadline = t0 + max(5.0, args.time_limit - write_buf)

    def set_params(m, tl):
        m.Params.OutputFlag = 0
        m.Params.Threads = 1
        m.Params.Seed = 0
        m.Params.MIPGap = 1e-4
        m.Params.NumericFocus = 0
        if tl is not None:
            m.Params.TimeLimit = max(0.5, tl)

    # ---------- context builder ----------
    def add_context(m, cd, dmn, ag, al, as_, binary_alpha):
        dem, cfs, cfw = cd["dem"], cd["cfs"], cd["cfw"]
        vt = GRB.BINARY if binary_alpha else GRB.CONTINUOUS
        alpha = m.addVars(GN, H, ub=1.0, vtype=vt)
        gamma = m.addVars(GN, H, ub=1.0)
        p = m.addVars(GN, H)
        s = m.addVars(GN, H)
        ff = m.addVars(LN, H) if LN else {}
        fb = m.addVars(LN, H) if LN else {}
        uu = m.addVars(SN, H) if SN else {}
        vv = m.addVars(SN, H) if SN else {}
        rr = m.addVars(SN, H) if SN else {}

        for gi, g in enumerate(gens):
            k = g["_k"]
            cf = cfw if k == "wind" else (cfs if k == "solar" else None)
            pmax = float(g["Pmax_MW"])
            pmin = float(g["Pmin_MW"])
            ramp = float(g["ramp_MW_per_h"])
            ut = max(1, int(g.get("min_on_h", 1)))
            dt_ = max(1, int(g.get("min_off_h", 1)))
            be = None
            if not g["existing"]:
                be = ag[g["id"]]
                if isinstance(be, (int, float)):
                    if be < 0.5:
                        for t in range(H):
                            alpha[gi, t].ub = 0.0
                    be = None
            gamma[gi, 0].ub = 0.0
            for t in range(H):
                cap_t = pmax * (cf[t] if cf is not None else 1.0)
                m.addConstr(p[gi, t] + s[gi, t] <= cap_t * alpha[gi, t])
                if pmin > 1e-12:
                    m.addConstr(p[gi, t] >= pmin * alpha[gi, t])
                if be is not None:
                    m.addConstr(alpha[gi, t] <= be)
                if t >= 1:
                    m.addConstr(gamma[gi, t] >= alpha[gi, t] - alpha[gi, t - 1])
                    m.addConstr(p[gi, t] - p[gi, t - 1] <= ramp)
                    m.addConstr(p[gi, t - 1] - p[gi, t] <= ramp)
                    if ut > 1:
                        for tau in range(t + 1, min(t + ut - 1, H - 1) + 1):
                            m.addConstr(alpha[gi, tau] >= alpha[gi, t] - alpha[gi, t - 1])
                    if dt_ > 1:
                        for tau in range(t + 1, min(t + dt_ - 1, H - 1) + 1):
                            m.addConstr(1.0 - alpha[gi, tau] >= alpha[gi, t - 1] - alpha[gi, t])

        for li, l in enumerate(lines):
            cap = float(l["flow_limit_MW"])
            if l["existing"]:
                for t in range(H):
                    ff[li, t].ub = cap
                    fb[li, t].ub = cap
            else:
                ae = al[l["id"]]
                if isinstance(ae, (int, float)):
                    c2 = cap if ae > 0.5 else 0.0
                    for t in range(H):
                        ff[li, t].ub = c2
                        fb[li, t].ub = c2
                else:
                    for t in range(H):
                        m.addConstr(ff[li, t] <= cap * ae)
                        m.addConstr(fb[li, t] <= cap * ae)

        for si, st_ in enumerate(stors):
            cap = float(st_["capacity_MW"])
            eff = float(st_.get("efficiency", glb.get("storage_efficiency", 1.0)))
            expr = None
            capv = None
            if st_["existing"]:
                capv = cap
            else:
                ae = as_[st_["id"]]
                if isinstance(ae, (int, float)):
                    capv = cap if ae > 0.5 else 0.0
                else:
                    expr = ae
            for t in range(H):
                if capv is not None:
                    uu[si, t].ub = capv
                    vv[si, t].ub = capv
                    rr[si, t].ub = capv
                else:
                    m.addConstr(uu[si, t] <= cap * expr)
                    m.addConstr(vv[si, t] <= cap * expr)
                    m.addConstr(rr[si, t] <= cap * expr)
                m.addConstr(vv[si, t] <= rr[si, t])
                if t >= 1:
                    m.addConstr(rr[si, t] == rr[si, t - 1] + eff * uu[si, t - 1] - vv[si, t - 1])
            rr[si, 0].ub = 0.0

        for b in buses:
            db = dem.get(b)
            gl_ = gens_at.get(b, [])
            sl_ = stor_at.get(b, [])
            lf = lfrom.get(b, [])
            lt = lto.get(b, [])
            for t in range(H):
                dembt = (db[t] * dmn) if db else 0.0
                lhs = quicksum(p[gi, t] for gi in gl_)
                lhs += quicksum(vv[si, t] - uu[si, t] for si in sl_)
                lhs += quicksum((1.0 - float(lines[li]["loss_factor"])) * ff[li, t] for li in lt)
                lhs += quicksum((1.0 - float(lines[li]["loss_factor"])) * fb[li, t] for li in lf)
                lhs -= quicksum(ff[li, t] for li in lf)
                lhs -= quicksum(fb[li, t] for li in lt)
                m.addConstr(lhs == dembt)
                if dembt > 0 and srp > 0:
                    m.addConstr(quicksum(s[gi, t] for gi in gl_) >= srp * dembt)

        obj = gp.LinExpr()
        for gi, g in enumerate(gens):
            mc = float(g["b_cost"]) + float(g["c_cost"]) * float(g["Pmax_MW"])
            su = float(g["startup_cost"])
            ac = float(g["a_cost"])
            for t in range(H):
                obj.add(gamma[gi, t], su)
                obj.add(alpha[gi, t], ac)
                obj.add(p[gi, t], mc)
        return (alpha, gamma, p, s, ff, fb, uu, vv, rr), obj

    # ---------- extraction ----------
    def extract(vars_):
        alpha, gamma, p, s, ff, fb, uu, vv, rr = vars_
        aval = [[1 if alpha[gi, t].X > 0.5 else 0 for t in range(H)] for gi in range(GN)]
        ent = {k: {} for k in VAR_KEYS}
        cost = 0.0
        for t in range(H):
            ts = str(t)
            da, dg, dp, ds = {}, {}, {}, {}
            for gi, g in enumerate(gens):
                av = aval[gi][t]
                gv = max(0, av - aval[gi][t - 1]) if t > 0 else 0
                pv = round(max(0.0, p[gi, t].X), 9)
                sv = round(max(0.0, s[gi, t].X), 9)
                gid = g["id"]
                da[gid] = av
                dg[gid] = gv
                dp[gid] = pv
                ds[gid] = sv
                cost += float(g["startup_cost"]) * gv + float(g["a_cost"]) * av + \
                        (float(g["b_cost"]) + float(g["c_cost"]) * float(g["Pmax_MW"])) * pv
            ent["alpha"][ts] = da
            ent["gamma"][ts] = dg
            ent["p_gen"][ts] = dp
            ent["s_res"][ts] = ds
            ent["f_fwd"][ts] = {l["id"]: round(max(0.0, ff[li, t].X), 9) for li, l in enumerate(lines)}
            ent["f_bwd"][ts] = {l["id"]: round(max(0.0, fb[li, t].X), 9) for li, l in enumerate(lines)}
            ent["u_stor"][ts] = {st_["id"]: round(max(0.0, uu[si, t].X), 9) for si, st_ in enumerate(stors)}
            ent["v_stor"][ts] = {st_["id"]: round(max(0.0, vv[si, t].X), 9) for si, st_ in enumerate(stors)}
            ent["r_stor"][ts] = {st_["id"]: round(max(0.0, rr[si, t].X), 9) for si, st_ in enumerate(stors)}
        return ent, cost

    def zero_entries():
        ent = {k: {} for k in VAR_KEYS}
        for t in range(H):
            ts = str(t)
            ent["alpha"][ts] = {g["id"]: 0 for g in gens}
            ent["gamma"][ts] = {g["id"]: 0 for g in gens}
            ent["p_gen"][ts] = {g["id"]: 0.0 for g in gens}
            ent["s_res"][ts] = {g["id"]: 0.0 for g in gens}
            ent["f_fwd"][ts] = {l["id"]: 0.0 for l in lines}
            ent["f_bwd"][ts] = {l["id"]: 0.0 for l in lines}
            ent["u_stor"][ts] = {s_["id"]: 0.0 for s_ in stors}
            ent["v_stor"][ts] = {s_["id"]: 0.0 for s_ in stors}
            ent["r_stor"][ts] = {s_["id"]: 0.0 for s_ in stors}
        return ent

    def unit_cost(item, kind, n):
        if kind == "g":
            capmw = item["Pmax_MW"]
        elif kind == "l":
            capmw = item["flow_limit_MW"]
        else:
            capmw = item["capacity_MW"]
        return float(item.get("capital_cost_per_kW", 0.0) or 0.0) * 1000.0 * float(capmw) * cmm[n]

    def new_built():
        return {n: {"g": {g["id"]: 0 for g in cand_g},
                    "l": {l["id"]: 0 for l in cand_l},
                    "s": {s_["id"]: 0 for s_ in cand_s}} for n in order}

    def build_all_root():
        b = new_built()
        for k in ("g", "l", "s"):
            for i in b[root_id][k]:
                b[root_id][k][i] = 1
        return b

    def compute_avail(built):
        av = {}
        for n in order:
            pth = path[n]
            ag = {g["id"]: (1.0 if any(built[m_]["g"][g["id"]] for m_ in pth) else 0.0) for g in cand_g}
            al = {l["id"]: (1.0 if any(built[m_]["l"][l["id"]] for m_ in pth) else 0.0) for l in cand_l}
            as_ = {s_["id"]: (1.0 if any(built[m_]["s"][s_["id"]] for m_ in pth) else 0.0) for s_ in cand_s}
            av[n] = (ag, al, as_)
        return av

    def inv_cost_of(built):
        tot = 0.0
        for n in order:
            pr = prob[n]
            for g in cand_g:
                tot += pr * built[n]["g"][g["id"]] * unit_cost(g, "g", n)
            for l in cand_l:
                tot += pr * built[n]["l"][l["id"]] * unit_cost(l, "l", n)
            for s_ in cand_s:
                tot += pr * built[n]["s"][s_["id"]] * unit_cost(s_, "s", n)
        return tot

    weight = {}
    for (n, did, sid) in ctx_list:
        weight[(n, did, sid)] = prob[n] / max(1, len(day_scens[did]))

    def assemble(entries_all, built, total):
        sol = {"objective_value": total}
        for key in VAR_KEYS:
            sol[key] = {}
        for (n, did, sid), ent in entries_all.items():
            ns, ds, ws = str(n), str(did), str(sid)
            for key in VAR_KEYS:
                sol[key].setdefault(ns, {}).setdefault(ds, {})[ws] = ent[key]
        inv = {}
        for n in order:
            inv[str(n)] = {"generators": {gid: int(v) for gid, v in built[n]["g"].items()},
                           "lines": {lid: int(v) for lid, v in built[n]["l"].items()},
                           "storage": {sid_: int(v) for sid_, v in built[n]["s"].items()}}
        sol["investments"] = inv
        return sol

    def write_sol(sol):
        with open(args.solution_path, "w") as f:
            json.dump(sol, f, separators=(",", ":"))

    built = new_built()
    wrote_ok = False
    try:
        # ---------- Phase A: investment sizing (relaxed commitment) ----------
        planok = not have_cands
        remA = deadline - time.time()
        do_A = have_cands and (remA - 1.0 * n_ctx > 8.0)
        reducedA = n_ctx * per_ctx > 250000
        if do_A:
            try:
                daysA = list(days)
                if reducedA:
                    while len(order) * len(daysA) * per_ctx > 2000000 and len(daysA) > 1:
                        daysA = daysA[::2]
                dayw = (float(len(days)) / max(1, len(daysA)))
                avg_ctx = {}
                if reducedA:
                    for d in daysA:
                        scs = day_scens[d["id"]]
                        Sn = max(1, len(scs))
                        ps = {"demand_multipliers": [sum(sc["demand_multipliers"][t] for sc in scs) / Sn if scs else 0.0 for t in range(H)],
                              "solar_multipliers": [sum(sc.get("solar_multipliers", [1.0] * H)[t] for sc in scs) / Sn if scs else 1.0 for t in range(H)],
                              "wind_power_factors": [sum(sc.get("wind_power_factors", [0.0] * H)[t] for sc in scs) / Sn if scs else 0.0 for t in range(H)]}
                        avg_ctx[d["id"]] = build_ctx_data(d, ps)

                mA = gp.Model("phaseA")
                set_params(mA, remA)
                mA.Params.MIPFocus = 1
                x = {}
                for n in order:
                    for g in cand_g:
                        x[("g", g["id"], n)] = mA.addVar(vtype=GRB.BINARY)
                    for l in cand_l:
                        x[("l", l["id"], n)] = mA.addVar(vtype=GRB.BINARY)
                    for s_ in cand_s:
                        x[("s", s_["id"], n)] = mA.addVar(vtype=GRB.BINARY)
                leaves = [n for n in order if not children_of.get(n)]
                for leaf in leaves:
                    pth = path[leaf]
                    for g in cand_g:
                        mA.addConstr(quicksum(x[("g", g["id"], m_)] for m_ in pth) <= 1)
                    for l in cand_l:
                        mA.addConstr(quicksum(x[("l", l["id"], m_)] for m_ in pth) <= 1)
                    for s_ in cand_s:
                        mA.addConstr(quicksum(x[("s", s_["id"], m_)] for m_ in pth) <= 1)
                bexp = {}
                for n in order:
                    pth = path[n]
                    for g in cand_g:
                        bexp[("g", g["id"], n)] = quicksum(x[("g", g["id"], m_)] for m_ in pth)
                    for l in cand_l:
                        bexp[("l", l["id"], n)] = quicksum(x[("l", l["id"], m_)] for m_ in pth)
                    for s_ in cand_s:
                        bexp[("s", s_["id"], n)] = quicksum(x[("s", s_["id"], m_)] for m_ in pth)

                objA = gp.LinExpr()
                for n in order:
                    pr = prob[n]
                    for g in cand_g:
                        objA.add(x[("g", g["id"], n)], pr * unit_cost(g, "g", n))
                    for l in cand_l:
                        objA.add(x[("l", l["id"], n)], pr * unit_cost(l, "l", n))
                    for s_ in cand_s:
                        objA.add(x[("s", s_["id"], n)], pr * unit_cost(s_, "s", n))
                exist_derated = sum(g["_df"] * float(g["Pmax_MW"]) for g in gens if g["existing"])
                total_peak = sum(peak.values())
                for n in order:
                    mA.addConstr(exist_derated +
                                 quicksum(g["_df"] * float(g["Pmax_MW"]) * bexp[("g", g["id"], n)] for g in cand_g)
                                 >= (1.0 + rm) * total_peak * dm[n])

                for n in order:
                    ag = {g["id"]: bexp[("g", g["id"], n)] for g in cand_g}
                    al = {l["id"]: bexp[("l", l["id"], n)] for l in cand_l}
                    as_ = {s_["id"]: bexp[("s", s_["id"], n)] for s_ in cand_s}
                    if reducedA:
                        for d in daysA:
                            _, oe = add_context(mA, avg_ctx[d["id"]], dm[n], ag, al, as_, False)
                            objA.add(oe, prob[n] * dayw)
                    else:
                        for d in days:
                            scs = day_scens[d["id"]]
                            for idx, sc in enumerate(scs):
                                sid = sc.get("scenario_id", idx)
                                _, oe = add_context(mA, ctx_data[(d["id"], sid)], dm[n], ag, al, as_, False)
                                objA.add(oe, prob[n] / max(1, len(scs)))
                mA.setObjective(objA, GRB.MINIMIZE)
                rem_now = deadline - time.time()
                tlA = max(3.0, min(0.5 * rem_now, rem_now - 1.0 * n_ctx - 5.0))
                mA.Params.TimeLimit = tlA
                mA.optimize()
                if mA.SolCount > 0:
                    for (k, i, n), var in x.items():
                        built[n][k][i] = 1 if var.X > 0.5 else 0
                    planok = True
                mA.dispose()
            except Exception:
                planok = False
        if not planok and have_cands:
            built = build_all_root()

        # ---------- Phase B: per-context unit commitment ----------
        def solve_ctx(n, did, sid, tl, availn):
            cdt = ctx_data[(did, sid)]
            ag, al, as_ = availn[n]
            m = gp.Model("ctx")
            set_params(m, tl)
            m.Params.FeasibilityTol = 1e-8
            m.Params.IntFeasTol = 1e-7
            vars_, oe = add_context(m, cdt, dm[n], ag, al, as_, True)
            m.setObjective(oe, GRB.MINIMIZE)
            m.optimize()
            if m.SolCount == 0:
                if m.Status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
                    m.dispose()
                    return "INF"
                m.Params.TimeLimit = max(10.0, tl * 3)
                m.Params.MIPFocus = 1
                m.optimize()
                if m.SolCount == 0:
                    st = m.Status
                    m.dispose()
                    return "INF" if st in (GRB.INFEASIBLE, GRB.INF_OR_UNBD) else None
            ent, cost = extract(vars_)
            try:
                gap = max(0.0, m.ObjVal - m.ObjBound)
            except Exception:
                gap = 0.0
            m.dispose()
            return ent, cost, gap

        def run_phaseB(availn):
            entries_all, costs, gaps = {}, {}, {}
            ninf = 0
            NC = len(ctx_list)
            for idx, (n, did, sid) in enumerate(ctx_list):
                rem = deadline - time.time()
                if rem > 1:
                    tl = max(0.8, min(60.0, 0.9 * rem / max(1, NC - idx)))
                else:
                    tl = 0.8
                res = solve_ctx(n, did, sid, tl, availn)
                key = (n, did, sid)
                if res in ("INF", None):
                    entries_all[key] = zero_entries()
                    costs[key] = 0.0
                    gaps[key] = 0.0
                    ninf += 1
                else:
                    entries_all[key], costs[key], gaps[key] = res
            return entries_all, costs, gaps, ninf

        availn = compute_avail(built)
        entries_all, costs, gaps, ninf = run_phaseB(availn)
        if ninf > 0 and have_cands and planok:
            built2 = build_all_root()
            availn2 = compute_avail(built2)
            e2, c2, g2, ninf2 = run_phaseB(availn2)
            if ninf2 < ninf:
                built, entries_all, costs, gaps, ninf = built2, e2, c2, g2, ninf2
                availn = availn2

        total = inv_cost_of(built) + sum(weight[c] * costs[c] for c in ctx_list)
        sol = assemble(entries_all, built, total)
        if logger:
            try:
                logger.log_solution(total, sol)
            except Exception:
                pass
        write_sol(sol)
        wrote_ok = True

        # ---------- improvement pass on contexts with residual gap ----------
        flagged = [c for c in ctx_list
                   if gaps.get(c, 0.0) > max(1e-6, 1.5e-4 * abs(costs.get(c, 0.0)))]
        flagged.sort(key=lambda c: -(gaps.get(c, 0.0) * weight[c]))
        improved = False
        for c in flagged:
            rem = deadline - time.time()
            if rem < 8:
                break
            res = solve_ctx(c[0], c[1], c[2], min(90.0, rem - 4.0), availn)
            if res not in (None, "INF"):
                e, cost2, g2 = res
                if cost2 < costs[c] - 1e-7:
                    costs[c] = cost2
                    entries_all[c] = e
                    improved = True
        if improved:
            total2 = inv_cost_of(built) + sum(weight[c] * costs[c] for c in ctx_list)
            if total2 < total - 1e-9:
                sol = assemble(entries_all, built, total2)
                if logger:
                    try:
                        logger.log_solution(total2, sol)
                    except Exception:
                        pass
                write_sol(sol)
    except Exception:
        if not wrote_ok:
            try:
                ents = {c: zero_entries() for c in ctx_list}
                sol = assemble(ents, built, 0.0)
                write_sol(sol)
            except Exception:
                try:
                    write_sol({"objective_value": 0.0, "alpha": {}, "gamma": {}, "p_gen": {},
                               "s_res": {}, "f_fwd": {}, "f_bwd": {}, "u_stor": {}, "v_stor": {},
                               "r_stor": {}, "investments": {}})
                except Exception:
                    pass


if __name__ == "__main__":
    main()
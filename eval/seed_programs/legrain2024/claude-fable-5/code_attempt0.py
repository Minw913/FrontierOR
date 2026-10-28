import argparse
import json
import time
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", required=True)
    ap.add_argument("--solution_path", required=True)
    ap.add_argument("--time_limit", type=int, default=300)
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
        inst = json.load(f)

    shifts = list(inst.get("shifts", []))
    skills = list(inst.get("skills", []))
    nurses = inst.get("nurses", [])
    nids = [n["id"] for n in nurses]
    nskills = {n["id"]: set(n.get("skills", []) or []) for n in nurses}
    prefs = {n["id"]: (n.get("preferences", []) or []) for n in nurses}
    demand = inst.get("demand", {}) or {}
    day_nums = [int(k.split("_")[-1]) for k in demand.keys()]
    D = max(day_nums) if day_nums else 0
    days = list(range(1, D + 1))

    def dem(d, s, k):
        return demand.get("day_%d" % d, {}).get(s, {}).get(k, 0)

    weekends = inst.get("weekends", []) or []
    wdef = (inst.get("weekend_definition") or "").lower()
    wk_days = []
    for wrec in weekends:
        ds = []
        sat = wrec.get("saturday")
        sun = wrec.get("sunday")
        if "friday" in wdef and sat is not None and sat - 1 >= 1:
            ds.append(sat - 1)
        if sat is not None:
            ds.append(sat)
        if sun is not None:
            ds.append(sun)
        ds = sorted(set(d for d in ds if 1 <= d <= D))
        if ds:
            wk_days.append(ds)

    if weekends:
        s0 = weekends[0].get("saturday", 6)
    else:
        s0 = 6
    daynames = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

    def dow(d):
        return (d - s0 + 5) % 7  # 0=Mon .. 5=Sat, 6=Sun

    def daytype_match(dt, d):
        t = str(dt).strip().lower()
        if t in ("any", "all", "*", "", "none"):
            return True
        if t in daynames:
            return daynames[dow(d)] == t
        if t == "weekend":
            return dow(d) >= 5
        if t == "weekday":
            return dow(d) < 5
        return True

    def shift_match(sn, a):
        t = str(sn)
        tl = t.lower()
        if tl in ("any", "*"):
            return a is not None
        if tl in ("none", "rest", "off", "-"):
            return a is None
        return a == t

    cons = inst.get("constraints", {}) or {}
    hc = cons.get("hard_constraints", {}) or {}
    sc = cons.get("soft_constraints", {}) or {}
    skill_hard = bool(hc.get("skill_requirement", True))
    dem_eq = bool(hc.get("demand_equality", False))
    hard_pats = hc.get("forbidden_patterns", []) or []
    soft_pats = sc.get("forbidden_patterns", []) or []
    skill_pen = sc.get("skill_requirement_penalty", 0) or 0
    pref_on = bool(sc.get("preferences_penalty", True))
    unsat_pen = inst.get("unsatisfied_demand_penalty", 0) or 0
    twd = sc.get("total_working_days")
    tww = sc.get("total_worked_weekends")
    cwd = sc.get("consecutive_working_days")
    cdo = sc.get("consecutive_days_off")
    css = sc.get("consecutive_same_shift")
    cww = sc.get("consecutive_worked_weekends")
    iw = sc.get("identical_weekend")

    BIGH = 1e7

    def streaks(bools):
        res = []
        i = 0
        n = len(bools)
        while i < n:
            if bools[i]:
                j = i
                while j < n and bools[j]:
                    j += 1
                res.append((i + 1, j - i))
                i = j
            else:
                i += 1
        return res

    def count_pat(rowmap, pattern):
        L = len(pattern)
        if L == 0:
            return 0
        c = 0
        for a in range(1, D - L + 2):
            ok = True
            for i in range(L):
                dt, sn = pattern[i][0], pattern[i][1]
                d = a + i
                if not daytype_match(dt, d) or not shift_match(sn, rowmap[d]):
                    ok = False
                    break
            if ok:
                c += 1
        return c

    def evaluate(sched, skas):
        tot = 0.0
        staff = {}
        for nid in nids:
            for d in days:
                s = sched[nid][d]
                if s is not None:
                    k = skas[nid][d]
                    staff[(d, s, k)] = staff.get((d, s, k), 0) + 1
                    if k not in nskills[nid]:
                        tot += BIGH if skill_hard else skill_pen
        for d in days:
            for s in shifts:
                for k in skills:
                    dv = dem(d, s, k)
                    st = staff.get((d, s, k), 0)
                    if dem_eq:
                        if st != dv:
                            tot += BIGH
                    else:
                        if st < dv:
                            tot += (dv - st) * unsat_pen
        for nid in nids:
            row = [sched[nid][d] for d in days]
            wb = [s is not None for s in row]
            nwork = sum(wb)
            if twd:
                tot += max(0, twd.get("min", 0) - nwork) * twd.get("penalty_under", 0)
                tot += max(0, nwork - twd.get("max", D)) * twd.get("penalty_over", 0)
            if cwd:
                mn = cwd.get("min", 1)
                mx = cwd.get("max", D)
                for (st, L) in streaks(wb):
                    tot += max(0, L - mx) * cwd.get("penalty_over", 0)
                    if L < mn and st > 1 and st + L - 1 < D:
                        tot += cwd.get("penalty_under", 0)
            if cdo:
                rb = [not b for b in wb]
                mn = cdo.get("min", 1)
                mx = cdo.get("max", D)
                for (st, L) in streaks(rb):
                    tot += max(0, L - mx) * cdo.get("penalty_over", 0)
                    if L < mn and st > 1 and st + L - 1 < D:
                        tot += cdo.get("penalty_under", 0)
            if css:
                mx = css.get("max", D)
                pen = css.get("penalty", 0)
                i = 0
                while i < D:
                    if row[i] is not None:
                        j = i
                        while j < D and row[j] == row[i]:
                            j += 1
                        tot += max(0, (j - i) - mx) * pen
                        i = j
                    else:
                        i += 1
            wwb = [any(sched[nid][d] is not None for d in ds) for ds in wk_days]
            if tww:
                tot += max(0, sum(wwb) - tww.get("max", len(wk_days))) * tww.get("penalty", 0)
            if cww:
                mx = cww.get("max", len(wk_days))
                pen = cww.get("penalty", 0)
                for (st, L) in streaks(wwb):
                    tot += max(0, L - mx) * pen
            if iw and iw.get("enabled"):
                pen = iw.get("penalty", 0)
                for ds in wk_days:
                    vals = set(str(sched[nid][d]) for d in ds)
                    if len(vals) > 1:
                        tot += pen
            rowmap = sched[nid]
            for pat in hard_pats:
                tot += BIGH * count_pat(rowmap, pat.get("pattern", []) or [])
            for pat in soft_pats:
                tot += pat.get("penalty", 0) * count_pat(rowmap, pat.get("pattern", []) or [])
            if pref_on:
                for p in prefs[nid]:
                    d = p.get("day")
                    if not d or d < 1 or d > D:
                        continue
                    pn = p.get("penalty", 0)
                    typ = str(p.get("type", "")).lower()
                    sn = p.get("shift")
                    a = sched[nid][d]
                    if typ == "day_off":
                        if a is not None:
                            tot += pn
                    elif typ == "day_on":
                        if a is None:
                            tot += pn
                    elif typ == "shift_on":
                        if sn in shifts:
                            if a != sn:
                                tot += pn
                        elif str(sn).lower() in ("none", "rest", "off", "-"):
                            if a is not None:
                                tot += pn
                        else:
                            if a is None:
                                tot += pn
                    elif typ == "shift_off":
                        if sn in shifts:
                            if a == sn:
                                tot += pn
                        elif str(sn).lower() in ("none", "rest", "off", "-"):
                            if a is None:
                                tot += pn
                        else:
                            if a is not None:
                                tot += pn
        return tot

    def soldict(sched, skas, obj):
        return {
            "objective_value": float(obj),
            "schedule": {
                nid: {("day_%d" % d): (sched[nid][d] if sched[nid][d] is not None else "None") for d in days}
                for nid in nids
            },
            "skill_assignments": {
                nid: {("day_%d" % d): (skas[nid][d] if skas[nid][d] is not None else "None") for d in days}
                for nid in nids
            },
        }

    def write_out(sched, skas, obj):
        with open(args.solution_path, "w") as f:
            json.dump(soldict(sched, skas, obj), f)

    # ---------------- Greedy fallback / warm start ----------------
    sched0 = {nid: {d: None for d in days} for nid in nids}
    skas0 = {nid: {d: None for d in days} for nid in nids}
    for d in days:
        used = set()
        for s in shifts:
            for k in skills:
                need = dem(d, s, k)
                if need <= 0:
                    continue
                cands = [nid for nid in nids if nid not in used and k in nskills[nid]]
                if not skill_hard:
                    cands += [nid for nid in nids if nid not in used and k not in nskills[nid]]
                for nid in cands[:need]:
                    sched0[nid][d] = s
                    skas0[nid][d] = k
                    used.add(nid)

    def hard_ok(sched, skas):
        for nid in nids:
            for d in days:
                s = sched[nid][d]
                if s is not None and skill_hard and skas[nid][d] not in nskills[nid]:
                    return False
            for pat in hard_pats:
                if count_pat(sched[nid], pat.get("pattern", []) or []) > 0:
                    return False
        if dem_eq:
            staff = {}
            for nid in nids:
                for d in days:
                    s = sched[nid][d]
                    if s is not None:
                        staff[(d, s, skas[nid][d])] = staff.get((d, s, skas[nid][d]), 0) + 1
            for d in days:
                for s in shifts:
                    for k in skills:
                        if staff.get((d, s, k), 0) != dem(d, s, k):
                            return False
        return True

    best = {"obj": float("inf"), "sol": None}
    if D == 0 or not nids:
        write_out(sched0, skas0, 0.0)
        if logger:
            logger.log_solution(0.0, soldict(sched0, skas0, 0.0))
        return

    g_ok = hard_ok(sched0, skas0)
    g_obj = evaluate(sched0, skas0)
    if g_ok:
        best["obj"] = g_obj
        best["sol"] = (sched0, skas0)
        if logger:
            logger.log_solution(g_obj, soldict(sched0, skas0, g_obj))

    # ---------------- MIP ----------------
    try:
        import gurobipy as gp
        from gurobipy import GRB

        m = gp.Model("nrp")
        m.Params.OutputFlag = 0
        m.Params.Seed = 0
        m.Params.MIPGap = 1e-4
        m.Params.NumericFocus = 0
        m.Params.Threads = 1
        m.Params.MIPFocus = 1

        obj = gp.LinExpr()
        x = {}
        wv = {}
        for nid in nids:
            for d in days:
                for s in shifts:
                    x[nid, d, s] = m.addVar(vtype=GRB.BINARY)
                x[nid, d, None] = m.addVar(vtype=GRB.BINARY)
                wv[nid, d] = m.addVar(vtype=GRB.BINARY)
                m.addConstr(gp.quicksum(x[nid, d, s] for s in shifts) == wv[nid, d])
                m.addConstr(wv[nid, d] + x[nid, d, None] == 1)

        y = {}
        for nid in nids:
            allowed = [k for k in skills if k in nskills[nid]] if skill_hard else list(skills)
            for d in days:
                for s in shifts:
                    for k in allowed:
                        v = m.addVar(vtype=GRB.BINARY)
                        y[nid, d, s, k] = v
                        if (not skill_hard) and k not in nskills[nid] and skill_pen:
                            obj += skill_pen * v
                    m.addConstr(gp.quicksum(y[nid, d, s, k] for k in allowed) == x[nid, d, s])

        for d in days:
            for s in shifts:
                for k in skills:
                    dv = dem(d, s, k)
                    expr = gp.quicksum(y[nid, d, s, k] for nid in nids if (nid, d, s, k) in y)
                    if dem_eq:
                        m.addConstr(expr == dv)
                    elif dv > 0:
                        u = m.addVar(lb=0.0)
                        m.addConstr(expr + u >= dv)
                        obj += unsat_pen * u

        # precompute pattern position specs & valid starts
        def pat_specs(pattern):
            L = len(pattern)
            specs = []
            for i in range(L):
                sn = str(pattern[i][1])
                tl = sn.lower()
                if tl in ("any", "*"):
                    specs.append(("w", None))
                elif tl in ("none", "rest", "off", "-"):
                    specs.append(("r", None))
                elif sn in shifts:
                    specs.append(("s", sn))
                else:
                    return None, None
            starts = []
            for a in range(1, D - L + 2):
                if all(daytype_match(pattern[i][0], a + i) for i in range(L)):
                    starts.append(a)
            return specs, starts

        hp_pre = []
        for pat in hard_pats:
            sp, st = pat_specs(pat.get("pattern", []) or [])
            if sp:
                hp_pre.append((sp, st))
        sp_pre = []
        for pat in soft_pats:
            sp, st = pat_specs(pat.get("pattern", []) or [])
            if sp:
                sp_pre.append((sp, st, pat.get("penalty", 0)))

        def ind(nid, d, spec):
            t, sn = spec
            if t == "w":
                return wv[nid, d]
            if t == "r":
                return x[nid, d, None]
            return x[nid, d, sn]

        Wk = len(wk_days)
        for nid in nids:
            Wl = [wv[nid, d] for d in days]
            Rl = [x[nid, d, None] for d in days]
            if twd:
                tot = gp.quicksum(Wl)
                if twd.get("penalty_under", 0):
                    u = m.addVar(lb=0.0)
                    m.addConstr(u >= twd.get("min", 0) - tot)
                    obj += twd["penalty_under"] * u
                if twd.get("penalty_over", 0):
                    o = m.addVar(lb=0.0)
                    m.addConstr(o >= tot - twd.get("max", D))
                    obj += twd["penalty_over"] * o

            def add_consec(lst, mn, mx, pu, po):
                if po and mx < D:
                    for a in range(0, D - mx):
                        o = m.addVar(lb=0.0)
                        m.addConstr(o >= gp.quicksum(lst[a:a + mx + 1]) - mx)
                        obj_ref.append((po, o))
                if pu and mn > 1:
                    for L in range(1, mn):
                        for a in range(1, D - L):
                            v = m.addVar(lb=0.0)
                            m.addConstr(v >= gp.quicksum(lst[a:a + L]) - lst[a - 1] - lst[a + L] - (L - 1))
                            obj_ref.append((pu, v))

            obj_ref = []
            if cwd:
                add_consec(Wl, cwd.get("min", 1), cwd.get("max", D),
                           cwd.get("penalty_under", 0), cwd.get("penalty_over", 0))
            if cdo:
                add_consec(Rl, cdo.get("min", 1), cdo.get("max", D),
                           cdo.get("penalty_under", 0), cdo.get("penalty_over", 0))
            if css and css.get("penalty", 0):
                mx = css.get("max", D)
                if mx < D:
                    for s in shifts:
                        Xl = [x[nid, d, s] for d in days]
                        for a in range(0, D - mx):
                            o = m.addVar(lb=0.0)
                            m.addConstr(o >= gp.quicksum(Xl[a:a + mx + 1]) - mx)
                            obj_ref.append((css["penalty"], o))
            for pen, v in obj_ref:
                obj += pen * v

            if Wk and (tww or cww):
                wwv = [m.addVar(vtype=GRB.BINARY) for _ in range(Wk)]
                for i, ds in enumerate(wk_days):
                    for d in ds:
                        m.addConstr(wwv[i] >= wv[nid, d])
                if tww and tww.get("penalty", 0):
                    o = m.addVar(lb=0.0)
                    m.addConstr(o >= gp.quicksum(wwv) - tww.get("max", Wk))
                    obj += tww["penalty"] * o
                if cww and cww.get("penalty", 0):
                    mx = cww.get("max", Wk)
                    if mx < Wk:
                        for a in range(0, Wk - mx):
                            o = m.addVar(lb=0.0)
                            m.addConstr(o >= gp.quicksum(wwv[a:a + mx + 1]) - mx)
                            obj += cww["penalty"] * o

            if iw and iw.get("enabled") and iw.get("penalty", 0):
                allsh = shifts + [None]
                for ds in wk_days:
                    if len(ds) >= 2:
                        dv = m.addVar(lb=0.0, ub=1.0)
                        for s in allsh:
                            for ii in range(len(ds)):
                                for jj in range(len(ds)):
                                    if ii != jj:
                                        m.addConstr(dv >= x[nid, ds[ii], s] - x[nid, ds[jj], s])
                        obj += iw["penalty"] * dv

            for sp, starts in hp_pre:
                L = len(sp)
                for a in starts:
                    m.addConstr(gp.quicksum(ind(nid, a + i, sp[i]) for i in range(L)) <= L - 1)
            for sp, starts, pen in sp_pre:
                if not pen:
                    continue
                L = len(sp)
                for a in starts:
                    v = m.addVar(lb=0.0)
                    m.addConstr(v >= gp.quicksum(ind(nid, a + i, sp[i]) for i in range(L)) - (L - 1))
                    obj += pen * v

            if pref_on:
                for p in prefs[nid]:
                    d = p.get("day")
                    pn = p.get("penalty", 0)
                    if not d or d < 1 or d > D or not pn:
                        continue
                    typ = str(p.get("type", "")).lower()
                    sn = p.get("shift")
                    if typ == "day_off":
                        obj += pn * wv[nid, d]
                    elif typ == "day_on":
                        obj += pn * (1 - wv[nid, d])
                    else:
                        if sn in shifts:
                            I = x[nid, d, sn]
                        elif str(sn).lower() in ("none", "rest", "off", "-"):
                            I = x[nid, d, None]
                        else:
                            I = wv[nid, d]
                        if typ == "shift_on":
                            obj += pn * (1 - I)
                        elif typ == "shift_off":
                            obj += pn * I

        m.setObjective(obj, GRB.MINIMIZE)

        # warm start
        for nid in nids:
            for d in days:
                s0v = sched0[nid][d]
                for s in shifts:
                    x[nid, d, s].Start = 1.0 if s == s0v else 0.0
                x[nid, d, None].Start = 1.0 if s0v is None else 0.0
                wv[nid, d].Start = 0.0 if s0v is None else 1.0
        for key, v in y.items():
            nid, d, s, k = key
            v.Start = 1.0 if (sched0[nid][d] == s and skas0[nid][d] == k) else 0.0

        xkeys = list(x.keys())
        xvars = [x[k] for k in xkeys]
        ykeys = list(y.keys())
        yvars = [y[k] for k in ykeys]

        def extract(xvals, yvals):
            sched = {nid: {d: None for d in days} for nid in nids}
            for key, val in zip(xkeys, xvals):
                nid, d, s = key
                if s is not None and val > 0.5:
                    sched[nid][d] = s
            skas = {nid: {d: None for d in days} for nid in nids}
            for key, val in zip(ykeys, yvals):
                nid, d, s, k = key
                if val > 0.5 and sched[nid][d] == s:
                    skas[nid][d] = k
            for nid in nids:
                for d in days:
                    if sched[nid][d] is not None and skas[nid][d] is None:
                        ks = nskills[nid]
                        skas[nid][d] = next(iter(ks)) if ks else (skills[0] if skills else "None")
            return sched, skas

        def cb(model, where):
            if where == GRB.Callback.MIPSOL:
                try:
                    xv = model.cbGetSolution(xvars)
                    yv = model.cbGetSolution(yvars)
                    sched, skas = extract(xv, yv)
                    ev = evaluate(sched, skas)
                    if ev < best["obj"]:
                        best["obj"] = ev
                        best["sol"] = (sched, skas)
                        if logger:
                            logger.log_solution(ev, soldict(sched, skas, ev))
                        write_out(sched, skas, ev)
                except Exception:
                    pass

        rem = args.time_limit - (time.time() - t0) - 3.0
        m.Params.TimeLimit = max(1.0, rem)
        m.optimize(cb)

        if m.SolCount > 0:
            xv = [v.X for v in xvars]
            yv = [v.X for v in yvars]
            sched, skas = extract(xv, yv)
            ev = evaluate(sched, skas)
            if ev < best["obj"]:
                best["obj"] = ev
                best["sol"] = (sched, skas)
                if logger:
                    logger.log_solution(ev, soldict(sched, skas, ev))
    except Exception:
        pass

    if best["sol"] is not None:
        sched, skas = best["sol"]
        write_out(sched, skas, best["obj"])
    else:
        write_out(sched0, skas0, g_obj)
        if logger:
            logger.log_solution(g_obj, soldict(sched0, skas0, g_obj))


if __name__ == "__main__":
    main()
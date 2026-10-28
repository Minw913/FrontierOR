import argparse
import json
import math
import random
import sys
import time

INF = float("inf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", required=True)
    ap.add_argument("--solution_path", required=True)
    ap.add_argument("--time_limit", type=int, default=300)
    ap.add_argument("--log_path", default=None)
    args = ap.parse_args()

    t0 = time.time()
    deadline = t0 + max(10, args.time_limit) - 2.0

    logger = None
    try:
        from solution_logger import SolutionLogger
        if args.log_path:
            logger = SolutionLogger(args.log_path, sense="minimize")
    except Exception:
        logger = None

    with open(args.instance_path) as f:
        inst = json.load(f)

    g = inst["global_parameters"]
    S = int(g["subperiods_per_period"])
    prep = float(g["preparation_time_minutes"])
    drate = float(g["driver_salary_per_minute"])
    vrate = float(g["vehicle_operating_cost_per_minute"])
    sal = float(g["employee_salary_per_subperiod"])
    setup_c = float(g["setup_cost"])
    minpaid_min = float(g["min_paid_time_minutes"])
    maxroute_min = float(g["max_shift_route_length_minutes"])
    maxtrips = int(g["max_trips_per_route"])
    Lsetup = int(g["setup_length_subperiods"])
    horizon_end = float(g["horizon_end_minutes"])

    pers = sorted(inst["periods"], key=lambda x: x["start_time"])
    P = len(pers)
    pstart = [float(p["start_time"]) for p in pers]
    pid_list = [int(p["period_id"]) for p in pers]

    subs = sorted(inst["subperiods"], key=lambda x: x["start_time"])
    Tn = len(subs)
    sid_list = [int(s["subperiod_id"]) for s in subs]
    spstart = [float(s["start_time"]) for s in subs]
    spend = [float(s["end_time"]) for s in subs]
    sid2idx = {sid_list[i]: i for i in range(Tn)}

    import bisect
    per_of_sub = [max(0, bisect.bisect_right(pstart, spstart[i] + 1e-6) - 1) for i in range(Tn)]

    M = max(1, int(math.ceil(float(g["min_paid_time_subperiods"]) / max(1, S))))
    Lmaxp = max(int(g["max_shift_route_length_periods"]), M, 1)

    nws = int(inst["num_workstations"])
    nemp = int(inst["num_production_employees"])
    UBe = max(0, min(nws, nemp))
    nveh = int(inst["num_vehicles"])
    ndrv = int(inst["num_drivers"])
    cap = float(inst["vehicle_capacity"])

    rate = {int(pr["product_id"]): float(pr["production_rate_per_subperiod"]) for pr in inst["products"]}

    orders = inst["orders"]
    n = len(orders)
    tt = inst["travel_time_matrix"]

    # node arrays (node = index in matrix, orders[k] -> node k+1)
    tws = [0.0] * (n + 1)
    twe = [0.0] * (n + 1)
    stime = [0.0] * (n + 1)
    space = [0.0] * (n + 1)
    oid_of = [0] * (n + 1)
    dem_of = [dict() for _ in range(n + 1)]
    for k, o in enumerate(orders):
        nd = k + 1
        tws[nd] = float(o["time_window_start"])
        twe[nd] = float(o["time_window_end"])
        stime[nd] = float(o["service_time"])
        space[nd] = float(o["space_requirement"])
        oid_of[nd] = int(o["order_id"])
        d = {}
        for ps, dv in o.get("demands", {}).items():
            if float(dv) > 1e-9:
                d[int(ps)] = float(dv)
        dem_of[nd] = d

    pw = inst.get("production_windows", {})
    valid_idx = {}
    for nd in range(1, n + 1):
        oid = oid_of[nd]
        po = pw.get(str(oid), {})
        for pid in dem_of[nd]:
            e = po.get(str(pid))
            if e:
                ids = sorted(sid2idx[s] for s in e.get("valid_subperiod_ids", []) if s in sid2idx)
                valid_idx[(oid, pid)] = ids

    # earliest feasible departure period for production (min valid subperiod must end by departure)
    plo = [0] * (n + 1)
    for nd in range(1, n + 1):
        thr = -INF
        for pid in dem_of[nd]:
            vids = valid_idx.get((oid_of[nd], pid), [])
            if vids:
                thr = max(thr, spend[vids[0]])
        if thr <= -INF:
            plo[nd] = 0
        else:
            p = bisect.bisect_left(pstart, thr - 1e-6)
            plo[nd] = min(p, P - 1)

    KAPPA = 0.3 * drate * minpaid_min

    # -------------------- routing primitives --------------------
    def sched(seq, p, strict=True):
        base = pstart[p]
        t = base + prep
        loc = 0
        travel = 0.0
        endmax = base
        for nd in seq:
            travel += tt[loc][nd]
            t += tt[loc][nd]
            d = t if t > tws[nd] else tws[nd]
            if strict and d > twe[nd] + 1e-6:
                return None
            t = d + stime[nd]
            rb = t + tt[nd][0]
            if rb > endmax:
                endmax = rb
            loc = nd
        travel += tt[loc][0]
        if strict and endmax - base > maxroute_min + 1e-6:
            return None
        return travel, endmax

    def latest_p(seq, lo, hi):
        for p in range(min(hi, P - 1), max(lo, 0) - 1, -1):
            r = sched(seq, p)
            if r:
                return p, r
        return None

    def make_state(seq):
        pl = max(plo[x] for x in seq)
        res = latest_p(seq, pl, P - 1)
        if res is None:
            return None
        p, (tv, en) = res
        return {"seq": list(seq), "p": p, "plo": pl, "travel": tv, "end": en,
                "load": sum(space[x] for x in seq)}

    def trip_cost(tr):
        return vrate * tr["travel"] + drate * (tr["end"] - pstart[tr["p"]]) + KAPPA

    def force_state(nd):
        p = min(plo[nd], P - 1)
        tv, en = sched([nd], p, strict=False)
        return {"seq": [nd], "p": p, "plo": p, "travel": tv, "end": en, "load": space[nd]}

    def best_solo(nd):
        best = None
        for p in range(plo[nd], P):
            r = sched([nd], p)
            if r:
                dur = r[1] - pstart[p]
                key = (dur, -p)
                if best is None or key < best[0]:
                    best = (key, p, r)
        return best

    def insert_node(trips, nd):
        best = None
        for tr in trips:
            if tr["load"] + space[nd] > cap + 1e-9:
                continue
            plo_new = max(tr["plo"], plo[nd])
            if plo_new > P - 1:
                continue
            cost_old = trip_cost(tr)
            seq = tr["seq"]
            for pos in range(len(seq) + 1):
                ns = seq[:pos] + [nd] + seq[pos:]
                res = latest_p(ns, plo_new, tr["p"])
                if res:
                    p2, (tv, en) = res
                    cost_new = vrate * tv + drate * (en - pstart[p2]) + KAPPA
                    delta = cost_new - cost_old
                    if best is None or delta < best[0]:
                        best = (delta, tr, ns, p2, tv, en, plo_new)
        solo = best_solo(nd)
        solo_cost = INF
        if solo:
            (dur, _np), p, (tv, en) = solo
            solo_cost = vrate * tv + drate * dur + KAPPA
        if best is not None and best[0] <= solo_cost:
            _, tr, ns, p2, tv, en, plo_new = best
            tr["seq"] = ns
            tr["p"] = p2
            tr["travel"] = tv
            tr["end"] = en
            tr["plo"] = plo_new
            tr["load"] += space[nd]
        elif solo:
            (durk, _np), p, (tv, en) = solo
            trips.append({"seq": [nd], "p": p, "plo": plo[nd], "travel": tv,
                          "end": en, "load": space[nd]})
        else:
            trips.append(force_state(nd))

    # -------------------- route chaining --------------------
    maxroutes = min(nveh, ndrv) if min(nveh, ndrv) > 0 else 10 ** 9

    def build_routes(trips):
        cop = []
        for t in trips:
            c = dict(t)
            c["seq"] = list(t["seq"])
            opts = []
            for p in range(c["plo"], P):
                r = sched(c["seq"], p)
                if r:
                    opts.append((p, r[1]))
            if not opts:
                opts = [(c["p"], c["end"])]
            c["opts"] = opts
            cop.append(c)
        cop.sort(key=lambda t: (t["opts"][0][0], t["end"]))
        routes = []
        for t in cop:
            bestapp = None
            for r in routes:
                if len(r["trips"]) >= maxtrips:
                    continue
                lastp = r["trips"][-1]["p"]
                for (p, end) in t["opts"]:
                    if p <= lastp:
                        continue
                    span_new = max(r["end"], end) - r["T0"]
                    if span_new > maxroute_min + 1e-6:
                        continue
                    marg = drate * (max(span_new, minpaid_min) - max(r["span"], minpaid_min))
                    key = (marg, -p)
                    if bestapp is None or key < bestapp[0]:
                        bestapp = (key, r, p, end)
            bestnew = None
            for (p, end) in t["opts"]:
                dur = end - pstart[p]
                c = drate * max(dur, minpaid_min)
                key = (c, -p)
                if bestnew is None or key < bestnew[0]:
                    bestnew = (key, p, end)
            use_new = False
            if bestapp is None:
                use_new = True
            elif len(routes) < maxroutes and bestnew is not None and bestnew[0][0] < bestapp[0][0]:
                use_new = True
            if use_new and bestnew is not None:
                _, p, end = bestnew
                t["p"] = p
                t["end"] = end
                routes.append({"trips": [t], "T0": pstart[p], "end": end,
                               "span": end - pstart[p]})
            else:
                _, r, p, end = bestapp
                t["p"] = p
                t["end"] = end
                r["trips"].append(t)
                r["end"] = max(r["end"], end)
                r["span"] = r["end"] - r["T0"]
        return routes

    # -------------------- production --------------------
    def compute_allowed(depart):
        allowed = {}
        dom = {}
        for nd in range(1, n + 1):
            oid = oid_of[nd]
            for pid, dv in dem_of[nd].items():
                vids = valid_idx.get((oid, pid), [])
                if not vids:
                    continue
                dp = depart.get(oid, horizon_end)
                al = [s for s in vids if spend[s] <= dp + 1e-6]
                if not al:
                    al = list(vids)
                allowed[(oid, pid)] = (al, dv)
                dom.setdefault(pid, set()).update(vids)
        return allowed, dom

    def empty_prod():
        return {"q": {}, "emp": {}, "setups": {}, "w": [0] * P, "z": [0] * P, "cost": 0.0}

    def fallback_production(depart):
        allowed, dom = compute_allowed(depart)
        if not allowed:
            return empty_prod()
        qmap = {}
        load = {}
        for (oid, pid), (al, dv) in allowed.items():
            peru = rate[pid] * max(UBe, 1)
            rem = dv
            for s in reversed(al):
                room = peru - load.get((pid, s), 0.0)
                if room <= 1e-9:
                    continue
                x = min(rem, room)
                qmap[(oid, pid, s)] = qmap.get((oid, pid, s), 0.0) + x
                load[(pid, s)] = load.get((pid, s), 0.0) + x
                rem -= x
                if rem <= 1e-9:
                    break
            if rem > 1e-9:
                s = al[-1]
                qmap[(oid, pid, s)] = qmap.get((oid, pid, s), 0.0) + rem
                load[(pid, s)] = load.get((pid, s), 0.0) + rem
        y = {}
        u = {}
        for pid, ss in dom.items():
            prev = 0
            for s in sorted(ss):
                need = int(math.ceil(load.get((pid, s), 0.0) / rate[pid] - 1e-9)) if rate[pid] > 0 else 0
                y[(pid, s)] = need
                u[(pid, s)] = max(0, need - prev)
                prev = need
        occ = [0] * Tn
        for (pid, s), v in y.items():
            occ[s] += v
        if Lsetup >= 1:
            for (pid, s), v in u.items():
                for s2 in range(s - Lsetup + 1, s + Lsetup):
                    if 0 <= s2 < Tn:
                        occ[s2] += v
        wneed = [0] * P
        for s in range(Tn):
            p = per_of_sub[s]
            if occ[s] > wneed[p]:
                wneed[p] = occ[s]
        w = [0] * P
        z = [0] * P
        used = [p for p in range(P) if wneed[p] > 0]
        if used:
            Wmax = max(wneed)
            pf, pl = min(used), max(used)
            if pl - pf + 1 < M:
                pf = max(0, pl - M + 1)
                pl = min(P - 1, pf + M - 1)
            pcur = pf
            while pcur <= pl:
                blk = min(Lmaxp, pl - pcur + 1)
                if blk < M:
                    blk = M
                z[pcur] += Wmax
                for p in range(pcur, min(pcur + blk, P)):
                    w[p] = Wmax
                pcur += blk
        cost = sal * S * sum(w) + setup_c * sum(u.values())
        return {"q": qmap, "emp": y, "setups": u, "w": w, "z": z, "cost": cost}

    def solve_production(depart, tl):
        allowed, dom = compute_allowed(depart)
        if not allowed:
            return empty_prod()
        if tl <= 0.5:
            return None
        try:
            import gurobipy as gpy
            from gurobipy import GRB
            m = gpy.Model("prod")
            m.Params.OutputFlag = 0
            m.Params.Seed = 0
            m.Params.MIPGap = 1e-4
            m.Params.NumericFocus = 0
            m.Params.Threads = 1
            m.Params.TimeLimit = max(1.0, tl)
            y = {}
            u = {}
            for pid, ss in dom.items():
                for s in ss:
                    y[(pid, s)] = m.addVar(vtype=GRB.INTEGER, ub=UBe)
                    u[(pid, s)] = m.addVar(vtype=GRB.INTEGER, ub=UBe)
            q = {}
            sl = {}
            for (oid, pid), (al, dv) in allowed.items():
                vs = []
                for s in al:
                    v = m.addVar()
                    q[(oid, pid, s)] = v
                    vs.append(v)
                sv = m.addVar()
                sl[(oid, pid)] = sv
                m.addConstr(gpy.quicksum(vs) + sv == dv)
            loadmap = {}
            for (oid, pid, s), v in q.items():
                loadmap.setdefault((pid, s), []).append(v)
            for (pid, s), vs in loadmap.items():
                m.addConstr(gpy.quicksum(vs) <= rate[pid] * y[(pid, s)])
            for pid, ss in dom.items():
                for s in sorted(ss):
                    prev = y.get((pid, s - 1), 0)
                    m.addConstr(u[(pid, s)] >= y[(pid, s)] - prev)
            w = [m.addVar(vtype=GRB.INTEGER, ub=UBe) for _ in range(P)]
            for s in range(Tn):
                terms = []
                for pid, ss in dom.items():
                    if s in ss:
                        terms.append(y[(pid, s)])
                    if Lsetup >= 1:
                        for s2 in range(s - Lsetup + 1, s + Lsetup):
                            if (pid, s2) in u:
                                terms.append(u[(pid, s2)])
                if terms:
                    m.addConstr(gpy.quicksum(terms) <= w[per_of_sub[s]])
            zv = [m.addVar(vtype=GRB.INTEGER, ub=UBe * P) for _ in range(P)]
            ev = [m.addVar(vtype=GRB.INTEGER, ub=UBe * P) for _ in range(P)]
            Zc = []
            Ec = []
            zex = gpy.LinExpr()
            eex = gpy.LinExpr()
            for p in range(P):
                zex = zex + zv[p]
                eex = eex + ev[p]
                Zc.append(gpy.LinExpr(zex))
                Ec.append(gpy.LinExpr(eex))
            for p in range(P):
                m.addConstr(w[p] == Zc[p] - (Ec[p - 1] if p >= 1 else 0))
                if p - M + 1 >= 0:
                    m.addConstr(Ec[p] <= Zc[p - M + 1])
                else:
                    m.addConstr(Ec[p] <= 0)
                if p - Lmaxp + 1 >= 0:
                    m.addConstr(Ec[p] >= Zc[p - Lmaxp + 1])
            m.addConstr(Ec[P - 1] == Zc[P - 1])
            m.setObjective(sal * S * gpy.quicksum(w) + setup_c * gpy.quicksum(u.values())
                           + 1e7 * gpy.quicksum(sl.values()))
            m.optimize()
            if m.SolCount == 0:
                return None
            qout = {k: v.X for k, v in q.items() if v.X > 1e-9}
            emp = {k: int(round(v.X)) for k, v in y.items()}
            setu = {k: int(round(v.X)) for k, v in u.items()}
            wl = [int(round(v.X)) for v in w]
            zl = [int(round(v.X)) for v in zv]
            cost = sal * S * sum(wl) + setup_c * sum(setu.values())
            return {"q": qout, "emp": emp, "setups": setu, "w": wl, "z": zl, "cost": cost}
        except Exception:
            return None

    # -------------------- solution assembly --------------------
    def build_dict(routes, prod, obj):
        rlist = []
        vi = 0
        for r in routes:
            tl = []
            for k, t in enumerate(sorted(r["trips"], key=lambda x: x["p"])):
                tl.append({"trip": k + 1,
                           "customers": [oid_of[nd] for nd in t["seq"]],
                           "start_period": pid_list[t["p"]]})
            rlist.append({"vehicle": vi, "trips": tl})
            vi += 1
        psched = [{"order": oid, "product": pid, "subperiod": sid_list[s],
                   "quantity": float(v)}
                  for (oid, pid, s), v in sorted(prod["q"].items()) if v > 1e-9]
        emp = [{"product": pid, "subperiod": sid_list[s], "employees": int(v)}
               for (pid, s), v in sorted(prod["emp"].items())]
        setu = [{"product": pid, "subperiod": sid_list[s], "setups": int(v)}
                for (pid, s), v in sorted(prod["setups"].items())]
        return {"objective_value": float(obj),
                "routes": rlist,
                "production_schedule": psched,
                "production_staffing": {
                    "employees_by_product_subperiod": emp,
                    "setups_by_product_subperiod": setu,
                    "workstations_by_period": [int(x) for x in prod["w"]],
                    "shift_starts_by_period": [int(x) for x in prod["z"]]}}

    best = {"obj": INF, "dict": None}
    prodcache = {}

    def evaluate(trips, prod_tl):
        routes = build_routes(trips)
        depart = {}
        for r in routes:
            for t in r["trips"]:
                dep = pstart[t["p"]]
                for nd in t["seq"]:
                    depart[oid_of[nd]] = dep
        key = tuple(sorted(depart.items()))
        if key in prodcache:
            prod = prodcache[key]
        else:
            tl = min(prod_tl, deadline - time.time() - 1.0)
            prod = solve_production(depart, tl)
            if prod is None:
                prod = fallback_production(depart)
            prodcache[key] = prod
        dcost = sum(drate * max(r["span"], minpaid_min) for r in routes)
        vcost = sum(vrate * t["travel"] for r in routes for t in r["trips"])
        obj = dcost + vcost + prod["cost"]
        if obj < best["obj"] - 1e-9:
            sd = build_dict(routes, prod, obj)
            best["obj"] = obj
            best["dict"] = sd
            if logger:
                try:
                    logger.log_solution(obj, sd)
                except Exception:
                    pass
        return obj

    # -------------------- local search --------------------
    def ls_pass(trips, tend):
        improved = False
        for nd in range(1, n + 1):
            if time.time() > tend:
                break
            tr_a = None
            for tr in trips:
                if nd in tr["seq"]:
                    tr_a = tr
                    break
            if tr_a is None:
                continue
            seq_a = [x for x in tr_a["seq"] if x != nd]
            cost_a_old = trip_cost(tr_a)
            st_a = None
            cost_a_new = 0.0
            if seq_a:
                st_a = make_state(seq_a)
                if st_a is None:
                    continue
                cost_a_new = trip_cost(st_a)
            bestmv = None  # (delta, kind, ...)
            for tr_b in trips:
                if tr_b is tr_a:
                    continue
                if tr_b["load"] + space[nd] > cap + 1e-9:
                    continue
                plo_b = max(tr_b["plo"], plo[nd])
                if plo_b > P - 1:
                    continue
                cost_b_old = trip_cost(tr_b)
                sq = tr_b["seq"]
                for pos in range(len(sq) + 1):
                    ns = sq[:pos] + [nd] + sq[pos:]
                    res = latest_p(ns, plo_b, tr_b["p"])
                    if res:
                        p2, (tv, en) = res
                        cost_b_new = vrate * tv + drate * (en - pstart[p2]) + KAPPA
                        delta = (cost_a_new - cost_a_old) + (cost_b_new - cost_b_old)
                        if bestmv is None or delta < bestmv[0]:
                            bestmv = (delta, "inter", tr_b, ns, p2, tv, en, plo_b)
            if seq_a:
                pl_full = max(max(plo[x] for x in seq_a), plo[nd])
                for pos in range(len(seq_a) + 1):
                    ns = seq_a[:pos] + [nd] + seq_a[pos:]
                    if ns == tr_a["seq"]:
                        continue
                    res = latest_p(ns, pl_full, P - 1)
                    if res:
                        p2, (tv, en) = res
                        cost_ns = vrate * tv + drate * (en - pstart[p2]) + KAPPA
                        delta = cost_ns - cost_a_old
                        if bestmv is None or delta < bestmv[0]:
                            bestmv = (delta, "intra", ns, p2, tv, en, pl_full)
            if bestmv is not None and bestmv[0] < -1e-6:
                improved = True
                if bestmv[1] == "inter":
                    _, _, tr_b, ns, p2, tv, en, plo_b = bestmv
                    tr_b["seq"] = ns
                    tr_b["p"] = p2
                    tr_b["travel"] = tv
                    tr_b["end"] = en
                    tr_b["plo"] = plo_b
                    tr_b["load"] += space[nd]
                    if st_a is None:
                        trips.remove(tr_a)
                    else:
                        tr_a.clear()
                        tr_a.update(st_a)
                else:
                    _, _, ns, p2, tv, en, pl_full = bestmv
                    tr_a["seq"] = ns
                    tr_a["p"] = p2
                    tr_a["travel"] = tv
                    tr_a["end"] = en
                    tr_a["plo"] = pl_full
        return improved

    def total_est(trips):
        return sum(trip_cost(t) for t in trips)

    def copy_trips(trips):
        return [dict(t, seq=list(t["seq"])) for t in trips]

    # -------------------- main flow --------------------
    if n == 0:
        sol = {"objective_value": 0.0, "routes": [], "production_schedule": [],
               "production_staffing": {"employees_by_product_subperiod": [],
                                       "setups_by_product_subperiod": [],
                                       "workstations_by_period": [0] * P,
                                       "shift_starts_by_period": [0] * P}}
        if logger:
            try:
                logger.log_solution(0.0, sol)
            except Exception:
                pass
        with open(args.solution_path, "w") as f:
            json.dump(sol, f)
        return

    # construction
    trips = []
    for nd in sorted(range(1, n + 1), key=lambda x: (twe[x], tws[x])):
        insert_node(trips, nd)

    try:
        evaluate(trips, min(40.0, max(5.0, 0.3 * (deadline - time.time()))))
    except Exception:
        pass

    # local search
    ls_end = min(deadline - 8.0, t0 + 0.5 * max(10, args.time_limit))
    while time.time() < ls_end:
        imp = ls_pass(trips, ls_end)
        if not imp:
            break
        try:
            evaluate(trips, 15.0)
        except Exception:
            pass

    # perturbation (ruin & recreate)
    rng = random.Random(0)
    proxy_best = total_est(trips)
    best_trips = copy_trips(trips)
    kdel = max(1, min(max(2, n // 6), 15))
    while time.time() < deadline - 6.0:
        cand = copy_trips(best_trips)
        rem = rng.sample(range(1, n + 1), min(kdel, n))
        remset = set(rem)
        newtrips = []
        extra = []
        for tr in cand:
            ns = [x for x in tr["seq"] if x not in remset]
            if not ns:
                continue
            if len(ns) == len(tr["seq"]):
                newtrips.append(tr)
                continue
            st = make_state(ns)
            if st is None:
                extra.extend(ns)
            else:
                newtrips.append(st)
        cand = newtrips
        pool = list(rem) + extra
        rng.shuffle(pool)
        for nd in pool:
            insert_node(cand, nd)
        ls_pass(cand, min(time.time() + 6.0, deadline - 5.0))
        pe = total_est(cand)
        if pe < proxy_best - 1e-6:
            proxy_best = pe
            best_trips = copy_trips(cand)
            try:
                evaluate(best_trips, 12.0)
            except Exception:
                pass

    if best["dict"] is None:
        # last resort with fallback production
        routes = build_routes(best_trips)
        depart = {}
        for r in routes:
            for t in r["trips"]:
                for nd in t["seq"]:
                    depart[oid_of[nd]] = pstart[t["p"]]
        prod = fallback_production(depart)
        dcost = sum(drate * max(r["span"], minpaid_min) for r in routes)
        vcost = sum(vrate * t["travel"] for r in routes for t in r["trips"])
        obj = dcost + vcost + prod["cost"]
        best["obj"] = obj
        best["dict"] = build_dict(routes, prod, obj)
        if logger:
            try:
                logger.log_solution(obj, best["dict"])
            except Exception:
                pass

    with open(args.solution_path, "w") as f:
        json.dump(best["dict"], f)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
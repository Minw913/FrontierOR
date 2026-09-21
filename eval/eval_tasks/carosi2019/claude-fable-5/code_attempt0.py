import argparse
import json
import time
import bisect
import re


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", required=True)
    ap.add_argument("--solution_path", required=True)
    ap.add_argument("--time_limit", type=int, default=300)
    ap.add_argument("--log_path", default=None)
    args = ap.parse_args()
    t0 = time.time()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path) as f:
        inst = json.load(f)

    depot = inst["topology"]["depot"]
    tw_info = {w["index"]: (float(w["ideal_headway_seconds"]),
                            float(w["min_headway_seconds"]),
                            float(w["max_headway_seconds"]))
               for w in inst["time_windows"]}

    stop_tab = {}
    for term, lst in inst.get("stopping_times", {}).items():
        d = stop_tab.setdefault(term, {})
        for e in lst:
            d[e["time_window_index"]] = (float(e["min_stopping_time_minutes"]),
                                         float(e["max_stopping_time_minutes"]))
    pull_tab = {}
    for term, lst in inst.get("pull_in_out_times", {}).items():
        d = pull_tab.setdefault(term, {})
        for e in lst:
            d[e["time_window_index"]] = (float(e["pull_out_time_minutes"]),
                                         float(e["pull_in_time_minutes"]))

    def near(d, tw):
        if tw in d:
            return d[tw]
        k = min(d, key=lambda w: abs(w - tw))
        return d[k]

    def get_stop(term, tw):
        d = stop_tab.get(term)
        if not d:
            return (0.0, 1e7)
        return near(d, tw)

    def get_pull(term, tw):
        d = pull_tab.get(term)
        if not d:
            return (0.0, 0.0)
        return near(d, tw)

    def depot_min(tw):
        d = stop_tab.get(depot)
        if not d:
            return 0.0
        return near(d, tw)[0]

    objf = inst.get("objective_function", {}) or {}
    alpha = float(objf.get("alpha", 1.0))
    desc = str(objf.get("description", ""))
    tt_weight = (1.0 - alpha) if re.search(r"\(\s*1(\.0)?\s*-\s*alpha\s*\)", desc, re.I) else 1.0

    def find_veh_cost(o):
        if isinstance(o, dict):
            for k, v in o.items():
                kl = k.lower()
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    if ((("cost" in kl or "price" in kl) and
                         ("vehicle" in kl or "fleet" in kl or "bus" in kl or
                          "deployment" in kl or "fixed" in kl)) or
                            kl in ("c_veh", "cveh", "vehicle_cost", "per_vehicle_cost")):
                        return float(v)
            for v in o.values():
                r = find_veh_cost(v)
                if r is not None:
                    return r
        elif isinstance(o, list):
            for v in o:
                r = find_veh_cost(v)
                if r is not None:
                    return r
        return None

    veh_cost = find_veh_cost(inst)
    if veh_cost is None:
        mm = re.search(r"(\d+(?:\.\d+)?)\s*[*x\u00d7]\s*(?:num_?vehicles?|n_?vehicles?|fleet|vehicles?|\|v\|)",
                       desc, re.I)
        if not mm:
            mm = re.search(r"(?:num_?vehicles?|n_?vehicles?|fleet_?size|vehicles?)\s*[*x\u00d7]\s*(\d+(?:\.\d+)?)",
                           desc, re.I)
        veh_cost = float(mm.group(1)) if mm else 10000.0

    pdesc = str((inst.get("penalty_function", {}) or {}).get("description", "")).lower()
    pen_minutes = ("minute" in pdesc and "second" not in pdesc)

    # ---------------- trips ----------------
    tr = {}
    for t in inst["potential_trips"]:
        tr[t["id"]] = {"s": t["start_terminal"], "e": t["end_terminal"],
                       "dep": float(t["departure_time_minutes"]),
                       "arr": float(t["arrival_time_minutes"]),
                       "ms": float(t["main_stop_arrival_time_minutes"]),
                       "tw": t["time_window_index"], "pat": t["pattern_id"]}
    ids = list(tr)
    po_time = {}
    ready = {}
    poc = {}
    pic = {}
    for i, t in tr.items():
        po, _ = get_pull(t["s"], t["tw"])
        _, pi = get_pull(t["e"], t["tw"])
        poc[i] = po
        pic[i] = pi
        po_time[i] = t["dep"] - po
        ready[i] = t["arr"] + pi + depot_min(t["tw"])

    def pen(a, b):
        ta, tb = tr[a], tr[b]
        ideal = tw_info.get(ta["tw"], (0.0, 0.0, 0.0))[0]
        d = 60.0 * (tb["ms"] - ta["ms"]) - ideal
        if pen_minutes:
            d /= 60.0
        return d * d

    pats = sorted({t["pat"] for t in tr.values()})
    init_tw = dict(inst.get("initial_trip_time_windows", {}) or {})
    fin_tw = dict(inst.get("final_trip_time_windows", {}) or {})
    pat_trips = {p: sorted([i for i in ids if tr[i]["pat"] == p],
                           key=lambda i: (tr[i]["ms"], i)) for p in pats}
    for p in pats:
        if p not in init_tw:
            init_tw[p] = min(tr[i]["tw"] for i in pat_trips[p])
        if p not in fin_tw:
            fin_tw[p] = max(tr[i]["tw"] for i in pat_trips[p])

    # ---------------- timetable arcs ----------------
    tt_out = {p: {} for p in pats}
    for p in pats:
        lst = pat_trips[p]
        ms_list = [tr[i]["ms"] for i in lst]
        for i in lst:
            _, mn, mx = tw_info.get(tr[i]["tw"], (0.0, 0.0, 0.0))
            lo = tr[i]["ms"] + mn / 60.0
            hi = tr[i]["ms"] + mx / 60.0
            a = bisect.bisect_left(ms_list, lo - 1e-9)
            b = bisect.bisect_right(ms_list, hi + 1e-9)
            out = []
            for q in range(a, b):
                j = lst[q]
                if tr[j]["ms"] > tr[i]["ms"] + 1e-9:
                    out.append((j, pen(i, j)))
            tt_out[p][i] = out

    # ---------------- in-line arcs ----------------
    starts_by = {}
    for i in ids:
        starts_by.setdefault(tr[i]["s"], []).append(i)
    for a in starts_by:
        starts_by[a].sort(key=lambda i: (tr[i]["dep"], i))
    dep_lists = {a: [tr[i]["dep"] for i in L] for a, L in starts_by.items()}

    inline = []
    for i in ids:
        a = tr[i]["e"]
        if a not in starts_by:
            continue
        mn, mx = get_stop(a, tr[i]["tw"])
        lo = tr[i]["arr"] + mn
        hi = tr[i]["arr"] + mx
        L = starts_by[a]
        D = dep_lists[a]
        u = bisect.bisect_left(D, lo - 1e-9)
        v = bisect.bisect_right(D, hi + 1e-9)
        for q in range(u, v):
            j = L[q]
            if j != i:
                inline.append((i, j, tr[j]["dep"] - tr[i]["arr"] - mn))

    # ---------------- solution builder ----------------
    def build_solution(tt_chains, veh_chains):
        tt_cost = 0.0
        tt_arcs_used = {}
        for p, ch in tt_chains.items():
            arcs = [["source", str(ch[0])]]
            for a, b in zip(ch, ch[1:]):
                arcs.append([str(a), str(b)])
                tt_cost += pen(a, b)
            arcs.append([str(ch[-1]), "sink"])
            tt_arcs_used[p] = arcs
        vs = veh_cost * len(veh_chains)
        vs_flows = {"%s-->%s" % (depot, depot): float(len(veh_chains))}
        sel = []
        for ch in veh_chains:
            sel.extend(ch)
            vs += poc[ch[0]] + pic[ch[-1]]
            vs_flows["%s-->%s" % (depot, ch[0])] = 1.0
            vs_flows["%s-->%s" % (ch[-1], depot)] = 1.0
            for t in ch:
                vs_flows["%s-->%s" % (t, t)] = 1.0
            for a, b in zip(ch, ch[1:]):
                vs_flows["%s-->%s" % (a, b)] = 1.0
                ti, tj = tr[a], tr[b]
                if ti["e"] == tj["s"]:
                    mn, _ = get_stop(ti["e"], ti["tw"])
                    vs += (tj["dep"] - ti["arr"] - mn)
                else:
                    vs += pic[a] + poc[b]
        obj = alpha * vs + tt_weight * tt_cost
        sol = {"objective_value": float(obj),
               "selected_trips": sorted(sel),
               "num_vehicles": len(veh_chains),
               "tt_arcs_used": tt_arcs_used,
               "vs_flows": vs_flows}
        return obj, sol

    # ---------------- fallback heuristic ----------------
    def tt_dp():
        chains = {}
        for p in pats:
            lst = pat_trips[p]
            INF = float("inf")
            dp = {i: INF for i in lst}
            par = {}
            for i in lst:
                if tr[i]["tw"] == init_tw[p]:
                    dp[i] = 0.0
            for i in lst:
                di = dp[i]
                if di == INF:
                    continue
                for j, pe in tt_out[p][i]:
                    if di + pe < dp[j]:
                        dp[j] = di + pe
                        par[j] = i
            best = None
            for i in lst:
                if tr[i]["tw"] == fin_tw[p] and dp[i] < INF:
                    if best is None or dp[i] < dp[best]:
                        best = i
            if best is None:
                return None
            ch = [best]
            while ch[-1] in par:
                ch.append(par[ch[-1]])
            ch.reverse()
            chains[p] = ch
        return chains

    def greedy_vs(sel):
        order = sorted(sel, key=lambda i: (tr[i]["dep"], i))
        chains = []
        last = []
        for j in order:
            tj = tr[j]
            best = None
            for vi, i in enumerate(last):
                ti = tr[i]
                if ti["e"] == tj["s"]:
                    mn, mx = get_stop(ti["e"], ti["tw"])
                    gap = tj["dep"] - ti["arr"]
                    if mn - 1e-9 <= gap <= mx + 1e-9:
                        c = alpha * (gap - mn)
                        if best is None or c < best[0]:
                            best = (c, vi)
                else:
                    if ready[i] <= po_time[j] + 1e-9:
                        c = alpha * (pic[i] + poc[j])
                        if best is None or c < best[0]:
                            best = (c, vi)
            cnew = alpha * (veh_cost + poc[j])
            if best is None or cnew < best[0]:
                chains.append([j])
                last.append(j)
            else:
                vi = best[1]
                chains[vi].append(j)
                last[vi] = j
        return chains

    state = {"obj": float("inf"), "sol": None}

    def accept(o, s):
        if o < state["obj"] - 1e-9:
            state["obj"] = o
            state["sol"] = s
            if logger:
                logger.log_solution(float(o), s)
            try:
                with open(args.solution_path, "w") as f:
                    json.dump(s, f)
            except Exception:
                pass

    tt_chains0 = tt_dp()
    veh0 = None
    if tt_chains0 is not None:
        sel0 = [i for ch in tt_chains0.values() for i in ch]
        veh0 = greedy_vs(sel0)
        o0, s0 = build_solution(tt_chains0, veh0)
        accept(o0, s0)

    # ---------------- MIP ----------------
    try:
        import gurobipy as gp
        from gurobipy import GRB

        m = gp.Model("itvs")
        m.Params.OutputFlag = 0
        m.Params.Seed = 0
        m.Params.MIPGap = 1e-4
        m.Params.NumericFocus = 0
        m.Params.Threads = 1
        rem = args.time_limit - (time.time() - t0) - 3.0
        m.Params.TimeLimit = max(1.0, rem)

        x = {i: m.addVar(vtype=GRB.BINARY) for i in ids}
        ttv = {}
        src = {}
        snk = {}
        tt_in = {i: [] for i in ids}
        tt_ov = {i: [] for i in ids}
        for p in pats:
            for i in pat_trips[p]:
                for j, pe in tt_out[p][i]:
                    v = m.addVar(vtype=GRB.BINARY, obj=tt_weight * pe)
                    ttv[(i, j)] = v
                    tt_ov[i].append(v)
                    tt_in[j].append(v)
            s_arcs = []
            k_arcs = []
            for i in pat_trips[p]:
                if tr[i]["tw"] == init_tw[p]:
                    v = m.addVar(vtype=GRB.BINARY)
                    src[(p, i)] = v
                    tt_in[i].append(v)
                    s_arcs.append(v)
                if tr[i]["tw"] == fin_tw[p]:
                    v = m.addVar(vtype=GRB.BINARY)
                    snk[(p, i)] = v
                    tt_ov[i].append(v)
                    k_arcs.append(v)
            m.addConstr(gp.quicksum(s_arcs) == 1)
            m.addConstr(gp.quicksum(k_arcs) == 1)
        for i in ids:
            m.addConstr(gp.quicksum(tt_in[i]) == x[i])
            m.addConstr(gp.quicksum(tt_ov[i]) == x[i])

        inl = {}
        vin = {i: [] for i in ids}
        vout = {i: [] for i in ids}
        for (i, j, w) in inline:
            v = m.addVar(vtype=GRB.BINARY, obj=alpha * w)
            inl[(i, j)] = v
            vout[i].append(v)
            vin[j].append(v)
        Fv = {i: m.addVar(vtype=GRB.BINARY, obj=alpha * poc[i]) for i in ids}
        PIv = {i: m.addVar(vtype=GRB.BINARY, obj=alpha * pic[i]) for i in ids}
        pull_terms = sorted({tr[i]["e"] for i in ids})
        tl_times = {}
        for a in pull_terms:
            times = set()
            for i in ids:
                if tr[i]["e"] == a:
                    times.add(ready[i])
                if tr[i]["s"] != a:
                    times.add(po_time[i])
            tl_times[a] = sorted(times)
        dh = {}
        sup_at = {}
        dem_at = {}
        for i in ids:
            sup_at.setdefault((tr[i]["e"], ready[i]), []).append(i)
            for a in pull_terms:
                if a != tr[i]["s"]:
                    v = m.addVar(vtype=GRB.BINARY, obj=alpha * poc[i])
                    dh[(a, i)] = v
                    dem_at.setdefault((a, po_time[i]), []).append(i)
        circ = m.addVar(lb=0.0, obj=alpha * veh_cost)
        toC = []
        for a in pull_terms:
            Ts = tl_times[a]
            prev = None
            for idx, tm in enumerate(Ts):
                v = m.addVar(lb=0.0)
                inflow = gp.LinExpr()
                if prev is not None:
                    inflow += prev
                for i in sup_at.get((a, tm), []):
                    inflow += PIv[i]
                outflow = gp.LinExpr()
                outflow += v
                for i in dem_at.get((a, tm), []):
                    outflow += dh[(a, i)]
                m.addConstr(inflow == outflow)
                prev = v
            if Ts:
                toC.append(prev)
        m.addConstr(circ == gp.quicksum(toC))
        m.addConstr(circ == gp.quicksum(Fv[i] for i in ids))
        for i in ids:
            m.addConstr(Fv[i] + gp.quicksum(dh[(a, i)] for a in pull_terms if a != tr[i]["s"]) +
                        gp.quicksum(vin[i]) == x[i])
            m.addConstr(PIv[i] + gp.quicksum(vout[i]) == x[i])
        m.ModelSense = GRB.MINIMIZE

        # warm start
        if tt_chains0 is not None and veh0 is not None:
            for v in list(ttv.values()) + list(src.values()) + list(snk.values()) + \
                    list(inl.values()) + list(Fv.values()) + list(PIv.values()) + list(dh.values()):
                v.Start = 0.0
            for i in ids:
                x[i].Start = 0.0
            for ch in tt_chains0.values():
                for i in ch:
                    x[i].Start = 1.0
            for p, ch in tt_chains0.items():
                src[(p, ch[0])].Start = 1.0
                snk[(p, ch[-1])].Start = 1.0
                for a2, b2 in zip(ch, ch[1:]):
                    if (a2, b2) in ttv:
                        ttv[(a2, b2)].Start = 1.0
            for ch in veh0:
                Fv[ch[0]].Start = 1.0
                PIv[ch[-1]].Start = 1.0
                for a2, b2 in zip(ch, ch[1:]):
                    if tr[a2]["e"] == tr[b2]["s"]:
                        if (a2, b2) in inl:
                            inl[(a2, b2)].Start = 1.0
                    else:
                        PIv[a2].Start = 1.0
                        if (tr[a2]["e"], b2) in dh:
                            dh[(tr[a2]["e"], b2)].Start = 1.0

        def extract(g):
            selset = {i for i in ids if g(x[i]) > 0.5}
            ttc = {}
            for p in pats:
                start = None
                for i in pat_trips[p]:
                    if (p, i) in src and g(src[(p, i)]) > 0.5:
                        start = i
                        break
                if start is None:
                    return None
                succ = {}
                for i in pat_trips[p]:
                    for j, pe in tt_out[p][i]:
                        if g(ttv[(i, j)]) > 0.5:
                            succ[i] = j
                ch = [start]
                seen = {start}
                while True:
                    nx = succ.get(ch[-1])
                    if nx is None or nx in seen:
                        break
                    ch.append(nx)
                    seen.add(nx)
                ttc[p] = ch
            isucc = {}
            for (i, j), v in inl.items():
                if i in selset and g(v) > 0.5:
                    isucc[i] = j
            piu = {i for i in selset if g(PIv[i]) > 0.5}
            starts = [i for i in selset if g(Fv[i]) > 0.5]
            dsucc = {}
            for a in pull_terms:
                sup = sorted([(ready[i], i) for i in piu if tr[i]["e"] == a])
                dem = sorted([(po_time[i], i) for i in selset
                              if tr[i]["s"] != a and g(dh[(a, i)]) > 0.5])
                k = 0
                for tm, j in dem:
                    if k < len(sup):
                        dsucc[sup[k][1]] = j
                        k += 1
            chains = []
            vis = set()

            def walk(t):
                ch = [t]
                vis.add(t)
                while True:
                    nx = isucc.get(ch[-1])
                    if nx is None:
                        nx = dsucc.get(ch[-1])
                    if nx is None or nx in vis:
                        break
                    ch.append(nx)
                    vis.add(nx)
                return ch

            for t in sorted(starts, key=lambda i: (tr[i]["dep"], i)):
                if t not in vis:
                    chains.append(walk(t))
            for t in sorted(selset, key=lambda i: (tr[i]["dep"], i)):
                if t not in vis:
                    chains.append(walk(t))
            return build_solution(ttc, chains)

        cbvars = None

        def cb(model, where):
            if where == GRB.Callback.MIPSOL:
                try:
                    vals = model.cbGetSolution(cbvars)
                    vm = {id(v): val for v, val in zip(cbvars, vals)}
                    res = extract(lambda v: vm[id(v)])
                    if res is not None:
                        accept(res[0], res[1])
                except Exception:
                    pass

        m.update()
        cbvars = m.getVars()
        m.optimize(cb)

        if m.SolCount > 0:
            try:
                res = extract(lambda v: v.X)
                if res is not None:
                    accept(res[0], res[1])
            except Exception:
                pass
    except Exception:
        pass

    if state["sol"] is None:
        state["sol"] = {"objective_value": 1e18, "selected_trips": [],
                        "num_vehicles": 0, "tt_arcs_used": {p: [] for p in pats},
                        "vs_flows": {}}
    with open(args.solution_path, "w") as f:
        json.dump(state["sol"], f)


if __name__ == "__main__":
    main()
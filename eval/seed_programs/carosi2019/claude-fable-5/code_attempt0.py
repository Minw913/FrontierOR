import argparse
import json
import time
import bisect
import re
from collections import defaultdict, deque


def find_vehicle_cost(inst):
    """Locate the per-vehicle deployment cost anywhere in the instance JSON."""
    found = []

    def rec(o):
        if isinstance(o, dict):
            for k, v in o.items():
                kl = k.lower()
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    if 'cost' in kl and any(s in kl for s in
                                            ('vehicle', 'veh', 'deploy', 'fleet', 'fixed')):
                        found.append(float(v))
                rec(v)
        elif isinstance(o, list):
            for v in o:
                rec(v)

    rec(inst)
    if found:
        return found[0]
    desc = str(inst.get('objective_function', {}).get('description', ''))
    nums = [float(t) for t in re.findall(r'(\d+(?:\.\d+)?)', desc)]
    big = [n for n in nums if n >= 100]
    if big:
        return max(big)
    return 100000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--instance_path', required=True)
    ap.add_argument('--solution_path', required=True)
    ap.add_argument('--time_limit', type=int, default=600)
    ap.add_argument('--log_path', default=None)
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

    depot = inst['topology']['depot']
    tws = sorted(inst['time_windows'], key=lambda w: w['index'])
    wmin = {w['index']: w['min_headway_seconds'] for w in tws}
    wmax = {w['index']: w['max_headway_seconds'] for w in tws}
    wideal = {w['index']: w['ideal_headway_seconds'] for w in tws}

    smin = defaultdict(dict)
    smax = defaultdict(dict)
    for t, lst in inst['stopping_times'].items():
        for e in lst:
            smin[t][e['time_window_index']] = e['min_stopping_time_minutes']
            smax[t][e['time_window_index']] = e['max_stopping_time_minutes']
    pot = defaultdict(dict)
    pit = defaultdict(dict)
    for t, lst in inst['pull_in_out_times'].items():
        for e in lst:
            pot[t][e['time_window_index']] = e['pull_out_time_minutes']
            pit[t][e['time_window_index']] = e['pull_in_time_minutes']

    def _lk(d, t, w):
        dd = d.get(t)
        if not dd:
            return 0
        if w in dd:
            return dd[w]
        # nearest window fallback
        ks = sorted(dd.keys())
        best = min(ks, key=lambda k: abs(k - w))
        return dd[best]

    def POT(t, w):
        return _lk(pot, t, w)

    def PIT(t, w):
        return _lk(pit, t, w)

    def SMIN(t, w):
        dd = smin.get(t)
        if not dd:
            return None
        if w in dd:
            return dd[w]
        ks = sorted(dd.keys())
        return dd[min(ks, key=lambda k: abs(k - w))]

    def SMAX(t, w):
        dd = smax.get(t)
        if not dd:
            return None
        if w in dd:
            return dd[w]
        ks = sorted(dd.keys())
        return dd[min(ks, key=lambda k: abs(k - w))]

    def DMIN(w):
        v = SMIN(depot, w)
        return v if v is not None else 0

    objf = inst.get('objective_function', {})
    alpha = float(objf.get('alpha', 1.0))
    ds = (str(objf.get('description', '')) + str(objf.get('type', ''))).replace(' ', '').lower()
    beta = 1.0
    if '1-alpha' in ds or '(1-a)' in ds:
        beta = 1.0 - alpha
    Cveh = find_vehicle_cost(inst)

    trips = inst['potential_trips']
    ids = [t['id'] for t in trips]
    DEP = {t['id']: t['departure_time_minutes'] for t in trips}
    ARR = {t['id']: t['arrival_time_minutes'] for t in trips}
    MS = {t['id']: t['main_stop_arrival_time_minutes'] for t in trips}
    TW = {t['id']: t['time_window_index'] for t in trips}
    STT = {t['id']: t['start_terminal'] for t in trips}
    ETT = {t['id']: t['end_terminal'] for t in trips}
    PAT = {t['id']: t['pattern_id'] for t in trips}

    init_w = inst['initial_trip_time_windows']
    fin_w = inst['final_trip_time_windows']
    pats = sorted(set(PAT.values()))
    trips_by_pat = {p: sorted([i for i in ids if PAT[i] == p],
                              key=lambda i: (MS[i], DEP[i], i)) for p in pats}

    # ---------------- timetable arcs ----------------
    tt_arcs = {p: [] for p in pats}  # (a, b, cost)
    for p in pats:
        pl = trips_by_pat[p]
        for k, i in enumerate(pl):
            w = TW[i]
            lo = wmin.get(w, 0)
            hi = wmax.get(w, 10 ** 9)
            ideal = wideal.get(w, 0)
            for j in pl[k + 1:]:
                h = (MS[j] - MS[i]) * 60
                if h > hi:
                    break
                if h < lo:
                    continue
                tt_arcs[p].append((i, j, float((h - ideal) ** 2) * beta))
        iw = init_w.get(p)
        fw = fin_w.get(p)
        for i in pl:
            if iw is None or TW[i] == iw:
                tt_arcs[p].append(('src', i, 0.0))
            if fw is None or TW[i] == fw:
                tt_arcs[p].append((i, 'snk', 0.0))

    # ---------------- in-line compatibility arcs ----------------
    by_start = defaultdict(list)
    for i in ids:
        by_start[STT[i]].append(i)
    for tt_ in by_start:
        by_start[tt_].sort(key=lambda i: (DEP[i], i))
    dep_lists = {tt_: [DEP[i] for i in lst] for tt_, lst in by_start.items()}

    inline = []  # (i, j, extra_wait)
    for i in ids:
        Te = ETT[i]
        w = TW[i]
        if Te not in by_start:
            continue
        mn = SMIN(Te, w)
        mx = SMAX(Te, w)
        if mn is None or mx is None:
            continue
        lo = ARR[i] + mn
        hi = ARR[i] + mx
        lst = by_start[Te]
        dl = dep_lists[Te]
        a = bisect.bisect_left(dl, lo)
        b = bisect.bisect_right(dl, hi)
        for j in lst[a:b]:
            if j != i:
                inline.append((i, j, DEP[j] - ARR[i] - mn))

    # ---------------- depot timelines (out-line connections) ----------------
    terms_end = sorted(set(ETT.values()))
    terms_start = sorted(set(STT.values()))
    timelines = {}
    for Te in terms_end:
        for Ts in terms_start:
            if Te == Ts:
                continue
            ent = [(i, ARR[i] + PIT(Te, TW[i]) + DMIN(TW[i]))
                   for i in ids if ETT[i] == Te]
            ext = [(j, DEP[j] - POT(Ts, TW[j])) for j in ids if STT[j] == Ts]
            if not ent or not ext:
                continue
            times = sorted(set([r for _, r in ent] + [s for _, s in ext]))
            timelines[(Te, Ts)] = {'entries': ent, 'exits': ext, 'times': times,
                                   'tidx': {tm: k for k, tm in enumerate(times)}}

    # ---------------- build MIP ----------------
    import gurobipy as gp
    from gurobipy import GRB

    m = gp.Model('tt_vs')
    m.Params.OutputFlag = 0
    m.Params.Seed = 0
    m.Params.MIPGap = 1e-4
    m.Params.NumericFocus = 0
    m.Params.Threads = 1

    x = {i: m.addVar(vtype=GRB.BINARY, name='x_%d' % i) for i in ids}
    yv = {}
    for p in pats:
        for (a, b, c) in tt_arcs[p]:
            yv[(p, a, b)] = m.addVar(vtype=GRB.BINARY)
    po = {i: m.addVar(vtype=GRB.BINARY) for i in ids}
    piv = {i: m.addVar(vtype=GRB.BINARY) for i in ids}
    gv = {}
    for (i, j, ex) in inline:
        if (i, j) not in gv:
            gv[(i, j)] = m.addVar(vtype=GRB.BINARY)
    entv = {}
    extv = {}
    waitv = {}
    for key, tl in timelines.items():
        for (i, r) in tl['entries']:
            entv[(key, i)] = m.addVar(vtype=GRB.BINARY)
        for (j, s) in tl['exits']:
            extv[(key, j)] = m.addVar(vtype=GRB.BINARY)
        nt = len(tl['times'])
        waitv[key] = [m.addVar(vtype=GRB.INTEGER, lb=0, ub=len(tl['entries']))
                      for _ in range(nt - 1)]
    z = m.addVar(vtype=GRB.INTEGER, lb=0, ub=len(ids), name='z')
    m.update()

    # timetable flow constraints
    for p in pats:
        inn = defaultdict(list)
        out = defaultdict(list)
        for (a, b, c) in tt_arcs[p]:
            v = yv[(p, a, b)]
            out[a].append(v)
            inn[b].append(v)
        m.addConstr(gp.quicksum(out['src']) == 1)
        m.addConstr(gp.quicksum(inn['snk']) == 1)
        for i in trips_by_pat[p]:
            m.addConstr(gp.quicksum(inn[i]) == x[i])
            m.addConstr(gp.quicksum(out[i]) == x[i])

    # vehicle-scheduling flow constraints
    sin = defaultdict(list)
    sout = defaultdict(list)
    for i in ids:
        sin[i].append(po[i])
        sout[i].append(piv[i])
    seen_pairs = set()
    for (i, j, ex) in inline:
        if (i, j) in seen_pairs:
            continue
        seen_pairs.add((i, j))
        v = gv[(i, j)]
        sout[i].append(v)
        sin[j].append(v)
    for key, tl in timelines.items():
        for (i, r) in tl['entries']:
            sout[i].append(entv[(key, i)])
        for (j, s) in tl['exits']:
            sin[j].append(extv[(key, j)])
    for i in ids:
        m.addConstr(gp.quicksum(sin[i]) == x[i])
        m.addConstr(gp.quicksum(sout[i]) == x[i])
    m.addConstr(gp.quicksum(po.values()) == z)
    m.addConstr(gp.quicksum(piv.values()) == z)
    for key, tl in timelines.items():
        ent_at = defaultdict(list)
        ext_at = defaultdict(list)
        for (i, r) in tl['entries']:
            ent_at[r].append(entv[(key, i)])
        for (j, s) in tl['exits']:
            ext_at[s].append(extv[(key, j)])
        wv = waitv[key]
        times = tl['times']
        for k, tm in enumerate(times):
            infl = list(ent_at.get(tm, []))
            outf = list(ext_at.get(tm, []))
            if k > 0:
                infl.append(wv[k - 1])
            if k < len(times) - 1:
                m.addConstr(gp.quicksum(infl) == gp.quicksum(outf) + wv[k])
            else:
                m.addConstr(gp.quicksum(infl) == gp.quicksum(outf))

    # objective
    obj = gp.LinExpr()
    for p in pats:
        for (a, b, c) in tt_arcs[p]:
            if c:
                obj.add(yv[(p, a, b)], c)
    obj.add(z, alpha * Cveh)
    added_inline = set()
    for (i, j, ex) in inline:
        if (i, j) in added_inline:
            continue
        added_inline.add((i, j))
        if ex:
            obj.add(gv[(i, j)], alpha * ex)
    for i in ids:
        obj.add(po[i], alpha * POT(STT[i], TW[i]))
        obj.add(piv[i], alpha * PIT(ETT[i], TW[i]))
    for key, tl in timelines.items():
        Te, Ts = key
        for (i, r) in tl['entries']:
            obj.add(entv[(key, i)], alpha * PIT(Te, TW[i]))
        for (j, s) in tl['exits']:
            obj.add(extv[(key, j)], alpha * POT(Ts, TW[j]))
    m.setObjective(obj, GRB.MINIMIZE)
    m.update()

    inline_pairs = sorted(gv.keys())
    inline_extra = {}
    for (i, j, ex) in inline:
        inline_extra[(i, j)] = ex

    # ---------------- solution builder ----------------
    def build_solution(getv):
        sel = [i for i in ids if getv(x[i]) > 0.5]
        tt_used = {}
        ttc = 0.0
        for p in pats:
            succ = {}
            for (a, b, c) in tt_arcs[p]:
                if getv(yv[(p, a, b)]) > 0.5:
                    succ[a] = b
                    ttc += c
            arcs = []
            cur = 'src'
            guard = 0
            while cur in succ and guard <= len(succ) + 2:
                nxt = succ[cur]
                fa = 'source' if cur == 'src' else str(cur)
                fb = 'sink' if nxt == 'snk' else str(nxt)
                arcs.append([fa, fb])
                cur = nxt
                guard += 1
            tt_used[p] = arcs
        vs = {}
        zval = int(round(getv(z)))
        vs['VS_O_plus-->VS_O_minus'] = zval
        vcost = Cveh * zval
        for i in sel:
            vs['%s-->%s' % (str((i, 'start')), str((i, 'end')))] = 1
        for i in ids:
            if getv(po[i]) > 0.5:
                vs['VS_O_minus-->%s' % str((i, 'start'))] = 1
                vcost += POT(STT[i], TW[i])
            if getv(piv[i]) > 0.5:
                vs['%s-->VS_O_plus' % str((i, 'end'))] = 1
                vcost += PIT(ETT[i], TW[i])
        for (i, j) in inline_pairs:
            if getv(gv[(i, j)]) > 0.5:
                vs['%s-->%s' % (str((i, 'end')), str((j, 'start')))] = 1
                vcost += inline_extra[(i, j)]
        for key, tl in timelines.items():
            Te, Ts = key
            ent = sorted([(r, i) for (i, r) in tl['entries']
                          if getv(entv[(key, i)]) > 0.5])
            ext = sorted([(s, j) for (j, s) in tl['exits']
                          if getv(extv[(key, j)]) > 0.5])
            k = 0
            avail = deque()
            for (s, j) in ext:
                while k < len(ent) and ent[k][0] <= s:
                    avail.append(ent[k][1])
                    k += 1
                if avail:
                    i = avail.popleft()
                    vs['%s-->%s' % (str((i, 'end')), str((j, 'start')))] = 1
                    vcost += PIT(Te, TW[i]) + POT(Ts, TW[j])
        objv = ttc + alpha * vcost
        sol = {
            'objective_value': float(objv),
            'selected_trips': sorted(sel),
            'num_vehicles': zval,
            'tt_arcs_used': tt_used,
            'vs_flows': vs,
        }
        return objv, sol

    # ---------------- heuristic (warm start / fallback) ----------------
    def heuristic():
        sv = {}

        def add(v, val=1.0):
            sv[v.index] = sv.get(v.index, 0.0) + val

        allsel = []
        for p in pats:
            adj = defaultdict(list)
            snk_from = set()
            for (a, b, c) in tt_arcs[p]:
                if b == 'snk':
                    snk_from.add(a)
                else:
                    adj[a].append((b, c))
            INF = float('inf')
            dist = {}
            par = {}
            for (b, c) in adj['src']:
                if dist.get(b, INF) > 0:
                    dist[b] = 0.0
                    par[b] = 'src'
            for i in trips_by_pat[p]:
                if i not in dist:
                    continue
                di = dist[i]
                for (b, c) in adj[i]:
                    nd = di + c
                    if nd < dist.get(b, INF):
                        dist[b] = nd
                        par[b] = i
            bi = None
            bd = INF
            for a in snk_from:
                if a in dist and dist[a] < bd:
                    bd = dist[a]
                    bi = a
            if bi is None:
                return None
            chain = []
            cur = bi
            while cur != 'src':
                chain.append(cur)
                cur = par[cur]
            chain.reverse()
            prev = 'src'
            for c in chain:
                add(yv[(p, prev, c)])
                prev = c
            add(yv[(p, prev, 'snk')])
            for c in chain:
                add(x[c])
            allsel += chain

        allsel.sort(key=lambda i: (DEP[i], i))
        last = []
        first = []
        conns = []
        for j in allsel:
            bestc = None
            bidx = None
            btype = None
            for vi, i in enumerate(last):
                gap = DEP[j] - ARR[i]
                if ETT[i] == STT[j]:
                    w = TW[i]
                    mn = SMIN(ETT[i], w)
                    mx = SMAX(ETT[i], w)
                    if mn is not None and mx is not None and mn <= gap <= mx:
                        if (i, j) in gv:
                            c = gap - mn
                            if bestc is None or c < bestc:
                                bestc, bidx, btype = c, vi, 'in'
                else:
                    key = (ETT[i], STT[j])
                    if key in timelines:
                        thr = PIT(ETT[i], TW[i]) + DMIN(TW[i]) + POT(STT[j], TW[j])
                        if gap >= thr:
                            c = PIT(ETT[i], TW[i]) + POT(STT[j], TW[j])
                            if bestc is None or c < bestc:
                                bestc, bidx, btype = c, vi, 'out'
            if bidx is None:
                last.append(j)
                first.append(j)
            else:
                conns.append((last[bidx], j, btype))
                last[bidx] = j
        for j in first:
            add(po[j])
        for i in last:
            add(piv[i])
        for (i, j, tp) in conns:
            if tp == 'in':
                add(gv[(i, j)])
            else:
                key = (ETT[i], STT[j])
                tl = timelines[key]
                add(entv[(key, i)])
                add(extv[(key, j)])
                r = ARR[i] + PIT(ETT[i], TW[i]) + DMIN(TW[i])
                s = DEP[j] - POT(STT[j], TW[j])
                ka = tl['tidx'][r]
                kb = tl['tidx'][s]
                for k in range(ka, kb):
                    add(waitv[key][k])
        sv[z.index] = float(len(last))
        return sv

    best = [float('inf')]
    best_sol = [None]

    sv = None
    try:
        sv = heuristic()
    except Exception:
        sv = None
    if sv is not None:
        try:
            getv = lambda v: sv.get(v.index, 0.0)
            hobj, hsol = build_solution(getv)
            best[0] = hobj
            best_sol[0] = hsol
            if logger:
                logger.log_solution(hobj, hsol)
            allv = m.getVars()
            m.setAttr('Start', allv, [0.0] * len(allv))
            for idx, val in sv.items():
                allv[idx].Start = val
        except Exception:
            pass

    m._allvars = m.getVars()

    def cb(model, where):
        if where == GRB.Callback.MIPSOL:
            try:
                ov = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if ov < best[0] - 1e-6:
                    vals = model.cbGetSolution(model._allvars)
                    getv = lambda v: vals[v.index]
                    o2, sol = build_solution(getv)
                    best[0] = o2
                    best_sol[0] = sol
                    if logger:
                        logger.log_solution(o2, sol)
            except Exception:
                pass

    rem = args.time_limit - (time.time() - t0) - 3
    if rem > 1:
        m.Params.TimeLimit = max(1.0, rem)
        try:
            m.optimize(cb)
        except Exception:
            pass
        try:
            if m.SolCount > 0:
                getv = lambda v: v.X
                o2, sol = build_solution(getv)
                if o2 < best[0] - 1e-9 or best_sol[0] is None:
                    if logger and o2 < best[0] - 1e-9:
                        logger.log_solution(o2, sol)
                    best[0] = o2
                    best_sol[0] = sol
        except Exception:
            pass

    if best_sol[0] is None:
        # last resort: empty structure (should not normally happen)
        best_sol[0] = {
            'objective_value': float('inf'),
            'selected_trips': [],
            'num_vehicles': 0,
            'tt_arcs_used': {p: [] for p in pats},
            'vs_flows': {'VS_O_plus-->VS_O_minus': 0},
        }

    with open(args.solution_path, 'w') as f:
        json.dump(best_sol[0], f)


if __name__ == '__main__':
    main()
import argparse
import json
import sys
import time
from collections import deque

INF = 10 ** 9


def hopcroft_karp(adj, nL, nR):
    """Maximum bipartite matching; adj[u] = list of right nodes."""
    matchL = [-1] * nL
    matchR = [-1] * nR
    dist = [0] * nL
    ptr = [0] * nL
    sys.setrecursionlimit(max(1000000, 8 * nL + 1000))
    dq = deque()

    def bfs():
        dq.clear()
        for u in range(nL):
            if matchL[u] == -1:
                dist[u] = 0
                dq.append(u)
            else:
                dist[u] = -1
        while dq:
            u = dq.popleft()
            du = dist[u] + 1
            for v in adj[u]:
                w = matchR[v]
                if w != -1 and dist[w] == -1:
                    dist[w] = du
                    dq.append(w)

    def dfs(u):
        au = adj[u]
        pu = ptr[u]
        la = len(au)
        while pu < la:
            v = au[pu]
            pu += 1
            w = matchR[v]
            if w == -1 or (dist[w] == dist[u] + 1 and dfs(w)):
                matchL[u] = v
                matchR[v] = u
                ptr[u] = pu
                return True
        ptr[u] = pu
        dist[u] = -1
        return False

    matching = 0
    while True:
        bfs()
        for u in range(nL):
            ptr[u] = 0
        found = False
        for u in range(nL):
            if matchL[u] == -1 and dfs(u):
                matching += 1
                found = True
        if not found:
            break
    return matchL, matchR, matching


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--instance_path', required=True)
    ap.add_argument('--solution_path', required=True)
    ap.add_argument('--time_limit', type=int, default=300)
    ap.add_argument('--log_path', default=None)
    args = ap.parse_args()
    t0 = time.time()
    TL = max(3, args.time_limit)

    logger = None
    if args.log_path:
        try:
            from solution_logger import SolutionLogger
            logger = SolutionLogger(args.log_path, sense="minimize")
        except Exception:
            logger = None

    with open(args.instance_path) as f:
        inst = json.load(f)

    # ---------------- data preparation ----------------
    trips = inst.get('trips', [])
    N = len(trips)
    locations = inst['locations']
    loc_names = list(locations.keys())
    loc_idx = {n: i for i, n in enumerate(loc_names)}
    L = len(loc_names)
    tt = [[0] * L for _ in range(L)]
    for o, dd in inst.get('travel_times', {}).items():
        oi = loc_idx.get(o)
        if oi is None:
            continue
        row = tt[oi]
        for dn2, v in dd.items():
            dj = loc_idx.get(dn2)
            if dj is not None:
                row[dj] = v

    depots = inst['depots']
    depot_loc = [loc_idx[d['name']] for d in depots]
    depot_names = [d['name'] for d in depots]
    nd = len(depots)

    order_trips = sorted(range(N), key=lambda i: (trips[i]['start_time'],
                                                  trips[i]['end_time'],
                                                  trips[i].get('trip_id', i)))
    st = [trips[i]['start_time'] for i in order_trips]
    et = [trips[i]['end_time'] for i in order_trips]
    sl = [loc_idx[trips[i]['start_location']] for i in order_trips]
    el = [loc_idx[trips[i]['end_location']] for i in order_trips]
    tid = [trips[i].get('trip_id', i) for i in order_trips]

    relief_idx = set()
    for r in inst.get('relief_locations', []) or []:
        if r in loc_idx:
            relief_idx.add(loc_idx[r])

    duty_types = inst['duty_types']
    cp = inst['crew_parameters']
    sign_on = cp['sign_on_time_depot_minutes']
    sign_off = cp['sign_off_time_depot_minutes']
    extra = cp['extra_time_non_depot_relief_minutes']
    extra_incl = bool(cp.get('extra_time_includes_deadhead_to_depot', False))

    def tval(T, k, dflt):
        v = T.get(k, dflt)
        return dflt if v is None else v

    all_types = list(duty_types.items())
    types2 = [(n, T) for n, T in all_types if T.get('num_pieces', 2) == 2]
    types1 = [(n, T) for n, T in all_types if T.get('num_pieces', 2) == 1]
    if not types2 and not types1:
        types2 = all_types
    tripper_name = types1[0][0] if types1 else (all_types[0][0] if all_types else 'tripper')
    tripper = types1[0][1] if types1 else None

    src2 = [T for _, T in types2] if types2 else [T for _, T in all_types]
    PL_min_t = max(tval(T, 'piece_length_min', 0) for _, T in all_types)
    PL_max_t = min(tval(T, 'piece_length_max', INF) for _, T in all_types)
    PL_min_w = min(tval(T, 'piece_length_min', 0) for T in src2)
    PL_max_w = max(tval(T, 'piece_length_max', INF) for T in src2)
    if PL_min_t > PL_max_t:
        PL_min_t, PL_max_t = PL_min_w, PL_max_w
    if PL_min_t > PL_max_t:
        PL_min_t, PL_max_t = 0, INF

    # ---------------- vehicle scheduling: min path cover ----------------
    succ_hk = [-1] * N
    K = 0
    edges_all = []
    if N > 0:
        cap = max(20, min(120, 300000 // max(1, N)))
        tt_min_from = [0] * L
        for r in range(L):
            bm = INF
            row = tt[r]
            for c in range(L):
                if c != r and row[c] < bm:
                    bm = row[c]
            tt_min_from[r] = 0 if bm >= INF else bm
        adj = [[] for _ in range(N)]
        for i in range(N):
            ei = et[i]
            row = tt[el[i]]
            tmin_i = tt_min_from[el[i]]
            ai = []
            cnt = 0
            for j in range(i + 1, N):
                sj = st[j]
                if sj < ei + tmin_i:
                    continue
                if ei + row[sl[j]] <= sj:
                    ai.append(j)
                    cnt += 1
                    if cnt >= cap:
                        break
            adj[i] = ai
        matchL, matchR, K = hopcroft_karp(adj, N, N)
        succ_hk = matchL[:]
        edges_all = [(i, j) for i in range(N) for j in adj[i]]

    # min-cost improvement of the max matching (idle / long-arc preference)
    succ = succ_hk
    remaining = TL - (time.time() - t0)
    if N > 0 and edges_all and len(edges_all) <= 500000 and remaining > 20:
        try:
            import gurobipy as gp
            from gurobipy import GRB
            m = gp.Model('veh')
            m.Params.Seed = 0
            m.Params.MIPGap = 1e-4
            m.Params.NumericFocus = 0
            m.Params.Threads = 1
            m.Params.OutputFlag = 0
            m.Params.TimeLimit = max(5.0, min(90.0, remaining * 0.25))
            costs = []
            for (i, j) in edges_all:
                gap = st[j] - et[i]
                via = min(tt[el[i]][depot_loc[d]] + tt[depot_loc[d]][sl[j]]
                          for d in range(nd))
                c = gap - tt[el[i]][sl[j]]
                if gap >= via:
                    c = 0
                if c < 0:
                    c = 0
                costs.append(c)
            x = m.addVars(edges_all, ub=1.0, obj=costs)
            m.ModelSense = GRB.MINIMIZE
            m.addConstrs((x.sum(i, '*') <= 1 for i in range(N)), name='out')
            m.addConstrs((x.sum('*', j) <= 1 for j in range(N)), name='in')
            m.addConstr(x.sum() == K, name='card')
            m.optimize()
            if m.Status == GRB.OPTIMAL:
                succ2 = [-1] * N
                for (i, j) in edges_all:
                    if x[(i, j)].X > 0.5:
                        succ2[i] = j
                if sum(1 for s in succ2 if s >= 0) == K:
                    succ = succ2
        except Exception:
            pass

    def build_chains(succv):
        pred = [-1] * N
        for i, s in enumerate(succv):
            if s >= 0:
                pred[s] = i
        chains = []
        for i in range(N):
            if pred[i] == -1:
                ch = []
                u = i
                while u != -1:
                    ch.append(u)
                    u = succv[u]
                chains.append(ch)
        return chains

    chains_main = build_chains(succ)

    # ---------------- crew piece generation ----------------
    def gen_pieces(chains, dl, variant, PL_min, PL_max, allow_short, min_pure):
        pieces = []
        for ci, chain in enumerate(chains):
            k = len(chain)
            po = [tt[dl][sl[t]] for t in chain]
            pin = [tt[el[t]][dl] for t in chain]
            longarc = [False] * max(0, k - 1)
            for i in range(k - 1):
                gap = st[chain[i + 1]] - et[chain[i]]
                longarc[i] = (gap >= pin[i] + po[i + 1])
            bnd = [False] * max(0, k - 1)
            for b in range(k - 1):
                if longarc[b]:
                    bnd[b] = True
                elif allow_short:
                    locb = sl[chain[b + 1]] if variant == 0 else el[chain[b]]
                    bnd[b] = (locb in relief_idx) or (locb == dl)
            ps_arr = [0.0] * k
            pe_arr = [0.0] * k
            for a in range(k):
                if a == 0 or longarc[a - 1]:
                    ps_arr[a] = st[chain[a]] - po[a]
                else:
                    ps_arr[a] = st[chain[a]] if variant == 0 else et[chain[a - 1]]
            for b in range(k):
                if b == k - 1 or longarc[b]:
                    pe_arr[b] = et[chain[b]] + pin[b]
                else:
                    pe_arr[b] = st[chain[b + 1]] if variant == 0 else et[chain[b]]
            dp = [(INF, INF, -1)] * (k + 1)
            dp[0] = (0, 0, -1)
            for j in range(1, k + 1):
                b = j - 1
                peb = pe_arr[b]
                etb = et[chain[b]]
                best = (INF, INF, -1)
                for a in range(b, -1, -1):
                    if dp[a][0] >= INF:
                        continue
                    if a > 0 and not bnd[a - 1]:
                        continue
                    dur_full = peb - ps_arr[a]
                    if dur_full > PL_max:
                        continue
                    if min_pure:
                        if etb - st[chain[a]] < PL_min:
                            continue
                    else:
                        if dur_full < PL_min:
                            continue
                    pen = 0 if (a == 0 or longarc[a - 1]) else 1
                    c0 = dp[a][0] + 1
                    c1 = dp[a][1] + pen
                    if (c0, c1) < (best[0], best[1]):
                        best = (c0, c1, a)
                dp[j] = best
            if dp[k][0] >= INF:
                return None
            segs = []
            j = k
            while j > 0:
                a = dp[j][2]
                if a < 0:
                    return None
                segs.append((a, j - 1))
                j = a
            segs.reverse()
            for (a, b) in segs:
                start_depot = (a == 0 or longarc[a - 1])
                end_depot = (b == k - 1 or longarc[b])
                if start_depot:
                    sloc = dl
                else:
                    sloc = sl[chain[a]] if variant == 0 else el[chain[a - 1]]
                if end_depot:
                    eloc = dl
                else:
                    eloc = sl[chain[b + 1]] if variant == 0 else el[chain[b]]
                pieces.append({
                    'chain': ci, 'a': a, 'b': b,
                    'trips': chain[a:b + 1],
                    'start': ps_arr[a], 'end': pe_arr[b],
                    'dur': pe_arr[b] - ps_arr[a],
                    'dur_pure': et[chain[b]] - st[chain[a]],
                    'start_depot': start_depot, 'end_depot': end_depot,
                    'sloc': sloc, 'eloc': eloc,
                })
        return pieces

    def gen_pieces_robust(chains, dl, variant, bounds, allow_short, min_pure):
        cfgs = [
            (bounds, allow_short, min_pure),
            (bounds, True, min_pure),
            (bounds, True, False),
            ((PL_min_w, PL_max_w), True, False),
            ((0, INF), True, False),
        ]
        seen = set()
        for cfg in cfgs:
            key = (cfg[0], cfg[1], cfg[2])
            if key in seen:
                continue
            seen.add(key)
            pieces = gen_pieces(chains, dl, variant, cfg[0][0], cfg[0][1],
                                cfg[1], cfg[2])
            if pieces is not None:
                return pieces
        pieces = []
        for ci, chain in enumerate(chains):
            ps = st[chain[0]] - tt[dl][sl[chain[0]]]
            pe = et[chain[-1]] + tt[el[chain[-1]]][dl]
            pieces.append({'chain': ci, 'a': 0, 'b': len(chain) - 1,
                           'trips': chain[:], 'start': ps, 'end': pe,
                           'dur': pe - ps,
                           'dur_pure': et[chain[-1]] - st[chain[0]],
                           'start_depot': True, 'end_depot': True,
                           'sloc': dl, 'eloc': dl})
        return pieces

    # ---------------- duty feasibility checks ----------------
    def oh_start(p, dl):
        if p['start_depot'] or p['sloc'] == dl:
            return sign_on
        v = extra
        if not extra_incl:
            v += tt[dl][p['sloc']]
        return v

    def oh_end(p, dl):
        if p['end_depot'] or p['eloc'] == dl:
            return sign_off
        v = extra
        if not extra_incl:
            v += tt[p['eloc']][dl]
        return v

    def piece_ok(T, p):
        pmn = tval(T, 'piece_length_min', 0)
        pmx = tval(T, 'piece_length_max', None)
        if p['dur_pure'] < pmn:
            return False
        if pmx is not None and p['dur'] > pmx:
            return False
        return True

    def feasible_type(p, q, ohs_p, ohe_q):
        for name, T in types2:
            if not piece_ok(T, p) or not piece_ok(T, q):
                continue
            brk = q['start'] - p['end']
            bm = T.get('break_length_min')
            if bm is not None and brk < bm:
                continue
            ds = p['start'] - ohs_p
            de = q['end'] + ohe_q
            dm = T.get('duty_length_max')
            if dm is not None and de - ds > dm:
                continue
            wm = T.get('work_time_max')
            if wm is not None and p['dur'] + q['dur'] + ohs_p + ohe_q > wm:
                continue
            smin = T.get('start_time_min')
            if smin is not None and ds < smin:
                continue
            smax = T.get('start_time_max')
            if smax is not None and p['start'] > smax:
                continue
            emax = T.get('end_time_max')
            if emax is not None and de > emax:
                continue
            return name
        return None

    def tripper_feasible(p, ohs, ohe):
        if tripper is None:
            return False
        T = tripper
        if not piece_ok(T, p):
            return False
        ds = p['start'] - ohs
        de = p['end'] + ohe
        dm = T.get('duty_length_max')
        if dm is not None and de - ds > dm:
            return False
        wm = T.get('work_time_max')
        if wm is not None and p['dur'] + ohs + ohe > wm:
            return False
        smin = T.get('start_time_min')
        if smin is not None and ds < smin:
            return False
        smax = T.get('start_time_max')
        if smax is not None and p['start'] > smax:
            return False
        emax = T.get('end_time_max')
        if emax is not None and de > emax:
            return False
        return True

    # ---------------- pairing of pieces into duties ----------------
    def make_duties(pieces, dl, mip_budget):
        P = len(pieces)
        if P == 0:
            return []
        ohs = [oh_start(p, dl) for p in pieces]
        ohe = [oh_end(p, dl) for p in pieces]
        order = sorted(range(P), key=lambda i: (pieces[i]['start'],
                                                pieces[i]['end'], i))
        max_span = 0
        for _, T in types2:
            dm = T.get('duty_length_max')
            if dm is None:
                max_span = INF
            else:
                max_span = max(max_span, dm)
        edges = []
        if types2:
            EDGE_CAP = 120
            for ii in range(P):
                i = order[ii]
                pi = pieces[i]
                cnt = 0
                for jj in range(ii + 1, P):
                    j = order[jj]
                    pj = pieces[j]
                    if max_span < INF and pj['start'] - pi['start'] + ohs[i] > max_span:
                        break
                    if pj['start'] < pi['end']:
                        continue
                    if pi['chain'] == pj['chain'] and pi['b'] + 1 == pj['a']:
                        continue
                    tn = feasible_type(pi, pj, ohs[i], ohe[j])
                    if tn is not None:
                        edges.append((i, j, tn))
                        cnt += 1
                        if cnt >= EDGE_CAP:
                            break
        must_pair = [i for i, p in enumerate(pieces)
                     if not tripper_feasible(p, ohs[i], ohe[i])]
        # greedy matching (guaranteed solution)
        adjd = {}
        for e, (i, j, tn) in enumerate(edges):
            adjd.setdefault(i, []).append((j, e))
            adjd.setdefault(j, []).append((i, e))
        matched = {}
        for v in order:
            if v in matched:
                continue
            for (w, e) in adjd.get(v, ()):
                if w not in matched:
                    matched[v] = w
                    matched[w] = v
                    break
        chosen = set()
        seen = set()
        for v, w in matched.items():
            if v in seen or w in seen:
                continue
            seen.add(v)
            seen.add(w)
            for (ww, e) in adjd[v]:
                if ww == w:
                    chosen.add(e)
                    break
        # MIP maximum matching improvement
        if edges and mip_budget > 1.0:
            try:
                import gurobipy as gp
                from gurobipy import GRB
                m = gp.Model('crew_pairing')
                m.Params.Seed = 0
                m.Params.MIPGap = 1e-4
                m.Params.NumericFocus = 0
                m.Params.Threads = 1
                m.Params.OutputFlag = 0
                m.Params.TimeLimit = max(1.0, mip_budget)
                E = len(edges)
                x = m.addVars(range(E), vtype=GRB.BINARY)
                u = m.addVars(range(len(must_pair)), vtype=GRB.BINARY)
                mind = {v: k for k, v in enumerate(must_pair)}
                inc = [[] for _ in range(P)]
                for e, (i, j, _) in enumerate(edges):
                    inc[i].append(e)
                    inc[j].append(e)
                for v in range(P):
                    if not inc[v] and v not in mind:
                        continue
                    expr = gp.quicksum(x[e] for e in inc[v])
                    if v in mind:
                        m.addConstr(expr + u[mind[v]] == 1)
                    else:
                        m.addConstr(expr <= 1)
                m.setObjective(x.sum() - 1000.0 * u.sum(), GRB.MAXIMIZE)
                m.optimize()
                if m.SolCount > 0:
                    ce = set(e for e in range(E) if x[e].X > 0.5)

                    def val(es):
                        used = [False] * P
                        for e in es:
                            i, j, _ = edges[e]
                            used[i] = True
                            used[j] = True
                        pen = sum(1 for v in mind if not used[v])
                        return len(es) - 1000 * pen

                    if val(ce) > val(chosen):
                        chosen = ce
            except Exception:
                pass
        used = [False] * P
        duties = []
        for e in sorted(chosen):
            i, j, tn = edges[e]
            used[i] = True
            used[j] = True
            duties.append((tn, pieces[i]['trips'] + pieces[j]['trips']))
        for i in range(P):
            if not used[i]:
                duties.append((tripper_name, pieces[i]['trips']))
        return duties

    # ---------------- solution building / validation ----------------
    def build_solution(chains, depot_id, duties):
        veh = {depot_names[d]: [] for d in range(nd)}
        veh[depot_names[depot_id]] = [[tid[t] for t in ch] for ch in chains]
        crew = [{'depot': depot_names[depot_id], 'type': tn,
                 'trips': [tid[t] for t in tr]} for (tn, tr) in duties]
        obj = len(chains) + len(duties)
        return {'objective_value': obj, 'vehicle_schedule': veh,
                'crew_schedule': crew}

    def validate(sol):
        try:
            allids = set(tid)
            seen = set()
            nv = 0
            for dn, chs in sol['vehicle_schedule'].items():
                for ch in chs:
                    nv += 1
                    for t in ch:
                        if t in seen:
                            return False
                        seen.add(t)
            if seen != allids:
                return False
            cs = set()
            for d in sol['crew_schedule']:
                for t in d['trips']:
                    if t in cs:
                        return False
                    cs.add(t)
            if cs != allids:
                return False
            if sol['objective_value'] != nv + len(sol['crew_schedule']):
                return False
            return True
        except Exception:
            return False

    def trivial_solution():
        d0 = depot_names[0]
        veh = {dn: [] for dn in depot_names}
        veh[d0] = [[tid[i]] for i in range(N)]
        crew = [{'depot': d0, 'type': tripper_name, 'trips': [tid[i]]}
                for i in range(N)]
        return {'objective_value': 2 * N, 'vehicle_schedule': veh,
                'crew_schedule': crew}

    # ---------------- candidate search loop ----------------
    veh_variants = [chains_main]
    if succ != succ_hk:
        veh_variants.append(build_chains(succ_hk))

    dep_cost = []
    for d in range(nd):
        dl = depot_loc[d]
        c = 0
        for ch in chains_main:
            c += tt[dl][sl[ch[0]]] + tt[el[ch[-1]]][dl]
        dep_cost.append((c, d))
    dep_cost.sort()
    depot_order = [d for _, d in dep_cost]

    tight = (PL_min_t, PL_max_t)
    wide = (PL_min_w, PL_max_w)
    d0 = depot_order[0]
    configs = [
        (0, d0, 0, tight, False, True),
        (0, d0, 0, tight, True, True),
        (0, d0, 0, tight, True, False),
        (0, d0, 0, wide, True, False),
        (0, d0, 1, tight, True, True),
    ]
    for d in depot_order[1:]:
        configs.append((0, d, 0, tight, True, False))
    if len(veh_variants) > 1:
        configs.append((1, d0, 0, tight, True, False))

    best_sol = None
    best_obj = INF
    total = len(configs)
    for ci_, (vi, d, variant, bounds, allow_short, min_pure) in enumerate(configs):
        remaining = TL - (time.time() - t0) - 2.0
        if remaining < 4 and best_sol is not None:
            break
        slice_t = remaining / max(1, (total - ci_))
        slice_t = max(2.0, min(slice_t, 60.0))
        try:
            dl = depot_loc[d]
            pieces = gen_pieces_robust(veh_variants[vi], dl, variant, bounds,
                                       allow_short, min_pure)
            duties = make_duties(pieces, dl, slice_t)
            sol = build_solution(veh_variants[vi], d, duties)
            if validate(sol) and sol['objective_value'] < best_obj:
                best_obj = sol['objective_value']
                best_sol = sol
                if logger:
                    logger.log_solution(best_obj, best_sol)
        except Exception:
            continue

    if best_sol is None:
        best_sol = trivial_solution()
        if logger:
            logger.log_solution(best_sol['objective_value'], best_sol)

    with open(args.solution_path, 'w') as f:
        json.dump(best_sol, f)


if __name__ == '__main__':
    main()
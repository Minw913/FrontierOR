import argparse
import json
import time
import random

EPS = 1e-6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--instance_path', required=True)
    ap.add_argument('--solution_path', required=True)
    ap.add_argument('--time_limit', type=int, default=60)
    ap.add_argument('--log_path', default=None)
    args = ap.parse_args()

    start = time.time()
    tl = max(3, args.time_limit)
    hard_dl = start + tl - max(1.0, 0.04 * tl)

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    rng = random.Random(0)

    with open(args.instance_path) as f:
        inst = json.load(f)

    n = int(inst['num_customers'])
    D = inst['distance_matrix']
    cap = float(inst['vehicle_capacity'])
    dep_lo = float(inst['depot']['exogenous_time_window'][0])
    dep_hi = float(inst['depot']['exogenous_time_window'][1])
    END = n + 1

    lo = [dep_lo] * (n + 2)
    hi = [dep_hi] * (n + 2)
    wid = [0.0] * (n + 2)
    gw = float(inst.get('endogenous_time_window_width', 0.0) or 0.0)
    ids = []
    for c in inst['customers']:
        i = int(c['id'])
        ids.append(i)
        lo[i] = float(c['exogenous_time_window'][0])
        hi[i] = float(c['exogenous_time_window'][1])
        wid[i] = float(c.get('endogenous_time_window_width', gw))
    ids.sort()

    scen = inst['scenarios']
    dem_s = {}
    probs = {}
    sids = []
    for s in scen:
        sid = s['scenario_id']
        sids.append(sid)
        v = [0.0] * (n + 2)
        for k, i in enumerate(ids):
            v[i] = float(s['demands'][k])
        dem_s[sid] = v
        probs[sid] = float(s['probability'])

    dmean = [0.0] * (n + 2)
    dmax = [0.0] * (n + 2)
    tp = sum(probs.values()) or 1.0
    for sid in sids:
        v = dem_s[sid]
        p = probs[sid] / tp
        for i in ids:
            dmean[i] += p * v[i]
            if v[i] > dmax[i]:
                dmax[i] = v[i]

    # ---------- core helpers ----------
    def rcost(r):
        prev = 0
        s = 0.0
        for x in r:
            s += D[prev][x]
            prev = x
        return s + D[prev][END]

    def tot_cost(routes):
        return sum(rcost(r) for r in routes if r)

    def sched(r, L, H):
        t = dep_lo
        prev = 0
        ts = []
        for c in r:
            t += D[prev][c]
            lc = L[c]
            if t < lc:
                t = lc
            if t > H[c] + EPS:
                return None
            ts.append(t)
            prev = c
        if t + D[prev][END] > dep_hi + EPS:
            return None
        return ts

    def insert_best(part, loads, c, dem, L, H):
        bd = None
        bri = -1
        bpos = -1
        dc = dem[c]
        for ri, r in enumerate(part):
            if loads[ri] + dc > cap + 1e-9:
                continue
            for pos in range(len(r) + 1):
                prev = r[pos - 1] if pos > 0 else 0
                nxt = r[pos] if pos < len(r) else END
                dta = D[prev][c] + D[c][nxt] - D[prev][nxt]
                if bd is None or dta < bd - 1e-12:
                    cand = r[:pos] + [c] + r[pos:]
                    if sched(cand, L, H) is not None:
                        bd = dta
                        bri = ri
                        bpos = pos
        nd = D[0][c] + D[c][END]
        if bd is not None and bd <= nd:
            part[bri].insert(bpos, c)
            loads[bri] += dc
            return True
        if sched([c], L, H) is not None:
            part.append([c])
            loads.append(dc)
            return True
        if bd is not None:
            part[bri].insert(bpos, c)
            loads[bri] += dc
            return True
        return False

    def construct(dem, L, H):
        part = []
        loads = []
        order = sorted(ids, key=lambda c: (L[c], H[c]))
        for c in order:
            if time.time() > hard_dl - 0.3:
                part.append([c])
                loads.append(dem[c])
                continue
            if not insert_best(part, loads, c, dem, L, H):
                part.append([c])
                loads.append(dem[c])
        return part

    def local_search(routes_in, dem, L, H, dl):
        routes = [r[:] for r in routes_in if r]
        loads = [sum(dem[c] for c in r) for r in routes]

        def try_relocate():
            for i in range(len(routes)):
                if time.time() > dl:
                    return False
                r1 = routes[i]
                for p in range(len(r1)):
                    c = r1[p]
                    a = r1[p - 1] if p > 0 else 0
                    b = r1[p + 1] if p + 1 < len(r1) else END
                    rem = D[a][c] + D[c][b] - D[a][b]
                    dc = dem[c]
                    for j in range(len(routes)):
                        r2 = routes[j]
                        if j != i and loads[j] + dc > cap + 1e-9:
                            continue
                        for q in range(len(r2) + 1):
                            if j == i and (q == p or q == p + 1):
                                continue
                            prev = r2[q - 1] if q > 0 else 0
                            nxt = r2[q] if q < len(r2) else END
                            add = D[prev][c] + D[c][nxt] - D[prev][nxt]
                            if add - rem >= -1e-9:
                                continue
                            if j == i:
                                base = r1[:p] + r1[p + 1:]
                                qq = q if q < p else q - 1
                                cand = base[:qq] + [c] + base[qq:]
                                if sched(cand, L, H) is None:
                                    continue
                                routes[i] = cand
                            else:
                                cand = r2[:q] + [c] + r2[q:]
                                if sched(cand, L, H) is None:
                                    continue
                                routes[j] = cand
                                loads[j] += dc
                                routes[i] = r1[:p] + r1[p + 1:]
                                loads[i] -= dc
                            return True
            return False

        def try_2optstar():
            R = len(routes)
            for i in range(R):
                if time.time() > dl:
                    return False
                r1 = routes[i]
                pl1 = [0.0]
                for c in r1:
                    pl1.append(pl1[-1] + dem[c])
                for j in range(R):
                    if j == i:
                        continue
                    r2 = routes[j]
                    pl2 = [0.0]
                    for c in r2:
                        pl2.append(pl2[-1] + dem[c])
                    for p in range(len(r1) + 1):
                        a = r1[p - 1] if p > 0 else 0
                        b = r1[p] if p < len(r1) else END
                        for q in range(len(r2) + 1):
                            c_ = r2[q - 1] if q > 0 else 0
                            d_ = r2[q] if q < len(r2) else END
                            delta = D[a][d_] + D[c_][b] - D[a][b] - D[c_][d_]
                            if delta >= -1e-9:
                                continue
                            l1 = pl1[p] + pl2[-1] - pl2[q]
                            l2 = pl2[q] + pl1[-1] - pl1[p]
                            if l1 > cap + 1e-9 or l2 > cap + 1e-9:
                                continue
                            n1 = r1[:p] + r2[q:]
                            n2 = r2[:q] + r1[p:]
                            if n1 and sched(n1, L, H) is None:
                                continue
                            if n2 and sched(n2, L, H) is None:
                                continue
                            routes[i] = n1
                            routes[j] = n2
                            loads[i] = l1
                            loads[j] = l2
                            return True
            return False

        def try_swap():
            R = len(routes)
            for i in range(R):
                if time.time() > dl:
                    return False
                r1 = routes[i]
                for j in range(i, R):
                    r2 = routes[j]
                    for p in range(len(r1)):
                        c1 = r1[p]
                        qs = p + 1 if i == j else 0
                        for q in range(qs, len(r2)):
                            c2 = r2[q]
                            if i != j:
                                if loads[i] - dem[c1] + dem[c2] > cap + 1e-9:
                                    continue
                                if loads[j] - dem[c2] + dem[c1] > cap + 1e-9:
                                    continue
                                a1 = r1[p - 1] if p > 0 else 0
                                b1 = r1[p + 1] if p + 1 < len(r1) else END
                                a2 = r2[q - 1] if q > 0 else 0
                                b2 = r2[q + 1] if q + 1 < len(r2) else END
                                delta = (D[a1][c2] + D[c2][b1] - D[a1][c1] - D[c1][b1]) + \
                                        (D[a2][c1] + D[c1][b2] - D[a2][c2] - D[c2][b2])
                                if delta >= -1e-9:
                                    continue
                                n1 = r1[:]
                                n1[p] = c2
                                n2 = r2[:]
                                n2[q] = c1
                                if sched(n1, L, H) is None or sched(n2, L, H) is None:
                                    continue
                                routes[i] = n1
                                routes[j] = n2
                                loads[i] += dem[c2] - dem[c1]
                                loads[j] += dem[c1] - dem[c2]
                                return True
                            else:
                                n1 = r1[:]
                                n1[p], n1[q] = n1[q], n1[p]
                                if rcost(n1) < rcost(r1) - 1e-9 and sched(n1, L, H) is not None:
                                    routes[i] = n1
                                    return True
            return False

        def try_2opt():
            for i in range(len(routes)):
                if time.time() > dl:
                    return False
                r = routes[i]
                m = len(r)
                if m < 3 or m > 45:
                    continue
                base = rcost(r)
                for p in range(m - 1):
                    for q in range(p + 1, m):
                        cand = r[:p] + r[p:q + 1][::-1] + r[q + 1:]
                        if rcost(cand) < base - 1e-9 and sched(cand, L, H) is not None:
                            routes[i] = cand
                            return True
            return False

        while time.time() < dl:
            moved = try_relocate() or try_2optstar() or try_swap() or try_2opt()
            if not moved:
                break
            k = 0
            while k < len(routes):
                if not routes[k]:
                    routes.pop(k)
                    loads.pop(k)
                else:
                    k += 1
        return [r for r in routes if r]

    def split_cap(routes, dem):
        out = []
        for r in routes:
            cur = []
            load = 0.0
            for c in r:
                if cur and load + dem[c] > cap + 1e-9:
                    out.append(cur)
                    cur = []
                    load = 0.0
                cur.append(c)
                load += dem[c]
            if cur:
                out.append(cur)
        return out

    def mk_tight(w):
        Lt = [dep_lo] * (n + 2)
        Ht = [dep_hi] * (n + 2)
        for c in ids:
            Lt[c] = w[c]
            Ht[c] = w[c] + wid[c]
        return Lt, Ht

    def derive_w(ref):
        w = {}
        for r in ref:
            ts = sched(r, lo, hi)
            if ts is None:
                ts = [max(lo[c], dep_lo) for c in r]
            for c, t in zip(r, ts):
                a = lo[c]
                b = max(a, hi[c] - wid[c])
                x = t - wid[c] / 2.0
                w[c] = min(max(x, a), b)
        Lt, Ht = mk_tight(w)
        ok = all(sched(r, Lt, Ht) is not None for r in ref)
        if not ok:
            for r in ref:
                ts = sched(r, lo, hi)
                if ts is None:
                    ts = [max(lo[c], dep_lo) for c in r]
                for c, t in zip(r, ts):
                    a = lo[c]
                    b = max(a, hi[c] - wid[c])
                    w[c] = min(max(t, a), b)
        return w

    def lns(routes0, dem, L, H, dl):
        best = [r[:] for r in routes0 if r]
        bc = tot_cost(best)
        cur = [r[:] for r in best]
        cc = bc
        allc = [c for r in best for c in r]
        if len(allc) <= 1:
            return best, bc
        fails = 0
        while time.time() < dl:
            kmax = min(max(2, len(allc) // 3), 10)
            k = rng.randint(2, max(2, kmax))
            k = min(k, len(allc))
            removed = rng.sample(allc, k)
            rs = set(removed)
            part = [[c for c in r if c not in rs] for r in cur]
            part = [r for r in part if r]
            loads = [sum(dem[c] for c in r) for r in part]
            order = removed[:]
            rng.shuffle(order)
            ok = True
            for c in order:
                if not insert_best(part, loads, c, dem, L, H):
                    ok = False
                    break
            if not ok:
                continue
            cost = tot_cost(part)
            if cost < bc - 1e-9:
                part = local_search(part, dem, L, H, min(dl, time.time() + 0.5))
                cost = tot_cost(part)
                best = [r[:] for r in part]
                bc = cost
                cur = [r[:] for r in part]
                cc = cost
                fails = 0
            elif cost < cc - 1e-9:
                cur = part
                cc = cost
                fails = 0
            else:
                fails += 1
                if fails >= 40:
                    cur = [r[:] for r in best]
                    cc = bc
                    fails = 0
        return best, bc

    # ---------- Phase 1: reference routing & window derivation ----------
    usable = hard_dl - time.time()
    cands = [dmean]
    if any(abs(dmax[i] - dmean[i]) > 1e-9 for i in ids):
        cands.append(dmax)
    ref_budget = max(1.0, 0.22 * usable)
    per_c = ref_budget / len(cands)

    best_ref = None
    best_val = None
    best_w = None
    for dref in cands:
        dl = min(hard_dl - 0.5, time.time() + per_c)
        r = construct(dref, lo, hi)
        r = local_search(r, dref, lo, hi, dl)
        w = derive_w(r)
        Lt, Ht = mk_tight(w)
        val = 0.0
        ok = True
        for sid in sids:
            segs = split_cap(r, dem_s[sid])
            for sg in segs:
                if sched(sg, Lt, Ht) is None:
                    ok = False
                    break
            if not ok:
                break
            val += probs[sid] * tot_cost(segs)
        if not ok:
            continue
        if best_val is None or val < best_val:
            best_val = val
            best_ref = r
            best_w = w

    if best_ref is None:
        r = construct(dmax, lo, hi)
        w = {}
        for rt in r:
            ts = sched(rt, lo, hi)
            if ts is None:
                ts = [max(lo[c], dep_lo) for c in rt]
            for c, t in zip(rt, ts):
                a = lo[c]
                b = max(a, hi[c] - wid[c])
                w[c] = min(max(t, a), b)
        best_ref = r
        best_w = w

    Lt, Ht = mk_tight(best_w)

    def build_solution(scen_sol):
        tw = {str(c): [best_w[c], best_w[c] + wid[c]] for c in ids}
        routes_out = {}
        obj = 0.0
        for sid in sids:
            lst = []
            tot = 0.0
            for r in scen_sol[sid]:
                if not r:
                    continue
                ts = sched(r, Lt, Ht)
                if ts is None:
                    ts = sched(r, lo, hi)
                if ts is None:
                    t = dep_lo
                    prev = 0
                    ts = []
                    for c in r:
                        t = max(t + D[prev][c], Lt[c])
                        ts.append(t)
                        prev = c
                at = {"0": dep_lo}
                for c, t in zip(r, ts):
                    at[str(c)] = t
                at[str(END)] = ts[-1] + D[r[-1]][END]
                cst = rcost(r)
                lst.append({"route": [0] + list(r) + [END],
                            "arrival_times": at, "cost": cst})
                tot += cst
            routes_out[str(sid)] = lst
            obj += probs[sid] * tot
        return obj, {"objective_value": obj, "time_windows": tw, "routes": routes_out}

    # quick baseline: split reference per scenario
    scen_sol = {sid: split_cap(best_ref, dem_s[sid]) for sid in sids}
    scen_cost = {sid: tot_cost(scen_sol[sid]) for sid in sids}
    obj, sol = build_solution(scen_sol)
    if logger:
        logger.log_solution(obj, sol)

    # ---------- Phase 2: per-scenario optimization under tight windows ----------
    S = len(sids)
    remaining = max(0.0, hard_dl - time.time())
    per = remaining * 0.5 / max(1, S)
    for sid in sids:
        d = dem_s[sid]
        dl1 = min(hard_dl, time.time() + per * 0.55)
        r1 = local_search(scen_sol[sid], d, Lt, Ht, dl1)
        c1 = tot_cost(r1)
        br, bcst = r1, c1
        if time.time() < hard_dl - 0.5:
            r2 = construct(d, Lt, Ht)
            r2 = local_search(r2, d, Lt, Ht, min(hard_dl, time.time() + per * 0.4))
            c2 = tot_cost(r2)
            if c2 < bcst - 1e-9:
                br, bcst = r2, c2
        if bcst < scen_cost[sid] - 1e-9:
            scen_sol[sid] = br
            scen_cost[sid] = bcst
            obj, sol = build_solution(scen_sol)
            if logger:
                logger.log_solution(obj, sol)

    obj, sol = build_solution(scen_sol)
    if logger:
        logger.log_solution(obj, sol)

    # ---------- Phase 3: LNS improvement round-robin ----------
    try:
        while time.time() < hard_dl - 0.2:
            order = sorted(sids, key=lambda s: -probs[s] * scen_cost[s])
            any_left = False
            for sid in order:
                now = time.time()
                if now >= hard_dl - 0.2:
                    break
                any_left = True
                chunk = max(0.3, min(3.0, (hard_dl - now) / (2.0 * S)))
                dl = min(hard_dl, now + chunk)
                r, c = lns(scen_sol[sid], dem_s[sid], Lt, Ht, dl)
                if c < scen_cost[sid] - 1e-9:
                    scen_sol[sid] = r
                    scen_cost[sid] = c
                    obj, sol = build_solution(scen_sol)
                    if logger:
                        logger.log_solution(obj, sol)
            if not any_left:
                break
    except Exception:
        pass

    obj, sol = build_solution(scen_sol)
    if logger:
        logger.log_solution(obj, sol)
    with open(args.solution_path, 'w') as f:
        json.dump(sol, f)


if __name__ == '__main__':
    main()
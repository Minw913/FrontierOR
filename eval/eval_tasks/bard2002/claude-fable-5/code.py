import argparse
import json
import math
import random
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--instance_path', required=True)
    ap.add_argument('--solution_path', required=True)
    ap.add_argument('--time_limit', type=int, default=60)
    ap.add_argument('--log_path', default=None)
    args = ap.parse_args()

    start = time.time()
    random.seed(0)

    try:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    except Exception:
        logger = None

    with open(args.instance_path) as f:
        inst = json.load(f)

    n = inst['num_customers']
    Q = inst['vehicle_capacity']
    dep = inst['depot']
    N = n + 1

    x = [0.0] * N; y = [0.0] * N
    demv = [0] * N; ready = [0.0] * N; due = [0.0] * N; serv = [0.0] * N
    x[0] = float(dep['x']); y[0] = float(dep['y'])
    ready[0] = float(dep['ready_time']); due[0] = float(dep['due_date'])
    for c in inst['customers']:
        i = c['id']
        x[i] = float(c['x']); y[i] = float(c['y'])
        demv[i] = c['demand']; ready[i] = float(c['ready_time'])
        due[i] = float(c['due_date']); serv[i] = float(c['service_time'])

    DM = inst.get('distance_matrix')
    if DM is None:
        DM = [[math.hypot(x[i] - x[j], y[i] - y[j]) for j in range(N)] for i in range(N)]
    # Euclidean distances for cost/reporting
    DE = [[math.hypot(x[i] - x[j], y[i] - y[j]) for j in range(N)] for i in range(N)]

    inc = bool(inst.get('travel_time_includes_service_time', False))
    # effective travel times
    T = [[0.0] * N for _ in range(N)]
    for i in range(N):
        Ti = T[i]; Di = DM[i]
        si = 0.0 if inc else serv[i]
        di = due[i]
        for j in range(N):
            t = math.floor(Di[j] * 10 + 1e-9) / 10.0 + si
            g = ready[j] - di
            if g > t:
                t = g
            Ti[j] = t

    # ---------------- route helpers ----------------
    def refresh(r):
        seq = r['seq']
        e = []; t = ready[0]; prev = 0
        for c in seq:
            t2 = t + T[prev][c]
            if t2 < ready[c]:
                t2 = ready[c]
            e.append(t2); t = t2; prev = c
        m = len(seq)
        lat = [0.0] * m
        nl = due[0]; nx = 0
        for i in range(m - 1, -1, -1):
            c = seq[i]
            l = nl - T[c][nx]
            if l > due[c]:
                l = due[c]
            lat[i] = l; nl = l; nx = c
        r['e'] = e; r['lat'] = lat
        r['load'] = sum(demv[c] for c in seq)

    def make_route(seq):
        r = {'seq': list(seq)}
        refresh(r)
        return r

    def feasible_insertions(v, routes):
        res = []
        dv = demv[v]
        rv = ready[v]; dvw = due[v]
        for ri, r in enumerate(routes):
            if r['load'] + dv > Q:
                continue
            seq = r['seq']; e = r['e']; lat = r['lat']; L = len(seq)
            for p in range(L + 1):
                if p > 0:
                    prev = seq[p - 1]; ep = e[p - 1]
                else:
                    prev = 0; ep = ready[0]
                ev = ep + T[prev][v]
                if ev < rv:
                    ev = rv
                if ev > dvw:
                    continue
                if p < L:
                    nxt = seq[p]
                    en = ev + T[v][nxt]
                    if en < ready[nxt]:
                        en = ready[nxt]
                    if en > lat[p]:
                        continue
                    cost = DE[prev][v] + DE[v][nxt] - DE[prev][nxt]
                else:
                    if ev + T[v][0] > due[0]:
                        continue
                    cost = DE[prev][v] + DE[v][0] - DE[prev][0]
                res.append((cost, ri, p))
        return res

    def eject_insert(v, routes, pcount):
        best = None
        dv = demv[v]; rv = ready[v]; dvw = due[v]
        for ri, r in enumerate(routes):
            seq = r['seq']; L = len(seq)
            if L == 0:
                continue
            load = r['load']
            for wi in range(L):
                w = seq[wi]
                if load - demv[w] + dv > Q:
                    continue
                pw = pcount.get(w, 0)
                if best is not None and pw > best[0]:
                    continue
                tmp = seq[:wi] + seq[wi + 1:]
                m = len(tmp)
                e = []; t = ready[0]; prev = 0
                for c in tmp:
                    t2 = t + T[prev][c]
                    if t2 < ready[c]:
                        t2 = ready[c]
                    e.append(t2); t = t2; prev = c
                lat = [0.0] * m
                nl = due[0]; nx = 0
                for i2 in range(m - 1, -1, -1):
                    c = tmp[i2]
                    l = nl - T[c][nx]
                    if l > due[c]:
                        l = due[c]
                    lat[i2] = l; nl = l; nx = c
                for p in range(m + 1):
                    if p > 0:
                        prevn = tmp[p - 1]; ep = e[p - 1]
                    else:
                        prevn = 0; ep = ready[0]
                    ev = ep + T[prevn][v]
                    if ev < rv:
                        ev = rv
                    if ev > dvw:
                        continue
                    if p < m:
                        nxt = tmp[p]
                        en = ev + T[v][nxt]
                        if en < ready[nxt]:
                            en = ready[nxt]
                        if en > lat[p]:
                            continue
                        cost = DE[prevn][v] + DE[v][nxt] - DE[prevn][nxt]
                    else:
                        if ev + T[v][0] > due[0]:
                            continue
                        cost = DE[prevn][v] + DE[v][0] - DE[prevn][0]
                    cand = (pw, cost, ri, wi, p)
                    if best is None or cand < best:
                        best = cand
        return best

    def perturb(routes, k):
        for _ in range(k):
            nonempty = [r for r in routes if r['seq']]
            if not nonempty:
                return
            r1 = random.choice(nonempty)
            i = random.randrange(len(r1['seq']))
            v = r1['seq'][i]
            del r1['seq'][i]
            refresh(r1)
            cands = feasible_insertions(v, routes)
            if cands:
                cost, ri, p = random.choice(cands)
                routes[ri]['seq'].insert(p, v)
                refresh(routes[ri])
            else:
                r1['seq'].insert(i, v)
                refresh(r1)

    def build_solution(seqs):
        routes_out = []; dtimes = {}; loads = {}; total = 0.0
        for seq in seqs:
            t = ready[0]; prev = 0; load = 0
            for c in seq:
                t2 = t + T[prev][c]
                if t2 < ready[c]:
                    t2 = ready[c]
                t = t2
                load += demv[c]
                dtimes[str(c)] = round(t, 6)
                loads[str(c)] = load
                total += DE[prev][c]
                prev = c
            total += DE[prev][0]
            routes_out.append([0] + list(seq) + [0])
        return {
            "objective_value": float(len(seqs)),
            "num_vehicles": len(seqs),
            "routes": routes_out,
            "total_distance": round(total, 6),
            "departure_times": dtimes,
            "loads": loads,
        }

    # ---------------- initial construction (nearest neighbor) ----------------
    unr = set(range(1, N))
    cur = []
    while unr:
        seq = []; load = 0; t = ready[0]; prev = 0
        while True:
            bestv = None; bestm = None
            for v in unr:
                if load + demv[v] > Q:
                    continue
                a0 = t + T[prev][v]
                a = a0 if a0 >= ready[v] else ready[v]
                if a > due[v]:
                    continue
                if a + T[v][0] > due[0]:
                    continue
                m = DE[prev][v] + (a - a0)
                if bestm is None or m < bestm:
                    bestm = m; bestv = v
            if bestv is None:
                break
            v = bestv
            unr.discard(v)
            seq.append(v); load += demv[v]
            a0 = t + T[prev][v]
            t = a0 if a0 >= ready[v] else ready[v]
            prev = v
        if not seq:
            # force-place a remaining customer alone (shouldn't happen on feasible instances)
            v = unr.pop()
            seq = [v]
        cur.append(make_route(seq))

    cur = [r for r in cur if r['seq']]
    best_seqs = [list(r['seq']) for r in cur]
    if logger:
        logger.log_solution(float(len(best_seqs)), build_solution(best_seqs))

    total_dem = sum(demv[1:])
    LB = max(1, int(math.ceil(total_dem / float(Q)))) if n > 0 else 0

    tl = max(1, args.time_limit)
    deadline = start + tl - 0.3
    min_end = min(deadline, start + tl * 0.92)

    # ---------------- route minimization via ejection pool ----------------
    while n > 0 and time.time() < min_end and len(cur) > LB and len(cur) > 1:
        # choose route to eliminate
        if random.random() < 0.3:
            ri = random.randrange(len(cur))
        else:
            ri = min(range(len(cur)), key=lambda i: (len(cur[i]['seq']), random.random()))
        removed = cur.pop(ri)
        backup = [list(r['seq']) for r in cur] + [list(removed['seq'])]
        EP = list(removed['seq'])
        random.shuffle(EP)
        pcount = {}
        maxit = min(6000, max(1500, 60 * len(EP)))
        rem = min_end - time.time()
        att_end = min(min_end, time.time() + max(2.0, 0.2 * rem))
        it = 0
        while EP:
            it += 1
            if it > maxit:
                break
            if (it & 31) == 0 and time.time() > att_end:
                break
            v = EP.pop()
            cands = feasible_insertions(v, cur)
            if cands:
                cost, ri2, p = random.choice(cands)
                r = cur[ri2]
                r['seq'].insert(p, v)
                refresh(r)
                continue
            pcount[v] = pcount.get(v, 0) + 1
            ej = eject_insert(v, cur, pcount)
            if ej is not None:
                pw, cost, ri2, wi, p = ej
                r = cur[ri2]
                seq = r['seq']
                w = seq[wi]
                tmp = seq[:wi] + seq[wi + 1:]
                r['seq'] = tmp[:p] + [v] + tmp[p:]
                refresh(r)
                EP.append(w)
            else:
                EP.append(v)
                perturb(cur, 10)
                cur = [r for r in cur if r['seq']]
        if not EP:
            cur = [r for r in cur if r['seq']]
            best_seqs = [list(r['seq']) for r in cur]
            if logger:
                logger.log_solution(float(len(best_seqs)), build_solution(best_seqs))
        else:
            cur = [make_route(s) for s in backup if s]

    # ---------------- distance improvement (relocate) ----------------
    def distance_ls(seqs, end_time):
        routes = [make_route(list(s)) for s in seqs]
        loc = {}
        for r in routes:
            for c in r['seq']:
                loc[c] = r
        ids = [c for r in routes for c in r['seq']]
        improved = True
        while improved and time.time() < end_time:
            improved = False
            random.shuffle(ids)
            for v in ids:
                if time.time() > end_time:
                    break
                r = loc[v]
                seq = r['seq']
                try:
                    i = seq.index(v)
                except ValueError:
                    continue
                prev = seq[i - 1] if i > 0 else 0
                nxt = seq[i + 1] if i + 1 < len(seq) else 0
                gain = DE[prev][v] + DE[v][nxt] - DE[prev][nxt]
                del seq[i]
                refresh(r)
                cands = feasible_insertions(v, routes)
                best = min(cands) if cands else None
                if best is not None and best[0] < gain - 1e-7:
                    cost, ri2, p = best
                    r2 = routes[ri2]
                    r2['seq'].insert(p, v)
                    refresh(r2)
                    loc[v] = r2
                    improved = True
                else:
                    seq.insert(i, v)
                    refresh(r)
            routes = [r for r in routes if r['seq']]
        return [list(r['seq']) for r in routes if r['seq']]

    if n > 0 and time.time() < deadline:
        new_seqs = distance_ls(best_seqs, deadline)
        # validate coverage
        seen = set()
        ok = True
        for s in new_seqs:
            for c in s:
                if c in seen:
                    ok = False
                seen.add(c)
        if ok and len(seen) == n:
            if len(new_seqs) < len(best_seqs):
                best_seqs = new_seqs
                if logger:
                    logger.log_solution(float(len(best_seqs)), build_solution(best_seqs))
            else:
                best_seqs = new_seqs

    # final validation; fallback if anything went wrong
    seen = set()
    valid = True
    for s in best_seqs:
        r = make_route(s)
        if r['load'] > Q:
            valid = False
        for idx, c in enumerate(s):
            if c in seen:
                valid = False
            seen.add(c)
            if r['e'][idx] > due[c] + 1e-6:
                valid = False
        if s and r['e'][-1] + T[s[-1]][0] > due[0] + 1e-6:
            valid = False
    if len(seen) != n:
        valid = False
    if not valid and n > 0:
        best_seqs = [[c] for c in range(1, N)]

    sol = build_solution(best_seqs)
    if logger:
        logger.log_solution(sol["objective_value"], sol)
    with open(args.solution_path, 'w') as f:
        json.dump(sol, f, indent=2)


if __name__ == '__main__':
    main()
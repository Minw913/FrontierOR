import argparse
import json
import math
import random
import time

from solution_logger import SolutionLogger

EPS = 1e-9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--instance_path', required=True)
    ap.add_argument('--solution_path', required=True)
    ap.add_argument('--time_limit', type=int, default=60)
    ap.add_argument('--log_path', default=None)
    args = ap.parse_args()

    t_start = time.time()
    deadline = t_start + max(1, args.time_limit) - 1.0

    random.seed(0)

    with open(args.instance_path) as f:
        inst = json.load(f)

    n = inst['num_customers']
    V = inst['num_vehicles']
    Q = inst['vehicle_capacity']
    nodes = sorted(inst['nodes'], key=lambda nd: nd['id'])
    tws = [nd['time_window_start'] for nd in nodes]
    twe = [nd['time_window_end'] for nd in nodes]
    srv = [nd['service_time'] for nd in nodes]
    dem = [nd['demand'] for nd in nodes]
    D = inst['distance_matrix']

    access = {}
    emap = {}
    for es in inst.get('edge_sets', []):
        sid = es['id']
        access[sid] = float(es['access_cost'])
        for e in es['edges']:
            i, j = int(e[0]), int(e[1])
            if i > j:
                i, j = j, i
            emap[(i, j)] = sid

    def eset(i, j):
        if i > j:
            i, j = j, i
        return emap.get((i, j), -1)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # ---------- trivial case ----------
    if n == 0:
        sol = {'objective_value': 0.0, 'routes': [], 'edge_sets_used': []}
        if logger:
            logger.log_solution(0.0, sol)
        with open(args.solution_path, 'w') as f:
            json.dump(sol, f, indent=2)
        return

    # ---------- route time-window machinery ----------
    def compute_TL(seq):
        """Earliest service-start times T and latest feasible start times L.
        Returns (T, L) or (None, None) if infeasible."""
        m = len(seq)
        T = [0.0] * m
        t = tws[0]
        prev = 0
        for idx in range(m):
            u = seq[idx]
            t = t + srv[prev] + D[prev][u]
            if t < tws[u]:
                t = tws[u]
            if t > twe[u] + EPS:
                return None, None
            T[idx] = t
            prev = u
        last = seq[-1]
        if T[-1] + srv[last] + D[last][0] > twe[0] + EPS:
            return None, None
        L = [0.0] * m
        nxt = 0
        lat = twe[0]
        for idx in range(m - 1, -1, -1):
            u = seq[idx]
            lat = lat - srv[u] - D[u][nxt]
            if lat > twe[u]:
                lat = twe[u]
            if lat < T[idx] - EPS:
                return None, None
            L[idx] = lat
            nxt = u
        return T, L

    def build_state(seqs):
        routes = []
        cnt = {}
        dist = 0.0
        for seq in seqs:
            if not seq:
                continue
            T, L = compute_TL(seq)
            if T is None:
                return None
            load = sum(dem[u] for u in seq)
            routes.append({'seq': seq, 'load': load, 'T': T, 'L': L})
            prev = 0
            for u in seq:
                dist += D[prev][u]
                s = eset(prev, u)
                if s >= 0:
                    cnt[s] = cnt.get(s, 0) + 1
                prev = u
            dist += D[prev][0]
            s = eset(prev, 0)
            if s >= 0:
                cnt[s] = cnt.get(s, 0) + 1
        return routes, cnt, dist

    def total_cost(cnt, dist):
        c = dist
        for s, k in cnt.items():
            if k > 0:
                c += access[s]
        return c

    def access_delta(cnt, adds, rems):
        tmp = {}
        for s in adds:
            tmp[s] = tmp.get(s, 0) + 1
        for s in rems:
            tmp[s] = tmp.get(s, 0) - 1
        d = 0.0
        for s, dc in tmp.items():
            c = cnt.get(s, 0)
            if c == 0 and c + dc > 0:
                d += access[s]
            elif c > 0 and c + dc <= 0:
                d -= access[s]
        return d

    # ---------- insertion evaluation ----------
    def try_insert_eval(rd, u, cnt):
        seq = rd['seq']
        m = len(seq)
        if rd['load'] + dem[u] > Q:
            return None
        T = rd['T']
        L = rd['L']
        best = None
        du = D[u]
        for p in range(m + 1):
            if p > 0:
                a = seq[p - 1]
                Ta = T[p - 1]
            else:
                a = 0
                Ta = tws[0]
            if p < m:
                b = seq[p]
                Lb = L[p]
            else:
                b = 0
                Lb = twe[0]
            arr = Ta + srv[a] + D[a][u]
            if arr < tws[u]:
                arr = tws[u]
            if arr > twe[u] + EPS:
                continue
            if arr + srv[u] + du[b] > Lb + EPS:
                continue
            dd = D[a][u] + du[b] - D[a][b]
            adds = []
            s1 = eset(a, u)
            if s1 >= 0:
                adds.append(s1)
            s2 = eset(u, b)
            if s2 >= 0:
                adds.append(s2)
            rems = []
            s0 = eset(a, b)
            if s0 >= 0:
                rems.append(s0)
            dc = dd + access_delta(cnt, adds, rems)
            if best is None or dc < best[0]:
                best = (dc, p, dd, adds, rems)
        return best

    def new_route_eval(u, cnt):
        arr = tws[0] + srv[0] + D[0][u]
        if arr < tws[u]:
            arr = tws[u]
        if arr > twe[u] + EPS:
            return None
        if arr + srv[u] + D[u][0] > twe[0] + EPS:
            return None
        if dem[u] > Q:
            return None
        dd = D[0][u] + D[u][0]
        adds = []
        s = eset(0, u)
        if s >= 0:
            adds.append(s)
            adds.append(s)
        dc = dd + access_delta(cnt, adds, [])
        return (dc, dd, adds)

    # ---------- repair (greedy cheapest insertion with cache) ----------
    def repair(routes, cnt, dist, unassigned, noise=0.0):
        U = list(unassigned)
        random.shuffle(U)

        def best_for(u):
            best = None
            for ri in range(len(routes)):
                r = try_insert_eval(routes[ri], u, cnt)
                if r is not None and (best is None or r[0] < best[0]):
                    best = (r[0], ri, r[1], r[2], r[3], r[4])
            if len(routes) < V:
                r = new_route_eval(u, cnt)
                if r is not None and (best is None or r[0] < best[0]):
                    best = (r[0], -1, 0, r[1], r[2], [])
            return best

        cache = {u: best_for(u) for u in U}
        while U:
            pick_u = None
            pick_key = None
            for u in U:
                c = cache[u]
                if c is None:
                    continue
                key = c[0] * (1.0 + noise * random.random()) if noise > 0 else c[0]
                if pick_key is None or key < pick_key:
                    pick_key = key
                    pick_u = u
            if pick_u is None:
                return None
            u = pick_u
            c = best_for(u)  # refresh (cache may be stale)
            if c is None:
                return None
            dc, ri, pos, dd, adds, rems = c
            if ri == -1:
                seq = [u]
                T, L = compute_TL(seq)
                if T is None:
                    return None
                routes.append({'seq': seq, 'load': dem[u], 'T': T, 'L': L})
                mod = len(routes) - 1
            else:
                rd = routes[ri]
                rd['seq'].insert(pos, u)
                rd['load'] += dem[u]
                T, L = compute_TL(rd['seq'])
                if T is None:
                    return None
                rd['T'], rd['L'] = T, L
                mod = ri
            dist += dd
            affected = set(adds) | set(rems)
            pre = {s: cnt.get(s, 0) > 0 for s in affected}
            for s in adds:
                cnt[s] = cnt.get(s, 0) + 1
            for s in rems:
                cnt[s] = cnt.get(s, 0) - 1
            status_changed = any(pre[s] != (cnt.get(s, 0) > 0) for s in affected)
            U.remove(u)
            if status_changed:
                for v in U:
                    cache[v] = best_for(v)
            else:
                for v in U:
                    cb = cache[v]
                    if cb is None or cb[1] == mod or cb[1] == -1:
                        cache[v] = best_for(v)
        return routes, cnt, dist

    # ---------- destroy operators ----------
    def destroy(cur_seqs, q):
        all_cust = [u for s in cur_seqs for u in s]
        if not all_cust:
            return set()
        q = min(q, len(all_cust))
        r = random.random()
        removed = set()
        if r < 0.30:
            removed = set(random.sample(all_cust, q))
        elif r < 0.55:
            seed = random.choice(all_cust)
            removed = {seed}
            cands = sorted((c for c in all_cust if c != seed),
                           key=lambda c: D[seed][c])
            while len(removed) < q and cands:
                i = int((random.random() ** 3) * len(cands))
                removed.add(cands.pop(i))
        elif r < 0.75:
            gains = {}
            for s in cur_seqs:
                m = len(s)
                for idx in range(m):
                    u = s[idx]
                    a = s[idx - 1] if idx > 0 else 0
                    b = s[idx + 1] if idx < m - 1 else 0
                    g = D[a][u] + D[u][b] - D[a][b]
                    sa = eset(a, u)
                    sb = eset(u, b)
                    if sa >= 0:
                        g += 0.3 * access[sa]
                    if sb >= 0:
                        g += 0.3 * access[sb]
                    gains[u] = g
            cands = sorted(gains, key=lambda u: -gains[u])
            while len(removed) < q and cands:
                i = int((random.random() ** 3) * len(cands))
                removed.add(cands.pop(i))
        elif r < 0.90 or not access:
            rt = random.choice(cur_seqs)
            if len(rt) <= q:
                removed = set(rt)
            else:
                st = random.randint(0, len(rt) - q)
                removed = set(rt[st:st + q])
        else:
            # try to close an accessed edge set
            used = set()
            for s in cur_seqs:
                prev = 0
                for u in s:
                    ss = eset(prev, u)
                    if ss >= 0:
                        used.add(ss)
                    prev = u
                ss = eset(prev, 0)
                if ss >= 0:
                    used.add(ss)
            if not used:
                removed = set(random.sample(all_cust, q))
            else:
                target = random.choice(list(used))
                for s in cur_seqs:
                    prev = 0
                    for u in s:
                        if eset(prev, u) == target:
                            if prev != 0:
                                removed.add(prev)
                            removed.add(u)
                        prev = u
                    if eset(prev, 0) == target and prev != 0:
                        removed.add(prev)
                if not removed:
                    removed = set(random.sample(all_cust, q))
                removed = set(list(removed)[:max(q, 6)])
        return removed

    # ---------- solution export ----------
    def make_sol(seqs):
        routes_out = []
        used = set()
        total = 0.0
        for k, seq in enumerate(seqs):
            T, L = compute_TL(seq)
            vt = {'0': float(tws[0])}
            for idx, u in enumerate(seq):
                vt[str(u)] = float(T[idx])
            arcs = []
            prev = 0
            for u in seq + [0]:
                arcs.append([prev, u])
                total += D[prev][u]
                s = eset(prev, u)
                if s >= 0:
                    used.add(s)
                prev = u
            routes_out.append({'vehicle': k,
                               'sequence': [0] + seq + [0],
                               'arcs': arcs,
                               'visit_times': vt})
        for s in used:
            total += access[s]
        return {'objective_value': total,
                'routes': routes_out,
                'edge_sets_used': sorted(used)}, total

    # ---------- initial solution ----------
    init = None
    for attempt in range(30):
        res = repair([], {}, 0.0, list(range(1, n + 1)),
                     noise=0.0 if attempt == 0 else 0.4)
        if res is not None:
            init = res
            break
    if init is None:
        # last resort: singleton routes (ignore V limit rather than fail)
        seqs = [[u] for u in range(1, n + 1)]
        st = build_state(seqs)
        init = st

    routes, cnt, dist = init
    cur_seqs = [list(rd['seq']) for rd in routes]
    cur_cost = total_cost(cnt, dist)

    best_seqs = [list(s) for s in cur_seqs]
    best_cost = cur_cost
    sol, chk = make_sol(best_seqs)
    best_cost = chk
    cur_cost = chk
    if logger:
        logger.log_solution(best_cost, sol)

    # ---------- LNS / simulated annealing ----------
    qmax = min(max(4, n // 4), 40, n)
    T0 = max(1e-6, 0.04 * best_cost)
    Tend = max(1e-9, T0 * 1e-3)
    horizon = max(1.0, deadline - time.time())
    it = 0
    while time.time() < deadline:
        it += 1
        frac = min(1.0, (time.time() - t_start) / max(1.0, horizon))
        Temp = T0 * ((Tend / T0) ** frac)

        q = random.randint(2, qmax)
        removed = destroy(cur_seqs, q)
        if not removed:
            continue
        new_seqs = []
        for s in cur_seqs:
            ns = [u for u in s if u not in removed]
            if ns:
                new_seqs.append(ns)
        st = build_state(new_seqs)
        if st is None:
            continue
        r2, c2, d2 = st
        noise = 0.15 if random.random() < 0.5 else 0.0
        res = repair(r2, c2, d2, list(removed), noise=noise)
        if res is None:
            continue
        r2, c2, d2 = res
        new_cost = total_cost(c2, d2)
        acc = False
        if new_cost < cur_cost - 1e-9:
            acc = True
        elif Temp > 1e-12:
            delta = new_cost - cur_cost
            if delta / Temp < 30 and random.random() < math.exp(-delta / Temp):
                acc = True
        if acc:
            cur_seqs = [list(rd['seq']) for rd in r2]
            cur_cost = new_cost
            if new_cost < best_cost - 1e-9:
                best_seqs = [list(s) for s in cur_seqs]
                sol, chk = make_sol(best_seqs)
                best_cost = chk
                if logger:
                    logger.log_solution(best_cost, sol)
        # occasional restart from best to intensify
        if it % 2000 == 0:
            cur_seqs = [list(s) for s in best_seqs]
            cur_cost = best_cost

    # ---------- write output ----------
    sol, chk = make_sol(best_seqs)
    sol['objective_value'] = chk
    with open(args.solution_path, 'w') as f:
        json.dump(sol, f, indent=2)


if __name__ == '__main__':
    main()
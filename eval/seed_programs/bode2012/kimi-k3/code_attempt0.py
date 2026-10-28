import argparse
import json
import time
import random
import heapq
from collections import Counter

from solution_logger import SolutionLogger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--instance_path', required=True)
    ap.add_argument('--solution_path', required=True)
    ap.add_argument('--time_limit', type=int, required=True)
    ap.add_argument('--log_path', default=None)
    args = ap.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    t_start = time.time()
    time_limit = max(0, args.time_limit)
    reserve = min(0.8, max(0.05, 0.04 * time_limit + 0.05))
    deadline = t_start + time_limit - reserve

    random.seed(20240517)

    with open(args.instance_path) as f:
        inst = json.load(f)

    def expired():
        return time.time() >= deadline

    def time_left():
        return deadline - time.time()

    # ---------------- instance parsing ----------------
    graph = inst['graph']
    n = graph['num_nodes']
    edges = inst['edges']
    depot = inst['depot']
    K = max(1, inst['fleet']['num_vehicles'])
    Q = inst['fleet']['vehicle_capacity']

    req = [e for e in edges if e.get('is_required', False)]
    E = len(req)
    svc_const = float(sum(e.get('service_cost', 0) for e in req))

    adj = [[] for _ in range(n)]
    for e in edges:
        u, v = e['endpoints']
        c = e['cost']
        eid = e['edge_id']
        adj[u].append((v, c, eid))
        adj[v].append((u, c, eid))

    if E == 0:
        out = {'objective_value': float(svc_const), 'routes': []}
        with open(args.solution_path, 'w') as f:
            json.dump(out, f)
        return

    # Distinguished nodes: depot + endpoints of required edges
    DN = [depot]
    dindex = {depot: 0}
    for e in req:
        u, v = e['endpoints']
        if u not in dindex:
            dindex[u] = len(DN)
            DN.append(u)
        if v not in dindex:
            dindex[v] = len(DN)
            DN.append(v)
    Dcnt = len(DN)

    # Shortest paths from each distinguished node
    Dmat = [None] * Dcnt
    parN = [None] * Dcnt
    parE = [None] * Dcnt

    def dijkstra(src):
        dist = [float('inf')] * n
        pn = [-1] * n
        pe = [-1] * n
        dist[src] = 0
        h = [(0, src)]
        while h:
            d, u = heapq.heappop(h)
            if d > dist[u]:
                continue
            for v, c, eid in adj[u]:
                nd = d + c
                if nd < dist[v]:
                    dist[v] = nd
                    pn[v] = u
                    pe[v] = eid
                    heapq.heappush(h, (nd, v))
        return dist, pn, pe

    for i, s in enumerate(DN):
        dist, pn, pe = dijkstra(s)
        Dmat[i] = [dist[DN[j]] for j in range(Dcnt)]
        parN[i] = pn
        parE[i] = pe

    RequestId = [e['edge_id'] for e in req]
    U = [e['endpoints'][0] for e in req]
    V = [e['endpoints'][1] for e in req]
    Dem = [e.get('demand', 0) for e in req]
    Uidx = [dindex[u] for u in U]
    Vidx = [dindex[v] for v in V]

    # ---------------- route helpers ----------------
    def item_entry(it):
        r, o = it
        return Uidx[r] if o == 0 else Vidx[r]

    def item_exit(it):
        r, o = it
        return Vidx[r] if o == 0 else Uidx[r]

    def route_cost(items):
        c = 0.0
        pos = 0
        for r, o in items:
            if o == 0:
                c += Dmat[pos][Uidx[r]]
                pos = Vidx[r]
            else:
                c += Dmat[pos][Vidx[r]]
                pos = Uidx[r]
        c += Dmat[pos][0]
        return c

    def route_dem(items):
        d = 0
        for r, _ in items:
            d += Dem[r]
        return d

    def make_route(items):
        return {'items': items, 'dem': route_dem(items), 'cost': route_cost(items)}

    def best_insertion(items, r):
        u0 = Uidx[r]
        x0 = Vidx[r]
        best_d = float('inf')
        best_pos = 0
        best_o = 0
        prev_exit = 0
        row = Dmat[0]
        nI = len(items)
        for posn in range(nI + 1):
            if posn == nI:
                next_entry = 0
            else:
                it = items[posn]
                next_entry = Uidx[it[0]] if it[1] == 0 else Vidx[it[0]]
            base = row[next_entry]
            d0 = row[u0] + Dmat[x0][next_entry] - base
            if d0 < best_d:
                best_d = d0
                best_pos = posn
                best_o = 0
            d1 = row[x0] + Dmat[u0][next_entry] - base
            if d1 < best_d:
                best_d = d1
                best_pos = posn
                best_o = 1
            if posn < nI:
                it = items[posn]
                prev_exit = Vidx[it[0]] if it[1] == 0 else Uidx[it[0]]
                row = Dmat[prev_exit]
        return best_d, best_pos, best_o

    def insertion_delta_at(items, r, posn, oo):
        if posn == 0:
            prev_exit = 0
        else:
            prev_exit = item_exit(items[posn - 1])
        if posn == len(items):
            next_entry = 0
        else:
            next_entry = item_entry(items[posn])
        base = Dmat[prev_exit][next_entry]
        if oo == 0:
            ent = Uidx[r]
            ext = Vidx[r]
        else:
            ent = Vidx[r]
            ext = Uidx[r]
        return Dmat[prev_exit][ent] + Dmat[ext][next_entry] - base

    def removal_delta(items, i):
        prev_exit = 0 if i == 0 else item_exit(items[i - 1])
        ce = item_entry(items[i])
        cx = item_exit(items[i])
        next_entry = 0 if i == len(items) - 1 else item_entry(items[i + 1])
        old = Dmat[prev_exit][ce] + Dmat[cx][next_entry]
        new = Dmat[prev_exit][next_entry]
        return new - old

    # ---------------- construction ----------------
    def construct(rand):
        unassigned = set(range(E))
        routes = []
        while unassigned:
            items = []
            load = 0
            pos = 0
            while True:
                r = -1
                if rand:
                    top = []
                    for rr in unassigned:
                        if load + Dem[rr] > Q:
                            continue
                        v1 = Dmat[pos][Uidx[rr]]
                        v2 = Dmat[pos][Vidx[rr]]
                        vcur = v1 if v1 < v2 else v2
                        if len(top) < 3:
                            top.append((vcur, rr))
                            top.sort()
                        elif vcur < top[-1][0]:
                            top[-1] = (vcur, rr)
                            top.sort()
                    if top:
                        r = random.choice(top)[1]
                else:
                    bestv = float('inf')
                    bestdem = -1
                    for rr in unassigned:
                        if load + Dem[rr] > Q:
                            continue
                        v1 = Dmat[pos][Uidx[rr]]
                        v2 = Dmat[pos][Vidx[rr]]
                        vcur = v1 if v1 < v2 else v2
                        if (vcur < bestv - 1e-12 or
                                (abs(vcur - bestv) <= 1e-12 and Dem[rr] > bestdem)):
                            bestv = vcur
                            r = rr
                            bestdem = Dem[rr]
                if r < 0:
                    break
                if Dmat[pos][Uidx[r]] <= Dmat[pos][Vidx[r]]:
                    items.append([r, 0])
                    pos = Vidx[r]
                else:
                    items.append([r, 1])
                    pos = Uidx[r]
                load += Dem[r]
                unassigned.discard(r)
            if not items:
                # safety guard (only reachable if an edge demand exceeds capacity)
                r = min(unassigned, key=lambda rr: Dem[rr])
                if Dmat[0][Uidx[r]] + Dmat[Vidx[r]][0] <= Dmat[0][Vidx[r]] + Dmat[Uidx[r]][0]:
                    items.append([r, 0])
                else:
                    items.append([r, 1])
                unassigned.discard(r)
            routes.append(make_route(items))
        return routes

    def merge_two(A, B):
        items = [list(it) for it in B['items']]
        for it in A['items']:
            r = it[0]
            dd, posn, oo = best_insertion(items, r)
            items.insert(posn, [r, oo])
        return make_route(items)

    def initial_fallback():
        order = sorted(range(E), key=lambda r: -Dem[r])
        bins = [{'items': [], 'dem': 0, 'cost': 0.0} for _ in range(K)]
        for r in order:
            bt = -1
            bestrem = float('inf')
            for bi, b in enumerate(bins):
                rem = Q - b['dem'] - Dem[r]
                if rem >= 0 and rem < bestrem:
                    bestrem = rem
                    bt = bi
            if bt < 0:
                bt = min(range(K), key=lambda bi: bins[bi]['dem'])
            b = bins[bt]
            dd, posn, oo = best_insertion(b['items'], r)
            b['items'].insert(posn, [r, oo])
            b['dem'] += Dem[r]
            b['cost'] += dd
        # repair any overflow (safety net; no-op on feasible greedy results)
        guard = 0
        while any(b['dem'] > Q for b in bins) and guard < 10000:
            guard += 1
            b = next(b for b in bins if b['dem'] > Q)
            fixed = False
            for it in list(b['items']):
                r = it[0]
                for c in bins:
                    if c is b or c['dem'] + Dem[r] > Q:
                        continue
                    b['items'].remove(it)
                    b['dem'] -= Dem[r]
                    dd, posn, oo = best_insertion(c['items'], r)
                    c['items'].insert(posn, [r, oo])
                    c['dem'] += Dem[r]
                    c['cost'] += dd
                    b['cost'] = route_cost(b['items'])
                    fixed = True
                    break
                if fixed:
                    break
            if not fixed:
                done = False
                for it in list(b['items']):
                    r = it[0]
                    for c in bins:
                        if c is b:
                            continue
                        for it2 in list(c['items']):
                            r2 = it2[0]
                            if Dem[r2] >= Dem[r]:
                                continue
                            if b['dem'] - Dem[r] + Dem[r2] > Q:
                                continue
                            if c['dem'] - Dem[r2] + Dem[r] > Q:
                                continue
                            b['items'].remove(it)
                            b['dem'] -= Dem[r]
                            c['items'].remove(it2)
                            c['dem'] -= Dem[r2]
                            dd, posn, oo = best_insertion(c['items'], r)
                            c['items'].insert(posn, [r, oo])
                            c['dem'] += Dem[r]
                            dd2, posn2, oo2 = best_insertion(b['items'], r2)
                            b['items'].insert(posn2, [r2, oo2])
                            b['dem'] += Dem[r2]
                            b['cost'] = route_cost(b['items'])
                            c['cost'] = route_cost(c['items'])
                            done = True
                            fixed = True
                            break
                        if done:
                            break
                    if done:
                        break
            if not fixed:
                break
        return [b for b in bins if b['items']]

    def enforce_K(routes):
        while len(routes) > K and not expired():
            best = None
            best_cost = float('inf')
            nb = len(routes)
            for a in range(nb - 1):
                Ra = routes[a]
                for b in range(a + 1, nb):
                    Rb = routes[b]
                    if Ra['dem'] + Rb['dem'] > Q:
                        continue
                    M = merge_two(Ra, Rb) if len(Ra['items']) <= len(Rb['items']) else merge_two(Rb, Ra)
                    if M['cost'] < best_cost:
                        best_cost = M['cost']
                        best = (a, b, M)
                if expired():
                    break
            if best is None:
                return initial_fallback()
            a, b, M = best
            routes = [routes[i] for i in range(nb) if i != a and i != b]
            routes.append(M)
        return routes

    # ---------------- local search ----------------
    def relocate(routes):
        for a in range(len(routes)):
            A = routes[a]
            nA = len(A['items'])
            for i in range(nA):
                if expired():
                    return False
                r = A['items'][i][0]
                rd = removal_delta(A['items'], i)
                bestd = float('inf')
                bt = -1
                bp = 0
                bo = 0
                for b in range(len(routes)):
                    if b == a:
                        continue
                    B = routes[b]
                    if B['dem'] + Dem[r] > Q:
                        continue
                    dd, posn, oo = best_insertion(B['items'], r)
                    tot = rd + dd
                    if tot < bestd:
                        bestd = tot
                        bt = b
                        bp = posn
                        bo = oo
                if bt >= 0 and bestd < -1e-9:
                    A['items'].pop(i)
                    A['dem'] -= Dem[r]
                    A['cost'] += rd
                    B = routes[bt]
                    B['items'].insert(bp, [r, bo])
                    B['dem'] += Dem[r]
                    B['cost'] += bestd - rd
                    return True
        return False

    def swap_move(routes):
        for a in range(len(routes)):
            A = routes[a]
            for b in range(a + 1, len(routes)):
                B = routes[b]
                for ia in range(len(A['items'])):
                    if expired():
                        return False
                    r1 = A['items'][ia][0]
                    da = removal_delta(A['items'], ia)
                    for ib in range(len(B['items'])):
                        r2 = B['items'][ib][0]
                        if A['dem'] - Dem[r1] + Dem[r2] > Q:
                            continue
                        if B['dem'] - Dem[r2] + Dem[r1] > Q:
                            continue
                        db = removal_delta(B['items'], ib)
                        Arem = A['items'][:ia] + A['items'][ia + 1:]
                        Brem = B['items'][:ib] + B['items'][ib + 1:]
                        d1, p1, o1 = best_insertion(Arem, r2)
                        d2, p2, o2 = best_insertion(Brem, r1)
                        tot = da + db + d1 + d2
                        if tot < -1e-9:
                            Arem.insert(p1, [r2, o1])
                            Brem.insert(p2, [r1, o2])
                            A['items'] = Arem
                            A['dem'] = A['dem'] - Dem[r1] + Dem[r2]
                            A['cost'] += tot
                            B['items'] = Brem
                            B['dem'] = B['dem'] - Dem[r2] + Dem[r1]
                            return True
        return False

    def reposition(R):
        items = R['items']
        for i in range(len(items)):
            r = items[i][0]
            rd = removal_delta(items, i)
            rem = items[:i] + items[i + 1:]
            dd, posn, oo = best_insertion(rem, r)
            if rd + dd < -1e-9:
                rem.insert(posn, [r, oo])
                R['items'] = rem
                R['cost'] += rd + dd
                return True
        return False

    def flip(R):
        items = R['items']
        m = len(items)
        changed = False
        for i in range(m):
            r, o = items[i]
            prev_exit = 0 if i == 0 else item_exit(items[i - 1])
            next_entry = 0 if i == m - 1 else item_entry(items[i + 1])
            ent = Uidx[r] if o == 0 else Vidx[r]
            ext = Vidx[r] if o == 0 else Uidx[r]
            cur = Dmat[prev_exit][ent] + Dmat[ext][next_entry]
            alt = Dmat[prev_exit][ext] + Dmat[ent][next_entry]
            if alt < cur - 1e-9:
                items[i][1] = 1 - o
                R['cost'] += alt - cur
                changed = True
        return changed

    def prune(routes):
        routes[:] = [R for R in routes if R['items']]

    def local_search(routes):
        improved = True
        while improved:
            improved = False
            if relocate(routes):
                improved = True
                prune(routes)
                continue
            if swap_move(routes):
                improved = True
                continue
            any_imp = False
            for R in routes:
                if expired():
                    break
                if reposition(R) or flip(R):
                    any_imp = True
            if any_imp:
                improved = True
        for R in routes:
            R['cost'] = route_cost(R['items'])
            R['dem'] = route_dem(R['items'])
        prune(routes)
        return routes

    # ---------------- exact route optimization (DP over subsets) ----------------
    def solve_exact(rs):
        k = len(rs)
        if k == 1:
            r = rs[0]
            c0 = Dmat[0][Uidx[r]] + Dmat[Vidx[r]][0]
            c1 = Dmat[0][Vidx[r]] + Dmat[Uidx[r]][0]
            if c0 <= c1:
                return [[r, 0]], c0
            return [[r, 1]], c1
        N = 1 << k
        en2 = []
        ex2 = []
        for r in rs:
            en2.append(Uidx[r])
            en2.append(Vidx[r])
            ex2.append(Vidx[r])
            ex2.append(Uidx[r])
        kk2 = 2 * k
        C = [[0.0] * kk2 for _ in range(kk2)]
        tail = [0.0] * kk2
        for a in range(kk2):
            rowa = Dmat[ex2[a]]
            for b in range(kk2):
                C[a][b] = rowa[en2[b]]
            tail[a] = rowa[0]
        INF = float('inf')
        size = N * kk2
        dp = [INF] * size
        par = [-1] * size
        stride = kk2
        for i in range(k):
            m = (1 << i)
            dp[m * stride + 2 * i] = Dmat[0][en2[2 * i]]
            dp[m * stride + 2 * i + 1] = Dmat[0][en2[2 * i + 1]]
        for mask in range(1, N):
            base_off = mask * stride
            m = mask
            while m:
                lb = m & -m
                i = lb.bit_length() - 1
                m ^= lb
                io0 = 2 * i
                for o in (0, 1):
                    a = io0 + o
                    ci = dp[base_off + a]
                    if ci == INF:
                        continue
                    Ca = C[a]
                    rem = (~mask) & (N - 1)
                    mm = rem
                    while mm:
                        lb2 = mm & -mm
                        j = lb2.bit_length() - 1
                        mm ^= lb2
                        noff = ((mask | lb2) * stride) + 2 * j
                        v0 = ci + Ca[2 * j]
                        if v0 < dp[noff]:
                            dp[noff] = v0
                            par[noff] = base_off + a
                        v1 = ci + Ca[2 * j + 1]
                        if v1 < dp[noff + 1]:
                            dp[noff + 1] = v1
                            par[noff + 1] = base_off + a
        full = (N - 1) * stride
        best = INF
        bi = -1
        for a in range(kk2):
            v = dp[full + a] + tail[a]
            if v < best:
                best = v
                bi = full + a
        order = []
        idx = bi
        while idx >= 0:
            a = idx % stride
            i = a >> 1
            o = a & 1
            order.append([rs[i], o])
            idx = par[idx]
        order.reverse()
        return order, best

    def polish(routes, maxk):
        for R in routes:
            kk = len(R['items'])
            if 2 <= kk <= maxk and not expired():
                est = (1 << kk) * kk * kk
                if est > 2e6 and time_left() < est / 2_000_000 + 0.2:
                    continue
                rs = [it[0] for it in R['items']]
                items, c = solve_exact(rs)
                if c < R['cost'] - 1e-9:
                    R['items'] = items
                    R['cost'] = c
        return routes

    def polish_merge(routes, maxk):
        while not expired():
            nb = len(routes)
            done = True
            for a in range(nb - 1):
                if expired():
                    return routes
                A = routes[a]
                for b in range(a + 1, nb):
                    B = routes[b]
                    if A['dem'] + B['dem'] > Q:
                        continue
                    if len(A['items']) + len(B['items']) > maxk:
                        continue
                    rs = [it[0] for it in A['items']] + [it[0] for it in B['items']]
                    items, c = solve_exact(rs)
                    if c < A['cost'] + B['cost'] - 1e-9:
                        A['items'] = items
                        A['dem'] = A['dem'] + B['dem']
                        A['cost'] = c
                        del routes[b]
                        done = False
                        break
                if not done:
                    break
            if done:
                return routes
        return routes

    # ---------------- perturbation ----------------
    def copy_routes(routes):
        return [{'items': [list(it) for it in R['items']],
                 'dem': R['dem'], 'cost': R['cost']} for R in routes]

    def perturb(routes):
        rs = copy_routes(routes)
        total_items = sum(len(R['items']) for R in rs)
        if total_items == 0:
            return rs
        p = random.randint(1, max(2, total_items // 3))
        for _ in range(p):
            nonempty = [i for i, R in enumerate(rs) if R['items']]
            if not nonempty:
                break
            a = random.choice(nonempty)
            A = rs[a]
            i = random.randrange(len(A['items']))
            r = A['items'][i][0]
            b = random.randrange(len(rs))
            B = rs[b]
            if b != a and B['dem'] + Dem[r] > Q:
                continue
            A['items'].pop(i)
            posn = random.randrange(len(B['items']) + 1)
            oo = random.randint(0, 1)
            B['items'].insert(posn, [r, oo])
            if b != a:
                A['dem'] -= Dem[r]
                B['dem'] += Dem[r]
        for R in rs:
            R['items'] = R['items']
            R['dem'] = route_dem(R['items'])
            R['cost'] = route_cost(R['items'])
        prune(rs)
        return rs

    # ---------------- output building ----------------
    def full_cost(routes):
        return svc_const + sum(R['cost'] for R in routes)

    def deadhead_counter_route(R):
        dead = Counter()
        pos = 0
        items = R['items']
        for r, o in items:
            entry = Uidx[r] if o == 0 else Vidx[r]
            if entry != pos:
                pn = parN[pos]
                pe = parE[pos]
                snode = DN[pos]
                x = DN[entry]
                while x != snode and x >= 0:
                    e = pe[x]
                    if e < 0:
                        break
                    dead[e] += 1
                    x = pn[x]
            pos = Vidx[r] if o == 0 else Uidx[r]
        if pos != 0:
            pn = parN[pos]
            pe = parE[pos]
            snode = DN[pos]
            x = DN[0]
            while x != snode and x >= 0:
                e = pe[x]
                if e < 0:
                    break
                dead[e] += 1
                x = pn[x]
        return dead

    def build_solution_dict(routes, obj):
        out_routes = []
        vi = 0
        for R in routes:
            if not R['items']:
                continue
            dead = deadhead_counter_route(R)
            ddl = [{'edge_id': eid, 'times': t} for eid, t in sorted(dead.items())]
            out_routes.append({
                'vehicle': vi,
                'serviced_edges': [RequestId[r] for r, _ in R['items']],
                'deadheaded_edges': ddl,
            })
            vi += 1
        return {'objective_value': float(obj), 'routes': out_routes}

    def log_best(best_obj, best_routes):
        if logger:
            try:
                logger.log_solution(best_obj, build_solution_dict(best_routes, best_obj))
            except Exception:
                pass

    # ---------------- main pipeline ----------------
    routes = construct(False)
    if len(routes) > K:
        routes = enforce_K(routes)
    if len(routes) > K:
        routes = initial_fallback()
    routes = local_search(routes)
    if not expired():
        polish(routes, 12)
    for R in routes:
        R['cost'] = route_cost(R['items'])
    best_routes = routes
    best_obj = full_cost(best_routes)
    log_best(best_obj, best_routes)

    it_no = 0
    while not expired():
        it_no += 1
        if it_no % 3 == 1 and best_routes:
            cand = perturb(best_routes)
        else:
            cand = construct(True)
            if len(cand) > K:
                cand = enforce_K(cand)
            if len(cand) > K:
                cand = initial_fallback()
        cand = local_search(cand)
        cobj = full_cost(cand)
        if cobj < best_obj * 1.03 and not expired():
            polish(cand, 12)
            for R in cand:
                R['cost'] = route_cost(R['items'])
            cobj = full_cost(cand)
        if cobj < best_obj - 1e-9:
            best_obj = cobj
            best_routes = cand
            log_best(best_obj, best_routes)

    if not expired():
        polish(best_routes, 15)
    if not expired():
        polish_merge(best_routes, 12)
    for R in best_routes:
        R['cost'] = route_cost(R['items'])
    best_obj = full_cost(best_routes)

    out = build_solution_dict(best_routes, best_obj)
    if logger:
        try:
            logger.log_solution(best_obj, out)
        except Exception:
            pass
    with open(args.solution_path, 'w') as f:
        json.dump(out, f)


if __name__ == '__main__':
    main()
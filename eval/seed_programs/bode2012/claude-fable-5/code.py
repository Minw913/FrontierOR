import argparse
import json
import heapq
import random
import time
from collections import defaultdict, Counter

INF = float('inf')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--instance_path', required=True)
    ap.add_argument('--solution_path', required=True)
    ap.add_argument('--time_limit', type=int, default=60)
    ap.add_argument('--log_path', default=None)
    args = ap.parse_args()

    t0 = time.time()
    deadline = t0 + max(1, args.time_limit) - 1.0

    try:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    except Exception:
        logger = None

    random.seed(0)
    rng = random.Random(0)

    with open(args.instance_path) as f:
        data = json.load(f)

    edges = data['edges']
    depot = data['depot']
    K = int(data['fleet']['num_vehicles'])
    cap = data['fleet']['vehicle_capacity']

    adj = defaultdict(list)
    ecost = {}
    for e in edges:
        u, v = e['endpoints']
        adj[u].append((v, e['cost'], e['edge_id']))
        adj[v].append((u, e['cost'], e['edge_id']))
        ecost[e['edge_id']] = e['cost']

    req = [e for e in edges if e['is_required']]
    n = len(req)
    total_service = sum(e.get('service_cost', 0) for e in req)

    # -------- trivial case --------
    if n == 0:
        sol = {"objective_value": 0.0,
               "routes": [{"vehicle": v, "serviced_edges": [], "deadheaded_edges": []}
                          for v in range(K)]}
        if logger:
            logger.log_solution(0.0, sol)
        with open(args.solution_path, 'w') as f:
            json.dump(sol, f)
        return

    teid = [e['edge_id'] for e in req]
    tu = [e['endpoints'][0] for e in req]
    tv = [e['endpoints'][1] for e in req]
    tdem = [e['demand'] for e in req]
    tsc = [e.get('service_cost', 0) for e in req]

    # -------- shortest paths from important nodes --------
    important = {depot}
    for i in range(n):
        important.add(tu[i]); important.add(tv[i])

    Ddist = {}
    Dpar = {}
    for s in important:
        dist = {s: 0}
        par = {}
        pq = [(0, s)]
        while pq:
            dd, u = heapq.heappop(pq)
            if dd > dist.get(u, INF):
                continue
            for v, c, eid in adj[u]:
                nd = dd + c
                if nd < dist.get(v, INF):
                    dist[v] = nd
                    par[v] = (u, eid)
                    heapq.heappush(pq, (nd, v))
        Ddist[s] = dist
        Dpar[s] = par

    def d(a, b):
        if a == b:
            return 0
        return Ddist[a].get(b, INF)

    def path_edges(a, b):
        if a == b:
            return []
        res = []
        par = Dpar[a]
        cur = b
        while cur != a:
            u, eid = par[cur]
            res.append(eid)
            cur = u
        return res

    def route_dh(r):
        c = 0
        cur = depot
        for (i, s, e) in r:
            c += d(cur, s)
            cur = e
        return c + d(cur, depot)

    def total_dh(routes):
        return sum(route_dh(r) for r in routes)

    # -------- path scanning construction --------
    def path_scan(rule):
        unserved = set(range(n))
        routes = []
        while unserved:
            r = []
            load = 0
            cur = depot
            while True:
                cand = []
                bd = INF
                for i in unserved:
                    if load + tdem[i] > cap:
                        continue
                    for s, e in ((tu[i], tv[i]), (tv[i], tu[i])):
                        dd = d(cur, s)
                        if dd < bd:
                            bd = dd
                            cand = [(i, s, e)]
                        elif dd == bd:
                            cand.append((i, s, e))
                if not cand:
                    break
                if len(cand) == 1:
                    pick = cand[0]
                elif rule == 5:
                    pick = rng.choice(cand)
                else:
                    rr = rule
                    if rule == 4:
                        rr = 0 if load < cap / 2.0 else 1
                    if rr == 0:
                        pick = max(cand, key=lambda t: d(t[2], depot))
                    elif rr == 1:
                        pick = min(cand, key=lambda t: d(t[2], depot))
                    elif rr == 2:
                        pick = max(cand, key=lambda t: tdem[t[0]] / max(tsc[t[0]], 1e-9))
                    else:
                        pick = min(cand, key=lambda t: tdem[t[0]] / max(tsc[t[0]], 1e-9))
                i, s, e = pick
                r.append(pick)
                load += tdem[i]
                cur = e
                unserved.discard(i)
            routes.append(r)
        return routes

    def merge_to_K(routes):
        routes = [list(r) for r in routes if r]
        loads = [sum(tdem[t[0]] for t in r) for r in routes]
        while len(routes) > K:
            best = None
            L = len(routes)
            for a in range(L):
                for b in range(L):
                    if a == b or loads[a] + loads[b] > cap:
                        continue
                    delta = route_dh(routes[a] + routes[b]) - route_dh(routes[a]) - route_dh(routes[b])
                    if best is None or delta < best[0]:
                        best = (delta, a, b)
            if best is None:
                return None
            _, a, b = best
            routes[a] = routes[a] + routes[b]
            loads[a] += loads[b]
            del routes[b]
            del loads[b]
        return routes

    def binpack_build():
        order = sorted(range(n), key=lambda i: -tdem[i])
        routes = [[] for _ in range(K)]
        loads = [0] * K
        for i in order:
            best = None
            for ri in range(K):
                if loads[ri] + tdem[i] > cap:
                    continue
                r = routes[ri]
                for q in range(len(r) + 1):
                    prev = r[q - 1][2] if q > 0 else depot
                    nxt = r[q][1] if q < len(r) else depot
                    for s, e in ((tu[i], tv[i]), (tv[i], tu[i])):
                        delta = d(prev, s) + d(e, nxt) - d(prev, nxt)
                        if best is None or delta < best[0]:
                            best = (delta, ri, q, s, e)
            if best is None:
                return None
            _, ri, q, s, e = best
            routes[ri].insert(q, (i, s, e))
            loads[ri] += tdem[i]
        return routes

    # -------- local search --------
    def local_search(routes, loads, deadline):
        Ks = len(routes)
        improved = True
        while improved:
            if time.time() > deadline:
                return
            improved = False
            # 2-opt intra (includes single-task orientation flip)
            for r in routes:
                m = len(r)
                changed = True
                while changed:
                    changed = False
                    for i in range(m):
                        prev = r[i - 1][2] if i > 0 else depot
                        si = r[i][1]
                        for j in range(i, m):
                            nxt = r[j + 1][1] if j + 1 < m else depot
                            ej = r[j][2]
                            delta = d(prev, ej) + d(si, nxt) - d(prev, si) - d(ej, nxt)
                            if delta < -1e-9:
                                r[i:j + 1] = [(t[0], t[2], t[1]) for t in reversed(r[i:j + 1])]
                                changed = True
                                improved = True
                                prev = r[i - 1][2] if i > 0 else depot
                                si = r[i][1]
                if time.time() > deadline:
                    return
            # relocate
            for ra in range(Ks):
                if time.time() > deadline:
                    return
                p = 0
                while p < len(routes[ra]):
                    r = routes[ra]
                    i, s, e = r[p]
                    prev = r[p - 1][2] if p > 0 else depot
                    nxt = r[p + 1][1] if p + 1 < len(r) else depot
                    rm = d(prev, nxt) - d(prev, s) - d(e, nxt)
                    bestdelta = -1e-9
                    bestmv = None
                    for rb in range(Ks):
                        if rb == ra:
                            tmp = r[:p] + r[p + 1:]
                            lt = len(tmp)
                            for qq in range(lt + 1):
                                prevb = tmp[qq - 1][2] if qq > 0 else depot
                                nxtb = tmp[qq][1] if qq < lt else depot
                                base = d(prevb, nxtb)
                                for s2, e2 in ((tu[i], tv[i]), (tv[i], tu[i])):
                                    delta = rm + d(prevb, s2) + d(e2, nxtb) - base
                                    if delta < bestdelta:
                                        bestdelta = delta
                                        bestmv = (rb, qq, s2, e2)
                        else:
                            if loads[rb] + tdem[i] > cap:
                                continue
                            r2 = routes[rb]
                            l2 = len(r2)
                            for q in range(l2 + 1):
                                prevb = r2[q - 1][2] if q > 0 else depot
                                nxtb = r2[q][1] if q < l2 else depot
                                base = d(prevb, nxtb)
                                for s2, e2 in ((tu[i], tv[i]), (tv[i], tu[i])):
                                    delta = rm + d(prevb, s2) + d(e2, nxtb) - base
                                    if delta < bestdelta:
                                        bestdelta = delta
                                        bestmv = (rb, q, s2, e2)
                    if bestmv is not None:
                        rb, q, s2, e2 = bestmv
                        if rb == ra:
                            tmp = r[:p] + r[p + 1:]
                            routes[ra] = tmp[:q] + [(i, s2, e2)] + tmp[q:]
                        else:
                            routes[ra].pop(p)
                            routes[rb].insert(q, (i, s2, e2))
                            loads[ra] -= tdem[i]
                            loads[rb] += tdem[i]
                        improved = True
                    else:
                        p += 1
            if time.time() > deadline:
                return
            # swap between routes
            for ra in range(Ks):
                for rb in range(ra + 1, Ks):
                    r1 = routes[ra]
                    r2 = routes[rb]
                    p = 0
                    while p < len(r1):
                        i1, s1, e1 = r1[p]
                        preva = r1[p - 1][2] if p > 0 else depot
                        nxta = r1[p + 1][1] if p + 1 < len(r1) else depot
                        olda = d(preva, s1) + d(e1, nxta)
                        applied = False
                        q = 0
                        while q < len(r2):
                            i2, s2, e2 = r2[q]
                            if (loads[ra] - tdem[i1] + tdem[i2] > cap or
                                    loads[rb] - tdem[i2] + tdem[i1] > cap):
                                q += 1
                                continue
                            prevb = r2[q - 1][2] if q > 0 else depot
                            nxtb = r2[q + 1][1] if q + 1 < len(r2) else depot
                            oldb = d(prevb, s2) + d(e2, nxtb)
                            na = None
                            for ss, ee in ((tu[i2], tv[i2]), (tv[i2], tu[i2])):
                                c = d(preva, ss) + d(ee, nxta)
                                if na is None or c < na[0]:
                                    na = (c, ss, ee)
                            nb = None
                            for ss, ee in ((tu[i1], tv[i1]), (tv[i1], tu[i1])):
                                c = d(prevb, ss) + d(ee, nxtb)
                                if nb is None or c < nb[0]:
                                    nb = (c, ss, ee)
                            delta = na[0] + nb[0] - olda - oldb
                            if delta < -1e-9:
                                r1[p] = (i2, na[1], na[2])
                                r2[q] = (i1, nb[1], nb[2])
                                loads[ra] += tdem[i2] - tdem[i1]
                                loads[rb] += tdem[i1] - tdem[i2]
                                improved = True
                                applied = True
                                break
                            q += 1
                        if not applied:
                            p += 1
                    if time.time() > deadline:
                        return

    def perturb(routes, loads, strength):
        all_pos = [(ra, p) for ra in range(len(routes)) for p in range(len(routes[ra]))]
        rng.shuffle(all_pos)
        chosen = sorted(all_pos[:strength], key=lambda x: (x[0], -x[1]))
        removed = []
        for ra, p in chosen:
            i, s, e = routes[ra].pop(p)
            loads[ra] -= tdem[i]
            removed.append(i)
        rng.shuffle(removed)
        for i in removed:
            best = None
            for rb in range(len(routes)):
                if loads[rb] + tdem[i] > cap:
                    continue
                r2 = routes[rb]
                for q in range(len(r2) + 1):
                    prevb = r2[q - 1][2] if q > 0 else depot
                    nxtb = r2[q][1] if q < len(r2) else depot
                    base = d(prevb, nxtb)
                    for s2, e2 in ((tu[i], tv[i]), (tv[i], tu[i])):
                        delta = d(prevb, s2) + d(e2, nxtb) - base
                        if best is None or delta < best[0]:
                            best = (delta, rb, q, s2, e2)
            if best is None:
                rb = min(range(len(routes)), key=lambda x: loads[x])
                best = (0, rb, len(routes[rb]), tu[i], tv[i])
            _, rb, q, s2, e2 = best
            routes[rb].insert(q, (i, s2, e2))
            loads[rb] += tdem[i]

    def build_solution(routes, num_out):
        out_routes = []
        total = float(total_service)
        for vidx in range(num_out):
            r = routes[vidx] if vidx < len(routes) else []
            serviced = [teid[t[0]] for t in r]
            cnt = Counter()
            cur = depot
            for (i, s, e) in r:
                for eid in path_edges(cur, s):
                    cnt[eid] += 1
                cur = e
            for eid in path_edges(cur, depot):
                cnt[eid] += 1
            dh = [{"edge_id": k, "times": v} for k, v in sorted(cnt.items())]
            total += sum(ecost[k] * v for k, v in cnt.items())
            out_routes.append({"vehicle": vidx, "serviced_edges": serviced,
                               "deadheaded_edges": dh})
        return {"objective_value": float(total), "routes": out_routes}

    # -------- build initial solution --------
    best_init = None
    best_init_cost = INF
    for rule in [0, 1, 2, 3, 4, 5, 5, 5]:
        if time.time() > deadline:
            break
        rs = path_scan(rule)
        rs2 = merge_to_K(rs)
        if rs2 is None:
            continue
        c = total_dh(rs2)
        if c < best_init_cost:
            best_init_cost = c
            best_init = rs2
    if best_init is None:
        best_init = binpack_build()
    if best_init is None:
        best_init = path_scan(0)  # last resort (may exceed K vehicles)

    routes = [list(r) for r in best_init]
    while len(routes) < K:
        routes.append([])
    Kslots = len(routes)
    loads = [sum(tdem[t[0]] for t in r) for r in routes]

    local_search(routes, loads, deadline)
    best_routes = [list(r) for r in routes]
    best_cost = total_dh(best_routes)
    num_out = max(K, Kslots)
    best_sol = build_solution(best_routes, num_out)
    if logger:
        logger.log_solution(best_sol["objective_value"], best_sol)

    # -------- iterated local search --------
    try:
        while time.time() < deadline:
            routes = [list(r) for r in best_routes]
            loads = [sum(tdem[t[0]] for t in r) for r in routes]
            strength = rng.randint(2, max(2, min(8, n)))
            perturb(routes, loads, strength)
            local_search(routes, loads, deadline)
            # verify capacity feasibility
            feasible = all(sum(tdem[t[0]] for t in r) <= cap for r in routes)
            if not feasible:
                continue
            c = total_dh(routes)
            if c < best_cost - 1e-9:
                best_cost = c
                best_routes = [list(r) for r in routes]
                best_sol = build_solution(best_routes, num_out)
                if logger:
                    logger.log_solution(best_sol["objective_value"], best_sol)
    finally:
        with open(args.solution_path, 'w') as f:
            json.dump(best_sol, f)


if __name__ == '__main__':
    main()
import argparse
import json
import random
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path) as f:
        inst = json.load(f)

    t0 = time.time()
    deadline = t0 + max(0.2, args.time_limit - 0.5)

    D = inst['distance_matrix']
    N = int(inst['num_total_customers'])
    K = max(1, int(inst['num_vehicles']))
    Q = float(inst['vehicle_capacity'])

    demand = [0.0] * (N + 1)
    isL = [False] * (N + 1)
    linehauls = []
    backhauls = []
    for c in inst.get('linehaul_customers', []):
        i = int(c['index'])
        demand[i] = float(c['demand'])
        isL[i] = True
        linehauls.append(i)
    for c in inst.get('backhaul_customers', []):
        i = int(c['index'])
        demand[i] = float(c['demand'])
        isL[i] = False
        backhauls.append(i)
    L = len(linehauls)

    rng = random.Random(0)
    EPS = 1e-9

    def rcost(r):
        if not r:
            return 0.0
        c = D[0][r[0]]
        for i in range(len(r) - 1):
            c += D[r[i]][r[i + 1]]
        c += D[r[-1]][0]
        return float(c)

    def time_up():
        return time.time() >= deadline

    # ---------------- state ----------------
    routes = [[] for _ in range(K)]
    nL = [0] * K
    ld = [0.0] * K
    bd = [0.0] * K
    rc = [0.0] * K

    def recalc(k):
        r = routes[k]
        n = 0
        sl = 0.0
        sb = 0.0
        for u in r:
            if isL[u]:
                n += 1
                sl += demand[u]
            else:
                sb += demand[u]
        nL[k] = n
        ld[k] = sl
        bd[k] = sb
        rc[k] = rcost(r)

    def recalc_all():
        for k in range(K):
            recalc(k)

    def total():
        return sum(rc)

    def solution_dict(rts, obj):
        return {
            "objective_value": float(obj),
            "routes": [[0] + [int(x) for x in r] + [0] for r in rts],
        }

    def log_incumbent(obj, rts):
        if logger:
            logger.log_solution(float(obj), solution_dict(rts, obj))

    # ---------------- construction (seeded regret insertion) ----------------
    def seed_order():
        if not linehauls:
            return []
        m = min(K, L)
        first = max(linehauls, key=lambda u: D[0][u])
        order = [first]
        inchosen = {first}
        while len(order) < m:
            best = None
            bestd = -1
            for u in linehauls:
                if u in inchosen:
                    continue
                md = min(D[u][v] for v in order)
                if md > bestd:
                    bestd = md
                    best = u
            order.append(best)
            inchosen.add(best)
        return order

    def feasible_construct(seeds, relax=False):
        routes[:] = [[] for _ in range(K)]
        nL_ = [0] * K
        ld_ = [0.0] * K
        bd_ = [0.0] * K
        assigned = set()
        for k, u in enumerate(seeds[:min(K, L)]):
            routes[k].append(u)
            assigned.add(u)
            nL_[k] += 1
            ld_[k] += demand[u]
        unassigned = [u for u in linehauls if u not in assigned] + list(backhauls)
        while unassigned:
            best_metric = None
            chosen = None
            for u in unassigned:
                du = demand[u]
                uL = isL[u]
                c1 = None
                c2 = None
                for k in range(K):
                    r = routes[k]
                    if uL:
                        if not relax and ld_[k] + du > Q + EPS:
                            continue
                        prange = range(0, nL_[k] + 1)
                    else:
                        if not relax and (nL_[k] == 0 or bd_[k] + du > Q + EPS):
                            continue
                        prange = range(nL_[k], len(r) + 1)
                    for p in prange:
                        a = r[p - 1] if p > 0 else 0
                        b = r[p] if p < len(r) else 0
                        inc = D[a][u] + D[u][b] - D[a][b]
                        if c1 is None or inc < c1[0]:
                            c2 = c1
                            c1 = (inc, k, p)
                        elif c2 is None or inc < c2[0]:
                            c2 = (inc, k, p)
                if c1 is None:
                    return False
                regret = (c2[0] - c1[0]) if c2 else 1e18
                metric = (regret, -c1[0])
                if best_metric is None or metric > best_metric:
                    best_metric = metric
                    chosen = (u, c1[1], c1[2])
            u, k, p = chosen
            routes[k].insert(p, u)
            if isL[u]:
                nL_[k] += 1
                ld_[k] += demand[u]
            else:
                bd_[k] += demand[u]
            unassigned.remove(u)
        recalc_all()
        return True

    # ---------------- local search (best improvement) ----------------
    def apply_move(mv):
        t = mv[0]
        if t == 'rel':
            _, A, i, B, p = mv
            u = routes[A].pop(i)
            routes[B].insert(p, u)
            recalc(A)
            recalc(B)
        elif t == 'swp':
            _, A, i, B, j = mv
            routes[A][i], routes[B][j] = routes[B][j], routes[A][i]
            recalc(A)
            recalc(B)
        elif t == 'wrel':
            _, A, i, p = mv
            u = routes[A].pop(i)
            routes[A].insert(p, u)
            recalc(A)
        elif t == '2opt':
            _, A, i, j = mv
            routes[A] = routes[A][:i] + routes[A][i:j + 1][::-1] + routes[A][j + 1:]
            recalc(A)
        elif t == '2opt*':
            _, A, i, B, j = mv
            newA = routes[A][:i] + routes[B][j:]
            newB = routes[B][:j] + routes[A][i:]
            routes[A] = newA
            routes[B] = newB
            recalc(A)
            recalc(B)

    def local_search():
        while not time_up():
            best_delta = -1e-6
            best_move = None

            # 1) cross-route relocate
            for A in range(K):
                if time_up():
                    return
                rA = routes[A]
                lenA = len(rA)
                if lenA == 0:
                    continue
                for i in range(lenA):
                    u = rA[i]
                    uL = isL[u]
                    du = demand[u]
                    if uL and nL[A] == 1 and lenA > 1:
                        continue  # would leave a backhaul-only route
                    a = rA[i - 1] if i > 0 else 0
                    b = rA[i + 1] if i < lenA - 1 else 0
                    rem = D[a][b] - D[a][u] - D[u][b]
                    for B in range(K):
                        if B == A:
                            continue
                        rB = routes[B]
                        if uL:
                            if ld[B] + du > Q + EPS:
                                continue
                            p0, p1 = 0, nL[B]
                        else:
                            if nL[B] == 0 or bd[B] + du > Q + EPS:
                                continue
                            p0, p1 = nL[B], len(rB)
                        for p in range(p0, p1 + 1):
                            x = rB[p - 1] if p > 0 else 0
                            y = rB[p] if p < len(rB) else 0
                            delta = rem + D[x][u] + D[u][y] - D[x][y]
                            if delta < best_delta:
                                best_delta = delta
                                best_move = ('rel', A, i, B, p)

            # 2) cross-route swap (same type keeps positions valid)
            for A in range(K):
                if time_up():
                    return
                rA = routes[A]
                if not rA:
                    continue
                for B in range(A + 1, K):
                    rB = routes[B]
                    if not rB:
                        continue
                    for i in range(len(rA)):
                        u = rA[i]
                        uL = isL[u]
                        du = demand[u]
                        a = rA[i - 1] if i > 0 else 0
                        b = rA[i + 1] if i < len(rA) - 1 else 0
                        for j in range(len(rB)):
                            v = rB[j]
                            if isL[v] != uL:
                                continue
                            dv = demand[v]
                            if uL:
                                if ld[A] - du + dv > Q + EPS or ld[B] - dv + du > Q + EPS:
                                    continue
                            else:
                                if bd[A] - du + dv > Q + EPS or bd[B] - dv + du > Q + EPS:
                                    continue
                            x = rB[j - 1] if j > 0 else 0
                            y = rB[j + 1] if j < len(rB) - 1 else 0
                            delta = (D[a][v] + D[v][b] - D[a][u] - D[u][b]) + \
                                    (D[x][u] + D[u][y] - D[x][v] - D[v][y])
                            if delta < best_delta:
                                best_delta = delta
                                best_move = ('swp', A, i, B, j)

            # 3) within-route relocate
            for A in range(K):
                if time_up():
                    return
                rA = routes[A]
                m = len(rA)
                if m < 2:
                    continue
                nLa = nL[A]
                for i in range(m):
                    u = rA[i]
                    uL = isL[u]
                    a = rA[i - 1] if i > 0 else 0
                    b = rA[i + 1] if i < m - 1 else 0
                    rem = D[a][b] - D[a][u] - D[u][b]
                    rest = rA[:i] + rA[i + 1:]
                    if uL:
                        p0, p1 = 0, nLa - 1
                    else:
                        p0, p1 = nLa, m - 1
                    for p in range(p0, p1 + 1):
                        x = rest[p - 1] if p > 0 else 0
                        y = rest[p] if p < m - 1 else 0
                        delta = rem + D[x][u] + D[u][y] - D[x][y]
                        if delta < best_delta:
                            best_delta = delta
                            best_move = ('wrel', A, i, p)

            # 4) within-route 2-opt on same-type segments
            for A in range(K):
                if time_up():
                    return
                rA = routes[A]
                m = len(rA)
                if m < 2:
                    continue
                nLa = nL[A]
                F = [0.0] * m
                G = [0.0] * m
                for t in range(m - 1):
                    F[t + 1] = F[t] + D[rA[t]][rA[t + 1]]
                    G[t + 1] = G[t] + D[rA[t + 1]][rA[t]]
                for i in range(m - 1):
                    ri = rA[i]
                    prev = rA[i - 1] if i > 0 else 0
                    for j in range(i + 1, m):
                        if i < nLa and j >= nLa:
                            continue  # mixed segment reversal would break ordering
                        rj = rA[j]
                        nxt = rA[j + 1] if j < m - 1 else 0
                        delta = (D[prev][rj] - D[prev][ri]) + (D[ri][nxt] - D[rj][nxt]) + \
                                (G[j] - G[i]) - (F[j] - F[i])
                        if delta < best_delta:
                            best_delta = delta
                            best_move = ('2opt', A, i, j)

            # 5) cross-route 2-opt* (tail exchange)
            for A in range(K):
                if time_up():
                    return
                for B in range(A + 1, K):
                    rA = routes[A]
                    rB = routes[B]
                    lenA = len(rA)
                    lenB = len(rB)
                    if lenA == 0 and lenB == 0:
                        continue
                    headA = [0.0] * (lenA + 1)
                    tailA = [0.0] * (lenA + 1)
                    cumLA = [0.0] * (lenA + 1)
                    cumBA = [0.0] * (lenA + 1)
                    for t in range(lenA):
                        u = rA[t]
                        headA[t + 1] = headA[t] + (D[0][u] if t == 0 else D[rA[t - 1]][u])
                        cumLA[t + 1] = cumLA[t]
                        cumBA[t + 1] = cumBA[t]
                        if isL[u]:
                            cumLA[t + 1] += demand[u]
                        else:
                            cumBA[t + 1] += demand[u]
                    for t in range(lenA - 1, -1, -1):
                        tailA[t] = tailA[t + 1] + (D[rA[t]][0] if t == lenA - 1 else D[rA[t]][rA[t + 1]])
                    headB = [0.0] * (lenB + 1)
                    tailB = [0.0] * (lenB + 1)
                    cumLB = [0.0] * (lenB + 1)
                    cumBB = [0.0] * (lenB + 1)
                    for t in range(lenB):
                        u = rB[t]
                        headB[t + 1] = headB[t] + (D[0][u] if t == 0 else D[rB[t - 1]][u])
                        cumLB[t + 1] = cumLB[t]
                        cumBB[t + 1] = cumBB[t]
                        if isL[u]:
                            cumLB[t + 1] += demand[u]
                        else:
                            cumBB[t + 1] += demand[u]
                    for t in range(lenB - 1, -1, -1):
                        tailB[t] = tailB[t + 1] + (D[rB[t]][0] if t == lenB - 1 else D[rB[t]][rB[t + 1]])
                    base = rc[A] + rc[B]
                    nLa = nL[A]
                    nLb = nL[B]
                    ldA = ld[A]
                    bdA = bd[A]
                    ldB = ld[B]
                    bdB = bd[B]
                    for i in range(lenA + 1):
                        plA_i = i if i < nLa else nLa
                        pbA_i = i - plA_i
                        tlA_i = nLa - plA_i  # linehauls in A[i:]
                        for j in range(lenB + 1):
                            if (i == 0 and j == 0) or (i == lenA and j == lenB):
                                continue
                            plB_j = j if j < nLb else nLb
                            pbB_j = j - plB_j
                            tlB_j = nLb - plB_j
                            # newA = A[:i] + B[j:]
                            naLen = i + (lenB - j)
                            naL = plA_i + tlB_j
                            if naLen > 0 and naL == 0:
                                continue
                            if pbA_i > 0 and tlB_j > 0:
                                continue
                            nald = cumLA[i] + (ldB - cumLB[j])
                            nabd = cumBA[i] + (bdB - cumBB[j])
                            if nald > Q + EPS or nabd > Q + EPS:
                                continue
                            # newB = B[:j] + A[i:]
                            nbLen = j + (lenA - i)
                            nbL = plB_j + tlA_i
                            if nbLen > 0 and nbL == 0:
                                continue
                            if pbB_j > 0 and tlA_i > 0:
                                continue
                            nbld = cumLB[j] + (ldA - cumLA[i])
                            nbbd = cumBB[j] + (bdA - cumBA[i])
                            if nbld > Q + EPS or nbbd > Q + EPS:
                                continue
                            # costs
                            if naLen == 0:
                                cA = 0.0
                            elif i == 0:
                                cA = D[0][rB[j]] + tailB[j]
                            elif j == lenB:
                                cA = headA[i] + D[rA[i - 1]][0]
                            else:
                                cA = headA[i] + D[rA[i - 1]][rB[j]] + tailB[j]
                            if nbLen == 0:
                                cB = 0.0
                            elif j == 0:
                                cB = D[0][rA[i]] + tailA[i]
                            elif i == lenA:
                                cB = headB[j] + D[rB[j - 1]][0]
                            else:
                                cB = headB[j] + D[rB[j - 1]][rA[i]] + tailA[i]
                            delta = cA + cB - base
                            if delta < best_delta:
                                best_delta = delta
                                best_move = ('2opt*', A, i, B, j)

            if best_move is None:
                break
            apply_move(best_move)

    # ---------------- perturbation (remove + regret reinsert) ----------------
    def perturb(strength):
        removed = []
        tries = 0
        while len(removed) < strength and tries < strength * 12:
            tries += 1
            nonempty = [k for k in range(K) if routes[k]]
            if not nonempty:
                break
            A = rng.choice(nonempty)
            i = rng.randrange(len(routes[A]))
            u = routes[A][i]
            if isL[u] and nL[A] == 1 and len(routes[A]) > 1:
                continue
            routes[A].pop(i)
            recalc(A)
            removed.append(u)
        while removed:
            best_metric = None
            chosen = None
            for u in removed:
                du = demand[u]
                uL = isL[u]
                c1 = None
                c2 = None
                for k in range(K):
                    r = routes[k]
                    if uL:
                        if ld[k] + du > Q + EPS:
                            continue
                        prange = range(0, nL[k] + 1)
                    else:
                        if nL[k] == 0 or bd[k] + du > Q + EPS:
                            continue
                        prange = range(nL[k], len(r) + 1)
                    for p in prange:
                        a = r[p - 1] if p > 0 else 0
                        b = r[p] if p < len(r) else 0
                        inc = D[a][u] + D[u][b] - D[a][b]
                        if c1 is None or inc < c1[0]:
                            c2 = c1
                            c1 = (inc, k, p)
                        elif c2 is None or inc < c2[0]:
                            c2 = (inc, k, p)
                if c1 is None:
                    return False
                regret = (c2[0] - c1[0]) if c2 else 1e18
                metric = (regret, -c1[0])
                if best_metric is None or metric > best_metric:
                    best_metric = metric
                    chosen = (u, c1[1], c1[2])
            u, k, p = chosen
            routes[k].insert(p, u)
            recalc(k)
            removed.remove(u)
        return True

    # ---------------- verification ----------------
    def verify(rts):
        seen = []
        for r in rts:
            nl = 0
            sl = 0.0
            sb = 0.0
            seen_b = False
            for u in r:
                if isL[u]:
                    if seen_b:
                        return False
                    nl += 1
                    sl += demand[u]
                else:
                    seen_b = True
                    sb += demand[u]
                seen.append(u)
            if sl > Q + EPS or sb > Q + EPS:
                return False
            if len(r) > 0 and nl == 0:
                return False
        return sorted(seen) == sorted(linehauls + backhauls)

    # ---------------- main ILS ----------------
    ok = feasible_construct(seed_order(), relax=False)
    att = 0
    while not ok and att < 40 and not time_up():
        att += 1
        s = linehauls[:]
        rng.shuffle(s)
        ok = feasible_construct(s, relax=False)
    if not ok:
        feasible_construct(seed_order(), relax=True)
    recalc_all()

    best_routes = [r[:] for r in routes]
    best_cost = sum(rcost(r) for r in best_routes)
    log_incumbent(best_cost, best_routes)

    if not time_up():
        local_search()
        c = total()
        if c < best_cost - EPS:
            best_cost = c
            best_routes = [r[:] for r in routes]
            log_incumbent(best_cost, best_routes)

    while not time_up():
        routes[:] = [r[:] for r in best_routes]
        recalc_all()
        strength = rng.randint(2, max(2, min(8, N)))
        if rng.random() < 0.25:
            strength = max(strength, min(max(N, 2), 14))
        if not perturb(strength):
            continue
        local_search()
        c = total()
        if c < best_cost - EPS:
            best_cost = c
            best_routes = [r[:] for r in routes]
            log_incumbent(best_cost, best_routes)

    # final safety check and output
    if not verify(best_routes):
        # fall back to a freshly constructed feasible solution
        feasible_construct(seed_order(), relax=True)
        recalc_all()
        cand = [r[:] for r in routes]
        if verify(cand):
            best_routes = cand
            best_cost = sum(rcost(r) for r in cand)

    final_obj = float(sum(rcost(r) for r in best_routes))
    out = solution_dict(best_routes, final_obj)
    with open(args.solution_path, 'w') as f:
        json.dump(out, f)


if __name__ == '__main__':
    main()
import argparse
import json
import math
import random
import time

INF = float('inf')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--instance_path', required=True)
    ap.add_argument('--solution_path', required=True)
    ap.add_argument('--time_limit', type=int, default=60)
    ap.add_argument('--log_path', default=None)
    args = ap.parse_args()

    t0 = time.time()
    deadline = t0 + max(1.0, args.time_limit - 1.0)
    random.seed(0)

    logger = None
    if args.log_path:
        try:
            from solution_logger import SolutionLogger
            logger = SolutionLogger(args.log_path, sense="minimize")
        except Exception:
            logger = None

    with open(args.instance_path) as f:
        data = json.load(f)

    n = data['num_customers']
    K = data['num_vehicles']
    cap = data['vehicle_capacity']
    depot = data['depot']
    dend = data['depot_end']
    e0 = float(depot['time_window'][0])
    H = float(dend['time_window'][1])
    END = n + 1
    D = data['distance_matrix']
    rho = float(data['inconvenience_cost_function'].get('rho', 1.0))
    default_st = data.get('service_time', 0)

    e = [0.0] * (n + 2)
    l = [0.0] * (n + 2)
    p = [0.0] * (n + 2)
    st = [0.0] * (n + 2)
    dem = [0] * (n + 2)
    for c in data['customers']:
        i = c['id']
        e[i] = float(c['time_window'][0])
        l[i] = float(c['time_window'][1])
        p[i] = float(c.get('preferred_time', 0.5 * (e[i] + l[i])))
        st[i] = float(c.get('service_time', default_st))
        dem[i] = c['demand']

    if n == 0:
        sol = {'objective_value': 0.0, 'routes': [], 'schedule': {}}
        with open(args.solution_path, 'w') as f:
            json.dump(sol, f)
        if logger:
            logger.log_solution(0.0, sol)
        return

    VPEN = 1e7   # penalty per extra vehicle
    SPEN = 1e6   # penalty for time-infeasible route

    # ------------------------------------------------------------------
    # Route evaluation: optimal schedule via bounded isotonic regression
    # (generalized PAVA with box constraints). Returns:
    # (cost, routing_cost, inconvenience_cost, times, feasible)
    # ------------------------------------------------------------------
    def eval_route(seq):
        k = len(seq)
        if k == 0:
            return (0.0, 0.0, 0.0, [], True)
        c = seq[0]
        off = D[0][c]
        rc = off
        offs = [0.0] * k
        offs[0] = off
        prev = c
        for i in range(1, k):
            c = seq[i]
            d = D[prev][c]
            rc += d
            off += st[prev] + d
            offs[i] = off
            prev = c
        rc += D[prev][END]
        A = [0.0] * k
        B = [0.0] * k
        q = [0.0] * k
        m = e0
        for i in range(k):
            c = seq[i]
            ai = e[c] - offs[i]
            if ai > m:
                m = ai
            A[i] = m
            q[i] = p[c] - offs[i]
        last = seq[-1]
        mb = H - st[last] - D[last][END] - offs[-1]
        lb = l[last] - offs[-1]
        if lb < mb:
            mb = lb
        for i in range(k - 1, -1, -1):
            c = seq[i]
            bi = l[c] - offs[i]
            if bi < mb:
                mb = bi
            B[i] = mb
        feas = True
        for i in range(k):
            if A[i] > B[i] + 1e-9:
                feas = False
                break
        if feas:
            blocks = []
            for i in range(k):
                s = q[i]
                cnt = 1
                am = A[i]
                bm = B[i]
                v = s
                if v < am:
                    v = am
                elif v > bm:
                    v = bm
                while blocks and blocks[-1][4] > v:
                    ps, pc_, pa, pb, _ = blocks.pop()
                    s += ps
                    cnt += pc_
                    if pa > am:
                        am = pa
                    if pb < bm:
                        bm = pb
                    v = s / cnt
                    if v < am:
                        v = am
                    elif v > bm:
                        v = bm
                blocks.append((s, cnt, am, bm, v))
            ic = 0.0
            times = [0.0] * k
            i = 0
            for _, cnt, _, _, v in blocks:
                for _ in range(cnt):
                    dq = v - q[i]
                    ic += dq * dq
                    times[i] = v + offs[i]
                    i += 1
            ic *= rho
            return (rc + ic, rc, ic, times, True)
        else:
            # soft (penalized) schedule for infeasible routes
            u = [0.0] * k
            viol = 0.0
            prevu = -INF
            for i in range(k):
                lo = A[i]
                if prevu > lo:
                    lo = prevu
                v = q[i]
                if v < lo:
                    v = lo
                bi = B[i]
                if bi >= lo and v > bi:
                    v = bi
                c = seq[i]
                ub = l[c] - offs[i]
                if i == k - 1:
                    ub2 = H - st[c] - D[c][END] - offs[i]
                    if ub2 < ub:
                        ub = ub2
                if v > ub:
                    viol += v - ub
                u[i] = v
                prevu = v
            ic = 0.0
            times = [0.0] * k
            for i in range(k):
                dq = u[i] - q[i]
                ic += dq * dq
                times[i] = u[i] + offs[i]
            ic *= rho
            return (rc + ic + SPEN + 1e4 * viol, rc, ic, times, False)

    custs = list(range(1, n + 1))
    nb = [[] for _ in range(n + 2)]
    for i in range(1, n + 1):
        nb[i] = sorted((j for j in custs if j != i), key=lambda j: D[i][j])

    # ------------------------------------------------------------------
    # Insertion machinery
    # ------------------------------------------------------------------
    def best_insertion(c, routes, loads, costs):
        bestd = INF
        second = INF
        best = None
        dc = dem[c]
        for ri in range(len(routes)):
            if loads[ri] + dc > cap:
                continue
            r = routes[ri]
            base = costs[ri]
            for pos in range(len(r) + 1):
                ns = r[:pos] + [c] + r[pos:]
                cst = eval_route(ns)[0]
                dta = cst - base
                if dta < bestd:
                    second = bestd
                    bestd = dta
                    best = (ri, ns, cst)
                elif dta < second:
                    second = dta
        cst = eval_route([c])[0]
        dta = cst + (VPEN if len(routes) >= K else 0.0)
        if dta < bestd:
            second = bestd
            bestd = dta
            best = (-1, [c], cst)
        elif dta < second:
            second = dta
        return bestd, best, second

    def apply_insertion(best, c, routes, loads, costs):
        ri, ns, cst = best
        if ri == -1:
            routes.append(ns)
            loads.append(dem[c])
            costs.append(cst)
        else:
            routes[ri] = ns
            loads[ri] += dem[c]
            costs[ri] = cst

    def greedy_repair(removed, routes, loads, costs):
        for c in removed:
            _, best, _ = best_insertion(c, routes, loads, costs)
            apply_insertion(best, c, routes, loads, costs)

    def regret_repair(removed, routes, loads, costs):
        rem = list(removed)
        while rem:
            bestc = None
            bestreg = -INF
            bestbest = None
            bestd0 = INF
            for c in rem:
                d0, best, d1 = best_insertion(c, routes, loads, costs)
                reg = (d1 - d0) if d1 < INF else 1e12
                if reg > bestreg or (reg == bestreg and d0 < bestd0):
                    bestreg = reg
                    bestc = c
                    bestbest = best
                    bestd0 = d0
            apply_insertion(bestbest, bestc, routes, loads, costs)
            rem.remove(bestc)

    def remove_customers(S, routes, loads, costs):
        Sset = set(S)
        i = 0
        while i < len(routes):
            r = routes[i]
            hit = False
            for c in r:
                if c in Sset:
                    hit = True
                    break
            if not hit:
                i += 1
                continue
            nr2 = [c for c in r if c not in Sset]
            if nr2:
                routes[i] = nr2
                loads[i] = sum(dem[c] for c in nr2)
                costs[i] = eval_route(nr2)[0]
                i += 1
            else:
                routes.pop(i)
                loads.pop(i)
                costs.pop(i)

    def shaw_removal(qn):
        s0 = custs[random.randrange(n)]
        S = [s0]
        ss = {s0}
        while len(S) < min(qn, n):
            r = S[random.randrange(len(S))]
            added = False
            for j in nb[r]:
                if j not in ss:
                    S.append(j)
                    ss.add(j)
                    added = True
                    break
            if not added:
                break
        return S

    def worst_removal(routes, costs, qn):
        gains = []
        for ri, r in enumerate(routes):
            base = costs[ri]
            for idx in range(len(r)):
                ns = r[:idx] + r[idx + 1:]
                cst = eval_route(ns)[0] if ns else 0.0
                gains.append((base - cst, r[idx]))
        gains.sort(key=lambda x: -x[0])
        S = []
        while len(S) < qn and gains:
            idx = int((random.random() ** 3) * len(gains))
            S.append(gains.pop(idx)[1])
        return S

    def total_cost(routes, costs):
        t = sum(costs)
        if len(routes) > K:
            t += VPEN * (len(routes) - K)
        return t

    def solution_feasible(routes):
        if len(routes) > K:
            return False
        for r in routes:
            if not eval_route(r)[4]:
                return False
        return True

    def build_solution(routes):
        routes_out = []
        sched = {}
        rc_tot = 0.0
        ic_tot = 0.0
        for r in routes:
            if not r:
                continue
            cst, rc, ic, times, fe = eval_route(r)
            routes_out.append([0] + list(r) + [END])
            rc_tot += rc
            ic_tot += ic
            for cc, tt in zip(r, times):
                sched[str(cc)] = float(tt)
        return {'objective_value': rc_tot + ic_tot,
                'routes': routes_out, 'schedule': sched}

    # ------------------------------------------------------------------
    # Initial solution
    # ------------------------------------------------------------------
    routes, loads, costs = [], [], []
    order = sorted(custs, key=lambda c: (p[c], l[c]))
    if n <= 250:
        greedy_repair(order, routes, loads, costs)
    else:
        for c in order:
            bestd = INF
            best = None
            for ri in range(len(routes)):
                if loads[ri] + dem[c] > cap:
                    continue
                ns = routes[ri] + [c]
                res = eval_route(ns)
                if res[4]:
                    dta = res[0] - costs[ri]
                    if dta < bestd:
                        bestd = dta
                        best = (ri, ns, res[0])
            cst = eval_route([c])[0]
            dta = cst + (VPEN if len(routes) >= K else 0.0)
            if dta < bestd:
                best = (-1, [c], cst)
            apply_insertion(best, c, routes, loads, costs)

    cur_total = total_cost(routes, costs)
    best_total = cur_total
    best_routes = [list(r) for r in routes]
    best_feas_sol = None
    best_feas_obj = INF
    if solution_feasible(routes):
        sol = build_solution(routes)
        best_feas_sol = sol
        best_feas_obj = sol['objective_value']
        if logger:
            logger.log_solution(sol['objective_value'], sol)

    # ------------------------------------------------------------------
    # LNS with simulated annealing acceptance
    # ------------------------------------------------------------------
    qmin = 1 if n < 4 else 2
    qmax = max(qmin, min(n, max(5, n // 7), 30))
    T0v = max(1.0, 0.04 * min(cur_total, 1e5))
    Tend = max(1e-3, 1e-3 * T0v)
    stall = 0

    while True:
        now = time.time()
        if now >= deadline:
            break
        frac = (now - t0) / max(1e-9, deadline - t0)
        T = T0v * (Tend / T0v) ** min(1.0, frac)

        nr = [list(r) for r in routes]
        nl = list(loads)
        nc = list(costs)
        qn = random.randint(qmin, qmax)
        rnd = random.random()
        if rnd < 0.45:
            S = random.sample(custs, min(qn, n))
        elif rnd < 0.85:
            S = shaw_removal(qn)
        else:
            S = worst_removal(nr, nc, qn)
        remove_customers(S, nr, nl, nc)
        Slist = list(S)
        random.shuffle(Slist)
        if random.random() < 0.25 and len(Slist) <= 12:
            regret_repair(Slist, nr, nl, nc)
        else:
            greedy_repair(Slist, nr, nl, nc)

        newtot = total_cost(nr, nc)
        d = newtot - cur_total
        accept = False
        if d < -1e-9:
            accept = True
        elif T > 1e-12 and d < 50 * T and random.random() < math.exp(-d / T):
            accept = True
        if accept:
            routes, loads, costs = nr, nl, nc
            cur_total = newtot
            if newtot < best_total - 1e-9:
                best_total = newtot
                best_routes = [list(r) for r in routes]
                stall = 0
                if solution_feasible(best_routes):
                    sol = build_solution(best_routes)
                    if sol['objective_value'] < best_feas_obj - 1e-9:
                        best_feas_obj = sol['objective_value']
                        best_feas_sol = sol
                        if logger:
                            logger.log_solution(sol['objective_value'], sol)
            else:
                stall += 1
        else:
            stall += 1

        if stall >= 1000:
            routes = [list(r) for r in best_routes]
            loads = [sum(dem[c] for c in r) for r in routes]
            costs = [eval_route(r)[0] for r in routes]
            cur_total = total_cost(routes, costs)
            stall = 0

    if best_feas_sol is None:
        best_feas_sol = build_solution(best_routes)
        if logger:
            logger.log_solution(best_feas_sol['objective_value'], best_feas_sol)

    with open(args.solution_path, 'w') as f:
        json.dump(best_feas_sol, f)


if __name__ == '__main__':
    main()
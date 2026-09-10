import argparse
import json
import math
import time
import random

import numpy as np

from solution_logger import SolutionLogger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--instance_path', required=True)
    ap.add_argument('--solution_path', required=True)
    ap.add_argument('--time_limit', type=int, default=60)
    ap.add_argument('--log_path', default=None)
    args = ap.parse_args()

    t0 = time.time()
    time_limit = max(5, int(args.time_limit))
    deadline = t0 + time_limit - max(1.0, 0.03 * time_limit)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        data = json.load(f)
    p = data['parameters']
    T = float(p['T_planning_horizon_minutes'])
    D = int(p['D_num_train_services'])
    NT = int(p['NT_num_demand_periods'])
    theta = float(p['theta_period_length_minutes'])
    hmin = float(p['h_min_minutes'])
    hmax = float(p['h_max_minutes'])
    C = float(p['C_train_capacity_passengers'])
    periods = sorted(data['demand_periods'], key=lambda e: e['period'])
    lam = [float(e['lambda_pax_per_min']) for e in periods]
    Cum = [0.0]
    for j in range(NT):
        Cum.append(Cum[-1] + lam[j] * theta)
    Atot = Cum[-1]

    EPS_END = 1e-6
    MARGIN = 1e-6

    def A(t):
        if t <= 0.0:
            return 0.0
        j = int(t / theta)
        if j >= NT:
            return Atot
        return Cum[j] + lam[j] * (t - j * theta)

    def evaluate(ts):
        total = 0.0
        W = 0.0
        pt = 0.0
        pA = 0.0
        for t in ts:
            a = A(t)
            N = a - pA
            H = t - pt
            total += H * (W + 0.5 * N)
            W = W + N - C
            if W < 0.0:
                W = 0.0
            pt = t
            pA = a
        H = T - pt
        N = Atot - pA
        total += H * (W + 0.5 * N)
        return total

    def feasible(ts):
        if ts[0] < -1e-9:
            return False
        if ts[-1] > T - 1e-9:
            return False
        for d in range(1, D):
            h = ts[d] - ts[d - 1]
            if h < hmin - 1e-6 or h > hmax + 1e-6:
                return False
        return True

    best_obj = math.inf
    best_ts = None

    def make_solution(v, ts):
        return {
            "objective_value": float(v),
            "departure_schedule_minutes": {str(i + 1): float(ts[i]) for i in range(D)}
        }

    def record(ts):
        nonlocal best_obj, best_ts
        v = evaluate(ts)
        if v < best_obj - 1e-9 and feasible(ts):
            best_obj = v
            best_ts = list(ts)
            if logger:
                logger.log_solution(v, make_solution(v, ts))
        return v

    # ---------- initial feasible schedule ----------
    if D == 1:
        ts0 = [min(max(0.0, T / 2.0), T - EPS_END)]
    else:
        h = min(hmax, max(hmin, T / (D + 1)))
        if (D - 1) * h > T - 1.0:
            h = max(hmin, (T - 1.0) / (D - 1))
        span = (D - 1) * h
        t1 = max(0.0, (T - span) / 2.0)
        if t1 + span > T - 1e-3:
            t1 = max(0.0, T - 1e-3 - span)
        ts0 = [t1 + k * h for k in range(D)]
        if not feasible(ts0):
            # fall back to tight-packed schedule
            ts0 = [k * hmin for k in range(D)]
            if ts0[-1] > T - 1e-3:
                scale = (T - 1e-3) / max(ts0[-1], 1e-9)
                ts0 = [t * scale for t in ts0]
    record(ts0)

    # ---------- DP (no capacity binding: Atot <= C) ----------
    def dp_simple(g, dl):
        M = int(math.floor(T / g + 1e-9))
        top = M if M * g < T - 1e-9 else M - 1
        wlo = max(1, int(math.ceil(hmin / g - 1e-9)))
        whi = int(math.floor(hmax / g + 1e-9))
        if whi < wlo or top < 0:
            return None

        def imin(d):
            return (d - 1) * wlo

        def imax(d):
            return top - (D - d) * wlo

        if imin(D) > imax(D):
            return None
        n = top + 1
        tg = np.arange(n) * g
        Ag = np.array([A(t) for t in tg])
        INF = math.inf
        V = np.full(n, INF)
        lo1, hi1 = imin(1), imax(1)
        V[lo1:hi1 + 1] = tg[lo1:hi1 + 1] * 0.5 * Ag[lo1:hi1 + 1]
        parents = [None] * D
        for d in range(2, D + 1):
            newV = np.full(n, INF)
            par = np.full(n, -1, dtype=np.int64)
            jl, jh = imin(d), imax(d)
            pl, ph = imin(d - 1), imax(d - 1)
            for j in range(jl, jh + 1):
                a = max(j - whi, pl)
                b = min(j - wlo, ph)
                if b < a:
                    continue
                seg = V[a:b + 1] + (tg[j] - tg[a:b + 1]) * (0.5 * (Ag[j] - Ag[a:b + 1]))
                k = int(np.argmin(seg))
                newV[j] = seg[k]
                par[j] = a + k
            if time.time() > dl:
                return None
            V = newV
            parents[d - 1] = par
        jl, jh = imin(D), imax(D)
        tot = V[jl:jh + 1] + (T - tg[jl:jh + 1]) * (0.5 * (Atot - Ag[jl:jh + 1]))
        k = int(np.argmin(tot))
        if not math.isfinite(float(tot[k])):
            return None
        j = jl + k
        ts = [0.0] * D
        ts[D - 1] = float(tg[j])
        for d in range(D - 1, 0, -1):
            j = int(parents[d][j])
            if j < 0:
                return None
            ts[d - 1] = float(tg[j])
        return ts

    # ---------- DP with capacity (Pareto over (cost, waiting)) ----------
    def dp_pareto(g, dl, K=6):
        M = int(math.floor(T / g + 1e-9))
        top = M if M * g < T - 1e-9 else M - 1
        wlo = max(1, int(math.ceil(hmin / g - 1e-9)))
        whi = int(math.floor(hmax / g + 1e-9))
        if whi < wlo or top < 0:
            return None

        def imin(d):
            return (d - 1) * wlo

        def imax(d):
            return top - (D - d) * wlo

        if imin(D) > imax(D):
            return None
        n = top + 1
        tg = [i * g for i in range(n)]
        Ag = [A(t) for t in tg]
        layers = []
        layer = [None] * n
        for i in range(imin(1), imax(1) + 1):
            a = Ag[i]
            W = a - C
            if W < 0.0:
                W = 0.0
            layer[i] = [(tg[i] * 0.5 * a, W, -1, -1)]
        layers.append(layer)
        for d in range(2, D + 1):
            nl = [None] * n
            jl_d, jh_d = imin(d), imax(d)
            for i in range(imin(d - 1), imax(d - 1) + 1):
                ents = layer[i]
                if not ents:
                    continue
                jlo = i + wlo if i + wlo > jl_d else jl_d
                jhi = i + whi if i + whi < jh_d else jh_d
                if jhi < jlo:
                    continue
                ti = tg[i]
                ai = Ag[i]
                for j in range(jlo, jhi + 1):
                    dt_ = tg[j] - ti
                    Nn = Ag[j] - ai
                    lst = nl[j]
                    if lst is None:
                        lst = nl[j] = []
                    for k in range(len(ents)):
                        e0 = ents[k]
                        c = e0[0]
                        W = e0[1]
                        nc = c + dt_ * (W + 0.5 * Nn)
                        nW = W + Nn - C
                        if nW < 0.0:
                            nW = 0.0
                        dominated = False
                        for e in lst:
                            if e[0] <= nc + 1e-9 and e[1] <= nW + 1e-9:
                                dominated = True
                                break
                        if dominated:
                            continue
                        if lst:
                            lst[:] = [e for e in lst
                                      if not (nc <= e[0] + 1e-9 and nW <= e[1] + 1e-9)]
                        lst.append((nc, nW, i, k))
                        if len(lst) > K:
                            lst.sort(key=lambda e: e[0])
                            del lst[K:]
                if (i & 63) == 0 and time.time() > dl:
                    return None
            layers.append(nl)
            layer = nl
        best = math.inf
        bi = -1
        bk = -1
        for i in range(imin(D), imax(D) + 1):
            ents = layer[i]
            if not ents:
                continue
            tail = T - tg[i]
            Nn = Atot - Ag[i]
            for k, e in enumerate(ents):
                tot = e[0] + tail * (e[1] + 0.5 * Nn)
                if tot < best:
                    best = tot
                    bi = i
                    bk = k
        if bi < 0:
            return None
        ts = [0.0] * D
        i, k = bi, bk
        for d in range(D - 1, -1, -1):
            ts[d] = tg[i]
            e = layers[d][i][k]
            i, k = e[2], e[3]
        return ts

    def choose_g_simple():
        cap = 3e5 if time_limit >= 30 else 1e5
        cands = [0.25, 0.5, 1.0, 2.0, 4.0, 5.0, 8.0, 10.0, 16.0, 20.0]
        pick = None
        for g in cands:
            wlo = max(1, int(math.ceil(hmin / g - 1e-9)))
            whi = int(math.floor(hmax / g + 1e-9))
            if whi < wlo:
                continue
            M = T / g
            if D * M <= cap:
                return g
            pick = g
        return pick

    def choose_g_pareto():
        cap = 5e5 if time_limit >= 30 else 1.5e5
        cands = [0.5, 1.0, 2.0, 4.0, 5.0, 8.0, 10.0, 16.0, 20.0, 30.0]
        pick = None
        for g in cands:
            wlo = max(1, int(math.ceil(hmin / g - 1e-9)))
            whi = int(math.floor(hmax / g + 1e-9))
            if whi < wlo:
                continue
            M = int(math.floor(T / g + 1e-9))
            top = M if M * g < T - 1e-9 else M - 1
            if (D - 1) * wlo > top:
                continue
            window = whi - wlo + 1
            est = D * (top + 1) * window
            if est <= cap:
                return g
            if pick is None:
                pick = g
        return pick

    dp_deadline = min(deadline - 1.0, t0 + 0.55 * time_limit)
    starts = []
    if Atot <= C + 1e-9:
        g = choose_g_simple()
        if g is not None:
            ts = dp_simple(g, dp_deadline)
            if ts is not None and feasible(ts):
                starts.append(ts)
                record(ts)
    else:
        g = choose_g_pareto()
        if g is not None:
            ts = dp_pareto(g, dp_deadline)
            if ts is not None and feasible(ts):
                starts.append(ts)
                record(ts)
        # also try capacity-ignoring DP as an extra start (exact eval fixes cost)
        if time.time() < dp_deadline:
            g2 = choose_g_simple()
            if g2 is not None:
                ts2 = dp_simple(g2, dp_deadline)
                if ts2 is not None and feasible(ts2):
                    starts.append(ts2)
                    record(ts2)

    if best_ts is None:
        best_ts = list(ts0)
        best_obj = evaluate(ts0)

    # ---------- coordinate descent (continuous refinement, exact objective) ----------
    def cd(ts_in, dl):
        ts = list(ts_in)
        cur = evaluate(ts)
        pass_dt = 0.25
        while time.time() < dl:
            improved_any = False
            for d in range(D):
                if time.time() > dl:
                    break
                lo = 0.0 if d == 0 else ts[d - 1] + hmin + MARGIN
                hi = (T - EPS_END) if d == D - 1 else ts[d + 1] - hmin - MARGIN
                if d > 0:
                    hi = min(hi, ts[d - 1] + hmax - MARGIN)
                if d < D - 1:
                    lo = max(lo, ts[d + 1] - hmax + MARGIN)
                if hi < lo:
                    continue
                rng = hi - lo
                step = max(pass_dt, rng / 800.0) if rng > 0 else 1.0
                cands = set()
                if rng > 0:
                    x = lo
                    while x < hi:
                        cands.add(x)
                        x += step
                cands.add(lo)
                cands.add(hi)
                curx = min(max(ts[d], lo), hi)
                cands.add(curx)
                jb = int(lo // theta) + 1
                while jb * theta < hi:
                    if jb * theta > lo:
                        cands.add(jb * theta)
                    jb += 1
                bestx = curx
                ts[d] = curx
                bestv = evaluate(ts)
                for x in cands:
                    ts[d] = x
                    v = evaluate(ts)
                    if v < bestv - 1e-10:
                        bestv = v
                        bestx = x
                # local refinement around best candidate
                l2 = max(lo, bestx - step)
                h2 = min(hi, bestx + step)
                if h2 > l2:
                    s2 = (h2 - l2) / 40.0
                    x = l2
                    while x <= h2 + 1e-12:
                        ts[d] = x
                        v = evaluate(ts)
                        if v < bestv - 1e-10:
                            bestv = v
                            bestx = x
                        x += s2
                    l3 = max(lo, bestx - s2)
                    h3 = min(hi, bestx + s2)
                    if h3 > l3:
                        s3 = (h3 - l3) / 20.0
                        x = l3
                        while x <= h3 + 1e-12:
                            ts[d] = x
                            v = evaluate(ts)
                            if v < bestv - 1e-10:
                                bestv = v
                                bestx = x
                            x += s3
                ts[d] = bestx
                if bestv < cur - 1e-9:
                    cur = bestv
                    improved_any = True
            record(ts)
            if not improved_any:
                if pass_dt <= 0.021:
                    break
                pass_dt /= 5.0
        return ts, cur

    # run CD from the best available start
    start = list(best_ts)
    if starts:
        for s in starts:
            if evaluate(s) < evaluate(start):
                start = list(s)
    if time.time() < deadline:
        cd(start, deadline)

    # ---------- perturbation restarts ----------
    random.seed(0)

    def perturb(ts):
        sigma = random.uniform(0.2, 1.5) * max(1.0, hmax - hmin)
        new = sorted(t + random.gauss(0.0, sigma) for t in ts)
        m = EPS_END
        new[0] = min(max(new[0], 0.0), T - m)
        for d in range(1, D):
            lo = new[d - 1] + hmin
            hi2 = new[d - 1] + hmax
            new[d] = min(max(new[d], lo), hi2)
        if new[-1] > T - m:
            new[-1] = T - m
            for d in range(D - 2, -1, -1):
                hi2 = new[d + 1] - hmin
                lo = new[d + 1] - hmax
                if new[d] > hi2:
                    new[d] = hi2
                if new[d] < lo:
                    new[d] = lo
            if new[0] < 0.0:
                return None
        return new if feasible(new) else None

    fails = 0
    while time.time() < deadline - 0.5 and fails < 200:
        cand = perturb(best_ts)
        if cand is None:
            fails += 1
            continue
        fails = 0
        cd(cand, deadline)

    # ---------- final output ----------
    if best_ts is None:
        best_ts = list(ts0)
        best_obj = evaluate(ts0)
    # safety clamp
    out_ts = [min(max(t, 0.0), T - EPS_END) for t in best_ts]
    out_ts.sort()
    sol = make_solution(evaluate(out_ts), out_ts)
    with open(args.solution_path, 'w') as f:
        json.dump(sol, f, indent=2)


if __name__ == '__main__':
    main()
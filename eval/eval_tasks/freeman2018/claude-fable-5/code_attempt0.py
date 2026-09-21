import argparse
import json
import math
import random
import time

import numpy as np

from solution_logger import SolutionLogger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--instance_path', required=True)
    ap.add_argument('--solution_path', required=True)
    ap.add_argument('--time_limit', type=int, default=60)
    ap.add_argument('--log_path', default=None)
    args = ap.parse_args()

    t_start = time.time()
    total_time = max(1.0, float(args.time_limit) - 0.5)
    deadline = t_start + total_time

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    rng = random.Random(0)
    np.random.seed(0)

    with open(args.instance_path) as fh:
        inst = json.load(fh)

    S = int(inst['num_sites'])
    C = int(inst['num_customers'])
    T = int(inst['num_periods'])
    prm = inst['parameters']
    delta = int(prm['delta'])
    eps = int(prm['epsilon'])
    L = float(prm['L'])
    ccost = float(prm['c'])
    q = float(prm['q'])
    wcycle = list(prm['weekly_attraction_cycle'])
    avgc = float(prm['avg_businesses_customers'])
    avgs = float(prm['avg_businesses_sites'])

    sites = sorted(inst['sites'], key=lambda x: x['id'])
    site_ids = [int(st['id']) for st in sites]
    cap = np.array([st['capacity'] for st in sites], dtype=float)
    rev = np.array([st['revenue_per_customer'] for st in sites], dtype=float)
    fix = np.array([st['fixed_cost'] for st in sites], dtype=float)
    maxev = np.array([st['max_events'] for st in sites], dtype=np.int64)
    nbs = np.array([st['num_businesses'] for st in sites], dtype=float)

    custs = sorted(inst['customers'], key=lambda x: x['id'])
    cust_ids = [int(cu['id']) for cu in custs]
    pop = np.array([cu['population'] for cu in custs], dtype=float)
    nbc = np.array([cu['num_businesses'] for cu in custs], dtype=float)

    # site-to-site distances (indices 0..S-1 correspond to site_ids)
    ss = inst['site_to_site_distances']
    dss = np.zeros((S, S))
    for a in range(S):
        for b in range(S):
            k1 = f"{site_ids[a]}_{site_ids[b]}"
            k2 = f"{site_ids[b]}_{site_ids[a]}"
            if k1 in ss:
                dss[a, b] = float(ss[k1])
            elif k2 in ss:
                dss[a, b] = float(ss[k2])
            else:
                dss[a, b] = 0.0

    csd = inst['customer_to_site_distances']
    dcs = np.zeros((C, S))
    for i in range(C):
        for j in range(S):
            k1 = f"{cust_ids[i]}_{site_ids[j]}"
            dcs[i, j] = float(csd.get(k1, 0.0))
    dcs = np.maximum(dcs, 1e-9)

    # weekly attractiveness: period 1 -> Monday (index 0)
    w = np.array([wcycle[t % 7] for t in range(T)], dtype=float)

    alpha = nbs / max(avgs, 1e-12)                  # site attraction (constant over time)
    base = alpha[None, :] * (dcs ** (-q))           # (C, S)
    A = base[:, :, None] * w[None, None, :]         # (C, S, T) attraction a_ijt
    u = (nbc / max(avgc, 1e-12))[:, None] * w[None, :]  # (C, T) self-attraction
    u = np.maximum(u, 1e-12)

    tt = np.arange(T)
    lo_idx = np.maximum(0, tt - eps)
    hi_idx = np.minimum(T, tt + eps + 1)
    R = dss <= L + 1e-9  # reachability within one period

    wlen = delta + 2

    # ---------- evaluation ----------
    def evaluate(s, y):
        yf = y.astype(float)
        b = A[:, s, tt] * yf[None, :]                # (C, T)
        P = np.zeros((C, T + 1))
        np.cumsum(b, axis=1, out=P[:, 1:])
        W = P[:, hi_idx] - P[:, lo_idx]              # shadow window sums
        den = u + W
        prop = b / den
        np.minimum(prop, 1.0, out=prop)
        attsum = pop @ prop                          # (T,)
        att = np.minimum(cap[s], attsum) * yf
        obj = float(np.dot(rev[s], att) - np.dot(fix[s], yf)
                    - ccost * dss[s[:-1], s[1:]].sum())
        return obj, att

    def window_ok(y):
        if T < wlen:
            return True
        cy = np.concatenate(([0], np.cumsum(y)))
        return bool(np.all(cy[wlen:] - cy[:-wlen] <= delta))

    def counts_ok(s, y):
        mask = y.astype(bool)
        if not mask.any():
            return True
        cnt = np.bincount(s[mask], minlength=S)
        return bool(np.all(cnt <= maxev))

    # ---------- standalone event profit p0[j,t] ----------
    p0 = np.zeros((S, T))
    for t in range(T):
        At = A[:, :, t]
        prop = np.minimum(At / (u[:, t:t + 1] + At), 1.0)
        attsum = pop @ prop
        att = np.minimum(cap, attsum)
        p0[:, t] = rev * att - fix

    # ---------- DP construction (ignoring shadow + window; repaired after) ----------
    NEG = -1e17
    gain = np.maximum(p0, 0.0)
    V = np.full((T, S), NEG)
    par = np.zeros((T, S), dtype=np.int64)
    V[0] = gain[:, 0]  # dummy start reaches any site at zero distance
    for t in range(1, T):
        M = V[t - 1][:, None] - ccost * dss
        M[~R] = NEG
        par[t] = np.argmax(M, axis=0)
        V[t] = M[par[t], np.arange(S)] + gain[:, t]

    s0 = np.zeros(T, dtype=np.int64)
    s0[T - 1] = int(np.argmax(V[T - 1]))
    for t in range(T - 1, 0, -1):
        s0[t - 1] = par[t, s0[t]]
    y0 = (p0[s0, tt] > 1e-9).astype(np.int64)

    # verify distance feasibility of DP path; fallback to staying put
    ok_path = all(R[s0[t - 1], s0[t]] for t in range(1, T))
    if not ok_path:
        stay = 0
        for j in range(S):
            if R[j, j]:
                stay = j
                break
        s0 = np.full(T, stay, dtype=np.int64)
        y0 = np.zeros(T, dtype=np.int64)

    # repair per-site max event counts
    for j in range(S):
        idxs = [t for t in range(T) if s0[t] == j and y0[t] == 1]
        if len(idxs) > maxev[j]:
            idxs.sort(key=lambda t: p0[j, t])
            for t in idxs[:len(idxs) - int(maxev[j])]:
                y0[t] = 0

    # repair sliding-window rest constraint
    if T >= wlen:
        while True:
            cy = np.concatenate(([0], np.cumsum(y0)))
            ws = cy[wlen:] - cy[:-wlen]
            viol = np.flatnonzero(ws > delta)
            if len(viol) == 0:
                break
            st = int(viol[0])
            cand = [t for t in range(st, st + wlen) if y0[t] == 1]
            tmin = min(cand, key=lambda t: p0[s0[t], t])
            y0[tmin] = 0

    cur_s, cur_y = s0, y0
    cur_obj, cur_att = evaluate(cur_s, cur_y)

    best_obj = cur_obj
    best_s = cur_s.copy()
    best_y = cur_y.copy()
    best_att = cur_att.copy()

    def build_solution(s, y, att, obj):
        tour = {str(t + 1): int(site_ids[s[t]]) for t in range(T)}
        events = {str(t + 1): int(y[t]) for t in range(T)}
        event_sites = {str(t + 1): int(site_ids[s[t]]) for t in range(T) if y[t]}
        attendance = {f"{site_ids[s[t]]}_{t + 1}": float(att[t]) for t in range(T) if y[t]}
        return {
            "objective_value": float(obj),
            "tour": tour,
            "events": events,
            "event_sites": event_sites,
            "attendance": attendance,
        }

    if logger:
        logger.log_solution(best_obj, build_solution(best_s, best_y, best_att, best_obj))

    # ---------- SA moves ----------
    def gen_move(s, y):
        r = rng.random()
        if r < 0.30:
            # change site at one period
            t = rng.randrange(T)
            if t == 0:
                mask = np.ones(S, dtype=bool)
            else:
                mask = R[s[t - 1]].copy()
            if t < T - 1:
                mask &= R[:, s[t + 1]]
            mask[s[t]] = False
            cands = np.flatnonzero(mask)
            if len(cands) == 0:
                return None
            j = int(cands[rng.randrange(len(cands))])
            s2 = s.copy()
            s2[t] = j
            return s2, y, False, bool(y[t])
        elif r < 0.58:
            # toggle event
            t = rng.randrange(T)
            y2 = y.copy()
            y2[t] ^= 1
            turned_on = bool(y2[t])
            return s, y2, turned_on, turned_on
        elif r < 0.80:
            # relocate a block of consecutive periods to one site
            t1 = rng.randrange(T)
            length = 1 + rng.randrange(min(7, T - t1))
            t2 = t1 + length - 1
            mask = np.ones(S, dtype=bool)
            if t1 > 0:
                mask &= R[s[t1 - 1]]
            if t2 < T - 1:
                mask &= R[:, s[t2 + 1]]
            cands = np.flatnonzero(mask)
            if len(cands) == 0:
                return None
            j = int(cands[rng.randrange(len(cands))])
            s2 = s.copy()
            s2[t1:t2 + 1] = j
            return s2, y, False, True
        else:
            # shift an event to another period
            on = np.flatnonzero(y)
            off = np.flatnonzero(y == 0)
            if len(on) == 0 or len(off) == 0:
                return None
            t1 = int(on[rng.randrange(len(on))])
            t2 = int(off[rng.randrange(len(off))])
            y2 = y.copy()
            y2[t1] = 0
            y2[t2] = 1
            return s, y2, True, True

    # ---------- estimate initial temperature ----------
    deltas = []
    for _ in range(80):
        mv = gen_move(cur_s, cur_y)
        if mv is None:
            continue
        s2, y2, chkw, chkc = mv
        if chkw and not window_ok(y2):
            continue
        if chkc and not counts_ok(s2, y2):
            continue
        obj2, _ = evaluate(s2, y2)
        d = abs(obj2 - cur_obj)
        if d > 1e-9:
            deltas.append(d)
    T0 = max(1.0, float(np.median(deltas))) if deltas else 100.0
    Tend = max(1e-6, T0 * 1e-4)
    ratio = Tend / T0

    # ---------- simulated annealing ----------
    temp = T0
    it = 0
    stall = 0
    while True:
        if (it & 63) == 0:
            now = time.time()
            if now > deadline:
                break
            frac = min(1.0, (now - t_start) / total_time)
            temp = T0 * (ratio ** frac)
        it += 1

        mv = gen_move(cur_s, cur_y)
        if mv is None:
            continue
        s2, y2, chkw, chkc = mv
        if chkw and not window_ok(y2):
            continue
        if chkc and not counts_ok(s2, y2):
            continue
        obj2, att2 = evaluate(s2, y2)
        d = obj2 - cur_obj
        if d >= 0 or (temp > 1e-12 and d / temp > -60 and rng.random() < math.exp(d / temp)):
            cur_s, cur_y, cur_obj = s2, y2, obj2
            if obj2 > best_obj + 1e-9:
                best_obj = obj2
                best_s = s2.copy()
                best_y = y2.copy()
                best_att = att2.copy()
                stall = 0
                if logger:
                    logger.log_solution(best_obj,
                                        build_solution(best_s, best_y, best_att, best_obj))
            else:
                stall += 1
        else:
            stall += 1

        if stall > 40000:
            # restart from best with a mild reheat
            cur_s = best_s.copy()
            cur_y = best_y.copy()
            cur_obj = best_obj
            temp = max(temp, T0 * 0.1)
            stall = 0

    # ---------- greedy polish (improving moves only) ----------
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # try toggling each period
        for t in range(T):
            if time.time() > deadline:
                break
            y2 = best_y.copy()
            y2[t] ^= 1
            if y2[t] and (not window_ok(y2) or not counts_ok(best_s, y2)):
                continue
            obj2, att2 = evaluate(best_s, y2)
            if obj2 > best_obj + 1e-9:
                best_obj, best_y, best_att = obj2, y2, att2
                improved = True
                if logger:
                    logger.log_solution(best_obj,
                                        build_solution(best_s, best_y, best_att, best_obj))
        # try changing single sites
        for t in range(T):
            if time.time() > deadline:
                break
            if t == 0:
                mask = np.ones(S, dtype=bool)
            else:
                mask = R[best_s[t - 1]].copy()
            if t < T - 1:
                mask &= R[:, best_s[t + 1]]
            mask[best_s[t]] = False
            for j in np.flatnonzero(mask):
                s2 = best_s.copy()
                s2[t] = int(j)
                if best_y[t] and not counts_ok(s2, best_y):
                    continue
                obj2, att2 = evaluate(s2, best_y)
                if obj2 > best_obj + 1e-9:
                    best_obj, best_s, best_att = obj2, s2, att2
                    improved = True
                    if logger:
                        logger.log_solution(best_obj,
                                            build_solution(best_s, best_y, best_att, best_obj))
                    break

    sol = build_solution(best_s, best_y, best_att, best_obj)
    with open(args.solution_path, 'w') as fh:
        json.dump(sol, fh, indent=2)


if __name__ == '__main__':
    main()
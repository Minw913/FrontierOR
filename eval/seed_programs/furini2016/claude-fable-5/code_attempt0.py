import argparse
import json
import time
import bisect
from collections import defaultdict

import numpy as np

try:
    from solution_logger import SolutionLogger
except Exception:
    SolutionLogger = None


# ----------------------------------------------------------------------
# Discretization point generation (normal patterns / raster points)
# ----------------------------------------------------------------------
def bit_points(pairs, limit):
    """Reachable positive integer sums (bounded knapsack) via bitset."""
    mask = (1 << (limit + 1)) - 1
    reach = 1
    for s, c in pairs:
        if s > limit or c <= 0:
            continue
        step, rem = 1, c
        while rem > 0:
            use = min(step, rem)
            new = reach | ((reach << (s * use)) & mask)
            if new == reach:
                break
            reach = new
            rem -= use
            step <<= 1
    nbytes = limit // 8 + 2
    b = reach.to_bytes(nbytes, "little")
    bits = np.unpackbits(np.frombuffer(b, dtype=np.uint8), bitorder="little")
    pts = np.nonzero(bits[1: limit + 1])[0] + 1
    return pts.tolist()


def set_points(pairs, limit, maxsize=60000):
    pts = {0}
    for s, c in pairs:
        if s > limit or c <= 0:
            continue
        frontier = pts
        for _ in range(c):
            frontier = {p + s for p in frontier if p + s <= limit}
            frontier -= pts
            if not frontier:
                break
            pts |= frontier
            if len(pts) > maxsize:
                break
        if len(pts) > maxsize:
            break
    pts.discard(0)
    return sorted(pts)


def thin_points(pts, keep_vals, cap):
    pts = sorted(pts)
    if len(pts) <= cap:
        return pts
    ptset = set(pts)
    keep = sorted((set(keep_vals) & ptset) | {pts[-1]})
    if len(keep) >= cap:
        idx = np.linspace(0, len(keep) - 1, cap).astype(int)
        sel = {keep[i] for i in idx}
        sel.add(pts[-1])
        return sorted(sel)
    others = sorted(ptset - set(keep))
    m = cap - len(keep)
    if m > 0 and others:
        idx = np.linspace(0, len(others) - 1, min(m, len(others))).astype(int)
        keep = sorted(set(keep) | {others[i] for i in idx})
    return sorted(set(keep))


# ----------------------------------------------------------------------
# Guillotine DP (unconstrained w.r.t. copies; copies handled by pricing)
# ----------------------------------------------------------------------
def run_dp(F0, nL, nW, kL, cL, ysW, cW, deadline):
    F = F0.copy()
    dirr = np.zeros((nL, nW), dtype=np.int8)
    arg = np.full((nL, nW), -1, dtype=np.int32)
    for i in range(nL):
        if time.time() > deadline:
            return None
        ks = kL[i]
        if ks.size:
            cand = F[ks, :] + F[cL[i], :]
            cmax = cand.max(axis=0)
            carg = cand.argmax(axis=0)
            upd = cmax > F[i]
            if upd.any():
                F[i][upd] = cmax[upd]
                dirr[i][upd] = 1
                arg[i][upd] = ks[carg[upd]].astype(np.int32)
        if i:
            upd = F[i - 1] > F[i]
            if upd.any():
                F[i][upd] = F[i - 1][upd]
                dirr[i][upd] = 3
                arg[i][upd] = i - 1
        row = F[i]
        drow = dirr[i]
        arow = arg[i]
        for j in range(nW):
            ys = ysW[j]
            if ys.size:
                cand = row[ys] + row[cW[j]]
                m = int(cand.argmax())
                v = cand[m]
                if v > row[j]:
                    row[j] = v
                    drow[j] = 2
                    arow[j] = int(ys[m])
            if j and row[j - 1] > row[j]:
                row[j] = row[j - 1]
                drow[j] = 4
                arow[j] = j - 1
    return F, dirr, arg


def extract(F, dirr, arg, Pl, Ql, bid, order_profit, ilen, iwid, iprof, icop):
    """Walk the DP cut tree; enforce copy limits greedily -> feasible sol."""
    nL, nW = len(Pl), len(Ql)
    remaining = list(icop)
    wanted = [0] * len(icop)
    taken = defaultdict(int)
    obj = 0.0
    stack = [(nL - 1, nW - 1)]
    while stack:
        i, j = stack.pop()
        d = dirr[i, j]
        if d == 1:
            k = int(arg[i, j])
            c = bisect.bisect_right(Pl, Pl[i] - Pl[k]) - 1
            stack.append((k, j))
            if c >= 0:
                stack.append((c, j))
        elif d == 2:
            y = int(arg[i, j])
            c = bisect.bisect_right(Ql, Ql[j] - Ql[y]) - 1
            stack.append((i, y))
            if c >= 0:
                stack.append((i, c))
        elif d == 3:
            stack.append((i - 1, j))
        elif d == 4:
            stack.append((i, j - 1))
        else:
            t = int(bid[i, j])
            if t >= 0 and F[i, j] > 1e-9:
                wanted[t] += 1
            pl, pw = Pl[i], Ql[j]
            for tt in order_profit:
                if remaining[tt] > 0 and ilen[tt] <= pl and iwid[tt] <= pw:
                    remaining[tt] -= 1
                    taken[tt] += 1
                    obj += iprof[tt]
                    break
    return obj, taken, wanted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", type=str, required=True)
    ap.add_argument("--solution_path", type=str, required=True)
    ap.add_argument("--time_limit", type=int, default=60)
    ap.add_argument("--log_path", type=str, default=None)
    args = ap.parse_args()

    t0 = time.time()
    deadline = t0 + max(2, args.time_limit) - 1.0
    logger = SolutionLogger(args.log_path, sense="maximize") if (args.log_path and SolutionLogger) else None

    with open(args.instance_path) as f:
        data = json.load(f)

    L = int(data["panel"]["length"])
    W = int(data["panel"]["width"])

    items = []
    for it in data["items"]:
        l, w = int(it["length"]), int(it["width"])
        p, c = int(it["profit"]), int(it["copies"])
        if l <= L and w <= W and p > 0 and c > 0:
            items.append((int(it["id"]), l, w, p, c))
    n = len(items)
    ids = [it[0] for it in items]
    ilen = [it[1] for it in items]
    iwid = [it[2] for it in items]
    iprof = [it[3] for it in items]
    icop = [min(it[4], (L // it[1]) * (W // it[2])) for it in items]

    best = {"obj": 0.0, "sel": {}}

    def make_solution(obj, sel):
        return {
            "objective_value": float(obj),
            "items_selected": [
                {"item_id": int(ids[t]), "plate_dims": [int(ilen[t]), int(iwid[t])], "copies": int(c)}
                for t, c in sorted(sel.items()) if c > 0
            ],
        }

    def emit(obj, sel):
        if obj > best["obj"] + 1e-9:
            best["obj"] = obj
            best["sel"] = dict(sel)
            if logger:
                logger.log_solution(float(obj), make_solution(obj, sel))

    def finish():
        with open(args.solution_path, "w") as f:
            json.dump(make_solution(best["obj"], best["sel"]), f, indent=2)

    if n == 0:
        finish()
        return

    # ------------------------------------------------------------------
    # Quick greedy fallbacks (grid of one type, shelf packing)
    # ------------------------------------------------------------------
    for t in range(n):
        cnt = min(icop[t], (L // ilen[t]) * (W // iwid[t]))
        if cnt > 0:
            emit(cnt * iprof[t], {t: cnt})

    order_dens = sorted(range(n), key=lambda t: -iprof[t] / (ilen[t] * iwid[t]))
    rem = list(icop)
    y = 0
    sel = defaultdict(int)
    obj = 0.0
    while True:
        starter = None
        for t in order_dens:
            if rem[t] > 0 and iwid[t] <= W - y:
                starter = t
                break
        if starter is None:
            break
        h = iwid[starter]
        x = 0
        for t in order_dens:
            if iwid[t] <= h:
                while rem[t] > 0 and x + ilen[t] <= L:
                    x += ilen[t]
                    rem[t] -= 1
                    sel[t] += 1
                    obj += iprof[t]
        y += h
    emit(obj, sel)

    # ------------------------------------------------------------------
    # Discretization
    # ------------------------------------------------------------------
    tl = args.time_limit
    cap = 250 if tl < 15 else (350 if tl < 45 else (480 if tl < 150 else 600))

    pairsL = sorted(set((ilen[t], icop[t]) for t in range(n)))
    pairsW = sorted(set((iwid[t], icop[t]) for t in range(n)))
    try:
        if L <= 4_000_000:
            Praw = bit_points(pairsL, L)
        else:
            Praw = set_points(pairsL, L)
        if W <= 4_000_000:
            Qraw = bit_points(pairsW, W)
        else:
            Qraw = set_points(pairsW, W)
    except Exception:
        Praw = set_points(pairsL, L)
        Qraw = set_points(pairsW, W)

    Pl = thin_points(Praw, ilen, cap)
    Ql = thin_points(Qraw, iwid, cap)
    if not Pl or not Ql:
        finish()
        return

    P = np.array(Pl, dtype=np.int64)
    Q = np.array(Ql, dtype=np.int64)
    nL, nW = len(Pl), len(Ql)

    # cut candidates (x <= l/2) and complements rounded down to points
    kL, cL = [], []
    for i in range(nL):
        cnt = int(np.searchsorted(P, Pl[i] // 2, side="right"))
        ks = np.arange(cnt, dtype=np.int64)
        comp = np.searchsorted(P, Pl[i] - P[ks], side="right") - 1
        kL.append(ks)
        cL.append(comp.astype(np.int64))
    ysW, cW = [], []
    for j in range(nW):
        cnt = int(np.searchsorted(Q, Ql[j] // 2, side="right"))
        ys = np.arange(cnt, dtype=np.int64)
        comp = np.searchsorted(Q, Ql[j] - Q[ys], side="right") - 1
        ysW.append(ys)
        cW.append(comp.astype(np.int64))

    order_profit = sorted(range(n), key=lambda t: -iprof[t])
    prof = np.array(iprof, dtype=np.float64)
    cop = np.array(icop, dtype=np.int64)
    lam = np.zeros(n, dtype=np.float64)
    rng = np.random.default_rng(0)

    # precompute item -> grid index offsets
    i0s = [bisect.bisect_left(Pl, ilen[t]) for t in range(n)]
    j0s = [bisect.bisect_left(Ql, iwid[t]) for t in range(n)]

    def build_F0(mp):
        F0 = np.zeros((nL, nW), dtype=np.float64)
        bid = np.full((nL, nW), -1, dtype=np.int32)
        for t in np.argsort(mp):
            t = int(t)
            v = mp[t]
            if v <= 1e-9:
                continue
            i0, j0 = i0s[t], j0s[t]
            if i0 >= nL or j0 >= nW:
                continue
            sub = F0[i0:, j0:]
            msk = sub < v
            if msk.any():
                sub[msk] = v
                bid[i0:, j0:][msk] = t
        return F0, bid

    # ------------------------------------------------------------------
    # Lagrangian-pricing loop around the guillotine DP
    # ------------------------------------------------------------------
    ub = None
    it = 0
    last_dur = 0.0
    while True:
        now = time.time()
        if now + max(last_dur * 1.15, 0.5) > deadline:
            break
        t_iter = now
        mp = prof - lam
        if it > 0:
            mp = mp - rng.random(n) * (1e-7 * prof)  # tiny tie-breaking noise
        F0, bid = build_F0(mp)
        res = run_dp(F0, nL, nW, kL, cL, ysW, cW, deadline)
        if res is None:
            break
        F, dirr, arg = res
        obj, taken, wanted = extract(F, dirr, arg, Pl, Ql, bid,
                                     order_profit, ilen, iwid, iprof, icop)
        emit(obj, taken)
        if it == 0:
            ub = float(F[-1, -1])
        feasible_dp = all(wanted[t] <= icop[t] for t in range(n))
        if it == 0 and feasible_dp:
            break  # DP solution already respects copy limits -> done
        if ub is not None and best["obj"] >= ub - 1e-6:
            break
        # subgradient-style multiplier update
        alpha = 0.4 / (1.0 + 0.3 * it)
        for t in range(n):
            over = wanted[t] - icop[t]
            if over > 0:
                step = alpha * prof[t] * min(2.0, over / max(1, icop[t]) + 0.25)
                lam[t] = min(prof[t], lam[t] + step)
            elif wanted[t] < icop[t] and lam[t] > 0:
                lam[t] = max(0.0, lam[t] - 0.5 * alpha * prof[t])
        it += 1
        last_dur = time.time() - t_iter

    finish()


if __name__ == "__main__":
    main()
import argparse
import json
import math
import random
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--instance_path', required=True)
    ap.add_argument('--solution_path', required=True)
    ap.add_argument('--time_limit', type=int, required=True)
    ap.add_argument('--log_path', default=None)
    args = ap.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path) as f:
        inst = json.load(f)

    n = int(inst['n'])
    m = int(inst['m'])
    jobs = inst['jobs']
    ids = [j['id'] for j in jobs]
    p = [int(j['p']) for j in jobs]
    d = [int(j['d']) for j in jobs]
    w = [int(j['w']) for j in jobs]

    t0 = time.time()
    TL = max(1, args.time_limit)
    deadline = t0 + TL - 0.3
    rng = random.Random(0)

    pbar = sum(p) / max(1, n)

    def machine_obj(seq):
        t = 0
        o = 0
        for jb in seq:
            t += p[jb]
            r = t - d[jb]
            if r > 0:
                o += w[jb] * r
        return o

    def total_obj(sched):
        return sum(machine_obj(s) for s in sched)

    def build_solution(sched):
        sol = {}
        for seq in sched:
            t = 0
            for jb in seq:
                sol[str(ids[jb])] = t
                t += p[jb]
        return sol

    # ---------- Construction heuristics ----------
    def atc(K):
        sched = [[] for _ in range(m)]
        mt = [0] * m
        un = set(range(n))
        kp = K * pbar
        while un:
            i = min(range(m), key=lambda z: mt[z])
            t = mt[i]
            bj = None
            bs = -1.0
            for jb in un:
                sl = d[jb] - p[jb] - t
                if sl < 0:
                    sl = 0
                sc = (w[jb] / p[jb]) * math.exp(-sl / kp)
                if sc > bs:
                    bs = sc
                    bj = jb
            sched[i].append(bj)
            mt[i] += p[bj]
            un.discard(bj)
        return sched

    def list_sched(order):
        sched = [[] for _ in range(m)]
        mt = [0] * m
        for jb in order:
            i = min(range(m), key=lambda z: mt[z])
            sched[i].append(jb)
            mt[i] += p[jb]
        return sched

    best_obj = None
    best_sched = None

    # ---------- Local search (relocation + swap, O(L) delta evaluation) ----------
    def local_search(sched, dl):
        nonlocal best_obj, best_sched
        pref = [None] * m
        loc = [None] * n
        mobj = [0] * m

        def recompute(i):
            seq = sched[i]
            pr = [0] * (len(seq) + 1)
            for k, jb in enumerate(seq):
                pr[k + 1] = pr[k] + p[jb]
                loc[jb] = (i, k)
            pref[i] = pr
            mobj[i] = machine_obj(seq)

        for i in range(m):
            recompute(i)

        def shift_delta(i, lo, hi, sh):
            seq = sched[i]
            pr = pref[i]
            s = 0
            for k in range(lo, hi):
                jb = seq[k]
                old = pr[k + 1] - d[jb]
                if old > 0:
                    s -= w[jb] * old
                new = old + sh
                if new > 0:
                    s += w[jb] * new
            return s

        POS_CAP = 80
        SWAP_CAP = 250
        tot = sum(mobj)
        order = list(range(n))
        improved = True
        while improved:
            improved = False
            if time.time() > dl:
                break
            rng.shuffle(order)
            for x in order:
                if time.time() > dl:
                    return tot
                i, a = loc[x]
                px = p[x]; dx = d[x]; wx = w[x]
                pr_i = pref[i]
                oldCx = pr_i[a + 1]
                oldTx = wx * (oldCx - dx) if oldCx > dx else 0
                Li = len(sched[i])
                best_delta = -1e-9
                best = None

                # Relocation moves of x to any machine/position
                for j in range(m):
                    pr_j = pref[j]
                    Lj = len(sched[j])
                    if j == i:
                        for b in range(Lj):
                            if b == a:
                                continue
                            if b < a:
                                Cn = pr_j[b] + px
                                delta = (wx * (Cn - dx) if Cn > dx else 0) - oldTx \
                                        + shift_delta(i, b, a, px)
                            else:
                                Cn = pr_j[b + 1]
                                delta = (wx * (Cn - dx) if Cn > dx else 0) - oldTx \
                                        + shift_delta(i, a + 1, b + 1, -px)
                            if delta < best_delta:
                                best_delta = delta
                                best = (0, j, b)
                    else:
                        rem = -oldTx + shift_delta(i, a + 1, Li, -px)
                        if Lj + 1 <= POS_CAP:
                            positions = range(Lj + 1)
                        else:
                            positions = rng.sample(range(Lj + 1), POS_CAP)
                        for b in positions:
                            Cn = pr_j[b] + px
                            delta = rem + (wx * (Cn - dx) if Cn > dx else 0) \
                                    + shift_delta(j, b, Lj, px)
                            if delta < best_delta:
                                best_delta = delta
                                best = (0, j, b)

                # Swap moves of x with another job y
                if n <= SWAP_CAP:
                    ys = order
                else:
                    ys = rng.sample(order, SWAP_CAP)
                for y in ys:
                    if y == x:
                        continue
                    j, b = loc[y]
                    py = p[y]; dy = d[y]; wy = w[y]
                    if j == i:
                        if a < b:
                            lo, hi = a, b
                            jlo, jhi = x, y
                        else:
                            lo, hi = b, a
                            jlo, jhi = y, x
                        plo = p[jlo]; phi = p[jhi]
                        pr = pr_i
                        Clo = pr[lo + 1]
                        Chi = pr[hi + 1]
                        delta = 0
                        Cn = pr[hi] + phi
                        dj = d[jlo]
                        if Cn > dj:
                            delta += w[jlo] * (Cn - dj)
                        if Clo > dj:
                            delta -= w[jlo] * (Clo - dj)
                        Cn2 = pr[lo] + phi
                        dj2 = d[jhi]
                        if Cn2 > dj2:
                            delta += w[jhi] * (Cn2 - dj2)
                        if Chi > dj2:
                            delta -= w[jhi] * (Chi - dj2)
                        delta += shift_delta(i, lo + 1, hi, phi - plo)
                        if delta < best_delta:
                            best_delta = delta
                            best = (1, j, b)
                    else:
                        pr_j = pref[j]
                        oldCy = pr_j[b + 1]
                        oldTy = wy * (oldCy - dy) if oldCy > dy else 0
                        Cxn = pr_j[b] + px
                        Cyn = pr_i[a] + py
                        delta = ((wx * (Cxn - dx) if Cxn > dx else 0) - oldTx) \
                                + ((wy * (Cyn - dy) if Cyn > dy else 0) - oldTy) \
                                + shift_delta(i, a + 1, Li, py - px) \
                                + shift_delta(j, b + 1, len(sched[j]), px - py)
                        if delta < best_delta:
                            best_delta = delta
                            best = (1, j, b)

                if best is not None:
                    if best[0] == 0:
                        _, j, b = best
                        sched[i].pop(a)
                        sched[j].insert(b, x)
                        recompute(i)
                        if j != i:
                            recompute(j)
                    else:
                        _, j, b = best
                        y = sched[j][b]
                        sched[i][a] = y
                        sched[j][b] = x
                        recompute(i)
                        if j != i:
                            recompute(j)
                    tot += best_delta
                    improved = True
                    if tot < best_obj - 1e-9:
                        best_obj = tot
                        best_sched = [s[:] for s in sched]
                        if logger:
                            logger.log_solution(float(tot), build_solution(sched))
        return tot

    # ---------- Build and score initial solutions ----------
    initials = []
    for K in (0.5, 1.0, 2.0, 4.0):
        initials.append(atc(K))
    initials.append(list_sched(sorted(range(n), key=lambda j: (d[j], -w[j]))))
    initials.append(list_sched(sorted(range(n), key=lambda j: (p[j] / w[j], d[j]))))
    initials.append(list_sched(sorted(range(n), key=lambda j: (d[j] - p[j], -w[j]))))

    scored = sorted(((total_obj(s), idx) for idx, s in enumerate(initials)),
                    key=lambda t: t[0])
    best_obj = scored[0][0]
    best_sched = [s[:] for s in initials[scored[0][1]]]
    if logger:
        logger.log_solution(float(best_obj), build_solution(best_sched))

    # ---------- Local search on the most promising distinct starts ----------
    fracs = [0.2, 0.35, 0.5]
    seen = set()
    runs = []
    for obj, idx in scored:
        key = tuple(tuple(s) for s in initials[idx])
        if key in seen:
            continue
        seen.add(key)
        runs.append(idx)
        if len(runs) >= 3:
            break
    for r, idx in enumerate(runs):
        sd = min(deadline, t0 + TL * fracs[r])
        if time.time() > sd:
            break
        local_search([s[:] for s in initials[idx]], sd)

    # ---------- Iterated local search until the time limit ----------
    def perturb(sched, k):
        ns = [s[:] for s in sched]
        for _ in range(k):
            i = rng.randrange(m)
            while not ns[i]:
                i = rng.randrange(m)
            a = rng.randrange(len(ns[i]))
            x = ns[i].pop(a)
            j = rng.randrange(m)
            b = rng.randrange(len(ns[j]) + 1)
            ns[j].insert(b, x)
        return ns

    fails = 0
    base_k = max(2, n // 10)
    while time.time() < deadline:
        prev = best_obj
        if fails >= 30:
            cs = list_sched(rng.sample(range(n), n))
            fails = 0
        else:
            k = min(max(2, n), base_k + fails // 2)
            cs = perturb(best_sched, k)
        local_search(cs, deadline)
        if best_obj < prev - 1e-9:
            fails = 0
        else:
            fails += 1

    # ---------- Write final solution ----------
    out = {
        "objective_value": float(best_obj),
        "schedule": build_solution(best_sched),
    }
    with open(args.solution_path, 'w') as f:
        json.dump(out, f)


if __name__ == '__main__':
    main()
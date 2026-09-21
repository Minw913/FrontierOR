import argparse
import json
import time
import random
import heapq

from solution_logger import SolutionLogger


class TimeoutErr(Exception):
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", required=True)
    ap.add_argument("--solution_path", required=True)
    ap.add_argument("--time_limit", type=int, default=60)
    ap.add_argument("--log_path", default=None)
    args = ap.parse_args()

    t0 = time.time()
    deadline = t0 + max(1, args.time_limit) - 1.0

    with open(args.instance_path) as f:
        data = json.load(f)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    containers = data["containers"]
    trades_in = data["trades"]
    num_periods = data["problem_parameters"]["num_periods"]

    ids = [c["id"] for c in containers]
    avs = {c["id"]: c["availability_start"] for c in containers}
    ave = {c["id"]: c["availability_end"] for c in containers}
    adj = {c["id"]: list(c.get("adjacent_containers", [])) for c in containers}

    nT = len(trades_in)
    tid = [t["id"] for t in trades_in]
    tstart = [t["start_period"] for t in trades_in]
    tend = [t["end_period"] for t in trades_in]
    tdem = [t["container_demand"] for t in trades_in]
    tdisp = [t["max_dispersion"] for t in trades_in]

    maxP = num_periods
    if ids:
        maxP = max(maxP, max(ave.values()))
    if nT:
        maxP = max(maxP, max(tend))

    # ------------------------------------------------------------------
    # Exact feasibility set builder: build a set of q containers from F
    # forming at most d adjacency-connected components. Returns None iff
    # no such set exists (exact test via top-d region capacity).
    # ------------------------------------------------------------------
    def build_set(F, q, d, nb, rng):
        if q == 0:
            return set()
        if len(F) < q:
            return None
        # connected regions of F
        regions = []
        seen = set()
        for c in F:
            if c in seen:
                continue
            stack = [c]
            seen.add(c)
            comp = []
            while stack:
                u = stack.pop()
                comp.append(u)
                for v in adj[u]:
                    if v in F and v not in seen:
                        seen.add(v)
                        stack.append(v)
            regions.append(comp)
        regions.sort(key=len, reverse=True)
        if sum(len(r) for r in regions[:d]) < q:
            return None
        S = set()
        frontier = set()
        comps_used = 0
        region_idx = 0
        while len(S) < q:
            if frontier:
                best = None
                bs = -1e18
                for c in frontier:
                    seln = 0
                    for v in adj[c]:
                        if v in S:
                            seln += 1
                    sc = 1.0 * seln - 0.02 * nb.get(c, 0) + 0.6 * rng.random()
                    if sc > bs:
                        bs = sc
                        best = c
                c = best
            else:
                while region_idx < len(regions) and regions[region_idx][0] in S:
                    region_idx += 1
                if region_idx >= len(regions) or comps_used >= d:
                    return None
                reg = regions[region_idx]
                best = None
                bs = -1e18
                for u in reg:
                    dg = 0
                    for v in adj[u]:
                        if v in F:
                            dg += 1
                    sc = dg + rng.random()
                    if sc > bs:
                        bs = sc
                        best = u
                c = best
                comps_used += 1
            S.add(c)
            frontier.discard(c)
            for v in adj[c]:
                if v in F and v not in S:
                    frontier.add(v)
        return S

    def fallback_pick(F0, q, rng):
        seen = set()
        regions = []
        for c in F0:
            if c in seen:
                continue
            stack = [c]
            seen.add(c)
            comp = []
            while stack:
                u = stack.pop()
                comp.append(u)
                for v in adj[u]:
                    if v in F0 and v not in seen:
                        seen.add(v)
                        stack.append(v)
            regions.append(comp)
        regions.sort(key=len, reverse=True)
        S = []
        for reg in regions:
            for u in reg:
                if len(S) < q:
                    S.append(u)
                else:
                    break
        return frozenset(S)

    # ------------------------------------------------------------------
    # Plan one maximal-length segment for trade i starting at period p,
    # given current reservations. Returns (end_period, frozenset) or None.
    # ------------------------------------------------------------------
    def plan_segment(i, p, prev, reserved, rng):
        q = tdem[i]
        d = tdisp[i]
        cap = tend[i]
        if q == 0:
            return cap, frozenset()
        nb = {}
        for c in ids:
            if avs[c] <= p <= ave[c] and c not in reserved[p]:
                lim = min(ave[c], cap)
                t = p + 1
                while t <= lim and c not in reserved[t]:
                    t += 1
                nb[c] = t  # first blocked period after p (capped)
        if len(nb) < q:
            return None
        # Try to keep the previous set unchanged (zero cost) as long as possible.
        if prev is not None and len(prev) == q:
            ok = True
            for c in prev:
                if c not in nb:
                    ok = False
                    break
            if ok:
                e = min(cap, min(nb[c] for c in prev) - 1)
                return e, frozenset(prev)

        def feas(end):
            Fe = {c for c, v in nb.items() if v > end}
            return build_set(Fe, q, d, nb, rng)

        s0 = feas(p)
        if s0 is None:
            return None
        cand = sorted({min(v - 1, cap) for v in nb.values()})
        s = feas(cand[-1])
        if s is not None:
            return cand[-1], frozenset(s)
        best_e, best_s = p, s0
        lo, hi = 0, len(cand) - 2
        while lo <= hi:
            mid = (lo + hi) // 2
            s = feas(cand[mid])
            if s is not None:
                best_e, best_s = cand[mid], s
                lo = mid + 1
            else:
                hi = mid - 1
        return best_e, frozenset(best_s)

    # ------------------------------------------------------------------
    def construct(rng, hard_deadline):
        reserved = [set() for _ in range(maxP + 2)]
        owner = [dict() for _ in range(maxP + 2)]
        assign = [dict() for _ in range(nT)]
        next_needed = [tstart[i] for i in range(nT)]
        heap = [(tstart[i], rng.random(), i) for i in range(nT)]
        heapq.heapify(heap)
        budget = 50 + 12 * nT
        infeas = False
        while heap:
            if hard_deadline is not None and time.time() > hard_deadline:
                raise TimeoutErr()
            p, r, i = heapq.heappop(heap)
            if p != next_needed[i] or p > tend[i]:
                continue
            prev = assign[i].get(p - 1)
            res = plan_segment(i, p, prev, reserved, rng)
            if res is None:
                owners_at_p = set(owner[p].values())
                owners_at_p.discard(i)
                if owners_at_p and budget > 0:
                    budget -= 1
                    j = rng.choice(sorted(owners_at_p))
                    for t in range(p, tend[j] + 1):
                        St = assign[j].pop(t, None)
                        if St:
                            for c in St:
                                reserved[t].discard(c)
                                owner[t].pop(c, None)
                    next_needed[j] = p
                    heapq.heappush(heap, (p, 1.0 + rng.random(), j))
                    heapq.heappush(heap, (p, -1.0, i))
                    continue
                else:
                    F0 = {c for c in ids
                          if avs[c] <= p <= ave[c] and c not in reserved[p]}
                    S = fallback_pick(F0, tdem[i], rng)
                    e = p
                    infeas = True
            else:
                e, S = res
            for t in range(p, e + 1):
                assign[i][t] = S
                reserved[t].update(S)
                for c in S:
                    owner[t][c] = i
            next_needed[i] = e + 1
            if e + 1 <= tend[i]:
                heapq.heappush(heap, (e + 1, rng.random(), i))
        return assign, reserved, owner, infeas

    def changes(assign, i):
        cnt = 0
        ai = assign[i]
        for t in range(tstart[i] + 1, tend[i] + 1):
            if ai.get(t) != ai.get(t - 1):
                cnt += 1
        return cnt

    # ------------------------------------------------------------------
    # Local improvement: re-optimize each trade solo given the others fixed.
    # ------------------------------------------------------------------
    def reopt(assign, reserved, owner, rng, dl):
        improved = True
        while improved:
            improved = False
            order = list(range(nT))
            rng.shuffle(order)
            for i in order:
                if time.time() > dl:
                    return
                oc = changes(assign, i)
                if oc == 0:
                    continue
                saved = assign[i]
                for t, S in saved.items():
                    for c in S:
                        reserved[t].discard(c)
                        owner[t].pop(c, None)
                new = {}
                prev = None
                p = tstart[i]
                ok = True
                while p <= tend[i]:
                    res = plan_segment(i, p, prev, reserved, rng)
                    if res is None:
                        ok = False
                        break
                    e, S = res
                    for t in range(p, e + 1):
                        new[t] = S
                    prev = S
                    p = e + 1
                nc = None
                if ok:
                    assign[i] = new
                    nc = changes(assign, i)
                if ok and nc < oc:
                    for t, S in new.items():
                        reserved[t].update(S)
                        for c in S:
                            owner[t][c] = i
                    improved = True
                else:
                    assign[i] = saved
                    for t, S in saved.items():
                        reserved[t].update(S)
                        for c in S:
                            owner[t][c] = i

    # ------------------------------------------------------------------
    # Lower bound: each trade solo with no other reservations.
    # ------------------------------------------------------------------
    def solo_lb():
        rng_lb = random.Random(12345)
        empty = [set() for _ in range(maxP + 2)]
        lb = 0
        for i in range(nT):
            new = {}
            prev = None
            p = tstart[i]
            ok = True
            while p <= tend[i]:
                res = plan_segment(i, p, prev, empty, rng_lb)
                if res is None:
                    ok = False
                    break
                e, S = res
                for t in range(p, e + 1):
                    new[t] = S
                prev = S
                p = e + 1
            if ok:
                cnt = 0
                for t in range(tstart[i] + 1, tend[i] + 1):
                    if new.get(t) != new.get(t - 1):
                        cnt += 1
                lb += cnt
        return lb

    # ------------------------------------------------------------------
    # Very fast period-by-period emergency construction.
    # ------------------------------------------------------------------
    def emergency(rng):
        assign = [dict() for _ in range(nT)]
        infeas = False
        lo = min(tstart) if nT else 1
        hi = max(tend) if nT else 0
        nb0 = {}
        for t in range(lo, hi + 1):
            resv = set()
            active = [i for i in range(nT) if tstart[i] <= t <= tend[i]]
            keepers = []
            builders = []
            for i in active:
                prev = assign[i].get(t - 1)
                if prev is not None and all(
                        avs[c] <= t <= ave[c] for c in prev):
                    keepers.append(i)
                else:
                    builders.append(i)
            for i in keepers:
                prev = assign[i][t - 1]
                if not (prev & resv):
                    assign[i][t] = prev
                    resv.update(prev)
                else:
                    builders.append(i)
            for i in builders:
                F = {c for c in ids
                     if avs[c] <= t <= ave[c] and c not in resv}
                S = build_set(F, tdem[i], tdisp[i], nb0, rng)
                if S is None:
                    S = fallback_pick(F, tdem[i], rng)
                    infeas = True
                S = frozenset(S)
                assign[i][t] = S
                resv.update(S)
        return assign, infeas

    def make_sol(assign, obj):
        out = {}
        for i in range(nT):
            per = {}
            for t in range(tstart[i], tend[i] + 1):
                per[str(t)] = sorted(assign[i].get(t, frozenset()))
            out[str(tid[i])] = per
        return {"objective_value": float(obj), "assignments": out}

    # ------------------------------------------------------------------
    # Main search loop: randomized restarts + local improvement.
    # ------------------------------------------------------------------
    rng = random.Random(0)
    lb = 0
    try:
        lb = solo_lb()
    except Exception:
        lb = 0

    best_obj = None
    best_sol = None
    best_feas = False
    attempt = 0

    while True:
        try:
            assign, reserved, owner, infeas = construct(rng, deadline)
            if not infeas and time.time() < deadline:
                reopt(assign, reserved, owner, rng, deadline)
        except TimeoutErr:
            if attempt == 0 and best_sol is None:
                try:
                    assign, infeas = emergency(rng)
                    obj = sum(changes(assign, i) for i in range(nT))
                    best_obj = obj
                    best_feas = not infeas
                    best_sol = make_sol(assign, obj)
                    if best_feas and logger:
                        logger.log_solution(float(obj), best_sol)
                except Exception:
                    pass
            break

        obj = sum(changes(assign, i) for i in range(nT))
        feas = not infeas
        take = False
        if best_sol is None:
            take = True
        elif feas and not best_feas:
            take = True
        elif feas == best_feas and obj < best_obj:
            take = True
        if take:
            best_obj = obj
            best_feas = feas
            best_sol = make_sol(assign, obj)
            if feas and logger:
                logger.log_solution(float(obj), best_sol)

        attempt += 1
        if best_feas and best_obj is not None and best_obj <= lb:
            break
        if time.time() >= deadline:
            break

    if best_sol is None:
        # absolute last resort: empty structure respecting schema shape
        try:
            assign, infeas = emergency(rng)
            obj = sum(changes(assign, i) for i in range(nT))
            best_sol = make_sol(assign, obj)
            if not infeas and logger:
                logger.log_solution(float(obj), best_sol)
        except Exception:
            best_sol = {"objective_value": 0.0, "assignments": {
                str(tid[i]): {str(t): [] for t in range(tstart[i], tend[i] + 1)}
                for i in range(nT)}}

    with open(args.solution_path, "w") as f:
        json.dump(best_sol, f)


if __name__ == "__main__":
    main()
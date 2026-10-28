import argparse
import json
import math
import random
import time
from array import array
from bisect import bisect_right

from solution_logger import SolutionLogger


class TDTSPSolver:
    """Time-dependent TSP solver.

    - Exact subset DP (using the FIFO / non-decreasing arrival property) for
      small instances.
    - Time-dependent nearest-neighbour construction + candidate-list local
      search (relocate, or-opt, swap, 2-opt) + iterated local search for
      larger instances.
    """

    def __init__(self, data):
        params = data["parameters"]
        self.H = int(params["H"])
        self.bounds = [float(x) for x in params["period_boundaries"]]
        self.T = float(params.get("time_horizon_T", self.bounds[-1]))
        self.depot = int(data.get("depot", {}).get("index", 0))
        arcs = data["arcs"]
        self.dist = [[float(x) for x in row] for row in arcs["distances"]]
        self.n = int(params.get("total_vertices") or len(self.dist))
        sp = arcs.get("speeds_v_ijh")
        if sp is not None:
            self.v = [[tuple(float(x) for x in vij) for vij in vi] for vi in sp]
        else:
            delta = arcs["delta_ijh"]
            u = arcs["max_speeds_u_ij"]
            bh = [float(x) for x in params["b_h"]]
            H = self.H
            self.v = [
                [tuple(float(delta[i][j][h]) * bh[h] * float(u[i][j])
                       for h in range(H))
                 for j in range(self.n)]
                for i in range(self.n)
            ]
        self.customers = [i for i in range(self.n) if i != self.depot]
        self.nc = len(self.customers)
        self.cand = None

    # ---------------- time-dependent travel time ----------------
    def arc_arrival(self, i, j, t):
        d = self.dist[i][j]
        if d <= 0.0:
            return t
        bounds = self.bounds
        vlist = self.v[i][j]
        H = self.H
        if t >= bounds[H]:
            v = vlist[H - 1]
            return (t + d / v) if v > 1e-12 else float("inf")
        h = bisect_right(bounds, t) - 1
        if h < 0:
            h = 0
        rem = d
        while True:
            v = vlist[h]
            e = bounds[h + 1]
            if v > 1e-12:
                a = t + rem / v
                if a <= e:
                    return a
                rem -= v * (e - t)
                if rem <= 1e-9:
                    return e
            t = e
            h += 1
            if h >= H:
                v = vlist[H - 1]
                return (t + rem / v) if v > 1e-12 else float("inf")

    def eval_tail(self, route, k0, t):
        """Simulate route from position k0 (departing route[k0] at time t)."""
        D = self.dist
        V = self.v
        bounds = self.bounds
        H = self.H
        bH = bounds[H]
        _bisect = bisect_right
        for k in range(k0, len(route) - 1):
            i = route[k]
            j = route[k + 1]
            d = D[i][j]
            if d > 0.0:
                vlist = V[i][j]
                if t >= bH:
                    v = vlist[H - 1]
                    t = (t + d / v) if v > 1e-12 else float("inf")
                    continue
                h = _bisect(bounds, t) - 1
                if h < 0:
                    h = 0
                rem = d
                while True:
                    v = vlist[h]
                    e = bounds[h + 1]
                    if v > 1e-12:
                        a = t + rem / v
                        if a <= e:
                            t = a
                            break
                        rem -= v * (e - t)
                        if rem <= 1e-9:
                            t = e
                            break
                    t = e
                    h += 1
                    if h >= H:
                        v = vlist[H - 1]
                        t = (t + rem / v) if v > 1e-12 else float("inf")
                        break
        return t

    def eval_full(self, route):
        """Full evaluation, returns (total_duration, arrival_times)."""
        D = self.dist
        V = self.v
        bounds = self.bounds
        H = self.H
        bH = bounds[H]
        _bisect = bisect_right
        arrivals = [0.0] * len(route)
        t = 0.0
        for k in range(len(route) - 1):
            i = route[k]
            j = route[k + 1]
            d = D[i][j]
            if d > 0.0:
                vlist = V[i][j]
                if t >= bH:
                    v = vlist[H - 1]
                    t = (t + d / v) if v > 1e-12 else float("inf")
                    arrivals[k + 1] = t
                    continue
                h = _bisect(bounds, t) - 1
                if h < 0:
                    h = 0
                rem = d
                while True:
                    v = vlist[h]
                    e = bounds[h + 1]
                    if v > 1e-12:
                        a = t + rem / v
                        if a <= e:
                            t = a
                            break
                        rem -= v * (e - t)
                        if rem <= 1e-9:
                            t = e
                            break
                    t = e
                    h += 1
                    if h >= H:
                        v = vlist[H - 1]
                        t = (t + rem / v) if v > 1e-12 else float("inf")
                        break
            arrivals[k + 1] = t
        return t, arrivals

    def eff(self, obj):
        """Search objective: duration + large penalty if beyond horizon."""
        return obj if obj <= self.T else obj + 1000.0 * (obj - self.T)

    # ---------------- construction ----------------
    def build_candidates(self):
        n = self.n
        k = 20 if n <= 120 else (15 if n <= 400 else 10)
        kk = max(1, min(k, n - 1))
        cand = []
        for i in range(n):
            row = self.dist[i]
            order = sorted(range(n), key=row.__getitem__)
            cand.append([j for j in order if j != i][:kk])
        return cand

    def nn_construct(self, deadline, rng=None):
        d0 = self.depot
        unvisited = set(self.customers)
        route = [d0]
        t = 0.0
        cur = d0
        aa = self.arc_arrival
        now = time.time
        while unvisited:
            if rng is not None and len(unvisited) >= 4:
                scored = sorted((aa(cur, j, t), j) for j in unvisited)
                t, nxt = scored[rng.randrange(3)]
            else:
                bt = float("inf")
                nxt = -1
                for j in unvisited:
                    a = aa(cur, j, t)
                    if a < bt or nxt < 0:
                        bt = a
                        nxt = j
                t = bt
            route.append(nxt)
            unvisited.remove(nxt)
            cur = nxt
            if (len(route) & 15) == 0 and now() > deadline:
                route.extend(sorted(unvisited, key=self.dist[cur].__getitem__))
                break
        route.append(d0)
        return route

    # ---------------- local search ----------------
    def local_search(self, route, deadline):
        route = route[:]
        obj, arrivals = self.eval_full(route)
        ec = self.eff(obj)
        cand = self.cand
        depot = self.depot
        n = self.n
        et = self.eval_tail
        ef = self.eval_full
        eff = self.eff
        now = time.time
        eps = 1e-9
        improved = True
        while improved:
            improved = False
            if now() > deadline:
                break
            L = len(route)
            pos = [0] * n
            for idx in range(L):
                pos[route[idx]] = idx

            # ---------- relocate (single node) ----------
            moved = False
            for p in range(1, L - 1):
                if (p & 7) == 1 and now() > deadline:
                    return route, obj
                a = route[p]
                for b in cand[a]:
                    if b == depot:
                        continue
                    q = pos[b]
                    if q == p - 1 or q >= L - 1:
                        continue
                    base = route[:p] + route[p + 1:]
                    ins = q + 1 if q < p else q
                    nr = base[:ins] + [a] + base[ins:]
                    fd = p if p < ins else ins
                    st = fd - 1
                    no = et(nr, st, arrivals[st])
                    ne = eff(no)
                    if ne < ec - eps:
                        route = nr
                        obj, arrivals = ef(nr)
                        ec = ne
                        moved = True
                        break
                if moved:
                    break
            if moved:
                improved = True
                continue

            # ---------- or-opt (segments of length 2 and 3) ----------
            for sl in (2, 3):
                if moved:
                    break
                for p in range(1, L - sl):
                    if (p & 7) == 1 and now() > deadline:
                        return route, obj
                    seg = route[p:p + sl]
                    for b in cand[seg[0]]:
                        if b == depot:
                            continue
                        q0 = pos[b]
                        if p <= q0 < p + sl:
                            continue
                        base = route[:p] + route[p + sl:]
                        ins = q0 + 1 if q0 < p else q0 - sl + 1
                        if ins == p:
                            continue
                        nr = base[:ins] + seg + base[ins:]
                        fd = p if p < ins else ins
                        st = fd - 1
                        no = et(nr, st, arrivals[st])
                        ne = eff(no)
                        if ne < ec - eps:
                            route = nr
                            obj, arrivals = ef(nr)
                            ec = ne
                            moved = True
                            break
                    if moved:
                        break
            if moved:
                improved = True
                continue

            # ---------- swap ----------
            for p in range(1, L - 2):
                if (p & 7) == 1 and now() > deadline:
                    return route, obj
                a = route[p]
                for b in cand[a]:
                    if b == depot:
                        continue
                    q = pos[b]
                    if q <= p or q >= L - 1:
                        continue
                    nr = route[:]
                    nr[p] = b
                    nr[q] = a
                    st = p - 1
                    no = et(nr, st, arrivals[st])
                    ne = eff(no)
                    if ne < ec - eps:
                        route = nr
                        obj, arrivals = ef(nr)
                        ec = ne
                        moved = True
                        break
                if moved:
                    break
            if moved:
                improved = True
                continue

            # ---------- 2-opt ----------
            for p in range(0, L - 2):
                if (p & 7) == 1 and now() > deadline:
                    return route, obj
                a = route[p]
                for b in cand[a]:
                    if b == depot:
                        continue
                    q = pos[b]
                    if q <= p + 1 or q >= L - 1:
                        continue
                    nr = route[:p + 1] + route[p + 1:q + 1][::-1] + route[q + 1:]
                    no = et(nr, p, arrivals[p])
                    ne = eff(no)
                    if ne < ec - eps:
                        route = nr
                        obj, arrivals = ef(nr)
                        ec = ne
                        moved = True
                        break
                if moved:
                    break
            if moved:
                improved = True
                continue

        return route, obj

    # ---------------- perturbation / ILS ----------------
    def perturb(self, route, rng):
        d0 = self.depot
        L = len(route)
        if L <= 4:
            return route[:]
        if rng.random() < 0.15:
            return [d0] + route[1:-1][::-1] + [d0]
        r = route[:]
        if L > 7:
            a, b, c = sorted(rng.sample(range(2, L - 2), 3))
            s1 = r[1:a]
            segs = [r[a:b], r[b:c], r[c:L - 1]]
            rng.shuffle(segs)
            r = [d0] + s1 + segs[0] + segs[1] + segs[2] + [d0]
        else:
            p = rng.randrange(1, L - 1)
            q = rng.randrange(1, L - 1)
            r[p], r[q] = r[q], r[p]
        for _ in range(rng.randrange(1, 4)):
            L2 = len(r)
            if L2 <= 3:
                break
            p = rng.randrange(1, L2 - 1)
            node = r[p]
            base = r[:p] + r[p + 1:]
            q = rng.randrange(1, len(base))
            r = base[:q] + [node] + base[q:]
        return r

    def ils(self, best_route, best_obj, deadline, logger, rng):
        cur_route, cur_obj = best_route, best_obj
        best_eff = self.eff(best_obj)
        stall = 0
        while time.time() < deadline:
            pr = self.perturb(cur_route, rng)
            r2, o2 = self.local_search(pr, deadline)
            e2 = self.eff(o2)
            if e2 < best_eff - 1e-9:
                best_route, best_obj, best_eff = r2, o2, e2
                cur_route, cur_obj = r2, o2
                self.log_best(logger, best_obj, best_route)
                stall = 0
            else:
                stall += 1
                if e2 <= best_eff * 1.01:
                    cur_route, cur_obj = r2, o2
                else:
                    cur_route, cur_obj = best_route, best_obj
                if stall >= 50:
                    stall = 0
                    rr = self.nn_construct(deadline, rng=rng)
                    cur_route, cur_obj = self.local_search(rr, deadline)
                    ce = self.eff(cur_obj)
                    if ce < best_eff - 1e-9:
                        best_route, best_obj, best_eff = cur_route, cur_obj, ce
                        self.log_best(logger, best_obj, best_route)
        return best_route, best_obj

    # ---------------- exact DP (small instances) ----------------
    def solve_exact(self, deadline):
        C = self.customers
        nc = self.nc
        d0 = self.depot
        if nc == 0:
            return [d0, d0], 0.0
        size = 1 << nc
        INF = float("inf")
        n = self.n
        dp = [array("d", [INF]) * n for _ in range(size)]
        par = [array("b", [-1]) * n for _ in range(size)]
        bit = [1 << k for k in range(nc)]
        vbit = {v: bit[k] for k, v in enumerate(C)}
        aa = self.arc_arrival
        for k, v in enumerate(C):
            dp[bit[k]][v] = aa(d0, v, 0.0)
            par[bit[k]][v] = d0
        now = time.time
        for mask in range(1, size):
            if (mask & 255) == 0 and now() > deadline:
                return None
            ks = []
            m = mask
            while m:
                bb = m & -m
                m ^= bb
                ks.append(bb.bit_length() - 1)
            if len(ks) < 2:
                continue
            dpm = dp[mask]
            pam = par[mask]
            for kj in ks:
                j = C[kj]
                sub = mask ^ bit[kj]
                dps = dp[sub]
                best = INF
                bj = -1
                m2 = sub
                while m2:
                    bb = m2 & -m2
                    m2 ^= bb
                    i = C[bb.bit_length() - 1]
                    ti = dps[i]
                    if ti == INF:
                        continue
                    a = aa(i, j, ti)
                    if a < best:
                        best = a
                        bj = i
                dpm[j] = best
                pam[j] = bj
        full = size - 1
        best = INF
        last = -1
        for v in C:
            ti = dp[full][v]
            if ti == INF:
                continue
            a = aa(v, d0, ti)
            if a < best:
                best = a
                last = v
        if last < 0 or best == INF:
            return None
        seq = []
        mask = full
        j = last
        while True:
            seq.append(j)
            pj = par[mask][j]
            if pj == d0 or pj < 0:
                break
            mask ^= vbit[j]
            j = pj
        route = [d0] + seq[::-1] + [d0]
        obj, _ = self.eval_full(route)
        return route, obj

    # ---------------- driver ----------------
    def log_best(self, logger, obj, route):
        if logger is None:
            return
        o = obj if math.isfinite(obj) else 1e15
        sol = {"objective_value": float(o), "tour": [int(x) for x in route]}
        logger.log_solution(float(o), sol)

    def solve(self, time_limit, logger):
        start = time.time()
        total = max(1.0, float(time_limit))
        deadline = start + total - 0.5
        rng = random.Random(0)
        self.cand = self.build_candidates()

        route = self.nn_construct(deadline)
        obj, _ = self.eval_full(route)
        best_route, best_obj = route, obj
        self.log_best(logger, best_obj, best_route)

        cap = min(deadline, start + 0.25 * total)
        if time.time() < cap:
            r, o = self.local_search(route, cap)
            if self.eff(o) < self.eff(best_obj) - 1e-9:
                best_route, best_obj = r, o
                self.log_best(logger, best_obj, best_route)

        # reversed orientation may be much better under time dependence
        if self.nc > 1 and time.time() < deadline:
            rev = [self.depot] + best_route[1:-1][::-1] + [self.depot]
            orv, _ = self.eval_full(rev)
            if self.eff(orv) < self.eff(best_obj) - 1e-9:
                r, o = self.local_search(rev, min(deadline, time.time() + 0.05 * total))
                if self.eff(o) < self.eff(best_obj) - 1e-9:
                    best_route, best_obj = r, o
                    self.log_best(logger, best_obj, best_route)

        # exact DP for small instances (optimal if it finishes)
        if self.nc <= 16 and time.time() < deadline:
            dp_deadline = min(deadline, start + 0.8 * total)
            res = self.solve_exact(dp_deadline)
            if res is not None:
                route, obj = res
                if self.eff(obj) < self.eff(best_obj) - 1e-9:
                    best_route, best_obj = route, obj
                    self.log_best(logger, best_obj, best_route)
                return best_route, best_obj  # provably optimal

        if time.time() < deadline:
            best_route, best_obj = self.ils(best_route, best_obj, deadline, logger, rng)
        return best_route, best_obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    solver = TDTSPSolver(data)
    try:
        best_route, best_obj = solver.solve(args.time_limit, logger)
    except Exception:
        best_route = [solver.depot] + solver.customers + [solver.depot]
        best_obj, _ = solver.eval_full(best_route)

    if not math.isfinite(best_obj):
        best_obj = 1e15
    solution = {
        "objective_value": float(best_obj),
        "tour": [int(x) for x in best_route],
    }
    with open(args.solution_path, "w") as f:
        json.dump(solution, f)

    if logger is not None:
        try:
            logger.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
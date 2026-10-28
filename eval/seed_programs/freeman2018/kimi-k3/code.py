#!/usr/bin/env python3
"""
Touring entertainer optimization.

Approach:
  Solution space is decomposed into:
    (a) the event schedule: for each period, no event or an event at one site
        (subject to per-site event limits and the rest-day window constraint);
    (b) the tour route: for a fixed event schedule the cheapest feasible tour is
        computed exactly with a shortest-path style DP over (period, site),
        where moving to the event site is forced in event periods and each step
        must respect the per-period travel limit.
  Given an event schedule, the gravity-style attendance model (with the
  performance-shadow denominator) is evaluated exactly and vectorized with
  numpy. A greedy construction plus hill-climbing gives a strong start, followed
  by simulated annealing over event schedules. Every improved incumbent is
  reported to the solution logger.
"""

import argparse
import json
import math
import random
import time

import numpy as np

from solution_logger import SolutionLogger

INF_COST = 1e15
BAD_OBJ = -1e15


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    return parser.parse_args()


def window_sum(att, e):
    """att: (C, n); returns (C, n) where column t sums columns in [t-e, t+e]."""
    n = att.shape[1]
    if n == 0:
        return att.copy()
    cs = np.cumsum(att, axis=1)
    idx = np.arange(n)
    hi = np.minimum(idx + e, n - 1)
    lo = idx - e - 1  # subtract prefix ending at lo (if lo >= 0)
    out = cs[:, hi]
    mask = lo >= 0
    if mask.any():
        corr = np.zeros_like(out)
        corr[:, mask] = cs[:, lo[mask]]
        out = out - corr
    return out


class Model:
    def __init__(self, data):
        params = data["parameters"]
        self.T = int(params.get("T", data.get("num_periods", 0)))
        self.delta = int(params.get("delta", 0))
        self.epsilon = int(params.get("epsilon", 0))
        self.L = float(params.get("L", 0.0))
        self.c = float(params.get("c", 0.0))
        self.q = float(params.get("q", 2.0))
        cycle = list(params.get("weekly_attraction_cycle", [1.0] * 7))
        if not cycle:
            cycle = [1.0]
        self.avg_c = float(params.get("avg_businesses_customers", 1.0))
        self.avg_s = float(params.get("avg_businesses_sites", 1.0))

        sites = sorted(data.get("sites", []), key=lambda s: s.get("id", 0))
        customers = sorted(data.get("customers", []), key=lambda c: c.get("id", 0))
        self.S = len(sites)
        self.C = len(customers)

        # site attributes
        self.cap = np.array([float(s.get("capacity", 0)) for s in sites])
        self.rev = np.array([float(s.get("revenue_per_customer", 0)) for s in sites])
        self.fixed = np.array([float(s.get("fixed_cost", 0)) for s in sites])
        self.maxev = np.array([int(s.get("max_events", 0)) for s in sites])
        site_attr = np.array(
            [float(s.get("num_businesses", 1)) / max(self.avg_s, 1e-9) for s in sites]
        )
        self.sx = np.array([float(s.get("x", 0.0)) for s in sites])
        self.sy = np.array([float(s.get("y", 0.0)) for s in sites])
        self.w = np.array([float(cycle[t % len(cycle)]) for t in range(self.T)])
        # site_attraction(site, period) = normalized_businesses * weekly factor
        self.sa_w = site_attr[:, None] * self.w[None, :] if self.T > 0 else np.zeros((self.S, 0))

        # customer attributes
        self.pop = np.array([float(c.get("population", 0)) for c in customers])
        nb = np.array([float(c.get("num_businesses", 0)) for c in customers])
        self.selfattr = np.maximum(nb / max(self.avg_c, 1e-9), 0.0)
        self.cx = np.array([float(c.get("x", 0.0)) for c in customers])
        self.cy = np.array([float(c.get("y", 0.0)) for c in customers])

        # site-to-site distances, index 0 = dummy start (distance 0 to all sites)
        D = np.zeros((self.S + 1, self.S + 1))
        s2s = data.get("site_to_site_distances", {}) or {}
        for key, val in s2s.items():
            try:
                a_s, b_s = key.split("_")
                a, b = int(a_s), int(b_s)
                if 0 <= a <= self.S and 0 <= b <= self.S:
                    D[a, b] = float(val)
            except Exception:
                continue
        # mirror missing symmetric entries; coordinate fallback for missing pairs
        for a in range(1, self.S + 1):
            for b in range(1, self.S + 1):
                if a == b:
                    continue
                if D[a, b] == 0.0 and D[b, a] != 0.0:
                    D[a, b] = D[b, a]
                elif D[a, b] == 0.0 and D[b, a] == 0.0:
                    d = math.hypot(self.sx[a - 1] - self.sx[b - 1], self.sy[a - 1] - self.sy[b - 1])
                    if d > 0.0:
                        D[a, b] = d
        self.D = D
        d_sub = D[1:, 1:] if self.S > 0 else np.zeros((0, 0))
        self.stepM = np.where(d_sub <= self.L, d_sub, INF_COST)
        self.d0 = D[0, 1:].copy() if self.S > 0 else np.zeros(0)
        self.d0 = np.where(self.d0 <= self.L, self.d0, INF_COST)

        # customer-to-site distances
        DCS = np.zeros((self.C, self.S))
        c2s = data.get("customer_to_site_distances", {}) or {}
        for key, val in c2s.items():
            try:
                a_s, b_s = key.split("_")
                a, b = int(a_s), int(b_s)
                if 1 <= a <= self.C and 1 <= b <= self.S:
                    DCS[a - 1, b - 1] = float(val)
            except Exception:
                continue
        # coordinate fallback for missing entries
        missing = DCS == 0.0
        if missing.any() and self.C > 0 and self.S > 0:
            for i in range(self.C):
                for j in range(self.S):
                    if DCS[i, j] == 0.0:
                        d = math.hypot(self.cx[i] - self.sx[j], self.cy[i] - self.sy[j])
                        if d > 0.0:
                            DCS[i, j] = d
        DCS = np.maximum(DCS, 1e-9)
        self.inv = 1.0 / np.power(DCS, self.q)  # C x S base distance term

        # rest-day windows: orderings t = 1..T-delta-1, window t..t+delta+1
        self.n_starts = max(0, self.T - self.delta - 1)

    # -------------------------------------------------- feasibility --------
    def feasible(self, ev):
        """ev: int array length T with values -1 (no event) or a site index."""
        evn = ev[ev >= 0]
        if evn.size:
            cnt = np.bincount(evn, minlength=self.S)
            if np.any(cnt > self.maxev):
                return False
        if self.n_starts > 0:
            b = (ev >= 0).astype(np.int64)
            cs = np.zeros(self.T + 1, dtype=np.int64)
            cs[1:] = np.cumsum(b)
            lo = np.arange(self.n_starts)
            hi = lo + self.delta + 2  # half-open end of window (inclusive delta+1)
            sums = cs[hi] - cs[lo]
            if np.any(sums > self.delta):
                return False
        return True

    # -------------------------------------------------- routing ------------
    def route_distance(self, ev):
        """Minimum total distance of a tour consistent with the event schedule.

        ev[t] >= 0 forces the tour to be at that site in period t (0-based).
        Returns None if infeasible.
        """
        T, S = self.T, self.S
        if T == 0:
            return 0.0
        if S == 0:
            return INF_COST
        cur = self.d0.copy()
        if ev[0] >= 0:
            tmp = np.full(S, INF_COST)
            tmp[ev[0]] = cur[ev[0]]
            cur = tmp
        for t in range(1, T):
            nxt = np.min(cur[:, None] + self.stepM, axis=0)
            if ev[t] >= 0:
                tmp = np.full(S, INF_COST)
                tmp[ev[t]] = nxt[ev[t]]
                nxt = tmp
            cur = nxt
        best = float(np.min(cur))
        if not np.isfinite(best) or best >= INF_COST:
            return None
        return best

    def route_positions(self, ev):
        """Returns (positions array length T, total distance) or (None, None)."""
        T, S = self.T, self.S
        if T == 0:
            return np.zeros(0, dtype=int), 0.0
        if S == 0:
            return None, None
        hist = []
        cur = self.d0.copy()
        if ev[0] >= 0:
            tmp = np.full(S, INF_COST)
            tmp[ev[0]] = cur[ev[0]]
            cur = tmp
        hist.append(cur.copy())
        for t in range(1, T):
            nxt = np.min(cur[:, None] + self.stepM, axis=0)
            if ev[t] >= 0:
                tmp = np.full(S, INF_COST)
                tmp[ev[t]] = nxt[ev[t]]
                nxt = tmp
            cur = nxt
            hist.append(cur.copy())
        if not np.isfinite(float(np.min(cur))) or float(np.min(cur)) >= INF_COST:
            return None, None
        pos = np.zeros(T, dtype=int)
        b = int(np.argmin(cur))
        pos[T - 1] = b
        for t in range(T - 1, 0, -1):
            prev = hist[t - 1] + self.stepM[:, b]
            b = int(np.argmin(prev))
            pos[t - 1] = b
        return pos, float(np.min(hist[-1]))

    # -------------------------------------------------- evaluation ---------
    def evaluate(self, ev, collect=False):
        events = np.flatnonzero(ev >= 0)
        revenue = 0.0
        fixed_cost = 0.0
        attendance = {}
        if events.size:
            Att = np.zeros((self.C, self.T))
            for t in events:
                j = ev[t]
                Att[:, t] = self.sa_w[j, t] * self.inv[:, j]
            W = window_sum(Att, self.epsilon)
            for t in events:
                j = ev[t]
                A = Att[:, t]
                denom = self.selfattr + W[:, t]
                np.maximum(denom, 1e-12, out=denom)
                r = A / denom
                np.clip(r, 0.0, 1.0, out=r)
                att = float(self.pop @ r)
                if att > self.cap[j]:
                    att = float(self.cap[j])
                revenue += self.rev[j] * att
                fixed_cost += self.fixed[j]
                if collect:
                    attendance[(int(j), int(t))] = att
        dist = self.route_distance(ev)
        if dist is None:
            if collect:
                return BAD_OBJ, attendance
            return BAD_OBJ
        obj = revenue - fixed_cost - self.c * dist
        if collect:
            return obj, attendance
        return obj


# -------------------------------------------------- solution building ------
def build_solution(model, ev):
    T, S = model.T, model.S
    obj, att = model.evaluate(ev, collect=True)
    pos, dist = model.route_positions(ev)
    if pos is None:
        # safety fallback (should not happen for a feasible incumbent)
        pos = np.zeros(T, dtype=int)
        for t in range(T):
            if ev[t] >= 0:
                pos[t] = ev[t]
            elif t > 0:
                pos[t] = pos[t - 1]
    tour = {str(t + 1): int(pos[t]) + 1 for t in range(T)}
    events = {str(t + 1): (1 if ev[t] >= 0 else 0) for t in range(T)}
    event_sites = {str(t + 1): int(ev[t]) + 1 for t in range(T) if ev[t] >= 0}
    attendance = {}
    for t in range(T):
        for j in range(S):
            attendance["{}_{}".format(j + 1, t + 1)] = float(att.get((j, t), 0.0))
    return {
        "objective_value": float(obj),
        "tour": tour,
        "events": events,
        "event_sites": event_sites,
        "attendance": attendance,
    }


# -------------------------------------------------- greedy -----------------
def greedy_initial(model, deadline):
    T, S = model.T, model.S
    profit_mat = np.zeros((T, S))
    for t in range(T):
        A = model.sa_w[:, t][None, :] * model.inv  # C x S attractions at period t
        denom = model.selfattr[:, None] + A
        r = np.clip(A / np.maximum(denom, 1e-12), 0.0, 1.0)
        att = np.minimum(model.pop @ r, model.cap)
        profit_mat[t] = model.rev * att - model.fixed
    order = np.argsort(-profit_mat, axis=None)
    ev = -np.ones(T, dtype=int)
    cur_obj = model.evaluate(ev)
    for k in order:
        if time.time() >= deadline:
            break
        if profit_mat.flat[k] <= 0:
            break
        t = int(k) // S
        j = int(k) % S
        if ev[t] != -1:
            continue
        ev[t] = j
        if not model.feasible(ev):
            ev[t] = -1
            continue
        o = model.evaluate(ev)
        if o > cur_obj + 1e-9:
            cur_obj = o
        else:
            ev[t] = -1
    return ev, cur_obj


# -------------------------------------------------- hill climbing ----------
def hill_climb(model, ev, obj, deadline):
    T, S = model.T, model.S
    improved = True
    while improved:
        improved = False
        if time.time() >= deadline:
            break
        for t in range(T):
            if time.time() >= deadline:
                break
            cur = int(ev[t])
            best_choice = cur
            best_obj = obj
            for new in range(-1, S):
                if new == cur:
                    continue
                ev2 = ev.copy()
                ev2[t] = new
                if not model.feasible(ev2):
                    continue
                o = model.evaluate(ev2)
                if o > best_obj + 1e-9:
                    best_obj = o
                    best_choice = new
            if best_choice != cur:
                ev[t] = best_choice
                obj = best_obj
                improved = True
    return obj


# -------------------------------------------------- temperature estimate ---
def estimate_temperature(model, ev, obj, rng, samples=150):
    T, S = model.T, model.S
    deltas = []
    for _ in range(samples):
        t = rng.randrange(T)
        new = rng.randrange(S + 1) - 1
        if new == ev[t]:
            continue
        ev2 = ev.copy()
        ev2[t] = new
        if not model.feasible(ev2):
            continue
        o = model.evaluate(ev2)
        if o > BAD_OBJ / 2:
            d = abs(o - obj)
            if d > 1e-9:
                deltas.append(d)
    base = abs(obj) * 0.01 + 1e-3
    if deltas:
        return max(2.0 * (sum(deltas) / len(deltas)), base, 1e-3)
    return max(base, 1e-3)


# -------------------------------------------------- main solve -------------
def solve(model, hard_deadline, logger):
    T, S = model.T, model.S
    rng = random.Random(0)

    if T == 0 or S == 0:
        return -np.ones(T, dtype=int)

    best_ev = -np.ones(T, dtype=int)
    best_obj = model.evaluate(best_ev)
    last_log = [0.0]

    def commit(ev, obj, force_log=False):
        nonlocal best_ev, best_obj
        if obj > best_obj + 1e-9:
            best_obj = obj
            best_ev = ev.copy()
            if logger and (force_log or time.time() - last_log[0] > 0.05):
                last_log[0] = time.time()
                logger.log_solution(float(obj), build_solution(model, best_ev))

    if logger:
        logger.log_solution(float(best_obj), build_solution(model, best_ev))

    # greedy + hill climb
    ev, obj = greedy_initial(model, hard_deadline)
    hc_deadline = min(hard_deadline, time.time() + max(0.3, 0.25 * (hard_deadline - time.time())))
    obj = hill_climb(model, ev, obj, hc_deadline)
    commit(ev, obj)

    if time.time() >= hard_deadline:
        return best_ev

    # simulated annealing
    T0 = estimate_temperature(model, best_ev, best_obj, rng)
    Tmin = T0 * 1e-4
    sa_start = time.time()
    sa_budget = max(1e-3, hard_deadline - sa_start)
    frac = 0.0
    ev = best_ev.copy()
    obj = best_obj
    it = 0
    since_improve = 0
    restart_limit = 20000

    while True:
        it += 1
        if it % 128 == 0:
            now = time.time()
            if now >= hard_deadline:
                break
            frac = min(1.0, (now - sa_start) / sa_budget)
        temp = max(T0 * (Tmin / T0) ** frac, 1e-12)

        t = rng.randrange(T)
        new = rng.randrange(S + 1) - 1
        if new == ev[t]:
            continue
        old = ev[t]
        ev[t] = new
        since_improve += 1
        feas = model.feasible(ev)
        o = model.evaluate(ev) if feas else BAD_OBJ
        d = o - obj
        if o <= BAD_OBJ / 2:
            ev[t] = old
        elif d >= 0 or rng.random() < math.exp(d / temp):
            obj = o
            if o > best_obj + 1e-9:
                commit(ev, o)
                since_improve = 0
        else:
            ev[t] = old

        if since_improve > restart_limit:
            # restart from incumbent with re-heated exploration
            ev = best_ev.copy()
            obj = best_obj
            since_improve = 0
            frac = min(frac, 0.4)

    # final polish
    if time.time() < hard_deadline:
        pol = best_ev.copy()
        o = hill_climb(model, pol, best_obj, hard_deadline)
        commit(pol, o)

    if logger:
        logger.log_solution(float(best_obj), build_solution(model, best_ev))
    return best_ev


def main():
    args = parse_args()
    with open(args.instance_path, "r") as f:
        data = json.load(f)

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    model = Model(data)
    start = time.time()
    reserve = min(0.5, max(0.05, 0.1 * args.time_limit))
    hard_deadline = start + max(0.1, args.time_limit - reserve)

    best_ev = solve(model, hard_deadline, logger)
    sol = build_solution(model, best_ev)
    if logger:
        logger.log_solution(float(sol["objective_value"]), sol)

    with open(args.solution_path, "w") as f:
        json.dump(sol, f)


if __name__ == "__main__":
    main()
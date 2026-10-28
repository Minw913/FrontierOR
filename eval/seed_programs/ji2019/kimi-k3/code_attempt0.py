import json
import argparse
import math
import time
import random

from solution_logger import SolutionLogger

EPS = 1e-6
INF = 1e18


class Solver:
    def __init__(self, inst, t_end, logger):
        self.t_end = t_end
        self.reserve = 1.0
        self.logger = logger

        ships = inst.get("ships", []) or []
        berths = inst.get("berths", []) or []
        self.n = n = len(ships)
        self.K = K = len(berths)
        self.W = float(inst.get("chamber_width_W", 0.0) or 0.0)
        self.L = float(inst.get("chamber_length_L", 0.0) or 0.0)
        self.Du = float(inst.get("lockage_duration_Du", 0.0) or 0.0)
        self.MT = float(inst.get("min_time_between_lockages_MT", 0.0) or 0.0)
        self.sc = float(inst.get("lock_start_availability_sc", 0.0) or 0.0)
        self.ek = float(inst.get("berth_end_availability_ek", 1e18) or 1e18)

        self.sid = [s.get("id", i) for i, s in enumerate(ships)]
        id2i = {self.sid[i]: i for i in range(n)}
        self.a = [float(s.get("arrival_time", 0.0) or 0.0) for s in ships]
        self.w = [float(s.get("width", 0.0) or 0.0) for s in ships]
        self.ln = [float(s.get("length", 0.0) or 0.0) for s in ships]
        self.pen = [float(s.get("transshipment_penalty", 0.0) or 0.0) for s in ships]
        self.MR = []
        for s in ships:
            ms = set()
            for x in (s.get("mooring_set_MR", []) or []):
                if x in id2i:
                    ms.add(id2i[x])
                elif isinstance(x, int) and 0 <= x < n:
                    ms.add(x)
            self.MR.append(ms)

        self.bid = [b.get("id", k) for k, b in enumerate(berths)]
        bid2k = {self.bid[k]: k for k in range(K)}
        self.sk = [float(b.get("start_availability_sk", 0.0) or 0.0) for b in berths]

        HT = inst.get("handling_times", []) or []
        self.p = [[0.0] * K for _ in range(n)]
        for i in range(n):
            row = HT[i] if i < len(HT) and isinstance(HT[i], list) else []
            for k in range(K):
                if k < len(row) and row[k] is not None:
                    try:
                        self.p[i][k] = float(row[k])
                    except Exception:
                        self.p[i][k] = 0.0

        self.elig = []
        for i, s in enumerate(ships):
            es = set()
            for b in (s.get("eligible_berths_B", []) or []):
                if b in bid2k:
                    es.add(bid2k[b])
                elif isinstance(b, int) and 0 <= b < K:
                    es.add(b)
            self.elig.append(sorted(es))

        self.fits = [self.w[i] <= self.W + EPS and self.ln[i] <= self.L + EPS
                     for i in range(n)]
        self.forced_lock = [len(self.elig[i]) == 0 for i in range(n)]
        self.forced_trans = [(not self.fits[i]) and len(self.elig[i]) > 0
                             for i in range(n)]
        self.free = [(not self.forced_lock[i]) and (not self.forced_trans[i])
                     for i in range(n)]
        self.free_list = [i for i in range(n) if self.free[i]]

        def ni(x):
            return abs(x - round(x)) < 1e-6
        self.int_ok = ni(self.W) and ni(self.L) and all(
            ni(self.w[i]) and ni(self.ln[i]) for i in range(n))

        self.base_idx = 1 if (n > 0 and min(self.sid) >= 1) else 0

        # caches
        self.pack_cache = {}
        self.lock_cache = {}
        self.berth_cache = {}
        self.dp_time_total = 0.0
        self.dp_count = 0

        # state
        self.modes = [0] * n
        self.berth_of = [-1] * n
        self.best_total = INF
        self.best_modes = None
        self.best_berth_of = None

    # ------------------ packing feasibility ------------------
    def pack_feasible(self, seg):
        r = self.pack_cache.get(seg)
        if r is None:
            r = self._pack_feasible(seg)
            self.pack_cache[seg] = r
        return r

    def _pack_feasible(self, seg):
        ln = self.ln; w = self.w; W = self.W; L = self.L; MR = self.MR
        for i in seg:
            if not self.fits[i]:
                return False
        if not self.int_ok:
            # conservative: one ship per shelf, integer y stacking with ceilings
            y = 0.0
            for i in sorted(seg, key=lambda i: (-ln[i], i)):
                if y + ln[i] > L + EPS:
                    return False
                y = math.ceil(y + ln[i] - 1e-9)
            return True
        suml = 0.0
        area = 0.0
        for i in seg:
            suml += ln[i]
            area += w[i] * ln[i]
        if suml <= L + EPS:
            return True  # one wall-moored shelf per ship
        if area > W * L + EPS:
            return False
        o1 = sorted(seg, key=lambda i: (-ln[i], -w[i], i))
        o2 = sorted(seg, key=lambda i: (-ln[i], w[i], i))
        attempts = ((o1, 0), (o1, 1), (o2, 0))
        for order, pref in attempts:
            shelves = []  # [band_len, left_last, left_w, right_last, right_w]
            total = 0.0
            ok = True
            for i in order:
                li = ln[i]; wi = w[i]
                best = None  # (leftover, shelf_idx, side 0=left,1=right)
                for si in range(len(shelves)):
                    band, llast, lw, rlast, rw = shelves[si]
                    if li > band + EPS:
                        continue
                    if (llast is None or llast in MR[i]) and lw + rw + wi <= W + EPS:
                        lo = W - (lw + rw + wi)
                        if best is None or lo < best[0]:
                            best = (lo, si, 0)
                    if (rlast is None or rlast in MR[i]) and lw + rw + wi <= W + EPS:
                        lo = W - (lw + rw + wi)
                        if best is None or lo < best[0]:
                            best = (lo, si, 1)
                if best is not None:
                    _, si, side = best
                    sh = shelves[si]
                    if side == 0:
                        sh[1] = i
                        sh[2] += wi
                    else:
                        sh[3] = i
                        sh[4] += wi
                else:
                    total += li
                    if total > L + EPS:
                        ok = False
                        break
                    if pref == 0:
                        shelves.append([li, i, wi, None, 0.0])
                    else:
                        shelves.append([li, None, 0.0, i, wi])
            if ok and total <= L + EPS:
                return True
        return False

    # ------------------ lock batching (optimal partition under FCFS) ------------------
    def solve_lock(self, lock_set):
        key = tuple(sorted(lock_set))
        r = self.lock_cache.get(key)
        if r is not None:
            return r
        ta = time.time()
        r = self._solve_lock(key)
        self.dp_time_total += time.time() - ta
        self.dp_count += 1
        self.lock_cache[key] = r
        return r

    def _solve_lock(self, ids_key):
        ids = sorted(ids_key, key=lambda i: (self.a[i], i))
        m = len(ids)
        if m == 0:
            return (0.0, [], [])
        a = self.a
        for i in ids:
            if not self.fits[i]:
                return (INF, None, None)
        aa = [a[i] for i in ids]
        A = [0.0] * (m + 1)
        for t in range(m):
            A[t + 1] = A[t] + aa[t]
        W = self.W; L = self.L; ln = self.ln; w = self.w
        maxJ = [-1] * m
        for i0 in range(m):
            suml = 0.0
            area = 0.0
            for j0 in range(i0, m):
                sj = ids[j0]
                suml += ln[sj]
                area += w[sj] * ln[sj]
                if area > W * L + EPS:
                    break
                if suml <= L + EPS:
                    maxJ[i0] = j0
                    continue
                seg = tuple(sorted(ids[i0:j0 + 1]))
                if self.pack_feasible(seg):
                    maxJ[i0] = j0
                else:
                    break
        Du = self.Du; MT = self.MT; sc = self.sc
        frontier = [None] * (m + 1)
        frontier[0] = [(-1e18, 0.0, -1, -1)]  # (completion, cost, prev_j, prev_ei)
        for i in range(1, m + 1):
            base = (sc if sc > aa[i - 1] else aa[i - 1]) + Du
            cands = []
            for j in range(0, i):
                if maxJ[j] < i - 1:
                    continue
                fj = frontier[j]
                if not fj:
                    continue
                cnt = i - j
                segsum = A[i] - A[j]
                for ei in range(len(fj)):
                    Cprev, costprev, _, _ = fj[ei]
                    Cnew = Cprev + MT
                    if base > Cnew:
                        Cnew = base
                    cands.append((Cnew, costprev + cnt * Cnew - segsum, j, ei))
            cands.sort(key=lambda x: (x[0], x[1]))
            fr = []
            bestc = INF
            for C, c, j, ei in cands:
                if c < bestc - 1e-9:
                    fr.append((C, c, j, ei))
                    bestc = c
                if len(fr) >= 60:
                    break
            frontier[i] = fr
        fm = frontier[m]
        if not fm:
            return (INF, None, None)
        bei = min(range(len(fm)), key=lambda e: fm[e][1])
        batches = []
        comps = []
        i = m
        ei = bei
        while i > 0:
            C, c, j, pei = frontier[i][ei]
            batches.append(ids[j:i])
            comps.append(C)
            i = j
            ei = pei
        batches.reverse()
        comps.reverse()
        return (fm[bei][1], batches, comps)

    # ------------------ berth sequencing ------------------
    def _simulate(self, seq, k):
        C = self.sk[k]
        tot = 0.0
        a = self.a
        pk = self.p
        for i in seq:
            st = C if C > a[i] else a[i]
            C = st + pk[i][k]
            tot += C - a[i]
        return tot, C

    def berth_eval(self, k, S):
        if not S:
            return (0.0, [])
        key = (k, S)
        r = self.berth_cache.get(key)
        if r is None:
            r = self._berth_eval(k, list(S))
            self.berth_cache[key] = r
        return r

    def _berth_eval(self, k, S):
        m = len(S)
        sk = self.sk[k]
        a = self.a; p = self.p
        erd = sorted(S, key=lambda i: (a[i], i))
        _, cmax = self._simulate(erd, k)
        if cmax > self.ek + EPS:
            return (INF, None)  # work-conserving makespan is order-invariant
        cands = [erd,
                 sorted(S, key=lambda i: (p[i][k], a[i], i)),
                 sorted(S, key=lambda i: (a[i] + p[i][k], i))]
        # greedy non-delay SPT
        rem = set(S)
        t = sk
        g = []
        while rem:
            rel = [i for i in rem if a[i] <= t + EPS]
            if not rel:
                t = min(a[i] for i in rem)
                rel = [i for i in rem if a[i] <= t + EPS]
            i = min(rel, key=lambda i: (p[i][k], a[i], i))
            g.append(i)
            rem.discard(i)
            t = t + p[i][k]
        cands.append(g)
        bestseq = None
        bestc = INF
        for seq in cands:
            c, _ = self._simulate(seq, k)
            if c < bestc - 1e-9:
                bestc = c
                bestseq = seq
        if m <= 70:
            improved = True
            rounds = 0
            while improved and rounds < 20:
                improved = False
                rounds += 1
                for x in range(m - 1):
                    seq2 = list(bestseq)
                    seq2[x], seq2[x + 1] = seq2[x + 1], seq2[x]
                    c2, _ = self._simulate(seq2, k)
                    if c2 < bestc - 1e-9:
                        bestc = c2
                        bestseq = seq2
                        improved = True
        if m <= 25:
            improved = True
            rounds = 0
            while improved and rounds < 6:
                improved = False
                rounds += 1
                for x in range(m):
                    if improved:
                        break
                    for y in range(m):
                        if x == y:
                            continue
                        seq2 = list(bestseq)
                        v = seq2.pop(x)
                        seq2.insert(y, v)
                        c2, _ = self._simulate(seq2, k)
                        if c2 < bestc - 1e-9:
                            bestc = c2
                            bestseq = seq2
                            improved = True
                            break
        return (bestc, bestseq)

    # ------------------ state helpers ------------------
    def derive(self, modes=None, berth_of=None):
        modes = self.modes if modes is None else modes
        berth_of = self.berth_of if berth_of is None else berth_of
        lock_set = set()
        bsets = [set() for _ in range(self.K)]
        psum = 0.0
        for i in range(self.n):
            if modes[i] == 0:
                lock_set.add(i)
            else:
                k = berth_of[i]
                if k is None or k < 0 or k >= self.K:
                    k = self.elig[i][0] if self.elig[i] else 0
                if self.K > 0:
                    bsets[k].add(i)
                psum += self.pen[i]
        return lock_set, bsets, psum

    def total_current(self):
        lock_set, bsets, psum = self.derive()
        lc = self.solve_lock(lock_set)[0]
        if lc >= INF:
            return INF
        tot = lc + psum
        for k in range(self.K):
            c = self.berth_eval(k, tuple(sorted(bsets[k])))[0]
            if c >= INF:
                return INF
            tot += c
        return tot

    def set_best(self, total):
        self.best_total = total
        self.best_modes = list(self.modes)
        self.best_berth_of = list(self.berth_of)
        if self.logger is not None:
            sol = self.build_solution(self.best_modes, self.best_berth_of)
            self.logger.log_solution(float(sol["objective_value"]), sol)

    def assign_initial_berths(self):
        n = self.n; K = self.K
        bsets = [set() for _ in range(K)]
        for i in range(n):
            if self.modes[i] == 1 and 0 <= self.berth_of[i] < K:
                bsets[self.berth_of[i]].add(i)
        costs = [0.0] * K
        for k in range(K):
            if bsets[k]:
                costs[k] = self.berth_eval(k, tuple(sorted(bsets[k])))[0]
        todo = [i for i in range(n) if self.modes[i] == 1 and
                not (0 <= self.berth_of[i] < K)]
        todo.sort(key=lambda i: (len(self.elig[i]), self.a[i], i))
        for i in todo:
            bestk = -1
            bestd = INF
            for k in self.elig[i]:
                c, _ = self.berth_eval(k, tuple(sorted(bsets[k] | {i})))
                if c >= INF:
                    continue
                d = c - costs[k]
                if d < bestd:
                    bestd = d
                    bestk = k
            if bestk < 0:
                if self.fits[i]:
                    self.modes[i] = 0
                    self.berth_of[i] = -1
                    continue
                if self.elig[i]:
                    bestk = min(self.elig[i],
                                key=lambda k: self.berth_eval(
                                    k, tuple(sorted(bsets[k] | {i})))[0])
                else:
                    self.modes[i] = 0
                    self.berth_of[i] = -1
                    continue
            self.berth_of[i] = bestk
            bsets[bestk].add(i)
            costs[bestk] = self.berth_eval(bestk, tuple(sorted(bsets[bestk])))[0]

    # ------------------ local search ------------------
    def descend(self):
        n = self.n; K = self.K
        while True:
            if time.time() > self.t_end - self.reserve:
                return
            lock_set, bsets, _ = self.derive()
            lc_cur = self.solve_lock(lock_set)[0]
            if lc_cur >= INF:
                return
            bc_cur = [self.berth_eval(k, tuple(sorted(bsets[k])))[0]
                      for k in range(K)]
            order = [i for i in range(n) if self.free[i]]
            random.shuffle(order)
            avg = (self.dp_time_total / self.dp_count) if self.dp_count > 0 else 0.01
            remain = self.t_end - self.reserve - time.time()
            max_nei = int(max(10, (remain * 0.5) / max(avg, 1e-3)))
            if len(order) > max_nei:
                order = order[:max_nei]
            improved = False
            for i in order:
                if time.time() > self.t_end - self.reserve:
                    return
                if self.modes[i] == 0:
                    nlc = self.solve_lock(lock_set - {i})[0]
                    if nlc >= INF:
                        continue
                    dl = nlc - lc_cur
                    bestk = -1
                    bestd = -1e-7
                    for k in self.elig[i]:
                        nbc = self.berth_eval(k, tuple(sorted(bsets[k] | {i})))[0]
                        if nbc >= INF:
                            continue
                        d = dl + (nbc - bc_cur[k]) + self.pen[i]
                        if d < bestd:
                            bestd = d
                            bestk = k
                    if bestk >= 0:
                        self.modes[i] = 1
                        self.berth_of[i] = bestk
                        improved = True
                        break
                else:
                    ki = self.berth_of[i]
                    if ki < 0 or ki >= K:
                        ki = self.elig[i][0] if self.elig[i] else 0
                    rem = bsets[ki] - {i}
                    nbc_ki = self.berth_eval(ki, tuple(sorted(rem)))[0]
                    if self.fits[i]:
                        nlc = self.solve_lock(lock_set | {i})[0]
                        if nlc < INF:
                            d = (nlc - lc_cur) + (nbc_ki - bc_cur[ki]) - self.pen[i]
                            if d < -1e-7:
                                self.modes[i] = 0
                                self.berth_of[i] = -1
                                improved = True
                                break
                    for k2 in self.elig[i]:
                        if k2 == ki:
                            continue
                        nbc2 = self.berth_eval(k2, tuple(sorted(bsets[k2] | {i})))[0]
                        if nbc2 >= INF:
                            continue
                        d = (nbc_ki - bc_cur[ki]) + (nbc2 - bc_cur[k2])
                        if d < -1e-7:
                            self.berth_of[i] = k2
                            improved = True
                            break
                if improved:
                    break
            if not improved:
                return
            cur = self.total_current()
            if cur < self.best_total - 1e-9:
                self.set_best(cur)

    def perturb(self, r):
        applied = 0
        attempts = 0
        fl = self.free_list
        while applied < r and attempts < r * 8 + 10:
            attempts += 1
            if time.time() > self.t_end - self.reserve:
                return
            i = random.choice(fl)
            lock_set, bsets, _ = self.derive()
            if self.modes[i] == 0:
                if not self.elig[i]:
                    continue
                k = random.choice(self.elig[i])
                if self.berth_eval(k, tuple(sorted(bsets[k] | {i})))[0] >= INF:
                    continue
                self.modes[i] = 1
                self.berth_of[i] = k
                applied += 1
            else:
                if self.fits[i] and random.random() < 0.6:
                    self.modes[i] = 0
                    self.berth_of[i] = -1
                    applied += 1
                else:
                    ki = self.berth_of[i]
                    others = [k for k in self.elig[i] if k != ki]
                    if not others:
                        continue
                    k2 = random.choice(others)
                    if self.berth_eval(k2, tuple(sorted(bsets[k2] | {i})))[0] >= INF:
                        continue
                    self.berth_of[i] = k2
                    applied += 1

    # ------------------ solution building ------------------
    def build_solution(self, modes, berth_of):
        n = self.n; K = self.K
        lock_set = set()
        bsets = [set() for _ in range(K)]
        for i in range(n):
            if modes[i] == 0:
                lock_set.add(i)
            else:
                k = berth_of[i]
                if k is None or k < 0 or k >= K:
                    k = self.elig[i][0] if self.elig[i] else 0
                if K > 0:
                    bsets[k].add(i)
        lc, batches, comps = self.solve_lock(lock_set)
        if batches is None:
            batches = []
            comps = []
            Cprev = None
            for i in sorted(lock_set, key=lambda i: (self.a[i], i)):
                C = (self.sc if self.sc > self.a[i] else self.a[i]) + self.Du
                if Cprev is not None and C < Cprev + self.MT:
                    C = Cprev + self.MT
                batches.append([i])
                comps.append(C)
                Cprev = C
        obj = 0.0
        comp_times = {}
        lock_assign = {}
        lock_comps = {}
        for bi, (batch, C) in enumerate(zip(batches, comps)):
            lock_comps[str(bi + self.base_idx)] = float(C)
            for i in batch:
                lock_assign[str(self.sid[i])] = int(bi + self.base_idx)
                comp_times[str(self.sid[i])] = float(C)
                obj += C - self.a[i]
        berth_assign = {}
        for k in range(K):
            if not bsets[k]:
                continue
            c, seq = self.berth_eval(k, tuple(sorted(bsets[k])))
            if seq is None:
                seq = sorted(bsets[k], key=lambda i: (self.a[i], i))
            C = self.sk[k]
            for pos, i in enumerate(seq):
                st = C if C > self.a[i] else self.a[i]
                C = st + self.p[i][k]
                obj += C - self.a[i]
                berth_assign[str(self.sid[i])] = {
                    "berth": self.bid[k],
                    "order": int(pos + self.base_idx),
                }
        for i in range(n):
            if modes[i] == 1:
                obj += self.pen[i]
                comp_times[str(self.sid[i])] = float(self.a[i])
        sol = {
            "objective_value": float(round(obj, 6)),
            "modes": {str(self.sid[i]): int(modes[i]) for i in range(n)},
            "completion_times": comp_times,
            "lockage_assignments": lock_assign,
            "lockage_completion_times": lock_comps,
            "berth_assignments": berth_assign,
        }
        return sol

    # ------------------ main driver ------------------
    def run(self):
        n = self.n
        # initial: everything that can go through the lock goes through the lock
        self.modes = [1 if self.forced_trans[i] else 0 for i in range(n)]
        self.berth_of = [-1] * n
        self.assign_initial_berths()
        cur = self.total_current()
        if cur < INF:
            self.set_best(cur)
        self.descend()
        # alternative start: everything free goes to transshipment
        if (time.time() < self.t_end - self.reserve and self.free_list
                and self.K > 0):
            self.modes = [0 if self.forced_lock[i] else 1 for i in range(n)]
            self.berth_of = [-1] * n
            self.assign_initial_berths()
            cur = self.total_current()
            if cur < INF and cur < self.best_total - 1e-9:
                self.set_best(cur)
            self.descend()
        # perturbation loop (basin hopping)
        while (time.time() < self.t_end - self.reserve and self.free_list):
            if self.best_modes is not None:
                self.modes = list(self.best_modes)
                self.berth_of = list(self.best_berth_of)
            r = random.randint(1, max(1, min(4, len(self.free_list))))
            if random.random() < 0.15:
                r = max(r, len(self.free_list) // 2)
            self.perturb(r)
            self.descend()
            cur = self.total_current()
            if cur < INF and cur < self.best_total - 1e-9:
                self.set_best(cur)


def emergency_solution(inst):
    ships = inst.get("ships", []) or []
    W = float(inst.get("chamber_width_W", 0.0) or 0.0)
    L = float(inst.get("chamber_length_L", 0.0) or 0.0)
    Du = float(inst.get("lockage_duration_Du", 0.0) or 0.0)
    MT = float(inst.get("min_time_between_lockages_MT", 0.0) or 0.0)
    sc = float(inst.get("lock_start_availability_sc", 0.0) or 0.0)
    berths = inst.get("berths", []) or []
    modes = {}
    ct = {}
    la = {}
    lct = {}
    ba = {}
    obj = 0.0
    C = None
    idx = 0
    for i, s in enumerate(ships):
        sid = s.get("id", i)
        a = float(s.get("arrival_time", 0.0) or 0.0)
        w = float(s.get("width", 0.0) or 0.0)
        l = float(s.get("length", 0.0) or 0.0)
        if w <= W + EPS and l <= L + EPS:
            modes[str(sid)] = 0
            c = (sc if sc > a else a) + Du
            if C is not None and c < C + MT:
                c = C + MT
            C = c
            ct[str(sid)] = c
            la[str(sid)] = idx
            lct[str(idx)] = c
            idx += 1
            obj += c - a
        else:
            modes[str(sid)] = 1
            ct[str(sid)] = a
            eb = s.get("eligible_berths_B", []) or []
            b0 = eb[0] if eb else (berths[0].get("id", 0) if berths else 0)
            ba[str(sid)] = {"berth": b0, "order": 0}
            obj += float(s.get("transshipment_penalty", 0.0) or 0.0)
    return {
        "objective_value": float(round(obj, 6)),
        "modes": modes,
        "completion_times": ct,
        "lockage_assignments": la,
        "lockage_completion_times": lct,
        "berth_assignments": ba,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, default=60)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    random.seed(0)
    t_end = time.time() + max(1, int(args.time_limit))
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    try:
        solver = Solver(inst, t_end, logger)
        solver.run()
        if solver.best_modes is None:
            solver.best_modes = list(solver.modes)
            solver.best_berth_of = list(solver.berth_of)
        sol = solver.build_solution(solver.best_modes, solver.best_berth_of)
    except Exception:
        sol = emergency_solution(inst)

    with open(args.solution_path, "w") as f:
        json.dump(sol, f)


if __name__ == "__main__":
    main()
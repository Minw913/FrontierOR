import argparse
import bisect
import json
import random
import time

from solution_logger import SolutionLogger


class StochasticScheduler:
    """Two-stage stochastic cumulative scheduling.

    First stage: job -> facility assignment (scenario independent).
    Second stage: per facility & scenario, schedule jobs with release dates
    under a cumulative resource-capacity constraint to minimize makespan.

    Approach:
      * Greedy load-balancing construction for the assignment.
      * Serial schedule-generation scheme (earliest feasible insertion with
        several priority rules) as the per-facility scheduling engine.
      * Local search (relocate / swap moves) with incremental evaluation:
        only the two affected facilities are rescheduled per scenario.
      * Perturbation / random restarts around the incumbent until the time
        limit, plus a final polishing pass on bottleneck facilities.
    """

    def __init__(self, data, time_limit, logger):
        self.n = int(data["n"])
        self.m = int(data["m"])
        self.S = int(data["num_scenarios"])
        self.C = list(data["facility_capacity_limits"])
        self.req = data["capacity_requirements"]          # [m][n]
        self.rel = list(data["release_times"])            # [n]
        scen = data["scenarios"]
        self.prob = [float(sc["probability"]) for sc in scen]
        self.proc = [sc["processing_times"] for sc in scen]  # [s][m][n]
        self.time_limit = max(1, int(time_limit))
        self.logger = logger
        self.rng = random.Random(12345)

        n, m, S = self.n, self.m, self.S

        # Expected processing times ep[i][j].
        ep = [[0.0] * n for _ in range(m)]
        for s in range(S):
            p = self.prob[s]
            ps = self.proc[s]
            for i in range(m):
                row = ps[i]
                epi = ep[i]
                for j in range(n):
                    epi[j] += p * row[j]
        self.ep = ep

        # Feasible facilities per job (resource requirement <= capacity).
        self.allowed = []
        self.allowed_set = []
        for j in range(n):
            al = [i for i in range(m) if self.req[i][j] <= self.C[i]]
            if not al:
                al = list(range(m))  # fallback; scheduler handles r > C
            self.allowed.append(al)
            self.allowed_set.append(set(al))

        # Candidate target facilities per job for relocate moves.
        self.cand_fac = []
        for j in range(n):
            al = self.allowed[j]
            if len(al) <= 8:
                self.cand_fac.append(list(al))
            else:
                scored = sorted(
                    al,
                    key=lambda i: self.ep[i][j] * self.req[i][j]
                    / max(1.0, float(self.C[i])),
                )
                self.cand_fac.append(scored[:8])

        # Adapt the number of scheduling priority rules to the instance size.
        avgk = max(1.0, n / max(1, m))
        unit = avgk * avgk * max(1, S)
        if unit <= 3e4:
            self.n_rules_search = 3
        elif unit <= 4e5:
            self.n_rules_search = 2
        else:
            self.n_rules_search = 1
        self.search_rules = list(range(self.n_rules_search))
        self.swap_budget = max(60, min(1500, 4 * n))

        # State.
        self.assign = [0] * n
        self.fac_jobs = [set() for _ in range(m)]
        self.fm = [[0] * S for _ in range(m)]        # facility makespans
        self.fsched = [[None] * S for _ in range(m)]  # job -> start dicts
        self.scen_ms = [0] * S
        self.cur_obj = 0.0
        self.crit = [0.0] * n
        self.best_obj = float("inf")
        self.best_assign = None
        self.best_fac_jobs = None
        self.best_fm = None
        self.best_fsched = None
        self.best_scen_ms = None
        self.deadline = None

    # ------------------------------------------------------------------
    # Scheduling engine
    # ------------------------------------------------------------------
    def _sgs(self, i, s, order):
        """Serial SGS: insert each job (in the given order) at its earliest
        feasible start time >= release such that the cumulative resource
        profile never exceeds the facility capacity."""
        C = self.C[i]
        ps = self.proc[s][i]
        rs = self.req[i]
        rel = self.rel
        times = [0]   # segment start times (sorted); last segment -> +inf
        used = [0]    # resource usage per segment; invariant used[-1] == 0
        starts = {}
        ms = 0
        br = bisect.bisect_right
        bl = bisect.bisect_left
        for j in order:
            p = ps[j]
            r = rs[j]
            t = rel[j]
            if t < 0:
                t = 0
            if p <= 0:
                starts[j] = t
                continue
            lim = C if r <= C else r
            while True:
                k = br(times, t) - 1
                if k < 0:
                    k = 0
                end = t + p
                feas = True
                kk = k
                nt = len(times)
                while kk < nt:
                    if times[kk] >= end:
                        break
                    if used[kk] + r > lim:
                        # Busy segment; any start before its end is infeasible.
                        t = times[kk + 1]
                        feas = False
                        break
                    kk += 1
                if feas:
                    break
            starts[j] = t
            fin = t + p
            if fin > ms:
                ms = fin
            if r > 0:
                k = bl(times, t)
                if k >= len(times) or times[k] != t:
                    prev = used[k - 1] if k > 0 else 0
                    times.insert(k, t)
                    used.insert(k, prev)
                k2 = bl(times, fin)
                if k2 >= len(times) or times[k2] != fin:
                    prev = used[k2 - 1] if k2 > 0 else 0
                    times.insert(k2, fin)
                    used.insert(k2, prev)
                ka = bl(times, t)
                kb = bl(times, fin)
                for kk in range(ka, kb):
                    used[kk] += r
        return ms, starts

    def _sgs_seq(self, i, s, jobs):
        """Trivial feasible fallback: run jobs one at a time in release order."""
        ps = self.proc[s][i]
        rel = self.rel
        order = sorted(jobs, key=lambda j: rel[j])
        starts = {}
        t = 0
        ms = 0
        for j in order:
            tj = rel[j]
            if tj > t:
                t = tj
            starts[j] = t
            t = t + ps[j]
            if t > ms:
                ms = t
        return ms, starts

    def sched_fac(self, i, s, jobs, rule_ids):
        """Best schedule over several priority rules for one facility/scenario."""
        if not jobs:
            return 0, {}
        ps = self.proc[s][i]
        rs = self.req[i]
        rel = self.rel
        best_ms = None
        best_st = None
        for rid in rule_ids:
            if rid == 0:
                order = sorted(jobs, key=lambda j: (rel[j], -ps[j]))
            elif rid == 1:
                order = sorted(jobs, key=lambda j: (rel[j], -rs[j]))
            elif rid == 2:
                order = sorted(jobs, key=lambda j: (rel[j], -(ps[j] * rs[j])))
            elif rid == 3:
                order = sorted(jobs, key=lambda j: (-(ps[j] * rs[j]), rel[j]))
            elif rid == 98:
                ms, st = self._sgs_seq(i, s, jobs)
                if best_ms is None or ms < best_ms:
                    best_ms, best_st = ms, st
                continue
            else:
                order = list(jobs)
                self.rng.shuffle(order)
                order.sort(key=lambda j: rel[j])
            ms, st = self._sgs(i, s, order)
            if best_ms is None or ms < best_ms:
                best_ms = ms
                best_st = st
        return best_ms, best_st

    # ------------------------------------------------------------------
    # Evaluation of complete / partial assignments
    # ------------------------------------------------------------------
    def full_evaluate(self):
        S = self.S
        scen_ms = [0] * S
        deadline = self.deadline
        rules = self.search_rules
        for i in range(self.m):
            jobs = sorted(self.fac_jobs[i])
            fmi = self.fm[i]
            fsi = self.fsched[i]
            for s in range(S):
                if deadline is not None and time.perf_counter() > deadline - 0.3:
                    ms, st = self.sched_fac(i, s, jobs, [98])
                else:
                    ms, st = self.sched_fac(i, s, jobs, rules)
                fmi[s] = ms
                fsi[s] = st
                if ms > scen_ms[s]:
                    scen_ms[s] = ms
        self.scen_ms = scen_ms
        obj = 0.0
        for s in range(S):
            obj += self.prob[s] * scen_ms[s]
        self.cur_obj = obj

    def eval_change(self, a, jobs_a, b, jobs_b, cutoff):
        """Evaluate objective when facility a holds jobs_a and b holds jobs_b
        (all other facilities unchanged). Aborts early above `cutoff`."""
        S = self.S
        m = self.m
        fm = self.fm
        prob = self.prob
        rules = self.search_rules
        la = sorted(jobs_a)
        lb = sorted(jobs_b)
        data_a = [None] * S
        data_b = [None] * S
        new_scen = [0] * S
        obj = 0.0
        for s in range(S):
            msa, sta = self.sched_fac(a, s, la, rules)
            msb, stb = self.sched_fac(b, s, lb, rules)
            data_a[s] = (msa, sta)
            data_b[s] = (msb, stb)
            best = msa if msa > msb else msb
            for i in range(m):
                if i != a and i != b:
                    v = fm[i][s]
                    if v > best:
                        best = v
            new_scen[s] = best
            obj += prob[s] * best
            if obj >= cutoff:
                return obj, None, None, None
        return obj, new_scen, data_a, data_b

    def apply_change(self, a, jobs_a, b, jobs_b, da, db, new_scen, obj):
        S = self.S
        self.fac_jobs[a] = set(jobs_a)
        self.fac_jobs[b] = set(jobs_b)
        fma = self.fm[a]
        fmb = self.fm[b]
        fsa = self.fsched[a]
        fsb = self.fsched[b]
        for s in range(S):
            fma[s] = da[s][0]
            fsa[s] = da[s][1]
            fmb[s] = db[s][0]
            fsb[s] = db[s][1]
        self.scen_ms = new_scen
        self.cur_obj = obj

    # ------------------------------------------------------------------
    # Construction heuristics
    # ------------------------------------------------------------------
    def greedy_assign(self, mode):
        n, m = self.n, self.m
        ep, req, C, rel = self.ep, self.req, self.C, self.rel
        load = [0.0] * m
        relmax = [0.0] * m
        assign = [-1] * n
        diff = [0.0] * n
        for j in range(n):
            diff[j] = min(
                ep[i][j] * req[i][j] / max(1.0, float(C[i]))
                for i in self.allowed[j]
            )
        if mode == 0:
            order = sorted(range(n), key=lambda j: -diff[j])
        else:
            order = sorted(range(n), key=lambda j: (rel[j], -diff[j]))
        for j in order:
            best_i = -1
            best_key = None
            for i in self.allowed[j]:
                ci = float(C[i]) if C[i] > 0 else 1.0
                nl = load[i] + ep[i][j] * req[i][j]
                rj = rel[j] + ep[i][j]
                nr = relmax[i] if relmax[i] > rj else rj
                lbi = nl / ci
                if nr > lbi:
                    lbi = nr
                cur = 0.0
                for ii in range(m):
                    if ii == i:
                        v = lbi
                    else:
                        cii = float(C[ii]) if C[ii] > 0 else 1.0
                        v = load[ii] / cii
                        if relmax[ii] > v:
                            v = relmax[ii]
                    if v > cur:
                        cur = v
                key = (cur, lbi) if mode == 0 else (lbi, cur)
                if best_key is None or key < best_key:
                    best_key = key
                    best_i = i
            i = best_i
            assign[j] = i
            load[i] += ep[i][j] * req[i][j]
            rj = rel[j] + ep[i][j]
            if rj > relmax[i]:
                relmax[i] = rj
        self.assign = assign
        self.fac_jobs = [set() for _ in range(m)]
        for j in range(n):
            self.fac_jobs[assign[j]].add(j)

    def random_assign(self):
        rng = self.rng
        self.assign = [
            self.allowed[j][rng.randrange(len(self.allowed[j]))]
            for j in range(self.n)
        ]
        self.fac_jobs = [set() for _ in range(self.m)]
        for j, i in enumerate(self.assign):
            self.fac_jobs[i].add(j)

    # ------------------------------------------------------------------
    # Local search
    # ------------------------------------------------------------------
    def update_crit(self):
        m, S = self.m, self.S
        cf = [0.0] * m
        fm = self.fm
        for s in range(S):
            ms = self.scen_ms[s]
            p = self.prob[s]
            if p == 0.0:
                continue
            for i in range(m):
                if fm[i][s] >= ms - 1e-9:
                    cf[i] += p
        self.crit = [cf[self.assign[j]] for j in range(self.n)]

    def local_search(self, deadline):
        eps = 1e-9
        rng = self.rng
        improved = True
        while improved:
            if time.perf_counter() >= deadline:
                return
            improved = False
            self.update_crit()
            order = list(range(self.n))
            rng.shuffle(order)
            order.sort(key=lambda j: -self.crit[j])
            for j in order:
                if time.perf_counter() >= deadline:
                    return
                a = self.assign[j]
                cf = list(self.cand_fac[j])
                rng.shuffle(cf)
                for b in cf:
                    if b == a:
                        continue
                    jobs_a = self.fac_jobs[a] - {j}
                    jobs_b = self.fac_jobs[b] | {j}
                    obj, ns, da, db = self.eval_change(
                        a, jobs_a, b, jobs_b, self.cur_obj - eps
                    )
                    if ns is not None and obj < self.cur_obj - eps:
                        self.assign[j] = b
                        self.apply_change(a, jobs_a, b, jobs_b, da, db, ns, obj)
                        improved = True
                        if obj < self.best_obj - eps:
                            self.incumbent()
                        break
            if not improved:
                improved = self.swap_pass(deadline)

    def swap_pass(self, deadline):
        eps = 1e-9
        n = self.n
        rng = self.rng
        budget = self.swap_budget
        cnt = 0
        order = sorted(range(n), key=lambda j: -self.crit[j])
        K = 14 if n > 14 else n
        for j1 in order:
            if cnt >= budget or time.perf_counter() >= deadline:
                break
            a = self.assign[j1]
            for _ in range(K):
                j2 = rng.randrange(n)
                if j2 == j1:
                    continue
                b = self.assign[j2]
                if b == a:
                    continue
                if b not in self.allowed_set[j1] or a not in self.allowed_set[j2]:
                    continue
                cnt += 1
                jobs_a = (self.fac_jobs[a] - {j1}) | {j2}
                jobs_b = (self.fac_jobs[b] - {j2}) | {j1}
                obj, ns, da, db = self.eval_change(
                    a, jobs_a, b, jobs_b, self.cur_obj - eps
                )
                if ns is not None and obj < self.cur_obj - eps:
                    self.assign[j1] = b
                    self.assign[j2] = a
                    self.apply_change(a, jobs_a, b, jobs_b, da, db, ns, obj)
                    if obj < self.best_obj - eps:
                        self.incumbent()
                    return True
        return False

    # ------------------------------------------------------------------
    # Perturbation and polishing
    # ------------------------------------------------------------------
    def perturb(self, strength):
        rng = self.rng
        n = self.n
        moved = 0
        tries = 0
        while moved < strength and tries < strength * 6:
            tries += 1
            j = rng.randrange(n)
            a = self.assign[j]
            cand = [b for b in self.cand_fac[j] if b != a]
            if not cand:
                continue
            b = cand[rng.randrange(len(cand))]
            jobs_a = self.fac_jobs[a] - {j}
            jobs_b = self.fac_jobs[b] | {j}
            obj, ns, da, db = self.eval_change(
                a, jobs_a, b, jobs_b, float("inf")
            )
            self.assign[j] = b
            self.apply_change(a, jobs_a, b, jobs_b, da, db, ns, obj)
            moved += 1

    def polish(self, deadline):
        """Re-schedule bottleneck facilities with extra (incl. randomized)
        priority rules to shave the scenario makespans."""
        eps = 1e-9
        S, m = self.S, self.m
        kmax = 1
        for i in range(m):
            li = len(self.fac_jobs[i])
            if li > kmax:
                kmax = li
        tries = int(2e5 / (kmax * kmax + 1.0))
        if tries < 2:
            tries = 2
        if tries > 25:
            tries = 25
        rules = [0, 1, 2, 3] + [99] * tries
        improved = True
        while improved and time.perf_counter() < deadline:
            improved = False
            for s in range(S):
                if time.perf_counter() >= deadline:
                    break
                sms = self.scen_ms[s]
                for i in range(m):
                    if time.perf_counter() >= deadline:
                        break
                    if self.fm[i][s] >= sms - 1e-9 and self.fac_jobs[i]:
                        jobs = sorted(self.fac_jobs[i])
                        ms, st = self.sched_fac(i, s, jobs, rules)
                        if ms < self.fm[i][s]:
                            self.fm[i][s] = ms
                            self.fsched[i][s] = st
                            improved = True
                self.scen_ms[s] = max(self.fm[i][s] for i in range(m))
        obj = 0.0
        for s in range(S):
            obj += self.prob[s] * self.scen_ms[s]
        self.cur_obj = obj
        if self.cur_obj < self.best_obj - eps:
            self.incumbent()

    # ------------------------------------------------------------------
    # Incumbent bookkeeping / output
    # ------------------------------------------------------------------
    def incumbent(self):
        self.best_obj = self.cur_obj
        self.best_assign = list(self.assign)
        self.best_fac_jobs = [set(x) for x in self.fac_jobs]
        self.best_fm = [row[:] for row in self.fm]
        self.best_fsched = [
            [(d.copy() if d is not None else None) for d in row]
            for row in self.fsched
        ]
        self.best_scen_ms = list(self.scen_ms)
        if self.logger is not None:
            sol = self.build_solution_from(self.assign, self.fsched, self.cur_obj)
            self.logger.log_solution(float(self.cur_obj), sol)

    def restore_best(self):
        self.assign = list(self.best_assign)
        self.fac_jobs = [set(x) for x in self.best_fac_jobs]
        self.fm = [row[:] for row in self.best_fm]
        self.fsched = [
            [(d.copy() if d is not None else None) for d in row]
            for row in self.best_fsched
        ]
        self.scen_ms = list(self.best_scen_ms)
        self.cur_obj = self.best_obj

    def build_solution_from(self, assign, fsched, obj):
        n, S = self.n, self.S
        sched = {}
        for s in range(S):
            proc_s = self.proc[s]
            ent = {}
            for j in range(n):
                i = assign[j]
                st = fsched[i][s][j]
                ent[str(j)] = {
                    "facility": int(i),
                    "start": st,
                    "finish": st + proc_s[i][j],
                }
            sched[str(s)] = ent
        return {
            "objective_value": float(obj),
            "assignment": {str(j): int(assign[j]) for j in range(n)},
            "schedule": sched,
        }

    def build_solution(self):
        return self.build_solution_from(self.assign, self.fsched, self.cur_obj)

    # ------------------------------------------------------------------
    # Main driver
    # ------------------------------------------------------------------
    def solve(self):
        t0 = time.perf_counter()
        total = float(self.time_limit)
        self.deadline = t0 + total
        deadline = self.deadline

        # --- initial construction ---
        self.greedy_assign(0)
        self.full_evaluate()
        eval_time = time.perf_counter() - t0
        self.incumbent()

        if 6.0 * eval_time < 0.4 * total and time.perf_counter() < deadline - 0.2:
            self.greedy_assign(1)
            self.full_evaluate()
            if self.cur_obj < self.best_obj - 1e-9:
                self.incumbent()
        self.restore_best()

        reserve = min(0.35 * total, max(0.5, 3.0 * eval_time + 0.3))
        work_deadline = deadline - reserve
        if work_deadline < time.perf_counter():
            work_deadline = time.perf_counter()

        # --- first descent ---
        self.local_search(work_deadline)

        # --- perturbation / restart loop ---
        it = 0
        while time.perf_counter() < work_deadline:
            it += 1
            self.restore_best()
            self.polish(work_deadline)
            if self.m > 1:
                if it % 6 == 0:
                    self.random_assign()
                    self.full_evaluate()
                else:
                    self.restore_best()
                    smax = max(3, min(15, self.n // 4 + 2))
                    self.perturb(self.rng.randint(2, smax))
                self.local_search(work_deadline)
            # For m == 1 the loop keeps polishing the single facility with
            # fresh randomized priority orders until the deadline.

        # --- final polish of the incumbent ---
        self.restore_best()
        self.polish(deadline - 0.1)
        self.restore_best()
        return self.build_solution()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", required=True)
    ap.add_argument("--solution_path", required=True)
    ap.add_argument("--time_limit", type=int, required=True)
    ap.add_argument("--log_path", default=None)
    args = ap.parse_args()

    with open(args.instance_path) as f:
        data = json.load(f)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    solver = StochasticScheduler(data, args.time_limit, logger)
    sol = solver.solve()

    with open(args.solution_path, "w") as f:
        json.dump(sol, f)


if __name__ == "__main__":
    main()
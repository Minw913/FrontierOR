import argparse
import json
import math
import random
import time

from solution_logger import SolutionLogger

BIGPEN = 1e6  # penalty for using more vehicles than available at a depot on a day
EPS = 1e-9


class Route:
    __slots__ = ('seq', 'load', 'cost', 'serv')

    def __init__(self, seq, load, cost, serv):
        self.seq = seq      # ordered list of customer ids
        self.load = load    # total demand
        self.cost = cost    # travel cost
        self.serv = serv    # total service time


class Solver:
    def __init__(self, inst, time_limit, logger):
        self.t0 = time.time()
        self.tl = time_limit
        self.logger = logger
        self.rng = random.Random(0)

        self.D = inst['d']
        self.N = inst['n']
        self.T = inst['t']
        self.M = inst['m']
        self.cap = inst['vehicle_capacity']
        L = inst.get('route_duration_limit', 0)
        self.L = float(L) if L and L > 0 else float('inf')

        deps = sorted(inst['depots'], key=lambda z: z['id'])
        cus = sorted(inst['customers'], key=lambda z: z['id'])
        self.dem = [c['demand'] for c in cus]
        self.serv = [c['service_duration'] for c in cus]
        self.pats = [[tuple(sorted(p)) for p in c['pattern_list']] for c in cus]

        dxy = [(d['x'], d['y']) for d in deps]
        cxy = [(c['x'], c['y']) for c in cus]
        self.dcd = [[math.hypot(dxy[j][0] - cxy[i][0], dxy[j][1] - cxy[i][1])
                     for i in range(self.N)] for j in range(self.D)]
        self.dcc = [[math.hypot(cxy[a][0] - cxy[b][0], cxy[a][1] - cxy[b][1])
                     for b in range(self.N)] for a in range(self.N)]

        self.routes = {(j, t): [] for j in range(self.D) for t in range(1, self.T + 1)}
        self.cdep = [-1] * self.N
        self.cpat = [None] * self.N

        self.best_cost = float('inf')
        self.best_snap = None

    # ---------------- utilities ----------------
    def time_up(self):
        return time.time() - self.t0 > self.tl - 1.0

    def dd(self, dep, a, b):
        if a is None:
            return 0.0 if b is None else self.dcd[dep][b]
        if b is None:
            return self.dcd[dep][a]
        return self.dcc[a][b]

    def rcost(self, dep, seq):
        if not seq:
            return 0.0
        c = self.dcd[dep][seq[0]] + self.dcd[dep][seq[-1]]
        for a, b in zip(seq, seq[1:]):
            c += self.dcc[a][b]
        return c

    def total_cost(self):
        tot = 0.0
        for (dep, t), rs in self.routes.items():
            for r in rs:
                tot += r.cost
            if len(rs) > self.M:
                tot += BIGPEN * (len(rs) - self.M)
        return tot

    def travel_cost(self):
        tot = 0.0
        for rs in self.routes.values():
            for r in rs:
                tot += r.cost
        return tot

    def snapshot(self):
        routes = {k: [Route(list(r.seq), r.load, r.cost, r.serv) for r in rs]
                  for k, rs in self.routes.items()}
        return (routes, list(self.cdep), list(self.cpat))

    def restore(self, snap):
        routes, cdep, cpat = snap
        self.routes = {k: [Route(list(r.seq), r.load, r.cost, r.serv) for r in rs]
                       for k, rs in routes.items()}
        self.cdep = list(cdep)
        self.cpat = list(cpat)

    # ---------------- insertion / removal ----------------
    def best_insert_day(self, c, dep, t):
        rs = self.routes[(dep, t)]
        dm = self.dem[c]
        sv = self.serv[c]
        best = None
        for ri, r in enumerate(rs):
            if r.load + dm > self.cap:
                continue
            maxdelta = self.L - (r.cost + r.serv + sv)
            if maxdelta < -1e-9:
                continue
            seq = r.seq
            n = len(seq)
            for pos in range(n + 1):
                prev = seq[pos - 1] if pos > 0 else None
                nxt = seq[pos] if pos < n else None
                delta = self.dd(dep, prev, c) + self.dd(dep, c, nxt) - self.dd(dep, prev, nxt)
                if delta <= maxdelta + 1e-9 and (best is None or delta < best[0]):
                    best = (delta, ri, pos)
        # new route option
        nd = 2.0 * self.dcd[dep][c]
        if dm <= self.cap and nd + sv <= self.L + 1e-9:
            pen = 0.0 if len(rs) < self.M else BIGPEN
            delta = nd + pen
            if best is None or delta < best[0]:
                best = (delta, -1, 0)
        return best

    def do_insert(self, dep, t, c, ri, pos):
        rs = self.routes[(dep, t)]
        if ri == -1:
            rs.append(Route([c], self.dem[c], 2.0 * self.dcd[dep][c], self.serv[c]))
        else:
            r = rs[ri]
            seq = r.seq
            prev = seq[pos - 1] if pos > 0 else None
            nxt = seq[pos] if pos < len(seq) else None
            r.cost += self.dd(dep, prev, c) + self.dd(dep, c, nxt) - self.dd(dep, prev, nxt)
            r.load += self.dem[c]
            r.serv += self.serv[c]
            seq.insert(pos, c)

    def remove_day(self, dep, t, c):
        rs = self.routes[(dep, t)]
        for ri, r in enumerate(rs):
            if c in r.seq:
                pos = r.seq.index(c)
                seq = r.seq
                prev = seq[pos - 1] if pos > 0 else None
                nxt = seq[pos + 1] if pos + 1 < len(seq) else None
                gain = self.dd(dep, prev, c) + self.dd(dep, c, nxt) - self.dd(dep, prev, nxt)
                seq.pop(pos)
                r.cost -= gain
                r.load -= self.dem[c]
                r.serv -= self.serv[c]
                if not seq:
                    rs.pop(ri)
                    if len(rs) >= self.M:  # route count was above the limit before removal
                        gain += BIGPEN
                return gain
        return 0.0

    def remove_full(self, c):
        gain = 0.0
        dep = self.cdep[c]
        for t in self.cpat[c]:
            gain += self.remove_day(dep, t, c)
        return gain

    def best_assignment(self, c):
        best = None
        for dep in range(self.D):
            for pat in self.pats[c]:
                tot = 0.0
                plan = []
                ok = True
                for t in pat:
                    b = self.best_insert_day(c, dep, t)
                    if b is None:
                        ok = False
                        break
                    tot += b[0]
                    plan.append((t, b[1], b[2]))
                    if best is not None and tot >= best[0]:
                        pass
                if ok and (best is None or tot < best[0]):
                    best = (tot, dep, pat, plan)
        return best

    def apply_assignment(self, c, dep, pat, plan):
        self.cdep[c] = dep
        self.cpat[c] = pat
        for (t, ri, pos) in plan:
            self.do_insert(dep, t, c, ri, pos)

    def force_insert(self, c, dep, pat):
        self.cdep[c] = dep
        self.cpat[c] = pat
        for t in pat:
            self.routes[(dep, t)].append(
                Route([c], self.dem[c], 2.0 * self.dcd[dep][c], self.serv[c]))

    def reassign(self, c):
        gain = self.remove_full(c)
        best = self.best_assignment(c)
        if best is None:
            self.force_insert(c, self.cdep[c], self.cpat[c])
            return 0.0
        self.apply_assignment(c, best[1], best[2], best[3])
        return best[0] - gain

    # ---------------- local search ----------------
    def two_opt_route(self, dep, r):
        seq = r.seq
        n = len(seq)
        if n < 3:
            return
        improved = True
        while improved:
            improved = False
            for i in range(n - 1):
                a = seq[i - 1] if i > 0 else None
                for j in range(i + 1, n):
                    dnode = seq[j + 1] if j + 1 < n else None
                    old = self.dd(dep, a, seq[i]) + self.dd(dep, seq[j], dnode)
                    new = self.dd(dep, a, seq[j]) + self.dd(dep, seq[i], dnode)
                    if new < old - 1e-9:
                        seq[i:j + 1] = reversed(seq[i:j + 1])
                        improved = True
        r.cost = self.rcost(dep, seq)

    def two_opt_all(self):
        for (dep, t), rs in self.routes.items():
            for r in rs:
                self.two_opt_route(dep, r)

    def improve_pass_loop(self, max_passes):
        order = list(range(self.N))
        for _ in range(max_passes):
            if self.time_up():
                return
            self.rng.shuffle(order)
            imp = False
            for c in order:
                if self.time_up():
                    return
                if self.reassign(c) < -1e-7:
                    imp = True
            self.two_opt_all()
            if not imp:
                return

    def perturb(self):
        n = self.N
        kmax = max(4, min(30, max(4, n // 3)))
        k = self.rng.randint(3, kmax) if kmax > 3 else min(3, n)
        k = min(k, n)
        seed = self.rng.randrange(n)
        others = sorted(range(n), key=lambda x: self.dcc[seed][x])
        S = others[:k]
        for c in S:
            self.remove_full(c)
        self.rng.shuffle(S)
        for c in S:
            b = self.best_assignment(c)
            if b is None:
                self.force_insert(c, self.cdep[c], self.cpat[c])
            else:
                self.apply_assignment(c, b[1], b[2], b[3])

    # ---------------- construction ----------------
    def construct(self):
        order = sorted(range(self.N),
                       key=lambda c: -min(self.dcd[j][c] for j in range(self.D)))
        for c in order:
            b = self.best_assignment(c)
            if b is None:
                dep = min(range(self.D), key=lambda j: self.dcd[j][c])
                self.force_insert(c, dep, self.pats[c][0])
            else:
                self.apply_assignment(c, b[1], b[2], b[3])

    # ---------------- solution output ----------------
    def build_solution(self):
        routes_out = []
        for dep in range(self.D):
            for t in range(1, self.T + 1):
                for v, r in enumerate(self.routes[(dep, t)]):
                    if r.seq:
                        routes_out.append({
                            'depot': dep,
                            'period': t,
                            'vehicle': v,
                            'customers': list(r.seq)
                        })
        assignments = {str(c): {'depot': int(self.cdep[c]),
                                'pattern': [int(x) for x in self.cpat[c]]}
                       for c in range(self.N)}
        return {'objective_value': self.travel_cost(),
                'routes': routes_out,
                'assignments': assignments}

    def log_best(self):
        if self.logger:
            sol = self.build_solution()
            self.logger.log_solution(sol['objective_value'], sol)

    # ---------------- main loop ----------------
    def solve(self):
        self.construct()
        self.two_opt_all()
        self.improve_pass_loop(max_passes=50)
        cur = self.total_cost()
        self.best_cost = cur
        self.best_snap = self.snapshot()
        self.log_best()

        while not self.time_up():
            self.perturb()
            self.improve_pass_loop(max_passes=3)
            cur = self.total_cost()
            if cur < self.best_cost - 1e-7:
                self.best_cost = cur
                self.best_snap = self.snapshot()
                self.log_best()
            elif cur > self.best_cost * 1.03 + 1e-6:
                self.restore(self.best_snap)

        self.restore(self.best_snap)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--instance_path', type=str, required=True)
    ap.add_argument('--solution_path', type=str, required=True)
    ap.add_argument('--time_limit', type=int, default=60)
    ap.add_argument('--log_path', type=str, default=None)
    args = ap.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        inst = json.load(f)

    solver = Solver(inst, args.time_limit, logger)
    solver.solve()
    sol = solver.build_solution()

    with open(args.solution_path, 'w') as f:
        json.dump(sol, f, indent=2)


if __name__ == '__main__':
    main()
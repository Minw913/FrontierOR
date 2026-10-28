import argparse
import json
import math
import random
import time

from solution_logger import SolutionLogger

EPS = 1e-6


class _TimeUp(Exception):
    pass


class Solver:
    def __init__(self, instance, time_limit, logger):
        self.logger = logger
        self.start = time.time()
        self.time_limit = max(1, int(time_limit))
        reserve = max(0.3, min(5.0, 0.05 * self.time_limit))
        self.deadline = self.start + max(0.2, self.time_limit - reserve)
        self._evals = 0

        inst = instance
        self.n = int(inst["num_customers"])
        self.D = int(inst["num_days"])
        self.V = int(inst["num_vehicles"])
        self.cap = float(inst["vehicle_capacity"])
        self.maxdur = float(inst["max_route_duration"])
        depot = inst["depot"]
        self.dep_e = float(depot["time_window"][0])
        self.dep_l = float(depot["time_window"][1])
        custs = inst["customers"]
        self.cid = [c["id"] for c in custs]
        m = self.n + 1
        self.DEP = self.n
        xs = [float(c["x"]) for c in custs] + [float(depot["x"])]
        ys = [float(c["y"]) for c in custs] + [float(depot["y"])]
        self.dist = [[0.0] * m for _ in range(m)]
        for i in range(m):
            di = self.dist[i]
            xi = xs[i]
            yi = ys[i]
            for j in range(m):
                dx = xi - xs[j]
                dy = yi - ys[j]
                di[j] = math.sqrt(dx * dx + dy * dy)
        self.serv = [float(c.get("service_time", 0)) for c in custs] + [float(depot.get("service_time", 0))]
        self.twe = [float(c["time_window"][0]) for c in custs]
        self.twl = [float(c["time_window"][1]) for c in custs]
        self.dpd = [float(c.get("demand_per_day", 0)) for c in custs]

        self.sched = []
        dayvals = []
        for c in custs:
            slist = []
            for s in c.get("schedules", []) or []:
                parts = []
                for p in s.get("parts", []) or []:
                    vd = int(p["visit_day"])
                    dcv = p.get("days_covered", 1)
                    dpv = p.get("demand_per_visit")
                    if dpv is None:
                        dpv = c.get("demand_per_day", 0) * dcv
                    parts.append({"visit_day": vd, "days_covered": dcv, "demand_per_visit": dpv})
                if not parts:
                    for vd in s.get("days", []) or []:
                        parts.append({"visit_day": int(vd), "days_covered": 1,
                                      "demand_per_visit": c.get("demand_per_day", 0)})
                days = s.get("days")
                if days is None:
                    days = [p["visit_day"] for p in parts]
                vf = s.get("visit_frequency")
                if vf is None:
                    vf = len(days)
                slist.append({"days": list(days), "visit_frequency": vf, "parts": parts})
                for p in parts:
                    dayvals.append(p["visit_day"])
            if not slist:
                slist = [{"days": [], "visit_frequency": 0, "parts": []}]
            self.sched.append(slist)
        self.base = 0 if (dayvals and min(dayvals) == 0) else 1

        self.choice = [0] * self.n
        self.routes = [[] for _ in range(self.D)]
        self.loads = [[] for _ in range(self.D)]
        self.dem = [dict() for _ in range(self.D)]
        self.part = [dict() for _ in range(self.D)]
        self.best_obj = float("inf")
        self.best_choice = None
        self.best_routes = None

    # ------------------------------------------------------------------
    # Route evaluation
    # ------------------------------------------------------------------
    def route_cost(self, r):
        dist = self.dist
        DEP = self.DEP
        prev = DEP
        tot = 0.0
        for c in r:
            tot += dist[prev][c]
            prev = c
        tot += dist[prev][DEP]
        return tot

    def eval_route(self, r):
        """Return (feasible, cost). Travel time i->j = dist(i,j) + service(i)."""
        self._evals += 1
        if (self._evals & 127) == 0 and time.time() > self.deadline:
            raise _TimeUp()
        dist = self.dist
        serv = self.serv
        twe = self.twe
        twl = self.twl
        DEP = self.DEP
        L = len(r)
        prev = DEP
        cost = 0.0
        for c in r:
            cost += dist[prev][c]
            prev = c
        cost += dist[prev][DEP]
        # forward pass from earliest depot departure
        t = self.dep_e
        prev = DEP
        for k in range(L):
            c = r[k]
            arr = t + dist[prev][c] + serv[prev]
            st = twe[c] if arr < twe[c] else arr
            if st > twl[c] + EPS:
                return (False, cost)
            t = st
            prev = c
        ret = t + dist[prev][DEP] + serv[prev]
        if ret > self.dep_l + EPS:
            return (False, cost)
        # backward pass: latest feasible service-start times
        latest = [0.0] * L
        c = r[L - 1]
        v = self.dep_l - dist[c][DEP] - serv[c]
        latest[L - 1] = twl[c] if twl[c] < v else v
        for k in range(L - 2, -1, -1):
            c = r[k]
            nx = r[k + 1]
            v = latest[k + 1] - dist[c][nx] - serv[c]
            latest[k] = twl[c] if twl[c] < v else v
        # latest feasible depot departure (minimizes duration)
        acc = 0.0
        prev = DEP
        t0 = 1e18
        for k in range(L):
            c = r[k]
            acc += dist[prev][c] + serv[prev]
            v = latest[k] - acc
            if v < t0:
                t0 = v
            prev = c
        if t0 < self.dep_e:
            t0 = self.dep_e
        t = t0
        prev = DEP
        for k in range(L):
            c = r[k]
            arr = t + dist[prev][c] + serv[prev]
            st = twe[c] if arr < twe[c] else arr
            if st > twl[c] + EPS:
                return (False, cost)
            t = st
            prev = c
        ret = t + dist[prev][DEP] + serv[prev]
        if ret > self.dep_l + EPS:
            return (False, cost)
        if ret - t0 > self.maxdur + EPS:
            return (False, cost)
        return (True, cost)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def initial_choice(self):
        choice = [0] * self.n
        dayload = [0.0] * self.D
        totalcap = self.V * self.cap
        order = sorted(range(self.n), key=lambda c: -(self.dpd[c] * self.D))
        for c in order:
            best = None
            for si, s in enumerate(self.sched[c]):
                parts = s["parts"]
                maxd = max((p["demand_per_visit"] for p in parts), default=0.0)
                pen = 1 if maxd > self.cap + EPS else 0
                over = 0.0
                for p in parts:
                    d = p["visit_day"] - self.base
                    if 0 <= d < self.D:
                        ov = dayload[d] + p["demand_per_visit"] - totalcap
                        if ov > 0:
                            over += ov
                key = (pen, over, s["visit_frequency"], maxd)
                if best is None or key < best[0]:
                    best = (key, si)
            si = best[1]
            choice[c] = si
            for p in self.sched[c][si]["parts"]:
                d = p["visit_day"] - self.base
                if 0 <= d < self.D:
                    dayload[d] += p["demand_per_visit"]
        return choice

    def best_insertion(self, routes, loads, c, dc, max_routes):
        best = None
        cap = self.cap
        for ri, r in enumerate(routes):
            if loads[ri] + dc > cap + EPS:
                continue
            base = self.route_cost(r)
            for pos in range(len(r) + 1):
                nr = r[:pos] + [c] + r[pos:]
                feas, cst = self.eval_route(nr)
                if not feas:
                    continue
                delta = cst - base
                if best is None or delta < best[0] - 1e-12:
                    best = (delta, ri, pos)
        if best is None and len(routes) < max_routes:
            feas, cst = self.eval_route([c])
            if feas:
                best = (cst, -1, 0)
        return best

    def build_state(self):
        self.routes = [[] for _ in range(self.D)]
        self.loads = [[] for _ in range(self.D)]
        self.dem = [dict() for _ in range(self.D)]
        self.part = [dict() for _ in range(self.D)]
        for c, si in enumerate(self.choice):
            for p in self.sched[c][si]["parts"]:
                d = p["visit_day"] - self.base
                if 0 <= d < self.D:
                    self.dem[d][c] = p["demand_per_visit"]
                    self.part[d][c] = p
        for d in range(self.D):
            self.construct_day(d)

    def construct_day(self, d):
        custs = list(self.dem[d].keys())
        routes = []
        loads = []
        order = sorted(custs, key=lambda c: (self.twl[c], -self.dem[d][c]))
        for c in order:
            dc = self.dem[d][c]
            ins = self.best_insertion(routes, loads, c, dc, 10 ** 9)
            if ins is None:
                routes.append([c])
                loads.append(dc)
            else:
                _, ri, pos = ins
                if ri == -1:
                    routes.append([c])
                    loads.append(dc)
                else:
                    routes[ri].insert(pos, c)
                    loads[ri] += dc
        self.routes[d] = routes
        self.loads[d] = loads
        self.reduce_routes(d)

    def reduce_routes(self, d):
        routes = self.routes[d]
        loads = self.loads[d]
        while len(routes) > self.V:
            ri = min(range(len(routes)), key=lambda i: loads[i])
            trial_r = [r[:] for r in routes]
            trial_l = loads[:]
            saved = trial_r.pop(ri)
            trial_l.pop(ri)
            ok = True
            for c in sorted(saved, key=lambda x: -self.dem[d][x]):
                ins = self.best_insertion(trial_r, trial_l, c, self.dem[d][c], self.V)
                if ins is None or ins[1] == -1:
                    ok = False
                    break
                _, rj, pos = ins
                trial_r[rj].insert(pos, c)
                trial_l[rj] += self.dem[d][c]
            if ok:
                self.routes[d] = trial_r
                self.loads[d] = trial_l
                routes = trial_r
                loads = trial_l
            else:
                break

    def find_route(self, d, c):
        for ri, r in enumerate(self.routes[d]):
            if c in r:
                return ri
        return -1

    # ------------------------------------------------------------------
    # Repair (schedule changes to eliminate excess routes)
    # ------------------------------------------------------------------
    def repair_feasibility(self):
        for d in range(self.D):
            guard = 0
            while len(self.routes[d]) > self.V and guard < 200:
                guard += 1
                best = None
                for c in list(self.dem[d].keys()):
                    for si in range(len(self.sched[c])):
                        if si == self.choice[c]:
                            continue
                        skip = False
                        for p in self.sched[c][si]["parts"]:
                            if p["visit_day"] - self.base == d:
                                skip = True
                                break
                        if skip:
                            continue
                        res = self.eval_schedule_move(c, si)
                        if res is None:
                            continue
                        delta, plan = res
                        removes = any(op[0] == d and op[1] == "remove_route" for op in plan)
                        key = (0 if removes else 1, delta)
                        if best is None or key < best[0]:
                            best = (key, c, si, plan)
                if best is None:
                    break
                self.apply_plan(best[1], best[2], best[3])
                self.reduce_routes(d)

    # ------------------------------------------------------------------
    # Routing local search operators (per day)
    # ------------------------------------------------------------------
    def two_opt_day(self, d):
        routes = self.routes[d]
        for ri, r in enumerate(routes):
            L = len(r)
            if L < 3:
                continue
            base = self.route_cost(r)
            for i in range(L - 1):
                for j in range(i + 1, L):
                    nr = r[:i] + r[i:j + 1][::-1] + r[j + 1:]
                    feas, cst = self.eval_route(nr)
                    if feas and cst < base - 1e-9:
                        routes[ri] = nr
                        return True
        return False

    def relocate_day(self, d):
        routes = self.routes[d]
        loads = self.loads[d]
        dem = self.dem[d]
        cap = self.cap
        R = len(routes)
        costs = [self.route_cost(r) for r in routes]
        for ri in range(R):
            r = routes[ri]
            base_r = costs[ri]
            for i in range(len(r)):
                c = r[i]
                dc = dem[c]
                rem = r[:i] + r[i + 1:]
                cost_rem = self.route_cost(rem) if rem else 0.0
                for rj in range(R):
                    if rj == ri:
                        tgt = rem
                        base_t = 0.0
                    else:
                        if loads[rj] + dc > cap + EPS:
                            continue
                        tgt = routes[rj]
                        base_t = costs[rj]
                    for pos in range(len(tgt) + 1):
                        if rj == ri and pos == i:
                            continue
                        nr = tgt[:pos] + [c] + tgt[pos:]
                        feas, cst = self.eval_route(nr)
                        if not feas:
                            continue
                        if rj == ri:
                            delta = cst - base_r
                        else:
                            delta = cost_rem + cst - base_r - base_t
                        if delta < -1e-9:
                            if rj == ri:
                                routes[ri] = nr
                            else:
                                routes[rj] = nr
                                loads[rj] += dc
                                if rem:
                                    routes[ri] = rem
                                    loads[ri] -= dc
                                else:
                                    del routes[ri]
                                    del loads[ri]
                            return True
        return False

    def or_opt_day(self, d):
        routes = self.routes[d]
        loads = self.loads[d]
        dem = self.dem[d]
        cap = self.cap
        R = len(routes)
        costs = [self.route_cost(r) for r in routes]
        for sl in (2, 3):
            for ri in range(R):
                r = routes[ri]
                L = len(r)
                if L < sl:
                    continue
                base_r = costs[ri]
                for i in range(L - sl + 1):
                    seg = r[i:i + sl]
                    sd = sum(dem[x] for x in seg)
                    rem = r[:i] + r[i + sl:]
                    cost_rem = self.route_cost(rem) if rem else 0.0
                    for rj in range(R):
                        if rj == ri:
                            tgt = rem
                            base_t = 0.0
                        else:
                            if loads[rj] + sd > cap + EPS:
                                continue
                            tgt = routes[rj]
                            base_t = costs[rj]
                        for pos in range(len(tgt) + 1):
                            if rj == ri and pos == i:
                                continue
                            nr = tgt[:pos] + seg + tgt[pos:]
                            feas, cst = self.eval_route(nr)
                            if not feas:
                                continue
                            if rj == ri:
                                delta = cst - base_r
                            else:
                                delta = cost_rem + cst - base_r - base_t
                            if delta < -1e-9:
                                if rj == ri:
                                    routes[ri] = nr
                                else:
                                    routes[rj] = nr
                                    loads[rj] += sd
                                    if rem:
                                        routes[ri] = rem
                                        loads[ri] -= sd
                                    else:
                                        del routes[ri]
                                        del loads[ri]
                                return True
        return False

    def swap_day(self, d):
        routes = self.routes[d]
        loads = self.loads[d]
        dem = self.dem[d]
        cap = self.cap
        R = len(routes)
        costs = [self.route_cost(r) for r in routes]
        for ri in range(R):
            for rj in range(ri, R):
                if ri == rj:
                    r = routes[ri]
                    L = len(r)
                    base = costs[ri]
                    for i in range(L):
                        for j in range(i + 1, L):
                            nr = r[:]
                            nr[i], nr[j] = nr[j], nr[i]
                            feas, cst = self.eval_route(nr)
                            if feas and cst < base - 1e-9:
                                routes[ri] = nr
                                return True
                else:
                    a = routes[ri]
                    b = routes[rj]
                    base = costs[ri] + costs[rj]
                    for i in range(len(a)):
                        c1 = a[i]
                        d1 = dem[c1]
                        for j in range(len(b)):
                            c2 = b[j]
                            d2 = dem[c2]
                            if loads[ri] - d1 + d2 > cap + EPS:
                                continue
                            if loads[rj] - d2 + d1 > cap + EPS:
                                continue
                            na = a[:]
                            nb = b[:]
                            na[i] = c2
                            nb[j] = c1
                            fa, ca = self.eval_route(na)
                            if not fa:
                                continue
                            fb, cb = self.eval_route(nb)
                            if not fb:
                                continue
                            if ca + cb < base - 1e-9:
                                routes[ri] = na
                                routes[rj] = nb
                                loads[ri] += d2 - d1
                                loads[rj] += d1 - d2
                                return True
        return False

    def tail_exchange_day(self, d):
        routes = self.routes[d]
        loads = self.loads[d]
        dem = self.dem[d]
        R = len(routes)
        if R < 2:
            return False
        costs = [self.route_cost(r) for r in routes]
        cap = self.cap
        for ri in range(R):
            a = routes[ri]
            La = len(a)
            sufa = [0.0] * (La + 1)
            for k in range(La - 1, -1, -1):
                sufa[k] = sufa[k + 1] + dem[a[k]]
            for rj in range(ri + 1, R):
                b = routes[rj]
                Lb = len(b)
                sufb = [0.0] * (Lb + 1)
                for k in range(Lb - 1, -1, -1):
                    sufb[k] = sufb[k + 1] + dem[b[k]]
                base = costs[ri] + costs[rj]
                for i in range(La + 1):
                    damoved = sufa[i]
                    for j in range(Lb + 1):
                        if (i == La and j == Lb) or (i == 0 and j == Lb) or (i == La and j == 0):
                            continue
                        dbmoved = sufb[j]
                        nla = loads[ri] - damoved + dbmoved
                        nlb = loads[rj] - dbmoved + damoved
                        if nla > cap + EPS or nlb > cap + EPS:
                            continue
                        na = a[:i] + b[j:]
                        nb = b[:j] + a[i:]
                        if not na or not nb:
                            continue
                        fa, ca = self.eval_route(na)
                        if not fa:
                            continue
                        fb, cb = self.eval_route(nb)
                        if not fb:
                            continue
                        if ca + cb < base - 1e-9:
                            routes[ri] = na
                            routes[rj] = nb
                            loads[ri] = nla
                            loads[rj] = nlb
                            return True
        return False

    def improve_day(self, d):
        if not self.routes[d]:
            return
        while True:
            if time.time() > self.deadline:
                return
            if self.two_opt_day(d):
                continue
            if self.relocate_day(d):
                continue
            if self.or_opt_day(d):
                continue
            if self.swap_day(d):
                continue
            if self.tail_exchange_day(d):
                continue
            return

    # ------------------------------------------------------------------
    # Schedule selection moves
    # ------------------------------------------------------------------
    def eval_schedule_move(self, c, si):
        cur = self.choice[c]
        curparts = {}
        for p in self.sched[c][cur]["parts"]:
            curparts[p["visit_day"] - self.base] = p
        altparts = {}
        for p in self.sched[c][si]["parts"]:
            altparts[p["visit_day"] - self.base] = p
        delta = 0.0
        plan = []
        days = set(curparts) | set(altparts)
        for d in days:
            if d < 0 or d >= self.D:
                return None
            routes = self.routes[d]
            loads = self.loads[d]
            cp = curparts.get(d)
            ap = altparts.get(d)
            if cp is not None and ap is not None:
                if ap["demand_per_visit"] != cp["demand_per_visit"]:
                    ri = self.find_route(d, c)
                    if ri < 0:
                        return None
                    nd = loads[ri] - cp["demand_per_visit"] + ap["demand_per_visit"]
                    if nd > self.cap + EPS:
                        return None
                plan.append((d, "stay", ap))
            elif cp is not None:
                ri = self.find_route(d, c)
                if ri < 0:
                    return None
                r = routes[ri]
                idx = r.index(c)
                base = self.route_cost(r)
                if len(r) == 1:
                    delta -= base
                    plan.append((d, "remove_route", ri))
                else:
                    nr = r[:idx] + r[idx + 1:]
                    cst = self.route_cost(nr)
                    delta += cst - base
                    plan.append((d, "remove", ri, idx))
            else:
                ins = self.best_insertion(routes, loads, c, ap["demand_per_visit"], self.V)
                if ins is None:
                    return None
                dlt, ri, pos = ins
                delta += dlt
                plan.append((d, "insert", ri, pos, ap))
        return (delta, plan)

    def apply_plan(self, c, si, plan):
        for op in plan:
            d = op[0]
            kind = op[1]
            if kind == "stay":
                ap = op[2]
                old = self.part[d][c]
                dd = ap["demand_per_visit"] - old["demand_per_visit"]
                if dd != 0:
                    ri = self.find_route(d, c)
                    if ri >= 0:
                        self.loads[d][ri] += dd
                self.dem[d][c] = ap["demand_per_visit"]
                self.part[d][c] = ap
            elif kind == "remove":
                ri = op[2]
                idx = op[3]
                self.loads[d][ri] -= self.dem[d][c]
                del self.routes[d][ri][idx]
                del self.dem[d][c]
                del self.part[d][c]
            elif kind == "remove_route":
                ri = op[2]
                del self.routes[d][ri]
                del self.loads[d][ri]
                del self.dem[d][c]
                del self.part[d][c]
            elif kind == "insert":
                ri = op[2]
                pos = op[3]
                ap = op[4]
                if ri == -1:
                    self.routes[d].append([c])
                    self.loads[d].append(ap["demand_per_visit"])
                else:
                    self.routes[d][ri].insert(pos, c)
                    self.loads[d][ri] += ap["demand_per_visit"]
                self.dem[d][c] = ap["demand_per_visit"]
                self.part[d][c] = ap
        self.choice[c] = si

    def schedule_pass(self):
        order = list(range(self.n))
        random.shuffle(order)
        for c in order:
            if time.time() > self.deadline:
                return
            cur = self.choice[c]
            best = None
            for si in range(len(self.sched[c])):
                if si == cur:
                    continue
                res = self.eval_schedule_move(c, si)
                if res is None:
                    continue
                delta, plan = res
                if delta < -1e-6 and (best is None or delta < best[0]):
                    best = (delta, si, plan)
            if best is not None:
                self.apply_plan(c, best[1], best[2])
                days = set()
                for p in self.sched[c][best[1]]["parts"]:
                    days.add(p["visit_day"] - self.base)
                for p in self.sched[c][cur]["parts"]:
                    days.add(p["visit_day"] - self.base)
                for d in days:
                    if 0 <= d < self.D:
                        self.improve_day(d)

    # ------------------------------------------------------------------
    # Perturbation
    # ------------------------------------------------------------------
    def perturb(self):
        k = random.randint(2, 5)
        applied = 0
        tries = 0
        while applied < k and tries < 40:
            tries += 1
            c = random.randrange(self.n)
            if len(self.sched[c]) < 2:
                continue
            si = random.randrange(len(self.sched[c]))
            if si == self.choice[c]:
                continue
            res = self.eval_schedule_move(c, si)
            if res is None:
                continue
            self.apply_plan(c, si, res[1])
            applied += 1
        # destroy & repair a random day
        d = random.randrange(self.D)
        custs = list(self.dem[d].keys())
        if custs:
            kk = min(len(custs), random.randint(2, max(2, len(custs) // 3 + 1)))
            removed = random.sample(custs, kk)
            saved_routes = [r[:] for r in self.routes[d]]
            saved_loads = self.loads[d][:]
            saved_dem = dict(self.dem[d])
            saved_part = dict(self.part[d])
            for c in removed:
                ri = self.find_route(d, c)
                if ri < 0:
                    continue
                r = self.routes[d][ri]
                self.loads[d][ri] -= self.dem[d][c]
                r.remove(c)
                if not r:
                    del self.routes[d][ri]
                    del self.loads[d][ri]
                del self.dem[d][c]
                del self.part[d][c]
            ok = True
            for c in removed:
                ins = self.best_insertion(self.routes[d], self.loads[d], c, saved_dem[c], self.V)
                if ins is None:
                    ok = False
                    break
                _, ri, pos = ins
                if ri == -1:
                    self.routes[d].append([c])
                    self.loads[d].append(saved_dem[c])
                else:
                    self.routes[d][ri].insert(pos, c)
                    self.loads[d][ri] += saved_dem[c]
                self.dem[d][c] = saved_dem[c]
                self.part[d][c] = saved_part[c]
            if not ok:
                self.routes[d] = saved_routes
                self.loads[d] = saved_loads
                self.dem[d] = saved_dem
                self.part[d] = saved_part

    # ------------------------------------------------------------------
    # Bookkeeping
    # ------------------------------------------------------------------
    def total_cost(self):
        tot = 0.0
        for d in range(self.D):
            for r in self.routes[d]:
                tot += self.route_cost(r)
        return tot

    def save_best(self, obj):
        self.best_obj = obj
        self.best_choice = self.choice[:]
        self.best_routes = [[r[:] for r in self.routes[d]] for d in range(self.D)]

    def restore_best(self):
        self.choice = self.best_choice[:]
        self.dem = [dict() for _ in range(self.D)]
        self.part = [dict() for _ in range(self.D)]
        for c, si in enumerate(self.choice):
            for p in self.sched[c][si]["parts"]:
                d = p["visit_day"] - self.base
                if 0 <= d < self.D:
                    self.dem[d][c] = p["demand_per_visit"]
                    self.part[d][c] = p
        self.routes = [[r[:] for r in self.best_routes[d]] for d in range(self.D)]
        self.loads = [[sum(self.dem[d][c] for c in r) for r in self.routes[d]]
                      for d in range(self.D)]

    def log_best(self):
        if self.logger:
            sol = self.build_solution(self.best_choice, self.best_routes)
            self.logger.log_solution(self.best_obj, sol)

    # ------------------------------------------------------------------
    # Main solve loop
    # ------------------------------------------------------------------
    def solve(self):
        try:
            self.choice = self.initial_choice()
            self.build_state()
            self.repair_feasibility()
            cur = self.total_cost()
            self.save_best(cur)
            self.log_best()
            for d in range(self.D):
                self.improve_day(d)
            cur = self.total_cost()
            if cur < self.best_obj - 1e-6:
                self.save_best(cur)
                self.log_best()
            stall = 0
            while time.time() < self.deadline:
                prev = cur
                for d in range(self.D):
                    self.improve_day(d)
                self.schedule_pass()
                cur = self.total_cost()
                if cur < self.best_obj - 1e-6:
                    self.save_best(cur)
                    self.log_best()
                    stall = 0
                if cur >= prev - 1e-6:
                    stall += 1
                    if stall >= 4:
                        self.restore_best()
                        cur = self.best_obj
                        stall = 0
                    self.perturb()
                    cur = self.total_cost()
                else:
                    stall = 0
        except _TimeUp:
            pass
        if self.best_choice is None:
            self.best_choice = self.choice[:]
            self.best_routes = [[r[:] for r in self.routes[d]] for d in range(self.D)]
        return self.build_solution(self.best_choice, self.best_routes)

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    def build_solution(self, choice, routes):
        parts_by_day = [dict() for _ in range(self.D)]
        for c, si in enumerate(choice):
            for p in self.sched[c][si]["parts"]:
                d = p["visit_day"] - self.base
                if 0 <= d < self.D:
                    parts_by_day[d][c] = p
        routes_out = {}
        total = 0.0
        for d in range(self.D):
            arr = []
            pmap = parts_by_day[d]
            for r in routes[d]:
                cost = self.route_cost(r)
                dem = 0
                pr = {}
                seq = []
                for c in r:
                    seq.append(self.cid[c])
                    p = pmap[c]
                    dem += p["demand_per_visit"]
                    pr[str(self.cid[c])] = {
                        "visit_day": p["visit_day"],
                        "days_covered": p["days_covered"],
                        "demand_per_visit": p["demand_per_visit"],
                    }
                arr.append({"customers": seq, "cost": cost, "demand": dem, "parts": pr})
                total += cost
            routes_out[str(self.base + d)] = arr
        sched_out = {}
        for c, si in enumerate(choice):
            s = self.sched[c][si]
            sched_out[str(self.cid[c])] = {
                "schedule_index": si,
                "days": list(s["days"]),
                "visit_frequency": s["visit_frequency"],
            }
        return {"objective_value": total, "routes": routes_out, "schedules": sched_out}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()
    with open(args.instance_path, "r") as f:
        instance = json.load(f)
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    random.seed(0)
    solver = Solver(instance, args.time_limit, logger)
    solution = solver.solve()
    with open(args.solution_path, "w") as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()
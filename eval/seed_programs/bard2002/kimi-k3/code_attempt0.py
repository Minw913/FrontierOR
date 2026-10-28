import argparse
import json
import math
import random
import time
from functools import lru_cache

from solution_logger import SolutionLogger

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception:
    gp = None
    GRB = None


class MinFleetVRPTW:
    def __init__(self, instance, time_limit, logger):
        self.instance = instance
        self.time_limit = max(0, int(time_limit))
        self.logger = logger
        self.rng = random.Random(0)

        start = time.monotonic()
        reserve = 0.0 if self.time_limit <= 0 else max(0.05, min(1.0, 0.03 * self.time_limit))
        self.deadline = start + max(0.0, self.time_limit - reserve)

        self._load_instance(instance)
        self.route_pool = {}
        self.pool_cap = 30000
        self.best_key = None
        self.best_routes = None
        self.best_dict = None

    @staticmethod
    def _trunc1(x):
        return math.floor(x * 10.0 + 1e-9) / 10.0

    def _load_instance(self, inst):
        self.n = int(inst.get("num_customers", 0))
        self.Q = int(inst.get("vehicle_capacity", 0))
        depot = inst.get("depot", {})
        customers = list(inst.get("customers", []))
        customers.sort(key=lambda c: int(c.get("id", 0)))

        self.customer_ids = [int(c.get("id")) for c in customers]
        self.node_id = [0] + self.customer_ids
        self.ids = list(range(1, self.n + 1))

        self.x = [float(depot.get("x", 0.0))]
        self.y = [float(depot.get("y", 0.0))]
        self.q = [int(depot.get("demand", 0))]
        self.r = [int(depot.get("ready_time", 0))]
        horizon = int(inst.get("scheduling_horizon", depot.get("due_date", 0)))
        self.r0 = int(depot.get("ready_time", 0))
        self.l0 = min(int(depot.get("due_date", horizon)), horizon)
        self.l = [self.l0]
        self.s = [int(depot.get("service_time", 0))]

        for c in customers:
            self.x.append(float(c.get("x", 0.0)))
            self.y.append(float(c.get("y", 0.0)))
            self.q.append(int(c.get("demand", 0)))
            self.r.append(int(c.get("ready_time", 0)))
            self.l.append(int(c.get("due_date", 0)))
            self.s.append(int(c.get("service_time", 0)))

        dm = inst.get("distance_matrix")
        ok = isinstance(dm, list) and len(dm) == self.n + 1
        if ok:
            for row in dm:
                if not isinstance(row, list) or len(row) != self.n + 1:
                    ok = False
                    break
        if ok:
            self.D = [[float(v) for v in row] for row in dm]
        else:
            self.D = [[0.0 for _ in range(self.n + 1)] for __ in range(self.n + 1)]
            for i in range(self.n + 1):
                for j in range(i + 1, self.n + 1):
                    d = self._trunc1(math.hypot(self.x[i] - self.x[j], self.y[i] - self.y[j]))
                    self.D[i][j] = d
                    self.D[j][i] = d

        self.dist_one_decimal = all(
            abs(self.D[i][j] * 10.0 - round(self.D[i][j] * 10.0)) < 1e-8
            for i in range(self.n + 1)
            for j in range(self.n + 1)
        )

        includes_service = bool(inst.get("travel_time_includes_service_time", False))
        self.T = [[0.0 for _ in range(self.n + 1)] for __ in range(self.n + 1)]
        for i in range(self.n + 1):
            for j in range(self.n + 1):
                if i == j:
                    continue
                travel = self.D[i][j]
                if not includes_service:
                    travel += self.s[i]
                diff = float(self.r[j] - self.l[i])
                if diff > travel:
                    travel = diff
                self.T[i][j] = float(travel)

        self.maxD = max((max(row) for row in self.D), default=0.0)
        total_demand = sum(self.q[1:])
        self.lb_routes = (total_demand + self.Q - 1) // self.Q if self.Q > 0 else self.n
        self.bigM = float(self.maxD * max(1, 2 * self.n) + 1.0)

        neigh_k = 30 if self.n <= 200 else 15
        self.neighbors = {}
        for i in self.ids:
            arr = [j for j in self.ids if j != i and self.q[i] + self.q[j] <= self.Q]
            arr.sort(key=lambda j: (self.D[i][j] + self.D[j][i], self.T[i][j] + self.T[j][i], self.l[j]))
            self.neighbors[i] = arr[:neigh_k]

    def time_left(self):
        return self.deadline - time.monotonic()

    @lru_cache(maxsize=250000)
    def eval_route(self, rt):
        rt = tuple(int(v) for v in rt)
        tol = 1e-7
        dist = 0.0
        load = 0
        deps = []
        loads = []
        prev = 0
        prev_dep = float(self.r[0])
        for node in rt:
            if node == 0:
                continue
            arr = prev_dep + self.T[prev][node]
            dep = float(self.r[node]) if arr < self.r[node] else arr
            if dep > self.l[node] + tol:
                return (False, dist, tuple(deps), tuple(loads))
            load += self.q[node]
            if load > self.Q:
                return (False, dist, tuple(deps), tuple(loads))
            deps.append(dep)
            loads.append(load)
            dist += self.D[prev][node]
            prev = node
            prev_dep = dep
        arr = prev_dep + self.T[prev][0]
        if arr > self.l[0] + tol:
            return (False, dist, tuple(deps), tuple(loads))
        dist += self.D[prev][0]
        return (True, dist, tuple(deps), tuple(loads))

    def route_load(self, route):
        if not route:
            return 0
        ev = self.eval_route(tuple(route))
        return ev[3][-1] if ev[3] else 0

    @staticmethod
    def _clean_num(x):
        xf = float(x)
        xr = round(xf)
        if abs(xf - xr) < 1e-9:
            return int(xr)
        return float(round(xf, 6))

    def solution_data(self, routes):
        seen = set()
        total = 0.0
        out_routes = []
        dep = {}
        load = {}
        for route in routes:
            rt = tuple(int(v) for v in route if int(v) != 0)
            ev = self.eval_route(rt)
            if not ev[0]:
                return None
            total += ev[1]
            out_routes.append([0] + [self.node_id[i] for i in rt] + [0])
            for node, dpt, ld in zip(rt, ev[2], ev[3]):
                if node in seen:
                    return None
                seen.add(node)
                dep[str(self.node_id[node])] = self._clean_num(dpt)
                load[str(self.node_id[node])] = int(ld)
        if len(seen) != self.n:
            return None
        return total, out_routes, dep, load

    def make_solution_dict(self, data):
        total, out_routes, dep, load = data
        if self.dist_one_decimal:
            total_out = float(round(total + 1e-9, 1))
        else:
            total_out = float(round(total, 6))
        return {
            "objective_value": float(len(out_routes)),
            "num_vehicles": int(len(out_routes)),
            "routes": out_routes,
            "total_distance": total_out,
            "departure_times": dep,
            "loads": load,
        }

    def add_routes_to_pool(self, routes):
        for route in routes:
            rt = tuple(int(v) for v in route if int(v) != 0)
            if not rt or rt in self.route_pool or len(self.route_pool) >= self.pool_cap:
                continue
            ev = self.eval_route(rt)
            if ev[0]:
                self.route_pool[rt] = ev

    def consider(self, routes):
        clean = [tuple(int(v) for v in route if int(v) != 0) for route in routes]
        clean = [rt for rt in clean if rt]
        data = self.solution_data(clean)
        if data is None:
            return False
        key = (len(data[1]), data[0])
        self.add_routes_to_pool(clean)
        primary_improved = self.best_key is None or key[0] < self.best_key[0]
        better = (
            self.best_key is None
            or key[0] < self.best_key[0]
            or (key[0] == self.best_key[0] and key[1] < self.best_key[1] - 1e-7)
        )
        if better:
            sol_dict = self.make_solution_dict(data)
            self.best_key = key
            self.best_routes = [list(rt) for rt in clean]
            self.best_dict = sol_dict
            if primary_improved and self.logger:
                self.logger.log_solution(float(key[0]), sol_dict)
            return True
        return False

    def insertion_positions(self, route, c, full=False):
        m = len(route)
        if (full and m <= 120) or ((not full) and m <= 35):
            return range(m + 1)
        pos = {0, m}
        for v in self.neighbors.get(c, ()):  # nearby customers only
            if v in route:
                idx = route.index(v)
                pos.add(idx)
                pos.add(idx + 1)
        return sorted(pos)

    def insertion_options(self, routes, c, full=False):
        opts = []
        for ri, route in enumerate(routes):
            route = tuple(route)
            ev0 = self.eval_route(route)
            if not ev0[0]:
                continue
            load0 = ev0[3][-1] if route else 0
            if load0 + self.q[c] > self.Q:
                continue
            for pos in self.insertion_positions(route, c, full=full):
                nr = route[:pos] + (c,) + route[pos:]
                ev = self.eval_route(nr)
                if ev[0]:
                    opts.append((ev[1] - ev0[1], ri, pos, ev))
        opts.sort(key=lambda o: (o[0], -(o[3][3][-1] if o[3][3] else 0)))
        return opts

    def construct(self, order):
        routes = []
        route_evs = []
        for c in order:
            if self.time_left() <= 0:
                break
            opts = self.insertion_options(routes, c, full=False)
            if opts:
                best = min(
                    opts,
                    key=lambda o: (
                        o[0] - 1e-4 * (o[3][3][-1] if o[3][3] else 0),
                        o[0],
                    ),
                )
                _, ri, pos, ev = best
                routes[ri] = routes[ri][:pos] + (c,) + routes[ri][pos:]
                route_evs[ri] = ev
            else:
                ev = self.eval_route((c,))
                if not ev[0]:
                    raise ValueError("infeasible customer")
                routes.append((c,))
                route_evs.append(ev)
        return [list(rt) for rt in routes if rt]

    def make_orders(self):
        orders = []
        orders.append(sorted(self.ids, key=lambda i: (self.l[i], self.r[i], -self.q[i])))
        orders.append(sorted(self.ids, key=lambda i: (self.r[i], self.l[i], -self.q[i])))
        orders.append(sorted(self.ids, key=lambda i: (-self.q[i], self.l[i], self.r[i])))
        orders.append(sorted(self.ids, key=lambda i: (self.l[i] - self.r[i], self.l[i], -self.q[i])))
        orders.append(sorted(self.ids, key=lambda i: (self.D[0][i], self.l[i])))
        orders.append(sorted(self.ids, key=lambda i: (-self.D[0][i], self.l[i])))
        orders.append(sorted(self.ids, key=lambda i: (math.atan2(self.y[i] - self.y[0], self.x[i] - self.x[0]), self.l[i])))
        orders.append(sorted(self.ids, key=lambda i: (self.l[i], self.D[0][i], -self.q[i])))
        if self.n > 300:
            orders = orders[:4]
        return orders

    def random_order(self):
        ids = self.ids[:]
        choice = self.rng.randrange(4)
        if choice == 0:
            self.rng.shuffle(ids)
        elif choice == 1:
            ids.sort(key=lambda i: (self.l[i] + self.rng.uniform(-0.15, 0.15) * (self.l[i] - self.r[i] + 1), -self.q[i]))
        elif choice == 2:
            ids.sort(key=lambda i: (-self.q[i] + self.rng.uniform(0.0, 0.1 * max(1, self.Q)), self.l[i]))
        else:
            remaining = set(ids)
            cur = 0
            order = []
            while remaining:
                if len(remaining) <= 20:
                    sample = list(remaining)
                else:
                    sample = self.rng.sample(list(remaining), 20)
                nxt = min(sample, key=lambda j: self.T[cur][j] + 0.05 * self.l[j] + self.rng.uniform(0.0, 3.0))
                order.append(nxt)
                remaining.remove(nxt)
                cur = nxt
            ids = order
        return ids

    def optimize_route(self, route, deadline):
        rt = tuple(route)
        ev = self.eval_route(rt)
        if not ev[0] or len(rt) <= 1:
            return rt
        tol = 1e-7
        improved = True
        rounds = 0
        while improved and rounds < 60 and time.monotonic() < deadline:
            improved = False
            rounds += 1
            m = len(rt)
            base = ev[1]
            max_seg = 3 if m <= 60 else 1
            for L in range(1, max_seg + 1):
                for i in range(0, m - L + 1):
                    seg = rt[i:i + L]
                    rest = rt[:i] + rt[i + L:]
                    positions = range(len(rest) + 1) if len(rest) <= 70 else self.insertion_positions(rest, seg[0], full=False)
                    for pos in positions:
                        nr = rest[:pos] + seg + rest[pos:]
                        if nr == rt:
                            continue
                        ev2 = self.eval_route(nr)
                        if ev2[0] and ev2[1] < base - tol:
                            rt, ev = nr, ev2
                            improved = True
                            break
                        if time.monotonic() > deadline:
                            return rt
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue
            m = len(rt)
            base = ev[1]
            if m <= 80:
                for i in range(m - 1):
                    for j in range(i + 1, m):
                        nr = rt[:i] + rt[i:j + 1][::-1] + rt[j + 1:]
                        ev2 = self.eval_route(nr)
                        if ev2[0] and ev2[1] < base - tol:
                            rt, ev = nr, ev2
                            improved = True
                            break
                        if time.monotonic() > deadline:
                            return rt
                    if improved:
                        break
        return rt

    def find_relocate_move(self, routes, deadline):
        if len(routes) <= 1:
            return None
        route_of = {}
        for ri, route in enumerate(routes):
            for c in route:
                route_of[c] = ri
        least_loaded = sorted(range(len(routes)), key=lambda i: self.route_load(routes[i]))[:5]
        ridx = list(range(len(routes)))
        self.rng.shuffle(ridx)
        tol = 1e-7
        for A in ridx:
            if time.monotonic() > deadline:
                return None
            a = tuple(routes[A])
            if not a:
                continue
            evA = self.eval_route(a)
            custs = list(a)
            self.rng.shuffle(custs)
            for u in custs:
                targets = set(least_loaded)
                for v in self.neighbors.get(u, ()):
                    B = route_of.get(v)
                    if B is not None and B != A:
                        targets.add(B)
                targets.discard(A)
                ai = a.index(u)
                a_rem = a[:ai] + a[ai + 1:]
                evA_rem = self.eval_route(a_rem) if a_rem else (True, 0.0, (), ())
                if a_rem and not evA_rem[0]:
                    continue
                for B in targets:
                    if B >= len(routes) or B == A:
                        continue
                    b = tuple(routes[B])
                    evB = self.eval_route(b)
                    if not evB[0]:
                        continue
                    if (evB[3][-1] if b else 0) + self.q[u] > self.Q:
                        continue
                    old = evA[1] + evB[1]
                    for pos in self.insertion_positions(b, u, full=(len(b) <= 100)):
                        nb = b[:pos] + (u,) + b[pos:]
                        evNB = self.eval_route(nb)
                        if not evNB[0]:
                            continue
                        new = (evA_rem[1] if a_rem else 0.0) + evNB[1]
                        if (not a_rem) or new < old - tol:
                            out = []
                            for idx, route in enumerate(routes):
                                if idx == A:
                                    if a_rem:
                                        out.append(a_rem)
                                elif idx == B:
                                    out.append(nb)
                                else:
                                    out.append(tuple(route))
                            return out
        return None

    def find_swap_move(self, routes, deadline):
        if len(routes) <= 1:
            return None
        route_of = {}
        for ri, route in enumerate(routes):
            for c in route:
                route_of[c] = ri
        ridx = list(range(len(routes)))
        self.rng.shuffle(ridx)
        tol = 1e-7
        for A in ridx:
            if time.monotonic() > deadline:
                return None
            a = tuple(routes[A])
            evA = self.eval_route(a)
            loadA = evA[3][-1] if a else 0
            custs = list(a)
            self.rng.shuffle(custs)
            for u in custs:
                for v in self.neighbors.get(u, ()):
                    B = route_of.get(v)
                    if B is None or B <= A or B >= len(routes):
                        continue
                    b = tuple(routes[B])
                    evB = self.eval_route(b)
                    loadB = evB[3][-1] if b else 0
                    if loadA - self.q[u] + self.q[v] > self.Q or loadB - self.q[v] + self.q[u] > self.Q:
                        continue
                    ai = a.index(u)
                    bi = b.index(v)
                    na = a[:ai] + (v,) + a[ai + 1:]
                    nb = b[:bi] + (u,) + b[bi + 1:]
                    evNA = self.eval_route(na)
                    evNB = self.eval_route(nb)
                    if evNA[0] and evNB[0] and evNA[1] + evNB[1] < evA[1] + evB[1] - tol:
                        out = []
                        for idx, route in enumerate(routes):
                            if idx == A:
                                out.append(na)
                            elif idx == B:
                                out.append(nb)
                            else:
                                out.append(tuple(route))
                        return out
        return None

    def local_search(self, routes, deadline):
        routes = [tuple(r) for r in routes if r]
        for i in range(len(routes)):
            if time.monotonic() > deadline:
                break
            routes[i] = self.optimize_route(routes[i], deadline)
        it = 0
        while it < 120 and time.monotonic() < deadline:
            it += 1
            nr = self.find_relocate_move(routes, deadline)
            if nr is None:
                nr = self.find_swap_move(routes, deadline)
            if nr is None:
                break
            routes = [tuple(r) for r in nr if r]
            for i in range(len(routes)):
                if time.monotonic() > deadline:
                    break
                routes[i] = self.optimize_route(routes[i], deadline)
        return [list(r) for r in routes if r]

    def try_empty_route(self, routes, ri, deadline, node_limit=2500):
        routes = [tuple(r) for r in routes if r]
        if ri >= len(routes):
            return None
        target = tuple(routes[ri])
        others = tuple(routes[k] for k in range(len(routes)) if k != ri)
        if not target:
            return [list(r) for r in others]
        state = [0]

        def rec(current, unplaced):
            state[0] += 1
            if state[0] > node_limit or time.monotonic() > deadline:
                return None
            if not unplaced:
                return current
            all_opts = []
            for c in unplaced:
                opts = self.insertion_options(current, c, full=True)
                if not opts:
                    return None
                opts = sorted(opts, key=lambda o: (o[0] + self.rng.uniform(0.0, 1e-5), -(o[3][3][-1] if o[3][3] else 0)))
                all_opts.append((c, opts))

            def ckey(item):
                c, opts = item
                regret = opts[1][0] - opts[0][0] if len(opts) > 1 else float("inf")
                return (len(opts), -regret, self.l[c] - self.r[c], -self.q[c])

            all_opts.sort(key=ckey)
            for c, opts in all_opts[:2]:
                for _, rj, pos, _ in opts[:6]:
                    new_current = list(current)
                    rt = new_current[rj]
                    new_current[rj] = rt[:pos] + (c,) + rt[pos:]
                    res = rec(tuple(new_current), tuple(x for x in unplaced if x != c))
                    if res is not None:
                        return res
            return None

        res = rec(others, target)
        if res is None:
            return None
        return [list(r) for r in res if r]

    def eliminate_all(self, routes, deadline, consider_callback=None, max_route_tries=None):
        routes = [tuple(r) for r in routes if r]
        while len(routes) > 1 and time.monotonic() < deadline:
            idxs = list(range(len(routes)))
            self.rng.shuffle(idxs)
            idxs.sort(key=lambda i: (len(routes[i]), self.route_load(routes[i]), self.rng.random()))
            success = False
            tried = 0
            for ri in idxs:
                if max_route_tries is not None and tried >= max_route_tries:
                    break
                tried += 1
                if time.monotonic() > deadline:
                    break
                new_routes = self.try_empty_route(routes, ri, deadline)
                if new_routes is not None:
                    new_routes = self.local_search(new_routes, min(deadline, time.monotonic() + 0.35))
                    routes = [tuple(r) for r in new_routes if r]
                    if consider_callback:
                        consider_callback([list(r) for r in routes])
                    success = True
                    break
            if not success:
                break
        return [list(r) for r in routes if r]

    def ruin_recreate(self, routes, deadline):
        routes = [tuple(r) for r in routes if r]
        allcust = [c for r in routes for c in r]
        if not allcust:
            return None
        if self.rng.random() < 0.55:
            scored = []
            for r in routes:
                ev = self.eval_route(r)
                for idx, c in enumerate(r):
                    rem = r[:idx] + r[idx + 1:]
                    evr = self.eval_route(rem) if rem else (True, 0.0, (), ())
                    saving = ev[1] - evr[1]
                    scored.append((saving + self.rng.uniform(0.0, 1e-3), c))
            scored.sort(reverse=True, key=lambda x: x[0])
            m = max(1, int(len(allcust) * self.rng.uniform(0.08, 0.22)))
            removed = set(c for _, c in scored[:m])
        else:
            m = max(1, int(len(allcust) * self.rng.uniform(0.08, 0.25)))
            removed = set(self.rng.sample(allcust, min(m, len(allcust))))

        kept = []
        for r in routes:
            nr = tuple(c for c in r if c not in removed)
            if nr:
                kept.append(nr)
        if not kept:
            return None
        unplaced = list(removed)
        while unplaced:
            if time.monotonic() > deadline:
                return None
            cand = []
            for c in unplaced:
                opts = self.insertion_options(kept, c, full=False)
                if not opts:
                    return None
                regret = opts[1][0] - opts[0][0] if len(opts) > 1 else 1e9
                hardness = self.l[c] - self.r[c]
                cand.append((regret - 0.001 * hardness + 0.0001 * self.q[c], c, opts[0]))
            cand.sort(reverse=True, key=lambda x: x[0])
            _, c, opt = cand[0]
            _, ri, pos, _ = opt
            kept[ri] = kept[ri][:pos] + (c,) + kept[ri][pos:]
            unplaced.remove(c)
        return [list(r) for r in kept if r]

    def solve_pool_master(self, time_budget):
        if gp is None or GRB is None or time_budget < 0.35 or not self.route_pool:
            return None
        items = list(self.route_pool.items())
        if not any(len(rt) > 1 for rt, _ in items):
            return None

        max_cols = min(8000, max(2500, 40 * self.n + 1000))
        if len(items) > max_cols:
            best_set = set(tuple(r) for r in self.best_routes) if self.best_routes else set()
            singles = [it for it in items if len(it[0]) == 1]
            rest = [it for it in items if len(it[0]) > 1]

            def score(item):
                rt, ev = item
                load = ev[3][-1] if ev[3] else sum(self.q[i] for i in rt)
                return (-(load / (ev[1] + 1e-6)), -len(rt), ev[1])

            rest.sort(key=score)
            items = singles + rest[: max(0, max_cols - len(singles))]
            have = set(rt for rt, _ in items)
            for rt in best_set:
                if rt in self.route_pool and rt not in have:
                    items.append((rt, self.route_pool[rt]))
                    have.add(rt)

        cols = [tuple(rt) for rt, _ in items]
        evs = [ev for _, ev in items]
        by_customer = [[] for _ in range(self.n + 1)]
        for j, rt in enumerate(cols):
            for c in rt:
                if 1 <= c <= self.n:
                    by_customer[c].append(j)
        for c in self.ids:
            if not by_customer[c]:
                return None

        try:
            model = gp.Model("min_fleet_set_partition")
            model.setParam("OutputFlag", 0)
            model.setParam("LogToConsole", 0)
            model.setParam("Seed", 0)
            model.setParam("MIPGap", 1e-4)
            model.setParam("NumericFocus", 0)
            model.setParam("Threads", 1)
            model.setParam("TimeLimit", max(0.05, float(time_budget)))

            x = model.addVars(len(cols), vtype=GRB.BINARY, obj=0.0, name="x")
            for c in self.ids:
                model.addConstr(gp.quicksum(x[j] for j in by_customer[c]) == 1, name=f"cover_{c}")
            model.setObjective(gp.quicksum((self.bigM + evs[j][1]) * x[j] for j in range(len(cols))), GRB.MINIMIZE)
            model.optimize()
            if model.SolCount > 0:
                selected = [cols[j] for j in range(len(cols)) if x[j].X > 0.5]
                if selected:
                    return [list(rt) for rt in selected]
        except Exception:
            return None
        return None

    def solve(self):
        if self.n == 0:
            self.best_dict = {
                "objective_value": 0.0,
                "num_vehicles": 0,
                "routes": [],
                "total_distance": 0.0,
                "departure_times": {},
                "loads": {},
            }
            return self.best_dict

        singles = [(i,) for i in self.ids]
        for rt in singles:
            ev = self.eval_route(rt)
            if ev[0]:
                self.route_pool[rt] = ev
        self.consider([list(rt) for rt in singles])

        if self.time_limit <= 0:
            return self.best_dict
        if self.best_key and self.best_key[0] <= self.lb_routes:
            return self.best_dict

        initial_budget = max(0.5, min(12.0, 0.25 * self.time_limit))
        initial_deadline = min(self.deadline, time.monotonic() + initial_budget)

        for order in self.make_orders():
            if time.monotonic() > initial_deadline or (self.best_key and self.best_key[0] <= self.lb_routes):
                break
            try:
                routes = self.construct(order)
            except Exception:
                routes = [list(rt) for rt in singles]
            rem = max(0.0, initial_deadline - time.monotonic())
            routes = self.local_search(routes, min(self.deadline, time.monotonic() + min(1.2, 0.45 * rem)))
            routes = self.eliminate_all(
                routes,
                min(self.deadline, time.monotonic() + min(1.8, 0.55 * rem)),
                consider_callback=self.consider,
            )
            self.consider(routes)

        while time.monotonic() < initial_deadline and (not self.best_key or self.best_key[0] > self.lb_routes):
            try:
                routes = self.construct(self.random_order())
            except Exception:
                break
            rem = max(0.0, initial_deadline - time.monotonic())
            if rem <= 0.05:
                break
            routes = self.local_search(routes, min(self.deadline, time.monotonic() + min(0.8, 0.35 * rem)))
            routes = self.eliminate_all(
                routes,
                min(self.deadline, time.monotonic() + min(1.2, 0.55 * rem)),
                consider_callback=self.consider,
            )
            self.consider(routes)

        if self.time_left() > 1.0 and (not self.best_key or self.best_key[0] > self.lb_routes):
            routes = self.solve_pool_master(min(self.time_left() - 0.4, max(1.0, 0.15 * self.time_limit)))
            if routes:
                self.consider(routes)

        it = 0
        while self.time_left() > 0.45 and (not self.best_key or self.best_key[0] > self.lb_routes):
            it += 1
            remaining = self.time_left()
            if self.best_routes:
                routes = self.eliminate_all(
                    [list(r) for r in self.best_routes],
                    time.monotonic() + min(3.0, 0.25 * remaining),
                    consider_callback=self.consider,
                )
                self.consider(routes)
                if self.best_key and self.best_key[0] <= self.lb_routes:
                    break

            remaining = self.time_left()
            if remaining <= 0.45:
                break
            if self.best_routes:
                rr = self.ruin_recreate([list(r) for r in self.best_routes], time.monotonic() + min(2.0, 0.20 * remaining))
                if rr is not None:
                    rr = self.local_search(rr, time.monotonic() + min(0.9, 0.18 * self.time_left()))
                    rr = self.eliminate_all(
                        rr,
                        time.monotonic() + min(1.4, 0.22 * self.time_left()),
                        consider_callback=self.consider,
                    )
                    self.consider(rr)
                    if self.best_key and self.best_key[0] <= self.lb_routes:
                        break

            remaining = self.time_left()
            if remaining <= 0.45:
                break
            if it % 2 == 0:
                try:
                    routes = self.construct(self.random_order())
                except Exception:
                    routes = None
                if routes is not None:
                    routes = self.local_search(routes, time.monotonic() + min(0.9, 0.18 * remaining))
                    routes = self.eliminate_all(
                        routes,
                        time.monotonic() + min(1.4, 0.22 * self.time_left()),
                        consider_callback=self.consider,
                    )
                    self.consider(routes)

            if self.time_left() > 1.0 and it % 3 == 0:
                routes = self.solve_pool_master(min(self.time_left() - 0.4, max(0.8, 0.10 * self.time_limit)))
                if routes:
                    self.consider(routes)
            if self.best_key and self.best_key[0] <= self.lb_routes:
                break

        if self.time_left() > 1.0 and self.best_key and self.best_key[0] > self.lb_routes:
            routes = self.solve_pool_master(self.time_left() - 0.4)
            if routes:
                self.consider(routes)

        return self.best_dict


def emergency_solution(solver):
    routes = [[0, solver.node_id[i], 0] for i in solver.ids]
    departure_times = {str(solver.node_id[i]): solver._clean_num(solver.r[i]) for i in solver.ids}
    loads = {str(solver.node_id[i]): int(solver.q[i]) for i in solver.ids}
    total = sum(solver.D[0][i] + solver.D[i][0] for i in solver.ids)
    if solver.dist_one_decimal:
        total = float(round(total + 1e-9, 1))
    else:
        total = float(round(total, 6))
    return {
        "objective_value": float(len(routes)),
        "num_vehicles": int(len(routes)),
        "routes": routes,
        "total_distance": total,
        "departure_times": departure_times,
        "loads": loads,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as f:
        instance = json.load(f)

    solver = MinFleetVRPTW(instance, args.time_limit, logger)
    solution = solver.solve()
    if solution is None:
        solution = emergency_solution(solver)

    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(solution, f, separators=(",", ":"))


if __name__ == "__main__":
    main()
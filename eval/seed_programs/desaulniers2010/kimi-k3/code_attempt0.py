import argparse
import json
import math
import random
import time
from collections import defaultdict

try:
    from solution_logger import SolutionLogger
except Exception:
    SolutionLogger = None

EPS = 1e-6


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--instance_path", required=True)
    p.add_argument("--solution_path", required=True)
    p.add_argument("--time_limit", type=int, required=True)
    p.add_argument("--log_path", default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Instance data
# ---------------------------------------------------------------------------
class Data:
    def __init__(self, inst):
        self.n = int(inst["num_customers"])
        self.Q = float(inst["vehicle_capacity"])
        n = self.n
        kmax = inst.get("num_vehicles_upper_bound")
        self.D = [list(map(float, row)) for row in inst["distance_matrix"]]
        dep = inst["depot"]
        self.a0 = float(dep["time_window"][0])
        self.b0 = float(dep["time_window"][1])
        self.demand = [0.0] * (n + 1)
        self.serv = [0.0] * (n + 1)
        self.twa = [0.0] * (n + 1)
        self.twb = [0.0] * (n + 1)
        for c in inst["customers"]:
            i = int(c["id"])
            self.demand[i] = float(c["demand"])
            self.serv[i] = float(c["service_time"])
            self.twa[i] = float(c["time_window"][0])
            self.twb[i] = float(c["time_window"][1])
        q_int = max(1, int(round(self.Q)))
        tot_int = sum(int(round(self.demand[i])) for i in range(1, n + 1))
        self.Kmin = (tot_int + q_int - 1) // q_int if tot_int > 0 else 0
        if kmax is None:
            kmax = sum((int(round(self.demand[i])) + q_int - 1) // q_int
                       for i in range(1, n + 1))
        self.Kmax = max(int(kmax), self.Kmin)


def make_copies(data, rng, mode):
    """Split each customer demand into ceil(d/Q) delivery atoms (copies)."""
    copies = []
    Q = max(1, int(round(data.Q)))
    for i in range(1, data.n + 1):
        d = int(round(data.demand[i]))
        if d <= 0:
            continue
        m = int(math.ceil(d / Q))
        if m == 1:
            copies.append((i, d))
        elif mode == "front":
            r = d
            while r > Q:
                copies.append((i, Q))
                r -= Q
            copies.append((i, r))
        elif mode == "random":
            rem = d
            for k in range(m):
                slots = m - k - 1
                lo = max(1, rem - slots * Q)
                hi = min(Q, rem - slots)
                if hi < lo:
                    hi = lo
                val = rng.randint(lo, hi)
                copies.append((i, val))
                rem -= val
        else:  # balanced
            base = d // m
            r = d % m
            for k in range(m):
                copies.append((i, base + 1 if k < r else base))
    return copies


# ---------------------------------------------------------------------------
# Solution helpers (visits format: list of routes, each a list of (cust, qty))
# ---------------------------------------------------------------------------
def route_cost_of(data, visit_route):
    D = data.D
    if not visit_route:
        return 0.0
    c = D[0][visit_route[0][0]]
    for k in range(1, len(visit_route)):
        c += D[visit_route[k - 1][0]][visit_route[k][0]]
    c += D[visit_route[-1][0]][0]
    return c


def validate(data, visits):
    tot = defaultdict(float)
    nroutes = 0
    for route in visits:
        if not route:
            continue
        nroutes += 1
        if sum(q for _, q in route) > data.Q + 1e-6:
            return False
        t = data.a0
        prev = 0
        for cust, q in route:
            arr = t + data.serv[prev] + data.D[prev][cust]
            st = arr if arr > data.twa[cust] else data.twa[cust]
            if st > data.twb[cust] + 1e-6:
                return False
            t = st
            prev = cust
        if t + data.serv[prev] + data.D[prev][0] > data.b0 + 1e-6:
            return False
        for cust, q in route:
            tot[cust] += q
    if nroutes > data.Kmax:
        return False
    for i in range(1, data.n + 1):
        if tot.get(i, 0.0) < data.demand[i] - 1e-6:
            return False
    return True


def _num(x):
    return int(round(x)) if abs(x - round(x)) < 1e-9 else float(x)


def build_solution(data, visits):
    D = data.D
    n = data.n
    routes_out = []
    total_del = defaultdict(float)
    obj = 0.0
    for vi, route in enumerate(visits, start=1):
        if not route:
            continue
        seq = [0]
        deliv = {}
        prev = 0
        for cust, qty in route:
            seq.append(cust)
            deliv[str(cust)] = deliv.get(str(cust), 0.0) + qty
            total_del[cust] += qty
            obj += D[prev][cust]
            prev = cust
        obj += D[prev][0]
        seq.append(n + 1)
        routes_out.append({
            "vehicle": vi,
            "route": seq,
            "deliveries": {k: _num(v) for k, v in deliv.items()},
        })
    deliveries = {}
    for c in range(1, n + 1):
        deliveries[str(c)] = {
            "demand": _num(data.demand[c]),
            "total_delivered": _num(total_del.get(c, 0.0)),
        }
    return {
        "objective_value": float(obj),
        "num_vehicles_used": len(routes_out),
        "routes": routes_out,
        "deliveries": deliveries,
    }


def trivial_solution(data):
    copies = make_copies(data, random.Random(0), "balanced")
    visits = [[(c, q)] for (c, q) in copies]
    cost = sum(route_cost_of(data, r) for r in visits)
    return (cost, visits)


# ---------------------------------------------------------------------------
# Heuristic solver (copies = split delivery atoms; VRPTW local search)
# ---------------------------------------------------------------------------
class Solver:
    def __init__(self, data, copies):
        self.data = data
        self.copies = copies
        self.cc = [c for c, _ in copies]
        self.cq = [q for _, q in copies]
        self.n2 = len(copies)
        self.deadline = float("inf")

    def route_cost(self, route):
        D = self.data.D
        cc = self.cc
        if not route:
            return 0.0
        c = D[0][cc[route[0]]]
        for k in range(1, len(route)):
            c += D[cc[route[k - 1]]][cc[route[k]]]
        c += D[cc[route[-1]]][0]
        return c

    def total_cost(self, routes):
        return sum(self.route_cost(r) for r in routes)

    def route_demand(self, route):
        return sum(self.cq[c] for c in route)

    def time_feasible(self, route):
        data = self.data
        D = data.D
        cc = self.cc
        t = data.a0
        prev = 0
        for cp in route:
            i = cc[cp]
            arr = t + data.serv[prev] + D[prev][i]
            st = arr if arr > data.twa[i] else data.twa[i]
            if st > data.twb[i] + EPS:
                return False
            t = st
            prev = i
        return t + data.serv[prev] + D[prev][0] <= data.b0 + EPS

    def to_visits(self, routes):
        return [[(self.cc[c], self.cq[c]) for c in r] for r in routes if r]

    # ---- construction helpers ----
    def insertion_options(self, routes, dem_sum, c, want=1):
        data = self.data
        D = data.D
        cust = self.cc[c]
        qty = self.cq[c]
        cands = []
        if len(routes) < data.Kmax:
            cands.append((D[0][cust] + D[cust][0], -1, 0))
        for ri, r in enumerate(routes):
            if dem_sum[ri] + qty > data.Q + EPS:
                continue
            rset = set(self.cc[cp] for cp in r)
            if cust in rset:
                continue
            for p in range(len(r) + 1):
                u = 0 if p == 0 else self.cc[r[p - 1]]
                v = 0 if p == len(r) else self.cc[r[p]]
                cands.append((D[u][cust] + D[cust][v] - D[u][v], ri, p))
        cands.sort(key=lambda x: x[0])
        feas = []
        checked = 0
        for delta, ri, p in cands:
            if ri == -1:
                if self.time_feasible([c]):
                    feas.append((delta, ri, p))
            else:
                r = routes[ri]
                if self.time_feasible(r[:p] + [c] + r[p:]):
                    feas.append((delta, ri, p))
            checked += 1
            if len(feas) >= want or checked >= 25:
                break
        return feas

    def force_insert(self, routes, dem_sum, c):
        data = self.data
        D = data.D
        cust = self.cc[c]
        qty = self.cq[c]
        best = None
        for ri, r in enumerate(routes):
            if dem_sum[ri] + qty > data.Q + EPS:
                continue
            rset = set(self.cc[cp] for cp in r)
            if cust in rset:
                continue
            for p in range(len(r) + 1):
                u = 0 if p == 0 else self.cc[r[p - 1]]
                v = 0 if p == len(r) else self.cc[r[p]]
                delta = D[u][cust] + D[cust][v] - D[u][v]
                if best is None or delta < best[0]:
                    best = (delta, ri, p)
        if best is not None:
            _, ri, p = best
            routes[ri].insert(p, c)
            dem_sum[ri] += qty
        else:
            routes.append([c])
            dem_sum.append(qty)

    def _apply(self, routes, dem_sum, c, opt):
        _, ri, p = opt
        if ri == -1:
            routes.append([c])
            dem_sum.append(self.cq[c])
        else:
            routes[ri].insert(p, c)
            dem_sum[ri] += self.cq[c]

    def construct(self, rng, deadline):
        order = list(range(self.n2))
        order.sort(key=lambda c: (self.data.twb[self.cc[c]], -self.cq[c], rng.random()))
        routes = []
        dem_sum = []
        for c in order:
            opts = self.insertion_options(routes, dem_sum, c, want=1)
            if opts:
                self._apply(routes, dem_sum, c, opts[0])
            else:
                self.force_insert(routes, dem_sum, c)
        return routes

    def construct_regret(self, rng, deadline):
        unassigned = list(range(self.n2))
        routes = []
        dem_sum = []
        while unassigned:
            if time.time() > deadline:
                for c in unassigned:
                    opts = self.insertion_options(routes, dem_sum, c, want=1)
                    if opts:
                        self._apply(routes, dem_sum, c, opts[0])
                    else:
                        self.force_insert(routes, dem_sum, c)
                break
            best_choice = None
            for c in unassigned:
                opts = self.insertion_options(routes, dem_sum, c, want=2)
                if not opts:
                    key = (float("inf"), 0.0)
                    b = None
                else:
                    b = opts[0]
                    regret = (opts[1][0] - opts[0][0]) if len(opts) > 1 else float("inf")
                    key = (regret, -opts[0][0])
                if best_choice is None or key > best_choice[0]:
                    best_choice = (key, c, b)
            _, c, b = best_choice
            if b is None:
                self.force_insert(routes, dem_sum, c)
            else:
                self._apply(routes, dem_sum, c, b)
            unassigned.remove(c)
        return routes

    # ---- local search ----
    def two_opt(self, route):
        D = self.data.D
        cc = self.cc
        L = len(route)
        if L < 2:
            return False
        best_delta = -EPS
        bi = None
        for i in range(L - 1):
            ci = cc[route[i]]
            u = 0 if i == 0 else cc[route[i - 1]]
            for j in range(i + 1, L):
                cj = cc[route[j]]
                v = 0 if j == L - 1 else cc[route[j + 1]]
                delta = D[u][cj] + D[ci][v] - D[u][ci] - D[cj][v]
                if delta < best_delta:
                    nr = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                    if self.time_feasible(nr):
                        best_delta = delta
                        bi = (i, j)
        if bi is not None:
            i, j = bi
            route[i:j + 1] = route[i:j + 1][::-1]
            return True
        return False

    def relocate_intra(self, route):
        D = self.data.D
        cc = self.cc
        L = len(route)
        if L < 2:
            return False
        best_delta = -EPS
        best = None
        for ia in range(L):
            c = route[ia]
            cust = cc[c]
            pu = 0 if ia == 0 else cc[route[ia - 1]]
            nu = 0 if ia == L - 1 else cc[route[ia + 1]]
            rem = D[pu][nu] - D[pu][cust] - D[cust][nu]
            base = route[:ia] + route[ia + 1:]
            for p in range(L):
                if p == ia:
                    continue
                u = 0 if p == 0 else cc[base[p - 1]]
                v = 0 if p == L - 1 else cc[base[p]]
                delta = rem + D[u][cust] + D[cust][v] - D[u][v]
                if delta < best_delta:
                    nr = base[:p] + [c] + base[p:]
                    if self.time_feasible(nr):
                        best_delta = delta
                        best = (ia, p)
        if best is not None:
            ia, p = best
            c = route.pop(ia)
            route.insert(p, c)
            return True
        return False

    def relocate(self, routes):
        data = self.data
        D = data.D
        cc = self.cc
        cq = self.cq
        Q = data.Q
        dem_sums = [sum(cq[c] for c in r) for r in routes]
        cust_sets = [set(cc[c] for c in r) for r in routes]
        best = None
        best_delta = -EPS
        for a in range(len(routes)):
            if time.time() > self.deadline:
                return False
            ra = routes[a]
            for ia in range(len(ra)):
                c = ra[ia]
                cust = cc[c]
                qty = cq[c]
                pu = 0 if ia == 0 else cc[ra[ia - 1]]
                nu = 0 if ia == len(ra) - 1 else cc[ra[ia + 1]]
                rem_delta = D[pu][nu] - D[pu][cust] - D[cust][nu]
                for b in range(len(routes)):
                    if a == b:
                        continue
                    if cust in cust_sets[b]:
                        continue
                    if dem_sums[b] + qty > Q + EPS:
                        continue
                    rb = routes[b]
                    for p in range(len(rb) + 1):
                        u = 0 if p == 0 else cc[rb[p - 1]]
                        v = 0 if p == len(rb) else cc[rb[p]]
                        delta = rem_delta + D[u][cust] + D[cust][v] - D[u][v]
                        if delta < best_delta:
                            na = ra[:ia] + ra[ia + 1:]
                            nb = rb[:p] + [c] + rb[p:]
                            if self.time_feasible(na) and self.time_feasible(nb):
                                best_delta = delta
                                best = (a, ia, b, p)
        if best is not None:
            a, ia, b, p = best
            c = routes[a].pop(ia)
            routes[b].insert(p, c)
            if not routes[a]:
                routes.pop(a)
            return True
        return False

    def swap(self, routes):
        data = self.data
        D = data.D
        cc = self.cc
        cq = self.cq
        Q = data.Q
        dem_sums = [sum(cq[c] for c in r) for r in routes]
        cust_sets = [set(cc[c] for c in r) for r in routes]
        best = None
        best_delta = -EPS
        for a in range(len(routes)):
            if time.time() > self.deadline:
                return False
            ra = routes[a]
            for b in range(a + 1, len(routes)):
                rb = routes[b]
                for ia in range(len(ra)):
                    c1 = ra[ia]
                    cu1 = cc[c1]
                    q1 = cq[c1]
                    pa = 0 if ia == 0 else cc[ra[ia - 1]]
                    na_ = 0 if ia == len(ra) - 1 else cc[ra[ia + 1]]
                    for ib in range(len(rb)):
                        c2 = rb[ib]
                        cu2 = cc[c2]
                        q2 = cq[c2]
                        if cu2 == cu1:
                            continue
                        if cu2 in cust_sets[a] or cu1 in cust_sets[b]:
                            continue
                        if dem_sums[a] - q1 + q2 > Q + EPS:
                            continue
                        if dem_sums[b] - q2 + q1 > Q + EPS:
                            continue
                        pb = 0 if ib == 0 else cc[rb[ib - 1]]
                        nb_ = 0 if ib == len(rb) - 1 else cc[rb[ib + 1]]
                        delta = (D[pa][cu2] + D[cu2][na_] - D[pa][cu1] - D[cu1][na_]
                                 + D[pb][cu1] + D[cu1][nb_] - D[pb][cu2] - D[cu2][nb_])
                        if delta < best_delta:
                            nra = ra[:ia] + [c2] + ra[ia + 1:]
                            nrb = rb[:ib] + [c1] + rb[ib + 1:]
                            if self.time_feasible(nra) and self.time_feasible(nrb):
                                best_delta = delta
                                best = (a, ia, b, ib)
        if best is not None:
            a, ia, b, ib = best
            routes[a][ia], routes[b][ib] = routes[b][ib], routes[a][ia]
            return True
        return False

    def local_search(self, routes, deadline):
        self.deadline = deadline
        while time.time() < deadline:
            improved = False
            for r in list(routes):
                if self.two_opt(r):
                    improved = True
                if self.relocate_intra(r):
                    improved = True
            if self.relocate(routes):
                improved = True
            if time.time() > deadline:
                break
            if self.swap(routes):
                improved = True
            if not improved:
                break
        routes[:] = [r for r in routes if r]

    # ---- perturbation ----
    def perturb(self, routes, rng):
        routes = [r[:] for r in routes if r]
        flat = [(ri, i) for ri, r in enumerate(routes) for i in range(len(r))]
        if not flat:
            return routes
        k = rng.randint(2, max(3, min(len(flat), 6)))
        k = min(k, len(flat))
        chosen = set(rng.sample(flat, k))
        removed = []
        new_routes = []
        for ri, r in enumerate(routes):
            nr = []
            for i, c in enumerate(r):
                if (ri, i) in chosen:
                    removed.append(c)
                else:
                    nr.append(c)
            if nr:
                new_routes.append(nr)
        dem_sum = [self.route_demand(r) for r in new_routes]
        rng.shuffle(removed)
        for c in removed:
            opts = self.insertion_options(new_routes, dem_sum, c, want=1)
            if opts:
                self._apply(new_routes, dem_sum, c, opts[0])
            else:
                self.force_insert(new_routes, dem_sum, c)
        return new_routes


def heuristic(data, deadline, logger, start_best):
    rng = random.Random(12345)
    best = start_best
    last_logged = best[0] if best else None
    modes = ["balanced", "front", "random"]
    epoch = 0
    first = True
    while first or time.time() < deadline:
        first = False
        mode = "balanced" if epoch == 0 else modes[rng.randrange(3)]
        copies = make_copies(data, rng, mode)
        if not copies:
            break
        solver = Solver(data, copies)
        if solver.n2 <= 80:
            routes = solver.construct_regret(rng, deadline)
        else:
            routes = solver.construct(rng, deadline)
        solver.local_search(routes, deadline)
        cost = solver.total_cost(routes)
        v = solver.to_visits(routes)
        if (best is None or cost < best[0] - EPS) and validate(data, v):
            best = (cost, v)
            if logger and (last_logged is None or cost < last_logged - EPS):
                logger.log_solution(cost, build_solution(data, v))
                last_logged = cost
        cur = routes
        cur_cost = cost
        no_improve = 0
        while time.time() < deadline and no_improve < 60:
            pr = solver.perturb(cur, rng)
            solver.local_search(pr, deadline)
            c = solver.total_cost(pr)
            if c < cur_cost * 1.03 + 1e-9:
                cur = pr
                cur_cost = c
            else:
                no_improve += 1
            if best is None or c < best[0] - EPS:
                vv = solver.to_visits(pr)
                if validate(data, vv):
                    best = (c, vv)
                    if logger and (last_logged is None or c < last_logged - EPS):
                        logger.log_solution(c, build_solution(data, vv))
                        last_logged = c
                    no_improve = 0
        epoch += 1
    return best


# ---------------------------------------------------------------------------
# MIP (Gurobi) for small instances
# ---------------------------------------------------------------------------
def extract_mip_solution(xv, qv, K, arcs, N, RET, data):
    na = len(arcs)
    nn = len(N)
    visits = []
    for k in range(K):
        adj = {}
        base = k * na
        for ai, (i, j) in enumerate(arcs):
            if xv[base + ai] > 0.5:
                adj[i] = j
        if 0 not in adj:
            continue
        route = []
        cur = 0
        guard = 0
        while cur != RET and guard <= nn + 1:
            nxt = adj.get(cur)
            if nxt is None:
                break
            if nxt != RET:
                qty = qv[k * nn + (nxt - 1)]
                if qty > 1e-6:
                    route.append((nxt, qty))
                cur = nxt
            else:
                cur = RET
            guard += 1
        if cur != RET or not route:
            continue
        visits.append(route)
    if not visits:
        return None
    if not validate(data, visits):
        return None
    cost = sum(route_cost_of(data, r) for r in visits)
    return (cost, visits)


def solve_with_mip(data, warm_visits, time_budget, logger, current_best):
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception:
        return None
    n = data.n
    if n < 1:
        return None
    Q = data.Q
    D = data.D
    RET = n + 1
    N = list(range(1, n + 1))
    Kmin = max(1, data.Kmin)
    K = min(data.Kmax, max(len(warm_visits), Kmin) + 2)
    K = max(K, Kmin)
    # Arc set: arc exists iff service started at the origin's earliest time
    # still allows arrival before the destination's latest time.
    arcs = []
    for j in N:
        if data.a0 + D[0][j] <= data.twb[j] + EPS:
            arcs.append((0, j))
    arcs.append((0, RET))
    for i in N:
        ai = data.twa[i]
        si = data.serv[i]
        Di = D[i]
        for j in N:
            if i != j and ai + si + Di[j] <= data.twb[j] + EPS:
                arcs.append((i, j))
        if ai + si + Di[0] <= data.b0 + EPS:
            arcs.append((i, RET))
    if len(arcs) * K > 6000:
        return None
    cost = {}
    for (i, j) in arcs:
        if j == RET:
            cost[(i, j)] = 0.0 if i == 0 else D[i][0]
        else:
            cost[(i, j)] = D[i][j]
    try:
        m = gp.Model("SDVRPTW")
        m.Params.Seed = 0
        m.Params.MIPGap = 1e-4
        m.Params.NumericFocus = 0
        m.Params.Threads = 1
        m.Params.OutputFlag = 0
        m.Params.TimeLimit = max(1, int(time_budget))
        Ks = list(range(K))
        xkeys = [(k, i, j) for k in Ks for (i, j) in arcs]
        x = m.addVars(xkeys, vtype=GRB.BINARY, name="x")
        qkeys = [(k, j) for k in Ks for j in N]
        q = m.addVars(qkeys, lb=0.0, name="q")
        tkeys = [(k, nd) for k in Ks for nd in ([0] + N + [RET])]
        t = m.addVars(tkeys, lb=0.0, name="t")
        used = m.addVars(Ks, vtype=GRB.BINARY, name="used")
        m.setObjective(
            gp.quicksum(cost[(i, j)] * x[k, i, j] for k in Ks for (i, j) in arcs),
            GRB.MINIMIZE)
        out0 = [a for a in arcs if a[0] == 0]
        inR = [a for a in arcs if a[1] == RET]
        U = {j: min(data.demand[j], Q) for j in N}
        for k in Ks:
            m.addConstr(gp.quicksum(x[k, a[0], a[1]] for a in out0) == 1)
            m.addConstr(gp.quicksum(x[k, a[0], a[1]] for a in inR) == 1)
            m.addConstr(used[k] == gp.quicksum(x[k, a[0], a[1]] for a in out0 if a[1] != RET))
            m.addConstr(gp.quicksum(q[k, j] for j in N) <= Q)
            t[k, 0].lb = data.a0
            t[k, 0].ub = data.b0
            t[k, RET].lb = data.a0
            t[k, RET].ub = data.b0
            for j in N:
                inj = [a for a in arcs if a[1] == j]
                outj = [a for a in arcs if a[0] == j]
                yj = gp.quicksum(x[k, a[0], a[1]] for a in inj)
                m.addConstr(yj == gp.quicksum(x[k, a[0], a[1]] for a in outj))
                m.addConstr(yj <= 1)
                m.addConstr(q[k, j] <= U[j] * yj)
                m.addConstr(t[k, j] >= data.twa[j] * yj)
                m.addConstr(t[k, j] <= data.twb[j] * yj)
            for (i, j) in arcs:
                si = data.serv[i] if i != 0 else 0.0
                tau = D[i][j] if j != RET else D[i][0]
                ui = data.twb[i] if i != 0 else data.b0
                lj = data.a0 if j == RET else 0.0
                M = ui + si + tau - lj
                if M < 0.0:
                    M = 0.0
                m.addConstr(t[k, j] >= t[k, i] + si + tau - M * (1 - x[k, i, j]))
        for j in N:
            m.addConstr(gp.quicksum(q[k, j] for k in Ks) >= data.demand[j])
        m.addConstr(gp.quicksum(used[k] for k in Ks) >= Kmin)
        for k in range(K - 1):
            m.addConstr(used[k] >= used[k + 1])
        # Warm start from the heuristic solution.
        for k in Ks:
            if k < len(warm_visits):
                route = [r for r in warm_visits[k] if r[1] > EPS]
                used[k].Start = 1.0
                t[k, 0].Start = data.a0
                seq = [c for c, _ in route]
                arcs_seq = set()
                prev = 0
                for c in seq:
                    arcs_seq.add((prev, c))
                    prev = c
                arcs_seq.add((prev, RET))
                for (i, j) in arcs:
                    x[k, i, j].Start = 1.0 if (i, j) in arcs_seq else 0.0
                tt = data.a0
                prev = 0
                for (cust, qty) in route:
                    arr = tt + data.serv[prev] + D[prev][cust]
                    st = arr if arr > data.twa[cust] else data.twa[cust]
                    t[k, cust].Start = st
                    q[k, cust].Start = qty
                    tt = st
                    prev = cust
                t[k, RET].Start = tt + data.serv[prev] + D[prev][0]
            else:
                used[k].Start = 0.0
                t[k, 0].Start = data.a0
                t[k, RET].Start = data.a0
                for (i, j) in arcs:
                    x[k, i, j].Start = 1.0 if (i, j) == (0, RET) else 0.0
        model_x = [x[k, i, j] for k in Ks for (i, j) in arcs]
        model_q = [q[k, j] for k in Ks for j in N]
        m._xvars = model_x
        m._qvars = model_q
        state = {"best": current_best}

        def cb(model, where):
            if where == GRB.Callback.MIPSOL:
                try:
                    xv = model.cbGetSolution(model._xvars)
                    qv = model.cbGetSolution(model._qvars)
                    res = extract_mip_solution(xv, qv, K, arcs, N, RET, data)
                    if res is not None:
                        cur = state["best"]
                        if cur is None or res[0] < cur[0] - EPS:
                            state["best"] = res
                            if logger:
                                logger.log_solution(res[0], build_solution(data, res[1]))
                except Exception:
                    pass

        m.optimize(cb)
        result = state["best"]
        if m.SolCount > 0:
            try:
                xv = [v.X for v in model_x]
                qv = [v.X for v in model_q]
                res = extract_mip_solution(xv, qv, K, arcs, N, RET, data)
                if res is not None and (result is None or res[0] < result[0] - EPS):
                    result = res
            except Exception:
                pass
        return result
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    with open(args.instance_path) as f:
        inst = json.load(f)
    logger = None
    if args.log_path and SolutionLogger is not None:
        try:
            logger = SolutionLogger(args.log_path, sense="minimize")
        except Exception:
            logger = None
    t0 = time.time()
    total = max(1, args.time_limit)
    data = Data(inst)

    if data.n == 0 or data.Kmin == 0:
        sol = {"objective_value": 0.0, "num_vehicles_used": 0,
               "routes": [], "deliveries": {}}
        if logger:
            logger.log_solution(0.0, sol)
        with open(args.solution_path, "w") as f:
            json.dump(sol, f)
        return

    pre_K = min(data.Kmax, data.Kmin + 4)
    use_mip_est = data.n <= 12 and pre_K <= 8 and total >= 20
    heur_budget = min(total * 0.25, 25.0) if use_mip_est else total - 0.5

    best = None
    try:
        best = heuristic(data, t0 + heur_budget, logger, None)
    except Exception:
        best = None

    if best is not None:
        rem = t0 + total - time.time()
        K = min(data.Kmax, max(len(best[1]), data.Kmin) + 2)
        did_mip = False
        if rem > 5 and data.n <= 12 and K <= 9:
            try:
                res = solve_with_mip(data, best[1], rem - 1.0, logger, best)
                did_mip = True
                if res is not None and res[0] < best[0] - EPS:
                    best = res
            except Exception:
                pass
        if not did_mip:
            rem = t0 + total - time.time()
            if rem > 1.0:
                try:
                    best = heuristic(data, t0 + total - 0.5, logger, best)
                except Exception:
                    pass

    if best is None or not validate(data, best[1]):
        best = trivial_solution(data)

    sol = build_solution(data, best[1])
    if logger:
        try:
            logger.log_solution(best[0], sol)
        except Exception:
            pass
    with open(args.solution_path, "w") as f:
        json.dump(sol, f)


if __name__ == "__main__":
    main()
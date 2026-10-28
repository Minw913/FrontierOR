import argparse
import json
import math
import random
import time

from solution_logger import SolutionLogger

BIG = 1e18
TOL = 1e-9


class DARPSolver:
    def __init__(self, inst):
        self.n = int(inst["num_users"])
        self.K = int(inst["num_vehicles"])
        self.Q = float(inst["vehicle_capacity"])
        self.Lmax = float(inst["maximum_ride_time"])
        self.maxDur = float(inst["maximum_route_duration"])
        nodes = inst["nodes"]
        N = len(nodes)
        self.N = N
        self.END = N - 1  # destination depot id
        e = [0.0] * N
        l = [0.0] * N
        s = [0.0] * N
        q = [0] * N
        xs = [0.0] * N
        ys = [0.0] * N
        self.pick = {}
        self.drop = {}
        self.pick_user = {}
        for nd in nodes:
            i = int(nd["node_id"])
            xs[i] = float(nd["x"])
            ys[i] = float(nd["y"])
            e[i] = float(nd["earliest_time"])
            l[i] = float(nd["latest_time"])
            s[i] = float(nd["service_duration"])
            q[i] = int(nd["load"])
            t = nd.get("node_type")
            if t == "pickup":
                u = nd["user_id"]
                self.pick[u] = i
                self.pick_user[i] = u
            elif t == "dropoff":
                self.drop[nd["user_id"]] = i
        self.e, self.l, self.s, self.q = e, l, s, q
        self.users = sorted(self.pick.keys())
        dist = [[0.0] * N for _ in range(N)]
        for i in range(N):
            xi, yi, row = xs[i], ys[i], dist[i]
            for j in range(N):
                dx = xi - xs[j]
                dy = yi - ys[j]
                row[j] = math.sqrt(dx * dx + dy * dy)
        self.dist = dist
        self.ver = [0] * self.K
        self.icache = {}
        self.deadline = float("inf")

    # ------------------------------------------------------------------
    def route_cost(self, route):
        c = 0.0
        dist = self.dist
        prev = route[0]
        for nd in route[1:]:
            c += dist[prev][nd]
            prev = nd
        return c

    def total_cost(self, routes):
        return sum(self.route_cost(r) for r in routes)

    # ------------------------------------------------------------------
    def schedule(self, route):
        """Feasible service-begin times for a route, or None if infeasible."""
        dist = self.dist; e = self.e; l = self.l; s = self.s; q = self.q
        Q = self.Q; Lmax = self.Lmax; maxDur = self.maxDur
        m = len(route)
        load = 0
        for nd in route:
            qv = q[nd]
            load += qv
            lo = qv if qv > 0 else 0
            hi = Q if qv >= 0 else Q + qv
            if load < lo - TOL or load > hi + TOL:
                return None
        B = [0.0] * m
        W = [0.0] * m
        B[0] = e[0]
        for i in range(1, m):
            pv = route[i - 1]; nd = route[i]
            arr = B[i - 1] + s[pv] + dist[pv][nd]
            bi = e[nd] if arr < e[nd] else arr
            if bi > l[nd] + TOL:
                return None
            B[i] = bi
            W[i] = bi - arr
        if B[-1] - B[0] > maxDur + TOL:
            B0 = B[-1] - maxDur
            if B0 < e[0]:
                B0 = e[0]
            if B0 > l[0] + TOL:
                return None
            B[0] = B0
            for i in range(1, m):
                pv = route[i - 1]; nd = route[i]
                arr = B[i - 1] + s[pv] + dist[pv][nd]
                bi = e[nd] if arr < e[nd] else arr
                if bi > l[nd] + TOL:
                    return None
                B[i] = bi
                W[i] = bi - arr
            if B[-1] - B[0] > maxDur + TOL:
                return None
        # ride-time repair: delay pickups using available waiting slack
        idx = {}
        for i, nd in enumerate(route):
            idx[nd] = i
        pairs = []
        pu = self.pick_user; dr = self.drop
        for i, nd in enumerate(route):
            u = pu.get(nd)
            if u is not None:
                d = dr[u]
                pairs.append((i, idx[d], nd, d))
        for _ in range(2 * m + 3):
            allok = True
            for (ip, jd, pn, dn) in pairs:
                R = B[jd] - B[ip] - s[pn]
                if R > Lmax + TOL:
                    allok = False
                    excess = R - Lmax
                    cumW = 0.0
                    for t in range(ip + 1, jd + 1):
                        cumW += W[t]
                    delta = excess
                    if cumW < delta:
                        delta = cumW
                    wl = l[pn] - B[ip]
                    if wl < delta:
                        delta = wl
                    if delta > TOL:
                        B[ip] += delta
                        for i in range(ip + 1, m):
                            pv = route[i - 1]; nd = route[i]
                            arr = B[i - 1] + s[pv] + dist[pv][nd]
                            bi = e[nd] if arr < e[nd] else arr
                            if bi > l[nd] + TOL:
                                return None
                            B[i] = bi
                            W[i] = bi - arr
                    else:
                        return None
            if allok:
                break
        if B[-1] - B[0] > maxDur + TOL:
            return None
        for (ip, jd, pn, dn) in pairs:
            if B[jd] - B[ip] - s[pn] > Lmax + TOL:
                return None
        return B

    # ------------------------------------------------------------------
    def try_insert(self, route, u, limit=BIG, max_check=6):
        """Cheapest feasible insertion of user's (pickup, dropoff) into route.

        Returns (delta_cost, pickup_pos, dropoff_pos) or None."""
        p = self.pick[u]; d = self.drop[u]
        dist = self.dist
        dp = dist[p]; dd = dist[d]
        m = len(route)
        cands = []
        for i in range(1, m):
            a = route[i - 1]; b = route[i]
            di = dist[a][p] + dp[b] - dist[a][b]
            for j in range(i + 1, m + 1):
                a2 = p if j == i + 1 else route[j - 2]
                b2 = route[j - 1]
                dj = dist[a2][d] + dd[b2] - dist[a2][b2]
                cands.append((di + dj, i, j))
        cands.sort()
        checked = 0
        for delta, i, j in cands:
            if delta >= limit:
                break
            r2 = route[:i] + [p] + route[i:j - 1] + [d] + route[j - 1:]
            if self.schedule(r2) is not None:
                return (delta, i, j)
            checked += 1
            if checked >= max_check:
                break
        return None

    def insert_eval(self, routes, ri, u, exhaustive=False):
        key = (u, ri)
        ent = self.icache.get(key)
        v = self.ver[ri]
        if ent is not None and ent[0] == v and (ent[2] or not exhaustive):
            return ent[1]
        res = self.try_insert(routes[ri], u,
                              max_check=(10 ** 9 if exhaustive else 6))
        self.icache[key] = (v, res, exhaustive)
        return res

    def apply_insertion(self, routes, ri, u, ins):
        _, i, j = ins
        r = routes[ri]
        p = self.pick[u]; d = self.drop[u]
        routes[ri] = r[:i] + [p] + r[i:j - 1] + [d] + r[j - 1:]
        self.ver[ri] += 1

    # ------------------------------------------------------------------
    def construct(self, rng):
        self.icache.clear()
        self.ver = [0] * self.K
        routes = [[0, self.END] for _ in range(self.K)]
        order = list(self.users)
        rng.shuffle(order)
        order.sort(key=lambda u: min(self.e[self.pick[u]], self.e[self.drop[u]]))
        for u in order:
            best = None
            for ri in range(self.K):
                r = self.insert_eval(routes, ri, u)
                if r is not None and (best is None or r[0] < best[0]):
                    best = (r[0], ri, r)
            if best is None:
                for ri in range(self.K):
                    r = self.insert_eval(routes, ri, u, exhaustive=True)
                    if r is not None and (best is None or r[0] < best[0]):
                        best = (r[0], ri, r)
            if best is None:
                return None
            _, ri, ins = best
            self.apply_insertion(routes, ri, u, ins)
        return routes

    def build_veh(self, routes):
        veh = {}
        pu = self.pick_user
        for ri, r in enumerate(routes):
            for nd in r:
                u = pu.get(nd)
                if u is not None:
                    veh[u] = ri
        return veh

    def remove_users(self, routes, removed):
        rem = set()
        for u in removed:
            rem.add(self.pick[u])
            rem.add(self.drop[u])
        new = []
        for ri, r in enumerate(routes):
            touched = False
            for nd in r:
                if nd in rem:
                    touched = True
                    break
            if touched:
                new.append([nd for nd in r if nd not in rem])
                self.ver[ri] += 1
            else:
                new.append(r)
        return new

    # ------------------------------------------------------------------
    def destroy(self, routes, veh_of, qn, op, rng):
        qn = min(qn, self.n)
        if op == 0:
            return rng.sample(self.users, qn)
        if op == 1:
            # worst (most costly) removal
            scored = []
            dist = self.dist
            for u in self.users:
                ri = veh_of[u]
                r = routes[ri]
                p = self.pick[u]; d = self.drop[u]
                red_cost = 0.0
                prev = None
                for nd in r:
                    if nd == p or nd == d:
                        continue
                    if prev is not None:
                        red_cost += dist[prev][nd]
                    prev = nd
                scored.append((self.route_cost(r) - red_cost, u))
            scored.sort(reverse=True)
            pool = [u for _, u in scored[:max(qn * 3, qn)]]
            return rng.sample(pool, min(qn, len(pool)))
        # related removal (Shaw-like)
        seed = rng.choice(self.users)
        ps = self.pick[seed]; ds = self.drop[seed]
        rel = []
        for u in self.users:
            rel.append((self.dist[ps][self.pick[u]]
                        + self.dist[ds][self.drop[u]], u))
        rel.sort()
        return [u for _, u in rel[:qn]]

    def repair(self, routes, removed, rng):
        unassigned = list(removed)
        while unassigned:
            if time.time() > self.deadline:
                return None
            best = None
            for u in unassigned:
                o1 = None
                o2 = None
                for ri in range(self.K):
                    r = self.insert_eval(routes, ri, u)
                    if r is not None:
                        if o1 is None or r[0] < o1[0]:
                            o2 = o1
                            o1 = (r[0], ri, r)
                        elif o2 is None or r[0] < o2[0]:
                            o2 = (r[0], ri, r)
                if o1 is None:
                    for ri in range(self.K):
                        r = self.insert_eval(routes, ri, u, exhaustive=True)
                        if r is not None and (o1 is None or r[0] < o1[0]):
                            o1 = (r[0], ri, r)
                    if o1 is None:
                        return None
                c1 = o1[0]
                c2 = o2[0] if o2 is not None else BIG
                reg = c2 - c1
                if best is None or reg > best[0] + 1e-12:
                    best = (reg, u, o1)
            _, u, (c1, ri, ins) = best
            self.apply_insertion(routes, ri, u, ins)
            unassigned.remove(u)
        for r in routes:
            if self.schedule(r) is None:
                return None
        return routes

    # ------------------------------------------------------------------
    def relocate_ls(self, routes, veh_of, source_users, rng, max_passes=8):
        K = self.K
        pick = self.pick; drop = self.drop
        any_improved = False
        src = list(source_users)
        rng.shuffle(src)
        for _ in range(max_passes):
            if time.time() > self.deadline:
                break
            improved = False
            for u in src:
                if time.time() > self.deadline:
                    return any_improved or improved
                ri = veh_of[u]
                r = routes[ri]
                p = pick[u]; d = drop[u]
                red = [nd for nd in r if nd != p and nd != d]
                if self.schedule(red) is None:
                    continue
                saving = self.route_cost(r) - self.route_cost(red)
                best = None
                for rj in range(K):
                    tgt = red if rj == ri else routes[rj]
                    ins = self.try_insert(tgt, u, limit=saving - TOL)
                    if ins is not None and (best is None or ins[0] < best[0]):
                        best = (ins[0], rj, ins)
                if best is not None:
                    _, rj, ins = best
                    routes[ri] = red
                    self.ver[ri] += 1
                    self.apply_insertion(routes, rj, u, ins)
                    veh_of[u] = rj
                    improved = True
            any_improved = any_improved or improved
            if not improved:
                break
        return any_improved

    def swap_ls(self, routes, veh_of, source_users, rng, attempts=50):
        pick = self.pick; drop = self.drop
        improved_any = False
        if not source_users:
            return False
        for _ in range(attempts):
            if time.time() > self.deadline:
                break
            u = rng.choice(source_users)
            v = rng.choice(self.users)
            if u == v:
                continue
            ri = veh_of[u]; rj = veh_of[v]
            if ri == rj:
                continue
            rA = routes[ri]; rB = routes[rj]
            pu, du = pick[u], drop[u]
            pv, dv = pick[v], drop[v]
            redA = [nd for nd in rA if nd != pu and nd != du]
            redB = [nd for nd in rB if nd != pv and nd != dv]
            base = self.route_cost(rA) + self.route_cost(rB)
            cA = self.route_cost(redA)
            cB = self.route_cost(redB)
            budget = base - cA - cB - TOL
            if budget <= 0:
                continue
            insU = self.try_insert(redB, u, limit=budget)
            if insU is None:
                continue
            insV = self.try_insert(redA, v, limit=budget - insU[0])
            if insV is None:
                continue
            routes[rj] = redB
            self.ver[rj] += 1
            self.apply_insertion(routes, rj, u, insU)
            routes[ri] = redA
            self.ver[ri] += 1
            self.apply_insertion(routes, ri, v, insV)
            veh_of[u] = rj
            veh_of[v] = ri
            improved_any = True
        return improved_any

    # ------------------------------------------------------------------
    def build_solution(self, routes):
        st = {}
        rt = {}
        out_routes = {}
        total = 0.0
        s = self.s; e = self.e; dist = self.dist
        for k, route in enumerate(routes):
            out_routes[str(k)] = [int(x) for x in route]
            total += self.route_cost(route)
            B = self.schedule(route)
            if B is None:  # should not happen; emergency fallback
                B = [0.0] * len(route)
                B[0] = e[0]
                for i in range(1, len(route)):
                    pv = route[i - 1]; nd = route[i]
                    arr = B[i - 1] + s[pv] + dist[pv][nd]
                    B[i] = e[nd] if arr < e[nd] else arr
            st["depot_start_%d" % k] = B[0]
            st["depot_end_%d" % k] = B[-1]
            for i, nd in enumerate(route):
                if nd == 0 or nd == self.END:
                    continue
                st[str(nd)] = B[i]
            idx = {nd: i for i, nd in enumerate(route)}
            for u in self.users:
                p = self.pick[u]
                if p in idx:
                    ip = idx[p]
                    jd = idx[self.drop[u]]
                    rt[str(u)] = B[jd] - B[ip] - s[p]
        return {"objective_value": total,
                "routes": out_routes,
                "service_times": st,
                "ride_times": rt}

    # ------------------------------------------------------------------
    def solve(self, deadline, logger):
        self.deadline = deadline
        rng = random.Random(987654321)
        routes = None
        attempts = 0
        while routes is None and attempts < 100:
            routes = self.construct(rng)
            attempts += 1
            if routes is None and time.time() > deadline:
                break
        if routes is None:
            self.icache.clear()
            self.ver = [0] * self.K
            routes = self.repair([[0, self.END] for _ in range(self.K)],
                                 list(self.users), rng)
        if routes is None:
            raise RuntimeError("No feasible solution could be constructed.")

        veh_of = self.build_veh(routes)
        self.relocate_ls(routes, veh_of, list(self.users), rng, max_passes=8)
        self.swap_ls(routes, veh_of, list(self.users), rng, attempts=100)

        cur = [r[:] for r in routes]
        cur_cost = self.total_cost(cur)
        best = [r[:] for r in cur]
        best_cost = cur_cost
        if logger:
            logger.log_solution(best_cost, self.build_solution(best))

        weights = [1.0, 1.0, 1.0]
        t0 = time.time()
        T0 = max(1e-6, 0.01 * max(1.0, best_cost))
        since_best = 0
        it = 0
        while time.time() < deadline:
            it += 1
            tot_w = weights[0] + weights[1] + weights[2]
            x = rng.random() * tot_w
            if x < weights[0]:
                op = 0
            elif x < weights[0] + weights[1]:
                op = 1
            else:
                op = 2
            lo_q = max(1, int(0.05 * self.n))
            hi_q = max(lo_q + 1, int(0.30 * self.n))
            qn = rng.randint(lo_q, hi_q)
            removed = self.destroy(cur, veh_of, qn, op, rng)
            partial = self.remove_users(cur, removed)
            new = self.repair(partial, removed, rng)
            if new is None:
                since_best += 1
                continue
            nveh = self.build_veh(new)
            self.relocate_ls(new, nveh, removed, rng, max_passes=4)
            if it % 4 == 0:
                self.swap_ls(new, nveh, removed, rng, attempts=30)
            nc = self.total_cost(new)
            rem_frac = (deadline - time.time()) / max(1e-9, deadline - t0)
            T = T0 * max(0.002, rem_frac)
            if nc <= cur_cost + 1e-12:
                accept = True
            else:
                accept = rng.random() < math.exp(-(nc - cur_cost) / T)
            if accept:
                cur = new
                cur_cost = nc
                veh_of = nveh
            if nc < best_cost - TOL:
                best_cost = nc
                best = [r[:] for r in new]
                since_best = 0
                weights[op] += 3.0
                if logger:
                    logger.log_solution(best_cost, self.build_solution(best))
            elif accept:
                weights[op] += 1.0
                since_best += 1
            else:
                since_best += 1
            if since_best >= 400:
                cur = [r[:] for r in best]
                cur_cost = best_cost
                veh_of = self.build_veh(cur)
                since_best = 0

        # Final polishing local search on the incumbent.
        veh_of = self.build_veh(best)
        improved = True
        while improved and time.time() < deadline:
            improved = self.relocate_ls(best, veh_of, list(self.users), rng,
                                        max_passes=10)
            if self.swap_ls(best, veh_of, list(self.users), rng, attempts=200):
                improved = True
            bc = self.total_cost(best)
            if bc < best_cost - TOL:
                best_cost = bc
                if logger:
                    logger.log_solution(best_cost, self.build_solution(best))
        return best, best_cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", required=True)
    ap.add_argument("--solution_path", required=True)
    ap.add_argument("--time_limit", type=int, required=True)
    ap.add_argument("--log_path", default=None)
    args = ap.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path) as f:
        inst = json.load(f)

    start = time.time()
    deadline = start + max(1, args.time_limit) - 0.5

    solver = DARPSolver(inst)
    best, _ = solver.solve(deadline, logger)
    sol = solver.build_solution(best)

    with open(args.solution_path, "w") as f:
        json.dump(sol, f)


if __name__ == "__main__":
    main()
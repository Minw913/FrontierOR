import argparse
import json
import math
import random
import time


class Solver:
    def __init__(self, inst, time_limit, logger):
        self.t0 = time.time()
        self.deadline = self.t0 + max(1.0, time_limit - 0.5)
        self.logger = logger

        self.N = inst["num_customers"]
        self.Q = inst["vehicle_capacity"]
        self.a = inst["cost_function"]["a"]
        self.b = inst["cost_function"]["b"]
        self.D = inst["distance_matrix"]

        dep = inst["depot"]
        horizon = inst.get("scheduling_horizon", dep["time_window"][1])

        n_nodes = self.N + 1
        self.e = [0.0] * n_nodes
        self.l = [0.0] * n_nodes
        self.s = [0.0] * n_nodes
        self.dem = [0] * n_nodes

        self.e[0] = dep["time_window"][0]
        self.l[0] = min(dep["time_window"][1], horizon)
        self.s[0] = dep.get("service_time", 0)

        for c in inst["customers"]:
            i = c["id"]
            self.e[i] = c["time_window"][0]
            self.l[i] = c["time_window"][1]
            self.s[i] = c["service_time"]
            self.dem[i] = c["demand"]

        # Build chunks (split demands exceeding capacity into near-even parts)
        self.cnode = []
        self.cq = []
        for i in range(1, n_nodes):
            d = self.dem[i]
            if d <= 0:
                continue
            k = (d + self.Q - 1) // self.Q
            base = d // k
            rem = d - base * k
            for j in range(k):
                q = base + (1 if j < rem else 0)
                self.cnode.append(i)
                self.cq.append(q)
        self.nc = len(self.cnode)

        self.singleton_cost = [self._cost_seq([c]) for c in range(self.nc)]

        self.routes = []
        self.costs = []
        self.loads = []
        self.csets = []

        self.best_routes = None
        self.best_cost = float("inf")

    # ---------- basic route evaluation ----------
    def timeout(self):
        return time.time() >= self.deadline

    def _cost_seq(self, seq):
        if not seq:
            return 0.0
        D, a, b = self.D, self.a, self.b
        load = 0
        for c in seq:
            load += self.cq[c]
        cost = 0.0
        prev = 0
        for c in seq:
            n = self.cnode[c]
            cost += D[prev][n] * (a * load + b)
            load -= self.cq[c]
            prev = n
        cost += D[prev][0] * (a * load + b)
        return cost

    def _tw_feasible(self, seq):
        D, e, l, s = self.D, self.e, self.l, self.s
        t = e[0]
        prev = 0
        for c in seq:
            n = self.cnode[c]
            arr = t + D[prev][n]
            if arr < e[n]:
                arr = e[n]
            if arr > l[n] + 1e-9:
                return False
            t = arr + s[n]
            prev = n
        return t + D[prev][0] <= self.l[0] + 1e-9

    # ---------- construction ----------
    def construct(self):
        unrouted = set(range(self.nc))
        routes = []
        while unrouted:
            if self.timeout():
                for c in unrouted:
                    routes.append([c])
                break
            seed = max(unrouted, key=lambda c: self.D[0][self.cnode[c]])
            unrouted.remove(seed)
            route = [seed]
            load = self.cq[seed]
            cset = {self.cnode[seed]}
            while True:
                if self.timeout():
                    break
                basec = self._cost_seq(route)
                best = None
                bestdelta = float("inf")
                for c in unrouted:
                    if load + self.cq[c] > self.Q:
                        continue
                    if self.cnode[c] in cset:
                        continue
                    for j in range(len(route) + 1):
                        seq = route[:j] + [c] + route[j:]
                        if not self._tw_feasible(seq):
                            continue
                        delta = self._cost_seq(seq) - basec
                        if delta < bestdelta:
                            bestdelta = delta
                            best = (c, seq)
                if best is None:
                    break
                c, seq = best
                if bestdelta >= self.singleton_cost[c] - 1e-9:
                    break
                route = seq
                unrouted.remove(c)
                load += self.cq[c]
                cset.add(self.cnode[c])
            routes.append(route)
        return routes

    def rebuild(self):
        self.routes = [r for r in self.routes if r]
        self.costs = [self._cost_seq(r) for r in self.routes]
        self.loads = [sum(self.cq[c] for c in r) for r in self.routes]
        self.csets = [set(self.cnode[c] for c in r) for r in self.routes]

    def total_cost(self):
        return sum(self.costs)

    # ---------- local search moves ----------
    def relocate_pass(self):
        improved = False
        i1 = 0
        while i1 < len(self.routes):
            if self.timeout():
                return improved
            pos1 = 0
            while pos1 < len(self.routes[i1]):
                r1 = self.routes[i1]
                c = r1[pos1]
                q = self.cq[c]
                cust = self.cnode[c]
                r1new = r1[:pos1] + r1[pos1 + 1:]
                c1new = self._cost_seq(r1new)
                base1 = self.costs[i1]
                best = None
                bestdelta = -1e-7
                # intra-route reinsertion
                for j in range(len(r1new) + 1):
                    if j == pos1:
                        continue
                    seq = r1new[:j] + [c] + r1new[j:]
                    if not self._tw_feasible(seq):
                        continue
                    delta = self._cost_seq(seq) - base1
                    if delta < bestdelta:
                        bestdelta = delta
                        best = ("intra", seq)
                # inter-route
                for i2 in range(len(self.routes)):
                    if i2 == i1:
                        continue
                    if self.loads[i2] + q > self.Q:
                        continue
                    if cust in self.csets[i2]:
                        continue
                    r2 = self.routes[i2]
                    for j in range(len(r2) + 1):
                        seq2 = r2[:j] + [c] + r2[j:]
                        if not self._tw_feasible(seq2):
                            continue
                        delta = c1new + self._cost_seq(seq2) - base1 - self.costs[i2]
                        if delta < bestdelta:
                            bestdelta = delta
                            best = ("inter", i2, seq2)
                # new dedicated route
                if len(r1) > 1:
                    delta = c1new + self.singleton_cost[c] - base1
                    if delta < bestdelta:
                        bestdelta = delta
                        best = ("new",)
                if best is None:
                    pos1 += 1
                    continue
                improved = True
                if best[0] == "intra":
                    self.routes[i1] = best[1]
                    self.costs[i1] = self._cost_seq(best[1])
                elif best[0] == "inter":
                    i2, seq2 = best[1], best[2]
                    self.routes[i1] = r1new
                    self.costs[i1] = c1new
                    self.loads[i1] -= q
                    self.csets[i1].discard(cust)
                    self.routes[i2] = seq2
                    self.costs[i2] = self._cost_seq(seq2)
                    self.loads[i2] += q
                    self.csets[i2].add(cust)
                else:
                    self.routes[i1] = r1new
                    self.costs[i1] = c1new
                    self.loads[i1] -= q
                    self.csets[i1].discard(cust)
                    self.routes.append([c])
                    self.costs.append(self.singleton_cost[c])
                    self.loads.append(q)
                    self.csets.append({cust})
                if self.timeout():
                    return improved
            i1 += 1
        return improved

    def swap_pass(self):
        improved = False
        nr = len(self.routes)
        for i1 in range(nr - 1):
            if self.timeout():
                return improved
            for p1 in range(len(self.routes[i1])):
                for i2 in range(i1 + 1, nr):
                    r1 = self.routes[i1]
                    if p1 >= len(r1):
                        break
                    c1 = r1[p1]
                    q1 = self.cq[c1]
                    n1 = self.cnode[c1]
                    r2 = self.routes[i2]
                    for p2 in range(len(r2)):
                        c2 = r2[p2]
                        q2 = self.cq[c2]
                        n2 = self.cnode[c2]
                        if self.loads[i1] - q1 + q2 > self.Q:
                            continue
                        if self.loads[i2] - q2 + q1 > self.Q:
                            continue
                        if n2 != n1 and (n2 in self.csets[i1] or n1 in self.csets[i2]):
                            continue
                        new1 = r1[:]
                        new1[p1] = c2
                        new2 = r2[:]
                        new2[p2] = c1
                        if not self._tw_feasible(new1) or not self._tw_feasible(new2):
                            continue
                        cn1 = self._cost_seq(new1)
                        cn2 = self._cost_seq(new2)
                        if cn1 + cn2 < self.costs[i1] + self.costs[i2] - 1e-7:
                            self.routes[i1] = new1
                            self.routes[i2] = new2
                            self.costs[i1] = cn1
                            self.costs[i2] = cn2
                            self.loads[i1] += q2 - q1
                            self.loads[i2] += q1 - q2
                            self.csets[i1].discard(n1)
                            self.csets[i1].add(n2)
                            self.csets[i2].discard(n2)
                            self.csets[i2].add(n1)
                            improved = True
                            break
                    if self.timeout():
                        return improved
        return improved

    def two_opt_pass(self):
        improved = False
        for i in range(len(self.routes)):
            if self.timeout():
                return improved
            r = self.routes[i]
            n = len(r)
            if n < 3:
                continue
            changed = True
            while changed:
                if self.timeout():
                    return improved
                changed = False
                base = self.costs[i]
                for p in range(n - 1):
                    for q in range(p + 1, n):
                        seq = r[:p] + r[p:q + 1][::-1] + r[q + 1:]
                        if not self._tw_feasible(seq):
                            continue
                        c = self._cost_seq(seq)
                        if c < base - 1e-7:
                            self.routes[i] = seq
                            self.costs[i] = c
                            r = seq
                            changed = True
                            improved = True
                            break
                    if changed:
                        break
        return improved

    def local_search(self):
        while not self.timeout():
            imp = False
            if self.relocate_pass():
                imp = True
            if self.timeout():
                break
            if self.two_opt_pass():
                imp = True
            if self.timeout():
                break
            if self.swap_pass():
                imp = True
            if not imp:
                break
        self.rebuild()

    # ---------- perturbation ----------
    def perturb(self, rng):
        all_pos = []
        for i, r in enumerate(self.routes):
            for p in range(len(r)):
                all_pos.append((i, p))
        if not all_pos:
            return
        k = max(1, min(len(all_pos) - 1, int(0.15 * len(all_pos)) + rng.randint(0, 2)))
        chosen = rng.sample(all_pos, k)
        removed = []
        by_route = {}
        for (i, p) in chosen:
            by_route.setdefault(i, []).append(p)
        for i, ps in by_route.items():
            for p in sorted(ps, reverse=True):
                removed.append(self.routes[i].pop(p))
        self.rebuild()
        rng.shuffle(removed)
        for c in removed:
            q = self.cq[c]
            cust = self.cnode[c]
            best = None
            bestdelta = self.singleton_cost[c]
            for i2 in range(len(self.routes)):
                if self.loads[i2] + q > self.Q:
                    continue
                if cust in self.csets[i2]:
                    continue
                r2 = self.routes[i2]
                for j in range(len(r2) + 1):
                    seq2 = r2[:j] + [c] + r2[j:]
                    if not self._tw_feasible(seq2):
                        continue
                    delta = self._cost_seq(seq2) - self.costs[i2]
                    if delta < bestdelta:
                        bestdelta = delta
                        best = (i2, seq2)
                if self.timeout():
                    break
            if best is None:
                self.routes.append([c])
                self.costs.append(self.singleton_cost[c])
                self.loads.append(q)
                self.csets.append({cust})
            else:
                i2, seq2 = best
                self.routes[i2] = seq2
                self.costs[i2] = self._cost_seq(seq2)
                self.loads[i2] += q
                self.csets[i2].add(cust)

    # ---------- output ----------
    def to_solution(self, routes):
        out_routes = []
        total = 0.0
        vid = 0
        for seq in routes:
            if not seq:
                continue
            nodes = [0] + [self.cnode[c] for c in seq] + [0]
            deliveries = {}
            for c in seq:
                key = str(self.cnode[c])
                deliveries[key] = deliveries.get(key, 0) + self.cq[c]
            edges = []
            load = sum(self.cq[c] for c in seq)
            rc = 0.0
            prev = 0
            for c in seq:
                n = self.cnode[c]
                edges.append({"from": prev, "to": n, "load_weight": float(load)})
                rc += self.D[prev][n] * (self.a * load + self.b)
                load -= self.cq[c]
                prev = n
            edges.append({"from": prev, "to": 0, "load_weight": float(load)})
            rc += self.D[prev][0] * (self.a * load + self.b)
            total += rc
            out_routes.append({
                "vehicle": vid,
                "route": nodes,
                "deliveries": deliveries,
                "edges": edges,
            })
            vid += 1
        return {"objective_value": total, "routes": out_routes}

    def maybe_update_best(self):
        tc = self.total_cost()
        if tc < self.best_cost - 1e-7:
            self.best_cost = tc
            self.best_routes = [r[:] for r in self.routes if r]
            if self.logger:
                sol = self.to_solution(self.best_routes)
                self.logger.log_solution(sol["objective_value"], sol)
            return True
        return False

    def solve(self):
        rng = random.Random(0)
        self.routes = self.construct()
        self.rebuild()
        self.maybe_update_best()
        self.local_search()
        self.maybe_update_best()
        while not self.timeout():
            self.routes = [r[:] for r in self.best_routes]
            self.rebuild()
            self.perturb(rng)
            self.local_search()
            self.maybe_update_best()
        return self.to_solution(self.best_routes if self.best_routes else [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", required=True)
    ap.add_argument("--solution_path", required=True)
    ap.add_argument("--time_limit", type=int, default=60)
    ap.add_argument("--log_path", default=None)
    args = ap.parse_args()

    logger = None
    if args.log_path:
        try:
            from solution_logger import SolutionLogger
            logger = SolutionLogger(args.log_path, sense="minimize")
        except Exception:
            logger = None

    with open(args.instance_path) as f:
        inst = json.load(f)

    random.seed(0)
    solver = Solver(inst, args.time_limit, logger)
    solution = solver.solve()

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()
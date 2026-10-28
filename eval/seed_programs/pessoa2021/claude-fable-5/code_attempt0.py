import argparse
import json
import math
import random
import time

from solution_logger import SolutionLogger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--instance_path', required=True)
    ap.add_argument('--solution_path', required=True)
    ap.add_argument('--time_limit', type=int, default=60)
    ap.add_argument('--log_path', default=None)
    args = ap.parse_args()

    start_time = time.time()
    deadline = start_time + max(5, args.time_limit) - 1.5

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path) as f:
        inst = json.load(f)

    n = int(inst['num_customers'])
    K = int(inst['num_vehicles'])
    Q = int(inst['vehicle_capacity'])
    Gamma = int(inst['uncertainty_set']['Gamma'])
    D = inst['distance_matrix']

    N = n + 1
    nom = [0] * N
    dev = [0] * N
    customers = []
    for c in inst['customers']:
        cid = int(c['id'])
        nom[cid] = int(c['nominal_demand'])
        dev[cid] = int(c['demand_deviation'])
        customers.append(cid)

    random.seed(0)

    maxD = max(max(row) for row in D) if D else 1
    PEN = 50.0 * (maxD + 1)

    # ---------- helpers ----------
    def route_cost(r):
        if not r:
            return 0
        c = D[0][r[0]] + D[r[-1]][0]
        for i in range(len(r) - 1):
            c += D[r[i]][r[i + 1]]
        return c

    def worst_demand(r):
        s = 0
        for c in r:
            s += nom[c]
        if Gamma > 0 and r:
            ds = sorted((dev[c] for c in r), reverse=True)
            s += sum(ds[:Gamma])
        return s

    class Sol:
        __slots__ = ('routes', 'rcost', 'rnom', 'rtop', 'rworst')

        def __init__(self, routes, blank=False):
            if blank:
                return
            self.routes = [list(r) for r in routes]
            self.rcost = [route_cost(r) for r in self.routes]
            self.rnom = [sum(nom[c] for c in r) for r in self.routes]
            self.rtop = [sorted((dev[c] for c in r), reverse=True)[:Gamma]
                         for r in self.routes]
            self.rworst = [self.rnom[i] + sum(self.rtop[i]) for i in range(len(self.routes))]

        def update(self, i):
            r = self.routes[i]
            self.rcost[i] = route_cost(r)
            self.rnom[i] = sum(nom[c] for c in r)
            self.rtop[i] = sorted((dev[c] for c in r), reverse=True)[:Gamma]
            self.rworst[i] = self.rnom[i] + sum(self.rtop[i])

        def cost(self):
            return sum(self.rcost)

        def overload(self):
            return sum(max(0, w - Q) for w in self.rworst)

        def pobj(self):
            return self.cost() + PEN * self.overload()

        def copy(self):
            s = Sol(None, blank=True)
            s.routes = [r[:] for r in self.routes]
            s.rcost = self.rcost[:]
            s.rnom = self.rnom[:]
            s.rtop = [t[:] for t in self.rtop]
            s.rworst = self.rworst[:]
            return s

    def new_worst(sol, ri, c):
        # worst-case demand if c is added to route ri
        base = sol.rnom[ri] + nom[c]
        if Gamma <= 0:
            return base
        merged = sol.rtop[ri] + [dev[c]]
        merged.sort(reverse=True)
        return base + sum(merged[:Gamma])

    def best_insertion(route, c):
        if not route:
            return D[0][c] + D[c][0], 0
        best = None
        bp = 0
        m = len(route)
        for i in range(m + 1):
            a = route[i - 1] if i > 0 else 0
            b = route[i] if i < m else 0
            delta = D[a][c] + D[c][b] - D[a][b]
            if best is None or delta < best:
                best = delta
                bp = i
        return best, bp

    def intra_opt(r):
        # asymmetric-safe intra-route improvement: relocation + 2-opt (recompute)
        if len(r) < 2:
            return r
        improved = True
        guard = 0
        while improved and guard < 60:
            guard += 1
            improved = False
            m = len(r)
            # single-customer relocation
            base_cost = route_cost(r)
            for i in range(m):
                c = r[i]
                rest = r[:i] + r[i + 1:]
                for j in range(len(rest) + 1):
                    if j == i:
                        continue
                    cand = rest[:j] + [c] + rest[j:]
                    cc = route_cost(cand)
                    if cc < base_cost - 1e-9:
                        r[:] = cand
                        base_cost = cc
                        improved = True
                        m = len(r)
                        break
                if improved:
                    break
            if improved:
                continue
            # 2-opt with full recompute (handles asymmetric matrices)
            for i in range(m - 1):
                for j in range(i + 1, m):
                    cand = r[:i] + r[i:j + 1][::-1] + r[j + 1:]
                    cc = route_cost(cand)
                    if cc < base_cost - 1e-9:
                        r[:] = cand
                        base_cost = cc
                        improved = True
                        break
                if improved:
                    break
        return r

    # ---------- repair (regret-2 insertion) ----------
    def repair(sol, removed):
        removed = list(removed)
        random.shuffle(removed)
        while removed:
            pick = None          # (regret, -delta, cust, ri, pos)
            forced = None        # customer with no feasible route
            forced_choice = None
            for c in removed:
                feas_opts = []
                all_opts = []
                for ri in range(K):
                    nw = new_worst(sol, ri, c)
                    delta, pos = best_insertion(sol.routes[ri], c)
                    all_opts.append((delta, ri, pos, nw))
                    if nw <= Q:
                        feas_opts.append((delta, ri, pos))
                if feas_opts:
                    feas_opts.sort(key=lambda o: o[0])
                    d1 = feas_opts[0][0]
                    d2 = feas_opts[1][0] if len(feas_opts) > 1 else d1 + 1e7
                    regret = d2 - d1
                    cand = (regret, -d1, c, feas_opts[0][1], feas_opts[0][2])
                    if pick is None or cand > pick:
                        pick = cand
                else:
                    # penalized fallback
                    best_p = None
                    for (delta, ri, pos, nw) in all_opts:
                        inc = max(0, nw - Q) - max(0, sol.rworst[ri] - Q)
                        pcost = delta + PEN * inc
                        if best_p is None or pcost < best_p[0]:
                            best_p = (pcost, c, ri, pos)
                    if forced is None:
                        forced = c
                        forced_choice = best_p
            if forced is not None:
                _, c, ri, pos = forced_choice
            else:
                _, _, c, ri, pos = pick
            sol.routes[ri].insert(pos, c)
            sol.update(ri)
            removed.remove(c)

    # ---------- destroy ----------
    def destroy(sol, q):
        removed = []
        op = random.random()
        nonempty = [i for i in range(K) if sol.routes[i]]
        if not nonempty:
            return removed
        if op < 0.35:
            # random removal
            present = [c for r in sol.routes for c in r]
            random.shuffle(present)
            removed = present[:q]
        elif op < 0.80:
            # related removal
            present = [c for r in sol.routes for c in r]
            seed = random.choice(present)
            present.sort(key=lambda c: D[seed][c] + D[c][seed])
            removed = present[:q]
        else:
            # route removal
            ri = random.choice(nonempty)
            removed = list(sol.routes[ri])
        rset = set(removed)
        touched = set()
        for i in range(K):
            r = sol.routes[i]
            nr = [c for c in r if c not in rset]
            if len(nr) != len(r):
                sol.routes[i] = nr
                touched.add(i)
        for i in touched:
            sol.update(i)
        return removed

    # ---------- initial solution ----------
    cur = Sol([[] for _ in range(K)])
    repair(cur, customers)
    for i in range(K):
        intra_opt(cur.routes[i])
        cur.update(i)

    best = cur.copy()
    best_cost = best.cost() if best.overload() == 0 else float('inf')

    def build_solution(sol):
        routes_out = []
        total = 0
        for r in sol.routes:
            rc = route_cost(r)
            routes_out.append({
                "customers": [int(c) for c in r],
                "cost": int(rc),
                "worst_case_demand": int(worst_demand(r))
            })
            total += rc
        return {"objective_value": float(total), "routes": routes_out}

    if best_cost < float('inf') and logger:
        logger.log_solution(best_cost, build_solution(best))

    # ---------- LNS with simulated annealing ----------
    T0 = max(1.0, 0.05 * max(1.0, cur.cost()))
    T = T0
    qmax = max(3, min(25, max(3, n // 3)))
    it = 0
    while time.time() < deadline and n > 0:
        it += 1
        cand = cur.copy()
        q = random.randint(2, qmax) if n >= 2 else 1
        removed = destroy(cand, q)
        if removed:
            repair(cand, removed)
        oc = cand.pobj()
        ocur = cur.pobj()
        accept = False
        if oc <= ocur + 1e-9:
            accept = True
        else:
            try:
                if random.random() < math.exp(-(oc - ocur) / max(T, 1e-9)):
                    accept = True
            except OverflowError:
                accept = False
        if accept:
            cur = cand
            if cur.overload() == 0 and cur.cost() < best_cost:
                # polish routes
                for i in range(K):
                    if cur.routes[i]:
                        intra_opt(cur.routes[i])
                        cur.update(i)
                if cur.cost() < best_cost:
                    best = cur.copy()
                    best_cost = best.cost()
                    if logger:
                        logger.log_solution(best_cost, build_solution(best))
        T *= 0.9997
        if T < T0 * 0.002:
            T = T0 * 0.4
        # occasional restart from best to intensify
        if it % 4000 == 0 and best_cost < float('inf'):
            cur = best.copy()

    # ---------- output ----------
    final = best if best_cost < float('inf') else cur
    sol_dict = build_solution(final)
    if logger:
        logger.log_solution(sol_dict["objective_value"], sol_dict)
    with open(args.solution_path, 'w') as f:
        json.dump(sol_dict, f, indent=2)


if __name__ == '__main__':
    main()
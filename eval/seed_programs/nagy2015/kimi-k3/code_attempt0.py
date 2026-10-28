import argparse
import json
import math
import random
import time

from solution_logger import SolutionLogger


class VRPBHeuristic:
    """Heuristic solver for the VRP with linehaul/backhaul (pickup & delivery)
    requests expanded into separate nodes (1..n linehaul, n+1..2n backhaul)."""

    EPS = 1e-9

    def __init__(self, instance):
        self.N = int(instance["num_customers"])
        self.C = float(instance["vehicle_capacity"])
        N = self.N
        # original index (0 = depot, 1..N customers) for every expanded node id
        self.orig = [0] * (2 * N + 1)
        for u in range(1, 2 * N + 1):
            self.orig[u] = u if u <= N else u - N
        M = instance.get("distance_matrix")
        if M is None:
            # fallback: rounded euclidean (schema promises distance_matrix)
            pts = {0: (instance["depot"]["x"], instance["depot"]["y"])}
            for c in instance["customers"]:
                pts[c["id"]] = (c["x"], c["y"])
            M = [
                [
                    int(
                        round(
                            math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1])
                        )
                    )
                    for j in range(N + 1)
                ]
                for i in range(N + 1)
            ]
        # expanded distance matrix (backhaul nodes reuse their customer's column)
        o = self.orig
        self.d = [[M[o[u]][o[v]] for v in range(2 * N + 1)] for u in range(2 * N + 1)]
        # per-node demands: dq>0 for linehaul nodes, pq>0 for backhaul nodes
        self.dq = [0.0] * (2 * N + 1)
        self.pq = [0.0] * (2 * N + 1)
        for c in instance["customers"]:
            i = int(c["id"])
            self.dq[i] = float(c.get("delivery_demand", 0))
            self.pq[N + i] = float(c.get("pickup_demand", 0))
        # active requests (skip zero demand requests)
        self.nodes = [
            u
            for u in range(1, 2 * N + 1)
            if (u <= N and self.dq[u] > 0) or (u > N and self.pq[u] > 0)
        ]

    # ---------- basic helpers ----------
    def rdist(self, r):
        if not r:
            return 0.0
        d = self.d
        total = d[0][r[0]] + d[r[-1]][0]
        for a, b in zip(r, r[1:]):
            total += d[a][b]
        return total

    def total(self, routes):
        return sum(self.rdist(r) for r in routes if r)

    # statistics for capacity feasibility of a route
    def stats(self, r):
        """Returns (loads, prefix_max, suffix_max, feasible). loads[t] = load
        after visiting first t stops (t=0..len)."""
        N = self.N
        dq = self.dq
        pq = self.pq
        L = 0.0
        for u in r:
            if u <= N:
                L += dq[u]
        loads = [L]
        mx = L
        for u in r:
            if u <= N:
                L -= dq[u]
            else:
                L += pq[u]
            loads.append(L)
            if L > mx:
                mx = L
        T = len(r)
        pmax = [0.0] * (T + 1)
        smax = [0.0] * (T + 1)
        run = loads[0]
        pmax[0] = run
        for t in range(1, T + 1):
            if loads[t] > run:
                run = loads[t]
            pmax[t] = run
        run = loads[T]
        smax[T] = run
        for t in range(T - 1, -1, -1):
            if loads[t] > run:
                run = loads[t]
            smax[t] = run
        return (loads, pmax, smax, mx <= self.C + self.EPS)

    def route_ok(self, r):
        return self.stats(r)[3]

    def ins_ok(self, u, r, st, pos):
        """O(1) feasibility check of inserting node u at position pos."""
        qd = self.dq[u]
        qp = self.pq[u]
        loads, pmax, smax, _ = st
        if qd > 0:  # linehaul node
            if pos == 0:
                return loads[0] + qd <= self.C + self.EPS
            return pmax[pos - 1] + qd <= self.C + self.EPS
        else:  # backhaul node
            if pos == len(r):
                return loads[-1] + qp <= self.C + self.EPS
            return smax[pos] + qp <= self.C + self.EPS

    # ---------- construction: Clarke-Wright savings ----------
    def initial_solution(self, rng, deadline):
        nodes = self.nodes
        m = len(nodes)
        if m == 0:
            return []
        routes = [[u] for u in nodes]
        route_of = {u: i for i, u in enumerate(nodes)}
        d = self.d
        # candidate pairs (restrict neighborhood for very large instances)
        limit = 100 if m > 900 else m
        pairs = []
        for i in nodes:
            others = sorted((d[i][j], j) for j in nodes if j != i)
            for _dist, j in others[:limit]:
                if j > i:
                    s = d[i][0] + d[0][j] - d[i][j]
                    pairs.append((s, i, j))
        pairs.sort(key=lambda x: -x[0])
        for _s, i, j in pairs:
            if time.time() > deadline:
                break
            ri = route_of.get(i)
            rj = route_of.get(j)
            if ri is None or rj is None or ri == rj:
                continue
            A = routes[ri]
            B = routes[rj]
            if not A or not B:
                continue
            # orient A so that i is last, B so that j is first
            if A[0] == i:
                A = A[::-1]
            elif A[-1] != i:
                continue
            if B[-1] == j:
                B = B[::-1]
            elif B[0] != j:
                continue
            new = A + B
            if self.route_ok(new):
                routes[ri] = new
                routes[rj] = []
                for u in new:
                    route_of[u] = ri
        return [r for r in routes if r]

    # ---------- local search ----------
    def two_opt(self, r, deadline):
        any_change = False
        d = self.d
        improved = True
        L = len(r)
        while improved and time.time() < deadline:
            improved = False
            L = len(r)
            for i in range(L):
                restart = False
                for j in range(i + 1, L):
                    a = r[i - 1] if i > 0 else 0
                    b = r[j + 1] if j + 1 < L else 0
                    delta = d[a][r[j]] + d[r[i]][b] - d[a][r[i]] - d[r[j]][b]
                    if delta < -self.EPS:
                        cand = r[:i] + r[i : j + 1][::-1] + r[j + 1 :]
                        if self.route_ok(cand):
                            r[:] = cand
                            improved = True
                            any_change = True
                            restart = True
                            break
                if restart:
                    break
        return any_change

    def relocate_sweep(self, routes, stats, deadline):
        any_change = False
        d = self.d
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for ri in range(len(routes)):
                r = routes[ri]
                if not r:
                    continue
                restart = False
                for pos in range(len(r)):
                    u = r[pos]
                    without = r[:pos] + r[pos + 1 :]
                    saved = self.rdist(r) - self.rdist(without)
                    best_d = math.inf
                    best_tj = -1
                    best_p = 0
                    for tj in range(len(routes)):
                        if tj == ri:
                            continue
                        tr = routes[tj]
                        st = stats[tj]
                        L = len(tr)
                        for p2 in range(L + 1):
                            a = tr[p2 - 1] if p2 > 0 else 0
                            b = tr[p2] if p2 < L else 0
                            delta = d[a][u] + d[u][b] - d[a][b]
                            if delta < best_d and self.ins_ok(u, tr, st, p2):
                                best_d = delta
                                best_tj = tj
                                best_p = p2
                    if best_tj >= 0 and best_d < saved - self.EPS:
                        node = r.pop(pos)
                        routes[best_tj].insert(best_p, node)
                        stats[ri] = self.stats(r)
                        stats[best_tj] = self.stats(routes[best_tj])
                        improved = True
                        any_change = True
                        restart = True
                        break
                if restart:
                    break
        return any_change

    def swap_sweep(self, routes, stats, deadline):
        any_change = False
        d = self.d
        improved = True
        while improved and time.time() < deadline:
            improved = False
            restart_outer = False
            for ri in range(len(routes)):
                A = routes[ri]
                if not A:
                    continue
                for tj in range(len(routes)):
                    if tj == ri or not routes[tj]:
                        continue
                    B = routes[tj]
                    moved = False
                    for i, u in enumerate(A):
                        a1 = A[i - 1] if i > 0 else 0
                        b1 = A[i + 1] if i + 1 < len(A) else 0
                        base1 = d[a1][u] + d[u][b1] - d[a1][b1]
                        for j, v in enumerate(B):
                            a2 = B[j - 1] if j > 0 else 0
                            b2 = B[j + 1] if j + 1 < len(B) else 0
                            base2 = d[a2][v] + d[v][b2] - d[a2][b2]
                            delta = (
                                d[a1][v] + d[v][b1] - d[a1][b1] - base1
                            ) + (
                                d[a2][u] + d[u][b2] - d[a2][b2] - base2
                            )
                            if delta < -self.EPS:
                                newA = A[:]
                                newB = B[:]
                                newA[i] = v
                                newB[j] = u
                                if self.route_ok(newA) and self.route_ok(newB):
                                    A[i] = v
                                    B[j] = u
                                    stats[ri] = self.stats(A)
                                    stats[tj] = self.stats(B)
                                    improved = True
                                    any_change = True
                                    moved = True
                                    break
                        if moved:
                            break
                    if moved:
                        restart_outer = True
                        break
                if restart_outer:
                    break
        return any_change

    def local_search(self, routes, deadline):
        routes[:] = [r for r in routes if r]
        if not routes:
            return
        stats = [self.stats(r) for r in routes]
        while time.time() < deadline:
            improved = False
            for ri in range(len(routes)):
                if routes[ri] and self.two_opt(routes[ri], deadline):
                    improved = True
                    stats[ri] = self.stats(routes[ri])
            if time.time() > deadline:
                break
            if self.relocate_sweep(routes, stats, deadline):
                improved = True
            if time.time() > deadline:
                break
            if len(self.nodes) <= 800 and self.swap_sweep(routes, stats, deadline):
                improved = True
            if not improved:
                break

    # ---------- LNS destroy / repair ----------
    def destroy(self, routes, k, rng):
        nonempt = [ri for ri in range(len(routes)) if routes[ri]]
        if not nonempt:
            return []
        all_nodes = [
            (ri, pos)
            for ri in nonempt
            for pos in range(len(routes[ri]))
        ]
        op = rng.random()
        removed = []
        if op < 0.35 and len(nonempt) > 1:
            # remove whole route
            ri = rng.choice(nonempt)
            removed = list(routes[ri])
        elif op < 0.75:
            # proximity cluster removal
            seed_ri, seed_pos = rng.choice(all_nodes)
            seed = routes[seed_ri][seed_pos]
            d = self.d
            all_nodes.sort(key=lambda rp: d[seed][routes[rp[0]][rp[1]]])
            kk = min(k, len(all_nodes))
            removed = [routes[ri][pos] for ri, pos in all_nodes[:kk]]
        else:
            kk = min(k, len(all_nodes))
            chosen = rng.sample(all_nodes, kk)
            removed = [routes[ri][pos] for ri, pos in chosen]
        rem = set(removed)
        for ri in range(len(routes)):
            if routes[ri]:
                routes[ri] = [v for v in routes[ri] if v not in rem]
        routes[:] = [r for r in routes if r]
        return removed

    def insertion_options(self, u, routes, stats):
        d = self.d
        opts = []
        for ri in range(len(routes)):
            r = routes[ri]
            if not r:
                continue
            st = stats[ri]
            L = len(r)
            best_d = math.inf
            best_p = 0
            found = False
            for pos in range(L + 1):
                a = r[pos - 1] if pos > 0 else 0
                b = r[pos] if pos < L else 0
                delta = d[a][u] + d[u][b] - d[a][b]
                if delta < best_d and self.ins_ok(u, r, st, pos):
                    best_d = delta
                    best_p = pos
                    found = True
            if found:
                opts.append((best_d, ri, best_p))
        # option of opening a new route
        opts.append((d[0][u] + d[u][0], -1, 0))
        opts.sort(key=lambda x: x[0])
        return opts

    def repair(self, routes, removed, rng):
        stats = [self.stats(r) for r in routes]
        while removed:
            best_choice = None
            best_regret = -1.0
            for u in removed:
                opts = self.insertion_options(u, routes, stats)
                if len(opts) == 1:
                    regret = float("inf")
                else:
                    regret = opts[1][0] - opts[0][0]
                if regret > best_regret:
                    best_regret = regret
                    best_choice = (u, opts[0][1], opts[0][2])
            u, ri, pos = best_choice
            if ri == -1 or ri >= len(routes) or not routes[ri] and ri == -1:
                routes.append([u])
                stats.append(self.stats([u]))
            elif ri == -1:
                routes.append([u])
                stats.append(self.stats([u]))
            elif not routes[ri]:
                routes[ri] = [u]
                stats[ri] = self.stats([u])
            else:
                routes[ri].insert(pos, u)
                stats[ri] = self.stats(routes[ri])
            removed.remove(u)

    # ---------- output ----------
    def build_solution(self, routes, obj):
        routes_out = []
        detailed = []
        for r in routes:
            if not r:
                continue
            path = [0] + list(r) + [0]
            routes_out.append(path)
            det = []
            for node in path:
                if node == 0:
                    det.append(
                        {
                            "node_id": 0,
                            "role": "depot",
                            "customer_id": 0,
                            "type": "depot",
                            "quantity": 0,
                        }
                    )
                elif node <= self.N:
                    det.append(
                        {
                            "node_id": node,
                            "role": "linehaul",
                            "customer_id": node,
                            "type": "delivery",
                            "quantity": self.dq[node],
                        }
                    )
                else:
                    det.append(
                        {
                            "node_id": node,
                            "role": "backhaul",
                            "customer_id": node - self.N,
                            "type": "pickup",
                            "quantity": self.pq[node],
                        }
                    )
            detailed.append(det)
        return {
            "objective_value": float(obj),
            "routes": routes_out,
            "routes_detailed": detailed,
        }

    # ---------- main solve ----------
    def solve(self, time_limit, logger):
        start = time.time()
        deadline = start + max(time_limit, 0)
        rng = random.Random(0)
        m = len(self.nodes)
        if m == 0:
            return [], 0.0

        routes = self.initial_solution(rng, deadline)
        self.local_search(routes, deadline)

        cur = [r[:] for r in routes]
        cur_obj = self.total(cur)
        best = [r[:] for r in cur]
        best_obj = cur_obj
        if logger:
            logger.log_solution(best_obj, self.build_solution(best, best_obj))

        kmax = max(2, min(40, max(3, m // 5)))
        no_improve = 0
        buffer = 0.05
        while time.time() < deadline - buffer:
            cand = [r[:] for r in cur]
            k = rng.randint(2, kmax)
            removed = self.destroy(cand, k, rng)
            if not removed:
                no_improve += 1
                if no_improve >= 200:
                    cur = [r[:] for r in best]
                    cur_obj = best_obj
                    no_improve = 0
                continue
            self.repair(cand, removed, rng)
            if time.time() < deadline - buffer:
                self.local_search(cand, deadline)
            obj = self.total(cand)
            accepted = obj <= cur_obj * (1.0 + 0.002)
            if accepted:
                cur = cand
                cur_obj = obj
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= 200:
                    cur = [r[:] for r in best]
                    cur_obj = best_obj
                    no_improve = 0
            if obj < best_obj - self.EPS:
                best = [r[:] for r in cand]
                best_obj = obj
                if logger:
                    logger.log_solution(
                        best_obj, self.build_solution(best, best_obj)
                    )
        return best, best_obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    logger = (
        SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    )

    solver = VRPBHeuristic(instance)
    best_routes, best_obj = solver.solve(args.time_limit, logger)
    solution = solver.build_solution(best_routes, best_obj)

    with open(args.solution_path, "w") as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()
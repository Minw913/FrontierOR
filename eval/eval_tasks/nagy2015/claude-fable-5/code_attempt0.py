import argparse
import json
import math
import random
import time
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", required=True)
    ap.add_argument("--solution_path", required=True)
    ap.add_argument("--time_limit", type=int, default=60)
    ap.add_argument("--log_path", default=None)
    args = ap.parse_args()

    t_start = time.time()
    deadline = t_start + max(1, args.time_limit) - 1.0

    logger = None
    if args.log_path:
        try:
            from solution_logger import SolutionLogger
            logger = SolutionLogger(args.log_path, sense="minimize")
        except Exception:
            logger = None

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    n = data["num_customers"]
    C = data["vehicle_capacity"]
    D = data["distance_matrix"]

    dd = [0] * (n + 1)  # delivery demand per customer
    pp = [0] * (n + 1)  # pickup demand per customer
    for c in data["customers"]:
        i = c["id"]
        dd[i] = c["delivery_demand"]
        pp[i] = c["pickup_demand"]

    # Expanded nodes: 1..n linehaul (delivery), n+1..2n backhaul (pickup)
    loc = [0] * (2 * n + 1)
    for i in range(1, n + 1):
        loc[i] = i
        loc[n + i] = i
    nodes = list(range(1, 2 * n + 1))
    N = len(nodes)

    rng = random.Random(0)

    def node_amt(u):
        return dd[u] if u <= n else pp[u - n]

    def total_cost(routes):
        c = 0
        for r in routes:
            prev = 0
            for u in r:
                lu = loc[u]
                c += D[prev][lu]
                prev = lu
            c += D[prev][0]
        return c

    def route_stats(r):
        L = 0
        for u in r:
            if u <= n:
                L += dd[u]
        loads = [L]
        cur = L
        for u in r:
            if u <= n:
                cur -= dd[u]
            else:
                cur += pp[u - n]
            loads.append(cur)
        m = len(loads)
        maxpref = [0] * m
        mp = -1
        for k in range(m):
            if loads[k] > mp:
                mp = loads[k]
            maxpref[k] = mp
        maxsuf = [0] * m
        ms = -1
        for k in range(m - 1, -1, -1):
            if loads[k] > ms:
                ms = loads[k]
            maxsuf[k] = ms
        return maxpref, maxsuf

    def feasible(r):
        L = 0
        for u in r:
            if u <= n:
                L += dd[u]
        if L > C:
            return False
        for u in r:
            if u <= n:
                L -= dd[u]
            else:
                L += pp[u - n]
                if L > C:
                    return False
        return True

    def repair(routes, pending, noise_amp):
        caches = [route_stats(r) for r in routes]
        order = pending[:]
        rng.shuffle(order)
        for u in order:
            amt = node_amt(u)
            isdel = u <= n
            lu = loc[u]
            bestv = float("inf")
            bestri = -1
            bestpos = -1
            for ri in range(len(routes)):
                r = routes[ri]
                mp, ms = caches[ri]
                rl = len(r)
                prevloc = 0
                for pos in range(rl + 1):
                    nxtloc = loc[r[pos]] if pos < rl else 0
                    if isdel:
                        if mp[pos] + amt > C:
                            break  # maxpref nondecreasing
                    else:
                        if ms[pos] + amt > C:
                            prevloc = nxtloc
                            continue
                    delta = D[prevloc][lu] + D[lu][nxtloc] - D[prevloc][nxtloc]
                    v = delta
                    if noise_amp:
                        v += noise_amp * (rng.random() - 0.5)
                    if v < bestv:
                        bestv = v
                        bestri = ri
                        bestpos = pos
                    prevloc = nxtloc
            # new route option
            vnew = 2 * D[0][lu]
            if noise_amp:
                vnew += noise_amp * (rng.random() - 0.5)
            if bestri == -1 or vnew < bestv:
                routes.append([u])
                caches.append(route_stats([u]))
            else:
                routes[bestri].insert(bestpos, u)
                caches[bestri] = route_stats(routes[bestri])

    def two_opt(r):
        m = len(r)
        if m < 3:
            return r
        passes = 0
        improved = True
        while improved and passes < 6:
            passes += 1
            improved = False
            p = [0] + [loc[u] for u in r] + [0]
            a = 0
            while a < m - 1:
                b = a + 1
                while b < m:
                    delta = (D[p[a]][p[b + 1]] + D[p[a + 1]][p[b + 2]]
                             - D[p[a]][p[a + 1]] - D[p[b + 1]][p[b + 2]])
                    if delta < 0:
                        nr = r[:a] + r[a:b + 1][::-1] + r[b + 1:]
                        if feasible(nr):
                            r = nr
                            p = [0] + [loc[u] for u in r] + [0]
                            improved = True
                    b += 1
                a += 1
        return r

    def build_solution(routes, obj):
        routes_out = []
        detailed = []
        for r in routes:
            if not r:
                continue
            seq = [0] + list(r) + [0]
            routes_out.append(seq)
            det = []
            for u in seq:
                if u == 0:
                    det.append({"node": 0, "role": "depot",
                                "customer": None, "quantity": 0})
                elif u <= n:
                    det.append({"node": u, "role": "linehaul",
                                "customer": u, "quantity": dd[u]})
                else:
                    det.append({"node": u, "role": "backhaul",
                                "customer": u - n, "quantity": pp[u - n]})
            detailed.append(det)
        return {"objective_value": float(obj),
                "routes": routes_out,
                "routes_detailed": detailed}

    def write_solution(sol):
        with open(args.solution_path, "w") as f:
            json.dump(sol, f)

    if n == 0:
        sol = {"objective_value": 0.0, "routes": [], "routes_detailed": []}
        if logger:
            logger.log_solution(0.0, sol)
        write_solution(sol)
        return

    # ---------- initial solution ----------
    order = sorted(nodes, key=lambda u: -D[0][loc[u]])
    init_routes = []
    repair(init_routes, order, 0)
    init_routes = [two_opt(r) for r in init_routes if r]
    cur_routes = [r[:] for r in init_routes]
    cur_cost = total_cost(cur_routes)
    best_routes = [r[:] for r in cur_routes]
    best_cost = cur_cost
    best_sol = build_solution(best_routes, best_cost)
    if logger:
        logger.log_solution(float(best_cost), best_sol)
    write_solution(best_sol)

    # average distance scale for noise
    avg_d = 0.0
    cnt = 0
    step = max(1, n // 30)
    for i in range(0, n + 1, step):
        for j in range(0, n + 1, step):
            avg_d += D[i][j]
            cnt += 1
    avg_d = avg_d / max(1, cnt)

    # ---------- LNS with simulated annealing ----------
    T0 = max(1.0, 0.04 * cur_cost)
    Tend = max(0.01, 0.0004 * cur_cost)
    total_time = max(1e-6, deadline - time.time())
    t_anneal_start = time.time()

    qmax = min(60, max(4, int(0.20 * N)))
    it = 0
    since_best = 0

    while time.time() < deadline:
        it += 1
        since_best += 1

        q = rng.randint(3, qmax) if qmax > 3 else min(3, N)
        mode = rng.random()

        if mode < 0.40:
            # random removal
            S = set(rng.sample(nodes, min(q, N)))
        elif mode < 0.80:
            # Shaw (relatedness) removal
            S = set()
            start = rng.choice(nodes)
            S.add(start)
            rem = [u for u in nodes if u != start]
            while len(S) < q and rem:
                ref = rng.choice(tuple(S))
                k = min(len(rem), 40)
                cands = rng.sample(rem, k)
                lref = loc[ref]
                cands.sort(key=lambda v: D[lref][loc[v]])
                idx = int((rng.random() ** 3) * len(cands))
                if idx >= len(cands):
                    idx = len(cands) - 1
                v = cands[idx]
                S.add(v)
                rem.remove(v)
        else:
            # route / segment removal
            ri = rng.randrange(len(cur_routes))
            r = cur_routes[ri]
            if len(r) <= q:
                S = set(r)
            else:
                st = rng.randrange(len(r) - q + 1)
                S = set(r[st:st + q])
            # possibly extend with random nodes
            need = q - len(S)
            if need > 0:
                extra = [u for u in nodes if u not in S]
                if extra:
                    S.update(rng.sample(extra, min(need, len(extra))))

        new_routes = []
        for r in cur_routes:
            nr = [u for u in r if u not in S]
            if nr:
                new_routes.append(nr)

        noise_amp = 0.15 * avg_d if rng.random() < 0.5 else 0.0
        repair(new_routes, list(S), noise_amp)
        new_cost = total_cost(new_routes)

        elapsed = time.time() - t_anneal_start
        frac = min(1.0, elapsed / total_time)
        T = T0 * ((Tend / T0) ** frac)

        accept = False
        if new_cost <= cur_cost:
            accept = True
        else:
            diff = new_cost - cur_cost
            if diff / max(T, 1e-9) < 40 and rng.random() < math.exp(-diff / max(T, 1e-9)):
                accept = True

        if accept:
            cur_routes = new_routes
            cur_cost = new_cost

            if cur_cost < best_cost:
                # polish with intra-route 2-opt
                polished = [two_opt(r) for r in cur_routes if r]
                pc = total_cost(polished)
                if pc < cur_cost:
                    cur_routes = polished
                    cur_cost = pc
                best_routes = [r[:] for r in cur_routes]
                best_cost = cur_cost
                since_best = 0
                best_sol = build_solution(best_routes, best_cost)
                if logger:
                    logger.log_solution(float(best_cost), best_sol)

        if since_best > 4000:
            cur_routes = [r[:] for r in best_routes]
            cur_cost = best_cost
            since_best = 0

    # final polish
    polished = [two_opt(r) for r in best_routes if r]
    pc = total_cost(polished)
    if pc < best_cost:
        best_routes = polished
        best_cost = pc
        best_sol = build_solution(best_routes, best_cost)
        if logger:
            logger.log_solution(float(best_cost), best_sol)
    else:
        best_sol = build_solution(best_routes, best_cost)

    write_solution(best_sol)


if __name__ == "__main__":
    main()
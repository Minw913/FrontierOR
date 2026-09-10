import argparse
import json
import math
import random
import time

try:
    from solution_logger import SolutionLogger
except Exception:
    SolutionLogger = None

EPS = 1e-6


class DARP:
    def __init__(self, inst):
        self.n = inst['num_users']
        self.m = inst['num_vehicles']
        self.Q = inst['vehicle_capacity']
        self.Lmax = inst['maximum_ride_time']
        self.Tmax = inst['maximum_route_duration']
        self.N = inst['num_nodes']
        nodes = sorted(inst['nodes'], key=lambda nd: nd['node_id'])
        N = self.N
        self.e = [0.0] * N
        self.l = [0.0] * N
        self.d = [0.0] * N
        self.q = [0] * N
        xs = [0.0] * N
        ys = [0.0] * N
        self.pair = [-1] * N
        self.user = [None] * N
        self.requests = []  # (user_id, pickup, dropoff)
        for nd in nodes:
            i = nd['node_id']
            self.e[i] = float(nd['earliest_time'])
            self.l[i] = float(nd['latest_time'])
            self.d[i] = float(nd['service_duration'])
            self.q[i] = int(nd['load'])
            xs[i] = float(nd['x'])
            ys[i] = float(nd['y'])
            if nd.get('node_type') == 'pickup':
                self.pair[i] = nd['paired_node']
                self.user[i] = nd['user_id']
                self.requests.append((nd['user_id'], i, nd['paired_node']))
            elif nd.get('node_type') == 'dropoff':
                self.pair[i] = nd['paired_node']
                self.user[i] = nd['user_id']
        self.end = N - 1
        # distance matrix
        t = [[0.0] * N for _ in range(N)]
        for i in range(N):
            xi, yi = xs[i], ys[i]
            ti = t[i]
            for j in range(i + 1, N):
                dd = math.hypot(xi - xs[j], yi - ys[j])
                ti[j] = dd
                t[j][i] = dd
        self.t = t

    def route_cost(self, seq):
        t = self.t
        c = 0.0
        prev = 0
        for v in seq:
            c += t[prev][v]
            prev = v
        c += t[prev][self.end]
        return c

    def total_cost(self, routes):
        return sum(self.route_cost(seq) for seq in routes)

    def eval_route(self, seq, check=True):
        """Eight-step evaluation.  Returns list B of service start times
        (including both depots) if feasible (or check=False), else None."""
        t = self.t
        e = self.e
        l = self.l
        d = self.d
        q = self.q
        route = [0] + seq + [self.end]
        L = len(route)
        if check:
            load = 0
            Q = self.Q
            for v in seq:
                load += q[v]
                if load > Q or load < 0:
                    return None
        A = [0.0] * L
        B = [0.0] * L
        W = [0.0] * L
        D = [0.0] * L
        B[0] = e[0]
        D[0] = e[0] + d[0]

        def forward(s):
            for i in range(s, L):
                ni = route[i]
                ai = D[i - 1] + t[route[i - 1]][ni]
                ei = e[ni]
                bi = ai if ai > ei else ei
                A[i] = ai
                B[i] = bi
                W[i] = bi - ai
                D[i] = bi + d[ni]

        forward(1)
        if check:
            # time-window upper bounds can never be repaired by delaying
            for i in range(1, L):
                if B[i] > l[route[i]] + EPS:
                    return None
        pos = {route[i]: i for i in range(L)}
        pair = self.pair
        Lmax = self.Lmax
        INF = float('inf')

        def fslack(i):
            cum = 0.0
            F = INF
            for j in range(i, L):
                if j > i:
                    cum += W[j]
                node = route[j]
                s = l[node] - B[j]
                if q[node] < 0:
                    pi = pos.get(pair[node], -1)
                    if 0 <= pi < i:
                        rs = Lmax - (B[j] - D[pi])
                        if rs < s:
                            s = rs
                if s < 0.0:
                    s = 0.0
                tot = cum + s
                if tot < F:
                    F = tot
                if F <= 0.0:
                    return 0.0
            return F

        F0 = fslack(0)
        sw = 0.0
        for p_ in range(1, L - 1):
            sw += W[p_]
        sh = F0 if F0 < sw else sw
        if sh > 1e-12:
            B[0] = e[0] + sh
            D[0] = B[0] + d[0]
            forward(1)

        Tmax = self.Tmax

        def violated():
            for i2 in range(L):
                if B[i2] > l[route[i2]] + EPS:
                    return True
            for i2 in range(1, L - 1):
                nd = route[i2]
                if q[nd] > 0:
                    j2 = pos[pair[nd]]
                    if B[j2] - D[i2] > Lmax + EPS:
                        return True
            if B[L - 1] - B[0] > Tmax + EPS:
                return True
            return False

        if violated():
            for i2 in range(1, L - 1):
                nd = route[i2]
                if q[nd] > 0:
                    F = fslack(i2)
                    sw = 0.0
                    for p_ in range(i2 + 1, L - 1):
                        sw += W[p_]
                    sh = F if F < sw else sw
                    if sh > 1e-12:
                        W[i2] += sh
                        B[i2] += sh
                        D[i2] = B[i2] + d[nd]
                        forward(i2 + 1)
            if check and violated():
                return None
        return B


def candidates_for_route(darp, seq, p, dr, noise, rng):
    t = darp.t
    Q = darp.Q
    q = darp.q
    ext = [0] + seq + [darp.end]
    n0 = len(seq)
    cl = [0] * (n0 + 1)
    for k in range(n0):
        cl[k + 1] = cl[k] + q[seq[k]]
    qp = q[p]
    out = []
    tp = t[p]
    td = t[dr]
    for i in range(n0 + 1):
        if cl[i] + qp > Q:
            continue
        a = ext[i]
        b = ext[i + 1]
        ta = t[a]
        rem = ta[b]
        # dropoff immediately after pickup
        delta = ta[p] + tp[dr] + td[b] - rem
        if noise > 0.0:
            out.append((delta * (1.0 + rng.random() * noise), delta, i, i))
        else:
            out.append((delta, delta, i, i))
        dp = ta[p] + tp[b] - rem
        maxl = cl[i]
        for j in range(i + 1, n0 + 1):
            if cl[j] > maxl:
                maxl = cl[j]
            if maxl + qp > Q:
                break
            c = ext[j]
            f = ext[j + 1]
            delta = dp + t[c][dr] + td[f] - t[c][f]
            if noise > 0.0:
                out.append((delta * (1.0 + rng.random() * noise), delta, i, j))
            else:
                out.append((delta, delta, i, j))
    return out


def best_insertion(darp, routes, p, dr, rng, noise=0.0):
    cands = []
    for v, seq in enumerate(routes):
        for key, delta, i, j in candidates_for_route(darp, seq, p, dr, noise, rng):
            cands.append((key, delta, v, i, j))
    cands.sort(key=lambda x: x[0])
    for key, delta, v, i, j in cands:
        seq = routes[v]
        newseq = seq[:i] + [p] + seq[i:j] + [dr] + seq[j:]
        if darp.eval_route(newseq) is not None:
            return v, newseq, delta
    return None


def construct(darp, rng, deadline):
    reqs = list(darp.requests)
    orders = []
    orders.append(sorted(reqs, key=lambda r: darp.e[r[1]]))
    orders.append(sorted(reqs, key=lambda r: darp.l[r[2]]))
    orders.append(sorted(reqs, key=lambda r: darp.l[r[1]] - darp.e[r[1]]))
    attempt = 0
    while time.time() < deadline and attempt < 60:
        if attempt < len(orders):
            order = orders[attempt]
        else:
            order = list(reqs)
            rng.shuffle(order)
        attempt += 1
        routes = [[] for _ in range(darp.m)]
        ok = True
        for uid, p, dr in order:
            res = best_insertion(darp, routes, p, dr, rng, 0.0)
            if res is None:
                ok = False
                break
            v, newseq, _ = res
            routes[v] = newseq
        if ok:
            return routes
    return None


def forced_solution(darp):
    routes = [[] for _ in range(darp.m)]
    for k, (uid, p, dr) in enumerate(darp.requests):
        v = k % darp.m
        routes[v] = routes[v] + [p, dr]
    return routes


def removal_saving(darp, seq, p, dr):
    t = darp.t
    ext = [0] + seq + [darp.end]
    ip = seq.index(p) + 1
    idd = seq.index(dr) + 1
    if idd == ip + 1:
        a = ext[ip - 1]
        b = ext[idd + 1]
        return t[a][p] + t[p][dr] + t[dr][b] - t[a][b]
    a = ext[ip - 1]
    b = ext[ip + 1]
    c = ext[idd - 1]
    f = ext[idd + 1]
    return t[a][p] + t[p][b] - t[a][b] + t[c][dr] + t[dr][f] - t[c][f]


def do_removal(darp, routes, k, rng, mode):
    served = []
    for v, seq in enumerate(routes):
        for node in seq:
            if darp.q[node] > 0:
                served.append((darp.user[node], node, darp.pair[node], v))
    if not served:
        return [list(r) for r in routes], []
    k = min(k, len(served))
    if mode == 'worst':
        scored = []
        for uid, p, dr, v in served:
            sv = removal_saving(darp, routes[v], p, dr)
            scored.append((sv * (0.75 + 0.5 * rng.random()), uid, p, dr, v))
        scored.sort(key=lambda x: -x[0])
        sel = [(uid, p, dr) for _, uid, p, dr, _ in scored[:k]]
    elif mode == 'shaw':
        seed_ = rng.choice(served)
        t = darp.t
        sp = seed_[1]
        rel = []
        for uid, p, dr, v in served:
            r = t[sp][p] + t[seed_[2]][dr] + 0.2 * abs(darp.e[sp] - darp.e[p])
            rel.append((r * (0.8 + 0.4 * rng.random()), uid, p, dr))
        rel.sort(key=lambda x: x[0])
        sel = [(uid, p, dr) for _, uid, p, dr in rel[:k]]
    else:
        picks = rng.sample(served, k)
        sel = [(uid, p, dr) for uid, p, dr, _ in picks]
    rem_nodes = set()
    for uid, p, dr in sel:
        rem_nodes.add(p)
        rem_nodes.add(dr)
    new = [[x for x in seq if x not in rem_nodes] for seq in routes]
    return new, sel


def build_solution(darp, routes):
    obj = 0.0
    routes_out = {}
    service = {}
    rides = {}
    for v in range(darp.m):
        seq = routes[v]
        full = [0] + seq + [darp.end]
        routes_out[str(v)] = full
        obj += darp.route_cost(seq)
        B = darp.eval_route(seq, check=False)
        if B is None:
            B = [darp.e[0]] * (len(seq) + 2)
        service['depot_start_%d' % v] = float(B[0])
        service['depot_end_%d' % v] = float(B[-1])
        pos = {}
        for idx, node in enumerate(seq):
            service[str(node)] = float(B[idx + 1])
            pos[node] = idx + 1
        for node in seq:
            if darp.q[node] > 0:
                j = pos[darp.pair[node]]
                rt = B[j] - (B[pos[node]] + darp.d[node])
                rides[str(darp.user[node])] = float(rt)
    return {
        'objective_value': obj,
        'routes': routes_out,
        'service_times': service,
        'ride_times': rides,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, default=60)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    start = time.time()
    deadline = start + max(2.0, args.time_limit - 1.5)

    logger = None
    if args.log_path and SolutionLogger is not None:
        logger = SolutionLogger(args.log_path, sense="minimize")

    with open(args.instance_path) as f:
        inst = json.load(f)
    darp = DARP(inst)
    rng = random.Random(0)

    n = darp.n
    if n == 0:
        routes = [[] for _ in range(darp.m)]
        sol = build_solution(darp, routes)
        if logger:
            logger.log_solution(sol['objective_value'], sol)
        with open(args.solution_path, 'w') as f:
            json.dump(sol, f)
        return

    routes = construct(darp, rng, deadline)
    forced = False
    if routes is None:
        routes = forced_solution(darp)
        forced = True

    cur = [list(r) for r in routes]
    cur_cost = darp.total_cost(cur)
    best = [list(r) for r in cur]
    best_cost = cur_cost
    if logger and not forced:
        logger.log_solution(best_cost, build_solution(darp, best))

    if not forced:
        # LNS with simulated annealing acceptance
        T0 = max(1e-3, 0.04 * cur_cost)
        Tend = max(1e-6, 0.0005 * cur_cost)
        total_time = max(1.0, deadline - time.time())
        t_start = time.time()
        kmax = max(2, min(10, max(3, n // 4)))
        it = 0
        last_improve = time.time()
        while True:
            now = time.time()
            if now >= deadline:
                break
            it += 1
            frac = min(1.0, (now - t_start) / total_time)
            T = T0 * ((Tend / T0) ** frac)
            r = rng.random()
            if r < 0.45:
                mode = 'random'
            elif r < 0.75:
                mode = 'worst'
            else:
                mode = 'shaw'
            k = rng.randint(2, kmax)
            newroutes, removed = do_removal(darp, cur, k, rng, mode)
            if not removed:
                break
            rng.shuffle(removed)
            noise = 0.15 if rng.random() < 0.5 else 0.0
            fail = False
            for uid, p, dr in removed:
                res = best_insertion(darp, newroutes, p, dr, rng, noise)
                if res is None:
                    fail = True
                    break
                v, newseq, _ = res
                newroutes[v] = newseq
            if fail:
                continue
            ncost = darp.total_cost(newroutes)
            accept = False
            if ncost < cur_cost - 1e-9:
                accept = True
            else:
                dlt = ncost - cur_cost
                try:
                    if rng.random() < math.exp(-dlt / max(T, 1e-9)):
                        accept = True
                except OverflowError:
                    accept = False
            if accept:
                cur = newroutes
                cur_cost = ncost
                if ncost < best_cost - 1e-9:
                    best = [list(rr) for rr in newroutes]
                    best_cost = ncost
                    last_improve = time.time()
                    if logger:
                        logger.log_solution(best_cost, build_solution(darp, best))
            # restart from best on stagnation
            if time.time() - last_improve > 20.0:
                cur = [list(rr) for rr in best]
                cur_cost = best_cost
                last_improve = time.time()

    sol = build_solution(darp, best)
    if logger and forced:
        logger.log_solution(sol['objective_value'], sol)
    with open(args.solution_path, 'w') as f:
        json.dump(sol, f)


if __name__ == '__main__':
    main()
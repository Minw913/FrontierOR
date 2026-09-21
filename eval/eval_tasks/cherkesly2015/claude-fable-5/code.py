import argparse
import json
import math
import random
import time

from solution_logger import SolutionLogger

INF = float('inf')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--instance_path', required=True)
    ap.add_argument('--solution_path', required=True)
    ap.add_argument('--time_limit', type=int, default=60)
    ap.add_argument('--log_path', default=None)
    args = ap.parse_args()

    t_start = time.time()
    deadline = t_start + max(2, args.time_limit) - 1.0

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    random.seed(0)

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    n = data['n_requests']
    N = data['n_nodes']
    Q = data['vehicle_capacity']
    END = N - 1

    nd = sorted(data['nodes'], key=lambda z: z['node_id'])
    ex = [z['x'] for z in nd]
    ey = [z['y'] for z in nd]
    qv = [z['load'] for z in nd]
    stv = [z['service_time'] for z in nd]
    ev = [z['tw_early'] for z in nd]
    lv = [z['tw_late'] for z in nd]

    C = [[INF] * N for _ in range(N)]
    TT = [[INF] * N for _ in range(N)]
    for a in data['arcs']:
        C[a['from']][a['to']] = float(a['cost'])
        TT[a['from']][a['to']] = float(a['travel_time'])

    # ---------------- Route representation ----------------
    class Route(object):
        __slots__ = ('seq', 'full', 'T', 'LT', 'LD', 'DP', 'cost')

        def __init__(self, seq):
            self.seq = seq
            self.recompute()

        def recompute(self):
            full = [0] + self.seq + [END]
            M = len(full)
            T = [0.0] * M
            LD = [0] * M
            DP = [0] * M
            t = float(ev[0])
            T[0] = t
            c = 0.0
            ok = True
            for k in range(1, M):
                u = full[k - 1]
                v = full[k]
                cv = C[u][v]
                if cv == INF:
                    ok = False
                    break
                c += cv
                t = t + stv[u] + TT[u][v]
                if t < ev[v]:
                    t = float(ev[v])
                if t > lv[v]:
                    ok = False
                    break
                T[k] = t
                LD[k] = LD[k - 1] + qv[v]
                if LD[k] > Q or LD[k] < 0:
                    ok = False
                    break
                if 1 <= v <= n:
                    DP[k] = DP[k - 1] + 1
                elif n < v < END:
                    DP[k] = DP[k - 1] - 1
                else:
                    DP[k] = DP[k - 1]
            LT = [0.0] * M
            LT[M - 1] = float(lv[END])
            for k in range(M - 2, -1, -1):
                u = full[k]
                v = full[k + 1]
                tv = TT[u][v]
                if tv == INF:
                    LT[k] = float(lv[u])
                else:
                    LT[k] = min(float(lv[u]), LT[k + 1] - stv[u] - tv)
            self.full = full
            self.T = T
            self.LT = LT
            self.LD = LD
            self.DP = DP
            self.cost = c
            return ok

    # ---------------- Insertion evaluation ----------------
    def eval_insert(rt, r, bound):
        """Best (delta,a,b) insertion of request r into route rt with delta < bound."""
        p = r
        d = r + n
        qr = qv[p]
        full = rt.full
        T = rt.T
        LT = rt.LT
        LD = rt.LD
        DP = rt.DP
        M = len(full)
        best = None
        Cp = C[p]
        for a in range(M - 1):
            u = full[a]
            cup = C[u][p]
            if cup == INF:
                continue
            tp = T[a] + stv[u] + TT[u][p]
            if tp < ev[p]:
                tp = float(ev[p])
            if tp > lv[p]:
                continue
            da = DP[a]
            mind = 10 ** 9
            maxload = LD[a]
            t_cur = tp
            prev = p
            for b in range(a, M - 1):
                if b > a:
                    v = full[b]
                    cpv = C[prev][v]
                    if cpv == INF:
                        break
                    t_cur = t_cur + stv[prev] + TT[prev][v]
                    if t_cur < ev[v]:
                        t_cur = float(ev[v])
                    if t_cur > lv[v]:
                        break
                    if DP[b] < mind:
                        mind = DP[b]
                    if LD[b] > maxload:
                        maxload = LD[b]
                    prev = v
                if maxload + qr > Q:
                    break
                if b > a and (DP[b] != da or mind < da):
                    continue
                cpd = C[prev][d]
                if cpd == INF:
                    continue
                td = t_cur + stv[prev] + TT[prev][d]
                if td < ev[d]:
                    td = float(ev[d])
                if td > lv[d]:
                    break  # travel times satisfy triangle inequality -> monotone
                w = full[b + 1]
                cdw = C[d][w]
                if cdw == INF:
                    continue
                tw_ = td + stv[d] + TT[d][w]
                if tw_ > LT[b + 1]:
                    continue
                if b == a:
                    if Cp[d] == INF:
                        continue
                    delta = cup + Cp[d] + cdw - C[u][full[a + 1]]
                else:
                    v1 = full[a + 1]
                    delta = (cup + Cp[v1] - C[u][v1]
                             + C[prev][d] + cdw - C[prev][full[b + 1]])
                if delta < bound - 1e-9:
                    best = (delta, a, b)
                    bound = delta
        return best

    def new_route_delta(r):
        p = r
        d = r + n
        if C[0][p] == INF or C[p][d] == INF or C[d][END] == INF:
            return None
        if qv[p] > Q:
            return None
        t = ev[0] + stv[0] + TT[0][p]
        if t < ev[p]:
            t = float(ev[p])
        if t > lv[p]:
            return None
        t = t + stv[p] + TT[p][d]
        if t < ev[d]:
            t = float(ev[d])
        if t > lv[d]:
            return None
        t = t + stv[d] + TT[d][END]
        if t > lv[END]:
            return None
        return C[0][p] + C[p][d] + C[d][END]

    def insert_request(routes, r, noise=0.0):
        p = r
        d = r + n
        best_score = INF
        best_true = INF
        best_choice = None
        for rt in routes:
            bound = INF if noise > 0 else best_true
            if noise > 0 and best_true < INF:
                bound = best_true * (1.0 + noise) + 1e-6
            res = eval_insert(rt, r, bound)
            if res is not None:
                delta, a, b = res
                score = delta * (1.0 + random.random() * noise) if noise > 0 else delta
                if score < best_score:
                    best_score = score
                    best_true = min(best_true, delta)
                    best_choice = (rt, a, b)
        nr = new_route_delta(r)
        if nr is not None:
            score = nr * (1.0 + random.random() * noise) if noise > 0 else nr
            if score < best_score:
                best_score = score
                best_choice = ('new',)
        if best_choice is None:
            return False
        if best_choice[0] == 'new':
            routes.append(Route([p, d]))
        else:
            rt, a, b = best_choice
            s = rt.seq
            rt.seq = s[:a] + [p] + s[a:b] + [d] + s[b:]
            rt.recompute()
        return True

    # ---------------- Removal operators ----------------
    def remove_requests(routes, removed):
        rem = set()
        for r in removed:
            rem.add(r)
            rem.add(r + n)
        newroutes = []
        for rt in routes:
            ns = [v for v in rt.seq if v not in rem]
            if len(ns) != len(rt.seq):
                if ns:
                    rt.seq = ns
                    rt.recompute()
                    newroutes.append(rt)
            else:
                newroutes.append(rt)
        routes[:] = newroutes

    def random_removal(routes, k):
        reqs = []
        for rt in routes:
            for v in rt.seq:
                if v <= n:
                    reqs.append(v)
        k = min(k, len(reqs))
        return random.sample(reqs, k) if k > 0 else []

    def removal_saving(rt, r):
        ns = [v for v in rt.seq if v != r and v != r + n]
        c = 0.0
        prev = 0
        for v in ns:
            c += C[prev][v]
            prev = v
        c += C[prev][END]
        return rt.cost - c

    def worst_removal(routes, k):
        savings = []
        for rt in routes:
            for v in rt.seq:
                if v <= n:
                    savings.append((removal_saving(rt, v), v))
        savings.sort(reverse=True)
        removed = []
        while len(removed) < k and savings:
            idx = int(len(savings) * (random.random() ** 3))
            removed.append(savings.pop(idx)[1])
        return removed

    def shaw_removal(routes, k):
        reqs = []
        for rt in routes:
            for v in rt.seq:
                if v <= n:
                    reqs.append(v)
        if not reqs:
            return []
        seed = random.choice(reqs)
        removed = [seed]
        pool = [r for r in reqs if r != seed]
        while len(removed) < k and pool:
            ref = random.choice(removed)
            pool.sort(key=lambda r: (math.hypot(ex[ref] - ex[r], ey[ref] - ey[r]) +
                                     math.hypot(ex[ref + n] - ex[r + n], ey[ref + n] - ey[r + n])))
            idx = int(len(pool) * (random.random() ** 4))
            removed.append(pool.pop(idx))
        return removed

    # ---------------- Solution build ----------------
    def build_solution(seqs):
        routes_out = []
        total = 0.0
        vid = 0
        for s in seqs:
            rt = Route(list(s))
            total += rt.cost
            details = []
            for k2, v in enumerate(rt.full):
                details.append({
                    "node_id": int(v),
                    "arrival_time": float(rt.T[k2]),
                    "load_after": float(rt.LD[k2]),
                })
            routes_out.append({
                "vehicle_id": vid,
                "route_sequence": [int(v) for v in rt.full],
                "route_details": details,
            })
            vid += 1
        return {"objective_value": float(total), "routes": routes_out}

    # ---------------- Initial construction ----------------
    routes = []
    order = sorted(range(1, n + 1), key=lambda r: (ev[r], lv[r]))
    for r in order:
        ok = insert_request(routes, r, noise=0.0)
        if not ok:
            # forced singleton route (best effort)
            routes.append(Route([r, r + n]))

    cur_cost = sum(rt.cost for rt in routes)
    best_seqs = [list(rt.seq) for rt in routes]
    best_cost = cur_cost
    if logger:
        logger.log_solution(best_cost, build_solution(best_seqs))

    # ---------------- LNS / SA loop ----------------
    if n > 0:
        T0 = max(1.0, 0.03 * best_cost)
        Tend = max(1e-3, 1e-4 * best_cost)
        qmax = min(n, max(3, min(12, n // 5 + 2)))
        no_improve = 0
        total_time = max(1e-6, deadline - t_start)

        while time.time() < deadline:
            frac = min(1.0, (time.time() - t_start) / total_time)
            Temp = T0 * ((Tend / T0) ** frac)

            snapshot = [list(rt.seq) for rt in routes]
            snap_cost = cur_cost

            k = random.randint(2 if n >= 2 else 1, qmax)
            op = random.random()
            if op < 0.4:
                removed = random_removal(routes, k)
            elif op < 0.7:
                removed = shaw_removal(routes, k)
            else:
                removed = worst_removal(routes, k)

            if not removed:
                continue

            remove_requests(routes, removed)
            random.shuffle(removed)
            noise = 0.1 if random.random() < 0.5 else 0.0
            failed = False
            for r in removed:
                if not insert_request(routes, r, noise=noise):
                    failed = True
                    break

            if failed:
                routes = [Route(s) for s in snapshot]
                cur_cost = snap_cost
                continue

            new_cost = sum(rt.cost for rt in routes)
            accept = False
            if new_cost < cur_cost - 1e-9:
                accept = True
            else:
                dc = (new_cost - cur_cost) / max(Temp, 1e-12)
                if dc < 40 and random.random() < math.exp(-dc):
                    accept = True

            if accept:
                cur_cost = new_cost
                if new_cost < best_cost - 1e-6:
                    best_cost = new_cost
                    best_seqs = [list(rt.seq) for rt in routes]
                    no_improve = 0
                    if logger:
                        logger.log_solution(best_cost, build_solution(best_seqs))
                else:
                    no_improve += 1
            else:
                routes = [Route(s) for s in snapshot]
                cur_cost = snap_cost
                no_improve += 1

            if no_improve >= 2000:
                routes = [Route(list(s)) for s in best_seqs]
                cur_cost = best_cost
                no_improve = 0

    # ---------------- Output ----------------
    sol = build_solution(best_seqs)
    if logger:
        logger.log_solution(sol["objective_value"], sol)
    with open(args.solution_path, 'w') as f:
        json.dump(sol, f, indent=2)


if __name__ == '__main__':
    main()
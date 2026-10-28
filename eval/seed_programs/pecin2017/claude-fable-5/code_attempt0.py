import argparse
import json
import math
import random
import time


def main():
    t_start = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument('--instance_path', required=True)
    ap.add_argument('--solution_path', required=True)
    ap.add_argument('--time_limit', type=int, default=60)
    ap.add_argument('--log_path', default=None)
    args = ap.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    random.seed(0)

    with open(args.instance_path) as f:
        inst = json.load(f)

    pp = inst['problem_parameters']
    cap = pp['vehicle_capacity']
    horizon = float(pp['planning_horizon'])
    nodes = inst['nodes']
    n = len(nodes)

    xs = [0.0] * n
    ys = [0.0] * n
    demand = [0] * n
    serv = [0.0] * n
    tw_open = [0.0] * n
    tw_close = [0.0] * n
    dep = 0
    snk = n - 1
    customers = []
    for nd in nodes:
        i = nd['id']
        xs[i] = float(nd['x'])
        ys[i] = float(nd['y'])
        demand[i] = nd['demand']
        serv[i] = float(nd['service_time'])
        tw_open[i] = float(nd['time_window_open'])
        tw_close[i] = float(nd['time_window_close'])
        t = nd['type']
        if t == 'depot_source':
            dep = i
        elif t == 'depot_sink':
            snk = i
        else:
            customers.append(i)

    nc = len(customers)
    deadline = t_start + args.time_limit - 0.4
    INF = float('inf')

    # ---- lazy distance rows ----
    rows = [None] * n

    def Drow(i):
        r = rows[i]
        if r is None:
            xi = xs[i]
            yi = ys[i]
            r = [math.hypot(xi - xs[j], yi - ys[j]) for j in range(n)]
            rows[i] = r
        return r

    def write_solution(nodelists, obj):
        rts = [[dep] + nl + [snk] for nl in nodelists]
        sol = {"objective_value": obj, "routes": rts}
        with open(args.solution_path, 'w') as f:
            json.dump(sol, f)
        return sol

    if nc == 0:
        write_solution([], 0.0)
        if logger:
            logger.log_solution(0.0, {"objective_value": 0.0, "routes": []})
        return

    # ---- route object ----
    class Route:
        __slots__ = ('nodes', 'load', 'E', 'L', 'cost')

        def __init__(self, nl):
            self.nodes = nl
            route_update(self)

    def route_update(r):
        nl = r.nodes
        m = len(nl)
        E = [0.0] * (m + 2)
        L = [0.0] * (m + 2)
        t = 0.0
        prev = dep
        cost = 0.0
        load = 0
        for k in range(m):
            c = nl[k]
            dpc = Drow(prev)[c]
            cost += dpc
            arr = t + serv[prev] + dpc
            oc = tw_open[c]
            t = arr if arr > oc else oc
            E[k + 1] = t
            load += demand[c]
            prev = c
        dret = Drow(prev)[snk]
        cost += dret
        E[m + 1] = t + serv[prev] + dret
        lt = min(horizon, tw_close[snk])
        L[m + 1] = lt
        nxt = snk
        for k in range(m - 1, -1, -1):
            c = nl[k]
            v = lt - Drow(c)[nxt] - serv[c]
            cc = tw_close[c]
            lt = cc if cc < v else v
            L[k + 1] = lt
            nxt = c
        L[0] = lt - Drow(dep)[nxt]
        r.E = E
        r.L = L
        r.cost = cost
        r.load = load

    def singleton_feasible(u):
        du = Drow(u)
        d0 = du[dep]
        st = d0 if d0 > tw_open[u] else tw_open[u]
        return st <= tw_close[u] and st + serv[u] + du[snk] <= horizon

    def best_insertion_full(u, routes):
        du = Drow(u)
        ddu = demand[u]
        ou = tw_open[u]
        cu = tw_close[u]
        su = serv[u]
        b1 = INF
        b2 = INF
        bri = -2
        bpos = 0
        d0 = du[dep]
        st = d0 if d0 > ou else ou
        if st <= cu and st + su + du[snk] <= horizon:
            b1 = d0 + du[snk]
            bri = -1
            bpos = 0
        for ri in range(len(routes)):
            r = routes[ri]
            if r.load + ddu > cap:
                continue
            nl = r.nodes
            E = r.E
            L = r.L
            m = len(nl)
            a = dep
            for pos in range(m + 1):
                b = nl[pos] if pos < m else snk
                dau = du[a]
                dub = du[b]
                delta = dau + dub - Drow(a)[b]
                if delta < b2:
                    s = E[pos] + serv[a] + dau
                    if s < ou:
                        s = ou
                    if s <= cu:
                        arrb = s + su + dub
                        ob = tw_open[b]
                        sb = arrb if arrb > ob else ob
                        if sb <= L[pos + 1]:
                            if delta < b1:
                                b2 = b1
                                b1 = delta
                                bri = ri
                                bpos = pos
                            elif delta < b2:
                                b2 = delta
                a = b
        return b1, bri, bpos, b2

    def do_insert(u, ri, pos, routes):
        if ri == -1 or ri == -2:
            routes.append(Route([u]))
        else:
            r = routes[ri]
            r.nodes.insert(pos, u)
            route_update(r)

    def greedy_insertion(pool, routes):
        random.shuffle(pool)
        for u in pool:
            b1, ri, pos, _ = best_insertion_full(u, routes)
            do_insert(u, ri, pos, routes)

    def regret_insertion(pool, routes):
        pool = list(pool)
        while pool:
            best_u = None
            best_key = None
            best_ins = None
            for u in pool:
                b1, ri, pos, b2 = best_insertion_full(u, routes)
                if b2 == INF:
                    regret = 1e18 - b1
                else:
                    regret = b2 - b1
                key = (-regret, b1)
                if best_key is None or key < best_key:
                    best_key = key
                    best_u = u
                    best_ins = (ri, pos)
            pool.remove(best_u)
            do_insert(best_u, best_ins[0], best_ins[1], routes)

    def remove_customers(routes, S):
        Sset = set(S)
        newroutes = []
        for r in routes:
            hit = False
            for c in r.nodes:
                if c in Sset:
                    hit = True
                    break
            if hit:
                r.nodes = [c for c in r.nodes if c not in Sset]
                if r.nodes:
                    route_update(r)
                    newroutes.append(r)
            else:
                newroutes.append(r)
        return newroutes

    def shaw_removal(routes, q):
        seed = random.choice(customers)
        Ds = Drow(seed)
        lst = sorted((c for c in customers if c != seed), key=lambda c: Ds[c])
        S = [seed]
        while lst and len(S) < q:
            idx = int((random.random() ** 5) * len(lst))
            S.append(lst.pop(idx))
        return S

    def worst_removal(routes, q):
        gains = []
        for r in routes:
            nl = r.nodes
            a = dep
            m = len(nl)
            for k in range(m):
                c = nl[k]
                b = nl[k + 1] if k + 1 < m else snk
                g = Drow(a)[c] + Drow(c)[b] - Drow(a)[b]
                gains.append((g, c))
                a = c
        gains.sort(reverse=True)
        S = []
        while gains and len(S) < q:
            idx = int((random.random() ** 3) * len(gains))
            S.append(gains.pop(idx)[1])
        return S

    def snapshot(routes):
        return [list(r.nodes) for r in routes]

    def total_cost(routes):
        return sum(r.cost for r in routes)

    # ---- initial construction: sequential best insertion ----
    order = sorted(customers, key=lambda c: (tw_close[c], -Drow(dep)[c]))
    routes = []
    for u in order:
        b1, ri, pos, _ = best_insertion_full(u, routes)
        do_insert(u, ri, pos, routes)

    cur_cost = total_cost(routes)
    best_cost = cur_cost
    best_nl = snapshot(routes)
    if logger:
        logger.log_solution(best_cost, {"objective_value": best_cost,
                                        "routes": [[dep] + nl + [snk] for nl in best_nl]})
    write_solution(best_nl, best_cost)

    # ---- LNS with simulated annealing acceptance ----
    T0 = max(1e-9, 0.01 * cur_cost)
    Tend = max(1e-12, T0 * 1e-4)
    total_time = max(1e-9, deadline - time.time())
    loop_start = time.time()

    qlow = 1 if nc < 4 else 4
    qhigh = max(qlow, min(45, nc // 3 if nc >= 12 else nc))
    last_write = time.time()

    while time.time() < deadline:
        frac = min(1.0, (time.time() - loop_start) / total_time)
        T = T0 * ((Tend / T0) ** frac)

        snap = snapshot(routes)
        q = random.randint(qlow, qhigh)
        q = min(q, nc)

        rr = random.random()
        if rr < 0.4:
            S = shaw_removal(routes, q)
        elif rr < 0.75:
            S = random.sample(customers, q)
        else:
            S = worst_removal(routes, q)

        routes = remove_customers(routes, S)

        use_regret = (random.random() < 0.4) and (q * max(1, nc) <= 30000)
        if use_regret:
            regret_insertion(S, routes)
        else:
            greedy_insertion(S, routes)

        new_cost = total_cost(routes)
        accept = False
        if new_cost < cur_cost - 1e-9:
            accept = True
        else:
            d = new_cost - cur_cost
            if T > 0 and random.random() < math.exp(-d / T):
                accept = True

        if accept:
            cur_cost = new_cost
            if new_cost < best_cost - 1e-6:
                best_cost = new_cost
                best_nl = snapshot(routes)
                if logger:
                    logger.log_solution(best_cost, {"objective_value": best_cost,
                                                    "routes": [[dep] + nl + [snk] for nl in best_nl]})
                if time.time() - last_write > 2.0:
                    write_solution(best_nl, best_cost)
                    last_write = time.time()
        else:
            routes = [Route(nl) for nl in snap]

    # ---- final output ----
    # recompute exact objective from best node lists
    obj = 0.0
    for nl in best_nl:
        prev = dep
        for c in nl:
            obj += Drow(prev)[c]
            prev = c
        obj += Drow(prev)[snk]
    write_solution(best_nl, obj)


if __name__ == '__main__':
    main()
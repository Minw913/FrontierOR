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
    random.seed(0)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path) as f:
        inst = json.load(f)

    n = inst['num_customers']
    Q = inst['vehicle_capacity']
    depot = inst['depot']
    D = inst['distance_matrix']
    customers = inst['customers']

    # ---------- build virtual nodes (split demands > Q) ----------
    orig = [0]
    dem = [0]
    E = [depot['time_window'][0]]
    L = [depot['time_window'][1]]
    S = [depot.get('service_time', 0)]
    for c in customers:
        d = c['demand']
        while d > Q:
            orig.append(c['id']); dem.append(Q)
            E.append(c['time_window'][0]); L.append(c['time_window'][1])
            S.append(c['service_time'])
            d -= Q
        if d > 0:
            orig.append(c['id']); dem.append(d)
            E.append(c['time_window'][0]); L.append(c['time_window'][1])
            S.append(c['service_time'])
    m = len(orig) - 1  # number of virtual customer nodes

    # virtual distance matrix
    DV = [[D[orig[i]][orig[j]] for j in range(m + 1)] for i in range(m + 1)]
    E0 = E[0]
    EPS = 1e-9

    def feasible(r):
        t = E0
        prev = 0
        for v in r:
            arr = t + S[prev] + DV[prev][v]
            if arr > L[v] + EPS:
                return False
            t = arr if arr > E[v] else E[v]
            prev = v
        return t + S[prev] + DV[prev][0] <= L[0] + EPS

    def rcost(r):
        if not r:
            return 0.0
        c = DV[0][r[0]]
        for i in range(len(r) - 1):
            c += DV[r[i]][r[i + 1]]
        return c + DV[r[-1]][0]

    def total_cost(routes):
        return sum(rcost(r) for r in routes)

    def build_solution(routes):
        cost = 0.0
        routes_out = []
        totals = {c['id']: 0 for c in customers}
        vk = 0
        for r in routes:
            if not r:
                continue
            vk += 1
            cost += rcost(r)
            seq = [0] + [orig[v] for v in r] + [n + 1]
            dl = {}
            for v in r:
                key = str(orig[v])
                dl[key] = dl.get(key, 0) + dem[v]
                totals[orig[v]] += dem[v]
            routes_out.append({"vehicle": vk, "route": seq, "deliveries": dl})
        deliveries = {str(c['id']): {"demand": c['demand'],
                                     "total_delivered": float(totals[c['id']])}
                      for c in customers}
        return {"objective_value": cost,
                "num_vehicles_used": vk,
                "routes": routes_out,
                "deliveries": deliveries}

    if m == 0:
        sol = {"objective_value": 0.0, "num_vehicles_used": 0, "routes": [], "deliveries": {}}
        if logger:
            logger.log_solution(0.0, sol)
        with open(args.solution_path, 'w') as f:
            json.dump(sol, f, indent=2)
        return

    # ---------- construction: sequential cheapest insertion ----------
    unrouted = list(range(1, m + 1))
    routes = []
    while unrouted:
        seed = min(unrouted, key=lambda v: (L[v], -DV[0][v]))
        unrouted.remove(seed)
        route = [seed]
        load = dem[seed]
        origset = {orig[seed]}
        while True:
            best_delta = float('inf')
            best_v = None
            best_pos = None
            for v in unrouted:
                if load + dem[v] > Q or orig[v] in origset:
                    continue
                for pos in range(len(route) + 1):
                    a = route[pos - 1] if pos > 0 else 0
                    b = route[pos] if pos < len(route) else 0
                    delta = DV[a][v] + DV[v][b] - DV[a][b]
                    if delta < best_delta - EPS:
                        cand = route[:pos] + [v] + route[pos:]
                        if feasible(cand):
                            best_delta = delta
                            best_v = v
                            best_pos = pos
            if best_v is None:
                break
            route.insert(best_pos, best_v)
            load += dem[best_v]
            origset.add(orig[best_v])
            unrouted.remove(best_v)
        routes.append(route)

    # ---------- intra-route improvement ----------
    def improve_route(r):
        if len(r) < 2:
            return r
        improved = True
        while improved:
            improved = False
            base = rcost(r)
            Lr = len(r)
            # relocate within route
            for i in range(Lr):
                for j in range(Lr):
                    if i == j:
                        continue
                    cand = r[:i] + r[i + 1:]
                    cand.insert(j, r[i])
                    if rcost(cand) < base - 1e-7 and feasible(cand):
                        r = cand
                        base = rcost(r)
                        improved = True
            # 2-opt (segment reversal)
            Lr = len(r)
            for i in range(Lr - 1):
                for j in range(i + 1, Lr):
                    cand = r[:i] + r[i:j + 1][::-1] + r[j + 1:]
                    if rcost(cand) < base - 1e-7 and feasible(cand):
                        r = cand
                        base = rcost(r)
                        improved = True
        return r

    routes = [improve_route(r) for r in routes if r]
    cur_routes = [r[:] for r in routes]
    cur_cost = total_cost(cur_routes)
    best_routes = [r[:] for r in cur_routes]
    best_cost = cur_cost
    if logger:
        logger.log_solution(best_cost, build_solution(best_routes))

    # ---------- LNS with simulated-annealing acceptance ----------
    def cheapest_insert(routes_, loads_, origsets_, v):
        best_delta = float('inf')
        best_ri = None
        best_pos = None
        for ri, r in enumerate(routes_):
            if loads_[ri] + dem[v] > Q or orig[v] in origsets_[ri]:
                continue
            for pos in range(len(r) + 1):
                a = r[pos - 1] if pos > 0 else 0
                b = r[pos] if pos < len(r) else 0
                delta = DV[a][v] + DV[v][b] - DV[a][b]
                if delta < best_delta - EPS:
                    if feasible(r[:pos] + [v] + r[pos:]):
                        best_delta = delta
                        best_ri = ri
                        best_pos = pos
        nd = DV[0][v] + DV[v][0]
        if nd < best_delta - EPS and feasible([v]):
            return (nd, -1, 0)
        if best_ri is None:
            if feasible([v]):
                return (nd, -1, 0)
            return None
        return (best_delta, best_ri, best_pos)

    T0 = max(1.0, 0.02 * best_cost)
    T = T0
    iter_count = 0

    while time.time() < deadline:
        iter_count += 1
        new_routes = [r[:] for r in cur_routes]

        # --- destroy ---
        mode = random.random()
        removed = []
        if mode < 0.15 and len(new_routes) > 1:
            # remove a whole route
            ri = random.randrange(len(new_routes))
            removed = new_routes.pop(ri)
        else:
            q = random.randint(2, max(2, min(14, max(2, m // 4))))
            all_nodes = [(ri, i) for ri, r in enumerate(new_routes) for i in range(len(r))]
            if len(all_nodes) <= q:
                continue
            if mode < 0.55:
                picks = random.sample(all_nodes, q)
            else:
                # related removal: pick a seed, remove nearest nodes
                sr, si = random.choice(all_nodes)
                seedv = new_routes[sr][si]
                all_nodes.sort(key=lambda p: DV[seedv][new_routes[p[0]][p[1]]])
                picks = all_nodes[:q]
            picks_by_route = {}
            for ri, i in picks:
                picks_by_route.setdefault(ri, []).append(i)
            for ri, idxs in picks_by_route.items():
                for i in sorted(idxs, reverse=True):
                    removed.append(new_routes[ri].pop(i))
            new_routes = [r for r in new_routes if r]

        if not removed:
            continue

        # --- repair ---
        random.shuffle(removed)
        loads_ = [sum(dem[v] for v in r) for r in new_routes]
        origsets_ = [set(orig[v] for v in r) for r in new_routes]
        ok = True
        for v in removed:
            res = cheapest_insert(new_routes, loads_, origsets_, v)
            if res is None:
                ok = False
                break
            _, ri, pos = res
            if ri == -1:
                new_routes.append([v])
                loads_.append(dem[v])
                origsets_.append({orig[v]})
            else:
                new_routes[ri].insert(pos, v)
                loads_[ri] += dem[v]
                origsets_[ri].add(orig[v])
        if not ok:
            continue

        new_cost = total_cost(new_routes)

        if new_cost < best_cost - 1e-6:
            new_routes = [improve_route(r) for r in new_routes if r]
            new_cost = total_cost(new_routes)
            if new_cost < best_cost - 1e-6:
                best_routes = [r[:] for r in new_routes]
                best_cost = new_cost
                if logger:
                    logger.log_solution(best_cost, build_solution(best_routes))

        # acceptance
        if new_cost < cur_cost - 1e-9:
            cur_routes = new_routes
            cur_cost = new_cost
        else:
            diff = new_cost - cur_cost
            if T > 1e-6 and random.random() < math.exp(-diff / T):
                cur_routes = new_routes
                cur_cost = new_cost

        T *= 0.9995
        if T < 1e-3 * T0:
            # restart from best
            T = T0 * 0.5
            cur_routes = [r[:] for r in best_routes]
            cur_cost = best_cost

    # ---------- final polish ----------
    best_routes = [improve_route(r) for r in best_routes if r]
    final_cost = total_cost(best_routes)
    if final_cost < best_cost - 1e-9:
        best_cost = final_cost
        if logger:
            logger.log_solution(best_cost, build_solution(best_routes))

    sol = build_solution(best_routes)
    with open(args.solution_path, 'w') as f:
        json.dump(sol, f, indent=2)


if __name__ == '__main__':
    main()
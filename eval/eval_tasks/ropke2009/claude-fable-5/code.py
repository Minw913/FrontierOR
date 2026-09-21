import json
import math
import random
import argparse
import time
import sys

from solution_logger import SolutionLogger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, default=60)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()
    deadline = start_time + max(1, args.time_limit) - 0.5

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    random.seed(0)

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    V = inst["num_vehicles"]
    cap = inst["vehicle_capacity"]
    nodes = sorted(inst["nodes"], key=lambda x: x["id"])
    N = len(nodes)

    X = [0.0] * N
    Y = [0.0] * N
    srv = [0.0] * N
    twa = [0.0] * N
    twb = [float("inf")] * N
    dem = [0] * N
    ntype = [""] * N
    for nd in nodes:
        i = nd["id"]
        X[i] = float(nd["x"])
        Y[i] = float(nd["y"])
        srv[i] = float(nd.get("service_time", 0) or 0)
        tw = nd.get("time_window", [0, inst.get("planning_horizon", 10 ** 9)])
        twa[i] = float(tw[0])
        twb[i] = float(tw[1])
        dem[i] = int(nd.get("demand", 0) or 0)
        ntype[i] = nd.get("type", "")

    O = None
    Dd = None
    for i in range(N):
        if ntype[i] == "depot_origin":
            O = i
        elif ntype[i] == "depot_destination":
            Dd = i
    if O is None:
        O = 0
    if Dd is None:
        Dd = O
    srv[O] = 0.0

    # requests
    reqs = []  # (pickup, delivery)
    for r in inst.get("requests", []):
        reqs.append((int(r["pickup_node"]), int(r["delivery_node"])))
    R = len(reqs)

    # distance matrix
    D = [[0.0] * N for _ in range(N)]
    for i in range(N):
        xi, yi = X[i], Y[i]
        Di = D[i]
        for j in range(i + 1, N):
            d = math.hypot(xi - X[j], yi - Y[j])
            Di[j] = d
            D[j][i] = d

    def feasible(inner):
        b = twa[O]
        load = 0
        prev = O
        Dp = D[prev]
        for v in inner:
            b += srv[prev] + Dp[v]
            if b < twa[v]:
                b = twa[v]
            if b > twb[v]:
                return False
            load += dem[v]
            if load > cap:
                return False
            prev = v
            Dp = D[prev]
        b += srv[prev] + Dp[Dd]
        return b <= twb[Dd]

    def rcost(inner):
        c = 0.0
        prev = O
        for v in inner:
            c += D[prev][v]
            prev = v
        return c + D[prev][Dd]

    def best_insertion(inner, p, d, bound=float("inf")):
        """Return (delta, i, j) of cheapest feasible insertion, or None."""
        n = len(inner)
        seq = [O] + inner + [Dd]
        best = None
        bestdelta = bound
        for i in range(n + 1):
            a = seq[i]
            b = seq[i + 1]
            dap = D[a][p]
            # j == i case
            delta0 = dap + D[p][d] + D[d][b] - D[a][b]
            if delta0 < bestdelta:
                cand = inner[:i] + [p, d] + inner[i:]
                if feasible(cand):
                    bestdelta = delta0
                    best = (i, i)
            addp = dap + D[p][b] - D[a][b]
            if addp >= bestdelta:
                # deltas for j>i are addp + non-negative (triangle ineq)
                continue
            for j in range(i + 1, n + 1):
                c = seq[j]
                e = seq[j + 1]
                delta = addp + D[c][d] + D[d][e] - D[c][e]
                if delta >= bestdelta:
                    continue
                cand = inner[:i] + [p] + inner[i:j] + [d] + inner[j:]
                if feasible(cand):
                    bestdelta = delta
                    best = (i, j)
        if best is None:
            return None
        return (bestdelta, best[0], best[1])

    def insert_req(inner, p, d, i, j):
        return inner[:i] + [p] + inner[i:j] + [d] + inner[j:]

    def remove_req(inner, p, d):
        return [v for v in inner if v != p and v != d]

    # ------------ repair ------------
    def repair(routes, unserved, mode):
        """Insert all requests in unserved (set of request indices). Returns True on success."""
        unserved = list(unserved)
        # table[rid] = list over routes: (delta,i,j) or None
        table = {}
        for rid in unserved:
            p, d = reqs[rid]
            table[rid] = [best_insertion(routes[k], p, d) for k in range(V)]
        while unserved:
            chosen = None
            chosen_route = None
            if mode == "greedy":
                bestval = float("inf")
                for rid in unserved:
                    row = table[rid]
                    for k in range(V):
                        e = row[k]
                        if e is not None and e[0] < bestval:
                            bestval = e[0]
                            chosen = rid
                            chosen_route = k
                if chosen is None:
                    return False
            else:  # regret-2
                bestreg = -float("inf")
                besttie = float("inf")
                for rid in unserved:
                    row = table[rid]
                    b1 = float("inf")
                    b2 = float("inf")
                    bk = -1
                    for k in range(V):
                        e = row[k]
                        if e is None:
                            continue
                        if e[0] < b1:
                            b2 = b1
                            b1 = e[0]
                            bk = k
                        elif e[0] < b2:
                            b2 = e[0]
                    if bk < 0:
                        return False
                    reg = (b2 - b1) if b2 < float("inf") else 1e12
                    if reg > bestreg or (reg == bestreg and b1 < besttie):
                        bestreg = reg
                        besttie = b1
                        chosen = rid
                        chosen_route = bk
                if chosen is None:
                    return False
            e = table[chosen][chosen_route]
            if e is None:
                return False
            p, d = reqs[chosen]
            routes[chosen_route] = insert_req(routes[chosen_route], p, d, e[1], e[2])
            unserved.remove(chosen)
            del table[chosen]
            # recompute affected column
            for rid in unserved:
                p2, d2 = reqs[rid]
                table[rid][chosen_route] = best_insertion(routes[chosen_route], p2, d2)
        return True

    # ------------ destroy operators ------------
    def route_of_request(routes):
        m = {}
        for k, r in enumerate(routes):
            for v in r:
                for rid, (p, d) in enumerate(reqs):
                    pass
        # faster: node -> request
        return m

    node_req = {}
    for rid, (p, d) in enumerate(reqs):
        node_req[p] = rid
        node_req[d] = rid

    def get_req_route_map(routes):
        m = {}
        for k, r in enumerate(routes):
            for v in r:
                m[node_req[v]] = k
        return m

    def random_removal(routes, q):
        served = list(get_req_route_map(routes).keys())
        random.shuffle(served)
        removed = served[:q]
        rs = set()
        for rid in removed:
            rs.add(rid)
        for k in range(V):
            routes[k] = [v for v in routes[k] if node_req[v] not in rs]
        return removed

    def worst_removal(routes, q):
        m = get_req_route_map(routes)
        gains = []
        rc = [rcost(r) for r in routes]
        for rid, k in m.items():
            p, d = reqs[rid]
            gains.append((rc[k] - rcost(remove_req(routes[k], p, d)), rid))
        gains.sort(reverse=True)
        removed = []
        while len(removed) < q and gains:
            idx = int((random.random() ** 3) * len(gains))
            removed.append(gains.pop(idx)[1])
        rs = set(removed)
        for k in range(V):
            routes[k] = [v for v in routes[k] if node_req[v] not in rs]
        return removed

    def shaw_removal(routes, q):
        served = list(get_req_route_map(routes).keys())
        if not served:
            return []
        seed = random.choice(served)
        removed = [seed]
        remset = {seed}
        while len(removed) < q:
            ref = random.choice(removed)
            pr, dr = reqs[ref]
            cands = []
            for rid in served:
                if rid in remset:
                    continue
                p, d = reqs[rid]
                rel = D[pr][p] + D[dr][d]
                cands.append((rel, rid))
            if not cands:
                break
            cands.sort()
            idx = int((random.random() ** 4) * len(cands))
            rid = cands[idx][1]
            removed.append(rid)
            remset.add(rid)
        for k in range(V):
            routes[k] = [v for v in routes[k] if node_req[v] not in remset]
        return removed

    # ------------ solution helpers ------------
    def total_cost(routes):
        return sum(rcost(r) for r in routes)

    def build_solution(routes, cost):
        out_routes = []
        for k in range(V):
            out_routes.append({"vehicle": k, "nodes": [O] + list(routes[k]) + [Dd]})
        return {"objective_value": float(cost), "routes": out_routes}

    # ------------ initial construction ------------
    routes = [[] for _ in range(V)]
    all_ids = list(range(R))
    ok = repair(routes, all_ids, "greedy")
    if not ok:
        routes = [[] for _ in range(V)]
        ok = repair(routes, all_ids, "regret")
    if not ok:
        # fallback: try single-request-per-vehicle greedy fill, then force
        routes = [[] for _ in range(V)]
        remaining = []
        order = sorted(range(R), key=lambda rid: twa[reqs[rid][0]])
        for rid in order:
            p, d = reqs[rid]
            best = None
            bestdelta = float("inf")
            for k in range(V):
                e = best_insertion(routes[k], p, d, bestdelta)
                if e is not None and e[0] < bestdelta:
                    bestdelta = e[0]
                    best = (k, e[1], e[2])
            if best is not None:
                k, i, j = best
                routes[k] = insert_req(routes[k], p, d, i, j)
            else:
                remaining.append(rid)
        for rid in remaining:
            p, d = reqs[rid]
            k = min(range(V), key=lambda kk: len(routes[kk]))
            routes[k] = routes[k] + [p, d]

    cur_cost = total_cost(routes)
    best_routes = [list(r) for r in routes]
    best_cost = cur_cost
    if logger:
        logger.log_solution(best_cost, build_solution(best_routes, best_cost))

    # ------------ LNS with simulated annealing ------------
    if R > 0:
        T0 = max(1e-6, 0.03 * cur_cost)
        Tend = max(1e-9, 0.0002 * cur_cost)
        destroy_ops = [random_removal, worst_removal, shaw_removal]
        repair_modes = ["greedy", "regret"]
        qmax = min(R, max(3, min(15, R // 2 if R >= 4 else R)))

        while time.time() < deadline:
            elapsed_frac = min(1.0, (time.time() - start_time) / max(1.0, (deadline - start_time)))
            T = T0 * ((Tend / T0) ** elapsed_frac)

            new_routes = [list(r) for r in routes]
            q = random.randint(2, qmax) if qmax >= 2 else 1
            op = random.choice(destroy_ops)
            removed = op(new_routes, q)
            if not removed:
                continue
            mode = random.choice(repair_modes)
            if not repair(new_routes, removed, mode):
                continue
            new_cost = total_cost(new_routes)
            accept = False
            if new_cost < cur_cost - 1e-9:
                accept = True
            else:
                diff = new_cost - cur_cost
                if T > 1e-12 and random.random() < math.exp(-diff / T):
                    accept = True
            if accept:
                routes = new_routes
                cur_cost = new_cost
                if cur_cost < best_cost - 1e-9:
                    best_cost = cur_cost
                    best_routes = [list(r) for r in routes]
                    if logger:
                        logger.log_solution(best_cost, build_solution(best_routes, best_cost))

    sol = build_solution(best_routes, best_cost)
    with open(args.solution_path, "w") as f:
        json.dump(sol, f, indent=2)


if __name__ == "__main__":
    main()
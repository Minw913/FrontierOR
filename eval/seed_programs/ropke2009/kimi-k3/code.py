import argparse
import json
import math
import random
import time

def solve(inst, time_limit, logger):
    random.seed(12345)
    start_clock = time.time()
    deadline = start_clock + max(0.05, float(time_limit) - 0.5)

    nodes = inst.get("nodes", [])
    if not nodes:
        return 0.0, []
    N = max(n["id"] for n in nodes) + 1

    X = [0.0] * N
    Y = [0.0] * N
    SRV = [0.0] * N
    DEM = [0.0] * N
    EW = [0.0] * N
    LW = [1e18] * N
    dep_o, dep_d = 0, N - 1
    for n in nodes:
        i = n["id"]
        X[i] = float(n.get("x", 0.0))
        Y[i] = float(n.get("y", 0.0))
        SRV[i] = float(n.get("service_time", 0.0))
        DEM[i] = float(n.get("demand", 0))
        tw = n.get("time_window", [0.0, 1e18])
        EW[i] = float(tw[0])
        LW[i] = float(tw[1])
        if n.get("type") == "depot_origin":
            dep_o = i
        elif n.get("type") == "depot_destination":
            dep_d = i

    requests = inst.get("requests", [])
    R = len(requests)
    PU = [r["pickup_node"] for r in requests]
    DE = [r["delivery_node"] for r in requests]
    LD = [float(r.get("load", 0)) for r in requests]
    REQ_OF_NODE = {}
    IS_PU_NODE = [False] * N
    for rid, r in enumerate(requests):
        REQ_OF_NODE[r["pickup_node"]] = rid
        REQ_OF_NODE[r["delivery_node"]] = rid
        IS_PU_NODE[r["pickup_node"]] = True
        IS_PU_NODE[r["delivery_node"]] = False

    CAP = float(inst.get("vehicle_capacity", 10**18))
    K = int(inst.get("num_vehicles", max(1, R)))
    K = max(1, K)
    DEPEARLY = EW[dep_o]
    RET_LATE = LW[dep_d]

    # distance / travel matrices (travel includes service time at origin)
    DIST = [[0.0] * N for _ in range(N)]
    for a in range(N):
        xa, ya = X[a], Y[a]
        rowa = DIST[a]
        for b in range(N):
            dx = xa - X[b]
            dy = ya - Y[b]
            rowa[b] = math.sqrt(dx * dx + dy * dy)
            _ = rowa[b]
    TRAV = [[0.0] * N for _ in range(N)]
    for a in range(N):
        sa = SRV[a]
        drow = DIST[a]
        trow = TRAV[a]
        for b in range(N):
            trow[b] = sa + drow[b]

    INF = float("inf")

    # ---------------- route evaluation ----------------
    def route_data(route):
        """returns (start_times, B_latest_arrival_bounds, prefix_loads, feasible, cost)"""
        L = len(route)
        start_times = []
        P = [0.0] * (L + 1)
        D = DIST
        T = TRAV
        load = 0.0
        cost = 0.0
        prev = dep_o
        tcur = DEPEARLY
        feasible = True
        for idx in range(L):
            node = route[idx]
            arr = tcur + T[prev][node]
            st = arr if arr > EW[node] else EW[node]
            if st > LW[node]:
                feasible = False
            start_times.append(st)
            load += DEM[node]
            if load > CAP or load < 0:
                feasible = False
            cost += D[prev][node]
            if cost < 0:
                pass
            P[idx + 1] = load
            tcur = st
            prev = node
        arrdep = tcur + T[prev][dep_d]
        if arrdep > RET_LATE:
            feasible = False
        cost += D[prev][dep_d]
        # backward latest-arrival bounds
        B = [0.0] * L
        nextBound = RET_LATE
        nextNode = dep_d
        for idx in range(L - 1, -1, -1):
            node = route[idx]
            b = LW[node]
            v = nextBound - T[node][nextNode]
            if v < b:
                b = v
            if b < EW[node]:
                feasible = False
            B[idx] = b
            nextBound = b
            nextNode = node
        return (start_times, B, P, feasible, cost)

    # ---------------- pair insertion (pickup before delivery) ----------------
    def best_insertion(route, data, rid):
        p = PU[rid]
        d = DE[rid]
        q = LD[rid]
        start, B, P, feas, oldcost = data
        L = len(route)
        D = DIST
        T = TRAV
        best_delta = INF
        best_i = -1
        best_j = -1
        early_p = EW[p]
        late_p = LW[p]
        early_d = EW[d]
        late_d = LW[d]
        for i in range(L + 1):
            if i == 0:
                prevNode = dep_o
                prevStart = DEPEARLY
                preLoad = 0.0
            else:
                prevNode = route[i - 1]
                prevStart = start[i - 1]
                preLoad = P[i]
            if preLoad + q > CAP:
                continue
            arrP = prevStart + T[prevNode][p]
            if arrP > late_p:
                continue
            pStart = arrP if arrP > early_p else early_p
            if pStart > late_p:
                continue
            curTime = pStart
            lastNode = p
            midLoad = preLoad + q
            # candidate with delivery directly after pickup (j == i)
            arrD = curTime + T[lastNode][d]
            dStart = arrD if arrD > early_d else early_d
            if dStart <= late_d:
                if i == L:
                    nxtNode = dep_d
                    bound = RET_LATE
                else:
                    nxtNode = route[i]
                    bound = B[i]
                arrNext = dStart + T[d][nxtNode]
                if arrNext <= bound:
                    delta = (
                        D[prevNode][p] + D[p][d] + D[d][nxtNode]
                        - D[prevNode][nxtNode]
                    )
                    if delta < best_delta:
                        best_delta = delta
                        best_i = i
                        best_j = i
            # scan delivery position further ahead
            for j in range(i + 1, L + 1):
                midNode = route[j - 1]
                arr = curTime + T[lastNode][midNode]
                st = arr if arr > EW[midNode] else EW[midNode]
                if st > LW[midNode]:
                    break  # arrival times only get worse downstream
                midLoad += DEM[midNode]
                if midLoad > CAP:
                    break  # this node will stay between pickup and delivery
                curTime = st
                lastNode = midNode
                arrD = curTime + T[lastNode][d]
                dStart = arrD if arrD > early_d else early_d
                if dStart > late_d:
                    break  # more intermediate nodes won't help
                if j == L:
                    nxtNode = dep_d
                    bound = RET_LATE
                else:
                    nxtNode = route[j]
                    bound = B[j]
                arrNext = dStart + T[d][nxtNode]
                if arrNext <= bound:
                    firstMid = route[i]
                    lastMid = route[j - 1]
                    delta = (
                        D[prevNode][p] + D[p][firstMid]
                        + D[lastMid][d] + D[d][nxtNode]
                        - D[prevNode][firstMid] - D[lastMid][nxtNode]
                    )
                    if delta < best_delta:
                        best_delta = delta
                        best_i = i
                        best_j = j
        if best_i < 0:
            return None
        return (best_delta, best_i, best_j)

    def apply_insert(route, rid, i, j):
        p = PU[rid]
        d = DE[rid]
        return route[:i] + [p] + route[i:j] + [d] + route[j:]

    def insert_best(routes, data, rid):
        best = None
        for ri in range(K):
            res = best_insertion(routes[ri], data[ri], rid)
            if res is not None and (best is None or res[0] < best[0]):
                best = (res[0], ri, res[1], res[2])
        if best is None:
            return False
        delta, ri, i, j = best
        routes[ri] = apply_insert(routes[ri], rid, i, j)
        data[ri] = route_data(routes[ri])
        return True

    # ---------------- insertion-based repair procedures ----------------
    def greedy_repair(routes, data, unassigned):
        while unassigned:
            best = None
            for rid in unassigned:
                for ri in range(K):
                    res = best_insertion(routes[ri], data[ri], rid)
                    if res is not None and (best is None or res[0] < best[0]):
                        best = (res[0], rid, ri, res[1], res[2])
            if best is None:
                return False
            delta, rid, ri, i, j = best
            routes[ri] = apply_insert(routes[ri], rid, i, j)
            data[ri] = route_data(routes[ri])
            unassigned.remove(rid)
        return True

    def regret_repair(routes, data, unassigned, reg=2):
        while unassigned:
            chosen = None
            chosen_regret = -INF
            chosen_detail = None
            for pos, rid in enumerate(unassigned):
                options = []
                for ri in range(K):
                    res = best_insertion(routes[ri], data[ri], rid)
                    if res is not None:
                        options.append((res[0], ri, res[1], res[2]))
                if not options:
                    return False
                options.sort(key=lambda x: x[0])
                if len(options) >= reg:
                    regret = options[reg - 1][0] - options[0][0]
                else:
                    regret = -options[0][0]
                if regret > chosen_regret:
                    chosen_regret = regret
                    chosen = pos
                    chosen_detail = options[0]
            delta, ri, i, j = chosen_detail
            rid = unassigned.pop(chosen)
            routes[ri] = apply_insert(routes[ri], rid, i, j)
            data[ri] = route_data(routes[ri])
        return True

    # ---------------- removal operators ----------------
    def assigned_requests(routes):
        seen = set()
        out = []
        for route in routes:
            for n in route:
                r = REQ_OF_NODE.get(n)
                if r is not None and r not in seen:
                    seen.add(r)
                    out.append(r)
        return out

    def remove_reqs(routes, removed):
        if removed:
            removed_nodes = set()
            for rid in removed:
                removed_nodes.add(PU[rid])
                removed_nodes.add(DE[rid])
            new_routes = []
            for route in routes:
                new_routes.append([n for n in route if n not in removed_nodes])
            return new_routes, list(removed)
        return [r[:] for r in routes], []

    def op_random(routes, k):
        assigned = assigned_requests(routes)
        if not assigned:
            return [r[:] for r in routes], []
        k = min(k, len(assigned))
        removed = random.sample(assigned, k)
        return remove_reqs(routes, removed)

    def op_worst(routes, k):
        gains = []
        for ri in range(K):
            route = routes[ri]
            if not route:
                continue
            data = route_data(route)
            old = data[4]
            req_in_route = []
            seen = set()
            for n in route:
                rid = REQ_OF_NODE[n]
                if rid not in seen:
                    seen.add(rid)
                    req_in_route.append(rid)
            for rid in req_in_route:
                pruned = [n for n in route if n != PU[rid] and n != DE[rid]]
                gain = old - route_data(pruned)[4]
                gains.append((gain + random.random() * 1e-6, rid))
        gains.sort(reverse=True)
        removed = [g[1] for g in gains[:k]]
        return remove_reqs(routes, removed)

    def op_shaw(routes, k):
        assigned = assigned_requests(routes)
        if len(assigned) <= 1:
            return [r[:] for r in routes], []
        base = random.choice(assigned)
        rel = []
        for rid in assigned:
            if rid == base:
                continue
            v = (
                DIST[PU[base]][PU[rid]] + DIST[DE[base]][DE[rid]]
                + abs(EW[PU[base]] - EW[PU[rid]])
                + abs(EW[DE[base]] - EW[DE[rid]])
            )
            rel.append((v + random.random() * 1e-6, rid))
        rel.sort()
        removed = [r for _, r in rel[:k]]
        return remove_reqs(routes, removed)

    destroy_ops = [op_random, op_worst, op_shaw]

    # ---------------- initial construction ----------------
    def build_initial():
        routes = [[] for _ in range(K)]
        data = [route_data(r) for r in routes]
        order = list(range(R))
        order.sort(key=lambda rid: (min(EW[PU[rid]], EW[DE[rid]]) + random.random() * 5.0))
        for rid in order:
            if not insert_best(routes, data, rid):
                return None, None
        return routes, data

    routes, data = build_initial()
    if routes is None:
        # fallback: try again with pure random order
        routes = [[] for _ in range(K)]
        data = [route_data(r) for r in routes]
        order = list(range(R))
        random.shuffle(order)
        ok = True
        for rid in order:
            if not insert_best(routes, data, rid):
                ok = False
                break
        if not ok:
            routes = [[] for _ in range(K)]
            data = [route_data(r) for r in routes]

    def total_cost(data):
        s = 0.0
        for d in data:
            s += d[4]
        return s

    def finalize(routes, obj):
        out = {
            "objective_value": obj,
            "routes": [
                {
                    "vehicle": i,
                    "nodes": [dep_o] + list(route) + [dep_d],
                }
                for i, route in enumerate(routes)
            ],
        }
        return out

    best_routes = [r[:] for r in routes]
    best_data = list(data)
    best_obj = total_cost(data)
    cur_routes = [r[:] for r in routes]
    cur_data = list(data)
    cur_obj = best_obj

    if logger:
        logger.log_solution(best_obj, finalize(best_routes, best_obj))

    # ---------------- relocation-based local improvement ----------------
    def improve(routes, data, budget_check):
        improved = True
        while improved and time.time() < deadline - budget_check:
            improved = False
            order = list(range(R))
            random.shuffle(order)
            for rid in order:
                if time.time() < deadline - budget_check:
                    pass
                else:
                    return routes, data
                # locate request
                ri_of = -1
                for ri in range(K):
                    rt = routes[ri]
                    for n in rt:
                        if n == PU[rid] or n == DE[rid]:
                            ri_of = ri
                            break
                    if ri_of >= 0:
                        break
                # occupancy map per node -> request
                ri_of = -1
            # re-derive location explicitly
            for rid in order:
                ri_of = -1
                for ri in range(K):
                    for n in routes[ri]:
                        rn = REQ_OF_NODE.get(n)
                        if rn == rid:
                            ri_of = ri
                            break
                    if ri_of >= 0:
                        break
                if ri_of < 0:
                    continue
                src_route = routes[ri_of]
                src_data = data[ri_of]
                old_cost = src_data[4]
                pruned = [n for n in src_route if n != PU[rid] and n != DE[rid]]
                pruned_data = route_data(pruned)
                gain = old_cost - pruned_data[4]
                best = None
                for ri in range(K):
                    if ri == ri_of:
                        cand_route = pruned
                        cand_data = pruned_data
                    else:
                        cand_route = routes[ri]
                        cand_data = data[ri]
                    res = best_insertion(cand_route, cand_data, rid)
                    if res is not None and (best is None or res[0] < best[0]):
                        best = (res[0], ri, res[1], res[2], ri == ri_of)
                if best is None:
                    continue
                delta, ri, i, j, same = best
                if gain - delta < 1e-7:
                    continue
                if ri == ri_of:
                    routes[ri] = apply_insert(pruned, rid, i, j)
                    data[ri] = route_data(routes[ri])
                else:
                    routes[ri_of] = pruned
                    data[ri_of] = route_data(pruned)
                    routes[ri] = apply_insert(routes[ri], rid, i, j)
                    data[ri] = route_data(routes[ri])
                improved = True
        return routes, data

    # ---------------- ALNS main loop ----------------
    temperature = max(1.0, best_obj * 0.03)
    cooling = 0.999
    it = 0
    while time.time() < deadline - 1.0:
        it += 1
        if it % 20 == 0 and time.time() > deadline - 1.0:
            break
        op = destroy_ops[it % len(destroy_ops)] if random.random() < 0.5 else random.choice(destroy_ops)
        k = max(1, int(random.random() * max(2, min(20, int(0.15 * max(R, 1))))) + 1)
        if op == op_worst and random.random() < 0.5:
            op = op_random if random.random() < 0.5 else op_shaw
        new_routes, removed = op(cur_routes, k)
        if not removed:
            continue
        new_data = [route_data(r) for r in new_routes]
        use_regret = random.random() < 0.5
        if use_regret:
            ok = regret_repair(new_routes, new_data, removed, reg=2 + (random.random() < 0.5))
        else:
            ok = greedy_repair(new_routes, new_data, removed)
        if not ok:
            continue
        new_obj = total_cost(new_data)
        diff = new_obj - cur_obj
        accept = False
        if diff < 0:
            accept = True
        elif temperature > 1e-9 and random.random() < math.exp(-diff / temperature):
            accept = True
        if accept:
            cur_routes = new_routes
            cur_data = new_data
            cur_obj = new_obj
            if new_obj < best_obj - 1e-9:
                best_obj = new_obj
                best_routes = [r[:] for r in new_routes]
                best_data = list(new_data)
                if logger:
                    logger.log_solution(best_obj, finalize(best_routes, best_obj))
        temperature *= cooling

    # final polish
    try:
        best_routes, best_data = improve([r[:] for r in best_routes], list(best_data), 0.15)
        best_obj = total_cost(best_data)
        if logger:
            logger.log_solution(best_obj, finalize(best_routes, best_obj))
    except Exception:
        pass

    return best_obj, best_routes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    obj, routes = solve(instance, args.time_limit, logger)

    solution = {
        "objective_value": obj,
        "routes": [
            {
                "vehicle": i,
                "nodes": build_route_nodes(instance, route),
            }
            for i, route in enumerate(routes)
        ],
    }
    with open(args.solution_path, "w") as f:
        json.dump(solution, f)


def build_route_nodes(instance, route):
    nodes = instance.get("nodes", [])
    dep_o, dep_d = 0, 0
    if nodes:
        ids = [n["id"] for n in nodes]
        dep_o = ids[0]
        dep_d = ids[-1]
        for n in nodes:
            if n.get("type") == "depot_origin":
                dep_o = n["id"]
            elif n.get("type") == "depot_destination":
                dep_d = n["id"]
    return [dep_o] + list(route) + [dep_d]


if __name__ == "__main__":
    main()
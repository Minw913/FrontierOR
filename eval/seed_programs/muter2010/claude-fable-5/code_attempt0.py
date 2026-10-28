import argparse
import json
import math
import random
import time

from solution_logger import SolutionLogger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, default=60)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()
    deadline = start_time + max(1, args.time_limit) - 0.35

    random.seed(0)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    depot = inst["depot"]
    customers = inst["customers"]
    n = len(customers)
    Q = inst["vehicle_capacity"]

    # node 0 = depot, nodes 1..n = customers
    xs = [depot["x"]] + [c["x"] for c in customers]
    ys = [depot["y"]] + [c["y"] for c in customers]
    dem = [0] + [c["demand"] for c in customers]
    ready = [depot["ready_time"]] + [c["ready_time"] for c in customers]
    due = [depot["due_date"]] + [c["due_date"] for c in customers]
    serv = [0] + [c["service_time"] for c in customers]
    ids = [depot["id"]] + [c["id"] for c in customers]
    ready0 = ready[0]
    due0 = due[0]

    N = n + 1
    D = [[0.0] * N for _ in range(N)]
    for i in range(N):
        Di = D[i]
        xi, yi = xs[i], ys[i]
        for j in range(i + 1, N):
            d = math.hypot(xi - xs[j], yi - ys[j])
            Di[j] = d
            D[j][i] = d

    maxD = max(max(row) for row in D) if N > 1 else 1.0
    if maxD <= 0:
        maxD = 1.0
    horizon = max(1.0, float(due0 - ready0))

    depot_id = ids[0]

    def route_cost(seq):
        c = D[0][seq[0]]
        prev = seq[0]
        for node in seq[1:]:
            c += D[prev][node]
            prev = node
        c += D[prev][0]
        return c

    def total_cost(routes):
        return sum(route_cost(s) for s in routes)

    def route_eval(seq):
        """Return (arr, L, load) if feasible else None."""
        t = ready0
        load = 0
        prev = 0
        arr = []
        for c in seq:
            t = t + D[prev][c]
            if t > due[c] + 1e-9:
                return None
            arr.append(t)
            t = max(t, ready[c]) + serv[c]
            load += dem[c]
            prev = c
        t += D[prev][0]
        if t > due0 + 1e-9 or load > Q:
            return None
        m = len(seq)
        L = [0.0] * m
        nxt = 0
        Ln = float(due0)
        for k in range(m - 1, -1, -1):
            c = seq[k]
            Ln = min(due[c], Ln - serv[c] - D[c][nxt])
            L[k] = Ln
            nxt = c
        return (arr, L, load)

    def build_solution_dict(routes, obj):
        out_routes = []
        for seq in routes:
            out_routes.append({
                "customer_ids": [ids[c] for c in seq],
                "path": [depot_id] + [ids[c] for c in seq] + [depot_id],
            })
        return {"objective_value": float(obj), "routes": out_routes}

    if n == 0:
        sol = {"objective_value": 0.0, "routes": []}
        if logger:
            logger.log_solution(0.0, sol)
        with open(args.solution_path, "w") as f:
            json.dump(sol, f, indent=2)
        return

    # ------------------- Initial solution: nearest-neighbor construction -------------------
    unrouted = set(range(1, n + 1))
    routes = []
    while unrouted:
        seq = []
        t = ready0
        load = 0
        prev = 0
        while True:
            best = None
            for c in unrouted:
                if load + dem[c] > Q:
                    continue
                a = t + D[prev][c]
                if a > due[c] + 1e-9:
                    continue
                startv = max(a, ready[c])
                fin = startv + serv[c]
                if fin + D[c][0] > due0 + 1e-9:
                    continue
                key = D[prev][c] + 0.2 * (startv - t)
                if best is None or key < best[0]:
                    best = (key, c, fin)
            if best is None:
                break
            _, c, fin = best
            seq.append(c)
            unrouted.discard(c)
            load += dem[c]
            t = fin
            prev = c
        if seq:
            routes.append(seq)
        else:
            # force a single-customer route (instance may be tight)
            c = next(iter(unrouted))
            unrouted.discard(c)
            routes.append([c])

    cur = [list(s) for s in routes]
    cur_obj = total_cost(cur)
    best_routes = [list(s) for s in cur]
    best_obj = cur_obj
    if logger:
        logger.log_solution(best_obj, build_solution_dict(best_routes, best_obj))

    # ------------------- Repair (greedy / regret-2 insertion) -------------------
    def repair(cand, datas, removed, use_regret):
        BIG = 1e18
        remaining = list(removed)
        rem_set = set(remaining)
        bp = {c: {} for c in remaining}

        def comp(c, r):
            seq = cand[r]
            dat = datas[r]
            if dat is None:
                bp[c][r] = None
                return
            arr, L, load = dat
            if load + dem[c] > Q:
                bp[c][r] = None
                return
            m = len(seq)
            bst = None
            duec = due[c]
            readyc = ready[c]
            servc = serv[c]
            Dc = D[c]
            for i in range(m + 1):
                if i > 0:
                    p = seq[i - 1]
                    dep_p = max(arr[i - 1], ready[p]) + serv[p]
                else:
                    p = 0
                    dep_p = ready0
                ac = dep_p + D[p][c]
                if ac > duec + 1e-9:
                    continue
                if i < m:
                    s = seq[i]
                    Ls = L[i]
                else:
                    s = 0
                    Ls = due0
                ns = max(ac, readyc) + servc + Dc[s]
                if ns > Ls + 1e-9:
                    continue
                d = D[p][c] + Dc[s] - D[p][s]
                if bst is None or d < bst[0]:
                    bst = (d, i)
            bp[c][r] = bst

        nr = len(cand)
        for c in remaining:
            for r in range(nr):
                comp(c, r)

        while rem_set:
            chosen = None
            if use_regret:
                best_key = None
                for c in rem_set:
                    cands = sorted((v[0], r, v[1]) for r, v in bp[c].items() if v is not None)
                    if not cands:
                        chosen = (c, None, None)
                        best_key = (BIG, -BIG)
                        break
                    if len(cands) == 1:
                        regret = BIG
                    else:
                        regret = cands[1][0] - cands[0][0]
                    key = (regret, -cands[0][0])
                    if best_key is None or key > best_key:
                        best_key = key
                        chosen = (c, cands[0][1], cands[0][2])
            else:
                best_key = None
                for c in rem_set:
                    bst = None
                    for r, v in bp[c].items():
                        if v is not None and (bst is None or v[0] < bst[0]):
                            bst = (v[0], r, v[1])
                    if bst is None:
                        chosen = (c, None, None)
                        best_key = None
                        break
                    if best_key is None or bst[0] < best_key:
                        best_key = bst[0]
                        chosen = (c, bst[1], bst[2])

            c, r, pos = chosen
            rem_set.discard(c)
            if r is None:
                # open a new route
                cand.append([c])
                datas.append(route_eval([c]))
                new_r = len(cand) - 1
                for c2 in rem_set:
                    comp(c2, new_r)
            else:
                cand[r].insert(pos, c)
                datas[r] = route_eval(cand[r])
                for c2 in rem_set:
                    comp(c2, r)

    # ------------------- Removal operators -------------------
    def random_removal(cand, q):
        allc = [c for seq in cand for c in seq]
        return random.sample(allc, min(q, len(allc)))

    def worst_removal(cand, q):
        gains = []
        for seq in cand:
            m = len(seq)
            for i in range(m):
                c = seq[i]
                p = seq[i - 1] if i > 0 else 0
                s = seq[i + 1] if i < m - 1 else 0
                g = D[p][c] + D[c][s] - D[p][s]
                gains.append((g, c))
        gains.sort(key=lambda t: -t[0])
        removed = []
        used = set()
        while len(removed) < q and gains:
            idx = int((random.random() ** 3) * len(gains))
            g, c = gains.pop(idx)
            if c not in used:
                used.add(c)
                removed.append(c)
        return removed

    def shaw_removal(cand, q):
        allc = [c for seq in cand for c in seq]
        seed = random.choice(allc)
        removed = [seed]
        rem_set = {seed}
        pool = [c for c in allc if c != seed]
        while len(removed) < q and pool:
            ref = random.choice(removed)
            pool.sort(key=lambda c: D[ref][c] / maxD +
                      abs(ready[ref] - ready[c]) / horizon)
            idx = int((random.random() ** 4) * len(pool))
            c = pool.pop(idx)
            removed.append(c)
            rem_set.add(c)
        return removed

    # ------------------- LNS main loop -------------------
    qmin = min(3, n)
    qmax = max(qmin, min(45, max(6, n // 5)))

    T0 = max(1e-6, 0.05 * cur_obj / math.log(2))
    Tend = max(1e-9, T0 * 0.002)
    no_improve = 0
    total_span = max(1e-6, deadline - start_time)

    while time.time() < deadline:
        progress = min(1.0, (time.time() - start_time) / total_span)
        T = T0 * ((Tend / T0) ** progress)

        cand = [list(s) for s in cur]
        q = random.randint(qmin, qmax)
        op = random.random()
        if op < 0.4:
            removed = random_removal(cand, q)
        elif op < 0.7:
            removed = shaw_removal(cand, q)
        else:
            removed = worst_removal(cand, q)

        rem_set = set(removed)
        cand = [[c for c in seq if c not in rem_set] for seq in cand]
        cand = [seq for seq in cand if seq]
        datas = [route_eval(seq) for seq in cand]

        use_regret = random.random() < 0.5
        repair(cand, datas, removed, use_regret)
        cand = [seq for seq in cand if seq]
        new_obj = total_cost(cand)

        accept = False
        if new_obj < cur_obj - 1e-9:
            accept = True
        else:
            try:
                if random.random() < math.exp((cur_obj - new_obj) / T):
                    accept = True
            except OverflowError:
                accept = False

        if accept:
            cur = cand
            cur_obj = new_obj

        if new_obj < best_obj - 1e-9:
            best_obj = new_obj
            best_routes = [list(s) for s in cand]
            no_improve = 0
            if logger:
                logger.log_solution(best_obj, build_solution_dict(best_routes, best_obj))
        else:
            no_improve += 1
            if no_improve >= 3000:
                cur = [list(s) for s in best_routes]
                cur_obj = best_obj
                no_improve = 0

    # ------------------- Output -------------------
    sol = build_solution_dict(best_routes, best_obj)
    with open(args.solution_path, "w") as f:
        json.dump(sol, f, indent=2)


if __name__ == "__main__":
    main()
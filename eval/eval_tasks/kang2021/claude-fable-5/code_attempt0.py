import argparse
import json
import math
import random
import time
from collections import defaultdict

from solution_logger import SolutionLogger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--instance_path', required=True)
    ap.add_argument('--solution_path', required=True)
    ap.add_argument('--time_limit', type=int, default=60)
    ap.add_argument('--log_path', default=None)
    args = ap.parse_args()

    start = time.time()
    random.seed(0)

    with open(args.instance_path) as f:
        inst = json.load(f)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    N = inst["num_nodes"]
    D = inst["parameters"]["num_drones"]
    caps = inst["parameters"]["drone_battery_capacities"]
    depot = inst["depot_index"]
    customers = list(inst["customer_indices"])
    tt = inst["truck_travel_times"]
    ddt = inst["drone_delivery_time"]
    dbc = inst["drone_battery_consumption"]

    deadline = start + max(3.0, args.time_limit * 0.97 - 0.5)

    # ---------- helpers ----------
    def route_cost(r):
        return sum(tt[r[i]][r[i + 1]] for i in range(len(r) - 1))

    def build_solution(rt, asg):
        arcs = [[rt[i], rt[i + 1]] for i in range(len(rt) - 1)]
        travel_ = sum(tt[a][b] for a, b in arcs)
        ndt_ = defaultdict(float)
        for c, (d, v) in asg.items():
            ndt_[(v, d)] += ddt[d][v][c]
        waits = {}
        for v in rt[:-1]:
            w = 0.0
            for d in range(D):
                if ndt_[(v, d)] > w:
                    w = ndt_[(v, d)]
            waits[str(v)] = float(w)
        obj = float(travel_ + sum(waits.values()))
        das = [{"drone": int(d), "dispatch_node": int(v), "delivery_node": int(c)}
               for c, (d, v) in sorted(asg.items())]
        return {"objective_value": obj,
                "truck_route": [[int(a), int(b)] for a, b in arcs],
                "drone_assignments": das,
                "waiting_times": waits}

    # ---------- initial TSP tour ----------
    unv = set(customers)
    route = [depot]
    cur = depot
    while unv:
        nxt = min(unv, key=lambda j: tt[cur][j])
        route.append(nxt)
        unv.remove(nxt)
        cur = nxt
    route.append(depot)

    sym = all(tt[i][j] == tt[j][i] for i in range(N) for j in range(i + 1, N))

    tsp_deadline = min(deadline, start + max(2.0, 0.12 * args.time_limit))
    if sym and len(route) > 4:
        improved = True
        while improved and time.time() < tsp_deadline:
            improved = False
            n = len(route)
            for i in range(1, n - 2):
                a = route[i - 1]
                ci = route[i]
                for j in range(i + 1, n - 1):
                    cj = route[j]
                    b = route[j + 1]
                    delta = tt[a][cj] + tt[ci][b] - tt[a][ci] - tt[cj][b]
                    if delta < -1e-9:
                        route[i:j + 1] = reversed(route[i:j + 1])
                        improved = True
                        ci = route[i]
                if time.time() > tsp_deadline:
                    break

    # ---------- state ----------
    assign = {}                      # c -> (d, v)
    ndt = defaultdict(float)         # (v,d) -> total delivery time
    disp_cnt = defaultdict(int)      # v -> number of deliveries dispatched
    batt = [0.0] * D
    wait = {v: 0.0 for v in route[:-1]}
    travel = route_cost(route)
    cost = travel

    def wait_of(v, mods=None):
        best = 0.0
        for d in range(D):
            x = ndt[(v, d)]
            if mods and d in mods:
                x += mods[d]
            if x > best:
                best = x
        return best

    def best_truck_to_drone(c, i):
        p = route[i - 1]
        nx = route[i + 1]
        dtr = tt[p][nx] - tt[p][c] - tt[c][nx]
        best = None
        for v in route[:-1]:
            if v == c:
                continue
            wv = wait[v]
            for d in range(D):
                cons = dbc[d][v][c]
                if batt[d] + cons > caps[d] + 1e-9:
                    continue
                dw = ndt[(v, d)] + ddt[d][v][c] - wv
                if dw < 0:
                    dw = 0.0
                delta = dtr + dw
                if best is None or delta < best[0]:
                    best = (delta, d, v)
        return best

    def apply_truck_to_drone(c, i, delta, d, v):
        nonlocal travel, cost
        p = route[i - 1]
        nx = route[i + 1]
        travel += tt[p][nx] - tt[p][c] - tt[c][nx]
        route.pop(i)
        wait.pop(c, None)
        assign[c] = (d, v)
        ndt[(v, d)] += ddt[d][v][c]
        if ndt[(v, d)] > wait[v]:
            wait[v] = ndt[(v, d)]
        disp_cnt[v] += 1
        batt[d] += dbc[d][v][c]
        cost += delta

    def best_drone_to_truck(c):
        d, v = assign[c]
        t = ddt[d][v][c]
        neww = wait_of(v, {d: -t})
        dw = neww - wait[v]
        besti = None
        bestins = None
        for i in range(len(route) - 1):
            a = route[i]
            b = route[i + 1]
            dins = tt[a][c] + tt[c][b] - tt[a][b]
            if bestins is None or dins < bestins:
                bestins = dins
                besti = i
        return (dw + bestins, besti, neww)

    def apply_drone_to_truck(c, i, neww, delta):
        nonlocal travel, cost
        d, v = assign.pop(c)
        ndt[(v, d)] -= ddt[d][v][c]
        batt[d] -= dbc[d][v][c]
        disp_cnt[v] -= 1
        wait[v] = neww
        a = route[i]
        b = route[i + 1]
        travel += tt[a][c] + tt[c][b] - tt[a][b]
        route.insert(i + 1, c)
        wait[c] = 0.0
        cost += delta

    def best_reassign(c):
        d, v = assign[c]
        t = ddt[d][v][c]
        cons = dbc[d][v][c]
        best = None
        for v2 in route[:-1]:
            for d2 in range(D):
                if v2 == v and d2 == d:
                    continue
                cons2 = dbc[d2][v2][c]
                nb = batt[d2] + cons2 - (cons if d2 == d else 0)
                if nb > caps[d2] + 1e-9:
                    continue
                t2 = ddt[d2][v2][c]
                if v2 == v:
                    mods = {d: -t}
                    mods[d2] = mods.get(d2, 0.0) + t2
                    dw = wait_of(v, mods) - wait[v]
                else:
                    dw = (wait_of(v, {d: -t}) - wait[v]) + \
                         (max(wait[v2], ndt[(v2, d2)] + t2) - wait[v2])
                if best is None or dw < best[0]:
                    best = (dw, d2, v2)
        return best

    def apply_reassign(c, delta, d2, v2):
        nonlocal cost
        d, v = assign[c]
        ndt[(v, d)] -= ddt[d][v][c]
        batt[d] -= dbc[d][v][c]
        disp_cnt[v] -= 1
        assign[c] = (d2, v2)
        ndt[(v2, d2)] += ddt[d2][v2][c]
        batt[d2] += dbc[d2][v2][c]
        disp_cnt[v2] += 1
        wait[v] = wait_of(v)
        wait[v2] = wait_of(v2)
        cost += delta

    def load_state(rt, asg):
        nonlocal travel, cost
        route[:] = rt
        assign.clear()
        assign.update(asg)
        ndt.clear()
        disp_cnt.clear()
        for d in range(D):
            batt[d] = 0.0
        wait.clear()
        for v in route[:-1]:
            wait[v] = 0.0
        for c, (d, v) in assign.items():
            ndt[(v, d)] += ddt[d][v][c]
            batt[d] += dbc[d][v][c]
            disp_cnt[v] += 1
        for v in route[:-1]:
            wait[v] = wait_of(v)
        travel = route_cost(route)
        cost = travel + sum(wait.values())

    # ---------- initial log ----------
    best_cost = cost
    best_route = route[:]
    best_assign = dict(assign)
    if logger:
        sol0 = build_solution(best_route, best_assign)
        logger.log_solution(sol0["objective_value"], sol0)

    # ---------- greedy truck -> drone ----------
    changed = True
    rounds = 0
    while changed and rounds < 8 and time.time() < deadline:
        changed = False
        rounds += 1
        order = route[1:-1][:]
        random.shuffle(order)
        for c in order:
            if time.time() > deadline:
                break
            if c in assign or disp_cnt[c] > 0:
                continue
            try:
                i = route.index(c)
            except ValueError:
                continue
            res = best_truck_to_drone(c, i)
            if res and res[0] < -1e-9:
                apply_truck_to_drone(c, i, *res)
                changed = True

    if cost < best_cost - 1e-9:
        best_cost = cost
        best_route = route[:]
        best_assign = dict(assign)
        if logger:
            sol0 = build_solution(best_route, best_assign)
            logger.log_solution(sol0["objective_value"], sol0)

    # ---------- simulated annealing ----------
    sa_start = time.time()
    horizon = max(1.0, deadline - sa_start)
    T0 = max(1.0, 0.03 * max(best_cost, 1.0))
    Tend = max(1e-3, 1e-4 * max(best_cost, 1.0))
    T = T0
    it = 0
    last_imp = sa_start
    stall_limit = max(5.0, 0.12 * horizon)

    def accept(delta):
        if delta <= 1e-12:
            return True
        if T <= 1e-12:
            return False
        try:
            return random.random() < math.exp(-delta / T)
        except OverflowError:
            return False

    while True:
        it += 1
        if it % 64 == 0:
            now = time.time()
            if now >= deadline:
                break
            frac = min(1.0, (now - sa_start) / horizon)
            T = T0 * (Tend / T0) ** frac
            if now - last_imp > stall_limit:
                load_state(best_route[:], dict(best_assign))
                last_imp = now

        r = random.random()
        applied = False

        if r < 0.28 and len(route) > 2:
            i = random.randint(1, len(route) - 2)
            c = route[i]
            if disp_cnt[c] == 0:
                res = best_truck_to_drone(c, i)
                if res and accept(res[0]):
                    apply_truck_to_drone(c, i, *res)
                    applied = True
        elif r < 0.44 and assign:
            c = random.choice(list(assign.keys()))
            delta, i, neww = best_drone_to_truck(c)
            if accept(delta):
                apply_drone_to_truck(c, i, neww, delta)
                applied = True
        elif r < 0.62 and assign:
            c = random.choice(list(assign.keys()))
            res = best_reassign(c)
            if res and accept(res[0]):
                apply_reassign(c, *res)
                applied = True
        elif r < 0.84 or not sym:
            n = len(route)
            if n >= 4:
                L = random.randint(1, min(3, n - 3))
                i = random.randint(1, n - 1 - L)
                seg = route[i:i + L]
                a = route[i - 1]
                b = route[i + L]
                drem = tt[a][b] - tt[a][seg[0]] - tt[seg[-1]][b]
                rest = route[:i] + route[i + L:]
                bestj = None
                bestins = None
                for j in range(len(rest) - 1):
                    x = rest[j]
                    y = rest[j + 1]
                    dins = tt[x][seg[0]] + tt[seg[-1]][y] - tt[x][y]
                    if bestins is None or dins < bestins:
                        bestins = dins
                        bestj = j
                delta = drem + bestins
                if accept(delta):
                    route[:] = rest[:bestj + 1] + seg + rest[bestj + 1:]
                    travel += delta
                    cost += delta
                    applied = True
        else:
            n = len(route)
            if n >= 5:
                i = random.randint(1, n - 3)
                j = random.randint(i + 1, n - 2)
                a = route[i - 1]
                b = route[j + 1]
                delta = tt[a][route[j]] + tt[route[i]][b] \
                    - tt[a][route[i]] - tt[route[j]][b]
                if accept(delta):
                    route[i:j + 1] = reversed(route[i:j + 1])
                    travel += delta
                    cost += delta
                    applied = True

        if applied and cost < best_cost - 1e-9:
            best_cost = cost
            best_route = route[:]
            best_assign = dict(assign)
            last_imp = time.time()
            if logger:
                sol = build_solution(best_route, best_assign)
                logger.log_solution(sol["objective_value"], sol)

    # ---------- final polish: greedy improving moves on best ----------
    load_state(best_route[:], dict(best_assign))
    changed = True
    while changed and time.time() < deadline + 0.3:
        changed = False
        for c in list(assign.keys()):
            res = best_reassign(c)
            if res and res[0] < -1e-9:
                apply_reassign(c, *res)
                changed = True
    if cost < best_cost - 1e-9:
        best_cost = cost
        best_route = route[:]
        best_assign = dict(assign)

    solution = build_solution(best_route, best_assign)
    if logger:
        logger.log_solution(solution["objective_value"], solution)

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()
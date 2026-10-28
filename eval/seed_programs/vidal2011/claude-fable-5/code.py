import argparse
import json
import math
import random
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", required=True)
    ap.add_argument("--solution_path", required=True)
    ap.add_argument("--time_limit", type=int, default=60)
    ap.add_argument("--log_path", default=None)
    args = ap.parse_args()

    logger = None
    if args.log_path:
        try:
            from solution_logger import SolutionLogger
            logger = SolutionLogger(args.log_path, sense="minimize")
        except Exception:
            logger = None

    start = time.perf_counter()
    deadline = start + max(1, args.time_limit) - 0.5
    random.seed(0)

    with open(args.instance_path) as f:
        inst = json.load(f)

    n = inst["n"]
    D = inst["d"]
    T = inst["t"]
    m = inst["m"]
    cap = inst["vehicle_capacity"]
    durlim = inst["route_duration_limit"]

    depots = sorted(inst["depots"], key=lambda z: z["id"])
    custs = sorted(inst["customers"], key=lambda z: z["id"])
    coords = [(dp["x"], dp["y"]) for dp in depots] + [(c["x"], c["y"]) for c in custs]
    N = D + n
    dist = [[0.0] * N for _ in range(N)]
    for i in range(N):
        xi, yi = coords[i]
        row = dist[i]
        for j in range(i + 1, N):
            xj, yj = coords[j]
            dd = math.hypot(xi - xj, yi - yj)
            row[j] = dd
            dist[j][i] = dd

    demand = [c["demand"] for c in custs]
    serv = [c["service_duration"] for c in custs]
    patterns = [[tuple(sorted(int(x) for x in p)) for p in c["pattern_list"]] for c in custs]

    # State
    assign = [None] * n  # (depot, pattern_index)
    routes = {(dep, day): [] for dep in range(D) for day in range(1, T + 1)}

    def now():
        return time.perf_counter()

    def r_cost(dep, r):
        if not r:
            return 0.0
        c = dist[dep][D + r[0]] + dist[D + r[-1]][dep]
        for i in range(len(r) - 1):
            c += dist[D + r[i]][D + r[i + 1]]
        return c

    def r_load(r):
        return sum(demand[c] for c in r)

    def r_dur(dep, r):
        return r_cost(dep, r) + sum(serv[c] for c in r)

    def total_cost():
        tot = 0.0
        for (dep, day), rl in routes.items():
            for r in rl:
                tot += r_cost(dep, r)
        return tot

    def best_insertion(dep, day, c):
        """Return (delta_cost, route_index, position) or None. route_index == len(rl) means new route."""
        rl = routes[(dep, day)]
        nc = D + c
        best = None
        for ri, r in enumerate(rl):
            if r_load(r) + demand[c] > cap:
                continue
            dur = r_dur(dep, r)
            L = len(r)
            for pos in range(L + 1):
                a = dep if pos == 0 else D + r[pos - 1]
                b = dep if pos == L else D + r[pos]
                delta = dist[a][nc] + dist[nc][b] - dist[a][b]
                if dur + delta + serv[c] <= durlim + 1e-9:
                    if best is None or delta < best[0] - 1e-12:
                        best = (delta, ri, pos)
        if len(rl) < m:
            delta = 2.0 * dist[dep][nc]
            if delta + serv[c] <= durlim + 1e-9:
                if best is None or delta < best[0] - 1e-12:
                    best = (delta, len(rl), 0)
        return best

    def do_insert(dep, day, c, ri, pos):
        rl = routes[(dep, day)]
        if ri >= len(rl):
            rl.append([c])
        else:
            rl[ri].insert(pos, c)

    def force_insert(dep, day, c):
        ins = best_insertion(dep, day, c)
        if ins is not None:
            do_insert(dep, day, c, ins[1], ins[2])
        else:
            routes[(dep, day)].append([c])  # last resort (may exceed fleet)

    # ---------------- Construction ----------------
    def construct():
        order = sorted(range(n), key=lambda c: (-demand[c], -serv[c]))
        for c in order:
            best = None
            if now() < deadline:
                for dep in range(D):
                    for pi, pat in enumerate(patterns[c]):
                        total = 0.0
                        plan = []
                        ok = True
                        for day in pat:
                            ins = best_insertion(dep, day, c)
                            if ins is None:
                                ok = False
                                break
                            total += ins[0]
                            plan.append((day, ins))
                        if ok and (best is None or total < best[0]):
                            best = (total, dep, pi, plan)
            if best is not None:
                _, dep, pi, plan = best
                assign[c] = (dep, pi)
                for day, (dl, ri, pos) in plan:
                    do_insert(dep, day, c, ri, pos)
            else:
                dep = min(range(D), key=lambda dp: dist[dp][D + c])
                assign[c] = (dep, 0)
                for day in patterns[c][0]:
                    force_insert(dep, day, c)

    # ---------------- Local search ----------------
    def two_opt(dep, r):
        any_imp = False
        improved = True
        while improved and now() < deadline:
            improved = False
            L = len(r)
            for i in range(L - 1):
                a = dep if i == 0 else D + r[i - 1]
                ni = D + r[i]
                for j in range(i + 1, L):
                    b = dep if j == L - 1 else D + r[j + 1]
                    nj = D + r[j]
                    delta = dist[a][nj] + dist[ni][b] - dist[a][ni] - dist[nj][b]
                    if delta < -1e-9:
                        r[i:j + 1] = r[i:j + 1][::-1]
                        improved = True
                        any_imp = True
                        break
                if improved:
                    break
        return any_imp

    def relocate_pass(dep, rl):
        moved = False
        changed = True
        while changed and now() < deadline:
            changed = False
            loads = [r_load(r) for r in rl]
            durs = [r_dur(dep, r) for r in rl]
            nr = len(rl)
            for i1 in range(nr):
                r1 = rl[i1]
                for p1 in range(len(r1)):
                    x = r1[p1]
                    nx = D + x
                    a = dep if p1 == 0 else D + r1[p1 - 1]
                    b = dep if p1 == len(r1) - 1 else D + r1[p1 + 1]
                    rem = dist[a][b] - dist[a][nx] - dist[nx][b]
                    for i2 in range(nr):
                        if i2 == i1:
                            continue
                        r2 = rl[i2]
                        if loads[i2] + demand[x] > cap:
                            continue
                        L2 = len(r2)
                        for p2 in range(L2 + 1):
                            a2 = dep if p2 == 0 else D + r2[p2 - 1]
                            b2 = dep if p2 == L2 else D + r2[p2]
                            add = dist[a2][nx] + dist[nx][b2] - dist[a2][b2]
                            if rem + add < -1e-9 and durs[i2] + add + serv[x] <= durlim + 1e-9:
                                r1.pop(p1)
                                r2.insert(p2, x)
                                moved = True
                                changed = True
                                break
                        if changed:
                            break
                    if changed:
                        break
                if changed:
                    break
        rl[:] = [r for r in rl if r]
        return moved

    def swap_pass(dep, rl):
        moved = False
        changed = True
        while changed and now() < deadline:
            changed = False
            loads = [r_load(r) for r in rl]
            durs = [r_dur(dep, r) for r in rl]
            nr = len(rl)
            for i1 in range(nr):
                r1 = rl[i1]
                for i2 in range(i1 + 1, nr):
                    r2 = rl[i2]
                    for p1 in range(len(r1)):
                        x = r1[p1]
                        nx = D + x
                        a1 = dep if p1 == 0 else D + r1[p1 - 1]
                        b1 = dep if p1 == len(r1) - 1 else D + r1[p1 + 1]
                        for p2 in range(len(r2)):
                            y = r2[p2]
                            if loads[i1] - demand[x] + demand[y] > cap:
                                continue
                            if loads[i2] - demand[y] + demand[x] > cap:
                                continue
                            ny = D + y
                            a2 = dep if p2 == 0 else D + r2[p2 - 1]
                            b2 = dep if p2 == len(r2) - 1 else D + r2[p2 + 1]
                            d1 = dist[a1][ny] + dist[ny][b1] - dist[a1][nx] - dist[nx][b1]
                            d2 = dist[a2][nx] + dist[nx][b2] - dist[a2][ny] - dist[ny][b2]
                            if d1 + d2 < -1e-9:
                                if (durs[i1] + d1 - serv[x] + serv[y] <= durlim + 1e-9 and
                                        durs[i2] + d2 - serv[y] + serv[x] <= durlim + 1e-9):
                                    r1[p1] = y
                                    r2[p2] = x
                                    moved = True
                                    changed = True
                                    break
                        if changed:
                            break
                    if changed:
                        break
                if changed:
                    break
        return moved

    def local_search():
        for key in routes:
            if now() > deadline:
                return
            dep, day = key
            rl = routes[key]
            rl[:] = [r for r in rl if r]
            improved = True
            while improved and now() < deadline:
                improved = False
                for r in rl:
                    if two_opt(dep, r):
                        improved = True
                if relocate_pass(dep, rl):
                    improved = True
                if swap_pass(dep, rl):
                    improved = True
            rl[:] = [r for r in rl if r]

    # ---------------- Depot/pattern reassignment ----------------
    def remove_customer(c):
        dep0, pi0 = assign[c]
        removed = []
        gain = 0.0
        for day in patterns[c][pi0]:
            rl = routes[(dep0, day)]
            for ri, r in enumerate(rl):
                if c in r:
                    pos = r.index(c)
                    a = dep0 if pos == 0 else D + r[pos - 1]
                    b = dep0 if pos == len(r) - 1 else D + r[pos + 1]
                    gain += dist[a][D + c] + dist[D + c][b] - dist[a][b]
                    r.pop(pos)
                    removed.append((day, ri, pos))
                    break
        return removed, gain

    def try_reassign(c):
        dep0, pi0 = assign[c]
        removed, gain = remove_customer(c)
        best = None
        for dep in range(D):
            for pi, pat in enumerate(patterns[c]):
                total = 0.0
                plan = []
                ok = True
                for day in pat:
                    ins = best_insertion(dep, day, c)
                    if ins is None:
                        ok = False
                        break
                    total += ins[0]
                    plan.append((day, ins))
                if ok and (best is None or total < best[0]):
                    best = (total, dep, pi, plan)
        if best is not None and best[0] < gain - 1e-9:
            _, dep, pi, plan = best
            assign[c] = (dep, pi)
            for day, (dl, ri, pos) in plan:
                do_insert(dep, day, c, ri, pos)
            return gain - best[0]
        # revert
        for day, ri, pos in removed:
            routes[(dep0, day)][ri].insert(pos, c)
        return 0.0

    def reassign_all():
        order = list(range(n))
        random.shuffle(order)
        gain = 0.0
        for c in order:
            if now() > deadline:
                break
            gain += try_reassign(c)
        for key in routes:
            routes[key] = [r for r in routes[key] if r]
        return gain

    # ---------------- Perturbation ----------------
    def perturb(k):
        cs = random.sample(range(n), min(k, n))
        for c in cs:
            dep0, pi0 = assign[c]
            remove_customer(c)
            cands = [(dep, pi) for dep in range(D) for pi in range(len(patterns[c]))]
            random.shuffle(cands)
            placed = False
            for dep, pi in cands:
                plan = []
                ok = True
                for day in patterns[c][pi]:
                    ins = best_insertion(dep, day, c)
                    if ins is None:
                        ok = False
                        break
                    plan.append((day, ins))
                if ok:
                    assign[c] = (dep, pi)
                    for day, (dl, ri, pos) in plan:
                        do_insert(dep, day, c, ri, pos)
                    placed = True
                    break
            if not placed:
                assign[c] = (dep0, pi0)
                for day in patterns[c][pi0]:
                    force_insert(dep0, day, c)
        for key in routes:
            routes[key] = [r for r in routes[key] if r]

    # ---------------- Solution building ----------------
    def build_solution(assign_s, routes_s):
        cost = 0.0
        out_routes = []
        for (dep, day) in sorted(routes_s.keys()):
            rl = routes_s[(dep, day)]
            veh = 0
            for r in rl:
                if not r:
                    continue
                veh += 1
                cost += r_cost(dep, r)
                out_routes.append({
                    "depot": dep,
                    "period": day,
                    "vehicle": veh,
                    "customers": list(r),
                })
        assignments = {}
        for c in range(n):
            dep, pi = assign_s[c]
            assignments[str(c)] = {"depot": dep, "pattern": list(patterns[c][pi])}
        return {"objective_value": cost, "routes": out_routes, "assignments": assignments}

    def snapshot():
        return (list(assign), {k: [list(r) for r in v] for k, v in routes.items()})

    def restore(snap):
        nonlocal assign
        a, rt = snap
        assign = list(a)
        for k in routes:
            routes[k] = [list(r) for r in rt[k]]

    # ---------------- Run ----------------
    construct()
    local_search()
    while now() < deadline:
        g = reassign_all()
        local_search()
        if g <= 1e-9:
            break

    best_snap = snapshot()
    best_cost = total_cost()
    if logger:
        logger.log_solution(best_cost, build_solution(*best_snap))

    # Iterated local search
    while now() < deadline:
        restore(best_snap)
        k = random.randint(2, max(3, n // 10))
        perturb(k)
        local_search()
        rounds = 0
        while now() < deadline and rounds < 3:
            g = reassign_all()
            local_search()
            rounds += 1
            if g <= 1e-9:
                break
        cur = total_cost()
        if cur < best_cost - 1e-9:
            best_cost = cur
            best_snap = snapshot()
            if logger:
                logger.log_solution(best_cost, build_solution(*best_snap))

    sol = build_solution(*best_snap)
    if logger:
        logger.log_solution(sol["objective_value"], sol)
    with open(args.solution_path, "w") as f:
        json.dump(sol, f, indent=2)


if __name__ == "__main__":
    main()
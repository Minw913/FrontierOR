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
    ap.add_argument('--time_limit', type=int, default=300)
    ap.add_argument('--log_path', default=None)
    args = ap.parse_args()

    t_start = time.time()
    deadline = t_start + max(3, args.time_limit) - 1.0
    random.seed(0)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path) as f:
        inst = json.load(f)

    Q1 = inst['Q1']
    Q2 = inst['Q2']
    T1 = inst['T1']
    T2 = inst['T2']
    dep_id = inst['depot']['id']

    coords = {dep_id: (inst['depot']['x'], inst['depot']['y'])}
    sats = []
    cap = {}
    maxveh = {}
    hcost = {}
    for s in inst['satellites']:
        sid = s['id']
        sats.append(sid)
        coords[sid] = (s['x'], s['y'])
        cap[sid] = s['capacity']
        maxveh[sid] = s['max_vehicles']
        hcost[sid] = s['handling_cost']

    dem = {}
    for c in inst['customers']:
        coords[c['id']] = (c['x'], c['y'])
        dem[c['id']] = c['demand']

    groups = {}
    for g in inst['groups']:
        groups[g['group_id']] = list(g['customer_ids'])
    gdem = {gid: sum(dem[c] for c in cs) for gid, cs in groups.items()}
    gids_all = list(groups.keys())

    hyp = math.hypot

    def dist(a, b):
        xa, ya = coords[a]
        xb, yb = coords[b]
        return hyp(xa - xb, ya - yb)

    def route_cost(dnode, route):
        c = dist(dnode, route[0]) + dist(route[-1], dnode)
        for i in range(len(route) - 1):
            c += dist(route[i], route[i + 1])
        return c

    def two_opt(dnode, route):
        n = len(route)
        if n < 3:
            return route
        improved = True
        while improved:
            improved = False
            for i in range(n - 1):
                a = dnode if i == 0 else route[i - 1]
                di_a = dist(a, route[i])
                for j in range(i + 1, n):
                    b = dnode if j == n - 1 else route[j + 1]
                    delta = dist(a, route[j]) + dist(route[i], b) - di_a - dist(route[j], b)
                    if delta < -1e-9:
                        route[i:j + 1] = route[i:j + 1][::-1]
                        improved = True
                        di_a = dist(a, route[i])
        return route

    def relocate(dnode, R, L, cap_v, dm):
        passes = 0
        improved = True
        while improved and passes < 4:
            improved = False
            passes += 1
            ri = 0
            while ri < len(R):
                ci = 0
                while ci < len(R[ri]):
                    c = R[ri][ci]
                    prev = dnode if ci == 0 else R[ri][ci - 1]
                    nxt = dnode if ci == len(R[ri]) - 1 else R[ri][ci + 1]
                    gain = dist(prev, c) + dist(c, nxt) - dist(prev, nxt)
                    bestdelta = -1e-9
                    bestpos = None
                    for rj in range(len(R)):
                        if rj == ri or L[rj] + dm[c] > cap_v:
                            continue
                        rr = R[rj]
                        for pos in range(len(rr) + 1):
                            p = dnode if pos == 0 else rr[pos - 1]
                            q = dnode if pos == len(rr) else rr[pos]
                            add = dist(p, c) + dist(c, q) - dist(p, q)
                            if add - gain < bestdelta:
                                bestdelta = add - gain
                                bestpos = (rj, pos)
                    if bestpos is not None:
                        rj, pos = bestpos
                        R[ri].pop(ci)
                        L[ri] -= dm[c]
                        R[rj].insert(pos, c)
                        L[rj] += dm[c]
                        improved = True
                        if not R[ri]:
                            break
                    else:
                        ci += 1
                ri += 1
            keep = [k for k in range(len(R)) if R[k]]
            R[:] = [R[k] for k in keep]
            L[:] = [L[k] for k in keep]

    def solve_cvrp(dnode, custs, dm, cap_v, max_routes):
        custs = list(custs)
        if not custs:
            return [], 0.0
        for c in custs:
            if dm[c] > cap_v:
                return None
        rid = {}
        routes = {}
        loads = {}
        for i, c in enumerate(custs):
            rid[c] = i
            routes[i] = [c]
            loads[i] = dm[c]
        d0 = {c: dist(dnode, c) for c in custs}
        sav = []
        n = len(custs)
        for i in range(n):
            a = custs[i]
            for j in range(i + 1, n):
                b = custs[j]
                sav.append((d0[a] + d0[b] - dist(a, b), a, b))
        sav.sort(key=lambda x: -x[0])
        for s, a, b in sav:
            ra, rb = rid[a], rid[b]
            if ra == rb or loads[ra] + loads[rb] > cap_v:
                continue
            Ra, Rb = routes[ra], routes[rb]
            if Ra[-1] == a and Rb[0] == b:
                new = Ra + Rb
            elif Rb[-1] == b and Ra[0] == a:
                new = Rb + Ra
            elif Ra[0] == a and Rb[0] == b:
                new = Ra[::-1] + Rb
            elif Ra[-1] == a and Rb[-1] == b:
                new = Ra + Rb[::-1]
            else:
                continue
            routes[ra] = new
            loads[ra] += loads[rb]
            for c in Rb:
                rid[c] = ra
            del routes[rb]
            del loads[rb]
        R = list(routes.values())
        L = [sum(dm[c] for c in r) for r in R]
        while len(R) > max_routes:
            best = None
            for i in range(len(R)):
                for j in range(i + 1, len(R)):
                    if L[i] + L[j] > cap_v:
                        continue
                    base = route_cost(dnode, R[i]) + route_cost(dnode, R[j])
                    for new in (R[i] + R[j], R[i] + R[j][::-1],
                                R[j] + R[i], R[i][::-1] + R[j]):
                        delta = route_cost(dnode, new) - base
                        if best is None or delta < best[0]:
                            best = (delta, i, j, new)
            if best is None:
                return None
            _, i, j, new = best
            R[i] = new
            L[i] += L[j]
            del R[j]
            del L[j]
        for k in range(len(R)):
            R[k] = two_opt(dnode, R[k])
        relocate(dnode, R, L, cap_v, dm)
        for k in range(len(R)):
            R[k] = two_opt(dnode, R[k])
        if len(R) > max_routes:
            return None
        cost = sum(route_cost(dnode, r) for r in R)
        return R, cost

    # ---------- first echelon ----------
    fe_cache = {}

    def solve_fe(sat_dem):
        key = tuple(sorted((s, d) for s, d in sat_dem.items() if d > 0))
        if key in fe_cache:
            return fe_cache[key]
        vehicles = []
        rem = {}
        for s, d in key:
            nf = d // Q1
            r = d - nf * Q1
            if r == 0 and nf > 0:
                nf -= 1
                r = Q1
            for _ in range(nf):
                vehicles.append(([s], {s: Q1}))
            if r > 0:
                rem[s] = r
        fulls_cost = sum(2 * dist(dep_id, v[0][0]) for v in vehicles)
        max_r = T1 - len(vehicles)
        if rem:
            res = solve_cvrp(dep_id, list(rem.keys()), rem, Q1, max_r)
            if res is None:
                fe_cache[key] = None
                return None
            R, c = res
        else:
            R, c = [], 0.0
        for r in R:
            vehicles.append((r, {s: rem[s] for s in r}))
        out = (vehicles, fulls_cost + c)
        fe_cache[key] = out
        return out

    # ---------- second echelon ----------
    se_cache = {}

    def solve_se(s, custs_key):
        key = (s, custs_key)
        if key in se_cache:
            return se_cache[key]
        res = solve_cvrp(s, list(custs_key), dem, Q2, maxveh[s])
        se_cache[key] = res
        return res

    def evaluate(assign):
        per_sat = {s: [] for s in sats}
        for gid, s in assign.items():
            per_sat[s].extend(groups[gid])
        sat_dem = {}
        for s in sats:
            d = sum(dem[c] for c in per_sat[s])
            if d > cap[s]:
                return None
            sat_dem[s] = d
        se = {}
        tot_routes = 0
        se_cost = 0.0
        for s in sats:
            ck = tuple(sorted(per_sat[s]))
            res = solve_se(s, ck)
            if res is None:
                return None
            se[s] = res
            tot_routes += len(res[0])
            se_cost += res[1]
        if tot_routes > T2:
            se = {s: ([list(r) for r in se[s][0]], se[s][1]) for s in sats}
            while tot_routes > T2:
                best = None
                for s in sats:
                    R = se[s][0]
                    for i in range(len(R)):
                        li = sum(dem[c] for c in R[i])
                        for j in range(i + 1, len(R)):
                            lj = sum(dem[c] for c in R[j])
                            if li + lj > Q2:
                                continue
                            base = route_cost(s, R[i]) + route_cost(s, R[j])
                            for new in (R[i] + R[j], R[i] + R[j][::-1]):
                                delta = route_cost(s, new) - base
                                if best is None or delta < best[0]:
                                    best = (delta, s, i, j, new)
                if best is None:
                    return None
                _, s, i, j, new = best
                R = se[s][0]
                R[i] = two_opt(s, list(new))
                del R[j]
                se[s] = (R, sum(route_cost(s, r) for r in R))
                tot_routes -= 1
            se_cost = sum(se[s][1] for s in sats)
        fe = solve_fe(sat_dem)
        if fe is None:
            return None
        h = sum(hcost[s] * sat_dem[s] for s in sats)
        total = fe[1] + h + se_cost
        return total, se, fe, sat_dem

    def centroid_dist(s, cs):
        return sum(dist(s, c) for c in cs) / len(cs)

    def initial_assign():
        order = sorted(gids_all, key=lambda g: -gdem[g])
        assign = {}
        used = {s: 0 for s in sats}
        for gid in order:
            cs = groups[gid]
            best_s = None
            best_sc = None
            for s in sats:
                if used[s] + gdem[gid] > cap[s]:
                    continue
                nveh = math.ceil((used[s] + gdem[gid]) / Q2) if Q2 > 0 else 1
                if nveh > maxveh[s]:
                    continue
                sc = hcost[s] * gdem[gid] + 2.0 * centroid_dist(s, cs) * max(1, math.ceil(gdem[gid] / Q2))
                if best_sc is None or sc < best_sc:
                    best_sc = sc
                    best_s = s
            if best_s is None:
                for s in sats:
                    if used[s] + gdem[gid] <= cap[s]:
                        best_s = s
                        break
            if best_s is None:
                return None
            assign[gid] = best_s
            used[best_s] += gdem[gid]
        return assign

    def random_assign():
        assign = {}
        used = {s: 0 for s in sats}
        order = list(gids_all)
        random.shuffle(order)
        for gid in order:
            cands = [s for s in sats if used[s] + gdem[gid] <= cap[s]]
            if not cands:
                return None
            s = random.choice(cands)
            assign[gid] = s
            used[s] += gdem[gid]
        return assign

    def build_solution(res, assign):
        total, se, fe, sat_dem = res
        fe_routes = []
        for vi, (r, deliv) in enumerate(fe[0]):
            fe_routes.append({
                "vehicle": vi,
                "route": [dep_id] + list(r) + [dep_id],
                "deliveries": {str(s): int(q) for s, q in deliv.items()}
            })
        se_routes = []
        for s in sats:
            for r in se[s][0]:
                se_routes.append({"satellite": s, "route": [s] + list(r) + [s]})
        return {
            "objective_value": float(total),
            "first_echelon_routes": fe_routes,
            "second_echelon_routes": se_routes,
            "group_assignments": {str(g): int(assign[g]) for g in assign}
        }

    # ---------- construct initial solution ----------
    assign = initial_assign()
    res = evaluate(assign) if assign is not None else None
    tries = 0
    while res is None and tries < 500 and time.time() < deadline:
        a = random_assign()
        if a is not None:
            r = evaluate(a)
            if r is not None:
                assign = a
                res = r
        tries += 1

    if res is None:
        # last resort: dump minimal structure
        sol = {"objective_value": float('inf'), "first_echelon_routes": [],
               "second_echelon_routes": [], "group_assignments": {}}
        with open(args.solution_path, 'w') as f:
            json.dump(sol, f, indent=2)
        return

    best_cost = res[0]
    best_res = res
    best_assign = dict(assign)
    if logger:
        logger.log_solution(best_cost, build_solution(best_res, best_assign))

    cur_assign = dict(assign)
    cur_cost = best_cost

    # ---------- local search on group assignments ----------
    if len(sats) > 1 and len(gids_all) > 0:
        gids = list(gids_all)
        while time.time() < deadline:
            improved = False
            random.shuffle(gids)
            for gid in gids:
                if time.time() >= deadline:
                    break
                cur_s = cur_assign[gid]
                cand = sorted(sats, key=lambda s: centroid_dist(s, groups[gid]))
                for s in cand:
                    if s == cur_s:
                        continue
                    if time.time() >= deadline:
                        break
                    cur_assign[gid] = s
                    r = evaluate(cur_assign)
                    if r is not None and r[0] < cur_cost - 1e-9:
                        cur_cost = r[0]
                        improved = True
                        if cur_cost < best_cost - 1e-9:
                            best_cost = cur_cost
                            best_res = r
                            best_assign = dict(cur_assign)
                            if logger:
                                logger.log_solution(best_cost, build_solution(best_res, best_assign))
                        break
                    else:
                        cur_assign[gid] = cur_s
            if time.time() >= deadline:
                break
            if not improved:
                did = False
                if len(gids) >= 2:
                    for _ in range(min(60, len(gids) * (len(gids) - 1))):
                        if time.time() >= deadline:
                            break
                        g1, g2 = random.sample(gids, 2)
                        s1, s2 = cur_assign[g1], cur_assign[g2]
                        if s1 == s2:
                            continue
                        cur_assign[g1], cur_assign[g2] = s2, s1
                        r = evaluate(cur_assign)
                        if r is not None and r[0] < cur_cost - 1e-9:
                            cur_cost = r[0]
                            did = True
                            if cur_cost < best_cost - 1e-9:
                                best_cost = cur_cost
                                best_res = r
                                best_assign = dict(cur_assign)
                                if logger:
                                    logger.log_solution(best_cost, build_solution(best_res, best_assign))
                            break
                        cur_assign[g1], cur_assign[g2] = s1, s2
                if not did:
                    # perturbation: restart from best, kick a few groups
                    cur_assign = dict(best_assign)
                    ok = False
                    for _attempt in range(20):
                        if time.time() >= deadline:
                            break
                        trial = dict(best_assign)
                        k = random.randint(1, min(3, len(gids)))
                        for _ in range(k):
                            g = random.choice(gids)
                            trial[g] = random.choice(sats)
                        r = evaluate(trial)
                        if r is not None:
                            cur_assign = trial
                            cur_cost = r[0]
                            ok = True
                            if cur_cost < best_cost - 1e-9:
                                best_cost = cur_cost
                                best_res = r
                                best_assign = dict(cur_assign)
                                if logger:
                                    logger.log_solution(best_cost, build_solution(best_res, best_assign))
                            break
                    if not ok:
                        cur_assign = dict(best_assign)
                        cur_cost = best_cost

    sol = build_solution(best_res, best_assign)
    with open(args.solution_path, 'w') as f:
        json.dump(sol, f, indent=2)


if __name__ == '__main__':
    main()
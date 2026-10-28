import argparse
import json
import math
import random
import time
from array import array


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    logger = None
    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize")

    t0 = time.time()
    BUDGET = max(2.0, float(args.time_limit))
    random.seed(0)

    with open(args.instance_path) as f:
        inst = json.load(f)

    depot_id = inst["depot"]["id"]
    sats = inst.get("satellites", []) or []
    custs = inst.get("customers", []) or []
    Q1 = int(inst["Q1"])
    Q2 = int(inst["Q2"])
    T1 = int(inst["T1"])
    T2 = int(inst["T2"])

    sat_ids = [s["id"] for s in sats]
    CAP = {s["id"]: s["capacity"] for s in sats}
    MAXV = {s["id"]: s["max_vehicles"] for s in sats}
    HC = {s["id"]: float(s["handling_cost"]) for s in sats}

    DEMC = {}
    GROUPS = {}
    for c in custs:
        DEMC[c["id"]] = c["demand"]
        GROUPS.setdefault(c["group"], []).append(c["id"])
    group_ids = sorted(GROUPS.keys())
    GDEM = {g: sum(DEMC[c] for c in GROUPS[g]) for g in group_ids}
    cust_ids = [c["id"] for c in custs]

    # Degenerate case: nothing to serve.
    if not sat_ids or not cust_ids or not group_ids:
        sol = {
            "objective_value": 0.0,
            "first_echelon_routes": [],
            "second_echelon_routes": [],
            "group_assignments": {str(g): (sat_ids[0] if sat_ids else 0) for g in group_ids},
        }
        if logger:
            logger.log_solution(0.0, sol)
        with open(args.solution_path, "w") as f:
            json.dump(sol, f)
        return

    all_ids = [depot_id] + sat_ids + cust_ids
    GI = {nid: i for i, nid in enumerate(all_ids)}
    N = len(all_ids)
    xs = [0.0] * N
    ys = [0.0] * N
    xs[GI[depot_id]] = float(inst["depot"]["x"])
    ys[GI[depot_id]] = float(inst["depot"]["y"])
    for s in sats:
        xs[GI[s["id"]]] = float(s["x"])
        ys[GI[s["id"]]] = float(s["y"])
    for c in custs:
        xs[GI[c["id"]]] = float(c["x"])
        ys[GI[c["id"]]] = float(c["y"])

    D = [array("d", [0.0] * N) for _ in range(N)]
    for i in range(N):
        Di = D[i]
        xi = xs[i]
        yi = ys[i]
        for j in range(i + 1, N):
            d = math.hypot(xi - xs[j], yi - ys[j])
            Di[j] = d
            D[j][i] = d

    depot_i = GI[depot_id]
    SATI = {sid: GI[sid] for sid in sat_ids}
    CUSTI = {cid: GI[cid] for cid in cust_ids}
    CAPNODE = {SATI[sid]: CAP[sid] for sid in sat_ids}

    # ------------------------------------------------------------------
    # Generic VRP utilities. Clients are referenced by positions; the
    # actual node index is nodes[position], demand is dems[position].
    # ------------------------------------------------------------------
    def route_dist(dep, nodes, r):
        if not r:
            return 0.0
        c = D[dep][nodes[r[0]]]
        for i in range(len(r) - 1):
            c += D[nodes[r[i]]][nodes[r[i + 1]]]
        c += D[nodes[r[-1]]][dep]
        return c

    def route_dist_g(dep, seq):
        if not seq:
            return 0.0
        c = D[dep][seq[0]]
        for i in range(len(seq) - 1):
            c += D[seq[i]][seq[i + 1]]
        c += D[seq[-1]][dep]
        return c

    def two_opt(dep, nodes, r):
        m = len(r)
        if m < 3:
            return list(r)
        pos = list(r)
        g = [nodes[p] for p in pos]
        improved = True
        it = 0
        while improved and it < 60:
            improved = False
            it += 1
            for i in range(m):
                a = dep if i == 0 else g[i - 1]
                b = g[i]
                dab = D[a][b]
                for j in range(i + 1, m):
                    c = g[j]
                    d = dep if j == m - 1 else g[j + 1]
                    if D[a][c] + D[b][d] < dab + D[c][d] - 1e-9:
                        g[i:j + 1] = g[i:j + 1][::-1]
                        pos[i:j + 1] = pos[i:j + 1][::-1]
                        improved = True
                        break
        return pos

    def savings_build(dep, nodes, dems, Q, max_same=None):
        n = len(nodes)
        if n == 0:
            return []
        if n == 1:
            return [[0]]
        dep_d = [0.0] * n
        for i in range(n):
            dep_d[i] = D[dep][nodes[i]]
        sav = []
        for i in range(n):
            gi = nodes[i]
            di = dep_d[i]
            for j in range(i + 1, n):
                sav.append((di + dep_d[j] - D[gi][nodes[j]], i, j))
        sav.sort(key=lambda t: t[0], reverse=True)
        next_ = [-1] * n
        prev_ = [-1] * n
        rid = list(range(n))
        active = [True] * n
        rdem = list(dems)
        first = list(range(n))
        last = list(range(n))
        rcnt = None
        if max_same is not None:
            rcnt = [{nodes[i]: dems[i]} for i in range(n)]

        def reverse(r):
            k = first[r]
            while k != -1:
                next_[k], prev_[k] = prev_[k], next_[k]
                k = prev_[k]
            first[r], last[r] = last[r], first[r]

        for sv, i, j in sav:
            ri = rid[i]
            rj = rid[j]
            if ri == rj or not (active[ri] and active[rj]):
                continue
            if rdem[ri] + rdem[rj] > Q:
                continue
            if first[ri] != i and last[ri] != i:
                continue
            if first[rj] != j and last[rj] != j:
                continue
            if rcnt is not None:
                ca = rcnt[ri]
                cb = rcnt[rj]
                ok = True
                for nd, vv in cb.items():
                    if nd in ca and ca[nd] + vv > max_same[nd]:
                        ok = False
                        break
                if not ok:
                    continue
            if first[ri] == i:
                reverse(ri)
            if last[rj] == j:
                reverse(rj)
            next_[i] = j
            prev_[j] = i
            k = j
            while k != -1:
                rid[k] = ri
                k = next_[k]
            rdem[ri] += rdem[rj]
            last[ri] = last[rj]
            active[rj] = False
            if rcnt is not None:
                ca = rcnt[ri]
                for nd, vv in rcnt[rj].items():
                    ca[nd] = ca.get(nd, 0) + vv
        routes = []
        for r in range(n):
            if active[r]:
                seq = []
                k = first[r]
                while k != -1:
                    seq.append(k)
                    k = next_[k]
                routes.append(seq)
        return routes

    def merge_to_k(dep, nodes, dems, Q, routes, K, cap_of_node=None):
        routes = [list(r) for r in routes]
        while len(routes) > K:
            R = len(routes)
            rd = [0] * R
            for a in range(R):
                s = 0
                for p in routes[a]:
                    s += dems[p]
                rd[a] = s
            best = None
            for a in range(R):
                la = routes[a]
                ga0 = nodes[la[0]]
                ga1 = nodes[la[-1]]
                for b in range(a + 1, R):
                    if rd[a] + rd[b] > Q:
                        continue
                    if cap_of_node is not None:
                        ok = True
                        cnt = {}
                        for p in la:
                            cnt[nodes[p]] = cnt.get(nodes[p], 0) + dems[p]
                        for p in routes[b]:
                            nd = nodes[p]
                            v = cnt.get(nd, 0) + dems[p]
                            if v > cap_of_node[nd]:
                                ok = False
                                break
                            cnt[nd] = v
                        if not ok:
                            continue
                    lb = routes[b]
                    gb0 = nodes[lb[0]]
                    gb1 = nodes[lb[-1]]
                    s1 = D[dep][ga1] + D[gb0][dep] - D[ga1][gb0]
                    s2 = D[dep][ga1] + D[gb1][dep] - D[ga1][gb1]
                    s3 = D[dep][ga0] + D[gb0][dep] - D[ga0][gb0]
                    s4 = D[dep][ga0] + D[gb1][dep] - D[ga0][gb1]
                    for sv, ra, rb in ((s1, 0, 0), (s2, 0, 1), (s3, 1, 0), (s4, 1, 1)):
                        if best is None or sv > best[0]:
                            best = (sv, a, b, ra, rb)
            if best is None:
                break
            _, a, b, ra, rb = best
            na = routes[a][::-1] if ra else routes[a]
            nb = routes[b][::-1] if rb else routes[b]
            merged = na + nb
            routes = [routes[k] for k in range(R) if k != a and k != b]
            routes.append(merged)
        return routes

    def eliminate_to_k(dep, nodes, dems, Q, routes, K, cap_of_node=None):
        routes = [list(r) for r in routes]
        while len(routes) > K:
            rd = [sum(dems[p] for p in r) for r in routes]
            ri = min(range(len(routes)), key=lambda i: rd[i])
            removed = routes[ri]
            rest = routes[:ri] + routes[ri + 1:]
            rest_dem = rd[:ri] + rd[ri + 1:]
            ok = True
            for c in sorted(removed, key=lambda p: -dems[p]):
                cg = nodes[c]
                cd = dems[c]
                best = None
                for k in range(len(rest)):
                    if rest_dem[k] + cd > Q:
                        continue
                    if cap_of_node is not None:
                        ssum = 0
                        for p in rest[k]:
                            if nodes[p] == cg:
                                ssum += dems[p]
                        if ssum + cd > cap_of_node[cg]:
                            continue
                    r = rest[k]
                    pv = dep
                    for p in range(len(r) + 1):
                        nx = nodes[r[p]] if p < len(r) else dep
                        delta = D[pv][cg] + D[cg][nx] - D[pv][nx]
                        if best is None or delta < best[0]:
                            best = (delta, k, p)
                        if p < len(r):
                            pv = nodes[r[p]]
                if best is None:
                    ok = False
                    break
                _, k, p = best
                rest[k].insert(p, c)
                rest_dem[k] += cd
            if not ok:
                break
            routes = rest
        return routes

    def ffd_routes(nodes, dems, Q, cap_of_node=None):
        order = sorted(range(len(nodes)), key=lambda i: -dems[i])
        bins = []  # [remaining, [positions], {node: amt}]
        for i in order:
            d = dems[i]
            nd = nodes[i]
            bi = -1
            brem = None
            for k, b in enumerate(bins):
                if b[0] < d:
                    continue
                if cap_of_node is not None and b[2].get(nd, 0) + d > cap_of_node[nd]:
                    continue
                if brem is None or b[0] < brem:
                    brem = b[0]
                    bi = k
            if bi == -1:
                bins.append([Q - d, [i], {nd: d}])
            else:
                bins[bi][0] -= d
                bins[bi][1].append(i)
                bins[bi][2][nd] = bins[bi][2].get(nd, 0) + d
        return [b[1] for b in bins]

    def improve_heavy(dep, nodes, dems, Q, routes, deadline):
        routes = [list(r) for r in routes if r]
        if not routes:
            return 0.0, []
        improved = True
        while improved:
            improved = False
            if time.time() > deadline:
                break
            # Relocate move.
            R = len(routes)
            rd = [sum(dems[p] for p in r) for r in routes]
            moved = False
            for ri in range(R):
                if moved:
                    break
                p = 0
                while p < len(routes[ri]):
                    if time.time() > deadline:
                        moved = True
                        break
                    c = routes[ri][p]
                    cg = nodes[c]
                    r = routes[ri]
                    prev = dep if p == 0 else nodes[r[p - 1]]
                    nxt = dep if p == len(r) - 1 else nodes[r[p + 1]]
                    rem = D[prev][cg] + D[cg][nxt] - D[prev][nxt]
                    best = None
                    for rj in range(R):
                        if rj == ri or rd[rj] + dems[c] > Q:
                            continue
                        r2 = routes[rj]
                        pv = dep
                        for q in range(len(r2) + 1):
                            nx = nodes[r2[q]] if q < len(r2) else dep
                            ins = D[pv][cg] + D[cg][nx] - D[pv][nx]
                            if ins - rem < -1e-9 and (best is None or ins < best[0]):
                                best = (ins, rj, q)
                            if q < len(r2):
                                pv = nodes[r2[q]]
                    if best is not None:
                        _, rj, q = best
                        routes[ri].pop(p)
                        routes[rj].insert(q, c)
                        moved = True
                        improved = True
                        break
                    p += 1
            routes = [r for r in routes if r]
            if moved:
                continue
            if time.time() > deadline:
                break
            # Swap move.
            R = len(routes)
            rd = [sum(dems[p] for p in r) for r in routes]
            stop = False
            for ri in range(R):
                if stop:
                    break
                for rj in range(ri + 1, R):
                    if stop:
                        break
                    r1 = routes[ri]
                    r2 = routes[rj]
                    for p in range(len(r1)):
                        if stop:
                            break
                        c1 = r1[p]
                        g1 = nodes[c1]
                        for q in range(len(r2)):
                            c2 = r2[q]
                            g2 = nodes[c2]
                            if rd[ri] - dems[c1] + dems[c2] > Q:
                                continue
                            if rd[rj] - dems[c2] + dems[c1] > Q:
                                continue
                            pv1 = dep if p == 0 else nodes[r1[p - 1]]
                            nx1 = dep if p == len(r1) - 1 else nodes[r1[p + 1]]
                            pv2 = dep if q == 0 else nodes[r2[q - 1]]
                            nx2 = dep if q == len(r2) - 1 else nodes[r2[q + 1]]
                            delta = (D[pv1][g2] + D[g2][nx1] - D[pv1][g1] - D[g1][nx1]
                                     + D[pv2][g1] + D[g1][nx2] - D[pv2][g2] - D[g2][nx2])
                            if delta < -1e-9:
                                r1[p], r2[q] = c2, c1
                                rd[ri] += dems[c2] - dems[c1]
                                rd[rj] += dems[c1] - dems[c2]
                                improved = True
                                stop = True
                                break
                        if time.time() > deadline:
                            stop = True
                            break
                    if time.time() > deadline:
                        stop = True
                        break
        routes = [two_opt(dep, nodes, r) for r in routes if r]
        routes = [r for r in routes if r]
        cost = 0.0
        for r in routes:
            cost += route_dist(dep, nodes, r)
        return cost, routes

    # ------------------------------------------------------------------
    # Second echelon solver (cached).
    # ------------------------------------------------------------------
    sat_cache = {}   # (sid, frozenset(customer ids)) -> (cost, routes as id lists)
    sat_gs_cache = {}  # (sid, frozenset(group ids)) -> (cost, routes, load)

    def solve_second_light(sid, clist):
        if not clist:
            return 0.0, []
        nodes = [CUSTI[c] for c in clist]
        dems = [DEMC[c] for c in clist]
        K = MAXV[sid]
        dep = SATI[sid]
        routes = savings_build(dep, nodes, dems, Q2)
        if len(routes) > K:
            routes = merge_to_k(dep, nodes, dems, Q2, routes, K)
        if len(routes) > K:
            routes = eliminate_to_k(dep, nodes, dems, Q2, routes, K)
        if len(routes) > K:
            routes = ffd_routes(nodes, dems, Q2)
        routes = [two_opt(dep, nodes, r) for r in routes]
        cost = 0.0
        for r in routes:
            cost += route_dist(dep, nodes, r)
        return cost, [[clist[p] for p in r] for r in routes]

    def sat_eval(sid, clist):
        key = (sid, frozenset(clist))
        v = sat_cache.get(key)
        if v is None:
            v = solve_second_light(sid, clist)
            sat_cache[key] = v
        return v

    def sat_eval_groups(sid, gset):
        key = (sid, gset)
        v = sat_gs_cache.get(key)
        if v is None:
            cl = []
            for g in gset:
                cl.extend(GROUPS[g])
            cost, rts = sat_eval(sid, cl)
            L = sum(GDEM[g] for g in gset)
            v = (cost, rts, L)
            sat_gs_cache[key] = v
        return v

    # ------------------------------------------------------------------
    # First echelon solver (cached).
    # ------------------------------------------------------------------
    first_cache = {}

    def nn_two_opt(dep, gnodes):
        if len(gnodes) <= 1:
            return list(gnodes)
        remaining = set(range(len(gnodes)))
        order_pos = []
        cur = dep
        while remaining:
            nxt = min(remaining, key=lambda p: D[cur][gnodes[p]])
            remaining.remove(nxt)
            order_pos.append(nxt)
            cur = gnodes[nxt]
        order_pos = two_opt(dep, gnodes, order_pos)
        return [gnodes[p] for p in order_pos]

    def sweep_fill_first(items):
        routes = []
        cur = []
        load = 0
        cur_sid = None
        cur_sid_amt = 0
        for sid, q in items:
            qq = q
            while qq > 0:
                if load >= Q1:
                    routes.append(cur)
                    cur = []
                    load = 0
                    cur_sid = None
                    cur_sid_amt = 0
                if sid != cur_sid:
                    cur_sid = sid
                    cur_sid_amt = 0
                take = min(qq, Q1 - load, CAP[sid] - cur_sid_amt)
                if take <= 0:
                    routes.append(cur)
                    cur = []
                    load = 0
                    cur_sid = None
                    cur_sid_amt = 0
                    continue
                cur.append((sid, take))
                load += take
                cur_sid_amt += take
                qq -= take
        if cur:
            routes.append(cur)
        return routes

    def solve_first(loads):
        items = []
        for sid in sorted(loads):
            L = loads[sid]
            if L <= 0:
                continue
            mc = max(1, min(Q1, CAP[sid]))
            rem = L
            while rem > mc:
                items.append((sid, mc))
                rem -= mc
            if rem > 0:
                items.append((sid, rem))
        if not items:
            return 0.0, []
        nodes = [SATI[s] for s, _ in items]
        dems = [q for _, q in items]
        routes = savings_build(depot_i, nodes, dems, Q1, max_same=CAPNODE)
        if len(routes) > T1:
            routes = merge_to_k(depot_i, nodes, dems, Q1, routes, T1, cap_of_node=CAPNODE)
        if len(routes) > T1:
            routes = eliminate_to_k(depot_i, nodes, dems, Q1, routes, T1, cap_of_node=CAPNODE)
        if len(routes) > T1:
            pair_routes = sweep_fill_first(items)
        else:
            pair_routes = [[(items[p][0], items[p][1]) for p in r] for r in routes]
        result = []
        cost = 0.0
        for pr in pair_routes:
            deliv = {}
            for sid, q in pr:
                deliv[sid] = deliv.get(sid, 0) + q
            snodes = [SATI[sid] for sid in deliv]
            order = nn_two_opt(depot_i, snodes)
            cost += route_dist_g(depot_i, order)
            result.append((order, deliv))
        return cost, result

    def first_eval(loads):
        key = tuple(sorted((sid, loads[sid]) for sid in loads if loads[sid] > 0))
        v = first_cache.get(key)
        if v is None:
            v = solve_first(dict(key))
            first_cache[key] = v
        return v

    # ------------------------------------------------------------------
    # Full assignment evaluation.
    # ------------------------------------------------------------------
    def evaluate(assign, strict=True):
        bysat = {sid: [] for sid in sat_ids}
        for g, sid in assign.items():
            bysat[sid].append(g)
        routes2 = {}
        loads = {}
        total2 = 0.0
        handling = 0.0
        nroutes = 0
        for sid in sat_ids:
            gl = bysat[sid]
            if not gl:
                continue
            cost, rts, L = sat_eval_groups(sid, frozenset(gl))
            if strict and (L > CAP[sid] or len(rts) > MAXV[sid]):
                return None
            routes2[sid] = rts
            loads[sid] = L
            total2 += cost
            handling += HC[sid] * L
            nroutes += len(rts)
        if strict and nroutes > T2:
            return None
        c1, r1 = first_eval(loads)
        if strict and len(r1) > T1:
            return None
        return {"obj": c1 + handling + total2, "routes2": routes2,
                "loads": loads, "r1": r1}

    def build_output(assign, ev):
        fe = []
        for k, (order, deliv) in enumerate(ev["r1"]):
            fe.append({
                "vehicle": k + 1,
                "route": [depot_id] + [all_ids[g] for g in order] + [depot_id],
                "deliveries": {str(sid): deliv[sid] for sid in deliv},
            })
        se = []
        for sid in sat_ids:
            for r in ev["routes2"].get(sid, []):
                se.append({"satellite": sid, "route": [sid] + list(r) + [sid]})
        return {
            "objective_value": float(ev["obj"]),
            "first_echelon_routes": fe,
            "second_echelon_routes": se,
            "group_assignments": {str(g): assign[g] for g in group_ids},
        }

    incumbent = {"obj": float("inf"), "sol": None}

    def log_incumbent(assign, ev):
        if ev["obj"] < incumbent["obj"] - 1e-9:
            sol = build_output(assign, ev)
            incumbent["obj"] = ev["obj"]
            incumbent["sol"] = sol
            if logger:
                logger.log_solution(ev["obj"], sol)

    # ------------------------------------------------------------------
    # Assignment helpers.
    # ------------------------------------------------------------------
    def greedy_assign(est_cost, est_veh):
        assign = {}
        remcap = dict(CAP)
        remveh = dict(MAXV)
        totveh = 0
        for g in sorted(group_ids, key=lambda gg: -GDEM[gg]):
            chosen = None
            best_c = None
            for sid in sat_ids:
                if remcap[sid] >= GDEM[g] and remveh[sid] >= est_veh[g, sid] \
                        and totveh + est_veh[g, sid] <= T2:
                    if best_c is None or est_cost[g, sid] < best_c:
                        best_c = est_cost[g, sid]
                        chosen = sid
            if chosen is None:
                for sid in sat_ids:
                    if remcap[sid] >= GDEM[g]:
                        if best_c is None or est_cost[g, sid] < best_c:
                            best_c = est_cost[g, sid]
                            chosen = sid
            if chosen is None:
                chosen = min(sat_ids, key=lambda s: est_cost[g, s])
            assign[g] = chosen
            remcap[chosen] -= GDEM[g]
            remveh[chosen] -= est_veh[g, chosen]
            totveh += est_veh[g, chosen]
        return assign

    def mip_assign(est_cost, est_veh, tlim):
        import gurobipy as gp
        from gurobipy import GRB
        model = gp.Model("assign")
        model.Params.OutputFlag = 0
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1
        model.Params.TimeLimit = max(1.0, tlim)
        x = {}
        for g in group_ids:
            for sid in sat_ids:
                x[g, sid] = model.addVar(vtype=GRB.BINARY)
        model.setObjective(
            sum(est_cost[g, sid] * x[g, sid] for g in group_ids for sid in sat_ids),
            GRB.MINIMIZE)
        for g in group_ids:
            model.addConstr(sum(x[g, sid] for sid in sat_ids) == 1)
        for sid in sat_ids:
            model.addConstr(sum(GDEM[g] * x[g, sid] for g in group_ids) <= CAP[sid])
            model.addConstr(sum(est_veh[g, sid] * x[g, sid] for g in group_ids) <= MAXV[sid])
        model.addConstr(sum(est_veh[g, sid] * x[g, sid]
                            for g in group_ids for sid in sat_ids) <= T2)
        model.optimize()
        if model.SolCount > 0:
            assign = {}
            for g in group_ids:
                best_sid = sat_ids[0]
                best_val = -1.0
                for sid in sat_ids:
                    v = x[g, sid].X
                    if v > best_val:
                        best_val = v
                        best_sid = sid
                assign[g] = best_sid
            return assign
        return None

    def count_routes_feas(assign):
        bysat = {sid: [] for sid in sat_ids}
        for g, sid in assign.items():
            bysat[sid].append(g)
        tot = 0
        feas = True
        for sid in sat_ids:
            gl = bysat[sid]
            if not gl:
                continue
            cost, rts, L = sat_eval_groups(sid, frozenset(gl))
            if L > CAP[sid]:
                feas = False
            tot += len(rts)
            if len(rts) > MAXV[sid]:
                feas = False
        return tot, feas

    def repair(assign, est_cost, deadline):
        ev = evaluate(assign)
        tries = 0
        while ev is None and tries < 30 and time.time() < deadline:
            tries += 1
            base_tot, base_feas = count_routes_feas(assign)
            best = None
            for g in group_ids:
                if time.time() > deadline:
                    break
                s0 = assign[g]
                for sid in sat_ids:
                    if sid == s0:
                        continue
                    cand = dict(assign)
                    cand[g] = sid
                    tot, feas = count_routes_feas(cand)
                    better = (feas and not base_feas) or (feas == base_feas and tot < base_tot)
                    if better:
                        score = (1 if feas else 0, -tot, -est_cost[g, sid])
                        if best is None or score > best[0]:
                            best = (score, cand)
            if best is None:
                break
            assign = best[1]
            ev = evaluate(assign)
        return assign, ev

    def compute_loads_g(assign):
        lg = {sid: 0 for sid in sat_ids}
        for g, sid in assign.items():
            lg[sid] += GDEM[g]
        return lg

    def heavy_polish(assign, phase_deadline):
        bysat = {sid: [] for sid in sat_ids}
        for g, sid in assign.items():
            bysat[sid].append(g)
        used = [sid for sid in sat_ids if bysat[sid]]
        changed = False
        for k, sid in enumerate(used):
            now_t = time.time()
            if now_t >= phase_deadline:
                break
            share = phase_deadline - now_t
            per = now_t + share / max(1, len(used) - k)
            gkey = frozenset(bysat[sid])
            cl = []
            for g in gkey:
                cl.extend(GROUPS[g])
            cost0, rts0, L = sat_eval_groups(sid, gkey)
            nodes = [CUSTI[c] for c in cl]
            dems = [DEMC[c] for c in cl]
            pos = {CUSTI[c]: i for i, c in enumerate(cl)}
            rpos = [[pos[CUSTI[c]] for c in r] for r in rts0]
            c1, r1 = improve_heavy(SATI[sid], nodes, dems, Q2, rpos,
                                   min(per, phase_deadline))
            if c1 < cost0 - 1e-9:
                new_routes = [[cl[p] for p in r] for r in r1]
                sat_cache[(sid, frozenset(cl))] = (c1, new_routes)
                sat_gs_cache[(sid, gkey)] = (c1, new_routes, L)
                changed = True
        return changed

    def improvement_pass(assign, ev, loads_g, cand_order, deadline):
        gs = list(group_ids)
        random.shuffle(gs)
        for g in gs:
            if time.time() > deadline:
                return assign, ev, True
            s0 = assign[g]
            dg = GDEM[g]
            for sid in cand_order[g]:
                if sid == s0:
                    continue
                if loads_g[sid] + dg > CAP[sid]:
                    continue
                cand = dict(assign)
                cand[g] = sid
                evc = evaluate(cand)
                if evc is None:
                    continue
                if evc["obj"] < ev["obj"] - 1e-6:
                    loads_g[s0] -= dg
                    loads_g[sid] += dg
                    return cand, evc, True
        ng = len(group_ids)
        for i in range(ng):
            if time.time() > deadline:
                return assign, ev, True
            g1 = group_ids[i]
            s1 = assign[g1]
            d1 = GDEM[g1]
            for j in range(i + 1, ng):
                if time.time() > deadline:
                    return assign, ev, True
                g2 = group_ids[j]
                s2 = assign[g2]
                if s1 == s2:
                    continue
                d2 = GDEM[g2]
                if loads_g[s1] - d1 + d2 > CAP[s1]:
                    continue
                if loads_g[s2] - d2 + d1 > CAP[s2]:
                    continue
                cand = dict(assign)
                cand[g1] = s2
                cand[g2] = s1
                evc = evaluate(cand)
                if evc is None:
                    continue
                if evc["obj"] < ev["obj"] - 1e-6:
                    loads_g[s1] += d2 - d1
                    loads_g[s2] += d1 - d2
                    return cand, evc, True
        return assign, ev, False

    # ------------------------------------------------------------------
    # Main pipeline.
    # ------------------------------------------------------------------
    try:
        # Phase 1: cost estimates for every (group, satellite) pair.
        est_cost = {}
        est_veh = {}
        est_deadline = t0 + 0.25 * BUDGET
        use_proxy = False
        for g in group_ids:
            for sid in sat_ids:
                if not use_proxy and time.time() > est_deadline:
                    use_proxy = True
                if not use_proxy:
                    c0, r0 = sat_eval(sid, GROUPS[g])
                    veh = max(1, len(r0))
                    rc = c0
                else:
                    si = SATI[sid]
                    sd = 0.0
                    for c in GROUPS[g]:
                        sd += D[si][CUSTI[c]]
                    k = max(1, int(math.ceil(GDEM[g] / float(Q2))))
                    veh = k
                    rc = 2.0 * sd / k
                est_cost[g, sid] = (rc + HC[sid] * GDEM[g]
                                    + 2.0 * D[depot_i][SATI[sid]] * GDEM[g] / float(Q1))
                est_veh[g, sid] = veh

        cand_order = {g: sorted(sat_ids, key=lambda sid: (est_cost[g, sid], sid))
                      for g in group_ids}

        # Phase 2: initial group-to-satellite assignment.
        assign = None
        tlim = min(30.0, 0.2 * BUDGET)
        if (t0 + BUDGET - time.time()) > tlim + 3:
            try:
                assign = mip_assign(est_cost, est_veh, tlim)
            except Exception:
                assign = None
        if assign is None:
            assign = greedy_assign(est_cost, est_veh)

        assign, ev = repair(assign, est_cost, t0 + 0.45 * BUDGET)
        if ev is None:
            ev = evaluate(assign, strict=False)

        # Phase 3: local search over group assignments.
        reserve = max(2.0, 0.12 * BUDGET)
        ls_deadline = t0 + BUDGET - reserve

        heavy_polish(assign, min(ls_deadline, time.time() + max(1.0, 0.08 * BUDGET)))
        ev = evaluate(assign)
        if ev is None:
            ev = evaluate(assign, strict=False)
        best_assign = dict(assign)
        best_ev = ev
        log_incumbent(best_assign, best_ev)

        loads_g = compute_loads_g(assign)
        stagnant = 0
        while time.time() < ls_deadline:
            assign, ev, improved = improvement_pass(assign, ev, loads_g,
                                                    cand_order, ls_deadline)
            if improved:
                if ev["obj"] < best_ev["obj"] - 1e-9:
                    best_assign = dict(assign)
                    best_ev = ev
                    log_incumbent(best_assign, best_ev)
                    stagnant = 0
            else:
                stagnant += 1
                if stagnant > 6:
                    break
                # Perturbation: restart from incumbent and apply random kicks.
                assign = dict(best_assign)
                ev = best_ev
                loads_g = compute_loads_g(assign)
                nk = 1 + stagnant
                done_kicks = 0
                for _ in range(60):
                    if done_kicks >= nk:
                        break
                    g = random.choice(group_ids)
                    sid = random.choice(sat_ids)
                    if sid == assign[g]:
                        continue
                    if loads_g[sid] + GDEM[g] > CAP[sid]:
                        continue
                    cand = dict(assign)
                    cand[g] = sid
                    evc = evaluate(cand)
                    if evc is None:
                        continue
                    loads_g[assign[g]] -= GDEM[g]
                    loads_g[sid] += GDEM[g]
                    assign = cand
                    ev = evc
                    done_kicks += 1
                if done_kicks == 0:
                    break

        # Phase 4: final heavy polish + improvement passes.
        final_deadline = t0 + BUDGET - 0.5
        while time.time() < final_deadline:
            changed = heavy_polish(best_assign, final_deadline)
            ev2 = evaluate(best_assign)
            if ev2 is None:
                ev2 = evaluate(best_assign, strict=False)
            improved_any = False
            if ev2["obj"] < best_ev["obj"] - 1e-9:
                best_ev = ev2
                log_incumbent(best_assign, best_ev)
                improved_any = True
            if time.time() < final_deadline:
                lg = compute_loads_g(best_assign)
                na, ne, imp = improvement_pass(best_assign, best_ev, lg,
                                               cand_order, final_deadline)
                if imp and ne["obj"] < best_ev["obj"] - 1e-9:
                    best_assign = dict(na)
                    best_ev = ne
                    log_incumbent(best_assign, best_ev)
                    improved_any = True
            if not (changed or improved_any):
                break

        if incumbent["sol"] is None:
            log_incumbent(best_assign, best_ev)
        sol = incumbent["sol"]
    except Exception:
        sol = incumbent["sol"]
        if sol is None:
            # Emergency fallback: simple capacity-respecting assignment.
            try:
                assign = {}
                rem = dict(CAP)
                for g in group_ids:
                    placed = None
                    for sid in sat_ids:
                        if rem.get(sid, 0) >= GDEM[g]:
                            placed = sid
                            break
                    if placed is None:
                        placed = sat_ids[0]
                    assign[g] = placed
                    rem[placed] = rem.get(placed, 0) - GDEM[g]
                ev = evaluate(assign, strict=False)
                sol = build_output(assign, ev)
            except Exception:
                sol = {
                    "objective_value": float("inf"),
                    "first_echelon_routes": [],
                    "second_echelon_routes": [],
                    "group_assignments": {str(g): sat_ids[0] for g in group_ids},
                }

    with open(args.solution_path, "w") as f:
        json.dump(sol, f)


if __name__ == "__main__":
    main()
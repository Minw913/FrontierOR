import argparse
import json
import random
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--instance_path', required=True)
    ap.add_argument('--solution_path', required=True)
    ap.add_argument('--time_limit', type=int, required=True)
    ap.add_argument('--log_path', default=None)
    args = ap.parse_args()

    t0 = time.time()
    T = float(args.time_limit)
    deadline = t0 + T
    MARGIN = 0.6
    random.seed(0)

    with open(args.instance_path, 'r') as f:
        inst = json.load(f)

    logger = None
    last_logged = [float('inf')]
    last_log_t = [0.0]
    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize")

    S = int(inst['num_delivery_sites'])
    C = int(inst['num_customers'])
    Q = float(inst['vehicle_capacity'])
    V = int(inst['num_vehicles'])
    site_ids = [int(s['id']) for s in inst['delivery_sites']]
    cust_ids = [int(c['id']) for c in inst['customers']]
    dem = [float(c['demand']) for c in inst['customers']]
    rcm = inst['routing_cost_matrix']
    acm = inst['assignment_cost_matrix']
    depot_id = int(inst['depot']['id'])

    # internal node numbering: 0 = depot, 1..S = sites (position in list)
    node_id = [depot_id] + site_ids
    row_of = {depot_id: 0}
    for pos, sid in enumerate(site_ids, start=1):
        row_of[sid] = pos
    Cmat = [[float(rcm[row_of[node_id[i]]][row_of[node_id[j]]]) for j in range(S + 1)]
            for i in range(S + 1)]
    ac = [[float(acm[i][s]) for s in range(S)] for i in range(C)]

    cand_k = min(S, 30)
    cand_sites = [sorted(range(S), key=lambda s: ac[i][s])[:cand_k] for i in range(C)]

    INF = float('inf')

    # ---------------- TSP helpers ----------------
    tsp_memo = {}

    def tour_cost(order):
        tot = Cmat[0][order[0]]
        for t in range(len(order) - 1):
            tot += Cmat[order[t]][order[t + 1]]
        tot += Cmat[order[-1]][0]
        return tot

    def held_karp(sites):
        m = len(sites)
        size = 1 << m
        dp = [[INF] * m for _ in range(size)]
        par = [[-1] * m for _ in range(size)]
        for i in range(m):
            dp[1 << i][i] = Cmat[0][sites[i]]
        for mask in range(size):
            dprow = dp[mask]
            for j in range(m):
                cj = dprow[j]
                if cj == INF:
                    continue
                sj = sites[j]
                crowsj = Cmat[sj]
                for l in range(m):
                    if not ((mask >> l) & 1):
                        nm = mask | (1 << l)
                        v = cj + crowsj[sites[l]]
                        if v < dp[nm][l] - 1e-12:
                            dp[nm][l] = v
                            par[nm][l] = j
        full = size - 1
        best = INF
        bj = 0
        for j in range(m):
            v = dp[full][j] + Cmat[sites[j]][0]
            if v < best:
                best = v
                bj = j
        order = []
        mask = full
        j = bj
        while j != -1:
            order.append(sites[j])
            pj = par[mask][j]
            mask ^= (1 << j)
            j = pj
        order.reverse()
        return (best, tuple(order))

    def two_opt_route(sites):
        m = len(sites)
        remset = set(sites)
        order = []
        cur = 0
        while remset:
            nxt = min(remset, key=lambda s: Cmat[cur][s])
            order.append(nxt)
            remset.discard(nxt)
            cur = nxt
        for _ in range(60):
            improved = False
            for i in range(m - 1):
                a0 = order[i - 1] if i > 0 else 0
                b0 = order[i]
                cab = Cmat[a0][b0]
                for j in range(i + 1, m):
                    c0 = order[j]
                    d0 = order[j + 1] if j + 1 < m else 0
                    if Cmat[a0][c0] + Cmat[b0][d0] < cab + Cmat[c0][d0] - 1e-9:
                        order[i:j + 1] = order[i:j + 1][::-1]
                        improved = True
            for i in range(m):
                nd = order[i]
                p0 = order[i - 1] if i > 0 else 0
                n0 = order[i + 1] if i + 1 < m else 0
                base = Cmat[p0][n0] - Cmat[p0][nd] - Cmat[nd][n0]
                rest = order[:i] + order[i + 1:]
                mm = m - 1
                best_pos = -1
                best_d = -1e-9
                for q in range(mm + 1):
                    a2 = rest[q - 1] if q > 0 else 0
                    b2 = rest[q] if q < mm else 0
                    d = base + Cmat[a2][nd] + Cmat[nd][b2] - Cmat[a2][b2]
                    if d < best_d:
                        best_d = d
                        best_pos = q
                if best_pos >= 0:
                    order = rest[:best_pos] + [nd] + rest[best_pos:]
                    improved = True
            if not improved:
                break
        return (tour_cost(order), tuple(order))

    def solve_tsp(nodes_sorted):
        res = tsp_memo.get(nodes_sorted)
        if res is None:
            m = len(nodes_sorted)
            if m == 0:
                res = (0.0, ())
            elif m == 1:
                s = nodes_sorted[0]
                res = (Cmat[0][s] + Cmat[s][0], nodes_sorted)
            elif m <= 8:
                res = held_karp(nodes_sorted)
            else:
                res = two_opt_route(nodes_sorted)
            if len(tsp_memo) > 300000:
                tsp_memo.clear()
            tsp_memo[nodes_sorted] = res
        return res

    def ins_delta(nd, order):
        m = len(order)
        bd = INF
        for p in range(m + 1):
            a_ = order[p - 1] if p > 0 else 0
            b_ = order[p] if p < m else 0
            dd = Cmat[a_][nd] + Cmat[nd][b_] - Cmat[a_][b_]
            if dd < bd:
                bd = dd
        return bd

    # ---------------- state ----------------
    assign = [-1] * C
    site_dem = [0.0] * S
    site_customers = [set() for _ in range(S)]
    routes = []                 # list of sorted tuples of internal nodes (1..S)
    site_route = [-1] * (S + 1)
    route_dem = []
    route_costs = []
    assign_cost = [0.0]

    best_obj = [INF]
    best_snap = [None]

    def set_route(r, nodes_sorted):
        routes[r] = nodes_sorted
        route_costs[r] = solve_tsp(nodes_sorted)[0]
        route_dem[r] = 0.0
        for nd in nodes_sorted:
            route_dem[r] += site_dem[nd - 1]

    def detach_node(nd):
        r = site_route[nd]
        if r < 0:
            return
        site_route[nd] = -1
        newset = tuple(x for x in routes[r] if x != nd)
        set_route(r, newset)

    def attach_node(nd, r):
        site_route[nd] = r
        newset = tuple(sorted(routes[r] + (nd,)))
        set_route(r, newset)

    def refresh_dem(r):
        s = 0.0
        for nd in routes[r]:
            s += site_dem[nd - 1]
        route_dem[r] = s

    def active_routes():
        cnt = 0
        for rt in routes:
            if rt:
                cnt += 1
        return cnt

    def find_slot():
        for r in range(len(routes)):
            if not routes[r]:
                return r
        routes.append(())
        route_dem.append(0.0)
        route_costs.append(0.0)
        return len(routes) - 1

    def apply_move(mv):
        t = mv[0]
        if t == 0:
            _, i, b, rb = mv
            a_ = assign[i]
            di = dem[i]
            na = a_ + 1
            nb = b + 1
            assign_cost[0] += ac[i][b] - ac[i][a_]
            assign[i] = b
            site_customers[a_].discard(i)
            site_customers[b].add(i)
            site_dem[a_] -= di
            site_dem[b] += di
            if site_dem[a_] <= 1e-9:
                detach_node(na)
            if site_route[nb] >= 0:
                refresh_dem(site_route[nb])
            if site_route[na] >= 0:
                refresh_dem(site_route[na])
        elif t == 1:
            _, i, b, target = mv
            a_ = assign[i]
            di = dem[i]
            na = a_ + 1
            nb = b + 1
            assign_cost[0] += ac[i][b] - ac[i][a_]
            assign[i] = b
            site_customers[a_].discard(i)
            site_customers[b].add(i)
            site_dem[a_] -= di
            site_dem[b] += di
            if site_dem[a_] <= 1e-9:
                detach_node(na)
            elif site_route[na] >= 0:
                refresh_dem(site_route[na])
            if target is None or target < 0:
                target = find_slot()
            attach_node(nb, target)
        elif t == 2:
            _, nd, r2 = mv
            detach_node(nd)
            if r2 is None or r2 < 0:
                r2 = find_slot()
            attach_node(nd, r2)
        elif t == 3:
            _, n1, n2 = mv
            r1 = site_route[n1]
            r2 = site_route[n2]
            detach_node(n1)
            detach_node(n2)
            attach_node(n1, r2)
            attach_node(n2, r1)
        elif t == 4:
            _, s, plan = mv
            for (i, b) in plan:
                apply_move((0, i, b, site_route[b + 1]))
            ns = s + 1
            if site_route[ns] >= 0:
                detach_node(ns)
        elif t == 5:
            _, i, ip = mv
            a_, b = assign[i], assign[ip]
            assign_cost[0] += (ac[i][b] - ac[i][a_]) + (ac[ip][a_] - ac[ip][b])
            assign[i], assign[ip] = b, a_
            site_customers[a_].discard(i)
            site_customers[a_].add(ip)
            site_customers[b].discard(ip)
            site_customers[b].add(i)
            site_dem[a_] += dem[ip] - dem[i]
            site_dem[b] += dem[i] - dem[ip]
            refresh_dem(site_route[a_ + 1])
            refresh_dem(site_route[b + 1])

    def snapshot():
        return (assign[:], site_dem[:], [set(sc) for sc in site_customers],
                routes[:], site_route[:], route_dem[:], route_costs[:], assign_cost[0])

    def restore(snap):
        assign[:] = snap[0]
        site_dem[:] = snap[1]
        for idx in range(S):
            site_customers[idx] = set(snap[2][idx])
        routes[:] = list(snap[3])
        site_route[:] = snap[4]
        route_dem[:] = snap[5]
        route_costs[:] = snap[6]
        assign_cost[0] = snap[7]

    def build_solution():
        routes_out = {}
        veh_of_route = {}
        veh = 0
        for r in range(len(routes)):
            if not routes[r]:
                continue
            order = solve_tsp(routes[r])[1]
            seq = [node_id[0]] + [node_id[nd] for nd in order] + [node_id[0]]
            arcs = [[seq[t], seq[t + 1]] for t in range(len(seq) - 1)]
            routes_out[str(veh)] = {"arcs": arcs, "sequence": seq}
            veh_of_route[r] = veh
            veh += 1
        assignments = {}
        cust_tours = {}
        tot_a = 0.0
        for i in range(C):
            s = assign[i]
            sid = site_ids[s]
            assignments[str(cust_ids[i])] = sid
            cust_tours[str(cust_ids[i])] = {"vehicle": str(veh_of_route[site_route[s + 1]]),
                                            "site": sid}
            tot_a += ac[i][s]
        obj = tot_a + sum(route_costs)
        return obj, {"objective_value": obj, "routes": routes_out,
                     "assignments": assignments, "customer_tours": cust_tours}

    def maybe_record(force=False):
        obj = assign_cost[0] + sum(route_costs)
        if obj < best_obj[0] - 1e-9:
            best_obj[0] = obj
            best_snap[0] = snapshot()
            if logger is not None and (force or time.time() - last_log_t[0] > 0.3):
                o, sol = build_solution()
                logger.log_solution(o, sol)
                last_logged[0] = o
                last_log_t[0] = time.time()

    # ---------------- construction ----------------
    def build_routes_savings():
        for s in range(S):
            if site_dem[s] > 1e-9:
                nd = s + 1
                r = find_slot()
                site_route[nd] = r
                set_route(r, (nd,))
        sav = []
        for a_ in range(1, S + 1):
            if site_route[a_] < 0:
                continue
            for b_ in range(a_ + 1, S + 1):
                if site_route[b_] < 0:
                    continue
                sv = Cmat[0][a_] + Cmat[b_][0] - Cmat[a_][b_]
                sav.append((sv, a_, b_))
        sav.sort(reverse=True)
        for sv, a_, b_ in sav:
            ra = site_route[a_]
            rb = site_route[b_]
            if ra == rb or ra < 0 or rb < 0:
                continue
            if route_dem[ra] + route_dem[rb] > Q + 1e-9:
                continue
            oa = list(solve_tsp(routes[ra])[1])
            ob = list(solve_tsp(routes[rb])[1])
            if oa[-1] != a_:
                if oa[0] == a_:
                    oa.reverse()
                else:
                    continue
            if ob[0] != b_:
                if ob[-1] == b_:
                    ob.reverse()
                else:
                    continue
            merged = tuple(sorted(routes[ra] + routes[rb]))
            for nd in routes[rb]:
                site_route[nd] = ra
            set_route(ra, merged)
            set_route(rb, ())

    def pack_routes_bfd():
        open_nodes = [nd for nd in range(1, S + 1) if site_route[nd] >= 0]
        for nd in open_nodes:
            site_route[nd] = -1
        routes.clear()
        route_dem.clear()
        route_costs.clear()
        for nd in sorted(open_nodes, key=lambda x: -site_dem[x - 1]):
            best_r = -1
            best_res = INF
            for r in range(len(routes)):
                res = Q - route_dem[r]
                if site_dem[nd - 1] <= res + 1e-9 and res < best_res:
                    best_res = res
                    best_r = r
            if best_r < 0:
                best_r = find_slot()
            attach_node(nd, best_r)

    def redistribute_site(s):
        custs = sorted(site_customers[s], key=lambda i: dem[i])
        resid = {r: Q - route_dem[r] for r in range(len(routes)) if routes[r]}
        rs = site_route[s + 1]
        if rs >= 0:
            resid[rs] = Q - route_dem[rs] + site_dem[s]
        plan = []
        for i in custs:
            di = dem[i]
            bb = -1
            bd = INF
            for b in range(S):
                if b == s or site_dem[b] <= 1e-9:
                    continue
                rb = site_route[b + 1]
                if resid.get(rb, 0.0) < di - 1e-9:
                    continue
                dd = ac[i][b]
                if dd < bd:
                    bd = dd
                    bb = b
            if bb < 0:
                return False
            resid[site_route[bb + 1]] -= di
            plan.append((i, bb))
        for (i, b) in plan:
            apply_move((0, i, b, site_route[b + 1]))
        return True

    def repair_vehicle_limit():
        guard = 0
        while active_routes() > V and guard < 1000:
            guard += 1
            cand = min((r for r in range(len(routes)) if routes[r]),
                       key=lambda r: route_dem[r])
            for nd in sorted(routes[cand], key=lambda x: site_dem[x - 1]):
                best_r = -1
                best_d = INF
                for r2 in range(len(routes)):
                    if r2 == cand or not routes[r2]:
                        continue
                    if route_dem[r2] + site_dem[nd - 1] <= Q + 1e-9:
                        d = ins_delta(nd, solve_tsp(routes[r2])[1])
                        if d < best_d:
                            best_d = d
                            best_r = r2
                if best_r >= 0:
                    detach_node(nd)
                    attach_node(nd, best_r)
                else:
                    ok = redistribute_site(nd - 1)
                    if ok and site_route[nd] >= 0:
                        detach_node(nd)
            if routes[cand]:
                break

    def construct():
        for i in range(C):
            assign[i] = -1
        for s in range(S):
            site_dem[s] = 0.0
            site_customers[s] = set()
        routes.clear()
        route_dem.clear()
        route_costs.clear()
        for nd in range(S + 1):
            site_route[nd] = -1
        order_i = sorted(range(C), key=lambda i: (-dem[i], i))
        for i in order_i:
            di = dem[i]
            best_s = -1
            best_c = INF
            rowi = ac[i]
            for s in range(S):
                if site_dem[s] + di <= Q + 1e-9:
                    c = rowi[s]
                    if c < best_c:
                        best_c = c
                        best_s = s
            if best_s < 0:
                best_s = max(range(S), key=lambda s: Q - site_dem[s])
            assign[i] = best_s
            site_dem[best_s] += di
            site_customers[best_s].add(i)
        # fix over-capacity sites at assignment level
        for s in range(S):
            while site_dem[s] > Q + 1e-9:
                best = None
                for i in list(site_customers[s]):
                    for b in range(S):
                        if b == s:
                            continue
                        if site_dem[b] + dem[i] <= Q + 1e-9:
                            dd = ac[i][b] - ac[i][s]
                            if best is None or dd < best[0]:
                                best = (dd, i, b)
                if best is None:
                    break
                _, i, b = best
                site_customers[s].discard(i)
                site_customers[b].add(i)
                site_dem[s] -= dem[i]
                site_dem[b] += dem[i]
                assign[i] = b
        build_routes_savings()
        if active_routes() > V:
            pack_routes_bfd()
            repair_vehicle_limit()
        assign_cost[0] = sum(ac[i][assign[i]] for i in range(C))

    # ---------------- local search ----------------
    def descent(ts_limit):
        while time.time() < ts_limit:
            active_rs = [r for r in range(len(routes)) if routes[r]]
            n_active = len(active_rs)
            can_new = n_active < V
            orders = {}
            rem_info = {}
            for r in active_rs:
                o = list(solve_tsp(routes[r])[1])
                orders[r] = o
                m = len(o)
                for pos in range(m):
                    nd = o[pos]
                    p0 = o[pos - 1] if pos > 0 else 0
                    n0 = o[pos + 1] if pos + 1 < m else 0
                    if m > 1:
                        delta = Cmat[p0][n0] - Cmat[p0][nd] - Cmat[nd][n0]
                    else:
                        delta = -route_costs[r]
                    rem_info[(r, nd)] = (delta, o[:pos] + o[pos + 1:])
            open_insert = {}
            for nb in range(1, S + 1):
                if site_route[nb] >= 0:
                    continue
                opts = []
                for r in active_rs:
                    opts.append((ins_delta(nb, orders[r]), r))
                opts.sort()
                open_insert[nb] = opts

            best_delta = -1e-6
            best_move = None
            time_up = False

            # 1) customer relocation
            for i in range(C):
                if (i & 127) == 0 and time.time() >= ts_limit:
                    time_up = True
                    break
                a_ = assign[i]
                di = dem[i]
                na = a_ + 1
                ra = site_route[na]
                a_closes = (site_dem[a_] <= di + 1e-9)
                base_removal = rem_info[(ra, na)][0] if a_closes else 0.0
                aci = ac[i]
                base_ai = aci[a_]
                for b in cand_sites[i]:
                    if b == a_:
                        continue
                    nb = b + 1
                    dA = aci[b] - base_ai
                    rb = site_route[nb]
                    if rb >= 0:
                        if rb != ra and route_dem[rb] + di > Q + 1e-9:
                            continue
                        delta = dA + base_removal
                        if delta < best_delta:
                            best_delta = delta
                            best_move = (0, i, b, rb)
                    else:
                        chosen = None
                        for (dd, r) in open_insert[nb]:
                            if r == ra and a_closes:
                                continue
                            if di <= Q - route_dem[r] + 1e-9:
                                chosen = (dd, r)
                                break
                        if a_closes and ra >= 0:
                            o_wo = rem_info[(ra, na)][1]
                            if di <= Q - route_dem[ra] + di + 1e-9:
                                dd = ins_delta(nb, o_wo)
                                if chosen is None or dd < chosen[0]:
                                    chosen = (dd, ra)
                        if can_new:
                            dd = Cmat[0][nb] + Cmat[nb][0]
                            if chosen is None or dd < chosen[0]:
                                chosen = (dd, -1)
                        if chosen is None:
                            continue
                        delta = dA + base_removal + chosen[0]
                        if delta < best_delta:
                            best_delta = delta
                            best_move = (1, i, b, chosen[1])
            if time_up:
                if best_move is not None:
                    apply_move(best_move)
                    maybe_record()
                break

            # 2) site relocation between routes
            for r1 in active_rs:
                for nd in routes[r1]:
                    sd = site_dem[nd - 1]
                    rd1, o1 = rem_info[(r1, nd)]
                    for r2 in active_rs:
                        if r2 == r1:
                            continue
                        if route_dem[r2] + sd > Q + 1e-9:
                            continue
                        d = rd1 + ins_delta(nd, orders[r2])
                        if d < best_delta:
                            best_delta = d
                            best_move = (2, nd, r2)
                    if can_new:
                        d = rd1 + Cmat[0][nd] + Cmat[nd][0]
                        if d < best_delta:
                            best_delta = d
                            best_move = (2, nd, -1)

            # 3) site swaps between routes
            for i1 in range(len(active_rs)):
                r1 = active_rs[i1]
                for i2 in range(i1 + 1, len(active_rs)):
                    r2 = active_rs[i2]
                    for n1 in routes[r1]:
                        d1 = site_dem[n1 - 1]
                        rd1, o1 = rem_info[(r1, n1)]
                        for n2 in routes[r2]:
                            d2 = site_dem[n2 - 1]
                            if route_dem[r1] - d1 + d2 > Q + 1e-9:
                                continue
                            if route_dem[r2] - d2 + d1 > Q + 1e-9:
                                continue
                            rd2, o2 = rem_info[(r2, n2)]
                            d = rd1 + rd2 + ins_delta(n1, o2) + ins_delta(n2, o1)
                            if d < best_delta:
                                best_delta = d
                                best_move = (3, n1, n2)

            # 4) customer swaps
            top = {}
            for i in range(C):
                a_ = assign[i]
                base_ai = ac[i][a_]
                rowi = ac[i]
                for b in cand_sites[i]:
                    if b == a_:
                        continue
                    g = base_ai - rowi[b]
                    if g <= 1e-9:
                        continue
                    key = (a_, b)
                    lst = top.get(key)
                    if lst is None:
                        top[key] = [(g, i)]
                    else:
                        lst.append((g, i))
            for key in top:
                lst = top[key]
                lst.sort(reverse=True)
                del lst[4:]
            open_list = [s for s in range(S) if site_dem[s] > 1e-9]
            for ai in range(len(open_list)):
                a_ = open_list[ai]
                ra = site_route[a_ + 1]
                for bi in range(ai + 1, len(open_list)):
                    b_ = open_list[bi]
                    rb = site_route[b_ + 1]
                    l1 = top.get((a_, b_))
                    l2 = top.get((b_, a_))
                    if not l1 or not l2:
                        continue
                    for (g1, i) in l1:
                        for (g2, ip) in l2:
                            if ra != rb:
                                if route_dem[rb] + dem[i] - dem[ip] > Q + 1e-9:
                                    continue
                                if route_dem[ra] + dem[ip] - dem[i] > Q + 1e-9:
                                    continue
                            d = -(g1 + g2)
                            if d < best_delta:
                                best_delta = d
                                best_move = (5, i, ip)

            # 5) site closing
            for s in range(S):
                if site_dem[s] <= 1e-9:
                    continue
                ns = s + 1
                rs = site_route[ns]
                if rs < 0:
                    continue
                resid = {r: Q - route_dem[r] for r in active_rs}
                resid[rs] = Q - route_dem[rs] + site_dem[s]
                newset = tuple(x for x in routes[rs] if x != ns)
                delta = solve_tsp(newset)[0] - route_costs[rs]
                plan = []
                ok = True
                for i in sorted(site_customers[s], key=lambda i: dem[i]):
                    di = dem[i]
                    bb = -1
                    bd = INF
                    rowi = ac[i]
                    base_i = rowi[s]
                    for b in range(S):
                        if b == s or site_dem[b] <= 1e-9:
                            continue
                        rb2 = site_route[b + 1]
                        if resid.get(rb2, 0.0) < di - 1e-9:
                            continue
                        dd = rowi[b] - base_i
                        if dd < bd:
                            bd = dd
                            bb = b
                    if bb < 0:
                        ok = False
                        break
                    resid[site_route[bb + 1]] -= di
                    delta += bd
                    plan.append((i, bb))
                if ok and delta < best_delta:
                    best_delta = delta
                    best_move = (4, s, plan)

            if best_move is None:
                break
            apply_move(best_move)
            maybe_record()

    def perturb(strength):
        for _ in range(strength):
            if random.random() < 0.65:
                i = random.randrange(C)
                a_ = assign[i]
                b = random.randrange(S)
                if b == a_:
                    continue
                di = dem[i]
                na = a_ + 1
                nb = b + 1
                ra = site_route[na]
                rb = site_route[nb]
                if rb >= 0:
                    if rb != ra and route_dem[rb] + di > Q + 1e-9:
                        continue
                    apply_move((0, i, b, rb))
                else:
                    cands = [r for r in range(len(routes))
                             if routes[r] and route_dem[r] + di <= Q + 1e-9]
                    if cands:
                        apply_move((1, i, b, random.choice(cands)))
                    elif active_routes() < V:
                        apply_move((1, i, b, -1))
            else:
                open_nodes = [nd for nd in range(1, S + 1) if site_route[nd] >= 0]
                if not open_nodes:
                    continue
                nd = random.choice(open_nodes)
                r1 = site_route[nd]
                sd = site_dem[nd - 1]
                cands = [r for r in range(len(routes))
                         if routes[r] and r != r1 and route_dem[r] + sd <= Q + 1e-9]
                if cands:
                    apply_move((2, nd, random.choice(cands)))
                elif active_routes() < V:
                    apply_move((2, nd, -1))

    # ---------------- MIP (compact formulation) ----------------
    def run_mip(time_left):
        import gurobipy as gp
        from gurobipy import GRB

        model = gp.Model("lrp")
        p = model.Params
        p.Seed = 0
        p.MIPGap = 1e-4
        p.NumericFocus = 0
        p.Threads = 1
        p.OutputFlag = 0
        p.TimeLimit = max(1.0, time_left)

        nodes = list(range(S + 1))
        Js = list(range(1, S + 1))
        Ks = list(range(V))
        arc_index = [(uu, vv, k) for k in Ks for uu in nodes for vv in nodes if uu != vv]
        a = model.addVars(arc_index, vtype=GRB.BINARY, name='a')
        z = model.addVars([(j, k) for j in Js for k in Ks], vtype=GRB.BINARY, name='z')
        y = model.addVars([(i, j) for i in range(C) for j in Js], vtype=GRB.BINARY, name='y')
        w = model.addVars([(j, k) for j in Js for k in Ks], lb=0.0, ub=Q, name='w')
        u = model.addVars([(j, k) for j in Js for k in Ks], lb=0.0, ub=Q, name='u')
        model.setObjective(
            gp.quicksum(Cmat[uu][vv] * a[uu, vv, k] for (uu, vv, k) in arc_index)
            + gp.quicksum(ac[i][j - 1] * y[i, j] for i in range(C) for j in Js),
            GRB.MINIMIZE)
        model.update()

        model.addConstrs([y.sum(i, '*') == 1 for i in range(C)], name='assign')
        model.addConstrs([y[i, j] <= z.sum(j, '*') for i in range(C) for j in Js], name='yz')
        model.addConstrs([z.sum(j, '*') <= 1 for j in Js], name='z1')
        model.addConstrs([gp.quicksum(a[0, vv, k] for vv in Js) <= 1 for k in Ks], name='dep')
        model.addConstrs([gp.quicksum(a[uu, j, k] for uu in nodes if uu != j)
                          == gp.quicksum(a[j, vv, k] for vv in nodes if vv != j)
                          for k in Ks for j in nodes], name='flow')
        model.addConstrs([gp.quicksum(a[uu, j, k] for uu in nodes if uu != j) == z[j, k]
                          for j in Js for k in Ks], name='zin')
        sd_expr = {j: gp.quicksum(dem[i] * y[i, j] for i in range(C)) for j in Js}
        model.addConstrs([w[j, k] >= sd_expr[j] - Q * (1 - z[j, k])
                          for j in Js for k in Ks], name='wlink')
        model.addConstrs([w.sum('*', k) <= Q for k in Ks], name='cap')
        model.addConstrs([u[j, k] >= w[j, k] for j in Js for k in Ks], name='uw')
        model.addConstrs([u[j, k] <= Q * gp.quicksum(a[j, vv, k] for vv in nodes if vv != j)
                          for j in Js for k in Ks], name='uQ')
        mtz = []
        for k in Ks:
            for uu in nodes:
                for vv in Js:
                    if uu == vv:
                        continue
                    lhs = (u[uu, k] if uu != 0 else 0.0) - u[vv, k] + Q * a[uu, vv, k] + w[vv, k]
                    mtz.append(lhs <= Q)
        model.addConstrs(mtz, name='mtz')
        model.addConstrs([z.sum('*', k) >= z.sum('*', k + 1) for k in range(V - 1)], name='sym')
        model.update()

        # warm start from current (best) state
        veh_of_route = {}
        veh = 0
        for r in range(len(routes)):
            if routes[r]:
                veh_of_route[r] = veh
                veh += 1
        for r, k in veh_of_route.items():
            order = solve_tsp(routes[r])[1]
            seq = [0] + list(order) + [0]
            for t in range(len(seq) - 1):
                a[seq[t], seq[t + 1], k].Start = 1.0
            cum = 0.0
            for nd in order:
                z[nd, k].Start = 1.0
                cum += site_dem[nd - 1]
                w[nd, k].Start = site_dem[nd - 1]
                u[nd, k].Start = cum
        for i in range(C):
            y[i, assign[i] + 1].Start = 1.0
        model.update()

        a_var_list = [a[key] for key in arc_index]
        y_keys_list = [(i, j) for i in range(C) for j in Js]
        y_var_list = [y[key] for key in y_keys_list]

        def extract(getter):
            avals = getter(a_var_list)
            yvals = getter(y_var_list)
            routes_out = {}
            site_veh = {}
            veh = 0
            arcs_by_k = {}
            for idx in range(len(arc_index)):
                if avals[idx] > 0.5:
                    uu, vv, k = arc_index[idx]
                    arcs_by_k.setdefault(k, []).append((uu, vv))
            obj = 0.0
            for k in sorted(arcs_by_k):
                nxt = {}
                for (uu, vv) in arcs_by_k[k]:
                    nxt[uu] = vv
                    obj += Cmat[uu][vv]
                if 0 not in nxt:
                    continue
                seq = [0]
                cur = 0
                guard = 0
                while cur in nxt and guard <= S + 1:
                    cur = nxt[cur]
                    seq.append(cur)
                    if cur == 0:
                        break
                    guard += 1
                if seq[-1] != 0:
                    continue
                ext_seq = [node_id[nd] for nd in seq]
                arcs_out = [[ext_seq[t], ext_seq[t + 1]] for t in range(len(ext_seq) - 1)]
                routes_out[str(veh)] = {"arcs": arcs_out, "sequence": ext_seq}
                for nd in seq[1:-1]:
                    site_veh[nd] = veh
                veh += 1
            assignments = {}
            cust_tours = {}
            for idx in range(len(y_keys_list)):
                if yvals[idx] > 0.5:
                    i, j = y_keys_list[idx]
                    sid = site_ids[j - 1]
                    assignments[str(cust_ids[i])] = sid
                    cust_tours[str(cust_ids[i])] = {"vehicle": str(site_veh.get(j, 0)),
                                                    "site": sid}
                    obj += ac[i][j - 1]
            sol = {"objective_value": obj, "routes": routes_out,
                   "assignments": assignments, "customer_tours": cust_tours}
            return obj, sol

        mip_best = [None]
        last_inc = [INF]

        def cb(model, where):
            if where == GRB.Callback.MIPSOL:
                try:
                    obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                except Exception:
                    return
                lim = min(best_obj[0], last_inc[0])
                if obj < lim - 1e-6:
                    try:
                        o2, sol = extract(model.cbGetSolution)
                    except Exception:
                        return
                    last_inc[0] = obj
                    mip_best[0] = (o2, sol)
                    if logger is not None and o2 < last_logged[0] - 1e-9:
                        logger.log_solution(o2, sol)
                        last_logged[0] = o2
                        last_log_t[0] = time.time()

        model.optimize(cb)

        if model.SolCount > 0:
            try:
                o2, sol = extract(lambda vl: [v.X for v in vl])
                if mip_best[0] is None or o2 < mip_best[0][0] - 1e-9:
                    mip_best[0] = (o2, sol)
                    if logger is not None and o2 < last_logged[0] - 1e-9:
                        logger.log_solution(o2, sol)
                        last_logged[0] = o2
            except Exception:
                pass
        return mip_best[0]

    # ---------------- main flow ----------------
    construct()
    maybe_record(force=True)

    xcount = C * S * V
    arccount = (S + 1) * S * V
    use_mip = (T >= 8.0 and S <= 40 and xcount <= 30000 and arccount <= 40000 and C > 0)
    if use_mip:
        ils_end = t0 + min(0.30 * T, 20.0)
        ils_end = max(ils_end, t0 + min(3.0, 0.5 * T))
    else:
        ils_end = deadline - MARGIN

    descent(ils_end)
    maybe_record(force=True)
    restore(best_snap[0])

    fails = 0
    while time.time() < ils_end:
        pre = best_obj[0]
        perturb(3 + min(fails // 3, 5))
        descent(ils_end)
        if best_obj[0] < pre - 1e-9:
            fails = 0
        else:
            fails += 1
        restore(best_snap[0])

    mip_res = None
    if use_mip:
        tl = deadline - MARGIN - time.time()
        if tl > 2.0:
            restore(best_snap[0])
            try:
                mip_res = run_mip(tl)
            except Exception:
                mip_res = None
    if mip_res is None or mip_res[0] >= best_obj[0] - 1e-9:
        end2 = deadline - MARGIN
        restore(best_snap[0])
        while time.time() < end2:
            pre = best_obj[0]
            perturb(3 + min(fails // 3, 5))
            descent(end2)
            if best_obj[0] < pre - 1e-9:
                fails = 0
            else:
                fails += 1
            restore(best_snap[0])

    if mip_res is not None and mip_res[0] < best_obj[0] - 1e-9:
        final_obj, final_sol = mip_res
    else:
        restore(best_snap[0])
        final_obj, final_sol = build_solution()

    if logger is not None and final_obj < last_logged[0] - 1e-9:
        logger.log_solution(final_obj, final_sol)
        last_logged[0] = final_obj

    with open(args.solution_path, 'w') as f:
        json.dump(final_sol, f)


if __name__ == '__main__':
    main()
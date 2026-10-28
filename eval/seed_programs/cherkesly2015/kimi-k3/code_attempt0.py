import argparse
import json
import math
import random
import time

INF = float('inf')
EPS = 1e-9


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    logger = None
    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize")

    t_start = time.time()
    time_limit = float(args.time_limit)
    hard_end = t_start + time_limit
    if time_limit > 8.0:
        mip_reserve = max(2.0, min(25.0, 0.12 * time_limit))
        polish_budget = max(0.5, min(5.0, 0.05 * time_limit))
    else:
        mip_reserve = 1.0
        polish_budget = 0.3
    alns_end = hard_end - mip_reserve - polish_budget - 0.2

    random.seed(0)

    with open(args.instance_path) as f:
        data = json.load(f)

    N = int(data['n_nodes'])
    Q = float(data['vehicle_capacity'])
    fixed_cost = float(data.get('vehicle_fixed_cost', 0) or 0.0)

    load = [0.0] * N
    serv = [0.0] * N
    twe = [0.0] * N
    twl = [0.0] * N
    is_pickup = [False] * N
    is_delivery = [False] * N
    pair = [-1] * N
    req_pick = {}
    req_del = {}
    depot0 = 0
    depotT = N - 1
    for node in data['nodes']:
        i = int(node['node_id'])
        load[i] = float(node['load'])
        serv[i] = float(node['service_time'])
        twe[i] = float(node['tw_early'])
        twl[i] = float(node['tw_late'])
        nt = node['node_type']
        if nt == 'depot_origin':
            depot0 = i
        elif nt == 'depot_destination':
            depotT = i
        elif nt == 'pickup':
            is_pickup[i] = True
            req_pick[node['request_id']] = i
        elif nt == 'delivery':
            is_delivery[i] = True
            req_del[node['request_id']] = i

    requests = sorted(req_pick.keys())
    nreq = len(requests)
    req_p = [req_pick[r] for r in requests]
    req_d = [req_del[r] for r in requests]
    req_q = [load[req_pick[r]] for r in requests]
    node_to_ridx = [0] * N
    for idx in range(nreq):
        node_to_ridx[req_p[idx]] = idx
        node_to_ridx[req_d[idx]] = idx
    for r in requests:
        pair[req_pick[r]] = req_del[r]
        pair[req_del[r]] = req_pick[r]

    C = [[INF] * N for _ in range(N)]
    Tm = [[INF] * N for _ in range(N)]
    for arc in data['arcs']:
        u = int(arc['from'])
        v = int(arc['to'])
        C[u][v] = float(arc['cost'])
        Tm[u][v] = float(arc['travel_time'])

    # ---------------- route evaluation ----------------
    def compute_rd(seq):
        C_l = C; T_l = Tm; twe_l = twe; twl_l = twl
        serv_l = serv; load_l = load
        m = len(seq)
        rt = [0.0] * m
        wait = [0.0] * m
        load_a = [0.0] * m
        depth = [0] * m
        rt[0] = twe_l[seq[0]]
        load_a[0] = load_l[seq[0]]
        stack = []
        prev = seq[0]
        cost_total = 0.0
        for k in range(1, m):
            v = seq[k]
            c = C_l[prev][v]
            if c == INF:
                return None
            a = rt[k - 1] + serv_l[prev] + T_l[prev][v]
            st = a if a > twe_l[v] else twe_l[v]
            if st > twl_l[v] + EPS:
                return None
            l = load_a[k - 1] + load_l[v]
            if l < -EPS or l > Q + EPS:
                return None
            if is_pickup[v]:
                stack.append(v)
            elif is_delivery[v]:
                if not stack or stack[-1] != pair[v]:
                    return None
                stack.pop()
            rt[k] = st
            wait[k] = st - a
            load_a[k] = l
            depth[k] = len(stack)
            cost_total += c
            prev = v
        if stack:
            return None
        slack = [0.0] * m
        prefWait = [0.0] * m
        pw = 0.0
        for k in range(m):
            slack[k] = twl_l[seq[k]] - rt[k]
            pw += wait[k]
            prefWait[k] = pw
        suffMin = [0.0] * m
        run = INF
        for k in range(m - 1, -1, -1):
            val = slack[k] + prefWait[k]
            if val < run:
                run = val
            suffMin[k] = run
        return [rt, wait, load_a, depth, slack, prefWait, suffMin, cost_total]

    # ---------------- best (cost) feasible insertion of request r into route ----------------
    def best_insertion(seq, rd, r):
        C_l = C; T_l = Tm; twe_l = twe; twl_l = twl; serv_l = serv
        p = req_p[r]; d = req_d[r]; q = req_q[r]
        m = len(seq)
        rt = rd[0]; wait = rd[1]; load_a = rd[2]; depth = rd[3]
        slack = rd[4]; prefWait = rd[5]; suffMin = rd[6]
        sp = serv_l[p]; sd = serv_l[d]
        ep = twe_l[p]; lp = twl_l[p]; ed = twe_l[d]; ld = twl_l[d]
        Cp = C_l[p]; Cd = C_l[d]; Tp = T_l[p]; Td = T_l[d]
        best_delta = INF; best_i = -1; best_j = -1
        for i in range(1, m):
            previ = seq[i - 1]
            nxt = seq[i]
            c_pp = C_l[previ][p]
            if c_pp == INF:
                continue
            t_pp = T_l[previ][p]
            if t_pp == INF:
                continue
            arr_p = rt[i - 1] + serv_l[previ] + t_pp
            start_p = arr_p if arr_p > ep else ep
            if start_p > lp + EPS:
                continue
            base_depth = depth[i - 1]
            base_load = load_a[i - 1]
            if base_load + q > Q + EPS:
                continue
            # case j == i: pickup and delivery adjacent in gap i
            c_pd = Cp[d]; t_pd = Tp[d]
            if c_pd != INF and t_pd != INF:
                c_dv = Cd[nxt]; t_dv = Td[nxt]
                if c_dv != INF and t_dv != INF:
                    arr_d = start_p + sp + t_pd
                    start_d = arr_d if arr_d > ed else ed
                    if start_d <= ld + EPS:
                        arr_v = start_d + sd + t_dv
                        nsv = arr_v if arr_v > twe_l[nxt] else twe_l[nxt]
                        dj = nsv - rt[i]
                        if dj < 0.0:
                            dj = 0.0
                        if dj <= suffMin[i] - prefWait[i] + 1e-7:
                            dc = c_pp + c_pd + c_dv - C_l[previ][nxt]
                            if dc < best_delta:
                                best_delta = dc; best_i = i; best_j = i
            # cases j > i
            c_pu = Cp[nxt]; t_pu = Tp[nxt]
            if c_pu == INF or t_pu == INF:
                continue
            arr_u = start_p + sp + t_pu
            nsu = arr_u if arr_u > twe_l[nxt] else twe_l[nxt]
            di = nsu - rt[i]
            if di < 0.0:
                di = 0.0
            rem_i = C_l[previ][nxt]
            minSlackW = INF
            minDepth = 1 << 30
            maxLoad = base_load
            dk = di
            pw_i = prefWait[i]
            add_i = c_pp + c_pu
            for j in range(i + 1, m):
                k = j - 1
                val = slack[k] + prefWait[k]
                if val < minSlackW:
                    minSlackW = val
                dpk = depth[k]
                if dpk < minDepth:
                    minDepth = dpk
                lk = load_a[k]
                if lk > maxLoad:
                    maxLoad = lk
                if minDepth < base_depth:
                    break
                if maxLoad + q > Q + EPS:
                    break
                if di > minSlackW - pw_i + 1e-7:
                    break
                nk = seq[k]
                c_kd = C_l[nk][d]; t_kd = T_l[nk][d]
                if c_kd != INF and t_kd != INF:
                    v = seq[j]
                    c_dv = Cd[v]; t_dv = Td[v]
                    if c_dv != INF and t_dv != INF:
                        arr_d = rt[k] + dk + serv_l[nk] + t_kd
                        start_d = arr_d if arr_d > ed else ed
                        if start_d <= ld + EPS:
                            arr_v = start_d + sd + t_dv
                            nsv = arr_v if arr_v > twe_l[v] else twe_l[v]
                            dj = nsv - rt[j]
                            if dj < 0.0:
                                dj = 0.0
                            if dj <= suffMin[j] - prefWait[j] + 1e-7:
                                dc = add_i + c_kd + c_dv - rem_i - C_l[nk][v]
                                if dc < best_delta:
                                    best_delta = dc; best_i = i; best_j = j
                dk = dk - wait[j]
                if dk < 0.0:
                    dk = 0.0
        if best_i < 0:
            return None
        return (best_delta, best_i, best_j)

    def apply_insertion(seq, r, i, j):
        return seq[:i] + [req_p[r]] + seq[i:j] + [req_d[r]] + seq[j:]

    def add_new_route(routes, rds, r):
        seq = [depot0, req_p[r], req_d[r], depotT]
        rd = compute_rd(seq)
        if rd is None:
            return False
        routes.append(seq)
        rds.append(rd)
        return True

    def do_remove(routes, rds, removed):
        rem_nodes = set()
        for r in removed:
            rem_nodes.add(req_p[r])
            rem_nodes.add(req_d[r])
        new_routes = []
        new_rds = []
        for seq in routes:
            nseq = [x for x in seq if x not in rem_nodes]
            if len(nseq) > 2:
                nrd = compute_rd(nseq)
                if nrd is not None:
                    new_routes.append(nseq)
                    new_rds.append(nrd)
        routes[:] = new_routes
        rds[:] = new_rds

    def repair(routes, rds, removed, allow_new, use_regret):
        removed = list(removed)
        if not removed:
            return True
        if not use_regret:
            random.shuffle(removed)
            for r in removed:
                best = None
                best_rt = -1
                for ri in range(len(routes)):
                    res = best_insertion(routes[ri], rds[ri], r)
                    if res is not None and (best is None or res[0] < best[0]):
                        best = res
                        best_rt = ri
                if best is None:
                    if not allow_new:
                        return False
                    if not add_new_route(routes, rds, r):
                        return False
                else:
                    nseq = apply_insertion(routes[best_rt], r, best[1], best[2])
                    nrd = compute_rd(nseq)
                    if nrd is None:
                        return False
                    routes[best_rt] = nseq
                    rds[best_rt] = nrd
            return True
        # regret-2 repair
        while removed:
            chosen = None
            no_route = None
            for r in removed:
                cands = []
                for ri in range(len(routes)):
                    res = best_insertion(routes[ri], rds[ri], r)
                    if res is not None:
                        cands.append((res[0], ri, res[1], res[2]))
                if not cands:
                    no_route = r
                    continue
                cands.sort(key=lambda x: x[0])
                b = cands[0]
                second = cands[1][0] if len(cands) > 1 else 1e18
                regret = second - b[0]
                if chosen is None or regret > chosen[0]:
                    chosen = (regret, r, b[1], b[2], b[3])
            if chosen is None:
                if not allow_new:
                    return False
                if not add_new_route(routes, rds, no_route):
                    return False
                removed.remove(no_route)
                continue
            _, r, ri, i, j = chosen
            nseq = apply_insertion(routes[ri], r, i, j)
            nrd = compute_rd(nseq)
            if nrd is None:
                return False
            routes[ri] = nseq
            rds[ri] = nrd
            removed.remove(r)
        return True

    # ---------------- destroy operators ----------------
    def all_requests(routes):
        out = []
        for seq in routes:
            for node in seq:
                if is_pickup[node]:
                    out.append(node_to_ridx[node])
        return out

    def destroy_random(routes, rds, k):
        reqs = all_requests(routes)
        k = min(k, len(reqs))
        removed = random.sample(reqs, k)
        do_remove(routes, rds, removed)
        return removed

    def destroy_shaw(routes, rds, k):
        reqs = all_requests(routes)
        if not reqs:
            return []
        k = min(k, len(reqs))
        r0 = random.choice(reqs)
        p0 = req_p[r0]; d0 = req_d[r0]
        ep0 = twe[p0]; ed0 = twe[d0]

        def rel(r):
            a = C[p0][req_p[r]]
            b = C[d0][req_d[r]]
            if a == INF:
                a = 1e12
            if b == INF:
                b = 1e12
            return a + b + abs(ep0 - twe[req_p[r]]) + abs(ed0 - twe[req_d[r]])

        removed = sorted(reqs, key=rel)[:k]
        do_remove(routes, rds, removed)
        return removed

    def destroy_worst(routes, rds, k):
        gains = []
        for seq in routes:
            pos = {}
            for idx, node in enumerate(seq):
                pos[node] = idx
            m = len(seq)
            for idx in range(1, m - 1):
                node = seq[idx]
                if not is_pickup[node]:
                    continue
                r = node_to_ridx[node]
                d = pair[node]
                idd = pos[d]
                prevp = seq[idx - 1]; nextp = seq[idx + 1]
                prevd = seq[idd - 1]; nextd = seq[idd + 1]
                g1 = C[prevp][node] + C[node][nextp]
                c1 = C[prevp][nextp]
                if c1 != INF:
                    g1 -= c1
                g2 = C[prevd][d] + C[d][nextd]
                c2 = C[prevd][nextd]
                if c2 != INF:
                    g2 -= c2
                gains.append((g1 + g2, r))
        gains.sort(key=lambda x: x[0], reverse=True)
        k = min(k, len(gains))
        removed = [r for _, r in gains[:k]]
        do_remove(routes, rds, removed)
        return removed

    def destroy_smallest_route(routes, rds):
        if len(routes) < 2:
            return None
        ri = min(range(len(routes)), key=lambda i: (len(routes[i]), -rds[i][7]))
        removed = [node_to_ridx[node] for node in routes[ri] if is_pickup[node]]
        do_remove(routes, rds, removed)
        return removed

    def eliminate_pass(routes, rds, deadline):
        while len(routes) >= 2 and time.time() < deadline:
            ri = min(range(len(routes)), key=lambda i: (len(routes[i]), -rds[i][7]))
            removed = [node_to_ridx[node] for node in routes[ri] if is_pickup[node]]
            backup = [list(s) for s in routes]
            do_remove(routes, rds, removed)
            if not repair(routes, rds, removed, allow_new=False, use_regret=True):
                routes[:] = backup
                rds[:] = [compute_rd(s) for s in routes]
                break

    def build_solution(route_list):
        sol_routes = []
        total = 0.0
        for vid, seq in enumerate(route_list):
            rd = compute_rd(seq)
            if rd is None:
                continue
            details = []
            for idx, node in enumerate(seq):
                details.append({
                    'node_id': int(node),
                    'arrival_time': float(rd[0][idx]),
                    'load_after': float(rd[2][idx]),
                })
            sol_routes.append({
                'vehicle_id': int(vid),
                'route_sequence': [int(x) for x in seq],
                'route_details': details,
            })
            total += rd[7]
        return {'objective_value': float(total), 'routes': sol_routes}

    # ---------------- construction ----------------
    routes = []
    rds = []
    order = sorted(range(nreq), key=lambda r: (twe[req_p[r]], twl[req_p[r]]))
    for r in order:
        best = None
        best_rt = -1
        for ri in range(len(routes)):
            res = best_insertion(routes[ri], rds[ri], r)
            if res is not None and (best is None or res[0] < best[0]):
                best = res
                best_rt = ri
        if best is None:
            add_new_route(routes, rds, r)
        else:
            nseq = apply_insertion(routes[best_rt], r, best[1], best[2])
            nrd = compute_rd(nseq)
            if nrd is not None:
                routes[best_rt] = nseq
                rds[best_rt] = nrd
            else:
                add_new_route(routes, rds, r)

    if nreq > 0:
        elim_deadline = min(alns_end, time.time() + max(1.0, 0.1 * max(alns_end - t_start, 1.0)))
        eliminate_pass(routes, rds, elim_deadline)

    cur_total = sum(rd[7] for rd in rds)
    best_routes = [list(s) for s in routes]
    best_total = cur_total
    if logger:
        try:
            logger.log_solution(best_total, build_solution(best_routes))
        except Exception:
            logger.log(best_total)

    # ---------------- ALNS ----------------
    pool = {}

    def add_to_pool(rts, rds_):
        if len(pool) >= 10000:
            return
        for seq, rd in zip(rts, rds_):
            key = frozenset(node_to_ridx[node] for node in seq if is_pickup[node])
            old = pool.get(key)
            if old is None or rd[7] < old[0]:
                pool[key] = (rd[7], tuple(seq))

    try:
        if nreq > 0:
            add_to_pool(routes, rds)
            travel0 = cur_total - fixed_cost * len(routes)
            T0 = 0.05 * max(travel0, 1.0)
            Tmin = max(T0 * 1e-4, 1e-9)
            alns_start = time.time()
            alns_budget = max(alns_end - alns_start, 0.1)
            T = T0
            d_w = [1.0, 1.0, 1.0, 0.6]
            r_w = [1.0, 1.0]
            d_score = [0.0] * 4
            d_use = [0] * 4
            r_score = [0.0] * 2
            r_use = [0] * 2
            it = 0
            since_best = 0
            maxK = max(2, min(20, nreq // 4 if nreq >= 8 else max(nreq, 1)))

            while time.time() < alns_end:
                it += 1
                sw = sum(d_w); x = random.random() * sw; acc = 0.0; di = 3
                for a in range(4):
                    acc += d_w[a]
                    if x <= acc:
                        di = a
                        break
                sw = sum(r_w); x = random.random() * sw; acc = 0.0; rop = 1
                for a in range(2):
                    acc += r_w[a]
                    if x <= acc:
                        rop = a
                        break

                backup = [list(s) for s in routes]
                if di == 3:
                    removed = destroy_smallest_route(routes, rds)
                    if removed is None:
                        continue
                    ok = repair(routes, rds, removed, allow_new=False, use_regret=True)
                else:
                    k = random.randint(1, maxK)
                    if di == 0:
                        removed = destroy_random(routes, rds, k)
                    elif di == 1:
                        removed = destroy_shaw(routes, rds, k)
                    else:
                        removed = destroy_worst(routes, rds, k)
                    ok = repair(routes, rds, removed, allow_new=True, use_regret=(rop == 1))
                if not ok:
                    routes[:] = backup
                    rds[:] = [compute_rd(s) for s in routes]
                    continue
                new_total = 0.0
                for rd in rds:
                    new_total += rd[7]
                delta = new_total - cur_total
                if delta < -1e-9:
                    accepted = True
                    d_score[di] += 9.0; r_score[rop] += 9.0
                elif delta < 1e-9:
                    accepted = True
                    d_score[di] += 3.0; r_score[rop] += 3.0
                elif random.random() < math.exp(-delta / max(T, 1e-9)):
                    accepted = True
                    d_score[di] += 13.0; r_score[rop] += 13.0
                else:
                    accepted = False
                if accepted:
                    cur_total = new_total
                    add_to_pool(routes, rds)
                    if new_total < best_total - 1e-9:
                        best_total = new_total
                        best_routes = [list(s) for s in routes]
                        d_score[di] += 33.0; r_score[rop] += 33.0
                        since_best = 0
                        if logger:
                            try:
                                logger.log_solution(best_total, build_solution(best_routes))
                            except Exception:
                                logger.log(best_total)
                    else:
                        since_best += 1
                else:
                    routes[:] = backup
                    rds[:] = [compute_rd(s) for s in routes]
                    since_best += 1
                d_use[di] += 1
                r_use[rop] += 1
                frac = (time.time() - alns_start) / alns_budget
                if frac > 1.0:
                    frac = 1.0
                T = T0 * (Tmin / T0) ** frac
                if since_best > 2500:
                    T = max(T, 0.3 * T0)
                    since_best = 0
                if it % 50 == 0:
                    for a in range(4):
                        if d_use[a]:
                            d_w[a] = d_w[a] * 0.7 + 0.3 * (d_score[a] / d_use[a])
                    for a in range(2):
                        if r_use[a]:
                            r_w[a] = r_w[a] * 0.7 + 0.3 * (r_score[a] / r_use[a])
                    d_score = [0.0] * 4
                    d_use = [0] * 4
                    r_score = [0.0] * 2
                    r_use = [0] * 2
    except Exception:
        pass

    # ---------------- final polish (best-improvement relocate) ----------------
    def polish(routes, rds, deadline):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            req_route = [-1] * nreq
            for rix, seq in enumerate(routes):
                for node in seq:
                    if is_pickup[node]:
                        req_route[node_to_ridx[node]] = rix
            for r in range(nreq):
                if time.time() >= deadline:
                    return
                rix = req_route[r]
                if rix < 0:
                    continue
                seq = routes[rix]
                old_cost = rds[rix][7]
                p = req_p[r]; d = req_d[r]
                nseq = [x for x in seq if x != p and x != d]
                if len(nseq) > 2:
                    nrd = compute_rd(nseq)
                    if nrd is None:
                        continue
                    rem_cost = nrd[7]
                else:
                    nrd = None
                    rem_cost = 0.0
                rem_delta = rem_cost - old_cost
                best = None
                best_rt = -1
                for rj in range(len(routes)):
                    if rj == rix:
                        if nrd is None:
                            continue
                        res = best_insertion(nseq, nrd, r)
                    else:
                        res = best_insertion(routes[rj], rds[rj], r)
                    if res is not None and (best is None or res[0] < best[0]):
                        best = res
                        best_rt = rj
                if best is None:
                    continue
                if rem_delta + best[0] < -1e-9:
                    if best_rt == rix:
                        routes[rix] = apply_insertion(nseq, r, best[1], best[2])
                        rds[rix] = compute_rd(routes[rix])
                    else:
                        routes[best_rt] = apply_insertion(routes[best_rt], r, best[1], best[2])
                        rds[best_rt] = compute_rd(routes[best_rt])
                        if nrd is None:
                            routes.pop(rix)
                            rds.pop(rix)
                        else:
                            routes[rix] = nseq
                            rds[rix] = nrd
                    req_route = [-1] * nreq
                    for a, sq in enumerate(routes):
                        for node in sq:
                            if is_pickup[node]:
                                req_route[node_to_ridx[node]] = a
                    improved = True

    try:
        if nreq > 0:
            polish_deadline = hard_end - mip_reserve - 0.1
            routes = [list(s) for s in best_routes]
            rds = [compute_rd(s) for s in routes]
            polish(routes, rds, polish_deadline)
            ptotal = sum(rd[7] for rd in rds)
            if ptotal < best_total - 1e-9:
                best_total = ptotal
                best_routes = [list(s) for s in routes]
                if logger:
                    try:
                        logger.log_solution(best_total, build_solution(best_routes))
                    except Exception:
                        logger.log(best_total)
            add_to_pool(best_routes, [compute_rd(s) for s in best_routes])
    except Exception:
        pass

    # ---------------- set-partitioning exact re-optimization over route pool ----------------
    def run_set_partition(pool, t_remaining):
        try:
            import gurobipy as gp
            from gurobipy import GRB
        except Exception:
            return None
        try:
            items = list(pool.items())
            model = gp.Model("spp")
            model.Params.Seed = 0
            model.Params.MIPGap = 1e-4
            model.Params.NumericFocus = 0
            model.Params.Threads = 1
            model.Params.OutputFlag = 0
            model.Params.TimeLimit = max(1.0, t_remaining)
            xvars = []
            for idx, (key, (cst, seq)) in enumerate(items):
                xvars.append(model.addVar(vtype=GRB.BINARY, obj=float(cst)))
            model.update()
            cover = [[] for _ in range(nreq)]
            for idx, (key, _) in enumerate(items):
                for r in key:
                    cover[r].append(xvars[idx])
            for r in range(nreq):
                if not cover[r]:
                    return None
                model.addConstr(gp.quicksum(cover[r]) == 1.0)
            model.optimize()
            if model.SolCount > 0:
                chosen = [idx for idx in range(len(items)) if xvars[idx].X > 0.5]
                routes_out = [list(items[idx][1][1]) for idx in chosen]
                return routes_out
        except Exception:
            return None
        return None

    try:
        t_now = time.time()
        remaining = hard_end - t_now - 0.3
        if remaining > 1.5 and nreq > 0 and pool:
            routes_mip = run_set_partition(pool, remaining)
            if routes_mip is not None:
                cnt = [0] * nreq
                total_mip = 0.0
                ok = True
                for seq in routes_mip:
                    rd = compute_rd(seq)
                    if rd is None:
                        ok = False
                        break
                    total_mip += rd[7]
                    for node in seq:
                        if is_pickup[node]:
                            cnt[node_to_ridx[node]] += 1
                if ok and all(c == 1 for c in cnt) and total_mip < best_total - 1e-6:
                    best_total = total_mip
                    best_routes = routes_mip
                    if logger:
                        try:
                            logger.log_solution(best_total, build_solution(best_routes))
                        except Exception:
                            logger.log(best_total)
    except Exception:
        pass

    # ---------------- output ----------------
    try:
        sol = build_solution(best_routes)
    except Exception:
        # fallback: one route per request
        fallback = [[depot0, req_p[r], req_d[r], depotT] for r in range(nreq)]
        sol = build_solution(fallback)
    with open(args.solution_path, 'w') as f:
        json.dump(sol, f)


if __name__ == '__main__':
    main()
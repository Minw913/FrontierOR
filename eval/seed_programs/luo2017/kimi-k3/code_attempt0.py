import argparse
import json
import math
import random
import time

try:
    from solution_logger import SolutionLogger
except Exception:
    SolutionLogger = None

EPS = 1e-9
COST_EPS = 1e-6


def solve(data, time_limit, logger):
    rng = random.Random(0)
    start_time = time.perf_counter()
    end_time = start_time + max(0.0, float(time_limit))
    # Time reserved at the very end for final polish + writing output.
    polish_time = max(0.4, 0.15 * time_limit)
    alns_end = end_time - polish_time - 0.2
    if alns_end < start_time:
        alns_end = start_time

    n = data['num_customers']
    Q = data['vehicle_capacity']
    A = float(data['cost_function']['a'])
    B = float(data['cost_function']['b'])
    depot = data['depot']
    DEP = depot['id']
    NV = n + 1  # virtual depot entrance for output
    D = data['distance_matrix']
    T = data.get('time_matrix') or data.get('travel_time_matrix') or D
    tw0 = depot['time_window']
    E0 = tw0[0]
    L0 = min(tw0[1], data.get('scheduling_horizon', tw0[1]))

    customers = data['customers']
    ids = [c['id'] for c in customers]
    max_id = max([DEP] + ids) if ids else DEP
    E = [0.0] * (max_id + 1)
    L = [0.0] * (max_id + 1)
    S = [0.0] * (max_id + 1)
    DEM = [0] * (max_id + 1)
    for c in customers:
        i = c['id']
        DEM[i] = c['demand']
        E[i] = c['time_window'][0]
        L[i] = c['time_window'][1]
        S[i] = c['service_time']

    DMAX = max((max(row) for row in D), default=1.0) or 1.0
    if ids:
        span = max(L[i] for i in ids) - min(E[i] for i in ids)
    else:
        span = 1.0
    SPAN = span if span > 0 else 1.0

    # ------------------------------------------------------------------
    # Route evaluation: cost = sum_e dist_e * (A * load_e + B); also checks
    # time-window feasibility assuming departure from depot at E0 (waiting
    # at customers is allowed, so an earliest-schedule check is exact).
    # ------------------------------------------------------------------
    def eval_route(seq, delv):
        load = 0.0
        for q in delv.values():
            load += q
        prev = DEP
        t = E0
        sprev = 0.0
        cost = 0.0
        Dr = D[prev]
        Tr = T[prev]
        for i in seq:
            d = Dr[i]
            cost += d * (A * load + B)
            arr = t + sprev + Tr[i]
            ei = E[i]
            st = arr if arr > ei else ei
            if st > L[i] + 1e-9:
                return cost, False
            t = st
            sprev = S[i]
            load -= delv[i]
            prev = i
            Dr = D[i]
            Tr = T[i]
        cost += Dr[DEP] * (A * load + B)
        if t + sprev + Tr[DEP] > L0 + 1e-9:
            return cost, False
        return cost, True

    def make_route(seq, delv):
        c, _ = eval_route(seq, delv)
        return {'seq': list(seq), 'del': dict(delv), 'cost': c,
                'load': sum(delv.values())}

    def clone(routes):
        return [{'seq': r['seq'][:], 'del': dict(r['del']),
                 'cost': r['cost'], 'load': r['load']} for r in routes]

    def total_cost(routes):
        s = 0.0
        for r in routes:
            s += r['cost']
        return s

    # ------------------------------------------------------------------
    # Insertion machinery
    # ------------------------------------------------------------------
    def insertion_options(routes, i, qrem):
        opts = []
        for idx in range(len(routes)):
            r = routes[idx]
            if i in r['del']:
                continue
            res = Q - r['load']
            if res <= EPS:
                continue
            q = qrem if qrem < res else res
            seq = r['seq']
            base = r['cost']
            nd = dict(r['del'])
            nd[i] = q
            for pos in range(len(seq) + 1):
                ns = seq[:pos] + [i] + seq[pos:]
                c, f = eval_route(ns, nd)
                if f:
                    opts.append((c - base, idx, pos, q))
        q = qrem if qrem < Q else Q
        c, f = eval_route([i], {i: q})
        if f:
            opts.append((c, -1, 0, q))
        return opts

    def apply_opt(routes, i, opt, remaining, touched=None):
        _, idx, pos, q = opt
        if idx == -1:
            routes.append(make_route([i], {i: q}))
            if touched is not None:
                touched.add(len(routes) - 1)
        else:
            r = routes[idx]
            r['seq'].insert(pos, i)
            r['del'][i] = q
            r['load'] += q
            c, _ = eval_route(r['seq'], r['del'])
            r['cost'] = c
            if touched is not None:
                touched.add(idx)
        nrem = remaining[i] - q
        if nrem <= EPS:
            remaining.pop(i, None)
        else:
            remaining[i] = nrem

    def repair(routes, remaining, use_regret):
        cust_list = [i for i, q in remaining.items() if q > EPS]
        if use_regret and len(cust_list) > 1:
            key = {}
            for i in cust_list:
                opts = insertion_options(routes, i, remaining[i])
                if opts:
                    opts.sort(key=lambda x: x[0])
                    reg = (opts[1][0] - opts[0][0]) if len(opts) > 1 else 1e18
                    key[i] = -reg
                else:
                    key[i] = -1e18
            cust_list.sort(key=lambda i: (key[i], -remaining[i]))
        else:
            cust_list.sort(key=lambda i: -remaining[i])
        touched = set()
        for i in cust_list:
            guard = 0
            while remaining.get(i, 0) > EPS:
                opts = insertion_options(routes, i, remaining[i])
                if not opts:
                    q = remaining[i] if remaining[i] < Q else Q
                    opt = (0.0, -1, 0, q)  # fallback (should not happen)
                else:
                    opt = min(opts, key=lambda x: x[0])
                apply_opt(routes, i, opt, remaining, touched)
                guard += 1
                if guard > 10000:
                    break
        return touched

    # ------------------------------------------------------------------
    # Initial construction: sequential cheapest-insertion route building
    # ------------------------------------------------------------------
    nbr = None
    if n > 120 and n <= 1500:
        nbr = {}
        for i in [DEP] + ids:
            nbr[i] = sorted(ids, key=lambda j: D[i][j])[:60]

    def build_initial():
        remaining = {i: DEM[i] for i in ids if DEM[i] > EPS}
        routes = []
        while remaining:
            seed = max(remaining, key=lambda i: (remaining[i], D[DEP][i]))
            q = remaining[seed] if remaining[seed] < Q else Q
            r = make_route([seed], {seed: q})
            rem = remaining[seed] - q
            if rem <= EPS:
                del remaining[seed]
            else:
                remaining[seed] = rem
            while r['load'] < Q - EPS:
                resid = Q - r['load']
                if n <= 120 or nbr is None:
                    pool = [i for i in remaining if i not in r['del']]
                    if n > 1500 and len(pool) > 80:
                        pool = rng.sample(pool, 80)
                else:
                    ps = set(nbr[DEP])
                    for u in r['seq']:
                        ps.update(nbr[u])
                    pool = [i for i in ps if i in remaining and i not in r['del']]
                    if not pool:
                        pool = [i for i in remaining if i not in r['del']]
                best = None  # (delta - solo, delta, i, pos, q)
                seq = r['seq']
                base = r['cost']
                for i in pool:
                    q = remaining[i] if remaining[i] < resid else resid
                    solo, fs = eval_route([i], {i: q})
                    if not fs:
                        solo = 1e18
                    nd = dict(r['del'])
                    nd[i] = q
                    for pos in range(len(seq) + 1):
                        ns = seq[:pos] + [i] + seq[pos:]
                        c, f = eval_route(ns, nd)
                        if not f:
                            continue
                        delta = c - base
                        kkey = delta - solo
                        if best is None or kkey < best[0]:
                            best = (kkey, delta, i, pos, q)
                if best is None or best[0] >= -1e-9:
                    break
                _, _, i, pos, q = best
                r['seq'].insert(pos, i)
                r['del'][i] = q
                r['load'] += q
                c, _ = eval_route(r['seq'], r['del'])
                r['cost'] = c
                rem = remaining[i] - q
                if rem <= EPS:
                    del remaining[i]
                else:
                    remaining[i] = rem
            routes.append(r)
        return routes

    # ------------------------------------------------------------------
    # Destroy operators
    # ------------------------------------------------------------------
    def all_visits(routes):
        v = []
        for ri in range(len(routes)):
            for p, i in enumerate(routes[ri]['seq']):
                v.append((ri, p, i))
        return v

    def remove_chosen(routes, chosen, remaining):
        by = {}
        for ri, p in chosen:
            by.setdefault(ri, []).append(p)
        for ri in sorted(by.keys(), reverse=True):
            r = routes[ri]
            for p in sorted(by[ri], reverse=True):
                i = r['seq'].pop(p)
                q = r['del'].pop(i)
                r['load'] -= q
                remaining[i] = remaining.get(i, 0) + q
            if r['seq']:
                c, _ = eval_route(r['seq'], r['del'])
                r['cost'] = c
            else:
                routes.pop(ri)

    def destroy_random(routes, k, remaining):
        vis = all_visits(routes)
        if not vis:
            return
        k = min(k, len(vis))
        chosen = rng.sample(vis, k)
        remove_chosen(routes, [(ri, p) for ri, p, _ in chosen], remaining)

    def destroy_worst(routes, k, remaining):
        gains = []
        for ri in range(len(routes)):
            r = routes[ri]
            seq = r['seq']
            base = r['cost']
            for p, i in enumerate(seq):
                ns = seq[:p] + seq[p + 1:]
                if ns:
                    nd = dict(r['del'])
                    nd.pop(i)
                    c, _ = eval_route(ns, nd)
                else:
                    c = 0.0
                gains.append((base - c + rng.random() * 1e-9, ri, p))
        gains.sort(reverse=True)
        chosen = [(ri, p) for _, ri, p in gains[:min(k, len(gains))]]
        remove_chosen(routes, chosen, remaining)

    def destroy_shaw(routes, k, remaining):
        vis = all_visits(routes)
        if not vis:
            return
        _, _, i0 = rng.choice(vis)
        scored = []
        for ri, p, i in vis:
            rel = D[i0][i] / DMAX + abs(E[i0] - E[i]) / SPAN
            scored.append((rel * (0.8 + 0.4 * rng.random()), ri, p))
        scored.sort()
        chosen = [(ri, p) for _, ri, p in scored[:min(k, len(scored))]]
        remove_chosen(routes, chosen, remaining)

    # ------------------------------------------------------------------
    # Local search
    # ------------------------------------------------------------------
    def best_2opt(r):
        seq = r['seq']
        m = len(seq)
        if m < 2:
            return False
        bestc = r['cost']
        bestseq = None
        delv = r['del']
        for a in range(m - 1):
            for b in range(a + 1, m):
                ns = seq[:a] + seq[a:b + 1][::-1] + seq[b + 1:]
                c, f = eval_route(ns, delv)
                if f and c < bestc - 1e-9:
                    bestc = c
                    bestseq = ns
        if bestseq is not None:
            r['seq'] = bestseq
            r['cost'] = bestc
            return True
        return False

    def best_relocate_intra(r):
        seq = r['seq']
        m = len(seq)
        if m < 2:
            return False
        bestc = r['cost']
        bestseq = None
        delv = r['del']
        for p in range(m):
            node = seq[p]
            rest = seq[:p] + seq[p + 1:]
            for pos in range(m):
                if pos == p:
                    continue
                ns = rest[:pos] + [node] + rest[pos:]
                c, f = eval_route(ns, delv)
                if f and c < bestc - 1e-9:
                    bestc = c
                    bestseq = ns
        if bestseq is not None:
            r['seq'] = bestseq
            r['cost'] = bestc
            return True
        return False

    def ls_intra_route(r, deadline):
        improved = False
        while time.perf_counter() < deadline:
            if best_2opt(r) or best_relocate_intra(r):
                improved = True
            else:
                break
        return improved

    def ls_inter(routes, deadline, restrict=None):
        improved_any = False
        improved = True
        while improved:
            improved = False
            if time.perf_counter() >= deadline:
                break
            # ---- relocate between routes ----
            applied = False
            R = len(routes)
            for r1 in range(R):
                if applied or time.perf_counter() >= deadline:
                    break
                if restrict is not None and r1 not in restrict:
                    continue
                r1o = routes[r1]
                for p1 in range(len(r1o['seq'])):
                    if applied:
                        break
                    i = r1o['seq'][p1]
                    qi = r1o['del'][i]
                    seq1 = r1o['seq']
                    seq1wo = seq1[:p1] + seq1[p1 + 1:]
                    nd1 = dict(r1o['del'])
                    nd1.pop(i)
                    if seq1wo:
                        c1, f1 = eval_route(seq1wo, nd1)
                    else:
                        c1, f1 = 0.0, True
                    for r2 in range(R):
                        if r2 == r1:
                            continue
                        r2o = routes[r2]
                        if i in r2o['del']:
                            continue
                        if r2o['load'] + qi > Q + EPS:
                            continue
                        base = r1o['cost'] + r2o['cost']
                        seq2 = r2o['seq']
                        for pos in range(len(seq2) + 1):
                            ns2 = seq2[:pos] + [i] + seq2[pos:]
                            nd2 = dict(r2o['del'])
                            nd2[i] = qi
                            c2, f2 = eval_route(ns2, nd2)
                            if f2 and c1 + c2 < base - COST_EPS:
                                r2o['seq'] = ns2
                                r2o['del'] = nd2
                                r2o['load'] += qi
                                r2o['cost'] = c2
                                if seq1wo:
                                    r1o['seq'] = seq1wo
                                    r1o['del'] = nd1
                                    r1o['load'] -= qi
                                    r1o['cost'] = c1
                                else:
                                    routes.pop(r1)
                                applied = True
                                improved = True
                                improved_any = True
                                break
                        if applied:
                            break
                    if applied:
                        break
            if applied:
                continue
            # ---- swap between routes ----
            applied = False
            R = len(routes)
            for r1 in range(R):
                if applied or time.perf_counter() >= deadline:
                    break
                r1o = routes[r1]
                for r2 in range(r1 + 1, R):
                    if applied:
                        break
                    if restrict is not None and (r1 not in restrict and r2 not in restrict):
                        continue
                    r2o = routes[r2]
                    seq1 = r1o['seq']
                    seq2 = r2o['seq']
                    base = r1o['cost'] + r2o['cost']
                    for p1 in range(len(seq1)):
                        if applied:
                            break
                        i = seq1[p1]
                        qi = r1o['del'][i]
                        for p2 in range(len(seq2)):
                            j = seq2[p2]
                            qj = r2o['del'][j]
                            if j in r1o['del'] or i in r2o['del']:
                                continue
                            if r1o['load'] - qi + qj > Q + EPS:
                                continue
                            if r2o['load'] - qj + qi > Q + EPS:
                                continue
                            ns1 = seq1[:p1] + [j] + seq1[p1 + 1:]
                            nd1 = dict(r1o['del'])
                            nd1.pop(i)
                            nd1[j] = qj
                            c1, f1 = eval_route(ns1, nd1)
                            if not f1:
                                continue
                            ns2 = seq2[:p2] + [i] + seq2[p2 + 1:]
                            nd2 = dict(r2o['del'])
                            nd2.pop(j)
                            nd2[i] = qi
                            c2, f2 = eval_route(ns2, nd2)
                            if not f2:
                                continue
                            if c1 + c2 < base - COST_EPS:
                                r1o['seq'] = ns1
                                r1o['del'] = nd1
                                r1o['load'] = r1o['load'] - qi + qj
                                r1o['cost'] = c1
                                r2o['seq'] = ns2
                                r2o['del'] = nd2
                                r2o['load'] = r2o['load'] - qj + qi
                                r2o['cost'] = c2
                                applied = True
                                improved = True
                                improved_any = True
                                break
                        if applied:
                            break
        return improved_any

    def consolidate(routes, deadline):
        """Merge split visits: move a whole delivery of customer i from one
        route into another route that already visits i, if beneficial."""
        improved = False
        changed = True
        while changed and time.perf_counter() < deadline:
            changed = False
            loc = {}
            for ri, r in enumerate(routes):
                for i in r['seq']:
                    loc.setdefault(i, []).append(ri)
            done = False
            for i, rlist in loc.items():
                if len(rlist) < 2 or done:
                    continue
                for ai in range(len(rlist)):
                    if done:
                        break
                    for bi in range(len(rlist)):
                        if ai == bi:
                            continue
                        ra = routes[rlist[ai]]
                        rb = routes[rlist[bi]]
                        q = rb['del'][i]
                        if ra['load'] + q > Q + EPS:
                            continue
                        nda = dict(ra['del'])
                        nda[i] += q
                        ca, fa = eval_route(ra['seq'], nda)
                        if not fa:
                            continue
                        seqb = rb['seq']
                        pb = seqb.index(i)
                        nsb = seqb[:pb] + seqb[pb + 1:]
                        ndb = dict(rb['del'])
                        ndb.pop(i)
                        if nsb:
                            cb, fb = eval_route(nsb, ndb)
                        else:
                            cb, fb = 0.0, True
                        if fb and ca + cb < ra['cost'] + rb['cost'] - COST_EPS:
                            ra['del'] = nda
                            ra['load'] += q
                            ra['cost'] = ca
                            if nsb:
                                rb['seq'] = nsb
                                rb['del'] = ndb
                                rb['load'] -= q
                                rb['cost'] = cb
                            else:
                                routes.pop(rlist[bi])
                            changed = True
                            improved = True
                            done = True
                            break
        return improved

    def full_polish(routes, deadline):
        while time.perf_counter() < deadline:
            changed = False
            for r in routes:
                if time.perf_counter() >= deadline:
                    break
                if ls_intra_route(r, deadline):
                    changed = True
            if time.perf_counter() < deadline and ls_inter(routes, deadline):
                changed = True
            if time.perf_counter() < deadline and consolidate(routes, deadline):
                changed = True
            if not changed:
                break

    # ------------------------------------------------------------------
    # Output builder
    # ------------------------------------------------------------------
    def build_solution(routes):
        out_routes = []
        total = 0.0
        for vi, r in enumerate(routes):
            seq = r['seq']
            delv = r['del']
            load = 0.0
            for q in delv.values():
                load += q
            edges = []
            prev = DEP
            cost = 0.0
            for i in seq:
                d = D[prev][i]
                cost += d * (A * load + B)
                edges.append({'from': prev, 'to': i,
                              'load_weight': float(load)})
                load -= delv[i]
                prev = i
            d = D[prev][DEP]
            cost += d * (A * load + B)
            edges.append({'from': prev, 'to': NV,
                          'load_weight': float(load)})
            total += cost
            deliveries = {str(i): delv[i] for i in seq}
            out_routes.append({
                'vehicle': vi,
                'route': [DEP] + list(seq) + [NV],
                'deliveries': deliveries,
                'edges': edges,
            })
        return {'objective_value': float(total), 'routes': out_routes}

    # ------------------------------------------------------------------
    # Main procedure
    # ------------------------------------------------------------------
    routes = build_initial()
    init_ls_deadline = min(alns_end, start_time + 0.08 * time_limit + 0.2)
    if init_ls_deadline > start_time:
        full_polish(routes, init_ls_deadline)

    current = routes
    best = clone(routes)
    best_cost = total_cost(best)
    last_logged = best_cost
    if logger:
        logger.log_solution(best_cost, build_solution(best))

    destroy_ops = [destroy_random, destroy_worst, destroy_shaw]
    d_w = [1.0, 1.0, 1.0]
    r_w = [1.0, 1.0]
    d_sco = [0.0, 0.0, 0.0]
    r_sco = [0.0, 0.0]
    d_use = [0, 0, 0]
    r_use = [0, 0]

    def roulette(w):
        s = sum(w)
        x = rng.random() * s
        acc = 0.0
        for idx, wj in enumerate(w):
            acc += wj
            if x <= acc:
                return idx
        return len(w) - 1

    T0 = max(1.0, 0.02 * best_cost)
    alns_span = max(1e-9, alns_end - time.perf_counter())
    alns_t0 = time.perf_counter()

    it = 0
    while time.perf_counter() < alns_end:
        it += 1
        cand = clone(current)
        remaining = {}
        nvisits = sum(len(r['seq']) for r in cand)
        if nvisits == 0:
            break
        kmax = max(2, min(30, nvisits // 3 if nvisits >= 6 else nvisits))
        k = rng.randint(1, kmax)
        di = roulette(d_w)
        ri = roulette(r_w)
        destroy_ops[di](cand, k, remaining)
        touched = repair(cand, remaining, use_regret=(ri == 1))
        # light intra-route improvement on touched routes
        for ti in touched:
            if ti < len(cand):
                best_2opt(cand[ti])
                best_relocate_intra(cand[ti])
        c = total_cost(cand)
        cur = total_cost(current)
        score = 0.0
        accept = False
        if c < cur - 1e-9:
            accept = True
            score = 10.0
        else:
            frac = (time.perf_counter() - alns_t0) / alns_span
            if frac > 1.0:
                frac = 1.0
            Temp = T0 * (0.001 ** frac)
            if Temp > 1e-12 and rng.random() < math.exp(-(c - cur) / Temp):
                accept = True
                score = 3.0
        if accept:
            current = cand
        if c < best_cost - 1e-6:
            score = 20.0
            best_cost = c
            best = clone(cand)
            last_logged = c
            if logger:
                logger.log_solution(best_cost, build_solution(best))
        d_sco[di] += score
        d_use[di] += 1
        r_sco[ri] += score
        r_use[ri] += 1
        if it % 100 == 0:
            for j in range(3):
                if d_use[j] > 0:
                    d_w[j] = 0.7 * d_w[j] + 0.3 * (d_sco[j] / d_use[j])
            for j in range(2):
                if r_use[j] > 0:
                    r_w[j] = 0.7 * r_w[j] + 0.3 * (r_sco[j] / r_use[j])
            d_sco = [0.0, 0.0, 0.0]
            r_sco = [0.0, 0.0]
            d_use = [0, 0, 0]
            r_use = [0, 0]
        if it % 40 == 0:
            dl = time.perf_counter() + 0.02
            if dl < alns_end and ls_inter(current, dl):
                cc = total_cost(current)
                if cc < best_cost - 1e-6:
                    best_cost = cc
                    best = clone(current)
                    last_logged = cc
                    if logger:
                        logger.log_solution(best_cost, build_solution(best))

    # ------------------------------------------------------------------
    # Final polish on the incumbent
    # ------------------------------------------------------------------
    polish_deadline = end_time - 0.2
    full_polish(best, polish_deadline)
    best_cost = total_cost(best)
    sol = build_solution(best)
    best_cost = sol['objective_value']
    if logger and best_cost < last_logged - 1e-6:
        logger.log_solution(best_cost, sol)
    return sol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    logger = None
    if args.log_path and SolutionLogger is not None:
        logger = SolutionLogger(args.log_path, sense="minimize")

    with open(args.instance_path) as f:
        data = json.load(f)

    solution = solve(data, args.time_limit, logger)

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == '__main__':
    main()
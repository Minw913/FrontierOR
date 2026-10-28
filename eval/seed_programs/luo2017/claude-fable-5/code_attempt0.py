import argparse
import json
import time
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--instance_path', required=True)
    ap.add_argument('--solution_path', required=True)
    ap.add_argument('--time_limit', type=int, default=60)
    ap.add_argument('--log_path', default=None)
    args = ap.parse_args()

    start_time = time.time()
    deadline = start_time + max(2, args.time_limit) - 1.0

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path) as f:
        inst = json.load(f)

    n = inst['num_customers']
    Q = inst['vehicle_capacity']
    a = inst['cost_function']['a']
    b = inst['cost_function']['b']
    dep = inst['depot']
    depot_id = dep.get('id', 0)
    D = inst['distance_matrix']
    N = n + 1

    e = [0.0] * N
    l = [0.0] * N
    sv = [0.0] * N
    dem = [0] * N
    e[0], l[0] = dep['time_window'][0], dep['time_window'][1]
    sv[0] = 0.0
    for c in inst['customers']:
        i = c['id']
        e[i], l[i] = c['time_window'][0], c['time_window'][1]
        sv[i] = c['service_time']
        dem[i] = c['demand']
    e0 = e[0]
    l0 = min(l[0], inst.get('scheduling_horizon', l[0]))

    random.seed(0)

    # ---- pre-split: full truckload direct trips for demands >= capacity ----
    fixed = []           # list of customer ids, one entry per full-load direct trip
    resid = {}           # customer -> residual demand (0 < resid < = Q-1 or original demand < Q)
    for i in range(1, N):
        d = dem[i]
        if d <= 0:
            continue
        k = d // Q
        r = d - k * Q
        if r == 0 and k > 0:
            for _ in range(k):
                fixed.append(i)
        else:
            for _ in range(k):
                fixed.append(i)
            if r > 0:
                resid[i] = r

    fixed_cost = sum(D[0][i] * (a * Q + b) + D[i][0] * b for i in fixed)

    # ---- route evaluation (cost + feasibility) ----
    def eval_route(seq, qty):
        tot = 0
        for c in seq:
            tot += qty[c]
        if tot > Q + 1e-9:
            return None
        w = tot
        cost = 0.0
        prev = 0
        t = e0
        for c in seq:
            dpc = D[prev][c]
            cost += dpc * (a * w + b)
            t += dpc
            ec = e[c]
            if t < ec:
                t = ec
            if t > l[c] + 1e-9:
                return None
            t += sv[c]
            w -= qty[c]
            prev = c
        cost += D[prev][0] * (a * w + b)
        if t + D[prev][0] > l0 + 1e-9:
            return None
        return cost

    def clone(routes):
        return [{'seq': list(r['seq']), 'qty': dict(r['qty']),
                 'cost': r['cost'], 'load': r['load']} for r in routes]

    def total_cost(routes):
        return fixed_cost + sum(r['cost'] for r in routes)

    # ---- greedy construction ----
    def best_insert(routes, c, q):
        best = None
        for ri, r in enumerate(routes):
            if r['load'] + q > Q or c in r['qty']:
                continue
            nq = dict(r['qty'])
            nq[c] = q
            seq = r['seq']
            for pos in range(len(seq) + 1):
                ns = seq[:pos] + [c] + seq[pos:]
                nc = eval_route(ns, nq)
                if nc is None:
                    continue
                delta = nc - r['cost']
                if best is None or delta < best[0]:
                    best = (delta, ri, ns, nq, nc)
        single = eval_route([c], {c: q})
        if single is None:
            single = D[0][c] * (a * q + b) + D[c][0] * b
        if best is None or single <= best[0]:
            routes.append({'seq': [c], 'qty': {c: q}, 'cost': single, 'load': q})
        else:
            _, ri, ns, nq, nc = best
            r = routes[ri]
            r['seq'] = ns
            r['qty'] = nq
            r['cost'] = nc
            r['load'] += q

    routes = []
    for c in sorted(resid.keys(), key=lambda x: -resid[x]):
        best_insert(routes, c, resid[c])

    # ---- neighborhood moves (first improvement) ----
    def move_relocate(routes):
        for i1 in range(len(routes)):
            if time.time() > deadline:
                return False
            r1 = routes[i1]
            seq1 = r1['seq']
            for k1 in range(len(seq1)):
                c = seq1[k1]
                q = r1['qty'][c]
                rem_seq = seq1[:k1] + seq1[k1 + 1:]
                rem_qty = {x: r1['qty'][x] for x in rem_seq}
                rem_cost = eval_route(rem_seq, rem_qty) if rem_seq else 0.0
                for i2 in range(len(routes)):
                    r2 = routes[i2]
                    if i2 == i1:
                        for pos in range(len(rem_seq) + 1):
                            if pos == k1:
                                continue
                            ns = rem_seq[:pos] + [c] + rem_seq[pos:]
                            nc = eval_route(ns, r1['qty'])
                            if nc is not None and nc < r1['cost'] - 1e-7:
                                r1['seq'] = ns
                                r1['cost'] = nc
                                return True
                    else:
                        if c in r2['qty'] or r2['load'] + q > Q:
                            continue
                        seq2 = r2['seq']
                        nq2 = dict(r2['qty'])
                        nq2[c] = q
                        thresh = r1['cost'] + r2['cost'] - rem_cost - 1e-7
                        for pos in range(len(seq2) + 1):
                            ns2 = seq2[:pos] + [c] + seq2[pos:]
                            nc2 = eval_route(ns2, nq2)
                            if nc2 is not None and nc2 < thresh:
                                r1['seq'] = rem_seq
                                r1['qty'] = rem_qty
                                r1['cost'] = rem_cost
                                r1['load'] -= q
                                r2['seq'] = ns2
                                r2['qty'] = nq2
                                r2['cost'] = nc2
                                r2['load'] += q
                                if not rem_seq:
                                    routes.pop(i1)
                                return True
        return False

    def move_swap(routes):
        for i1 in range(len(routes)):
            if time.time() > deadline:
                return False
            r1 = routes[i1]
            for i2 in range(i1 + 1, len(routes)):
                r2 = routes[i2]
                for k1, c1 in enumerate(r1['seq']):
                    if c1 in r2['qty']:
                        continue
                    q1 = r1['qty'][c1]
                    for k2, c2 in enumerate(r2['seq']):
                        if c2 in r1['qty']:
                            continue
                        q2 = r2['qty'][c2]
                        if r1['load'] - q1 + q2 > Q or r2['load'] - q2 + q1 > Q:
                            continue
                        ns1 = list(r1['seq'])
                        ns1[k1] = c2
                        nq1 = dict(r1['qty'])
                        del nq1[c1]
                        nq1[c2] = q2
                        nc1 = eval_route(ns1, nq1)
                        if nc1 is None:
                            continue
                        ns2 = list(r2['seq'])
                        ns2[k2] = c1
                        nq2 = dict(r2['qty'])
                        del nq2[c2]
                        nq2[c1] = q1
                        nc2 = eval_route(ns2, nq2)
                        if nc2 is None:
                            continue
                        if nc1 + nc2 < r1['cost'] + r2['cost'] - 1e-7:
                            r1['seq'], r1['qty'], r1['cost'] = ns1, nq1, nc1
                            r1['load'] += q2 - q1
                            r2['seq'], r2['qty'], r2['cost'] = ns2, nq2, nc2
                            r2['load'] += q1 - q2
                            return True
        return False

    def move_2opt(routes):
        for r in routes:
            seq = r['seq']
            L = len(seq)
            if L < 3:
                continue
            for i in range(L - 1):
                for j in range(i + 1, L):
                    ns = seq[:i] + seq[i:j + 1][::-1] + seq[j + 1:]
                    nc = eval_route(ns, r['qty'])
                    if nc is not None and nc < r['cost'] - 1e-7:
                        r['seq'] = ns
                        r['cost'] = nc
                        return True
        return False

    def move_transfer(routes):
        # shift quantity between two visits of the same customer
        locs = {}
        for ri, r in enumerate(routes):
            for c in r['seq']:
                locs.setdefault(c, []).append(ri)
        for c, ris in locs.items():
            if len(ris) < 2:
                continue
            for i1 in ris:
                for i2 in ris:
                    if i1 == i2:
                        continue
                    r1, r2 = routes[i1], routes[i2]
                    q1 = r1['qty'][c]
                    spare = Q - r2['load']
                    m = min(q1, spare)
                    if m <= 0:
                        continue
                    nq2 = dict(r2['qty'])
                    nq2[c] += m
                    nc2 = eval_route(r2['seq'], nq2)
                    if nc2 is None:
                        continue
                    if m == q1:
                        k1 = r1['seq'].index(c)
                        ns1 = r1['seq'][:k1] + r1['seq'][k1 + 1:]
                        nq1 = {x: r1['qty'][x] for x in ns1}
                        nc1 = eval_route(ns1, nq1) if ns1 else 0.0
                    else:
                        ns1 = list(r1['seq'])
                        nq1 = dict(r1['qty'])
                        nq1[c] = q1 - m
                        nc1 = eval_route(ns1, nq1)
                    if nc1 is None:
                        continue
                    if nc1 + nc2 < r1['cost'] + r2['cost'] - 1e-7:
                        r1['seq'], r1['qty'], r1['cost'] = ns1, nq1, nc1
                        r1['load'] -= m
                        r2['qty'], r2['cost'] = nq2, nc2
                        r2['load'] += m
                        if not ns1:
                            routes.pop(i1)
                        return True
        return False

    def move_split(routes):
        for i1 in range(len(routes)):
            if time.time() > deadline:
                return False
            r1 = routes[i1]
            for c in list(r1['seq']):
                q = r1['qty'][c]
                if q < 2:
                    continue
                for i2, r2 in enumerate(routes):
                    if i2 == i1 or c in r2['qty']:
                        continue
                    spare = Q - r2['load']
                    if spare <= 0:
                        continue
                    m = min(spare, q - 1)
                    nq1 = dict(r1['qty'])
                    nq1[c] = q - m
                    nc1 = eval_route(r1['seq'], nq1)
                    if nc1 is None:
                        continue
                    nq2 = dict(r2['qty'])
                    nq2[c] = m
                    thresh = r1['cost'] + r2['cost'] - nc1 - 1e-7
                    seq2 = r2['seq']
                    for pos in range(len(seq2) + 1):
                        ns2 = seq2[:pos] + [c] + seq2[pos:]
                        nc2 = eval_route(ns2, nq2)
                        if nc2 is not None and nc2 < thresh:
                            r1['qty'], r1['cost'] = nq1, nc1
                            r1['load'] -= m
                            r2['seq'], r2['qty'], r2['cost'] = ns2, nq2, nc2
                            r2['load'] += m
                            return True
        return False

    moves = [move_relocate, move_2opt, move_swap, move_transfer, move_split]

    def local_search(routes):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for mv in moves:
                while mv(routes):
                    improved = True
                    if time.time() > deadline:
                        return

    def perturb(routes):
        custs = set()
        for r in routes:
            custs.update(r['seq'])
        if not custs:
            return
        k = min(len(custs), random.randint(2, 4))
        removed = random.sample(sorted(custs), k)
        for c in removed:
            for ri in range(len(routes) - 1, -1, -1):
                r = routes[ri]
                if c in r['qty']:
                    q = r['qty'].pop(c)
                    r['seq'].remove(c)
                    r['load'] -= q
                    if r['seq']:
                        r['cost'] = eval_route(r['seq'], r['qty'])
                    else:
                        routes.pop(ri)
        random.shuffle(removed)
        for c in removed:
            best_insert(routes, c, resid[c])

    # ---- solution export ----
    def build_solution(routes):
        out = []
        vid = 0
        tot_obj = 0.0
        vend = n + 1
        for c in fixed:
            cost = D[0][c] * (a * Q + b) + D[c][0] * b
            tot_obj += cost
            out.append({'vehicle': vid,
                        'route': [depot_id, c, vend],
                        'deliveries': {str(c): float(Q)},
                        'edges': [{'from': depot_id, 'to': c, 'load_weight': float(Q)},
                                  {'from': c, 'to': vend, 'load_weight': 0.0}]})
            vid += 1
        for r in routes:
            if not r['seq']:
                continue
            w = sum(r['qty'].values())
            prev = 0
            path = [depot_id]
            edges = []
            for c in r['seq']:
                edges.append({'from': depot_id if prev == 0 else prev,
                              'to': c, 'load_weight': float(w)})
                tot_obj += D[prev][c] * (a * w + b)
                w -= r['qty'][c]
                prev = c
                path.append(c)
            edges.append({'from': prev, 'to': vend, 'load_weight': float(w)})
            tot_obj += D[prev][0] * (a * w + b)
            path.append(vend)
            out.append({'vehicle': vid, 'route': path,
                        'deliveries': {str(c): float(r['qty'][c]) for c in r['seq']},
                        'edges': edges})
            vid += 1
        return {'objective_value': tot_obj, 'routes': out}

    # ---- optimize ----
    sol = build_solution(routes)
    if logger:
        logger.log_solution(sol['objective_value'], sol)

    local_search(routes)
    best = clone(routes)
    best_cost = total_cost(routes)
    if logger:
        logger.log_solution(best_cost, build_solution(best))

    cur = clone(best)
    cur_cost = best_cost
    while time.time() < deadline and resid:
        cand = clone(cur)
        perturb(cand)
        local_search(cand)
        cc = total_cost(cand)
        if cc < best_cost - 1e-7:
            best = clone(cand)
            best_cost = cc
            if logger:
                logger.log_solution(best_cost, build_solution(best))
        if cc < cur_cost + 1e-7 or random.random() < 0.05:
            cur, cur_cost = cand, cc
        elif random.random() < 0.2:
            cur = clone(best)
            cur_cost = best_cost

    final = build_solution(best)
    with open(args.solution_path, 'w') as f:
        json.dump(final, f, indent=2)


if __name__ == '__main__':
    main()
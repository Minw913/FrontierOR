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
    ap.add_argument('--time_limit', type=int, default=60)
    ap.add_argument('--log_path', default=None)
    args = ap.parse_args()

    t_start = time.time()
    buffer = max(1.0, min(3.0, 0.03 * args.time_limit))
    deadline = t_start + args.time_limit - buffer
    random.seed(0)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path) as f:
        data = json.load(f)

    n = data['num_customers']
    H = data['num_days']
    V = data['num_vehicles']
    Q = data['vehicle_capacity']
    Dmax = data['max_route_duration']
    dep = data['depot']
    custs = data['customers']

    xs = [dep['x']] + [c['x'] for c in custs]
    ys = [dep['y']] + [c['y'] for c in custs]
    service = [dep.get('service_time', 0)] + [c['service_time'] for c in custs]
    twE = [dep['time_window'][0]] + [c['time_window'][0] for c in custs]
    twL = [dep['time_window'][1]] + [c['time_window'][1] for c in custs]
    ids = [dep['id']] + [c['id'] for c in custs]
    N = n + 1

    dist = [[0.0] * N for _ in range(N)]
    for i in range(N):
        xi, yi = xs[i], ys[i]
        for j in range(i + 1, N):
            d = math.hypot(xi - xs[j], yi - ys[j])
            dist[i][j] = d
            dist[j][i] = d

    # day labels / offset
    all_labels = set()
    for c in custs:
        for s in c['schedules']:
            for p in s['parts']:
                all_labels.add(p['visit_day'])
            for d in s['days']:
                all_labels.add(d)
    off = min(all_labels) if all_labels else 0

    # internal schedule representation: scheds[i] = list over schedules of list of
    # (day_index, day_label, days_covered, demand)
    scheds = [None] * (n + 1)
    for i in range(1, n + 1):
        L = []
        for s in custs[i - 1]['schedules']:
            vis = []
            for p in s['parts']:
                di = p['visit_day'] - off
                if di < 0:
                    di = 0
                if di >= H:
                    di = H - 1
                vis.append((di, p['visit_day'], p['days_covered'], p['demand_per_visit']))
            L.append(vis)
        scheds[i] = L

    # solution state
    routes = [[] for _ in range(H)]          # per day: list of {'seq','load','cost'}
    cur_dem = [dict() for _ in range(H)]     # per day: cust -> demand delivered
    sched_choice = [-1] * (n + 1)

    dtw0 = twE[0]
    dtw1 = twL[0]

    def eval_route(di, seq, oc=0, od=0):
        """Return route cost if feasible else None."""
        dem = cur_dem[di]
        load = 0
        cost = 0.0
        t = dtw0
        prev = 0
        W = 0.0
        cap = 1e18
        for c in seq:
            load += od if c == oc else dem[c]
            if load > Q:
                return None
            e = dist[prev][c]
            cost += e
            arr = t + e + service[prev]
            a = twE[c]
            st = a if arr < a else arr
            if st > twL[c] + 1e-9:
                return None
            W += st - arr
            cc = W + twL[c] - st
            if cc < cap:
                cap = cc
            t = st
            prev = c
        e = dist[prev][0]
        cost += e
        ret = t + e + service[prev]
        if ret > dtw1 + 1e-9:
            return None
        delta = W if W < cap else cap
        if delta < 0:
            delta = 0.0
        if ret - dtw0 - delta > Dmax + 1e-9:
            return None
        return cost

    def best_insertion(di, c, demand):
        """Return (delta_cost, route_idx or -1 for new route, pos) or None."""
        bestd = 1e18
        bri = None
        bpos = 0
        R = routes[di]
        for ri, r in enumerate(R):
            seq = r['seq']
            if c in seq:
                continue
            if r['load'] + demand > Q:
                continue
            m = len(seq)
            for pos in range(m + 1):
                a = seq[pos - 1] if pos > 0 else 0
                b = seq[pos] if pos < m else 0
                dd = dist[a][c] + dist[c][b] - dist[a][b]
                if dd >= bestd - 1e-12:
                    continue
                if eval_route(di, seq[:pos] + [c] + seq[pos:], c, demand) is not None:
                    bestd = dd
                    bri = ri
                    bpos = pos
        if len(R) < V:
            r0 = eval_route(di, [c], c, demand)
            if r0 is not None and r0 < bestd:
                bestd = r0
                bri = -1
                bpos = 0
        if bri is None:
            return None
        return (bestd, bri, bpos)

    def do_insert(di, c, demand, ri, pos):
        if ri == -1:
            routes[di].append({'seq': [c], 'load': demand,
                               'cost': dist[0][c] + dist[c][0]})
        else:
            r = routes[di][ri]
            seq = r['seq']
            a = seq[pos - 1] if pos > 0 else 0
            b = seq[pos] if pos < len(seq) else 0
            r['cost'] += dist[a][c] + dist[c][b] - dist[a][b]
            seq.insert(pos, c)
            r['load'] += demand
        cur_dem[di][c] = demand

    def remove_visit(di, c):
        R = routes[di]
        for ri, r in enumerate(R):
            seq = r['seq']
            if c in seq:
                pos = seq.index(c)
                a = seq[pos - 1] if pos > 0 else 0
                b = seq[pos + 1] if pos + 1 < len(seq) else 0
                r['cost'] -= dist[a][c] + dist[c][b] - dist[a][b]
                seq.pop(pos)
                r['load'] -= cur_dem[di].get(c, 0)
                if c in cur_dem[di]:
                    del cur_dem[di][c]
                if not seq:
                    R.pop(ri)
                return

    def remove_customer(c):
        si = sched_choice[c]
        if si < 0:
            return
        for (di, _, _, _) in scheds[c][si]:
            remove_visit(di, c)
        sched_choice[c] = -1

    def insert_customer_best(c):
        """Choose schedule minimizing total insertion cost, apply. Return cost or None."""
        best = None
        for si, vis in enumerate(scheds[c]):
            tot = 0.0
            plan = []
            ok = True
            for (di, _, _, dem) in vis:
                bi = best_insertion(di, c, dem)
                if bi is None:
                    ok = False
                    break
                tot += bi[0]
                plan.append((di, dem, bi[1], bi[2]))
            if ok and (best is None or tot < best[0]):
                best = (tot, si, plan)
        if best is None:
            return None
        tot, si, plan = best
        for (di, dem, ri, pos) in plan:
            do_insert(di, c, dem, ri, pos)
        sched_choice[c] = si
        return tot

    def force_insert(di, c, dem):
        bi = best_insertion(di, c, dem)
        if bi is not None:
            do_insert(di, c, dem, bi[1], bi[2])
            return
        bestd = 1e18
        bri = None
        bpos = 0
        for ri, r in enumerate(routes[di]):
            seq = r['seq']
            if c in seq or r['load'] + dem > Q:
                continue
            for pos in range(len(seq) + 1):
                a = seq[pos - 1] if pos > 0 else 0
                b = seq[pos] if pos < len(seq) else 0
                dd = dist[a][c] + dist[c][b] - dist[a][b]
                if dd < bestd:
                    bestd = dd
                    bri = ri
                    bpos = pos
        if bri is None:
            routes[di].append({'seq': [c], 'load': dem,
                               'cost': dist[0][c] + dist[c][0]})
            cur_dem[di][c] = dem
        else:
            do_insert(di, c, dem, bri, bpos)

    # ---------------- local search ----------------
    def relocate_pass(di):
        R = routes[di]
        for ri in range(len(R)):
            r = R[ri]
            seq = r['seq']
            for pos in range(len(seq)):
                c = seq[pos]
                a = seq[pos - 1] if pos > 0 else 0
                b = seq[pos + 1] if pos + 1 < len(seq) else 0
                gain = dist[a][c] + dist[c][b] - dist[a][b]
                if gain <= 1e-9:
                    continue
                dem = cur_dem[di][c]
                for rj in range(len(R)):
                    if rj == ri:
                        continue
                    r2 = R[rj]
                    if r2['load'] + dem > Q:
                        continue
                    s2 = r2['seq']
                    for p2 in range(len(s2) + 1):
                        a2 = s2[p2 - 1] if p2 > 0 else 0
                        b2 = s2[p2] if p2 < len(s2) else 0
                        dd = dist[a2][c] + dist[c][b2] - dist[a2][b2]
                        if dd < gain - 1e-7 and \
                                eval_route(di, s2[:p2] + [c] + s2[p2:]) is not None:
                            r['cost'] -= gain
                            seq.pop(pos)
                            r['load'] -= dem
                            r2['cost'] += dd
                            s2.insert(p2, c)
                            r2['load'] += dem
                            if not seq:
                                R.pop(ri)
                            return True
        return False

    def intra_relocate_pass(di):
        for r in routes[di]:
            seq = r['seq']
            m = len(seq)
            for pos in range(m):
                c = seq[pos]
                a = seq[pos - 1] if pos > 0 else 0
                b = seq[pos + 1] if pos + 1 < m else 0
                gain = dist[a][c] + dist[c][b] - dist[a][b]
                if gain <= 1e-9:
                    continue
                base = seq[:pos] + seq[pos + 1:]
                for p2 in range(len(base) + 1):
                    if p2 == pos:
                        continue
                    a2 = base[p2 - 1] if p2 > 0 else 0
                    b2 = base[p2] if p2 < len(base) else 0
                    dd = dist[a2][c] + dist[c][b2] - dist[a2][b2]
                    if dd < gain - 1e-7:
                        news = base[:p2] + [c] + base[p2:]
                        nc = eval_route(di, news)
                        if nc is not None:
                            r['seq'] = news
                            r['cost'] = nc
                            return True
        return False

    def twoopt_pass(di):
        for r in routes[di]:
            seq = r['seq']
            m = len(seq)
            for i in range(m - 1):
                a = seq[i - 1] if i > 0 else 0
                for j in range(i + 1, m):
                    b = seq[j + 1] if j + 1 < m else 0
                    dd = dist[a][seq[j]] + dist[seq[i]][b] \
                        - dist[a][seq[i]] - dist[seq[j]][b]
                    if dd < -1e-7:
                        news = seq[:i] + seq[i:j + 1][::-1] + seq[j + 1:]
                        nc = eval_route(di, news)
                        if nc is not None:
                            r['seq'] = news
                            r['cost'] = nc
                            return True
        return False

    def improve_day(di, dl):
        while time.time() < dl:
            if relocate_pass(di):
                continue
            if intra_relocate_pass(di):
                continue
            if twoopt_pass(di):
                continue
            break

    # ---------------- snapshot / restore ----------------
    def snapshot():
        return ([[(list(r['seq']), r['load'], r['cost']) for r in routes[d]]
                 for d in range(H)],
                list(sched_choice),
                [dict(cur_dem[d]) for d in range(H)])

    def restore(s):
        rt, sc, cd = s
        for d in range(H):
            routes[d] = [{'seq': list(q), 'load': l, 'cost': c} for (q, l, c) in rt[d]]
            cur_dem[d] = dict(cd[d])
        sched_choice[:] = sc

    def total_cost():
        return sum(r['cost'] for d in range(H) for r in routes[d])

    def build_solution():
        sol_routes = {}
        for d in range(H):
            lab = str(d + off)
            lst = []
            for r in routes[d]:
                parts = {}
                dem_tot = 0
                for c in r['seq']:
                    si = sched_choice[c]
                    if si < 0:
                        si = 0
                    for (di2, vlab, cov, dm) in scheds[c][si]:
                        if di2 == d:
                            parts[str(ids[c])] = {'visit_day': vlab,
                                                  'days_covered': cov,
                                                  'demand_per_visit': dm}
                            dem_tot += dm
                            break
                cost = 0.0
                prev = 0
                for c in r['seq']:
                    cost += dist[prev][c]
                    prev = c
                cost += dist[prev][0]
                lst.append({'customers': [ids[c] for c in r['seq']],
                            'cost': cost, 'demand': dem_tot, 'parts': parts})
            sol_routes[lab] = lst
        sol_sched = {}
        for c in range(1, n + 1):
            si = sched_choice[c]
            if si < 0:
                si = 0
            s = custs[c - 1]['schedules'][si]
            sol_sched[str(ids[c])] = {'schedule_index': si,
                                      'days': s['days'],
                                      'visit_frequency': s['visit_frequency']}
        obj = sum(rr['cost'] for day in sol_routes.values() for rr in day)
        return {'objective_value': obj, 'routes': sol_routes, 'schedules': sol_sched}

    # ---------------- construction ----------------
    order = sorted(range(1, n + 1),
                   key=lambda c: -max(v[3] for s in scheds[c] for v in s))
    for c in order:
        if insert_customer_best(c) is None:
            sched_choice[c] = 0
            for (di, _, _, dem) in scheds[c][0]:
                force_insert(di, c, dem)

    sol = build_solution()
    if logger:
        logger.log_solution(sol['objective_value'], sol)

    # initial local search (bounded)
    ls_end = min(deadline, time.time() + 0.2 * args.time_limit)
    for d in range(H):
        improve_day(d, ls_end)

    cur = total_cost()
    best = cur
    best_snap = snapshot()
    sol = build_solution()
    if logger:
        logger.log_solution(sol['objective_value'], sol)

    # ---------------- LNS with SA acceptance ----------------
    if n >= 1:
        T0 = max(1e-6, 0.02 * cur)
        Tend = max(1e-9, 0.0005 * cur)
        while time.time() < deadline:
            frac = min(1.0, (time.time() - t_start) / max(1, args.time_limit))
            T = T0 * ((Tend / T0) ** frac)
            snap = snapshot()
            qmax = min(12, n)
            q = random.randint(1, qmax) if qmax >= 1 else 1
            if q < 2 and n >= 2:
                q = 2
            if random.random() < 0.5 or n < 3:
                rem = random.sample(range(1, n + 1), min(q, n))
            else:
                seedc = random.randint(1, n)
                rem = sorted(range(1, n + 1),
                             key=lambda x: dist[seedc][x])[:min(q, n)]
            touched = set()
            for c in rem:
                si = sched_choice[c]
                if si >= 0:
                    for (di, _, _, _) in scheds[c][si]:
                        touched.add(di)
                remove_customer(c)
            rem = list(rem)
            random.shuffle(rem)
            ok = True
            for c in rem:
                if insert_customer_best(c) is None:
                    ok = False
                    break
                for (di, _, _, _) in scheds[c][sched_choice[c]]:
                    touched.add(di)
            if not ok:
                restore(snap)
                continue
            new = total_cost()
            accept = False
            if new < cur - 1e-9:
                accept = True
            else:
                arg = -(new - cur) / max(T, 1e-12)
                if arg > -50 and random.random() < math.exp(arg):
                    accept = True
            if accept:
                if new < best * 1.01:
                    dl = min(deadline, time.time() + 1.0)
                    for d in touched:
                        improve_day(d, dl)
                    new = total_cost()
                cur = new
                if new < best - 1e-7:
                    best = new
                    best_snap = snapshot()
                    sol = build_solution()
                    if logger:
                        logger.log_solution(sol['objective_value'], sol)
            else:
                restore(snap)

    restore(best_snap)
    sol = build_solution()
    if logger:
        logger.log_solution(sol['objective_value'], sol)
    with open(args.solution_path, 'w') as f:
        json.dump(sol, f, indent=2)


if __name__ == '__main__':
    main()
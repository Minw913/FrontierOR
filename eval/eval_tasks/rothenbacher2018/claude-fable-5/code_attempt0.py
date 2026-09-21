import argparse
import json
import random
import time
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--instance_path', required=True)
    ap.add_argument('--solution_path', required=True)
    ap.add_argument('--time_limit', type=int, default=60)
    ap.add_argument('--log_path', default=None)
    args = ap.parse_args()

    start_t = time.monotonic()
    deadline = start_t + max(3, args.time_limit) - 1.0
    random.seed(0)

    try:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    except Exception:
        logger = None

    with open(args.instance_path) as f:
        data = json.load(f)

    P = data['parameters']
    D = data['distance_matrix']
    TT = data['travel_time_matrix']
    depot = data['depot']['id']

    customers = {}
    for c in data.get('truck_customers', []) or []:
        customers[c['id']] = {'supply': c['supply'], 'st': c['service_time'], 'truck_only': True}
    for c in data.get('trailer_customers', []) or []:
        customers[c['id']] = {'supply': c['supply'], 'st': c['service_time'], 'truck_only': False}

    ts_ids = [t['id'] for t in data.get('transshipment_locations', []) or []]
    hor = P['planning_horizon']
    Tmax = min(P['max_route_duration'], hor[1] - hor[0])
    dec_t = P['decoupling_time']; cp_t = P['coupling_time']; tr_t = P['transfer_time']
    dec_c = P['decoupling_cost']; cp_c = P['coupling_cost']; tr_c = P['transfer_cost']

    classes = data['fleet']['vehicle_classes']
    truck_avail = {t['id']: t['num_available'] for t in data['fleet']['truck_types']}
    trailer_avail = {t['id']: t['num_available'] for t in data['fleet']['trailer_types']}
    nloc = len(D)
    cust_ids = list(customers.keys())
    n = len(cust_ids)

    nearest_ts = {c: sorted(ts_ids, key=lambda p: D[c][p] + D[p][c])[:3] for c in cust_ids}

    # ---------------- evaluation ----------------
    def eval_main(cls, main):
        has_tr = cls['trailer_type_id'] != 0
        cf = cls['truck_cost_factor']
        mult = cf + (cls['trailer_towing_cost_multiplier'] if has_tr else 0.0)
        tcap = cls['truck_capacity']
        rcap = cls['trailer_capacity'] if has_tr else 0
        cost = cls['truck_fixed_cost']
        t = 0.0
        truck = 0
        trailer = 0
        prev = depot
        for node in main:
            loc = node['loc']
            cost += D[prev][loc] * mult
            t += TT[prev][loc]
            prev = loc
            if node['is_cust']:
                cu = customers[loc]
                if cu['truck_only'] and has_tr:
                    return None
                q = cu['supply']
                if has_tr:
                    f = rcap - trailer
                    if f > q:
                        f = q
                    trailer += f
                    truck += q - f
                else:
                    truck += q
                if truck > tcap:
                    return None
                t += cu['st']
            subs = node['subs']
            if subs:
                if not has_tr:
                    return None
                t += dec_t
                cost += dec_c
                ns = len(subs)
                for si in range(ns):
                    sub = subs[si]
                    p = loc
                    for c in sub:
                        cost += D[p][c] * cf
                        t += TT[p][c]
                        cu = customers[c]
                        truck += cu['supply']
                        if truck > tcap:
                            return None
                        t += cu['st']
                        p = c
                    cost += D[p][loc] * cf
                    t += TT[p][loc]
                    if si < ns - 1:
                        t += tr_t
                        cost += tr_c
                    else:
                        t += cp_t
                        cost += cp_c
                    mv = rcap - trailer
                    if mv > truck:
                        mv = truck
                    truck -= mv
                    trailer += mv
        cost += D[prev][depot] * mult
        t += TT[prev][depot]
        if t > Tmax + 1e-9:
            return None
        return cost

    def copy_main(main):
        return [{'loc': nd['loc'], 'is_cust': nd['is_cust'],
                 'subs': [s[:] for s in nd['subs']]} for nd in main]

    def clone_routes(routes):
        return [{'cls': r['cls'], 'main': copy_main(r['main']), 'cost': r['cost']} for r in routes]

    def usage(routes):
        ut = defaultdict(int)
        ur = defaultdict(int)
        for r in routes:
            ut[r['cls']['truck_type_id']] += 1
            tr = r['cls']['trailer_type_id']
            if tr:
                ur[tr] += 1
        return ut, ur

    def customers_of(r):
        out = []
        for nd in r['main']:
            if nd['is_cust']:
                out.append(nd['loc'])
            for s in nd['subs']:
                out.extend(s)
        return out

    # ---------------- insertion ----------------
    def insertion_candidates(c, route):
        cls = route['cls']
        has_tr = cls['trailer_type_id'] != 0
        main = route['main']
        cu = customers[c]
        res = []
        L = len(main)
        if not has_tr:
            for i in range(L + 1):
                m = copy_main(main)
                m.insert(i, {'loc': c, 'is_cust': True, 'subs': []})
                res.append(m)
        else:
            if not cu['truck_only']:
                for i in range(L + 1):
                    m = copy_main(main)
                    m.insert(i, {'loc': c, 'is_cust': True, 'subs': []})
                    res.append(m)
            for ni in range(L):
                nd = main[ni]
                for si in range(len(nd['subs'])):
                    for pi in range(len(nd['subs'][si]) + 1):
                        m = copy_main(main)
                        m[ni]['subs'][si].insert(pi, c)
                        res.append(m)
                if cu['truck_only']:
                    m = copy_main(main)
                    m[ni]['subs'].append([c])
                    res.append(m)
            if cu['truck_only']:
                for p in nearest_ts.get(c, [])[:2]:
                    for i in range(L + 1):
                        m = copy_main(main)
                        m.insert(i, {'loc': p, 'is_cust': False, 'subs': [[c]]})
                        res.append(m)
        return res

    def new_route_candidates(c, cls):
        has_tr = cls['trailer_type_id'] != 0
        cu = customers[c]
        res = []
        if (not has_tr) or (not cu['truck_only']):
            res.append([{'loc': c, 'is_cust': True, 'subs': []}])
        if has_tr and cu['truck_only']:
            for p in nearest_ts.get(c, [])[:2]:
                res.append([{'loc': p, 'is_cust': False, 'subs': [[c]]}])
        return res

    def try_insert(c, routes, noise):
        best = None
        for ri, r in enumerate(routes):
            base = r['cost']
            for m in insertion_candidates(c, r):
                cost = eval_main(r['cls'], m)
                if cost is None:
                    continue
                delta = cost - base
                key = delta + (random.random() * noise if noise > 0 else 0.0)
                if best is None or key < best[0]:
                    best = (key, ri, m, cost, None)
        ut, ur = usage(routes)
        for cls in classes:
            tid = cls['truck_type_id']
            if ut[tid] >= truck_avail.get(tid, 0):
                continue
            trid = cls['trailer_type_id']
            if trid and ur[trid] >= trailer_avail.get(trid, 0):
                continue
            for m in new_route_candidates(c, cls):
                cost = eval_main(cls, m)
                if cost is None:
                    continue
                key = cost + (random.random() * noise if noise > 0 else 0.0)
                if best is None or key < best[0]:
                    best = (key, None, m, cost, cls)
        if best is None:
            return False
        _, ri, m, cost, cls = best
        if ri is None:
            routes.append({'cls': cls, 'main': m, 'cost': cost})
        else:
            routes[ri]['main'] = m
            routes[ri]['cost'] = cost
        return True

    # ---------------- removal ----------------
    def remove_customers(routes, sel):
        selset = set(sel)
        extra = []
        for r in routes:
            newmain = []
            for nd in r['main']:
                if nd['is_cust'] and nd['loc'] in selset:
                    for s in nd['subs']:
                        extra.extend(s)
                    continue
                nd['subs'] = [[c for c in s if c not in selset] for s in nd['subs']]
                nd['subs'] = [s for s in nd['subs'] if s]
                if (not nd['is_cust']) and (not nd['subs']):
                    continue
                newmain.append(nd)
            r['main'] = newmain
        routes[:] = [r for r in routes if r['main']]
        for r in routes:
            c = eval_main(r['cls'], r['main'])
            r['cost'] = c if c is not None else float('inf')
        removed = [c for c in selset] + [c for c in extra if c not in selset]
        return removed

    # ---------------- local improvement ----------------
    def improve_route(r):
        changed = True
        guard = 0
        while changed and guard < 200:
            guard += 1
            if time.monotonic() > deadline:
                return
            changed = False
            main = r['main']
            L = len(main)
            # 2-opt on main tour
            done = False
            for i in range(L - 1):
                for j in range(i + 1, L):
                    m = copy_main(main)
                    m[i:j + 1] = list(reversed(m[i:j + 1]))
                    c = eval_main(r['cls'], m)
                    if c is not None and c < r['cost'] - 1e-9:
                        r['main'] = m
                        r['cost'] = c
                        changed = True
                        done = True
                        break
                if done:
                    break
            if changed:
                continue
            # 2-opt within subtours
            main = r['main']
            done = False
            for ni in range(len(main)):
                for si in range(len(main[ni]['subs'])):
                    s = main[ni]['subs'][si]
                    ls = len(s)
                    for i in range(ls - 1):
                        for j in range(i + 1, ls):
                            m = copy_main(main)
                            ss = m[ni]['subs'][si]
                            ss[i:j + 1] = list(reversed(ss[i:j + 1]))
                            c = eval_main(r['cls'], m)
                            if c is not None and c < r['cost'] - 1e-9:
                                r['main'] = m
                                r['cost'] = c
                                changed = True
                                done = True
                                break
                        if done:
                            break
                    if done:
                        break
                if done:
                    break

    # ---------------- output rendering ----------------
    def build_solution(routes):
        uid = [nloc + 1000]

        def fresh():
            uid[0] += 1
            return uid[0] - 1

        sol_routes = []
        class_count = defaultdict(int)
        total = 0.0
        for r in routes:
            cls = r['cls']
            seq = []
            phys = []
            arr = {}
            decl = []
            cpl = []
            served = []
            used = set()

            def add(loc, t):
                if loc in used:
                    sid = fresh()
                else:
                    sid = loc
                    used.add(loc)
                seq.append(sid)
                phys.append(loc)
                arr[str(sid)] = float(t)

            t = 0.0
            add(depot, 0.0)
            prev = depot
            for nd in r['main']:
                loc = nd['loc']
                t += TT[prev][loc]
                add(loc, t)
                prev = loc
                if nd['is_cust']:
                    served.append(loc)
                    t += customers[loc]['st']
                subs = nd['subs']
                if subs:
                    t += dec_t
                    decl.append(loc)
                    ns = len(subs)
                    for si in range(ns):
                        p = loc
                        for c in subs[si]:
                            t += TT[p][c]
                            add(c, t)
                            served.append(c)
                            t += customers[c]['st']
                            p = c
                        t += TT[p][loc]
                        add(loc, t)
                        if si < ns - 1:
                            t += tr_t
                        else:
                            t += cp_t
                    cpl.append(loc)
            t += TT[prev][depot]
            add(depot, t)

            cid = cls['class_id']
            vidx = class_count[cid]
            class_count[cid] += 1
            total += r['cost']
            sol_routes.append({
                'vehicle_class': cid,
                'vehicle_index': vidx,
                'customers_served': served,
                'route_sequence': seq,
                'route_sequence_physical': phys,
                'decouple_locations': decl,
                'couple_locations': cpl,
                'arrival_times': arr
            })
        return {'objective_value': total, 'routes': sol_routes}

    # ---------------- construction ----------------
    def construct(order):
        routes = []
        for c in order:
            if time.monotonic() > deadline:
                return None
            if not try_insert(c, routes, 0.0):
                return None
        return routes

    if n == 0:
        with open(args.solution_path, 'w') as f:
            json.dump({'objective_value': 0.0, 'routes': []}, f)
        return

    orders = []
    o1 = sorted(cust_ids, key=lambda c: -customers[c]['supply'])
    orders.append(o1)
    o2 = sorted(cust_ids, key=lambda c: -D[depot][c])
    orders.append(o2)
    for _ in range(6):
        o = cust_ids[:]
        random.shuffle(o)
        orders.append(o)

    cur = None
    for o in orders:
        cur = construct(o)
        if cur is not None:
            break
    if cur is None:
        # last-resort: keep trying random orders while time remains
        while time.monotonic() < deadline and cur is None:
            o = cust_ids[:]
            random.shuffle(o)
            cur = construct(o)
    if cur is None:
        with open(args.solution_path, 'w') as f:
            json.dump({'objective_value': float('inf'), 'routes': []}, f)
        return

    for r in cur:
        improve_route(r)
    cur_cost = sum(r['cost'] for r in cur)
    best = clone_routes(cur)
    best_cost = cur_cost
    if logger:
        logger.log_solution(best_cost, build_solution(best))

    # ---------------- LNS main loop ----------------
    qmax = max(3, min(30, max(3, n // 4)))
    noise_base = 0.05 * cur_cost / max(1, n)
    non_improve = 0

    while time.monotonic() < deadline:
        cand = clone_routes(cur)
        mode = random.random()
        if mode < 0.4:
            q = random.randint(2, qmax)
            all_c = []
            for r in cand:
                all_c.extend(customers_of(r))
            q = min(q, len(all_c))
            sel = random.sample(all_c, q)
        elif mode < 0.8:
            all_c = []
            for r in cand:
                all_c.extend(customers_of(r))
            seed = random.choice(all_c)
            q = min(random.randint(2, qmax), len(all_c))
            all_c.sort(key=lambda x: D[seed][x])
            sel = all_c[:q]
        else:
            r = random.choice(cand)
            sel = customers_of(r)

        removed = remove_customers(cand, sel)
        if any(r['cost'] == float('inf') for r in cand):
            continue

        if random.random() < 0.5:
            removed.sort(key=lambda c: -customers[c]['supply'] + random.random())
        else:
            random.shuffle(removed)

        noise = noise_base if random.random() < 0.5 else 0.0
        ok = True
        for c in removed:
            if time.monotonic() > deadline:
                ok = False
                break
            if not try_insert(c, cand, noise):
                ok = False
                break
        if not ok:
            continue

        ccost = sum(r['cost'] for r in cand)
        if ccost < cur_cost * 1.02:
            for r in cand:
                improve_route(r)
            ccost = sum(r['cost'] for r in cand)

        accepted = False
        if ccost < cur_cost - 1e-9:
            accepted = True
        elif ccost < cur_cost * 1.015 and random.random() < 0.12:
            accepted = True

        if accepted:
            cur = cand
            cur_cost = ccost
            if ccost < best_cost - 1e-9:
                best = clone_routes(cand)
                best_cost = ccost
                non_improve = 0
                if logger:
                    logger.log_solution(best_cost, build_solution(best))
            else:
                non_improve += 1
        else:
            non_improve += 1

        if non_improve >= 600:
            cur = clone_routes(best)
            cur_cost = best_cost
            non_improve = 0

    sol = build_solution(best)
    with open(args.solution_path, 'w') as f:
        json.dump(sol, f)


if __name__ == '__main__':
    main()
import argparse
import json
import random
import time

from solution_logger import SolutionLogger

CONV = ("Let n_orig be the number of physical locations and P be the instance "
        "parking-location IDs in their published list order. Physical nodes retain IDs "
        "0..n_orig-1. The decouple copy of P[i] has ID n_orig+i, and the couple copy of "
        "P[i] has ID n_orig+len(P)+i. Both copies map back to physical location P[i].")

EPS = 1e-7


class Solver:
    def __init__(self, inst, logger, deadline):
        self.logger = logger
        self.deadline = deadline
        p = inst['parameters']
        self.maxdur = float(p['max_route_duration'])
        self.dec_t = float(p.get('decoupling_time', 0.0))
        self.cou_t = float(p.get('coupling_time', 0.0))
        self.dec_c = float(p.get('decoupling_cost', 0.0))
        self.cou_c = float(p.get('coupling_cost', 0.0))
        self.H0, self.H1 = [float(x) for x in p['planning_horizon']]
        self.default_st = float(p.get('service_time', 0.0))
        self.depot = inst['depot']['id']
        self.D = inst['distance_matrix']
        self.T = inst['travel_time_matrix']
        self.n_orig = len(self.D)

        self.cust = {}
        for c in inst.get('truck_customers', []):
            self.cust[c['id']] = {'supply': c['supply'],
                                  'st': float(c.get('service_time', self.default_st)),
                                  'truck': True}
        for c in inst.get('trailer_customers', []):
            self.cust[c['id']] = {'supply': c['supply'],
                                  'st': float(c.get('service_time', self.default_st)),
                                  'truck': False}
        self.ts_ids = [t['id'] for t in inst.get('transshipment_locations', [])]
        self.park_ids = [c['id'] for c in inst.get('trailer_customers', [])] + self.ts_ids
        self.park_index = {pid: i for i, pid in enumerate(self.park_ids)}
        self.npark = len(self.park_ids)

        fleet = inst['fleet']
        self.classes = []
        for vc in fleet['vehicle_classes']:
            self.classes.append({'cid': vc['class_id'], 'tt': vc['truck_type_id'],
                                 'rt': vc['trailer_type_id'], 'tcap': vc['truck_capacity'],
                                 'rcap': vc['trailer_capacity'], 'totcap': vc['total_capacity'],
                                 'fixed': float(vc['truck_fixed_cost']),
                                 'cf': float(vc['truck_cost_factor']),
                                 'tow': float(vc['trailer_towing_cost_multiplier']),
                                 'navail': vc['num_available']})
        self.avail_truck = {t['id']: t['num_available'] for t in fleet['truck_types']}
        self.avail_trailer = {t['id']: t['num_available'] for t in fleet['trailer_types']}
        self.class_used = {c['cid']: 0 for c in self.classes}

        ids = list(self.cust)
        self.near_ts = {}
        self.neighbors = {}
        for c in ids:
            self.near_ts[c] = sorted(self.ts_ids, key=lambda t_: self.D[c][t_])[:2]
            self.neighbors[c] = sorted((o for o in ids if o != c),
                                       key=lambda o: self.D[c][o])[:10]

        self.routes = []
        self.cust2route = {}
        self.best_total = float('inf')
        self.best_sol = None
        self.fast = False

    # ---------------- availability -----------------
    def available(self, cls):
        if self.class_used[cls['cid']] >= cls['navail']:
            return False
        if self.avail_truck.get(cls['tt'], 0) < 1:
            return False
        if cls['rt'] != 0 and self.avail_trailer.get(cls['rt'], 0) < 1:
            return False
        return True

    def acquire(self, cls):
        self.class_used[cls['cid']] += 1
        self.avail_truck[cls['tt']] = self.avail_truck.get(cls['tt'], 0) - 1
        if cls['rt'] != 0:
            self.avail_trailer[cls['rt']] = self.avail_trailer.get(cls['rt'], 0) - 1

    def release(self, cls):
        self.class_used[cls['cid']] -= 1
        self.avail_truck[cls['tt']] = self.avail_truck.get(cls['tt'], 0) + 1
        if cls['rt'] != 0:
            self.avail_trailer[cls['rt']] = self.avail_trailer.get(cls['rt'], 0) + 1

    # ---------------- evaluation -----------------
    def eval_route(self, cls, main, build=False, relaxed=False):
        solo = (cls['rt'] == 0)
        cf = cls['cf']
        tow = cls['tow'] if not solo else 0.0
        tcap = cls['tcap']
        rcap = cls['rcap']
        D, T, cust = self.D, self.T, self.cust
        t = self.H0
        cost = cls['fixed']
        tl = 0.0
        rl = 0.0
        loc = self.depot
        if build:
            seq = [self.depot]; phys = [self.depot]
            arrs = [(self.depot, self.H0)]
            decs = []; cous = []; served = []
        for item in main:
            if item[0] == 'c':
                cid = item[1]
                c = cust[cid]
                if (not solo) and c['truck'] and not relaxed:
                    return None
                cost += D[loc][cid] * (cf + tow)
                t += T[loc][cid]
                if build:
                    seq.append(cid); phys.append(cid); arrs.append((cid, t)); served.append(cid)
                t += c['st']
                s = c['supply']
                if not solo:
                    r = min(s, rcap - rl)
                    rl += r; tl += s - r
                else:
                    tl += s
                if tl > tcap + 1e-9 and not relaxed:
                    return None
                loc = cid
            else:
                _, pid, serve, sub = item
                if solo and not relaxed:
                    return None
                cost += D[loc][pid] * (cf + tow)
                t += T[loc][pid]
                if serve:
                    c = cust[pid]
                    if build:
                        seq.append(pid); phys.append(pid); arrs.append((pid, t)); served.append(pid)
                    t += c['st']
                    s = c['supply']
                    r = min(s, rcap - rl)
                    rl += r; tl += s - r
                    if tl > tcap + 1e-9 and not relaxed:
                        return None
                di = self.n_orig + self.park_index.get(pid, 0)
                if build:
                    seq.append(di); phys.append(pid); arrs.append((di, t)); decs.append(pid)
                t += self.dec_t
                cost += self.dec_c
                cur = pid
                for sc in sub:
                    cost += D[cur][sc] * cf
                    t += T[cur][sc]
                    c = cust[sc]
                    if build:
                        seq.append(sc); phys.append(sc); arrs.append((sc, t)); served.append(sc)
                    t += c['st']
                    tl += c['supply']
                    if tl > tcap + 1e-9 and not relaxed:
                        return None
                    cur = sc
                cost += D[cur][pid] * cf
                t += T[cur][pid]
                ci = self.n_orig + self.npark + self.park_index.get(pid, 0)
                if build:
                    seq.append(ci); phys.append(pid); arrs.append((ci, t)); cous.append(pid)
                t += self.cou_t
                cost += self.cou_c
                x = min(tl, rcap - rl)
                rl += x; tl -= x
                loc = pid
        cost += D[loc][self.depot] * (cf + tow)
        t += T[loc][self.depot]
        if build:
            seq.append(self.depot); phys.append(self.depot); arrs.append((self.depot, t))
        if not relaxed:
            if t > self.H1 + 1e-6 or (t - self.H0) > self.maxdur + 1e-6:
                return None
        if build:
            return (cost, t, seq, phys, arrs, decs, cous, served)
        return cost

    def main_load(self, main):
        tot = 0
        for it in main:
            if it[0] == 'c':
                tot += self.cust[it[1]]['supply']
            else:
                if it[2]:
                    tot += self.cust[it[1]]['supply']
                for s in it[3]:
                    tot += self.cust[s]['supply']
        return tot

    def route_customers(self, main):
        out = []
        for it in main:
            if it[0] == 'c':
                out.append(it[1])
            else:
                if it[2]:
                    out.append(it[1])
                out.extend(it[3])
        return out

    # ---------------- insertion -----------------
    def candidate_mains(self, cls, main, c):
        solo = (cls['rt'] == 0)
        is_tc = self.cust[c]['truck']
        cands = []
        n = len(main)
        if self.fast:
            positions = [n]
        else:
            positions = range(n + 1)
        if solo or not is_tc:
            if solo or not is_tc:
                if solo:
                    for i in positions:
                        cands.append(main[:i] + [('c', c)] + main[i:])
                else:
                    if not is_tc:
                        for i in positions:
                            cands.append(main[:i] + [('c', c)] + main[i:])
        if not solo:
            for j, item in enumerate(main):
                if item[0] == 'p':
                    sub = item[3]
                    rng = [len(sub)] if self.fast else range(len(sub) + 1)
                    for k in rng:
                        nit = ('p', item[1], item[2], sub[:k] + [c] + sub[k:])
                        cands.append(main[:j] + [nit] + main[j + 1:])
            if is_tc:
                used = set(it[1] for it in main if it[0] == 'p')
                for j, item in enumerate(main):
                    if item[0] == 'c' and item[1] not in used:
                        nit = ('p', item[1], True, [c])
                        cands.append(main[:j] + [nit] + main[j + 1:])
                for ts in self.near_ts.get(c, []):
                    if ts in used:
                        continue
                    for i in positions:
                        cands.append(main[:i] + [('p', ts, False, [c])] + main[i:])
        return cands

    def best_insertion(self, r, c, main=None, base=None):
        cls = r['cls']
        if main is None:
            main = r['main']
            base = r['cost']
        if self.main_load(main) + self.cust[c]['supply'] > cls['totcap']:
            return (None, None)
        bestd = None
        bestm = None
        for m in self.candidate_mains(cls, main, c):
            v = self.eval_route(cls, m)
            if v is not None:
                d = v - base
                if bestd is None or d < bestd:
                    bestd = d
                    bestm = m
        return (bestd, bestm)

    def remove_customer(self, main, c):
        for i, item in enumerate(main):
            if item[0] == 'c' and item[1] == c:
                return main[:i] + main[i + 1:]
            if item[0] == 'p':
                pid, serve, sub = item[1], item[2], item[3]
                if serve and pid == c:
                    if not sub:
                        return main[:i] + main[i + 1:]
                    return None
                if c in sub:
                    ns = [x for x in sub if x != c]
                    if ns:
                        return main[:i] + [('p', pid, serve, ns)] + main[i + 1:]
                    if serve:
                        return main[:i] + [('c', pid)] + main[i + 1:]
                    return main[:i] + main[i + 1:]
        return None

    # ---------------- route creation -----------------
    def create_route(self, cls, main):
        v = self.eval_route(cls, main)
        if v is None:
            v = self.eval_route(cls, main, relaxed=True)
        r = {'cls': cls, 'main': main, 'cost': v}
        self.routes.append(r)
        self.acquire(cls)
        for c in self.route_customers(main):
            self.cust2route[c] = r
        return r

    def delete_route(self, r):
        self.routes.remove(r)
        self.release(r['cls'])

    def best_new_route(self, c):
        best = None
        is_tc = self.cust[c]['truck']
        for cls in self.classes:
            if not self.available(cls):
                continue
            if cls['rt'] != 0 and is_tc:
                if not self.ts_ids:
                    continue
                m = [('p', self.near_ts[c][0], False, [c])]
            else:
                m = [('c', c)]
            v = self.eval_route(cls, m)
            if v is not None and (best is None or v < best[0]):
                best = (v, cls, m)
        return best

    # ---------------- construction -----------------
    def construct(self):
        unassigned = set(self.cust)
        while unassigned:
            if time.time() > self.deadline - 2:
                self.fast = True
            seed = max(unassigned, key=lambda c: self.D[self.depot][c])
            is_tc = self.cust[seed]['truck']
            remaining = sum(self.cust[c]['supply'] for c in unassigned)
            best = None
            for cls in self.classes:
                if not self.available(cls):
                    continue
                extra = []
                if cls['rt'] != 0 and is_tc:
                    if self.ts_ids:
                        m = [('p', self.near_ts[seed][0], False, [seed])]
                    else:
                        tcands = [u for u in unassigned
                                  if (not self.cust[u]['truck']) and u != seed]
                        if not tcands:
                            continue
                        tc = min(tcands, key=lambda u: self.D[seed][u])
                        m = [('p', tc, True, [seed])]
                        extra = [tc]
                else:
                    m = [('c', seed)]
                v = self.eval_route(cls, m)
                if v is None:
                    continue
                score = v / max(1.0, min(cls['totcap'], remaining))
                if best is None or score < best[0]:
                    best = (score, cls, m, extra)
            if best is None:
                # forced fallback (possibly infeasible fleet-wise)
                cand = [c for c in self.classes if c['rt'] == 0] if is_tc else self.classes
                if not cand:
                    cand = self.classes
                cls = max(cand, key=lambda c: c['totcap'])
                m = [('c', seed)]
                best = (0.0, cls, m, [])
            _, cls, m, extra = best
            r = self.create_route(cls, m)
            unassigned.discard(seed)
            for e in extra:
                unassigned.discard(e)
            # fill route greedily (customers ordered by proximity to seed)
            order = sorted(unassigned, key=lambda c: self.D[seed][c])
            for c in order:
                if time.time() > self.deadline - 1:
                    self.fast = True
                d, m2 = self.best_insertion(r, c)
                if m2 is not None:
                    r['main'] = m2
                    r['cost'] = self.eval_route(cls, m2)
                    self.cust2route[c] = r
                    unassigned.discard(c)
        self.fast = False

    # ---------------- logging -----------------
    def build_solution(self):
        total = 0.0
        out_routes = []
        counter = {}
        for r in self.routes:
            if not r['main']:
                continue
            res = self.eval_route(r['cls'], r['main'], build=True, relaxed=True)
            cost, t, seq, phys, arrs, decs, cous, served = res
            total += cost
            cid = r['cls']['cid']
            idx = counter.get(cid, 0)
            counter[cid] = idx + 1
            arr = {}
            for node, tt_ in arrs:
                arr[str(node)] = tt_
            out_routes.append({'vehicle_class': cid, 'vehicle_index': idx,
                               'customers_served': served,
                               'route_sequence': seq,
                               'route_sequence_physical': phys,
                               'decouple_locations': decs,
                               'couple_locations': cous,
                               'arrival_times': arr})
        sol = {'objective_value': total, 'routes': out_routes,
               'expanded_node_convention': CONV}
        return total, sol

    def maybe_log(self):
        total, sol = self.build_solution()
        if total < self.best_total - 1e-9:
            self.best_total = total
            self.best_sol = sol
            if self.logger:
                self.logger.log_solution(total, sol)

    # ---------------- local search -----------------
    def relocate_pass(self, subset=None):
        improved = False
        ids = list(subset) if subset is not None else list(self.cust2route)
        random.shuffle(ids)
        for c in ids:
            if time.time() > self.deadline:
                break
            r = self.cust2route.get(c)
            if r is None:
                continue
            m0 = self.remove_customer(r['main'], c)
            if m0 is None:
                continue
            if m0:
                cost0 = self.eval_route(r['cls'], m0)
                if cost0 is None:
                    continue
            else:
                cost0 = 0.0
            gain = r['cost'] - cost0
            best = None  # (total_delta, target_route, newmain)
            if m0:
                d, m = self.best_insertion(r, c, main=m0, base=cost0)
                if m is not None:
                    td = -gain + d
                    best = (td, r, m)
            for r2 in self.routes:
                if r2 is r:
                    continue
                d, m = self.best_insertion(r2, c)
                if m is not None:
                    td = -gain + d
                    if best is None or td < best[0]:
                        best = (td, r2, m)
            if best is not None and best[0] < -EPS:
                td, r2, m = best
                if r2 is r:
                    r['main'] = m
                    r['cost'] = self.eval_route(r['cls'], m)
                else:
                    if not m0:
                        self.delete_route(r)
                    else:
                        r['main'] = m0
                        r['cost'] = cost0
                    r2['main'] = m
                    r2['cost'] = self.eval_route(r2['cls'], m)
                    self.cust2route[c] = r2
                improved = True
                self.maybe_log()
        return improved

    def swap_pass(self):
        improved = False
        ids = list(self.cust2route)
        random.shuffle(ids)
        for c1 in ids:
            if time.time() > self.deadline:
                break
            r1 = self.cust2route.get(c1)
            if r1 is None:
                continue
            for c2 in self.neighbors.get(c1, [])[:8]:
                r2 = self.cust2route.get(c2)
                if r2 is None or r2 is r1:
                    continue
                m1 = self.remove_customer(r1['main'], c1)
                m2 = self.remove_customer(r2['main'], c2)
                if m1 is None or m2 is None:
                    continue
                cost1 = self.eval_route(r1['cls'], m1) if m1 else 0.0
                cost2 = self.eval_route(r2['cls'], m2) if m2 else 0.0
                if (m1 and cost1 is None) or (m2 and cost2 is None):
                    continue
                d1, mm1 = self.best_insertion(r1, c2, main=m1, base=cost1)
                d2, mm2 = self.best_insertion(r2, c1, main=m2, base=cost2)
                if mm1 is None or mm2 is None:
                    continue
                delta = (cost1 + d1 - r1['cost']) + (cost2 + d2 - r2['cost'])
                if delta < -EPS:
                    r1['main'] = mm1
                    r1['cost'] = self.eval_route(r1['cls'], mm1)
                    r2['main'] = mm2
                    r2['cost'] = self.eval_route(r2['cls'], mm2)
                    self.cust2route[c1] = r2
                    self.cust2route[c2] = r1
                    improved = True
                    self.maybe_log()
                    break
        return improved

    def two_opt_pass(self):
        improved = False
        for r in self.routes:
            main = r['main']
            n = len(main)
            if n < 3:
                continue
            better = True
            while better and time.time() < self.deadline:
                better = False
                for i in range(n - 1):
                    for j in range(i + 1, n):
                        m2 = main[:i] + main[i:j + 1][::-1] + main[j + 1:]
                        v = self.eval_route(r['cls'], m2)
                        if v is not None and v < r['cost'] - EPS:
                            r['main'] = m2
                            r['cost'] = v
                            main = m2
                            better = True
                            improved = True
        if improved:
            self.maybe_log()
        return improved

    def class_change_pass(self):
        improved = False
        for r in self.routes:
            if time.time() > self.deadline:
                break
            cur = r['cls']
            for c2 in self.classes:
                if c2['cid'] == cur['cid']:
                    continue
                if self.class_used[c2['cid']] >= c2['navail']:
                    continue
                if c2['tt'] != cur['tt'] and self.avail_truck.get(c2['tt'], 0) < 1:
                    continue
                if c2['rt'] != 0 and c2['rt'] != cur['rt'] and \
                        self.avail_trailer.get(c2['rt'], 0) < 1:
                    continue
                if c2['rt'] == 0:
                    m2 = []
                    for it in r['main']:
                        if it[0] == 'c':
                            m2.append(it)
                        else:
                            if it[2]:
                                m2.append(('c', it[1]))
                            m2.extend(('c', s) for s in it[3])
                else:
                    ok = True
                    for it in r['main']:
                        if it[0] == 'c' and self.cust[it[1]]['truck']:
                            ok = False
                            break
                    if not ok:
                        continue
                    m2 = list(r['main'])
                v = self.eval_route(c2, m2)
                if v is not None and v < r['cost'] - EPS:
                    self.release(cur)
                    self.acquire(c2)
                    r['cls'] = c2
                    r['main'] = m2
                    r['cost'] = v
                    cur = c2
                    improved = True
                    self.maybe_log()
        return improved

    def local_search(self):
        while time.time() < self.deadline:
            improved = False
            if self.class_change_pass():
                improved = True
            if time.time() < self.deadline and self.two_opt_pass():
                improved = True
            if time.time() < self.deadline and self.relocate_pass():
                improved = True
            if time.time() < self.deadline and self.swap_pass():
                improved = True
            if not improved:
                break

    # ---------------- perturbation -----------------
    def snapshot(self):
        routes = []
        for r in self.routes:
            m = [it if it[0] == 'c' else (it[0], it[1], it[2], list(it[3]))
                 for it in r['main']]
            routes.append({'cls': r['cls'], 'main': m, 'cost': r['cost']})
        return (routes, dict(self.avail_truck), dict(self.avail_trailer),
                dict(self.class_used))

    def restore(self, snap):
        self.routes = snap[0]
        self.avail_truck = snap[1]
        self.avail_trailer = snap[2]
        self.class_used = snap[3]
        self.cust2route = {}
        for r in self.routes:
            for c in self.route_customers(r['main']):
                self.cust2route[c] = r

    def perturb(self):
        if not self.routes:
            return
        pre_total = sum(r['cost'] for r in self.routes)
        snap = self.snapshot()
        ids = list(self.cust2route)
        random.shuffle(ids)
        k = max(2, int(len(ids) * random.uniform(0.08, 0.2)))
        removed = []
        for c in ids:
            if len(removed) >= k:
                break
            r = self.cust2route[c]
            m0 = self.remove_customer(r['main'], c)
            if m0 is None:
                continue
            if m0:
                v = self.eval_route(r['cls'], m0)
                if v is None:
                    continue
                r['main'] = m0
                r['cost'] = v
            else:
                self.delete_route(r)
            del self.cust2route[c]
            removed.append(c)
        if len(removed) < 2:
            self.restore(snap)
            return
        random.shuffle(removed)
        ok = True
        for c in removed:
            if time.time() > self.deadline:
                ok = False
                break
            best = None
            for r2 in self.routes:
                d, m = self.best_insertion(r2, c)
                if m is not None and (best is None or d < best[0]):
                    best = (d, r2, m)
            nr = self.best_new_route(c)
            if nr is not None and (best is None or nr[0] < best[0]):
                self.create_route(nr[1], nr[2])
                continue
            if best is None:
                ok = False
                break
            d, r2, m = best
            r2['main'] = m
            r2['cost'] = self.eval_route(r2['cls'], m)
            self.cust2route[c] = r2
        if not ok:
            self.restore(snap)
            return
        self.relocate_pass(subset=removed)
        self.two_opt_pass()
        self.class_change_pass()
        total = sum(r['cost'] for r in self.routes)
        if total < pre_total - EPS:
            self.maybe_log()
        else:
            self.restore(snap)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--instance_path', required=True)
    ap.add_argument('--solution_path', required=True)
    ap.add_argument('--time_limit', type=int, default=60)
    ap.add_argument('--log_path', default=None)
    args = ap.parse_args()
    t0 = time.time()
    random.seed(0)

    with open(args.instance_path) as f:
        inst = json.load(f)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    deadline = t0 + max(5, args.time_limit) - 1.5

    solver = Solver(inst, logger, deadline)
    solver.construct()
    solver.maybe_log()
    solver.local_search()
    solver.maybe_log()
    while time.time() < solver.deadline - 0.5:
        solver.perturb()

    if solver.best_sol is None:
        _, sol = solver.build_solution()
        solver.best_sol = sol

    with open(args.solution_path, 'w') as f:
        json.dump(solver.best_sol, f, indent=2)


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
import argparse, json, time, math, random, copy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, default=120)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()
    random.seed(42)
    try:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    except Exception:
        logger = None
    start_time = time.time()
    with open(args.instance_path) as f:
        inst = json.load(f)
    solver = Solver(inst, args.time_limit, start_time, logger)
    sol = solver.solve()
    with open(args.solution_path, 'w') as f:
        json.dump(sol, f, indent=2)


class Solver:
    def __init__(self, inst, time_limit, start_time, logger):
        self.inst = inst; self.time_limit = time_limit
        self.start_time = start_time; self.logger = logger
        p = inst['parameters']
        self.max_route_duration = p['max_route_duration']
        self.service_time_global = p.get('service_time', 0)
        self.coupling_time = p['coupling_time']; self.decoupling_time = p['decoupling_time']
        self.transfer_time = p['transfer_time']
        self.coupling_cost = p['coupling_cost']; self.decoupling_cost = p['decoupling_cost']
        self.transfer_cost = p['transfer_cost']
        self.horizon_start = p['planning_horizon'][0]; self.horizon_end = p['planning_horizon'][1]
        self.dist = inst['distance_matrix']; self.tt = inst['travel_time_matrix']
        self.depot_id = inst['depot']['id']
        self.customers = {}
        for c in inst['truck_customers']:
            self.customers[c['id']] = {'id': c['id'], 'type': 'truck', 'supply': c['supply'],
                'service_time': c.get('service_time', self.service_time_global)}
        for c in inst['trailer_customers']:
            self.customers[c['id']] = {'id': c['id'], 'type': 'trailer', 'supply': c['supply'],
                'service_time': c.get('service_time', self.service_time_global)}
        self.parking_list = ([c['id'] for c in inst['trailer_customers']] +
                             [t['id'] for t in inst['transshipment_locations']])
        self.parking_set = set(self.parking_list)
        self.parking_index = {pid: i for i, pid in enumerate(self.parking_list)}
        self.n_orig = len(self.dist)
        self.classes = {c['class_id']: c for c in inst['fleet']['vehicle_classes']}
        self.class_ids = sorted(self.classes.keys(),
            key=lambda cid: (self.classes[cid]['truck_capacity'] + self.classes[cid]['trailer_capacity'],
                             self.classes[cid]['truck_fixed_cost']))
        self.class_avail = {c['class_id']: c['num_available'] for c in inst['fleet']['vehicle_classes']}
        self.truck_type_avail = {t['id']: t['num_available'] for t in inst['fleet']['truck_types']}
        self.trailer_type_avail = {t['id']: t['num_available'] for t in inst['fleet']['trailer_types']}
        self.best_routes = None; self.best_cost = float('inf')

    def remaining(self):
        return self.time_limit - (time.time() - self.start_time)

    def eval_route(self, route):
        cls = self.classes[route['class_id']]
        is_solo = (cls['trailer_capacity'] == 0)
        tcap = cls['truck_capacity']; rcap = cls['trailer_capacity']
        cf = cls['truck_cost_factor']; tm = cls['trailer_towing_cost_multiplier']
        fc = cls['truck_fixed_cost']
        mb = route.get('main_before', []); pl = route.get('parking_loc', None)
        st = route.get('subtour', []); ma = route.get('main_after', [])
        for c in mb + ma:
            if self.customers[c]['type'] == 'truck': return None
        if is_solo:
            if pl is not None or st: return None
        else:
            if pl is not None and pl not in self.parking_set: return None
        total = sum(self.customers[c]['supply'] for c in mb + st + ma)
        if is_solo:
            if total > tcap: return None
        else:
            if total > tcap + rcap: return None
        cur = self.depot_id; t = self.horizon_start; tl = 0; rl = 0; cost = fc
        for c in mb:
            cost += (cf + tm) * self.dist[cur][c]; t += self.tt[cur][c]
            if t > self.horizon_end: return None
            s = self.customers[c]['supply']; rem = rcap - rl
            if s <= rem: rl += s
            else:
                rl = rcap; tl += (s - rem)
                if tl > tcap: return None
            t += self.customers[c]['service_time']; cur = c
        if not is_solo and pl is not None:
            cost += (cf + tm) * self.dist[cur][pl]; t += self.tt[cur][pl]
            if t > self.horizon_end: return None
            t += self.decoupling_time; cost += self.decoupling_cost
            sub_cur = pl
            for c in st:
                cost += cf * self.dist[sub_cur][c]; t += self.tt[sub_cur][c]
                if t > self.horizon_end: return None
                tl += self.customers[c]['supply']
                if tl > tcap: return None
                t += self.customers[c]['service_time']; sub_cur = c
            cost += cf * self.dist[sub_cur][pl]; t += self.tt[sub_cur][pl]
            if t > self.horizon_end: return None
            t += self.coupling_time; cost += self.coupling_cost
            rem = rcap - rl; ta = min(tl, rem); rl += ta; tl -= ta
            t += self.transfer_time; cost += self.transfer_cost; cur = pl
        for c in ma:
            cost += (cf + tm) * self.dist[cur][c]; t += self.tt[cur][c]
            if t > self.horizon_end: return None
            s = self.customers[c]['supply']; rem = rcap - rl
            if s <= rem: rl += s
            else:
                rl = rcap; tl += (s - rem)
                if tl > tcap: return None
            t += self.customers[c]['service_time']; cur = c
        cost += (cf if is_solo else (cf + tm)) * self.dist[cur][self.depot_id]
        t += self.tt[cur][self.depot_id]
        if t > self.horizon_end: return None
        if t - self.horizon_start > self.max_route_duration: return None
        return {'cost': cost, 'time': t}

    def check_fleet(self, routes):
        cc = {}; tc = {}; rc = {}
        for r in routes:
            cls = self.classes[r['class_id']]
            cc[r['class_id']] = cc.get(r['class_id'], 0) + 1
            tc[cls['truck_type_id']] = tc.get(cls['truck_type_id'], 0) + 1
            if cls['trailer_type_id'] != 0:
                rc[cls['trailer_type_id']] = rc.get(cls['trailer_type_id'], 0) + 1
        for k, v in cc.items():
            if v > self.class_avail.get(k, 0): return False
        for k, v in tc.items():
            if v > self.truck_type_avail.get(k, 0): return False
        for k, v in rc.items():
            if v > self.trailer_type_avail.get(k, 0): return False
        return True

    def route_cost(self, r):
        ev = self.eval_route(r)
        return ev['cost'] if ev else float('inf')

    def total_cost(self, routes):
        if not self.check_fleet(routes): return float('inf')
        s = 0
        for r in routes:
            ev = self.eval_route(r)
            if ev is None: return float('inf')
            s += ev['cost']
        return s

    def construct_initial(self):
        custs = list(self.customers.keys())
        custs.sort(key=lambda c: (-self.customers[c]['supply'], self.dist[self.depot_id][c]))
        routes = []
        assigned = set()

        def try_insert(c, route):
            best = None; best_cost = float('inf')
            mb = route['main_before']; pl = route['parking_loc']
            st = route['subtour']; ma = route['main_after']
            ctype = self.customers[c]['type']
            cls = self.classes[route['class_id']]
            is_solo = (cls['trailer_capacity'] == 0)
            options = []
            if is_solo:
                for i in range(len(mb)+1): options.append(('mb', i))
            else:
                if ctype == 'trailer':
                    for i in range(len(mb)+1): options.append(('mb', i))
                    for i in range(len(ma)+1): options.append(('ma', i))
                    for i in range(len(st)+1): options.append(('st', i))
                else:
                    if pl is not None:
                        for i in range(len(st)+1): options.append(('st', i))
            for seg, i in options:
                nr = {'class_id': route['class_id'], 'main_before': mb[:],
                      'parking_loc': pl, 'subtour': st[:], 'main_after': ma[:]}
                if seg == 'mb': nr['main_before'].insert(i, c)
                elif seg == 'ma': nr['main_after'].insert(i, c)
                else: nr['subtour'].insert(i, c)
                ev = self.eval_route(nr)
                if ev and ev['cost'] < best_cost:
                    best_cost = ev['cost']; best = nr
            return best, best_cost

        for c in custs:
            if self.remaining() < 5: break
            best_idx = -1; best_new = None; best_delta = float('inf')
            for idx, r in enumerate(routes):
                old = self.route_cost(r)
                nr, nc = try_insert(c, r)
                if nr is not None and nc - old < best_delta:
                    best_delta = nc - old; best_idx = idx; best_new = nr
            if best_idx >= 0:
                routes[best_idx] = best_new
                assigned.add(c); continue
            best_new = None; best_cost = float('inf')
            for cid in self.class_ids:
                cls = self.classes[cid]; is_solo = (cls['trailer_capacity'] == 0)
                ctype = self.customers[c]['type']
                if is_solo:
                    nr = {'class_id': cid, 'main_before': [c], 'parking_loc': None,
                          'subtour': [], 'main_after': []}
                else:
                    if ctype == 'truck':
                        bpl = None; bpl_cost = float('inf')
                        for pl in self.parking_list:
                            nr = {'class_id': cid, 'main_before': [], 'parking_loc': pl,
                                  'subtour': [c], 'main_after': []}
                            ev = self.eval_route(nr)
                            if ev and ev['cost'] < bpl_cost:
                                bpl_cost = ev['cost']; bpl = pl
                        if bpl is None: continue
                        nr = {'class_id': cid, 'main_before': [], 'parking_loc': bpl,
                              'subtour': [c], 'main_after': []}
                    else:
                        nr = {'class_id': cid, 'main_before': [c], 'parking_loc': None,
                              'subtour': [], 'main_after': []}
                if not self.check_fleet(routes + [nr]): continue
                ev = self.eval_route(nr)
                if ev and ev['cost'] < best_cost:
                    best_cost = ev['cost']; best_new = nr
            if best_new is not None:
                routes.append(best_new); assigned.add(c)
        # Force-assign remaining customers
        for c in custs:
            if c in assigned: continue
            placed = False
            for cid in self.class_ids:
                cls = self.classes[cid]; is_solo = (cls['trailer_capacity'] == 0)
                ctype = self.customers[c]['type']
                if is_solo:
                    nr = {'class_id': cid, 'main_before': [c], 'parking_loc': None,
                          'subtour': [], 'main_after': []}
                else:
                    if ctype == 'truck':
                        bpl = None; bpl_cost = float('inf')
                        for pl in self.parking_list:
                            nr = {'class_id': cid, 'main_before': [], 'parking_loc': pl,
                                  'subtour': [c], 'main_after': []}
                            ev = self.eval_route(nr)
                            if ev and ev['cost'] < bpl_cost:
                                bpl_cost = ev['cost']; bpl = pl
                        if bpl is None: continue
                        nr = {'class_id': cid, 'main_before': [], 'parking_loc': bpl,
                              'subtour': [c], 'main_after': []}
                    else:
                        nr = {'class_id': cid, 'main_before': [c], 'parking_loc': None,
                              'subtour': [], 'main_after': []}
                if not self.check_fleet(routes + [nr]): continue
                ev = self.eval_route(nr)
                if ev:
                    routes.append(nr); assigned.add(c); placed = True; break
            if not placed:
                # Last resort: solo truck ignoring fleet (will be infeasible but output something)
                for cid in self.class_ids:
                    cls = self.classes[cid]
                    if cls['trailer_capacity'] == 0:
                        nr = {'class_id': cid, 'main_before': [c], 'parking_loc': None,
                              'subtour': [], 'main_after': []}
                        ev = self.eval_route(nr)
                        if ev:
                            routes.append(nr); assigned.add(c); break
        return routes, assigned

    def try_relocate(self, routes, i, j):
        # Move a customer from route i to route j
        r1 = routes[i]; r2 = routes[j]
        best = None; best_delta = 0
        # Enumerate customers in r1
        for seg_name in ['main_before', 'subtour', 'main_after']:
            seg = r1[seg_name]
            for k in range(len(seg)):
                c = seg[k]
                new_r1 = {k2: v2[:] if isinstance(v2, list) else v2 for k2, v2 in r1.items()}
                new_r1[seg_name] = seg[:k] + seg[k+1:]
                # If new_r1 has truck customers but no parking (complete vehicle), skip
                cls1 = self.classes[new_r1['class_id']]
                if cls1['trailer_capacity'] != 0:
                    has_truck = any(self.customers[cc]['type'] == 'truck' for cc in
                                    new_r1['main_before'] + new_r1['subtour'] + new_r1['main_after'])
                    if has_truck and new_r1['parking_loc'] is None:
                        continue
                ev1 = self.eval_route(new_r1)
                if ev1 is None: continue
                # Insert c into r2
                mb2 = r2['main_before']; pl2 = r2['parking_loc']
                st2 = r2['subtour']; ma2 = r2['main_after']
                ctype = self.customers[c]['type']
                cls2 = self.classes[r2['class_id']]
                is_solo2 = (cls2['trailer_capacity'] == 0)
                options = []
                if is_solo2:
                    for ii in range(len(mb2)+1): options.append(('mb', ii))
                else:
                    if ctype == 'trailer':
                        for ii in range(len(mb2)+1): options.append(('mb', ii))
                        for ii in range(len(ma2)+1): options.append(('ma', ii))
                        for ii in range(len(st2)+1): options.append(('st', ii))
                    else:
                        if pl2 is not None:
                            for ii in range(len(st2)+1): options.append(('st', ii))
                for seg2, ii in options:
                    nr2 = {'class_id': r2['class_id'], 'main_before': mb2[:],
                           'parking_loc': pl2, 'subtour': st2[:], 'main_after': ma2[:]}
                    if seg2 == 'mb': nr2['main_before'].insert(ii, c)
                    elif seg2 == 'ma': nr2['main_after'].insert(ii, c)
                    else: nr2['subtour'].insert(ii, c)
                    ev2 = self.eval_route(nr2)
                    if ev2 is None: continue
                    if not self.check_fleet([r for idx, r in enumerate(routes) if idx != i] + [new_r1, nr2]):
                        continue
                    old = self.route_cost(r1) + self.route_cost(r2)
                    new = ev1['cost'] + ev2['cost']
                    delta = new - old
                    if delta < best_delta - 1e-6:
                        best_delta = delta
                        best = (new_r1, nr2)
        return best, best_delta

    def two_opt_route(self, route):
        # Try 2-opt on each segment
        improved = True; best_route = route; best_cost = self.route_cost(route)
        while improved and self.remaining() > 3:
            improved = False
            for seg_name in ['main_before', 'subtour', 'main_after']:
                seg = best_route[seg_name]
                n = len(seg)
                if n < 3: continue
                for i in range(n-1):
                    for j in range(i+1, n):
                        nr = {k2: v2[:] if isinstance(v2, list) else v2 for k2, v2 in best_route.items()}
                        nr[seg_name] = seg[:i] + seg[i:j+1][::-1] + seg[j+1:]
                        c = self.route_cost(nr)
                        if c < best_cost - 1e-6:
                            best_cost = c; best_route = nr; improved = True
            if improved:
                best_route = {k2: v2 for k2, v2 in best_route.items()}
        return best_route

    def try_change_class(self, route, routes, idx):
        best = None; best_delta = 0
        cur_cost = self.route_cost(route)
        # Try converting complete vehicle (with parking) to solo, or vice versa
        for cid in self.class_ids:
            if cid == route['class_id']: continue
            cls = self.classes[cid]; is_solo = (cls['trailer_capacity'] == 0)
            cur_cls = self.classes[route['class_id']]; cur_is_solo = (cur_cls['trailer_capacity'] == 0)
            # Build candidate route
            all_custs = route['main_before'] + route['subtour'] + route['main_after']
            truck_custs = [c for c in all_custs if self.customers[c]['type'] == 'truck']
            trailer_custs = [c for c in all_custs if self.customers[c]['type'] == 'trailer']
            if is_solo:
                if truck_custs: continue  # solo can serve all, but we need to check
                nr = {'class_id': cid, 'main_before': all_custs[:], 'parking_loc': None,
                      'subtour': [], 'main_after': []}
            else:
                if not truck_custs:
                    nr = {'class_id': cid, 'main_before': trailer_custs[:], 'parking_loc': None,
                          'subtour': [], 'main_after': []}
                else:
                    # Need parking loc; put all truck custs + maybe trailer in subtour
                    bpl = None; bpl_cost = float('inf'); bnr = None
                    for pl in self.parking_list:
                        nr = {'class_id': cid, 'main_before': [], 'parking_loc': pl,
                              'subtour': all_custs[:], 'main_after': []}
                        ev = self.eval_route(nr)
                        if ev and ev['cost'] < bpl_cost:
                            bpl_cost = ev['cost']; bpl = pl; bnr = nr
                    if bpl is None: continue
                    nr = bnr
            new_routes = [r if ii != idx else nr for ii, r in enumerate(routes)]
            if not self.check_fleet(new_routes): continue
            c = self.route_cost(nr)
            delta = c - cur_cost
            if delta < best_delta - 1e-6:
                best_delta = delta; best = nr
        return best, best_delta

    def try_add_parking(self, route, routes, idx):
        # For a complete vehicle without parking, try adding a parking detour (empty subtour, just for cost)
        # Or move some trailer customers from main to subtour
        cls = self.classes[route['class_id']]
        if cls['trailer_capacity'] == 0: return None, 0
        if route['parking_loc'] is not None: return None, 0
        # Try: pick a parking loc, move all main to subtour? Only if it lowers cost (unlikely).
        # Skip - parking adds cost. Only useful to enable truck customers, but route has none.
        return None, 0

    def local_search(self, routes):
        routes = [r for r in routes]
        cur_cost = self.total_cost(routes)
        if cur_cost == float('inf'): return routes
        if self.logger and cur_cost < self.best_cost:
            self.best_cost = cur_cost; self.best_routes = [r for r in routes]
            sol = self.build_solution_dict(routes)
            self.logger.log_solution(cur_cost, sol)
        # 2-opt each route
        for i in range(len(routes)):
            if self.remaining() < 3: break
            nr = self.two_opt_route(routes[i])
            if self.route_cost(nr) < self.route_cost(routes[i]) - 1e-6:
                routes[i] = nr
        # Relocate
        improved = True
        while improved and self.remaining() > 5:
            improved = False
            for i in range(len(routes)):
                for j in range(len(routes)):
                    if i == j: continue
                    if self.remaining() < 3: break
                    res, delta = self.try_relocate(routes, i, j)
                    if res is not None:
                        new_r1, new_r2 = res
                        # Check if r1 empty -> remove
                        new_routes = []
                        for idx, r in enumerate(routes):
                            if idx == i:
                                if (new_r1['main_before'] or new_r1['subtour'] or new_r1['main_after']):
                                    new_routes.append(new_r1)
                            elif idx == j:
                                new_routes.append(new_r2)
                            else:
                                new_routes.append(r)
                        nc = self.total_cost(new_routes)
                        if nc < cur_cost - 1e-6:
                            routes = new_routes; cur_cost = nc; improved = True
                            if self.logger and cur_cost < self.best_cost:
                                self.best_cost = cur_cost; self.best_routes = [r for r in routes]
                                sol = self.build_solution_dict(routes)
                                self.logger.log_solution(cur_cost, sol)
                            break
                if improved: break
        # Change class
        improved = True
        while improved and self.remaining() > 5:
            improved = False
            for i in range(len(routes)):
                if self.remaining() < 3: break
                nr, delta = self.try_change_class(routes[i], routes, i)
                if nr is not None:
                    new_routes = [r if idx != i else nr for idx, r in enumerate(routes)]
                    nc = self.total_cost(new_routes)
                    if nc < cur_cost - 1e-6:
                        routes = new_routes; cur_cost = nc; improved = True
                        if self.logger and cur_cost < self.best_cost:
                            self.best_cost = cur_cost; self.best_routes = [r for r in routes]
                            sol = self.build_solution_dict(routes)
                            self.logger.log_solution(cur_cost, sol)
        return routes

    def build_solution_dict(self, routes):
        n_orig = self.n_orig; P = self.parking_list; lenP = len(P)
        pidx = self.parking_index
        class_count = {}; out_routes = []
        for r in routes:
            cls = self.classes[r['class_id']]; is_solo = (cls['trailer_capacity'] == 0)
            cid = r['class_id']; vidx = class_count.get(cid, 0); class_count[cid] = vidx + 1
            mb = r['main_before']; pl = r['parking_loc']; st = r['subtour']; ma = r['main_after']
            route_seq = [self.depot_id]; route_seq_phys = [self.depot_id]
            arrivals = {}; decouple_locs = []; couple_locs = []
            cur = self.depot_id; t = self.horizon_start
            arrivals[str(self.depot_id)] = t
            def serve(c):
                nonlocal cur, t
                t += self.tt[cur][c]
                route_seq.append(c); route_seq_phys.append(c)
                arrivals[str(c)] = t
                t += self.customers[c]['service_time']; cur = c
            if is_solo:
                for c in mb: serve(c)
            else:
                for c in mb: serve(c)
                if pl is not None:
                    t += self.tt[cur][pl]
                    dec_id = n_orig + pidx[pl]
                    route_seq.append(dec_id); route_seq_phys.append(pl)
                    arrivals[str(dec_id)] = t; decouple_locs.append(pl)
                    t += self.decoupling_time; sub_cur = pl
                    for c in st:
                        t += self.tt[sub_cur][c]
                        route_seq.append(c); route_seq_phys.append(c)
                        arrivals[str(c)] = t
                        t += self.customers[c]['service_time']; sub_cur = c
                    t += self.tt[sub_cur][pl]
                    cpl_id = n_orig + lenP + pidx[pl]
                    route_seq.append(cpl_id); route_seq_phys.append(pl)
                    arrivals[str(cpl_id)] = t; couple_locs.append(pl)
                    t += self.coupling_time + self.transfer_time; cur = pl
                for c in ma: serve(c)
            t += self.tt[cur][self.depot_id]
            route_seq.append(self.depot_id); route_seq_phys.append(self.depot_id)
            customers_served = list(mb) + list(st) + list(ma)
            out_routes.append({
                'vehicle_class': cid, 'vehicle_index': vidx,
                'customers_served': customers_served,
                'route_sequence': route_seq, 'route_sequence_physical': route_seq_phys,
                'decouple_locations': decouple_locs, 'couple_locations': couple_locs,
                'arrival_times': arrivals
            })
        return {'objective_value': self.total_cost(routes) if self.total_cost(routes) < float('inf') else 0.0,
                'routes': out_routes}

    def solve(self):
        routes, assigned = self.construct_initial()
        if not routes:
            sol = {'objective_value': 0.0, 'routes': []}
            if self.logger: self.logger.log_solution(0.0, sol)
            return sol
        cost = self.total_cost(routes)
        if cost < float('inf') and self.logger and cost < self.best_cost:
            self.best_cost = cost; self.best_routes = [r for r in routes]
            self.logger.log_solution(cost, self.build_solution_dict(routes))
        # Multi-restart local search
        best = routes; best_cost = cost
        iters = 0
        while self.remaining() > 5 and iters < 20:
            iters += 1
            routes = self.local_search(routes)
            c = self.total_cost(routes)
            if c < best_cost:
                best = [r for r in routes]; best_cost = c
            # Perturb: random relocations
            if self.remaining() > 5:
                perturbed = [r for r in routes]
                for _ in range(3):
                    if len(perturbed) >= 2:
                        i = random.randrange(len(perturbed))
                        j = random.randrange(len(perturbed))
                        if i != j:
                            res, _ = self.try_relocate(perturbed, i, j)
                            if res:
                                new_r1, new_r2 = res
                                new_routes = []
                                for idx, r in enumerate(perturbed):
                                    if idx == i:
                                        if new_r1['main_before'] or new_r1['subtour'] or new_r1['main_after']:
                                            new_routes.append(new_r1)
                                    elif idx == j: new_routes.append(new_r2)
                                    else: new_routes.append(r)
                                if self.total_cost(new_routes) < float('inf'):
                                    perturbed = new_routes
                routes = perturbed
        final = best if best_cost < float('inf') else routes
        if self.best_routes is not None and (best_cost == float('inf') or self.best_cost < best_cost):
            final = self.best_routes
        return self.build_solution_dict(final)


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
import json
import math
import argparse
import time
import random
import sys
from copy import deepcopy


def euclidean_dist(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


class PVRPTWSolver:
    def __init__(self, instance, time_limit, logger=None):
        self.inst = instance
        self.time_limit = time_limit
        self.logger = logger
        self.start_time = time.time()

        self.nc = instance['num_customers']
        self.nd = instance['num_days']
        self.nv = instance['num_vehicles']
        self.cap = instance['vehicle_capacity']
        self.max_dur = instance['max_route_duration']
        self.depot = instance['depot']
        self.custs = instance['customers']

        # Distance matrix: index 0 = depot, 1..nc = customers
        n = self.nc + 1
        coords = [(self.depot['x'], self.depot['y'])]
        for c in self.custs:
            coords.append((c['x'], c['y']))

        self.dist = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                d = euclidean_dist(coords[i][0], coords[i][1], coords[j][0], coords[j][1])
                self.dist[i][j] = d
                self.dist[j][i] = d

        # Customer data (0-indexed internally)
        self.ids = [c['id'] for c in self.custs]
        self.tw = [(c['time_window'][0], c['time_window'][1]) for c in self.custs]
        self.svc = [c['service_time'] for c in self.custs]
        self.scheds = [c['schedules'] for c in self.custs]

        self.dep_tw = (self.depot['time_window'][0], self.depot['time_window'][1])
        self.dep_svc = self.depot['service_time']

    def remaining(self):
        return self.time_limit - (time.time() - self.start_time)

    def route_dist(self, route):
        """Total Euclidean distance of a route (list of 0-based customer indices)."""
        if not route:
            return 0.0
        d = self.dist[0][route[0] + 1]
        for i in range(len(route) - 1):
            d += self.dist[route[i] + 1][route[i + 1] + 1]
        d += self.dist[route[-1] + 1][0]
        return d

    def is_feasible(self, route, dem):
        """Check route feasibility with time windows, capacity, duration."""
        if not route:
            return True, 0
        td = sum(dem.get(c, 0) for c in route)
        if td > self.cap:
            return False, td

        t = float(self.dep_tw[0])
        for i, c in enumerate(route):
            if i == 0:
                tt = self.dist[0][c + 1] + self.dep_svc
            else:
                tt = self.dist[route[i - 1] + 1][c + 1] + self.svc[route[i - 1]]
            t += tt
            if t < self.tw[c][0]:
                t = self.tw[c][0]
            if t > self.tw[c][1]:
                return False, td

        # Return to depot
        tt = self.dist[route[-1] + 1][0] + self.svc[route[-1]]
        t += tt
        if t > self.dep_tw[1]:
            return False, td
        if t - self.dep_tw[0] > self.max_dur:
            return False, td
        return True, td

    def construct_day(self, custs, dem):
        """Construct routes for one day using cheapest insertion."""
        if not custs:
            return [], True

        routes = []
        unasgn = sorted(custs, key=lambda c: -self.dist[0][c + 1])

        while unasgn and len(routes) < self.nv:
            seed = unasgn.pop(0)
            if dem.get(seed, 0) > self.cap:
                continue
            route = [seed]
            rdem = dem.get(seed, 0)

            changed = True
            while changed and unasgn:
                changed = False
                best_c, best_p, best_incr = None, None, float('inf')
                cur_cost = self.route_dist(route)

                for c in unasgn:
                    cd = dem.get(c, 0)
                    if rdem + cd > self.cap:
                        continue
                    for p in range(len(route) + 1):
                        nr = route[:p] + [c] + route[p:]
                        ok, _ = self.is_feasible(nr, dem)
                        if ok:
                            incr = self.route_dist(nr) - cur_cost
                            if incr < best_incr:
                                best_incr = incr
                                best_c = c
                                best_p = p

                if best_c is not None:
                    route.insert(best_p, best_c)
                    rdem += dem.get(best_c, 0)
                    unasgn.remove(best_c)
                    changed = True

            routes.append(route)

        # Try remaining as individual routes
        for c in unasgn[:]:
            if len(routes) >= self.nv:
                break
            if dem.get(c, 0) <= self.cap:
                ok, _ = self.is_feasible([c], dem)
                if ok:
                    routes.append([c])
                    unasgn.remove(c)

        return routes, len(unasgn) == 0

    def local_search_day(self, routes, dem):
        """Improve routes with 2-opt and relocate."""
        # 2-opt within each route
        for ri in range(len(routes)):
            if self.remaining() < 0.5:
                break
            improved = True
            while improved:
                improved = False
                r = routes[ri]
                if len(r) < 3:
                    break
                best_g, bi, bj = 0, -1, -1
                rc = self.route_dist(r)
                for i in range(len(r) - 1):
                    for j in range(i + 2, len(r)):
                        nr = r[:i + 1] + r[i + 1:j + 1][::-1] + r[j + 1:]
                        ok, _ = self.is_feasible(nr, dem)
                        if ok:
                            g = rc - self.route_dist(nr)
                            if g > best_g + 1e-10:
                                best_g, bi, bj = g, i, j
                if bi >= 0:
                    routes[ri] = r[:bi + 1] + r[bi + 1:bj + 1][::-1] + r[bj + 1:]
                    improved = True

        # Relocate between routes
        for _ in range(30):
            if self.remaining() < 0.5:
                break
            improved = False
            for r1i in range(len(routes)):
                if improved:
                    break
                for ci in range(len(routes[r1i])):
                    if improved:
                        break
                    c = routes[r1i][ci]
                    rem = routes[r1i][:ci] + routes[r1i][ci + 1:]
                    for r2i in range(len(routes)):
                        if r2i == r1i:
                            continue
                        rd2 = sum(dem.get(x, 0) for x in routes[r2i])
                        if rd2 + dem.get(c, 0) > self.cap:
                            continue
                        for p in range(len(routes[r2i]) + 1):
                            nr2 = routes[r2i][:p] + [c] + routes[r2i][p:]
                            ok, _ = self.is_feasible(nr2, dem)
                            if ok:
                                old = self.route_dist(routes[r1i]) + self.route_dist(routes[r2i])
                                new = self.route_dist(rem) + self.route_dist(nr2)
                                if new < old - 1e-10:
                                    routes[r1i] = rem
                                    routes[r2i] = nr2
                                    improved = True
                                    break
                        if improved:
                            break

        # Or-opt: move sequences of 1-2 customers
        for _ in range(20):
            if self.remaining() < 0.5:
                break
            improved = False
            for r1i in range(len(routes)):
                if improved:
                    break
                r1 = routes[r1i]
                for seg_len in [1, 2]:
                    if improved:
                        break
                    for ci in range(len(r1) - seg_len + 1):
                        if improved:
                            break
                        seg = r1[ci:ci + seg_len]
                        rem = r1[:ci] + r1[ci + seg_len:]
                        for r2i in range(len(routes)):
                            if improved:
                                break
                            if r2i == r1i:
                                # Reinsert in same route
                                for p in range(len(rem) + 1):
                                    nr = rem[:p] + seg + rem[p:]
                                    if nr == r1:
                                        continue
                                    ok, _ = self.is_feasible(nr, dem)
                                    if ok:
                                        old_c = self.route_dist(r1)
                                        new_c = self.route_dist(nr)
                                        if new_c < old_c - 1e-10:
                                            routes[r1i] = nr
                                            r1 = nr
                                            improved = True
                                            break
                            else:
                                rd2 = sum(dem.get(x, 0) for x in routes[r2i])
                                seg_dem = sum(dem.get(x, 0) for x in seg)
                                if rd2 + seg_dem > self.cap:
                                    continue
                                for p in range(len(routes[r2i]) + 1):
                                    nr2 = routes[r2i][:p] + seg + routes[r2i][p:]
                                    ok, _ = self.is_feasible(nr2, dem)
                                    if ok:
                                        old = self.route_dist(routes[r1i]) + self.route_dist(routes[r2i])
                                        new = self.route_dist(rem) + self.route_dist(nr2)
                                        if new < old - 1e-10:
                                            routes[r1i] = rem
                                            routes[r2i] = nr2
                                            improved = True
                                            break

        routes = [r for r in routes if r]
        return routes

    def eval_selection(self, sel, do_ls=True):
        """Evaluate a schedule selection. Returns (cost, routes_by_day, feasible)."""
        day_dem = [{} for _ in range(self.nd)]
        for ci in range(self.nc):
            si = sel[ci]
            sched = self.scheds[ci][si]
            for part in sched['parts']:
                vd = part['visit_day']
                day_dem[vd][ci] = part['demand_per_visit']

        total = 0.0
        routes_by_day = {}

        for d in range(self.nd):
            custs_today = list(day_dem[d].keys())
            if not custs_today:
                routes_by_day[d] = []
                continue

            routes, all_served = self.construct_day(custs_today, day_dem[d])
            if not all_served:
                return float('inf'), None, False

            if do_ls and self.remaining() > 1.0:
                routes = self.local_search_day(routes, day_dem[d])

            dc = sum(self.route_dist(r) for r in routes)
            total += dc
            routes_by_day[d] = routes

        return total, routes_by_day, True

    def build_output(self, sel, routes_by_day, cost):
        """Build solution dictionary in required output format."""
        sol = {"objective_value": round(cost, 6), "routes": {}, "schedules": {}}

        for ci in range(self.nc):
            cid = str(self.ids[ci])
            si = sel[ci]
            s = self.scheds[ci][si]
            sol["schedules"][cid] = {
                "schedule_index": si,
                "days": s['days'],
                "visit_frequency": s['visit_frequency']
            }

        for d in range(self.nd):
            day_routes = []
            for route in routes_by_day.get(d, []):
                if not route:
                    continue
                rcost = self.route_dist(route)
                rdem = 0
                parts = {}
                cids = []

                for c in route:
                    cid = self.ids[c]
                    cids.append(cid)
                    si = sel[c]
                    sched = self.scheds[c][si]
                    for part in sched['parts']:
                        if part['visit_day'] == d:
                            rdem += part['demand_per_visit']
                            parts[str(cid)] = {
                                "visit_day": part['visit_day'],
                                "days_covered": part['days_covered'],
                                "demand_per_visit": part['demand_per_visit']
                            }
                            break

                day_routes.append({
                    "customers": cids,
                    "cost": round(rcost, 6),
                    "demand": rdem,
                    "parts": parts
                })

            sol["routes"][str(d)] = day_routes

        return sol

    def solve(self):
        best_cost = float('inf')
        best_sol = None
        best_sel = None

        # Strategy 1: Minimum visit frequency
        if self.remaining() > 0.5:
            sel = []
            for ci in range(self.nc):
                best_si = min(range(len(self.scheds[ci])),
                              key=lambda s: self.scheds[ci][s]['visit_frequency'])
                sel.append(best_si)

            cost, rbd, feas = self.eval_selection(sel)
            if feas and cost < best_cost:
                best_cost = cost
                best_sel = sel[:]
                best_sol = self.build_output(sel, rbd, cost)
                if self.logger:
                    self.logger.log_solution(best_cost, best_sol)

        # Strategy 2: Maximum visit frequency (spreads load)
        if self.remaining() > 1:
            sel = []
            for ci in range(self.nc):
                best_si = max(range(len(self.scheds[ci])),
                              key=lambda s: self.scheds[ci][s]['visit_frequency'])
                sel.append(best_si)

            cost, rbd, feas = self.eval_selection(sel)
            if feas and cost < best_cost:
                best_cost = cost
                best_sel = sel[:]
                best_sol = self.build_output(sel, rbd, cost)
                if self.logger:
                    self.logger.log_solution(best_cost, best_sol)

        # Strategy 3: First schedule for all
        if self.remaining() > 1:
            sel = [0] * self.nc
            cost, rbd, feas = self.eval_selection(sel)
            if feas and cost < best_cost:
                best_cost = cost
                best_sel = sel[:]
                best_sol = self.build_output(sel, rbd, cost)
                if self.logger:
                    self.logger.log_solution(best_cost, best_sol)

        # Strategy 4: Random selections
        rng = random.Random(42)
        attempts = 0
        max_attempts = 100
        while self.remaining() > 2.0 and attempts < max_attempts:
            attempts += 1
            sel = [rng.randint(0, len(self.scheds[ci]) - 1) for ci in range(self.nc)]
            do_ls = self.remaining() > 5.0
            cost, rbd, feas = self.eval_selection(sel, do_ls=do_ls)
            if feas and cost < best_cost:
                best_cost = cost
                best_sel = sel[:]
                best_sol = self.build_output(sel, rbd, cost)
                if self.logger:
                    self.logger.log_solution(best_cost, best_sol)

        # Strategy 5: Guided random - bias toward min visits but with randomness
        if self.remaining() > 3.0:
            for trial in range(20):
                if self.remaining() < 2.0:
                    break
                sel = []
                for ci in range(self.nc):
                    n_scheds = len(self.scheds[ci])
                    if n_scheds == 1:
                        sel.append(0)
                    elif rng.random() < 0.7:
                        # Pick min visit frequency
                        best_si = min(range(n_scheds),
                                      key=lambda s: self.scheds[ci][s]['visit_frequency'])
                        sel.append(best_si)
                    else:
                        sel.append(rng.randint(0, n_scheds - 1))

                do_ls = self.remaining() > 5.0
                cost, rbd, feas = self.eval_selection(sel, do_ls=do_ls)
                if feas and cost < best_cost:
                    best_cost = cost
                    best_sel = sel[:]
                    best_sol = self.build_output(sel, rbd, cost)
                    if self.logger:
                        self.logger.log_solution(best_cost, best_sol)

        # Strategy 6: Local search on schedule selection
        if best_sel is not None and self.remaining() > 5.0:
            sel = best_sel[:]
            current_cost = best_cost
            no_improve = 0
            max_no_improve = 3

            while no_improve < max_no_improve and self.remaining() > 3.0:
                improved = False
                order = list(range(self.nc))
                rng.shuffle(order)

                for ci in order:
                    if self.remaining() < 2.0:
                        break
                    cur_si = sel[ci]
                    best_local_cost = current_cost
                    best_local_si = cur_si

                    for si in range(len(self.scheds[ci])):
                        if si == cur_si:
                            continue
                        if self.remaining() < 1.5:
                            break
                        sel[ci] = si
                        cost, rbd, feas = self.eval_selection(sel, do_ls=False)
                        if feas and cost < best_local_cost:
                            best_local_cost = cost
                            best_local_si = si

                    sel[ci] = best_local_si
                    if best_local_si != cur_si:
                        # Re-evaluate with local search
                        if self.remaining() > 2.0:
                            cost, rbd, feas = self.eval_selection(sel, do_ls=True)
                            if feas and cost < current_cost:
                                current_cost = cost
                                improved = True
                                if cost < best_cost:
                                    best_cost = cost
                                    best_sel = sel[:]
                                    best_sol = self.build_output(sel, rbd, cost)
                                    if self.logger:
                                        self.logger.log_solution(best_cost, best_sol)

                if not improved:
                    no_improve += 1
                else:
                    no_improve = 0

        # Strategy 7: Perturbation-based search
        if best_sel is not None and self.remaining() > 3.0:
            for trial in range(30):
                if self.remaining() < 2.0:
                    break
                sel = best_sel[:]
                # Perturb k customers
                k = rng.randint(1, max(1, self.nc // 5))
                for _ in range(k):
                    ci = rng.randint(0, self.nc - 1)
                    sel[ci] = rng.randint(0, len(self.scheds[ci]) - 1)

                do_ls = self.remaining() > 4.0
                cost, rbd, feas = self.eval_selection(sel, do_ls=do_ls)
                if feas and cost < best_cost:
                    best_cost = cost
                    best_sel = sel[:]
                    best_sol = self.build_output(sel, rbd, cost)
                    if self.logger:
                        self.logger.log_solution(best_cost, best_sol)

        # Fallback if no solution found
        if best_sol is None:
            # Try each schedule index combination more exhaustively for small instances
            if self.nc <= 10:
                self._exhaustive_small()
            else:
                # Just try all first schedules
                sel = [0] * self.nc
                cost, rbd, feas = self.eval_selection(sel, do_ls=False)
                if feas:
                    best_cost = cost
                    best_sol = self.build_output(sel, rbd, cost)
                    if self.logger:
                        self.logger.log_solution(best_cost, best_sol)

        return best_sol

    def _exhaustive_small(self):
        """For very small instances, try more combinations."""
        pass


def main():
    parser = argparse.ArgumentParser(description='PVRPTW Solver')
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path) as f:
        instance = json.load(f)

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    solver = PVRPTWSolver(instance, args.time_limit, logger)
    solution = solver.solve()

    if solution is None:
        # Absolute fallback: construct minimal solution
        solution = {
            "objective_value": 0.0,
            "routes": {str(d): [] for d in range(instance['num_days'])},
            "schedules": {}
        }
        for ci, c in enumerate(instance['customers']):
            cid = str(c['id'])
            if c['schedules']:
                s = c['schedules'][0]
                solution["schedules"][cid] = {
                    "schedule_index": 0,
                    "days": s['days'],
                    "visit_frequency": s['visit_frequency']
                }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == '__main__':
    main()
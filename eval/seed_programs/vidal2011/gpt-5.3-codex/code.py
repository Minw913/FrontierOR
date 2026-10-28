import argparse
import json
import math
import random
import time
from collections import defaultdict

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def euclidean(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


class MDPVRPSolver:
    def __init__(self, inst, time_limit, logger=None):
        self.inst = inst
        self.n = inst["n"]
        self.d = inst["d"]
        self.t = inst["t"]
        self.m = inst["m"]
        self.Q = inst["vehicle_capacity"]
        self.D = inst["route_duration_limit"]
        self.depots = inst["depots"]
        self.customers = inst["customers"]
        self.time_limit = time_limit
        self.logger = logger
        self.rng = random.Random(0)

        self.start_time = time.time()
        self.deadline = self.start_time + time_limit - 0.05

        self.dep_xy = [(dp["x"], dp["y"]) for dp in self.depots]
        self.cus_xy = [(c["x"], c["y"]) for c in self.customers]
        self.dem = [c["demand"] for c in self.customers]
        self.srv = [c["service_duration"] for c in self.customers]
        self.patterns = []
        for c in self.customers:
            pats = []
            for p in c["pattern_list"]:
                pats.append(tuple(sorted(int(x) for x in p)))
            # remove duplicates while preserving order
            seen = set()
            uniq = []
            for p in pats:
                if p not in seen:
                    seen.add(p)
                    uniq.append(p)
            self.patterns.append(uniq)

        self.dist_dc = [[euclidean(self.dep_xy[p], self.cus_xy[i]) for i in range(self.n)] for p in range(self.d)]
        self.dist_cc = [[euclidean(self.cus_xy[i], self.cus_xy[j]) for j in range(self.n)] for i in range(self.n)]

        # options per customer: list[(depot, pattern_tuple)]
        self.options = []
        for i in range(self.n):
            opts = []
            for dep in range(self.d):
                for pat in self.patterns[i]:
                    opts.append((dep, pat))
            self.options.append(opts)

        # scale for greedy penalty
        all_d = [self.dist_dc[p][i] for p in range(self.d) for i in range(self.n)]
        self.dist_scale = sum(all_d) / max(1, len(all_d))

    def time_left(self):
        return self.deadline - time.time()

    # ---------- route utilities ----------
    def route_travel_cost(self, dep, route):
        if not route:
            return 0.0
        c = self.dist_dc[dep][route[0]]
        for a, b in zip(route, route[1:]):
            c += self.dist_cc[a][b]
        c += self.dist_dc[dep][route[-1]]
        return c

    def route_duration(self, dep, route):
        return self.route_travel_cost(dep, route) + sum(self.srv[i] for i in route)

    def route_demand(self, route):
        return sum(self.dem[i] for i in route)

    def route_feasible(self, dep, route):
        return self.route_demand(route) <= self.Q + 1e-9 and self.route_duration(dep, route) <= self.D + 1e-9

    def two_opt(self, dep, route):
        if len(route) < 4:
            return route
        improved = True
        best = route[:]
        best_cost = self.route_travel_cost(dep, best)
        while improved and self.time_left() > 0:
            improved = False
            n = len(best)
            for i in range(0, n - 2):
                for j in range(i + 2, n):
                    if j - i == 1:
                        continue
                    new_r = best[:i + 1] + list(reversed(best[i + 1:j + 1])) + best[j + 1:]
                    new_cost = self.route_travel_cost(dep, new_r)
                    if new_cost + 1e-9 < best_cost and self.route_duration(dep, new_r) <= self.D + 1e-9:
                        best = new_r
                        best_cost = new_cost
                        improved = True
                        break
                if improved:
                    break
        return best

    # ---------- routing heuristics ----------
    def order_sweep(self, dep, customers, reverse=False):
        dx, dy = self.dep_xy[dep]
        arr = []
        for c in customers:
            x, y = self.cus_xy[c]
            ang = math.atan2(y - dy, x - dx)
            arr.append((ang, c))
        arr.sort(reverse=reverse)
        return [c for _, c in arr]

    def order_nearest_neighbor(self, dep, customers):
        un = set(customers)
        order = []
        # start farthest from depot
        cur = max(un, key=lambda i: self.dist_dc[dep][i])
        order.append(cur)
        un.remove(cur)
        while un:
            nxt = min(un, key=lambda j: self.dist_cc[cur][j])
            order.append(nxt)
            un.remove(nxt)
            cur = nxt
        return order

    def split_order_dp(self, dep, order):
        n = len(order)
        if n == 0:
            return True, [], 0.0

        pref_dem = [0] * (n + 1)
        pref_srv = [0.0] * (n + 1)
        for i in range(n):
            c = order[i]
            pref_dem[i + 1] = pref_dem[i] + self.dem[c]
            pref_srv[i + 1] = pref_srv[i] + self.srv[c]

        pref_tr = [0.0] * n  # between consecutive order points
        for i in range(1, n):
            pref_tr[i] = pref_tr[i - 1] + self.dist_cc[order[i - 1]][order[i]]

        feasible_seg = [[False] * n for _ in range(n)]
        seg_cost = [[0.0] * n for _ in range(n)]
        for a in range(n):
            for b in range(a, n):
                dem = pref_dem[b + 1] - pref_dem[a]
                if dem > self.Q + 1e-9:
                    continue
                travel = self.dist_dc[dep][order[a]]
                if b > a:
                    travel += pref_tr[b] - pref_tr[a]
                travel += self.dist_dc[dep][order[b]]
                dur = travel + (pref_srv[b + 1] - pref_srv[a])
                if dur <= self.D + 1e-9:
                    feasible_seg[a][b] = True
                    seg_cost[a][b] = travel

        INF = 1e100
        dp = [[INF] * (n + 1) for _ in range(self.m + 1)]
        prev = [[None] * (n + 1) for _ in range(self.m + 1)]
        dp[0][0] = 0.0

        for r in range(1, self.m + 1):
            for j in range(1, n + 1):
                best = INF
                best_i = None
                for i in range(0, j):
                    if dp[r - 1][i] >= INF:
                        continue
                    if feasible_seg[i][j - 1]:
                        val = dp[r - 1][i] + seg_cost[i][j - 1]
                        if val < best:
                            best = val
                            best_i = i
                dp[r][j] = best
                prev[r][j] = best_i

        best_r = None
        best_cost = INF
        for r in range(1, self.m + 1):
            if dp[r][n] < best_cost:
                best_cost = dp[r][n]
                best_r = r
        if best_r is None or best_cost >= INF:
            return False, None, None

        routes = []
        r = best_r
        j = n
        while j > 0:
            i = prev[r][j]
            if i is None:
                return False, None, None
            routes.append(order[i:j])
            j = i
            r -= 1
        routes.reverse()

        # local 2-opt improvement
        for idx in range(len(routes)):
            routes[idx] = self.two_opt(dep, routes[idx])
            if not self.route_feasible(dep, routes[idx]):
                return False, None, None

        cost = sum(self.route_travel_cost(dep, rt) for rt in routes)
        return True, routes, cost

    def insertion_construct(self, dep, customers, attempts=20):
        best = None
        best_cost = 1e100
        cust_list = list(customers)

        for a in range(attempts):
            if self.time_left() <= 0:
                break
            if a == 0:
                order = sorted(cust_list, key=lambda c: self.dist_dc[dep][c], reverse=True)
            elif a == 1:
                order = self.order_sweep(dep, cust_list)
            else:
                order = cust_list[:]
                self.rng.shuffle(order)

            routes = []
            feasible = True
            for c in order:
                best_ins = None  # (delta, ridx, pos, new_route)
                # try insert in existing routes
                for ridx, rt in enumerate(routes):
                    for pos in range(len(rt) + 1):
                        new_rt = rt[:pos] + [c] + rt[pos:]
                        if self.route_demand(new_rt) > self.Q + 1e-9:
                            continue
                        dur = self.route_duration(dep, new_rt)
                        if dur > self.D + 1e-9:
                            continue
                        old_cost = self.route_travel_cost(dep, rt)
                        new_cost = self.route_travel_cost(dep, new_rt)
                        delta = new_cost - old_cost
                        if best_ins is None or delta < best_ins[0]:
                            best_ins = (delta, ridx, pos, new_rt)

                if best_ins is not None:
                    _, ridx, _, new_rt = best_ins
                    routes[ridx] = new_rt
                else:
                    if len(routes) < self.m and self.route_feasible(dep, [c]):
                        routes.append([c])
                    else:
                        feasible = False
                        break

            if feasible:
                # improve
                for r in range(len(routes)):
                    routes[r] = self.two_opt(dep, routes[r])
                    if not self.route_feasible(dep, routes[r]):
                        feasible = False
                        break
            if feasible:
                cost = sum(self.route_travel_cost(dep, rt) for rt in routes)
                if cost < best_cost:
                    best_cost = cost
                    best = [rt[:] for rt in routes]

        if best is None:
            return False, None, None
        return True, best, best_cost

    def solve_day_routes(self, dep, day_customers):
        customers = list(day_customers)
        if not customers:
            return True, [], 0.0

        # quick impossible checks
        if sum(self.dem[c] for c in customers) > self.m * self.Q + 1e-9:
            return False, None, None
        for c in customers:
            if self.dem[c] > self.Q + 1e-9:
                return False, None, None
            if self.route_duration(dep, [c]) > self.D + 1e-9:
                return False, None, None

        candidates = []
        # deterministic candidate orders
        candidates.append(self.order_sweep(dep, customers, reverse=False))
        candidates.append(self.order_sweep(dep, customers, reverse=True))
        candidates.append(self.order_nearest_neighbor(dep, customers))

        # randomized sweep jitter
        base = self.order_sweep(dep, customers, reverse=False)
        rand_trials = 8 if len(customers) <= 60 else 4
        for _ in range(rand_trials):
            arr = base[:]
            self.rng.shuffle(arr)
            candidates.append(arr)

        best_routes = None
        best_cost = 1e100
        for ord_ in candidates:
            if self.time_left() <= 0:
                break
            ok, routes, cost = self.split_order_dp(dep, ord_)
            if ok and cost < best_cost:
                best_cost = cost
                best_routes = routes

        # insertion fallback
        ok2, routes2, cost2 = self.insertion_construct(dep, customers, attempts=18)
        if ok2 and cost2 < best_cost:
            best_cost = cost2
            best_routes = routes2

        if best_routes is None:
            return False, None, None
        return True, best_routes, best_cost

    # ---------- assignment ----------
    def initial_assignment_mip(self, tl):
        model = gp.Model("assign")
        model.Params.OutputFlag = 0
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1
        model.Params.TimeLimit = max(1.0, tl)

        x = {}
        for i in range(self.n):
            for o, (dep, pat) in enumerate(self.options[i]):
                x[(i, o)] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{o}")

        # assign each customer once
        for i in range(self.n):
            model.addConstr(gp.quicksum(x[(i, o)] for o in range(len(self.options[i]))) == 1)

        # aggregate daily capacity / duration proxies
        for dep in range(self.d):
            for day in range(1, self.t + 1):
                model.addConstr(
                    gp.quicksum(
                        self.dem[i] * x[(i, o)]
                        for i in range(self.n)
                        for o, (dp, pat) in enumerate(self.options[i])
                        if dp == dep and day in pat
                    ) <= self.m * self.Q
                )
                model.addConstr(
                    gp.quicksum(
                        (self.srv[i] + 2.0 * self.dist_dc[dep][i]) * x[(i, o)]
                        for i in range(self.n)
                        for o, (dp, pat) in enumerate(self.options[i])
                        if dp == dep and day in pat
                    ) <= self.m * self.D
                )

        # objective proxy
        obj = gp.LinExpr()
        for i in range(self.n):
            for o, (dep, pat) in enumerate(self.options[i]):
                c = sum(2.0 * self.dist_dc[dep][i] for _ in pat)
                obj += c * x[(i, o)]
        model.setObjective(obj, GRB.MINIMIZE)

        try:
            model.optimize()
        except gp.GurobiError:
            return None

        if model.SolCount == 0:
            return None

        assignment = {}
        for i in range(self.n):
            chosen = None
            for o in range(len(self.options[i])):
                if x[(i, o)].X > 0.5:
                    chosen = self.options[i][o]
                    break
            if chosen is None:
                chosen = self.options[i][0]
            assignment[i] = chosen
        return assignment

    def greedy_assignment(self, seed_shift=0):
        rng = random.Random(1000 + seed_shift)
        assignment = {}
        load_dem = [[0.0 for _ in range(self.t + 1)] for _ in range(self.d)]
        load_dur = [[0.0 for _ in range(self.t + 1)] for _ in range(self.d)]

        cust_order = list(range(self.n))
        # harder first
        cust_order.sort(key=lambda i: (self.dem[i], self.srv[i], len(self.patterns[i])), reverse=True)
        rng.shuffle(cust_order[:max(1, self.n // 3)])

        beta = 6.0 * self.dist_scale
        for i in cust_order:
            best = None
            best_score = 1e100
            opts = self.options[i][:]
            rng.shuffle(opts)
            for dep, pat in opts:
                base = sum(2.0 * self.dist_dc[dep][i] for _ in pat)
                penalty = 0.0
                for day in pat:
                    nd = load_dem[dep][day] + self.dem[i]
                    nu = load_dur[dep][day] + (self.srv[i] + 2.0 * self.dist_dc[dep][i])
                    rd = nd / (self.m * self.Q + 1e-9)
                    ru = nu / (self.m * self.D + 1e-9)
                    penalty += beta * (rd ** 4 + ru ** 4)
                sc = base + penalty
                if sc < best_score:
                    best_score = sc
                    best = (dep, pat)
            assignment[i] = best
            dep, pat = best
            for day in pat:
                load_dem[dep][day] += self.dem[i]
                load_dur[dep][day] += self.srv[i] + 2.0 * self.dist_dc[dep][i]
        return assignment

    # ---------- build full solution ----------
    def build_active_sets(self, assignment):
        active = [[set() for _ in range(self.t + 1)] for _ in range(self.d)]
        for i in range(self.n):
            dep, pat = assignment[i]
            for day in pat:
                active[dep][day].add(i)
        return active

    def route_all_days(self, assignment):
        active = self.build_active_sets(assignment)
        routes_map = {}
        costs_map = {}
        infeas = []

        keys = [(dep, day) for dep in range(self.d) for day in range(1, self.t + 1)]
        # route larger sets first (fail fast)
        keys.sort(key=lambda k: len(active[k[0]][k[1]]), reverse=True)

        for dep, day in keys:
            if self.time_left() <= 0:
                return False, None, None, None
            ok, routes, cost = self.solve_day_routes(dep, active[dep][day])
            if not ok:
                infeas.append((dep, day))
            else:
                routes_map[(dep, day)] = routes
                costs_map[(dep, day)] = cost

        feasible = len(infeas) == 0
        return feasible, routes_map, costs_map, active

    def objective_from_costs(self, costs_map):
        return float(sum(costs_map.values()))

    def make_solution_dict(self, assignment, routes_map, obj):
        routes_out = []
        for dep in range(self.d):
            for day in range(1, self.t + 1):
                rts = routes_map.get((dep, day), [])
                for v_idx, rt in enumerate(rts, start=1):
                    routes_out.append({
                        "depot": dep,
                        "period": day,
                        "vehicle": v_idx,
                        "customers": [int(c) for c in rt]
                    })

        assign_out = {}
        for i in range(self.n):
            dep, pat = assignment[i]
            assign_out[str(i)] = {
                "depot": int(dep),
                "pattern": [int(x) for x in pat]
            }

        return {
            "objective_value": float(obj),
            "routes": routes_out,
            "assignments": assign_out
        }

    def local_improve_assignment(self, assignment, routes_map, costs_map):
        # simple hill-climb: single-customer option change, reroute impacted days
        current_obj = self.objective_from_costs(costs_map)
        active = self.build_active_sets(assignment)
        improved = True

        while improved and self.time_left() > 0:
            improved = False
            custs = list(range(self.n))
            self.rng.shuffle(custs)

            for i in custs:
                if self.time_left() <= 0:
                    break
                cur_dep, cur_pat = assignment[i]
                cur_opts = self.options[i]
                self.rng.shuffle(cur_opts)

                for new_dep, new_pat in cur_opts:
                    if (new_dep, new_pat) == (cur_dep, cur_pat):
                        continue
                    # quick aggregate demand filter on affected days
                    bad = False
                    affected = set()
                    for dday in cur_pat:
                        affected.add((cur_dep, dday))
                    for dday in new_pat:
                        affected.add((new_dep, dday))

                    # apply temporary active update
                    for dday in cur_pat:
                        active[cur_dep][dday].remove(i)
                    for dday in new_pat:
                        active[new_dep][dday].add(i)

                    new_routes = {}
                    new_costs = {}
                    feasible = True
                    for key in affected:
                        dep, day = key
                        if sum(self.dem[c] for c in active[dep][day]) > self.m * self.Q + 1e-9:
                            feasible = False
                            break
                        ok, rts, cst = self.solve_day_routes(dep, active[dep][day])
                        if not ok:
                            feasible = False
                            break
                        new_routes[key] = rts
                        new_costs[key] = cst

                    if feasible:
                        cand_obj = current_obj - sum(costs_map.get(k, 0.0) for k in affected) + sum(new_costs.values())
                        if cand_obj + 1e-9 < current_obj:
                            assignment[i] = (new_dep, new_pat)
                            for k, v in new_routes.items():
                                routes_map[k] = v
                            for k, v in new_costs.items():
                                costs_map[k] = v
                            current_obj = cand_obj
                            improved = True

                            sol = self.make_solution_dict(assignment, routes_map, current_obj)
                            if self.logger:
                                self.logger.log_solution(current_obj, sol)
                            break

                    # revert active update
                    for dday in new_pat:
                        active[new_dep][dday].remove(i)
                    for dday in cur_pat:
                        active[cur_dep][dday].add(i)

                if improved:
                    break

        return assignment, routes_map, costs_map

    def solve(self):
        best_sol = None
        best_obj = 1e100

        # candidate assignments: 1 MIP + several greedy randomized
        assignments = []
        mip_tl = max(1, int(self.time_limit * 0.2))
        if self.time_left() > 1.0:
            a0 = self.initial_assignment_mip(min(mip_tl, max(1, int(self.time_left() - 0.5))))
            if a0 is not None:
                assignments.append(a0)

        # multiple greedy starts
        starts = 10
        for s in range(starts):
            if self.time_left() <= 0:
                break
            assignments.append(self.greedy_assignment(seed_shift=s))

        for ass in assignments:
            if self.time_left() <= 0:
                break
            feasible, routes_map, costs_map, _ = self.route_all_days(ass)
            if not feasible:
                continue
            obj = self.objective_from_costs(costs_map)
            if obj + 1e-9 < best_obj:
                best_obj = obj
                # copy structures
                best_assignment = {i: ass[i] for i in range(self.n)}
                best_routes = {k: [rt[:] for rt in v] for k, v in routes_map.items()}
                best_costs = dict(costs_map)
                best_sol = self.make_solution_dict(best_assignment, best_routes, best_obj)
                if self.logger:
                    self.logger.log_solution(best_obj, best_sol)

        # if feasible found, improve
        if best_sol is not None and self.time_left() > 0.5:
            best_assignment = {i: (best_sol["assignments"][str(i)]["depot"],
                                   tuple(best_sol["assignments"][str(i)]["pattern"])) for i in range(self.n)}
            # reconstruct map
            routes_map = defaultdict(list)
            for r in best_sol["routes"]:
                routes_map[(r["depot"], r["period"])].append(list(r["customers"]))
            costs_map = {}
            for dep in range(self.d):
                for day in range(1, self.t + 1):
                    costs_map[(dep, day)] = sum(self.route_travel_cost(dep, rt) for rt in routes_map.get((dep, day), []))

            best_assignment, routes_map, costs_map = self.local_improve_assignment(best_assignment, routes_map, costs_map)
            best_obj = self.objective_from_costs(costs_map)
            best_sol = self.make_solution_dict(best_assignment, routes_map, best_obj)

        # fallback (very unlikely): output a greedy assignment with empty routes if still none
        if best_sol is None:
            ass = self.greedy_assignment(seed_shift=999)
            feasible, routes_map, costs_map, _ = self.route_all_days(ass)
            if feasible:
                obj = self.objective_from_costs(costs_map)
                best_sol = self.make_solution_dict(ass, routes_map, obj)
                if self.logger:
                    self.logger.log_solution(obj, best_sol)
            else:
                # last-resort schema-valid output (may be infeasible for the optimization problem)
                assign_out = {}
                for i in range(self.n):
                    dep, pat = ass[i]
                    assign_out[str(i)] = {"depot": int(dep), "pattern": list(map(int, pat))}
                best_sol = {"objective_value": float(1e18), "routes": [], "assignments": assign_out}

        return best_sol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    solver = MDPVRPSolver(instance, args.time_limit, logger=logger)
    sol = solver.solve()

    with open(args.solution_path, "w") as f:
        json.dump(sol, f, indent=2)


if __name__ == "__main__":
    main()
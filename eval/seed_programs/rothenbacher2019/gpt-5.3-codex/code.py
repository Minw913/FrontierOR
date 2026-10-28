import argparse
import json
import math
import random
import time
from copy import deepcopy

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def euclid(a, b):
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


def detect_day_list(instance):
    num_days = instance["num_days"]
    day_vals = set()
    for c in instance["customers"]:
        for s in c["schedules"]:
            for p in s.get("parts", []):
                day_vals.add(int(p["visit_day"]))
            for d in s.get("days", []):
                day_vals.add(int(d))
    if 0 in day_vals:
        return list(range(0, num_days))
    return list(range(1, num_days + 1))


class Solver:
    def __init__(self, instance, time_limit, logger=None):
        self.instance = instance
        self.time_limit = time_limit
        self.logger = logger
        self.start_time = time.time()

        self.num_days = instance["num_days"]
        self.num_vehicles = instance["num_vehicles"]
        self.vehicle_capacity = instance["vehicle_capacity"]
        self.max_route_duration = instance["max_route_duration"]

        self.depot = instance["depot"]
        self.customers = instance["customers"]

        self.customer_ids = [c["id"] for c in self.customers]
        self.cust_by_id = {c["id"]: c for c in self.customers}

        self.day_list = detect_day_list(instance)

        self.nodes = {self.depot["id"]: self.depot}
        for c in self.customers:
            self.nodes[c["id"]] = c

        self.dist = {}
        self.travel_time = {}
        for i in self.nodes:
            self.dist[i] = {}
            self.travel_time[i] = {}
            for j in self.nodes:
                d = euclid(self.nodes[i], self.nodes[j])
                self.dist[i][j] = d
                self.travel_time[i][j] = d + self.nodes[i]["service_time"]

        self.base_schedule_cost = self._compute_base_schedule_costs()
        random.seed(0)

    def elapsed(self):
        return time.time() - self.start_time

    def remaining(self):
        return max(0.0, self.time_limit - self.elapsed())

    def _compute_base_schedule_costs(self):
        depot_id = self.depot["id"]
        base = {}
        for c in self.customers:
            cid = c["id"]
            base[cid] = []
            rt = 2.0 * self.dist[depot_id][cid]
            for s in c["schedules"]:
                v = len(s.get("parts", []))
                if v == 0:
                    v = s.get("visit_frequency", 0)
                base[cid].append(rt * v)
        return base

    def initial_selection_mip(self):
        # Quick MIP for schedule selection with daily overload penalties.
        if self.remaining() <= 0:
            return None

        try:
            m = gp.Model("schedule_selection")
            m.Params.OutputFlag = 0
            m.Params.Threads = 1
            m.Params.TimeLimit = min(10.0, max(1.0, 0.25 * self.remaining()))

            y = {}
            for c in self.customers:
                cid = c["id"]
                for sidx, _ in enumerate(c["schedules"]):
                    y[cid, sidx] = m.addVar(vtype=GRB.BINARY, name=f"y_{cid}_{sidx}")

            over = {d: m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"over_{d}") for d in self.day_list}

            # exactly one schedule per customer
            for c in self.customers:
                cid = c["id"]
                m.addConstr(gp.quicksum(y[cid, sidx] for sidx in range(len(c["schedules"]))) == 1)

            # daily capacity soft constraints
            cap_day = self.num_vehicles * self.vehicle_capacity
            for d in self.day_list:
                expr = gp.LinExpr()
                for c in self.customers:
                    cid = c["id"]
                    for sidx, s in enumerate(c["schedules"]):
                        q = 0
                        for p in s.get("parts", []):
                            if int(p["visit_day"]) == d:
                                q += p["demand_per_visit"]
                        if q > 0:
                            expr += q * y[cid, sidx]
                m.addConstr(expr <= cap_day + over[d])

            obj = gp.LinExpr()
            for c in self.customers:
                cid = c["id"]
                for sidx, _ in enumerate(c["schedules"]):
                    obj += self.base_schedule_cost[cid][sidx] * y[cid, sidx]
            obj += 10000.0 * gp.quicksum(over[d] for d in self.day_list)

            m.setObjective(obj, GRB.MINIMIZE)
            m.optimize()

            if m.SolCount == 0:
                return None

            selection = {}
            for c in self.customers:
                cid = c["id"]
                best = 0
                bestv = -1.0
                for sidx, _ in enumerate(c["schedules"]):
                    v = y[cid, sidx].X
                    if v > bestv:
                        bestv = v
                        best = sidx
                selection[cid] = best
            return selection
        except Exception:
            return None

    def greedy_selection(self):
        sel = {}
        for c in self.customers:
            cid = c["id"]
            costs = self.base_schedule_cost[cid]
            best = min(range(len(costs)), key=lambda i: costs[i])
            sel[cid] = best
        return sel

    def random_selection(self):
        sel = {}
        for c in self.customers:
            cid = c["id"]
            k = len(c["schedules"])
            if k == 1:
                sel[cid] = 0
                continue
            weights = []
            for sidx in range(k):
                w = 1.0 / (1.0 + self.base_schedule_cost[cid][sidx])
                weights.append(w)
            s = sum(weights)
            r = random.random() * s
            acc = 0.0
            pick = 0
            for i, w in enumerate(weights):
                acc += w
                if acc >= r:
                    pick = i
                    break
            sel[cid] = pick
        return sel

    def perturb_selection(self, base_sel, strength=0.1):
        sel = dict(base_sel)
        n = len(self.customer_ids)
        k = max(1, int(strength * n))
        change_customers = random.sample(self.customer_ids, min(k, n))
        for cid in change_customers:
            schs = self.cust_by_id[cid]["schedules"]
            if len(schs) <= 1:
                continue
            cur = sel[cid]
            opts = [i for i in range(len(schs)) if i != cur]
            sel[cid] = random.choice(opts)
        return sel

    def build_daily_visits(self, selection):
        # day -> customer -> visit info
        daily = {d: {} for d in self.day_list}
        for c in self.customers:
            cid = c["id"]
            sidx = selection[cid]
            s = c["schedules"][sidx]
            for p in s.get("parts", []):
                d = int(p["visit_day"])
                if d not in daily:
                    continue
                if cid in daily[d]:
                    # combine if duplicated day in same schedule (rare)
                    daily[d][cid]["demand_per_visit"] += p["demand_per_visit"]
                    daily[d][cid]["days_covered"] += p["days_covered"]
                else:
                    daily[d][cid] = {
                        "visit_day": d,
                        "days_covered": int(p["days_covered"]),
                        "demand_per_visit": float(p["demand_per_visit"]),
                    }
        return daily

    def route_cost_distance(self, seq):
        depot_id = self.depot["id"]
        if not seq:
            return 0.0
        c = self.dist[depot_id][seq[0]]
        for i in range(len(seq) - 1):
            c += self.dist[seq[i]][seq[i + 1]]
        c += self.dist[seq[-1]][depot_id]
        return c

    def build_route(self, seq, day_visits):
        # capacity
        demand = 0.0
        seen = set()
        for cid in seq:
            if cid in seen:
                return None
            seen.add(cid)
            demand += day_visits[cid]["demand_per_visit"]
        if demand > self.vehicle_capacity + 1e-9:
            return None

        depot_id = self.depot["id"]
        a0, b0 = self.depot["time_window"][0], self.depot["time_window"][1]

        # latest times backward
        n = len(seq)
        L = [0.0] * n
        last = seq[-1]
        b_last = min(self.cust_by_id[last]["time_window"][1], b0 - self.travel_time[last][depot_id])
        L[-1] = b_last
        if L[-1] < self.cust_by_id[last]["time_window"][0] - 1e-9:
            return None

        for idx in range(n - 2, -1, -1):
            i = seq[idx]
            j = seq[idx + 1]
            bi = self.cust_by_id[i]["time_window"][1]
            L[idx] = min(bi, L[idx + 1] - self.travel_time[i][j])
            ai = self.cust_by_id[i]["time_window"][0]
            if L[idx] < ai - 1e-9:
                return None

        latest_dep = min(b0, L[0] - self.travel_time[depot_id][seq[0]])
        if latest_dep < a0 - 1e-9:
            return None

        # choose latest feasible departure to reduce waiting/duration
        s = latest_dep

        # forward earliest starts with fixed departure s
        t = [0.0] * n
        first = seq[0]
        t[0] = max(self.cust_by_id[first]["time_window"][0], s + self.travel_time[depot_id][first])
        if t[0] > min(L[0], self.cust_by_id[first]["time_window"][1]) + 1e-9:
            return None

        for idx in range(1, n):
            prev = seq[idx - 1]
            cur = seq[idx]
            t[idx] = max(self.cust_by_id[cur]["time_window"][0], t[idx - 1] + self.travel_time[prev][cur])
            if t[idx] > min(L[idx], self.cust_by_id[cur]["time_window"][1]) + 1e-9:
                return None

        ret = t[-1] + self.travel_time[seq[-1]][depot_id]
        if ret > b0 + 1e-9:
            return None

        dur = ret - s
        if dur > self.max_route_duration + 1e-9:
            return None

        cost = self.route_cost_distance(seq)
        return {
            "seq": list(seq),
            "cost": float(cost),
            "demand": float(demand),
            "duration": float(dur),
        }

    def solve_day_routes(self, day, day_visits):
        # no visits
        if not day_visits:
            return []

        # initial singleton routes
        routes = []
        for cid in day_visits.keys():
            r = self.build_route([cid], day_visits)
            if r is None:
                return None
            routes.append(r)

        # merge heuristic
        K = self.num_vehicles
        while True:
            best = None
            best_saving = -1e18
            m = len(routes)
            for i in range(m):
                for j in range(i + 1, m):
                    a = routes[i]["seq"]
                    b = routes[j]["seq"]
                    candidates = [
                        a + b,
                        a + list(reversed(b)),
                        list(reversed(a)) + b,
                        list(reversed(a)) + list(reversed(b)),
                    ]
                    # deduplicate
                    unique = []
                    seen = set()
                    for cand in candidates:
                        tup = tuple(cand)
                        if tup not in seen:
                            seen.add(tup)
                            unique.append(cand)

                    for cand in unique:
                        rnew = self.build_route(cand, day_visits)
                        if rnew is None:
                            continue
                        saving = routes[i]["cost"] + routes[j]["cost"] - rnew["cost"]
                        if saving > best_saving:
                            best_saving = saving
                            best = (i, j, rnew)

            if best is None:
                break

            if len(routes) <= K and best_saving <= 1e-9:
                break

            i, j, rnew = best
            if i > j:
                i, j = j, i
            new_routes = []
            for idx, r in enumerate(routes):
                if idx == i or idx == j:
                    continue
                new_routes.append(r)
            new_routes.append(rnew)
            routes = new_routes

        # if too many routes, try route elimination by insertion
        if len(routes) > K:
            ok = self.reduce_route_count(routes, day_visits, K)
            if not ok:
                return None
            routes = ok

        return routes

    def reduce_route_count(self, routes, day_visits, K):
        routes = list(routes)
        while len(routes) > K:
            improved = False
            order = sorted(range(len(routes)), key=lambda idx: (len(routes[idx]["seq"]), routes[idx]["demand"]))
            for ridx in order:
                source = routes[ridx]
                customers = list(source["seq"])
                other_idxs = [i for i in range(len(routes)) if i != ridx]

                working = {i: routes[i] for i in other_idxs}
                success = True

                for cid in customers:
                    best_target = None
                    best_new_route = None
                    best_delta = 1e18

                    for tidx in other_idxs:
                        r = working[tidx]
                        seq = r["seq"]
                        for pos in range(len(seq) + 1):
                            cand = seq[:pos] + [cid] + seq[pos:]
                            rc = self.build_route(cand, day_visits)
                            if rc is None:
                                continue
                            delta = rc["cost"] - r["cost"]
                            if delta < best_delta:
                                best_delta = delta
                                best_target = tidx
                                best_new_route = rc

                    if best_target is None:
                        success = False
                        break

                    working[best_target] = best_new_route

                if success:
                    new_routes = []
                    for i in range(len(routes)):
                        if i == ridx:
                            continue
                        new_routes.append(working[i] if i in working else routes[i])
                    routes = new_routes
                    improved = True
                    break

            if not improved:
                return None
        return routes

    def evaluate_selection(self, selection):
        daily = self.build_daily_visits(selection)
        all_day_routes = {}
        total_cost = 0.0

        for d in self.day_list:
            routes = self.solve_day_routes(d, daily[d])
            if routes is None:
                return None
            all_day_routes[d] = routes
            total_cost += sum(r["cost"] for r in routes)

        return total_cost, daily, all_day_routes

    def build_solution_json(self, objective, selection, daily_visits, all_day_routes):
        sol = {"objective_value": float(objective), "routes": {}, "schedules": {}}

        for d in self.day_list:
            day_key = str(d)
            sol["routes"][day_key] = []
            for r in all_day_routes[d]:
                parts = {}
                for cid in r["seq"]:
                    p = daily_visits[d][cid]
                    parts[str(cid)] = {
                        "visit_day": int(p["visit_day"]),
                        "days_covered": int(p["days_covered"]),
                        "demand_per_visit": float(p["demand_per_visit"]),
                    }
                sol["routes"][day_key].append(
                    {
                        "customers": [int(x) for x in r["seq"]],
                        "cost": float(r["cost"]),
                        "demand": float(r["demand"]),
                        "parts": parts,
                    }
                )

        for c in self.customers:
            cid = c["id"]
            sidx = selection[cid]
            sch = c["schedules"][sidx]
            sol["schedules"][str(cid)] = {
                "schedule_index": int(sidx),
                "days": [int(x) for x in sch.get("days", [])],
                "visit_frequency": int(sch.get("visit_frequency", len(sch.get("parts", [])))),
            }

        return sol

    def solve(self):
        best = None
        best_pack = None

        candidates = []
        mip_sel = self.initial_selection_mip()
        if mip_sel is not None:
            candidates.append(mip_sel)
        candidates.append(self.greedy_selection())
        for _ in range(3):
            candidates.append(self.random_selection())

        idx = 0
        while self.remaining() > 0.05:
            if idx < len(candidates):
                sel = candidates[idx]
                idx += 1
            else:
                if best_pack is None:
                    sel = self.random_selection()
                else:
                    strength = 0.05 + 0.20 * random.random()
                    sel = self.perturb_selection(best_pack["selection"], strength=strength)

            ev = self.evaluate_selection(sel)
            if ev is None:
                continue

            obj, daily_visits, all_day_routes = ev
            if (best is None) or (obj < best - 1e-9):
                best = obj
                best_pack = {
                    "objective": obj,
                    "selection": deepcopy(sel),
                    "daily_visits": daily_visits,
                    "all_day_routes": all_day_routes,
                }
                if self.logger:
                    self.logger.log(float(obj))

        if best_pack is None:
            # Fallback (possibly infeasible in real world, but keep schema-compliant output)
            sel = self.greedy_selection()
            daily_visits = self.build_daily_visits(sel)
            empty_routes = {d: [] for d in self.day_list}
            return self.build_solution_json(0.0, sel, daily_visits, empty_routes)

        return self.build_solution_json(
            best_pack["objective"],
            best_pack["selection"],
            best_pack["daily_visits"],
            best_pack["all_day_routes"],
        )


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

    solver = Solver(instance, time_limit=args.time_limit, logger=logger)
    solution = solver.solve()

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()
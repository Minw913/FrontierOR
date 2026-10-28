import argparse
import json
import math
import random
import time
import itertools

EPS = 1e-6
INF = 1e100

MAX_PATTERNS = 20
MAX_EVAL_OPTIONS = 60

try:
    from solution_logger import SolutionLogger
except Exception:
    SolutionLogger = None


class Route:
    __slots__ = ("custs", "demand", "duration", "cost")

    def __init__(self, custs=None, demand=0, duration=0.0, cost=0.0):
        self.custs = [] if custs is None else list(custs)
        self.demand = demand
        self.duration = duration
        self.cost = cost

    def copy(self):
        return Route(list(self.custs), self.demand, self.duration, self.cost)


class Instance:
    def __init__(self, data):
        self.n = int(data.get("n", len(data.get("customers", []))))
        self.d = int(data.get("d", len(data.get("depots", []))))
        self.t = int(data.get("t", 0))
        self.m = int(data.get("m", 0))
        self.capacity = int(data.get("vehicle_capacity", 0))
        self.duration_limit = float(data.get("route_duration_limit", 0))

        self.depots = data.get("depots", [])
        self.customers = data.get("customers", [])

        self.demand = []
        self.service = []
        self.frequency = []

        for i in range(self.n):
            c = self.customers[i] if i < len(self.customers) else {}
            self.demand.append(int(c.get("demand", 0)))
            self.service.append(float(c.get("service_duration", 0.0)))
            self.frequency.append(int(c.get("frequency", 0)))

        self.dist = self._build_distance_matrix(data)
        self._build_options()

        total = 0.0
        cnt = 0
        for dep in range(self.d):
            for c in range(self.n):
                node = self.d + c
                total += self.dist[dep][node] + self.dist[node][dep]
                cnt += 1
        self.avg_dist = total / (2.0 * cnt) if cnt else 1.0

    def _build_distance_matrix(self, data):
        total = self.d + self.n

        mat = None
        for key in (
            "distance_matrix",
            "distances",
            "travel_times",
            "travel_time_matrix",
            "travel_costs",
            "travel_cost_matrix",
            "cost_matrix",
            "time_matrix",
            "costs",
        ):
            if key in data:
                mat = data[key]
                break

        if mat is not None:
            try:
                if len(mat) == total and all(len(row) == total for row in mat):
                    return [[float(mat[i][j]) for j in range(total)] for i in range(total)]
            except Exception:
                pass

        coords = []
        for dep in self.depots[: self.d]:
            coords.append((float(dep.get("x", 0.0)), float(dep.get("y", 0.0))))
        while len(coords) < self.d:
            coords.append((0.0, 0.0))

        for cust in self.customers[: self.n]:
            coords.append((float(cust.get("x", 0.0)), float(cust.get("y", 0.0))))
        while len(coords) < total:
            coords.append((0.0, 0.0))
        coords = coords[:total]

        dist = [[0.0] * total for _ in range(total)]
        for i in range(total):
            xi, yi = coords[i]
            for j in range(total):
                if i == j:
                    dist[i][j] = 0.0
                else:
                    xj, yj = coords[j]
                    dist[i][j] = math.hypot(xi - xj, yi - yj)
        return dist

    def _build_options(self):
        self.patterns = []
        self.pattern_sets = []
        self.options = []

        for c in range(self.n):
            freq = self.frequency[c]
            cust = self.customers[c] if c < len(self.customers) else {}
            raw_patterns = cust.get("pattern_list", [])

            pats = []
            for pat in raw_patterns:
                days = []
                for p in pat:
                    try:
                        p = int(p)
                    except Exception:
                        continue
                    if 1 <= p <= self.t:
                        days.append(p)
                days = tuple(sorted(set(days)))

                if not days:
                    if freq <= 0:
                        pats.append(())
                else:
                    if freq > 0:
                        pats.append(days)

            if freq <= 0:
                pats = [()]
            else:
                exact = [p for p in pats if len(p) == freq]
                if exact:
                    pats = exact

                unique = []
                seen = set()
                for p in pats:
                    if p not in seen:
                        seen.add(p)
                        unique.append(p)
                pats = unique

                if not pats:
                    generated = []
                    if 0 < freq <= self.t:
                        for comb in itertools.combinations(range(1, self.t + 1), freq):
                            generated.append(tuple(comb))
                            if len(generated) >= MAX_PATTERNS:
                                break
                    if generated:
                        pats = generated
                    else:
                        pats = [tuple(range(1, self.t + 1))]

                if len(pats) > MAX_PATTERNS:
                    pats_sorted = sorted(pats)
                    keep = [pats_sorted[0], pats_sorted[-1]]
                    for k in range(MAX_PATTERNS):
                        idx = k * len(pats_sorted) // MAX_PATTERNS
                        keep.append(pats_sorted[idx])
                    unique = []
                    seen = set()
                    for p in keep:
                        if p not in seen:
                            seen.add(p)
                            unique.append(p)
                    pats = unique

            self.patterns.append(tuple(pats))
            self.pattern_sets.append(set(pats))

            opts = []
            for dep in range(self.d):
                for pat in pats:
                    opts.append((dep, pat))
            self.options.append(opts)


class Solution:
    def __init__(self, inst):
        self.assign_depot = [-1] * inst.n
        self.assign_pattern = [()] * inst.n
        self.routes = [[[] for _ in range(inst.t)] for __ in range(inst.d)]
        self.objective = 0.0

    def copy(self):
        new = Solution.__new__(Solution)
        new.assign_depot = list(self.assign_depot)
        new.assign_pattern = list(self.assign_pattern)
        new.objective = self.objective
        new.routes = []
        for depot_routes in self.routes:
            new_depot = []
            for day_routes in depot_routes:
                new_depot.append([r.copy() for r in day_routes])
            new.routes.append(new_depot)
        return new


def evaluate_seq(inst, depot, seq):
    demand = 0
    duration = 0.0
    cost = 0.0

    if not seq:
        return demand, duration, cost

    prev = depot
    off = inst.d
    dist = inst.dist

    for c in seq:
        node = off + c
        demand += inst.demand[c]
        leg = dist[prev][node]
        cost += leg
        duration += leg + inst.service[c]
        prev = node

    leg = dist[prev][depot]
    cost += leg
    duration += leg

    return demand, duration, cost


def make_route(inst, depot, seq):
    if not seq:
        return Route()
    demand, duration, cost = evaluate_seq(inst, depot, seq)
    if demand > inst.capacity + EPS:
        return None
    if duration > inst.duration_limit + EPS:
        return None
    return Route(seq, demand, duration, cost)


def insertion_delta(inst, depot, route, c, pos):
    seq = route.custs
    L = len(seq)
    cn = inst.d + c
    dist = inst.dist

    pred = depot if pos == 0 else inst.d + seq[pos - 1]
    succ = depot if pos == L else inst.d + seq[pos]

    add_cost = dist[pred][cn] + dist[cn][succ] - dist[pred][succ]
    add_dur = add_cost + inst.service[c]

    feasible = (
        route.demand + inst.demand[c] <= inst.capacity + EPS
        and route.duration + add_dur <= inst.duration_limit + EPS
    )
    return add_cost, add_dur, feasible


def removal_delta(inst, depot, route, idx):
    seq = route.custs
    c = seq[idx]
    cn = inst.d + c
    dist = inst.dist

    pred = depot if idx == 0 else inst.d + seq[idx - 1]
    succ = depot if idx == len(seq) - 1 else inst.d + seq[idx + 1]

    rem_cost = dist[pred][cn] + dist[cn][succ] - dist[pred][succ]
    rem_dur = rem_cost + inst.service[c]
    return rem_cost, rem_dur


def apply_insert(inst, depot, route, c, pos, add_cost, add_dur):
    route.custs.insert(pos, c)
    route.demand += inst.demand[c]
    route.duration += add_dur
    route.cost += add_cost


def apply_remove(inst, depot, route, idx, rem_cost, rem_dur):
    c = route.custs.pop(idx)
    route.demand -= inst.demand[c]
    route.duration -= rem_dur
    route.cost -= rem_cost
    return c


def best_insert_in_route(inst, depot, route, c):
    if c in route.custs:
        return None
    if route.demand + inst.demand[c] > inst.capacity + EPS:
        return None

    best = None
    for pos in range(len(route.custs) + 1):
        add_cost, add_dur, feasible = insertion_delta(inst, depot, route, c, pos)
        if feasible:
            if best is None or add_cost < best[0] - EPS:
                best = (add_cost, add_dur, pos)
    return best


def best_insert_in_routes(inst, routes, depot, c, allow_new=True):
    best = None

    for ri, route in enumerate(routes):
        res = best_insert_in_route(inst, depot, route, c)
        if res is None:
            continue
        add_cost, add_dur, pos = res
        score = add_cost
        if best is None or score < best[0] - EPS:
            best = (score, add_cost, add_dur, ri, pos, False)

    if allow_new and inst.m > 0 and len(routes) < inst.m:
        empty = Route()
        add_cost, add_dur, feasible = insertion_delta(inst, depot, empty, c, 0)
        if feasible:
            score = add_cost + 1e-9
            if best is None or score < best[0] - EPS:
                best = (score, add_cost, add_dur, -1, 0, True)

    return best


def apply_res_action(inst, routes, depot, c, action):
    if action is None:
        return False

    _, add_cost, add_dur, ri, pos, is_new = action

    if is_new:
        if inst.m <= 0 or len(routes) >= inst.m:
            return False
        r = Route()
        ac, ad, feasible = insertion_delta(inst, depot, r, c, 0)
        if not feasible:
            return False
        apply_insert(inst, depot, r, c, 0, ac, ad)
        routes.append(r)
        return True

    if ri >= len(routes):
        return False
    route = routes[ri]
    if pos > len(route.custs):
        return False

    ac, ad, feasible = insertion_delta(inst, depot, route, c, pos)
    if not feasible:
        return False

    apply_insert(inst, depot, route, c, pos, ac, ad)
    return True


def estimate_option(inst, c, depot, pattern):
    if not pattern:
        return 0.0
    node = inst.d + c
    roundtrip = inst.dist[depot][node] + inst.dist[node][depot]
    return len(pattern) * roundtrip + len(pattern) * inst.service[c]


def evaluate_option(inst, sol, c, depot, pattern, rng=None, noise=0.0):
    pattern = tuple(pattern)

    if not pattern:
        score = 0.0
        if noise > 0.0 and rng is not None:
            score += rng.uniform(-noise, noise) * max(1.0, inst.avg_dist)
        return (score, 0.0, depot, pattern, [])

    actions = []
    total = 0.0
    new_routes = 0

    for p in pattern:
        day_idx = int(p) - 1
        if day_idx < 0 or day_idx >= inst.t:
            return None

        routes = sol.routes[depot][day_idx]
        res = best_insert_in_routes(inst, routes, depot, c, allow_new=True)
        if res is None:
            return None

        score, add_cost, add_dur, ri, pos, is_new = res
        actions.append((score, add_cost, add_dur, ri, pos, is_new))
        total += add_cost
        if is_new:
            new_routes += 1

    score = total + 1e-7 * new_routes * max(1.0, inst.avg_dist)
    if noise > 0.0 and rng is not None:
        score += rng.uniform(-noise, noise) * max(1.0, abs(total))

    return (score, total, depot, pattern, actions)


def evaluate_all_options(inst, sol, c, rng=None, noise=0.0):
    opts = inst.options[c]
    if not opts:
        return []

    if len(opts) > MAX_EVAL_OPTIONS:
        scored = sorted(opts, key=lambda o: estimate_option(inst, c, o[0], o[1]))
        keep = scored[: MAX_EVAL_OPTIONS - 10]
        rest = scored[MAX_EVAL_OPTIONS - 10:]
        if rng is not None and rest:
            keep.extend(rng.sample(rest, min(10, len(rest))))
        else:
            keep.extend(rest[:10])

        unique = []
        seen = set()
        for dep, pat in keep:
            key = (dep, tuple(pat))
            if key not in seen:
                seen.add(key)
                unique.append((dep, pat))
        opts = unique

    cands = []
    for depot, pattern in opts:
        opt = evaluate_option(inst, sol, c, depot, pattern, rng=rng, noise=noise)
        if opt is not None:
            cands.append(opt)

    cands.sort(key=lambda x: x[0])
    return cands


def remove_customer_from_day(inst, sol, depot, day_idx, c):
    routes = sol.routes[depot][day_idx]

    for route in routes:
        while c in route.custs:
            idx = route.custs.index(c)
            rem_cost, rem_dur = removal_delta(inst, depot, route, idx)
            apply_remove(inst, depot, route, idx, rem_cost, rem_dur)

    routes[:] = [r for r in routes if r.custs]


def remove_customer(inst, sol, c):
    for dep in range(inst.d):
        for day_idx in range(inst.t):
            remove_customer_from_day(inst, sol, dep, day_idx, c)
    sol.assign_depot[c] = -1
    sol.assign_pattern[c] = ()


def apply_option(inst, sol, c, option):
    _, _, depot, pattern, actions = option
    inserted_days = []

    for action in actions:
        day_idx = action[3] if len(action) == 6 else None
        # action format: (score, add_cost, add_dur, ri, pos, is_new)
        day_idx = action[3]
        # The above is wrong because day_idx is not stored in action.
        # We therefore store day_idx separately in evaluate_option.
        # To keep compatibility, actions are rebuilt below.
        pass

    # Rebuild with explicit day index to avoid ambiguity.
    inserted_days = []
    for score, add_cost, add_dur, ri, pos, is_new in actions:
        # This loop is replaced below by a safer construction from evaluate_option.
        pass

    # Safe application using actions generated by evaluate_option_safe.
    # To avoid maintaining two formats, evaluate_option_safe is used instead.
    return False


# Replace evaluate_option/apply_option with safer versions that store day_idx explicitly.
def evaluate_option_safe(inst, sol, c, depot, pattern, rng=None, noise=0.0):
    pattern = tuple(pattern)

    if not pattern:
        score = 0.0
        if noise > 0.0 and rng is not None:
            score += rng.uniform(-noise, noise) * max(1.0, inst.avg_dist)
        return (score, 0.0, depot, pattern, [])

    actions = []
    total = 0.0
    new_routes = 0

    for p in pattern:
        day_idx = int(p) - 1
        if day_idx < 0 or day_idx >= inst.t:
            return None

        routes = sol.routes[depot][day_idx]
        res = best_insert_in_routes(inst, routes, depot, c, allow_new=True)
        if res is None:
            return None

        score_action, add_cost, add_dur, ri, pos, is_new = res
        actions.append((day_idx, add_cost, add_dur, ri, pos, is_new))
        total += add_cost
        if is_new:
            new_routes += 1

    score = total + 1e-7 * new_routes * max(1.0, inst.avg_dist)
    if noise > 0.0 and rng is not None:
        score += rng.uniform(-noise, noise) * max(1.0, abs(total))

    return (score, total, depot, pattern, actions)


def evaluate_all_options_safe(inst, sol, c, rng=None, noise=0.0):
    opts = inst.options[c]
    if not opts:
        return []

    if len(opts) > MAX_EVAL_OPTIONS:
        scored = sorted(opts, key=lambda o: estimate_option(inst, c, o[0], o[1]))
        keep = scored[: MAX_EVAL_OPTIONS - 10]
        rest = scored[MAX_EVAL_OPTIONS - 10:]
        if rng is not None and rest:
            keep.extend(rng.sample(rest, min(10, len(rest))))
        else:
            keep.extend(rest[:10])

        unique = []
        seen = set()
        for dep, pat in keep:
            key = (dep, tuple(pat))
            if key not in seen:
                seen.add(key)
                unique.append((dep, pat))
        opts = unique

    cands = []
    for depot, pattern in opts:
        opt = evaluate_option_safe(inst, sol, c, depot, pattern, rng=rng, noise=noise)
        if opt is not None:
            cands.append(opt)

    cands.sort(key=lambda x: x[0])
    return cands


def apply_option_safe(inst, sol, c, option):
    _, _, depot, pattern, actions = option
    inserted_days = []

    for day_idx, add_cost, add_dur, ri, pos, is_new in actions:
        routes = sol.routes[depot][day_idx]
        action_tuple = (None, add_cost, add_dur, ri, pos, is_new)

        ok = apply_res_action(inst, routes, depot, c, action_tuple)
        if not ok:
            res = best_insert_in_routes(inst, routes, depot, c, allow_new=True)
            if res is not None:
                ok = apply_res_action(inst, routes, depot, c, res)

        if not ok:
            for d_idx in inserted_days:
                remove_customer_from_day(inst, sol, depot, d_idx, c)
            return False

        inserted_days.append(day_idx)

    sol.assign_depot[c] = depot
    sol.assign_pattern[c] = tuple(pattern)
    return True


def recompute_solution(inst, sol):
    obj = 0.0

    for dep in range(inst.d):
        for day_idx in range(inst.t):
            routes = sol.routes[dep][day_idx]
            new_routes = []
            for r in routes:
                if not r.custs:
                    continue
                demand, duration, cost = evaluate_seq(inst, dep, r.custs)
                r.demand = demand
                r.duration = duration
                r.cost = cost
                new_routes.append(r)
                obj += cost
            sol.routes[dep][day_idx] = new_routes

    sol.objective = obj


def optimize_route_seq(inst, depot, seq):
    base = make_route(inst, depot, seq)
    if base is None:
        return None

    if len(seq) <= 1 or len(seq) > 50:
        return base

    best_seq = list(seq)
    best_route = base

    passes = 1 if len(best_seq) > 30 else 2

    for _ in range(passes):
        improved = False
        current = make_route(inst, depot, best_seq)
        if current is None:
            break

        L = len(best_seq)

        # Single-customer relocation.
        for i in range(L):
            c = best_seq[i]
            rem_cost, rem_dur = removal_delta(inst, depot, current, i)
            seq2 = best_seq[:i] + best_seq[i+1:]

            base_cost = current.cost - rem_cost
            base_dur = current.duration - rem_dur

            best_pos = None
            best_add = None
            best_new_cost = current.cost

            for pos in range(len(seq2) + 1):
                pred = depot if pos == 0 else inst.d + seq2[pos - 1]
                succ = depot if pos == len(seq2) else inst.d + seq2[pos]
                cn = inst.d + c

                add_cost = inst.dist[pred][cn] + inst.dist[cn][succ] - inst.dist[pred][succ]
                add_dur = add_cost + inst.service[c]

                if base_dur + add_dur > inst.duration_limit + EPS:
                    continue

                new_cost = base_cost + add_cost
                if new_cost < best_new_cost - EPS:
                    best_new_cost = new_cost
                    best_pos = pos
                    best_add = (add_cost, add_dur)

            if best_pos is not None:
                cand_seq = seq2[:best_pos] + [c] + seq2[best_pos:]
                cand_route = make_route(inst, depot, cand_seq)
                if cand_route is not None and cand_route.cost < best_route.cost - EPS:
                    best_seq = cand_seq
                    best_route = cand_route
                    improved = True
                    break

        # Limited 2-opt for small routes.
        if not improved and len(best_seq) <= 20:
            L = len(best_seq)
            for i in range(L - 1):
                for j in range(i + 1, L):
                    cand_seq = best_seq[:i] + list(reversed(best_seq[i:j + 1])) + best_seq[j + 1:]
                    cand_route = make_route(inst, depot, cand_seq)
                    if cand_route is not None and cand_route.cost < best_route.cost - EPS:
                        best_seq = cand_seq
                        best_route = cand_route
                        improved = True
                        break
                if improved:
                    break

        if not improved:
            break

    return best_route


def best_merge_two(inst, depot, r1, r2):
    if set(r1.custs) & set(r2.custs):
        return None

    base = r1.cost + r2.cost
    best = None

    seqs1 = [r1.custs, list(reversed(r1.custs))]
    seqs2 = [r2.custs, list(reversed(r2.custs))]
    seen = set()

    for s1 in seqs1:
        t1 = tuple(s1)
        for s2 in seqs2:
            t2 = tuple(s2)
            for cand in (t1 + t2, t2 + t1):
                if cand in seen:
                    continue
                seen.add(cand)

                nr = make_route(inst, depot, list(cand))
                if nr is None:
                    continue

                saving = base - nr.cost
                if best is None or saving > best[0]:
                    best = (saving, nr)

    return best


def merge_positive_routes(inst, depot, routes, deadline):
    improved = True
    while improved and len(routes) > 1 and time.time() < deadline:
        improved = False
        best = None

        for i in range(len(routes)):
            for j in range(i + 1, len(routes)):
                res = best_merge_two(inst, depot, routes[i], routes[j])
                if res is None:
                    continue
                saving, nr = res
                if saving > EPS and (best is None or saving > best[0]):
                    best = (saving, i, j, nr)

        if best is not None:
            _, i, j, nr = best
            routes.pop(j)
            routes.pop(i)
            routes.append(nr)
            improved = True


def greedy_insert_order(inst, depot, order):
    routes = []

    for c in order:
        res = best_insert_in_routes(inst, routes, depot, c, allow_new=True)
        if res is None:
            return None
        if not apply_res_action(inst, routes, depot, c, res):
            return None

    return routes


def reduce_fleet(inst, depot, routes):
    routes = [r.copy() for r in routes]

    while len(routes) > inst.m:
        best = None

        for i in range(len(routes)):
            for j in range(i + 1, len(routes)):
                res = best_merge_two(inst, depot, routes[i], routes[j])
                if res is None:
                    continue
                saving, nr = res
                if best is None or saving > best[0]:
                    best = (saving, i, j, nr)

        if best is None:
            return routes

        _, i, j, nr = best
        routes.pop(j)
        routes.pop(i)
        routes.append(nr)

    return routes


def savings_builder(inst, depot, custs):
    if inst.m <= 0 and custs:
        return None
    if not custs:
        return []

    routes = []
    for c in custs:
        r = make_route(inst, depot, [c])
        if r is None:
            return None
        routes.append(r)

    while len(routes) > inst.m:
        best = None
        for i in range(len(routes)):
            for j in range(i + 1, len(routes)):
                res = best_merge_two(inst, depot, routes[i], routes[j])
                if res is None:
                    continue
                saving, nr = res
                if best is None or saving > best[0]:
                    best = (saving, i, j, nr)

        if best is None:
            return None

        _, i, j, nr = best
        routes.pop(j)
        routes.pop(i)
        routes.append(nr)

    merge_positive_routes(inst, depot, routes, time.time() + 3600)
    return routes


def build_day_routes(inst, depot, custs, rng=None):
    if not custs:
        return []
    if inst.m <= 0:
        return None

    # Single-customer feasibility.
    for c in custs:
        if inst.demand[c] > inst.capacity + EPS:
            return None
        node = inst.d + c
        dur = inst.dist[depot][node] + inst.dist[node][depot] + inst.service[c]
        if dur > inst.duration_limit + EPS:
            return None

    best = None
    best_cost = INF

    orders = []
    orders.append(sorted(custs, key=lambda x: -inst.demand[x]))
    orders.append(sorted(custs, key=lambda x: -(inst.dist[depot][inst.d + x] + inst.dist[inst.d + x][depot])))
    orders.append(sorted(custs, key=lambda x: -(inst.service[x] + inst.demand[x])))

    if rng is not None and len(custs) > 1:
        sh = custs[:]
        rng.shuffle(sh)
        orders.append(sh)

        sh2 = custs[:]
        rng.shuffle(sh2)
        orders.append(sh2)

    seen = set()
    for order in orders:
        key = tuple(order)
        if key in seen:
            continue
        seen.add(key)

        routes = greedy_insert_order(inst, depot, order)
        if routes is None:
            continue

        if len(custs) <= 35:
            improved_routes = []
            for r in routes:
                nr = optimize_route_seq(inst, depot, r.custs)
                improved_routes.append(nr if nr is not None else r)
            routes = improved_routes

        cost = sum(r.cost for r in routes)
        if best is None or cost < best_cost - EPS:
            best = routes
            best_cost = cost

    if len(custs) <= 50:
        routes = savings_builder(inst, depot, custs)
        if routes is not None:
            if len(custs) <= 35:
                improved_routes = []
                for r in routes:
                    nr = optimize_route_seq(inst, depot, r.custs)
                    improved_routes.append(nr if nr is not None else r)
                routes = improved_routes

            cost = sum(r.cost for r in routes)
            if best is None or cost < best_cost - EPS:
                best = routes
                best_cost = cost

    if best is None and rng is not None:
        for _ in range(3):
            order = custs[:]
            rng.shuffle(order)
            routes = greedy_insert_order(inst, depot, order)
            if routes is not None:
                cost = sum(r.cost for r in routes)
                if best is None or cost < best_cost - EPS:
                    best = routes
                    best_cost = cost

    if best is not None and len(best) > inst.m:
        best = reduce_fleet(inst, depot, best)

    if best is None or len(best) > inst.m:
        return None

    return best


def build_routes_from_assignment(inst, sol, rng=None):
    custs_by = [[[] for _ in range(inst.t)] for __ in range(inst.d)]

    for c in range(inst.n):
        dep = sol.assign_depot[c]
        pat = sol.assign_pattern[c]
        if dep < 0 or dep >= inst.d or not pat:
            continue

        for p in pat:
            day_idx = int(p) - 1
            if 0 <= day_idx < inst.t:
                custs_by[dep][day_idx].append(c)

    new_routes = [[[] for _ in range(inst.t)] for __ in range(inst.d)]

    for dep in range(inst.d):
        for day_idx in range(inst.t):
            routes = build_day_routes(inst, dep, custs_by[dep][day_idx], rng)
            if routes is None:
                return False
            new_routes[dep][day_idx] = routes

    sol.routes = new_routes
    recompute_solution(inst, sol)
    return True


def optimize_day_routes(inst, sol, depot, day_idx, deadline):
    routes = sol.routes[depot][day_idx]
    if not routes:
        return

    # Intra-route improvement.
    for i, r in enumerate(routes):
        if time.time() >= deadline:
            break
        nr = optimize_route_seq(inst, depot, r.custs)
        if nr is not None and nr.cost < r.cost - EPS:
            routes[i] = nr

    # Positive merges.
    merge_positive_routes(inst, depot, routes, deadline)

    # Inter-route relocation.
    max_rel_iters = 3
    it = 0
    while it < max_rel_iters and time.time() < deadline:
        it += 1
        moved_global = False

        for i, route_i in enumerate(routes):
            if time.time() >= deadline:
                break

            moved = False
            for k, c in enumerate(route_i.custs):
                rem_cost, rem_dur = removal_delta(inst, depot, route_i, k)

                best_gain = EPS
                best_route = None
                best_pos = None
                best_add = None

                for route_j in routes:
                    if route_j is route_i:
                        continue
                    if c in route_j.custs:
                        continue

                    ins = best_insert_in_route(inst, depot, route_j, c)
                    if ins is None:
                        continue

                    add_cost, add_dur, pos = ins
                    gain = rem_cost - add_cost

                    if gain > best_gain:
                        best_gain = gain
                        best_route = route_j
                        best_pos = pos
                        best_add = (add_cost, add_dur)

                if best_route is not None:
                    apply_remove(inst, depot, route_i, k, rem_cost, rem_dur)
                    apply_insert(inst, depot, best_route, c, best_pos, best_add[0], best_add[1])

                    if not route_i.custs:
                        try:
                            routes.remove(route_i)
                        except ValueError:
                            pass

                    moved = True
                    moved_global = True
                    break

            if moved:
                routes[:] = [r for r in routes if r.custs]
                break

        if not moved_global:
            break

    # Positive merges again after relocations.
    merge_positive_routes(inst, depot, routes, deadline)

    # Final intra-route improvement.
    for i, r in enumerate(routes):
        if time.time() >= deadline:
            break
        nr = optimize_route_seq(inst, depot, r.custs)
        if nr is not None and nr.cost < r.cost - EPS:
            routes[i] = nr

    sol.routes[depot][day_idx] = routes


def optimize_all(inst, sol, deadline):
    for dep in range(inst.d):
        for day_idx in range(inst.t):
            if time.time() >= deadline:
                break
            optimize_day_routes(inst, sol, dep, day_idx, deadline)
    recompute_solution(inst, sol)


def quick_key(inst, sol):
    fleet = 0
    route_infeas = 0
    structural = 0
    unserved = 0
    pattern_mismatch = 0

    visit_days = [set() for _ in range(inst.n)]

    for dep, depot_routes in enumerate(sol.routes):
        for day_idx, routes in enumerate(depot_routes):
            if len(routes) > inst.m:
                fleet += len(routes) - inst.m

            for r in routes:
                if not r.custs:
                    continue

                if r.demand > inst.capacity + EPS or r.duration > inst.duration_limit + EPS:
                    route_infeas += 1

                seen = set()
                for c in r.custs:
                    if c in seen:
                        structural += 1
                    seen.add(c)

                    if 0 <= c < inst.n:
                        if dep < inst.d and sol.assign_depot[c] != dep:
                            structural += 1

                        day = day_idx + 1
                        if day in visit_days[c]:
                            structural += 1
                        else:
                            visit_days[c].add(day)
                    else:
                        route_infeas += 1

    for c in range(inst.n):
        dep = sol.assign_depot[c]
        pat = tuple(sol.assign_pattern[c] or ())

        if dep < 0 or dep >= inst.d:
            structural += 1
        elif pat not in inst.pattern_sets[c]:
            structural += 1

        freq = inst.frequency[c]
        if freq <= 0:
            if visit_days[c]:
                unserved += len(visit_days[c])
            if pat != ():
                pattern_mismatch += 1
        else:
            if len(pat) != freq:
                pattern_mismatch += 1
            unserved += abs(freq - len(visit_days[c]))
            if set(pat) != visit_days[c]:
                pattern_mismatch += 1

    return (fleet, route_infeas, structural, unserved, pattern_mismatch, float(sol.objective))


def is_feasible(inst, sol):
    if inst.d == 0:
        return all(inst.frequency[c] <= 0 for c in range(inst.n))

    key = quick_key(inst, sol)
    if key[0] != 0 or key[1] != 0 or key[2] != 0 or key[3] != 0 or key[4] != 0:
        return False

    visit_days = [set() for _ in range(inst.n)]

    for dep, depot_routes in enumerate(sol.routes):
        for day_idx, routes in enumerate(depot_routes):
            if len(routes) > inst.m:
                return False

            for r in routes:
                if not r.custs:
                    continue

                demand, duration, _ = evaluate_seq(inst, dep, r.custs)
                if demand > inst.capacity + EPS or duration > inst.duration_limit + EPS:
                    return False

                seen = set()
                for c in r.custs:
                    if c in seen:
                        return False
                    seen.add(c)

                    if not (0 <= c < inst.n):
                        return False
                    if sol.assign_depot[c] != dep:
                        return False

                    day = day_idx + 1
                    if day in visit_days[c]:
                        return False
                    visit_days[c].add(day)

    for c in range(inst.n):
        pat = tuple(sol.assign_pattern[c] or ())
        if pat not in inst.pattern_sets[c]:
            return False

        freq = inst.frequency[c]
        if freq <= 0:
            if pat != () or visit_days[c]:
                return False
        else:
            if len(pat) != freq:
                return False
            if len(visit_days[c]) != freq:
                return False
            if set(pat) != visit_days[c]:
                return False

    return True


def solution_to_dict(inst, sol):
    routes_out = []

    for dep, depot_routes in enumerate(sol.routes):
        for day_idx, routes in enumerate(depot_routes):
            vehicle = 0
            for r in routes:
                if not r.custs:
                    continue
                vehicle += 1

                if inst.m > 0:
                    vehicle_id = vehicle if vehicle <= inst.m else ((vehicle - 1) % inst.m) + 1
                else:
                    vehicle_id = 1

                routes_out.append({
                    "depot": dep,
                    "period": day_idx + 1,
                    "vehicle": vehicle_id,
                    "customers": list(r.custs),
                })

    assignments = {}
    for c in range(inst.n):
        dep = sol.assign_depot[c]
        if dep < 0 or dep >= inst.d:
            dep = 0 if inst.d > 0 else 0

        pat = list(sol.assign_pattern[c] or [])
        assignments[str(c)] = {
            "depot": dep,
            "pattern": pat,
        }

    return {
        "objective_value": float(sol.objective),
        "routes": routes_out,
        "assignments": assignments,
    }


class Reporter:
    def __init__(self, inst, logger):
        self.inst = inst
        self.logger = logger
        self.best_obj = None

    def report(self, sol):
        if not is_feasible(self.inst, sol):
            return False

        obj = float(sol.objective)
        if self.best_obj is None or obj < self.best_obj - EPS:
            self.best_obj = obj
            if self.logger is not None:
                try:
                    self.logger.log_solution(obj, solution_to_dict(self.inst, sol))
                except Exception:
                    pass
            return True

        return False


def deterministic_assignment(inst, variant, rng):
    sol = Solution(inst)

    for c in range(inst.n):
        opts = inst.options[c]
        if not opts:
            sol.assign_depot[c] = 0 if inst.d > 0 else -1
            sol.assign_pattern[c] = ()
            continue

        if variant == 0:
            dep, pat = min(opts, key=lambda o: estimate_option(inst, c, o[0], o[1]))
        elif variant == 1:
            dep, pat = min(opts, key=lambda o: (len(o[1]), estimate_option(inst, c, o[0], o[1])))
        elif variant == 2:
            scored = sorted(opts, key=lambda o: estimate_option(inst, c, o[0], o[1]))
            if rng is not None:
                dep, pat = rng.choice(scored[: min(3, len(scored))])
            else:
                dep, pat = scored[0]
        elif variant == 3:
            scored = sorted(opts, key=lambda o: (len(o[1]), estimate_option(inst, c, o[0], o[1])))
            if rng is not None and len(scored) > 1:
                dep, pat = rng.choice(scored[: min(5, len(scored))])
            else:
                dep, pat = scored[0]
        else:
            dep, pat = opts[0]

        sol.assign_depot[c] = dep
        sol.assign_pattern[c] = tuple(pat)

    return sol


def incremental_greedy(inst, rng, noise=0.0):
    sol = Solution(inst)

    for c in range(inst.n):
        if inst.frequency[c] <= 0:
            if inst.options[c]:
                dep, pat = inst.options[c][0]
            else:
                dep = 0 if inst.d > 0 else -1
                pat = ()
            sol.assign_depot[c] = dep
            sol.assign_pattern[c] = tuple(pat)

    customers = [c for c in range(inst.n) if inst.frequency[c] > 0]

    def sort_key(c):
        min_est = INF
        for dep, pat in inst.options[c]:
            val = estimate_option(inst, c, dep, pat)
            if val < min_est:
                min_est = val
        if min_est == INF:
            min_est = 0.0
        return (
            inst.frequency[c],
            inst.demand[c],
            inst.service[c],
            -len(inst.options[c]),
            min_est,
        )

    customers.sort(key=sort_key, reverse=True)

    for c in customers:
        if not inst.options[c]:
            sol.assign_depot[c] = 0 if inst.d > 0 else -1
            sol.assign_pattern[c] = ()
            continue

        cands = evaluate_all_options_safe(inst, sol, c, rng=rng, noise=noise)
        applied = False

        for cand in cands[:12]:
            if apply_option_safe(inst, sol, c, cand):
                applied = True
                break

        if not applied:
            dep, pat = min(inst.options[c], key=lambda o: estimate_option(inst, c, o[0], o[1]))
            sol.assign_depot[c] = dep
            sol.assign_pattern[c] = tuple(pat)

    recompute_solution(inst, sol)
    return sol


def generate_initial(inst, rng, deadline, reporter):
    best = None
    best_key = None

    def consider(sol, opt_budget=0.0):
        nonlocal best, best_key

        recompute_solution(inst, sol)

        if opt_budget > 0.0 and time.time() < deadline:
            optimize_all(inst, sol, min(deadline, time.time() + opt_budget))
            recompute_solution(inst, sol)

        key = quick_key(inst, sol)
        if best_key is None or key < best_key:
            best = sol.copy()
            best_key = key

        if reporter is not None:
            reporter.report(sol)

    # Incremental constructions.
    for noise in (0.0, 0.04, 0.10):
        if time.time() >= deadline:
            break

        sol = incremental_greedy(inst, rng, noise=noise)
        consider(sol, 0.5)

        sol2 = sol.copy()
        if time.time() < deadline and build_routes_from_assignment(inst, sol2, rng):
            consider(sol2, 0.5)

    # Assignment-first constructions.
    for variant in range(4):
        if time.time() >= deadline:
            break

        sol = deterministic_assignment(inst, variant, rng)
        if build_routes_from_assignment(inst, sol, rng):
            consider(sol, 0.5)

    # Fallback.
    if best is None:
        sol = Solution(inst)
        for c in range(inst.n):
            if inst.options[c]:
                dep, pat = min(inst.options[c], key=lambda o: estimate_option(inst, c, o[0], o[1]))
            else:
                dep = 0 if inst.d > 0 else -1
                pat = ()
            sol.assign_depot[c] = dep
            sol.assign_pattern[c] = tuple(pat)

        build_routes_from_assignment(inst, sol, rng)
        consider(sol, 0.25)

    return best


def improve_by_reinsertion(inst, sol, rng, deadline, noise=0.0, max_passes=1):
    recompute_solution(inst, sol)

    best = sol.copy()
    best_key = quick_key(inst, best)

    customers = [c for c in range(inst.n) if inst.frequency[c] > 0]
    if not customers:
        return best

    passes = 0
    while passes < max_passes and time.time() < deadline:
        passes += 1
        order = customers[:]
        rng.shuffle(order)

        improved = False

        for c in order:
            if time.time() >= deadline:
                break

            backup = sol.copy()
            old_key = quick_key(inst, backup)

            remove_customer(inst, sol, c)
            recompute_solution(inst, sol)

            cands = evaluate_all_options_safe(inst, sol, c, rng=rng, noise=noise)
            inserted = False

            for cand in cands[:8]:
                if apply_option_safe(inst, sol, c, cand):
                    recompute_solution(inst, sol)
                    inserted = True
                    break
                recompute_solution(inst, sol)

            if inserted:
                new_key = quick_key(inst, sol)
                if new_key < old_key:
                    if new_key < best_key:
                        best = sol.copy()
                        best_key = new_key
                    improved = True
                    continue

            sol = backup

        if not improved:
            break

        cur_key = quick_key(inst, sol)
        if cur_key < best_key:
            best = sol.copy()
            best_key = cur_key

    return best


def perturb_solution(inst, sol, rng, deadline, fraction=0.15, noise=0.08):
    recompute_solution(inst, sol)

    customers = [c for c in range(inst.n) if inst.frequency[c] > 0]
    if not customers:
        return sol

    k = max(1, min(len(customers), int(len(customers) * fraction) + 1))
    removed = rng.sample(customers, k)

    for c in removed:
        remove_customer(inst, sol, c)

    recompute_solution(inst, sol)

    rng.shuffle(removed)
    for c in removed:
        if time.time() >= deadline:
            break

        cands = evaluate_all_options_safe(inst, sol, c, rng=rng, noise=noise)
        applied = False

        for cand in cands[:8]:
            if apply_option_safe(inst, sol, c, cand):
                applied = True
                break

        if not applied:
            if inst.options[c]:
                dep, pat = min(inst.options[c], key=lambda o: estimate_option(inst, c, o[0], o[1]))
                sol.assign_depot[c] = dep
                sol.assign_pattern[c] = tuple(pat)

    recompute_solution(inst, sol)
    return sol


def search(inst, best, rng, deadline, reporter):
    current_best = best.copy()
    best_key = quick_key(inst, current_best)

    while time.time() < deadline - 1.0:
        segment = min(5.0, max(1.0, (deadline - time.time()) * 0.10))

        cand = current_best.copy()
        cand = improve_by_reinsertion(
            inst,
            cand,
            rng,
            min(deadline, time.time() + segment),
            noise=0.0,
            max_passes=1,
        )

        optimize_all(inst, cand, min(deadline, time.time() + 0.5))
        recompute_solution(inst, cand)

        key = quick_key(inst, cand)
        if key < best_key:
            current_best = cand
            best_key = key
            if reporter is not None:
                reporter.report(current_best)

        if time.time() < deadline - 3.0:
            pert = current_best.copy()
            pert = perturb_solution(
                inst,
                pert,
                rng,
                min(deadline, time.time() + segment),
                fraction=0.15,
                noise=0.08,
            )

            pert = improve_by_reinsertion(
                inst,
                pert,
                rng,
                min(deadline, time.time() + segment),
                noise=0.02,
                max_passes=1,
            )

            optimize_all(inst, pert, min(deadline, time.time() + 0.5))
            recompute_solution(inst, pert)

            keyp = quick_key(inst, pert)
            if keyp < best_key:
                current_best = pert
                best_key = keyp
                if reporter is not None:
                    reporter.report(current_best)
        else:
            break

    return current_best


def finalize_solution(inst, sol, rng=None):
    recompute_solution(inst, sol)

    # Remove empty routes and try to satisfy fleet limits.
    for dep in range(inst.d):
        for day_idx in range(inst.t):
            routes = [r for r in sol.routes[dep][day_idx] if r.custs]
            if len(routes) > inst.m:
                routes = reduce_fleet(inst, dep, routes)
            sol.routes[dep][day_idx] = routes

    recompute_solution(inst, sol)

    # Ensure assignment fields are present and syntactically valid.
    for c in range(inst.n):
        if sol.assign_depot[c] < 0 or sol.assign_depot[c] >= inst.d:
            if inst.options[c]:
                sol.assign_depot[c] = inst.options[c][0][0]
            else:
                sol.assign_depot[c] = 0

        pat = tuple(sol.assign_pattern[c] or ())
        if pat not in inst.pattern_sets[c]:
            if inst.patterns[c]:
                sol.assign_pattern[c] = inst.patterns[c][0]
            else:
                sol.assign_pattern[c] = ()

    recompute_solution(inst, sol)
    return sol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, default=60)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    start = time.time()

    logger = None
    if args.log_path and SolutionLogger is not None:
        try:
            logger = SolutionLogger(args.log_path, sense="minimize")
        except Exception:
            logger = None

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    inst = Instance(data)
    rng = random.Random(0)

    deadline = start + max(1, int(args.time_limit)) - 0.5
    reporter = Reporter(inst, logger)

    best = generate_initial(inst, rng, deadline, reporter)
    if best is None:
        best = Solution(inst)

    recompute_solution(inst, best)

    if time.time() < deadline:
        best = search(inst, best, rng, deadline, reporter)

    best = finalize_solution(inst, best, rng)
    recompute_solution(inst, best)

    if reporter is not None:
        reporter.report(best)

    sol_dict = solution_to_dict(inst, best)

    with open(args.solution_path, "w") as f:
        json.dump(sol_dict, f, indent=2)


if __name__ == "__main__":
    main()
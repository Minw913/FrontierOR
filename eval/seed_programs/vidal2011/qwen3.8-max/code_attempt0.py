import argparse
import json
import math
import random
import time
import itertools

EPS = 1e-6
INF = 1e100

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
        return Route(self.custs, self.demand, self.duration, self.cost)


class Problem:
    def __init__(self, path):
        with open(path, "r") as f:
            data = json.load(f)

        self.n = int(data.get("n", 0))
        self.d = int(data.get("d", 0))
        self.t = int(data.get("t", 0))
        self.m = int(data.get("m", 0))
        self.capacity = int(data.get("vehicle_capacity", 0))
        self.duration_limit = float(data.get("route_duration_limit", 0))

        self.depots = data.get("depots", [])
        self.customers = data.get("customers", [])

        self.demand = [0] * self.n
        self.service = [0.0] * self.n
        self.frequency = [0] * self.n

        for i, c in enumerate(self.customers):
            self.demand[i] = int(c.get("demand", 0))
            self.service[i] = float(c.get("service_duration", 0.0))
            self.frequency[i] = int(c.get("frequency", 0))

        self.dist = self._build_distance_matrix(data)
        self.options = self._build_options()

        total = 0.0
        cnt = 0
        for c in range(self.n):
            node = self.d + c
            for k in range(self.d):
                total += self.dist[k][node]
                cnt += 1
        self.avg_dist = total / cnt if cnt else 1.0

    def _build_distance_matrix(self, data):
        total_nodes = self.d + self.n

        mat = None
        for key in ("distance_matrix", "travel_times", "travel_costs", "cost_matrix"):
            if key in data:
                mat = data[key]
                break

        if mat is not None:
            try:
                if len(mat) == total_nodes and all(len(row) == total_nodes for row in mat):
                    return [[float(mat[i][j]) for j in range(total_nodes)] for i in range(total_nodes)]
            except Exception:
                pass

        coords = []
        for dep in self.depots:
            coords.append((float(dep.get("x", 0.0)), float(dep.get("y", 0.0))))
        for cust in self.customers:
            coords.append((float(cust.get("x", 0.0)), float(cust.get("y", 0.0))))

        if len(coords) != total_nodes:
            # Fallback to zero distances if malformed; instances should not trigger this.
            return [[0.0] * total_nodes for _ in range(total_nodes)]

        dist = [[0.0] * total_nodes for _ in range(total_nodes)]
        for i in range(total_nodes):
            xi, yi = coords[i]
            for j in range(total_nodes):
                if i == j:
                    dist[i][j] = 0.0
                else:
                    xj, yj = coords[j]
                    dist[i][j] = math.hypot(xi - xj, yi - yj)
        return dist

    def _build_options(self):
        options = []
        for i, cust in enumerate(self.customers):
            freq = self.frequency[i]
            raw_patterns = cust.get("pattern_list", [])

            pats = []
            for pat in raw_patterns:
                days = tuple(sorted(set(int(p) for p in pat if 1 <= int(p) <= self.t)))
                if freq == 0:
                    if not days:
                        pats.append(())
                    elif len(days) == freq:
                        pats.append(days)
                else:
                    if days:
                        pats.append(days)

            if freq > 0:
                exact_len = [p for p in pats if len(p) == freq]
                if exact_len:
                    pats = exact_len

            # Deduplicate while preserving order.
            unique_pats = []
            seen = set()
            for p in pats:
                if p not in seen:
                    seen.add(p)
                    unique_pats.append(p)
            pats = unique_pats

            if not pats:
                if freq == 0:
                    pats = [()]
                else:
                    # Fallback only for malformed instances. Limit count to avoid explosion.
                    generated = []
                    if 0 < freq <= self.t:
                        for comb in itertools.combinations(range(1, self.t + 1), freq):
                            generated.append(tuple(comb))
                            if len(generated) >= 50:
                                break
                    else:
                        generated = [tuple(range(1, self.t + 1))]
                    pats = generated if generated else [tuple(range(1, self.t + 1))]

            cust_opts = []
            for dep in range(self.d):
                for pat in pats:
                    cust_opts.append((dep, pat))
            options.append(cust_opts)
        return options


class Solution:
    def __init__(self, prob):
        self.prob = prob
        self.assign_depot = [-1] * prob.n
        self.assign_pattern = [None] * prob.n
        self.routes = [[[] for _ in range(prob.t)] for __ in range(prob.d)]
        self.objective = INF

    def copy(self):
        prob = self.prob
        new = Solution(prob)
        new.assign_depot = self.assign_depot.copy()
        new.assign_pattern = list(self.assign_pattern)
        new.objective = self.objective
        new.routes = [
            [[r.copy() for r in self.routes[dep][p]] for p in range(prob.t)]
            for dep in range(prob.d)
        ]
        return new

    def recompute_objective(self):
        prob = self.prob
        obj = 0.0
        for dep in range(prob.d):
            for pidx in range(prob.t):
                routes = self.routes[dep][pidx]
                routes[:] = [r for r in routes if r.custs]
                for r in routes:
                    demand, duration, cost = compute_route_metrics(prob, dep, r.custs)
                    r.demand = demand
                    r.duration = duration
                    r.cost = cost
                    obj += cost
        self.objective = obj

    def build_routes(self, rng=None):
        prob = self.prob
        custs_by = [[[] for _ in range(prob.t)] for __ in range(prob.d)]

        for c in range(prob.n):
            dep = self.assign_depot[c]
            pat = self.assign_pattern[c]
            if dep < 0 or dep >= prob.d or not pat:
                continue
            for p in pat:
                pidx = int(p) - 1
                if 0 <= pidx < prob.t:
                    custs_by[dep][pidx].append(c)

        new_routes = [[[] for _ in range(prob.t)] for __ in range(prob.d)]
        for dep in range(prob.d):
            for pidx in range(prob.t):
                routes = build_routes_for_customers(prob, dep, custs_by[dep][pidx], rng)
                if routes is None:
                    return False
                new_routes[dep][pidx] = routes

        self.routes = new_routes
        self.recompute_objective()
        return True


def compute_route_metrics(prob, depot, custs):
    if not custs:
        return 0, 0.0, 0.0

    demand = 0
    service = 0.0
    travel = 0.0
    prev = depot
    off = prob.d
    dist = prob.dist

    for c in custs:
        node = off + c
        demand += prob.demand[c]
        service += prob.service[c]
        travel += dist[prev][node]
        prev = node

    travel += dist[prev][depot]
    return demand, travel + service, travel


def make_route(prob, depot, custs):
    if not custs:
        return Route()
    demand, duration, cost = compute_route_metrics(prob, depot, custs)
    if demand > prob.capacity + EPS:
        return None
    if duration > prob.duration_limit + EPS:
        return None
    return Route(custs, demand, duration, cost)


def insertion_delta(prob, route, depot, c, pos):
    r = route.custs
    L = len(r)
    cn = prob.d + c
    dist = prob.dist

    pred = depot if pos == 0 else prob.d + r[pos - 1]
    succ = depot if pos == L else prob.d + r[pos]

    add_cost = dist[pred][cn] + dist[cn][succ] - dist[pred][succ]
    add_dur = add_cost + prob.service[c]

    feasible = (
        route.demand + prob.demand[c] <= prob.capacity + EPS
        and route.duration + add_dur <= prob.duration_limit + EPS
    )
    return add_cost, add_dur, feasible


def removal_delta(prob, route, depot, idx):
    r = route.custs
    c = r[idx]
    cn = prob.d + c
    dist = prob.dist

    pred = depot if idx == 0 else prob.d + r[idx - 1]
    succ = depot if idx == len(r) - 1 else prob.d + r[idx + 1]

    rem_cost = dist[pred][cn] + dist[cn][succ] - dist[pred][succ]
    rem_dur = rem_cost + prob.service[c]
    return rem_cost, rem_dur


def apply_insert(prob, route, c, pos, add_cost, add_dur):
    route.custs.insert(pos, c)
    route.demand += prob.demand[c]
    route.duration += add_dur
    route.cost += add_cost


def apply_remove(prob, route, idx, rem_cost, rem_dur):
    c = route.custs.pop(idx)
    route.demand -= prob.demand[c]
    route.duration -= rem_dur
    route.cost -= rem_cost
    return c


def greedy_insertion(prob, depot, order):
    if prob.m <= 0 and order:
        return None

    routes = []
    for c in order:
        best = None
        best_key = INF

        for ri, route in enumerate(routes):
            if route.demand + prob.demand[c] > prob.capacity + EPS:
                continue
            for pos in range(len(route.custs) + 1):
                add_cost, add_dur, feas = insertion_delta(prob, route, depot, c, pos)
                if feas and add_cost < best_key:
                    best_key = add_cost
                    best = (add_cost, add_dur, ri, pos)

        if best is not None:
            add_cost, add_dur, ri, pos = best
            apply_insert(prob, routes[ri], c, pos, add_cost, add_dur)
        else:
            if prob.m > 0 and len(routes) < prob.m:
                empty = Route()
                add_cost, add_dur, feas = insertion_delta(prob, empty, depot, c, 0)
                if not feas:
                    return None
                r = Route()
                apply_insert(prob, r, c, 0, add_cost, add_dur)
                routes.append(r)
            else:
                return None

    return routes


def best_merge_two(prob, depot, r1, r2):
    base = r1.cost + r2.cost
    best = None

    seqs1 = [r1.custs, list(reversed(r1.custs))]
    seqs2 = [r2.custs, list(reversed(r2.custs))]
    seen = set()

    for s1 in seqs1:
        t1 = tuple(s1)
        for s2 in seqs2:
            t2 = tuple(s2)
            cand1 = t1 + t2
            cand2 = t2 + t1
            for cand in (cand1, cand2):
                if cand in seen:
                    continue
                seen.add(cand)
                nr = make_route(prob, depot, list(cand))
                if nr is not None:
                    saving = base - nr.cost
                    if best is None or saving > best[0]:
                        best = (saving, nr)
    return best


def savings_builder(prob, depot, custs):
    if prob.m <= 0 and custs:
        return None
    if not custs:
        return []

    routes = []
    for c in custs:
        r = make_route(prob, depot, [c])
        if r is None:
            return None
        routes.append(r)

    # Forced merges to satisfy fleet size.
    while len(routes) > prob.m:
        best_saving = -INF
        best_i = -1
        best_j = -1
        best_route = None

        for i in range(len(routes)):
            for j in range(i + 1, len(routes)):
                res = best_merge_two(prob, depot, routes[i], routes[j])
                if res is not None:
                    saving, nr = res
                    if best_route is None or saving > best_saving:
                        best_saving = saving
                        best_i = i
                        best_j = j
                        best_route = nr

        if best_route is None:
            return None

        routes.pop(best_j)
        routes.pop(best_i)
        routes.append(best_route)

    # Optional positive-saving merges.
    while len(routes) > 1:
        best_saving = EPS
        best_i = -1
        best_j = -1
        best_route = None

        for i in range(len(routes)):
            for j in range(i + 1, len(routes)):
                res = best_merge_two(prob, depot, routes[i], routes[j])
                if res is not None:
                    saving, nr = res
                    if saving > best_saving:
                        best_saving = saving
                        best_i = i
                        best_j = j
                        best_route = nr

        if best_route is None:
            break

        routes.pop(best_j)
        routes.pop(best_i)
        routes.append(best_route)

    return routes


def build_routes_for_customers(prob, depot, custs, rng=None):
    if not custs:
        return []

    best = None
    best_cost = INF

    orders = []
    orders.append(sorted(custs, key=lambda c: -prob.demand[c]))
    orders.append(sorted(custs, key=lambda c: -(2.0 * prob.dist[depot][prob.d + c] + prob.service[c])))
    orders.append(sorted(custs, key=lambda c: -prob.service[c]))
    if rng is not None and len(custs) > 1:
        orders.append(rng.sample(custs, len(custs)))

    seen = set()
    for order in orders:
        key = tuple(order)
        if key in seen:
            continue
        seen.add(key)

        routes = greedy_insertion(prob, depot, order)
        if routes is not None:
            routes = [r for r in routes if r.custs]
            if len(routes) <= prob.m:
                cost = sum(r.cost for r in routes)
                if cost < best_cost:
                    best_cost = cost
                    best = [r.copy() for r in routes]

    if len(custs) <= 45:
        routes = savings_builder(prob, depot, custs)
        if routes is not None:
            cost = sum(r.cost for r in routes)
            if cost < best_cost:
                best = routes

    return best


def best_insert_action(prob, sol, depot, day_idx, c):
    routes = sol.routes[depot][day_idx]
    best_key = INF
    best = None

    for ri, route in enumerate(routes):
        if route.demand + prob.demand[c] > prob.capacity + EPS:
            continue
        for pos in range(len(route.custs) + 1):
            add_cost, add_dur, feas = insertion_delta(prob, route, depot, c, pos)
            if feas and add_cost < best_key:
                best_key = add_cost
                best = (add_cost, add_dur, ri, pos, False)

    if prob.m > 0 and len(routes) < prob.m:
        empty = Route()
        add_cost, add_dur, feas = insertion_delta(prob, empty, depot, c, 0)
        if feas:
            penalty = max(1e-6, prob.avg_dist * 0.01)
            if add_cost + penalty < best_key:
                best = (add_cost, add_dur, -1, 0, True)

    return best


def apply_insert_action(prob, sol, depot, day_idx, c, action):
    add_cost, add_dur, ri, pos, is_new = action
    routes = sol.routes[depot][day_idx]

    if is_new:
        r = Route()
        apply_insert(prob, r, c, pos, add_cost, add_dur)
        routes.append(r)
    else:
        if ri >= len(routes):
            action2 = best_insert_action(prob, sol, depot, day_idx, c)
            if action2 is None:
                return False
            return apply_insert_action(prob, sol, depot, day_idx, c, action2)
        apply_insert(prob, routes[ri], c, pos, add_cost, add_dur)
    return True


def customer_insert_candidates(prob, sol, c, rng, noise=0.0):
    cands = []

    for depot, pattern in prob.options[c]:
        if depot < 0 or depot >= prob.d:
            continue

        if not pattern:
            cands.append((0.0, depot, pattern, []))
            continue

        actions = []
        total = 0.0
        feasible = True

        for p in pattern:
            pidx = int(p) - 1
            if not (0 <= pidx < prob.t):
                feasible = False
                break
            action = best_insert_action(prob, sol, depot, pidx, c)
            if action is None:
                feasible = False
                break
            total += action[0]
            actions.append((pidx, action))

        if feasible:
            score = total
            if noise > 0.0:
                score += rng.uniform(-noise, noise) * max(1.0, abs(total))
            cands.append((score, depot, pattern, actions))

    cands.sort(key=lambda x: x[0])
    return cands


def apply_candidate(prob, sol, c, cand):
    _, depot, pattern, actions = cand
    done = []
    for pidx, action in actions:
        if not apply_insert_action(prob, sol, depot, pidx, c, action):
            for old_pidx in done:
                remove_customer_from_day(prob, sol, depot, old_pidx, c)
            sol.recompute_objective()
            return False
        done.append(pidx)
    sol.assign_depot[c] = depot
    sol.assign_pattern[c] = tuple(pattern)
    return True


def remove_customer_from_day(prob, sol, depot, day_idx, c):
    routes = sol.routes[depot][day_idx]
    for route in routes:
        while c in route.custs:
            idx = route.custs.index(c)
            rem_cost, rem_dur = removal_delta(prob, route, depot, idx)
            apply_remove(prob, route, idx, rem_cost, rem_dur)


def remove_customer(prob, sol, c):
    for dep in range(prob.d):
        for day_idx in range(prob.t):
            remove_customer_from_day(prob, sol, dep, day_idx, c)
    sol.assign_depot[c] = -1
    sol.assign_pattern[c] = None


def unserved_count(prob, sol):
    served = [0] * prob.n
    for dep in range(prob.d):
        for day_idx in range(prob.t):
            for route in sol.routes[dep][day_idx]:
                for c in route.custs:
                    if 0 <= c < prob.n:
                        served[c] += 1

    bad = 0
    for c in range(prob.n):
        freq = prob.frequency[c]
        if freq <= 0:
            if served[c] != 0:
                bad += served[c]
        else:
            if served[c] != freq:
                bad += abs(freq - served[c])
    return bad


def solution_key(prob, sol):
    return (unserved_count(prob, sol), float(sol.objective))


def initial_incremental(prob, rng, noise=0.0):
    sol = Solution(prob)
    sol.objective = 0.0

    customers = list(range(prob.n))

    def customer_sort_key(c):
        freq = prob.frequency[c]
        demand = prob.demand[c]
        opts = len(prob.options[c])
        nearest = min((prob.dist[dep][prob.d + c] for dep, _ in prob.options[c]), default=0.0)
        return (freq, demand, -opts, -nearest)

    customers.sort(key=customer_sort_key, reverse=True)

    for c in customers:
        cands = customer_insert_candidates(prob, sol, c, rng, noise=noise)
        applied = False

        for cand in cands[:8]:
            backup = sol.copy()
            if apply_candidate(prob, sol, c, cand):
                sol.recompute_objective()
                applied = True
                break
            sol = backup

        if not applied and prob.options[c]:
            def option_key(opt):
                dep, pat = opt
                visits = len(pat)
                if visits == 0:
                    return (0, prob.dist[dep][prob.d + c])
                return (1, visits, prob.dist[dep][prob.d + c] * visits + prob.service[c] * visits)

            dep, pat = min(prob.options[c], key=option_key)
            sol.assign_depot[c] = dep
            sol.assign_pattern[c] = tuple(pat)

    sol.recompute_objective()
    return sol


def deterministic_assignment(prob, variant=0, rng=None):
    sol = Solution(prob)

    for c in range(prob.n):
        opts = prob.options[c]
        if not opts:
            continue

        if variant == 0:
            def key(opt):
                dep, pat = opt
                visits = len(pat)
                if visits == 0:
                    return (0, prob.dist[dep][prob.d + c])
                return (1, visits, prob.dist[dep][prob.d + c] * visits + prob.service[c] * visits)
            dep, pat = min(opts, key=key)
        elif variant == 1:
            def key(opt):
                dep, pat = opt
                visits = len(pat)
                if visits == 0:
                    return (0, prob.dist[dep][prob.d + c])
                return (1, prob.dist[dep][prob.d + c] * visits, visits)
            dep, pat = min(opts, key=key)
        else:
            dep, pat = rng.choice(opts) if rng is not None else opts[0]

        sol.assign_depot[c] = dep
        sol.assign_pattern[c] = tuple(pat)

    return sol


def local_search(prob, sol, rng, time_limit=5.0, max_passes=2):
    start = time.time()
    sol.recompute_objective()

    best = sol.copy()
    best_key = solution_key(prob, best)

    customers = [c for c in range(prob.n) if prob.frequency[c] > 0]
    if not customers:
        return best

    for pass_idx in range(max_passes):
        if time.time() - start >= time_limit:
            break

        order = customers[:]
        rng.shuffle(order)
        improved = False

        for c in order:
            if time.time() - start >= time_limit:
                break

            backup = sol.copy()
            old_key = solution_key(prob, backup)

            remove_customer(prob, sol, c)
            sol.recompute_objective()

            noise = 0.02 if pass_idx == 0 else 0.08
            cands = customer_insert_candidates(prob, sol, c, rng, noise=noise)
            inserted = False

            for cand in cands[:6]:
                if apply_candidate(prob, sol, c, cand):
                    sol.recompute_objective()
                    inserted = True
                    break
                sol.recompute_objective()

            if inserted:
                new_key = solution_key(prob, sol)
                accept = False

                if new_key[0] < old_key[0]:
                    accept = True
                elif new_key[0] == old_key[0] and new_key[1] < old_key[1] - EPS:
                    accept = True

                if accept:
                    if new_key < best_key:
                        best = sol.copy()
                        best_key = new_key
                    improved = True
                    continue

            sol = backup

        if not improved:
            break

    return best


def solve(prob, seed=0, time_limit=10.0):
    rng = random.Random(seed)
    start = time.time()
    sols = []

    def consider(sol):
        sol.recompute_objective()
        sols.append(sol)

    for noise in (0.0, 0.05, 0.10):
        if time.time() - start >= time_limit:
            break
        sol = initial_incremental(prob, rng, noise=noise)
        consider(sol)

        rebuilt = sol.copy()
        if rebuilt.build_routes(rng):
            consider(rebuilt)

    for variant in range(3):
        if time.time() - start >= time_limit:
            break
        sol = deterministic_assignment(prob, variant, rng)
        if sol.build_routes(rng):
            consider(sol)

    if not sols:
        sol = Solution(prob)
        for c in range(prob.n):
            if prob.options[c]:
                dep, pat = prob.options[c][0]
                sol.assign_depot[c] = dep
                sol.assign_pattern[c] = tuple(pat)
        sol.build_routes(rng)
        consider(sol)

    best = None
    best_key = None
    for sol in sols:
        sol.recompute_objective()
        key = solution_key(prob, sol)
        if best_key is None or key < best_key:
            best_key = key
            best = sol.copy()

    if best is None:
        best = Solution(prob)

    remain = max(0.25, time_limit - (time.time() - start))
    best = local_search(
        prob,
        best,
        rng,
        time_limit=min(remain, max(1.0, time_limit * 0.5)),
        max_passes=2,
    )
    best.recompute_objective()
    return best


def solution_to_dict(prob, sol):
    sol.recompute_objective()

    routes_out = []
    for dep in range(prob.d):
        for day_idx in range(prob.t):
            for route in sol.routes[dep][day_idx]:
                if not route.custs:
                    continue
                routes_out.append({
                    "depot": dep,
                    "day": day_idx + 1,
                    "customers": list(route.custs),
                    "demand": int(route.demand),
                    "duration": float(route.duration),
                    "cost": float(route.cost),
                })

    assignments = []
    for c in range(prob.n):
        assignments.append({
            "customer": c,
            "depot": sol.assign_depot[c],
            "pattern": list(sol.assign_pattern[c] or []),
        })

    return {
        "objective": float(sol.objective),
        "assignments": assignments,
        "routes": routes_out,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_pos", nargs="?", default=None)
    parser.add_argument("output_pos", nargs="?", default=None)
    parser.add_argument("--input", dest="input_flag", default=None)
    parser.add_argument("--output", dest="output_flag", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--time-limit", type=float, default=10.0)
    args = parser.parse_args()

    input_path = args.input_flag or args.input_pos or "problem.json"
    output_path = args.output_flag or args.output_pos

    prob = Problem(input_path)
    sol = solve(prob, seed=args.seed, time_limit=args.time_limit)
    result = solution_to_dict(prob, sol)

    text = json.dumps(result, indent=2)
    if output_path:
        with open(output_path, "w") as f:
            f.write(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
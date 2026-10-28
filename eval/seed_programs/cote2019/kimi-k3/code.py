import argparse
import json
import math
import random
import time
import itertools
import os

from solution_logger import SolutionLogger


def _to_int(x, default=0):
    try:
        return int(x)
    except Exception:
        try:
            return int(round(float(x)))
        except Exception:
            return default


def _to_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def normalize_realizations(item):
    reals = item.get("realizations") or []
    out = []
    for r in reals:
        p = max(0.0, _to_float(r.get("probability", 0.0), 0.0))
        h = max(0, _to_int(r.get("height", 0)))
        w = max(0, _to_int(r.get("width", 0)))
        wt = max(0, _to_int(r.get("weight", 0)))
        out.append((p, h, w, wt))
    if not out:
        out = [(1.0, 0, 0, 0)]
    s = sum(p for p, _, _, _ in out)
    if s <= 1e-15:
        q = 1.0 / len(out)
        return [(q, h, w, wt) for _, h, w, wt in out]
    return [(p / s, h, w, wt) for p, h, w, wt in out]


def strip_height(items, W, H):
    # Sufficient height for packing one customer's items into a band of width W.
    # The route uses one such band per customer; bands are stacked in service order.
    if not items:
        return 0
    for h, w in items:
        if h > H or w > W:
            return H + 1
    if sum(w for _, w in items) <= W:
        return max(h for h, _ in items)

    keys = [
        lambda a: (-a[0], -a[1], -(a[0] * a[1])),
        lambda a: (-a[1], -a[0], -(a[0] * a[1])),
        lambda a: (-(a[0] * a[1]), -a[0], -a[1]),
    ]
    best = 10**18
    for key in keys:
        arr = sorted(items, key=key)

        # Next-fit decreasing height.
        total = 0
        rem = 0
        row_h = 0
        for h, w in arr:
            if w <= rem:
                rem -= w
                if h > row_h:
                    row_h = h
            else:
                total += row_h
                rem = W - w
                row_h = h
        total += row_h
        if total < best:
            best = total

        # First-fit decreasing height into shelves.
        shelves = []  # [remaining_width, shelf_height]
        for h, w in arr:
            placed = False
            for sh in shelves:
                if sh[0] >= w and sh[1] >= h:
                    sh[0] -= w
                    placed = True
                    break
            if not placed:
                shelves.append([W - w, h])
        total = sum(sh[1] for sh in shelves)
        if total < best:
            best = total
    return best


def choose_certain_scenario(item_lists, W, H):
    items = []
    weight = 0
    area = 0
    for reals in item_lists:
        p, h, w, wt = max(reals, key=lambda x: x[0])
        items.append((h, w))
        weight += wt
        area += h * w
    return (1.0, strip_height(items, W, H), weight, area)


def customer_scenarios(cust, W, H, cust_enum_limit=20000, sample_scen=256):
    is_stoch = bool(cust.get("is_stochastic", False))
    item_lists = [normalize_realizations(it) for it in (cust.get("items") or [])]
    if not item_lists:
        return [(1.0, 0, 0, 0)]

    if not is_stoch:
        return [choose_certain_scenario(item_lists, W, H)]

    log_count = sum(math.log(max(1, len(rl))) for rl in item_lists)
    total_items = sum(len(rl) for rl in item_lists)
    scen = []

    if log_count <= math.log(cust_enum_limit):
        for combo in itertools.product(*item_lists):
            prob = 1.0
            items = []
            weight = 0
            area = 0
            for p, h, w, wt in combo:
                prob *= p
                items.append((h, w))
                weight += wt
                area += h * w
            scen.append((prob, strip_height(items, W, H), weight, area))
    else:
        nsamp = sample_scen
        if total_items > 0 and nsamp * total_items > 2_000_000:
            nsamp = max(8, 2_000_000 // total_items)
        seed = (_to_int(cust.get("id", 0)) * 2654435761 + 7919) & 0xFFFFFFFF
        rng = random.Random(seed)
        for _ in range(nsamp):
            items = []
            weight = 0
            area = 0
            prob = 1.0
            for reals in item_lists:
                x = rng.random()
                acc = 0.0
                picked = reals[-1]
                for r in reals:
                    acc += r[0]
                    if x <= acc + 1e-15:
                        picked = r
                        break
                p, h, w, wt = picked
                prob *= p
                items.append((h, w))
                weight += wt
                area += h * w
            scen.append((1.0 / nsamp, strip_height(items, W, H), weight, area))

    # Normalize probabilities defensively.
    s = sum(p for p, _, _, _ in scen)
    if s <= 1e-15:
        q = 1.0 / len(scen)
        return [(q, h, g, a) for _, h, g, a in scen]
    return [(p / s, h, g, a) for p, h, g, a in scen]


def scenario_stats(scen):
    if not scen:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    mh = sum(p * h for p, h, _, _ in scen)
    mg = sum(p * g for p, _, g, _ in scen)
    ma = sum(p * a for p, _, _, a in scen)
    vh = sum(p * (h - mh) ** 2 for p, h, _, _ in scen)
    vg = sum(p * (g - mg) ** 2 for p, _, g, _ in scen)
    va = sum(p * (a - ma) ** 2 for p, _, _, a in scen)
    return (mh, vh, mg, vg, ma, va)


def min_unserved(hs, gs, H, Q):
    m = len(hs)
    if m == 0:
        return 0
    th = sum(hs)
    tg = sum(gs)
    eps = 1e-9
    if th <= H + eps and tg <= Q + eps:
        return 0

    # Try one removed customer exactly.
    for i in range(m):
        if th - hs[i] <= H + eps and tg - gs[i] <= Q + eps:
            return 1

    # Try pairs for small routes only.
    if m <= 16:
        for i in range(m):
            hi = hs[i]
            gi = gs[i]
            for j in range(i + 1, m):
                if th - hi - hs[j] <= H + eps and tg - gi - gs[j] <= Q + eps:
                    return 2

    # Greedy fallback: monotone removal, so this is a valid number of unserved customers.
    served = list(range(m))
    removed = 0
    while (th > H + eps or tg > Q + eps) and served:
        oh = max(0.0, th - H)
        ow = max(0.0, tg - Q)
        best_i = None
        best_key = None
        for i in served:
            red = 0.0
            if oh > eps:
                red += min(hs[i], oh) / oh
            if ow > eps:
                red += min(gs[i], ow) / ow
            post = max(0.0, th - hs[i] - H) + max(0.0, tg - gs[i] - Q)
            key = (red, -post, hs[i] + gs[i])
            if best_key is None or key > best_key:
                best_key = key
                best_i = i
        if best_i is None:
            break
        served.remove(best_i)
        th -= hs[best_i]
        tg -= gs[best_i]
        removed += 1
    return removed


def prob_exceed(mu, var, cap):
    if var <= 1e-12:
        return 1.0 if mu > cap + 1e-9 else 0.0
    sd = math.sqrt(var)
    if sd <= 0:
        return 1.0 if mu > cap else 0.0
    z = (cap - mu) / sd
    p = 0.5 * math.erfc(z / math.sqrt(2.0))
    if p < 0.0:
        p = 0.0
    elif p > 1.0:
        p = 1.0
    return p


class Evaluator:
    def __init__(self, customers, D, H, W, Q, cf, route_enum_limit=40000):
        self.customers = customers
        self.by_id = {c["id"]: c for c in customers}
        self.D = D
        self.H = H
        self.W = W
        self.Q = Q
        self.Acap = H * W
        self.cf = float(cf)
        self.route_enum_limit = route_enum_limit
        self.log_limit = math.log(max(1, route_enum_limit))
        self.rec_cache = {}

    def route_distance(self, route):
        if not route:
            return 0.0
        D = self.D
        cost = D[0][route[0]]
        for a, b in zip(route, route[1:]):
            cost += D[a][b]
        cost += D[route[-1]][0]
        return float(cost)

    def recourse_penalty(self, route):
        if not route:
            return 0.0
        key = tuple(sorted(route))
        val = self.rec_cache.get(key)
        if val is not None:
            return val

        scens = [self.by_id[i]["scenarios"] for i in key]
        log_combos = sum(math.log(max(1, len(s))) for s in scens)

        if log_combos <= self.log_limit:
            expected_unserved = 0.0
            for combo in itertools.product(*scens):
                prob = 1.0
                hs = []
                gs = []
                for p, h, g, _a in combo:
                    prob *= p
                    hs.append(h)
                    gs.append(g)
                expected_unserved += prob * min_unserved(hs, gs, self.H, self.Q)
            val = self.cf * expected_unserved
        else:
            mh = sum(self.by_id[i]["h_mean"] for i in key)
            vh = sum(self.by_id[i]["h_var"] for i in key)
            mg = sum(self.by_id[i]["g_mean"] for i in key)
            vg = sum(self.by_id[i]["g_var"] for i in key)
            ph = prob_exceed(mh, vh, self.H)
            pg = prob_exceed(mg, vg, self.Q)
            pfail = 1.0 - (1.0 - ph) * (1.0 - pg)
            base = min_unserved([self.by_id[i]["h_mean"] for i in key],
                                [self.by_id[i]["g_mean"] for i in key],
                                self.H, self.Q)
            expected_unserved = float(base) if base > 0 else float(pfail)
            val = self.cf * expected_unserved

        self.rec_cache[key] = val
        return val

    def route_objective(self, route):
        return self.route_distance(route) + self.recourse_penalty(route)

    def solution_objective(self, routes):
        return sum(self.route_objective(r) for r in routes)


def build_initial(customers, D, K, H, W, Q, cf, deadline):
    ids = [c["id"] for c in customers]
    n = len(ids)
    if K <= 0:
        return []
    if n == 0:
        return [[] for _ in range(K)]
    if n <= K:
        routes = [[i] for i in ids]
        while len(routes) < K:
            routes.append([])
        return routes[:K]

    by_id = {c["id"]: c for c in customers}
    area = {i: by_id[i]["exp_area"] for i in ids}
    weight = {i: by_id[i]["exp_weight"] for i in ids}
    Acap = float(H * W)

    def reserve_time():
        return time.time() < deadline - 0.05

    routes = [[] for _ in range(K)]
    ra = [0.0] * K
    rw = [0.0] * K

    unassigned = set(ids)
    seeds = [max(ids, key=lambda i: D[0][i])]
    unassigned.remove(seeds[0])
    while len(seeds) < K and unassigned:
        nx = max(unassigned, key=lambda i: min(D[i][s] for s in seeds) + 0.15 * D[0][i])
        seeds.append(nx)
        unassigned.remove(nx)
    for r, i in enumerate(seeds):
        routes[r] = [i]
        ra[r] = area[i]
        rw[r] = weight[i]

    while unassigned and reserve_time():
        best_choice = None
        best_regret = -1e100
        # A deterministic order keeps runtime stable.
        for u in sorted(unassigned):
            cand = []
            for r in range(K):
                rr = routes[r]
                for pos in range(len(rr) + 1):
                    na = ra[r] + area[u]
                    nw = rw[r] + weight[u]
                    overload = 0.0
                    if Acap > 0:
                        overload += max(0.0, na - Acap) / Acap
                    if Q > 0:
                        overload += max(0.0, nw - Q) / Q
                    prev = 0 if pos == 0 else rr[pos - 1]
                    nxt = 0 if pos == len(rr) else rr[pos]
                    dc = D[prev][u] + D[u][nxt] - D[prev][nxt]
                    cost = dc + (abs(cf) * 25.0 + 1.0) * overload
                    cand.append((cost, overload, dc, r, pos))
            if not cand:
                continue
            cand.sort(key=lambda x: (x[0], x[4], x[3]))
            first = cand[0]
            second = cand[1] if len(cand) > 1 else (first[0] + abs(first[2]) + 1.0, 0, 0, 0, 0)
            regret = second[0] - first[0]
            if regret > best_regret:
                best_regret = regret
                best_choice = (first, u)
        if best_choice is None:
            break
        (cost, overload, dc, r, pos), u = best_choice
        routes[r].insert(pos, u)
        ra[r] += area[u]
        rw[r] += weight[u]
        unassigned.remove(u)

    # If time ran out during regret insertion, still assign all customers.
    for u in sorted(unassigned):
        best = None
        for r in range(K):
            na = ra[r] + area[u]
            nw = rw[r] + weight[u]
            overload = 0.0
            if Acap > 0:
                overload += max(0.0, na - Acap) / Acap
            if Q > 0:
                overload += max(0.0, nw - Q) / Q
            rr = routes[r]
            prev = rr[-1] if rr else 0
            dc = D[prev][u] + D[u][0] - D[prev][0]
            cost = dc + (abs(cf) * 25.0 + 1.0) * overload
            if best is None or cost < best[0]:
                best = (cost, r)
        r = best[1]
        routes[r].append(u)
        ra[r] += area[u]
        rw[r] += weight[u]

    return routes[:K] + ([[] for _ in range(K - len(routes))] if len(routes) < K else [])


def two_opt(route, D, deadline):
    r = list(route)
    m = len(r)
    if m < 4:
        return r
    improved = True
    passes = 0
    while improved and passes < 30 and time.time() < deadline - 0.03:
        improved = False
        best_delta = -1e-10
        bi = bj = -1
        for i in range(m - 1):
            a = r[i - 1] if i > 0 else 0
            b = r[i]
            dab = D[a][b]
            for j in range(i + 1, m):
                c = r[j]
                d = r[j + 1] if j + 1 < m else 0
                delta = D[a][c] + D[b][d] - dab - D[c][d]
                if delta < best_delta:
                    best_delta = delta
                    bi, bj = i, j
        if bi >= 0:
            r[bi:bj + 1] = reversed(r[bi:bj + 1])
            improved = True
        passes += 1
    return r


def local_search(routes, ev, cf, deadline, logger):
    routes = [list(r) for r in routes]
    while len(routes) < 1:
        routes.append([])

    def sol_dict(obj):
        return {
            "objective_value": float(obj),
            "routes": [[int(x) for x in r] for r in routes],
        }

    best_obj = ev.solution_objective(routes)
    if logger:
        logger.log_solution(best_obj, sol_dict(best_obj))

    # Intra-route optimization. Recourse is set-based in this model, so distance-only 2-opt is safe.
    for idx in range(len(routes)):
        if time.time() >= deadline - 0.05:
            break
        old = routes[idx]
        new = two_opt(old, ev.D, deadline)
        if new != old:
            routes[idx] = new
            obj = ev.solution_objective(routes)
            if obj < best_obj - 1e-9:
                best_obj = obj
                if logger:
                    logger.log_solution(best_obj, sol_dict(best_obj))

    # Inter-route relocation.
    while time.time() < deadline - 0.08:
        K = len(routes)
        route_obj = [ev.route_objective(r) for r in routes]
        route_dist = [ev.route_distance(r) for r in routes]
        route_area = [sum(ev.by_id[i]["exp_area"] for i in r) for r in routes]
        route_weight = [sum(ev.by_id[i]["exp_weight"] for i in r) for r in routes]
        Acap = float(ev.H * ev.W)
        Q = float(ev.Q)

        best_delta = -1e-8
        best_move = None

        for a in range(K):
            if len(routes[a]) <= 1:
                continue
            ra = routes[a]
            for pu, u in enumerate(ra):
                if time.time() >= deadline - 0.05:
                    break
                prev = ra[pu - 1] if pu > 0 else 0
                nxt = ra[pu + 1] if pu + 1 < len(ra) else 0
                remove_delta = ev.D[prev][nxt] - ev.D[prev][u] - ev.D[u][nxt]
                a_minus = ra[:pu] + ra[pu + 1:]
                obj_a_minus = route_dist[a] + remove_delta + ev.recourse_penalty(a_minus)

                for b in range(K):
                    if b == a:
                        continue
                    new_area = route_area[b] + ev.by_id[u]["exp_area"]
                    new_weight = route_weight[b] + ev.by_id[u]["exp_weight"]
                    # Soft screen: do not spend time on extreme overload moves unless already bad.
                    if (Acap > 0 and new_area > 1.75 * Acap) or (Q > 0 and new_weight > 1.75 * Q):
                        continue
                    b_add = routes[b] + [u]
                    rec_b_add = ev.recourse_penalty(b_add)
                    rb = routes[b]
                    best_ins = 1e100
                    best_pos = 0
                    for pos in range(len(rb) + 1):
                        prevb = rb[pos - 1] if pos > 0 else 0
                        nxtb = rb[pos] if pos < len(rb) else 0
                        ins = ev.D[prevb][u] + ev.D[u][nxtb] - ev.D[prevb][nxtb]
                        if ins < best_ins:
                            best_ins = ins
                            best_pos = pos
                    delta = (obj_a_minus + route_dist[b] + best_ins + rec_b_add) - (route_obj[a] + route_obj[b])
                    if delta < best_delta:
                        best_delta = delta
                        best_move = (a, pu, u, b, best_pos)

        if best_move is None:
            break
        a, pu, u, b, pos = best_move
        routes[a].pop(pu)
        insert_pos = pos
        if b > a and routes[b] is not routes[a]:
            # After popping from a, positions in b are unchanged unless a and b are same (not here).
            pass
        routes[b].insert(insert_pos, u)

        # Re-optimize the two touched routes.
        for idx in (a, b):
            routes[idx] = two_opt(routes[idx], ev.D, deadline)

        obj = ev.solution_objective(routes)
        if obj < best_obj - 1e-9:
            best_obj = obj
            if logger:
                logger.log_solution(best_obj, sol_dict(best_obj))
        else:
            break

    return routes, best_obj


def solve_instance(instance, time_limit, logger):
    start = time.time()
    deadline = start + max(0, int(time_limit))

    veh = instance.get("vehicle", {}) or {}
    H = max(1, _to_int(veh.get("H", 1), 1))
    W = max(1, _to_int(veh.get("W", 1), 1))
    Q = max(1, _to_int(veh.get("Q", 1), 1))
    K = max(0, _to_int(veh.get("K", 0), 0))

    params = instance.get("parameters", {}) or {}
    cf = _to_float(params.get("recourse_cost_cf", 0), 0.0)

    D = instance.get("distance_matrix", []) or []
    raw_customers = instance.get("customers", []) or []

    customers = []
    for pos, c in enumerate(raw_customers, start=1):
        cid = _to_int(c.get("id", pos), pos)
        if cid <= 0 or cid >= len(D):
            cid = pos
        items = c.get("items") or []
        exp_area = 0.0
        exp_weight = 0.0
        normalized_items = []
        for it in items:
            rl = normalize_realizations(it)
            normalized_items.append(rl)
            for p, h, w, wt in rl:
                exp_area += p * h * w
                exp_weight += p * wt
        scen = customer_scenarios({**c, "items": items}, W, H)
        mh, vh, mg, vg, _ma, _va = scenario_stats(scen)
        customers.append({
            "id": cid,
            "exp_area": float(exp_area),
            "exp_weight": float(exp_weight),
            "scenarios": scen,
            "h_mean": mh,
            "h_var": vh,
            "g_mean": mg,
            "g_var": vg,
        })

    if D and len(D) < len(customers) + 1:
        # Defensive fallback: pad missing distances with a large value.
        n = len(customers) + 1
        maxd = 1
        for row in D:
            for v in row:
                if v + v > maxd:
                    maxd = v + v
        D = [[(D[i][j] if i < len(D) and j < len(D[i]) else maxd) for j in range(n)] for i in range(n)]

    ev = Evaluator(customers, D, H, W, Q, cf)

    routes = build_initial(customers, D, K, H, W, Q, cf, deadline)
    if K > len(routes):
        routes += [[] for _ in range(K - len(routes))]
    routes = routes[:K]

    routes, best_obj = local_search(routes, ev, cf, deadline, logger)

    # Final projection. Keep exactly K route lists.
    final_routes = [[int(x) for x in r] for r in routes]
    if len(final_routes) < K:
        final_routes += [[] for _ in range(K - len(final_routes))]
    final_routes = final_routes[:K]

    final_obj = ev.solution_object(final_routes)
    solution = {
        "objective_value": float(final_obj),
        "routes": final_routes,
    }
    if logger:
        logger.log_solution(final_obj, solution)
    return solution


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", required=True)
    ap.add_argument("--solution_path", required=True)
    ap.add_argument("--time_limit", type=int, required=True)
    ap.add_argument("--log_path", default=None)
    args = ap.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    sol = solve_instance(instance, args.time_limit, logger)

    out_dir = os.path.dirname(args.solution_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.solution_path, "w") as f:
        json.dump(sol, f)


if __name__ == "__main__":
    main()
import argparse
import json
import math
import random
import time
from typing import List, Dict, Tuple

from solution_logger import SolutionLogger


INF = 10**12


def shelf_depth(items_hw: List[Tuple[int, int]], W: int, H: int) -> float:
    """
    Compute packed depth using a simple FFDH shelf heuristic in a strip of width W.
    Item orientation is fixed (no rotation). Returns INF if impossible.
    """
    if not items_hw:
        return 0.0
    for h, w in items_hw:
        if w > W or h > H:
            return INF

    # Sort by height desc, then width desc
    items = sorted(items_hw, key=lambda x: (-x[0], -x[1]))

    # Shelves: [shelf_height, remaining_width]
    shelves = []
    total_depth = 0.0

    for h, w in items:
        placed = False
        for sh in shelves:
            if h <= sh[0] and w <= sh[1]:
                sh[1] -= w
                placed = True
                break
        if not placed:
            shelves.append([h, W - w])
            total_depth += h
            if total_depth > H:
                return INF

    return total_depth


def sample_item_realization(item, rng: random.Random):
    reals = item["realizations"]
    if len(reals) == 1:
        r = reals[0]
        return r["height"], r["width"], r["weight"]
    x = rng.random()
    cum = 0.0
    for r in reals:
        cum += r["probability"]
        if x <= cum + 1e-12:
            return r["height"], r["width"], r["weight"]
    r = reals[-1]
    return r["height"], r["width"], r["weight"]


def generate_scenarios(customers, S, W, H, rng):
    """
    Returns:
      depth_s[cid] = list length S
      weight_s[cid] = list length S
      exp_weight[cid], exp_area[cid]
    """
    depth_s: Dict[int, List[float]] = {}
    weight_s: Dict[int, List[float]] = {}
    exp_weight: Dict[int, float] = {}
    exp_area: Dict[int, float] = {}

    # exact expected weight/area from distributions
    for c in customers:
        cid = c["id"]
        ew = 0.0
        ea = 0.0
        for it in c["items"]:
            for r in it["realizations"]:
                p = r["probability"]
                ew += p * r["weight"]
                ea += p * r["height"] * r["width"]
        exp_weight[cid] = ew
        exp_area[cid] = ea

    for c in customers:
        cid = c["id"]
        depth_s[cid] = []
        weight_s[cid] = []
        for _ in range(S):
            items_hw = []
            tw = 0.0
            for it in c["items"]:
                h, w, wt = sample_item_realization(it, rng)
                items_hw.append((h, w))
                tw += wt
            d = shelf_depth(items_hw, W, H)
            depth_s[cid].append(d)
            weight_s[cid].append(tw)

    return depth_s, weight_s, exp_weight, exp_area


def travel_cost(route: List[int], dist: List[List[int]]) -> float:
    if not route:
        return 0.0
    c = dist[0][route[0]]
    for i in range(len(route) - 1):
        c += dist[route[i]][route[i + 1]]
    c += dist[route[-1]][0]
    return float(c)


def route_cost(
    route: List[int],
    dist: List[List[int]],
    depth_s: Dict[int, List[float]],
    weight_s: Dict[int, List[float]],
    H: int,
    Q: int,
    recourse_cf: float,
    cache: Dict[Tuple[int, ...], float],
) -> float:
    key = tuple(route)
    if key in cache:
        return cache[key]

    tc = travel_cost(route, dist)
    if not route:
        cache[key] = tc
        return tc

    S = len(next(iter(depth_s.values())))
    unserved_sum = 0.0
    m = len(route)

    for s in range(S):
        cum_d = 0.0
        cum_w = 0.0
        served = 0
        for cid in route:
            d = depth_s[cid][s]
            w = weight_s[cid][s]
            if d >= INF / 10:
                break
            if cum_d + d <= H + 1e-9 and cum_w + w <= Q + 1e-9:
                cum_d += d
                cum_w += w
                served += 1
            else:
                break
        unserved_sum += (m - served)

    pen = recourse_cf * (unserved_sum / S)
    val = tc + pen
    cache[key] = val
    return val


def nearest_neighbor_order(customers: List[int], dist: List[List[int]]) -> List[int]:
    if not customers:
        return []
    rem = set(customers)
    cur = 0
    route = []
    while rem:
        nxt = min(rem, key=lambda x: dist[cur][x])
        route.append(nxt)
        rem.remove(nxt)
        cur = nxt
    return route


def build_initial_routes(
    customer_ids: List[int],
    coords: Dict[int, Tuple[float, float]],
    K: int,
    exp_weight: Dict[int, float],
    exp_depth_proxy: Dict[int, float],
    Q: int,
    H: int,
    dist: List[List[int]],
) -> List[List[int]]:
    n = len(customer_ids)
    if K <= 0:
        return []
    if n == 0:
        return [[] for _ in range(K)]

    depot = coords[0]
    ordered = sorted(
        customer_ids,
        key=lambda cid: math.atan2(coords[cid][1] - depot[1], coords[cid][0] - depot[0]),
    )

    total_norm = 0.0
    for cid in ordered:
        total_norm += max(exp_weight[cid] / max(1.0, Q), exp_depth_proxy[cid] / max(1.0, H))
    target = total_norm / K if K > 0 else total_norm

    routes = []
    idx = 0
    remaining = n
    remaining_routes = K

    for k in range(K):
        if idx >= n:
            routes.append([])
            remaining_routes -= 1
            continue

        if k == K - 1:
            chunk = ordered[idx:]
            routes.append(nearest_neighbor_order(chunk, dist))
            break

        min_take = 1 if remaining >= remaining_routes else 0
        max_take = max(0, remaining - (remaining_routes - 1))
        take = 0
        acc = 0.0
        while idx + take < n and take < max_take:
            cid = ordered[idx + take]
            acc += max(exp_weight[cid] / max(1.0, Q), exp_depth_proxy[cid] / max(1.0, H))
            take += 1
            if take >= min_take and acc >= target:
                break
        if take == 0 and max_take > 0:
            take = 1
        chunk = ordered[idx: idx + take]
        routes.append(nearest_neighbor_order(chunk, dist))
        idx += take
        remaining -= take
        remaining_routes -= 1

    while len(routes) < K:
        routes.append([])

    return routes


def total_solution_cost(routes, *args):
    return sum(route_cost(r, *args) for r in routes)


def solve(instance, time_limit, logger=None):
    t0 = time.time()
    rng = random.Random(42)

    vehicle = instance["vehicle"]
    H, W, Q, K = vehicle["H"], vehicle["W"], vehicle["Q"], vehicle["K"]
    recourse_cf = instance["parameters"]["recourse_cost_cf"]

    customers = instance["customers"]
    n = len(customers)
    dist = instance["distance_matrix"]

    coords = {0: (instance["depot"]["x"], instance["depot"]["y"])}
    customer_ids = []
    any_stochastic = False
    for c in customers:
        cid = c["id"]
        customer_ids.append(cid)
        coords[cid] = (0.0, 0.0)  # fallback if coords not provided
        if "x" in c and "y" in c:
            coords[cid] = (c["x"], c["y"])
        if c.get("is_stochastic", False):
            any_stochastic = True
        else:
            for it in c["items"]:
                if it.get("is_stochastic", False) or it.get("num_realizations", 1) > 1:
                    any_stochastic = True

    # number of Monte Carlo scenarios
    if any_stochastic:
        S = max(30, min(120, int(2000 / max(1, n))))
    else:
        S = 1

    depth_s, weight_s, exp_weight, exp_area = generate_scenarios(customers, S, W, H, rng)

    exp_depth_proxy = {}
    for cid in customer_ids:
        exp_depth_proxy[cid] = sum(depth_s[cid]) / len(depth_s[cid]) if depth_s[cid] else 0.0
        if exp_depth_proxy[cid] >= INF / 10:
            exp_depth_proxy[cid] = H + 1.0

    routes = build_initial_routes(
        customer_ids, coords, K, exp_weight, exp_depth_proxy, Q, H, dist
    )

    # Ensure all customers appear exactly once
    seen = set()
    for r in routes:
        for cid in r:
            seen.add(cid)
    missing = [cid for cid in customer_ids if cid not in seen]
    for cid in missing:
        rid = min(range(K), key=lambda x: len(routes[x]))
        routes[rid].append(cid)

    # if n >= K, try to avoid empty routes
    if n >= K:
        empties = [i for i in range(K) if len(routes[i]) == 0]
        while empties:
            e = empties.pop()
            donor = max(range(K), key=lambda x: len(routes[x]))
            if len(routes[donor]) <= 1:
                break
            routes[e].append(routes[donor].pop())

    cache = {}
    cur_obj = total_solution_cost(routes, dist, depth_s, weight_s, H, Q, recourse_cf, cache)
    best_routes = [r[:] for r in routes]
    best_obj = cur_obj
    if logger:
        logger.log(best_obj)

    require_nonempty = n >= K
    max_no_improve = 400
    no_improve = 0

    while time.time() - t0 < time_limit - 0.05:
        improved = False

        # random neighborhood move type
        mv = rng.randint(0, 2)

        if mv == 0:
            # relocate
            src = rng.randrange(K)
            if len(routes[src]) == 0:
                no_improve += 1
                continue
            if require_nonempty and len(routes[src]) <= 1:
                no_improve += 1
                continue
            i = rng.randrange(len(routes[src]))
            cid = routes[src][i]

            dst = rng.randrange(K)
            j = rng.randrange(len(routes[dst]) + 1)

            if src == dst and (j == i or j == i + 1):
                no_improve += 1
                continue

            old_src = routes[src]
            old_dst = routes[dst]

            new_src = old_src[:i] + old_src[i + 1:]
            new_dst = old_dst[:j] + [cid] + old_dst[j:] if src != dst else None

            if src == dst:
                tmp = old_src[:]
                tmp.pop(i)
                if j > i:
                    j -= 1
                tmp.insert(j, cid)
                new_src = tmp
                delta = route_cost(new_src, dist, depth_s, weight_s, H, Q, recourse_cf, cache) - \
                        route_cost(old_src, dist, depth_s, weight_s, H, Q, recourse_cf, cache)
                cand_obj = cur_obj + delta
                if cand_obj + 1e-9 < cur_obj:
                    routes[src] = new_src
                    cur_obj = cand_obj
                    improved = True
            else:
                delta = (
                    route_cost(new_src, dist, depth_s, weight_s, H, Q, recourse_cf, cache)
                    + route_cost(new_dst, dist, depth_s, weight_s, H, Q, recourse_cf, cache)
                    - route_cost(old_src, dist, depth_s, weight_s, H, Q, recourse_cf, cache)
                    - route_cost(old_dst, dist, depth_s, weight_s, H, Q, recourse_cf, cache)
                )
                cand_obj = cur_obj + delta
                if cand_obj + 1e-9 < cur_obj:
                    routes[src] = new_src
                    routes[dst] = new_dst
                    cur_obj = cand_obj
                    improved = True

        elif mv == 1:
            # swap between routes
            a = rng.randrange(K)
            b = rng.randrange(K)
            if a == b or len(routes[a]) == 0 or len(routes[b]) == 0:
                no_improve += 1
                continue
            i = rng.randrange(len(routes[a]))
            j = rng.randrange(len(routes[b]))

            old_a = routes[a]
            old_b = routes[b]
            new_a = old_a[:]
            new_b = old_b[:]
            new_a[i], new_b[j] = new_b[j], new_a[i]

            delta = (
                route_cost(new_a, dist, depth_s, weight_s, H, Q, recourse_cf, cache)
                + route_cost(new_b, dist, depth_s, weight_s, H, Q, recourse_cf, cache)
                - route_cost(old_a, dist, depth_s, weight_s, H, Q, recourse_cf, cache)
                - route_cost(old_b, dist, depth_s, weight_s, H, Q, recourse_cf, cache)
            )
            cand_obj = cur_obj + delta
            if cand_obj + 1e-9 < cur_obj:
                routes[a] = new_a
                routes[b] = new_b
                cur_obj = cand_obj
                improved = True

        else:
            # intra-route 2-opt
            r = rng.randrange(K)
            m = len(routes[r])
            if m < 4:
                no_improve += 1
                continue
            i = rng.randrange(0, m - 2)
            j = rng.randrange(i + 2, m)
            old_r = routes[r]
            new_r = old_r[:i] + old_r[i:j][::-1] + old_r[j:]

            delta = (
                route_cost(new_r, dist, depth_s, weight_s, H, Q, recourse_cf, cache)
                - route_cost(old_r, dist, depth_s, weight_s, H, Q, recourse_cf, cache)
            )
            cand_obj = cur_obj + delta
            if cand_obj + 1e-9 < cur_obj:
                routes[r] = new_r
                cur_obj = cand_obj
                improved = True

        if improved:
            no_improve = 0
            if cur_obj + 1e-9 < best_obj:
                best_obj = cur_obj
                best_routes = [r[:] for r in routes]
                if logger:
                    logger.log(best_obj)
        else:
            no_improve += 1

        # diversification
        if no_improve >= max_no_improve and time.time() - t0 < time_limit - 0.2:
            no_improve = 0
            a = rng.randrange(K)
            b = rng.randrange(K)
            if a != b and len(routes[a]) > 0 and len(routes[b]) > 0:
                i = rng.randrange(len(routes[a]))
                j = rng.randrange(len(routes[b]))
                routes[a][i], routes[b][j] = routes[b][j], routes[a][i]
                cur_obj = total_solution_cost(
                    routes, dist, depth_s, weight_s, H, Q, recourse_cf, cache
                )

    # final cleanup: ensure all customers exactly once
    assigned = []
    for r in best_routes:
        assigned.extend(r)
    count = {}
    for cid in assigned:
        count[cid] = count.get(cid, 0) + 1

    # remove duplicates
    for k in range(K):
        newr = []
        for cid in best_routes[k]:
            if count.get(cid, 0) > 1:
                count[cid] -= 1
            else:
                newr.append(cid)
        best_routes[k] = newr

    # insert missing
    missing = [cid for cid in customer_ids if cid not in set(sum(best_routes, []))]
    for cid in missing:
        rid = min(range(K), key=lambda x: len(best_routes[x]))
        best_routes[rid].append(cid)

    if require_nonempty:
        empties = [i for i in range(K) if len(best_routes[i]) == 0]
        for e in empties:
            donor = max(range(K), key=lambda x: len(best_routes[x]))
            if len(best_routes[donor]) > 1:
                best_routes[e].append(best_routes[donor].pop())

    best_obj = total_solution_cost(
        best_routes, dist, depth_s, weight_s, H, Q, recourse_cf, {}
    )

    return best_obj, best_routes


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

    obj, routes = solve(instance, args.time_limit, logger)

    sol = {
        "objective_value": float(obj),
        "routes": [[int(cid) for cid in r] for r in routes],
    }

    with open(args.solution_path, "w") as f:
        json.dump(sol, f)


if __name__ == "__main__":
    main()
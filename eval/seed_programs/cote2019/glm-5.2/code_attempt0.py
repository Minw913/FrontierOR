#!/usr/bin/env python3
import json
import argparse
import itertools
import random
import time
from solution_logger import SolutionLogger


def shelf_pack_height(items, W):
    """Compute strip height using shelf-based packing (FFD by height)."""
    if not items:
        return 0
    sorted_items = sorted(items, key=lambda x: -x[0])
    total_h = 0
    cur_h = 0
    cur_w = 0
    for h, w in sorted_items:
        if cur_w + w <= W:
            cur_w += w
            if h > cur_h:
                cur_h = h
        else:
            total_h += cur_h
            cur_h = h
            cur_w = w
    return total_h + cur_h


def check_packing(route_custs, scenario, H, W):
    """Check LIFO packing feasibility using layer approach."""
    total_h = 0
    for c in route_custs:
        items = []
        for it in c['items']:
            if it['is_stochastic']:
                r = it['realizations'][scenario[(c['id'], it['item_id'])]]
            else:
                r = it['realizations'][0]
            items.append((r['height'], r['width']))
        h = shelf_pack_height(items, W)
        total_h += h
        if total_h > H:
            return False
    return True


def check_weight(route_custs, scenario, Q):
    total_w = 0
    for c in route_custs:
        for it in c['items']:
            if it['is_stochastic']:
                r = it['realizations'][scenario[(c['id'], it['item_id'])]]
            else:
                r = it['realizations'][0]
            total_w += r['weight']
    return total_w <= Q


def evaluate_route(route_custs, H, W, Q, penalty, max_scenarios=5000):
    """Returns (is_allowed, expected_penalty)."""
    stoch = []
    for c in route_custs:
        for it in c['items']:
            if it['is_stochastic']:
                stoch.append((c['id'], it['item_id'], len(it['realizations']),
                             [r['probability'] for r in it['realizations']]))

    if not stoch:
        scenario = {}
        ok = check_packing(route_custs, scenario, H, W) and check_weight(route_custs, scenario, Q)
        if ok:
            return True, 0.0
        else:
            return True, float(len(route_custs) * penalty)

    total_comb = 1
    for s in stoch:
        total_comb *= s[2]

    if total_comb <= max_scenarios:
        any_ok = False
        exp_pen = 0.0
        for combo in itertools.product(*[range(s[2]) for s in stoch]):
            scenario = {}
            prob = 1.0
            for i, s in enumerate(stoch):
                scenario[(s[0], s[1])] = combo[i]
                prob *= s[3][combo[i]]
            ok = check_packing(route_custs, scenario, H, W) and check_weight(route_custs, scenario, Q)
            if ok:
                any_ok = True
            else:
                exp_pen += prob * len(route_custs) * penalty
        if not any_ok:
            return False, 0.0
        return True, exp_pen
    else:
        random.seed(42)
        any_ok = False
        total_pen = 0.0
        n_samples = max_scenarios
        for _ in range(n_samples):
            scenario = {}
            for s in stoch:
                idx = random.choices(range(s[2]), weights=s[3])[0]
                scenario[(s[0], s[1])] = idx
            ok = check_packing(route_custs, scenario, H, W) and check_weight(route_custs, scenario, Q)
            if ok:
                any_ok = True
            else:
                total_pen += len(route_custs) * penalty
        if not any_ok:
            return False, 0.0
        return True, total_pen / n_samples


def route_cost(route_ids, dist):
    if not route_ids:
        return 0
    c = dist[0][route_ids[0]]
    for i in range(len(route_ids) - 1):
        c += dist[route_ids[i]][route_ids[i + 1]]
    c += dist[route_ids[-1]][0]
    return c


def two_opt(route_ids, dist):
    if len(route_ids) <= 2:
        return route_ids
    best = route_ids[:]
    best_cost = route_cost(best, dist)
    improved = True
    while improved:
        improved = False
        for i in range(len(best)):
            for j in range(i + 1, len(best)):
                nr = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                nc = route_cost(nr, dist)
                if nc < best_cost - 1e-6:
                    best = nr
                    best_cost = nc
                    improved = True
    return best


def nearest_neighbor(ids, dist):
    if len(ids) <= 1:
        return ids
    unvisited = set(ids)
    route = []
    current = 0
    while unvisited:
        nearest = min(unvisited, key=lambda x: dist[current][x])
        route.append(nearest)
        current = nearest
        unvisited.remove(nearest)
    return route


def solve(instance, time_limit, logger):
    t0 = time.time()
    K = instance['vehicle']['K']
    H = instance['vehicle']['H']
    W = instance['vehicle']['W']
    Q = instance['vehicle']['Q']
    penalty = instance['parameters']['recourse_cost_cf']
    dist = instance['distance_matrix']
    customers = instance['customers']
    cust_by_id = {c['id']: c for c in customers}
    cust_ids = [c['id'] for c in customers]
    n = len(cust_ids)

    # Expected area and weight
    ea = {}
    ew = {}
    for c in customers:
        a, w = 0.0, 0.0
        for item in c['items']:
            for r in item['realizations']:
                p = r['probability']
                a += p * r['height'] * r['width']
                w += p * r['weight']
        ea[c['id']] = a
        ew[c['id']] = w

    # Route evaluation cache
    eval_cache = {}

    def eval_cached(route_ids):
        key = tuple(route_ids)
        if key in eval_cache:
            return eval_cache[key]
        rc = [cust_by_id[c] for c in route_ids]
        result = evaluate_route(rc, H, W, Q, penalty)
        eval_cache[key] = result
        return result

    def compute_obj(rts):
        tc = 0
        tp = 0
        for r in rts:
            if r:
                tc += route_cost(r, dist)
                allowed, pen = eval_cached(r)
                if allowed:
                    tp += pen
                else:
                    tp += penalty * len(r)
        return tc + tp

    # Clarke-Wright savings
    savings = []
    for i in range(n):
        for j in range(i + 1, n):
            ci, cj = cust_ids[i], cust_ids[j]
            s = dist[0][ci] + dist[0][cj] - dist[ci][cj]
            savings.append((s, ci, cj))
    savings.sort(reverse=True)

    routes = [[cid] for cid in cust_ids]
    route_of = {cid: idx for idx, cid in enumerate(cust_ids)}
    r_area = [ea[cid] for cid in cust_ids]
    r_weight = [ew[cid] for cid in cust_ids]

    for s, ci, cj in savings:
        if time.time() - t0 > time_limit * 0.3:
            break
        non_empty = sum(1 for r in routes if r is not None)
        if non_empty <= K:
            break
        ri, rj = route_of[ci], route_of[cj]
        if ri == rj or routes[ri] is None or routes[rj] is None:
            continue
        ca = r_area[ri] + r_area[rj]
        cw = r_weight[ri] + r_weight[rj]
        if ca > H * W + 1e-6 or cw > Q + 1e-6:
            continue
        a, b = routes[ri], routes[rj]
        merged = None
        if a[-1] == ci and b[0] == cj:
            merged = a + b
        elif a[0] == ci and b[-1] == cj:
            merged = b + a
        elif a[-1] == ci and b[-1] == cj:
            merged = a + b[::-1]
        elif a[0] == ci and b[0] == cj:
            merged = a[::-1] + b
        if merged is None:
            continue
        routes[ri] = merged
        r_area[ri] = ca
        r_weight[ri] = cw
        routes[rj] = None
        for cid in merged:
            route_of[cid] = ri

    routes = [r for r in routes if r is not None]

    # Ensure exactly K routes
    while len(routes) < K:
        if routes and any(len(r) > 1 for r in routes):
            idx = max(range(len(routes)), key=lambda i: len(routes[i]))
            mid = len(routes[idx]) // 2
            if mid > 0:
                routes.append(routes[idx][mid:])
                routes[idx] = routes[idx][:mid]
            else:
                routes.append([])
        else:
            routes.append([])
    while len(routes) > K:
        routes.sort(key=len)
        routes[-2] = routes[-2] + routes[-1]
        routes.pop()

    # TSP improvement
    for i in range(len(routes)):
        if len(routes[i]) > 8:
            routes[i] = nearest_neighbor(routes[i], dist)
        routes[i] = two_opt(routes[i], dist)

    obj = compute_obj(routes)
    sol = {"objective_value": obj, "routes": [r[:] for r in routes]}
    if logger:
        logger.log_solution(obj, sol)

    # Local search: relocate customers between routes
    improved = True
    while improved and time.time() - t0 < time_limit * 0.7:
        improved = False
        for ri in range(len(routes)):
            if not routes[ri]:
                continue
            for pos in range(len(routes[ri])):
                if time.time() - t0 > time_limit * 0.7:
                    break
                cust = routes[ri][pos]
                new_ri = routes[ri][:pos] + routes[ri][pos + 1:]
                for rj in range(len(routes)):
                    if ri == rj:
                        continue
                    new_rj = routes[rj] + [cust]
                    area = sum(ea[c] for c in new_rj)
                    wt = sum(ew[c] for c in new_rj)
                    if area > H * W + 1e-6 or wt > Q + 1e-6:
                        continue
                    old = [r[:] for r in routes]
                    routes[ri] = new_ri
                    routes[rj] = two_opt(new_rj, dist) if len(new_rj) > 2 else new_rj
                    if new_ri:
                        routes[ri] = two_opt(new_ri, dist) if len(new_ri) > 2 else new_ri
                    new_obj = compute_obj(routes)
                    if new_obj < obj - 1e-6:
                        obj = new_obj
                        sol = {"objective_value": obj, "routes": [r[:] for r in routes]}
                        if logger:
                            logger.log_solution(obj, sol)
                        improved = True
                        break
                    else:
                        routes = old
                if improved:
                    break
            if improved:
                break

    # Local search: swap customers between routes
    improved = True
    while improved and time.time() - t0 < time_limit * 0.85:
        improved = False
        for ri in range(len(routes)):
            for rj in range(ri + 1, len(routes)):
                if time.time() - t0 > time_limit * 0.85:
                    break
                if not routes[ri] or not routes[rj]:
                    continue
                for pi in range(len(routes[ri])):
                    for pj in range(len(routes[rj])):
                        ci = routes[ri][pi]
                        cj = routes[rj][pj]
                        new_ri = routes[ri][:pi] + [cj] + routes[ri][pi + 1:]
                        new_rj = routes[rj][:pj] + [ci] + routes[rj][pj + 1:]
                        ai = sum(ea[c] for c in new_ri)
                        wi = sum(ew[c] for c in new_ri)
                        aj = sum(ea[c] for c in new_rj)
                        wj = sum(ew[c] for c in new_rj)
                        if ai > H * W + 1e-6 or wi > Q + 1e-6 or aj > H * W + 1e-6 or wj > Q + 1e-6:
                            continue
                        old = [r[:] for r in routes]
                        routes[ri] = two_opt(new_ri, dist) if len(new_ri) > 2 else new_ri
                        routes[rj] = two_opt(new_rj, dist) if len(new_rj) > 2 else new_rj
                        new_obj = compute_obj(routes)
                        if new_obj < obj - 1e-6:
                            obj = new_obj
                            sol = {"objective_value": obj, "routes": [r[:] for r in routes]}
                            if logger:
                                logger.log_solution(obj, sol)
                            improved = True
                            break
                        else:
                            routes = old
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

    # Local search: or-opt (move segments of 2-3 customers)
    improved = True
    while improved and time.time() - t0 < time_limit * 0.92:
        improved = False
        for ri in range(len(routes)):
            if not routes[ri]:
                continue
            for seg_len in [2, 3]:
                if time.time() - t0 > time_limit * 0.92:
                    break
                for pos in range(len(routes[ri]) - seg_len + 1):
                    seg = routes[ri][pos:pos + seg_len]
                    new_ri = routes[ri][:pos] + routes[ri][pos + seg_len:]
                    for rj in range(len(routes)):
                        if ri == rj:
                            continue
                        for insert_pos in range(len(routes[rj]) + 1):
                            new_rj = routes[rj][:insert_pos] + seg + routes[rj][insert_pos:]
                            area = sum(ea[c] for c in new_rj)
                            wt = sum(ew[c] for c in new_rj)
                            if area > H * W + 1e-6 or wt > Q + 1e-6:
                                continue
                            old = [r[:] for r in routes]
                            routes[ri] = new_ri
                            routes[rj] = two_opt(new_rj, dist) if len(new_rj) > 2 else new_rj
                            if new_ri:
                                routes[ri] = two_opt(new_ri, dist) if len(new_ri) > 2 else new_ri
                            new_obj = compute_obj(routes)
                            if new_obj < obj - 1e-6:
                                obj = new_obj
                                sol = {"objective_value": obj, "routes": [r[:] for r in routes]}
                                if logger:
                                    logger.log_solution(obj, sol)
                                improved = True
                                break
                            else:
                                routes = old
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

    # Final 2-opt on all routes
    for i in range(len(routes)):
        routes[i] = two_opt(routes[i], dist)

    obj = compute_obj(routes)
    sol = {"objective_value": obj, "routes": routes}
    if logger:
        logger.log_solution(obj, sol)

    return sol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, default=60)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path) as f:
        instance = json.load(f)

    sol = solve(instance, args.time_limit, logger)

    with open(args.solution_path, 'w') as f:
        json.dump(sol, f, indent=2)


if __name__ == '__main__':
    main()
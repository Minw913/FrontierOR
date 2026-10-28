import json
import argparse
import math
import time
import heapq
import random
from solution_logger import SolutionLogger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    start_time = time.time()
    deadline = start_time + args.time_limit

    with open(args.instance_path) as f:
        inst = json.load(f)

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    num_sites = inst['num_sites']
    num_customers = inst['num_customers']
    T = inst['parameters']['T']
    delta = inst['parameters']['delta']
    epsilon = inst['parameters']['epsilon']
    L = inst['parameters']['L']
    c_travel = inst['parameters']['c']
    q = inst['parameters']['q']
    weekly = inst['parameters']['weekly_attraction_cycle']
    avg_biz_c = inst['parameters']['avg_businesses_customers']
    avg_biz_s = inst['parameters']['avg_businesses_sites']

    sites = inst['sites']
    customers = inst['customers']

    site_ids = [s['id'] for s in sites]
    site_capacity = [s['capacity'] for s in sites]
    site_biz = [s['num_businesses'] for s in sites]
    site_rev = [s['revenue_per_customer'] for s in sites]
    site_fc = [s['fixed_cost'] for s in sites]
    site_max_events = [s['max_events'] for s in sites]

    cust_pop = [cu['population'] for cu in customers]
    cust_biz = [cu['num_businesses'] for cu in customers]
    cust_ids = [cu['id'] for cu in customers]

    s2s = inst['site_to_site_distances']
    c2s = inst['customer_to_site_distances']

    def get_s2s(i, j):
        return s2s[f"{i}_{j}"]

    def get_c2s(c_id, s_id):
        return c2s[f"{c_id}_{s_id}"]

    site_attr = [(site_biz[s] / avg_biz_s) if avg_biz_s > 0 else 0.0 for s in range(num_sites)]
    self_attr = [(cust_biz[c] / avg_biz_c) if avg_biz_c > 0 else 1e-6 for c in range(num_customers)]
    self_attr = [max(s, 1e-6) for s in self_attr]

    # A[c][s][t] = site_attr[s] * weekly[t%7] / dist(c,s)^q
    A = [[[0.0] * T for _ in range(num_sites)] for _ in range(num_customers)]
    for ci in range(num_customers):
        for si in range(num_sites):
            d = get_c2s(cust_ids[ci], site_ids[si])
            if d <= 1e-9:
                d = 1e-6
            base = site_attr[si] / (d ** q)
            for t in range(T):
                A[ci][si][t] = base * weekly[t % 7]

    pop = cust_pop

    # Distance matrix between real sites (0-indexed by site index)
    dist_matrix = [[0.0] * num_sites for _ in range(num_sites)]
    for i in range(num_sites):
        for j in range(num_sites):
            dist_matrix[i][j] = get_s2s(site_ids[i], site_ids[j])

    # State
    events = {}  # t -> site_idx
    event_set = set()
    event_count_per_site = [0] * num_sites
    D_cache = {}   # t -> list[D per customer]
    att_cache = {} # t -> attendance

    def rebuild_D_for_period(t):
        D = list(self_attr)
        for t2 in event_set:
            if t2 != t and abs(t2 - t) <= epsilon:
                s2 = events[t2]
                a_t2 = [A[ci][s2][t2] for ci in range(num_customers)]
                for ci in range(num_customers):
                    D[ci] += a_t2[ci]
        D_cache[t] = D

    def recompute_att_for_period(t):
        s = events[t]
        D = D_cache[t]
        total_att = 0.0
        a_st = [A[ci][s][t] for ci in range(num_customers)]
        for ci in range(num_customers):
            d_val = D[ci]
            if d_val > 0:
                p = a_st[ci] / d_val
            else:
                p = 0.0
            if p > 1.0:
                p = 1.0
            total_att += pop[ci] * p
        att_cache[t] = min(total_att, site_capacity[s])

    def add_event(t, s):
        events[t] = s
        event_set.add(t)
        event_count_per_site[s] += 1
        a_new = [A[ci][s][t] for ci in range(num_customers)]
        for t2 in list(event_set):
            if abs(t2 - t) <= epsilon:
                if t2 == t:
                    rebuild_D_for_period(t)
                else:
                    Dc = D_cache[t2]
                    for ci in range(num_customers):
                        Dc[ci] += a_new[ci]
        for t2 in list(event_set):
            if abs(t2 - t) <= epsilon:
                recompute_att_for_period(t2)

    def remove_event(t):
        s = events[t]
        a_rem = [A[ci][s][t] for ci in range(num_customers)]
        for t2 in list(event_set):
            if t2 != t and abs(t2 - t) <= epsilon:
                Dc = D_cache[t2]
                for ci in range(num_customers):
                    Dc[ci] -= a_rem[ci]
        del events[t]
        event_set.discard(t)
        event_count_per_site[s] -= 1
        if t in D_cache:
            del D_cache[t]
        if t in att_cache:
            del att_cache[t]
        for t2 in list(event_set):
            if abs(t2 - t) <= epsilon:
                recompute_att_for_period(t2)

    def is_rest_feasible_set(es):
        st = sorted(es)
        n = len(st)
        for i in range(n - delta):
            if st[i + delta] - st[i] < delta + 2:
                return False
        return True

    def can_add(t, s):
        if t in event_set:
            return False
        if event_count_per_site[s] + 1 > site_max_events[s]:
            return False
        if not is_rest_feasible_set(event_set | {t}):
            return False
        return True

    def marginal_add(s, t):
        if not can_add(t, s):
            return -float('inf')
        a_new = [A[ci][s][t] for ci in range(num_customers)]
        aff = [t2 for t2 in event_set if abs(t2 - t) <= epsilon]
        delta_rev = 0.0
        for t2 in aff:
            s2 = events[t2]
            D = D_cache[t2]
            a_t2 = [A[ci][s2][t2] for ci in range(num_customers)]
            total_att_new = 0.0
            for ci in range(num_customers):
                d_new = D[ci] + a_new[ci]
                if d_new > 0:
                    p = a_t2[ci] / d_new
                else:
                    p = 0.0
                if p > 1.0:
                    p = 1.0
                total_att_new += pop[ci] * p
            new_att = min(total_att_new, site_capacity[s2])
            delta_rev += site_rev[s2] * (new_att - att_cache[t2])
        # New event's own D
        D_new = list(self_attr)
        for t2 in aff:
            s2 = events[t2]
            for ci in range(num_customers):
                D_new[ci] += A[ci][s2][t2]
        total_att_self = 0.0
        for ci in range(num_customers):
            d_val = D_new[ci] + a_new[ci]
            if d_val > 0:
                p = a_new[ci] / d_val
            else:
                p = 0.0
            if p > 1.0:
                p = 1.0
            total_att_self += pop[ci] * p
        new_att_self = min(total_att_self, site_capacity[s])
        delta_rev += site_rev[s] * new_att_self
        return delta_rev - site_fc[s]

    def marginal_remove(t):
        if t not in event_set:
            return -float('inf')
        s = events[t]
        a_rem = [A[ci][s][t] for ci in range(num_customers)]
        delta_rev = -site_rev[s] * att_cache[t]
        aff = [t2 for t2 in event_set if t2 != t and abs(t2 - t) <= epsilon]
        for t2 in aff:
            s2 = events[t2]
            D = D_cache[t2]
            a_t2 = [A[ci][s2][t2] for ci in range(num_customers)]
            total_att_new = 0.0
            for ci in range(num_customers):
                d_new = D[ci] - a_rem[ci]
                if d_new > 0:
                    p = a_t2[ci] / d_new
                else:
                    p = 0.0
                if p > 1.0:
                    p = 1.0
                total_att_new += pop[ci] * p
            new_att = min(total_att_new, site_capacity[s2])
            delta_rev += site_rev[s2] * (new_att - att_cache[t2])
        return delta_rev + site_fc[s]

    def compute_routing(events_dict):
        if num_sites == 0 or T == 0:
            return [0] * T, 0.0
        INF = float('inf')
        dp = [[INF] * num_sites for _ in range(T)]
        parent = [[-1] * num_sites for _ in range(T)]
        for s in range(num_sites):
            dp[0][s] = 0.0
        for t in range(T):
            if t in events_dict:
                fixed_s = events_dict[t]
                for s in range(num_sites):
                    if s != fixed_s:
                        dp[t][s] = INF
            if t == T - 1:
                break
            for s in range(num_sites):
                if dp[t][s] >= INF:
                    continue
                for s2 in range(num_sites):
                    d = dist_matrix[s][s2]
                    if d > L:
                        continue
                    nc = dp[t][s] + d
                    if nc < dp[t + 1][s2]:
                        dp[t + 1][s2] = nc
                        parent[t + 1][s2] = s
        best_s = -1
        best_cost = INF
        for s in range(num_sites):
            if dp[T - 1][s] < best_cost:
                best_cost = dp[T - 1][s]
                best_s = s
        if best_s == -1:
            return None, INF
        tour = [0] * T
        tour[T - 1] = best_s
        for t in range(T - 1, 0, -1):
            tour[t - 1] = parent[t][tour[t]]
            if tour[t - 1] == -1:
                return None, INF
        return tour, best_cost

    def compute_full_attendance(events_dict):
        event_periods = sorted(events_dict.keys())
        attendances = {}
        for t in event_periods:
            s = events_dict[t]
            D = list(self_attr)
            for t2 in event_periods:
                if abs(t2 - t) <= epsilon:
                    s2 = events_dict[t2]
                    for ci in range(num_customers):
                        D[ci] += A[ci][s2][t2]
            total_att = 0.0
            for ci in range(num_customers):
                if D[ci] > 0:
                    p = A[ci][s][t] / D[ci]
                else:
                    p = 0.0
                if p > 1.0:
                    p = 1.0
                total_att += pop[ci] * p
            attendances[t] = min(total_att, site_capacity[s])
        return attendances

    def full_objective(events_dict, travel_cost):
        attendances = compute_full_attendance(events_dict)
        rev = sum(site_rev[events_dict[t]] * attendances[t] for t in events_dict)
        fc = sum(site_fc[events_dict[t]] for t in events_dict)
        return rev - fc - c_travel * travel_cost, attendances

    def build_solution_dict(events_dict, tour, attendances, obj):
        sol = {
            "objective_value": obj,
            "tour": {},
            "events": {},
            "event_sites": {},
            "attendance": {}
        }
        for t in range(T):
            si = tour[t]
            sol["tour"][str(t + 1)] = site_ids[si]
            if t in events_dict:
                sol["events"][str(t + 1)] = 1
                sol["event_sites"][str(t + 1)] = site_ids[events_dict[t]]
                sol["attendance"][f"{site_ids[events_dict[t]]}_{t + 1}"] = attendances[t]
            else:
                sol["events"][str(t + 1)] = 0
        return sol

    best_obj = -float('inf')
    best_events = {}
    best_tour = [0] * T if num_sites > 0 else []
    best_attendances = {}

    def update_best():
        nonlocal best_obj, best_events, best_tour, best_attendances
        tour, travel = compute_routing(events)
        if tour is None:
            return False
        obj, attendances = full_objective(events, travel)
        if obj > best_obj:
            best_obj = obj
            best_events = dict(events)
            best_tour = tour
            best_attendances = attendances
            if logger:
                sol = build_solution_dict(best_events, best_tour, best_attendances, obj)
                logger.log_solution(obj, sol)
            return True
        return False

    # Initial empty solution
    update_best()

    # Greedy add
    while time.time() < deadline - 8:
        best_delta = 1e-4
        best_st = None
        for s in range(num_sites):
            for t in range(T):
                d = marginal_add(s, t)
                if d > best_delta:
                    best_delta = d
                    best_st = (s, t)
        if best_st is None:
            break
        s, t = best_st
        add_event(t, s)
        update_best()

    # Local search
    improved = True
    iter_count = 0
    while improved and time.time() < deadline - 3:
        improved = False
        iter_count += 1
        # Try removes
        for t in list(event_set):
            if time.time() >= deadline - 3:
                break
            mp = marginal_remove(t)
            if mp > 1e-4:
                remove_event(t)
                if update_best():
                    improved = True
        # Try adds again
        if time.time() < deadline - 3:
            for s in range(num_sites):
                for t in range(T):
                    if time.time() >= deadline - 3:
                        break
                    mp = marginal_add(s, t)
                    if mp > 1e-4:
                        add_event(t, s)
                        if update_best():
                            improved = True
                if time.time() >= deadline - 3:
                    break
        # Try site swaps at same period
        if time.time() < deadline - 3:
            for t in list(event_set):
                if time.time() >= deadline - 3:
                    break
                old_s = events[t]
                mp_remove = marginal_remove(t)
                remove_event(t)
                best_new_s = None
                best_add = 1e-4
                for new_s in range(num_sites):
                    mp_add = marginal_add(new_s, t)
                    if mp_remove + mp_add > best_add:
                        best_add = mp_remove + mp_add
                        best_new_s = new_s
                if best_new_s is not None and best_new_s != old_s:
                    add_event(t, best_new_s)
                    if update_best():
                        improved = True
                else:
                    add_event(t, old_s)
        # Try period moves
        if time.time() < deadline - 3:
            for t in list(event_set):
                if time.time() >= deadline - 3:
                    break
                old_s = events[t]
                mp_remove = marginal_remove(t)
                remove_event(t)
                best_new_t = None
                best_new_s = None
                best_gain = 1e-4
                for new_t in range(T):
                    if new_t == t:
                        continue
                    if not can_add(new_t, old_s):
                        continue
                    mp_add = marginal_add(old_s, new_t)
                    if mp_remove + mp_add > best_gain:
                        best_gain = mp_remove + mp_add
                        best_new_t = new_t
                        best_new_s = old_s
                if best_new_t is not None:
                    add_event(best_new_t, best_new_s)
                    if update_best():
                        improved = True
                else:
                    add_event(t, old_s)

    # Final: ensure best solution is logged and written
    if best_events:
        tour, travel = compute_routing(best_events)
        if tour is not None:
            obj, attendances = full_objective(best_events, travel)
            best_obj = obj
            best_tour = tour
            best_attendances = attendances
    else:
        best_obj = 0.0
        best_attendances = {}

    sol = build_solution_dict(best_events, best_tour, best_attendances, best_obj)
    with open(args.solution_path, 'w') as f:
        json.dump(sol, f, indent=2)

    if logger:
        logger.log_solution(best_obj, sol)


if __name__ == "__main__":
    main()
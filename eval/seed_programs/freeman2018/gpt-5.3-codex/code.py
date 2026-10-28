import argparse
import json
import math
import random
import time
from typing import List, Tuple

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def get_keyed_distance(dct, a, b, default=None):
    k1 = f"{a}_{b}"
    if k1 in dct:
        return float(dct[k1])
    k2 = f"{b}_{a}"
    if k2 in dct:
        return float(dct[k2])
    return default


def build_distance_matrix(data, S):
    dct = data["site_to_site_distances"]
    # matrix with dummy node 0 and sites 1..S
    D = np.full((S + 1, S + 1), np.inf, dtype=float)
    for i in range(S + 1):
        D[i, i] = 0.0
    for i in range(S + 1):
        for j in range(S + 1):
            if i == j:
                continue
            v = get_keyed_distance(dct, i, j, default=None)
            if v is not None:
                D[i, j] = v
    return D


def build_customer_site_matrix(data, C, S):
    dct = data["customer_to_site_distances"]
    M = np.full((C, S), np.inf, dtype=float)
    for i in range(1, C + 1):
        for s in range(1, S + 1):
            v = get_keyed_distance(dct, i, s, default=None)
            if v is None:
                raise ValueError(f"Missing customer_to_site distance for {i}_{s}")
            M[i - 1, s - 1] = float(v)
    return M


def compute_static_arrays(data):
    params = data["parameters"]
    S = data["num_sites"]
    C = data["num_customers"]
    T = data["num_periods"]

    weekly = params["weekly_attraction_cycle"]
    q = float(params["q"])
    avg_c = float(params["avg_businesses_customers"])
    avg_s = float(params["avg_businesses_sites"])

    sites = data["sites"]
    customers = data["customers"]

    site_capacity = np.array([int(s["capacity"]) for s in sites], dtype=float)
    site_rev = np.array([float(s["revenue_per_customer"]) for s in sites], dtype=float)
    site_fixed = np.array([float(s["fixed_cost"]) for s in sites], dtype=float)
    site_max_events = np.array([int(s["max_events"]) for s in sites], dtype=int)
    site_business = np.array([float(s["num_businesses"]) for s in sites], dtype=float)

    cust_pop = np.array([float(c["population"]) for c in customers], dtype=float)
    cust_business = np.array([float(c["num_businesses"]) for c in customers], dtype=float)

    # Self attraction (strictly positive)
    self_attr = cust_business / max(avg_c, 1e-6)
    self_attr = np.maximum(self_attr, 1e-6)

    # Site-period attraction coefficient
    site_base_attr = site_business / max(avg_s, 1e-6)
    weekly_factor = np.array([float(weekly[t % 7]) for t in range(T)], dtype=float)
    site_period_coeff = np.outer(site_base_attr, weekly_factor)  # S x T

    # Customer-site distance matrix
    cs_dist = build_customer_site_matrix(data, C, S)
    cs_dist = np.maximum(cs_dist, 1e-3)
    base_attract = 1.0 / np.power(cs_dist, q)  # C x S

    # Full event attraction if site s hosts at period t: C x S x T (implicit via base_attract * coeff)
    return (
        S, C, T, params,
        site_capacity, site_rev, site_fixed, site_max_events,
        cust_pop, self_attr, site_period_coeff, base_attract
    )


def evaluate_solution(
    tour: np.ndarray, events: np.ndarray,
    site_capacity: np.ndarray, site_rev: np.ndarray, site_fixed: np.ndarray,
    cust_pop: np.ndarray, self_attr: np.ndarray,
    site_period_coeff: np.ndarray, base_attract: np.ndarray,
    eps: int, travel_cost_per_dist: float, dist_mat: np.ndarray
) -> Tuple[float, np.ndarray]:
    """
    tour: (T,) site indices 0..S-1
    events: (T,) 0/1
    returns objective, attendance_matrix (S x T)
    """
    T = len(tour)
    S = len(site_capacity)

    idx_t = np.arange(T)
    coeff_t = site_period_coeff[tour, idx_t]  # (T,)
    # event attraction matrix E: C x T
    E = base_attract[:, tour] * coeff_t[np.newaxis, :]
    E = E * events[np.newaxis, :]

    # denominator with epsilon shadow window
    prefix = np.cumsum(E, axis=1)
    denom = np.empty_like(E)
    for t in range(T):
        l = max(0, t - eps)
        r = min(T - 1, t + eps)
        ws = prefix[:, r] - (prefix[:, l - 1] if l > 0 else 0.0)
        denom[:, t] = self_attr + ws

    shares = np.divide(E, denom, out=np.zeros_like(E), where=denom > 0)
    demand = (cust_pop[:, None] * shares).sum(axis=0)  # (T,)

    caps = site_capacity[tour]
    attendance_t = np.minimum(demand, caps) * events

    revenue = (site_rev[tour] * attendance_t).sum()
    fixed = (site_fixed[tour] * events).sum()

    tour1 = tour + 1  # convert to 1..S for dist matrix with dummy 0
    travel_dist = dist_mat[0, tour1[0]] + np.sum(dist_mat[tour1[:-1], tour1[1:]])
    travel = travel_cost_per_dist * travel_dist

    obj = float(revenue - fixed - travel)

    attendance_matrix = np.zeros((S, T), dtype=float)
    ev_periods = np.where(events > 0.5)[0]
    attendance_matrix[tour[ev_periods], ev_periods] = attendance_t[ev_periods]

    return obj, attendance_matrix


def is_feasible(
    tour: np.ndarray, events: np.ndarray,
    L: float, delta: int, site_max_events: np.ndarray, dist_mat: np.ndarray
) -> bool:
    T = len(tour)
    S = len(site_max_events)

    # travel feasibility
    tour1 = tour + 1
    if dist_mat[0, tour1[0]] > L + 1e-9:
        return False
    if T > 1:
        if np.any(dist_mat[tour1[:-1], tour1[1:]] > L + 1e-9):
            return False

    # site max events
    counts = np.zeros(S, dtype=int)
    ev_idx = np.where(events > 0.5)[0]
    for t in ev_idx:
        counts[tour[t]] += 1
    if np.any(counts > site_max_events):
        return False

    # rest-day window: periods t..t+delta+1 <= delta
    # t in 1..T-delta-1 (1-indexed) => 0..T-delta-2 (0-indexed)
    w = delta + 2
    if T - delta - 1 > 0:
        pref = np.cumsum(events)
        for start in range(0, T - delta - 1):
            end = start + w - 1
            s = pref[end] - (pref[start - 1] if start > 0 else 0)
            if s > delta + 1e-9:
                return False

    return True


def build_initial_mip(
    T, S, L, delta, c_travel,
    site_max_events, approx_profit, dist_mat, time_limit_sec
):
    model = gp.Model("tour_surrogate")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = max(1, int(time_limit_sec))
    model.Params.MIPFocus = 1
    model.Params.Threads = 1

    x = model.addVars(S, T, vtype=GRB.BINARY, name="x")
    y = model.addVars(S, T, vtype=GRB.BINARY, name="y")

    # arc vars for transitions t>=1 (0-based)
    arcs = {}
    for t in range(1, T):
        for i in range(S):
            for j in range(S):
                d = dist_mat[i + 1, j + 1]
                if d <= L + 1e-9:
                    arcs[(i, j, t)] = model.addVar(vtype=GRB.BINARY, name=f"a_{i}_{j}_{t}")

    model.update()

    # one visited site each period
    for t in range(T):
        model.addConstr(gp.quicksum(x[s, t] for s in range(S)) == 1)

    # first period reachable from dummy
    for s in range(S):
        if dist_mat[0, s + 1] > L + 1e-9:
            model.addConstr(x[s, 0] == 0)

    # flow linking arcs and x
    for t in range(1, T):
        for j in range(S):
            in_arcs = [arcs[(i, j, t)] for i in range(S) if (i, j, t) in arcs]
            if in_arcs:
                model.addConstr(gp.quicksum(in_arcs) == x[j, t])
            else:
                model.addConstr(x[j, t] == 0)

        for i in range(S):
            out_arcs = [arcs[(i, j, t)] for j in range(S) if (i, j, t) in arcs]
            if out_arcs:
                model.addConstr(gp.quicksum(out_arcs) == x[i, t - 1])
            else:
                model.addConstr(x[i, t - 1] == 0)

    # event only at visited site
    for s in range(S):
        for t in range(T):
            model.addConstr(y[s, t] <= x[s, t])

    # max events per site
    for s in range(S):
        model.addConstr(gp.quicksum(y[s, t] for t in range(T)) <= int(site_max_events[s]))

    # rest windows
    w = delta + 2
    if T - delta - 1 > 0:
        for start in range(0, T - delta - 1):
            model.addConstr(
                gp.quicksum(y[s, t] for s in range(S) for t in range(start, start + w)) <= delta
            )

    # objective
    obj = gp.LinExpr()
    for s in range(S):
        for t in range(T):
            obj += float(approx_profit[s, t]) * y[s, t]
    for (i, j, t), var in arcs.items():
        obj += -c_travel * float(dist_mat[i + 1, j + 1]) * var
    # first-period travel from dummy
    for s in range(S):
        obj += -c_travel * float(dist_mat[0, s + 1]) * x[s, 0]

    model.setObjective(obj, GRB.MAXIMIZE)
    model.optimize()

    if model.SolCount == 0:
        return None, None

    tour = np.zeros(T, dtype=int)
    events = np.zeros(T, dtype=float)
    for t in range(T):
        vals = np.array([x[s, t].X for s in range(S)])
        s_star = int(np.argmax(vals))
        tour[t] = s_star
        events[t] = 1.0 if y[s_star, t].X > 0.5 else 0.0

    return tour, events


def heuristic_fallback(T, S, L, dist_mat, site_max_events, delta, approx_profit):
    # choose best start reachable from dummy, then greedy transitions
    feasible_start = [s for s in range(S) if dist_mat[0, s + 1] <= L + 1e-9]
    if not feasible_start:
        feasible_start = [0]
    tour = np.zeros(T, dtype=int)
    start = max(feasible_start, key=lambda s: approx_profit[s, 0])
    tour[0] = start
    for t in range(1, T):
        prev = tour[t - 1]
        cand = [j for j in range(S) if dist_mat[prev + 1, j + 1] <= L + 1e-9]
        if not cand:
            cand = [prev]
        tour[t] = max(cand, key=lambda s: approx_profit[s, t])

    events = np.zeros(T, dtype=float)
    # greedy event activation respecting constraints
    order = list(range(T))
    order.sort(key=lambda t: approx_profit[tour[t], t], reverse=True)
    for t in order:
        if approx_profit[tour[t], t] <= 0:
            continue
        events[t] = 1.0
        if not is_feasible(tour, events, L, delta, site_max_events, dist_mat):
            events[t] = 0.0

    return tour, events


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    start_time = time.time()
    random.seed(42)
    np.random.seed(42)

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    (
        S, C, T, params,
        site_capacity, site_rev, site_fixed, site_max_events,
        cust_pop, self_attr, site_period_coeff, base_attract
    ) = compute_static_arrays(data)

    L = float(params["L"])
    delta = int(params["delta"])
    eps = int(params["epsilon"])
    c_travel = float(params["c"])

    dist_mat = build_distance_matrix(data, S)

    # surrogate profit for initialization (single-event approximation)
    approx_profit = np.zeros((S, T), dtype=float)
    for s in range(S):
        for t in range(T):
            attr = base_attract[:, s] * site_period_coeff[s, t]
            denom = self_attr + attr
            share = np.divide(attr, denom, out=np.zeros_like(attr), where=denom > 0)
            dem = float(np.sum(cust_pop * share))
            dem = min(dem, site_capacity[s])
            approx_profit[s, t] = site_rev[s] * dem - site_fixed[s]

    # Initial solution via MIP surrogate
    remaining = max(1, args.time_limit - int(time.time() - start_time))
    mip_budget = max(1, int(0.45 * remaining))
    tour, events = build_initial_mip(
        T, S, L, delta, c_travel,
        site_max_events, approx_profit, dist_mat, mip_budget
    )
    if tour is None:
        tour, events = heuristic_fallback(T, S, L, dist_mat, site_max_events, delta, approx_profit)

    if not is_feasible(tour, events, L, delta, site_max_events, dist_mat):
        # hard fallback: stay at one feasible site, no events
        feasible_start = [s for s in range(S) if dist_mat[0, s + 1] <= L + 1e-9]
        s0 = feasible_start[0] if feasible_start else 0
        tour = np.array([s0] * T, dtype=int)
        events = np.zeros(T, dtype=float)

    best_obj, best_att = evaluate_solution(
        tour, events, site_capacity, site_rev, site_fixed,
        cust_pop, self_attr, site_period_coeff, base_attract,
        eps, c_travel, dist_mat
    )
    best_tour = tour.copy()
    best_events = events.copy()
    if logger:
        logger.log(best_obj)

    # Local search
    while time.time() - start_time < args.time_limit - 0.2:
        improved = False

        # 1) Toggle events
        for t in np.random.permutation(T):
            if time.time() - start_time >= args.time_limit - 0.2:
                break
            cand_events = best_events.copy()
            cand_events[t] = 1.0 - cand_events[t]
            if not is_feasible(best_tour, cand_events, L, delta, site_max_events, dist_mat):
                continue
            obj, att = evaluate_solution(
                best_tour, cand_events, site_capacity, site_rev, site_fixed,
                cust_pop, self_attr, site_period_coeff, base_attract,
                eps, c_travel, dist_mat
            )
            if obj > best_obj + 1e-8:
                best_obj, best_att = obj, att
                best_events = cand_events
                improved = True
                if logger:
                    logger.log(best_obj)
                break
        if improved:
            continue

        # 2) Change visited site in one period
        for t in np.random.permutation(T):
            if time.time() - start_time >= args.time_limit - 0.2:
                break
            cur = best_tour[t]

            # feasible candidates by adjacency
            cand_sites = []
            for s in range(S):
                if s == cur:
                    continue
                ok_prev = True
                ok_next = True
                if t == 0:
                    ok_prev = dist_mat[0, s + 1] <= L + 1e-9
                else:
                    ok_prev = dist_mat[best_tour[t - 1] + 1, s + 1] <= L + 1e-9
                if t < T - 1:
                    ok_next = dist_mat[s + 1, best_tour[t + 1] + 1] <= L + 1e-9
                if ok_prev and ok_next:
                    cand_sites.append(s)

            if not cand_sites:
                continue

            # prioritize promising candidates
            cand_sites.sort(key=lambda s: approx_profit[s, t], reverse=True)
            cand_sites = cand_sites[:8]

            for s in cand_sites:
                cand_tour = best_tour.copy()
                cand_tour[t] = s
                if not is_feasible(cand_tour, best_events, L, delta, site_max_events, dist_mat):
                    continue
                obj, att = evaluate_solution(
                    cand_tour, best_events, site_capacity, site_rev, site_fixed,
                    cust_pop, self_attr, site_period_coeff, base_attract,
                    eps, c_travel, dist_mat
                )
                if obj > best_obj + 1e-8:
                    best_obj, best_att = obj, att
                    best_tour = cand_tour
                    improved = True
                    if logger:
                        logger.log(best_obj)
                    break
            if improved:
                break

        if not improved:
            # small random perturbation to escape local optimum (if time remains)
            if time.time() - start_time < args.time_limit - 1.0:
                t = random.randrange(T)
                cand_events = best_events.copy()
                cand_events[t] = 1.0 - cand_events[t]
                if is_feasible(best_tour, cand_events, L, delta, site_max_events, dist_mat):
                    obj, att = evaluate_solution(
                        best_tour, cand_events, site_capacity, site_rev, site_fixed,
                        cust_pop, self_attr, site_period_coeff, base_attract,
                        eps, c_travel, dist_mat
                    )
                    if obj > best_obj:
                        best_obj, best_att = obj, att
                        best_events = cand_events
                        if logger:
                            logger.log(best_obj)
            else:
                break

    # Build output
    tour_out = {str(t + 1): int(best_tour[t] + 1) for t in range(T)}
    events_out = {str(t + 1): int(round(best_events[t])) for t in range(T)}
    event_sites_out = {
        str(t + 1): int(best_tour[t] + 1) if best_events[t] > 0.5 else 0
        for t in range(T)
    }

    attendance_out = {}
    for s in range(S):
        for t in range(T):
            attendance_out[f"{s+1}_{t+1}"] = float(best_att[s, t])

    sol = {
        "objective_value": float(best_obj),
        "tour": tour_out,
        "events": events_out,
        "event_sites": event_sites_out,
        "attendance": attendance_out,
    }

    with open(args.solution_path, "w") as f:
        json.dump(sol, f, indent=2)


if __name__ == "__main__":
    main()
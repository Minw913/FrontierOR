import argparse
import json
import time
from copy import deepcopy

from solution_logger import SolutionLogger


def clip(x, lo, hi):
    return max(lo, min(hi, x))


def eval_sequence(seq, dist, tw_lb, tw_ub, depot_lb, depot_ub, end_idx):
    """
    Evaluate feasibility, earliest service times, and travel cost for route:
    0 -> seq... -> end_idx
    """
    nodes = [0] + seq + [end_idx]
    times = {}
    t = depot_lb
    times[0] = t
    cost = 0.0
    prev = 0

    for node in nodes[1:]:
        cost += dist[prev][node]
        arr = t + dist[prev][node]
        lb = depot_lb if node == end_idx else tw_lb[node]
        ub = depot_ub if node == end_idx else tw_ub[node]
        t = max(arr, lb)
        if t > ub + 1e-9:
            return False, None, None
        times[node] = t
        prev = node

    return True, times, cost


def build_initial_endogenous_windows(customers, depot_tw, dist, end_idx, global_width):
    """
    Build customer endogenous windows (fixed across scenarios), trying to keep them
    depot-feasible for singleton routes.
    """
    depot_lb, depot_ub = depot_tw
    tw_start = {}
    tw_end = {}
    widths = {}

    # exogenous by customer id
    exo = {}
    for c in customers:
        cid = c["id"]
        a, b = c["exogenous_time_window"]
        w = c.get("endogenous_time_window_width", global_width)
        exo[cid] = (a, b)
        widths[cid] = w

    for cid in exo:
        a, b = exo[cid]
        w = widths[cid]
        s_lo = a
        s_hi = b - w

        # Depot-feasible singleton service interval [L, U]
        L = max(a, depot_lb + dist[0][cid])
        U = min(b, depot_ub - dist[cid][end_idx])

        # Start from centered candidate around [L, U]
        if L <= U:
            s = (L + U - w) / 2.0
        else:
            # fallback to exogenous center
            s = (a + b - w) / 2.0

        s = clip(s, s_lo, s_hi)

        # Ensure overlap with [L, U] when possible
        if L <= U:
            if s > U:
                s = clip(U, s_lo, s_hi)
            if s + w < L:
                s = clip(L, s_lo, s_hi)

        tw_start[cid] = s
        tw_end[cid] = s + w

    return tw_start, tw_end, widths


def build_singleton_routes(customer_ids, demands_by_cust, dist, tw_lb, tw_ub, depot_lb, depot_ub, end_idx):
    routes = []
    for cid in customer_ids:
        feas, times, cost = eval_sequence([cid], dist, tw_lb, tw_ub, depot_lb, depot_ub, end_idx)
        if not feas:
            return None  # caller handles fallback/infeasible
        routes.append(
            {
                "custs": [cid],
                "load": demands_by_cust[cid],
                "cost": cost,
                "times": times,
            }
        )
    return routes


def scenario_cost(routes):
    return sum(r["cost"] for r in routes)


def improve_routes_pairwise_merge(
    routes,
    demands_by_cust,
    capacity,
    dist,
    tw_lb,
    tw_ub,
    depot_lb,
    depot_ub,
    end_idx,
    deadline,
    on_improvement=None,
):
    """
    Greedy best-improving concatenation merge with route reversals.
    """
    if routes is None:
        return None

    while time.time() < deadline:
        m = len(routes)
        best = None
        best_saving = 1e-9

        for i in range(m):
            if time.time() >= deadline:
                break
            ri = routes[i]
            for j in range(i + 1, m):
                rj = routes[j]
                if ri["load"] + rj["load"] > capacity:
                    continue

                base = ri["cost"] + rj["cost"]

                for oi in (1, -1):
                    ci = ri["custs"] if oi == 1 else list(reversed(ri["custs"]))
                    for oj in (1, -1):
                        cj = rj["custs"] if oj == 1 else list(reversed(rj["custs"]))
                        seq = ci + cj

                        feas, times, new_cost = eval_sequence(seq, dist, tw_lb, tw_ub, depot_lb, depot_ub, end_idx)
                        if not feas:
                            continue

                        saving = base - new_cost
                        if saving > best_saving:
                            best_saving = saving
                            best = (i, j, seq, times, new_cost)

        if best is None:
            break

        i, j, seq, times, new_cost = best
        # merge i and j
        if i > j:
            i, j = j, i
        ri = routes[i]
        rj = routes[j]
        new_route = {
            "custs": seq,
            "load": ri["load"] + rj["load"],
            "cost": new_cost,
            "times": times,
        }

        routes.pop(j)
        routes.pop(i)
        routes.append(new_route)

        if on_improvement:
            on_improvement(best_saving)

    return routes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    t0 = time.time()
    deadline = t0 + max(1, args.time_limit) - 1e-3

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    n = inst["num_customers"]
    end_idx = n + 1
    dist = inst["distance_matrix"]

    depot_tw = inst["depot"]["exogenous_time_window"]
    depot_lb, depot_ub = depot_tw
    capacity = inst["vehicle_capacity"]
    global_w = inst.get("endogenous_time_window_width", 0.0)

    customers = inst["customers"]
    customer_ids = sorted([c["id"] for c in customers])

    # Build fixed endogenous windows
    tw_start, tw_end, widths = build_initial_endogenous_windows(customers, depot_tw, dist, end_idx, global_w)

    # Time window arrays for fast access (customer only; depot handled separately)
    max_idx = end_idx
    tw_lb = [0.0] * (max_idx + 1)
    tw_ub = [0.0] * (max_idx + 1)

    # default from exogenous (for safety)
    exo_map = {c["id"]: c["exogenous_time_window"] for c in customers}
    for cid in customer_ids:
        a, b = exo_map[cid]
        tw_lb[cid] = max(a, tw_start[cid])
        tw_ub[cid] = min(b, tw_end[cid])

    scenarios = inst["scenarios"]

    # Build initial feasible solution: singleton routes for all scenarios
    routes_by_s = {}
    scen_prob = {}
    scen_cost = {}
    total_obj = 0.0

    # Customer id order for scenario demand vectors
    # (schema says ordered by customer id)
    sorted_ids_for_demands = sorted(customer_ids)

    for s in scenarios:
        sid = s["scenario_id"]
        p = s["probability"]
        scen_prob[sid] = p
        dem_vec = s["demands"]

        demands_by_cust = {}
        for pos, cid in enumerate(sorted_ids_for_demands):
            demands_by_cust[cid] = dem_vec[pos]

        # Initial singleton routes (always capacity-feasible since each demand <= Q by schema)
        routes = build_singleton_routes(
            customer_ids,
            demands_by_cust,
            dist,
            tw_lb,
            tw_ub,
            depot_lb,
            depot_ub,
            end_idx,
        )

        # If somehow infeasible, fallback to direct exogenous intersection slack attempt
        if routes is None:
            routes = []
            for cid in customer_ids:
                # emergency route with customer exogenous only (still must satisfy endogenous by schema,
                # but we keep same tw_lb/tw_ub; if this fails, instance likely infeasible with chosen windows)
                feas, times, cost = eval_sequence([cid], dist, tw_lb, tw_ub, depot_lb, depot_ub, end_idx)
                if not feas:
                    # last-resort: keep route structure with zeroed times (avoid crash)
                    times = {0: depot_lb, cid: tw_lb[cid], end_idx: depot_lb}
                    cost = dist[0][cid] + dist[cid][end_idx]
                routes.append({"custs": [cid], "load": demands_by_cust[cid], "cost": cost, "times": times})

        routes_by_s[sid] = {
            "routes": routes,
            "demands": demands_by_cust,
        }
        cst = scenario_cost(routes)
        scen_cost[sid] = cst
        total_obj += p * cst

    best_obj = total_obj
    if logger:
        logger.log(best_obj)

    # Improve each scenario independently with greedy merges
    # Prioritize high-probability scenarios
    ordered_scenarios = sorted(scenarios, key=lambda x: -x["probability"])

    for s in ordered_scenarios:
        if time.time() >= deadline:
            break

        sid = s["scenario_id"]
        p = scen_prob[sid]
        sr = routes_by_s[sid]["routes"]
        demands_by_cust = routes_by_s[sid]["demands"]

        def on_improvement(saving):
            nonlocal best_obj, total_obj
            total_obj -= p * saving
            if total_obj + 1e-9 < best_obj:
                best_obj = total_obj
                if logger:
                    logger.log(best_obj)

        improved = improve_routes_pairwise_merge(
            sr,
            demands_by_cust,
            capacity,
            dist,
            tw_lb,
            tw_ub,
            depot_lb,
            depot_ub,
            end_idx,
            deadline,
            on_improvement=on_improvement,
        )
        routes_by_s[sid]["routes"] = improved if improved is not None else sr
        scen_cost[sid] = scenario_cost(routes_by_s[sid]["routes"])

    # Recompute final objective exactly
    final_obj = 0.0
    for s in scenarios:
        sid = s["scenario_id"]
        final_obj += scen_prob[sid] * scenario_cost(routes_by_s[sid]["routes"])

    # Build output
    out_time_windows = {}
    for cid in customer_ids:
        out_time_windows[str(cid)] = [float(tw_start[cid]), float(tw_end[cid])]

    out_routes = {}
    for s in scenarios:
        sid = s["scenario_id"]
        rlist = []
        for r in routes_by_s[sid]["routes"]:
            seq = r["custs"]
            feas, times, cost = eval_sequence(seq, dist, tw_lb, tw_ub, depot_lb, depot_ub, end_idx)
            if not feas:
                # keep stored values if re-eval fails for numerical reasons
                times = r["times"]
                cost = r["cost"]

            route_nodes = [0] + seq + [end_idx]
            arr_dict = {str(node): float(times[node]) for node in route_nodes}
            rlist.append(
                {
                    "route": route_nodes,
                    "arrival_times": arr_dict,
                    "cost": float(cost),
                }
            )
        out_routes[str(sid)] = rlist

    solution = {
        "objective_value": float(final_obj),
        "time_windows": out_time_windows,
        "routes": out_routes,
    }

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()
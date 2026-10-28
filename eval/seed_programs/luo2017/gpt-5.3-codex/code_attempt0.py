import argparse
import json
import time
from copy import deepcopy

from solution_logger import SolutionLogger


def solve_instance(instance, time_limit, logger=None):
    start_time = time.time()

    n = instance["num_customers"]
    Q = float(instance["vehicle_capacity"])
    a = float(instance["cost_function"]["a"])
    b = float(instance["cost_function"]["b"])

    depot = instance["depot"]
    depot_id = int(depot["id"])
    virtual_depot_id = n + 1

    dist_mat = instance["distance_matrix"]

    customers = instance["customers"]
    cust_ids = [int(c["id"]) for c in customers]

    demand = {int(c["id"]): float(c["demand"]) for c in customers}
    service = {int(c["id"]): float(c["service_time"]) for c in customers}
    tw_early = {int(c["id"]): float(c["time_window"][0]) for c in customers}
    tw_late = {int(c["id"]): float(c["time_window"][1]) for c in customers}

    depot_early = float(depot["time_window"][0])
    depot_late = min(float(depot["time_window"][1]), float(instance["scheduling_horizon"]))
    depot_service = float(depot.get("service_time", 0))

    def idx(node):
        if node == virtual_depot_id:
            return depot_id
        return node

    def dist(i, j):
        return float(dist_mat[idx(i)][idx(j)])

    def route_cost(seq, qty_map):
        if not seq:
            return 0.0
        load = sum(qty_map[c] for c in seq)
        cost = 0.0
        prev = depot_id
        for c in seq:
            cost += dist(prev, c) * (a * load + b)
            load -= qty_map[c]
            prev = c
        cost += dist(prev, depot_id) * (a * load + b)  # typically load=0 on return
        return cost

    def route_feasible_time(seq):
        # Earliest-departure schedule check (sufficient for this setting)
        t = depot_early + depot_service
        prev = depot_id
        for c in seq:
            t += dist(prev, c)  # travel time assumed equal to distance
            if t < tw_early[c]:
                t = tw_early[c]
            if t > tw_late[c] + 1e-9:
                return False
            t += service[c]
            prev = c
        t += dist(prev, depot_id)
        return t <= depot_late + 1e-9

    def build_solution_dict(routes):
        sol_routes = []
        total_obj = 0.0

        vehicle_id = 0
        for r in routes:
            seq = r["seq"]
            if not seq:
                continue

            qty_map = r["qty"]
            rcost = route_cost(seq, qty_map)
            total_obj += rcost

            route_nodes = [depot_id] + seq + [virtual_depot_id]
            deliveries = {str(c): float(qty_map[c]) for c in seq}

            edges = []
            load = sum(qty_map[c] for c in seq)
            prev = depot_id
            for c in seq:
                edges.append(
                    {
                        "from": int(prev),
                        "to": int(c),
                        "load_weight": float(load),
                    }
                )
                load -= qty_map[c]
                prev = c
            edges.append(
                {
                    "from": int(prev),
                    "to": int(virtual_depot_id),
                    "load_weight": float(load),
                }
            )

            sol_routes.append(
                {
                    "vehicle": int(vehicle_id),
                    "route": [int(x) for x in route_nodes],
                    "deliveries": deliveries,
                    "edges": edges,
                }
            )
            vehicle_id += 1

        return {"objective_value": float(total_obj), "routes": sol_routes}

    def total_cost(routes):
        return sum(route_cost(r["seq"], r["qty"]) for r in routes if r["seq"])

    # ----------------------------
    # Build split-delivery tasks
    # ----------------------------
    tasks = []
    for c in cust_ids:
        d = demand[c]
        while d > 1e-9:
            q = min(Q, d)
            tasks.append((c, q))
            d -= q

    # Harder tasks first
    tasks.sort(key=lambda x: (tw_late[x[0]], tw_early[x[0]], -x[1]))

    # ----------------------------
    # Constructive insertion heuristic
    # ----------------------------
    routes = []

    def current_time_exceeded():
        return (time.time() - start_time) >= time_limit

    for c, q in tasks:
        if current_time_exceeded():
            break

        # Option 1: standalone route
        standalone_seq = [c]
        standalone_qty = {c: q}
        standalone_feas = route_feasible_time(standalone_seq)
        standalone_cost = route_cost(standalone_seq, standalone_qty)

        best_choice = ("new", None, None, standalone_cost if standalone_feas else float("inf"))

        # Option 2: insert into existing routes
        for ridx, r in enumerate(routes):
            if c in r["qty"]:
                continue
            if r["load"] + q > Q + 1e-9:
                continue

            old_seq = r["seq"]
            old_qty = r["qty"]
            old_cost = route_cost(old_seq, old_qty)

            for pos in range(len(old_seq) + 1):
                new_seq = old_seq[:pos] + [c] + old_seq[pos:]
                if not route_feasible_time(new_seq):
                    continue

                new_qty = dict(old_qty)
                new_qty[c] = q
                new_cost = route_cost(new_seq, new_qty)
                delta = new_cost - old_cost

                if new_cost < best_choice[3] and delta < float("inf"):
                    best_choice = ("insert", ridx, pos, new_cost)

        if best_choice[0] == "new" and not standalone_feas:
            # If infeasible even standalone, still place it standalone to keep demand satisfied.
            # (Instance is likely infeasible in time windows.)
            routes.append({"seq": [c], "qty": {c: q}, "load": q})
        elif best_choice[0] == "new":
            routes.append({"seq": [c], "qty": {c: q}, "load": q})
        else:
            _, ridx, pos, _ = best_choice
            r = routes[ridx]
            r["seq"] = r["seq"][:pos] + [c] + r["seq"][pos:]
            r["qty"][c] = q
            r["load"] += q

    # If time ended before assigning all tasks, assign remainder standalone quickly
    assigned_total = sum(sum(rt["qty"].values()) for rt in routes)
    required_total = sum(demand.values())
    if assigned_total + 1e-6 < required_total:
        # Identify unassigned by tracking counts
        assigned_per_customer = {c: 0.0 for c in cust_ids}
        for rt in routes:
            for c, q in rt["qty"].items():
                assigned_per_customer[c] += q
        for c in cust_ids:
            rem = demand[c] - assigned_per_customer[c]
            while rem > 1e-9:
                q = min(Q, rem)
                routes.append({"seq": [c], "qty": {c: q}, "load": q})
                rem -= q

    # ----------------------------
    # Local improvement
    # ----------------------------
    best_routes = deepcopy(routes)
    best_obj = total_cost(best_routes)
    best_solution = build_solution_dict(best_routes)
    if logger:
        logger.log_solution(best_obj, best_solution)

    # Intra-route relocate
    def improve_route_order(r):
        improved = True
        while improved and not current_time_exceeded():
            improved = False
            seq = r["seq"]
            if len(seq) <= 2:
                break
            base_cost = route_cost(seq, r["qty"])
            best_seq = seq
            best_cost = base_cost
            m = len(seq)
            for i in range(m):
                node = seq[i]
                tmp = seq[:i] + seq[i + 1 :]
                for j in range(m):
                    cand = tmp[:j] + [node] + tmp[j:]
                    if cand == seq:
                        continue
                    if not route_feasible_time(cand):
                        continue
                    cc = route_cost(cand, r["qty"])
                    if cc + 1e-9 < best_cost:
                        best_cost = cc
                        best_seq = cand
            if best_seq != seq:
                r["seq"] = best_seq
                improved = True

    # route elimination by reinsertion
    improved_global = True
    while improved_global and not current_time_exceeded():
        improved_global = False

        # First, improve each route order
        for r in routes:
            if current_time_exceeded():
                break
            improve_route_order(r)

        routes = [r for r in routes if r["seq"]]
        routes.sort(key=lambda rr: (len(rr["seq"]), rr["load"]))

        for rem_idx in range(len(routes)):
            if current_time_exceeded():
                break

            candidate_routes = deepcopy(routes)
            removed = candidate_routes.pop(rem_idx)
            rem_items = [(c, removed["qty"][c]) for c in removed["seq"]]

            feasible_reinsert = True
            for c, q in rem_items:
                best_ins = None  # (new_cost, ridx, pos)
                for ridx, rr in enumerate(candidate_routes):
                    if c in rr["qty"]:
                        continue
                    if rr["load"] + q > Q + 1e-9:
                        continue
                    old_seq = rr["seq"]
                    old_cost = route_cost(old_seq, rr["qty"])
                    for pos in range(len(old_seq) + 1):
                        cand_seq = old_seq[:pos] + [c] + old_seq[pos:]
                        if not route_feasible_time(cand_seq):
                            continue
                        cand_qty = dict(rr["qty"])
                        cand_qty[c] = q
                        nc = route_cost(cand_seq, cand_qty)
                        delta = nc - old_cost
                        if (best_ins is None) or (delta < best_ins[0]):
                            best_ins = (delta, ridx, pos)
                if best_ins is None:
                    feasible_reinsert = False
                    break
                _, ridx, pos = best_ins
                rr = candidate_routes[ridx]
                rr["seq"] = rr["seq"][:pos] + [c] + rr["seq"][pos:]
                rr["qty"][c] = q
                rr["load"] += q

            if not feasible_reinsert:
                continue

            old_total = total_cost(routes)
            new_total = total_cost(candidate_routes)

            if new_total + 1e-9 < old_total:
                routes = candidate_routes
                improved_global = True

                if new_total + 1e-9 < best_obj:
                    best_obj = new_total
                    best_routes = deepcopy(routes)
                    best_solution = build_solution_dict(best_routes)
                    if logger:
                        logger.log_solution(best_obj, best_solution)
                break

    # Final best
    final_solution = build_solution_dict(best_routes)
    final_solution["objective_value"] = float(best_obj)
    return final_solution


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    solution = solve_instance(instance, args.time_limit, logger=logger)

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()
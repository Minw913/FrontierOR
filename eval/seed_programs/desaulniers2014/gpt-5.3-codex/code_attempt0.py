import argparse
import json
import math
import time
from typing import Dict, List, Tuple, Optional

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


EPS = 1e-6


def nearest_neighbor_route(customers, dist):
    if not customers:
        return [0, 0]
    remaining = set(customers)
    route = [0]
    cur = 0
    while remaining:
        nxt = min(remaining, key=lambda j: dist[cur][j])
        route.append(nxt)
        remaining.remove(nxt)
        cur = nxt
    route.append(0)
    return route


def route_cost(route, dist):
    return sum(dist[route[i]][route[i + 1]] for i in range(len(route) - 1))


def exact_binpack(items, m, Q, time_limit=2.0):
    # items: list of (customer_id, qty)
    n = len(items)
    if n == 0:
        return [[] for _ in range(m)]
    mdl = gp.Model("binpack")
    mdl.Params.OutputFlag = 0
    mdl.Params.Threads = 1
    mdl.Params.TimeLimit = max(0.1, time_limit)

    a = mdl.addVars(n, m, vtype=GRB.BINARY, name="a")
    y = mdl.addVars(m, vtype=GRB.BINARY, name="y")

    for i in range(n):
        mdl.addConstr(gp.quicksum(a[i, k] for k in range(m)) == 1)
    for k in range(m):
        mdl.addConstr(gp.quicksum(items[i][1] * a[i, k] for i in range(n)) <= Q * y[k])

    mdl.setObjective(gp.quicksum(y[k] for k in range(m)), GRB.MINIMIZE)
    mdl.optimize()

    if mdl.SolCount == 0:
        return None

    bins = [[] for _ in range(m)]
    for i in range(n):
        for k in range(m):
            if a[i, k].X > 0.5:
                bins[k].append(items[i][0])
                break
    return bins


def greedy_pack(items, m, Q, remaining_time=2.0):
    # items: list of (customer_id, qty)
    items = sorted(items, key=lambda x: x[1], reverse=True)
    bins = []
    loads = []

    for cid, q in items:
        best_k = -1
        best_rem = float("inf")
        for k in range(len(bins)):
            rem = Q - loads[k]
            if q <= rem + EPS and rem - q < best_rem:
                best_rem = rem - q
                best_k = k
        if best_k >= 0:
            bins[best_k].append(cid)
            loads[best_k] += q
        else:
            if len(bins) < m:
                bins.append([cid])
                loads.append(q)
            else:
                # Try exact bin packing as fallback
                ex = exact_binpack(items, m, Q, time_limit=min(2.0, remaining_time))
                if ex is None:
                    return None
                return ex

    while len(bins) < m:
        bins.append([])
    return bins


def build_greedy_solution(data, time_budget=5.0):
    n = data["num_customers"]
    T = data["num_periods"]
    m = data["num_vehicles"]
    Q = data["vehicle_capacity"]
    dist = data["distance_matrix"]

    depot = data["depot"]
    customers = {c["id"]: c for c in data["customers"]}

    inv_depot = float(depot["initial_inventory"])
    inv_c = {i: float(customers[i]["initial_inventory"]) for i in range(1, n + 1)}

    prod = float(depot["production_per_period"])
    h_depot = float(depot["holding_cost"])

    objective = 0.0
    periods = {}

    start_data = {
        "routes_by_tk": {},       # (t,k) -> route list
        "delivery_by_tik": {},    # (t,i,k) -> qty
        "inv_depot_t": {},
        "inv_c_ti": {}
    }

    t0 = time.time()

    for t in range(1, T + 1):
        inv_depot += prod

        deliveries = {}
        req_items = []

        # Minimal deliveries to satisfy end-of-period min inventory
        for i in range(1, n + 1):
            c = customers[i]
            d = float(c["demand_per_period"])
            min_inv = float(c.get("min_inventory", 0))
            max_inv = float(c["max_inventory"])

            req = max(0.0, d + min_inv - inv_c[i])
            req = min(req, max_inv - inv_c[i])
            if req > EPS:
                deliveries[i] = req
                req_items.append((i, req))

        total_req = sum(deliveries.values())
        if total_req > inv_depot + 1e-5:
            return None  # infeasible for this greedy policy
        if any(q > Q + 1e-5 for _, q in req_items):
            return None

        rem = max(0.1, time_budget - (time.time() - t0))
        bins = greedy_pack(req_items, m, Q, remaining_time=rem)
        if bins is None:
            return None

        # Build routes for non-empty bins
        routes = []
        veh_idx = 0
        for kbin in bins:
            if not kbin:
                veh_idx += 1
                continue
            r = nearest_neighbor_route(kbin, dist)
            routes.append(r)
            start_data["routes_by_tk"][(t, veh_idx + 1)] = r
            veh_idx += 1

        # Assign deliveries to first vehicle route that contains customer
        # (for MIP start consistency)
        for i, q in deliveries.items():
            assigned = False
            for k in range(1, m + 1):
                r = start_data["routes_by_tk"].get((t, k), None)
                if r and i in r:
                    start_data["delivery_by_tik"][(t, i, k)] = q
                    assigned = True
                    break
            if not assigned:
                return None

        # Update inventories
        inv_depot -= total_req
        if inv_depot < -1e-6 or inv_depot > float(depot["max_inventory"]) + 1e-6:
            return None

        for i in range(1, n + 1):
            d = float(customers[i]["demand_per_period"])
            q = deliveries.get(i, 0.0)
            inv_c[i] = inv_c[i] + q - d
            if inv_c[i] < float(customers[i].get("min_inventory", 0)) - 1e-6:
                return None
            if inv_c[i] > float(customers[i]["max_inventory"]) + 1e-6:
                return None

        # Objective terms
        travel = sum(route_cost(r, dist) for r in routes)
        hold = h_depot * inv_depot + sum(float(customers[i]["holding_cost"]) * inv_c[i] for i in range(1, n + 1))
        objective += travel + hold

        inv_out = {"depot": float(inv_depot)}
        for i in range(1, n + 1):
            inv_out[str(i)] = float(inv_c[i])

        periods[str(t)] = {
            "routes": routes,
            "deliveries": {str(i): float(q) for i, q in deliveries.items() if q > EPS},
            "inventories": inv_out
        }

        start_data["inv_depot_t"][t] = float(inv_depot)
        for i in range(1, n + 1):
            start_data["inv_c_ti"][(t, i)] = float(inv_c[i])

    return {"objective_value": float(objective), "solution_details": {"periods": periods}}, start_data


def solve_instance(data, time_limit, logger=None):
    n = data["num_customers"]
    T = data["num_periods"]
    m = data["num_vehicles"]
    Q = data["vehicle_capacity"]
    dist = data["distance_matrix"]

    depot = data["depot"]
    customers_list = data["customers"]
    customers = {c["id"]: c for c in customers_list}
    C = list(range(1, n + 1))
    V = [0] + C
    K = list(range(1, m + 1))
    P = list(range(1, T + 1))

    start_time = time.time()

    # Build a quick greedy feasible solution for warm start / fallback
    greedy = build_greedy_solution(data, time_budget=min(5.0, 0.15 * time_limit))
    greedy_sol = None
    greedy_start = None
    best_logged = float("inf")
    if greedy is not None:
        greedy_sol, greedy_start = greedy
        if logger:
            logger.log(greedy_sol["objective_value"])
        best_logged = greedy_sol["objective_value"]

    mdl = gp.Model("IRP_single_product")
    mdl.Params.OutputFlag = 0
    mdl.Params.Threads = 1
    mdl.Params.MIPFocus = 1

    remaining = max(1, int(time_limit - (time.time() - start_time)))
    mdl.Params.TimeLimit = remaining

    # Variables
    x = mdl.addVars(
        [(i, j, k, t) for t in P for k in K for i in V for j in V if i != j],
        vtype=GRB.BINARY, name="x"
    )
    y = mdl.addVars([(i, k, t) for t in P for k in K for i in C], vtype=GRB.BINARY, name="y")
    z = mdl.addVars([(k, t) for t in P for k in K], vtype=GRB.BINARY, name="z")
    q = mdl.addVars([(i, k, t) for t in P for k in K for i in C], lb=0.0, vtype=GRB.CONTINUOUS, name="q")
    I0 = mdl.addVars(P, lb=0.0, ub=float(depot["max_inventory"]), vtype=GRB.CONTINUOUS, name="I0")
    Ic = mdl.addVars([(i, t) for t in P for i in C], vtype=GRB.CONTINUOUS, name="Ic")
    u = mdl.addVars([(i, k, t) for t in P for k in K for i in C], lb=0.0, ub=float(n), vtype=GRB.CONTINUOUS, name="u")

    # Inventory constraints
    prod = float(depot["production_per_period"])
    init0 = float(depot["initial_inventory"])
    max0 = float(depot["max_inventory"])

    for t in P:
        outgoing = gp.quicksum(q[i, k, t] for i in C for k in K)
        if t == 1:
            mdl.addConstr(I0[t] == init0 + prod - outgoing)
        else:
            mdl.addConstr(I0[t] == I0[t - 1] + prod - outgoing)
        mdl.addConstr(I0[t] <= max0)
        mdl.addConstr(I0[t] >= 0.0)

    for i in C:
        c = customers[i]
        init_i = float(c["initial_inventory"])
        max_i = float(c["max_inventory"])
        min_i = float(c.get("min_inventory", 0))
        dem_i = float(c["demand_per_period"])

        for t in P:
            delivered = gp.quicksum(q[i, k, t] for k in K)
            if t == 1:
                mdl.addConstr(Ic[i, t] == init_i + delivered - dem_i)
            else:
                mdl.addConstr(Ic[i, t] == Ic[i, t - 1] + delivered - dem_i)
            mdl.addConstr(Ic[i, t] <= max_i)
            mdl.addConstr(Ic[i, t] >= min_i)

    # Delivery/visit linking and vehicle capacity
    for t in P:
        for i in C:
            mdl.addConstr(gp.quicksum(y[i, k, t] for k in K) <= 1)
            ub_i = min(Q, float(customers[i]["max_inventory"]))
            for k in K:
                mdl.addConstr(q[i, k, t] <= ub_i * y[i, k, t])

        for k in K:
            mdl.addConstr(gp.quicksum(q[i, k, t] for i in C) <= Q * z[k, t])

    # Routing degree constraints
    for t in P:
        for k in K:
            mdl.addConstr(gp.quicksum(x[0, j, k, t] for j in C) == z[k, t])
            mdl.addConstr(gp.quicksum(x[j, 0, k, t] for j in C) == z[k, t])

            for i in C:
                mdl.addConstr(gp.quicksum(x[i, j, k, t] for j in V if j != i) == y[i, k, t])
                mdl.addConstr(gp.quicksum(x[j, i, k, t] for j in V if j != i) == y[i, k, t])
                mdl.addConstr(y[i, k, t] <= z[k, t])

    # MTZ subtour elimination
    for t in P:
        for k in K:
            for i in C:
                mdl.addConstr(u[i, k, t] <= n * y[i, k, t])
                mdl.addConstr(u[i, k, t] >= y[i, k, t])
            for i in C:
                for j in C:
                    if i != j:
                        mdl.addConstr(u[i, k, t] - u[j, k, t] + n * x[i, j, k, t] <= n - 1)

    # Objective
    travel_cost = gp.quicksum(dist[i][j] * x[i, j, k, t] for t in P for k in K for i in V for j in V if i != j)
    holding_cost = gp.quicksum(float(depot["holding_cost"]) * I0[t] for t in P) + gp.quicksum(
        float(customers[i]["holding_cost"]) * Ic[i, t] for t in P for i in C
    )
    mdl.setObjective(travel_cost + holding_cost, GRB.MINIMIZE)

    # Warm start from greedy solution
    if greedy_start is not None:
        for t in P:
            if t in greedy_start["inv_depot_t"]:
                I0[t].Start = greedy_start["inv_depot_t"][t]
            for i in C:
                key = (t, i)
                if key in greedy_start["inv_c_ti"]:
                    Ic[i, t].Start = greedy_start["inv_c_ti"][key]

            for k in K:
                route = greedy_start["routes_by_tk"].get((t, k))
                if route and len(route) >= 2:
                    z[k, t].Start = 1.0
                    for idx in range(len(route) - 1):
                        a, b = route[idx], route[idx + 1]
                        if a != b and (a, b, k, t) in x:
                            x[a, b, k, t].Start = 1.0
                    for i in route:
                        if i != 0:
                            y[i, k, t].Start = 1.0
                else:
                    z[k, t].Start = 0.0

            for i in C:
                for k in K:
                    q_key = (t, i, k)
                    if q_key in greedy_start["delivery_by_tik"]:
                        q[i, k, t].Start = greedy_start["delivery_by_tik"][q_key]
                        y[i, k, t].Start = 1.0

    # Callback for incumbent logging
    incumbent_best = [best_logged]

    def cb(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj + 1e-6 < incumbent_best[0]:
                incumbent_best[0] = obj
                if logger:
                    logger.log(float(obj))

    mdl.optimize(cb)

    if mdl.SolCount == 0:
        # Fallback to greedy if available
        if greedy_sol is not None:
            return greedy_sol
        # Last resort schema-valid empty output
        periods = {}
        for t in P:
            invs = {"depot": 0.0}
            for i in C:
                invs[str(i)] = 0.0
            periods[str(t)] = {"routes": [], "deliveries": {}, "inventories": invs}
        return {"objective_value": float("inf"), "solution_details": {"periods": periods}}

    # Extract best MIP solution
    objective_value = float(mdl.ObjVal)
    if logger and objective_value + 1e-6 < incumbent_best[0]:
        logger.log(objective_value)

    periods_out = {}

    for t in P:
        routes_t = []
        deliveries_t = {}

        # Deliveries
        for i in C:
            qtot = sum(q[i, k, t].X for k in K)
            if qtot > EPS:
                deliveries_t[str(i)] = float(qtot)

        # Routes from x
        for k in K:
            if z[k, t].X <= 0.5:
                continue
            # Build successor mapping
            succ = {}
            for i in V:
                for j in V:
                    if i != j and (i, j, k, t) in x and x[i, j, k, t].X > 0.5:
                        succ[i] = j

            route = [0]
            cur = 0
            safety = 0
            while safety <= n + 2:
                safety += 1
                if cur not in succ:
                    break
                nxt = succ[cur]
                route.append(nxt)
                cur = nxt
                if cur == 0:
                    break
            if len(route) >= 2 and route[0] == 0 and route[-1] == 0:
                routes_t.append(route)

        invs = {"depot": float(I0[t].X)}
        for i in C:
            invs[str(i)] = float(Ic[i, t].X)

        periods_out[str(t)] = {
            "routes": routes_t,
            "deliveries": deliveries_t,
            "inventories": invs
        }

    return {"objective_value": objective_value, "solution_details": {"periods": periods_out}}


def main():
    parser = argparse.ArgumentParser(description="Solve multi-period inventory routing problem with Gurobi.")
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, required=False, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    solution = solve_instance(data, args.time_limit, logger=logger)

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()
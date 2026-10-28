import argparse
import json
import math
from collections import defaultdict

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def solve_instance(instance, time_limit, logger=None):
    n = int(instance["num_customers"])
    Q = float(instance["vehicle_capacity"])
    K = int(instance["num_vehicles_upper_bound"])

    depot = instance["depot"]
    customers_raw = instance["customers"]
    dist_matrix = instance["distance_matrix"]

    # Customers indexed 1..n
    cust_by_id = {int(c["id"]): c for c in customers_raw}
    customers = [cust_by_id[i] for i in range(1, n + 1)]

    start_depot = 0
    end_depot = n + 1
    C = list(range(1, n + 1))
    Vh = list(range(K))

    # Time windows, service, demand on expanded node set {0,1..n,n+1}
    e = {}
    l = {}
    s = {}
    dem = {}

    e[start_depot], l[start_depot] = depot["time_window"]
    s[start_depot] = float(depot.get("service_time", 0))
    dem[start_depot] = 0.0

    for c in customers:
        i = int(c["id"])
        e[i], l[i] = c["time_window"]
        s[i] = float(c["service_time"])
        dem[i] = float(c["demand"])

    # End depot duplicates depot TW/service/demand
    e[end_depot], l[end_depot] = depot["time_window"]
    s[end_depot] = 0.0
    dem[end_depot] = 0.0

    def d(i, j):
        ii = 0 if i == end_depot else i
        jj = 0 if j == end_depot else j
        return float(dist_matrix[ii][jj])

    # Build feasible arc set by TW screening
    A = []
    outgoing = defaultdict(list)
    incoming = defaultdict(list)
    M = {}

    candidate_from = [start_depot] + C
    candidate_to = C + [end_depot]

    for i in candidate_from:
        for j in candidate_to:
            if i == j:
                continue
            if i == start_depot and j == end_depot:
                feasible = True  # idle vehicle arc always allowed
            else:
                travel = d(i, j)
                feasible = (e[i] + s[i] + travel <= l[j] + 1e-9)
            if feasible:
                A.append((i, j))
                outgoing[i].append(j)
                incoming[j].append(i)
                M[(i, j)] = max(0.0, l[i] + s[i] + d(i, j) - e[j])

    # Basic feasibility sanity: every customer must be reachable and can leave
    for i in C:
        if i not in incoming or i not in outgoing or len(incoming[i]) == 0 or len(outgoing[i]) == 0:
            return {
                "objective_value": 1e30,
                "num_vehicles_used": 0,
                "routes": [],
                "deliveries": {
                    str(cid): {"demand": int(dem[cid]), "total_delivered": 0.0}
                    for cid in C
                },
            }

    total_demand = sum(dem[i] for i in C)
    min_veh_lb = int(math.ceil(total_demand / Q - 1e-12))

    model = gp.Model("SDVRPTW")
    model.Params.TimeLimit = int(time_limit)
    model.Params.Threads = 1
    model.Params.OutputFlag = 0

    x = model.addVars(((k, i, j) for k in Vh for (i, j) in A), vtype=GRB.BINARY, name="x")
    y = model.addVars(Vh, vtype=GRB.BINARY, name="y")
    q = model.addVars(((k, i) for k in Vh for i in C), lb=0.0, vtype=GRB.CONTINUOUS, name="q")
    t = model.addVars(((k, i) for k in Vh for i in [start_depot] + C + [end_depot]),
                      lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="t")

    # Objective: total travel distance
    model.setObjective(gp.quicksum(d(i, j) * x[k, i, j] for k in Vh for (i, j) in A), GRB.MINIMIZE)

    # Bounds on time vars
    for k in Vh:
        for i in [start_depot] + C + [end_depot]:
            t[k, i].LB = e[i]
            t[k, i].UB = l[i]
        model.addConstr(t[k, start_depot] == e[start_depot], name=f"t0fix_{k}")

    # Route/path constraints per vehicle
    for k in Vh:
        # exactly one arc leaves start depot
        model.addConstr(gp.quicksum(x[k, start_depot, j] for j in outgoing[start_depot]) == 1, name=f"start_out_{k}")
        # exactly one arc enters end depot
        model.addConstr(gp.quicksum(x[k, i, end_depot] for i in incoming[end_depot]) == 1, name=f"end_in_{k}")

        # used vehicle if leaves start to a customer
        arcs0cust = [j for j in C if (start_depot, j) in A]
        model.addConstr(y[k] == gp.quicksum(x[k, start_depot, j] for j in arcs0cust), name=f"use_def_{k}")

        # flow conservation at customer nodes, at most one visit by this vehicle
        for i in C:
            out_expr = gp.quicksum(x[k, i, j] for j in outgoing[i])
            in_expr = gp.quicksum(x[k, h, i] for h in incoming[i])
            model.addConstr(out_expr == in_expr, name=f"flow_{k}_{i}")
            model.addConstr(out_expr <= 1, name=f"visit_once_{k}_{i}")

            # delivery only if visited
            model.addConstr(q[k, i] <= dem[i] * out_expr, name=f"q_visit_{k}_{i}")

        # vehicle capacity
        model.addConstr(gp.quicksum(q[k, i] for i in C) <= Q * y[k], name=f"cap_{k}")

    # Demand satisfaction (split deliveries allowed)
    for i in C:
        model.addConstr(gp.quicksum(q[k, i] for k in Vh) >= dem[i], name=f"demand_{i}")

    # Vehicle count bounds
    model.addConstr(gp.quicksum(y[k] for k in Vh) >= min_veh_lb, name="veh_lb")
    model.addConstr(gp.quicksum(y[k] for k in Vh) <= K, name="veh_ub")

    # Symmetry breaking: use lower-index vehicles first
    for k in range(K - 1):
        model.addConstr(y[k] >= y[k + 1], name=f"sym_{k}")

    # Time propagation constraints
    for k in Vh:
        for (i, j) in A:
            model.addConstr(
                t[k, j] >= t[k, i] + s[i] + d(i, j) - M[(i, j)] * (1 - x[k, i, j]),
                name=f"time_{k}_{i}_{j}"
            )

    # Simple feasible start: one-customer routes per needed trip
    # (if possible from arc availability)
    remaining = {i: dem[i] for i in C}
    veh_plan = []  # list of tuples (vehicle k, customer i, qty)
    k_ptr = 0
    feasible_start = True
    for i in C:
        while remaining[i] > 1e-9:
            if k_ptr >= K:
                feasible_start = False
                break
            if (start_depot, i) not in A or (i, end_depot) not in A:
                feasible_start = False
                break
            qty = min(Q, remaining[i])
            veh_plan.append((k_ptr, i, qty))
            remaining[i] -= qty
            k_ptr += 1
        if not feasible_start:
            break

    if feasible_start:
        # set y start
        used_set = set(k for (k, _, _) in veh_plan)
        for k in Vh:
            y[k].Start = 1.0 if k in used_set else 0.0

        # arcs and times
        for (k, i, qty) in veh_plan:
            x[k, start_depot, i].Start = 1.0
            x[k, i, end_depot].Start = 1.0
            q[k, i].Start = qty

            ti = max(e[i], e[start_depot] + d(start_depot, i))
            te = max(e[end_depot], ti + s[i] + d(i, end_depot))
            t[k, start_depot].Start = e[start_depot]
            t[k, i].Start = ti
            t[k, end_depot].Start = te

        for k in Vh:
            if k not in used_set:
                if (start_depot, end_depot) in A:
                    x[k, start_depot, end_depot].Start = 1.0
                t[k, start_depot].Start = e[start_depot]
                t[k, end_depot].Start = e[end_depot]

        # log heuristic value
        init_obj = 0.0
        for (k, i, qty) in veh_plan:
            init_obj += d(start_depot, i) + d(i, end_depot)
        if logger:
            logger.log(float(init_obj))

    # Incumbent callback logging
    model._best_logged = float("inf")

    def cb(m, where):
        if where == GRB.Callback.MIPSOL and logger is not None:
            obj = m.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj + 1e-9 < m._best_logged:
                m._best_logged = obj
                logger.log(float(obj))

    model.optimize(cb)

    # Build output
    if model.SolCount == 0:
        return {
            "objective_value": 1e30,
            "num_vehicles_used": 0,
            "routes": [],
            "deliveries": {
                str(cid): {"demand": int(dem[cid]), "total_delivered": 0.0}
                for cid in C
            },
        }

    routes_out = []
    total_delivered = {i: 0.0 for i in C}

    def get_next(k, cur):
        for j in outgoing[cur]:
            if x[k, cur, j].X > 0.5:
                return j
        return None

    for k in Vh:
        if y[k].X > 0.5:
            route = [start_depot]
            cur = start_depot
            guard = 0
            while cur != end_depot and guard <= n + 5:
                nxt = get_next(k, cur)
                if nxt is None:
                    break
                route.append(int(nxt))
                cur = nxt
                guard += 1

            deliv = {}
            for i in C:
                qv = q[k, i].X
                if qv > 1e-8:
                    deliv[str(i)] = float(qv)
                    total_delivered[i] += float(qv)

            routes_out.append({
                "vehicle": int(k + 1),
                "route": route,
                "deliveries": deliv
            })

    deliveries_out = {
        str(i): {
            "demand": int(dem[i]),
            "total_delivered": float(total_delivered[i])
        } for i in C
    }

    return {
        "objective_value": float(model.ObjVal),
        "num_vehicles_used": int(len(routes_out)),
        "routes": routes_out,
        "deliveries": deliveries_out
    }


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
import argparse
import json
import math
import time
from collections import defaultdict

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def remaining_time(deadline):
    return max(0.0, deadline - time.time())


def euclidean(a, b):
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


def build_distance_matrix(depot, satellites, customers):
    nodes = {depot["id"]: depot}
    for s in satellites:
        nodes[s["id"]] = s
    for c in customers:
        nodes[c["id"]] = c

    ids = list(nodes.keys())
    dist = {i: {} for i in ids}
    for i in ids:
        for j in ids:
            if j in dist[i]:
                continue
            d = euclidean(nodes[i], nodes[j])
            dist[i][j] = d
            if i != j:
                dist[j][i] = d
    return dist


def route_cost(route, dist):
    return sum(dist[route[i]][route[i + 1]] for i in range(len(route) - 1))


def nearest_neighbor_cycle(start, customers, dist):
    if not customers:
        return [start, start]
    unvisited = set(customers)
    curr = start
    seq = []
    while unvisited:
        nxt = min(unvisited, key=lambda x: dist[curr][x])
        seq.append(nxt)
        unvisited.remove(nxt)
        curr = nxt
    return [start] + seq + [start]


def two_opt_with_fixed_start(route, dist):
    # route format: [start, ..., start]
    if len(route) <= 4:
        return route
    start = route[0]
    best = route[:]
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best) - 1):
                if j - i == 1:
                    continue
                cand = best[:]
                cand[i:j] = reversed(cand[i:j])
                if route_cost(cand, dist) + 1e-9 < route_cost(best, dist):
                    best = cand
                    improved = True
    if best[0] != start or best[-1] != start:
        return route
    return best


def solve_group_assignment(
    instance,
    dist,
    strict_mode,
    deadline,
):
    rem = remaining_time(deadline)
    if rem < 1.0:
        return None

    groups = instance["groups"]
    satellites = instance["satellites"]
    customers = instance["customers"]
    depot = instance["depot"]

    Q2 = instance["Q2"]
    T2 = instance["T2"]

    sat_ids = [s["id"] for s in satellites]
    group_ids = [g["group_id"] for g in groups]

    customer_map = {c["id"]: c for c in customers}
    sat_map = {s["id"]: s for s in satellites}

    group_demand = {}
    group_route_lb = {}
    group_customers = {}
    for g in groups:
        gid = g["group_id"]
        cids = g["customer_ids"]
        group_customers[gid] = cids
        d = sum(customer_map[cid]["demand"] for cid in cids)
        group_demand[gid] = d
        group_route_lb[gid] = int(math.ceil(d / Q2)) if d > 0 else 0

    model = gp.Model("group_assignment")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = max(1.0, min(rem - 0.05, rem))

    z = model.addVars(group_ids, sat_ids, vtype=GRB.BINARY, name="z")

    # each group assigned to exactly one satellite
    for g in group_ids:
        model.addConstr(gp.quicksum(z[g, s] for s in sat_ids) == 1)

    # satellite throughput capacity
    for s in sat_ids:
        cap = sat_map[s]["capacity"]
        model.addConstr(gp.quicksum(group_demand[g] * z[g, s] for g in group_ids) <= cap)

    if strict_mode:
        # conservative vehicle constraints from group-level lower bound
        for s in sat_ids:
            maxv = sat_map[s]["max_vehicles"]
            model.addConstr(gp.quicksum(group_route_lb[g] * z[g, s] for g in group_ids) <= maxv)
        model.addConstr(gp.quicksum(group_route_lb[g] * z[g, s] for g in group_ids for s in sat_ids) <= T2)
    else:
        # relaxed demand-based lower bound
        v = model.addVars(sat_ids, vtype=GRB.INTEGER, lb=0, name="v")
        for s in sat_ids:
            maxv = sat_map[s]["max_vehicles"]
            model.addConstr(v[s] <= maxv)
            model.addConstr(
                Q2 * v[s] >= gp.quicksum(group_demand[g] * z[g, s] for g in group_ids)
            )
        model.addConstr(gp.quicksum(v[s] for s in sat_ids) <= T2)

    # objective approximation: second-echelon radial + handling + rough first-echelon term
    obj = gp.LinExpr()
    Q1 = instance["Q1"]
    for g in group_ids:
        dem = group_demand[g]
        cids = group_customers[g]
        for s in sat_ids:
            radial = sum(2.0 * dist[s][cid] for cid in cids)
            handling = sat_map[s]["handling_cost"] * dem
            first_approx = (2.0 * dist[depot["id"]][s] / max(Q1, 1)) * dem
            coef = radial + handling + 0.5 * first_approx
            obj += coef * z[g, s]

    model.setObjective(obj, GRB.MINIMIZE)
    model.optimize()

    if model.SolCount == 0:
        return None

    assignment = {}
    for g in group_ids:
        for s in sat_ids:
            if z[g, s].X > 0.5:
                assignment[g] = s
                break
        if g not in assignment:
            return None
    return assignment


def solve_bin_packing(customers, demand_map, capacity, max_bins, deadline, name="binpack"):
    # returns list of bins (each bin is list of customer ids), or None if infeasible/no solution
    n = len(customers)
    if n == 0:
        return []
    if max_bins <= 0:
        return None

    rem = remaining_time(deadline)
    if rem < 0.5:
        return None

    B = min(n, max_bins)
    model = gp.Model(name)
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = max(0.5, min(rem - 0.05, rem))

    x = model.addVars(n, B, vtype=GRB.BINARY, name="x")
    y = model.addVars(B, vtype=GRB.BINARY, name="y")

    for i in range(n):
        model.addConstr(gp.quicksum(x[i, b] for b in range(B)) == 1)

    for b in range(B):
        model.addConstr(
            gp.quicksum(demand_map[customers[i]] * x[i, b] for i in range(n)) <= capacity * y[b]
        )

    for i in range(n):
        for b in range(B):
            model.addConstr(x[i, b] <= y[b])

    # symmetry breaking
    for b in range(B - 1):
        model.addConstr(y[b] >= y[b + 1])

    model.setObjective(gp.quicksum(y[b] for b in range(B)), GRB.MINIMIZE)
    model.optimize()

    if model.SolCount == 0:
        return None

    used = [b for b in range(B) if y[b].X > 0.5]
    bins = {b: [] for b in used}
    for i in range(n):
        assigned = None
        for b in range(B):
            if x[i, b].X > 0.5:
                assigned = b
                break
        if assigned is None:
            return None
        if assigned not in bins:
            bins[assigned] = []
        bins[assigned].append(customers[i])

    return [bins[b] for b in sorted(bins.keys()) if len(bins[b]) > 0]


def solve_first_echelon_packing(items, Q1, T1, deadline):
    # items: list of dict {sat, qty}
    # returns bins: list of list of item indices
    if not items:
        return []
    if T1 <= 0:
        return None

    n = len(items)
    B = min(T1, n)

    rem = remaining_time(deadline)
    if rem < 0.5:
        return None

    sats = sorted(set(it["sat"] for it in items))
    sat_to_items = defaultdict(list)
    for i, it in enumerate(items):
        sat_to_items[it["sat"]].append(i)

    model = gp.Model("first_pack")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = max(0.5, min(rem - 0.05, rem))

    x = model.addVars(n, B, vtype=GRB.BINARY, name="x")
    y = model.addVars(B, vtype=GRB.BINARY, name="y")

    for i in range(n):
        model.addConstr(gp.quicksum(x[i, b] for b in range(B)) == 1)

    for b in range(B):
        model.addConstr(gp.quicksum(items[i]["qty"] * x[i, b] for i in range(n)) <= Q1 * y[b])

    # at most one visit per satellite per first-echelon vehicle
    for s in sats:
        idxs = sat_to_items[s]
        for b in range(B):
            model.addConstr(gp.quicksum(x[i, b] for i in idxs) <= 1)

    for i in range(n):
        for b in range(B):
            model.addConstr(x[i, b] <= y[b])

    for b in range(B - 1):
        model.addConstr(y[b] >= y[b + 1])

    model.setObjective(gp.quicksum(y[b] for b in range(B)), GRB.MINIMIZE)
    model.optimize()

    if model.SolCount == 0:
        return None

    used = [b for b in range(B) if y[b].X > 0.5]
    bins = {b: [] for b in used}
    for i in range(n):
        bb = None
        for b in range(B):
            if x[i, b].X > 0.5:
                bb = b
                break
        if bb is None:
            return None
        if bb not in bins:
            bins[bb] = []
        bins[bb].append(i)

    return [bins[b] for b in sorted(bins.keys()) if len(bins[b]) > 0]


def build_complete_solution(instance, dist, group_assignment, deadline):
    depot = instance["depot"]
    satellites = instance["satellites"]
    customers = instance["customers"]
    groups = instance["groups"]

    sat_map = {s["id"]: s for s in satellites}
    cust_map = {c["id"]: c for c in customers}
    group_map = {g["group_id"]: g["customer_ids"] for g in groups}

    Q2 = instance["Q2"]
    T2 = instance["T2"]
    Q1 = instance["Q1"]
    T1 = instance["T1"]

    # customer -> assigned satellite by group
    customer_sat = {}
    for gid, sat in group_assignment.items():
        for cid in group_map[gid]:
            customer_sat[cid] = sat

    sat_customers = defaultdict(list)
    sat_demand = defaultdict(float)
    for c in customers:
        sat = customer_sat[c["id"]]
        sat_customers[sat].append(c["id"])
        sat_demand[sat] += c["demand"]

    # second echelon: exact/min bin packing by capacity at each satellite
    second_routes = []
    total_second_vehicles = 0
    second_travel = 0.0

    sat_ids = [s["id"] for s in satellites]
    demand_map = {c["id"]: c["demand"] for c in customers}

    for s in sat_ids:
        cids = sat_customers.get(s, [])
        if not cids:
            continue

        maxv = sat_map[s]["max_vehicles"]
        bins = solve_bin_packing(
            cids,
            demand_map,
            Q2,
            maxv,
            deadline,
            name=f"binpack_sat_{s}",
        )
        if bins is None:
            return None

        total_second_vehicles += len(bins)
        if total_second_vehicles > T2:
            return None

        for b in bins:
            r = nearest_neighbor_cycle(s, b, dist)
            r = two_opt_with_fixed_start(r, dist)
            second_routes.append({"satellite": s, "route": r})
            second_travel += route_cost(r, dist)

    # first echelon demand items
    items = []
    for s in sat_ids:
        d = sat_demand.get(s, 0.0)
        if d <= 1e-9:
            continue
        k = int(d // Q1)
        rem = d - k * Q1
        for _ in range(k):
            items.append({"sat": s, "qty": float(Q1)})
        if rem > 1e-9:
            items.append({"sat": s, "qty": float(rem)})

    first_bins = solve_first_echelon_packing(items, Q1, T1, deadline)
    if first_bins is None:
        return None

    first_routes = []
    first_travel = 0.0
    vehicle_id = 1
    for bin_items in first_bins:
        sat_qty = defaultdict(float)
        for i in bin_items:
            sat_qty[items[i]["sat"]] += items[i]["qty"]
        sats = list(sat_qty.keys())

        # build route depot -> sats -> depot
        if sats:
            rr = nearest_neighbor_cycle(depot["id"], sats, dist)
            rr = two_opt_with_fixed_start(rr, dist)
        else:
            rr = [depot["id"], depot["id"]]

        first_travel += route_cost(rr, dist)
        first_routes.append(
            {
                "vehicle": vehicle_id,
                "route": rr,
                "deliveries": {str(k): float(v) for k, v in sat_qty.items()},
            }
        )
        vehicle_id += 1

    # handling cost
    handling = 0.0
    for s in sat_ids:
        d = sat_demand.get(s, 0.0)
        if d > 0:
            handling += sat_map[s]["handling_cost"] * d

    obj = first_travel + second_travel + handling

    # output format
    sol = {
        "objective_value": float(obj),
        "first_echelon_routes": first_routes,
        "second_echelon_routes": second_routes,
        "group_assignments": {str(g): int(s) for g, s in group_assignment.items()},
    }
    return sol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, required=False, default=None)
    args = parser.parse_args()

    start = time.time()
    deadline = start + max(1, args.time_limit)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    depot = instance["depot"]
    satellites = instance["satellites"]
    customers = instance["customers"]

    dist = build_distance_matrix(depot, satellites, customers)

    best_sol = None
    best_obj = float("inf")

    # Try strict assignment first, then relaxed if time permits
    modes = [True, False]
    for strict in modes:
        if remaining_time(deadline) < 1.0:
            break

        assignment = solve_group_assignment(instance, dist, strict, deadline)
        if assignment is None:
            continue

        cand = build_complete_solution(instance, dist, assignment, deadline)
        if cand is None:
            continue

        if cand["objective_value"] + 1e-9 < best_obj:
            best_obj = cand["objective_value"]
            best_sol = cand
            if logger:
                logger.log(best_obj)

        # if strict already good and little time, stop
        if strict and remaining_time(deadline) < 2.0:
            break

    # Fallback minimal schema-compliant output if no feasible found
    if best_sol is None:
        best_sol = {
            "objective_value": float("inf"),
            "first_echelon_routes": [],
            "second_echelon_routes": [],
            "group_assignments": {},
        }

    with open(args.solution_path, "w") as f:
        json.dump(best_sol, f, indent=2)


if __name__ == "__main__":
    main()
import argparse
import json
import math
import time
from collections import defaultdict

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def euclidean(a, b):
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


def build_and_solve(instance, time_limit, logger=None):
    K = int(instance["num_vehicles"])
    C = int(instance["vehicle_capacity"])
    max_ride_time = float(instance["maximum_ride_time"])
    max_route_duration = float(instance["maximum_route_duration"])

    nodes_raw = {int(n["node_id"]): n for n in instance["nodes"]}
    node_ids = sorted(nodes_raw.keys())

    origin = next(i for i in node_ids if nodes_raw[i]["node_type"] == "origin_depot")
    destination = next(i for i in node_ids if nodes_raw[i]["node_type"] == "destination_depot")
    pickups = [i for i in node_ids if nodes_raw[i]["node_type"] == "pickup"]
    dropoffs = [i for i in node_ids if nodes_raw[i]["node_type"] == "dropoff"]
    non_depots = [i for i in node_ids if i not in (origin, destination)]

    users = sorted({nodes_raw[p]["user_id"] for p in pickups})
    user_pickup = {}
    user_dropoff = {}
    for p in pickups:
        u = nodes_raw[p]["user_id"]
        user_pickup[u] = p
    for d in dropoffs:
        u = nodes_raw[d]["user_id"]
        user_dropoff[u] = d

    # Distances / travel times / costs
    dist = {}
    for i in node_ids:
        for j in node_ids:
            if i != j:
                dist[(i, j)] = euclidean(nodes_raw[i], nodes_raw[j])

    # Arc set: no incoming to origin, no outgoing from destination
    arcs = []
    for i in node_ids:
        if i == destination:
            continue
        for j in node_ids:
            if j == origin or i == j:
                continue
            arcs.append((i, j))

    E = {i: float(nodes_raw[i]["earliest_time"]) for i in node_ids}
    L = {i: float(nodes_raw[i]["latest_time"]) for i in node_ids}
    S = {i: float(nodes_raw[i]["service_duration"]) for i in node_ids}
    Qload = {i: int(nodes_raw[i]["load"]) for i in node_ids}

    M_time = {}
    for (i, j) in arcs:
        M_time[(i, j)] = max(0.0, L[i] + S[i] + dist[(i, j)] - E[j])

    M_load = C + max(abs(Qload[i]) for i in node_ids) + 1.0

    m = gp.Model("DARP")
    m.Params.OutputFlag = 0
    m.Params.TimeLimit = float(time_limit)
    m.Params.Threads = 1

    vehs = range(K)

    x = m.addVars([(k, i, j) for k in vehs for (i, j) in arcs], vtype=GRB.BINARY, name="x")
    y = m.addVars(vehs, vtype=GRB.BINARY, name="y")
    w = m.addVars([(k, i) for k in vehs for i in non_depots], vtype=GRB.BINARY, name="w")
    z = m.addVars([(k, u) for k in vehs for u in users], vtype=GRB.BINARY, name="z")

    B = m.addVars([(k, i) for k in vehs for i in node_ids], lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="B")
    Qv = m.addVars([(k, i) for k in vehs for i in node_ids], lb=0.0, ub=float(C), vtype=GRB.CONTINUOUS, name="Q")
    ride = m.addVars(users, lb=0.0, vtype=GRB.CONTINUOUS, name="ride")

    # Objective
    m.setObjective(gp.quicksum(dist[(i, j)] * x[k, i, j] for k in vehs for (i, j) in arcs), GRB.MINIMIZE)

    # Depot degree
    for k in vehs:
        m.addConstr(gp.quicksum(x[k, origin, j] for j in node_ids if (origin, j) in dist and (origin, j) in {(a, b) for (a, b) in arcs}) == y[k])
        m.addConstr(gp.quicksum(x[k, i, destination] for i in node_ids if (i, destination) in dist and (i, destination) in {(a, b) for (a, b) in arcs}) == y[k])

    # Flow with visit vars
    out_arcs_from = defaultdict(list)
    in_arcs_to = defaultdict(list)
    for (i, j) in arcs:
        out_arcs_from[i].append(j)
        in_arcs_to[j].append(i)

    for k in vehs:
        for i in non_depots:
            m.addConstr(gp.quicksum(x[k, i, j] for j in out_arcs_from[i]) == w[k, i])
            m.addConstr(gp.quicksum(x[k, j, i] for j in in_arcs_to[i]) == w[k, i])

    # Serve each pickup/dropoff exactly once
    for p in pickups:
        m.addConstr(gp.quicksum(w[k, p] for k in vehs) == 1)
    for d in dropoffs:
        m.addConstr(gp.quicksum(w[k, d] for k in vehs) == 1)

    # Pairing pickup-dropoff on same vehicle
    for u in users:
        p = user_pickup[u]
        d = user_dropoff[u]
        m.addConstr(gp.quicksum(z[k, u] for k in vehs) == 1)
        for k in vehs:
            m.addConstr(z[k, u] == w[k, p])
            m.addConstr(z[k, u] == w[k, d])

    # Time windows
    for k in vehs:
        for i in node_ids:
            m.addConstr(B[k, i] >= E[i])
            m.addConstr(B[k, i] <= L[i])

    # Time propagation
    for k in vehs:
        for (i, j) in arcs:
            m.addConstr(B[k, j] >= B[k, i] + S[i] + dist[(i, j)] - M_time[(i, j)] * (1 - x[k, i, j]))

    # Route duration
    for k in vehs:
        m.addConstr(B[k, destination] - B[k, origin] <= max_route_duration)
        m.addConstr(B[k, destination] >= B[k, origin])

    # Load initialization and bounds
    for k in vehs:
        m.addConstr(Qv[k, origin] == 0.0)
        m.addConstr(Qv[k, destination] == 0.0)

    for k in vehs:
        for i in non_depots:
            lb_i = max(0, Qload[i])
            ub_i = min(C, C + Qload[i])
            m.addConstr(Qv[k, i] >= lb_i * w[k, i])
            m.addConstr(Qv[k, i] <= ub_i * w[k, i] + C * (1 - w[k, i]))

    # Load propagation
    for k in vehs:
        for (i, j) in arcs:
            m.addConstr(Qv[k, j] >= Qv[k, i] + Qload[j] - M_load * (1 - x[k, i, j]))
            m.addConstr(Qv[k, j] <= Qv[k, i] + Qload[j] + M_load * (1 - x[k, i, j]))

    # Ride times
    for u in users:
        p = user_pickup[u]
        d = user_dropoff[u]
        direct = dist[(p, d)]
        m.addConstr(ride[u] >= direct)
        m.addConstr(ride[u] <= max_ride_time)

        M_ride = max_ride_time + L[d] - E[p] + S[p] + 10.0
        for k in vehs:
            expr = B[k, d] - B[k, p] - S[p]
            m.addConstr(ride[u] >= expr - M_ride * (1 - z[k, u]))
            m.addConstr(ride[u] <= expr + M_ride * (1 - z[k, u]))

    # Logging callback
    m._logger = logger
    m._best_inc = float("inf")

    def cb(model, where):
        if where == GRB.Callback.MIPSOL and model._logger is not None:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj + 1e-9 < model._best_inc:
                model._best_inc = obj
                model._logger.log(float(obj))

    m.optimize(cb)

    if m.SolCount == 0:
        return {
            "objective_value": 1e18,
            "routes": {str(k): [origin, destination] for k in vehs},
            "service_times": {},
            "ride_times": {},
        }

    # Extract solution
    x_val = {(k, i, j): x[k, i, j].X for k in vehs for (i, j) in arcs}
    w_val = {(k, i): w[k, i].X for k in vehs for i in non_depots}
    B_val = {(k, i): B[k, i].X for k in vehs for i in node_ids}
    ride_val = {u: ride[u].X for u in users}

    routes = {}
    arcset = {(i, j) for (i, j) in arcs}

    for k in vehs:
        succ = {}
        for (i, j) in arcset:
            if x_val.get((k, i, j), 0.0) > 0.5:
                succ[i] = j

        if origin not in succ:
            routes[str(k)] = [origin, destination]
            continue

        route = [origin]
        cur = origin
        visited = set([origin])
        steps = 0
        while cur != destination and steps <= len(node_ids) + 5:
            nxt = succ.get(cur, None)
            if nxt is None:
                break
            route.append(nxt)
            if nxt in visited and nxt != destination:
                break
            visited.add(nxt)
            cur = nxt
            steps += 1

        if route[-1] != destination:
            route.append(destination)
        routes[str(k)] = route

    service_times = {}
    # Per vehicle depot times
    for k in vehs:
        service_times[f"depot_start_{k}"] = float(B_val[(k, origin)])
        service_times[f"depot_end_{k}"] = float(B_val[(k, destination)])

    # Unique pickup/dropoff node times
    for i in non_depots:
        assigned_k = None
        for k in vehs:
            if w_val[(k, i)] > 0.5:
                assigned_k = k
                break
        if assigned_k is not None:
            service_times[str(i)] = float(B_val[(assigned_k, i)])

    ride_times = {str(u): float(ride_val[u]) for u in users}

    return {
        "objective_value": float(m.ObjVal),
        "routes": routes,
        "service_times": service_times,
        "ride_times": ride_times,
    }


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

    start = time.time()
    solution = build_and_solve(instance, args.time_limit, logger=logger)
    _ = time.time() - start

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()
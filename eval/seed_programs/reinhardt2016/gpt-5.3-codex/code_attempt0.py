import argparse
import json
import math
from collections import defaultdict

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def build_and_solve(instance, time_limit, logger=None):
    n_customers = instance["num_customers"]
    max_vehicles = min(instance["num_vehicles"], n_customers)
    nodes_data = instance["nodes"]
    distance = instance["distance_matrix"]
    edge_sets_data = instance["edge_sets"]

    node_ids = [nd["id"] for nd in nodes_data]  # expected 0..n
    N = len(node_ids)  # n_customers + 1
    customers = [i for i in node_ids if i != 0]
    K = list(range(max_vehicles))

    demand = {nd["id"]: nd["demand"] for nd in nodes_data}
    tw_start = {nd["id"]: nd["time_window_start"] for nd in nodes_data}
    tw_end = {nd["id"]: nd["time_window_end"] for nd in nodes_data}
    service = {nd["id"]: nd["service_time"] for nd in nodes_data}
    capacity = instance["vehicle_capacity"]

    # Map each undirected edge -> edge set id (None = free edge)
    edge_to_set = {}
    access_cost = {}
    edge_set_ids = []
    for es in edge_sets_data:
        sid = es["id"]
        edge_set_ids.append(sid)
        access_cost[sid] = float(es["access_cost"])
        for a, b in es["edges"]:
            edge_to_set[tuple(sorted((a, b)))] = sid

    # Free edges exist but no fixed-cost var needed; they remain absent from edge_to_set map
    # Build feasible directed arc list using time-window quick filtering
    arcs = []
    arc_set_id = {}  # directed arc -> set id or None
    succ = defaultdict(list)
    pred = defaultdict(list)

    def travel_time(i, j):
        return float(distance[i][j]) + float(service[i])

    for i in node_ids:
        for j in node_ids:
            if i == j:
                continue
            tt = travel_time(i, j)
            # Necessary feasibility: earliest possible at i then to j must not exceed latest of j
            if tw_start[i] + tt > tw_end[j]:
                continue
            arcs.append((i, j))
            succ[i].append(j)
            pred[j].append(i)
            sid = edge_to_set.get(tuple(sorted((i, j))), None)
            arc_set_id[(i, j)] = sid

    m = gp.Model("VRPTW_EdgeSet")
    m.Params.OutputFlag = 0
    m.Params.Threads = 1
    m.Params.TimeLimit = int(time_limit)

    # Decision variables
    x = m.addVars(K, arcs, vtype=GRB.BINARY, name="x")        # route arcs
    z = m.addVars(K, customers, vtype=GRB.BINARY, name="z")   # customer assigned to vehicle
    u = m.addVars(K, vtype=GRB.BINARY, name="u")              # vehicle used
    y = m.addVars(edge_set_ids, vtype=GRB.BINARY, name="y") if edge_set_ids else {}
    t = m.addVars(K, node_ids, vtype=GRB.CONTINUOUS, name="t")

    # Time window bounds (for all nodes and vehicles, as required)
    for k in K:
        for i in node_ids:
            m.addConstr(t[k, i] >= tw_start[i], name=f"tw_lb_{k}_{i}")
            m.addConstr(t[k, i] <= tw_end[i], name=f"tw_ub_{k}_{i}")
        # single depot time as route start anchor
        m.addConstr(t[k, 0] == tw_start[0], name=f"depot_start_{k}")

    # Customer assignment exactly once
    for i in customers:
        m.addConstr(gp.quicksum(z[k, i] for k in K) == 1, name=f"visit_once_{i}")

    # Flow linking at customers
    for k in K:
        for i in customers:
            m.addConstr(
                gp.quicksum(x[k, i, j] for j in succ[i]) == z[k, i],
                name=f"out_link_{k}_{i}"
            )
            m.addConstr(
                gp.quicksum(x[k, j, i] for j in pred[i]) == z[k, i],
                name=f"in_link_{k}_{i}"
            )

    # Depot flow and vehicle use
    for k in K:
        m.addConstr(gp.quicksum(x[k, 0, j] for j in succ[0]) == u[k], name=f"depot_out_{k}")
        m.addConstr(gp.quicksum(x[k, j, 0] for j in pred[0]) == u[k], name=f"depot_in_{k}")
        m.addConstr(gp.quicksum(z[k, i] for i in customers) <= n_customers * u[k], name=f"use_link_{k}")

    # Capacity
    for k in K:
        m.addConstr(
            gp.quicksum(demand[i] * z[k, i] for i in customers) <= capacity,
            name=f"cap_{k}"
        )

    # Optional symmetry breaking
    for k in range(max_vehicles - 1):
        m.addConstr(u[k] >= u[k + 1], name=f"sym_u_{k}")

    # Time propagation
    for k in K:
        for (i, j) in arcs:
            tt = travel_time(i, j)
            if j == 0:
                # return to depot before depot latest time
                M = max(0.0, tw_end[i] + tt - tw_start[0])
                m.addConstr(
                    t[k, i] + tt <= tw_end[0] + M * (1 - x[k, i, j]),
                    name=f"time_ret_{k}_{i}_{j}"
                )
            else:
                M = max(0.0, tw_end[i] + tt - tw_start[j])
                m.addConstr(
                    t[k, j] >= t[k, i] + tt - M * (1 - x[k, i, j]),
                    name=f"time_{k}_{i}_{j}"
                )

    # Edge set access constraints: arc can be used only if set accessed
    if edge_set_ids:
        for (i, j) in arcs:
            sid = arc_set_id[(i, j)]
            if sid is not None:
                for k in K:
                    m.addConstr(x[k, i, j] <= y[sid], name=f"set_access_{k}_{i}_{j}_{sid}")

    # Objective
    travel_cost = gp.quicksum(float(distance[i][j]) * x[k, i, j] for k in K for (i, j) in arcs)
    fixed_cost = gp.quicksum(access_cost[sid] * y[sid] for sid in edge_set_ids) if edge_set_ids else 0.0
    m.setObjective(travel_cost + fixed_cost, GRB.MINIMIZE)

    # Incumbent logger callback
    class IncumbentCallback:
        def __init__(self, ext_logger):
            self.logger = ext_logger
            self.best = math.inf

        def __call__(self, model, where):
            if where == GRB.Callback.MIPSOL and self.logger is not None:
                obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if obj + 1e-9 < self.best:
                    self.best = obj
                    self.logger.log(float(obj))

    cb = IncumbentCallback(logger)
    m.optimize(cb)

    # Extract solution
    if m.SolCount == 0:
        return {
            "objective_value": 1e30,
            "routes": [],
            "edge_sets_used": []
        }

    objective_value = float(m.ObjVal)

    edge_sets_used = []
    if edge_set_ids:
        for sid in edge_set_ids:
            if y[sid].X > 0.5:
                edge_sets_used.append(int(sid))

    routes = []
    for k in K:
        if u[k].X <= 0.5:
            continue

        # Build successor map for this vehicle
        sel_succ = {}
        for (i, j) in arcs:
            if x[k, i, j].X > 0.5:
                sel_succ[i] = j

        if 0 not in sel_succ:
            continue

        seq = [0]
        arc_list = []
        cur = 0
        guard = 0
        while True:
            if cur not in sel_succ:
                break
            nxt = sel_succ[cur]
            arc_list.append([int(cur), int(nxt)])
            seq.append(int(nxt))
            cur = nxt
            guard += 1
            if cur == 0 or guard > (N + 5):
                break

        visit_times = {str(i): float(t[k, i].X) for i in node_ids}

        routes.append({
            "vehicle": int(k),
            "sequence": seq,
            "arcs": arc_list,
            "visit_times": visit_times
        })

    return {
        "objective_value": objective_value,
        "routes": routes,
        "edge_sets_used": edge_sets_used
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, required=False, default=None)
    args = parser.parse_args()

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    solution = build_and_solve(instance, args.time_limit, logger)

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()
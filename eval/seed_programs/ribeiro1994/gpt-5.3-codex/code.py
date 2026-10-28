import argparse
import json
import math
import time
from collections import deque, defaultdict

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def hopcroft_karp(n, adj):
    """Maximum bipartite matching on graph (U=trips, V=trips) with edges u->v."""
    pair_u = [-1] * n
    pair_v = [-1] * n
    dist = [0] * n
    INF = 10**18

    def bfs():
        q = deque()
        for u in range(n):
            if pair_u[u] == -1:
                dist[u] = 0
                q.append(u)
            else:
                dist[u] = INF
        found_augmenting = False
        while q:
            u = q.popleft()
            for v in adj[u]:
                pu = pair_v[v]
                if pu == -1:
                    found_augmenting = True
                elif dist[pu] == INF:
                    dist[pu] = dist[u] + 1
                    q.append(pu)
        return found_augmenting

    def dfs(u):
        for v in adj[u]:
            pu = pair_v[v]
            if pu == -1 or (dist[pu] == dist[u] + 1 and dfs(pu)):
                pair_u[u] = v
                pair_v[v] = u
                return True
        dist[u] = INF
        return False

    while bfs():
        for u in range(n):
            if pair_u[u] == -1:
                dfs(u)

    return pair_u, pair_v


def build_heuristic_start(n, m, succ_adj, arc_cost_map, start_cost, end_cost, capacities):
    """
    Build a feasible solution (if possible) via:
    1) min path cover in DAG via maximum matching
    2) assign each path to a depot greedily by cheapest start/end cost with capacities
    """
    match_u, match_v = hopcroft_karp(n, succ_adj)

    succ = [-1] * n
    pred = [-1] * n
    for u in range(n):
        v = match_u[u]
        if v != -1:
            succ[u] = v
            pred[v] = u

    visited = [False] * n
    paths = []
    for i in range(n):
        if pred[i] == -1:
            p = []
            cur = i
            while cur != -1 and not visited[cur]:
                visited[cur] = True
                p.append(cur)
                cur = succ[cur]
            if p:
                paths.append(p)

    # Safety for any leftover (shouldn't happen in DAG path cover)
    for i in range(n):
        if not visited[i]:
            p = []
            cur = i
            while cur != -1 and not visited[cur]:
                visited[cur] = True
                p.append(cur)
                cur = succ[cur]
            if p:
                paths.append(p)

    rem = capacities[:]
    path_depot = []
    for p in paths:
        first, last = p[0], p[-1]
        best_d, best_c = None, None
        for d in range(m):
            if rem[d] <= 0:
                continue
            c = start_cost[d][first] + end_cost[last][d]
            if best_c is None or c < best_c:
                best_c = c
                best_d = d
        if best_d is None:
            return None  # infeasible under capacities
        rem[best_d] -= 1
        path_depot.append(best_d)

    s_sel = set()
    t_sel = set()
    x_sel = set()
    total_obj = 0.0
    for p, d in zip(paths, path_depot):
        first, last = p[0], p[-1]
        s_sel.add((d, first))
        t_sel.add((d, last))
        total_obj += start_cost[d][first] + end_cost[last][d]
        for k in range(len(p) - 1):
            u, v = p[k], p[k + 1]
            x_sel.add((d, u, v))
            total_obj += arc_cost_map[(u, v)]

    return {
        "s_sel": s_sel,
        "t_sel": t_sel,
        "x_sel": x_sel,
        "objective": total_obj,
        "num_vehicles": len(paths),
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
        inst = json.load(f)

    trips = inst["trips"]
    n = len(trips)
    depots = inst["depots"]
    m = len(depots)
    capacities = list(inst["vehicle_capacities"])

    trip_ids = [t["trip_id"] for t in trips]
    id_to_idx = {tid: i for i, tid in enumerate(trip_ids)}

    # Start and end costs
    start_cost = [[10**12] * n for _ in range(m)]   # depot d -> trip i
    for rec in inst["depot_to_trip_costs"]:
        d = rec["depot"] - 1
        i = id_to_idx[rec["trip"]]
        start_cost[d][i] = int(rec["cost"])

    end_cost = [[10**12] * m for _ in range(n)]     # trip i -> depot d
    for rec in inst["trip_to_depot_costs"]:
        i = id_to_idx[rec["trip"]]
        d = rec["depot"] - 1
        end_cost[i][d] = int(rec["cost"])

    # Compatible arcs
    arc_list = []  # (u, v, cost)
    arc_cost_map = {}
    succ_adj = [[] for _ in range(n)]
    pred_arcs = [[] for _ in range(n)]   # arc indices ending at i
    succ_arcs = [[] for _ in range(n)]   # arc indices starting from i

    for rec in inst["compatible_arcs"]:
        u = id_to_idx[rec["from_trip"]]
        v = id_to_idx[rec["to_trip"]]
        c = int(rec["cost"])
        a = len(arc_list)
        arc_list.append((u, v, c))
        arc_cost_map[(u, v)] = c
        succ_adj[u].append(v)
        pred_arcs[v].append(a)
        succ_arcs[u].append(a)

    # Build model
    model = gp.Model("mdvsp_multi_depot")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = float(args.time_limit)

    A = len(arc_list)
    x = model.addVars(m, A, vtype=GRB.BINARY, name="x")    # depot-specific trip->trip arcs
    s = model.addVars(m, n, vtype=GRB.BINARY, name="s")    # depot start arc depot->trip
    t = model.addVars(m, n, vtype=GRB.BINARY, name="t")    # depot end arc trip->depot

    # Each trip has exactly one predecessor globally
    for i in range(n):
        model.addConstr(
            gp.quicksum(s[d, i] for d in range(m)) +
            gp.quicksum(x[d, a] for d in range(m) for a in pred_arcs[i]) == 1,
            name=f"in_{i}"
        )

    # Each trip has exactly one successor globally
    for i in range(n):
        model.addConstr(
            gp.quicksum(t[d, i] for d in range(m)) +
            gp.quicksum(x[d, a] for d in range(m) for a in succ_arcs[i]) == 1,
            name=f"out_{i}"
        )

    # Depot-consistent flow through each trip
    for d in range(m):
        for i in range(n):
            model.addConstr(
                s[d, i] + gp.quicksum(x[d, a] for a in pred_arcs[i]) ==
                t[d, i] + gp.quicksum(x[d, a] for a in succ_arcs[i]),
                name=f"bal_d{d}_i{i}"
            )

    # Depot vehicle capacities
    for d in range(m):
        model.addConstr(gp.quicksum(s[d, i] for i in range(n)) <= capacities[d], name=f"cap_{d}")

    obj = gp.LinExpr()
    for d in range(m):
        for i in range(n):
            obj += start_cost[d][i] * s[d, i]
            obj += end_cost[i][d] * t[d, i]
    for d in range(m):
        for a, (_, _, c) in enumerate(arc_list):
            obj += c * x[d, a]
    model.setObjective(obj, GRB.MINIMIZE)

    # Heuristic MIP start
    heuristic = build_heuristic_start(n, m, succ_adj, arc_cost_map, start_cost, end_cost, capacities)
    if heuristic is not None:
        # map (u,v) -> arc index
        arc_index = {(u, v): a for a, (u, v, _) in enumerate(arc_list)}
        for d, i in heuristic["s_sel"]:
            s[d, i].Start = 1.0
        for d, i in heuristic["t_sel"]:
            t[d, i].Start = 1.0
        for d, u, v in heuristic["x_sel"]:
            a = arc_index.get((u, v), None)
            if a is not None:
                x[d, a].Start = 1.0
        if logger:
            logger.log(float(heuristic["objective"]))

    best_logged = [float("inf")]

    def cb(model_cb, where):
        if where == GRB.Callback.MIPSOL and logger is not None:
            val = model_cb.cbGet(GRB.Callback.MIPSOL_OBJ)
            if val + 1e-6 < best_logged[0]:
                best_logged[0] = val
                logger.log(float(val))

    model.optimize(cb)

    if model.SolCount == 0:
        sol = {
            "objective_value": 1e18,
            "routes": [],
            "num_vehicles": 0
        }
        with open(args.solution_path, "w") as f:
            json.dump(sol, f, indent=2)
        return

    # Extract solution
    routes = []
    for d in range(m):
        starts = [i for i in range(n) if s[d, i].X > 0.5]
        succ_map = {}
        for a, (u, v, _) in enumerate(arc_list):
            if x[d, a].X > 0.5:
                succ_map[u] = v

        for st in starts:
            trip_seq_idx = []
            cur = st
            seen = set()
            while cur is not None and cur not in seen:
                seen.add(cur)
                trip_seq_idx.append(cur)
                nxt = succ_map.get(cur, None)
                cur = nxt
            routes.append({
                "depot": d + 1,
                "trips": [trip_ids[i] for i in trip_seq_idx]
            })

    solution = {
        "objective_value": float(model.ObjVal),
        "routes": routes,
        "num_vehicles": len(routes)
    }

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()
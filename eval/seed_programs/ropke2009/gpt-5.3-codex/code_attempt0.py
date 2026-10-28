import argparse
import json
import math
import time
from typing import Dict, List, Tuple, Optional

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def euclidean(a, b):
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


def find_matrix(data: dict, candidate_keys: List[str]):
    for k in candidate_keys:
        if k in data:
            return data[k]
    if "matrices" in data and isinstance(data["matrices"], dict):
        for k in candidate_keys:
            if k in data["matrices"]:
                return data["matrices"][k]
    return None


def build_cost_time(data: dict, nodes: List[dict], id_pos: Dict[int, int]):
    n = len(nodes)
    node_by_id = {nd["id"]: nd for nd in nodes}
    node_ids = [nd["id"] for nd in nodes]
    max_id = max(node_ids)

    cost_mat = find_matrix(
        data, ["routing_cost_matrix", "cost_matrix", "distance_matrix", "routing_costs"]
    )
    time_mat = find_matrix(
        data, ["travel_time_matrix", "time_matrix", "travel_times"]
    )

    cost = {}
    travel = {}

    def read_from_matrix(mat, i, j):
        # prefer row/col by node order
        if len(mat) == n and len(mat[0]) == n:
            return float(mat[id_pos[i]][id_pos[j]])
        # fallback contiguous IDs
        if max_id + 1 == len(mat) and len(mat[0]) == len(mat):
            return float(mat[i][j])
        return None

    for i in node_ids:
        for j in node_ids:
            if i == j:
                cij = 0.0
                tij = 0.0
            else:
                cij = None
                tij = None
                if cost_mat is not None:
                    cij = read_from_matrix(cost_mat, i, j)
                if time_mat is not None:
                    tij = read_from_matrix(time_mat, i, j)

                # fallback to Euclidean if matrix not available
                if cij is None:
                    cij = euclidean(node_by_id[i], node_by_id[j])
                if tij is None:
                    # travel time includes service time at origin node
                    tij = euclidean(node_by_id[i], node_by_id[j]) + float(node_by_id[i].get("service_time", 0))

            cost[(i, j)] = float(cij)
            travel[(i, j)] = float(tij)

    return cost, travel


def simulate_route(
    stops: List[int],
    origin: int,
    dest: int,
    e: Dict[int, float],
    l: Dict[int, float],
    demand: Dict[int, int],
    capacity: int,
    travel: Dict[Tuple[int, int], float],
    pair_pick_to_del: Dict[int, int],
    pair_del_to_pick: Dict[int, int],
):
    seq = [origin] + stops + [dest]

    # precedence check (deliveries after pickups)
    seen_pickups = set()
    for nid in stops:
        if nid in pair_pick_to_del:
            seen_pickups.add(nid)
        elif nid in pair_del_to_pick:
            if pair_del_to_pick[nid] not in seen_pickups:
                return False, float("inf")

    t = max(e[origin], 0.0)
    if t > l[origin]:
        return False, float("inf")

    load = 0
    total_cost_like = 0.0

    for idx in range(len(seq) - 1):
        i, j = seq[idx], seq[idx + 1]
        arr = t + travel[(i, j)]
        start = max(arr, e[j])
        if start > l[j] + 1e-9:
            return False, float("inf")
        t = start

        if j != origin and j != dest:
            load += demand[j]
            if load < -1e-9 or load > capacity + 1e-9:
                return False, float("inf")
        total_cost_like += travel[(i, j)]

    # with paired pickup-delivery on same route, end load should be 0
    if abs(load) > 1e-6:
        return False, float("inf")
    return True, total_cost_like


def greedy_initial_solution(
    num_vehicles: int,
    requests: List[dict],
    origin: int,
    dest: int,
    e: Dict[int, float],
    l: Dict[int, float],
    demand: Dict[int, int],
    capacity: int,
    travel: Dict[Tuple[int, int], float],
    cost: Dict[Tuple[int, int], float],
):
    pair_pick_to_del = {r["pickup_node"]: r["delivery_node"] for r in requests}
    pair_del_to_pick = {r["delivery_node"]: r["pickup_node"] for r in requests}

    routes = [[] for _ in range(num_vehicles)]  # stops only (no depots)

    # sort by tightness
    reqs = sorted(
        requests,
        key=lambda r: (r.get("pickup_time_window", [e[r["pickup_node"]], l[r["pickup_node"]]])[1], -r["load"]),
    )

    def route_cost(stops):
        seq = [origin] + stops + [dest]
        return sum(cost[(seq[i], seq[i + 1])] for i in range(len(seq) - 1))

    for r in reqs:
        p = r["pickup_node"]
        d = r["delivery_node"]
        best = None

        for k in range(num_vehicles):
            base = routes[k]
            m = len(base)
            base_cost = route_cost(base)

            for i in range(m + 1):
                for j in range(i + 1, m + 2):
                    cand = list(base)
                    cand.insert(i, p)
                    cand.insert(j, d)

                    ok, _ = simulate_route(
                        cand, origin, dest, e, l, demand, capacity, travel, pair_pick_to_del, pair_del_to_pick
                    )
                    if not ok:
                        continue

                    new_cost = route_cost(cand)
                    delta = new_cost - base_cost
                    if best is None or delta < best[0]:
                        best = (delta, k, cand)

        if best is None:
            return None
        _, best_k, best_route = best
        routes[best_k] = best_route

    # final feasibility check
    for k in range(num_vehicles):
        ok, _ = simulate_route(
            routes[k], origin, dest, e, l, demand, capacity, travel, pair_pick_to_del, pair_del_to_pick
        )
        if not ok:
            return None

    return routes


def build_and_solve(instance: dict, time_limit: int, logger: Optional[SolutionLogger]):
    start_time = time.time()

    num_vehicles = int(instance["num_vehicles"])
    vehicle_capacity = int(instance["vehicle_capacity"])

    nodes = instance["nodes"]
    node_ids = [nd["id"] for nd in nodes]
    id_pos = {nid: i for i, nid in enumerate(node_ids)}
    node_by_id = {nd["id"]: nd for nd in nodes}

    # identify depots
    origin_candidates = [nd["id"] for nd in nodes if nd.get("type") == "depot_origin"]
    dest_candidates = [nd["id"] for nd in nodes if nd.get("type") == "depot_destination"]

    if not origin_candidates:
        origin = node_ids[0]
    else:
        origin = origin_candidates[0]

    if not dest_candidates:
        dest = origin
    else:
        dest = dest_candidates[0]

    requests = instance["requests"]
    req_pairs = [(r["pickup_node"], r["delivery_node"]) for r in requests]
    pickup_nodes = [p for p, _ in req_pairs]
    delivery_nodes = [d for _, d in req_pairs]
    non_depot_nodes = [nid for nid in node_ids if nid not in (origin, dest)]

    demand = {nd["id"]: int(nd.get("demand", 0)) for nd in nodes}
    e = {nd["id"]: float(nd["time_window"][0]) for nd in nodes}
    l = {nd["id"]: float(nd["time_window"][1]) for nd in nodes}

    # if depot window exists in top-level and depot nodes are missing/tighter, apply cautiously
    if "depot" in instance and "time_window" in instance["depot"]:
        dep_e, dep_l = instance["depot"]["time_window"]
        e[origin] = max(e.get(origin, float(dep_e)), float(dep_e))
        l[origin] = min(l.get(origin, float(dep_l)), float(dep_l))
        if dest in e and dest in l:
            e[dest] = max(e[dest], float(dep_e))
            l[dest] = min(l[dest], float(dep_l))

    cost, travel = build_cost_time(instance, nodes, id_pos)

    pair_pick_to_del = {p: d for p, d in req_pairs}
    pair_del_to_pick = {d: p for p, d in req_pairs}

    # quick greedy for warm start
    init_routes = greedy_initial_solution(
        num_vehicles, requests, origin, dest, e, l, demand, vehicle_capacity, travel, cost
    )

    def routes_to_objective(routes):
        total = 0.0
        for r in routes:
            seq = [origin] + r + [dest]
            total += sum(cost[(seq[i], seq[i + 1])] for i in range(len(seq) - 1))
        return total

    if init_routes is not None and logger is not None:
        logger.log(float(routes_to_objective(init_routes)))

    # Build feasible arcs
    arcs = []
    for i in node_ids:
        if i == dest:
            continue
        for j in node_ids:
            if j == origin or i == j:
                continue
            # minimum-time feasibility filter
            if e[i] + travel[(i, j)] > l[j] + 1e-9:
                continue
            arcs.append((i, j))

    model = gp.Model("pdptw")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = max(1, int(time_limit))
    model.Params.MIPFocus = 1

    K = range(num_vehicles)

    x = model.addVars(
        [(k, i, j) for k in K for (i, j) in arcs],
        vtype=GRB.BINARY,
        name="x",
    )
    y = model.addVars(
        [(k, i) for k in K for i in non_depot_nodes],
        vtype=GRB.BINARY,
        name="y",
    )
    t = model.addVars(
        [(k, i) for k in K for i in node_ids],
        vtype=GRB.CONTINUOUS,
        lb=-GRB.INFINITY,
        name="t",
    )
    q = model.addVars(
        [(k, i) for k in K for i in node_ids],
        vtype=GRB.CONTINUOUS,
        lb=0.0,
        ub=vehicle_capacity,
        name="q",
    )

    # Objective: travel distance/cost
    model.setObjective(gp.quicksum(cost[(i, j)] * x[k, i, j] for k in K for (i, j) in arcs), GRB.MINIMIZE)

    # node visit exactly once (all pickups/deliveries)
    for i in non_depot_nodes:
        model.addConstr(gp.quicksum(y[k, i] for k in K) == 1, name=f"visit_once_{i}")

    # each vehicle departs origin once and arrives dest once
    for k in K:
        out_origin = gp.quicksum(x[k, i, j] for (i, j) in arcs if i == origin)
        in_dest = gp.quicksum(x[k, i, j] for (i, j) in arcs if j == dest)
        model.addConstr(out_origin == 1, name=f"dep_{k}")
        model.addConstr(in_dest == 1, name=f"ret_{k}")

    # flow conservation at non-depot nodes
    for k in K:
        for i in non_depot_nodes:
            in_i = gp.quicksum(x[k, u, v] for (u, v) in arcs if v == i)
            out_i = gp.quicksum(x[k, u, v] for (u, v) in arcs if u == i)
            model.addConstr(in_i == y[k, i], name=f"in_link_{k}_{i}")
            model.addConstr(out_i == y[k, i], name=f"out_link_{k}_{i}")

    # pickup-delivery same vehicle
    for p, d in req_pairs:
        for k in K:
            model.addConstr(y[k, p] == y[k, d], name=f"pair_{k}_{p}_{d}")

    # Time windows and depot time
    min_e = min(e.values())
    max_l = max(l.values())
    max_tt = max(travel.values()) if travel else 0.0
    M_time = (max_l - min_e) + max_tt + 10.0

    for k in K:
        # Fix departure time at depot earliest (waiting at depot is equivalent to waiting later)
        model.addConstr(t[k, origin] == e[origin], name=f"t_origin_{k}")
        model.addConstr(q[k, origin] == 0.0, name=f"q_origin_{k}")
        model.addConstr(q[k, dest] == 0.0, name=f"q_dest_{k}")

        for i in non_depot_nodes:
            model.addConstr(t[k, i] >= e[i] - M_time * (1 - y[k, i]), name=f"tw_lb_{k}_{i}")
            model.addConstr(t[k, i] <= l[i] + M_time * (1 - y[k, i]), name=f"tw_ub_{k}_{i}")

        # depot destination time window
        model.addConstr(t[k, dest] >= e[dest], name=f"tw_dest_lb_{k}")
        model.addConstr(t[k, dest] <= l[dest], name=f"tw_dest_ub_{k}")

    # time propagation on arcs
    for k in K:
        for (i, j) in arcs:
            M_ij = max(0.0, l[i] + travel[(i, j)] - e[j]) + 1.0
            model.addConstr(
                t[k, j] >= t[k, i] + travel[(i, j)] - M_ij * (1 - x[k, i, j]),
                name=f"time_{k}_{i}_{j}",
            )

    # load propagation
    max_dem_abs = max(abs(v) for v in demand.values()) if demand else 0
    M_load = vehicle_capacity + max_dem_abs + 1

    for k in K:
        for (i, j) in arcs:
            dj = demand.get(j, 0)
            model.addConstr(
                q[k, j] >= q[k, i] + dj - M_load * (1 - x[k, i, j]),
                name=f"load_lb_{k}_{i}_{j}",
            )
            model.addConstr(
                q[k, j] <= q[k, i] + dj + M_load * (1 - x[k, i, j]),
                name=f"load_ub_{k}_{i}_{j}",
            )

    # precedence pickup before delivery on same vehicle
    for p, d in req_pairs:
        for k in K:
            model.addConstr(
                t[k, d] >= t[k, p] - M_time * (1 - y[k, p]),
                name=f"prec_{k}_{p}_{d}",
            )

    # warm start
    if init_routes is not None:
        for k in K:
            # set default starts
            for i in non_depot_nodes:
                y[k, i].Start = 0.0

        for k, stops in enumerate(init_routes):
            seq = [origin] + stops + [dest]
            visited = set(stops)
            for i in visited:
                y[k, i].Start = 1.0
            for a, b in zip(seq[:-1], seq[1:]):
                if (a, b) in arcs:
                    x[k, a, b].Start = 1.0

    # incumbent logger callback
    model._logger = logger
    model._best_obj = float("inf")

    def cb(m, where):
        if where == GRB.Callback.MIPSOL:
            obj = m.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj + 1e-6 < m._best_obj:
                m._best_obj = obj
                if m._logger is not None:
                    m._logger.log(float(obj))

    remaining = max(1, int(time_limit - (time.time() - start_time)))
    model.Params.TimeLimit = remaining
    model.optimize(cb)

    # Extract best solution
    best_routes = None

    if model.SolCount > 0:
        best_routes = []
        for k in K:
            succ = {}
            for (i, j) in arcs:
                val = x[k, i, j].X
                if val > 0.5:
                    succ[i] = j

            route = [origin]
            cur = origin
            seen = {origin}
            for _ in range(len(node_ids) + 5):
                if cur not in succ:
                    break
                nxt = succ[cur]
                route.append(nxt)
                cur = nxt
                if cur == dest:
                    break
                if cur in seen:
                    # break cycle if any
                    break
                seen.add(cur)

            if route[-1] != dest:
                route.append(dest)

            best_routes.append(route)
    else:
        # fallback to greedy (if available)
        if init_routes is not None:
            best_routes = [[origin] + r + [dest] for r in init_routes]
        else:
            # minimal safe fallback
            best_routes = [[origin, dest] for _ in range(num_vehicles)]

    # objective value requested by schema: total routing distance
    objective_value = 0.0
    for route in best_routes:
        for i in range(len(route) - 1):
            objective_value += cost[(route[i], route[i + 1])]

    out_routes = []
    for k, route in enumerate(best_routes):
        out_routes.append({"vehicle": int(k), "nodes": [int(v) for v in route]})

    return {"objective_value": float(objective_value), "routes": out_routes}


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
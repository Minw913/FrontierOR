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


def build_distance_matrix(nodes_by_id: Dict[int, dict], node_ids: List[int]) -> Dict[Tuple[int, int], float]:
    dist = {}
    for i in node_ids:
        for j in node_ids:
            if i == j:
                dist[(i, j)] = 0.0
            else:
                dist[(i, j)] = euclidean(nodes_by_id[i], nodes_by_id[j])
    return dist


def compute_route_schedule_cost(
    route_customers: List[int],
    source: int,
    sink: int,
    nodes: Dict[int, dict],
    dist: Dict[Tuple[int, int], float],
    Q: float,
    H: float,
    arc_allowed: set,
):
    load = 0.0
    t = 0.0
    prev = source
    cost = 0.0
    start_times = {}

    for c in route_customers:
        if (prev, c) not in arc_allowed:
            return False, None, None
        load += nodes[c]["demand"]
        if load > Q + 1e-9:
            return False, None, None

        arrive = t + dist[(prev, c)]
        start = max(arrive, nodes[c]["time_window_open"])
        if start > nodes[c]["time_window_close"] + 1e-9:
            return False, None, None

        start_times[c] = start
        t = start + nodes[c]["service_time"]
        cost += dist[(prev, c)]
        prev = c

    if (prev, sink) not in arc_allowed:
        return False, None, None

    ret = t + dist[(prev, sink)]
    if ret > H + 1e-9:
        return False, None, None

    cost += dist[(prev, sink)]
    return True, cost, start_times


def initial_solution_insertion(
    customers: List[int],
    source: int,
    sink: int,
    nodes: Dict[int, dict],
    dist: Dict[Tuple[int, int], float],
    Q: float,
    H: float,
    arc_allowed: set,
):
    # Solomon-style greedy insertion into existing routes
    ordered = sorted(customers, key=lambda c: (nodes[c]["time_window_close"], nodes[c]["time_window_open"]))
    routes = []  # list of customer sequences
    route_costs = []

    for c in ordered:
        best = None  # (delta, r_idx, pos, new_cost)
        for r_idx, r in enumerate(routes):
            old_cost = route_costs[r_idx]
            for pos in range(len(r) + 1):
                cand = r[:pos] + [c] + r[pos:]
                feas, new_cost, _ = compute_route_schedule_cost(cand, source, sink, nodes, dist, Q, H, arc_allowed)
                if not feas:
                    continue
                delta = new_cost - old_cost
                if (best is None) or (delta < best[0]):
                    best = (delta, r_idx, pos, new_cost)

        if best is None:
            feas, cst, _ = compute_route_schedule_cost([c], source, sink, nodes, dist, Q, H, arc_allowed)
            if not feas:
                return None, None
            routes.append([c])
            route_costs.append(cst)
        else:
            _, r_idx, pos, new_cost = best
            routes[r_idx] = routes[r_idx][:pos] + [c] + routes[r_idx][pos:]
            route_costs[r_idx] = new_cost

    return routes, route_costs


def local_search_relocate(
    routes: List[List[int]],
    route_costs: List[float],
    source: int,
    sink: int,
    nodes: Dict[int, dict],
    dist: Dict[Tuple[int, int], float],
    Q: float,
    H: float,
    arc_allowed: set,
    end_time: float,
):
    improved = True
    while improved and time.time() < end_time:
        improved = False
        n_routes = len(routes)

        for r1 in range(n_routes):
            if time.time() >= end_time:
                break
            if len(routes[r1]) == 0:
                continue

            for i in range(len(routes[r1])):
                if time.time() >= end_time:
                    break
                cust = routes[r1][i]
                base_r1 = routes[r1][:i] + routes[r1][i + 1 :]

                feas1, cost1, _ = compute_route_schedule_cost(base_r1, source, sink, nodes, dist, Q, H, arc_allowed)
                if not feas1:
                    continue

                for r2 in range(n_routes):
                    if time.time() >= end_time:
                        break
                    # allow moving into same route at different position too
                    for pos in range(len(routes[r2]) + 1):
                        if r1 == r2 and (pos == i or pos == i + 1):
                            continue

                        if r1 == r2:
                            rr = routes[r2][:]
                            rr.pop(i)
                            adj_pos = pos if pos <= i else pos - 1
                            rr = rr[:adj_pos] + [cust] + rr[adj_pos:]
                            feas2, cost2, _ = compute_route_schedule_cost(rr, source, sink, nodes, dist, Q, H, arc_allowed)
                            if not feas2:
                                continue
                            old = route_costs[r1]
                            new = cost2
                            if new < old - 1e-6:
                                routes[r1] = rr
                                route_costs[r1] = new
                                improved = True
                                break
                        else:
                            new_r2 = routes[r2][:pos] + [cust] + routes[r2][pos:]
                            feas2, cost2, _ = compute_route_schedule_cost(new_r2, source, sink, nodes, dist, Q, H, arc_allowed)
                            if not feas2:
                                continue

                            old_total = route_costs[r1] + route_costs[r2]
                            new_total = cost1 + cost2
                            if new_total < old_total - 1e-6:
                                routes[r1] = base_r1
                                route_costs[r1] = cost1
                                routes[r2] = new_r2
                                route_costs[r2] = cost2
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                # remove empty routes
                kept_routes = []
                kept_costs = []
                for rr, cc in zip(routes, route_costs):
                    if len(rr) > 0:
                        kept_routes.append(rr)
                        kept_costs.append(cc)
                routes, route_costs = kept_routes, kept_costs
                break

    return routes, route_costs


def routes_to_full(routes_customers: List[List[int]], source: int, sink: int) -> List[List[int]]:
    return [[source] + r + [sink] for r in routes_customers]


def compute_total_cost(full_routes: List[List[int]], dist: Dict[Tuple[int, int], float]) -> float:
    total = 0.0
    for r in full_routes:
        for i in range(len(r) - 1):
            total += dist[(r[i], r[i + 1])]
    return total


def reconstruct_routes_from_x(
    x_sol: Dict[Tuple[int, int], float],
    source: int,
    sink: int,
    customers: List[int],
):
    succ = {}
    starts = []
    for (i, j), v in x_sol.items():
        if v > 0.5:
            if i == source:
                starts.append(j)
            elif i in customers:
                succ[i] = j

    routes = []
    for st in sorted(starts):
        route = [source, st]
        cur = st
        guard = 0
        while cur != sink and guard <= len(customers) + 5:
            nxt = succ.get(cur, None)
            if nxt is None:
                break
            route.append(nxt)
            cur = nxt
            guard += 1
        routes.append(route)
    return routes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()
    end_time = start_time + args.time_limit
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    params = inst["problem_parameters"]
    Q = float(params["vehicle_capacity"])
    H = float(params["planning_horizon"])

    nodes_list = inst["nodes"]
    nodes = {n["id"]: n for n in nodes_list}
    node_ids = [n["id"] for n in nodes_list]

    source = None
    sink = None
    customers = []
    for n in nodes_list:
        if n["type"] == "depot_source":
            source = n["id"]
        elif n["type"] == "depot_sink":
            sink = n["id"]
        elif n["type"] == "customer":
            customers.append(n["id"])

    if source is None or sink is None:
        raise ValueError("Instance must contain one depot_source and one depot_sink.")

    dist = build_distance_matrix(nodes, node_ids)

    # Build feasible directed arcs
    arc_allowed = set()
    for i in node_ids:
        if i == sink:
            continue
        for j in node_ids:
            if j == source:
                continue
            if i == j:
                continue
            if i == source and j == sink:
                continue

            # Pair-demand compatibility
            if nodes[i]["demand"] + nodes[j]["demand"] > Q + 1e-9:
                continue

            # Earliest-start reachability check
            earliest_i = nodes[i]["time_window_open"]
            if earliest_i + nodes[i]["service_time"] + dist[(i, j)] > nodes[j]["time_window_close"] + 1e-9:
                continue

            arc_allowed.add((i, j))

    # Heuristic incumbent
    best_routes_full = []
    best_obj = float("inf")

    if len(customers) == 0:
        best_routes_full = []
        best_obj = 0.0
        if logger:
            logger.log(best_obj)
    else:
        h_routes, h_costs = initial_solution_insertion(customers, source, sink, nodes, dist, Q, H, arc_allowed)
        if h_routes is not None:
            h_routes, h_costs = local_search_relocate(
                h_routes,
                h_costs,
                source,
                sink,
                nodes,
                dist,
                Q,
                H,
                arc_allowed,
                end_time=min(end_time, time.time() + max(1.0, 0.2 * args.time_limit)),
            )
            full = routes_to_full(h_routes, source, sink)
            obj = sum(h_costs)
            best_routes_full = full
            best_obj = obj
            if logger:
                logger.log(best_obj)

    # Build and solve MILP (if time remains)
    rem = end_time - time.time()
    if rem > 0.5 and len(customers) > 0:
        model = gp.Model("VRPTW")
        model.Params.OutputFlag = 0
        model.Params.Threads = 1
        model.Params.TimeLimit = rem

        A = list(arc_allowed)
        x = model.addVars(A, vtype=GRB.BINARY, name="x")

        # time vars
        t = {}
        for i in node_ids:
            lb = nodes[i]["time_window_open"]
            ub = nodes[i]["time_window_close"]
            if i == source:
                t[i] = model.addVar(lb=0.0, ub=0.0, vtype=GRB.CONTINUOUS, name=f"t_{i}")
            else:
                t[i] = model.addVar(lb=lb, ub=ub, vtype=GRB.CONTINUOUS, name=f"t_{i}")

        # load vars for customers
        u = {}
        for i in customers:
            u[i] = model.addVar(lb=nodes[i]["demand"], ub=Q, vtype=GRB.CONTINUOUS, name=f"u_{i}")

        model.update()

        # Degree constraints
        incoming = {j: [] for j in customers}
        outgoing = {i: [] for i in customers}
        out_source = []
        in_sink = []

        for (i, j) in A:
            if j in customers:
                incoming[j].append((i, j))
            if i in customers:
                outgoing[i].append((i, j))
            if i == source:
                out_source.append((i, j))
            if j == sink:
                in_sink.append((i, j))

        for j in customers:
            model.addConstr(gp.quicksum(x[a] for a in incoming[j]) == 1, name=f"in_{j}")
            model.addConstr(gp.quicksum(x[a] for a in outgoing[j]) == 1, name=f"out_{j}")

        model.addConstr(gp.quicksum(x[a] for a in out_source) == gp.quicksum(x[a] for a in in_sink), name="fleet_balance")

        # Time precedence
        for (i, j) in A:
            M = max(
                0.0,
                nodes[i]["time_window_close"] + nodes[i]["service_time"] + dist[(i, j)] - nodes[j]["time_window_open"],
            )
            model.addConstr(
                t[j] >= t[i] + nodes[i]["service_time"] + dist[(i, j)] - M * (1 - x[(i, j)]),
                name=f"time_{i}_{j}",
            )

        # Capacity propagation
        for (i, j) in A:
            if i in customers and j in customers:
                model.addConstr(
                    u[j] >= u[i] + nodes[j]["demand"] - Q * (1 - x[(i, j)]),
                    name=f"load_{i}_{j}",
                )

        # Objective
        model.setObjective(gp.quicksum(dist[(i, j)] * x[(i, j)] for (i, j) in A), GRB.MINIMIZE)

        # MIP start from heuristic
        if best_obj < float("inf") and best_routes_full:
            for a in A:
                x[a].Start = 0.0
            for route in best_routes_full:
                for k in range(len(route) - 1):
                    a = (route[k], route[k + 1])
                    if a in x:
                        x[a].Start = 1.0

            max_arrival_sink = 0.0
            for route in best_routes_full:
                tt = 0.0
                load_val = 0.0
                for k in range(1, len(route) - 1):
                    prev = route[k - 1]
                    cur = route[k]
                    tt = max(tt + dist[(prev, cur)], nodes[cur]["time_window_open"])
                    if cur in t:
                        t[cur].Start = tt
                    if cur in u:
                        load_val += nodes[cur]["demand"]
                        u[cur].Start = load_val
                    tt += nodes[cur]["service_time"]
                last = route[-2]
                arr_sink = tt + dist[(last, sink)]
                max_arrival_sink = max(max_arrival_sink, arr_sink)
            if sink in t:
                t[sink].Start = min(max_arrival_sink, nodes[sink]["time_window_close"])

        incumbent_holder = [best_obj]

        def cb(m, where):
            if where == GRB.Callback.MIPSOL:
                try:
                    obj = m.cbGet(GRB.Callback.MIPSOL_OBJ)
                    if obj + 1e-6 < incumbent_holder[0]:
                        incumbent_holder[0] = obj
                        if logger:
                            logger.log(float(obj))
                except gp.GurobiError:
                    pass

        model.optimize(cb)

        if model.SolCount > 0:
            x_sol = {a: x[a].X for a in A}
            cand_routes = reconstruct_routes_from_x(x_sol, source, sink, customers)

            # validate coverage
            served = []
            for r in cand_routes:
                served.extend([n for n in r if n in customers])
            if len(served) == len(customers) and len(set(served)) == len(customers):
                cand_obj = compute_total_cost(cand_routes, dist)
                if cand_obj < best_obj - 1e-6:
                    best_obj = cand_obj
                    best_routes_full = cand_routes

    # Final fallback for infeasible / no solution
    if len(customers) > 0 and (best_obj == float("inf") or not best_routes_full):
        # keep schema-valid output
        solution = {
            "objective_value": 1e30,
            "routes": []
        }
    else:
        solution = {
            "objective_value": float(best_obj),
            "routes": best_routes_full
        }

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()
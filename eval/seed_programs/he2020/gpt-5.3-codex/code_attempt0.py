import argparse
import json
from typing import Dict, List, Tuple, Optional

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def read_instance(path: str):
    with open(path, "r") as f:
        return json.load(f)


def build_route_from_x(n: int, end_node: int, x_vals: Dict[Tuple[int, int], float]) -> List[List[int]]:
    succ = {}
    for (i, j), v in x_vals.items():
        if v > 0.5:
            succ.setdefault(i, []).append(j)

    starts = sorted(succ.get(0, []))
    routes = []
    for s in starts:
        route = [0, s]
        cur = s
        seen = {0, s}
        safety = 0
        while cur != end_node and safety <= n + 5:
            safety += 1
            nxts = succ.get(cur, [])
            if not nxts:
                break
            nxt = nxts[0]
            route.append(nxt)
            if nxt in seen and nxt != end_node:
                break
            seen.add(nxt)
            cur = nxt
        routes.append(route)
    return routes


def make_solution_dict(
    n: int,
    end_node: int,
    customer_ids_by_index: Dict[int, int],
    x_vals: Dict[Tuple[int, int], float],
    t_vals: Dict[int, float],
    objective_value: float,
):
    routes = build_route_from_x(n, end_node, x_vals)
    schedule = {}
    for i in range(1, n + 1):
        cid = customer_ids_by_index[i]
        schedule[str(cid)] = float(t_vals.get(i, 0.0))
    return {
        "objective_value": float(objective_value),
        "routes": routes,
        "schedule": schedule,
    }


def earliest_schedule_feasible(
    seq: List[int],
    earliest: Dict[int, float],
    latest: Dict[int, float],
    service: Dict[int, float],
    travel: List[List[float]],
    end_node: int,
    cap_demand: Dict[int, int],
    Q: int,
):
    load = sum(cap_demand[c] for c in seq)
    if load > Q:
        return False, None

    t = earliest[0]
    prev = 0
    sched = {}
    for c in seq:
        t = max(earliest[c], t + service[prev] + travel[prev][c])
        if t > latest[c]:
            return False, None
        sched[c] = t
        prev = c

    if seq:
        ret = t + service[prev] + travel[prev][end_node]
    else:
        ret = earliest[0]
    if ret > latest[end_node]:
        return False, None
    return True, sched


def build_greedy_start(
    n: int,
    K: int,
    Q: int,
    travel: List[List[float]],
    route_cost: List[List[float]],
    demands: Dict[int, int],
    earliest: Dict[int, float],
    latest: Dict[int, float],
    service: Dict[int, float],
    preferred: Dict[int, float],
    rho: float,
    end_node: int,
):
    # Initial singleton routes
    routes = []
    for c in range(1, n + 1):
        feas, _ = earliest_schedule_feasible(
            [c], earliest, latest, service, travel, end_node, demands, Q
        )
        if feas:
            routes.append([c])

    if len(routes) == 0:
        return None

    # Greedy merges (end of r1 with start of r2)
    improved = True
    while improved and len(routes) > K:
        improved = False
        best_pair = None
        best_saving = -1e18

        for i in range(len(routes)):
            for j in range(len(routes)):
                if i == j:
                    continue
                r1 = routes[i]
                r2 = routes[j]
                merged = r1 + r2
                feas, _ = earliest_schedule_feasible(
                    merged, earliest, latest, service, travel, end_node, demands, Q
                )
                if not feas:
                    continue

                old = (
                    route_cost[0][r1[0]]
                    + sum(route_cost[r1[k]][r1[k + 1]] for k in range(len(r1) - 1))
                    + route_cost[r1[-1]][end_node]
                    + route_cost[0][r2[0]]
                    + sum(route_cost[r2[k]][r2[k + 1]] for k in range(len(r2) - 1))
                    + route_cost[r2[-1]][end_node]
                )
                new = (
                    route_cost[0][merged[0]]
                    + sum(route_cost[merged[k]][merged[k + 1]] for k in range(len(merged) - 1))
                    + route_cost[merged[-1]][end_node]
                )
                saving = old - new
                if saving > best_saving:
                    best_saving = saving
                    best_pair = (i, j, merged)

        if best_pair is not None:
            i, j, merged = best_pair
            new_routes = []
            for idx, r in enumerate(routes):
                if idx not in (i, j):
                    new_routes.append(r)
            new_routes.append(merged)
            routes = new_routes
            improved = True

    if len(routes) > K:
        return None

    # Build starts
    x_start = {}
    t_start = {}
    for r in routes:
        if not r:
            continue
        x_start[(0, r[0])] = 1
        for k in range(len(r) - 1):
            x_start[(r[k], r[k + 1])] = 1
        x_start[(r[-1], end_node)] = 1

        _, sched = earliest_schedule_feasible(
            r, earliest, latest, service, travel, end_node, demands, Q
        )
        if sched:
            for c, tt in sched.items():
                t_start[c] = tt

    # rough objective (for diagnostics only)
    obj = 0.0
    for (i, j), v in x_start.items():
        if v > 0.5:
            obj += route_cost[i][j]
    for c in range(1, n + 1):
        tc = t_start.get(c, preferred[c])
        obj += rho * (tc - preferred[c]) * (tc - preferred[c])

    return x_start, t_start, obj


def solve(instance: dict, time_limit: int, logger: Optional[SolutionLogger] = None):
    n = int(instance["num_customers"])
    K = int(instance["num_vehicles"])
    Q = int(instance["vehicle_capacity"])
    end_node = n + 1

    depot = instance["depot"]
    depot_end = instance["depot_end"]
    customers = instance["customers"]
    dist = instance["distance_matrix"]

    rho = 1.0
    if "inconvenience_cost_function" in instance:
        rho = float(instance["inconvenience_cost_function"].get("rho", 1.0))

    # index-based data (matrix index 1..n corresponds to customers list order)
    customer_ids_by_index = {i: int(customers[i - 1]["id"]) for i in range(1, n + 1)}
    demands = {0: 0, end_node: 0}
    earliest = {
        0: float(depot["time_window"][0]),
        end_node: float(depot_end["time_window"][0]),
    }
    latest = {
        0: float(depot["time_window"][1]),
        end_node: float(depot_end["time_window"][1]),
    }
    service = {0: 0.0, end_node: 0.0}
    preferred = {}

    global_service = float(instance.get("service_time", 0))
    for i in range(1, n + 1):
        c = customers[i - 1]
        demands[i] = int(c["demand"])
        tw = c["time_window"]
        earliest[i] = float(tw[0])
        latest[i] = float(tw[1])
        service[i] = float(c.get("service_time", global_service))
        preferred[i] = float(c["preferred_time"])

    nodes = list(range(0, n + 2))
    cust_nodes = list(range(1, n + 1))

    # Build feasible arc set
    arcs = []
    big_m = {}
    for i in nodes:
        if i == end_node:
            continue
        for j in nodes:
            if j == 0:
                continue
            if i == j:
                continue
            if i == 0 and j == end_node:
                continue  # avoid empty route symmetry

            # quick feasibility screen using earliest departure at i
            if earliest[i] + service[i] + float(dist[i][j]) > latest[j] + 1e-9:
                continue

            arcs.append((i, j))
            mij = latest[i] + service[i] + float(dist[i][j]) - earliest[j]
            big_m[(i, j)] = max(0.0, mij)

    # Basic sanity check for connectivity
    in_count = {j: 0 for j in cust_nodes}
    out_count = {i: 0 for i in cust_nodes}
    for i, j in arcs:
        if j in in_count:
            in_count[j] += 1
        if i in out_count:
            out_count[i] += 1
    for c in cust_nodes:
        if in_count[c] == 0 or out_count[c] == 0:
            # No feasible route due to time windows / graph
            sol = {"objective_value": float("inf"), "routes": [], "schedule": {}}
            return sol

    model = gp.Model("VRPTW_QuadraticInconvenience")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = float(time_limit)
    model.Params.OutputFlag = 0

    x = model.addVars(arcs, vtype=GRB.BINARY, name="x")
    t = model.addVars(nodes, vtype=GRB.CONTINUOUS, name="t")
    u = model.addVars(cust_nodes, vtype=GRB.CONTINUOUS, name="u")

    # Time bounds
    t[0].lb = earliest[0]
    t[0].ub = earliest[0]  # fixed departure reference (waiting can happen implicitly)
    t[end_node].lb = earliest[end_node]
    t[end_node].ub = latest[end_node]
    for i in cust_nodes:
        t[i].lb = earliest[i]
        t[i].ub = latest[i]

    # Degree constraints
    for i in cust_nodes:
        model.addConstr(gp.quicksum(x[i, j] for (ii, j) in arcs if ii == i) == 1, name=f"out_{i}")
        model.addConstr(gp.quicksum(x[j, i] for (j, jj) in arcs if jj == i) == 1, name=f"in_{i}")

    model.addConstr(
        gp.quicksum(x[0, j] for (i, j) in arcs if i == 0) <= K,
        name="veh_out_limit",
    )
    model.addConstr(
        gp.quicksum(x[i, end_node] for (i, j) in arcs if j == end_node) <= K,
        name="veh_in_limit",
    )
    model.addConstr(
        gp.quicksum(x[0, j] for (i, j) in arcs if i == 0)
        == gp.quicksum(x[i, end_node] for (i, j) in arcs if j == end_node),
        name="veh_balance",
    )

    # Capacity / MTZ-like load constraints
    for i in cust_nodes:
        model.addConstr(u[i] >= demands[i], name=f"u_lb_{i}")
        model.addConstr(u[i] <= Q, name=f"u_ub_{i}")

    for (i, j) in arcs:
        if j in cust_nodes:
            if i == 0:
                model.addConstr(
                    u[j] >= demands[j] - Q * (1 - x[i, j]),
                    name=f"load_0_{j}",
                )
            elif i in cust_nodes:
                model.addConstr(
                    u[j] >= u[i] + demands[j] - Q * (1 - x[i, j]),
                    name=f"load_{i}_{j}",
                )

    # Time precedence constraints
    for (i, j) in arcs:
        model.addConstr(
            t[j] >= t[i] + service[i] + float(dist[i][j]) - big_m[(i, j)] * (1 - x[i, j]),
            name=f"time_{i}_{j}",
        )

    # Objective
    routing_expr = gp.quicksum(float(dist[i][j]) * x[i, j] for (i, j) in arcs)
    inconvenience_expr = gp.quicksum(
        rho * (t[i] - preferred[i]) * (t[i] - preferred[i]) for i in cust_nodes
    )
    model.setObjective(routing_expr + inconvenience_expr, GRB.MINIMIZE)

    # Warm start (optional)
    warm = build_greedy_start(
        n=n,
        K=K,
        Q=Q,
        travel=dist,
        route_cost=dist,
        demands=demands,
        earliest=earliest,
        latest=latest,
        service=service,
        preferred=preferred,
        rho=rho,
        end_node=end_node,
    )
    if warm is not None:
        x_start, t_start, _ = warm
        for (i, j) in arcs:
            x[i, j].Start = 1.0 if (i, j) in x_start else 0.0
        t[0].Start = earliest[0]
        t[end_node].Start = min(latest[end_node], max(earliest[end_node], earliest[0]))
        for i in cust_nodes:
            if i in t_start:
                t[i].Start = min(latest[i], max(earliest[i], t_start[i]))
            else:
                t[i].Start = preferred[i]

    # Callback for incumbent logging
    model._logger = logger
    model._best_logged = float("inf")
    model._arcs = arcs
    model._x_var_list = [x[a] for a in arcs]
    model._t_var_list = [t[i] for i in nodes]
    model._nodes = nodes
    model._n = n
    model._end_node = end_node
    model._customer_ids_by_index = customer_ids_by_index

    def incumbent_cb(m, where):
        if where == GRB.Callback.MIPSOL:
            if m._logger is None:
                return
            try:
                obj = float(m.cbGet(GRB.Callback.MIPSOL_OBJ))
                if obj + 1e-9 >= m._best_logged:
                    return

                xv = m.cbGetSolution(m._x_var_list)
                tv = m.cbGetSolution(m._t_var_list)

                xvals = {}
                for idx, a in enumerate(m._arcs):
                    xvals[a] = xv[idx]

                tvals = {}
                for idx, node in enumerate(m._nodes):
                    tvals[node] = tv[idx]

                sol_dict = make_solution_dict(
                    n=m._n,
                    end_node=m._end_node,
                    customer_ids_by_index=m._customer_ids_by_index,
                    x_vals=xvals,
                    t_vals=tvals,
                    objective_value=obj,
                )
                m._logger.log_solution(obj, sol_dict)
                m._best_logged = obj
            except Exception:
                # Never break optimization due to logging issues
                pass

    model.optimize(incumbent_cb)

    if model.SolCount == 0:
        return {"objective_value": float("inf"), "routes": [], "schedule": {}}

    x_vals = {(i, j): x[i, j].X for (i, j) in arcs}
    t_vals = {i: t[i].X for i in nodes}
    obj = float(model.ObjVal)

    sol = make_solution_dict(
        n=n,
        end_node=end_node,
        customer_ids_by_index=customer_ids_by_index,
        x_vals=x_vals,
        t_vals=t_vals,
        objective_value=obj,
    )
    return sol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, required=False, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    instance = read_instance(args.instance_path)
    solution = solve(instance, args.time_limit, logger=logger)

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()
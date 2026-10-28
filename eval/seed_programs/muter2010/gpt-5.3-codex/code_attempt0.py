import argparse
import json
import math
import random
import time
from typing import List, Tuple, Dict, Optional

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def euclidean(a, b):
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


class BestTracker:
    def __init__(self, logger: Optional[SolutionLogger] = None):
        self.best = float("inf")
        self.logger = logger

    def consider(self, obj: float):
        if obj + 1e-9 < self.best:
            self.best = obj
            if self.logger:
                self.logger.log(float(obj))


def build_distance_matrix(nodes: List[Dict]) -> List[List[float]]:
    n = len(nodes)
    d = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            dist = euclidean(nodes[i], nodes[j])
            d[i][j] = dist
            d[j][i] = dist
    return d


def route_distance(route: List[int], dist: List[List[float]]) -> float:
    if not route:
        return 0.0
    total = dist[0][route[0]]
    for i in range(len(route) - 1):
        total += dist[route[i]][route[i + 1]]
    total += dist[route[-1]][0]
    return total


def check_route_feasible(
    route: List[int],
    dist: List[List[float]],
    demand: List[int],
    ready: List[int],
    due: List[int],
    service: List[int],
    capacity: int,
) -> bool:
    load = sum(demand[i] for i in route)
    if load > capacity:
        return False

    t = ready[0]  # depot earliest departure
    prev = 0
    for c in route:
        arr = t + dist[prev][c]
        if arr > due[c] + 1e-9:
            return False
        t = max(arr, ready[c]) + service[c]
        prev = c
    # return to depot by depot due_date
    if t + dist[prev][0] > due[0] + 1e-9:
        return False
    return True


def greedy_construct(
    customer_indices: List[int],
    dist: List[List[float]],
    demand: List[int],
    ready: List[int],
    due: List[int],
    service: List[int],
    capacity: int,
    max_routes: int,
    randomized: bool = False,
) -> List[List[int]]:
    unserved = set(customer_indices)
    routes: List[List[int]] = []

    while unserved and len(routes) < max_routes:
        # Seed: earliest due date (or random among top few)
        cand = sorted(unserved, key=lambda c: (due[c], dist[0][c]))
        if randomized and len(cand) > 1:
            seed = random.choice(cand[: min(4, len(cand))])
        else:
            seed = cand[0]

        route = [seed]
        if not check_route_feasible(route, dist, demand, ready, due, service, capacity):
            # If single customer route infeasible, skip (instance likely infeasible)
            unserved.remove(seed)
            continue
        unserved.remove(seed)

        while True:
            insertions = []
            base_dist = route_distance(route, dist)
            base_load = sum(demand[i] for i in route)

            for c in list(unserved):
                if base_load + demand[c] > capacity:
                    continue
                for pos in range(len(route) + 1):
                    new_route = route[:pos] + [c] + route[pos:]
                    if check_route_feasible(new_route, dist, demand, ready, due, service, capacity):
                        delta = route_distance(new_route, dist) - base_dist
                        insertions.append((delta, c, pos))

            if not insertions:
                break

            insertions.sort(key=lambda x: x[0])
            if randomized:
                choice = random.choice(insertions[: min(5, len(insertions))])
            else:
                choice = insertions[0]
            _, c_best, p_best = choice
            route = route[:p_best] + [c_best] + route[p_best:]
            unserved.remove(c_best)

        routes.append(route)

    # Try to place leftovers into existing routes if any
    if unserved:
        progress = True
        while unserved and progress:
            progress = False
            for c in list(unserved):
                best_move = None
                for r_idx, r in enumerate(routes):
                    if sum(demand[i] for i in r) + demand[c] > capacity:
                        continue
                    base = route_distance(r, dist)
                    for pos in range(len(r) + 1):
                        nr = r[:pos] + [c] + r[pos:]
                        if check_route_feasible(nr, dist, demand, ready, due, service, capacity):
                            delta = route_distance(nr, dist) - base
                            if best_move is None or delta < best_move[0]:
                                best_move = (delta, r_idx, pos)
                if best_move is not None:
                    _, r_idx, pos = best_move
                    routes[r_idx] = routes[r_idx][:pos] + [c] + routes[r_idx][pos:]
                    unserved.remove(c)
                    progress = True

    # If still unserved, create singleton routes (may exceed max_routes if needed to ensure service)
    for c in list(unserved):
        if check_route_feasible([c], dist, demand, ready, due, service, capacity):
            routes.append([c])
            unserved.remove(c)

    return routes


def solution_objective(routes: List[List[int]], dist: List[List[float]]) -> float:
    return float(sum(route_distance(r, dist) for r in routes))


def build_mip_and_solve(
    routes_start: List[List[int]],
    nodes: List[Dict],
    dist: List[List[float]],
    demand: List[int],
    ready: List[int],
    due: List[int],
    service: List[int],
    capacity: int,
    num_vehicles: int,
    time_limit: float,
    tracker: BestTracker,
):
    n = len(nodes) - 1
    cust = list(range(1, n + 1))
    all_nodes = list(range(0, n + 1))

    model = gp.Model("VRPTW")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = max(0.1, time_limit)
    model.Params.MIPFocus = 1

    x = model.addVars(
        [(i, j) for i in all_nodes for j in all_nodes if i != j],
        vtype=GRB.BINARY,
        name="x",
    )
    t = model.addVars(cust, lb=0.0, vtype=GRB.CONTINUOUS, name="t")
    l = model.addVars(cust, lb=0.0, vtype=GRB.CONTINUOUS, name="l")

    # Objective
    model.setObjective(gp.quicksum(dist[i][j] * x[i, j] for i, j in x.keys()), GRB.MINIMIZE)

    # Visit constraints
    for j in cust:
        model.addConstr(gp.quicksum(x[i, j] for i in all_nodes if i != j) == 1, name=f"in_{j}")
        model.addConstr(gp.quicksum(x[j, k] for k in all_nodes if k != j) == 1, name=f"out_{j}")

    # Depot flow / fleet size
    model.addConstr(gp.quicksum(x[0, j] for j in cust) == gp.quicksum(x[i, 0] for i in cust), name="depot_balance")
    if num_vehicles > 0:
        model.addConstr(gp.quicksum(x[0, j] for j in cust) <= num_vehicles, name="fleet_limit")

    # Time windows
    for i in cust:
        model.addConstr(t[i] >= ready[i], name=f"ready_{i}")
        model.addConstr(t[i] <= due[i], name=f"due_{i}")

    big_m = due[0] + max(service) + max(max(row) for row in dist) + 10.0

    # Depot -> customer
    for j in cust:
        model.addConstr(
            t[j] >= ready[0] + dist[0][j] - big_m * (1 - x[0, j]),
            name=f"time_0_{j}",
        )

    # customer -> customer
    for i in cust:
        for j in cust:
            if i == j:
                continue
            model.addConstr(
                t[j] >= t[i] + service[i] + dist[i][j] - big_m * (1 - x[i, j]),
                name=f"time_{i}_{j}",
            )

    # customer -> depot return bound
    for i in cust:
        model.addConstr(
            t[i] + service[i] + dist[i][0] <= due[0] + big_m * (1 - x[i, 0]),
            name=f"return_{i}",
        )

    # Capacity/load constraints
    for i in cust:
        model.addConstr(l[i] >= demand[i], name=f"load_lb_{i}")
        model.addConstr(l[i] <= capacity, name=f"load_ub_{i}")

    # From depot
    for j in cust:
        model.addConstr(l[j] >= demand[j] - capacity * (1 - x[0, j]), name=f"load0_lb_{j}")
        model.addConstr(l[j] <= demand[j] + capacity * (1 - x[0, j]), name=f"load0_ub_{j}")

    # Between customers
    for i in cust:
        for j in cust:
            if i == j:
                continue
            model.addConstr(
                l[j] >= l[i] + demand[j] - capacity * (1 - x[i, j]),
                name=f"load_lb_{i}_{j}",
            )
            model.addConstr(
                l[j] <= l[i] + demand[j] + capacity * (1 - x[i, j]),
                name=f"load_ub_{i}_{j}",
            )

    # Warm-start from heuristic routes
    start_arcs = set()
    for r in routes_start:
        if not r:
            continue
        start_arcs.add((0, r[0]))
        for a, b in zip(r[:-1], r[1:]):
            start_arcs.add((a, b))
        start_arcs.add((r[-1], 0))

    for (i, j), var in x.items():
        var.Start = 1.0 if (i, j) in start_arcs else 0.0

    # Optional starts for t,l from simulation
    for r in routes_start:
        tcur = ready[0]
        prev = 0
        loadcur = 0
        for c in r:
            arr = tcur + dist[prev][c]
            start = max(arr, ready[c])
            t[c].Start = start
            loadcur += demand[c]
            l[c].Start = loadcur
            tcur = start + service[c]
            prev = c

    def cb(m, where):
        if where == GRB.Callback.MIPSOL:
            obj = m.cbGet(GRB.Callback.MIPSOL_OBJ)
            tracker.consider(float(obj))

    model.optimize(cb)

    if model.SolCount == 0:
        return None, None

    x_sol = {(i, j): x[i, j].X for i, j in x.keys()}
    return model.ObjVal, x_sol


def arcs_to_routes(x_sol: Dict[Tuple[int, int], float], n_customers: int) -> List[List[int]]:
    succ = {}
    for (i, j), v in x_sol.items():
        if v > 0.5:
            succ[i] = succ.get(i, []) + [j]

    routes = []
    starts = succ.get(0, [])
    visited = set()

    for s in starts:
        route = []
        cur = s
        guard = 0
        while cur != 0 and guard <= n_customers + 5:
            guard += 1
            if cur in visited:
                break
            visited.add(cur)
            route.append(cur)
            nxts = succ.get(cur, [])
            if not nxts:
                break
            cur = nxts[0]
        routes.append(route)

    # In rare cases, catch unvisited cycles and output them too
    for c in range(1, n_customers + 1):
        if c not in visited:
            route = []
            cur = c
            guard = 0
            while cur != 0 and cur not in visited and guard <= n_customers + 5:
                guard += 1
                visited.add(cur)
                route.append(cur)
                nxts = succ.get(cur, [])
                if not nxts:
                    break
                cur = nxts[0]
            if route:
                routes.append(route)

    return routes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    t_start = time.monotonic()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    tracker = BestTracker(logger)

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    depot = inst["depot"]
    customers = inst["customers"]
    n = inst["num_customers"]
    capacity = int(inst["vehicle_capacity"])
    num_vehicles = int(inst.get("num_vehicles", n))
    if num_vehicles <= 0:
        num_vehicles = n

    # Internal node indexing: 0 depot, 1..n customers in given order
    nodes = [depot] + customers
    id_of = [depot["id"]] + [c["id"] for c in customers]

    demand = [int(nodes[i]["demand"]) for i in range(n + 1)]
    ready = [int(nodes[i]["ready_time"]) for i in range(n + 1)]
    due = [int(nodes[i]["due_date"]) for i in range(n + 1)]
    service = [int(nodes[i]["service_time"]) for i in range(n + 1)]

    dist = build_distance_matrix(nodes)
    cust_indices = list(range(1, n + 1))

    # Heuristic phase
    heuristic_budget = min(5.0, max(0.5, 0.2 * args.time_limit))
    best_routes = None
    best_obj = float("inf")

    it = 0
    while time.monotonic() - t_start < heuristic_budget:
        randomized = it > 0
        routes = greedy_construct(
            cust_indices,
            dist,
            demand,
            ready,
            due,
            service,
            capacity,
            num_vehicles,
            randomized=randomized,
        )
        covered = sum(len(r) for r in routes)
        if covered == n:
            obj = solution_objective(routes, dist)
            if obj + 1e-9 < best_obj:
                best_obj = obj
                best_routes = routes
                tracker.consider(obj)
        it += 1
        if it >= 20:
            break

    # If heuristic somehow failed, fallback singleton feasible
    if best_routes is None:
        fallback = []
        for c in cust_indices:
            if check_route_feasible([c], dist, demand, ready, due, service, capacity):
                fallback.append([c])
        if sum(len(r) for r in fallback) == n:
            best_routes = fallback
            best_obj = solution_objective(best_routes, dist)
            tracker.consider(best_obj)
        else:
            best_routes = []
            best_obj = float("inf")

    # MIP phase
    remaining = args.time_limit - (time.monotonic() - t_start)
    mip_obj = None
    mip_routes = None
    if remaining > 0.5 and best_routes is not None:
        try:
            obj, x_sol = build_mip_and_solve(
                best_routes,
                nodes,
                dist,
                demand,
                ready,
                due,
                service,
                capacity,
                num_vehicles,
                remaining,
                tracker,
            )
            if x_sol is not None:
                candidate_routes = arcs_to_routes(x_sol, n)
                if sum(len(r) for r in candidate_routes) == n:
                    cand_obj = solution_objective(candidate_routes, dist)
                    if cand_obj + 1e-6 < best_obj:
                        best_obj = cand_obj
                        best_routes = candidate_routes
                mip_obj = obj
                mip_routes = candidate_routes
        except Exception:
            pass

    # Build output
    output_routes = []
    for r in best_routes:
        customer_ids = [int(id_of[i]) for i in r]
        path = [int(id_of[0])] + customer_ids + [int(id_of[0])]
        output_routes.append(
            {
                "customer_ids": customer_ids,
                "path": path,
            }
        )

    if math.isinf(best_obj):
        # last resort
        best_obj = 0.0
        output_routes = []

    sol = {
        "objective_value": float(best_obj),
        "routes": output_routes,
    }

    with open(args.solution_path, "w") as f:
        json.dump(sol, f, indent=2)


if __name__ == "__main__":
    main()
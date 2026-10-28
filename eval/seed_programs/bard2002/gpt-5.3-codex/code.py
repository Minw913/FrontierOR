import argparse
import json
import time
from copy import deepcopy

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def load_instance(path):
    with open(path, "r") as f:
        return json.load(f)


def build_indexed_data(instance):
    n = int(instance["num_customers"])
    depot = instance["depot"]
    customers = instance["customers"]

    demand = [0] * (n + 1)
    ready = [0.0] * (n + 1)
    due = [0.0] * (n + 1)
    service = [0.0] * (n + 1)

    demand[0] = int(depot["demand"])
    ready[0] = float(depot["ready_time"])
    due[0] = float(depot["due_date"])
    service[0] = float(depot["service_time"])

    # customers are IDs 1..n
    by_id = {c["id"]: c for c in customers}
    for i in range(1, n + 1):
        c = by_id[i]
        demand[i] = int(c["demand"])
        ready[i] = float(c["ready_time"])
        due[i] = float(c["due_date"])
        service[i] = float(c["service_time"])

    base_dist = instance["distance_matrix"]
    base_dist = [[float(x) for x in row] for row in base_dist]

    horizon = float(instance.get("scheduling_horizon", due[0]))
    horizon = min(horizon, due[0])

    cap = int(instance["vehicle_capacity"])
    includes_service = bool(instance.get("travel_time_includes_service_time", False))

    return n, demand, ready, due, service, base_dist, horizon, cap, includes_service


def build_effective_travel_time(n, ready, due, service, base_dist, includes_service):
    tau = [[0.0] * (n + 1) for _ in range(n + 1)]

    if includes_service:
        # Assume matrix already contains effective travel times
        for i in range(n + 1):
            for j in range(n + 1):
                tau[i][j] = float(base_dist[i][j])
        return tau

    # Otherwise apply transformation from description
    for i in range(n + 1):
        for j in range(n + 1):
            if i == j:
                tau[i][j] = 0.0
                continue

            t = float(base_dist[i][j]) + float(service[i])

            if i != 0:
                gap = float(ready[j]) - float(due[i])
                if gap > t:
                    t = gap

            tau[i][j] = t

    return tau


def check_route_feasible(route, tau, demand, ready, due, cap, depot_ready, horizon):
    load = 0
    dep_prev = depot_ready
    prev = 0

    dep_times = {}
    loads = {}

    for c in route:
        load += demand[c]
        if load > cap:
            return False, None, None

        dep_c = max(ready[c], dep_prev + tau[prev][c])
        if dep_c > due[c] + 1e-9:
            return False, None, None

        dep_times[c] = dep_c
        loads[c] = load

        dep_prev = dep_c
        prev = c

    # Return to depot before horizon
    if route:
        if dep_prev + tau[route[-1]][0] > horizon + 1e-9:
            return False, None, None

    return True, dep_times, loads


def route_distance_with_depot(route, base_dist):
    if not route:
        return 0.0
    dist = base_dist[0][route[0]]
    for i in range(len(route) - 1):
        dist += base_dist[route[i]][route[i + 1]]
    dist += base_dist[route[-1]][0]
    return dist


def heuristic_merge_routes(n, tau, base_dist, demand, ready, due, cap, depot_ready, horizon, logger=None):
    # Start with singleton routes
    routes = []
    for c in range(1, n + 1):
        feas, _, _ = check_route_feasible([c], tau, demand, ready, due, cap, depot_ready, horizon)
        if not feas:
            return None  # infeasible instance (or no feasible singleton)
        routes.append([c])

    best_logged = len(routes)
    if logger:
        logger.log(float(best_logged))

    improved = True
    while improved:
        improved = False
        best_pair = None
        best_savings = -1e18

        m = len(routes)
        for i in range(m):
            for j in range(m):
                if i == j:
                    continue
                r1 = routes[i]
                r2 = routes[j]
                candidate = r1 + r2

                feas, _, _ = check_route_feasible(candidate, tau, demand, ready, due, cap, depot_ready, horizon)
                if not feas:
                    continue

                # Clarke-Wright style saving for concatenation
                s = base_dist[r1[-1]][0] + base_dist[0][r2[0]] - base_dist[r1[-1]][r2[0]]
                if s > best_savings:
                    best_savings = s
                    best_pair = (i, j, candidate)

        if best_pair is not None:
            i, j, merged = best_pair
            # remove max index first
            new_routes = []
            for idx, r in enumerate(routes):
                if idx != i and idx != j:
                    new_routes.append(r)
            new_routes.append(merged)
            routes = new_routes
            improved = True

            if logger and len(routes) < best_logged:
                best_logged = len(routes)
                logger.log(float(best_logged))

    # Build departure/load dictionaries
    dep_all = {}
    load_all = {}
    total_distance = 0.0
    output_routes = []

    for r in routes:
        feas, dep, loads = check_route_feasible(r, tau, demand, ready, due, cap, depot_ready, horizon)
        if not feas:
            return None
        dep_all.update(dep)
        load_all.update(loads)
        total_distance += route_distance_with_depot(r, base_dist)
        output_routes.append([0] + r + [0])

    return {
        "num_vehicles": len(routes),
        "routes_customers": routes,
        "routes": output_routes,
        "departure_times": dep_all,
        "loads": load_all,
        "total_distance": total_distance,
    }


def build_feasible_arcs(n, tau, demand, ready, due, cap, depot_ready, horizon):
    arcs = []
    pred = {j: [] for j in range(1, n + 1)}
    succ = {i: [] for i in range(1, n + 1)}
    depot_out = []
    depot_in = []

    # depot -> customer
    for j in range(1, n + 1):
        if depot_ready + tau[0][j] <= due[j] + 1e-9:
            arcs.append((0, j))
            pred[j].append(0)
            depot_out.append(j)

    # customer -> depot
    for i in range(1, n + 1):
        if ready[i] + tau[i][0] <= horizon + 1e-9:
            arcs.append((i, 0))
            succ[i].append(0)
            depot_in.append(i)

    # customer -> customer
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if i == j:
                continue
            if demand[i] + demand[j] > cap:
                continue
            if ready[i] + tau[i][j] > due[j] + 1e-9:
                continue
            arcs.append((i, j))
            succ[i].append(j)
            pred[j].append(i)

    return arcs, pred, succ, depot_out, depot_in


def reconstruct_routes_from_x(n, xvals):
    # xvals: dict (i,j)->0/1 approx
    starts = [j for j in range(1, n + 1) if xvals.get((0, j), 0.0) > 0.5]
    succ_map = {}
    for i in range(1, n + 1):
        nxt = None
        for j in range(0, n + 1):
            if i == j:
                continue
            if xvals.get((i, j), 0.0) > 0.5:
                nxt = j
                break
        succ_map[i] = nxt

    routes = []
    visited = set()
    for s in starts:
        route = []
        cur = s
        guard = 0
        while cur != 0 and cur is not None and guard <= n + 5:
            if cur in visited:
                break
            visited.add(cur)
            route.append(cur)
            cur = succ_map.get(cur, None)
            guard += 1
        routes.append(route)

    # If for any reason some customers are missing, append as singleton fallback
    for c in range(1, n + 1):
        if c not in visited:
            routes.append([c])

    return routes


def evaluate_routes(routes_customers, tau, base_dist, demand, ready, due, cap, depot_ready, horizon):
    dep_all = {}
    load_all = {}
    total_distance = 0.0
    output_routes = []

    for r in routes_customers:
        feas, dep, loads = check_route_feasible(r, tau, demand, ready, due, cap, depot_ready, horizon)
        if not feas:
            return None
        dep_all.update(dep)
        load_all.update(loads)
        total_distance += route_distance_with_depot(r, base_dist)
        output_routes.append([0] + r + [0])

    return {
        "num_vehicles": len(routes_customers),
        "routes_customers": routes_customers,
        "routes": output_routes,
        "departure_times": dep_all,
        "loads": load_all,
        "total_distance": total_distance,
    }


def solve_with_mip(instance, time_limit, logger=None):
    n, demand, ready, due, service, base_dist, horizon, cap, includes_service = build_indexed_data(instance)
    tau = build_effective_travel_time(n, ready, due, service, base_dist, includes_service)

    t0 = time.time()

    # Heuristic warm start (and fallback)
    heuristic_sol = heuristic_merge_routes(
        n=n,
        tau=tau,
        base_dist=base_dist,
        demand=demand,
        ready=ready,
        due=due,
        cap=cap,
        depot_ready=ready[0],
        horizon=horizon,
        logger=logger,
    )

    if time.time() - t0 >= time_limit:
        return heuristic_sol

    arcs, pred, succ, depot_out, depot_in = build_feasible_arcs(
        n, tau, demand, ready, due, cap, ready[0], horizon
    )

    # Quick feasibility checks for degree availability
    for c in range(1, n + 1):
        if len(pred[c]) == 0 or len(succ[c]) == 0:
            # fallback to heuristic if available
            return heuristic_sol

    model = gp.Model("VRPTW_min_vehicles")
    model.Params.Threads = 1
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = max(0.0, time_limit - (time.time() - t0))
    model.Params.MIPFocus = 1

    x = {}
    for (i, j) in arcs:
        x[(i, j)] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

    # departure time and cumulative load at customers
    t = {}
    q = {}
    for i in range(1, n + 1):
        t[i] = model.addVar(lb=ready[i], ub=due[i], vtype=GRB.CONTINUOUS, name=f"t_{i}")
        q[i] = model.addVar(lb=demand[i], ub=cap, vtype=GRB.CONTINUOUS, name=f"q_{i}")

    model.update()

    # Visit constraints
    for j in range(1, n + 1):
        model.addConstr(gp.quicksum(x[(i, j)] for i in pred[j]) == 1, name=f"in_{j}")

    for i in range(1, n + 1):
        model.addConstr(gp.quicksum(x[(i, j)] for j in succ[i]) == 1, name=f"out_{i}")

    # Depot flow balance
    k_out = gp.quicksum(x[(0, j)] for j in depot_out if (0, j) in x)
    k_in = gp.quicksum(x[(i, 0)] for i in depot_in if (i, 0) in x)
    model.addConstr(k_out == k_in, name="depot_balance")

    # Time propagation
    for i in range(1, n + 1):
        for j in succ[i]:
            if j == 0:
                continue
            if (i, j) not in x:
                continue
            Mij = max(0.0, due[i] + tau[i][j] - ready[j])
            model.addConstr(
                t[j] >= t[i] + tau[i][j] - Mij * (1 - x[(i, j)]),
                name=f"time_{i}_{j}",
            )

    for j in depot_out:
        if (0, j) in x:
            M0j = max(0.0, ready[0] + tau[0][j] - ready[j])
            model.addConstr(
                t[j] >= ready[0] + tau[0][j] - M0j * (1 - x[(0, j)]),
                name=f"time_0_{j}",
            )

    for i in depot_in:
        if (i, 0) in x:
            Mi0 = max(0.0, due[i] + tau[i][0] - horizon)
            model.addConstr(
                t[i] + tau[i][0] <= horizon + Mi0 * (1 - x[(i, 0)]),
                name=f"time_{i}_0",
            )

    # Load propagation
    for i in range(1, n + 1):
        for j in succ[i]:
            if j == 0:
                continue
            if (i, j) not in x:
                continue
            model.addConstr(
                q[j] >= q[i] + demand[j] - cap * (1 - x[(i, j)]),
                name=f"load_{i}_{j}",
            )

    for j in depot_out:
        if (0, j) in x:
            model.addConstr(
                q[j] >= demand[j] - cap * (1 - x[(0, j)]),
                name=f"load_0_{j}",
            )

    # Lexicographic-like weighted objective: minimize vehicles first, distance second
    max_d = max(max(row) for row in base_dist) if n > 0 else 0.0
    big_m = (2 * n + 2) * max_d + 1.0

    distance_expr = gp.quicksum(base_dist[i][j] * x[(i, j)] for (i, j) in x.keys())
    model.setObjective(big_m * k_out + distance_expr, GRB.MINIMIZE)

    # MIP start from heuristic
    if heuristic_sol is not None:
        for var in x.values():
            var.Start = 0.0

        for r in heuristic_sol["routes_customers"]:
            prev = 0
            for c in r:
                if (prev, c) in x:
                    x[(prev, c)].Start = 1.0
                prev = c
            if (prev, 0) in x:
                x[(prev, 0)].Start = 1.0

        dep_h = heuristic_sol["departure_times"]
        load_h = heuristic_sol["loads"]
        for i in range(1, n + 1):
            if i in dep_h:
                t[i].Start = dep_h[i]
            if i in load_h:
                q[i].Start = load_h[i]

    best_logged = [float("inf")]
    if heuristic_sol is not None:
        best_logged[0] = heuristic_sol["num_vehicles"]

    x0_vars = {j: x[(0, j)] for j in depot_out if (0, j) in x}

    def callback(m, where):
        if where == GRB.Callback.MIPSOL and logger is not None:
            kval = 0.0
            for j, var in x0_vars.items():
                kval += m.cbGetSolution(var)
            k_int = int(round(kval))
            if k_int < best_logged[0]:
                best_logged[0] = k_int
                logger.log(float(k_int))

    model.optimize(callback)

    best_sol = deepcopy(heuristic_sol)

    if model.SolCount > 0:
        xvals = {(i, j): x[(i, j)].X for (i, j) in x.keys()}
        routes_customers = reconstruct_routes_from_x(n, xvals)
        mip_eval = evaluate_routes(
            routes_customers, tau, base_dist, demand, ready, due, cap, ready[0], horizon
        )
        if mip_eval is not None:
            if best_sol is None:
                best_sol = mip_eval
            else:
                # prioritize fewer vehicles, then shorter distance
                a = (mip_eval["num_vehicles"], mip_eval["total_distance"])
                b = (best_sol["num_vehicles"], best_sol["total_distance"])
                if a < b:
                    best_sol = mip_eval

    return best_sol


def write_solution(path, sol, n):
    if sol is None:
        # Infeasible fallback (schema-compliant)
        out = {
            "objective_value": 1e18,
            "num_vehicles": 0,
            "routes": [],
            "total_distance": 0.0,
            "departure_times": {str(i): 0.0 for i in range(1, n + 1)},
            "loads": {str(i): 0 for i in range(1, n + 1)},
        }
    else:
        dep = {str(k): float(v) for k, v in sol["departure_times"].items()}
        loads = {str(k): int(v) for k, v in sol["loads"].items()}

        # Ensure all customer keys are present
        for i in range(1, n + 1):
            dep.setdefault(str(i), 0.0)
            loads.setdefault(str(i), 0)

        out = {
            "objective_value": float(sol["num_vehicles"]),
            "num_vehicles": int(sol["num_vehicles"]),
            "routes": sol["routes"],
            "total_distance": float(sol["total_distance"]),
            "departure_times": dep,
            "loads": loads,
        }

    with open(path, "w") as f:
        json.dump(out, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    instance = load_instance(args.instance_path)
    n = int(instance["num_customers"])

    sol = solve_with_mip(instance, time_limit=args.time_limit, logger=logger)
    write_solution(args.solution_path, sol, n)


if __name__ == "__main__":
    main()
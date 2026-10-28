import argparse
import json
import math
import random
import time
from copy import deepcopy

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def robust_worst_case_demand(route, nominal, deviation, gamma):
    if not route:
        return 0
    nom_sum = sum(nominal[i] for i in route)
    if gamma <= 0:
        return nom_sum
    devs = sorted((deviation[i] for i in route), reverse=True)
    g = min(gamma, len(devs))
    return nom_sum + sum(devs[:g])


def route_cost(route, dist):
    if not route:
        return 0
    c = dist[0][route[0]] + dist[route[-1]][0]
    for i in range(len(route) - 1):
        c += dist[route[i]][route[i + 1]]
    return c


def total_cost(routes, dist):
    return sum(route_cost(r, dist) for r in routes)


def best_insertion_position(route, customer, dist):
    if not route:
        return 0, dist[0][customer] + dist[customer][0]
    best_pos = 0
    best_delta = dist[0][customer] + dist[customer][route[0]] - dist[0][route[0]]
    # Between
    for p in range(1, len(route)):
        prev_n = route[p - 1]
        next_n = route[p]
        delta = dist[prev_n][customer] + dist[customer][next_n] - dist[prev_n][next_n]
        if delta < best_delta:
            best_delta = delta
            best_pos = p
    # End
    delta = dist[route[-1]][customer] + dist[customer][0] - dist[route[-1]][0]
    if delta < best_delta:
        best_delta = delta
        best_pos = len(route)
    return best_pos, best_delta


def construct_initial_solution(customer_ids, K, Q, gamma, nominal, deviation, dist, time_end):
    # Multi-start randomized greedy insertion
    best_routes = None
    best_obj = float("inf")

    attempts = 0
    while time.time() < time_end:
        attempts += 1
        routes = [[] for _ in range(K)]

        # Randomized order, biased by larger uncertain demand first
        customers = customer_ids[:]
        random.shuffle(customers)
        customers.sort(key=lambda i: nominal[i] + deviation[i], reverse=True)

        feasible = True
        for c in customers:
            candidates = []
            for k in range(K):
                pos, delta = best_insertion_position(routes[k], c, dist)
                new_route = routes[k][:pos] + [c] + routes[k][pos:]
                if robust_worst_case_demand(new_route, nominal, deviation, gamma) <= Q:
                    candidates.append((delta, k, pos))
            if not candidates:
                feasible = False
                break
            candidates.sort(key=lambda x: x[0])
            # Randomized selection among top few
            pick = candidates[min(len(candidates) - 1, random.randint(0, min(2, len(candidates) - 1)))]
            _, k, pos = pick
            routes[k].insert(pos, c)

        if feasible:
            obj = total_cost(routes, dist)
            if obj < best_obj:
                best_obj = obj
                best_routes = routes

        if attempts >= 30 and best_routes is not None:
            break

    return best_routes, best_obj


def local_search(routes, Q, gamma, nominal, deviation, dist, time_end):
    K = len(routes)
    improved = True
    current = deepcopy(routes)
    current_obj = total_cost(current, dist)

    while improved and time.time() < time_end:
        improved = False

        # Relocate (inter + intra), first improvement
        for r1 in range(K):
            if improved or time.time() >= time_end:
                break
            for i in range(len(current[r1])):
                if improved or time.time() >= time_end:
                    break
                c = current[r1][i]
                old_r1 = current[r1]
                base_r1 = old_r1[:i] + old_r1[i + 1 :]

                for r2 in range(K):
                    if improved or time.time() >= time_end:
                        break
                    old_r2 = current[r2]
                    max_pos = len(base_r1) if r1 == r2 else len(old_r2)
                    for pos in range(max_pos + 1):
                        if r1 == r2:
                            if pos == i or pos == i + 1:
                                continue
                            new_r = base_r1[:pos] + [c] + base_r1[pos:]
                            # same route, robust unchanged (same set), always feasible if old feasible
                            delta = route_cost(new_r, dist) - route_cost(old_r1, dist)
                            if delta < -1e-9:
                                current[r1] = new_r
                                current_obj += delta
                                improved = True
                                break
                        else:
                            new_r1 = base_r1
                            new_r2 = old_r2[:pos] + [c] + old_r2[pos:]
                            if robust_worst_case_demand(new_r1, nominal, deviation, gamma) > Q:
                                continue
                            if robust_worst_case_demand(new_r2, nominal, deviation, gamma) > Q:
                                continue
                            delta = (
                                route_cost(new_r1, dist)
                                + route_cost(new_r2, dist)
                                - route_cost(old_r1, dist)
                                - route_cost(old_r2, dist)
                            )
                            if delta < -1e-9:
                                current[r1] = new_r1
                                current[r2] = new_r2
                                current_obj += delta
                                improved = True
                                break

        # Swap (inter-route), first improvement
        if not improved:
            for r1 in range(K):
                if improved or time.time() >= time_end:
                    break
                for r2 in range(r1 + 1, K):
                    if improved or time.time() >= time_end:
                        break
                    for i in range(len(current[r1])):
                        if improved or time.time() >= time_end:
                            break
                        for j in range(len(current[r2])):
                            c1 = current[r1][i]
                            c2 = current[r2][j]
                            new_r1 = current[r1][:]
                            new_r2 = current[r2][:]
                            new_r1[i] = c2
                            new_r2[j] = c1

                            if robust_worst_case_demand(new_r1, nominal, deviation, gamma) > Q:
                                continue
                            if robust_worst_case_demand(new_r2, nominal, deviation, gamma) > Q:
                                continue

                            delta = (
                                route_cost(new_r1, dist)
                                + route_cost(new_r2, dist)
                                - route_cost(current[r1], dist)
                                - route_cost(current[r2], dist)
                            )
                            if delta < -1e-9:
                                current[r1] = new_r1
                                current[r2] = new_r2
                                current_obj += delta
                                improved = True
                                break

    return current, current_obj


def build_and_solve_mip(
    customer_ids,
    K,
    Q,
    gamma,
    nominal,
    deviation,
    dist,
    time_limit,
    initial_routes=None,
    logger=None,
):
    n = len(customer_ids)
    all_nodes = [0] + customer_ids

    model = gp.Model("RobustCVRP")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = max(0.0, time_limit)
    model.Params.MIPFocus = 1

    # Arc variables x[k,i,j]
    x = {}
    for k in range(K):
        for i in all_nodes:
            for j in all_nodes:
                if i == j and i != 0:
                    continue
                # allow 0->0 for empty vehicle route
                x[k, i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{k}_{i}_{j}")

    # Assignment vars y[k,i] for customers
    y = {}
    for k in range(K):
        for i in customer_ids:
            y[k, i] = model.addVar(vtype=GRB.BINARY, name=f"y_{k}_{i}")

    # MTZ order vars
    u = {}
    for k in range(K):
        for i in customer_ids:
            u[k, i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=n, name=f"u_{k}_{i}")

    # Robust vars
    theta = {}
    p = {}
    for k in range(K):
        theta[k] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"theta_{k}")
        for i in customer_ids:
            p[k, i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"p_{k}_{i}")

    model.update()

    # Objective
    model.setObjective(
        gp.quicksum(dist[i][j] * x[k, i, j] for (k, i, j) in x.keys()),
        GRB.MINIMIZE
    )

    # Depot degree for each vehicle
    for k in range(K):
        model.addConstr(gp.quicksum(x[k, 0, j] for j in all_nodes) == 1, name=f"dep_out_{k}")
        model.addConstr(gp.quicksum(x[k, j, 0] for j in all_nodes) == 1, name=f"dep_in_{k}")

    # Flow linking with y on customers
    for k in range(K):
        for i in customer_ids:
            model.addConstr(
                gp.quicksum(x[k, i, j] for j in all_nodes if j != i) == y[k, i],
                name=f"out_{k}_{i}",
            )
            model.addConstr(
                gp.quicksum(x[k, j, i] for j in all_nodes if j != i) == y[k, i],
                name=f"in_{k}_{i}",
            )

    # Each customer assigned exactly once
    for i in customer_ids:
        model.addConstr(gp.quicksum(y[k, i] for k in range(K)) == 1, name=f"assign_{i}")

    # MTZ subtour elimination
    for k in range(K):
        for i in customer_ids:
            model.addConstr(u[k, i] >= y[k, i], name=f"u_lb_{k}_{i}")
            model.addConstr(u[k, i] <= n * y[k, i], name=f"u_ub_{k}_{i}")

        for i in customer_ids:
            for j in customer_ids:
                if i == j:
                    continue
                if (k, i, j) in x:
                    model.addConstr(
                        u[k, i] - u[k, j] + n * x[k, i, j] <= n - 1,
                        name=f"mtz_{k}_{i}_{j}",
                    )

    # Robust capacity constraints (Bertsimas-Sim counterpart)
    for k in range(K):
        model.addConstr(
            gp.quicksum(nominal[i] * y[k, i] for i in customer_ids)
            + gamma * theta[k]
            + gp.quicksum(p[k, i] for i in customer_ids)
            <= Q,
            name=f"rob_cap_{k}",
        )
        for i in customer_ids:
            model.addConstr(
                p[k, i] >= deviation[i] * y[k, i] - theta[k],
                name=f"p_def_{k}_{i}",
            )

    # MIP start from heuristic
    if initial_routes is not None and len(initial_routes) == K:
        # reset starts
        for var in model.getVars():
            var.Start = 0.0

        assigned = set()
        for k, r in enumerate(initial_routes):
            if not r:
                if (k, 0, 0) in x:
                    x[k, 0, 0].Start = 1.0
            else:
                x[k, 0, r[0]].Start = 1.0
                for t in range(len(r) - 1):
                    x[k, r[t], r[t + 1]].Start = 1.0
                x[k, r[-1], 0].Start = 1.0
                for c in r:
                    y[k, c].Start = 1.0
                    assigned.add(c)

        # Ensure all assigned (safety)
        for c in customer_ids:
            if c not in assigned:
                # leave it free if mismatch
                pass

    model._logger = logger
    model._best_logged = float("inf")

    def cb(m, where):
        if where == GRB.Callback.MIPSOL:
            obj = m.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj + 1e-9 < m._best_logged:
                m._best_logged = obj
                if m._logger:
                    m._logger.log(float(obj))

    model.optimize(cb)

    if model.SolCount == 0:
        return None, None

    # Extract routes
    routes = [[] for _ in range(K)]
    for k in range(K):
        succ = {}
        for i in all_nodes:
            for j in all_nodes:
                if i == j and i != 0:
                    continue
                var = x.get((k, i, j))
                if var is not None and var.X > 0.5:
                    succ[i] = j

        # Build route from depot
        if 0 not in succ:
            routes[k] = []
            continue
        cur = succ[0]
        if cur == 0:
            routes[k] = []
            continue

        seen = set()
        r = []
        while cur != 0 and cur not in seen:
            seen.add(cur)
            r.append(cur)
            cur = succ.get(cur, 0)
        routes[k] = r

    obj = total_cost(routes, dist)
    return routes, obj


def write_solution(solution_path, routes, nominal, deviation, gamma, dist):
    sol_routes = []
    for r in routes:
        sol_routes.append({
            "customers": r,
            "cost": int(round(route_cost(r, dist))),
            "worst_case_demand": int(round(robust_worst_case_demand(r, nominal, deviation, gamma))),
        })

    solution = {
        "objective_value": float(total_cost(routes, dist)),
        "routes": sol_routes,
    }

    with open(solution_path, "w") as f:
        json.dump(solution, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    random.seed(42)
    start_time = time.time()
    end_time = start_time + max(1, args.time_limit)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    K = int(inst["num_vehicles"])
    Q = int(inst["vehicle_capacity"])
    gamma = int(inst["uncertainty_set"]["Gamma"])
    customers = inst["customers"]
    dist = inst["distance_matrix"]

    customer_ids = [int(c["id"]) for c in customers]
    nominal = {int(c["id"]): int(c["nominal_demand"]) for c in customers}
    deviation = {int(c["id"]): int(c["demand_deviation"]) for c in customers}

    # Quick infeasibility check for singletons
    for c in customer_ids:
        if nominal[c] + min(gamma, 1) * deviation[c] > Q:
            # No robust-feasible assignment possible for this customer
            routes = [[] for _ in range(K)]
            # place customer anyway into first route to keep schema valid
            if K > 0:
                routes[0] = [c]
            write_solution(args.solution_path, routes, nominal, deviation, gamma, dist)
            return

    best_routes = None
    best_obj = float("inf")

    # Heuristic phase
    heuristic_end = min(end_time, start_time + max(1.0, args.time_limit * 0.35))
    init_routes, init_obj = construct_initial_solution(
        customer_ids, K, Q, gamma, nominal, deviation, dist, heuristic_end
    )

    if init_routes is None:
        # fallback trivial random packing with retries
        init_routes = [[] for _ in range(K)]
        shuffled = customer_ids[:]
        random.shuffle(shuffled)
        for c in shuffled:
            placed = False
            for k in range(K):
                trial = init_routes[k] + [c]
                if robust_worst_case_demand(trial, nominal, deviation, gamma) <= Q:
                    init_routes[k].append(c)
                    placed = True
                    break
            if not placed and K > 0:
                init_routes[0].append(c)
        init_obj = total_cost(init_routes, dist)

    # Local improvement (if feasible)
    feasible_init = all(
        robust_worst_case_demand(r, nominal, deviation, gamma) <= Q for r in init_routes
    ) and sorted([c for r in init_routes for c in r]) == sorted(customer_ids)

    if feasible_init:
        ls_end = min(end_time, start_time + max(1.0, args.time_limit * 0.55))
        ls_routes, ls_obj = local_search(
            init_routes, Q, gamma, nominal, deviation, dist, ls_end
        )
        best_routes, best_obj = ls_routes, ls_obj
        if logger:
            logger.log(float(best_obj))
    else:
        best_routes, best_obj = init_routes, init_obj

    # MIP phase (if not too large and time remains)
    n = len(customer_ids)
    est_vars = K * (n + 1) * (n + 1)
    remaining = end_time - time.time()

    if remaining > 1.0 and est_vars <= 400000:
        mip_routes, mip_obj = build_and_solve_mip(
            customer_ids=customer_ids,
            K=K,
            Q=Q,
            gamma=gamma,
            nominal=nominal,
            deviation=deviation,
            dist=dist,
            time_limit=remaining,
            initial_routes=best_routes if feasible_init else None,
            logger=logger,
        )
        if mip_routes is not None:
            # Safety: ensure customer coverage; if malformed, keep heuristic
            all_mip_customers = [c for r in mip_routes for c in r]
            if sorted(all_mip_customers) == sorted(customer_ids):
                best_routes = mip_routes
                best_obj = mip_obj

    # Final fallback
    if best_routes is None:
        best_routes = [[] for _ in range(K)]

    # Ensure exactly K routes
    if len(best_routes) < K:
        best_routes += [[] for _ in range(K - len(best_routes))]
    elif len(best_routes) > K:
        best_routes = best_routes[:K]

    write_solution(args.solution_path, best_routes, nominal, deviation, gamma, dist)


if __name__ == "__main__":
    main()
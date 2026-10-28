import argparse
import json
import time
from typing import Dict, List, Tuple, Optional

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


EPS = 1e-6


def build_inventory_lp(data, time_limit: float):
    n = data["num_nodes"]
    T = data["num_periods"]
    m = data["num_vehicles"]
    Q = data["vehicle_capacity"]

    demands = data["customer_demands"]  # [num_customers][T], customer i => node i+1
    prod = data["supplier_production"]
    U = data["inventory_capacities"]
    init = data["initial_inventories"]
    h = data["holding_costs"]

    cust_nodes = list(range(1, n))

    mdl = gp.Model("inventory_lp")
    mdl.Params.OutputFlag = 0
    mdl.Params.Threads = 1
    if time_limit is not None and time_limit > 0:
        mdl.Params.TimeLimit = time_limit

    q = mdl.addVars(T, cust_nodes, lb=0.0, vtype=GRB.CONTINUOUS, name="q")
    I = mdl.addVars(T, range(n), lb=0.0, vtype=GRB.CONTINUOUS, name="I")

    # bounds on inventories
    for t in range(T):
        for i in range(n):
            I[t, i].UB = U[i]

    for t in range(T):
        # customer inventory + immediate post-delivery cap
        for i in cust_nodes:
            dem = demands[i - 1][t]
            prev = init[i] if t == 0 else I[t - 1, i]
            mdl.addConstr(I[t, i] == prev + q[t, i] - dem, name=f"cust_bal_{t}_{i}")
            mdl.addConstr(prev + q[t, i] <= U[i], name=f"cust_postcap_{t}_{i}")
            mdl.addConstr(q[t, i] <= min(U[i], Q), name=f"cust_qcap_{t}_{i}")

        # depot inventory
        prev0 = init[0] if t == 0 else I[t - 1, 0]
        mdl.addConstr(
            I[t, 0] == prev0 + prod[t] - gp.quicksum(q[t, i] for i in cust_nodes),
            name=f"dep_bal_{t}",
        )

        # fleet aggregate capacity
        mdl.addConstr(
            gp.quicksum(q[t, i] for i in cust_nodes) <= m * Q,
            name=f"fleet_cap_{t}",
        )

    mdl.setObjective(
        gp.quicksum(h[i] * I[t, i] for t in range(T) for i in range(n)),
        GRB.MINIMIZE,
    )

    mdl.optimize()

    if mdl.Status not in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL] or mdl.SolCount == 0:
        return None

    q_sol = {(t, i): float(q[t, i].X) for t in range(T) for i in cust_nodes}
    I_sol = {(t, i): float(I[t, i].X) for t in range(T) for i in range(n)}
    return q_sol, I_sol


def assign_customers_to_vehicles(
    customers: List[int], qty: Dict[int, float], m: int, Q: float, dist0: List[float], time_limit: float = 2.0
) -> Optional[List[List[int]]]:
    # First-fit decreasing
    sorted_customers = sorted(customers, key=lambda c: qty[c], reverse=True)
    bins = []  # list of (load, customers)
    for c in sorted_customers:
        placed = False
        best_idx = None
        best_rem = None
        for bi, (load, clist) in enumerate(bins):
            rem = Q - load
            if qty[c] <= rem + 1e-9:
                if best_rem is None or rem - qty[c] < best_rem:
                    best_rem = rem - qty[c]
                    best_idx = bi
        if best_idx is not None:
            bins[best_idx][0] += qty[c]
            bins[best_idx][1].append(c)
            placed = True
        elif len(bins) < m:
            bins.append([qty[c], [c]])
            placed = True

        if not placed:
            break

    if len(customers) == sum(len(b[1]) for b in bins):
        return [b[1] for b in bins if b[1]]

    # Fallback assignment MIP
    mdl = gp.Model("assign")
    mdl.Params.OutputFlag = 0
    mdl.Params.Threads = 1
    mdl.Params.TimeLimit = max(0.5, time_limit)

    V = list(range(m))
    a = mdl.addVars(V, customers, vtype=GRB.BINARY, name="a")
    for c in customers:
        mdl.addConstr(gp.quicksum(a[v, c] for v in V) == 1)

    for v in V:
        mdl.addConstr(gp.quicksum(qty[c] * a[v, c] for c in customers) <= Q)

    mdl.setObjective(
        gp.quicksum(dist0[c] * a[v, c] for v in V for c in customers), GRB.MINIMIZE
    )
    mdl.optimize()

    if mdl.SolCount == 0:
        return None

    veh_customers = [[] for _ in V]
    for v in V:
        for c in customers:
            if a[v, c].X > 0.5:
                veh_customers[v].append(c)

    return [lst for lst in veh_customers if lst]


def nearest_neighbor_route(customers: List[int], dist: List[List[int]]) -> List[int]:
    if not customers:
        return [0, 0]
    rem = set(customers)
    route = [0]
    cur = 0
    while rem:
        nxt = min(rem, key=lambda j: dist[cur][j])
        route.append(nxt)
        rem.remove(nxt)
        cur = nxt
    route.append(0)
    return route


def build_routes_from_quantities(data, q_sol: Dict[Tuple[int, int], float], per_period_time: float = 2.0):
    n = data["num_nodes"]
    T = data["num_periods"]
    m = data["num_vehicles"]
    Q = data["vehicle_capacity"]
    dist = data["distance_matrix"]

    period_routes = {t: [] for t in range(T)}
    travel_cost = 0.0

    for t in range(T):
        qty = {i: q_sol.get((t, i), 0.0) for i in range(1, n)}
        active = [i for i in range(1, n) if qty[i] > EPS]

        if not active:
            continue

        # sanity
        if any(qty[i] > Q + 1e-6 for i in active):
            return None

        bins = assign_customers_to_vehicles(
            active, qty, m, Q, dist[0], time_limit=per_period_time
        )
        if bins is None:
            return None

        # build routes
        for cust_list in bins:
            route = nearest_neighbor_route(cust_list, dist)
            dmap = {str(i): float(qty[i]) for i in cust_list if qty[i] > EPS}
            period_routes[t].append({"route": route, "deliveries": dmap})

            for a, b in zip(route[:-1], route[1:]):
                travel_cost += dist[a][b]

    return period_routes, travel_cost


def inventories_from_quantities(data, q_sol: Dict[Tuple[int, int], float]):
    n = data["num_nodes"]
    T = data["num_periods"]
    demands = data["customer_demands"]
    prod = data["supplier_production"]
    init = data["initial_inventories"]

    I = {(t, i): 0.0 for t in range(T) for i in range(n)}

    for t in range(T):
        # customers
        for i in range(1, n):
            prev = init[i] if t == 0 else I[(t - 1, i)]
            I[(t, i)] = prev + q_sol.get((t, i), 0.0) - demands[i - 1][t]

        # depot
        prev0 = init[0] if t == 0 else I[(t - 1, 0)]
        shipped = sum(q_sol.get((t, i), 0.0) for i in range(1, n))
        I[(t, 0)] = prev0 + prod[t] - shipped

    return I


def heuristic_solution(data, total_time: float):
    # 1) LP for deliveries/inventories
    lp = build_inventory_lp(data, time_limit=max(1.0, min(30.0, total_time * 0.4)))
    if lp is None:
        return None
    q_sol, I_sol_lp = lp

    # 2) routing heuristic
    routed = build_routes_from_quantities(data, q_sol, per_period_time=max(0.5, total_time * 0.05))
    if routed is None:
        return None
    period_routes, travel = routed

    I_sol = inventories_from_quantities(data, q_sol)
    h = data["holding_costs"]
    T = data["num_periods"]
    n = data["num_nodes"]

    holding = sum(h[i] * I_sol[(t, i)] for t in range(T) for i in range(n))
    obj = float(travel + holding)

    return {
        "objective_value": obj,
        "q": q_sol,
        "I": I_sol,
        "period_routes": period_routes,
    }


def build_output_from_routes_and_inventory(data, obj, period_routes, I_sol):
    T = data["num_periods"]
    n = data["num_nodes"]

    routes_out = {}
    for t in range(T):
        tkey = str(t)
        routes_out[tkey] = {}
        veh_id = 0
        for r in period_routes.get(t, []):
            routes_out[tkey][str(veh_id)] = {
                "route": [int(x) for x in r["route"]],
                "deliveries": {str(k): float(v) for k, v in r["deliveries"].items()},
            }
            veh_id += 1

    inv_out = {}
    for i in range(n):
        ikey = str(i)
        inv_out[ikey] = {}
        for t in range(T):
            inv_out[ikey][str(t)] = float(I_sol[(t, i)])

    return {
        "objective_value": float(obj),
        "routes": routes_out,
        "inventories": inv_out,
    }


def solve_integrated_mip(data, time_limit: float, logger: Optional[SolutionLogger], incumbent_obj: Optional[float], warm_start=None):
    n = data["num_nodes"]
    T = data["num_periods"]
    m = data["num_vehicles"]
    Q = data["vehicle_capacity"]

    dist = data["distance_matrix"]
    demands = data["customer_demands"]
    prod = data["supplier_production"]
    U = data["inventory_capacities"]
    init = data["initial_inventories"]
    h = data["holding_costs"]

    nodes = list(range(n))
    cust_nodes = list(range(1, n))
    arcs = [(i, j) for i in nodes for j in nodes if i != j]

    out_nei = {i: [j for j in nodes if j != i] for i in nodes}
    in_nei = {i: [j for j in nodes if j != i] for i in nodes}

    mdl = gp.Model("integrated_irp")
    mdl.Params.OutputFlag = 0
    mdl.Params.Threads = 1
    mdl.Params.TimeLimit = max(1.0, time_limit)
    mdl.Params.MIPFocus = 1

    x = mdl.addVars(T, arcs, vtype=GRB.BINARY, name="x")
    g = mdl.addVars(T, arcs, lb=0.0, vtype=GRB.CONTINUOUS, name="g")
    y = mdl.addVars(T, cust_nodes, vtype=GRB.BINARY, name="y")
    q = mdl.addVars(T, cust_nodes, lb=0.0, vtype=GRB.CONTINUOUS, name="q")
    I = mdl.addVars(T, nodes, lb=0.0, vtype=GRB.CONTINUOUS, name="I")

    # inventory bounds
    for t in range(T):
        for i in nodes:
            I[t, i].UB = U[i]

    for t in range(T):
        # customer degree + linking
        for i in cust_nodes:
            mdl.addConstr(gp.quicksum(x[t, i, j] for j in out_nei[i]) == y[t, i], name=f"outdeg_{t}_{i}")
            mdl.addConstr(gp.quicksum(x[t, j, i] for j in in_nei[i]) == y[t, i], name=f"indeg_{t}_{i}")

            mdl.addConstr(q[t, i] <= min(U[i], Q) * y[t, i], name=f"q_visit_{t}_{i}")

            prev = init[i] if t == 0 else I[t - 1, i]
            dem = demands[i - 1][t]
            mdl.addConstr(I[t, i] == prev + q[t, i] - dem, name=f"cust_bal_{t}_{i}")
            mdl.addConstr(prev + q[t, i] <= U[i], name=f"cust_postcap_{t}_{i}")

            # flow conservation at customer
            mdl.addConstr(
                gp.quicksum(g[t, j, i] for j in in_nei[i]) - gp.quicksum(g[t, i, j] for j in out_nei[i]) == q[t, i],
                name=f"flow_c_{t}_{i}",
            )

        # depot degree / number of vehicles
        mdl.addConstr(
            gp.quicksum(x[t, 0, j] for j in cust_nodes) == gp.quicksum(x[t, j, 0] for j in cust_nodes),
            name=f"depot_deg_bal_{t}",
        )
        mdl.addConstr(
            gp.quicksum(x[t, 0, j] for j in cust_nodes) <= m,
            name=f"veh_lim_{t}",
        )

        # depot inventory
        prev0 = init[0] if t == 0 else I[t - 1, 0]
        mdl.addConstr(
            I[t, 0] == prev0 + prod[t] - gp.quicksum(q[t, i] for i in cust_nodes),
            name=f"dep_bal_{t}",
        )
        mdl.addConstr(
            gp.quicksum(q[t, i] for i in cust_nodes) <= m * Q,
            name=f"fleet_qty_{t}",
        )

        # flow from depot and zero return load
        mdl.addConstr(gp.quicksum(g[t, 0, j] for j in cust_nodes) == gp.quicksum(q[t, i] for i in cust_nodes),
                      name=f"dep_flow_out_{t}")
        for i in cust_nodes:
            mdl.addConstr(g[t, i, 0] == 0.0, name=f"ret_zero_{t}_{i}")

        # arc capacity linking
        for i, j in arcs:
            mdl.addConstr(g[t, i, j] <= Q * x[t, i, j], name=f"cap_{t}_{i}_{j}")

    travel = gp.quicksum(dist[i][j] * x[t, i, j] for t in range(T) for (i, j) in arcs)
    holding = gp.quicksum(h[i] * I[t, i] for t in range(T) for i in nodes)
    mdl.setObjective(travel + holding, GRB.MINIMIZE)

    # warm start
    if warm_start is not None:
        q_ws = warm_start["q"]
        I_ws = warm_start["I"]
        period_routes = warm_start["period_routes"]

        for t in range(T):
            for i in cust_nodes:
                val = q_ws.get((t, i), 0.0)
                q[t, i].Start = val
                y[t, i].Start = 1.0 if val > EPS else 0.0
            for i in nodes:
                I[t, i].Start = I_ws[(t, i)]

        for t in range(T):
            for r in period_routes.get(t, []):
                route = r["route"]
                dmap = {int(k): float(v) for k, v in r["deliveries"].items()}
                cust_seq = [v for v in route if v != 0]
                rem = sum(dmap.get(c, 0.0) for c in cust_seq)
                for a, b in zip(route[:-1], route[1:]):
                    x[t, a, b].Start = 1.0
                    if a == 0:
                        g[t, a, b].Start = rem
                    elif a in dmap:
                        rem -= dmap[a]
                        g[t, a, b].Start = max(0.0, rem)
                    else:
                        g[t, a, b].Start = max(0.0, rem)

    best_logged = [incumbent_obj if incumbent_obj is not None else float("inf")]

    def cb(model, where):
        if where == GRB.Callback.MIPSOL and logger is not None:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj + 1e-6 < best_logged[0]:
                best_logged[0] = obj
                logger.log(float(obj))

    mdl.optimize(cb)

    if mdl.SolCount == 0:
        return None

    # extract solution
    x_sol = {(t, i, j): x[t, i, j].X for t in range(T) for (i, j) in arcs}
    q_sol = {(t, i): q[t, i].X for t in range(T) for i in cust_nodes}
    I_sol = {(t, i): I[t, i].X for t in range(T) for i in nodes}

    period_routes = {t: [] for t in range(T)}
    for t in range(T):
        succ = {}
        depot_starts = []
        for i, j in arcs:
            if x_sol[(t, i, j)] > 0.5:
                succ[i] = succ.get(i, []) + [j]
                if i == 0 and j != 0:
                    depot_starts.append(j)

        used_customers = set()
        for s in depot_starts:
            route = [0, s]
            cur = s
            guard = 0
            while cur != 0 and guard < n + 5:
                used_customers.add(cur)
                nxts = succ.get(cur, [])
                if not nxts:
                    break
                nxt = nxts[0]
                route.append(nxt)
                cur = nxt
                guard += 1
            if route[-1] != 0:
                route.append(0)

            dmap = {}
            for v in route:
                if v != 0:
                    qv = q_sol[(t, v)]
                    if qv > EPS:
                        dmap[str(v)] = float(qv)
            period_routes[t].append({"route": route, "deliveries": dmap})

    obj = float(mdl.ObjVal)
    return {
        "objective_value": obj,
        "q": q_sol,
        "I": I_sol,
        "period_routes": period_routes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, required=False, default=None)
    args = parser.parse_args()

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    start = time.time()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    best = None
    best_obj = float("inf")

    # Heuristic incumbent
    time_left = max(0.0, args.time_limit - (time.time() - start))
    if time_left > 0.5:
        heur = heuristic_solution(data, total_time=max(1.0, min(30.0, time_left * 0.35)))
        if heur is not None:
            best = heur
            best_obj = heur["objective_value"]
            if logger:
                logger.log(float(best_obj))

    # Integrated MIP improvement
    time_left = max(0.0, args.time_limit - (time.time() - start))
    if time_left > 1.0:
        mip = solve_integrated_mip(
            data,
            time_limit=time_left,
            logger=logger,
            incumbent_obj=best_obj if best is not None else None,
            warm_start=best,
        )
        if mip is not None and mip["objective_value"] + 1e-6 < best_obj:
            best = mip
            best_obj = mip["objective_value"]

    # Final fallback
    if best is None:
        # Minimal emergency output (should be rare; potentially infeasible instance)
        n = data["num_nodes"]
        T = data["num_periods"]
        routes = {str(t): {} for t in range(T)}
        inv = {str(i): {str(t): float(data["initial_inventories"][i]) for t in range(T)} for i in range(n)}
        out = {
            "objective_value": float(1e18),
            "routes": routes,
            "inventories": inv,
        }
    else:
        out = build_output_from_routes_and_inventory(
            data,
            best["objective_value"],
            best["period_routes"],
            best["I"],
        )

    with open(args.solution_path, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
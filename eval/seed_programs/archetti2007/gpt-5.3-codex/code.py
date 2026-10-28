import argparse
import json
import time
from collections import defaultdict, deque

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def build_and_solve(instance, time_limit, logger=None):
    n = instance["num_retailers"]
    H = instance["time_horizon"]

    supplier = instance["supplier"]
    retailers = instance["retailers"]
    Q = instance["vehicle_capacity"]
    dist = instance["distance_matrix"]

    # Indexing by matrix position: 0=supplier, 1..n=retailers
    retailer_ids = [r["id"] for r in retailers]
    supplier_id = supplier["id"]
    U = {i + 1: retailers[i]["max_inventory"] for i in range(n)}
    d = {i + 1: retailers[i]["demand"] for i in range(n)}
    h_r = {i + 1: retailers[i]["inventory_cost"] for i in range(n)}
    I0_r = {i + 1: retailers[i]["starting_inventory"] for i in range(n)}

    h_s = supplier["inventory_cost"]
    S0 = supplier["starting_inventory"]
    p = supplier["production_rate"]

    m = gp.Model("IRP_fillup_single_vehicle")
    m.Params.OutputFlag = 0
    m.Params.TimeLimit = time_limit
    m.Params.Threads = 1
    m.Params.MIPFocus = 1
    m.Params.LazyConstraints = 1

    T = range(1, H + 1)
    T_ext = range(1, H + 2)
    R = range(1, n + 1)

    # Inventory variables
    S = m.addVars(T_ext, lb=0.0, vtype=GRB.CONTINUOUS, name="S")  # supplier inventory at period start
    I = m.addVars(R, T_ext, lb=0.0, vtype=GRB.CONTINUOUS, name="I")  # retailer inventory at period start
    P = m.addVars(R, T, lb=0.0, vtype=GRB.CONTINUOUS, name="P")  # retailer inventory right after delivery
    q = m.addVars(R, T, lb=0.0, vtype=GRB.CONTINUOUS, name="q")  # delivery quantity
    y = m.addVars(R, T, vtype=GRB.BINARY, name="y")  # visit

    # Routing variables
    z = m.addVars(T, vtype=GRB.BINARY, name="z")  # route active
    x0 = m.addVars(R, T, lb=0.0, ub=2.0, vtype=GRB.INTEGER, name="x0")  # depot-retailer multiplicity
    x = {}
    for i in R:
        for j in R:
            if i < j:
                for t in T:
                    x[(i, j, t)] = m.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}_{t}")
    m.update()

    # Initial conditions
    m.addConstr(S[1] == S0, name="S_init")
    for i in R:
        m.addConstr(I[i, 1] == I0_r[i], name=f"I_init_{i}")

    # Supplier flow and shipment availability
    for t in T:
        ship_t = gp.quicksum(q[i, t] for i in R)
        m.addConstr(S[t] >= ship_t, name=f"S_avail_{t}")
        m.addConstr(S[t + 1] == S[t] + p - ship_t, name=f"S_bal_{t}")

    # Retailer flow, fill-up policy, and capacity
    for t in T:
        m.addConstr(gp.quicksum(q[i, t] for i in R) <= Q, name=f"veh_cap_{t}")
        for i in R:
            # Fill-up linearization via post-delivery inventory P
            m.addConstr(P[i, t] <= U[i], name=f"P_ubU_{i}_{t}")
            m.addConstr(P[i, t] >= I[i, t], name=f"P_geI_{i}_{t}")
            m.addConstr(P[i, t] <= I[i, t] + U[i] * y[i, t], name=f"P_leIUy_{i}_{t}")
            m.addConstr(P[i, t] >= U[i] * y[i, t], name=f"P_geUy_{i}_{t}")

            m.addConstr(q[i, t] == P[i, t] - I[i, t], name=f"q_def_{i}_{t}")
            m.addConstr(I[i, t + 1] == P[i, t] - d[i], name=f"I_bal_{i}_{t}")

    # Retailer non-negativity already enforced by var lb=0 on I including period H+1

    # Routing degree/activation constraints
    for t in T:
        # Depot degree
        m.addConstr(gp.quicksum(x0[i, t] for i in R) == 2 * z[t], name=f"deg_depot_{t}")

        m.addConstr(gp.quicksum(y[i, t] for i in R) >= z[t], name=f"z_lb_{t}")
        m.addConstr(gp.quicksum(y[i, t] for i in R) <= n * z[t], name=f"z_ub_{t}")

        for i in R:
            incident_rr = gp.quicksum(x[(j, i, t)] for j in R if j < i) + gp.quicksum(
                x[(i, j, t)] for j in R if i < j
            )
            m.addConstr(x0[i, t] + incident_rr == 2 * y[i, t], name=f"deg_{i}_{t}")
            m.addConstr(y[i, t] <= z[t], name=f"visit_active_{i}_{t}")
            m.addConstr(x0[i, t] <= 2 * y[i, t], name=f"x0_visit_{i}_{t}")

    # Objective
    inv_supplier_cost = h_s * gp.quicksum(S[t] for t in T_ext)
    inv_retailer_cost = gp.quicksum(h_r[i] * I[i, t] for i in R for t in T_ext)
    transport_cost = gp.quicksum(dist[0][i] * x0[i, t] for i in R for t in T) + gp.quicksum(
        dist[i][j] * x[(i, j, t)] for i in R for j in R if i < j for t in T
    )
    m.setObjective(inv_supplier_cost + inv_retailer_cost + transport_cost, GRB.MINIMIZE)

    # Store for callback
    m._R = list(R)
    m._T = list(T)
    m._x = x
    m._x0 = x0
    m._y = y
    m._logger = logger
    m._best_logged = float("inf")

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            # Log incumbent improvements
            try:
                obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if model._logger is not None and obj + 1e-9 < model._best_logged:
                    model._best_logged = obj
                    model._logger.log(float(obj))
            except Exception:
                pass

            # Lazy subtour elimination per period
            for t in model._T:
                visited = [i for i in model._R if model.cbGetSolution(model._y[i, t]) > 0.5]
                if len(visited) <= 1:
                    continue

                # Build adjacency among visited + depot (0)
                adj = defaultdict(list)

                # depot-retailer edges
                for i in visited:
                    if model.cbGetSolution(model._x0[i, t]) > 0.5:
                        adj[0].append(i)
                        adj[i].append(0)

                # retailer-retailer edges
                for idx_i in range(len(visited)):
                    i = visited[idx_i]
                    for idx_j in range(idx_i + 1, len(visited)):
                        j = visited[idx_j]
                        a, b = (i, j) if i < j else (j, i)
                        if model.cbGetSolution(model._x[a, b, t]) > 0.5:
                            adj[i].append(j)
                            adj[j].append(i)

                # Reachable from depot
                reached = set()
                if 0 in adj:
                    dq = deque([0])
                    reached.add(0)
                    while dq:
                        u = dq.popleft()
                        for v in adj[u]:
                            if v not in reached:
                                reached.add(v)
                                dq.append(v)

                unreached = [i for i in visited if i not in reached]
                if not unreached:
                    continue

                # Components in unreached subgraph using retailer-retailer edges only
                unreached_set = set(unreached)
                seen = set()
                for start in unreached:
                    if start in seen:
                        continue
                    comp = []
                    dq = deque([start])
                    seen.add(start)
                    while dq:
                        u = dq.popleft()
                        comp.append(u)
                        for v in adj[u]:
                            if v in unreached_set and v not in seen:
                                seen.add(v)
                                dq.append(v)

                    if len(comp) >= 2:
                        comp_sorted = sorted(comp)
                        lhs = gp.quicksum(
                            model._x[(i, j, t)]
                            for ii, i in enumerate(comp_sorted)
                            for j in comp_sorted[ii + 1 :]
                        )
                        rhs = gp.quicksum(model._y[i, t] for i in comp_sorted) - 1
                        model.cbLazy(lhs <= rhs)

    m.optimize(callback)

    if m.SolCount == 0:
        return None

    # Build solution
    obj_val = float(m.ObjVal)

    deliveries = {}
    routes = {}
    supplier_inventory = {}
    retailer_inventory = {}

    for t in T:
        t_key = str(t)
        deliveries[t_key] = {}
        for i in R:
            qv = q[i, t].X
            if qv > 1e-6:
                rid = retailer_ids[i - 1]
                deliveries[t_key][str(rid)] = float(qv)

    for t in T:
        t_key = str(t)
        segs = []

        # Depot-retailer segments
        for i in R:
            xv = int(round(x0[i, t].X))
            if xv > 0:
                rid = retailer_ids[i - 1]
                segs.append([supplier_id, rid, xv])

        # Retailer-retailer segments
        for i in R:
            for j in R:
                if i < j:
                    xv = x[(i, j, t)].X
                    if xv > 0.5:
                        rid_i = retailer_ids[i - 1]
                        rid_j = retailer_ids[j - 1]
                        segs.append([rid_i, rid_j, 1])

        routes[t_key] = segs

    for t in T_ext:
        supplier_inventory[str(t)] = float(S[t].X)

    for i in R:
        rid = retailer_ids[i - 1]
        retailer_inventory[str(rid)] = {}
        for t in T_ext:
            retailer_inventory[str(rid)][str(t)] = float(I[i, t].X)

    return {
        "objective_value": obj_val,
        "deliveries": deliveries,
        "routes": routes,
        "supplier_inventory": supplier_inventory,
        "retailer_inventory": retailer_inventory,
    }


def fallback_solution(instance):
    n = instance["num_retailers"]
    H = instance["time_horizon"]
    supplier = instance["supplier"]
    retailers = instance["retailers"]

    deliveries = {str(t): {} for t in range(1, H + 1)}
    routes = {str(t): [] for t in range(1, H + 1)}

    supplier_inventory = {str(1): float(supplier["starting_inventory"])}
    p = supplier["production_rate"]
    for t in range(1, H + 1):
        supplier_inventory[str(t + 1)] = float(supplier_inventory[str(t)] + p)

    retailer_inventory = {}
    for r in retailers:
        rid = str(r["id"])
        retailer_inventory[rid] = {}
        retailer_inventory[rid]["1"] = float(r["starting_inventory"])
        for t in range(1, H + 1):
            retailer_inventory[rid][str(t + 1)] = float(retailer_inventory[rid][str(t)] - r["demand"])

    return {
        "objective_value": 1e30,
        "deliveries": deliveries,
        "routes": routes,
        "supplier_inventory": supplier_inventory,
        "retailer_inventory": retailer_inventory,
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
    elapsed = time.time() - start

    if solution is None:
        solution = fallback_solution(instance)

    # Ensure objective is plain float
    solution["objective_value"] = float(solution["objective_value"])

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()
import argparse
import json
import math
import time
from typing import Dict, Tuple

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def build_and_solve(instance: dict, time_limit: int, logger: SolutionLogger = None):
    # --------------------------
    # Read instance data
    # --------------------------
    n = int(instance["n"])                  # number of customers
    T = int(instance["T"])                  # periods
    m = int(instance["m"])                  # vehicles
    Q = float(instance["Q"])                # vehicle capacity
    C = float(instance["C"])                # production capacity
    f = float(instance["f"])                # setup cost
    u_cost = float(instance["u"])           # unit production cost

    c = instance["transportation_costs"]    # (n+1)x(n+1), symmetric
    h = [float(v) for v in instance["h"]]   # holding costs, node 0..n
    L = [float(v) for v in instance["L"]]   # storage capacities, node 0..n
    I0 = [float(v) for v in instance["I0"]] # initial inventories, node 0..n
    sigma = [float(v) for v in instance["sigma"]]  # penalty for unmet demand, customer 1..n

    S = int(instance["n_scenarios"])
    probs = [float(p) for p in instance["scenario_probabilities"]]
    demand_scenarios = instance["demand_scenarios"]  # [S][n][T]

    nodes = list(range(n + 1))
    customers = list(range(1, n + 1))
    periods = list(range(1, T + 1))
    vehicles = list(range(1, m + 1))
    scenarios = list(range(1, S + 1))

    # Precompute future demand by scenario/customer/period (1-based period)
    future_dem = {}
    for w in scenarios:
        widx = w - 1
        for i in customers:
            iidx = i - 1
            running = 0.0
            for t in range(T, 0, -1):
                running += float(demand_scenarios[widx][iidx][t - 1])
                future_dem[(i, t, w)] = running

    # --------------------------
    # Build model
    # --------------------------
    model = gp.Model("stochastic_irp")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = float(time_limit)

    # First-stage variables
    y = model.addVars(periods, vtype=GRB.BINARY, name="y")  # setup
    z = model.addVars(nodes, vehicles, periods, vtype=GRB.BINARY, name="z")  # visit (node 0 = dispatch)
    a = model.addVars(
        [(i, j, k, t) for i in nodes for j in nodes if i != j for k in vehicles for t in periods],
        vtype=GRB.BINARY,
        name="a",
    )  # directed arc usage for route construction

    # MTZ order variables (per customer/vehicle/period)
    u_mtz = model.addVars(customers, vehicles, periods, lb=0.0, ub=float(n), vtype=GRB.CONTINUOUS, name="u_mtz")

    # Second-stage (scenario dependent)
    p = model.addVars(periods, scenarios, lb=0.0, vtype=GRB.CONTINUOUS, name="p")
    I = model.addVars(nodes, periods, scenarios, lb=0.0, vtype=GRB.CONTINUOUS, name="I")
    e = model.addVars(customers, periods, scenarios, lb=0.0, vtype=GRB.CONTINUOUS, name="e")
    q = model.addVars(customers, vehicles, periods, scenarios, lb=0.0, vtype=GRB.CONTINUOUS, name="q")

    # --------------------------
    # Routing/visit constraints
    # --------------------------
    # At most one vehicle visits each customer each period
    for i in customers:
        for t in periods:
            model.addConstr(gp.quicksum(z[i, k, t] for k in vehicles) <= 1, name=f"visit_once_{i}_{t}")

    for k in vehicles:
        for t in periods:
            # Depot degree (if dispatched)
            model.addConstr(gp.quicksum(a[0, j, k, t] for j in customers) == z[0, k, t], name=f"dep_out_{k}_{t}")
            model.addConstr(gp.quicksum(a[j, 0, k, t] for j in customers) == z[0, k, t], name=f"dep_in_{k}_{t}")

            for i in customers:
                # Customer degree if visited
                model.addConstr(
                    gp.quicksum(a[i, j, k, t] for j in nodes if j != i) == z[i, k, t],
                    name=f"out_{i}_{k}_{t}",
                )
                model.addConstr(
                    gp.quicksum(a[j, i, k, t] for j in nodes if j != i) == z[i, k, t],
                    name=f"in_{i}_{k}_{t}",
                )
                # Can visit customer only if vehicle dispatched
                model.addConstr(z[i, k, t] <= z[0, k, t], name=f"dispatch_link_{i}_{k}_{t}")

            # MTZ subtour elimination among customers
            for i in customers:
                model.addConstr(u_mtz[i, k, t] >= z[i, k, t], name=f"mtz_lb_{i}_{k}_{t}")
                model.addConstr(u_mtz[i, k, t] <= n * z[i, k, t], name=f"mtz_ub_{i}_{k}_{t}")

            for i in customers:
                for j in customers:
                    if i != j:
                        model.addConstr(
                            u_mtz[i, k, t] - u_mtz[j, k, t] + n * a[i, j, k, t] <= n - 1,
                            name=f"mtz_{i}_{j}_{k}_{t}",
                        )

    # --------------------------
    # Inventory/production/delivery constraints per scenario
    # --------------------------
    for w in scenarios:
        for t in periods:
            # Production capacity and setup link
            model.addConstr(p[t, w] <= C * y[t], name=f"prod_cap_{t}_{w}")

            # Plant inventory balance
            shipped_t = gp.quicksum(q[i, k, t, w] for i in customers for k in vehicles)
            if t == 1:
                model.addConstr(I0[0] + p[t, w] == shipped_t + I[0, t, w], name=f"plant_bal_{t}_{w}")
            else:
                model.addConstr(I[0, t - 1, w] + p[t, w] == shipped_t + I[0, t, w], name=f"plant_bal_{t}_{w}")
            model.addConstr(I[0, t, w] <= L[0], name=f"plant_store_{t}_{w}")

            # Vehicle capacity in period
            for k in vehicles:
                model.addConstr(
                    gp.quicksum(q[i, k, t, w] for i in customers) <= Q * z[0, k, t],
                    name=f"veh_cap_{k}_{t}_{w}",
                )

            # Customer balances and bounds
            for i in customers:
                demand_i_t = float(demand_scenarios[w - 1][i - 1][t - 1])
                delivered = gp.quicksum(q[i, k, t, w] for k in vehicles)

                if t == 1:
                    model.addConstr(
                        I0[i] + delivered + e[i, t, w] == demand_i_t + I[i, t, w],
                        name=f"cust_bal_{i}_{t}_{w}",
                    )
                else:
                    model.addConstr(
                        I[i, t - 1, w] + delivered + e[i, t, w] == demand_i_t + I[i, t, w],
                        name=f"cust_bal_{i}_{t}_{w}",
                    )

                # Storage feasibility at end of period considering arrivals before demand
                model.addConstr(I[i, t, w] + demand_i_t <= L[i], name=f"cust_store_{i}_{t}_{w}")

                # Delivery only if visited; also bounded by future demand from t onward
                fd = future_dem[(i, t, w)]
                Mq = max(0.0, min(Q, L[i], fd))
                for k in vehicles:
                    model.addConstr(q[i, k, t, w] <= Mq * z[i, k, t], name=f"deliver_link_{i}_{k}_{t}_{w}")

    # --------------------------
    # Objective
    # --------------------------
    first_stage_cost = gp.quicksum(f * y[t] for t in periods) + gp.quicksum(
        c[i][j] * a[i, j, k, t]
        for i in nodes for j in nodes if i != j
        for k in vehicles for t in periods
    )

    recourse_cost = gp.quicksum(
        probs[w - 1] * (
            gp.quicksum(u_cost * p[t, w] for t in periods)
            + gp.quicksum(h[0] * I[0, t, w] for t in periods)
            + gp.quicksum(h[i] * I[i, t, w] for i in customers for t in periods)
            + gp.quicksum(sigma[i - 1] * e[i, t, w] for i in customers for t in periods)
        )
        for w in scenarios
    )

    model.setObjective(first_stage_cost + recourse_cost, GRB.MINIMIZE)

    # --------------------------
    # Incumbent logging callback
    # --------------------------
    best_logged = [float("inf")]

    def cb(m, where):
        if logger is None:
            return
        if where == GRB.Callback.MIPSOL:
            obj = m.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj + 1e-9 < best_logged[0]:
                best_logged[0] = obj
                logger.log(float(obj))

    model.optimize(cb)

    # --------------------------
    # Build output solution
    # --------------------------
    sol = {
        "objective_value": None,
        "model_variables": {},
        "y": {},
        "z": {},
        "x": {},
    }

    if model.SolCount == 0:
        sol["objective_value"] = 1e30
        return sol

    sol["objective_value"] = float(model.ObjVal)

    tol = 1e-6
    model_vars = {}

    # y output (complete by period)
    for t in periods:
        vy = y[t].X
        sol["y"][f"{t}"] = int(vy > 0.5)
        if abs(vy) > tol:
            model_vars[f"y_{t}"] = float(vy)

    # z output (sparse nonzero)
    for i in nodes:
        for k in vehicles:
            for t in periods:
                vz = z[i, k, t].X
                if abs(vz) > tol:
                    key = f"{i}_{k}_{t}"
                    sol["z"][key] = int(vz > 0.5)
                    model_vars[f"z_{i}_{k}_{t}"] = float(vz)

    # x output from directed arc variables, as undirected traversal count
    for k in vehicles:
        for t in periods:
            for i in nodes:
                for j in range(i + 1, n + 1):
                    val = a[i, j, k, t].X + a[j, i, k, t].X
                    if abs(val) > tol:
                        key = f"{i}_{j}_{k}_{t}"
                        sol["x"][key] = int(round(val))
                        model_vars[f"x_{i}_{j}_{k}_{t}"] = float(val)

    # scenario variables in model_variables
    for t in periods:
        for w in scenarios:
            vp = p[t, w].X
            if abs(vp) > tol:
                model_vars[f"p_{t}_{w}"] = float(vp)

    for i in nodes:
        for t in periods:
            for w in scenarios:
                vI = I[i, t, w].X
                if abs(vI) > tol:
                    model_vars[f"I_{i}_{t}_{w}"] = float(vI)

    for i in customers:
        for t in periods:
            for w in scenarios:
                ve = e[i, t, w].X
                if abs(ve) > tol:
                    model_vars[f"e_{i}_{t}_{w}"] = float(ve)

    for i in customers:
        for k in vehicles:
            for t in periods:
                for w in scenarios:
                    vq = q[i, k, t, w].X
                    if abs(vq) > tol:
                        model_vars[f"q_{i}_{k}_{t}_{w}"] = float(vq)

    sol["model_variables"] = model_vars

    if logger is not None:
        logger.log_solution(sol["objective_value"], sol)

    return sol


def main():
    parser = argparse.ArgumentParser(description="Solve stochastic production-routing-inventory problem.")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to input instance JSON")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to output solution JSON")
    parser.add_argument("--time_limit", type=int, required=True, help="Time limit in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Optional JSONL path for incumbent logging")
    args = parser.parse_args()

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    solution = build_and_solve(instance, args.time_limit, logger)

    with open(args.solution_path, "w") as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()
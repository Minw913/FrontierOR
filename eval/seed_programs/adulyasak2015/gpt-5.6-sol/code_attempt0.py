import argparse
import json
import os
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


ZERO_TOL = 1e-8


def write_json(path, data):
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, allow_nan=False)


def clean_continuous(value):
    value = float(value)
    if abs(value) <= ZERO_TOL:
        return 0.0
    return value


def make_initial_solution(data):
    """Feasible no-production/no-delivery policy using unmet demand."""
    n = int(data["n"])
    T = int(data["T"])
    m = int(data["m"])
    W = int(data["n_scenarios"])

    h = [float(v) for v in data["h"]]
    I0 = [float(v) for v in data["I0"]]
    sigma = [float(v) for v in data["sigma"]]
    probs = [float(v) for v in data["scenario_probabilities"]]
    demand = data["demand_scenarios"]

    model_variables = {}
    y_out = {str(t + 1): 0 for t in range(T)}
    z_out = {}
    x_out = {}

    scenario_costs = [0.0] * W

    for w in range(W):
        plant_inventory = I0[0]
        for t in range(T):
            if plant_inventory > ZERO_TOL:
                model_variables[f"I_0_{t + 1}_{w + 1}"] = plant_inventory
            scenario_costs[w] += h[0] * plant_inventory

        for i in range(n):
            inventory = I0[i + 1]
            for t in range(T):
                d = float(demand[w][i][t])
                served = min(inventory, d)
                inventory -= served
                unmet = d - served

                if inventory > ZERO_TOL:
                    model_variables[f"I_{i + 1}_{t + 1}_{w + 1}"] = inventory
                if unmet > ZERO_TOL:
                    model_variables[f"e_{i + 1}_{t + 1}_{w + 1}"] = unmet

                scenario_costs[w] += (
                    h[i + 1] * inventory + sigma[i] * unmet
                )

    objective = sum(probs[w] * scenario_costs[w] for w in range(W))

    return {
        "objective_value": float(objective),
        "model_variables": model_variables,
        "y": y_out,
        "z": z_out,
        "x": x_out,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    start_time = time.monotonic()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    n = int(data["n"])
    T = int(data["T"])
    m = int(data["m"])
    Q = float(data["Q"])
    C = float(data["C"])
    fixed_cost = float(data["f"])
    production_cost = float(data["u"])
    W = int(data["n_scenarios"])

    travel = data["transportation_costs"]
    h = [float(v) for v in data["h"]]
    L = [float(v) for v in data["L"]]
    I0 = [float(v) for v in data["I0"]]
    sigma = [float(v) for v in data["sigma"]]
    probs = [float(v) for v in data["scenario_probabilities"]]
    demand = [
        [
            [float(data["demand_scenarios"][w][i][t]) for t in range(T)]
            for i in range(n)
        ]
        for w in range(W)
    ]

    fallback_solution = make_initial_solution(data)
    best_logged_objective = [float(fallback_solution["objective_value"])]

    if logger:
        logger.log_solution(best_logged_objective[0], fallback_solution)

    if args.time_limit <= 0 or time.monotonic() - start_time >= args.time_limit:
        write_json(args.solution_path, fallback_solution)
        return

    model = gp.Model("stochastic_production_inventory_routing")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1

    # First-stage variables.
    y = {
        t: model.addVar(vtype=GRB.BINARY, name=f"y_{t + 1}")
        for t in range(T)
    }

    z = {}
    for i in range(n + 1):
        for k in range(m):
            for t in range(T):
                z[i, k, t] = model.addVar(
                    vtype=GRB.BINARY,
                    name=f"z_{i}_{k + 1}_{t + 1}",
                )

    # Undirected edge variables. Depot-customer edges can be traversed twice
    # for a route serving one customer.
    x = {}
    for i in range(n + 1):
        for j in range(i + 1, n + 1):
            ub = 2.0 if i == 0 else 1.0
            for k in range(m):
                for t in range(T):
                    x[i, j, k, t] = model.addVar(
                        vtype=GRB.INTEGER,
                        lb=0.0,
                        ub=ub,
                        name=f"x_{i}_{j}_{k + 1}_{t + 1}",
                    )

    # Continuous connectivity flow variables, excluded from output.
    flow = {}
    for i in range(n + 1):
        for j in range(n + 1):
            if i == j:
                continue
            a, b = min(i, j), max(i, j)
            for k in range(m):
                for t in range(T):
                    flow[i, j, k, t] = model.addVar(
                        lb=0.0,
                        ub=float(n),
                        vtype=GRB.CONTINUOUS,
                        name=f"flow_{i}_{j}_{k + 1}_{t + 1}",
                    )

    # Scenario-dependent recourse variables.
    p = {}
    inv = {}
    emergency = {}
    q = {}

    for w in range(W):
        for t in range(T):
            p[t, w] = model.addVar(
                lb=0.0,
                ub=C,
                name=f"p_{t + 1}_{w + 1}",
            )

            for i in range(n + 1):
                inv[i, t, w] = model.addVar(
                    lb=0.0,
                    ub=L[i],
                    name=f"I_{i}_{t + 1}_{w + 1}",
                )

            for i in range(1, n + 1):
                d = demand[w][i - 1][t]
                emergency[i, t, w] = model.addVar(
                    lb=0.0,
                    ub=max(0.0, d),
                    name=f"e_{i}_{t + 1}_{w + 1}",
                )

                for k in range(m):
                    q[i, k, t, w] = model.addVar(
                        lb=0.0,
                        name=f"q_{i}_{k + 1}_{t + 1}_{w + 1}",
                    )

    model.update()

    # Routing degree and dispatch constraints.
    for k in range(m):
        for t in range(T):
            depot_degree = gp.quicksum(x[0, j, k, t] for j in range(1, n + 1))
            model.addConstr(
                depot_degree == 2 * z[0, k, t],
                name=f"depot_degree_{k}_{t}",
            )

            for i in range(1, n + 1):
                incident = gp.quicksum(
                    x[min(i, j), max(i, j), k, t]
                    for j in range(n + 1)
                    if j != i
                )
                model.addConstr(
                    incident == 2 * z[i, k, t],
                    name=f"degree_{i}_{k}_{t}",
                )
                model.addConstr(
                    z[i, k, t] <= z[0, k, t],
                    name=f"dispatch_link_{i}_{k}_{t}",
                )

            # Single-commodity connectivity flow.
            for i in range(n + 1):
                for j in range(n + 1):
                    if i == j:
                        continue
                    a, b = min(i, j), max(i, j)
                    model.addConstr(
                        flow[i, j, k, t] <= n * x[a, b, k, t],
                        name=f"flow_edge_{i}_{j}_{k}_{t}",
                    )

            for i in range(1, n + 1):
                inflow = gp.quicksum(
                    flow[j, i, k, t] for j in range(n + 1) if j != i
                )
                outflow = gp.quicksum(
                    flow[i, j, k, t] for j in range(n + 1) if j != i
                )
                model.addConstr(
                    inflow - outflow == z[i, k, t],
                    name=f"flow_balance_{i}_{k}_{t}",
                )

    # At most one vehicle visits each customer in a period.
    for i in range(1, n + 1):
        for t in range(T):
            model.addConstr(
                gp.quicksum(z[i, k, t] for k in range(m)) <= 1,
                name=f"one_vehicle_{i}_{t}",
            )

    # Vehicle symmetry breaking.
    for t in range(T):
        for k in range(m - 1):
            model.addConstr(
                z[0, k, t] >= z[0, k + 1, t],
                name=f"vehicle_order_{k}_{t}",
            )

    # Scenario-dependent production, inventory, delivery, and capacity constraints.
    for w in range(W):
        for t in range(T):
            model.addConstr(
                p[t, w] <= C * y[t],
                name=f"production_setup_{t}_{w}",
            )

            total_shipments = gp.quicksum(
                q[i, k, t, w]
                for i in range(1, n + 1)
                for k in range(m)
            )
            previous_plant_inventory = I0[0] if t == 0 else inv[0, t - 1, w]
            model.addConstr(
                previous_plant_inventory + p[t, w]
                == total_shipments + inv[0, t, w],
                name=f"plant_balance_{t}_{w}",
            )

            for k in range(m):
                vehicle_load = gp.quicksum(
                    q[i, k, t, w] for i in range(1, n + 1)
                )
                model.addConstr(
                    vehicle_load <= Q * z[0, k, t],
                    name=f"vehicle_capacity_{k}_{t}_{w}",
                )

            for i in range(1, n + 1):
                previous_inventory = I0[i] if t == 0 else inv[i, t - 1, w]
                delivered = gp.quicksum(q[i, k, t, w] for k in range(m))
                d = demand[w][i - 1][t]

                model.addConstr(
                    previous_inventory
                    + delivered
                    + emergency[i, t, w]
                    == d + inv[i, t, w],
                    name=f"customer_balance_{i}_{t}_{w}",
                )

                # Deliveries arrive before demand is consumed.
                model.addConstr(
                    previous_inventory + delivered <= L[i],
                    name=f"arrival_storage_{i}_{t}_{w}",
                )

                # Storage convention stated in the problem.
                model.addConstr(
                    inv[i, t, w] + d <= L[i],
                    name=f"ending_storage_{i}_{t}_{w}",
                )

                future_demand = sum(demand[w][i - 1][r] for r in range(t, T))
                delivery_bound = max(0.0, min(Q, L[i], future_demand))

                for k in range(m):
                    model.addConstr(
                        q[i, k, t, w]
                        <= delivery_bound * z[i, k, t],
                        name=f"delivery_visit_{i}_{k}_{t}_{w}",
                    )

    # Nonanticipativity: scenarios with identical demand history through period t
    # share all recourse decisions made in period t.
    for t in range(T):
        history_groups = {}
        for w in range(W):
            history = tuple(
                demand[w][i][r]
                for r in range(t + 1)
                for i in range(n)
            )
            history_groups.setdefault(history, []).append(w)

        for group in history_groups.values():
            if len(group) <= 1:
                continue
            representative = group[0]
            for w in group[1:]:
                model.addConstr(p[t, w] == p[t, representative])
                for i in range(n + 1):
                    model.addConstr(
                        inv[i, t, w] == inv[i, t, representative]
                    )
                for i in range(1, n + 1):
                    model.addConstr(
                        emergency[i, t, w]
                        == emergency[i, t, representative]
                    )
                    for k in range(m):
                        model.addConstr(
                            q[i, k, t, w]
                            == q[i, k, t, representative]
                        )

    first_stage_cost = (
        fixed_cost * gp.quicksum(y[t] for t in range(T))
        + gp.quicksum(
            float(travel[i][j]) * x[i, j, k, t]
            for i in range(n + 1)
            for j in range(i + 1, n + 1)
            for k in range(m)
            for t in range(T)
        )
    )

    expected_recourse_cost = gp.LinExpr()
    for w in range(W):
        scenario_cost = gp.LinExpr()
        for t in range(T):
            scenario_cost += production_cost * p[t, w]
            scenario_cost += h[0] * inv[0, t, w]
            for i in range(1, n + 1):
                scenario_cost += h[i] * inv[i, t, w]
                scenario_cost += sigma[i - 1] * emergency[i, t, w]
        expected_recourse_cost += probs[w] * scenario_cost

    model.setObjective(first_stage_cost + expected_recourse_cost, GRB.MINIMIZE)

    # Output-variable metadata.
    output_meta = []

    for t in range(T):
        output_meta.append((f"y_{t + 1}", y[t], "binary", "y", (t,)))

    for i in range(n + 1):
        for k in range(m):
            for t in range(T):
                output_meta.append(
                    (
                        f"z_{i}_{k + 1}_{t + 1}",
                        z[i, k, t],
                        "binary",
                        "z",
                        (i, k, t),
                    )
                )

    for i in range(n + 1):
        for j in range(i + 1, n + 1):
            for k in range(m):
                for t in range(T):
                    output_meta.append(
                        (
                            f"x_{i}_{j}_{k + 1}_{t + 1}",
                            x[i, j, k, t],
                            "integer",
                            "x",
                            (i, j, k, t),
                        )
                    )

    for w in range(W):
        for t in range(T):
            output_meta.append(
                (
                    f"p_{t + 1}_{w + 1}",
                    p[t, w],
                    "continuous",
                    "other",
                    (),
                )
            )
            for i in range(n + 1):
                output_meta.append(
                    (
                        f"I_{i}_{t + 1}_{w + 1}",
                        inv[i, t, w],
                        "continuous",
                        "other",
                        (),
                    )
                )
            for i in range(1, n + 1):
                output_meta.append(
                    (
                        f"e_{i}_{t + 1}_{w + 1}",
                        emergency[i, t, w],
                        "continuous",
                        "other",
                        (),
                    )
                )
                for k in range(m):
                    output_meta.append(
                        (
                            f"q_{i}_{k + 1}_{t + 1}_{w + 1}",
                            q[i, k, t, w],
                            "continuous",
                            "other",
                            (),
                        )
                    )

    output_vars = [entry[1] for entry in output_meta]

    def build_solution(values, objective):
        model_variables = {}
        y_result = {str(t + 1): 0 for t in range(T)}
        z_result = {}
        x_result = {}

        for entry, raw_value in zip(output_meta, values):
            name, _, kind, category, indices = entry

            if kind in ("binary", "integer"):
                value = int(round(float(raw_value)))
            else:
                value = clean_continuous(raw_value)
                if value < 0.0 and value > -1e-7:
                    value = 0.0

            if category == "y":
                t = indices[0]
                y_result[str(t + 1)] = int(value)
            elif category == "z" and value != 0:
                i, k, t = indices
                z_result[f"{i}_{k + 1}_{t + 1}"] = int(value)
            elif category == "x" and value != 0:
                i, j, k, t = indices
                x_result[f"{i}_{j}_{k + 1}_{t + 1}"] = int(value)

            if value != 0:
                model_variables[name] = value

        return {
            "objective_value": float(objective),
            "model_variables": model_variables,
            "y": y_result,
            "z": z_result,
            "x": x_result,
        }

    # Complete no-action MIP start.
    for var in model.getVars():
        var.Start = 0.0

    for w in range(W):
        plant_inventory = I0[0]
        for t in range(T):
            inv[0, t, w].Start = plant_inventory

        for i in range(1, n + 1):
            inventory = I0[i]
            for t in range(T):
                d = demand[w][i - 1][t]
                served = min(inventory, d)
                inventory -= served
                unmet = d - served
                inv[i, t, w].Start = inventory
                emergency[i, t, w].Start = unmet

    remaining_time = max(0.01, args.time_limit - (time.monotonic() - start_time))
    model.Params.TimeLimit = remaining_time

    def incumbent_callback(cb_model, where):
        if where != GRB.Callback.MIPSOL or logger is None:
            return
        try:
            objective = float(cb_model.cbGet(GRB.Callback.MIPSOL_OBJ))
            if objective >= best_logged_objective[0] - 1e-7:
                return
            values = cb_model.cbGetSolution(output_vars)
            solution = build_solution(values, objective)
            logger.log_solution(objective, solution)
            best_logged_objective[0] = objective
        except Exception:
            # Logging must not interrupt optimization.
            pass

    try:
        model.optimize(incumbent_callback)
    except gp.GurobiError:
        write_json(args.solution_path, fallback_solution)
        return

    if model.SolCount > 0:
        final_values = [var.X for var in output_vars]
        final_solution = build_solution(final_values, model.ObjVal)

        if logger and final_solution["objective_value"] < best_logged_objective[0] - 1e-7:
            logger.log_solution(final_solution["objective_value"], final_solution)
            best_logged_objective[0] = final_solution["objective_value"]
    else:
        final_solution = fallback_solution

    write_json(args.solution_path, final_solution)


if __name__ == "__main__":
    main()
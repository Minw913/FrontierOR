import argparse
import json
import os
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    return parser.parse_args()


def write_json(path, data):
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"), allow_nan=False)
    os.replace(tmp_path, path)


def main():
    args = parse_args()
    start_time = time.monotonic()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    I = range(int(data["num_demand_sites"]))
    J = range(int(data["num_production_sites"]))
    S = range(int(data["num_scenarios"]))
    T = range(int(data["num_time_periods"]))
    M = range(int(data["num_module_sizes"]))
    K = range(int(data["units_per_module_size"]))

    num_units_per_size = int(data["units_per_module_size"])
    u_max = float(data["u_max"])

    capacities = [
        float(data["module_sizes"][m]["capacity"]) for m in M
    ]

    default_capital_costs = [
        float(data["module_sizes"][m]["capital_cost_per_unit"]) for m in M
    ]
    capital_costs = {
        (m, k): default_capital_costs[m] for m in M for k in K
    }
    for unit in data.get("module_units", []):
        m = int(unit["module_size_id"])
        k = int(unit["unit_id"])
        if m in M and k in K:
            capital_costs[m, k] = float(
                unit.get("capital_cost", default_capital_costs[m])
            )

    production_costs = data["production_costs"]
    relocation_costs = data["relocation_costs"]
    penalty_costs = data["penalty_costs"]

    scenarios = data["scenarios"]
    demands = {}
    probabilities = {}
    top_probabilities = data.get("scenario_probabilities", [])

    for s in S:
        scenario = scenarios[s]
        probabilities[s] = float(
            scenario.get(
                "probability",
                top_probabilities[s] if s < len(top_probabilities) else 0.0,
            )
        )
        scenario_demands = scenario["demands"]
        for i in I:
            for t in T:
                demands[i, t, s] = float(scenario_demands[i][t])

    model = gp.Model("modular_supply_chain")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1

    # First-stage purchase decisions.
    y = model.addVars(M, K, vtype=GRB.BINARY, name="y")

    # Scenario copies of the purchased count, included to match the canonical
    # witness and to connect first- and second-stage decisions.
    z = model.addVars(
        M, S, lb=0, ub=num_units_per_size, vtype=GRB.INTEGER, name="z"
    )

    # Served and unmet demand fractions.
    x = model.addVars(I, J, T, S, lb=0.0, ub=1.0, name="x")
    q = model.addVars(I, T, S, lb=0.0, ub=1.0, name="q")

    # Inventories.
    v = model.addVars(
        J, M, T, S,
        lb=0,
        ub=num_units_per_size,
        vtype=GRB.INTEGER,
        name="v",
    )
    v_depot = model.addVars(
        M, T, S,
        lb=0,
        ub=num_units_per_size,
        vtype=GRB.INTEGER,
        name="v_depot",
    )

    # Relocations.
    distinct_pairs = [(j, jp) for j in J for jp in J if j != jp]
    w = model.addVars(
        distinct_pairs, M, T, S,
        lb=0,
        ub=num_units_per_size,
        vtype=GRB.INTEGER,
        name="w",
    )
    w_fd = model.addVars(
        J, M, T, S,
        lb=0,
        ub=num_units_per_size,
        vtype=GRB.INTEGER,
        name="w_fd",
    )
    w_td = model.addVars(
        J, M, T, S,
        lb=0,
        ub=num_units_per_size,
        vtype=GRB.INTEGER,
        name="w_td",
    )

    # Purchased count must be identical in every scenario.
    for m in M:
        purchased = gp.quicksum(y[m, k] for k in K)
        for s in S:
            model.addConstr(z[m, s] == purchased)

    # Since units of a size are operationally interchangeable, an optimal
    # solution can always buy them in nondecreasing capital-cost order.
    for m in M:
        ordered_units = sorted(K, key=lambda k: (capital_costs[m, k], k))
        for p in range(len(ordered_units) - 1):
            model.addConstr(
                y[m, ordered_units[p]] >= y[m, ordered_units[p + 1]]
            )

    # Every demand fraction is either served or unmet.
    for i in I:
        for t in T:
            for s in S:
                model.addConstr(
                    gp.quicksum(x[i, j, t, s] for j in J) + q[i, t, s] == 1.0
                )
                if abs(demands[i, t, s]) <= 1e-12:
                    for j in J:
                        x[i, j, t, s].UB = 0.0
                    q[i, t, s].LB = 1.0

    # Inventory conservation at the depot and production sites.
    for m in M:
        for s in S:
            for t in T:
                previous_depot = z[m, s] if t == 0 else v_depot[m, t - 1, s]
                model.addConstr(
                    v_depot[m, t, s]
                    == previous_depot
                    + gp.quicksum(w_td[j, m, t, s] for j in J)
                    - gp.quicksum(w_fd[j, m, t, s] for j in J)
                )

                for j in J:
                    previous_site = 0 if t == 0 else v[j, m, t - 1, s]
                    incoming = (
                        w_fd[j, m, t, s]
                        + gp.quicksum(
                            w[jp, j, m, t, s] for jp in J if jp != j
                        )
                    )
                    outgoing = (
                        w_td[j, m, t, s]
                        + gp.quicksum(
                            w[j, jp, m, t, s] for jp in J if jp != j
                        )
                    )
                    model.addConstr(
                        v[j, m, t, s] == previous_site + incoming - outgoing
                    )

    # Production capacity constraints.
    for j in J:
        for t in T:
            for s in S:
                production = gp.quicksum(
                    demands[i, t, s] * x[i, j, t, s] for i in I
                )
                installed_capacity = gp.quicksum(
                    capacities[m] * v[j, m, t, s] for m in M
                )
                model.addConstr(production <= installed_capacity)
                model.addConstr(production <= u_max)

    capital_cost_expr = gp.quicksum(
        capital_costs[m, k] * y[m, k] for m in M for k in K
    )

    expected_operating_cost = gp.LinExpr()
    for s in S:
        scenario_cost = gp.LinExpr()

        scenario_cost += gp.quicksum(
            float(production_costs[i][j])
            * demands[i, t, s]
            * x[i, j, t, s]
            for i in I for j in J for t in T
        )

        scenario_cost += gp.quicksum(
            float(relocation_costs[j][jp][m]) * w[j, jp, m, t, s]
            for j, jp in distinct_pairs for m in M for t in T
        )

        scenario_cost += gp.quicksum(
            float(penalty_costs[i][t])
            * demands[i, t, s]
            * q[i, t, s]
            for i in I for t in T
        )

        expected_operating_cost += probabilities[s] * scenario_cost

    model.setObjective(capital_cost_expr + expected_operating_cost, GRB.MINIMIZE)
    model.update()

    all_vars = model.getVars()

    def integer_value(get_value, var):
        value = float(get_value(var))
        rounded = int(round(value))
        return max(0, rounded)

    def continuous_value(get_value, var):
        value = float(get_value(var))
        if abs(value) < 1e-12:
            return 0.0
        if abs(value - 1.0) < 1e-12:
            return 1.0
        return value

    def build_solution(get_value):
        model_variables = {}
        y_output = {}

        y_values = {}
        z_values = {}
        x_values = {}
        q_values = {}
        v_values = {}
        depot_values = {}
        w_values = {}
        fd_values = {}
        td_values = {}

        for m in M:
            for k in K:
                value = integer_value(get_value, y[m, k])
                y_values[m, k] = value
                y_output[f"y_{m}_{k}"] = value
                if value:
                    model_variables[f"y[{m},{k}]"] = value

        for m in M:
            for s in S:
                value = integer_value(get_value, z[m, s])
                z_values[m, s] = value
                if value:
                    model_variables[f"z[{m},{s}]"] = value

        for i in I:
            for j in J:
                for t in T:
                    for s in S:
                        value = continuous_value(get_value, x[i, j, t, s])
                        x_values[i, j, t, s] = value
                        if abs(value) > 1e-12:
                            model_variables[f"x[{i},{j},{t},{s}]"] = value

        for i in I:
            for t in T:
                for s in S:
                    value = continuous_value(get_value, q[i, t, s])
                    q_values[i, t, s] = value
                    if abs(value) > 1e-12:
                        model_variables[f"q[{i},{t},{s}]"] = value

        for j in J:
            for m in M:
                for t in T:
                    for s in S:
                        value = integer_value(get_value, v[j, m, t, s])
                        v_values[j, m, t, s] = value
                        if value:
                            model_variables[f"v[{j},{m},{t},{s}]"] = value

        for m in M:
            for t in T:
                for s in S:
                    value = integer_value(get_value, v_depot[m, t, s])
                    depot_values[m, t, s] = value
                    if value:
                        model_variables[f"v_depot[{m},{t},{s}]"] = value

        for j, jp in distinct_pairs:
            for m in M:
                for t in T:
                    for s in S:
                        value = integer_value(get_value, w[j, jp, m, t, s])
                        w_values[j, jp, m, t, s] = value
                        if value:
                            model_variables[
                                f"w[{j},{jp},{m},{t},{s}]"
                            ] = value

        for j in J:
            for m in M:
                for t in T:
                    for s in S:
                        value = integer_value(get_value, w_fd[j, m, t, s])
                        fd_values[j, m, t, s] = value
                        if value:
                            model_variables[
                                f"w_fd[{j},{m},{t},{s}]"
                            ] = value

                        value = integer_value(get_value, w_td[j, m, t, s])
                        td_values[j, m, t, s] = value
                        if value:
                            model_variables[
                                f"w_td[{j},{m},{t},{s}]"
                            ] = value

        objective = sum(
            capital_costs[m, k] * y_values[m, k] for m in M for k in K
        )

        for s in S:
            scenario_cost = 0.0

            for i in I:
                for j in J:
                    serving_cost = float(production_costs[i][j])
                    for t in T:
                        scenario_cost += (
                            serving_cost
                            * demands[i, t, s]
                            * x_values[i, j, t, s]
                        )

            for j, jp in distinct_pairs:
                for m in M:
                    move_cost = float(relocation_costs[j][jp][m])
                    for t in T:
                        scenario_cost += (
                            move_cost * w_values[j, jp, m, t, s]
                        )

            for i in I:
                for t in T:
                    scenario_cost += (
                        float(penalty_costs[i][t])
                        * demands[i, t, s]
                        * q_values[i, t, s]
                    )

            objective += probabilities[s] * scenario_cost

        return {
            "objective_value": float(objective),
            "model_variables": model_variables,
            "y": y_output,
        }

    # A guaranteed feasible MIP start: buy nothing and leave all demand unmet.
    for var in all_vars:
        var.Start = 0.0
    for i in I:
        for t in T:
            for s in S:
                q[i, t, s].Start = 1.0

    def fallback_get_value(var):
        name = var.VarName
        return 1.0 if name.startswith("q[") else 0.0

    best_solution = build_solution(fallback_get_value)
    best_logged_objective = [best_solution["objective_value"]]

    if logger:
        logger.log_solution(best_solution["objective_value"], best_solution)

    remaining = max(0.0, float(args.time_limit) - (time.monotonic() - start_time))

    if remaining > 0.001:
        model.Params.TimeLimit = remaining

        def callback(cb_model, where):
            if where != GRB.Callback.MIPSOL:
                return
            try:
                values = cb_model.cbGetSolution(all_vars)
                by_index = {
                    var.index: float(value)
                    for var, value in zip(all_vars, values)
                }

                def callback_get_value(var):
                    return by_index[var.index]

                candidate = build_solution(callback_get_value)
                objective = candidate["objective_value"]
                tolerance = 1e-7 * max(1.0, abs(best_logged_objective[0]))
                if objective < best_logged_objective[0] - tolerance:
                    best_logged_objective[0] = objective
                    if logger:
                        logger.log_solution(objective, candidate)
            except Exception:
                # Logging must not interrupt the optimization.
                pass

        try:
            model.optimize(callback)
        except gp.GurobiError:
            pass

        if model.SolCount > 0:
            solution_values = [var.X for var in all_vars]
            by_index = {
                var.index: float(value)
                for var, value in zip(all_vars, solution_values)
            }

            def final_get_value(var):
                return by_index[var.index]

            candidate = build_solution(final_get_value)
            if (
                candidate["objective_value"]
                <= best_solution["objective_value"] + 1e-7
            ):
                best_solution = candidate

            tolerance = 1e-7 * max(
                1.0, abs(best_logged_objective[0])
            )
            if (
                logger
                and best_solution["objective_value"]
                < best_logged_objective[0] - tolerance
            ):
                logger.log_solution(
                    best_solution["objective_value"], best_solution
                )

    write_json(args.solution_path, best_solution)


if __name__ == "__main__":
    main()
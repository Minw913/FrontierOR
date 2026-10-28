import argparse
import json
import math
import sys
import time

from solution_logger import SolutionLogger

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception:
    gp = None
    GRB = None


PATTERNS = {
    "y[m,k]": "m=0..num_module_sizes-1, k=0..units_per_module_size-1",
    "z[m,s]": "module size m, scenario s",
    "x[i,j,t,s]": "demand site i, production site j, period t, scenario s",
    "q[i,t,s]": "unmet-demand fraction",
    "v[j,m,t,s]": "units at production site",
    "v_depot[m,t,s]": "units at depot",
    "w[j,jp,m,t,s]": "relocation between distinct production sites j and jp",
    "w_fd[j,m,t,s]": "depot-to-production-site relocation",
    "w_td[j,m,t,s]": "production-site-to-depot relocation",
}


def as_period_vector(cell, num_periods):
    """Return a length-num_periods vector, broadcasting scalar schema values."""
    if isinstance(cell, (list, tuple)):
        values = [float(v) for v in cell]
        if len(values) == num_periods:
            return values
        if len(values) == 1:
            return values * num_periods
        if len(values) > num_periods:
            return values[:num_periods]
        return values + [values[-1]] * (num_periods - len(values))
    return [float(cell)] * num_periods


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()
    logger = (
        SolutionLogger(args.log_path, sense="minimize")
        if args.log_path
        else None
    )

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    num_demand_sites = int(data.get("num_demand_sites", len(data.get("demand_sites", []))))
    num_production_sites = int(data.get("num_production_sites", len(data.get("production_sites", []))))
    num_module_sizes = int(data.get("num_module_sizes", len(data.get("module_sizes", []))))
    num_scenarios = int(data.get("num_scenarios", len(data.get("scenarios", []))))
    num_periods = int(data["num_time_periods"])
    units_per_size = int(data.get("units_per_module_size", 0))
    u_max = float(data.get("u_max", math.inf))

    demand_sites = data.get("demand_sites", [])
    production_sites = data.get("production_sites", [])
    module_sizes = data.get("module_sizes", [])
    scenarios = data.get("scenarios", [])

    I = range(num_demand_sites)
    J = range(num_production_sites)
    M = range(num_module_sizes)
    T = range(num_periods)
    S = range(num_scenarios)
    K = range(units_per_size)

    module_capacity = [
        float(module_sizes[m].get("capacity", 0.0))
        for m in range(num_module_sizes)
    ]
    fallback_unit_cost = [
        float(module_sizes[m].get("capital_cost_per_unit", 0.0))
        for m in range(num_module_sizes)
    ]

    explicit_unit_cost = {}
    for unit in data.get("module_units", []):
        m_id = int(unit.get("module_size_id", -1))
        k_id = int(unit.get("unit_id", -1))
        if 0 <= m_id < num_module_sizes and 0 <= k_id < units_per_size:
            explicit_unit_cost[(m_id, k_id)] = float(
                unit.get("capital_cost", fallback_unit_cost[m_id])
            )

    unit_cost = [
        [
            explicit_unit_cost.get((m, k), fallback_unit_cost[m])
            for k in K
        ]
        for m in M
    ]

    production_costs_raw = data.get("production_costs", [])
    service_cost = [
        [
            as_period_vector(production_costs_raw[i][j], num_periods)
            for j in J
        ]
        for i in I
    ]

    relocation_costs_raw = data.get("relocation_costs", [])
    relocation_cost = []
    for j in J:
        relocation_cost.append([])
        for jp in J:
            if j == jp:
                relocation_cost[j].append([[0.0] * num_periods for _ in M])
            else:
                relocation_cost[j].append(
                    [
                        as_period_vector(relocation_costs_raw[j][jp][m], num_periods)
                        for m in M
                    ]
                )

    penalty_cost = [
        [float(v) for v in data["penalty_costs"][i]]
        for i in I
    ]

    demands = []
    probabilities = []
    top_level_probabilities = data.get("scenario_probabilities", [])
    for s in S:
        scenario = scenarios[s]
        probabilities.append(
            float(
                scenario.get(
                    "probability",
                    top_level_probabilities[s] if s < len(top_level_probabilities) else 0.0,
                )
            )
        )
        demands.append(
            [
                [float(v) for v in scenario["demands"][i]]
                for i in I
            ]
        )

    # A penalty-only solution is always feasible: purchase nothing and leave all
    # demand unmet.  This guarantees output even under a very strict time limit.
    best_objective = 0.0
    for s in S:
        for i in I:
            for t in T:
                best_objective += (
                    probabilities[s]
                    * penalty_cost[i][t]
                    * demands[s][i][t]
                )

    y_dict = {
        f"y_{m}_{k}": 0
        for m in M
        for k in K
    }
    trivial_model_variables = {
        f"q[{i},{t},{s}]": 1.0
        for s in S
        for i in I
        for t in T
    }
    best_payload = {
        "objective_value": float(best_objective),
        "model_variables": trivial_model_variables,
        "y": y_dict,
        "model_variable_name_patterns": PATTERNS,
    }

    if logger:
        logger.log_solution(best_objective, best_payload)

    epsilon = 1e-9
    integer_tolerance = 1e-6
    model = None
    variable_specs = []
    model_ready = False

    def build_payload(objective_value, values):
        model_variables = {}
        complete_y = {
            f"y_{m}_{k}": 0
            for m in M
            for k in K
        }

        for (var, key_name, kind), raw_value in zip(variable_specs, values):
            raw_value = float(raw_value)
            if kind == "fraction":
                value = raw_value
                if value < 0.0 and value > -integer_tolerance:
                    value = 0.0
                if value > 1.0 and value < 1.0 + integer_tolerance:
                    value = 1.0
                value = min(1.0, max(0.0, value))
                if value > epsilon:
                    model_variables[key_name] = value
            else:
                value = int(round(raw_value))
                if kind == "y":
                    underscore_key = key_name.replace("[", "_").replace("]", "")
                    underscore_key = underscore_key.replace(",", "_")
                    complete_y[underscore_key] = value
                if value != 0:
                    model_variables[key_name] = value

        return {
            "objective_value": float(objective_value),
            "model_variables": model_variables,
            "y": complete_y,
            "model_variable_name_patterns": PATTERNS,
        }

    remaining = args.time_limit - (time.time() - start_time)
    if gp is not None and remaining > 0:
        try:
            model = gp.Model("modular_supply_chain")
            model.Params.Seed = 0
            model.Params.MIPGap = 1e-4
            model.Params.NumericFocus = 0
            model.Params.Threads = 1
            model.Params.TimeLimit = max(0.0, remaining)

            y_index = [(m, k) for m in M for k in K]
            z_index = [(m, s) for m in M for s in S]
            x_index = [
                (i, j, t, s)
                for s in S
                for i in I
                for j in J
                for t in T
            ]
            q_index = [(i, t, s) for s in S for i in I for t in T]
            v_index = [
                (j, m, t, s)
                for s in S
                for j in J
                for m in M
                for t in T
            ]
            depot_index = [(m, t, s) for s in S for m in M for t in T]
            production_pair_index = [
                (j, jp, m, t, s)
                for s in S
                for j in J
                for jp in J
                if j != jp
                for m in M
                for t in T
            ]
            from_depot_index = [
                (j, m, t, s)
                for s in S
                for j in J
                for m in M
                for t in T
            ]

            y = model.addVars(
                y_index,
                vtype=GRB.BINARY,
                name="y",
            )
            z = model.addVars(
                z_index,
                vtype=GRB.INTEGER,
                lb=0,
                ub=units_per_size,
                name="z",
            )
            x = model.addVars(
                x_index,
                vtype=GRB.CONTINUOUS,
                lb=0.0,
                ub=1.0,
                name="x",
            )
            q = model.addVars(
                q_index,
                vtype=GRB.CONTINUOUS,
                lb=0.0,
                ub=1.0,
                name="q",
            )
            v = model.addVars(
                v_index,
                vtype=GRB.INTEGER,
                lb=0,
                ub=units_per_size,
                name="v",
            )
            v_depot = model.addVars(
                depot_index,
                vtype=GRB.INTEGER,
                lb=0,
                ub=units_per_size,
                name="v_depot",
            )
            w = model.addVars(
                production_pair_index,
                vtype=GRB.INTEGER,
                lb=0,
                ub=units_per_size,
                name="w",
            )
            w_fd = model.addVars(
                from_depot_index,
                vtype=GRB.INTEGER,
                lb=0,
                ub=units_per_size,
                name="w_fd",
            )
            w_td = model.addVars(
                from_depot_index,
                vtype=GRB.INTEGER,
                lb=0,
                ub=units_per_size,
                name="w_td",
            )

            variable_specs = []
            for (m, k), var in y.items():
                variable_specs.append((var, f"y[{m},{k}]", "y"))
            for (m, s), var in z.items():
                variable_specs.append((var, f"z[{m},{s}]", "integer"))
            for (i, j, t, s), var in x.items():
                variable_specs.append((var, f"x[{i},{j},{t},{s}]", "fraction"))
            for (i, t, s), var in q.items():
                variable_specs.append((var, f"q[{i},{t},{s}]", "fraction"))
            for (j, m, t, s), var in v.items():
                variable_specs.append((var, f"v[{j},{m},{t},{s}]", "integer"))
            for (m, t, s), var in v_depot.items():
                variable_specs.append((var, f"v_depot[{m},{t},{s}]", "integer"))
            for (j, jp, m, t, s), var in w.items():
                variable_specs.append((var, f"w[{j},{jp},{m},{t},{s}]", "integer"))
            for (j, m, t, s), var in w_fd.items():
                variable_specs.append((var, f"w_fd[{j},{m},{t},{s}]", "integer"))
            for (j, m, t, s), var in w_td.items():
                variable_specs.append((var, f"w_td[{j},{m},{t},{s}]", "integer"))

            flat_variables = [spec[0] for spec in variable_specs]

            purchase_expr = {
                m: gp.quicksum(y[m, k] for k in K)
                for m in M
            }

            # Identical equal-cost units create symmetry; selecting the first
            # available unit indices is an equivalent symmetry-breaking choice.
            for m in M:
                if units_per_size > 1 and all(
                    abs(unit_cost[m][k] - unit_cost[m][0]) < 1e-9
                    for k in K
                ):
                    for k in range(units_per_size - 1):
                        model.addConstr(y[m, k] >= y[m, k + 1])

            for s in S:
                for m in M:
                    # Aggregate purchased pool available in this scenario.
                    model.addConstr(z[m, s] == purchase_expr[m])

                for i in I:
                    for t in T:
                        model.addConstr(
                            gp.quicksum(x[i, j, t, s] for j in J)
                            + q[i, t, s]
                            == 1.0
                        )

                for j in J:
                    for t in T:
                        production_expr = gp.quicksum(
                            demands[s][i][t] * x[i, j, t, s]
                            for i in I
                        )
                        capacity_expr = gp.quicksum(
                            module_capacity[m] * v[j, m, t, s]
                            for m in M
                        )
                        model.addConstr(production_expr <= capacity_expr)
                        if not math.isinf(u_max):
                            model.addConstr(production_expr <= u_max)

                for m in M:
                    for t in T:
                        depot_net = gp.quicksum(
                            w_td[j, m, t, s] for j in J
                        ) - gp.quicksum(
                            w_fd[j, m, t, s] for j in J
                        )
                        if t == 0:
                            model.addConstr(
                                v_depot[m, t, s] == z[m, s] + depot_net
                            )
                        else:
                            model.addConstr(
                                v_depot[m, t, s]
                                == v_depot[m, t - 1, s] + depot_net
                            )

                    for j in J:
                        for t in T:
                            site_net = (
                                w_fd[j, m, t, s]
                                - w_td[j, m, t, s]
                                + gp.quicksum(
                                    w[jp, j, m, t, s]
                                    for jp in J
                                    if jp != j
                                )
                                - gp.quicksum(
                                    w[j, jp, m, t, s]
                                    for jp in J
                                    if jp != j
                                )
                            )
                            if t == 0:
                                model.addConstr(
                                    v[j, m, t, s] == site_net
                                )
                            else:
                                model.addConstr(
                                    v[j, m, t, s]
                                    == v[j, m, t - 1, s] + site_net
                                )

            objective = gp.quicksum(
                unit_cost[m][k] * y[m, k]
                for m in M
                for k in K
            )

            for s in S:
                operating_expr = gp.LinExpr()
                for j in J:
                    for t in T:
                        for i in I:
                            operating_expr += (
                                service_cost[i][j][t]
                                * demands[s][i][t]
                                * x[i, j, t, s]
                            )
                for j in J:
                    for jp in J:
                        if j == jp:
                            continue
                        for m in M:
                            for t in T:
                                operating_expr += (
                                    relocation_cost[j][jp][m][t]
                                    * w[j, jp, m, t, s]
                                )
                for i in I:
                    for t in T:
                        operating_expr += (
                            penalty_cost[i][t]
                            * demands[s][i][t]
                            * q[i, t, s]
                        )
                objective += probabilities[s] * operating_expr

            model.setObjective(objective, GRB.MINIMIZE)
            model_ready = True

            def callback(cb_model, where):
                nonlocal best_objective, best_payload
                if where == GRB.Callback.MIPSOL:
                    if time.time() - start_time >= args.time_limit:
                        cb_model.terminate()
                        return
                    candidate_objective = cb_model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    if candidate_objective < best_objective - 1e-7:
                        values = cb_model.cbGet(flat_variables)
                        payload = build_payload(candidate_objective, values)
                        best_objective = float(candidate_objective)
                        best_payload = payload
                        if logger:
                            logger.log_solution(best_objective, best_payload)

            model.optimize(callback)

            if model.SolCount > 0 and model.ObjVal < best_objective - 1e-7:
                values = [var.X for var in flat_variables]
                best_objective = float(model.ObjVal)
                best_payload = build_payload(best_objective, values)
                if logger:
                    logger.log_solution(best_objective, best_payload)

        except Exception as exc:
            print(f"Optimization could not complete: {exc}", file=sys.stderr)
            if model_ready and model is not None and model.SolCount > 0:
                try:
                    values = [var.X for var in flat_variables]
                    if model.ObjVal < best_objective - 1e-7:
                        best_objective = float(model.ObjVal)
                        best_payload = build_payload(best_objective, values)
                        if logger:
                            logger.log_solution(best_objective, best_payload)
                except Exception:
                    pass

    with open(args.solution_path, "w") as f:
        json.dump(best_payload, f, indent=2)


if __name__ == "__main__":
    main()
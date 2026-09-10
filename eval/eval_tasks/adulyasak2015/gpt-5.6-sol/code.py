import argparse
import json
import os
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


TOL = 1e-7


def clean_number(value):
    value = float(value)
    if abs(value) < TOL:
        return 0.0
    rounded = round(value)
    if abs(value - rounded) < 1e-6:
        return int(rounded)
    return round(value, 10)


def write_json(path, data):
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    temporary_path = path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"), allow_nan=False)
    os.replace(temporary_path, path)


def prepare_instance(data):
    n = int(data["n"])
    periods = int(data["T"])
    vehicles = int(data["m"])

    scenario_count = int(data.get("n_scenarios", 0))
    scenarios = data.get("demand_scenarios", [])
    probabilities = data.get("scenario_probabilities", [])

    if scenario_count <= 0 or not scenarios:
        scenario_count = 1
        scenarios = [[
            [float(data["d_bar"][i][t]) for t in range(periods)]
            for i in range(n)
        ]]
        probabilities = [1.0]

    probabilities = [float(p) for p in probabilities]
    if len(probabilities) != scenario_count:
        probabilities = [1.0 / scenario_count] * scenario_count

    probability_sum = sum(probabilities)
    if probability_sum <= 0:
        probabilities = [1.0 / scenario_count] * scenario_count
    elif abs(probability_sum - 1.0) > 1e-10:
        probabilities = [p / probability_sum for p in probabilities]

    demand = [
        [
            [float(scenarios[s][i][t]) for t in range(periods)]
            for i in range(n)
        ]
        for s in range(scenario_count)
    ]

    return n, periods, vehicles, scenario_count, probabilities, demand


def make_zero_route_fields(n, periods, vehicles):
    y = {str(t): 0 for t in range(periods)}
    z = {
        f"{i}_{k}_{t}": 0
        for t in range(periods)
        for k in range(vehicles)
        for i in range(n + 1)
    }
    x = {
        f"{i}_{j}_{k}_{t}": 0
        for t in range(periods)
        for k in range(vehicles)
        for i in range(n + 1)
        for j in range(i + 1, n + 1)
    }
    return y, z, x


def build_baseline_solution(data, n, periods, vehicles, scenario_count,
                            probabilities, demand):
    """
    Feasible no-production/no-delivery policy. Initial customer inventory is
    used to satisfy demand, and all residual demand is lost.
    """
    initial_inventory = [float(v) for v in data["I0"]]
    holding = [float(v) for v in data["h"]]
    penalties = [float(v) for v in data["sigma"]]

    y, z, x = make_zero_route_fields(n, periods, vehicles)
    model_variables = {}
    objective = 0.0

    for s in range(scenario_count):
        probability = probabilities[s]

        plant_inventory = initial_inventory[0]
        for t in range(periods):
            if plant_inventory > TOL:
                model_variables[f"inv[{s},0,{t}]"] = clean_number(
                    plant_inventory
                )
            objective += probability * holding[0] * plant_inventory

        for i in range(n):
            inventory = initial_inventory[i + 1]
            for t in range(periods):
                served = min(inventory, demand[s][i][t])
                shortage = max(0.0, demand[s][i][t] - served)
                inventory = max(0.0, inventory - served)

                if inventory > TOL:
                    model_variables[f"inv[{s},{i + 1},{t}]"] = clean_number(
                        inventory
                    )
                if shortage > TOL:
                    model_variables[f"short[{s},{i + 1},{t}]"] = clean_number(
                        shortage
                    )

                objective += probability * (
                    holding[i + 1] * inventory
                    + penalties[i] * shortage
                )

    return {
        "objective_value": clean_number(objective),
        "model_variables": model_variables,
        "y": y,
        "z": z,
        "x": x,
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

    n, periods, vehicles, scenario_count, probabilities, demand = \
        prepare_instance(data)

    baseline = build_baseline_solution(
        data, n, periods, vehicles, scenario_count, probabilities, demand
    )
    best_solution = baseline
    best_objective = float(baseline["objective_value"])

    if logger:
        logger.log_solution(best_objective, baseline)

    # If virtually no time remains, return the guaranteed baseline immediately.
    remaining = args.time_limit - (time.monotonic() - start_time)
    if remaining <= 0.05:
        write_json(args.solution_path, best_solution)
        return

    nodes = range(n + 1)
    customers = range(1, n + 1)
    period_set = range(periods)
    vehicle_set = range(vehicles)
    scenario_set = range(scenario_count)

    capacity = float(data["Q"])
    production_capacity = float(data["C"])
    fixed_cost = float(data["f"])
    production_cost = float(data["u"])
    travel = data["transportation_costs"]
    holding = [float(v) for v in data["h"]]
    storage = [float(v) for v in data["L"]]
    initial_inventory = [float(v) for v in data["I0"]]
    penalties = [float(v) for v in data["sigma"]]

    model = gp.Model("stochastic_production_inventory_routing")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = max(
        0.01, args.time_limit - (time.monotonic() - start_time)
    )

    # First-stage decisions.
    y = {
        t: model.addVar(vtype=GRB.BINARY, name=f"y[{t}]")
        for t in period_set
    }
    z = {
        (i, k, t): model.addVar(
            vtype=GRB.BINARY, name=f"z[{i},{k},{t}]"
        )
        for t in period_set
        for k in vehicle_set
        for i in nodes
    }
    x = {}
    for t in period_set:
        for k in vehicle_set:
            for i in nodes:
                for j in range(i + 1, n + 1):
                    x[i, j, k, t] = model.addVar(
                        vtype=GRB.INTEGER if i == 0 else GRB.BINARY,
                        lb=0.0,
                        ub=2.0 if i == 0 else 1.0,
                        name=f"x[{i},{j},{k},{t}]",
                    )

    # Scenario-dependent recourse decisions.
    production = {
        (s, t): model.addVar(
            lb=0.0, ub=production_capacity, name=f"p[{s},{t}]"
        )
        for s in scenario_set
        for t in period_set
    }
    delivery = {
        (s, i, k, t): model.addVar(
            lb=0.0, ub=capacity, name=f"q[{s},{i},{k},{t}]"
        )
        for s in scenario_set
        for i in customers
        for k in vehicle_set
        for t in period_set
    }
    inventory = {
        (s, i, t): model.addVar(
            lb=0.0, ub=storage[i], name=f"inv[{s},{i},{t}]"
        )
        for s in scenario_set
        for i in nodes
        for t in period_set
    }
    shortage = {
        (s, i, t): model.addVar(
            lb=0.0,
            ub=max(0.0, demand[s][i - 1][t]),
            name=f"short[{s},{i},{t}]",
        )
        for s in scenario_set
        for i in customers
        for t in period_set
    }

    # Directed single-commodity flow used only to eliminate disconnected tours.
    route_flow = {
        (i, j, k, t): model.addVar(
            lb=0.0, ub=float(n), name=f"_flow[{i},{j},{k},{t}]"
        )
        for t in period_set
        for k in vehicle_set
        for i in nodes
        for j in nodes
        if i != j
    }

    model.update()

    def edge_var(i, j, k, t):
        if i < j:
            return x[i, j, k, t]
        return x[j, i, k, t]

    # Route degree, dispatch, connectivity, and customer assignment.
    for t in period_set:
        for i in customers:
            model.addConstr(
                gp.quicksum(z[i, k, t] for k in vehicle_set) <= 1,
                name=f"one_vehicle[{i},{t}]",
            )

        for k in vehicle_set:
            model.addConstr(
                gp.quicksum(x[0, j, k, t] for j in customers)
                == 2 * z[0, k, t],
                name=f"depot_degree[{k},{t}]",
            )
            model.addConstr(
                z[0, k, t]
                <= gp.quicksum(z[i, k, t] for i in customers),
                name=f"nonempty_route[{k},{t}]",
            )

            for i in customers:
                model.addConstr(
                    gp.quicksum(
                        edge_var(i, j, k, t) for j in nodes if j != i
                    ) == 2 * z[i, k, t],
                    name=f"customer_degree[{i},{k},{t}]",
                )
                model.addConstr(
                    z[i, k, t] <= z[0, k, t],
                    name=f"visit_requires_dispatch[{i},{k},{t}]",
                )

                model.addConstr(
                    gp.quicksum(
                        route_flow[j, i, k, t] for j in nodes if j != i
                    )
                    - gp.quicksum(
                        route_flow[i, j, k, t] for j in nodes if j != i
                    )
                    == z[i, k, t],
                    name=f"flow_balance[{i},{k},{t}]",
                )

            model.addConstr(
                gp.quicksum(
                    route_flow[0, j, k, t] for j in customers
                )
                - gp.quicksum(
                    route_flow[j, 0, k, t] for j in customers
                )
                == gp.quicksum(z[i, k, t] for i in customers),
                name=f"depot_flow[{k},{t}]",
            )

            for i in nodes:
                for j in nodes:
                    if i != j:
                        model.addConstr(
                            route_flow[i, j, k, t]
                            <= n * edge_var(i, j, k, t),
                            name=f"flow_link[{i},{j},{k},{t}]",
                        )

        # Vehicle-label symmetry breaking within each period.
        for k in range(vehicles - 1):
            model.addConstr(
                z[0, k, t] >= z[0, k + 1, t],
                name=f"vehicle_order[{k},{t}]",
            )

    # Production, vehicle loading, and inventory balances.
    for s in scenario_set:
        for t in period_set:
            model.addConstr(
                production[s, t] <= production_capacity * y[t],
                name=f"production_setup[{s},{t}]",
            )

            previous_plant = (
                initial_inventory[0]
                if t == 0
                else inventory[s, 0, t - 1]
            )
            total_shipped = gp.quicksum(
                delivery[s, i, k, t]
                for i in customers
                for k in vehicle_set
            )
            model.addConstr(
                previous_plant + production[s, t]
                == total_shipped + inventory[s, 0, t],
                name=f"plant_balance[{s},{t}]",
            )

            for k in vehicle_set:
                model.addConstr(
                    gp.quicksum(
                        delivery[s, i, k, t] for i in customers
                    ) <= capacity,
                    name=f"vehicle_capacity[{s},{k},{t}]",
                )

            for i in customers:
                previous_customer = (
                    initial_inventory[i]
                    if t == 0
                    else inventory[s, i, t - 1]
                )
                delivered = gp.quicksum(
                    delivery[s, i, k, t] for k in vehicle_set
                )
                realized_demand = demand[s][i - 1][t]

                model.addConstr(
                    previous_customer + delivered + shortage[s, i, t]
                    == realized_demand + inventory[s, i, t],
                    name=f"customer_balance[{s},{i},{t}]",
                )
                model.addConstr(
                    inventory[s, i, t] + realized_demand <= storage[i],
                    name=f"customer_post_delivery_storage[{s},{i},{t}]",
                )
                model.addConstr(
                    previous_customer + delivered <= storage[i],
                    name=f"customer_receipt_storage[{s},{i},{t}]",
                )

                future_demand = sum(
                    demand[s][i - 1][tau] for tau in range(t, periods)
                )
                delivery_bound = min(capacity, storage[i], future_demand)
                for k in vehicle_set:
                    model.addConstr(
                        delivery[s, i, k, t]
                        <= delivery_bound * z[i, k, t],
                        name=f"delivery_visit[{s},{i},{k},{t}]",
                    )

    # Nonanticipativity of recourse decisions for scenarios with identical
    # observed demand histories through the current period.
    for t in period_set:
        history_groups = {}
        for s in scenario_set:
            history = tuple(
                demand[s][i][tau]
                for tau in range(t + 1)
                for i in range(n)
            )
            history_groups.setdefault(history, []).append(s)

        for group in history_groups.values():
            representative = group[0]
            for s in group[1:]:
                model.addConstr(
                    production[s, t] == production[representative, t],
                    name=f"na_production[{s},{representative},{t}]",
                )
                for i in customers:
                    for k in vehicle_set:
                        model.addConstr(
                            delivery[s, i, k, t]
                            == delivery[representative, i, k, t],
                            name=(
                                f"na_delivery[{s},{representative},"
                                f"{i},{k},{t}]"
                            ),
                        )

    first_stage_cost = (
        gp.quicksum(fixed_cost * y[t] for t in period_set)
        + gp.quicksum(
            float(travel[i][j]) * x[i, j, k, t]
            for (i, j, k, t) in x
        )
    )

    expected_recourse_cost = gp.quicksum(
        probabilities[s]
        * (
            gp.quicksum(
                production_cost * production[s, t]
                for t in period_set
            )
            + gp.quicksum(
                holding[i] * inventory[s, i, t]
                for i in nodes
                for t in period_set
            )
            + gp.quicksum(
                penalties[i - 1] * shortage[s, i, t]
                for i in customers
                for t in period_set
            )
        )
        for s in scenario_set
    )

    model.setObjective(first_stage_cost + expected_recourse_cost, GRB.MINIMIZE)

    # Install the baseline as a MIP start.
    for var in y.values():
        var.Start = 0.0
    for var in z.values():
        var.Start = 0.0
    for var in x.values():
        var.Start = 0.0
    for var in production.values():
        var.Start = 0.0
    for var in delivery.values():
        var.Start = 0.0
    for var in route_flow.values():
        var.Start = 0.0

    for s in scenario_set:
        for t in period_set:
            inventory[s, 0, t].Start = initial_inventory[0]

        for i in customers:
            current_inventory = initial_inventory[i]
            for t in period_set:
                served = min(current_inventory, demand[s][i - 1][t])
                unmet = max(0.0, demand[s][i - 1][t] - served)
                current_inventory = max(0.0, current_inventory - served)
                inventory[s, i, t].Start = current_inventory
                shortage[s, i, t].Start = unmet

    named_variables = (
        list(y.values())
        + list(z.values())
        + list(x.values())
        + list(production.values())
        + list(delivery.values())
        + list(inventory.values())
        + list(shortage.values())
    )

    all_solution_variables = named_variables

    def construct_solution(value_map, objective_value):
        def value(var):
            return float(value_map[var])

        output_y = {
            str(t): int(round(value(y[t])))
            for t in period_set
        }
        output_z = {
            f"{i}_{k}_{t}": int(round(value(z[i, k, t])))
            for t in period_set
            for k in vehicle_set
            for i in nodes
        }
        output_x = {
            f"{i}_{j}_{k}_{t}": int(round(value(x[i, j, k, t])))
            for t in period_set
            for k in vehicle_set
            for i in nodes
            for j in range(i + 1, n + 1)
        }

        sparse_variables = {}
        for var in named_variables:
            var_value = value(var)
            if abs(var_value) > TOL:
                sparse_variables[var.VarName] = clean_number(var_value)

        return {
            "objective_value": clean_number(objective_value),
            "model_variables": sparse_variables,
            "y": output_y,
            "z": output_z,
            "x": output_x,
        }

    callback_best = [best_objective]

    def incumbent_callback(cb_model, where):
        nonlocal best_solution, best_objective
        if where != GRB.Callback.MIPSOL:
            return
        try:
            incumbent_objective = float(
                cb_model.cbGet(GRB.Callback.MIPSOL_OBJ)
            )
            if incumbent_objective >= callback_best[0] - 1e-7:
                return

            values = cb_model.cbGetSolution(all_solution_variables)
            value_map = dict(zip(all_solution_variables, values))
            solution = construct_solution(value_map, incumbent_objective)

            callback_best[0] = incumbent_objective
            best_objective = incumbent_objective
            best_solution = solution

            if logger:
                logger.log_solution(incumbent_objective, solution)
        except Exception:
            # Logging or snapshot construction must not interrupt optimization.
            pass

    try:
        model.optimize(incumbent_callback)

        if model.SolCount > 0:
            objective_value = float(model.ObjVal)
            if objective_value < best_objective - 1e-7:
                value_map = {
                    var: float(var.X) for var in all_solution_variables
                }
                final_solution = construct_solution(
                    value_map, objective_value
                )
                best_solution = final_solution
                best_objective = objective_value
                if logger:
                    logger.log_solution(objective_value, final_solution)
            elif abs(objective_value - best_objective) <= 1e-6:
                # Prefer the final solver values, which have full numerical
                # precision and satisfy the final incumbent tolerances.
                value_map = {
                    var: float(var.X) for var in all_solution_variables
                }
                best_solution = construct_solution(
                    value_map, objective_value
                )
    except gp.GurobiError:
        # The precomputed baseline remains available if optimization fails.
        pass

    write_json(args.solution_path, best_solution)


if __name__ == "__main__":
    main()
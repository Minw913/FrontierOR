import argparse
import json
import os
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Solve the stochastic capacitated facility location problem."
    )
    parser.add_argument("--instance_path", required=True, type=str)
    parser.add_argument("--solution_path", required=True, type=str)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None, type=str)
    return parser.parse_args()


def build_solution(
    x_values,
    y_values,
    facility_ids,
    customer_ids,
    scenario_ids,
    opening_costs,
    transportation_costs,
    probabilities,
):
    num_facilities = len(facility_ids)
    num_customers = len(customer_ids)
    num_scenarios = len(scenario_ids)

    rounded_x = [1 if x_values[i] >= 0.5 else 0 for i in range(num_facilities)]

    x_output = {
        str(facility_ids[i]): rounded_x[i]
        for i in range(num_facilities)
    }
    open_facilities = [
        facility_ids[i] for i in range(num_facilities) if rounded_x[i] == 1
    ]

    y_output = {}
    transportation_objective = 0.0
    k = 0

    for s in range(num_scenarios):
        scenario_key = str(scenario_ids[s])
        y_output[scenario_key] = {}

        for i in range(num_facilities):
            facility_key = str(facility_ids[i])
            customer_shipments = {}

            for j in range(num_customers):
                value = float(y_values[k])
                k += 1

                # Remove harmless negative numerical noise.
                if value < 0.0 and value > -1e-7:
                    value = 0.0
                elif abs(value) < 1e-12:
                    value = 0.0

                customer_shipments[str(customer_ids[j])] = value
                transportation_objective += (
                    probabilities[s] * transportation_costs[i][j] * value
                )

            y_output[scenario_key][facility_key] = customer_shipments

    fixed_objective = sum(
        opening_costs[i] * rounded_x[i] for i in range(num_facilities)
    )
    objective = float(fixed_objective + transportation_objective)

    return {
        "objective_value": objective,
        "open_facilities": open_facilities,
        "x": x_output,
        "y": y_output,
    }


def create_all_open_solution(
    capacities,
    demands,
    transportation_costs,
):
    """
    Construct a guaranteed feasible solution when total available capacity is
    sufficient in every scenario. All facilities are opened, and shipments are
    greedily assigned in increasing transportation-cost order.
    """
    num_scenarios = len(demands)
    num_facilities = len(capacities)
    num_customers = len(demands[0]) if num_scenarios else 0

    edge_order = sorted(
        (
            (transportation_costs[i][j], i, j)
            for i in range(num_facilities)
            for j in range(num_customers)
        ),
        key=lambda item: item[0],
    )

    y_values = [0.0] * (
        num_scenarios * num_facilities * num_customers
    )

    for s in range(num_scenarios):
        remaining_demand = [max(0.0, float(v)) for v in demands[s]]
        remaining_capacity = [max(0.0, float(v)) for v in capacities]

        for _, i, j in edge_order:
            if remaining_capacity[i] <= 1e-12 or remaining_demand[j] <= 1e-12:
                continue

            quantity = min(remaining_capacity[i], remaining_demand[j])
            index = (s * num_facilities + i) * num_customers + j
            y_values[index] = quantity

            remaining_capacity[i] -= quantity
            remaining_demand[j] -= quantity

        if sum(remaining_demand) > 1e-6:
            return None

    return [1.0] * num_facilities, y_values


def main():
    args = parse_args()
    start_time = time.monotonic()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as f:
        instance = json.load(f)

    num_facilities = int(instance["num_facilities"])
    num_customers = int(instance["num_customers"])
    num_scenarios = int(instance["num_scenarios"])

    facilities = instance["facilities"]
    customers = instance["customers"]
    scenarios = instance["scenarios"]

    facility_ids = [int(facilities[i]["id"]) for i in range(num_facilities)]
    customer_ids = [int(customers[j]["id"]) for j in range(num_customers)]
    scenario_ids = [int(scenarios[s]["id"]) for s in range(num_scenarios)]

    opening_costs = [
        float(facilities[i]["opening_cost"]) for i in range(num_facilities)
    ]
    capacities = [
        float(facilities[i]["capacity"]) for i in range(num_facilities)
    ]
    transportation_costs = [
        [float(instance["transportation_costs"][i][j])
         for j in range(num_customers)]
        for i in range(num_facilities)
    ]
    probabilities = [
        float(scenarios[s]["probability"]) for s in range(num_scenarios)
    ]
    demands = [
        [float(scenarios[s]["demands"][j]) for j in range(num_customers)]
        for s in range(num_scenarios)
    ]

    maximum_total_demand = max(
        (sum(demands[s]) for s in range(num_scenarios)),
        default=0.0,
    )

    # Produce an immediate feasible incumbent and use it as a complete MIP start.
    warm_start = create_all_open_solution(
        capacities,
        demands,
        transportation_costs,
    )

    best_logged_objective = [float("inf")]
    fallback_solution = None

    if warm_start is not None:
        warm_x, warm_y = warm_start
        fallback_solution = build_solution(
            warm_x,
            warm_y,
            facility_ids,
            customer_ids,
            scenario_ids,
            opening_costs,
            transportation_costs,
            probabilities,
        )
        best_logged_objective[0] = fallback_solution["objective_value"]
        if logger:
            logger.log_solution(
                fallback_solution["objective_value"],
                fallback_solution,
            )

    model = gp.Model("stochastic_capacitated_facility_location")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1

    x = model.addVars(
        num_facilities,
        vtype=GRB.BINARY,
        name="x",
    )
    y = model.addVars(
        num_scenarios,
        num_facilities,
        num_customers,
        lb=0.0,
        vtype=GRB.CONTINUOUS,
        name="y",
    )

    fixed_cost_expression = gp.quicksum(
        opening_costs[i] * x[i] for i in range(num_facilities)
    )
    expected_transport_expression = gp.quicksum(
        probabilities[s]
        * transportation_costs[i][j]
        * y[s, i, j]
        for s in range(num_scenarios)
        for i in range(num_facilities)
        for j in range(num_customers)
    )

    model.setObjective(
        fixed_cost_expression + expected_transport_expression,
        GRB.MINIMIZE,
    )

    model.addConstrs(
        (
            gp.quicksum(y[s, i, j] for i in range(num_facilities))
            >= demands[s][j]
            for s in range(num_scenarios)
            for j in range(num_customers)
        ),
        name="demand",
    )

    model.addConstrs(
        (
            gp.quicksum(y[s, i, j] for j in range(num_customers))
            <= capacities[i] * x[i]
            for s in range(num_scenarios)
            for i in range(num_facilities)
        ),
        name="scenario_capacity",
    )

    model.addConstr(
        gp.quicksum(capacities[i] * x[i] for i in range(num_facilities))
        >= maximum_total_demand,
        name="maximum_demand_capacity",
    )

    model.update()

    x_vars = [x[i] for i in range(num_facilities)]
    y_vars = [
        y[s, i, j]
        for s in range(num_scenarios)
        for i in range(num_facilities)
        for j in range(num_customers)
    ]

    if warm_start is not None:
        warm_x, warm_y = warm_start
        for i, var in enumerate(x_vars):
            var.Start = warm_x[i]
        for k, var in enumerate(y_vars):
            var.Start = warm_y[k]

    elapsed = time.monotonic() - start_time
    remaining_time = max(0.001, float(args.time_limit) - elapsed)
    model.Params.TimeLimit = remaining_time

    def incumbent_callback(cb_model, where):
        if where != GRB.Callback.MIPSOL:
            return

        incumbent_objective = float(
            cb_model.cbGet(GRB.Callback.MIPSOL_OBJ)
        )
        tolerance = 1e-8 * max(1.0, abs(best_logged_objective[0]))

        if incumbent_objective >= best_logged_objective[0] - tolerance:
            return

        incumbent_x = cb_model.cbGetSolution(x_vars)
        incumbent_y = cb_model.cbGetSolution(y_vars)

        incumbent_solution = build_solution(
            incumbent_x,
            incumbent_y,
            facility_ids,
            customer_ids,
            scenario_ids,
            opening_costs,
            transportation_costs,
            probabilities,
        )

        best_logged_objective[0] = incumbent_solution["objective_value"]

        if logger:
            logger.log_solution(
                incumbent_solution["objective_value"],
                incumbent_solution,
            )

    model.optimize(incumbent_callback)

    if model.SolCount > 0:
        final_x = [var.X for var in x_vars]
        final_y = [var.X for var in y_vars]

        final_solution = build_solution(
            final_x,
            final_y,
            facility_ids,
            customer_ids,
            scenario_ids,
            opening_costs,
            transportation_costs,
            probabilities,
        )
    elif fallback_solution is not None:
        final_solution = fallback_solution
    else:
        raise RuntimeError(
            "No feasible solution was found. The instance may be infeasible "
            "because total facility capacity is insufficient."
        )

    final_tolerance = 1e-8 * max(1.0, abs(best_logged_objective[0]))
    if (
        logger
        and final_solution["objective_value"]
        < best_logged_objective[0] - final_tolerance
    ):
        logger.log_solution(
            final_solution["objective_value"],
            final_solution,
        )
        best_logged_objective[0] = final_solution["objective_value"]

    solution_directory = os.path.dirname(os.path.abspath(args.solution_path))
    os.makedirs(solution_directory, exist_ok=True)

    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(final_solution, f, separators=(",", ":"), allow_nan=False)


if __name__ == "__main__":
    main()
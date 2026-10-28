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


def construct_shipments(
    open_vector,
    capacities,
    demands,
    transportation_costs,
):
    """Construct feasible shipments using cheapest open arcs first."""
    num_scenarios = len(demands)
    num_facilities = len(capacities)
    num_customers = len(demands[0]) if num_scenarios else 0

    open_indices = [
        i for i in range(num_facilities) if open_vector[i] >= 0.5
    ]
    total_capacity = sum(capacities[i] for i in open_indices)
    required_capacity = max(
        (sum(scenario_demand) for scenario_demand in demands),
        default=0.0,
    )

    if total_capacity + 1e-7 < required_capacity:
        return None

    edge_order = sorted(
        (
            (transportation_costs[i][j], i, j)
            for i in open_indices
            for j in range(num_customers)
        ),
        key=lambda item: item[0],
    )

    y_values = [0.0] * (
        num_scenarios * num_facilities * num_customers
    )

    for s in range(num_scenarios):
        remaining_capacity = [
            capacities[i] if open_vector[i] >= 0.5 else 0.0
            for i in range(num_facilities)
        ]
        remaining_demand = [
            max(0.0, float(demands[s][j]))
            for j in range(num_customers)
        ]

        for _, i, j in edge_order:
            if remaining_capacity[i] <= 1e-12:
                continue
            if remaining_demand[j] <= 1e-12:
                continue

            quantity = min(
                remaining_capacity[i],
                remaining_demand[j],
            )
            index = (
                (s * num_facilities + i) * num_customers + j
            )
            y_values[index] = quantity
            remaining_capacity[i] -= quantity
            remaining_demand[j] -= quantity

        if any(value > 1e-6 for value in remaining_demand):
            return None

    return y_values


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

    rounded_x = [
        1 if float(x_values[i]) >= 0.5 else 0
        for i in range(num_facilities)
    ]

    x_output = {
        str(facility_ids[i]): rounded_x[i]
        for i in range(num_facilities)
    }
    open_facilities = [
        facility_ids[i]
        for i in range(num_facilities)
        if rounded_x[i] == 1
    ]

    y_output = {}
    transportation_objective = 0.0
    position = 0

    for s in range(num_scenarios):
        scenario_key = str(scenario_ids[s])
        y_output[scenario_key] = {}

        for i in range(num_facilities):
            facility_key = str(facility_ids[i])
            y_output[scenario_key][facility_key] = {}

            for j in range(num_customers):
                value = float(y_values[position])
                position += 1

                if abs(value) < 1e-10:
                    value = 0.0
                elif value < 0.0 and value > -1e-6:
                    value = 0.0

                y_output[scenario_key][facility_key][
                    str(customer_ids[j])
                ] = value

                transportation_objective += (
                    probabilities[s]
                    * transportation_costs[i][j]
                    * value
                )

    fixed_objective = sum(
        opening_costs[i] * rounded_x[i]
        for i in range(num_facilities)
    )

    objective_value = float(
        fixed_objective + transportation_objective
    )

    return {
        "objective_value": objective_value,
        "open_facilities": open_facilities,
        "x": x_output,
        "y": y_output,
    }


def generate_fallback_solutions(
    capacities,
    opening_costs,
    demands,
    transportation_costs,
    probabilities,
    facility_ids,
    customer_ids,
    scenario_ids,
    logger,
    deadline,
):
    """
    Generate several feasible facility sets without relying on a solver.
    This also ensures a valid result when Gurobi cannot acquire a license.
    """
    num_facilities = len(capacities)
    num_customers = len(customer_ids)
    maximum_demand = max(
        (sum(row) for row in demands),
        default=0.0,
    )

    if sum(capacities) + 1e-7 < maximum_demand:
        return None, None, None

    expected_demands = [
        sum(
            probabilities[s] * demands[s][j]
            for s in range(len(demands))
        )
        for j in range(num_customers)
    ]
    total_expected_demand = sum(expected_demands)

    average_shipping_cost = []
    for i in range(num_facilities):
        if total_expected_demand > 1e-12:
            value = sum(
                expected_demands[j] * transportation_costs[i][j]
                for j in range(num_customers)
            ) / total_expected_demand
        elif num_customers:
            value = sum(transportation_costs[i]) / num_customers
        else:
            value = 0.0
        average_shipping_cost.append(value)

    rankings = [
        list(range(num_facilities)),
        sorted(
            range(num_facilities),
            key=lambda i: (
                opening_costs[i] / max(capacities[i], 1e-12),
                opening_costs[i],
            ),
        ),
        sorted(
            range(num_facilities),
            key=lambda i: (
                opening_costs[i] / max(capacities[i], 1e-12)
                + average_shipping_cost[i],
                opening_costs[i],
            ),
        ),
        sorted(
            range(num_facilities),
            key=lambda i: (
                average_shipping_cost[i],
                opening_costs[i] / max(capacities[i], 1e-12),
            ),
        ),
        sorted(
            range(num_facilities),
            key=lambda i: (-capacities[i], opening_costs[i]),
        ),
    ]

    candidate_vectors = []
    seen = set()

    all_open = tuple(1 for _ in range(num_facilities))
    candidate_vectors.append(list(all_open))
    seen.add(all_open)

    for ranking in rankings:
        selected = [0] * num_facilities
        selected_capacity = 0.0

        for i in ranking:
            if selected_capacity + 1e-7 >= maximum_demand:
                break
            selected[i] = 1
            selected_capacity += capacities[i]

        key = tuple(selected)
        if selected_capacity + 1e-7 >= maximum_demand and key not in seen:
            candidate_vectors.append(selected)
            seen.add(key)

    best_solution = None
    best_x = None
    best_y = None

    def evaluate(open_vector):
        nonlocal best_solution, best_x, best_y

        shipments = construct_shipments(
            open_vector,
            capacities,
            demands,
            transportation_costs,
        )
        if shipments is None:
            return None

        solution = build_solution(
            open_vector,
            shipments,
            facility_ids,
            customer_ids,
            scenario_ids,
            opening_costs,
            transportation_costs,
            probabilities,
        )

        if (
            best_solution is None
            or solution["objective_value"]
            < best_solution["objective_value"] - 1e-8
        ):
            best_solution = solution
            best_x = list(open_vector)
            best_y = shipments

            if logger:
                logger.log_solution(
                    solution["objective_value"],
                    solution,
                )

        return solution

    for candidate in candidate_vectors:
        evaluate(candidate)
        if time.monotonic() >= deadline:
            break

    # Limited facility-removal improvement around the best heuristic set.
    if best_x is not None and time.monotonic() < deadline:
        current_x = list(best_x)
        current_objective = best_solution["objective_value"]

        for _ in range(2):
            improved_x = None
            improved_objective = current_objective

            open_indices = [
                i for i in range(num_facilities) if current_x[i] == 1
            ]
            open_indices.sort(
                key=lambda i: opening_costs[i],
                reverse=True,
            )

            for i in open_indices:
                if time.monotonic() >= deadline:
                    break

                candidate = list(current_x)
                candidate[i] = 0

                if (
                    sum(
                        capacities[k]
                        for k in range(num_facilities)
                        if candidate[k] == 1
                    )
                    + 1e-7
                    < maximum_demand
                ):
                    continue

                solution = evaluate(candidate)
                if (
                    solution is not None
                    and solution["objective_value"]
                    < improved_objective - 1e-8
                ):
                    improved_objective = solution["objective_value"]
                    improved_x = candidate

            if improved_x is None:
                break

            current_x = improved_x
            current_objective = improved_objective

    return best_solution, best_x, best_y


def write_solution(path, solution):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)

    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(
            solution,
            output_file,
            separators=(",", ":"),
            allow_nan=False,
        )


def main():
    args = parse_args()
    start_time = time.monotonic()
    deadline = start_time + max(0, args.time_limit)

    logger = (
        SolutionLogger(args.log_path, sense="minimize")
        if args.log_path
        else None
    )

    with open(args.instance_path, "r", encoding="utf-8") as input_file:
        instance = json.load(input_file)

    facilities = instance["facilities"]
    customers = instance["customers"]
    scenarios = instance["scenarios"]

    num_facilities = int(instance["num_facilities"])
    num_customers = int(instance["num_customers"])
    num_scenarios = int(instance["num_scenarios"])

    facility_ids = [
        int(facilities[i]["id"]) for i in range(num_facilities)
    ]
    customer_ids = [
        int(customers[j]["id"]) for j in range(num_customers)
    ]
    scenario_ids = [
        int(scenarios[s]["id"]) for s in range(num_scenarios)
    ]

    opening_costs = [
        float(facilities[i]["opening_cost"])
        for i in range(num_facilities)
    ]
    capacities = [
        float(facilities[i]["capacity"])
        for i in range(num_facilities)
    ]
    transportation_costs = [
        [
            float(instance["transportation_costs"][i][j])
            for j in range(num_customers)
        ]
        for i in range(num_facilities)
    ]
    probabilities = [
        float(scenarios[s]["probability"])
        for s in range(num_scenarios)
    ]
    demands = [
        [
            float(scenarios[s]["demands"][j])
            for j in range(num_customers)
        ]
        for s in range(num_scenarios)
    ]

    maximum_total_demand = max(
        (sum(demands[s]) for s in range(num_scenarios)),
        default=0.0,
    )

    # Reserve most of the available time for Gurobi, while creating a valid
    # fallback first. For very short limits, fallback construction is still
    # attempted because a solution must be returned.
    heuristic_deadline = min(
        deadline,
        start_time + max(0.05, min(2.0, args.time_limit * 0.10)),
    )

    fallback_solution, fallback_x, fallback_y = generate_fallback_solutions(
        capacities,
        opening_costs,
        demands,
        transportation_costs,
        probabilities,
        facility_ids,
        customer_ids,
        scenario_ids,
        logger,
        heuristic_deadline,
    )

    if fallback_solution is None:
        raise RuntimeError(
            "The instance is infeasible: total facility capacity is less "
            "than the maximum total demand."
        )

    best_solution = fallback_solution
    best_logged_objective = [fallback_solution["objective_value"]]

    # Model construction can itself fail if no Gurobi license is available.
    # In that case, retain and write the feasible fallback solution.
    try:
        remaining_time = deadline - time.monotonic()
        if remaining_time <= 0.0:
            write_solution(args.solution_path, best_solution)
            return

        model = gp.Model(
            "stochastic_capacitated_facility_location"
        )
        model.Params.OutputFlag = 0
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1
        model.Params.TimeLimit = max(0.001, remaining_time)

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

        model.setObjective(
            gp.quicksum(
                opening_costs[i] * x[i]
                for i in range(num_facilities)
            )
            + gp.quicksum(
                probabilities[s]
                * transportation_costs[i][j]
                * y[s, i, j]
                for s in range(num_scenarios)
                for i in range(num_facilities)
                for j in range(num_customers)
            ),
            GRB.MINIMIZE,
        )

        model.addConstrs(
            (
                gp.quicksum(
                    y[s, i, j] for i in range(num_facilities)
                )
                >= demands[s][j]
                for s in range(num_scenarios)
                for j in range(num_customers)
            ),
            name="demand",
        )

        model.addConstrs(
            (
                gp.quicksum(
                    y[s, i, j] for j in range(num_customers)
                )
                <= capacities[i] * x[i]
                for s in range(num_scenarios)
                for i in range(num_facilities)
            ),
            name="capacity",
        )

        model.addConstr(
            gp.quicksum(
                capacities[i] * x[i]
                for i in range(num_facilities)
            )
            >= maximum_total_demand,
            name="maximum_total_demand_capacity",
        )

        model.update()

        x_vars = [x[i] for i in range(num_facilities)]
        y_vars = [
            y[s, i, j]
            for s in range(num_scenarios)
            for i in range(num_facilities)
            for j in range(num_customers)
        ]

        for i, variable in enumerate(x_vars):
            variable.Start = fallback_x[i]

        for index, variable in enumerate(y_vars):
            variable.Start = fallback_y[index]

        def incumbent_callback(callback_model, where):
            if where != GRB.Callback.MIPSOL:
                return

            callback_objective = float(
                callback_model.cbGet(GRB.Callback.MIPSOL_OBJ)
            )
            tolerance = 1e-8 * max(
                1.0, abs(best_logged_objective[0])
            )

            if (
                callback_objective
                >= best_logged_objective[0] - tolerance
            ):
                return

            incumbent_x = callback_model.cbGetSolution(x_vars)
            incumbent_y = callback_model.cbGetSolution(y_vars)

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

            if (
                incumbent_solution["objective_value"]
                < best_logged_objective[0] - tolerance
            ):
                best_logged_objective[0] = incumbent_solution[
                    "objective_value"
                ]
                if logger:
                    logger.log_solution(
                        incumbent_solution["objective_value"],
                        incumbent_solution,
                    )

        model.optimize(incumbent_callback)

        if model.SolCount > 0:
            solved_solution = build_solution(
                [variable.X for variable in x_vars],
                [variable.X for variable in y_vars],
                facility_ids,
                customer_ids,
                scenario_ids,
                opening_costs,
                transportation_costs,
                probabilities,
            )

            if (
                solved_solution["objective_value"]
                < best_solution["objective_value"] - 1e-7
            ):
                best_solution = solved_solution

                tolerance = 1e-8 * max(
                    1.0, abs(best_logged_objective[0])
                )
                if (
                    logger
                    and best_solution["objective_value"]
                    < best_logged_objective[0] - tolerance
                ):
                    logger.log_solution(
                        best_solution["objective_value"],
                        best_solution,
                    )
                    best_logged_objective[0] = best_solution[
                        "objective_value"
                    ]

    except (gp.GurobiError, RuntimeError, AttributeError):
        # Typical cause: Gurobi cannot obtain a license in the execution
        # environment. The previously constructed fallback remains feasible.
        pass

    write_solution(args.solution_path, best_solution)


if __name__ == "__main__":
    main()
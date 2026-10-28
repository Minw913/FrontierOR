import argparse
import json
import math
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def make_solution(opened, allocation, demands, fixed_costs, shipping_costs):
    """Convert an allocation matrix into the required output schema."""
    opened = sorted(int(j) for j in opened)
    opened_set = set(opened)

    assignments = {}
    shipping_total = 0.0

    for i, row in enumerate(allocation):
        customer_assignments = {}
        for j in opened:
            value = float(row[j])
            if value > 0.0:
                customer_assignments[str(j)] = value
                shipping_total += shipping_costs[i][j] * value
        assignments[str(i)] = customer_assignments

    fixed_total = sum(fixed_costs[j] for j in opened_set)
    objective = float(fixed_total + shipping_total)

    return {
        "objective_value": objective,
        "open_plants": opened,
        "assignments": assignments,
    }


def greedy_transport(opened, demands, capacities, shipping_costs, customer_order):
    """
    Construct a feasible splittable transportation plan for a given set of
    open plants. Customers are processed in the supplied order and use the
    cheapest plants with remaining capacity.
    """
    n = len(demands)
    m = len(capacities)
    remaining_capacity = [0.0] * m
    for j in opened:
        remaining_capacity[j] = float(capacities[j])

    allocation = [[0.0] * m for _ in range(n)]
    tolerance = 1e-9

    for i in customer_order:
        demand = float(demands[i])
        remaining = demand
        plant_order = sorted(opened, key=lambda j: shipping_costs[i][j])

        for j in plant_order:
            if remaining <= tolerance:
                break
            amount = min(remaining, max(0.0, remaining_capacity[j]))
            if amount > 0.0:
                allocation[i][j] += amount / demand
                remaining_capacity[j] -= amount
                remaining -= amount

        if remaining > max(tolerance, 1e-8 * max(1.0, demand)):
            return None

        # Correct harmless floating-point residue so each customer's fractions
        # sum exactly to one in the serialized solution.
        row_sum = sum(allocation[i][j] for j in opened)
        if row_sum > 0.0 and abs(row_sum - 1.0) <= 1e-8:
            largest = max(opened, key=lambda j: allocation[i][j])
            allocation[i][largest] += 1.0 - row_sum

    return allocation


def construct_initial_solutions(
    demands,
    capacities,
    fixed_costs,
    shipping_costs,
    logger,
    best_holder,
):
    """Generate several inexpensive feasible MIP starts."""
    n = len(demands)
    m = len(capacities)
    total_demand = float(sum(demands))

    all_plants = list(range(m))
    candidate_sets = [all_plants]

    sum_shipping = [
        sum(shipping_costs[i][j] for i in range(n)) for j in range(m)
    ]
    weighted_shipping = [
        sum(shipping_costs[i][j] * demands[i] for i in range(n))
        / max(1.0, total_demand)
        for j in range(m)
    ]

    orders = [
        sorted(
            range(m),
            key=lambda j: fixed_costs[j] / max(float(capacities[j]), 1e-12),
        ),
        sorted(
            range(m),
            key=lambda j: (
                fixed_costs[j] + sum_shipping[j] / max(1, n)
            ) / max(float(capacities[j]), 1e-12),
        ),
        sorted(
            range(m),
            key=lambda j: (
                fixed_costs[j] + weighted_shipping[j]
            ) / max(float(capacities[j]), 1e-12),
        ),
        sorted(range(m), key=lambda j: fixed_costs[j] + weighted_shipping[j]),
    ]

    for order in orders:
        selected = []
        selected_capacity = 0.0
        for j in order:
            selected.append(j)
            selected_capacity += capacities[j]
            if selected_capacity >= total_demand:
                break
        if selected_capacity >= total_demand:
            candidate_sets.append(sorted(selected))

    for j in range(m):
        if capacities[j] >= total_demand:
            candidate_sets.append([j])

    unique_sets = []
    seen = set()
    for plants in candidate_sets:
        key = tuple(sorted(plants))
        if key not in seen:
            seen.add(key)
            unique_sets.append(list(key))

    natural_order = list(range(n))

    for opened in unique_sets:
        if sum(capacities[j] for j in opened) + 1e-9 < total_demand:
            continue

        customer_orders = [natural_order]

        if len(opened) >= 2:
            regrets = []
            for i in range(n):
                costs = sorted(shipping_costs[i][j] for j in opened)
                regrets.append(costs[1] - costs[0])
            customer_orders.append(
                sorted(
                    range(n),
                    key=lambda i: (regrets[i], demands[i]),
                    reverse=True,
                )
            )
        else:
            customer_orders.append(
                sorted(range(n), key=lambda i: demands[i], reverse=True)
            )

        for customer_order in customer_orders:
            allocation = greedy_transport(
                opened,
                demands,
                capacities,
                shipping_costs,
                customer_order,
            )
            if allocation is None:
                continue

            solution = make_solution(
                opened,
                allocation,
                demands,
                fixed_costs,
                shipping_costs,
            )
            objective = solution["objective_value"]

            if objective < best_holder["objective"] - 1e-9:
                best_holder["objective"] = objective
                best_holder["solution"] = solution
                best_holder["allocation"] = allocation
                best_holder["opened"] = opened

                if logger:
                    logger.log_solution(objective, solution)


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
        instance = json.load(f)

    n = int(instance["num_customers"])
    m = int(instance["num_plants"])
    demands = [float(v) for v in instance["customers"]["demands"]]
    capacities = [float(v) for v in instance["plants"]["capacities"]]
    fixed_costs = [float(v) for v in instance["plants"]["fixed_costs"]]
    shipping_costs = [
        [float(v) for v in row] for row in instance["shipping_costs"]
    ]

    best_holder = {
        "objective": math.inf,
        "solution": None,
        "allocation": None,
        "opened": None,
    }

    # Produce a guaranteed feasible fallback and a useful MIP start.
    construct_initial_solutions(
        demands,
        capacities,
        fixed_costs,
        shipping_costs,
        logger,
        best_holder,
    )

    try:
        model = gp.Model("capacitated_facility_location")
        model.Params.OutputFlag = 0
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1

        y = model.addVars(
            m,
            vtype=GRB.BINARY,
            obj=fixed_costs,
            name="open",
        )
        x = model.addVars(
            n,
            m,
            lb=0.0,
            ub=1.0,
            vtype=GRB.CONTINUOUS,
            obj={
                (i, j): shipping_costs[i][j]
                for i in range(n)
                for j in range(m)
            },
            name="fraction",
        )

        model.ModelSense = GRB.MINIMIZE

        model.addConstrs(
            (gp.quicksum(x[i, j] for j in range(m)) == 1.0 for i in range(n)),
            name="demand_fulfillment",
        )

        model.addConstrs(
            (
                gp.quicksum(demands[i] * x[i, j] for i in range(n))
                <= capacities[j] * y[j]
                for j in range(m)
            ),
            name="capacity",
        )

        # These constraints strengthen the LP relaxation when a plant's
        # capacity is large relative to an individual customer's demand.
        model.addConstrs(
            (x[i, j] <= y[j] for i in range(n) for j in range(m)),
            name="open_link",
        )

        # Supply the best constructive solution as a complete MIP start.
        if best_holder["allocation"] is not None:
            opened_set = set(best_holder["opened"])
            allocation = best_holder["allocation"]
            for j in range(m):
                y[j].Start = 1.0 if j in opened_set else 0.0
            for i in range(n):
                for j in range(m):
                    x[i, j].Start = allocation[i][j]

        model.update()

        elapsed = time.monotonic() - start_time
        remaining_time = args.time_limit - elapsed

        flat_x = [x[i, j] for i in range(n) for j in range(m)]
        y_list = [y[j] for j in range(m)]

        def incumbent_callback(cb_model, where):
            if where != GRB.Callback.MIPSOL:
                return

            try:
                y_values = cb_model.cbGetSolution(y_list)
                x_values = cb_model.cbGetSolution(flat_x)

                opened = [j for j in range(m) if y_values[j] > 0.5]
                opened_set = set(opened)
                allocation = [[0.0] * m for _ in range(n)]

                index = 0
                for i in range(n):
                    for j in range(m):
                        value = float(x_values[index])
                        index += 1
                        if j in opened_set and value > 0.0:
                            allocation[i][j] = value

                solution = make_solution(
                    opened,
                    allocation,
                    demands,
                    fixed_costs,
                    shipping_costs,
                )
                objective = solution["objective_value"]

                if objective < best_holder["objective"] - 1e-9:
                    best_holder["objective"] = objective
                    best_holder["solution"] = solution
                    best_holder["allocation"] = allocation
                    best_holder["opened"] = opened

                    if logger:
                        logger.log_solution(objective, solution)
            except Exception:
                # Logging or extraction must not interrupt the optimization.
                pass

        if remaining_time > 0.0:
            model.Params.TimeLimit = max(0.001, remaining_time)
            model.optimize(incumbent_callback)

            if model.SolCount > 0:
                opened = [j for j in range(m) if y[j].X > 0.5]
                opened_set = set(opened)
                allocation = [[0.0] * m for _ in range(n)]

                for i in range(n):
                    for j in range(m):
                        value = float(x[i, j].X)
                        if j in opened_set and value > 0.0:
                            allocation[i][j] = value

                solution = make_solution(
                    opened,
                    allocation,
                    demands,
                    fixed_costs,
                    shipping_costs,
                )
                objective = solution["objective_value"]

                if objective < best_holder["objective"] - 1e-9:
                    best_holder["objective"] = objective
                    best_holder["solution"] = solution
                    best_holder["allocation"] = allocation
                    best_holder["opened"] = opened
                    if logger:
                        logger.log_solution(objective, solution)

    except gp.GurobiError:
        # The constructive solution remains available as a valid fallback.
        pass

    if best_holder["solution"] is None:
        raise RuntimeError("Unable to construct a feasible solution.")

    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(best_holder["solution"], f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
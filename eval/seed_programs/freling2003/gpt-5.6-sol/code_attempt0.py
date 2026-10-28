import argparse
import json
import math
import os
import random
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


def canonical_inventory(loads, seasonal, capacities):
    """Minimum feasible inventory for fixed facility loads and nonnegative costs."""
    m = len(capacities)
    t_count = len(seasonal)
    result = [[0.0] * t_count for _ in range(m)]

    for i in range(m):
        inv = result[i]
        inv[t_count - 1] = 0.0
        for t in range(t_count - 1, 0, -1):
            inv[t - 1] = max(
                0.0,
                loads[i] * seasonal[t] - capacities[i][t] + inv[t],
            )
    return result


def make_solution(assignment, inventory, aggregate_costs, holding_costs):
    m = len(inventory)
    n = len(assignment)
    t_count = len(inventory[0]) if m else 0

    objective = 0.0
    for j in range(n):
        objective += aggregate_costs[assignment[j]][j]

    clean_inventory = {}
    for i in range(m):
        values = []
        for t in range(t_count):
            value = float(inventory[i][t])
            if -1e-8 < value < 0.0:
                value = 0.0
            values.append(value)
            objective += holding_costs[i][t] * value
        clean_inventory[str(i)] = values

    return {
        "objective_value": float(objective),
        "assignment": {str(j): int(assignment[j]) for j in range(n)},
        "inventory": clean_inventory,
    }


def assignment_loads(assignment, demands, num_facilities):
    loads = [0.0] * num_facilities
    for j, i in enumerate(assignment):
        loads[i] += demands[j]
    return loads


def compute_load_limits(seasonal, capacities):
    """Maximum total-horizon demand assignable to each facility."""
    limits = []
    for row in capacities:
        cumulative_factor = 0.0
        cumulative_capacity = 0.0
        limit = math.inf
        for s, cap in zip(seasonal, row):
            cumulative_factor += s
            cumulative_capacity += cap
            if cumulative_factor > 1e-14:
                limit = min(limit, cumulative_capacity / cumulative_factor)
        limits.append(limit)
    return limits


def greedy_assignment(
    order,
    strategy,
    demands,
    aggregate_costs,
    load_limits,
    rng,
):
    m = len(load_limits)
    assignment = [-1] * len(demands)
    loads = [0.0] * m

    for j in order:
        feasible = [
            i
            for i in range(m)
            if loads[i] + demands[j] <= load_limits[i] + 1e-9
        ]
        if not feasible:
            return None

        if strategy == 0:
            # Cheapest feasible facility.
            selected = min(
                feasible,
                key=lambda i: (
                    aggregate_costs[i][j],
                    load_limits[i] - loads[i] - demands[j],
                ),
            )
        elif strategy == 1:
            # Best fit, with cost as tie breaker.
            selected = min(
                feasible,
                key=lambda i: (
                    load_limits[i] - loads[i] - demands[j],
                    aggregate_costs[i][j],
                ),
            )
        elif strategy == 2:
            # Preserve large-capacity facilities where possible.
            selected = max(
                feasible,
                key=lambda i: (
                    load_limits[i] - loads[i] - demands[j],
                    -aggregate_costs[i][j],
                ),
            )
        else:
            # Randomized compromise between cost and packing tightness.
            costs = [aggregate_costs[i][j] for i in feasible]
            c_min = min(costs)
            c_max = max(costs)
            scored = []
            for i in feasible:
                cost_norm = (
                    (aggregate_costs[i][j] - c_min) / (c_max - c_min)
                    if c_max > c_min + 1e-15
                    else 0.0
                )
                residual = load_limits[i] - loads[i] - demands[j]
                residual_norm = residual / max(1.0, load_limits[i])
                scored.append(
                    (
                        cost_norm + rng.uniform(0.15, 1.0) * residual_norm
                        + rng.random() * 1e-7,
                        i,
                    )
                )
            selected = min(scored)[1]

        assignment[j] = selected
        loads[selected] += demands[j]

    return assignment


def main():
    args = parse_args()
    start_time = time.monotonic()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    m = int(data["parameters"]["num_facilities"])
    n = int(data["parameters"]["num_customers"])
    t_count = int(data["parameters"]["num_periods"])

    seasonal = [float(v) for v in data["seasonal_factors"]]
    demands = [float(v) for v in data["demands"]]
    capacities = [
        [float(v) for v in row] for row in data["capacities"]
    ]
    holding_costs = [
        [float(v) for v in row] for row in data["holding_costs"]
    ]
    transportation_costs = data["transportation_costs"]

    # Assignment is fixed across periods, so period supply costs can be summed.
    aggregate_costs = [
        [
            sum(float(transportation_costs[i][j][t]) for t in range(t_count))
            for j in range(n)
        ]
        for i in range(m)
    ]

    load_limits = compute_load_limits(seasonal, capacities)
    nonnegative_holding = all(
        holding_costs[i][t] >= -1e-12
        for i in range(m)
        for t in range(t_count)
    )

    best_solution = None
    best_assignment = None
    best_objective = math.inf

    def register_solution(solution, assignment=None):
        nonlocal best_solution, best_assignment, best_objective
        objective = float(solution["objective_value"])
        if objective < best_objective - 1e-7:
            best_objective = objective
            best_solution = solution
            best_assignment = list(assignment) if assignment is not None else None
            if logger:
                logger.log_solution(objective, solution)

    # Build several inexpensive feasible starts.
    rng = random.Random(0)
    base_desc = sorted(range(n), key=lambda j: (-demands[j], j))
    cost_regret_order = sorted(
        range(n),
        key=lambda j: (
            -(
                sorted(aggregate_costs[i][j] for i in range(m))[1]
                - sorted(aggregate_costs[i][j] for i in range(m))[0]
                if m > 1
                else 0.0
            ),
            -demands[j],
        ),
    )

    heuristic_budget = min(0.75, max(0.05, args.time_limit * 0.05))
    attempt = 0
    while (
        time.monotonic() - start_time < heuristic_budget
        and attempt < 250
    ):
        if attempt < 4:
            order = base_desc
            strategy = attempt
        elif attempt < 8:
            order = cost_regret_order
            strategy = attempt % 4
        else:
            # Mostly retain largest-first behavior while adding diversity.
            order = list(base_desc)
            block_size = max(2, min(12, int(math.sqrt(max(1, n))) + 1))
            for begin in range(0, n, block_size):
                block = order[begin:begin + block_size]
                rng.shuffle(block)
                order[begin:begin + block_size] = block
            strategy = 3

        candidate = greedy_assignment(
            order,
            strategy,
            demands,
            aggregate_costs,
            load_limits,
            rng,
        )
        if candidate is not None:
            loads = assignment_loads(candidate, demands, m)
            inventory = canonical_inventory(loads, seasonal, capacities)
            solution = make_solution(
                candidate, inventory, aggregate_costs, holding_costs
            )
            register_solution(solution, candidate)

        attempt += 1

    model = gp.Model("single_source_production_inventory")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1

    remaining_time = max(
        0.01, float(args.time_limit) - (time.monotonic() - start_time)
    )
    model.Params.TimeLimit = remaining_time

    x = model.addVars(m, n, vtype=GRB.BINARY, name="assign")
    inventory_vars = model.addVars(
        m, t_count, lb=0.0, vtype=GRB.CONTINUOUS, name="inventory"
    )

    model.setObjective(
        gp.quicksum(
            aggregate_costs[i][j] * x[i, j]
            for i in range(m)
            for j in range(n)
        )
        + gp.quicksum(
            holding_costs[i][t] * inventory_vars[i, t]
            for i in range(m)
            for t in range(t_count)
        ),
        GRB.MINIMIZE,
    )

    model.addConstrs(
        (
            gp.quicksum(x[i, j] for i in range(m)) == 1
            for j in range(n)
        ),
        name="single_source",
    )

    for i in range(m):
        for t in range(t_count):
            period_demand = gp.quicksum(
                demands[j] * seasonal[t] * x[i, j] for j in range(n)
            )
            previous_inventory = (
                inventory_vars[i, t - 1] if t > 0 else 0.0
            )
            model.addConstr(
                period_demand + inventory_vars[i, t]
                <= capacities[i][t] + previous_inventory,
                name=f"balance_{i}_{t}",
            )

        # Strong implied cumulative-capacity constraint.
        if math.isfinite(load_limits[i]):
            model.addConstr(
                gp.quicksum(demands[j] * x[i, j] for j in range(n))
                <= load_limits[i],
                name=f"total_load_{i}",
            )

    # Eliminate assignment choices that cannot fit even by themselves.
    for i in range(m):
        for j in range(n):
            if demands[j] > load_limits[i] + 1e-9:
                x[i, j].UB = 0.0

    # Supply a complete MIP start when the constructive heuristic succeeded.
    if best_assignment is not None:
        start_loads = assignment_loads(best_assignment, demands, m)
        start_inventory = canonical_inventory(
            start_loads, seasonal, capacities
        )
        for i in range(m):
            for j in range(n):
                x[i, j].Start = 1.0 if best_assignment[j] == i else 0.0
            for t in range(t_count):
                inventory_vars[i, t].Start = start_inventory[i][t]

    callback_best = [best_objective]

    def incumbent_callback(cb_model, where):
        if where != GRB.Callback.MIPSOL:
            return

        try:
            x_values = cb_model.cbGetSolution(x)
            inv_values = cb_model.cbGetSolution(inventory_vars)

            assignment = []
            for j in range(n):
                selected = max(range(m), key=lambda i: x_values[i, j])
                assignment.append(selected)

            extracted_inventory = [
                [
                    max(0.0, float(inv_values[i, t]))
                    for t in range(t_count)
                ]
                for i in range(m)
            ]

            # Canonical inventory avoids arbitrary excess stock when all
            # holding costs are nonnegative.
            if nonnegative_holding:
                loads = assignment_loads(assignment, demands, m)
                extracted_inventory = canonical_inventory(
                    loads, seasonal, capacities
                )

            solution = make_solution(
                assignment,
                extracted_inventory,
                aggregate_costs,
                holding_costs,
            )
            objective = solution["objective_value"]

            if objective < callback_best[0] - 1e-7:
                callback_best[0] = objective
                register_solution(solution, assignment)
        except Exception:
            # A logging failure must not interrupt the optimization.
            return

    model.optimize(incumbent_callback)

    if model.SolCount > 0:
        final_assignment = []
        for j in range(n):
            selected = max(range(m), key=lambda i: x[i, j].X)
            final_assignment.append(selected)

        if nonnegative_holding:
            final_loads = assignment_loads(final_assignment, demands, m)
            final_inventory = canonical_inventory(
                final_loads, seasonal, capacities
            )
        else:
            final_inventory = [
                [
                    max(0.0, float(inventory_vars[i, t].X))
                    for t in range(t_count)
                ]
                for i in range(m)
            ]

        final_solution = make_solution(
            final_assignment,
            final_inventory,
            aggregate_costs,
            holding_costs,
        )
        register_solution(final_solution, final_assignment)

    if best_solution is None:
        raise RuntimeError(
            "No feasible solution was found within the time limit. "
            "The instance may be infeasible."
        )

    output_dir = os.path.dirname(os.path.abspath(args.solution_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(best_solution, f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
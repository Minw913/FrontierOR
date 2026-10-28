import argparse
import json
import math
import os
import random
import sys
import time

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception:
    gp = None
    GRB = None

from solution_logger import SolutionLogger


EPS = 1e-8


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    return parser.parse_args()


def calculate_load_limits(seasonal_factors, capacities):
    limits = []
    for capacity_row in capacities:
        cumulative_demand_factor = 0.0
        cumulative_capacity = 0.0
        limit = math.inf

        for factor, capacity in zip(seasonal_factors, capacity_row):
            cumulative_demand_factor += factor
            cumulative_capacity += capacity
            if cumulative_demand_factor > EPS:
                limit = min(
                    limit,
                    cumulative_capacity / cumulative_demand_factor,
                )

        limits.append(limit)

    return limits


def minimum_inventory(load, seasonal_factors, capacity_row):
    """Minimum end-of-period inventory for a fixed facility load."""
    num_periods = len(seasonal_factors)
    inventory = [0.0] * num_periods

    if num_periods == 0:
        return inventory

    inventory[-1] = 0.0
    for t in range(num_periods - 1, 0, -1):
        inventory[t - 1] = max(
            0.0,
            load * seasonal_factors[t]
            + inventory[t]
            - capacity_row[t],
        )

    return inventory


def inventory_cost(
    facility,
    load,
    seasonal_factors,
    capacities,
    holding_costs,
):
    inventory = minimum_inventory(
        load,
        seasonal_factors,
        capacities[facility],
    )
    return sum(
        holding_costs[facility][t] * inventory[t]
        for t in range(len(seasonal_factors))
    )


def get_loads(assignment, demands, num_facilities):
    loads = [0.0] * num_facilities
    for customer, facility in enumerate(assignment):
        loads[facility] += demands[customer]
    return loads


def build_solution(
    assignment,
    demands,
    aggregate_costs,
    seasonal_factors,
    capacities,
    holding_costs,
):
    num_facilities = len(capacities)
    num_periods = len(seasonal_factors)
    loads = get_loads(assignment, demands, num_facilities)

    inventory = []
    objective = 0.0

    for customer, facility in enumerate(assignment):
        objective += aggregate_costs[facility][customer]

    for facility in range(num_facilities):
        values = minimum_inventory(
            loads[facility],
            seasonal_factors,
            capacities[facility],
        )
        cleaned = []
        for t in range(num_periods):
            value = max(0.0, float(values[t]))
            if value < 1e-10:
                value = 0.0
            cleaned.append(value)
            objective += holding_costs[facility][t] * value
        inventory.append(cleaned)

    return {
        "objective_value": float(objective),
        "assignment": {
            str(customer): int(facility)
            for customer, facility in enumerate(assignment)
        },
        "inventory": {
            str(facility): inventory[facility]
            for facility in range(num_facilities)
        },
    }


def is_feasible_assignment(assignment, demands, load_limits):
    if assignment is None or any(facility < 0 for facility in assignment):
        return False

    loads = get_loads(assignment, demands, len(load_limits))
    return all(
        loads[i] <= load_limits[i] + 1e-7
        for i in range(len(load_limits))
    )


def greedy_assignment(
    customer_order,
    demands,
    aggregate_costs,
    load_limits,
    mode,
    rng,
):
    num_facilities = len(load_limits)
    assignment = [-1] * len(demands)
    loads = [0.0] * num_facilities

    for customer in customer_order:
        demand = demands[customer]
        feasible = [
            facility
            for facility in range(num_facilities)
            if loads[facility] + demand
            <= load_limits[facility] + EPS
        ]

        if not feasible:
            return None

        if mode == 0:
            # Cheapest feasible facility.
            facility = min(
                feasible,
                key=lambda i: (
                    aggregate_costs[i][customer],
                    loads[i],
                ),
            )
        elif mode == 1:
            # Tightest feasible packing.
            facility = min(
                feasible,
                key=lambda i: (
                    load_limits[i] - loads[i] - demand,
                    aggregate_costs[i][customer],
                ),
            )
        elif mode == 2:
            # Preserve scarce residual capacity.
            facility = max(
                feasible,
                key=lambda i: (
                    load_limits[i] - loads[i] - demand,
                    -aggregate_costs[i][customer],
                ),
            )
        else:
            minimum_cost = min(
                aggregate_costs[i][customer] for i in feasible
            )
            maximum_cost = max(
                aggregate_costs[i][customer] for i in feasible
            )

            scored = []
            for i in feasible:
                if maximum_cost > minimum_cost + EPS:
                    normalized_cost = (
                        aggregate_costs[i][customer] - minimum_cost
                    ) / (maximum_cost - minimum_cost)
                else:
                    normalized_cost = 0.0

                residual_ratio = (
                    load_limits[i] - loads[i] - demand
                ) / max(1.0, load_limits[i])

                score = (
                    normalized_cost
                    + rng.uniform(0.1, 1.0) * residual_ratio
                    + rng.random() * 1e-8
                )
                scored.append((score, i))

            facility = min(scored)[1]

        assignment[customer] = facility
        loads[facility] += demand

    return assignment


def improve_by_relocation(
    assignment,
    demands,
    aggregate_costs,
    seasonal_factors,
    capacities,
    holding_costs,
    load_limits,
    deadline,
    register_assignment,
):
    """First-improvement relocation local search."""
    num_facilities = len(load_limits)
    num_customers = len(demands)
    loads = get_loads(assignment, demands, num_facilities)

    facility_inventory_cost = [
        inventory_cost(
            i,
            loads[i],
            seasonal_factors,
            capacities,
            holding_costs,
        )
        for i in range(num_facilities)
    ]

    improved = True
    while improved and time.monotonic() < deadline:
        improved = False

        customer_order = sorted(
            range(num_customers),
            key=lambda j: -demands[j],
        )

        for customer in customer_order:
            if time.monotonic() >= deadline:
                break

            old_facility = assignment[customer]
            demand = demands[customer]

            best_delta = -EPS
            best_facility = old_facility
            best_old_inventory_cost = facility_inventory_cost[old_facility]
            best_new_inventory_cost = None

            old_after_load = loads[old_facility] - demand
            old_after_cost = inventory_cost(
                old_facility,
                old_after_load,
                seasonal_factors,
                capacities,
                holding_costs,
            )

            for new_facility in range(num_facilities):
                if new_facility == old_facility:
                    continue
                if (
                    loads[new_facility] + demand
                    > load_limits[new_facility] + EPS
                ):
                    continue

                new_after_load = loads[new_facility] + demand
                new_after_cost = inventory_cost(
                    new_facility,
                    new_after_load,
                    seasonal_factors,
                    capacities,
                    holding_costs,
                )

                delta = (
                    aggregate_costs[new_facility][customer]
                    - aggregate_costs[old_facility][customer]
                    + old_after_cost
                    - facility_inventory_cost[old_facility]
                    + new_after_cost
                    - facility_inventory_cost[new_facility]
                )

                if delta < best_delta:
                    best_delta = delta
                    best_facility = new_facility
                    best_old_inventory_cost = old_after_cost
                    best_new_inventory_cost = new_after_cost

            if best_facility != old_facility:
                assignment[customer] = best_facility
                loads[old_facility] -= demand
                loads[best_facility] += demand
                facility_inventory_cost[old_facility] = (
                    best_old_inventory_cost
                )
                facility_inventory_cost[best_facility] = (
                    best_new_inventory_cost
                )

                register_assignment(assignment)
                improved = True
                break

    return assignment


def main():
    args = parse_args()
    start_time = time.monotonic()
    deadline = start_time + max(0, args.time_limit)

    logger = (
        SolutionLogger(args.log_path, sense="minimize")
        if args.log_path
        else None
    )

    with open(args.instance_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    num_facilities = int(data["parameters"]["num_facilities"])
    num_customers = int(data["parameters"]["num_customers"])
    num_periods = int(data["parameters"]["num_periods"])

    seasonal_factors = [
        float(value) for value in data["seasonal_factors"]
    ]
    demands = [float(value) for value in data["demands"]]
    capacities = [
        [float(value) for value in row]
        for row in data["capacities"]
    ]
    holding_costs = [
        [float(value) for value in row]
        for row in data["holding_costs"]
    ]
    transportation_costs = data["transportation_costs"]

    aggregate_costs = [
        [
            sum(
                float(transportation_costs[i][j][t])
                for t in range(num_periods)
            )
            for j in range(num_customers)
        ]
        for i in range(num_facilities)
    ]

    load_limits = calculate_load_limits(
        seasonal_factors,
        capacities,
    )

    best_solution = None
    best_assignment = None
    best_objective = math.inf

    def register_assignment(assignment):
        nonlocal best_solution, best_assignment, best_objective

        candidate = list(assignment)
        if not is_feasible_assignment(
            candidate,
            demands,
            load_limits,
        ):
            return False

        solution = build_solution(
            candidate,
            demands,
            aggregate_costs,
            seasonal_factors,
            capacities,
            holding_costs,
        )
        objective = solution["objective_value"]

        if objective < best_objective - 1e-7:
            best_objective = objective
            best_solution = solution
            best_assignment = candidate
            if logger:
                logger.log_solution(objective, solution)
            return True

        return False

    # Construct several initial solutions before attempting to invoke Gurobi.
    rng = random.Random(0)

    descending_demand = sorted(
        range(num_customers),
        key=lambda j: (-demands[j], j),
    )

    regrets = []
    for customer in range(num_customers):
        costs = sorted(
            aggregate_costs[i][customer]
            for i in range(num_facilities)
        )
        regret = costs[1] - costs[0] if len(costs) > 1 else 0.0
        regrets.append(regret)

    regret_order = sorted(
        range(num_customers),
        key=lambda j: (-regrets[j], -demands[j], j),
    )

    heuristic_deadline = min(
        deadline,
        start_time + min(1.0, max(0.05, args.time_limit * 0.08)),
    )

    attempt = 0
    while time.monotonic() < heuristic_deadline and attempt < 300:
        if attempt < 4:
            order = descending_demand
            mode = attempt
        elif attempt < 8:
            order = regret_order
            mode = attempt - 4
        else:
            order = list(descending_demand)
            block_size = max(
                2,
                min(16, int(math.sqrt(max(1, num_customers))) + 1),
            )
            for begin in range(0, num_customers, block_size):
                block = order[begin:begin + block_size]
                rng.shuffle(block)
                order[begin:begin + block_size] = block
            mode = 3

        assignment = greedy_assignment(
            order,
            demands,
            aggregate_costs,
            load_limits,
            mode,
            rng,
        )
        if assignment is not None:
            register_assignment(assignment)

        attempt += 1

    # A pure-Python branch-and-bound fallback is used if Gurobi cannot be
    # initialized (for example, when no license is available). It also finds
    # a feasible solution when the greedy construction failed.
    def run_python_branch_and_bound(search_deadline):
        nonlocal best_assignment

        if num_customers == 0:
            register_assignment([])
            return

        sys.setrecursionlimit(max(2000, num_customers + 100))

        order = sorted(
            range(num_customers),
            key=lambda j: (
                -demands[j],
                -regrets[j],
                j,
            ),
        )

        minimum_transport = [
            min(aggregate_costs[i][j] for i in range(num_facilities))
            for j in order
        ]
        suffix_minimum = [0.0] * (num_customers + 1)
        for position in range(num_customers - 1, -1, -1):
            suffix_minimum[position] = (
                suffix_minimum[position + 1]
                + minimum_transport[position]
            )

        assignment = [-1] * num_customers
        loads = [0.0] * num_facilities
        current_inventory_costs = [0.0] * num_facilities

        nonnegative_holding = all(
            holding_costs[i][t] >= -EPS
            for i in range(num_facilities)
            for t in range(num_periods)
        )

        nodes = [0]

        def search(position, current_cost):
            nodes[0] += 1
            if nodes[0] % 256 == 0 and time.monotonic() >= search_deadline:
                return
            if time.monotonic() >= search_deadline:
                return

            if position == num_customers:
                register_assignment(assignment)
                return

            if (
                nonnegative_holding
                and best_solution is not None
                and current_cost + suffix_minimum[position]
                >= best_objective - 1e-8
            ):
                return

            customer = order[position]
            demand = demands[customer]

            choices = []
            for facility in range(num_facilities):
                new_load = loads[facility] + demand
                if new_load > load_limits[facility] + EPS:
                    continue

                new_inventory_cost = inventory_cost(
                    facility,
                    new_load,
                    seasonal_factors,
                    capacities,
                    holding_costs,
                )
                delta = (
                    aggregate_costs[facility][customer]
                    + new_inventory_cost
                    - current_inventory_costs[facility]
                )
                choices.append(
                    (delta, facility, new_inventory_cost)
                )

            choices.sort(key=lambda item: item[0])

            for delta, facility, new_inventory_cost in choices:
                if time.monotonic() >= search_deadline:
                    return

                if (
                    nonnegative_holding
                    and best_solution is not None
                    and current_cost
                    + delta
                    + suffix_minimum[position + 1]
                    >= best_objective - 1e-8
                ):
                    continue

                old_inventory_cost = current_inventory_costs[facility]
                loads[facility] += demand
                current_inventory_costs[facility] = new_inventory_cost
                assignment[customer] = facility

                search(position + 1, current_cost + delta)

                assignment[customer] = -1
                current_inventory_costs[facility] = old_inventory_cost
                loads[facility] -= demand

        search(0, 0.0)

    # If no greedy solution was found, spend a small amount of time obtaining
    # one before starting the MIP.
    if best_solution is None and time.monotonic() < deadline:
        feasibility_deadline = min(
            deadline,
            time.monotonic() + max(0.05, min(2.0, args.time_limit * 0.15)),
        )
        run_python_branch_and_bound(feasibility_deadline)

    gurobi_succeeded = False

    if gp is not None and time.monotonic() < deadline:
        model = None
        try:
            # Model construction itself can fail when a Gurobi license is not
            # available, so it is deliberately inside this try block.
            model = gp.Model("single_source_production_inventory")
            model.Params.OutputFlag = 0
            model.Params.Seed = 0
            model.Params.MIPGap = 1e-4
            model.Params.NumericFocus = 0
            model.Params.Threads = 1
            model.Params.TimeLimit = max(
                0.01,
                deadline - time.monotonic(),
            )

            x = model.addVars(
                num_facilities,
                num_customers,
                vtype=GRB.BINARY,
                name="assignment",
            )
            inv = model.addVars(
                num_facilities,
                num_periods,
                lb=0.0,
                vtype=GRB.CONTINUOUS,
                name="inventory",
            )

            model.setObjective(
                gp.quicksum(
                    aggregate_costs[i][j] * x[i, j]
                    for i in range(num_facilities)
                    for j in range(num_customers)
                )
                + gp.quicksum(
                    holding_costs[i][t] * inv[i, t]
                    for i in range(num_facilities)
                    for t in range(num_periods)
                ),
                GRB.MINIMIZE,
            )

            model.addConstrs(
                (
                    gp.quicksum(
                        x[i, j] for i in range(num_facilities)
                    ) == 1
                    for j in range(num_customers)
                ),
                name="single_source",
            )

            for i in range(num_facilities):
                for t in range(num_periods):
                    previous_inventory = inv[i, t - 1] if t > 0 else 0.0
                    demand_expression = gp.quicksum(
                        demands[j] * seasonal_factors[t] * x[i, j]
                        for j in range(num_customers)
                    )
                    model.addConstr(
                        demand_expression + inv[i, t]
                        <= capacities[i][t] + previous_inventory,
                        name=f"capacity_{i}_{t}",
                    )

                if math.isfinite(load_limits[i]):
                    model.addConstr(
                        gp.quicksum(
                            demands[j] * x[i, j]
                            for j in range(num_customers)
                        )
                        <= load_limits[i],
                        name=f"load_limit_{i}",
                    )

            for i in range(num_facilities):
                for j in range(num_customers):
                    if demands[j] > load_limits[i] + EPS:
                        x[i, j].UB = 0.0

            if best_assignment is not None:
                start_loads = get_loads(
                    best_assignment,
                    demands,
                    num_facilities,
                )
                for i in range(num_facilities):
                    start_inventory = minimum_inventory(
                        start_loads[i],
                        seasonal_factors,
                        capacities[i],
                    )
                    for j in range(num_customers):
                        x[i, j].Start = (
                            1.0 if best_assignment[j] == i else 0.0
                        )
                    for t in range(num_periods):
                        inv[i, t].Start = start_inventory[t]

            callback_best = [best_objective]

            def callback(cb_model, where):
                if where != GRB.Callback.MIPSOL:
                    return

                try:
                    values = cb_model.cbGetSolution(x)
                    candidate_assignment = []

                    for j in range(num_customers):
                        facility = max(
                            range(num_facilities),
                            key=lambda i: values[i, j],
                        )
                        candidate_assignment.append(facility)

                    if is_feasible_assignment(
                        candidate_assignment,
                        demands,
                        load_limits,
                    ):
                        solution = build_solution(
                            candidate_assignment,
                            demands,
                            aggregate_costs,
                            seasonal_factors,
                            capacities,
                            holding_costs,
                        )
                        objective = solution["objective_value"]

                        if objective < callback_best[0] - 1e-7:
                            callback_best[0] = objective
                            register_assignment(candidate_assignment)
                except Exception:
                    pass

            model.optimize(callback)
            gurobi_succeeded = True

            if model.SolCount > 0:
                candidate_assignment = []
                for j in range(num_customers):
                    facility = max(
                        range(num_facilities),
                        key=lambda i: x[i, j].X,
                    )
                    candidate_assignment.append(facility)
                register_assignment(candidate_assignment)

        except Exception:
            # Continue with the license-independent algorithm.
            gurobi_succeeded = False
        finally:
            if model is not None:
                try:
                    model.dispose()
                except Exception:
                    pass

    # If Gurobi was unavailable, use the remaining runtime for pure-Python
    # branch-and-bound.
    if not gurobi_succeeded and time.monotonic() < deadline:
        run_python_branch_and_bound(deadline)

    # Improve the best result with inexpensive assignment relocations.
    if best_assignment is not None and time.monotonic() < deadline:
        local_assignment = list(best_assignment)
        improve_by_relocation(
            local_assignment,
            demands,
            aggregate_costs,
            seasonal_factors,
            capacities,
            holding_costs,
            load_limits,
            deadline,
            register_assignment,
        )

    if best_solution is None:
        raise RuntimeError(
            "No feasible solution was found. The instance may be infeasible "
            "or the time limit may be too small."
        )

    solution_directory = os.path.dirname(
        os.path.abspath(args.solution_path)
    )
    if solution_directory:
        os.makedirs(solution_directory, exist_ok=True)

    with open(args.solution_path, "w", encoding="utf-8") as file:
        json.dump(
            best_solution,
            file,
            indent=2,
            allow_nan=False,
        )


if __name__ == "__main__":
    main()
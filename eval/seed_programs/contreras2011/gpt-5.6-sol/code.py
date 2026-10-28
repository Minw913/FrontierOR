import argparse
import json
import math
import os
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


EPS = 1e-6


def configure_gurobi(model, time_limit):
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = max(0.01, float(time_limit))


def exact_objective(
    assignment,
    flow,
    distance,
    setup_costs,
    chi,
    alpha,
    delta,
):
    hubs = set(assignment)
    value = sum(setup_costs[k] for k in hubs)
    n = len(assignment)

    for i in range(n):
        hi = assignment[i]
        for j in range(n):
            quantity = flow[i][j]
            if quantity == 0.0:
                continue
            hj = assignment[j]
            value += quantity * (
                chi * distance[i][hi]
                + alpha * distance[hi][hj]
                + delta * distance[hj][j]
            )

    return float(value)


def build_solution(
    assignment,
    flow,
    distance,
    setup_costs,
    chi,
    alpha,
    delta,
):
    assignment = [int(k) for k in assignment]
    return {
        "objective_value": exact_objective(
            assignment,
            flow,
            distance,
            setup_costs,
            chi,
            alpha,
            delta,
        ),
        "hubs": sorted(set(assignment)),
        "assignment": assignment,
    }


def is_feasible(assignment, demands, capacities, required_total_capacity):
    n = len(demands)

    if assignment is None or len(assignment) != n:
        return False

    for k in assignment:
        if not isinstance(k, int) or k < 0 or k >= n:
            return False

    hubs = set(assignment)

    # Every opened hub must be assigned to itself.
    for k in hubs:
        if assignment[k] != k:
            return False

    loads = [0.0] * n
    for i, k in enumerate(assignment):
        loads[k] += demands[i]

    for k in hubs:
        tolerance = EPS * max(1.0, abs(capacities[k]))
        if loads[k] > capacities[k] + tolerance:
            return False

    total_open_capacity = sum(capacities[k] for k in hubs)
    if total_open_capacity + EPS * max(1.0, required_total_capacity) < required_total_capacity:
        return False

    return True


def update_incumbent(candidate, incumbent, logger):
    if candidate is None:
        return incumbent

    if (
        incumbent is None
        or candidate["objective_value"] < incumbent["objective_value"] - 1e-7
    ):
        if logger:
            logger.log_solution(candidate["objective_value"], candidate)
        return candidate

    return incumbent


def greedy_feasible_assignments(
    demands,
    capacities,
    access_cost,
    required_total_capacity,
):
    """
    Generate several feasible assignments. All hubs that can legally serve
    themselves are initially opened, after which remaining nodes are packed.
    """
    n = len(demands)
    viable_hubs = [
        k for k in range(n)
        if capacities[k] + EPS >= demands[k]
    ]

    if not viable_hubs:
        return []

    orderings = [
        sorted(range(n), key=lambda i: (-demands[i], i)),
        sorted(
            range(n),
            key=lambda i: (
                min(
                    (
                        access_cost[i][k]
                        for k in viable_hubs
                        if capacities[k] + EPS >= demands[i]
                    ),
                    default=math.inf,
                ),
                -demands[i],
                i,
            ),
        ),
        list(range(n)),
    ]

    results = []
    seen = set()

    for ordering in orderings:
        assignment = [-1] * n
        residual = [0.0] * n

        # Opening every viable hub and assigning it to itself creates maximum
        # flexibility for packing nonviable nodes.
        for k in viable_hubs:
            assignment[k] = k
            residual[k] = capacities[k] - demands[k]

        success = True
        for i in ordering:
            if assignment[i] >= 0:
                continue

            choices = [
                k for k in viable_hubs
                if residual[k] + EPS >= demands[i]
            ]
            if not choices:
                success = False
                break

            # Prefer low access cost, then a tight capacity fit.
            selected = min(
                choices,
                key=lambda k: (
                    access_cost[i][k],
                    residual[k] - demands[i],
                    k,
                ),
            )
            assignment[i] = selected
            residual[selected] -= demands[i]

        if success and is_feasible(
            assignment,
            demands,
            capacities,
            required_total_capacity,
        ):
            key = tuple(assignment)
            if key not in seen:
                seen.add(key)
                results.append(assignment)

    return results


def solve_assignment_surrogate(
    n,
    demands,
    capacities,
    required_total_capacity,
    setup_costs,
    access_cost,
    time_limit,
    start_assignment,
    solution_builder,
    incumbent_value,
    logger,
):
    """
    Solve a capacitated facility-location surrogate. It omits inter-hub
    transfer costs but generally identifies a strong, compact hub set.
    """
    if time_limit <= 0.01:
        return None

    try:
        model = gp.Model("hub_surrogate")
        configure_gurobi(model, time_limit)

        x = model.addVars(n, n, vtype=GRB.BINARY, name="x")

        for i in range(n):
            model.addConstr(
                gp.quicksum(x[i, k] for k in range(n)) == 1
            )

        for i in range(n):
            for k in range(n):
                if i != k:
                    model.addConstr(x[i, k] <= x[k, k])

        for k in range(n):
            model.addConstr(
                gp.quicksum(demands[i] * x[i, k] for i in range(n))
                <= capacities[k] * x[k, k]
            )

        model.addConstr(
            gp.quicksum(capacities[k] * x[k, k] for k in range(n))
            >= required_total_capacity
        )

        objective = gp.LinExpr()
        for k in range(n):
            objective += setup_costs[k] * x[k, k]

        for i in range(n):
            for k in range(n):
                coefficient = access_cost[i][k]
                if coefficient != 0.0:
                    objective += coefficient * x[i, k]

        model.setObjective(objective, GRB.MINIMIZE)

        if start_assignment is not None:
            for i in range(n):
                assigned_hub = start_assignment[i]
                for k in range(n):
                    x[i, k].Start = 1.0 if k == assigned_hub else 0.0

        callback_best = [float(incumbent_value)]
        flat_keys = [(i, k) for i in range(n) for k in range(n)]
        flat_vars = [x[i, k] for i, k in flat_keys]

        def callback(cb_model, where):
            if where != GRB.Callback.MIPSOL:
                return

            try:
                values = cb_model.cbGetSolution(flat_vars)
                assignment = [0] * n

                for i in range(n):
                    begin = i * n
                    assignment[i] = max(
                        range(n),
                        key=lambda k: values[begin + k],
                    )

                solution = solution_builder(assignment)
                if (
                    solution is not None
                    and solution["objective_value"] < callback_best[0] - 1e-7
                ):
                    callback_best[0] = solution["objective_value"]
                    if logger:
                        logger.log_solution(
                            solution["objective_value"],
                            solution,
                        )
            except Exception:
                # Logging or numerical extraction must not interrupt Gurobi.
                return

        model.optimize(callback)

        if model.SolCount <= 0:
            return None

        assignment = [
            max(range(n), key=lambda k: x[i, k].X)
            for i in range(n)
        ]

        return assignment

    except gp.GurobiError:
        return None
    except Exception:
        return None


def select_candidate_hubs(
    n,
    demands,
    capacities,
    setup_costs,
    access_cost,
    preferred_hubs,
    maximum_count,
):
    viable = [
        k for k in range(n)
        if capacities[k] + EPS >= demands[k]
    ]

    preferred = [k for k in preferred_hubs if k in viable]
    chosen = set(preferred[:maximum_count])

    centrality_order = sorted(
        viable,
        key=lambda k: (
            setup_costs[k] + sum(access_cost[i][k] for i in range(n)),
            k,
        ),
    )

    capacity_order = sorted(
        viable,
        key=lambda k: (
            setup_costs[k] / max(capacities[k], EPS),
            -capacities[k],
            k,
        ),
    )

    a = 0
    b = 0
    while len(chosen) < maximum_count:
        added = False

        while a < len(centrality_order):
            k = centrality_order[a]
            a += 1
            if k not in chosen:
                chosen.add(k)
                added = True
                break

        if len(chosen) >= maximum_count:
            break

        while b < len(capacity_order):
            k = capacity_order[b]
            b += 1
            if k not in chosen:
                chosen.add(k)
                added = True
                break

        if not added:
            break

    return sorted(chosen)


def solve_exact_restricted(
    candidates,
    n,
    demands,
    incoming,
    capacities,
    required_total_capacity,
    setup_costs,
    flow,
    distance,
    chi,
    alpha,
    delta,
    time_limit,
    start_assignment,
    solution_builder,
    incumbent_value,
    logger,
):
    """
    Exact linear formulation over a restricted set of possible hubs.

    q[i,k,m] is the amount of origin i's traffic transported from origin hub
    k to destination hub m. Since assignments are binary and unique, the
    transportation balance constraints recover the exact routing cost.
    """
    if time_limit <= 0.01 or not candidates:
        return None

    candidates = list(candidates)

    try:
        model = gp.Model("restricted_exact_hub_model")
        configure_gurobi(model, time_limit)

        x = model.addVars(
            range(n),
            candidates,
            vtype=GRB.BINARY,
            name="x",
        )
        q = model.addVars(
            range(n),
            candidates,
            candidates,
            lb=0.0,
            vtype=GRB.CONTINUOUS,
            name="q",
        )

        for i in range(n):
            model.addConstr(
                gp.quicksum(x[i, k] for k in candidates) == 1
            )

        for i in range(n):
            for k in candidates:
                if i != k:
                    model.addConstr(x[i, k] <= x[k, k])

        for k in candidates:
            model.addConstr(
                gp.quicksum(demands[i] * x[i, k] for i in range(n))
                <= capacities[k] * x[k, k]
            )

        model.addConstr(
            gp.quicksum(
                capacities[k] * x[k, k]
                for k in candidates
            )
            >= required_total_capacity
        )

        for i in range(n):
            for k in candidates:
                model.addConstr(
                    gp.quicksum(q[i, k, m] for m in candidates)
                    == demands[i] * x[i, k]
                )

            for m in candidates:
                model.addConstr(
                    gp.quicksum(q[i, k, m] for k in candidates)
                    == gp.quicksum(
                        flow[i][j] * x[j, m]
                        for j in range(n)
                    )
                )

        objective = gp.LinExpr()

        for k in candidates:
            objective += setup_costs[k] * x[k, k]

        for i in range(n):
            for k in candidates:
                coefficient = (
                    chi * demands[i] * distance[i][k]
                    + delta * incoming[i] * distance[k][i]
                )
                if coefficient != 0.0:
                    objective += coefficient * x[i, k]

        if alpha != 0.0:
            for i in range(n):
                for k in candidates:
                    for m in candidates:
                        coefficient = alpha * distance[k][m]
                        if coefficient != 0.0:
                            objective += coefficient * q[i, k, m]

        model.setObjective(objective, GRB.MINIMIZE)

        if (
            start_assignment is not None
            and all(k in candidates for k in start_assignment)
        ):
            for i in range(n):
                assigned_hub = start_assignment[i]
                for k in candidates:
                    x[i, k].Start = 1.0 if k == assigned_hub else 0.0

        callback_best = [float(incumbent_value)]
        flat_keys = [
            (i, k)
            for i in range(n)
            for k in candidates
        ]
        flat_vars = [x[i, k] for i, k in flat_keys]
        candidate_count = len(candidates)

        def callback(cb_model, where):
            if where != GRB.Callback.MIPSOL:
                return

            try:
                values = cb_model.cbGetSolution(flat_vars)
                assignment = [0] * n

                for i in range(n):
                    begin = i * candidate_count
                    best_position = max(
                        range(candidate_count),
                        key=lambda p: values[begin + p],
                    )
                    assignment[i] = candidates[best_position]

                solution = solution_builder(assignment)
                if (
                    solution is not None
                    and solution["objective_value"] < callback_best[0] - 1e-7
                ):
                    callback_best[0] = solution["objective_value"]
                    if logger:
                        logger.log_solution(
                            solution["objective_value"],
                            solution,
                        )
            except Exception:
                return

        model.optimize(callback)

        if model.SolCount <= 0:
            return None

        assignment = [
            max(candidates, key=lambda k: x[i, k].X)
            for i in range(n)
        ]

        return assignment

    except gp.GurobiError:
        return None
    except Exception:
        return None


def solve_feasibility_mip(
    n,
    demands,
    capacities,
    required_total_capacity,
    time_limit,
):
    """Last-resort feasibility model used if greedy packing fails."""
    if time_limit <= 0.0:
        return None

    try:
        model = gp.Model("hub_feasibility")
        configure_gurobi(model, time_limit)

        x = model.addVars(n, n, vtype=GRB.BINARY, name="x")

        for i in range(n):
            model.addConstr(
                gp.quicksum(x[i, k] for k in range(n)) == 1
            )

        for i in range(n):
            for k in range(n):
                if i != k:
                    model.addConstr(x[i, k] <= x[k, k])

        for k in range(n):
            model.addConstr(
                gp.quicksum(demands[i] * x[i, k] for i in range(n))
                <= capacities[k] * x[k, k]
            )

        model.addConstr(
            gp.quicksum(capacities[k] * x[k, k] for k in range(n))
            >= required_total_capacity
        )

        # Prefer fewer hubs while searching for feasibility.
        model.setObjective(
            gp.quicksum(x[k, k] for k in range(n)),
            GRB.MINIMIZE,
        )

        model.Params.MIPGap = 0.05
        model.optimize()

        if model.SolCount <= 0:
            return None

        return [
            max(range(n), key=lambda k: x[i, k].X)
            for i in range(n)
        ]

    except gp.GurobiError:
        return None
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    logger = (
        SolutionLogger(args.log_path, sense="minimize")
        if args.log_path
        else None
    )

    start_time = time.monotonic()
    deadline = start_time + max(0, args.time_limit)

    with open(args.instance_path, "r", encoding="utf-8") as file:
        instance = json.load(file)

    n = int(instance["n"])
    flow = [
        [float(value) for value in row]
        for row in instance["flow_matrix"]
    ]
    distance = [
        [float(value) for value in row]
        for row in instance["distance_matrix"]
    ]
    setup_costs = [float(value) for value in instance["setup_costs"]]
    capacities = [float(value) for value in instance["capacities"]]

    if "outgoing_flow" in instance:
        demands = [float(value) for value in instance["outgoing_flow"]]
    else:
        demands = [sum(row) for row in flow]

    incoming = [
        sum(flow[i][j] for i in range(n))
        for j in range(n)
    ]

    required_total_capacity = float(
        instance.get("total_flow_D", sum(demands))
    )

    parameters = instance["cost_parameters"]
    chi = float(parameters["collection_cost_chi"])
    alpha = float(parameters["transfer_cost_alpha"])
    delta = float(parameters["distribution_cost_delta"])

    access_cost = [
        [
            chi * demands[i] * distance[i][k]
            + delta * incoming[i] * distance[k][i]
            for k in range(n)
        ]
        for i in range(n)
    ]

    def solution_builder(assignment):
        assignment = [int(k) for k in assignment]
        if not is_feasible(
            assignment,
            demands,
            capacities,
            required_total_capacity,
        ):
            return None

        return build_solution(
            assignment,
            flow,
            distance,
            setup_costs,
            chi,
            alpha,
            delta,
        )

    incumbent = None
    best_greedy_assignment = None

    for assignment in greedy_feasible_assignments(
        demands,
        capacities,
        access_cost,
        required_total_capacity,
    ):
        solution = solution_builder(assignment)
        old_incumbent = incumbent
        incumbent = update_incumbent(solution, incumbent, logger)
        if incumbent is not old_incumbent:
            best_greedy_assignment = assignment

    # If greedy packing failed, prioritize finding any feasible assignment.
    if incumbent is None:
        remaining = max(0.0, deadline - time.monotonic())
        feasibility_assignment = solve_feasibility_mip(
            n,
            demands,
            capacities,
            required_total_capacity,
            remaining,
        )
        feasibility_solution = (
            solution_builder(feasibility_assignment)
            if feasibility_assignment is not None
            else None
        )
        incumbent = update_incumbent(
            feasibility_solution,
            incumbent,
            logger,
        )
        best_greedy_assignment = feasibility_assignment

    surrogate_assignment = None
    remaining = max(0.0, deadline - time.monotonic())

    if remaining > 0.03:
        if incumbent is None:
            surrogate_time = remaining
        else:
            surrogate_time = min(
                remaining * 0.30,
                max(0.10, min(20.0, remaining - 0.02)),
            )

        surrogate_assignment = solve_assignment_surrogate(
            n=n,
            demands=demands,
            capacities=capacities,
            required_total_capacity=required_total_capacity,
            setup_costs=setup_costs,
            access_cost=access_cost,
            time_limit=surrogate_time,
            start_assignment=best_greedy_assignment,
            solution_builder=solution_builder,
            incumbent_value=(
                incumbent["objective_value"]
                if incumbent is not None
                else math.inf
            ),
            logger=logger,
        )

        if surrogate_assignment is not None:
            surrogate_solution = solution_builder(surrogate_assignment)
            incumbent = update_incumbent(
                surrogate_solution,
                incumbent,
                logger,
            )

    remaining = max(0.0, deadline - time.monotonic())

    if remaining > 0.03:
        # Keep n * |C|^2 continuous variables within a practical range.
        variable_budget = 350000
        budget_count = int(
            math.sqrt(max(1, variable_budget // max(1, n)))
        )
        maximum_candidates = min(
            n,
            max(6, min(35, budget_count)),
        )

        preferred_hubs = []
        if surrogate_assignment is not None:
            preferred_hubs.extend(sorted(set(surrogate_assignment)))
        if incumbent is not None:
            preferred_hubs.extend(incumbent["hubs"])

        # Preserve order while removing duplicates.
        preferred_hubs = list(dict.fromkeys(preferred_hubs))

        candidates = select_candidate_hubs(
            n=n,
            demands=demands,
            capacities=capacities,
            setup_costs=setup_costs,
            access_cost=access_cost,
            preferred_hubs=preferred_hubs,
            maximum_count=maximum_candidates,
        )

        start_assignment = None
        if (
            incumbent is not None
            and all(k in candidates for k in incumbent["assignment"])
        ):
            start_assignment = incumbent["assignment"]
        elif (
            surrogate_assignment is not None
            and all(k in candidates for k in surrogate_assignment)
        ):
            start_assignment = surrogate_assignment

        exact_assignment = solve_exact_restricted(
            candidates=candidates,
            n=n,
            demands=demands,
            incoming=incoming,
            capacities=capacities,
            required_total_capacity=required_total_capacity,
            setup_costs=setup_costs,
            flow=flow,
            distance=distance,
            chi=chi,
            alpha=alpha,
            delta=delta,
            time_limit=remaining,
            start_assignment=start_assignment,
            solution_builder=solution_builder,
            incumbent_value=(
                incumbent["objective_value"]
                if incumbent is not None
                else math.inf
            ),
            logger=logger,
        )

        if exact_assignment is not None:
            exact_solution = solution_builder(exact_assignment)
            incumbent = update_incumbent(
                exact_solution,
                incumbent,
                logger,
            )

    if incumbent is None:
        raise RuntimeError(
            "No feasible solution was found. The instance may be infeasible "
            "or the time limit may be too small to establish feasibility."
        )

    # Recompute from the final discrete assignment to avoid reporting a solver
    # objective with numerical tolerances.
    incumbent = build_solution(
        incumbent["assignment"],
        flow,
        distance,
        setup_costs,
        chi,
        alpha,
        delta,
    )

    output_directory = os.path.dirname(
        os.path.abspath(args.solution_path)
    )
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)

    with open(args.solution_path, "w", encoding="utf-8") as file:
        json.dump(incumbent, file, separators=(",", ":"))


if __name__ == "__main__":
    main()
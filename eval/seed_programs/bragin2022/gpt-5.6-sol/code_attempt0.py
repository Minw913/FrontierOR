import argparse
import json
import math
import os
import random
import time

import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def make_solution(costs, assignment):
    objective = float(sum(costs[assignment[j]][j] for j in range(len(assignment))))
    return {
        "objective_value": objective,
        "assignments": {str(j): int(assignment[j]) for j in range(len(assignment))}
    }


def validate_instance(data):
    m = int(data["num_machines"])
    n = int(data["num_jobs"])
    costs = data["cost_matrix"]
    resources = data["resource_matrix"]
    capacities = data["capacities"]

    if m < 0 or n < 0:
        raise ValueError("num_machines and num_jobs must be nonnegative")
    if len(costs) != m or any(len(row) != n for row in costs):
        raise ValueError("cost_matrix has an invalid shape")
    if len(resources) != m or any(len(row) != n for row in resources):
        raise ValueError("resource_matrix has an invalid shape")
    if len(capacities) != m:
        raise ValueError("capacities has an invalid length")
    if n > 0 and m == 0:
        raise ValueError("No machine is available for nonempty job set")

    return m, n, costs, resources, capacities


def improve_by_relocation(assignment, costs, resources, capacities):
    """Improve a feasible assignment using capacity-feasible single-job moves."""
    n = len(assignment)
    m = len(capacities)
    loads = [0] * m

    for j, i in enumerate(assignment):
        loads[i] += resources[i][j]

    for _ in range(4):
        moves = []
        for j in range(n):
            old = assignment[j]
            for new in range(m):
                if new == old:
                    continue
                saving = costs[old][j] - costs[new][j]
                if saving > 0:
                    moves.append((saving, j, old, new))

        moves.sort(reverse=True)
        changed = False

        for saving, j, old, new in moves:
            if assignment[j] != old:
                continue
            if loads[new] + resources[new][j] <= capacities[new]:
                loads[old] -= resources[old][j]
                loads[new] += resources[new][j]
                assignment[j] = new
                changed = True

        if not changed:
            break

    return assignment


def find_greedy_incumbents(
    m, n, costs, resources, capacities, deadline, logger
):
    """Generate inexpensive feasible starts for the MIP solver."""
    if n == 0:
        solution = {"objective_value": 0.0, "assignments": {}}
        if logger:
            logger.log_solution(0.0, solution)
        return [], solution

    eligible = []
    for j in range(n):
        machines = [
            i for i in range(m)
            if resources[i][j] <= capacities[i]
        ]
        if not machines:
            return None, None
        eligible.append(machines)

    def normalized_size(i, j):
        cap = capacities[i]
        if cap > 0:
            return resources[i][j] / cap
        return 0.0 if resources[i][j] <= 0 else math.inf

    min_size = [
        min(normalized_size(i, j) for i in eligible[j])
        for j in range(n)
    ]

    base_orders = [
        sorted(range(n), key=lambda j: (len(eligible[j]), -min_size[j])),
        sorted(range(n), key=lambda j: -min_size[j]),
        sorted(
            range(n),
            key=lambda j: -(
                sorted(costs[i][j] for i in eligible[j])[1]
                - sorted(costs[i][j] for i in eligible[j])[0]
                if len(eligible[j]) > 1 else float("inf")
            )
        ),
    ]

    rng = random.Random(0)
    alphas = [0.85, 0.65, 0.45, 0.25, 1.0, 0.0]
    best_assignment = None
    best_solution = None
    best_value = math.inf

    heuristic_deadline = min(deadline, time.monotonic() + 1.0)
    trial = 0

    while trial < 64 and time.monotonic() < heuristic_deadline:
        if trial < len(base_orders):
            order = list(base_orders[trial])
        else:
            order = list(base_orders[trial % len(base_orders)])
            # Keep difficult jobs broadly near the front while varying ties/order.
            order.sort(
                key=lambda j: (
                    len(eligible[j]),
                    -min_size[j],
                    rng.random() * (0.25 + 0.02 * trial)
                )
            )

        alpha = alphas[trial % len(alphas)]
        remaining = list(capacities)
        assignment = [-1] * n
        feasible = True

        for j in order:
            choices = [
                i for i in eligible[j]
                if resources[i][j] <= remaining[i]
            ]
            if not choices:
                feasible = False
                break

            job_costs = [costs[i][j] for i in choices]
            cmin = min(job_costs)
            cmax = max(job_costs)
            cspan = max(1.0, float(cmax - cmin))

            def score(i):
                normalized_cost = (costs[i][j] - cmin) / cspan
                if remaining[i] > 0:
                    packing = resources[i][j] / remaining[i]
                else:
                    packing = 0.0 if resources[i][j] <= 0 else math.inf
                noise = rng.random() * 1e-7 if trial >= len(base_orders) else 0.0
                return alpha * normalized_cost + (1.0 - alpha) * packing + noise

            selected = min(choices, key=score)
            assignment[j] = selected
            remaining[selected] -= resources[selected][j]

        if feasible:
            assignment = improve_by_relocation(
                assignment, costs, resources, capacities
            )
            solution = make_solution(costs, assignment)
            value = solution["objective_value"]

            if value < best_value - 1e-9:
                best_value = value
                best_assignment = list(assignment)
                best_solution = solution
                if logger:
                    logger.log_solution(value, solution)

        trial += 1

    return best_assignment, best_solution


def main():
    program_start = time.monotonic()

    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    deadline = program_start + max(0, args.time_limit)

    with open(args.instance_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    m, n, costs, resources, capacities = validate_instance(data)

    if n == 0:
        final_solution = {"objective_value": 0.0, "assignments": {}}
        if logger:
            logger.log_solution(0.0, final_solution)
        output_dir = os.path.dirname(os.path.abspath(args.solution_path))
        os.makedirs(output_dir, exist_ok=True)
        with open(args.solution_path, "w", encoding="utf-8") as f:
            json.dump(final_solution, f)
        return

    best_assignment, best_solution = find_greedy_incumbents(
        m, n, costs, resources, capacities, deadline, logger
    )
    best_value = (
        best_solution["objective_value"]
        if best_solution is not None else math.inf
    )

    model = gp.Model("generalized_assignment")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1

    x = model.addVars(m, n, vtype=GRB.BINARY, name="x")

    model.setObjective(
        gp.quicksum(costs[i][j] * x[i, j] for i in range(m) for j in range(n)),
        GRB.MINIMIZE
    )

    model.addConstrs(
        (gp.quicksum(x[i, j] for i in range(m)) == 1 for j in range(n)),
        name="assign"
    )

    model.addConstrs(
        (
            gp.quicksum(resources[i][j] * x[i, j] for j in range(n))
            <= capacities[i]
            for i in range(m)
        ),
        name="capacity"
    )

    if best_assignment is not None:
        for i in range(m):
            for j in range(n):
                x[i, j].Start = 1.0 if best_assignment[j] == i else 0.0

    incumbent_state = {
        "value": best_value,
        "assignment": best_assignment,
        "solution": best_solution,
    }

    def incumbent_callback(cb_model, where):
        if where != GRB.Callback.MIPSOL:
            return

        callback_obj = cb_model.cbGet(GRB.Callback.MIPSOL_OBJ)
        if callback_obj >= incumbent_state["value"] - 1e-7:
            return

        values = cb_model.cbGetSolution(x)
        assignment = []
        for j in range(n):
            selected = max(range(m), key=lambda i: values[i, j])
            assignment.append(selected)

        solution = make_solution(costs, assignment)
        value = solution["objective_value"]

        if value < incumbent_state["value"] - 1e-7:
            incumbent_state["value"] = value
            incumbent_state["assignment"] = assignment
            incumbent_state["solution"] = solution
            if logger:
                logger.log_solution(value, solution)

    remaining = deadline - time.monotonic()
    if remaining > 0:
        model.Params.TimeLimit = remaining
        try:
            model.optimize(incumbent_callback)
        except gp.GurobiError:
            if incumbent_state["solution"] is None:
                raise

    if model.SolCount > 0:
        assignment = []
        for j in range(n):
            selected = max(range(m), key=lambda i: x[i, j].X)
            assignment.append(selected)

        solution = make_solution(costs, assignment)
        value = solution["objective_value"]

        if value < incumbent_state["value"] - 1e-7:
            incumbent_state["value"] = value
            incumbent_state["assignment"] = assignment
            incumbent_state["solution"] = solution
            if logger:
                logger.log_solution(value, solution)

    final_solution = incumbent_state["solution"]
    if final_solution is None:
        raise RuntimeError(
            "No feasible assignment was found within the time limit, "
            "or the instance is infeasible."
        )

    output_dir = os.path.dirname(os.path.abspath(args.solution_path))
    os.makedirs(output_dir, exist_ok=True)
    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(final_solution, f)


if __name__ == "__main__":
    main()
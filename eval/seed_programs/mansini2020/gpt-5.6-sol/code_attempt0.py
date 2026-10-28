import argparse
import json
import math
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def build_solution(groups, selection):
    selected_items = []
    objective = 0

    for g_idx, item_idx in enumerate(selection):
        group = groups[g_idx]
        item = group["items"][item_idx]
        profit = int(item["profit"])
        objective += profit

        selected_items.append({
            "group_id": int(group["group_id"]),
            "item_id": int(item["item_id"]),
            "profit": profit,
            "weights": [int(w) for w in item["weights"]],
        })

    return {
        "objective_value": float(objective),
        "selected_items": selected_items,
    }


def is_feasible(groups, capacities, selection):
    usage = [0] * len(capacities)

    for g_idx, item_idx in enumerate(selection):
        weights = groups[g_idx]["items"][item_idx]["weights"]
        for r, weight in enumerate(weights):
            usage[r] += int(weight)
            if usage[r] > capacities[r]:
                return False

    return True


def improve_feasible_selection(groups, capacities, selection, deadline):
    """Perform deterministic profit-improving one-group exchanges."""
    m = len(capacities)
    usage = [0] * m

    for g_idx, item_idx in enumerate(selection):
        item = groups[g_idx]["items"][item_idx]
        for r in range(m):
            usage[r] += int(item["weights"][r])

    # A few passes are generally enough for this exchange neighborhood.
    for _ in range(3):
        changed = False

        for g_idx, group in enumerate(groups):
            if time.monotonic() >= deadline:
                return selection

            old_idx = selection[g_idx]
            old_item = group["items"][old_idx]
            old_profit = int(old_item["profit"])

            best_idx = old_idx
            best_profit = old_profit

            for item_idx, item in enumerate(group["items"]):
                profit = int(item["profit"])
                if profit <= best_profit:
                    continue

                feasible = True
                for r in range(m):
                    new_usage = (
                        usage[r]
                        - int(old_item["weights"][r])
                        + int(item["weights"][r])
                    )
                    if new_usage > capacities[r]:
                        feasible = False
                        break

                if feasible:
                    best_idx = item_idx
                    best_profit = profit

            if best_idx != old_idx:
                new_item = group["items"][best_idx]
                for r in range(m):
                    usage[r] += (
                        int(new_item["weights"][r])
                        - int(old_item["weights"][r])
                    )
                selection[g_idx] = best_idx
                changed = True

        if not changed:
            break

    return selection


def greedy_feasible_solution(groups, capacities, deadline):
    """
    Construct a low-consumption selection, then repair aggregate capacity
    violations through exchanges that strictly reduce normalized violation.
    """
    n = len(groups)
    m = len(capacities)

    if n == 0:
        return []

    denominators = [max(1, int(c)) for c in capacities]

    selection = []
    usage = [0] * m

    # Start with the item minimizing normalized aggregate resource usage.
    for group in groups:
        best_idx = min(
            range(len(group["items"])),
            key=lambda idx: (
                sum(
                    int(group["items"][idx]["weights"][r]) / denominators[r]
                    for r in range(m)
                ),
                -int(group["items"][idx]["profit"]),
                idx,
            ),
        )
        selection.append(best_idx)
        item = group["items"][best_idx]
        for r in range(m):
            usage[r] += int(item["weights"][r])

    def violation_value(candidate_usage):
        return sum(
            max(0, candidate_usage[r] - capacities[r]) / denominators[r]
            for r in range(m)
        )

    current_violation = violation_value(usage)
    max_iterations = min(max(1, 3 * n), 10000)

    for _ in range(max_iterations):
        if current_violation <= 1e-12:
            selection = improve_feasible_selection(
                groups, capacities, selection, deadline
            )
            return selection

        if time.monotonic() >= deadline:
            break

        best_move = None
        best_reduction = 0.0
        best_profit_change = -math.inf

        for g_idx, group in enumerate(groups):
            old_idx = selection[g_idx]
            old_item = group["items"][old_idx]
            old_profit = int(old_item["profit"])

            for new_idx, new_item in enumerate(group["items"]):
                if new_idx == old_idx:
                    continue

                new_violation = 0.0
                for r in range(m):
                    candidate_usage = (
                        usage[r]
                        - int(old_item["weights"][r])
                        + int(new_item["weights"][r])
                    )
                    new_violation += (
                        max(0, candidate_usage - capacities[r]) / denominators[r]
                    )

                reduction = current_violation - new_violation
                profit_change = int(new_item["profit"]) - old_profit

                if (
                    reduction > best_reduction + 1e-12
                    or (
                        abs(reduction - best_reduction) <= 1e-12
                        and reduction > 1e-12
                        and profit_change > best_profit_change
                    )
                ):
                    best_reduction = reduction
                    best_profit_change = profit_change
                    best_move = (g_idx, new_idx, new_violation)

        if best_move is None or best_reduction <= 1e-12:
            break

        g_idx, new_idx, new_violation = best_move
        old_idx = selection[g_idx]
        old_item = groups[g_idx]["items"][old_idx]
        new_item = groups[g_idx]["items"][new_idx]

        for r in range(m):
            usage[r] += (
                int(new_item["weights"][r])
                - int(old_item["weights"][r])
            )

        selection[g_idx] = new_idx
        current_violation = new_violation

    if current_violation <= 1e-12 and is_feasible(
        groups, capacities, selection
    ):
        return selection

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    start_time = time.monotonic()
    time_limit = max(0, args.time_limit)

    logger = (
        SolutionLogger(args.log_path, sense="maximize")
        if args.log_path
        else None
    )

    with open(args.instance_path, "r", encoding="utf-8") as f:
        instance = json.load(f)

    capacities = [int(c) for c in instance["capacities"]]
    groups = instance["groups"]
    m = len(capacities)

    for group in groups:
        if not group["items"]:
            raise RuntimeError(
                f"Group {group['group_id']} contains no items, so the instance "
                "has no feasible selection."
            )
        for item in group["items"]:
            if len(item["weights"]) != m:
                raise ValueError(
                    f"Item {item['item_id']} in group {group['group_id']} has "
                    "an invalid weights vector length."
                )

    best_solution = None
    best_objective = -math.inf

    # Spend only a small fraction of the budget on a deterministic warm start.
    heuristic_budget = min(
        2.0,
        max(0.02, 0.05 * time_limit) if time_limit > 0 else 0.0,
    )
    heuristic_deadline = start_time + heuristic_budget

    selection = greedy_feasible_solution(
        groups, capacities, heuristic_deadline
    )

    if selection is not None and is_feasible(groups, capacities, selection):
        best_solution = build_solution(groups, selection)
        best_objective = best_solution["objective_value"]
        if logger:
            logger.log_solution(best_objective, best_solution)

    # Empty instances are solved directly.
    if not groups:
        if best_solution is None:
            best_solution = {
                "objective_value": 0.0,
                "selected_items": [],
            }
            if logger:
                logger.log_solution(0.0, best_solution)

        with open(args.solution_path, "w", encoding="utf-8") as f:
            json.dump(best_solution, f, indent=2)
        return

    model = gp.Model("multiple_choice_multidimensional_knapsack")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1

    variables = []
    records = []

    for g_idx, group in enumerate(groups):
        group_vars = []
        for item_idx, item in enumerate(group["items"]):
            var = model.addVar(
                vtype=GRB.BINARY,
                obj=float(item["profit"]),
                name=f"x_{g_idx}_{item_idx}",
            )
            group_vars.append(var)
            variables.append(var)
            records.append((g_idx, item_idx, item))
        model.addConstr(gp.quicksum(group_vars) == 1, name=f"group_{g_idx}")

    for r, capacity in enumerate(capacities):
        model.addConstr(
            gp.quicksum(
                int(item["weights"][r]) * var
                for var, (_, _, item) in zip(variables, records)
            )
            <= capacity,
            name=f"capacity_{r}",
        )

    model.ModelSense = GRB.MAXIMIZE

    # Supply the heuristic incumbent as a MIP start.
    if selection is not None and is_feasible(groups, capacities, selection):
        for var, (g_idx, item_idx, _) in zip(variables, records):
            var.Start = 1.0 if selection[g_idx] == item_idx else 0.0

    model._variables = variables
    model._records = records
    model._groups = groups
    model._logger = logger
    model._best_solution = best_solution
    model._best_objective = best_objective

    def incumbent_callback(cb_model, where):
        if where != GRB.Callback.MIPSOL:
            return

        values = cb_model.cbGetSolution(cb_model._variables)
        chosen = [-1] * len(cb_model._groups)
        chosen_value = [-1.0] * len(cb_model._groups)

        for value, (g_idx, item_idx, _) in zip(values, cb_model._records):
            if value > chosen_value[g_idx]:
                chosen_value[g_idx] = value
                chosen[g_idx] = item_idx

        if any(item_idx < 0 for item_idx in chosen):
            return

        solution = build_solution(cb_model._groups, chosen)
        objective = solution["objective_value"]

        if objective > cb_model._best_objective + 1e-9:
            cb_model._best_objective = objective
            cb_model._best_solution = solution
            if cb_model._logger:
                cb_model._logger.log_solution(objective, solution)

    elapsed = time.monotonic() - start_time
    remaining = max(0.001, time_limit - elapsed)
    model.Params.TimeLimit = remaining

    try:
        model.optimize(incumbent_callback)
    except gp.GurobiError:
        # Preserve and return any feasible heuristic or callback incumbent.
        if model._best_solution is None:
            raise

    # Extract the final incumbent as a safeguard, including cases where the
    # callback was not invoked for a supplied MIP start.
    if model.SolCount > 0:
        final_selection = [-1] * len(groups)
        final_values = [-1.0] * len(groups)

        for var, (g_idx, item_idx, _) in zip(variables, records):
            value = var.X
            if value > final_values[g_idx]:
                final_values[g_idx] = value
                final_selection[g_idx] = item_idx

        if all(item_idx >= 0 for item_idx in final_selection):
            final_solution = build_solution(groups, final_selection)
            final_objective = final_solution["objective_value"]

            if final_objective > model._best_objective + 1e-9:
                model._best_objective = final_objective
                model._best_solution = final_solution
                if logger:
                    logger.log_solution(final_objective, final_solution)

    best_solution = model._best_solution

    if best_solution is None:
        if model.Status == GRB.INFEASIBLE:
            raise RuntimeError("The instance is infeasible.")
        raise RuntimeError(
            "No feasible solution was found within the time limit."
        )

    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(best_solution, f, indent=2)


if __name__ == "__main__":
    main()
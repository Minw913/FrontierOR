import argparse
import json
import math
import os
import time
from itertools import combinations

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def build_solution(assignments_by_order, order_weights, slab_weights):
    assignments = [
        {"order": int(order), "slab": int(slab)}
        for order, slab in sorted(assignments_by_order.items())
    ]
    slabs_used = sorted({int(slab) for slab in assignments_by_order.values()})

    assigned_weight = math.fsum(order_weights[o] for o in assignments_by_order)
    used_slab_weight = math.fsum(slab_weights[s] for s in slabs_used)
    objective_value = 2.0 * assigned_weight - used_slab_weight

    if abs(objective_value) < 1e-10:
        objective_value = 0.0

    return {
        "objective_value": float(objective_value),
        "assignments": assignments,
        "slabs_used": slabs_used,
    }


def greedy_pack(order_ids, order_weights, capacity):
    """Return a high-weight feasible subset using several deterministic orders."""
    feasible = [o for o in order_ids if order_weights[o] <= capacity + 1e-9]
    if not feasible:
        return [], 0.0

    candidate_orders = [
        sorted(feasible, key=lambda o: (-order_weights[o], o)),
        sorted(feasible, key=lambda o: (order_weights[o], o)),
    ]

    # A closeness-first ordering can improve packing when one large order almost
    # fills the slab.
    candidate_orders.append(
        sorted(feasible, key=lambda o: (abs(capacity - order_weights[o]), -order_weights[o], o))
    )

    best_orders = []
    best_weight = 0.0

    for ordered in candidate_orders:
        chosen = []
        total = 0.0
        for o in ordered:
            w = order_weights[o]
            if total + w <= capacity + 1e-9:
                chosen.append(o)
                total += w

        if total > best_weight + 1e-9:
            best_orders = chosen
            best_weight = total

    return best_orders, best_weight


def best_greedy_load_for_slab(
    slab,
    available_orders,
    orders_by_slab_color,
    order_weights,
    slab_weights,
    max_colors,
    deadline,
):
    if max_colors <= 0 or time.monotonic() >= deadline:
        return [], 0.0

    color_map = orders_by_slab_color.get(slab, {})
    available_by_color = {}

    for color, orders in color_map.items():
        filtered = [o for o in orders if o in available_orders]
        if filtered:
            available_by_color[color] = filtered

    if not available_by_color:
        return [], 0.0

    ranked_colors = sorted(
        available_by_color,
        key=lambda c: (
            -sum(order_weights[o] for o in available_by_color[c]),
            c,
        ),
    )

    # Limit pair enumeration so the startup heuristic remains inexpensive.
    pair_colors = ranked_colors[:14]
    singleton_colors = ranked_colors[:24]

    color_sets = [(c,) for c in singleton_colors]

    if max_colors >= 2:
        color_sets.extend(combinations(pair_colors, 2))

    if max_colors > 2:
        # Include several multi-color prefixes without combinatorial explosion.
        limit = min(max_colors, len(ranked_colors))
        for size in range(3, limit + 1):
            color_sets.append(tuple(ranked_colors[:size]))

    best_orders = []
    best_weight = 0.0
    capacity = slab_weights[slab]

    for color_set in color_sets:
        if time.monotonic() >= deadline:
            break

        pool = []
        for color in color_set:
            pool.extend(available_by_color[color])

        selected, load = greedy_pack(pool, order_weights, capacity)
        if load > best_weight + 1e-9:
            best_orders = selected
            best_weight = load

    return best_orders, best_weight


def construct_greedy_solution(
    num_orders,
    active_slabs,
    orders_by_slab_color,
    order_weights,
    slab_weights,
    max_colors,
    deadline,
):
    available_orders = set(range(num_orders))
    assignments = {}

    # Process slabs with the largest optimistic standalone contribution first.
    slab_priority = []
    for slab in active_slabs:
        eligible_weight = 0.0
        for orders in orders_by_slab_color.get(slab, {}).values():
            eligible_weight += sum(order_weights[o] for o in orders)
        optimistic_load = min(slab_weights[slab], eligible_weight)
        optimistic_gain = 2.0 * optimistic_load - slab_weights[slab]
        slab_priority.append((optimistic_gain, optimistic_load, -slab))

    slab_priority.sort(reverse=True)

    for _, _, neg_slab in slab_priority:
        if time.monotonic() >= deadline:
            break

        slab = -neg_slab
        selected, load = best_greedy_load_for_slab(
            slab,
            available_orders,
            orders_by_slab_color,
            order_weights,
            slab_weights,
            max_colors,
            deadline,
        )

        # Each newly opened slab must make a strictly positive contribution.
        if selected and 2.0 * load - slab_weights[slab] > 1e-9:
            for order in selected:
                assignments[order] = slab
                available_orders.remove(order)

    return assignments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    start_time = time.monotonic()
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as f:
        instance = json.load(f)

    num_slabs = int(instance["num_slabs"])
    num_orders = int(instance["num_orders"])
    max_colors = int(instance["max_colors_per_knapsack"])

    slab_weights = [float(v) for v in instance["slab_weights"]]
    order_weights = [float(v) for v in instance["order_weights"]]
    order_colors = [int(v) for v in instance["order_colors"]]
    raw_eligibility = instance["eligible_slabs_per_order"]

    if len(slab_weights) != num_slabs:
        raise ValueError("slab_weights length does not match num_slabs")
    if len(order_weights) != num_orders:
        raise ValueError("order_weights length does not match num_orders")
    if len(order_colors) != num_orders:
        raise ValueError("order_colors length does not match num_orders")
    if len(raw_eligibility) != num_orders:
        raise ValueError("eligible_slabs_per_order length does not match num_orders")
    if any(w < 0 for w in slab_weights) or any(w < 0 for w in order_weights):
        raise ValueError("Slab and order weights must be nonnegative")

    # Eliminate duplicate, invalid, and capacity-infeasible assignment arcs.
    eligible_slabs_per_order = [[] for _ in range(num_orders)]
    orders_by_slab = [[] for _ in range(num_slabs)]
    orders_by_slab_color = {}

    for order in range(num_orders):
        seen = set()
        for raw_slab in raw_eligibility[order]:
            slab = int(raw_slab)
            if slab in seen:
                continue
            seen.add(slab)

            if not 0 <= slab < num_slabs:
                continue
            if order_weights[order] > slab_weights[slab] + 1e-9:
                continue

            eligible_slabs_per_order[order].append(slab)
            orders_by_slab[slab].append(order)
            orders_by_slab_color.setdefault(slab, {}).setdefault(
                order_colors[order], []
            ).append(order)

    active_slabs = [s for s in range(num_slabs) if orders_by_slab[s]]

    empty_solution = build_solution({}, order_weights, slab_weights)
    best_logged_value = empty_solution["objective_value"]
    if logger:
        logger.log_solution(best_logged_value, empty_solution)

    total_limit = max(0, args.time_limit)
    heuristic_budget = min(1.0, 0.10 * total_limit)
    heuristic_deadline = min(
        start_time + total_limit,
        time.monotonic() + max(0.0, heuristic_budget),
    )

    greedy_assignments = {}
    if total_limit > 0 and max_colors > 0:
        greedy_assignments = construct_greedy_solution(
            num_orders,
            active_slabs,
            orders_by_slab_color,
            order_weights,
            slab_weights,
            max_colors,
            heuristic_deadline,
        )

    greedy_solution = build_solution(
        greedy_assignments, order_weights, slab_weights
    )
    if greedy_solution["objective_value"] > best_logged_value + 1e-8:
        best_logged_value = greedy_solution["objective_value"]
        if logger:
            logger.log_solution(best_logged_value, greedy_solution)

    model = gp.Model("surplus_slab_assignment")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1

    remaining_time = max(0.0, start_time + total_limit - time.monotonic())
    model.Params.TimeLimit = remaining_time

    x = {}
    y = {}
    z = {}

    for slab in active_slabs:
        y[slab] = model.addVar(vtype=GRB.BINARY, name=f"y_{slab}")
        for color in orders_by_slab_color[slab]:
            z[slab, color] = model.addVar(
                vtype=GRB.BINARY, name=f"z_{slab}_{color}"
            )

    for order in range(num_orders):
        for slab in eligible_slabs_per_order[order]:
            x[order, slab] = model.addVar(
                vtype=GRB.BINARY, name=f"x_{order}_{slab}"
            )

    model.update()

    # Each order is assigned to at most one slab.
    for order in range(num_orders):
        arcs = [x[order, slab] for slab in eligible_slabs_per_order[order]]
        if len(arcs) > 1:
            model.addConstr(gp.quicksum(arcs) <= 1)
        elif len(arcs) == 1:
            # The binary bound already enforces this constraint.
            pass

    for slab in active_slabs:
        slab_vars = [x[order, slab] for order in orders_by_slab[slab]]

        model.addConstr(
            gp.quicksum(
                order_weights[order] * x[order, slab]
                for order in orders_by_slab[slab]
            )
            <= slab_weights[slab] * y[slab]
        )

        # Strengthening constraints: an assignment directly forces its slab open.
        for order in orders_by_slab[slab]:
            model.addConstr(x[order, slab] <= y[slab])
            model.addConstr(
                x[order, slab] <= z[slab, order_colors[order]]
            )

        model.addConstr(
            gp.quicksum(
                z[slab, color] for color in orders_by_slab_color[slab]
            )
            <= max_colors * y[slab]
        )

        # Do not allow an open slab with no assigned order.
        model.addConstr(y[slab] <= gp.quicksum(slab_vars))

    objective = gp.quicksum(
        2.0 * order_weights[order] * var
        for (order, _), var in x.items()
    ) - gp.quicksum(
        slab_weights[slab] * y[slab] for slab in active_slabs
    )
    model.setObjective(objective, GRB.MAXIMIZE)

    # Supply a complete MIP start from the startup heuristic.
    used_in_greedy = set(greedy_assignments.values())
    greedy_colors = {}
    for order, slab in greedy_assignments.items():
        greedy_colors.setdefault(slab, set()).add(order_colors[order])

    for (order, slab), var in x.items():
        var.Start = 1.0 if greedy_assignments.get(order) == slab else 0.0
    for slab, var in y.items():
        var.Start = 1.0 if slab in used_in_greedy else 0.0
    for (slab, color), var in z.items():
        var.Start = 1.0 if color in greedy_colors.get(slab, set()) else 0.0

    x_items = list(x.items())
    x_vars = [var for _, var in x_items]

    model._x_items = x_items
    model._x_vars = x_vars
    model._order_weights = order_weights
    model._slab_weights = slab_weights
    model._logger = logger
    model._best_logged_value = best_logged_value
    model._logging_disabled = False

    def incumbent_callback(cb_model, where):
        if where != GRB.Callback.MIPSOL:
            return
        if cb_model._logger is None or cb_model._logging_disabled:
            return

        try:
            values = cb_model.cbGetSolution(cb_model._x_vars)
            incumbent_assignments = {}

            for ((order, slab), _), value in zip(cb_model._x_items, values):
                if value > 0.5:
                    incumbent_assignments[order] = slab

            solution = build_solution(
                incumbent_assignments,
                cb_model._order_weights,
                cb_model._slab_weights,
            )
            value = solution["objective_value"]

            if value > cb_model._best_logged_value + 1e-8:
                cb_model._best_logged_value = value
                cb_model._logger.log_solution(value, solution)
        except Exception:
            # Logging must not abort the optimization if the log destination
            # becomes unavailable during a callback.
            cb_model._logging_disabled = True

    if remaining_time > 0.0:
        try:
            model.optimize(incumbent_callback if logger else None)
        except gp.GurobiError:
            # The validated greedy solution remains available as a fallback.
            pass

    final_solution = greedy_solution

    if model.SolCount > 0:
        final_assignments = {}
        for (order, slab), var in x.items():
            if var.X > 0.5:
                final_assignments[order] = slab

        candidate_solution = build_solution(
            final_assignments, order_weights, slab_weights
        )
        if (
            candidate_solution["objective_value"]
            >= final_solution["objective_value"] - 1e-8
        ):
            final_solution = candidate_solution

    if (
        logger
        and not model._logging_disabled
        and final_solution["objective_value"] > model._best_logged_value + 1e-8
    ):
        logger.log_solution(
            final_solution["objective_value"], final_solution
        )

    solution_dir = os.path.dirname(os.path.abspath(args.solution_path))
    if solution_dir:
        os.makedirs(solution_dir, exist_ok=True)

    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(final_solution, f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
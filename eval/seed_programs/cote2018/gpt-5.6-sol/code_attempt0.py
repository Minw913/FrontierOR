import argparse
import bisect
import json
import math
import os
import time
from collections import defaultdict
from functools import reduce

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


LOSS_LABEL = 0


def variable_name(u, v, label):
    return f"x[{u},{v},{label}]"


def write_solution(path, solution):
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    temporary_path = path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(solution, handle, separators=(",", ":"))
    os.replace(temporary_path, path)


def make_solution(objective, model_variables):
    return {
        "objective_value": int(objective),
        "model_variables": {
            key: int(value)
            for key, value in model_variables.items()
            if int(value) != 0
        },
    }


def expand_scaled_flow(flow, scale, scaled_capacity):
    """Convert a flow on gcd-scaled capacity states to original states."""
    original_capacity = scaled_capacity * scale
    result = defaultdict(int)

    for (u, v, label), value in flow.items():
        value = int(value)
        if value == 0:
            continue

        if u == scaled_capacity and v == 0 and label == LOSS_LABEL:
            result[variable_name(original_capacity, 0, LOSS_LABEL)] += value
        elif label == LOSS_LABEL:
            start = u * scale
            end = v * scale
            for position in range(start, end):
                result[variable_name(position, position + 1, LOSS_LABEL)] += value
        else:
            result[variable_name(u * scale, v * scale, label)] += value

    return dict(result)


def add_pattern_to_flow(flow, labels, widths, capacity, multiplicity=1):
    position = 0
    for label in labels:
        next_position = position + widths[label - 1]
        flow[(position, next_position, label)] += multiplicity
        position = next_position

    while position < capacity:
        flow[(position, position + 1, LOSS_LABEL)] += multiplicity
        position += 1

    flow[(capacity, 0, LOSS_LABEL)] += multiplicity


def best_fit_decreasing(widths, demands, capacity):
    """
    Best-fit decreasing heuristic for moderate item counts.

    Returns an aggregated arc-flow witness on the supplied capacity scale.
    """
    flow = defaultdict(int)
    total_items = sum(demands)

    if total_items == 0:
        return 0, flow

    # Entries are (remaining capacity, unique bin id).
    open_bins = []
    bin_items = []
    next_bin_id = 0

    order = sorted(range(len(widths)), key=lambda i: (-widths[i], i))

    for item_index in order:
        width = widths[item_index]
        label = item_index + 1

        for _ in range(demands[item_index]):
            location = bisect.bisect_left(open_bins, (width, -1))

            if location == len(open_bins):
                bin_id = next_bin_id
                next_bin_id += 1
                bin_items.append([label])
                remaining = capacity - width
            else:
                remaining, bin_id = open_bins.pop(location)
                bin_items[bin_id].append(label)
                remaining -= width

            bisect.insort(open_bins, (remaining, bin_id))

    for labels in bin_items:
        # A deterministic order makes the resulting path easy to validate.
        labels.sort()
        add_pattern_to_flow(flow, labels, widths, capacity)

    return len(bin_items), flow


def bulk_decreasing_heuristic(widths, demands, capacity):
    """
    Count-based decreasing heuristic for instances with very large demands.
    Repeated identical bins are added to the flow in one operation.
    """
    remaining = list(demands)
    order = sorted(range(len(widths)), key=lambda i: (-widths[i], i))
    flow = defaultdict(int)
    objective = 0

    while any(value > 0 for value in remaining):
        free_space = capacity
        quantities = [0] * len(widths)

        for item_index in order:
            if remaining[item_index] <= 0:
                continue
            count = min(remaining[item_index], free_space // widths[item_index])
            if count:
                quantities[item_index] = count
                free_space -= count * widths[item_index]

        if not any(quantities):
            raise ValueError("The instance contains an item wider than the bin capacity")

        # Repeat the current pattern as many times as is certainly valid.
        multiplicity = min(
            remaining[i] // quantities[i]
            for i in range(len(widths))
            if quantities[i] > 0
        )
        multiplicity = max(1, multiplicity)

        labels = []
        for item_index in order:
            count = quantities[item_index]
            if count:
                labels.extend([item_index + 1] * count)
                remaining[item_index] -= count * multiplicity

        labels.sort()
        add_pattern_to_flow(
            flow, labels, widths, capacity, multiplicity=multiplicity
        )
        objective += multiplicity

    return objective, flow


def initial_solution(widths, demands, capacity):
    total_items = sum(demands)
    if total_items <= 250_000:
        return best_fit_decreasing(widths, demands, capacity)
    return bulk_decreasing_heuristic(widths, demands, capacity)


def flow_to_solution(flow, objective, scale, scaled_capacity):
    variables = expand_scaled_flow(flow, scale, scaled_capacity)
    return make_solution(objective, variables)


def lower_bound(widths, demands, capacity):
    total_width = sum(w * d for w, d in zip(widths, demands))
    bound = (total_width + capacity - 1) // capacity if capacity else 0

    for width, demand in zip(widths, demands):
        copies_per_bin = capacity // width
        if copies_per_bin <= 0 and demand > 0:
            raise ValueError("The instance contains an item wider than the bin capacity")
        if demand > 0:
            bound = max(bound, (demand + copies_per_bin - 1) // copies_per_bin)

    return bound


def build_and_solve_arc_flow(
    widths,
    demands,
    capacity,
    scale,
    initial_objective,
    initial_flow,
    deadline,
    logger,
):
    model = gp.Model("cutting_stock_arc_flow")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = max(0.01, deadline - time.monotonic())

    outgoing = [gp.LinExpr() for _ in range(capacity + 1)]
    incoming = [gp.LinExpr() for _ in range(capacity + 1)]
    edges = []
    variables = []

    def add_arc(u, v, label):
        var = model.addVar(
            lb=0.0,
            vtype=GRB.INTEGER,
            name=variable_name(u, v, label),
        )
        edges.append((u, v, label))
        variables.append(var)
        outgoing[u] += var
        incoming[v] += var
        return var

    # Unit loss arcs ensure that every path can be completed to the sink.
    for u in range(capacity):
        add_arc(u, u + 1, LOSS_LABEL)

    item_arc_indices = [[] for _ in widths]
    for item_index, width in enumerate(widths):
        label = item_index + 1
        for u in range(capacity - width + 1):
            index = len(variables)
            add_arc(u, u + width, label)
            item_arc_indices[item_index].append(index)

    feedback_index = len(variables)
    feedback = add_arc(capacity, 0, LOSS_LABEL)

    model.update()

    for node in range(capacity + 1):
        model.addConstr(
            outgoing[node] == incoming[node],
            name=f"flow[{node}]",
        )

    for item_index, demand in enumerate(demands):
        expression = gp.LinExpr()
        for arc_index in item_arc_indices[item_index]:
            expression += variables[arc_index]
        # Exact production is without loss of optimality and tightens the model.
        model.addConstr(
            expression == demand,
            name=f"demand[{item_index + 1}]",
        )

    model.setObjective(feedback, GRB.MINIMIZE)
    model.addConstr(feedback <= initial_objective, name="incumbent_upper_bound")

    for edge, var in zip(edges, variables):
        var.Start = float(initial_flow.get(edge, 0))

    best_logged = [initial_objective]

    def callback(cb_model, where):
        if where != GRB.Callback.MIPSOL:
            return

        objective = int(round(cb_model.cbGet(GRB.Callback.MIPSOL_OBJ)))
        if objective >= best_logged[0]:
            return

        values = cb_model.cbGetSolution(variables)
        candidate_flow = defaultdict(int)
        for edge, value in zip(edges, values):
            rounded = int(round(value))
            if rounded > 0:
                candidate_flow[edge] = rounded

        candidate = flow_to_solution(
            candidate_flow, objective, scale, capacity
        )
        best_logged[0] = objective
        if logger:
            logger.log_solution(objective, candidate)

    model.optimize(callback)

    if model.SolCount <= 0:
        return initial_objective, initial_flow

    objective = int(round(model.ObjVal))
    if objective >= initial_objective:
        return initial_objective, initial_flow

    values = model.getAttr("X", variables)
    best_flow = defaultdict(int)
    for edge, value in zip(edges, values):
        rounded = int(round(value))
        if rounded > 0:
            best_flow[edge] = rounded

    return objective, best_flow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    start_time = time.monotonic()
    deadline = start_time + max(0, args.time_limit)

    with open(args.instance_path, "r", encoding="utf-8") as handle:
        instance = json.load(handle)

    original_capacity = int(instance["bin_capacity"])
    items = list(instance["items"])

    if original_capacity <= 0:
        raise ValueError("bin_capacity must be positive")

    original_widths = [int(item["width"]) for item in items]
    demands = [int(item["demand"]) for item in items]

    if any(width <= 0 for width in original_widths):
        raise ValueError("All item widths must be positive")
    if any(demand < 0 for demand in demands):
        raise ValueError("All demands must be nonnegative")
    if any(width > original_capacity and demand > 0
           for width, demand in zip(original_widths, demands)):
        raise ValueError("The instance is infeasible: an item exceeds bin capacity")

    if not items or sum(demands) == 0:
        solution = make_solution(0, {})
        if logger:
            logger.log_solution(0, solution)
        write_solution(args.solution_path, solution)
        return

    divisor = reduce(math.gcd, [original_capacity] + original_widths)
    scaled_capacity = original_capacity // divisor
    scaled_widths = [width // divisor for width in original_widths]

    incumbent_objective, incumbent_flow = initial_solution(
        scaled_widths, demands, scaled_capacity
    )
    incumbent_solution = flow_to_solution(
        incumbent_flow,
        incumbent_objective,
        divisor,
        scaled_capacity,
    )

    if logger:
        logger.log_solution(incumbent_objective, incumbent_solution)

    write_solution(args.solution_path, incumbent_solution)

    bound = lower_bound(scaled_widths, demands, scaled_capacity)
    remaining_time = deadline - time.monotonic()

    # Avoid constructing an excessively large state-space model when it is
    # unlikely to fit comfortably in memory or when virtually no time remains.
    estimated_item_arcs = sum(
        max(0, scaled_capacity - width + 1) for width in scaled_widths
    )
    estimated_arcs = estimated_item_arcs + scaled_capacity + 1

    if (
        incumbent_objective > bound
        and remaining_time > 0.25
        and estimated_arcs <= 2_000_000
    ):
        best_objective, best_flow = build_and_solve_arc_flow(
            scaled_widths,
            demands,
            scaled_capacity,
            divisor,
            incumbent_objective,
            incumbent_flow,
            deadline,
            logger,
        )

        if best_objective < incumbent_objective:
            incumbent_objective = best_objective
            incumbent_flow = best_flow
            incumbent_solution = flow_to_solution(
                incumbent_flow,
                incumbent_objective,
                divisor,
                scaled_capacity,
            )

    write_solution(args.solution_path, incumbent_solution)


if __name__ == "__main__":
    main()
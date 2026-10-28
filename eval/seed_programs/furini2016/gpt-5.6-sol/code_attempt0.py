import argparse
import json
import math
import os
import time

import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def make_solution(items, counts):
    selected = []
    objective = 0
    for item, count in zip(items, counts):
        count = int(count)
        if count > 0:
            selected.append({
                "item_id": int(item["id"]),
                "plate_dims": [int(item["length"]), int(item["width"])],
                "copies": count
            })
            objective += count * int(item["profit"])

    return {
        "objective_value": float(objective),
        "items_selected": selected
    }


def solution_value(items, counts):
    return sum(int(item["profit"]) * int(count)
               for item, count in zip(items, counts))


def greedy_strip_solution(panel_l, panel_w, items, horizontal, key_function):
    """
    Constructs a valid staged guillotine pattern.

    Horizontal mode cuts the panel into horizontal strips. Each strip contains
    copies of one type placed from left to right. Vertical mode is symmetric.
    """
    n = len(items)
    counts = [0] * n
    remaining_primary = panel_w if horizontal else panel_l

    feasible = []
    for i, item in enumerate(items):
        l = int(item["length"])
        w = int(item["width"])
        copies = int(item["copies"])
        if copies <= 0 or l > panel_l or w > panel_w:
            continue

        per_strip = panel_l // l if horizontal else panel_w // w
        strip_size = w if horizontal else l
        if per_strip > 0 and strip_size <= remaining_primary:
            feasible.append(i)

    feasible.sort(key=lambda i: key_function(items[i], horizontal), reverse=True)

    for i in feasible:
        item = items[i]
        l = int(item["length"])
        w = int(item["width"])
        available = int(item["copies"])

        per_strip = panel_l // l if horizontal else panel_w // w
        strip_size = w if horizontal else l
        max_strips = remaining_primary // strip_size
        if per_strip <= 0 or max_strips <= 0:
            continue

        take = min(available, max_strips * per_strip)
        if take <= 0:
            continue

        strips_used = (take + per_strip - 1) // per_strip
        counts[i] = take
        remaining_primary -= strips_used * strip_size

    return counts


def initial_heuristics(panel_l, panel_w, items, report):
    n = len(items)
    best_counts = [0] * n
    best_value = 0

    # Homogeneous rectangular grids.
    for i, item in enumerate(items):
        l = int(item["length"])
        w = int(item["width"])
        if l <= panel_l and w <= panel_w:
            count = min(
                int(item["copies"]),
                (panel_l // l) * (panel_w // w)
            )
            value = count * int(item["profit"])
            if value > best_value:
                best_value = value
                best_counts = [0] * n
                best_counts[i] = count
                report(best_counts)

    def density(item, horizontal):
        area = int(item["length"]) * int(item["width"])
        return int(item["profit"]) / max(1, area)

    def strip_efficiency(item, horizontal):
        l = int(item["length"])
        w = int(item["width"])
        p = int(item["profit"])
        copies = int(item["copies"])
        if horizontal:
            q = panel_l // l
            strip_size = w
        else:
            q = panel_w // w
            strip_size = l
        return min(copies, q) * p / max(1, strip_size)

    def profit_key(item, horizontal):
        return int(item["profit"])

    def potential_key(item, horizontal):
        return int(item["profit"]) * int(item["copies"])

    key_functions = [density, strip_efficiency, profit_key, potential_key]

    for horizontal in (True, False):
        for key_function in key_functions:
            counts = greedy_strip_solution(
                panel_l, panel_w, items, horizontal, key_function
            )
            value = solution_value(items, counts)
            if value > best_value:
                best_value = value
                best_counts = counts
                report(best_counts)

    return best_counts


def bounded_subset_sums(limit, items, dimension_name):
    if limit <= 0:
        return []
    if limit > 250000:
        return []

    bits = 1
    mask = (1 << (limit + 1)) - 1

    for item in items:
        size = int(item[dimension_name])
        if size <= 0 or size > limit:
            continue

        copies = min(int(item["copies"]), limit // size)
        chunk = 1
        while copies > 0:
            use = min(chunk, copies)
            bits |= bits << (size * use)
            bits &= mask
            copies -= use
            chunk <<= 1

    return [v for v in range(1, limit + 1) if (bits >> v) & 1]


def choose_core_items(items, cap):
    if len(items) <= cap:
        return list(range(len(items)))

    rankings = [
        sorted(
            range(len(items)),
            key=lambda i: (
                int(items[i]["profit"]) /
                max(1, int(items[i]["length"]) * int(items[i]["width"]))
            ),
            reverse=True
        ),
        sorted(
            range(len(items)),
            key=lambda i: int(items[i]["profit"]),
            reverse=True
        ),
        sorted(
            range(len(items)),
            key=lambda i: int(items[i]["profit"]) * int(items[i]["copies"]),
            reverse=True
        ),
        sorted(
            range(len(items)),
            key=lambda i: int(items[i]["length"]) * int(items[i]["width"])
        )
    ]

    selected = []
    selected_set = set()
    position = 0
    while len(selected) < cap:
        added = False
        for ranking in rankings:
            if position < len(ranking):
                i = ranking[position]
                if i not in selected_set:
                    selected_set.add(i)
                    selected.append(i)
                    added = True
                    if len(selected) >= cap:
                        break
        if position >= len(items) and not added:
            break
        position += 1

    return selected


def select_axis_values(limit, items, dimension_name, core_indices, cap):
    mandatory = {limit}
    for i in core_indices:
        value = int(items[i][dimension_name])
        if 0 < value <= limit:
            mandatory.add(value)

    # Useful multiples of promising item dimensions.
    extras = set()
    for i in core_indices[:40]:
        size = int(items[i][dimension_name])
        if size <= 0:
            continue
        max_k = min(int(items[i]["copies"]), limit // size, 20)
        for k in range(1, max_k + 1):
            extras.add(k * size)

    sums = bounded_subset_sums(limit, items, dimension_name)
    pool = sorted((set(sums) | extras | mandatory) - {0})

    if len(mandatory) >= cap:
        retained = sorted(mandatory)
        if len(retained) > cap:
            # Always retain the original panel dimension.
            others = [v for v in retained if v != limit]
            step = max(1.0, len(others) / max(1, cap - 1))
            sampled = {
                others[min(len(others) - 1, int(k * step))]
                for k in range(cap - 1)
            } if others else set()
            retained = sorted(sampled | {limit})
        return retained

    retained = set(mandatory)
    candidates = [v for v in pool if v not in retained]
    slots = cap - len(retained)

    if len(candidates) <= slots:
        retained.update(candidates)
    elif slots > 0:
        # Quantile sampling preserves both small and large attainable sizes.
        for k in range(slots):
            pos = round(k * (len(candidates) - 1) / max(1, slots - 1))
            retained.add(candidates[pos])

    return sorted(retained)


def solve_with_plate_flow(panel_l, panel_w, items, deadline,
                          incumbent_counts, report):
    remaining = deadline - time.monotonic()
    if remaining <= 1.0:
        return incumbent_counts

    axis_cap = 105
    core_indices = choose_core_items(items, axis_cap - 1)

    x_values = select_axis_values(
        panel_l, items, "length", core_indices, axis_cap
    )
    y_values = select_axis_values(
        panel_w, items, "width", core_indices, axis_cap
    )

    x_set = set(x_values)
    y_set = set(y_values)

    represented_items = [
        i for i, item in enumerate(items)
        if int(item["length"]) in x_set
        and int(item["width"]) in y_set
        and int(item["length"]) <= panel_l
        and int(item["width"]) <= panel_w
        and int(item["copies"]) > 0
    ]
    if not represented_items:
        return incumbent_counts

    states = [(x, y) for x in x_values for y in y_values]
    state_index = {state: k for k, state in enumerate(states)}

    root = (panel_l, panel_w)
    if root not in state_index:
        states.append(root)
        state_index[root] = len(states) - 1

    decompositions_x = {}
    for x in x_values:
        pairs = []
        for left in x_values:
            if left > x // 2:
                break
            right = x - left
            if right in x_set:
                pairs.append((left, right))
        decompositions_x[x] = pairs

    decompositions_y = {}
    for y in y_values:
        pairs = []
        for lower in y_values:
            if lower > y // 2:
                break
            upper = y - lower
            if upper in y_set:
                pairs.append((lower, upper))
        decompositions_y[y] = pairs

    try:
        model = gp.Model("bounded_guillotine_plate_flow")
        model.Params.OutputFlag = 0
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1

        max_pieces = max(
            1,
            sum(min(
                int(item["copies"]),
                (panel_l // int(item["length"])) *
                (panel_w // int(item["width"]))
            ) for item in items
                if int(item["length"]) <= panel_l
                and int(item["width"]) <= panel_w)
        )
        variable_ub = max(1, min(max_pieces * 2 + 1, 2_000_000_000))

        balances = [gp.LinExpr() for _ in states]
        balances[state_index[root]].addConstant(1.0)

        extraction_vars = []
        extraction_item_indices = []

        items_at_state = {}
        for i in represented_items:
            item = items[i]
            state = (int(item["length"]), int(item["width"]))
            items_at_state.setdefault(state, []).append(i)

        for state, item_indices in items_at_state.items():
            s = state_index[state]
            for i in item_indices:
                item = items[i]
                var = model.addVar(
                    vtype=GRB.INTEGER,
                    lb=0,
                    ub=int(item["copies"]),
                    obj=int(item["profit"]),
                    name=f"item_{i}"
                )
                balances[s].addTerms(-1.0, var)
                extraction_vars.append(var)
                extraction_item_indices.append(i)

        # Trimming actions to the immediately smaller modeled dimension.
        for xi, x in enumerate(x_values):
            for yi, y in enumerate(y_values):
                parent_idx = state_index[(x, y)]

                if xi > 0:
                    child = (x_values[xi - 1], y)
                    var = model.addVar(
                        vtype=GRB.INTEGER, lb=0, ub=variable_ub
                    )
                    balances[parent_idx].addTerms(-1.0, var)
                    balances[state_index[child]].addTerms(1.0, var)

                if yi > 0:
                    child = (x, y_values[yi - 1])
                    var = model.addVar(
                        vtype=GRB.INTEGER, lb=0, ub=variable_ub
                    )
                    balances[parent_idx].addTerms(-1.0, var)
                    balances[state_index[child]].addTerms(1.0, var)

        # Guillotine cuts. A cut consumes one parent and creates two children.
        for x in x_values:
            for y in y_values:
                parent_idx = state_index[(x, y)]

                for left, right in decompositions_x[x]:
                    var = model.addVar(
                        vtype=GRB.INTEGER, lb=0, ub=variable_ub
                    )
                    balances[parent_idx].addTerms(-1.0, var)
                    balances[state_index[(left, y)]].addTerms(1.0, var)
                    balances[state_index[(right, y)]].addTerms(1.0, var)

                for lower, upper in decompositions_y[y]:
                    var = model.addVar(
                        vtype=GRB.INTEGER, lb=0, ub=variable_ub
                    )
                    balances[parent_idx].addTerms(-1.0, var)
                    balances[state_index[(x, lower)]].addTerms(1.0, var)
                    balances[state_index[(x, upper)]].addTerms(1.0, var)

        for expression in balances:
            model.addConstr(expression >= 0.0)

        # This is redundant for distinct extraction variables, but explicitly
        # enforces each type's availability and remains valid if formulations
        # are later extended to several extraction states.
        variables_by_item = {}
        for var, i in zip(extraction_vars, extraction_item_indices):
            variables_by_item.setdefault(i, []).append(var)
        for i, variables in variables_by_item.items():
            model.addConstr(
                gp.quicksum(variables) <= int(items[i]["copies"])
            )

        model.ModelSense = GRB.MAXIMIZE
        model.update()

        remaining = deadline - time.monotonic()
        if remaining <= 0.05:
            return incumbent_counts
        model.Params.TimeLimit = max(0.05, remaining)

        callback_best = [solution_value(items, incumbent_counts)]

        def callback(cb_model, where):
            if where != GRB.Callback.MIPSOL:
                return
            try:
                values = cb_model.cbGetSolution(extraction_vars)
                counts = [0] * len(items)
                for value, item_index in zip(values, extraction_item_indices):
                    count = max(0, int(round(value)))
                    count = min(count, int(items[item_index]["copies"]))
                    counts[item_index] += count

                value = solution_value(items, counts)
                if value > callback_best[0]:
                    callback_best[0] = value
                    report(counts)
            except Exception:
                pass

        model.optimize(callback)

        best_counts = list(incumbent_counts)
        best_value = solution_value(items, best_counts)

        if model.SolCount > 0:
            counts = [0] * len(items)
            for var, item_index in zip(extraction_vars, extraction_item_indices):
                count = max(0, int(round(var.X)))
                counts[item_index] += count

            for i in range(len(items)):
                counts[i] = min(counts[i], int(items[i]["copies"]))

            value = solution_value(items, counts)
            if value > best_value:
                best_counts = counts
                report(best_counts)

        return best_counts

    except gp.GurobiError:
        return incumbent_counts
    except Exception:
        return incumbent_counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    start_time = time.monotonic()
    deadline = start_time + max(0, args.time_limit)

    logger = SolutionLogger(
        args.log_path, sense="maximize"
    ) if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as f:
        instance = json.load(f)

    panel_l = int(instance["panel"]["length"])
    panel_w = int(instance["panel"]["width"])
    items = list(instance["items"])

    best_counts = [0] * len(items)
    best_value = -1

    def report(counts):
        nonlocal best_counts, best_value
        value = solution_value(items, counts)
        if value > best_value:
            best_value = value
            best_counts = list(counts)
            solution = make_solution(items, best_counts)
            if logger:
                logger.log_solution(float(value), solution)

    # The empty cutting pattern is always feasible.
    report(best_counts)

    if time.monotonic() < deadline:
        heuristic_counts = initial_heuristics(
            panel_l, panel_w, items, report
        )
        if solution_value(items, heuristic_counts) > best_value:
            report(heuristic_counts)

    if time.monotonic() < deadline:
        mip_counts = solve_with_plate_flow(
            panel_l, panel_w, items, deadline, best_counts, report
        )
        if solution_value(items, mip_counts) > best_value:
            report(mip_counts)

    final_solution = make_solution(items, best_counts)

    solution_directory = os.path.dirname(os.path.abspath(args.solution_path))
    if solution_directory:
        os.makedirs(solution_directory, exist_ok=True)

    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(final_solution, f, indent=2)


if __name__ == "__main__":
    main()
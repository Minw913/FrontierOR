import argparse
import bisect
import json
import math
import os
import time
from collections import Counter

from solution_logger import SolutionLogger


def make_solution(bin_patterns):
    bins = []
    for pattern in bin_patterns:
        packed = []
        for item_index, count in sorted(pattern.items()):
            packed.extend([ITEM_IDS[item_index]] * int(count))
        bins.append(packed)
    return {
        "objective_value": float(len(bins)),
        "num_bins": len(bins),
        "bins": bins,
    }


def pattern_key(pattern):
    return tuple(sorted((int(i), int(c)) for i, c in pattern.items() if c > 0))


def first_fit_decreasing(type_order, weights, demands, capacity):
    total_items = sum(demands)
    if total_items == 0:
        return []

    size = 1
    while size < total_items:
        size <<= 1
    tree = [0] * (2 * size)
    remaining = []
    patterns = []

    def add_bin(item_index, weight):
        bin_index = len(remaining)
        rem = capacity - weight
        remaining.append(rem)
        patterns.append({item_index: 1})

        pos = size + bin_index
        tree[pos] = rem
        pos >>= 1
        while pos:
            new_value = max(tree[pos << 1], tree[(pos << 1) | 1])
            if tree[pos] == new_value:
                break
            tree[pos] = new_value
            pos >>= 1

    def find_first(required):
        if not remaining or tree[1] < required:
            return -1
        pos = 1
        while pos < size:
            left = pos << 1
            if tree[left] >= required:
                pos = left
            else:
                pos = left | 1
        idx = pos - size
        return idx if idx < len(remaining) else -1

    def update_bin(bin_index, item_index, weight):
        remaining[bin_index] -= weight
        patterns[bin_index][item_index] = patterns[bin_index].get(item_index, 0) + 1

        pos = size + bin_index
        tree[pos] = remaining[bin_index]
        pos >>= 1
        while pos:
            new_value = max(tree[pos << 1], tree[(pos << 1) | 1])
            if tree[pos] == new_value:
                break
            tree[pos] = new_value
            pos >>= 1

    for item_index in type_order:
        weight = weights[item_index]
        for _ in range(demands[item_index]):
            bin_index = find_first(weight)
            if bin_index < 0:
                add_bin(item_index, weight)
            else:
                update_bin(bin_index, item_index, weight)

    return patterns


def best_fit_decreasing(type_order, weights, demands, capacity):
    # This implementation is used only on moderate-size instances.
    sorted_remaining = []  # (remaining capacity, bin index)
    patterns = []

    for item_index in type_order:
        weight = weights[item_index]
        for _ in range(demands[item_index]):
            pos = bisect.bisect_left(sorted_remaining, (weight, -1))
            if pos == len(sorted_remaining):
                bin_index = len(patterns)
                patterns.append({item_index: 1})
                bisect.insort(sorted_remaining, (capacity - weight, bin_index))
            else:
                rem, bin_index = sorted_remaining.pop(pos)
                patterns[bin_index][item_index] = (
                    patterns[bin_index].get(item_index, 0) + 1
                )
                bisect.insort(sorted_remaining, (rem - weight, bin_index))

    return patterns


def simple_lower_bound(weights, demands, capacity):
    total_weight = sum(w * d for w, d in zip(weights, demands))
    lower_bound = (total_weight + capacity - 1) // capacity

    # Threshold bounds: items heavier than W/(k+1) occur at most k per bin.
    max_k = min(30, max(1, capacity))
    for k in range(1, max_k + 1):
        count = sum(
            d for w, d in zip(weights, demands)
            if w * (k + 1) > capacity
        )
        lower_bound = max(lower_bound, (count + k - 1) // k)

    return lower_bound


def configure_gurobi(model):
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1


def solve_with_patterns(
    weights,
    demands,
    capacity,
    initial_pattern_sets,
    current_best,
    logger,
    start_time,
    time_limit,
):
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception:
        return current_best

    m = len(weights)
    all_patterns = []
    pattern_to_index = {}

    def register_pattern(pattern):
        key = pattern_key(pattern)
        if not key or key in pattern_to_index:
            return False
        pattern_to_index[key] = len(all_patterns)
        all_patterns.append(key)
        return True

    for pattern_set in initial_pattern_sets:
        for pattern in pattern_set:
            register_pattern(pattern)

    # Singleton patterns guarantee feasibility of the equality master.
    for i in range(m):
        if demands[i] > 0:
            register_pattern({i: 1})

    # Homogeneous maximal patterns often substantially strengthen the start.
    for i in range(m):
        if demands[i] > 0:
            count = min(demands[i], capacity // weights[i])
            if count > 0:
                register_pattern({i: count})

    try:
        master = gp.Model("bin_packing_master")
        configure_gurobi(master)
        master.Params.Method = 1

        constraints = [
            master.addConstr(gp.LinExpr() == demands[i], name=f"demand_{i}")
            for i in range(m)
        ]

        variables = []

        def add_master_variable(key):
            column = gp.Column()
            for i, count in key:
                column.addTerms(float(count), constraints[i])
            var = master.addVar(
                lb=0.0,
                obj=1.0,
                vtype=GRB.CONTINUOUS,
                column=column,
                name=f"p_{len(variables)}",
            )
            variables.append(var)

        for key in all_patterns:
            add_master_variable(key)
        master.update()

        # Warm start corresponding to the current incumbent.
        incumbent_frequency = Counter(pattern_key(p) for p in current_best)
        for key, count in incumbent_frequency.items():
            idx = pattern_to_index.get(key)
            if idx is not None:
                variables[idx].Start = float(count)

        total_deadline = start_time + max(0, time_limit)
        column_generation_deadline = start_time + max(0, time_limit) * 0.38
        max_iterations = 120

        for _ in range(max_iterations):
            now = time.monotonic()
            if now >= column_generation_deadline or now >= total_deadline:
                break

            master.Params.TimeLimit = max(
                0.05, min(column_generation_deadline, total_deadline) - now
            )
            master.optimize()

            if master.Status != GRB.OPTIMAL:
                break

            duals = [constraint.Pi for constraint in constraints]

            pricing = gp.Model("pricing")
            configure_gurobi(pricing)
            pricing.Params.PoolSearchMode = 1
            pricing.Params.PoolSolutions = 8
            pricing.Params.PoolGap = 0.10

            pricing_vars = []
            for i in range(m):
                upper = min(demands[i], capacity // weights[i])
                pricing_vars.append(
                    pricing.addVar(
                        lb=0,
                        ub=upper,
                        vtype=GRB.INTEGER,
                        obj=duals[i],
                        name=f"x_{i}",
                    )
                )

            pricing.addConstr(
                gp.quicksum(weights[i] * pricing_vars[i] for i in range(m))
                <= capacity
            )
            pricing.ModelSense = GRB.MAXIMIZE

            remaining_cg = min(column_generation_deadline, total_deadline) - time.monotonic()
            if remaining_cg <= 0:
                break
            pricing.Params.TimeLimit = max(0.05, min(3.0, remaining_cg))
            pricing.optimize()

            if pricing.SolCount == 0:
                break

            new_keys = []
            number_solutions = min(pricing.SolCount, 8)
            for solution_number in range(number_solutions):
                pricing.Params.SolutionNumber = solution_number
                pattern = {}
                dual_value = 0.0
                for i, var in enumerate(pricing_vars):
                    count = int(round(var.Xn))
                    if count > 0:
                        pattern[i] = count
                        dual_value += duals[i] * count

                if dual_value > 1.0 + 1e-6:
                    key = pattern_key(pattern)
                    if key and key not in pattern_to_index:
                        pattern_to_index[key] = len(all_patterns)
                        all_patterns.append(key)
                        new_keys.append(key)

            if not new_keys:
                break

            for key in new_keys:
                add_master_variable(key)
            master.update()

        remaining = total_deadline - time.monotonic()
        if remaining <= 0:
            return current_best

        # Convert the restricted master to an integer model.
        for var in variables:
            var.VType = GRB.INTEGER
        master.update()

        incumbent_frequency = Counter(pattern_key(p) for p in current_best)
        for idx, key in enumerate(all_patterns):
            variables[idx].Start = float(incumbent_frequency.get(key, 0))

        best_holder = {
            "patterns": current_best,
            "objective": len(current_best),
        }

        def selected_patterns_from_values(values):
            selected = []
            for idx, value in enumerate(values):
                multiplicity = int(round(value))
                if multiplicity <= 0:
                    continue
                pattern = dict(all_patterns[idx])
                for _ in range(multiplicity):
                    selected.append(pattern.copy())
            return selected

        def callback(model, where):
            if where != GRB.Callback.MIPSOL:
                return

            objective = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            integer_objective = int(round(objective))
            if integer_objective >= best_holder["objective"]:
                return

            values = model.cbGetSolution(variables)
            selected = selected_patterns_from_values(values)
            if len(selected) != integer_objective:
                return

            best_holder["patterns"] = selected
            best_holder["objective"] = integer_objective
            if logger:
                solution = make_solution(selected)
                logger.log_solution(float(integer_objective), solution)

        master.Params.TimeLimit = max(0.05, remaining)
        master.optimize(callback)

        if master.SolCount > 0:
            values = [var.X for var in variables]
            selected = selected_patterns_from_values(values)
            if len(selected) < best_holder["objective"]:
                best_holder["patterns"] = selected
                best_holder["objective"] = len(selected)
                if logger:
                    solution = make_solution(selected)
                    logger.log_solution(float(len(selected)), solution)

        return best_holder["patterns"]

    except Exception:
        # The heuristic incumbent remains a valid solution if Gurobi is
        # unavailable, interrupted, or cannot solve the generated model.
        return current_best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    start_time = time.monotonic()

    with open(args.instance_path, "r", encoding="utf-8") as f:
        instance = json.load(f)

    global ITEM_IDS

    capacity = int(instance["parameters"]["W"])
    raw_items = instance["items"]

    # Merge repeated identifiers defensively while retaining schema identifiers.
    merged = {}
    for item in raw_items:
        item_id = int(item["item_type"])
        weight = int(item["weight"])
        demand = int(item["demand"])
        if demand <= 0:
            continue
        if item_id in merged:
            if merged[item_id][0] != weight:
                raise ValueError(f"Inconsistent weights for item type {item_id}")
            merged[item_id][1] += demand
        else:
            merged[item_id] = [weight, demand]

    ordered_ids = sorted(merged)
    ITEM_IDS = ordered_ids
    weights = [merged[item_id][0] for item_id in ordered_ids]
    demands = [merged[item_id][1] for item_id in ordered_ids]

    if not weights:
        solution = {
            "objective_value": 0.0,
            "num_bins": 0,
            "bins": [],
        }
        if logger:
            logger.log_solution(0.0, solution)
        with open(args.solution_path, "w", encoding="utf-8") as f:
            json.dump(solution, f)
        return

    for item_id, weight in zip(ITEM_IDS, weights):
        if weight <= 0 or weight > capacity:
            raise ValueError(f"Invalid weight for item type {item_id}: {weight}")

    total_items = sum(demands)
    base_order = sorted(
        range(len(weights)),
        key=lambda i: (-weights[i], ITEM_IDS[i]),
    )
    reverse_tie_order = sorted(
        range(len(weights)),
        key=lambda i: (-weights[i], -ITEM_IDS[i]),
    )

    heuristic_pattern_sets = []

    best_patterns = first_fit_decreasing(
        base_order, weights, demands, capacity
    )
    heuristic_pattern_sets.append(best_patterns)

    if logger:
        initial_solution = make_solution(best_patterns)
        logger.log_solution(float(len(best_patterns)), initial_solution)

    lower_bound = simple_lower_bound(weights, demands, capacity)

    # Alternative type tie order supplies useful extra patterns at low cost.
    if (
        len(best_patterns) > lower_bound
        and time.monotonic() - start_time < max(0, args.time_limit) * 0.15
    ):
        alternative = first_fit_decreasing(
            reverse_tie_order, weights, demands, capacity
        )
        heuristic_pattern_sets.append(alternative)
        if len(alternative) < len(best_patterns):
            best_patterns = alternative
            if logger:
                solution = make_solution(best_patterns)
                logger.log_solution(float(len(best_patterns)), solution)

    # Best-fit decreasing is effective but uses a sorted-list structure, so it
    # is restricted to moderate-size expanded instances.
    if (
        total_items <= 25000
        and len(best_patterns) > lower_bound
        and time.monotonic() - start_time < max(0, args.time_limit) * 0.22
    ):
        bfd = best_fit_decreasing(base_order, weights, demands, capacity)
        heuristic_pattern_sets.append(bfd)
        if len(bfd) < len(best_patterns):
            best_patterns = bfd
            if logger:
                solution = make_solution(best_patterns)
                logger.log_solution(float(len(best_patterns)), solution)

    if (
        len(best_patterns) > lower_bound
        and time.monotonic() - start_time < max(0, args.time_limit) - 0.05
    ):
        best_patterns = solve_with_patterns(
            weights=weights,
            demands=demands,
            capacity=capacity,
            initial_pattern_sets=heuristic_pattern_sets,
            current_best=best_patterns,
            logger=logger,
            start_time=start_time,
            time_limit=max(0, args.time_limit),
        )

    final_solution = make_solution(best_patterns)

    solution_directory = os.path.dirname(os.path.abspath(args.solution_path))
    if solution_directory:
        os.makedirs(solution_directory, exist_ok=True)

    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(final_solution, f)


if __name__ == "__main__":
    main()
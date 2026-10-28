import argparse
import json
import math
import time
from array import array
from collections import Counter

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


EPS = 1e-6


def configure_gurobi(model, output=False):
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.OutputFlag = 1 if output else 0


def make_solution(bin_types, weights):
    assignments = [[weights[t] for t in bin_items] for bin_items in bin_types]
    count = len(assignments)
    return {
        "objective_value": float(count),
        "num_bins": count,
        "bin_assignments": assignments,
    }


def pattern_to_items(pattern):
    items = []
    for item_type, count in pattern:
        if count > 0:
            items.extend([item_type] * count)
    return items


def patterns_to_solution(patterns, multiplicities, weights):
    bins = []
    for pattern, count in zip(patterns, multiplicities):
        count = int(round(count))
        if count <= 0:
            continue
        items = pattern_to_items(pattern)
        for _ in range(count):
            bins.append([weights[t] for t in items])

    return {
        "objective_value": float(len(bins)),
        "num_bins": len(bins),
        "bin_assignments": bins,
    }


def validate_solution(solution, capacity, weights, demands):
    expected = Counter()
    for w, d in zip(weights, demands):
        expected[w] += d

    actual = Counter()
    for bin_items in solution["bin_assignments"]:
        if sum(bin_items) > capacity:
            return False
        actual.update(bin_items)

    return (
        actual == expected
        and solution["num_bins"] == len(solution["bin_assignments"])
        and int(round(solution["objective_value"])) == solution["num_bins"]
    )


def first_fit_decreasing(weights, demands, capacity):
    """
    First-fit decreasing using a segment tree. Each leaf stores a bin's
    remaining capacity, and internal nodes store the maximum remaining
    capacity in their subtree.
    """
    total_items = sum(demands)
    if total_items == 0:
        return []

    tree_size = 1
    while tree_size < total_items:
        tree_size <<= 1

    tree = array("q", [0]) * (2 * tree_size)
    bins = []
    bin_count = 0

    def update(index, value):
        pos = tree_size + index
        tree[pos] = value
        pos >>= 1
        while pos:
            new_value = max(tree[pos << 1], tree[(pos << 1) | 1])
            if tree[pos] == new_value:
                break
            tree[pos] = new_value
            pos >>= 1

    def find_first(required):
        pos = 1
        while pos < tree_size:
            left = pos << 1
            if tree[left] >= required:
                pos = left
            else:
                pos = left | 1
        return pos - tree_size

    order = sorted(range(len(weights)), key=lambda i: (-weights[i], i))

    for item_type in order:
        width = weights[item_type]
        for _ in range(demands[item_type]):
            if tree[1] < width:
                idx = bin_count
                bin_count += 1
                bins.append([item_type])
                update(idx, capacity - width)
            else:
                idx = find_first(width)
                bins[idx].append(item_type)
                remaining = tree[tree_size + idx] - width
                update(idx, remaining)

    return bins


def elementary_lower_bound(weights, demands, capacity):
    total_width = sum(w * d for w, d in zip(weights, demands))
    lower_bound = (total_width + capacity - 1) // capacity

    for w, d in zip(weights, demands):
        max_per_bin = capacity // w
        lower_bound = max(lower_bound, (d + max_per_bin - 1) // max_per_bin)

    # For every item-size threshold, bound the number of threshold-sized
    # items that can fit in one bin.
    distinct = sorted(set(weights), reverse=True)
    running_items = 0
    demand_by_weight = Counter()
    for w, d in zip(weights, demands):
        demand_by_weight[w] += d

    for threshold in distinct:
        running_items += demand_by_weight[threshold]
        max_count = capacity // threshold
        if max_count > 0:
            lower_bound = max(
                lower_bound,
                (running_items + max_count - 1) // max_count,
            )

    return lower_bound


def solve_direct_bin_mip(
    weights,
    demands,
    capacity,
    heuristic_bins,
    incumbent_solution,
    logger,
    deadline,
):
    m = len(weights)
    upper_bound = len(heuristic_bins)
    best_solution = incumbent_solution
    best_value = upper_bound

    remaining = deadline - time.monotonic()
    if remaining <= 0.05 or upper_bound == 0:
        return best_solution

    # Ordering bins by load makes the heuristic start satisfy load symmetry.
    start_bins = sorted(
        heuristic_bins,
        key=lambda b: -sum(weights[t] for t in b),
    )

    model = gp.Model("grouped_bin_packing")
    configure_gurobi(model)

    y = model.addVars(upper_bound, vtype=GRB.BINARY, name="used")
    x = {}

    for i in range(m):
        upper = min(demands[i], capacity // weights[i])
        for b in range(upper_bound):
            x[i, b] = model.addVar(
                vtype=GRB.INTEGER,
                lb=0,
                ub=upper,
                name=f"x_{i}_{b}",
            )

    model.update()

    for i in range(m):
        model.addConstr(
            gp.quicksum(x[i, b] for b in range(upper_bound)) == demands[i],
            name=f"demand_{i}",
        )

    loads = []
    for b in range(upper_bound):
        load = gp.quicksum(weights[i] * x[i, b] for i in range(m))
        loads.append(load)
        model.addConstr(load <= capacity * y[b], name=f"capacity_{b}")

    for b in range(upper_bound - 1):
        model.addConstr(y[b] >= y[b + 1], name=f"used_order_{b}")
        model.addConstr(loads[b] >= loads[b + 1], name=f"load_order_{b}")

    model.setObjective(gp.quicksum(y[b] for b in range(upper_bound)), GRB.MINIMIZE)

    start_counts = []
    for bin_items in start_bins:
        start_counts.append(Counter(bin_items))

    for b in range(upper_bound):
        y[b].Start = 1.0
        counts = start_counts[b]
        for i in range(m):
            x[i, b].Start = float(counts.get(i, 0))

    model.Params.TimeLimit = max(0.01, deadline - time.monotonic())
    model.Params.Cutoff = upper_bound + 0.5

    def extract_from_values(get_value):
        packed = []
        for b in range(upper_bound):
            if get_value(y[b]) < 0.5:
                continue
            one_bin = []
            for i in range(m):
                count = int(round(get_value(x[i, b])))
                if count > 0:
                    one_bin.extend([weights[i]] * count)
            if one_bin:
                packed.append(one_bin)

        return {
            "objective_value": float(len(packed)),
            "num_bins": len(packed),
            "bin_assignments": packed,
        }

    def callback(cb_model, where):
        nonlocal best_solution, best_value
        if where != GRB.Callback.MIPSOL:
            return
        try:
            objective = int(round(cb_model.cbGet(GRB.Callback.MIPSOL_OBJ)))
            if objective >= best_value:
                return
            candidate = extract_from_values(cb_model.cbGetSolution)
            if candidate["num_bins"] != objective:
                return
            if not validate_solution(candidate, capacity, weights, demands):
                return

            best_value = objective
            best_solution = candidate
            if logger:
                logger.log_solution(float(objective), candidate)
        except Exception:
            # A callback should not abort optimization because of a transient
            # solution-extraction issue.
            return

    try:
        model.optimize(callback)

        if model.SolCount > 0:
            candidate = extract_from_values(lambda var: var.X)
            if (
                candidate["num_bins"] < best_value
                and validate_solution(candidate, capacity, weights, demands)
            ):
                best_value = candidate["num_bins"]
                best_solution = candidate
                if logger:
                    logger.log_solution(float(best_value), best_solution)
    except gp.GurobiError:
        pass
    finally:
        model.dispose()

    return best_solution


def build_initial_patterns(weights, demands, capacity, heuristic_bins):
    patterns = []
    pattern_set = set()
    heuristic_frequency = Counter()

    def add(pattern):
        pattern = tuple((i, int(v)) for i, v in pattern if v > 0)
        if not pattern or pattern in pattern_set:
            return
        pattern_set.add(pattern)
        patterns.append(pattern)

    for i, (w, d) in enumerate(zip(weights, demands)):
        if d <= 0:
            continue
        add(((i, 1),))
        homogeneous_count = min(d, capacity // w)
        add(((i, homogeneous_count),))

    for bin_items in heuristic_bins:
        counts = Counter(bin_items)
        pattern = tuple(sorted(counts.items()))
        add(pattern)
        heuristic_frequency[pattern] += 1

    return patterns, pattern_set, heuristic_frequency


def column_generation(
    weights,
    demands,
    capacity,
    heuristic_bins,
    deadline,
    phase_deadline,
):
    m = len(weights)
    patterns, pattern_set, heuristic_frequency = build_initial_patterns(
        weights, demands, capacity, heuristic_bins
    )

    master = gp.Model("cutting_stock_lp")
    configure_gurobi(master)

    variables = [
        master.addVar(lb=0.0, obj=1.0, vtype=GRB.CONTINUOUS, name=f"p_{j}")
        for j in range(len(patterns))
    ]
    master.update()

    incidence = [[] for _ in range(m)]
    for j, pattern in enumerate(patterns):
        for i, count in pattern:
            incidence[i].append(count * variables[j])

    constraints = []
    for i in range(m):
        constraints.append(
            master.addConstr(
                gp.quicksum(incidence[i]) == demands[i],
                name=f"demand_{i}",
            )
        )

    master.ModelSense = GRB.MINIMIZE

    pricing = gp.Model("pricing_knapsack")
    configure_gurobi(pricing)

    z = []
    for i in range(m):
        upper = min(demands[i], capacity // weights[i])
        z.append(
            pricing.addVar(
                lb=0,
                ub=upper,
                vtype=GRB.INTEGER,
                name=f"z_{i}",
            )
        )

    pricing.addConstr(
        gp.quicksum(weights[i] * z[i] for i in range(m)) <= capacity
    )
    pricing.ModelSense = GRB.MAXIMIZE
    pricing.Params.PoolSearchMode = 1
    pricing.Params.PoolSolutions = 12

    try:
        while time.monotonic() < phase_deadline - 0.05:
            master.Params.TimeLimit = max(
                0.01, min(phase_deadline, deadline) - time.monotonic()
            )
            master.optimize()
            if master.Status != GRB.OPTIMAL:
                break

            duals = [constraints[i].Pi for i in range(m)]
            pricing.setObjective(
                gp.quicksum(duals[i] * z[i] for i in range(m)),
                GRB.MAXIMIZE,
            )

            remaining_phase = phase_deadline - time.monotonic()
            if remaining_phase <= 0.05:
                break
            pricing.Params.TimeLimit = max(0.01, min(1.5, remaining_phase))
            pricing.optimize()

            if pricing.SolCount == 0:
                break

            added = 0
            best_pricing_value = -GRB.INFINITY

            solution_count = min(pricing.SolCount, 12)
            for solution_number in range(solution_count):
                pricing.Params.SolutionNumber = solution_number
                pattern_list = []
                dual_value = 0.0
                used_capacity = 0

                for i in range(m):
                    count = int(round(z[i].Xn))
                    if count > 0:
                        pattern_list.append((i, count))
                        dual_value += duals[i] * count
                        used_capacity += weights[i] * count

                best_pricing_value = max(best_pricing_value, dual_value)
                pattern = tuple(pattern_list)

                if (
                    not pattern
                    or used_capacity > capacity
                    or dual_value <= 1.0 + 1e-7
                    or pattern in pattern_set
                ):
                    continue

                column = gp.Column()
                for i, count in pattern:
                    column.addTerms(count, constraints[i])

                variable = master.addVar(
                    lb=0.0,
                    obj=1.0,
                    vtype=GRB.CONTINUOUS,
                    column=column,
                    name=f"p_{len(patterns)}",
                )
                patterns.append(pattern)
                pattern_set.add(pattern)
                variables.append(variable)
                added += 1

            if added:
                master.update()
            else:
                # If pricing was solved to near optimality and found no
                # negative reduced-cost pattern, the LP column generation is
                # converged. Otherwise, stop to preserve time for the integer
                # master.
                if (
                    pricing.Status == GRB.OPTIMAL
                    or best_pricing_value <= 1.0 + 1e-7
                ):
                    break
                break
    except gp.GurobiError:
        pass
    finally:
        pricing.dispose()
        master.dispose()

    return patterns, heuristic_frequency


def solve_integer_pattern_master(
    weights,
    demands,
    capacity,
    patterns,
    heuristic_frequency,
    incumbent_solution,
    logger,
    deadline,
):
    best_solution = incumbent_solution
    best_value = incumbent_solution["num_bins"]
    m = len(weights)

    if time.monotonic() >= deadline - 0.02:
        return best_solution

    model = gp.Model("integer_pattern_master")
    configure_gurobi(model)

    variables = []
    for j, pattern in enumerate(patterns):
        # No pattern can be used more often than the tightest demand ratio.
        upper = min(demands[i] // count for i, count in pattern)
        var = model.addVar(
            lb=0,
            ub=upper,
            vtype=GRB.INTEGER,
            obj=1.0,
            name=f"p_{j}",
        )
        variables.append(var)

    model.update()

    incidence = [[] for _ in range(m)]
    for j, pattern in enumerate(patterns):
        for i, count in pattern:
            incidence[i].append(count * variables[j])

    for i in range(m):
        model.addConstr(
            gp.quicksum(incidence[i]) == demands[i],
            name=f"demand_{i}",
        )

    model.ModelSense = GRB.MINIMIZE

    for j, pattern in enumerate(patterns):
        variables[j].Start = float(heuristic_frequency.get(pattern, 0))

    model.Params.TimeLimit = max(0.01, deadline - time.monotonic())
    model.Params.Cutoff = best_value + 0.5

    def candidate_from_values(get_value):
        multiplicities = [
            int(round(get_value(variable))) for variable in variables
        ]
        return patterns_to_solution(patterns, multiplicities, weights)

    def callback(cb_model, where):
        nonlocal best_solution, best_value
        if where != GRB.Callback.MIPSOL:
            return

        try:
            objective = int(round(cb_model.cbGet(GRB.Callback.MIPSOL_OBJ)))
            if objective >= best_value:
                return

            candidate = candidate_from_values(cb_model.cbGetSolution)
            if candidate["num_bins"] != objective:
                return
            if not validate_solution(candidate, capacity, weights, demands):
                return

            best_value = objective
            best_solution = candidate
            if logger:
                logger.log_solution(float(objective), candidate)
        except Exception:
            return

    try:
        model.optimize(callback)

        if model.SolCount > 0:
            candidate = candidate_from_values(lambda var: var.X)
            if (
                candidate["num_bins"] < best_value
                and validate_solution(candidate, capacity, weights, demands)
            ):
                best_value = candidate["num_bins"]
                best_solution = candidate
                if logger:
                    logger.log_solution(float(best_value), best_solution)
    except gp.GurobiError:
        pass
    finally:
        model.dispose()

    return best_solution


def write_solution(path, solution):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(solution, f, separators=(",", ":"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    start_time = time.monotonic()
    deadline = start_time + max(0, args.time_limit)

    logger = (
        SolutionLogger(args.log_path, sense="minimize")
        if args.log_path
        else None
    )

    with open(args.instance_path, "r", encoding="utf-8") as f:
        instance = json.load(f)

    capacity = int(instance["c"])
    weights = [int(v) for v in instance["weights"]]
    demands = [int(v) for v in instance["demands"]]

    if len(weights) != int(instance["m"]) or len(demands) != len(weights):
        raise ValueError("Instance dimensions do not match m")
    if capacity <= 0:
        raise ValueError("Bin capacity must be positive")
    if any(w <= 0 or w >= capacity for w in weights):
        raise ValueError("Each item width must be strictly between 0 and capacity")
    if any(d < 0 for d in demands):
        raise ValueError("Demands must be nonnegative")

    total_items = sum(demands)
    if total_items == 0:
        solution = {
            "objective_value": 0.0,
            "num_bins": 0,
            "bin_assignments": [],
        }
        if logger:
            logger.log_solution(0.0, solution)
        write_solution(args.solution_path, solution)
        return

    heuristic_bins = first_fit_decreasing(weights, demands, capacity)
    best_solution = make_solution(heuristic_bins, weights)

    if logger:
        logger.log_solution(best_solution["objective_value"], best_solution)

    lower_bound = elementary_lower_bound(weights, demands, capacity)
    if best_solution["num_bins"] == lower_bound or time.monotonic() >= deadline:
        write_solution(args.solution_path, best_solution)
        return

    m = len(weights)
    upper_bound = len(heuristic_bins)
    direct_variable_count = m * upper_bound

    # A compact bin-indexed model can prove optimality effectively on small
    # and medium grouped instances. Larger instances use cutting-stock column
    # generation followed by an integer restricted master.
    use_direct = (
        upper_bound <= 500
        and direct_variable_count <= 60000
        and time.monotonic() < deadline - 0.05
    )

    if use_direct:
        best_solution = solve_direct_bin_mip(
            weights,
            demands,
            capacity,
            heuristic_bins,
            best_solution,
            logger,
            deadline,
        )
    else:
        remaining = max(0.0, deadline - time.monotonic())
        # Reserve most of the runtime for the integer master.
        cg_budget = min(0.40 * remaining, max(0.0, remaining - 0.5))
        phase_deadline = time.monotonic() + cg_budget

        patterns, heuristic_frequency = column_generation(
            weights,
            demands,
            capacity,
            heuristic_bins,
            deadline,
            phase_deadline,
        )

        best_solution = solve_integer_pattern_master(
            weights,
            demands,
            capacity,
            patterns,
            heuristic_frequency,
            best_solution,
            logger,
            deadline,
        )

    if not validate_solution(best_solution, capacity, weights, demands):
        best_solution = make_solution(heuristic_bins, weights)

    write_solution(args.solution_path, best_solution)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Bin Packing Problem Solver
Combines heuristics (FFD, BFD), ILP via Gurobi (pattern-based, column generation,
assignment-based), and local search for high-quality solutions.
"""

import json
import argparse
import time
import math
import sys
from collections import Counter


def first_fit_decreasing(items, capacity):
    """First Fit Decreasing heuristic."""
    sorted_items = sorted(items, reverse=True)
    bins = []
    bin_remaining = []
    for item in sorted_items:
        placed = False
        for i in range(len(bins)):
            if bin_remaining[i] >= item:
                bins[i].append(item)
                bin_remaining[i] -= item
                placed = True
                break
        if not placed:
            bins.append([item])
            bin_remaining.append(capacity - item)
    return bins


def best_fit_decreasing(items, capacity):
    """Best Fit Decreasing heuristic."""
    sorted_items = sorted(items, reverse=True)
    bins = []
    bin_remaining = []
    for item in sorted_items:
        best_bin = -1
        best_remaining = float('inf')
        for i in range(len(bins)):
            if bin_remaining[i] >= item and bin_remaining[i] - item < best_remaining:
                best_bin = i
                best_remaining = bin_remaining[i] - item
        if best_bin >= 0:
            bins[best_bin].append(item)
            bin_remaining[best_bin] -= item
        else:
            bins.append([item])
            bin_remaining.append(capacity - item)
    return bins


def worst_fit_decreasing(items, capacity):
    """Worst Fit Decreasing heuristic."""
    sorted_items = sorted(items, reverse=True)
    bins = []
    bin_remaining = []
    for item in sorted_items:
        best_bin = -1
        best_remaining = -1
        for i in range(len(bins)):
            if bin_remaining[i] >= item and bin_remaining[i] > best_remaining:
                best_bin = i
                best_remaining = bin_remaining[i]
        if best_bin >= 0:
            bins[best_bin].append(item)
            bin_remaining[best_bin] -= item
        else:
            bins.append([item])
            bin_remaining.append(capacity - item)
    return bins


def format_solution(bins):
    """Format bins into the solution schema."""
    bin_assignments = []
    for b in bins:
        bin_assignments.append({
            "items": list(b),
            "total_size": sum(b)
        })
    return {
        "objective_value": len(bins),
        "bin_assignments": bin_assignments,
        "num_bins": len(bins)
    }


def try_eliminate_bin(bins, bin_idx, capacity):
    """Try to eliminate a bin by redistributing its items using best-fit."""
    items_to_place = sorted(bins[bin_idx], reverse=True)
    other_bins = [list(bins[i]) for i in range(len(bins)) if i != bin_idx]
    remaining = [capacity - sum(b) for b in other_bins]

    for item in items_to_place:
        best_bin = -1
        best_rem = float('inf')
        for i in range(len(other_bins)):
            if remaining[i] >= item and remaining[i] - item < best_rem:
                best_bin = i
                best_rem = remaining[i] - item
        if best_bin < 0:
            return None
        other_bins[best_bin].append(item)
        remaining[best_bin] -= item

    return other_bins


def local_search(bins, capacity, time_limit, start_time):
    """Improve solution through local search: merging and bin elimination."""
    bins = [list(b) for b in bins]

    improved = True
    while improved and time.time() - start_time < time_limit - 0.1:
        improved = False

        # Phase 1: Try to merge pairs of bins
        i = 0
        while i < len(bins) and time.time() - start_time < time_limit - 0.1:
            j = i + 1
            merged = False
            while j < len(bins) and time.time() - start_time < time_limit - 0.1:
                if sum(bins[i]) + sum(bins[j]) <= capacity:
                    bins[i].extend(bins[j])
                    bins.pop(j)
                    improved = True
                    merged = True
                else:
                    j += 1
            i += 1

        if improved:
            continue

        # Phase 2: Try to eliminate bins (least full first)
        bin_sums = [(sum(bins[i]), i) for i in range(len(bins))]
        bin_sums.sort()

        for _, idx in bin_sums:
            if time.time() - start_time >= time_limit - 0.1:
                break
            result = try_eliminate_bin(bins, idx, capacity)
            if result is not None:
                bins = result
                improved = True
                break

    return bins


def generate_patterns(sizes, demands, capacity, max_patterns=200000):
    """Generate all feasible patterns via backtracking."""
    patterns = []
    k = len(sizes)
    count = [0] * k

    def backtrack(idx, remaining):
        if len(patterns) >= max_patterns:
            return
        if idx == k:
            patterns.append(tuple(count))
            return
        max_count = min(demands[idx], remaining // sizes[idx])
        for c in range(max_count + 1):
            count[idx] = c
            backtrack(idx + 1, remaining - c * sizes[idx])
        count[idx] = 0

    backtrack(0, capacity)

    if len(patterns) >= max_patterns:
        return None
    return patterns


def solve_pattern_ilp(sizes, demands, capacity, time_limit):
    """Solve using pattern-based ILP with Gurobi."""
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError:
        return None

    patterns = generate_patterns(sizes, demands, capacity)
    if patterns is None:
        return None

    patterns = [p for p in patterns if sum(p) > 0]
    if len(patterns) == 0:
        return None

    k = len(sizes)
    num_patterns = len(patterns)

    model = gp.Model("bin_packing_pattern")
    model.setParam('OutputFlag', 0)
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', time_limit)

    x = model.addVars(num_patterns, vtype=GRB.INTEGER, lb=0, name="x")
    model.setObjective(gp.quicksum(x[j] for j in range(num_patterns)), GRB.MINIMIZE)

    for i in range(k):
        model.addConstr(
            gp.quicksum(patterns[j][i] * x[j] for j in range(num_patterns)) >= demands[i],
            name=f"demand_{i}"
        )

    model.optimize()

    if model.SolCount > 0:
        bins = []
        for j in range(num_patterns):
            cnt = int(round(x[j].X))
            for _ in range(cnt):
                bin_items = []
                for i in range(k):
                    bin_items.extend([sizes[i]] * patterns[j][i])
                bins.append(bin_items)
        return bins

    return None


def solve_pricing_dp(sizes, demands, capacity, duals):
    """Solve bounded knapsack pricing problem using DP."""
    k = len(sizes)

    # Limit DP size
    if capacity > 100000 or k * (capacity + 1) > 10_000_000:
        return None

    # dp[i][w] = max value using types 0..i-1 with capacity w
    dp = [[0.0] * (capacity + 1) for _ in range(k + 1)]

    for i in range(k):
        for w in range(capacity + 1):
            dp[i + 1][w] = dp[i][w]
            if duals[i] > 1e-10:
                max_c = min(demands[i], w // sizes[i])
                for c in range(1, max_c + 1):
                    val = dp[i][w - c * sizes[i]] + c * duals[i]
                    if val > dp[i + 1][w]:
                        dp[i + 1][w] = val

    # Find best value
    best_w = 0
    best_val = 0.0
    for w in range(capacity + 1):
        if dp[k][w] > best_val:
            best_val = dp[k][w]
            best_w = w

    if best_val <= 1.0 + 1e-6:
        return None

    # Reconstruct pattern
    pattern = [0] * k
    w = best_w
    for i in range(k - 1, -1, -1):
        max_c = min(demands[i], w // sizes[i])
        for c in range(max_c, -1, -1):
            if c == 0:
                if abs(dp[i + 1][w] - dp[i][w]) < 1e-9:
                    pattern[i] = 0
                    break
            else:
                expected = dp[i][w - c * sizes[i]] + c * duals[i]
                if abs(dp[i + 1][w] - expected) < 1e-9:
                    pattern[i] = c
                    w -= c * sizes[i]
                    break

    if sum(pattern) == 0:
        return None
    return tuple(pattern)


def solve_column_generation(sizes, demands, capacity, time_limit, initial_bins):
    """Column generation approach for bin packing."""
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError:
        return None

    k = len(sizes)
    size_to_idx = {s: i for i, s in enumerate(sizes)}

    # Generate initial patterns from heuristic solution
    patterns_set = set()
    for bin_items in initial_bins:
        pattern = [0] * k
        for item in bin_items:
            if item in size_to_idx:
                pattern[size_to_idx[item]] += 1
        if sum(pattern) > 0:
            patterns_set.add(tuple(pattern))

    # Add single-item patterns
    for i in range(k):
        p = [0] * k
        p[i] = 1
        patterns_set.add(tuple(p))

    patterns = list(patterns_set)
    cg_start = time.time()
    max_cg_time = time_limit * 0.6

    # Column generation loop
    while time.time() - cg_start < max_cg_time:
        num_patterns = len(patterns)

        model = gp.Model("master_lp")
        model.setParam('OutputFlag', 0)
        model.setParam('Threads', 1)

        x = model.addVars(num_patterns, lb=0.0, name="x")
        model.setObjective(gp.quicksum(x[j] for j in range(num_patterns)), GRB.MINIMIZE)

        constrs = []
        for i in range(k):
            constrs.append(model.addConstr(
                gp.quicksum(patterns[j][i] * x[j] for j in range(num_patterns)) >= demands[i]
            ))

        model.optimize()

        if model.status != GRB.OPTIMAL:
            break

        duals = [constrs[i].Pi for i in range(k)]

        new_pattern = solve_pricing_dp(sizes, demands, capacity, duals)

        if new_pattern is None:
            break

        if new_pattern in patterns_set:
            break

        patterns_set.add(new_pattern)
        patterns.append(new_pattern)

    # Solve final IP
    remaining_time = max(1.0, time_limit - (time.time() - cg_start))
    num_patterns = len(patterns)

    model = gp.Model("bin_packing_ip")
    model.setParam('OutputFlag', 0)
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', remaining_time)

    x = model.addVars(num_patterns, vtype=GRB.INTEGER, lb=0, name="x")
    model.setObjective(gp.quicksum(x[j] for j in range(num_patterns)), GRB.MINIMIZE)

    for i in range(k):
        model.addConstr(
            gp.quicksum(patterns[j][i] * x[j] for j in range(num_patterns)) >= demands[i]
        )

    model.optimize()

    if model.SolCount > 0:
        bins = []
        for j in range(num_patterns):
            cnt = int(round(x[j].X))
            for _ in range(cnt):
                bin_items = []
                for i in range(k):
                    bin_items.extend([sizes[i]] * patterns[j][i])
                bins.append(bin_items)
        return bins

    return None


def solve_assignment_ilp(items, capacity, time_limit, upper_bound):
    """Assignment-based ILP for small instances."""
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError:
        return None

    n = len(items)
    UB = upper_bound

    model = gp.Model("bin_packing_assignment")
    model.setParam('OutputFlag', 0)
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', time_limit)

    x = model.addVars(n, UB, vtype=GRB.BINARY, name="x")
    y = model.addVars(UB, vtype=GRB.BINARY, name="y")

    model.setObjective(gp.quicksum(y[j] for j in range(UB)), GRB.MINIMIZE)

    for i in range(n):
        model.addConstr(gp.quicksum(x[i, j] for j in range(UB)) == 1)

    for j in range(UB):
        model.addConstr(
            gp.quicksum(items[i] * x[i, j] for i in range(n)) <= capacity * y[j]
        )

    # Symmetry breaking
    for j in range(UB - 1):
        model.addConstr(y[j] >= y[j + 1])

    model.optimize()

    if model.SolCount > 0:
        bins = [[] for _ in range(UB)]
        for i in range(n):
            for j in range(UB):
                if x[i, j].X > 0.5:
                    bins[j].append(items[i])
                    break
        bins = [b for b in bins if len(b) > 0]
        return bins

    return None


def main():
    parser = argparse.ArgumentParser(description="Bin Packing Problem Solver")
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    start_time = time.time()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    capacity = instance['bin_capacity']
    items = instance['items']
    n = len(items)

    # Count distinct sizes and demands
    size_counts = Counter(items)
    sizes = sorted(size_counts.keys(), reverse=True)
    demands = [size_counts[s] for s in sizes]
    k = len(sizes)

    # Lower bound
    total_size = sum(items)
    lb = math.ceil(total_size / capacity)

    best_bins = None
    best_obj = float('inf')

    def update_best(bins):
        nonlocal best_bins, best_obj
        if bins is not None and len(bins) < best_obj:
            # Verify feasibility
            all_items = []
            for b in bins:
                assert sum(b) <= capacity, f"Bin exceeds capacity: {sum(b)} > {capacity}"
                all_items.extend(b)
            if sorted(all_items) == sorted(items):
                best_bins = bins
                best_obj = len(bins)
                if logger:
                    logger.log_solution(best_obj, format_solution(best_bins))
                return True
        return False

    # Heuristic 1: FFD
    ffd_bins = first_fit_decreasing(items, capacity)
    update_best(ffd_bins)

    # Heuristic 2: BFD
    bfd_bins = best_fit_decreasing(items, capacity)
    update_best(bfd_bins)

    # Heuristic 3: WFD
    wfd_bins = worst_fit_decreasing(items, capacity)
    update_best(wfd_bins)

    if best_obj == lb:
        solution = format_solution(best_bins)
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f)
        return

    # Local search on heuristic solution
    elapsed = time.time() - start_time
    remaining = args.time_limit - elapsed
    if remaining > 1.0:
        ls_bins = local_search(best_bins, capacity, args.time_limit, start_time)
        update_best(ls_bins)

    if best_obj == lb:
        solution = format_solution(best_bins)
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f)
        return

    # Try Gurobi-based methods
    elapsed = time.time() - start_time
    remaining = args.time_limit - elapsed

    if remaining > 2.0:
        gurobi_bins = None

        # Strategy 1: Pattern-based ILP (for small number of distinct sizes)
        if k <= 12:
            try:
                time_for_gurobi = min(remaining - 1.0, args.time_limit * 0.6)
                gurobi_bins = solve_pattern_ilp(sizes, demands, capacity, time_for_gurobi)
                if gurobi_bins is not None:
                    update_best(gurobi_bins)
            except Exception:
                pass

        # Strategy 2: Column generation (for moderate instances)
        if best_obj > lb and time.time() - start_time < args.time_limit - 2:
            try:
                remaining = args.time_limit - (time.time() - start_time)
                if remaining > 3:
                    cg_bins = solve_column_generation(
                        sizes, demands, capacity, remaining - 1.0, best_bins
                    )
                    if cg_bins is not None:
                        update_best(cg_bins)
            except Exception:
                pass

        # Strategy 3: Assignment-based ILP (for very small instances)
        if n <= 60 and best_obj > lb and time.time() - start_time < args.time_limit - 2:
            try:
                remaining = args.time_limit - (time.time() - start_time)
                if remaining > 2:
                    assign_bins = solve_assignment_ilp(items, capacity, remaining - 1.0, best_obj)
                    if assign_bins is not None:
                        update_best(assign_bins)
            except Exception:
                pass

    # Final local search
    elapsed = time.time() - start_time
    remaining = args.time_limit - elapsed
    if remaining > 0.5:
        ls_bins = local_search(best_bins, capacity, args.time_limit, start_time)
        update_best(ls_bins)

    # Write solution
    solution = format_solution(best_bins)
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()
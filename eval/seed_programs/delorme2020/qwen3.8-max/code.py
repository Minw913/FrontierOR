import json
import argparse
import time
import math
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, default=60)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    start_time = time.time()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    n = instance['n']
    m = instance['m']
    c = instance['c']
    weights = instance['weights']
    demands = instance['demands']

    total_weight = sum(weights[i] * demands[i] for i in range(m))
    lb = math.ceil(total_weight / c)

    best_bins = None
    best_num = float('inf')

    def update_solution(bin_assignments):
        nonlocal best_bins, best_num
        # Validate
        for b in bin_assignments:
            if sum(b) > c:
                return
        num = len(bin_assignments)
        if num < best_num:
            best_num = num
            best_bins = [b[:] for b in bin_assignments]
            if logger:
                solution_dict = {
                    "objective_value": float(num),
                    "num_bins": num,
                    "bin_assignments": best_bins
                }
                logger.log_solution(num, solution_dict)

    # Phase 1: Heuristics
    ffd_sol = first_fit_decreasing(weights, demands, c, m)
    update_solution(ffd_sol)

    bfd_sol = best_fit_decreasing(weights, demands, c, m)
    update_solution(bfd_sol)

    # Try a variant: FFD with different tie-breaking
    ffd2_sol = first_fit_decreasing_v2(weights, demands, c, m)
    update_solution(ffd2_sol)

    # Phase 2: Exact/Improved methods
    elapsed = time.time() - start_time
    remaining = args.time_limit - elapsed

    if remaining > 2:
        try:
            # Check if pattern enumeration is feasible
            est_patterns = 1
            for i in range(m):
                est_patterns *= (min(c // weights[i], demands[i]) + 1)
                if est_patterns > 200000:
                    break

            if est_patterns <= 100000:
                patterns = enumerate_patterns(weights, demands, c, m)
                if patterns and len(patterns) > 0:
                    remaining = args.time_limit - (time.time() - start_time)
                    if remaining > 1:
                        solve_pattern_mip(patterns, weights, demands, c, m, remaining, update_solution)
            else:
                # Column generation approach
                remaining = args.time_limit - (time.time() - start_time)
                if remaining > 2:
                    solve_column_generation(weights, demands, c, m, lb, remaining,
                                           update_solution, start_time, args.time_limit)
        except Exception as e:
            pass

    # Phase 3: Local search improvement
    elapsed = time.time() - start_time
    if elapsed < args.time_limit - 0.5 and best_bins is not None:
        improved = local_search(best_bins, weights, demands, c, m,
                               args.time_limit - elapsed, start_time, args.time_limit)
        if improved is not None:
            update_solution(improved)

    # Write final solution
    if best_bins is None:
        # Fallback: put each item in its own bin
        best_bins = [[weights[i]] for i in range(m) for _ in range(demands[i])]
        best_num = len(best_bins)

    solution_dict = {
        "objective_value": float(best_num),
        "num_bins": best_num,
        "bin_assignments": best_bins
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution_dict, f)


def first_fit_decreasing(weights, demands, c, m):
    items = []
    for i in range(m):
        for _ in range(demands[i]):
            items.append(weights[i])
    items.sort(reverse=True)

    bins = []  # [load, items_list]
    for item in items:
        placed = False
        for b in bins:
            if b[0] + item <= c:
                b[0] += item
                b[1].append(item)
                placed = True
                break
        if not placed:
            bins.append([item, [item]])

    return [b[1] for b in bins]


def best_fit_decreasing(weights, demands, c, m):
    items = []
    for i in range(m):
        for _ in range(demands[i]):
            items.append(weights[i])
    items.sort(reverse=True)

    bins = []
    for item in items:
        best_idx = -1
        best_rem = c + 1
        for idx, b in enumerate(bins):
            rem = c - b[0] - item
            if 0 <= rem < best_rem:
                best_rem = rem
                best_idx = idx
        if best_idx >= 0:
            bins[best_idx][0] += item
            bins[best_idx][1].append(item)
        else:
            bins.append([item, [item]])

    return [b[1] for b in bins]


def first_fit_decreasing_v2(weights, demands, c, m):
    """FFD but process by type, placing multiple items at once."""
    # Sort types by weight descending
    type_order = sorted(range(m), key=lambda i: weights[i], reverse=True)

    bins = []  # [load, items_list]
    for i in type_order:
        w = weights[i]
        d = demands[i]
        for _ in range(d):
            placed = False
            # Try to place in bin with least remaining space that still fits
            best_idx = -1
            best_rem = c + 1
            for idx, b in enumerate(bins):
                rem = c - b[0] - w
                if 0 <= rem < best_rem:
                    best_rem = rem
                    best_idx = idx
            if best_idx >= 0:
                bins[best_idx][0] += w
                bins[best_idx][1].append(w)
            else:
                bins.append([w, [w]])

    return [b[1] for b in bins]


def enumerate_patterns(weights, demands, c, m):
    patterns = []

    def gen(idx, rem, pattern):
        if idx == m:
            if any(p > 0 for p in pattern):
                patterns.append(tuple(pattern))
            return
        max_cnt = min(rem // weights[idx], demands[idx])
        for cnt in range(max_cnt + 1):
            pattern.append(cnt)
            gen(idx + 1, rem - cnt * weights[idx], pattern)
            pattern.pop()

    gen(0, c, [])
    return patterns


def solve_pattern_mip(patterns, weights, demands, c, m, time_limit, update_solution):
    import gurobipy as gp
    from gurobipy import GRB

    P = len(patterns)

    model = gp.Model("bpp")
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', time_limit)
    model.setParam('OutputFlag', 0)

    x = model.addVars(P, vtype=GRB.INTEGER, lb=0, name="x")

    model.setObjective(gp.quicksum(x[j] for j in range(P)), GRB.MINIMIZE)

    for i in range(m):
        model.addConstr(
            gp.quicksum(patterns[j][i] * x[j] for j in range(P) if patterns[j][i] > 0) >= demands[i],
            name=f"demand_{i}"
        )

    model.optimize()

    if model.SolCount > 0:
        bin_assignments = []
        for j in range(P):
            val = x[j].X
            if val > 0.5:
                cnt = int(round(val))
                for _ in range(cnt):
                    items_in_bin = []
                    for i in range(m):
                        for _ in range(patterns[j][i]):
                            items_in_bin.append(weights[i])
                    bin_assignments.append(items_in_bin)

        update_solution(bin_assignments)


def solve_column_generation(weights, demands, c, m, lb, time_limit, update_solution, start_time, total_time_limit):
    import gurobipy as gp
    from gurobipy import GRB

    # Initialize patterns
    patterns_set = set()

    # Single-type patterns
    for i in range(m):
        pattern = [0] * m
        pattern[i] = min(demands[i], c // weights[i])
        if sum(pattern) > 0:
            patterns_set.add(tuple(pattern))

    # Patterns from FFD solution
    ffd_sol = first_fit_decreasing(weights, demands, c, m)
    for bin_items in ffd_sol:
        pattern = [0] * m
        for w in bin_items:
            for i in range(m):
                if weights[i] == w:
                    pattern[i] += 1
                    break
        patterns_set.add(tuple(pattern))

    # Patterns from BFD solution
    bfd_sol = best_fit_decreasing(weights, demands, c, m)
    for bin_items in bfd_sol:
        pattern = [0] * m
        for w in bin_items:
            for i in range(m):
                if weights[i] == w:
                    pattern[i] += 1
                    break
        patterns_set.add(tuple(pattern))

    patterns = list(patterns_set)

    # Column generation iterations
    max_iter = 200
    for iteration in range(max_iter):
        if time.time() - start_time > total_time_limit * 0.6:
            break

        P = len(patterns)

        # Solve LP relaxation
        model = gp.Model("cg_master")
        model.setParam('Seed', 0)
        model.setParam('Threads', 1)
        model.setParam('OutputFlag', 0)
        model.setParam('TimeLimit', min(10, time_limit / 3))

        x = model.addVars(P, vtype=GRB.CONTINUOUS, lb=0, name="x")
        model.setObjective(gp.quicksum(x[j] for j in range(P)), GRB.MINIMIZE)

        constraints = []
        for i in range(m):
            con = model.addConstr(
                gp.quicksum(patterns[j][i] * x[j] for j in range(P) if patterns[j][i] > 0) >= demands[i],
                name=f"demand_{i}"
            )
            constraints.append(con)

        model.optimize()

        if model.status != GRB.OPTIMAL:
            break

        # Get duals
        pi = [constraints[i].Pi for i in range(m)]

        # Solve pricing problem
        new_pattern = solve_pricing_problem(weights, demands, c, m, pi)

        if new_pattern is None:
            break

        # Check reduced cost
        rc = 1.0 - sum(pi[i] * new_pattern[i] for i in range(m))
        if rc >= -1e-6:
            break

        new_tuple = tuple(new_pattern)
        if new_tuple in patterns_set:
            break
        patterns_set.add(new_tuple)
        patterns.append(new_tuple)

    # Solve IP with final pattern set
    remaining = total_time_limit - (time.time() - start_time)
    if remaining > 0.5 and len(patterns) > 0:
        solve_pattern_mip(patterns, weights, demands, c, m, remaining, update_solution)


def solve_pricing_problem(weights, demands, c, m, pi):
    """Solve bounded knapsack: max sum(pi[i]*a[i]) s.t. sum(w[i]*a[i]) <= c, 0<=a[i]<=demands[i]"""
    # DP approach: dp[j] = max value with capacity j
    # Use 2D DP for reconstruction
    # dp[i][j] = max value using types 0..i-1 with capacity j

    # For memory efficiency, use rolling array but store choices for reconstruction
    # Actually, let's use full 2D for correctness

    # If c is too large, use a simpler approach
    if c > 50000:
        return solve_pricing_greedy(weights, demands, c, m, pi)

    dp = [[0.0] * (c + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        wi = weights[i - 1]
        pi_i = pi[i - 1]
        di = demands[i - 1]
        if pi_i <= 0:
            for j in range(c + 1):
                dp[i][j] = dp[i - 1][j]
            continue
        for j in range(c + 1):
            dp[i][j] = dp[i - 1][j]
            max_k = min(di, j // wi)
            for k in range(1, max_k + 1):
                val = dp[i - 1][j - k * wi] + k * pi_i
                if val > dp[i][j]:
                    dp[i][j] = val

    if dp[m][c] <= 1.0 + 1e-6:
        return None

    # Reconstruct
    pattern = [0] * m
    j = c
    for i in range(m, 0, -1):
        wi = weights[i - 1]
        pi_i = pi[i - 1]
        di = demands[i - 1]
        best_k = 0
        best_val = dp[i - 1][j]
        max_k = min(di, j // wi)
        for k in range(1, max_k + 1):
            val = dp[i - 1][j - k * wi] + k * pi_i
            if val > best_val + 1e-10:
                best_val = val
                best_k = k
        pattern[i - 1] = best_k
        j -= best_k * wi

    if sum(pattern) == 0:
        return None

    return pattern


def solve_pricing_greedy(weights, demands, c, m, pi):
    """Greedy approach for pricing when c is too large for DP."""
    # Sort by value/weight ratio
    items = [(pi[i] / weights[i], pi[i], weights[i], demands[i], i) for i in range(m) if pi[i] > 0]
    items.sort(reverse=True)

    pattern = [0] * m
    remaining = c
    for ratio, pi_i, wi, di, i in items:
        if remaining < wi:
            continue
        cnt = min(di, remaining // wi)
        if cnt > 0:
            pattern[i] = cnt
            remaining -= cnt * wi

    if sum(pattern) == 0:
        return None

    # Check if reduced cost is negative
    val = sum(pi[i] * pattern[i] for i in range(m))
    if val <= 1.0 + 1e-6:
        return None

    return pattern


def local_search(bin_assignments, weights, demands, c, m, time_limit, start_time, total_time_limit):
    """Try to reduce number of bins by moving items."""
    import random
    random.seed(0)

    bins = [b[:] for b in bin_assignments]
    improved = True

    while improved and (time.time() - start_time < total_time_limit - 0.5):
        improved = False
        num_bins = len(bins)

        # Try to empty a bin by moving its items to other bins
        for i in range(num_bins - 1, -1, -1):
            if time.time() - start_time > total_time_limit - 0.5:
                break

            items_to_move = bins[i][:]
            if not items_to_move:
                bins.pop(i)
                improved = True
                continue

            # Try to redistribute items from bin i to other bins
            other_bins = [j for j in range(len(bins)) if j != i]
            success = try_redistribute(items_to_move, bins, i, other_bins, c)

            if success:
                bins.pop(i)
                improved = True

    return bins if len(bins) < len(bin_assignments) else None


def try_redistribute(items, bins, source_idx, other_indices, capacity):
    """Try to place all items from source bin into other bins."""
    # Sort items by size (largest first for better packing)
    sorted_items = sorted(enumerate(items), key=lambda x: -x[1])

    # Try to place each item
    placements = []  # (item_idx, target_bin_idx)
    temp_loads = {j: sum(bins[j]) for j in other_indices}

    for orig_idx, item in sorted_items:
        placed = False
        # Find best fit
        best_target = -1
        best_rem = capacity + 1
        for j in other_indices:
            rem = capacity - temp_loads[j] - item
            if 0 <= rem < best_rem:
                best_rem = rem
                best_target = j
        if best_target >= 0:
            temp_loads[best_target] += item
            placements.append((orig_idx, best_target))
            placed = True

        if not placed:
            return False

    # Apply placements
    for orig_idx, target in placements:
        bins[target].append(items[orig_idx])

    return True


if __name__ == "__main__":
    main()
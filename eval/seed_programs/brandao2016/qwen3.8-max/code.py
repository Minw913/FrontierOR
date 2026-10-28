import json
import argparse
import time
import sys
from solution_logger import SolutionLogger


def ffd_heuristic(items_expanded, W):
    """First Fit Decreasing heuristic."""
    items_sorted = sorted(items_expanded, key=lambda x: -x[0])
    bins = []
    bin_remaining = []

    for weight, item_type in items_sorted:
        placed = False
        for i in range(len(bins)):
            if bin_remaining[i] >= weight:
                bins[i].append(item_type)
                bin_remaining[i] -= weight
                placed = True
                break
        if not placed:
            bins.append([item_type])
            bin_remaining.append(W - weight)

    return bins


def bfd_heuristic(items_expanded, W):
    """Best Fit Decreasing heuristic."""
    items_sorted = sorted(items_expanded, key=lambda x: -x[0])
    bins = []
    bin_remaining = []

    for weight, item_type in items_sorted:
        best_idx = -1
        best_remaining = W + 1
        for i in range(len(bins)):
            rem = bin_remaining[i] - weight
            if rem >= 0 and rem < best_remaining:
                best_idx = i
                best_remaining = rem
        if best_idx >= 0:
            bins[best_idx].append(item_type)
            bin_remaining[best_idx] -= weight
        else:
            bins.append([item_type])
            bin_remaining.append(W - weight)

    return bins


def wfd_heuristic(items_expanded, W):
    """Worst Fit Decreasing heuristic."""
    items_sorted = sorted(items_expanded, key=lambda x: -x[0])
    bins = []
    bin_remaining = []

    for weight, item_type in items_sorted:
        best_idx = -1
        best_remaining = -1
        for i in range(len(bins)):
            rem = bin_remaining[i] - weight
            if rem >= 0 and rem > best_remaining:
                best_idx = i
                best_remaining = rem
        if best_idx >= 0:
            bins[best_idx].append(item_type)
            bin_remaining[best_idx] -= weight
        else:
            bins.append([item_type])
            bin_remaining.append(W - weight)

    return bins


def local_search(bins, weights, W, time_limit, start_time):
    """Try to reduce number of bins by emptying light bins."""
    while time.time() - start_time < time_limit - 3:
        improved = False

        # Compute bin weights
        bin_weights = [sum(weights[t] for t in b) for b in bins]

        # Sort bins by weight ascending - try to empty lightest bins first
        bin_order = sorted(range(len(bins)), key=lambda i: bin_weights[i])

        for idx in bin_order:
            if time.time() - start_time > time_limit - 3:
                break
            if len(bins[idx]) == 0:
                continue

            # Try to redistribute items from bin idx to other bins
            items_to_move = sorted(bins[idx], key=lambda t: -weights[t])
            other_bins = [i for i in range(len(bins)) if i != idx and len(bins[i]) > 0]

            if not other_bins:
                break

            # Try BFD assignment to other bins
            temp_remaining = {i: W - sum(weights[t] for t in bins[i]) for i in other_bins}
            assignment = {i: [] for i in other_bins}

            success = True
            for item in items_to_move:
                best_i = -1
                best_rem = W + 1
                for i in other_bins:
                    rem = temp_remaining[i] - weights[item]
                    if rem >= 0 and rem < best_rem:
                        best_i = i
                        best_rem = rem
                if best_i >= 0:
                    temp_remaining[best_i] -= weights[item]
                    assignment[best_i].append(item)
                else:
                    success = False
                    break

            if success:
                for i in other_bins:
                    bins[i] = bins[i] + assignment[i]
                bins[idx] = []
                improved = True

        # Remove empty bins
        bins = [b for b in bins if len(b) > 0]

        if not improved:
            break

    return bins


def local_search_pairwise(bins, weights, W, time_limit, start_time):
    """Try to merge pairs of bins."""
    while time.time() - start_time < time_limit - 3:
        improved = False
        bin_weights = [sum(weights[t] for t in b) for b in bins]

        # Sort by weight ascending
        order = sorted(range(len(bins)), key=lambda i: bin_weights[i])

        for i_idx in range(len(order)):
            if time.time() - start_time > time_limit - 3:
                break
            i = order[i_idx]
            if len(bins[i]) == 0:
                continue
            for j_idx in range(i_idx + 1, len(order)):
                if time.time() - start_time > time_limit - 3:
                    break
                j = order[j_idx]
                if len(bins[j]) == 0:
                    continue
                # Try to move items from lighter bin i to bin j
                if bin_weights[i] + bin_weights[j] <= W:
                    bins[j] = bins[j] + bins[i]
                    bins[i] = []
                    improved = True
                    break
                else:
                    # Try partial move
                    temp_items = list(bins[i])
                    temp_items.sort(key=lambda t: -weights[t])
                    remaining_j = W - bin_weights[j]
                    moved_any = False
                    new_bin_i = []
                    for item in temp_items:
                        if remaining_j >= weights[item]:
                            bins[j].append(item)
                            remaining_j -= weights[item]
                            moved_any = True
                        else:
                            new_bin_i.append(item)
                    if moved_any:
                        bins[i] = new_bin_i
                        if len(bins[i]) == 0:
                            improved = True
                            break

        bins = [b for b in bins if len(b) > 0]
        if not improved:
            break

    return bins


def solve_mip_compact(bins_heuristic, weights, demands, W, m, time_remaining, logger=None):
    """Solve using Gurobi compact formulation."""
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError:
        return None

    K = len(bins_heuristic)

    # Lower bound
    total_weight = sum(weights[i] * demands[i] for i in range(1, m + 1))
    LB = (total_weight + W - 1) // W

    if K <= LB:
        return bins_heuristic  # Already optimal

    # Limit K for tractability
    max_bins_for_mip = 300
    if K > max_bins_for_mip:
        K = max_bins_for_mip

    try:
        model = gp.Model("bin_packing")
        model.setParam('Seed', 0)
        model.setParam('MIPGap', 1e-4)
        model.setParam('NumericFocus', 0)
        model.setParam('Threads', 1)
        model.setParam('TimeLimit', max(time_remaining, 1))
        model.setParam('OutputFlag', 0)

        item_types = list(range(1, m + 1))
        bin_indices = list(range(K))

        # Variables
        y = model.addVars(bin_indices, vtype=GRB.BINARY, name="y")
        x = model.addVars(item_types, bin_indices, vtype=GRB.INTEGER, lb=0, name="x")

        # Objective
        model.setObjective(gp.quicksum(y[j] for j in bin_indices), GRB.MINIMIZE)

        # Capacity constraints
        for j in bin_indices:
            model.addConstr(
                gp.quicksum(weights[i] * x[i, j] for i in item_types) <= W * y[j],
                name=f"cap_{j}"
            )

        # Demand constraints
        for i in item_types:
            model.addConstr(
                gp.quicksum(x[i, j] for j in bin_indices) >= demands[i],
                name=f"demand_{i}"
            )

        # Symmetry breaking
        for j in range(K - 1):
            model.addConstr(y[j] >= y[j + 1], name=f"sym_{j}")

        # Upper bounds on x
        for i in item_types:
            max_items_i = min(demands[i], W // weights[i])
            for j in bin_indices:
                model.addConstr(x[i, j] <= max_items_i * y[j], name=f"ub_{i}_{j}")

        # Warm start from heuristic
        for j in range(min(K, len(bins_heuristic))):
            y[j].Start = 1.0
            bin_items = bins_heuristic[j]
            for i in item_types:
                count = bin_items.count(i)
                x[i, j].Start = count
        for j in range(len(bins_heuristic), K):
            y[j].Start = 0.0

        model.optimize()

        if model.SolCount > 0:
            result_bins = []
            for j in bin_indices:
                if y[j].X > 0.5:
                    bin_items = []
                    for i in item_types:
                        count = int(round(x[i, j].X))
                        bin_items.extend([i] * count)
                    if bin_items:
                        result_bins.append(bin_items)

            # Verify solution
            total_packed = {}
            for b in result_bins:
                for item in b:
                    total_packed[item] = total_packed.get(item, 0) + 1

            valid = True
            for i in item_types:
                if total_packed.get(i, 0) < demands[i]:
                    valid = False
                    break
            for b in result_bins:
                if sum(weights[t] for t in b) > W:
                    valid = False
                    break

            if valid:
                return result_bins

        return None
    except Exception as e:
        return None


def solve_mip_patterns(bins_heuristic, weights, demands, W, m, time_remaining, logger=None):
    """Pattern-based formulation with column generation using Gurobi."""
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError:
        return None

    item_types = list(range(1, m + 1))
    K_heuristic = len(bins_heuristic)

    # Lower bound
    total_weight = sum(weights[i] * demands[i] for i in item_types)
    LB = (total_weight + W - 1) // W

    if K_heuristic <= LB:
        return bins_heuristic

    # Generate initial patterns from heuristic solution
    patterns = []
    for b in bins_heuristic:
        pattern = tuple(sorted(b))
        if pattern not in patterns:
            patterns.append(pattern)

    # Add some additional patterns
    # Single item patterns
    for i in item_types:
        max_count = W // weights[i]
        if max_count > 0:
            p = tuple([i] * max_count)
            if p not in patterns:
                patterns.append(p)

    # Column generation loop
    start_cg = time.time()
    max_cg_time = min(time_remaining * 0.6, 60)  # Use at most 60% of time for CG

    while time.time() - start_cg < max_cg_time:
        num_patterns = len(patterns)

        # Build pattern matrix
        # a[p][i] = number of items of type i in pattern p
        a = {}
        for p_idx, pattern in enumerate(patterns):
            for i in item_types:
                a[p_idx, i] = pattern.count(i)

        # Solve LP relaxation
        try:
            model = gp.Model("master_lp")
            model.setParam('OutputFlag', 0)
            model.setParam('Threads', 1)
            model.setParam('TimeLimit', 10)

            x = model.addVars(num_patterns, vtype=GRB.CONTINUOUS, lb=0, name="x")

            model.setObjective(gp.quicksum(x[p] for p in range(num_patterns)), GRB.MINIMIZE)

            demand_constrs = {}
            for i in item_types:
                demand_constrs[i] = model.addConstr(
                    gp.quicksum(a[p, i] * x[p] for p in range(num_patterns) if a[p, i] > 0) >= demands[i],
                    name=f"demand_{i}"
                )

            model.optimize()

            if model.status != GRB.OPTIMAL:
                break

            # Get dual values
            pi = {}
            for i in item_types:
                pi[i] = demand_constrs[i].Pi

            # Pricing problem: find pattern with negative reduced cost
            # min 1 - sum(pi[i] * a[i]) s.t. sum(w[i] * a[i]) <= W, a[i] >= 0 integer
            # Equivalently: max sum(pi[i] * a[i]) s.t. sum(w[i] * a[i]) <= W
            # This is a knapsack problem

            # Solve knapsack using DP
            # dp[c] = max value with capacity c
            dp = [0.0] * (W + 1)
            choice = [[0] * (W + 1) for _ in range(m + 1)]

            for idx, i in enumerate(item_types):
                w_i = weights[i]
                v_i = pi[i]
                if v_i <= 0:
                    continue
                for c in range(W, w_i - 1, -1):
                    if dp[c - w_i] + v_i > dp[c]:
                        dp[c] = dp[c - w_i] + v_i
                        choice[idx + 1][c] = 1

            # Reconstruct pattern
            best_val = dp[W]
            if best_val > 1.0 + 1e-6:  # Negative reduced cost
                # Reconstruct
                c = W
                new_pattern_items = []
                for idx in range(m, 0, -1):
                    i = item_types[idx - 1]
                    w_i = weights[i]
                    while c >= w_i and abs(dp[c] - dp[c - w_i] - pi[i]) < 1e-9:
                        new_pattern_items.append(i)
                        c -= w_i

                new_pattern = tuple(sorted(new_pattern_items))
                if new_pattern and new_pattern not in patterns:
                    patterns.append(new_pattern)
                else:
                    break
            else:
                break

        except Exception:
            break

    # Now solve IP with all patterns
    num_patterns = len(patterns)
    a = {}
    for p_idx, pattern in enumerate(patterns):
        for i in item_types:
            a[p_idx, i] = pattern.count(i)

    try:
        model = gp.Model("master_ip")
        model.setParam('Seed', 0)
        model.setParam('MIPGap', 1e-4)
        model.setParam('NumericFocus', 0)
        model.setParam('Threads', 1)
        remaining_time = time_remaining - (time.time() - start_cg)
        model.setParam('TimeLimit', max(remaining_time, 1))
        model.setParam('OutputFlag', 0)

        x = model.addVars(num_patterns, vtype=GRB.INTEGER, lb=0, name="x")

        model.setObjective(gp.quicksum(x[p] for p in range(num_patterns)), GRB.MINIMIZE)

        for i in item_types:
            model.addConstr(
                gp.quicksum(a[p, i] * x[p] for p in range(num_patterns) if a[p, i] > 0) >= demands[i],
                name=f"demand_{i}"
            )

        model.optimize()

        if model.SolCount > 0:
            result_bins = []
            for p in range(num_patterns):
                count = int(round(x[p].X))
                for _ in range(count):
                    result_bins.append(list(patterns[p]))

            # Verify
            total_packed = {}
            for b in result_bins:
                for item in b:
                    total_packed[item] = total_packed.get(item, 0) + 1

            valid = True
            for i in item_types:
                if total_packed.get(i, 0) < demands[i]:
                    valid = False
                    break
            for b in result_bins:
                if sum(weights[t] for t in b) > W:
                    valid = False
                    break

            if valid and len(result_bins) < K_heuristic:
                return result_bins

        return None
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    start_time = time.time()

    with open(args.instance_path) as f:
        instance = json.load(f)

    n = instance['parameters']['n']
    m = instance['parameters']['m']
    W = instance['parameters']['W']

    weights = {}
    demands = {}
    for item in instance['items']:
        weights[item['item_type']] = item['weight']
        demands[item['item_type']] = item['demand']

    # Expand items
    items_expanded = []
    for it in range(1, m + 1):
        for _ in range(demands[it]):
            items_expanded.append((weights[it], it))

    # Get heuristic solutions
    bins_ffd = ffd_heuristic(items_expanded, W)
    bins_bfd = bfd_heuristic(items_expanded, W)
    bins_wfd = wfd_heuristic(items_expanded, W)

    # Take the best
    candidates = [bins_ffd, bins_bfd, bins_wfd]
    bins = min(candidates, key=lambda b: len(b))

    # Log initial solution
    solution = {
        "objective_value": float(len(bins)),
        "num_bins": len(bins),
        "bins": bins
    }
    if logger:
        logger.log_solution(float(len(bins)), solution)

    # Local search: try to empty bins
    elapsed = time.time() - start_time
    if elapsed < args.time_limit - 5:
        bins = local_search(bins, weights, W, args.time_limit, start_time)

        if len(bins) < solution["num_bins"]:
            solution = {
                "objective_value": float(len(bins)),
                "num_bins": len(bins),
                "bins": bins
            }
            if logger:
                logger.log_solution(float(len(bins)), solution)

    # Pairwise merge local search
    elapsed = time.time() - start_time
    if elapsed < args.time_limit - 5:
        bins = local_search_pairwise(bins, weights, W, args.time_limit, start_time)

        if len(bins) < solution["num_bins"]:
            solution = {
                "objective_value": float(len(bins)),
                "num_bins": len(bins),
                "bins": bins
            }
            if logger:
                logger.log_solution(float(len(bins)), solution)

    # Try MIP if time allows
    elapsed = time.time() - start_time
    time_remaining = args.time_limit - elapsed - 2

    if time_remaining > 5:
        # Choose approach based on problem size
        num_bins_current = len(bins)

        if num_bins_current <= 200 and m <= 50:
            # Use compact formulation
            mip_bins = solve_mip_compact(bins, weights, demands, W, m, time_remaining, logger)
            if mip_bins is not None and len(mip_bins) < len(bins):
                bins = mip_bins
                solution = {
                    "objective_value": float(len(bins)),
                    "num_bins": len(bins),
                    "bins": bins
                }
                if logger:
                    logger.log_solution(float(len(bins)), solution)
        elif num_bins_current <= 500:
            # Try pattern-based approach
            pattern_bins = solve_mip_patterns(bins, weights, demands, W, m, time_remaining, logger)
            if pattern_bins is not None and len(pattern_bins) < len(bins):
                bins = pattern_bins
                solution = {
                    "objective_value": float(len(bins)),
                    "num_bins": len(bins),
                    "bins": bins
                }
                if logger:
                    logger.log_solution(float(len(bins)), solution)

    # Write final solution
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == '__main__':
    main()
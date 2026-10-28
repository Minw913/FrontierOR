import json
import argparse
import time
import math
import random
from copy import deepcopy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    start_time = time.time()
    time_limit = args.time_limit

    # Read instance
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    n = instance['n']
    W = instance['W']
    V = instance['V']
    items = instance['items']
    weights = [item['w'] for item in items]
    volumes = [item['v'] for item in items]

    def elapsed():
        return time.time() - start_time

    def time_remaining():
        return time_limit - elapsed()

    # Lower bound
    total_weight = sum(weights)
    total_volume = sum(volumes)
    lb = max(math.ceil(total_weight / W), math.ceil(total_volume / V))

    def build_solution(bins_list):
        """Build solution dict from list of bins, each bin is list of item indices."""
        solution_bins = []
        for bin_items in bins_list:
            tw = sum(weights[i] for i in bin_items)
            tv = sum(volumes[i] for i in bin_items)
            solution_bins.append({
                "items": bin_items,
                "total_weight": tw,
                "total_volume": tv
            })
        return {
            "objective_value": len(bins_list),
            "bins": solution_bins
        }

    def is_valid(bins_list):
        """Check if solution is valid."""
        all_items = set()
        for bin_items in bins_list:
            tw = sum(weights[i] for i in bin_items)
            tv = sum(volumes[i] for i in bin_items)
            if tw > W or tv > V:
                return False
            for i in bin_items:
                if i in all_items:
                    return False
                all_items.add(i)
        return len(all_items) == n

    # Best Fit Decreasing heuristic
    def solve_bfd(order):
        bins = []  # list of [current_weight, current_volume, [item_indices]]
        for idx in order:
            w, v = weights[idx], volumes[idx]
            best_bin = -1
            best_score = float('inf')
            for i in range(len(bins)):
                if bins[i][0] + w <= W and bins[i][1] + v <= V:
                    # Score: prefer bins with less remaining space (tighter fit)
                    score = (W - bins[i][0] - w) + (V - bins[i][1] - v)
                    if score < best_score:
                        best_score = score
                        best_bin = i
            if best_bin >= 0:
                bins[best_bin][0] += w
                bins[best_bin][1] += v
                bins[best_bin][2].append(idx)
            else:
                bins.append([w, v, [idx]])
        return [b[2] for b in bins]

    # First Fit Decreasing heuristic
    def solve_ffd(order):
        bins = []
        for idx in order:
            w, v = weights[idx], volumes[idx]
            placed = False
            for i in range(len(bins)):
                if bins[i][0] + w <= W and bins[i][1] + v <= V:
                    bins[i][0] += w
                    bins[i][1] += v
                    bins[i][2].append(idx)
                    placed = True
                    break
            if not placed:
                bins.append([w, v, [idx]])
        return [b[2] for b in bins]

    # Generate multiple sorting orders
    sortings = [
        sorted(range(n), key=lambda i: weights[i], reverse=True),
        sorted(range(n), key=lambda i: volumes[i], reverse=True),
        sorted(range(n), key=lambda i: weights[i] + volumes[i], reverse=True),
        sorted(range(n), key=lambda i: weights[i] / W + volumes[i] / V, reverse=True),
        sorted(range(n), key=lambda i: max(weights[i] / W, volumes[i] / V), reverse=True),
        sorted(range(n), key=lambda i: weights[i] * volumes[i], reverse=True),
        sorted(range(n), key=lambda i: weights[i] / W, reverse=True),
        sorted(range(n), key=lambda i: volumes[i] / V, reverse=True),
    ]

    best_bins = None
    best_count = float('inf')

    for order in sortings:
        if time_remaining() < 1:
            break
        for solver in [solve_bfd, solve_ffd]:
            bins = solver(order)
            if len(bins) < best_count:
                best_count = len(bins)
                best_bins = [b[:] for b in bins]

    # Log initial solution
    if best_bins and logger:
        sol = build_solution(best_bins)
        logger.log_solution(sol['objective_value'], sol)

    # Local search: try to reduce number of bins
    def local_search(bins_list, max_time=None):
        """Improve solution by moving items between bins and trying to empty bins."""
        improved = True
        while improved and (max_time is None or time_remaining() > max_time):
            improved = False

            # Try to empty each bin by moving its items to other bins
            for bi in range(len(bins_list)):
                if time_remaining() < 0.5:
                    break
                if len(bins_list) <= lb:
                    break
                bin_items = bins_list[bi][:]
                # Try to redistribute items of bin bi into other bins
                can_empty = True
                temp_bins = [b[:] for b in bins_list if b != bins_list[bi]]
                # Sort items in this bin by difficulty (larger first)
                bin_items_sorted = sorted(bin_items, key=lambda i: weights[i] / W + volumes[i] / V, reverse=True)

                for item in bin_items_sorted:
                    placed = False
                    best_bin_idx = -1
                    best_score = float('inf')
                    for j in range(len(temp_bins)):
                        tw = sum(weights[k] for k in temp_bins[j])
                        tv = sum(volumes[k] for k in temp_bins[j])
                        if tw + weights[item] <= W and tv + volumes[item] <= V:
                            score = (W - tw - weights[item]) + (V - tv - volumes[item])
                            if score < best_score:
                                best_score = score
                                best_bin_idx = j
                    if best_bin_idx >= 0:
                        temp_bins[best_bin_idx].append(item)
                    else:
                        can_empty = False
                        break

                if can_empty:
                    bins_list = temp_bins
                    improved = True
                    break

            # Try moving single items to enable future merges
            if not improved:
                for bi in range(len(bins_list)):
                    if time_remaining() < 0.5:
                        break
                    for ii in range(len(bins_list[bi])):
                        item = bins_list[bi][ii]
                        for bj in range(len(bins_list)):
                            if bj == bi:
                                continue
                            tw = sum(weights[k] for k in bins_list[bj])
                            tv = sum(volumes[k] for k in bins_list[bj])
                            if tw + weights[item] <= W and tv + volumes[item] <= V:
                                # Move item
                                new_bins = [b[:] for b in bins_list]
                                new_bins[bi].remove(item)
                                new_bins[bj].append(item)
                                # Check if source bin is now empty
                                if len(new_bins[bi]) == 0:
                                    new_bins.pop(bi)
                                    bins_list = new_bins
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break

        return bins_list

    best_bins = local_search(best_bins)

    if best_bins and logger:
        sol = build_solution(best_bins)
        if sol['objective_value'] < best_count:
            best_count = sol['objective_value']
            logger.log_solution(sol['objective_value'], sol)

    # More aggressive local search with random restarts
    def random_local_search(bins_list, iterations=1000):
        """Random perturbation + local search."""
        current = [b[:] for b in bins_list]
        best = [b[:] for b in bins_list]
        best_obj = len(best)

        for _ in range(iterations):
            if time_remaining() < 1:
                break

            # Random perturbation: move a random item to a random valid bin
            perturbed = [b[:] for b in current]
            if len(perturbed) < 2:
                break

            # Pick a random bin with items
            non_empty = [i for i in range(len(perturbed)) if len(perturbed[i]) > 0]
            if not non_empty:
                break
            bi = random.choice(non_empty)
            if not perturbed[bi]:
                continue
            item_idx = random.randint(0, len(perturbed[bi]) - 1)
            item = perturbed[bi][item_idx]

            # Find valid destination bins
            valid_dests = []
            for bj in range(len(perturbed)):
                if bj == bi:
                    continue
                tw = sum(weights[k] for k in perturbed[bj])
                tv = sum(volumes[k] for k in perturbed[bj])
                if tw + weights[item] <= W and tv + volumes[item] <= V:
                    valid_dests.append(bj)

            if not valid_dests:
                continue

            bj = random.choice(valid_dests)
            perturbed[bi].pop(item_idx)
            perturbed[bj].append(item)
            if len(perturbed[bi]) == 0:
                perturbed.pop(bi)

            # Apply local search to perturbed solution
            perturbed = local_search(perturbed, max_time=0.5)

            if len(perturbed) < best_obj:
                best_obj = len(perturbed)
                best = [b[:] for b in perturbed]

        return best

    if time_remaining() > 2:
        best_bins = random_local_search(best_bins, iterations=500)
        if best_bins and logger:
            sol = build_solution(best_bins)
            if sol['objective_value'] < best_count:
                best_count = sol['objective_value']
                logger.log_solution(sol['objective_value'], sol)

    # MIP approach for small instances
    def solve_mip(num_bins, time_left):
        """Try to find a feasible solution with num_bins bins using Gurobi."""
        try:
            import gurobipy as gp
            from gurobipy import GRB

            model = gp.Model("bin_packing")
            model.Params.Seed = 0
            model.Params.MIPGap = 1e-4
            model.Params.NumericFocus = 0
            model.Params.Threads = 1
            model.Params.TimeLimit = time_left
            model.Params.OutputFlag = 0

            # Variables
            x = {}
            for i in range(n):
                for j in range(num_bins):
                    x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

            y = {}
            for j in range(num_bins):
                y[j] = model.addVar(vtype=GRB.BINARY, name=f"y_{j}")

            model.update()

            # Objective: minimize number of bins used
            model.setObjective(gp.quicksum(y[j] for j in range(num_bins)), GRB.MINIMIZE)

            # Each item assigned to exactly one bin
            for i in range(n):
                model.addConstr(gp.quicksum(x[i, j] for j in range(num_bins)) == 1)

            # Weight and volume constraints
            for j in range(num_bins):
                model.addConstr(gp.quicksum(weights[i] * x[i, j] for i in range(n)) <= W * y[j])
                model.addConstr(gp.quicksum(volumes[i] * x[i, j] for i in range(n)) <= V * y[j])

            # Symmetry breaking
            for j in range(num_bins - 1):
                model.addConstr(y[j] >= y[j + 1])

            model.optimize()

            if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
                if model.SolCount > 0:
                    bins_result = [[] for _ in range(num_bins)]
                    for i in range(n):
                        for j in range(num_bins):
                            if x[i, j].X > 0.5:
                                bins_result[j].append(i)
                                break
                    # Remove empty bins
                    bins_result = [b for b in bins_result if len(b) > 0]
                    return bins_result
            return None
        except Exception:
            return None

    # Try MIP if instance is small enough
    if n <= 50 and time_remaining() > 5:
        # Try to solve with current best number of bins or fewer
        target_bins = min(best_count, lb + 5)
        for k in range(lb, target_bins + 1):
            if time_remaining() < 2:
                break
            result = solve_mip(k, min(time_remaining() - 1, 30))
            if result is not None and is_valid(result):
                if len(result) < best_count:
                    best_bins = result
                    best_count = len(result)
                    if logger:
                        sol = build_solution(best_bins)
                        logger.log_solution(sol['objective_value'], sol)
                break

    # For medium instances, try MIP with fixed number of bins = current best
    elif n <= 100 and time_remaining() > 10 and best_count <= lb + 3:
        result = solve_mip(best_count, min(time_remaining() - 1, 60))
        if result is not None and is_valid(result):
            if len(result) < best_count:
                best_bins = result
                best_count = len(result)
                if logger:
                    sol = build_solution(best_bins)
                    logger.log_solution(sol['objective_value'], sol)

    # Final validation and output
    if best_bins is None:
        # Fallback: put each item in its own bin
        best_bins = [[i] for i in range(n)]

    # Validate
    assert is_valid(best_bins), "Solution is not valid!"

    solution = build_solution(best_bins)

    # Write solution
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

    if logger:
        logger.log_solution(solution['objective_value'], solution)


if __name__ == "__main__":
    main()
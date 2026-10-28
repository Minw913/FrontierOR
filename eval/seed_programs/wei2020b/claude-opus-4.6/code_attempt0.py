import argparse
import json
import time
import random
from solution_logger import SolutionLogger


def solve(instance_path, solution_path, time_limit, log_path):
    with open(instance_path, 'r') as f:
        data = json.load(f)

    n = data['n']
    W = data['W']
    V = data['V']
    items = data['items']

    start_time = time.time()
    deadline = start_time + time_limit

    logger = SolutionLogger(log_path, sense="minimize") if log_path else None

    if n == 0:
        solution = {"objective_value": 0, "bins": []}
        with open(solution_path, 'w') as f:
            json.dump(solution, f)
        return

    weights = [item['w'] for item in items]
    volumes = [item['v'] for item in items]

    # Lower bound: max of weight-based and volume-based
    total_w = sum(weights)
    total_v = sum(volumes)
    lb = max((total_w + W - 1) // W, (total_v + V - 1) // V, 1)

    # --- First Fit Decreasing heuristic (by weight+volume combined score) ---
    def first_fit_decreasing(order):
        bins_w = []
        bins_v = []
        bins_items = []
        for idx in order:
            w, v = weights[idx], volumes[idx]
            placed = False
            for b in range(len(bins_w)):
                if bins_w[b] + w <= W and bins_v[b] + v <= V:
                    bins_w[b] += w
                    bins_v[b] += v
                    bins_items[b].append(idx)
                    placed = True
                    break
            if not placed:
                bins_w.append(w)
                bins_v.append(v)
                bins_items.append([idx])
        return bins_items

    # Try several orderings for FFD
    indices = list(range(n))

    # Sort by decreasing weight
    order_w = sorted(indices, key=lambda i: weights[i], reverse=True)
    # Sort by decreasing volume
    order_v = sorted(indices, key=lambda i: volumes[i], reverse=True)
    # Sort by decreasing max(w/W, v/V)
    order_max = sorted(indices, key=lambda i: max(weights[i] / W, volumes[i] / V), reverse=True)
    # Sort by decreasing w/W + v/V
    order_sum = sorted(indices, key=lambda i: weights[i] / W + volumes[i] / V, reverse=True)

    best_bins = None
    best_obj = n + 1

    for order in [order_w, order_v, order_max, order_sum]:
        result = first_fit_decreasing(order)
        if len(result) < best_obj:
            best_obj = len(result)
            best_bins = result
            if logger:
                logger.log(best_obj)

    # --- Best Fit Decreasing ---
    def best_fit_decreasing(order):
        bins_w = []
        bins_v = []
        bins_items = []
        for idx in order:
            w, v = weights[idx], volumes[idx]
            best_b = -1
            best_remaining = float('inf')
            for b in range(len(bins_w)):
                if bins_w[b] + w <= W and bins_v[b] + v <= V:
                    # Minimize remaining space (sum of normalized remainders)
                    rem = (W - bins_w[b] - w) / W + (V - bins_v[b] - v) / V
                    if rem < best_remaining:
                        best_remaining = rem
                        best_b = b
            if best_b >= 0:
                bins_w[best_b] += w
                bins_v[best_b] += v
                bins_items[best_b].append(idx)
            else:
                bins_w.append(w)
                bins_v.append(v)
                bins_items.append([idx])
        return bins_items

    for order in [order_w, order_v, order_max, order_sum]:
        result = best_fit_decreasing(order)
        if len(result) < best_obj:
            best_obj = len(result)
            best_bins = result
            if logger:
                logger.log(best_obj)

    if best_obj <= lb:
        # Already optimal
        solution = format_solution(best_bins, weights, volumes)
        with open(solution_path, 'w') as f:
            json.dump(solution, f)
        return

    # --- Local search improvement ---
    # Try to reduce number of bins by redistributing items from least-filled bins

    def try_reduce_bins(current_bins):
        """Try to empty the least-loaded bin by moving its items to other bins."""
        num_bins = len(current_bins)
        if num_bins <= lb:
            return current_bins, False

        # Compute loads
        bin_loads = []
        for b_items in current_bins:
            tw = sum(weights[i] for i in b_items)
            tv = sum(volumes[i] for i in b_items)
            bin_loads.append((tw, tv))

        # Sort bins by total "fullness" ascending - try to empty least full first
        bin_order = sorted(range(num_bins), key=lambda b: bin_loads[b][0] / W + bin_loads[b][1] / V)

        for src_idx in bin_order:
            src_items = current_bins[src_idx]
            if not src_items:
                continue

            # Try to move all items from src to other bins
            # Use a greedy assignment
            other_bins = [b for b in range(num_bins) if b != src_idx]
            other_w = [bin_loads[b][0] for b in other_bins]
            other_v = [bin_loads[b][1] for b in other_bins]

            # Sort items in src by difficulty (largest first)
            src_sorted = sorted(src_items, key=lambda i: max(weights[i] / W, volumes[i] / V), reverse=True)

            assignments = {}
            success = True
            for item in src_sorted:
                w, v = weights[item], volumes[item]
                best_b = -1
                best_rem = float('inf')
                for j, ob in enumerate(other_bins):
                    if other_w[j] + w <= W and other_v[j] + v <= V:
                        rem = (W - other_w[j] - w) / W + (V - other_v[j] - v) / V
                        if rem < best_rem:
                            best_rem = rem
                            best_b = j
                if best_b >= 0:
                    other_w[best_b] += w
                    other_v[best_b] += v
                    assignments[item] = other_bins[best_b]
                else:
                    success = False
                    break

            if success:
                # Apply the moves
                new_bins = [list(b) for b in current_bins]
                new_bins[src_idx] = []
                for item, target in assignments.items():
                    new_bins[target].append(item)
                new_bins = [b for b in new_bins if b]
                return new_bins, True

        return current_bins, False

    improved = True
    while improved and time.time() < deadline:
        best_bins, improved = try_reduce_bins(best_bins)
        if improved:
            best_obj = len(best_bins)
            if logger:
                logger.log(best_obj)
            if best_obj <= lb:
                break

    # --- More aggressive local search: swap items between bins to create space ---
    if best_obj > lb and time.time() < deadline:
        # Try a more sophisticated approach: for each pair of bins, try swapping items
        # to free up space, then try to empty a third bin

        def compute_bin_loads(bins_list):
            loads = []
            for b_items in bins_list:
                tw = sum(weights[i] for i in b_items)
                tv = sum(volumes[i] for i in b_items)
                loads.append((tw, tv))
            return loads

        # Perturbation + reconstruction approach
        random.seed(42)
        best_global_bins = [list(b) for b in best_bins]
        best_global_obj = best_obj

        iteration = 0
        max_no_improve = 200
        no_improve_count = 0

        while time.time() < deadline and best_global_obj > lb and no_improve_count < max_no_improve:
            iteration += 1

            # Generate a random permutation with some structure
            if iteration % 4 == 0:
                order = list(range(n))
                random.shuffle(order)
            elif iteration % 4 == 1:
                # Shuffle within blocks of similar items
                order = sorted(range(n), key=lambda i: weights[i] / W + volumes[i] / V + random.uniform(-0.3, 0.3), reverse=True)
            elif iteration % 4 == 2:
                order = sorted(range(n), key=lambda i: max(weights[i] / W, volumes[i] / V) + random.uniform(-0.2, 0.2), reverse=True)
            else:
                order = sorted(range(n), key=lambda i: min(weights[i] / W, volumes[i] / V) + random.uniform(-0.3, 0.3), reverse=True)

            # Use best fit with this order
            result = best_fit_decreasing(order)
            obj = len(result)

            if obj < best_global_obj:
                best_global_obj = obj
                best_global_bins = result
                if logger:
                    logger.log(best_global_obj)
                no_improve_count = 0

                # Try to reduce further
                improved2 = True
                while improved2 and time.time() < deadline:
                    best_global_bins, improved2 = try_reduce_bins(best_global_bins)
                    if improved2:
                        best_global_obj = len(best_global_bins)
                        if logger:
                            logger.log(best_global_obj)
                        if best_global_obj <= lb:
                            break
            else:
                no_improve_count += 1

        best_bins = best_global_bins
        best_obj = best_global_obj

    # --- Try Gurobi for small instances or if time remains ---
    remaining_time = deadline - time.time()
    if best_obj > lb and remaining_time > 2 and n <= 200:
        try:
            import gurobipy as gp
            from gurobipy import GRB

            num_bins_ub = best_obj
            num_bins_try = best_obj - 1  # Try with one fewer bin

            # Binary search on number of bins
            lo_bins = lb
            hi_bins = best_obj - 1

            while lo_bins <= hi_bins and time.time() < deadline - 1:
                mid_bins = (lo_bins + hi_bins) // 2
                remaining = deadline - time.time() - 0.5
                if remaining < 1:
                    break

                K = mid_bins
                model = gp.Model("binpacking")
                model.setParam('OutputFlag', 0)
                model.setParam('TimeLimit', min(remaining, max(remaining / 2, 10)))
                model.setParam('Threads', 1)

                # x[i,k] = 1 if item i is in bin k
                x = model.addVars(n, K, vtype=GRB.BINARY, name="x")

                # Each item in exactly one bin
                for i in range(n):
                    model.addConstr(gp.quicksum(x[i, k] for k in range(K)) == 1)

                # Weight constraints
                for k in range(K):
                    model.addConstr(gp.quicksum(weights[i] * x[i, k] for i in range(n)) <= W)

                # Volume constraints
                for k in range(K):
                    model.addConstr(gp.quicksum(volumes[i] * x[i, k] for i in range(n)) <= V)

                # Symmetry breaking: bin k must be used if bin k+1 is used
                # Use: item 0 goes to bin 0 (WLOG)
                if K > 0:
                    model.addConstr(x[0, 0] == 1)
                # Lexicographic-like: for bins, enforce ordering by first item
                for k in range(1, K):
                    model.addConstr(gp.quicksum(x[i, k] for i in range(n)) >= gp.quicksum(x[i, k-1] for i in range(n)) * 0)  # weak
                    # Actually just ensure bin k is used only if bin k-1 is used
                    model.addConstr(gp.quicksum(x[i, k] for i in range(n)) <= gp.quicksum(x[i, k-1] for i in range(n)))

                # Provide initial solution from best_bins if K == best_obj - 1
                if K >= len(best_bins):
                    for k, b_items in enumerate(best_bins):
                        for i in b_items:
                            if k < K:
                                x[i, k].Start = 1.0

                model.setParam('MIPFocus', 1)  # Focus on feasibility
                model.optimize()

                if model.Status == GRB.OPTIMAL or (model.Status == GRB.TIME_LIMIT and model.SolCount > 0):
                    # Found a feasible solution with K bins
                    new_bins = [[] for _ in range(K)]
                    for i in range(n):
                        for k in range(K):
                            if x[i, k].X > 0.5:
                                new_bins[k].append(i)
                                break
                    new_bins = [b for b in new_bins if b]
                    if len(new_bins) < best_obj:
                        best_obj = len(new_bins)
                        best_bins = new_bins
                        if logger:
                            logger.log(best_obj)
                    hi_bins = mid_bins - 1
                elif model.Status == GRB.INFEASIBLE:
                    lo_bins = mid_bins + 1
                else:
                    # Timeout without solution - can't conclude, try next
                    lo_bins = mid_bins + 1

        except ImportError:
            pass
        except Exception:
            pass

    # If still above lb and we have enough time, try MIP with minimization objective
    remaining_time = deadline - time.time()
    if best_obj > lb and remaining_time > 2 and n <= 300:
        try:
            import gurobipy as gp
            from gurobipy import GRB

            K = best_obj  # Use current best as upper bound
            model = gp.Model("binpacking_min")
            model.setParam('OutputFlag', 0)
            model.setParam('TimeLimit', remaining_time - 0.5)
            model.setParam('Threads', 1)

            x = model.addVars(n, K, vtype=GRB.BINARY, name="x")
            y = model.addVars(K, vtype=GRB.BINARY, name="y")  # y[k]=1 if bin k is used

            # Minimize number of bins
            model.setObjective(gp.quicksum(y[k] for k in range(K)), GRB.MINIMIZE)

            # Each item in exactly one bin
            for i in range(n):
                model.addConstr(gp.quicksum(x[i, k] for k in range(K)) == 1)

            # Capacity constraints
            for k in range(K):
                model.addConstr(gp.quicksum(weights[i] * x[i, k] for i in range(n)) <= W * y[k])
                model.addConstr(gp.quicksum(volumes[i] * x[i, k] for i in range(n)) <= V * y[k])

            # Symmetry breaking
            for k in range(1, K):
                model.addConstr(y[k] <= y[k - 1])

            # Provide warm start
            for k in range(K):
                if k < len(best_bins):
                    y[k].Start = 1.0
                    for i in best_bins[k]:
                        x[i, k].Start = 1.0
                else:
                    y[k].Start = 0.0

            def callback(model, where):
                if where == GRB.Callback.MIPSOL:
                    obj_val = int(round(model.cbGet(GRB.Callback.MIPSOL_OBJ)))
                    if logger:
                        logger.log(obj_val)

            model.optimize(callback)

            if model.SolCount > 0:
                new_bins = [[] for _ in range(K)]
                for i in range(n):
                    for k in range(K):
                        if x[i, k].X > 0.5:
                            new_bins[k].append(i)
                            break
                new_bins = [b for b in new_bins if b]
                if len(new_bins) < best_obj:
                    best_obj = len(new_bins)
                    best_bins = new_bins
                    if logger:
                        logger.log(best_obj)

        except ImportError:
            pass
        except Exception:
            pass

    solution = format_solution(best_bins, weights, volumes)
    with open(solution_path, 'w') as f:
        json.dump(solution, f)


def format_solution(bins_list, weights, volumes):
    result_bins = []
    for b_items in bins_list:
        if not b_items:
            continue
        tw = sum(weights[i] for i in b_items)
        tv = sum(volumes[i] for i in b_items)
        result_bins.append({
            "items": b_items,
            "total_weight": tw,
            "total_volume": tv
        })
    return {
        "objective_value": len(result_bins),
        "bins": result_bins
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)
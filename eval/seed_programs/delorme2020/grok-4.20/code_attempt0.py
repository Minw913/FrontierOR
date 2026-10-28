import json
import argparse
import time
import sys
from solution_logger import SolutionLogger
import gurobipy as gp
from gurobipy import GRB
from itertools import combinations_with_replacement, product
import math

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    n = instance['n']
    m = instance['m']
    c = instance['c']
    weights = instance['weights']
    demands = instance['demands']

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    start_time = time.time()
    best_ub = float('inf')
    best_packing = None

    # Lower bound
    total_width = sum(w * d for w, d in zip(weights, demands))
    lb = math.ceil(total_width / c)
    best_lb = lb

    # Greedy First Fit Decreasing for initial solution
    items = []
    for i in range(m):
        items.extend([weights[i]] * demands[i])
    items.sort(reverse=True)

    bins = []
    for item in items:
        placed = False
        for b in bins:
            if sum(b) + item <= c:
                b.append(item)
                placed = True
                break
        if not placed:
            bins.append([item])

    initial_bins = len(bins)
    if initial_bins < best_ub:
        best_ub = initial_bins
        best_packing = [sorted(b) for b in bins]
        if logger:
            logger.log(float(best_ub))

    # If small enough, try exact ILP with column generation like patterns but limited
    # Generate feasible patterns (since m and demands can be large, limit to reasonable)
    # Use Gurobi with pattern-based formulation but generate patterns on the fly or limited enumeration

    try:
        model = gp.Model("BinPacking")
        model.setParam(GRB.Param.TimeLimit, max(1, args.time_limit - (time.time() - start_time) - 2))
        model.setParam(GRB.Param.Threads, 1)
        model.setParam(GRB.Param.MIPGap, 0.0)
        model.setParam(GRB.Param.LogToConsole, 0)

        # Use one var per possible item count vector but only if m is small
        # Otherwise use assignment if n is small
        if n <= 40:  # Assignment model for small n
            # Binary variable x[i][k] = 1 if item i in bin k
            # But need to decide max bins
            max_bins = best_ub
            x = {}
            for i in range(n):
                for k in range(max_bins):
                    x[i, k] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{k}")

            y = [model.addVar(vtype=GRB.BINARY, name=f"y_{k}") for k in range(max_bins)]

            # Each item in exactly one bin
            for i in range(n):
                model.addConstr(sum(x[i, k] for k in range(max_bins)) == 1)

            # Capacity
            item_widths = []
            for i in range(m):
                item_widths.extend([weights[i]] * demands[i])

            for k in range(max_bins):
                model.addConstr(sum(item_widths[i] * x[i, k] for i in range(n)) <= c * y[k])

            model.setObjective(sum(y), GRB.MINIMIZE)

            # Callback to log incumbents
            def callback(model, where):
                nonlocal best_ub, best_packing
                if where == GRB.Callback.MIPSOL:
                    obj = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
                    if obj < best_ub - 1e-6:
                        best_ub = obj
                        if logger:
                            logger.log(float(best_ub))
                        # Could extract solution but skip for time

            model.optimize(callback)

            if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SOLUTION_LIMIT):
                if model.objVal < best_ub:
                    best_ub = model.objVal
                    # Reconstruct packing
                    bin_assignments = []
                    if model.status == GRB.OPTIMAL or model.solCount > 0:
                        for k in range(max_bins):
                            if y[k].X > 0.5:
                                bin_items = []
                                for i in range(n):
                                    if x[i, k].X > 0.5:
                                        bin_items.append(item_widths[i])
                                if bin_items:
                                    bin_assignments.append(sorted(bin_items))
                    best_packing = bin_assignments

        else:
            # Larger instances: use column generation like but since no column gen lib, use simple model + heuristic
            # Use Gurobi with a simple relaxation or stick with heuristic and improve with local search

            # Improve initial solution with local search / bin reduction heuristic
            current_packing = [b[:] for b in bins]  # copy
            current_num = len(current_packing)

            improved = True
            while improved and (time.time() - start_time < args.time_limit - 5):
                improved = False
                # Try to find two bins that can be merged into one or redistribute
                for i in range(len(current_packing)):
                    for j in range(i+1, len(current_packing)):
                        bin1 = current_packing[i]
                        bin2 = current_packing[j]
                        if sum(bin1) + sum(bin2) <= c:
                            # Can merge
                            merged = sorted(bin1 + bin2)
                            new_packing = [current_packing[k] for k in range(len(current_packing)) if k != i and k != j]
                            new_packing.append(merged)
                            current_packing = new_packing
                            current_num -= 1
                            improved = True
                            if current_num < best_ub:
                                best_ub = current_num
                                best_packing = [sorted(b) for b in current_packing]
                                if logger:
                                    logger.log(float(best_ub))
                            break
                    if improved:
                        break
                if not improved:
                    # Simple First Fit improvement by reshuffling
                    all_items = []
                    for b in current_packing:
                        all_items.extend(b)
                    all_items.sort(reverse=True)
                    new_bins = []
                    for item in all_items:
                        placed = False
                        for b_idx in range(len(new_bins)):
                            if sum(new_bins[b_idx]) + item <= c:
                                new_bins[b_idx].append(item)
                                placed = True
                                break
                        if not placed:
                            new_bins.append([item])
                    if len(new_bins) < current_num:
                        current_packing = new_bins
                        current_num = len(new_bins)
                        improved = True
                        if current_num < best_ub:
                            best_ub = current_num
                            best_packing = [sorted(b) for b in current_packing]
                            if logger:
                                logger.log(float(best_ub))

            # Try to use Gurobi only for bounding if possible
            if time.time() - start_time < args.time_limit - 10 and best_ub > lb:
                # Simple LP relaxation for lower bound
                try:
                    rel = gp.Model("Relax")
                    rel.setParam(GRB.Param.TimeLimit, 30)
                    rel.setParam(GRB.Param.Threads, 1)
                    rel.setParam(GRB.Param.OutputFlag, 0)
                    # Very crude: just use variables per type per bin is too big
                    # Use fractional bin count per pattern is complex
                    # Skip advanced and use current best
                    pass
                except:
                    pass

    except Exception as e:
        # If solver fails, return best from heuristic
        pass

    # Final best_packing should exist from heuristic at minimum
    if best_packing is None:
        best_packing = [sorted(b) for b in bins]
        best_ub = len(best_packing)

    objective = float(best_ub)
    num_bins = int(round(best_ub))

    solution = {
        "objective_value": objective,
        "num_bins": num_bins,
        "bin_assignments": best_packing
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

    remaining = args.time_limit - (time.time() - start_time)
    if remaining > 0:
        time.sleep(min(0.1, remaining))

if __name__ == "__main__":
    main()
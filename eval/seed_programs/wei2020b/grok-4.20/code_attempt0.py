import json
import argparse
import time
import sys
from solution_logger import SolutionLogger
import gurobipy as gp
from gurobipy import GRB

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Read instance
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    n = instance['n']
    W = instance['W']
    V = instance['V']
    items = instance['items']
    weights = [item['w'] for item in items]
    volumes = [item['v'] for item in items]

    # Upper bound on bins
    max_bins = n
    # Lower bound
    lower_bound = max(
        (sum(weights) + W - 1) // W,
        (sum(volumes) + V - 1) // V
    )

    start_time = time.time()
    best_obj = n + 1
    best_solution = None

    # Try to solve with increasing number of bins
    for k in range(lower_bound, max_bins + 1):
        if time.time() - start_time > args.time_limit:
            break

        remaining_time = max(1, int(args.time_limit - (time.time() - start_time)))
        
        try:
            model = gp.Model("BinPacking")
            model.Params.TimeLimit = remaining_time
            model.Params.Threads = 1
            model.Params.MIPFocus = 1
            model.Params.LogToConsole = 0

            # Variables
            x = model.addVars(n, k, vtype=GRB.BINARY, name="x")
            y = model.addVars(k, vtype=GRB.BINARY, name="y")

            # Objective
            model.setObjective(gp.quicksum(y[j] for j in range(k)), GRB.MINIMIZE)

            # Each item in exactly one bin
            for i in range(n):
                model.addConstr(gp.quicksum(x[i, j] for j in range(k)) == 1)

            # Capacity constraints
            for j in range(k):
                model.addConstr(gp.quicksum(weights[i] * x[i, j] for i in range(n)) <= W * y[j])
                model.addConstr(gp.quicksum(volumes[i] * x[i, j] for i in range(n)) <= V * y[j])

            # If using fewer than k bins is possible, we let solver decide via y variables

            # Callback to log improved solutions
            def callback(model, where):
                nonlocal best_obj, best_solution
                if where == GRB.Callback.MIPSOL:
                    obj = int(round(model.cbGet(GRB.Callback.MIPSOL_OBJBST)))
                    if obj < best_obj:
                        best_obj = obj
                        if logger:
                            logger.log(best_obj)
                        # Extract solution
                        sol_x = model.cbGetSolution(x)
                        bins = []
                        for j in range(k):
                            if sum(sol_x[i, j] for i in range(n)) > 0.5:
                                bin_items = [i for i in range(n) if sol_x[i, j] > 0.5]
                                tw = sum(weights[i] for i in bin_items)
                                tv = sum(volumes[i] for i in bin_items)
                                bins.append({
                                    "items": bin_items,
                                    "total_weight": tw,
                                    "total_volume": tv
                                })
                        nonlocal best_solution
                        best_solution = {
                            "objective_value": obj,
                            "bins": bins
                        }

            model.optimize(callback)

            if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT):
                if model.objVal < best_obj - 0.5:
                    best_obj = int(round(model.objVal))
                    # Extract best solution
                    bins = []
                    for j in range(k):
                        if y[j].X > 0.5:
                            bin_items = [i for i in range(n) if x[i, j].X > 0.5]
                            if bin_items:
                                tw = sum(weights[i] for i in bin_items)
                                tv = sum(volumes[i] for i in bin_items)
                                bins.append({
                                    "items": bin_items,
                                    "total_weight": tw,
                                    "total_volume": tv
                                })
                    best_solution = {
                        "objective_value": best_obj,
                        "bins": bins
                    }
                    if logger:
                        logger.log(best_obj)

            # If we found optimal for current k, no need to try larger k
            if model.status == GRB.OPTIMAL:
                break

        except Exception as e:
            continue

    # If no solution found yet, use a greedy heuristic as fallback
    if best_solution is None:
        # Simple First Fit Decreasing heuristic
        # Sort items by volume + weight
        indexed_items = list(range(n))
        indexed_items.sort(key=lambda i: -(weights[i] * V + volumes[i] * W))

        bins = []
        bin_weights = []
        bin_volumes = []

        for i in indexed_items:
            placed = False
            for b in range(len(bins)):
                if bin_weights[b] + weights[i] <= W and bin_volumes[b] + volumes[i] <= V:
                    bins[b].append(i)
                    bin_weights[b] += weights[i]
                    bin_volumes[b] += volumes[i]
                    placed = True
                    break
            if not placed:
                bins.append([i])
                bin_weights.append(weights[i])
                bin_volumes.append(volumes[i])

        best_obj = len(bins)
        solution_bins = []
        for b_idx, bin_items in enumerate(bins):
            solution_bins.append({
                "items": sorted(bin_items),
                "total_weight": bin_weights[b_idx],
                "total_volume": bin_volumes[b_idx]
            })
        best_solution = {
            "objective_value": best_obj,
            "bins": solution_bins
        }
        if logger:
            logger.log(best_obj)

    # Ensure solution is valid and write output
    if best_solution is None:
        # Extremely unlikely fallback
        best_solution = {
            "objective_value": n,
            "bins": [{"items": [i], "total_weight": weights[i], "total_volume": volumes[i]} for i in range(n)]
        }

    with open(args.solution_path, 'w') as f:
        json.dump(best_solution, f, indent=2)

if __name__ == "__main__":
    main()
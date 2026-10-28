import json
import argparse
import time
import sys
from typing import List, Dict, Any
from solution_logger import SolutionLogger

try:
    from gurobipy import Model, GRB, quicksum
except ImportError:
    print("Gurobi not available. Please ensure gurobipy is installed.")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    # Load instance
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    W = instance['bin_capacity']
    items = instance['items']
    n = len(items)
    if n == 0:
        solution = {
            "objective_value": 0,
            "bin_assignments": [],
            "num_bins": 0
        }
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
        return

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Upper bound: First Fit Decreasing
    sorted_items = sorted(items, reverse=True)
    bins = []
    for size in sorted_items:
        placed = False
        for b in bins:
            if b['remaining'] >= size:
                b['items'].append(size)
                b['remaining'] -= size
                placed = True
                break
        if not placed:
            bins.append({'items': [size], 'remaining': W - size})
    ub = len(bins)
    best_obj = ub
    best_assignment = None

    if logger:
        logger.log(best_obj)

    # Lower bound: ceil(total_size / W)
    total_size = sum(items)
    lb = (total_size + W - 1) // W
    if lb == ub:
        # Optimal
        construct_solution(bins, W, args.solution_path, best_obj)
        return

    # Try to solve with Gurobi Column Generation / MIP with time limit
    time_start = time.time()
    remaining_time = args.time_limit - (time.time() - time_start)
    if remaining_time <= 0:
        construct_solution(bins, W, args.solution_path, best_obj)
        return

    # Model: assign items to bins with limit on number of bins
    model = Model("BinPacking")
    model.setParam('TimeLimit', remaining_time)
    model.setParam('Threads', 1)
    model.setParam('MIPGap', 1e-4)
    model.setParam('LogToConsole', 0)

    # We use a formulation with possible bins up to ub
    max_bins = ub
    x = model.addVars(n, max_bins, vtype=GRB.BINARY, name="x")
    y = model.addVars(max_bins, vtype=GRB.BINARY, name="y")

    # Each item in exactly one bin
    for i in range(n):
        model.addConstr(quicksum(x[i, j] for j in range(max_bins)) == 1)

    # Capacity
    for j in range(max_bins):
        model.addConstr(quicksum(items[i] * x[i, j] for i in range(n)) <= W * y[j])

    # Objective
    model.setObjective(quicksum(y[j] for j in range(max_bins)), GRB.MINIMIZE)

    # Callback to log improving solutions
    def callback(model, where):
        nonlocal best_obj, best_assignment
        if where == GRB.Callback.MIPSOL:
            obj = int(round(model.cbGet(GRB.Callback.MIPSOL_OBJBST)))
            if obj < best_obj:
                best_obj = obj
                # Extract solution
                sol_x = model.cbGetSolution(x)
                assignment = []
                for j in range(max_bins):
                    if sol_x[0, j] > 0.5 or any(sol_x[i, j] > 0.5 for i in range(n)):
                        bin_items = [items[i] for i in range(n) if sol_x[i, j] > 0.5]
                        assignment.append({
                            'items': bin_items,
                            'total_size': sum(bin_items)
                        })
                best_assignment = assignment
                if logger:
                    logger.log(best_obj)

    model.optimize(callback)

    if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED):
        if model.SolCount > 0:
            current_obj = int(model.objVal)
            if current_obj < best_obj:
                best_obj = current_obj
                # Extract final solution
                assignment = []
                for j in range(max_bins):
                    if y[j].X > 0.5:
                        bin_items = [items[i] for i in range(n) if x[i, j].X > 0.5]
                        if bin_items:
                            assignment.append({
                                'items': bin_items,
                                'total_size': sum(bin_items)
                            })
                best_assignment = assignment

    # If we didn't get a solution from MIP, use heuristic
    if best_assignment is None:
        best_assignment = [
            {'items': b['items'], 'total_size': W - b['remaining']} for b in bins
        ]

    construct_solution(best_assignment, W, args.solution_path, best_obj)


def construct_solution(bin_assignments: List[Dict], W: int, solution_path: str, obj: int):
    solution = {
        "objective_value": obj,
        "bin_assignments": bin_assignments,
        "num_bins": len(bin_assignments)
    }
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()
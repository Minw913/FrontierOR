import argparse
import heapq
import json
import time
import gurobipy as grb
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()

    # Read instance data
    with open(args.instance_path, 'r') as f:
        data = json.load(f)
    n = data['n']
    m = data['m']
    c = data['c']
    weights = data['weights']
    demands = data['demands']

    # Run heuristic to get initial solution and upper bound
    total_bins_heuristic = 0
    bin_counts_heuristic = []  # List to store counts per bin per item type
    heap = []  # Min-heap for (-remaining_capacity, bin_index)

    # Sort item types by weight descending
    item_types = sorted(range(m), key=lambda i: weights[i], reverse=True)

    for i in item_types:
        w = weights[i]
        d = demands[i]
        temp = []  # Store bins popped that are too small

        # Assign to existing non-full bins
        while d > 0 and heap:
            cap_neg, bin_idx = heapq.heappop(heap)
            cap = -cap_neg
            if cap < w:
                temp.append((cap_neg, bin_idx))
            else:
                num = min(cap // w, d)
                bin_counts_heuristic[bin_idx][i] += num
                new_cap = cap - num * w
                d -= num
                if new_cap > 0:
                    heapq.heappush(heap, (-new_cap, bin_idx))

        # Push back bins that were too small
        for item in temp:
            heapq.heappush(heap, item)

        # Open new bins for remaining demand
        while d > 0:
            num = min(c // w, d)
            bin_idx = total_bins_heuristic
            total_bins_heuristic += 1
            # Initialize new bin with zeros for all item types
            bin_counts_heuristic.append([0] * m)
            bin_counts_heuristic[bin_idx][i] = num
            new_cap = c - num * w
            d -= num
            if new_cap > 0:
                heapq.heappush(heap, (-new_cap, bin_idx))

    # Expand bin_counts to get bin_assignments_heuristic
    bin_assignments_heuristic = []
    for j in range(total_bins_heuristic):
        bin_j = []
        for i in range(m):
            bin_j.extend([weights[i]] * bin_counts_heuristic[j][i])
        bin_assignments_heuristic.append(bin_j)

    elapsed = time.time() - start_time
    time_remaining = args.time_limit - elapsed

    # If no time left, return heuristic solution
    if time_remaining <= 0:
        bins_used = total_bins_heuristic
        bin_assignments = bin_assignments_heuristic
    else:
        # Initialize logger if provided
        logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
        
        try:
            model = grb.Model()
            model.setParam('OutputFlag', 0)
            model.setParam('TimeLimit', time_remaining)
            B = total_bins_heuristic  # Upper bound on bins

            # Variables
            y = model.addVars(B, vtype=grb.GRB.BINARY, name="y")
            x = model.addVars(m, B, vtype=grb.GRB.INTEGER, name="x")
            model.update()

            # Constraints
            for j in range(B):
                model.addConstr(
                    grb.quicksum(weights[i] * x[i, j] for i in range(m)) <= c * y[j],
                    f"capacity_{j}"
                )
            
            for i in range(m):
                model.addConstr(
                    grb.quicksum(x[i, j] for j in range(B)) == demands[i],
                    f"demand_{i}"
                )

            # Objective: minimize number of bins used
            model.setObjective(grb.quicksum(y[j] for j in range(B)), grb.GRB.MINIMIZE)

            # Set initial solution from heuristic
            for j in range(B):
                y[j].Start = 1  # All B bins are initially used in heuristic
            for i in range(m):
                for j in range(B):
                    if j < len(bin_counts_heuristic):
                        x[i, j].Start = bin_counts_heuristic[j][i]
                    else:
                        x[i, j].Start = 0

            # Callback for logging
            def callback(model, where):
                if where == grb.GRB.Callback.MIPSOL:
                    objval = model.cbGet(grb.GRB.Callback.MIPSOL_OBJ)
                    if logger:
                        logger.log(objval)

            # Solve
            if logger:
                model.optimize(callback)
            else:
                model.optimize()

            # Extract solution if found
            if model.SolCount > 0:
                bins_used = 0
                bin_assignments = []
                for j in range(B):
                    if y[j].X > 0.5:  # Bin j is used
                        bins_used += 1
                        bin_j = []
                        for i in range(m):
                            count = int(round(x[i, j].X))
                            bin_j.extend([weights[i]] * count)
                        bin_assignments.append(bin_j)
                # Ensure we use the best solution found
                if bins_used > total_bins_heuristic:
                    bins_used = total_bins_heuristic
                    bin_assignments = bin_assignments_heuristic
            else:
                bins_used = total_bins_heuristic
                bin_assignments = bin_assignments_heuristic

        except (grb.GurobiError, AttributeError):
            bins_used = total_bins_heuristic
            bin_assignments = bin_assignments_heuristic

    # Write solution to file
    solution = {
        "objective_value": float(bins_used),
        "num_bins": bins_used,
        "bin_assignments": bin_assignments
    }
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == "__main__":
    main()
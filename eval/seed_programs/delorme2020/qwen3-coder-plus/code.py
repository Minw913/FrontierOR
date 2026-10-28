import argparse
import json
import math
import time
from solution_logger import SolutionLogger


def solve_bin_packing(instance, time_limit):
    c = instance['c']
    weights = instance['weights']
    demands = instance['demands']
    n_items = len(weights)

    # Initialize result
    best_solution = []
    min_bins = float('inf')

    # Get a lower bound for pruning
    total_load = sum(w * d for w, d in zip(weights, demands))
    lower_bound = math.ceil(total_load / c)

    # Try First-Fit Decreasing heuristic and more
    start_time = time.time()

    def generate_items_list():
        items = []
        for i in range(n_items):
            for j in range(demands[i]):
                items.append(weights[i])
        return items

    def first_fit_decreasing(items):
        items_sorted = sorted(items, reverse=True)
        bins_used = []

        for item in items_sorted:
            placed = False
            # Try to place item in existing bins
            for bin_idx in range(len(bins_used)):
                if sum(bins_used[bin_idx]) + item <= c:
                    bins_used[bin_idx].append(item)
                    placed = True
                    break
            # If not placed, create a new bin
            if not placed:
                bins_used.append([item])
        return bins_used

    # Run FFD to get upper bound and initial solution
    all_items = generate_items_list()
    ffd_solution = first_fit_decreasing(all_items)
    current_best_bin_count = len(ffd_solution)
    best_solution = [list(bin_content) for bin_content in ffd_solution]
    min_bins = current_best_bin_count

    # Define improvement algorithms
    def swap_improvement(current_bins):
        improved = False
        while True:
            improved_locally = False
            for bin_a_idx in range(len(current_bins)):
                for bin_b_idx in range(bin_a_idx + 1, len(current_bins)):
                    # Try moving items from bin_a to bin_b
                    for i, item in enumerate(current_bins[bin_a_idx]):
                        if sum(current_bins[bin_b_idx]) + item <= c:
                            new_bin_b = current_bins[bin_b_idx] + [item]
                            new_bin_a = current_bins[bin_a_idx][:i] + current_bins[bin_a_idx][i+1:]
                            
                            # Check if we can merge two bins
                            temp_bins = current_bins[:]
                            merged_bins = temp_bins[:bin_a_idx] + temp_bins[bin_a_idx+1:bin_b_idx] + \
                                          temp_bins[bin_b_idx+1:] + [new_bin_a, new_bin_b] if new_bin_a else \
                                          [new_bin_b] + [temp_bins[:bin_a_idx] + temp_bins[bin_a_idx+1:bin_b_idx] + temp_preds + temp_bins[bin_b_idx+1:]]

                            # Clean up empty bins
                            new_bins = [bin_list for bin_list in temp_bins if len(temp_bins[bin_idx]) > 0]
                            if len(new_bins) < len(current_bins):
                                current_bins = new_bins
                                improved = True
                                improved_locally = True
                                break
                    if improved_locally:
                        break
                if improved_locally:
                    break
            if not improved_locally:
                break
        
        return current_bins

    # Now try more sophisticated greedy approaches to improve solution
    def best_fit_decreasing(items):
        items_sorted = sorted(items, reverse=True)
        bins_used = []

        for item in items_sorted:
            best_bin_idx = -1
            min_space_left = c + 1  # Initialize with impossible value

            for bin_idx in range(len(bins_used)):
                space_left = c - sum(bins_used[bin_idx])
                if space_left >= item and space_left < min_space_left:
                    best_bin_idx = bin_idx
                    min_space_left = space_left

            if best_bin_idx != -1:
                bins_used[best_bin_idx].append(item)
            else:
                bins_used.append([item])
        
        return bins_used

    bbd_solution = best_fit_decreasing(all_items)
    if len(bbd_solution) < current_best_bin_count:
        current_best_bin_count = len(bbd_solution)
        best_solution = [list(bin_content) for bin_content in bbd_solution]
        min_bins = current_best_bin_count

    # Try next-fit-decreasing for comparison
    def next_fit_decreasing(items):
        items_sorted = sorted(items, reverse=True)
        bins_used = []

        if not items_sorted:
            return []

        current_bin = [items_sorted[0]]
        for item in items_sorted[1:]:
            if sum(current_bin) + item <= c:
                current_bin.append(item)
            else:
                bins_used.append(current_bin)
                current_bin = [item]
        bins_used.append(current_bin)
        return bins_used

    nfd_solution = next_fit_decreasing(all_items)
    if len(nfd_solution) < current_best_bin_count:
        current_best_bin_count = len(nfd_solution)
        best_solution = [list(bin_content) for bin_content in nfd_solution]
        min_bins = current_best_bin_count

    return {
        "objective_value": min_bins,
        "num_bins": min_bins,
        "bin_assignments": best_solution
    }, min_bins


def main():
    parser = argparse.ArgumentParser(description='Solve the Bin Packing Problem')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the instance file')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to the solution file')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds')
    parser.add_argument('--log_path', type=str, default=None, help='Path to the log file')
    
    args = parser.parse_args()

    # Load instance
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    # Setup logger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    start_time = time.time()

    # Solve the problem
    try:
        solution, _ = solve_bin_packing(instance, args.time_limit)
        if logger:
            logger.log(solution["objective_value"])
    except Exception as e:
        print(f"Error occurred during solving: {e}")
        raise

    # Write solution
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == '__main__':
    main()
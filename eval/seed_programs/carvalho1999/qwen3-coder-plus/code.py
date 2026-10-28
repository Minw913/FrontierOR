import argparse
import json
import math
from collections import Counter
from solution_logger import SolutionLogger
from typing import List, Tuple


def first_fit_decreasing(items: List[int], bin_capacity: int) -> Tuple[int, List[dict]]:
    """First Fit Decreasing heuristic for Bin Packing"""
    # Sort items in decreasing order keeping track of original indices (we just need sizes here)
    sorted_items = sorted(items, reverse=True)
    
    bins_total_size = []  # Total sizes of active bins
    bin_contents = []  # Contents of active bins

    for item in sorted_items:
        placed = False
        
        # Try placing the item in existing bins
        for i, current_size in enumerate(bins_total_size):
            if current_size + item <= bin_capacity:
                bins_total_size[i] += item
                bin_contents[i].append(item)
                placed = True
                break
        
        # If not placed, create a new bin
        if not placed:
            bins_total_size.append(item)
            bin_contents.append([item])
    
    result_bins = []
    for i in range(len(bin_contents)):
        result_bins.append({
            'items': bin_contents[i],
            'total_size': bins_total_size[i]
        })
    
    return len(result_bins), result_bins


def solve_with_heuristics(items: List[int], bin_capacity: int, time_limit: int, logger=None):
    """
    Enhanced heuristic solution incorporating several classical heuristics
    since Gurobi may not be available in all environments.
    """
    # Start with First-Fit Decreasing
    best_num_bins, best_bins = first_fit_decreasing(items, bin_capacity)
    
    # Log the initial solution
    if logger:
        logger.log(best_num_bins)
    
    # Apply Best-Fit Decreasing as well
    sorted_items = sorted(items, reverse=True)
    
    bin_spaces = [bin_capacity for _ in range(len(sorted_items))]
    bin_contents = [[] for _ in range(len(sorted_items))]
    
    for item in sorted_items:
        best_bin_idx = -1
        min_space_left = bin_capacity + 1
        
        for i, space in enumerate(bin_spaces):
            if space >= item and space - item < min_space_left:
                min_space_left = space - item
                best_bin_idx = i
        
        if best_bin_idx != -1:
            bin_spaces[best_bin_idx] -= item
            bin_contents[best_bin_idx].append(item)
        else:
            # Add new bin
            idx = len(bin_spaces)  # Find first empty slot among the extra bins
            if len(bin_spaces) < len(sorted_items):
                bin_spaces.append(bin_capacity - item)
                bin_contents.append([item])
            else:
                # We shouldn't reach here but just in case
                break

    # Remove unused bins and construct solution
    used_bins = [(space, content) for space, content in zip(bin_spaces, bin_contents) if content]
    num_bins_bfd = len(used_bins)
    bins_bfd = [{'items': content, 'total_size': bin_capacity - space}
                for space, content in used_bins]

    # Check if BFD provided better solution than FFD
    if num_bins_bfd < best_num_bins:
        best_num_bins = num_bins_bfd
        best_bins = bins_bfd
        if logger:
            logger.log(best_num_bins)
    
    return best_num_bins, best_bins


def main():
    parser = argparse.ArgumentParser(description='Solve Bin Packing Problem')
    parser.add_argument('--instance_path', type=str, required=True,
                        help='Path to the JSON file containing the problem instance.')
    parser.add_argument('--solution_path', type=str, required=True,
                        help='Path where the final solution JSON file must be written.')
    parser.add_argument('--time_limit', type=int, required=True,
                        help='Maximum runtime in seconds.')
    parser.add_argument('--log_path', type=str, default=None,
                        help='Path to a JSONL file for logging intermediate solutions (optional).')

    args = parser.parse_args()

    # Read instance data
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)

    bin_capacity = instance_data['bin_capacity']
    items = instance_data['items']

    # Initialize logger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Solve the problem using heuristics
    num_bins, bin_assignments = solve_with_heuristics(
        items, bin_capacity, args.time_limit, logger
    )

    # Prepare solution object
    solution = {
        "objective_value": num_bins,
        "bin_assignments": bin_assignments,
        "num_bins": num_bins
    }

    # Write solution to file
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == '__main__':
    main()
import argparse
import json
import sys
from itertools import permutations
from collections import defaultdict
import heapq
import math

def solve_2d_bin_packing(instance):
    """
    Solve the 2D Bin Packing problem with weight and volume constraints.
    This is a heuristic implementation with multiple techniques applied.
    """
    n = instance["n"]
    W = instance["W"]  # Weight capacity per bin
    V = instance["V"]  # Volume capacity per bin
    items = instance["items"]

    # Precompute all ratios and properties of each item
    for i, item in enumerate(items):
        item['id'] = i
        item['efficiency'] = (item['w'] + item['v']) / (W + V)
        item['weight_ratio'] = item['w'] / W
        item['volume_ratio'] = item['v'] / V
        
    # Sort items in descending order by utilization ratio (density)
    sorted_items = sorted(enumerate(items), key=lambda x: max(x[1]['weight_ratio'], x[1]['volume_ratio']), reverse=True)
    
    def calculate_utilization(bin_data):
        """Calculate how full a bin is compared to its capacities."""
        total_w = bin_data['total_weight']
        total_v = bin_data['total_volume']
        util_weight = total_w / W
        util_volume = total_v / V
        return max(util_weight, util_volume)
    
    # Best-first greedy approach with different orders
    best_solution = None
    best_bins_count = float('inf')
    
    # Define several heuristics to try
    def get_sorted_items_by_weight():
        return sorted(enumerate(items), key=lambda x: x[1]['w'], reverse=True)
        
    def get_sorted_items_by_volume():
        return sorted(enumerate(items), key=lambda x: x[1]['v'], reverse=True)
        
    def get_sorted_items_by_sum():
        return sorted(enumerate(items), key=lambda x: x[1]['w'] + x[1]['v'], reverse=True)
        
    def get_sorted_items_by_max():
        return sorted(enumerate(items), key=lambda x: max(x[1]['w']/W, x[1]['v']/V), reverse=True)
        
    def get_sorted_items_random():
        ids = list(range(len(items)))
        import random
        random.shuffle(ids)
        return [(i, items[i]) for i in ids]
    
    heuristics = [
        get_sorted_items_by_weight,
        get_sorted_items_by_volume,
        get_sorted_items_by_sum,
        get_sorted_items_by_max
    ]
    
    for heuristic in heuristics:
        # First fit decreasing
        remaining_items = [item_id for item_id, _ in heuristic()]
        bins = []
        placed = [False] * n
        
        for item_id in remaining_items:
            item = items[item_id]
            
            # Try to place the item in existing bins
            placed_in_bin = False
            for bin_idx in range(len(bins)):
                bin_data = bins[bin_idx]
                if bin_data['total_weight'] + item['w'] <= W and \
                   bin_data['total_volume'] + item['v'] <= V:
                    # Found a suitable bin
                    bin_data['items'].append(item_id)
                    bin_data['total_weight'] += item['w']
                    bin_data['total_volume'] += item['v']
                    placed[item_id] = True
                    placed_in_bin = True
                    break
            
            if not placed_in_bin:
                # Create a new bin
                new_bin = {
                    'items': [item_id],
                    'total_weight': item['w'],
                    'total_volume': item['v']
                }
                bins.append(new_bin)
        
        # Check if current solution is better
        if len(bins) < best_bins_count:
            best_bins_count = len(bins)
            best_solution = bins
    
    return best_solution

def validate_and_format_solution(solution, instance):
    """Validate the solution format and return formatted output."""
    # Calculate the total number of items that were successfully assigned
    all_assigned_items = set()
    for bin_data in solution:
        for item_idx in bin_data['items']:
            if item_idx < instance['n']:
                all_assigned_items.add(item_idx)
    
    # Validate: all items should be assigned exactly once
    assert len(all_assigned_items) == instance['n'], f"Not all items were assigned: {len(all_assigned_items)} out of {instance['n']} assigned"
    assert all(item < instance['n'] for bin in solution for item in bin['items']), "Invalid item index in solution"

    formatted_solution = {
        "objective_value": len(solution),
        "bins": []
    }
    
    for bin_data in solution:
        formatted_bin = {
            "items": sorted(bin_data['items']),
            "total_weight": bin_data['total_weight'],
            "total_volume": bin_data['total_volume']
        }
        
        # Verify bin constraints
        assert formatted_bin['total_weight'] <= instance['W'], "Weight constraint violated"
        assert formatted_bin['total_volume'] <= instance['V'], "Volume constraint violated"
        
        formatted_solution['bins'].append(formatted_bin)
    
    return formatted_solution

def main():
    parser = argparse.ArgumentParser(description='Solve 2D Bin Packing Problem')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the input instance file')
    parser.add_argument('--solution_path', type=str, required=True, help='Path where the solution JSON will be written')
    parser.add_argument('--time_limit', type=int, required=True, help='Maximum allowed runtime in seconds')
    parser.add_argument('--log_path', type=str, help='Path to the log file for intermediate solutions')
    
    args = parser.parse_args()
    
    # Initialize the solution logger if provided
    logger = None
    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize")
    
    # Read instance data
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    # Start timer
    import time
    start_time = time.time()
    
    try:
        # Solve the problem
        raw_solution = solve_2d_bin_packing(instance)
        
        # Validate and format the solution
        solution = validate_and_format_solution(raw_solution, instance)
        
        if logger:
            logger.log(solution["objective_value"])
        
    except Exception as e:
        print(f"Error during solving: {e}", file=sys.stderr)
        raise
    
    # Write the solution to file
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)
    
    print(f"Solution with {solution['objective_value']} bins written to {args.solution_path}")

if __name__ == '__main__':
    main()
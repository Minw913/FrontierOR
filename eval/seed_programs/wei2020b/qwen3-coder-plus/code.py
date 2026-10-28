import argparse
import json
import sys
import time


def solve_2d_bin_packing(instance, timeout_start, timeout_seconds):
    """
    Solve the 2D Bin Packing problem with weight and volume constraints.
    This is a heuristic implementation with First Fit Decreasing and Best Fit Decreasing approaches.
    """
    n = instance["n"]
    W = instance["W"]  # Weight capacity per bin
    V = instance["V"]  # Volume capacity per bin
    items = instance["items"]

    # Create items with original indices
    indexed_items = [(i, items[i]) for i in range(n)]

    # Sort items in decreasing order - several possible strategies
    def weight_violation(w, v):
        return w/W + v/V  # Combined ratio metric
    
    sorted_items = sorted(indexed_items, key=lambda x: weight_violation(x[1]['w'], x[1]['v']), reverse=True)
    
    # First Fit Decreasing approach
    bins_ffd = []
    for item_idx, item in sorted_items:
        placed = False
        for bin_data in bins_ffd:
            if bin_data['total_weight'] + item['w'] <= W and \
               bin_data['total_volume'] + item['v'] <= V:
                # Pack the item in this bin
                bin_data['items'].append(item_idx)
                bin_data['total_weight'] += item['w']
                bin_data['total_volume'] += item['v']
                placed = True
                break
                
        if not placed:
            # Create a new bin
            bins_ffd.append({
                'items': [item_idx],
                'total_weight': item['w'],
                'total_volume': item['v']
            })
    
    # Best Fit Decreasing approach  
    bins_bfd = []
    for item_idx, item in sorted_items:
        best_bin_idx = -1
        min_remaining_space = float('inf')  # Minimize remaining unused space
        
        # Find bin that gives minimal remaining space after adding current item
        for i, bin_data in enumerate(bins_bfd):
            if bin_data['total_weight'] + item['w'] <= W and \
               bin_data['total_volume'] + item['v'] <= V:
                
                # Combined measure of space left
                remaining_weight = W - (bin_data['total_weight'] + item['w'])
                remaining_volume = V - (bin_data['total_volume'] + item['v'])
                remaining_combined = remaining_weight + remaining_volume
                
                if remaining_combined < min_remaining_space:
                    min_remaining_space = remaining_combined
                    best_bin_idx = i
        
        if best_bin_idx != -1:
            # Place the item in the best bin
            bins_bfd[best_bin_idx]['items'].append(item_idx)
            bins_bfd[best_bin_idx]['total_weight'] += item['w']
            bins_bfd[best_bin_idx]['total_volume'] += item['v']
        else:
            # Create a new bin
            bins_bfd.append({
                'items': [item_idx],
                'total_weight': item['w'],
                'total_volume': item['v']
            })

    # Return the better of the two
    if len(bins_ffd) <= len(bins_bfd):
        return bins_ffd
    else:
        return bins_bfd


def validate_and_format_solution(solution, instance):
    """Validate the solution format and return formatted output."""
    n = instance['n']
    W = instance['W']
    V = instance['V']
    
    # Create a set of all assigned item indices
    all_assigned_items = set()
    for bin_data in solution:
        for item_idx in bin_data['items']:
            # Ensure item index is valid
            if not (0 <= item_idx < n):
                raise AssertionError(f"Invalid item index {item_idx} in solution; must be between 0 and {n-1}")
            all_assigned_items.add(item_idx)
    
    # Validate: all items should be assigned exactly once
    if len(all_assigned_items) != n:
        raise AssertionError(f"Not all items were assigned: {len(all_assigned_items)} out of {n} assigned")
    
    formatted_solution = {
        "objective_value": len(solution),
        "bins": []
    }
    
    for bin_data in solution:
        formatted_bin = {
            "items": sorted(bin_data['items']),  # Ensure items are sorted in each bin
            "total_weight": bin_data['total_weight'],
            "total_volume": bin_data['total_volume']
        }
        
        # Verify bin constraints
        if formatted_bin['total_weight'] > W:
            raise AssertionError(f"Weight constraint violated in bin: {formatted_bin['total_weight']} > {W}")
        if formatted_bin['total_volume'] > V:
            raise AssertionError(f"Volume constraint violated in bin: {formatted_bin['total_volume']} > {V}")
        
        formatted_solution['bins'].append(formatted_bin)
    
    # Check that no item was missed during the assignment process
    all_item_indices_in_solution = set()
    for bin_data in formatted_solution['bins']:
        for item_idx in bin_data['items']:
            all_item_indices_in_solution.add(item_idx)
    
    if len(all_item_indices_in_solution) != n:
        raise AssertionError(f"Item count mismatch after formatting: {len(all_item_indices_in_solution)} items found in solution vs {n} expected")
    
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
    start_time = time.time()
    
    try:
        # Solve the problem
        raw_solution = solve_2d_bin_packing(instance, start_time, args.time_limit)
        
        # Validate and format the solution
        solution = validate_and_format_solution(raw_solution, instance)
        
        # Log if logger exists
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
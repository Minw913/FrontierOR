import argparse
import json
import math
from collections import defaultdict
from solution_logger import SolutionLogger


def solve_instance(instance_data, time_limit, log_path):
    # Extract instance data
    num_slabs = instance_data['num_slabs']
    num_orders = instance_data['num_orders']
    num_colors = instance_data['num_colors']
    max_colors_per_knapsack = instance_data['max_colors_per_knapsack']
    slab_weights = instance_data['slab_weights']
    order_weights = instance_data['order_weights']
    order_colors = instance_data['order_colors']
    eligible_slabs_per_order = instance_data['eligible_slabs_per_order']

    # Calculate profit density for orders (weight / possible slab capacities)
    order_profits = [
        (2 * order_weights[i], i) 
        for i in range(num_orders)
    ]
    order_profits.sort(reverse=True)  # Sort by profit descending
    
    # Initialize solution
    order_assignment = [-1] * num_orders  # Assign each order to a slab, -1 means unassigned
    slab_orders = [[] for _ in range(num_slabs)]
    slab_colors = [set() for _ in range(num_slabs)]
    slab_loads = [0.0 for _ in range(num_slabs)]
    slabs_used = set()

    # Pack orders greedily
    for _, order_idx in order_profits:
        possible_slabs = [
            slab_idx for slab_idx in eligible_slabs_per_order[order_idx]
            if slab_loads[slab_idx] + order_weights[order_idx] <= slab_weights[slab_idx]
        ]
        
        # Filter to slabs that would be valid considering color constraint
        valid_slabs = []
        for slab_idx in possible_slabs:
            current_colors = slab_colors[slab_idx]
            order_color = order_colors[order_idx]
            
            # Check if current assignment already uses both allowed colors
            if len(current_colors) < max_colors_per_knapsack:
                # We can add this color regardless
                valid_slabs.append((slab_idx, 0))
            elif order_color in current_colors:
                # We can reuse an existing color
                valid_slabs.append((slab_idx, 0))
        
        if valid_slabs:
            # Select the best slab (simple heuristic: prefer less loaded first)
            best_slab_idx = min(valid_slabs, key=lambda x: slab_loads[x[0]])[0]
            
            # Add order to best slab
            order_assignment[order_idx] = best_slab_idx
            slab_orders[best_slab_idx].append(order_idx)
            slab_loads[best_slab_idx] += order_weights[order_idx]
            slab_colors[best_slab_idx].add(order_colors[order_idx])
            slabs_used.add(best_slab_idx)

    # Refine solution through local search improvement
    improved = True
    iteration = 0

    logger = SolutionLogger(log_path, sense="maximize") if log_path else None
    current_obj = calculate_current_objective(order_assignment, order_weights, slabs_used, slab_weights)
    if logger:
        logger.log(current_obj)
    
    # Local search with limited iterations for performance
    while improved and iteration < 10:  # Limit iterations for performance
        improved = False
        iteration += 1
        
        for order_idx in range(len(order_assignment)):
            if order_assignment[order_idx] != -1:
                # Temporarily remove the order
                current_slab = order_assignment[order_idx]
                slab_orders[current_slab].remove(order_idx)
                slab_loads[current_slab] -= order_weights[order_idx]
                
                # Update color tracking
                color_removed = order_colors[order_idx]
                if not any(order_colors[o] == color_removed for o in slab_orders[current_slab]):
                    slab_colors[current_slab].remove(color_removed)
                    
                if not slab_orders[current_slab]:  # Slab is now empty
                    slabs_used.discard(current_slab)
                    
                order_assignment[order_idx] = -1
                
                # Try to re-assign to another slab
                best_new_assignment = -1
                best_new_slab = -1
                
                possible_slabs = [
                    slab_idx for slab_idx in eligible_slabs_per_order[order_idx]
                    if slab_loads[slab_idx] + order_weights[order_idx] <= slab_weights[slab_idx]
                ]
                
                for slab_idx in possible_slabs:
                    current_colors = slab_colors[slab_idx]
                    order_color = order_colors[order_idx]
                    
                    # Check color constraint
                    if len(current_colors) < max_colors_per_knapsack or order_color in current_colors:
                        # This slab accepts this color -> good candidate
                        if best_new_assignment == -1 or slab_loads[slab_idx] > slab_loads[best_new_slab]:
                            best_new_assignment = order_idx
                            best_new_slab = slab_idx
                
                # If we found a better assignment, make it
                if best_new_assignment != -1:
                    # Assign the order to the best slab
                    order_assignment[order_idx] = best_new_slab
                    slab_orders[best_new_slab].append(order_idx)
                    slab_loads[best_new_slab] += order_weights[order_idx]
                    slab_colors[best_new_slab].add(order_colors[order_idx])
                    slabs_used.add(best_new_slab)
                    improved = True
                else:
                    # No better assignment, put back in original spot
                    order_assignment[order_idx] = current_slab
                    slab_orders[current_slab].append(order_idx)
                    slab_loads[current_slab] += order_weights[order_idx]
                    slab_colors[current_slab].add(order_colors[order_idx])
                    slabs_used.add(current_slab)

        # Log if improved
        new_obj = calculate_current_objective(order_assignment, order_weights, slabs_used, slab_weights)
        if new_obj > current_obj:
            current_obj = new_obj
            improved = True
            if logger:
                logger.log(new_obj)

    # Finalize solution
    assignments = []
    for i, slab_id in enumerate(order_assignment):
        if slab_id != -1:
            assignments.append({"order": i, "slab": slab_id})
    
    slabs_used_final = [slab_id for slab_id in range(num_slabs) if any(order_assignment[order_idx] == slab_id for order_idx in range(num_orders))]
    
    # Compute objective value
    total_weight_rewards = 0
    for i, slab_id in enumerate(order_assignment):
        if slab_id != -1:
            total_weight_rewards += 2 * order_weights[i]
    
    total_slab_penalties = 0
    for slab_id in slabs_used:
        total_slab_penalties += slab_weights[slab_id]
    
    objective_value = total_weight_rewards - total_slab_penalties
    
    # Log final objective if needed
    if logger:
        logger.log(objective_value)
    
    return {
        "objective_value": objective_value,
        "assignments": assignments,
        "slabs_used": sorted(list(slabs_used_final))
    }


def calculate_current_objective(order_assignment, order_weights, slabs_used, slab_weights):
    total_order_reward = sum(2 * order_weights[i] for i in range(len(order_assignment)) if order_assignment[i] != -1)
    total_slab_penalty = sum(slab_weights[j] for j in slabs_used)
    return total_order_reward - total_slab_penalty


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON file containing the problem instance")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path where the final solution JSON file must be written")
    parser.add_argument("--time_limit", type=int, required=True,
                        help="Maximum runtime in seconds")
    parser.add_argument("--log_path", type=str, required=False,
                        help="Path to a JSONL file for logging intermediate solutions")

    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)

    solution = solve_instance(instance_data, args.time_limit, args.log_path)

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)
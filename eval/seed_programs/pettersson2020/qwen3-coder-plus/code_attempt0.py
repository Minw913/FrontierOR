import argparse
import json
import sys
from itertools import combinations
from collections import defaultdict
import time
from solution_logger import SolutionLogger


def is_dominated(a, b):
    """Check if vector a is dominated by vector b."""
    # b dominates a if b[i] <= a[i] for all i and b[i] < a[i] for at least one i
    equal = True
    for ai, bi in zip(a, b):
        if bi > ai:
            return False  # b is not dominated by b
        elif bi < ai:
            equal = False
    
    return not equal  # a is dominated by b if they're not equal in terms of better values


def remove_dominated(solutions):
    """Remove dominated solutions from list of (objective_vector, selected_items)."""
    n = len(solutions)
    if n <= 1:
        return solutions
        
    to_remove = set()
    
    for i in range(n):
        for j in range(i + 1, n):
            obj_i, _, _ = solutions[i]
            obj_j, _, _ = solutions[j]
            
            if is_dominated(obj_i, obj_j):
                to_remove.add(i)
            elif is_dominated(obj_j, obj_i):
                to_remove.add(j)
                
    result = []
    for i in range(n):
        if i not in to_remove:
            result.append(solutions[i])
            
    return result


def find_non_dominated_solutions_brute_force(num_items, num_objectives, weights, capacity, objective_coefficients):
    """Find non-dominated solutions using a brute force approach."""
    solutions = []
    
    # Generate all possible item combinations (up to a reasonable limit)
    max_combinations = 2 ** min(20, num_items)  # avoid combinatorial explosion
    for i in range(min(max_combinations, 65537)):  # cap iterations to avoid hanging
        selected_indices = []
        
        mask = i
        idx = 0
        while mask > 0 and idx < num_items:
            if mask & 1:
                selected_indices.append(idx)
            mask >>= 1
            idx += 1
            
        # Calculate total weight
        total_weight = sum(weights[idx] for idx in selected_indices)
        if total_weight > capacity:
            continue
            
        # Calculate objective values
        objectives = []
        for obj_idx in range(num_objectives):
            val = sum(objective_coefficients[obj_idx][idx] for idx in selected_indices)
            objectives.append(val)
            
        # Create solution vector
        solution_vector = [0] * num_items
        for idx in selected_indices:
            solution_vector[idx] = 1
        
        solutions.append((tuple(objectives), solution_vector))
    
    # Now remove dominated solutions
    final_solutions = remove_dominated([(obj, [], sol) for obj, sol in solutions])
    
    # Convert back to desired format
    nvov = [list(sol[0]) for sol in final_solutions]
    sols = [sol[2] for sol in final_solutions]
    
    return len(nvov), nvov, sols


def find_non_dominated_solutions_dp_like(num_items, num_objectives, weights, capacity, objective_coefficients):
    """Use an extended DP approach inspired by multi-objective dynamic programming."""
    
    # dp[w][items_selected] -> objectives vector achieved with this weight and state
    # Since too many states, use a pruned search
    
    # Initialize with empty solution
    states = {(0, tuple([0]*num_items)): tuple([0]*num_objectives)}
    
    for i in range(num_items):
        new_states = {}
        
        for (w, selection), objectives in states.items():
            # Don't take item i
            state_key_notake = (w, selection)
            if state_key_notake not in new_states or is_better_objective(states.get(state_key_notake, []), objectives):
                new_states[state_key_notake] = objectives

            # Take item i if possible
            if w + weights[i] <= capacity:
                new_w = w + weights[i]
                
                # Update selection vector
                new_selection_list = list(selection)
                new_selection_list[i] = 1
                new_selection = tuple(new_selection_list)
                
                # Update objectives
                new_objectives_list = list(objectives)
                for j in range(num_objectives):
                    new_objectives_list[j] += objective_coefficients[j][i]
                new_objectives = tuple(new_objectives_list)
                
                # Check dominance here to reduce state space
                key = (new_w, new_selection)
                new_states[key] = new_objectives
        
        # Prune dominated states by checking for same objectives (very aggressive pruning might miss some)
        # For now just take all
        states = new_states
    
    # Extract and filter the solutions - collect unique objectives
    temp_solutions = {}
    for (weight, selection_tuple), obj_tuple in states.items():
        obj_list = list(obj_tuple)
        sel_list = list(selection_tuple)
        
        if tuple(obj_list) not in temp_solutions:
            temp_solutions[tuple(obj_list)] = sel_list
        else:
            # In case of objective duplicates (rare but possible), keep one valid selection
            pass
    
    list_of_sols = [(list(k), v) for k, v in temp_solutions.items()]
    
    # Now we need to do post-filtering to remove dominated solutions
    valid_solutions = []
    for obj_vec, sel_vec in list_of_sols:
        valid_solutions.append((obj_vec, [], sel_vec))  # second element is unused
    
    pruned = remove_dominated(valid_solutions)
    
    nvov = [sol[0] for sol in pruned]
    sols = [sol[2] for sol in pruned]
    
    return len(nvov), nvov, sols


def find_non_dominated_solutions_optimized_search(num_items, num_objectives, weights, capacity, objective_coefficients):
    """
    A more efficient optimized method - explore solutions systematically,
    building up combinations while pruning dominated partial solutions.
    This uses a lexicographic filtering strategy.
    """
    start_time = time.time()
    
    all_solutions = []
    
    # Start with all subset exploration but with pruning
    q = [([0] * num_items, 0, [0] * num_objectives, 0)]  # [selection, current_weight, current_objectives, current_idx]
    
    while q and time.time() - start_time < capacity*0.1 + 10:  # Add time constraint
        selection, total_weight, total_objectives, idx = q.pop()
        
        # Add the current state to our solutions if feasible
        if idx == num_items:
            all_solutions.append((total_objectives[:], selection[:]))
        else:
            # Not taking the current item
            q.append((selection[:], total_weight, total_objectives[:], idx+1))

            # Taking the current item (if possible)
            if total_weight + weights[idx] <= capacity:
                new_selection = selection[:]
                new_selection[idx] = 1
                
                new_objectives = []
                for obj_id in range(num_objectives):
                    new_objectives.append(total_objectives[obj_id] + objective_coefficients[obj_id][idx])
                
                new_weight = total_weight + weights[idx]
                q.append((new_selection, new_weight, new_objectives, idx+1))
    
    # Now filter for non-domination
    final_solutions = []
    for obj_vals, sel_vals in all_solutions:
        is_dominated_flag = False
        for i in range(len(final_solutions)):
            existing_obj = final_solutions[i][0]
            if is_dominated(obj_vals, existing_obj):
                is_dominated_flag = True
                break
            elif is_dominated(existing_obj, obj_vals):
                del final_solutions[i]
                # Continue since we removed from middle
                
        if not is_dominated_flag:
            # Check if this dominates any existing
            non_dominated_to_add = []
            for existing_obj, existing_sel in final_solutions:
                if is_dominated(existing_obj, obj_vals):
                    continue  # skip adding if this dominates existing
                else:
                    non_dominated_to_add.append((existing_obj, existing_sel))
                    
            non_dominated_to_add.append((obj_vals, sel_vals))
            final_solutions = non_dominated_to_add
            
    if final_solutions:
        nvov = [s[0] for s in final_solutions]
        sols = [s[1] for s in final_solutions]
        
        return len(nvov), nvov, sols
    else:
        return 0, [], []


def weighted_sum_method(num_items, num_objectives, weights, capacity, objective_coefficients, logger, time_limit, start_time):
    """
    Use weighted sum scalarization to approximate the Pareto front.
    """
    import itertools
    from collections import defaultdict
    
    all_solutions = set()
    
    # Try different combinations of weight vectors
    # Simple method: iterate across different weights
    step_size = 0.2  # To speed things up
    max_steps = 20   # Max total tries per weight combination
    
    for attempt_weights in itertools.product([i/10.0 for i in range(0, 11, 2)], repeat=num_objectives):
        if sum(attempt_weights) == 0:
            continue
        if time.time() - start_time >= time_limit:
            break
            
        # Solve the aggregated problem
        # We need to maximize sum of weighted objectives (since we're minimizing originals)
        selected_set = set()
        total_weight = 0
        
        # Effective costs per item for the weighted combination
        cost_per_item = []
        for i in range(num_items):
            eff_cost = 0
            for j in range(num_objectives):
                eff_cost += attempt_weights[j] * objective_coefficients[j][i]
            # For minimization with knapsack, we want lower cost items, so take as much as possible 
            # Actually, minimize the aggregate function
            cost_per_item.append((eff_cost, i))
        
        sorted_by_cost = sorted(cost_per_item, key=lambda x: x[0])  # Ascending order for minimization
        
        current_solution = [0]*num_items
        current_objectives = [0]*num_objectives
        
        for eff_cost, item_idx in sorted_by_cost:
            if time.time() - start_time >= time_limit:
                break
            # Try to add this item if it fits
            if total_weight + weights[item_idx] <= capacity:
                total_weight += weights[item_idx]
                current_solution[item_idx] = 1
                for j in range(num_objectives):
                    current_objectives[j] += objective_coefficients[j][item_idx]
                    
        # Calculate final objectives
        if current_solution.count(1) > 0:  # Only consider if any items were selected
            final_obj = []
            for j in range(num_objectives):
                final_obj.append(current_objectives[j])
            
            sol_tuple = tuple(current_solution)
            obj_tuple = tuple(final_obj)
            all_solutions.add((obj_tuple, sol_tuple))
    
    # Also try some standard approaches: pick one objective at a time
    for obj_index in range(min(num_objectives, 4)):  # Limit to avoid timeout
        if time.time() - start_time >= time_limit:
            break
        
        # Sort items by this objective's coefficients and greedy fill
        obj_coeffs = [(objective_coefficients[obj_index][i], i) for i in range(num_items)]
        sorted_coeffs = sorted(obj_coeffs, key=lambda x: x[0])
        
        current_solution = [0]*num_items
        total_weight = 0
        current_obj_values = [0]*num_objectives
        
        for coeff, item_idx in sorted_coeffs:
            if total_weight + weights[item_idx] <= capacity:
                total_weight += weights[item_idx]
                current_solution[item_idx] = 1
                for j in range(num_objectives):
                    current_obj_values[j] += objective_coefficients[j][item_idx]
        
        # Record this solution as well
        solution_obj = tuple(current_obj_values)
        solution_vec = tuple(current_solution)
        all_solutions.add((solution_obj, solution_vec))
    
    # Now remove dominated solutions
    solution_list = [(list(obj), list(sel)) for obj, sel in all_solutions]
    
    # Remove dominated
    pruned = remove_dominated([(obj, [], sel) for obj, sel in solution_list])
    
    nvov = [s[0] for s in pruned]
    sols = [s[2] for s in pruned]
    
    return len(nvov), nvov, sols


def solve_multiconstraint_knapsack(instance_data, time_limit):
    """Solve the multi-objective knapsack problem."""
    start_time = time.time()
    
    num_items = instance_data["num_items"]
    num_objectives = instance_data["num_objectives"]
    weights = instance_data["weights"]
    capacity = instance_data["capacity"]
    objective_coefficients = instance_data["objective_coefficients"]
    
    # For small instances, we could use brute force approach
    # For larger ones, we'll use heuristic methods
    if num_items <= 15:
        # Small enough for exhaustive search
        total_nd, nvov, sols = find_non_dominated_solutions_brute_force(
            num_items, num_objectives, weights, capacity, objective_coefficients
        )
    else:
        # Use weighted sum heuristic + greedy for larger instances
        total_nd, nvov, sols = weighted_sum_method(
            num_items, num_objectives, weights, capacity, objective_coefficients,
            None, time_limit, start_time
        )
        
    return {
        "objective_value": total_nd,
        "non_dominated_objective_vectors": nvov,
        "solutions": sols
    }


def main():
    parser = argparse.ArgumentParser(description='Solve Multi-Objective Knapsack Problem')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the input instance JSON file')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to save the output solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Maximum allowed runtime in seconds')
    parser.add_argument('--log_path', type=str, required=False, help='Path to log intermediate solutions')
    
    args = parser.parse_args()
    
    # Initialize logger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    # Load instance data
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)
    
    start_time = time.time()
    solution = solve_multiconstraint_knapsack(instance_data, args.time_limit)
    
    # Ensure we respect the time limit
    remaining_time = args.time_limit - (time.time() - start_time)
    if remaining_time <= 0:
        print("Time limit exceeded", file=sys.stderr)
        
    # Save the solution
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)
        
    # Log the final objective value if logging is enabled
    if logger:
        logger.log(solution["objective_value"])


if __name__ == '__main__':
    main()
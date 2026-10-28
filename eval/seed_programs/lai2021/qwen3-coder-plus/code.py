import argparse
import json
import math
import time
from itertools import combinations
from solution_logger import SolutionLogger

def solve_diversity_partitioning(instance_path, time_limit, log_path):
    # Initialize logger
    logger = SolutionLogger(log_path, sense="maximize") if log_path else None
    
    # Read instance
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    
    N = instance['N']
    m = instance['m']
    group_capacities = []
    for i in range(m):
        group_capacities.append({
            "L_g": instance["group_capacities"][i]["L_g"],
            "U_g": instance["group_capacities"][i]["U_g"]
        })
    
    distance_upper_triangle = instance["distances_upper_triangular"]
    
    # Convert upper triangular to 2D distance matrix lookup function
    def get_distance(i, j):
        if i == j:
            return 0
        if i > j:
            i, j = j, i  # Ensure i < j for upper triangular access
        # Calculate index in upper triangular matrix
        index = i * (N - 1) - (i * (i + 1)) // 2 + (j - i - 1)
        return distance_upper_triangle[index]
    
    start_time = time.time()
    elapsed = lambda: time.time() - start_time
    
    # Start with a greedy initial assignment
    assignment = {}
    groups = {g: [] for g in range(m)}
    
    # Track current group sizes
    group_sizes = [0] * m
    min_caps = [cap['L_g'] for cap in group_capacities]
    max_caps = [cap['U_g'] for cap in group_capacities]
    
    # Sort elements by some heuristic: here, just randomly initially
    import random
    elements = list(range(N))
    random.shuffle(elements)
    
    # Assign each element to satisfy minimum capacity first
    element_idx = 0
    for g in range(m):
        for _ in range(min_caps[g]):
            if element_idx < N:
                e = elements[element_idx]
                assignment[e] = g
                groups[g].append(e)
                group_sizes[g] += 1
                element_idx += 1
                
    # Assign remaining elements to groups that haven't reached upper bound
    while element_idx < N:
        assigned = False
        for g in range(m):
            if group_sizes[g] < max_caps[g]:
                e = elements[element_idx]
                assignment[e] = g
                groups[g].append(e)
                group_sizes[g] += 1
                element_idx += 1
                assigned = True
                break
        if not assigned:
            break  # All groups reached max capacity
            
    # Verify all elements are assigned
    unassigned = []
    for e in range(N):
        if e not in assignment or assignment[e] == -1:
            unassigned.append(e)
            
    # Backfill unassigned elements to groups that still allow more
    for e in unassigned:
        for g in range(m):
            if group_sizes[g] < max_caps[g]:
                assignment[e] = g
                groups[g].append(e)
                group_sizes[g] += 1
                break
    
    # Calculate initial objective value
    current_obj = calculate_objective(assignment, get_distance)
    
    if logger:
        logger.log(current_obj)
        
    # Perform local search to improve the solution
    no_improvement_counter = 0
    max_no_improve_iters = 100
    
    best_obj = current_obj
    best_assignment = assignment.copy()
    best_groups = {k: v[:] for k, v in groups.items()}
    
    iter_start_time = time.time()
    while elapsed() < time_limit * 0.8 and no_improvement_counter < max_no_improve_iters:
        improved = False
        
        # Try swapping elements between two different groups
        for _ in range(min(200, N*m//2)):  # Cap the number of swaps tried per iteration
            if elapsed() >= time_limit * 0.8:
                break
            
            # Get two elements from different groups
            groups_with_items = [g for g in range(m) if len(groups[g]) > 0]
            if len(groups_with_items) < 2:
                break
            g1_i = random.randint(0, len(groups_with_items)-1)
            g2_i = random.randint(0, len(groups_with_items)-2)
            if g2_i >= g1_i:
                g2_i += 1
            g1, g2 = groups_with_items[g1_i], groups_with_items[g2_i]
            
            if len(groups[g1]) == 0 or len(groups[g2]) == 0:
                continue
                
            e1 = random.choice(groups[g1])
            e2 = random.choice(groups[g2])
            
            # Check if swap preserves capacity constraints
            g1_size_after = group_sizes[g1] - 1 + 1  # removing one and adding one -> stays same
            g2_size_after = group_sizes[g2] - 1 + 1  # stays same
            g1_ok = min_caps[g1] <= g1_size_after <= max_caps[g1]
            g2_ok = min_caps[g2] <= g2_size_after <= max_caps[g2]
            
            if g1_ok and g2_ok:
                # Calculate change in objective by swapping e1 and e2
                old_div_in_g1 = sum(get_distance(e1, other) for other in groups[g1] if other != e1)
                old_div_in_g2 = sum(get_distance(e2, other) for other in groups[g2] if other != e2)
                
                new_div_for_e1_in_g2 = sum(get_distance(e1, other) for other in groups[g2] if other != e2)
                new_div_for_e2_in_g1 = sum(get_distance(e2, other) for other in groups[g1] if other != e1)
                
                old_total = old_div_in_g1 + old_div_in_g2
                new_total = new_div_for_e1_in_g2 + new_div_for_e2_in_g1
                diff = new_total - old_total
                
                if diff > 1e-9:  # Improvement found
                    # Perform the swap
                    groups[g1].remove(e1)
                    groups[g1].append(e2)
                    groups[g2].remove(e2)
                    groups[g2].append(e1)
                    
                    assignment[e1] = g2
                    assignment[e2] = g1
                    
                    current_obj += diff
                    improved = True
                    
                    if logger and current_obj > best_obj:
                        logger.log(current_obj)
                        best_obj = current_obj
                        best_assignment = assignment.copy()
                        best_groups = {k: v[:] for k, v in groups.items()}
                        
                    if abs(current_obj - best_obj) < 1e-6:
                        best_obj = current_obj
                        best_assignment = assignment.copy()
                        best_groups = {k: v[:] for k, v in groups.items()}
                        
        # Try moving one element from one group to another
        for _ in range(min(150, N)):
            if elapsed() >= time_limit * 0.8:
                break
                
            # Pick a random element from any group
            all_assigned = list(assignment.keys())
            if not all_assigned:
                continue
            e = random.choice(all_assigned)
            src_g = assignment[e]
            
            # Pick a random destination group other than current
            possible_dst = [g for g in range(m) if g != src_g]
            if not possible_dst:
                continue
            dst_g = random.choice(possible_dst)
            
            # Check capacity after move
            src_ok_after_move = (group_sizes[src_g] - 1 >= min_caps[src_g]) and \
                               (group_sizes[src_g] - 1 <= max_caps[src_g])
            dst_ok_after_move = (group_sizes[dst_g] + 1 >= min_caps[dst_g]) and \
                               (group_sizes[dst_g] + 1 <= max_caps[dst_g])
            
            if src_ok_after_move and dst_ok_after_move:
                # Calculate objective change due to this move
                old_div_from_src = sum(get_distance(e, other) for other in groups[src_g] if other != e)
                new_div_in_dst = sum(get_distance(e, other) for other in groups[dst_g])
                
                div_change = new_div_in_dst - old_div_from_src
                
                if div_change > 1e-9:
                    # Perform the move
                    groups[src_g].remove(e)
                    groups[dst_g].append(e)
                    
                    assignment[e] = dst_g
                    
                    current_obj += div_change
                    improved = True
                    
                    if logger and current_obj > best_obj:
                        logger.log(current_obj)
                        best_obj = current_obj
                        best_assignment = assignment.copy()
                        best_groups = {k: v[:] for k, v in groups.items()}
                        
                    if abs(current_obj - best_obj) < 1e-6:
                        best_obj = current_obj
                        best_assignment = assignment.copy()
                        best_groups = {k: v[:] for k, v in groups.items()}
        
        if improved:
            no_improvement_counter = 0
        else:
            no_improvement_counter += 1
    
    # Return the best solution found
    final_assignment = best_assignment
    final_groups = {k: sorted(v) for k, v in best_groups.items()}  # Sort group contents
    
    # Final verification that solution respects constraints
    for g in range(m):
        size = len(final_groups[g])
        assert min_caps[g] <= size <= max_caps[g], f"Group {g} has size {size}, required {min_caps[g]} - {max_caps[g]}"
    
    final_obj = calculate_objective(final_assignment, get_distance)
    
    return {
        "objective_value": final_obj,
        "assignment": final_assignment,
        "groups": final_groups
    }

def calculate_objective(assignment, get_distance):
    from collections import defaultdict
    
    # Group elements by their assigned group
    grouped_elements = defaultdict(list)
    for element, group_id in assignment.items():
        grouped_elements[group_id].append(element)
    
    total_diversity = 0.0
    for group_elem_list in grouped_elements.values():
        n_in_group = len(group_elem_list)
        for i in range(n_in_group):
            for j in range(i + 1, n_in_group):
                elem_i = group_elem_list[i]
                elem_j = group_elem_list[j]
                total_diversity += get_distance(elem_i, elem_j)
                
    return total_diversity

def main():
    parser = argparse.ArgumentParser(description='Solve Diversity Partitioning Problem')
    parser.add_argument('--instance_path', type=str, required=True, 
                        help='Path to the input instance JSON file')
    parser.add_argument('--solution_path', type=str, required=True, 
                        help='Path where the solution JSON file will be written')
    parser.add_argument('--time_limit', type=int, required=True, 
                        help='Maximum runtime in seconds')
    parser.add_argument('--log_path', type=str, required=False, 
                        help='Path to log file for intermediate solutions')
    
    args = parser.parse_args()
    
    solution = solve_diversity_partitioning(
        args.instance_path, args.time_limit, args.log_path
    )
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == "__main__":
    main()
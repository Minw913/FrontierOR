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
    
    # Convert upper triangular to 2D distance matrix
    def get_distance(i, j):
        if i == j:
            return 0
        if i > j:
            i, j = j, i  # Ensure i < j for upper triangular access
        index = i * (N - 1) - (i * (i + 1)) // 2 + (j - i - 1)
        return distance_upper_triangle[index]
    
    start_time = time.time()
    elapsed = lambda: time.time() - start_time
    
    # Start with a greedy initial solution
    assignment = {}
    groups = {g: [] for g in range(m)}
    
    # Calculate diversity for the current assignment of an element to each group
    def calculate_diversity_add(element, group):
        diversity_gain = 0
        for e in groups[group]:
            diversity_gain += get_distance(element, e)
        return diversity_gain
    
    # Initially randomly distribute elements respecting capacity constraints
    from random import shuffle
    
    elements = list(range(N))
    shuffle(elements)
    
    # Initialize empty assignment
    for e in range(N):
        assignment[e] = -1
    
    for g in range(m):
        # Fill with minimum requirements first
        for _ in range(group_capacities[g]['L_g']):
            if not elements:
                break
            e = elements.pop()
            assignment[e] = g
            groups[g].append(e)
        
    # Assign the remaining elements to meet capacity constraints
    for g in range(m):
        while len(groups[g]) < group_capacities[g]['U_g'] and elements:
            e = elements.pop()
            assignment[e] = g
            groups[g].append(e)
            
    # Check if all elements are assigned
    unassigned = [e for e in range(N) if assignment[e] == -1]
    for e in unassigned:
        for g in range(m):
            if len(groups[g]) < group_capacities[g]['U_g']:
                assignment[e] = g
                groups[g].append(e)
                break
                
    # Now implement a local search with 2-opt or swap moves since we're tight on time
    current_obj = calculate_objective(assignment, get_distance)
    
    if logger:
        logger.log(current_obj)
        
    last_improvement_time = time.time()
    
    # Local search improvement: try swapping elements between groups to increase diversity
    improved = True
    while improved and elapsed() < time_limit * 0.9:  # Use 90% of time for optimization
        improved = False
        
        # Randomly check swaps
        swaps_to_try = min(500, N * m)  # Limit swaps per iteration
        for _ in range(swaps_to_try):
            if elapsed() >= time_limit * 0.9:
                break
            
            # Pick two random elements in different groups
            elem1 = elements[int(len(elements) * (hash(str(time.time())) % 1000000) / 1000000)]
            group1 = assignment[elem1]
            
            # Find another element in a different group
            possible_elem2 = [(e, assignment[e]) for e in range(N) if assignment[e] != group1 and assignment[e] != -1]
            if not possible_elem2:
                continue
                
            elem2, group2 = possible_elem2[hash(str(elem1) + str(time.time())) % len(possible_elem2)]
            
            # Only consider swap if capacities allow it
            if (len(groups[group1]) - 1 >= group_capacities[group1]['L_g'] and 
                len(groups[group1]) - 1 <= group_capacities[group1]['U_g'] and
                len(groups[group2]) - 1 >= group_capacities[group2]['L_g'] and
                len(groups[group2]) - 1 <= group_capacities[group2]['U_g']):
                
                old_diversity = (
                    # Original within group diversity for both groups
                    sum(get_distance(e1, e2) for e1, e2 in combinations(groups[group1], 2)) +
                    sum(get_distance(e1, e2) for e1, e2 in combinations(groups[group2], 2))
                )
                
                # Temporarily perform swap for objective calculation
                temp_group1 = [e for e in groups[group1] if e != elem1] + [elem2]
                temp_group2 = [e for e in groups[group2] if e != elem2] + [elem1]
                
                new_diversity = (
                    sum(get_distance(e1, e2) for e1, e2 in combinations(temp_group1, 2)) +
                    sum(get_distance(e1, e2) for e1, e2 in combinations(temp_group2, 2))
                )
                
                if new_diversity > old_diversity:
                    # Perform the swap
                    groups[group1] = temp_group1
                    groups[group2] = temp_group2
                    assignment[elem1] = group2
                    assignment[elem2] = group1
                    
                    improved = True
                    current_obj = current_obj - old_diversity + new_diversity
                    
                    if logger:
                        logger.log(current_obj)
                    
                    if elapsed() >= time_limit * 0.8:
                        break

        # Also try moving one element from one group to another if capacities allow
        for _ in range(min(100, N)):
            if elapsed() >= time_limit * 0.9:
                break
            
            # Select a random element
            e = hash(str(time.time()) + str(_)) % N
            source_g = assignment[e]
            
            # Select a random different group to move to
            possible_dst_groups = [g for g in range(m) if g != source_g and
                                 len(groups[g]) + 1 <= group_capacities[g]['U_g'] and
                                 (len(groups[source_g]) - 1) >= group_capacities[source_g]['L_g']]
            
            if not possible_dst_groups:
                continue
                
            dest_g = possible_dst_groups[hash(str(time.time()) + str(e)) % len(possible_dst_groups)]
            
            # Calculate diversity change
            old_diversity_from_src = sum(get_distance(e, other_e) for other_e in groups[source_g] if other_e != e)
            new_diversity_in_dst = sum(get_distance(e, other_e) for other_e in groups[dest_g])
            
            diversity_change = new_diversity_in_dst - old_diversity_from_src
            
            if diversity_change > 0:
                # Perform move
                groups[source_g].remove(e)
                groups[dest_g].append(e)
                assignment[e] = dest_g
                current_obj += diversity_change
                improved = True
                
                if logger:
                    logger.log(current_obj)
                
                if elapsed() >= time_limit * 0.8:
                    break


    # At this point, either we've stopped improving or hit time limit
    # Final cleanup: ensure all capacities are respected and all elements assigned
    assignment, groups = verify_and_fix_assignment(
        assignment,
        groups,
        group_capacities
    )
    
    current_obj = calculate_objective(assignment, get_distance)
    
    return {
        "objective_value": current_obj,
        "assignment": assignment,
        "groups": groups
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


def verify_and_fix_assignment(assignment, groups, group_capacities):
    N = sum(len(lst) for lst in groups.values())
    m = len(group_capacities)
    
    # Reconstruct from assignment (to sync)
    groups_reconstructed = {g: [] for g in range(m)}
    for element, group_id in assignment.items():
        if group_id >= 0:
            groups_reconstructed[group_id].append(element)

    # Try to fix capacity issues by rebalancing elements if needed
    under_capacity = []
    over_capacity = []
    
    for g in range(m):
        low = group_capacities[g]['L_g']
        high = group_capacities[g]['U_g']
        actual_size = len(groups_reconstructed[g])
        
        if actual_size < low:
            needed = low - actual_size
            under_capacity.extend([g] * needed)
        elif actual_size > high:
            excess = actual_size - high
            over_capacity.extend([(el, g) for el in groups_reconstructed[g][:excess]])
    
    # Move excess elements to under-capatcity groups
    for element, excess_group in over_capacity:
        if under_capacity:
            target_group = under_capacity.pop()
            # Move element from excess_group to target_group
            if element in groups_reconstructed[excess_group]:
                groups_reconstructed[excess_group].remove(element)
                groups_reconstructed[target_group].append(element)
                assignment[element] = target_group
    
    # Any remaining assignments issue might have occurred - make sure every element is assigned
    for element in range(N):
        if assignment[element] == -1:
            # Assign to the first available group that isn't at max capacity
            for g in range(m):
                if len(groups_reconstructed[g]) < group_capacities[g]['U_g']:
                    assignment[element] = g
                    groups_reconstructed[g].append(element)
                    break
    
    # Sanitize: sort elements in each group
    for g in groups_reconstructed:
        groups_reconstructed[g].sort()
    
    return assignment, groups_reconstructed

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
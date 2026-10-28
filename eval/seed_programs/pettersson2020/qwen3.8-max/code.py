import json
import argparse
import time


def dominates(a, b):
    """Check if objective vector a dominates b (minimization)."""
    return all(a[i] <= b[i] for i in range(len(a))) and any(a[i] < b[i] for i in range(len(a)))


def solve():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    num_items = instance['num_items']
    num_objectives = instance['num_objectives']
    weights = instance['weights']
    capacity = instance['capacity']
    obj_coeffs = instance['objective_coefficients']

    # Check if all coefficients are non-negative
    all_non_negative = all(
        obj_coeffs[k][i] >= 0
        for k in range(num_objectives)
        for i in range(num_items)
    )

    if all_non_negative:
        # The empty set (selecting nothing) gives (0,0,...,0) which dominates all others
        solution_dict = {
            "objective_value": 1,
            "non_dominated_objective_vectors": [[0] * num_objectives],
            "solutions": [[0] * num_items]
        }
        if logger:
            logger.log_solution(1, solution_dict)
    else:
        # General case: enumerate all feasible subsets and find non-dominated ones
        # Use dynamic programming or enumeration depending on size
        non_dominated_vecs = []
        non_dominated_sols = []

        if num_items <= 25:
            # Full enumeration for small instances
            for mask in range(1 << num_items):
                if time.time() - start_time > args.time_limit - 2:
                    break

                # Compute total weight
                total_weight = 0
                for i in range(num_items):
                    if mask & (1 << i):
                        total_weight += weights[i]
                        if total_weight > capacity:
                            break

                if total_weight > capacity:
                    continue

                # Compute objective vector
                obj_vec = [0] * num_objectives
                for i in range(num_items):
                    if mask & (1 << i):
                        for k in range(num_objectives):
                            obj_vec[k] += obj_coeffs[k][i]

                # Check dominance
                is_dominated = False
                for existing in non_dominated_vecs:
                    if dominates(existing, obj_vec):
                        is_dominated = True
                        break

                if not is_dominated:
                    # Remove vectors dominated by the new one
                    new_vecs = []
                    new_sols = []
                    for idx, existing in enumerate(non_dominated_vecs):
                        if not dominates(obj_vec, existing):
                            new_vecs.append(existing)
                            new_sols.append(non_dominated_sols[idx])

                    selection = [(mask >> i) & 1 for i in range(num_items)]
                    new_vecs.append(obj_vec)
                    new_sols.append(selection)

                    non_dominated_vecs = new_vecs
                    non_dominated_sols = new_sols
        else:
            # For larger instances, use a label-setting approach with DP on weight
            # dp[w] = list of non-dominated objective vectors achievable with weight exactly w
            # We'll track which items were selected for reconstruction

            # Since weights are in [60,100] and capacity = floor(sum_weights/2),
            # capacity can be large. Use a dictionary-based approach.
            # State: (weight) -> list of (obj_vec, selection_mask_or_list)
            
            # For large instances, use iterative Pareto front construction
            # Start with empty set, add items one by one
            # Each state is (obj_vector, selection)
            
            # Pareto front: list of (obj_vec, selection)
            pareto_front = [([0] * num_objectives, [0] * num_items)]

            for item_idx in range(num_items):
                if time.time() - start_time > args.time_limit - 2:
                    break

                new_candidates = []
                for obj_vec, selection in pareto_front:
                    # Try adding item_idx
                    new_weight = sum(weights[i] for i in range(num_items) if selection[i] == 1) + weights[item_idx]
                    if new_weight <= capacity:
                        new_obj = [obj_vec[k] + obj_coeffs[k][item_idx] for k in range(num_objectives)]
                        new_sel = selection[:]
                        new_sel[item_idx] = 1
                        new_candidates.append((new_obj, new_sel))

                # Merge candidates into pareto front
                combined = pareto_front + new_candidates
                # Filter to non-dominated
                filtered = []
                for i, (vec_i, sol_i) in enumerate(combined):
                    is_dom = False
                    for j, (vec_j, sol_j) in enumerate(combined):
                        if i != j and dominates(vec_j, vec_i):
                            is_dom = True
                            break
                    if not is_dom:
                        filtered.append((vec_i, sol_i))
                pareto_front = filtered

            non_dominated_vecs = [v for v, s in pareto_front]
            non_dominated_sols = [s for v, s in pareto_front]

        solution_dict = {
            "objective_value": len(non_dominated_vecs),
            "non_dominated_objective_vectors": non_dominated_vecs,
            "solutions": non_dominated_sols
        }
        if logger:
            logger.log_solution(len(non_dominated_vecs), solution_dict)

    with open(args.solution_path, 'w') as f:
        json.dump(solution_dict, f, indent=2)


if __name__ == '__main__':
    solve()
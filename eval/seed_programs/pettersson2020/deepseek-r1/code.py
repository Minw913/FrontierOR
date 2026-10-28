import argparse
import json
import time

def dominates(vecA, vecB):
    """Check if vecA dominates vecB (minimization)."""
    if vecA == vecB:
        return False
    for a, b in zip(vecA, vecB):
        if a > b:
            return False
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    # Initialize solution logger if log_path is provided
    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize")
    else:
        logger = None

    start_time = time.time()

    # Read instance data
    with open(args.instance_path, 'r') as f:
        data = json.load(f)
    
    num_items = data['num_items']
    num_objectives = data['num_objectives']
    weights = data['weights']
    capacity = data['capacity']
    objective_coefficients = data['objective_coefficients']

    # Initialize DP: list of dictionaries for weights 0 to capacity
    # Each dp[w] is a dictionary: objective_vector_tuple -> selection_mask (integer bitmask)
    dp = [dict() for _ in range(capacity + 1)]
    # Start with empty solution: weight 0, objective vector (0,...,0), mask=0
    dp[0][tuple([0] * num_objectives)] = 0

    i_processed = 0  # Number of items processed
    for i in range(num_items):
        # Check time limit
        current_time = time.time()
        if current_time - start_time > args.time_limit:
            i_processed = i  # Break at current item (processed items: 0 to i-1)
            break

        # Process item i: iterate weights in reverse to avoid reusing the same item
        for w in range(capacity, -1, -1):
            if not dp[w]:
                continue
            w2 = w + weights[i]
            if w2 > capacity:
                continue

            # Generate new states by including item i
            new_states = {}
            for obj_vec, mask in dp[w].items():
                new_obj = tuple(obj_vec[j] + objective_coefficients[j][i] for j in range(num_objectives))
                new_mask = mask | (1 << i)
                new_states[new_obj] = new_mask

            # Merge new_states into dp[w2] and apply within-weight dominance pruning
            merged_states = dp[w2].copy()
            merged_states.update(new_states)  # Overwrite existing with new mask if same objective vector

            # Skip pruning if no states or only one state
            if len(merged_states) <= 1:
                dp[w2] = merged_states
                continue

            # Within-weight dominance pruning using skyline approach
            keys = list(merged_states.keys())
            keys.sort(key=lambda x: x[0])  # Sort by first objective
            skyline = []
            for vec in keys:
                dominated = False
                # Check against current skyline
                for s in skyline:
                    if dominates(s, vec):
                        dominated = True
                        break
                if not dominated:
                    # Remove any solution in skyline dominated by vec
                    skyline = [s for s in skyline if not dominates(vec, s)]
                    skyline.append(vec)
            # Update dp[w2] with non-dominated states
            dp[w2] = {vec: merged_states[vec] for vec in skyline}

        i_processed = i + 1  # Mark item i as processed

    # Collect all objective vectors from all weights (without duplicates)
    all_vectors = {}
    for w in range(capacity + 1):
        for obj_vec, mask in dp[w].items():
            if obj_vec not in all_vectors:
                all_vectors[obj_vec] = mask

    # Global dominance pruning: find non-dominated vectors from all_vectors
    candidates = list(all_vectors.keys())
    if not candidates:
        non_dominated_vectors = []
    else:
        candidates.sort(key=lambda x: x[0])  # Sort by first objective
        skyline = []
        for vec in candidates:
            dominated = False
            for s in skyline:
                if dominates(s, vec):
                    dominated = True
                    break
            if not dominated:
                # Remove any solution in skyline dominated by vec
                skyline = [s for s in skyline if not dominates(vec, s)]
                skyline.append(vec)
        non_dominated_vectors = skyline

    # Prepare output
    objective_value = len(non_dominated_vectors)
    non_dominated_objective_vectors = [list(vec) for vec in non_dominated_vectors]
    solutions = []
    for vec in non_dominated_vectors:
        mask = all_vectors[vec]
        selection = [0] * num_items
        # Set bits for processed items (0 to i_processed-1)
        for j in range(i_processed):
            if mask & (1 << j):
                selection[j] = 1
        solutions.append(selection)

    # Log the final objective_value (count of non-dominated vectors)
    if logger:
        logger.log(objective_value)

    # Write solution
    solution = {
        "objective_value": objective_value,
        "non_dominated_objective_vectors": non_dominated_objective_vectors,
        "solutions": solutions
    }
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == '__main__':
    main()
import argparse
import json
import numpy as np
from solution_logger import SolutionLogger


def calculate_value(selected_indices, instance_data):
    """
    Calculate the value of a selected subset based on the instance type.
    """
    instance_type = instance_data["instance_type"]
    data_matrix = np.array(instance_data["data_matrix"])
    node_weights = np.array(instance_data["node_weights"])
    
    if len(selected_indices) == 0:
        return 0.0
    
    if instance_type == "LOC":
        # Sum over all entities of max benefit from any selected element in the subset
        selected_data = data_matrix[:, selected_indices]
        max_benefits = np.max(selected_data, axis=1)
        return float(np.sum(max_benefits))
    
    elif instance_type == "COV":
        # Sum of weights of items covered by at least one selected sensor
        selected_data = data_matrix[:, selected_indices]
        is_covered = np.any(selected_data > 0, axis=1)
        weighted_coverage = node_weights * is_covered.astype(float)
        return float(np.sum(weighted_coverage))
    
    elif instance_type == "INF":
        # Sum over all targets of probability of being reached by at least one selected source
        selected_element_weights = np.array(instance_data["element_weights"])[selected_indices]
        selected_data = data_matrix[:, selected_indices]
        
        probabilities = []
        for i in range(data_matrix.shape[0]):  # For each target
            connected_sources = selected_data[i] > 0
            connected_probs = selected_element_weights[connected_sources]
            if len(connected_probs) == 0:
                prob = 0.0
            else:
                # Probability of not being reached by any source = product(1-prob)
                prob = 1 - np.prod(1 - connected_probs)
            probabilities.append(prob)
        
        return float(np.sum(probabilities))
    
    else:
        raise ValueError(f"Unknown instance type: {instance_type}")


def greedy_algorithm(instance_data):
    """
    Perform greedy selection based on marginal gains.
    """
    n = instance_data["n"]
    k = instance_data["k"]
    element_values = []  # Store the best value and set found for different sizes up to k
    
    # Start with empty set
    current_set = []
    current_value = 0.0
    
    # Best solution found so far
    best_value = 0.0
    best_set = []
    
    for step in range(k):
        best_candidate = -1
        best_marginal_gain = -1
        
        for candidate_idx in range(n):
            if candidate_idx in current_set:
                continue
                
            test_set = current_set + [candidate_idx]
            test_value = calculate_value(test_set, instance_data)
            marginal_gain = test_value - current_value
            
            if marginal_gain > best_marginal_gain:
                best_marginal_gain = marginal_gain
                best_candidate = candidate_idx
        
        if best_candidate != -1 and best_marginal_gain > 0:
            current_set.append(best_candidate)
            current_value = calculate_value(current_set, instance_data)
            
            if current_value > best_value:
                best_value = current_value
                best_set = current_set[:]
        else:
            # No improvement possible
            break
    
    return best_value, best_set


def solve_instance(instance_data, time_limit, log_path):
    """
    Solve the optimization problem given instance data.
    """
    logger = SolutionLogger(log_path, sense="maximize") if log_path else None
    
    # Run the greedy algorithm
    best_value, best_set = greedy_algorithm(instance_data)
    
    if logger:
        logger.log(obj=best_value, runtime=0)  # Log once when solution is found
    
    # Format the solution accordingly
    solution = {
        "objective_value": best_value,
        "selected_elements": best_set
    }
    
    return solution


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True,
                        help='Path to the JSON file containing the problem instance')
    parser.add_argument('--solution_path', type=str, required=True,
                        help='Path where the final solution JSON file must be written')
    parser.add_argument('--time_limit', type=int, required=True,
                        help='Maximum runtime in seconds')
    parser.add_argument('--log_path', type=str, default=None,
                        help='Path to a JSONL file for logging intermediate solutions')
    
    args = parser.parse_args()
    
    # Load instance data
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)
    
    # Solve the instance
    solution = solve_instance(instance_data, args.time_limit, args.log_path)
    
    # Write solution to file
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == '__main__':
    main()
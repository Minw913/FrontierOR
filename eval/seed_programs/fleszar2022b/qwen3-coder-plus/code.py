import argparse
import json
import time
from solution_logger import SolutionLogger


class KnapsackPairSolver:
    def __init__(self, instance_data):
        self.n = instance_data['n']
        self.m = instance_data['m']
        self.capacity = instance_data['capacity']
        self.weights = instance_data['weights']
        self.values = instance_data['values']
        self.pairwise_values = instance_data['pairwise_values']
        
    def calculate_objective(self, assignment):
        """Calculate the value for given assignment, where assignment[i] = knapsack index for item i, or -1 if unassigned"""
        total_value = 0
        for k in range(self.m):
            items_in_k = [i for i in range(self.n) if assignment[i] == k]
            
            # Calculate individual values
            if len(items_in_k) > 0:
                for i in items_in_k:
                    total_value += self.values[i]
                
                # Calculate pairwise values for items in same knapsack
                for idx_i in range(len(items_in_k)):
                    for idx_j in range(idx_i + 1, len(items_in_k)):
                        i = items_in_k[idx_i]
                        j = items_in_k[idx_j]
                        total_value += self.pairwise_values[i][j]
                        
        return total_value
    
    def is_feasible(self, assignment):
        """Check if the assignment satisfies knapsack capacity constraints"""
        for k in range(self.m):
            items_in_k = [i for i in range(self.n) if assignment[i] == k]
            total_weight = sum([self.weights[i] for i in items_in_k])
            if total_weight > self.capacity:
                return False
        return True
    
    def local_search_heuristic(self, time_limit, logger=None):
        start_time = time.time()
        
        # Initial greedy solution
        assignment = [-1] * self.n  # -1 means unassigned
        knapsacks_weights = [0] * self.m
        
        # Sort items by benefit-to-weight ratio as heuristic
        scores = []
        for i in range(self.n):
            # Calculate score based just on value/weight initially
            score = self.values[i] / self.weights[i] if self.weights[i] > 0 else self.values[i]
            scores.append((score, i))
        
        # Sort in descending order of benefit per weight
        sorted_items = sorted(scores, key=lambda x: x[0], reverse=True)
        
        # Greedy assign items to knapsacks with room
        for _, i in sorted_items:
            placed = False
            for k in range(self.m):
                if knapsacks_weights[k] + self.weights[i] <= self.capacity:
                    # Evaluate potential benefit in knapsack k
                    value = self.values[i]
                    for idx in range(len(assignment)):
                        if assignment[idx] == k and idx != i:
                            value += self.pairwise_values[i][idx]
                    
                    assignment[i] = k
                    knapsacks_weights[k] += self.weights[i]
                    placed = True
                    break
                    
            # If no knapsack could fit the item, leave it unassigned (handled by initialization)
        
        best_assignment = assignment[:]
        best_value = self.calculate_objective(best_assignment)
        if logger:
            logger.log(best_value)
        
        improvement_found = True
        
        while improvement_found and (time.time() - start_time) < time_limit:
            improvement_found = False
            
            # Move neighborhood: try moving each item to different knapsacks or unassign
            for i in range(self.n):
                original_k = best_assignment[i]
                
                # Temporarily remove item i from its current knapsack
                items_in_orig_k = [idx for idx in range(len(best_assignment)) 
                                  if best_assignment[idx] == original_k and idx != i]
                
                # Calculate the value lost by removing item i
                value_lost = self.values[i]
                for other_idx in items_in_orig_k:
                    value_lost += self.pairwise_values[i][other_idx]
                
                # Temporarily unassign to simulate removal
                best_assignment[i] = -1
                knapsacks_weights[original_k] -= self.weights[i]
                
                # Try assignments to all valid knapsacks
                for k in range(self.m):
                    if k == original_k:
                        continue
                        
                    if knapsacks_weights[k] + self.weights[i] <= self.capacity:
                        # Calculate gain by adding item to knapsack k
                        value_gained = self.values[i]
                        items_in_target = [idx for idx in range(len(best_assignment)) 
                                          if best_assignment[idx] == k]
                                          
                        for other_idx in items_in_target:
                            value_gained += self.pairwise_values[i][other_idx] 
                        
                        # Calculate true value change: old_value - value_lost + value_gained
                        new_assignment = best_assignment[:]
                        new_assignment[i] = k
                        temp_knapsack_weights = knapsacks_weights[:]
                        temp_knapsack_weights[original_k] += self.weights[i]  # Restore
                        temp_knapsack_weights[k] += self.weights[i]          # Add to new
                        
                        # Actually compute new objective to avoid complex incremental calc
                        new_value = self.calculate_objective(new_assignment)
                        
                        if new_value > best_value:
                            best_value = new_value
                            best_assignment = new_assignment[:]
                            # Update weights appropriately
                            knapsacks_weights[original_k] += self.weights[i]
                            knapsacks_weights[k] += self.weights[i]
                            
                            if logger:
                                logger.log(best_value)
                            improvement_found = True
                            break  # After improvement found, go to next i
                                    
                # If this item was not reassigned (no improvement found in reassignment), restore original
                if best_assignment[i] == -1:
                    best_assignment[i] = original_k
                    knapsacks_weights[original_k] += self.weights[i]
                    
            # If no improvement across all neighborhoods, end search          
            if not improvement_found:
                break

        # Convert to desired output format
        assignment_dict = {}
        for idx, kns in enumerate(best_assignment):
            if kns != -1:
                assignment_dict[str(idx)] = int(kns)
        
        return {
            "objective_value": float(best_value),
            "assignment": assignment_dict
        }
    
    def solve(self, time_limit, logger=None):
        return self.local_search_heuristic(time_limit, logger)


def main():
    parser = argparse.ArgumentParser(description='Solve knapsack problem with pairing values.')
    parser.add_argument('--instance_path', type=str, help='Path to instance JSON file')
    parser.add_argument('--solution_path', type=str, help='Path to output solution JSON file')
    parser.add_argument('--time_limit', type=int, help='Maximum runtime in seconds')
    parser.add_argument('--log_path', type=str, help='Path to log file (JSONL)')

    args = parser.parse_args()

    # Load instance data
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)

    # Initialize logger
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    # Create solver instance
    solver = KnapsackPairSolver(instance_data)

    # Solve the problem
    solution = solver.solve(args.time_limit, logger)

    # Write solution to file
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

    if solution["objective_value"] == float("-inf"):
        print("No feasible solution found.")
    else:
        print(f"Heuristic optimization completed. Best objective value: {solution['objective_value']}")


if __name__ == '__main__':
    main()
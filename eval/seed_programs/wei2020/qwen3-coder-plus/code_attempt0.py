import argparse
import json
import sys
from itertools import combinations
import gurobipy as gp
from gurobipy import GRB
import signal


def solve_bin_packing_with_conflicts(n, c, weights, time_limit):
    # Create model
    model = gp.Model("bin_packing_with_conflicts")
    
    # Set time limit
    model.setParam('TimeLimit', time_limit)
    model.setParam('MIPFocus', 1)  # Focus on finding good feasible solutions
    model.setParam('Heuristics', 0.05)  # Moderate emphasis on heuristics
    
    # Maximum possible bins needed would be n (one item per bin worst case)
    max_bins = n
    # Decision variables
    # x[i][j] = 1 if item i is assigned to bin j
    x = {}
    # y[j] = 1 if bin j is used
    y = {}
    
    for i in range(n):
        for j in range(max_bins):
            x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
    
    for j in range(max_bins):
        y[j] = model.addVar(vtype=GRB.BINARY, name=f"y_{j}")
        
    # Each item must be assigned to exactly one bin
    for i in range(n):
        model.addConstr(gp.quicksum(x[i, j] for j in range(max_bins)) == 1)
    
    # Bin capacity constraint
    for j in range(max_bins):
        model.addConstr(gp.quicksum(weights[i] * x[i, j] for i in range(n)) <= c * y[j])
    
    # Linking constraint: if any item is in bin j, then bin j is used
    for j in range(max_bins):
        for i in range(n):
            model.addConstr(x[i, j] <= y[j])
    
    # No two conflicting items in the same bin
    # Since we don't have the conflicts parameter, we'll ignore this for now
    # This is a placeholder implementation based on the basic formulation
    
    # Minimize number of used bins
    model.setObjective(gp.quicksum(y[j] for j in range(max_bins)), GRB.MINIMIZE)
    
    # Optimize
    model.optimize()
    
    # Extract solution
    if model.SolCount > 0:
        bins_used = []
        for j in range(max_bins):
            if y[j].X > 0.5:  # Bin j is used
                current_bin_items = []
                for i in range(n):
                    if x[i, j].X > 0.5:  # Item i assigned to bin j
                        current_bin_items.append(i)
                if current_bin_items:  # Only add non-empty bins
                    bins_used.append(current_bin_items)
        return bins_used
    else:
        return None


class TimeoutHandler:
    def __init__(self, timeout_duration):
        self.timeout_duration = timeout_duration

    def handle_timeout(self, signum, frame):
        raise TimeoutError("Time limit exceeded")

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.timeout_duration)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.alarm(0)  # Cancel the alarm


def improved_solve_bin_packing_with_conflicts(n, c, weights, conflicts=None, time_limit=3600):
    # Preprocessing: calculate lower bound using total weight
    total_weight = sum(weights)
    min_bins_needed = (total_weight + c - 1) // c  # ceil division
    max_bins = n  # Upper bound: one item per bin

    # Create model
    model = gp.Model("bin_packing_with_conflicts")
    
    # Set time limit
    model.setParam('TimeLimit', time_limit)
    model.setParam('MIPFocus', 1)
    model.setParam('OutputFlag', 0)  # Suppress Gurobi logs unless needed

    # Decision variables
    # x[i][j] = 1 if item i is assigned to bin j
    x = {}
    # y[j] = 1 if bin j is used
    y = {}
    
    for i in range(n):
        for j in range(min(max_bins, n)):  # Limit max bins to actual items for performance
            x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
    
    for j in range(min(max_bins, n)):
        y[j] = model.addVar(vtype=GRB.BINARY, name=f"y_{j}")
        
    # Each item must be assigned to exactly one bin
    for i in range(n):
        model.addConstr(gp.quicksum(x[i, j] for j in range(min(max_bins, n))) == 1, name=f"assign_item_{i}")
    
    # Bin capacity constraint
    for j in range(min(max_bins, n)):
        model.addConstr(
            gp.quicksum(weights[i] * x[i, j] for i in range(n)) <= c * y[j],
            name=f"capacity_bin_{j}"
        )
    
    # Linking constraint: if any item is in bin j, then bin j is used
    for j in range(min(max_bins, n)):
        for i in range(n):
            model.addConstr(x[i, j] <= y[j], name=f"use_bin_if_item_{j}_has_{i}")
    
    # Add conflicts if provided
    if conflicts is not None:
        for conflict_pair in conflicts:
            item1, item2 = conflict_pair
            for j in range(min(max_bins, n)):
                # Both items cannot be in the same bin
                model.addConstr(x[item1, j] + x[item2, j] <= 1, f"conflict_{item1}_{item2}_in_bin_{j}")
    
    # Optimization priority: try to meet the theoretical minimum
    model.setObjective(gp.quicksum(y[j] for j in range(min(max_bins, n))), GRB.MINIMIZE)
    
    # Optimize
    try:
        model.optimize()
    except gp.GurobiError as e:
        print(f"Gurobi Error: {e}")
        return None
    except Exception as e:
        print(f"Error during optimization: {e}")
        return None
    
    # Extract solution
    if model.status == GRB.INFEASIBLE:
        print("Model is infeasible")
        return None
    elif model.status == GRB.TIME_LIMIT and model.SolCount == 0:
        print("Time limit reached without any solution found")
        return None
    
    if model.SolCount > 0:
        bins_used = []
        for j in range(min(max_bins, n)):
            if y[j].X > 0.5:  # Bin j is used
                current_bin_items = []
                for i in range(n):
                    if x[i, j].X > 0.5:  # Item i assigned to bin j
                        current_bin_items.append(i)
                if current_bin_items:  # Only add non-empty bins
                    bins_used.append(current_bin_items)
        
        # Sort the bins to maintain consistency in output order
        return bins_used
    else:
        print("No solution found")
        return None


def greedy_solution(n, c, weights, conflicts=None):
    """
    Fallback greedy heuristic solution
    """
    item_indices = list(range(n))
    # Sort items in descending order of weight (First Fit Decreasing heuristic)
    item_indices.sort(key=lambda i: weights[i], reverse=True)

    bins = []  # List of bins, each bin is a list of items
    bin_remaining_capacities = []

    # Convert conflicts to sets for faster lookup
    conflict_dict = {}
    for i in range(n):
        conflict_dict[i] = set()
    if conflicts:
        for a, b in conflicts:
            conflict_dict[a].add(b)
            conflict_dict[b].add(a)

    for item in item_indices:
        # Find first bin where item fits with capacity and compatibility constraints
        placed = False
        for i in range(len(bins)):
            # Check capacity
            if bin_remaining_capacities[i] >= weights[item]:
                # Check conflicts
                conflicts_exist = False
                for existing_item in bins[i]:
                    if existing_item in conflict_dict.get(item, set()):
                        conflicts_exist = True
                        break
                
                if not conflicts_exist:
                    # Place item in this bin
                    bins[i].append(item)
                    bin_remaining_capacities[i] -= weights[item]
                    placed = True
                    break
        
        if not placed:
            # Create new bin
            bins.append([item])
            bin_remaining_capacities.append(c - weights[item])
    
    # Return bins with items sorted by original indices
    for bin_items in bins:
        bin_items.sort()
    return bins


def main():
    parser = argparse.ArgumentParser(description='Solve bin packing with conflicts')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to instance JSON file')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to output solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds')
    parser.add_argument('--log_path', type=str, help='Path to log file for intermediate solutions (optional)')
    args = parser.parse_args()
    
    # Initialize logger
    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize")
    else:
        logger = None

    # Read instance
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    n = instance['n']
    c = instance['c']
    weights = instance['weights']
    
    # Note: The schema does not mention conflicts in the instance, 
    # but the description says some instances may have them.
    # For general applicability, assume conflicts exist in some extended version
    conflicts = instance.get('conflicts', None)

    # Use the improved solver
    solution_bins = improved_solve_bin_packing_with_conflicts(n, c, weights, conflicts, time_limit=args.time_limit)
    
    # Fallback to greedy if no solution was found by MIP
    if solution_bins is None:
        solution_bins = greedy_solution(n, c, weights, conflicts)

    if solution_bins is not None:
        # Calculate objective value (number of bins used)
        objective_value = len(solution_bins)
        
        # Log solution if logger exists
        if logger:
            logger.log(objective_value)
        
        # Prepare solution output
        solution = {
            "objective_value": float(objective_value),
            "bins": solution_bins
        }
        
        # Write solution
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f)
    else:
        # Even if we don't find a solution, at least write placeholders 
        solution = {
            "objective_value": float(n),  # Fallback: worst case is one item per bin
            "bins": [[i] for i in range(n)]  # One bin per item
        }
        
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f)


if __name__ == "__main__":
    main()
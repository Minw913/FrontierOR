import argparse
import json
import math
from solution_logger import SolutionLogger


def first_fit_decreasing(items, bin_capacity):
    """First Fit Decreasing heuristic for Bin Packing"""
    # Sort items in decreasing order
    sorted_items = sorted(enumerate(items), key=lambda x: x[1], reverse=True)
    
    bins = []
    bin_assignments = []
    
    for original_idx, item_size in sorted_items:
        placed = False
        
        # Try placing the item in existing bins
        for i, bin_total in enumerate(bins):
            if bin_total + item_size <= bin_capacity:
                bins[i] += item_size
                bin_assignments[i].append((original_idx, item_size))
                placed = True
                break
        
        # If not placed, create a new bin
        if not placed:
            bins.append(item_size)
            bin_assignments.append([(original_idx, item_size)])
    
    result_bins = []
    for bin_items in bin_assignments:
        items_list = [size for idx, size in bin_items]
        total_size = sum(items_list)
        result_bins.append({
            'items': items_list,
            'total_size': total_size
        })
    
    return len(result_bins), result_bins


def generate_cutting_patterns(items, bin_capacity):
    """
    Generate initial cutting patterns using greedy heuristics
    """
    patterns = []
    
    # Add First-Fit-Decreasing solution as initial patterns
    _, ffd_bins = first_fit_decreasing(items, bin_capacity)
    for b in ffd_bins:
        # Create a pattern vector representing one instance of this bin
        pattern = []
        temp_items = items[:]
        
        # For this specific bin composition
        for item_size in b['items']:
            pos = -1
            for i, orig_item in enumerate(temp_items):
                if orig_item == item_size:
                    pos = i
                    break
            if pos != -1:
                temp_items.pop(pos)
        
        # Reconstruct pattern from items in original list
        pattern = [0] * len(set(items))
        unique_sizes = list(set(items))
        unique_sizes.sort(reverse=True)
        
        for item_size in b['items']:
            idx = unique_sizes.index(item_size)
            pattern[idx] += 1
    
    return patterns


def solve_column_generation(items, bin_capacity, time_limit, logger=None):
    """
    Solve using column generation approach with simple greedy pricing
    """
    # Group items by size and record their demands
    from collections import Counter
    size_demand = Counter(items)
    unique_sizes = sorted(size_demand.keys(), reverse=True)  # Largest first
    demands = [size_demand[size] for size in unique_sizes]
    
    # Initialize with trivial patterns (each item type gets its own pattern as much as needed)
    initial_patterns = []
    for i, (size, demand) in enumerate(zip(unique_sizes, demands)):
        max_items_per_bin = bin_capacity // size
        for _ in range(math.ceil(demand / max_items_per_bin)):
            pattern = [0] * len(unique_sizes)
            pattern[i] = min(max_items_per_bin, demand)
            demand -= max_items_per_bin
            if demand > 0:
                continue  # Continue creating new bins if demand is still there
            initial_patterns.append(pattern)
    
    # Solve restricted master problem using linear programming
    try:
        import gurobipy as gp
        from gurobipy import GRB
        
        # Set time limit
        env = gp.Env()
        env.setParam('TimeLimit', time_limit)
        
        # Create model
        model = gp.Model("cutting_stock", env=env)
        model.setParam(GRB.Param.OutputFlag, 0)
        
        # Variables for each column (pattern)
        num_patterns = len(initial_patterns)
        x = model.addVars(num_patterns, vtype=GRB.CONTINUOUS, name="pattern_use")
        
        # Constraints: meet the demand for each item type
        for j in range(len(unique_sizes)):
            row_constraint = gp.quicksum(initial_patterns[i][j] * x[i] 
                                        for i in range(len(initial_patterns)) if initial_patterns[i][j] > 0) >= demands[j]
            model.addConstr(row_constraint, name=f"demand_{j}")
        
        # Objective: minimize the total number of bins
        model.setObjective(gp.quicksum(x[i] for i in range(len(initial_patterns))), GRB.MINIMIZE)
        
        # Solve initial LP relaxation
        model.optimize()
        
        # Column generation loop
        iteration = 0
        while iteration < 50:  # Limit iterations to ensure termination
            iteration += 1
            
            if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
                # Get dual values for generating new columns
                duals = [model.getConstrByName(f"demand_{j}").Pi for j in range(len(unique_sizes))]
                
                # Subproblem: maximize reduced cost (knapsack problem)
                knapsack_model = gp.Model("subproblem", env=env)
                knapsack_model.setParam(GRB.Param.OutputFlag, 0)
                
                y = knapsack_model.addVars(len(unique_sizes), vtype=GRB.INTEGER, lb=0, name="item_count")
                
                knapsack_model.addConstr(
                    gp.quicksum(unique_sizes[i] * y[i] for i in range(len(unique_sizes))) <= bin_capacity,
                    name="capacity"
                )
                
                knapsack_model.setObjective(
                    gp.quicksum(duals[i] * y[i] for i in range(len(unique_sizes))) - 1,  # Reduced cost
                    GRB.MAXIMIZE
                )
                
                knapsack_model.optimize()
                
                # Check if the new column has negative reduced cost
                if knapsack_model.ObjVal <= 0 + 1e-6:  # Optimal if no more negative reduced cost columns
                    break
                
                # Add new column to the master problem
                new_pattern = [int(y[i].x + 0.5) for i in range(len(unique_sizes))]
                initial_patterns.append(new_pattern)
                
                col = model.addVar(vtype=GRB.CONTINUOUS, name=f"new_pattern_{len(initial_patterns)}")
                
                model.update()
                
                # Add the constraint coefficients for this new pattern
                constr_refs = []
                for j in range(len(unique_sizes)):
                    if new_pattern[j] > 0:
                        constr_refs.append((f"demand_{j}", new_pattern[j]))
                
                for constr_name, coeff in constr_refs:
                    model.chgCoeff(model.getConstrByName(constr_name), col, coeff)
                
                # Update the objective coefficient for new column
                # (coefficient in minimization function is always 1 as each var represents 1 bin)
                
                # Re-optimize the master problem
                model.optimize()
            
            else:
                break
        
        # Now we have the LP solution, need to get integer solution
        # Convert to integer variables for final solution
        for i in range(len(initial_patterns)):
            x[i].setAttr('vtype', GRB.INTEGER)
        
        model.setParam('TimeLimit', max(1, time_limit - int(env.get(GRB.DoubleParam.Runtime))))
        model.optimize()
        
        if model.SolCount > 0:
            result_bins = []
            for i in range(len(initial_patterns)):
                count = int(round(x[i].x))
                pattern = initial_patterns[i]
                for _ in range(count):
                    actual_items = []
                    total_size = 0
                    for j, item_count in enumerate(pattern):
                        item_size = unique_sizes[j]
                        for _ in range(item_count):
                            actual_items.append(item_size)
                            total_size += item_size
                    
                    if actual_items:  # Only add non-empty bins
                        result_bins.append({
                            'items': actual_items,
                            'total_size': total_size
                        })

            # Apply logger if exists and solution is improved
            if logger and len(result_bins) > 0:
                logger.log(len(result_bins))

            return len(result_bins), result_bins
    
    except ImportError:
        # Fallback: use greedy heuristics if Gurobi is not available
        pass
    
    # Heuristic fallback when Gurobi unavailable or timeout
    _, result_bins = first_fit_decreasing(items, bin_capacity)
    return len(result_bins), result_bins


def main():
    parser = argparse.ArgumentParser(description='Solve Bin Packing Problem')
    parser.add_argument('--instance_path', type=str, required=True,
                        help='Path to the JSON file containing the problem instance.')
    parser.add_argument('--solution_path', type=str, required=True,
                        help='Path where the final solution JSON file must be written.')
    parser.add_argument('--time_limit', type=int, required=True,
                        help='Maximum runtime in seconds.')
    parser.add_argument('--log_path', type=str, default=None,
                        help='Path to a JSONL file for logging intermediate solutions (optional).')

    args = parser.parse_args()

    # Read instance data
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)

    bin_capacity = instance_data['bin_capacity']
    items = instance_data['items']

    # Initialize logger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Solve the problem
    num_bins, bin_assignments = solve_column_generation(
        items, bin_capacity, args.time_limit, logger
    )

    # Prepare solution object
    solution = {
        "objective_value": num_bins,
        "bin_assignments": bin_assignments,
        "num_bins": num_bins
    }

    # Write solution to file
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == '__main__':
    main()
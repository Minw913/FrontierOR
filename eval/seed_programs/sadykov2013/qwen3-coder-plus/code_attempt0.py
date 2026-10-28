import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def solve_bppc(instance_data, time_limit, logger):
    """
    Solve the Bin Packing Problem with Conflicts (BPPC) using Gurobi.

    Args:
        instance_data: The instance data in the given schema format
        time_limit: Maximum time allowed in seconds
        logger: SolutionLogger object to log intermediate solutions

    Returns:
        Dictionary containing the solution in the required format
    """
    
    # Extract problem data
    n = instance_data["num_items"]
    capacity = instance_data["bin_capacity"]
    items = {item["id"]: item["weight"] for item in instance_data["items"]}
    edges = [(u, v) for u, v in instance_data["conflict_edges"]]
    
    item_ids = sorted(items.keys())
    item_weights = [items[i] for i in item_ids]

    # Create model
    model = gp.Model("BPPC")
    model.params.LogToConsole = 0
    model.params.TimeLimit = time_limit
    
    # Decision variables
    # x[i][j] = 1 if item i is assigned to bin j
    # y[j] = 1 if bin j is used
    max_bins = n  # At most n bins would require as much as we have items
    x = {}  # x[item_id][bin_id]
    for i in item_ids:
        for j in range(max_bins):
            x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

    y = {}  # y[bin_id]
    for j in range(max_bins):
        y[j] = model.addVar(vtype=GRB.BINARY, name=f"y_{j}")

    # Constraints
    # 1. Each item is assigned to exactly one bin
    for i in item_ids:
        model.addConstr(sum(x[i, j] for j in range(max_bins)) == 1, name=f"assign_item_{i}")

    # 2. Capacity constraint for each bin
    for j in range(max_bins):
        model.addConstr(
            sum(x[i, j] * item_weights[item_ids.index(i)] for i in item_ids) <= 
            capacity * y[j], 
            name=f"capacity_bin_{j}"
        )

    # 3. Conflict constraints
    for u, v in edges:
        for j in range(max_bins):
            model.addConstr(x[u, j] + x[v, j] <= 1, name=f"conflict_{u}_{v}_bin_{j}")

    # 4. Link y variables to actual assignment
    for i in item_ids:
        for j in range(max_bins):
            model.addConstr(x[i, j] <= y[j], name=f"activate_bin_{i}_{j}")

    # Objective function
    model.setObjective(sum(y[j] for j in range(max_bins)), GRB.MINIMIZE)

    # Optimal callback function to log solutions
    best_obj_val = float('inf')

    def mycallback(model, where):
        nonlocal best_obj_val
        if where == GRB.Callback.MIPSOL:
            obj = round(model.cbGet(GRB.Callback.MIPSOL_OBJ))
            if obj < best_obj_val and model.cbGet(GRB.Callback.MIPSOL_SOLCNT) > 0:
                best_obj_val = obj
                if logger:
                    logger.log(obj)

    # Optimize
    model.optimize(mycallback)

    # Extract solution
    bins = {}
    active_bins = set()
    
    for j in range(max_bins):
        bin_items = []
        bin_weight = 0
        for i in item_ids:
            if x[i, j].X > 0.5:
                bin_items.append(i)
                bin_weight += items[i]
        
        if bin_items:  # If the bin is actually used
            active_bins.add(j)
            bins[str(j)] = {
                "items": bin_items,
                "total_weight": bin_weight
            }

    # Calculate objective value (number of used bins)
    objective_value = len(active_bins)
    
    # Ensure result is valid in case optimization was interrupted
    if model.SolCount < 1:
        raise Exception("No feasible solution found within time limit")
        
    solution = {
        "objective_value": objective_value,
        "bins": bins
    }
    
    return solution


def main():
    parser = argparse.ArgumentParser(description='Solve Bin Packing Problem with Conflicts')
    parser.add_argument('--instance_path', required=True, help='Path to the input instance JSON file')
    parser.add_argument('--solution_path', required=True, help='Path to save the output solution JSON file')
    parser.add_argument('--time_limit', required=True, type=int, help='Time limit in seconds')
    parser.add_argument('--log_path', required=False, help='Path to log intermediate solutions')
    
    args = parser.parse_args()

    # Initialize logger if specified
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Load instance data
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)

    # Solve the problem
    try:
        solution = solve_bppc(instance_data, args.time_limit, logger)
    except Exception as e:
        print(f"Error solving instance: {e}")
        raise

    # Save the solution
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()
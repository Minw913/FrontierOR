import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def solve_class_knapsack_problem(instance_data, time_limit, log_path):
    # Extract instance data
    N = instance_data["N"]           # Number of classes
    n_star = instance_data["n_star"] # Total number of items
    b = instance_data["b"]           # Knapsack capacity
    
    classes_data = instance_data["classes"]
    
    # Create the model
    model = gp.Model("ClassKnapsack")
    model.setParam('TimeLimit', time_limit)
    model.setParam('OutputFlag', 0)  # Suppress Gurobi output
    
    # Decision variables
    # y_i: whether class i is activated (1) or not (0)
    y_vars = []
    # x_ij: whether item j in class i is selected (1) or not (0)
    x_vars = []  # List of lists for x_{ij}
    
    # Create decision variables
    all_items_list = []  # To store (i, j, item_data) tuples
    
    for i_idx, class_data in enumerate(classes_data):
        class_id = class_data["class_id"]
        n_i = class_data["n_i"]
        items = class_data["items"]
        
        # Add binary decision variable for whether the class is activated
        y_var = model.addVar(vtype=GRB.BINARY, name=f"y_{class_id}")
        y_vars.append(y_var)
        
        x_class_vars = []
        for j_idx, item in enumerate(items):
            item_id = item["item_id"]
            x_var = model.addVar(vtype=GRB.BINARY, name=f"x_{class_id}_{item_id}")
            x_class_vars.append(x_var)
            all_items_list.append((i_idx, j_idx, item))
        
        x_vars.append(x_class_vars)
    
    # Update model with added variables
    model.update()
    
    # Constraints:
    
    # For each class i, if any item in class i is selected, then class i must be activated
    for i_idx in range(N):
        class_data = classes_data[i_idx]
        n_i = class_data["n_i"]
        items = class_data["items"]
        
        # Sum of x_ij for class i <= n_i * y_i (big-M constraint)
        lhs = gp.LinExpr()
        for j_idx in range(n_i):
            lhs += x_vars[i_idx][j_idx]
        
        model.addConstr(lhs <= n_i * y_vars[i_idx], name=f"class_activation_{i_idx+1}")
    
    # Capacity constraint: total weight of selected items + total setup capacity consumption
    capacity_expr = gp.LinExpr()
    
    # Add capacity consumption from selected items
    for i_idx, j_idx, item in all_items_list:
        weight = item["a_ij"]
        capacity_expr += weight * x_vars[i_idx][j_idx]
    
    # Add setup capacity consumption from activated classes  
    for i_idx in range(N):
        class_data = classes_data[i_idx]
        d_i = class_data["d_i"]
        capacity_expr += d_i * y_vars[i_idx]
    
    model.addConstr(capacity_expr <= b, name="capacity")
    
    # Objective function: maximize (sum of item profits + sum of class setup costs)
    obj_expr = gp.LinExpr()
    
    # Add profits from selected items
    for i_idx, j_idx, item in all_items_list:
        profit = item["c_ij"]
        obj_expr += profit * x_vars[i_idx][j_idx]
    
    # Add setup costs from activated classes
    for i_idx in range(N):
        class_data = classes_data[i_idx]
        f_i = class_data["f_i"]  # This is a negative value
        obj_expr += f_i * y_vars[i_idx]
    
    model.setObjective(obj_expr, GRB.MAXIMIZE)
    
    # Initialize solution logger
    logger = SolutionLogger(log_path, sense="maximize") if log_path else None

    def mycallback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj_val)

    # Optimize the model
    model.optimize(mycallback)
    
    # Prepare the solution dictionary
    solution = {"classes": []}
    
    # Get solution values
    y_vals = [model.getVarByName(f"y_{classes_data[i]['class_id']}").X for i in range(N)]
    x_vals = []  # List of lists for x_ij values
    
    for i_idx, class_data in enumerate(classes_data):
        class_id = class_data["class_id"]
        n_i = class_data["n_i"]
        items = class_data["items"]
        
        y_val = y_vals[i_idx]
        x_class_vals = []
        
        for j_idx, item in enumerate(items):
            item_id = item["item_id"]
            x_val = x_vars[i_idx][j_idx].X
            x_class_vals.append(x_val)
        
        x_vals.append(x_class_vals)
        
        # Round values to integer (as they might be slightly off due to numerical precision)
        y_int = round(y_val)
        
        # Process items for this class
        class_solution = {
            "class_id": class_id,
            "y_i": y_int,
            "items": []
        }
        
        for j_idx, item in enumerate(items):
            item_id = item["item_id"]
            x_val = x_vals[i_idx][j_idx]
            x_int = round(x_val)
            
            class_solution["items"].append({
                "item_id": item_id,
                "x_ij": x_int
            })
        
        solution["classes"].append(class_solution)
    
    # Compute final objective value from the solution
    if model.SolCount > 0:
        final_obj_val = model.ObjVal
        solution["objective_value"] = final_obj_val
    else:
        # If no solution was found within the time limit, still create a valid solution with zero value
        solution["objective_value"] = float('-inf')  # Or another placeholder indicating infeasibility
    
    return solution


def main():
    parser = argparse.ArgumentParser(description="Solve Class Knapsack Problem")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to the instance JSON file")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to save the solution JSON file")
    parser.add_argument("--time_limit", type=int, required=True, help="Time limit in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log intermediate solutions (JSONL)")
    
    args = parser.parse_args()
    
    # Load instance data
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)
    
    # Solve the problem
    solution = solve_class_knapsack_problem(instance_data, args.time_limit, args.log_path)
    
    # Write solution to file
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)
    
    print(f"Solution saved to {args.solution_path}")


if __name__ == "__main__":
    main()
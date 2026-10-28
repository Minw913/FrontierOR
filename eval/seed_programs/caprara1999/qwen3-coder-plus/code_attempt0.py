import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def solve_knapsack_with_pairwise_profits(instance_data, time_limit, log_path):
    # Extract instance parameters
    n = instance_data["n"]
    capacity = instance_data["capacity"]
    weights = instance_data["weights"]
    profit_matrix = instance_data["profit_matrix"]

    # Create model
    model = gp.Model("knapsack_with_pairwise_profits")
    
    # Set log path if logger exists
    if log_path:
        # We need to manually handle logging since it's an event callback
        pass  # Will handle logging within callback function
        
    # Create binary variables for each item selection
    x = []
    for i in range(n):
        x.append(model.addVar(vtype=GRB.BINARY, name=f"x_{i}"))
    
    # Add knapsack capacity constraint
    model.addConstr(gp.quicksum(weights[i] * x[i] for i in range(n)) <= capacity, name="knapsack_capacity")
    
    # Linearize the quadratic objective: profit_matrix[i][j] * x[i] * x[j]
    # Introduce new variables z[i][j] to represent product x[i] * x[j]
    z = [[None for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                # Diagonal terms are just profit_matrix[i][i] * x[i]
                continue
            else:
                # z[i][j] represents x[i] * x[j] for off-diagonal terms
                z[i][j] = model.addVar(vtype=GRB.BINARY, name=f"z_{i}_{j}")
                
                # Linearization constraints:
                # Since we are maximizing and profit values are non-negative,
                # the standard linearization constraints work:
                model.addConstr(z[i][j] <= x[i])
                model.addConstr(z[i][j] <= x[j])
                model.addConstr(z[i][j] >= x[i] + x[j] - 1)
    
    # Objective function: sum of diagonal terms + off-diagonal terms
    obj_terms = []
    
    # Individual profits (diagonal terms)
    for i in range(n):
        if profit_matrix[i][i] != 0:
            obj_terms.append(profit_matrix[i][i] * x[i])
            
    # Joint profits (off-diagonal terms)
    for i in range(n):
        for j in range(i+1, n):  # Only consider upper triangle to avoid double counting 
            if profit_matrix[i][j] != 0:
                # Since matrix is symmetric, this term represents: profit_matrix[i][j]*x[i]*x[j] + profit_matrix[j][i]*x[j]*x[i]
                # And profit_matrix[i][j] = profit_matrix[j][i], so contribution is 2*profit_matrix[i][j]*x[i]*x[j]
                # Actually, based on problem statement: "the profit contribution from each pair of distinct selected items equals twice the value of a single off-diagonal entry for that pair"
                # So the objective will sum over all entries including duplicates, so each off-diag element is counted as 2*matrix_val
                obj_terms.append(profit_matrix[i][j] * z[i][j])  # This already accounts for the symm contribution
                obj_terms.append(profit_matrix[j][i] * z[j][i])  # This adds the other part
    
    # Note: Above may double count off-diagonal entries. According to the problem:
    # "each diagonal entry contributes when its item is selected. Because the matrix is symmetric and every off-diagonal entry appears in both orderings, the profit contribution from each pair of distinct selected items equals twice the value of a single off-diagonal entry for that pair."
    # This suggests total objective = sum over selected i,j of (2 * P[i][j]) where P is the symmetric matrix
    
    # Let's rebuild more clearly:
    lin_obj_expr = gp.LinExpr()
    
    # Diagonal elements 
    for i in range(n):
        lin_obj_expr += profit_matrix[i][i] * x[i]

    # Off-diagonal elements: For each pair (i,j) both P[i][j] and P[j][i] are added if both items are selected.
    # Since Matrix is symmetric, P[i][j] == P[j][i]
    for i in range(n):
        for j in range(i+1, n):  # Use upper triangle only but multiply properly
            # If both selected, the total contribution is P[i][j] + P[j][i] = 2 * P[i][j]
            # Using our z variable, the contribution is 2 * profit_matrix[i][j] * z[i][j]
            # But per problem text "contribution from each pair...equals twice the value of single off-diagonal entry":
            # so contribution is 2 * P[i][j] * x[i] * x[j]
            # In linearized form it's 2 * P[i][j] * z[i][j]
            lin_obj_expr += 2 * profit_matrix[i][j] * z[i][j]
    
    # Update model objective
    model.setObjective(lin_obj_expr, GRB.MAXIMIZE)
    
    # Initialize logger
    logger = SolutionLogger(log_path, sense="maximize") if log_path else None
    
    # Callback function for logging
    def mycallback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj_val)

    # Set time limit
    model.setParam('TimeLimit', time_limit)
    
    # Optimize the model with callback for logging
    model.optimize(mycallback)
    
    # Extract the solution
    if model.SolCount > 0:
        # Check which variables are set to 1
        x_vals = [int(x[i].X) for i in range(n)]
        
        # Calculate actual objective value to ensure correctness after any approximations
        # Objective = sum of all P[i][j] where i and j are both selected
        actual_obj_val = 0.0
        for i in range(n):
            if x_vals[i] == 1:
                for j in range(n):
                    if x_vals[j] == 1:
                        actual_obj_val += profit_matrix[i][j]
        
        return {
            "objective_value": actual_obj_val,
            "selected_items": x_vals
        }
    else:
        # No solution found - return an empty solution if timeout occurred
        x_vals = [0 for _ in range(n)]
        return {
            "objective_value": 0.0,
            "selected_items": x_vals
        }


def main():
    parser = argparse.ArgumentParser(description='Solve Knapsack with Pairwise Profits')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to instance JSON file')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to save solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds')
    parser.add_argument('--log_path', type=str, required=False, help='Path to log intermediate solutions')

    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)

    solution = solve_knapsack_with_pairwise_profits(instance_data, args.time_limit, args.log_path)

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == '__main__':
    main()
import argparse
import json
import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None

def main():
    parser = argparse.ArgumentParser(description="Solve the Quadratic Knapsack Problem.")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to the instance JSON file.")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to the output solution JSON file.")
    parser.add_argument("--time_limit", type=int, required=True, help="Time limit in seconds.")
    parser.add_argument("--log_path", type=str, default=None, help="Path to the log JSONL file.")
    args = parser.parse_args()

    # Initialize the solution logger
    logger = None
    if args.log_path and SolutionLogger is not None:
        logger = SolutionLogger(args.log_path, sense="maximize")

    # Read instance data from JSON
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    n = instance['n']
    capacity = instance['capacity']
    weights = instance['weights']
    profit_matrix = instance['profit_matrix']

    # Initialize Gurobi model
    model = gp.Model("Quadratic_Knapsack")
    
    # Configure model parameters
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = args.time_limit
    model.Params.Threads = 1

    # Add decision variables: x[i] = 1 if item i is selected, 0 otherwise
    x = model.addVars(n, vtype=GRB.BINARY, name="x")

    # Add weight capacity constraint
    model.addConstr(gp.LinExpr(weights, [x[i] for i in range(n)]) <= capacity, name="capacity_constraint")

    # Construct the quadratic objective efficiently
    obj = gp.QuadExpr()
    
    # Linear terms (diagonal entries)
    l_coeffs = []
    l_vars = []
    for i in range(n):
        if profit_matrix[i][i] > 0:
            l_coeffs.append(float(profit_matrix[i][i]))
            l_vars.append(x[i])
    if l_coeffs:
        obj.addTerms(l_coeffs, l_vars)

    # Quadratic terms (off-diagonal entries, capturing the symmetry and factor of 2)
    q_coeffs = []
    q_vars1 = []
    q_vars2 = []
    for i in range(n):
        for j in range(i + 1, n):
            if profit_matrix[i][j] > 0:
                q_coeffs.append(2.0 * profit_matrix[i][j])
                q_vars1.append(x[i])
                q_vars2.append(x[j])
    if q_coeffs:
        obj.addTerms(q_coeffs, q_vars1, q_vars2)

    model.setObjective(obj, GRB.MAXIMIZE)

    # Callback function to log incumbent solutions
    def cbfn(mod, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = mod.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger is not None:
                logger.log(obj_val)

    # Optimize the model
    model.optimize(cbfn)

    # Extract results and populate output dictionary
    if model.SolCount > 0:
        objective_value = model.ObjVal
        selected_items = [1 if x[i].X > 0.5 else 0 for i in range(n)]
    else:
        # Fallback to an empty knapsack in the unlikely event no solution is found (guaranteed feasible)
        objective_value = 0.0
        selected_items = [0] * n

    solution = {
        "objective_value": float(objective_value),
        "selected_items": selected_items
    }

    # Save output solution to JSON
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=4)

if __name__ == "__main__":
    main()
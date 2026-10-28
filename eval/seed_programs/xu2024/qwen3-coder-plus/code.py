import argparse
import json
import numpy as np
from gurobipy import *
import sys

def solve_portfolio_optimization(n, k, b, r, Q):
    # Create model
    model = Model('portfolio_optimization')
    model.setParam("Threads", 1)
    
    # Set Gurobi parameters for numerical stability and performance
    model.Params.OutputFlag = 0
    model.setParam(GRB.Param.NonConvex, 2)  # Allow nonconvex problems
    
    # Decision Variables
    x = model.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='x')  # Portfolio weights
    z = model.addVars(n, vtype=GRB.BINARY, name='z')  # Selection of assets
    
    # Objective function: minimize 0.5 * x'Qx
    obj_quad = QuadExpr()
    for i in range(n):
        for j in range(n):
            obj_quad += 0.5 * Q[i][j] * x[i] * x[j]
    model.setObjective(obj_quad, GRB.MINIMIZE)
    
    # Constraint: Sum of weights equals 1
    model.addConstr(sum(x[i] for i in range(n)) == 1, 'budget_constraint')
    
    # Constraint: Minimum return constraint
    model.addConstr(sum(r[i] * x[i] for i in range(n)) >= b, 'return_constraint')
    
    # Constraint: At most k non-zero weights
    model.addConstr(sum(z[i] for i in range(n)) <= k, 'cardinality_constraint')
    
    # Constraints linking x and z variables
    # Define big-M values
    # We can derive loose bounds based on the budget constraint and min return requirement
    big_M_pos = 1000.0  # Upper bound for positive weights
    big_M_neg = -1000.0  # Lower bound for negative weights
    
    for i in range(n):
        # If z[i] = 0, then x[i] = 0
        # If z[i] = 1, then x[i] can be non-zero
        
        # x[i] <= M_pos * z[i]
        model.addConstr(x[i] <= big_M_pos * z[i], f'indicator_pos_{i}')
        
        # x[i] >= M_neg * z[i]
        model.addConstr(x[i] >= big_M_neg * z[i], f'indicator_neg_{i}')

    return model, x, z

def extract_solution(model, x, n):
    x_vals = []
    for i in range(n):
        x_vals.append(x[i].X)
    
    # Calculate actual objective value since x'Qx can have numerical errors with quadratic terms
    x_array = np.array(x_vals)
    Q_np = np.array(instance_data['Q'])
    portfolio_variance = np.dot(np.dot(x_array.T, Q_np), x_array)
    
    return {
        "x": x_vals,
        "objective_value": 0.5 * portfolio_variance
    }

def main():
    parser = argparse.ArgumentParser(description='Solve portfolio optimization problem.')
    parser.add_argument('--instance_path', type=str, required=True,
                        help='Path to the JSON file containing the problem instance.')
    parser.add_argument('--solution_path', type=str, required=True,
                        help='Path where the final solution JSON file must be written.')
    parser.add_argument('--time_limit', type=int, required=True,
                        help='Maximum runtime in seconds.')
    parser.add_argument('--log_path', type=str, default=None,
                        help='Path to log file for intermediate solutions.')
    
    args = parser.parse_args()
    
    global instance_data
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)
    
    n = instance_data["n"]
    k = instance_data["k"]
    b = instance_data["b"]
    r = instance_data["r"]
    Q = instance_data["Q"]
    
    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Callback function to log new incumbent solutions
    def mycallback(model, where):
        if where == GRB.Callback.MIPSOL and logger:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            logger.log(obj_val)

    # Solve the model
    model, x, z = solve_portfolio_optimization(n, k, b, r, Q)
    
    # Set time limit
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('OutputFlag', 0)  # Suppress Gurobi output
    
    model.optimize(mycallback)
    
    # Extract solution
    solution = extract_solution(model, x, n)
    
    # Write solution to file
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == "__main__":
    main()
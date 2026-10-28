import argparse
import json
import sys

# Attempt to import SolutionLogger as required
try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None

import gurobipy as gp
from gurobipy import GRB

def get_fallback_solution(n, m, opening_costs, allocation_costs):
    """
    Provides a trivial feasible solution if the solver fails to find one.
    Opens all facilities and assigns each customer fully to their cheapest facility.
    """
    y_out = [1] * n
    x_out = [[0.0] * m for _ in range(n)]
    total_alloc_cost = 0.0
    
    for j in range(m):
        best_i = 0
        min_c = allocation_costs[0][j]
        for i in range(1, n):
            if allocation_costs[i][j] < min_c:
                min_c = allocation_costs[i][j]
                best_i = i
        x_out[best_i][j] = 1.0
        total_alloc_cost += min_c
    
    obj_val = sum(opening_costs) + total_alloc_cost
    return obj_val, y_out, x_out

def main():
    parser = argparse.ArgumentParser(description="Solve Quadratic Uncapacitated Facility Location (qUFL).")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to the instance JSON file.")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to write the solution JSON file.")
    parser.add_argument("--time_limit", type=int, required=True, help="Time limit in seconds.")
    parser.add_argument("--log_path", type=str, default=None, help="Path for logging intermediate solutions.")
    
    args = parser.parse_args()
    
    # Read instance data
    with open(args.instance_path, 'r') as f:
        data = json.load(f)
        
    n = data["num_facilities"]
    m = data["num_customers"]
    opening_costs = data["opening_costs"]
    allocation_costs = data["allocation_costs"]
    
    # Initialize logger if provided
    logger = None
    if args.log_path and SolutionLogger:
        logger = SolutionLogger(args.log_path, sense="minimize")
        
    # Setup Gurobi environment
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 1)
    env.start()
    
    # Initialize Gurobi model
    model = gp.Model("qUFL", env=env)
    
    # Required Gurobi parameters
    model.setParam("TimeLimit", args.time_limit)
    model.setParam("Seed", 0)
    model.setParam("MIPGap", 1e-4)
    model.setParam("NumericFocus", 0)
    model.setParam("Threads", 1)
    
    # Decision Variables
    y = model.addVars(n, vtype=GRB.BINARY, name="y")
    x = model.addVars(n, m, vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="x")
    
    # Construct Objective Function
    # We use QuadExpr.addTerms for efficiency given potentially large matrix instances
    coeffs = []
    x_vars = []
    for i in range(n):
        coeffs.extend(allocation_costs[i])
        x_vars.extend([x[i, j] for j in range(m)])
        
    quad_expr = gp.QuadExpr()
    quad_expr.addTerms(coeffs, x_vars, x_vars)
    
    lin_expr = gp.LinExpr()
    lin_expr.addTerms(opening_costs, [y[i] for i in range(n)])
    
    model.setObjective(lin_expr + quad_expr, GRB.MINIMIZE)
    
    # Constraints
    # 1. Each customer's demand must be completely met (sum of fractions = 1)
    for j in range(m):
        model.addLConstr(gp.LinExpr([1.0] * n, [x[i, j] for i in range(n)]), GRB.EQUAL, 1.0, name=f"demand_{j}")
        
    # 2. Demand can only be served by open facilities
    # Note: Generator expression in addConstrs is highly optimized in gurobipy
    model.addConstrs((x[i, j] <= y[i] for i in range(n) for j in range(m)), name="open")
    
    # Store references for callback
    model._n = n
    model._m = m
    model._y = y
    model._x = x
    model._logger = logger
    
    # Define Callback for tracking incumbents
    def callback(cb_model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = cb_model.cbGet(GRB.Callback.MIPSOL_OBJ)
            y_val = cb_model.cbGetSolution(cb_model._y)
            x_val = cb_model.cbGetSolution(cb_model._x)
            
            y_out = [int(round(y_val[i])) for i in range(cb_model._n)]
            x_out = [[max(0.0, float(x_val[i, j])) for j in range(cb_model._m)] for i in range(cb_model._n)]
            
            sol_dict = {
                "objective_value": float(obj_val),
                "variant": "quadratic",
                "y": y_out,
                "x": x_out
            }
            if cb_model._logger:
                cb_model._logger.log_solution(float(obj_val), sol_dict)
                
    # Optimize
    model.optimize(callback)
    
    # Extract final optimal or best feasible solution
    if model.SolCount > 0:
        obj_val = float(model.ObjVal)
        y_out = [int(round(y[i].X)) for i in range(n)]
        x_out = [[max(0.0, float(x[i, j].X)) for j in range(m)] for i in range(n)]
    else:
        obj_val, y_out, x_out = get_fallback_solution(n, m, opening_costs, allocation_costs)
        
    # Project to the exact required Solution Output Schema
    final_solution = {
        "objective_value": obj_val,
        "variant": "quadratic",
        "y": y_out,
        "x": x_out
    }
    
    # Write solution to JSON
    with open(args.solution_path, 'w') as f:
        json.dump(final_solution, f, indent=2)

if __name__ == "__main__":
    main()
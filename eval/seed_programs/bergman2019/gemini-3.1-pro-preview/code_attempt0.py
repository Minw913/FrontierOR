import argparse
import json
import sys

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    pass

try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None


def solve():
    parser = argparse.ArgumentParser(description="Solve Quadratic Multiple Knapsack Problem")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to the instance JSON file")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to the output solution JSON file")
    parser.add_argument("--time_limit", type=int, required=True, help="Maximum runtime in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log intermediate solutions")
    
    args = parser.parse_args()
    
    # Read instance data
    with open(args.instance_path, "r") as f:
        data = json.load(f)
        
    n = data["n"]
    m = data["m"]
    p = data["profits"]
    Q = data["pairwise_profits"]
    w = data["weights"]
    C = data["capacities"]
    
    # Initialize solution logger if provided
    if args.log_path and SolutionLogger is not None:
        logger = SolutionLogger(args.log_path, sense="maximize")
    else:
        logger = None
        
    # Callback to log new incumbents
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj)
                
    # Initialize Gurobi model
    model = gp.Model("QMKP")
    
    # Set parameters: single CPU core and time limit
    model.setParam('Threads', 1)
    if args.time_limit:
        model.setParam('TimeLimit', args.time_limit)
        
    # Decision Variables: x[i, k] = 1 if item i is placed in knapsack k
    x = {}
    for i in range(n):
        for k in range(m):
            x[i, k] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{k}")
            
    # Set initial feasible solution (all zeros) to ensure a rapid starting incumbent
    for i in range(n):
        for k in range(m):
            x[i, k].Start = 0.0
            
    # Constraint 1: Each item can be assigned to at most one knapsack
    for i in range(n):
        model.addConstr(gp.LinExpr([1.0]*m, [x[i, k] for k in range(m)]) <= 1)
        
    # Constraint 2: Capacity constraints for each knapsack
    for k in range(m):
        coeffs = []
        vars_k = []
        for i in range(n):
            if w[i] > 0:
                coeffs.append(w[i])
                vars_k.append(x[i, k])
        if vars_k:
            model.addConstr(gp.LinExpr(coeffs, vars_k) <= C[k])
            
    # Objective Configuration: Fast expression building with arrays
    
    # Linear component (individual profits)
    p_coeffs = []
    p_vars = []
    for i in range(n):
        if p[i] != 0:
            for k in range(m):
                p_coeffs.append(p[i])
                p_vars.append(x[i, k])
                
    obj = gp.LinExpr(p_coeffs, p_vars)
    
    # Quadratic component (pairwise profits)
    quad_obj = gp.QuadExpr()
    
    q_coeffs = []
    qx_i = []
    qx_j = []
    for i in range(n):
        for j in range(i + 1, n):
            q_val = Q[i][j]
            if q_val != 0:
                for k in range(m):
                    q_coeffs.append(q_val)
                    qx_i.append(x[i, k])
                    qx_j.append(x[j, k])
                    
    quad_obj.addTerms(q_coeffs, qx_i, qx_j)
    
    # Set Objective: Maximize total combined profit
    model.setObjective(obj + quad_obj, GRB.MAXIMIZE)
    
    # Optimize
    model.optimize(callback)
    
    # Extract Solution
    if model.SolCount > 0:
        assignment = []
        for i in range(n):
            for k in range(m):
                if x[i, k].X > 0.5:
                    assignment.append([i, k])
        objective_value = float(model.ObjVal)
    else:
        # Fallback to empty assignment, conceptually 0 profit
        assignment = []
        objective_value = 0.0
        
    # Build dictionary corresponding to the solution schema
    solution = {
        "objective_value": objective_value,
        "assignment": assignment
    }
    
    # Output projection formatting
    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=4)


if __name__ == "__main__":
    solve()
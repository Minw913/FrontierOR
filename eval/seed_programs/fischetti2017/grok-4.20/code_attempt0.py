import json
import argparse
import time
import sys
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    # Load instance
    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    n = data['num_facilities']
    m = data['num_customers']
    opening_costs = data['opening_costs']
    allocation_costs = data['allocation_costs']

    # Initialize logger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Create Gurobi model
    model = gp.Model("qUFL")
    
    # Set parameters as required
    model.setParam(GRB.Param.Seed, 0)
    model.setParam(GRB.Param.MIPGap, 1e-4)
    model.setParam(GRB.Param.NumericFocus, 0)
    model.setParam(GRB.Param.Threads, 1)
    model.setParam(GRB.Param.TimeLimit, args.time_limit)
    
    # Variables
    y = model.addVars(n, vtype=GRB.BINARY, name="y")
    x = model.addVars(n, m, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="x")
    
    # Objective: sum(f_j * y_j) + sum(c_{ij} * x_{ij}^2)
    obj = gp.quicksum(opening_costs[j] * y[j] for j in range(n))
    for j in range(n):
        for i in range(m):
            obj += allocation_costs[j][i] * x[j, i] * x[j, i]
    model.setObjective(obj, GRB.MINIMIZE)
    
    # Constraints
    # Each customer demand fully covered
    for i in range(m):
        model.addConstr(gp.quicksum(x[j, i] for j in range(n)) == 1.0, name=f"demand_{i}")
    
    # Linking constraints: x_ij <= y_j
    for j in range(n):
        for i in range(m):
            model.addConstr(x[j, i] <= y[j], name=f"link_{j}_{i}")
    
    # Callback for logging incumbents
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            # Extract solution
            y_sol = [int(model.cbGetSolution(y[j]) + 0.5) for j in range(n)]
            x_sol = [[model.cbGetSolution(x[j, i]) for i in range(m)] for j in range(n)]
            
            solution_dict = {
                "variant": "quadratic",
                "y": y_sol,
                "x": x_sol
            }
            
            if logger:
                logger.log_solution(obj_val, solution_dict)
    
    # Optimize
    start_time = time.time()
    model.optimize(callback)
    
    # Extract final solution
    if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
        objective_value = model.objVal
        y_sol = [int(y[j].X + 0.5) for j in range(n)]
        x_sol = [[x[j, i].X for i in range(m)] for j in range(n)]
        
        solution_dict = {
            "objective_value": float(objective_value),
            "variant": "quadratic",
            "y": y_sol,
            "x": x_sol
        }
        
        # Log final solution
        if logger:
            logger.log_solution(objective_value, solution_dict)
        
        # Write solution
        with open(args.solution_path, 'w') as f:
            json.dump(solution_dict, f, indent=2)
    else:
        # Fallback: open no facilities is invalid, so we must provide something
        # Use a simple greedy heuristic as last resort
        print("Solver failed, using greedy fallback", file=sys.stderr)
        y_sol = [1] * n
        x_sol = [[0.0 for _ in range(m)] for _ in range(n)]
        for i in range(m):
            best_j = 0
            best_c = float('inf')
            for j in range(n):
                if allocation_costs[j][i] < best_c:
                    best_c = allocation_costs[j][i]
                    best_j = j
            x_sol[best_j][i] = 1.0
        
        obj_val = sum(opening_costs) + sum(allocation_costs[j][i] * x_sol[j][i]**2 
                                          for j in range(n) for i in range(m))
        
        solution_dict = {
            "objective_value": float(obj_val),
            "variant": "quadratic",
            "y": y_sol,
            "x": x_sol
        }
        
        if logger:
            logger.log_solution(obj_val, solution_dict)
        
        with open(args.solution_path, 'w') as f:
            json.dump(solution_dict, f, indent=2)

if __name__ == "__main__":
    main()
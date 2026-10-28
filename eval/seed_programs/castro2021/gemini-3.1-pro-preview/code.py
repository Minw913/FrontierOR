import argparse
import json

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    pass

from solution_logger import SolutionLogger

def solve(args):
    # Load instance data
    with open(args.instance_path, 'r') as f:
        data = json.load(f)
        
    n = data['n']
    m = data['m']
    supplies = data['supplies']
    demands = data['demands']
    arc_cap = data['arc_capacity']
    linear_costs = data['linear_costs']
    cost_type = data.get('cost_type', '')
    quadratic_costs = data.get('quadratic_costs', [])
    
    # Initialize logger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Initialize Gurobi model
    model = gp.Model("TransportationNetwork")
    model.setParam("Threads", 1)
    if args.time_limit:
        model.setParam('TimeLimit', args.time_limit)
        
    # Disable verbose logging to strictly use custom solution logger
    model.setParam('OutputFlag', 0)
    
    # Variables: Flow from each supplier to each customer follows upper bound and non-negativity
    x = model.addVars(n, m, lb=0.0, ub=arc_cap, vtype=GRB.CONTINUOUS, name="x")
    
    # Build the objective expression efficiently
    obj = gp.QuadExpr()
    is_quad = (cost_type == 'quadratic')
    
    linear_vars = []
    linear_coeffs = []
    quad_vars1 = []
    quad_vars2 = []
    quad_coeffs = []
    
    for i in range(n):
        for j in range(m):
            v = x[i, j]
            # Accumulate linear components
            c = linear_costs[i][j]
            if c != 0:
                linear_coeffs.append(c)
                linear_vars.append(v)
            # Accumulate quadratic components if applicable     
            if is_quad:
                q = quadratic_costs[i][j]
                if q != 0:
                    quad_coeffs.append(q)
                    quad_vars1.append(v)
                    quad_vars2.append(v)
                    
    obj.addTerms(linear_coeffs, linear_vars)
    if is_quad:
        obj.addTerms(quad_coeffs, quad_vars1, quad_vars2)
        
    model.setObjective(obj, GRB.MINIMIZE)
    
    # Constraints:
    # 1. Total flow bounded by supply limits
    model.addConstrs((x.sum(i, '*') <= supplies[i] for i in range(n)), name="supply")
    
    # 2. Total flow exactly resolves customer demand
    model.addConstrs((x.sum('*', j) == demands[j] for j in range(m)), name="demand")
    
    # Solve model
    model.optimize()
    
    # Check if a feasible solution was found within the time limit
    if model.SolCount > 0:
        obj_val = model.ObjVal
        if logger:
            logger.log(obj_val)
            
        # Extract variables satisfying output schema
        flows = {}
        for i in range(n):
            for j in range(m):
                flows[f"x_{i}_{j}"] = float(x[i, j].X)
                
        sol = {
            "objective_value": float(obj_val),
            "flows": flows
        }
        
        # Save output to JSON
        with open(args.solution_path, 'w') as f:
            json.dump(sol, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimal Transportation Network Solver")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to instance JSON file.")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to write the solution JSON file.")
    parser.add_argument("--time_limit", type=int, required=True, help="Maximum runtime in seconds.")
    parser.add_argument("--log_path", type=str, default=None, help="Path to a JSONL file for logging intermediate solutions.")
    
    args = parser.parse_args()
    solve(args)
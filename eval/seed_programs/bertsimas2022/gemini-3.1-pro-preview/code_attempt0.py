import argparse
import json
import gurobipy as gp
from gurobipy import GRB

def solve():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    # Attempt to initialize logging for solutions
    try:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    except ImportError:
        logger = None

    # Load problem instance
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    n = instance['n']
    k = instance['k']
    r = instance['r']
    gamma = instance['gamma']
    kappa = instance['kappa']
    mu = instance['mu']
    F = instance['factor_loadings']
    D = instance['idiosyncratic_variance']
    
    cons = instance.get('constraints', {})
    has_min_ret = cons.get('has_min_return_constraint', False)
    r_bar = cons.get('r_bar', None)
    has_min_inv = cons.get('has_min_investment_constraint', False)
    l_min = cons.get('l_min_investment', None)
    u_max = cons.get('u_max_investment', None)

    # Initialize Gurobi environment natively
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    
    model = gp.Model("portfolio_optimization", env=env)
    
    # Restrict to a single CPU core and enforce hard time limit to ensure compliant containerized execution
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", args.time_limit)

    # Define Decision Variables
    x = model.addVars(n, vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="x")
    z = model.addVars(n, vtype=GRB.BINARY, name="z")
    
    # Introduce helper variables for the systematic (factor) part of the covariance form
    y = model.addVars(r, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="y")

    # 1. Budget constraint: full capital must be invested
    model.addConstr(gp.quicksum(x[i] for i in range(n)) == 1.0, name="budget")

    # 2. Maximum number of non-zero allocation securities
    model.addConstr(gp.quicksum(z[i] for i in range(n)) <= k, name="cardinality")

    # 3. y vector definition for systematic risk calculation: y = F^T * x
    if r > 0:
        for j in range(r):
            model.addConstr(y[j] == gp.quicksum(F[i][j] * x[i] for i in range(n)), name=f"y_{j}")

    # 4. Logical implications determining minimum and maximum investment per security
    if has_min_inv and l_min is not None and u_max is not None:
        for i in range(n):
            model.addConstr(x[i] >= l_min[i] * z[i], name=f"l_min_{i}")
            model.addConstr(x[i] <= u_max[i] * z[i], name=f"u_max_{i}")
    else:
        for i in range(n):
            model.addConstr(x[i] <= z[i], name=f"z_limit_{i}")

    # 5. Minimum portfolio return constraint (optional)
    if has_min_ret and r_bar is not None:
        model.addConstr(gp.quicksum(mu[i] * x[i] for i in range(n)) >= r_bar, name="min_return")

    # Build QuadExpr Objective Function
    obj = gp.QuadExpr()
    
    # Add Variance Component (Systematic Risk)
    if r > 0:
        for j in range(r):
            obj.add(0.5 * y[j] * y[j])

    for i in range(n):
        # Add Variance Component (Idiosyncratic Risk) & Concentration Penalty (1 / 2*gamma)
        coeff = 0.5 * (D[i] + 1.0 / gamma)
        obj.add(coeff * x[i] * x[i])
        
        # Add Negative Rewarded Return
        if kappa != 0:
            obj.add(-kappa * mu[i] * x[i])

    model.setObjective(obj, GRB.MINIMIZE)

    # Solution Logging Callback Tracker
    def make_callback(logged_writer):
        def callback(model, where):
            if where == GRB.Callback.MIPSOL:
                try:
                    obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    logged_writer.log(obj_val)
                except Exception:
                    pass
        return callback

    # Trigger Optimization Procedure
    if logger is not None:
        model.optimize(make_callback(logger))
    else:
        model.optimize()

    # Extract and persist optimal assignments if a feasible solution was found
    if model.SolCount > 0:
        sol_x = [x[i].X for i in range(n)]
        out = {
            "objective_value": model.ObjVal,
            "x": sol_x
        }
        with open(args.solution_path, 'w') as f:
            json.dump(out, f, indent=4)
    else:
        print("No feasible solution could be identified within the imposed limits.", flush=True)

if __name__ == "__main__":
    solve()
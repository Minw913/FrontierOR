import argparse
import json
import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None

def solve_portfolio(instance_path, solution_path, time_limit, log_path):
    # Read the instance data
    with open(instance_path, 'r') as f:
        data = json.load(f)
        
    N = data['N']
    S = data['S']
    k = data['k']
    beta = data['beta']
    gamma = data['gamma']
    mu_bar = data['mu_bar']
    mu = data['mu']
    p_s = data['p_s']
    scenarios = data['scenarios']
    
    # Initialize the callback logger if a path is provided
    if SolutionLogger is not None and log_path:
        logger = SolutionLogger(log_path, sense="minimize")
    else:
        logger = None
    
    # Prepare the Gurobi environment and model
    env = gp.Env(empty=True)
    env.setParam("LogToConsole", 0)
    env.start()
    
    model = gp.Model("PortfolioOptimization", env=env)
    model.setParam("TimeLimit", time_limit)
    model.setParam("Threads", 1)  # Constrained to a single CPU core as per requirements
    
    # Define optimization variables
    x = [model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"x_{i}") for i in range(N)]
    z = [model.addVar(vtype=GRB.BINARY, name=f"z_{i}") for i in range(N)]
    a = model.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="a")
    v = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="v")
    u = [model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"u_{s}") for s in range(S)]
    
    # Total portfolio weight must equal 1
    model.addConstr(gp.quicksum(x) == 1.0, name="sum_x_equals_one")
    
    # Expected return requirement (must equal or exceed mu_bar)
    model.addConstr(gp.LinExpr(mu, x) >= mu_bar, name="expected_return_limit")
    
    # Cardinality constraints limit how many assets can be included to k
    model.addConstr(gp.quicksum(z) <= k, name="max_included_assets")
    
    # Weight must be 0 if the asset is not included
    for i in range(N):
        model.addConstr(x[i] <= z[i], name=f"inclusion_enforcement_{i}")
        
    # Scenario constraints to determine loss strictly beyond the generic VaR threshold 'a'
    for s in range(S):
        model.addConstr(u[s] >= -gp.LinExpr(scenarios[s], x) - a, name=f"scenario_loss_margin_{s}")
        
    # CVaR formulation binding 'v' to the computed probability-weighted scenario tail losses
    model.addConstr(v >= (p_s / (1.0 - beta)) * gp.quicksum(u), name="cvar_expectation_bound")
    
    # Define objective: sum of regularization component + VaR + CVaR
    obj_expr = (1.0 / (2.0 * gamma)) * gp.quicksum(x[i] * x[i] for i in range(N)) + a + v
    model.setObjective(obj_expr, GRB.MINIMIZE)
    
    # Logging callback function checking for newly found incumbent solutions
    def callback(cb_model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = cb_model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj_val)
                
    # Run optimization
    model.optimize(callback)
    
    if model.SolCount > 0:
        # Extract best found solution formatted exactly to standard expectations
        solution = {
            "objective_value": float(model.ObjVal),
            "x": [float(x[i].X) for i in range(N)],
            "z": [int(round(z[i].X)) for i in range(N)],
            "a": float(a.X),
            "v": float(v.X)
        }
        
        with open(solution_path, 'w') as f:
            json.dump(solution, f, indent=4)
    else:
        print("No feasible solution was found within the time limit.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve Portfolio Optimization CVaR with cardinality constraints")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to the JSON file containing the problem instance.")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to write the optimal portfolio solution JSON.")
    parser.add_argument("--time_limit", type=int, required=True, help="Maximum allowed runtime duration in seconds.")
    parser.add_argument("--log_path", type=str, required=False, default=None, help="Path to write logging information.")
    
    args = parser.parse_args()
    
    solve_portfolio(args.instance_path, args.solution_path, args.time_limit, args.log_path)
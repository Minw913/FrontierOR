import argparse
import json
import gurobipy as gp
from gurobipy import GRB

# Attempt to import SolutionLogger from the environment
try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None

def solve():
    parser = argparse.ArgumentParser(description="Solve the robust two-stage knapsack optimization problem.")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to the instance JSON file.")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to the output solution JSON file.")
    parser.add_argument("--time_limit", type=int, required=True, help="Maximum runtime in seconds.")
    parser.add_argument("--log_path", type=str, default=None, help="Path to the logging JSONL file.")
    args = parser.parse_args()

    # Initialize logger if provided
    logger = SolutionLogger(args.log_path, sense="minimize") if SolutionLogger and args.log_path else None

    # Load instance data
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    capacity = instance['problem_parameters']['knapsack_capacity']
    budget = instance['problem_parameters']['uncertainty_budget']
    items = instance['items']
    
    # Create Gurobi environment and suppress console output for clean logging
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    
    model = gp.Model("RobustOptimization", env=env)
    
    # Set required operational parameters
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)

    # Dictionaries to hold decision variables
    x = {}
    y = {}
    z = {}
    mu = {}
    item_ids = []
    
    # First stage: Selection variables
    # Second stage: Continuous variables for as-is production and repair
    # Dual variables (mu, theta): Enforce adversarial maximum degradation within uncertainty budget
    for item in items:
        i = item['id']
        item_ids.append(i)
        x[i] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}")
        y[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name=f"y_{i}")
        z[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name=f"z_{i}")
        mu[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"mu_{i}")
        
    theta = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="theta")
    
    # Build Objective Function
    obj_expr = gp.LinExpr()
    obj_expr += budget * theta
    
    for item in items:
        i = item['id']
        o_i = item['outsource_penalty']
        p_i = item['nominal_profit']
        
        # Selection cost baseline (outsourcing penalty minus nominal profit)
        obj_expr += (o_i - p_i) * x[i]
        # Offset outsourcing penalty when produced
        obj_expr += -o_i * y[i]
        # Dual penalty component for the adversarial robustness
        obj_expr += mu[i]
        
    model.setObjective(obj_expr, GRB.MINIMIZE)
    
    # Add Constraints
    cap_expr = gp.LinExpr()
    for item in items:
        i = item['id']
        w_i = item['weight']
        r_i = item['repair_capacity']
        d_i = item['max_degradation']
        
        # Capacity consumed by production and repair activities
        cap_expr += w_i * y[i] + r_i * z[i]
        
        # Dual robustness constraint limiting adversarial effectiveness
        model.addConstr(theta + mu[i] - d_i * y[i] + d_i * z[i] >= 0, name=f"Dual_{i}")
        
        # Logical bounds consistency 
        model.addConstr(z[i] <= y[i], name=f"zy_{i}")
        model.addConstr(y[i] <= x[i], name=f"yx_{i}")

        # Trivial MIP start point: assume nothing is selected
        x[i].Start = 0

    model.addConstr(cap_expr <= capacity, name="Capacity")

    # Inject model properties to use inside the callback context
    model._x = x
    model._item_ids = item_ids
    model._logger = logger

    def cb(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            x_vals = model.cbGetSolution(model._x)
            sol_x = {str(i): int(round(x_vals[i])) for i in model._item_ids}
            solution_dict = {
                "objective_value": obj,
                "x": sol_x
            }
            if model._logger:
                model._logger.log_solution(obj, solution_dict)

    # Optimize the model
    model.optimize(cb)

    # Extract final optimal or best feasible solution 
    if model.SolCount > 0:
        best_obj = model.ObjVal
        sol_x = {str(i): int(round(x[i].X)) for i in item_ids}
        solution = {
            "objective_value": best_obj,
            "x": sol_x
        }
    else:
        # Fallback empty solution in extreme circumstances
        solution = {
            "objective_value": 0.0,
            "x": {str(i): 0 for i in item_ids}
        }

    # Write output to the targeted JSON schema strictly
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    solve()
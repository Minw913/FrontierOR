import argparse
import json
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

    # Read instance data
    with open(args.instance_path, 'r') as f:
        data = json.load(f)
    
    problem_params = data['problem_parameters']
    knapsack_capacity = problem_params['knapsack_capacity']
    uncertainty_budget = problem_params['uncertainty_budget']
    items = data['items']
    n = len(items)
    
    # Initialize solution logger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Create model
    model = gp.Model()
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', args.time_limit)

    # Add variables
    x = model.addVars(n, vtype=GRB.BINARY, name="x")
    y = model.addVars(n, lb=0, ub=1, name="y")
    z = model.addVars(n, lb=0, ub=1, name="z")
    r = model.addVars(n, lb=0, name="r")
    s = model.addVars(n, lb=0, name="s")
    w = model.addVar(lb=0, name="w")

    # Set objective
    obj = gp.quicksum((item['outsource_penalty'] - item['nominal_profit']) * x[i] for i, item in enumerate(items))
    obj -= gp.quicksum(item['outsource_penalty'] * y[i] for i, item in enumerate(items))
    obj += gp.quicksum(r[i] for i in range(n))
    obj += uncertainty_budget * w
    model.setObjective(obj)

    # Add constraints
    for i in range(n):
        model.addConstr(y[i] <= x[i], name=f"prod_sel_{i}")
        model.addConstr(z[i] <= y[i], name=f"repair_prod_{i}")

    capacity_constr = gp.quicksum(
        items[i]['weight'] * y[i] + items[i]['repair_capacity'] * z[i] 
        for i in range(n)
    ) <= knapsack_capacity
    model.addConstr(capacity_constr, name="knapsack_capacity")

    for i, item in enumerate(items):
        max_deg = item['max_degradation']
        model.addConstr(
            r[i] + s[i] + w >= max_deg * (y[i] - z[i]),
            name=f"dual_constraint_{i}"
        )

    # Callback function for logging incumbents
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            x_vals = {str(i): int(round(model.cbGet(GRB.Callback.MIPSOL, x[i]))) for i in range(n)}
            solution_dict = {"x": x_vals}
            if logger:
                logger.log_solution(obj_val, solution_dict)

    # Optimize
    if logger:
        model._callback = callback
        model.optimize()
    else:
        model.optimize()

    # Prepare solution
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        if model.SolCount > 0:
            obj_value = model.ObjVal
            x_sol = {str(i): int(round(x[i].X)) for i in range(n)}
            solution = {
                "objective_value": obj_value,
                "x": x_sol
            }
        else:
            raise RuntimeError("No solution found within time limit.")
    else:
        raise RuntimeError(f"Optimization failed with status {model.status}")

    # Write solution
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()
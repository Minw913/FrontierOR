import gurobipy as gp
from gurobipy import GRB
import json
import argparse
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser(description="Portfolio Optimization with Cardinality Constraint")
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the instance JSON file')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to write the solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Maximum runtime in seconds')
    parser.add_argument('--log_path', type=str, default=None, help='Path to log intermediate solutions (optional)')
    args = parser.parse_args()

    # Read instance data
    with open(args.instance_path, 'r') as f:
        data = json.load(f)
    n = data['n']
    k = data['k']
    b = data['b']
    r = data['r']
    Q = data['Q']

    # Initialize solution logger if log_path is provided
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Create Gurobi model
    model = gp.Model()
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('Threads', 1)

    # Add variables
    x = model.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="x")
    y = model.addVars(n, vtype=GRB.BINARY, name="y")

    # Budget constraint: sum(x) = 1
    model.addConstr(gp.quicksum(x[i] for i in range(n)) == 1, "budget")

    # Return constraint: r'x >= b
    model.addConstr(gp.quicksum(r[i] * x[i] for i in range(n)) >= b, "return")

    # Cardinality constraint: sum(y) <= k
    model.addConstr(gp.quicksum(y[i] for i in range(n)) <= k, "cardinality")

    # Indicator constraints: y[i] = 0 => x[i] = 0
    for i in range(n):
        model.addGenConstrIndicator(y[i], 0, x[i] == 0, name=f"indicator_{i}")

    # Objective: minimize (1/2) * x'Qx
    obj_expr = 0.0
    for i in range(n):
        for j in range(n):
            obj_expr += 0.5 * Q[i][j] * x[i] * x[j]
    model.setObjective(obj_expr, GRB.MINIMIZE)

    # Optimize with callback for logging if logger is provided
    if logger:
        model._logger = logger
        model._best_logged_obj = float('inf')
        def mycallback(m, where):
            if where == GRB.Callback.MIPSOL:
                obj_val = m.cbGet(GRB.Callback.MIPSOL_OBJ)
                if obj_val < m._best_logged_obj:
                    m._best_logged_obj = obj_val
                    m._logger.log(obj_val)
        model.optimize(mycallback)
    else:
        model.optimize()

    # Extract solution
    if model.SolCount > 0:
        x_vals = [x[i].X for i in range(n)]
        obj_val = model.ObjVal
    else:
        # Fallback: invest in the asset with the highest expected return
        max_r = max(r)
        idx = r.index(max_r)
        x_vals = [0.0] * n
        x_vals[idx] = 1.0
        obj_val = 0.5 * Q[idx][idx]

    # Write solution to JSON
    solution = {
        "objective_value": obj_val,
        "x": x_vals
    }
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == "__main__":
    main()
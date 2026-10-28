import argparse
import json
import time
from gurobipy import *
from solution_logger import SolutionLogger


def build_solution_from_vars(y_vals, x_vals, classes, obj_val):
    """Build solution dictionary from variable values."""
    solution = {"classes": [], "objective_value": float(obj_val)}
    for cls in classes:
        i = cls['class_id']
        class_entry = {
            "class_id": i,
            "y_i": int(y_vals[i]),
            "items": []
        }
        for item in cls['items']:
            j = item['item_id']
            class_entry["items"].append({
                "item_id": j,
                "x_ij": int(x_vals[i, j])
            })
        solution["classes"].append(class_entry)
    return solution


def main():
    parser = argparse.ArgumentParser(description="Knapsack Problem with Setup Costs")
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    # Read instance
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    # Initialize logger
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    # Parse instance data
    N = instance['N']
    b = instance['b']
    classes = instance['classes']

    # Build Gurobi model
    model = Model("KPS")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = args.time_limit
    model.Params.OutputFlag = 0

    # Decision variables
    y = {}  # y[i] = 1 if class i is activated
    x = {}  # x[i,j] = 1 if item j in class i is selected

    for cls in classes:
        i = cls['class_id']
        y[i] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}")
        for item in cls['items']:
            j = item['item_id']
            x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

    model.update()

    # Objective: maximize sum of item profits + sum of setup costs (negative)
    obj = LinExpr()
    for cls in classes:
        i = cls['class_id']
        for item in cls['items']:
            j = item['item_id']
            obj += item['c_ij'] * x[i, j]
        obj += cls['f_i'] * y[i]
    model.setObjective(obj, GRB.MAXIMIZE)

    # Capacity constraint: sum of item weights + sum of setup capacities <= b
    cap_expr = LinExpr()
    for cls in classes:
        i = cls['class_id']
        for item in cls['items']:
            j = item['item_id']
            cap_expr += item['a_ij'] * x[i, j]
        cap_expr += cls['d_i'] * y[i]
    model.addConstr(cap_expr <= b, name="capacity")

    # Linking constraints: x_ij <= y_i
    for cls in classes:
        i = cls['class_id']
        for item in cls['items']:
            j = item['item_id']
            model.addConstr(x[i, j] <= y[i], name=f"link_{i}_{j}")

    model.update()

    # Callback to log incumbent solutions
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            y_vals = {}
            x_vals = {}
            for cls in classes:
                i = cls['class_id']
                y_vals[i] = int(round(model.cbGetSolution(y[i])))
                for item in cls['items']:
                    j = item['item_id']
                    x_vals[i, j] = int(round(model.cbGetSolution(x[i, j])))
            solution = build_solution_from_vars(y_vals, x_vals, classes, obj_val)
            if logger:
                logger.log_solution(obj_val, solution)

    # Optimize
    model.optimize(callback)

    # Extract final solution
    if model.SolCount > 0:
        obj_val = model.ObjVal
        y_vals = {}
        x_vals = {}
        for cls in classes:
            i = cls['class_id']
            y_vals[i] = int(round(y[i].X))
            for item in cls['items']:
                j = item['item_id']
                x_vals[i, j] = int(round(x[i, j].X))
        solution = build_solution_from_vars(y_vals, x_vals, classes, obj_val)
    else:
        # No feasible solution found; return all zeros
        obj_val = 0.0
        y_vals = {}
        x_vals = {}
        for cls in classes:
            i = cls['class_id']
            y_vals[i] = 0
            for item in cls['items']:
                j = item['item_id']
                x_vals[i, j] = 0
        solution = build_solution_from_vars(y_vals, x_vals, classes, obj_val)

    # Log final solution if not already logged
    if logger:
        logger.log_solution(solution["objective_value"], solution)

    # Write solution
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()
import json
import argparse
import gurobipy as gp
from gurobipy import GRB

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    # Load instance data
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    N = instance['N']
    b = instance['b']
    classes = instance['classes']

    # Initialize logger
    logger = None
    if args.log_path:
        try:
            from solution_logger import SolutionLogger
            logger = SolutionLogger(args.log_path, sense="maximize")
        except ImportError:
            pass

    # Initialize Gurobi model
    model = gp.Model()
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = float(args.time_limit)

    # Variables
    y = model.addVars(N, vtype=GRB.BINARY, name="y")
    
    x_items = []
    for i, cls in enumerate(classes):
        for j, item in enumerate(cls['items']):
            x_items.append((i, j))
            
    x = model.addVars(x_items, vtype=GRB.BINARY, name="x")
    x_vars_list = [x[i, j] for i, j in x_items]
    x_idx = {(i, j): idx for idx, (i, j) in enumerate(x_items)}

    # Capacity constraint
    cap_expr = gp.LinExpr()
    for i, cls in enumerate(classes):
        cap_expr += cls['d_i'] * y[i]
        for j, item in enumerate(cls['items']):
            cap_expr += item['a_ij'] * x[i, j]
    model.addConstr(cap_expr <= b, name="capacity")

    # Activation constraints
    for i, cls in enumerate(classes):
        for j, item in enumerate(cls['items']):
            model.addConstr(x[i, j] <= y[i], name=f"act_{i}_{j}")

    # Objective
    obj_expr = gp.LinExpr()
    for i, cls in enumerate(classes):
        obj_expr += cls['f_i'] * y[i]
        for j, item in enumerate(cls['items']):
            obj_expr += item['c_ij'] * x[i, j]
    model.setObjective(obj_expr, GRB.MAXIMIZE)

    # Callback for logging intermediate solutions
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            y_vals = model.cbGetSolution(model._y)
            x_vals = model.cbGetSolution(model._x_vars_list)
            
            classes_out = []
            for i, cls in enumerate(model._classes):
                items_out = []
                for j, item in enumerate(cls['items']):
                    val = x_vals[model._x_idx[(i, j)]]
                    items_out.append({
                        "item_id": item['item_id'],
                        "x_ij": int(round(val))
                    })
                classes_out.append({
                    "class_id": cls['class_id'],
                    "y_i": int(round(y_vals[i])),
                    "items": items_out
                })
            sol_dict = {
                "classes": classes_out,
                "objective_value": obj
            }
            if model._logger:
                model._logger.log_solution(obj, sol_dict)

    model._y = y
    model._x = x
    model._x_vars_list = x_vars_list
    model._x_idx = x_idx
    model._classes = classes
    model._logger = logger

    # Optimize
    model.optimize(callback)

    # Output final solution
    if model.SolCount > 0:
        y_vals = model.getAttr('X', y)
        x_vals = model.getAttr('X', x)
        obj_val = model.ObjVal
        
        classes_out = []
        for i, cls in enumerate(classes):
            items_out = []
            for j, item in enumerate(cls['items']):
                items_out.append({
                    "item_id": item['item_id'],
                    "x_ij": int(round(x_vals[i, j]))
                })
            classes_out.append({
                "class_id": cls['class_id'],
                "y_i": int(round(y_vals[i])),
                "items": items_out
            })
        sol_dict = {
            "classes": classes_out,
            "objective_value": obj_val
        }
        
        if logger:
            logger.log_solution(obj_val, sol_dict)
            
        with open(args.solution_path, 'w') as f:
            json.dump(sol_dict, f, indent=2)
    else:
        # Fallback empty solution
        classes_out = []
        for i, cls in enumerate(classes):
            items_out = []
            for j, item in enumerate(cls['items']):
                items_out.append({
                    "item_id": item['item_id'],
                    "x_ij": 0
                })
            classes_out.append({
                "class_id": cls['class_id'],
                "y_i": 0,
                "items": items_out
            })
        sol_dict = {
            "classes": classes_out,
            "objective_value": 0.0
        }
        with open(args.solution_path, 'w') as f:
            json.dump(sol_dict, f, indent=2)

if __name__ == "__main__":
    main()
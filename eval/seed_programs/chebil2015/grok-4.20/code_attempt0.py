import json
import argparse
import time
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

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    N = instance['N']
    b = instance['b']
    classes = instance['classes']
    problem_name = instance.get('problem', 'MKSP')

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    model = gp.Model("Multiple_Knapsack_with_Setup")
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('Threads', 1)
    model.setParam('MIPGap', 0.01)
    model.setParam('LogToConsole', 0)

    # Decision variables
    y = {}
    x = {}
    for cls in classes:
        i = cls['class_id']
        y[i] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}", lb=0, ub=1)
        for item in cls['items']:
            j = item['item_id']
            x[(i, j)] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}", lb=0, ub=1)

    # Objective: maximize sum profits - sum |setup costs|
    obj = gp.LinExpr()
    for cls in classes:
        i = cls['class_id']
        obj.addTerms(cls['f_i'], y[i])  # f_i is negative
        for item in cls['items']:
            j = item['item_id']
            obj.addTerms(item['c_ij'], x[(i, j)])
    model.setObjective(obj, GRB.MAXIMIZE)

    # Capacity constraint
    cap_expr = gp.LinExpr()
    for cls in classes:
        i = cls['class_id']
        cap_expr.addTerms(cls['d_i'], y[i])
        for item in cls['items']:
            j = item['item_id']
            cap_expr.addTerms(item['a_ij'], x[(i, j)])
    model.addConstr(cap_expr <= b, name="capacity")

    # Linking constraints: x_ij <= y_i
    for cls in classes:
        i = cls['class_id']
        for item in cls['items']:
            j = item['item_id']
            model.addConstr(x[(i, j)] <= y[i], name=f"link_{i}_{j}")

    # Callback to log improved solutions
    best_obj = float('-inf')
    start_time = time.time()

    def callback(model, where):
        nonlocal best_obj
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if obj_val > best_obj + 1e-6:
                best_obj = obj_val
                if logger:
                    logger.log(float(obj_val))

    # Optimize
    model.optimize(callback)

    # Extract solution
    solution_classes = []
    final_obj = 0.0
    if model.SolCount > 0:
        final_obj = model.objVal
        for cls in classes:
            i = cls['class_id']
            y_val = int(round(y[i].X))
            item_list = []
            for item in cls['items']:
                j = item['item_id']
                x_val = int(round(x[(i, j)].X))
                item_list.append({
                    "item_id": j,
                    "x_ij": x_val
                })
            solution_classes.append({
                "class_id": i,
                "y_i": y_val,
                "items": item_list
            })
    else:
        # No solution found, return all zeros
        for cls in classes:
            i = cls['class_id']
            item_list = []
            for item in cls['items']:
                j = item['item_id']
                item_list.append({
                    "item_id": j,
                    "x_ij": 0
                })
            solution_classes.append({
                "class_id": i,
                "y_i": 0,
                "items": item_list
            })

    output = {
        "classes": solution_classes,
        "objective_value": float(final_obj)
    }

    with open(args.solution_path, 'w') as f:
        json.dump(output, f, indent=2)

if __name__ == "__main__":
    main()
import json
import argparse
from gurobipy import Model, GRB, quicksum

def main():
    parser = argparse.ArgumentParser(description='Solve the knapsack problem with setup costs and classes.')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the instance JSON file.')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to write the solution JSON file.')
    parser.add_argument('--time_limit', type=int, required=True, help='Maximum runtime in seconds.')
    parser.add_argument('--log_path', type=str, default=None, help='Path to log intermediate solutions.')
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        data = json.load(f)
    
    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="maximize")
    else:
        logger = None

    model = Model("knapsack_with_setups")
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('Threads', 1)
    model.setParam('OutputFlag', 0)

    y = {}
    x = {}

    for cls in data['classes']:
        class_id = cls['class_id']
        y[class_id] = model.addVar(vtype=GRB.BINARY, name=f"y_{class_id}")
        x[class_id] = {}
        for item in cls['items']:
            item_id = item['item_id']
            x[class_id][item_id] = model.addVar(vtype=GRB.BINARY, name=f"x_{class_id}_{item_id}")

    for cls in data['classes']:
        class_id = cls['class_id']
        for item in cls['items']:
            model.addConstr(x[class_id][item['item_id']] <= y[class_id])

    capacity_expr = quicksum(cls['d_i'] * y[cls['class_id']] for cls in data['classes'])
    for cls in data['classes']:
        class_id = cls['class_id']
        for item in cls['items']:
            capacity_expr += item['a_ij'] * x[class_id][item['item_id']]
    model.addConstr(capacity_expr <= data['b'])

    obj_expr = quicksum(cls['f_i'] * y[cls['class_id']] for cls in data['classes'])
    for cls in data['classes']:
        class_id = cls['class_id']
        for item in cls['items']:
            obj_expr += item['c_ij'] * x[class_id][item['item_id']]
    model.setObjective(obj_expr, GRB.MAXIMIZE)

    if logger:
        def callback(m, where):
            if where == GRB.Callback.MIPSOL:
                obj_val = m.cbGet(GRB.Callback.MIPSOL_OBJ)
                logger.log(obj_val)
        model.optimize(callback)
    else:
        model.optimize()

    if model.SolCount > 0:
        sol = {
            "classes": [],
            "objective_value": float(model.ObjVal)
        }
        for cls in data['classes']:
            class_id = cls['class_id']
            y_val = int(round(y[class_id].X))
            items_sol = []
            for item in cls['items']:
                item_id = item['item_id']
                x_val = int(round(x[class_id][item_id].X))
                items_sol.append({
                    "item_id": item_id,
                    "x_ij": x_val
                })
            sol['classes'].append({
                "class_id": class_id,
                "y_i": y_val,
                "items": items_sol
            })
    else:
        sol = {
            "classes": [],
            "objective_value": 0.0
        }
        for cls in data['classes']:
            class_id = cls['class_id']
            items_sol = []
            for item in cls['items']:
                items_sol.append({
                    "item_id": item['item_id'],
                    "x_ij": 0
                })
            sol['classes'].append({
                "class_id": class_id,
                "y_i": 0,
                "items": items_sol
            })

    with open(args.solution_path, 'w') as f:
        json.dump(sol, f, indent=2)

if __name__ == "__main__":
    main()
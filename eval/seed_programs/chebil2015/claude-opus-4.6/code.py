import argparse
import json
import time
import sys

def solve():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    N = instance['N']
    b = instance['b']
    classes = instance['classes']

    start_time = time.time()

    try:
        import gurobipy as gp
        from gurobipy import GRB

        model = gp.Model("knapsack_setup")
        model.setParam('TimeLimit', max(1, args.time_limit - 2))
        model.setParam('Threads', 1)
        model.setParam('MIPFocus', 1)  # Focus on finding good feasible solutions

        # Decision variables
        y = {}  # class activation
        x = {}  # item selection

        for cls in classes:
            cid = cls['class_id']
            y[cid] = model.addVar(vtype=GRB.BINARY, name=f"y_{cid}")
            for item in cls['items']:
                iid = item['item_id']
                x[(cid, iid)] = model.addVar(vtype=GRB.BINARY, name=f"x_{cid}_{iid}")

        model.update()

        # Objective: maximize sum of profits - setup costs
        obj = gp.LinExpr()
        for cls in classes:
            cid = cls['class_id']
            obj += cls['f_i'] * y[cid]  # f_i is negative, so this subtracts
            for item in cls['items']:
                iid = item['item_id']
                obj += item['c_ij'] * x[(cid, iid)]

        model.setObjective(obj, GRB.MAXIMIZE)

        # Capacity constraint
        cap_expr = gp.LinExpr()
        for cls in classes:
            cid = cls['class_id']
            cap_expr += cls['d_i'] * y[cid]
            for item in cls['items']:
                iid = item['item_id']
                cap_expr += item['a_ij'] * x[(cid, iid)]

        model.addConstr(cap_expr <= b, "capacity")

        # Linking constraints: item can only be selected if class is activated
        for cls in classes:
            cid = cls['class_id']
            for item in cls['items']:
                iid = item['item_id']
                model.addConstr(x[(cid, iid)] <= y[cid], f"link_{cid}_{iid}")

        # Callback to log incumbent solutions
        def callback(model, where):
            if where == GRB.Callback.MIPSOL:
                obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if logger:
                    logger.log(obj_val)

        model.optimize(callback)

        # Extract solution
        solution_classes = []
        total_obj = 0.0

        if model.SolCount > 0:
            total_obj = model.ObjVal
            for cls in classes:
                cid = cls['class_id']
                y_val = int(round(y[cid].X))
                items_sol = []
                for item in cls['items']:
                    iid = item['item_id']
                    x_val = int(round(x[(cid, iid)].X))
                    items_sol.append({
                        "item_id": iid,
                        "x_ij": x_val
                    })
                solution_classes.append({
                    "class_id": cid,
                    "y_i": y_val,
                    "items": items_sol
                })
        else:
            # No feasible solution found, return all zeros
            total_obj = 0.0
            for cls in classes:
                cid = cls['class_id']
                items_sol = []
                for item in cls['items']:
                    iid = item['item_id']
                    items_sol.append({
                        "item_id": iid,
                        "x_ij": 0
                    })
                solution_classes.append({
                    "class_id": cid,
                    "y_i": 0,
                    "items": items_sol
                })

        solution = {
            "classes": solution_classes,
            "objective_value": float(total_obj)
        }

        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)

    except Exception as e:
        # Fallback: greedy approach
        print(f"Gurobi failed: {e}, falling back to greedy", file=sys.stderr)
        
        # Greedy: for each class, compute the best subset of items if class is activated
        # Then use a knapsack-like greedy over classes
        
        class_options = []
        for cls in classes:
            cid = cls['class_id']
            f_i = cls['f_i']
            d_i = cls['d_i']
            
            # Sort items by profit/weight ratio (descending)
            items_sorted = sorted(cls['items'], 
                                  key=lambda it: it['c_ij'] / max(it['a_ij'], 1e-9), 
                                  reverse=True)
            
            # Greedily pick items
            remaining_cap = b - d_i  # hypothetical
            total_profit = f_i  # start with setup cost
            selected_items = []
            
            for item in items_sorted:
                if item['a_ij'] <= remaining_cap and item['c_ij'] > 0:
                    selected_items.append(item['item_id'])
                    total_profit += item['c_ij']
                    remaining_cap -= item['a_ij']
            
            total_weight = d_i + sum(it['a_ij'] for it in cls['items'] if it['item_id'] in selected_items)
            
            if total_profit > 0:
                class_options.append((cid, total_profit, total_weight, selected_items))
        
        # Sort by profit density
        class_options.sort(key=lambda c: c[1] / max(c[2], 1e-9), reverse=True)
        
        remaining = b
        activated = {}
        selected = {}
        total_obj = 0.0
        
        for cid, profit, weight, items_sel in class_options:
            if weight <= remaining:
                activated[cid] = 1
                selected[cid] = set(items_sel)
                remaining -= weight
                total_obj += profit
        
        if logger:
            logger.log(total_obj)
        
        solution_classes = []
        for cls in classes:
            cid = cls['class_id']
            y_val = activated.get(cid, 0)
            items_sol = []
            sel_set = selected.get(cid, set())
            for item in cls['items']:
                iid = item['item_id']
                x_val = 1 if iid in sel_set else 0
                items_sol.append({"item_id": iid, "x_ij": x_val})
            solution_classes.append({
                "class_id": cid,
                "y_i": y_val,
                "items": items_sol
            })
        
        solution = {
            "classes": solution_classes,
            "objective_value": float(total_obj)
        }
        
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)

if __name__ == "__main__":
    solve()
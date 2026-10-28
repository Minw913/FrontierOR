import argparse
import json
import time
import sys
import gurobipy as gp
from gurobipy import GRB

def main():
    start_time = time.time()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, required=False)
    args = parser.parse_args()

    # Solution Logger
    try:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None
    except ImportError:
        class SolutionLogger:
            def __init__(self, *args, **kwargs): pass
            def log(self, obj): pass
        logger = SolutionLogger()

    # Load instance
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    W = instance['panel']['length']
    H = instance['panel']['width']
    items = instance.get('items', [])

    if not items:
        solution = {"objective_value": 0.0, "items_selected": []}
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
        return

    time_limit = args.time_limit if args.time_limit else 60

    min_L = min((item['length'] for item in items), default=W)
    min_W = min((item['width'] for item in items), default=H)

    # 1. DP to find canonical lengths (combinations of item lengths)
    dp_x = [False] * (W + 1)
    dp_x[0] = True
    for i in range(1, W + 1):
        for item in items:
            L = item['length']
            if i >= L and dp_x[i - L]:
                dp_x[i] = True
    X_norm = [i for i in range(1, W + 1) if dp_x[i]]

    # 2. DP to find canonical widths (combinations of item widths)
    dp_y = [False] * (H + 1)
    dp_y[0] = True
    for i in range(1, H + 1):
        for item in items:
            w_item = item['width']
            if i >= w_item and dp_y[i - w_item]:
                dp_y[i] = True
    Y_norm = [i for i in range(1, H + 1) if dp_y[i]]

    x_norm_sorted = sorted([x for x in X_norm if x >= min_L and x <= W])
    y_norm_sorted = sorted([y for y in Y_norm if y >= min_W and y <= H])

    # 3. Explore state space DAG (top-down, ensuring strictly valid cuts)
    memo_vars = set()
    memo_vars.add((W, H))
    stack = [(W, H)]

    while stack:
        # Time safety bound: Truncate exploration if taking too long to guarantee formulation solves.
        if len(memo_vars) % 2000 == 0:
            if time.time() - start_time > time_limit * 0.15:
                break

        w, h = stack.pop()

        seen_x = set()
        for x in x_norm_sorted:
            if x >= w: break
            w_x = w - x
            pair = (min(x, w_x), max(x, w_x))
            if pair not in seen_x:
                seen_x.add(pair)
                if x >= min_L and (x, h) not in memo_vars:
                    memo_vars.add((x, h))
                    stack.append((x, h))
                if w_x >= min_L and (w_x, h) not in memo_vars:
                    memo_vars.add((w_x, h))
                    stack.append((w_x, h))

        seen_y = set()
        for y in y_norm_sorted:
            if y >= h: break
            h_y = h - y
            pair = (min(y, h_y), max(y, h_y))
            if pair not in seen_y:
                seen_y.add(pair)
                if y >= min_W and (w, y) not in memo_vars:
                    memo_vars.add((w, y))
                    stack.append((w, y))
                if h_y >= min_W and (w, h_y) not in memo_vars:
                    memo_vars.add((w, h_y))
                    stack.append((w, h_y))

    # 4. Generate ILP variables properly bounded to explored DAG
    E_keys = []
    E_obj = {}
    for w, h in memo_vars:
        for item in items:
            if item['length'] == w and item['width'] == h:
                E_keys.append((w, h, item['id']))
                E_obj[(w, h, item['id'])] = item['profit']

    V_keys = []
    for w, h in memo_vars:
        seen_x = set()
        for x in x_norm_sorted:
            if x >= w: break
            w_x = w - x
            if x < min_L and w_x < min_L: continue 
            pair = (min(x, w_x), max(x, w_x))
            if pair not in seen_x:
                seen_x.add(pair)
                valid = True
                if x >= min_L and (x, h) not in memo_vars: valid = False
                if w_x >= min_L and (w_x, h) not in memo_vars: valid = False
                if valid:
                    V_keys.append((w, h, x))

    H_keys = []
    for w, h in memo_vars:
        seen_y = set()
        for y in y_norm_sorted:
            if y >= h: break
            h_y = h - y
            if y < min_W and h_y < min_W: continue
            pair = (min(y, h_y), max(y, h_y))
            if pair not in seen_y:
                seen_y.add(pair)
                valid = True
                if y >= min_W and (w, y) not in memo_vars: valid = False
                if h_y >= min_W and (w, h_y) not in memo_vars: valid = False
                if valid:
                    H_keys.append((w, h, y))

    # 5. Build and Solve Gurobi Model
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    m = gp.Model(env=env)
    m.setParam("Threads", 1)

    elapsed = time.time() - start_time
    rem = max(1, time_limit - elapsed - 2)
    m.Params.TimeLimit = rem
    m.Params.MIPFocus = 1 

    E_vars = m.addVars(E_keys, vtype=GRB.INTEGER, obj=E_obj, name="E")
    V_vars = m.addVars(V_keys, vtype=GRB.INTEGER, obj=0.0, name="V")
    H_vars = m.addVars(H_keys, vtype=GRB.INTEGER, obj=0.0, name="H")

    inflow = {state: gp.LinExpr() for state in memo_vars}
    outflow = {state: gp.LinExpr() for state in memo_vars}

    inflow[(W, H)] += 1.0

    for w, h, x in V_keys:
        var = V_vars[w, h, x]
        outflow[(w, h)] += var
        if x >= min_L: inflow[(x, h)] += var
        if w - x >= min_L: inflow[(w - x, h)] += var

    for w, h, y in H_keys:
        var = H_vars[w, h, y]
        outflow[(w, h)] += var
        if y >= min_W: inflow[(w, y)] += var
        if h - y >= min_W: inflow[(w, h - y)] += var

    for w, h, k in E_keys:
        var = E_vars[w, h, k]
        outflow[(w, h)] += var

    m.addConstrs((outflow[state] <= inflow[state] for state in memo_vars), name="flow")
    
    for item in items:
        m.addConstr(E_vars.sum('*', '*', item['id']) <= item['copies'], name=f"copies_{item['id']}")
    
    m.ModelSense = GRB.MAXIMIZE

    def progress_callback(model, where):
        if where == GRB.Callback.MIPSOL:
            try:
                obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if logger:
                    logger.log(obj)
            except Exception:
                pass

    m.optimize(progress_callback)

    # 6. Assemble solution formatted cleanly payload
    obj_val = 0.0
    items_selected = []

    if m.SolCount > 0:
        obj_val = m.ObjVal
        if logger:
            logger.log(obj_val)
            
        final_items = {}
        for w, h, k in E_keys:
            val = E_vars[w, h, k].X
            if val >= 0.5:
                qty = int(round(val))
                if k not in final_items:
                    final_items[k] = {"item_id": k, "plate_dims": [w, h], "copies": qty}
                else:
                    final_items[k]["copies"] += qty
        items_selected = list(final_items.values())

    solution = {
        "objective_value": float(obj_val),
        "items_selected": items_selected
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()
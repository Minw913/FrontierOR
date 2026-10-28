import sys
import json
import argparse
import gurobipy as gp

try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None

class CallbackData:
    def __init__(self):
        self.best_obj = float('inf')

cb_data = CallbackData()

def my_callback(model, where):
    if where == gp.GRB.Callback.MIPSOL:
        try:
            x_vals = model.cbGetSolution(model._x)
            y_vals = model.cbGetSolution(model._y)
            z_vals = model.cbGetSolution(model._z)
            
            lazy_added = False
            
            for t in range(1, model._H + 1):
                if z_vals[t] < 0.5:
                    continue
                    
                edges = []
                vertices = set()
                for i in range(model._N + 1):
                    for j in range(i + 1, model._N + 1):
                        if x_vals[t, i, j] > 0.5:
                            edges.append((i, j))
                            vertices.add(i)
                            vertices.add(j)
                            
                if not edges:
                    continue
                    
                # Build adjacency list
                adj = {n: [] for n in vertices}
                for u, v in edges:
                    adj[u].append(v)
                    adj[v].append(u)
                    
                comps = []
                visited = set()
                for n in adj:
                    if n not in visited:
                        comp = set()
                        q = [n]
                        while q:
                            curr = q.pop(0)
                            if curr not in comp:
                                comp.add(curr)
                                visited.add(curr)
                                for nbr in adj[curr]:
                                    if nbr not in comp:
                                        q.append(nbr)
                        comps.append(comp)
                        
                # Check each pure-retailer component for SEC violations
                for comp in comps:
                    if 0 not in comp and len(comp) >= 2:
                        k = list(comp)[0]
                        expr = gp.LinExpr()
                        for i in comp:
                            for j in comp:
                                if i < j:
                                    if (t, i, j) in model._x:
                                        expr.addTerms(1.0, model._x[t, i, j])
                        
                        y_sum = gp.LinExpr()
                        for i in comp:
                            if (i, t) in model._y:
                                y_sum.addTerms(1.0, model._y[i, t])
                                
                        model.cbLazy(expr <= y_sum - model._y[k, t])
                        lazy_added = True
                        
            # If no lazy constraint was generated, we found a valid integer solution
            if not lazy_added and model._logger is not None:
                obj = model.cbGet(gp.GRB.Callback.MIPSOL_OBJ)
                if obj < cb_data.best_obj - 1e-5:
                    cb_data.best_obj = obj
                    model._logger.log(obj)
                    
        except Exception:
            pass

def solve():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, default=300)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    N = data["num_retailers"]
    H = data["time_horizon"]
    supplier = data["supplier"]
    retailers = data["retailers"]
    vehicle_capacity = data["vehicle_capacity"]
    distance_matrix = data["distance_matrix"]

    ret_dict = {ret["id"]: ret for ret in retailers}

    env = gp.Env(empty=True)
    env.start()
    model = gp.Model("IRP", env=env)
    
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('Threads', 1)
    model.setParam('LazyConstraints', 1)
    
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path and SolutionLogger else None
    
    # Define variables
    I = model.addVars(
        [(i, t) for i in range(1, N + 1) for t in range(1, H + 2)],
        lb=0.0,
        vtype=gp.GRB.CONTINUOUS,
        name="I"
    )
    for i in range(1, N + 1):
        for t in range(1, H + 2):
            I[i, t].ub = ret_dict[i]["max_inventory"]

    I0 = model.addVars(
        range(1, H + 2),
        lb=0.0,
        vtype=gp.GRB.CONTINUOUS,
        name="I0"
    )

    Q = model.addVars(
        [(i, t) for i in range(1, N + 1) for t in range(1, H + 1)],
        lb=0.0,
        vtype=gp.GRB.CONTINUOUS,
        name="Q"
    )

    y = model.addVars(
        [(i, t) for i in range(1, N + 1) for t in range(1, H + 1)],
        vtype=gp.GRB.BINARY,
        name="y"
    )

    z = model.addVars(
        range(1, H + 1),
        vtype=gp.GRB.BINARY,
        name="z"
    )

    x = model.addVars(
        [(t, i, j) for t in range(1, H + 1) for i in range(N + 1) for j in range(i + 1, N + 1)],
        vtype=gp.GRB.INTEGER,
        lb=0,
        name="x"
    )
    
    for t in range(1, H + 1):
        for j in range(1, N + 1):
            x[t, 0, j].ub = 2
        for i in range(1, N + 1):
            for j in range(i + 1, N + 1):
                x[t, i, j].ub = 1

    # Constraints
    # 1. Retailer inventory balances & bounds
    for i in range(1, N + 1):
        ret = ret_dict[i]
        Ui = ret["max_inventory"]
        Di = ret["demand"]
        
        model.addConstr(I[i, 1] == ret["starting_inventory"])
        
        for t in range(1, H + 1):
            model.addConstr(I[i, t+1] == I[i, t] + Q[i, t] - Di)
            
            # Delivery quantity (Order-up-to level policy exactly when visited, 0 else)
            model.addConstr(Q[i, t] >= Ui - I[i, t] - Ui * (1 - y[i, t]))
            model.addConstr(Q[i, t] <= Ui * y[i, t])
            model.addConstr(Q[i, t] <= Ui - I[i, t] + Ui * (1 - y[i, t]))

    # 2. Supplier inventory balances & delivery bounds
    model.addConstr(I0[1] == supplier["starting_inventory"])
    P = supplier["production_rate"]
    
    for t in range(1, H + 1):
        sum_Q = gp.quicksum(Q[i, t] for i in range(1, N + 1))
        model.addConstr(I0[t+1] == I0[t] + P - sum_Q)
        model.addConstr(I0[t] >= sum_Q)
        model.addConstr(sum_Q <= vehicle_capacity)

    # 3. Routing variables constraints
    for t in range(1, H + 1):
        for i in range(1, N + 1):
            model.addConstr(z[t] >= y[i, t])
        model.addConstr(z[t] <= gp.quicksum(y[i, t] for i in range(1, N + 1)))
        
        # Proper vertex degree
        for i in range(1, N + 1):
            deg = gp.quicksum(x[t, k, i] for k in range(i)) + gp.quicksum(x[t, i, j] for j in range(i + 1, N + 1))
            model.addConstr(deg == 2 * y[i, t])
            
        model.addConstr(gp.quicksum(x[t, 0, j] for j in range(1, N + 1)) == 2 * z[t])
        
        # 2-node symmetric SEC & continuous bounds
        for i in range(1, N + 1):
            for j in range(i + 1, N + 1):
                model.addConstr(x[t, i, j] <= y[i, t])
                model.addConstr(x[t, i, j] <= y[j, t])

    # 4. Objective definition
    h0 = supplier["inventory_cost"]
    obj_inv_supp = gp.quicksum(h0 * I0[t] for t in range(1, H + 2))
    
    obj_inv_ret = gp.LinExpr()
    for i in range(1, N + 1):
        hi = ret_dict[i]["inventory_cost"]
        obj_inv_ret += gp.quicksum(hi * I[i, t] for t in range(1, H + 2))
        
    obj_routing = gp.LinExpr()
    for t in range(1, H + 1):
        for i in range(N + 1):
            for j in range(i + 1, N + 1):
                obj_routing += distance_matrix[i][j] * x[t, i, j]

    model.setObjective(obj_inv_supp + obj_inv_ret + obj_routing, gp.GRB.MINIMIZE)

    # Save components to model for callback usage
    model._x = x
    model._y = y
    model._z = z
    model._N = N
    model._H = H
    model._logger = logger
    
    # Solve the model with lazy subtour breaks
    model.optimize(my_callback)

    # Extract format
    result = {}
    if model.SolCount > 0:
        result["objective_value"] = float(model.ObjVal)
        result["deliveries"] = {}
        result["routes"] = {}
        result["supplier_inventory"] = {}
        result["retailer_inventory"] = {}

        for t in range(1, H + 1):
            str_t = str(t)
            result["deliveries"][str_t] = {}
            for i in range(1, N + 1):
                if y[i, t].X > 0.5:
                    val = max(0.0, Q[i, t].X)
                    result["deliveries"][str_t][str(i)] = round(val, 2)
                    
            route_t = []
            for i in range(N + 1):
                for j in range(i + 1, N + 1):
                    if x[t, i, j].X > 0.5:
                        k_val = int(round(x[t, i, j].X))
                        if k_val > 0:
                            route_t.append([i, j, k_val])
            result["routes"][str_t] = route_t
            
        for t in range(1, H + 2):
            result["supplier_inventory"][str(t)] = round(max(0.0, I0[t].X), 2)
            
        for i in range(1, N + 1):
            str_i = str(i)
            result["retailer_inventory"][str_i] = {}
            for t in range(1, H + 2):
                result["retailer_inventory"][str_i][str(t)] = round(max(0.0, I[i, t].X), 2)
    else:
        # Fallback when absolutely no solution is found
        result["objective_value"] = -1.0
        result["deliveries"] = {str(t): {} for t in range(1, H + 1)}
        result["routes"] = {str(t): [] for t in range(1, H + 1)}
        result["supplier_inventory"] = {str(t): 0.0 for t in range(1, H + 2)}
        result["retailer_inventory"] = {str(i): {str(t): 0.0 for t in range(1, H + 2)} for i in range(1, N + 1)}
        
    with open(args.solution_path, 'w') as f:
        json.dump(result, f, indent=2)

if __name__ == "__main__":
    solve()
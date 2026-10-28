import sys
import json
import argparse
import gurobipy as gp
from gurobipy import GRB

def build_solution_dict(model, cb_get_solution=None):
    sol = {
        "objective_value": 0.0,
        "model_variables": {},
        "y": {},
        "z": {},
        "x": {},
        "model_variable_name_patterns": {
            "y_{t}": "setup in one-based period t",
            "z_{i}_{k}_{t}": "visit node i by vehicle k in period t",
            "x_{i}_{j}_{k}_{t}": "edge traversal",
            "p_{t}_{omega}": "scenario production",
            "I_{i}_{t}_{omega}": "scenario inventory",
            "e_{i}_{t}_{omega}": "scenario emergency quantity",
            "q_{i}_{k}_{t}_{omega}": "scenario delivery quantity"
        }
    }
    
    def get_val(var):
        if cb_get_solution:
            return cb_get_solution(var)
        else:
            return var.X

    for var in model._all_vars:
        try:
            val = get_val(var)
        except gp.GurobiError:
            continue
            
        if abs(val) > 1e-4:
            name = var.VarName
            sol["model_variables"][name] = val
            if name.startswith("y_"):
                parts = name.split('_')
                sol["y"][parts[1]] = int(round(val))
            elif name.startswith("z_"):
                parts = name.split('_')
                sol["z"][f"{parts[1]}_{parts[2]}_{parts[3]}"] = int(round(val))
            elif name.startswith("x_"):
                parts = name.split('_')
                sol["x"][f"{parts[1]}_{parts[2]}_{parts[3]}_{parts[4]}"] = int(round(val))
                
    return sol

best_obj = float('inf')

def subtour_elimination(model, where):
    global best_obj
    if where == GRB.Callback.MIPSOL:
        try:
            x_vals = model.cbGetSolution(model._x)
            z_vals = model.cbGetSolution(model._z)
            
            subtours_added = 0
            for t in model._T_set:
                for k in model._V:
                    if z_vals.get((0, k, t), 0) < 0.5:
                        continue
                        
                    active_nodes = [i for i in model._C if z_vals.get((i, k, t), 0) > 0.5]
                    if not active_nodes:
                        continue
                        
                    adj = {i: [] for i in model._N}
                    for i in model._N:
                        for j in range(i + 1, model._n + 1):
                            if x_vals.get((i, j, k, t), 0) > 0.5:
                                adj[i].append(j)
                                adj[j].append(i)
                                
                    visited = set()
                    for i in active_nodes:
                        if i not in visited:
                            comp = []
                            q_nodes = [i]
                            visited.add(i)
                            while q_nodes:
                                curr = q_nodes.pop(0)
                                comp.append(curr)
                                for nxt in adj[curr]:
                                    if nxt != 0 and nxt not in visited and nxt in active_nodes:
                                        visited.add(nxt)
                                        q_nodes.append(nxt)
                            
                            has_zero = any(0 in adj[node] for node in comp)
                            
                            if not has_zero and len(comp) >= 2:
                                v = comp[0]
                                expr = gp.LinExpr()
                                for idx1 in range(len(comp)):
                                    for idx2 in range(idx1 + 1, len(comp)):
                                        ni, nj = comp[idx1], comp[idx2]
                                        if ni > nj:
                                            ni, nj = nj, ni
                                        if (ni, nj, k, t) in model._x:
                                            expr.add(model._x[ni, nj, k, t])
                                
                                rhs = gp.LinExpr()
                                for node in comp:
                                    if node != v:
                                        rhs.add(model._z[node, k, t])
                                
                                model.cbLazy(expr <= rhs)
                                subtours_added += 1
                                
            if subtours_added == 0:
                obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if obj_val < best_obj - 1e-5:
                    best_obj = obj_val
                    if model._logger is not None:
                        sol_dict = build_solution_dict(model, cb_get_solution=model.cbGetSolution)
                        sol_dict["objective_value"] = obj_val
                        model._logger.log_solution(obj_val, sol_dict)
        except Exception as e:
            pass

def solve():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default="")
    args = parser.parse_args()

    try:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    except ImportError:
        logger = None

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    n = data["n"]
    T = data["T"]
    m = data["m"]
    Q = data["Q"]
    C_cap = data["C"]
    f_setup = data["f"]
    u_prod = data["u"]
    c = data["transportation_costs"]
    h = data["h"]
    L = data["L"]
    I0 = data["I0"]
    sigma = data["sigma"]
    n_scenarios = data["n_scenarios"]
    probs = data["scenario_probabilities"]
    demands = data["demand_scenarios"]

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    model = gp.Model("PRP_Demand_Uncertainty", env=env)

    model.setParam("TimeLimit", args.time_limit)
    model.setParam("Seed", 0)
    model.setParam("MIPGap", 1e-4)
    model.setParam("NumericFocus", 0)
    model.setParam("Threads", 1)
    model.setParam("LazyConstraints", 1)

    N = list(range(n + 1))
    C = list(range(1, n + 1))
    V = list(range(1, m + 1))
    T_set = list(range(1, T + 1))
    Omega = list(range(n_scenarios))

    y, z, x, p, I, e, q = {}, {}, {}, {}, {}, {}, {}
    all_vars = []

    for t in T_set:
        y[t] = model.addVar(vtype=GRB.BINARY, name=f"y_{t}")
        all_vars.append(y[t])
        for k in V:
            for i in N:
                z[i, k, t] = model.addVar(vtype=GRB.BINARY, name=f"z_{i}_{k}_{t}")
                all_vars.append(z[i, k, t])
            for i in N:
                for j in range(i + 1, n + 1):
                    ub = 2 if i == 0 else 1
                    x[i, j, k, t] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=ub, name=f"x_{i}_{j}_{k}_{t}")
                    all_vars.append(x[i, j, k, t])

    for t in T_set:
        for omega in Omega:
            p[t, omega] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"p_{t}_{omega}")
            all_vars.append(p[t, omega])
            for i in N:
                I[i, t, omega] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"I_{i}_{t}_{omega}")
                all_vars.append(I[i, t, omega])
            for i in C:
                e[i, t, omega] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"e_{i}_{t}_{omega}")
                all_vars.append(e[i, t, omega])
                for k in V:
                    q[i, k, t, omega] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"q_{i}_{k}_{t}_{omega}")
                    all_vars.append(q[i, k, t, omega])

    # Store for callbacks
    model._all_vars = all_vars
    model._x = x
    model._z = z
    model._n = n
    model._N = N
    model._C = C
    model._V = V
    model._T_set = T_set
    model._logger = logger

    obj = gp.LinExpr()
    for t in T_set:
        obj.add(f_setup * y[t])
        for k in V:
            for i in N:
                for j in range(i + 1, n + 1):
                    obj.add(c[i][j] * x[i, j, k, t])

    for omega in Omega:
        prob = probs[omega]
        scen_obj = gp.LinExpr()
        for t in T_set:
            scen_obj.add(u_prod * p[t, omega])
            for i in N:
                scen_obj.add(h[i] * I[i, t, omega])
            for i in C:
                scen_obj.add(sigma[i-1] * e[i, t, omega])
        obj.add(prob * scen_obj)

    model.setObjective(obj, GRB.MINIMIZE)

    for t in T_set:
        for k in V:
            for i in N:
                expr = gp.LinExpr()
                for j in range(i + 1, n + 1):
                    expr.add(x[i, j, k, t])
                for j in range(0, i):
                    expr.add(x[j, i, k, t])
                model.addConstr(expr == 2 * z[i, k, t])
                
            for i in C:
                model.addConstr(z[i, k, t] <= z[0, k, t])

    for t in T_set:
        for i in C:
            model.addConstr(gp.quicksum(z[i, k, t] for k in V) <= 1)
        for k in range(1, m):
            model.addConstr(z[0, k, t] >= z[0, k + 1, t])

    for omega in Omega:
        for t in T_set:
            I_prev_0 = I0[0] if t == 1 else I[0, t - 1, omega]
            sum_q_all = gp.quicksum(q[i, k, t, omega] for i in C for k in V)
            model.addConstr(I_prev_0 + p[t, omega] == sum_q_all + I[0, t, omega])
            model.addConstr(I[0, t, omega] <= L[0])
            model.addConstr(p[t, omega] <= C_cap * y[t])
            
            for i in C:
                I_prev_i = I0[i] if t == 1 else I[i, t - 1, omega]
                sum_q_i = gp.quicksum(q[i, k, t, omega] for k in V)
                d_val = demands[omega][i-1][t-1]
                
                model.addConstr(I_prev_i + sum_q_i + e[i, t, omega] == d_val + I[i, t, omega])
                model.addConstr(I_prev_i + sum_q_i <= L[i])
                
                for k in V:
                    model.addConstr(q[i, k, t, omega] <= Q * z[i, k, t])
                    future_d = sum(demands[omega][i-1][tau-1] for tau in range(t, T + 1))
                    model.addConstr(q[i, k, t, omega] <= future_d * z[i, k, t])

            for k in V:
                model.addConstr(gp.quicksum(q[i, k, t, omega] for i in C) <= Q * z[0, k, t])

    model.optimize(subtour_elimination)

    if model.SolCount > 0:
        sol_dict = build_solution_dict(model)
        sol_dict["objective_value"] = model.ObjVal
        with open(args.solution_path, 'w') as f:
            f.write(json.dumps(sol_dict, indent=2))

if __name__ == "__main__":
    solve()
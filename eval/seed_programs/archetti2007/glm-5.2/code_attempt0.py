import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def solve(args):
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    n = instance['num_retailers']
    H = instance['time_horizon']
    supplier = instance['supplier']
    retailers = instance['retailers']
    C = instance['vehicle_capacity']
    dist = instance['distance_matrix']
    
    I_S0 = supplier['starting_inventory']
    P = supplier['production_rate']
    c_S = supplier['inventory_cost']
    
    I_R0 = [r['starting_inventory'] for r in retailers]
    d = [r['demand'] for r in retailers]
    M = [r['max_inventory'] for r in retailers]
    c_R = [r['inventory_cost'] for r in retailers]
    
    H_ext = H + 1
    M_big = max((M[i] + H * d[i] for i in range(n)), default=0)
    
    model = gp.Model("IRP")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    if args.time_limit:
        model.Params.TimeLimit = args.time_limit
    
    y = {}
    z = {}
    S = {}
    Z = {}
    x = {}
    pos = {}
    
    for t in range(1, H + 1):
        for i in range(n):
            y[i, t] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{t}")
        z[t] = model.addVar(vtype=GRB.BINARY, name=f"z_{t}")
        
    for i in range(n):
        S[i, 0] = model.addVar(lb=0, ub=0, name=f"S_{i}_0")
        for t in range(1, H + 1):
            S[i, t] = model.addVar(lb=0, name=f"S_{i}_{t}")
            
    Z[0] = model.addVar(lb=0, ub=0, name="Z_0")
    for t in range(1, H + 1):
        Z[t] = model.addVar(lb=0, name=f"Z_{t}")
        
    for t in range(1, H + 1):
        for i in range(n + 1):
            for j in range(n + 1):
                if i != j:
                    x[i, j, t] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}_{t}")
        for i in range(1, n + 1):
            pos[i, t] = model.addVar(lb=0, name=f"pos_{i}_{t}")
            
    model.update()
    
    for t in range(1, H + 1):
        for i in range(n):
            S[i, t].setAttr(GRB.Attr.Ub, M[i] - I_R0[i] + t * d[i])
            model.addConstr(S[i, t] >= M[i] - I_R0[i] + t * d[i] - M_big * (1 - y[i, t]))
            model.addConstr(S[i, t] - S[i, t - 1] <= M_big * y[i, t])
            model.addConstr(S[i, t] >= S[i, t - 1])
            model.addConstr(S[i, t] >= (t + 1) * d[i] - I_R0[i])
            
        model.addConstr(Z[t] == gp.quicksum(S[i, t] for i in range(n)))
        model.addConstr(Z[t] <= I_S0 + t * P)
        model.addConstr(Z[t] - Z[t - 1] <= C * z[t])
        
        for i in range(n):
            model.addConstr(z[t] >= y[i, t])
        model.addConstr(z[t] <= gp.quicksum(y[i, t] for i in range(n)))
        
        for i in range(1, n + 1):
            model.addConstr(gp.quicksum(x[j, i, t] for j in range(n + 1) if j != i) == y[i - 1, t])
            model.addConstr(gp.quicksum(x[i, j, t] for j in range(n + 1) if j != i) == y[i - 1, t])
        model.addConstr(gp.quicksum(x[0, j, t] for j in range(1, n + 1)) == z[t])
        model.addConstr(gp.quicksum(x[j, 0, t] for j in range(1, n + 1)) == z[t])
        
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                if i != j:
                    model.addConstr(x[i, j, t] <= y[i - 1, t])
                    model.addConstr(pos[j, t] >= pos[i, t] + 1 - n * (1 - x[i, j, t]))
        for i in range(1, n + 1):
            model.addConstr(pos[i, t] >= 1 - n * (1 - y[i - 1, t]))
            model.addConstr(pos[i, t] <= n * y[i - 1, t])
            model.addConstr(x[0, i, t] <= z[t])
            model.addConstr(x[i, 0, t] <= z[t])
            
    obj = gp.LinExpr()
    for t in range(1, H + 1):
        for i in range(n + 1):
            for j in range(n + 1):
                if i != j:
                    obj += dist[i][j] * x[i, j, t]
    for t in range(0, H + 1):
        obj -= c_S * Z[t]
        for i in range(n):
            obj += c_R[i] * S[i, t]
            
    const = c_S * ((H + 1) * I_S0 + P * (H + 1) * (H + 2) / 2)
    for i in range(n):
        const += c_R[i] * ((H + 1) * I_R0[i] - d[i] * (H + 1) * (H + 2) / 2)
    obj += const
    
    model.setObjective(obj, GRB.MINIMIZE)
    
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    def callback(model, where):
        if where == GRB.Callback.MIP and model.cbGet(GRB.Callback.MIP_SOLCNT) > 0:
            objbst = model.cbGet(GRB.Callback.MIP_OBJBST)
            if logger:
                logger.log(objbst)
    
    model.optimize(callback)
    
    solution = {}
    if model.SolCount > 0:
        obj_val = model.ObjVal
        solution['objective_value'] = obj_val
        
        deliveries = {}
        for t in range(1, H + 1):
            t_deliveries = {}
            for i in range(n):
                if y[i, t].X > 0.5:
                    val = S[i, t].X - S[i, t - 1].X
                    t_deliveries[str(retailers[i]['id'])] = val
            if t_deliveries:
                deliveries[str(t)] = t_deliveries
        solution['deliveries'] = deliveries
        
        routes = {}
        for t in range(1, H + 1):
            t_routes = []
            if z[t].X > 0.5:
                for i in range(n + 1):
                    for j in range(i + 1, n + 1):
                        if i == 0:
                            k = round(x[0, j, t].X + x[j, 0, t].X)
                        else:
                            k = round(x[i, j, t].X + x[j, i, t].X)
                        if k > 0:
                            t_routes.append([i, j, k])
            routes[str(t)] = t_routes
        solution['routes'] = routes
        
        supplier_inventory = {}
        for t in range(1, H + 2):
            inv = I_S0 + t * P - Z[t - 1].X
            supplier_inventory[str(t)] = inv
        solution['supplier_inventory'] = supplier_inventory
        
        retailer_inventory = {}
        for i in range(n):
            r_inv = {}
            for t in range(1, H + 2):
                inv = I_R0[i] - t * d[i] + S[i, t - 1].X
                r_inv[str(t)] = inv
            retailer_inventory[str(retailers[i]['id'])] = r_inv
        solution['retailer_inventory'] = retailer_inventory
        
        if logger:
            logger.log_solution(obj_val, solution)
    else:
        solution = {
            "objective_value": 0.0,
            "deliveries": {},
            "routes": {},
            "supplier_inventory": {str(t): I_S0 + t * P for t in range(1, H + 2)},
            "retailer_inventory": {str(retailers[i]['id']): {str(t): I_R0[i] - t * d[i] for t in range(1, H + 2)} for i in range(n)}
        }
        
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, default=60)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()
    solve(args)
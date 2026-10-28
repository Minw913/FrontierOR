import argparse
import json
import gurobipy as gp
from gurobipy import GRB

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True, help="Path to instance JSON file")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to write the solution JSON file")
    parser.add_argument("--time_limit", type=int, required=True, help="Maximum runtime in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log intermediate solutions")
    args = parser.parse_args()

    # Initialize solution logger
    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize")
    else:
        logger = None

    # Load instance data
    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    # Problem dimensions and structure
    R = data["num_retailers"]
    W = data["num_warehouses"]
    T = data["num_periods"]
    
    r_to_w = data["retailer_warehouse_assignment"]
    w_to_rs = {w: [] for w in range(W)}
    for r, w in enumerate(r_to_w):
        w_to_rs[w].append(r)
        
    demands = data["demands"]
    
    # Costs
    fc_p = data["fixed_costs"]["plant"]
    fc_w = data["fixed_costs"]["warehouses"]
    fc_r = data["fixed_costs"]["retailers"]
    
    hc_p = data["holding_costs"]["plant"]
    hc_w = data["holding_costs"]["warehouses"]
    hc_r = data["holding_costs"]["retailers"]

    # Gurobi Environment and Model Setup
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    
    model = gp.Model("SupplyChain", env=env)
    
    # Apply time limit and settings for single core execution
    if args.time_limit > 1:
        model.setParam("TimeLimit", args.time_limit - 1.0) # 1 second buffer for graceful output
    else:
        model.setParam("TimeLimit", float(args.time_limit))
        
    model.setParam("Threads", 1)
    
    # Define Callback for logging
    def cb(model, where):
        if where == GRB.Callback.MIPSOL:
            if logger:
                obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                logger.log(obj)
                
    # Decision Variables
    # Plant
    x_p = model.addVars(T, vtype=GRB.CONTINUOUS, name="x_p", lb=0)
    s_p = model.addVars(T, vtype=GRB.CONTINUOUS, name="s_p", lb=0)
    y_p = model.addVars(T, vtype=GRB.BINARY, name="y_p")
    
    # Warehouses
    x_w = model.addVars(W, T, vtype=GRB.CONTINUOUS, name="x_w", lb=0)
    s_w = model.addVars(W, T, vtype=GRB.CONTINUOUS, name="s_w", lb=0)
    y_w = model.addVars(W, T, vtype=GRB.BINARY, name="y_w")
    
    # Retailers
    x_r = model.addVars(R, T, vtype=GRB.CONTINUOUS, name="x_r", lb=0)
    s_r = model.addVars(R, T, vtype=GRB.CONTINUOUS, name="s_r", lb=0)
    y_r = model.addVars(R, T, vtype=GRB.BINARY, name="y_r")
    
    # Compile Big-M upper bound dynamically: Sum of remaining demands
    M_r_t = [[0]*T for _ in range(R)]
    for r in range(R):
        for t in range(T-1, -1, -1):
            if t == T-1:
                M_r_t[r][t] = demands[r][t]
            else:
                M_r_t[r][t] = demands[r][t] + M_r_t[r][t+1]
                
    M_w_t = [[0]*T for _ in range(W)]
    for w in range(W):
        for t in range(T):
            M_w_t[w][t] = sum(M_r_t[r][t] for r in w_to_rs[w])
            
    M_p_t = [0]*T
    for t in range(T):
        M_p_t[t] = sum(M_w_t[w][t] for w in range(W))
        
    # Big-M Setup logic constraints and Variable Upper Bounds
    for t in range(T):
        x_p[t].ub = M_p_t[t]
        model.addConstr(x_p[t] <= M_p_t[t] * y_p[t], name=f"BigM_plant_{t}")
        
    for w in range(W):
        for t in range(T):
            x_w[w, t].ub = M_w_t[w][t]
            model.addConstr(x_w[w, t] <= M_w_t[w][t] * y_w[w, t], name=f"BigM_wh_{w}_{t}")
            
    for r in range(R):
        for t in range(T):
            x_r[r, t].ub = M_r_t[r][t]
            model.addConstr(x_r[r, t] <= M_r_t[r][t] * y_r[r, t], name=f"BigM_ret_{r}_{t}")
            
    # Flow and Inventory Balance Constraints
    for t in range(T):
        prev_s_p = s_p[t-1] if t > 0 else 0
        model.addConstr(prev_s_p + x_p[t] == gp.quicksum(x_w[w, t] for w in range(W)) + s_p[t], name=f"InvBal_plant_{t}")
        
    for w in range(W):
        for t in range(T):
            prev_s_w = s_w[w, t-1] if t > 0 else 0
            model.addConstr(prev_s_w + x_w[w, t] == gp.quicksum(x_r[r, t] for r in w_to_rs[w]) + s_w[w, t], name=f"InvBal_wh_{w}_{t}")
            
    for r in range(R):
        for t in range(T):
            prev_s_r = s_r[r, t-1] if t > 0 else 0
            model.addConstr(prev_s_r + x_r[r, t] == demands[r][t] + s_r[r, t], name=f"InvBal_ret_{r}_{t}")
            
    # Objective Formulation: Minimize total fixed and holding costs
    obj = gp.quicksum(fc_p[t] * y_p[t] + hc_p * s_p[t] for t in range(T))
    obj += gp.quicksum(fc_w[w][t] * y_w[w, t] + hc_w * s_w[w, t] for w in range(W) for t in range(T))
    obj += gp.quicksum(fc_r[r][t] * y_r[r, t] + hc_r[r] * s_r[r, t] for r in range(R) for t in range(T))
    
    model.setObjective(obj, GRB.MINIMIZE)

    # Establish an explicit Lot-for-Lot initial MIP start to guarantee at least one valid solution
    obj_lfl = 0
    res_x_p_lfl = [0.0] * T
    res_s_p_lfl = [0.0] * T
    res_y_p_lfl = [0] * T
    
    res_x_w_lfl = [[0.0] * T for _ in range(W)]
    res_s_w_lfl = [[0.0] * T for _ in range(W)]
    res_y_w_lfl = [[0] * T for _ in range(W)]
    
    res_x_r_lfl = [[0.0] * T for _ in range(R)]
    res_s_r_lfl = [[0.0] * T for _ in range(R)]
    res_y_r_lfl = [[0] * T for _ in range(R)]

    for t in range(T):
        p_qty = 0
        for w in range(W):
            w_qty = 0
            for r in w_to_rs[w]:
                rqty = demands[r][t]
                res_x_r_lfl[r][t] = float(rqty)
                res_y_r_lfl[r][t] = 1 if rqty > 0 else 0
                obj_lfl += fc_r[r][t] * res_y_r_lfl[r][t]
                w_qty += rqty
                
                # Provide start hint for solver
                x_r[r, t].start = float(rqty)
                y_r[r, t].start = res_y_r_lfl[r][t]
                s_r[r, t].start = 0.0
                
            res_x_w_lfl[w][t] = float(w_qty)
            res_y_w_lfl[w][t] = 1 if w_qty > 0 else 0
            obj_lfl += fc_w[w][t] * res_y_w_lfl[w][t]
            p_qty += w_qty
            
            x_w[w, t].start = float(w_qty)
            y_w[w, t].start = res_y_w_lfl[w][t]
            s_w[w, t].start = 0.0
            
        res_x_p_lfl[t] = float(p_qty)
        res_y_p_lfl[t] = 1 if p_qty > 0 else 0
        obj_lfl += fc_p[t] * res_y_p_lfl[t]
        
        x_p[t].start = float(p_qty)
        y_p[t].start = res_y_p_lfl[t]
        s_p[t].start = 0.0

    # Execute Model
    model.optimize(cb)
    
    # Process solution matching Solution Output Schema
    if model.SolCount > 0:
        solution = {
            "objective_value": float(model.ObjVal),
            "x_plant": [float(max(0.0, x_p[t].X)) for t in range(T)],
            "s_plant": [float(max(0.0, s_p[t].X)) for t in range(T)],
            "y_plant": [int(round(y_p[t].X)) for t in range(T)],
            "x_warehouses": [[float(max(0.0, x_w[w, t].X)) for t in range(T)] for w in range(W)],
            "s_warehouses": [[float(max(0.0, s_w[w, t].X)) for t in range(T)] for w in range(W)],
            "y_warehouses": [[int(round(y_w[w, t].X)) for t in range(T)] for w in range(W)],
            "x_retailers": [[float(max(0.0, x_r[r, t].X)) for t in range(T)] for r in range(R)],
            "s_retailers": [[float(max(0.0, s_r[r, t].X)) for t in range(T)] for r in range(R)],
            "y_retailers": [[int(round(y_r[r, t].X)) for t in range(T)] for r in range(R)]
        }
    else:
        # Fallback to lot-for-lot if solver aborts entirely without capturing start point
        solution = {
            "objective_value": float(obj_lfl),
            "x_plant": res_x_p_lfl,
            "s_plant": res_s_p_lfl,
            "y_plant": res_y_p_lfl,
            "x_warehouses": res_x_w_lfl,
            "s_warehouses": res_s_w_lfl,
            "y_warehouses": res_y_w_lfl,
            "x_retailers": res_x_r_lfl,
            "s_retailers": res_s_r_lfl,
            "y_retailers": res_y_r_lfl
        }

    # Write definitive outcome to configured destination
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=4)

if __name__ == '__main__':
    main()
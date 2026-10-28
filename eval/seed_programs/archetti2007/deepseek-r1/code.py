import argparse
import json
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

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        data = json.load(f)
    
    n = data['num_retailers']
    H = data['time_horizon']
    supplier = data['supplier']
    retailers = data['retailers']
    vehicle_capacity = data['vehicle_capacity']
    dist_matrix = data['distance_matrix']
    
    supplier_id = supplier['id']
    supplier_starting_inv = supplier['starting_inventory']
    supplier_production = supplier['production_rate']
    supplier_inv_cost = supplier['inventory_cost']
    
    retailer_ids = [r['id'] for r in retailers]
    max_inventory = [r['max_inventory'] for r in retailers]
    starting_inventory = [r['starting_inventory'] for r in retailers]
    demand = [r['demand'] for r in retailers]
    retailer_inv_cost = [r['inventory_cost'] for r in retailers]
    
    external_id = {}
    external_id[0] = supplier_id
    for i in range(1, n+1):
        external_id[i] = retailer_ids[i-1]
    
    model = gp.Model("IRP")
    model.setParam('OutputFlag', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', args.time_limit)
    
    nodes = [0] + list(range(1, n+1))
    
    I_s_var = model.addVars(H+1, lb=0, vtype=GRB.INTEGER, name="I_s")
    I_r_var = model.addVars(n, H+1, lb=0, vtype=GRB.INTEGER, name="I_r")
    x = model.addVars(n, H, vtype=GRB.BINARY, name="x")
    z = model.addVars(n, H, vtype=GRB.INTEGER, lb=0, name="z")
    w = model.addVars(H, vtype=GRB.BINARY, name="w")
    
    a = {}
    for t in range(H):
        for i in nodes:
            for j in nodes:
                if i != j:
                    a[i, j, t] = model.addVar(vtype=GRB.BINARY, name=f"a_{i}_{j}_{t}")
    
    u = {}
    for t in range(H):
        for i in range(1, n+1):
            u[i, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=n, name=f"u_{i}_{t}")
    
    model.addConstr(I_s_var[0] == supplier_starting_inv, "I_s0")
    for i in range(n):
        model.addConstr(I_r_var[i, 0] == starting_inventory[i], f"I_r_{i}_0")
    
    for t in range(H):
        model.addConstrs((w[t] >= x[i, t] for i in range(n)), name=f"w_ge_{t}")
        model.addConstr(w[t] <= gp.quicksum(x[i, t] for i in range(n)), name=f"w_le_{t}")
    
    for t in range(H):
        for i in range(n):
            model.addConstr(
                I_r_var[i, t+1] == I_r_var[i, t] - demand[i] + (max_inventory[i] * x[i, t] - z[i, t]),
                f"ret_inv_{i}_{t}"
            )
            model.addConstr(z[i, t] <= max_inventory[i] * x[i, t], f"z_ub1_{i}_{t}")
            model.addConstr(z[i, t] <= I_r_var[i, t], f"z_ub2_{i}_{t}")
            model.addConstr(z[i, t] >= I_r_var[i, t] - max_inventory[i] * (1 - x[i, t]), f"z_lb_{i}_{t}")
    
    for t in range(H):
        total_delivery = gp.quicksum(max_inventory[i] * x[i, t] - z[i, t] for i in range(n))
        model.addConstr(I_s_var[t] >= total_delivery, f"supplier_inv_min_{t}")
        model.addConstr(I_s_var[t+1] == I_s_var[t] - total_delivery + supplier_production, f"supplier_inv_{t}")
        model.addConstr(total_delivery <= vehicle_capacity, f"capacity_{t}")
    
    for t in range(H):
        model.addConstr(gp.quicksum(a[0, j, t] for j in range(1, n+1)) == w[t], f"supplier_out_{t}")
        model.addConstr(gp.quicksum(a[j, 0, t] for j in range(1, n+1)) == w[t], f"supplier_in_{t}")
        
        for k in range(1, n+1):
            i_ret = k - 1
            model.addConstr(gp.quicksum(a[k, j, t] for j in nodes if j != k) == x[i_ret, t], f"out_{k}_{t}")
            model.addConstr(gp.quicksum(a[j, k, t] for j in nodes if j != k) == x[i_ret, t], f"in_{k}_{t}")
        
        for i in nodes:
            for j in nodes:
                if i == j:
                    continue
                if i == 0 and j != 0:
                    model.addConstr(-u[j, t] + n * a[i, j, t] <= n-1, f"mtz_{i}_{j}_{t}")
                elif j == 0 and i != 0:
                    model.addConstr(u[i, t] + n * a[i, j, t] <= n-1, f"mtz_{i}_{j}_{t}")
                elif i != 0 and j != 0:
                    model.addConstr(u[i, t] - u[j, t] + n * a[i, j, t] <= n-1, f"mtz_{i}_{j}_{t}")
        
        for k in range(1, n+1):
            i_ret = k - 1
            model.addConstr(u[k, t] <= n * x[i_ret, t], f"u_upper_{k}_{t}")
    
    supplier_inv_cost_expr = supplier_inv_cost * gp.quicksum(I_s_var[t] for t in range(H+1))
    retailer_inv_cost_expr = 0
    for i in range(n):
        retailer_inv_cost_expr += retailer_inv_cost[i] * gp.quicksum(I_r_var[i, t] for t in range(H+1))
    trans_cost_expr = 0
    for t in range(H):
        for i in nodes:
            for j in nodes:
                if i != j:
                    trans_cost_expr += a[i, j, t] * dist_matrix[i][j]
    
    total_cost = supplier_inv_cost_expr + retailer_inv_cost_expr + trans_cost_expr
    model.setObjective(total_cost, GRB.MINIMIZE)
    
    def mycallback(model, where):
        if where == GRB.Callback.MIPSOL:
            if logger:
                obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                logger.log(obj_val)
    
    try:
        if logger:
            model.optimize(mycallback)
        else:
            model.optimize()
    except Exception as e:
        print(f"Optimization failed: {str(e)}")
        if model.SolCount > 0:
            model.setParam(GRB.Param.SolutionNumber, 0)
    
    solution = {
        "objective_value": float('inf'),
        "deliveries": {},
        "routes": {},
        "supplier_inventory": {},
        "retailer_inventory": {}
    }
    
    if model.SolCount > 0:
        solution["objective_value"] = model.ObjVal
        
        deliveries_dict = {}
        for t in range(H):
            period_str = str(t)
            d_dict = {}
            for i in range(n):
                if x[i, t].X > 0.5:
                    quantity = max_inventory[i] * x[i, t].X - z[i, t].X
                    d_dict[str(retailer_ids[i])] = float(quantity)
            deliveries_dict[period_str] = d_dict
        solution["deliveries"] = deliveries_dict
        
        routes_dict = {}
        for t in range(H):
            period_str = str(t)
            segment_list = []
            for i in nodes:
                for j in nodes:
                    if i < j:
                        a_ij = a.get((i, j, t), None)
                        a_ji = a.get((j, i, t), None)
                        if a_ij is not None and a_ji is not None:
                            k_val = a_ij.X + a_ji.X
                            if k_val > 0.5:
                                k_val_int = round(k_val)
                                seg = [int(external_id[i]), int(external_id[j]), int(k_val_int)]
                                segment_list.append(seg)
            routes_dict[period_str] = segment_list
        solution["routes"] = routes_dict
        
        supplier_inv_dict = {}
        for t in range(H+1):
            supplier_inv_dict[str(t)] = float(I_s_var[t].X)
        solution["supplier_inventory"] = supplier_inv_dict
        
        retailer_inv_dict = {}
        for i in range(n):
            inv_dict = {}
            for t in range(H+1):
                inv_dict[str(t)] = float(I_r_var[i, t].X)
            retailer_inv_dict[str(retailer_ids[i])] = inv_dict
        solution["retailer_inventory"] = retailer_inv_dict
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()
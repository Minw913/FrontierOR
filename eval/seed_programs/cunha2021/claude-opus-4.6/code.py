import argparse
import json
import time
import gurobipy as gp
from gurobipy import GRB

def solve(instance_path, solution_path, time_limit, log_path):
    from solution_logger import SolutionLogger
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None

    with open(instance_path, 'r') as f:
        data = json.load(f)

    num_retailers = data['num_retailers']
    num_warehouses = data['num_warehouses']
    num_periods = data['num_periods']
    retailer_warehouse_assignment = data['retailer_warehouse_assignment']
    warehouse_retailer_sets = data['warehouse_retailer_sets']
    demands = data['demands']  # [num_retailers][num_periods]
    fixed_costs = data['fixed_costs']
    holding_costs = data['holding_costs']

    T = num_periods
    R = num_retailers
    W = num_warehouses

    # Precompute cumulative demands
    # For retailer r from period t onward
    cum_demand_retailer = [[0]*T for _ in range(R)]
    for r in range(R):
        cum_demand_retailer[r][T-1] = demands[r][T-1]
        for t in range(T-2, -1, -1):
            cum_demand_retailer[r][t] = cum_demand_retailer[r][t+1] + demands[r][t]

    # For warehouse w from period t onward
    cum_demand_warehouse = [[0]*T for _ in range(W)]
    for w in range(W):
        retailers_of_w = warehouse_retailer_sets[str(w)]
        for t in range(T):
            cum_demand_warehouse[w][t] = sum(cum_demand_retailer[r][t] for r in retailers_of_w)

    # For plant from period t onward
    cum_demand_plant = [0]*T
    for t in range(T):
        cum_demand_plant[t] = sum(cum_demand_retailer[r][t] for r in range(R))

    # Build Gurobi model
    start_time = time.time()
    
    model = gp.Model("supply_chain")
    model.Params.TimeLimit = max(1, time_limit - 2)
    model.Params.Threads = 1
    model.Params.MIPFocus = 1  # Focus on finding good feasible solutions
    
    # Decision variables
    # Plant
    x_plant = model.addVars(T, lb=0, name="x_plant")
    s_plant = model.addVars(T, lb=0, name="s_plant")
    y_plant = model.addVars(T, vtype=GRB.BINARY, name="y_plant")

    # Warehouses
    x_wh = model.addVars(W, T, lb=0, name="x_wh")
    s_wh = model.addVars(W, T, lb=0, name="s_wh")
    y_wh = model.addVars(W, T, vtype=GRB.BINARY, name="y_wh")

    # Retailers
    x_rt = model.addVars(R, T, lb=0, name="x_rt")
    s_rt = model.addVars(R, T, lb=0, name="s_rt")
    y_rt = model.addVars(R, T, vtype=GRB.BINARY, name="y_rt")

    # Objective
    obj = gp.LinExpr()
    
    # Plant fixed + holding costs
    plant_fc = fixed_costs['plant']
    plant_hc = holding_costs['plant']
    for t in range(T):
        obj += plant_fc[t] * y_plant[t]
        obj += plant_hc * s_plant[t]

    # Warehouse fixed + holding costs
    wh_fc = fixed_costs['warehouses']
    wh_hc = holding_costs['warehouses']
    for w in range(W):
        for t in range(T):
            obj += wh_fc[w][t] * y_wh[w, t]
            obj += wh_hc * s_wh[w, t]

    # Retailer fixed + holding costs
    rt_fc = fixed_costs['retailers']
    rt_hc = holding_costs['retailers']
    for r in range(R):
        for t in range(T):
            obj += rt_fc[r][t] * y_rt[r, t]
            obj += rt_hc[r] * s_rt[r, t]

    model.setObjective(obj, GRB.MINIMIZE)

    # Constraints
    # Plant flow balance: s_plant[t-1] + x_plant[t] = sum_w(x_wh[w,t]) + s_plant[t]
    for t in range(T):
        prev_s = s_plant[t-1] if t > 0 else 0
        model.addConstr(
            prev_s + x_plant[t] == gp.quicksum(x_wh[w, t] for w in range(W)) + s_plant[t],
            name=f"plant_balance_{t}"
        )

    # Plant setup
    for t in range(T):
        model.addConstr(x_plant[t] <= cum_demand_plant[t] * y_plant[t], name=f"plant_setup_{t}")

    # Warehouse flow balance: s_wh[w,t-1] + x_wh[w,t] = sum_r_in_w(x_rt[r,t]) + s_wh[w,t]
    for w in range(W):
        retailers_of_w = warehouse_retailer_sets[str(w)]
        for t in range(T):
            prev_s = s_wh[w, t-1] if t > 0 else 0
            model.addConstr(
                prev_s + x_wh[w, t] == gp.quicksum(x_rt[r, t] for r in retailers_of_w) + s_wh[w, t],
                name=f"wh_balance_{w}_{t}"
            )

    # Warehouse setup
    for w in range(W):
        for t in range(T):
            model.addConstr(x_wh[w, t] <= cum_demand_warehouse[w][t] * y_wh[w, t], name=f"wh_setup_{w}_{t}")

    # Retailer flow balance: s_rt[r,t-1] + x_rt[r,t] = demands[r][t] + s_rt[r,t]
    for r in range(R):
        for t in range(T):
            prev_s = s_rt[r, t-1] if t > 0 else 0
            model.addConstr(
                prev_s + x_rt[r, t] == demands[r][t] + s_rt[r, t],
                name=f"rt_balance_{r}_{t}"
            )

    # Retailer setup
    for r in range(R):
        for t in range(T):
            model.addConstr(x_rt[r, t] <= cum_demand_retailer[r][t] * y_rt[r, t], name=f"rt_setup_{r}_{t}")

    # Callback for logging
    best_obj = [float('inf')]
    
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj_val < best_obj[0] - 1e-6:
                best_obj[0] = obj_val
                if logger:
                    logger.log(obj_val)

    model.optimize(callback)

    # Extract solution
    if model.SolCount > 0:
        obj_val = model.ObjVal

        sol_x_plant = [x_plant[t].X for t in range(T)]
        sol_s_plant = [s_plant[t].X for t in range(T)]
        sol_y_plant = [int(round(y_plant[t].X)) for t in range(T)]

        sol_x_wh = [[x_wh[w, t].X for t in range(T)] for w in range(W)]
        sol_s_wh = [[s_wh[w, t].X for t in range(T)] for w in range(W)]
        sol_y_wh = [[int(round(y_wh[w, t].X)) for t in range(T)] for w in range(W)]

        sol_x_rt = [[x_rt[r, t].X for t in range(T)] for r in range(R)]
        sol_s_rt = [[s_rt[r, t].X for t in range(T)] for r in range(R)]
        sol_y_rt = [[int(round(y_rt[r, t].X)) for t in range(T)] for r in range(R)]

        solution = {
            "objective_value": obj_val,
            "x_plant": sol_x_plant,
            "s_plant": sol_s_plant,
            "y_plant": sol_y_plant,
            "x_warehouses": sol_x_wh,
            "s_warehouses": sol_s_wh,
            "y_warehouses": sol_y_wh,
            "x_retailers": sol_x_rt,
            "s_retailers": sol_s_rt,
            "y_retailers": sol_y_rt
        }
    else:
        # Return empty/infeasible solution
        solution = {
            "objective_value": float('inf'),
            "x_plant": [0.0]*T,
            "s_plant": [0.0]*T,
            "y_plant": [0]*T,
            "x_warehouses": [[0.0]*T for _ in range(W)],
            "s_warehouses": [[0.0]*T for _ in range(W)],
            "y_warehouses": [[0]*T for _ in range(W)],
            "x_retailers": [[0.0]*T for _ in range(R)],
            "s_retailers": [[0.0]*T for _ in range(R)],
            "y_retailers": [[0]*T for _ in range(R)]
        }

    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)
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

    num_products = data['dimensions']['num_products']
    num_periods = data['dimensions']['num_periods']
    
    demand = data['parameters']['demand']
    holding_cost = data['parameters']['holding_cost']
    setup_cost = data['parameters']['setup_cost']
    var_prod_cost = data['parameters']['variable_production_cost']
    var_prod_time = data['parameters']['variable_production_time']
    setup_time = data['parameters']['setup_time']
    capacity = data['parameters']['capacity']
    init_inv_cost = data['parameters']['initial_inventory_cost']

    # Precompute cumulative remaining demand for each product from period t onwards
    cum_demand = [[0]*num_periods for _ in range(num_products)]
    for i in range(num_products):
        cum_demand[i][num_periods-1] = demand[i][num_periods-1]
        for t in range(num_periods-2, -1, -1):
            cum_demand[i][t] = cum_demand[i][t+1] + demand[i][t]

    start_time = time.time()

    model = gp.Model("CLSP")
    model.setParam('TimeLimit', max(1, time_limit - 2))
    model.setParam('Threads', 1)
    model.setParam('MIPFocus', 1)  # Focus on finding good feasible solutions

    # Decision variables
    production = {}
    setup = {}
    inventory = {}
    init_inv = {}

    for i in range(num_products):
        init_inv[i] = model.addVar(lb=0, obj=init_inv_cost[i], vtype=GRB.CONTINUOUS, name=f"init_inv_{i}")
        for t in range(num_periods):
            # Upper bound on production
            cap_bound = capacity[t] / var_prod_time[i][t] if var_prod_time[i][t] > 0 else cum_demand[i][t]
            ub = min(cum_demand[i][t], cap_bound)
            
            production[i,t] = model.addVar(lb=0, ub=ub, obj=var_prod_cost[i][t], 
                                            vtype=GRB.CONTINUOUS, name=f"prod_{i}_{t}")
            setup[i,t] = model.addVar(vtype=GRB.BINARY, obj=setup_cost[i][t], name=f"setup_{i}_{t}")
            inventory[i,t] = model.addVar(lb=0, obj=holding_cost[i][t], 
                                           vtype=GRB.CONTINUOUS, name=f"inv_{i}_{t}")

    model.update()

    # Flow balance constraints
    for i in range(num_products):
        # First period
        model.addConstr(
            init_inv[i] + production[i,0] == demand[i][0] + inventory[i,0],
            name=f"balance_{i}_0"
        )
        for t in range(1, num_periods):
            model.addConstr(
                inventory[i,t-1] + production[i,t] == demand[i][t] + inventory[i,t],
                name=f"balance_{i}_{t}"
            )

    # Setup linking constraints
    for i in range(num_products):
        for t in range(num_periods):
            # Production upper bound linked to setup
            cap_after_setup = (capacity[t] - setup_time[i][t]) / var_prod_time[i][t] if var_prod_time[i][t] > 0 else cum_demand[i][t]
            M = min(cum_demand[i][t], max(0, cap_after_setup))
            model.addConstr(
                production[i,t] <= M * setup[i,t],
                name=f"setup_link_{i}_{t}"
            )

    # Capacity constraints
    for t in range(num_periods):
        model.addConstr(
            gp.quicksum(setup_time[i][t] * setup[i,t] + var_prod_time[i][t] * production[i,t] 
                        for i in range(num_products)) <= capacity[t],
            name=f"capacity_{t}"
        )

    model.setAttr('ModelSense', GRB.MINIMIZE)

    # Callback for logging
    best_obj = [float('inf')]
    
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj < best_obj[0] - 1e-6:
                best_obj[0] = obj
                if logger:
                    logger.log(obj)

    model.optimize(callback)

    # Extract solution
    if model.SolCount > 0:
        obj_val = model.ObjVal
        
        prod_sol = [[0.0]*num_periods for _ in range(num_products)]
        setup_sol = [[0]*num_periods for _ in range(num_products)]
        inv_sol = [[0.0]*num_periods for _ in range(num_products)]
        init_inv_sol = [0.0]*num_products
        
        for i in range(num_products):
            init_inv_sol[i] = init_inv[i].X
            for t in range(num_periods):
                prod_sol[i][t] = production[i,t].X
                setup_sol[i][t] = int(round(setup[i,t].X))
                inv_sol[i][t] = inventory[i,t].X
        
        solution = {
            "objective_value": obj_val,
            "variables": {
                "production": prod_sol,
                "setup": setup_sol,
                "inventory": inv_sol,
                "initial_inventory": init_inv_sol
            }
        }
    else:
        # Fallback: produce exactly what's needed using initial inventory if necessary
        prod_sol = [[0.0]*num_periods for _ in range(num_products)]
        setup_sol = [[0]*num_periods for _ in range(num_products)]
        inv_sol = [[0.0]*num_periods for _ in range(num_products)]
        init_inv_sol = [0.0]*num_products
        
        for i in range(num_products):
            init_inv_sol[i] = float(cum_demand[i][0])
            inv_val = cum_demand[i][0]
            for t in range(num_periods):
                inv_val -= demand[i][t]
                inv_sol[i][t] = inv_val
        
        obj_val = 0.0
        for i in range(num_products):
            obj_val += init_inv_cost[i] * init_inv_sol[i]
            for t in range(num_periods):
                obj_val += holding_cost[i][t] * inv_sol[i][t]
        
        solution = {
            "objective_value": obj_val,
            "variables": {
                "production": prod_sol,
                "setup": setup_sol,
                "inventory": inv_sol,
                "initial_inventory": init_inv_sol
            }
        }
        if logger:
            logger.log(obj_val)

    with open(solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)
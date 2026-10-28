import argparse
import json
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def solve(instance_path, solution_path, time_limit, log_path=None):
    start_time = time.time()
    
    # Read instance data
    with open(instance_path, 'r') as f:
        data = json.load(f)
    
    NI = data['parameters']['NI']
    NK = data['parameters']['NK']
    NT = data['parameters']['NT']
    
    setup_cost_f = data['costs']['setup_cost_f']
    startup_cost_g = data['costs']['startup_cost_g']
    holding_cost_h = data['costs']['holding_cost_h']
    backlogging_cost_e = data['costs']['backlogging_cost_e']
    
    capacity_C = data['machine_data']['capacity_C']
    startup_time_sigma = data['machine_data']['startup_time_sigma']
    
    demand = data['demand']
    initial_stock_s0 = data['initial_conditions']['initial_stock_s0']
    initial_backlog_r0 = data['initial_conditions']['initial_backlog_r0']
    
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    # Create model
    model = gp.Model("CLSP")
    model.setParam('TimeLimit', max(1, time_limit - 5))
    model.setParam('Threads', 1)
    model.setParam('MIPFocus', 1)  # Focus on finding feasible solutions
    
    # Index ranges
    items = range(NI)
    machines = range(NK)
    periods = range(NT)
    
    # Decision variables
    # Production quantity x[i,k,t]
    x = {}
    for i in items:
        for k in machines:
            for t in periods:
                x[i, k, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"x_{i}_{k}_{t}")
    
    # Inventory s[i,t]
    s = {}
    for i in items:
        for t in periods:
            s[i, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"s_{i}_{t}")
    
    # Backlog r[i,t]
    r = {}
    for i in items:
        for t in periods:
            r[i, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"r_{i}_{t}")
    
    # Setup y[i,k,t] = 1 if machine k is set up for item i in period t
    y = {}
    for i in items:
        for k in machines:
            for t in periods:
                y[i, k, t] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{k}_{t}")
    
    # Startup z[i,k,t] = 1 if startup occurs
    z = {}
    for i in items:
        for k in machines:
            for t in periods:
                z[i, k, t] = model.addVar(vtype=GRB.BINARY, name=f"z_{i}_{k}_{t}")
    
    model.update()
    
    # Objective function
    obj = gp.LinExpr()
    for i in items:
        for k in machines:
            for t in periods:
                obj += setup_cost_f * y[i, k, t]
                obj += startup_cost_g * z[i, k, t]
    for i in items:
        for t in periods:
            obj += holding_cost_h[i] * s[i, t]
            obj += backlogging_cost_e[i] * r[i, t]
    
    model.setObjective(obj, GRB.MINIMIZE)
    
    # Constraints
    
    # 1. Flow balance: s_{i,t-1} + sum_k x_{i,k,t} - r_{i,t-1} = d_{i,t} + s_{i,t} - r_{i,t}
    for i in items:
        for t in periods:
            prev_stock = initial_stock_s0[i] if t == 0 else s[i, t - 1]
            prev_backlog = initial_backlog_r0[i] if t == 0 else r[i, t - 1]
            prod_sum = gp.quicksum(x[i, k, t] for k in machines)
            model.addConstr(
                prev_stock + prod_sum - prev_backlog == demand[i][t] + s[i, t] - r[i, t],
                name=f"balance_{i}_{t}"
            )
    
    # 2. Each machine produces at most one item per period
    for k in machines:
        for t in periods:
            model.addConstr(
                gp.quicksum(y[i, k, t] for i in items) <= 1,
                name=f"one_item_{k}_{t}"
            )
    
    # 3. Production only if set up: x[i,k,t] <= C_k * y[i,k,t]
    for i in items:
        for k in machines:
            for t in periods:
                model.addConstr(
                    x[i, k, t] <= capacity_C[k] * y[i, k, t],
                    name=f"setup_prod_{i}_{k}_{t}"
                )
    
    # 4. Capacity with startup loss: x[i,k,t] + sigma_k * z[i,k,t] <= C_k * y[i,k,t]
    # This is actually: x[i,k,t] <= C_k * y[i,k,t] - sigma_k * z[i,k,t]
    # But we also need that total on machine k in period t respects capacity
    # Since at most one item per machine per period, we can write:
    # sum_i x[i,k,t] + sigma_k * sum_i z[i,k,t] <= C_k
    for k in machines:
        for t in periods:
            model.addConstr(
                gp.quicksum(x[i, k, t] for i in items) + startup_time_sigma[k] * gp.quicksum(z[i, k, t] for i in items) <= capacity_C[k],
                name=f"capacity_{k}_{t}"
            )
    
    # 5. Startup definition: z[i,k,t] >= y[i,k,t] - y[i,k,t-1]
    # For t=0: no machine is set up for any item before period 0, so y[i,k,-1]=0
    # z[i,k,t] >= y[i,k,t] - y[i,k,t-1]
    for i in items:
        for k in machines:
            for t in periods:
                if t == 0:
                    model.addConstr(
                        z[i, k, t] >= y[i, k, t],
                        name=f"startup_def_{i}_{k}_{t}"
                    )
                else:
                    model.addConstr(
                        z[i, k, t] >= y[i, k, t] - y[i, k, t - 1],
                        name=f"startup_def_{i}_{k}_{t}"
                    )
    
    # 6. z[i,k,t] <= y[i,k,t] (can only startup if set up)
    for i in items:
        for k in machines:
            for t in periods:
                model.addConstr(
                    z[i, k, t] <= y[i, k, t],
                    name=f"startup_setup_{i}_{k}_{t}"
                )
    
    # 7. If set up for item i in previous period, no startup for item i:
    # z[i,k,t] <= 1 - y[i,k,t-1] for t >= 1
    for i in items:
        for k in machines:
            for t in periods:
                if t >= 1:
                    model.addConstr(
                        z[i, k, t] <= 1 - y[i, k, t - 1],
                        name=f"no_startup_same_{i}_{k}_{t}"
                    )
    
    # Callback for logging
    best_obj = [float('inf')]
    
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj_val < best_obj[0] - 1e-6:
                best_obj[0] = obj_val
                if logger:
                    logger.log(obj_val)
    
    # Optimize
    model.optimize(callback)
    
    # Extract solution
    if model.SolCount > 0:
        obj_val = model.ObjVal
        
        production = {}
        for i in items:
            for k in machines:
                for t in periods:
                    val = x[i, k, t].X
                    if val > 1e-8:
                        production[f"{i}_{k}_{t}"] = val
        
        stock = {}
        for i in items:
            for t in periods:
                stock[f"{i}_{t}"] = s[i, t].X
        
        backlog = {}
        for i in items:
            for t in periods:
                backlog[f"{i}_{t}"] = r[i, t].X
        
        setup = {}
        for i in items:
            for k in machines:
                for t in periods:
                    val = round(y[i, k, t].X)
                    if val > 0.5:
                        setup[f"{i}_{k}_{t}"] = 1
        
        startup = {}
        for i in items:
            for k in machines:
                for t in periods:
                    val = round(z[i, k, t].X)
                    if val > 0.5:
                        startup[f"{i}_{k}_{t}"] = 1
        
        solution = {
            "objective_value": obj_val,
            "production": production,
            "stock": stock,
            "backlog": backlog,
            "setup": setup,
            "startup": startup
        }
    else:
        # No feasible solution found - return empty with high cost
        solution = {
            "objective_value": float('inf'),
            "production": {},
            "stock": {},
            "backlog": {},
            "setup": {},
            "startup": {}
        }
    
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)
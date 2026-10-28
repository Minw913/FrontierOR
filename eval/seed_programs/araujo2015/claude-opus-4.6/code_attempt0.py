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
    
    num_items = data['num_items']
    num_periods = data['num_periods']
    demand = data['demand']
    setup_cost = data['setup_cost']
    holding_cost = data['holding_cost']
    variable_cost = data['variable_cost']
    setup_time = data['setup_time']
    variable_time = data['variable_time']
    capacity = data['capacity']
    initial_inventory_cost = data['initial_inventory_cost']
    
    # Precompute cumulative demand from period t to end for each item
    cum_demand = [[0] * num_periods for _ in range(num_items)]
    for i in range(num_items):
        cum_demand[i][num_periods - 1] = demand[i][num_periods - 1]
        for t in range(num_periods - 2, -1, -1):
            cum_demand[i][t] = cum_demand[i][t + 1] + demand[i][t]
    
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    # Build MIP model
    model = gp.Model("CLSP")
    model.setParam('TimeLimit', max(1, time_limit - int(time.time() - start_time) - 2))
    model.setParam('Threads', 1)
    model.setParam('MIPFocus', 0)
    
    # Decision variables
    # Production quantity x[i][t]
    x = {}
    for i in range(num_items):
        for t in range(num_periods):
            ub_cap = (capacity[t] - setup_time[i]) / variable_time[i] if variable_time[i] > 0 else float('inf')
            ub_dem = cum_demand[i][t]
            ub = max(0, min(ub_cap, ub_dem))
            x[i, t] = model.addVar(lb=0, ub=ub, vtype=GRB.CONTINUOUS, name=f"x_{i}_{t}")
    
    # Setup indicator y[i][t]
    y = {}
    for i in range(num_items):
        for t in range(num_periods):
            y[i, t] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{t}")
    
    # Inventory s[i][t] where t=0 is initial inventory, t=1..num_periods is end-of-period
    s = {}
    for i in range(num_items):
        # Initial inventory
        s[i, 0] = model.addVar(lb=0, ub=cum_demand[i][0], vtype=GRB.CONTINUOUS, name=f"s_{i}_0")
        for t in range(1, num_periods + 1):
            if t < num_periods:
                ub_inv = cum_demand[i][t]
            else:
                ub_inv = 0  # End inventory must be 0
            s[i, t] = model.addVar(lb=0, ub=ub_inv, vtype=GRB.CONTINUOUS, name=f"s_{i}_{t}")
    
    # Objective
    obj = gp.LinExpr()
    for i in range(num_items):
        obj += initial_inventory_cost[i] * s[i, 0]
        for t in range(num_periods):
            obj += setup_cost[i] * y[i, t]
            obj += variable_cost[i] * x[i, t]
            obj += holding_cost[i] * s[i, t + 1]
    
    model.setObjective(obj, GRB.MINIMIZE)
    
    # Flow balance constraints
    for i in range(num_items):
        for t in range(num_periods):
            model.addConstr(
                s[i, t] + x[i, t] == demand[i][t] + s[i, t + 1],
                name=f"balance_{i}_{t}"
            )
    
    # Terminal inventory = 0
    for i in range(num_items):
        model.addConstr(s[i, num_periods] == 0, name=f"terminal_{i}")
    
    # Setup linking constraints
    for i in range(num_items):
        for t in range(num_periods):
            ub_cap = (capacity[t] - setup_time[i]) / variable_time[i] if variable_time[i] > 0 else float('inf')
            ub_dem = cum_demand[i][t]
            M = max(0, min(ub_cap, ub_dem))
            if M > 0:
                model.addConstr(x[i, t] <= M * y[i, t], name=f"setup_{i}_{t}")
            # If M == 0, x is already bounded to 0
    
    # Capacity constraints
    for t in range(num_periods):
        lhs = gp.LinExpr()
        for i in range(num_items):
            lhs += setup_time[i] * y[i, t]
            lhs += variable_time[i] * x[i, t]
        model.addConstr(lhs <= capacity[t], name=f"capacity_{t}")
    
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
        
        production = [[0.0] * num_periods for _ in range(num_items)]
        setup = [[0] * num_periods for _ in range(num_items)]
        inventory = [[0.0] * (num_periods + 1) for _ in range(num_items)]
        
        for i in range(num_items):
            inventory[i][0] = s[i, 0].X
            for t in range(num_periods):
                production[i][t] = x[i, t].X
                setup[i][t] = int(round(y[i, t].X))
                inventory[i][t + 1] = s[i, t + 1].X
        
        solution = {
            "objective_value": obj_val,
            "production": production,
            "setup": setup,
            "inventory": inventory
        }
    else:
        # No feasible solution found - return empty/default
        solution = {
            "objective_value": float('inf'),
            "production": [[0.0] * num_periods for _ in range(num_items)],
            "setup": [[0] * num_periods for _ in range(num_items)],
            "inventory": [[0.0] * (num_periods + 1) for _ in range(num_items)]
        }
    
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)


if __name__ == '__main__':
    main()
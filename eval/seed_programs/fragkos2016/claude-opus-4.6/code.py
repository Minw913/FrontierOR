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
    
    n_items = data['n_items']
    n_periods = data['n_periods']
    capacity = data['capacity']
    initial_inventory = data['initial_inventory']
    demands = data['demands']
    setup_costs = data['setup_costs']
    holding_costs = data['holding_costs']
    variable_costs = data['variable_costs']
    setup_times = data['setup_times']
    variable_times = data['variable_times']
    big_M = data['big_M']
    
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    # Create model
    model = gp.Model("lot_sizing")
    model.setParam('TimeLimit', max(1, time_limit - 2))
    model.setParam('Threads', 1)
    model.setParam('MIPFocus', 1)  # Focus on finding feasible solutions
    
    # Decision variables
    x = {}  # production quantities
    y = {}  # setup binary variables
    s = {}  # inventory levels
    
    for i in range(n_items):
        # Initial inventory
        s[i, 0] = model.addVar(lb=initial_inventory[i], ub=initial_inventory[i], 
                                name=f's_{i}_0')
        for t in range(n_periods):
            x[i, t] = model.addVar(lb=0, ub=big_M[i][t], name=f'x_{i}_{t}')
            y[i, t] = model.addVar(vtype=GRB.BINARY, name=f'y_{i}_{t}')
            s[i, t + 1] = model.addVar(lb=0, name=f's_{i}_{t+1}')
        # Final inventory must be zero
        s[i, n_periods].ub = 0
    
    model.update()
    
    # Objective: minimize total cost
    obj = gp.LinExpr()
    for i in range(n_items):
        for t in range(n_periods):
            obj += setup_costs[i][t] * y[i, t]
            obj += variable_costs[i][t] * x[i, t]
            obj += holding_costs[i][t] * s[i, t + 1]
    
    model.setObjective(obj, GRB.MINIMIZE)
    
    # Constraints
    # Flow balance: s[i,t] + x[i,t] = d[i,t] + s[i,t+1]
    for i in range(n_items):
        for t in range(n_periods):
            model.addConstr(
                s[i, t] + x[i, t] == demands[i][t] + s[i, t + 1],
                name=f'balance_{i}_{t}'
            )
    
    # Setup constraint: x[i,t] <= big_M[i][t] * y[i,t]
    for i in range(n_items):
        for t in range(n_periods):
            model.addConstr(
                x[i, t] <= big_M[i][t] * y[i, t],
                name=f'setup_{i}_{t}'
            )
    
    # Capacity constraint
    for t in range(n_periods):
        model.addConstr(
            gp.quicksum(setup_times[i][t] * y[i, t] + variable_times[i][t] * x[i, t]
                        for i in range(n_items)) <= capacity[t],
            name=f'capacity_{t}'
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
    
    model.optimize(callback)
    
    # Extract solution
    if model.SolCount > 0:
        objective_value = model.ObjVal
        
        x_sol = [[0.0] * n_periods for _ in range(n_items)]
        y_sol = [[0] * n_periods for _ in range(n_items)]
        s_sol = [[0.0] * (n_periods + 1) for _ in range(n_items)]
        
        for i in range(n_items):
            for t in range(n_periods):
                x_sol[i][t] = x[i, t].X
                y_sol[i][t] = int(round(y[i, t].X))
            for t in range(n_periods + 1):
                s_sol[i][t] = s[i, t].X
        
        solution = {
            "objective_value": objective_value,
            "x": x_sol,
            "s": s_sol,
            "y": y_sol
        }
    else:
        solution = {
            "objective_value": float('inf'),
            "x": [[0.0] * n_periods for _ in range(n_items)],
            "s": [[0.0] * (n_periods + 1) for _ in range(n_items)],
            "y": [[0] * n_periods for _ in range(n_items)]
        }
    
    with open(solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)
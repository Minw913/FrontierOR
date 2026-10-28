import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def solve_manufacturing_planning(instance_path, solution_path, time_limit, log_path):
    # Load the instance data
    with open(instance_path, 'r') as f:
        instance_data = json.load(f)
    
    n_items = instance_data['n_items']
    n_periods = instance_data['n_periods']
    capacities = instance_data['capacity']
    initial_inventory = instance_data['initial_inventory']
    demands = instance_data['demands']
    setup_costs = instance_data['setup_costs']
    holding_costs = instance_data['holding_costs']
    variable_costs = instance_data['variable_costs']
    setup_times = instance_data['setup_times']
    variable_times = instance_data['variable_times']
    big_M = instance_data['big_M']
    
    # Initialize the solution logger
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    # Create model
    model = gp.Model("ManufacturingPlanning")
    model.setParam('TimeLimit', time_limit)
    model.setParam('OutputFlag', 0)  # Suppress Gurobi console output
    
    # Decision variables:
    # x[i][t]: production quantity of item i in period t
    # y[i][t]: binary, 1 if item i is setup in period t, 0 otherwise
    # s[i][t]: inventory of item i at the beginning of period t
    x = [[model.addVar(name=f"x_{i}_{t}", vtype=GRB.CONTINUOUS, lb=0.0) 
          for t in range(n_periods)] for i in range(n_items)]
    
    y = [[model.addVar(name=f"y_{i}_{t}", vtype=GRB.BINARY) 
          for t in range(n_periods)] for i in range(n_items)]
    
    s = [[model.addVar(name=f"s_{i}_{t}", vtype=GRB.CONTINUOUS, lb=0.0) 
          for t in range(n_periods + 1)] for i in range(n_items)]
    
    # Objective function
    obj_expr = gp.LinExpr()
    
    # Setup costs
    for i in range(n_items):
        for t in range(n_periods):
            obj_expr += setup_costs[i][t] * y[i][t]
    
    # Production costs
    for i in range(n_items):
        for t in range(n_periods):
            obj_expr += variable_costs[i][t] * x[i][t]
    
    # Holding costs
    for i in range(n_items):
        for t in range(1, n_periods + 1):  # From period 1 (actually beginning inventory from s[i][1])
            obj_expr += holding_costs[i][t-1] * s[i][t]
    
    model.setObjective(obj_expr, GRB.MINIMIZE)
    
    # Constraints
    
    # Initial inventory constraints
    for i in range(n_items):
        model.addConstr(s[i][0] == initial_inventory[i], name=f"initial_{i}")
    
    # Flow conservation (inventory balance) constraint
    for i in range(n_items):
        for t in range(n_periods):
            model.addConstr(
                s[i][t] + x[i][t] == demands[i][t] + s[i][t+1],
                name=f"inventory_balance_{i}_{t}"
            )
    
    # Production capacity constraints (time limit)
    for t in range(n_periods):
        time_expr = gp.LinExpr()
        
        # Add setup times
        for i in range(n_items):
            time_expr += setup_times[i][t] * y[i][t]
        
        # Add production times
        for i in range(n_items):
            time_expr += variable_times[i][t] * x[i][t]
            
        model.addConstr(time_expr <= capacities[t], name=f"time_capacity_{t}")
    
    # Logical constraint: production is only allowed if setup is performed
    for i in range(n_items):
        for t in range(n_periods):
            model.addConstr(x[i][t] <= big_M[i][t] * y[i][t], name=f"setup_production_{i}_{t}")
    
    # Final inventory must be zero
    for i in range(n_items):
        model.addConstr(s[i][n_periods] == 0, name=f"final_inventory_{i}")
    
    # Callback function to log incumbent solutions
    def solution_callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj_val)
    
    # Optimize the model
    model.optimize(solution_callback)
    
    # Extract solution values
    if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        x_vals = [[x[i][t].X for t in range(n_periods)] for i in range(n_items)]
        y_vals = [[int(y[i][t].X) for t in range(n_periods)] for i in range(n_items)]
        s_vals = [[s[i][t].X for t in range(n_periods + 1)] for i in range(n_items)]
        objective_value = model.objVal
        
        # Create solution dictionary
        solution = {
            "objective_value": objective_value,
            "x": x_vals,
            "s": s_vals,
            "y": y_vals
        }
        
        # Write solution to file
        with open(solution_path, 'w') as f:
            json.dump(solution, f)
    else:
        # Handle case when optimization failed
        solution = {
            "objective_value": float('inf'),
            "x": [[0.0 for _ in range(n_periods)] for _ in range(n_items)],
            "s": [[0.0 for _ in range(n_periods + 1)] for _ in range(n_items)],
            "y": [[0 for _ in range(n_periods)] for _ in range(n_items)]
        }
        
        with open(solution_path, 'w') as f:
            json.dump(solution, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Solve Manufacturing Planning Problem')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the instance file')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to save the solution')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds')
    parser.add_argument('--log_path', type=str, help='Path to the log file (optional)')
    
    args = parser.parse_args()
    
    solve_manufacturing_planning(args.instance_path, args.solution_path, args.time_limit, args.log_path)
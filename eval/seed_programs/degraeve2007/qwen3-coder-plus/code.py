import argparse
import json
import math
from gurobipy import *
import sys
def solve_bom_instance(instance_data, time_limit):
    # Extract dimensions
    num_products = instance_data["dimensions"]["num_products"]
    num_periods = instance_data["dimensions"]["num_periods"]
    
    # Extract parameters
    demand = instance_data["parameters"]["demand"]
    holding_cost = instance_data["parameters"]["holding_cost"]
    setup_cost = instance_data["parameters"]["setup_cost"]
    variable_production_cost = instance_data["parameters"]["variable_production_cost"]
    variable_production_time = instance_data["parameters"]["variable_production_time"]
    setup_time = instance_data["parameters"]["setup_time"]
    capacity = instance_data["parameters"]["capacity"]
    initial_inventory_cost = instance_data["parameters"]["initial_inventory_cost"]

    # Construct model
    model = Model("BOM")
    model.setParam("Seed", 0)
    model.setParam("MIPGap", 1e-4)
    model.setParam("NumericFocus", 0)
    model.setParam("Threads", 1)
    
    # Decision variables
    production = {}
    setup = {}
    inventory = {}
    initial_inventory = {}

    for i in range(num_products):
        initial_inventory[i] = model.addVar(name=f"initial_inventory_{i}", lb=0.0)
        
        for t in range(num_periods):
            production[i,t] = model.addVar(name=f"production_{i}_{t}", lb=0.0)
            setup[i,t] = model.addVar(name=f"setup_{i}_{t}", vtype=GRB.BINARY)
            inventory[i,t] = model.addVar(name=f"inventory_{i}_{t}", lb=0.0)
            
    
    # Constraints
    # 1. Flow balance constraints (for each product i, for each period t)
    for i in range(num_products):
        # Period 0 needs special treatment with initial_inventory
        inv_prev = initial_inventory[i]
        model.addConstr(inv_prev + production[i,0] == demand[i][0] + inventory[i,0],
                        name=f"flow_balance_{i}_0")
        
        for t in range(1, num_periods):
            prev_inventory = inventory[i, t-1]
            model.addConstr(prev_inventory + production[i,t] == demand[i][t] + inventory[i,t],
                            name=f"flow_balance_{i}_{t}")

    # 2. Production enabled only when setup occurs
    for i in range(num_products):
        for t in range(num_periods):
            # We need an upper bound on production to make big-M constraint valid
            # This could be sum of remaining demands: demand[i][t:] 
            remaining_demand = sum(demand[i][k] for k in range(t, num_periods))
            max_possible_with_capacity = (capacity[t] - setup_time[i][t]) / variable_production_time[i][t] if capacity[t] >= setup_time[i][t] else 0.0
            max_allowable_prod = min(remaining_demand, max_possible_with_capacity) if remaining_demand > max_possible_with_capacity else remaining_demand
            model.addConstr(production[i,t] <= max_allowable_prod * setup[i,t], 
                            name=f"prod_enabled_{i}_{t}")

    # 3. Capacity constraints (for each period t)
    for t in range(num_periods):
        capacity_expr = LinExpr()
        for i in range(num_products):
            capacity_expr += setup[i,t] * setup_time[i][t] + production[i,t] * variable_production_time[i][t]
        model.addConstr(capacity_expr <= capacity[t], name=f"cap_const_{t}")
    
    # Objective function
    total_initial_inventory_cost = quicksum(
        initial_inventory_cost[i] * initial_inventory[i] 
        for i in range(num_products)
    )
    
    total_setup_cost = quicksum(
        setup_cost[i][t] * setup[i,t] 
        for i in range(num_products) 
        for t in range(num_periods)
    )
    
    total_production_cost = quicksum(
        variable_production_cost[i][t] * production[i,t] 
        for i in range(num_products) 
        for t in range(num_periods)
    )
    
    total_holding_cost = quicksum(
        holding_cost[i][t] * inventory[i,t] 
        for i in range(num_products) 
        for t in range(num_periods)
    )
    
    model.setObjective(
        total_initial_inventory_cost +
        total_setup_cost +
        total_production_cost +
        total_holding_cost,
        GRB.MINIMIZE
    )

    # Set time limit
    model.setParam('TimeLimit', time_limit)
    model.setParam('OutputFlag', 0)  # Suppress solver output
    
    # Optimize
    model.optimize()

    # Extract solution
    if model.Status == GRB.INFEASIBLE:
        raise ValueError("Model is infeasible!")
    elif model.Status == GRB.UNBOUNDED:
        raise ValueError("Model is unbounded!")
    else:
        # Check if time limit was reached and the solution is available yet
        if model.SolCount == 0:
            raise ValueError("No solution found")
        
        solution_vars = {
            "production": [[0.0 for _ in range(num_periods)] for _ in range(num_products)],
            "setup": [[0 for _ in range(num_periods)] for _ in range(num_products)],
            "inventory": [[0.0 for _ in range(num_periods)] for _ in range(num_products)],
            "initial_inventory": [0.0 for _ in range(num_products)]
        }

        for i in range(num_products):
            solution_vars["initial_inventory"][i] = initial_inventory[i].X

            for t in range(num_periods):
                solution_vars["production"][i][t] = production[i,t].X
                solution_vars["setup"][i][t] = round(setup[i,t].X)
                solution_vars["inventory"][i][t] = inventory[i,t].X
        
        result = {
            "objective_value": model.ObjVal,
            "variables": solution_vars
        }
        
        return result
    

def main():
    parser = argparse.ArgumentParser(description="Solve BOM instance")
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    
    args = parser.parse_args()
    
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)
    
    logger = None
    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize")
    
    # Define callback function to log solutions when found
    global_logger = logger
    def log_callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            global_logger.log(obj_val)
    
    # Try solving the instance
    try:
        result = solve_bom_instance(instance_data, args.time_limit)
        
        # Only save the final solution
        with open(args.solution_path, 'w') as f:
            json.dump(result, f)
        
    except Exception as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
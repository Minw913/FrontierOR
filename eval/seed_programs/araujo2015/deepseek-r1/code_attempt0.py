import argparse
import json
import gurobipy as gp
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    # Read instance data
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    num_items = instance['num_items']
    num_periods = instance['num_periods']
    demand = instance['demand']
    setup_cost = instance['setup_cost']
    holding_cost = instance['holding_cost']
    variable_cost = instance['variable_cost']
    setup_time = instance['setup_time']
    variable_time = instance['variable_time']
    capacity = instance['capacity']
    initial_inventory_cost = instance['initial_inventory_cost']

    # Precompute total demand per item
    total_demand = [sum(dem) for dem in demand]
    
    # Precompute total remaining demand per item and period
    total_remaining_demand = []
    for i in range(num_items):
        rem_dem = []
        cum_demand = 0
        for t in range(num_periods):
            cum_demand = sum(demand[i][t:])
            rem_dem.append(cum_demand)
        total_remaining_demand.append(rem_dem)
    
    # Precompute big-M values for setup constraints
    M = []
    for i in range(num_items):
        m_i = []
        for t in range(num_periods):
            if variable_time[i] == 0:
                cap_limit = float('inf')
            else:
                cap_limit = (capacity[t] - setup_time[i]) / variable_time[i]
            m_val = min(cap_limit, total_remaining_demand[i][t])
            if m_val < 0:
                m_val = 0
            m_i.append(m_val)
        M.append(m_i)
    
    # Initialize solution logger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Create optimization model
    model = gp.Model("CapacitatedLotSizing")
    model.setParam('OutputFlag', 0)  # Disable solver output
    model.setParam('TimeLimit', args.time_limit)
    
    # Create decision variables
    s = model.addVars(num_items, num_periods+1, lb=0, name="inventory")
    x = model.addVars(num_items, num_periods, lb=0, name="production")
    y = model.addVars(num_items, num_periods, vtype=gp.GRB.BINARY, name="setup")
    
    # Set initial solution: zero production with initial inventory covering all demand
    for i in range(num_items):
        s[i,0].Start = total_demand[i]
        cum = 0
        for t in range(num_periods):
            x[i,t].Start = 0
            y[i,t].Start = 0
            cum += demand[i][t]
            s[i,t+1].Start = total_demand[i] - cum
    
    # Flow balance constraints
    for i in range(num_items):
        for t in range(num_periods):
            model.addConstr(s[i,t] + x[i,t] == demand[i][t] + s[i,t+1])
    
    # Final inventory must be zero
    for i in range(num_items):
        model.addConstr(s[i,num_periods] == 0)
    
    # Setup constraints
    for i in range(num_items):
        for t in range(num_periods):
            model.addConstr(x[i,t] <= M[i][t] * y[i,t])
    
    # Capacity constraints
    for t in range(num_periods):
        model.addConstr(
            gp.quicksum(setup_time[i]*y[i,t] + variable_time[i]*x[i,t] for i in range(num_items)) <= capacity[t]
        )
    
    # Objective function
    initial_inv_cost = gp.quicksum(initial_inventory_cost[i] * s[i,0] for i in range(num_items))
    setup_costs = gp.quicksum(setup_cost[i] * y[i,t] for i in range(num_items) for t in range(num_periods))
    production_costs = gp.quicksum(variable_cost[i] * x[i,t] for i in range(num_items) for t in range(num_periods))
    holding_costs = gp.quicksum(holding_cost[i] * s[i,t] for i in range(num_items) for t in range(1, num_periods+1))
    model.setObjective(initial_inv_cost + setup_costs + production_costs + holding_costs)
    
    # Callback for logging intermediate solutions
    def callback(model, where):
        if where == gp.GRB.Callback.MIPSOL:
            if logger:
                obj_val = model.cbGet(gp.GRB.Callback.MIPSOL_OBJ)
                logger.log(obj_val)
    
    # Optimize
    model.optimize(callback)
    
    # Extract solution
    production = []
    setup = []
    inventory = []
    
    if model.SolCount > 0:
        # Use best solution found by solver
        for i in range(num_items):
            production.append([x[i,t].X for t in range(num_periods)])
            setup.append([int(round(y[i,t].X)) for t in range(num_periods)])
            inventory.append([s[i,t].X for t in range(num_periods+1)])
        objective_value = model.ObjVal
    else:
        # Fallback to initial solution
        for i in range(num_items):
            production.append([x[i,t].Start for t in range(num_periods)])
            setup.append([int(round(y[i,t].Start)) for t in range(num_periods)])
            inv_i = [s[i,0].Start]
            cum = 0
            for t in range(num_periods):
                cum += demand[i][t]
                inv_i.append(total_demand[i] - cum)
            inventory.append(inv_i)
        # Compute objective for initial solution
        objective_value = 0
        for i in range(num_items):
            objective_value += initial_inventory_cost[i] * inventory[i][0]
            for t in range(num_periods):
                objective_value += setup_cost[i] * setup[i][t]
                objective_value += variable_cost[i] * production[i][t]
            for t in range(1, num_periods+1):
                objective_value += holding_cost[i] * inventory[i][t]
    
    # Write solution to file
    solution = {
        "objective_value": objective_value,
        "production": production,
        "setup": setup,
        "inventory": inventory
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == '__main__':
    main()
import argparse
import json
import numpy as np
from gurobipy import *
import math

def solve_distribution_network(instance_path, solution_path, time_limit, log_path=None):
    # Read the instance data
    with open(instance_path, 'r') as f:
        instance_data = json.load(f)
    
    num_facilities = instance_data['parameters']['num_facilities']
    num_customers = instance_data['parameters']['num_customers']
    num_periods = instance_data['parameters']['num_periods']
    
    seasonal_factors = np.array(instance_data['seasonal_factors'])
    demands = np.array(instance_data['demains'])
    holding_costs = np.array(instance_data['holding_costs'])
    capacities = np.array(instance_data['capacities'])
    transportation_costs = np.array(instance_data['transportation_costs'])
    
    # Create Gurobi model
    model = Model('Logistics Distribution Network')
    model.setParam('TimeLimit', time_limit)
    model.setParam('MIPFocus', 0)  # General focus
    model.setParam('Threads', 1)
    
    # Variables
    # Customer assignment variables (binary): x_cf 
    x = {}  # x[customer, facility]
    for c in range(num_customers):
        for f in range(num_facilities):
            x[c, f] = model.addVar(vtype=GRB.BINARY, name=f'x_{c}_{f}')
    
    # Inventory variables: I_[f, t]
    I = {}  # I[facility, period]
    for f in range(num_facilities):
        for t in range(num_periods):
            I[f, t] = model.addVar(vtype=GRB.CONTINUOUS, name=f'I_{f}_{t}', lb=0.0)
    
    # Supply quantity variables (derived from assignments)
    supply = {}  # supply[facility, customer, period]
    for f in range(num_facilities):
        for c in range(num_customers):
            for t in range(num_periods):
                # Calculate period-specific demand for customer c
                period_demand = demands[c] * seasonal_factors[t]
                # This variable represents the supply cost impact if assigned
                supply[f, c, t] = x[c, f] * period_demand
    
    model.update()
    
    # Constraints
    # 1. Each customer is assigned to exactly one facility
    for c in range(num_customers):
        model.addConstr(quicksum(x[c, f] for f in range(num_facilities)) == 1, name=f'assign_cust_{c}')
    
    # 2. Capacity constraints: Production + inventory >= Demand + inventory_next
    for f in range(num_facilities):
        for t in range(num_periods):
            # Sum of demands assigned to facility f in period t (i.e., incoming flow)
            total_demand_f_t = quicksum(demands[c] * seasonal_factors[t] * x[c, f] 
                                       for c in range(num_customers))
            
            if t == 0:
                # Initial inventory is zero
                inventory_last_period = 0
            else:
                inventory_last_period = I[f, t-1]
            
            # Constraint: production_cap + inventory_from_prev >= demand + inventory_now
            # Which we rearrange to: inventory_last_period + production_capacity >= total_demand + inventory_now
            model.addConstr(inventory_last_period + capacities[f, t] >= total_demand_f_t + I[f, t],
                           name=f'capacity_{f}_{t}')
    
    # Objective function components
    # 1. Supply costs
    supply_cost = quicksum(
        transportation_costs[f, c, t] * demands[c] * seasonal_factors[t] * x[c, f]
        for f in range(num_facilities)
        for c in range(num_customers)
        for t in range(num_periods)
    )
    
    # 2. Holding costs
    holding_cost = quicksum(
        holding_costs[f, t] * I[f, t]
        for f in range(num_facilities)
        for t in range(num_periods)
    )
    
    # Set objective
    model.setObjective(supply_cost + holding_cost, GRB.MINIMIZE)
    
    # Initialize logger if provided
    from solution_logger import SolutionLogger
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    # Callback to log intermediate solutions
    def log_solution_cb(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_value = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj_value)
    
    # Optimize
    model.optimize(log_solution_cb)
    
    # Extract solution
    assignment = {}
    for c in range(num_customers):
        for f in range(num_facilities):
            if x[c, f].x > 0.5:  # Assignment is made
                assignment[str(c)] = f
                break
                
    inventory = {}
    for f in range(num_facilities):
        inventory[str(f)] = []
        for t in range(num_periods):
            inventory[str(f)].append(max(0, I[f, t].x))  # Ensure it's non-negative
    		
    # Prepare solution object
    solution = {
        'objective_value': model.objVal if model.solCount > 0 else float('inf'),
        'assignment': assignment,
        'inventory': inventory
    }
    
    # Write solution to output file
    with open(solution_path, 'w') as f:
        json.dump(solution, f)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Solve Logistics Distribution Network Problem')
    parser.add_argument('--instance_path', required=True, help='Path to the instance JSON file')
    parser.add_argument('--solution_path', required=True, help='Path to write the solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Maximum runtime in seconds')
    parser.add_argument('--log_path', help='Path to log intermediate solutions (optional)')
    
    args = parser.parse_args()
    
    solve_distribution_network(args.instance_path, args.solution_path, args.time_limit, args.log_path)
import argparse
import json
import time
import numpy as np
from solution_logger import SolutionLogger

def solve(instance_path, solution_path, time_limit, log_path):
    with open(instance_path, 'r') as f:
        data = json.load(f)
    
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    start_time = time.time()
    
    num_customers = data['num_customers']
    num_plants = data['num_plants']
    demands = data['customers']['demands']
    capacities = data['plants']['capacities']
    fixed_costs = data['plants']['fixed_costs']
    shipping_costs = data['shipping_costs']  # [customer][plant]
    
    # Convert to numpy for efficiency
    demands_np = np.array(demands, dtype=np.float64)
    capacities_np = np.array(capacities, dtype=np.float64)
    fixed_costs_np = np.array(fixed_costs, dtype=np.float64)
    shipping_np = np.array(shipping_costs, dtype=np.float64)  # shape: (num_customers, num_plants)
    
    # Use Gurobi to solve
    import gurobipy as gp
    from gurobipy import GRB
    
    model = gp.Model("facility_location")
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', max(1, time_limit - (time.time() - start_time) - 1))
    model.setParam('Threads', 1)
    
    # Decision variables
    # y[j] = 1 if plant j is open
    y = model.addVars(num_plants, vtype=GRB.BINARY, name="y")
    
    # x[i,j] = fraction of customer i's demand served by plant j
    x = model.addVars(num_customers, num_plants, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="x")
    
    # Objective: minimize total fixed + shipping cost
    obj = gp.LinExpr()
    for j in range(num_plants):
        obj += fixed_costs_np[j] * y[j]
    for i in range(num_customers):
        for j in range(num_plants):
            obj += shipping_np[i, j] * x[i, j]
    
    model.setObjective(obj, GRB.MINIMIZE)
    
    # Constraints: each customer's demand must be fully met
    for i in range(num_customers):
        model.addConstr(gp.quicksum(x[i, j] for j in range(num_plants)) == 1.0, name=f"demand_{i}")
    
    # Constraints: capacity of each plant
    for j in range(num_plants):
        model.addConstr(
            gp.quicksum(demands_np[i] * x[i, j] for i in range(num_customers)) <= capacities_np[j] * y[j],
            name=f"capacity_{j}"
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
        
        open_plants = []
        for j in range(num_plants):
            if y[j].X > 0.5:
                open_plants.append(j)
        
        assignments = {}
        for i in range(num_customers):
            customer_assignments = {}
            for j in range(num_plants):
                val = x[i, j].X
                if val > 1e-8:
                    customer_assignments[str(j)] = val
            assignments[str(i)] = customer_assignments
        
        solution = {
            "objective_value": objective_value,
            "open_plants": open_plants,
            "assignments": assignments
        }
    else:
        # Fallback: open all plants and assign greedily
        open_plants = list(range(num_plants))
        assignments = {}
        remaining_cap = capacities_np.copy()
        total_cost = sum(fixed_costs)
        
        for i in range(num_customers):
            # Find cheapest plant with capacity
            j_best = np.argmin(shipping_np[i])
            assignments[str(i)] = {str(int(j_best)): 1.0}
            remaining_cap[j_best] -= demands_np[i]
            total_cost += shipping_np[i, j_best]
        
        if logger:
            logger.log(total_cost)
        
        solution = {
            "objective_value": total_cost,
            "open_plants": open_plants,
            "assignments": assignments
        }
    
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
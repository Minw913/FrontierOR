import argparse
import json
import time
import numpy as np
from gurobipy import Model, GRB, quicksum

def solve(instance_path, solution_path, time_limit, log_path):
    from solution_logger import SolutionLogger
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    with open(instance_path, 'r') as f:
        data = json.load(f)
    
    num_facilities = data['num_facilities']
    num_customers = data['num_customers']
    num_scenarios = data['num_scenarios']
    
    facilities = data['facilities']
    customers = data['customers']
    transport_costs = data['transportation_costs']
    scenarios = data['scenarios']
    
    opening_costs = [f['opening_cost'] for f in facilities]
    capacities = [f['capacity'] for f in facilities]
    
    probabilities = [s['probability'] for s in scenarios]
    demands = [s['demands'] for s in scenarios]  # scenarios x customers
    
    start_time = time.time()
    
    # Create Gurobi model
    model = Model("facility_location")
    model.setParam('OutputFlag', 1)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', max(time_limit - 5, 10))
    
    # Decision variables
    # x[i] = 1 if facility i is opened
    x = model.addVars(num_facilities, vtype=GRB.BINARY, name="x")
    
    # y[s][i][j] = amount shipped from facility i to customer j in scenario s
    y = {}
    for s in range(num_scenarios):
        for i in range(num_facilities):
            for j in range(num_customers):
                y[s, i, j] = model.addVar(lb=0.0, name=f"y_{s}_{i}_{j}")
    
    model.update()
    
    # Objective: minimize opening costs + expected transportation costs
    obj = quicksum(opening_costs[i] * x[i] for i in range(num_facilities))
    for s in range(num_scenarios):
        prob = probabilities[s]
        obj += prob * quicksum(transport_costs[i][j] * y[s, i, j] 
                               for i in range(num_facilities) 
                               for j in range(num_customers))
    model.setObjective(obj, GRB.MINIMIZE)
    
    # Demand satisfaction constraints
    for s in range(num_scenarios):
        for j in range(num_customers):
            model.addConstr(
                quicksum(y[s, i, j] for i in range(num_facilities)) >= demands[s][j],
                name=f"demand_{s}_{j}"
            )
    
    # Capacity constraints
    for s in range(num_scenarios):
        for i in range(num_facilities):
            model.addConstr(
                quicksum(y[s, i, j] for j in range(num_customers)) <= capacities[i] * x[i],
                name=f"cap_{s}_{i}"
            )
    
    # Total capacity constraint
    max_total_demand = max(sum(demands[s][j] for j in range(num_customers)) for s in range(num_scenarios))
    model.addConstr(
        quicksum(capacities[i] * x[i] for i in range(num_facilities)) >= max_total_demand,
        name="total_cap"
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
        obj_val = model.ObjVal
        
        open_facilities = []
        x_dict = {}
        for i in range(num_facilities):
            val = x[i].X
            x_dict[str(i)] = int(round(val))
            if round(val) > 0.5:
                open_facilities.append(i)
        
        y_dict = {}
        for s in range(num_scenarios):
            y_dict[str(s)] = {}
            for i in range(num_facilities):
                y_dict[str(s)][str(i)] = {}
                for j in range(num_customers):
                    val = y[s, i, j].X
                    y_dict[str(s)][str(i)][str(j)] = val
        
        solution = {
            "objective_value": obj_val,
            "open_facilities": open_facilities,
            "x": x_dict,
            "y": y_dict
        }
    else:
        solution = {
            "objective_value": float('inf'),
            "open_facilities": [],
            "x": {str(i): 0 for i in range(num_facilities)},
            "y": {str(s): {str(i): {str(j): 0.0 for j in range(num_customers)} for i in range(num_facilities)} for s in range(num_scenarios)}
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
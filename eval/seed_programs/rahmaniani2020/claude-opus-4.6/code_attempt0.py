import argparse
import json
import time
import numpy as np
from itertools import combinations

def solve(instance_path, solution_path, time_limit, log_path):
    from solution_logger import SolutionLogger
    
    start_time = time.time()
    
    # Read instance
    with open(instance_path, 'r') as f:
        data = json.load(f)
    
    num_facilities = data['num_facilities']
    num_customers = data['num_customers']
    num_scenarios = data['scenarios']['num_scenarios']
    
    fixed_costs = data['facilities']['fixed_costs']
    capacities = data['facilities']['capacities']
    routing_costs = data['routing_costs']
    probabilities = data['scenarios']['probabilities']
    demands = data['scenarios']['demands']
    max_total_demand = data['max_total_demand']
    
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    # Convert to numpy for efficiency
    rc = np.array(routing_costs)  # [num_facilities][num_customers]
    probs = np.array(probabilities)  # [num_scenarios]
    dems = np.array(demands)  # [num_scenarios][num_customers]
    caps = np.array(capacities)
    fc = np.array(fixed_costs)
    
    # For small instances, use Gurobi MIP directly
    # For larger instances, we might need heuristics
    
    use_gurobi = True
    
    if use_gurobi:
        import gurobipy as gp
        from gurobipy import GRB
        
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time <= 1:
            # Fallback: open all facilities
            y = [1] * num_facilities
            obj = sum(fixed_costs)
            solution = {"objective_value": obj, "y": y}
            with open(solution_path, 'w') as f:
                json.dump(solution, f)
            return
        
        model = gp.Model("stochastic_facility_location")
        model.setParam('TimeLimit', remaining_time - 1)
        model.setParam('Threads', 1)
        model.setParam('MIPFocus', 1)  # Focus on finding feasible solutions
        
        # Decision variables
        y = model.addVars(num_facilities, vtype=GRB.BINARY, name="y")
        
        # Flow variables x[i,j,s] - flow from facility i to customer j in scenario s
        # For efficiency with many scenarios, we use continuous variables
        x = {}
        for s in range(num_scenarios):
            for i in range(num_facilities):
                for j in range(num_customers):
                    x[i, j, s] = model.addVar(lb=0.0, name=f"x_{i}_{j}_{s}")
        
        model.update()
        
        # Objective: minimize fixed costs + expected routing costs
        obj = gp.quicksum(fc[i] * y[i] for i in range(num_facilities))
        for s in range(num_scenarios):
            p = probs[s]
            for i in range(num_facilities):
                for j in range(num_customers):
                    obj += p * rc[i][j] * x[i, j, s]
        
        model.setObjective(obj, GRB.MINIMIZE)
        
        # Constraints
        # 1. Demand satisfaction: for each scenario s and customer j
        for s in range(num_scenarios):
            for j in range(num_customers):
                model.addConstr(
                    gp.quicksum(x[i, j, s] for i in range(num_facilities)) >= dems[s][j],
                    name=f"demand_{s}_{j}"
                )
        
        # 2. Capacity constraints: for each scenario s and facility i
        for s in range(num_scenarios):
            for i in range(num_facilities):
                model.addConstr(
                    gp.quicksum(x[i, j, s] for j in range(num_customers)) <= caps[i] * y[i],
                    name=f"cap_{s}_{i}"
                )
        
        # 3. Total capacity constraint
        model.addConstr(
            gp.quicksum(caps[i] * y[i] for i in range(num_facilities)) >= max_total_demand,
            name="total_cap"
        )
        
        # Store best solution found
        best_obj = [float('inf')]
        best_y = [None]
        
        def callback(model, where):
            if where == GRB.Callback.MIPSOL:
                obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if obj_val < best_obj[0]:
                    best_obj[0] = obj_val
                    best_y[0] = [int(round(model.cbGetSolution(y[i]))) for i in range(num_facilities)]
                    if logger:
                        logger.log(obj_val)
        
        model.optimize(callback)
        
        if model.SolCount > 0:
            obj_val = model.ObjVal
            y_sol = [int(round(y[i].X)) for i in range(num_facilities)]
            if logger and obj_val < best_obj[0]:
                logger.log(obj_val)
            solution = {"objective_value": obj_val, "y": y_sol}
        elif best_y[0] is not None:
            solution = {"objective_value": best_obj[0], "y": best_y[0]}
        else:
            # Fallback: open all facilities
            y_sol = [1] * num_facilities
            obj_val = sum(fixed_costs)
            # Compute routing cost with simple greedy assignment
            for s in range(num_scenarios):
                for j in range(num_customers):
                    min_cost = min(rc[i][j] for i in range(num_facilities))
                    obj_val += probs[s] * min_cost * dems[s][j]
            solution = {"objective_value": obj_val, "y": y_sol}
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
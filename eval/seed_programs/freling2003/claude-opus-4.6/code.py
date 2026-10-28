import argparse
import json
import time
import numpy as np
from itertools import product as iterproduct

def solve(instance_path, solution_path, time_limit, log_path):
    from solution_logger import SolutionLogger
    
    start_time = time.time()
    
    with open(instance_path, 'r') as f:
        data = json.load(f)
    
    num_facilities = data['parameters']['num_facilities']
    num_customers = data['parameters']['num_customers']
    num_periods = data['parameters']['num_periods']
    
    seasonal_factors = np.array(data['seasonal_factors'])
    demands = np.array(data['demands'])
    holding_costs = np.array(data['holding_costs'])  # [num_facilities][num_periods]
    capacities = np.array(data['capacities'])  # [num_facilities][num_periods]
    transportation_costs = np.array(data['transportation_costs'])  # [num_facilities][num_customers][num_periods]
    
    # Customer demand in each period: d[j][t] = demands[j] * seasonal_factors[t]
    # demand_jt[j][t]
    demand_jt = np.outer(demands, seasonal_factors)  # [num_customers, num_periods]
    
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    # For each facility-customer pair, compute the total transportation cost over all periods
    # trans_cost_total[i][j] = sum_t transportation_costs[i][j][t]
    # But actually, the supply cost for assigning customer j to facility i is:
    # sum_t transportation_costs[i][j][t] (since the customer is assigned for all periods)
    trans_cost_total = transportation_costs.sum(axis=2)  # [num_facilities, num_customers]
    
    # Use Gurobi MIP formulation
    import gurobipy as gp
    from gurobipy import GRB
    
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time <= 0:
        remaining_time = 1
    
    model = gp.Model("logistics")
    model.setParam('TimeLimit', max(1, remaining_time - 5))
    model.setParam('Threads', 1)
    model.setParam('MIPFocus', 1)  # Focus on finding good feasible solutions
    
    # Decision variables
    # x[i,j] = 1 if customer j is assigned to facility i
    x = {}
    for i in range(num_facilities):
        for j in range(num_customers):
            x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
    
    # inv[i,t] = inventory at facility i at end of period t
    inv = {}
    for i in range(num_facilities):
        for t in range(num_periods):
            inv[i, t] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"inv_{i}_{t}")
    
    model.update()
    
    # Objective: minimize total supply cost + total holding cost
    obj = gp.LinExpr()
    
    # Supply cost: sum over i,j,t of transportation_costs[i][j][t] * x[i,j]
    # = sum over i,j of trans_cost_total[i][j] * x[i,j]
    for i in range(num_facilities):
        for j in range(num_customers):
            obj += trans_cost_total[i, j] * x[i, j]
    
    # Holding cost
    for i in range(num_facilities):
        for t in range(num_periods):
            obj += holding_costs[i][t] * inv[i, t]
    
    model.setObjective(obj, GRB.MINIMIZE)
    
    # Constraints
    # Each customer assigned to exactly one facility
    for j in range(num_customers):
        model.addConstr(
            gp.quicksum(x[i, j] for i in range(num_facilities)) == 1,
            name=f"assign_{j}"
        )
    
    # Capacity/inventory balance constraints for each facility and period
    # In period t: demand on facility i = sum_j demand_jt[j][t] * x[i,j]
    # Balance: production + inv[i,t-1] >= demand_on_i_t + inv[i,t]
    # => demand_on_i_t + inv[i,t] <= capacity[i][t] + inv[i,t-1]
    # With inv[i,-1] = 0
    for i in range(num_facilities):
        for t in range(num_periods):
            demand_expr = gp.quicksum(demand_jt[j, t] * x[i, j] for j in range(num_customers))
            prev_inv = inv[i, t - 1] if t > 0 else 0
            model.addConstr(
                demand_expr + inv[i, t] <= capacities[i][t] + prev_inv,
                name=f"cap_{i}_{t}"
            )
    
    # Callback to log solutions
    best_obj = [float('inf')]
    best_assignment = [None]
    best_inventory = [None]
    
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj_val < best_obj[0] - 1e-6:
                best_obj[0] = obj_val
                if logger:
                    logger.log(obj_val)
                # Extract solution
                assignment = {}
                for j in range(num_customers):
                    for i in range(num_facilities):
                        val = model.cbGetSolution(x[i, j])
                        if val > 0.5:
                            assignment[str(j)] = i
                            break
                best_assignment[0] = assignment
                
                inventory = {}
                for i in range(num_facilities):
                    inv_list = []
                    for t in range(num_periods):
                        inv_list.append(model.cbGetSolution(inv[i, t]))
                    inventory[str(i)] = inv_list
                best_inventory[0] = inventory
    
    model.optimize(callback)
    
    # Extract final solution
    if model.SolCount > 0:
        obj_val = model.ObjVal
        
        assignment = {}
        for j in range(num_customers):
            for i in range(num_facilities):
                if x[i, j].X > 0.5:
                    assignment[str(j)] = i
                    break
        
        inventory = {}
        for i in range(num_facilities):
            inv_list = []
            for t in range(num_periods):
                inv_list.append(max(0, inv[i, t].X))
            inventory[str(i)] = inv_list
        
        solution = {
            "objective_value": obj_val,
            "assignment": assignment,
            "inventory": inventory
        }
    elif best_assignment[0] is not None:
        solution = {
            "objective_value": best_obj[0],
            "assignment": best_assignment[0],
            "inventory": best_inventory[0]
        }
    else:
        # Fallback: assign each customer to cheapest facility ignoring capacity
        assignment = {}
        for j in range(num_customers):
            best_i = int(np.argmin(trans_cost_total[:, j]))
            assignment[str(j)] = best_i
        
        # Compute inventory
        facility_demand = np.zeros((num_facilities, num_periods))
        for j in range(num_customers):
            i = assignment[str(j)]
            for t in range(num_periods):
                facility_demand[i, t] += demand_jt[j, t]
        
        inventory = {}
        obj_val = 0.0
        for i in range(num_facilities):
            inv_list = []
            prev_inv = 0.0
            for t in range(num_periods):
                current_inv = max(0, prev_inv + capacities[i, t] - facility_demand[i, t])
                # Actually: inv[i,t] = prev_inv + cap[i,t] - demand => but we need feasibility
                # Just set inv to max(0, ...)
                current_inv = prev_inv + capacities[i, t] - facility_demand[i, t]
                if current_inv < 0:
                    current_inv = 0  # infeasible but best we can do
                inv_list.append(current_inv)
                obj_val += holding_costs[i][t] * current_inv
                prev_inv = current_inv
            inventory[str(i)] = inv_list
        
        for j in range(num_customers):
            i = assignment[str(j)]
            obj_val += trans_cost_total[i, j]
        
        if logger:
            logger.log(obj_val)
        
        solution = {
            "objective_value": obj_val,
            "assignment": assignment,
            "inventory": inventory
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
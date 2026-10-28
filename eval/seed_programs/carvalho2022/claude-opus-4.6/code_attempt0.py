import argparse
import json
import time
import math
from itertools import product

def solve(instance_path, solution_path, time_limit, log_path):
    from solution_logger import SolutionLogger
    
    with open(instance_path, 'r') as f:
        data = json.load(f)
    
    start_time = time.time()
    
    # Parse dimensions
    n = data['dimensions']['n']
    m = data['dimensions']['m']
    p = data['dimensions']['p']
    
    # Parse parameters
    demands = data['demands']  # [n][p]
    inventory_costs = data['inventory_costs']  # [n]
    processing_time = data['processing_time']  # scalar
    setup_times = data['setup_times']  # [n][n][m]
    setup_costs = data['setup_costs']  # [n][n][m]
    machine_capacities = data['machine_capacities']  # [m][p]
    max_production = data['max_production']  # [n][p]
    max_setups = data['max_setups_per_item']  # [n][m][p]
    minimum_lot_sizes = data['minimum_lot_sizes']  # [n]
    machine_eligibility = data['machine_eligibility']  # [n][m]
    
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    import gurobipy as gp
    from gurobipy import GRB
    
    model = gp.Model("CLSP")
    model.setParam('TimeLimit', max(1, time_limit - 5))
    model.setParam('Threads', 1)
    model.setParam('MIPGap', 0.005)
    # Use moderate settings for tractability
    model.setParam('MIPFocus', 1)  # Focus on finding feasible solutions
    
    # Decision variables
    # x[i,k,t,s] = quantity of item i produced on machine k in period t for demand period s
    # y[i,j,k,t] = number of changeovers from item i to item j on machine k in period t
    # z[i,k,t] = 1 if machine k is configured for item i at start of period t
    
    # Precompute eligible (item, machine) pairs
    eligible = {}
    for i in range(n):
        for k in range(m):
            if machine_eligibility[i][k] == 1:
                eligible[(i, k)] = True
    
    # Precompute which (i,k,t) have max_setups > 0
    can_produce = {}
    for i in range(n):
        for k in range(m):
            for t in range(p):
                if (i, k) in eligible and max_setups[i][k][t] > 0:
                    can_produce[(i, k, t)] = max_setups[i][k][t]
    
    # For production variables, we need i produced in period t for demand in period s >= t
    # x[i,k,t,s] where s >= t and demands[i][s] > 0
    x = {}
    for i in range(n):
        for t in range(p):
            for s in range(t, p):
                if demands[i][s] > 0:
                    for k in range(m):
                        if (i, k, t) in can_produce:
                            # Also check if machine can be carried over (z) - handled by constraints
                            x[(i, k, t, s)] = model.addVar(
                                vtype=GRB.CONTINUOUS, lb=0,
                                name=f"x_{i}_{k}_{t}_{s}"
                            )
    
    # y[i,j,k,t] - changeover variables (integer)
    y = {}
    for k in range(m):
        for t in range(p):
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if (j, k, t) in can_produce:
                        # Item j needs to be producible on machine k in period t
                        y[(i, j, k, t)] = model.addVar(
                            vtype=GRB.INTEGER, lb=0,
                            ub=can_produce.get((j, k, t), 0),
                            name=f"y_{i}_{j}_{k}_{t}"
                        )
    
    # z[i,k,t] - carryover binary
    z = {}
    for k in range(m):
        for t in range(p):
            for i in range(n):
                z[(i, k, t)] = model.addVar(
                    vtype=GRB.BINARY, lb=0, ub=1,
                    name=f"z_{i}_{k}_{t}"
                )
    
    # w[i,k,t] = total production of item i on machine k in period t
    w = {}
    for i in range(n):
        for k in range(m):
            for t in range(p):
                w[(i, k, t)] = model.addVar(
                    vtype=GRB.CONTINUOUS, lb=0,
                    name=f"w_{i}_{k}_{t}"
                )
    
    # gamma[i,k,t] = number of setups for item i on machine k in period t (via changeovers, not carryover)
    gamma = {}
    for i in range(n):
        for k in range(m):
            for t in range(p):
                gamma[(i, k, t)] = model.addVar(
                    vtype=GRB.INTEGER, lb=0,
                    ub=can_produce.get((i, k, t), 0),
                    name=f"gamma_{i}_{k}_{t}"
                )
    
    model.update()
    
    # Objective: minimize inventory holding costs + setup costs
    obj = gp.LinExpr()
    
    # Inventory holding cost: for item i produced in period t for demand in period s,
    # the holding cost is inventory_costs[i] * (s - t) * x[i,k,t,s]
    for (i, k, t, s), var in x.items():
        if s > t:
            obj.addTerms(inventory_costs[i] * (s - t), var)
    
    # Setup costs
    for (i, j, k, t), var in y.items():
        obj.addTerms(setup_costs[i][j][k], var)
    
    model.setObjective(obj, GRB.MINIMIZE)
    
    # Constraints
    
    # 1. Demand satisfaction: for each item i and demand period s,
    # sum over k, t<=s of x[i,k,t,s] = demands[i][s]
    for i in range(n):
        for s in range(p):
            if demands[i][s] > 0:
                expr = gp.LinExpr()
                for t in range(s + 1):
                    for k in range(m):
                        if (i, k, t, s) in x:
                            expr.addTerms(1, x[(i, k, t, s)])
                model.addConstr(expr == demands[i][s], name=f"demand_{i}_{s}")
    
    # 2. Link w to x: w[i,k,t] = sum_s x[i,k,t,s]
    for i in range(n):
        for k in range(m):
            for t in range(p):
                expr = gp.LinExpr()
                for s in range(t, p):
                    if (i, k, t, s) in x:
                        expr.addTerms(1, x[(i, k, t, s)])
                model.addConstr(w[(i, k, t)] == expr, name=f"link_w_{i}_{k}_{t}")
    
    # 3. Capacity constraint: for each machine k and period t,
    # sum_i processing_time * w[i,k,t] + sum_{i,j} setup_times[i][j][k] * y[i,j,k,t] <= capacity[k][t]
    for k in range(m):
        for t in range(p):
            expr = gp.LinExpr()
            for i in range(n):
                expr.addTerms(processing_time, w[(i, k, t)])
            for i in range(n):
                for j in range(n):
                    if i != j and (i, j, k, t) in y:
                        expr.addTerms(setup_times[i][j][k], y[(i, j, k, t)])
            model.addConstr(expr <= machine_capacities[k][t], name=f"cap_{k}_{t}")
    
    # 4. Each machine configured for exactly one item at start of each period
    for k in range(m):
        for t in range(p):
            expr = gp.LinExpr()
            for i in range(n):
                expr.addTerms(1, z[(i, k, t)])
            model.addConstr(expr == 1, name=f"one_config_{k}_{t}")
    
    # 5. gamma[i,k,t] = number of changeovers INTO item i on machine k in period t
    for i in range(n):
        for k in range(m):
            for t in range(p):
                expr = gp.LinExpr()
                for j in range(n):
                    if j != i and (j, i, k, t) in y:
                        expr.addTerms(1, y[(j, i, k, t)])
                model.addConstr(gamma[(i, k, t)] == expr, name=f"gamma_def_{i}_{k}_{t}")
    
    # 6. Max setups per item: z[i,k,t] + gamma[i,k,t] <= max_setups[i][k][t]
    # Also, if max_setups[i][k][t] == 0, then z[i,k,t] should be 0 if item i can't be on machine k
    for i in range(n):
        for k in range(m):
            for t in range(p):
                ms = max_setups[i][k][t] if (i, k) in eligible else 0
                model.addConstr(z[(i, k, t)] + gamma[(i, k, t)] <= ms,
                                name=f"max_setup_{i}_{k}_{t}")
    
    # 7. Production only if set up: w[i,k,t] <= M * (z[i,k,t] + gamma[i,k,t])
    for i in range(n):
        for k in range(m):
            for t in range(p):
                M_val = machine_capacities[k][t] / processing_time if processing_time > 0 else 0
                M_val = min(M_val, max_production[i][t] if t < p else 0)
                model.addConstr(w[(i, k, t)] <= M_val * (z[(i, k, t)] + gamma[(i, k, t)]),
                                name=f"prod_setup_{i}_{k}_{t}")
    
    # 8. Minimum lot size: if item is produced via changeover (not carryover),
    # the amount must be >= minimum_lot_sizes[i] * gamma[i,k,t]
    # But we need to handle the case where production can also come from carryover.
    # Actually, the constraint is: production from changeover setups must meet min lot.
    # This is tricky. The constraint says: when item i is produced as result of a setup changeover
    # (not carryover), the amount produced must be >= min_lot.
    # 
    # A common formulation: w[i,k,t] >= min_lot * gamma[i,k,t]
    # This ensures that each changeover into i requires at least min_lot production.
    for i in range(n):
        for k in range(m):
            for t in range(p):
                if minimum_lot_sizes[i] > 0:
                    model.addConstr(w[(i, k, t)] >= minimum_lot_sizes[i] * gamma[(i, k, t)],
                                    name=f"min_lot_{i}_{k}_{t}")
    
    # 9. Flow balance on changeovers:
    # At start of period t, machine k is configured for item z[i,k,t].
    # During the period, changeovers happen. The number of times we leave item i =
    # z[i,k,t] (if we leave from initial config) + (times we change INTO i then leave) 
    # This is complex. Let's use a flow conservation approach.
    #
    # For each item i, machine k, period t:
    # (inflow to i) = z[i,k,t] + gamma[i,k,t]  (carry-over + changeovers into i)
    # (outflow from i) = sum_j y[i,j,k,t] + z_end[i,k,t]
    # where z_end[i,k,t] is the ending configuration.
    # And z[i,k,t+1] = z_end[i,k,t] (carryover to next period).
    # 
    # Inflow = Outflow: z[i,k,t] + gamma[i,k,t] = sum_j!=i y[i,j,k,t] + z_end[i,k,t]
    # 
    # For the last period, z_end doesn't matter (or we can ignore).
    # For linking: z[i,k,t+1] = z_end[i,k,t] = z[i,k,t] + gamma[i,k,t] - sum_j!=i y[i,j,k,t]
    
    for k in range(m):
        for t in range(p):
            for i in range(n):
                inflow = gp.LinExpr()
                inflow.addTerms(1, z[(i, k, t)])
                inflow.addTerms(1, gamma[(i, k, t)])
                
                outflow_changeover = gp.LinExpr()
                for j in range(n):
                    if j != i and (i, j, k, t) in y:
                        outflow_changeover.addTerms(1, y[(i, j, k, t)])
                
                if t < p - 1:
                    # z[i,k,t+1] = inflow - outflow_changeover
                    model.addConstr(
                        z[(i, k, t + 1)] == inflow - outflow_changeover,
                        name=f"flow_{i}_{k}_{t}"
                    )
                else:
                    # Last period: inflow >= outflow (ending config absorbs remainder)
                    model.addConstr(
                        inflow >= outflow_changeover,
                        name=f"flow_last_{i}_{k}_{t}"
                    )
    
    # 10. Max production constraint
    for i in range(n):
        for t in range(p):
            expr = gp.LinExpr()
            for k in range(m):
                expr.addTerms(1, w[(i, k, t)])
            model.addConstr(expr <= max_production[i][t], name=f"max_prod_{i}_{t}")
    
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
    solution = {
        "objective_value": 0.0,
        "production": {},
        "setups": {},
        "carryover": {}
    }
    
    if model.SolCount > 0:
        solution["objective_value"] = model.ObjVal
        
        for (i, k, t, s), var in x.items():
            val = var.X
            if val > 1e-6:
                key = f"x_{i}_{k}_{t}_{s}"
                solution["production"][key] = val
        
        for (i, j, k, t), var in y.items():
            val = round(var.X)
            if val > 0:
                key = f"y_{i}_{j}_{k}_{t}"
                solution["setups"][key] = val
        
        for (i, k, t), var in z.items():
            val = round(var.X)
            if val > 0:
                key = f"z_{i}_{k}_{t}"
                solution["carryover"][key] = val
    
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()
    
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)
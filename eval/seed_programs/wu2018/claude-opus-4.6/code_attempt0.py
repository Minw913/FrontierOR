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
    
    num_items = data['num_items']
    num_periods = data['num_periods']
    num_machines = data['num_machines']
    max_machines = data['max_machines_per_item_per_period']
    machine_capabilities = data['machine_capabilities']
    demand = data['demand']
    production_time = data['production_time']
    holding_cost = data['holding_cost']
    setup_cost = data['setup_cost']
    setup_time = data['setup_time']
    production_cost = data['production_cost']
    capacity = data['capacity']
    
    # Initialize logger
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    # Build model
    model = gp.Model("CLSP")
    model.setParam('TimeLimit', max(1, time_limit - 5))
    model.setParam('Threads', 1)
    model.setParam('MIPFocus', 1)  # Focus on finding feasible solutions
    
    # Precompute remaining demand for big-M bounds
    # remaining_demand[i][t] = sum of demand[i][t..T-1]
    remaining_demand = []
    for i in range(num_items):
        rd = [0] * num_periods
        rd[num_periods - 1] = demand[i][num_periods - 1]
        for t in range(num_periods - 2, -1, -1):
            rd[t] = rd[t + 1] + demand[i][t]
        remaining_demand.append(rd)
    
    # Initial inventory (assumed zero if not specified)
    initial_inventory = [0.0] * num_items
    if 'initial_inventory' in data:
        initial_inventory = data['initial_inventory']
    
    # Decision variables
    X = {}  # production quantities
    Y = {}  # setup decisions (binary)
    I = {}  # inventory levels
    
    # Create variables
    for i in range(num_items):
        for t in range(num_periods):
            I[i, t] = model.addVar(lb=0, obj=holding_cost[i], vtype=GRB.CONTINUOUS,
                                    name=f"I_{i}_{t}")
            for m in machine_capabilities[i]:
                m_idx = m  # 1-indexed
                key = f"item_{i+1}_machine_{m_idx}"
                pt = production_time[key]
                st = setup_time[key]
                pc = production_cost[key]
                sc = setup_cost[key]
                cap = capacity[m_idx - 1][t]
                
                # Upper bound on production
                ub1 = remaining_demand[i][t]
                if pt > 0:
                    ub2 = max(0, (cap - st) / pt)
                else:
                    ub2 = remaining_demand[i][t]
                ub = min(ub1, ub2)
                ub = max(0, ub)
                
                X[i, m_idx, t] = model.addVar(lb=0, ub=ub, obj=pc, vtype=GRB.CONTINUOUS,
                                                name=f"X_{i}_{m_idx}_{t}")
                Y[i, m_idx, t] = model.addVar(vtype=GRB.BINARY, obj=sc,
                                                name=f"Y_{i}_{m_idx}_{t}")
    
    model.update()
    
    # Constraints
    # 1. Flow balance: production + inventory_prev - inventory = demand
    for i in range(num_items):
        for t in range(num_periods):
            prod_sum = gp.quicksum(X[i, m, t] for m in machine_capabilities[i])
            inv_prev = I[i, t - 1] if t > 0 else initial_inventory[i]
            model.addConstr(prod_sum + inv_prev - I[i, t] == demand[i][t],
                           name=f"balance_{i}_{t}")
    
    # 2. Setup linking: X[i,m,t] <= UB * Y[i,m,t]
    for i in range(num_items):
        for t in range(num_periods):
            for m in machine_capabilities[i]:
                key = f"item_{i+1}_machine_{m}"
                pt = production_time[key]
                st = setup_time[key]
                cap = capacity[m - 1][t]
                
                ub1 = remaining_demand[i][t]
                if pt > 0:
                    ub2 = max(0, (cap - st) / pt)
                else:
                    ub2 = remaining_demand[i][t]
                ub = min(ub1, ub2)
                ub = max(0, ub)
                
                model.addConstr(X[i, m, t] <= ub * Y[i, m, t],
                               name=f"setup_{i}_{m}_{t}")
    
    # 3. Capacity constraints per machine per period
    for m_idx in range(1, num_machines + 1):
        for t in range(num_periods):
            terms = []
            for i in range(num_items):
                if m_idx in machine_capabilities[i]:
                    key = f"item_{i+1}_machine_{m_idx}"
                    pt = production_time[key]
                    st = setup_time[key]
                    if pt > 0 or st > 0:
                        terms.append((i, pt, st))
            
            if terms:
                expr = gp.LinExpr()
                for (i, pt, st) in terms:
                    if pt > 0:
                        expr.addTerms(pt, X[i, m_idx, t])
                    if st > 0:
                        expr.addTerms(st, Y[i, m_idx, t])
                model.addConstr(expr <= capacity[m_idx - 1][t],
                               name=f"capacity_{m_idx}_{t}")
    
    # 4. Max machines per item per period
    for i in range(num_items):
        if max_machines[i] < len(machine_capabilities[i]):
            for t in range(num_periods):
                model.addConstr(
                    gp.quicksum(Y[i, m, t] for m in machine_capabilities[i]) <= max_machines[i],
                    name=f"max_machines_{i}_{t}")
    
    # Callback for logging
    best_obj = [float('inf')]
    
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj < best_obj[0] - 1e-6:
                best_obj[0] = obj
                if logger:
                    logger.log(obj)
    
    model.optimize(callback)
    
    # Extract solution
    solution = {
        "objective_value": 0.0,
        "production": {},
        "inventory": {},
        "setups": {}
    }
    
    if model.SolCount > 0:
        solution["objective_value"] = model.ObjVal
        
        for i in range(num_items):
            for t in range(num_periods):
                inv_val = I[i, t].X
                solution["inventory"][f"I_item{i+1}_period{t+1}"] = inv_val
                
                for m in machine_capabilities[i]:
                    x_val = X[i, m, t].X
                    y_val = round(Y[i, m, t].X)
                    solution["production"][f"X_item{i+1}_machine{m}_period{t+1}"] = x_val
                    solution["setups"][f"Y_item{i+1}_machine{m}_period{t+1}"] = int(y_val)
    else:
        # No feasible solution found; output zeros
        for i in range(num_items):
            for t in range(num_periods):
                solution["inventory"][f"I_item{i+1}_period{t+1}"] = 0.0
                for m in machine_capabilities[i]:
                    solution["production"][f"X_item{i+1}_machine{m}_period{t+1}"] = 0.0
                    solution["setups"][f"Y_item{i+1}_machine{m}_period{t+1}"] = 0
    
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
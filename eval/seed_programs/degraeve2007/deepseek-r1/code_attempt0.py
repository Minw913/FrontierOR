import json
import gurobipy as gp
from gurobipy import GRB
import argparse
import os

try:
    from solution_logger import SolutionLogger
except ImportError:
    class SolutionLogger:
        def __init__(self, filepath, sense):
            self.filepath = filepath
            self.sense = sense
            if filepath:
                open(filepath, 'w').close()

        def log(self, objective_value):
            if self.filepath:
                with open(self.filepath, 'a') as f:
                    f.write(json.dumps({"objective_value": objective_value}) + '\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        data = json.load(f)
    
    dims = data['dimensions']
    P = dims['num_products']
    T = dims['num_periods']
    params = data['parameters']
    demand = params['demand']
    holding_cost = params['holding_cost']
    setup_cost = params['setup_cost']
    variable_production_cost = params['variable_production_cost']
    variable_production_time = params['variable_production_time']
    setup_time = params['setup_time']
    capacity_list = params['capacity']
    initial_inventory_cost = params['initial_inventory_cost']

    cum_demand = []
    for i in range(P):
        arr = [0] * T
        arr[T-1] = demand[i][T-1]
        for t in range(T-2, -1, -1):
            arr[t] = arr[t+1] + demand[i][t]
        cum_demand.append(arr)
    
    fixed_to_zero = [[False] * T for _ in range(P)]
    U = [[0] * T for _ in range(P)]
    for i in range(P):
        for t in range(T):
            st = setup_time[i][t]
            vt = variable_production_time[i][t]
            cap_t = capacity_list[t]
            if st > cap_t:
                fixed_to_zero[i][t] = True
            else:
                prod_bound = (cap_t - st) / vt if vt > 1e-6 else float('inf')
                U[i][t] = min(cum_demand[i][t], prod_bound)
    
    model = gp.Model("LotSizing")
    I0 = model.addVars(P, name="I0", lb=0.0)
    inventory = model.addVars(P, T, name="inventory", lb=0.0)
    x = model.addVars(P, T, name="production", lb=0.0)
    y = model.addVars(P, T, vtype=GRB.BINARY, name="setup")
    
    for i in range(P):
        for t in range(T):
            if fixed_to_zero[i][t]:
                x[i,t].ub = 0
                y[i,t].ub = 0
                y[i,t].lb = 0
    
    for i in range(P):
        model.addConstr(I0[i] + x[i,0] == demand[i][0] + inventory[i,0], name=f"balance_{i}_0")
        for t in range(1, T):
            model.addConstr(inventory[i,t-1] + x[i,t] == demand[i][t] + inventory[i,t], name=f"balance_{i}_{t}")
    
    for i in range(P):
        for t in range(T):
            if not fixed_to_zero[i][t]:
                model.addConstr(x[i,t] <= U[i][t] * y[i,t], name=f"setup_force_{i}_{t}")
    
    for t in range(T):
        cap_constr_expr = gp.quicksum(
            setup_time[i][t] * y[i,t] + variable_production_time[i][t] * x[i,t] 
            for i in range(P)
        )
        model.addConstr(cap_constr_expr <= capacity_list[t], name=f"capacity_{t}")
    
    obj = gp.quicksum(initial_inventory_cost[i] * I0[i] for i in range(P))
    obj += gp.quicksum(setup_cost[i][t] * y[i,t] for i in range(P) for t in range(T))
    obj += gp.quicksum(variable_production_cost[i][t] * x[i,t] for i in range(P) for t in range(T))
    obj += gp.quicksum(holding_cost[i][t] * inventory[i,t] for i in range(P) for t in range(T))
    model.setObjective(obj)
    
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('Threads', 1)
    
    logger = SolutionLogger(args.log_path, "minimize") if args.log_path else None
    
    for i in range(P):
        I0_val = cum_demand[i][0]
        I0[i].start = I0_val
        for t in range(T):
            if t < T - 1:
                inv_val = cum_demand[i][t+1]
            else:
                inv_val = 0.0
            inventory[i, t].start = inv_val
            x[i, t].start = 0.0
            y[i, t].start = 0.0
    
    def callback(model, where):
        if where == GRB.Callback.MIPSOL and logger:
            obj_value = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            logger.log(obj_value)
    
    model.optimize(callback if logger else None)
    
    status = model.status
    if status == GRB.INFEASIBLE:
        print("Infeasible model. Using fallback solution.")
        sol_I0 = [cum_demand[i][0] for i in range(P)]
        sol_inventory = [[0] * T for _ in range(P)]
        for i in range(P):
            inv = sol_I0[i]
            for t in range(T):
                inv_after = inv - demand[i][t]
                sol_inventory[i][t] = inv_after
                inv = inv_after
        sol_x = [[0.0] * T for _ in range(P)]
        sol_y = [[0] * T for _ in range(P)]
        total_cost = 0.0
        for i in range(P):
            total_cost += initial_inventory_cost[i] * sol_I0[i]
            for t in range(T):
                total_cost += holding_cost[i][t] * sol_inventory[i][t]
    elif status == GRB.OPTIMAL or status == GRB.TIME_LIMIT:
        total_cost = model.ObjVal
        sol_I0 = [I0[i].X if model.SolCount > 0 else 0.0 for i in range(P)]
        sol_inventory = []
        sol_x = []
        sol_y = []
        for i in range(P):
            inv_row = []
            x_row = []
            y_row = []
            for t in range(T):
                if model.SolCount > 0:
                    inv_row.append(inventory[i,t].X)
                    x_row.append(x[i,t].X)
                    y_row.append(int(round(y[i,t].X)))
                else:
                    inv_row.append(0.0)
                    x_row.append(0.0)
                    y_row.append(0)
            sol_inventory.append(inv_row)
            sol_x.append(x_row)
            sol_y.append(y_row)
    else:
        print(f"Unexpected status: {status}. Using fallback solution.")
        sol_I0 = [cum_demand[i][0] for i in range(P)]
        sol_inventory = [[0] * T for _ in range(P)]
        for i in range(P):
            inv = sol_I0[i]
            for t in range(T):
                inv_after = inv - demand[i][t]
                sol_inventory[i][t] = inv_after
                inv = inv_after
        sol_x = [[0.0] * T for _ in range(P)]
        sol_y = [[0] * T for _ in range(P)]
        total_cost = 0.0
        for i in range(P):
            total_cost += initial_inventory_cost[i] * sol_I0[i]
            for t in range(T):
                total_cost += holding_cost[i][t] * sol_inventory[i][t]
    
    solution = {
        "objective_value": total_cost,
        "variables": {
            "production": sol_x,
            "setup": sol_y,
            "inventory": sol_inventory,
            "initial_inventory": sol_I0
        }
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == '__main__':
    main()
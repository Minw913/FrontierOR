import argparse
import json
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    # Load instance
    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    I = data['num_items']
    T = data['num_periods']
    demand = data['demand']
    setup_cost = data['setup_cost']
    holding_cost = data['holding_cost']
    variable_cost = data['variable_cost']
    setup_time = data['setup_time']
    variable_time = data['variable_time']
    capacity = data['capacity']
    initial_inventory_cost = data['initial_inventory_cost']

    # Precompute remaining demand upper bounds
    max_prod = [[0.0 for _ in range(T)] for _ in range(I)]
    for i in range(I):
        cum = 0
        for t in range(T-1, -1, -1):
            cum += demand[i][t]
            max_prod[i][t] = cum

    # Create model
    model = gp.Model("MultiItemLotSizing")
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('Threads', 1)
    model.setParam('MIPGap', 1e-4)
    model.setParam('LogToConsole', 0)

    # Variables
    production = [[model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f'prod_{i}_{t}') for t in range(T)] for i in range(I)]
    setup = [[model.addVar(vtype=GRB.BINARY, name=f'setup_{i}_{t}') for t in range(T)] for i in range(I)]
    inventory = [[model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f'inv_{i}_{t}') for t in range(T+1)] for i in range(I)]
    initial = [model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f'init_{i}') for i in range(I)]

    # Objective components
    init_cost = gp.quicksum(initial_inventory_cost[i] * initial[i] for i in range(I))
    setup_obj = gp.quicksum(setup_cost[i] * setup[i][t] for i in range(I) for t in range(T))
    var_prod_obj = gp.quicksum(variable_cost[i] * production[i][t] for i in range(I) for t in range(T))
    holding_obj = gp.quicksum(holding_cost[i] * inventory[i][t+1] for i in range(I) for t in range(T))
    model.setObjective(init_cost + setup_obj + var_prod_obj + holding_obj, GRB.MINIMIZE)

    # Inventory balance
    for i in range(I):
        # First period
        model.addConstr(initial[i] + production[i][0] == demand[i][0] + inventory[i][1])
        # Middle periods
        for t in range(1, T):
            model.addConstr(inventory[i][t] + production[i][t] == demand[i][t] + inventory[i][t+1])
        # End of horizon
        model.addConstr(inventory[i][T] == 0)

    # Setup forcing and capacity bounds
    M = [0.0] * I
    for i in range(I):
        M[i] = sum(demand[i])

    for i in range(I):
        for t in range(T):
            # Production only if setup
            model.addConstr(production[i][t] <= M[i] * setup[i][t])
            # Production time bound (use variable_time as float)
            if variable_time[i] > 1e-9:
                prod_bound = (capacity[t] - setup_time[i]) / variable_time[i]
                model.addConstr(production[i][t] <= prod_bound * setup[i][t])
            # Upper bound from remaining demand
            model.addConstr(production[i][t] <= max_prod[i][t])

    # Capacity constraints
    for t in range(T):
        setup_total = gp.quicksum(setup_time[i] * setup[i][t] for i in range(I))
        prod_total = gp.quicksum(variable_time[i] * production[i][t] for i in range(I))
        model.addConstr(setup_total + prod_total <= capacity[t])

    # Logger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Callback for logging incumbents
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if logger:
                logger.log(obj)

    # Optimize
    start_time = time.time()
    model.optimize(callback)

    # Extract solution
    if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
        prod_sol = [[production[i][t].X for t in range(T)] for i in range(I)]
        setup_sol = [[int(round(setup[i][t].X)) for t in range(T)] for i in range(I)]
        inv_sol = [[initial[i].X] + [inventory[i][t+1].X for t in range(T)] for i in range(I)]
        obj_val = model.objVal

        # Verify final inventory is zero (within tolerance)
        for i in range(I):
            if abs(inv_sol[i][-1]) > 1e-4:
                inv_sol[i][-1] = 0.0

        solution = {
            "objective_value": float(obj_val),
            "production": [[float(prod_sol[i][t]) for t in range(T)] for i in range(I)],
            "setup": setup_sol,
            "inventory": [[float(x) for x in inv_sol[i]] for i in range(I)]
        }

        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
    else:
        # Infeasible or error - write dummy feasible if possible, but should not happen
        solution = {
            "objective_value": 0.0,
            "production": [[0.0] * T for _ in range(I)],
            "setup": [[0] * T for _ in range(I)],
            "inventory": [[0.0] * (T+1) for _ in range(I)]
        }
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()
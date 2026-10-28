import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    n_items = data['num_items']
    n_periods = data['num_periods']
    n_machines = data['num_machines']
    max_mach_per_item = data['max_machines_per_item_per_period']
    machine_capab = data['machine_capabilities']
    demand = data['demand']
    holding_cost = data['holding_cost']
    capacity = data['capacity']
    
    # Convert capabilities to sets (1-indexed in data)
    can_produce = [[set() for _ in range(n_machines)] for _ in range(n_items)]
    for i in range(n_items):
        for m1 in machine_capab[i]:
            m = m1 - 1  # 0-index
            can_produce[i][m] = True

    # Dicts with string keys
    prod_time = {}
    setup_cost = {}
    setup_time = {}
    prod_cost = {}
    for k, v in data['production_time'].items():
        prod_time[k] = v
    for k, v in data['setup_cost'].items():
        setup_cost[k] = v
    for k, v in data['setup_time'].items():
        setup_time[k] = v
    for k, v in data['production_cost'].items():
        prod_cost[k] = v

    # Precompute total remaining demand for upper bounds
    total_remaining = [[0.0] * n_periods for _ in range(n_items)]
    for i in range(n_items):
        cum = 0.0
        for t in range(n_periods - 1, -1, -1):
            cum += demand[i][t]
            total_remaining[i][t] = cum

    # Initial inventory
    I0 = data.get('initial_inventory', [0.0] * n_items)

    # Create model
    model = gp.Model("MultiItemMultiMachineLotSizing")
    model.Params.TimeLimit = args.time_limit
    model.Params.LogToConsole = 0
    model.Params.Threads = 1
    model.Params.MIPFocus = 1
    model.Params.Heuristics = 0.2
    model.Params.MIPGap = 0.01

    # Variables
    X = {}  # production
    Y = {}  # setup binary
    I = {}  # inventory

    for i in range(n_items):
        for t in range(n_periods):
            I[i, t] = model.addVar(lb=0.0, name=f"I_{i}_{t}")
            for m in range(n_machines):
                if can_produce[i][m]:
                    key = f"item_{i+1}_machine_{m+1}"
                    ub_x = min(total_remaining[i][t], 
                               (capacity[m][t] - setup_time[key]) / prod_time[key] if setup_time[key] < capacity[m][t] else 0.0)
                    X[i, m, t] = model.addVar(lb=0.0, ub=ub_x, name=f"X_{i}_{m}_{t}")
                    Y[i, m, t] = model.addVar(vtype=GRB.BINARY, name=f"Y_{i}_{m}_{t}")

    # Objective
    setup_expr = gp.LinExpr()
    prod_expr = gp.LinExpr()
    hold_expr = gp.LinExpr()

    for i in range(n_items):
        for t in range(n_periods):
            hold_expr.add(I[i, t], holding_cost[i])
            for m in range(n_machines):
                if can_produce[i][m]:
                    key = f"item_{i+1}_machine_{m+1}"
                    setup_expr.add(Y[i, m, t], setup_cost[key])
                    prod_expr.add(X[i, m, t], prod_cost[key])

    model.setObjective(setup_expr + prod_expr + hold_expr, GRB.MINIMIZE)

    # Constraints
    # 1. Inventory balance
    for i in range(n_items):
        # First period
        model.addConstr(I0[i] + gp.quicksum(X[i, m, 0] for m in range(n_machines) if can_produce[i][m]) 
                        == demand[i][0] + I[i, 0])
        for t in range(1, n_periods):
            model.addConstr(I[i, t-1] + gp.quicksum(X[i, m, t] for m in range(n_machines) if can_produce[i][m]) 
                            == demand[i][t] + I[i, t])

    # 2. Capacity
    for m in range(n_machines):
        for t in range(n_periods):
            time_used = gp.LinExpr()
            for i in range(n_items):
                if can_produce[i][m]:
                    key = f"item_{i+1}_machine_{m+1}"
                    time_used.add(X[i, m, t], prod_time[key])
                    time_used.add(Y[i, m, t], setup_time[key])
            model.addConstr(time_used <= capacity[m][t])

    # 3. Setup forcing
    for i in range(n_items):
        for m in range(n_machines):
            if can_produce[i][m]:
                key = f"item_{i+1}_machine_{m+1}"
                bigM = total_remaining[i][0] + 1
                for t in range(n_periods):
                    model.addConstr(X[i, m, t] <= bigM * Y[i, m, t])

    # 4. Max machines per item per period
    for i in range(n_items):
        for t in range(n_periods):
            model.addConstr(gp.quicksum(Y[i, m, t] for m in range(n_machines) if can_produce[i][m]) 
                            <= max_mach_per_item[i])

    # Logger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    best_obj = None

    def callback(model, where):
        nonlocal best_obj
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if best_obj is None or obj < best_obj:
                best_obj = obj
                if logger:
                    logger.log(float(obj))

    # Optimize
    model.optimize(callback)

    # Extract solution
    if model.SolCount > 0:
        objective_value = model.objVal
        production = {}
        inventory = {}
        setups = {}

        for i in range(n_items):
            for t in range(n_periods):
                inv_key = f"I_item{i+1}_period{t+1}"
                inventory[inv_key] = float(I[i, t].X)
                for m in range(n_machines):
                    if can_produce[i][m]:
                        x_key = f"X_item{i+1}_machine{m+1}_period{t+1}"
                        y_key = f"Y_item{i+1}_machine{m+1}_period{t+1}"
                        production[x_key] = float(X[i, m, t].X)
                        setups[y_key] = int(round(Y[i, m, t].X))

        solution = {
            "objective_value": float(objective_value),
            "production": production,
            "inventory": inventory,
            "setups": setups
        }

        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
    else:
        # No solution found - write empty feasible if possible, but per spec we must produce valid JSON
        solution = {
            "objective_value": 0.0,
            "production": {},
            "inventory": {},
            "setups": {}
        }
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()
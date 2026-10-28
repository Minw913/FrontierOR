import argparse
import json
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    prog_start = time.time()

    # Load instance
    with open(args.instance_path, 'r') as f:
        inst = json.load(f)

    P = inst['dimensions']['num_products']
    T = inst['dimensions']['num_periods']

    demand = inst['parameters']['demand']
    holding_cost = inst['parameters']['holding_cost']
    setup_cost = inst['parameters']['setup_cost']
    var_prod_cost = inst['parameters']['variable_production_cost']
    var_prod_time = inst['parameters']['variable_production_time']
    setup_time = inst['parameters']['setup_time']
    capacity = inst['parameters']['capacity']
    init_inv_cost = inst['parameters']['initial_inventory_cost']

    # Build model
    model = gp.Model('lotsizing')
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1

    # Variables
    y = model.addVars(P, T, vtype=GRB.BINARY, name='y')
    x = model.addVars(P, T, vtype=GRB.CONTINUOUS, lb=0.0, name='x')
    s = model.addVars(P, T, vtype=GRB.CONTINUOUS, lb=0.0, name='s')
    I0 = model.addVars(P, vtype=GRB.CONTINUOUS, lb=0.0, name='I0')

    # Inventory flow balance
    for i in range(P):
        model.addConstr(I0[i] + x[i, 0] == demand[i][0] + s[i, 0], name=f'flow_{i}_0')
        for t in range(1, T):
            model.addConstr(s[i, t - 1] + x[i, t] == demand[i][t] + s[i, t], name=f'flow_{i}_{t}')

    # Cumulative remaining demand
    cum_dem = [[0.0] * T for _ in range(P)]
    for i in range(P):
        c = 0.0
        for t in range(T - 1, -1, -1):
            c += demand[i][t]
            cum_dem[i][t] = c

    # Setup linking and production upper bounds
    for i in range(P):
        for t in range(T):
            vpt = var_prod_time[i][t]
            st = setup_time[i][t]
            if vpt > 1e-9:
                cap_bound = (capacity[t] - st) / vpt
            else:
                cap_bound = 0.0
            if cap_bound < 0:
                cap_bound = 0.0
            ub = min(cap_bound, cum_dem[i][t])
            if ub < 1e-9:
                ub = 0.0
            model.addConstr(x[i, t] <= ub * y[i, t], name=f'link_{i}_{t}')

    # Capacity constraints per period
    for t in range(T):
        expr = gp.LinExpr()
        for i in range(P):
            expr += setup_time[i][t] * y[i, t] + var_prod_time[i][t] * x[i, t]
        model.addConstr(expr <= capacity[t], name=f'cap_{t}')

    # Objective
    obj = gp.LinExpr()
    for i in range(P):
        obj += init_inv_cost[i] * I0[i]
    for i in range(P):
        for t in range(T):
            obj += setup_cost[i][t] * y[i, t]
            obj += var_prod_cost[i][t] * x[i, t]
            obj += holding_cost[i][t] * s[i, t]
    model.setObjective(obj, GRB.MINIMIZE)

    # Feasible MIP start: use initial inventory to cover all demand, no production
    start_y = {}
    start_x = {}
    start_s = {}
    start_I0 = {}
    for i in range(P):
        start_I0[i] = cum_dem[i][0]
        for t in range(T):
            start_y[i, t] = 0
            start_x[i, t] = 0.0
            rem = 0.0
            for tau in range(t + 1, T):
                rem += demand[i][tau]
            start_s[i, t] = rem

    model.update()
    start_vars = []
    start_vals = []
    for i in range(P):
        start_vars.append(I0[i])
        start_vals.append(start_I0[i])
    for i in range(P):
        for t in range(T):
            start_vars.append(y[i, t])
            start_vals.append(start_y[i, t])
            start_vars.append(x[i, t])
            start_vals.append(start_x[i, t])
            start_vars.append(s[i, t])
            start_vals.append(start_s[i, t])
    model.setStart(start_vars, start_vals)

    # Time limit for solver (leave room for I/O)
    elapsed = time.time() - prog_start
    solver_time = max(1.0, args.time_limit - elapsed - 2.0)
    model.Params.TimeLimit = solver_time

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Callback to log incumbents
    def callback(m, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = m.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                try:
                    y_vals = m.cbGetSolution(y)
                    x_vals = m.cbGetSolution(x)
                    s_vals = m.cbGetSolution(s)
                    i0_vals = m.cbGetSolution(I0)
                    production = [[float(x_vals[i, t]) for t in range(T)] for i in range(P)]
                    setup = [[int(round(y_vals[i, t])) for t in range(T)] for i in range(P)]
                    inventory = [[float(s_vals[i, t]) for t in range(T)] for i in range(P)]
                    initial_inventory = [float(i0_vals[i]) for i in range(P)]
                    sol_dict = {
                        "objective_value": float(obj_val),
                        "variables": {
                            "production": production,
                            "setup": setup,
                            "inventory": inventory,
                            "initial_inventory": initial_inventory
                        }
                    }
                    logger.log_solution(float(obj_val), sol_dict)
                except Exception:
                    logger.log(float(obj_val))

    model.optimize(callback)

    # Extract final solution
    if model.SolCount == 0:
        # Fallback to trivial feasible solution
        production = [[0.0] * T for _ in range(P)]
        setup = [[0] * T for _ in range(P)]
        inventory = [[0.0] * T for _ in range(P)]
        initial_inventory = [0.0] * P
        for i in range(P):
            initial_inventory[i] = cum_dem[i][0]
            for t in range(T):
                rem = 0.0
                for tau in range(t + 1, T):
                    rem += demand[i][tau]
                inventory[i][t] = rem
        obj_val = 0.0
        for i in range(P):
            obj_val += init_inv_cost[i] * initial_inventory[i]
            for t in range(T):
                obj_val += holding_cost[i][t] * inventory[i][t]
    else:
        y_vals = model.getAttr('X', y)
        x_vals = model.getAttr('X', x)
        s_vals = model.getAttr('X', s)
        i0_vals = model.getAttr('X', I0)
        production = []
        setup = []
        inventory = []
        for i in range(P):
            prow = []
            yrow = []
            srow = []
            for t in range(T):
                xv = float(x_vals[i, t])
                if xv < 1e-6:
                    xv = 0.0
                prow.append(xv)
                yrow.append(int(round(y_vals[i, t])))
                sv = float(s_vals[i, t])
                if sv < 1e-6:
                    sv = 0.0
                srow.append(sv)
            production.append(prow)
            setup.append(yrow)
            inventory.append(srow)
        initial_inventory = [float(i0_vals[i]) for i in range(P)]

        # Recompute objective from variables for consistency
        obj_val = 0.0
        for i in range(P):
            obj_val += init_inv_cost[i] * initial_inventory[i]
            for t in range(T):
                obj_val += setup_cost[i][t] * setup[i][t]
                obj_val += var_prod_cost[i][t] * production[i][t]
                obj_val += holding_cost[i][t] * inventory[i][t]

    solution = {
        "objective_value": float(obj_val),
        "variables": {
            "production": production,
            "setup": setup,
            "inventory": inventory,
            "initial_inventory": initial_inventory
        }
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

    if logger:
        logger.log_solution(float(obj_val), solution)


if __name__ == '__main__':
    main()
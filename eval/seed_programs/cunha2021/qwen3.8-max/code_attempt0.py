import json
import argparse
import time
import sys


def main():
    parser = argparse.ArgumentParser(description='Multi-echelon lot-sizing solver')
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    solution = solve(instance, args.time_limit, logger)

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


def wagner_whitin(demands, fixed_costs, holding_cost, T):
    """Solve uncapacitated lot-sizing via Wagner-Whitin DP."""
    cum = [0] * (T + 1)
    for t in range(T):
        cum[t + 1] = cum[t] + demands[t]

    dp = [0.0] * (T + 1)
    choice = [0] * T

    for t in range(T - 1, -1, -1):
        if cum[T] - cum[t] == 0:
            dp[t] = 0.0
            choice[t] = t
            continue
        dp[t] = float('inf')
        for t_end in range(t, T):
            qty = cum[t_end + 1] - cum[t]
            if qty == 0:
                cost = dp[t_end + 1]
            else:
                hold = 0.0
                for k in range(t, t_end):
                    hold += cum[t_end + 1] - cum[k + 1]
                cost = fixed_costs[t] + holding_cost * hold + dp[t_end + 1]
            if cost < dp[t]:
                dp[t] = cost
                choice[t] = t_end

    orders = [0.0] * T
    inventory = [0.0] * T
    setup = [0] * T
    t = 0
    while t < T:
        t_end = choice[t]
        qty = cum[t_end + 1] - cum[t]
        if qty > 0:
            orders[t] = float(qty)
            setup[t] = 1
            for k in range(t, t_end):
                inventory[k] = float(cum[t_end + 1] - cum[k + 1])
        t = t_end + 1

    return orders, inventory, setup


def compute_heuristic(instance):
    """Bottom-up Wagner-Whitin heuristic."""
    num_r = instance['num_retailers']
    num_w = instance['num_warehouses']
    T = instance['num_periods']
    r_to_w = instance['retailer_warehouse_assignment']
    demands = instance['demands']
    fc = instance['fixed_costs']
    hc = instance['holding_costs']

    w_to_r = {w: [] for w in range(num_w)}
    for r in range(num_r):
        w_to_r[r_to_w[r]].append(r)

    # Solve each retailer
    ret_orders = [[0.0] * T for _ in range(num_r)]
    ret_inv = [[0.0] * T for _ in range(num_r)]
    ret_setup = [[0] * T for _ in range(num_r)]

    for r in range(num_r):
        o, inv, s = wagner_whitin(demands[r], fc['retailers'][r], hc['retailers'][r], T)
        ret_orders[r] = o
        ret_inv[r] = inv
        ret_setup[r] = s

    # Compute warehouse outflow requirements
    wh_outflow = [[0.0] * T for _ in range(num_w)]
    for r in range(num_r):
        w = r_to_w[r]
        for t in range(T):
            wh_outflow[w][t] += ret_orders[r][t]

    # Solve each warehouse
    wh_orders = [[0.0] * T for _ in range(num_w)]
    wh_inv = [[0.0] * T for _ in range(num_w)]
    wh_setup = [[0] * T for _ in range(num_w)]

    for w in range(num_w):
        o, inv, s = wagner_whitin(wh_outflow[w], fc['warehouses'][w], hc['warehouses'], T)
        wh_orders[w] = o
        wh_inv[w] = inv
        wh_setup[w] = s

    # Compute plant outflow requirements
    plant_outflow = [0.0] * T
    for w in range(num_w):
        for t in range(T):
            plant_outflow[t] += wh_orders[w][t]

    # Solve plant
    p_orders, p_inv, p_setup = wagner_whitin(plant_outflow, fc['plant'], hc['plant'], T)

    # Compute objective
    obj = 0.0
    for t in range(T):
        if p_setup[t]:
            obj += fc['plant'][t]
        obj += hc['plant'] * p_inv[t]
    for w in range(num_w):
        for t in range(T):
            if wh_setup[w][t]:
                obj += fc['warehouses'][w][t]
            obj += hc['warehouses'] * wh_inv[w][t]
    for r in range(num_r):
        for t in range(T):
            if ret_setup[r][t]:
                obj += fc['retailers'][r][t]
            obj += hc['retailers'][r] * ret_inv[r][t]

    return {
        "objective_value": obj,
        "x_plant": p_orders,
        "s_plant": p_inv,
        "y_plant": p_setup,
        "x_warehouses": wh_orders,
        "s_warehouses": wh_inv,
        "y_warehouses": wh_setup,
        "x_retailers": ret_orders,
        "s_retailers": ret_inv,
        "y_retailers": ret_setup
    }


def solve(instance, time_limit, logger):
    from gurobipy import Model, GRB, quicksum

    num_r = instance['num_retailers']
    num_w = instance['num_warehouses']
    T = instance['num_periods']
    r_to_w = instance['retailer_warehouse_assignment']
    demands = instance['demands']
    fc = instance['fixed_costs']
    hc = instance['holding_costs']

    w_to_r = {w: [] for w in range(num_w)}
    for r in range(num_r):
        w_to_r[r_to_w[r]].append(r)

    # Compute remaining demands
    rem_r = [[0] * T for _ in range(num_r)]
    for r in range(num_r):
        rem_r[r][T - 1] = demands[r][T - 1]
        for t in range(T - 2, -1, -1):
            rem_r[r][t] = rem_r[r][t + 1] + demands[r][t]

    rem_w = [[0] * T for _ in range(num_w)]
    for w in range(num_w):
        for t in range(T):
            rem_w[w][t] = sum(rem_r[r][t] for r in w_to_r[w])

    rem_p = [0] * T
    for t in range(T):
        rem_p[t] = sum(rem_w[w][t] for w in range(num_w))

    # Get heuristic solution for warm start
    heuristic = compute_heuristic(instance)
    if logger:
        logger.log_solution(heuristic['objective_value'], heuristic)

    # Build MIP
    model = Model("multi_echelon_lotsizing")
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', max(time_limit - 2, 1))
    model.setParam('OutputFlag', 0)

    # Variables
    xp = [model.addVar(lb=0, ub=max(rem_p[t], 0), name=f"xp_{t}") for t in range(T)]
    sp = [model.addVar(lb=0, name=f"sp_{t}") for t in range(T)]
    yp = [model.addVar(vtype=GRB.BINARY, name=f"yp_{t}") for t in range(T)]

    xw = [[model.addVar(lb=0, ub=max(rem_w[w][t], 0), name=f"xw_{w}_{t}") for t in range(T)]
          for w in range(num_w)]
    sw = [[model.addVar(lb=0, name=f"sw_{w}_{t}") for t in range(T)] for w in range(num_w)]
    yw = [[model.addVar(vtype=GRB.BINARY, name=f"yw_{w}_{t}") for t in range(T)]
          for w in range(num_w)]

    xr = [[model.addVar(lb=0, ub=max(rem_r[r][t], 0), name=f"xr_{r}_{t}") for t in range(T)]
          for r in range(num_r)]
    sr = [[model.addVar(lb=0, name=f"sr_{r}_{t}") for t in range(T)] for r in range(num_r)]
    yr = [[model.addVar(vtype=GRB.BINARY, name=f"yr_{r}_{t}") for t in range(T)]
          for r in range(num_r)]

    model.update()

    # Store variables as model attributes for callback access
    model._xp = xp
    model._sp = sp
    model._yp = yp
    model._xw = xw
    model._sw = sw
    model._yw = yw
    model._xr = xr
    model._sr = sr
    model._yr = yr
    model._instance = instance
    model._logger = logger

    # Objective
    obj = 0
    for t in range(T):
        obj += fc['plant'][t] * yp[t] + hc['plant'] * sp[t]
    for w in range(num_w):
        for t in range(T):
            obj += fc['warehouses'][w][t] * yw[w][t] + hc['warehouses'] * sw[w][t]
    for r in range(num_r):
        for t in range(T):
            obj += fc['retailers'][r][t] * yr[r][t] + hc['retailers'][r] * sr[r][t]

    model.setObjective(obj, GRB.MINIMIZE)

    # Plant flow balance
    for t in range(T):
        s_prev = sp[t - 1] if t > 0 else 0
        model.addConstr(s_prev + xp[t] == quicksum(xw[w][t] for w in range(num_w)) + sp[t],
                        name=f"plant_bal_{t}")

    # Warehouse flow balance
    for w in range(num_w):
        for t in range(T):
            s_prev = sw[w][t - 1] if t > 0 else 0
            model.addConstr(s_prev + xw[w][t] == quicksum(xr[r][t] for r in w_to_r[w]) + sw[w][t],
                            name=f"wh_bal_{w}_{t}")

    # Retailer flow balance
    for r in range(num_r):
        for t in range(T):
            s_prev = sr[r][t - 1] if t > 0 else 0
            model.addConstr(s_prev + xr[r][t] == demands[r][t] + sr[r][t],
                            name=f"ret_bal_{r}_{t}")

    # Big-M constraints (setup linking)
    for t in range(T):
        if rem_p[t] > 0:
            model.addConstr(xp[t] <= rem_p[t] * yp[t], name=f"plant_bm_{t}")
        else:
            model.addConstr(yp[t] == 0)

    for w in range(num_w):
        for t in range(T):
            if rem_w[w][t] > 0:
                model.addConstr(xw[w][t] <= rem_w[w][t] * yw[w][t], name=f"wh_bm_{w}_{t}")
            else:
                model.addConstr(yw[w][t] == 0)

    for r in range(num_r):
        for t in range(T):
            if rem_r[r][t] > 0:
                model.addConstr(xr[r][t] <= rem_r[r][t] * yr[r][t], name=f"ret_bm_{r}_{t}")
            else:
                model.addConstr(yr[r][t] == 0)

    # Inventory upper bounds
    for t in range(T - 1):
        model.addConstr(sp[t] <= rem_p[t + 1], name=f"plant_inv_ub_{t}")
    for w in range(num_w):
        for t in range(T - 1):
            model.addConstr(sw[w][t] <= rem_w[w][t + 1], name=f"wh_inv_ub_{w}_{t}")
    for r in range(num_r):
        for t in range(T - 1):
            model.addConstr(sr[r][t] <= rem_r[r][t + 1], name=f"ret_inv_ub_{r}_{t}")

    # End of horizon: zero inventory
    model.addConstr(sp[T - 1] == 0)
    for w in range(num_w):
        model.addConstr(sw[w][T - 1] == 0)
    for r in range(num_r):
        model.addConstr(sr[r][T - 1] == 0)

    # Set warm start from heuristic
    for t in range(T):
        xp[t].Start = heuristic['x_plant'][t]
        sp[t].Start = heuristic['s_plant'][t]
        yp[t].Start = heuristic['y_plant'][t]
    for w in range(num_w):
        for t in range(T):
            xw[w][t].Start = heuristic['x_warehouses'][w][t]
            sw[w][t].Start = heuristic['s_warehouses'][w][t]
            yw[w][t].Start = heuristic['y_warehouses'][w][t]
    for r in range(num_r):
        for t in range(T):
            xr[r][t].Start = heuristic['x_retailers'][r][t]
            sr[r][t].Start = heuristic['s_retailers'][r][t]
            yr[r][t].Start = heuristic['y_retailers'][r][t]

    # Callback for logging
    best_obj = [heuristic['objective_value']]

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj_val < best_obj[0] - 1e-6:
                best_obj[0] = obj_val
                sol = extract_solution_cb(model)
                if model._logger:
                    model._logger.log_solution(obj_val, sol)

    model.optimize(callback)

    # Extract final solution
    if model.SolCount > 0:
        solution = extract_solution(model)
        if logger and solution['objective_value'] < best_obj[0] - 1e-6:
            logger.log_solution(solution['objective_value'], solution)
        return solution
    else:
        return heuristic


def extract_solution_cb(model):
    """Extract solution from callback."""
    inst = model._instance
    num_r = inst['num_retailers']
    num_w = inst['num_warehouses']
    T = inst['num_periods']

    xp = [model.cbGetSolution(model._xp[t]) for t in range(T)]
    sp = [model.cbGetSolution(model._sp[t]) for t in range(T)]
    yp = [int(round(model.cbGetSolution(model._yp[t]))) for t in range(T)]

    xw = [[model.cbGetSolution(model._xw[w][t]) for t in range(T)] for w in range(num_w)]
    sw = [[model.cbGetSolution(model._sw[w][t]) for t in range(T)] for w in range(num_w)]
    yw = [[int(round(model.cbGetSolution(model._yw[w][t]))) for t in range(T)] for w in range(num_w)]

    xr = [[model.cbGetSolution(model._xr[r][t]) for t in range(T)] for r in range(num_r)]
    sr = [[model.cbGetSolution(model._sr[r][t]) for t in range(T)] for r in range(num_r)]
    yr = [[int(round(model.cbGetSolution(model._yr[r][t]))) for t in range(T)] for r in range(num_r)]

    obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)

    return {
        "objective_value": obj,
        "x_plant": xp,
        "s_plant": sp,
        "y_plant": yp,
        "x_warehouses": xw,
        "s_warehouses": sw,
        "y_warehouses": yw,
        "x_retailers": xr,
        "s_retailers": sr,
        "y_retailers": yr
    }


def extract_solution(model):
    """Extract solution from model after optimization."""
    inst = model._instance
    num_r = inst['num_retailers']
    num_w = inst['num_warehouses']
    T = inst['num_periods']

    xp = [model._xp[t].X for t in range(T)]
    sp = [model._sp[t].X for t in range(T)]
    yp = [int(round(model._yp[t].X)) for t in range(T)]

    xw = [[model._xw[w][t].X for t in range(T)] for w in range(num_w)]
    sw = [[model._sw[w][t].X for t in range(T)] for w in range(num_w)]
    yw = [[int(round(model._yw[w][t].X)) for t in range(T)] for w in range(num_w)]

    xr = [[model._xr[r][t].X for t in range(T)] for r in range(num_r)]
    sr = [[model._sr[r][t].X for t in range(T)] for r in range(num_r)]
    yr = [[int(round(model._yr[r][t].X)) for t in range(T)] for r in range(num_r)]

    obj = model.ObjVal

    return {
        "objective_value": obj,
        "x_plant": xp,
        "s_plant": sp,
        "y_plant": yp,
        "x_warehouses": xw,
        "s_warehouses": sw,
        "y_warehouses": yw,
        "x_retailers": xr,
        "s_retailers": sr,
        "y_retailers": yr
    }


if __name__ == '__main__':
    main()
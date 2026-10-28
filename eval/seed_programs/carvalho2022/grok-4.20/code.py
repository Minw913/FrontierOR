import json
import argparse
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

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    n = data['dimensions']['n']
    m = data['dimensions']['m']
    p = data['dimensions']['p']

    demands = data['demands']
    inventory_costs = data['inventory_costs']
    processing_time = data['processing_time']
    setup_times = data['setup_times']
    setup_costs = data['setup_costs']
    machine_capacities = data['machine_capacities']
    max_production = data['max_production']
    max_setups_per_item = data['max_setups_per_item']
    minimum_lot_sizes = data['minimum_lot_sizes']
    machine_eligibility = data['machine_eligibility']

    total_demand_from = [[0] * p for _ in range(n)]
    for i in range(n):
        cum = 0
        for t in range(p - 1, -1, -1):
            cum += demands[i][t]
            total_demand_from[i][t] = cum

    model = gp.Model("MultiItemLotSizingWithSetupCarryover")
    model.Params.TimeLimit = args.time_limit
    model.Params.Threads = 1
    model.Params.LogToConsole = 0
    model.Params.MIPGap = 1e-4
    model.Params.Heuristics = 0.2
    model.Params.MIPFocus = 1
    model.Params.LazyConstraints = 1

    # Variables
    x = model.addVars(n, m, p, p, vtype=GRB.CONTINUOUS, lb=0, name="x")
    y = model.addVars(n, n, m, p, vtype=GRB.INTEGER, lb=0, name="y")
    z = model.addVars(n, m, p, vtype=GRB.BINARY, name="z")
    I = model.addVars(n, p+1, vtype=GRB.CONTINUOUS, lb=0, name="I")
    prod_qty = model.addVars(n, m, p, vtype=GRB.CONTINUOUS, lb=0, name="prod")
    w = model.addVars(n, m, p, vtype=GRB.BINARY, name="w")

    # Objective
    inventory_cost_expr = gp.quicksum(inventory_costs[i] * I[i, t] for i in range(n) for t in range(p))
    setup_cost_expr = gp.quicksum(setup_costs[i][j][k] * y[i, j, k, t]
                                  for i in range(n) for j in range(n) for k in range(m) for t in range(p) if i != j)
    model.setObjective(inventory_cost_expr + setup_cost_expr, GRB.MINIMIZE)

    # Inventory balance and production linking
    for i in range(n):
        model.addConstr(I[i, 0] == 0, f"init_inv_{i}")
        for t in range(p):
            prod_sum = gp.quicksum(x[i, k, t, s] for k in range(m) for s in range(t, p))
            model.addConstr(I[i, t] + prod_sum == demands[i][t] + I[i, t+1], f"balance_{i}_{t}")
            model.addConstr(prod_qty[i, k, t] == gp.quicksum(x[i, k, t, s] for s in range(t, p)),
                            f"link_prod_{i}_{k}_{t}")
        model.addConstr(I[i, p] == 0, f"final_inv_{i}")

    # Capacity
    for k in range(m):
        for t in range(p):
            prod_time = processing_time * gp.quicksum(prod_qty[i, k, t] for i in range(n))
            setup_time = gp.quicksum(setup_times[i][j][k] * y[i, j, k, t]
                                     for i in range(n) for j in range(n) if i != j)
            model.addConstr(prod_time + setup_time <= machine_capacities[k][t], f"cap_{k}_{t}")

    # Production only if setup
    bigM = [sum(demands[i]) + 1 for i in range(n)]
    for i in range(n):
        for k in range(m):
            for t in range(p):
                model.addConstr(prod_qty[i, k, t] <= bigM[i] * w[i, k, t], f"setup_link1_{i}_{k}_{t}")

    # One carryover state per machine per period
    for k in range(m):
        for t in range(p):
            model.addConstr(gp.quicksum(z[i, k, t] for i in range(n)) == 1, f"one_state_{k}_{t}")

    # Flow conservation for setups and w (corrected)
    for k in range(m):
        for t in range(p):
            for j in range(n):
                inflow = (gp.quicksum(y[i, j, k, t] for i in range(n) if i != j) + z[j, k, t])
                outflow = gp.quicksum(y[j, l, k, t] for l in range(n) if l != j)
                model.addConstr(inflow == w[j, k, t] + outflow, f"flow_{j}_{k}_{t}")

    # Carryover linking between periods
    for k in range(m):
        for t in range(1, p):
            for i in range(n):
                incoming = (z[i, k, t-1] + gp.quicksum(y[j, i, k, t-1] for j in range(n) if j != i))
                model.addConstr(z[i, k, t] <= incoming, f"carry_link_{i}_{k}_{t}")

    # Eligibility
    for i in range(n):
        for k in range(m):
            if machine_eligibility[i][k] == 0:
                for t in range(p):
                    model.addConstr(w[i, k, t] == 0, f"elig_w_{i}_{k}_{t}")
                    model.addConstr(z[i, k, t] == 0, f"elig_z_{i}_{k}_{t}")
                for t_val in range(p):
                    for j_val in range(n):
                        if i != j_val:
                            model.addConstr(y[i, j_val, k, t_val] == 0, f"elig_yout_{i}_{j_val}_{k}_{t_val}")
                            model.addConstr(y[j_val, i, k, t_val] == 0, f"elig_yin_{j_val}_{i}_{k}_{t_val}")

    # Minimum lot sizes for new setups
    for i in range(n):
        minlot = minimum_lot_sizes[i]
        if minlot > 0:
            for k in range(m):
                for t in range(p):
                    model.addConstr(prod_qty[i, k, t] >= minlot * (w[i, k, t] - z[i, k, t]),
                                    f"minlot_{i}_{k}_{t}")

    # Max production (cumulative from period t onward)
    for i in range(n):
        for t in range(p):
            total_prod = gp.quicksum(x[i, k, tau, s]
                                     for k in range(m)
                                     for tau in range(t, p)
                                     for s in range(tau, p))
            model.addConstr(total_prod <= max_production[i][t], f"maxprod_{i}_{t}")

    # Max setups per item and upper bounds on y
    for i in range(n):
        for k in range(m):
            for t in range(p):
                max_set = max_setups_per_item[i][k][t]
                if max_set == 0:
                    for j in range(n):
                        if i != j:
                            model.addConstr(y[i, j, k, t] == 0, f"maxsetup_y_{i}_{j}_{k}_{t}")
                            model.addConstr(y[j, i, k, t] == 0, f"maxsetup_yin_{j}_{i}_{k}_{t}")
                else:
                    outgoing = gp.quicksum(y[i, j, k, t] for j in range(n) if i != j)
                    incoming = gp.quicksum(y[j, i, k, t] for j in range(n) if j != i)
                    model.addConstr(outgoing + incoming <= max_set, f"maxsetup_{i}_{k}_{t}")

    # Upper bounds on x and y
    for i in range(n):
        for k in range(m):
            for t in range(p):
                for s in range(t, p):
                    x[i, k, t, s].ub = total_demand_from[i][t]
                for j in range(n):
                    if i != j:
                        y[i, j, k, t].ub = max_setups_per_item[i][k][t]

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    best_obj = None

    def callback(model, where):
        nonlocal best_obj
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if best_obj is None or obj < best_obj - 1e-4:
                best_obj = obj
                if logger:
                    logger.log(float(obj))

    try:
        model.optimize(callback)
    except gp.GurobiError as e:
        print("Gurobi Error:", e, file=sys.stderr)
        solution = {
            "objective_value": 0.0,
            "production": {},
            "setups": {},
            "carryover": {}
        }
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
        return

    if model.SolCount > 0:
        objective_value = model.ObjVal

        production = {}
        for i_idx in range(n):
            for k_idx in range(m):
                for t_idx in range(p):
                    for s_idx in range(t_idx, p):
                        val = x[i_idx, k_idx, t_idx, s_idx].X
                        if val > 1e-6:
                            production[f"x_{i_idx}_{k_idx}_{t_idx}_{s_idx}"] = float(round(val, 6))

        setups = {}
        for i_idx in range(n):
            for j_idx in range(n):
                if i_idx == j_idx: continue
                for k_idx in range(m):
                    for t_idx in range(p):
                        val = y[i_idx, j_idx, k_idx, t_idx].X
                        if val > 0.5:
                            setups[f"y_{i_idx}_{j_idx}_{k_idx}_{t_idx}"] = int(round(val))

        carryover = {}
        for i_idx in range(n):
            for k_idx in range(m):
                for t_idx in range(p):
                    val = z[i_idx, k_idx, t_idx].X
                    if val > 0.5:
                        carryover[f"z_{i_idx}_{k_idx}_{t_idx}"] = 1

        solution = {
            "objective_value": float(objective_value),
            "production": production,
            "setups": setups,
            "carryover": carryover
        }
    else:
        solution = {
            "objective_value": 0.0,
            "production": {},
            "setups": {},
            "carryover": {}
        }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()
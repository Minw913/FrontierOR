import json
import argparse
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

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    T = data['num_periods']
    W = data['num_warehouses']
    R = data['num_retailers']
    S = data['num_scenarios']
    prob = data['scenario_probabilities']
    retailer_to_warehouse = data['retailer_to_warehouse']
    h_p = data['holding_cost_plant']
    h_w = data['holding_cost_warehouse']
    h_r = data['holding_costs_retailers']
    setup_p = data['setup_costs_plant']
    setup_w = data['setup_costs_warehouses']
    setup_r = data['setup_costs_retailers']
    d = data['demands']
    initial_inventory = data.get('initial_inventory', 0)
    initial_setups_imposed = data.get('initial_setups_imposed', True)

    # Precompute big-M per retailer-scenario (tighter)
    M = [[0] * R for _ in range(S)]
    for s in range(S):
        for r in range(R):
            M[s][r] = sum(d[s][r][t] for t in range(T))

    model = gp.Model("DistributionPlanning")
    model.setParam(GRB.Param.TimeLimit, args.time_limit)
    model.setParam(GRB.Param.Threads, 1)
    model.setParam(GRB.Param.MIPGap, 1e-4)
    model.setParam(GRB.Param.LogToConsole, 0)

    # Setup binaries
    y_plant = model.addVars(T, vtype=GRB.BINARY, name="y_plant")
    y_ware = model.addVars(W, T, vtype=GRB.BINARY, name="y_ware")
    y_ret = model.addVars(R, T, vtype=GRB.BINARY, name="y_ret")

    # Production / delivery / inventory (aggregated)
    prod = model.addVars(T, S, lb=0, name="prod")
    deli_w = model.addVars(W, T, S, lb=0, name="deli_w")
    deli_r = model.addVars(R, T, S, lb=0, name="deli_r")
    inv_p = model.addVars(T, S, lb=0, name="inv_p")
    inv_w = model.addVars(W, T, S, lb=0, name="inv_w")
    inv_r = model.addVars(R, T, S, lb=0, name="inv_r")

    # Objective
    setup_cost = (gp.quicksum(setup_p[t] * y_plant[t] for t in range(T)) +
                  gp.quicksum(setup_w[w][t] * y_ware[w, t] for w in range(W) for t in range(T)) +
                  gp.quicksum(setup_r[r][t] * y_ret[r, t] for r in range(R) for t in range(T)))

    holding = 0.0
    for s in range(S):
        ps = prob[s]
        for t in range(T):
            holding += ps * h_p * inv_p[t, s]
            for w in range(W):
                holding += ps * h_w * inv_w[w, t, s]
            for r in range(R):
                holding += ps * h_r[r] * inv_r[r, t, s]

    model.setObjective(setup_cost + holding, GRB.MINIMIZE)

    # Initial inventory (usually 0)
    for s in range(S):
        model.addConstr(inv_p[0, s] == initial_inventory)
        for w in range(W):
            model.addConstr(inv_w[w, 0, s] == initial_inventory)
        for r in range(R):
            model.addConstr(inv_r[r, 0, s] == initial_inventory)

    if initial_setups_imposed:
        model.addConstr(y_plant[0] == 1)
        for w in range(W):
            model.addConstr(y_ware[w, 0] == 1)
        for r in range(R):
            model.addConstr(y_ret[r, 0] == 1)

    # Balance constraints
    for s in range(S):
        # Plant
        for t in range(T):
            lhs = prod[t, s] + (inv_p[t-1, s] if t > 0 else 0)
            rhs = inv_p[t, s]
            for w in range(W):
                rhs += deli_w[w, t, s]
            model.addConstr(lhs == rhs, f"bal_p_{t}_{s}")

        # Warehouses
        for w in range(W):
            for t in range(T):
                lhs = deli_w[w, t, s] + (inv_w[w, t-1, s] if t > 0 else 0)
                rhs = inv_w[w, t, s]
                for rr in range(R):
                    if retailer_to_warehouse[rr] == w:
                        rhs += deli_r[rr, t, s]
                model.addConstr(lhs == rhs, f"bal_w_{w}_{t}_{s}")

        # Retailers
        for r in range(R):
            for t in range(T):
                lhs = deli_r[r, t, s] + (inv_r[r, t-1, s] if t > 0 else 0)
                rhs = inv_r[r, t, s] + d[s][r][t]
                model.addConstr(lhs == rhs, f"bal_r_{r}_{t}_{s}")

    # Setup linking
    for s in range(S):
        for t in range(T):
            model.addConstr(prod[t, s] <= M[s][0] * y_plant[t])
            for w in range(W):
                model.addConstr(deli_w[w, t, s] <= M[s][0] * y_ware[w, t])
            for r in range(R):
                model.addConstr(deli_r[r, t, s] <= M[s][r] * y_ret[r, t])

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(float(obj))

    model.optimize(callback)

    status = model.status
    if status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT):
        # If infeasible or no solution, write minimal feasible dict
        obj_val = 0.0
        sol = {"objective_value": obj_val, "setup_variables": {}, "production_plant": {},
               "delivery_warehouse": {}, "delivery_retailer": {}, "inventory_plant": {},
               "inventory_warehouse": {}, "inventory_retailer": {}}
        for t in range(T):
            sol["setup_variables"][f"y_0_{t}"] = 1
            for w in range(W):
                sol["setup_variables"][f"y_{w+1}_{t}"] = 1
            for r in range(R):
                sol["setup_variables"][f"y_{W+1+r}_{t}"] = 1
        for t in range(T):
            for s in range(S):
                sol["production_plant"][f"{t}_{s}"] = 0
                sol["inventory_plant"][f"{t}_{s}"] = 0
                for w in range(W):
                    sol["delivery_warehouse"][f"{w}_{t}_{s}"] = 0
                    sol["inventory_warehouse"][f"{w}_{t}_{s}"] = 0
                for r in range(R):
                    sol["delivery_retailer"][f"{r}_{t}_{s}"] = 0
                    sol["inventory_retailer"][f"{r}_{t}_{s}"] = 0
        with open(args.solution_path, 'w') as f:
            json.dump(sol, f, indent=2)
        return

    obj_val = model.objVal

    solution = {
        "objective_value": float(obj_val),
        "setup_variables": {},
        "production_plant": {},
        "delivery_warehouse": {},
        "delivery_retailer": {},
        "inventory_plant": {},
        "inventory_warehouse": {},
        "inventory_retailer": {}
    }

    # Extract setups
    for t in range(T):
        solution["setup_variables"][f"y_0_{t}"] = int(round(y_plant[t].X))
    for w in range(W):
        for t in range(T):
            solution["setup_variables"][f"y_{w+1}_{t}"] = int(round(y_ware[w, t].X))
    for r in range(R):
        for t in range(T):
            solution["setup_variables"][f"y_{W + 1 + r}_{t}"] = int(round(y_ret[r, t].X))

    # Extract continuous variables
    for t in range(T):
        for s in range(S):
            solution["production_plant"][f"{t}_{s}"] = round(prod[t, s].X)
            solution["inventory_plant"][f"{t}_{s}"] = round(inv_p[t, s].X)
            for w in range(W):
                solution["delivery_warehouse"][f"{w}_{t}_{s}"] = round(deli_w[w, t, s].X)
                solution["inventory_warehouse"][f"{w}_{t}_{s}"] = round(inv_w[w, t, s].X)
            for r in range(R):
                solution["delivery_retailer"][f"{r}_{t}_{s}"] = round(deli_r[r, t, s].X)
                solution["inventory_retailer"][f"{r}_{t}_{s}"] = round(inv_r[r, t, s].X)

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()
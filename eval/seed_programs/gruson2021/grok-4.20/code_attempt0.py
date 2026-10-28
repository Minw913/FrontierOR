import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
import sys

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
    initial_inventory = data['initial_inventory']
    initial_setups_imposed = data['initial_setups_imposed']

    # Precompute max demand for big-M
    M = [[0 for r in range(R)] for s in range(S)]
    for s in range(S):
        for r in range(R):
            M[s][r] = sum(d[s][r][t] for t in range(T))

    model = gp.Model("MultiStageDistribution")

    # Setup variables
    y_plant = model.addVars(T, vtype=GRB.BINARY, name="y_plant")
    y_ware = model.addVars(W, T, vtype=GRB.BINARY, name="y_ware")
    y_ret = model.addVars(R, T, vtype=GRB.BINARY, name="y_ret")

    # Production and delivery variables (aggregated)
    prod = model.addVars(T, S, lb=0, name="prod")
    deli_w = model.addVars(W, T, S, lb=0, name="deli_w")
    deli_r = model.addVars(R, T, S, lb=0, name="deli_r")

    # Inventory variables (aggregated)
    inv_p = model.addVars(T, S, lb=0, name="inv_p")
    inv_w = model.addVars(W, T, S, lb=0, name="inv_w")
    inv_r = model.addVars(R, T, S, lb=0, name="inv_r")

    # Objective: setup + expected holding
    setup_cost = gp.quicksum(setup_p[t] * y_plant[t] for t in range(T))
    for w in range(W):
        setup_cost += gp.quicksum(setup_w[w][t] * y_ware[w, t] for t in range(T))
    for r in range(R):
        setup_cost += gp.quicksum(setup_r[r][t] * y_ret[r, t] for t in range(T))

    holding_cost = 0
    for s in range(S):
        for t in range(T):
            holding_cost += prob[s] * h_p * inv_p[t, s]
            for w in range(W):
                holding_cost += prob[s] * h_w * inv_w[w, t, s]
            for r in range(R):
                holding_cost += prob[s] * h_r[r] * inv_r[r, t, s]

    model.setObjective(setup_cost + holding_cost, GRB.MINIMIZE)

    # Initial inventory and setups
    for s in range(S):
        model.addConstr(inv_p[0, s] == initial_inventory, name=f"init_inv_p_{s}")
        for w in range(W):
            model.addConstr(inv_w[w, 0, s] == initial_inventory, name=f"init_inv_w_{w}_{s}")
        for r in range(R):
            model.addConstr(inv_r[r, 0, s] == initial_inventory, name=f"init_inv_r_{r}_{s}")

    if initial_setups_imposed:
        model.addConstr(y_plant[0] == 1)
        for w in range(W):
            model.addConstr(y_ware[w, 0] == 1)
        for r in range(R):
            model.addConstr(y_ret[r, 0] == 1)

    # Flow balance constraints
    for s in range(S):
        # Plant
        for t in range(T):
            inflow = prod[t, s] if t == 0 else prod[t, s] + inv_p[t-1, s]
            outflow = inv_p[t, s]
            for w_idx in range(W):
                outflow += deli_w[w_idx, t, s]
            model.addConstr(inflow == outflow, name=f"bal_p_{t}_{s}")

        # Warehouses
        for w in range(W):
            for t in range(T):
                inflow = deli_w[w, t, s] if t == 0 else deli_w[w, t, s] + inv_w[w, t-1, s]
                outflow = inv_w[w, t, s]
                for rr in range(R):
                    if retailer_to_warehouse[rr] == w:
                        outflow += deli_r[rr, t, s]
                model.addConstr(inflow == outflow, name=f"bal_w_{w}_{t}_{s}")

        # Retailers
        for r in range(R):
            for t in range(T):
                inflow = deli_r[r, t, s] if t == 0 else deli_r[r, t, s] + inv_r[r, t-1, s]
                if t == 0:
                    model.addConstr(inflow == d[s][r][0] + inv_r[r, 0, s], name=f"bal_r_{r}_{0}_{s}")
                else:
                    model.addConstr(inflow == d[s][r][t] + inv_r[r, t, s], name=f"bal_r_{r}_{t}_{s}")

    # Linking constraints with setups
    for t in range(T):
        for s in range(S):
            model.addConstr(prod[t, s] <= M[s][0] * y_plant[t], name=f"link_prod_{t}_{s}")
            for w in range(W):
                model.addConstr(deli_w[w, t, s] <= M[s][0] * y_ware[w, t], name=f"link_deli_w_{w}_{t}_{s}")
            for r in range(R):
                model.addConstr(deli_r[r, t, s] <= M[s][r] * y_ret[r, t], name=f"link_deli_r_{r}_{t}_{s}")

    # Non-negativity already set via lb=0

    # Set time limit
    model.setParam(GRB.Param.TimeLimit, args.time_limit)
    model.setParam(GRB.Param.Threads, 1)
    model.setParam(GRB.Param.MIPGap, 0.01)  # Allow 1% gap for efficiency

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj)

    model.optimize(callback)

    # Extract solution
    solution = {
        "objective_value": model.objVal if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SOLUTION_LIMIT] else 0.0,
        "setup_variables": {},
        "production_plant": {},
        "delivery_warehouse": {},
        "delivery_retailer": {},
        "inventory_plant": {},
        "inventory_warehouse": {},
        "inventory_retailer": {}
    }

    # Setup variables
    for t in range(T):
        val = 1 if y_plant[t].X > 0.5 else 0
        solution["setup_variables"][f"y_0_{t}"] = val
    for w in range(W):
        for t in range(T):
            val = 1 if y_ware[w, t].X > 0.5 else 0
            solution["setup_variables"][f"y_{w+1}_{t}"] = val
    for r in range(R):
        for t in range(T):
            val = 1 if y_ret[r, t].X > 0.5 else 0
            solution["setup_variables"][f"y_{W+1+r}_{t}"] = val

    # Production, delivery, inventory
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
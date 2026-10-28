import argparse
import json
from typing import Dict, Any

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def solve(instance: Dict[str, Any], time_limit: int, logger: SolutionLogger = None):
    # -----------------------------
    # Read instance data
    # -----------------------------
    T = int(instance["num_periods"])
    W = int(instance["num_warehouses"])
    R = int(instance["num_retailers"])
    S = int(instance["num_scenarios"])

    initial_inventory = float(instance["initial_inventory"])
    initial_setups_imposed = bool(instance["initial_setups_imposed"])

    retailer_to_warehouse = instance["retailer_to_warehouse"]
    scenario_probabilities = instance["scenario_probabilities"]

    holding_cost_plant = float(instance["holding_cost_plant"])
    holding_cost_warehouse = float(instance["holding_cost_warehouse"])
    holding_costs_retailers = instance["holding_costs_retailers"]

    setup_costs_plant = instance["setup_costs_plant"]
    setup_costs_warehouses = instance["setup_costs_warehouses"]
    setup_costs_retailers = instance["setup_costs_retailers"]

    demands = instance["demands"]  # [s][r][t]

    # Retailers served by each warehouse
    retailers_by_w = [[] for _ in range(W)]
    for r in range(R):
        retailers_by_w[retailer_to_warehouse[r]].append(r)

    # Big-M values (scenario-dependent, tighter than global)
    total_demand_s = [sum(demands[s][r][t] for r in range(R) for t in range(T)) for s in range(S)]
    total_demand_wr_s = [[sum(demands[s][r][t] for t in range(T)) for r in range(R)] for s in range(S)]
    total_demand_ww_s = [
        [sum(demands[s][r][t] for r in retailers_by_w[w] for t in range(T)) for w in range(W)]
        for s in range(S)
    ]

    # -----------------------------
    # Build model
    # -----------------------------
    m = gp.Model("stochastic_3level_lotsizing")
    m.Params.TimeLimit = time_limit
    m.Params.Threads = 1
    m.Params.OutputFlag = 0

    # First-stage setup variables
    y_plant = m.addVars(T, vtype=GRB.BINARY, name="y_plant")
    y_wh = m.addVars(W, T, vtype=GRB.BINARY, name="y_wh")
    y_ret = m.addVars(R, T, vtype=GRB.BINARY, name="y_ret")

    # Second-stage variables (scenario-dependent)
    prod = m.addVars(T, S, lb=0.0, vtype=GRB.CONTINUOUS, name="prod")
    ship_pw = m.addVars(W, T, S, lb=0.0, vtype=GRB.CONTINUOUS, name="ship_pw")
    ship_wr = m.addVars(R, T, S, lb=0.0, vtype=GRB.CONTINUOUS, name="ship_wr")

    inv_p = m.addVars(T, S, lb=0.0, vtype=GRB.CONTINUOUS, name="inv_p")
    inv_w = m.addVars(W, T, S, lb=0.0, vtype=GRB.CONTINUOUS, name="inv_w")
    inv_r = m.addVars(R, T, S, lb=0.0, vtype=GRB.CONTINUOUS, name="inv_r")

    # Initial setups if imposed
    if initial_setups_imposed:
        m.addConstr(y_plant[0] == 1, name="init_setup_plant")
        for w in range(W):
            m.addConstr(y_wh[w, 0] == 1, name=f"init_setup_wh_{w}")
        for r in range(R):
            m.addConstr(y_ret[r, 0] == 1, name=f"init_setup_ret_{r}")

    # Inventory-flow balance constraints
    for s in range(S):
        for t in range(T):
            # Plant balance
            prev_inv_p = initial_inventory if t == 0 else inv_p[t - 1, s]
            m.addConstr(
                prev_inv_p + prod[t, s] == inv_p[t, s] + gp.quicksum(ship_pw[w, t, s] for w in range(W)),
                name=f"bal_p_t{t}_s{s}"
            )

            # Warehouse balances
            for w in range(W):
                prev_inv_w = initial_inventory if t == 0 else inv_w[w, t - 1, s]
                m.addConstr(
                    prev_inv_w + ship_pw[w, t, s]
                    == inv_w[w, t, s] + gp.quicksum(ship_wr[r, t, s] for r in retailers_by_w[w]),
                    name=f"bal_w{w}_t{t}_s{s}"
                )

            # Retailer balances
            for r in range(R):
                prev_inv_r = initial_inventory if t == 0 else inv_r[r, t - 1, s]
                m.addConstr(
                    prev_inv_r + ship_wr[r, t, s] == inv_r[r, t, s] + demands[s][r][t],
                    name=f"bal_r{r}_t{t}_s{s}"
                )

            # Setup-linking constraints
            m.addConstr(
                prod[t, s] <= total_demand_s[s] * y_plant[t],
                name=f"setup_prod_t{t}_s{s}"
            )

            for w in range(W):
                m.addConstr(
                    ship_pw[w, t, s] <= total_demand_ww_s[s][w] * y_wh[w, t],
                    name=f"setup_wh_{w}_t{t}_s{s}"
                )

            for r in range(R):
                m.addConstr(
                    ship_wr[r, t, s] <= total_demand_wr_s[s][r] * y_ret[r, t],
                    name=f"setup_ret_{r}_t{t}_s{s}"
                )

    # Objective: setup costs + expected holding costs
    setup_cost = (
        gp.quicksum(setup_costs_plant[t] * y_plant[t] for t in range(T))
        + gp.quicksum(setup_costs_warehouses[w][t] * y_wh[w, t] for w in range(W) for t in range(T))
        + gp.quicksum(setup_costs_retailers[r][t] * y_ret[r, t] for r in range(R) for t in range(T))
    )

    expected_holding = gp.quicksum(
        scenario_probabilities[s] * (
            gp.quicksum(
                holding_cost_plant * inv_p[t, s]
                + gp.quicksum(holding_cost_warehouse * inv_w[w, t, s] for w in range(W))
                + gp.quicksum(holding_costs_retailers[r] * inv_r[r, t, s] for r in range(R))
                for t in range(T)
            )
        )
        for s in range(S)
    )

    m.setObjective(setup_cost + expected_holding, GRB.MINIMIZE)

    # Incumbent logging callback
    best_logged = {"obj": float("inf")}

    def cb(model, where):
        if where == GRB.Callback.MIPSOL and logger is not None:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj + 1e-9 < best_logged["obj"]:
                best_logged["obj"] = obj
                logger.log(float(obj))

    m.optimize(cb)

    # -----------------------------
    # Build output
    # -----------------------------
    def val(x):
        v = x.X
        if abs(v) < 1e-9:
            return 0.0
        return float(v)

    solution = {
        "objective_value": None,
        "setup_variables": {},
        "production_plant": {},
        "delivery_warehouse": {},
        "delivery_retailer": {},
        "inventory_plant": {},
        "inventory_warehouse": {},
        "inventory_retailer": {},
    }

    if m.SolCount == 0:
        # No feasible solution found
        solution["objective_value"] = float("inf")
        return solution

    solution["objective_value"] = float(m.ObjVal)

    # setup_variables: facility index convention
    # 0 = plant
    # 1..W = warehouses
    # W+1 .. W+R = retailers
    for t in range(T):
        solution["setup_variables"][f"y_0_{t}"] = int(round(y_plant[t].X))

    for w in range(W):
        fac = 1 + w
        for t in range(T):
            solution["setup_variables"][f"y_{fac}_{t}"] = int(round(y_wh[w, t].X))

    for r in range(R):
        fac = 1 + W + r
        for t in range(T):
            solution["setup_variables"][f"y_{fac}_{t}"] = int(round(y_ret[r, t].X))

    # scenario-dependent quantities
    for s in range(S):
        for t in range(T):
            solution["production_plant"][f"{t}_{s}"] = val(prod[t, s])
            solution["inventory_plant"][f"{t}_{s}"] = val(inv_p[t, s])

        for w in range(W):
            for t in range(T):
                solution["delivery_warehouse"][f"{w}_{t}_{s}"] = val(ship_pw[w, t, s])
                solution["inventory_warehouse"][f"{w}_{t}_{s}"] = val(inv_w[w, t, s])

        for r in range(R):
            for t in range(T):
                solution["delivery_retailer"][f"{r}_{t}_{s}"] = val(ship_wr[r, t, s])
                solution["inventory_retailer"][f"{r}_{t}_{s}"] = val(inv_r[r, t, s])

    return solution


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    solution = solve(instance, args.time_limit, logger)

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()
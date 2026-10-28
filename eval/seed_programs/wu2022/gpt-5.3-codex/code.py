import argparse
import json
from typing import Dict, List, Tuple

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def build_and_solve(instance: dict, time_limit: int, logger: SolutionLogger = None):
    dims = instance["dimensions"]
    I = dims["num_items"]
    T = dims["num_periods"]
    P = dims["num_plants"]
    C = dims["num_customers"]

    plant_capabilities: List[List[int]] = instance["plant_capabilities"]
    demand = instance["demand"]
    setup_cost = instance["setup_cost"]
    production_cost = instance["production_cost"]
    holding_cost = instance["holding_cost"]
    setup_time = instance["setup_time"]
    production_time = instance["production_time"]
    capacity = instance["capacity"]
    transportation_cost = instance["transportation_cost"]

    if "plant_opening_cost" in instance and instance["plant_opening_cost"] is not None:
        plant_opening_cost = instance["plant_opening_cost"]
    else:
        # Fallback if only per-period opening cost is present
        plant_opening_cost = [sum(instance["plant_opening_cost_per_period"][p]) for p in range(P)]

    # Aggregate demand by item-period and by item total
    demand_it = [[0.0 for _ in range(T)] for _ in range(I)]
    total_demand_i = [0.0 for _ in range(I)]
    for i in range(I):
        for c in range(C):
            for t in range(T):
                d = demand[i][c][t]
                demand_it[i][t] += d
                total_demand_i[i] += d

    # Aggregate transportation cost into assignment-dependent term:
    # if item i is assigned to plant p, all demand goes from p to each customer c in each period t
    trans_cost_ip = [[0.0 for _ in range(P)] for _ in range(I)]
    for i in range(I):
        for p in range(P):
            s = 0.0
            for c in range(C):
                for t in range(T):
                    s += demand[i][c][t] * transportation_cost[i][c][p][t]
            trans_cost_ip[i][p] = s

    # Capability helpers
    capable_set = [set(plant_capabilities[i]) for i in range(I)]
    items_by_plant = [[] for _ in range(P)]
    for i in range(I):
        for p in plant_capabilities[i]:
            items_by_plant[p].append(i)

    m = gp.Model("two_echelon_single_source_lot_sizing")
    m.Params.OutputFlag = 0
    m.Params.TimeLimit = int(time_limit)
    m.Params.Threads = 1

    # Decision variables
    Z = {p: m.addVar(vtype=GRB.BINARY, name=f"Z_{p}") for p in range(P)}

    U = {}
    Y = {}
    X = {}
    Inv = {}

    for i in range(I):
        for p in plant_capabilities[i]:
            U[(i, p)] = m.addVar(vtype=GRB.BINARY, name=f"U_{i}_{p}")
            for t in range(T):
                Y[(i, p, t)] = m.addVar(vtype=GRB.BINARY, name=f"Y_{i}_{p}_{t}")
                X[(i, p, t)] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"X_{i}_{p}_{t}")
                Inv[(i, p, t)] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"Inv_{i}_{p}_{t}")

    # Constraints

    # Single-sourcing: exactly one capable plant per item
    for i in range(I):
        m.addConstr(gp.quicksum(U[(i, p)] for p in plant_capabilities[i]) == 1, name=f"assign_{i}")

    # Assignment implies plant open
    for i in range(I):
        for p in plant_capabilities[i]:
            m.addConstr(U[(i, p)] <= Z[p], name=f"open_link_{i}_{p}")

    # Setup only if assigned, and production only if setup
    for i in range(I):
        M_i = total_demand_i[i]  # tight enough and safe
        for p in plant_capabilities[i]:
            for t in range(T):
                m.addConstr(Y[(i, p, t)] <= U[(i, p)], name=f"setup_link_{i}_{p}_{t}")
                m.addConstr(X[(i, p, t)] <= M_i * Y[(i, p, t)], name=f"prod_setup_link_{i}_{p}_{t}")

    # Inventory flow for each (item, capable plant)
    for i in range(I):
        for p in plant_capabilities[i]:
            # t = 0
            m.addConstr(
                X[(i, p, 0)] - demand_it[i][0] * U[(i, p)] == Inv[(i, p, 0)],
                name=f"inv_bal_{i}_{p}_0",
            )
            # t >= 1
            for t in range(1, T):
                m.addConstr(
                    Inv[(i, p, t - 1)] + X[(i, p, t)] - demand_it[i][t] * U[(i, p)] == Inv[(i, p, t)],
                    name=f"inv_bal_{i}_{p}_{t}",
                )

    # Plant-period capacity
    for p in range(P):
        for t in range(T):
            lhs = gp.quicksum(
                setup_time[i][p][t] * Y[(i, p, t)] + production_time[i][p][t] * X[(i, p, t)]
                for i in items_by_plant[p]
            )
            m.addConstr(lhs <= capacity[p][t] * Z[p], name=f"cap_{p}_{t}")

    # Objective
    obj = gp.LinExpr()

    # Plant opening
    obj += gp.quicksum(plant_opening_cost[p] * Z[p] for p in range(P))

    # Assignment-driven transportation
    obj += gp.quicksum(trans_cost_ip[i][p] * U[(i, p)] for i in range(I) for p in plant_capabilities[i])

    # Setup, production, holding
    obj += gp.quicksum(
        setup_cost[i][p][t] * Y[(i, p, t)]
        + production_cost[i][p][t] * X[(i, p, t)]
        + holding_cost[i][p][t] * Inv[(i, p, t)]
        for i in range(I)
        for p in plant_capabilities[i]
        for t in range(T)
    )

    m.setObjective(obj, GRB.MINIMIZE)

    # Incumbent logging callback
    m._logger = logger
    m._best_logged = float("inf")

    def cb(model, where):
        if where == GRB.Callback.MIPSOL:
            val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if val + 1e-9 < model._best_logged:
                model._best_logged = val
                if model._logger is not None:
                    model._logger.log(float(val))

    m.optimize(cb)

    has_sol = m.SolCount > 0

    if has_sol and logger is not None:
        final_obj = float(m.ObjVal)
        if final_obj + 1e-9 < m._best_logged:
            logger.log(final_obj)

    # Build output dictionaries with required full key formats
    Z_out = {}
    for p in range(P):
        Z_out[str(p)] = int(round(Z[p].X)) if has_sol else 0

    U_out = {}
    for i in range(I):
        for p in range(P):
            key = f"{i}_{p}"
            if (i, p) in U and has_sol:
                U_out[key] = int(round(U[(i, p)].X))
            else:
                U_out[key] = 0

    Y_out = {}
    for i in range(I):
        for p in range(P):
            for t in range(T):
                key = f"{i}_{p}_{t}"
                if (i, p, t) in Y and has_sol:
                    Y_out[key] = int(round(Y[(i, p, t)].X))
                else:
                    Y_out[key] = 0

    objective_value = float(m.ObjVal) if has_sol else 1e30

    return {
        "objective_value": objective_value,
        "Z": Z_out,
        "U": U_out,
        "Y": Y_out,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    solution = build_and_solve(instance, args.time_limit, logger)

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()
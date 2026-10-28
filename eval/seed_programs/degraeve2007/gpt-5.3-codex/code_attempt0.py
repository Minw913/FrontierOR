import argparse
import json
import math
from typing import Any, Dict, List, Optional, Tuple

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def compute_tail_demands(demand: List[List[float]]) -> List[List[float]]:
    p_count = len(demand)
    t_count = len(demand[0]) if p_count > 0 else 0
    tail = [[0.0] * t_count for _ in range(p_count)]
    for p in range(p_count):
        running = 0.0
        for t in range(t_count - 1, -1, -1):
            running += demand[p][t]
            tail[p][t] = running
    return tail


def build_fallback_solution(data: Dict[str, Any]) -> Dict[str, Any]:
    demand = data["parameters"]["demand"]
    holding_cost = data["parameters"]["holding_cost"]
    setup_cost = data["parameters"]["setup_cost"]
    variable_cost = data["parameters"]["variable_production_cost"]
    initial_inventory_cost = data["parameters"]["initial_inventory_cost"]

    P = len(demand)
    T = len(demand[0]) if P > 0 else 0

    production = [[0.0 for _ in range(T)] for _ in range(P)]
    setup = [[0 for _ in range(T)] for _ in range(P)]
    inventory = [[0.0 for _ in range(T)] for _ in range(P)]
    initial_inventory = [0.0 for _ in range(P)]

    for p in range(P):
        total_d = float(sum(demand[p]))
        initial_inventory[p] = total_d
        remaining = total_d
        for t in range(T):
            remaining -= float(demand[p][t])
            inventory[p][t] = max(0.0, remaining)

    obj = 0.0
    for p in range(P):
        obj += float(initial_inventory_cost[p]) * initial_inventory[p]
        for t in range(T):
            obj += float(setup_cost[p][t]) * setup[p][t]
            obj += float(variable_cost[p][t]) * production[p][t]
            obj += float(holding_cost[p][t]) * inventory[p][t]

    return {
        "objective_value": float(obj),
        "variables": {
            "production": production,
            "setup": setup,
            "inventory": inventory,
            "initial_inventory": initial_inventory,
        },
    }


def solve_instance(
    data: Dict[str, Any],
    time_limit: int,
    logger: Optional[SolutionLogger] = None,
) -> Dict[str, Any]:
    params = data["parameters"]
    demand = params["demand"]
    holding_cost = params["holding_cost"]
    setup_cost = params["setup_cost"]
    variable_cost = params["variable_production_cost"]
    variable_time = params["variable_production_time"]
    setup_time = params["setup_time"]
    capacity = params["capacity"]
    initial_inventory_cost = params["initial_inventory_cost"]

    P = len(demand)
    T = len(demand[0]) if P > 0 else 0

    tail_demand = compute_tail_demands(demand)

    m = gp.Model("multi_item_capacitated_lot_sizing")
    m.Params.TimeLimit = max(1, int(time_limit))
    m.Params.Threads = 1
    m.Params.OutputFlag = 0
    m.Params.MIPFocus = 1

    x = m.addVars(P, T, lb=0.0, vtype=GRB.CONTINUOUS, name="production")
    y = m.addVars(P, T, vtype=GRB.BINARY, name="setup")
    inv = m.addVars(P, T, lb=0.0, vtype=GRB.CONTINUOUS, name="inventory")
    init_inv = m.addVars(P, lb=0.0, vtype=GRB.CONTINUOUS, name="initial_inventory")

    # Inventory balance
    for p in range(P):
        if T > 0:
            m.addConstr(
                init_inv[p] + x[p, 0] == float(demand[p][0]) + inv[p, 0],
                name=f"balance_p{p}_t0",
            )
        for t in range(1, T):
            m.addConstr(
                inv[p, t - 1] + x[p, t] == float(demand[p][t]) + inv[p, t],
                name=f"balance_p{p}_t{t}",
            )

    # Capacity constraints
    for t in range(T):
        m.addConstr(
            gp.quicksum(
                float(setup_time[p][t]) * y[p, t] + float(variable_time[p][t]) * x[p, t]
                for p in range(P)
            )
            <= float(capacity[t]),
            name=f"capacity_t{t}",
        )

    # Production-linking upper bounds: x[p,t] <= M[p,t] * y[p,t]
    # M is min(cumulative remaining demand, (capacity_t - setup_time)/variable_time)
    for p in range(P):
        for t in range(T):
            cum_dem = float(tail_demand[p][t])
            vt = float(variable_time[p][t])
            st = float(setup_time[p][t])
            cap_t = float(capacity[t])

            if vt <= 1e-12:
                cap_based = cum_dem
            else:
                cap_based = (cap_t - st) / vt

            M = min(cum_dem, max(0.0, cap_based))
            m.addConstr(x[p, t] <= M * y[p, t], name=f"link_p{p}_t{t}")
            m.addConstr(x[p, t] <= cum_dem, name=f"cumdem_p{p}_t{t}")

    obj = gp.quicksum(float(initial_inventory_cost[p]) * init_inv[p] for p in range(P))
    obj += gp.quicksum(float(setup_cost[p][t]) * y[p, t] for p in range(P) for t in range(T))
    obj += gp.quicksum(float(variable_cost[p][t]) * x[p, t] for p in range(P) for t in range(T))
    obj += gp.quicksum(float(holding_cost[p][t]) * inv[p, t] for p in range(P) for t in range(T))
    m.setObjective(obj, GRB.MINIMIZE)

    # Feasible warm start: satisfy all demand using initial inventory only.
    start_initial = [float(sum(demand[p])) for p in range(P)]
    start_obj = 0.0
    for p in range(P):
        init_inv[p].Start = start_initial[p]
        start_obj += float(initial_inventory_cost[p]) * start_initial[p]
        remaining = start_initial[p]
        for t in range(T):
            y[p, t].Start = 0.0
            x[p, t].Start = 0.0
            remaining -= float(demand[p][t])
            inv_val = max(0.0, remaining)
            inv[p, t].Start = inv_val
            start_obj += float(holding_cost[p][t]) * inv_val

    best_incumbent = [math.inf]
    if logger is not None:
        logger.log(float(start_obj))
        best_incumbent[0] = float(start_obj)

    def cb(model, where):
        if logger is None:
            return
        if where == GRB.Callback.MIPSOL:
            try:
                val = float(model.cbGet(GRB.Callback.MIPSOL_OBJ))
                if val + 1e-9 < best_incumbent[0]:
                    best_incumbent[0] = val
                    logger.log(val)
            except gp.GurobiError:
                pass

    m.optimize(cb)

    if m.SolCount <= 0:
        return build_fallback_solution(data)

    production = [[0.0 for _ in range(T)] for _ in range(P)]
    setup = [[0 for _ in range(T)] for _ in range(P)]
    inventory = [[0.0 for _ in range(T)] for _ in range(P)]
    initial_inventory = [0.0 for _ in range(P)]

    for p in range(P):
        initial_inventory[p] = float(init_inv[p].X)
        for t in range(T):
            production[p][t] = float(x[p, t].X)
            setup[p][t] = int(round(float(y[p, t].X)))
            inventory[p][t] = float(inv[p, t].X)

    solution = {
        "objective_value": float(m.ObjVal),
        "variables": {
            "production": production,
            "setup": setup,
            "inventory": inventory,
            "initial_inventory": initial_inventory,
        },
    }
    return solution


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, required=False, default=None)
    args = parser.parse_args()

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    try:
        solution = solve_instance(data, args.time_limit, logger)
    except Exception:
        # Guaranteed fallback output
        solution = build_fallback_solution(data)

    with open(args.solution_path, "w") as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()
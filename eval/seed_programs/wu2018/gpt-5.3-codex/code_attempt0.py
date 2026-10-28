import argparse
import json
from typing import Dict, List, Tuple

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def get_param(param_dict: Dict[str, float], i: int, m: int, name: str) -> float:
    key = f"item_{i}_machine_{m}"
    if key not in param_dict:
        raise KeyError(f"Missing key '{key}' in '{name}'.")
    return float(param_dict[key])


def build_and_solve(instance: Dict, time_limit: int, logger: SolutionLogger = None):
    # -----------------------------
    # Read instance
    # -----------------------------
    n_items = int(instance["num_items"])
    n_periods = int(instance["num_periods"])
    n_machines = int(instance["num_machines"])

    demand: List[List[float]] = instance["demand"]
    capacity: List[List[float]] = instance["capacity"]
    machine_capabilities: List[List[int]] = instance["machine_capabilities"]
    max_machines_item = instance["max_machines_per_item_per_period"]

    production_time_dict = instance["production_time"]
    setup_time_dict = instance["setup_time"]
    production_cost_dict = instance["production_cost"]
    setup_cost_dict = instance["setup_cost"]
    holding_cost = instance["holding_cost"]

    initial_inventory = instance.get("initial_inventory", [0.0] * n_items)
    if len(initial_inventory) != n_items:
        raise ValueError("initial_inventory must have length num_items if provided.")

    # sanitize capabilities
    M_i = {}
    for i in range(1, n_items + 1):
        caps = sorted(set(int(m) for m in machine_capabilities[i - 1]))
        for m in caps:
            if m < 1 or m > n_machines:
                raise ValueError(f"Invalid machine id {m} for item {i}.")
        M_i[i] = caps

    # items that each machine can produce
    I_m = {m: [] for m in range(1, n_machines + 1)}
    for i in range(1, n_items + 1):
        for m in M_i[i]:
            I_m[m].append(i)

    # remaining demand for tighter bounds
    rem_demand = [[0.0] * (n_periods + 2) for _ in range(n_items + 1)]  # 1-indexed
    for i in range(1, n_items + 1):
        running = 0.0
        for t in range(n_periods, 0, -1):
            running += float(demand[i - 1][t - 1])
            rem_demand[i][t] = running

    # -----------------------------
    # Build model
    # -----------------------------
    model = gp.Model("multi_item_multi_machine_lot_sizing")
    model.Params.TimeLimit = float(time_limit)
    model.Params.Threads = 1
    model.Params.OutputFlag = 0

    # Variables
    x = {}  # production
    y = {}  # setup
    inv = {}  # inventory

    # Inventory variables
    for i in range(1, n_items + 1):
        for t in range(1, n_periods + 1):
            inv[i, t] = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"I_item{i}_period{t}")

    # Production/setup variables with tight UB
    for i in range(1, n_items + 1):
        for m in M_i[i]:
            pt = get_param(production_time_dict, i, m, "production_time")
            st = get_param(setup_time_dict, i, m, "setup_time")
            if pt <= 0:
                raise ValueError(f"production_time must be > 0 for item {i}, machine {m}.")
            for t in range(1, n_periods + 1):
                cap_mt = float(capacity[m - 1][t - 1])
                rem = rem_demand[i][t]
                u_cap = max(0.0, (cap_mt - st) / pt)
                ub = min(rem, u_cap)

                y_ub = 1.0 if ub > 1e-12 else 0.0
                y[i, m, t] = model.addVar(lb=0.0, ub=y_ub, vtype=GRB.BINARY,
                                          name=f"Y_item{i}_machine{m}_period{t}")
                x[i, m, t] = model.addVar(lb=0.0, ub=ub, vtype=GRB.CONTINUOUS,
                                          name=f"X_item{i}_machine{m}_period{t}")

    model.update()

    # -----------------------------
    # Constraints
    # -----------------------------

    # Flow balance
    for i in range(1, n_items + 1):
        init_i = float(initial_inventory[i - 1])
        for t in range(1, n_periods + 1):
            prod_sum = gp.quicksum(x[i, m, t] for m in M_i[i])
            prev_inv = init_i if t == 1 else inv[i, t - 1]
            model.addConstr(
                prod_sum + prev_inv - inv[i, t] == float(demand[i - 1][t - 1]),
                name=f"balance_item{i}_period{t}"
            )

    # Capacity per machine/period
    for m in range(1, n_machines + 1):
        for t in range(1, n_periods + 1):
            expr = gp.LinExpr()
            for i in I_m[m]:
                pt = get_param(production_time_dict, i, m, "production_time")
                st = get_param(setup_time_dict, i, m, "setup_time")
                expr += pt * x[i, m, t] + st * y[i, m, t]
            model.addConstr(
                expr <= float(capacity[m - 1][t - 1]),
                name=f"capacity_machine{m}_period{t}"
            )

    # Linking x and y
    for (i, m, t), xv in x.items():
        ub = xv.UB
        model.addConstr(xv <= ub * y[i, m, t], name=f"link_item{i}_machine{m}_period{t}")

    # Max machines per item/period
    for i in range(1, n_items + 1):
        kmax = int(max_machines_item[i - 1])
        for t in range(1, n_periods + 1):
            model.addConstr(
                gp.quicksum(y[i, m, t] for m in M_i[i]) <= kmax,
                name=f"max_machines_item{i}_period{t}"
            )

    # Objective
    obj_setup = gp.quicksum(
        get_param(setup_cost_dict, i, m, "setup_cost") * y[i, m, t]
        for (i, m, t) in y
    )
    obj_prod = gp.quicksum(
        get_param(production_cost_dict, i, m, "production_cost") * x[i, m, t]
        for (i, m, t) in x
    )
    obj_hold = gp.quicksum(
        float(holding_cost[i - 1]) * inv[i, t]
        for i in range(1, n_items + 1)
        for t in range(1, n_periods + 1)
    )

    model.setObjective(obj_setup + obj_prod + obj_hold, GRB.MINIMIZE)

    # Incumbent logger callback
    model._logger = logger
    model._best_obj_logged = float("inf")

    def mip_callback(cb_model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = cb_model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj_val + 1e-9 < cb_model._best_obj_logged:
                cb_model._best_obj_logged = obj_val
                if cb_model._logger is not None:
                    cb_model._logger.log(float(obj_val))

    model.optimize(mip_callback)

    return model, x, y, inv, M_i


def write_solution(
    solution_path: str,
    model: gp.Model,
    x: Dict[Tuple[int, int, int], gp.Var],
    y: Dict[Tuple[int, int, int], gp.Var],
    inv: Dict[Tuple[int, int], gp.Var],
    M_i: Dict[int, List[int]],
    n_items: int,
    n_periods: int,
):
    has_sol = model.SolCount > 0

    production = {}
    setups = {}
    inventory = {}

    # Fill production + setup (capable arcs only)
    for i in range(1, n_items + 1):
        for m in M_i[i]:
            for t in range(1, n_periods + 1):
                x_key = f"X_item{i}_machine{m}_period{t}"
                y_key = f"Y_item{i}_machine{m}_period{t}"
                if has_sol:
                    xv = max(0.0, float(x[i, m, t].X))
                    yv = 1 if float(y[i, m, t].X) > 0.5 else 0
                else:
                    xv = 0.0
                    yv = 0
                production[x_key] = xv
                setups[y_key] = yv

    # Fill inventory
    for i in range(1, n_items + 1):
        for t in range(1, n_periods + 1):
            key = f"I_item{i}_period{t}"
            if has_sol:
                iv = max(0.0, float(inv[i, t].X))
            else:
                iv = 0.0
            inventory[key] = iv

    objective_value = float(model.ObjVal) if has_sol else 1e100

    out = {
        "objective_value": objective_value,
        "production": production,
        "inventory": inventory,
        "setups": setups,
    }

    with open(solution_path, "w") as f:
        json.dump(out, f)


def main():
    parser = argparse.ArgumentParser(description="Solve multi-item multi-machine lot-sizing with setups.")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to input instance JSON.")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to output solution JSON.")
    parser.add_argument("--time_limit", type=int, required=True, help="Max runtime in seconds.")
    parser.add_argument("--log_path", type=str, default=None, help="Optional path to JSONL convergence log.")
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    model, x, y, inv, M_i = build_and_solve(instance, args.time_limit, logger)

    write_solution(
        args.solution_path,
        model,
        x,
        y,
        inv,
        M_i,
        n_items=int(instance["num_items"]),
        n_periods=int(instance["num_periods"]),
    )


if __name__ == "__main__":
    main()
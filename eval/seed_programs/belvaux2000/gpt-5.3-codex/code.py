import argparse
import json
from typing import Dict, Any

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def solve_instance(instance: Dict[str, Any], time_limit: int, logger: SolutionLogger = None) -> Dict[str, Any]:
    # ---------------------------
    # Read instance data
    # ---------------------------
    NI = int(instance["parameters"]["NI"])
    NK = int(instance["parameters"]["NK"])
    NT = int(instance["parameters"]["NT"])

    f = float(instance["costs"]["setup_cost_f"])
    g = float(instance["costs"]["startup_cost_g"])
    h = [float(v) for v in instance["costs"]["holding_cost_h"]]
    e = [float(v) for v in instance["costs"]["backlogging_cost_e"]]

    C = [float(v) for v in instance["machine_data"]["capacity_C"]]
    sigma = [float(v) for v in instance["machine_data"]["startup_time_sigma"]]

    demand = instance["demand"]
    s0 = [float(v) for v in instance["initial_conditions"]["initial_stock_s0"]]
    r0 = [float(v) for v in instance["initial_conditions"]["initial_backlog_r0"]]

    I = range(NI)
    K = range(NK)
    T = range(NT)

    # ---------------------------
    # Build model
    # ---------------------------
    model = gp.Model("lot_sizing_multi_item_multi_machine")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = int(time_limit)
    model.Params.Threads = 1

    # Decision variables
    x = model.addVars(NI, NK, NT, vtype=GRB.CONTINUOUS, lb=0.0, name="production")
    s = model.addVars(NI, NT, vtype=GRB.CONTINUOUS, lb=0.0, name="stock")
    r = model.addVars(NI, NT, vtype=GRB.CONTINUOUS, lb=0.0, name="backlog")
    y = model.addVars(NI, NK, NT, vtype=GRB.BINARY, name="setup")
    z = model.addVars(NI, NK, NT, vtype=GRB.BINARY, name="startup")

    # Inventory/backlog flow balance
    for i in I:
        for t in T:
            prev_s = s0[i] if t == 0 else s[i, t - 1]
            prev_r = r0[i] if t == 0 else r[i, t - 1]
            model.addConstr(
                prev_s + gp.quicksum(x[i, k, t] for k in K) - prev_r
                == float(demand[i][t]) + s[i, t] - r[i, t],
                name=f"balance_{i}_{t}",
            )

    # Machine setup and capacity constraints
    for k in K:
        for t in T:
            # At most one item setup per machine-period
            model.addConstr(gp.quicksum(y[i, k, t] for i in I) <= 1, name=f"one_item_{k}_{t}")

            # Capacity with startup capacity loss
            model.addConstr(
                gp.quicksum(x[i, k, t] for i in I) + sigma[k] * gp.quicksum(z[i, k, t] for i in I) <= C[k],
                name=f"cap_{k}_{t}",
            )

            for i in I:
                # Production only if setup
                model.addConstr(x[i, k, t] <= C[k] * y[i, k, t], name=f"link_prod_setup_{i}_{k}_{t}")

    # Startup logic
    for i in I:
        for k in K:
            # No setup before first period -> startup iff setup in period 0
            model.addConstr(z[i, k, 0] == y[i, k, 0], name=f"startup_init_{i}_{k}")
            for t in range(1, NT):
                model.addConstr(z[i, k, t] >= y[i, k, t] - y[i, k, t - 1], name=f"startup_lb_{i}_{k}_{t}")
                model.addConstr(z[i, k, t] <= y[i, k, t], name=f"startup_ub1_{i}_{k}_{t}")
                model.addConstr(z[i, k, t] <= 1 - y[i, k, t - 1], name=f"startup_ub2_{i}_{k}_{t}")

    # Objective
    obj_setup = f * gp.quicksum(y[i, k, t] for i in I for k in K for t in T)
    obj_startup = g * gp.quicksum(z[i, k, t] for i in I for k in K for t in T)
    obj_hold = gp.quicksum(h[i] * s[i, t] for i in I for t in T)
    obj_backlog = gp.quicksum(e[i] * r[i, t] for i in I for t in T)
    model.setObjective(obj_setup + obj_startup + obj_hold + obj_backlog, GRB.MINIMIZE)

    # ---------------------------
    # Incumbent logging callback
    # ---------------------------
    best_logged = {"obj": float("inf")}

    def callback(cb_model, where):
        if where == GRB.Callback.MIPSOL and logger is not None:
            obj_val = cb_model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj_val + 1e-9 < best_logged["obj"]:
                best_logged["obj"] = float(obj_val)
                logger.log(float(obj_val))

    # ---------------------------
    # Optimize
    # ---------------------------
    model.optimize(callback)

    # ---------------------------
    # Extract best feasible solution
    # ---------------------------
    if model.SolCount == 0:
        # Should rarely happen; return schema-compliant empty solution
        return {
            "objective_value": float("inf"),
            "production": {},
            "stock": {},
            "backlog": {},
            "setup": {},
            "startup": {},
        }

    def val(v):
        vv = v.X
        if abs(vv) < 1e-9:
            return 0.0
        return float(vv)

    production = {}
    setup = {}
    startup = {}
    stock = {}
    backlog = {}

    for i in I:
        for t in T:
            stock[f"{i}_{t}"] = val(s[i, t])
            backlog[f"{i}_{t}"] = val(r[i, t])
        for k in K:
            for t in T:
                production[f"{i}_{k}_{t}"] = val(x[i, k, t])
                setup[f"{i}_{k}_{t}"] = 1 if y[i, k, t].X > 0.5 else 0
                startup[f"{i}_{k}_{t}"] = 1 if z[i, k, t].X > 0.5 else 0

    solution = {
        "objective_value": float(model.ObjVal),
        "production": production,
        "stock": stock,
        "backlog": backlog,
        "setup": setup,
        "startup": startup,
    }
    return solution


def main():
    parser = argparse.ArgumentParser(description="Solve multi-item multi-machine lot-sizing with setup/startup decisions.")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to input instance JSON")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to output solution JSON")
    parser.add_argument("--time_limit", type=int, required=True, help="Maximum runtime in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Optional path to JSONL convergence log")
    args = parser.parse_args()

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    solution = solve_instance(instance, args.time_limit, logger=logger)

    with open(args.solution_path, "w") as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()
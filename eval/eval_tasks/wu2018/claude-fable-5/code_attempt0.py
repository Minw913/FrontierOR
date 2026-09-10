import argparse
import json
import math
import time

import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except Exception:
    SolutionLogger = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, default=300)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()

    logger = None
    if args.log_path and SolutionLogger is not None:
        logger = SolutionLogger(args.log_path, sense="minimize")

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    N = data["num_items"]
    T = data["num_periods"]
    M = data["num_machines"]
    max_mach = data["max_machines_per_item_per_period"]
    caps_list = data["machine_capabilities"]  # 1-indexed machine ids per item
    demand = data["demand"]  # [item][period]
    prod_time = data["production_time"]
    hold_cost = data["holding_cost"]
    setup_cost = data["setup_cost"]
    setup_time = data["setup_time"]
    prod_cost = data["production_cost"]
    capacity = data["capacity"]  # [machine][period]
    init_inv = data.get("initial_inventory", [0.0] * N)
    if init_inv is None:
        init_inv = [0.0] * N

    def key(i, m):
        # i, m are 1-indexed
        return f"item_{i}_machine_{m}"

    # Pairs of (item, machine), 1-indexed
    pairs = []
    for i in range(1, N + 1):
        for m in caps_list[i - 1]:
            pairs.append((i, m))

    # Remaining demand from period t to end, adjusted for initial inventory usage
    rem_demand = {}  # (i, t) 1-indexed
    for i in range(1, N + 1):
        tot = 0.0
        for t in range(T, 0, -1):
            tot += demand[i - 1][t - 1]
            rem_demand[(i, t)] = tot

    model = gp.Model("CLSP_PM")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    elapsed = time.time() - start_time
    model.Params.TimeLimit = max(1.0, args.time_limit - elapsed - 3.0)

    # Variables
    X = {}
    Y = {}
    for (i, m) in pairs:
        k = key(i, m)
        pt = prod_time[k]
        st = setup_time[k]
        for t in range(1, T + 1):
            cap = capacity[m - 1][t - 1]
            if pt > 1e-12:
                cap_bound = max(0.0, (cap - st) / pt)
            else:
                cap_bound = float("inf")
            ub = min(rem_demand[(i, t)], cap_bound)
            ub = max(0.0, ub)
            X[(i, m, t)] = model.addVar(lb=0.0, ub=ub, obj=prod_cost[k],
                                        name=f"X_{i}_{m}_{t}")
            Y[(i, m, t)] = model.addVar(vtype=GRB.BINARY, obj=setup_cost[k],
                                        name=f"Y_{i}_{m}_{t}")

    I = {}
    for i in range(1, N + 1):
        for t in range(1, T + 1):
            I[(i, t)] = model.addVar(lb=0.0, obj=hold_cost[i - 1],
                                     name=f"I_{i}_{t}")

    model.ModelSense = GRB.MINIMIZE
    model.update()

    # Linking: X <= UB * Y
    for (i, m) in pairs:
        for t in range(1, T + 1):
            ub = X[(i, m, t)].UB
            if ub <= 1e-12:
                model.addConstr(X[(i, m, t)] == 0)
                model.addConstr(Y[(i, m, t)] == 0)
            else:
                model.addConstr(X[(i, m, t)] <= ub * Y[(i, m, t)])

    # Capacity per machine per period
    items_on_machine = {m: [] for m in range(1, M + 1)}
    for (i, m) in pairs:
        items_on_machine[m].append(i)
    for m in range(1, M + 1):
        for t in range(1, T + 1):
            expr = gp.LinExpr()
            for i in items_on_machine[m]:
                k = key(i, m)
                expr += prod_time[k] * X[(i, m, t)] + setup_time[k] * Y[(i, m, t)]
            model.addConstr(expr <= capacity[m - 1][t - 1],
                            name=f"cap_{m}_{t}")

    # Flow balance
    for i in range(1, N + 1):
        for t in range(1, T + 1):
            prev = init_inv[i - 1] if t == 1 else I[(i, t - 1)]
            model.addConstr(
                gp.quicksum(X[(i, m, t)] for m in caps_list[i - 1])
                + prev - I[(i, t)] == demand[i - 1][t - 1],
                name=f"bal_{i}_{t}")

    # Max machines per item per period
    for i in range(1, N + 1):
        for t in range(1, T + 1):
            model.addConstr(
                gp.quicksum(Y[(i, m, t)] for m in caps_list[i - 1])
                <= max_mach[i - 1], name=f"maxm_{i}_{t}")

    # Callback to log incumbents
    def build_solution(xvals, yvals, ivals, obj):
        sol = {
            "objective_value": obj,
            "production": {},
            "inventory": {},
            "setups": {},
        }
        for (i, m) in pairs:
            for t in range(1, T + 1):
                v = xvals[(i, m, t)]
                sol["production"][f"X_item{i}_machine{m}_period{t}"] = max(0.0, float(v))
                yv = 1 if yvals[(i, m, t)] > 0.5 else 0
                sol["setups"][f"Y_item{i}_machine{m}_period{t}"] = yv
        for i in range(1, N + 1):
            for t in range(1, T + 1):
                sol["inventory"][f"I_item{i}_period{t}"] = max(0.0, float(ivals[(i, t)]))
        return sol

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger is not None:
                try:
                    xvals = {kk: model.cbGetSolution(v) for kk, v in X.items()}
                    yvals = {kk: model.cbGetSolution(v) for kk, v in Y.items()}
                    ivals = {kk: model.cbGetSolution(v) for kk, v in I.items()}
                    sol = build_solution(xvals, yvals, ivals, obj)
                    logger.log_solution(obj, sol)
                except Exception:
                    try:
                        logger.log(obj)
                    except Exception:
                        pass

    model.optimize(callback)

    # Extract final solution
    solution = {
        "objective_value": float("inf"),
        "production": {},
        "inventory": {},
        "setups": {},
    }

    if model.SolCount > 0:
        xvals = {kk: v.X for kk, v in X.items()}
        yvals = {kk: v.X for kk, v in Y.items()}
        ivals = {kk: v.X for kk, v in I.items()}
        obj = model.ObjVal
        solution = build_solution(xvals, yvals, ivals, obj)
        # Recompute objective consistently with rounded setups
        total = 0.0
        for (i, m) in pairs:
            k = key(i, m)
            for t in range(1, T + 1):
                total += prod_cost[k] * solution["production"][f"X_item{i}_machine{m}_period{t}"]
                total += setup_cost[k] * solution["setups"][f"Y_item{i}_machine{m}_period{t}"]
        for i in range(1, N + 1):
            for t in range(1, T + 1):
                total += hold_cost[i - 1] * solution["inventory"][f"I_item{i}_period{t}"]
        solution["objective_value"] = total
        if logger is not None:
            try:
                logger.log_solution(total, solution)
            except Exception:
                pass

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()
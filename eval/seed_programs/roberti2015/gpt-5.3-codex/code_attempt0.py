import argparse
import json
import math
import os
import time
from typing import List, Tuple

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def read_instance(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def compute_objective(
    flow: List[List[float]],
    arcs_used: List[List[int]],
    variable_cost: List[List[float]],
    fixed_cost: List[List[float]],
) -> float:
    m = len(flow)
    n = len(flow[0]) if m > 0 else 0
    obj = 0.0
    for i in range(m):
        for j in range(n):
            obj += variable_cost[i][j] * flow[i][j] + fixed_cost[i][j] * arcs_used[i][j]
    return float(obj)


def constructive_heuristic(
    supply: List[float],
    demand: List[float],
    variable_cost: List[List[float]],
    fixed_cost: List[List[float]],
    capacity: List[List[float]],
) -> Tuple[List[List[float]], List[List[int]], float]:
    """
    Cost-aware greedy constructive solution.
    Because the network is complete and capacities are min(s_i, d_j), this always yields feasibility.
    """
    m = len(supply)
    n = len(demand)
    rem_s = [float(x) for x in supply]
    rem_d = [float(x) for x in demand]
    flow = [[0.0 for _ in range(n)] for _ in range(m)]

    # Proxy score: variable + fixed spread over potential amount
    arcs = []
    for i in range(m):
        for j in range(n):
            denom = max(1.0, min(float(supply[i]), float(demand[j])))
            score = float(variable_cost[i][j]) + float(fixed_cost[i][j]) / denom
            arcs.append((score, i, j))
    arcs.sort(key=lambda t: t[0])

    for _, i, j in arcs:
        if rem_s[i] <= 1e-12 or rem_d[j] <= 1e-12:
            continue
        q = min(rem_s[i], rem_d[j], float(capacity[i][j]))
        if q > 1e-12:
            flow[i][j] += q
            rem_s[i] -= q
            rem_d[j] -= q

    # Safety pass (should usually be unnecessary)
    for i in range(m):
        if rem_s[i] <= 1e-10:
            continue
        for j in range(n):
            if rem_d[j] <= 1e-10:
                continue
            residual_cap = float(capacity[i][j]) - flow[i][j]
            if residual_cap <= 1e-12:
                continue
            q = min(rem_s[i], rem_d[j], residual_cap)
            if q > 1e-12:
                flow[i][j] += q
                rem_s[i] -= q
                rem_d[j] -= q
            if rem_s[i] <= 1e-10:
                break

    # Build arc usage
    arcs_used = [[1 if flow[i][j] > 1e-9 else 0 for j in range(n)] for i in range(m)]
    obj = compute_objective(flow, arcs_used, variable_cost, fixed_cost)
    return flow, arcs_used, obj


def build_and_solve_mip(
    data: dict,
    time_limit: int,
    logger: SolutionLogger = None,
):
    m = int(data["num_sources"])
    n = int(data["num_sinks"])
    supply = [float(x) for x in data["supply"]]
    demand = [float(x) for x in data["demand"]]
    fixed_cost = data["fixed_cost"]
    variable_cost = data["variable_cost"]
    capacity = data["capacity"]

    # Initial feasible solution (also fallback if solver times out too early)
    h_flow, h_used, h_obj = constructive_heuristic(
        supply, demand, variable_cost, fixed_cost, capacity
    )

    if logger:
        logger.log(float(h_obj))

    # If no time, return heuristic immediately
    if time_limit <= 0:
        return h_flow, h_used, h_obj

    model = gp.Model("fixed_charge_transport")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = float(time_limit)

    # Decision variables
    x = model.addVars(m, n, lb=0.0, vtype=GRB.CONTINUOUS, name="x")
    y = model.addVars(m, n, vtype=GRB.BINARY, name="y")

    # Constraints
    for i in range(m):
        model.addConstr(gp.quicksum(x[i, j] for j in range(n)) == supply[i], name=f"supply_{i}")
    for j in range(n):
        model.addConstr(gp.quicksum(x[i, j] for i in range(m)) == demand[j], name=f"demand_{j}")

    for i in range(m):
        for j in range(n):
            cap = float(capacity[i][j])
            model.addConstr(x[i, j] <= cap * y[i, j], name=f"link_{i}_{j}")

    # Optional strengthening: each source and sink must use at least one arc (positive supply/demand)
    for i in range(m):
        model.addConstr(gp.quicksum(y[i, j] for j in range(n)) >= 1, name=f"use_source_{i}")
    for j in range(n):
        model.addConstr(gp.quicksum(y[i, j] for i in range(m)) >= 1, name=f"use_sink_{j}")

    # Objective
    model.setObjective(
        gp.quicksum(variable_cost[i][j] * x[i, j] + fixed_cost[i][j] * y[i, j] for i in range(m) for j in range(n)),
        GRB.MINIMIZE,
    )

    # MIP start from heuristic
    for i in range(m):
        for j in range(n):
            x[i, j].Start = float(h_flow[i][j])
            y[i, j].Start = float(h_used[i][j])

    model._logger = logger
    model._best_logged = float(h_obj)

    def incumbent_callback(cb_model, where):
        if where == GRB.Callback.MIPSOL:
            obj = cb_model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj + 1e-9 < cb_model._best_logged:
                cb_model._best_logged = float(obj)
                if cb_model._logger:
                    cb_model._logger.log(float(obj))

    model.optimize(incumbent_callback)

    # Extract best available solution
    if model.SolCount > 0:
        flow = [[0.0 for _ in range(n)] for _ in range(m)]
        arcs_used = [[0 for _ in range(n)] for _ in range(m)]
        for i in range(m):
            for j in range(n):
                xv = float(x[i, j].X)
                yv = float(y[i, j].X)
                if abs(xv) < 1e-10:
                    xv = 0.0
                flow[i][j] = xv
                arcs_used[i][j] = 1 if yv >= 0.5 or xv > 1e-9 else 0
        obj = compute_objective(flow, arcs_used, variable_cost, fixed_cost)
        return flow, arcs_used, obj

    # Fallback (should be rare)
    return h_flow, h_used, h_obj


def main():
    parser = argparse.ArgumentParser(description="Solve fixed-charge transportation problem.")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to input instance JSON.")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to output solution JSON.")
    parser.add_argument("--time_limit", type=int, required=True, help="Maximum runtime in seconds.")
    parser.add_argument("--log_path", type=str, default=None, help="Optional JSONL path for incumbent logging.")
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    data = read_instance(args.instance_path)

    start = time.time()
    flow, arcs_used, obj = build_and_solve_mip(
        data=data,
        time_limit=args.time_limit,
        logger=logger,
    )
    _ = time.time() - start  # kept for potential debugging; intentionally unused

    solution = {
        "objective_value": float(obj),
        "flow": [[float(v) for v in row] for row in flow],
        "arcs_used": [[int(v) for v in row] for row in arcs_used],
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.solution_path)), exist_ok=True)
    with open(args.solution_path, "w") as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()
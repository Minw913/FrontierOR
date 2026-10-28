import argparse
import json
from typing import Any, Dict, List, Tuple

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def build_retailers_by_warehouse(num_warehouses: int, assignment: List[int]) -> List[List[int]]:
    retailers_by_w = [[] for _ in range(num_warehouses)]
    for r, w in enumerate(assignment):
        retailers_by_w[w].append(r)
    return retailers_by_w


def compute_remaining_demands(
    demands: List[List[int]],
    retailers_by_w: List[List[int]],
) -> Tuple[List[List[float]], List[List[float]], List[float]]:
    num_retailers = len(demands)
    num_periods = len(demands[0]) if num_retailers > 0 else 0
    num_warehouses = len(retailers_by_w)

    rem_r = [[0.0] * num_periods for _ in range(num_retailers)]
    for r in range(num_retailers):
        run = 0.0
        for t in range(num_periods - 1, -1, -1):
            run += float(demands[r][t])
            rem_r[r][t] = run

    rem_w = [[0.0] * num_periods for _ in range(num_warehouses)]
    for w in range(num_warehouses):
        for t in range(num_periods):
            rem_w[w][t] = sum(rem_r[r][t] for r in retailers_by_w[w])

    rem_p = [0.0] * num_periods
    for t in range(num_periods):
        rem_p[t] = sum(rem_r[r][t] for r in range(num_retailers))

    return rem_r, rem_w, rem_p


def heuristic_jit_solution(instance: Dict[str, Any], retailers_by_w: List[List[int]]) -> Dict[str, Any]:
    R = instance["num_retailers"]
    W = instance["num_warehouses"]
    T = instance["num_periods"]
    demands = instance["demands"]
    fixed = instance["fixed_costs"]
    hold = instance["holding_costs"]

    x_retailers = [[float(demands[r][t]) for t in range(T)] for r in range(R)]
    s_retailers = [[0.0 for _ in range(T)] for _ in range(R)]
    y_retailers = [[1 if demands[r][t] > 0 else 0 for t in range(T)] for r in range(R)]

    x_warehouses = [[0.0 for _ in range(T)] for _ in range(W)]
    for w in range(W):
        for t in range(T):
            x_warehouses[w][t] = float(sum(demands[r][t] for r in retailers_by_w[w]))
    s_warehouses = [[0.0 for _ in range(T)] for _ in range(W)]
    y_warehouses = [[1 if x_warehouses[w][t] > 0 else 0 for t in range(T)] for w in range(W)]

    x_plant = [0.0 for _ in range(T)]
    for t in range(T):
        x_plant[t] = float(sum(x_warehouses[w][t] for w in range(W)))
    s_plant = [0.0 for _ in range(T)]
    y_plant = [1 if x_plant[t] > 0 else 0 for t in range(T)]

    obj = 0.0
    for t in range(T):
        obj += fixed["plant"][t] * y_plant[t]
        obj += hold["plant"] * s_plant[t]
    for w in range(W):
        for t in range(T):
            obj += fixed["warehouses"][w][t] * y_warehouses[w][t]
            obj += hold["warehouses"] * s_warehouses[w][t]
    for r in range(R):
        for t in range(T):
            obj += fixed["retailers"][r][t] * y_retailers[r][t]
            obj += hold["retailers"][r] * s_retailers[r][t]

    return {
        "objective_value": float(obj),
        "x_plant": x_plant,
        "s_plant": s_plant,
        "y_plant": y_plant,
        "x_warehouses": x_warehouses,
        "s_warehouses": s_warehouses,
        "y_warehouses": y_warehouses,
        "x_retailers": x_retailers,
        "s_retailers": s_retailers,
        "y_retailers": y_retailers,
    }


def clean_float(v: float, tol: float = 1e-8) -> float:
    if abs(v) <= tol:
        return 0.0
    return float(v)


def extract_solution_from_model(
    model: gp.Model,
    xP, sP, yP,
    xW, sW, yW,
    xR, sR, yR,
    W: int, R: int, T: int
) -> Dict[str, Any]:
    x_plant = [clean_float(xP[t].X) for t in range(T)]
    s_plant = [clean_float(sP[t].X) for t in range(T)]
    y_plant = [1 if yP[t].X >= 0.5 else 0 for t in range(T)]

    x_warehouses = [[clean_float(xW[w, t].X) for t in range(T)] for w in range(W)]
    s_warehouses = [[clean_float(sW[w, t].X) for t in range(T)] for w in range(W)]
    y_warehouses = [[1 if yW[w, t].X >= 0.5 else 0 for t in range(T)] for w in range(W)]

    x_retailers = [[clean_float(xR[r, t].X) for t in range(T)] for r in range(R)]
    s_retailers = [[clean_float(sR[r, t].X) for t in range(T)] for r in range(R)]
    y_retailers = [[1 if yR[r, t].X >= 0.5 else 0 for t in range(T)] for r in range(R)]

    return {
        "objective_value": float(model.ObjVal),
        "x_plant": x_plant,
        "s_plant": s_plant,
        "y_plant": y_plant,
        "x_warehouses": x_warehouses,
        "s_warehouses": s_warehouses,
        "y_warehouses": y_warehouses,
        "x_retailers": x_retailers,
        "s_retailers": s_retailers,
        "y_retailers": y_retailers,
    }


def solve_instance(instance: Dict[str, Any], time_limit: int, logger: SolutionLogger = None) -> Dict[str, Any]:
    R = instance["num_retailers"]
    W = instance["num_warehouses"]
    T = instance["num_periods"]

    assignment = instance["retailer_warehouse_assignment"]
    demands = instance["demands"]
    fixed = instance["fixed_costs"]
    hold = instance["holding_costs"]

    retailers_by_w = build_retailers_by_warehouse(W, assignment)
    rem_r, rem_w, rem_p = compute_remaining_demands(demands, retailers_by_w)

    heuristic_sol = heuristic_jit_solution(instance, retailers_by_w)
    if logger:
        logger.log(heuristic_sol["objective_value"])

    model = gp.Model("three_echelon_lotsizing")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = max(1, int(time_limit))
    model.Params.Threads = 1

    xP = model.addVars(T, lb=0.0, ub=[rem_p[t] for t in range(T)], vtype=GRB.CONTINUOUS, name="xP")
    sP = model.addVars(T, lb=0.0, vtype=GRB.CONTINUOUS, name="sP")
    yP = model.addVars(T, vtype=GRB.BINARY, name="yP")

    xW = model.addVars(W, T, lb=0.0, vtype=GRB.CONTINUOUS, name="xW")
    sW = model.addVars(W, T, lb=0.0, vtype=GRB.CONTINUOUS, name="sW")
    yW = model.addVars(W, T, vtype=GRB.BINARY, name="yW")

    xR = model.addVars(R, T, lb=0.0, vtype=GRB.CONTINUOUS, name="xR")
    sR = model.addVars(R, T, lb=0.0, vtype=GRB.CONTINUOUS, name="sR")
    yR = model.addVars(R, T, vtype=GRB.BINARY, name="yR")

    # Tight upper bounds by remaining demand
    for w in range(W):
        for t in range(T):
            xW[w, t].UB = rem_w[w][t]
    for r in range(R):
        for t in range(T):
            xR[r, t].UB = rem_r[r][t]

    # Flow balance at plant
    for t in range(T):
        lhs = (sP[t - 1] if t > 0 else 0.0) + xP[t]
        rhs = gp.quicksum(xW[w, t] for w in range(W)) + sP[t]
        model.addConstr(lhs == rhs, name=f"plant_bal_{t}")

    # Flow balance at warehouses
    for w in range(W):
        for t in range(T):
            lhs = (sW[w, t - 1] if t > 0 else 0.0) + xW[w, t]
            rhs = gp.quicksum(xR[r, t] for r in retailers_by_w[w]) + sW[w, t]
            model.addConstr(lhs == rhs, name=f"wh_bal_{w}_{t}")

    # Flow balance at retailers
    for r in range(R):
        for t in range(T):
            lhs = (sR[r, t - 1] if t > 0 else 0.0) + xR[r, t]
            rhs = float(demands[r][t]) + sR[r, t]
            model.addConstr(lhs == rhs, name=f"ret_bal_{r}_{t}")

    # Setup linking with per-period remaining demand bounds
    for t in range(T):
        model.addConstr(xP[t] <= rem_p[t] * yP[t], name=f"setup_p_{t}")
    for w in range(W):
        for t in range(T):
            model.addConstr(xW[w, t] <= rem_w[w][t] * yW[w, t], name=f"setup_w_{w}_{t}")
    for r in range(R):
        for t in range(T):
            model.addConstr(xR[r, t] <= rem_r[r][t] * yR[r, t], name=f"setup_r_{r}_{t}")

    # Objective
    obj = gp.LinExpr()
    for t in range(T):
        obj += float(fixed["plant"][t]) * yP[t]
        obj += float(hold["plant"]) * sP[t]
    for w in range(W):
        for t in range(T):
            obj += float(fixed["warehouses"][w][t]) * yW[w, t]
            obj += float(hold["warehouses"]) * sW[w, t]
    for r in range(R):
        for t in range(T):
            obj += float(fixed["retailers"][r][t]) * yR[r, t]
            obj += float(hold["retailers"][r]) * sR[r, t]

    model.setObjective(obj, GRB.MINIMIZE)

    # MIP start from simple feasible JIT plan
    for t in range(T):
        xP[t].Start = heuristic_sol["x_plant"][t]
        sP[t].Start = heuristic_sol["s_plant"][t]
        yP[t].Start = heuristic_sol["y_plant"][t]
    for w in range(W):
        for t in range(T):
            xW[w, t].Start = heuristic_sol["x_warehouses"][w][t]
            sW[w, t].Start = heuristic_sol["s_warehouses"][w][t]
            yW[w, t].Start = heuristic_sol["y_warehouses"][w][t]
    for r in range(R):
        for t in range(T):
            xR[r, t].Start = heuristic_sol["x_retailers"][r][t]
            sR[r, t].Start = heuristic_sol["s_retailers"][r][t]
            yR[r, t].Start = heuristic_sol["y_retailers"][r][t]

    cb_state = {"best": heuristic_sol["objective_value"]}

    def callback(m: gp.Model, where: int):
        if logger is None:
            return
        if where == GRB.Callback.MIPSOL:
            obj_val = m.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj_val + 1e-9 < cb_state["best"]:
                cb_state["best"] = obj_val
                logger.log(float(obj_val))

    if logger:
        model.optimize(callback)
    else:
        model.optimize()

    if model.SolCount > 0:
        return extract_solution_from_model(model, xP, sP, yP, xW, sW, yW, xR, sR, yR, W, R, T)

    # Fallback (should be rare): return heuristic feasible plan
    return heuristic_sol


def main():
    parser = argparse.ArgumentParser(description="Solve 3-level uncapacitated lot-sizing problem.")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to instance JSON file.")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to write solution JSON.")
    parser.add_argument("--time_limit", type=int, required=True, help="Maximum runtime in seconds.")
    parser.add_argument("--log_path", type=str, default=None, help="Optional path to JSONL convergence log.")
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    solution = solve_instance(instance, args.time_limit, logger=logger)

    # Ensure exact output schema keys
    output = {
        "objective_value": float(solution["objective_value"]),
        "x_plant": [float(v) for v in solution["x_plant"]],
        "s_plant": [float(v) for v in solution["s_plant"]],
        "y_plant": [int(v) for v in solution["y_plant"]],
        "x_warehouses": [[float(v) for v in row] for row in solution["x_warehouses"]],
        "s_warehouses": [[float(v) for v in row] for row in solution["s_warehouses"]],
        "y_warehouses": [[int(v) for v in row] for row in solution["y_warehouses"]],
        "x_retailers": [[float(v) for v in row] for row in solution["x_retailers"]],
        "s_retailers": [[float(v) for v in row] for row in solution["s_retailers"]],
        "y_retailers": [[int(v) for v in row] for row in solution["y_retailers"]],
    }

    with open(args.solution_path, "w") as f:
        json.dump(output, f)


if __name__ == "__main__":
    main()
import argparse
import json
import math
from typing import Dict, Tuple, Any

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def mip_incumbent_logger(model, where):
    """Gurobi callback to log every improved incumbent objective."""
    if where == GRB.Callback.MIPSOL:
        obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
        if obj > model._best_logged_obj + 1e-9:
            model._best_logged_obj = obj
            if model._logger is not None:
                model._logger.log(float(obj))


def solve_instance(data: Dict[str, Any], time_limit: int, logger: SolutionLogger = None):
    classes = data["classes"]
    capacity = data["b"]

    model = gp.Model("family_knapsack_with_setups")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = max(0, int(time_limit))
    model.Params.MIPFocus = 1  # favor finding good feasible solutions quickly

    # Decision variables
    y = {}  # class activation
    x = {}  # item selection

    for cls in classes:
        cid = cls["class_id"]
        y[cid] = model.addVar(vtype=GRB.BINARY, name=f"y_{cid}")

        for it in cls["items"]:
            iid = it["item_id"]
            x[(cid, iid)] = model.addVar(vtype=GRB.BINARY, name=f"x_{cid}_{iid}")

    model.update()

    # Linking constraints: item can be chosen only if class is activated
    for cls in classes:
        cid = cls["class_id"]
        for it in cls["items"]:
            iid = it["item_id"]
            model.addConstr(x[(cid, iid)] <= y[cid], name=f"link_{cid}_{iid}")

    # Capacity constraint
    cap_expr = gp.quicksum(cls["d_i"] * y[cls["class_id"]] for cls in classes)
    cap_expr += gp.quicksum(
        it["a_ij"] * x[(cls["class_id"], it["item_id"])]
        for cls in classes
        for it in cls["items"]
    )
    model.addConstr(cap_expr <= capacity, name="capacity")

    # Objective: item profits + setup costs (setup costs are negative integers)
    obj_expr = gp.quicksum(
        it["c_ij"] * x[(cls["class_id"], it["item_id"])]
        for cls in classes
        for it in cls["items"]
    )
    obj_expr += gp.quicksum(cls["f_i"] * y[cls["class_id"]] for cls in classes)
    model.setObjective(obj_expr, GRB.MAXIMIZE)

    # Provide trivial feasible MIP start (all zeros)
    for cls in classes:
        cid = cls["class_id"]
        y[cid].Start = 0.0
        for it in cls["items"]:
            x[(cid, it["item_id"])].Start = 0.0

    # Attach logger data for callback
    model._logger = logger
    model._best_logged_obj = -math.inf

    model.optimize(mip_incumbent_logger)

    # Ensure final best objective is logged even if callback missed it
    if logger is not None and model.SolCount > 0:
        best_obj = float(model.ObjVal)
        if best_obj > model._best_logged_obj + 1e-9:
            logger.log(best_obj)
            model._best_logged_obj = best_obj

    return model, x, y


def build_solution(data: Dict[str, Any], model: gp.Model, x, y) -> Dict[str, Any]:
    classes_out = []
    has_solution = model.SolCount > 0

    objective_value = 0.0

    for cls in data["classes"]:
        cid = cls["class_id"]
        y_val = int(round(y[cid].X)) if has_solution else 0

        cls_out = {
            "class_id": cid,
            "y_i": y_val,
            "items": []
        }

        if y_val == 1:
            objective_value += float(cls["f_i"])

        for it in cls["items"]:
            iid = it["item_id"]
            x_val = int(round(x[(cid, iid)].X)) if has_solution else 0

            cls_out["items"].append({
                "item_id": iid,
                "x_ij": x_val
            })

            if x_val == 1:
                objective_value += float(it["c_ij"])

        classes_out.append(cls_out)

    return {
        "classes": classes_out,
        "objective_value": float(objective_value)
    }


def main():
    parser = argparse.ArgumentParser(description="Solve family knapsack with setup costs/capacity using Gurobi.")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to input instance JSON.")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to output solution JSON.")
    parser.add_argument("--time_limit", type=int, required=True, help="Maximum runtime in seconds.")
    parser.add_argument("--log_path", type=str, default=None, help="Optional JSONL path for incumbent logging.")
    args = parser.parse_args()

    with open(args.instance_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    model, x, y = solve_instance(data, args.time_limit, logger=logger)
    solution = build_solution(data, model, x, y)

    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(solution, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
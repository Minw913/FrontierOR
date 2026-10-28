import argparse
import json
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, default=300)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    classes = inst["classes"]
    b = inst["b"]

    model = gp.Model("knapsack_setup")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1

    y = {}
    x = {}
    for cls in classes:
        ci = cls["class_id"]
        y[ci] = model.addVar(vtype=GRB.BINARY, name=f"y_{ci}")
        for item in cls["items"]:
            ij = item["item_id"]
            x[(ci, ij)] = model.addVar(vtype=GRB.BINARY, name=f"x_{ci}_{ij}")

    # capacity constraint
    cap_expr = gp.LinExpr()
    obj_expr = gp.LinExpr()
    for cls in classes:
        ci = cls["class_id"]
        cap_expr += cls["d_i"] * y[ci]
        obj_expr += cls["f_i"] * y[ci]
        for item in cls["items"]:
            ij = item["item_id"]
            cap_expr += item["a_ij"] * x[(ci, ij)]
            obj_expr += item["c_ij"] * x[(ci, ij)]
            model.addConstr(x[(ci, ij)] <= y[ci])

    model.addConstr(cap_expr <= b, name="capacity")
    model.setObjective(obj_expr, GRB.MAXIMIZE)

    # remaining time budget (reserve small margin for I/O)
    elapsed = time.time() - start_time
    remaining = max(1.0, args.time_limit - elapsed - 2.0)
    model.Params.TimeLimit = remaining

    def build_solution(get_val):
        sol_classes = []
        for cls in classes:
            ci = cls["class_id"]
            yv = 1 if get_val(y[ci]) > 0.5 else 0
            items_out = []
            for item in cls["items"]:
                ij = item["item_id"]
                xv = 1 if (yv == 1 and get_val(x[(ci, ij)]) > 0.5) else 0
                items_out.append({"item_id": ij, "x_ij": xv})
            sol_classes.append({"class_id": ci, "y_i": yv, "items": items_out})
        return sol_classes

    def compute_obj(sol_classes):
        obj = 0
        cls_map = {c["class_id"]: c for c in classes}
        for sc in sol_classes:
            cd = cls_map[sc["class_id"]]
            if sc["y_i"] == 1:
                obj += cd["f_i"]
                item_map = {it["item_id"]: it for it in cd["items"]}
                for si in sc["items"]:
                    if si["x_ij"] == 1:
                        obj += item_map[si["item_id"]]["c_ij"]
        return obj

    best = {"obj": None, "sol": None}

    def callback(m, where):
        if where == GRB.Callback.MIPSOL:
            sol_classes = build_solution(lambda v: m.cbGetSolution(v))
            obj = compute_obj(sol_classes)
            if best["obj"] is None or obj > best["obj"]:
                best["obj"] = obj
                best["sol"] = sol_classes
                if logger:
                    logger.log_solution(float(obj), {
                        "classes": sol_classes,
                        "objective_value": float(obj),
                    })

    model.optimize(callback)

    # Fallback / final extraction
    if model.SolCount > 0:
        sol_classes = build_solution(lambda v: v.X)
        obj = compute_obj(sol_classes)
        if best["obj"] is None or obj > best["obj"]:
            best["obj"] = obj
            best["sol"] = sol_classes
            if logger:
                logger.log_solution(float(obj), {
                    "classes": sol_classes,
                    "objective_value": float(obj),
                })

    if best["sol"] is None:
        # trivial feasible solution: nothing activated
        sol_classes = []
        for cls in classes:
            sol_classes.append({
                "class_id": cls["class_id"],
                "y_i": 0,
                "items": [{"item_id": it["item_id"], "x_ij": 0} for it in cls["items"]],
            })
        best["sol"] = sol_classes
        best["obj"] = 0
        if logger:
            logger.log_solution(0.0, {"classes": sol_classes, "objective_value": 0.0})

    output = {
        "classes": best["sol"],
        "objective_value": float(best["obj"]),
    }
    with open(args.solution_path, "w") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()
import argparse
import json

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def main():
    parser = argparse.ArgumentParser(
        description="Fixed-charge multi-class knapsack solver"
    )
    parser.add_argument("--instance_path", required=True,
                        help="Path to the instance JSON file.")
    parser.add_argument("--solution_path", required=True,
                        help="Path where the solution JSON file is written.")
    parser.add_argument("--time_limit", type=int, required=True,
                        help="Maximum runtime in seconds.")
    parser.add_argument("--log_path", default=None,
                        help="Optional path to a JSONL convergence log file.")
    args = parser.parse_args()

    logger = (SolutionLogger(args.log_path, sense="maximize")
              if args.log_path else None)

    with open(args.instance_path, "r") as fh:
        data = json.load(fh)

    b = int(data["b"])
    classes = data["classes"]

    class_ids = [cl["class_id"] for cl in classes]
    item_keys = [(cl["class_id"], it["item_id"])
                 for cl in classes for it in cl["items"]]

    f_cost = {cl["class_id"]: cl["f_i"] for cl in classes}
    d_setup = {cl["class_id"]: cl["d_i"] for cl in classes}
    a_w = {(cl["class_id"], it["item_id"]): it["a_ij"]
           for cl in classes for it in cl["items"]}
    c_p = {(cl["class_id"], it["item_id"]): it["c_ij"]
           for cl in classes for it in cl["items"]}

    # The select-nothing solution is always feasible (objective 0).
    def empty_solution():
        return {
            "classes": [
                {
                    "class_id": cl["class_id"],
                    "y_i": 0,
                    "items": [{"item_id": it["item_id"], "x_ij": 0}
                              for it in cl["items"]],
                }
                for cl in classes
            ],
            "objective_value": 0.0,
        }

    best = {"obj": 0.0}
    if logger:
        logger.log_solution(0.0, empty_solution())

    # ------------------------- MIP formulation ------------------------- #
    model = gp.Model("class_setup_knapsack")
    params = model.Params
    params.Seed = 0
    params.MIPGap = 1e-4
    params.NumericFocus = 0
    params.Threads = 1
    params.TimeLimit = max(args.time_limit, 0)
    params.OutputFlag = 0

    y = model.addVars(class_ids, vtype=GRB.BINARY, name="y")
    x = model.addVars(item_keys, vtype=GRB.BINARY, name="x")

    model.setObjective(
        gp.quicksum(c_p[k] * x[k] for k in item_keys)
        + gp.quicksum(f_cost[i] * y[i] for i in class_ids),
        GRB.MAXIMIZE,
    )

    model.addConstr(
        gp.quicksum(a_w[k] * x[k] for k in item_keys)
        + gp.quicksum(d_setup[i] * y[i] for i in class_ids) <= b,
        name="capacity",
    )

    model.addConstrs(
        (x[(i, j)] <= y[i] for (i, j) in item_keys),
        name="link",
    )
    model.update()

    # Ordered variable list for fast solution extraction (also in callbacks).
    var_list = [y[i] for i in class_ids] + [x[k] for k in item_keys]
    var_keys = ([("y", i, None) for i in class_ids]
                + [("x", i, j) for (i, j) in item_keys])

    def build_solution(values):
        """Project raw variable values onto the solution schema.

        The activation flag is derived from the selected items (a class is
        activated iff at least one of its items is selected, or the solver
        activated it and doing so is profitable), which removes any spurious
        activations and never hurts feasibility.
        """
        xvals = {}
        yvals = {}
        for (kind, i, j), v in zip(var_keys, values):
            if kind == "y":
                yvals[i] = v
            else:
                xvals[(i, j)] = v

        sol_classes = []
        total = 0.0
        for cl in classes:
            i = cl["class_id"]
            items_out = []
            any_sel = False
            profit = 0.0
            for it in cl["items"]:
                j = it["item_id"]
                sel = 1 if xvals.get((i, j), 0.0) >= 0.5 else 0
                if sel:
                    any_sel = True
                    profit += it["c_ij"]
                items_out.append({"item_id": j, "x_ij": sel})
            yv = 1 if (any_sel or
                       (yvals.get(i, 0.0) >= 0.5 and cl["f_i"] > 0)) else 0
            if yv:
                total += profit + cl["f_i"]
            sol_classes.append({"class_id": i, "y_i": yv, "items": items_out})
        return {"classes": sol_classes, "objective_value": float(total)}

    def incumbent_cb(m, where):
        if where == GRB.Callback.MIPSOL and logger is not None:
            try:
                vals = m.cbGetSolution(var_list)
                sol = build_solution(vals)
                objv = sol["objective_value"]
                if objv > best["obj"] + 1e-9:
                    best["obj"] = objv
                    logger.log_solution(objv, sol)
            except gp.GurobiError:
                pass

    # --------------------------- Solve --------------------------- #
    model.optimize(incumbent_cb)

    if model.SolCount > 0:
        final_sol = build_solution([v.X for v in var_list])
    else:
        # No incumbent (e.g., time limit hit immediately): use trivial one.
        final_sol = empty_solution()

    if logger and final_sol["objective_value"] > best["obj"] + 1e-9:
        best["obj"] = final_sol["objective_value"]
        logger.log_solution(final_sol["objective_value"], final_sol)

    with open(args.solution_path, "w") as fh:
        json.dump(final_sol, fh)


if __name__ == "__main__":
    main()
import argparse
import json

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

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    capacity = data["knapsack_capacity"]
    families = data["families"]

    model = gp.Model("family_knapsack")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = max(1, args.time_limit - 2)
    model.Params.OutputFlag = 0

    y = {}  # family activation
    x = {}  # item selection
    fam_ids = []
    item_keys = []

    for fam in families:
        fid = fam["family_id"]
        fam_ids.append(fid)
        y[fid] = model.addVar(vtype=GRB.BINARY, name=f"y_{fid}")
        for it in fam["items"]:
            iid = it["item_id"]
            x[(fid, iid)] = model.addVar(vtype=GRB.BINARY, name=f"x_{fid}_{iid}")
            item_keys.append((fid, iid))
            # item only if family activated
            model.addConstr(x[(fid, iid)] <= y[fid])

    # capacity constraint
    cap_expr = gp.LinExpr()
    obj_expr = gp.LinExpr()
    for fam in families:
        fid = fam["family_id"]
        cap_expr.addTerms(fam["setup_capacity"], y[fid])
        obj_expr.addTerms(-fam["setup_cost"], y[fid])
        for it in fam["items"]:
            cap_expr.addTerms(it["weight"], x[(fid, it["item_id"])])
            obj_expr.addTerms(it["profit"], x[(fid, it["item_id"])])
    model.addConstr(cap_expr <= capacity)
    model.setObjective(obj_expr, GRB.MAXIMIZE)

    def build_solution(fam_vals, item_vals, obj):
        fams_sel = [fid for fid in fam_ids if fam_vals[fid] > 0.5]
        items_sel = [{"family": fid, "item": iid}
                     for (fid, iid) in item_keys if item_vals[(fid, iid)] > 0.5]
        return {
            "objective_value": float(obj),
            "families_selected": fams_sel,
            "items_selected": items_sel,
        }

    best = {"obj": None, "sol": None}

    def callback(m, where):
        if where == GRB.Callback.MIPSOL:
            obj = m.cbGet(GRB.Callback.MIPSOL_OBJ)
            if best["obj"] is None or obj > best["obj"] + 1e-9:
                fam_vals = {fid: m.cbGetSolution(y[fid]) for fid in fam_ids}
                item_vals = {k: m.cbGetSolution(x[k]) for k in item_keys}
                sol = build_solution(fam_vals, item_vals, obj)
                best["obj"] = obj
                best["sol"] = sol
                if logger:
                    logger.log_solution(obj, sol)

    model.optimize(callback)

    if model.SolCount > 0:
        fam_vals = {fid: y[fid].X for fid in fam_ids}
        item_vals = {k: x[k].X for k in item_keys}
        obj = model.ObjVal
        sol = build_solution(fam_vals, item_vals, obj)
        if best["obj"] is None or obj > best["obj"] + 1e-9:
            if logger:
                logger.log_solution(obj, sol)
        best["sol"] = sol
    elif best["sol"] is None:
        # trivial empty solution
        sol = {"objective_value": 0.0, "families_selected": [], "items_selected": []}
        best["sol"] = sol
        if logger:
            logger.log_solution(0.0, sol)

    with open(args.solution_path, "w") as f:
        json.dump(best["sol"], f, indent=2)


if __name__ == "__main__":
    main()
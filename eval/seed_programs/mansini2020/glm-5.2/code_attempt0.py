import argparse
import json
import time

import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except Exception:
    SolutionLogger = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    return parser.parse_args()


def load_instance(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


class IncumbentLogger:
    """Gurobi callback that logs every new incumbent solution to the SolutionLogger."""
    def __init__(self, model, logger, var_info):
        self.model = model
        self.logger = logger
        self.var_info = var_info  # list of dict: {var, group_id, item_id, profit, weights}
        self.vars = [d["var"] for d in var_info]

    def __call__(self):
        try:
            if self.model.cbGet(GRB.Callback.MIPSOL) is None:
                return
            where = self.model.cbGet(GRB.Callback.MIPSOL)
        except Exception:
            return

        if where == GRB.Callback.MIPSOL:
            try:
                obj = self.model.cbGet(GRB.Callback.MIPSOL_OBJ)
            except Exception:
                obj = None

            x = self.model.cbGetSolution(self.vars)

            selected = []
            for d, val in zip(self.var_info, x):
                if val is not None and val > 0.5:
                    selected.append({
                        "group_id": d["group_id"],
                        "item_id": d["item_id"],
                        "profit": d["profit"],
                        "weights": d["weights"],
                    })

            if self.logger and obj is not None:
                solution_dict = {
                    "objective_value": float(obj),
                    "selected_items": selected,
                }
                self.logger.log_solution(float(obj), solution_dict)


def build_and_solve(instance, time_limit, logger=None):
    params = instance.get("parameters", {})
    groups = instance.get("groups", [])
    capacities = instance.get("capacities", [])
    m_resources = len(capacities)

    env = gp.Env(empty=True)
    env.setParam("Seed", 0)
    env.setParam("Threads", 1)
    env.setParam("OutputFlag", 0)
    env.start()

    model = gp.Model(env=env)
    model.setParam("MIPGap", 1e-4)
    model.setParam("NumericFocus", 0)
    model.setParam("TimeLimit", float(time_limit))

    var_info = []
    # group constraint vars
    group_vars = {}

    for g in groups:
        gid = g["group_id"]
        gvars = []
        for it in g["items"]:
            iid = it["item_id"]
            profit = it["profit"]
            weights = it["weights"]
            x = model.addVar(vtype=GRB.BINARY, obj=profit, name=f"x_{gid}_{iid}")
            var_info.append({
                "var": x,
                "group_id": gid,
                "item_id": iid,
                "profit": profit,
                "weights": weights,
            })
            gvars.append(x)
        group_vars[gid] = gvars

    # exactly one per group
    for gid, gvars in group_vars.items():
        model.addConstr(gp.quicksum(gvars) == 1, name=f"one_per_group_{gid}")

    # resource constraints
    if m_resources > 0:
        for r in range(m_resources):
            terms = []
            for d in var_info:
                w = d["weights"][r] if r < len(d["weights"]) else 0
                if w != 0:
                    terms.append(w * d["var"])
            if terms:
                model.addConstr(gp.quicksum(terms) <= capacities[r], name=f"resource_{r}")

    model.ModelSense = GRB.MAXIMIZE

    callback = IncumbentLogger(model, logger, var_info) if logger is not None else None
    model.optimize(callback)

    status = model.Status
    best_obj = None
    selected = []

    if model.SolCount > 0:
        try:
            best_obj = model.ObjVal
        except Exception:
            best_obj = None
        for d in var_info:
            try:
                val = d["var"].Xn if hasattr(d["var"], "Xn") else d["var"].X
            except Exception:
                val = d["var"].X
            if val is not None and val > 0.5:
                selected.append({
                    "group_id": d["group_id"],
                    "item_id": d["item_id"],
                    "profit": d["profit"],
                    "weights": d["weights"],
                })

    # If somehow no solution, try to return empty feasible-ish structure
    if best_obj is None:
        best_obj = 0.0
        selected = []

    return float(best_obj), selected, status


def main():
    args = parse_args()
    instance = load_instance(args.instance_path)

    logger = None
    if args.log_path and SolutionLogger is not None:
        logger = SolutionLogger(args.log_path, sense="maximize")

    start_time = time.time()
    remaining_time = max(1.0, args.time_limit - (time.time() - start_time))

    obj, selected, status = build_and_solve(instance, remaining_time, logger=logger)

    solution = {
        "objective_value": float(obj),
        "selected_items": selected,
    }

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()
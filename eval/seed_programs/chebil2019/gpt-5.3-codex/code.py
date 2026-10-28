import argparse
import json
import math
from typing import Dict, List, Tuple, Any

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def parse_args():
    parser = argparse.ArgumentParser(description="Solve family-setup knapsack problem with Gurobi.")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to input instance JSON.")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to output solution JSON.")
    parser.add_argument("--time_limit", type=int, required=True, help="Time limit in seconds.")
    parser.add_argument("--log_path", type=str, default=None, help="Optional JSONL path for incumbent logging.")
    return parser.parse_args()


def load_instance(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        data = json.load(f)
    return data


def build_and_solve(instance: Dict[str, Any], time_limit: int, logger: SolutionLogger = None):
    capacity = int(instance["knapsack_capacity"])
    families = instance["families"]

    # Build model
    model = gp.Model("family_setup_knapsack")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = max(0, int(time_limit))

    # Decision variables
    y = {}  # family activation
    x = {}  # item selection

    # Keep direct references for fast objective reconstruction
    fam_setup_cost = {}
    fam_setup_cap = {}
    item_profit = {}
    item_weight = {}

    for fam in families:
        fid = int(fam["family_id"])
        setup_cost = int(fam["setup_cost"])
        setup_cap = int(fam["setup_capacity"])

        fam_setup_cost[fid] = setup_cost
        fam_setup_cap[fid] = setup_cap

        y[fid] = model.addVar(vtype=GRB.BINARY, name=f"y_{fid}")

        for it in fam["items"]:
            iid = int(it["item_id"])
            w = int(it["weight"])
            p = int(it["profit"])

            x[(fid, iid)] = model.addVar(vtype=GRB.BINARY, name=f"x_{fid}_{iid}")
            item_weight[(fid, iid)] = w
            item_profit[(fid, iid)] = p

    model.update()

    # Capacity constraint
    model.addConstr(
        gp.quicksum(fam_setup_cap[fid] * y[fid] for fid in y.keys())
        + gp.quicksum(item_weight[key] * x[key] for key in x.keys())
        <= capacity,
        name="capacity"
    )

    # Linking constraints: item can be selected only if family activated
    for (fid, iid), var in x.items():
        model.addConstr(var <= y[fid], name=f"link_{fid}_{iid}")

    # Objective: maximize item profits - setup costs
    model.setObjective(
        gp.quicksum(item_profit[key] * x[key] for key in x.keys())
        - gp.quicksum(fam_setup_cost[fid] * y[fid] for fid in y.keys()),
        GRB.MAXIMIZE
    )

    # Incumbent logging callback
    class Tracker:
        def __init__(self):
            self.best_logged = -math.inf

        def maybe_log(self, val: float):
            if logger is not None and val > self.best_logged + 1e-9:
                logger.log(float(val))
                self.best_logged = float(val)

    tracker = Tracker()

    def cb(m: gp.Model, where):
        if where == GRB.Callback.MIPSOL:
            obj = m.cbGet(GRB.Callback.MIPSOL_OBJ)
            tracker.maybe_log(obj)

    model.optimize(cb)

    # Extract best solution found
    families_selected: List[int] = []
    items_selected: List[Dict[str, int]] = []
    objective_value = 0.0

    if model.SolCount > 0:
        selected_fams = []
        selected_items = []

        for fid, yvar in y.items():
            if yvar.X > 0.5:
                selected_fams.append(fid)

        for (fid, iid), xvar in x.items():
            if xvar.X > 0.5:
                selected_items.append((fid, iid))

        selected_fams.sort()
        selected_items.sort()

        # Recompute objective exactly from selected variables
        total_profit = sum(item_profit[(fid, iid)] for fid, iid in selected_items)
        total_setup_cost = sum(fam_setup_cost[fid] for fid in selected_fams)
        objective_value = float(total_profit - total_setup_cost)

        families_selected = selected_fams
        items_selected = [{"family": fid, "item": iid} for (fid, iid) in selected_items]

        tracker.maybe_log(objective_value)
    else:
        # Always-feasible empty solution
        objective_value = 0.0
        families_selected = []
        items_selected = []
        tracker.maybe_log(objective_value)

    return {
        "objective_value": float(objective_value),
        "families_selected": families_selected,
        "items_selected": items_selected
    }


def main():
    args = parse_args()
    instance = load_instance(args.instance_path)

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    solution = build_and_solve(instance, args.time_limit, logger=logger)

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, separators=(",", ":"))


if __name__ == "__main__":
    main()
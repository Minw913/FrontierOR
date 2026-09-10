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

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    m_resources = data["parameters"]["m_resources"]
    capacities = data["capacities"]
    groups = data["groups"]

    # Build MIP model: multiple-choice multidimensional knapsack
    model = gp.Model("mckp")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1

    elapsed = time.time() - start_time
    remaining = max(1.0, args.time_limit - elapsed - 2.0)
    model.Params.TimeLimit = remaining

    x = {}
    for g in groups:
        gid = g["group_id"]
        for it in g["items"]:
            x[(gid, it["item_id"])] = model.addVar(
                vtype=GRB.BINARY, obj=it["profit"], name=f"x_{gid}_{it['item_id']}"
            )
    model.ModelSense = GRB.MAXIMIZE

    # Exactly one item per group
    for g in groups:
        gid = g["group_id"]
        model.addConstr(
            gp.quicksum(x[(gid, it["item_id"])] for it in g["items"]) == 1,
            name=f"group_{gid}",
        )

    # Resource capacities
    for r in range(m_resources):
        model.addConstr(
            gp.quicksum(
                it["weights"][r] * x[(g["group_id"], it["item_id"])]
                for g in groups
                for it in g["items"]
            )
            <= capacities[r],
            name=f"res_{r}",
        )

    # Lookup for building solutions
    item_lookup = {}
    for g in groups:
        for it in g["items"]:
            item_lookup[(g["group_id"], it["item_id"])] = it

    def build_solution(selection):
        selected = []
        total = 0.0
        for gid, iid in selection:
            it = item_lookup[(gid, iid)]
            selected.append(
                {
                    "group_id": gid,
                    "item_id": iid,
                    "profit": it["profit"],
                    "weights": list(it["weights"]),
                }
            )
            total += it["profit"]
        return {"objective_value": float(total), "selected_items": selected}

    def callback(mdl, where):
        if where == GRB.Callback.MIPSOL:
            vals = mdl.cbGetSolution(mdl.getVars())
            selection = []
            for (gid, iid), var in x.items():
                if vals[var.index] > 0.5:
                    selection.append((gid, iid))
            selection.sort()
            sol = build_solution(selection)
            if logger:
                logger.log_solution(sol["objective_value"], sol)

    model.optimize(callback)

    solution = None
    if model.SolCount > 0:
        selection = []
        for (gid, iid), var in x.items():
            if var.X > 0.5:
                selection.append((gid, iid))
        selection.sort()
        solution = build_solution(selection)
    else:
        # Fallback: pick the minimum total-weight item per group (may be infeasible)
        selection = []
        for g in groups:
            best = min(g["items"], key=lambda it: sum(it["weights"]))
            selection.append((g["group_id"], best["item_id"]))
        selection.sort()
        solution = build_solution(selection)
        if logger:
            logger.log_solution(solution["objective_value"], solution)

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()
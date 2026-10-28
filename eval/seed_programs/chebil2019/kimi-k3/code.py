#!/usr/bin/env python3
"""Family-activation (setup) knapsack solver.

Formulates the problem as a MIP and solves it with Gurobi. A fast greedy
heuristic is used as a warm start and as a safeguard in the unlikely event
that Gurobi cannot produce an incumbent within the time limit. Every new
incumbent found by Gurobi is reported to the SolutionLogger.
"""

import argparse
import json
import time
from collections import defaultdict

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def parse_args():
    parser = argparse.ArgumentParser(description="Family-activation knapsack solver")
    parser.add_argument("--instance_path", required=True,
                        help="Path to the JSON problem instance.")
    parser.add_argument("--solution_path", required=True,
                        help="Path where the final solution JSON is written.")
    parser.add_argument("--time_limit", type=int, required=True,
                        help="Maximum runtime in seconds.")
    parser.add_argument("--log_path", default=None,
                        help="Optional path to a JSONL convergence log file.")
    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()

    with open(args.instance_path, "r") as fh:
        instance = json.load(fh)

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    capacity = int(instance.get("knapsack_capacity", 0))
    families = instance.get("families", [])

    # ---- Flatten instance data ------------------------------------------------
    fam_ids = []
    setup_cost = []
    setup_cap = []
    it_fam_idx = []   # item -> family index
    it_fam_id = []    # item -> family identifier (for output)
    it_id = []        # item -> item identifier (for output)
    it_w = []
    it_p = []

    for i, fam in enumerate(families):
        fid = fam.get("family_id", i)
        fam_ids.append(fid)
        setup_cost.append(int(fam.get("setup_cost", 0)))
        setup_cap.append(int(fam.get("setup_capacity", 0)))
        for it in fam.get("items", []):
            k = len(it_w)
            it_fam_idx.append(i)
            it_fam_id.append(fid)
            it_id.append(it.get("item_id", k))
            it_w.append(int(it.get("weight", 0)))
            it_p.append(int(it.get("profit", 0)))

    nF = len(fam_ids)
    nI = len(it_w)

    def build_solution(yvals, xvals):
        """Project raw variable values onto the solution schema."""
        total_profit = 0
        total_setup = 0
        fam_sel = []
        item_sel = []
        for i, v in enumerate(yvals):
            if v > 0.5:
                fam_sel.append(fam_ids[i])
                total_setup += setup_cost[i]
        for k, v in enumerate(xvals):
            if v > 0.5:
                item_sel.append({"family": it_fam_id[k], "item": it_id[k]})
                total_profit += it_p[k]
        return {
            "objective_value": float(total_profit - total_setup),
            "families_selected": fam_sel,
            "items_selected": item_sel,
        }

    # ---- Greedy heuristic (warm start / fallback) -----------------------------
    order = []
    for k in range(nI):
        w, p = it_w[k], it_p[k]
        if p <= 0:
            continue
        ratio = p / w if w > 0 else float("inf")
        order.append((ratio, p, k))
    order.sort(key=lambda t: (t[0], t[1]), reverse=True)

    rem = capacity
    active = set()
    chosen = set()
    for _, _, k in order:
        i = it_fam_idx[k]
        extra = setup_cap[i] if i not in active else 0
        if it_w[k] + extra <= rem:
            if i not in active:
                active.add(i)
                rem -= setup_cap[i]
            chosen.add(k)
            rem -= it_w[k]

    fam_gain = defaultdict(int)
    for k in chosen:
        fam_gain[it_fam_idx[k]] += it_p[k]
    greedy_fams = {i for i in active if fam_gain.get(i, 0) > setup_cost[i]}
    greedy_items = {k for k in chosen if it_fam_idx[k] in greedy_fams}

    greedy_y = [1.0 if i in greedy_fams else 0.0 for i in range(nF)]
    greedy_x = [1.0 if k in greedy_items else 0.0 for k in range(nI)]
    greedy_sol = build_solution(greedy_y, greedy_x)

    # ---- MIP model -------------------------------------------------------------
    model = gp.Model("family_activation_knapsack")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1

    yvars = model.addVars(nF, vtype=GRB.BINARY, name="y")
    xvars = model.addVars(nI, vtype=GRB.BINARY, name="x")

    obj = gp.quicksum(it_p[k] * xvars[k] for k in range(nI))
    obj -= gp.quicksum(setup_cost[i] * yvars[i] for i in range(nF))
    model.setObjective(obj, GRB.MAXIMIZE)

    cap_expr = gp.quicksum(it_w[k] * xvars[k] for k in range(nI))
    cap_expr += gp.quicksum(setup_cap[i] * yvars[i] for i in range(nF))
    model.addConstr(cap_expr <= capacity, name="capacity")

    model.addConstrs(
        (xvars[k] <= yvars[it_fam_idx[k]] for k in range(nI)),
        name="activation_link",
    )

    # Warm start from the greedy solution.
    for i in greedy_fams:
        yvars[i].Start = 1.0
    for k in greedy_items:
        xvars[k].Start = 1.0

    yvar_list = [yvars[i] for i in range(nF)]
    xvar_list = [xvars[k] for k in range(nI)]

    def incumbent_callback(m, where):
        if where == GRB.Callback.MIPSOL and logger is not None:
            try:
                yv = m.cbGetSolution(yvar_list)
                xv = m.cbGetSolution(xvar_list)
                sol = build_solution(yv, xv)
                logger.log_solution(sol["objective_value"], sol)
            except Exception:
                pass

    # ---- Solve within the remaining time budget --------------------------------
    elapsed = time.time() - start_time
    remaining = float(args.time_limit) - elapsed - 0.5  # reserve for I/O

    if remaining > 0.0:
        model.Params.TimeLimit = remaining
        model.optimize(incumbent_callback if logger is not None else None)

    # ---- Extract best available solution ---------------------------------------
    best = greedy_sol
    if model.SolCount > 0:
        yv = model.getAttr("X", yvar_list)
        xv = model.getAttr("X", xvar_list)
        mip_sol = build_solution(yv, xv)
        if mip_sol["objective_value"] >= best["objective_value"]:
            best = mip_sol
    elif logger is not None:
        # No incumbent was produced (e.g., extremely tight time limit):
        # at least record the greedy fallback solution.
        logger.log_solution(best["objective_value"], best)

    with open(args.solution_path, "w") as fh:
        json.dump(best, fh, indent=2)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Crew rotation selection: exact-cover (set partitioning) with optional base
labor-utilization constraints. Solved with Gurobi; a greedy exact-cover
fallback is used if the MIP fails to produce a feasible incumbent."""

import argparse
import json
import random
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


# ---------------------------------------------------------------------------
# Instance loading / helpers
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--instance_path", required=True)
    p.add_argument("--solution_path", required=True)
    p.add_argument("--time_limit", type=int, required=True)
    p.add_argument("--log_path", default=None)
    return p.parse_args()


def load_instance(path):
    with open(path, "r") as f:
        data = json.load(f)
    num_rows = data["dimensions"]["num_rows"]
    num_cols = data["dimensions"]["num_cols"]
    cost = [float(c) for c in data["cost_vector"]]
    col_rows = [list(map(int, r)) for r in data["constraint_matrix_A"]["columns"]]
    row_cols = [[] for _ in range(num_rows)]
    for j, rows in enumerate(col_rows):
        for i in rows:
            row_cols[i].append(j)

    base_data = None
    if data.get("has_base_constraints") and data.get("base_constraints"):
        bc = data["base_constraints"]
        D = bc["D_matrix"]["rows"]
        lb = bc["lower_bounds_d1"]
        ub = bc["upper_bounds_d2"]
        if D and lb and ub and len(lb) > 0:
            base_data = (D, lb, ub)
    return num_rows, num_cols, cost, col_rows, row_cols, base_data


def make_solution_dict(num_cols, cost, xvals):
    sel = [j for j, v in enumerate(xvals) if v >= 0.5]
    obj = float(sum(cost[j] for j in sel))
    return obj, {
        "objective_value": obj,
        "selected_rotations": sorted(sel),
        "variable_values": {str(j): (1.0 if xvals[j] >= 0.5 else 0.0)
                            for j in range(num_cols)},
    }


def base_feasible(chosen, base_data):
    if not base_data:
        return True
    D, lb, ub = base_data
    for b in range(len(lb)):
        s = sum(D[b][j] for j in chosen)
        if s < lb[b] - 1e-9 or s > ub[b] + 1e-9:
            return False
    return True


# ---------------------------------------------------------------------------
# Greedy exact-cover fallback (only used if the MIP yields no feasible point)
# ---------------------------------------------------------------------------

def greedy_fallback(num_rows, cost, col_rows, row_cols, base_data, deadline):
    rng = random.Random(0)
    col_sets = [set(r) for r in col_rows]
    best = None  # (obj, chosen)
    while time.time() < deadline:
        remaining = set(range(num_rows))
        chosen = []
        ok = True
        while remaining:
            best_row, best_cands = None, None
            for r in remaining:
                cands = [j for j in row_cols[r] if col_sets[j] <= remaining]
                if not cands:
                    best_cands = []
                    break
                if best_cands is None or len(cands) < len(best_cands):
                    best_cands = cands
                    if len(cands) == 1:
                        break
            if not best_cands:
                ok = False
                break
            j = min(best_cands,
                    key=lambda jj: (cost[jj] / float(len(col_sets[jj])),
                                    rng.random()))
            chosen.append(j)
            remaining -= col_sets[j]
        if not ok or not base_feasible(chosen, base_data):
            continue
        obj = float(sum(cost[j] for j in chosen))
        if best is None or obj < best[0]:
            best = (obj, sorted(chosen))
    return best


# ---------------------------------------------------------------------------
# Main MIP solution
# ---------------------------------------------------------------------------

def solve(num_rows, num_cols, cost, col_rows, row_cols, base_data,
          time_limit, logger):
    deadline = time.time() + time_limit
    best = {"obj": float("inf"), "x": None}

    model = gp.Model("crew_rotation")
    model.setParam("Seed", 0)
    model.setParam("MIPGap", 1e-4)
    model.setParam("NumericFocus", 0)
    model.setParam("Threads", 1)
    grb_time = max(1.0, time_limit - 1.0)
    model.setParam("TimeLimit", grb_time)

    x = model.addVars(num_cols, vtype=GRB.BINARY, obj=cost, name="x")
    model.ModelSense = GRB.MINIMIZE

    for i in range(num_rows):
        cols = row_cols[i]
        model.addConstr(gp.quicksum(x[j] for j in cols) == 1, name=f"cover_{i}")

    if base_data:
        D, lb, ub = base_data
        for b in range(len(lb)):
            expr = gp.LinExpr(D[b], [x[j] for j in range(num_cols)])
            model.addConstr(expr >= lb[b], name=f"base_lo_{b}")
            model.addConstr(expr <= ub[b], name=f"base_hi_{b}")

    model._xvars = [x[j] for j in range(num_cols)]

    def callback(m, where):
        if where == GRB.Callback.MIPSOL:
            try:
                xvals = m.cbGetSolution(m._xvars)
            except Exception:
                return
            obj, sol = make_solution_dict(num_cols, cost, xvals)
            if obj < best["obj"] - 1e-9:
                best["obj"] = obj
                best["x"] = xvals
                if logger:
                    logger.log_solution(obj, sol)

    model.optimize(callback)

    xvars = model._xvars
    if model.SolCount > 0:
        xvals = [v.X if hasattr(v, "X") else v.x for v in xvars]
        obj, _ = make_solution_dict(num_cols, cost, xvals)
        if obj < best["obj"]:
            best["obj"] = obj
            best["x"] = xvals

    if best["x"] is not None:
        obj, sol = make_solution_dict(num_cols, cost, best["x"])
        if logger:
            logger.log_solution(obj, sol)
        return sol

    # Fallback: greedy exact-cover with restarts for the remaining time.
    remaining = deadline - time.time() - 0.3
    if remaining > 0.1:
        fb = greedy_fallback(num_rows, cost, col_rows, row_cols, base_data,
                             time.time() + remaining)
        if fb is not None:
            obj, chosen = fb
            xvals = [0.0] * num_cols
            for j in chosen:
                xvals[j] = 1.0
            obj, sol = make_solution_dict(num_cols, cost, xvals)
            if logger:
                logger.log_solution(obj, sol)
            return sol

    # No feasible solution found; emit an empty solution to keep schema valid.
    return {
        "objective_value": 0.0,
        "selected_rotations": [],
        "variable_values": {str(j): 0.0 for j in range(num_cols)},
    }


def main():
    args = parse_args()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    num_rows, num_cols, cost, col_rows, row_cols, base_data = load_instance(
        args.instance_path)

    sol = solve(num_rows, num_cols, cost, col_rows, row_cols, base_data,
                args.time_limit, logger)

    with open(args.solution_path, "w") as f:
        json.dump(sol, f, indent=2)


if __name__ == "__main__":
    main()
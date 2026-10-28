#!/usr/bin/env python3
"""
Quadratic (profit) knapsack solver.

Maximize  sum_i p_ii * x_i  +  sum_{i<j} p_ij * x_i * x_j
subject to  sum_i w_i * x_i <= capacity,  x_i in {0,1}.

Approach:
  1. A greedy marginal-gain heuristic plus a short add/swap local search
     produces a strong feasible solution (also used as fallback).
  2. The problem is solved as a MILP via standard linearization of the
     quadratic terms (y_ij = x_i * x_j, only for nonzero interaction
     profits) using Gurobi. The heuristic solution is passed as a MIP
     start. Every improved incumbent found by Gurobi is recorded with the
     provided SolutionLogger.
"""

import argparse
import json
import time


def main():
    parser = argparse.ArgumentParser(description="Quadratic knapsack solver")
    parser.add_argument("--instance_path", required=True,
                        help="Path to the JSON instance file.")
    parser.add_argument("--solution_path", required=True,
                        help="Path where the solution JSON is written.")
    parser.add_argument("--time_limit", type=int, required=True,
                        help="Maximum runtime in seconds.")
    parser.add_argument("--log_path", default=None,
                        help="Optional JSONL path for incumbent logging.")
    args = parser.parse_args()

    t0 = time.time()
    time_limit = max(0, int(args.time_limit))

    logger = None
    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="maximize")

    # ------------------------- read instance -------------------------
    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    n = int(inst["n"])
    capacity = inst["capacity"]
    weights = inst["weights"]
    P = inst["profit_matrix"]

    # --------------------------- helpers -----------------------------
    def evaluate(sel):
        """Exact quadratic-knapsack objective of a 0/1 selection."""
        idx = [i for i, v in enumerate(sel) if v]
        total = 0
        for i in idx:
            total += P[i][i]
        m = len(idx)
        for a in range(m):
            row = P[idx[a]]
            for b in range(a + 1, m):
                total += row[idx[b]]
        return total

    def greedy():
        """Repeatedly add the fitting item with the best marginal gain/weight."""
        sel = [0] * n
        if n == 0:
            return sel
        load = 0
        gain = [float(P[i][i]) for i in range(n)]
        alive = [True] * n
        while True:
            best_i = -1
            best_key = 0.0
            for i in range(n):
                if not alive[i]:
                    continue
                wi = weights[i]
                if load + wi > capacity:
                    continue
                g = gain[i]
                if g <= 0.0:
                    continue
                key = float("inf") if wi == 0 else g / float(wi)
                if key > best_key:
                    best_key = key
                    best_i = i
            if best_i < 0:
                break
            i = best_i
            alive[i] = False
            sel[i] = 1
            load += weights[i]
            row = P[i]
            for j in range(n):
                if alive[j]:
                    gain[j] += row[j]
        return sel

    def local_search(sel, budget):
        """First pass best-improvement local search with add and 1-1 swap moves."""
        if n == 0 or budget <= 0.0:
            return sel
        sel = list(sel)
        deadline = time.time() + budget
        load = sum(weights[i] for i in range(n) if sel[i])
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # inter[j] = sum of interaction profits of j with selected items
            inter = [0.0] * n
            for i in range(n):
                if sel[i]:
                    row = P[i]
                    for j in range(n):
                        if j != i:
                            inter[j] += row[j]
            best_delta = 1e-9
            move = None
            for i in range(n):
                if sel[i]:
                    continue
                wi = weights[i]
                base_add = P[i][i] + inter[i]
                if load + wi <= capacity and base_add > best_delta:
                    best_delta = base_add
                    move = (i, -1)
                row = P[i]
                for k in range(n):
                    if sel[k] and load - weights[k] + wi <= capacity:
                        delta = base_add - row[k] - (P[k][k] + inter[k])
                        if delta > best_delta:
                            best_delta = delta
                            move = (i, k)
                if time.time() >= deadline:
                    break
            if move is not None:
                i, k = move
                sel[i] = 1
                load += weights[i]
                if k >= 0:
                    sel[k] = 0
                    load -= weights[k]
                improved = True
        return sel

    # ------------------- initial feasible solution --------------------
    best_sel = greedy()
    rem = time_limit - (time.time() - t0)
    best_sel = local_search(best_sel, min(2.0, max(0.0, 0.05 * rem)))
    best_val = evaluate(best_sel)

    last_logged = [float(best_val)]
    if logger:
        logger.log_solution(float(best_val),
                            {"objective_value": float(best_val),
                             "solution": list(best_sel)})

    # --------------------------- MILP stage ---------------------------
    rem = time_limit - (time.time() - t0)
    if n > 0 and rem > 1.5:
        try:
            import gurobipy as gp
            from gurobipy import GRB

            model = gp.Model("quadratic_knapsack")
            model.Params.Seed = 0
            model.Params.MIPGap = 1e-4
            model.Params.NumericFocus = 0
            model.Params.Threads = 1
            model.Params.OutputFlag = 0

            usable = [weights[i] <= capacity for i in range(n)]
            x = model.addVars(n, vtype=GRB.BINARY, name="x")
            for i in range(n):
                if not usable[i]:
                    x[i].UB = 0.0

            model.addConstr(
                gp.quicksum(weights[i] * x[i] for i in range(n) if weights[i])
                <= capacity,
                name="capacity",
            )

            # Collect nonzero interaction profits (upper triangle, each pair once).
            pair_coeff = {}
            for i in range(n):
                if not usable[i]:
                    continue
                row = P[i]
                for j in range(i + 1, n):
                    if usable[j]:
                        p = row[j]
                        if p:
                            pair_coeff[(i, j)] = float(p)

            diag_obj = gp.quicksum(P[i][i] * x[i] for i in range(n))

            MAX_LINEARIZED_PAIRS = 1_000_000
            if len(pair_coeff) <= MAX_LINEARIZED_PAIRS:
                # Standard linearization y_ij = x_i * x_j.  Since all pair
                # profits are non-negative and we maximize, only the upper
                # linking constraints are required.
                pair_keys = list(pair_coeff.keys())
                y = model.addVars(pair_keys, lb=0.0, ub=1.0,
                                  vtype=GRB.CONTINUOUS, name="y")
                model.addConstrs((y[i, j] <= x[i] for (i, j) in pair_keys),
                                 name="link_i")
                model.addConstrs((y[i, j] <= x[j] for (i, j) in pair_keys),
                                 name="link_j")
                model.setObjective(diag_obj + y.prod(pair_coeff),
                                   GRB.MAXIMIZE)
            else:
                # Extremely dense instance: hand the quadratic objective to
                # Gurobi directly (binary products are linearized in presolve).
                obj = diag_obj
                for (i, j), p in pair_coeff.items():
                    obj += p * x[i] * x[j]
                try:
                    model.Params.NonConvex = 2
                except Exception:
                    pass
                model.setObjective(obj, GRB.MAXIMIZE)

            # Warm start with the heuristic solution (all binary variables).
            for i in range(n):
                x[i].Start = best_sel[i]

            def _callback(m, where):
                if where == GRB.Callback.MIPSOL and logger is not None:
                    try:
                        objbst = m.cbGet(GRB.Callback.MIPSOL_OBJBST)
                        if objbst <= last_logged[0] + 1e-9:
                            return
                        xv = m.cbGetSolution(x)
                        sel = [1 if xv[i] >= 0.5 else 0 for i in range(n)]
                        val = float(evaluate(sel))
                        if val > last_logged[0] + 1e-9:
                            last_logged[0] = val
                            logger.log_solution(val,
                                                {"objective_value": val,
                                                 "solution": sel})
                    except Exception:
                        pass

            tl = time_limit - (time.time() - t0) - 0.3
            if tl > 0.2:
                model.Params.TimeLimit = tl
                model.optimize(_callback)

                if model.SolCount > 0:
                    sel = [1 if x[i].X >= 0.5 else 0 for i in range(n)]
                    val = evaluate(sel)
                    if val >= best_val:
                        best_val = val
                        best_sel = sel
        except Exception:
            # Fall back to the heuristic solution below.
            pass

    # Log the final incumbent in case it was not captured by the callback.
    if logger and float(best_val) > last_logged[0] + 1e-9:
        logger.log_solution(float(best_val),
                            {"objective_value": float(best_val),
                             "solution": list(best_sel)})

    # --------------------------- write output --------------------------
    out = {"objective_value": float(best_val),
           "solution": [int(v) for v in best_sel]}
    with open(args.solution_path, "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()
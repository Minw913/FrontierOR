import argparse
import json
import time
import math

import numpy as np


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", type=str, required=True)
    ap.add_argument("--solution_path", type=str, required=True)
    ap.add_argument("--time_limit", type=int, required=True)
    ap.add_argument("--log_path", type=str, default=None)
    return ap.parse_args()


def evaluate(P, x):
    xx = np.asarray(x, dtype=float)
    return float(xx @ P @ xx)


def greedy_initial(weights, P, capacity):
    """Construct a feasible solution by repeatedly adding the item with the
    best incremental-profit / weight ratio."""
    n = len(weights)
    x = np.zeros(n)
    total_w = 0.0
    diag = np.diag(P)
    while True:
        Px = P @ x
        add_delta = 2.0 * Px + diag
        feasible = (x == 0) & (total_w + weights <= capacity)
        if not np.any(feasible):
            break
        ratios = np.where(feasible, add_delta / np.maximum(weights, 1e-12), -np.inf)
        i = int(np.argmax(ratios))
        if ratios[i] <= 0.0 and feasible.sum() > 0:
            # still allow zero-delta free additions only if they occupy capacity
            # that nothing positive can use; simplest is to stop here.
            if add_delta[i] <= 1e-12:
                break
        x[i] = 1.0
        total_w += weights[i]
    return x, total_w


def local_search(weights, P, capacity, x, total_w, deadline):
    """Best-improvement single-flip (add/remove) local search."""
    diag = np.diag(P)
    obj = evaluate(P, x)
    while time.time() < deadline:
        Px = P @ x
        add_delta = 2.0 * Px + diag              # valid only for x_i == 0
        rem_delta = -(2.0 * Px - diag)           # valid only for x_i == 1
        feasible_add = (x == 0) & (total_w + weights <= capacity)
        cand = np.where(feasible_add, add_delta, -np.inf)
        cand = np.where(x == 1, rem_delta, cand)
        k = int(np.argmax(cand)) if cand.size else -1
        if k < 0 or cand[k] <= 1e-9:
            break
        if x[k] == 1.0:
            x[k] = 0.0
            total_w -= weights[k]
        else:
            x[k] = 1.0
            total_w += weights[k]
        obj += cand[k]
    return x, total_w, obj


def heuristic(weights, P, capacity, deadline, rng):
    n = len(weights)
    best_x = np.zeros(n)
    best_obj = 0.0
    first = True
    while time.time() < deadline:
        if first:
            x, total_w = greedy_initial(weights, P, capacity)
            first = False
        else:
            # random restart: random feasible subset biased toward greedy fill
            perm = rng.permutation(n)
            x = np.zeros(n)
            total_w = 0.0
            # with some probability take the greedy solution then perturb
            if rng.random() < 0.5:
                x, total_w = greedy_initial(weights, P, capacity)
                k_pert = max(1, int(0.1 * n))
                idx = rng.choice(n, size=k_pert, replace=False)
                for i in idx:
                    if x[i] == 1.0:
                        x[i] = 0.0
                        total_w -= weights[i]
            else:
                # random feasible packing
                order = perm
                for i in order:
                    if rng.random() < 0.5 and total_w + weights[i] <= capacity:
                        x[i] = 1.0
                        total_w += weights[i]
        x, total_w, obj = local_search(weights, P, capacity, x, total_w, deadline)
        if obj > best_obj + 1e-9:
            best_obj = obj
            best_x = x.copy()
    return best_x, best_obj


def main():
    args = parse_args()
    start_time = time.time()

    try:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None
    except Exception:
        logger = None

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    n = int(data["n"])
    weights = [float(w) for w in data["weights"]]
    capacity = float(data["capacity"])
    P = [[float(v) for v in row] for row in data["profit_matrix"]]
    P_np = np.asarray(P, dtype=float)
    weights_np = np.asarray(weights, dtype=float)

    best_x = np.zeros(n)
    best_obj = 0.0

    def make_solution_dict(x, obj):
        sel = [1 if x[i] >= 0.5 else 0 for i in range(len(x))]
        return {"objective_value": float(obj), "selected_items": sel}

    if n > 0:
        # ---- Heuristic phase (small time budget) ----
        total_budget = float(args.time_limit)
        heur_budget = min(10.0, max(0.2, 0.15 * total_budget))
        heur_deadline = time.time() + heur_budget
        rng = np.random.default_rng(0)
        hx, hobj = heuristic(weights_np, P_np, capacity, heur_deadline, rng)
        best_x = hx
        best_obj = evaluate(P_np, np.asarray(hx, dtype=float))
        if logger and best_obj > -np.inf:
            logger.log_solution(best_obj, make_solution_dict(best_x, best_obj))

        # ---- Gurobi phase ----
        remaining = args.time_limit - (time.time() - start_time) - 0.5
        if remaining > 0.05:
            try:
                import gurobipy as gp
                from gurobipy import GRB

                m = gp.Model("QuadraticKnapsack")
                m.Params.Threads = 1
                m.Params.Seed = 0
                m.Params.MIPGap = 1e-4
                m.Params.NumericFocus = 0
                m.Params.TimeLimit = max(remaining, 0.01)
                m.Params.OutputFlag = 0

                xvars = m.addVars(n, vtype=GRB.BINARY, name="x")
                var_list = [xvars[i] for i in range(n)]

                # capacity constraint
                m.addConstr(
                    gp.quicksum(weights[i] * xvars[i] for i in range(n)) <= capacity,
                    name="capacity",
                )

                # quadratic objective: sum_{i,j} P[i][j] x_i x_j
                obj_expr = gp.QuadExpr()
                diag = np.diag(P_np)
                for i in range(n):
                    if diag[i] != 0.0:
                        obj_expr.addTerms([diag[i]], [xvars[i]])
                    Pi_row = P_np[i]
                    for j in range(i + 1, n):
                        c = Pi_row[j]
                        if c != 0.0:
                            obj_expr.addTerms([2.0 * c], [xvars[i]], [xvars[j]])
                m.setObjective(obj_expr, GRB.MAXIMIZE)

                # warm start with heuristic solution
                for i in range(n):
                    xvars[i].Start = float(best_x[i])

                logged = {"last": best_obj}

                def callback(model, where):
                    if where == GRB.Callback.MIPSOL and logger:
                        try:
                            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                            if obj > logged["last"] + 1e-9:
                                logged["last"] = obj
                                sol = model.cbGetSolution(var_list)
                                sel = [1 if v > 0.5 else 0 for v in sol]
                                logger.log_solution(
                                    float(obj),
                                    {"objective_value": float(obj), "selected_items": sel},
                                )
                        except Exception:
                            pass

                m.optimize(callback)

                if m.SolCount > 0:
                    vals = m.getAttr("X", var_list)
                    x_found = np.asarray([1.0 if v > 0.5 else 0.0 for v in vals])
                    obj_found = evaluate(P_np, x_found)
                    if obj_found > best_obj + 1e-9 or m.Status in (GRB.OPTIMAL,):
                        if obj_found > best_obj - 1e-9:
                            # prefer incumbent consistency; take Gurobi's when no worse
                            best_obj = obj_found
                            best_x = x_found
            except Exception:
                pass

    sel = [1 if best_x[i] >= 0.5 else 0 for i in range(n)]
    result = {"objective_value": float(best_obj), "selected_items": sel}

    with open(args.solution_path, "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
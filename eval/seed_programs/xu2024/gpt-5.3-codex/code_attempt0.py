import argparse
import json
import time
from typing import List, Optional, Tuple

import numpy as np
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def compute_objective(Q: np.ndarray, x: np.ndarray) -> float:
    return float(0.5 * x @ Q @ x)


def solve_continuous_qp(
    Q: np.ndarray,
    r: np.ndarray,
    b: float,
    time_limit: float,
) -> Optional[Tuple[np.ndarray, float]]:
    n = len(r)
    m = gp.Model("continuous_qp")
    m.Params.OutputFlag = 0
    m.Params.Threads = 1
    if time_limit is not None and time_limit > 0:
        m.Params.TimeLimit = float(time_limit)

    x = m.addMVar(n, lb=-GRB.INFINITY, name="x")
    m.addConstr(np.ones(n) @ x == 1.0, name="budget")
    m.addConstr(r @ x >= b, name="return")
    m.setObjective(0.5 * (x @ Q @ x), GRB.MINIMIZE)

    m.optimize()
    if m.SolCount > 0:
        xv = x.X
        obj = compute_objective(Q, xv)
        return xv, obj
    return None


def solve_support_qp(
    Q: np.ndarray,
    r: np.ndarray,
    b: float,
    support: List[int],
    n: int,
    time_limit: float,
) -> Optional[Tuple[np.ndarray, float]]:
    if not support:
        return None

    m = gp.Model("support_qp")
    m.Params.OutputFlag = 0
    m.Params.Threads = 1
    if time_limit is not None and time_limit > 0:
        m.Params.TimeLimit = float(time_limit)

    s = support
    ns = len(s)
    Qs = Q[np.ix_(s, s)]
    rs = r[s]

    xs = m.addMVar(ns, lb=-GRB.INFINITY, name="xs")
    m.addConstr(np.ones(ns) @ xs == 1.0, name="budget")
    m.addConstr(rs @ xs >= b, name="return")
    m.setObjective(0.5 * (xs @ Qs @ xs), GRB.MINIMIZE)

    m.optimize()
    if m.SolCount > 0:
        x_full = np.zeros(n, dtype=float)
        x_full[s] = xs.X
        obj = compute_objective(Q, x_full)
        return x_full, obj
    return None


def build_miqp_model(
    Q: np.ndarray,
    r: np.ndarray,
    b: float,
    k: int,
    time_limit: float,
    logger: Optional[SolutionLogger],
):
    n = len(r)
    m = gp.Model("cardinality_portfolio")
    m.Params.OutputFlag = 0
    m.Params.Threads = 1
    if time_limit is not None and time_limit > 0:
        m.Params.TimeLimit = float(time_limit)

    x = m.addVars(n, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="x")
    z = m.addVars(n, vtype=GRB.BINARY, name="z")

    m.addConstr(gp.quicksum(x[i] for i in range(n)) == 1.0, name="budget")
    m.addConstr(gp.quicksum(float(r[i]) * x[i] for i in range(n)) >= float(b), name="return")
    m.addConstr(gp.quicksum(z[i] for i in range(n)) <= int(k), name="cardinality")

    # If z_i = 0 then x_i = 0 (exactly)
    for i in range(n):
        m.addGenConstrIndicator(z[i], 0, x[i], GRB.EQUAL, 0.0, name=f"ind_zero_{i}")

    # Objective: 0.5 x'Qx (using symmetry)
    obj = gp.QuadExpr()
    for i in range(n):
        obj += 0.5 * float(Q[i, i]) * x[i] * x[i]
        for j in range(i + 1, n):
            obj += float(Q[i, j]) * x[i] * x[j]
    m.setObjective(obj, GRB.MINIMIZE)

    # Callback logger for incumbents
    m._logger = logger
    m._best_logged = float("inf")

    def cb(model, where):
        if where == GRB.Callback.MIPSOL and model._logger is not None:
            val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if val + 1e-12 < model._best_logged:
                model._best_logged = val
                model._logger.log(float(val))

    return m, x, z, cb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    n = int(inst["n"])
    k = int(inst["k"])
    b = float(inst["b"])
    r = np.array(inst["r"], dtype=float)
    Q = np.array(inst["Q"], dtype=float)

    start = time.time()

    best_x = None
    best_obj = float("inf")

    def update_best(xcand: np.ndarray):
        nonlocal best_x, best_obj
        obj = compute_objective(Q, xcand)
        if obj + 1e-12 < best_obj:
            best_obj = obj
            best_x = xcand.copy()
            if logger:
                logger.log(float(best_obj))

    def remaining_time() -> float:
        return max(0.0, float(args.time_limit) - (time.time() - start))

    # Case 1: no effective cardinality restriction
    if k >= n:
        rem = remaining_time()
        sol = solve_continuous_qp(Q, r, b, rem)
        if sol is not None:
            update_best(sol[0])

        if best_x is None:
            out = {
                "objective_value": 1e100,
                "x": [0.0] * n,
            }
        else:
            out = {
                "objective_value": float(best_obj),
                "x": [float(v) for v in best_x.tolist()],
            }

        with open(args.solution_path, "w") as f:
            json.dump(out, f)
        return

    # Heuristic warm starts
    rem = remaining_time()
    if rem > 0.2:
        t_relax = min(2.0, rem * 0.25)
        relaxed = solve_continuous_qp(Q, r, b, t_relax)
        if relaxed is not None:
            x_relax, _ = relaxed

            # Candidate supports
            kk = min(k, n)
            cand_supports = []

            top_abs = np.argsort(-np.abs(x_relax))[:kk].tolist()
            cand_supports.append(sorted(set(top_abs)))

            top_ret = np.argsort(-r)[:kk].tolist()
            cand_supports.append(sorted(set(top_ret)))

            # Union-trim candidate
            union = sorted(set(top_abs) | set(top_ret))
            if len(union) > kk:
                # keep assets with largest |x_relax| in union
                union_sorted = sorted(union, key=lambda i: -abs(x_relax[i]))[:kk]
                union = sorted(union_sorted)
            if union:
                cand_supports.append(union)

            # Deduplicate
            unique_supports = []
            seen = set()
            for s in cand_supports:
                key = tuple(s)
                if key not in seen and len(s) <= k and len(s) > 0:
                    seen.add(key)
                    unique_supports.append(s)

            if unique_supports:
                per_t = max(0.2, min(1.0, remaining_time() * 0.2 / len(unique_supports)))
                for s in unique_supports:
                    if remaining_time() <= 0.05:
                        break
                    sol_s = solve_support_qp(Q, r, b, s, n, min(per_t, remaining_time()))
                    if sol_s is not None:
                        update_best(sol_s[0])

    # Build and solve MIQP
    rem = remaining_time()
    if rem > 0.01:
        model, xvars, zvars, cb = build_miqp_model(Q, r, b, k, rem, logger)

        # Warm start if available
        if best_x is not None:
            for i in range(n):
                xv = float(best_x[i])
                xvars[i].Start = xv
                zvars[i].Start = 1.0 if abs(xv) > 1e-8 else 0.0

        model.optimize(cb)

        if model.SolCount > 0:
            x_mip = np.array([xvars[i].X for i in range(n)], dtype=float)
            update_best(x_mip)

    # Final output
    if best_x is None:
        result = {
            "objective_value": 1e100,
            "x": [0.0] * n,
        }
    else:
        result = {
            "objective_value": float(best_obj),
            "x": [float(v) for v in best_x.tolist()],
        }

    with open(args.solution_path, "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
import argparse
import json
import math
import time
from typing import List, Optional, Tuple

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def build_covariance(instance: dict) -> np.ndarray:
    n = int(instance["n"])

    # Optional direct covariance support (robustness beyond schema)
    if "covariance" in instance and instance["covariance"] is not None:
        cov = np.array(instance["covariance"], dtype=float)
        return cov
    if "covariance_matrix" in instance and instance["covariance_matrix"] is not None:
        cov = np.array(instance["covariance_matrix"], dtype=float)
        return cov

    B_raw = instance.get("factor_loadings", [])
    B = np.array(B_raw, dtype=float)

    if B.size == 0:
        systematic = np.zeros((n, n), dtype=float)
    else:
        # Accept either n x r or r x n
        if B.shape[0] == n:
            systematic = B @ B.T
        elif B.shape[1] == n:
            systematic = B.T @ B
        else:
            raise ValueError(
                f"factor_loadings shape {B.shape} incompatible with n={n}."
            )

    idio = np.array(instance.get("idiosyncratic_variance", [0.0] * n), dtype=float)
    if idio.shape[0] != n:
        raise ValueError("idiosyncratic_variance length must equal n.")
    cov = systematic + np.diag(idio)
    return cov


def compute_objective(
    x: np.ndarray, cov: np.ndarray, gamma: float, kappa: float, mu: np.ndarray
) -> float:
    var_term = 0.5 * float(x @ cov @ x)
    ridge_term = 0.0
    if gamma is not None and gamma > 1e-12:
        ridge_term = 0.5 / gamma * float(x @ x)
    ret_term = float(kappa) * float(mu @ x)
    return var_term + ridge_term - ret_term


def try_construct_fallback(
    n: int,
    k: int,
    mu: np.ndarray,
    use_y: bool,
    lmin: Optional[np.ndarray],
    umax: Optional[np.ndarray],
) -> np.ndarray:
    # Best-effort feasible-ish fallback if solver has no incumbent.
    eps = 1e-12
    x = np.zeros(n, dtype=float)

    if umax is None:
        umax = np.ones(n, dtype=float)
    else:
        umax = np.array(umax, dtype=float)

    if lmin is None:
        lmin = np.zeros(n, dtype=float)
    else:
        lmin = np.array(lmin, dtype=float)

    order = np.argsort(-mu)  # descending by expected return

    if not use_y:
        # Fill greedily under upper bounds
        remaining = 1.0
        for i in order:
            take = min(umax[i], remaining)
            if take > 0:
                x[i] = take
                remaining -= take
            if remaining <= eps:
                break
        if remaining > 1e-8:
            # last-resort uniform
            x[:] = 1.0 / n
        return x

    # use_y case: need <=k active and (if lmin>0) x_i either 0 or >= lmin_i
    k_use = max(1, min(k, n))
    selected = []
    sum_l = 0.0
    sum_u = 0.0
    for i in order:
        if len(selected) >= k_use:
            break
        if umax[i] <= 0:
            continue
        selected.append(i)
        sum_l += lmin[i]
        sum_u += umax[i]
        if sum_u >= 1.0 and sum_l <= 1.0:
            break

    # If condition not met, try adding more up to k
    if (sum_u < 1.0 or sum_l > 1.0) and len(selected) < k_use:
        for i in order:
            if i in selected:
                continue
            if len(selected) >= k_use:
                break
            selected.append(i)
            sum_l += lmin[i]
            sum_u += umax[i]
            if sum_u >= 1.0 and sum_l <= 1.0:
                break

    if not selected:
        x[:] = 1.0 / n
        return x

    # Start with lower bounds
    for i in selected:
        x[i] = lmin[i]
    rem = 1.0 - float(np.sum(x[selected]))

    if rem < -1e-8:
        # infeasible lower bounds set; degrade to best-effort single asset
        x[:] = 0.0
        best = int(order[0])
        x[best] = 1.0
        return x

    # Distribute remaining by mu preference
    if rem > 1e-12:
        for i in selected:
            cap = umax[i] - x[i]
            if cap <= 0:
                continue
            take = min(cap, rem)
            x[i] += take
            rem -= take
            if rem <= 1e-12:
                break

    # If still leftover, attempt any asset (can violate k if needed as last resort)
    if rem > 1e-8:
        for i in order:
            cap = umax[i] - x[i]
            if cap <= 0:
                continue
            if use_y and x[i] <= 1e-12 and len(np.where(x > 1e-12)[0]) >= k_use:
                continue
            take = min(cap, rem)
            x[i] += take
            rem -= take
            if rem <= 1e-12:
                break

    # Normalize tiny numeric drift
    s = float(np.sum(x))
    if s > 1e-12:
        x /= s
    else:
        x[:] = 1.0 / n
    return x


def add_optional_linear_constraints(model: gp.Model, x, instance: dict):
    # Robust support for optional linear rules not explicitly in schema
    # Format 1: instance["linear_constraints"] = [{"a": [...], "lb": v, "ub": v}, ...]
    if "linear_constraints" in instance and instance["linear_constraints"] is not None:
        for j, c in enumerate(instance["linear_constraints"]):
            a = np.array(c.get("a", c.get("coefficients")), dtype=float)
            lb = c.get("lb", c.get("lower_bound", -GRB.INFINITY))
            ub = c.get("ub", c.get("upper_bound", GRB.INFINITY))
            expr = a @ x
            if lb is not None and lb != -math.inf:
                model.addConstr(expr >= float(lb), name=f"lin_lb_{j}")
            if ub is not None and ub != math.inf:
                model.addConstr(expr <= float(ub), name=f"lin_ub_{j}")

    # Format 2: instance["constraints"]["A"], ["lb"], ["ub"]
    cons = instance.get("constraints", {})
    if all(k in cons for k in ["A", "lb", "ub"]):
        A = np.array(cons["A"], dtype=float)
        lb = np.array(cons["lb"], dtype=float)
        ub = np.array(cons["ub"], dtype=float)
        m = A.shape[0]
        for j in range(m):
            expr = A[j] @ x
            if not np.isneginf(lb[j]):
                model.addConstr(expr >= float(lb[j]), name=f"A_lb_{j}")
            if not np.isposinf(ub[j]):
                model.addConstr(expr <= float(ub[j]), name=f"A_ub_{j}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    n = int(instance["n"])
    k = int(instance["k"])
    gamma = float(instance["gamma"])
    kappa = float(instance.get("kappa", 1))
    mu = np.array(instance["mu"], dtype=float)

    cons = instance.get("constraints", {})
    has_min_ret = bool(cons.get("has_min_return_constraint", False))
    r_bar = cons.get("r_bar", None)
    has_min_inv = bool(cons.get("has_min_investment_constraint", False))
    l_min = cons.get("l_min_investment", None)
    u_max = cons.get("u_max_investment", None)

    l_min_arr = np.array(l_min, dtype=float) if l_min is not None else None
    u_max_arr = np.array(u_max, dtype=float) if u_max is not None else None

    cov = build_covariance(instance)
    Q = cov.copy()
    if gamma > 1e-12:
        Q = Q + np.eye(n) / gamma

    model = gp.Model("sparse_portfolio")
    model.Params.TimeLimit = int(args.time_limit)
    model.Params.Threads = 1
    model.Params.OutputFlag = 0

    x = model.addMVar(n, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="x")

    use_y = (k < n) or has_min_inv
    y = None
    if use_y:
        y = model.addMVar(n, vtype=GRB.BINARY, name="y")

    model.addConstr(np.ones(n) @ x == 1.0, name="budget")

    if use_y:
        ub_vec = np.ones(n) if u_max_arr is None else u_max_arr
        for i in range(n):
            model.addConstr(x[i] <= float(ub_vec[i]) * y[i], name=f"link_ub_{i}")
            if has_min_inv and l_min_arr is not None:
                model.addConstr(x[i] >= float(l_min_arr[i]) * y[i], name=f"link_lb_{i}")
        model.addConstr(np.ones(n) @ y <= float(k), name="cardinality")
    else:
        if u_max_arr is not None:
            for i in range(n):
                model.addConstr(x[i] <= float(u_max_arr[i]), name=f"x_ub_{i}")

    if has_min_ret and r_bar is not None:
        model.addConstr(mu @ x >= float(r_bar), name="min_return")

    add_optional_linear_constraints(model, x, instance)

    obj = 0.5 * (x @ Q @ x) - kappa * (mu @ x)
    model.setObjective(obj, GRB.MINIMIZE)

    # Optional warm start
    x0 = try_construct_fallback(
        n=n, k=k, mu=mu, use_y=use_y, lmin=l_min_arr, umax=u_max_arr
    )
    try:
        x.Start = x0
        if use_y:
            y0 = (x0 > 1e-10).astype(float)
            y.Start = y0
    except Exception:
        pass

    best_logged = [float("inf")]

    def cb(m, where):
        if where == GRB.Callback.MIPSOL and logger is not None:
            v = float(m.cbGet(GRB.Callback.MIPSOL_OBJ))
            if v + 1e-12 < best_logged[0]:
                best_logged[0] = v
                logger.log(v)

    if use_y:
        model.optimize(cb)
    else:
        model.optimize()

    if model.SolCount > 0:
        x_sol = np.array(x.X, dtype=float)
        obj_val = compute_objective(x_sol, cov, gamma, kappa, mu)
        if logger is not None and obj_val + 1e-12 < best_logged[0]:
            logger.log(float(obj_val))
    else:
        x_sol = try_construct_fallback(
            n=n, k=k, mu=mu, use_y=use_y, lmin=l_min_arr, umax=u_max_arr
        )
        obj_val = compute_objective(x_sol, cov, gamma, kappa, mu)
        if logger is not None:
            logger.log(float(obj_val))

    out = {
        "objective_value": float(obj_val),
        "x": [float(v) for v in x_sol.tolist()],
    }

    with open(args.solution_path, "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()
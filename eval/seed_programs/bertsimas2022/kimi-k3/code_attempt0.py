import argparse
import json
import math
import time

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Cardinality-constrained mean-variance portfolio optimization."
    )
    parser.add_argument("--instance_path", required=True,
                        help="Path to the JSON problem instance.")
    parser.add_argument("--solution_path", required=True,
                        help="Path where the solution JSON is written.")
    parser.add_argument("--time_limit", type=int, required=True,
                        help="Maximum runtime in seconds.")
    parser.add_argument("--log_path", default=None,
                        help="Optional path to a JSONL convergence log.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Instance loading
# ---------------------------------------------------------------------------
def load_instance(path):
    with open(path, "r") as f:
        data = json.load(f)

    n = int(data["n"])
    k = int(data["k"])
    kappa = float(data.get("kappa", 1))
    mu = np.array(data.get("mu", [0.0] * n), dtype=float)

    gamma = data.get("gamma", None)
    if gamma is None:
        formula = data.get("gamma_formula") or "1.0/n"
        gamma = eval(str(formula), {"__builtins__": {}},
                     {"n": float(n), "sqrt": math.sqrt})
    gamma = float(gamma)
    if gamma <= 0.0:
        gamma = 1.0 / max(1, n)

    fl = data.get("factor_loadings", None)
    if fl is None or int(data.get("r", 0)) == 0:
        F = np.zeros((n, 0))
    else:
        F = np.array(fl, dtype=float)
        if F.ndim != 2:
            F = np.zeros((n, 0))
        elif F.shape[0] != n and F.shape[1] == n:
            F = F.T
        if F.shape[0] != n:
            F = np.zeros((n, 0))
    r = F.shape[1]

    d = np.array(data.get("idiosyncratic_variance", [0.0] * n), dtype=float)
    if d.shape[0] != n:
        d = np.zeros(n)

    cons = data.get("constraints", {}) or {}
    has_min_ret = bool(cons.get("has_min_return_constraint", False))
    r_bar = cons.get("r_bar", None)
    if not has_min_ret or r_bar is None:
        has_min_ret, r_bar = False, None
    else:
        r_bar = float(r_bar)

    has_min_inv = bool(cons.get("has_min_investment_constraint", False))
    l_min = cons.get("l_min_investment", None)
    if has_min_inv and l_min is not None:
        l_min = np.array(l_min, dtype=float)
    else:
        l_min, has_min_inv = None, False

    u_max = cons.get("u_max_investment", None)
    u_max = np.array(u_max, dtype=float) if u_max is not None else None

    return {
        "n": n, "k": k, "kappa": kappa, "gamma": gamma, "mu": mu,
        "F": F, "r": r, "d": d,
        "has_min_ret": has_min_ret, "r_bar": r_bar,
        "l_min": l_min, "u_max": u_max,
    }


# ---------------------------------------------------------------------------
# True objective: 0.5 x' Sigma x + ||x||^2 / (2 gamma) - kappa * mu' x
# with Sigma = F F' + diag(d)
# ---------------------------------------------------------------------------
def compute_objective(x, F, d, gamma, kappa, mu):
    x = np.asarray(x, dtype=float)
    sys_risk = 0.0
    if F.shape[1] > 0:
        t = F.T.dot(x)
        sys_risk = float(t.dot(t))
    var = sys_risk + float(d.dot(x * x))
    ridge = float(x.dot(x)) / (2.0 * gamma)
    ret = float(mu.dot(x))
    return 0.5 * var + ridge - kappa * ret


# ---------------------------------------------------------------------------
# Feasibility check
# ---------------------------------------------------------------------------
def check_feasible(x, k, mu, l_min, u_max, has_min_ret, r_bar, tol=1e-6):
    x = np.asarray(x, dtype=float)
    if abs(float(x.sum()) - 1.0) > tol:
        return False
    if float(x.min()) < -tol:
        return False
    u = u_max if u_max is not None else np.ones(len(x))
    if np.any(x > u + tol):
        return False
    pos = x > tol
    if int(np.count_nonzero(pos)) > k:
        return False
    if l_min is not None and np.any(x[pos] < l_min[pos] - tol):
        return False
    if has_min_ret and float(mu.dot(x)) < r_bar - tol:
        return False
    return True


# ---------------------------------------------------------------------------
# Greedy feasible construction (used as incumbent + MIP start)
# ---------------------------------------------------------------------------
def greedy_feasible(n, k, mu, l_min, u_max, has_min_ret, r_bar):
    m = int(min(k, n))
    if m <= 0:
        return None
    u = u_max if u_max is not None else np.ones(n)
    l = l_min if l_min is not None else np.zeros(n)

    order = list(np.argsort(-mu))
    S = order[:m]
    # Ensure the sum of lower bounds on the selected set fits the budget.
    while len(S) > 1 and sum(l[i] for i in S) > 1.0 - 1e-12:
        victim = max(S, key=lambda i: (l[i], -mu[i]))
        S.remove(victim)

    x = np.zeros(n)
    for i in S:
        x[i] = l[i]
    residual = 1.0 - float(x.sum())
    if residual < -1e-9:
        return None

    for i in sorted(S, key=lambda i: -mu[i]):
        if residual <= 1e-12:
            break
        room = u[i] - x[i]
        if room <= 0.0:
            continue
        add = min(room, residual)
        x[i] += add
        residual -= add

    if residual > 1e-9 and len(S) < m:
        in_S = set(S)
        for i in order:
            if residual <= 1e-12 or len(S) >= m:
                break
            if i in in_S:
                continue
            base, cap = l[i], u[i]
            if cap < base - 1e-12:
                continue
            add = min(cap, residual)
            if add < base - 1e-12:
                continue
            x[i] = add
            S.append(i)
            in_S.add(i)
            residual -= add

    if check_feasible(x, k, mu, l_min, u_max, has_min_ret, r_bar):
        return x
    return None


# ---------------------------------------------------------------------------
# Polish a solver solution: zero numerical noise and restore sum == 1
# ---------------------------------------------------------------------------
def polish(x, l_min, u_vec):
    x = np.array(x, dtype=float)
    x[np.abs(x) < 1e-10] = 0.0
    x = np.maximum(x, 0.0)
    x = np.minimum(x, u_vec)
    diff = 1.0 - float(x.sum())
    if abs(diff) > 1e-12:
        if diff > 0.0:
            for i in np.argsort(-x):
                if x[i] <= 0.0:
                    continue
                room = u_vec[i] - x[i]
                if room <= 0.0:
                    continue
                add = min(room, diff)
                x[i] += add
                diff -= add
                if diff <= 1e-12:
                    break
        else:
            for i in np.argsort(-x):
                if x[i] <= 0.0:
                    continue
                floor = l_min[i] if l_min is not None else 0.0
                red = min(x[i] - floor, -diff)
                if red > 0.0:
                    x[i] -= red
                    diff += red
                if diff >= -1e-12:
                    break
    return x


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    t_start = time.time()
    time_limit = max(1, int(args.time_limit))
    hard_stop = t_start + time_limit - 0.5

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    inst = load_instance(args.instance_path)
    n, k = inst["n"], inst["k"]
    kappa, gamma, mu = inst["kappa"], inst["gamma"], inst["mu"]
    F, r, d = inst["F"], inst["r"], inst["d"]
    has_min_ret, r_bar = inst["has_min_ret"], inst["r_bar"]
    l_min, u_max = inst["l_min"], inst["u_max"]

    u_vec = np.maximum(u_max, 0.0) if u_max is not None else np.ones(n)

    best = {"obj": math.inf, "x": None}

    def record_incumbent(obj, xs):
        if obj < best["obj"] - 1e-9:
            best["obj"] = float(obj)
            best["x"] = np.array(xs, dtype=float)
            if logger is not None:
                logger.log_solution(float(obj), {
                    "objective_value": float(obj),
                    "x": [float(v) for v in best["x"]],
                })

    # ---- Greedy incumbent / MIP start ------------------------------------
    x0 = greedy_feasible(n, k, mu, l_min, u_max, has_min_ret, r_bar)
    if x0 is not None:
        record_incumbent(compute_objective(x0, F, d, gamma, kappa, mu), x0)

    # ---- Gurobi MIQP ------------------------------------------------------
    try:
        remaining = hard_stop - time.time()
        if remaining > 0.5:
            model = gp.Model("cardinality_portfolio")
            model.Params.Seed = 0
            model.Params.MIPGap = 1e-4
            model.Params.NumericFocus = 0
            model.Params.Threads = 1
            model.Params.OutputFlag = 0
            model.Params.TimeLimit = max(0.5, remaining - 0.3)

            x = model.addMVar(n, lb=0.0, ub=u_vec, name="x")
            model.addConstr(x.sum() == 1.0, name="budget")

            need_bin = (k < n) or (l_min is not None)
            y = None
            if need_bin:
                y = model.addMVar(n, vtype=GRB.BINARY, name="y")
                model.addConstr(y.sum() <= float(k), name="cardinality")
                model.addConstrs(
                    (x[i] <= u_vec[i] * y[i] for i in range(n)), name="ub_link")
                if l_min is not None:
                    model.addConstrs(
                        (x[i] >= l_min[i] * y[i] for i in range(n)),
                        name="lb_link")

            if has_min_ret:
                model.addConstr(mu @ x >= r_bar, name="min_return")

            # Factor-model reformulation: z = F' x, so x' F F' x = z' z.
            z = None
            if r > 0:
                zlb = F.min(axis=0)
                zub = F.max(axis=0)
                z = model.addMVar(r, lb=zlb, ub=zub, name="z")
                model.addConstrs(
                    (z[j] == F[:, j] @ x for j in range(r)), name="factor")

            coef = d + 1.0 / gamma
            obj = gp.quicksum(0.5 * coef[i] * x[i] * x[i] for i in range(n))
            if z is not None:
                obj = obj + 0.5 * (z @ z)
            if kappa != 0.0:
                obj = obj - kappa * (mu @ x)
            model.setObjective(obj, GRB.MINIMIZE)

            if x0 is not None:
                try:
                    x.Start = x0
                    if y is not None:
                        y0 = np.zeros(n)
                        y0[x0 > 1e-9] = 1.0
                        y.Start = y0
                except Exception:
                    pass

            def callback(m, where):
                if time.time() > hard_stop:
                    m.cbTerminate()
                    return
                if where == GRB.Callback.MIPSOL:
                    try:
                        xv = np.array(m.cbGetSolution(m._xvars), dtype=float)
                    except Exception:
                        return
                    record_incumbent(
                        compute_objective(xv, F, d, gamma, kappa, mu), xv)

            model._xvars = x
            model.optimize(callback)

            if model.SolCount > 0:
                try:
                    xv = np.array(x.X, dtype=float)
                    record_incumbent(
                        compute_objective(xv, F, d, gamma, kappa, mu), xv)
                except Exception:
                    pass
    except Exception:
        pass

    # ---- Emergency fallback (should not normally be needed) ---------------
    if best["x"] is None:
        m = max(1, min(k, n))
        idx = np.argsort(-mu)[:m]
        xs = np.zeros(n)
        xs[idx] = 1.0 / m
        record_incumbent(compute_objective(xs, F, d, gamma, kappa, mu), xs)

    # ---- Final answer -----------------------------------------------------
    xf = polish(best["x"], l_min, u_vec)
    obj_f = compute_objective(xf, F, d, gamma, kappa, mu)
    solution = {
        "objective_value": float(obj_f),
        "x": [float(v) for v in xf],
    }
    with open(args.solution_path, "w") as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()
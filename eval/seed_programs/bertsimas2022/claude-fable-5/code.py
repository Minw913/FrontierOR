import argparse
import json
import time
import numpy as np
import scipy.sparse as sp
import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except Exception:
    SolutionLogger = None


def compute_objective(xs, F, q, mu, kappa):
    """Objective = 0.5 x' (FF'+D) x + (1/(2 gamma)) x'x - kappa * mu'x
       where q = d + 1/gamma (diagonal part combining idiosyncratic var and ridge)."""
    y = F.T @ xs if F.shape[1] > 0 else np.zeros(0)
    return 0.5 * float(y @ y) + 0.5 * float(np.sum(q * xs * xs)) - kappa * float(np.dot(mu, xs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, default=300)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    t_start = time.time()

    logger = SolutionLogger(args.log_path, sense="minimize") if (args.log_path and SolutionLogger) else None

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    n = int(data["n"])
    k = int(data["k"])
    r = int(data.get("r", 0))
    gamma = float(data["gamma"])
    kappa = float(data.get("kappa", 1))
    mu = np.array(data["mu"], dtype=float)

    fl = data.get("factor_loadings", None)
    if fl is not None and r > 0:
        F = np.array(fl, dtype=float)
        if F.shape == (r, n) and r != n:
            F = F.T
        F = F.reshape(n, -1)
    else:
        F = np.zeros((n, 0))
    r_eff = F.shape[1]

    d = np.array(data["idiosyncratic_variance"], dtype=float)
    q = d + 1.0 / gamma  # combined diagonal quadratic coefficient

    cons = data.get("constraints", {}) or {}
    has_min_ret = bool(cons.get("has_min_return_constraint", False))
    r_bar = cons.get("r_bar", None)
    has_min_inv = bool(cons.get("has_min_investment_constraint", False))
    l_min = cons.get("l_min_investment", None)
    u_max = cons.get("u_max_investment", None)

    if u_max is not None:
        u = np.minimum(np.array(u_max, dtype=float), 1.0)
    else:
        u = np.ones(n)
    if has_min_inv and l_min is not None:
        l = np.array(l_min, dtype=float)
    else:
        l = None

    best_obj = float("inf")
    best_x = None

    def record(xs):
        nonlocal best_obj, best_x
        obj = compute_objective(xs, F, q, mu, kappa)
        if obj < best_obj - 1e-12:
            best_obj = obj
            best_x = xs.copy()
            if logger:
                logger.log_solution(obj, {"objective_value": obj, "x": [float(v) for v in xs]})
            return True
        return False

    def remaining():
        return args.time_limit - (time.time() - t_start)

    # ---------- Heuristic warm start via continuous relaxation ----------
    start_x = None
    start_z = None
    try:
        if remaining() > 3:
            mr = gp.Model("relax")
            mr.Params.OutputFlag = 0
            mr.Params.Seed = 0
            mr.Params.Threads = 1
            mr.Params.NumericFocus = 0
            mr.Params.TimeLimit = max(1.0, min(15.0, remaining() * 0.2))
            xr = mr.addMVar(n, lb=0.0, ub=u)
            yr = mr.addMVar(r_eff, lb=-GRB.INFINITY) if r_eff > 0 else None
            mr.addConstr(xr.sum() == 1)
            if r_eff > 0:
                mr.addConstr(F.T @ xr - yr == 0)
            if has_min_ret and r_bar is not None:
                mr.addConstr(mu @ xr >= float(r_bar))
            Dq = sp.diags(q)
            objr = 0.5 * (xr @ Dq @ xr) - kappa * (mu @ xr)
            if r_eff > 0:
                objr = objr + 0.5 * (yr @ yr)
            mr.setObjective(objr, GRB.MINIMIZE)
            mr.optimize()
            if mr.SolCount > 0:
                xv = np.array(xr.X)
                # choose top-k support
                idx = np.argsort(-xv)[:min(k, n)]
                S = set(int(i) for i in idx)
                # restricted QP on support S
                mh = gp.Model("restr")
                mh.Params.OutputFlag = 0
                mh.Params.Seed = 0
                mh.Params.Threads = 1
                mh.Params.TimeLimit = max(1.0, min(10.0, remaining() * 0.2))
                ub_h = np.array([u[i] if i in S else 0.0 for i in range(n)])
                lb_h = np.zeros(n)
                if l is not None:
                    lb_h = np.array([l[i] if i in S else 0.0 for i in range(n)])
                xh = mh.addMVar(n, lb=lb_h, ub=ub_h)
                yh = mh.addMVar(r_eff, lb=-GRB.INFINITY) if r_eff > 0 else None
                mh.addConstr(xh.sum() == 1)
                if r_eff > 0:
                    mh.addConstr(F.T @ xh - yh == 0)
                if has_min_ret and r_bar is not None:
                    mh.addConstr(mu @ xh >= float(r_bar))
                objh = 0.5 * (xh @ Dq @ xh) - kappa * (mu @ xh)
                if r_eff > 0:
                    objh = objh + 0.5 * (yh @ yh)
                mh.setObjective(objh, GRB.MINIMIZE)
                mh.optimize()
                if mh.SolCount > 0:
                    xs = np.array(xh.X)
                    xs = np.maximum(xs, 0.0)
                    record(xs)
                    start_x = xs
                    start_z = np.array([1.0 if i in S else 0.0 for i in range(n)])
    except Exception:
        pass

    # ---------- Full MIQP ----------
    try:
        if remaining() > 2:
            m = gp.Model("miqp")
            m.Params.OutputFlag = 0
            m.Params.Seed = 0
            m.Params.MIPGap = 1e-4
            m.Params.NumericFocus = 0
            m.Params.Threads = 1
            m.Params.TimeLimit = max(1.0, remaining() - 2.0)

            x = m.addMVar(n, lb=0.0, ub=u, name="x")
            z = m.addMVar(n, vtype=GRB.BINARY, name="z")
            y = m.addMVar(r_eff, lb=-GRB.INFINITY, name="y") if r_eff > 0 else None

            m.addConstr(x.sum() == 1)
            m.addConstr(z.sum() <= k)
            m.addConstr(x <= u * z)
            if l is not None:
                m.addConstr(x >= l * z)
            if r_eff > 0:
                m.addConstr(F.T @ x - y == 0)
            if has_min_ret and r_bar is not None:
                m.addConstr(mu @ x >= float(r_bar))

            Dq = sp.diags(q)
            obj = 0.5 * (x @ Dq @ x) - kappa * (mu @ x)
            if r_eff > 0:
                obj = obj + 0.5 * (y @ y)
            m.setObjective(obj, GRB.MINIMIZE)

            if start_x is not None:
                x.Start = start_x
                z.Start = start_z

            m._x = x

            def cb(model, where):
                if where == GRB.Callback.MIPSOL:
                    try:
                        xs = np.array(model.cbGetSolution(model._x))
                        xs = np.maximum(xs, 0.0)
                        record(xs)
                    except Exception:
                        pass

            m.optimize(cb)

            if m.SolCount > 0:
                xs = np.maximum(np.array(x.X), 0.0)
                record(xs)
    except Exception:
        pass

    # ---------- Fallback ----------
    if best_x is None:
        # simple fallback: equal weight on top-k by mu (respecting bounds loosely)
        idx = np.argsort(-mu)[:min(k, n)]
        xs = np.zeros(n)
        xs[idx] = 1.0 / len(idx)
        record(xs)

    # normalize tiny numerical noise
    xs = np.maximum(best_x, 0.0)
    s = xs.sum()
    if s > 0:
        pass  # keep as solved; renormalizing could violate bounds
    final_obj = compute_objective(xs, F, q, mu, kappa)

    sol = {"objective_value": float(final_obj), "x": [float(v) for v in xs]}
    with open(args.solution_path, "w") as f:
        json.dump(sol, f)


if __name__ == "__main__":
    main()
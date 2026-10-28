import argparse
import json
import time

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def main():
    parser = argparse.ArgumentParser(
        description="Cardinality-constrained mean-variance portfolio optimization"
    )
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    t_start = time.time()
    time_limit = max(1, int(args.time_limit))

    with open(args.instance_path, "r") as fh:
        inst = json.load(fh)

    n = int(inst["n"])
    k = min(int(inst["k"]), n)
    b = float(inst["b"])
    r = np.asarray(inst["r"], dtype=float).reshape(-1)
    Q = np.asarray(inst["Q"], dtype=float)
    Q = 0.5 * (Q + Q.T)  # ensure symmetry

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    state = {"obj": float("inf"), "x": None}

    def objval(x):
        return float(0.5 * (x @ Q @ x))

    def record(x):
        """Feasibility check + incumbent update + convergence logging."""
        if x is None:
            return None
        x = np.asarray(x, dtype=float)
        if x.shape[0] != n or not np.all(np.isfinite(x)):
            return None
        if abs(float(x.sum()) - 1.0) > 1e-6:
            return None
        if float(r @ x) < b - 1e-6:
            return None
        if int(np.count_nonzero(np.abs(x) > 1e-7)) > k:
            return None
        obj = objval(x)
        if obj < state["obj"] - 1e-9:
            state["obj"] = obj
            state["x"] = x.copy()
            if logger is not None:
                logger.log_solution(
                    float(obj),
                    {"objective_value": float(obj), "x": [float(v) for v in x]},
                )
        return obj

    # ---------- Analytic QP solvers (no cardinality) ----------
    def solve_qp(Qs, rs):
        """min 0.5 x'Qs x  s.t. sum(x)=1, rs'x>=b ; Qs SPD."""
        m = rs.shape[0]
        ones = np.ones(m)
        qinv1 = np.linalg.solve(Qs, ones)
        A = float(ones @ qinv1)
        x = qinv1 / A
        if float(rs @ x) >= b - 1e-12:
            return x
        qinvr = np.linalg.solve(Qs, rs)
        B = float(ones @ qinvr)
        C = float(rs @ qinvr)
        lam, mu = np.linalg.solve(
            np.array([[A, B], [B, C]]), np.array([1.0, b])
        )
        return lam * qinv1 + mu * qinvr

    def solve_subset(S):
        idx = sorted(S)
        if not idx:
            return None
        try:
            xs = solve_qp(Q[np.ix_(idx, idx)], r[idx])
        except np.linalg.LinAlgError:
            return None
        x = np.zeros(n)
        x[idx] = xs
        x[np.abs(x) < 1e-12] = 0.0
        return x

    def polish(x):
        """Project a (possibly slightly dense/noisy) vector onto <=k support
        and re-solve the restricted QP exactly."""
        S = [i for i in range(n) if abs(x[i]) > 1e-7]
        if len(S) > k:
            S = sorted(S, key=lambda i: -abs(x[i]))[:k]
        if not S:
            return None
        return solve_subset(S)

    # ---------- Full (no cardinality) solution ----------
    x_full = None
    if n <= 2000:
        try:
            x_full = solve_qp(Q, r)
        except np.linalg.LinAlgError:
            x_full = None

    # If cardinality constraint is non-binding, the analytic QP is optimal.
    if k >= n:
        record(x_full)
        if state["x"] is None:
            state["x"] = np.ones(n) / n
            state["obj"] = objval(state["x"])
        with open(args.solution_path, "w") as fh:
            json.dump(
                {"objective_value": float(state["obj"]),
                 "x": [float(v) for v in state["x"]]},
                fh,
            )
        return

    # ---------- Heuristic incumbent: greedy subset + swap local search ----------
    heuristic_deadline = time.time() + min(0.25 * time_limit, 30.0)

    cand_subsets = []
    if x_full is not None:
        cand_subsets.append(np.argsort(-np.abs(x_full))[:k].tolist())
    if k >= 1:
        cand_subsets.append(np.argsort(-r)[:k].tolist())

    best_S, best_f = None, float("inf")
    for S in cand_subsets:
        x = solve_subset(S)
        f = record(x)
        if f is not None and f < best_f:
            best_f, best_S = f, S

    if best_S is not None:
        if x_full is not None:
            sc = np.abs(x_full)
            sc = sc / sc.max() if sc.max() > 0 else sc
            rr = (r - r.min()) / (r.max() - r.min()) if r.max() > r.min() else np.zeros(n)
            priority = sc + rr
        else:
            priority = (r - r.min()) / (r.max() - r.min()) if r.max() > r.min() else np.zeros(n)
        order = np.argsort(-priority).tolist()
        cache = {}

        def eval_S(Sf):
            key = frozenset(Sf)
            fval = cache.get(key)
            if fval is None:
                xs = solve_subset(key)
                fval = objval(xs) if xs is not None else float("inf")
                if len(cache) < 200000:
                    cache[key] = fval
            return fval

        S = frozenset(best_S)
        cur_f = eval_S(S)
        improved = True
        while improved and time.time() < heuristic_deadline:
            improved = False
            x_cur = solve_subset(S)
            w = np.abs(x_cur) if x_cur is not None else np.ones(n)
            inside = sorted(S, key=lambda i: w[i])
            outside = [j for j in order if j not in S][:250]
            for i in inside:
                if time.time() >= heuristic_deadline:
                    break
                for j in outside:
                    if time.time() >= heuristic_deadline:
                        break
                    f2 = eval_S((S - {i}) | {j})
                    if f2 < cur_f - 1e-9:
                        S = (S - {i}) | {j}
                        cur_f = f2
                        record(solve_subset(S))
                        improved = True
                        break
                if improved:
                    break

    # ---------- Gurobi MIQP with indicator constraints ----------
    remaining = time_limit - (time.time() - t_start)
    if remaining > 0.3:
        model = gp.Model("cardinality_portfolio")
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1
        model.Params.LogToConsole = 0
        model.Params.TimeLimit = remaining

        if n <= 1500:
            try:
                np.linalg.cholesky(Q)
            except np.linalg.LinAlgError:
                model.Params.NonConvex = 2

        X = model.addMVar(n, lb=-GRB.INFINITY, name="x")
        Z = model.addMVar(n, vtype=GRB.BINARY, name="z")

        model.addConstr(X.sum() == 1.0, name="budget")
        model.addConstr(X @ r >= b, name="target_return")
        model.addConstr(Z.sum() <= float(k), name="cardinality")
        for i in range(n):
            # z_i = 0  =>  x_i = 0
            model.addGenConstrIndicator(Z[i], False, X[i] == 0.0)

        model.setObjective(0.5 * (X @ Q @ X), GRB.MINIMIZE)

        xvars = [X[i] for i in range(n)]

        # MIP start from heuristic incumbent.
        if state["x"] is not None:
            bx = state["x"].copy()
            bx[np.abs(bx) < 1e-9] = 0.0
            for i in range(n):
                xvars[i].Start = float(bx[i])
                Z[i].Start = 1.0 if bx[i] != 0.0 else 0.0

        model._xvars = xvars

        def callback(mdl, where):
            if where == GRB.Callback.MIPSOL:
                try:
                    xv = np.asarray(mdl.cbGetSolution(mdl._xvars), dtype=float)
                    record(polish(xv))
                except Exception:
                    pass

        try:
            model.optimize(callback)
        except gp.GurobiError:
            # e.g. numerical non-convexity of Q: retry with NonConvex=2
            try:
                model.Params.NonConvex = 2
                model.Params.TimeLimit = max(
                    time_limit - (time.time() - t_start), 0.0
                )
                model.optimize(callback)
            except gp.GurobiError:
                pass

        if model.SolCount > 0:
            try:
                xv = np.asarray([v.X for v in xvars], dtype=float)
                record(polish(xv))
            except Exception:
                pass

    # ---------- Final fallback and output ----------
    if state["x"] is None:
        x = solve_subset(np.argsort(-r)[: max(1, k)].tolist())
        if x is None:
            x = np.zeros(n)
            x[int(np.argmax(r))] = 1.0
        state["x"] = x
        state["obj"] = objval(x)

    x_out = [float(v) for v in state["x"]]
    obj_out = float(objval(state["x"]))
    with open(args.solution_path, "w") as fh:
        json.dump({"objective_value": obj_out, "x": x_out}, fh)


if __name__ == "__main__":
    main()
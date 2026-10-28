import argparse
import itertools
import json
import math
import os
import time

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    start_time = time.monotonic()
    deadline = start_time + max(0, args.time_limit)

    with open(args.instance_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    n = int(data["n"])
    k = min(max(int(data["k"]), 0), n)
    b = float(data["b"])
    r = np.asarray(data["r"], dtype=float)
    Q = np.asarray(data["Q"], dtype=float)
    Q = 0.5 * (Q + Q.T)

    best_obj = math.inf
    best_x = None
    best_support = None

    def objective(x):
        return 0.5 * float(x @ Q @ x)

    def make_solution(x, obj=None):
        if obj is None:
            obj = objective(x)
        clean = np.asarray(x, dtype=float).copy()
        clean[np.abs(clean) < 1e-14] = 0.0
        return {
            "objective_value": float(obj),
            "x": [float(v) for v in clean],
        }

    def solve_support(support):
        """Exactly solve the convex QP restricted to the supplied support."""
        support = tuple(sorted(set(int(i) for i in support)))
        m = len(support)
        if m == 0:
            return None

        idx = np.asarray(support, dtype=int)
        QS = Q[np.ix_(idx, idx)]
        rs = r[idx]
        ones = np.ones(m)

        try:
            inv_one = np.linalg.solve(QS, ones)
        except np.linalg.LinAlgError:
            inv_one = np.linalg.lstsq(QS, ones, rcond=None)[0]

        denom = float(ones @ inv_one)
        if not np.isfinite(denom) or denom <= 0.0:
            return None

        xs = inv_one / denom
        ret = float(rs @ xs)
        ret_tol = 1e-10 * max(1.0, abs(b), float(np.max(np.abs(rs))))

        if ret < b - ret_tol:
            C = np.column_stack((ones, rs))
            try:
                inv_C = np.linalg.solve(QS, C)
            except np.linalg.LinAlgError:
                inv_C = np.linalg.lstsq(QS, C, rcond=None)[0]

            gram = C.T @ inv_C
            rhs = np.array([1.0, b], dtype=float)
            try:
                multipliers = np.linalg.solve(gram, rhs)
            except np.linalg.LinAlgError:
                multipliers = np.linalg.lstsq(gram, rhs, rcond=None)[0]

            xs = inv_C @ multipliers

        # Repair tiny numerical errors in the two linear constraints.
        xs = np.asarray(xs, dtype=float)
        xs += (1.0 - float(np.sum(xs))) / m

        current_return = float(rs @ xs)
        if current_return < b:
            direction = rs - float(np.mean(rs))
            gain = float(rs @ direction)
            if gain <= 1e-18 * max(1.0, float(rs @ rs)):
                return None
            xs += ((b - current_return) / gain) * direction
            xs += (1.0 - float(np.sum(xs))) / m

        if not np.all(np.isfinite(xs)):
            return None
        if abs(float(np.sum(xs)) - 1.0) > 1e-7:
            return None
        if float(rs @ xs) < b - 1e-8 * max(1.0, abs(b)):
            return None

        x = np.zeros(n, dtype=float)
        x[idx] = xs
        return x

    def consider(x, support=None):
        nonlocal best_obj, best_x, best_support
        if x is None or not np.all(np.isfinite(x)):
            return False

        obj = objective(x)
        tolerance = 1e-11 * max(1.0, abs(best_obj)) if math.isfinite(best_obj) else 0.0
        if obj < best_obj - tolerance:
            best_obj = obj
            best_x = np.asarray(x, dtype=float).copy()
            if support is None:
                support = np.flatnonzero(np.abs(best_x) > 1e-10)
            best_support = tuple(int(i) for i in support)
            sol = make_solution(best_x, best_obj)
            if logger:
                logger.log_solution(best_obj, sol)
            return True
        return False

    # Fast initial feasible solutions.
    if k >= 1:
        for i in range(n):
            if r[i] >= b - 1e-12 * max(1.0, abs(b)):
                x = np.zeros(n)
                x[i] = 1.0
                consider(x, (i,))

    if k >= 2 and n >= 2:
        order = np.argsort(r)
        pair_candidates = {
            (int(order[0]), int(order[-1])),
            (int(order[0]), int(order[1])),
            (int(order[-2]), int(order[-1])),
        }
        for pair in pair_candidates:
            if pair[0] != pair[1]:
                consider(solve_support(pair), pair)

    # Use the largest unconstrained solution to suggest a sparse support.
    if k >= 1 and time.monotonic() < deadline:
        full = solve_support(range(n))
        if full is not None:
            top = np.argsort(-np.abs(full))[:k]
            consider(solve_support(top), top)

    # Greedily enlarge the best support, testing each possible addition.
    heuristic_deadline = min(deadline, start_time + min(2.0, 0.15 * max(0, args.time_limit)))
    if best_support is not None and k > len(best_support):
        current = set(best_support)
        while len(current) < k and time.monotonic() < heuristic_deadline:
            chosen = None
            chosen_x = None
            chosen_obj = math.inf
            for j in range(n):
                if time.monotonic() >= heuristic_deadline:
                    break
                if j in current:
                    continue
                support = tuple(current | {j})
                candidate = solve_support(support)
                if candidate is not None:
                    obj = objective(candidate)
                    if obj < chosen_obj:
                        chosen_obj = obj
                        chosen = j
                        chosen_x = candidate
            if chosen is None:
                break
            current.add(chosen)
            consider(chosen_x, tuple(current))

    # Enumerate all maximum-size supports when the combinatorial space is small.
    exhaustive = False
    if k >= 1:
        combination_count = math.comb(n, k)
        enum_limit = 50000
        enum_deadline = min(deadline, start_time + max(0.25, 0.30 * max(0, args.time_limit)))
        if combination_count <= enum_limit:
            exhaustive = True
            for support in itertools.combinations(range(n), k):
                if time.monotonic() >= enum_deadline:
                    exhaustive = False
                    break
                consider(solve_support(support), support)

    # If enumeration completed, every support of cardinality <= k is represented
    # inside a k-element support, so the best solution is globally optimal.
    if not exhaustive and k >= 1 and time.monotonic() < deadline:
        try:
            model = gp.Model("cardinality_portfolio")
            model.Params.OutputFlag = 0
            model.Params.Seed = 0
            model.Params.MIPGap = 1e-4
            model.Params.NumericFocus = 0
            model.Params.Threads = 1
            model.Params.TimeLimit = max(0.001, deadline - time.monotonic())

            xvar = model.addMVar(n, lb=-GRB.INFINITY, name="x")
            zvar = model.addMVar(n, vtype=GRB.BINARY, name="z")

            model.addConstr(xvar.sum() == 1.0, name="budget")
            model.addConstr(r @ xvar >= b, name="return")
            model.addConstr(zvar.sum() <= k, name="cardinality")

            for i in range(n):
                model.addGenConstrIndicator(
                    zvar[i], False, xvar[i] == 0.0, name=f"zero_{i}"
                )

            model.setObjective(0.5 * (xvar @ Q @ xvar), GRB.MINIMIZE)

            if best_x is not None:
                active = np.abs(best_x) > 1e-10
                xvar.Start = best_x
                zvar.Start = active.astype(float)

            callback_state = {"last_solver_obj": math.inf}

            def callback(cb_model, where):
                if where != GRB.Callback.MIPSOL:
                    return
                try:
                    solver_obj = float(cb_model.cbGet(GRB.Callback.MIPSOL_OBJ))
                    if solver_obj >= callback_state["last_solver_obj"] - 1e-12:
                        return
                    callback_state["last_solver_obj"] = solver_obj

                    z_values = np.asarray(cb_model.cbGetSolution(zvar), dtype=float)
                    support = tuple(int(i) for i in np.flatnonzero(z_values > 0.5))
                    polished = solve_support(support)
                    if polished is not None:
                        consider(polished, support)
                    else:
                        raw = np.asarray(cb_model.cbGetSolution(xvar), dtype=float)
                        raw[z_values < 0.5] = 0.0
                        consider(raw, support)
                except Exception:
                    # A logging or numerical issue must not interrupt optimization.
                    pass

            model.optimize(callback)

            if model.SolCount > 0:
                z_values = np.asarray(zvar.X, dtype=float)
                support = tuple(int(i) for i in np.flatnonzero(z_values > 0.5))
                polished = solve_support(support)
                if polished is not None:
                    consider(polished, support)
                else:
                    raw = np.asarray(xvar.X, dtype=float)
                    raw[z_values < 0.5] = 0.0
                    consider(raw, support)

        except gp.GurobiError:
            # Return the best analytical incumbent if model construction or
            # optimization cannot be completed.
            pass

    # Defensive fallback for malformed or infeasible instances.
    if best_x is None:
        fallback = np.zeros(n, dtype=float)
        if n > 0:
            fallback[int(np.argmax(r))] = 1.0
        best_x = fallback
        best_obj = objective(best_x)

    final_solution = make_solution(best_x, best_obj)
    output_dir = os.path.dirname(os.path.abspath(args.solution_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(final_solution, f, separators=(",", ":"), allow_nan=False)


if __name__ == "__main__":
    main()
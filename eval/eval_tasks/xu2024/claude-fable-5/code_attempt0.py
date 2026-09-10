import argparse
import json
import time

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def solve_restricted_qp(Q, r, b, support, n, time_limit):
    """Solve continuous QP restricted to a given support set. Returns (obj, x) or None."""
    try:
        m = gp.Model("restricted")
        m.Params.OutputFlag = 0
        m.Params.Seed = 0
        m.Params.Threads = 1
        m.Params.NumericFocus = 0
        m.Params.TimeLimit = max(1.0, time_limit)
        idx = list(support)
        x = m.addVars(idx, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="x")
        m.addConstr(gp.quicksum(x[i] for i in idx) == 1.0)
        m.addConstr(gp.quicksum(r[i] * x[i] for i in idx) >= b)
        obj = gp.QuadExpr()
        for a in idx:
            for c in idx:
                if Q[a][c] != 0.0:
                    obj.add(0.5 * Q[a][c] * x[a] * x[c])
        m.setObjective(obj, GRB.MINIMIZE)
        m.optimize()
        if m.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT) and m.SolCount > 0:
            xs = [0.0] * n
            for i in idx:
                xs[i] = x[i].X
            return m.ObjVal, xs
    except gp.GurobiError:
        pass
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, default=300)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    n = int(data["n"])
    k = int(data["k"])
    b = float(data["b"])
    r = [float(v) for v in data["r"]]
    Q = [[float(v) for v in row] for row in data["Q"]]

    best_obj = None
    best_x = None

    def remaining():
        return args.time_limit - (time.time() - start_time)

    # ---------- Heuristic warm start ----------
    warm_x = None
    if k < n:
        res = solve_restricted_qp(Q, r, b, list(range(n)), n, min(30.0, max(2.0, remaining() * 0.1)))
        if res is not None:
            _, x_relax = res
            order = sorted(range(n), key=lambda i: -abs(x_relax[i]))
            support = order[:k]
            res2 = solve_restricted_qp(Q, r, b, support, n, min(30.0, max(2.0, remaining() * 0.1)))
            if res2 is not None:
                best_obj, best_x = res2
                warm_x = best_x
                if logger:
                    logger.log_solution(best_obj, {"objective_value": best_obj, "x": best_x})
    else:
        # No cardinality constraint binding: pure QP
        res = solve_restricted_qp(Q, r, b, list(range(n)), n, max(2.0, remaining() - 2.0))
        if res is not None:
            best_obj, best_x = res
            if logger:
                logger.log_solution(best_obj, {"objective_value": best_obj, "x": best_x})
        sol = {"objective_value": best_obj if best_obj is not None else 0.0,
               "x": best_x if best_x is not None else [0.0] * n}
        with open(args.solution_path, "w") as f:
            json.dump(sol, f)
        return

    # ---------- Exact MIQP ----------
    try:
        model = gp.Model("cardportfolio")
        model.Params.OutputFlag = 0
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1
        tl = max(1.0, remaining() - 3.0)
        model.Params.TimeLimit = tl

        x = model.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="x")
        z = model.addVars(n, vtype=GRB.BINARY, name="z")

        model.addConstr(gp.quicksum(x[i] for i in range(n)) == 1.0)
        model.addConstr(gp.quicksum(r[i] * x[i] for i in range(n)) >= b)
        model.addConstr(gp.quicksum(z[i] for i in range(n)) <= k)
        for i in range(n):
            model.addGenConstrIndicator(z[i], False, x[i] == 0.0)

        obj = gp.QuadExpr()
        for i in range(n):
            Qi = Q[i]
            for j in range(n):
                if Qi[j] != 0.0:
                    obj.add(0.5 * Qi[j] * x[i] * x[j])
        model.setObjective(obj, GRB.MINIMIZE)

        # Warm start
        if warm_x is not None:
            for i in range(n):
                x[i].Start = warm_x[i]
                z[i].Start = 1 if abs(warm_x[i]) > 1e-9 else 0

        incumbent = {"obj": best_obj, "x": best_x}

        def cb(m, where):
            if where == GRB.Callback.MIPSOL:
                objv = m.cbGet(GRB.Callback.MIPSOL_OBJ)
                if incumbent["obj"] is None or objv < incumbent["obj"] - 1e-12:
                    xv = m.cbGetSolution([x[i] for i in range(n)])
                    zv = m.cbGetSolution([z[i] for i in range(n)])
                    xs = [xv[i] if zv[i] > 0.5 else 0.0 for i in range(n)]
                    incumbent["obj"] = objv
                    incumbent["x"] = xs
                    if logger:
                        logger.log_solution(objv, {"objective_value": objv, "x": xs})

        model.optimize(cb)

        if model.SolCount > 0:
            objv = model.ObjVal
            if best_obj is None or objv < best_obj:
                xs = [x[i].X if z[i].X > 0.5 else 0.0 for i in range(n)]
                best_obj = objv
                best_x = xs
                if logger:
                    logger.log_solution(best_obj, {"objective_value": best_obj, "x": best_x})
        if incumbent["obj"] is not None and (best_obj is None or incumbent["obj"] < best_obj):
            best_obj = incumbent["obj"]
            best_x = incumbent["x"]
    except gp.GurobiError:
        pass

    # ---------- Polish: re-solve QP on final support for numerical accuracy ----------
    if best_x is not None and remaining() > 1.0:
        support = [i for i in range(n) if abs(best_x[i]) > 1e-9]
        if 0 < len(support) <= k:
            res = solve_restricted_qp(Q, r, b, support, n, min(10.0, remaining() - 0.5))
            if res is not None and res[0] < best_obj:
                best_obj, best_x = res
                if logger:
                    logger.log_solution(best_obj, {"objective_value": best_obj, "x": best_x})

    if best_x is None:
        best_x = [0.0] * n
        best_obj = 0.0

    # Recompute objective exactly from x for consistency
    xv = np.array(best_x)
    Qm = np.array(Q)
    obj_val = float(0.5 * xv.dot(Qm).dot(xv))

    sol = {"objective_value": obj_val, "x": [float(v) for v in best_x]}
    with open(args.solution_path, "w") as f:
        json.dump(sol, f)


if __name__ == "__main__":
    main()
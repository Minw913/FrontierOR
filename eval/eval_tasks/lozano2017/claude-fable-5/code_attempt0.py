import argparse
import json
import math
import time

import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except Exception:
    SolutionLogger = None


def set_params(m, tl):
    m.Params.OutputFlag = 0
    m.Params.Seed = 0
    m.Params.MIPGap = 1e-4
    m.Params.NumericFocus = 0
    m.Params.Threads = 1
    m.Params.TimeLimit = max(0.5, tl)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, default=600)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    start = time.time()
    reserve = min(3.0, 0.05 * args.time_limit + 1.0)
    deadline = start + max(2.0, args.time_limit - reserve)

    logger = None
    if args.log_path and SolutionLogger is not None:
        logger = SolutionLogger(args.log_path, sense="maximize")

    with open(args.instance_path) as f:
        data = json.load(f)

    n1 = data["n1"]
    n2 = data["n2"]
    m1 = data["m1"]
    m2 = data["m2"]
    c = data["c"]
    d2 = data["d2"]
    A1 = data["A1"]
    B1 = data["B1"]
    b1 = data["b1"]
    A2 = data["A2"]
    B2 = data["B2"]
    b2 = data["b2"]
    int_y = set(i - 1 for i in data.get("integer_follower_vars", []))

    CAP = 1e6

    def ubs_from(pairs, n):
        ub = [CAP] * n
        for M, b in pairs:
            for i in range(len(M)):
                bi = float(b[i])
                row = M[i]
                for j in range(n):
                    a = row[j]
                    if a > 0:
                        t = bi / a
                        if t < ub[j]:
                            ub[j] = t
        return ub

    ubx = [int(math.floor(u + 1e-9)) for u in ubs_from([(A1, b1), (A2, b2)], n1)]
    uby_f_raw = ubs_from([(B2, b2)], n2)
    uby_f = [math.floor(u + 1e-9) if j in int_y else u for j, u in enumerate(uby_f_raw)]
    uby_m_raw = ubs_from([(B1, b1), (B2, b2)], n2)
    uby_m = [math.floor(u + 1e-9) if j in int_y else u for j, u in enumerate(uby_m_raw)]

    def remaining():
        return deadline - time.time()

    # ---------- follower problem: given x, maximize d2'y s.t. B2 y <= b2 - A2 x ----------
    def solve_follower(xval, tl):
        rhs = [b2[i] - sum(A2[i][j] * xval[j] for j in range(n1)) for i in range(m2)]
        m = gp.Model("follower")
        set_params(m, tl)
        y = [m.addVar(lb=0.0, ub=uby_f[j],
                      vtype=(GRB.INTEGER if j in int_y else GRB.CONTINUOUS))
             for j in range(n2)]
        for i in range(m2):
            m.addConstr(gp.quicksum(B2[i][j] * y[j] for j in range(n2) if B2[i][j] != 0) <= rhs[i])
        m.setObjective(gp.quicksum(d2[j] * y[j] for j in range(n2) if d2[j] != 0), GRB.MAXIMIZE)
        m.optimize()
        if m.SolCount > 0:
            return m.ObjVal, [y[j].X for j in range(n2)]
        return None

    # ---------- check: exists y feasible for both levels with d2'y >= phi ----------
    def solve_check(xval, phi, tl):
        m = gp.Model("check")
        set_params(m, tl)
        y = [m.addVar(lb=0.0, ub=uby_m[j],
                      vtype=(GRB.INTEGER if j in int_y else GRB.CONTINUOUS))
             for j in range(n2)]
        for i in range(m2):
            rhs = b2[i] - sum(A2[i][j] * xval[j] for j in range(n1))
            m.addConstr(gp.quicksum(B2[i][j] * y[j] for j in range(n2) if B2[i][j] != 0) <= rhs)
        for i in range(m1):
            rhs = b1[i] - sum(A1[i][j] * xval[j] for j in range(n1))
            m.addConstr(gp.quicksum(B1[i][j] * y[j] for j in range(n2) if B1[i][j] != 0) <= rhs)
        eps = 1e-6 + 1e-9 * abs(phi)
        m.addConstr(gp.quicksum(d2[j] * y[j] for j in range(n2) if d2[j] != 0) >= phi - eps)
        m.setObjective(gp.quicksum(d2[j] * y[j] for j in range(n2) if d2[j] != 0), GRB.MAXIMIZE)
        m.optimize()
        if m.SolCount > 0:
            out = []
            for j in range(n2):
                v = y[j].X
                if j in int_y:
                    v = round(v)
                out.append(max(0.0, float(v)))
            return out
        return None

    def test_candidate(xval, tl):
        fres = solve_follower(xval, tl)
        if fres is None:
            return None
        phi, yhat = fres
        yfeas = solve_check(xval, phi, tl)
        return phi, yhat, yfeas

    best = {"obj": None, "x": None, "y": None}

    def record(xval, yval):
        obj = float(sum(c[j] * xval[j] for j in range(n1)))
        if best["obj"] is None or obj > best["obj"]:
            best["obj"] = obj
            best["x"] = [float(v) for v in xval]
            best["y"] = [float(v) for v in yval]
            fobj = float(sum(d2[j] * yval[j] for j in range(n2)))
            sol = {
                "objective_value": obj,
                "x": best["x"],
                "y": best["y"],
                "follower_objective_value": fobj,
            }
            if logger:
                logger.log_solution(obj, sol)
            return True
        return False

    fallback_y0 = None

    # initial candidate: x = 0
    try:
        if remaining() > 1:
            res0 = test_candidate([0] * n1, min(30.0, max(2.0, remaining())))
            if res0 is not None:
                phi0, yhat0, yfeas0 = res0
                fallback_y0 = yhat0
                if yfeas0 is not None:
                    record([0] * n1, yfeas0)
    except Exception:
        pass

    # ---------- master (high point relaxation + cuts) ----------
    master = gp.Model("master")
    set_params(master, max(1.0, remaining()))
    xv = [master.addVar(lb=0.0, ub=ubx[j], vtype=GRB.INTEGER) for j in range(n1)]
    yv = [master.addVar(lb=0.0, ub=uby_m[j],
                        vtype=(GRB.INTEGER if j in int_y else GRB.CONTINUOUS))
          for j in range(n2)]
    for i in range(m1):
        master.addConstr(
            gp.quicksum(A1[i][j] * xv[j] for j in range(n1) if A1[i][j] != 0)
            + gp.quicksum(B1[i][j] * yv[j] for j in range(n2) if B1[i][j] != 0) <= b1[i])
    for i in range(m2):
        master.addConstr(
            gp.quicksum(A2[i][j] * xv[j] for j in range(n1) if A2[i][j] != 0)
            + gp.quicksum(B2[i][j] * yv[j] for j in range(n2) if B2[i][j] != 0) <= b2[i])
    master.setObjective(gp.quicksum(c[j] * xv[j] for j in range(n1) if c[j] != 0), GRB.MAXIMIZE)

    LB_d2y = sum(min(0.0, d2[j] * uby_m[j]) for j in range(n2))

    def add_ls_cut(yhat):
        """If yhat stays follower-feasible at x, then d2'y >= d2'yhat."""
        dyhat = sum(d2[j] * yhat[j] for j in range(n2))
        rows = []
        for i in range(m2):
            rowA = A2[i]
            if all(a == 0 for a in rowA):
                continue
            r = b2[i] - sum(B2[i][j] * yhat[j] for j in range(n2))
            rhs = math.floor(r + 1e-9) + 1
            maxlhs = sum(rowA[j] * ubx[j] for j in range(n1))
            if rhs > maxlhs + 1e-9:
                continue
            rows.append((i, rhs))
        if not rows:
            master.addConstr(
                gp.quicksum(d2[j] * yv[j] for j in range(n2) if d2[j] != 0) >= dyhat - 1e-6)
            return
        Mk = max(0.0, dyhat - LB_d2y) + 1.0
        vs = []
        for (i, rhs) in rows:
            v = master.addVar(vtype=GRB.BINARY)
            master.addConstr(
                gp.quicksum(A2[i][j] * xv[j] for j in range(n1) if A2[i][j] != 0) >= rhs * v)
            vs.append(v)
        z = master.addVar(vtype=GRB.BINARY)
        master.addConstr(z <= gp.quicksum(vs))
        master.addConstr(
            gp.quicksum(d2[j] * yv[j] for j in range(n2) if d2[j] != 0) >= dyhat - 1e-6 - Mk * z)

    def add_nogood(xstar):
        bins = []
        for j, v in enumerate(xstar):
            ub = ubx[j]
            if ub - v >= 1:
                u = master.addVar(vtype=GRB.BINARY)
                master.addConstr(xv[j] >= (v + 1) * u)
                bins.append(u)
            if v >= 1:
                l = master.addVar(vtype=GRB.BINARY)
                master.addConstr(xv[j] <= (v - 1) + (ub - v + 1) * (1 - l))
                bins.append(l)
        if bins:
            master.addConstr(gp.quicksum(bins) >= 1)

    seen = set()
    proven_optimal = False
    tried_heuristic = False
    max_iters = 100000

    try:
        for _ in range(max_iters):
            rem = remaining()
            if rem < 1.0:
                break
            master.Params.TimeLimit = max(0.5, rem)
            master.optimize()
            st = master.Status
            if st == GRB.INFEASIBLE:
                proven_optimal = True
                break
            if master.SolCount == 0:
                break
            xval = [min(ubx[j], max(0, int(round(xv[j].X)))) for j in range(n1)]
            master_obj = sum(c[j] * xval[j] for j in range(n1))
            if best["obj"] is not None and master_obj <= best["obj"] + 1e-9:
                if st == GRB.OPTIMAL:
                    proven_optimal = True
                break
            key = tuple(xval)
            repeated = key in seen
            seen.add(key)

            rem = remaining()
            if rem < 1.0:
                break
            sub_tl = max(2.0, min(rem, 60.0))
            res = test_candidate(xval, sub_tl)
            if res is None:
                add_nogood(xval)
                continue
            phi, yhat, yfeas = res

            if yfeas is not None:
                record(xval, yfeas)
                if st == GRB.OPTIMAL:
                    proven_optimal = True
                    break
                inc = int(round(best["obj"]))
                master.addConstr(
                    gp.quicksum(c[j] * xv[j] for j in range(n1) if c[j] != 0) >= inc + 1)
                continue

            # bilevel infeasible candidate -> cut it off
            if repeated:
                add_nogood(xval)
            else:
                add_ls_cut(yhat)

            # cheap heuristic for an early incumbent
            if best["obj"] is None and not tried_heuristic and remaining() > 5.0:
                tried_heuristic = True
                for alpha in (0.75, 0.5, 0.25):
                    if remaining() < 3.0:
                        break
                    xh = [int(math.floor(alpha * v)) for v in xval]
                    if all(v == 0 for v in xh):
                        continue
                    rh = test_candidate(xh, max(2.0, min(remaining(), 20.0)))
                    if rh is not None and rh[2] is not None:
                        record(xh, rh[2])
                        inc = int(round(best["obj"]))
                        master.addConstr(
                            gp.quicksum(c[j] * xv[j] for j in range(n1) if c[j] != 0) >= inc + 1)
                        break
    except Exception:
        pass

    # ---------- write output ----------
    if best["obj"] is not None:
        xout = best["x"]
        yout = best["y"]
        obj = best["obj"]
    else:
        xout = [0.0] * n1
        if fallback_y0 is None:
            try:
                fr = solve_follower([0] * n1, 10.0)
                fallback_y0 = fr[1] if fr is not None else [0.0] * n2
            except Exception:
                fallback_y0 = [0.0] * n2
        yout = [round(v) if j in int_y else float(v) for j, v in enumerate(fallback_y0)]
        yout = [max(0.0, float(v)) for v in yout]
        obj = float(sum(c[j] * xout[j] for j in range(n1)))
        if logger:
            try:
                logger.log_solution(obj, {
                    "objective_value": obj,
                    "x": xout,
                    "y": yout,
                    "follower_objective_value": float(sum(d2[j] * yout[j] for j in range(n2))),
                })
            except Exception:
                pass

    fobj = float(sum(d2[j] * yout[j] for j in range(n2)))
    sol = {
        "objective_value": float(obj),
        "x": [float(v) for v in xout],
        "y": [float(v) for v in yout],
        "follower_objective_value": fobj,
    }
    with open(args.solution_path, "w") as f:
        json.dump(sol, f, indent=2)


if __name__ == "__main__":
    main()
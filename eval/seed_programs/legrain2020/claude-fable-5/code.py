import argparse
import json
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", required=True)
    ap.add_argument("--solution_path", required=True)
    ap.add_argument("--time_limit", type=int, default=300)
    ap.add_argument("--log_path", default=None)
    args = ap.parse_args()

    t0 = time.time()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path) as f:
        inst = json.load(f)

    D = inst["num_days"]
    W = inst["num_weeks"]
    shifts = [sh["name"] for sh in inst["shifts"]]
    cs_bounds = {sh["name"]: (sh["CS_minus"], sh["CS_plus"]) for sh in inst["shifts"]}
    pw = inst["penalty_weights"]
    nurses = inst["nurses"]
    N = len(nurses)
    demand = inst["demand"]
    forb = inst.get("forbidden_shift_successions", [])

    m = gp.Model("nurse_rostering")
    m.Params.OutputFlag = 0
    m.Params.Seed = 0
    m.Params.MIPGap = 1e-4
    m.Params.NumericFocus = 0
    m.Params.Threads = 1

    # x[i,d,s,k] : nurse i works shift s on day d performing skill k
    x = {}
    for i, nu in enumerate(nurses):
        for d in range(D):
            for s in shifts:
                for k in nu["skills"]:
                    x[i, d, s, k] = m.addVar(vtype=GRB.BINARY, name=f"x_{i}_{d}_{s}_{k}")

    # w[i,d] : nurse works on day d
    wv = {}
    for i, nu in enumerate(nurses):
        for d in range(D):
            wv[i, d] = m.addVar(vtype=GRB.BINARY, name=f"w_{i}_{d}")
            m.addConstr(
                wv[i, d]
                == gp.quicksum(x[i, d, s, k] for s in shifts for k in nu["skills"])
            )

    # y[i,d,s] as linear expression: nurse works shift s on day d
    def y_expr(i, d, s):
        return gp.quicksum(x[i, d, s, k] for k in nurses[i]["skills"])

    obj = gp.LinExpr()

    # ---------- Coverage (hard minimum + soft optimal) ----------
    # index nurses per skill
    nurses_with_skill = {k: [] for k in inst["skills"]}
    for i, nu in enumerate(nurses):
        for k in nu["skills"]:
            if k in nurses_with_skill:
                nurses_with_skill[k].append(i)
            else:
                nurses_with_skill[k] = [i]

    c_S1 = pw["c_S1"]
    for d in range(D):
        day_dem = demand[d]
        for s, skill_map in day_dem.items():
            for k, req in skill_map.items():
                mn = req.get("minimum", 0)
                op = req.get("optimal", 0)
                cover = gp.quicksum(
                    x[i, d, s, k] for i in nurses_with_skill.get(k, [])
                )
                if mn > 0:
                    m.addConstr(cover >= mn, name=f"min_{d}_{s}_{k}")
                if op > 0:
                    z = m.addVar(lb=0.0, name=f"slk_{d}_{s}_{k}")
                    m.addConstr(cover + z >= op)
                    obj += c_S1 * z

    # ---------- Forbidden shift successions (hard) ----------
    for (s1, s2) in forb:
        if s1 not in shifts or s2 not in shifts:
            continue
        for i in range(N):
            for d in range(D - 1):
                m.addConstr(y_expr(i, d, s1) + y_expr(i, d + 1, s2) <= 1)

    # ---------- Stretch penalty helper ----------
    def add_seq_penalties(seq, lo, hi, weight, name):
        """seq: list (len D) of vars/exprs in [0,1]. Penalize maximal runs of 1s
        shorter than lo or longer than hi, per unit deviation, at 'weight'."""
        nonlocal obj
        Dn = len(seq)
        if weight <= 0:
            return
        # max-length violations: each window of hi+1 fully filled -> 1 unit
        if hi < Dn:
            for d in range(Dn - hi):
                v = m.addVar(lb=0.0, name=f"{name}_mx_{d}")
                m.addConstr(
                    gp.quicksum(seq[t] for t in range(d, d + hi + 1)) - hi <= v
                )
                obj += weight * v
        # min-length violations: detect maximal runs of exact length j < lo
        if lo > 1:
            for j in range(1, lo):
                for d in range(0, Dn - j + 1):
                    expr = gp.quicksum(seq[t] for t in range(d, d + j))
                    if d > 0:
                        expr = expr - seq[d - 1]
                    if d + j < Dn:
                        expr = expr - seq[d + j]
                    v = m.addVar(lb=0.0, ub=1.0, name=f"{name}_mn_{j}_{d}")
                    m.addConstr(v >= expr - (j - 1))
                    obj += weight * (lo - j) * v

    c_S2a = pw["c_S2a"]
    c_S2b = pw["c_S2b"]
    c_S3 = pw["c_S3"]
    c_S4 = pw["c_S4"]
    c_S5 = pw["c_S5"]
    c_S6 = pw["c_S6"]
    c_S7 = pw["c_S7"]

    for i, nu in enumerate(nurses):
        ct = nu["contract"]
        work_seq = [wv[i, d] for d in range(D)]
        rest_seq = [1 - wv[i, d] for d in range(D)]
        # consecutive worked days
        add_seq_penalties(work_seq, ct["CD_minus"], ct["CD_plus"], c_S2a, f"cw_{i}")
        # consecutive days off
        add_seq_penalties(rest_seq, ct["CR_minus"], ct["CR_plus"], c_S3, f"cr_{i}")
        # consecutive same shift
        for s in shifts:
            lo, hi = cs_bounds[s]
            seq = [y_expr(i, d, s) for d in range(D)]
            add_seq_penalties(seq, lo, hi, c_S2b, f"cs_{i}_{s}")

        # preferences (preferred off assignments)
        for pref in nu.get("preferences", []):
            pd = pref["day"]
            ps = pref["shift"]
            if 0 <= pd < D and ps in shifts:
                obj += c_S4 * y_expr(i, pd, ps)

        # weekends
        ww_vars = []
        for wk in range(W):
            sat, sun = 7 * wk + 5, 7 * wk + 6
            if sat >= D or sun >= D:
                continue
            # incomplete weekend
            v = m.addVar(lb=0.0, ub=1.0, name=f"iw_{i}_{wk}")
            m.addConstr(v >= wv[i, sat] - wv[i, sun])
            m.addConstr(v >= wv[i, sun] - wv[i, sat])
            obj += c_S5 * v
            # worked weekend indicator
            ww = m.addVar(vtype=GRB.BINARY, name=f"ww_{i}_{wk}")
            m.addConstr(ww >= wv[i, sat])
            m.addConstr(ww >= wv[i, sun])
            ww_vars.append(ww)
        ex = m.addVar(lb=0.0, name=f"exww_{i}")
        m.addConstr(ex >= gp.quicksum(ww_vars) - ct["WE_plus"])
        obj += c_S7 * ex

        # total assignments
        tot = gp.quicksum(wv[i, d] for d in range(D))
        under = m.addVar(lb=0.0, name=f"un_{i}")
        over = m.addVar(lb=0.0, name=f"ov_{i}")
        m.addConstr(under >= ct["L_minus"] - tot)
        m.addConstr(over >= tot - ct["L_plus"])
        obj += c_S6 * (under + over)

    m.setObjective(obj, GRB.MINIMIZE)

    # ---------- Solution extraction ----------
    xkeys = list(x.keys())
    xvars = [x[kk] for kk in xkeys]

    def build_schedule(vmap):
        sched = []
        for i, nu in enumerate(nurses):
            assigns = []
            for d in range(D):
                for s in shifts:
                    for k in nu["skills"]:
                        if vmap.get((i, d, s, k), 0.0) > 0.5:
                            assigns.append({"day": d, "shift": s, "skill": k})
            sched.append(
                {
                    "nurse_id": nu["id"],
                    "nurse_name": nu["name"],
                    "assignments": assigns,
                }
            )
        return sched

    best_obj = [float("inf")]
    best_sol = [None]

    def cb(model, where):
        if where == GRB.Callback.MIPSOL:
            objval = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if objval < best_obj[0] - 1e-6:
                best_obj[0] = objval
                vals = model.cbGetSolution(xvars)
                vmap = dict(zip(xkeys, vals))
                sched = build_schedule(vmap)
                sol = {"objective_value": float(round(objval, 6)), "schedule": sched}
                best_sol[0] = sol
                if logger:
                    logger.log_solution(float(round(objval, 6)), sol)

    elapsed = time.time() - t0
    m.Params.TimeLimit = max(1.0, args.time_limit - elapsed - 2.0)
    m.optimize(cb)

    # ---------- Final output ----------
    if m.SolCount > 0:
        vmap = {kk: v.X for kk, v in zip(xkeys, xvars)}
        sched = build_schedule(vmap)
        objval = float(round(m.ObjVal, 6))
        sol = {"objective_value": objval, "schedule": sched}
        if logger and objval < best_obj[0] - 1e-6:
            logger.log_solution(objval, sol)
    elif best_sol[0] is not None:
        sol = best_sol[0]
    else:
        sol = {
            "objective_value": float("inf"),
            "schedule": [
                {"nurse_id": nu["id"], "nurse_name": nu["name"], "assignments": []}
                for nu in nurses
            ],
        }

    with open(args.solution_path, "w") as f:
        json.dump(sol, f, indent=2)


if __name__ == "__main__":
    main()
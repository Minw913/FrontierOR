import argparse
import json
import time
from collections import defaultdict

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

    start = time.time()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path) as f:
        data = json.load(f)

    H = data["num_days"]
    shifts = [st["id"] for st in data["shift_types"]]
    length = {st["id"]: st["length_minutes"] for st in data["shift_types"]}

    forb = defaultdict(list)
    for r in data.get("forbidden_shift_rotations", []):
        forb[r["shift_type"]].append(r["cannot_be_followed_by"])

    nurses = data["nurses"]
    num_weekends = data["num_weekends"]
    cov = data["coverage_requirements"]
    on_req = data.get("shift_on_requests", [])
    off_req = data.get("shift_off_requests", [])

    m = gp.Model("nrp")
    m.Params.OutputFlag = 0
    m.Params.Seed = 0
    m.Params.MIPGap = 1e-4
    m.Params.NumericFocus = 0
    m.Params.Threads = 1

    # Decision variables
    x = {}
    for n in nurses:
        nid = n["id"]
        offdays = set(n["day_off_requests"])
        for d in range(H):
            fixed_zero = d in offdays
            for s in shifts:
                v = m.addVar(vtype=GRB.BINARY, name=f"x_{nid}_{d}_{s}")
                if fixed_zero:
                    v.UB = 0.0
                x[(nid, d, s)] = v

    # Working-day expressions
    w = {}
    for n in nurses:
        nid = n["id"]
        for d in range(H):
            w[(nid, d)] = gp.quicksum(x[(nid, d, s)] for s in shifts)

    for n in nurses:
        nid = n["id"]
        # at most one shift per day
        for d in range(H):
            m.addConstr(w[(nid, d)] <= 1)
        # forbidden successions
        for d in range(H - 1):
            for s1, succs in forb.items():
                m.addConstr(
                    x[(nid, d, s1)]
                    + gp.quicksum(x[(nid, d + 1, s2)] for s2 in succs)
                    <= 1
                )
        # max shifts per type
        for s in shifts:
            cap = n["max_shifts_per_type"].get(str(s))
            if cap is not None:
                m.addConstr(gp.quicksum(x[(nid, d, s)] for d in range(H)) <= cap)
        # total minutes
        total_min = gp.quicksum(length[s] * x[(nid, d, s)] for d in range(H) for s in shifts)
        m.addConstr(total_min >= n["min_total_minutes"])
        m.addConstr(total_min <= n["max_total_minutes"])
        # max consecutive shifts
        cmax = n["max_consecutive_shifts"]
        for d in range(H - cmax):
            m.addConstr(gp.quicksum(w[(nid, d + j)] for j in range(cmax + 1)) <= cmax)
        # min consecutive shifts
        cmin = n["min_consecutive_shifts"]
        for g in range(1, cmin):
            for d in range(0, H - g - 1):
                m.addConstr(
                    gp.quicksum(w[(nid, d + j)] for j in range(1, g + 1))
                    - w[(nid, d)] - w[(nid, d + g + 1)]
                    <= g - 1
                )
        # min consecutive days off
        omin = n["min_consecutive_days_off"]
        for g in range(1, omin):
            for d in range(0, H - g - 1):
                m.addConstr(
                    w[(nid, d)] + w[(nid, d + g + 1)]
                    - gp.quicksum(w[(nid, d + j)] for j in range(1, g + 1))
                    <= 1
                )
        # max weekends
        kvars = []
        for wk in range(num_weekends):
            sat, sun = 7 * wk + 5, 7 * wk + 6
            k = m.addVar(vtype=GRB.BINARY, name=f"k_{nid}_{wk}")
            if sat < H:
                m.addConstr(w[(nid, sat)] <= k)
            if sun < H:
                m.addConstr(w[(nid, sun)] <= k)
            kvars.append(k)
        m.addConstr(gp.quicksum(kvars) <= n["max_weekends"])

    # Coverage
    obj = gp.LinExpr()
    for c in cov:
        d, s = c["day"], c["shift_type"]
        u = m.addVar(lb=0.0, name=f"u_{d}_{s}")
        o = m.addVar(lb=0.0, name=f"o_{d}_{s}")
        m.addConstr(
            gp.quicksum(x[(n["id"], d, s)] for n in nurses) - c["preferred"] == o - u
        )
        obj += c["under_weight"] * u + c["over_weight"] * o

    # Requests
    for r in on_req:
        obj += r["penalty"] * (1 - x[(r["nurse_id"], r["day"], r["shift_type"])])
    for r in off_req:
        obj += r["penalty"] * x[(r["nurse_id"], r["day"], r["shift_type"])]

    m.setObjective(obj, GRB.MINIMIZE)

    xkeys = list(x.keys())
    xlist = [x[k] for k in xkeys]

    def build_solution(vals):
        schedule = {str(n["id"]): {str(d): None for d in range(H)} for n in nurses}
        count = defaultdict(int)
        for key, v in zip(xkeys, vals):
            if v > 0.5:
                nid, d, s = key
                schedule[str(nid)][str(d)] = s
                count[(d, s)] += 1
        total = 0.0
        for r in on_req:
            if schedule[str(r["nurse_id"])][str(r["day"])] != r["shift_type"]:
                total += r["penalty"]
        for r in off_req:
            if schedule[str(r["nurse_id"])][str(r["day"])] == r["shift_type"]:
                total += r["penalty"]
        cu, co = {}, {}
        for c in cov:
            d, s = c["day"], c["shift_type"]
            cnt = count.get((d, s), 0)
            under = max(0, c["preferred"] - cnt)
            over = max(0, cnt - c["preferred"])
            if under:
                cu[f"{d}_{s}"] = under
                total += under * c["under_weight"]
            if over:
                co[f"{d}_{s}"] = over
                total += over * c["over_weight"]
        return total, {"schedule": schedule, "coverage_under": cu, "coverage_over": co}

    m._best = float("inf")

    def cb(model, where):
        if where == GRB.Callback.MIPSOL:
            vals = model.cbGetSolution(xlist)
            objv, sol = build_solution(vals)
            if objv < model._best - 1e-9:
                model._best = objv
                if logger:
                    logger.log_solution(objv, sol)

    elapsed = time.time() - start
    m.Params.TimeLimit = max(1.0, args.time_limit - elapsed - 3.0)

    m.optimize(cb)

    if m.SolCount > 0:
        vals = [v.X for v in xlist]
        objv, sol = build_solution(vals)
        if logger and objv < m._best - 1e-9:
            logger.log_solution(objv, sol)
        out = {"objective_value": float(objv), "solution": sol}
    else:
        # Fallback: empty schedule
        schedule = {str(n["id"]): {str(d): None for d in range(H)} for n in nurses}
        cu = {}
        total = 0.0
        for r in on_req:
            total += r["penalty"]
        for c in cov:
            if c["preferred"] > 0:
                cu[f"{c['day']}_{c['shift_type']}"] = c["preferred"]
                total += c["preferred"] * c["under_weight"]
        sol = {"schedule": schedule, "coverage_under": cu, "coverage_over": {}}
        if logger:
            logger.log_solution(total, sol)
        out = {"objective_value": float(total), "solution": sol}

    with open(args.solution_path, "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()
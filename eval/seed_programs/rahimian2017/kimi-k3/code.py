import argparse
import json
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def main():
    parser = argparse.ArgumentParser(description="Nurse rostering solver")
    parser.add_argument("--instance_path", required=True,
                        help="Path to the instance JSON file.")
    parser.add_argument("--solution_path", required=True,
                        help="Path where the solution JSON file is written.")
    parser.add_argument("--time_limit", type=int, required=True,
                        help="Maximum runtime in seconds.")
    parser.add_argument("--log_path", default=None,
                        help="Optional path to a JSONL file for incumbent logging.")
    args = parser.parse_args()

    t0 = time.time()

    # ------------------------------------------------------------------ #
    # Read instance
    # ------------------------------------------------------------------ #
    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    D = int(inst["num_days"])
    shift_list = inst["shift_types"]
    shift_ids = [s["id"] for s in shift_list]
    shift_id_set = set(shift_ids)
    length = {s["id"]: float(s["length_minutes"]) for s in shift_list}

    nurse_list = inst["nurses"]
    nurse_ids = [n["id"] for n in nurse_list]
    nurse_id_set = set(nurse_ids)

    forbidden = set()
    for fr in inst.get("forbidden_shift_rotations", []):
        forbidden.add((fr["shift_type"], fr["cannot_be_followed_by"]))

    cov = {}
    for c in inst.get("coverage_requirements", []):
        cov[(c["day"], c["shift_type"])] = {
            "preferred": c["preferred"],
            "under_weight": c["under_weight"],
            "over_weight": c["over_weight"],
        }

    on_req = {}
    for r in inst.get("shift_on_requests", []):
        key = (r["nurse_id"], r["day"], r["shift_type"])
        on_req[key] = on_req.get(key, 0.0) + r["penalty"]

    off_req = {}
    for r in inst.get("shift_off_requests", []):
        key = (r["nurse_id"], r["day"], r["shift_type"])
        off_req[key] = off_req.get(key, 0.0) + r["penalty"]

    # Weeks start on Monday: weekend `we` consists of days 7*we+5 (Sat) and
    # 7*we+6 (Sun).
    weekends = []
    for we in range(int(inst.get("num_weekends", 0))):
        sat = 7 * we + 5
        sun = 7 * we + 6
        if sat < D and sun < D:
            weekends.append((sat, sun))

    # ------------------------------------------------------------------ #
    # Build model
    # ------------------------------------------------------------------ #
    model = gp.Model("nurse_rostering")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.LogToConsole = 0

    # Decision variables: x[nurse, day, shift] = 1 if nurse works shift on day
    x = {}
    for nid in nurse_ids:
        for d in range(D):
            for s in shift_ids:
                x[(nid, d, s)] = model.addVar(vtype=GRB.BINARY,
                                              name="x_%s_%s_%s" % (nid, d, s))
    # w[nurse, day] = 1 if nurse works any shift on day
    w = {}
    for nid in nurse_ids:
        for d in range(D):
            w[(nid, d)] = model.addVar(vtype=GRB.BINARY,
                                       name="w_%s_%s" % (nid, d))

    model.update()

    # At most one shift per nurse per day (link w to x)
    for nid in nurse_ids:
        for d in range(D):
            model.addConstr(
                w[(nid, d)] == gp.quicksum(x[(nid, d, s)] for s in shift_ids),
                name="link_%s_%s" % (nid, d))

    # Forbidden shift rotations on consecutive days
    for nid in nurse_ids:
        for d in range(D - 1):
            for (s1, s2) in forbidden:
                model.addConstr(x[(nid, d, s1)] + x[(nid, d + 1, s2)] <= 1,
                                name="rot_%s_%s_%s_%s" % (nid, d, s1, s2))

    # Per-nurse constraints
    for n in nurse_list:
        nid = n["id"]

        # Maximum number of shifts of each type
        mspt = n.get("max_shifts_per_type", {}) or {}
        for s in shift_ids:
            mx = mspt.get(str(s))
            if mx is None:
                mx = mspt.get(s)
            if mx is not None:
                model.addConstr(
                    gp.quicksum(x[(nid, d, s)] for d in range(D)) <= int(mx),
                    name="maxtype_%s_%s" % (nid, s))

        # Total working minutes bounds
        total = gp.quicksum(length[s] * x[(nid, d, s)]
                            for d in range(D) for s in shift_ids)
        model.addConstr(total >= float(n["min_total_minutes"]),
                        name="minmin_%s" % nid)
        model.addConstr(total <= float(n["max_total_minutes"]),
                        name="maxmin_%s" % nid)

        # Requested days off (hard)
        for d in n.get("day_off_requests", []):
            if 0 <= d < D:
                model.addConstr(w[(nid, d)] == 0,
                                name="dayoff_%s_%s" % (nid, d))

        # Maximum consecutive working days:
        # any window of mxc+1 days contains at most mxc working days.
        mxc = int(n["max_consecutive_shifts"])
        if mxc < D:
            for d0 in range(0, D - mxc):
                model.addConstr(
                    gp.quicksum(w[(nid, dd)]
                                for dd in range(d0, d0 + mxc + 1)) <= mxc,
                    name="maxcons_%s_%s" % (nid, d0))

        # Minimum consecutive working days:
        # no working block of length k < mcs surrounded by off days.
        # Days outside the horizon count as working (infinite work stretch).
        mcs = int(n["min_consecutive_shifts"])
        for k in range(1, mcs):
            for d0 in range(0, D - k - 1):
                model.addConstr(
                    w[(nid, d0)] + w[(nid, d0 + k + 1)]
                    + gp.quicksum(1 - w[(nid, d0 + i)]
                                  for i in range(1, k + 1)) >= 1,
                    name="mincons_%s_%s_%s" % (nid, d0, k))

        # Minimum consecutive days off:
        # no off block of length k < mco surrounded by working days.
        # Days outside the horizon count as off (infinite off stretch).
        mco = int(n["min_consecutive_days_off"])
        for k in range(1, mco):
            for d0 in range(0, D - k - 1):
                model.addConstr(
                    (1 - w[(nid, d0)]) + (1 - w[(nid, d0 + k + 1)])
                    + gp.quicksum(w[(nid, d0 + i)]
                                  for i in range(1, k + 1)) >= 1,
                    name="minoff_%s_%s_%s" % (nid, d0, k))

        # Maximum number of worked weekends
        maxwe = int(n["max_weekends"])
        if maxwe < len(weekends):
            yvars = []
            for wi, (sat, sun) in enumerate(weekends):
                yv = model.addVar(vtype=GRB.BINARY,
                                  name="we_%s_%s" % (nid, wi))
                model.addConstr(yv >= w[(nid, sat)],
                                name="wea_%s_%s" % (nid, wi))
                model.addConstr(yv >= w[(nid, sun)],
                                name="web_%s_%s" % (nid, wi))
                yvars.append(yv)
            model.addConstr(gp.quicksum(yvars) <= maxwe,
                            name="maxwe_%s" % nid)

    # Coverage deviation variables
    under = {}
    over = {}
    for (d, s), c in cov.items():
        if not (0 <= d < D) or s not in shift_id_set:
            continue
        u = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0,
                         name="under_%s_%s" % (d, s))
        o = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0,
                         name="over_%s_%s" % (d, s))
        under[(d, s)] = u
        over[(d, s)] = o
        assigned = gp.quicksum(x[(nid, d, s)] for nid in nurse_ids)
        model.addConstr(u >= c["preferred"] - assigned,
                        name="cu_%s_%s" % (d, s))
        model.addConstr(o >= assigned - c["preferred"],
                        name="co_%s_%s" % (d, s))

    # Objective: request penalties + coverage deviations
    obj = gp.LinExpr()
    for (nid, d, s), p in on_req.items():
        if (nid, d, s) in x:
            obj += p * (1.0 - x[(nid, d, s)])
        else:
            obj += p
    for (nid, d, s), p in off_req.items():
        if (nid, d, s) in x:
            obj += p * x[(nid, d, s)]
    for (d, s), c in cov.items():
        if (d, s) in under:
            obj += c["under_weight"] * under[(d, s)]
            obj += c["over_weight"] * over[(d, s)]
    model.setObjective(obj, GRB.MINIMIZE)

    # ------------------------------------------------------------------ #
    # Helpers: build solution dictionary from x values
    # ------------------------------------------------------------------ #
    def build_solution(xvals):
        schedule = {}
        counts = {}
        for nid in nurse_ids:
            daymap = {}
            for d in range(D):
                val = None
                for s in shift_ids:
                    if xvals[(nid, d, s)] > 0.5:
                        val = s
                        break
                daymap[str(d)] = val
                if val is not None:
                    counts[(d, val)] = counts.get((d, val), 0) + 1
            schedule[str(nid)] = daymap

        cu = {}
        co = {}
        for (d, s), c in cov.items():
            a = counts.get((d, s), 0)
            cu["%s_%s" % (d, s)] = max(0, c["preferred"] - a)
            co["%s_%s" % (d, s)] = max(0, a - c["preferred"])

        objval = 0.0
        for (nid, d, s), p in on_req.items():
            if schedule.get(str(nid), {}).get(str(d)) != s:
                objval += p
        for (nid, d, s), p in off_req.items():
            if schedule.get(str(nid), {}).get(str(d)) == s:
                objval += p
        for (d, s), c in cov.items():
            key = "%s_%s" % (d, s)
            objval += cu[key] * c["under_weight"]
            objval += co[key] * c["over_weight"]

        return float(objval), {"schedule": schedule,
                               "coverage_under": cu,
                               "coverage_over": co}

    # Log every incumbent found by Gurobi
    def callback(m, where):
        if where == GRB.Callback.MIPSOL:
            try:
                xv = m.cbGetSolution(x)
            except gp.GurobiError:
                return
            try:
                ov, sol = build_solution(xv)
                if logger:
                    logger.log_solution(ov, sol)
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # Solve
    # ------------------------------------------------------------------ #
    remaining = float(args.time_limit) - (time.time() - t0) - 1.0
    model.Params.TimeLimit = max(1.0, remaining)

    try:
        model.optimize(callback)
    except gp.GurobiError:
        pass

    # ------------------------------------------------------------------ #
    # Extract best solution and write output
    # ------------------------------------------------------------------ #
    best_obj = None
    best_sol = None
    if model.SolCount > 0:
        try:
            xv = model.getAttr("X", x)
            best_obj, best_sol = build_solution(xv)
            if logger:
                logger.log_solution(best_obj, best_sol)
        except gp.GurobiError:
            best_obj = None
            best_sol = None

    if best_sol is None:
        # Fallback (no feasible solution found): everyone off.
        xv = {}
        for nid in nurse_ids:
            for d in range(D):
                for s in shift_ids:
                    xv[(nid, d, s)] = 0.0
        best_obj, best_sol = build_solution(xv)

    out = {"objective_value": float(best_obj), "solution": best_sol}
    with open(args.solution_path, "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()
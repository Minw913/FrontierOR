import argparse
import json
import math
import time
from collections import defaultdict

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger

# Activity types that are considered "rest-like" (days off that can be part of rest periods)
REST_TYPES = {"annual_leave", "desiderata"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", required=True)
    ap.add_argument("--solution_path", required=True)
    ap.add_argument("--time_limit", type=int, default=300)
    ap.add_argument("--log_path", default=None)
    args = ap.parse_args()

    t0 = time.time()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    H = data["planning_horizon_days"]
    wr = data.get("work_rules", {}) or {}
    weekly_h = wr.get("weekly_rest_period_hours", 48)
    add_h = wr.get("monthly_rest_additional_hours", 48)
    spread = wr.get("max_spread_between_breaks_days", 8)
    mfl = wr.get("monthly_flight_limit_hours", 77)
    rfl = wr.get("rolling_flight_limit_hours", 85)
    max_early = wr.get("max_consecutive_early_departures_before_8am", 2)
    max_cross = wr.get("max_cross_pairings_per_month", 1)

    wdays = max(1, int(math.ceil(weekly_h / 24.0)))  # days of consecutive free time for weekly rest

    pairings = data["pairings"]
    crews = data["crew_members"]

    # ---------- Pairing preprocessing ----------
    p_by_id = {}
    p_occ = {}      # pairing id -> set of occupied days (clipped to horizon)
    for p in pairings:
        pid = p["id"]
        p_by_id[pid] = p
        s = max(0, p["start_day"])
        e = min(H, p["start_day"] + p["duration_days"])
        p_occ[pid] = set(range(s, e))

    # ---------- Crew preprocessing ----------
    crew_info = {}
    for c in crews:
        cid = c["id"]
        quals = set(c.get("qualifications", []))
        blocked = {}         # day -> 'block' or 'rest'
        reserve_days = set()
        reserve_end = set()
        act_minutes = 0      # pre-assigned activity minutes within horizon
        inactive_days = 0
        for a in c.get("pre_assigned_activities", []):
            sd = a["start_day"]
            ed = a["end_day"]
            t = a.get("type", "")
            lo = max(0, sd)
            hi = min(H - 1, ed)
            ndays = max(0, hi - lo + 1)
            act_minutes += ndays * 1440
            kind = "rest" if t in REST_TYPES else "block"
            for d in range(lo, hi + 1):
                if kind == "block":
                    blocked[d] = "block"
                elif d not in blocked:
                    blocked[d] = "rest"
            if t == "reserve_block":
                reserve_days.update(range(lo, hi + 1))
                reserve_end.add(ed)
            if t != "transition":
                inactive_days += ndays
        for d in c.get("pre_assigned_days", []):
            if 0 <= d < H and d not in blocked:
                blocked[d] = "block"
                act_minutes += 1440

        pre_days = set(blocked.keys())

        # eligible pairings
        elig = []
        for p in pairings:
            pid = p["id"]
            if p["aircraft_type"] not in quals:
                continue
            if p_occ[pid] & pre_days:
                continue
            # reserve blocks must be followed by rest or pairings departing at noon or later
            if (p["start_day"] - 1) in reserve_end and p["departure_hour"] < 12.0:
                continue
            elig.append(pid)

        crew_info[cid] = {
            "crew": c,
            "quals": quals,
            "blocked": blocked,
            "reserve_days": reserve_days,
            "act_minutes": act_minutes,
            "inactive_days": inactive_days,
            "elig": elig,
            "prev15": float(c.get("flight_hours_previous_15_days", 0.0)),
        }

    # ---------- Solution builder ----------
    def build_solution(assign):
        """assign: dict crew_id -> list of pairing ids"""
        total_min = H * 1440
        rosters = {}
        counts = defaultdict(int)
        for c in crews:
            cid = c["id"]
            plist = sorted(assign.get(cid, []))
            pdur = 0
            for pid in plist:
                pdur += p_by_id[pid]["duration_minutes"]
                counts[pid] += 1
            unprod = total_min - crew_info[cid]["act_minutes"] - pdur
            rosters[str(cid)] = {
                "roster_index": 0,
                "pairings": [int(x) for x in plist],
                "unproductive_time": int(unprod),
            }
        uncovered = []
        obj = 0.0
        for p in pairings:
            pid = p["id"]
            unc = max(0, p["coverage_requirement"] - counts.get(pid, 0))
            if unc > 0:
                uncovered.append({
                    "pairing_id": int(pid),
                    "uncovered_count": int(unc),
                    "duration_minutes": int(p["duration_minutes"]),
                })
                obj += unc * p["duration_minutes"]
        sol = {
            "objective_value": float(obj),
            "rosters": rosters,
            "uncovered_pairings": uncovered,
        }
        return obj, sol

    # Trivial (empty) solution
    best_obj, best_sol = build_solution({})
    if logger:
        logger.log_solution(best_obj, best_sol)

    # ---------- Build MIP ----------
    try:
        m = gp.Model("rostering")
        m.Params.OutputFlag = 0
        m.Params.Seed = 0
        m.Params.MIPGap = 1e-4
        m.Params.NumericFocus = 0
        m.Params.Threads = 1

        x = {}          # (cid, pid) -> binary var
        xvars = []
        xmeta = []
        pairings_of_day = defaultdict(lambda: defaultdict(list))  # cid -> day -> [xvar]

        for cid, info in crew_info.items():
            for pid in info["elig"]:
                v = m.addVar(vtype=GRB.BINARY, name=f"x_{cid}_{pid}")
                v.Start = 0.0
                x[(cid, pid)] = v
                xvars.append(v)
                xmeta.append((cid, pid))
                for d in p_occ[pid]:
                    pairings_of_day[cid][d].append(v)

        # coverage / shortfall
        s_vars = {}
        obj_expr = gp.LinExpr()
        for p in pairings:
            pid = p["id"]
            cov = p["coverage_requirement"]
            sv = m.addVar(lb=0.0, ub=float(cov), name=f"s_{pid}")
            s_vars[pid] = sv
            expr = gp.LinExpr()
            for cid in crew_info:
                if (cid, pid) in x:
                    expr += x[(cid, pid)]
            m.addConstr(expr + sv == cov)
            obj_expr += p["duration_minutes"] * sv
        m.setObjective(obj_expr, GRB.MINIMIZE)

        # per-crew constraints
        for cid, info in crew_info.items():
            blocked = info["blocked"]
            reserve_days = info["reserve_days"]
            elig = info["elig"]

            # day capacity: 0 blocked / 1 rest-preassigned / 2 free (variable)
            cap = []
            for d in range(H):
                st = blocked.get(d)
                if st == "block":
                    cap.append(0)
                elif st == "rest":
                    cap.append(1)
                else:
                    cap.append(2)

            fvar = {}
            for d in range(H):
                if cap[d] == 2:
                    fv = m.addVar(lb=0.0, ub=1.0, name=f"f_{cid}_{d}")
                    fvar[d] = fv
                    m.addConstr(fv + gp.quicksum(pairings_of_day[cid][d]) <= 1)
                else:
                    # exclusivity on other days: no eligible pairing touches them anyway
                    if pairings_of_day[cid][d]:
                        m.addConstr(gp.quicksum(pairings_of_day[cid][d]) <= 1)

            def day_possible(d):
                return d >= H or cap[d] != 0

            # rest-start variables z[d]: wdays consecutive free days starting at d
            zvar = {}
            for d in range(H):
                ok = all(day_possible(d + k) for k in range(wdays))
                if not ok:
                    continue
                zv = m.addVar(lb=0.0, ub=1.0, name=f"z_{cid}_{d}")
                zvar[d] = zv
                for k in range(wdays):
                    dd = d + k
                    if dd < H and cap[dd] == 2:
                        m.addConstr(zv <= fvar[dd])

            # weekly rest: one rest start per full calendar week (day 0 assumed Monday)
            nweeks = H // 7
            for w in range(nweeks):
                cands = [zvar[d] for d in range(7 * w, 7 * w + 7) if d in zvar]
                if cands:
                    m.addConstr(gp.quicksum(cands) >= 1)

            # spread between breaks
            d = 0
            while d < H:
                wlen = spread + 1
                if any(dd in reserve_days for dd in range(d, min(H, d + wlen + 1))):
                    wlen += 1
                if d + wlen > H:
                    break
                cands = [zvar[dd] for dd in range(d, d + wlen) if dd in zvar]
                if cands:
                    m.addConstr(gp.quicksum(cands) >= 1)
                d += 1

            # monthly rest: extended free window once per month
            E = max(0, add_h - 12 * (info["inactive_days"] // 7))
            if E > 0:
                extra = int(math.ceil(E / 24.0))
                L = wdays + extra
                while L > wdays:
                    starts = []
                    for d0 in range(0, H - L + 1):
                        if all(cap[d0 + k] != 0 for k in range(L)):
                            starts.append(d0)
                    if starts:
                        mvars = []
                        for d0 in starts:
                            mv = m.addVar(lb=0.0, ub=1.0, name=f"m_{cid}_{d0}")
                            mvars.append(mv)
                            for k in range(L):
                                dd = d0 + k
                                if cap[dd] == 2:
                                    m.addConstr(mv <= fvar[dd])
                        m.addConstr(gp.quicksum(mvars) >= 1)
                        break
                    L -= 1  # relax if pre-assignments make it impossible

            # consecutive early departures
            early_by_day = defaultdict(list)
            for pid in elig:
                p = p_by_id[pid]
                if p["departure_hour"] < 8.0:
                    early_by_day[p["start_day"]].append(x[(cid, pid)])
            for d0 in range(0, H - max_early):
                vs = []
                for k in range(max_early + 1):
                    vs.extend(early_by_day.get(d0 + k, []))
                if len(vs) > max_early:
                    m.addConstr(gp.quicksum(vs) <= max_early)

            # monthly flight limit
            fh_expr = gp.quicksum(p_by_id[pid]["flight_hours"] * x[(cid, pid)] for pid in elig)
            if elig:
                m.addConstr(fh_expr <= mfl)

            # rolling flight limit (first 15 days of current month)
            roll = [pid for pid in elig if p_by_id[pid]["start_day"] <= 14]
            if roll:
                rhs = rfl - info["prev15"]
                if rhs <= 0:
                    for pid in roll:
                        x[(cid, pid)].UB = 0.0
                else:
                    m.addConstr(
                        gp.quicksum(p_by_id[pid]["flight_hours"] * x[(cid, pid)] for pid in roll) <= rhs
                    )

            # cross pairings
            cross = [x[(cid, pid)] for pid in elig if p_by_id[pid].get("is_cross_pairing", False)]
            if len(cross) > max_cross:
                m.addConstr(gp.quicksum(cross) <= max_cross)

        # ---------- Callback for incumbent logging ----------
        holder = {"obj": best_obj, "sol": best_sol}

        def cb(model, where):
            if where == GRB.Callback.MIPSOL:
                try:
                    vals = model.cbGetSolution(xvars)
                    assign = defaultdict(list)
                    for v, (cid, pid) in zip(vals, xmeta):
                        if v > 0.5:
                            assign[cid].append(pid)
                    obj, sol = build_solution(assign)
                    if obj < holder["obj"] - 1e-6:
                        holder["obj"] = obj
                        holder["sol"] = sol
                        if logger:
                            logger.log_solution(obj, sol)
                except Exception:
                    pass

        elapsed = time.time() - t0
        remaining = max(1.0, args.time_limit - elapsed - 3.0)
        m.Params.TimeLimit = remaining

        try:
            m.optimize(cb)
        except Exception:
            pass

        # Extract final solution from model if available (should match last incumbent)
        try:
            if m.SolCount > 0:
                assign = defaultdict(list)
                for v, (cid, pid) in zip(xvars, xmeta):
                    if v.X > 0.5:
                        assign[cid].append(pid)
                obj, sol = build_solution(assign)
                if obj < holder["obj"] - 1e-6:
                    holder["obj"] = obj
                    holder["sol"] = sol
                    if logger:
                        logger.log_solution(obj, sol)
        except Exception:
            pass

        best_obj = holder["obj"]
        best_sol = holder["sol"]

    except Exception:
        # Fall back to trivial solution already computed
        pass

    with open(args.solution_path, "w") as f:
        json.dump(best_sol, f, indent=2)


if __name__ == "__main__":
    main()
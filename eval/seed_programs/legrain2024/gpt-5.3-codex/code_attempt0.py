import argparse
import json
import math
import re
from typing import Dict, List, Tuple, Any

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def parse_args():
    parser = argparse.ArgumentParser(description="Nurse rostering solver (Gurobi).")
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    return parser.parse_args()


def load_instance(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def extract_day_count(inst: Dict[str, Any]) -> int:
    max_day = 0

    demand = inst.get("demand", {})
    for k in demand.keys():
        m = re.match(r"day_(\d+)$", k)
        if m:
            max_day = max(max_day, int(m.group(1)))

    for nurse in inst.get("nurses", []):
        for p in nurse.get("preferences", []):
            d = p.get("day")
            if isinstance(d, int):
                max_day = max(max_day, d)

    for w in inst.get("weekends", []):
        for kk, vv in w.items():
            if kk == "week":
                continue
            if isinstance(vv, int):
                max_day = max(max_day, vv)

    return max_day


def weekend_dow_set(weekend_definition: str):
    wd = (weekend_definition or "").lower()
    # Default Sat/Sun
    s = {5, 6}
    if "fri" in wd:
        s.add(4)
    return s


def daytype_matches(day: int, day_type: str, wkd_set: set) -> bool:
    if day_type is None:
        return True
    t = str(day_type).strip().lower()
    if t in ("any", "*", ""):
        return True

    dow = (day - 1) % 7  # 0=Mon ... 6=Sun
    name_map = {
        "monday": 0, "mon": 0,
        "tuesday": 1, "tue": 1, "tues": 1,
        "wednesday": 2, "wed": 2,
        "thursday": 3, "thu": 3, "thur": 3, "thurs": 3,
        "friday": 4, "fri": 4,
        "saturday": 5, "sat": 5,
        "sunday": 6, "sun": 6
    }
    if t in name_map:
        return dow == name_map[t]
    if t in ("weekday", "workingday"):
        return dow not in wkd_set
    if t in ("weekend",):
        return dow in wkd_set

    # Unknown token: treat as unrestricted to stay robust
    return True


def build_solution_from_model(
    nurses_ids: List[str],
    days: List[int],
    shifts: List[str],
    skills: List[str],
    o_vars: Dict[Tuple[str, int], gp.Var],
    w_vars: Dict[Tuple[str, int, str], gp.Var],
    z_vars: Dict[Tuple[str, int, str, str], gp.Var],
    get_val,
):
    schedule = {}
    skill_assignments = {}

    for n in nurses_ids:
        schedule[n] = {}
        skill_assignments[n] = {}
        for d in days:
            off_v = get_val(o_vars[(n, d)])
            if off_v > 0.5:
                schedule[n][f"day_{d}"] = "None"
                skill_assignments[n][f"day_{d}"] = "None"
                continue

            # Pick max shift
            best_s = None
            best_sv = -1.0
            for s in shifts:
                v = get_val(w_vars[(n, d, s)])
                if v > best_sv:
                    best_sv = v
                    best_s = s

            if best_s is None or best_sv < 0.5:
                # fallback
                schedule[n][f"day_{d}"] = "None"
                skill_assignments[n][f"day_{d}"] = "None"
                continue

            schedule[n][f"day_{d}"] = best_s

            best_k = None
            best_kv = -1.0
            for k in skills:
                var = z_vars.get((n, d, best_s, k))
                if var is None:
                    continue
                kv = get_val(var)
                if kv > best_kv:
                    best_kv = kv
                    best_k = k

            if best_k is None:
                # If numerically weird fallback
                skill_assignments[n][f"day_{d}"] = skills[0] if skills else "None"
            else:
                skill_assignments[n][f"day_{d}"] = best_k

    return schedule, skill_assignments


def solve(inst: Dict[str, Any], time_limit: int, logger: SolutionLogger = None):
    shifts = list(inst.get("shifts", []))
    skills = list(inst.get("skills", []))
    nurses = inst.get("nurses", [])
    nurses_ids = [n["id"] for n in nurses]
    nurse_skillset = {n["id"]: set(n.get("skills", [])) for n in nurses}

    D = extract_day_count(inst)
    days = list(range(1, D + 1))

    constraints = inst.get("constraints", {})
    hard = constraints.get("hard_constraints", {})
    soft = constraints.get("soft_constraints", {})

    demand = inst.get("demand", {})
    unsat_penalty = float(inst.get("unsatisfied_demand_penalty", 0))

    demand_equality = bool(hard.get("demand_equality", False))
    hard_skill_req = bool(hard.get("skill_requirement", False))

    weekend_definition = inst.get("weekend_definition", "")
    wkd_set = weekend_dow_set(weekend_definition)

    weekends_raw = inst.get("weekends", [])
    weekend_days = []
    for w in weekends_raw:
        ds = []
        for kk, vv in w.items():
            if kk == "week":
                continue
            if isinstance(vv, int):
                ds.append(vv)
        ds = sorted(set(ds))
        if ds:
            weekend_days.append(ds)
    W = len(weekend_days)

    m = gp.Model("nurse_rostering")

    # Required parameter settings
    m.Params.Seed = 0
    m.Params.MIPGap = 1e-4
    m.Params.NumericFocus = 0
    m.Params.Threads = 1
    m.Params.TimeLimit = int(time_limit)

    # Variables
    o = {}   # off day binary
    wd = {}  # working day binary
    w = {}   # shift assignment (work shift) binary
    z = {}   # shift-skill assignment binary

    for n in nurses_ids:
        skill_domain = skills if not hard_skill_req else [k for k in skills if k in nurse_skillset[n]]
        for d in days:
            o[(n, d)] = m.addVar(vtype=GRB.BINARY, name=f"o[{n},{d}]")
            wd[(n, d)] = m.addVar(vtype=GRB.BINARY, name=f"wd[{n},{d}]")
            for s in shifts:
                w[(n, d, s)] = m.addVar(vtype=GRB.BINARY, name=f"w[{n},{d},{s}]")
                for k in skill_domain:
                    z[(n, d, s, k)] = m.addVar(vtype=GRB.BINARY, name=f"z[{n},{d},{s},{k}]")

    # Basic assignment constraints
    for n in nurses_ids:
        skill_domain = skills if not hard_skill_req else [k for k in skills if k in nurse_skillset[n]]
        for d in days:
            m.addConstr(wd[(n, d)] == gp.quicksum(w[(n, d, s)] for s in shifts), name=f"wd_link[{n},{d}]")
            m.addConstr(o[(n, d)] + wd[(n, d)] == 1, name=f"one_shift[{n},{d}]")
            for s in shifts:
                m.addConstr(
                    gp.quicksum(z[(n, d, s, k)] for k in skill_domain) == w[(n, d, s)],
                    name=f"skill_one[{n},{d},{s}]"
                )

    obj = gp.LinExpr()

    # Demand constraints
    if D > 0:
        for d in days:
            dkey = f"day_{d}"
            day_dem = demand.get(dkey, {})
            for s in shifts:
                shift_dem = day_dem.get(s, {})
                for k in skills:
                    req = float(shift_dem.get(k, 0))
                    staff = gp.quicksum(z_var for (nn, dd, ss, kk), z_var in z.items()
                                        if dd == d and ss == s and kk == k)
                    if demand_equality:
                        m.addConstr(staff == req, name=f"demand_eq[{d},{s},{k}]")
                    else:
                        slack = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"slack[{d},{s},{k}]")
                        m.addConstr(staff + slack >= req, name=f"demand_ge[{d},{s},{k}]")
                        if unsat_penalty != 0:
                            obj += unsat_penalty * slack

    # Skill mismatch penalty (only relevant when skill requirement is soft)
    if not hard_skill_req:
        skill_pen = float(soft.get("skill_requirement_penalty", 0) or 0)
        if skill_pen != 0:
            for (n, d, s, k), var in z.items():
                if k not in nurse_skillset[n]:
                    obj += skill_pen * var

    # Total working days soft constraint
    tw = soft.get("total_working_days")
    if isinstance(tw, dict):
        tmin = tw.get("min", None)
        tmax = tw.get("max", None)
        p_under = float(tw.get("penalty_under", 0) or 0)
        p_over = float(tw.get("penalty_over", 0) or 0)

        for n in nurses_ids:
            total_work = gp.quicksum(wd[(n, d)] for d in days)
            if tmin is not None:
                u = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"tw_under[{n}]")
                m.addConstr(u >= float(tmin) - total_work, name=f"tw_under_c[{n}]")
                if p_under != 0:
                    obj += p_under * u
            if tmax is not None:
                v = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"tw_over[{n}]")
                m.addConstr(v >= total_work - float(tmax), name=f"tw_over_c[{n}]")
                if p_over != 0:
                    obj += p_over * v

    # Weekend worked indicators
    ww = {}
    if W > 0:
        for n in nurses_ids:
            for wi, ds in enumerate(weekend_days):
                ww[(n, wi)] = m.addVar(vtype=GRB.BINARY, name=f"ww[{n},{wi}]")
                for d in ds:
                    if 1 <= d <= D:
                        m.addConstr(ww[(n, wi)] >= wd[(n, d)], name=f"ww_lb[{n},{wi},{d}]")
                valid_ds = [d for d in ds if 1 <= d <= D]
                if valid_ds:
                    m.addConstr(
                        ww[(n, wi)] <= gp.quicksum(wd[(n, d)] for d in valid_ds),
                        name=f"ww_ub[{n},{wi}]"
                    )
                else:
                    m.addConstr(ww[(n, wi)] == 0, name=f"ww_zero[{n},{wi}]")

    # Total worked weekends max
    tww = soft.get("total_worked_weekends")
    if isinstance(tww, dict) and W > 0:
        max_w = tww.get("max", None)
        pen_w = float(tww.get("penalty", 0) or 0)
        if max_w is not None:
            for n in nurses_ids:
                total_ww = gp.quicksum(ww[(n, wi)] for wi in range(W))
                ov = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"tww_over[{n}]")
                m.addConstr(ov >= total_ww - float(max_w), name=f"tww_over_c[{n}]")
                if pen_w != 0:
                    obj += pen_w * ov

    # Consecutive working days
    cwd = soft.get("consecutive_working_days")
    if isinstance(cwd, dict) and D > 0:
        cmin = int(cwd.get("min", 0) or 0)
        cmax = int(cwd.get("max", 10**9) or 10**9)
        p_under = float(cwd.get("penalty_under", 0) or 0)
        p_over = float(cwd.get("penalty_over", 0) or 0)

        # under-min internal streaks only
        if cmin > 1 and p_under != 0:
            for n in nurses_ids:
                for L in range(1, cmin):
                    for t in range(2, D - L + 1):  # internal only: has day before and after
                        if t + L > D:
                            continue
                        if t - 1 < 1 or t + L > D:
                            continue
                        var = m.addVar(vtype=GRB.BINARY, name=f"cwd_under[{n},{t},{L}]")
                        lits = [o[(n, t - 1)], o[(n, t + L)]]
                        lits += [wd[(n, dd)] for dd in range(t, t + L)]
                        for lit in lits:
                            m.addConstr(var <= lit)
                        m.addConstr(var >= gp.quicksum(lits) - len(lits) + 1)
                        obj += p_under * var

        # over-max via windows of length cmax+1 all worked
        if cmax >= 0 and cmax < D and p_over != 0:
            Lw = cmax + 1
            for n in nurses_ids:
                for t in range(1, D - Lw + 2):
                    var = m.addVar(vtype=GRB.BINARY, name=f"cwd_over[{n},{t}]")
                    lits = [wd[(n, dd)] for dd in range(t, t + Lw)]
                    for lit in lits:
                        m.addConstr(var <= lit)
                    m.addConstr(var >= gp.quicksum(lits) - Lw + 1)
                    obj += p_over * var

    # Consecutive days off
    cdo = soft.get("consecutive_days_off")
    if isinstance(cdo, dict) and D > 0:
        cmin = int(cdo.get("min", 0) or 0)
        cmax = int(cdo.get("max", 10**9) or 10**9)
        p_under = float(cdo.get("penalty_under", 0) or 0)
        p_over = float(cdo.get("penalty_over", 0) or 0)

        # under-min internal off streaks only
        if cmin > 1 and p_under != 0:
            for n in nurses_ids:
                for L in range(1, cmin):
                    for t in range(2, D - L + 1):
                        if t + L > D:
                            continue
                        var = m.addVar(vtype=GRB.BINARY, name=f"cdo_under[{n},{t},{L}]")
                        lits = [wd[(n, t - 1)], wd[(n, t + L)]]
                        lits += [o[(n, dd)] for dd in range(t, t + L)]
                        for lit in lits:
                            m.addConstr(var <= lit)
                        m.addConstr(var >= gp.quicksum(lits) - len(lits) + 1)
                        obj += p_under * var

        # over-max via windows of length cmax+1 all off
        if cmax >= 0 and cmax < D and p_over != 0:
            Lw = cmax + 1
            for n in nurses_ids:
                for t in range(1, D - Lw + 2):
                    var = m.addVar(vtype=GRB.BINARY, name=f"cdo_over[{n},{t}]")
                    lits = [o[(n, dd)] for dd in range(t, t + Lw)]
                    for lit in lits:
                        m.addConstr(var <= lit)
                    m.addConstr(var >= gp.quicksum(lits) - Lw + 1)
                    obj += p_over * var

    # Consecutive same shift
    css = soft.get("consecutive_same_shift")
    if isinstance(css, dict) and D > 0:
        cmax = int(css.get("max", 10**9) or 10**9)
        pen = float(css.get("penalty", 0) or 0)
        if cmax >= 0 and cmax < D and pen != 0:
            Lw = cmax + 1
            for n in nurses_ids:
                for s in shifts:
                    for t in range(1, D - Lw + 2):
                        var = m.addVar(vtype=GRB.BINARY, name=f"css_over[{n},{s},{t}]")
                        lits = [w[(n, dd, s)] for dd in range(t, t + Lw)]
                        for lit in lits:
                            m.addConstr(var <= lit)
                        m.addConstr(var >= gp.quicksum(lits) - Lw + 1)
                        obj += pen * var

    # Consecutive worked weekends
    cww = soft.get("consecutive_worked_weekends")
    if isinstance(cww, dict) and W > 0:
        cmax = int(cww.get("max", 10**9) or 10**9)
        pen = float(cww.get("penalty", 0) or 0)
        if cmax >= 0 and cmax < W and pen != 0:
            Lw = cmax + 1
            for n in nurses_ids:
                for t in range(0, W - Lw + 1):
                    var = m.addVar(vtype=GRB.BINARY, name=f"cww_over[{n},{t}]")
                    lits = [ww[(n, wi)] for wi in range(t, t + Lw)]
                    for lit in lits:
                        m.addConstr(var <= lit)
                    m.addConstr(var >= gp.quicksum(lits) - Lw + 1)
                    obj += pen * var

    # Identical weekend
    iw = soft.get("identical_weekend")
    if isinstance(iw, dict) and iw.get("enabled", False) and W > 0:
        pen = float(iw.get("penalty", 0) or 0)
        if pen != 0:
            all_shift_labels = shifts + ["None"]
            for n in nurses_ids:
                for wi, ds in enumerate(weekend_days):
                    valid_ds = [d for d in ds if 1 <= d <= D]
                    if len(valid_ds) < 2:
                        continue
                    viol = m.addVar(vtype=GRB.BINARY, name=f"iw_viol[{n},{wi}]")
                    for i in range(len(valid_ds)):
                        for j in range(i + 1, len(valid_ds)):
                            d1, d2 = valid_ds[i], valid_ds[j]
                            for sh in all_shift_labels:
                                v1 = o[(n, d1)] if sh == "None" else w[(n, d1, sh)]
                                v2 = o[(n, d2)] if sh == "None" else w[(n, d2, sh)]
                                m.addConstr(v1 - v2 <= viol, name=f"iw1[{n},{wi},{d1},{d2},{sh}]")
                                m.addConstr(v2 - v1 <= viol, name=f"iw2[{n},{wi},{d1},{d2},{sh}]")
                    obj += pen * viol

    # Preferences
    pref_flag = soft.get("preferences_penalty", True)
    if pref_flag is not False:
        pref_mult = 1.0
        if isinstance(pref_flag, (int, float)):
            pref_mult = float(pref_flag)
        for nurse in nurses:
            n = nurse["id"]
            for p in nurse.get("preferences", []):
                d = p.get("day", None)
                if not isinstance(d, int) or d < 1 or d > D:
                    continue
                ptype = str(p.get("type", "")).strip().lower()
                sh = p.get("shift", "None")
                pen = float(p.get("penalty", 0) or 0) * pref_mult
                if pen == 0:
                    continue

                if ptype == "day_off":
                    obj += pen * wd[(n, d)]  # unmet iff works
                elif ptype == "day_on":
                    obj += pen * o[(n, d)]   # unmet iff off
                elif ptype == "shift_on":
                    if sh in shifts:
                        obj += pen * (1 - w[(n, d, sh)])
                    else:
                        # impossible request => always unmet
                        obj += pen
                elif ptype == "shift_off":
                    if sh in shifts:
                        obj += pen * w[(n, d, sh)]
                    else:
                        # requesting off for unknown shift => always honored
                        pass

    # Forbidden patterns (hard + soft)
    def add_pattern_terms(pattern_obj_list, hard_flag: bool):
        nonlocal obj
        for idx, pobj in enumerate(pattern_obj_list):
            pattern = pobj.get("pattern", [])
            L = len(pattern)
            if L == 0 or D == 0 or L > D:
                continue

            pen = None
            if not hard_flag:
                pen = float(pobj.get("penalty", pobj.get("cost", 0)) or 0)

            for n in nurses_ids:
                for t in range(1, D - L + 2):
                    impossible = False
                    lits = []
                    for j in range(L):
                        day = t + j
                        token = pattern[j]
                        if not isinstance(token, list) or len(token) < 2:
                            continue
                        day_type = token[0]
                        sh = token[1]

                        if not daytype_matches(day, day_type, wkd_set):
                            impossible = True
                            break

                        shs = str(sh)
                        if shs.lower() == "any":
                            continue
                        if shs == "None":
                            lits.append(o[(n, day)])
                        elif shs in shifts:
                            lits.append(w[(n, day, shs)])
                        else:
                            # unknown shift => cannot match
                            impossible = True
                            break

                    if impossible:
                        continue

                    if len(lits) == 0:
                        # Pattern always occurs at this start
                        if hard_flag:
                            m.addConstr(0 == 1, name=f"hard_forbid_const[{idx},{n},{t}]")
                        else:
                            if pen:
                                obj += pen
                        continue

                    occ = m.addVar(vtype=GRB.BINARY, name=f"pat_occ[{int(hard_flag)},{idx},{n},{t}]")
                    for lit in lits:
                        m.addConstr(occ <= lit)
                    m.addConstr(occ >= gp.quicksum(lits) - len(lits) + 1)

                    if hard_flag:
                        m.addConstr(occ == 0, name=f"hard_forbid[{idx},{n},{t}]")
                    else:
                        if pen:
                            obj += pen * occ

    hard_forbidden = hard.get("forbidden_patterns", []) or []
    add_pattern_terms(hard_forbidden, hard_flag=True)

    soft_forbidden = soft.get("forbidden_patterns", []) or []
    add_pattern_terms(soft_forbidden, hard_flag=False)

    m.setObjective(obj, GRB.MINIMIZE)

    # Incumbent logging callback
    best_logged = {"obj": math.inf}

    def cb(model, where):
        if where == GRB.Callback.MIPSOL and logger is not None:
            try:
                cur_obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if cur_obj + 1e-9 < best_logged["obj"]:
                    def gv(var):
                        return model.cbGetSolution(var)

                    schedule, skill_assign = build_solution_from_model(
                        nurses_ids, days, shifts, skills, o, w, z, gv
                    )
                    sol = {
                        "objective_value": float(cur_obj),
                        "schedule": schedule,
                        "skill_assignments": skill_assign
                    }
                    logger.log_solution(float(cur_obj), sol)
                    best_logged["obj"] = cur_obj
            except Exception:
                # Never crash optimization from callback logging issues
                pass

    m.optimize(cb)

    if m.SolCount > 0:
        def gvx(var):
            return var.X

        schedule, skill_assign = build_solution_from_model(
            nurses_ids, days, shifts, skills, o, w, z, gvx
        )
        objective_value = float(m.ObjVal)
    else:
        # Fallback schema-compliant output
        schedule = {}
        skill_assign = {}
        for n in nurses_ids:
            schedule[n] = {f"day_{d}": "None" for d in days}
            skill_assign[n] = {f"day_{d}": "None" for d in days}
        objective_value = float("inf")

    solution = {
        "objective_value": objective_value,
        "schedule": schedule,
        "skill_assignments": skill_assign
    }

    return solution


def main():
    args = parse_args()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    inst = load_instance(args.instance_path)
    sol = solve(inst, args.time_limit, logger=logger)

    with open(args.solution_path, "w") as f:
        json.dump(sol, f, indent=2)


if __name__ == "__main__":
    main()
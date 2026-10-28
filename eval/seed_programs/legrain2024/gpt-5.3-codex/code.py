import argparse
import json
import re
import time
from collections import defaultdict

HARD_PENALTY = 1_000_000.0


def parse_args():
    p = argparse.ArgumentParser(description="Nurse rostering solver")
    p.add_argument("--instance_path", type=str, required=True)
    p.add_argument("--solution_path", type=str, required=True)
    p.add_argument("--time_limit", type=int, required=True)
    p.add_argument("--log_path", type=str, default=None)
    return p.parse_args()


def safe_init_logger(log_path):
    """Initialize solution logger safely; never crash if unavailable."""
    if not log_path:
        return None
    try:
        from solution_logger import SolutionLogger
        return SolutionLogger(log_path, sense="minimize")
    except BaseException:
        return None


def load_instance(path):
    with open(path, "r") as f:
        return json.load(f)


def normalize_shift(v):
    if v is None:
        return "None"
    s = str(v)
    return "None" if s == "None" else s


def extract_day_count(inst):
    mx = 0
    for k in inst.get("demand", {}).keys():
        m = re.match(r"day_(\d+)$", str(k))
        if m:
            mx = max(mx, int(m.group(1)))
    for n in inst.get("nurses", []):
        for p in n.get("preferences", []):
            d = p.get("day")
            if isinstance(d, int):
                mx = max(mx, d)
    for w in inst.get("weekends", []):
        for kk, vv in w.items():
            if kk != "week" and isinstance(vv, int):
                mx = max(mx, vv)
    return mx


def weekend_dayset(weekend_definition):
    txt = str(weekend_definition or "").lower()
    ds = {5, 6}  # Saturday/Sunday
    if "fri" in txt:
        ds.add(4)
    return ds


def build_weekends(weekends_raw):
    weekends = []
    for w in weekends_raw:
        ds = []
        for k, v in w.items():
            if k != "week" and isinstance(v, int):
                ds.append(v)
        ds = sorted(set(ds))
        if ds:
            weekends.append(ds)
    return weekends


def daytype_matches(day, day_type, weekend_set):
    if day_type is None:
        return True
    t = str(day_type).strip().lower()
    if t in ("", "any", "*"):
        return True

    dow = (day - 1) % 7  # Mon=0
    name_to_idx = {
        "mon": 0, "monday": 0,
        "tue": 1, "tues": 1, "tuesday": 1,
        "wed": 2, "wednesday": 2,
        "thu": 3, "thur": 3, "thurs": 3, "thursday": 3,
        "fri": 4, "friday": 4,
        "sat": 5, "saturday": 5,
        "sun": 6, "sunday": 6,
    }
    if t in name_to_idx:
        return dow == name_to_idx[t]
    if t == "weekend":
        return dow in weekend_set
    if t in ("weekday", "workingday"):
        return dow not in weekend_set
    return True


def preference_penalty(preferences, assigned_shift):
    assigned_shift = normalize_shift(assigned_shift)
    c = 0.0
    for p in preferences:
        ptype = str(p.get("type", "")).lower()
        target = normalize_shift(p.get("shift", "None"))
        pen = float(p.get("penalty", 0) or 0)
        unmet = False
        if ptype == "day_off":
            unmet = assigned_shift != "None"
        elif ptype == "day_on":
            unmet = assigned_shift == "None"
        elif ptype == "shift_on":
            unmet = assigned_shift != target
        elif ptype == "shift_off":
            unmet = assigned_shift == target
        if unmet:
            c += pen
    return c


def closes_hard_pattern(nid, day, new_shift, partial_schedule, hard_patterns, weekend_set):
    for pobj in hard_patterns:
        pat = pobj.get("pattern", [])
        L = len(pat)
        if L <= 0:
            continue
        start = day - L + 1
        if start < 1:
            continue
        ok = True
        for j in range(L):
            d = start + j
            tok = pat[j]
            if not isinstance(tok, list) or len(tok) < 2:
                continue
            dt, req = tok[0], normalize_shift(tok[1])
            if not daytype_matches(d, dt, weekend_set):
                ok = False
                break
            got = new_shift if d == day else partial_schedule[nid].get(d, "None")
            got = normalize_shift(got)
            if req.lower() != "any" and got != req:
                ok = False
                break
        if ok:
            return True
    return False


def evaluate(inst, schedule, skill_assignments):
    shifts = [str(s) for s in inst.get("shifts", [])]
    skills = [str(k) for k in inst.get("skills", [])]
    nurses = inst.get("nurses", [])
    nurse_ids = [n["id"] for n in nurses]
    nurse_skillset = {n["id"]: set(map(str, n.get("skills", []))) for n in nurses}
    nurse_map = {n["id"]: n for n in nurses}

    constraints = inst.get("constraints", {})
    hard = constraints.get("hard_constraints", {})
    soft = constraints.get("soft_constraints", {})
    demand = inst.get("demand", {})

    hard_skill = bool(hard.get("skill_requirement", False))
    demand_eq = bool(hard.get("demand_equality", False))
    unsat_pen = float(inst.get("unsatisfied_demand_penalty", 0) or 0)
    soft_skill_pen = float(soft.get("skill_requirement_penalty", 0) or 0)

    D = extract_day_count(inst)
    days = list(range(1, D + 1))
    weekend_set = weekend_dayset(inst.get("weekend_definition", ""))
    weekends = build_weekends(inst.get("weekends", []))
    W = len(weekends)

    obj = 0.0

    # certificate shape & consistency
    for nid in nurse_ids:
        if nid not in schedule or nid not in skill_assignments:
            obj += HARD_PENALTY
            continue
        for d in days:
            k = f"day_{d}"
            if k not in schedule[nid] or k not in skill_assignments[nid]:
                obj += HARD_PENALTY
                continue
            sh = normalize_shift(schedule[nid][k])
            sk = normalize_shift(skill_assignments[nid][k])
            if sh == "None":
                if sk != "None":
                    obj += HARD_PENALTY
            else:
                if sh not in shifts:
                    obj += HARD_PENALTY
                if sk == "None" or sk not in skills:
                    obj += HARD_PENALTY

    # demand penalties
    for d in days:
        dd = demand.get(f"day_{d}", {})
        for s in shifts:
            sd = dd.get(s, {})
            for k in skills:
                req = float(sd.get(k, 0) or 0)
                staffed = 0.0
                day_key = f"day_{d}"
                for nid in nurse_ids:
                    if normalize_shift(schedule[nid][day_key]) == s and normalize_shift(skill_assignments[nid][day_key]) == k:
                        staffed += 1.0
                if demand_eq:
                    obj += HARD_PENALTY * abs(staffed - req)
                else:
                    obj += unsat_pen * max(0.0, req - staffed)

    # skill mismatch
    for nid in nurse_ids:
        own = nurse_skillset[nid]
        for d in days:
            k = f"day_{d}"
            sh = normalize_shift(schedule[nid][k])
            sk = normalize_shift(skill_assignments[nid][k])
            if sh != "None" and sk not in own:
                obj += HARD_PENALTY if hard_skill else soft_skill_pen

    # preferences
    pref_flag = soft.get("preferences_penalty", True)
    if pref_flag is not False:
        mult = float(pref_flag) if isinstance(pref_flag, (int, float)) else 1.0
        for nid in nurse_ids:
            for p in nurse_map[nid].get("preferences", []):
                d = p.get("day")
                if not isinstance(d, int) or d < 1 or d > D:
                    continue
                sh = normalize_shift(schedule[nid][f"day_{d}"])
                ptype = str(p.get("type", "")).lower()
                target = normalize_shift(p.get("shift", "None"))
                pen = float(p.get("penalty", 0) or 0) * mult
                unmet = False
                if ptype == "day_off":
                    unmet = sh != "None"
                elif ptype == "day_on":
                    unmet = sh == "None"
                elif ptype == "shift_on":
                    unmet = sh != target
                elif ptype == "shift_off":
                    unmet = sh == target
                if unmet:
                    obj += pen

    # total working days
    tw = soft.get("total_working_days")
    if isinstance(tw, dict):
        mn = tw.get("min", None)
        mx = tw.get("max", None)
        pu = float(tw.get("penalty_under", 0) or 0)
        po = float(tw.get("penalty_over", 0) or 0)
        for nid in nurse_ids:
            w = sum(1 for d in days if normalize_shift(schedule[nid][f"day_{d}"]) != "None")
            if mn is not None and w < int(mn):
                obj += pu * (int(mn) - w)
            if mx is not None and w > int(mx):
                obj += po * (w - int(mx))

    # worked weekends arrays
    weekend_work = {nid: [] for nid in nurse_ids}
    for nid in nurse_ids:
        for ds in weekends:
            wk = 0
            for d in ds:
                if 1 <= d <= D and normalize_shift(schedule[nid][f"day_{d}"]) != "None":
                    wk = 1
                    break
            weekend_work[nid].append(wk)

    # total worked weekends
    tww = soft.get("total_worked_weekends")
    if isinstance(tww, dict):
        mx = tww.get("max", None)
        pen = float(tww.get("penalty", 0) or 0)
        if mx is not None:
            mx = int(mx)
            for nid in nurse_ids:
                c = sum(weekend_work[nid])
                if c > mx:
                    obj += pen * (c - mx)

    # consecutive worked weekends
    cww = soft.get("consecutive_worked_weekends")
    if isinstance(cww, dict):
        mx = int(cww.get("max", 10**9) or 10**9)
        pen = float(cww.get("penalty", 0) or 0)
        if mx < 10**9 and W > 0:
            for nid in nurse_ids:
                arr = weekend_work[nid]
                i = 0
                while i < len(arr):
                    if arr[i] == 1:
                        j = i
                        while j < len(arr) and arr[j] == 1:
                            j += 1
                        L = j - i
                        if L > mx:
                            obj += pen * (L - mx)
                        i = j
                    else:
                        i += 1

    # identical weekend
    iw = soft.get("identical_weekend")
    if isinstance(iw, dict) and iw.get("enabled", False):
        pen = float(iw.get("penalty", 0) or 0)
        for nid in nurse_ids:
            for ds in weekends:
                vals = [normalize_shift(schedule[nid][f"day_{d}"]) for d in ds if 1 <= d <= D]
                if len(vals) >= 2 and len(set(vals)) > 1:
                    obj += pen

    # run helper
    def runs(arr):
        if not arr:
            return []
        out = []
        st = 0
        cur = arr[0]
        for i in range(1, len(arr)):
            if arr[i] != cur:
                out.append((st + 1, i, cur))
                st = i
                cur = arr[i]
        out.append((st + 1, len(arr), cur))
        return out

    cwd = soft.get("consecutive_working_days")
    cdo = soft.get("consecutive_days_off")

    for nid in nurse_ids:
        b = [1 if normalize_shift(schedule[nid][f"day_{d}"]) != "None" else 0 for d in days]
        rr = runs(b)

        if isinstance(cwd, dict):
            mn = int(cwd.get("min", 0) or 0)
            mx = int(cwd.get("max", 10**9) or 10**9)
            pu = float(cwd.get("penalty_under", 0) or 0)
            po = float(cwd.get("penalty_over", 0) or 0)
            for sday, eday, v in rr:
                if v != 1:
                    continue
                L = eday - sday + 1
                boundary = (sday == 1 or eday == D)
                if mn > 0 and L < mn and not boundary:
                    obj += pu
                if L > mx:
                    obj += po * (L - mx)

        if isinstance(cdo, dict):
            mn = int(cdo.get("min", 0) or 0)
            mx = int(cdo.get("max", 10**9) or 10**9)
            pu = float(cdo.get("penalty_under", 0) or 0)
            po = float(cdo.get("penalty_over", 0) or 0)
            for sday, eday, v in rr:
                if v != 0:
                    continue
                L = eday - sday + 1
                boundary = (sday == 1 or eday == D)
                if mn > 0 and L < mn and not boundary:
                    obj += pu
                if L > mx:
                    obj += po * (L - mx)

    # consecutive same shift
    css = soft.get("consecutive_same_shift")
    if isinstance(css, dict):
        mx = int(css.get("max", 10**9) or 10**9)
        pen = float(css.get("penalty", 0) or 0)
        if mx < 10**9:
            for nid in nurse_ids:
                for s in shifts:
                    L = 0
                    for d in days:
                        if normalize_shift(schedule[nid][f"day_{d}"]) == s:
                            L += 1
                        else:
                            if L > mx:
                                obj += pen * (L - mx)
                            L = 0
                    if L > mx:
                        obj += pen * (L - mx)

    # forbidden pattern counts
    def pattern_count(pattern):
        L = len(pattern)
        if L <= 0:
            return 0
        cnt = 0
        for nid in nurse_ids:
            for t in range(1, D - L + 2):
                ok = True
                for j in range(L):
                    day = t + j
                    tok = pattern[j]
                    if not isinstance(tok, list) or len(tok) < 2:
                        continue
                    dt, req = tok[0], normalize_shift(tok[1])
                    if not daytype_matches(day, dt, weekend_set):
                        ok = False
                        break
                    sh = normalize_shift(schedule[nid][f"day_{day}"])
                    if req.lower() != "any" and sh != req:
                        ok = False
                        break
                if ok:
                    cnt += 1
        return cnt

    for fp in hard.get("forbidden_patterns", []) or []:
        c = pattern_count(fp.get("pattern", []))
        if c > 0:
            obj += HARD_PENALTY * c
    for fp in soft.get("forbidden_patterns", []) or []:
        c = pattern_count(fp.get("pattern", []))
        pen = float(fp.get("penalty", fp.get("cost", 0)) or 0)
        obj += pen * c

    return float(obj)


def make_empty_solution(inst):
    nurses = inst.get("nurses", [])
    nurse_ids = [n["id"] for n in nurses]
    D = extract_day_count(inst)
    days = range(1, D + 1)
    schedule = {nid: {f"day_{d}": "None" for d in days} for nid in nurse_ids}
    skills = {nid: {f"day_{d}": "None" for d in days} for nid in nurse_ids}
    return schedule, skills


def construct_greedy(inst, start_time, time_limit):
    shifts = [str(s) for s in inst.get("shifts", [])]
    skills = [str(k) for k in inst.get("skills", [])]
    nurses = inst.get("nurses", [])
    nurse_ids = [n["id"] for n in nurses]
    nurse_skillset = {n["id"]: set(map(str, n.get("skills", []))) for n in nurses}

    constraints = inst.get("constraints", {})
    hard = constraints.get("hard_constraints", {})
    hard_skill = bool(hard.get("skill_requirement", False))
    hard_patterns = hard.get("forbidden_patterns", []) or []
    demand = inst.get("demand", {})

    D = extract_day_count(inst)
    days = list(range(1, D + 1))
    weekend_set = weekend_dayset(inst.get("weekend_definition", ""))
    weekends = build_weekends(inst.get("weekends", []))

    day_to_wi = {}
    for i, ds in enumerate(weekends):
        for d in ds:
            day_to_wi[d] = i

    pref_by_nd = defaultdict(list)
    for n in nurses:
        nid = n["id"]
        for p in n.get("preferences", []):
            d = p.get("day")
            if isinstance(d, int) and 1 <= d <= D:
                pref_by_nd[(nid, d)].append(p)

    sch = {nid: {d: "None" for d in days} for nid in nurse_ids}
    skl = {nid: {d: "None" for d in days} for nid in nurse_ids}
    total_work = {nid: 0 for nid in nurse_ids}
    weekend_worked = {nid: set() for nid in nurse_ids}

    # reserve a tiny tail for evaluation + output
    stop_at = start_time + max(1, time_limit - 1)

    for d in days:
        if time.time() >= stop_at:
            break

        unassigned = set(nurse_ids)
        dd = demand.get(f"day_{d}", {})
        units = []
        for s in shifts:
            sd = dd.get(s, {})
            for k in skills:
                req = int(sd.get(k, 0) or 0)
                for _ in range(max(0, req)):
                    units.append((s, k))

        # scarce skill first
        units.sort(key=lambda x: sum(1 for nid in nurse_ids if x[1] in nurse_skillset[nid]))

        for s, k in units:
            if time.time() >= stop_at or not unassigned:
                break

            cand = []
            for nid in unassigned:
                if hard_skill and (k not in nurse_skillset[nid]):
                    continue

                bad_pat = closes_hard_pattern(nid, d, s, sch, hard_patterns, weekend_set)

                pref_none = preference_penalty(pref_by_nd[(nid, d)], "None")
                pref_work = preference_penalty(pref_by_nd[(nid, d)], s)
                pref_delta = pref_work - pref_none

                w_bias = 0.2 * len(weekend_worked[nid]) if d in day_to_wi else 0.0
                score = total_work[nid] + pref_delta + w_bias + (10_000.0 if bad_pat else 0.0)
                cand.append((score, nid, bad_pat))

            if not cand:
                continue
            good = [x for x in cand if not x[2]]
            pool = good if good else cand
            pool.sort(key=lambda x: (x[0], x[1]))
            nid = pool[0][1]

            sch[nid][d] = s
            skl[nid][d] = k
            total_work[nid] += 1
            if d in day_to_wi:
                weekend_worked[nid].add(day_to_wi[d])
            unassigned.remove(nid)

    schedule = {nid: {f"day_{d}": sch[nid][d] for d in days} for nid in nurse_ids}
    skill_assignments = {nid: {f"day_{d}": skl[nid][d] for d in days} for nid in nurse_ids}
    return schedule, skill_assignments


def write_solution(path, obj, schedule, skill_assignments):
    out = {
        "objective_value": float(obj),
        "schedule": schedule,
        "skill_assignments": skill_assignments,
    }
    with open(path, "w") as f:
        json.dump(out, f, indent=2)


def main():
    # Absolute fail-safe: never crash with nonzero exit
    try:
        args = parse_args()
        logger = safe_init_logger(args.log_path)

        start = time.time()
        inst = load_instance(args.instance_path)

        # Build heuristic solution under time limit
        schedule, skill_assignments = construct_greedy(inst, start, args.time_limit)
        obj = evaluate(inst, schedule, skill_assignments)

        sol = {
            "objective_value": float(obj),
            "schedule": schedule,
            "skill_assignments": skill_assignments,
        }
        if logger:
            try:
                logger.log_solution(float(obj), sol)
            except BaseException:
                pass

        write_solution(args.solution_path, obj, schedule, skill_assignments)

    except BaseException:
        # Try best effort fallback output
        try:
            # If args parsing failed, we cannot know output path.
            # Re-parse in safest way.
            import sys
            ap = argparse.ArgumentParser(add_help=False)
            ap.add_argument("--instance_path", type=str, required=False)
            ap.add_argument("--solution_path", type=str, required=False)
            ap.add_argument("--time_limit", type=int, required=False)
            ap.add_argument("--log_path", type=str, required=False)
            known, _ = ap.parse_known_args(sys.argv[1:])
            instance_path = getattr(known, "instance_path", None)
            solution_path = getattr(known, "solution_path", None)
            if solution_path:
                if instance_path:
                    try:
                        inst = load_instance(instance_path)
                        schedule, skill_assignments = make_empty_solution(inst)
                        obj = evaluate(inst, schedule, skill_assignments)
                    except BaseException:
                        schedule, skill_assignments = {}, {}
                        obj = HARD_PENALTY
                else:
                    schedule, skill_assignments = {}, {}
                    obj = HARD_PENALTY
                write_solution(solution_path, obj, schedule, skill_assignments)
        except BaseException:
            pass
        return


if __name__ == "__main__":
    main()
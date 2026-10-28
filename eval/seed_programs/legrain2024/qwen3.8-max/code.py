import json
import argparse
import time as time_module
from collections import defaultdict


def safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value, default=None):
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def penalty_float(value, default=0.0):
    v = safe_float(value, default)
    return v if v > 0.0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Personalized nurse scheduling")
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    try:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    except Exception:
        logger = None

    start_time = time_module.time()

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    # ------------------------------------------------------------------
    # Basic instance parsing
    # ------------------------------------------------------------------
    rest_token = "None"

    shifts_raw = inst.get("shifts", [])
    work_shifts = [str(s) for s in shifts_raw if str(s) != rest_token]
    all_shifts = work_shifts + [rest_token]
    shift_set = set(work_shifts)
    shift_lower_map = {s.lower(): s for s in work_shifts}
    all_shifts_set = set(all_shifts)

    skills = [str(k) for k in inst.get("skills", [])]
    skill_set = set(skills)

    nurses_data = inst.get("nurses", [])
    nurse_ids = []
    nurse_skill_sets = {}
    raw_preferences = {}

    for i, nd in enumerate(nurses_data):
        nid = str(nd.get("id", f"nurse_{i}"))
        nurse_ids.append(nid)
        nurse_skill_sets[nid] = set(
            str(s) for s in nd.get("skills", []) if str(s) in skill_set
        )
        raw_preferences[nid] = nd.get("preferences", []) or []

    # Infer horizon length from demand, weekends, and preferences.
    num_days = 0

    demand_raw = inst.get("demand", {}) or {}
    for day_key in demand_raw.keys():
        d = safe_int(str(day_key).rsplit("_", 1)[-1], None)
        if d is not None and d > num_days:
            num_days = d

    weekends_raw = inst.get("weekends", []) or []
    for w in weekends_raw:
        for k in ("friday", "saturday", "sunday"):
            d = safe_int(w.get(k), None)
            if d is not None and d > num_days:
                num_days = d

    for nd in nurses_data:
        for pref in nd.get("preferences", []) or []:
            d = safe_int(pref.get("day"), None)
            if d is not None and d > num_days:
                num_days = d

    if num_days <= 0:
        num_days = 1

    days = list(range(1, num_days + 1))

    # Weekend structures.
    weekend_days_list = []
    all_weekend_days = set()
    friday_days = set()
    saturday_days = set()
    sunday_days = set()

    for w in weekends_raw:
        wd = []
        df = safe_int(w.get("friday"), None)
        ds = safe_int(w.get("saturday"), None)
        du = safe_int(w.get("sunday"), None)

        if df is not None and 1 <= df <= num_days:
            wd.append(df)
            friday_days.add(df)
        if ds is not None and 1 <= ds <= num_days:
            wd.append(ds)
            saturday_days.add(ds)
        if du is not None and 1 <= du <= num_days:
            wd.append(du)
            sunday_days.add(du)

        wd = sorted(set(wd))
        if wd:
            weekend_days_list.append(wd)
            all_weekend_days.update(wd)

    num_weekends = len(weekend_days_list)

    # Weekday inference for pattern day types.
    weekday_names = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6,
    }

    anchor_day = None
    anchor_weekday = None
    for w in weekends_raw:
        ds = safe_int(w.get("saturday"), None)
        if ds is not None and 1 <= ds <= num_days:
            anchor_day = ds
            anchor_weekday = 5
            break
        df = safe_int(w.get("friday"), None)
        if df is not None and 1 <= df <= num_days:
            anchor_day = df
            anchor_weekday = 4
            break
        du = safe_int(w.get("sunday"), None)
        if du is not None and 1 <= du <= num_days:
            anchor_day = du
            anchor_weekday = 6
            break

    def weekday_of(day):
        if anchor_day is None:
            return None
        return (anchor_weekday + (day - anchor_day)) % 7

    def is_weekend_day(day):
        if day in all_weekend_days:
            return True
        wd = weekday_of(day)
        return wd is not None and wd in (5, 6)

    def day_type_matches(day, day_type):
        if day_type is None:
            return True
        dt = str(day_type).lower().strip()
        if dt in ("", "any", "all", "*"):
            return True

        if (
            "non-weekend" in dt
            or "nonweekend" in dt
            or "weekday" in dt
            or "week_day" in dt
            or "workday" in dt
            or "work_day" in dt
        ):
            return not is_weekend_day(day)

        if "weekend" in dt:
            return is_weekend_day(day)

        if "sat" in dt:
            if day in saturday_days:
                return True
            wd = weekday_of(day)
            return wd == 5

        if "sun" in dt:
            if day in sunday_days:
                return True
            wd = weekday_of(day)
            return wd == 6

        if "fri" in dt:
            if day in friday_days:
                return True
            wd = weekday_of(day)
            return wd == 4

        for name, idx in weekday_names.items():
            if name in dt:
                wd = weekday_of(day)
                if wd is not None:
                    return wd == idx
                return True

        # Unknown day types are treated as matching to avoid making the model
        # infeasible because of unsupported nomenclature.
        return True

    def normalize_shift_name(val):
        if val is None:
            return rest_token
        sval = str(val)
        if sval == rest_token:
            return rest_token
        low = sval.lower()
        if low in ("none", "off", "rest", "day_off", "holiday"):
            return rest_token
        if sval in shift_set:
            return sval
        if low in shift_lower_map:
            return shift_lower_map[low]
        return sval

    # ------------------------------------------------------------------
    # Constraints / penalties
    # ------------------------------------------------------------------
    constraints = inst.get("constraints", {}) or {}
    hard = constraints.get("hard_constraints", {}) or {}
    soft = constraints.get("soft_constraints", {}) or {}

    hard_skill_requirement = bool(hard.get("skill_requirement", True))
    demand_equality = bool(hard.get("demand_equality", False))

    hard_forbidden_raw = hard.get("forbidden_patterns", []) or []
    if isinstance(hard_forbidden_raw, dict):
        hard_forbidden_raw = [hard_forbidden_raw]
    if not isinstance(hard_forbidden_raw, list):
        hard_forbidden_raw = []

    soft_forbidden_raw = soft.get("forbidden_patterns", []) or []
    if isinstance(soft_forbidden_raw, dict):
        soft_forbidden_raw = [soft_forbidden_raw]
    if not isinstance(soft_forbidden_raw, list):
        soft_forbidden_raw = []

    unsatisfied_demand_penalty = penalty_float(inst.get("unsatisfied_demand_penalty", 0), 0.0)
    skill_mismatch_penalty = penalty_float(soft.get("skill_requirement_penalty", 0), 0.0)

    prefs_setting = soft.get("preferences_penalty", True)
    preferences_enabled = True if prefs_setting is None else bool(prefs_setting)

    BIG = 1.0e9

    # ------------------------------------------------------------------
    # Demand parsing
    # ------------------------------------------------------------------
    demand = {}
    demanded_skills = {}
    demand_entries = []

    for day_key, shift_map in demand_raw.items():
        d = safe_int(str(day_key).rsplit("_", 1)[-1], None)
        if d is None or d < 1 or d > num_days:
            continue
        if not isinstance(shift_map, dict):
            continue

        for shift_name, skill_map in shift_map.items():
            s = str(shift_name)
            if s not in shift_set:
                continue
            if not isinstance(skill_map, dict):
                continue

            for skill_name, req in skill_map.items():
                k = str(skill_name)
                if k not in skill_set:
                    continue
                val = safe_int(req, 0)
                if val is None or val <= 0:
                    continue

                if d not in demand:
                    demand[d] = {}
                if s not in demand[d]:
                    demand[d][s] = {}

                demand[d][s][k] = demand[d][s].get(k, 0) + val

                if d not in demanded_skills:
                    demanded_skills[d] = {}
                if s not in demanded_skills[d]:
                    demanded_skills[d][s] = set()
                demanded_skills[d][s].add(k)

    for d in sorted(demand.keys()):
        for s in sorted(demand[d].keys()):
            for k in sorted(demand[d][s].keys()):
                req = demand[d][s][k]
                if req > 0:
                    demand_entries.append((d, s, k, req))

    demand_key_set = set((d, s, k) for (d, s, k, _) in demand_entries)

    # ------------------------------------------------------------------
    # Preferences normalization
    # ------------------------------------------------------------------
    preferences_by_nurse = {n: [] for n in nurse_ids}

    if preferences_enabled:
        for n in nurse_ids:
            for pref in raw_preferences.get(n, []) or []:
                d = safe_int(pref.get("day"), None)
                if d is None or d < 1 or d > num_days:
                    continue

                pen = penalty_float(pref.get("penalty", 0), 0.0)
                if pen <= 0:
                    continue

                ptype = str(pref.get("type", "")).lower().strip()
                if ptype not in ("day_off", "day_on", "shift_on", "shift_off"):
                    continue

                norm_shift = normalize_shift_name(pref.get("shift", rest_token))
                preferences_by_nurse[n].append((d, ptype, norm_shift, pen))

    # ------------------------------------------------------------------
    # Forbidden patterns parsing
    # ------------------------------------------------------------------
    def parse_pattern(pattern):
        parsed = []
        if not isinstance(pattern, list):
            return parsed

        for item in pattern:
            if isinstance(item, (list, tuple)):
                if len(item) >= 2:
                    parsed.append((item[0], item[1]))
                elif len(item) == 1:
                    parsed.append(("any", item[0]))
                else:
                    parsed.append(("any", "any"))
            elif isinstance(item, dict):
                parsed.append((item.get("day_type", "any"), item.get("shift", "any")))
            else:
                parsed.append(("any", item))

        return parsed

    def pattern_starts(pattern):
        L = len(pattern)
        if L == 0 or L > num_days:
            return []

        starts = []
        for st in range(0, num_days - L + 1):
            ok = True
            for i, (dt, _) in enumerate(pattern):
                if not day_type_matches(st + i + 1, dt):
                    ok = False
                    break
            if ok:
                starts.append(st)
        return starts

    hard_patterns = []
    for fp in hard_forbidden_raw:
        if not isinstance(fp, dict):
            continue
        pat = parse_pattern(fp.get("pattern", []))
        starts = pattern_starts(pat)
        if pat and starts:
            hard_patterns.append((pat, starts))

    soft_patterns = []
    for fp in soft_forbidden_raw:
        if not isinstance(fp, dict):
            continue
        pen = penalty_float(fp.get("penalty", 0), 0.0)
        if pen <= 0:
            continue
        pat = parse_pattern(fp.get("pattern", []))
        starts = pattern_starts(pat)
        if pat and starts:
            soft_patterns.append((pat, starts, pen))

    # ------------------------------------------------------------------
    # Soft-rule parameters
    # ------------------------------------------------------------------
    twd = soft.get("total_working_days", {}) or {}
    twd_min = max(0, safe_int(twd.get("min", 0), 0))
    twd_max = max(0, safe_int(twd.get("max", num_days), num_days))
    twd_pen_under = penalty_float(twd.get("penalty_under", 0), 0.0)
    twd_pen_over = penalty_float(twd.get("penalty_over", 0), 0.0)

    tww = soft.get("total_worked_weekends", {}) or {}
    tww_max = max(0, safe_int(tww.get("max", num_weekends), num_weekends))
    tww_penalty = penalty_float(tww.get("penalty", 0), 0.0)

    cwd = soft.get("consecutive_working_days", {}) or {}
    cwd_min = max(0, safe_int(cwd.get("min", 0), 0))
    cwd_max = max(0, safe_int(cwd.get("max", num_days), num_days))
    cwd_pen_under = penalty_float(cwd.get("penalty_under", 0), 0.0)
    cwd_pen_over = penalty_float(cwd.get("penalty_over", 0), 0.0)

    cdo = soft.get("consecutive_days_off", {}) or {}
    cdo_min = max(0, safe_int(cdo.get("min", 0), 0))
    cdo_max = max(0, safe_int(cdo.get("max", num_days), num_days))
    cdo_pen_under = penalty_float(cdo.get("penalty_under", 0), 0.0)
    cdo_pen_over = penalty_float(cdo.get("penalty_over", 0), 0.0)

    css = soft.get("consecutive_same_shift", {}) or {}
    css_max = max(0, safe_int(css.get("max", num_days), num_days))
    css_penalty = penalty_float(css.get("penalty", 0), 0.0)

    cww = soft.get("consecutive_worked_weekends", {}) or {}
    cww_max = max(0, safe_int(cww.get("max", num_weekends), num_weekends))
    cww_penalty = penalty_float(cww.get("penalty", 0), 0.0)

    iw = soft.get("identical_weekend", {}) or {}
    iw_enabled = bool(iw.get("enabled", False))
    iw_penalty = penalty_float(iw.get("penalty", 0), 0.0)

    # ------------------------------------------------------------------
    # Objective evaluator for complete solutions
    # ------------------------------------------------------------------
    def shift_token_matches_actual(token, actual_shift):
        if token is None:
            return None

        tok = str(token)
        if tok == rest_token:
            return actual_shift == rest_token

        low = tok.lower()
        if low in ("none", "off", "rest", "day_off", "holiday"):
            return actual_shift == rest_token

        if tok in shift_set:
            return actual_shift == tok
        if low in shift_lower_map:
            return actual_shift == shift_lower_map[low]

        if low in ("any", "all", "*", "work", "working"):
            return actual_shift != rest_token

        return None

    def compute_actual_objective(schedule, skill_assignments):
        cost = 0.0

        # Staffing counts.
        staffing = {}
        for n in nurse_ids:
            sched_n = schedule.get(n, {})
            skill_n = skill_assignments.get(n, {})
            for d in days:
                sh = sched_n.get(f"day_{d}", rest_token)
                if sh is None or sh == rest_token:
                    continue
                sk = skill_n.get(f"day_{d}", rest_token)
                if sk is None or sk == rest_token:
                    continue
                key = (d, str(sh), str(sk))
                staffing[key] = staffing.get(key, 0) + 1

        # Demand cost.
        if demand_equality:
            for d, s, k, req in demand_entries:
                staff = staffing.get((d, s, k), 0)
                if staff != req:
                    cost += BIG * abs(staff - req)

            # Under equality, staffing a skill with no positive demand is treated
            # as a hard equality violation for reporting purposes.
            for key, cnt in staffing.items():
                if key not in demand_key_set:
                    cost += BIG * cnt
        else:
            if unsatisfied_demand_penalty > 0:
                for d, s, k, req in demand_entries:
                    staff = staffing.get((d, s, k), 0)
                    if staff < req:
                        cost += unsatisfied_demand_penalty * (req - staff)

        # Sequences for roster rules and patterns.
        shift_sequences = {}
        work_sequences = {}

        for n in nurse_ids:
            sched_n = schedule.get(n, {})
            skill_n = skill_assignments.get(n, {})
            seq = []

            for d in days:
                sh = sched_n.get(f"day_{d}", rest_token)
                sh = rest_token if sh is None else str(sh)
                seq.append(sh)

                sk = skill_n.get(f"day_{d}", rest_token)
                sk = rest_token if sk is None else str(sk)

                # Skill mismatch / invalid skill cost.
                if sh != rest_token:
                    if sk == rest_token:
                        if hard_skill_requirement:
                            cost += BIG
                        else:
                            cost += skill_mismatch_penalty
                    elif hard_skill_requirement:
                        if sk not in nurse_skill_sets[n]:
                            cost += BIG
                    else:
                        if sk not in nurse_skill_sets[n]:
                            cost += skill_mismatch_penalty

            shift_sequences[n] = seq
            work_sequences[n] = [sh != rest_token for sh in seq]

        # Preferences.
        if preferences_enabled:
            for n in nurse_ids:
                seq = shift_sequences[n]
                for d, ptype, norm_shift, pen in preferences_by_nurse[n]:
                    actual = seq[d - 1]

                    if ptype == "day_off":
                        if actual != rest_token:
                            cost += pen
                    elif ptype == "day_on":
                        if actual == rest_token:
                            cost += pen
                    elif ptype == "shift_on":
                        if norm_shift in all_shifts_set:
                            if actual != norm_shift:
                                cost += pen
                        else:
                            cost += pen
                    elif ptype == "shift_off":
                        if norm_shift in all_shifts_set:
                            if actual == norm_shift:
                                cost += pen

        # Total working days.
        if twd_pen_under > 0 or twd_pen_over > 0:
            for n in nurse_ids:
                work_count = sum(work_sequences[n])
                if twd_pen_under > 0 and work_count < twd_min:
                    cost += twd_pen_under * (twd_min - work_count)
                if twd_pen_over > 0 and work_count > twd_max:
                    cost += twd_pen_over * (work_count - twd_max)

        # Weekend work.
        weekend_worked_by_nurse = {}
        for n in nurse_ids:
            work_seq = work_sequences[n]
            weekend_worked = []
            for wdays in weekend_days_list:
                weekend_worked.append(any(work_seq[d - 1] for d in wdays))
            weekend_worked_by_nurse[n] = weekend_worked

            if tww_penalty > 0:
                cnt = sum(weekend_worked)
                if cnt > tww_max:
                    cost += tww_penalty * (cnt - tww_max)

            if cww_penalty > 0 and num_weekends > 0:
                streak = 0
                for worked in weekend_worked:
                    if worked:
                        streak += 1
                    else:
                        if streak > cww_max:
                            cost += cww_penalty * (streak - cww_max)
                        streak = 0
                if streak > cww_max:
                    cost += cww_penalty * (streak - cww_max)

        # Consecutive working / rest streaks.
        if (
            cwd_pen_under > 0
            or cwd_pen_over > 0
            or cdo_pen_under > 0
            or cdo_pen_over > 0
        ):
            for n in nurse_ids:
                arr = work_sequences[n]
                i = 0
                L = len(arr)
                while i < L:
                    j = i
                    while j < L and arr[j] == arr[i]:
                        j += 1
                    length = j - i

                    if arr[i]:  # working streak
                        if cwd_pen_over > 0 and length > cwd_max:
                            cost += cwd_pen_over * (length - cwd_max)
                        if cwd_pen_under > 0 and length < cwd_min:
                            # Boundary-touching streaks are not penalized below min.
                            if i > 0 and j < L:
                                cost += cwd_pen_under
                    else:  # rest streak
                        if cdo_pen_over > 0 and length > cdo_max:
                            cost += cdo_pen_over * (length - cdo_max)
                        if cdo_pen_under > 0 and length < cdo_min:
                            if i > 0 and j < L:
                                cost += cdo_pen_under

                    i = j

        # Consecutive same shift.
        if css_penalty > 0 and work_shifts:
            for n in nurse_ids:
                seq = shift_sequences[n]
                i = 0
                L = len(seq)
                while i < L:
                    if seq[i] == rest_token:
                        i += 1
                        continue
                    j = i + 1
                    while j < L and seq[j] == seq[i]:
                        j += 1
                    length = j - i
                    if length > css_max:
                        cost += css_penalty * (length - css_max)
                    i = j

        # Identical weekend.
        if iw_enabled and iw_penalty > 0:
            for n in nurse_ids:
                seq = shift_sequences[n]
                for wdays in weekend_days_list:
                    if len(wdays) < 2:
                        continue
                    vals = [seq[d - 1] for d in wdays]
                    if len(set(vals)) > 1:
                        cost += iw_penalty

        # Forbidden patterns.
        if hard_patterns or soft_patterns:
            for n in nurse_ids:
                seq = shift_sequences[n]

                for pat, starts in hard_patterns:
                    L = len(pat)
                    for st in starts:
                        occ = True
                        for i, (_, sh_token) in enumerate(pat):
                            match = shift_token_matches_actual(sh_token, seq[st + i])
                            if match is not True:
                                occ = False
                                break
                        if occ:
                            cost += BIG

                for pat, starts, pen in soft_patterns:
                    L = len(pat)
                    for st in starts:
                        occ = True
                        for i, (_, sh_token) in enumerate(pat):
                            match = shift_token_matches_actual(sh_token, seq[st + i])
                            if match is not True:
                                occ = False
                                break
                        if occ:
                            cost += pen

        if cost < 0 and cost > -1e-6:
            cost = 0.0

        return float(cost)

    # ------------------------------------------------------------------
    # Fallback solution
    # ------------------------------------------------------------------
    def build_fallback_solution():
        schedule = {}
        skill_assignments = {}

        for n in nurse_ids:
            schedule[n] = {f"day_{d}": rest_token for d in days}
            skill_assignments[n] = {f"day_{d}": rest_token for d in days}

        obj = compute_actual_objective(schedule, skill_assignments)
        return {
            "objective_value": float(obj),
            "schedule": schedule,
            "skill_assignments": skill_assignments,
        }

    def time_left():
        return max(0.0, float(args.time_limit) - (time_module.time() - start_time))

    if time_left() <= 0.5:
        solution = build_fallback_solution()
        if logger:
            try:
                logger.log_solution(solution["objective_value"], solution)
            except Exception:
                pass
        with open(args.solution_path, "w") as f:
            json.dump(solution, f, indent=2)
        return

    # ------------------------------------------------------------------
    # Build compact skill-assignment options
    # ------------------------------------------------------------------
    dummy_token = -1

    option_keys = []
    options = {}
    demand_var_keys = defaultdict(list)
    dummy_actual = {}
    var_cost = {}

    for n in nurse_ids:
        owned = nurse_skill_sets[n]

        for d in days:
            dem_by_shift = demanded_skills.get(d, {})

            for s in work_shifts:
                dem = dem_by_shift.get(s, set())
                keys_for_cell = []

                if demand_equality:
                    if hard_skill_requirement:
                        dem_choices = owned.intersection(dem)
                    else:
                        dem_choices = dem

                    for k in dem_choices:
                        key = (n, d, s, k)
                        option_keys.append(key)
                        keys_for_cell.append(key)
                        demand_var_keys[(d, s, k)].append(key)

                        if (not hard_skill_requirement) and skill_mismatch_penalty > 0 and k not in owned:
                            var_cost[key] = skill_mismatch_penalty
                else:
                    if hard_skill_requirement:
                        dem_choices = owned.intersection(dem)
                    else:
                        dem_choices = dem

                    for k in dem_choices:
                        key = (n, d, s, k)
                        option_keys.append(key)
                        keys_for_cell.append(key)
                        demand_var_keys[(d, s, k)].append(key)

                        if (not hard_skill_requirement) and skill_mismatch_penalty > 0 and k not in owned:
                            var_cost[key] = skill_mismatch_penalty

                    non_dem = skill_set.difference(dem)
                    if non_dem:
                        owned_non = owned.intersection(non_dem)

                        if hard_skill_requirement:
                            if owned_non:
                                key = (n, d, s, dummy_token)
                                option_keys.append(key)
                                keys_for_cell.append(key)
                                dummy_actual[key] = sorted(owned_non)[0]
                        else:
                            key = (n, d, s, dummy_token)
                            option_keys.append(key)
                            keys_for_cell.append(key)

                            if owned_non:
                                dummy_actual[key] = sorted(owned_non)[0]
                            else:
                                dummy_actual[key] = sorted(non_dem)[0]
                                if skill_mismatch_penalty > 0:
                                    var_cost[key] = skill_mismatch_penalty

                if keys_for_cell:
                    options[(n, d, s)] = keys_for_cell

    def choose_emergency_skill(n, d, s):
        owned = nurse_skill_sets[n]
        dem = demanded_skills.get(d, {}).get(s, set())

        if demand_equality:
            if hard_skill_requirement:
                choices = owned.intersection(dem)
            else:
                choices = dem

            choices = sorted(choices)
            return choices[0] if choices else None

        if hard_skill_requirement:
            if not owned:
                return None
            inter = owned.intersection(dem)
            if inter:
                return sorted(inter)[0]
            return sorted(owned)[0]

        inter = owned.intersection(dem)
        if inter:
            return sorted(inter)[0]
        if dem:
            return sorted(dem)[0]
        if owned:
            return sorted(owned)[0]
        if skills:
            return skills[0]
        return None

    # ------------------------------------------------------------------
    # Gurobi model
    # ------------------------------------------------------------------
    solution = None

    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception:
        solution = None

    if solution is None and time_left() > 1.0:
        try:
            model = gp.Model("nurse_scheduling")
            model.setParam("Seed", 0)
            model.setParam("MIPGap", 1e-4)
            model.setParam("NumericFocus", 0)
            model.setParam("Threads", 1)
            model.setParam("OutputFlag", 0)
            model.setParam("MIPFocus", 1)

            if nurse_ids and days and all_shifts:
                x = model.addVars(
                    nurse_ids,
                    days,
                    all_shifts,
                    lb=0.0,
                    ub=1.0,
                    vtype=GRB.CONTINUOUS,
                )
            else:
                x = {}

            # Disable work shifts that have no feasible skill assignment.
            if nurse_ids and days and work_shifts:
                for n in nurse_ids:
                    for d in days:
                        for s in work_shifts:
                            if (n, d, s) not in options:
                                x[n, d, s].UB = 0.0

            # Exactly one shift per nurse-day.
            if nurse_ids and days:
                model.addConstrs(
                    (
                        gp.quicksum(x[n, d, s] for s in all_shifts) == 1
                        for n in nurse_ids
                        for d in days
                    )
                )

            if option_keys:
                y = model.addVars(option_keys, vtype=GRB.BINARY)
            else:
                y = {}

            # Link shift assignment to skill assignment.
            if options:
                model.addConstrs(
                    (
                        gp.quicksum(y[key] for key in keys) == x[n, d, s]
                        for (n, d, s), keys in options.items()
                    )
                )

            obj = gp.LinExpr()

            # Skill mismatch costs.
            if var_cost:
                for key, pen in var_cost.items():
                    if pen > 0:
                        obj += pen * y[key]

            # Demand constraints.
            if demand_entries:
                if demand_equality:
                    for d, s, k, req in demand_entries:
                        staff = gp.quicksum(y[key] for key in demand_var_keys.get((d, s, k), []))
                        under = model.addVar(lb=0.0)
                        over = model.addVar(lb=0.0)
                        model.addConstr(staff + under >= req)
                        model.addConstr(staff - over <= req)
                        obj += BIG * (under + over)
                else:
                    if unsatisfied_demand_penalty > 0:
                        for d, s, k, req in demand_entries:
                            staff = gp.quicksum(y[key] for key in demand_var_keys.get((d, s, k), []))
                            slack = model.addVar(lb=0.0)
                            model.addConstr(staff + slack >= req)
                            obj += unsatisfied_demand_penalty * slack

            # Total working days.
            if nurse_ids and days:
                if twd_pen_under > 0 and twd_min > 0:
                    for n in nurse_ids:
                        work_days = gp.quicksum(1 - x[n, d, rest_token] for d in days)
                        under = model.addVar(lb=0.0)
                        model.addConstr(under >= twd_min - work_days)
                        obj += twd_pen_under * under

                if twd_pen_over > 0 and twd_max < num_days:
                    for n in nurse_ids:
                        work_days = gp.quicksum(1 - x[n, d, rest_token] for d in days)
                        over = model.addVar(lb=0.0)
                        model.addConstr(over >= work_days - twd_max)
                        obj += twd_pen_over * over

            # Weekend-work indicators.
            tww_active = num_weekends > 0 and tww_penalty > 0 and tww_max < num_weekends
            cww_active = num_weekends > 0 and cww_penalty > 0 and cww_max < num_weekends

            if nurse_ids and (tww_active or cww_active):
                ww = model.addVars(
                    nurse_ids,
                    range(num_weekends),
                    lb=0.0,
                    ub=1.0,
                    vtype=GRB.CONTINUOUS,
                )

                for n in nurse_ids:
                    for wi, wdays in enumerate(weekend_days_list):
                        for d in wdays:
                            model.addConstr(ww[n, wi] >= 1 - x[n, d, rest_token])

                if tww_active:
                    for n in nurse_ids:
                        over = model.addVar(lb=0.0)
                        model.addConstr(
                            over >= gp.quicksum(ww[n, wi] for wi in range(num_weekends)) - tww_max
                        )
                        obj += tww_penalty * over

                if cww_active:
                    window_len = cww_max + 1
                    if window_len <= num_weekends:
                        for n in nurse_ids:
                            for start in range(0, num_weekends - window_len + 1):
                                over = model.addVar(lb=0.0)
                                model.addConstr(
                                    over
                                    >= gp.quicksum(
                                        ww[n, wi] for wi in range(start, start + window_len)
                                    )
                                    - cww_max
                                )
                                obj += cww_penalty * over

            # Consecutive working days.
            if nurse_ids and days:
                if cwd_pen_over > 0 and cwd_max < num_days:
                    window_len = cwd_max + 1
                    if window_len <= num_days:
                        for n in nurse_ids:
                            for start in range(1, num_days - window_len + 2):
                                over = model.addVar(lb=0.0)
                                window_sum = gp.quicksum(
                                    1 - x[n, d, rest_token]
                                    for d in range(start, start + window_len)
                                )
                                model.addConstr(over >= window_sum - cwd_max)
                                obj += cwd_pen_over * over

                if cwd_pen_under > 0 and cwd_min > 1:
                    max_short_len = min(cwd_min - 1, num_days - 2)
                    for length in range(1, max_short_len + 1):
                        for n in nurse_ids:
                            for start in range(2, num_days - length + 1):
                                expr = (
                                    x[n, start - 1, rest_token]
                                    + gp.quicksum(
                                        1 - x[n, start + i, rest_token]
                                        for i in range(length)
                                    )
                                    + x[n, start + length, rest_token]
                                )
                                viol = model.addVar(lb=0.0, ub=1.0)
                                model.addConstr(viol >= expr - length - 1)
                                obj += cwd_pen_under * viol

            # Consecutive days off.
            if nurse_ids and days:
                if cdo_pen_over > 0 and cdo_max < num_days:
                    window_len = cdo_max + 1
                    if window_len <= num_days:
                        for n in nurse_ids:
                            for start in range(1, num_days - window_len + 2):
                                over = model.addVar(lb=0.0)
                                window_sum = gp.quicksum(
                                    x[n, d, rest_token]
                                    for d in range(start, start + window_len)
                                )
                                model.addConstr(over >= window_sum - cdo_max)
                                obj += cdo_pen_over * over

                if cdo_pen_under > 0 and cdo_min > 1:
                    max_short_len = min(cdo_min - 1, num_days - 2)
                    for length in range(1, max_short_len + 1):
                        for n in nurse_ids:
                            for start in range(2, num_days - length + 1):
                                expr = (
                                    (1 - x[n, start - 1, rest_token])
                                    + gp.quicksum(
                                        x[n, start + i, rest_token] for i in range(length)
                                    )
                                    + (1 - x[n, start + length, rest_token])
                                )
                                viol = model.addVar(lb=0.0, ub=1.0)
                                model.addConstr(viol >= expr - length - 1)
                                obj += cdo_pen_under * viol

            # Consecutive same shift.
            if nurse_ids and days and work_shifts and css_penalty > 0 and css_max < num_days:
                window_len = css_max + 1
                if window_len <= num_days:
                    for n in nurse_ids:
                        for s in work_shifts:
                            for start in range(1, num_days - window_len + 2):
                                over = model.addVar(lb=0.0)
                                window_sum = gp.quicksum(
                                    x[n, d, s] for d in range(start, start + window_len)
                                )
                                model.addConstr(over >= window_sum - css_max)
                                obj += css_penalty * over

            # Identical weekend.
            if iw_enabled and iw_penalty > 0 and nurse_ids and num_weekends > 0:
                for n in nurse_ids:
                    for wi, wdays in enumerate(weekend_days_list):
                        if len(wdays) < 2:
                            continue
                        ref_day = wdays[0]
                        viol = model.addVar(lb=0.0, ub=1.0)

                        for other_day in wdays[1:]:
                            for s in all_shifts:
                                model.addConstr(x[n, ref_day, s] - x[n, other_day, s] <= viol)
                                model.addConstr(x[n, other_day, s] - x[n, ref_day, s] <= viol)

                        obj += iw_penalty * viol

            # Pattern helpers for model.
            def get_shift_match(n, d, token):
                if token is None:
                    return None

                tok = str(token)
                if tok == rest_token:
                    return x[n, d, rest_token]

                low = tok.lower()
                if low in ("none", "off", "rest", "day_off", "holiday"):
                    return x[n, d, rest_token]

                if tok in shift_set:
                    return x[n, d, tok]
                if low in shift_lower_map:
                    return x[n, d, shift_lower_map[low]]

                if low in ("any", "all", "*", "work", "working"):
                    return 1 - x[n, d, rest_token]

                return None

            # Hard forbidden patterns, enforced with a very large penalty.
            if nurse_ids:
                for pat, starts in hard_patterns:
                    L = len(pat)
                    for n in nurse_ids:
                        for st in starts:
                            exprs = []
                            const_sum = 0
                            valid = True

                            for i, (_, sh_token) in enumerate(pat):
                                match = get_shift_match(n, st + i + 1, sh_token)
                                if match is None:
                                    valid = False
                                    break
                                if isinstance(match, (int, float)):
                                    const_sum += int(match)
                                else:
                                    exprs.append(match)

                            if not valid:
                                continue

                            viol = model.addVar(lb=0.0, ub=1.0)
                            model.addConstr(gp.quicksum(exprs) + const_sum <= L - 1 + viol)
                            obj += BIG * viol

                # Soft forbidden patterns.
                for pat, starts, pen in soft_patterns:
                    L = len(pat)
                    for n in nurse_ids:
                        for st in starts:
                            exprs = []
                            const_sum = 0
                            valid = True

                            for i, (_, sh_token) in enumerate(pat):
                                match = get_shift_match(n, st + i + 1, sh_token)
                                if match is None:
                                    valid = False
                                    break
                                if isinstance(match, (int, float)):
                                    const_sum += int(match)
                                else:
                                    exprs.append(match)

                            if not valid:
                                continue

                            occ = model.addVar(lb=0.0, ub=1.0)
                            model.addConstr(gp.quicksum(exprs) + const_sum >= L - 1 + occ)
                            obj += pen * occ

            # Preferences.
            if preferences_enabled and nurse_ids:
                for n in nurse_ids:
                    for d, ptype, norm_shift, pen in preferences_by_nurse[n]:
                        if ptype == "day_off":
                            viol = model.addVar(lb=0.0, ub=1.0)
                            model.addConstr(viol >= 1 - x[n, d, rest_token])
                            obj += pen * viol

                        elif ptype == "day_on":
                            viol = model.addVar(lb=0.0, ub=1.0)
                            model.addConstr(viol >= x[n, d, rest_token])
                            obj += pen * viol

                        elif ptype == "shift_on":
                            if norm_shift in all_shifts_set:
                                viol = model.addVar(lb=0.0, ub=1.0)
                                model.addConstr(viol >= 1 - x[n, d, norm_shift])
                                obj += pen * viol
                            else:
                                obj += pen

                        elif ptype == "shift_off":
                            if norm_shift in all_shifts_set:
                                viol = model.addVar(lb=0.0, ub=1.0)
                                model.addConstr(viol >= x[n, d, norm_shift])
                                obj += pen * viol

            model.setObjective(obj, GRB.MINIMIZE)
            model.update()

            # ----------------------------------------------------------
            # Solution extraction
            # ----------------------------------------------------------
            def extract_solution(get_x, get_y):
                schedule = {}
                skill_assignments = {}

                for n in nurse_ids:
                    schedule[n] = {}
                    skill_assignments[n] = {}

                    for d in days:
                        best_shift = rest_token
                        try:
                            best_val = float(get_x(n, d, rest_token))
                        except Exception:
                            best_val = -1.0

                        for s in work_shifts:
                            try:
                                val = float(get_x(n, d, s))
                            except Exception:
                                val = 0.0
                            if val > best_val + 1e-6:
                                best_val = val
                                best_shift = s

                        if best_shift != rest_token and best_shift in shift_set:
                            keys_for_cell = options.get((n, d, best_shift), [])
                            best_key = None
                            best_y_val = -1.0

                            for key in keys_for_cell:
                                try:
                                    val = float(get_y(key))
                                except Exception:
                                    val = 0.0
                                if val > best_y_val:
                                    best_y_val = val
                                    best_key = key

                            if best_key is None:
                                skill = choose_emergency_skill(n, d, best_shift)
                                if skill is None:
                                    best_shift = rest_token
                                    skill = rest_token
                            else:
                                token = best_key[3]
                                if token == dummy_token:
                                    skill = dummy_actual.get(best_key)
                                    if skill is None:
                                        skill = choose_emergency_skill(n, d, best_shift)
                                    if skill is None:
                                        best_shift = rest_token
                                        skill = rest_token
                                else:
                                    skill = token

                            schedule[n][f"day_{d}"] = str(best_shift)
                            skill_assignments[n][f"day_{d}"] = str(skill)
                        else:
                            schedule[n][f"day_{d}"] = rest_token
                            skill_assignments[n][f"day_{d}"] = rest_token

                return schedule, skill_assignments

            # ----------------------------------------------------------
            # Callback logging
            # ----------------------------------------------------------
            best_cb_obj = [float("inf")]
            small_enough_for_callback_solution = (
                len(nurse_ids) * num_days <= 300 and len(option_keys) <= 5000
            )

            def callback(cb_model, where):
                if where != GRB.Callback.MIPSOL:
                    return

                try:
                    cb_obj = float(cb_model.cbGet(GRB.Callback.MIPSOL_OBJ))
                except Exception:
                    return

                if cb_obj >= best_cb_obj[0] - 1e-6:
                    return

                best_cb_obj[0] = cb_obj
                if not logger:
                    return

                if not small_enough_for_callback_solution:
                    try:
                        logger.log(cb_obj)
                    except Exception:
                        pass
                    return

                try:
                    def gx(n, d, s):
                        return float(cb_model.cbGetSolution(x[n, d, s]))

                    def gy(key):
                        if key in y:
                            return float(cb_model.cbGetSolution(y[key]))
                        return 0.0

                    sched, skills_out = extract_solution(gx, gy)
                    actual_obj = compute_actual_objective(sched, skills_out)
                    sol_dict = {
                        "objective_value": actual_obj,
                        "schedule": sched,
                        "skill_assignments": skills_out,
                    }
                    logger.log_solution(actual_obj, sol_dict)
                except Exception:
                    try:
                        logger.log(cb_obj)
                    except Exception:
                        pass

            # ----------------------------------------------------------
            # Optimize
            # ----------------------------------------------------------
            remaining = int(time_left()) - 1
            if remaining > 0:
                model.setParam("TimeLimit", remaining)
                model.optimize(callback)

            try:
                has_solution = model.SolCount > 0
            except Exception:
                has_solution = False

            if has_solution:
                def gx_final(n, d, s):
                    return float(x[n, d, s].X)

                def gy_final(key):
                    if key in y:
                        return float(y[key].X)
                    return 0.0

                sched, skills_out = extract_solution(gx_final, gy_final)
                actual_obj = compute_actual_objective(sched, skills_out)

                solution = {
                    "objective_value": float(actual_obj),
                    "schedule": sched,
                    "skill_assignments": skills_out,
                }

        except Exception:
            solution = None

    if solution is None:
        solution = build_fallback_solution()

    if logger:
        try:
            logger.log_solution(solution["objective_value"], solution)
        except Exception:
            try:
                logger.log(solution["objective_value"])
            except Exception:
                pass

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()
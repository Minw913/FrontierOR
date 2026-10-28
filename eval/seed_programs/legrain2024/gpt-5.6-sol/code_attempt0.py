import argparse
import json
import os
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


OFF = "None"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", default=None)
    return parser.parse_args()


def day_number(key):
    if isinstance(key, int):
        return key
    text = str(key)
    if text.startswith("day_"):
        return int(text[4:])
    return int(text)


def determine_horizon(instance):
    days = set()

    for key in instance.get("demand", {}):
        days.add(day_number(key))

    for nurse in instance.get("nurses", []):
        for pref in nurse.get("preferences", []):
            days.add(int(pref["day"]))

    for weekend in instance.get("weekends", []):
        if "saturday" in weekend:
            days.add(int(weekend["saturday"]))
        if "sunday" in weekend:
            days.add(int(weekend["sunday"]))

    return max(days) if days else 0


def calendar_day_names(instance, horizon):
    """Derive weekday names using any supplied Saturday as an anchor."""
    anchor_saturday = None
    weekends = instance.get("weekends", [])
    if weekends:
        anchor_saturday = int(weekends[0]["saturday"])

    names = {}
    weekday_names = ["monday", "tuesday", "wednesday", "thursday",
                     "friday", "saturday", "sunday"]

    if anchor_saturday is not None:
        # Monday=0, Saturday=5.
        for d in range(1, horizon + 1):
            idx = (5 + d - anchor_saturday) % 7
            names[d] = weekday_names[idx]
    else:
        # Fallback convention: day 1 is Monday.
        for d in range(1, horizon + 1):
            names[d] = weekday_names[(d - 1) % 7]

    for weekend in weekends:
        sat = int(weekend["saturday"])
        sun = int(weekend["sunday"])
        if 1 <= sat <= horizon:
            names[sat] = "saturday"
        if 1 <= sun <= horizon:
            names[sun] = "sunday"
        if 1 <= sat - 1 <= horizon:
            names[sat - 1] = "friday"

    return names


def day_type_matches(specification, actual_day_name):
    if specification is None:
        return True

    spec = str(specification).strip().lower()
    actual = actual_day_name.lower()

    if spec in {"*", "any", "all", "any_day", "any day", "day"}:
        return True
    if spec in {"weekday", "week_day", "working_day", "workday"}:
        return actual in {"monday", "tuesday", "wednesday", "thursday", "friday"}
    if spec in {"weekend", "weekend_day"}:
        return actual in {"saturday", "sunday"}
    if spec in {"mon", "monday"}:
        return actual == "monday"
    if spec in {"tue", "tues", "tuesday"}:
        return actual == "tuesday"
    if spec in {"wed", "wednesday"}:
        return actual == "wednesday"
    if spec in {"thu", "thur", "thurs", "thursday"}:
        return actual == "thursday"
    if spec in {"fri", "friday"}:
        return actual == "friday"
    if spec in {"sat", "saturday"}:
        return actual == "saturday"
    if spec in {"sun", "sunday"}:
        return actual == "sunday"

    return spec == actual


def normalized_shift_name(value):
    if value is None:
        return OFF
    text = str(value)
    if text.strip().lower() in {"none", "off", "rest", "day_off", "day off"}:
        return OFF
    return text


def main():
    args = parse_args()
    start_time = time.monotonic()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as f:
        instance = json.load(f)

    nurses_data = instance.get("nurses", [])
    nurse_ids = [str(n["id"]) for n in nurses_data]
    nurse_by_id = {str(n["id"]): n for n in nurses_data}

    work_shifts = list(instance.get("shifts", []))
    skills = list(instance.get("skills", []))
    all_shifts = work_shifts + [OFF]

    horizon = determine_horizon(instance)
    days = list(range(1, horizon + 1))
    day_names = calendar_day_names(instance, horizon)

    constraints = instance.get("constraints", {})
    hard = constraints.get("hard_constraints", {})
    soft = constraints.get("soft_constraints", {})

    hard_skill_requirement = bool(hard.get("skill_requirement", False))
    demand_equality = bool(hard.get("demand_equality", False))

    model = gp.Model("nurse_scheduling")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1

    objective = gp.LinExpr()

    # x[n,d,s] indicates the assigned shift, including OFF.
    x = {}
    for n in nurse_ids:
        for d in days:
            for s in all_shifts:
                x[n, d, s] = model.addVar(vtype=GRB.BINARY, name=f"x_{n}_{d}_{s}")

    # z[n,d] indicates a worked day.
    z = {}
    for n in nurse_ids:
        for d in days:
            z[n, d] = model.addVar(vtype=GRB.BINARY, name=f"work_{n}_{d}")

    # y[n,d,s,k] is the skill witness for a worked assignment.
    y = {}
    for n in nurse_ids:
        possessed = set(nurse_by_id[n].get("skills", []))
        for d in days:
            for s in work_shifts:
                for k in skills:
                    if hard_skill_requirement and k not in possessed:
                        continue
                    y[n, d, s, k] = model.addVar(
                        vtype=GRB.BINARY, name=f"y_{n}_{d}_{s}_{k}"
                    )

    model.update()

    # Exactly one shift per nurse-day and exactly one skill per worked shift.
    for n in nurse_ids:
        for d in days:
            model.addConstr(
                gp.quicksum(x[n, d, s] for s in all_shifts) == 1,
                name=f"one_shift_{n}_{d}",
            )
            model.addConstr(
                z[n, d] == gp.quicksum(x[n, d, s] for s in work_shifts),
                name=f"work_link_{n}_{d}",
            )

            for s in work_shifts:
                available_y = [
                    y[n, d, s, k]
                    for k in skills
                    if (n, d, s, k) in y
                ]
                if available_y:
                    model.addConstr(
                        gp.quicksum(available_y) == x[n, d, s],
                        name=f"skill_link_{n}_{d}_{s}",
                    )
                else:
                    model.addConstr(x[n, d, s] == 0)

    # Soft skill mismatch penalties.
    if not hard_skill_requirement:
        mismatch_penalty = float(soft.get("skill_requirement_penalty", 0))
        if mismatch_penalty:
            for n in nurse_ids:
                possessed = set(nurse_by_id[n].get("skills", []))
                for d in days:
                    for s in work_shifts:
                        for k in skills:
                            if k not in possessed and (n, d, s, k) in y:
                                objective += mismatch_penalty * y[n, d, s, k]

    # Demand constraints and unmet-demand slack.
    demand_data = instance.get("demand", {})
    unmet_penalty = float(instance.get("unsatisfied_demand_penalty", 0))
    demand_slack = {}

    for d in days:
        day_demand = demand_data.get(f"day_{d}", demand_data.get(str(d), {}))
        for s in work_shifts:
            shift_demand = day_demand.get(s, {})
            for k in skills:
                required = int(shift_demand.get(k, 0))
                staffing = gp.quicksum(
                    y[n, d, s, k]
                    for n in nurse_ids
                    if (n, d, s, k) in y
                )

                if demand_equality:
                    model.addConstr(
                        staffing == required,
                        name=f"demand_eq_{d}_{s}_{k}",
                    )
                else:
                    slack = model.addVar(
                        lb=0.0,
                        vtype=GRB.CONTINUOUS,
                        name=f"unmet_{d}_{s}_{k}",
                    )
                    demand_slack[d, s, k] = slack
                    model.addConstr(
                        staffing + slack >= required,
                        name=f"demand_{d}_{s}_{k}",
                    )
                    if unmet_penalty:
                        objective += unmet_penalty * slack

    # Total working days.
    total_work_cfg = soft.get("total_working_days")
    if total_work_cfg:
        min_work = int(total_work_cfg.get("min", 0))
        max_work = int(total_work_cfg.get("max", horizon))
        penalty_under = float(total_work_cfg.get("penalty_under", 0))
        penalty_over = float(total_work_cfg.get("penalty_over", 0))

        for n in nurse_ids:
            total_work = gp.quicksum(z[n, d] for d in days)

            if penalty_under:
                under = model.addVar(lb=0.0, name=f"work_under_{n}")
                model.addConstr(under >= min_work - total_work)
                objective += penalty_under * under

            if penalty_over:
                over = model.addVar(lb=0.0, name=f"work_over_{n}")
                model.addConstr(over >= total_work - max_work)
                objective += penalty_over * over

    weekend_definition = str(instance.get("weekend_definition", "")).lower()
    includes_friday = "fri" in weekend_definition

    weekend_day_lists = []
    for wi, weekend in enumerate(instance.get("weekends", [])):
        sat = int(weekend["saturday"])
        sun = int(weekend["sunday"])
        wd = []
        if includes_friday and 1 <= sat - 1 <= horizon:
            wd.append(sat - 1)
        if 1 <= sat <= horizon:
            wd.append(sat)
        if 1 <= sun <= horizon:
            wd.append(sun)
        weekend_day_lists.append((wi, wd))

    # Weekend work indicators.
    weekend_work = {}
    for n in nurse_ids:
        for wi, wd in weekend_day_lists:
            w = model.addVar(vtype=GRB.BINARY, name=f"weekend_work_{n}_{wi}")
            weekend_work[n, wi] = w
            if wd:
                for d in wd:
                    model.addConstr(w >= z[n, d])
                model.addConstr(w <= gp.quicksum(z[n, d] for d in wd))
            else:
                model.addConstr(w == 0)

    # Total worked weekends.
    total_weekend_cfg = soft.get("total_worked_weekends")
    if total_weekend_cfg:
        max_weekends = int(total_weekend_cfg.get("max", len(weekend_day_lists)))
        weekend_penalty = float(
            total_weekend_cfg.get("penalty", total_weekend_cfg.get("penalty_over", 0))
        )
        if weekend_penalty:
            for n in nurse_ids:
                over = model.addVar(lb=0.0, name=f"weekend_over_{n}")
                model.addConstr(
                    over >= gp.quicksum(
                        weekend_work[n, wi] for wi, _ in weekend_day_lists
                    ) - max_weekends
                )
                objective += weekend_penalty * over

    def add_binary_run_cost(binary_vars, prefix, cfg):
        """Add finite-window run-length costs for one binary sequence."""
        if not cfg or not binary_vars:
            return

        length = len(binary_vars)
        min_len = int(cfg.get("min", 0))
        max_len = int(cfg.get("max", length))
        penalty_under = float(cfg.get("penalty_under", 0))
        penalty_over = float(cfg.get("penalty_over", cfg.get("penalty", 0)))

        # A run of L contributes L-max through its max+1 all-one windows.
        if penalty_over and max_len >= 0 and max_len < length:
            window = max_len + 1
            for start in range(0, length - window + 1):
                violation = model.addVar(
                    vtype=GRB.BINARY, name=f"{prefix}_over_{start}"
                )
                model.addConstr(
                    violation >=
                    gp.quicksum(binary_vars[i] for i in range(start, start + window))
                    - max_len
                )
                objective.addTerms(penalty_over, violation)

        # Only fully enclosed short runs are penalized; boundary runs are exempt.
        if penalty_under and min_len > 1:
            for run_len in range(1, min_len):
                for start in range(1, length - run_len):
                    end = start + run_len - 1
                    if end + 1 >= length:
                        continue
                    violation = model.addVar(
                        vtype=GRB.BINARY,
                        name=f"{prefix}_under_{start}_{run_len}",
                    )
                    terms = (
                        (1 - binary_vars[start - 1])
                        + gp.quicksum(binary_vars[i] for i in range(start, end + 1))
                        + (1 - binary_vars[end + 1])
                    )
                    model.addConstr(violation >= terms - (run_len + 1))
                    objective.addTerms(penalty_under, violation)

    # Consecutive working-day and off-day runs.
    for n in nurse_ids:
        add_binary_run_cost(
            [z[n, d] for d in days],
            f"working_run_{n}",
            soft.get("consecutive_working_days"),
        )
        add_binary_run_cost(
            [x[n, d, OFF] for d in days],
            f"off_run_{n}",
            soft.get("consecutive_days_off"),
        )

    # Consecutive occurrences of the same work shift.
    same_shift_cfg = soft.get("consecutive_same_shift")
    if same_shift_cfg:
        same_max = int(same_shift_cfg.get("max", horizon))
        same_penalty = float(
            same_shift_cfg.get("penalty", same_shift_cfg.get("penalty_over", 0))
        )
        if same_penalty and same_max < horizon:
            window = same_max + 1
            for n in nurse_ids:
                for s in work_shifts:
                    for start in range(0, horizon - window + 1):
                        violation = model.addVar(
                            vtype=GRB.BINARY,
                            name=f"same_shift_{n}_{s}_{start}",
                        )
                        model.addConstr(
                            violation >=
                            gp.quicksum(
                                x[n, days[i], s]
                                for i in range(start, start + window)
                            ) - same_max
                        )
                        objective += same_penalty * violation

    # Consecutive worked weekends.
    consecutive_weekend_cfg = soft.get("consecutive_worked_weekends")
    if consecutive_weekend_cfg and weekend_day_lists:
        ww_max = int(
            consecutive_weekend_cfg.get("max", len(weekend_day_lists))
        )
        ww_penalty = float(
            consecutive_weekend_cfg.get(
                "penalty", consecutive_weekend_cfg.get("penalty_over", 0)
            )
        )
        weekend_indices = [wi for wi, _ in weekend_day_lists]
        if ww_penalty and ww_max < len(weekend_indices):
            window = ww_max + 1
            for n in nurse_ids:
                for start in range(0, len(weekend_indices) - window + 1):
                    violation = model.addVar(
                        vtype=GRB.BINARY,
                        name=f"consecutive_weekend_{n}_{start}",
                    )
                    model.addConstr(
                        violation >=
                        gp.quicksum(
                            weekend_work[n, weekend_indices[i]]
                            for i in range(start, start + window)
                        ) - ww_max
                    )
                    objective += ww_penalty * violation

    # Identical-weekend shift assignments.
    identical_cfg = soft.get("identical_weekend", {})
    if identical_cfg.get("enabled", False):
        identical_penalty = float(identical_cfg.get("penalty", 0))
        for n in nurse_ids:
            for wi, wd in weekend_day_lists:
                if len(wd) <= 1:
                    continue
                mismatch = model.addVar(
                    vtype=GRB.BINARY, name=f"weekend_mismatch_{n}_{wi}"
                )
                reference = wd[0]
                for d in wd[1:]:
                    for s in all_shifts:
                        model.addConstr(mismatch >= x[n, reference, s] - x[n, d, s])
                        model.addConstr(mismatch >= x[n, d, s] - x[n, reference, s])
                if identical_penalty:
                    objective += identical_penalty * mismatch

    def shift_match_expr(n, d, shift_spec):
        shift_name = normalized_shift_name(shift_spec)
        lower = str(shift_spec).strip().lower() if shift_spec is not None else "none"

        if lower in {"*", "any", "all", "any_shift", "any shift"}:
            return gp.LinExpr(1.0)
        if lower in {"work", "working", "any_work", "any work", "work_shift"}:
            return z[n, d]
        if shift_name in all_shifts:
            return x[n, d, shift_name]
        # Unknown shifts can never match.
        return gp.LinExpr(0.0)

    def add_forbidden_patterns(pattern_objects, hard_patterns):
        for pi, pattern_obj in enumerate(pattern_objects or []):
            pattern = pattern_obj.get("pattern", [])
            pattern_len = len(pattern)
            if pattern_len == 0 or pattern_len > horizon:
                continue

            penalty = float(
                pattern_obj.get(
                    "penalty",
                    pattern_obj.get("cost", pattern_obj.get("weight", 0)),
                )
            )

            for start in range(1, horizon - pattern_len + 2):
                calendar_matches = True
                for offset, pair in enumerate(pattern):
                    if not pair:
                        continue
                    day_spec = pair[0] if len(pair) >= 1 else None
                    if not day_type_matches(
                        day_spec, day_names[start + offset]
                    ):
                        calendar_matches = False
                        break

                if not calendar_matches:
                    continue

                for n in nurse_ids:
                    terms = []
                    for offset, pair in enumerate(pattern):
                        shift_spec = pair[1] if len(pair) >= 2 else OFF
                        terms.append(shift_match_expr(n, start + offset, shift_spec))

                    expression = gp.quicksum(terms)
                    if hard_patterns:
                        model.addConstr(
                            expression <= pattern_len - 1,
                            name=f"hard_pattern_{pi}_{n}_{start}",
                        )
                    elif penalty:
                        violation = model.addVar(
                            vtype=GRB.BINARY,
                            name=f"soft_pattern_{pi}_{n}_{start}",
                        )
                        model.addConstr(violation >= expression - pattern_len + 1)
                        objective += penalty * violation

    add_forbidden_patterns(hard.get("forbidden_patterns", []), True)
    add_forbidden_patterns(soft.get("forbidden_patterns", []), False)

    # Nurse preferences.
    pref_switch = soft.get("preferences_penalty", True)
    preferences_enabled = bool(pref_switch)

    if preferences_enabled:
        for n in nurse_ids:
            for pi, pref in enumerate(nurse_by_id[n].get("preferences", [])):
                d = int(pref["day"])
                if d < 1 or d > horizon:
                    continue

                penalty = float(pref.get("penalty", 0))
                if not penalty:
                    continue

                pref_type = str(pref.get("type", "")).lower()
                pref_shift = normalized_shift_name(pref.get("shift", OFF))

                if pref_type == "day_off":
                    unmet = 1 - x[n, d, OFF]
                elif pref_type == "day_on":
                    unmet = x[n, d, OFF]
                elif pref_type == "shift_on":
                    if pref_shift in all_shifts:
                        unmet = 1 - x[n, d, pref_shift]
                    else:
                        unmet = 1
                elif pref_type == "shift_off":
                    if pref_shift in all_shifts:
                        unmet = x[n, d, pref_shift]
                    else:
                        unmet = 0
                else:
                    continue

                objective += penalty * unmet

    model.setObjective(objective, GRB.MINIMIZE)

    # A useful warm start for non-equality instances: all nurses are off and
    # all demand is covered by slack.
    if not demand_equality:
        for n in nurse_ids:
            for d in days:
                for s in all_shifts:
                    x[n, d, s].Start = 1.0 if s == OFF else 0.0
                z[n, d].Start = 0.0
        for variable in y.values():
            variable.Start = 0.0
        for (d, s, k), slack in demand_slack.items():
            day_demand = demand_data.get(f"day_{d}", demand_data.get(str(d), {}))
            slack.Start = float(day_demand.get(s, {}).get(k, 0))

    model.update()

    x_keys = list(x.keys())
    x_vars = [x[key] for key in x_keys]
    y_keys = list(y.keys())
    y_vars = [y[key] for key in y_keys]

    def make_solution(x_values, y_values, objective_value):
        x_value_map = dict(zip(x_keys, x_values))
        y_value_map = dict(zip(y_keys, y_values))

        schedule = {n: {} for n in nurse_ids}
        skill_assignments = {n: {} for n in nurse_ids}

        for n in nurse_ids:
            for d in days:
                chosen_shift = max(
                    all_shifts, key=lambda s: x_value_map.get((n, d, s), 0.0)
                )
                schedule[n][f"day_{d}"] = chosen_shift

                if chosen_shift == OFF:
                    skill_assignments[n][f"day_{d}"] = OFF
                else:
                    chosen_skill = None
                    best_value = -1.0
                    for k in skills:
                        value = y_value_map.get((n, d, chosen_shift, k), -1.0)
                        if value > best_value:
                            best_value = value
                            chosen_skill = k
                    skill_assignments[n][f"day_{d}"] = (
                        chosen_skill if chosen_skill is not None else OFF
                    )

        rounded_objective = float(objective_value)
        if abs(rounded_objective - round(rounded_objective)) <= 1e-6:
            rounded_objective = float(round(rounded_objective))

        return {
            "objective_value": rounded_objective,
            "schedule": schedule,
            "skill_assignments": skill_assignments,
        }

    best_logged = [float("inf")]

    def callback(cb_model, where):
        if where != GRB.Callback.MIPSOL:
            return
        try:
            obj_value = cb_model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj_value + 1e-7 >= best_logged[0]:
                return
            best_logged[0] = obj_value

            if logger:
                incumbent_x = cb_model.cbGetSolution(x_vars)
                incumbent_y = cb_model.cbGetSolution(y_vars)
                solution = make_solution(incumbent_x, incumbent_y, obj_value)
                logger.log_solution(solution["objective_value"], solution)
        except Exception:
            # Logging must never interrupt the optimization.
            pass

    elapsed = time.monotonic() - start_time
    remaining = max(0.01, float(args.time_limit) - elapsed)
    model.Params.TimeLimit = remaining
    model.optimize(callback)

    if model.SolCount <= 0:
        raise RuntimeError("No feasible solution was found within the time limit.")

    final_x = [var.X for var in x_vars]
    final_y = [var.X for var in y_vars]
    final_solution = make_solution(final_x, final_y, model.ObjVal)

    if logger and final_solution["objective_value"] + 1e-7 < best_logged[0]:
        logger.log_solution(final_solution["objective_value"], final_solution)

    output_dir = os.path.dirname(os.path.abspath(args.solution_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(final_solution, f, indent=2)


if __name__ == "__main__":
    main()
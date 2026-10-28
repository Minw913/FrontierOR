import argparse
import json
import math
import os
import time
from collections import defaultdict

import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


EPS = 1e-7


def pairing_interval(pairing):
    start = 24.0 * pairing["start_day"] + float(pairing["departure_hour"])
    end = (
        24.0 * (pairing["start_day"] + pairing["duration_days"] - 1)
        + float(pairing["return_hour"])
    )
    if end <= start + EPS:
        end = start + float(pairing["duration_minutes"]) / 60.0
    return start, end


def activity_interval(activity):
    return 24.0 * activity["start_day"], 24.0 * (activity["end_day"] + 1)


def intervals_overlap(a_start, a_end, b_start, b_end):
    return a_start < b_end - EPS and b_start < a_end - EPS


def merge_intervals(intervals):
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        if start <= merged[-1][1] + EPS:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged]


def is_rest_like_activity(activity_type):
    t = str(activity_type).strip().lower()
    keywords = (
        "annual_leave",
        "annual leave",
        "vacation",
        "holiday",
        "day_off",
        "day off",
        "desiderata",
        "periodic_rest",
        "weekly_rest",
        "monthly_rest",
    )
    return any(k in t for k in keywords)


def is_reserve_activity(activity_type):
    t = str(activity_type).strip().lower()
    return "reserve" in t


def candidate_rest_starts(
    busy_intervals,
    earliest,
    latest,
    length,
    horizon_end,
    require_weekly_component_inside=False,
):
    if latest < earliest - EPS:
        return []

    search_start = earliest
    search_end = max(latest + length, horizon_end + length)
    clipped = []
    for a, b in busy_intervals:
        if b > search_start - EPS and a < search_end + EPS:
            clipped.append((max(a, search_start), min(b, search_end)))
    clipped = merge_intervals(clipped)

    gaps = []
    cursor = search_start
    for a, b in clipped:
        if a > cursor + EPS:
            gaps.append((cursor, a))
        cursor = max(cursor, b)
    if cursor < search_end + EPS:
        gaps.append((cursor, search_end))

    starts = set()
    for gap_start, gap_end in gaps:
        lo = max(earliest, gap_start)
        hi = min(latest, gap_end - length)
        if require_weekly_component_inside:
            hi = min(hi, horizon_end - 48.0)
        if hi < lo - EPS:
            continue

        starts.add(round(lo, 6))
        starts.add(round(hi, 6))

        # Hourly points plus exact gap boundaries provide a compact but robust
        # representation of the continuous rest-start domain.
        first_hour = int(math.ceil(lo - EPS))
        last_hour = int(math.floor(hi + EPS))
        for hour in range(first_hour, last_hour + 1):
            starts.add(float(hour))

        # Include quarter-hour boundary points near the endpoints.
        qlo = int(math.ceil(4.0 * lo - EPS))
        qhi = int(math.floor(4.0 * hi + EPS))
        for q in (qlo, qlo + 1, qhi - 1, qhi):
            value = q / 4.0
            if lo - EPS <= value <= hi + EPS:
                starts.add(round(value, 6))

    return sorted(starts)


class RosterValidator:
    def __init__(self, instance, pairings, pairing_intervals):
        self.instance = instance
        self.pairings = pairings
        self.pairing_intervals = pairing_intervals
        self.horizon_days = int(instance["planning_horizon_days"])
        self.horizon_end = 24.0 * self.horizon_days
        self.rules = instance.get("work_rules", {})
        self.cache = {}

    def validate(self, crew_index, selected_pairing_indices):
        key = (crew_index, tuple(sorted(selected_pairing_indices)))
        if key in self.cache:
            return self.cache[key]

        crew = self.instance["crew_members"][crew_index]
        selected = sorted(selected_pairing_indices)

        weekly_hours = float(self.rules.get("weekly_rest_period_hours", 48))
        additional_hours = float(
            self.rules.get("monthly_rest_additional_hours", 48)
        )
        max_spread_days = float(
            self.rules.get("max_spread_between_breaks_days", 8)
        )
        sunday_hour = float(
            self.rules.get("weekly_rest_latest_start_on_sunday_hour", 6)
        )

        activity_busy = []
        reserves = []
        inactive_days = set()

        for activity in crew.get("pre_assigned_activities", []):
            a, b = activity_interval(activity)
            activity_type = activity.get("type", "")
            for day in range(
                max(0, int(activity["start_day"])),
                min(self.horizon_days - 1, int(activity["end_day"])) + 1,
            ):
                inactive_days.add(day)

            if is_reserve_activity(activity_type):
                reserves.append((a, b))

            # Leave and designated days off can themselves provide uninterrupted
            # rest. Other pre-assigned activities interrupt a rest interval.
            if not is_rest_like_activity(activity_type):
                activity_busy.append((a, b))

        pairing_busy = [self.pairing_intervals[p] for p in selected]
        all_busy = merge_intervals(activity_busy + pairing_busy)

        reduction = 12.0 * (len(inactive_days) // 7)
        effective_additional = max(0.0, additional_hours - reduction)
        monthly_length = weekly_hours + effective_additional

        weeks = list(range(0, self.horizon_days, 7))
        candidates_by_week = []

        for week_start_day in weeks:
            week_start = 24.0 * week_start_day
            latest_start = week_start + 6.0 * 24.0 + sunday_hour
            latest_start = min(latest_start, self.horizon_end)

            normal_starts = candidate_rest_starts(
                all_busy,
                week_start,
                latest_start,
                weekly_hours,
                self.horizon_end,
                require_weekly_component_inside=False,
            )
            monthly_starts = candidate_rest_starts(
                all_busy,
                week_start,
                latest_start,
                monthly_length,
                self.horizon_end,
                require_weekly_component_inside=True,
            )

            candidates = []
            for start in normal_starts:
                candidates.append((start, start + weekly_hours, False))
            for start in monthly_starts:
                candidates.append((start, start + monthly_length, True))

            # Remove duplicate candidate records.
            candidates = sorted(set(candidates))
            if not candidates:
                self.cache[key] = (False, "rest")
                return self.cache[key]
            candidates_by_week.append(candidates)

        def spread_compatible(previous, current):
            prev_start, prev_end, _ = previous
            cur_start, _, _ = current
            if cur_start < prev_end - EPS:
                return False

            limit = 24.0 * max_spread_days
            if any(
                intervals_overlap(prev_end, cur_start, reserve_start, reserve_end)
                for reserve_start, reserve_end in reserves
            ):
                limit += 24.0

            return cur_start - prev_end <= limit + EPS

        # Dynamic programming over the weekly rest choices. State records
        # whether the required extended monthly rest has already been used.
        states = {(candidate, candidate[2]) for candidate in candidates_by_week[0]}

        for week_candidates in candidates_by_week[1:]:
            next_states = set()
            for current in week_candidates:
                for previous, monthly_used in states:
                    if spread_compatible(previous, current):
                        next_states.add(
                            (current, monthly_used or current[2])
                        )
            if not next_states:
                self.cache[key] = (False, "rest")
                return self.cache[key]
            states = next_states

        if not any(monthly_used for _, monthly_used in states):
            self.cache[key] = (False, "rest")
            return self.cache[key]

        # A reserve block must be followed either by enough free time for a
        # periodic rest or by a pairing departing at noon or later.
        sorted_pairings = sorted(
            selected, key=lambda p: self.pairing_intervals[p][0]
        )
        nonreserve_activity_busy = [
            activity_interval(a)
            for a in crew.get("pre_assigned_activities", [])
            if not is_reserve_activity(a.get("type", ""))
            and not is_rest_like_activity(a.get("type", ""))
        ]

        for reserve_start, reserve_end in reserves:
            following = [
                p
                for p in sorted_pairings
                if self.pairing_intervals[p][0] >= reserve_end - EPS
            ]
            if not following:
                continue

            next_pairing = following[0]
            pairing = self.pairings[next_pairing]
            next_start = self.pairing_intervals[next_pairing][0]

            if float(pairing["departure_hour"]) >= 12.0 - EPS:
                continue

            interruptions = []
            for p in sorted_pairings:
                a, b = self.pairing_intervals[p]
                if a >= reserve_end - EPS and b <= next_start + EPS:
                    interruptions.append((a, b))
            for a, b in nonreserve_activity_busy:
                if b > reserve_end + EPS and a < next_start - EPS:
                    interruptions.append(
                        (max(a, reserve_end), min(b, next_start))
                    )

            interruptions = merge_intervals(interruptions)
            cursor = reserve_end
            has_periodic_rest = False
            for a, b in interruptions:
                if a - cursor >= weekly_hours - EPS:
                    has_periodic_rest = True
                    break
                cursor = max(cursor, b)
            if next_start - cursor >= weekly_hours - EPS:
                has_periodic_rest = True

            if not has_periodic_rest:
                self.cache[key] = (False, "reserve")
                return self.cache[key]

        self.cache[key] = (True, "ok")
        return self.cache[key]


def build_solution(instance, pairings, selected_by_crew):
    crew_members = instance["crew_members"]
    horizon_minutes = int(instance["planning_horizon_days"]) * 24 * 60

    coverage = defaultdict(int)
    rosters = {}

    for c, crew in enumerate(crew_members):
        selected = sorted(
            selected_by_crew.get(c, []), key=lambda p: pairings[p]["id"]
        )
        pairing_ids = [int(pairings[p]["id"]) for p in selected]

        for p in selected:
            coverage[p] += 1

        preassigned_days = set()
        for activity in crew.get("pre_assigned_activities", []):
            for day in range(
                max(0, int(activity["start_day"])),
                min(
                    int(instance["planning_horizon_days"]) - 1,
                    int(activity["end_day"]),
                )
                + 1,
            ):
                preassigned_days.add(day)

        productive_minutes = 1440 * len(preassigned_days)
        productive_minutes += sum(
            int(pairings[p]["duration_minutes"]) for p in selected
        )
        unproductive = max(0, horizon_minutes - productive_minutes)

        rosters[str(crew["id"])] = {
            "roster_index": 0,
            "pairings": pairing_ids,
            "unproductive_time": int(unproductive),
        }

    uncovered = []
    objective = 0
    for p, pairing in enumerate(pairings):
        missing = max(
            0, int(pairing["coverage_requirement"]) - coverage[p]
        )
        if missing > 0:
            duration = int(pairing["duration_minutes"])
            objective += missing * duration
            uncovered.append(
                {
                    "pairing_id": int(pairing["id"]),
                    "uncovered_count": int(missing),
                    "duration_minutes": duration,
                }
            )

    return {
        "objective_value": float(objective),
        "rosters": rosters,
        "uncovered_pairings": uncovered,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    start_time = time.monotonic()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as f:
        instance = json.load(f)

    pairings = instance["pairings"]
    crews = instance["crew_members"]
    num_pairings = len(pairings)
    num_crews = len(crews)
    horizon_days = int(instance["planning_horizon_days"])
    pairing_intervals = [pairing_interval(p) for p in pairings]

    total_required_duration = sum(
        int(p["duration_minutes"]) * int(p["coverage_requirement"])
        for p in pairings
    )

    # The all-uncovered assignment is always a useful fallback and incumbent.
    fallback_selected = {c: [] for c in range(num_crews)}
    best_solution = build_solution(instance, pairings, fallback_selected)
    best_objective = best_solution["objective_value"]
    if logger:
        logger.log_solution(best_objective, best_solution)

    model = gp.Model("cabin_crew_rosters")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.LazyConstraints = 1

    x = {}
    eligible_by_crew = [[] for _ in range(num_crews)]
    vars_by_pairing = [[] for _ in range(num_pairings)]

    # Create only qualification- and activity-compatible assignment variables.
    for c, crew in enumerate(crews):
        qualifications = set(int(q) for q in crew.get("qualifications", []))
        blocked_days = set(int(d) for d in crew.get("pre_assigned_days", []))
        for activity in crew.get("pre_assigned_activities", []):
            blocked_days.update(
                range(int(activity["start_day"]), int(activity["end_day"]) + 1)
            )

        for p, pairing in enumerate(pairings):
            if int(pairing["aircraft_type"]) not in qualifications:
                continue

            pairing_days = range(
                int(pairing["start_day"]),
                int(pairing["start_day"]) + int(pairing["duration_days"]),
            )
            if any(day in blocked_days for day in pairing_days):
                continue

            variable = model.addVar(
                vtype=GRB.BINARY, name=f"x_{c}_{p}", obj=-int(pairing["duration_minutes"])
            )
            x[c, p] = variable
            eligible_by_crew[c].append(p)
            vars_by_pairing[p].append(variable)

    model.update()

    # Pairing coverage cannot exceed demand. Shortfalls are represented by the
    # uncovered count in the output and by the constant-minus-covered objective.
    for p, pairing in enumerate(pairings):
        if vars_by_pairing[p]:
            model.addConstr(
                gp.quicksum(vars_by_pairing[p])
                <= int(pairing["coverage_requirement"]),
                name=f"coverage_{p}",
            )

    rules = instance.get("work_rules", {})
    monthly_limit = float(rules.get("monthly_flight_limit_hours", 77))
    rolling_limit = float(rules.get("rolling_flight_limit_hours", 85))
    max_cross = int(rules.get("max_cross_pairings_per_month", 1))
    max_early = int(
        rules.get("max_consecutive_early_departures_before_8am", 2)
    )

    for c, crew in enumerate(crews):
        eligible = eligible_by_crew[c]

        if eligible:
            model.addConstr(
                gp.quicksum(
                    float(pairings[p]["flight_hours"]) * x[c, p]
                    for p in eligible
                )
                <= monthly_limit,
                name=f"monthly_hours_{c}",
            )

            first_half = [
                p for p in eligible if int(pairings[p]["start_day"]) <= 14
            ]
            if first_half:
                model.addConstr(
                    gp.quicksum(
                        float(pairings[p]["flight_hours"]) * x[c, p]
                        for p in first_half
                    )
                    + float(crew.get("flight_hours_previous_15_days", 0.0))
                    <= rolling_limit,
                    name=f"rolling_hours_{c}",
                )

            cross = [
                p for p in eligible if bool(pairings[p]["is_cross_pairing"])
            ]
            if cross:
                model.addConstr(
                    gp.quicksum(x[c, p] for p in cross) <= max_cross,
                    name=f"cross_{c}",
                )

        # Interval graph clique constraints prevent pairing overlap.
        groups = defaultdict(list)
        for p in eligible:
            groups[round(pairing_intervals[p][0], 8)].append(p)

        active = []
        for start in sorted(groups):
            active = [
                p for p in active if pairing_intervals[p][1] > start + EPS
            ]
            current_group = groups[start]
            clique = active + current_group
            if len(clique) > 1:
                model.addConstr(
                    gp.quicksum(x[c, p] for p in clique) <= 1,
                    name=f"overlap_{c}_{len(model.getConstrs())}",
                )
            active.extend(current_group)

        # Daily early-departure indicators allow multiple compatible departures
        # on one day to count as one early-departure day.
        early_indicator = {}
        for day in range(horizon_days):
            early_pairings = [
                p
                for p in eligible
                if int(pairings[p]["start_day"]) == day
                and float(pairings[p]["departure_hour"]) < 8.0 - EPS
            ]
            if early_pairings:
                y = model.addVar(vtype=GRB.BINARY, name=f"early_{c}_{day}")
                early_indicator[day] = y
                for p in early_pairings:
                    model.addConstr(x[c, p] <= y)
                model.addConstr(
                    y <= gp.quicksum(x[c, p] for p in early_pairings)
                )

        window_length = max_early + 1
        for first_day in range(0, horizon_days - window_length + 1):
            ys = [
                early_indicator[d]
                for d in range(first_day, first_day + window_length)
                if d in early_indicator
            ]
            if ys:
                model.addConstr(
                    gp.quicksum(ys) <= max_early,
                    name=f"early_window_{c}_{first_day}",
                )

    model.ModelSense = GRB.MINIMIZE
    model.setObjective(
        float(total_required_duration)
        - gp.quicksum(
            int(pairings[p]["duration_minutes"]) * var
            for (c, p), var in x.items()
        )
    )

    # Start from the guaranteed all-uncovered assignment.
    for variable in x.values():
        variable.Start = 0.0

    validator = RosterValidator(instance, pairings, pairing_intervals)

    model._x = x
    model._eligible_by_crew = eligible_by_crew
    model._validator = validator
    model._instance = instance
    model._pairings = pairings
    model._logger = logger
    model._best_objective = best_objective
    model._best_solution = best_solution

    def incumbent_callback(cb_model, where):
        if where != GRB.Callback.MIPSOL:
            return

        selected_by_crew = {}
        invalid = []

        for c in range(num_crews):
            selected = [
                p
                for p in cb_model._eligible_by_crew[c]
                if cb_model.cbGetSolution(cb_model._x[c, p]) > 0.5
            ]
            selected_by_crew[c] = selected
            valid, reason = cb_model._validator.validate(c, selected)
            if not valid:
                invalid.append((c, selected, reason))

        if invalid:
            for c, selected, reason in invalid:
                selected_set = set(selected)
                if reason == "rest":
                    # Rest-rule violations are monotone: adding more pairings
                    # cannot create additional free time.
                    if selected:
                        cb_model.cbLazy(
                            gp.quicksum(cb_model._x[c, p] for p in selected)
                            <= len(selected) - 1
                        )
                    else:
                        cb_model.cbLazy(gp.LinExpr(0.0) <= -1.0)
                else:
                    # The reserve-following rule is not monotone, because an
                    # inserted noon-or-later pairing can change the successor.
                    expression = gp.LinExpr()
                    for p in cb_model._eligible_by_crew[c]:
                        if p in selected_set:
                            expression += cb_model._x[c, p]
                        else:
                            expression -= cb_model._x[c, p]
                    cb_model.cbLazy(expression <= len(selected) - 1)
            return

        solution = build_solution(
            cb_model._instance, cb_model._pairings, selected_by_crew
        )
        objective = solution["objective_value"]

        if objective + EPS < cb_model._best_objective:
            cb_model._best_objective = objective
            cb_model._best_solution = solution
            if cb_model._logger:
                try:
                    cb_model._logger.log_solution(objective, solution)
                except Exception:
                    pass

    elapsed = time.monotonic() - start_time
    remaining = max(0.01, float(args.time_limit) - elapsed)
    model.Params.TimeLimit = remaining

    try:
        model.optimize(incumbent_callback)
    except gp.GurobiError:
        pass

    # Recover Gurobi's final incumbent as a safeguard in case it was accepted
    # without improving the callback-maintained solution.
    if model.SolCount > 0:
        selected_by_crew = {}
        all_valid = True
        for c in range(num_crews):
            selected = [
                p
                for p in eligible_by_crew[c]
                if x[c, p].X > 0.5
            ]
            selected_by_crew[c] = selected
            valid, _ = validator.validate(c, selected)
            all_valid = all_valid and valid

        if all_valid:
            final_candidate = build_solution(
                instance, pairings, selected_by_crew
            )
            if final_candidate["objective_value"] + EPS < model._best_objective:
                model._best_objective = final_candidate["objective_value"]
                model._best_solution = final_candidate
                if logger:
                    logger.log_solution(
                        final_candidate["objective_value"], final_candidate
                    )

    output_solution = model._best_solution

    solution_directory = os.path.dirname(os.path.abspath(args.solution_path))
    if solution_directory:
        os.makedirs(solution_directory, exist_ok=True)
    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(output_solution, f, indent=2)


if __name__ == "__main__":
    main()
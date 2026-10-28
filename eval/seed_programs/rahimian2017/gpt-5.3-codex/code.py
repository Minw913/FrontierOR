import argparse
import json
from collections import defaultdict
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger


def get_dict_value(dct, key, default=None):
    if dct is None:
        return default
    if key in dct:
        return dct[key]
    skey = str(key)
    if skey in dct:
        return dct[skey]
    return default


def build_and_solve(instance, time_limit, logger=None):
    # Basic sets
    D = int(instance["num_days"])
    days = list(range(D))

    nurses_data = instance["nurses"]
    nurse_ids = [n["id"] for n in nurses_data]
    nurse_by_id = {n["id"]: n for n in nurses_data}

    shift_types = instance["shift_types"]
    shift_ids = [s["id"] for s in shift_types]
    shift_len = {s["id"]: int(s["length_minutes"]) for s in shift_types}

    # Forbidden rotations
    forbidden_pairs = []
    for r in instance.get("forbidden_shift_rotations", []):
        a = r["shift_type"]
        b = r["cannot_be_followed_by"]
        if a in shift_len and b in shift_len:
            forbidden_pairs.append((a, b))

    # Coverage requirements (assume unique per (day, shift))
    coverage = {}
    for c in instance.get("coverage_requirements", []):
        d = int(c["day"])
        s = c["shift_type"]
        if d in days and s in shift_len:
            coverage[(d, s)] = (
                int(c["preferred"]),
                int(c["under_weight"]),
                int(c["over_weight"]),
            )
    cov_keys = list(coverage.keys())

    # Aggregate request penalties
    on_pen = defaultdict(int)
    off_pen = defaultdict(int)

    for r in instance.get("shift_on_requests", []):
        n, d, s, p = r["nurse_id"], int(r["day"]), r["shift_type"], int(r["penalty"])
        if n in nurse_by_id and d in days and s in shift_len:
            on_pen[(n, d, s)] += p

    for r in instance.get("shift_off_requests", []):
        n, d, s, p = r["nurse_id"], int(r["day"]), r["shift_type"], int(r["penalty"])
        if n in nurse_by_id and d in days and s in shift_len:
            off_pen[(n, d, s)] += p

    model = Model("nurse_rostering")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = int(time_limit)

    # Variables
    x = model.addVars(nurse_ids, days, shift_ids, vtype=GRB.BINARY, name="x")
    y = model.addVars(nurse_ids, days, vtype=GRB.BINARY, name="y")

    under = model.addVars(cov_keys, vtype=GRB.INTEGER, lb=0, name="under") if cov_keys else {}
    over = model.addVars(cov_keys, vtype=GRB.INTEGER, lb=0, name="over") if cov_keys else {}

    # Work indicator linking / at most one shift per day
    for n in nurse_ids:
        for d in days:
            model.addConstr(quicksum(x[n, d, s] for s in shift_ids) == y[n, d], name=f"link_{n}_{d}")

    # Forbidden successor shifts
    if D >= 2 and forbidden_pairs:
        for n in nurse_ids:
            for d in range(D - 1):
                for a, b in forbidden_pairs:
                    model.addConstr(x[n, d, a] + x[n, d + 1, b] <= 1, name=f"forbid_{n}_{d}_{a}_{b}")

    # Nurse-specific constraints
    num_weekends = int(instance.get("num_weekends", 0))
    weekend_indices = list(range(num_weekends))
    wworked = model.addVars(nurse_ids, weekend_indices, vtype=GRB.BINARY, name="w")

    for n in nurse_ids:
        nd = nurse_by_id[n]

        # Max shifts per type
        max_per_type = nd.get("max_shifts_per_type", {})
        for s in shift_ids:
            max_s = get_dict_value(max_per_type, s, D)
            model.addConstr(quicksum(x[n, d, s] for d in days) <= int(max_s), name=f"max_type_{n}_{s}")

        # Total minutes
        min_min = int(nd.get("min_total_minutes", 0))
        max_min = int(nd.get("max_total_minutes", 10**9))
        total_minutes = quicksum(shift_len[s] * x[n, d, s] for d in days for s in shift_ids)
        model.addConstr(total_minutes >= min_min, name=f"min_minutes_{n}")
        model.addConstr(total_minutes <= max_min, name=f"max_minutes_{n}")

        # Max consecutive shifts (boundary days treated as off)
        max_cons = int(nd.get("max_consecutive_shifts", D))
        if max_cons < D:
            win_len = max_cons + 1
            for start in range(0, D - win_len + 1):
                model.addConstr(
                    quicksum(y[n, t] for t in range(start, start + win_len)) <= max_cons,
                    name=f"max_cons_{n}_{start}",
                )

        # Min consecutive shifts (boundary before/after horizon treated as WORKING = 1)
        min_cons = int(nd.get("min_consecutive_shifts", 1))
        if min_cons > 1:
            for k in range(1, min_cons):
                for start in range(0, D - k + 1):
                    lhs = quicksum(y[n, t] for t in range(start, start + k))
                    rhs_const = k - 1

                    # left boundary day: start-1
                    if start - 1 < 0:
                        rhs_const += k * 1
                        rhs = rhs_const
                    else:
                        rhs = rhs_const + k * y[n, start - 1]

                    # right boundary day: start+k
                    if start + k >= D:
                        rhs = rhs + k
                    else:
                        rhs = rhs + k * y[n, start + k]

                    model.addConstr(lhs <= rhs, name=f"min_cons_work_{n}_{start}_{k}")

        # Min consecutive days off (boundary before/after horizon treated as OFF = 0)
        min_off = int(nd.get("min_consecutive_days_off", 1))
        if min_off > 1:
            for k in range(1, min_off):
                for start in range(0, D - k + 1):
                    lhs = quicksum(y[n, t] for t in range(start, start + k))
                    rhs = 1 - 2 * k
                    if start - 1 >= 0:
                        rhs = rhs + k * y[n, start - 1]
                    if start + k < D:
                        rhs = rhs + k * y[n, start + k]
                    model.addConstr(lhs >= rhs, name=f"min_cons_off_{n}_{start}_{k}")

        # Weekends worked
        max_weekends = int(nd.get("max_weekends", num_weekends))
        for w in weekend_indices:
            sat = 7 * w + 5
            sun = 7 * w + 6
            # whole number of weeks is given, so both should exist; still guard
            if sat < D and sun < D:
                model.addConstr(wworked[n, w] >= y[n, sat], name=f"w_sat_{n}_{w}")
                model.addConstr(wworked[n, w] >= y[n, sun], name=f"w_sun_{n}_{w}")
                model.addConstr(wworked[n, w] <= y[n, sat] + y[n, sun], name=f"w_ub_{n}_{w}")
            elif sat < D:
                model.addConstr(wworked[n, w] == y[n, sat], name=f"w_only_sat_{n}_{w}")
            elif sun < D:
                model.addConstr(wworked[n, w] == y[n, sun], name=f"w_only_sun_{n}_{w}")
            else:
                model.addConstr(wworked[n, w] == 0, name=f"w_none_{n}_{w}")
        model.addConstr(quicksum(wworked[n, w] for w in weekend_indices) <= max_weekends, name=f"max_wkend_{n}")

        # Hard day-off requests
        for d in nd.get("day_off_requests", []):
            d = int(d)
            if 0 <= d < D:
                model.addConstr(y[n, d] == 0, name=f"day_off_{n}_{d}")

    # Coverage balance
    for (d, s), (pref, _, _) in coverage.items():
        model.addConstr(
            quicksum(x[n, d, s] for n in nurse_ids) + under[(d, s)] - over[(d, s)] == pref,
            name=f"cov_{d}_{s}",
        )

    # Objective
    const_on = sum(on_pen.values())
    obj = const_on

    all_req_keys = set(on_pen.keys()) | set(off_pen.keys())
    for key in all_req_keys:
        p_on = on_pen.get(key, 0)
        p_off = off_pen.get(key, 0)
        n, d, s = key
        coeff = p_off - p_on
        if coeff != 0:
            obj += coeff * x[n, d, s]

    for key, (_, uw, ow) in coverage.items():
        obj += uw * under[key] + ow * over[key]

    model.setObjective(obj, GRB.MINIMIZE)

    # Incumbent logging callback
    model._logger = logger
    model._best_logged = float("inf")

    def cb(m, where):
        if where == GRB.Callback.MIPSOL and m._logger is not None:
            val = float(m.cbGet(GRB.Callback.MIPSOL_OBJ))
            if val + 1e-9 < m._best_logged:
                m._best_logged = val
                m._logger.log(val)

    model.optimize(cb)

    return model, x, y, coverage, nurse_ids, shift_ids, days


def extract_solution(instance, model, x, coverage, nurse_ids, shift_ids, days):
    # Default all off schedule
    schedule = {str(n): {str(d): None for d in days} for n in nurse_ids}
    assigned = {(n, d): None for n in nurse_ids for d in days}
    count_day_shift = defaultdict(int)

    if model.SolCount > 0:
        for n in nurse_ids:
            for d in days:
                chosen = None
                for s in shift_ids:
                    if x[n, d, s].X > 0.5:
                        chosen = s
                        break
                assigned[(n, d)] = chosen
                schedule[str(n)][str(d)] = chosen
                if chosen is not None:
                    count_day_shift[(d, chosen)] += 1

    # Recompute objective exactly from definition
    total_pen = 0

    for r in instance.get("shift_on_requests", []):
        n, d, s, p = r["nurse_id"], int(r["day"]), r["shift_type"], int(r["penalty"])
        if n in nurse_ids and d in days and s in shift_ids:
            if assigned[(n, d)] != s:
                total_pen += p

    for r in instance.get("shift_off_requests", []):
        n, d, s, p = r["nurse_id"], int(r["day"]), r["shift_type"], int(r["penalty"])
        if n in nurse_ids and d in days and s in shift_ids:
            if assigned[(n, d)] == s:
                total_pen += p

    coverage_under = {}
    coverage_over = {}
    for (d, s), (pref, uw, ow) in coverage.items():
        actual = count_day_shift.get((d, s), 0)
        u = max(0, pref - actual)
        o = max(0, actual - pref)
        if u > 0:
            coverage_under[f"{d}_{s}"] = int(u)
        if o > 0:
            coverage_over[f"{d}_{s}"] = int(o)
        total_pen += u * uw + o * ow

    return {
        "objective_value": float(total_pen),
        "solution": {
            "schedule": schedule,
            "coverage_under": coverage_under,
            "coverage_over": coverage_over,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    model, x, y, coverage, nurse_ids, shift_ids, days = build_and_solve(
        instance, args.time_limit, logger=logger
    )

    result = extract_solution(instance, model, x, coverage, nurse_ids, shift_ids, days)

    with open(args.solution_path, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
import argparse
import json
import math
import os
import time

import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    return parser.parse_args()


def write_json(path, data):
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    temp_path = path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"), allow_nan=False)
    os.replace(temp_path, path)


def main():
    args = parse_args()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    start_time = time.monotonic()

    with open(args.instance_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    num_days = int(data["num_days"])
    days = list(range(num_days))

    nurse_records = data["nurses"]
    shift_records = data["shift_types"]
    nurse_ids = [int(record["id"]) for record in nurse_records]
    shift_ids = [int(record["id"]) for record in shift_records]

    nurse_by_id = {int(record["id"]): record for record in nurse_records}
    shift_lengths = {
        int(record["id"]): int(record["length_minutes"])
        for record in shift_records
    }

    forbidden_rotations = {
        (int(record["shift_type"]), int(record["cannot_be_followed_by"]))
        for record in data.get("forbidden_shift_rotations", [])
    }

    shift_on_requests = data.get("shift_on_requests", [])
    shift_off_requests = data.get("shift_off_requests", [])

    # The schema specifies a requirement for every day/shift pair. Defaults make
    # the program robust to sparse representations as well.
    coverage = {
        (d, s): {
            "preferred": 0,
            "under_weight": 0,
            "over_weight": 0,
        }
        for d in days
        for s in shift_ids
    }
    for record in data.get("coverage_requirements", []):
        key = (int(record["day"]), int(record["shift_type"]))
        coverage[key] = {
            "preferred": int(record["preferred"]),
            "under_weight": int(record["under_weight"]),
            "over_weight": int(record["over_weight"]),
        }

    model = gp.Model("nurse_rostering")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1

    x_keys = [
        (nurse_id, day, shift_id)
        for nurse_id in nurse_ids
        for day in days
        for shift_id in shift_ids
    ]
    x = model.addVars(x_keys, vtype=GRB.BINARY, name="x")

    work_keys = [(nurse_id, day) for nurse_id in nurse_ids for day in days]
    work = model.addVars(work_keys, vtype=GRB.BINARY, name="work")

    # Exactly one shift or a day off.
    for nurse_id in nurse_ids:
        for day in days:
            model.addConstr(
                work[nurse_id, day]
                == gp.quicksum(x[nurse_id, day, s] for s in shift_ids)
            )

    # Forbidden next-day shift rotations.
    for nurse_id in nurse_ids:
        for day in range(num_days - 1):
            for first_shift, next_shift in forbidden_rotations:
                model.addConstr(
                    x[nurse_id, day, first_shift]
                    + x[nurse_id, day + 1, next_shift]
                    <= 1
                )

    for nurse_id in nurse_ids:
        nurse = nurse_by_id[nurse_id]
        max_per_type = nurse.get("max_shifts_per_type", {})

        # Maximum number of shifts of each type.
        for shift_id in shift_ids:
            maximum = max_per_type.get(
                str(shift_id), max_per_type.get(shift_id, num_days)
            )
            model.addConstr(
                gp.quicksum(x[nurse_id, day, shift_id] for day in days)
                <= int(maximum)
            )

        # Total working time.
        total_minutes = gp.quicksum(
            shift_lengths[shift_id] * x[nurse_id, day, shift_id]
            for day in days
            for shift_id in shift_ids
        )
        model.addConstr(total_minutes >= int(nurse["min_total_minutes"]))
        model.addConstr(total_minutes <= int(nurse["max_total_minutes"]))

        # Maximum consecutive working days. Outside the horizon is treated as off.
        max_consecutive = int(nurse["max_consecutive_shifts"])
        if max_consecutive < num_days:
            window_length = max_consecutive + 1
            for first_day in range(num_days - window_length + 1):
                model.addConstr(
                    gp.quicksum(
                        work[nurse_id, day]
                        for day in range(first_day, first_day + window_length)
                    )
                    <= max_consecutive
                )

        # Minimum consecutive working days. Before and after the horizon are
        # treated as an infinite stretch of working days, so only starts after
        # an in-horizon off-day need to be constrained.
        min_consecutive = int(nurse["min_consecutive_shifts"])
        if min_consecutive > 1:
            for first_day in range(1, num_days):
                last_in_horizon = min(
                    num_days - 1, first_day + min_consecutive - 1
                )
                in_horizon_count = last_in_horizon - first_day + 1
                assumed_work_after = min_consecutive - in_horizon_count
                model.addConstr(
                    gp.quicksum(
                        work[nurse_id, day]
                        for day in range(first_day, last_in_horizon + 1)
                    )
                    + assumed_work_after
                    >= min_consecutive
                    * (work[nurse_id, first_day] - work[nurse_id, first_day - 1])
                )

        # Minimum consecutive days off. Outside the horizon is treated as an
        # infinite stretch of days off.
        min_days_off = int(nurse["min_consecutive_days_off"])
        if min_days_off > 1:
            for first_day in range(1, num_days):
                last_in_horizon = min(
                    num_days - 1, first_day + min_days_off - 1
                )
                in_horizon_count = last_in_horizon - first_day + 1
                assumed_off_after = min_days_off - in_horizon_count
                model.addConstr(
                    gp.quicksum(
                        1 - work[nurse_id, day]
                        for day in range(first_day, last_in_horizon + 1)
                    )
                    + assumed_off_after
                    >= min_days_off
                    * (work[nurse_id, first_day - 1] - work[nurse_id, first_day])
                )

        # Mandatory days off.
        for day in nurse.get("day_off_requests", []):
            day = int(day)
            if 0 <= day < num_days:
                model.addConstr(work[nurse_id, day] == 0)

    # Weekend variables. Since the horizon starts on Monday, Saturday and Sunday
    # have offsets 5 and 6 in each complete week.
    weekend_pairs = []
    requested_weekends = int(data.get("num_weekends", data.get("num_weeks", 0)))
    for weekend_index in range(requested_weekends):
        saturday = 7 * weekend_index + 5
        sunday = 7 * weekend_index + 6
        weekend_days = [
            day for day in (saturday, sunday) if 0 <= day < num_days
        ]
        if weekend_days:
            weekend_pairs.append((weekend_index, weekend_days))

    weekend_keys = [
        (nurse_id, weekend_index)
        for nurse_id in nurse_ids
        for weekend_index, _ in weekend_pairs
    ]
    weekend_worked = model.addVars(
        weekend_keys, vtype=GRB.BINARY, name="weekend_worked"
    )

    for nurse_id in nurse_ids:
        for weekend_index, weekend_days in weekend_pairs:
            for day in weekend_days:
                model.addConstr(
                    weekend_worked[nurse_id, weekend_index]
                    >= work[nurse_id, day]
                )
            model.addConstr(
                weekend_worked[nurse_id, weekend_index]
                <= gp.quicksum(work[nurse_id, day] for day in weekend_days)
            )

        model.addConstr(
            gp.quicksum(
                weekend_worked[nurse_id, weekend_index]
                for weekend_index, _ in weekend_pairs
            )
            <= int(nurse_by_id[nurse_id]["max_weekends"])
        )

    coverage_keys = [(day, shift_id) for day in days for shift_id in shift_ids]
    under = model.addVars(
        coverage_keys,
        lb=0.0,
        ub=len(nurse_ids),
        vtype=GRB.CONTINUOUS,
        name="coverage_under",
    )
    over = model.addVars(
        coverage_keys,
        lb=0.0,
        ub=len(nurse_ids),
        vtype=GRB.CONTINUOUS,
        name="coverage_over",
    )

    for day, shift_id in coverage_keys:
        preferred = coverage[(day, shift_id)]["preferred"]
        model.addConstr(
            gp.quicksum(x[nurse_id, day, shift_id] for nurse_id in nurse_ids)
            + under[day, shift_id]
            - over[day, shift_id]
            == preferred
        )

    objective = gp.LinExpr()

    for request in shift_on_requests:
        nurse_id = int(request["nurse_id"])
        day = int(request["day"])
        shift_id = int(request["shift_type"])
        penalty = float(request["penalty"])
        objective += penalty * (1 - x[nurse_id, day, shift_id])

    for request in shift_off_requests:
        nurse_id = int(request["nurse_id"])
        day = int(request["day"])
        shift_id = int(request["shift_type"])
        penalty = float(request["penalty"])
        objective += penalty * x[nurse_id, day, shift_id]

    for day, shift_id in coverage_keys:
        requirement = coverage[(day, shift_id)]
        objective += requirement["under_weight"] * under[day, shift_id]
        objective += requirement["over_weight"] * over[day, shift_id]

    model.setObjective(objective, GRB.MINIMIZE)

    x_var_list = [x[key] for key in x_keys]

    def make_solution(x_values):
        assigned = {
            (nurse_id, day): None
            for nurse_id in nurse_ids
            for day in days
        }
        actual_coverage = {
            (day, shift_id): 0
            for day in days
            for shift_id in shift_ids
        }

        for key, value in zip(x_keys, x_values):
            if value > 0.5:
                nurse_id, day, shift_id = key
                assigned[nurse_id, day] = shift_id
                actual_coverage[day, shift_id] += 1

        schedule = {
            str(nurse_id): {
                str(day): assigned[nurse_id, day]
                for day in days
            }
            for nurse_id in nurse_ids
        }

        coverage_under = {}
        coverage_over = {}
        for day, shift_id in coverage_keys:
            actual = actual_coverage[day, shift_id]
            preferred = coverage[(day, shift_id)]["preferred"]
            coverage_under[f"{day}_{shift_id}"] = max(0, preferred - actual)
            coverage_over[f"{day}_{shift_id}"] = max(0, actual - preferred)

        objective_value = 0.0

        for request in shift_on_requests:
            nurse_id = int(request["nurse_id"])
            day = int(request["day"])
            shift_id = int(request["shift_type"])
            if assigned[nurse_id, day] != shift_id:
                objective_value += float(request["penalty"])

        for request in shift_off_requests:
            nurse_id = int(request["nurse_id"])
            day = int(request["day"])
            shift_id = int(request["shift_type"])
            if assigned[nurse_id, day] == shift_id:
                objective_value += float(request["penalty"])

        for day, shift_id in coverage_keys:
            key = f"{day}_{shift_id}"
            requirement = coverage[(day, shift_id)]
            objective_value += (
                coverage_under[key] * requirement["under_weight"]
            )
            objective_value += (
                coverage_over[key] * requirement["over_weight"]
            )

        result = {
            "objective_value": float(objective_value),
            "solution": {
                "schedule": schedule,
                "coverage_under": coverage_under,
                "coverage_over": coverage_over,
            },
        }
        return float(objective_value), result

    callback_best = [math.inf]
    callback_best_solution = [None]

    def incumbent_callback(cb_model, where):
        if where != GRB.Callback.MIPSOL:
            return
        model_objective = cb_model.cbGet(GRB.Callback.MIPSOL_OBJ)
        if model_objective >= callback_best[0] - 1e-7:
            return

        values = cb_model.cbGetSolution(x_var_list)
        true_objective, solution = make_solution(values)
        callback_best[0] = model_objective
        callback_best_solution[0] = solution

        if logger:
            logger.log_solution(true_objective, solution)

    elapsed = time.monotonic() - start_time
    model.Params.TimeLimit = max(0.01, float(args.time_limit) - elapsed)
    model.optimize(incumbent_callback)

    if model.SolCount > 0:
        final_values = [var.X for var in x_var_list]
        final_objective, final_solution = make_solution(final_values)

        # Normally every incumbent is seen by MIPSOL. This also handles solver
        # configurations where the final incumbent was not passed to the callback.
        if callback_best_solution[0] is None:
            if logger:
                logger.log_solution(final_objective, final_solution)
        else:
            logged_obj = float(callback_best_solution[0]["objective_value"])
            if final_objective < logged_obj - 1e-7 and logger:
                logger.log_solution(final_objective, final_solution)

        write_json(args.solution_path, final_solution)
        return

    # Valid benchmark instances are expected to be feasible. If no incumbent was
    # found within the allotted time, report the condition rather than emitting
    # a structurally valid but infeasible roster.
    raise RuntimeError(
        "No feasible solution was found within the allotted time limit."
    )


if __name__ == "__main__":
    main()
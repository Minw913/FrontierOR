import argparse
import json
import os
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def parse_args():
    parser = argparse.ArgumentParser(description="Solve the nurse scheduling problem.")
    parser.add_argument("--instance_path", required=True, help="Path to the instance JSON file.")
    parser.add_argument("--solution_path", required=True, help="Path for the solution JSON file.")
    parser.add_argument("--time_limit", required=True, type=int, help="Maximum runtime in seconds.")
    parser.add_argument("--log_path", default=None, help="Optional incumbent JSONL log path.")
    return parser.parse_args()


def add_sequence_penalties(model, sequence, minimum, maximum, prefix):
    """
    Return an exact linear penalty expression for maximal 1-runs in sequence.

    A run shorter than minimum contributes minimum - run_length.
    A run longer than maximum contributes run_length - maximum.
    Runs touching either horizon boundary are treated as complete runs within
    the planning horizon.
    """
    horizon = len(sequence)
    penalty = gp.LinExpr()

    minimum = max(0, int(minimum))
    maximum = max(0, int(maximum))

    # Short runs: an exact maximal run of length l < minimum incurs minimum-l.
    for length in range(1, min(minimum, horizon + 1)):
        deviation = minimum - length
        for start in range(horizon - length + 1):
            end = start + length - 1
            violation = model.addVar(
                lb=0.0,
                ub=1.0,
                vtype=GRB.CONTINUOUS,
                name=f"{prefix}_short_{start}_{length}",
            )

            segment_sum = gp.quicksum(sequence[start:end + 1])
            left = sequence[start - 1] if start > 0 else 0.0
            right = sequence[end + 1] if end + 1 < horizon else 0.0

            # Equals 1 exactly when [start,end] is a maximal all-one run.
            model.addConstr(
                violation >= segment_sum - left - right - (length - 1),
                name=f"{prefix}_short_c_{start}_{length}",
            )
            penalty += deviation * violation

    # Each all-one window of maximum+1 contributes one excess unit. A run of
    # length L therefore contributes max(0, L-maximum).
    window_length = maximum + 1
    if window_length <= horizon:
        for start in range(horizon - window_length + 1):
            violation = model.addVar(
                lb=0.0,
                ub=1.0,
                vtype=GRB.CONTINUOUS,
                name=f"{prefix}_long_{start}",
            )
            model.addConstr(
                violation
                >= gp.quicksum(sequence[start:start + window_length]) - maximum,
                name=f"{prefix}_long_c_{start}",
            )
            penalty += violation

    return penalty


def build_solution(instance, x_keys, x_values, objective_value):
    assignments_by_nurse = {n: [] for n in range(len(instance["nurses"]))}

    for key, value in zip(x_keys, x_values):
        if value <= 0.5:
            continue
        nurse_idx, day, shift_idx, skill_idx = key
        assignments_by_nurse[nurse_idx].append(
            {
                "day": int(day),
                "shift": instance["shifts"][shift_idx]["name"],
                "skill": instance["skills"][skill_idx],
            }
        )

    schedule = []
    for nurse_idx, nurse in enumerate(instance["nurses"]):
        assignments = assignments_by_nurse[nurse_idx]
        assignments.sort(key=lambda item: (item["day"], item["shift"], item["skill"]))
        schedule.append(
            {
                "nurse_id": int(nurse["id"]),
                "nurse_name": nurse["name"],
                "assignments": assignments,
            }
        )

    return {
        "objective_value": float(objective_value),
        "schedule": schedule,
    }


def main():
    args = parse_args()
    start_time = time.monotonic()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as handle:
        instance = json.load(handle)

    nurses = instance["nurses"]
    shifts = instance["shifts"]
    skills = instance["skills"]

    num_nurses = len(nurses)
    num_days = int(instance["num_days"])
    num_shifts = len(shifts)
    num_skills = len(skills)

    shift_index = {shift["name"]: idx for idx, shift in enumerate(shifts)}
    skill_index = {skill: idx for idx, skill in enumerate(skills)}

    qualified_skills = []
    for nurse in nurses:
        qualified_skills.append(
            sorted({skill_index[s] for s in nurse["skills"] if s in skill_index})
        )

    weights = instance["penalty_weights"]

    model = gp.Model("nurse_scheduling")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.MIPFocus = 1
    model.Params.Heuristics = 0.20

    # x[n,d,s,k] indicates assignment to shift s while performing skill k.
    x = {}
    x_keys = []
    x_vars = []

    for n in range(num_nurses):
        for d in range(num_days):
            for s in range(num_shifts):
                for k in qualified_skills[n]:
                    key = (n, d, s, k)
                    var = model.addVar(
                        vtype=GRB.BINARY,
                        name=f"x_{n}_{d}_{s}_{k}",
                    )
                    x[key] = var
                    x_keys.append(key)
                    x_vars.append(var)

    # z[n,d,s] indicates that nurse n works shift s on day d.
    z = {}
    for n in range(num_nurses):
        for d in range(num_days):
            for s in range(num_shifts):
                z[n, d, s] = model.addVar(
                    vtype=GRB.BINARY,
                    name=f"z_{n}_{d}_{s}",
                )
                model.addConstr(
                    z[n, d, s]
                    == gp.quicksum(
                        x[n, d, s, k]
                        for k in qualified_skills[n]
                        if (n, d, s, k) in x
                    ),
                    name=f"shift_link_{n}_{d}_{s}",
                )

    # y[n,d] indicates that nurse n works on day d.
    y = {}
    for n in range(num_nurses):
        for d in range(num_days):
            y[n, d] = model.addVar(
                vtype=GRB.BINARY,
                name=f"y_{n}_{d}",
            )
            model.addConstr(
                y[n, d] == gp.quicksum(z[n, d, s] for s in range(num_shifts)),
                name=f"day_link_{n}_{d}",
            )

    objective = gp.LinExpr()

    # Hard minimum coverage and soft optimal coverage.
    demand = instance["demand"]
    for d in range(num_days):
        for s, shift in enumerate(shifts):
            shift_name = shift["name"]
            shift_demand = demand[d].get(shift_name, {})

            for k, skill_name in enumerate(skills):
                skill_demand = shift_demand.get(
                    skill_name, {"minimum": 0, "optimal": 0}
                )
                minimum = int(skill_demand.get("minimum", 0))
                optimal = int(skill_demand.get("optimal", minimum))

                coverage = gp.quicksum(
                    x[n, d, s, k]
                    for n in range(num_nurses)
                    if (n, d, s, k) in x
                )

                if minimum > 0:
                    model.addConstr(
                        coverage >= minimum,
                        name=f"minimum_coverage_{d}_{s}_{k}",
                    )

                if optimal > 0:
                    shortfall = model.addVar(
                        lb=0.0,
                        ub=float(optimal),
                        vtype=GRB.CONTINUOUS,
                        name=f"optimal_shortfall_{d}_{s}_{k}",
                    )
                    model.addConstr(
                        shortfall >= optimal - coverage,
                        name=f"optimal_shortfall_c_{d}_{s}_{k}",
                    )
                    objective += float(weights["c_S1"]) * shortfall

    # Hard forbidden shift successions.
    for pair_idx, pair in enumerate(instance.get("forbidden_shift_successions", [])):
        if len(pair) != 2 or pair[0] not in shift_index or pair[1] not in shift_index:
            continue
        first = shift_index[pair[0]]
        second = shift_index[pair[1]]
        for n in range(num_nurses):
            for d in range(num_days - 1):
                model.addConstr(
                    z[n, d, first] + z[n, d + 1, second] <= 1,
                    name=f"forbidden_{pair_idx}_{n}_{d}",
                )

    # Consecutive worked days, days off, totals, preferences, and weekends.
    for n, nurse in enumerate(nurses):
        contract = nurse["contract"]
        work_sequence = [y[n, d] for d in range(num_days)]

        objective += float(weights["c_S2a"]) * add_sequence_penalties(
            model,
            work_sequence,
            int(contract["CD_minus"]),
            int(contract["CD_plus"]),
            f"workrun_{n}",
        )

        # Rest indicators equal 1-y. Linear expressions are sufficient.
        rest_sequence = [1 - y[n, d] for d in range(num_days)]
        objective += float(weights["c_S3"]) * add_sequence_penalties(
            model,
            rest_sequence,
            int(contract["CR_minus"]),
            int(contract["CR_plus"]),
            f"restrun_{n}",
        )

        total_work = gp.quicksum(work_sequence)

        under_total = model.addVar(
            lb=0.0,
            vtype=GRB.CONTINUOUS,
            name=f"total_under_{n}",
        )
        over_total = model.addVar(
            lb=0.0,
            vtype=GRB.CONTINUOUS,
            name=f"total_over_{n}",
        )
        model.addConstr(
            under_total >= int(contract["L_minus"]) - total_work,
            name=f"total_under_c_{n}",
        )
        model.addConstr(
            over_total >= total_work - int(contract["L_plus"]),
            name=f"total_over_c_{n}",
        )
        objective += float(weights["c_S6"]) * (under_total + over_total)

        preferences = {
            (int(pref["day"]), pref["shift"])
            for pref in nurse.get("preferences", [])
        }
        for day, shift_name in preferences:
            if 0 <= day < num_days and shift_name in shift_index:
                objective += (
                    float(weights["c_S4"])
                    * z[n, day, shift_index[shift_name]]
                )

        worked_weekends = []
        for week in range(int(instance["num_weeks"])):
            saturday = 7 * week + 5
            sunday = 7 * week + 6
            if sunday >= num_days:
                continue

            weekend_worked = model.addVar(
                vtype=GRB.BINARY,
                name=f"weekend_worked_{n}_{week}",
            )
            model.addConstr(
                weekend_worked >= y[n, saturday],
                name=f"weekend_sat_{n}_{week}",
            )
            model.addConstr(
                weekend_worked >= y[n, sunday],
                name=f"weekend_sun_{n}_{week}",
            )
            model.addConstr(
                weekend_worked <= y[n, saturday] + y[n, sunday],
                name=f"weekend_upper_{n}_{week}",
            )
            worked_weekends.append(weekend_worked)

            incomplete = model.addVar(
                lb=0.0,
                ub=1.0,
                vtype=GRB.CONTINUOUS,
                name=f"incomplete_weekend_{n}_{week}",
            )
            model.addConstr(
                incomplete >= y[n, saturday] - y[n, sunday],
                name=f"incomplete_sat_{n}_{week}",
            )
            model.addConstr(
                incomplete >= y[n, sunday] - y[n, saturday],
                name=f"incomplete_sun_{n}_{week}",
            )
            objective += float(weights["c_S5"]) * incomplete

        excess_weekends = model.addVar(
            lb=0.0,
            vtype=GRB.CONTINUOUS,
            name=f"excess_weekends_{n}",
        )
        model.addConstr(
            excess_weekends
            >= gp.quicksum(worked_weekends) - int(contract["WE_plus"]),
            name=f"excess_weekends_c_{n}",
        )
        objective += float(weights["c_S7"]) * excess_weekends

    # Consecutive assignments to the same shift.
    for n in range(num_nurses):
        for s, shift in enumerate(shifts):
            shift_sequence = [z[n, d, s] for d in range(num_days)]
            objective += float(weights["c_S2b"]) * add_sequence_penalties(
                model,
                shift_sequence,
                int(shift["CS_minus"]),
                int(shift["CS_plus"]),
                f"shiftrun_{n}_{s}",
            )

    model.setObjective(objective, GRB.MINIMIZE)

    elapsed = time.monotonic() - start_time
    model.Params.TimeLimit = max(0.001, float(args.time_limit) - elapsed)

    best_logged = [float("inf")]

    def incumbent_callback(cb_model, where):
        if where != GRB.Callback.MIPSOL or logger is None:
            return
        try:
            incumbent_objective = float(
                cb_model.cbGet(GRB.Callback.MIPSOL_OBJ)
            )
            if incumbent_objective >= best_logged[0] - 1e-7:
                return

            values = cb_model.cbGetSolution(x_vars)
            solution = build_solution(
                instance,
                x_keys,
                values,
                incumbent_objective,
            )
            logger.log_solution(incumbent_objective, solution)
            best_logged[0] = incumbent_objective
        except Exception:
            # Logging must never interrupt the optimization.
            pass

    if logger is not None:
        model.optimize(incumbent_callback)
    else:
        model.optimize()

    if model.SolCount <= 0:
        raise RuntimeError(
            "No feasible schedule was found within the supplied time limit."
        )

    final_objective = float(model.ObjVal)
    final_values = [var.X for var in x_vars]
    final_solution = build_solution(
        instance,
        x_keys,
        final_values,
        final_objective,
    )

    if logger is not None and final_objective < best_logged[0] - 1e-7:
        try:
            logger.log_solution(final_objective, final_solution)
        except Exception:
            pass

    solution_directory = os.path.dirname(os.path.abspath(args.solution_path))
    if solution_directory:
        os.makedirs(solution_directory, exist_ok=True)

    with open(args.solution_path, "w", encoding="utf-8") as handle:
        json.dump(final_solution, handle, indent=2)


if __name__ == "__main__":
    main()
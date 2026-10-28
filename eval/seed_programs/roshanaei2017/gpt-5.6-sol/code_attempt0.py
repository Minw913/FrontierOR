import argparse
import json
import os
import time
from collections import defaultdict
from itertools import combinations

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


def key_get(mapping, key, default=None):
    if mapping is None:
        return default
    if key in mapping:
        return mapping[key]
    skey = str(key)
    if skey in mapping:
        return mapping[skey]
    return default


def clean_number(value):
    value = float(value)
    if abs(value) < 1e-8:
        return 0.0
    return value


def main():
    args = parse_args()
    start_wall = time.monotonic()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    patients = data["patients"]
    surgeons = data["surgeons"]
    hospitals = data["hospitals"]
    num_days = int(data["num_days"])
    alpha = float(data.get("alpha", 0.0))
    days = list(range(1, num_days + 1))

    patient_by_index = {i: p for i, p in enumerate(patients)}
    surgeon_by_id = {int(s["surgeon_id"]): s for s in surgeons}
    hospital_by_id = {int(h["hospital_id"]): h for h in hospitals}

    room_by_key = {}
    room_day_data = {}
    for hospital in hospitals:
        h_id = int(hospital["hospital_id"])
        for room in hospital["ORs"]:
            r_id = int(room["or_id"])
            room_by_key[(h_id, r_id)] = room
            daily = room.get("daily", {})
            for day in days:
                day_info = key_get(daily, day, {}) or {}
                regular = int(day_info.get("regular_time", room["regular_time"]))
                max_ot = int(day_info.get("max_overtime", room["max_overtime"]))
                room_day_data[(h_id, day, r_id)] = {
                    "regular_time": regular,
                    "max_overtime": max_ot,
                    "fixed_cost": float(room.get("fixed_cost_per_hour", 0.0))
                    * regular
                    / 60.0,
                    "overtime_rate": float(room.get("overtime_cost_per_hour", 0.0))
                    / 60.0,
                }

    max_room_horizon = max(
        (
            rd["regular_time"] + rd["max_overtime"]
            for rd in room_day_data.values()
        ),
        default=1,
    )
    max_time = max(1.0, float(max_room_horizon))

    surgeon_availability = {}
    for s_id, surgeon in surgeon_by_id.items():
        operating_days = {int(d) for d in surgeon.get("operating_days", [])}
        availability_map = surgeon.get("availability_by_day", {})
        for day in days:
            availability = float(key_get(availability_map, day, 0) or 0)
            if day in operating_days and availability > 0:
                surgeon_availability[(s_id, day)] = availability

    model = gp.Model("coalition_surgery_scheduling")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1

    remaining_time = max(0.05, args.time_limit - (time.monotonic() - start_wall))
    model.Params.TimeLimit = remaining_time

    # Candidate assignment variables:
    # (patient index, surgeon id, hospital id, day, OR id)
    x = {}
    x_metadata = []

    room_presence = defaultdict(list)
    surgeon_presence = defaultdict(list)
    patient_candidates = defaultdict(list)

    for p_idx, patient in patient_by_index.items():
        due_date = int(patient["due_date"])
        mandatory = bool(patient["is_mandatory"])
        eligible_surgeons = [int(s) for s in patient.get("eligible_surgeons", [])]
        eligible_rooms_map = patient.get("eligible_ORs_by_hospital", {})
        surgeon_times = patient.get("surgeon_specific_times", {})

        allowed_days = [d for d in days if (not mandatory or d <= due_date)]

        for s_id in eligible_surgeons:
            if s_id not in surgeon_by_id:
                continue
            surgical_time = key_get(surgeon_times, s_id, None)
            if surgical_time is None:
                continue

            for day in allowed_days:
                if (s_id, day) not in surgeon_availability:
                    continue

                for h_id in hospital_by_id:
                    eligible_rooms = key_get(eligible_rooms_map, h_id, []) or []
                    for r_id_raw in eligible_rooms:
                        r_id = int(r_id_raw)
                        if (h_id, r_id) not in room_by_key:
                            continue

                        key = (p_idx, s_id, h_id, day, r_id)
                        var = model.addVar(vtype=GRB.BINARY, name=f"x_{p_idx}_{s_id}_{h_id}_{day}_{r_id}")
                        x[key] = var
                        x_metadata.append(key)
                        patient_candidates[p_idx].append(var)
                        room_presence[(p_idx, h_id, day, r_id)].append(var)
                        surgeon_presence[(p_idx, s_id, h_id, day)].append(var)

    z = {}
    room_open = {}
    overtime = {}
    room_completion = {}
    surgeon_deployed = {}
    surgeon_start = {}
    surgeon_end = {}

    room_assignment_vars = defaultdict(list)
    surgeon_assignment_vars = defaultdict(list)

    for key, variables in room_presence.items():
        p_idx, h_id, day, r_id = key
        room_assignment_vars[(h_id, day, r_id)].extend(variables)

    for key, variables in surgeon_presence.items():
        p_idx, s_id, h_id, day = key
        surgeon_assignment_vars[(s_id, h_id, day)].extend(variables)

    for p_idx, patient in patient_by_index.items():
        mandatory = bool(patient["is_mandatory"])
        if mandatory:
            z[p_idx] = model.addVar(lb=1.0, ub=1.0, vtype=GRB.BINARY, name=f"z_{p_idx}")
        else:
            z[p_idx] = model.addVar(vtype=GRB.BINARY, name=f"z_{p_idx}")

        candidates = patient_candidates.get(p_idx, [])
        model.addConstr(gp.quicksum(candidates) == z[p_idx], name=f"assign_once_{p_idx}")

    for h_id, day, r_id in room_assignment_vars:
        rd = room_day_data[(h_id, day, r_id)]
        room_open[(h_id, day, r_id)] = model.addVar(
            vtype=GRB.BINARY, name=f"open_{h_id}_{day}_{r_id}"
        )
        overtime[(h_id, day, r_id)] = model.addVar(
            lb=0.0,
            ub=float(rd["max_overtime"]),
            vtype=GRB.CONTINUOUS,
            name=f"ot_{h_id}_{day}_{r_id}",
        )
        room_completion[(h_id, day, r_id)] = model.addVar(
            lb=0.0,
            ub=float(rd["regular_time"] + rd["max_overtime"]),
            vtype=GRB.CONTINUOUS,
            name=f"completion_{h_id}_{day}_{r_id}",
        )

    for s_id, h_id, day in surgeon_assignment_vars:
        availability = surgeon_availability[(s_id, day)]
        surgeon_deployed[(s_id, h_id, day)] = model.addVar(
            vtype=GRB.BINARY, name=f"deploy_{s_id}_{h_id}_{day}"
        )
        surgeon_start[(s_id, h_id, day)] = model.addVar(
            lb=0.0, ub=max_time, vtype=GRB.CONTINUOUS,
            name=f"surgeon_start_{s_id}_{h_id}_{day}"
        )
        surgeon_end[(s_id, h_id, day)] = model.addVar(
            lb=0.0, ub=max_time, vtype=GRB.CONTINUOUS,
            name=f"surgeon_end_{s_id}_{h_id}_{day}"
        )
        model.addConstr(
            surgeon_end[(s_id, h_id, day)] - surgeon_start[(s_id, h_id, day)]
            <= availability,
            name=f"surgeon_span_{s_id}_{h_id}_{day}",
        )

    model.update()

    room_entry = {}
    surgery_start = {}
    finish = {}
    room_exit = {}

    for p_idx, patient in patient_by_index.items():
        room_entry[p_idx] = model.addVar(
            lb=0.0, ub=max_time, vtype=GRB.CONTINUOUS, name=f"entry_{p_idx}"
        )
        surgery_start[p_idx] = model.addVar(
            lb=0.0, ub=max_time, vtype=GRB.CONTINUOUS, name=f"surgery_start_{p_idx}"
        )
        finish[p_idx] = model.addVar(
            lb=0.0, ub=max_time, vtype=GRB.CONTINUOUS, name=f"finish_{p_idx}"
        )
        room_exit[p_idx] = model.addVar(
            lb=0.0, ub=max_time, vtype=GRB.CONTINUOUS, name=f"exit_{p_idx}"
        )

    model.update()

    # Timing identities.
    for p_idx, patient in patient_by_index.items():
        prep = float(patient["preparation_time"])
        cleaning = float(patient["cleaning_time"])
        surgeon_times = patient.get("surgeon_specific_times", {})

        duration_expr = gp.quicksum(
            float(key_get(surgeon_times, s_id, 0.0)) * var
            for (pp, s_id, h_id, day, r_id), var in x.items()
            if pp == p_idx
        )

        model.addConstr(
            room_entry[p_idx] == surgery_start[p_idx] - prep * z[p_idx],
            name=f"entry_identity_{p_idx}",
        )
        model.addConstr(
            finish[p_idx] == surgery_start[p_idx] + duration_expr,
            name=f"finish_identity_{p_idx}",
        )
        model.addConstr(
            room_exit[p_idx] == finish[p_idx] + cleaning * z[p_idx],
            name=f"exit_identity_{p_idx}",
        )
        model.addConstr(
            room_exit[p_idx] <= max_time * z[p_idx],
            name=f"inactive_time_{p_idx}",
        )

    # Assignment activation and aggregate room capacity.
    for room_key, assigned_vars in room_assignment_vars.items():
        h_id, day, r_id = room_key
        u = room_open[room_key]
        ot = overtime[room_key]
        completion = room_completion[room_key]
        rd = room_day_data[room_key]

        for var in assigned_vars:
            model.addConstr(var <= u)

        model.addConstr(u <= gp.quicksum(assigned_vars), name=f"no_empty_room_{h_id}_{day}_{r_id}")
        model.addConstr(ot <= rd["max_overtime"] * u)
        model.addConstr(completion <= (rd["regular_time"] + rd["max_overtime"]) * u)
        model.addConstr(completion <= rd["regular_time"] + ot)

        capacity_terms = []
        for key, var in x.items():
            p_idx, s_id, hh, dd, rr = key
            if (hh, dd, rr) != room_key:
                continue
            patient = patient_by_index[p_idx]
            surg = float(key_get(patient["surgeon_specific_times"], s_id, 0.0))
            total_time = (
                float(patient["preparation_time"])
                + surg
                + float(patient["cleaning_time"])
            )
            capacity_terms.append(total_time * var)

        model.addConstr(
            gp.quicksum(capacity_terms)
            <= rd["regular_time"] * u + ot,
            name=f"room_capacity_{h_id}_{day}_{r_id}",
        )

    # Room completion lower bounds.
    for (p_idx, h_id, day, r_id), presence_vars in room_presence.items():
        presence = gp.quicksum(presence_vars)
        completion = room_completion[(h_id, day, r_id)]
        model.addConstr(
            completion >= room_exit[p_idx] - max_time * (1 - presence),
            name=f"room_completion_lb_{p_idx}_{h_id}_{day}_{r_id}",
        )

    # Surgeon deployment, workload, and span constraints.
    for surgeon_key, assigned_vars in surgeon_assignment_vars.items():
        s_id, h_id, day = surgeon_key
        v = surgeon_deployed[surgeon_key]
        ss = surgeon_start[surgeon_key]
        se = surgeon_end[surgeon_key]
        availability = surgeon_availability[(s_id, day)]

        for var in assigned_vars:
            model.addConstr(var <= v)

        model.addConstr(v <= gp.quicksum(assigned_vars))
        model.addConstr(ss <= max_time * v)
        model.addConstr(se <= max_time * v)
        model.addConstr(ss <= se)

        workload_terms = []
        for key, var in x.items():
            p_idx, ss_id, hh, dd, r_id = key
            if (ss_id, hh, dd) != surgeon_key:
                continue
            patient = patient_by_index[p_idx]
            surgical = float(key_get(patient["surgeon_specific_times"], s_id, 0.0))
            total = (
                float(patient["preparation_time"])
                + surgical
                + float(patient["cleaning_time"])
            )
            weighted = alpha * surgical + (1.0 - alpha) * total
            workload_terms.append(weighted * var)

        model.addConstr(
            gp.quicksum(workload_terms) <= availability * v,
            name=f"surgeon_workload_{s_id}_{h_id}_{day}",
        )

    for (p_idx, s_id, h_id, day), presence_vars in surgeon_presence.items():
        presence = gp.quicksum(presence_vars)
        ss = surgeon_start[(s_id, h_id, day)]
        se = surgeon_end[(s_id, h_id, day)]
        model.addConstr(
            ss <= surgery_start[p_idx] + max_time * (1 - presence),
            name=f"surgeon_start_bound_{p_idx}_{s_id}_{h_id}_{day}",
        )
        model.addConstr(
            se >= finish[p_idx] - max_time * (1 - presence),
            name=f"surgeon_end_bound_{p_idx}_{s_id}_{h_id}_{day}",
        )

    # A surgeon can work at at most one hospital per day.
    for s_id in surgeon_by_id:
        for day in days:
            deployments = [
                v for (ss, h_id, dd), v in surgeon_deployed.items()
                if ss == s_id and dd == day
            ]
            if deployments:
                model.addConstr(
                    gp.quicksum(deployments) <= 1,
                    name=f"one_hospital_{s_id}_{day}",
                )

    # Pairwise room non-overlap.
    room_groups = defaultdict(list)
    for (p_idx, h_id, day, r_id), variables in room_presence.items():
        room_groups[(h_id, day, r_id)].append((p_idx, variables))

    for (h_id, day, r_id), members in room_groups.items():
        members.sort(key=lambda item: item[0])
        for (p_idx, p_vars), (k_idx, k_vars) in combinations(members, 2):
            p_present = gp.quicksum(p_vars)
            k_present = gp.quicksum(k_vars)
            order = model.addVar(
                vtype=GRB.BINARY,
                name=f"room_order_{p_idx}_{k_idx}_{h_id}_{day}_{r_id}",
            )
            inactive = max_time * (2 - p_present - k_present)
            model.addConstr(
                room_exit[p_idx]
                <= room_entry[k_idx] + max_time * (1 - order) + inactive
            )
            model.addConstr(
                room_exit[k_idx]
                <= room_entry[p_idx] + max_time * order + inactive
            )

    # Pairwise surgeon non-overlap.
    surgeon_groups = defaultdict(list)
    for (p_idx, s_id, h_id, day), variables in surgeon_presence.items():
        surgeon_groups[(s_id, h_id, day)].append((p_idx, variables))

    for (s_id, h_id, day), members in surgeon_groups.items():
        members.sort(key=lambda item: item[0])
        for (p_idx, p_vars), (k_idx, k_vars) in combinations(members, 2):
            p_present = gp.quicksum(p_vars)
            k_present = gp.quicksum(k_vars)
            order = model.addVar(
                vtype=GRB.BINARY,
                name=f"surgeon_order_{p_idx}_{k_idx}_{s_id}_{h_id}_{day}",
            )
            inactive = max_time * (2 - p_present - k_present)
            model.addConstr(
                finish[p_idx]
                <= surgery_start[k_idx] + max_time * (1 - order) + inactive
            )
            model.addConstr(
                finish[k_idx]
                <= surgery_start[p_idx] + max_time * order + inactive
            )

    # Objective.
    objective = gp.LinExpr()

    for room_key, u in room_open.items():
        rd = room_day_data[room_key]
        objective += rd["fixed_cost"] * u
        objective += rd["overtime_rate"] * overtime[room_key]

    for (s_id, h_id, day), v in surgeon_deployed.items():
        hospital = hospital_by_id[h_id]
        fixed_costs = hospital.get("surgeon_fixed_costs_per_day", {})
        cost = float(key_get(fixed_costs, s_id, 0.0) or 0.0)
        objective += cost * v

    for p_idx, patient in patient_by_index.items():
        if not bool(patient["is_mandatory"]):
            objective -= float(patient.get("reward", 0.0)) * z[p_idx]

    model.setObjective(objective, GRB.MINIMIZE)
    model.update()

    x_keys = list(x.keys())
    x_vars = [x[k] for k in x_keys]
    patient_indices = list(patient_by_index.keys())
    entry_vars = [room_entry[p] for p in patient_indices]
    start_vars = [surgery_start[p] for p in patient_indices]
    finish_vars = [finish[p] for p in patient_indices]
    exit_vars = [room_exit[p] for p in patient_indices]

    def build_solution(x_values, entry_values, start_values, finish_values, exit_values):
        entry_map = dict(zip(patient_indices, entry_values))
        start_map = dict(zip(patient_indices, start_values))
        finish_map = dict(zip(patient_indices, finish_values))
        exit_map = dict(zip(patient_indices, exit_values))

        selected = []
        for key, value in zip(x_keys, x_values):
            if value <= 0.5:
                continue

            p_idx, s_id, h_id, day, r_id = key
            patient = patient_by_index[p_idx]
            surgical = float(key_get(patient["surgeon_specific_times"], s_id, 0.0))
            total_time = (
                float(patient["preparation_time"])
                + surgical
                + float(patient["cleaning_time"])
            )

            selected.append({
                "_p_idx": p_idx,
                "patient_id": int(patient["patient_id"]),
                "surgeon_id": int(s_id),
                "hospital_id": int(h_id),
                "day": int(day),
                "or_id": int(r_id),
                "is_mandatory": bool(patient["is_mandatory"]),
                "T_ps": clean_number(total_time),
                "finish_time": clean_number(finish_map[p_idx]),
                "surgery_start_time": clean_number(start_map[p_idx]),
                "room_entry_time": clean_number(entry_map[p_idx]),
                "room_exit_time": clean_number(exit_map[p_idx]),
            })

        selected.sort(key=lambda a: (a["day"], a["hospital_id"], a["or_id"],
                                     a["room_entry_time"], a["patient_id"]))

        room_cases = defaultdict(list)
        surgeon_cases = defaultdict(list)
        for assignment in selected:
            room_key = (
                assignment["hospital_id"],
                assignment["day"],
                assignment["or_id"],
            )
            surgeon_key = (
                assignment["surgeon_id"],
                assignment["hospital_id"],
                assignment["day"],
            )
            room_cases[room_key].append(assignment)
            surgeon_cases[surgeon_key].append(assignment)

        opened_ors = []
        total_room_cost = 0.0
        total_overtime_cost = 0.0

        for room_key, cases in sorted(room_cases.items()):
            h_id, day, r_id = room_key
            rd = room_day_data[room_key]
            completion = max(case["room_exit_time"] for case in cases)
            overtime_used = max(0.0, completion - rd["regular_time"])
            if overtime_used < 1e-7:
                overtime_used = 0.0

            opened_ors.append({
                "hospital_id": int(h_id),
                "day": int(day),
                "or_id": int(r_id),
                "regular_time": int(rd["regular_time"]),
                "completion_time": clean_number(completion),
                "overtime": clean_number(overtime_used),
            })
            total_room_cost += rd["fixed_cost"]
            total_overtime_cost += rd["overtime_rate"] * overtime_used

        surgeon_assignments = []
        total_surgeon_cost = 0.0

        for surgeon_key, cases in sorted(surgeon_cases.items()):
            s_id, h_id, day = surgeon_key
            actual_start = min(case["surgery_start_time"] for case in cases)
            actual_end = max(case["finish_time"] for case in cases)

            surgeon_assignments.append({
                "surgeon_id": int(s_id),
                "hospital_id": int(h_id),
                "day": int(day),
                "start_time": clean_number(actual_start),
                "end_time": clean_number(actual_end),
            })

            fixed_costs = hospital_by_id[h_id].get("surgeon_fixed_costs_per_day", {})
            total_surgeon_cost += float(key_get(fixed_costs, s_id, 0.0) or 0.0)

        or_sequences = []
        for (h_id, day, r_id), cases in sorted(room_cases.items()):
            by_id = sorted(cases, key=lambda c: c["patient_id"])
            for p_case, k_case in combinations(by_id, 2):
                p_after_k = int(
                    p_case["room_entry_time"] > k_case["room_entry_time"] + 1e-7
                )
                or_sequences.append({
                    "hospital_id": int(h_id),
                    "day": int(day),
                    "or_id": int(r_id),
                    "patient_p": int(p_case["patient_id"]),
                    "patient_k": int(k_case["patient_id"]),
                    "p_after_k": p_after_k,
                })

        surgeon_sequences = []
        for (s_id, h_id, day), cases in sorted(surgeon_cases.items()):
            by_id = sorted(cases, key=lambda c: c["patient_id"])
            for p_case, k_case in combinations(by_id, 2):
                p_after_k = int(
                    p_case["surgery_start_time"]
                    > k_case["surgery_start_time"] + 1e-7
                )
                surgeon_sequences.append({
                    "hospital_id": int(h_id),
                    "day": int(day),
                    "surgeon_id": int(s_id),
                    "patient_p": int(p_case["patient_id"]),
                    "patient_k": int(k_case["patient_id"]),
                    "p_after_k": p_after_k,
                })

        reward = 0.0
        for assignment in selected:
            if not assignment["is_mandatory"]:
                patient = patient_by_index[assignment["_p_idx"]]
                reward += float(patient.get("reward", 0.0))

        objective_value = (
            total_room_cost
            + total_surgeon_cost
            + total_overtime_cost
            - reward
        )

        assignments_output = []
        for assignment in selected:
            assignments_output.append({
                "patient_id": assignment["patient_id"],
                "surgeon_id": assignment["surgeon_id"],
                "hospital_id": assignment["hospital_id"],
                "day": assignment["day"],
                "or_id": assignment["or_id"],
                "is_mandatory": assignment["is_mandatory"],
                "T_ps": assignment["T_ps"],
                "finish_time": assignment["finish_time"],
                "surgery_start_time": assignment["surgery_start_time"],
                "room_entry_time": assignment["room_entry_time"],
                "room_exit_time": assignment["room_exit_time"],
            })

        return {
            "objective_value": clean_number(objective_value),
            "assignments": assignments_output,
            "opened_ors": opened_ors,
            "surgeon_assignments": surgeon_assignments,
            "or_sequences": or_sequences,
            "surgeon_sequences": surgeon_sequences,
        }

    best_logged = [float("inf")]

    def incumbent_callback(cb_model, where):
        if where != GRB.Callback.MIPSOL or logger is None:
            return
        try:
            x_values = cb_model.cbGetSolution(x_vars)
            entry_values = cb_model.cbGetSolution(entry_vars)
            start_values = cb_model.cbGetSolution(start_vars)
            finish_values = cb_model.cbGetSolution(finish_vars)
            exit_values = cb_model.cbGetSolution(exit_vars)

            solution = build_solution(
                x_values,
                entry_values,
                start_values,
                finish_values,
                exit_values,
            )
            value = float(solution["objective_value"])
            if value < best_logged[0] - 1e-7:
                logger.log_solution(value, solution)
                best_logged[0] = value
        except Exception:
            # Logging must not interrupt optimization.
            pass

    model.optimize(incumbent_callback)

    if model.SolCount > 0:
        final_solution = build_solution(
            [var.X for var in x_vars],
            [var.X for var in entry_vars],
            [var.X for var in start_vars],
            [var.X for var in finish_vars],
            [var.X for var in exit_vars],
        )

        if logger and final_solution["objective_value"] < best_logged[0] - 1e-7:
            logger.log_solution(final_solution["objective_value"], final_solution)
    else:
        # This can occur only when the instance is infeasible or no incumbent
        # was found within an extremely short time limit.
        final_solution = {
            "objective_value": 0.0,
            "assignments": [],
            "opened_ors": [],
            "surgeon_assignments": [],
            "or_sequences": [],
            "surgeon_sequences": [],
        }

    solution_dir = os.path.dirname(os.path.abspath(args.solution_path))
    if solution_dir:
        os.makedirs(solution_dir, exist_ok=True)

    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(final_solution, f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
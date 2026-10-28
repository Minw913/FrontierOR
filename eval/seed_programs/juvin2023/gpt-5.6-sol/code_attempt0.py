import argparse
import json
import os
import random
import time

import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def solution_key(job_id, operation_id):
    return f"({job_id},{operation_id})"


def make_solution(objective, assignments, schedules):
    return {
        "objective_value": float(objective),
        "assignment": {
            solution_key(j, o): int(machine)
            for (j, o), machine in assignments.items()
        },
        "schedule": {
            solution_key(j, o): [int(t) for t in sorted(periods)]
            for (j, o), periods in schedules.items()
        },
    }


def parse_instance(data):
    jobs = []
    operations = {}
    job_operations = {}

    for job_data in sorted(data["jobs"], key=lambda item: item["job_id"]):
        job_id = int(job_data["job_id"])
        ordered = sorted(
            job_data.get("operations", []),
            key=lambda item: item["operation_id"],
        )

        op_keys = []
        for op_data in ordered:
            operation_id = int(op_data["operation_id"])
            key = (job_id, operation_id)
            machines = [int(m) for m in op_data["eligible_machines"]]
            processing_times = {
                int(m): int(p)
                for m, p in op_data["processing_times"].items()
            }

            if not machines:
                raise ValueError(f"Operation {key} has no eligible machine")
            for machine in machines:
                if machine not in processing_times:
                    raise ValueError(
                        f"Missing processing time for operation {key}, "
                        f"machine {machine}"
                    )
                if processing_times[machine] <= 0:
                    raise ValueError(
                        f"Processing times must be positive for operation {key}"
                    )

            operations[key] = {
                "machines": machines,
                "processing_times": processing_times,
            }
            op_keys.append(key)

        jobs.append(job_id)
        job_operations[job_id] = op_keys

    return jobs, operations, job_operations


def earliest_contiguous_start(busy_periods, release, duration):
    start = max(1, int(release))
    while True:
        conflict = None
        for period in range(start, start + duration):
            if period in busy_periods:
                conflict = period
                break
        if conflict is None:
            return start
        start = conflict + 1


def greedy_schedule(
    jobs,
    operations,
    job_operations,
    num_machines,
    mode,
    seed,
):
    rng = random.Random(seed)
    machine_busy = [set() for _ in range(num_machines)]
    machine_load = [0] * num_machines
    next_index = {job_id: 0 for job_id in jobs}
    job_ready = {job_id: 1 for job_id in jobs}

    remaining_min = {}
    for job_id in jobs:
        suffix = 0
        for op_key in reversed(job_operations[job_id]):
            suffix += min(operations[op_key]["processing_times"].values())
            remaining_min[op_key] = suffix

    assignments = {}
    schedules = {}
    number_remaining = sum(len(job_operations[j]) for j in jobs)

    while number_remaining:
        candidates = []

        for job_id in jobs:
            index = next_index[job_id]
            if index >= len(job_operations[job_id]):
                continue

            op_key = job_operations[job_id][index]
            op = operations[op_key]
            release = job_ready[job_id]

            for machine in op["machines"]:
                duration = op["processing_times"][machine]
                start = earliest_contiguous_start(
                    machine_busy[machine], release, duration
                )
                finish = start + duration

                if mode == 0:
                    priority = (finish, start, duration, machine_load[machine])
                elif mode == 1:
                    priority = (start, finish, duration, machine_load[machine])
                elif mode == 2:
                    priority = (duration, finish, start, machine_load[machine])
                elif mode == 3:
                    priority = (
                        -remaining_min[op_key],
                        finish,
                        start,
                        duration,
                    )
                elif mode == 4:
                    priority = (
                        machine_load[machine] + duration,
                        finish,
                        start,
                    )
                elif mode == 5:
                    priority = (
                        finish + 0.20 * remaining_min[op_key],
                        finish,
                        start,
                    )
                else:
                    noise = rng.random() * max(1, duration)
                    priority = (
                        finish + noise,
                        rng.random(),
                        start,
                        duration,
                    )

                candidates.append(
                    (priority, job_id, op_key, machine, start, finish)
                )

        if not candidates:
            raise RuntimeError("Unable to construct a feasible schedule")

        _, job_id, op_key, machine, start, finish = min(
            candidates, key=lambda item: item[0]
        )
        duration = operations[op_key]["processing_times"][machine]
        periods = list(range(start, start + duration))

        assignments[op_key] = machine
        schedules[op_key] = periods
        machine_busy[machine].update(periods)
        machine_load[machine] += duration
        job_ready[job_id] = finish
        next_index[job_id] += 1
        number_remaining -= 1

    objective = max(
        (max(periods) + 1 for periods in schedules.values()),
        default=1,
    )
    return make_solution(objective, assignments, schedules)


def estimate_model_size(
    preemption_allowed,
    operations,
    earliest_period,
    latest_period,
    upper_bound,
):
    estimate = 0
    for op_key, op in operations.items():
        earliest = earliest_period[op_key]
        latest = latest_period[op_key]

        for machine in op["machines"]:
            duration = op["processing_times"][machine]
            if preemption_allowed:
                estimate += max(0, latest - earliest + 1)
            else:
                last_start = min(
                    latest - duration + 1,
                    upper_bound - duration,
                )
                estimate += max(0, last_start - earliest + 1)
    return estimate


def solve_with_gurobi(
    data,
    jobs,
    operations,
    job_operations,
    incumbent_solution,
    logger,
    deadline,
):
    num_machines = int(data["num_machines"])
    preemption_allowed = bool(data.get("preemption_allowed", True))
    upper_bound = int(round(incumbent_solution["objective_value"]))
    last_period = upper_bound - 1

    if not operations or last_period < 1:
        return incumbent_solution

    min_duration = {
        op_key: min(op["processing_times"].values())
        for op_key, op in operations.items()
    }

    earliest_period = {}
    latest_period = {}
    for job_id in jobs:
        prefix = 0
        for op_key in job_operations[job_id]:
            earliest_period[op_key] = 1 + prefix
            prefix += min_duration[op_key]

        suffix = 0
        for op_key in reversed(job_operations[job_id]):
            latest_period[op_key] = upper_bound - 1 - suffix
            suffix += min_duration[op_key]

    remaining = deadline - time.monotonic()
    if remaining <= 0.05:
        return incumbent_solution

    estimated_variables = estimate_model_size(
        preemption_allowed,
        operations,
        earliest_period,
        latest_period,
        upper_bound,
    )

    variable_limit = min(
        2_000_000,
        max(250_000, int(250_000 + 30_000 * remaining)),
    )
    if estimated_variables > variable_limit:
        return incumbent_solution

    model = gp.Model("flexible_job_shop")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1

    assignment_vars = {}
    for op_key, op in operations.items():
        for machine in op["machines"]:
            assignment_vars[op_key, machine] = model.addVar(
                vtype=GRB.BINARY,
                name=f"y_{op_key[0]}_{op_key[1]}_{machine}",
            )

    makespan = model.addVar(
        lb=1.0,
        ub=float(upper_bound),
        vtype=GRB.CONTINUOUS,
        name="makespan",
    )

    period_vars = {}
    start_vars = {}

    if preemption_allowed:
        operation_start = {}
        operation_completion = {}

        for op_key in operations:
            operation_start[op_key] = model.addVar(
                lb=float(earliest_period[op_key]),
                ub=float(max(earliest_period[op_key], latest_period[op_key])),
                vtype=GRB.CONTINUOUS,
                name=f"S_{op_key[0]}_{op_key[1]}",
            )
            operation_completion[op_key] = model.addVar(
                lb=float(earliest_period[op_key] + min_duration[op_key]),
                ub=float(upper_bound),
                vtype=GRB.CONTINUOUS,
                name=f"C_{op_key[0]}_{op_key[1]}",
            )

        for op_key, op in operations.items():
            first = earliest_period[op_key]
            last = latest_period[op_key]
            for machine in op["machines"]:
                for period in range(first, last + 1):
                    period_vars[op_key, machine, period] = model.addVar(
                        vtype=GRB.BINARY,
                        name=(
                            f"x_{op_key[0]}_{op_key[1]}_"
                            f"{machine}_{period}"
                        ),
                    )
    else:
        for op_key, op in operations.items():
            first = earliest_period[op_key]
            latest_active = latest_period[op_key]

            for machine in op["machines"]:
                duration = op["processing_times"][machine]
                last_start = min(
                    latest_active - duration + 1,
                    upper_bound - duration,
                )
                for start in range(first, last_start + 1):
                    start_vars[op_key, machine, start] = model.addVar(
                        vtype=GRB.BINARY,
                        name=(
                            f"z_{op_key[0]}_{op_key[1]}_"
                            f"{machine}_{start}"
                        ),
                    )

    model.update()

    for op_key, op in operations.items():
        model.addConstr(
            gp.quicksum(
                assignment_vars[op_key, machine]
                for machine in op["machines"]
            )
            == 1,
            name=f"assign_once_{op_key[0]}_{op_key[1]}",
        )

    if preemption_allowed:
        for op_key, op in operations.items():
            processing_expr = gp.LinExpr()
            required_expr = gp.LinExpr()

            for machine in op["machines"]:
                y = assignment_vars[op_key, machine]
                duration = op["processing_times"][machine]
                required_expr += duration * y

                machine_period_vars = [
                    var
                    for (key, m, _), var in period_vars.items()
                    if key == op_key and m == machine
                ]

                for var in machine_period_vars:
                    model.addConstr(var <= y)

                processing_expr += gp.quicksum(machine_period_vars)

            model.addConstr(processing_expr == required_expr)

        for machine in range(num_machines):
            for period in range(1, last_period + 1):
                active = [
                    var
                    for (op_key, m, t), var in period_vars.items()
                    if m == machine and t == period
                ]
                if active:
                    model.addConstr(gp.quicksum(active) <= 1)

        big_m = float(upper_bound)
        for (op_key, machine, period), var in period_vars.items():
            model.addConstr(
                operation_start[op_key]
                <= period + big_m * (1 - var)
            )
            model.addConstr(
                operation_completion[op_key]
                >= period + 1 - big_m * (1 - var)
            )

        for job_id in jobs:
            sequence = job_operations[job_id]
            for predecessor, successor in zip(sequence, sequence[1:]):
                model.addConstr(
                    operation_completion[predecessor]
                    <= operation_start[successor]
                )
            if sequence:
                model.addConstr(
                    makespan >= operation_completion[sequence[-1]]
                )
    else:
        for op_key, op in operations.items():
            vars_for_operation = [
                var
                for (key, _, _), var in start_vars.items()
                if key == op_key
            ]
            model.addConstr(gp.quicksum(vars_for_operation) == 1)

            for machine in op["machines"]:
                y = assignment_vars[op_key, machine]
                starts = [
                    var
                    for (key, m, _), var in start_vars.items()
                    if key == op_key and m == machine
                ]
                model.addConstr(gp.quicksum(starts) == y)

        for machine in range(num_machines):
            for period in range(1, last_period + 1):
                covering = []
                for (op_key, m, start), var in start_vars.items():
                    if m != machine:
                        continue
                    duration = operations[op_key]["processing_times"][machine]
                    if start <= period < start + duration:
                        covering.append(var)
                if covering:
                    model.addConstr(gp.quicksum(covering) <= 1)

        for job_id in jobs:
            sequence = job_operations[job_id]
            for predecessor, successor in zip(sequence, sequence[1:]):
                predecessor_completion = gp.quicksum(
                    (
                        start
                        + operations[predecessor]["processing_times"][machine]
                    )
                    * var
                    for (key, machine, start), var in start_vars.items()
                    if key == predecessor
                )
                successor_start = gp.quicksum(
                    start * var
                    for (key, _, start), var in start_vars.items()
                    if key == successor
                )
                model.addConstr(predecessor_completion <= successor_start)

            if sequence:
                final_op = sequence[-1]
                final_completion = gp.quicksum(
                    (
                        start
                        + operations[final_op]["processing_times"][machine]
                    )
                    * var
                    for (key, machine, start), var in start_vars.items()
                    if key == final_op
                )
                model.addConstr(makespan >= final_completion)

    for machine in range(num_machines):
        load = gp.quicksum(
            operations[op_key]["processing_times"][machine]
            * assignment_vars[op_key, machine]
            for op_key, op in operations.items()
            if machine in op["machines"]
        )
        if load.size() > 0:
            model.addConstr(makespan >= 1 + load)

    job_lower_bound = 1
    for job_id in jobs:
        job_lower_bound = max(
            job_lower_bound,
            1 + sum(min_duration[op] for op in job_operations[job_id]),
        )
    makespan.LB = float(job_lower_bound)

    model.setObjective(makespan, GRB.MINIMIZE)

    incumbent_assignment = {
        key: int(value)
        for key, value in incumbent_solution["assignment"].items()
    }
    incumbent_schedule = {
        key: [int(t) for t in value]
        for key, value in incumbent_solution["schedule"].items()
    }

    for op_key, op in operations.items():
        text_key = solution_key(*op_key)
        chosen_machine = incumbent_assignment[text_key]
        chosen_periods = incumbent_schedule[text_key]

        for machine in op["machines"]:
            assignment_vars[op_key, machine].Start = (
                1.0 if machine == chosen_machine else 0.0
            )

        if preemption_allowed:
            chosen_set = set(chosen_periods)
            for machine in op["machines"]:
                for period in range(
                    earliest_period[op_key],
                    latest_period[op_key] + 1,
                ):
                    var = period_vars.get((op_key, machine, period))
                    if var is not None:
                        var.Start = (
                            1.0
                            if machine == chosen_machine
                            and period in chosen_set
                            else 0.0
                        )

            operation_start[op_key].Start = float(min(chosen_periods))
            operation_completion[op_key].Start = float(max(chosen_periods) + 1)
        else:
            chosen_start = min(chosen_periods)
            for (key, machine, start), var in start_vars.items():
                if key == op_key:
                    var.Start = (
                        1.0
                        if machine == chosen_machine and start == chosen_start
                        else 0.0
                    )

    makespan.Start = float(upper_bound)

    best_holder = {
        "objective": float(incumbent_solution["objective_value"]),
        "solution": incumbent_solution,
    }

    def extract_solution(value_getter):
        assignments = {}
        schedules = {}

        for op_key, op in operations.items():
            chosen_machine = max(
                op["machines"],
                key=lambda machine: value_getter(
                    assignment_vars[op_key, machine]
                ),
            )
            assignments[op_key] = chosen_machine

            if preemption_allowed:
                periods = [
                    period
                    for (key, machine, period), var in period_vars.items()
                    if key == op_key
                    and machine == chosen_machine
                    and value_getter(var) > 0.5
                ]
            else:
                chosen_start = None
                for (key, machine, start), var in start_vars.items():
                    if (
                        key == op_key
                        and machine == chosen_machine
                        and value_getter(var) > 0.5
                    ):
                        chosen_start = start
                        break

                if chosen_start is None:
                    return None

                duration = operations[op_key]["processing_times"][
                    chosen_machine
                ]
                periods = list(
                    range(chosen_start, chosen_start + duration)
                )

            required = operations[op_key]["processing_times"][chosen_machine]
            if len(periods) != required:
                return None
            schedules[op_key] = sorted(periods)

        objective = max(
            (max(periods) + 1 for periods in schedules.values()),
            default=1,
        )
        return make_solution(objective, assignments, schedules)

    def callback(cb_model, where):
        if where != GRB.Callback.MIPSOL:
            return
        try:
            candidate_obj = cb_model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if candidate_obj >= best_holder["objective"] - 1e-6:
                return

            candidate = extract_solution(cb_model.cbGetSolution)
            if candidate is None:
                return

            actual_obj = candidate["objective_value"]
            if actual_obj < best_holder["objective"] - 1e-6:
                best_holder["objective"] = actual_obj
                best_holder["solution"] = candidate
                if logger:
                    logger.log_solution(actual_obj, candidate)
        except Exception:
            pass

    remaining = deadline - time.monotonic()
    if remaining <= 0.01:
        return incumbent_solution

    model.Params.TimeLimit = max(0.01, remaining)

    try:
        model.optimize(callback)
    except gp.GurobiError:
        return best_holder["solution"]

    if model.SolCount > 0:
        final_candidate = extract_solution(lambda var: var.X)
        if final_candidate is not None:
            final_obj = final_candidate["objective_value"]
            if final_obj < best_holder["objective"] - 1e-6:
                best_holder["objective"] = final_obj
                best_holder["solution"] = final_candidate
                if logger:
                    logger.log_solution(final_obj, final_candidate)

    return best_holder["solution"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    logger = (
        SolutionLogger(args.log_path, sense="minimize")
        if args.log_path
        else None
    )

    start_time = time.monotonic()
    deadline = start_time + max(0, args.time_limit)

    with open(args.instance_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    jobs, operations, job_operations = parse_instance(data)
    num_machines = int(data["num_machines"])

    if not operations:
        best_solution = {
            "objective_value": 1.0,
            "assignment": {},
            "schedule": {},
        }
        if logger:
            logger.log_solution(1.0, best_solution)
    else:
        best_solution = None

        heuristic_modes = [0, 1, 2, 3, 4, 5, 6, 6, 6, 6, 6, 6]
        for run, mode in enumerate(heuristic_modes):
            if run > 0 and time.monotonic() >= deadline:
                break

            candidate = greedy_schedule(
                jobs,
                operations,
                job_operations,
                num_machines,
                mode=mode,
                seed=run,
            )

            if (
                best_solution is None
                or candidate["objective_value"]
                < best_solution["objective_value"] - 1e-9
            ):
                best_solution = candidate
                if logger:
                    logger.log_solution(
                        candidate["objective_value"], candidate
                    )

        if time.monotonic() < deadline:
            try:
                best_solution = solve_with_gurobi(
                    data,
                    jobs,
                    operations,
                    job_operations,
                    best_solution,
                    logger,
                    deadline,
                )
            except (gp.GurobiError, MemoryError):
                pass

    output_directory = os.path.dirname(os.path.abspath(args.solution_path))
    os.makedirs(output_directory, exist_ok=True)
    with open(args.solution_path, "w", encoding="utf-8") as file:
        json.dump(best_solution, file, indent=2)


if __name__ == "__main__":
    main()
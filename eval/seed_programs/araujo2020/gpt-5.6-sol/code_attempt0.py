import argparse
import json
import math
import os
import time
import heapq

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def configure_gurobi(model, time_limit=None):
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    if time_limit is not None:
        model.Params.TimeLimit = max(0.01, float(time_limit))


def topological_order(job_ids, successors, predecessors, priority=None):
    indegree = {j: len(predecessors[j]) for j in job_ids}
    heap = []

    for j in job_ids:
        if indegree[j] == 0:
            key = priority(j) if priority is not None else j
            heapq.heappush(heap, (key, j))

    order = []
    while heap:
        _, j = heapq.heappop(heap)
        order.append(j)
        for s in successors[j]:
            indegree[s] -= 1
            if indegree[s] == 0:
                key = priority(s) if priority is not None else s
                heapq.heappush(heap, (key, s))

    if len(order) != len(job_ids):
        raise ValueError("The precedence graph contains a cycle.")
    return order


def make_solution(job_ids, chosen_modes, starts, sink_by_project, cp, release):
    schedule = [
        {
            "job_id": int(j),
            "mode_id": int(chosen_modes[j]),
            "start_time": int(starts[j]),
        }
        for j in sorted(job_ids)
    ]

    completion = {
        p: int(starts[sink_by_project[p]])
        for p in sink_by_project
    }
    delay = sum(completion[p] - release[p] - cp[p] for p in completion)
    makespan = max(completion.values()) if completion else 0

    return {
        "objective_value": float(delay),
        "schedule": schedule,
        "makespan": int(makespan),
    }


def solve_mode_assignment(
    job_ids,
    modes_by_job,
    renewable_capacities,
    nonrenewable_capacities,
    time_limit,
):
    model = gp.Model("mode_assignment")
    configure_gurobi(model, time_limit)

    y = {}
    valid_modes = {}

    for j in job_ids:
        valid_modes[j] = []
        for m in modes_by_job[j]:
            renewable_ok = all(
                m["renewable_consumption"][r] <= renewable_capacities[r]
                for r in range(len(renewable_capacities))
            )
            if renewable_ok:
                valid_modes[j].append(m)
                y[j, m["mode_id"]] = model.addVar(vtype=GRB.BINARY)

        if not valid_modes[j]:
            raise ValueError(f"Job {j} has no mode compatible with renewable capacities.")

    model.update()

    for j in job_ids:
        model.addConstr(
            gp.quicksum(y[j, m["mode_id"]] for m in valid_modes[j]) == 1
        )

    for r, capacity in enumerate(nonrenewable_capacities):
        model.addConstr(
            gp.quicksum(
                m["nonrenewable_consumption"][r] * y[j, m["mode_id"]]
                for j in job_ids
                for m in valid_modes[j]
            )
            <= capacity
        )

    # Primarily prefer short modes; the tiny second term weakly favors lower
    # nonrenewable use without changing integer duration priorities.
    total_nr_capacity = max(1, sum(nonrenewable_capacities))
    model.setObjective(
        gp.quicksum(
            (
                m["duration"]
                + 1e-6
                * sum(m["nonrenewable_consumption"])
                / total_nr_capacity
            )
            * y[j, m["mode_id"]]
            for j in job_ids
            for m in valid_modes[j]
        ),
        GRB.MINIMIZE,
    )
    model.optimize()

    if model.SolCount == 0:
        return None, valid_modes

    chosen = {}
    for j in job_ids:
        chosen[j] = max(
            valid_modes[j],
            key=lambda m: y[j, m["mode_id"]].X,
        )["mode_id"]

    return chosen, valid_modes


def build_serial_sgs(
    job_ids,
    jobs,
    successors,
    predecessors,
    chosen_modes,
    mode_lookup,
    project_release,
    project_cp,
    tail_min,
    renewable_capacities,
):
    def priority(j):
        p = jobs[j]["project_id"]
        urgency = project_release[p] + project_cp[p] - tail_min[j]
        return urgency, project_release[p], j

    order = topological_order(job_ids, successors, predecessors, priority)

    num_renewable = len(renewable_capacities)
    usage = [[] for _ in range(num_renewable)]
    starts = {}
    finishes = {}

    def ensure_profile_length(length):
        for r in range(num_renewable):
            if len(usage[r]) < length:
                usage[r].extend([0] * (length - len(usage[r])))

    for j in order:
        p = jobs[j]["project_id"]
        m = mode_lookup[j][chosen_modes[j]]
        duration = int(m["duration"])
        consumption = m["renewable_consumption"]

        earliest = project_release[p]
        if predecessors[j]:
            earliest = max(earliest, max(finishes[q] for q in predecessors[j]))

        if duration == 0:
            starts[j] = int(earliest)
            finishes[j] = int(earliest)
            continue

        t = int(earliest)
        while True:
            ensure_profile_length(t + duration)
            feasible = True
            jump_to = None

            for tau in range(t, t + duration):
                for r in range(num_renewable):
                    if usage[r][tau] + consumption[r] > renewable_capacities[r]:
                        feasible = False
                        jump_to = tau + 1
                        break
                if not feasible:
                    break

            if feasible:
                break
            t = max(t + 1, jump_to)

        starts[j] = t
        finishes[j] = t + duration
        for tau in range(t, t + duration):
            for r in range(num_renewable):
                usage[r][tau] += consumption[r]

    return starts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    start_clock = time.monotonic()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as f:
        instance = json.load(f)

    projects = {int(p["project_id"]): p for p in instance["projects"]}
    jobs = {int(j["job_id"]): j for j in instance["jobs"]}
    job_ids = list(jobs.keys())

    release = {
        p: int(projects[p]["release_date"])
        for p in projects
    }
    sink_by_project = {
        p: int(projects[p]["artificial_sink_job_id"])
        for p in projects
    }

    renewable_resources = instance["resources"].get("renewable", [])
    nonrenewable_resources = instance["resources"].get("nonrenewable", [])
    renewable_capacities = [int(r["capacity"]) for r in renewable_resources]
    nonrenewable_capacities = [int(r["capacity"]) for r in nonrenewable_resources]

    successors = {
        j: [int(s) for s in jobs[j].get("successors", [])]
        for j in job_ids
    }
    predecessors = {j: [] for j in job_ids}
    for j in job_ids:
        for s in successors[j]:
            if s not in predecessors:
                raise ValueError(f"Unknown successor job ID {s}.")
            predecessors[s].append(j)

    modes_by_job = {}
    mode_lookup = {}
    min_duration = {}

    for j in job_ids:
        modes = []
        lookup = {}
        for raw_mode in jobs[j]["modes"]:
            mode = {
                "mode_id": int(raw_mode["mode_id"]),
                "duration": int(raw_mode["duration"]),
                "renewable_consumption": [
                    int(v) for v in raw_mode["renewable_consumption"]
                ],
                "nonrenewable_consumption": [
                    int(v) for v in raw_mode["nonrenewable_consumption"]
                ],
            }
            modes.append(mode)
            lookup[mode["mode_id"]] = mode

        if not modes:
            raise ValueError(f"Job {j} has no execution mode.")

        modes_by_job[j] = modes
        mode_lookup[j] = lookup
        min_duration[j] = min(m["duration"] for m in modes)

    global_order = topological_order(job_ids, successors, predecessors)

    # Earliest starts under shortest job durations and release dates.
    earliest_start = {}
    for j in global_order:
        p = int(jobs[j]["project_id"])
        value = release[p]
        if predecessors[j]:
            value = max(
                value,
                max(earliest_start[q] + min_duration[q] for q in predecessors[j]),
            )
        earliest_start[j] = int(value)

    # Longest minimum-duration tail, including the current job.
    tail_min = {}
    for j in reversed(global_order):
        same_project_successors = [
            s for s in successors[j]
            if jobs[s]["project_id"] == jobs[j]["project_id"]
        ]
        after = max((tail_min[s] for s in same_project_successors), default=0)
        tail_min[j] = min_duration[j] + after

    cp = {}
    for p, sink in sink_by_project.items():
        cp[p] = max(0, int(earliest_start[sink] - release[p]))

    elapsed = time.monotonic() - start_clock
    remaining = max(0.01, args.time_limit - elapsed)
    mode_time = min(5.0, max(0.1, 0.15 * remaining))

    chosen_modes, valid_modes = solve_mode_assignment(
        job_ids,
        modes_by_job,
        renewable_capacities,
        nonrenewable_capacities,
        mode_time,
    )

    if chosen_modes is None:
        # Give the mode-feasibility model the remaining time if its initial
        # short run did not find an incumbent.
        elapsed = time.monotonic() - start_clock
        remaining = max(0.01, args.time_limit - elapsed)
        chosen_modes, valid_modes = solve_mode_assignment(
            job_ids,
            modes_by_job,
            renewable_capacities,
            nonrenewable_capacities,
            remaining,
        )

    if chosen_modes is None:
        raise RuntimeError("No feasible mode assignment was found.")

    initial_starts = build_serial_sgs(
        job_ids,
        jobs,
        successors,
        predecessors,
        chosen_modes,
        mode_lookup,
        release,
        cp,
        tail_min,
        renewable_capacities,
    )

    best_solution = make_solution(
        job_ids,
        chosen_modes,
        initial_starts,
        sink_by_project,
        cp,
        release,
    )
    best_combined = None

    if logger:
        logger.log_solution(best_solution["objective_value"], best_solution)

    initial_delay = int(round(best_solution["objective_value"]))
    max_total_delay = max(0, initial_delay)

    theoretical_horizon = max(
        (
            release[p] + cp[p] + max_total_delay
            for p in projects
        ),
        default=0,
    )
    incumbent_end = max(
        (
            initial_starts[j]
            + mode_lookup[j][chosen_modes[j]]["duration"]
            for j in job_ids
        ),
        default=0,
    )
    horizon = max(theoretical_horizon, incumbent_end)

    elapsed = time.monotonic() - start_clock
    remaining = args.time_limit - elapsed

    if remaining > 0.05:
        model = gp.Model("multi_project_multimode_rcpsp")
        configure_gurobi(model, remaining)

        # Limit the time-indexed model size on unusually large horizons while
        # always retaining the complete initial incumbent.
        total_valid_modes = sum(len(valid_modes[j]) for j in job_ids)
        target_variables = min(
            500000,
            max(100000, int(100000 + 20000 * max(0.0, remaining))),
        )

        full_count = 0
        full_latest = {}
        for j in job_ids:
            p = int(jobs[j]["project_id"])
            same_project_successors = [
                s for s in successors[j]
                if jobs[s]["project_id"] == p
            ]
            remaining_after = max(
                (tail_min[s] for s in same_project_successors),
                default=0,
            )
            for m in valid_modes[j]:
                latest = (
                    release[p]
                    + cp[p]
                    + max_total_delay
                    - m["duration"]
                    - remaining_after
                )
                full_latest[j, m["mode_id"]] = int(latest)
                if latest >= earliest_start[j]:
                    full_count += latest - earliest_start[j] + 1

        if full_count > target_variables:
            window_slack = max(
                5,
                target_variables // max(1, total_valid_modes),
            )
        else:
            window_slack = None

        x = {}
        choices_by_job = {j: [] for j in job_ids}
        start_expr = {}
        duration_expr = {}

        for j in job_ids:
            es = int(earliest_start[j])
            for m in valid_modes[j]:
                mid = m["mode_id"]
                latest = full_latest[j, mid]

                if window_slack is not None:
                    latest = min(latest, es + window_slack)

                if mid == chosen_modes[j]:
                    latest = max(latest, initial_starts[j])

                if latest < es:
                    continue

                for t in range(es, latest + 1):
                    var = model.addVar(vtype=GRB.BINARY)
                    x[j, mid, t] = var
                    choices_by_job[j].append((m, t, var))
                    var.Start = 1.0 if (
                        mid == chosen_modes[j] and t == initial_starts[j]
                    ) else 0.0

            if not choices_by_job[j]:
                raise RuntimeError(f"No time-indexed choice remains for job {j}.")

        model.update()

        for j in job_ids:
            model.addConstr(
                gp.quicksum(var for _, _, var in choices_by_job[j]) == 1
            )
            start_expr[j] = gp.quicksum(
                t * var for _, t, var in choices_by_job[j]
            )
            duration_expr[j] = gp.quicksum(
                m["duration"] * var
                for m, _, var in choices_by_job[j]
            )

        for j in job_ids:
            p = int(jobs[j]["project_id"])
            model.addConstr(start_expr[j] >= release[p])
            for s in successors[j]:
                model.addConstr(
                    start_expr[s] >= start_expr[j] + duration_expr[j]
                )

        # Renewable cumulative constraints.
        active = {}
        for j in job_ids:
            for m, t, var in choices_by_job[j]:
                duration = m["duration"]
                if duration <= 0:
                    continue
                for r, amount in enumerate(m["renewable_consumption"]):
                    if amount == 0:
                        continue
                    for tau in range(t, t + duration):
                        active.setdefault((r, tau), []).append((amount, var))

        for (r, tau), terms in active.items():
            model.addConstr(
                gp.quicksum(amount * var for amount, var in terms)
                <= renewable_capacities[r]
            )

        # Nonrenewable capacities.
        for r, capacity in enumerate(nonrenewable_capacities):
            model.addConstr(
                gp.quicksum(
                    m["nonrenewable_consumption"][r] * var
                    for j in job_ids
                    for m, _, var in choices_by_job[j]
                )
                <= capacity
            )

        makespan_var = model.addVar(
            lb=0.0,
            ub=max(1, horizon),
            vtype=GRB.INTEGER,
        )
        makespan_var.Start = best_solution["makespan"]

        for p, sink in sink_by_project.items():
            model.addConstr(makespan_var >= start_expr[sink])

        delay_expression = gp.quicksum(
            start_expr[sink_by_project[p]] - release[p] - cp[p]
            for p in projects
        )

        epsilon = 1.0 / (max(1, horizon) + 2.0)
        model.setObjective(
            delay_expression + epsilon * makespan_var,
            GRB.MINIMIZE,
        )

        best_combined = (
            best_solution["objective_value"]
            + epsilon * best_solution["makespan"]
        )

        def incumbent_callback(cb_model, where):
            nonlocal best_solution, best_combined
            if where != GRB.Callback.MIPSOL:
                return

            try:
                incumbent_modes = {}
                incumbent_starts = {}

                for j in job_ids:
                    selected = None
                    best_value = -1.0
                    for m, t, var in choices_by_job[j]:
                        value = cb_model.cbGetSolution(var)
                        if value > best_value:
                            best_value = value
                            selected = (m["mode_id"], t)

                    if selected is None:
                        return
                    incumbent_modes[j] = selected[0]
                    incumbent_starts[j] = selected[1]

                candidate = make_solution(
                    job_ids,
                    incumbent_modes,
                    incumbent_starts,
                    sink_by_project,
                    cp,
                    release,
                )
                combined = (
                    candidate["objective_value"]
                    + epsilon * candidate["makespan"]
                )

                if best_combined is None or combined < best_combined - 1e-7:
                    best_combined = combined
                    best_solution = candidate
                    if logger:
                        logger.log_solution(
                            candidate["objective_value"],
                            candidate,
                        )
            except Exception:
                # Logging or callback extraction must not terminate the solve.
                return

        model.optimize(incumbent_callback)

        if model.SolCount > 0:
            final_modes = {}
            final_starts = {}

            for j in job_ids:
                selected = max(
                    choices_by_job[j],
                    key=lambda item: item[2].X,
                )
                final_modes[j] = selected[0]["mode_id"]
                final_starts[j] = selected[1]

            candidate = make_solution(
                job_ids,
                final_modes,
                final_starts,
                sink_by_project,
                cp,
                release,
            )
            candidate_combined = (
                candidate["objective_value"]
                + epsilon * candidate["makespan"]
            )

            if best_combined is None or candidate_combined < best_combined - 1e-7:
                best_solution = candidate
                best_combined = candidate_combined
                if logger:
                    logger.log_solution(
                        candidate["objective_value"],
                        candidate,
                    )

    output_dir = os.path.dirname(os.path.abspath(args.solution_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(best_solution, f, indent=2)


if __name__ == "__main__":
    main()
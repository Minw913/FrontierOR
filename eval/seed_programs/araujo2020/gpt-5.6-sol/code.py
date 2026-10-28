import argparse
import heapq
import json
import os
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def configure_gurobi(model, time_limit):
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = max(0.01, float(time_limit))


def topological_order(job_ids, successors, predecessors, priority=None):
    indegree = {j: len(predecessors[j]) for j in job_ids}
    queue = []

    for j in job_ids:
        if indegree[j] == 0:
            key = priority(j) if priority else j
            heapq.heappush(queue, (key, j))

    order = []
    while queue:
        _, j = heapq.heappop(queue)
        order.append(j)

        for successor in successors[j]:
            indegree[successor] -= 1
            if indegree[successor] == 0:
                key = priority(successor) if priority else successor
                heapq.heappush(queue, (key, successor))

    if len(order) != len(job_ids):
        raise ValueError("The precedence graph contains a cycle.")

    return order


def normalize_vector(values, length):
    if values is None:
        values = []
    result = [int(v) for v in values]
    if len(result) < length:
        result.extend([0] * (length - len(result)))
    return result[:length]


def mode_assignment_is_feasible(
    job_ids,
    chosen_modes,
    mode_lookup,
    renewable_capacities,
    nonrenewable_capacities,
):
    if chosen_modes is None or len(chosen_modes) != len(job_ids):
        return False

    nonrenewable_use = [0] * len(nonrenewable_capacities)

    for j in job_ids:
        if j not in chosen_modes:
            return False
        mode = mode_lookup[j].get(chosen_modes[j])
        if mode is None:
            return False

        for r, capacity in enumerate(renewable_capacities):
            if mode["renewable_consumption"][r] > capacity:
                return False

        for r in range(len(nonrenewable_capacities)):
            nonrenewable_use[r] += mode["nonrenewable_consumption"][r]

    return all(
        nonrenewable_use[r] <= nonrenewable_capacities[r]
        for r in range(len(nonrenewable_capacities))
    )


def heuristic_mode_assignment(
    job_ids,
    valid_modes,
    nonrenewable_capacities,
):
    chosen = {
        j: min(
            valid_modes[j],
            key=lambda m: (
                m["duration"],
                sum(m["nonrenewable_consumption"]),
                m["mode_id"],
            ),
        )["mode_id"]
        for j in job_ids
    }

    mode_by_id = {
        j: {m["mode_id"]: m for m in valid_modes[j]}
        for j in job_ids
    }

    def resource_use():
        result = [0] * len(nonrenewable_capacities)
        for job in job_ids:
            mode = mode_by_id[job][chosen[job]]
            for resource in range(len(result)):
                result[resource] += mode["nonrenewable_consumption"][resource]
        return result

    # Greedily change modes until all nonrenewable violations are removed.
    for _ in range(max(1, len(job_ids) * 10)):
        use = resource_use()
        violation = [
            max(0, use[r] - nonrenewable_capacities[r])
            for r in range(len(nonrenewable_capacities))
        ]

        if not any(violation):
            return chosen

        best_change = None
        best_key = None

        for j in job_ids:
            old_mode = mode_by_id[j][chosen[j]]
            for new_mode in valid_modes[j]:
                if new_mode["mode_id"] == old_mode["mode_id"]:
                    continue

                reduction = 0.0
                total_reduction = 0
                for r in range(len(nonrenewable_capacities)):
                    delta = (
                        old_mode["nonrenewable_consumption"][r]
                        - new_mode["nonrenewable_consumption"][r]
                    )
                    total_reduction += max(0, delta)
                    if violation[r] > 0 and delta > 0:
                        reduction += min(violation[r], delta) / max(
                            1, nonrenewable_capacities[r]
                        )

                if reduction <= 0:
                    continue

                duration_increase = max(
                    0, new_mode["duration"] - old_mode["duration"]
                )
                key = (
                    reduction / (1.0 + duration_increase),
                    reduction,
                    total_reduction,
                    -duration_increase,
                    -new_mode["duration"],
                )

                if best_key is None or key > best_key:
                    best_key = key
                    best_change = (j, new_mode["mode_id"])

        if best_change is None:
            return None

        chosen[best_change[0]] = best_change[1]

    return None


def gurobi_mode_assignment(
    job_ids,
    valid_modes,
    nonrenewable_capacities,
    time_limit,
):
    model = gp.Model("mode_assignment")
    configure_gurobi(model, time_limit)

    y = {}
    for j in job_ids:
        for mode in valid_modes[j]:
            y[j, mode["mode_id"]] = model.addVar(vtype=GRB.BINARY)

    for j in job_ids:
        model.addConstr(
            gp.quicksum(y[j, mode["mode_id"]] for mode in valid_modes[j]) == 1
        )

    for r, capacity in enumerate(nonrenewable_capacities):
        model.addConstr(
            gp.quicksum(
                mode["nonrenewable_consumption"][r] * y[j, mode["mode_id"]]
                for j in job_ids
                for mode in valid_modes[j]
            )
            <= capacity
        )

    model.setObjective(
        gp.quicksum(
            (
                1000000 * mode["duration"]
                + sum(mode["nonrenewable_consumption"])
            )
            * y[j, mode["mode_id"]]
            for j in job_ids
            for mode in valid_modes[j]
        ),
        GRB.MINIMIZE,
    )

    model.optimize()

    if model.SolCount == 0:
        return None

    chosen = {}
    for j in job_ids:
        selected = max(
            valid_modes[j],
            key=lambda mode: y[j, mode["mode_id"]].X,
        )
        chosen[j] = selected["mode_id"]

    return chosen


def depth_first_mode_assignment(
    job_ids,
    valid_modes,
    nonrenewable_capacities,
    deadline,
):
    num_resources = len(nonrenewable_capacities)

    if num_resources == 0:
        return {
            j: min(
                valid_modes[j],
                key=lambda m: (m["duration"], m["mode_id"]),
            )["mode_id"]
            for j in job_ids
        }

    ordered_jobs = sorted(
        job_ids,
        key=lambda j: (
            -len(valid_modes[j]),
            -max(
                (
                    sum(m["nonrenewable_consumption"])
                    for m in valid_modes[j]
                ),
                default=0,
            ),
            j,
        ),
    )

    ordered_modes = {}
    for j in ordered_jobs:
        ordered_modes[j] = sorted(
            valid_modes[j],
            key=lambda m: (
                m["duration"],
                sum(m["nonrenewable_consumption"]),
                m["mode_id"],
            ),
        )

    suffix_minimum = [
        [0] * num_resources for _ in range(len(ordered_jobs) + 1)
    ]
    for index in range(len(ordered_jobs) - 1, -1, -1):
        j = ordered_jobs[index]
        for r in range(num_resources):
            suffix_minimum[index][r] = (
                suffix_minimum[index + 1][r]
                + min(m["nonrenewable_consumption"][r] for m in ordered_modes[j])
            )

    current = {}
    use = [0] * num_resources
    result = None

    def search(index):
        nonlocal result

        if result is not None:
            return True
        if time.monotonic() >= deadline:
            return False

        for r in range(num_resources):
            if (
                use[r] + suffix_minimum[index][r]
                > nonrenewable_capacities[r]
            ):
                return False

        if index == len(ordered_jobs):
            result = dict(current)
            return True

        j = ordered_jobs[index]

        for mode in ordered_modes[j]:
            feasible = True
            for r in range(num_resources):
                if (
                    use[r] + mode["nonrenewable_consumption"][r]
                    > nonrenewable_capacities[r]
                ):
                    feasible = False
                    break

            if not feasible:
                continue

            current[j] = mode["mode_id"]
            for r in range(num_resources):
                use[r] += mode["nonrenewable_consumption"][r]

            if search(index + 1):
                return True

            for r in range(num_resources):
                use[r] -= mode["nonrenewable_consumption"][r]
            del current[j]

        return False

    search(0)
    return result


def find_mode_assignment(
    job_ids,
    valid_modes,
    mode_lookup,
    renewable_capacities,
    nonrenewable_capacities,
    time_limit,
):
    chosen = heuristic_mode_assignment(
        job_ids,
        valid_modes,
        nonrenewable_capacities,
    )

    if mode_assignment_is_feasible(
        job_ids,
        chosen,
        mode_lookup,
        renewable_capacities,
        nonrenewable_capacities,
    ):
        return chosen

    start = time.monotonic()

    try:
        chosen = gurobi_mode_assignment(
            job_ids,
            valid_modes,
            nonrenewable_capacities,
            min(max(0.05, time_limit), 10.0),
        )
    except (gp.GurobiError, ValueError, TypeError, IndexError):
        chosen = None

    if mode_assignment_is_feasible(
        job_ids,
        chosen,
        mode_lookup,
        renewable_capacities,
        nonrenewable_capacities,
    ):
        return chosen

    remaining = max(0.01, time_limit - (time.monotonic() - start))
    chosen = depth_first_mode_assignment(
        job_ids,
        valid_modes,
        nonrenewable_capacities,
        time.monotonic() + remaining,
    )

    if mode_assignment_is_feasible(
        job_ids,
        chosen,
        mode_lookup,
        renewable_capacities,
        nonrenewable_capacities,
    ):
        return chosen

    return None


def serial_schedule_generation(
    job_ids,
    jobs,
    successors,
    predecessors,
    chosen_modes,
    mode_lookup,
    project_release,
    project_cp,
    minimum_tail,
    renewable_capacities,
):
    def priority(j):
        project = jobs[j]["project_id"]
        latest_critical_start = (
            project_release[project]
            + project_cp[project]
            - minimum_tail[j]
        )
        return latest_critical_start, project_release[project], j

    order = topological_order(
        job_ids,
        successors,
        predecessors,
        priority,
    )

    profiles = [[] for _ in renewable_capacities]
    starts = {}
    finishes = {}

    def extend_profiles(length):
        for r in range(len(profiles)):
            if len(profiles[r]) < length:
                profiles[r].extend([0] * (length - len(profiles[r])))

    for j in order:
        project = jobs[j]["project_id"]
        mode = mode_lookup[j][chosen_modes[j]]
        duration = mode["duration"]

        earliest = project_release[project]
        if predecessors[j]:
            earliest = max(
                earliest,
                max(finishes[pred] for pred in predecessors[j]),
            )

        if duration == 0:
            starts[j] = int(earliest)
            finishes[j] = int(earliest)
            continue

        t = int(earliest)
        while True:
            extend_profiles(t + duration)
            feasible = True
            next_t = t + 1

            for period in range(t, t + duration):
                period_feasible = True
                for r, capacity in enumerate(renewable_capacities):
                    if (
                        profiles[r][period]
                        + mode["renewable_consumption"][r]
                        > capacity
                    ):
                        period_feasible = False
                        break

                if not period_feasible:
                    feasible = False
                    next_t = period + 1
                    break

            if feasible:
                break
            t = next_t

        starts[j] = t
        finishes[j] = t + duration

        for period in range(t, t + duration):
            for r in range(len(renewable_capacities)):
                profiles[r][period] += mode["renewable_consumption"][r]

    return starts


def build_solution(
    job_ids,
    chosen_modes,
    starts,
    sink_by_project,
    release,
    critical_path,
):
    completion_times = {
        project: int(starts[sink])
        for project, sink in sink_by_project.items()
    }

    objective = sum(
        completion_times[p] - release[p] - critical_path[p]
        for p in completion_times
    )

    makespan = max(completion_times.values(), default=0)

    return {
        "objective_value": float(objective),
        "schedule": [
            {
                "job_id": int(j),
                "mode_id": int(chosen_modes[j]),
                "start_time": int(starts[j]),
            }
            for j in sorted(job_ids)
        ],
        "makespan": int(makespan),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    program_start = time.monotonic()
    logger = (
        SolutionLogger(args.log_path, sense="minimize")
        if args.log_path
        else None
    )

    with open(args.instance_path, "r", encoding="utf-8") as file:
        instance = json.load(file)

    project_data = {
        int(project["project_id"]): project
        for project in instance.get("projects", [])
    }
    job_data = {
        int(job["job_id"]): job
        for job in instance.get("jobs", [])
    }

    if not project_data:
        raise ValueError("The instance contains no projects.")
    if not job_data:
        raise ValueError("The instance contains no jobs.")

    job_ids = sorted(job_data)
    project_ids = sorted(project_data)

    renewable_resources = instance.get("resources", {}).get("renewable", [])
    nonrenewable_resources = instance.get("resources", {}).get(
        "nonrenewable", []
    )

    renewable_capacities = [
        int(resource["capacity"]) for resource in renewable_resources
    ]
    nonrenewable_capacities = [
        int(resource["capacity"]) for resource in nonrenewable_resources
    ]

    num_renewable = len(renewable_capacities)
    num_nonrenewable = len(nonrenewable_capacities)

    release = {
        p: int(project_data[p]["release_date"])
        for p in project_ids
    }
    sink_by_project = {
        p: int(project_data[p]["artificial_sink_job_id"])
        for p in project_ids
    }

    jobs = {}
    successors = {}
    predecessors = {j: [] for j in job_ids}
    modes_by_job = {}
    mode_lookup = {}

    for j in job_ids:
        raw_job = job_data[j]
        project = int(raw_job["project_id"])

        if project not in project_data:
            raise ValueError(f"Job {j} refers to unknown project {project}.")

        jobs[j] = {
            "project_id": project,
            "type": raw_job.get("type", "real"),
        }

        successors[j] = [int(s) for s in raw_job.get("successors", [])]

        modes = []
        lookup = {}
        for raw_mode in raw_job.get("modes", []):
            mode = {
                "mode_id": int(raw_mode["mode_id"]),
                "duration": int(raw_mode["duration"]),
                "renewable_consumption": normalize_vector(
                    raw_mode.get("renewable_consumption", []),
                    num_renewable,
                ),
                "nonrenewable_consumption": normalize_vector(
                    raw_mode.get("nonrenewable_consumption", []),
                    num_nonrenewable,
                ),
            }
            modes.append(mode)
            lookup[mode["mode_id"]] = mode

        if not modes:
            raise ValueError(f"Job {j} has no mode.")

        modes_by_job[j] = modes
        mode_lookup[j] = lookup

    for j in job_ids:
        for successor in successors[j]:
            if successor not in predecessors:
                raise ValueError(
                    f"Job {j} refers to unknown successor {successor}."
                )
            predecessors[successor].append(j)

    for project, sink in sink_by_project.items():
        if sink not in job_data:
            raise ValueError(
                f"Project {project} refers to unknown sink job {sink}."
            )

    order = topological_order(job_ids, successors, predecessors)

    minimum_duration = {
        j: min(mode["duration"] for mode in modes_by_job[j])
        for j in job_ids
    }

    earliest_start = {}
    for j in order:
        project = jobs[j]["project_id"]
        earliest = release[project]
        if predecessors[j]:
            earliest = max(
                earliest,
                max(
                    earliest_start[pred] + minimum_duration[pred]
                    for pred in predecessors[j]
                ),
            )
        earliest_start[j] = int(earliest)

    critical_path = {
        p: max(0, earliest_start[sink_by_project[p]] - release[p])
        for p in project_ids
    }

    minimum_tail = {}
    for j in reversed(order):
        same_project_successors = [
            successor
            for successor in successors[j]
            if jobs[successor]["project_id"] == jobs[j]["project_id"]
        ]
        minimum_tail[j] = minimum_duration[j] + max(
            (minimum_tail[s] for s in same_project_successors),
            default=0,
        )

    valid_modes = {}
    for j in job_ids:
        valid_modes[j] = [
            mode
            for mode in modes_by_job[j]
            if all(
                mode["renewable_consumption"][r]
                <= renewable_capacities[r]
                for r in range(num_renewable)
            )
            and all(
                mode["nonrenewable_consumption"][r]
                <= nonrenewable_capacities[r]
                for r in range(num_nonrenewable)
            )
        ]

        if not valid_modes[j]:
            raise RuntimeError(
                f"Job {j} has no individually resource-feasible mode."
            )

    remaining = max(
        0.05,
        args.time_limit - (time.monotonic() - program_start),
    )
    mode_budget = min(remaining, max(1.0, 0.2 * args.time_limit))

    chosen_modes = find_mode_assignment(
        job_ids,
        valid_modes,
        mode_lookup,
        renewable_capacities,
        nonrenewable_capacities,
        mode_budget,
    )

    if chosen_modes is None:
        raise RuntimeError("No feasible mode assignment could be found.")

    initial_starts = serial_schedule_generation(
        job_ids,
        jobs,
        successors,
        predecessors,
        chosen_modes,
        mode_lookup,
        release,
        critical_path,
        minimum_tail,
        renewable_capacities,
    )

    best_solution = build_solution(
        job_ids,
        chosen_modes,
        initial_starts,
        sink_by_project,
        release,
        critical_path,
    )

    if logger:
        logger.log_solution(
            best_solution["objective_value"],
            best_solution,
        )

    remaining = args.time_limit - (time.monotonic() - program_start)

    if remaining > 0.1:
        try:
            model = gp.Model("time_indexed_project_scheduling")
            configure_gurobi(model, remaining)

            maximum_total_delay = max(
                0, int(round(best_solution["objective_value"]))
            )
            incumbent_horizon = max(
                initial_starts[j]
                + mode_lookup[j][chosen_modes[j]]["duration"]
                for j in job_ids
            )
            horizon = max(
                incumbent_horizon,
                max(
                    release[p]
                    + critical_path[p]
                    + maximum_total_delay
                    for p in project_ids
                ),
            )

            total_mode_count = sum(len(valid_modes[j]) for j in job_ids)
            maximum_variables = min(
                400000,
                max(50000, int(50000 + 10000 * remaining)),
            )
            window_limit = max(
                5,
                maximum_variables // max(1, total_mode_count),
            )

            choices = {j: [] for j in job_ids}
            x = {}

            for j in job_ids:
                project = jobs[j]["project_id"]

                successor_tail = max(
                    (
                        minimum_tail[s]
                        for s in successors[j]
                        if jobs[s]["project_id"] == project
                    ),
                    default=0,
                )

                for mode in valid_modes[j]:
                    mode_id = mode["mode_id"]
                    earliest = earliest_start[j]
                    latest = (
                        release[project]
                        + critical_path[project]
                        + maximum_total_delay
                        - mode["duration"]
                        - successor_tail
                    )

                    latest = min(latest, earliest + window_limit)

                    if mode_id == chosen_modes[j]:
                        latest = max(latest, initial_starts[j])

                    if latest < earliest:
                        continue

                    for start in range(earliest, latest + 1):
                        variable = model.addVar(
                            vtype=GRB.BINARY,
                            name=f"x_{j}_{mode_id}_{start}",
                        )
                        variable.Start = (
                            1.0
                            if mode_id == chosen_modes[j]
                            and start == initial_starts[j]
                            else 0.0
                        )
                        x[j, mode_id, start] = variable
                        choices[j].append((mode, start, variable))

                if not choices[j]:
                    mode = mode_lookup[j][chosen_modes[j]]
                    start = initial_starts[j]
                    variable = model.addVar(
                        vtype=GRB.BINARY,
                        name=f"x_{j}_{mode['mode_id']}_{start}",
                    )
                    variable.Start = 1.0
                    x[j, mode["mode_id"], start] = variable
                    choices[j].append((mode, start, variable))

            model.update()

            start_expression = {}
            duration_expression = {}

            for j in job_ids:
                model.addConstr(
                    gp.quicksum(
                        variable
                        for _, _, variable in choices[j]
                    )
                    == 1
                )

                start_expression[j] = gp.quicksum(
                    start * variable
                    for _, start, variable in choices[j]
                )
                duration_expression[j] = gp.quicksum(
                    mode["duration"] * variable
                    for mode, _, variable in choices[j]
                )

            for j in job_ids:
                project = jobs[j]["project_id"]
                model.addConstr(start_expression[j] >= release[project])

                for successor in successors[j]:
                    model.addConstr(
                        start_expression[successor]
                        >= start_expression[j] + duration_expression[j]
                    )

            active_terms = {}
            for j in job_ids:
                for mode, start, variable in choices[j]:
                    for r in range(num_renewable):
                        amount = mode["renewable_consumption"][r]
                        if amount == 0:
                            continue
                        for period in range(
                            start, start + mode["duration"]
                        ):
                            active_terms.setdefault((r, period), []).append(
                                (amount, variable)
                            )

            for (r, period), terms in active_terms.items():
                model.addConstr(
                    gp.quicksum(
                        amount * variable
                        for amount, variable in terms
                    )
                    <= renewable_capacities[r]
                )

            for r, capacity in enumerate(nonrenewable_capacities):
                model.addConstr(
                    gp.quicksum(
                        mode["nonrenewable_consumption"][r] * variable
                        for j in job_ids
                        for mode, _, variable in choices[j]
                    )
                    <= capacity
                )

            makespan = model.addVar(
                lb=0,
                ub=max(1, horizon),
                vtype=GRB.INTEGER,
                name="makespan",
            )
            makespan.Start = best_solution["makespan"]

            for p in project_ids:
                model.addConstr(
                    makespan >= start_expression[sink_by_project[p]]
                )

            delay_expression = gp.quicksum(
                start_expression[sink_by_project[p]]
                - release[p]
                - critical_path[p]
                for p in project_ids
            )

            epsilon = 1.0 / (max(1, horizon) + 1.0)
            model.setObjective(
                delay_expression + epsilon * makespan,
                GRB.MINIMIZE,
            )

            best_combined = (
                best_solution["objective_value"]
                + epsilon * best_solution["makespan"]
            )

            def callback(cb_model, where):
                nonlocal best_solution, best_combined

                if where != GRB.Callback.MIPSOL:
                    return

                try:
                    incumbent_modes = {}
                    incumbent_starts = {}

                    for job in job_ids:
                        selected_mode = None
                        selected_start = None
                        selected_value = -1.0

                        for mode, start, variable in choices[job]:
                            value = cb_model.cbGetSolution(variable)
                            if value > selected_value:
                                selected_value = value
                                selected_mode = mode["mode_id"]
                                selected_start = start

                        if selected_mode is None:
                            return

                        incumbent_modes[job] = selected_mode
                        incumbent_starts[job] = selected_start

                    candidate = build_solution(
                        job_ids,
                        incumbent_modes,
                        incumbent_starts,
                        sink_by_project,
                        release,
                        critical_path,
                    )
                    combined = (
                        candidate["objective_value"]
                        + epsilon * candidate["makespan"]
                    )

                    if combined < best_combined - 1e-7:
                        best_combined = combined
                        best_solution = candidate
                        if logger:
                            logger.log_solution(
                                candidate["objective_value"],
                                candidate,
                            )
                except Exception:
                    return

            model.optimize(callback)

            if model.SolCount > 0:
                final_modes = {}
                final_starts = {}

                for j in job_ids:
                    mode, start, _ = max(
                        choices[j],
                        key=lambda item: item[2].X,
                    )
                    final_modes[j] = mode["mode_id"]
                    final_starts[j] = start

                candidate = build_solution(
                    job_ids,
                    final_modes,
                    final_starts,
                    sink_by_project,
                    release,
                    critical_path,
                )
                combined = (
                    candidate["objective_value"]
                    + epsilon * candidate["makespan"]
                )

                if combined < best_combined - 1e-7:
                    best_solution = candidate
                    if logger:
                        logger.log_solution(
                            candidate["objective_value"],
                            candidate,
                        )

        except (gp.GurobiError, ValueError, TypeError, IndexError):
            # The serial schedule remains a valid fallback if the optional
            # time-indexed improvement model cannot be built or solved.
            pass

    output_directory = os.path.dirname(
        os.path.abspath(args.solution_path)
    )
    os.makedirs(output_directory, exist_ok=True)

    with open(args.solution_path, "w", encoding="utf-8") as file:
        json.dump(best_solution, file, indent=2)


if __name__ == "__main__":
    main()
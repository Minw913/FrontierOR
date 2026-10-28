import argparse
import json
import math
import os
import time

import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def task_sort_key(task):
    try:
        return (0, int(task))
    except (TypeError, ValueError):
        return (1, str(task))


def make_solution(assignments, processing_times, num_stations, cycle_time):
    loads = [0] * (num_stations + 1)
    for task, station in assignments.items():
        loads[int(station)] += processing_times[task]

    objective = max(cycle_time - loads[s] for s in range(1, num_stations + 1))
    return {
        "objective_value": float(objective),
        "task_assignments": {
            str(task): int(assignments[task])
            for task in sorted(assignments, key=task_sort_key)
        },
    }


def validate_assignment(
    assignments,
    tasks,
    processing_times,
    earliest,
    latest,
    arcs,
    num_stations,
    cycle_time,
):
    if len(assignments) != len(tasks):
        return False

    loads = [0] * (num_stations + 1)
    for task in tasks:
        if task not in assignments:
            return False
        station = assignments[task]
        if not isinstance(station, int):
            return False
        if station < earliest[task] or station > latest[task]:
            return False
        if station < 1 or station > num_stations:
            return False
        loads[station] += processing_times[task]
        if loads[station] > cycle_time:
            return False

    for predecessor, successor in arcs:
        if assignments[predecessor] > assignments[successor]:
            return False

    return True


def greedy_assignment(
    tasks,
    processing_times,
    earliest,
    latest,
    arcs,
    num_stations,
    cycle_time,
    priority_mode,
):
    predecessors = {task: set() for task in tasks}
    successors = {task: set() for task in tasks}

    for predecessor, successor in arcs:
        predecessors[successor].add(predecessor)
        successors[predecessor].add(successor)

    downstream_score = {task: len(successors[task]) for task in tasks}
    unscheduled = set(tasks)
    scheduled = set()
    assignment = {}

    def priority(task):
        if priority_mode == 0:
            return (latest[task], -processing_times[task], -downstream_score[task],
                    task_sort_key(task))
        if priority_mode == 1:
            return (latest[task], earliest[task], processing_times[task],
                    task_sort_key(task))
        if priority_mode == 2:
            return (latest[task], -downstream_score[task], processing_times[task],
                    task_sort_key(task))
        return (latest[task], processing_times[task], -downstream_score[task],
                task_sort_key(task))

    for station in range(1, num_stations + 1):
        remaining_capacity = cycle_time

        while True:
            # Any task whose latest station has already passed makes this
            # construction infeasible.
            if any(latest[task] < station for task in unscheduled):
                return None

            candidates = []
            for task in unscheduled:
                if earliest[task] > station or latest[task] < station:
                    continue
                if processing_times[task] > remaining_capacity:
                    continue
                if predecessors[task].issubset(scheduled):
                    candidates.append(task)

            if not candidates:
                break

            task = min(candidates, key=priority)
            assignment[task] = station
            remaining_capacity -= processing_times[task]
            unscheduled.remove(task)
            scheduled.add(task)

    if unscheduled:
        return None
    return assignment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    start_time = time.monotonic()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as instance_file:
        data = json.load(instance_file)

    num_stations = int(data["num_stations"])
    cycle_time = int(data["cycle_time"])

    processing_times = {
        str(task): int(value)
        for task, value in data["task_processing_times"].items()
    }
    tasks = sorted(processing_times.keys(), key=task_sort_key)
    task_set = set(tasks)

    earliest_data = data.get("earliest_station", {})
    latest_data = data.get("latest_station", {})

    earliest = {
        task: max(1, int(earliest_data.get(task, 1)))
        for task in tasks
    }
    latest = {
        task: min(num_stations, int(latest_data.get(task, num_stations)))
        for task in tasks
    }

    arcs = []
    for arc in data.get("precedence_arcs", []):
        predecessor = str(arc[0])
        successor = str(arc[1])
        if predecessor not in task_set or successor not in task_set:
            raise ValueError(
                f"Precedence arc references an unknown task: {arc}"
            )
        arcs.append((predecessor, successor))

    best_solution = None
    best_objective = math.inf

    def record_solution(solution):
        nonlocal best_solution, best_objective
        objective = float(solution["objective_value"])
        if objective < best_objective - 1e-9:
            best_objective = objective
            best_solution = solution
            if logger:
                logger.log_solution(objective, solution)

    # Generate deterministic feasible starts. These are also fallbacks if the
    # optimization time limit is too short for Gurobi to produce an incumbent.
    if len(tasks) <= 3000:
        for mode in range(4):
            if time.monotonic() - start_time >= max(0, args.time_limit):
                break
            candidate = greedy_assignment(
                tasks,
                processing_times,
                earliest,
                latest,
                arcs,
                num_stations,
                cycle_time,
                mode,
            )
            if candidate is not None and validate_assignment(
                candidate,
                tasks,
                processing_times,
                earliest,
                latest,
                arcs,
                num_stations,
                cycle_time,
            ):
                record_solution(
                    make_solution(
                        candidate,
                        processing_times,
                        num_stations,
                        cycle_time,
                    )
                )

    model = gp.Model("assembly_line_balancing")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1

    x = {}
    allowed_stations = {}

    for task in tasks:
        if earliest[task] > latest[task]:
            raise ValueError(f"Task {task} has an empty station domain")
        allowed_stations[task] = list(
            range(earliest[task], latest[task] + 1)
        )
        for station in allowed_stations[task]:
            x[task, station] = model.addVar(vtype=GRB.BINARY)

    total_processing_time = sum(processing_times.values())
    total_slack = num_stations * cycle_time - total_processing_time
    average_idle_lb = max(
        0,
        math.ceil(total_slack / num_stations)
    )

    max_idle = model.addVar(
        lb=average_idle_lb,
        ub=cycle_time,
        vtype=GRB.INTEGER,
        name="max_idle",
    )

    for task in tasks:
        model.addConstr(
            gp.quicksum(x[task, station]
                        for station in allowed_stations[task]) == 1
        )

    loads = {}
    for station in range(1, num_stations + 1):
        load = gp.quicksum(
            processing_times[task] * x[task, station]
            for task in tasks
            if (task, station) in x
        )
        loads[station] = load
        model.addConstr(load <= cycle_time)
        model.addConstr(load + max_idle >= cycle_time)

    station_expression = {
        task: gp.quicksum(
            station * x[task, station]
            for station in allowed_stations[task]
        )
        for task in tasks
    }

    for predecessor, successor in arcs:
        model.addConstr(
            station_expression[predecessor]
            <= station_expression[successor]
        )

    model.setObjective(max_idle, GRB.MINIMIZE)

    # Supply the best constructive solution as a MIP start.
    if best_solution is not None:
        start_assignments = best_solution["task_assignments"]
        for task in tasks:
            selected_station = int(start_assignments[task])
            for station in allowed_stations[task]:
                x[task, station].Start = (
                    1.0 if station == selected_station else 0.0
                )
        max_idle.Start = best_solution["objective_value"]

    flat_variables = []
    flat_metadata = []
    for task in tasks:
        for station in allowed_stations[task]:
            flat_variables.append(x[task, station])
            flat_metadata.append((task, station))

    remaining_time = args.time_limit - (time.monotonic() - start_time)
    model.Params.TimeLimit = max(0.001, remaining_time)

    def incumbent_callback(callback_model, where):
        if time.monotonic() - start_time >= args.time_limit:
            callback_model.terminate()
            return

        if where != GRB.Callback.MIPSOL:
            return

        try:
            values = callback_model.cbGetSolution(flat_variables)
            assignment = {}
            best_values = {}

            for value, (task, station) in zip(values, flat_metadata):
                if task not in best_values or value > best_values[task]:
                    best_values[task] = value
                    assignment[task] = station

            if len(assignment) != len(tasks):
                return

            solution = make_solution(
                assignment,
                processing_times,
                num_stations,
                cycle_time,
            )
            record_solution(solution)
        except Exception:
            # Logging or incumbent extraction must not terminate optimization.
            return

    model.optimize(incumbent_callback)

    if model.SolCount > 0:
        assignment = {}
        for task in tasks:
            assignment[task] = max(
                allowed_stations[task],
                key=lambda station: x[task, station].X,
            )

        if validate_assignment(
            assignment,
            tasks,
            processing_times,
            earliest,
            latest,
            arcs,
            num_stations,
            cycle_time,
        ):
            record_solution(
                make_solution(
                    assignment,
                    processing_times,
                    num_stations,
                    cycle_time,
                )
            )

    if best_solution is None:
        raise RuntimeError(
            "No feasible solution was found within the time limit."
        )

    solution_directory = os.path.dirname(os.path.abspath(args.solution_path))
    if solution_directory:
        os.makedirs(solution_directory, exist_ok=True)

    with open(args.solution_path, "w", encoding="utf-8") as solution_file:
        json.dump(best_solution, solution_file, indent=2)
        solution_file.write("\n")


if __name__ == "__main__":
    main()
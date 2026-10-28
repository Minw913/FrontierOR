import argparse
import heapq
import json
import math
import os
import time

from solution_logger import SolutionLogger


def build_solution(jobs, starts, machine_count):
    """Color the interval schedule onto identical machines."""
    available = list(range(machine_count))
    heapq.heapify(available)
    busy = []  # (completion_time, machine_id)

    assignments = {}
    intervals = sorted(
        (
            int(starts[j["id"]]),
            int(starts[j["id"]]) + j["processing_time"],
            j["id"],
        )
        for j in jobs
    )

    for start, completion, job_id in intervals:
        while busy and busy[0][0] <= start:
            _, machine_id = heapq.heappop(busy)
            heapq.heappush(available, machine_id)

        if not available:
            raise ValueError("Interval schedule requires more machines than available")

        machine_id = heapq.heappop(available)
        assignments[job_id] = machine_id
        heapq.heappush(busy, (completion, machine_id))

    jobs_by_id = {j["id"]: j for j in jobs}
    machine_jobs = [[] for _ in range(machine_count)]
    for job_id, machine_id in assignments.items():
        job = jobs_by_id[job_id]
        start = int(starts[job_id])
        completion = start + job["processing_time"]
        machine_jobs[machine_id].append((start, completion, job_id))

    prev_job = {}
    for machine_id in range(machine_count):
        machine_jobs[machine_id].sort(key=lambda x: (x[0], x[2]))
        previous = 0
        for _, _, job_id in machine_jobs[machine_id]:
            prev_job[job_id] = previous
            previous = job_id

    objective = 0
    schedule = []
    for job in sorted(jobs, key=lambda x: x["id"]):
        job_id = job["id"]
        start = int(starts[job_id])
        completion = start + job["processing_time"]
        weighted_tardiness = job["weight"] * max(0, completion - job["due_date"])
        objective += weighted_tardiness

        schedule.append(
            {
                "machine_id": int(assignments[job_id]),
                "prev_job": int(prev_job[job_id]),
                "job": int(job_id),
                "start_time": int(start),
                "completion_time": int(completion),
                "weighted_tardiness": int(weighted_tardiness),
            }
        )

    return {
        "objective_value": float(objective),
        "schedule": schedule,
    }


def list_schedule(order, machine_count):
    """Schedule a fixed job order on the currently least-loaded machine."""
    heap = [(0, machine_id) for machine_id in range(machine_count)]
    heapq.heapify(heap)

    starts = {}
    sequences = [[] for _ in range(machine_count)]

    for job in order:
        load, machine_id = heapq.heappop(heap)
        starts[job["id"]] = load
        sequences[machine_id].append(job["id"])
        heapq.heappush(heap, (load + job["processing_time"], machine_id))

    return starts, sequences


def atc_schedule(jobs, machine_count, kappa, deadline):
    """Apparent tardiness cost dispatching heuristic."""
    remaining = list(jobs)
    loads = [0] * machine_count
    starts = {}
    sequences = [[] for _ in range(machine_count)]
    average_p = max(1.0, sum(j["processing_time"] for j in jobs) / max(1, len(jobs)))

    while remaining:
        machine_id = min(range(machine_count), key=lambda m: (loads[m], m))
        current_time = loads[machine_id]

        best_index = 0
        best_key = None
        for idx, job in enumerate(remaining):
            p = job["processing_time"]
            urgency = max(0, job["due_date"] - p - current_time)
            score = (job["weight"] / p) * math.exp(
                -urgency / max(1e-9, kappa * average_p)
            )
            key = (score, job["weight"], -job["due_date"], -job["id"])
            if best_key is None or key > best_key:
                best_key = key
                best_index = idx

        job = remaining.pop(best_index)
        starts[job["id"]] = current_time
        sequences[machine_id].append(job["id"])
        loads[machine_id] += job["processing_time"]

        if time.monotonic() >= deadline:
            # Complete quickly using a fixed dispatch rule.
            remaining.sort(
                key=lambda j: (
                    j["due_date"],
                    j["processing_time"] / j["weight"],
                    j["id"],
                )
            )
            for rest in remaining:
                machine_id = min(range(machine_count), key=lambda m: (loads[m], m))
                starts[rest["id"]] = loads[machine_id]
                sequences[machine_id].append(rest["id"])
                loads[machine_id] += rest["processing_time"]
            break

    return starts, sequences


def starts_from_sequences(sequences, jobs_by_id):
    starts = {}
    for sequence in sequences:
        current = 0
        for job_id in sequence:
            starts[job_id] = current
            current += jobs_by_id[job_id]["processing_time"]
    return starts


def sequence_cost(sequence, jobs_by_id):
    current = 0
    cost = 0
    for job_id in sequence:
        job = jobs_by_id[job_id]
        current += job["processing_time"]
        cost += job["weight"] * max(0, current - job["due_date"])
    return cost, current


def local_search(
    sequences,
    jobs_by_id,
    horizon,
    deadline,
    on_improvement,
):
    """
    First-improvement insertion search. Jobs may move within a machine or
    between machines. All generated schedules are non-idling schedules.
    """
    sequences = [list(s) for s in sequences]
    machine_count = len(sequences)

    while time.monotonic() < deadline:
        costs_loads = [sequence_cost(s, jobs_by_id) for s in sequences]
        improved = False

        # Intra-machine insertion moves.
        for machine_id in range(machine_count):
            sequence = sequences[machine_id]
            old_cost = costs_loads[machine_id][0]

            for source_pos in range(len(sequence)):
                job_id = sequence[source_pos]
                shortened = sequence[:source_pos] + sequence[source_pos + 1 :]

                for target_pos in range(len(shortened) + 1):
                    if target_pos == source_pos:
                        continue

                    candidate = (
                        shortened[:target_pos]
                        + [job_id]
                        + shortened[target_pos:]
                    )
                    new_cost, new_load = sequence_cost(candidate, jobs_by_id)

                    if new_load <= horizon and new_cost < old_cost:
                        sequences[machine_id] = candidate
                        starts = starts_from_sequences(sequences, jobs_by_id)
                        on_improvement(starts, sequences)
                        improved = True
                        break

                    if time.monotonic() >= deadline:
                        return sequences

                if improved or time.monotonic() >= deadline:
                    break
            if improved or time.monotonic() >= deadline:
                break

        if improved:
            continue

        # Inter-machine insertion moves.
        for source_machine in range(machine_count):
            source_sequence = sequences[source_machine]
            source_cost = costs_loads[source_machine][0]

            for source_pos, job_id in enumerate(source_sequence):
                new_source = (
                    source_sequence[:source_pos]
                    + source_sequence[source_pos + 1 :]
                )
                new_source_cost, _ = sequence_cost(new_source, jobs_by_id)

                for target_machine in range(machine_count):
                    if target_machine == source_machine:
                        continue

                    target_sequence = sequences[target_machine]
                    old_pair_cost = source_cost + costs_loads[target_machine][0]

                    for target_pos in range(len(target_sequence) + 1):
                        new_target = (
                            target_sequence[:target_pos]
                            + [job_id]
                            + target_sequence[target_pos:]
                        )
                        new_target_cost, new_target_load = sequence_cost(
                            new_target, jobs_by_id
                        )

                        if (
                            new_target_load <= horizon
                            and new_source_cost + new_target_cost < old_pair_cost
                        ):
                            sequences[source_machine] = new_source
                            sequences[target_machine] = new_target
                            starts = starts_from_sequences(sequences, jobs_by_id)
                            on_improvement(starts, sequences)
                            improved = True
                            break

                        if time.monotonic() >= deadline:
                            return sequences

                    if improved or time.monotonic() >= deadline:
                        break
                if improved or time.monotonic() >= deadline:
                    break
            if improved or time.monotonic() >= deadline:
                break

        if not improved:
            break

    return sequences


def write_solution(path, solution):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(solution, f, indent=2)


def main():
    start_wall = time.monotonic()

    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as f:
        instance = json.load(f)

    jobs = [
        {
            "id": int(j["id"]),
            "processing_time": int(j["processing_time"]),
            "weight": int(j["weight"]),
            "due_date": int(j["due_date"]),
        }
        for j in instance["jobs"]
    ]
    machine_count = int(instance["parameters"]["m"])
    horizon = int(instance["time_horizon"])
    jobs_by_id = {j["id"]: j for j in jobs}

    best_solution = None
    best_starts = None
    best_sequences = None
    best_objective = float("inf")

    def consider(starts, sequences=None):
        nonlocal best_solution, best_starts, best_sequences, best_objective
        try:
            solution = build_solution(jobs, starts, machine_count)
        except ValueError:
            return False

        objective = solution["objective_value"]
        if objective + 1e-9 < best_objective:
            best_objective = objective
            best_solution = solution
            best_starts = dict(starts)
            best_sequences = [list(s) for s in sequences] if sequences is not None else None
            if logger:
                logger.log_solution(objective, solution)
            return True
        return False

    # A valid incumbent is produced immediately.
    initial_order = sorted(
        jobs,
        key=lambda j: (
            j["due_date"],
            j["processing_time"] / j["weight"],
            j["id"],
        ),
    )
    starts, sequences = list_schedule(initial_order, machine_count)
    consider(starts, sequences)

    # Several inexpensive deterministic priority rules.
    fixed_orders = [
        sorted(jobs, key=lambda j: (j["due_date"], j["id"])),
        sorted(jobs, key=lambda j: (j["due_date"] - j["processing_time"], j["id"])),
        sorted(
            jobs,
            key=lambda j: (
                j["processing_time"] / j["weight"],
                j["due_date"],
                j["id"],
            ),
        ),
        sorted(
            jobs,
            key=lambda j: (
                -j["weight"] / j["processing_time"],
                j["due_date"],
                j["id"],
            ),
        ),
        sorted(
            jobs,
            key=lambda j: (
                j["due_date"] / max(1, j["weight"]),
                j["processing_time"],
                j["id"],
            ),
        ),
    ]

    for order in fixed_orders:
        if time.monotonic() - start_wall >= args.time_limit:
            break
        starts, sequences = list_schedule(order, machine_count)
        consider(starts, sequences)

    heuristic_deadline = min(
        start_wall + max(0, args.time_limit),
        time.monotonic() + min(1.5, max(0.05, args.time_limit * 0.08)),
    )

    if len(jobs) <= 1000:
        for kappa in (0.5, 1.0, 2.0, 4.0):
            if time.monotonic() >= heuristic_deadline:
                break
            starts, sequences = atc_schedule(
                jobs, machine_count, kappa, heuristic_deadline
            )
            consider(starts, sequences)

    # Small local search before creating the exact model.
    if (
        best_sequences is not None
        and len(jobs) <= 120
        and time.monotonic() < heuristic_deadline
    ):
        local_search(
            best_sequences,
            jobs_by_id,
            horizon,
            heuristic_deadline,
            consider,
        )

    remaining = args.time_limit - (time.monotonic() - start_wall)

    # Number of binary variables in the time-indexed model.
    variable_count = sum(
        max(0, horizon - job["processing_time"] + 1) for job in jobs
    )

    # Avoid spending the entire limit constructing an excessively large model.
    if remaining > 0.25:
        if remaining < 2:
            model_limit = 150_000
        elif remaining < 10:
            model_limit = 400_000
        else:
            model_limit = 750_000
    else:
        model_limit = 0

    if variable_count <= model_limit and variable_count > 0:
        try:
            import gurobipy as gp
            from gurobipy import GRB

            model = gp.Model("parallel_machine_weighted_tardiness")
            model.Params.Seed = 0
            model.Params.MIPGap = 1e-4
            model.Params.NumericFocus = 0
            model.Params.Threads = 1
            model.Params.OutputFlag = 0
            model.Params.TimeLimit = max(
                0.01, args.time_limit - (time.monotonic() - start_wall)
            )
            model.ModelSense = GRB.MINIMIZE

            x_by_job = {}
            start_vars = {}
            starts_at = [[] for _ in range(horizon)]
            ends_at = [[] for _ in range(horizon + 1)]

            for job in jobs:
                job_id = job["id"]
                p = job["processing_time"]
                due = job["due_date"]
                weight = job["weight"]

                variables = []
                for t in range(horizon - p + 1):
                    coefficient = weight * max(0, t + p - due)
                    var = model.addVar(
                        vtype=GRB.BINARY,
                        obj=coefficient,
                        name=f"x_{job_id}_{t}",
                    )
                    variables.append(var)
                    starts_at[t].append(var)
                    ends_at[t + p].append(var)

                x_by_job[job_id] = variables

            model.update()

            for job in jobs:
                job_id = job["id"]
                variables = x_by_job[job_id]
                model.addConstr(gp.quicksum(variables) == 1, name=f"assign_{job_id}")

                start_var = model.addVar(
                    lb=0.0,
                    ub=float(horizon - job["processing_time"]),
                    vtype=GRB.CONTINUOUS,
                    name=f"start_{job_id}",
                )
                start_vars[job_id] = start_var
                model.addConstr(
                    start_var
                    == gp.quicksum(t * variables[t] for t in range(len(variables))),
                    name=f"start_link_{job_id}",
                )

            # Recursive machine-occupancy equations use only two nonzeros per
            # start variable rather than explicitly listing every active period.
            occupancy = []
            for t in range(horizon):
                occ = model.addVar(
                    lb=0.0,
                    ub=float(machine_count),
                    vtype=GRB.CONTINUOUS,
                    name=f"occupancy_{t}",
                )
                occupancy.append(occ)

                expression = gp.quicksum(starts_at[t]) - gp.quicksum(ends_at[t])
                if t == 0:
                    model.addConstr(occ == expression, name=f"flow_{t}")
                else:
                    model.addConstr(
                        occ == occupancy[t - 1] + expression,
                        name=f"flow_{t}",
                    )

            model.update()

            # Warm start from the best heuristic schedule.
            if best_starts is not None:
                for job in jobs:
                    job_id = job["id"]
                    selected_t = int(best_starts[job_id])
                    if 0 <= selected_t < len(x_by_job[job_id]):
                        x_by_job[job_id][selected_t].Start = 1.0
                    start_vars[job_id].Start = float(selected_t)

            callback_state = {
                "best_objective": best_objective,
                "best_solution": best_solution,
                "best_starts": best_starts,
            }

            def callback(cb_model, where):
                if where != GRB.Callback.MIPSOL:
                    return

                callback_objective = cb_model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if callback_objective + 1e-7 >= callback_state["best_objective"]:
                    return

                try:
                    candidate_starts = {
                        job["id"]: int(
                            round(cb_model.cbGetSolution(start_vars[job["id"]]))
                        )
                        for job in jobs
                    }
                    candidate_solution = build_solution(
                        jobs, candidate_starts, machine_count
                    )
                    exact_objective = candidate_solution["objective_value"]

                    if exact_objective + 1e-9 < callback_state["best_objective"]:
                        callback_state["best_objective"] = exact_objective
                        callback_state["best_solution"] = candidate_solution
                        callback_state["best_starts"] = candidate_starts
                        if logger:
                            logger.log_solution(exact_objective, candidate_solution)
                except Exception:
                    # A callback logging failure must not prevent optimization.
                    pass

            model.optimize(callback)

            if model.SolCount > 0:
                candidate_starts = {
                    job["id"]: int(round(start_vars[job["id"]].X))
                    for job in jobs
                }
                consider(candidate_starts)

            if (
                callback_state["best_solution"] is not None
                and callback_state["best_objective"] + 1e-9 < best_objective
            ):
                best_objective = callback_state["best_objective"]
                best_solution = callback_state["best_solution"]
                best_starts = callback_state["best_starts"]

        except Exception:
            # Preserve and return the heuristic incumbent if Gurobi is
            # unavailable, model construction fails, or the limit is exhausted.
            pass

    if best_solution is None:
        # Defensive fallback; normally the initial list schedule always exists.
        starts, sequences = list_schedule(jobs, machine_count)
        best_solution = build_solution(jobs, starts, machine_count)

    write_solution(args.solution_path, best_solution)


if __name__ == "__main__":
    main()
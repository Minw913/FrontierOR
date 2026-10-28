import argparse
import json
import os
import random
import time

from solution_logger import SolutionLogger


def read_instance(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    n = int(data["n_jobs"])
    m = int(data["n_machines"])
    processing_times = [
        [int(v) for v in row] for row in data["processing_times"]
    ]

    if len(processing_times) != n:
        raise ValueError("processing_times must contain n_jobs rows")
    if any(len(row) != m for row in processing_times):
        raise ValueError("Each processing_times row must contain n_machines values")
    if any(v < 0 for row in processing_times for v in row):
        raise ValueError("Processing times must be nonnegative")

    flags = data.get("no_idle_flags")
    if flags is not None and len(flags) == m:
        no_idle = [bool(v) for v in flags]
    else:
        no_idle = [False] * m
        for machine in data.get("no_idle_machines", []):
            idx = int(machine) - 1
            if not 0 <= idx < m:
                raise ValueError("Invalid no-idle machine index")
            no_idle[idx] = True

    return n, m, no_idle, processing_times


def compute_schedule(sequence, processing_times, no_idle):
    """Return the earliest feasible schedule for a fixed permutation."""
    n = len(sequence)
    m = len(no_idle)

    if n == 0:
        return 0, [[] for _ in range(m)]

    previous_machine = [0] * n
    all_completion_times = []

    for machine in range(m):
        if no_idle[machine]:
            # If the first job starts at shift, position k starts at
            # shift + sum of processing times in positions before k.
            shift = 0
            prefix_before = 0
            for pos, job in enumerate(sequence):
                required_shift = previous_machine[pos] - prefix_before
                if required_shift > shift:
                    shift = required_shift
                prefix_before += processing_times[job][machine]

            completion = []
            current = shift
            for job in sequence:
                current += processing_times[job][machine]
                completion.append(current)
        else:
            completion = []
            current = 0
            for pos, job in enumerate(sequence):
                current = max(current, previous_machine[pos])
                current += processing_times[job][machine]
                completion.append(current)

        all_completion_times.append(completion)
        previous_machine = completion

    objective = previous_machine[-1] if m > 0 else 0
    return objective, all_completion_times


def make_solution(sequence, objective, completion_times):
    return {
        "objective_value": float(objective),
        "sequence": [int(job) + 1 for job in sequence],
        "completion_times": {
            f"machine_{machine + 1}": [
                int(value) if isinstance(value, int) or float(value).is_integer()
                else float(value)
                for value in completion_times[machine]
            ]
            for machine in range(len(completion_times))
        },
    }


def neh_construction(order, processing_times, no_idle, deadline):
    sequence = []
    processed = 0

    for idx, job in enumerate(order):
        if time.monotonic() >= deadline:
            break

        best_partial = None
        best_value = None
        for position in range(len(sequence) + 1):
            candidate = sequence[:position] + [job] + sequence[position:]
            value, _ = compute_schedule(candidate, processing_times, no_idle)
            if best_value is None or value < best_value:
                best_value = value
                best_partial = candidate

            if time.monotonic() >= deadline:
                break

        if best_partial is None:
            break

        sequence = best_partial
        processed = idx + 1

    # Preserve a complete permutation if the construction was interrupted.
    sequence.extend(order[processed:])
    return sequence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    start_time = time.monotonic()
    hard_deadline = start_time + max(0, args.time_limit)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    n, m, no_idle, processing_times = read_instance(args.instance_path)

    best_sequence = list(range(n))
    best_objective, best_completion = compute_schedule(
        best_sequence, processing_times, no_idle
    )
    best_solution = make_solution(
        best_sequence, best_objective, best_completion
    )

    if logger:
        logger.log_solution(best_objective, best_solution)

    def consider(sequence):
        nonlocal best_sequence, best_objective, best_completion, best_solution

        objective, completion = compute_schedule(
            sequence, processing_times, no_idle
        )
        if objective < best_objective - 1e-9:
            best_sequence = list(sequence)
            best_objective = objective
            best_completion = completion
            best_solution = make_solution(
                best_sequence, best_objective, best_completion
            )
            if logger:
                logger.log_solution(best_objective, best_solution)
            return True
        return False

    if n > 0 and m > 0:
        total_work = [sum(processing_times[j]) for j in range(n)]

        # Use the MIP only when its assignment formulation is reasonably sized.
        estimated_size = n * n + n * m
        plan_mip = (
            args.time_limit >= 2
            and n <= 140
            and estimated_size <= 120000
        )

        if plan_mip:
            heuristic_seconds = min(3.0, max(0.08, args.time_limit * 0.25))
            heuristic_deadline = min(
                hard_deadline - 0.08,
                start_time + heuristic_seconds,
            )
        else:
            heuristic_deadline = hard_deadline - 0.03

        # Several inexpensive deterministic priority rules.
        candidate_orders = [
            sorted(range(n), key=lambda j: (-total_work[j], j)),
            sorted(range(n), key=lambda j: (total_work[j], j)),
            sorted(
                range(n),
                key=lambda j: (
                    processing_times[j][0],
                    -processing_times[j][-1],
                    j,
                ),
            ),
            sorted(
                range(n),
                key=lambda j: (
                    -processing_times[j][-1],
                    processing_times[j][0],
                    j,
                ),
            ),
        ]

        # Johnson-style aggregate rules for selected machine splits.
        if m >= 2:
            splits = {1, m // 2, m - 1}
            for split in sorted(s for s in splits if 0 < s < m):
                first = {
                    j: sum(processing_times[j][:split]) for j in range(n)
                }
                second = {
                    j: sum(processing_times[j][split:]) for j in range(n)
                }
                front = [j for j in range(n) if first[j] <= second[j]]
                back = [j for j in range(n) if first[j] > second[j]]
                front.sort(key=lambda j: (first[j], j))
                back.sort(key=lambda j: (-second[j], j))
                candidate_orders.append(front + back)

        seen_orders = set()
        for order in candidate_orders:
            if time.monotonic() >= heuristic_deadline:
                break
            key = tuple(order)
            if key not in seen_orders:
                seen_orders.add(key)
                consider(order)

        # NEH insertion construction.
        if time.monotonic() < heuristic_deadline:
            neh_order = sorted(range(n), key=lambda j: (-total_work[j], j))
            neh_sequence = neh_construction(
                neh_order,
                processing_times,
                no_idle,
                heuristic_deadline,
            )
            consider(neh_sequence)

        # Insertion local search. Small instances use complete neighborhoods;
        # larger instances use deterministic randomized sampling.
        rng = random.Random(0)
        current = list(best_sequence)
        current_value = best_objective

        if n > 1 and time.monotonic() < heuristic_deadline:
            if n <= 45:
                improved = True
                while improved and time.monotonic() < heuristic_deadline:
                    improved = False
                    move_sequence = None
                    move_value = current_value

                    for source in range(n):
                        if time.monotonic() >= heuristic_deadline:
                            break
                        reduced = current[:source] + current[source + 1:]
                        job = current[source]

                        for target in range(n):
                            if target == source:
                                continue
                            candidate = reduced[:target] + [job] + reduced[target:]
                            value, _ = compute_schedule(
                                candidate, processing_times, no_idle
                            )
                            if value < move_value:
                                move_value = value
                                move_sequence = candidate
                            if time.monotonic() >= heuristic_deadline:
                                break

                    if move_sequence is not None:
                        current = move_sequence
                        current_value = move_value
                        consider(current)
                        improved = True
            else:
                failures = 0
                failure_limit = max(100, 8 * n)
                while (
                    time.monotonic() < heuristic_deadline
                    and failures < failure_limit
                ):
                    source = rng.randrange(n)
                    target = rng.randrange(n)
                    if source == target:
                        continue

                    candidate = list(current)
                    job = candidate.pop(source)
                    candidate.insert(target, job)
                    value, _ = compute_schedule(
                        candidate, processing_times, no_idle
                    )

                    if value < current_value:
                        current = candidate
                        current_value = value
                        consider(current)
                        failures = 0
                    else:
                        failures += 1

        # Position-assignment MIP. It exactly models the scheduling constraints
        # because the processing time at each position is linear in assignment.
        if plan_mip and time.monotonic() < hard_deadline - 0.12:
            try:
                import gurobipy as gp
                from gurobipy import GRB

                model = gp.Model("mixed_no_idle_permutation_flowshop")
                model.Params.OutputFlag = 0
                model.Params.Seed = 0
                model.Params.MIPGap = 1e-4
                model.Params.NumericFocus = 0
                model.Params.Threads = 1

                horizon = max(
                    1, sum(sum(row) for row in processing_times)
                )

                x = model.addVars(
                    n, n, vtype=GRB.BINARY, name="assign"
                )
                completion = model.addVars(
                    m,
                    n,
                    lb=0.0,
                    ub=float(horizon),
                    vtype=GRB.CONTINUOUS,
                    name="completion",
                )

                for pos in range(n):
                    model.addConstr(
                        gp.quicksum(x[pos, job] for job in range(n)) == 1
                    )
                for job in range(n):
                    model.addConstr(
                        gp.quicksum(x[pos, job] for pos in range(n)) == 1
                    )

                processing_expr = {}
                for machine in range(m):
                    for pos in range(n):
                        processing_expr[machine, pos] = gp.quicksum(
                            processing_times[job][machine] * x[pos, job]
                            for job in range(n)
                        )

                for machine in range(m):
                    for pos in range(n):
                        q = processing_expr[machine, pos]

                        if pos == 0:
                            model.addConstr(completion[machine, pos] >= q)
                        elif no_idle[machine]:
                            model.addConstr(
                                completion[machine, pos]
                                == completion[machine, pos - 1] + q
                            )
                        else:
                            model.addConstr(
                                completion[machine, pos]
                                >= completion[machine, pos - 1] + q
                            )

                        if machine > 0:
                            model.addConstr(
                                completion[machine, pos]
                                >= completion[machine - 1, pos] + q
                            )

                model.setObjective(
                    completion[m - 1, n - 1], GRB.MINIMIZE
                )

                # Complete MIP start from the incumbent heuristic schedule.
                incumbent_positions = {
                    job: pos for pos, job in enumerate(best_sequence)
                }
                for pos in range(n):
                    for job in range(n):
                        x[pos, job].Start = (
                            1.0 if incumbent_positions[job] == pos else 0.0
                        )
                for machine in range(m):
                    for pos in range(n):
                        completion[machine, pos].Start = float(
                            best_completion[machine][pos]
                        )

                model.update()
                remaining = hard_deadline - time.monotonic() - 0.05

                if remaining > 0:
                    model.Params.TimeLimit = max(0.01, remaining)

                    def mip_callback(cb_model, where):
                        if where != GRB.Callback.MIPSOL:
                            return
                        try:
                            sequence = []
                            used = set()
                            for pos in range(n):
                                values = cb_model.cbGetSolution(
                                    [x[pos, job] for job in range(n)]
                                )
                                job = max(range(n), key=lambda j: values[j])
                                if job in used or values[job] < 0.5:
                                    return
                                used.add(job)
                                sequence.append(job)

                            if len(sequence) == n:
                                consider(sequence)
                        except Exception:
                            # Logging or extraction failures must not interrupt
                            # the optimization run.
                            return

                    model.optimize(mip_callback)

                    if model.SolCount > 0:
                        mip_sequence = []
                        used = set()
                        valid = True
                        for pos in range(n):
                            job = max(
                                range(n), key=lambda j: x[pos, j].X
                            )
                            if job in used or x[pos, job].X < 0.5:
                                valid = False
                                break
                            used.add(job)
                            mip_sequence.append(job)
                        if valid:
                            consider(mip_sequence)

            except Exception:
                # A heuristic incumbent is always available, including in case
                # of a Gurobi license, construction, or runtime failure.
                pass

    output_dir = os.path.dirname(os.path.abspath(args.solution_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(best_solution, f, indent=2)


if __name__ == "__main__":
    main()
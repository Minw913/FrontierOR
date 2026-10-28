import argparse
import json
import math
import random
import time
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def machine_cost(sequence, jobs):
    completion = 0
    value = 0
    for j in sequence:
        completion += jobs[j]["p"]
        value += jobs[j]["w"] * max(0, completion - jobs[j]["d"])
    return value


def schedule_cost(sequences, jobs):
    return sum(machine_cost(seq, jobs) for seq in sequences)


def machine_load(sequence, jobs):
    return sum(jobs[j]["p"] for j in sequence)


def make_solution(sequences, jobs):
    starts = {}
    objective = 0

    for sequence in sequences:
        current_time = 0
        for j in sequence:
            starts[str(jobs[j]["id"])] = int(current_time)
            current_time += jobs[j]["p"]
            objective += jobs[j]["w"] * max(0, current_time - jobs[j]["d"])

    return {
        "objective_value": float(objective),
        "schedule": starts,
    }


def fixed_order_schedule(order, machine_count, jobs, horizon, assignment_rule):
    sequences = [[] for _ in range(machine_count)]
    loads = [0] * machine_count

    for j in order:
        feasible = [
            k for k in range(machine_count)
            if loads[k] + jobs[j]["p"] <= horizon
        ]
        if not feasible:
            feasible = list(range(machine_count))

        if assignment_rule == "earliest":
            k = min(feasible, key=lambda q: (loads[q], len(sequences[q]), q))
        else:
            k = min(
                feasible,
                key=lambda q: (
                    jobs[j]["w"] * max(
                        0, loads[q] + jobs[j]["p"] - jobs[j]["d"]
                    ),
                    loads[q],
                    len(sequences[q]),
                    q,
                ),
            )

        sequences[k].append(j)
        loads[k] += jobs[j]["p"]

    return sequences


def lpt_partition_with_sequence_rule(machine_count, jobs, horizon, sequence_key):
    sequences = [[] for _ in range(machine_count)]
    loads = [0] * machine_count

    order = sorted(
        range(len(jobs)),
        key=lambda j: (-jobs[j]["p"], jobs[j]["d"], -jobs[j]["w"], j),
    )

    for j in order:
        feasible = [
            k for k in range(machine_count)
            if loads[k] + jobs[j]["p"] <= horizon
        ]
        if not feasible:
            feasible = list(range(machine_count))
        k = min(feasible, key=lambda q: (loads[q], q))
        sequences[k].append(j)
        loads[k] += jobs[j]["p"]

    for k in range(machine_count):
        sequences[k].sort(key=sequence_key)

    return sequences


def normalize_interval_solution(start_times, machine_count, jobs):
    """Color intervals onto machines, then left-shift each machine."""
    sequences = [[] for _ in range(machine_count)]
    ends = [0] * machine_count

    ordered = sorted(
        range(len(jobs)),
        key=lambda j: (start_times[j], start_times[j] + jobs[j]["p"], j),
    )

    for j in ordered:
        s = start_times[j]
        available = [k for k in range(machine_count) if ends[k] <= s]

        if available:
            k = max(available, key=lambda q: (ends[q], -q))
        else:
            # This should not happen for a feasible time-indexed incumbent.
            k = min(range(machine_count), key=lambda q: (ends[q], q))

        sequences[k].append(j)
        ends[k] = s + jobs[j]["p"]

    return sequences


def estimate_time_indexed_size(jobs, horizon):
    variables = 0
    nonzeros = 0

    for job in jobs:
        starts = max(0, horizon - job["p"] + 1)
        variables += starts
        nonzeros += starts * (job["p"] + 1)

    return variables, nonzeros


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    start_clock = time.monotonic()
    deadline = start_clock + max(0, args.time_limit)

    with open(args.instance_path, "r", encoding="utf-8") as f:
        instance = json.load(f)

    jobs = []
    for item in instance["jobs"]:
        jobs.append(
            {
                "id": item["id"],
                "p": int(item["p"]),
                "d": int(item["d"]),
                "w": int(item["w"]),
            }
        )

    n = len(jobs)
    m = int(instance["m"])
    if m <= 0:
        raise ValueError("The number of machines must be positive")

    if n == 0:
        solution = {"objective_value": 0.0, "schedule": {}}
        if logger:
            logger.log_solution(0.0, solution)
        Path(args.solution_path).parent.mkdir(parents=True, exist_ok=True)
        with open(args.solution_path, "w", encoding="utf-8") as f:
            json.dump(solution, f)
        return

    total_processing = sum(job["p"] for job in jobs)
    declared_horizon = int(instance.get("T", total_processing))
    horizon = max(max(job["p"] for job in jobs), declared_horizon)

    best_sequences = None
    best_value = math.inf

    def record_candidate(sequences):
        nonlocal best_sequences, best_value

        if len(sequences) != m or sum(len(s) for s in sequences) != n:
            return False

        seen = [j for seq in sequences for j in seq]
        if len(set(seen)) != n:
            return False

        value = schedule_cost(sequences, jobs)
        if value < best_value:
            best_value = value
            best_sequences = [list(seq) for seq in sequences]
            solution = make_solution(best_sequences, jobs)
            if logger:
                logger.log_solution(float(value), solution)
            return True
        return False

    indices = list(range(n))
    priority_orders = [
        sorted(indices, key=lambda j: (jobs[j]["d"], jobs[j]["p"], -jobs[j]["w"], j)),
        sorted(indices, key=lambda j: (jobs[j]["p"] / jobs[j]["w"], jobs[j]["d"], j)),
        sorted(indices, key=lambda j: (jobs[j]["d"] / jobs[j]["w"], jobs[j]["p"], j)),
        sorted(indices, key=lambda j: (-jobs[j]["w"] / jobs[j]["p"], jobs[j]["d"], j)),
        sorted(indices, key=lambda j: (jobs[j]["d"] - jobs[j]["p"], jobs[j]["p"], j)),
        sorted(indices, key=lambda j: (-jobs[j]["w"], jobs[j]["d"], jobs[j]["p"], j)),
        sorted(indices, key=lambda j: (jobs[j]["p"], jobs[j]["d"], -jobs[j]["w"], j)),
        sorted(indices, key=lambda j: (-jobs[j]["p"], jobs[j]["d"], -jobs[j]["w"], j)),
    ]

    # Fast constructive incumbents.
    for order in priority_orders:
        if time.monotonic() >= deadline:
            break
        record_candidate(fixed_order_schedule(order, m, jobs, horizon, "earliest"))
        if time.monotonic() >= deadline:
            break
        record_candidate(fixed_order_schedule(order, m, jobs, horizon, "tardiness"))

    sequence_keys = [
        lambda j: (jobs[j]["d"], jobs[j]["p"], -jobs[j]["w"], j),
        lambda j: (jobs[j]["p"] / jobs[j]["w"], jobs[j]["d"], j),
        lambda j: (-jobs[j]["w"] / jobs[j]["p"], jobs[j]["d"], j),
    ]
    for key in sequence_keys:
        if time.monotonic() >= deadline:
            break
        record_candidate(lpt_partition_with_sequence_rule(m, jobs, horizon, key))

    if best_sequences is None:
        # Guaranteed fallback.
        best_sequences = fixed_order_schedule(priority_orders[0], m, jobs, total_processing, "earliest")
        best_value = schedule_cost(best_sequences, jobs)
        if logger:
            logger.log_solution(best_value, make_solution(best_sequences, jobs))

    variable_estimate, nonzero_estimate = estimate_time_indexed_size(jobs, horizon)
    use_mip = (
        variable_estimate <= 120000
        and nonzero_estimate <= 4000000
        and time.monotonic() + 0.15 < deadline
    )

    # Randomized relocation/swap search. Reserve most time for the exact MIP
    # when the time-indexed formulation is reasonably sized.
    remaining = max(0.0, deadline - time.monotonic())
    if use_mip:
        heuristic_end = min(deadline, time.monotonic() + min(2.0, 0.18 * remaining))
    else:
        heuristic_end = deadline

    rng = random.Random(0)
    current = [list(seq) for seq in best_sequences]
    current_value = schedule_cost(current, jobs)
    current_loads = [machine_load(seq, jobs) for seq in current]

    base_scale = max(
        1.0,
        sum(job["w"] * max(1, job["p"]) for job in jobs) / max(1, n),
    )
    temperature = base_scale
    iteration = 0

    while time.monotonic() < heuristic_end:
        iteration += 1
        nonempty = [k for k in range(m) if current[k]]
        if not nonempty:
            break

        do_swap = n > 1 and rng.random() < 0.30

        if do_swap:
            a = rng.choice(nonempty)
            b = rng.randrange(m)
            if not current[b]:
                continue

            ia = rng.randrange(len(current[a]))
            ib = rng.randrange(len(current[b]))

            if a == b and ia == ib:
                continue

            old_cost = machine_cost(current[a], jobs)
            if b != a:
                old_cost += machine_cost(current[b], jobs)

            ja = current[a][ia]
            jb = current[b][ib]

            if a != b:
                new_load_a = current_loads[a] - jobs[ja]["p"] + jobs[jb]["p"]
                new_load_b = current_loads[b] - jobs[jb]["p"] + jobs[ja]["p"]
                if new_load_a > horizon or new_load_b > horizon:
                    continue

            current[a][ia], current[b][ib] = current[b][ib], current[a][ia]

            new_cost = machine_cost(current[a], jobs)
            if b != a:
                new_cost += machine_cost(current[b], jobs)
            delta = new_cost - old_cost

            accept = delta <= 0 or rng.random() < math.exp(
                -min(700.0, delta / max(1e-9, temperature))
            )
            if accept:
                current_value += delta
                if a != b:
                    current_loads[a] = new_load_a
                    current_loads[b] = new_load_b
            else:
                current[a][ia], current[b][ib] = current[b][ib], current[a][ia]

        else:
            a = rng.choice(nonempty)
            ia = rng.randrange(len(current[a]))
            j = current[a][ia]
            b = rng.randrange(m)

            if a != b and current_loads[b] + jobs[j]["p"] > horizon:
                continue

            old_cost = machine_cost(current[a], jobs)
            if b != a:
                old_cost += machine_cost(current[b], jobs)

            current[a].pop(ia)
            insert_position = rng.randrange(len(current[b]) + 1)
            current[b].insert(insert_position, j)

            new_cost = machine_cost(current[a], jobs)
            if b != a:
                new_cost += machine_cost(current[b], jobs)
            delta = new_cost - old_cost

            accept = delta <= 0 or rng.random() < math.exp(
                -min(700.0, delta / max(1e-9, temperature))
            )

            if accept:
                current_value += delta
                if a != b:
                    current_loads[a] -= jobs[j]["p"]
                    current_loads[b] += jobs[j]["p"]
            else:
                current[b].pop(insert_position)
                current[a].insert(ia, j)

        if current_value < best_value:
            record_candidate(current)

        temperature *= 0.9995
        if temperature < 0.01 * base_scale:
            temperature = base_scale
            current = [list(seq) for seq in best_sequences]
            current_value = best_value
            current_loads = [machine_load(seq, jobs) for seq in current]

        if iteration % 256 == 0 and time.monotonic() >= heuristic_end:
            break

    # Time-indexed MIP for instances where it is compact enough.
    if use_mip and time.monotonic() < deadline:
        try:
            model = gp.Model("parallel_machine_weighted_tardiness")
            model.Params.OutputFlag = 0
            model.Params.Seed = 0
            model.Params.MIPGap = 1e-4
            model.Params.NumericFocus = 0
            model.Params.Threads = 1
            model.Params.TimeLimit = max(0.01, deadline - time.monotonic())
            model.ModelSense = GRB.MINIMIZE

            assignment_constraints = [
                model.addConstr(gp.LinExpr() == 1.0, name=f"assign_{j}")
                for j in range(n)
            ]
            capacity_constraints = [
                model.addConstr(gp.LinExpr() <= float(m), name=f"capacity_{t}")
                for t in range(horizon)
            ]

            x = []
            start_values = []
            warm_solution = make_solution(best_sequences, jobs)["schedule"]

            for j, job in enumerate(jobs):
                row_vars = []
                row_starts = []
                latest = horizon - job["p"]

                for s in range(latest + 1):
                    constraints = [assignment_constraints[j]]
                    coefficients = [1.0]

                    for t in range(s, s + job["p"]):
                        constraints.append(capacity_constraints[t])
                        coefficients.append(1.0)

                    tardiness_cost = job["w"] * max(0, s + job["p"] - job["d"])
                    column = gp.Column(coefficients, constraints)
                    var = model.addVar(
                        vtype=GRB.BINARY,
                        obj=float(tardiness_cost),
                        column=column,
                        name=f"x_{j}_{s}",
                    )

                    expected_start = warm_solution.get(str(job["id"]))
                    var.Start = 1.0 if expected_start == s else 0.0
                    row_vars.append(var)
                    row_starts.append(s)

                x.append(row_vars)
                start_values.append(row_starts)

            model.update()
            callback_best = [best_value]

            def mip_callback(cb_model, where):
                if where != GRB.Callback.MIPSOL:
                    return
                try:
                    starts = [0] * n
                    for j in range(n):
                        values = cb_model.cbGetSolution(x[j])
                        chosen = max(range(len(values)), key=lambda q: values[q])
                        starts[j] = start_values[j][chosen]

                    sequences = normalize_interval_solution(starts, m, jobs)
                    value = schedule_cost(sequences, jobs)

                    if value < callback_best[0]:
                        callback_best[0] = value
                        record_candidate(sequences)
                except Exception:
                    # Never interrupt optimization merely because an incumbent
                    # could not be decoded during a callback.
                    pass

            model.optimize(mip_callback)

            if model.SolCount > 0:
                mip_starts = [0] * n
                for j in range(n):
                    chosen = max(range(len(x[j])), key=lambda q: x[j][q].X)
                    mip_starts[j] = start_values[j][chosen]
                mip_sequences = normalize_interval_solution(mip_starts, m, jobs)
                record_candidate(mip_sequences)

        except gp.GurobiError:
            # The heuristic incumbent remains valid if model construction or
            # optimization cannot be completed.
            pass
        except Exception:
            pass

    final_solution = make_solution(best_sequences, jobs)
    final_solution["objective_value"] = float(best_value)

    solution_parent = Path(args.solution_path).parent
    solution_parent.mkdir(parents=True, exist_ok=True)
    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(final_solution, f, separators=(",", ":"))


if __name__ == "__main__":
    main()
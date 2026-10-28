import argparse
import json
import math
import random
import time

import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    start_time = time.monotonic()
    deadline = start_time + max(0, args.time_limit)

    with open(args.instance_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    n = int(data["n_jobs"])
    m = int(data["n_machines"])
    processing = data["processing_times"]
    due_dates = [int(x) for x in data["due_dates"]]

    if n == 0:
        solution = {"objective_value": 0.0, "sequence": []}
        if logger:
            logger.log_solution(0.0, solution)
        with open(args.solution_path, "w", encoding="utf-8") as f:
            json.dump(solution, f)
        return

    # Prefix processing time before each machine.
    prefix = [[0] * n for _ in range(m)]
    for machine in range(1, m):
        for job in range(n):
            prefix[machine][job] = (
                prefix[machine - 1][job] + int(processing[machine - 1][job])
            )

    last_offset = prefix[m - 1]
    last_processing = [int(processing[m - 1][j]) for j in range(n)]
    total_processing = [
        last_offset[j] + last_processing[j] for j in range(n)
    ]

    # If i immediately precedes j, their machine-1 start times must differ
    # by at least delta[i][j]. In last-machine coordinates this gives arc
    # length transition[i][j].
    transition = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            delta = max(
                prefix[k][i] + int(processing[k][i]) - prefix[k][j]
                for k in range(m)
            )
            transition[i][j] = delta + last_offset[j] - last_offset[i]

    def evaluate(sequence):
        """Return (feasible, makespan, last-machine start times)."""
        if not sequence:
            return True, 0, []
        starts = [0] * len(sequence)
        starts[0] = last_offset[sequence[0]]
        if starts[0] + last_processing[sequence[0]] > due_dates[sequence[0]]:
            return False, starts[0] + last_processing[sequence[0]], starts

        for pos in range(1, len(sequence)):
            prev_job = sequence[pos - 1]
            job = sequence[pos]
            starts[pos] = starts[pos - 1] + transition[prev_job][job]
            if starts[pos] + last_processing[job] > due_dates[job]:
                return False, starts[pos] + last_processing[job], starts

        return True, starts[-1] + last_processing[sequence[-1]], starts

    best_sequence = None
    best_objective = math.inf

    def submit(sequence):
        nonlocal best_sequence, best_objective
        feasible, objective, _ = evaluate(sequence)
        if feasible and len(sequence) == n and objective < best_objective - 1e-7:
            best_sequence = list(sequence)
            best_objective = float(objective)
            solution = {
                "objective_value": float(objective),
                "sequence": [j + 1 for j in sequence],
            }
            if logger:
                logger.log_solution(float(objective), solution)
            return True
        return False

    # Basic deterministic candidates.
    candidates = [
        sorted(range(n), key=lambda j: (due_dates[j], total_processing[j], j)),
        sorted(
            range(n),
            key=lambda j: (
                due_dates[j] - total_processing[j],
                due_dates[j],
                j,
            ),
        ),
        sorted(range(n), key=lambda j: (total_processing[j], due_dates[j], j)),
        sorted(range(n), key=lambda j: (last_offset[j], due_dates[j], j)),
    ]
    for candidate in candidates:
        submit(candidate)

    heuristic_deadline = min(
        deadline,
        start_time + max(0.03, min(2.0, 0.15 * max(1, args.time_limit))),
    )
    rng = random.Random(0)

    def insertion_construct(order):
        sequence = []
        for job in order:
            chosen = None
            chosen_obj = math.inf
            for pos in range(len(sequence) + 1):
                trial = sequence[:pos] + [job] + sequence[pos:]
                feasible, objective, _ = evaluate(trial)
                if feasible and objective < chosen_obj:
                    chosen = trial
                    chosen_obj = objective
            if chosen is None:
                return None
            sequence = chosen
        return sequence

    # Due-date-oriented insertion constructions.
    insertion_orders = [
        sorted(range(n), key=lambda j: (due_dates[j], j)),
        sorted(range(n), key=lambda j: (due_dates[j] - total_processing[j], j)),
    ]
    for order in insertion_orders:
        if time.monotonic() >= heuristic_deadline:
            break
        candidate = insertion_construct(order)
        if candidate is not None:
            submit(candidate)

    # Randomized insertion orders provide inexpensive diversification.
    while time.monotonic() < heuristic_deadline and n > 1:
        order = list(range(n))
        if rng.random() < 0.7:
            order.sort(
                key=lambda j: (
                    due_dates[j] + rng.uniform(-1.0, 1.0) * max(1, due_dates[j]) * 0.15,
                    j,
                )
            )
        else:
            rng.shuffle(order)
        candidate = insertion_construct(order)
        if candidate is not None:
            submit(candidate)
        if n > 80:
            break

    # Beam search is mainly used to obtain an initial feasible solution when
    # simple constructions fail. A state with smaller arrival time dominates
    # another state having the same visited set and last job.
    if best_sequence is None and time.monotonic() < heuristic_deadline:
        if n <= 62:
            beam_width = 2500 if n <= 25 else 700
            states = []
            for j in range(n):
                t = last_offset[j]
                if t + last_processing[j] <= due_dates[j]:
                    mask = 1 << j
                    slack = due_dates[j] - (t + last_processing[j])
                    states.append((t - 0.15 * slack, t, mask, j, (j,)))
            states.sort(key=lambda x: x[0])
            states = states[:beam_width]

            for depth in range(1, n):
                if time.monotonic() >= heuristic_deadline or not states:
                    break
                next_best = {}
                for _, arrival, mask, last, seq in states:
                    remaining_mask = ((1 << n) - 1) ^ mask
                    while remaining_mask:
                        bit = remaining_mask & -remaining_mask
                        remaining_mask -= bit
                        job = bit.bit_length() - 1
                        new_arrival = arrival + transition[last][job]
                        completion = new_arrival + last_processing[job]
                        if completion > due_dates[job]:
                            continue
                        new_mask = mask | bit
                        key = (new_mask, job)
                        old = next_best.get(key)
                        if old is None or new_arrival < old[1]:
                            slack = due_dates[job] - completion
                            score = new_arrival - 0.15 * slack
                            next_best[key] = (
                                score,
                                new_arrival,
                                new_mask,
                                job,
                                seq + (job,),
                            )
                    if time.monotonic() >= heuristic_deadline:
                        break
                states = sorted(next_best.values(), key=lambda x: x[0])[:beam_width]

            for state in states:
                if len(state[4]) == n:
                    submit(list(state[4]))

    # Feasible local improvement by swaps and relocations.
    if best_sequence is not None and time.monotonic() < heuristic_deadline:
        improved = True
        while improved and time.monotonic() < heuristic_deadline:
            improved = False
            base = list(best_sequence)

            # Adjacent swaps.
            for i in range(n - 1):
                trial = list(base)
                trial[i], trial[i + 1] = trial[i + 1], trial[i]
                if submit(trial):
                    improved = True
                    break
                if time.monotonic() >= heuristic_deadline:
                    break
            if improved:
                continue

            # Relocations, restricted for large instances.
            attempts = 0
            max_attempts = min(n * n, 3000)
            indices = list(range(n))
            rng.shuffle(indices)
            for i in indices:
                if time.monotonic() >= heuristic_deadline:
                    break
                targets = list(range(n))
                rng.shuffle(targets)
                for new_pos in targets:
                    if new_pos == i:
                        continue
                    trial = list(base)
                    job = trial.pop(i)
                    trial.insert(new_pos, job)
                    attempts += 1
                    if submit(trial):
                        improved = True
                        break
                    if attempts >= max_attempts or time.monotonic() >= heuristic_deadline:
                        break
                if improved or attempts >= max_attempts:
                    break

    # Exact MIP: a source-to-sink Hamiltonian path with deadline-constrained
    # last-machine arrival times.
    remaining_time = deadline - time.monotonic()
    if remaining_time > 0.01:
        latest_start = [
            due_dates[j] - last_processing[j] for j in range(n)
        ]

        # A negative latest last-machine start makes the instance infeasible.
        bounds_valid = all(x >= 0 for x in latest_start)

        if bounds_valid:
            model = gp.Model("no_wait_flow_shop")
            model.Params.OutputFlag = 0
            model.Params.Seed = 0
            model.Params.MIPGap = 1e-4
            model.Params.NumericFocus = 0
            model.Params.Threads = 1
            model.Params.TimeLimit = max(0.01, remaining_time)

            source = model.addVars(n, vtype=GRB.BINARY, name="source")
            sink = model.addVars(n, vtype=GRB.BINARY, name="sink")
            arc = {
                (i, j): model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
                for i in range(n)
                for j in range(n)
                if i != j
            }
            arrival = model.addVars(
                n,
                lb=0.0,
                ub={j: latest_start[j] for j in range(n)},
                vtype=GRB.CONTINUOUS,
                name="arrival",
            )

            model.addConstr(gp.quicksum(source[j] for j in range(n)) == 1)
            model.addConstr(gp.quicksum(sink[j] for j in range(n)) == 1)

            for j in range(n):
                model.addConstr(
                    source[j]
                    + gp.quicksum(arc[i, j] for i in range(n) if i != j)
                    == 1
                )
                model.addConstr(
                    sink[j]
                    + gp.quicksum(arc[j, k] for k in range(n) if k != j)
                    == 1
                )
                model.addConstr(arrival[j] >= last_offset[j] * source[j])

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    big_m = latest_start[i] + transition[i][j]
                    model.addConstr(
                        arrival[j]
                        >= arrival[i]
                        + transition[i][j]
                        - big_m * (1 - arc[i, j])
                    )

            objective = (
                gp.quicksum(last_offset[j] * source[j] for j in range(n))
                + gp.quicksum(
                    transition[i][j] * arc[i, j]
                    for i in range(n)
                    for j in range(n)
                    if i != j
                )
                + gp.quicksum(last_processing[j] * sink[j] for j in range(n))
            )
            model.setObjective(objective, GRB.MINIMIZE)

            # Supply the heuristic incumbent as a complete MIP start.
            if best_sequence is not None:
                position = {job: pos for pos, job in enumerate(best_sequence)}
                _, _, start_values = evaluate(best_sequence)

                for j in range(n):
                    source[j].Start = 1.0 if position[j] == 0 else 0.0
                    sink[j].Start = 1.0 if position[j] == n - 1 else 0.0
                    arrival[j].Start = float(start_values[position[j]])

                successor_pairs = set(
                    zip(best_sequence[:-1], best_sequence[1:])
                )
                for key, variable in arc.items():
                    variable.Start = 1.0 if key in successor_pairs else 0.0

            arc_items = list(arc.items())
            source_vars = [source[j] for j in range(n)]

            def extract_sequence(get_value):
                source_values = get_value(source_vars)
                first = max(range(n), key=lambda j: source_values[j])

                successor = {}
                arc_variables = [var for _, var in arc_items]
                values = get_value(arc_variables)
                for ((i, j), _), value in zip(arc_items, values):
                    if value > 0.5:
                        successor[i] = j

                sequence = [first]
                seen = {first}
                while len(sequence) < n:
                    nxt = successor.get(sequence[-1])
                    if nxt is None or nxt in seen:
                        return None
                    sequence.append(nxt)
                    seen.add(nxt)
                return sequence

            def callback(cb_model, where):
                if where != GRB.Callback.MIPSOL:
                    return
                try:
                    sequence = extract_sequence(cb_model.cbGetSolution)
                    if sequence is not None:
                        submit(sequence)
                except gp.GurobiError:
                    pass

            try:
                model.optimize(callback)

                if model.SolCount > 0:
                    def model_values(variables):
                        return [v.X for v in variables]

                    sequence = extract_sequence(model_values)
                    if sequence is not None:
                        submit(sequence)
            except gp.GurobiError:
                pass

    # If no feasible solution was found, emit a complete permutation anyway.
    # This branch is relevant only for infeasible instances or an extremely
    # small runtime limit; feasible benchmark instances normally never use it.
    if best_sequence is None:
        fallback = sorted(range(n), key=lambda j: (due_dates[j], j))
        _, fallback_objective, _ = evaluate(fallback)
        best_sequence = fallback
        best_objective = float(fallback_objective)

    final_solution = {
        "objective_value": float(best_objective),
        "sequence": [j + 1 for j in best_sequence],
    }
    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(final_solution, f)


if __name__ == "__main__":
    main()
import argparse
import json
import math
import os
import time
from functools import cmp_to_key

from solution_logger import SolutionLogger

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception:
    gp = None
    GRB = None


EPS = 1e-9


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    return parser.parse_args()


def read_instance(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_solution(path, solution):
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(solution, f, separators=(",", ":"), allow_nan=False)


def normalized_number(value):
    value = float(value)
    rounded = round(value)
    if abs(value - rounded) <= 1e-8:
        return float(rounded)
    return value


def smith_comparator(machine, processing_times, weights):
    def compare(a, b):
        # p_a / w_a < p_b / w_b, avoiding floating-point division.
        left = processing_times[a][machine] * weights[b]
        right = processing_times[b][machine] * weights[a]
        if left < right - EPS:
            return -1
        if left > right + EPS:
            return 1
        if a < b:
            return -1
        if a > b:
            return 1
        return 0

    return compare


def completion_solution_from_assignment(assignment, processing_times, weights):
    m = len(assignment)
    schedule = {}
    completion_times = {}
    objective = 0.0

    for i in range(m):
        jobs = list(assignment[i])
        jobs.sort(key=cmp_to_key(smith_comparator(i, processing_times, weights)))
        schedule[str(i)] = jobs

        current_time = 0.0
        for j in jobs:
            current_time += processing_times[j][i]
            completion_times[str(j)] = normalized_number(current_time)
            objective += weights[j] * current_time

    return {
        "objective_value": normalized_number(objective),
        "schedule": schedule,
        "completion_times": completion_times,
        "tardy_jobs": [],
        "on_time_jobs": [],
    }


def tardy_solution_from_assignment(assignment, processing_times, weights, due_dates):
    n = len(weights)
    m = len(assignment)
    schedule = {}
    completion_times = {}
    on_time = []
    assigned = set()

    for i in range(m):
        jobs = sorted(assignment[i], key=lambda j: (due_dates[j], j))
        schedule[str(i)] = jobs
        current_time = 0.0

        for j in jobs:
            current_time += processing_times[j][i]
            completion_times[str(j)] = normalized_number(current_time)
            on_time.append(j)
            assigned.add(j)

    on_time.sort()
    tardy = [j for j in range(n) if j not in assigned]
    objective = sum(weights[j] for j in tardy)

    return {
        "objective_value": normalized_number(objective),
        "schedule": schedule,
        "completion_times": completion_times,
        "tardy_jobs": tardy,
        "on_time_jobs": on_time,
    }


def interaction_cost(machine, a, b, processing_times, weights):
    return min(
        processing_times[a][machine] * weights[b],
        processing_times[b][machine] * weights[a],
    )


def assignment_completion_objective(assignment, processing_times, weights):
    total = 0.0
    for i, jobs in enumerate(assignment):
        for j in jobs:
            total += weights[j] * processing_times[j][i]
        for pos, j in enumerate(jobs):
            for k in jobs[pos + 1:]:
                total += interaction_cost(i, j, k, processing_times, weights)
    return total


def greedy_completion_assignment(order, processing_times, weights, exact=True):
    n = len(weights)
    m = len(processing_times[0])
    assignment = [[] for _ in range(m)]

    fastest = [
        sorted(range(m), key=lambda i: (processing_times[j][i], i))
        for j in range(n)
    ]

    for j in order:
        if exact or m <= 6:
            candidates = range(m)
        else:
            candidate_set = set(fastest[j][: min(5, m)])
            candidate_set.add(min(range(m), key=lambda i: (len(assignment[i]), i)))
            candidates = sorted(candidate_set)

        best_machine = None
        best_increment = math.inf

        for i in candidates:
            increment = weights[j] * processing_times[j][i]
            for k in assignment[i]:
                increment += interaction_cost(i, j, k, processing_times, weights)

            if increment < best_increment - EPS or (
                abs(increment - best_increment) <= EPS
                and (best_machine is None or i < best_machine)
            ):
                best_increment = increment
                best_machine = i

        assignment[best_machine].append(j)

    return assignment


def improve_completion_relocations(
    assignment, processing_times, weights, deadline, max_work=3_000_000
):
    n = len(weights)
    m = len(assignment)
    estimated_work = max(1, n * n * m)
    use_all_machines = estimated_work <= max_work

    fastest = [
        sorted(range(m), key=lambda i: (processing_times[j][i], i))
        for j in range(n)
    ]

    improved = True
    passes = 0

    while improved and time.monotonic() < deadline and passes < 5:
        improved = False
        passes += 1

        job_machine = [-1] * n
        for i, jobs in enumerate(assignment):
            for j in jobs:
                job_machine[j] = i

        for j in range(n):
            if time.monotonic() >= deadline:
                break

            source = job_machine[j]
            removal_saving = weights[j] * processing_times[j][source]
            for k in assignment[source]:
                if k != j:
                    removal_saving += interaction_cost(
                        source, j, k, processing_times, weights
                    )

            if use_all_machines:
                targets = range(m)
            else:
                target_set = set(fastest[j][: min(5, m)])
                target_set.add(min(range(m), key=lambda i: (len(assignment[i]), i)))
                target_set.add(source)
                targets = sorted(target_set)

            best_target = source
            best_delta = 0.0

            for target in targets:
                if target == source:
                    continue

                addition = weights[j] * processing_times[j][target]
                for k in assignment[target]:
                    addition += interaction_cost(
                        target, j, k, processing_times, weights
                    )

                delta = addition - removal_saving
                if delta < best_delta - EPS:
                    best_delta = delta
                    best_target = target

            if best_target != source:
                assignment[source].remove(j)
                assignment[best_target].append(j)
                job_machine[j] = best_target
                improved = True

    return assignment


def solve_weighted_completion(
    instance, logger, start_time, time_limit
):
    weights = [float(x) for x in instance["jobs"]["weights"]]
    processing_times = [
        [float(x) for x in row]
        for row in instance["jobs"]["processing_times"]
    ]
    n = len(weights)
    m = instance["num_machines"]
    deadline = start_time + max(0, time_limit)

    min_processing = [min(processing_times[j]) for j in range(n)]
    average_processing = [
        sum(processing_times[j]) / max(1, m) for j in range(n)
    ]
    machine_regret = []
    for j in range(n):
        row = sorted(processing_times[j])
        machine_regret.append(row[1] - row[0] if m > 1 else row[0])

    orders = [
        sorted(
            range(n),
            key=lambda j: (
                -(weights[j] / max(min_processing[j], EPS)),
                -weights[j],
                j,
            ),
        ),
        sorted(
            range(n),
            key=lambda j: (-weights[j], min_processing[j], j),
        ),
        sorted(
            range(n),
            key=lambda j: (
                -(weights[j] * average_processing[j]),
                -weights[j],
                j,
            ),
        ),
        sorted(
            range(n),
            key=lambda j: (-machine_regret[j], -weights[j], j),
        ),
    ]

    best_solution = None
    best_assignment = None
    best_objective = math.inf
    exact_greedy = n * n * m <= 6_000_000

    for order in orders:
        if best_solution is not None and time.monotonic() >= deadline:
            break

        assignment = greedy_completion_assignment(
            order, processing_times, weights, exact=exact_greedy
        )
        solution = completion_solution_from_assignment(
            assignment, processing_times, weights
        )
        value = float(solution["objective_value"])

        if value < best_objective - EPS:
            best_objective = value
            best_solution = solution
            best_assignment = [list(x) for x in assignment]
            if logger:
                logger.log_solution(value, solution)

    if best_solution is None:
        assignment = [[] for _ in range(m)]
        for j in range(n):
            machine = min(
                range(m), key=lambda i: (processing_times[j][i], i)
            )
            assignment[machine].append(j)
        best_assignment = assignment
        best_solution = completion_solution_from_assignment(
            assignment, processing_times, weights
        )
        best_objective = float(best_solution["objective_value"])
        if logger:
            logger.log_solution(best_objective, best_solution)

    if time.monotonic() < deadline:
        local_deadline = min(deadline, time.monotonic() + max(0.05, 0.15 * time_limit))
        improved_assignment = improve_completion_relocations(
            [list(x) for x in best_assignment],
            processing_times,
            weights,
            local_deadline,
        )
        solution = completion_solution_from_assignment(
            improved_assignment, processing_times, weights
        )
        value = float(solution["objective_value"])
        if value < best_objective - EPS:
            best_objective = value
            best_solution = solution
            best_assignment = improved_assignment
            if logger:
                logger.log_solution(value, solution)

    # For moderate instances, improve/prove the solution with a binary
    # quadratic assignment model. Smith's rule fixes the optimal order on
    # each machine, and each same-machine job pair has a fixed interaction.
    pair_terms = m * n * max(0, n - 1) // 2
    remaining = deadline - time.monotonic()

    if (
        gp is not None
        and remaining > 0.25
        and n * m <= 20_000
        and pair_terms <= 180_000
    ):
        try:
            model = gp.Model("weighted_completion_time")
            model.Params.OutputFlag = 0
            model.Params.Seed = 0
            model.Params.MIPGap = 1e-4
            model.Params.NumericFocus = 0
            model.Params.Threads = 1
            model.Params.NonConvex = 2
            model.Params.TimeLimit = max(0.01, remaining)

            x = model.addVars(n, m, vtype=GRB.BINARY, name="x")
            for j in range(n):
                model.addConstr(gp.quicksum(x[j, i] for i in range(m)) == 1)

            objective = gp.QuadExpr()
            for j in range(n):
                for i in range(m):
                    objective += weights[j] * processing_times[j][i] * x[j, i]

            for i in range(m):
                for j in range(n):
                    for k in range(j + 1, n):
                        coefficient = interaction_cost(
                            i, j, k, processing_times, weights
                        )
                        objective += coefficient * x[j, i] * x[k, i]

            model.setObjective(objective, GRB.MINIMIZE)

            incumbent_machine = {}
            for i, jobs in enumerate(best_assignment):
                for j in jobs:
                    incumbent_machine[j] = i

            for j in range(n):
                for i in range(m):
                    x[j, i].Start = 1.0 if incumbent_machine.get(j) == i else 0.0

            callback_best = [best_objective]
            flat_vars = [x[j, i] for j in range(n) for i in range(m)]

            def callback(model_cb, where):
                if where != GRB.Callback.MIPSOL:
                    return
                try:
                    values = model_cb.cbGetSolution(flat_vars)
                    assignment_cb = [[] for _ in range(m)]
                    pos = 0
                    for j_cb in range(n):
                        selected = None
                        selected_value = -math.inf
                        for i_cb in range(m):
                            val = values[pos]
                            pos += 1
                            if val > selected_value:
                                selected_value = val
                                selected = i_cb
                        assignment_cb[selected].append(j_cb)

                    solution_cb = completion_solution_from_assignment(
                        assignment_cb, processing_times, weights
                    )
                    value_cb = float(solution_cb["objective_value"])
                    if value_cb < callback_best[0] - EPS:
                        callback_best[0] = value_cb
                        if logger:
                            logger.log_solution(value_cb, solution_cb)
                except Exception:
                    pass

            model.optimize(callback)

            if model.SolCount > 0:
                assignment_model = [[] for _ in range(m)]
                for j in range(n):
                    chosen = max(range(m), key=lambda i: x[j, i].X)
                    assignment_model[chosen].append(j)

                solution_model = completion_solution_from_assignment(
                    assignment_model, processing_times, weights
                )
                value_model = float(solution_model["objective_value"])
                if value_model < best_objective - EPS:
                    best_objective = value_model
                    best_solution = solution_model
                    if logger:
                        logger.log_solution(value_model, solution_model)
        except Exception:
            pass

    return best_solution


def is_machine_due_date_feasible(
    jobs, machine, processing_times, due_dates
):
    current = 0.0
    for j in sorted(jobs, key=lambda x: (due_dates[x], x)):
        current += processing_times[j][machine]
        if current > due_dates[j] + EPS:
            return False
    return True


def greedy_tardy_edd(processing_times, weights, due_dates):
    n = len(weights)
    m = len(processing_times[0])
    assignment = [[] for _ in range(m)]
    loads = [0.0] * m

    order = sorted(range(n), key=lambda j: (due_dates[j], -weights[j], j))
    for j in order:
        feasible = []
        for i in range(m):
            completion = loads[i] + processing_times[j][i]
            if completion <= due_dates[j] + EPS:
                feasible.append((completion, processing_times[j][i], i))
        if feasible:
            _, _, machine = min(feasible)
            assignment[machine].append(j)
            loads[machine] += processing_times[j][machine]

    return assignment


def greedy_tardy_insertion(
    order, processing_times, weights, due_dates, deadline
):
    m = len(processing_times[0])
    assignment = [[] for _ in range(m)]

    for j in order:
        if time.monotonic() >= deadline:
            break

        best_machine = None
        best_score = None

        for i in range(m):
            trial = assignment[i] + [j]
            if is_machine_due_date_feasible(
                trial, i, processing_times, due_dates
            ):
                ordered = sorted(trial, key=lambda x: (due_dates[x], x))
                load = sum(processing_times[k][i] for k in ordered)
                slack = due_dates[ordered[-1]] - load if ordered else 0.0
                score = (processing_times[j][i], -slack, i)
                if best_score is None or score < best_score:
                    best_score = score
                    best_machine = i

        if best_machine is not None:
            assignment[best_machine].append(j)

    return assignment


def solve_weighted_tardy(instance, logger, start_time, time_limit):
    weights = [float(x) for x in instance["jobs"]["weights"]]
    processing_times = [
        [float(x) for x in row]
        for row in instance["jobs"]["processing_times"]
    ]
    due_dates = [float(x) for x in instance["jobs"]["due_dates"]]
    n = len(weights)
    m = instance["num_machines"]
    deadline = start_time + max(0, time_limit)

    best_assignment = greedy_tardy_edd(
        processing_times, weights, due_dates
    )
    best_solution = tardy_solution_from_assignment(
        best_assignment, processing_times, weights, due_dates
    )
    best_objective = float(best_solution["objective_value"])

    if logger:
        logger.log_solution(best_objective, best_solution)

    min_processing = [min(processing_times[j]) for j in range(n)]
    orders = [
        sorted(range(n), key=lambda j: (-weights[j], due_dates[j], j)),
        sorted(
            range(n),
            key=lambda j: (
                -(weights[j] / max(min_processing[j], EPS)),
                due_dates[j],
                j,
            ),
        ),
        sorted(
            range(n),
            key=lambda j: (
                due_dates[j],
                -(weights[j] / max(min_processing[j], EPS)),
                j,
            ),
        ),
    ]

    if n * n * m <= 4_000_000:
        heuristic_deadline = min(
            deadline, time.monotonic() + max(0.05, 0.15 * time_limit)
        )
        for order in orders:
            if time.monotonic() >= heuristic_deadline:
                break
            assignment = greedy_tardy_insertion(
                order,
                processing_times,
                weights,
                due_dates,
                heuristic_deadline,
            )
            solution = tardy_solution_from_assignment(
                assignment, processing_times, weights, due_dates
            )
            value = float(solution["objective_value"])
            if value < best_objective - EPS:
                best_objective = value
                best_solution = solution
                best_assignment = assignment
                if logger:
                    logger.log_solution(value, solution)

    remaining = deadline - time.monotonic()

    # Cumulative EDD constraints characterize feasibility of the selected
    # on-time jobs on each machine.
    estimated_nonzeros = m * n * (n + 1) // 2
    if (
        gp is not None
        and remaining > 0.25
        and n * m <= 50_000
        and estimated_nonzeros <= 700_000
    ):
        try:
            model = gp.Model("weighted_tardy_jobs")
            model.Params.OutputFlag = 0
            model.Params.Seed = 0
            model.Params.MIPGap = 1e-4
            model.Params.NumericFocus = 0
            model.Params.Threads = 1
            model.Params.TimeLimit = max(0.01, remaining)

            x = model.addVars(n, m, vtype=GRB.BINARY, name="x")

            for j in range(n):
                model.addConstr(gp.quicksum(x[j, i] for i in range(m)) <= 1)

            sorted_jobs = sorted(range(n), key=lambda j: (due_dates[j], j))
            unique_due_dates = sorted(set(due_dates))

            for i in range(m):
                prefix_expr = gp.LinExpr()
                position = 0
                for due in unique_due_dates:
                    while (
                        position < n
                        and due_dates[sorted_jobs[position]] <= due + EPS
                    ):
                        j = sorted_jobs[position]
                        prefix_expr += processing_times[j][i] * x[j, i]
                        position += 1
                    model.addConstr(prefix_expr <= due)

            total_weight = sum(weights)
            saved_weight = gp.quicksum(
                weights[j] * x[j, i] for j in range(n) for i in range(m)
            )
            model.setObjective(total_weight - saved_weight, GRB.MINIMIZE)

            incumbent_machine = {}
            for i, jobs in enumerate(best_assignment):
                for j in jobs:
                    incumbent_machine[j] = i

            for j in range(n):
                for i in range(m):
                    x[j, i].Start = (
                        1.0 if incumbent_machine.get(j) == i else 0.0
                    )

            callback_best = [best_objective]
            flat_vars = [x[j, i] for j in range(n) for i in range(m)]

            def callback(model_cb, where):
                if where != GRB.Callback.MIPSOL:
                    return
                try:
                    values = model_cb.cbGetSolution(flat_vars)
                    assignment_cb = [[] for _ in range(m)]
                    pos = 0

                    for j_cb in range(n):
                        selected = None
                        selected_value = 0.5
                        for i_cb in range(m):
                            val = values[pos]
                            pos += 1
                            if val > selected_value:
                                selected_value = val
                                selected = i_cb
                        if selected is not None:
                            assignment_cb[selected].append(j_cb)

                    solution_cb = tardy_solution_from_assignment(
                        assignment_cb,
                        processing_times,
                        weights,
                        due_dates,
                    )
                    value_cb = float(solution_cb["objective_value"])
                    if value_cb < callback_best[0] - EPS:
                        callback_best[0] = value_cb
                        if logger:
                            logger.log_solution(value_cb, solution_cb)
                except Exception:
                    pass

            model.optimize(callback)

            if model.SolCount > 0:
                assignment_model = [[] for _ in range(m)]
                for j in range(n):
                    values = [x[j, i].X for i in range(m)]
                    selected = max(range(m), key=lambda i: values[i])
                    if values[selected] > 0.5:
                        assignment_model[selected].append(j)

                solution_model = tardy_solution_from_assignment(
                    assignment_model,
                    processing_times,
                    weights,
                    due_dates,
                )
                value_model = float(solution_model["objective_value"])
                if value_model < best_objective - EPS:
                    best_objective = value_model
                    best_solution = solution_model
                    if logger:
                        logger.log_solution(value_model, solution_model)
        except Exception:
            pass

    return best_solution


def main():
    args = parse_args()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    start_time = time.monotonic()
    instance = read_instance(args.instance_path)

    problem_type = instance.get("problem_type")
    objective = instance.get("objective")

    if (
        problem_type == "weighted_completion_time"
        or objective == "minimize_total_weighted_completion_time"
    ):
        solution = solve_weighted_completion(
            instance, logger, start_time, args.time_limit
        )
    elif (
        problem_type == "weighted_tardy_jobs"
        or objective == "minimize_weighted_number_of_tardy_jobs"
    ):
        solution = solve_weighted_tardy(
            instance, logger, start_time, args.time_limit
        )
    else:
        raise ValueError(
            f"Unsupported problem type/objective: {problem_type!r}, {objective!r}"
        )

    write_solution(args.solution_path, solution)


if __name__ == "__main__":
    main()
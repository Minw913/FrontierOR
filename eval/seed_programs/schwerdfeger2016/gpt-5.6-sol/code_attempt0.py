import argparse
import json
import math
import os
import time

from solution_logger import SolutionLogger


def objective_value(loads, average):
    variance_sum = math.fsum((load - average) ** 2 for load in loads)
    return float(math.sqrt(max(0.0, variance_sum)) / average)


def build_solution(job_machine, processing_times, num_machines):
    assignments = [[] for _ in range(num_machines)]
    loads = [0] * num_machines

    for job, machine in enumerate(job_machine):
        assignments[machine].append(job)
        loads[machine] += processing_times[job]

    average = sum(processing_times) / num_machines
    obj = objective_value(loads, average)

    return {
        "objective_value": obj,
        "assignment": {
            str(machine): assignments[machine]
            for machine in range(num_machines)
        },
        "machine_completion_times": {
            str(machine): loads[machine]
            for machine in range(num_machines)
        },
    }


def lpt_assignment(processing_times, num_machines):
    import heapq

    heap = [(0, machine) for machine in range(num_machines)]
    heapq.heapify(heap)
    job_machine = [-1] * len(processing_times)

    for job, processing_time in enumerate(processing_times):
        load, machine = heapq.heappop(heap)
        job_machine[job] = machine
        heapq.heappush(heap, (load + processing_time, machine))

    return job_machine


def target_best_fit_assignment(processing_times, num_machines):
    total = sum(processing_times)
    average = total / num_machines
    loads = [0] * num_machines
    job_machine = [-1] * len(processing_times)

    for job, processing_time in enumerate(processing_times):
        feasible = [
            machine
            for machine in range(num_machines)
            if loads[machine] + processing_time <= average
        ]

        if feasible:
            machine = max(feasible, key=lambda k: (loads[k], -k))
        else:
            machine = min(range(num_machines), key=lambda k: (loads[k], k))

        job_machine[job] = machine
        loads[machine] += processing_time

    return job_machine


def local_search(
    initial_assignment,
    processing_times,
    num_machines,
    deadline,
    incumbent_callback,
):
    n = len(processing_times)
    job_machine = list(initial_assignment)
    loads = [0] * num_machines

    for job, machine in enumerate(job_machine):
        loads[machine] += processing_times[job]

    current_q = sum(load * load for load in loads)
    incumbent_callback(job_machine, loads, current_q)

    max_iterations = 100
    allow_full_swaps = n <= 600

    for _ in range(max_iterations):
        if time.monotonic() >= deadline:
            break

        # Best single-job relocation.
        order = sorted(range(num_machines), key=lambda k: (loads[k], k))
        minimum_machine = order[0]
        second_minimum_machine = order[1]

        best_delta = 0
        best_operation = None

        for job, source in enumerate(job_machine):
            destination = (
                second_minimum_machine
                if source == minimum_machine
                else minimum_machine
            )
            p = processing_times[job]
            source_load = loads[source]
            destination_load = loads[destination]

            delta = (
                (source_load - p) ** 2
                + (destination_load + p) ** 2
                - source_load ** 2
                - destination_load ** 2
            )

            if delta < best_delta:
                best_delta = delta
                best_operation = ("move", job, source, destination)

        # Best pairwise swap for moderate-sized instances.
        if allow_full_swaps and time.monotonic() < deadline:
            checks = 0
            timed_out = False

            for first_job in range(n):
                first_machine = job_machine[first_job]
                first_p = processing_times[first_job]

                for second_job in range(first_job + 1, n):
                    second_machine = job_machine[second_job]
                    if first_machine == second_machine:
                        continue

                    second_p = processing_times[second_job]
                    first_load = loads[first_machine]
                    second_load = loads[second_machine]
                    difference = first_p - second_p

                    delta = (
                        (first_load - difference) ** 2
                        + (second_load + difference) ** 2
                        - first_load ** 2
                        - second_load ** 2
                    )

                    if delta < best_delta:
                        best_delta = delta
                        best_operation = (
                            "swap",
                            first_job,
                            second_job,
                            first_machine,
                            second_machine,
                        )

                    checks += 1
                    if checks % 4096 == 0 and time.monotonic() >= deadline:
                        timed_out = True
                        break

                if timed_out:
                    break

        if best_operation is None:
            break

        if best_operation[0] == "move":
            _, job, source, destination = best_operation
            p = processing_times[job]
            job_machine[job] = destination
            loads[source] -= p
            loads[destination] += p
        else:
            _, first_job, second_job, first_machine, second_machine = best_operation
            first_p = processing_times[first_job]
            second_p = processing_times[second_job]

            job_machine[first_job] = second_machine
            job_machine[second_job] = first_machine
            loads[first_machine] += second_p - first_p
            loads[second_machine] += first_p - second_p

        current_q += best_delta
        incumbent_callback(job_machine, loads, current_q)

    return job_machine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    start_time = time.monotonic()
    deadline = start_time + max(0, args.time_limit)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as input_file:
        instance = json.load(input_file)

    num_machines = int(instance["num_machines"])
    num_jobs = int(instance["num_jobs"])
    processing_times = [int(value) for value in instance["processing_times"]]

    if len(processing_times) != num_jobs:
        raise ValueError("processing_times length does not match num_jobs")

    total_processing_time = sum(processing_times)
    average = total_processing_time / num_machines

    best_assignment = None
    best_loads = None
    best_q = None
    best_solution = None

    def consider_incumbent(job_machine, loads=None, q=None):
        nonlocal best_assignment, best_loads, best_q, best_solution

        if loads is None:
            loads = [0] * num_machines
            for job, machine in enumerate(job_machine):
                loads[machine] += processing_times[job]

        if q is None:
            q = sum(load * load for load in loads)

        if best_q is not None and q >= best_q:
            return False

        best_assignment = list(job_machine)
        best_loads = list(loads)
        best_q = int(q)
        best_solution = build_solution(
            best_assignment, processing_times, num_machines
        )

        if logger:
            logger.log_solution(best_solution["objective_value"], best_solution)

        return True

    # Always create an immediate feasible solution.
    lpt = lpt_assignment(processing_times, num_machines)
    consider_incumbent(lpt)

    if time.monotonic() < deadline:
        target_fit = target_best_fit_assignment(
            processing_times, num_machines
        )
        consider_incumbent(target_fit)

    variable_count = num_jobs * num_machines
    use_gurobi = variable_count <= 250000

    if use_gurobi:
        heuristic_budget = min(
            1.5,
            max(0.05, args.time_limit * 0.08),
        )
    else:
        heuristic_budget = min(
            3.0,
            max(0.05, args.time_limit * 0.8),
        )

    heuristic_deadline = min(deadline, time.monotonic() + heuristic_budget)

    if time.monotonic() < heuristic_deadline:
        local_search(
            lpt,
            processing_times,
            num_machines,
            heuristic_deadline,
            consider_incumbent,
        )

    if time.monotonic() < heuristic_deadline:
        target_fit = target_best_fit_assignment(
            processing_times, num_machines
        )
        local_search(
            target_fit,
            processing_times,
            num_machines,
            heuristic_deadline,
            consider_incumbent,
        )

    # Universal lower bound obtained by distributing the total load as evenly
    # as possible among integer-valued machine loads.
    quotient, remainder = divmod(total_processing_time, num_machines)
    load_balance_lower_bound_q = (
        (num_machines - remainder) * quotient * quotient
        + remainder * (quotient + 1) * (quotient + 1)
    )

    # If the lower bound is attained, the incumbent is globally optimal.
    proven_by_load_bound = best_q == load_balance_lower_bound_q

    if (
        use_gurobi
        and not proven_by_load_bound
        and time.monotonic() < deadline
    ):
        try:
            import gurobipy as gp
            from gurobipy import GRB

            remaining = deadline - time.monotonic()

            if remaining > 0.02:
                model = gp.Model("parallel_machine_squared_deviation")
                model.Params.OutputFlag = 0
                model.Params.Seed = 0
                model.Params.MIPGap = 1e-4
                model.Params.NumericFocus = 0
                model.Params.Threads = 1
                model.Params.TimeLimit = max(0.01, remaining)

                x = model.addVars(
                    num_jobs,
                    num_machines,
                    vtype=GRB.BINARY,
                    name="x",
                )

                incumbent_variance = math.fsum(
                    (load - average) ** 2 for load in best_loads
                )
                max_deviation = math.sqrt(max(0.0, incumbent_variance))

                load_lower_bound = max(0.0, average - max_deviation)
                load_upper_bound = average + max_deviation

                completion = model.addVars(
                    num_machines,
                    lb=load_lower_bound,
                    ub=load_upper_bound,
                    vtype=GRB.CONTINUOUS,
                    name="completion",
                )

                model.addConstrs(
                    (
                        gp.quicksum(x[job, machine] for machine in range(num_machines))
                        == 1
                        for job in range(num_jobs)
                    ),
                    name="assign",
                )

                model.addConstrs(
                    (
                        completion[machine]
                        == gp.quicksum(
                            processing_times[job] * x[job, machine]
                            for job in range(num_jobs)
                        )
                        for machine in range(num_machines)
                    ),
                    name="load",
                )

                # Every optimal solution uses every machine because n > m and
                # all processing times are positive.
                model.addConstrs(
                    (
                        gp.quicksum(
                            x[job, machine] for job in range(num_jobs)
                        )
                        >= 1
                        for machine in range(num_machines)
                    ),
                    name="nonempty",
                )

                # Break machine-label symmetry by sorting completion times.
                model.addConstrs(
                    (
                        completion[machine] >= completion[machine + 1]
                        for machine in range(num_machines - 1)
                    ),
                    name="load_order",
                )

                objective = gp.QuadExpr()
                for machine in range(num_machines):
                    deviation = completion[machine] - average
                    objective += deviation * deviation

                model.setObjective(objective, GRB.MINIMIZE)

                # Relabel the heuristic incumbent to satisfy load ordering.
                ordered_old_machines = sorted(
                    range(num_machines),
                    key=lambda k: (-best_loads[k], k),
                )
                old_to_new = {
                    old_machine: new_machine
                    for new_machine, old_machine in enumerate(ordered_old_machines)
                }
                warm_assignment = [
                    old_to_new[machine] for machine in best_assignment
                ]

                flat_variables = []
                for job in range(num_jobs):
                    for machine in range(num_machines):
                        variable = x[job, machine]
                        variable.Start = (
                            1.0 if warm_assignment[job] == machine else 0.0
                        )
                        flat_variables.append(variable)

                for new_machine, old_machine in enumerate(ordered_old_machines):
                    completion[new_machine].Start = best_loads[old_machine]

                model._flat_variables = flat_variables

                def callback(cb_model, where):
                    if where != GRB.Callback.MIPSOL:
                        return

                    try:
                        values = cb_model.cbGetSolution(
                            cb_model._flat_variables
                        )
                        candidate_assignment = [-1] * num_jobs
                        candidate_loads = [0] * num_machines

                        position = 0
                        for job in range(num_jobs):
                            selected_machine = 0
                            selected_value = -1.0

                            for machine in range(num_machines):
                                value = values[position]
                                position += 1
                                if value > selected_value:
                                    selected_value = value
                                    selected_machine = machine

                            candidate_assignment[job] = selected_machine
                            candidate_loads[selected_machine] += processing_times[job]

                        candidate_q = sum(
                            load * load for load in candidate_loads
                        )
                        consider_incumbent(
                            candidate_assignment,
                            candidate_loads,
                            candidate_q,
                        )
                    except Exception:
                        # Logging or callback extraction failures must not
                        # terminate the optimization.
                        pass

                model.optimize(callback)
                model.dispose()

        except Exception:
            # Retain and output the best heuristic solution if Gurobi cannot
            # be initialized or the model cannot be solved.
            pass

    if best_solution is None:
        best_solution = build_solution(
            lpt, processing_times, num_machines
        )

    solution_directory = os.path.dirname(os.path.abspath(args.solution_path))
    if solution_directory:
        os.makedirs(solution_directory, exist_ok=True)

    with open(args.solution_path, "w", encoding="utf-8") as output_file:
        json.dump(best_solution, output_file, indent=2)


if __name__ == "__main__":
    main()
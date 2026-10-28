import argparse
import copy
import json
import math
import os
import random
import time

from solution_logger import SolutionLogger


def clone_schedule(schedule):
    return [[list(batch) for batch in machine] for machine in schedule]


def evaluate_schedule(schedule, jobs):
    objective = 0
    for machine_batches in schedule:
        available = 0
        for batch in machine_batches:
            if not batch:
                continue
            release = max(jobs[j]["release_date"] for j in batch)
            duration = max(jobs[j]["processing_time"] for j in batch)
            start = max(available, release)
            available = start + duration
            objective += len(batch) * available
    return objective


def build_solution(schedule, jobs):
    batches_output = []
    assignments = {}
    objective = 0

    for machine_index, machine_batches in enumerate(schedule, start=1):
        available = 0
        batch_index = 0

        for batch in machine_batches:
            if not batch:
                continue

            batch_index += 1
            release = max(jobs[j]["release_date"] for j in batch)
            duration = max(jobs[j]["processing_time"] for j in batch)
            start = max(available, release)
            completion = start + duration
            available = completion
            objective += len(batch) * completion

            job_ids = [jobs[j]["job_id"] for j in batch]
            batches_output.append({
                "batch_id": batch_index,
                "machine": machine_index,
                "start_time": start,
                "processing_time": duration,
                "jobs": job_ids
            })

            for j in batch:
                assignments[str(jobs[j]["job_id"])] = {
                    "batch": batch_index,
                    "machine": machine_index,
                    "completion_time": completion
                }

    return {
        "objective_value": float(objective),
        "batches": batches_output,
        "job_assignments": assignments
    }


def group_statistics(group, jobs):
    release = max(jobs[j]["release_date"] for j in group)
    duration = max(jobs[j]["processing_time"] for j in group)
    return release, duration, len(group)


def next_fit_partition(order, jobs, capacity):
    groups = []
    current = []
    current_size = 0

    for j in order:
        size = jobs[j]["size"]
        if current and current_size + size > capacity:
            groups.append(current)
            current = []
            current_size = 0
        current.append(j)
        current_size += size

    if current:
        groups.append(current)
    return groups


def best_fit_partition(order, jobs, capacity):
    groups = []
    group_sizes = []

    for j in order:
        job = jobs[j]
        own_cost = job["release_date"] + job["processing_time"]
        best_group = None
        best_delta = own_cost

        for g_index, group in enumerate(groups):
            if group_sizes[g_index] + job["size"] > capacity:
                continue

            old_r, old_p, old_w = group_statistics(group, jobs)
            new_r = max(old_r, job["release_date"])
            new_p = max(old_p, job["processing_time"])
            delta = (old_w + 1) * (new_r + new_p) - old_w * (old_r + old_p)

            if delta < best_delta:
                best_delta = delta
                best_group = g_index

        if best_group is None:
            groups.append([j])
            group_sizes.append(job["size"])
        else:
            groups[best_group].append(j)
            group_sizes[best_group] += job["size"]

    return groups


def first_fit_partition(order, jobs, capacity):
    groups = []
    group_sizes = []

    for j in order:
        feasible = []
        for g_index, group in enumerate(groups):
            if group_sizes[g_index] + jobs[j]["size"] <= capacity:
                old_r, old_p, old_w = group_statistics(group, jobs)
                new_r = max(old_r, jobs[j]["release_date"])
                new_p = max(old_p, jobs[j]["processing_time"])
                increase = (
                    (old_w + 1) * (new_r + new_p)
                    - old_w * (old_r + old_p)
                )
                feasible.append((increase, capacity - group_sizes[g_index], g_index))

        if feasible:
            _, _, g_index = min(feasible)
            groups[g_index].append(j)
            group_sizes[g_index] += jobs[j]["size"]
        else:
            groups.append([j])
            group_sizes.append(jobs[j]["size"])

    return groups


def schedule_groups(groups, jobs, num_machines, mode):
    data = []
    for group in groups:
        release, duration, weight = group_statistics(group, jobs)
        size = sum(jobs[j]["size"] for j in group)
        data.append((list(group), release, duration, weight, size))

    if mode == 0:
        data.sort(key=lambda x: (x[1], x[2] / x[3], x[2], -x[3]))
    elif mode == 1:
        data.sort(key=lambda x: (x[2] / x[3], x[1], x[2], -x[3]))
    elif mode == 2:
        data.sort(key=lambda x: (x[1] + x[2], x[2] / x[3], -x[3]))
    elif mode == 3:
        data.sort(key=lambda x: (x[1], -x[3], x[2]))
    else:
        data.sort(key=lambda x: (x[2], x[1], -x[3]))

    schedule = [[] for _ in range(num_machines)]
    available = [0] * num_machines

    for group, release, duration, _, _ in data:
        machine = min(
            range(num_machines),
            key=lambda m: (max(available[m], release) + duration,
                           available[m], m)
        )
        schedule[machine].append(group)
        available[machine] = max(available[machine], release) + duration

    return schedule


def job_locations(schedule):
    locations = {}
    for m, machine_batches in enumerate(schedule):
        for b, batch in enumerate(machine_batches):
            for pos, j in enumerate(batch):
                locations[j] = (m, b, pos)
    return locations


def batch_size(batch, jobs):
    return sum(jobs[j]["size"] for j in batch)


def remove_job(schedule, location):
    m, b, pos = location
    result = clone_schedule(schedule)
    result[m][b].pop(pos)
    if not result[m][b]:
        result[m].pop(b)
    return result


def best_job_relocation(schedule, current_value, selected_job,
                        jobs, capacity, deadline):
    locations = job_locations(schedule)
    if selected_job not in locations:
        return schedule, current_value

    base = remove_job(schedule, locations[selected_job])
    best_schedule = schedule
    best_value = current_value
    selected_size = jobs[selected_job]["size"]

    # Insert into an existing batch.
    for m, machine_batches in enumerate(base):
        for b, batch in enumerate(machine_batches):
            if time.monotonic() >= deadline:
                return best_schedule, best_value
            if batch_size(batch, jobs) + selected_size <= capacity:
                candidate = clone_schedule(base)
                candidate[m][b].append(selected_job)
                value = evaluate_schedule(candidate, jobs)
                if value < best_value:
                    best_value = value
                    best_schedule = candidate

    # Create a singleton batch in every possible sequence position.
    for m, machine_batches in enumerate(base):
        for position in range(len(machine_batches) + 1):
            if time.monotonic() >= deadline:
                return best_schedule, best_value
            candidate = clone_schedule(base)
            candidate[m].insert(position, [selected_job])
            value = evaluate_schedule(candidate, jobs)
            if value < best_value:
                best_value = value
                best_schedule = candidate

    return best_schedule, best_value


def best_batch_relocation(schedule, current_value, source_machine,
                          source_batch, jobs, deadline):
    if source_machine >= len(schedule) or source_batch >= len(schedule[source_machine]):
        return schedule, current_value

    moved_batch = list(schedule[source_machine][source_batch])
    base = clone_schedule(schedule)
    base[source_machine].pop(source_batch)

    best_schedule = schedule
    best_value = current_value

    for m, machine_batches in enumerate(base):
        for position in range(len(machine_batches) + 1):
            if time.monotonic() >= deadline:
                return best_schedule, best_value
            candidate = clone_schedule(base)
            candidate[m].insert(position, list(moved_batch))
            value = evaluate_schedule(candidate, jobs)
            if value < best_value:
                best_value = value
                best_schedule = candidate

    return best_schedule, best_value


def best_batch_merge(schedule, current_value, source_machine, source_batch,
                     jobs, capacity, deadline):
    if source_machine >= len(schedule) or source_batch >= len(schedule[source_machine]):
        return schedule, current_value

    source_jobs = list(schedule[source_machine][source_batch])
    source_size = batch_size(source_jobs, jobs)
    base = clone_schedule(schedule)
    base[source_machine].pop(source_batch)

    best_schedule = schedule
    best_value = current_value

    for m, machine_batches in enumerate(base):
        for b, target in enumerate(machine_batches):
            if time.monotonic() >= deadline:
                return best_schedule, best_value
            if source_size + batch_size(target, jobs) <= capacity:
                candidate = clone_schedule(base)
                candidate[m][b].extend(source_jobs)
                value = evaluate_schedule(candidate, jobs)
                if value < best_value:
                    best_value = value
                    best_schedule = candidate

    return best_schedule, best_value


def random_swap_improvement(schedule, current_value, jobs, capacity,
                            rng, attempts, deadline):
    locations = job_locations(schedule)
    all_jobs = list(locations)
    best_schedule = schedule
    best_value = current_value

    if len(all_jobs) < 2:
        return best_schedule, best_value

    for _ in range(attempts):
        if time.monotonic() >= deadline:
            break

        j1, j2 = rng.sample(all_jobs, 2)
        m1, b1, p1 = locations[j1]
        m2, b2, p2 = locations[j2]

        if m1 == m2 and b1 == b2:
            continue

        batch1 = schedule[m1][b1]
        batch2 = schedule[m2][b2]
        new_size1 = batch_size(batch1, jobs) - jobs[j1]["size"] + jobs[j2]["size"]
        new_size2 = batch_size(batch2, jobs) - jobs[j2]["size"] + jobs[j1]["size"]

        if new_size1 > capacity or new_size2 > capacity:
            continue

        candidate = clone_schedule(schedule)
        candidate[m1][b1][p1] = j2
        candidate[m2][b2][p2] = j1
        value = evaluate_schedule(candidate, jobs)

        if value < best_value:
            best_value = value
            best_schedule = candidate

    return best_schedule, best_value


def random_perturbation(schedule, jobs, capacity, rng, moves):
    result = clone_schedule(schedule)
    num_jobs = len(jobs)

    for _ in range(moves):
        if num_jobs == 0:
            break

        locations = job_locations(result)
        j = rng.randrange(num_jobs)
        if j not in locations:
            continue

        base = remove_job(result, locations[j])
        feasible_targets = []

        for m, machine_batches in enumerate(base):
            for b, batch in enumerate(machine_batches):
                if batch_size(batch, jobs) + jobs[j]["size"] <= capacity:
                    feasible_targets.append(("existing", m, b))

        for m, machine_batches in enumerate(base):
            feasible_targets.append(
                ("new", m, rng.randrange(len(machine_batches) + 1))
            )

        if not feasible_targets:
            continue

        kind, m, index = rng.choice(feasible_targets)
        if kind == "existing":
            base[m][index].append(j)
        else:
            base[m].insert(index, [j])
        result = base

    return result


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

    with open(args.instance_path, "r", encoding="utf-8") as f:
        instance = json.load(f)

    num_jobs = int(instance["num_jobs"])
    num_machines = int(instance["num_machines"])
    capacity = int(instance["batch_capacity"])

    raw_jobs = instance["jobs"]
    jobs = []
    for item in raw_jobs:
        jobs.append({
            "job_id": item["job_id"],
            "release_date": int(item["release_date"]),
            "processing_time": int(item["processing_time"]),
            "size": int(item["size"])
        })

    if num_jobs != len(jobs):
        num_jobs = len(jobs)

    if num_machines <= 0:
        raise ValueError("num_machines must be positive")

    for job in jobs:
        if job["size"] > capacity:
            raise ValueError(
                f"Job {job['job_id']} has size larger than batch capacity"
            )

    if num_jobs == 0:
        solution = {
            "objective_value": 0.0,
            "batches": [],
            "job_assignments": {}
        }
        if logger:
            logger.log_solution(0.0, solution)
        os.makedirs(os.path.dirname(os.path.abspath(args.solution_path)), exist_ok=True)
        with open(args.solution_path, "w", encoding="utf-8") as f:
            json.dump(solution, f, indent=2)
        return

    rng = random.Random(0)

    # A fast, always-available initial feasible solution.
    initial_order = sorted(
        range(num_jobs),
        key=lambda j: (
            jobs[j]["release_date"],
            jobs[j]["processing_time"],
            jobs[j]["size"],
            jobs[j]["job_id"]
        )
    )
    singleton_groups = [[j] for j in initial_order]
    best_schedule = schedule_groups(singleton_groups, jobs, num_machines, 0)
    best_value = evaluate_schedule(best_schedule, jobs)

    if logger:
        logger.log_solution(
            float(best_value),
            build_solution(best_schedule, jobs)
        )

    def register_candidate(candidate):
        nonlocal best_schedule, best_value
        value = evaluate_schedule(candidate, jobs)
        if value < best_value:
            best_schedule = clone_schedule(candidate)
            best_value = value
            if logger:
                logger.log_solution(
                    float(best_value),
                    build_solution(best_schedule, jobs)
                )
            return True
        return False

    # Diverse batching orders.
    orders = [
        sorted(range(num_jobs),
               key=lambda j: (jobs[j]["processing_time"],
                              jobs[j]["release_date"],
                              -jobs[j]["size"])),
        sorted(range(num_jobs),
               key=lambda j: (jobs[j]["release_date"],
                              jobs[j]["processing_time"],
                              -jobs[j]["size"])),
        sorted(range(num_jobs),
               key=lambda j: (jobs[j]["release_date"] +
                              jobs[j]["processing_time"],
                              jobs[j]["processing_time"],
                              -jobs[j]["size"])),
        sorted(range(num_jobs),
               key=lambda j: (jobs[j]["processing_time"] /
                              max(1, jobs[j]["size"]),
                              jobs[j]["release_date"])),
        sorted(range(num_jobs),
               key=lambda j: (-jobs[j]["size"],
                              jobs[j]["processing_time"],
                              jobs[j]["release_date"])),
    ]

    construction_budget = min(
        deadline,
        start_time + max(0.05, min(3.0, args.time_limit * 0.25))
    )

    for order in orders:
        if time.monotonic() >= construction_budget:
            break

        partitions = [
            next_fit_partition(order, jobs, capacity),
            best_fit_partition(order, jobs, capacity),
            first_fit_partition(order, jobs, capacity)
        ]

        for groups in partitions:
            for mode in range(5):
                if time.monotonic() >= construction_budget:
                    break
                candidate = schedule_groups(groups, jobs, num_machines, mode)
                register_candidate(candidate)

    # Local improvement and repeated perturbation.
    current_schedule = clone_schedule(best_schedule)
    current_value = best_value
    no_improvement = 0

    while time.monotonic() < deadline:
        operation = rng.random()
        candidate_schedule = current_schedule
        candidate_value = current_value

        if operation < 0.58:
            selected_job = rng.randrange(num_jobs)
            candidate_schedule, candidate_value = best_job_relocation(
                current_schedule, current_value, selected_job,
                jobs, capacity, deadline
            )

        elif operation < 0.75:
            nonempty = [
                (m, b)
                for m, machine_batches in enumerate(current_schedule)
                for b in range(len(machine_batches))
            ]
            if nonempty:
                m, b = rng.choice(nonempty)
                candidate_schedule, candidate_value = best_batch_merge(
                    current_schedule, current_value, m, b,
                    jobs, capacity, deadline
                )

        elif operation < 0.90:
            nonempty = [
                (m, b)
                for m, machine_batches in enumerate(current_schedule)
                for b in range(len(machine_batches))
            ]
            if nonempty:
                m, b = rng.choice(nonempty)
                candidate_schedule, candidate_value = best_batch_relocation(
                    current_schedule, current_value, m, b, jobs, deadline
                )

        else:
            candidate_schedule, candidate_value = random_swap_improvement(
                current_schedule, current_value, jobs, capacity,
                rng, min(80, max(10, num_jobs)), deadline
            )

        if candidate_value < current_value:
            current_schedule = candidate_schedule
            current_value = candidate_value
            no_improvement = 0

            if current_value < best_value:
                best_schedule = clone_schedule(current_schedule)
                best_value = current_value
                if logger:
                    logger.log_solution(
                        float(best_value),
                        build_solution(best_schedule, jobs)
                    )
        else:
            no_improvement += 1

        # Escape relocation/merge local optima.
        if no_improvement >= max(20, min(100, num_jobs)):
            moves = 2 + rng.randrange(4)
            current_schedule = random_perturbation(
                best_schedule, jobs, capacity, rng, moves
            )
            current_value = evaluate_schedule(current_schedule, jobs)
            no_improvement = 0

    final_solution = build_solution(best_schedule, jobs)

    os.makedirs(os.path.dirname(os.path.abspath(args.solution_path)), exist_ok=True)
    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(final_solution, f, indent=2)


if __name__ == "__main__":
    main()
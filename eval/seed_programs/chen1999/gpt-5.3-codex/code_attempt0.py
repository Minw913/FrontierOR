import argparse
import json
import time
import random
import heapq
from copy import deepcopy

from solution_logger import SolutionLogger


def build_processing_times(data):
    n = data["num_jobs"]
    m = data["num_machines"]
    jobs = data["jobs"]
    p = jobs.get("processing_times", None)

    if p is not None:
        return p

    # Fallback if processing_times absent (schema says present, but keep robust)
    machine_type = data.get("machine_type", "unrelated")
    base = jobs.get("base_processing_times", None)
    speeds = (data.get("machines") or {}).get("speeds", None)
    if base is None:
        raise ValueError("processing_times missing and base_processing_times unavailable.")
    if machine_type == "uniform":
        if speeds is None:
            raise ValueError("uniform machine type requires machine speeds.")
        return [[max(1, int(round(base[j] / speeds[i]))) for i in range(m)] for j in range(n)]
    # identical fallback
    return [[base[j] for _ in range(m)] for j in range(n)]


def completion_machine_cost_order(machine, jobset, p, w, cache):
    fs = frozenset(jobset)
    if fs in cache[machine]:
        return cache[machine][fs]

    jobs = list(fs)
    def key_fn(j):
        if w[j] <= 0:
            return (float("inf"), p[j][machine], j)
        return (p[j][machine] / w[j], p[j][machine], j)

    order = sorted(jobs, key=key_fn)
    t = 0.0
    cost = 0.0
    for j in order:
        t += p[j][machine]
        cost += w[j] * t

    cache[machine][fs] = (cost, order)
    return cost, order


def solve_weighted_completion_time(data, time_limit, logger):
    start = time.time()
    deadline = start + max(0, time_limit) - 1e-3

    n = data["num_jobs"]
    m = data["num_machines"]
    w = data["jobs"]["weights"]
    p = build_processing_times(data)

    machine_sets = [set() for _ in range(m)]
    assignment = [-1] * n
    cache = [dict() for _ in range(m)]
    machine_costs = [0.0] * m
    machine_orders = [[] for _ in range(m)]

    # Greedy construction: assign each job to machine with minimum marginal increase.
    jobs_order = sorted(range(n), key=lambda j: (-w[j], min(p[j]), j))
    for j in jobs_order:
        if time.time() > deadline:
            break
        best_m = None
        best_delta = float("inf")
        best_cost = None
        best_order = None

        for mm in range(m):
            new_set = machine_sets[mm] | {j}
            c_new, o_new = completion_machine_cost_order(mm, new_set, p, w, cache)
            delta = c_new - machine_costs[mm]
            if delta < best_delta:
                best_delta = delta
                best_m = mm
                best_cost = c_new
                best_order = o_new

        machine_sets[best_m].add(j)
        assignment[j] = best_m
        machine_costs[best_m] = best_cost
        machine_orders[best_m] = best_order

    # If some jobs not assigned because time was tiny, place quickly
    for j in range(n):
        if assignment[j] == -1:
            best_m = min(range(m), key=lambda mm: p[j][mm])
            machine_sets[best_m].add(j)
            assignment[j] = best_m
    # Recompute all costs/orders to ensure consistency
    for mm in range(m):
        c, o = completion_machine_cost_order(mm, machine_sets[mm], p, w, cache)
        machine_costs[mm], machine_orders[mm] = c, o

    current_obj = sum(machine_costs)
    best_obj = current_obj
    if logger:
        logger.log(best_obj)

    # Local search: job moves (first-improvement)
    improved = True
    while improved and time.time() <= deadline:
        improved = False
        job_scan = list(range(n))
        random.shuffle(job_scan)

        for j in job_scan:
            if time.time() > deadline:
                break
            m_from = assignment[j]
            old_from = machine_costs[m_from]
            set_from_minus = machine_sets[m_from] - {j}
            c_from_new, o_from_new = completion_machine_cost_order(m_from, set_from_minus, p, w, cache)

            best_delta = 0.0
            best_to = None
            best_to_cost = None
            best_to_order = None

            for m_to in range(m):
                if m_to == m_from:
                    continue
                old_to = machine_costs[m_to]
                set_to_plus = machine_sets[m_to] | {j}
                c_to_new, o_to_new = completion_machine_cost_order(m_to, set_to_plus, p, w, cache)

                delta = (c_from_new + c_to_new) - (old_from + old_to)
                if delta < best_delta - 1e-12:
                    best_delta = delta
                    best_to = m_to
                    best_to_cost = c_to_new
                    best_to_order = o_to_new

            if best_to is not None:
                # Apply improving move
                machine_sets[m_from].remove(j)
                machine_sets[best_to].add(j)
                assignment[j] = best_to

                machine_costs[m_from] = c_from_new
                machine_orders[m_from] = o_from_new
                machine_costs[best_to] = best_to_cost
                machine_orders[best_to] = best_to_order

                current_obj += best_delta
                improved = True

                if current_obj + 1e-9 < best_obj:
                    best_obj = current_obj
                    if logger:
                        logger.log(best_obj)

    # Build output schedule & completion times
    schedule = {str(mm): [int(j) for j in machine_orders[mm]] for mm in range(m)}
    completion_times = {}
    for mm in range(m):
        t = 0.0
        for j in machine_orders[mm]:
            t += p[j][mm]
            completion_times[str(j)] = t

    # objective recompute for numerical cleanliness
    obj = 0.0
    for j in range(n):
        obj += w[j] * completion_times[str(j)]

    return {
        "objective_value": float(obj),
        "schedule": schedule,
        "completion_times": completion_times,
        "tardy_jobs": [],
        "on_time_jobs": [int(j) for j in range(n)],
    }


def weighted_moore_machine(machine, jobset, p, d, w, cache):
    fs = frozenset(jobset)
    if fs in cache[machine]:
        return cache[machine][fs]

    if not fs:
        cache[machine][fs] = ([], 0.0)
        return [], 0.0

    jobs = sorted(list(fs), key=lambda j: (d[j], -w[j], j))
    selected = set()
    heap = []  # min-heap by "remove priority": smaller w/p removed first
    t = 0.0

    for j in jobs:
        pj = p[j][machine]
        t += pj
        selected.add(j)
        score = (w[j] / pj) if pj > 0 else float("inf")
        heapq.heappush(heap, (score, j, pj))

        while t > d[j] and heap:
            _, rj, rp = heapq.heappop(heap)
            if rj in selected:
                selected.remove(rj)
                t -= rp

    order = [j for j in jobs if j in selected]
    total_w = float(sum(w[j] for j in order))
    cache[machine][fs] = (order, total_w)
    return order, total_w


def solve_weighted_tardy_jobs(data, time_limit, logger):
    start = time.time()
    deadline = start + max(0, time_limit) - 1e-3

    n = data["num_jobs"]
    m = data["num_machines"]
    w = data["jobs"]["weights"]
    d = data["jobs"]["due_dates"]
    p = build_processing_times(data)

    total_weight = float(sum(w))

    ontime_sets = [set() for _ in range(m)]
    orders = [[] for _ in range(m)]
    machine_ontime_weight = [0.0] * m
    cache = [dict() for _ in range(m)]

    # Initial greedy construction: add jobs if positive gain
    jobs_order = sorted(range(n), key=lambda j: (-w[j], min(p[j]), d[j], j))
    for j in jobs_order:
        if time.time() > deadline:
            break
        best_gain = 0.0
        best_m = None
        best_order = None
        best_w = None

        for mm in range(m):
            cand = ontime_sets[mm] | {j}
            o_new, w_new = weighted_moore_machine(mm, cand, p, d, w, cache)
            gain = w_new - machine_ontime_weight[mm]
            if gain > best_gain + 1e-12:
                best_gain = gain
                best_m = mm
                best_order = o_new
                best_w = w_new

        if best_m is not None:
            ontime_sets[best_m] = set(best_order)
            orders[best_m] = best_order
            machine_ontime_weight[best_m] = best_w

    current_obj = total_weight - sum(machine_ontime_weight)
    best_obj = current_obj
    if logger:
        logger.log(best_obj)

    # Local improvement: move job (tardy or on-time) to another machine if improves on-time weight
    improved = True
    while improved and time.time() <= deadline:
        improved = False
        job_to_machine = {}
        for mm in range(m):
            for j in ontime_sets[mm]:
                job_to_machine[j] = mm

        scan_jobs = list(range(n))
        random.shuffle(scan_jobs)

        for j in scan_jobs:
            if time.time() > deadline:
                break
            cur = job_to_machine.get(j, None)

            best_gain = 0.0
            best_target = None
            best_cur_order = None
            best_cur_w = None
            best_tgt_order = None
            best_tgt_w = None

            if cur is None:
                # tardy -> some machine
                for mm in range(m):
                    cand = ontime_sets[mm] | {j}
                    o_new, w_new = weighted_moore_machine(mm, cand, p, d, w, cache)
                    gain = w_new - machine_ontime_weight[mm]
                    if gain > best_gain + 1e-12:
                        best_gain = gain
                        best_target = mm
                        best_tgt_order = o_new
                        best_tgt_w = w_new
            else:
                # on-time move cur -> mm
                cur_minus = ontime_sets[cur] - {j}
                o_cur_new, w_cur_new = weighted_moore_machine(cur, cur_minus, p, d, w, cache)

                for mm in range(m):
                    if mm == cur:
                        continue
                    cand = ontime_sets[mm] | {j}
                    o_t_new, w_t_new = weighted_moore_machine(mm, cand, p, d, w, cache)
                    gain = (w_cur_new - machine_ontime_weight[cur]) + (w_t_new - machine_ontime_weight[mm])
                    if gain > best_gain + 1e-12:
                        best_gain = gain
                        best_target = mm
                        best_cur_order = o_cur_new
                        best_cur_w = w_cur_new
                        best_tgt_order = o_t_new
                        best_tgt_w = w_t_new

            if best_target is not None and best_gain > 1e-12:
                if cur is None:
                    mm = best_target
                    ontime_sets[mm] = set(best_tgt_order)
                    orders[mm] = best_tgt_order
                    machine_ontime_weight[mm] = best_tgt_w
                else:
                    ontime_sets[cur] = set(best_cur_order)
                    orders[cur] = best_cur_order
                    machine_ontime_weight[cur] = best_cur_w

                    mm = best_target
                    ontime_sets[mm] = set(best_tgt_order)
                    orders[mm] = best_tgt_order
                    machine_ontime_weight[mm] = best_tgt_w

                current_obj = total_weight - sum(machine_ontime_weight)
                improved = True
                if current_obj + 1e-9 < best_obj:
                    best_obj = current_obj
                    if logger:
                        logger.log(best_obj)

    # Finalize schedules (recompute from sets for consistency)
    for mm in range(m):
        o, ww = weighted_moore_machine(mm, ontime_sets[mm], p, d, w, cache)
        orders[mm] = o
        ontime_sets[mm] = set(o)
        machine_ontime_weight[mm] = ww

    schedule = {str(mm): [int(j) for j in orders[mm]] for mm in range(m)}

    completion_times = {}
    on_time_set = set()
    for mm in range(m):
        t = 0.0
        for j in orders[mm]:
            t += p[j][mm]
            completion_times[str(j)] = t
            on_time_set.add(j)

    on_time_jobs = sorted(list(on_time_set))
    tardy_jobs = sorted([j for j in range(n) if j not in on_time_set])
    obj = float(sum(w[j] for j in tardy_jobs))

    return {
        "objective_value": obj,
        "schedule": schedule,
        "completion_times": completion_times,
        "tardy_jobs": [int(j) for j in tardy_jobs],
        "on_time_jobs": [int(j) for j in on_time_jobs],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    random.seed(0)

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    objective_name = data.get("objective", "")
    problem_type = data.get("problem_type", "")

    tardy_variant = (
        objective_name == "minimize_weighted_number_of_tardy_jobs"
        or problem_type == "weighted_tardy_jobs"
    )

    if tardy_variant:
        solution = solve_weighted_tardy_jobs(data, args.time_limit, logger)
    else:
        solution = solve_weighted_completion_time(data, args.time_limit, logger)

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()
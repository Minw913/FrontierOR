import os
import sys

# Suppress D-Bus and GTK warnings before any other imports
os.environ['DBUS_SESSION_BUS_ADDRESS'] = '/dev/null'
os.environ['GIO_USE_VFS'] = 'local'
os.environ['NO_AT_BRIDGE'] = '1'

import argparse
import json
import time
import random
import math
import warnings
warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    return parser.parse_args()


def read_instance(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def compute_machine_wt(machine_jobs, proc, weight, due):
    wt = 0
    t = 0
    for jid in machine_jobs:
        t += proc[jid]
        tardiness = t - due[jid]
        if tardiness > 0:
            wt += weight[jid] * tardiness
    return wt


def compute_total_wt(machines_seq, proc, weight, due):
    total = 0
    for seq in machines_seq:
        total += compute_machine_wt(seq, proc, weight, due)
    return total


def atc_schedule(job_ids, m, proc, weight, due, k_param=1.0):
    n = len(job_ids)
    if n == 0:
        return [[] for _ in range(m)]
    p_avg = sum(proc[j] for j in job_ids) / n

    machines_seq = [[] for _ in range(m)]
    machines_time = [0] * m
    remaining = list(job_ids)

    while remaining:
        mi = min(range(m), key=lambda i: machines_time[i])
        t = machines_time[mi]

        best_job_idx = -1
        best_priority = -1.0

        denom = k_param * p_avg if k_param * p_avg > 0 else 1e-10

        for idx, jid in enumerate(remaining):
            p = proc[jid]
            w = weight[jid]
            d = due[jid]

            slack = d - p - t
            if slack < 0:
                slack = 0
            priority = (w / max(p, 1e-10)) * math.exp(-slack / denom)

            if priority > best_priority:
                best_priority = priority
                best_job_idx = idx

        jid = remaining.pop(best_job_idx)
        machines_seq[mi].append(jid)
        machines_time[mi] += proc[jid]

    return machines_seq


def wspt_schedule(job_ids, m, proc, weight):
    sorted_jobs = sorted(job_ids, key=lambda j: proc[j] / max(weight[j], 1e-10))

    machines_seq = [[] for _ in range(m)]
    machines_time = [0] * m

    for jid in sorted_jobs:
        mi = min(range(m), key=lambda i: machines_time[i])
        machines_seq[mi].append(jid)
        machines_time[mi] += proc[jid]

    return machines_seq


def wedd_schedule(job_ids, m, proc, weight, due):
    sorted_jobs = sorted(job_ids, key=lambda j: due[j] / max(weight[j], 1e-10))

    machines_seq = [[] for _ in range(m)]
    machines_time = [0] * m

    for jid in sorted_jobs:
        mi = min(range(m), key=lambda i: machines_time[i])
        machines_seq[mi].append(jid)
        machines_time[mi] += proc[jid]

    return machines_seq


def optimize_single_machine_swap(seq, proc, weight, due):
    if len(seq) <= 1:
        return list(seq)

    best_seq = list(seq)
    best_wt = compute_machine_wt(best_seq, proc, weight, due)
    improved = True

    while improved:
        improved = False
        for i in range(len(best_seq) - 1):
            best_seq[i], best_seq[i+1] = best_seq[i+1], best_seq[i]
            new_wt = compute_machine_wt(best_seq, proc, weight, due)
            if new_wt < best_wt:
                best_wt = new_wt
                improved = True
            else:
                best_seq[i], best_seq[i+1] = best_seq[i+1], best_seq[i]

    return best_seq


def optimize_single_machine_insert(seq, proc, weight, due):
    if len(seq) <= 1:
        return list(seq)

    best_seq = list(seq)
    best_wt = compute_machine_wt(best_seq, proc, weight, due)
    improved = True

    while improved:
        improved = False
        for i in range(len(best_seq)):
            job = best_seq[i]
            remaining = best_seq[:i] + best_seq[i+1:]
            for j in range(len(remaining) + 1):
                if j == i:
                    continue
                new_seq = remaining[:j] + [job] + remaining[j:]
                new_wt = compute_machine_wt(new_seq, proc, weight, due)
                if new_wt < best_wt:
                    best_seq = new_seq
                    best_wt = new_wt
                    improved = True
                    break
            if improved:
                break

    return best_seq


def local_search_move(machines_seq, proc, weight, due, deadline):
    m = len(machines_seq)
    best_seq = [list(s) for s in machines_seq]

    machine_wt = [compute_machine_wt(best_seq[i], proc, weight, due) for i in range(m)]
    best_obj = sum(machine_wt)

    improved = True
    while improved and time.time() < deadline:
        improved = False
        for mi in range(m):
            if time.time() >= deadline:
                break
            for i in range(len(best_seq[mi])):
                if time.time() >= deadline:
                    break
                job = best_seq[mi][i]
                new_mi = best_seq[mi][:i] + best_seq[mi][i+1:]
                new_wt_mi = compute_machine_wt(new_mi, proc, weight, due)

                best_delta = 0
                best_move = None

                for mj in range(m):
                    if mj == mi:
                        continue
                    for pos in range(len(best_seq[mj]) + 1):
                        new_mj = best_seq[mj][:pos] + [job] + best_seq[mj][pos:]
                        new_wt_mj = compute_machine_wt(new_mj, proc, weight, due)
                        delta = (new_wt_mi + new_wt_mj) - (machine_wt[mi] + machine_wt[mj])
                        if delta < best_delta:
                            best_delta = delta
                            best_move = (mj, pos, new_mi, new_mj, new_wt_mi, new_wt_mj)

                if best_move is not None:
                    mj, pos, new_mi_seq, new_mj_seq, nwt_mi, nwt_mj = best_move
                    best_seq[mi] = new_mi_seq
                    best_seq[mj] = new_mj_seq
                    machine_wt[mi] = nwt_mi
                    machine_wt[mj] = nwt_mj
                    best_obj += best_delta
                    improved = True
                    break
            if improved:
                break

    return best_seq, best_obj


def local_search_swap(machines_seq, proc, weight, due, deadline):
    m = len(machines_seq)
    best_seq = [list(s) for s in machines_seq]

    machine_wt = [compute_machine_wt(best_seq[i], proc, weight, due) for i in range(m)]
    best_obj = sum(machine_wt)

    improved = True
    while improved and time.time() < deadline:
        improved = False
        for mi in range(m):
            if time.time() >= deadline:
                break
            for mj in range(mi + 1, m):
                if time.time() >= deadline:
                    break
                for i in range(len(best_seq[mi])):
                    if time.time() >= deadline:
                        break
                    for j in range(len(best_seq[mj])):
                        new_mi = list(best_seq[mi])
                        new_mj = list(best_seq[mj])
                        new_mi[i], new_mj[j] = new_mj[j], new_mi[i]

                        new_wt_mi = compute_machine_wt(new_mi, proc, weight, due)
                        new_wt_mj = compute_machine_wt(new_mj, proc, weight, due)

                        delta = (new_wt_mi + new_wt_mj) - (machine_wt[mi] + machine_wt[mj])
                        if delta < 0:
                            best_seq[mi] = new_mi
                            best_seq[mj] = new_mj
                            machine_wt[mi] = new_wt_mi
                            machine_wt[mj] = new_wt_mj
                            best_obj += delta
                            improved = True

    return best_seq, best_obj


def perturbation(machines_seq, proc, rng, strength=2):
    m = len(machines_seq)
    new_seq = [list(s) for s in machines_seq]

    for _ in range(strength):
        non_empty = [i for i in range(m) if len(new_seq[i]) > 0]
        if len(non_empty) < 1:
            break

        action = rng.random()

        if action < 0.5 and len(non_empty) >= 2:
            mi, mj = rng.sample(non_empty, 2)
            if len(new_seq[mi]) > 0 and len(new_seq[mj]) > 0:
                i = rng.randint(0, len(new_seq[mi]) - 1)
                j = rng.randint(0, len(new_seq[mj]) - 1)
                new_seq[mi][i], new_seq[mj][j] = new_seq[mj][j], new_seq[mi][i]
        else:
            mi = rng.choice(non_empty)
            if len(new_seq[mi]) > 0:
                i = rng.randint(0, len(new_seq[mi]) - 1)
                job = new_seq[mi].pop(i)
                mj = rng.randint(0, m - 1)
                pos = rng.randint(0, len(new_seq[mj]))
                new_seq[mj].insert(pos, job)

    return new_seq


def build_solution_dict(machines_seq, proc, weight, due, m):
    schedule = []
    total_wt = 0

    for mi in range(m):
        t = 0
        prev_job = 0
        for jid in machines_seq[mi]:
            start = t
            ct = start + proc[jid]
            tardiness = max(0, ct - due[jid])
            wt = weight[jid] * tardiness
            total_wt += wt

            schedule.append({
                'machine_id': mi,
                'prev_job': prev_job,
                'job': jid,
                'start_time': start,
                'completion_time': ct,
                'weighted_tardiness': wt
            })

            prev_job = jid
            t = ct

    return {
        'objective_value': total_wt,
        'schedule': schedule
    }


def solve(instance, time_limit, logger):
    start_time = time.time()
    deadline = start_time + time_limit - 1.0

    jobs = instance['jobs']
    n = instance['parameters']['n']
    m = instance['parameters']['m']

    max_id = max(j['id'] for j in jobs)
    proc = [0] * (max_id + 1)
    weight = [0] * (max_id + 1)
    due = [0] * (max_id + 1)
    job_ids = []

    for j in jobs:
        jid = j['id']
        proc[jid] = j['processing_time']
        weight[jid] = j['weight']
        due[jid] = j['due_date']
        job_ids.append(jid)

    best_seq = None
    best_obj = float('inf')

    # Try different ATC k values
    for k in [0.1, 0.5, 1.0, 2.0, 3.0, 5.0]:
        if time.time() >= deadline:
            break
        try:
            seq = atc_schedule(job_ids, m, proc, weight, due, k)
            for mi in range(m):
                seq[mi] = optimize_single_machine_swap(seq[mi], proc, weight, due)
            obj = compute_total_wt(seq, proc, weight, due)
            if obj < best_obj:
                best_obj = obj
                best_seq = seq
                sol = build_solution_dict(best_seq, proc, weight, due, m)
                if logger:
                    logger.log_solution(best_obj, sol)
        except Exception:
            pass

    # WSPT
    if time.time() < deadline:
        try:
            seq = wspt_schedule(job_ids, m, proc, weight)
            for mi in range(m):
                seq[mi] = optimize_single_machine_swap(seq[mi], proc, weight, due)
            obj = compute_total_wt(seq, proc, weight, due)
            if obj < best_obj:
                best_obj = obj
                best_seq = seq
                sol = build_solution_dict(best_seq, proc, weight, due, m)
                if logger:
                    logger.log_solution(best_obj, sol)
        except Exception:
            pass

    # WEDD
    if time.time() < deadline:
        try:
            seq = wedd_schedule(job_ids, m, proc, weight, due)
            for mi in range(m):
                seq[mi] = optimize_single_machine_swap(seq[mi], proc, weight, due)
            obj = compute_total_wt(seq, proc, weight, due)
            if obj < best_obj:
                best_obj = obj
                best_seq = seq
                sol = build_solution_dict(best_seq, proc, weight, due, m)
                if logger:
                    logger.log_solution(best_obj, sol)
        except Exception:
            pass

    if best_seq is None:
        best_seq = [[] for _ in range(m)]
        for i, jid in enumerate(job_ids):
            best_seq[i % m].append(jid)
        best_obj = compute_total_wt(best_seq, proc, weight, due)
        sol = build_solution_dict(best_seq, proc, weight, due, m)
        if logger:
            logger.log_solution(best_obj, sol)

    # Insertion-based single machine optimization
    if time.time() < deadline and n <= 500:
        for mi in range(m):
            if time.time() >= deadline:
                break
            best_seq[mi] = optimize_single_machine_insert(best_seq[mi], proc, weight, due)
        new_obj = compute_total_wt(best_seq, proc, weight, due)
        if new_obj < best_obj:
            best_obj = new_obj
            sol = build_solution_dict(best_seq, proc, weight, due, m)
            if logger:
                logger.log_solution(best_obj, sol)

    # Inter-machine move local search
    phase_deadline = min(deadline, start_time + time_limit * 0.3)
    if time.time() < phase_deadline:
        best_seq, new_obj = local_search_move(best_seq, proc, weight, due, phase_deadline)
        if new_obj < best_obj:
            best_obj = new_obj
            sol = build_solution_dict(best_seq, proc, weight, due, m)
            if logger:
                logger.log_solution(best_obj, sol)

    # Inter-machine swap local search
    phase_deadline = min(deadline, start_time + time_limit * 0.4)
    if time.time() < phase_deadline:
        best_seq, new_obj = local_search_swap(best_seq, proc, weight, due, phase_deadline)
        if new_obj < best_obj:
            best_obj = new_obj
            sol = build_solution_dict(best_seq, proc, weight, due, m)
            if logger:
                logger.log_solution(best_obj, sol)

    # Optimize each machine again
    if time.time() < deadline and n <= 500:
        for mi in range(m):
            if time.time() >= deadline:
                break
            best_seq[mi] = optimize_single_machine_insert(best_seq[mi], proc, weight, due)
        new_obj = compute_total_wt(best_seq, proc, weight, due)
        if new_obj < best_obj:
            best_obj = new_obj
            sol = build_solution_dict(best_seq, proc, weight, due, m)
            if logger:
                logger.log_solution(best_obj, sol)

    # ILS
    rng = random.Random(42)
    global_best_seq = [list(s) for s in best_seq]
    global_best_obj = best_obj

    no_improve_count = 0

    while time.time() < deadline:
        strength = 2 if no_improve_count <= 20 else min(4, max(2, n // 10))

        perturbed_seq = perturbation(best_seq, proc, rng, strength)

        for mi in range(m):
            if time.time() >= deadline:
                break
            perturbed_seq[mi] = optimize_single_machine_swap(perturbed_seq[mi], proc, weight, due)

        if time.time() >= deadline:
            break

        # Quick move search
        iter_deadline = min(deadline, time.time() + max(0.5, (deadline - time.time()) * 0.1))
        perturbed_seq, perturbed_obj = local_search_move(perturbed_seq, proc, weight, due, iter_deadline)

        if time.time() >= deadline:
            break

        # Single machine insertion optimization (skip for very large instances)
        if n <= 300:
            for mi in range(m):
                if time.time() >= deadline:
                    break
                perturbed_seq[mi] = optimize_single_machine_insert(perturbed_seq[mi], proc, weight, due)
            perturbed_obj = compute_total_wt(perturbed_seq, proc, weight, due)

        if perturbed_obj < best_obj:
            best_seq = perturbed_seq
            best_obj = perturbed_obj
            no_improve_count = 0

            if best_obj < global_best_obj:
                global_best_obj = best_obj
                global_best_seq = [list(s) for s in best_seq]
                sol = build_solution_dict(global_best_seq, proc, weight, due, m)
                if logger:
                    logger.log_solution(global_best_obj, sol)
        else:
            no_improve_count += 1
            if no_improve_count > 50:
                best_seq = [list(s) for s in global_best_seq]
                best_obj = global_best_obj
                no_improve_count = 0

    solution = build_solution_dict(global_best_seq, proc, weight, due, m)
    return solution


def main():
    args = parse_args()

    logger = None
    if args.log_path:
        try:
            from solution_logger import SolutionLogger
            logger = SolutionLogger(args.log_path, sense="minimize")
        except Exception:
            logger = None

    instance = read_instance(args.instance_path)

    solution = solve(instance, args.time_limit, logger)

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == '__main__':
    main()
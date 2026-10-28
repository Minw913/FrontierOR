import argparse
import json
import time
import sys

# Try to import Gurobi
try:
    import gurobipy as gp
    from gurobipy import GRB
    HAS_GUROBI = True
except ImportError:
    HAS_GUROBI = False

from solution_logger import SolutionLogger


def read_instance(path):
    with open(path, 'r') as f:
        data = json.load(f)
    jobs = []
    for j in data['jobs']:
        jobs.append({
            'job_id': j['job_id'],
            'r': j['release_date'],
            'p': j['processing_time'],
            's': j['size']
        })
    return {
        'num_jobs': data['num_jobs'],
        'num_machines': data['num_machines'],
        'batch_capacity': data['batch_capacity'],
        'jobs': jobs
    }


# ----------------- Batch selection rules -----------------
def select_batch_spt_full(available, jobs, Cap):
    sorted_jobs = sorted(available, key=lambda j: jobs[j]['p'])
    selected = []
    total_size = 0
    for j in sorted_jobs:
        if total_size + jobs[j]['s'] <= Cap:
            selected.append(j)
            total_size += jobs[j]['s']
    return selected


def select_batch_lpt_full(available, jobs, Cap):
    sorted_jobs = sorted(available, key=lambda j: jobs[j]['p'], reverse=True)
    selected = []
    total_size = 0
    for j in sorted_jobs:
        if total_size + jobs[j]['s'] <= Cap:
            selected.append(j)
            total_size += jobs[j]['s']
    return selected


def select_batch_spt_threshold(available, jobs, Cap, factor):
    sorted_jobs = sorted(available, key=lambda j: jobs[j]['p'])
    selected = []
    total_size = 0
    current_max = 0
    for j in sorted_jobs:
        if current_max == 0:
            if total_size + jobs[j]['s'] <= Cap:
                selected.append(j)
                total_size += jobs[j]['s']
                current_max = jobs[j]['p']
        else:
            if jobs[j]['p'] <= current_max * factor and total_size + jobs[j]['s'] <= Cap:
                selected.append(j)
                total_size += jobs[j]['s']
                current_max = max(current_max, jobs[j]['p'])
            else:
                break
    return selected


def select_batch_lpt_threshold(available, jobs, Cap, factor):
    sorted_jobs = sorted(available, key=lambda j: jobs[j]['p'], reverse=True)
    selected = []
    total_size = 0
    current_min = float('inf')
    for j in sorted_jobs:
        if current_min == float('inf'):
            if total_size + jobs[j]['s'] <= Cap:
                selected.append(j)
                total_size += jobs[j]['s']
                current_min = jobs[j]['p']
        else:
            if jobs[j]['p'] >= current_min * factor and total_size + jobs[j]['s'] <= Cap:
                selected.append(j)
                total_size += jobs[j]['s']
                current_min = min(current_min, jobs[j]['p'])
            else:
                break
    return selected


def select_batch_erf_full(available, jobs, Cap):
    sorted_jobs = sorted(available, key=lambda j: (jobs[j]['r'], jobs[j]['p']))
    selected = []
    total_size = 0
    for j in sorted_jobs:
        if total_size + jobs[j]['s'] <= Cap:
            selected.append(j)
            total_size += jobs[j]['s']
    return selected


# ----------------- Simulation based heuristic -----------------
def build_schedule(jobs, M, Cap, rule):
    N = len(jobs)
    unscheduled = set(range(N))
    machine_free = [0] * M
    schedule = [[] for _ in range(M)]

    while unscheduled:
        t = min(machine_free)
        available = [j for j in unscheduled if jobs[j]['r'] <= t]
        if not available:
            if not unscheduled:
                break
            next_r = min(jobs[j]['r'] for j in unscheduled)
            t = max(t, next_r)
            available = [j for j in unscheduled if jobs[j]['r'] <= t]
            if not available:
                continue

        free_machines = [m for m in range(M) if machine_free[m] <= t]
        if not free_machines:
            t = min(machine_free)
            free_machines = [m for m in range(M) if machine_free[m] <= t]

        m = min(free_machines, key=lambda i: machine_free[i])
        start_time = t

        # wait_full logic
        if rule == 'wait_full':
            total_size = sum(jobs[j]['s'] for j in available)
            future = any(jobs[j]['r'] > t for j in unscheduled)
            if total_size < Cap * 0.8 and future:
                next_r = min(jobs[j]['r'] for j in unscheduled if jobs[j]['r'] > t)
                machine_free[m] = next_r
                continue

        if rule == 'spt_full':
            selected = select_batch_spt_full(available, jobs, Cap)
        elif rule == 'lpt_full':
            selected = select_batch_lpt_full(available, jobs, Cap)
        elif rule == 'spt_thr_1.5':
            selected = select_batch_spt_threshold(available, jobs, Cap, 1.5)
        elif rule == 'spt_thr_2.0':
            selected = select_batch_spt_threshold(available, jobs, Cap, 2.0)
        elif rule == 'spt_thr_3.0':
            selected = select_batch_spt_threshold(available, jobs, Cap, 3.0)
        elif rule == 'lpt_thr_0.5':
            selected = select_batch_lpt_threshold(available, jobs, Cap, 0.5)
        elif rule == 'lpt_thr_0.7':
            selected = select_batch_lpt_threshold(available, jobs, Cap, 0.7)
        elif rule == 'erf_full':
            selected = select_batch_erf_full(available, jobs, Cap)
        else:
            selected = available[:1]

        if not selected:
            selected = [min(available, key=lambda j: jobs[j]['s'])]

        proc_time = max(jobs[j]['p'] for j in selected)
        batch = {'start': start_time, 'p': proc_time, 'jobs': selected}
        schedule[m].append(batch)
        machine_free[m] = start_time + proc_time
        for j in selected:
            unscheduled.remove(j)

    objective = 0
    for m in range(M):
        for b in schedule[m]:
            comp = b['start'] + b['p']
            for j in b['jobs']:
                objective += comp
    return schedule, objective


def schedule_to_solution(schedule, jobs):
    batches = []
    job_assignments = {}
    obj = 0
    for m, batches_m in enumerate(schedule):
        for idx, b in enumerate(batches_m):
            batch_id = idx + 1
            comp = b['start'] + b['p']
            batch_dict = {
                'batch_id': batch_id,
                'machine': m,
                'start_time': int(b['start']),
                'processing_time': int(b['p']),
                'jobs': [jobs[j]['job_id'] for j in b['jobs']]
            }
            batches.append(batch_dict)
            for j in b['jobs']:
                job_id_str = str(jobs[j]['job_id'])
                job_assignments[job_id_str] = {
                    'batch': batch_id,
                    'machine': m,
                    'completion_time': int(comp)
                }
                obj += comp
    batches.sort(key=lambda x: (x['machine'], x['batch_id']))
    return {
        'objective_value': float(obj),
        'batches': batches,
        'job_assignments': job_assignments
    }


# ----------------- MIP solver -----------------
def solve_mip(jobs, M, Cap, time_limit, init_schedule=None, logger=None):
    N = len(jobs)
    if N == 0:
        return {}, 0.0
    H = max(j['r'] for j in jobs) + sum(j['p'] for j in jobs)
    max_p = max(j['p'] for j in jobs)
    K = N  # max batches per machine

    env = gp.Env(empty=True)
    env.setParam('Seed', 0)
    env.setParam('Threads', 1)
    env.setParam('NumericFocus', 0)
    env.start()

    model = gp.Model(env=env)
    model.setParam('MIPGap', 1e-4)
    model.setParam('TimeLimit', time_limit)
    model.setParam('OutputFlag', 0)

    x = model.addVars(N, K, M, vtype=GRB.BINARY, name="x")
    S = model.addVars(K, M, lb=0, name="S")
    P = model.addVars(K, M, lb=0, name="P")
    C = model.addVars(N, lb=0, name="C")

    # each job exactly once
    for j in range(N):
        model.addConstr(gp.quicksum(x[j, k, m] for k in range(K) for m in range(M)) == 1)

    # capacity
    for k in range(K):
        for m in range(M):
            model.addConstr(gp.quicksum(jobs[j]['s'] * x[j, k, m] for j in range(N)) <= Cap)

    # processing time
    for j in range(N):
        for k in range(K):
            for m in range(M):
                model.addConstr(P[k, m] >= jobs[j]['p'] - H * (1 - x[j, k, m]))

    # P = 0 if no jobs
    for k in range(K):
        for m in range(M):
            model.addConstr(P[k, m] <= max_p * gp.quicksum(x[j, k, m] for j in range(N)))

    # release date
    for j in range(N):
        for k in range(K):
            for m in range(M):
                model.addConstr(S[k, m] >= jobs[j]['r'] - H * (1 - x[j, k, m]))

    # S = 0 if no jobs
    for k in range(K):
        for m in range(M):
            model.addConstr(S[k, m] <= H * gp.quicksum(x[j, k, m] for j in range(N)))

    # ordering on same machine
    for m in range(M):
        for k in range(K - 1):
            model.addConstr(S[k + 1, m] >= S[k, m] + P[k, m])

    # completion time
    for j in range(N):
        for k in range(K):
            for m in range(M):
                model.addConstr(C[j] >= S[k, m] + P[k, m] - H * (1 - x[j, k, m]))

    model.setObjective(gp.quicksum(C[j] for j in range(N)), GRB.MINIMIZE)

    # warm start
    if init_schedule:
        for v in x.values():
            v.Start = 0.0
        for v in S.values():
            v.Start = 0.0
        for v in P.values():
            v.Start = 0.0
        for v in C.values():
            v.Start = 0.0

        for m in range(M):
            for k, batch in enumerate(init_schedule[m]):
                for j in batch['jobs']:
                    x[j, k, m].Start = 1.0
                S[k, m].Start = batch['start']
                P[k, m].Start = batch['p']
                comp = batch['start'] + batch['p']
                for j in batch['jobs']:
                    C[j].Start = comp

    model.optimize()

    if model.SolCount > 0:
        schedule = [[] for _ in range(M)]
        for m in range(M):
            for k in range(K):
                jobs_in_batch = [j for j in range(N) if x[j, k, m].X > 0.5]
                if jobs_in_batch:
                    start = round(S[k, m].X)
                    proc = round(P[k, m].X)
                    schedule[m].append({'start': start, 'p': proc, 'jobs': jobs_in_batch})
        obj = 0
        for m in range(M):
            for b in schedule[m]:
                comp = b['start'] + b['p']
                for j in b['jobs']:
                    obj += comp
        return schedule, float(obj)
    else:
        return None, None


# ----------------- Main -----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    start_time = time.time()

    instance = read_instance(args.instance_path)
    jobs = instance['jobs']
    M = instance['num_machines']
    Cap = instance['batch_capacity']

    rules = [
        'spt_full',
        'lpt_full',
        'spt_thr_1.5',
        'spt_thr_2.0',
        'spt_thr_3.0',
        'lpt_thr_0.5',
        'lpt_thr_0.7',
        'erf_full',
        'wait_full'
    ]

    best_schedule = None
    best_obj = float('inf')

    for rule in rules:
        sched, obj = build_schedule(jobs, M, Cap, rule)
        if obj < best_obj:
            best_obj = obj
            best_schedule = sched
            if logger:
                solution = schedule_to_solution(best_schedule, jobs)
                logger.log_solution(best_obj, solution)

    # Try MIP for small instances if time remains
    if HAS_GUROBI:
        N = len(jobs)
        elapsed = time.time() - start_time
        if N <= 40 and elapsed < args.time_limit - 1:
            remaining = max(1, args.time_limit - elapsed)
            try:
                mip_schedule, mip_obj = solve_mip(jobs, M, Cap, remaining, best_schedule, logger)
                if mip_schedule is not None and mip_obj < best_obj:
                    best_obj = mip_obj
                    best_schedule = mip_schedule
                    if logger:
                        solution = schedule_to_solution(best_schedule, jobs)
                        logger.log_solution(best_obj, solution)
            except Exception as e:
                sys.stderr.write(f"MIP failed: {e}\n")

    solution = schedule_to_solution(best_schedule, jobs)
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()
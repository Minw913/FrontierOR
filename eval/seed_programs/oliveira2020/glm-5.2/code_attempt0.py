import argparse
import json
import math
import sys
import time
from collections import defaultdict

try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path and SolutionLogger else None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    H = instance['time_horizon']
    jobs = instance['jobs']
    n = instance['parameters']['n']
    m = instance['parameters']['m']

    job_map = {j['id']: j for j in jobs}
    job_ids = [j['id'] for j in jobs]
    p = {j['id']: j['processing_time'] for j in jobs}
    w = {j['id']: j['weight'] for j in jobs}
    d = {j['id']: j['due_date'] for j in jobs}

    wall_start = time.time()

    # --- Heuristics ---
    best_schedule, best_obj = run_heuristics(jobs, m, H, job_map)
    if logger:
        logger.log_solution(best_obj, best_schedule)

    remaining = max(1, args.time_limit - (time.time() - wall_start) - 2)

    # --- MIP (time-indexed) for small/medium instances ---
    n_vars = sum(max(0, H - j['processing_time'] + 1) for j in jobs)
    if remaining > 3 and n_vars <= 300000:
        try:
            mip_result = solve_mip(jobs, m, H, remaining, best_schedule, best_obj, logger, job_map, p, w, d)
            if mip_result is not None:
                mip_schedule, mip_obj = mip_result
                if mip_obj < best_obj:
                    best_schedule = mip_schedule
                    best_obj = mip_obj
        except Exception as e:
            print(f"MIP failed: {e}", file=sys.stderr)

    # --- Local search if time remains ---
    remaining = max(1, args.time_limit - (time.time() - wall_start) - 2)
    if remaining > 2:
        ls_schedule, ls_obj = local_search(jobs, m, H, best_schedule, best_obj, remaining, logger, job_map, p, w, d)
        if ls_obj < best_obj:
            best_schedule = ls_schedule
            best_obj = ls_obj

    # --- Output ---
    output = {
        'objective_value': float(best_obj),
        'schedule': best_schedule
    }
    with open(args.solution_path, 'w') as f:
        json.dump(output, f, indent=2)


# ---------- Heuristics ----------

def run_heuristics(jobs, m, H, job_map):
    best_schedule, best_obj = None, float('inf')
    p_bar = sum(j['processing_time'] for j in jobs) / max(1, len(jobs))

    rules = [
        lambda j: -j['weight'] / max(1, j['processing_time']),          # WSPT
        lambda j: j['due_date'],                                        # EDD
        lambda j: j['due_date'] / max(1, j['weight']),                  # WEDD
        lambda j: j['processing_time'],                                 # SPT
        lambda j: -j['processing_time'],                                # LPT
        lambda j: -j['weight'],                                          # LW
    ]

    for rule in rules:
        schedule, obj = list_scheduling(jobs, m, H, rule, job_map)
        if obj < best_obj:
            best_obj, best_schedule = obj, schedule

    for k in [0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
        schedule, obj = atc_heuristic(jobs, m, H, k, p_bar, job_map)
        if obj < best_obj:
            best_obj, best_schedule = obj, schedule

    for rule in rules:
        schedule, obj = greedy_insertion(jobs, m, H, rule, job_map)
        if obj < best_obj:
            best_obj, best_schedule = obj, schedule

    return best_schedule, best_obj


def list_scheduling(jobs, m, H, priority_fn, job_map):
    sorted_jobs = sorted(jobs, key=priority_fn)
    machine_avail = [0] * m
    machine_jobs = [[] for _ in range(m)]

    for job in sorted_jobs:
        jid, pj = job['id'], job['processing_time']
        best_m = min(range(m), key=lambda k: machine_avail[k])
        machine_jobs[best_m].append(jid)
        machine_avail[best_m] += pj

    return build_schedule_from_machine_jobs(machine_jobs, job_map, m)


def atc_heuristic(jobs, m, H, k, p_bar, job_map):
    unassigned = set(j['id'] for j in jobs)
    machine_avail = [0] * m
    machine_jobs = [[] for _ in range(m)]

    while unassigned:
        best_m = min(range(m), key=lambda k: machine_avail[k])
        t = machine_avail[best_m]

        best_jid, best_idx = None, -float('inf')
        for jid in unassigned:
            j = job_map[jid]
            slack = max(0, j['due_date'] - j['processing_time'] - t)
            denom = max(k * p_bar, 1e-9)
            idx = (j['weight'] / max(j['processing_time'], 1)) * math.exp(-slack / denom)
            if idx > best_idx:
                best_idx, best_jid = idx, jid

        machine_jobs[best_m].append(best_jid)
        machine_avail[best_m] += job_map[best_jid]['processing_time']
        unassigned.remove(best_jid)

    return build_schedule_from_machine_jobs(machine_jobs, job_map, m)


def greedy_insertion(jobs, m, H, priority_fn, job_map):
    sorted_jobs = sorted(jobs, key=priority_fn)
    machine_avail = [0] * m
    machine_jobs = [[] for _ in range(m)]

    for job in sorted_jobs:
        jid, pj, wj, dj = job['id'], job['processing_time'], job['weight'], job['due_date']
        best_m, best_cost = -1, float('inf')
        for mk in range(m):
            comp = machine_avail[mk] + pj
            cost = wj * max(0, comp - dj)
            if cost < best_cost or (cost == best_cost and machine_avail[mk] < machine_avail[best_m]):
                best_cost, best_m = cost, mk
        machine_jobs[best_m].append(jid)
        machine_avail[best_m] += pj

    return build_schedule_from_machine_jobs(machine_jobs, job_map, m)


# ---------- Schedule builders ----------

def build_schedule_from_machine_jobs(machine_jobs, job_map, m):
    schedule = []
    total_wt = 0
    for mk in range(m):
        t = 0
        for idx, jid in enumerate(machine_jobs[mk]):
            prev = machine_jobs[mk][idx - 1] if idx > 0 else 0
            j = job_map[jid]
            pj = j['processing_time']
            start, comp = t, t + pj
            tard = max(0, comp - j['due_date'])
            wt = j['weight'] * tard
            schedule.append({
                'machine_id': mk, 'prev_job': prev, 'job': jid,
                'start_time': start, 'completion_time': comp,
                'weighted_tardiness': wt
            })
            total_wt += wt
            t = comp
    return schedule, total_wt


def build_schedule_from_start_times(start_times, job_map, m):
    sorted_jids = sorted(start_times.keys(), key=lambda jid: (start_times[jid], jid))
    machine_avail = [0] * m
    machine_jobs = [[] for _ in range(m)]
    actual_starts = {}

    for jid in sorted_jids:
        st = start_times[jid]
        chosen = -1
        for mk in range(m):
            if machine_avail[mk] <= st:
                chosen = mk
                break
        if chosen == -1:
            chosen = min(range(m), key=lambda k: machine_avail[k])
            st = max(st, machine_avail[chosen])
        machine_jobs[chosen].append(jid)
        actual_starts[jid] = st
        machine_avail[chosen] = st + job_map[jid]['processing_time']

    schedule = []
    total_wt = 0
    for mk in range(m):
        for idx, jid in enumerate(machine_jobs[mk]):
            prev = machine_jobs[mk][idx - 1] if idx > 0 else 0
            j = job_map[jid]
            pj = j['processing_time']
            st = actual_starts[jid]
            comp = st + pj
            tard = max(0, comp - j['due_date'])
            wt = j['weight'] * tard
            schedule.append({
                'machine_id': mk, 'prev_job': prev, 'job': jid,
                'start_time': st, 'completion_time': comp,
                'weighted_tardiness': wt
            })
            total_wt += wt
    return schedule, total_wt


# ---------- MIP ----------

def solve_mip(jobs, m, H, time_limit, initial_schedule, initial_obj, logger, job_map, p, w, d):
    import gurobipy as gp
    from gurobipy import GRB

    model = gp.Model('wt')
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = time_limit
    model.Params.OutputFlag = 0

    y = {}
    for j in jobs:
        jid, pj = j['id'], j['processing_time']
        max_start = H - pj
        if max_start < 0:
            return None
        for t in range(max_start + 1):
            y[jid, t] = model.addVar(vtype=GRB.BINARY, name=f'y_{jid}_{t}')

    T = {j['id']: model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f'T_{j["id"]}') for j in jobs}
    model.update()

    for j in jobs:
        jid, pj = j['id'], j['processing_time']
        model.addConstr(gp.quicksum(y[jid, t] for t in range(H - pj + 1)) == 1)

    # Capacity constraints built efficiently
    cap_exprs = [gp.LinExpr() for _ in range(H)]
    for j in jobs:
        jid, pj = j['id'], j['processing_time']
        for t in range(H - pj + 1):
            for s in range(t, min(t + pj, H)):
                cap_exprs[s].add(y[jid, t], 1.0)
    for s in range(H):
        if cap_exprs[s].size() > 0:
            model.addConstr(cap_exprs[s] <= m)

    for j in jobs:
        jid, pj, dj = j['id'], j['processing_time'], j['due_date']
        comp_expr = gp.quicksum((t + pj) * y[jid, t] for t in range(H - pj + 1))
        model.addConstr(T[jid] >= comp_expr - dj)

    model.setObjective(gp.quicksum(w[j['id']] * T[j['id']] for j in jobs), GRB.MINIMIZE)

    # Warm start
    if initial_schedule:
        for s in initial_schedule:
            jid, st = s['job'], s['start_time']
            if (jid, st) in y:
                y[jid, st].Start = 1

    best_found = [initial_obj]

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj < best_found[0] - 1e-6:
                best_found[0] = obj
                start_times = {}
                for j in jobs:
                    jid, pj = j['id'], j['processing_time']
                    for t in range(H - pj + 1):
                        if model.cbGetSolution(y[jid, t]) > 0.5:
                            start_times[jid] = t
                            break
                sched, _ = build_schedule_from_start_times(start_times, job_map, m)
                if logger:
                    logger.log_solution(obj, sched)

    model.optimize(callback)

    if model.SolCount > 0:
        start_times = {}
        for j in jobs:
            jid, pj = j['id'], j['processing_time']
            for t in range(H - pj + 1):
                if y[jid, t].X > 0.5:
                    start_times[jid] = t
                    break
        sched, obj = build_schedule_from_start_times(start_times, job_map, m)
        if obj <= best_found[0] + 1e-6:
            return sched, obj
        else:
            return sched, obj

    return None


# ---------- Local search ----------

def local_search(jobs, m, H, initial_schedule, initial_obj, time_limit, logger, job_map, p, w, d):
    deadline = time.time() + time_limit

    machine_jobs = defaultdict(list)
    for s in initial_schedule:
        machine_jobs[s['machine_id']].append(s['job'])
    for mk in range(m):
        if mk not in machine_jobs:
            machine_jobs[mk] = []

    current_obj = initial_obj
    improved = True

    while improved and time.time() < deadline:
        improved = False

        # Adjacent swaps on each machine
        for mk in range(m):
            seq = machine_jobs[mk]
            for idx in range(len(seq) - 1):
                seq[idx], seq[idx + 1] = seq[idx + 1], seq[idx]
                schedule, obj = build_schedule_from_machine_jobs(machine_jobs, job_map, m)
                if obj < current_obj - 1e-9:
                    current_obj = obj
                    improved = True
                    if logger:
                        logger.log_solution(obj, schedule)
                else:
                    seq[idx], seq[idx + 1] = seq[idx + 1], seq[idx]
                if time.time() >= deadline:
                    break
            if time.time() >= deadline:
                break

        # Move job to end of another machine
        for mk1 in range(m):
            for idx in range(len(machine_jobs[mk1])):
                if time.time() >= deadline:
                    break
                jid = machine_jobs[mk1][idx]
                for mk2 in range(m):
                    if mk2 == mk1:
                        continue
                    machine_jobs[mk1].pop(idx)
                    machine_jobs[mk2].append(jid)
                    schedule, obj = build_schedule_from_machine_jobs(machine_jobs, job_map, m)
                    if obj < current_obj - 1e-9:
                        current_obj = obj
                        improved = True
                        if logger:
                            logger.log_solution(obj, schedule)
                        break
                    else:
                        machine_jobs[mk2].pop()
                        machine_jobs[mk1].insert(idx, jid)
                if improved:
                    break
            if improved:
                break

        # Insert job at position on another machine
        for mk1 in range(m):
            for idx in range(len(machine_jobs[mk1])):
                if time.time() >= deadline:
                    break
                jid = machine_jobs[mk1][idx]
                for mk2 in range(m):
                    if mk2 == mk1:
                        continue
                    for pos in range(len(machine_jobs[mk2]) + 1):
                        machine_jobs[mk1].pop(idx)
                        machine_jobs[mk2].insert(pos, jid)
                        schedule, obj = build_schedule_from_machine_jobs(machine_jobs, job_map, m)
                        if obj < current_obj - 1e-9:
                            current_obj = obj
                            improved = True
                            if logger:
                                logger.log_solution(obj, schedule)
                            break
                        else:
                            machine_jobs[mk2].pop(pos)
                            machine_jobs[mk1].insert(idx, jid)
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

        # Reorder each machine by EDD
        for mk in range(m):
            if time.time() >= deadline:
                break
            old_seq = list(machine_jobs[mk])
            machine_jobs[mk] = sorted(old_seq, key=lambda jid: job_map[jid]['due_date'])
            schedule, obj = build_schedule_from_machine_jobs(machine_jobs, job_map, m)
            if obj < current_obj - 1e-9:
                current_obj = obj
                improved = True
                if logger:
                    logger.log_solution(obj, schedule)
            else:
                machine_jobs[mk] = old_seq

    schedule, _ = build_schedule_from_machine_jobs(machine_jobs, job_map, m)
    return schedule, current_obj


if __name__ == '__main__':
    main()
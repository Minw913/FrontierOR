import argparse
import json
import time
import sys
from math import inf
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def compute_schedule(seq, cum, D, due_dates, n_machines):
    n = len(seq)
    if n == 0:
        return 0, []
    s = [0] * n
    job0 = seq[0]
    comp0 = s[0] + cum[job0][n_machines-1]
    if comp0 > due_dates[job0]:
        return None, None
    for i in range(1, n):
        s[i] = s[i-1] + D[seq[i-1]][seq[i]]
        job_i = seq[i]
        comp_i = s[i] + cum[job_i][n_machines-1]
        if comp_i > due_dates[job_i]:
            return None, None
    makespan = s[-1] + cum[seq[-1]][n_machines-1]
    return makespan, s

def solve_mip(n_jobs, n_machines, processing_times, due_dates, cum, D, time_limit, logger):
    try:
        model = gp.Model("nowait_flowshop")
        model.setParam("Threads", 1)
        model.setParam('OutputFlag', 0)
        model.setParam('TimeLimit', max(1, time_limit))

        nodes = [0] + list(range(n_jobs)) + [n_jobs+1]
        real_jobs = list(range(n_jobs))

        arcs = []
        for j in real_jobs:
            arcs.append((0, j+1))
        for i in real_jobs:
            arcs.append((i+1, n_jobs+1))
        for i in real_jobs:
            for j in real_jobs:
                if i != j:
                    arcs.append((i+1, j+1))

        x = model.addVars(arcs, vtype=GRB.BINARY, name="x")
        s = model.addVars(nodes, vtype=GRB.CONTINUOUS, lb=0, name="s")

        makespan = s[n_jobs+1]
        model.setObjective(makespan, GRB.MINIMIZE)

        model.addConstr(s[0] == 0)

        model.addConstr(gp.quicksum(x[0, j+1] for j in real_jobs) == 1)
        model.addConstr(gp.quicksum(x[i+1, n_jobs+1] for i in real_jobs) == 1)
        for j in real_jobs:
            model.addConstr(gp.quicksum(x[i, j+1] for (i, j1) in arcs if j1 == j+1) == 1)
        for i in real_jobs:
            model.addConstr(gp.quicksum(x[i+1, j] for (i1, j) in arcs if i1 == i+1) == 1)

        bigM = 10**6
        for (i, j) in arcs:
            if j == n_jobs+1:
                d_val = cum[i-1][n_machines-1] if i != 0 else 0
            else:
                if i == 0:
                    d_val = 0
                else:
                    d_val = D[i-1][j-1]
            model.addConstr(s[j] >= s[i] + d_val - bigM * (1 - x[i, j]))

        for j in real_jobs:
            model.addConstr(s[j+1] + cum[j][n_machines-1] <= due_dates[j])

        model.optimize()

        if model.SolCount > 0:
            obj_value = model.ObjVal
            seq = []
            current = 0
            while current != n_jobs+1:
                for j in nodes:
                    if (current, j) in x and x[current, j].X > 0.5:
                        if 1 <= j <= n_jobs:
                            seq.append(j-1)
                        current = j
                        break
            if logger:
                logger.log(obj_value)
            return seq, obj_value
        else:
            return None, None
    except gp.GurobiError as e:
        print(f"Gurobi error: {e}")
        return None, None
    except Exception as e:
        print(f"Error in MIP: {e}")
        return None, None

def greedy_insertion(n_jobs, cum, D, due_dates, n_machines, time_remaining, start_time):
    jobs = list(range(n_jobs))
    seq = []
    while jobs:
        if time.time() - start_time > time_remaining:
            break
        best_makespan = float('inf')
        best_job = None
        best_pos = None
        for j in jobs:
            for pos in range(0, len(seq)+1):
                candidate = seq[:pos] + [j] + seq[pos:]
                makespan, _ = compute_schedule(candidate, cum, D, due_dates, n_machines)
                if makespan is None:
                    continue
                if makespan < best_makespan:
                    best_makespan = makespan
                    best_job = j
                    best_pos = pos
        if best_job is not None:
            seq = seq[:best_pos] + [best_job] + seq[best_pos:]
            jobs.remove(best_job)
        else:
            j = jobs[0]
            seq.append(j)
            jobs.remove(j)
    return seq

def local_search(initial_seq, cum, D, due_dates, n_machines, time_remaining, global_start, time_limit, logger):
    best_seq = initial_seq
    best_makespan, _ = compute_schedule(best_seq, cum, D, due_dates, n_machines)
    if best_makespan is None:
        best_makespan = float('inf')
    improved = True
    while improved and time.time() - global_start < time_limit:
        improved = False
        time_remaining = time_limit - (time.time() - global_start)
        if time_remaining <= 0:
            break

        for i in range(len(best_seq)-1):
            candidate = best_seq[:i] + [best_seq[i+1], best_seq[i]] + best_seq[i+2:]
            makespan, _ = compute_schedule(candidate, cum, D, due_dates, n_machines)
            if makespan is not None and makespan < best_makespan:
                best_seq = candidate
                best_makespan = makespan
                improved = True
                if logger:
                    logger.log(best_makespan)
                break
        if improved:
            continue

        for i in range(len(best_seq)):
            for j in range(i+1, len(best_seq)):
                if time.time() - global_start > time_limit:
                    break
                candidate = best_seq[:]
                candidate[i], candidate[j] = candidate[j], candidate[i]
                makespan, _ = compute_schedule(candidate, cum, D, due_dates, n_machines)
                if makespan is not None and makespan < best_makespan:
                    best_seq = candidate
                    best_makespan = makespan
                    improved = True
                    if logger:
                        logger.log(best_makespan)
                    break
            if improved:
                break
        if improved:
            continue

        for i in range(len(best_seq)):
            for j in range(len(best_seq)):
                if i == j:
                    continue
                if time.time() - global_start > time_limit:
                    break
                candidate = best_seq[:i] + best_seq[i+1:]
                candidate = candidate[:j] + [best_seq[i]] + candidate[j:]
                makespan, _ = compute_schedule(candidate, cum, D, due_dates, n_machines)
                if makespan is not None and makespan < best_makespan:
                    best_seq = candidate
                    best_makespan = makespan
                    improved = True
                    if logger:
                        logger.log(best_makespan)
                    break
            if improved:
                break
    return best_seq, best_makespan

def main(instance_path, solution_path, time_limit, log_path):
    global_start = time.time()
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    with open(instance_path, 'r') as f:
        data = json.load(f)
    n_jobs = data['n_jobs']
    n_machines = data['n_machines']
    processing_times = data['processing_times']
    due_dates = data['due_dates']

    cum = []
    for j in range(n_jobs):
        c = 0
        cum_j = []
        for i in range(n_machines):
            c += processing_times[i][j]
            cum_j.append(c)
        cum.append(cum_j)

    D = [[0]*n_jobs for _ in range(n_jobs)]
    for i in range(n_jobs):
        for j in range(n_jobs):
            if i == j:
                D[i][j] = 0
            else:
                max_val = -10**15
                for k in range(n_machines):
                    if k == 0:
                        term = cum[i][0]
                    else:
                        term = cum[i][k] - cum[j][k-1]
                    if term > max_val:
                        max_val = term
                D[i][j] = max_val

    time_remaining = time_limit - (time.time() - global_start)
    if time_remaining <= 0:
        seq_ed = list(range(n_jobs))
        seq_ed.sort(key=lambda j: due_dates[j])
        makespan_ed, _ = compute_schedule(seq_ed, cum, D, due_dates, n_machines)
        if makespan_ed is None:
            seq_nat = list(range(n_jobs))
            makespan_nat, _ = compute_schedule(seq_nat, cum, D, due_dates, n_machines)
            if makespan_nat is None:
                seq = seq_ed
                makespan, s = compute_schedule(seq, cum, D, due_dates, n_machines)
            else:
                seq = seq_nat
                makespan = makespan_nat
        else:
            seq = seq_ed
            makespan = makespan_ed
    else:
        if n_jobs <= 20:
            seq, makespan = solve_mip(n_jobs, n_machines, processing_times, due_dates, cum, D, time_remaining, logger)
            if seq is None:
                seq_ed = list(range(n_jobs))
                seq_ed.sort(key=lambda j: due_dates[j])
                makespan_ed, _ = compute_schedule(seq_ed, cum, D, due_dates, n_machines)
                if makespan_ed is None:
                    seq_nat = list(range(n_jobs))
                    makespan_nat, _ = compute_schedule(seq_nat, cum, D, due_dates, n_machines)
                    if makespan_nat is None:
                        seq = seq_ed
                        makespan, s = compute_schedule(seq, cum, D, due_dates, n_machines)
                    else:
                        seq = seq_nat
                        makespan = makespan_nat
                else:
                    seq = seq_ed
                    makespan = makespan_ed
        else:
            seq_ed = list(range(n_jobs))
            seq_ed.sort(key=lambda j: due_dates[j])
            makespan_ed, _ = compute_schedule(seq_ed, cum, D, due_dates, n_machines)
            if makespan_ed is None:
                seq_nat = list(range(n_jobs))
                makespan_nat, _ = compute_schedule(seq_nat, cum, D, due_dates, n_machines)
                if makespan_nat is None:
                    seq = seq_ed
                else:
                    seq = seq_nat
            else:
                seq = seq_ed

        time_remaining_after_init = time_limit - (time.time() - global_start)
        if time_remaining_after_init > 0:
            initial_seq = greedy_insertion(n_jobs, cum, D, due_dates, n_machines, time_remaining_after_init, global_start)
            time_remaining_after_greedy = time_limit - (time.time() - global_start)
            if time_remaining_after_greedy > 0:
                seq, makespan = local_search(initial_seq, cum, D, due_dates, n_machines, time_remaining_after_greedy, global_start, time_limit, logger)
            else:
                makespan, _ = compute_schedule(initial_seq, cum, D, due_dates, n_machines)
                seq = initial_seq
        else:
            makespan, _ = compute_schedule(seq, cum, D, due_dates, n_machines)

    if makespan is None:
        makespan, _ = compute_schedule(seq, cum, D, due_dates, n_machines)

    sequence_1based = [job_index+1 for job_index in seq]
    solution = {
        "objective_value": float(makespan) if makespan is not None else 0.0,
        "sequence": sequence_1based
    }
    with open(solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    main(args.instance_path, args.solution_path, args.time_limit, args.log_path)
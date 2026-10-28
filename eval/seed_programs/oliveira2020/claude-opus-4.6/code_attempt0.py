import argparse
import json
import time
import random
import math
import copy
from solution_logger import SolutionLogger

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

def compute_objective(machines, jobs_dict):
    total_wt = 0
    for machine_jobs in machines:
        t = 0
        for jid, start in machine_jobs:
            job = jobs_dict[jid]
            ct = start + job['processing_time']
            tardiness = max(0, ct - job['due_date'])
            total_wt += job['weight'] * tardiness
            t = ct
    return total_wt

def compute_machine_wt(machine_jobs, jobs_dict):
    """Compute weighted tardiness for a single machine's job sequence (compact schedule)."""
    wt = 0
    t = 0
    for jid in machine_jobs:
        job = jobs_dict[jid]
        ct = t + job['processing_time']
        tardiness = max(0, ct - job['due_date'])
        wt += job['weight'] * tardiness
        t = ct
    return wt

def make_compact_schedule(machines_seq, jobs_dict):
    """Given machines as lists of job ids, create compact schedule with start times."""
    machines = []
    for seq in machines_seq:
        machine = []
        t = 0
        for jid in seq:
            machine.append((jid, t))
            t += jobs_dict[jid]['processing_time']
        machines.append(machine)
    return machines

def schedule_to_seq(machines):
    """Convert machines with (jid, start) to just sequences of jids."""
    return [[jid for jid, _ in m] for m in machines]

def compute_total_wt_from_seq(machines_seq, jobs_dict):
    total = 0
    for seq in machines_seq:
        total += compute_machine_wt(seq, jobs_dict)
    return total

def atc_schedule(jobs, m, jobs_dict, k_param=1.0):
    """Apparent Tardiness Cost scheduling heuristic."""
    n = len(jobs)
    p_avg = sum(j['processing_time'] for j in jobs) / max(n, 1)
    
    machines_seq = [[] for _ in range(m)]
    machines_time = [0] * m
    remaining = set(j['id'] for j in jobs)
    
    while remaining:
        # Find machine with earliest available time
        min_time = min(machines_time)
        available_machines = [i for i in range(m) if machines_time[i] == min_time]
        
        for mi in available_machines:
            if not remaining:
                break
            t = machines_time[mi]
            
            best_job = None
            best_priority = -1
            
            for jid in remaining:
                job = jobs_dict[jid]
                p = job['processing_time']
                w = job['weight']
                d = job['due_date']
                
                slack = max(0, d - p - t)
                if k_param * p_avg > 0:
                    priority = (w / p) * math.exp(-slack / (k_param * p_avg + 1e-10))
                else:
                    priority = w / p
                
                if priority > best_priority:
                    best_priority = priority
                    best_job = jid
            
            if best_job is not None:
                remaining.remove(best_job)
                machines_seq[mi].append(best_job)
                machines_time[mi] += jobs_dict[best_job]['processing_time']
    
    return machines_seq

def wspt_schedule(jobs, m, jobs_dict):
    """Weighted Shortest Processing Time first."""
    sorted_jobs = sorted(jobs, key=lambda j: j['processing_time'] / max(j['weight'], 1e-10))
    
    machines_seq = [[] for _ in range(m)]
    machines_time = [0] * m
    
    for job in sorted_jobs:
        mi = min(range(m), key=lambda i: machines_time[i])
        machines_seq[mi].append(job['id'])
        machines_time[mi] += job['processing_time']
    
    return machines_seq

def wedd_schedule(jobs, m, jobs_dict):
    """Weighted Earliest Due Date."""
    sorted_jobs = sorted(jobs, key=lambda j: j['due_date'] / max(j['weight'], 1e-10))
    
    machines_seq = [[] for _ in range(m)]
    machines_time = [0] * m
    
    for job in sorted_jobs:
        mi = min(range(m), key=lambda i: machines_time[i])
        machines_seq[mi].append(job['id'])
        machines_time[mi] += job['processing_time']
    
    return machines_seq

def optimize_single_machine(seq, jobs_dict, time_budget=0.01):
    """Optimize job order on a single machine using adjacent swap."""
    if len(seq) <= 1:
        return seq
    
    best_seq = list(seq)
    best_wt = compute_machine_wt(best_seq, jobs_dict)
    improved = True
    
    while improved:
        improved = False
        for i in range(len(best_seq) - 1):
            new_seq = list(best_seq)
            new_seq[i], new_seq[i+1] = new_seq[i+1], new_seq[i]
            new_wt = compute_machine_wt(new_seq, jobs_dict)
            if new_wt < best_wt:
                best_seq = new_seq
                best_wt = new_wt
                improved = True
    
    return best_seq

def optimize_single_machine_insert(seq, jobs_dict):
    """Optimize job order on a single machine using insertion moves."""
    if len(seq) <= 1:
        return seq
    
    best_seq = list(seq)
    best_wt = compute_machine_wt(best_seq, jobs_dict)
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
                new_wt = compute_machine_wt(new_seq, jobs_dict)
                if new_wt < best_wt:
                    best_seq = new_seq
                    best_wt = new_wt
                    improved = True
                    break
            if improved:
                break
    
    return best_seq

def local_search_swap_between(machines_seq, jobs_dict, deadline):
    """Swap jobs between different machines."""
    m = len(machines_seq)
    best_seq = [list(s) for s in machines_seq]
    best_obj = compute_total_wt_from_seq(best_seq, jobs_dict)
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
                        # Swap job i from mi with job j from mj
                        old_wt_mi = compute_machine_wt(best_seq[mi], jobs_dict)
                        old_wt_mj = compute_machine_wt(best_seq[mj], jobs_dict)
                        
                        new_mi = list(best_seq[mi])
                        new_mj = list(best_seq[mj])
                        new_mi[i], new_mj[j] = new_mj[j], new_mi[i]
                        
                        new_wt_mi = compute_machine_wt(new_mi, jobs_dict)
                        new_wt_mj = compute_machine_wt(new_mj, jobs_dict)
                        
                        if new_wt_mi + new_wt_mj < old_wt_mi + old_wt_mj:
                            best_seq[mi] = new_mi
                            best_seq[mj] = new_mj
                            best_obj = best_obj - (old_wt_mi + old_wt_mj) + (new_wt_mi + new_wt_mj)
                            improved = True
    
    return best_seq, best_obj

def local_search_move_between(machines_seq, jobs_dict, deadline):
    """Move a job from one machine to another."""
    m = len(machines_seq)
    best_seq = [list(s) for s in machines_seq]
    best_obj = compute_total_wt_from_seq(best_seq, jobs_dict)
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
                old_wt_mi = compute_machine_wt(best_seq[mi], jobs_dict)
                new_mi = best_seq[mi][:i] + best_seq[mi][i+1:]
                new_wt_mi = compute_machine_wt(new_mi, jobs_dict)
                
                best_move = None
                best_delta = 0
                
                for mj in range(m):
                    if mj == mi:
                        continue
                    old_wt_mj = compute_machine_wt(best_seq[mj], jobs_dict)
                    
                    for pos in range(len(best_seq[mj]) + 1):
                        new_mj = best_seq[mj][:pos] + [job] + best_seq[mj][pos:]
                        new_wt_mj = compute_machine_wt(new_mj, jobs_dict)
                        
                        delta = (new_wt_mi + new_wt_mj) - (old_wt_mi + old_wt_mj)
                        if delta < best_delta:
                            best_delta = delta
                            best_move = (mj, pos, new_mi, new_mj)
                
                if best_move is not None:
                    mj, pos, new_mi_seq, new_mj_seq = best_move
                    best_seq[mi] = new_mi_seq
                    best_seq[mj] = new_mj_seq
                    best_obj += best_delta
                    improved = True
                    break
            if improved:
                break
    
    return best_seq, best_obj

def perturbation(machines_seq, jobs_dict, rng, strength=2):
    """Random perturbation for ILS."""
    m = len(machines_seq)
    new_seq = [list(s) for s in machines_seq]
    
    for _ in range(strength):
        # Collect all non-empty machines
        non_empty = [i for i in range(m) if len(new_seq[i]) > 0]
        if len(non_empty) < 1:
            break
        
        action = rng.random()
        
        if action < 0.5 and len(non_empty) >= 2:
            # Swap two random jobs between machines
            mi, mj = rng.sample(non_empty, 2)
            if len(new_seq[mi]) > 0 and len(new_seq[mj]) > 0:
                i = rng.randint(0, len(new_seq[mi]) - 1)
                j = rng.randint(0, len(new_seq[mj]) - 1)
                new_seq[mi][i], new_seq[mj][j] = new_seq[mj][j], new_seq[mi][i]
        else:
            # Move a random job to a random position on another machine
            mi = rng.choice(non_empty)
            if len(new_seq[mi]) > 0:
                i = rng.randint(0, len(new_seq[mi]) - 1)
                job = new_seq[mi].pop(i)
                mj = rng.randint(0, m - 1)
                pos = rng.randint(0, len(new_seq[mj]))
                new_seq[mj].insert(pos, job)
    
    return new_seq

def build_solution_dict(machines_seq, jobs_dict, m):
    """Build the solution dictionary from machine sequences."""
    schedule = []
    total_wt = 0
    
    for mi in range(m):
        t = 0
        prev_job = 0
        for jid in machines_seq[mi]:
            job = jobs_dict[jid]
            start = t
            ct = start + job['processing_time']
            tardiness = max(0, ct - job['due_date'])
            wt = job['weight'] * tardiness
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
    deadline = start_time + time_limit - 1  # Leave 1 second buffer
    
    jobs = instance['jobs']
    n = instance['parameters']['n']
    m = instance['parameters']['m']
    
    jobs_dict = {j['id']: j for j in jobs}
    
    # Generate multiple initial solutions
    best_seq = None
    best_obj = float('inf')
    
    # Try different heuristics
    heuristics = []
    
    # ATC with different k values
    for k in [0.1, 0.5, 1.0, 2.0, 3.0, 5.0]:
        heuristics.append(('ATC_k={}'.format(k), lambda k=k: atc_schedule(jobs, m, jobs_dict, k)))
    
    heuristics.append(('WSPT', lambda: wspt_schedule(jobs, m, jobs_dict)))
    heuristics.append(('WEDD', lambda: wedd_schedule(jobs, m, jobs_dict)))
    
    for name, hfunc in heuristics:
        if time.time() >= deadline:
            break
        seq = hfunc()
        # Optimize each machine
        for mi in range(m):
            seq[mi] = optimize_single_machine(seq[mi], jobs_dict)
        obj = compute_total_wt_from_seq(seq, jobs_dict)
        if obj < best_obj:
            best_obj = obj
            best_seq = [list(s) for s in seq]
            sol = build_solution_dict(best_seq, jobs_dict, m)
            if logger:
                logger.log_solution(best_obj, sol)
    
    if best_seq is None:
        # Fallback
        best_seq = [[] for _ in range(m)]
        for i, job in enumerate(jobs):
            best_seq[i % m].append(job['id'])
        best_obj = compute_total_wt_from_seq(best_seq, jobs_dict)
        sol = build_solution_dict(best_seq, jobs_dict, m)
        if logger:
            logger.log_solution(best_obj, sol)
    
    # Local search on initial solution
    # First optimize each machine individually with insertion
    for mi in range(m):
        if time.time() >= deadline:
            break
        best_seq[mi] = optimize_single_machine_insert(best_seq[mi], jobs_dict)
    
    new_obj = compute_total_wt_from_seq(best_seq, jobs_dict)
    if new_obj < best_obj:
        best_obj = new_obj
        sol = build_solution_dict(best_seq, jobs_dict, m)
        if logger:
            logger.log_solution(best_obj, sol)
    
    # Inter-machine local search
    phase_deadline = min(deadline, start_time + time_limit * 0.3)
    best_seq, best_obj_new = local_search_move_between(best_seq, jobs_dict, phase_deadline)
    if best_obj_new < best_obj:
        best_obj = best_obj_new
        sol = build_solution_dict(best_seq, jobs_dict, m)
        if logger:
            logger.log_solution(best_obj, sol)
    
    phase_deadline = min(deadline, start_time + time_limit * 0.4)
    best_seq, best_obj_new = local_search_swap_between(best_seq, jobs_dict, phase_deadline)
    if best_obj_new < best_obj:
        best_obj = best_obj_new
        sol = build_solution_dict(best_seq, jobs_dict, m)
        if logger:
            logger.log_solution(best_obj, sol)
    
    # Optimize each machine again
    for mi in range(m):
        if time.time() >= deadline:
            break
        best_seq[mi] = optimize_single_machine_insert(best_seq[mi], jobs_dict)
    new_obj = compute_total_wt_from_seq(best_seq, jobs_dict)
    if new_obj < best_obj:
        best_obj = new_obj
        sol = build_solution_dict(best_seq, jobs_dict, m)
        if logger:
            logger.log_solution(best_obj, sol)
    
    # ILS (Iterated Local Search) for remaining time
    rng = random.Random(42)
    global_best_seq = [list(s) for s in best_seq]
    global_best_obj = best_obj
    
    no_improve_count = 0
    iteration = 0
    
    while time.time() < deadline:
        iteration += 1
        
        # Determine perturbation strength
        if no_improve_count > 20:
            strength = min(4, max(2, n // 5))
        else:
            strength = max(1, min(3, n // 10))
        
        # Perturb
        perturbed_seq = perturbation(best_seq, jobs_dict, rng, strength)
        
        # Optimize each machine
        for mi in range(m):
            if time.time() >= deadline:
                break
            perturbed_seq[mi] = optimize_single_machine(perturbed_seq[mi], jobs_dict)
        
        if time.time() >= deadline:
            break
        
        # Quick inter-machine move search
        iter_deadline = min(deadline, time.time() + max(0.5, (deadline - time.time()) * 0.1))
        perturbed_seq, perturbed_obj = local_search_move_between(perturbed_seq, jobs_dict, iter_deadline)
        
        if time.time() >= deadline:
            break
        
        # Optimize each machine with insertion
        for mi in range(m):
            if time.time() >= deadline:
                break
            perturbed_seq[mi] = optimize_single_machine_insert(perturbed_seq[mi], jobs_dict)
        
        perturbed_obj = compute_total_wt_from_seq(perturbed_seq, jobs_dict)
        
        # Quick swap search
        if time.time() < deadline:
            iter_deadline = min(deadline, time.time() + max(0.3, (deadline - time.time()) * 0.05))
            perturbed_seq, perturbed_obj = local_search_swap_between(perturbed_seq, jobs_dict, iter_deadline)
        
        # Acceptance criterion
        if perturbed_obj < best_obj:
            best_seq = [list(s) for s in perturbed_seq]
            best_obj = perturbed_obj
            no_improve_count = 0
            
            if best_obj < global_best_obj:
                global_best_obj = best_obj
                global_best_seq = [list(s) for s in best_seq]
                sol = build_solution_dict(global_best_seq, jobs_dict, m)
                if logger:
                    logger.log_solution(global_best_obj, sol)
        else:
            no_improve_count += 1
            # Restart from global best if stuck
            if no_improve_count > 50:
                best_seq = [list(s) for s in global_best_seq]
                best_obj = global_best_obj
                no_improve_count = 0
    
    # Build final solution
    solution = build_solution_dict(global_best_seq, jobs_dict, m)
    return solution

def main():
    args = parse_args()
    
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    instance = read_instance(args.instance_path)
    
    solution = solve(instance, args.time_limit, logger)
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == '__main__':
    main()
import argparse
import json
import time
import random
import math
from itertools import permutations
from solution_logger import SolutionLogger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    return parser.parse_args()


def compute_makespan(sequence, n_jobs, n_machines, processing_times, no_idle_flags):
    """
    Compute the makespan and completion times for a given sequence.
    Returns (makespan, completion_times_dict) or (makespan, completion_times_matrix).
    """
    # completion_times[pos][machine]
    C = [[0] * n_machines for _ in range(n_jobs)]
    
    # For no-idle machines, we need to handle them specially.
    # First, compute completion times assuming regular flow shop constraints,
    # then adjust for no-idle constraints.
    
    # Actually, for no-idle machines, we need to ensure no gaps between consecutive jobs.
    # This means the start time of position k on a no-idle machine m equals
    # the completion time of position k-1 on machine m.
    # But also, job at position k can't start on machine m until it finishes on machine m-1.
    
    # For no-idle machines, the start of the first job might be delayed to ensure
    # all subsequent jobs can be processed without idle time.
    
    # Strategy: For each no-idle machine, we need to find the earliest start time S
    # such that starting the first job at time S on that machine, and processing
    # all jobs consecutively, satisfies the constraint that each job's start time
    # on this machine >= its completion time on the previous machine.
    
    # Let's compute this properly.
    # We'll do a forward pass, but for no-idle machines we need special handling.
    
    # For machine 0 (first machine):
    # If it's a no-idle machine, jobs are processed consecutively starting at some time S.
    # C[k][0] = S + sum of processing times of jobs 0..k
    # Constraint: C[k][0] >= p[seq[k]][0] (trivially satisfied if S >= 0)
    # Since it's the first machine, the only constraint is that jobs start at time >= 0.
    # For no-idle: S >= 0, so S = 0 works for the first machine.
    
    # For machine m > 0:
    # If machine m is no-idle:
    #   C[k][m] = S_m + sum_{i=0}^{k} p[seq[i]][m]
    #   Constraint: C[k][m] >= C[k][m-1] + p[seq[k]][m]
    #   => S_m + sum_{i=0}^{k} p[seq[i]][m] >= C[k][m-1] + p[seq[k]][m]
    #   => S_m >= C[k][m-1] - sum_{i=0}^{k-1} p[seq[i]][m]
    #   So S_m = max over all k of (C[k][m-1] - sum_{i=0}^{k-1} p[seq[i]][m])
    #   where for k=0, sum is 0, so S_m >= C[0][m-1]
    
    # If machine m is regular:
    #   C[k][m] = max(C[k-1][m] + p[seq[k]][m], C[k][m-1] + p[seq[k]][m])
    #   (standard flow shop)
    #   For k=0: C[0][m] = max(0 + p[seq[0]][m], C[0][m-1] + p[seq[0]][m]) = C[0][m-1] + p[seq[0]][m]
    
    # We need to compute C[k][m-1] before we can compute C[k][m].
    # So we process machines in order.
    
    for m in range(n_machines):
        if no_idle_flags[m]:
            # Compute the required start time S_m
            # S_m = max over k of (C[k][m-1] - cumulative_p[k])
            # where cumulative_p[k] = sum_{i=0}^{k-1} p[seq[i]][m], and for k=0 it's 0
            # But for m=0, C[k][m-1] doesn't exist; we just need S_0 >= 0
            
            if m == 0:
                S_m = 0
            else:
                S_m = 0
                cum_p = 0
                for k in range(n_jobs):
                    job = sequence[k]
                    # S_m >= C[k][m-1] - cum_p
                    val = C[k][m - 1] - cum_p
                    if val > S_m:
                        S_m = val
                    cum_p += processing_times[job][m]
            
            # Now compute completion times
            cum_p = 0
            for k in range(n_jobs):
                job = sequence[k]
                cum_p += processing_times[job][m]
                C[k][m] = S_m + cum_p
        else:
            # Regular machine
            for k in range(n_jobs):
                job = sequence[k]
                p = processing_times[job][m]
                prev_machine = C[k][m - 1] if m > 0 else 0
                prev_job = C[k - 1][m] if k > 0 else 0
                C[k][m] = max(prev_machine + p, prev_job + p)
    
    makespan = C[n_jobs - 1][n_machines - 1]
    return makespan, C


def compute_makespan_fast(sequence, n_jobs, n_machines, processing_times, no_idle_flags):
    """Fast makespan computation without storing full completion times."""
    # We only need the last row/column relationships
    # For each machine, we need C[k][m] for all k to compute the next machine
    # So we need to store at least the current and previous machine columns.
    
    prev_C = [0] * n_jobs  # C[k][m-1]
    curr_C = [0] * n_jobs  # C[k][m]
    
    for m in range(n_machines):
        if no_idle_flags[m]:
            if m == 0:
                S_m = 0
            else:
                S_m = 0
                cum_p = 0
                for k in range(n_jobs):
                    job = sequence[k]
                    val = prev_C[k] - cum_p
                    if val > S_m:
                        S_m = val
                    cum_p += processing_times[job][m]
            
            cum_p = 0
            for k in range(n_jobs):
                job = sequence[k]
                cum_p += processing_times[job][m]
                curr_C[k] = S_m + cum_p
        else:
            for k in range(n_jobs):
                job = sequence[k]
                p = processing_times[job][m]
                prev_machine = prev_C[k] if m > 0 else 0
                prev_job = curr_C[k - 1] if k > 0 else 0
                curr_C[k] = max(prev_machine + p, prev_job + p)
        
        prev_C, curr_C = curr_C, prev_C
    
    # After the loop, prev_C holds the last machine's completion times
    return prev_C[n_jobs - 1]


def neh_heuristic(n_jobs, n_machines, processing_times, no_idle_flags):
    """NEH heuristic for flow shop scheduling."""
    # Step 1: Compute total processing time for each job
    total_p = [(sum(processing_times[j]), j) for j in range(n_jobs)]
    # Sort in descending order of total processing time
    total_p.sort(reverse=True)
    
    # Step 2: Build sequence by inserting jobs one at a time
    sequence = [total_p[0][1]]
    
    for i in range(1, n_jobs):
        job = total_p[i][1]
        best_ms = float('inf')
        best_pos = 0
        
        for pos in range(len(sequence) + 1):
            candidate = sequence[:pos] + [job] + sequence[pos:]
            ms = compute_makespan_fast(candidate, len(candidate), n_machines, processing_times, no_idle_flags)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        
        sequence = sequence[:best_pos] + [job] + sequence[best_pos:]
    
    return sequence


def neh_heuristic_tiebreak(n_jobs, n_machines, processing_times, no_idle_flags):
    """NEH with different tie-breaking strategies."""
    total_p = [(sum(processing_times[j]), j) for j in range(n_jobs)]
    total_p.sort(reverse=True)
    
    sequence = [total_p[0][1]]
    
    for i in range(1, n_jobs):
        job = total_p[i][1]
        best_ms = float('inf')
        best_positions = []
        
        for pos in range(len(sequence) + 1):
            candidate = sequence[:pos] + [job] + sequence[pos:]
            ms = compute_makespan_fast(candidate, len(candidate), n_machines, processing_times, no_idle_flags)
            if ms < best_ms:
                best_ms = ms
                best_positions = [pos]
            elif ms == best_ms:
                best_positions.append(pos)
        
        # Tie-break: choose the position that is first (leftmost insertion)
        best_pos = best_positions[0]
        sequence = sequence[:best_pos] + [job] + sequence[best_pos:]
    
    return sequence


def local_search_insert(sequence, n_jobs, n_machines, processing_times, no_idle_flags, current_ms, time_limit_abs):
    """Insertion-based local search."""
    improved = True
    best_seq = list(sequence)
    best_ms = current_ms
    
    while improved:
        improved = False
        for i in range(n_jobs):
            if time.time() > time_limit_abs:
                return best_seq, best_ms
            
            job = best_seq[i]
            remaining = best_seq[:i] + best_seq[i+1:]
            
            for pos in range(n_jobs):
                if pos == i:
                    continue
                candidate = remaining[:pos] + [job] + remaining[pos:]
                ms = compute_makespan_fast(candidate, n_jobs, n_machines, processing_times, no_idle_flags)
                if ms < best_ms:
                    best_ms = ms
                    best_seq = candidate
                    improved = True
                    break
            if improved:
                break
    
    return best_seq, best_ms


def local_search_swap(sequence, n_jobs, n_machines, processing_times, no_idle_flags, current_ms, time_limit_abs):
    """Swap-based local search."""
    improved = True
    best_seq = list(sequence)
    best_ms = current_ms
    
    while improved:
        improved = False
        for i in range(n_jobs - 1):
            if time.time() > time_limit_abs:
                return best_seq, best_ms
            for j in range(i + 1, n_jobs):
                candidate = list(best_seq)
                candidate[i], candidate[j] = candidate[j], candidate[i]
                ms = compute_makespan_fast(candidate, n_jobs, n_machines, processing_times, no_idle_flags)
                if ms < best_ms:
                    best_ms = ms
                    best_seq = candidate
                    improved = True
                    break
            if improved:
                break
    
    return best_seq, best_ms


def perturbation(sequence, n_jobs, strength=3):
    """Random perturbation for ILS."""
    seq = list(sequence)
    for _ in range(strength):
        i = random.randint(0, n_jobs - 1)
        j = random.randint(0, n_jobs - 1)
        while j == i:
            j = random.randint(0, n_jobs - 1)
        # Remove job at position i and insert at position j
        job = seq.pop(i)
        seq.insert(j, job)
    return seq


def local_search_combined(sequence, n_jobs, n_machines, processing_times, no_idle_flags, current_ms, time_limit_abs):
    """Combined insertion local search - more thorough."""
    best_seq = list(sequence)
    best_ms = current_ms
    
    improved = True
    while improved:
        improved = False
        for i in range(n_jobs):
            if time.time() > time_limit_abs:
                return best_seq, best_ms
            
            job = best_seq[i]
            remaining = best_seq[:i] + best_seq[i+1:]
            
            local_best_ms = best_ms
            local_best_pos = i
            
            for pos in range(n_jobs):
                candidate = remaining[:pos] + [job] + remaining[pos:]
                ms = compute_makespan_fast(candidate, n_jobs, n_machines, processing_times, no_idle_flags)
                if ms < local_best_ms:
                    local_best_ms = ms
                    local_best_pos = pos
            
            if local_best_ms < best_ms:
                best_seq = remaining[:local_best_pos] + [job] + remaining[local_best_pos:]
                best_ms = local_best_ms
                improved = True
    
    return best_seq, best_ms


def solve():
    args = parse_args()
    
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    n_jobs = instance['n_jobs']
    n_machines = instance['n_machines']
    processing_times = instance['processing_times']
    no_idle_flags = instance['no_idle_flags']
    
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    start_time = time.time()
    time_limit_abs = start_time + args.time_limit - 0.5  # Leave 0.5s buffer
    
    # For very small instances, try brute force
    if n_jobs <= 8:
        # Try all permutations for tiny instances
        best_ms = float('inf')
        best_seq = list(range(n_jobs))
        
        if n_jobs <= 8:
            count = 0
            for perm in permutations(range(n_jobs)):
                if time.time() > time_limit_abs:
                    break
                seq = list(perm)
                ms = compute_makespan_fast(seq, n_jobs, n_machines, processing_times, no_idle_flags)
                if ms < best_ms:
                    best_ms = ms
                    best_seq = seq
                    if logger:
                        logger.log(best_ms)
                count += 1
        
        # Compute full completion times for output
        _, C = compute_makespan(best_seq, n_jobs, n_machines, processing_times, no_idle_flags)
        
        completion_times = {}
        for m in range(n_machines):
            completion_times[f'machine_{m}'] = [C[k][m] for k in range(n_jobs)]
        
        solution = {
            'objective_value': best_ms,
            'sequence': best_seq,
            'completion_times': completion_times
        }
        
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f)
        return
    
    # Step 1: NEH heuristic
    neh_seq = neh_heuristic(n_jobs, n_machines, processing_times, no_idle_flags)
    neh_ms = compute_makespan_fast(neh_seq, n_jobs, n_machines, processing_times, no_idle_flags)
    
    best_seq = list(neh_seq)
    best_ms = neh_ms
    if logger:
        logger.log(best_ms)
    
    # Also try NEH with different initial orderings
    if time.time() < time_limit_abs:
        # Try sorting by different criteria
        # By processing time on first machine
        sorted_by_first = sorted(range(n_jobs), key=lambda j: processing_times[j][0], reverse=True)
        # By processing time on last machine
        sorted_by_last = sorted(range(n_jobs), key=lambda j: processing_times[j][n_machines-1], reverse=True)
        # By difference between first half and second half processing times
        sorted_by_slope = sorted(range(n_jobs), 
                                  key=lambda j: sum(processing_times[j][:n_machines//2]) - sum(processing_times[j][n_machines//2:]),
                                  reverse=True)
        
        for sort_order in [sorted_by_first, sorted_by_last, sorted_by_slope]:
            if time.time() > time_limit_abs:
                break
            seq = [sort_order[0]]
            for i in range(1, n_jobs):
                if time.time() > time_limit_abs:
                    break
                job = sort_order[i]
                best_pos_ms = float('inf')
                best_pos = 0
                for pos in range(len(seq) + 1):
                    candidate = seq[:pos] + [job] + seq[pos:]
                    ms = compute_makespan_fast(candidate, len(candidate), n_machines, processing_times, no_idle_flags)
                    if ms < best_pos_ms:
                        best_pos_ms = ms
                        best_pos = pos
                seq = seq[:best_pos] + [job] + seq[best_pos:]
            
            ms = compute_makespan_fast(seq, n_jobs, n_machines, processing_times, no_idle_flags)
            if ms < best_ms:
                best_ms = ms
                best_seq = list(seq)
                if logger:
                    logger.log(best_ms)
    
    # Step 2: Local search on best NEH solution
    if time.time() < time_limit_abs:
        ls_seq, ls_ms = local_search_combined(best_seq, n_jobs, n_machines, processing_times, no_idle_flags, best_ms, time_limit_abs)
        if ls_ms < best_ms:
            best_ms = ls_ms
            best_seq = ls_seq
            if logger:
                logger.log(best_ms)
    
    # Step 3: Iterated Local Search (ILS)
    ils_seq = list(best_seq)
    ils_ms = best_ms
    
    no_improve_count = 0
    temperature = best_ms * 0.01  # For SA-like acceptance
    
    while time.time() < time_limit_abs:
        # Perturbation strength increases with no improvement
        strength = min(3 + no_improve_count // 10, max(4, n_jobs // 3))
        
        # Perturb
        new_seq = perturbation(ils_seq, n_jobs, strength)
        new_ms = compute_makespan_fast(new_seq, n_jobs, n_machines, processing_times, no_idle_flags)
        
        # Local search
        if time.time() < time_limit_abs:
            new_seq, new_ms = local_search_combined(new_seq, n_jobs, n_machines, processing_times, no_idle_flags, new_ms, time_limit_abs)
        
        # Acceptance criterion (SA-like)
        if new_ms < ils_ms:
            ils_seq = new_seq
            ils_ms = new_ms
            no_improve_count = 0
        else:
            delta = new_ms - ils_ms
            if temperature > 0 and random.random() < math.exp(-delta / max(temperature, 1e-10)):
                ils_seq = new_seq
                ils_ms = new_ms
            no_improve_count += 1
        
        # Cool down
        temperature *= 0.995
        
        if ils_ms < best_ms:
            best_ms = ils_ms
            best_seq = list(ils_seq)
            if logger:
                logger.log(best_ms)
        
        # Restart if stuck for too long
        if no_improve_count > 50:
            # Random restart with NEH-like construction
            jobs_order = list(range(n_jobs))
            random.shuffle(jobs_order)
            seq = [jobs_order[0]]
            for i in range(1, n_jobs):
                if time.time() > time_limit_abs:
                    break
                job = jobs_order[i]
                best_pos_ms = float('inf')
                best_pos = 0
                for pos in range(len(seq) + 1):
                    candidate = seq[:pos] + [job] + seq[pos:]
                    ms = compute_makespan_fast(candidate, len(candidate), n_machines, processing_times, no_idle_flags)
                    if ms < best_pos_ms:
                        best_pos_ms = ms
                        best_pos = pos
                seq = seq[:best_pos] + [job] + seq[best_pos:]
            
            ms = compute_makespan_fast(seq, n_jobs, n_machines, processing_times, no_idle_flags)
            ils_seq = seq
            ils_ms = ms
            no_improve_count = 0
            temperature = best_ms * 0.01
            
            if ils_ms < best_ms:
                best_ms = ils_ms
                best_seq = list(ils_seq)
                if logger:
                    logger.log(best_ms)
    
    # Compute full completion times for output
    final_ms, C = compute_makespan(best_seq, n_jobs, n_machines, processing_times, no_idle_flags)
    
    completion_times = {}
    for m in range(n_machines):
        completion_times[f'machine_{m}'] = [C[k][m] for k in range(n_jobs)]
    
    solution = {
        'objective_value': final_ms,
        'sequence': best_seq,
        'completion_times': completion_times
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == '__main__':
    solve()
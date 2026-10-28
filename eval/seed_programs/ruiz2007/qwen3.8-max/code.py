import json
import argparse
import time
import random
from itertools import permutations
from solution_logger import SolutionLogger


def compute_completion_times(sequence, processing_times, n_machines, no_idle_flags):
    """Compute completion times for a given sequence."""
    n = len(sequence)
    if n == 0:
        return [[] for _ in range(n_machines)]
    
    C = [[0] * n for _ in range(n_machines)]
    
    # Machine 0 (first machine)
    C[0][0] = processing_times[sequence[0]][0]
    for k in range(1, n):
        C[0][k] = C[0][k - 1] + processing_times[sequence[k]][0]
    
    # Subsequent machines
    for j in range(1, n_machines):
        if no_idle_flags[j]:
            # No-idle machine: compute prefix sums of processing times
            prefix = [0] * n
            prefix[0] = processing_times[sequence[0]][j]
            for k in range(1, n):
                prefix[k] = prefix[k - 1] + processing_times[sequence[k]][j]
            
            # Find earliest valid start time
            start_j = C[j - 1][0]
            for k in range(1, n):
                required = C[j - 1][k] - prefix[k - 1]
                if required > start_j:
                    start_j = required
            
            for k in range(n):
                C[j][k] = start_j + prefix[k]
        else:
            # Regular machine
            C[j][0] = C[j - 1][0] + processing_times[sequence[0]][j]
            for k in range(1, n):
                C[j][k] = max(C[j][k - 1], C[j - 1][k]) + processing_times[sequence[k]][j]
    
    return C


def evaluate(sequence, processing_times, n_machines, no_idle_flags):
    """Evaluate makespan for a given sequence."""
    n = len(sequence)
    if n == 0:
        return 0
    C = compute_completion_times(sequence, processing_times, n_machines, no_idle_flags)
    return C[n_machines - 1][n - 1]


def build_solution_dict(sequence, C, makespan, n_machines):
    """Build solution dictionary in required format."""
    solution_dict = {
        "objective_value": float(makespan),
        "sequence": list(sequence),
        "completion_times": {}
    }
    for j in range(n_machines):
        solution_dict["completion_times"][f"machine_{j}"] = list(C[j])
    return solution_dict


def neh_heuristic(processing_times, n_jobs, n_machines, no_idle_flags, time_limit, start_time):
    """NEH construction heuristic."""
    # Order jobs by total processing time (descending)
    total_times = [(sum(processing_times[j]), j) for j in range(n_jobs)]
    total_times.sort(reverse=True)
    job_order = [j for _, j in total_times]
    
    sequence = [job_order[0]]
    
    for i in range(1, n_jobs):
        if time.time() - start_time > time_limit * 0.3:
            # If running low on time, just append remaining jobs
            sequence.extend(job_order[i:])
            break
        
        job = job_order[i]
        best_pos = 0
        best_makespan = float('inf')
        
        for pos in range(len(sequence) + 1):
            new_seq = sequence[:pos] + [job] + sequence[pos:]
            makespan = evaluate(new_seq, processing_times, n_machines, no_idle_flags)
            if makespan < best_makespan:
                best_makespan = makespan
                best_pos = pos
        
        sequence = sequence[:best_pos] + [job] + sequence[best_pos:]
    
    return sequence


def local_search_best_improvement(sequence, processing_times, n_machines, no_idle_flags, time_limit, start_time):
    """Best-improvement local search with insert neighborhood."""
    n = len(sequence)
    best_seq = sequence[:]
    best_makespan = evaluate(best_seq, processing_times, n_machines, no_idle_flags)
    
    improved = True
    while improved:
        if time.time() - start_time > time_limit * 0.85:
            break
        improved = False
        best_move = None
        best_move_makespan = best_makespan
        
        for i in range(n):
            if time.time() - start_time > time_limit * 0.85:
                break
            job = best_seq[i]
            for j in range(n):
                if i == j:
                    continue
                new_seq = best_seq[:i] + best_seq[i + 1:]
                new_seq.insert(j, job)
                makespan = evaluate(new_seq, processing_times, n_machines, no_idle_flags)
                if makespan < best_move_makespan:
                    best_move_makespan = makespan
                    best_move = (i, j)
        
        if best_move is not None:
            i, j = best_move
            job = best_seq[i]
            best_seq = best_seq[:i] + best_seq[i + 1:]
            best_seq.insert(j, job)
            best_makespan = best_move_makespan
            improved = True
    
    return best_seq, best_makespan


def local_search_first_improvement(sequence, processing_times, n_machines, no_idle_flags, time_limit, start_time):
    """First-improvement local search with insert neighborhood."""
    n = len(sequence)
    best_seq = sequence[:]
    best_makespan = evaluate(best_seq, processing_times, n_machines, no_idle_flags)
    
    improved = True
    while improved:
        if time.time() - start_time > time_limit * 0.85:
            break
        improved = False
        
        for i in range(n):
            if time.time() - start_time > time_limit * 0.85:
                break
            job = best_seq[i]
            for j in range(n):
                if i == j:
                    continue
                new_seq = best_seq[:i] + best_seq[i + 1:]
                new_seq.insert(j, job)
                makespan = evaluate(new_seq, processing_times, n_machines, no_idle_flags)
                if makespan < best_makespan:
                    best_makespan = makespan
                    best_seq = new_seq
                    improved = True
                    break
            if improved:
                break
    
    return best_seq, best_makespan


def local_search_swap(sequence, processing_times, n_machines, no_idle_flags, time_limit, start_time):
    """Local search with swap neighborhood."""
    n = len(sequence)
    best_seq = sequence[:]
    best_makespan = evaluate(best_seq, processing_times, n_machines, no_idle_flags)
    
    improved = True
    while improved:
        if time.time() - start_time > time_limit * 0.85:
            break
        improved = False
        
        for i in range(n):
            if time.time() - start_time > time_limit * 0.85:
                break
            for j in range(i + 1, n):
                new_seq = best_seq[:]
                new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
                makespan = evaluate(new_seq, processing_times, n_machines, no_idle_flags)
                if makespan < best_makespan:
                    best_makespan = makespan
                    best_seq = new_seq
                    improved = True
                    break
            if improved:
                break
    
    return best_seq, best_makespan


def perturbation(sequence, strength):
    """Random perturbation for ILS."""
    n = len(sequence)
    perturbed = sequence[:]
    for _ in range(strength):
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i != j:
            job = perturbed.pop(i)
            perturbed.insert(j, job)
    return perturbed


def ils(sequence, processing_times, n_machines, no_idle_flags, time_limit, start_time, logger):
    """Iterated Local Search."""
    n = len(sequence)
    best_seq = sequence[:]
    best_makespan = evaluate(best_seq, processing_times, n_machines, no_idle_flags)
    current_seq = best_seq[:]
    current_makespan = best_makespan
    
    # Determine perturbation strength based on problem size
    strength = max(1, min(4, n // 10))
    
    iteration = 0
    while time.time() - start_time < time_limit * 0.9:
        iteration += 1
        
        # Perturbation
        perturbed = perturbation(current_seq, strength)
        
        # Local search on perturbed solution
        if n <= 30:
            perturbed, perturbed_makespan = local_search_best_improvement(
                perturbed, processing_times, n_machines, no_idle_flags, time_limit, start_time)
        else:
            perturbed, perturbed_makespan = local_search_first_improvement(
                perturbed, processing_times, n_machines, no_idle_flags, time_limit, start_time)
        
        if perturbed_makespan < best_makespan:
            best_makespan = perturbed_makespan
            best_seq = perturbed[:]
            current_seq = perturbed[:]
            current_makespan = perturbed_makespan
            
            if logger:
                C = compute_completion_times(best_seq, processing_times, n_machines, no_idle_flags)
                solution_dict = build_solution_dict(best_seq, C, best_makespan, n_machines)
                logger.log_solution(best_makespan, solution_dict)
        elif perturbed_makespan <= current_makespan:
            # Accept if better than current (diversification)
            current_seq = perturbed[:]
            current_makespan = perturbed_makespan
        else:
            # Accept with small probability for diversification
            if random.random() < 0.1:
                current_seq = perturbed[:]
                current_makespan = perturbed_makespan
        
        # Check time after each iteration
        if time.time() - start_time > time_limit * 0.9:
            break
    
    return best_seq, best_makespan


def exact_enumeration(processing_times, n_jobs, n_machines, no_idle_flags, time_limit, start_time, logger):
    """Exact solution by enumerating all permutations (for very small instances)."""
    best_seq = None
    best_makespan = float('inf')
    
    jobs = list(range(n_jobs))
    count = 0
    total_perms = 1
    for i in range(2, n_jobs + 1):
        total_perms *= i
    
    for perm in permutations(jobs):
        count += 1
        if count % 1000 == 0 and time.time() - start_time > time_limit * 0.9:
            break
        
        seq = list(perm)
        makespan = evaluate(seq, processing_times, n_machines, no_idle_flags)
        if makespan < best_makespan:
            best_makespan = makespan
            best_seq = seq[:]
            
            if logger:
                C = compute_completion_times(seq, processing_times, n_machines, no_idle_flags)
                solution_dict = build_solution_dict(seq, C, best_makespan, n_machines)
                logger.log_solution(best_makespan, solution_dict)
    
    return best_seq, best_makespan


def solve():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    random.seed(42)
    
    # Read instance
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    n_jobs = instance['n_jobs']
    n_machines = instance['n_machines']
    no_idle_flags = instance['no_idle_flags']
    processing_times = instance['processing_times']
    
    start_time = time.time()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    # Choose strategy based on problem size
    if n_jobs <= 7:
        # Exact enumeration for very small instances
        sequence, best_makespan = exact_enumeration(
            processing_times, n_jobs, n_machines, no_idle_flags, args.time_limit, start_time, logger)
    else:
        # NEH construction
        sequence = neh_heuristic(processing_times, n_jobs, n_machines, no_idle_flags, args.time_limit, start_time)
        best_makespan = evaluate(sequence, processing_times, n_machines, no_idle_flags)
        
        # Log initial solution
        C = compute_completion_times(sequence, processing_times, n_machines, no_idle_flags)
        solution_dict = build_solution_dict(sequence, C, best_makespan, n_machines)
        if logger:
            logger.log_solution(best_makespan, solution_dict)
        
        # Local search
        if time.time() - start_time < args.time_limit * 0.7:
            if n_jobs <= 30:
                sequence, ls_makespan = local_search_best_improvement(
                    sequence, processing_times, n_machines, no_idle_flags, args.time_limit, start_time)
            else:
                sequence, ls_makespan = local_search_first_improvement(
                    sequence, processing_times, n_machines, no_idle_flags, args.time_limit, start_time)
            
            if ls_makespan < best_makespan:
                best_makespan = ls_makespan
                C = compute_completion_times(sequence, processing_times, n_machines, no_idle_flags)
                solution_dict = build_solution_dict(sequence, C, best_makespan, n_machines)
                if logger:
                    logger.log_solution(best_makespan, solution_dict)
        
        # Swap neighborhood
        if time.time() - start_time < args.time_limit * 0.75:
            sequence, swap_makespan = local_search_swap(
                sequence, processing_times, n_machines, no_idle_flags, args.time_limit, start_time)
            if swap_makespan < best_makespan:
                best_makespan = swap_makespan
                C = compute_completion_times(sequence, processing_times, n_machines, no_idle_flags)
                solution_dict = build_solution_dict(sequence, C, best_makespan, n_machines)
                if logger:
                    logger.log_solution(best_makespan, solution_dict)
        
        # ILS for further improvement
        if time.time() - start_time < args.time_limit * 0.8:
            sequence, ils_makespan = ils(
                sequence, processing_times, n_machines, no_idle_flags, args.time_limit, start_time, logger)
            if ils_makespan < best_makespan:
                best_makespan = ils_makespan
    
    # Final solution
    C = compute_completion_times(sequence, processing_times, n_machines, no_idle_flags)
    solution_dict = build_solution_dict(sequence, C, best_makespan, n_machines)
    
    # Write solution
    with open(args.solution_path, 'w') as f:
        json.dump(solution_dict, f)


if __name__ == '__main__':
    solve()
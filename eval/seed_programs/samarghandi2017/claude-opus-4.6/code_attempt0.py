import json
import argparse
import time
import itertools
import random

def solve(instance_path, solution_path, time_limit, log_path):
    from solution_logger import SolutionLogger
    
    with open(instance_path, 'r') as f:
        data = json.load(f)
    
    n_jobs = data['n_jobs']
    n_machines = data['n_machines']
    processing_times = data['processing_times']  # [m][j]
    due_dates = data['due_dates']
    
    start_time = time.time()
    
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    # For no-wait: once job j starts on machine 0 at time s_j,
    # it starts on machine m at time s_j + sum(processing_times[k][j] for k=0..m-1)
    # Completion on last machine: s_j + sum(processing_times[k][j] for k=0..n_machines-1)
    
    # Compute total processing time for each job (from start on machine 0 to finish on last machine)
    total_time = [0] * n_jobs
    for j in range(n_jobs):
        total_time[j] = sum(processing_times[m][j] for m in range(n_machines))
    
    # Compute cumulative processing times: cum[m][j] = sum of processing_times[0..m-1][j]
    # This is the offset of job j's start on machine m from its start on machine 0
    cum = [[0] * n_jobs for _ in range(n_machines)]
    for j in range(n_jobs):
        for m in range(1, n_machines):
            cum[m][j] = cum[m-1][j] + processing_times[m-1][j]
    
    # Minimum delay between job i (predecessor) and job j (successor):
    # On each machine m, job j's start on m >= job i's start on m + processing_times[m][i]
    # s_i + cum[m][i] + processing_times[m][i] <= s_j + cum[m][j]
    # s_j - s_i >= cum[m][i] + processing_times[m][i] - cum[m][j]
    # delay[i][j] = max over m of (cum[m][i] + processing_times[m][i] - cum[m][j])
    
    delay = [[0] * n_jobs for _ in range(n_jobs)]
    for i in range(n_jobs):
        for j in range(n_jobs):
            if i == j:
                continue
            max_d = 0
            for m in range(n_machines):
                d = cum[m][i] + processing_times[m][i] - cum[m][j]
                if d > max_d:
                    max_d = d
            delay[i][j] = max_d
    
    def compute_makespan_and_feasibility(seq):
        """Compute makespan for a given sequence and check due date feasibility."""
        if len(seq) == 0:
            return 0, True
        
        # Start times on machine 0
        starts = [0] * len(seq)
        starts[0] = 0
        for k in range(1, len(seq)):
            starts[k] = starts[k-1] + delay[seq[k-1]][seq[k]]
        
        # Check due dates and compute makespan
        makespan = 0
        feasible = True
        for k in range(len(seq)):
            j = seq[k]
            completion = starts[k] + total_time[j]
            if completion > due_dates[j]:
                feasible = False
                break
            if completion > makespan:
                makespan = completion
        
        return makespan, feasible
    
    def compute_makespan_fast(seq):
        """Compute makespan without due date check."""
        if len(seq) == 0:
            return 0
        n = len(seq)
        starts = [0] * n
        for k in range(1, n):
            starts[k] = starts[k-1] + delay[seq[k-1]][seq[k]]
        
        makespan = 0
        for k in range(n):
            c = starts[k] + total_time[seq[k]]
            if c > makespan:
                makespan = c
        return makespan
    
    def is_feasible(seq):
        if len(seq) == 0:
            return True
        n = len(seq)
        starts = [0] * n
        for k in range(1, n):
            starts[k] = starts[k-1] + delay[seq[k-1]][seq[k]]
        for k in range(n):
            if starts[k] + total_time[seq[k]] > due_dates[seq[k]]:
                return False
        return True
    
    def evaluate_insertion(seq, job, pos):
        """Evaluate inserting job at position pos in seq. Returns (makespan, feasible)."""
        new_seq = seq[:pos] + [job] + seq[pos:]
        return compute_makespan_and_feasibility(new_seq)
    
    # NEH heuristic
    # Sort jobs by decreasing total processing time
    sorted_jobs = sorted(range(n_jobs), key=lambda j: -total_time[j])
    
    best_seq = None
    best_makespan = float('inf')
    
    def try_neh(job_order):
        nonlocal best_seq, best_makespan
        seq = []
        for j in job_order:
            best_pos = None
            best_ms = float('inf')
            for pos in range(len(seq) + 1):
                ms, feas = evaluate_insertion(seq, j, pos)
                if feas and ms < best_ms:
                    best_ms = ms
                    best_pos = pos
            if best_pos is not None:
                seq = seq[:best_pos] + [j] + seq[best_pos:]
            else:
                # Try to find any feasible position
                # Insert at position that minimizes infeasibility (or just try all)
                found = False
                for pos in range(len(seq) + 1):
                    trial = seq[:pos] + [j] + seq[pos:]
                    if is_feasible(trial):
                        seq = trial
                        found = True
                        break
                if not found:
                    # Force insert at best makespan position regardless
                    best_pos_forced = 0
                    best_ms_forced = float('inf')
                    for pos in range(len(seq) + 1):
                        trial = seq[:pos] + [j] + seq[pos:]
                        ms = compute_makespan_fast(trial)
                        if ms < best_ms_forced:
                            best_ms_forced = ms
                            best_pos_forced = pos
                    seq = seq[:best_pos_forced] + [j] + seq[best_pos_forced:]
        
        ms, feas = compute_makespan_and_feasibility(seq)
        if feas and ms < best_makespan:
            best_makespan = ms
            best_seq = seq[:]
            return seq
        return seq if feas else None
    
    try_neh(sorted_jobs)
    
    # Also try sorted by due date (EDD)
    edd_order = sorted(range(n_jobs), key=lambda j: due_dates[j])
    try_neh(edd_order)
    
    # Try reverse order
    try_neh(list(reversed(sorted_jobs)))
    
    # Try some random orders
    random.seed(42)
    for _ in range(min(20, max(1, n_jobs))):
        if time.time() - start_time > time_limit * 0.1:
            break
        perm = list(range(n_jobs))
        random.shuffle(perm)
        try_neh(perm)
    
    if best_seq is not None and logger:
        sol_dict = {
            "objective_value": best_makespan,
            "sequence": [j + 1 for j in best_seq]
        }
        logger.log_solution(best_makespan, sol_dict)
    
    # Local search: insertion neighborhood
    def local_search_insertion(seq, deadline):
        nonlocal best_seq, best_makespan
        if seq is None:
            return None
        improved = True
        current_seq = seq[:]
        current_ms = compute_makespan_fast(current_seq)
        
        while improved:
            if time.time() > deadline:
                break
            improved = False
            for i in range(len(current_seq)):
                if time.time() > deadline:
                    break
                job = current_seq[i]
                remaining = current_seq[:i] + current_seq[i+1:]
                for pos in range(len(remaining) + 1):
                    if pos == i:
                        continue  # same position
                    trial = remaining[:pos] + [job] + remaining[pos:]
                    ms, feas = compute_makespan_and_feasibility(trial)
                    if feas and ms < current_ms:
                        current_seq = trial
                        current_ms = ms
                        improved = True
                        if ms < best_makespan:
                            best_makespan = ms
                            best_seq = trial[:]
                            if logger:
                                sol_dict = {
                                    "objective_value": best_makespan,
                                    "sequence": [j + 1 for j in best_seq]
                                }
                                logger.log_solution(best_makespan, sol_dict)
                        break
                if improved:
                    break
        return current_seq
    
    # Local search: swap neighborhood  
    def local_search_swap(seq, deadline):
        nonlocal best_seq, best_makespan
        if seq is None:
            return None
        improved = True
        current_seq = seq[:]
        current_ms = compute_makespan_fast(current_seq)
        
        while improved:
            if time.time() > deadline:
                break
            improved = False
            for i in range(len(current_seq) - 1):
                if time.time() > deadline:
                    break
                for j_idx in range(i + 1, len(current_seq)):
                    trial = current_seq[:]
                    trial[i], trial[j_idx] = trial[j_idx], trial[i]
                    ms, feas = compute_makespan_and_feasibility(trial)
                    if feas and ms < current_ms:
                        current_seq = trial
                        current_ms = ms
                        improved = True
                        if ms < best_makespan:
                            best_makespan = ms
                            best_seq = trial[:]
                            if logger:
                                sol_dict = {
                                    "objective_value": best_makespan,
                                    "sequence": [j + 1 for j in best_seq]
                                }
                                logger.log_solution(best_makespan, sol_dict)
                        break
                if improved:
                    break
        return current_seq
    
    if best_seq is not None:
        deadline = start_time + time_limit * 0.5
        best_seq = local_search_insertion(best_seq, deadline)
        
        deadline = start_time + time_limit * 0.7
        best_seq = local_search_swap(best_seq, deadline)
        
        deadline = start_time + time_limit * 0.85
        best_seq = local_search_insertion(best_seq, deadline)
    
    # Iterated perturbation + local search
    def perturb(seq, strength=3):
        seq = seq[:]
        for _ in range(strength):
            if len(seq) < 2:
                break
            i = random.randint(0, len(seq) - 1)
            job = seq.pop(i)
            j = random.randint(0, len(seq))
            seq.insert(j, job)
        return seq
    
    if best_seq is not None:
        deadline = start_time + time_limit * 0.98
        current_seq = best_seq[:]
        current_ms = best_makespan
        no_improve_count = 0
        
        while time.time() < deadline:
            strength = random.randint(2, max(2, min(n_jobs // 3, 5)))
            candidate = perturb(current_seq, strength)
            candidate = local_search_insertion(candidate, min(time.time() + 2.0, deadline))
            if candidate is None:
                continue
            cand_ms, cand_feas = compute_makespan_and_feasibility(candidate)
            if cand_feas:
                if cand_ms < current_ms:
                    current_seq = candidate
                    current_ms = cand_ms
                    no_improve_count = 0
                    if cand_ms < best_makespan:
                        best_makespan = cand_ms
                        best_seq = candidate[:]
                        if logger:
                            sol_dict = {
                                "objective_value": best_makespan,
                                "sequence": [j + 1 for j in best_seq]
                            }
                            logger.log_solution(best_makespan, sol_dict)
                else:
                    no_improve_count += 1
                    # Accept with some probability (simulated annealing style)
                    if no_improve_count > 10:
                        current_seq = best_seq[:]
                        current_ms = best_makespan
                        no_improve_count = 0
    
    # If no feasible solution found, try brute force for small instances
    if best_seq is None and n_jobs <= 10:
        deadline = start_time + time_limit * 0.98
        for perm in itertools.permutations(range(n_jobs)):
            if time.time() > deadline:
                break
            ms, feas = compute_makespan_and_feasibility(list(perm))
            if feas and ms < best_makespan:
                best_makespan = ms
                best_seq = list(perm)
                if logger:
                    sol_dict = {
                        "objective_value": best_makespan,
                        "sequence": [j + 1 for j in best_seq]
                    }
                    logger.log_solution(best_makespan, sol_dict)
    
    # If still no feasible solution, return best infeasible as fallback
    if best_seq is None:
        # Just use EDD order
        best_seq = sorted(range(n_jobs), key=lambda j: due_dates[j])
        best_makespan = compute_makespan_fast(best_seq)
    
    solution = {
        "objective_value": float(best_makespan),
        "sequence": [j + 1 for j in best_seq]
    }
    
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)
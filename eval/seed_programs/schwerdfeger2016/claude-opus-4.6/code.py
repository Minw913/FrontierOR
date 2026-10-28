import argparse
import json
import math
import time
import sys
from solution_logger import SolutionLogger


def solve(instance_path, solution_path, time_limit, log_path=None):
    start_time = time.time()
    
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    
    num_machines = instance['num_machines']
    num_jobs = instance['num_jobs']
    processing_times = instance['processing_times']
    
    m = num_machines
    n = num_jobs
    p = processing_times
    
    total_time = sum(p)
    avg_time = total_time / m
    
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    def compute_objective(loads):
        sum_sq_dev = sum((load - avg_time) ** 2 for load in loads)
        if avg_time == 0:
            return 0.0
        return math.sqrt(sum_sq_dev) / avg_time
    
    def compute_sum_sq(loads):
        return sum(load * load for load in loads)
    
    # --- LPT heuristic as initial solution ---
    def lpt_heuristic():
        loads = [0] * m
        assignment = [[] for _ in range(m)]
        # Jobs are already sorted in nonincreasing order
        for j in range(n):
            # Assign to machine with minimum load
            min_idx = 0
            min_load = loads[0]
            for i in range(1, m):
                if loads[i] < min_load:
                    min_load = loads[i]
                    min_idx = i
            loads[min_idx] += p[j]
            assignment[min_idx].append(j)
        return loads, assignment
    
    best_loads, best_assignment = lpt_heuristic()
    best_obj = compute_objective(best_loads)
    best_sum_sq = compute_sum_sq(best_loads)
    
    if logger:
        logger.log(best_obj)
    
    # --- Local search improvement ---
    def local_search(loads, assignment, deadline):
        """Improve solution using move and swap neighborhood."""
        loads = list(loads)
        assignment = [list(a) for a in assignment]
        current_sum_sq = compute_sum_sq(loads)
        improved = True
        
        while improved and time.time() < deadline:
            improved = False
            
            # Try move: move a job from one machine to another
            for i in range(m):
                if time.time() >= deadline:
                    break
                for j_idx in range(len(assignment[i])):
                    if time.time() >= deadline:
                        break
                    j = assignment[i][j_idx]
                    pj = p[j]
                    for k in range(m):
                        if k == i:
                            continue
                        # Compute new sum of squares
                        new_load_i = loads[i] - pj
                        new_load_k = loads[k] + pj
                        delta = (new_load_i**2 - loads[i]**2) + (new_load_k**2 - loads[k]**2)
                        if delta < -1e-9:
                            loads[i] = new_load_i
                            loads[k] = new_load_k
                            assignment[i].pop(j_idx)
                            assignment[k].append(j)
                            current_sum_sq += delta
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                # Try swap: swap jobs between two machines
                for i in range(m):
                    if time.time() >= deadline:
                        break
                    found = False
                    for k in range(i + 1, m):
                        if time.time() >= deadline:
                            break
                        for ji_idx in range(len(assignment[i])):
                            if found or time.time() >= deadline:
                                break
                            ji = assignment[i][ji_idx]
                            pji = p[ji]
                            for jk_idx in range(len(assignment[k])):
                                jk = assignment[k][jk_idx]
                                pjk = p[jk]
                                diff = pji - pjk
                                if diff == 0:
                                    continue
                                new_load_i = loads[i] - diff
                                new_load_k = loads[k] + diff
                                delta = (new_load_i**2 - loads[i]**2) + (new_load_k**2 - loads[k]**2)
                                if delta < -1e-9:
                                    loads[i] = new_load_i
                                    loads[k] = new_load_k
                                    assignment[i][ji_idx] = jk
                                    assignment[k][jk_idx] = ji
                                    current_sum_sq += delta
                                    improved = True
                                    found = True
                                    break
                        if found:
                            break
                    if found:
                        break
        
        return loads, assignment
    
    deadline = start_time + time_limit - 0.5
    
    best_loads, best_assignment = local_search(best_loads, best_assignment, deadline)
    best_obj = compute_objective(best_loads)
    best_sum_sq = compute_sum_sq(best_loads)
    
    if logger:
        logger.log(best_obj)
    
    # --- More advanced: try multiple restarts with perturbation ---
    import random
    random.seed(42)
    
    def perturb(loads, assignment, strength=3):
        """Randomly move some jobs to random machines."""
        loads = list(loads)
        assignment = [list(a) for a in assignment]
        moves = min(strength, n)
        for _ in range(moves):
            # Pick a random non-empty machine
            non_empty = [i for i in range(m) if len(assignment[i]) > 0]
            if not non_empty:
                break
            src = random.choice(non_empty)
            if len(assignment[src]) == 0:
                continue
            j_idx = random.randint(0, len(assignment[src]) - 1)
            j = assignment[src][j_idx]
            dst = random.randint(0, m - 2)
            if dst >= src:
                dst += 1
            loads[src] -= p[j]
            loads[dst] += p[j]
            assignment[src].pop(j_idx)
            assignment[dst].append(j)
        return loads, assignment
    
    # Iterated local search
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        # Perturb current best
        strength = random.randint(1, max(1, n // 3))
        new_loads, new_assignment = perturb(best_loads, best_assignment, strength)
        
        sub_deadline = min(deadline, time.time() + max(0.5, (deadline - time.time()) / 10))
        new_loads, new_assignment = local_search(new_loads, new_assignment, sub_deadline)
        
        new_sum_sq = compute_sum_sq(new_loads)
        if new_sum_sq < best_sum_sq - 1e-9:
            best_sum_sq = new_sum_sq
            best_loads = new_loads
            best_assignment = new_assignment
            best_obj = compute_objective(best_loads)
            if logger:
                logger.log(best_obj)
    
    # If perfect balance is possible, check it
    if total_time % m == 0:
        target = total_time // m
        # Try DP-based approach for small instances
        if n <= 30 and m <= 5 and time.time() < start_time + time_limit - 0.1:
            pass  # The local search should handle small instances well
    
    # Build output
    assignment_dict = {}
    completion_dict = {}
    for i in range(m):
        key = str(i)
        assignment_dict[key] = best_assignment[i]
        completion_dict[key] = best_loads[i]
    
    solution = {
        "objective_value": best_obj,
        "assignment": assignment_dict,
        "machine_completion_times": completion_dict
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
    
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)
import sys
import time
import json
import random
import math
import argparse
from solution_logger import SolutionLogger

def get_makespan(seq, p_times, no_idle_flags, num_machines):
    n = len(seq)
    C_prev = [0] * n
    C_curr = [0] * n
    
    for k in range(num_machines):
        is_no_idle = no_idle_flags[k]
        
        if not is_no_idle:
            c = C_prev[0] + p_times[seq[0]][k]
            C_curr[0] = c
            for i in range(1, n):
                prev_c = C_prev[i]
                p = p_times[seq[i]][k]
                if prev_c > c:
                    c = prev_c + p
                else:
                    c = c + p
                C_curr[i] = c
        else:
            max_S = 0
            curr_sum = 0
            for i in range(n):
                if C_prev[i] - curr_sum > max_S:
                    max_S = C_prev[i] - curr_sum
                curr_sum += p_times[seq[i]][k]
            
            c = max_S + p_times[seq[0]][k]
            C_curr[0] = c
            for i in range(1, n):
                c += p_times[seq[i]][k]
                C_curr[i] = c
                
        C_prev, C_curr = C_curr, C_prev
        
    return C_prev[-1]

def destruct_construct(seq, d, p_times, no_idle_flags, num_machines):
    removed_jobs = random.sample(seq, d)
    removed_set = set(removed_jobs)
    new_seq = [j for j in seq if j not in removed_set]
    
    best_mk = float('inf')
    
    for job in removed_jobs:
        best_pos = -1
        best_mk = float('inf')
        for pos in range(len(new_seq) + 1):
            new_seq.insert(pos, job)
            mk = get_makespan(new_seq, p_times, no_idle_flags, num_machines)
            if mk < best_mk:
                best_mk = mk
                best_pos = pos
            new_seq.pop(pos)
        new_seq.insert(best_pos, job)
        
    return new_seq, best_mk

def local_search_V2(seq, p_times, no_idle_flags, num_machines, mk_current, max_time_for_ig, global_start_time, global_best_mk, log_solution):
    n = len(seq)
    improved_overall = True
    best_seq_tracker = None
    
    while improved_overall and (time.time() - global_start_time < max_time_for_ig):
        improved_overall = False
        
        job_order = list(seq)
        random.shuffle(job_order)
        
        for job in job_order:
            if time.time() - global_start_time >= max_time_for_ig:
                break
            
            idx = seq.index(job)
            seq.pop(idx)
            best_pos = idx
            
            for pos in range(n):
                if pos == idx:
                    continue
                seq.insert(pos, job)
                mk = get_makespan(seq, p_times, no_idle_flags, num_machines)
                
                if mk < global_best_mk:
                    global_best_mk = mk
                    best_seq_tracker = list(seq)
                    log_solution(mk)
                    
                if mk < mk_current:
                    mk_current = mk
                    best_pos = pos
                    improved_overall = True
                    
                seq.pop(pos)
                
            seq.insert(best_pos, job)
            
    return seq, mk_current, global_best_mk, best_seq_tracker

def neh_initial(jobs, p_times, no_idle_flags, num_machines, max_time, global_start_time):
    sum_p = [sum(p_times[j]) for j in jobs]
    sorted_jobs = sorted(jobs, key=lambda j: sum_p[j], reverse=True)
    
    seq = [sorted_jobs[0]]
    for i in range(1, len(sorted_jobs)):
        if time.time() - global_start_time >= max_time:
            for rem_i in range(i, len(sorted_jobs)):
                seq.append(sorted_jobs[rem_i])
            return seq, get_makespan(seq, p_times, no_idle_flags, num_machines)
            
        job = sorted_jobs[i]
        best_pos = -1
        best_mk = float('inf')
        
        for pos in range(len(seq) + 1):
            seq.insert(pos, job)
            mk = get_makespan(seq, p_times, no_idle_flags, num_machines)
            if mk < best_mk:
                best_mk = mk
                best_pos = pos
            seq.pop(pos)
            
        seq.insert(best_pos, job)
        
    return seq, get_makespan(seq, p_times, no_idle_flags, num_machines)

def iterated_greedy(p_times, no_idle_flags, num_machines, num_jobs, max_time_for_ig, global_start_time, log_solution, initial_seq, initial_mk):
    current_seq = initial_seq.copy()
    current_mk = initial_mk
    best_seq_saved = initial_seq.copy()
    best_mk = initial_mk
    
    sum_p = sum(p_times[j][k] for j in range(num_jobs) for k in range(num_machines))
    T_0 = 0.4 * sum_p / (num_jobs * num_machines) if (num_jobs * num_machines) > 0 else 1.0
    d = min(4, num_jobs)
    
    while time.time() - global_start_time < max_time_for_ig:
        new_seq, new_mk = destruct_construct(current_seq, d, p_times, no_idle_flags, num_machines)
        
        if new_mk < best_mk:
            best_mk = new_mk
            best_seq_saved = list(new_seq)
            log_solution(best_mk)
            
        new_seq, new_mk, best_mk, tracker_seq = local_search_V2(
            new_seq, p_times, no_idle_flags, num_machines, new_mk,
            max_time_for_ig, global_start_time, best_mk, log_solution
        )
        
        if tracker_seq is not None:
            best_seq_saved = tracker_seq
            
        if new_mk <= current_mk:
            current_seq = new_seq
            current_mk = new_mk
        else:
            delta = new_mk - current_mk
            prob = math.exp(-delta / T_0) if T_0 > 0 else 0
            if random.random() <= prob:
                current_seq = new_seq
                current_mk = new_mk
                
    return best_seq_saved, best_mk

def compute_full_completion_times(seq, p_times, no_idle_flags, num_machines):
    n = len(seq)
    C = [[0] * num_machines for _ in range(n)]
    
    for k in range(num_machines):
        if not no_idle_flags[k]:
            C[0][k] = (C[0][k-1] if k > 0 else 0) + p_times[seq[0]][k]
            for i in range(1, n):
                prev_c = C[i][k-1] if k > 0 else 0
                C[i][k] = max(prev_c, C[i-1][k]) + p_times[seq[i]][k]
        else:
            max_S = 0
            curr_sum = 0
            for i in range(n):
                prev_c = C[i][k-1] if k > 0 else 0
                S_req = prev_c - curr_sum
                if S_req > max_S:
                    max_S = S_req
                curr_sum += p_times[seq[i]][k]
            
            C[0][k] = max_S + p_times[seq[0]][k]
            for i in range(1, n):
                C[i][k] = C[i-1][k] + p_times[seq[i]][k]
                
    comp_times_dict = {}
    for k in range(num_machines):
        comp_times_dict[f"machine_{k}"] = [C[i][k] for i in range(n)]
        
    return comp_times_dict

def solve(instance_data, time_limit, log_path):
    global_start_time = time.time()
    
    num_jobs = instance_data["n_jobs"]
    num_machines = instance_data["n_machines"]
    no_idle_flags = instance_data["no_idle_flags"]
    p_times = instance_data["processing_times"]
    
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    def log_solution(obj):
        if logger:
            logger.log(float(obj))
            
    max_time_for_ig = time_limit - 0.5 
            
    best_seq, best_mk = neh_initial(
        list(range(num_jobs)), p_times, no_idle_flags, num_machines,
        max_time_for_ig, global_start_time
    )
    log_solution(best_mk)
    
    if time.time() - global_start_time < max_time_for_ig:
        best_seq, best_mk = iterated_greedy(
            p_times, no_idle_flags, num_machines, num_jobs, 
            max_time_for_ig, global_start_time, log_solution, best_seq, best_mk
        )
        
    comp_times = compute_full_completion_times(best_seq, p_times, no_idle_flags, num_machines)
    
    return best_seq, best_mk, comp_times

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, required=False)
    args = parser.parse_args()
    
    with open(args.instance_path, "r") as f:
        instance_data = json.load(f)
        
    best_seq, best_mk, comp_times = solve(instance_data, args.time_limit, args.log_path)
    
    solution = {
        "objective_value": float(best_mk),
        "sequence": best_seq,
        "completion_times": comp_times
    }
    
    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)
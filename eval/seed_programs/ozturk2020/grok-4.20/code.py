import json
import argparse
import time
import sys
import math
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    N = instance['num_jobs']
    M = instance['num_machines']
    Cap = instance['batch_capacity']
    jobs = instance['jobs']
    
    # Preprocess jobs (0-based)
    job_list = []
    for j in jobs:
        job_list.append({
            'r': j['release_date'],
            'p': j['processing_time'],
            's': j['size'],
            'id': j['job_id']
        })
    
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    start_time = time.time()
    time_limit = args.time_limit
    
    best_obj = float('inf')
    best_batches = None
    best_assign = None
    best_job_assign_dict = None
    
    def build_full_solution(machine_batches):
        # machine_batches[m] = list of lists (batches), each batch is list of job indices (0..N-1)
        batch_list = []
        job_assign_dict = {}
        objective = 0.0
        machine_current_time = [0.0] * M
        
        batch_global_idx = 0
        for m in range(M):
            current_time = 0.0
            for b_idx, batch_jobs in enumerate(machine_batches[m]):
                if not batch_jobs:
                    continue
                max_r = max(job_list[j]['r'] for j in batch_jobs)
                max_p = max(job_list[j]['p'] for j in batch_jobs)
                start_t = max(current_time, float(max_r))
                finish_t = start_t + max_p
                batch_id = b_idx + 1
                machine_id = m + 1
                job_ids = [job_list[j]['id'] for j in batch_jobs]
                
                batch_list.append({
                    'batch_id': batch_id,
                    'machine': machine_id,
                    'start_time': start_t,
                    'processing_time': max_p,
                    'jobs': job_ids
                })
                
                for j in batch_jobs:
                    jid = str(job_list[j]['id'])
                    job_assign_dict[jid] = {
                        'batch': batch_id,
                        'machine': machine_id,
                        'completion_time': finish_t
                    }
                    objective += finish_t
                
                current_time = finish_t
            machine_current_time[m] = current_time
        
        return objective, batch_list, job_assign_dict
    
    # Multiple heuristic strategies
    strategies = [
        lambda i: (job_list[i]['r'], job_list[i]['p']),           # primarily by release
        lambda i: (job_list[i]['p'], job_list[i]['r']),           # primarily by processing time
        lambda i: job_list[i]['r'] + 0.1 * job_list[i]['p'],      # weighted
        lambda i: job_list[i]['p'] * (1.0 + job_list[i]['s']),    # size-aware
        lambda i: job_list[i]['r'] * job_list[i]['s'],            # release-size
    ]
    
    for strat_idx, sort_key in enumerate(strategies):
        if time.time() - start_time > time_limit * 0.85:
            break
        
        # Get job order
        job_order = sorted(range(N), key=sort_key)
        
        # Greedy assignment to machines and batches
        machine_batches = [[] for _ in range(M)]  # list of batches, each batch is list of job indices
        machine_finish = [0.0] * M
        job_to_machine_batch = [-1] * N  # temporary
        
        for j_idx in job_order:
            j = job_list[j_idx]
            best_cost = float('inf')
            best_m = -1
            best_b_idx = -1
            best_is_new = False
            
            for m in range(M):
                # Try appending to each existing batch on machine m
                for b_idx, batch in enumerate(machine_batches[m]):
                    total_size = sum(job_list[jj]['s'] for jj in batch)
                    if total_size + j['s'] > Cap:
                        continue
                    max_r = max([job_list[jj]['r'] for jj in batch] + [j['r']])
                    max_p = max([job_list[jj]['p'] for jj in batch] + [j['p']])
                    start_time = max(machine_finish[m] if b_idx == len(machine_batches[m])-1 else 0.0, float(max_r))
                    finish_time = start_time + max_p
                    cost = finish_time * 1.0 + (finish_time - j['r']) * 0.1  # primary finish, secondary flow
                    
                    if cost < best_cost:
                        best_cost = cost
                        best_m = m
                        best_b_idx = b_idx
                        best_is_new = False
                
                # Option: new batch at the end of this machine
                start_time_new = max(machine_finish[m], float(j['r']))
                finish_time_new = start_time_new + j['p']
                cost_new = finish_time_new * 1.2 + (finish_time_new - machine_finish[m]) * 0.3
                
                if cost_new < best_cost:
                    best_cost = cost_new
                    best_m = m
                    best_b_idx = len(machine_batches[m])
                    best_is_new = True
            
            if best_m == -1:
                # Emergency: put in own batch on least loaded machine
                best_m = int(min(range(M), key=lambda m2: machine_finish[m2]))
                best_b_idx = len(machine_batches[best_m])
                best_is_new = True
            
            if best_is_new:
                machine_batches[best_m].append([j_idx])
            else:
                machine_batches[best_m][best_b_idx].append(j_idx)
            
            # Update machine finish time (conservative)
            if best_is_new or best_b_idx == len(machine_batches[best_m]) - 1:
                max_r = max(job_list[jj]['r'] for jj in machine_batches[best_m][-1])
                max_p = max(job_list[jj]['p'] for jj in machine_batches[best_m][-1])
                start_t = max(machine_finish[best_m], float(max_r))
                machine_finish[best_m] = start_t + max_p
        
        # Compute exact objective
        obj, batch_list, job_dict = build_full_solution(machine_batches)
        
        if obj < best_obj:
            best_obj = obj
            best_batches = batch_list
            best_job_assign_dict = job_dict
            if logger:
                logger.log(float(obj))
        
        # Light local improvement: try moving one job at a time (limited iterations)
        if N <= 30 and time.time() - start_time < time_limit * 0.6:
            for iter_local in range(3):
                improved = False
                for j_move in range(N):
                    if time.time() - start_time > time_limit * 0.75:
                        break
                    current_m = None
                    current_b = None
                    for m_cand in range(M):
                        for b_cand, bjobs in enumerate(machine_batches[m_cand]):
                            if j_move in bjobs:
                                current_m = m_cand
                                current_b = b_cand
                                break
                        if current_m is not None:
                            break
                    if current_m is None:
                        continue
                    
                    original_batch = machine_batches[current_m][current_b][:]
                    machine_batches[current_m][current_b].remove(j_move)
                    if not machine_batches[current_m][current_b]:
                        machine_batches[current_m].pop(current_b)
                    
                    # Try all possible insertions
                    best_local_m = current_m
                    best_local_b = current_b
                    best_local_cost = obj
                    inserted = False
                    
                    for m_try in range(M):
                        for b_try in range(len(machine_batches[m_try]) + 1):
                            if b_try < len(machine_batches[m_try]):
                                test_batch = machine_batches[m_try][b_try]
                                if sum(job_list[jj]['s'] for jj in test_batch) + job_list[j_move]['s'] > Cap:
                                    continue
                            test_batches = [lst[:] for lst in machine_batches]
                            if b_try < len(test_batches[m_try]):
                                test_batches[m_try][b_try].append(j_move)
                            else:
                                test_batches[m_try].insert(b_try, [j_move])
                            
                            test_obj, _, _ = build_full_solution(test_batches)
                            if test_obj < best_local_cost - 0.001:
                                best_local_cost = test_obj
                                best_local_m = m_try
                                best_local_b = b_try
                                inserted = True
                    
                    if inserted:
                        # Apply best move
                        machine_batches = [lst[:] for lst in machine_batches]  # reset
                        # Rebuild from best
                        if best_local_b < len(machine_batches[best_local_m]):
                            machine_batches[best_local_m][best_local_b].append(j_move)
                        else:
                            machine_batches[best_local_m].append([j_move])
                        obj = best_local_cost
                        improved = True
                        if logger:
                            logger.log(float(obj))
                    else:
                        # Revert
                        machine_batches = [lst[:] for lst in machine_batches]
                        if current_b < len(machine_batches[current_m]):
                            machine_batches[current_m][current_b] = original_batch
                        else:
                            machine_batches[current_m].append(original_batch)
                
                if not improved:
                    break
    
    # Final fallback if nothing found
    if best_batches is None:
        # One job per batch, round-robin machines
        machine_batches_fallback = [[] for _ in range(M)]
        for i, j_idx in enumerate(range(N)):
            m = i % M
            machine_batches_fallback[m].append([j_idx])
        best_obj, best_batches, best_job_assign_dict = build_full_solution(machine_batches_fallback)
    
    output = {
        "objective_value": float(best_obj),
        "batches": best_batches,
        "job_assignments": best_job_assign_dict
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(output, f, indent=2)

if __name__ == "__main__":
    main()
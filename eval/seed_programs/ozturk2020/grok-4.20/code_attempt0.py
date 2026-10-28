import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
import sys
import math

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
    
    # Sort jobs by release date for some heuristics
    sorted_jobs = sorted(range(N), key=lambda i: job_list[i]['r'])
    
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    start_time = time.time()
    time_limit = args.time_limit
    
    # Simple greedy heuristic: sort by release time, batch compatible jobs
    # Use multiple passes with different sorting keys to find good solutions quickly
    
    best_obj = float('inf')
    best_solution = None
    best_assign = None
    best_batches = None
    
    def evaluate_solution(batches_per_machine, job_to_batch_machine):
        # Compute completion times
        completion = [0] * N
        obj = 0.0
        machine_time = [0.0] * M
        
        for m in range(M):
            batches = batches_per_machine[m]
            current_time = 0.0
            for b_idx, batch in enumerate(batches):
                # start time = max(current_time, max release in batch)
                max_r = 0
                max_p = 0
                for j in batch:
                    max_r = max(max_r, job_list[j]['r'])
                    max_p = max(max_p, job_list[j]['p'])
                start = max(current_time, float(max_r))
                finish = start + max_p
                for j in batch:
                    completion[j] = finish
                    obj += finish
                current_time = finish
            machine_time[m] = current_time
        return obj, completion
    
    def build_batches_from_assignment(assignment):
        # assignment: list of (machine, batch_idx) for each job
        machine_batches = [{} for _ in range(M)]  # batch_idx -> list of jobs
        for j in range(N):
            m, bidx = assignment[j]
            if bidx not in machine_batches[m]:
                machine_batches[m][bidx] = []
            machine_batches[m][bidx].append(j)
        
        batches_list = []
        job_assign_dict = {}
        obj = 0.0
        batch_counter = 0
        
        for m in range(M):
            batch_dict = machine_batches[m]
            if not batch_dict:
                continue
            max_bidx = max(batch_dict.keys())
            for bidx in range(max_bidx + 1):
                if bidx not in batch_dict or not batch_dict[bidx]:
                    continue
                batch_jobs = batch_dict[bidx]
                max_p = max(job_list[j]['p'] for j in batch_jobs)
                max_r = max(job_list[j]['r'] for j in batch_jobs)
                # For output, we need to compute actual start time considering previous batches on machine
                # But to validate, we'll recompute sequentially
                batches_list.append({
                    'batch_id': bidx + 1,  # 1-based as per description
                    'machine': m + 1,      # assuming 1-based machines?
                    'start_time': 0,       # will update later
                    'processing_time': max_p,
                    'jobs': [job_list[j]['id'] for j in batch_jobs]
                })
                for j in batch_jobs:
                    job_assign_dict[str(job_list[j]['id'])] = {
                        'batch': bidx + 1,
                        'machine': m + 1,
                        'completion_time': 0  # update later
                    }
                batch_counter += 1
        # We need to compute actual start times and objective correctly
        # Recompute everything properly
        machine_current = [0.0] * M
        batch_map = {}  # (m, bidx) -> batch entry index in batches_list
        idx = 0
        for m_idx, entry in enumerate(batches_list):
            m = entry['machine'] - 1
            bidx = entry['batch_id'] - 1
            batch_map[(m, bidx)] = m_idx
        
        for m in range(M):
            b = 0
            while True:
                key = (m, b)
                if key not in batch_map:
                    break
                b_entry_idx = batch_map[key]
                b_entry = batches_list[b_entry_idx]
                jobs_in_b = [j for j in range(N) if job_list[j]['id'] in b_entry['jobs']]
                max_r = max(job_list[j]['r'] for j in jobs_in_b)
                start_t = max(machine_current[m], float(max_r))
                p_time = b_entry['processing_time']
                finish_t = start_t + p_time
                b_entry['start_time'] = start_t
                for jid in b_entry['jobs']:
                    job_assign_dict[str(jid)]['completion_time'] = finish_t
                    obj += finish_t
                machine_current[m] = finish_t
                b += 1
        return obj, batches_list, job_assign_dict
    
    strategies = [
        lambda i: job_list[i]['r'],           # by release
        lambda i: job_list[i]['p'],           # by processing time
        lambda i: job_list[i]['r'] + job_list[i]['p'],  # combined
        lambda i: job_list[i]['p'] * job_list[i]['s'],  # weighted
    ]
    
    for strategy_idx, key_func in enumerate(strategies):
        if time.time() - start_time > time_limit * 0.8:
            break
            
        # Sort jobs
        job_order = sorted(range(N), key=key_func)
        
        # Greedy batching
        machine_batches = [[] for _ in range(M)]
        machine_current_time = [0.0] * M
        job_assignment = [None] * N  # (machine, batch_index_on_machine)
        
        for j_idx in job_order:
            j_info = job_list[j_idx]
            best_m = -1
            best_b = -1
            best_start = float('inf')
            best_fit = False
            
            for m in range(M):
                # Try to add to existing batches on this machine
                for b_idx, batch in enumerate(machine_batches[m]):
                    # Check capacity
                    current_size = sum(job_list[jj]['s'] for jj in batch)
                    if current_size + j_info['s'] > Cap:
                        continue
                    # Check if p is compatible? No, p determines batch p = max
                    # Compute new max_p and start
                    max_p_new = max([job_list[jj]['p'] for jj in batch] + [j_info['p']])
                    max_r_new = max([job_list[jj]['r'] for jj in batch] + [j_info['r']])
                    start_new = max(machine_current_time[m] if b_idx == len(machine_batches[m])-1 else 
                                   (machine_current_time[m] if False else 0), float(max_r_new))  # rough
                    if start_new < best_start:
                        best_start = start_new
                        best_m = m
                        best_b = b_idx
                        best_fit = True
                
                # Or create new batch at end of machine
                new_max_r = j_info['r']
                new_start = max(machine_current_time[m], float(new_max_r))
                if new_start < best_start or best_m == -1:
                    best_start = new_start
                    best_m = m
                    best_b = len(machine_batches[m])
                    best_fit = False
            
            if best_m != -1:
                if best_fit and best_b < len(machine_batches[best_m]):
                    machine_batches[best_m][best_b].append(j_idx)
                else:
                    machine_batches[best_m].append([j_idx])
                # Update current time roughly for ordering
                # We'll recompute exactly later
                job_assignment[j_idx] = (best_m, best_b)
        
        # Evaluate
        obj, batches_out, assign_out = build_batches_from_assignment(job_assignment)
        
        if obj < best_obj:
            best_obj = obj
            best_solution = (batches_out, assign_out)
            best_batches = machine_batches
            if logger:
                logger.log(float(obj))
        
        # Try to improve by local search: reassign single jobs
        improved = True
        iter_count = 0
        while improved and iter_count < 5 and (time.time() - start_time < time_limit * 0.7):
            improved = False
            iter_count += 1
            for j in range(N):
                current_m, current_bidx = job_assignment[j]
                current_obj = best_obj
                best_alt_m = current_m
                best_alt_b = current_bidx
                best_alt_obj = current_obj
                
                for alt_m in range(M):
                    for alt_b in range(len(machine_batches[alt_m]) + 1):  # existing + new
                        # This is expensive, so we skip deep simulation and just try simple moves
                        pass  # Simplified: skip heavy local search for time
                
                if best_alt_obj < current_obj - 0.1:
                    # Would require full rebuild - skip for speed
                    pass
    
    # MIP formulation if time permits (for small instances)
    elapsed = time.time() - start_time
    if N <= 12 and elapsed < time_limit * 0.6:  # only for small instances
        try:
            model = gp.Model("BatchFlow")
            model.setParam('TimeLimit', max(1, time_limit - elapsed - 5))
            model.setParam('MIPGap', 0.05)
            model.setParam('Threads', 1)
            
            # Decision vars: batch assignment is complex
            # Use position-based modeling with limited batches per machine (upper bound K = N)
            K = N  # max batches per machine
            bigM = 1000000
            
            # x[j][m][k] = 1 if job j in batch k on machine m
            x = model.addVars(N, M, K, vtype=GRB.BINARY, name="x")
            
            # y[m][k] = 1 if batch k on m is used
            y = model.addVars(M, K, vtype=GRB.BINARY, name="y")
            
            # s[m][k] start time of batch k on m
            s = model.addVars(M, K, vtype=GRB.CONTINUOUS, name="s")
            
            # p_batch[m][k]
            pbatch = model.addVars(M, K, vtype=GRB.CONTINUOUS, name="p")
            
            # Completion for each job
            c = model.addVars(N, vtype=GRB.CONTINUOUS, name="c")
            
            # Objective
            model.setObjective(gp.quicksum(c[j] for j in range(N)), GRB.MINIMIZE)
            
            # Each job in exactly one batch
            for j in range(N):
                model.addConstr(gp.quicksum(x[j, m, k] for m in range(M) for k in range(K)) == 1)
            
            # Capacity and pbatch
            for m in range(M):
                for k in range(K):
                    model.addConstr(gp.quicksum(job_list[j]['s'] * x[j, m, k] for j in range(N)) <= Cap * y[m, k])
                    for j in range(N):
                        model.addConstr(pbatch[m, k] >= job_list[j]['p'] * x[j, m, k])
                        model.addConstr(s[m, k] >= job_list[j]['r'] * x[j, m, k])
            
            # Batch ordering on machine
            for m in range(M):
                for k in range(K-1):
                    model.addConstr(s[m, k+1] >= s[m, k] + pbatch[m, k] - bigM * (2 - y[m, k] - y[m, k+1]))
                    model.addConstr(y[m, k] >= y[m, k+1])  # no gaps in batch indices
            
            # Link completion
            for j in range(N):
                model.addConstr(c[j] == gp.quicksum((s[m, k] + pbatch[m, k]) * x[j, m, k] for m in range(M) for k in range(K)))
            
            # Symmetry breaking
            for m in range(M):
                model.addConstr(y[m, 0] == 1)  # at least first batch if used? not exactly but helps a bit
            
            model.optimize()
            
            if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
                if model.SolCount > 0:
                    sol_obj = model.ObjVal
                    if sol_obj < best_obj:
                        best_obj = sol_obj
                        # Extract solution
                        assignment = [None] * N
                        batch_list = []
                        job_assign_out = {}
                        
                        for m in range(M):
                            for k in range(K):
                                if y[m, k].X > 0.5:
                                    batch_jobs = []
                                    for j in range(N):
                                        if x[j, m, k].X > 0.5:
                                            batch_jobs.append(j)
                                    if batch_jobs:
                                        max_p_val = max(job_list[jj]['p'] for jj in batch_jobs)
                                        start_val = s[m, k].X
                                        finish_val = start_val + max_p_val
                                        batch_id = k + 1
                                        mach_id = m + 1
                                        batch_list.append({
                                            'batch_id': batch_id,
                                            'machine': mach_id,
                                            'start_time': start_val,
                                            'processing_time': max_p_val,
                                            'jobs': [job_list[jj]['id'] for jj in batch_jobs]
                                        })
                                        for jj in batch_jobs:
                                            jid_str = str(job_list[jj]['id'])
                                            job_assign_out[jid_str] = {
                                                'batch': batch_id,
                                                'machine': mach_id,
                                                'completion_time': finish_val
                                            }
                                            assignment[jj] = (m, k)
                        if logger:
                            logger.log(float(sol_obj))
                        best_solution = (batch_list, job_assign_out)
        except:
            pass  # if MIP fails, use heuristic solution
    
    # Final best solution
    if best_solution is None:
        # Fallback: each job its own batch, round robin machines
        batch_list = []
        job_assign_out = {}
        obj = 0.0
        machine_load = [0.0] * M
        for idx, j in enumerate(sorted_jobs):
            m = idx % M
            r = job_list[j]['r']
            p = job_list[j]['p']
            start = max(machine_load[m], float(r))
            finish = start + p
            machine_load[m] = finish
            obj += finish
            batch_id = 1
            mach_id = m + 1
            batch_list.append({
                'batch_id': batch_id,  
                'machine': mach_id,
                'start_time': start,
                'processing_time': p,
                'jobs': [job_list[j]['id']]
            })
            job_assign_out[str(job_list[j]['id'])] = {
                'batch': batch_id,
                'machine': mach_id,
                'completion_time': finish
            }
        best_obj = obj
        best_solution = (batch_list, job_assign_out)
    else:
        batch_list, job_assign_out = best_solution
    
    output = {
        "objective_value": float(best_obj),
        "batches": batch_list,
        "job_assignments": job_assign_out
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(output, f, indent=2)

if __name__ == "__main__":
    main()
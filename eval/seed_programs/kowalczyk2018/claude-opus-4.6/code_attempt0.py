import argparse
import json
import time
import heapq
from solution_logger import SolutionLogger


def solve(instance_path, solution_path, time_limit, log_path=None):
    start_time = time.time()
    
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    
    n = instance['n']
    m = instance['m']
    jobs = instance['jobs']
    
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    # Sort jobs by w/p ratio (WSPT rule) - descending
    # Handle p=0 case: if p=0 and w>0, ratio is infinity (schedule first)
    # if p=0 and w=0, ratio is 0
    def wspt_key(job):
        if job['p'] == 0:
            return (-float('inf'), -job['w'])
        return (-job['w'] / job['p'], job['p'])
    
    sorted_jobs = sorted(jobs, key=wspt_key)
    
    # Greedy WSPT assignment: assign jobs in WSPT order to machine that minimizes
    # the increase in weighted completion time
    # For each machine, track load and current weighted sum contribution
    
    def compute_objective(machine_assignments):
        """Compute total weighted completion time given machine assignments."""
        total = 0
        for machine_jobs in machine_assignments:
            # Jobs on each machine should be sorted by WSPT
            t = 0
            for job in machine_jobs:
                t += job['p']
                total += job['w'] * t
        return total
    
    def greedy_wspt_assignment(sorted_jobs, m):
        """Assign jobs in WSPT order to machine minimizing marginal cost increase."""
        # machine_load[i] = current total processing time on machine i
        # machine_weight[i] = current total weight of jobs on machine i
        machine_loads = [0] * m
        machine_assignments = [[] for _ in range(m)]
        
        for job in sorted_jobs:
            p_j = job['p']
            w_j = job['w']
            
            # If we add job to machine k (at end, since WSPT order is maintained):
            # marginal cost = w_j * (machine_loads[k] + p_j)
            # because the job completes at machine_loads[k] + p_j
            # (all previous jobs on the machine are already in WSPT order and scheduled before)
            
            best_machine = 0
            best_cost = float('inf')
            for k in range(m):
                cost = w_j * (machine_loads[k] + p_j)
                if cost < best_cost:
                    best_cost = cost
                    best_machine = k
            
            machine_assignments[best_machine].append(job)
            machine_loads[best_machine] += p_j
        
        return machine_assignments
    
    machine_assignments = greedy_wspt_assignment(sorted_jobs, m)
    best_obj = compute_objective(machine_assignments)
    best_assignments = [list(ma) for ma in machine_assignments]
    
    if logger:
        logger.log(best_obj)
    
    # Try to improve with local search: swap and move operations
    def try_improve(assignments, current_obj, deadline):
        """Local search: try moving a job from one machine to another, or swapping two jobs."""
        improved = True
        obj = current_obj
        
        while improved and time.time() < deadline:
            improved = False
            
            # Precompute machine loads and per-machine data
            num_machines = len(assignments)
            machine_loads = [sum(j['p'] for j in assignments[k]) for k in range(num_machines)]
            
            # Try move: move job i from machine a to machine b
            for a in range(num_machines):
                if time.time() >= deadline:
                    break
                if len(assignments[a]) == 0:
                    continue
                for idx_a in range(len(assignments[a])):
                    if time.time() >= deadline:
                        break
                    job_a = assignments[a][idx_a]
                    p_a = job_a['p']
                    w_a = job_a['w']
                    
                    # Cost of removing job_a from machine a:
                    # Current cost contribution of job_a: w_a * C_a where C_a is its completion time
                    # Also, removing it reduces completion time of all jobs after it by p_a
                    # So savings = w_a * C_a + p_a * (sum of weights of jobs after idx_a on machine a)
                    # But recomputing is simpler for correctness
                    
                    # Cost on machine a without job_a
                    cost_a_without = 0
                    t = 0
                    for idx, j in enumerate(assignments[a]):
                        if idx == idx_a:
                            continue
                        t += j['p']
                        cost_a_without += j['w'] * t
                    
                    # Current cost on machine a
                    cost_a_current = 0
                    t = 0
                    for j in assignments[a]:
                        t += j['p']
                        cost_a_current += j['w'] * t
                    
                    savings_from_removal = cost_a_current - cost_a_without
                    
                    for b in range(num_machines):
                        if b == a:
                            continue
                        if time.time() >= deadline:
                            break
                        
                        # Cost of adding job_a to machine b (at optimal position in WSPT order)
                        # Since all jobs are in WSPT order, we insert at correct position
                        # Actually, we need to find best insertion position
                        
                        # Current cost on machine b
                        cost_b_current = 0
                        t = 0
                        for j in assignments[b]:
                            t += j['p']
                            cost_b_current += j['w'] * t
                        
                        # Try inserting at each position in machine b
                        best_insert_pos = 0
                        best_insert_cost = float('inf')
                        
                        # Weights suffix sum for machine b
                        n_b = len(assignments[b])
                        suffix_w = [0] * (n_b + 1)
                        for i in range(n_b - 1, -1, -1):
                            suffix_w[i] = suffix_w[i + 1] + assignments[b][i]['w']
                        
                        # Prefix processing time
                        prefix_p = [0] * (n_b + 1)
                        for i in range(n_b):
                            prefix_p[i + 1] = prefix_p[i] + assignments[b][i]['p']
                        
                        # Inserting at position pos means job_a is processed after pos jobs
                        # Cost increase = w_a * (prefix_p[pos] + p_a) + p_a * suffix_w[pos]
                        for pos in range(n_b + 1):
                            cost_increase = w_a * (prefix_p[pos] + p_a) + p_a * suffix_w[pos]
                            if cost_increase < best_insert_cost:
                                best_insert_cost = cost_increase
                                best_insert_pos = pos
                        
                        net_change = best_insert_cost - savings_from_removal
                        
                        if net_change < -1e-9:
                            # Apply move
                            assignments[a].pop(idx_a)
                            assignments[b].insert(best_insert_pos, job_a)
                            obj += net_change
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                # Try swap: swap job from machine a with job from machine b
                for a in range(num_machines):
                    if time.time() >= deadline:
                        break
                    for b in range(a + 1, num_machines):
                        if time.time() >= deadline:
                            break
                        for idx_a in range(len(assignments[a])):
                            if time.time() >= deadline:
                                break
                            for idx_b in range(len(assignments[b])):
                                if time.time() >= deadline:
                                    break
                                
                                job_a = assignments[a][idx_a]
                                job_b = assignments[b][idx_b]
                                
                                # Quick filter: if same p and w, skip
                                if job_a['p'] == job_b['p'] and job_a['w'] == job_b['w']:
                                    continue
                                
                                # Compute change in cost for machine a: remove job_a, add job_b
                                # and for machine b: remove job_b, add job_a
                                
                                # Machine a: effect of replacing job_a with job_b at optimal position
                                # Machine b: effect of replacing job_b with job_a at optimal position
                                
                                # Current costs
                                cost_a = 0
                                t = 0
                                for j in assignments[a]:
                                    t += j['p']
                                    cost_a += j['w'] * t
                                
                                cost_b = 0
                                t = 0
                                for j in assignments[b]:
                                    t += j['p']
                                    cost_b += j['w'] * t
                                
                                # New machine a: remove job_a, insert job_b at best position
                                new_a = [j for idx, j in enumerate(assignments[a]) if idx != idx_a]
                                n_new_a = len(new_a)
                                suffix_w_a = [0] * (n_new_a + 1)
                                for i in range(n_new_a - 1, -1, -1):
                                    suffix_w_a[i] = suffix_w_a[i + 1] + new_a[i]['w']
                                prefix_p_a = [0] * (n_new_a + 1)
                                for i in range(n_new_a):
                                    prefix_p_a[i + 1] = prefix_p_a[i] + new_a[i]['p']
                                
                                best_pos_a = 0
                                best_cost_a = float('inf')
                                for pos in range(n_new_a + 1):
                                    # base cost without job_b
                                    base = 0
                                    t2 = 0
                                    for j in new_a:
                                        t2 += j['p']
                                        base += j['w'] * t2
                                    add_cost = job_b['w'] * (prefix_p_a[pos] + job_b['p']) + job_b['p'] * suffix_w_a[pos]
                                    total_cost_a = base + add_cost
                                    if total_cost_a < best_cost_a:
                                        best_cost_a = total_cost_a
                                        best_pos_a = pos
                                
                                # New machine b: remove job_b, insert job_a at best position
                                new_b = [j for idx, j in enumerate(assignments[b]) if idx != idx_b]
                                n_new_b = len(new_b)
                                suffix_w_b = [0] * (n_new_b + 1)
                                for i in range(n_new_b - 1, -1, -1):
                                    suffix_w_b[i] = suffix_w_b[i + 1] + new_b[i]['w']
                                prefix_p_b = [0] * (n_new_b + 1)
                                for i in range(n_new_b):
                                    prefix_p_b[i + 1] = prefix_p_b[i] + new_b[i]['p']
                                
                                best_pos_b = 0
                                best_cost_b = float('inf')
                                for pos in range(n_new_b + 1):
                                    base = 0
                                    t2 = 0
                                    for j in new_b:
                                        t2 += j['p']
                                        base += j['w'] * t2
                                    add_cost = job_a['w'] * (prefix_p_b[pos] + job_a['p']) + job_a['p'] * suffix_w_b[pos]
                                    total_cost_b = base + add_cost
                                    if total_cost_b < best_cost_b:
                                        best_cost_b = total_cost_b
                                        best_pos_b = pos
                                
                                new_total = best_cost_a + best_cost_b
                                old_total = cost_a + cost_b
                                
                                if new_total < old_total - 1e-9:
                                    # Apply swap
                                    assignments[a] = new_a
                                    assignments[a].insert(best_pos_a, job_b)
                                    assignments[b] = new_b
                                    assignments[b].insert(best_pos_b, job_a)
                                    obj = obj - old_total + new_total
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
        
        return assignments, obj
    
    deadline = start_time + time_limit - 0.5  # Reserve 0.5s for output
    
    # Run local search
    best_assignments, best_obj_ls = try_improve(best_assignments, best_obj, deadline)
    if best_obj_ls < best_obj:
        best_obj = best_obj_ls
        if logger:
            logger.log(best_obj)
    
    # Also try LPT-like assignment and other heuristics if time permits
    if time.time() < deadline:
        # Try assigning jobs sorted by processing time (LPT) to least loaded machine
        lpt_sorted = sorted(jobs, key=lambda j: -j['p'])
        machine_loads_lpt = [0] * m
        machine_assignments_lpt = [[] for _ in range(m)]
        
        for job in lpt_sorted:
            # Assign to least loaded machine
            min_load = min(machine_loads_lpt)
            best_k = machine_loads_lpt.index(min_load)
            machine_assignments_lpt[best_k].append(job)
            machine_loads_lpt[best_k] += job['p']
        
        # Sort each machine's jobs by WSPT
        for k in range(m):
            machine_assignments_lpt[k].sort(key=wspt_key)
        
        obj_lpt = compute_objective(machine_assignments_lpt)
        if obj_lpt < best_obj:
            best_obj = obj_lpt
            best_assignments = machine_assignments_lpt
            if logger:
                logger.log(best_obj)
        
        if time.time() < deadline:
            machine_assignments_lpt, obj_lpt_ls = try_improve(
                [list(ma) for ma in machine_assignments_lpt], obj_lpt, deadline
            )
            if obj_lpt_ls < best_obj:
                best_obj = obj_lpt_ls
                best_assignments = machine_assignments_lpt
                if logger:
                    logger.log(best_obj)
    
    # Build solution output
    solution = {
        "objective_value": best_obj,
        "machines": []
    }
    
    total_obj_check = 0
    for k in range(m):
        machine_jobs = best_assignments[k] if k < len(best_assignments) else []
        # Ensure WSPT order on each machine
        machine_jobs.sort(key=wspt_key)
        
        schedule_cost = 0
        t = 0
        job_list = []
        for job in machine_jobs:
            t += job['p']
            c_j = t
            schedule_cost += job['w'] * c_j
            job_list.append({
                "j_orig": job['j'],
                "p": job['p'],
                "w": job['w'],
                "C_j": c_j
            })
        
        total_obj_check += schedule_cost
        solution["machines"].append({
            "schedule_cost": schedule_cost,
            "jobs": job_list
        })
    
    # Add empty machines if needed
    while len(solution["machines"]) < m:
        solution["machines"].append({
            "schedule_cost": 0,
            "jobs": []
        })
    
    solution["objective_value"] = total_obj_check
    
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)


if __name__ == '__main__':
    main()
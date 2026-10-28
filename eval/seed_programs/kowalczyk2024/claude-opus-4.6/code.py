import argparse
import json
import time
import random
import math
from copy import deepcopy

def solve(instance_path, solution_path, time_limit, log_path):
    from solution_logger import SolutionLogger
    
    with open(instance_path, 'r') as f:
        data = json.load(f)
    
    n = data['n']
    m = data['m']
    jobs = data['jobs']
    
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    start_time = time.time()
    
    # Index jobs
    job_map = {}
    for j in jobs:
        job_map[j['id']] = j
    
    job_ids = [j['id'] for j in jobs]
    
    def compute_objective(machines):
        """Compute total weighted tardiness given machine assignments (list of lists of job_ids)."""
        total_wt = 0
        for machine_jobs in machines:
            t = 0
            for jid in machine_jobs:
                j = job_map[jid]
                t += j['p']
                tardiness = max(0, t - j['d'])
                total_wt += j['w'] * tardiness
        return total_wt
    
    def compute_machine_wt(machine_jobs):
        """Compute weighted tardiness for a single machine."""
        total_wt = 0
        t = 0
        for jid in machine_jobs:
            j = job_map[jid]
            t += j['p']
            tardiness = max(0, t - j['d'])
            total_wt += j['w'] * tardiness
        return total_wt
    
    def get_start_times(machines):
        """Get start times for all jobs."""
        schedule = {}
        for machine_jobs in machines:
            t = 0
            for jid in machine_jobs:
                schedule[str(jid)] = t
                t += job_map[jid]['p']
        return schedule
    
    # Greedy initial solution using WSPT-like rule
    # Sort jobs by weighted shortest processing time (w/p ratio descending)
    sorted_jobs = sorted(job_ids, key=lambda jid: job_map[jid]['w'] / job_map[jid]['p'], reverse=True)
    
    # Also try EDD (earliest due date) and other heuristics
    def greedy_assign(job_order, num_machines):
        machines = [[] for _ in range(num_machines)]
        machine_times = [0] * num_machines
        for jid in job_order:
            # Assign to machine with earliest completion
            best_m = min(range(num_machines), key=lambda mi: machine_times[mi])
            machines[best_m].append(jid)
            machine_times[best_m] += job_map[jid]['p']
        return machines
    
    def greedy_assign_min_wt(job_order, num_machines):
        """Assign each job to the machine that minimizes incremental weighted tardiness."""
        machines = [[] for _ in range(num_machines)]
        machine_times = [0] * num_machines
        machine_wts = [0] * num_machines
        for jid in job_order:
            j = job_map[jid]
            best_m = -1
            best_inc = float('inf')
            for mi in range(num_machines):
                ct = machine_times[mi] + j['p']
                inc = j['w'] * max(0, ct - j['d'])
                if inc < best_inc or (inc == best_inc and machine_times[mi] < machine_times[best_m if best_m >= 0 else mi]):
                    best_inc = inc
                    best_m = mi
            machines[best_m].append(jid)
            machine_times[best_m] += j['p']
        return machines
    
    # Try multiple initial heuristics
    best_machines = None
    best_obj = float('inf')
    
    # WSPT order
    for order_func in [
        lambda: sorted(job_ids, key=lambda jid: job_map[jid]['w'] / job_map[jid]['p'], reverse=True),
        lambda: sorted(job_ids, key=lambda jid: job_map[jid]['d']),
        lambda: sorted(job_ids, key=lambda jid: job_map[jid]['d'] - job_map[jid]['p']),
        lambda: sorted(job_ids, key=lambda jid: job_map[jid]['w'] * job_map[jid]['p'], reverse=True),
    ]:
        order = order_func()
        for assign_func in [greedy_assign, greedy_assign_min_wt]:
            machines = assign_func(order, m)
            # Also try optimizing each machine's sequence with WSPT
            for mi in range(m):
                machines[mi] = optimize_single_machine(machines[mi], job_map) if len(machines[mi]) > 0 else machines[mi]
            obj = compute_objective(machines)
            if obj < best_obj:
                best_obj = obj
                best_machines = deepcopy(machines)
    
    if logger:
        logger.log(best_obj)
    
    # Local search improvement
    def optimize_single_machine_dp(job_list):
        """For small job lists, try all permutations or use DP-like approach."""
        if len(job_list) <= 1:
            return job_list
        # Use adjacent swap optimization (bubble sort style)
        improved = True
        current = list(job_list)
        while improved:
            improved = False
            for i in range(len(current) - 1):
                # Try swapping i and i+1
                # Compute contribution before swap
                new = current[:]
                new[i], new[i+1] = new[i+1], new[i]
                if compute_machine_wt(new) < compute_machine_wt(current):
                    current = new
                    improved = True
        return current
    
    # Optimize each machine's job order
    for mi in range(m):
        best_machines[mi] = optimize_single_machine_dp(best_machines[mi])
    
    new_obj = compute_objective(best_machines)
    if new_obj < best_obj:
        best_obj = new_obj
        if logger:
            logger.log(best_obj)
    
    # More sophisticated local search
    def try_move(machines, obj):
        """Try moving a job from one machine to another."""
        best_delta = 0
        best_move = None
        
        for mi in range(m):
            if len(machines[mi]) == 0:
                continue
            old_wt_mi = compute_machine_wt(machines[mi])
            
            for ji_idx in range(len(machines[mi])):
                jid = machines[mi][ji_idx]
                j = job_map[jid]
                
                # Remove job from machine mi
                new_mi = machines[mi][:ji_idx] + machines[mi][ji_idx+1:]
                new_wt_mi = compute_machine_wt(new_mi)
                saved = old_wt_mi - new_wt_mi
                
                for mk in range(m):
                    if mk == mi:
                        continue
                    old_wt_mk = compute_machine_wt(machines[mk])
                    
                    # Try inserting at each position
                    best_pos = -1
                    best_wt_mk = float('inf')
                    for pos in range(len(machines[mk]) + 1):
                        new_mk = machines[mk][:pos] + [jid] + machines[mk][pos:]
                        wt_mk = compute_machine_wt(new_mk)
                        if wt_mk < best_wt_mk:
                            best_wt_mk = wt_mk
                            best_pos = pos
                    
                    cost = best_wt_mk - old_wt_mk
                    delta = -(saved - cost)  # negative means improvement
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_move = (mi, ji_idx, mk, best_pos)
        
        return best_move, best_delta
    
    def try_swap(machines, obj):
        """Try swapping jobs between two machines."""
        best_delta = 0
        best_swap = None
        
        for mi in range(m):
            for mk in range(mi + 1, m):
                if len(machines[mi]) == 0 or len(machines[mk]) == 0:
                    continue
                old_wt_mi = compute_machine_wt(machines[mi])
                old_wt_mk = compute_machine_wt(machines[mk])
                
                for ji_idx in range(len(machines[mi])):
                    for jk_idx in range(len(machines[mk])):
                        jid_i = machines[mi][ji_idx]
                        jid_k = machines[mk][jk_idx]
                        
                        new_mi = machines[mi][:ji_idx] + [jid_k] + machines[mi][ji_idx+1:]
                        new_mk = machines[mk][:jk_idx] + [jid_i] + machines[mk][jk_idx+1:]
                        
                        new_wt_mi = compute_machine_wt(new_mi)
                        new_wt_mk = compute_machine_wt(new_mk)
                        
                        delta = (new_wt_mi + new_wt_mk) - (old_wt_mi + old_wt_mk)
                        
                        if delta < best_delta:
                            best_delta = delta
                            best_swap = (mi, ji_idx, mk, jk_idx)
        
        return best_swap, best_delta
    
    # Iterative local search
    improved = True
    iteration = 0
    while improved and time.time() - start_time < time_limit * 0.8:
        improved = False
        iteration += 1
        
        # Try moves
        move, delta = try_move(best_machines, best_obj)
        if move is not None:
            mi, ji_idx, mk, pos = move
            jid = best_machines[mi][ji_idx]
            best_machines[mi] = best_machines[mi][:ji_idx] + best_machines[mi][ji_idx+1:]
            best_machines[mk] = best_machines[mk][:pos] + [jid] + best_machines[mk][pos:]
            # Re-optimize both machines
            best_machines[mi] = optimize_single_machine_dp(best_machines[mi])
            best_machines[mk] = optimize_single_machine_dp(best_machines[mk])
            new_obj = compute_objective(best_machines)
            if new_obj < best_obj:
                best_obj = new_obj
                improved = True
                if logger:
                    logger.log(best_obj)
        
        if time.time() - start_time >= time_limit * 0.8:
            break
        
        # Try swaps
        swap, delta = try_swap(best_machines, best_obj)
        if swap is not None:
            mi, ji_idx, mk, jk_idx = swap
            jid_i = best_machines[mi][ji_idx]
            jid_k = best_machines[mk][jk_idx]
            best_machines[mi][ji_idx] = jid_k
            best_machines[mk][jk_idx] = jid_i
            # Re-optimize both machines
            best_machines[mi] = optimize_single_machine_dp(best_machines[mi])
            best_machines[mk] = optimize_single_machine_dp(best_machines[mk])
            new_obj = compute_objective(best_machines)
            if new_obj < best_obj:
                best_obj = new_obj
                improved = True
                if logger:
                    logger.log(best_obj)
    
    # Simulated annealing phase
    current_machines = deepcopy(best_machines)
    current_obj = best_obj
    
    temp = max(1.0, best_obj * 0.05) if best_obj > 0 else 100.0
    cooling_rate = 0.9995
    
    sa_iterations = 0
    while time.time() - start_time < time_limit * 0.95:
        sa_iterations += 1
        
        # Random perturbation
        action = random.random()
        new_machines = deepcopy(current_machines)
        
        if action < 0.4 and n > 1:
            # Move a random job to a random machine at a random position
            non_empty = [i for i in range(m) if len(new_machines[i]) > 0]
            if non_empty:
                mi = random.choice(non_empty)
                ji_idx = random.randint(0, len(new_machines[mi]) - 1)
                jid = new_machines[mi][ji_idx]
                new_machines[mi] = new_machines[mi][:ji_idx] + new_machines[mi][ji_idx+1:]
                
                if m > 1:
                    mk = random.choice([i for i in range(m) if i != mi or len(new_machines[i]) > 0])
                    if mk == mi and len(new_machines[mi]) == 0:
                        mk = random.randint(0, m - 1)
                else:
                    mk = 0
                pos = random.randint(0, len(new_machines[mk]))
                new_machines[mk] = new_machines[mk][:pos] + [jid] + new_machines[mk][pos:]
        elif action < 0.7:
            # Swap two jobs on same or different machines
            if n >= 2:
                all_positions = []
                for mi in range(m):
                    for ji in range(len(new_machines[mi])):
                        all_positions.append((mi, ji))
                if len(all_positions) >= 2:
                    pos1, pos2 = random.sample(all_positions, 2)
                    mi1, ji1 = pos1
                    mi2, ji2 = pos2
                    new_machines[mi1][ji1], new_machines[mi2][ji2] = new_machines[mi2][ji2], new_machines[mi1][ji1]
        else:
            # Reverse a segment on a random machine
            non_empty = [i for i in range(m) if len(new_machines[i]) >= 2]
            if non_empty:
                mi = random.choice(non_empty)
                l = len(new_machines[mi])
                i1 = random.randint(0, l - 2)
                i2 = random.randint(i1 + 1, l - 1)
                new_machines[mi][i1:i2+1] = reversed(new_machines[mi][i1:i2+1])
        
        new_obj = compute_objective(new_machines)
        delta = new_obj - current_obj
        
        if delta < 0 or (temp > 0 and random.random() < math.exp(-delta / max(temp, 1e-10))):
            current_machines = new_machines
            current_obj = new_obj
            
            if current_obj < best_obj:
                best_obj = current_obj
                best_machines = deepcopy(current_machines)
                if logger:
                    logger.log(best_obj)
        
        temp *= cooling_rate
        
        # Periodically do local optimization on machines
        if sa_iterations % 500 == 0:
            for mi in range(m):
                if len(current_machines[mi]) >= 2:
                    old_wt = compute_machine_wt(current_machines[mi])
                    optimized = optimize_single_machine_dp(current_machines[mi])
                    new_wt = compute_machine_wt(optimized)
                    if new_wt < old_wt:
                        current_machines[mi] = optimized
                        current_obj += (new_wt - old_wt)
                        if current_obj < best_obj:
                            best_obj = current_obj
                            best_machines = deepcopy(current_machines)
                            if logger:
                                logger.log(best_obj)
    
    # Final optimization pass on each machine
    for mi in range(m):
        best_machines[mi] = optimize_single_machine_dp(best_machines[mi])
    final_obj = compute_objective(best_machines)
    if final_obj < best_obj:
        best_obj = final_obj
        if logger:
            logger.log(best_obj)
    
    # Generate output
    schedule = get_start_times(best_machines)
    
    solution = {
        "objective_value": best_obj,
        "schedule": schedule
    }
    
    with open(solution_path, 'w') as f:
        json.dump(solution, f)


def optimize_single_machine(job_list, job_map):
    """Optimize single machine using adjacent swap."""
    if len(job_list) <= 1:
        return job_list
    current = list(job_list)
    improved = True
    while improved:
        improved = False
        for i in range(len(current) - 1):
            new = current[:]
            new[i], new[i+1] = new[i+1], new[i]
            if _compute_machine_wt(new, job_map) < _compute_machine_wt(current, job_map):
                current = new
                improved = True
    return current


def _compute_machine_wt(machine_jobs, job_map):
    total_wt = 0
    t = 0
    for jid in machine_jobs:
        j = job_map[jid]
        t += j['p']
        tardiness = max(0, t - j['d'])
        total_wt += j['w'] * tardiness
    return total_wt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)
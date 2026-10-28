import argparse
import json
import time
import itertools
import random
from copy import deepcopy
from solution_logger import SolutionLogger


def solve(instance, time_limit, logger=None):
    start_time = time.time()
    
    num_jobs = instance["num_jobs"]
    num_machines = instance["num_machines"]
    cap = instance["batch_capacity"]
    jobs = instance["jobs"]
    
    # Sort jobs by release date, then by processing time descending
    jobs_sorted = sorted(jobs, key=lambda j: (j["release_date"], -j["processing_time"]))
    
    def compute_schedule(machine_batches):
        """Given machine_batches: list of machines, each a list of batches (list of job indices),
        compute start times and objective value."""
        total_obj = 0
        batch_info = []
        job_completion = {}
        
        for m_idx, batches in enumerate(machine_batches):
            current_time = 0
            for b_idx, batch_jobs in enumerate(batches):
                if not batch_jobs:
                    continue
                # Get job details
                max_release = max(jobs[j]["release_date"] for j in batch_jobs)
                max_proc = max(jobs[j]["processing_time"] for j in batch_jobs)
                total_size = sum(jobs[j]["size"] for j in batch_jobs)
                
                start = max(current_time, max_release)
                end = start + max_proc
                current_time = end
                
                for j in batch_jobs:
                    job_completion[j] = end
                    total_obj += end
                
                batch_info.append({
                    "machine": m_idx,
                    "batch_idx": b_idx,
                    "start_time": start,
                    "processing_time": max_proc,
                    "jobs": batch_jobs
                })
        
        return total_obj, batch_info, job_completion
    
    def greedy_solution(jobs_order, num_machines, cap):
        """Build a greedy solution: assign jobs in given order to machines using BFD-like approach."""
        # Each machine has a list of batches
        # Each batch is a list of job indices + metadata
        machine_batches = [[] for _ in range(num_machines)]
        machine_end_times = [0] * num_machines
        
        for j_idx in jobs_order:
            job = jobs[j_idx]
            r = job["release_date"]
            p = job["processing_time"]
            s = job["size"]
            
            best_cost = float('inf')
            best_machine = -1
            best_batch = -1
            best_is_new = True
            
            # Try to fit into existing batches
            for m in range(num_machines):
                for b_idx, batch in enumerate(machine_batches[m]):
                    batch_jobs = batch["jobs"]
                    batch_size = batch["total_size"]
                    batch_max_proc = batch["max_proc"]
                    batch_max_release = batch["max_release"]
                    
                    if batch_size + s > cap:
                        continue
                    
                    # Estimate the cost of adding this job to this batch
                    new_max_proc = max(batch_max_proc, p)
                    new_max_release = max(batch_max_release, r)
                    
                    # We need to recompute the schedule for this machine
                    # Quick estimate: added cost = completion time of this job
                    # Plus any delay to subsequent batches
                    delta_proc = new_max_proc - batch_max_proc
                    delta_release = new_max_release - batch_max_release
                    
                    # Number of jobs in subsequent batches
                    subsequent_jobs = sum(len(machine_batches[m][bi]["jobs"]) for bi in range(b_idx + 1, len(machine_batches[m])))
                    
                    # Rough cost: completion of this job + delay * subsequent jobs
                    # We need actual computation for accuracy
                    # Let's do a quick simulation
                    cost = estimate_add_cost(machine_batches[m], b_idx, j_idx, s, p, r)
                    
                    if cost < best_cost:
                        best_cost = cost
                        best_machine = m
                        best_batch = b_idx
                        best_is_new = False
            
            # Try creating a new batch on each machine
            for m in range(num_machines):
                cost = estimate_new_batch_cost(machine_batches[m], j_idx, s, p, r)
                if cost < best_cost:
                    best_cost = cost
                    best_machine = m
                    best_batch = len(machine_batches[m])
                    best_is_new = True
            
            if best_is_new:
                machine_batches[best_machine].append({
                    "jobs": [j_idx],
                    "total_size": s,
                    "max_proc": p,
                    "max_release": r
                })
            else:
                batch = machine_batches[best_machine][best_batch]
                batch["jobs"].append(j_idx)
                batch["total_size"] += s
                batch["max_proc"] = max(batch["max_proc"], p)
                batch["max_release"] = max(batch["max_release"], r)
        
        # Convert to simple format
        result = []
        for m in range(num_machines):
            m_batches = []
            for batch in machine_batches[m]:
                m_batches.append(batch["jobs"])
            result.append(m_batches)
        
        return result
    
    def estimate_add_cost(machine_schedule, batch_idx, job_idx, size, proc, release):
        """Estimate the cost change of adding a job to an existing batch."""
        # Compute current machine cost
        current_cost = compute_machine_cost(machine_schedule)
        
        # Create modified schedule
        new_schedule = deepcopy(machine_schedule)
        new_schedule[batch_idx]["jobs"].append(job_idx)
        new_schedule[batch_idx]["total_size"] += size
        new_schedule[batch_idx]["max_proc"] = max(new_schedule[batch_idx]["max_proc"], proc)
        new_schedule[batch_idx]["max_release"] = max(new_schedule[batch_idx]["max_release"], release)
        
        new_cost = compute_machine_cost(new_schedule)
        return new_cost - current_cost
    
    def estimate_new_batch_cost(machine_schedule, job_idx, size, proc, release):
        """Estimate the cost of adding a new batch at the end of a machine."""
        current_cost = compute_machine_cost(machine_schedule)
        
        new_batch = {
            "jobs": [job_idx],
            "total_size": size,
            "max_proc": proc,
            "max_release": release
        }
        new_schedule = machine_schedule + [new_batch]
        new_cost = compute_machine_cost(new_schedule)
        return new_cost - current_cost
    
    def compute_machine_cost(machine_schedule):
        """Compute total completion time for all jobs on a machine."""
        current_time = 0
        total = 0
        for batch in machine_schedule:
            start = max(current_time, batch["max_release"])
            end = start + batch["max_proc"]
            current_time = end
            total += end * len(batch["jobs"])
        return total
    
    def compute_machine_cost_simple(batches_jobs):
        """Compute total completion time for a machine given list of batches (each batch = list of job indices)."""
        current_time = 0
        total = 0
        for batch_jobs in batches_jobs:
            if not batch_jobs:
                continue
            max_release = max(jobs[j]["release_date"] for j in batch_jobs)
            max_proc = max(jobs[j]["processing_time"] for j in batch_jobs)
            start = max(current_time, max_release)
            end = start + max_proc
            current_time = end
            total += end * len(batch_jobs)
        return total
    
    def full_objective(machine_batches):
        """Compute total objective for all machines."""
        total = 0
        for m_batches in machine_batches:
            total += compute_machine_cost_simple(m_batches)
        return total
    
    # Better greedy: sort by release date, then assign
    def greedy_bf(order):
        """Best-fit decreasing style greedy."""
        machine_batches = [[] for _ in range(num_machines)]  # list of (jobs_list, total_size, max_proc, max_release)
        machine_batch_meta = [[] for _ in range(num_machines)]
        
        for j_idx in order:
            job = jobs[j_idx]
            r, p, s = job["release_date"], job["processing_time"], job["size"]
            
            best_cost_delta = float('inf')
            best_m = -1
            best_b = -1
            best_new = True
            
            for m in range(num_machines):
                # Try each existing batch
                for b_idx in range(len(machine_batches[m])):
                    bm = machine_batch_meta[m][b_idx]
                    if bm[1] + s > cap:
                        continue
                    
                    # Quick: compute full machine cost before and after
                    old_cost = compute_machine_cost_meta(machine_batch_meta[m])
                    
                    # Temporarily modify
                    old_meta = bm[:]
                    machine_batch_meta[m][b_idx] = [
                        len(machine_batches[m][b_idx]) + 1,
                        bm[1] + s,
                        max(bm[2], p),
                        max(bm[3], r)
                    ]
                    new_cost = compute_machine_cost_meta(machine_batch_meta[m])
                    machine_batch_meta[m][b_idx] = old_meta
                    
                    delta = new_cost - old_cost
                    if delta < best_cost_delta:
                        best_cost_delta = delta
                        best_m = m
                        best_b = b_idx
                        best_new = False
                
                # Try new batch at end
                old_cost = compute_machine_cost_meta(machine_batch_meta[m])
                machine_batch_meta[m].append([1, s, p, r])
                new_cost = compute_machine_cost_meta(machine_batch_meta[m])
                machine_batch_meta[m].pop()
                
                delta = new_cost - old_cost
                if delta < best_cost_delta:
                    best_cost_delta = delta
                    best_m = m
                    best_b = len(machine_batches[m])
                    best_new = True
            
            if best_new:
                machine_batches[best_m].append([j_idx])
                machine_batch_meta[best_m].append([1, s, p, r])
            else:
                machine_batches[best_m][best_b].append(j_idx)
                bm = machine_batch_meta[best_m][best_b]
                bm[0] += 1
                bm[1] += s
                bm[2] = max(bm[2], p)
                bm[3] = max(bm[3], r)
        
        return machine_batches
    
    def compute_machine_cost_meta(meta_list):
        """Compute cost from metadata: list of [num_jobs, total_size, max_proc, max_release]."""
        current_time = 0
        total = 0
        for nj, ts, mp, mr in meta_list:
            start = max(current_time, mr)
            end = start + mp
            current_time = end
            total += end * nj
        return total
    
    # Try multiple orderings
    best_obj = float('inf')
    best_solution = None
    
    # Ordering 1: by release date, then processing time desc
    order1 = sorted(range(num_jobs), key=lambda j: (jobs[j]["release_date"], -jobs[j]["processing_time"]))
    sol1 = greedy_bf(order1)
    obj1 = full_objective(sol1)
    if obj1 < best_obj:
        best_obj = obj1
        best_solution = deepcopy(sol1)
        if logger:
            logger.log(best_obj)
    
    if time.time() - start_time < time_limit * 0.1:
        # Ordering 2: by release date, then processing time asc
        order2 = sorted(range(num_jobs), key=lambda j: (jobs[j]["release_date"], jobs[j]["processing_time"]))
        sol2 = greedy_bf(order2)
        obj2 = full_objective(sol2)
        if obj2 < best_obj:
            best_obj = obj2
            best_solution = deepcopy(sol2)
            if logger:
                logger.log(best_obj)
    
    if time.time() - start_time < time_limit * 0.1:
        # Ordering 3: by processing time desc
        order3 = sorted(range(num_jobs), key=lambda j: (-jobs[j]["processing_time"], jobs[j]["release_date"]))
        sol3 = greedy_bf(order3)
        obj3 = full_objective(sol3)
        if obj3 < best_obj:
            best_obj = obj3
            best_solution = deepcopy(sol3)
            if logger:
                logger.log(best_obj)
    
    if time.time() - start_time < time_limit * 0.1:
        # Ordering 4: by size desc
        order4 = sorted(range(num_jobs), key=lambda j: (-jobs[j]["size"], jobs[j]["release_date"]))
        sol4 = greedy_bf(order4)
        obj4 = full_objective(sol4)
        if obj4 < best_obj:
            best_obj = obj4
            best_solution = deepcopy(sol4)
            if logger:
                logger.log(best_obj)
    
    # Local search improvement
    def try_improve(solution):
        nonlocal best_obj, best_solution
        
        improved = True
        iteration = 0
        while improved and time.time() - start_time < time_limit * 0.95:
            improved = False
            iteration += 1
            
            # Move: take a job from one batch and move it to another batch (same or different machine)
            for m in range(num_machines):
                if time.time() - start_time >= time_limit * 0.95:
                    break
                for b_idx in range(len(solution[m])):
                    if time.time() - start_time >= time_limit * 0.95:
                        break
                    batch = solution[m][b_idx]
                    if not batch:
                        continue
                    
                    for j_pos in range(len(batch)):
                        if time.time() - start_time >= time_limit * 0.95:
                            break
                        j_idx = batch[j_pos]
                        job = jobs[j_idx]
                        
                        # Current cost of source machine
                        old_source_cost = compute_machine_cost_simple(solution[m])
                        
                        # Remove job from current batch
                        batch.pop(j_pos)
                        
                        if not batch:
                            # Remove empty batch
                            solution[m].pop(b_idx)
                        
                        new_source_cost = compute_machine_cost_simple(solution[m])
                        source_saving = old_source_cost - new_source_cost
                        
                        best_move_cost = float('inf')
                        best_move = None  # (machine, batch_idx, is_new)
                        
                        for m2 in range(num_machines):
                            if time.time() - start_time >= time_limit * 0.95:
                                break
                            
                            old_target_cost = compute_machine_cost_simple(solution[m2]) if m2 != m else new_source_cost
                            
                            # Try each existing batch on m2
                            for b2_idx in range(len(solution[m2])):
                                b2 = solution[m2][b2_idx]
                                if not b2:
                                    continue
                                total_size = sum(jobs[jj]["size"] for jj in b2)
                                if total_size + job["size"] > cap:
                                    continue
                                
                                # Add job
                                b2.append(j_idx)
                                new_target_cost = compute_machine_cost_simple(solution[m2])
                                b2.pop()
                                
                                if m2 == m:
                                    delta = new_target_cost - new_source_cost
                                else:
                                    delta = (new_target_cost - old_target_cost) - source_saving
                                
                                net_cost = -source_saving + (new_target_cost - old_target_cost) if m2 != m else (new_target_cost - new_source_cost) - source_saving
                                # Actually: total change = (new_source_cost - old_source_cost) + (new_target_cost - old_target_cost)
                                # But if m2 == m, the source IS the target
                                if m2 == m:
                                    # We already removed the job, new_source_cost is computed
                                    # Now add to batch b2_idx
                                    b2.append(j_idx)
                                    combined_cost = compute_machine_cost_simple(solution[m])
                                    b2.pop()
                                    net_cost = combined_cost - old_source_cost
                                else:
                                    net_cost = (new_source_cost - old_source_cost) + (new_target_cost - old_target_cost)
                                
                                if net_cost < best_move_cost:
                                    best_move_cost = net_cost
                                    best_move = (m2, b2_idx, False)
                            
                            # Try new batch on m2
                            # Find best insertion position
                            for ins_pos in range(len(solution[m2]) + 1):
                                solution[m2].insert(ins_pos, [j_idx])
                                new_target_cost = compute_machine_cost_simple(solution[m2])
                                solution[m2].pop(ins_pos)
                                
                                if m2 == m:
                                    net_cost = new_target_cost - old_source_cost
                                else:
                                    net_cost = (new_source_cost - old_source_cost) + (new_target_cost - old_target_cost)
                                
                                if net_cost < best_move_cost:
                                    best_move_cost = net_cost
                                    best_move = (m2, ins_pos, True)
                        
                        if best_move is not None and best_move_cost < -1e-9:
                            # Apply the move
                            m2, b2_idx, is_new = best_move
                            if is_new:
                                solution[m2].insert(b2_idx, [j_idx])
                            else:
                                solution[m2][b2_idx].append(j_idx)
                            
                            new_obj = full_objective(solution)
                            if new_obj < best_obj:
                                best_obj = new_obj
                                best_solution = deepcopy(solution)
                                if logger:
                                    logger.log(best_obj)
                            improved = True
                            break  # restart
                        else:
                            # Undo removal
                            if not batch:
                                solution[m].insert(b_idx, [j_idx])
                            else:
                                batch.insert(j_pos, j_idx)
                    
                    if improved:
                        break
                if improved:
                    break
        
        return solution
    
    # Optimize batch ordering on each machine
    def optimize_batch_order(solution):
        nonlocal best_obj, best_solution
        changed = True
        while changed:
            changed = False
            for m in range(num_machines):
                if time.time() - start_time >= time_limit * 0.95:
                    return solution
                batches = solution[m]
                if len(batches) <= 1:
                    continue
                
                # Try all pairwise swaps
                current_cost = compute_machine_cost_simple(batches)
                best_perm_cost = current_cost
                best_swap = None
                
                for i in range(len(batches)):
                    for j in range(i + 1, len(batches)):
                        batches[i], batches[j] = batches[j], batches[i]
                        new_cost = compute_machine_cost_simple(batches)
                        if new_cost < best_perm_cost:
                            best_perm_cost = new_cost
                            best_swap = (i, j)
                        batches[i], batches[j] = batches[j], batches[i]
                
                if best_swap is not None:
                    i, j = best_swap
                    batches[i], batches[j] = batches[j], batches[i]
                    changed = True
            
            if changed:
                new_obj = full_objective(solution)
                if new_obj < best_obj:
                    best_obj = new_obj
                    best_solution = deepcopy(solution)
                    if logger:
                        logger.log(best_obj)
        
        return solution
    
    # Optimize batch ordering first
    if time.time() - start_time < time_limit * 0.3:
        best_solution = optimize_batch_order(deepcopy(best_solution))
    
    # Local search
    if time.time() - start_time < time_limit * 0.9:
        try_improve(deepcopy(best_solution))
    
    # After local search, try batch order optimization again
    if time.time() - start_time < time_limit * 0.95:
        best_solution = optimize_batch_order(deepcopy(best_solution))
    
    # Try merge batches
    def try_merge_batches(solution):
        nonlocal best_obj, best_solution
        changed = True
        while changed and time.time() - start_time < time_limit * 0.95:
            changed = False
            for m in range(num_machines):
                if time.time() - start_time >= time_limit * 0.95:
                    break
                batches = solution[m]
                for i in range(len(batches)):
                    if time.time() - start_time >= time_limit * 0.95:
                        break
                    for j in range(i + 1, len(batches)):
                        if time.time() - start_time >= time_limit * 0.95:
                            break
                        # Check if we can merge batch i and j
                        size_i = sum(jobs[jj]["size"] for jj in batches[i])
                        size_j = sum(jobs[jj]["size"] for jj in batches[j])
                        if size_i + size_j <= cap:
                            old_cost = compute_machine_cost_simple(batches)
                            merged = batches[i] + batches[j]
                            new_batches = [batches[k] for k in range(len(batches)) if k != i and k != j]
                            # Try inserting merged batch at each position
                            best_pos = 0
                            best_pos_cost = float('inf')
                            for pos in range(len(new_batches) + 1):
                                new_batches.insert(pos, merged)
                                c = compute_machine_cost_simple(new_batches)
                                if c < best_pos_cost:
                                    best_pos_cost = c
                                    best_pos = pos
                                new_batches.pop(pos)
                            
                            if best_pos_cost < old_cost - 1e-9:
                                new_batches.insert(best_pos, merged)
                                solution[m] = new_batches
                                new_obj = full_objective(solution)
                                if new_obj < best_obj:
                                    best_obj = new_obj
                                    best_solution = deepcopy(solution)
                                    if logger:
                                        logger.log(best_obj)
                                changed = True
                                break
                    if changed:
                        break
        return solution
    
    if time.time() - start_time < time_limit * 0.95:
        best_solution = try_merge_batches(deepcopy(best_solution))
    
    # Final batch order optimization
    if time.time() - start_time < time_limit * 0.95:
        best_solution = optimize_batch_order(deepcopy(best_solution))
    
    # Second round of local search
    if time.time() - start_time < time_limit * 0.95:
        try_improve(deepcopy(best_solution))
    
    # Split large batches - try splitting a batch into two if it reduces cost
    def try_split_batches(solution):
        nonlocal best_obj, best_solution
        changed = True
        while changed and time.time() - start_time < time_limit * 0.95:
            changed = False
            for m in range(num_machines):
                if time.time() - start_time >= time_limit * 0.95:
                    break
                batches = solution[m]
                for b_idx in range(len(batches)):
                    if time.time() - start_time >= time_limit * 0.95:
                        break
                    batch = batches[b_idx]
                    if len(batch) <= 1:
                        continue
                    
                    old_cost = compute_machine_cost_simple(batches)
                    
                    # Try removing each job from the batch into its own new batch
                    best_split_cost = old_cost
                    best_split = None
                    
                    for j_pos in range(len(batch)):
                        j_idx = batch[j_pos]
                        remaining = batch[:j_pos] + batch[j_pos+1:]
                        
                        new_batches = batches[:b_idx] + [remaining] + batches[b_idx+1:]
                        
                        # Try inserting single job batch at each position
                        for pos in range(len(new_batches) + 1):
                            new_batches.insert(pos, [j_idx])
                            c = compute_machine_cost_simple(new_batches)
                            if c < best_split_cost - 1e-9:
                                best_split_cost = c
                                best_split = (j_pos, pos, b_idx)
                            new_batches.pop(pos)
                    
                    if best_split is not None:
                        j_pos, ins_pos, bi = best_split
                        j_idx = batch[j_pos]
                        remaining = batch[:j_pos] + batch[j_pos+1:]
                        new_batches = batches[:bi] + [remaining] + batches[bi+1:]
                        new_batches.insert(ins_pos, [j_idx])
                        solution[m] = new_batches
                        
                        new_obj = full_objective(solution)
                        if new_obj < best_obj:
                            best_obj = new_obj
                            best_solution = deepcopy(solution)
                            if logger:
                                logger.log(best_obj)
                        changed = True
                        break
        return solution
    
    if time.time() - start_time < time_limit * 0.95:
        best_solution = try_split_batches(deepcopy(best_solution))
    
    if time.time() - start_time < time_limit * 0.95:
        best_solution = optimize_batch_order(deepcopy(best_solution))
    
    # Another round of moves
    if time.time() - start_time < time_limit * 0.90:
        try_improve(deepcopy(best_solution))
    
    if time.time() - start_time < time_limit * 0.95:
        best_solution = optimize_batch_order(deepcopy(best_solution))
    
    if time.time() - start_time < time_limit * 0.95:
        best_solution = try_merge_batches(deepcopy(best_solution))
    
    if time.time() - start_time < time_limit * 0.95:
        best_solution = optimize_batch_order(deepcopy(best_solution))
    
    # Build output
    best_obj = full_objective(best_solution)
    
    output_batches = []
    job_assignments = {}
    
    for m_idx, batches in enumerate(best_solution):
        current_time = 0
        for b_idx, batch_jobs in enumerate(batches):
            if not batch_jobs:
                continue
            max_release = max(jobs[j]["release_date"] for j in batch_jobs)
            max_proc = max(jobs[j]["processing_time"] for j in batch_jobs)
            
            start = max(current_time, max_release)
            end = start + max_proc
            current_time = end
            
            output_batches.append({
                "batch_id": b_idx + 1,
                "machine": m_idx + 1,
                "start_time": start,
                "processing_time": max_proc,
                "jobs": [jobs[j]["job_id"] for j in batch_jobs]
            })
            
            for j in batch_jobs:
                job_assignments[str(jobs[j]["job_id"])] = {
                    "batch": b_idx + 1,
                    "machine": m_idx + 1,
                    "completion_time": end
                }
    
    return {
        "objective_value": best_obj,
        "batches": output_batches,
        "job_assignments": job_assignments
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()
    
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    solution = solve(instance, args.time_limit, logger)
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()
import argparse
import json
import math
import heapq
import time
from solution_logger import SolutionLogger


def solve_parallel_scheduling(instance_path, solution_path, time_limit, log_path):
    # Initialize the logger
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    start_time = time.time()
    
    # Read the instance
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    
    n = instance['n']
    m = instance['m']
    jobs = [(i, j['p'], j['w']) for i, j in enumerate(instance['jobs'], 1)]  # (index, processing_time, weight)
    
    # Precompute weight-to-processing-time ratios
    ratios = []
    for idx, p, w in jobs:
        ratio = w / p if p > 0 else float('inf')
        ratios.append((ratio, idx, p, w))
    
    # Sort jobs by weight-to-processing-time ratio in non-increasing order
    ratios.sort(reverse=True)
    
    # Initialize machines' cumulative processing times
    machine_times = [0] * m
    machine_jobs = [[] for _ in range(m)]
    
    # Assign jobs to machines in round-robin fashion after sorting
    # Then refine with local searches
    for _, idx, p, w in ratios:
        # Find machine with minimum current cumulative load
        min_load_idx = machine_times.index(min(machine_times))
        machine_jobs[min_load_idx].append({'j_orig': idx, 'p': p, 'w': w})
        machine_times[min_load_idx] += p

    def calculate_weighted_completion(jobs_list):
        """Calculate weighted completion time for a single machine."""
        c_time = 0
        total_wct = 0
        jobs_with_completion = []
        for job in jobs_list:
            c_time += job['p']
            completion_time = c_time
            weighted_ct = job['w'] * completion_time
            total_wct += weighted_ct
            jobs_with_completion.append({
                'j_orig': job['j_orig'],
                'p': job['p'],
                'w': job['w'],
                'C_j': completion_time
            })
        return total_wct, jobs_with_completion

    def evaluate_current_solution():
        """Evaluate the current assignment."""
        total_obj = 0
        schedules = []
        for i in range(m):
            schedule_cost, schedule = calculate_weighted_completion(machine_jobs[i])
            schedules.append({
                'schedule_cost': schedule_cost,
                'jobs': schedule
            })
            total_obj += schedule_cost
        return total_obj, schedules

    def improve_single_machine_local(job_list, machine_time):
        """Attempt improving job order on a single machine using WSPT principle."""
        # Within a machine, we can reorder jobs to minimize local WCt
        # The optimal order within a machine is still WSPT (Weighted Shortest Processing Time first)
        sorted_jobs = []
        for job in job_list:
            r = job['w'] / job['p'] if job['p'] > 0 else float('inf')
            sorted_jobs.append((r, job))

        # We want highest weight/processing first, so reverse sort by ratio
        sorted_jobs.sort(key=lambda x: x[0], reverse=True)
        improved_job_list = [job for ratio, job in sorted_jobs]
        
        return improved_job_list

    # Improve each machine's internal ordering based on WSPT
    for i in range(m):
        old_schedule_cost, _ = calculate_weighted_completion(machine_jobs[i])
        improved_jobs = improve_single_machine_local(machine_jobs[i], machine_times[i])
        new_schedule_cost, _ = calculate_weighted_completion(improved_jobs)
        
        if new_schedule_cost < old_schedule_cost:
            machine_jobs[i] = improved_jobs
            total_obj_temp, _ = evaluate_current_solution()

    # Perform further pairwise swaps between machines to find improvements
    def perform_local_search(max_iter=500):
        improved_any = True
        iter_count = 0
        while improved_any and iter_count < max_iter:
            improved_any = False
            current_obj, _ = evaluate_current_solution()
            
            for i in range(m):
                for j in range(i+1, m):
                    # Try swapping each job from machine i with each job from machine j
                    for pos_i in range(len(machine_jobs[i])):
                        for pos_j in range(len(machine_jobs[j])):
                            # Perform swap
                            job_i = machine_jobs[i][pos_i]
                            job_j = machine_jobs[j][pos_j]
                            
                            # Create temporary new assignments
                            temp_mach_i_jobs = machine_jobs[i][:pos_i] + [job_j] + machine_jobs[i][pos_i+1:]
                            temp_mach_j_jobs = machine_jobs[j][:pos_j] + [job_i] + machine_jobs[j][pos_j+1:]
                            
                            # Reorder to respect WSPT on both affected machines
                            temp_mach_i_jobs = improve_single_machine_local(temp_mach_i_jobs, machine_times[i] - job_i['p'] + job_j['p'])
                            temp_mach_j_jobs = improve_single_machine_local(temp_mach_j_jobs, machine_times[j] - job_j['p'] + job_i['p'])
                            
                            # Calculate potential new objective
                            old_i_cost, _ = calculate_weighted_completion(machine_jobs[i])
                            old_j_cost, _ = calculate_weighted_completion(machine_jobs[j])
                            new_i_cost, _ = calculate_weighted_completion(temp_mach_i_jobs)
                            new_j_cost, _ = calculate_weighted_completion(temp_mach_j_jobs)
                            
                            old_total = old_i_cost + old_j_cost
                            new_total = new_i_cost + new_j_cost
                            
                            if new_total < old_total:
                                machine_jobs[i] = temp_mach_i_jobs
                                machine_jobs[j] = temp_mach_j_jobs
                                
                                # Update machine times estimate (not critical since we recalculate obj)
                                updated_total, schedules_result = evaluate_current_solution()
                                
                                if logger and updated_total < current_obj:
                                    logger.log(updated_total)
                                
                                improved_any = True
                            
                            # Break early if time limit reached
                            if time.time() - start_time >= time_limit * 0.95:  # Exit if time is almost up
                                return
            iter_count += 1
            # End of iteration
    
    # Run improvement heuristic
    perform_local_search()
    
    # Final evaluation
    final_obj, final_schedules = evaluate_current_solution()
    
    # Prepare result
    result = {
        "objective_value": float(final_obj),
        "machines": final_schedules
    }
    
    # Write solution
    with open(solution_path, 'w') as f:
        json.dump(result, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Parallel Scheduling Optimizer")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to the instance JSON file")
    parser.add_argument("--solution_path", type=str, required=True, help="Path where the solution JSON file will be written")
    parser.add_argument("--time_limit", type=int, required=True, help="Maximum runtime in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Path to a JSONL file for logging intermediate solutions")
    
    args = parser.parse_args()
    
    solve_parallel_scheduling(
        instance_path=args.instance_path,
        solution_path=args.solution_path,
        time_limit=args.time_limit,
        log_path=args.log_path
    )


if __name__ == "__main__":
    main()
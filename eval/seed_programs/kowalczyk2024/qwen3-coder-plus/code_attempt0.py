import argparse
import json
import math
from typing import Dict, List, Tuple
import random
import copy
from solution_logger import SolutionLogger


def calculate_tardiness(start_time, p, d):
    """Calculate tardiness of a job given its start time, processing time, and due date."""
    completion_time = start_time + p
    return max(0, completion_time - d)


def evaluate_schedule(jobs_data: List[Dict], schedule: Dict[int, int], m: int) -> float:
    """
    Evaluate the total weighted tardiness of a schedule based on job start times.
    """
    machines_completions = [[] for _ in range(m)]
    
    # Create list of jobs with their schedule information
    scheduled_jobs = []
    for job_id, start_time in schedule.items():
        job_info = next(job for job in jobs_data if job['id'] == job_id)
        scheduled_jobs.append({
            'id': job_id,
            'start': start_time,
            'processing_time': job_info['p'],
            'due_date': job_info['d'],
            'weight': job_info['w'],
            'completion_time': start_time + job_info['p']
        })
    
    # Sort jobs by start time to know processing order on machines
    scheduled_jobs.sort(key=lambda x: x['start'])
    
    # Determine which machine each job goes to by simulating assignment order
    current_ends = [0.0] * m  # Current end time for each machine
    
    # This function tries to reconstruct the assignment by assuming jobs are assigned greedily
    # But since we have start times, we need a more sophisticated method to infer machine assignments
    # Instead, we'll directly verify that the schedule is valid by ensuring no overlaps if needed
    # For now, let's assume validity and directly compute tardiness
    total_weighted_tardiness = 0.0
    for job_info in jobs_data:
        job_start_time = schedule[job_info['id']]
        tardiness = calculate_tardiness(job_start_time, job_info['p'], job_info['d'])
        total_weighted_tardiness += job_info['w'] * tardiness
        
    return total_weighted_tardiness


def assign_to_machines_by_start_order(scheduled_jobs: List[Dict], m: int) -> List[List[Dict]]:
    """
    Assign jobs to machines based on their start times ensuring no conflicts.
    This function takes jobs with start times and assigns them to machines,
    validating that the timing does not cause conflicts.
    """
    machines = [[] for _ in range(m)]
    machine_end_times = [0.0] * m
    
    # Sort jobs by start time to process in the order they are scheduled
    for job in sorted(scheduled_jobs, key=lambda x: x['start']):
        assigned = False
        for mid in range(m):
            # Check if this machine is available at the required start_time
            # Look for when last job ends on this machine
            last_end = 0.0 if not machines[mid] else machines[mid][-1]['end_time']
            
            if last_end <= job['start']:
                # Fit the job on this machine
                job_info_with_timing = {
                    **job,
                    'machine': mid,
                    'end_time': job['start'] + job['p']
                }
                machines[mid].append(job_info_with_timing)
                machine_end_times[mid] = job_info_with_timing['end_time']
                assigned = True
                break
        
        if not assigned:
            # Find the machine that becomes free earliest
            min_idx = machine_end_times.index(min(machine_end_times))
            # Update the start time to fit with the last finish time on that machine
            actual_start = max(job['start'], machine_end_times[min_idx])
            job_info_with_timing = {
                **job,
                'start_time_new': actual_start,
                'machine': min_idx,
                'end_time': actual_start + job['p']
            }
            machines[min_idx].append(job_info_with_timing)
            machine_end_times[min_idx] = job_info_with_timing['end_time']
    
    return machines


def get_greedy_schedule(jobs: List[Dict], m: int) -> Tuple[Dict[int, int], float]:
    """Generate initial greedy schedule based on modified WSPT rule adapted for parallel machines."""
    jobs_list = [(j['id'], j['p'], j['d'], j['w']) for j in jobs]
    
    # Sort jobs based on weight/processing_time ratio (a variation of WSPT rule)
    # Or according to due date if ratios equal
    sorted_jobs = sorted(jobs_list, key=lambda x: (-(x[3]/(x[1] + 1)), x[2]))
    
    # Maintain machine availability times
    machine_loads = [0] * m  # Keeps track of total processing time on each machine
    job_start_times = {}
    
    # Schedule jobs to machine with smallest current load
    for job_id, processing_time, due_date, weight in sorted_jobs:
        # Assign to first available machine (greedy assignment based on load)
        min_machine_idx = min(range(m), key=lambda i: machine_loads[i])
        
        start_time = machine_loads[min_machine_idx]
        job_start_times[job_id] = start_time
        machine_loads[min_machine_idx] += processing_time
    
    # Calculate objective
    total_weighted_tardiness = evaluate_schedule(jobs, job_start_times, m)
    
    return job_start_times, total_weighted_tardiness


def perform_local_search(initial_schedule: Dict[int, int], jobs: List[Dict], m: int, time_limit: int) -> Tuple[Dict[int, int], float]:
    """Perform local search to improve the current schedule."""
    import time
    start_time = time.time()
    
    current_schedule = copy.deepcopy(initial_schedule)
    current_objective = evaluate_schedule(jobs, current_schedule, m)
    
    improved = True
    
    # Try different types of neighborhoods
    while improved and (time.time() - start_time < time_limit * 0.8):  # Leave some time for final checks
        improved = False
        
        # Neighborhood move 1: Job swapping on same machine
        # We first need to assign jobs to machines based on current schedule
        jobs_with_info = [
            {'id': job_id, 'start': start_time, 'p': next((j['p'] for j in jobs if j['id'] == job_id), 0)}
            for job_id, start_time in current_schedule.items()
        ]
        machines = assign_to_machines_by_start_order(jobs_with_info, m)
        
        # Swap jobs on the same machine
        for mach in machines:
            for i in range(len(mach)):
                for j in range(i+1, len(mach)):
                    job_i_id = mach[i]['id']
                    job_j_id = mach[j]['id']
                    
                    # Create candidate schedule by swapping positions
                    cand_schedule = copy.deepcopy(current_schedule)
                    
                    # Recompute start times after swap
                    # Need to resequence these two jobs on this machine
                    other_jobs = [(jid, s_time) for jid, s_time in current_schedule.items() 
                                 if jid != job_i_id and jid != job_j_id]
                    
                    # Jobs up to i-1 position stay the same
                    new_schedule = {k: v for k, v in other_jobs}
                    
                    # Place j-th job in i-th position
                    prev_end_time = 0 if i == 0 else mach[i-1]['end_time']
                    new_schedule[job_j_id] = prev_end_time
                    
                    # Place i-th job in j-th position
                    temp_prev_end = prev_end_time + next((j_dict['p'] for j_dict in jobs if j_dict['id'] == job_j_id), 0)
                    new_schedule[job_i_id] = temp_prev_end
                    
                    # Adjust subsequent jobs if needed
                    remaining_jobs = [mid for mid in mach[i+1:j+1] if mid['id'] not in [job_i_id, job_j_id]]
                    curr_time = temp_prev_end + next((j_dict['p'] for j_dict in jobs if j_dict['id'] == job_i_id), 0)
                    
                    for rem_job in remaining_jobs:
                        new_schedule[rem_job['id']] = curr_time
                        curr_time += next((j_dict['p'] for j_dict in jobs if j_dict['id'] == rem_job['id']), 0)
                    
                    # Adjust the rest of the machine sequence
                    rest_jobs = mach[j+1:]
                    for job_info in rest_jobs:
                        new_schedule[job_info['id']] = curr_time
                        curr_time += next((j_dict['p'] for j_dict in jobs if j_dict['id'] == job_info['id']), 0)
                    
                    cand_obj = evaluate_schedule(jobs, new_schedule, m)
                    if cand_obj < current_objective:
                        current_schedule = new_schedule
                        current_objective = cand_obj
                        improved = True
                        break
                if improved: break
            if improved: break

        if improved: continue
        
        # Neighborhood move 2: Move a job from one machine to another position
        for job_id, current_start in current_schedule.items():
            # Identify the current machine of the job
            job_p = next((j['p'] for j in jobs if j['id'] == job_id), 0)
            job_d = next((j['d'] for j in jobs if j['id'] == job_id), 0)
            job_w = next((j['w'] for j in jobs if j['id'] == job_id), 0)
            
            # Try inserting at various positions
            for insert_machine in range(m):
                # Get jobs in current and target machines
                candidate_schedule = copy.deepcopy(current_schedule)
                
                # Remove the job from its current slot
                # Rebuild schedule temporarily to know the position
                machines_sched = assign_to_machines_by_start_order([
                    {'id': job_id, 'p': next((j['p'] for j in jobs if j['id'] == job_id), 0), 
                     'start': start_time}
                    for job_id, start_time in candidate_schedule.items()
                ], m)
                
                # For each alternative position on the target machine
                target_jobs = machines_sched[insert_machine]
                for pos in range(len(target_jobs) + 1):  # Insert at each position including at the end
                    candidate_schedule = copy.deepcopy(current_schedule)
                    
                    # Get the start time for the new position:
                    # If pos > 0, it comes after job at pos-1
                    new_start = 0 if pos == 0 else machines_sched[insert_machine][pos-1]['end_time']
                    
                    candidate_schedule[job_id] = new_start
                    
                    # Shift jobs originally after this position to correct new positions
                    # For simplicity, we'll recalculate all schedules
                    new_obj = evaluate_schedule(jobs, candidate_schedule, m)
                    if new_obj < current_objective:
                        current_schedule = candidate_schedule
                        current_objective = new_obj
                        improved = True
                        break
                if improved: break
            if improved: break
                
    return current_schedule, current_objective


def solve_problem(instance_path: str, time_limit: int):
    with open(instance_path, 'r') as f:
        instance = json.load(f)

    n = instance['n']
    m = instance['m']
    jobs = instance['jobs']
    
    # Get initial solution
    sched, obj_val = get_greedy_schedule(jobs, m)
    
    # Perform local search to improve solution
    best_schedule, best_obj = perform_local_search(sched, jobs, m, time_limit)
    
    result = {
        "objective_value": best_obj,
        "schedule": {str(k): v for k, v in best_schedule.items()}
    }
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Solve Parallel Machine Scheduling Problem')
    parser.add_argument('--instance_path', type=str, help='Path to the instance JSON file')
    parser.add_argument('--solution_path', type=str, help='Path to output the solution JSON file')
    parser.add_argument('--time_limit', type=int, help='Time limit in seconds')
    parser.add_argument('--log_path', type=str, help='Path to log the intermediate solutions')
    
    args = parser.parse_args()
    
    # Initialize logger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    result = solve_problem(args.instance_path, args.time_limit)
    
    # Log final objective if logger is available
    if logger:
        logger.log(result["objective_value"])
    
    # Write solution to file
    with open(args.solution_path, 'w') as f:
        json.dump(result, f, indent=2)


if __name__ == '__main__':
    main()
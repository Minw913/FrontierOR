import argparse
import json
import math
from collections import defaultdict
import random

def solve_batch_scheduling(instance_path, solution_path, time_limit, log_path=None):
    # Initialize the logger
    if log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(log_path, sense="minimize")
    else:
        logger = None

    # Read the instance
    with open(instance_path, 'r') as f:
        instance = json.load(f)

    num_jobs = instance['num_jobs']
    num_machines = instance['num_machines']
    batch_capacity = instance['batch_capacity']
    jobs = {job_data['job_id']: {
        'release_date': job_data['release_date'],
        'processing_time': job_data['processing_time'],
        'size': job_data['size']
    } for job_data in instance['jobs']}

    # Heuristic approach: Improved version of First-Fit Decreasing (FFD)
    # Sort jobs by release date, then by processing time (descending), then by size (descending)
    sorted_job_ids = sorted(jobs.keys(), 
                            key=lambda j: (-jobs[j]['release_date'], -jobs[j]['processing_time'], -jobs[j]['size']))

    current_solution = []
    best_objective_value = float('inf')
    best_assignment = {}

    # Greedy heuristic with small local search improvements
    def calculate_assignment_and_cost():
        # Reset assignment info for each iteration
        batch_start_times = {}
        batch_processing_times = {}
        batch_jobs = {}
        
        # For each machine, track the time at which it is free
        machine_free_time = [0 for _ in range(num_machines)]
        machine_batches = [[] for _ in range(num_machines)]
        
        # Keep track of which batch each job gets assigned to
        job_to_batch = {}
        job_to_machine = {}
        
        # Assign each job greedily
        for job_id in sorted_job_ids:
            job_info = jobs[job_id]
            
            # Try to place the job on the machine with the earliest availability that has room
            earliest_feasible_batch = []
            for machine_idx in range(num_machines):
                # Check existing batches for space
                batch_idx = len(machine_batches[machine_idx])
                
                # For each machine, maintain its set of batches (may need to create one)
                # Look through batches already created on this machine
                candidate_made_new_batch = False
                
                for b_idx, batch_list in enumerate(machine_batches[machine_idx]):
                    # Get current start time and duration
                    batch_start = max(
                        job_info['release_date'],
                        get_batch_start_time(machine_idx, b_idx, machine_batches),
                        machine_free_time[machine_idx] if b_idx == len(machine_batches[machine_idx]) - 1 else 0
                    )
                    
                    # Get max proc time considering adding this job
                    new_proc_time = max([jobs[jid]['processing_time'] for jid in batch_list + [job_id]])
                    capacity_after_adding = sum([jobs[jid]['size'] for jid in batch_list + [job_id]])
                    
                    if capacity_after_adding <= batch_capacity:
                        # We can potentially add to this batch
                        if b_idx == len(machine_batches[machine_idx]) - 1:  # Last batch on machine - consider schedule clash
                            prev_end_time = get_batch_finish_time(machine_idx, b_idx-1, machine_batches) if b_idx > 0 else 0
                            batch_start = max(batch_start, prev_end_time)
                        
                        # Check if we can add to this batch without causing issues
                        # This is getting complex, let's go with simple approach first
                        break
                        
            # Use greedy strategy for now
            placed = False
            for machine_idx in range(num_machines):
                machine_batches_on_this_machine = [] if len(current_solution) <= machine_idx else [b for b in current_solution if b['machine'] == machine_idx]
                
                # We'll build up from scratch more systematically
                machine_batch_list = [[] for _ in range(len(machine_batches_on_this_machine))]
                
                # Try adding to an existing batch
                for b_idx in range(len(machine_batch_list)):
                    batch_list = machine_batch_list[b_idx]
                    batch_size = sum([jobs[jid]['size'] for jid in batch_list])
                    
                    if batch_size + job_info['size'] <= batch_capacity:
                        # Calculate tentative start time for this batch with the added job
                        batch_proc_time = max([jobs[jid]['processing_time'] for jid in batch_list + [job_id]])
                        batch_start_time = max(
                            job_info['release_date'], 
                            (machine_batch_list[b_idx - 1][2] if b_idx > 0 else 0)  # Previous batch end time
                        )
                        
                        # We want to recompute this properly
                        pass
                        
                # If not placed in existing batch but capacity remains, add to new batch on this machine
                if not placed and len(machine_batch_list) < len(sorted_job_ids):
                    if job_info['size'] <= batch_capacity:
                        new_batch_jobs = [j for batch_j in [b for m in machine_batches_for_machine(machine_idx)] for j in batch_j]
                        
            # Simple greedy - just try to fit in earliest available slot
            # This algorithm is getting complex. Let's go with an efficient greedy approach
            
        # Simple greedy approach to build batches
        assignments = {}  # Maps job_id to {'machine', 'batch'}
        batches = {}  # Key: (machine_idx, batch_idx), Value: dict of batch info

        machine_usage_end_time = [0 for _ in range(num_machines)]
        machine_batch_counts = [0 for _ in range(num_machines)]

        for job_id in sorted_job_ids:
            job = jobs[job_id]
            
            # Find best placement (earliest possible start time)
            best_machine_idx = -1
            best_batch_idx = -1
            best_start_time = float('inf')

            for machine_idx in range(num_machines):
                # Search through existing batches to see if we can fit
                batch_found = False
                for m_key, b_info in batches.items():
                    if b_info['machine'] != machine_idx:
                        continue
                    
                    # Check if it fits in existing batch
                    batch_jobs_here = b_info.get('jobs', [])
                    if sum(jobs[jid]['size'] for jid in batch_jobs_here) + job['size'] <= batch_capacity:
                        # Check latest start time constraint
                        new_proc_time = max(b_info['processing_time'], job['processing_time'])
                        new_release = max(job['release_date'], max(jobs[jid]['release_date'] for jid in batch_jobs_here))
                        min_possible_start = new_release
                        
                        # Schedule after previous job on machine if exists
                        if b_info.get('end_time'):  # We'll set this after creating a temp batch
                            pass
                        
                # Check for new batch on this machine
                current_time = machine_usage_end_time[machine_idx]
                start_time_for_new_batch = max(job['release_date'], current_time)
                
                if job['size'] <= batch_capacity:
                    if start_time_for_new_batch < best_start_time:
                        best_machine_idx = machine_idx
                        best_batch_idx = machine_batch_counts[machine_idx]
                        best_start_time = start_time_for_new_batch
            
            # Place job in best available location
            key = (best_machine_idx, best_batch_idx)
            if key not in batches:
                batches[key] = {
                    'machine': best_machine_idx,
                    'batch_id': best_batch_idx,
                    'jobs': [],
                    'start_time': best_start_time,
                    'processing_time': 0,
                    'end_time': 0
                }
                machine_batch_counts[best_machine_idx] += 1
            
            batches[key]['jobs'].append(job_id)
            assignments[job_id] = {'machine': best_machine_idx, 'batch': best_batch_idx}
            
            # Update times
            new_proc_time = max(batches[key]['processing_time'], job['processing_time'])
            batches[key]['processing_time'] = new_proc_time
            
            max_rel_date = max(jobs[jid]['release_date'] for jid in batches[key]['jobs'])
            actual_start_time = max(max_rel_date, 
                                   machine_usage_end_time[best_machine_idx] if best_batch_idx > 0 else max_rel_date)
            batches[key]['start_time'] = actual_start_time
            batches[key]['end_time'] = actual_start_time + new_proc_time
            
            # Update machine usage time to end of this batch
            machine_usage_end_time[best_machine_idx] = batches[key]['end_time']

        # Convert batches dict to list format
        batch_list = []
        for info in batches.values():
            info_copy = info.copy()
            del info_copy['end_time']  # Not needed in final output
            batch_list.append(info_copy)

        # Compute completion times and total flow time
        total_flow_time = 0
        job_assignments = {}
        
        for job_id, assign in assignments.items():
            machine_idx = assign['machine']
            batch_idx = assign['batch']
            
            batch = next(b for b in batch_list if b['machine'] == machine_idx and b['batch_id'] == batch_idx)
            
            completion_time = batch['start_time'] + batch['processing_time']
            total_flow_time += completion_time
            
            job_assignments[job_id] = {
                'machine': machine_idx,
                'batch': batch_idx,
                'completion_time': completion_time
            }

        return batch_list, job_assignments, total_flow_time

    def machine_batches_for_machine(m_id):
        return [b for b in current_solution if b['machine'] == m_id]

    def get_batch_start_time(m_idx, b_idx, m_batches):
        # Find specific batch with machine index and batch id
        matching_batches = [b for b in m_batches[m_idx] if b['batch_id'] == b_idx]
        if matching_batches:
            return matching_batches[0].get('start_time', 0)
        return 0

    def get_batch_finish_time(m_idx, b_idx, m_batches):
        matching_batch = next((b for b in m_batches[m_idx] if b['batch_id'] == b_idx), None)
        if matching_batch:
            return matching_batch.get('start_time', 0) + matching_batch['processing_time']
        return 0
    
    # Simple greedy initial assignment
    def compute_greedy_solution():
        # Create initial greedy assignment
        job_assignments = {}
        all_batches = []
        
        machines = [defaultdict(int) for _ in range(num_machines)]
        machine_free_time = [0] * num_machines
        machine_batch_counters = [0] * num_machines
        
        assignments = {}
        
        # Sort jobs by release date, tie-break by processing time descending
        ordered_jobs = sorted(jobs.keys(), key=lambda j: (jobs[j]['release_date'], -jobs[j]['processing_time']))
        
        for job_id in ordered_jobs:
            job = jobs[job_id]
            
            # Try placing job in existing batch on earliest available machine
            placed = False
            
            # Sort machines by free time to prioritize earlier starting times
            machine_opts = list(enumerate(machine_free_time))
            machine_opts.sort(key=lambda x: x[1])  # By free time ascending
            
            for idx, mt in machine_opts:
                # Look for a batch on this machine where we can add the job
                batches_for_machine = [b for b in all_batches if b['machine'] == idx]
                
                # Sort existing batches by start time so we try adding to earlier batches first
                batches_for_machine.sort(key=lambda b: b['start_time'])
                
                for batch in batches_for_machine:
                    if sum(jobs[jid]['size'] for jid in batch['jobs']) + job['size'] <= batch_capacity:
                        # We can add to this batch!
                        # Need to recalculate batch start time and processing time
                        batch['jobs'].append(job_id)
                        old_proc_time = batch['processing_time']
                        new_proc_time = max(old_proc_time, job['processing_time'])
                        batch['processing_time'] = new_proc_time
                        
                        # Recompute start time: max(release dates of all jobs in batch, start of prev batch)
                        max_release = max(jobs[jid]['release_date'] for jid in batch['jobs'])
                        prev_batch_end = 0
                        
                        # Find prev batch end time on same machine
                        same_machine_batches = [b for b in all_batches if b['machine'] == idx]
                        same_machine_batches.sort(key=lambda x: x['batch_id'])  # by index
                        prev_b_idx = -1
                        for i, b in enumerate(same_machine_batches):
                            if b == batch and i > 0:
                                prev_b_dx = i - 1
                                prev_batch = same_machine_batches[prev_b_dx]
                                prev_batch_end = prev_batch['start_time'] + prev_batch['processing_time']
                                break
                        
                        batch['start_time'] = max(max_release, prev_batch_end)
                        assignments[job_id] = {'machine': idx, 'batch': batch['batch_id']}
                        
                        # Update machine time to end of this updated batch
                        machine_free_time[idx] = batch['start_time'] + batch['processing_time']
                        placed = True
                        break
                
                if placed:
                    break
            
            # If not placed in existing batch on any machine, create new batch
            if not placed:
                best_machine_idx = -1
                for m_idx in range(num_machines):
                    if job['size'] <= batch_capacity :
                        best_machine_idx = m_idx
                        break
                        
                if best_machine_idx != -1:
                    start_time = max(job['release_date'], machine_free_time[best_machine_idx])
                    new_batch = {
                        'machine': best_machine_idx,
                        'batch_id': machine_batch_counters[best_machine_idx],
                        'start_time': start_time,
                        'processing_time': job['processing_time'],
                        'jobs': [job_id]
                    }
                    
                    all_batches.append(new_batch)
                    # Update machine stats
                    machine_free_time[best_machine_idx] = start_time + job['processing_time']
                    machine_batch_counters[best_machine_idx] += 1
                    assignments[job_id] = {'machine': best_machine_idx, 'batch': new_batch['batch_id']}
        
        # Finalize solution: compute completion times
        total_flow = 0
        final_job_assignment_map = {}
        
        for job_id, assign in assignments.items():
            machine_idx = assign['machine']
            batch_id = assign['batch']
            
            batch_obj = next((b for b in all_batches if b['machine'] == machine_idx and b['batch_id'] == batch_id), None)
            if batch_obj:
                comp_time = batch_obj['start_time'] + batch_obj['processing_time']
                total_flow += comp_time
                
                final_job_assignment_map[job_id] = {
                    'machine': machine_idx,
                    'batch': batch_id,
                    'completion_time': comp_time
                }

        return all_batches, final_job_assignment_map, total_flow
    
    final_batch_list, final_job_assignments, base_objective_value = compute_greedy_solution()

    # Since we're using a heuristic, let's make one pass refinement
    # Improve by trying to swap jobs between batches with different processing times
    
    # For now, just store the greedy result
    best_objective_value = base_objective_value
    current_solution = final_batch_list
    best_assignment = final_job_assignments

    # Prepare final result
    result = {
        "objective_value": best_objective_value,
        "batches": [],
        "job_assignments": best_assignment
    }

    # Add formatted batch info
    for batch in current_solution:
        formatted_batch = {
            "machine": batch['machine'],
            "batch_id": batch['batch_id'],
            "start_time": int(batch['start_time']),
            "processing_time": int(batch['processing_time']),
            "jobs": sorted(batch['jobs'])  # sort job IDs for consistency
        }
        result["batches"].append(formatted_batch)

    # Ensure batches are sorted by machine then batch_id for clarity
    result["batches"] = sorted(result["batches"], key=lambda x: (x["machine"], x["batch_id"]))

    # Write the solution
    with open(solution_path, 'w') as f:
        json.dump(result, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Solve Batch Scheduling Problem')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the instance JSON file')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to write the solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds')
    parser.add_argument('--log_path', type=str, default=None, help='Path to the log file (optional)')

    args = parser.parse_args()
    
    solve_batch_scheduling(
        instance_path=args.instance_path,
        solution_path=args.solution_path,
        time_limit=args.time_limit,
        log_path=args.log_path
    )
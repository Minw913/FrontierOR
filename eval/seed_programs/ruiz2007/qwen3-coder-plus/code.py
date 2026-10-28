import argparse
import json
import math
from itertools import permutations
import time
from solution_logger import SolutionLogger
import sys

def solve_flow_shop_scheduling(instance_data, time_limit):
    n_jobs = instance_data["n_jobs"]
    n_machines = instance_data["n_machines"]
    no_idle_machines = set(instance_data["no_idle_machines"])  # 1-indexed
    no_idle_flags = instance_data["no_idle_flags"]
    processing_times = instance_data["processing_times"]

    # Since gurobipy is not found, let's implement a heuristic approach instead.
    # For such problems, simple heuristics like NEH algorithm work well for flowshop scheduling.
    # We adapt the NEH heuristic with consideration for the no-idle constraint
    start_time = time.time()
    
    # Calculate total processing time per job as a priority factor
    total_processing_times = [sum(processing_times[i]) for i in range(n_jobs)]
    job_indices = list(range(n_jobs))
    
    # Sort jobs by descending total processing time (NEH-style)
    sorted_jobs = sorted(job_indices, key=lambda x: total_processing_times[x], reverse=True)
    
    # Initial sequence is just the first job
    current_sequence = [sorted_jobs[0]]
    
    for idx in range(1, len(sorted_jobs)):
        candidate_job = sorted_jobs[idx]
        best_makespan = float('inf')
        best_position = 0
        
        # Try inserting the candidate job at each possible position
        for pos in range(len(current_sequence) + 1):
            test_sequence = current_sequence[:pos] + [candidate_job] + current_sequence[pos:]
            makespan_val = calculate_makespan_with_no_idle(test_sequence, processing_times, no_idle_flags, n_jobs, n_machines)
            
            if makespan_val < best_makespan:
                best_makespan = makespan_val
                best_position = pos
                
        # Insert in the best position found
        current_sequence.insert(best_position, candidate_job)
        
        if time.time() - start_time >= time_limit * 0.95:  # Leave some buffer for return
            break
    
    # Calculate completion times for the final best sequence
    final_makespan, completion_times = calculate_detailed_completion_times(current_sequence, processing_times, no_idle_flags, n_jobs, n_machines)
    
    return {
        "objective_value": final_makespan,
        "sequence": current_sequence,
        "completion_times": completion_times
    }

def calculate_makespan_with_no_idle(sequence, processing_times, no_idle_flags, n_jobs, n_machines):
    """
    Helper function to calculate makespan of a given sequence considering no-idle constraints
    """
    if not sequence:
        return 0
    
    # c[j][k] = completion time of job number j on machine k
    # But actually, c[pos][k] = completion time of job at position pos on machine k
    c = [[0 for _ in range(n_machines)] for _ in range(len(sequence))]
    
    for pos, job_id in enumerate(sequence):
        for machine in range(n_machines):
            proc_time = processing_times[job_id][machine]
            
            job_comp_time_on_machine = 0
            if pos == 0:
                # First job for this machine
                if machine == 0:
                    job_comp_time_on_machine = proc_time
                else:
                    job_comp_time_on_machine = c[0][machine-1] + proc_time
            else:
                # Not first in sequence
                prev_job_comp_on_machine = c[pos-1][machine]
                if machine == 0:
                    job_comp_time_on_machine = max(prev_job_comp_on_machine, 0) + proc_time
                else:
                    job_available_time = c[pos][machine-1]  # Previous machine finish time
                    job_comp_time_on_machine = max(
                        prev_job_comp_on_machine, 
                        job_available_time
                    ) + proc_time
            
            # For no-idle machines, after the first job on that machine, 
            # the completion time must be exactly prev_completion + curr_proc_time
            if no_idle_flags[machine] and pos > 0:
                c[pos][machine] = c[pos-1][machine] + proc_time
            else:
                c[pos][machine] = job_comp_time_on_machine
                
    return c[len(sequence)-1][n_machines-1]  # Last job, last machine


def calculate_detailed_completion_times(sequence, processing_times, no_idle_flags, n_jobs, n_machines):
    """
    Helper function to calculate detailed completion times for each machine position
    """
    if not sequence:
        return 0, {}
    
    # c[j][k] = completion time of job at position j on machine k
    c = [[0 for _ in range(n_machines)] for _ in range(len(sequence))]
    
    for pos, job_id in enumerate(sequence):
        for machine in range(n_machines):
            proc_time = processing_times[job_id][machine]
            
            if pos == 0:
                # First in the sequence
                if machine == 0:
                    # First job, first machine
                    c[pos][machine] = proc_time
                else:
                    # First job, but not first machine - wait for previous machine
                    c[pos][machine] = c[pos][machine-1] + proc_time
            else:
                # Not first in sequence
                if machine == 0:
                    # First machine, but not first job - wait for previous job to finish on the same machine
                    c[pos][machine] = c[pos-1][machine] + proc_time
                else:
                    # Other machines - need to wait for both:
                    # 1. The previous job to finish this machine, AND
                    # 2. This job to finish the previous machine
                    c[pos][machine] = max(
                        c[pos-1][machine],  # Prev job completed
                        c[pos][machine-1]   # This job arrived from prev machine
                    ) + proc_time
                    
            # Apply no-idle constraint: for non-first job on this machine, must not have gaps
            if no_idle_flags[machine] and pos > 0:
                c[pos][machine] = c[pos-1][machine] + proc_time

    # Generate completion times structure
    completion_times = {}
    for k in range(n_machines):
        completion_times[f"machine_{k}"] = [c[pos][k] for pos in range(len(sequence))]
        
    makespan = c[len(sequence)-1][n_machines-1]
    return makespan, completion_times


def main():
    parser = argparse.ArgumentParser(description='Solve Flow Shop Scheduling with No-Idle Constraint')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to instance JSON file')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to output solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds')
    parser.add_argument('--log_path', type=str, help='Path to log file for intermediate solutions')

    args = parser.parse_args()
    
    # Initialize logger if provided
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)

    solution = solve_flow_shop_scheduling(instance_data, args.time_limit)
    
    if solution:
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f)
    else:
        print("No solution found within the given time limit")


if __name__ == "__main__":
    main()
import argparse
import json
import math
from itertools import permutations
from solution_logger import SolutionLogger


def solve_flowshop_no_wait(args):
    # Read instance data
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    n_jobs = instance['n_jobs']
    n_machines = instance['n_machines']
    processing_times = instance['processing_times']  # Shape: [n_machines][n_jobs]
    due_dates = instance['due_dates']

    # Transpose processing_times to get job-major indexing
    proc_by_job = [[processing_times[m][j] for m in range(n_machines)] for j in range(n_jobs)]

    def calculate_makespan(sequence):
        """Calculate makespan for a given sequence under no-wait constraints."""
        start_times = [[0] * n_machines for _ in range(n_jobs)]
        
        # Schedule first job completely
        j_idx = sequence[0] - 1  
        current_time = 0
        for m in range(n_machines):
            start_times[j_idx][m] = current_time
            current_time += proc_by_job[j_idx][m]
            
        # Schedule remaining jobs
        for i in range(1, n_jobs):
            # Calculate start times for this job based on no-wait constraint
            next_job_idx = sequence[i] - 1
            
            # Determine earliest start time on first machine
            # based on the completion time of previous job on various machines
            min_start_time = 0
            prev_job_idx = sequence[i-1] - 1
            
            for m in range(n_machines):
                # Start time needed on machine 0 such that machine m is clear
                # Prev job starts on machine m at start_times[prev_job_idx][m]
                # Completion time would be start_times[prev_job_idx][m] + proc_by_job[prev_job_idx][m]
                # We need our job to start on some earlier machine at least at that time
                completion_prev = start_times[prev_job_idx][m] + proc_by_job[prev_job_idx][m]
                
                # For this job to reach machine m starting from start_time_0:
                # start_time_0 + sum(proc_by_job[next_job_idx][:m+1]) >= completion_prev
                # So start_time_0 >= completion_prev - sum(proc_by_job[next_job_idx][:m+1])
                
                sum_time_to_machine = sum(proc_by_job[next_job_idx][:m+1])
                req_time_on_machine0 = completion_prev - sum_time_to_machine
                min_start_time = max(min_start_time, req_time_on_machine0)
                
            # Now calculate start times for this job based on start time on machine 0
            start_times[next_job_idx][0] = min_start_time
            for m in range(1, n_machines):
                start_times[next_job_idx][m] = start_times[next_job_idx][m-1] + proc_by_job[next_job_idx][m-1]
            
        # Verify due dates and calculate makespan
        makespan = 0
        for j in range(n_jobs):
            job_idx = j  # j is already 0-indexed
            completion_time_last_machine = start_times[job_idx][n_machines - 1] + proc_by_job[job_idx][n_machines - 1]
            
            if completion_time_last_machine > due_dates[job_idx]:
                return float('inf')
                
            makespan = max(makespan, completion_time_last_machine)
            
        return makespan
    
    def local_search_solution():
        """Use heuristic like NEH modified for due dates and makespan."""
        # Start with a greedy heuristic solution
        base_sequence = list(range(1, n_jobs + 1))
        best_seq = base_sequence[:]
        best_makespan = calculate_makespan(best_seq)
        
        # Simple insertion-based local search
        improved = True
        while improved:
            improved = False
            for i in range(n_jobs):
                for j in range(i + 1, n_jobs):
                    # Swap positions i and j
                    candidate = best_seq[:]
                    candidate[i], candidate[j] = candidate[j], candidate[i]
                    
                    mkspn = calculate_makespan(candidate)
                    if mkspn < best_makespan:
                        best_seq = candidate
                        best_makespan = mkspn
                        improved = True
                        
                        if logger:
                            logger.log(best_makespan)
        
        return best_seq, best_makespan

    # Find initial solution using basic heuristic
    # Sort jobs by due_date / processing_time_sum ratio (shortest processing time first may work well)
    job_process_sums = [(sum(proc_by_job[j]), j+1) for j in range(n_jobs)]
    
    # Try multiple initial sequences
    best_sequence = list(range(1, n_jobs + 1))
    best_makespan = calculate_makespan(best_sequence)
    
    if best_makespan != float('inf') and logger:
        logger.log(best_makespan)
    
    # Run the local search to improve
    improved_seq, improved_mkspn = local_search_solution()
    if improved_mkspn < best_makespan:
        best_sequence = improved_seq
        best_makespan = improved_mkspn

    # If problem size is small, brute force
    if n_jobs <= 8:  # Only if practical
        import time
        start_time = time.time()
        best_makespan = float('inf')
        best_sequence = []
        
        for perm in permutations(range(1, n_jobs + 1)):
            if time.time() - start_time > args.time_limit - 1:
                break
                
            mkspn = calculate_makespan(list(perm))
            if mkspn < best_makespan:
                best_makespan = mkspn
                best_sequence = list(perm)
                
                if best_makespan != float('inf') and logger:
                    logger.log(best_makespan)

    # Final result
    final_seq = best_sequence 
    final_makespan = best_makespan if best_makespan != float('inf') else -1
    
    # Write output
    solution = {
        "objective_value": float(final_makespan),
        "sequence": final_seq
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True, help="Path to input instance JSON")
    parser.add_argument("--solution_path", required=True, help="Path to output solution JSON")
    parser.add_argument("--time_limit", type=int, default=60, help="Time limit in seconds")
    parser.add_argument("--log_path", required=False, help="Path to log intermediate solutions")

    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    solve_flowshop_no_wait(args)
import argparse
import json
import math
import random
from datetime import datetime
import time
from collections import defaultdict

def solve_job_shop_scheduling(args):
    # Load the instance data
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    # Extract instance data
    num_jobs = instance["num_jobs"]
    num_machines = instance["num_machines"]
    jobs = instance["jobs"]
    
    # Precompute operation mappings
    job_machine_operations = {}  # (job_id, machine_id) -> operation index
    machine_jobs = [[] for _ in range(num_machines)]  # List of (job_id, op_idx) for each machine
    
    for j in range(num_jobs):
        for op_idx, op in enumerate(jobs[j]["operations"]):
            machine_id = op["machine"]
            job_machine_operations[(j, machine_id)] = op_idx
            machine_jobs[machine_id].append((j, op_idx))
    
    # Initial solution using simple forward scheduling respecting precedences
    def get_initial_solution():
        # Build precedence graph constraints
        start_times = {}
        
        # Initialize with release dates
        for j in range(num_jobs):
            job_release = jobs[j]["release_date"]
            first_op = 0
            start_times[(j, first_op)] = float(job_release)
        
        # Process operations based on technological order and availability using critical path method
        changed = True
        max_iter = 100
        iter_count = 0
        while changed and iter_count < max_iter:
            changed = False
            iter_count += 1
            
            # For each job and operation after first one, check tech order
            for j in range(num_jobs):
                op_list = jobs[j]["operations"]
                for op_idx in range(1, len(op_list)):  # Start from second operation
                    prev_op = op_idx - 1
                    prev_finish_time = start_times.get((j, prev_op), 0) + op_list[prev_op]["processing_time"]
                    current_earliest = max(start_times.get((j, op_idx), 0), prev_finish_time)
                    if current_earliest > start_times.get((j, op_idx), 0):
                        start_times[(j, op_idx)] = current_earliest
                        changed = True
        
        # Now handle machine capacity using greedy assignment 
        # First build a complete list of all operations ordered by their earliest time according to precedence
        all_ops = []
        for j in range(num_jobs):
            for op_idx, op in enumerate(jobs[j]["operations"]):
                start_time = start_times.get((j, op_idx), jobs[j]["release_date"])
                if op_idx > 0:  # Make sure technology order is satisfied
                    prev_end_time = start_times.get((j, op_idx-1), jobs[j]["release_date"]) + jobs[j]["operations"][op_idx-1]["processing_time"]
                    start_time = max(start_time, prev_end_time)
                all_ops.append((j, op_idx, op["machine"], start_time, op["processing_time"]))
        
        # Process all operations, resolving machine conflicts by rescheduling later operations
        machine_available_time = [0] * num_machines
        
        # Sort operations by the earliest possible start time per machine
        all_ops.sort(key=lambda x: (x[2], x[3]))  # Sort by machine and then by earliest start time
        
        # Now assign each operation to its machine ensuring no overlap
        for j, op_idx, machine_id, est, proc_time in all_ops:
            # Find earliest available slot on the machine after EST
            start_time = max(machine_available_time[machine_id], est)
            start_times[(j, op_idx)] = start_time
            machine_available_time[machine_id] = start_time + proc_time
        
        # Now construct proper schedule
        schedule = []
        total_tardiness = 0
        for j in range(num_jobs):
            job_schedule = {
                "job_id": j,
                "operations": []
            }
            
            last_end_time = 0
            for op_idx, op in enumerate(jobs[j]["operations"]):
                st = start_times.get((j, op_idx), jobs[j]["release_date"])
                # Ensure tech order is satisfied
                if op_idx > 0:
                    prev_end_time = start_times.get((j, op_idx-1), jobs[j]["release_date"]) + jobs[j]["operations"][op_idx-1]["processing_time"]
                    st = max(st, prev_end_time, jobs[j]["release_date"])
                
                op_info = {
                    "machine": op["machine"],
                    "start_time": float(st)
                }
                job_schedule["operations"].append(op_info)
                last_end_time = st + op["processing_time"]
        
            job_completion_time = float(last_end_time)
            job_schedule["completion_time"] = job_completion_time
            
            due_dt = jobs[j]["due_date"]
            job_tardiness = max(0, job_completion_time - due_dt)
            job_schedule["tardiness"] = float(job_tardiness)
            schedule.append(job_schedule)
            
            total_tardiness += jobs[j]["weight"] * job_tardiness
        
        return {
            "objective_value": total_tardiness,
            "schedule": schedule
        }
    
    # Evaluate a solution
    def evaluate_schedule(schedule):
        total_weighted_tardiness = 0
        for job_sched in schedule:
            tardiness = job_sched["tardiness"]
            job_info = jobs[job_sched["job_id"]]
            weight = job_info["weight"]
            total_weighted_tardiness += weight * tardiness
        return total_weighted_tardiness
    
    # Local search: Swap adjacent operations on the same machine
    def local_search_improvement(initial_solution):
        best_solution = initial_solution.copy()
        best_solution["schedule"] = [dict(item) for item in initial_solution["schedule"]]
        for i, job in enumerate(initial_solution["schedule"]):
            best_solution["schedule"][i] = dict(job)
            best_solution["schedule"][i]["operations"] = [dict(op) for op in job["operations"]]
        best_objective = initial_solution["objective_value"]
        
        start_time = time.time()
        improved = True
        
        while improved and (time.time() - start_time) < args.time_limit * 0.8:
            improved = False
            operations_per_machine = defaultdict(list)
            
            # Group operations by machine
            for job_idx, job_schedule in enumerate(best_solution["schedule"]):
                for op_idx, op in enumerate(job_schedule["operations"]):
                    operations_per_machine[op["machine"]].append({
                        "job_idx": job_idx,
                        "op_idx": op_idx,
                        "original_idx": op_idx,
                        "start_time": op["start_time"]
                    })
            
            # For each machine, try swaps of adjacent operations
            for machine_id, ops_on_machine in operations_per_machine.items():
                if len(ops_on_machine) < 2:
                    continue
                
                for i in range(len(ops_on_machine) - 1):
                    # Get the two operations indices
                    op1 = ops_on_machine[i]
                    op2 = ops_on_machine[i + 1]
                    
                    # Try swapping these two operations on the machine schedule
                    new_solution = perform_swap(best_solution, op1["job_idx"], op1["op_idx"],
                                                op2["job_idx"], op2["op_idx"])
                    
                    if new_solution:
                        new_obj = new_solution["objective_value"]
                        
                        if new_obj < best_objective:
                            best_objective = new_obj
                            best_solution = new_solution
                            if logger:
                                logger.log(best_objective)
                            improved = True
                            
                            # Break to restart scan from beginning since we changed solution
                            break
                if improved:
                    break  # Restart scanning from beginning after improvement
        
        return best_solution
    
    def perform_swap(solution, job1, op1, job2, op2):
        # Deep copy solution
        new_solution = {
            "objective_value": solution["objective_value"],
            "schedule": []
        }
        
        for j_idx, job_info in enumerate(solution["schedule"]):
            new_job = {"job_id": job_info["job_id"], "operations": [], 
                       "completion_time": job_info["completion_time"], 
                       "tardiness": job_info["tardiness"]}
            
            for idx, op_info in enumerate(job_info["operations"]):
                new_op = {
                    "machine": op_info["machine"],
                    "start_time": op_info["start_time"]
                }
                new_job["operations"].append(new_op)
            
            new_solution["schedule"].append(new_job)
        
        # Simply reschedule respecting machine orders for this swap
        # We'll reschedule all operations in a way that respects machine order after swap
        try:
            new_solution = reschedule_respecting_order(new_solution, job1, op1, job2, op2)
            return new_solution
        except:
            # If rescheduling failed, return None (no valid solution)
            return None
    
    def reschedule_respecting_order(current_sol, j1, o1, j2, o2):
        # For simplicity here, just swap the start times and propagate constraints properly
        # Swap the operations start times on the machine temporarily
        job1_orig_st = current_sol["schedule"][j1]["operations"][o1]["start_time"]
        job2_orig_st = current_sol["schedule"][j2]["operations"][o2]["start_time"]
        
        current_sol["schedule"][j1]["operations"][o1]["start_time"] = job2_orig_st  
        current_sol["schedule"][j2]["operations"][o2]["start_time"] = job1_orig_st
        
        # Then reschedule respecting tech order and machine capacity using a more comprehensive propagation
        # Reset all schedules and recompute with respect to machine sequences
        temp_solution = compute_updated_schedule_from_machine_sequences(current_sol)
        return temp_solution
    
    def compute_updated_schedule_from_machine_sequences(sol):
        # Build a consistent schedule from a base respecting both tech and machine orders
        # This function propagates both types of constraints from scratch
        
        # Temporarily keep the original start times for comparison and machine-respected scheduling
        # Start by enforcing that operation on j1,o1 and j2,o2 have swapped start times relative to machine
        
        # To make it practical let's do forward scheduling respecting technology and current machine order
        start_times = {}
        
        # Initialize first operation of each job with release date
        for j in range(num_jobs):
            start_times[(j, 0)] = float(jobs[j]["release_date"])
        
        # Propagate forward with both types of constraints in mind
        # Technology first
        for j in range(num_jobs):
            for op_idx in range(1, len(jobs[j]["operations"])):
                prev_st = start_times[(j, op_idx-1)]
                start_times[(j, op_idx)] = max(
                    jobs[j]["release_date"], 
                    start_times.get((j, op_idx), 0),
                    prev_st + jobs[j]["operations"][op_idx-1]["processing_time"]
                )
        
        # Then fix machines: schedule each machine in the order determined by start times in original solution
        machine_operations = [[] for _ in range(num_machines)]
        for j in range(num_jobs):
            for op_idx, _ in enumerate(jobs[j]["operations"]):
                machine_id = jobs[j]["operations"][op_idx]["machine"]
                # Include original time to know order
                orig_time = sol["schedule"][j]["operations"][op_idx]["start_time"]
                machine_operations[machine_id].append((orig_time, j, op_idx))
        
        for machine_id in range(num_machines):
            machine_operations[machine_id].sort(key=lambda x: x[0])  # Sort by original start time to preserve intended order
            prev_machine_time = 0
            for _, j, op_idx in machine_operations[machine_id]:
                st = start_times.get((j, op_idx), 0)
                # Enforce both technology order and machine non-overlapping
                tech_req = 0
                if op_idx > 0:
                    tech_req = start_times[(j, op_idx-1)] + jobs[j]["operations"][op_idx-1]["processing_time"]
                
                # Ensure no machine conflict
                machine_req = max(prev_machine_time, st)
                
                true_start = max(tech_req, machine_req, jobs[j]["release_date"])
                start_times[(j, op_idx)] = true_start
                prev_machine_time = true_start + jobs[j]["operations"][op_idx]["processing_time"]
        
        # Build final schedule
        new_result = {"schedule": [], "objective_value": 0}
        for j in range(num_jobs):
            job_schedule = {"job_id": j, "operations": []}
            
            for op_idx in range(len(jobs[j]["operations"])):
                op_start_time = start_times[(j, op_idx)]
                
                op_entry = {
                    "machine": jobs[j]["operations"][op_idx]["machine"],
                    "start_time": float(op_start_time)
                }
                job_schedule["operations"].append(op_entry)
            
            # Compute completion & tardiness 
            final_op_time = op_start_time + jobs[j]["operations"][-1]["processing_time"]  # last start + processing
            job_schedule["completion_time"] = float(final_op_time)
            tardiness = max(0, final_op_time - jobs[j]["due_date"])
            job_schedule["tardiness"] = float(tardiness)
            
            new_result["schedule"].append(job_schedule)
            
            # Accumulate objective
            new_result["objective_value"] += jobs[j]["weight"] * tardiness
        
        return new_result

    # Get initial solution
    initial_sol = get_initial_solution()
    
    # Log the initial solution if we have a logger
    if logger:
        logger.log(initial_sol["objective_value"])
    
    # Run local search improvement up to time limit
    final_solution = local_search_improvement(initial_sol)
    
    # Ensure final solution format matches schema
    for job_sched in final_solution["schedule"]:
        # Ensure values are the right type
        job_sched["completion_time"] = float(job_sched["completion_time"])
        job_sched["tardiness"] = float(job_sched["tardiness"])
        for op in job_sched["operations"]:
            op["start_time"] = float(op["start_time"])
    
    # Write the final solution to disk
    with open(args.solution_path, 'w') as f:
        json.dump(final_solution, f)
    
    return final_solution


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    # Initialize logger if path is provided
    logger = None
    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize")

    # Solve the instance
    solution = solve_job_shop_scheduling(args)
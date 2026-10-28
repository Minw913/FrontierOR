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
    
    # Initial solution using dispatching rules (Shortest Processing Time heuristic adapted for tardiness)
    def get_initial_solution():
        # Build precedence graph constraints
        start_times = {}
        
        # Initialize with release dates
        for j in range(num_jobs):
            job_release = jobs[j]["release_date"]
            first_op = 0
            start_times[(j, first_op)] = float(job_release)
        
        # Process operations based on technological order and availability
        changed = True
        while changed:
            changed = False
            # For each operation (except the first in each job), ensure technological precedence is respected
            for j in range(num_jobs):
                op_list = jobs[j]["operations"]
                for op_idx in range(1, len(op_list)):  # Start from second operation
                    prev_op = op_idx - 1
                    # Ensure previous operation finishes before current one starts
                    required_start = start_times.get((j, prev_op), 0) + op_list[prev_op]["processing_time"]
                    current_start = start_times.get((j, op_idx), 0)
                    if required_start > current_start:
                        start_times[(j, op_idx)] = required_start
                        changed = True
        
        # Now add machine capacity constraints incrementally
        max_iterations = 100  # Prevent infinite loops
        iteration = 0
        
        while iteration < max_iterations:
            changed = False
            iteration += 1
            
            # Sort operations per machine by some heuristic and adjust timing accordingly
            for m in range(num_machines):
                machine_ops = []
                for j, op_idx in machine_jobs[m]:
                    if (j, op_idx) in start_times:
                        machine_ops.append((j, op_idx, start_times[(j, op_idx)], 
                                           jobs[j]["operations"][op_idx]["processing_time"]))
                
                if len(machine_ops) <= 1:
                    continue
                
                # Sort by earliest start time respecting technological constraints again
                sorted_ops = sorted(machine_ops, key=lambda x: x[2])
                
                # Resolve machine conflicts by pushing later operations back
                for i in range(1, len(sorted_ops)):
                    earlier_op = sorted_ops[i-1]
                    later_op = sorted_ops[i]
                    
                    e_job, e_op_idx, e_start, e_duration = earlier_op
                    l_job, l_op_idx, l_start, l_duration = later_op
                    
                    e_end = e_start + e_duration
                    needed_start = max(l_start, e_end)
                    
                    if needed_start > l_start:
                        old_l_start = start_times[(l_job, l_op_idx)]
                        start_times[(l_job, l_op_idx)] = needed_start
                        
                        # Update all subsequent operations in that job to maintain tech order
                        for next_op_idx in range(l_op_idx + 1, len(jobs[l_job]["operations"])):
                            current_next = start_times.get((l_job, next_op_idx), 0)
                            required_next = start_times[(l_job, next_op_idx - 1)] + jobs[l_job]["operations"][next_op_idx-1]["processing_time"]
                            actual_next_start = max(current_next, required_next)
                            
                            if actual_next_start != current_next:
                                start_times[(l_job, next_op_idx)] = actual_next_start
                                changed = True
                        
                        changed = True
                        # Also move any subsequent machine ops for this machine forward appropriately
                        # This is handled in the next iterations
            
            if not changed:
                break
        
        # Construct the full schedule
        schedule = []
        for j in range(num_jobs):
            job_schedule = {
                "job_id": j,
                "operations": []
            }
            
            for op_idx, op in enumerate(jobs[j]["operations"]):
                st = start_times.get((j, op_idx), jobs[j]["release_date"])
                if op_idx > 0:
                    prev_end_time = start_times[(j, op_idx - 1)] + jobs[j]["operations"][op_idx-1]["processing_time"]
                    st = max(st, prev_end_time)
                st = max(st, jobs[j]["release_date"])
                
                op_info = {
                    "machine": op["machine"],
                    "start_time": st
                }
                job_schedule["operations"].append(op_info)
        
            # Calculate job completion time and tardiness
            final_op_time = start_times[(j, len(jobs[j]["operations"])-1)] + jobs[j]["operations"][len(jobs[j]["operations"])-1]["processing_time"]
            job_schedule["completion_time"] = final_op_time
            due_dt = jobs[j]["due_date"]
            tardi = max(0, final_op_time - due_dt)
            job_schedule["tardiness"] = tardi
            schedule.append(job_schedule)
        
        # Calculate objective value
        total_tardiness = sum(job["weight"] * job["tardiness"] for job in schedule)
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
    
    # Local search improvement
    def local_search_improvement(initial_solution):
        best_solution = initial_solution
        best_objective = initial_solution["objective_value"]
        
        # Define moves for local search: swap sequence within a machine
        machines_ops = [[] for _ in range(num_machines)]
        for job_scheduler in best_solution["schedule"]:
            jid = job_scheduler["job_id"]
            op_list = job_scheduler["operations"]
            for idx, op_info in enumerate(op_list):
                machine_id = op_info["machine"]
                start_time = op_info["start_time"]
                proc_time = jobs[jid]["operations"][idx]["processing_time"]
                machines_ops[machine_id].append({
                    "job_id": jid,
                    "op_idx": idx,
                    "start_time": start_time,
                    "proc_time": proc_time
                })
        
        for machine in range(num_machines):
            machines_ops[machine].sort(key=lambda x: x["start_time"])  # Order by current time slot
        
        # Perform swaps within each machine
        time_limit_reached = False
        local_start_time = time.time()
        
        for m in range(num_machines):
            ops_list = machines_ops[m]
            n_ops = len(ops_list)
            
            # Try swapping pairs of operations on the same machine
            for i in range(n_ops):
                for j in range(i+1, n_ops):
                    # Swap operations i and j in the ordering for machine m
                    if time.time() - local_start_time > args.time_limit * 0.8:  # Save time for final evaluation
                        time_limit_reached = True
                        break
                    
                    temp_ops_list = ops_list[:]
                    temp_ops_list[i], temp_ops_list[j] = temp_ops_list[j], temp_ops_list[i]
                    
                    # Create a new schedule based on this swap and reschedule respecting constraints
                    new_solution = reschedule_with_order(m, temp_ops_list, best_solution)
                    if not new_solution:
                        continue
                    
                    new_obj = new_solution["objective_value"]
                    
                    if new_obj < best_objective:
                        best_objective = new_obj
                        best_solution = new_solution
                        
                        # Update log
                        if logger:
                            logger.log(best_objective)
                    # Otherwise keep going
            if time_limit_reached:
                break
        
        return {
            "objective_value": best_objective,
            "schedule": best_solution["schedule"]
        }
    
    def reschedule_with_order(machine_id, ordered_ops, current_solution):
        # Rebuild schedule with specific ordering on one machine but retain others
        schedule = [None] * num_jobs
        
        # Copy over operation start times except for conflicting machine
        op_times = {}
        for job_sched in current_solution["schedule"]:
            j = job_sched["job_id"]
            for k, op_info in enumerate(job_sched["operations"]):
                m = op_info["machine"]
                existing_start = op_info["start_time"]
                if m != machine_id:
                    op_times[(j, k)] = existing_start
        
        # Assign times based on the provided order for the specific machine
        current_time = 0
        for op_info in ordered_ops:
            job_id = op_info["job_id"]
            op_idx = op_info["op_idx"]
            proc_time = op_info["proc_time"]
            
            # Determine earliest start time considering:
            # - release date
            # - previous operation in job completed
            # - machine available from current_time
            
            earliest_by_prev_job_op = 0
            if op_idx > 0:
                earliest_by_prev_job_op = op_times.get((job_id, op_idx-1), 0) + \
                                          jobs[job_id]["operations"][op_idx-1]["processing_time"]
            
            release_time = jobs[job_id]["release_date"]
            new_start_time = max(current_time, earliest_by_prev_job_op, release_time)
            
            op_times[(job_id, op_idx)] = new_start_time
            current_time = new_start_time + proc_time
        
        # Verify technology order holds for affected jobs (should still hold after swap)
        # Create final schedule
        for j in range(num_jobs):
            job_schedule = {"job_id": j, "operations": []}
            
            for op_idx in range(len(jobs[j]["operations"])):
                st = op_times.get((j, op_idx), jobs[j]["release_date"])
                # Check tech constraint again explicitly
                if op_idx > 0:
                    prev_op_end = op_times.get((j, op_idx-1), 0) + jobs[j]["operations"][op_idx-1]["processing_time"]
                    st = max(st, prev_op_end)
                
                op_info = {
                    "machine": jobs[j]["operations"][op_idx]["machine"],
                    "start_time": st
                }
                job_schedule["operations"].append(op_info)
            
            # Compute completion time and tardiness
            final_op = len(jobs[j]["operations"]) - 1
            comp_time = op_times.get((j, final_op), 0) + jobs[j]["operations"][final_op]["processing_time"]
            job_schedule["completion_time"] = float(comp_time)
            due_dt = jobs[j]["due_date"]
            tardy = max(0, comp_time - due_dt)
            job_schedule["tardiness"] = float(tardy)
            
            schedule[j] = job_schedule
                
        obj_val = sum(s["tardiness"] * jobs[s["job_id"]]["weight"] for s in schedule)
        return {
            "objective_value": obj_val,
            "schedule": schedule
        }

    # Get initial solution
    initial_sol = get_initial_solution()
    
    # Log the initial solution if we have a logger
    if logger:
        logger.log(initial_sol["objective_value"])
    
    # Run local search improvement up to time limit
    final_solution = local_search_improvement(initial_sol)
    
    # Verify that solution structure matches schema
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
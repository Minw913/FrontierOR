import math
import json
import argparse
import time
from collections import defaultdict
import gurobipy as gp
from solution_logger import SolutionLogger

def compute_remaining_processing_times(instance):
    num_jobs = instance['num_jobs']
    num_machines = instance['num_machines']
    jobs = instance['jobs']
    rem = [[0] * num_machines for _ in range(num_jobs)]
    for j in range(num_jobs):
        ops = jobs[j]['operations']
        rem[j][num_machines-1] = ops[num_machines-1]['processing_time']
        for i in range(num_machines-2, -1, -1):
            rem[j][i] = ops[i]['processing_time'] + rem[j][i+1]
    return rem

def run_heuristic_simulation(instance, k, time_limit=None):
    num_jobs = instance['num_jobs']
    num_machines = instance['num_machines']
    jobs = instance['jobs']
    start_time = time.time()
    
    # Precompute global average processing time
    total_processing_time = 0
    for j in range(num_jobs):
        for op in jobs[j]['operations']:
            total_processing_time += op['processing_time']
    global_avg_p = total_processing_time / (num_jobs * num_machines)
    
    # Precompute remaining processing times for each job and step
    rem = compute_remaining_processing_times(instance)
    
    # Initialize data structures
    free_time = [0] * num_machines  # when each machine becomes free
    available_time = [jobs[j]['release_date'] for j in range(num_jobs)]  # when each job becomes available
    next_op_index = [0] * num_jobs  # next operation index for each job
    finished = [False] * num_jobs   # whether the job is finished
    start_times = [[0] * num_machines for _ in range(num_jobs)]  # to record start times
    unscheduled_count = num_jobs * num_machines
    
    # Precompute machine index for each operation per job
    # For quick lookup: for job j, operation i -> machine
    op_machine = []
    for j in range(num_jobs):
        machine_list = [op['machine'] for op in jobs[j]['operations']]
        op_machine.append(machine_list)
    
    while unscheduled_count > 0:
        if time_limit is not None and time.time() - start_time > time_limit:
            break
            
        # Find next event time: min of machine free times and job available times (for unfinished jobs)
        min_free_time = min(free_time)
        min_avail_time = min([available_time[j] for j in range(num_jobs) if not finished[j]])
        t = min(min_free_time, min_avail_time)
        
        # Collect candidate jobs by machine: jobs available at time t and whose next operation's machine is free at t
        candidate_jobs_by_machine = defaultdict(list)
        for j in range(num_jobs):
            if finished[j] or available_time[j] > t:
                continue
            step = next_op_index[j]
            m = op_machine[j][step]
            if free_time[m] <= t:
                candidate_jobs_by_machine[m].append(j)
        
        if not candidate_jobs_by_machine:
            # No candidates at t, find next event time > t
            next_free_times = [ft for ft in free_time if ft > t]
            next_avail_times = [available_time[j] for j in range(num_jobs) if not finished[j] and available_time[j] > t]
            if not next_free_times and not next_avail_times:
                break  # should not happen
            next_t = min(next_free_times + next_avail_times) if next_free_times or next_avail_times else t+1
            # Force time to advance
            t = next_t
            continue
        
        # Schedule the best job for each machine with candidates
        for m, job_list in candidate_jobs_by_machine.items():
            best_job = None
            best_priority = -10**15
            for j in job_list:
                step = next_op_index[j]
                p_ij = jobs[j]['operations'][step]['processing_time']
                rem_time = rem[j][step]
                slack = jobs[j]['due_date'] - t - rem_time
                exponent = -max(0, slack) / (k * global_avg_p)
                I_j = (jobs[j]['weight'] / p_ij) * math.exp(exponent)
                if I_j > best_priority:
                    best_priority = I_j
                    best_job = j
            
            if best_job is None:
                continue
                
            j = best_job
            step = next_op_index[j]
            p_ij = jobs[j]['operations'][step]['processing_time']
            
            # Schedule job j on machine m at time t
            start_times[j][step] = t
            free_time[m] = t + p_ij
            available_time[j] = t + p_ij
            next_op_index[j] += 1
            if next_op_index[j] == num_machines:
                finished[j] = True
            unscheduled_count -= 1
    
    # Build the schedule in the output format
    schedule = []
    total_tardiness = 0.0
    for j in range(num_jobs):
        last_op_index = num_machines - 1
        completion_time = start_times[j][last_op_index] + jobs[j]['operations'][last_op_index]['processing_time']
        tardiness = max(0.0, completion_time - jobs[j]['due_date'])
        total_tardiness += jobs[j]['weight'] * tardiness
        
        operations = []
        for i in range(num_machines):
            machine = jobs[j]['operations'][i]['machine']
            operations.append({
                "machine": machine,
                "start_time": float(start_times[j][i])
            })
        
        schedule.append({
            "job_id": j,
            "completion_time": float(completion_time),
            "tardiness": float(tardiness),
            "operations": operations
        })
    
    return total_tardiness, schedule

def run_mip(instance, time_limit, initial_solution, logger):
    num_jobs = instance['num_jobs']
    num_machines = instance['num_machines']
    jobs = instance['jobs']
    
    # Precompute a large M: max(release_date) + total_processing_time of all jobs
    max_release = max(job['release_date'] for job in jobs)
    total_processing = sum(op['processing_time'] for job in jobs for op in job['operations'])
    H = max_release + total_processing
    
    # Precompute for each job j and machine m: the step index i and processing time
    machine_step = [[-1] * num_machines for _ in range(num_jobs)]
    machine_pt = [[0] * num_machines for _ in range(num_jobs)]
    for j in range(num_jobs):
        for i, op in enumerate(jobs[j]['operations']):
            m = op['machine']
            machine_step[j][m] = i
            machine_pt[j][m] = op['processing_time']
    
    model = gp.Model()
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', time_limit)
    
    # Variables: start times for each operation
    S = {}  # S[job_id, operation_index]
    for j in range(num_jobs):
        for i in range(num_machines):
            S[(j, i)] = model.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name=f'S_{j}_{i}')
    
    # Tardiness variables
    T = {}
    for j in range(num_jobs):
        T[j] = model.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name=f'T_{j}')
    
    # Binary variables for machine orders: for each machine m and job pair (j1, j2) with j1 < j2
    y = {}
    for m in range(num_machines):
        for j1 in range(num_jobs):
            for j2 in range(j1+1, num_jobs):
                y[(m, j1, j2)] = model.addVar(vtype=gp.GRB.BINARY, name=f'y_{m}_{j1}_{j2}')
    
    # Constraints
    # Release dates
    for j in range(num_jobs):
        model.addConstr(S[(j, 0)] >= jobs[j]['release_date'], name=f'release_{j}')
    
    # Precedence within a job
    for j in range(num_jobs):
        ops = jobs[j]['operations']
        for i in range(num_machines-1):
            model.addConstr(
                S[(j, i+1)] >= S[(j, i)] + ops[i]['processing_time'],
                name=f'prec_{j}_{i}'
            )
    
    # Machine constraints: disjunctive constraints for operations on same machine
    for m in range(num_machines):
        for j1 in range(num_jobs):
            i1 = machine_step[j1][m]
            p1 = machine_pt[j1][m]
            for j2 in range(j1+1, num_jobs):
                i2 = machine_step[j2][m]
                p2 = machine_pt[j2][m]
                model.addConstr(
                    S[(j1, i1)] + p1 <= S[(j2, i2)] + H * (1 - y[(m, j1, j2)]),
                    name=f'machine_{m}_{j1}_{j2}_1'
                )
                model.addConstr(
                    S[(j2, i2)] + p2 <= S[(j1, i1)] + H * y[(m, j1, j2)]),
                    name=f'machine_{m}_{j1}_{j2}_2'
                )
    
    # Tardiness constraints
    for j in range(num_jobs):
        last_op_index = num_machines - 1
        last_pt = jobs[j]['operations'][last_op_index]['processing_time']
        model.addConstr(
            T[j] >= S[(j, last_op_index)] + last_pt - jobs[j]['due_date'],
            name=f'tardiness_{j}'
        )
    
    # Objective: minimize total weighted tardiness
    obj = gp.quicksum(jobs[j]['weight'] * T[j] for j in range(num_jobs))
    model.setObjective(obj, gp.GRB.MINIMIZE)
    
    # Set initial solution if provided
    if initial_solution is not None:
        for j in range(num_jobs):
            for i in range(num_machines):
                start_val = initial_solution['start_times'][j][i]
                S[(j, i)].Start = start_val
        
        # Set y variables based on start times
        for m in range(num_machines):
            for j1 in range(num_jobs):
                i1 = machine_step[j1][m]
                s1 = initial_solution['start_times'][j1][i1]
                for j2 in range(j1+1, num_jobs):
                    i2 = machine_step[j2][m]
                    s2 = initial_solution['start_times'][j2][i2]
                    if s1 < s2:
                        y_val = 1
                    else:
                        y_val = 0
                    y[(m, j1, j2)].Start = y_val
    
    # Optimize
    model.optimize()
    
    # Extract solution
    if model.SolCount > 0:
        # Build schedule from best solution
        start_times_sol = [[0.0] * num_machines for _ in range(num_jobs)]
        for j in range(num_jobs):
            for i in range(num_machines):
                start_times_sol[j][i] = S[(j, i)].X
        
        schedule = []
        total_tardiness = 0.0
        for j in range(num_jobs):
            last_op_index = num_machines - 1
            completion_time = start_times_sol[j][last_op_index] + jobs[j]['operations'][last_op_index]['processing_time']
            tardiness = max(0.0, completion_time - jobs[j]['due_date'])
            total_tardiness += jobs[j]['weight'] * tardiness
            
            operations = []
            for i in range(num_machines):
                machine = jobs[j]['operations'][i]['machine']
                operations.append({
                    "machine": machine,
                    "start_time": float(start_times_sol[j][i])
                })
            
            schedule.append({
                "job_id": j,
                "completion_time": float(completion_time),
                "tardiness": float(tardiness),
                "operations": operations
            })
        
        return total_tardiness, schedule
    else:
        return None

def main():
    parser = argparse.ArgumentParser(description='Job Shop Scheduling with Weighted Tardiness Minimization')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the instance JSON file')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to the solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds')
    parser.add_argument('--log_path', type=str, default=None, help='Path to the log file (optional)')
    args = parser.parse_args()
    
    # Read instance
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    num_jobs = instance['num_jobs']
    num_machines = instance['num_machines']
    jobs = instance['jobs']
    
    # Initialize logger if log_path is provided
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    start_time = time.time()
    time_left = args.time_limit
    
    # Step 1: Run heuristic with multiple k values
    best_heuristic_obj = float('inf')
    best_heuristic_schedule = None
    k_list = [0.5, 1.0, 2.0, 5.0]
    
    for k in k_list:
        if time.time() - start_time > args.time_limit:
            break
            
        # Run heuristic simulation with time limit
        heuristic_time_left = max(0, args.time_limit - (time.time() - start_time))
        obj, schedule = run_heuristic_simulation(instance, k, heuristic_time_left)
        
        # Log if this solution is better
        if logger and obj < best_heuristic_obj:
            logger.log(obj)
        
        if obj < best_heuristic_obj:
            best_heuristic_obj = obj
            best_heuristic_schedule = schedule
    
    # Extract start times for MIP start if needed
    initial_solution_for_mip = None
    if best_heuristic_schedule:
        start_times_heuristic = [[0.0] * num_machines for _ in range(num_jobs)]
        for job_sched in best_heuristic_schedule:
            j = job_sched['job_id']
            for i, op in enumerate(job_sched['operations']):
                start_times_heuristic[j][i] = op['start_time']
        initial_solution_for_mip = {'start_times': start_times_heuristic}
    
    # Step 2: Run MIP for small instances if time permits
    final_schedule = best_heuristic_schedule
    final_obj = best_heuristic_obj
    
    if num_jobs <= 20:  # Only for small instances
        mip_time_left = max(0, args.time_limit - (time.time() - start_time))
        if mip_time_left > 1:  # Only if there's enough time left
            mip_result = run_mip(instance, mip_time_left, initial_solution_for_mip, logger)
            if mip_result is not None:
                mip_obj, mip_schedule = mip_result
                if mip_obj < final_obj:  # Only use if better than heuristic
                    final_obj = mip_obj
                    final_schedule = mip_schedule
    
    # If MIP didn't run or failed, use best heuristic
    if final_schedule is None and best_heuristic_schedule is not None:
        final_schedule = best_heuristic_schedule
        final_obj = best_heuristic_obj
    
    # Write solution
    solution = {
        "objective_value": float(final_obj),
        "schedule": final_schedule
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()
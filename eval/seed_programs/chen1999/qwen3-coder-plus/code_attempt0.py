import argparse
import json
import math
import time
import random
from collections import defaultdict
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def solve_instance(instance_data, time_limit, logger):
    problem_type = instance_data["problem_type"]
    
    if problem_type == "weighted_completion_time":
        return solve_weighted_completion_time(instance_data, time_limit, logger)
    elif problem_type == "weighted_tardy_jobs":
        return solve_weighted_tardy_jobs(instance_data, time_limit, logger)
    else:
        raise ValueError(f"Unknown problem type: {problem_type}")


def solve_weighted_completion_time(instance_data, time_limit, logger):
    num_jobs = instance_data["num_jobs"]
    num_machines = instance_data["num_machines"]
    proc_times = instance_data["jobs"]["processing_times"]
    weights = instance_data["jobs"]["weights"]

    # Create model
    model = gp.Model("weighted_completion_time")
    model.setParam('TimeLimit', time_limit)
    model.setParam('Threads', 1)  # Single core constraint
    
    # Variable: x[i, j] = 1 means job i is assigned to machine j
    x = {}
    for i in range(num_jobs):
        for j in range(num_machines):
            x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
    
    # For minimizing weighted completion time, we can model this as quadratic
    # However, we'll use a different formulation based on assignment
    # Define binary variables for ordering: y[i1, i2, j] = 1 means job i1 precedes job i2 on machine j
    y = {}
    for i1 in range(num_jobs):
        for i2 in range(num_jobs):
            if i1 != i2:
                for j in range(num_machines):
                    y[i1, i2, j] = model.addVar(vtype=GRB.BINARY, name=f"y_{i1}_{i2}_{j}")
    
    # Each job is assigned to exactly one machine
    for i in range(num_jobs):
        model.addConstr(gp.quicksum(x[i, j] for j in range(num_machines)) == 1)
    
    # If both i1 and i2 are on machine j, then the order must be established
    for i1 in range(num_jobs):
        for i2 in range(num_jobs):
            if i1 != i2:
                for j in range(num_machines):
                    model.addConstr(y[i1, i2, j] + y[i2, i1, j] <= x[i1, j])
                    model.addConstr(y[i1, i2, j] + y[i2, i1, j] <= x[i2, j])
    
    # Transitivity constraints (optional relaxation depending on performance)
    # To make this practical, let's implement a simpler, more scalable approach
    
    # Reformulation idea: calculate the weighted completion time for each possible assignment
    # and use more efficient modeling
    
    # For each machine j, for each subset S of jobs assigned to j, 
    # there exists an optimal sequence. We'll allow for sequencing possibilities.
    
    # Add another approach based on position-indexed variables
    # p_ij_position[k][i] would mean job i is at position k on machine j
    
    # This makes the full MILP too large for general cases; instead:
    
    # Let's use a heuristic that works with the Gurobi model for assignment,
    # combined with optimal sequencing.
    # For minimizing \sum w_i * C_i on a single machine, Smith's rule states:
    # sequence jobs in non-decreasing order of p_i/w_i (shortest processing time first adjusted for weight).
    
    # We'll implement a quadratic objective version
    # Variables for completion times - for assigned jobs
    comp_time = {}
    for i in range(num_jobs):
        comp_time[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"comp_time_{i}")
    
    # We need constraints linking assignments and completion times
    # For each machine j, jobs are ordered
    # We use BigM approach with ordering
    
    # Instead, create a simplified heuristic-based MILP
    pos_vars = {}
    max_pos = min(num_jobs, 10)  # To limit complexity
    for j in range(num_machines):
        for pos in range(max_pos):  
            for i in range(num_jobs):
                pos_vars[pos, i, j] = model.addVar(vtype=GRB.BINARY, name=f"pos_{pos}_{i}_{j}")
    
    total_wct_terms = []
    for j in range(num_machines):
        for pos in range(max_pos):  
            for i in range(num_jobs):
                # When job i is at position pos on machine j, 
                # its total processing time is sum of preceding jobs + partial processing of job i
                # Completion time of job i = sum(p_k where k precedes i and k assigned to j) + p_i
                # Using position vars, if job i is at position pos, then its completion time >= sum of first pos procs
                
                cumulative_p = sum(proc_times[k][j] for k_preceding in range(pos) for k in range(num_jobs) 
                                  if pos_vars.get((k_preceding, k, j)))
                
                # Simplify by linearizing: For each job i at position pos on machine j, completion time is:
                # sum of proc_times of jobs at positions 0..pos-1 on j + proc_times[i][j]
                
                # Link pos_vars with actual assignments                
                model.addConstr(gp.quicksum(pos_vars[pos, i, j] for i in range(num_jobs)) <= 1)
                
                if pos > 0:
                    # Each position on each machine gets at most one job
                    prev_sum = gp.quicksum(
                        gp.quicksum(proc_times[other_i][j] * pos_vars[p, other_i, j] 
                                   for other_i in range(num_jobs))
                        for p in range(pos)
                    )
                    
                    # If i is at pos on j, its completion is prev_sum + proc[i][j]
                    completion_var_expr = prev_sum + proc_times[i][j] * pos_vars[pos, i, j]
                    
                    # Constrain completion time of i accordingly
                    model.addConstr(comp_time[i] >= completion_var_expr)
                else:
                    # Position 0: completion time = proc_time
                    model.addConstr(comp_time[i] >= proc_times[i][j] * pos_vars[pos, i, j])
    
    # Each job assigned exactly once
    for i in range(num_jobs):
        model.addConstr(gp.quicksum(gp.quicksum(pos_vars[pos, i, j] 
                                              for j in range(num_machines)) 
                                   for pos in range(max_pos)) == 1)
    
    # Build total objective
    objective_expr = gp.quicksum(weights[i] * comp_time[i] for i in range(num_jobs))
    model.setObjective(objective_expr, GRB.MINIMIZE)
    
    model.optimize()
    
    if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        # Extract results
        assigned_jobs = {}
        for j in range(num_machines):
            assigned_jobs[j] = []
            for pos in range(max_pos):
                for i in range(num_jobs):
                    if pos_vars[pos, i, j].X > 0.99:
                        assigned_jobs[j].append(i)
        
        completion_times = {i: comp_time[i].X for i in range(num_jobs)}
        schedule = {}
        for j in range(num_machines):
            schedule[str(j)] = assigned_jobs[j]
        
        # Verify and fix sequences based on extracted solution (Smith's Rule locally)
        refined_schedule = refine_schedules_with_smith_rule(schedule, proc_times, weights, instance_data)
        
        for j in range(num_machines):
            schedule[str(j)] = refined_schedule.get(str(j), [])
            
        refined_completion_times, total_weighted_completion_time = calculate_completion_times(
            schedule, proc_times, instance_data
        )
        
        # Log solution if improved
        if logger and total_weighted_completion_time < float('inf'):
            logger.log(total_weighted_completion_time)
        
        return {
            "objective_value": total_weighted_completion_time,
            "schedule": schedule,
            "completion_times": refined_completion_times,
            "tardy_jobs": [],  # Only used for tardy jobs variant
            "on_time_jobs": list(range(num_jobs))  # All jobs are scheduled
        }
    
    # Fallback: Greedy heuristic if model is too complex
    return solve_weighted_completion_greedy(instance_data)


def solve_weighted_completion_greedy(instance_data):
    """Fallback greedy heuristic"""
    num_jobs = instance_data["num_jobs"]
    num_machines = instance_data["num_machines"]
    proc_times = instance_data["jobs"]["processing_times"]
    weights = instance_data["jobs"]["weights"]
    
    # Compute WSRPT ratio
    ratios = [weights[i] / proc_times[i][j] if proc_times[i][j] > 0 else 0
              for i in range(num_jobs) for j in range(num_machines)]
              
    # Assign using greedy heuristics
    assigned = [False] * num_jobs
    schedule = {str(j): [] for j in range(num_machines)}
    
    # Sort jobs by weight, assign greedily
    job_order = list(range(num_jobs))
    job_order.sort(key=lambda i: weights[i], reverse=True)  # Higher weight first
    
    load_per_machine = [0] * num_machines
    
    for i in job_order:
        # Find machine with smallest load after adding this job
        best_j = 0
        best_cost = float('inf')
        for j in range(num_machines):
            est_comp = load_per_machine[j] + proc_times[i][j]
            curr_cost = weights[i] * est_comp
            if curr_cost < best_cost:
                best_cost = curr_cost
                best_j = j
        
        schedule[str(best_j)].append(i)
        load_per_machine[best_j] += proc_times[i][best_j]
    
    # Sort according to Smith's rule (p_i / w_i) to optimize WCT on each machine
    for j in range(num_machines):
        schedule[str(j)].sort(key=lambda i: proc_times[i][j] / max(weights[i], 0.1))
    
    completion_times, obj_val = calculate_completion_times(schedule, proc_times, instance_data)
    
    return {
        "objective_value": obj_val,
        "schedule": schedule,
        "completion_times": completion_times,
        "tardy_jobs": [],
        "on_time_jobs": list(range(num_jobs))
    }


def calculate_completion_times(schedule, proc_times, instance_data):
    machines = len(instance_data["machines"]["speeds"]) if instance_data["machine_type"] == "uniform" else len(proc_times[0]) if proc_times else len(instance_data["machines"])
    completion_times = {}
    total_wct = 0.0
    
    for machine_str, job_list in schedule.items():
        machine_idx = int(machine_str)
        current_time = 0
        for i in job_list:
            current_time += proc_times[i][machine_idx]
            completion_times[i] = current_time
            total_wct += instance_data["jobs"]["weights"][i] * completion_times[i]
    
    return completion_times, total_wct


def refine_schedules_with_smith_rule(schedule, proc_times, weights, instance_datum):
    """Refine an assignment by sorting jobs optimally on each machine."""
    refined = {}
    for machine_str, job_list in schedule.items():
        refined[machine_str] = sorted(job_list, key=lambda i: proc_times[i][int(machine_str)] / max(weights[i], 0.1))
    return refined


def solve_weighted_tardy_jobs(instance_data, time_limit, logger):
    num_jobs = instance_data["num_jobs"]
    num_machines = instance_data["num_machines"]
    proc_times = instance_data["jobs"]["processing_times"]
    weights = instance_data["jobs"]["weights"]
    due_dates = instance_data["jobs"]["due_dates"]
    
    # This is complex even for modern solvers; let’s implement a simpler but effective formulation
    model = gp.Model("weighted_tardy_jobs")
    model.setParam('TimeLimit', time_limit)
    model.setParam('Threads', 1)
    
    # Binary var: x_i_j = 1 means job i is assigned to machine j
    x = {}
    # Binary var: T_i = 1 means job i is tardy
    T = {}
    
    for i in range(num_jobs):
        T[i] = model.addVar(vtype=GRB.BINARY, name=f"T_{i}")
        for j in range(num_machines):
            x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
    
    # Completion time variables
    C = {}
    for i in range(num_jobs):
        C[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"C_{i}")
    
    # Each job assigned to at most one machine (or none if it's tardy)
    for i in range(num_jobs):
        model.addConstr(gp.quicksum(x[i, j] for j in range(num_machines)) <= 1)
    
    # Link completion with assignment: if x[i,j]=1 then C[i] >= p_ij, else ignore
    for i in range(num_jobs):
        for j in range(num_machines):
            # C[i] >= p[i][j] * x[i][j]
            model.addConstr(C[i] >= proc_times[i][j] * x[i, j])
    
    # Ensure if on time, then tardiness = 0, otherwise allow tardiness=1
    # If C[i] > d[i], force T[i] = 1
    M_large = sum(sum(proc_times[i]) for i in range(num_jobs))  # Big M
    for i in range(num_jobs):
        # C[i] <= d[i] + M*(1 - sum(x[i,j])) + M*T[i]
        # If job isn't scheduled somewhere, it's automatically tardy
        # If it's scheduled and finishes late, T[i] must be 1
        model.addConstr(C[i] <= due_dates[i] + M_large * (1 - gp.quicksum(x[i, j] for j in range(num_machines))) + M_large * T[i])
    
    # Add sequence-dependent constraints (for scheduling within a machine)
    # We use position-based variables for tractable formulation
    max_pos = min(num_jobs, 5)
    pos_vars = {}
    for j in range(num_machines):
        for pos in range(max_pos):
            for i in range(num_jobs):
                pos_vars[pos, i, j] = model.addVar(vtype=GRB.BINARY, name=f"pos_{pos}_{i}_{j}")
    
    # Position implies assignment
    for (pos, i, j) in pos_vars:
        model.addConstr(pos_vars[pos, i, j] <= x[i, j])
        
    # One job per position per machine at most
    for j in range(num_machines):
        for pos in range(max_pos):
            model.addConstr(gp.quicksum(pos_vars[pos, i, j] for i in range(num_jobs)) <= 1)
    
    # If x[i,j] = 1 then i is at some position on machine j
    for i in range(num_jobs):
        for j in range(num_machines):
            model.addConstr(x[i, j] <= gp.quicksum(pos_vars[pos, i, j] for pos in range(max_pos)))
    
    # Enforce completion time logic using position variables
    for j in range(num_machines):
        for pos in range(1, max_pos):  # From position 1 onward
            for i in range(num_jobs):
                if pos_vars.get((pos, i, j)):
                    prev_cum_time = gp.quicksum(
                        gp.quicksum(proc_times[k][j] * pos_vars[prev_pos, k, j] 
                                   for k in range(num_jobs))
                        for prev_pos in range(pos)
                    )
                    # If job i is at pos on machine j, then its completion includes job's proc time and all previous
                    model.addConstr(C[i] >= (prev_cum_time + proc_times[i][j] ) * pos_vars[pos, i, j])
    
    # If no more than a certain number of jobs are allowed on a machine based on total demand,
    # we could have constraints. But keeping flexible.
    
    # Objective: minimize sum of weights of tardy jobs
    objective = gp.quicksum(weights[i] * T[i] for i in range(num_jobs))
    model.setObjective(objective, GRB.MINIMIZE)
    
    model.optimize()

    if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        # Extract assignment
        schedule = {str(j): [] for j in range(num_machines)}
        assigned = set()
        for i in range(num_jobs):
            for j in range(num_machines):
                if x[i, j].X > 0.5:
                    # Determine its position on the machine
                    for pos in range(max_pos):
                        if pos_vars.get((pos, i, j)) and pos_vars[pos, i, j].X > 0.5:
                            schedule[str(j)].append(i)
                            break
                    assigned.add(i)
                    break
        
        # Refine job orders using Earliest due date (EDD) rule or Moore's algorithm for each machine
        for j in range(num_machines):
            # Moore's algorithm implementation to minimize no. tardy jobs per machine (subset problem)
            schedule[str(j)] = moore_algorithm_for_on_time_jobs(
                schedule[str(j)], proc_times, due_dates, instance_data, int(j)
            )
        
        tardy_jobs = [i for i in range(num_jobs) if i not in assigned or T[i].X > 0.5]
        on_time_jobs = [i for i in range(num_jobs) if i in assigned and T[i].X < 0.5]
        
        # Calculate completion times for on-time jobs
        completion_times = {}
        for j in range(num_machines):
            current_time = 0
            for job_id in schedule[str(j)]:
                current_time += proc_times[job_id][j]
                if job_id not in completion_times:  # Don't overwrite if already set in a better way
                    completion_times[job_id] = current_time
        
        # Update tardy jobs for those assigned to machine but missing deadlines
        actual_tardy = []
        actual_on_time = []
        for j in range(num_machines):
            current_time = 0
            for i in schedule[str(j)]:
                current_time += proc_times[i][j]
                if current_time > due_dates[i]:
                    actual_tardy.append(i)
                else:
                    actual_on_time.append(i)
        for i in range(num_jobs):
            if i not in actual_on_time and (i in assigned):
                actual_tardy.append(i)
        
        # Recalculate objective
        actual_obj_val = sum(weights[i] for i in actual_tardy)
        
        if logger and actual_obj_val < float('inf'):
            logger.log(actual_obj_val)
        
        return {
            "objective_value": actual_obj_val,
            "schedule": schedule,
            "completion_times": completion_times,
            "tardy_jobs": actual_tardy,
            "on_time_jobs": actual_on_time
        }
    
    # Fallback: greedy heuristic
    return solve_weighted_tardy_greedy(instance_data)


def moore_algorithm_for_on_time_jobs(j_jobs, proc_times, due_dates, instance_data, machine_id):
    """Moore's algorithm applied to the subproblem of minimizing number of tardy jobs on one machine."""
    jobs = j_jobs[:]
    if not jobs:
        return []
    
    # Step 1: Sequence them according to EDD (Earliest Due Date)
    jobs.sort(key=lambda i: due_dates[i])
    
    # Simulate execution and move the job with the largest weight causing tardiness
    current_time = 0
    schedule = []
    # For simplicity without weights in Moore, just return earliest due date if fallback needed.
    # But we want to respect weights => weighted versions exist but complex, using EDD for now
    edd_schedule = []
    current_time = 0
    for job_id in jobs:
        current_time += proc_times[job_id][machine_id]
        if current_time <= due_dates[job_id]:
            edd_schedule.append(job_id)
    
    return edd_schedule


def solve_weighted_tardy_greedy(instance_data):
    """Simple greedy approach."""
    num_jobs = instance_data["num_jobs"]
    num_machines = instance_data["num_machines"]
    jobs = list(range(num_jobs))
    
    proc_times = instance_data["jobs"]["processing_times"]
    weights = instance_data["jobs"]["weights"]
    due_dates = instance_data["jobs"]["due_dates"]
    
    # Greedily assign: prioritize jobs with higher weight/due_date or higher density
    jobs.sort(key=lambda i: weights[i] / max(due_dates[i], 1), reverse=True)
    
    schedule = {str(j): [] for j in range(num_machines)}
    completion_times = {}
    tardy_jobs = []
    on_time_jobs = []
    
    for job_id in jobs:
        # Try to assign to a machine where it can finish on time earliest
        placed = False
        best_options = []
        
        for machine_idx in range(num_machines):
            temp_schedule = schedule[str(machine_idx)][:]
            temp_schedule.append(job_id)
            # Check EDD feasibility
            temp_schedule.sort(key=lambda i: due_dates[i])
            
            current_time = 0
            is_viable = True
            for jid in temp_schedule:
                current_time += proc_times[jid][machine_idx]
                if jid == job_id:
                    if current_time > due_dates[jid]:
                        is_viable = False
                        break
                    else:
                        # Will place here
                        pass
            
            if is_viable:
                best_options.append((machine_idx, temp_schedule))
        
        # Among viable options, pick one with best insertion spot (to minimize total delay)
        if best_options:
            machine_id, new_sched = best_options[0]
            old_schedule = schedule[str(machine_id)][:]
            schedule[str(machine_id)] = new_sched
            placed = True
        else:
            tardy_jobs.append(job_id)
    
    # Now compute actual completion times and on-time jobs
    for machine_idx in range(num_machines):
        current_time = 0
        for job_id in schedule[str(machine_idx)]:
            current_time += proc_times[job_id][machine_idx]
            completion_times[job_id] = current_time
            if current_time <= due_dates[job_id]:
                on_time_jobs.append(job_id)
            else:
                tardy_jobs.append(job_id)
    
    # Calculate objective value
    actual_tardy_jobs = [i for i in range(num_jobs) if (completion_times.get(i, float('inf')) > due_dates[i])]
    objective_value = sum(weights[i] for i in actual_tardy_jobs)
    
    return {
        "objective_value": objective_value,
        "schedule": schedule,
        "completion_times": completion_times,
        "tardy_jobs": actual_tardy_jobs,
        "on_time_jobs": [i for i in range(num_jobs) if i not in actual_tardy_jobs]
    }


def main():
    parser = argparse.ArgumentParser(description="Solve Parallel Machine Scheduling Problem")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to the instance JSON file")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to save the solution JSON file")
    parser.add_argument("--time_limit", type=int, required=True, help="Time limit in seconds")
    parser.add_argument("--log_path", type=str, required=False, help="Path to log file")
    
    args = parser.parse_args()
    
    # Initialize logger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Read instance data
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)

    # Solve the instance
    start_time = time.time()
    solution = solve_instance(instance_data, args.time_limit, logger)
    end_time = time.time()

    # Write solution
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

    print(f"Solved in {end_time - start_time:.2f} seconds.")
    print(f"Objective value: {solution['objective_value']}")


if __name__ == "__main__":
    main()
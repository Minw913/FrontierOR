import argparse
import json
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from solution_logger import SolutionLogger


def solve_two_stage_stochastic_job_scheduling(instance_path, solution_path, time_limit, log_path):
    # Read the instance
    with open(instance_path, 'r') as f:
        data = json.load(f)

    n = data['n']
    m = data['m']
    num_scenarios = data['num_scenarios']
    facility_capacity_limits = data['facility_capacity_limits']
    capacity_requirements = np.array(data['capacity_requirements'])
    release_times = data['release_times']
    scenarios = data['scenarios']

    # Extract all scenario probabilities and processing times
    scenario_probabilities = []
    scenario_processing_times = []
    for s in scenarios:
        scenario_probabilities.append(s['probability'])
        scenario_processing_times.append(np.array(s['processing_times']))

    logger = SolutionLogger(log_path, sense="minimize") if log_path else None

    # Define assignment only using Gurobi first to get assignment
    model = gp.Model("Assignment-Only")
    model.setParam("Threads", 1)
    model.setParam('TimeLimit', time_limit)
    model.setParam('OutputFlag', 0)  # Suppress Gurobi output

    # Decision variables for assignment
    x = model.addVars(n, m, vtype=GRB.BINARY, name="assign")

    # Constraint: Each job is assigned to exactly one facility
    for j in range(n):
        model.addConstr(gp.quicksum(x[j, i] for i in range(m)) == 1)

    # Since no cost in the assignment stage alone, just satisfy constraints efficiently
    # We add dummy objective and focus on feasibility and speed
    model.setObjective(0, GRB.MINIMIZE)

    model.optimize()

    # Now schedule each scenario based on fixed assignment
    assignment_solution = {}

    status = model.Status
    if status in [GRB.OPTIMAL, GRB.TIME_LIMIT] and model.SolCount > 0:
        # Extract assignment
        for j in range(n):
            for i in range(m):
                if x[j, i].x > 0.5:
                    assignment_solution[str(j)] = i
                    break
    else:
        # Default assignment if Gurobi fails to find a solution
        for j in range(n):
            assignment_solution[str(j)] = 0  # Assign all jobs to first facility

    # Fixed assignment - schedule each scenario separately
    schedule_solution = {}
    expected_makespan_real = 0.0
    
    # Process each scenario
    for s_idx in range(num_scenarios):
        prob_s = scenario_probabilities[s_idx]
        processing_times_s = scenario_processing_times[s_idx]

        # Build schedule for this scenario based on assignment_solution
        schedule_solution[f"{s_idx}"] = {}
        
        # Group jobs by assigned facility
        jobs_per_facility = [[] for _ in range(m)]
        for j in range(n):
            facility_i = assignment_solution[str(j)]
            jobs_per_facility[facility_i].append(j)
        
        # Schedule each facility independently
        max_completion_in_scenario = 0
        
        for i in range(m):
            jobs_on_facility_i = jobs_per_facility[i]
            if not jobs_on_facility_i:
                continue

            # Schedule these jobs respecting resource constraints
            resource_timeline = {}
            
            # Sort jobs by their release time for basic chronological processing
            sorted_jobs = sorted(jobs_on_facility_i, key=lambda j: release_times[j])
            
            for j in sorted_jobs:
                proc_time = int(processing_times_s[i][j])
                cap_req = capacity_requirements[i][j]
                earliest_start = release_times[j]
                
                # Find earliest start time when resource is available for duration
                start_time = earliest_start
                
                # Find valid starting position to fit this job's resource requirement
                while True:
                    # Check if there is enough capacity for the entire processing period
                    can_fit = True
                    for t_offset in range(proc_time):
                        t = start_time + t_offset
                        if resource_timeline.get(t, 0) + cap_req > facility_capacity_limits[i]:
                            can_fit = False
                            start_time = t + 1  # Move past the conflicting time
                            break
                            
                    if can_fit:
                        break
                    
                # Commit to running job j on facility i from start_time to finish_time
                finish_time = start_time + proc_time
                
                # Add resource usage over the interval
                for t in range(start_time, finish_time):
                    resource_timeline[t] = resource_timeline.get(t, 0) + cap_req
                
                # Store schedule info
                schedule_solution[f"{s_idx}"][f"{j}"] = {
                    "facility": i,
                    "start": start_time,
                    "finish": finish_time
                }
                
                max_completion_in_scenario = max(max_completion_in_scenario, finish_time)
        
        scenario_makespan = max_completion_in_scenario
        expected_makespan_real += prob_s * scenario_makespan
    
    # Final solution construction
    solution = {
        "objective_value": expected_makespan_real,
        "assignment": assignment_solution,
        "schedule": schedule_solution
    }
    
    # Write the final solution to file
    with open(solution_path, 'w') as f:
        json.dump(solution, f)
    
    # Log the value
    if logger:
        logger.log(expected_makespan_real)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Solve Two-Stage Stochastic Job Scheduling Problem')
    parser.add_argument('--instance_path', required=True, help='Path to the input instance JSON file')
    parser.add_argument('--solution_path', required=True, help='Path to save the output solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds')
    parser.add_argument('--log_path', required=False, help='Path to log file for intermediate solutions')

    args = parser.parse_args()
    
    solve_two_stage_stochastic_job_scheduling(args.instance_path, args.solution_path, args.time_limit, args.log_path)
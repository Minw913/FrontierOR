import argparse
import json
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import sys
from datetime import datetime
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

    scenario_probabilities = []
    scenario_processing_times = []  # [scenario_idx][facility][job]
    for s in scenarios:
        scenario_probabilities.append(s['probability'])
        scenario_processing_times.append(np.array(s['processing_times']))
    
    scenario_probabilities = np.array(scenario_probabilities)

    logger = SolutionLogger(log_path, sense="minimize") if log_path else None

    # Start building the model
    model = gp.Model("TwoStageStochasticScheduling")

    # Set time limit
    model.Params.TimeLimit = time_limit
    model.Params.MIPGap = 0.001

    # First stage: Job assignment decisions (binary variables x[job_id, facility_id])
    x_vars = {}
    for j in range(n):
        for i in range(m):
            x_vars[(j, i)] = model.addVar(vtype=GRB.BINARY, name=f"x_{j}_{i}")
    
    # Constraint: Each job is assigned to exactly one facility
    for j in range(n):
        model.addConstr(gp.quicksum(x_vars[(j, i)] for i in range(m)) == 1)
    
    # Determine upper bound for processing time across all scenarios
    max_processing_time = 0
    for s in range(len(scenarios)):
        for i_facility in range(m):
            for j_job in range(n):
                t = scenario_processing_times[s][i_facility][j_job]
                max_processing_time = max(max_processing_time, t)
    
    T_max = sum(release_times) + n * max_processing_time  # Roughly estimated due date for planning horizon
    
    # Discretize time grid (using smaller time steps would be more accurate but expensive)
    # We'll handle the scheduling constraints for each scenario separately in our heuristic after assignment
    max_finish_time = int(T_max)
    time_steps = list(range(max_finish_time + 1))

    # Create second-stage variables for each scenario
    start_vars = {}  # [scenario][job] -> start time vars
    finish_vars = {}  # [scenario][job] -> finish time vars
    for s in range(num_scenarios):
        start_vars[s] = {}
        finish_vars[s] = {}
        for j in range(n):
            start_vars[s][j] = model.addVar(vtype=GRB.INTEGER, lb=release_times[j], ub=max_finish_time, name=f"start_{s}_{j}")
            # For finish variables, compute an upper bound
            max_proc_on_j = max([scenario_processing_times[s][i][j] for i in range(m)])
            finish_vars[s][j] = model.addVar(vtype=GRB.INTEGER, lb=release_times[j], ub=max_finish_time, name=f"finish_{s}_{j}")


    # Link assignment to finish times for each scenario
    for s in range(num_scenarios):
        for j in range(n):
            for i in range(m):
                # finish_time >= start_time + proc_time if assigned to facility i
                # This is complex - let's reconsider the approach to avoid bilinearity
                
                # Instead, we will construct schedule and enforce resource constraints for each scenario
                # after obtaining the assignment decision
                pass
    
    # Objective: Minimize expected makespan
    makespan_vars = []
    for s in range(num_scenarios):
        # Create a variable for scenario makespan
        z_s = model.addVar(vtype=GRB.CONTINUOUS, name=f"makespan_{s}")
        
        # Scenario s makespan = max_j (finish_time[j] in scenario s)
        for j in range(n):
            model.addConstr(z_s >= finish_vars[s][j])
            
        makespan_vars.append(z_s)

    expected_makespan = gp.quicksum(scenario_probabilities[s] * makespan_vars[s] for s in range(num_scenarios))
    model.setObjective(expected_makespan, GRB.MINIMIZE)

    # Optimize the first stage (assignment) for now - we'll incorporate scheduling heuristically due to complexity of resource overlapping
    model.optimize()

    # Check solution status
    if model.SolCount == 0:
        # No feasible solution found during the time limit
        print("No solution found within time limit.")
        assignment_solution = {str(j): 0 for j in range(n)}  # Default assignment (all to facility 0)
        schedule_solution = {}
        for s_idx, _ in enumerate(scenarios):
            schedule_solution[f"{s_idx}"] = {}
            for j in range(n):
                schedule_solution[f"{s_idx}"][f"{j}"] = {"facility": 0, "start": release_times[j], "finish": release_times[j]}
        final_obj_val = 9999999  
    else:
        # Extract the assignment solution
        assignment_solution = {}
        for j in range(n):
            for i in range(m):
                if x_vars[(j, i)].X > 0.5:  # If job j is assigned to facility i
                    assignment_solution[str(j)] = i
                    break
        
        # For each scenario, now we need to compute the schedules respecting the resource constraints
        schedule_solution = {}
        obj_vals_by_scenario = []
        
        for s_idx, scenario_data in enumerate(scenarios):
            processing_times_scen = scenario_processing_times[s_idx]
            schedule_solution[str(s_idx)] = {}
            scenario_start_times = []
            
            # Build job lists per facility based on assignment
            jobs_per_facility = [[] for _ in range(m)]
            for j in range(n):
                facility_i = assignment_solution[str(j)]
                proc_time_here = processing_times_scen[facility_i][j]
                jobs_per_facility[facility_i].append((j, proc_time_here))
                
            # Schedule each facility independently with resource constraints
            actual_finish_times_scenario = []
            for facility_i in range(m):
                jobs_facility = jobs_per_facility[facility_i]
                if not jobs_facility:
                    continue
                
                # Sort jobs by earliest release time (simple heuristic)
                jobs_sorted = sorted(jobs_facility, key=lambda x: release_times[x[0]])
                
                # Now schedule considering resource constraints using timeline sweep
                # For each job in this facility, determine start time obeying release times and resource limits
                current_time_resource_usage = [0] * (T_max + 86400)  # Array of resource usage over time intervals
                
                for j_info in jobs_sorted:
                    job_id = j_info[0]
                    proc_time = j_info[1]
                    
                    cap_req = capacity_requirements[facility_i][job_id]
                    
                    # Find earliest start time possible given resource availability
                    candidate_t = release_times[job_id]
                    done = False
                    max_time_check = T_max
                    
                    while candidate_t <= max_time_check and not done:
                        feasible = True
                        # Check if we can allocate resources for job duration starting at candidate_t
                        for t in range(candidate_t, candidate_t + proc_time):
                            if t > len(current_time_resource_usage) - 1:
                                # Extend array to accommodate longer timeline
                                current_time_resource_usage.extend([0] * (t - len(current_time_resource_usage) + 10000))
                            if current_time_resource_usage[t] + cap_req > facility_capacity_limits[facility_i]:
                                feasible = False
                                break
                        
                        if feasible:
                            done = True
                            
                            # Store the schedule result
                            start_t = candidate_t
                            end_t = start_t + proc_time
                            
                            schedule_solution[f"{s_idx}"][f"{job_id}"] = {
                                "facility": facility_i,
                                "start": start_t,
                                "finish": end_t
                            }
                            
                            # Update resource usage for the interval
                            for t in range(start_t, end_t):
                                current_time_resource_usage[t] += cap_req
                                
                            actual_finish_times_scenario.append(end_t)
                        
                        candidate_t += 1
                        
                    if not done:
                        # Assign start and end times without considering resources - fallback
                        start_t = max(candidate_t - 1, release_times[job_id])
                        end_t = start_t + proc_time
                        schedule_solution[f"{s_idx}"][f"{job_id}"] = {
                            "facility": facility_i,
                            "start": start_t,
                            "finish": end_t
                        }
                        actual_finish_times_scenario.append(end_t)
                        
                    
            # Calculate makespan for this scenario
            scenario_makespan = max(actual_finish_times_scenario) if actual_finish_times_scenario else 0
            obj_vals_by_scenario.append(scenario_makespan)
            
        # Compute actual expected makespan
        expected_makespan_real = sum(
            scenario_probabilities[s_idx] * obj_vals_by_scenario[s_idx] 
            for s_idx in range(num_scenarios)
        )
        final_obj_val = expected_makespan_real
            
    # Prepare final solution
    solution = {
        "objective_value": final_obj_val,
        "assignment": assignment_solution,
        "schedule": schedule_solution
    }

    # Write solution
    with open(solution_path, 'w') as f:
        json.dump(solution, f)

    if logger:
        logger.log(final_obj_val)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Solve Two-Stage Stochastic Job Scheduling Problem')
    parser.add_argument('--instance_path', required=True, help='Path to the input instance JSON file')
    parser.add_argument('--solution_path', required=True, help='Path to save the output solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds')
    parser.add_argument('--log_path', required=False, help='Path to log file for intermediate solutions')

    args = parser.parse_args()

    solve_two_stage_stochastic_job_scheduling(args.instance_path, args.solution_path, args.time_limit, args.log_path)
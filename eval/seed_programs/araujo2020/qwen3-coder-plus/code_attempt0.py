import argparse
import json
import math
from collections import defaultdict
import gurobipy as gp
from gurobipy import GRB


def compute_forward_paths(jobs_by_project, job_modes, critical_precedences):
    """
    Compute forward earliest start times for each job using the critical 
    path method (ignoring resource constraints).
    """
    # Calculate the shortest duration for each job across all modes
    min_durations = {}
    for job_data in job_modes.values():
        job_id = job_data["job_id"]
        min_duration = float('inf')
        for mode in job_data["modes"]:
            if mode["duration"] < min_duration:
                min_duration = mode["duration"]
        if min_duration == float('inf'):  # For jobs without modes
            min_duration = 0
        min_durations[job_id] = min_duration

    # Initialize earliest start times
    est = {}
    for job_data in job_modes.values():
        job_id = job_data["job_id"]
        est[job_id] = 0

    changed = True
    while changed:
        changed = False
        for job_data in job_modes.values():
            job_id = job_data["job_id"]
            for succ_id in job_data.get("successors", []):
                new_earliest_start = max(est[succ_id], est[job_id] + min_durations[job_id])
                if new_earliest_start > est[succ_id]:
                    est[succ_id] = new_earliest_start
                    changed = True

    return est


def compute_backward_paths(project_end_times, jobs_by_project, job_modes, critical_precedences):
    """
    Compute late finish / start times backward from project end times.
    """
    # First map all succ -> pred relationships
    predecessors = {}
    for job_data in job_modes.values():
        job_id = job_data["job_id"]
        for succ_id in job_data.get("successors", []):
            if succ_id not in predecessors:
                predecessors[succ_id] = []
            predecessors[succ_id].append(job_id)

    # Find end jobs for the backwards pass
    end_jobs = set()
    for proj_data in jobs_by_project:
        project_id = proj_data["project_id"]
        end_job_id = proj_data["artificial_sink_job_id"]
        end_jobs.add(end_job_id)

    # Calculate the shortest duration for each job across all modes
    min_durations = {}
    for job_data in job_modes.values():
        job_id = job_data["job_id"]
        min_duration = float('inf')
        for mode in job_data["modes"]:
            if mode["duration"] < min_duration:
                min_duration = mode["duration"]
        if min_duration == float('inf'):  # For jobs without modes
            min_duration = 0
        min_durations[job_id] = min_duration

    # Initialize latest start and finish times
    lft = {}
    lst = {}

    # Start with project end times
    for job_id in end_jobs:
        lft[job_id] = project_end_times[job_id]

    changed = True
    while changed:
        changed = False
        # Process jobs backwards
        sorted_by_est = sorted(jobs_by_project, key=lambda x: x['release_date'])  # Simplify assumption
        for job_data in job_modes.values():
            job_id = job_data["job_id"]
            
            # Initialize LFT if not already initialized
            if job_id not in lft:
                continue

            # Calculate earliest start for this job based on predecessors
            # For now: get max finish of predecessors
            max_pred_finish = 0
            for pred_id in predecessors.get(job_id, []):
                f_time = lft.get(pred_id, 0)
                if f_time > max_pred_finish:
                    max_pred_finish = f_time
            lst[job_id] = max(0, lft[job_id] - min_durations[job_id])

            # Update predecessors with latest finish times based on current job's latest start
            for pred_id in predecessors.get(job_id, []):
                # The predecessor has to finish by the current job's start time
                latest_pred_finish_based_on_curr = lst.get(job_id, float('inf'))
                if pred_id not in lft or lft[pred_id] > latest_pred_finish_based_on_curr:
                    # Update only if more restrictive
                    old_finish = lft.get(pred_id, float('inf'))
                    lft[pred_id] = min(old_finish, latest_pred_finish_based_on_curr)
                    changed = True or (old_finish != min(old_finish, latest_pred_finish_based_on_curr))

    return lst


def compute_critical_path(project_info, job_modes):
    """
    Compute the critical path for each project to determine the project's duration if unlimited resources.
    """
    def calculate_project_duration(proj_project_id, jobs_for_project, job_modes):
        job_dict = {j['job_id']: j for j in job_modes.values() if j['project_id'] == proj_project_id}
        
        # Build graph for the project
        est = {} # Earliest start time
        for jid, jdata in job_dict.items():
            est[jid] = 0

        # For this computation, assume minimum duration across modes
        job_min_duration = {}
        for jid, jdata in job_dict.items():
            min_dur = float('inf') if jdata["modes"] else 0
            for m in jdata["modes"]:
                min_dur = min(min_dur, m["duration"])
            job_min_duration[jid] = min_dur
        
        updated = True
        while updated:
            updated = False
            for jid, jdata in job_dict.items():
                for succ_id in jdata.get("successors", []):
                    if succ_id in job_dict:  # Only project local successors
                        new_early_start = est[jid] + job_min_duration[jid]
                        if new_early_start > est[succ_id]:
                            est[succ_id] = new_early_start
                            updated = True
                        
        # Now do a backward pass to find latest finish time
        lft = {}
        end_job_id = jobs_for_project["artificial_sink_job_id"]
        lft[end_job_id] = est[end_job_id]  # Critical project duration
        
        # Reverse precedences
        succ_to_preds = defaultdict(list)
        for jid, jdata in job_dict.items():
            for succ_id in jdata.get("successors", []):
                if succ_id in job_dict:
                    succ_to_preds[succ_id].append(jid)
                    
        # Backward pass
        sorted_nodes = sorted(est.keys(), key=lambda x: est[x], reverse=True)
        for jid in sorted_nodes:
            for pred_id in succ_to_preds[jid]:
                pred_lf = lft[jid] - job_min_duration[pred_id]
                if pred_id not in lft or pred_lf < lft[pred_id]:
                    lft[pred_id] = pred_lf
                    
        return lft[end_job_id]

    critical_path_duration_per_project = {}
    for proj in project_info:
        project_id = proj["project_id"]
        critical_path_duration_per_project[project_id] = calculate_project_duration(
            project_id, proj, job_modes
        )
    
    return critical_path_duration_per_project


def create_model(instance_data, time_limit):
    """
    Create and solve the MIP model optimizing the Multi-Mode Resource-Constrained Project Scheduling Problem.
    """
    projects = instance_data["projects"]
    resources = instance_data["resources"]
    jobs = instance_data["jobs"]
    renewables = resources["renewable"]
    nonrenewables = resources["nonrenewable"]

    # Map structures for easier access
    job_modes = {j["job_id"]: j for j in jobs}
    project_info = {p["project_id"]: p for p in projects}
    renewable_capacities = [r["capacity"] for r in renewables]
    nonrenewable_capacities = [r["capacity"] for r in nonrenewables]

    # Compute critical paths for all projects
    critical_paths = compute_critical_path(projects, job_modes)
    
    # Calculate max allowable delay
    critical_lower_bounds = {
        p["project_id"]: p["release_date"] + critical_paths.get(p["project_id"], 0)
        for p in projects
    }
    upper_bound_completions = {p["project_id"]: critical_lower_bounds[p["project_id"]] for p in projects}
    total_max_allowable_delay = sum(
        max(upper_bound_completion - critical_lower_bounds[p_id], 0)
        for p_id, upper_bound_completion in upper_bound_completions.items()
    )
    total_horizon = max(critical_lower_bounds[p["project_id"]] + critical_paths.get(p["project_id"], 0) for p in projects)
    
    # Determine horizon as sum of release date, critical path duration and max allowable delay
    HORIZON = total_horizon + total_max_allowable_delay

    # Compute earliest start times using forward propagation (Critical Path Method)
    early_starts = compute_forward_paths(project_info, job_modes, None)
    # Adjust early starts based on release dates
    for job_id, job_data in job_modes.items():
        project_id = job_data["project_id"]
        release_date = project_info[project_id]["release_date"]
        early_starts[job_id] = max(early_starts[job_id], release_date)

    # Now we need latest finish times and start times based on LFT and total horizon
    end_jobs_lft = {}
    for proj_data in projects:
        end_job_id = proj_data["artificial_sink_job_id"]
        # End time based on release + critical + delay budget: 
        end_jobs_lft[end_job_id] = critical_lower_bounds[proj_data["project_id"]] + total_max_allowable_delay

    # Use these end times for computing late start times
    late_starts = compute_backward_paths(end_jobs_lft, projects, job_modes, None)
    
    # Final EST and LST
    job_time_windows = {}
    for job_id in job_modes:
        job_time_windows[job_id] = (
            early_starts[job_id],
            max(early_starts[job_id], late_starts.get(job_id, HORIZON))
        )

    # Initialize Gurobi model
    model = gp.Model("MMRCPSP")
    model.setParam('TimeLimit', time_limit)
    model.setParam('MIPFocus', 1)  # Feasibility first
    model.setParam('Threads', 1)

    # Define variables
    # Binary variables x[j, m, t] where j=job, m=mode, t=time
    x = {}
    start_times = {}
    for job_id, job_data in job_modes.items():
        # Determine the relevant modes for the job
        start_times[job_id] = model.addVar(vtype=GRB.INTEGER, name=f"start_{job_id}")
        
        es, ls = job_time_windows[job_id]
        for t in range(es, min(ls, HORIZON) + 1):
            for mode_idx, mode in enumerate(job_data["modes"]):
                # Add variable if it fits in the valid time window
                dur = mode["duration"]
                if t + dur <= HORIZON:
                    x[(job_id, mode_idx, t)] = model.addVar(vtype=GRB.BINARY, name=f"x_{job_id}_{mode_idx}_{t}")

    # Variable for project completion times
    comp_times = {}
    for proj in projects:
        end_job = proj["artificial_sink_job_id"]
        comp_times[proj["project_id"]] = model.addVar(vtype=GRB.CONTINUOUS, name=f"comp_{proj['project_id']}")

    # Makespan (for tie-breaking)
    makespan = model.addVar(vtype=GRB.INTEGER, name="makespan")

    # Constraint 1: Each job must be assigned exactly one mode and one start time
    for job_id, job_data in job_modes.items():
        es, ls = job_time_windows[job_id]
        model.addConstr(
            gp.quicksum(x[(job_id, mode_idx, t)] 
                       for mode_idx, mode in enumerate(job_data["modes"]) 
                       for t in range(es, min(ls, HORIZON) + 1) 
                       if t + mode["duration"] <= HORIZON) == 1,
            name=f"one_assignment_{job_id}"
        )

    # Constraint 2: Define relationship between assignment decisions and start times
    for job_id, job_data in job_modes.items():
        es, ls = job_time_windows[job_id]
        expr_sum = gp.LinExpr()
        for mode_idx, mode in enumerate(job_data["modes"]):
            for t in range(es, min(ls, HORIZON) + 1):
                if t + mode["duration"] <= HORIZON:
                    expr_sum.addTerms(t, x[(job_id, mode_idx, t)])
        model.addConstr(start_times[job_id] == expr_sum, name=f"start_def_{job_id}")

    # Constraint 3: Precedence relationships
    for job_id, job_data in job_modes.items():
        for successor_id in job_data["successors"]:
            # Sum over all possible assignments of the job
            start_of_successor_expr = gp.LinExpr()
            for mode_idx, mode in enumerate(job_modes[successor_id]["modes"]):
                es_s, ls_s = job_time_windows[successor_id]
                for t in range(es_s, min(ls_s, HORIZON) + 1):
                    if t + mode["duration"] <= HORIZON and (successor_id, mode_idx, t) in x:
                        start_of_successor_expr.addTerms(t, x[(successor_id, mode_idx, t)])

            finish_expr = gp.LinExpr()
            for mode_idx, mode in enumerate(job_data["modes"]):
                es_j, ls_j = job_time_windows[job_id]
                for t in range(es_j, min(ls_j, HORIZON) + 1):
                    if t + mode["duration"] <= HORIZON and (job_id, mode_idx, t) in x:
                        finish_expr.addTerms(t + mode["duration"], x[(job_id, mode_idx, t)])

            model.addConstr(
                # This is equivalent to: start_time[successor] >= job's finish time
                # So, sum of (t * x[jobsucc, mode, t]) >= sum of ((t+duration) * x[j, mode, t])
                start_of_successor_expr >= finish_expr,
                name=f"prec_{job_id}_{successor_id}"
            )

    # Constraint 5 & 6: Renewable resource constraints (cumulative constraints)
    num_renewables = len(renewable_capacities)
    for r_idx, cap in enumerate(renewable_capacities):
        for t in range(HORIZON + 1):
            resource_usage = gp.LinExpr()
            for job_id, job_data in job_modes.items():
                es, ls = job_time_windows[job_id]
                
                for mode_idx, mode in enumerate(job_data["modes"]):
                    renew_cons = mode["renewable_consumption"][r_idx] if r_idx < len(mode["renewable_consumption"]) else 0
                    for s in range(max(0, t - mode["duration"] + 1), min(t + 1, HORIZON - mode["duration"] + 1)):
                        if s >= es and s <= ls and ((job_id, mode_idx, s) in x):
                            resource_usage.addTerms(renew_cons, x[(job_id, mode_idx, s)])
            
            model.addConstr(resource_usage <= cap, name=f"renew_r{r_idx}_t{t}")

    # Constraint 7: Non-renewable resource constraints
    num_nonrenewables = len(nonrenewable_capacities)
    for nr_idx, ncap in enumerate(nonrenewable_capacities):
        total_usage = gp.LinExpr()
        for job_id, job_data in job_modes.items():
            es, ls = job_time_windows[job_id]
            for mode_idx, mode in enumerate(job_data["modes"]):
                nonrenew_cons = mode["nonrenewable_consumption"][nr_idx] if nr_idx < len(mode["nonrenewable_consumption"]) else 0
                for t in range(es, min(ls, HORIZON) + 1):
                    if t + mode["duration"] <= HORIZON and (job_id, mode_idx, t) in x:
                        total_usage.addTerms(nonrenew_cons, x[(job_id, mode_idx, t)])
        
        model.addConstr(total_usage <= ncap, name=f"nonrenew_nr{nr_idx}")

    # Define completion time of projects
    for proj in projects:
        end_job_id = proj["artificial_sink_job_id"]
        start_expr = gp.LinExpr()
        # The end job has 0 duration, so its start time is the project completion time
        es, ls = job_time_windows[end_job_id]
        for mode_idx, mode in enumerate(job_modes[end_job_id]["modes"]):
            for t in range(es, min(ls, HORIZON) + 1):
                if t + mode["duration"] <= HORIZON and (end_job_id, mode_idx, t) in x:
                    # Since duration is 0, it finishes at the same time it starts
                    start_expr.addTerms(t, x[(end_job_id, mode_idx, t)])

        model.addConstr(comp_times[proj["project_id"]] == start_expr, name=f"proj_comp_{proj['project_id']}")
    
    # Define makespan
    for proj in projects:
        model.addConstr(makespan >= comp_times[proj["project_id"]])

    # Objective function: Minimize total project delays + makespan tie breaker
    total_delay = gp.LinExpr()
    for proj in projects:
        # Actual completion - minimum required finish (release date + critical path)
        actual_completion = comp_times[proj["project_id"]]
        min_req_completion = critical_lower_bounds[proj["project_id"]]
        total_delay.add(actual_completion - min_req_completion)

    # Add a small penalty for makespan to break ties
    TIEBREAK_COEF = 0.0001
    model.setObjective(total_delay + TIEBREAK_COEF * makespan, GRB.MINIMIZE)

    model.update()

    return model, x, start_times, comp_times, makespan, job_modes, projects, critical_lower_bounds


def solve_instance(instance_data, time_limit):
    model, x, start_times_var, comp_times_var, makespan_var, job_modes, projects, critical_lower_bounds = create_model(instance_data, time_limit)
    
    # Optimize
    model.optimize()

    if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        # Extract solution
        schedule = []
        
        for job_id, job_data in job_modes.items():
            assigned = False
            for mode_idx, mode in enumerate(job_data["modes"]):
                for t_val in range(len([v for v in x if v[0] == job_id and v[1] == mode_idx])):
                    if (job_id, mode_idx, t_val) in x:
                        var = x[(job_id, mode_idx, t_val)]
                        if var.X > 0.9:  # Account for numerical tolerance
                            schedule.append({
                                "job_id": job_id,
                                "mode_id": mode_idx,
                                "start_time": t_val
                            })
                            assigned = True
                            break
                if assigned:
                    break
        
        # If schedule wasn't fully built from x variables, extract start times
        if not schedule:
            for job_id in start_times_var:
                start_time = int(round(start_times_var[job_id].X))
                # We also need to determine which mode was selected for each job.
                # For this we go back to the x variables
                job_mode_found = False
                for mode_idx, mode in enumerate(job_modes[job_id]["modes"]):
                    es, ls = 0, 0
                    # Approximate these since we don't have them easily accessible here
                    # but we know they exist from the modeling step
                    found_at_time = False
                    for t in range(HORIZON + 1):
                        if (job_id, mode_idx, t) in x:
                            if abs(x[(job_id, mode_idx, t)].X - 1.0) < 0.001:
                                schedule.append({
                                    "job_id": job_id,
                                    "mode_id": mode_idx,
                                    "start_time": t
                                })
                                found_at_time = True
                                break
                    if found_at_time:
                        job_mode_found = True
                        break
                assert job_mode_found, f"No mode found for job {job_id}"
        
        # Compute objective values
        project_completions = {}
        for proj in projects:
            project_id = proj["project_id"]
            end_job_id = proj["artificial_sink_job_id"]
            project_completions[project_id] = round(comp_times_var[project_id].X)

        total_delay = 0
        for proj in projects:
            project_id = proj["project_id"]
            min_req_completion = critical_lower_bounds[project_id]
            actual_completion = project_completions[project_id]
            delay = max(0, actual_completion - min_req_completion)
            total_delay += delay
        
        makespan = round(makespan_var.X) if makespan_var else max(project_completions.values())
        
        solution = {
            "objective_value": total_delay,
            "schedule": schedule,
            "makespan": makespan
        }
        
        return solution
    
    else:
        # No solution found within time limits
        return {
            "objective_value": float('inf'),
            "schedule": [],
            "makespan": 0
        }


def main():
    parser = argparse.ArgumentParser(description="Solve Multi-Mode Resource Constrained Project Scheduling Problem")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to the input JSON instance file")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to save the output JSON solution")
    parser.add_argument("--time_limit", type=int, required=True, help="Time limit in seconds")
    parser.add_argument("--log_path", type=str, help="Optional path for logging intermediate solutions")
    args = parser.parse_args()

    # Load instance
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)

    # Initialize logger if provided
    logger = None
    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize")

    # Solve the instance
    solution = solve_instance(instance_data, args.time_limit)
    
    # Log if applicable
    if logger and solution["objective_value"] != float('inf'):
        logger.log(solution["objective_value"])

    # Write solution to file
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()
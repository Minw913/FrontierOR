import argparse
import json
import time
import sys
import math
from solution_logger import SolutionLogger

def create_fallback_assignment(n, m, release_times, mean_processing_times, capacity_requirements, facility_capacity_limits):
    """
    Creates a feasible job-to-facility assignment using a greedy approach.
    Each job is assigned to the facility that minimizes (release_time + mean_processing_time) while respecting capacity constraints.
    """
    assignment = {}
    for j in range(n):
        best_f = None
        best_score = float('inf')
        for f in range(m):
            if capacity_requirements[f][j] > facility_capacity_limits[f]:
                continue
            score = release_times[j] + mean_processing_times[f][j]
            if score < best_score:
                best_score = score
                best_f = f
        if best_f is None:
            for f in range(m):
                if capacity_requirements[f][j] <= facility_capacity_limits[f]:
                    best_f = f
                    break
        assignment[j] = best_f
    return assignment

def compute_expected_makespan(assignment, scenarios, release_times, capacity_requirements, facility_capacity_limits):
    """
    Computes the expected makespan for a given job assignment across all scenarios.
    For each scenario, schedules jobs on each facility respecting release times and resource constraints.
    """
    n = len(assignment)
    m = len(facility_capacity_limits)
    expected_makespan = 0.0
    
    for scenario in scenarios:
        scenario_makespan = 0.0
        jobs_by_facility = {f: [] for f in range(m)}
        
        for j in range(n):
            f = assignment[j]
            jobs_by_facility[f].append(j)
            
        for f in range(m):
            jobs_f = jobs_by_facility[f]
            if not jobs_f:
                continue
                
            jobs_f_sorted = sorted(jobs_f, key=lambda j: (release_times[j], -capacity_requirements[f][j]))
            timeline = [(0, float('inf'), 0)]
            facility_makespan = 0
            
            for j in jobs_f_sorted:
                r_j = release_times[j]
                p_j = scenario['processing_times'][f][j]
                c_j = capacity_requirements[f][j]
                candidate_start = r_j
                
                while True:
                    found_violation = False
                    for seg_start, seg_end, usage in timeline:
                        if seg_end <= candidate_start:
                            continue
                        if seg_start >= candidate_start + p_j:
                            break
                        overlap_start = max(seg_start, candidate_start)
                        overlap_end = min(seg_end, candidate_start + p_j)
                        if overlap_start < overlap_end and usage + c_j > facility_capacity_limits[f]:
                            candidate_start = seg_end
                            found_violation = True
                            break
                    if not found_violation:
                        break
                
                finish_j = candidate_start + p_j
                if finish_j > facility_makespan:
                    facility_makespan = finish_j
                    
                new_timeline = []
                for seg in timeline:
                    s, e, u = seg
                    if e <= candidate_start:
                        new_timeline.append(seg)
                    elif s >= candidate_start + p_j:
                        new_timeline.append(seg)
                    else:
                        if s < candidate_start:
                            new_timeline.append((s, candidate_start, u))
                        mid_start = max(s, candidate_start)
                        mid_end = min(e, candidate_start + p_j)
                        new_timeline.append((mid_start, mid_end, u + c_j))
                        if e > candidate_start + p_j:
                            new_timeline.append((candidate_start + p_j, e, u))
                timeline = new_timeline
                
            if facility_makespan > scenario_makespan:
                scenario_makespan = facility_makespan
                
        expected_makespan += scenario['probability'] * scenario_makespan
        
    return expected_makespan

def compute_detailed_schedule(assignment, scenarios, release_times, capacity_requirements, facility_capacity_limits):
    """
    Computes detailed schedules for all jobs under each scenario for a given assignment.
    """
    n = len(assignment)
    m = len(facility_capacity_limits)
    detailed_schedules = {}
    
    for s_idx, scenario in enumerate(scenarios):
        scenario_schedule = {}
        jobs_by_facility = {f: [] for f in range(m)}
        
        for j in range(n):
            f = assignment[j]
            jobs_by_facility[f].append(j)
            
        for f in range(m):
            jobs_f = jobs_by_facility[f]
            if not jobs_f:
                continue
                
            jobs_f_sorted = sorted(jobs_f, key=lambda j: (release_times[j], -capacity_requirements[f][j]))
            timeline = [(0, float('inf'), 0)]
            
            for j in jobs_f_sorted:
                r_j = release_times[j]
                p_j = scenario['processing_times'][f][j]
                c_j = capacity_requirements[f][j]
                candidate_start = r_j
                
                while True:
                    found_violation = False
                    for seg in timeline:
                        seg_start, seg_end, usage = seg
                        if seg_end <= candidate_start:
                            continue
                        if seg_start >= candidate_start + p_j:
                            break
                        overlap_start = max(seg_start, candidate_start)
                        overlap_end = min(seg_end, candidate_start + p_j)
                        if overlap_start < overlap_end and usage + c_j > facility_capacity_limits[f]:
                            candidate_start = seg_end
                            found_violation = True
                            break
                    if not found_violation:
                        break
                
                finish_j = candidate_start + p_j
                new_timeline = []
                for seg in timeline:
                    s, e, u = seg
                    if e <= candidate_start:
                        new_timeline.append(seg)
                    elif s >= candidate_start + p_j:
                        new_timeline.append(seg)
                    else:
                        if s < candidate_start:
                            new_timeline.append((s, candidate_start, u))
                        mid_start = max(s, candidate_start)
                        mid_end = min(e, candidate_start + p_j)
                        new_timeline.append((mid_start, mid_end, u + c_j))
                        if e > candidate_start + p_j:
                            new_timeline.append((candidate_start + p_j, e, u))
                timeline = new_timeline
                
                scenario_schedule[j] = {
                    "facility": f,
                    "start": candidate_start,
                    "finish": finish_j
                }
                
        detailed_schedules[str(s_idx)] = scenario_schedule
        
    return detailed_schedules

def main(instance_path, solution_path, time_limit, log_path=None):
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    start_time = time.time()
    
    with open(instance_path, 'r') as f:
        data = json.load(f)
    
    n = data['n']
    m = data['m']
    facility_capacity_limits = data['facility_capacity_limits']
    capacity_requirements = data['capacity_requirements']
    release_times = data['release_times']
    mean_processing_times = data['mean_processing_times']
    scenarios = data['scenarios']
    
    # Create fallback assignment
    assignment = create_fallback_assignment(
        n, m, release_times, mean_processing_times, 
        capacity_requirements, facility_capacity_limits
    )
    
    # Compute expected makespan and detailed schedules
    expected_makespan = compute_expected_makespan(
        assignment, scenarios, release_times, 
        capacity_requirements, facility_capacity_limits
    )
    
    detailed_schedules = compute_detailed_schedule(
        assignment, scenarios, release_times, 
        capacity_requirements, facility_capacity_limits
    )
    
    # Log solution if logger is available
    if logger:
        logger.log(expected_makespan)
    
    # Prepare output
    output_assignment = {str(j): assignment[j] for j in range(n)}
    output_schedule = {}
    
    for s_idx in range(len(scenarios)):
        scenario_key = str(s_idx)
        scenario_schedule = detailed_schedules[scenario_key]
        output_schedule[scenario_key] = {str(j): scenario_schedule[j] for j in scenario_schedule}
    
    solution = {
        "objective_value": expected_makespan,
        "assignment": output_assignment,
        "schedule": output_schedule
    }
    
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    main(args.instance_path, args.solution_path, args.time_limit, args.log_path)
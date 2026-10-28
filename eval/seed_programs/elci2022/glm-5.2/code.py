import argparse
import json
import time
import random
import math
from solution_logger import SolutionLogger

def improve_schedule(jobs, capacity, starts):
    changed = True
    while changed:
        changed = False
        for j in jobs:
            idx, r, p, c = j
            current_start = starts[idx]
            if current_start <= r:
                continue
            
            events = {}
            for k in jobs:
                k_idx, k_r, k_p, k_c = k
                if k_idx == idx:
                    continue
                k_s = starts[k_idx]
                if k_s not in events:
                    events[k_s] = 0
                events[k_s] += k_c
                if k_s + k_p not in events:
                    events[k_s + k_p] = 0
                events[k_s + k_p] -= k_c
                
            sorted_times = sorted(events.keys())
            intervals = []
            prev_t = None
            prev_cap = 0
            for t_event in sorted_times:
                if prev_t is not None:
                    intervals.append((prev_t, t_event, prev_cap))
                prev_t = t_event
                prev_cap += events[t_event]
                
            candidate_starts = {r}
            for k in jobs:
                k_idx, k_r, k_p, k_c = k
                if k_idx == idx:
                    continue
                k_s = starts[k_idx]
                candidate_starts.add(k_s + k_p - p + 1)
                candidate_starts.add(k_s - p + 1)
            
            candidate_starts = sorted([t for t in candidate_starts if r <= t < current_start])
            
            for t in candidate_starts:
                feasible = True
                for start_i, end_i, cap_i in intervals:
                    overlap_start = max(start_i, t)
                    overlap_end = min(end_i - 1, t + p - 1)
                    if overlap_start <= overlap_end:
                        if cap_i + c > capacity:
                            feasible = False
                            break
                if feasible:
                    starts[idx] = t
                    changed = True
                    break
    return starts

def schedule_jobs(jobs, capacity):
    if not jobs:
        return 0, {}
    
    best_makespan = float('inf')
    best_starts = {}
    
    rules = [
        lambda j: -j[2] * j[3],  # energy desc
        lambda j: j[1],          # release asc
        lambda j: j[0]           # idx asc
    ]
    
    for rule in rules:
        t = min(j[1] for j in jobs)
        available = list(jobs)
        running = []
        starts = {}
        makespan = 0
        
        while available or running:
            available.sort(key=rule)
            cap_left = capacity - sum(r[1][3] for r in running)
            
            for j in list(available):
                if j[1] <= t and j[3] <= cap_left:
                    starts[j[0]] = t
                    running.append((t + j[2], j))
                    cap_left -= j[3]
                    available.remove(j)
                    makespan = max(makespan, t + j[2])
            
            if not running:
                if available:
                    t = min(j[1] for j in available)
                else:
                    break
            else:
                next_t = min(r[0] for r in running)
                for r in list(running):
                    if r[0] == next_t:
                        running.remove(r)
                t = next_t
                
        starts = improve_schedule(jobs, capacity, starts)
        makespan = max(starts[j[0]] + j[2] for j in jobs)
        
        if makespan < best_makespan:
            best_makespan = makespan
            best_starts = starts
            
    return best_makespan, best_starts

def evaluate_assignment(assignment, data):
    expected_makespan = 0.0
    schedules = {}
    for s_idx, scenario in enumerate(data['scenarios']):
        makespan_s = 0
        schedules[s_idx] = {}
        for i in range(data['m']):
            jobs_on_i = []
            for j in range(data['n']):
                if assignment[j] == i:
                    r = data['release_times'][j]
                    p = scenario['processing_times'][i][j]
                    c = data['capacity_requirements'][i][j]
                    jobs_on_i.append((j, r, p, c))
            
            ms, starts = schedule_jobs(jobs_on_i, data['facility_capacity_limits'][i])
            makespan_s = max(makespan_s, ms)
            for j, start in starts.items():
                p = scenario['processing_times'][i][j]
                schedules[s_idx][j] = {
                    "facility": i,
                    "start": start,
                    "finish": start + p
                }
        expected_makespan += scenario['probability'] * makespan_s
    return expected_makespan, schedules

def format_solution(obj, assignment, schedules, data):
    sol = {
        "objective_value": obj,
        "assignment": {str(j): assignment[j] for j in range(data['n'])},
        "schedule": {}
    }
    for s_idx in range(len(data['scenarios'])):
        sol["schedule"][str(s_idx)] = {}
        for j in range(data['n']):
            if j in schedules[s_idx]:
                sol["schedule"][str(s_idx)][str(j)] = schedules[s_idx][j]
    return sol

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    n = data['n']
    m = data['m']
    
    current_assignment = [0] * n
    for j in range(n):
        best_fac = 0
        min_cap = float('inf')
        for i in range(m):
            if data['capacity_requirements'][i][j] < min_cap:
                min_cap = data['capacity_requirements'][i][j]
                best_fac = i
        current_assignment[j] = best_fac
        
    current_obj, current_schedules = evaluate_assignment(current_assignment, data)
    best_assignment = list(current_assignment)
    best_obj = current_obj
    best_schedules = current_schedules
    
    if logger:
        logger.log_solution(best_obj, format_solution(best_obj, best_assignment, best_schedules, data))
    
    start_time = time.time()
    T = 1.0
    cooling_rate = 0.99
    
    random.seed(0)
    
    while time.time() - start_time < max(1, args.time_limit - 1):
        new_assignment = list(current_assignment)
        j = random.randint(0, n - 1)
        new_assignment[j] = random.randint(0, m - 1)
        
        new_obj, new_schedules = evaluate_assignment(new_assignment, data)
        
        if new_obj < current_obj or random.random() < math.exp((current_obj - new_obj) / T):
            current_assignment = new_assignment
            current_obj = new_obj
            current_schedules = new_schedules
            
            if new_obj < best_obj:
                best_obj = new_obj
                best_assignment = list(new_assignment)
                best_schedules = new_schedules
                if logger:
                    logger.log_solution(best_obj, format_solution(best_obj, best_assignment, best_schedules, data))
        
        T *= cooling_rate
        if T < 0.01:
            T = 1.0
            
    final_solution = format_solution(best_obj, best_assignment, best_schedules, data)
    with open(args.solution_path, 'w') as f:
        json.dump(final_solution, f, indent=4)

if __name__ == "__main__":
    main()
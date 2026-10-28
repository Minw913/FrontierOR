import argparse
import json
import math
import heapq
import time
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser(description='Schedule jobs to minimize normalized sum of squared deviations.')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the input instance JSON file.')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to the output solution JSON file.')
    parser.add_argument('--time_limit', type=int, required=True, help='Maximum runtime in seconds.')
    parser.add_argument('--log_path', type=str, default=None, help='Path to the log file for intermediate solutions.')
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    num_machines = instance['num_machines']
    num_jobs = instance['num_jobs']
    processing_times = instance['processing_times']

    total_start_time = time.time()
    T = sum(processing_times)
    A = T / num_machines

    heap_greedy = []
    for i in range(num_machines):
        heapq.heappush(heap_greedy, (0, i))
    
    machine_loads = [0] * num_machines
    job_machine = [-1] * num_jobs

    for j in range(num_jobs):
        p = processing_times[j]
        load, machine_index = heapq.heappop(heap_greedy)
        new_load = load + p
        machine_loads[machine_index] = new_load
        job_machine[j] = machine_index
        heapq.heappush(heap_greedy, (new_load, machine_index))
    
    S = sum(load * load for load in machine_loads)
    T_sq_over_m = T * T / num_machines
    total_sq_dev = max(0, S - T_sq_over_m)
    normalized_dev = math.sqrt(total_sq_dev) / A

    logger = None
    if args.log_path:
        logger = SolutionLogger(args.log_path, sense="minimize")
        logger.log(normalized_dev)

    time_used_so_far = time.time() - total_start_time
    time_remaining = args.time_limit - time_used_so_far

    if time_remaining > 0:
        heap_local = []
        for i in range(num_machines):
            heapq.heappush(heap_local, (machine_loads[i], i))
        
        improved = True
        local_start_time = time.time()
        while improved and time_remaining > 0:
            improved = False
            for j in range(num_jobs):
                current_time = time.time()
                elapsed_local = current_time - local_start_time
                if elapsed_local >= time_remaining:
                    break
                
                p_j = processing_times[j]
                current_machine = job_machine[j]
                current_load = machine_loads[current_machine]
                
                candidate = None
                while heap_local and elapsed_local < time_remaining:
                    load_val, machine_index = heapq.heappop(heap_local)
                    if abs(load_val - machine_loads[machine_index]) < 1e-5:
                        if machine_index != current_machine:
                            candidate = (load_val, machine_index)
                            break
                        else:
                            heapq.heappush(heap_local, (load_val, machine_index))
                    current_time_inner = time.time()
                    elapsed_local = current_time_inner - local_start_time
                    if elapsed_local >= time_remaining:
                        break
                
                if candidate is None:
                    break
                
                candidate_load, candidate_machine = candidate
                if candidate_load + p_j < current_load:
                    delta_S = 2 * p_j * (candidate_load - current_load + p_j)
                    S += delta_S
                    machine_loads[current_machine] = current_load - p_j
                    machine_loads[candidate_machine] = candidate_load + p_j
                    job_machine[j] = candidate_machine
                    
                    heapq.heappush(heap_local, (machine_loads[current_machine], current_machine))
                    heapq.heappush(heap_local, (machine_loads[candidate_machine], candidate_machine))
                    
                    total_sq_dev = max(0, S - T_sq_over_m)
                    new_normalized_dev = math.sqrt(total_sq_dev) / A
                    if logger:
                        logger.log(new_normalized_dev)
                    
                    improved = True
                    break
                else:
                    heapq.heappush(heap_local, (candidate_load, candidate_machine))
            
            current_time_after_loop = time.time()
            elapsed_total = current_time_after_loop - total_start_time
            time_remaining = args.time_limit - elapsed_total

    S_final = sum(load * load for load in machine_loads)
    total_sq_dev_final = max(0, S_final - T_sq_over_m)
    normalized_dev_final = math.sqrt(total_sq_dev_final) / A

    assignment_dict = {}
    machine_completion_times_dict = {}
    for i in range(num_machines):
        machine_key = str(i)
        jobs_on_machine = [j for j in range(num_jobs) if job_machine[j] == i]
        assignment_dict[machine_key] = jobs_on_machine
        machine_completion_times_dict[machine_key] = machine_loads[i]
    
    solution = {
        "objective_value": normalized_dev_final,
        "assignment": assignment_dict,
        "machine_completion_times": machine_completion_times_dict
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == '__main__':
    main()
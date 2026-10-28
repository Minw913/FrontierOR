import heapq
import json
import time
import argparse
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    num_jobs = instance['num_jobs']
    num_machines = instance['num_machines']
    batch_capacity = instance['batch_capacity']
    jobs = instance['jobs']

    events = []
    for job in jobs:
        heapq.heappush(events, (job['release_date'], 'job_release', job))
    
    free_machines = set(range(num_machines))
    available_jobs_set = set()
    job_info = {}
    machine_available_time = [0] * num_machines
    batch_index = [1] * num_machines
    batches_list = []
    job_assignments = {}

    start_time_real = time.time()
    time_limit = args.time_limit

    while events and (time.time() - start_time_real) < time_limit:
        current_time = events[0][0]
        events_at_time = []
        while events and events[0][0] == current_time:
            event = heapq.heappop(events)
            events_at_time.append(event)
        
        for event in events_at_time:
            if event[1] == 'job_release':
                job = event[2]
                job_id = job['job_id']
                available_jobs_set.add(job_id)
                job_info[job_id] = job
            elif event[1] == 'batch_completion':
                machine_id = event[2]
                free_machines.add(machine_id)
        
        if free_machines and available_jobs_set:
            jobs_to_schedule = [job_info[job_id] for job_id in available_jobs_set]
            jobs_to_schedule.sort(key=lambda job: (job['processing_time'], job['size']))
            free_machines_list = sorted(free_machines)
            batches_to_assign = []
            remaining_jobs = jobs_to_schedule

            for machine_id in free_machines_list:
                if not remaining_jobs:
                    break
                batch_jobs = []
                current_size = 0
                max_pt = 0
                new_remaining = []
                for job in remaining_jobs:
                    if current_size + job['size'] <= batch_capacity:
                        batch_jobs.append(job)
                        current_size += job['size']
                        if job['processing_time'] > max_pt:
                            max_pt = job['processing_time']
                    else:
                        new_remaining.append(job)
                batches_to_assign.append((machine_id, batch_jobs, max_pt))
                remaining_jobs = new_remaining
            
            for (machine_id, batch_jobs, batch_processing_time) in batches_to_assign:
                if machine_id not in free_machines:
                    continue
                free_machines.remove(machine_id)
                completion_time = current_time + batch_processing_time
                machine_available_time[machine_id] = completion_time

                batch_id = batch_index[machine_id]
                batch_index[machine_id] += 1

                batch_dict = {
                    'batch_id': batch_id,
                    'machine': machine_id,
                    'start_time': current_time,
                    'processing_time': batch_processing_time,
                    'jobs': [job['job_id'] for job in batch_jobs]
                }
                batches_list.append(batch_dict)

                for job in batch_jobs:
                    job_id = job['job_id']
                    available_jobs_set.remove(job_id)
                    job_assignments[job_id] = {
                        'batch': batch_id,
                        'machine': machine_id,
                        'completion_time': completion_time
                    }
                
                heapq.heappush(events, (completion_time, 'batch_completion', machine_id))

    if available_jobs_set and (time.time() - start_time_real) < time_limit + 10:
        jobs_remaining = [job_info[job_id] for job_id in available_jobs_set]
        jobs_remaining_sorted = sorted(jobs_remaining, key=lambda job: (job['processing_time'], job['size']))
        batches_remaining = []
        for job in jobs_remaining_sorted:
            placed = False
            for batch in batches_remaining:
                if sum(j['size'] for j in batch) + job['size'] <= batch_capacity:
                    batch.append(job)
                    placed = True
                    break
            if not placed:
                batches_remaining.append([job])
        
        machine_queue = []
        for m in range(num_machines):
            heapq.heappush(machine_queue, (machine_available_time[m], m))
        
        for batch in batches_remaining:
            if not machine_queue:
                break
            batch_processing_time = max(job['processing_time'] for job in batch)
            batch_release_date = max(job['release_date'] for job in batch)
            next_avail, machine_id = heapq.heappop(machine_queue)
            start_time = max(next_avail, batch_release_date)
            completion_time = start_time + batch_processing_time

            batch_id = batch_index[machine_id]
            batch_index[machine_id] += 1

            batch_dict = {
                'batch_id': batch_id,
                'machine': machine_id,
                'start_time': start_time,
                'processing_time': batch_processing_time,
                'jobs': [job['job_id'] for job in batch]
            }
            batches_list.append(batch_dict)

            for job in batch:
                job_id = job['job_id']
                job_assignments[job_id] = {
                    'batch': batch_id,
                    'machine': machine_id,
                    'completion_time': completion_time
                }
            
            heapq.heappush(machine_queue, (completion_time, machine_id))
            available_jobs_set.difference_update(job['job_id'] for job in batch)

    total_flow = sum(assignment['completion_time'] for assignment in job_assignments.values())
    if logger:
        logger.log(total_flow)
    
    job_assignments_output = {}
    for job_id, assignment in job_assignments.items():
        job_assignments_output[str(job_id)] = assignment
    
    output_dict = {
        "objective_value": float(total_flow),
        "batches": batches_list,
        "job_assignments": job_assignments_output
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(output_dict, f)

if __name__ == '__main__':
    main()
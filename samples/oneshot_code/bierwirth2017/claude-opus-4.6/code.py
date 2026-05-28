import argparse
import json
import time
import random
import copy
import math

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    return parser.parse_args()

def load_instance(path):
    with open(path, 'r') as f:
        return json.load(f)

def compute_schedule(instance, machine_orders):
    """
    Given machine_orders (for each machine, a list of job indices in processing order),
    compute the start times of all operations.
    Returns schedule info or None if infeasible.
    """
    num_jobs = instance['num_jobs']
    num_machines = instance['num_machines']
    jobs = instance['jobs']
    
    # Precompute operation info: for each job, list of (machine, processing_time)
    # job_op_idx[j][m] = index of machine m in job j's operation sequence
    job_ops = []
    for j in range(num_jobs):
        job_ops.append(jobs[j]['operations'])
    
    # start_times[j][op_idx] = start time of job j's op_idx-th operation
    start_times = [[0] * num_machines for _ in range(num_jobs)]
    
    # For each job, track when the job is available (after previous op finishes)
    job_available = [jobs[j]['release_date'] for j in range(num_jobs)]
    
    # For each machine, track when the machine is available
    machine_available = [0] * num_machines
    
    # We need to process operations respecting both job order and machine order
    # job_next_op[j] = next operation index to schedule for job j
    job_next_op = [0] * num_jobs
    
    # machine_next_idx[m] = next index in machine_orders[m] to process
    machine_next_idx = [0] * num_machines
    
    # For each job, which machine does each operation use
    # job_op_machine[j][op] = machine
    job_op_machine = []
    job_op_pt = []
    for j in range(num_jobs):
        machines = []
        pts = []
        for op in job_ops[j]:
            machines.append(op['machine'])
            pts.append(op['processing_time'])
        job_op_machine.append(machines)
        job_op_pt.append(pts)
    
    # For each machine, build the order of (job, op_index_in_job) pairs
    # We need to know for each job which operation index corresponds to each machine
    job_machine_to_op = [[0] * num_machines for _ in range(num_jobs)]
    for j in range(num_jobs):
        for op_idx in range(num_machines):
            m = job_op_machine[j][op_idx]
            job_machine_to_op[j][m] = op_idx
    
    # Process using a simulation approach
    # For each machine order, we process jobs in the given sequence
    # We iterate until all operations are scheduled
    
    total_ops = num_jobs * num_machines
    scheduled = 0
    
    # Reset
    job_available = [jobs[j]['release_date'] for j in range(num_jobs)]
    machine_available = [0] * num_machines
    job_next_op = [0] * num_jobs
    machine_next_idx = [0] * num_machines
    
    max_iter = total_ops * 10
    iteration = 0
    
    while scheduled < total_ops and iteration < max_iter:
        iteration += 1
        progress = False
        for m in range(num_machines):
            if machine_next_idx[m] >= len(machine_orders[m]):
                continue
            j = machine_orders[m][machine_next_idx[m]]
            op_idx = job_machine_to_op[j][m]
            # Check if this is the next operation for job j
            if job_next_op[j] != op_idx:
                continue
            # Schedule it
            st = max(job_available[j], machine_available[m])
            pt = job_op_pt[j][op_idx]
            start_times[j][op_idx] = st
            job_available[j] = st + pt
            machine_available[m] = st + pt
            job_next_op[j] = op_idx + 1
            machine_next_idx[m] += 1
            scheduled += 1
            progress = True
        
        if not progress:
            return None  # Deadlock
    
    if scheduled < total_ops:
        return None
    
    return start_times

def evaluate_schedule(instance, start_times):
    """Compute total weighted tardiness from start_times."""
    jobs = instance['jobs']
    num_jobs = instance['num_jobs']
    num_machines = instance['num_machines']
    
    total_wt = 0.0
    for j in range(num_jobs):
        last_op_idx = num_machines - 1
        pt = jobs[j]['operations'][last_op_idx]['processing_time']
        completion = start_times[j][last_op_idx] + pt
        tardiness = max(0, completion - jobs[j]['due_date'])
        total_wt += jobs[j]['weight'] * tardiness
    
    return total_wt

def build_machine_orders_from_priority(instance, priority):
    """
    Build machine orders using a priority-based dispatch.
    priority[j] = priority value for job j (lower = higher priority).
    We simulate dispatching: at each step, among all operations ready to be scheduled,
    pick the one with lowest priority value.
    """
    num_jobs = instance['num_jobs']
    num_machines = instance['num_machines']
    jobs = instance['jobs']
    
    job_op_machine = []
    job_op_pt = []
    for j in range(num_jobs):
        machines = []
        pts = []
        for op in jobs[j]['operations']:
            machines.append(op['machine'])
            pts.append(op['processing_time'])
        job_op_machine.append(machines)
        job_op_pt.append(pts)
    
    job_available = [jobs[j]['release_date'] for j in range(num_jobs)]
    machine_available = [0] * num_machines
    job_next_op = [0] * num_jobs
    
    machine_orders = [[] for _ in range(num_machines)]
    
    total_ops = num_jobs * num_machines
    scheduled = 0
    
    while scheduled < total_ops:
        # Find all dispatchable operations
        candidates = []
        for j in range(num_jobs):
            if job_next_op[j] < num_machines:
                op_idx = job_next_op[j]
                m = job_op_machine[j][op_idx]
                st = max(job_available[j], machine_available[m])
                candidates.append((st, priority[j], j, op_idx, m))
        
        if not candidates:
            break
        
        # Find minimum possible start time
        min_st = min(c[0] for c in candidates)
        
        # Among candidates that can start at or near min_st, pick by priority
        # Actually, let's use a greedy: pick the candidate with best (priority, start_time)
        # Or pick earliest start, break ties by priority
        candidates.sort(key=lambda c: (c[0], c[1]))
        
        # But we need to be careful: scheduling one op may enable others
        # Simple approach: schedule the best candidate
        _, _, j, op_idx, m = candidates[0]
        st = max(job_available[j], machine_available[m])
        pt = job_op_pt[j][op_idx]
        
        machine_orders[m].append(j)
        job_available[j] = st + pt
        machine_available[m] = st + pt
        job_next_op[j] = op_idx + 1
        scheduled += 1
    
    return machine_orders

def dispatch_schedule(instance, dispatch_rule='edd'):
    """Create schedule using dispatch rules."""
    num_jobs = instance['num_jobs']
    num_machines = instance['num_machines']
    jobs = instance['jobs']
    
    job_op_machine = []
    job_op_pt = []
    for j in range(num_jobs):
        machines = []
        pts = []
        for op in jobs[j]['operations']:
            machines.append(op['machine'])
            pts.append(op['processing_time'])
        job_op_machine.append(machines)
        job_op_pt.append(pts)
    
    job_machine_to_op = [[0] * num_machines for _ in range(num_jobs)]
    for j in range(num_jobs):
        for op_idx in range(num_machines):
            m = job_op_machine[j][op_idx]
            job_machine_to_op[j][m] = op_idx
    
    job_available = [float(jobs[j]['release_date']) for j in range(num_jobs)]
    machine_available = [0.0] * num_machines
    job_next_op = [0] * num_jobs
    
    start_times = [[0.0] * num_machines for _ in range(num_jobs)]
    machine_orders = [[] for _ in range(num_machines)]
    
    # Remaining processing time for each job
    job_remaining_pt = []
    for j in range(num_jobs):
        total = sum(job_op_pt[j])
        job_remaining_pt.append(total)
    
    total_ops = num_jobs * num_machines
    scheduled = 0
    
    while scheduled < total_ops:
        # Find all dispatchable operations
        candidates = []
        for j in range(num_jobs):
            if job_next_op[j] < num_machines:
                op_idx = job_next_op[j]
                m = job_op_machine[j][op_idx]
                st = max(job_available[j], machine_available[m])
                candidates.append((j, op_idx, m, st))
        
        if not candidates:
            break
        
        min_st = min(c[3] for c in candidates)
        
        # Filter to those that can start reasonably soon
        # Use "non-delay" or "active" schedule generation
        # For non-delay: only consider ops that can start at min_st
        # For active: consider ops whose machine is free and job is available before
        #   the earliest completion of any candidate
        
        # Active schedule: find earliest completion
        min_completion = min(c[3] + job_op_pt[c[0]][c[1]] for c in candidates)
        
        # Filter: candidates whose start time < min_completion on the same machine
        # Group by machine, find for each machine the earliest completion
        machine_earliest_completion = {}
        for j, op_idx, m, st in candidates:
            ct = st + job_op_pt[j][op_idx]
            if m not in machine_earliest_completion or ct < machine_earliest_completion[m]:
                machine_earliest_completion[m] = ct
        
        # For active schedule, pick a machine with earliest completion
        best_machine = min(machine_earliest_completion, key=machine_earliest_completion.get)
        threshold = machine_earliest_completion[best_machine]
        
        eligible = [(j, op_idx, m, st) for j, op_idx, m, st in candidates 
                     if m == best_machine and st < threshold]
        
        # Apply dispatch rule to select among eligible
        if dispatch_rule == 'edd':
            # Earliest due date
            eligible.sort(key=lambda c: jobs[c[0]]['due_date'])
        elif dispatch_rule == 'wspt':
            # Weighted shortest processing time
            eligible.sort(key=lambda c: job_op_pt[c[0]][c[1]] / max(jobs[c[0]]['weight'], 1))
        elif dispatch_rule == 'watc':
            # Weighted apparent tardiness cost
            avg_pt = sum(sum(job_op_pt[j]) for j in range(num_jobs)) / max(num_jobs * num_machines, 1)
            k = 2.0
            def watc_key(c):
                j, op_idx, m, st = c
                w = jobs[j]['weight']
                p = job_op_pt[j][op_idx]
                d = jobs[j]['due_date']
                remaining = job_remaining_pt[j]
                slack = max(d - st - remaining, 0)
                return -(w / max(p, 1)) * math.exp(-slack / (k * max(avg_pt, 1)))
            eligible.sort(key=watc_key)
        elif dispatch_rule == 'slack':
            def slack_key(c):
                j, op_idx, m, st = c
                d = jobs[j]['due_date']
                remaining = job_remaining_pt[j]
                slack = d - st - remaining
                return slack / max(jobs[j]['weight'], 1)
            eligible.sort(key=slack_key)
        
        # Schedule the best
        j, op_idx, m, st = eligible[0]
        pt = job_op_pt[j][op_idx]
        actual_st = max(job_available[j], machine_available[m])
        
        start_times[j][op_idx] = actual_st
        job_available[j] = actual_st + pt
        machine_available[m] = actual_st + pt
        job_next_op[j] = op_idx + 1
        job_remaining_pt[j] -= pt
        machine_orders[m].append(j)
        scheduled += 1
    
    return start_times, machine_orders

def compute_schedule_from_machine_orders(instance, machine_orders):
    """
    More efficient schedule computation given machine orders.
    """
    num_jobs = instance['num_jobs']
    num_machines = instance['num_machines']
    jobs = instance['jobs']
    
    job_op_machine = []
    job_op_pt = []
    for j in range(num_jobs):
        machines = []
        pts = []
        for op in jobs[j]['operations']:
            machines.append(op['machine'])
            pts.append(op['processing_time'])
        job_op_machine.append(machines)
        job_op_pt.append(pts)
    
    job_machine_to_op = [[0] * num_machines for _ in range(num_jobs)]
    for j in range(num_jobs):
        for op_idx in range(num_machines):
            m = job_op_machine[j][op_idx]
            job_machine_to_op[j][m] = op_idx
    
    # Build precedence constraints and compute start times
    # For each operation (j, op_idx), its start time >= job_available after previous op
    # and >= machine_available after previous job on same machine
    
    start_times = [[0.0] * num_machines for _ in range(num_jobs)]
    
    # We need topological order
    # Each operation depends on: previous operation of same job, previous job on same machine
    
    # in_degree for each operation
    in_degree = [[0] * num_machines for _ in range(num_jobs)]
    
    # For job precedence: op_idx > 0 depends on op_idx - 1
    for j in range(num_jobs):
        for op_idx in range(1, num_machines):
            in_degree[j][op_idx] += 1
    
    # For machine precedence
    machine_prev = {}  # (j, op_idx) -> previous (j', op_idx') on same machine
    for m in range(num_machines):
        for idx in range(1, len(machine_orders[m])):
            j_curr = machine_orders[m][idx]
            op_curr = job_machine_to_op[j_curr][m]
            in_degree[j_curr][op_curr] += 1
    
    # BFS/topological sort
    from collections import deque
    queue = deque()
    
    for j in range(num_jobs):
        if in_degree[j][0] == 0:
            start_times[j][0] = float(jobs[j]['release_date'])
            queue.append((j, 0))
    
    # Successors
    # Job successor: (j, op_idx) -> (j, op_idx+1)
    # Machine successor: using machine_orders
    
    job_available = [float(jobs[j]['release_date']) for j in range(num_jobs)]
    machine_available = [0.0] * num_machines
    
    # Actually let's just do iterative computation
    # Process in topological order
    
    # Reset
    for j in range(num_jobs):
        for op_idx in range(num_machines):
            start_times[j][op_idx] = 0.0
            in_degree[j][op_idx] = 0
    
    for j in range(num_jobs):
        for op_idx in range(1, num_machines):
            in_degree[j][op_idx] += 1
    
    for m in range(num_machines):
        for idx in range(1, len(machine_orders[m])):
            j_curr = machine_orders[m][idx]
            op_curr = job_machine_to_op[j_curr][m]
            in_degree[j_curr][op_curr] += 1
    
    # Initialize start times with release dates for first ops
    queue = deque()
    for j in range(num_jobs):
        op_idx = 0
        start_times[j][op_idx] = float(jobs[j]['release_date'])
        if in_degree[j][op_idx] == 0:
            queue.append((j, op_idx))
    
    while queue:
        j, op_idx = queue.popleft()
        m = job_op_machine[j][op_idx]
        pt = job_op_pt[j][op_idx]
        finish = start_times[j][op_idx] + pt
        
        # Job successor
        if op_idx + 1 < num_machines:
            next_op = op_idx + 1
            start_times[j][next_op] = max(start_times[j][next_op], finish)
            in_degree[j][next_op] -= 1
            if in_degree[j][next_op] == 0:
                queue.append((j, next_op))
        
        # Machine successor
        mo = machine_orders[m]
        # Find position of j in machine_orders[m]
        # This is slow; let's precompute
        # Actually let's build machine_order_pos
        pass
    
    # Redo with precomputed positions
    # Build position of each job in each machine's order
    machine_order_pos = {}
    for m in range(num_machines):
        for idx, j in enumerate(machine_orders[m]):
            machine_order_pos[(m, j)] = idx
    
    # Reset everything
    for j in range(num_jobs):
        for op_idx in range(num_machines):
            start_times[j][op_idx] = 0.0
            in_degree[j][op_idx] = 0
    
    for j in range(num_jobs):
        start_times[j][0] = float(jobs[j]['release_date'])
        for op_idx in range(1, num_machines):
            in_degree[j][op_idx] += 1
    
    for m in range(num_machines):
        for idx in range(1, len(machine_orders[m])):
            j_curr = machine_orders[m][idx]
            op_curr = job_machine_to_op[j_curr][m]
            in_degree[j_curr][op_curr] += 1
    
    queue = deque()
    for j in range(num_jobs):
        if in_degree[j][0] == 0:
            queue.append((j, 0))
    
    # Build machine successor mapping
    machine_succ = {}  # (m, j) -> next job on machine m
    for m in range(num_machines):
        for idx in range(len(machine_orders[m]) - 1):
            j_curr = machine_orders[m][idx]
            j_next = machine_orders[m][idx + 1]
            machine_succ[(m, j_curr)] = j_next
    
    while queue:
        j, op_idx = queue.popleft()
        m = job_op_machine[j][op_idx]
        pt = job_op_pt[j][op_idx]
        finish = start_times[j][op_idx] + pt
        
        # Job successor
        if op_idx + 1 < num_machines:
            next_op = op_idx + 1
            start_times[j][next_op] = max(start_times[j][next_op], finish)
            in_degree[j][next_op] -= 1
            if in_degree[j][next_op] == 0:
                queue.append((j, next_op))
        
        # Machine successor
        if (m, j) in machine_succ:
            j_next = machine_succ[(m, j)]
            op_next = job_machine_to_op[j_next][m]
            start_times[j_next][op_next] = max(start_times[j_next][op_next], finish)
            in_degree[j_next][op_next] -= 1
            if in_degree[j_next][op_next] == 0:
                queue.append((j_next, op_next))
    
    return start_times

def neighborhood_swap(machine_orders, machine, i, j):
    """Swap positions i and j in machine's order."""
    new_orders = [mo[:] for mo in machine_orders]
    new_orders[machine][i], new_orders[machine][j] = new_orders[machine][j], new_orders[machine][i]
    return new_orders

def format_solution(instance, start_times, objective_value):
    jobs = instance['jobs']
    num_jobs = instance['num_jobs']
    num_machines = instance['num_machines']
    
    schedule = []
    for j in range(num_jobs):
        last_op_idx = num_machines - 1
        pt = jobs[j]['operations'][last_op_idx]['processing_time']
        completion = start_times[j][last_op_idx] + pt
        tardiness = max(0, completion - jobs[j]['due_date'])
        
        ops = []
        for op_idx in range(num_machines):
            ops.append({
                'machine': jobs[j]['operations'][op_idx]['machine'],
                'start_time': start_times[j][op_idx]
            })
        
        schedule.append({
            'job_id': j,
            'completion_time': completion,
            'tardiness': tardiness,
            'operations': ops
        })
    
    return {
        'objective_value': objective_value,
        'schedule': schedule
    }

def is_valid_swap(instance, machine_orders, machine, i, j, job_machine_to_op, job_op_machine):
    """Check if swapping positions i and j on a machine could lead to deadlock."""
    # For job shop, swapping two adjacent jobs on a machine is always cycle-free
    # if they are adjacent. For non-adjacent swaps, we need to be more careful.
    # For simplicity, we'll just try and verify.
    return True

def local_search(instance, machine_orders, start_times, best_obj, time_limit, start_time, logger):
    """
    Perform local search using adjacent swap moves on critical path.
    """
    num_jobs = instance['num_jobs']
    num_machines = instance['num_machines']
    jobs = instance['jobs']
    
    job_op_machine = []
    job_op_pt = []
    for j in range(num_jobs):
        machines = []
        pts = []
        for op in jobs[j]['operations']:
            machines.append(op['machine'])
            pts.append(op['processing_time'])
        job_op_machine.append(machines)
        job_op_pt.append(pts)
    
    job_machine_to_op = [[0] * num_machines for _ in range(num_jobs)]
    for j in range(num_jobs):
        for op_idx in range(num_machines):
            m = job_op_machine[j][op_idx]
            job_machine_to_op[j][m] = op_idx
    
    current_orders = [mo[:] for mo in machine_orders]
    current_obj = best_obj
    current_st = start_times
    
    improved = True
    while improved:
        if time.time() - start_time > time_limit * 0.95:
            break
        improved = False
        
        # Try all adjacent swaps on each machine
        for m in range(num_machines):
            if time.time() - start_time > time_limit * 0.95:
                break
            mo = current_orders[m]
            for i in range(len(mo) - 1):
                if time.time() - start_time > time_limit * 0.95:
                    break
                # Try swapping i and i+1
                new_orders = [o[:] for o in current_orders]
                new_orders[m][i], new_orders[m][i+1] = new_orders[m][i+1], new_orders[m][i]
                
                try:
                    new_st = compute_schedule_from_machine_orders(instance, new_orders)
                    new_obj = evaluate_schedule(instance, new_st)
                    
                    if new_obj < current_obj:
                        current_orders = new_orders
                        current_obj = new_obj
                        current_st = new_st
                        improved = True
                        if current_obj < best_obj:
                            best_obj = current_obj
                            if logger:
                                logger.log(best_obj)
                except:
                    pass
    
    return current_orders, current_st, current_obj

def find_critical_operations(instance, start_times, machine_orders):
    """Find operations on the critical path of tardy jobs."""
    num_jobs = instance['num_jobs']
    num_machines = instance['num_machines']
    jobs = instance['jobs']
    
    job_op_machine = []
    job_op_pt = []
    for j in range(num_jobs):
        machines = []
        pts = []
        for op in jobs[j]['operations']:
            machines.append(op['machine'])
            pts.append(op['processing_time'])
        job_op_machine.append(machines)
        job_op_pt.append(pts)
    
    job_machine_to_op = [[0] * num_machines for _ in range(num_jobs)]
    for j in range(num_jobs):
        for op_idx in range(num_machines):
            m = job_op_machine[j][op_idx]
            job_machine_to_op[j][m] = op_idx
    
    critical_blocks = []  # (machine, position_in_order) pairs to try swapping
    
    # Find tardy jobs and trace their critical paths
    for j in range(num_jobs):
        last_op = num_machines - 1
        pt = job_op_pt[j][last_op]
        completion = start_times[j][last_op] + pt
        tardiness = max(0, completion - jobs[j]['due_date'])
        if tardiness <= 0:
            continue
        
        # Trace back: for each operation of this job, check if it's delayed by machine
        for op_idx in range(num_machines):
            m = job_op_machine[j][op_idx]
            mo = machine_orders[m]
            # Find position of j in machine order
            for pos in range(len(mo)):
                if mo[pos] == j:
                    if pos > 0:
                        critical_blocks.append((m, pos - 1))
                    if pos < len(mo) - 1:
                        critical_blocks.append((m, pos))
                    break
    
    return critical_blocks

def targeted_local_search(instance, machine_orders, start_times, best_obj, time_limit, start_time, logger):
    """Local search focusing on critical operations of tardy jobs."""
    num_jobs = instance['num_jobs']
    num_machines = instance['num_machines']
    
    current_orders = [mo[:] for mo in machine_orders]
    current_obj = best_obj
    current_st = start_times
    
    improved = True
    iteration = 0
    while improved:
        if time.time() - start_time > time_limit * 0.95:
            break
        improved = False
        iteration += 1
        
        # Find critical blocks
        critical_blocks = find_critical_operations(instance, current_st, current_orders)
        
        # Remove duplicates
        critical_blocks = list(set(critical_blocks))
        random.shuffle(critical_blocks)
        
        for m, pos in critical_blocks:
            if time.time() - start_time > time_limit * 0.95:
                break
            
            new_orders = [o[:] for o in current_orders]
            new_orders[m][pos], new_orders[m][pos+1] = new_orders[m][pos+1], new_orders[m][pos]
            
            try:
                new_st = compute_schedule_from_machine_orders(instance, new_orders)
                new_obj = evaluate_schedule(instance, new_st)
                
                if new_obj < current_obj:
                    current_orders = new_orders
                    current_obj = new_obj
                    current_st = new_st
                    improved = True
                    if current_obj < best_obj:
                        best_obj = current_obj
                        if logger:
                            logger.log(best_obj)
                    break  # Restart search from new solution
            except:
                pass
        
        # If no improvement from critical blocks, try broader search
        if not improved and iteration < 3:
            for m in range(num_machines):
                if time.time() - start_time > time_limit * 0.95:
                    break
                mo = current_orders[m]
                for i in range(len(mo) - 1):
                    if time.time() - start_time > time_limit * 0.95:
                        break
                    new_orders = [o[:] for o in current_orders]
                    new_orders[m][i], new_orders[m][i+1] = new_orders[m][i+1], new_orders[m][i]
                    
                    try:
                        new_st = compute_schedule_from_machine_orders(instance, new_orders)
                        new_obj = evaluate_schedule(instance, new_st)
                        
                        if new_obj < current_obj:
                            current_orders = new_orders
                            current_obj = new_obj
                            current_st = new_st
                            improved = True
                            if current_obj < best_obj:
                                best_obj = current_obj
                                if logger:
                                    logger.log(best_obj)
                            break
                    except:
                        pass
                if improved:
                    break
    
    return current_orders, current_st, current_obj

def simulated_annealing(instance, machine_orders, start_times, best_obj, time_limit, start_time, logger):
    """Simulated annealing with adjacent swaps."""
    num_jobs = instance['num_jobs']
    num_machines = instance['num_machines']
    
    current_orders = [mo[:] for mo in machine_orders]
    current_obj = best_obj
    current_st = start_times
    
    best_orders = [mo[:] for mo in current_orders]
    best_st = current_st
    global_best_obj = best_obj
    
    # SA parameters
    T_start = max(best_obj * 0.1, 100)
    T_end = 0.1
    remaining_time = time_limit * 0.90 - (time.time() - start_time)
    if remaining_time <= 0:
        return best_orders, best_st, global_best_obj
    
    iterations = 0
    max_iterations = 1000000
    
    # Cooling
    alpha = 0.9999
    T = T_start
    
    while time.time() - start_time < time_limit * 0.90 and iterations < max_iterations:
        iterations += 1
        
        # Pick random machine and random adjacent swap
        m = random.randint(0, num_machines - 1)
        if len(current_orders[m]) < 2:
            continue
        pos = random.randint(0, len(current_orders[m]) - 2)
        
        # Try swap
        new_orders = [o[:] for o in current_orders]
        new_orders[m][pos], new_orders[m][pos+1] = new_orders[m][pos+1], new_orders[m][pos]
        
        try:
            new_st = compute_schedule_from_machine_orders(instance, new_orders)
            new_obj = evaluate_schedule(instance, new_st)
        except:
            continue
        
        delta = new_obj - current_obj
        
        if delta < 0 or (T > 0 and random.random() < math.exp(-delta / max(T, 1e-10))):
            current_orders = new_orders
            current_obj = new_obj
            current_st = new_st
            
            if current_obj < global_best_obj:
                global_best_obj = current_obj
                best_orders = [o[:] for o in current_orders]
                best_st = current_st
                if logger:
                    logger.log(global_best_obj)
        
        T *= alpha
        if T < T_end:
            T = T_end
    
    return best_orders, best_st, global_best_obj

def extract_machine_orders_from_start_times(instance, start_times):
    """Extract machine orders from computed start times."""
    num_jobs = instance['num_jobs']
    num_machines = instance['num_machines']
    jobs = instance['jobs']
    
    machine_orders = [[] for _ in range(num_machines)]
    
    # For each machine, collect (start_time, job) and sort
    machine_ops = [[] for _ in range(num_machines)]
    for j in range(num_jobs):
        for op_idx in range(num_machines):
            m = jobs[j]['operations'][op_idx]['machine']
            machine_ops[m].append((start_times[j][op_idx], j))
    
    for m in range(num_machines):
        machine_ops[m].sort()
        machine_orders[m] = [j for _, j in machine_ops[m]]
    
    return machine_orders

def solve(instance, time_limit, logger):
    start_time = time.time()
    num_jobs = instance['num_jobs']
    num_machines = instance['num_machines']
    
    best_obj = float('inf')
    best_st = None
    best_orders = None
    
    # Try multiple dispatch rules
    rules = ['watc', 'edd', 'wspt', 'slack']
    
    for rule in rules:
        if time.time() - start_time > time_limit * 0.1:
            break
        try:
            st, mo = dispatch_schedule(instance, rule)
            obj = evaluate_schedule(instance, st)
            if obj < best_obj:
                best_obj = obj
                best_st = st
                best_orders = mo
                if logger:
                    logger.log(best_obj)
        except:
            pass
    
    if best_st is None:
        # Fallback: simple EDD
        st, mo = dispatch_schedule(instance, 'edd')
        best_obj = evaluate_schedule(instance, st)
        best_st = st
        best_orders = mo
        if logger:
            logger.log(best_obj)
    
    # Local search phase
    if time.time() - start_time < time_limit * 0.3:
        try:
            new_orders, new_st, new_obj = targeted_local_search(
                instance, best_orders, best_st, best_obj, time_limit * 0.3, start_time, logger)
            if new_obj < best_obj:
                best_obj = new_obj
                best_st = new_st
                best_orders = new_orders
        except:
            pass
    
    # Simulated annealing phase
    if time.time() - start_time < time_limit * 0.9:
        try:
            sa_orders, sa_st, sa_obj = simulated_annealing(
                instance, best_orders, best_st, best_obj, time_limit, start_time, logger)
            if sa_obj < best_obj:
                best_obj = sa_obj
                best_st = sa_st
                best_orders = sa_orders
        except:
            pass
    
    # Final local search
    if time.time() - start_time < time_limit * 0.95:
        try:
            new_orders, new_st, new_obj = targeted_local_search(
                instance, best_orders, best_st, best_obj, time_limit, start_time, logger)
            if new_obj < best_obj:
                best_obj = new_obj
                best_st = new_st
                best_orders = new_orders
        except:
            pass
    
    return best_st, best_obj

def main():
    args = parse_args()
    
    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    instance = load_instance(args.instance_path)
    
    start_times, objective_value = solve(instance, args.time_limit, logger)
    
    solution = format_solution(instance, start_times, objective_value)
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == '__main__':
    main()
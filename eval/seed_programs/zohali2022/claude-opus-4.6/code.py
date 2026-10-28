import argparse
import json
import time
import random
import math
from copy import deepcopy

def solve(instance_path, solution_path, time_limit, log_path=None):
    from solution_logger import SolutionLogger
    
    start_time = time.time()
    
    with open(instance_path, 'r') as f:
        data = json.load(f)
    
    num_tasks = data['num_tasks']
    num_stations = data['num_stations']
    cycle_time_ub = data['cycle_time']
    processing_times = data['processing_times']
    precedence_relations = data['precedence_relations']
    forward_setup = data['forward_setup_times']
    backward_setup = data['backward_setup_times']
    
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    # Determine if indices are 0-based or 1-based by checking range
    max_idx = 0
    min_idx = float('inf')
    for rel in precedence_relations:
        max_idx = max(max_idx, rel[0], rel[1])
        min_idx = min(min_idx, rel[0], rel[1])
    
    # If max index equals num_tasks, it's 1-based; if max index equals num_tasks-1, it's 0-based
    one_based = False
    if len(precedence_relations) > 0:
        if max_idx >= num_tasks:
            one_based = True
        elif min_idx >= 1 and max_idx == num_tasks:
            one_based = True
    
    # Build precedence graph (0-based internally)
    predecessors = [set() for _ in range(num_tasks)]
    successors = [set() for _ in range(num_tasks)]
    for rel in precedence_relations:
        i, j = rel[0], rel[1]
        if one_based:
            i -= 1
            j -= 1
        if 0 <= i < num_tasks and 0 <= j < num_tasks:
            predecessors[j].add(i)
            successors[i].add(j)
    
    # Compute transitive closure for precedence
    all_predecessors = [set() for _ in range(num_tasks)]
    all_successors = [set() for _ in range(num_tasks)]
    
    # Topological order
    in_degree = [0] * num_tasks
    for j in range(num_tasks):
        in_degree[j] = len(predecessors[j])
    
    topo_order = []
    queue = [i for i in range(num_tasks) if in_degree[i] == 0]
    temp_in = in_degree[:]
    while queue:
        node = queue.pop(0)
        topo_order.append(node)
        for s in successors[node]:
            temp_in[s] -= 1
            if temp_in[s] == 0:
                queue.append(s)
    
    # If topo_order doesn't contain all tasks, there's a cycle - just add remaining
    if len(topo_order) < num_tasks:
        remaining = set(range(num_tasks)) - set(topo_order)
        topo_order.extend(remaining)
    
    for node in topo_order:
        for p in predecessors[node]:
            all_predecessors[node].add(p)
            all_predecessors[node].update(all_predecessors[p])
    
    for node in reversed(topo_order):
        for s in successors[node]:
            all_successors[node].add(s)
            all_successors[node].update(all_successors[s])
    
    # Compute earliest and latest station for each task (0-based stations internally)
    earliest_station = [0] * num_tasks
    latest_station = [num_stations - 1] * num_tasks
    
    for node in topo_order:
        for p in predecessors[node]:
            earliest_station[node] = max(earliest_station[node], earliest_station[p])
    
    for node in reversed(topo_order):
        for s in successors[node]:
            latest_station[node] = min(latest_station[node], latest_station[s])
    
    def compute_station_time(task_sequence):
        """Compute the cycle time of a single station given its task sequence (0-based)."""
        if not task_sequence:
            return 0
        total = sum(processing_times[t] for t in task_sequence)
        for i in range(len(task_sequence) - 1):
            total += forward_setup[task_sequence[i]][task_sequence[i+1]]
        # backward setup from last to first
        total += backward_setup[task_sequence[-1]][task_sequence[0]]
        return total
    
    def find_best_sequence(tasks):
        """Find the best ordering of tasks at a station respecting precedence constraints within station."""
        if not tasks:
            return [], 0
        if len(tasks) == 1:
            t = tasks[0]
            return [t], processing_times[t] + backward_setup[t][t]
        
        task_set = set(tasks)
        # Local precedence within this station
        local_preds = {t: set() for t in tasks}
        for t in tasks:
            for p in predecessors[t]:
                if p in task_set:
                    local_preds[t].add(p)
        
        if len(tasks) <= 10:
            # Branch and bound / enumerate valid permutations
            best_time = float('inf')
            best_seq = None
            
            def generate_valid_sequences(remaining, current_seq, available):
                nonlocal best_time, best_seq
                if not remaining:
                    t = compute_station_time(current_seq)
                    if t < best_time:
                        best_time = t
                        best_seq = current_seq[:]
                    return
                
                # Pruning: lower bound
                partial_sum = sum(processing_times[t] for t in current_seq)
                for i in range(len(current_seq) - 1):
                    partial_sum += forward_setup[current_seq[i]][current_seq[i+1]]
                remaining_proc = sum(processing_times[t] for t in remaining)
                if partial_sum + remaining_proc >= best_time:
                    return
                
                for task in sorted(available):
                    new_remaining = remaining - {task}
                    current_seq.append(task)
                    new_available = set()
                    placed = set(current_seq)
                    for r in new_remaining:
                        if local_preds[r] <= placed:
                            new_available.add(r)
                    generate_valid_sequences(new_remaining, current_seq, new_available)
                    current_seq.pop()
            
            remaining = set(tasks)
            available = {t for t in tasks if not local_preds[t]}
            generate_valid_sequences(remaining, [], available)
            return best_seq if best_seq else tasks, best_time if best_time < float('inf') else compute_station_time(tasks)
        else:
            return greedy_sequence_with_local_search(tasks, local_preds)
    
    def greedy_sequence_with_local_search(tasks, local_preds):
        task_set = set(tasks)
        
        def greedy_sequence(tasks, local_preds, randomize=False):
            remaining = set(tasks)
            available = [t for t in tasks if not local_preds[t]]
            sequence = []
            while available:
                if randomize:
                    if len(sequence) == 0:
                        chosen = random.choice(available)
                    else:
                        last = sequence[-1]
                        scored = [(forward_setup[last][t], t) for t in available]
                        scored.sort()
                        top_k = min(3, len(scored))
                        chosen = scored[random.randint(0, top_k-1)][1]
                else:
                    if len(sequence) == 0:
                        chosen = available[0]
                    else:
                        last = sequence[-1]
                        best_t = None
                        best_cost = float('inf')
                        for t in available:
                            c = forward_setup[last][t]
                            if c < best_cost:
                                best_cost = c
                                best_t = t
                        chosen = best_t
                
                sequence.append(chosen)
                remaining.remove(chosen)
                available = [t for t in remaining if local_preds[t] <= set(sequence)]
            
            return sequence
        
        best_seq = greedy_sequence(tasks, local_preds, randomize=False)
        best_time = compute_station_time(best_seq)
        
        for _ in range(min(20, max(5, 100 // len(tasks)))):
            seq = greedy_sequence(tasks, local_preds, randomize=True)
            t = compute_station_time(seq)
            if t < best_time:
                best_time = t
                best_seq = seq
        
        # Local search: swap adjacent tasks if precedence allows
        improved = True
        while improved:
            improved = False
            for i in range(len(best_seq) - 1):
                t1, t2 = best_seq[i], best_seq[i+1]
                if t1 in local_preds[t2]:
                    continue
                new_seq = best_seq[:i] + [t2, t1] + best_seq[i+2:]
                pos = {t: p for p, t in enumerate(new_seq)}
                valid = True
                for t in [t1, t2]:
                    for pred in local_preds[t]:
                        if pos[pred] > pos[t]:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    new_time = compute_station_time(new_seq)
                    if new_time < best_time:
                        best_time = new_time
                        best_seq = new_seq
                        improved = True
        
        return best_seq, best_time
    
    def evaluate_solution(assignment, station_sequences):
        """Evaluate a full solution. Returns cycle time and per-station times."""
        max_time = 0
        station_times = {}
        for s in range(num_stations):
            seq = station_sequences.get(s, [])
            if not seq:
                return float('inf'), {}
            t = compute_station_time(seq)
            station_times[s] = t
            max_time = max(max_time, t)
        return max_time, station_times
    
    def build_initial_solution():
        assignment = [0] * num_tasks
        station_tasks = [[] for _ in range(num_stations)]
        
        # Distribute tasks respecting precedence, trying to balance
        # Calculate total processing time for rough balance target
        total_proc = sum(processing_times)
        target_per_station = total_proc / num_stations
        
        current_station = 0
        current_load = 0
        
        for task in topo_order:
            es = earliest_station[task]
            ls = latest_station[task]
            
            target = max(es, current_station)
            if target > ls:
                target = ls
            
            # Check if we should move to next station
            if current_load >= target_per_station and current_station < num_stations - 1:
                remaining_tasks_count = num_tasks - sum(len(st) for st in station_tasks)
                remaining_stations_count = num_stations - current_station - 1
                if remaining_tasks_count > remaining_stations_count and station_tasks[current_station]:
                    next_station = current_station + 1
                    if next_station <= ls:
                        target = next_station
                        current_station = next_station
                        current_load = 0
            
            assignment[task] = target
            station_tasks[target].append(task)
            current_load += processing_times[task]
            if target > current_station:
                current_station = target
                current_load = processing_times[task]
        
        # Make sure every station has at least one task
        empty_stations = [s for s in range(num_stations) if not station_tasks[s]]
        
        for es in sorted(empty_stations):
            best_donor = -1
            best_task = -1
            best_cost = float('inf')
            
            for s in range(num_stations):
                if len(station_tasks[s]) <= 1:
                    continue
                for task in station_tasks[s]:
                    if earliest_station[task] <= es <= latest_station[task]:
                        can_move = True
                        for p in predecessors[task]:
                            if assignment[p] > es:
                                can_move = False
                                break
                        if can_move:
                            for sc in successors[task]:
                                if assignment[sc] < es:
                                    can_move = False
                                    break
                        if can_move:
                            cost = processing_times[task]
                            if best_donor == -1 or cost < best_cost:
                                best_cost = cost
                                best_task = task
                                best_donor = s
            
            if best_donor >= 0:
                station_tasks[best_donor].remove(best_task)
                station_tasks[es].append(best_task)
                assignment[best_task] = es
        
        return assignment, station_tasks
    
    def optimize_sequences(station_tasks):
        """Find best sequence for each station."""
        station_sequences = {}
        for s in range(num_stations):
            if station_tasks[s]:
                seq, _ = find_best_sequence(station_tasks[s])
                station_sequences[s] = seq
            else:
                station_sequences[s] = []
        return station_sequences
    
    # Build initial solution
    assignment, station_tasks = build_initial_solution()
    
    # Second attempt to fill empty stations
    empty_stations = [s for s in range(num_stations) if not station_tasks[s]]
    if empty_stations:
        for es in sorted(empty_stations):
            largest = max(range(num_stations), key=lambda s: len(station_tasks[s]))
            if len(station_tasks[largest]) <= 1:
                continue
            for task in list(station_tasks[largest]):
                if earliest_station[task] <= es <= latest_station[task]:
                    can_move = True
                    for p in predecessors[task]:
                        if assignment[p] > es:
                            can_move = False
                            break
                    if can_move:
                        for sc in successors[task]:
                            if assignment[sc] < es:
                                can_move = False
                                break
                    if can_move:
                        station_tasks[largest].remove(task)
                        station_tasks[es].append(task)
                        assignment[task] = es
                        break
    
    station_sequences = optimize_sequences(station_tasks)
    best_obj, station_times = evaluate_solution(assignment, station_sequences)
    
    if logger:
        logger.log(best_obj)
    
    best_assignment = assignment[:]
    best_station_tasks = [list(st) for st in station_tasks]
    best_station_sequences = {k: list(v) for k, v in station_sequences.items()}
    
    def try_move_task(assignment, station_tasks, station_sequences, station_times):
        """Try moving a task from the bottleneck station to another."""
        bottleneck = max(range(num_stations), key=lambda s: station_times.get(s, 0))
        bottleneck_time = station_times.get(bottleneck, 0)
        
        best_improvement = 0
        best_move = None
        
        tasks_on_bottleneck = list(station_tasks[bottleneck])
        if len(tasks_on_bottleneck) <= 1:
            return None
        
        for task in tasks_on_bottleneck:
            for target_s in range(num_stations):
                if target_s == bottleneck:
                    continue
                if not (earliest_station[task] <= target_s <= latest_station[task]):
                    continue
                
                can_move = True
                for p in predecessors[task]:
                    if assignment[p] > target_s:
                        can_move = False
                        break
                if not can_move:
                    continue
                for sc in successors[task]:
                    if assignment[sc] < target_s:
                        can_move = False
                        break
                if not can_move:
                    continue
                
                new_bottleneck_tasks = [t for t in tasks_on_bottleneck if t != task]
                new_target_tasks = station_tasks[target_s] + [task]
                
                _, new_bottleneck_time = find_best_sequence(new_bottleneck_tasks)
                _, new_target_time = find_best_sequence(new_target_tasks)
                
                new_max = max(new_bottleneck_time, new_target_time)
                for s in range(num_stations):
                    if s != bottleneck and s != target_s:
                        new_max = max(new_max, station_times.get(s, 0))
                
                improvement = bottleneck_time - new_max
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_move = (task, bottleneck, target_s)
        
        return best_move
    
    def try_swap_tasks(assignment, station_tasks, station_sequences, station_times):
        """Try swapping tasks between two stations."""
        bottleneck = max(range(num_stations), key=lambda s: station_times.get(s, 0))
        bottleneck_time = station_times.get(bottleneck, 0)
        
        best_improvement = 0
        best_swap = None
        
        for task1 in station_tasks[bottleneck]:
            for target_s in range(num_stations):
                if target_s == bottleneck:
                    continue
                for task2 in station_tasks[target_s]:
                    if not (earliest_station[task1] <= target_s <= latest_station[task1]):
                        continue
                    if not (earliest_station[task2] <= bottleneck <= latest_station[task2]):
                        continue
                    
                    can_swap = True
                    orig1, orig2 = assignment[task1], assignment[task2]
                    assignment[task1] = target_s
                    assignment[task2] = bottleneck
                    
                    for p in predecessors[task1]:
                        if assignment[p] > target_s:
                            can_swap = False
                            break
                    if can_swap:
                        for sc in successors[task1]:
                            if assignment[sc] < target_s:
                                can_swap = False
                                break
                    if can_swap:
                        for p in predecessors[task2]:
                            if assignment[p] > bottleneck:
                                can_swap = False
                                break
                    if can_swap:
                        for sc in successors[task2]:
                            if assignment[sc] < bottleneck:
                                can_swap = False
                                break
                    
                    assignment[task1] = orig1
                    assignment[task2] = orig2
                    
                    if not can_swap:
                        continue
                    
                    new_bn_tasks = [t for t in station_tasks[bottleneck] if t != task1] + [task2]
                    new_ts_tasks = [t for t in station_tasks[target_s] if t != task2] + [task1]
                    
                    _, new_bn_time = find_best_sequence(new_bn_tasks)
                    _, new_ts_time = find_best_sequence(new_ts_tasks)
                    
                    new_max = max(new_bn_time, new_ts_time)
                    for s in range(num_stations):
                        if s != bottleneck and s != target_s:
                            new_max = max(new_max, station_times.get(s, 0))
                    
                    improvement = bottleneck_time - new_max
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_swap = (task1, bottleneck, task2, target_s)
        
        return best_swap
    
    def apply_move(task, from_s, to_s, assignment, station_tasks):
        assignment[task] = to_s
        station_tasks[from_s].remove(task)
        station_tasks[to_s].append(task)
    
    def apply_swap(task1, s1, task2, s2, assignment, station_tasks):
        assignment[task1] = s2
        assignment[task2] = s1
        station_tasks[s1].remove(task1)
        station_tasks[s2].remove(task2)
        station_tasks[s1].append(task2)
        station_tasks[s2].append(task1)
    
    # Greedy improvement loop
    max_iterations = 10000
    no_improve_limit = 200
    no_improve_count = 0
    
    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit - 2:
            break
        
        move = try_move_task(assignment, station_tasks, station_sequences, station_times)
        if move:
            task, from_s, to_s = move
            apply_move(task, from_s, to_s, assignment, station_tasks)
            station_sequences = optimize_sequences(station_tasks)
            obj, station_times = evaluate_solution(assignment, station_sequences)
            
            if obj < best_obj:
                best_obj = obj
                best_assignment = assignment[:]
                best_station_tasks = [list(st) for st in station_tasks]
                best_station_sequences = {k: list(v) for k, v in station_sequences.items()}
                no_improve_count = 0
                if logger:
                    logger.log(best_obj)
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
        
        if time.time() - start_time > time_limit - 2:
            break
        
        swap = try_swap_tasks(assignment, station_tasks, station_sequences, station_times)
        if swap:
            task1, s1, task2, s2 = swap
            apply_swap(task1, s1, task2, s2, assignment, station_tasks)
            station_sequences = optimize_sequences(station_tasks)
            obj, station_times = evaluate_solution(assignment, station_sequences)
            
            if obj < best_obj:
                best_obj = obj
                best_assignment = assignment[:]
                best_station_tasks = [list(st) for st in station_tasks]
                best_station_sequences = {k: list(v) for k, v in station_sequences.items()}
                no_improve_count = 0
                if logger:
                    logger.log(best_obj)
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
        
        if no_improve_count >= no_improve_limit:
            break
    
    # Simulated annealing phase
    elapsed = time.time() - start_time
    remaining_time = time_limit - elapsed - 2
    
    if remaining_time > 1:
        assignment = best_assignment[:]
        station_tasks = [list(st) for st in best_station_tasks]
        station_sequences = {k: list(v) for k, v in best_station_sequences.items()}
        _, station_times = evaluate_solution(assignment, station_sequences)
        current_obj = best_obj
        
        temperature = max(best_obj * 0.1, 1.0)
        cooling_rate = 0.9995
        
        while time.time() - start_time < time_limit - 1:
            temperature *= cooling_rate
            if temperature < 0.01:
                temperature = max(best_obj * 0.02, 0.5)
            
            action = random.random()
            
            if action < 0.6:
                # Random move
                task = random.randint(0, num_tasks - 1)
                from_s = assignment[task]
                if len(station_tasks[from_s]) <= 1:
                    continue
                
                valid_targets = []
                for s in range(num_stations):
                    if s == from_s:
                        continue
                    if earliest_station[task] <= s <= latest_station[task]:
                        can_move = True
                        for p in predecessors[task]:
                            if assignment[p] > s:
                                can_move = False
                                break
                        if can_move:
                            for sc in successors[task]:
                                if assignment[sc] < s:
                                    can_move = False
                                    break
                        if can_move:
                            valid_targets.append(s)
                
                if not valid_targets:
                    continue
                
                to_s = random.choice(valid_targets)
                
                new_from_tasks = [t for t in station_tasks[from_s] if t != task]
                new_to_tasks = station_tasks[to_s] + [task]
                
                seq_from, time_from = find_best_sequence(new_from_tasks)
                seq_to, time_to = find_best_sequence(new_to_tasks)
                
                new_max = max(time_from, time_to)
                for s in range(num_stations):
                    if s != from_s and s != to_s:
                        new_max = max(new_max, station_times.get(s, 0))
                
                delta = new_max - current_obj
                if delta < 0 or random.random() < math.exp(-delta / max(temperature, 0.001)):
                    apply_move(task, from_s, to_s, assignment, station_tasks)
                    station_sequences[from_s] = seq_from
                    station_sequences[to_s] = seq_to
                    station_times[from_s] = time_from
                    station_times[to_s] = time_to
                    current_obj = new_max
                    
                    if current_obj < best_obj:
                        best_obj = current_obj
                        best_assignment = assignment[:]
                        best_station_tasks = [list(st) for st in station_tasks]
                        best_station_sequences = {k: list(v) for k, v in station_sequences.items()}
                        if logger:
                            logger.log(best_obj)
            
            else:
                # Random swap
                if num_tasks < 2:
                    continue
                task1 = random.randint(0, num_tasks - 1)
                s1 = assignment[task1]
                
                other_stations = [s for s in range(num_stations) if s != s1 and station_tasks[s]]
                if not other_stations:
                    continue
                s2 = random.choice(other_stations)
                task2 = random.choice(station_tasks[s2])
                
                if not (earliest_station[task1] <= s2 <= latest_station[task1]):
                    continue
                if not (earliest_station[task2] <= s1 <= latest_station[task2]):
                    continue
                
                can_swap = True
                orig1, orig2 = assignment[task1], assignment[task2]
                assignment[task1] = s2
                assignment[task2] = s1
                
                for p in predecessors[task1]:
                    if assignment[p] > s2:
                        can_swap = False
                        break
                if can_swap:
                    for sc in successors[task1]:
                        if assignment[sc] < s2:
                            can_swap = False
                            break
                if can_swap:
                    for p in predecessors[task2]:
                        if assignment[p] > s1:
                            can_swap = False
                            break
                if can_swap:
                    for sc in successors[task2]:
                        if assignment[sc] < s1:
                            can_swap = False
                            break
                
                assignment[task1] = orig1
                assignment[task2] = orig2
                
                if not can_swap:
                    continue
                
                new_s1_tasks = [t for t in station_tasks[s1] if t != task1] + [task2]
                new_s2_tasks = [t for t in station_tasks[s2] if t != task2] + [task1]
                
                seq_s1, time_s1 = find_best_sequence(new_s1_tasks)
                seq_s2, time_s2 = find_best_sequence(new_s2_tasks)
                
                new_max = max(time_s1, time_s2)
                for s in range(num_stations):
                    if s != s1 and s != s2:
                        new_max = max(new_max, station_times.get(s, 0))
                
                delta = new_max - current_obj
                if delta < 0 or random.random() < math.exp(-delta / max(temperature, 0.001)):
                    apply_swap(task1, s1, task2, s2, assignment, station_tasks)
                    station_sequences[s1] = seq_s1
                    station_sequences[s2] = seq_s2
                    station_times[s1] = time_s1
                    station_times[s2] = time_s2
                    current_obj = new_max
                    
                    if current_obj < best_obj:
                        best_obj = current_obj
                        best_assignment = assignment[:]
                        best_station_tasks = [list(st) for st in station_tasks]
                        best_station_sequences = {k: list(v) for k, v in station_sequences.items()}
                        if logger:
                            logger.log(best_obj)
    
    # Build output (convert to 1-based indexing for output)
    assignment_out = {}
    for t in range(num_tasks):
        assignment_out[str(t)] = best_assignment[t] + 1  # 1-based station
    
    station_sequences_out = {}
    for s in range(num_stations):
        seq = best_station_sequences.get(s, [])
        station_sequences_out[str(s + 1)] = [t + 1 for t in seq]  # 1-based tasks
    
    solution = {
        "objective_value": float(best_obj),
        "assignment": assignment_out,
        "station_sequences": station_sequences_out,
        "cycle_time": float(best_obj)
    }
    
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()
    
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)
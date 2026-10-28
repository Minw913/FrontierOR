import json
import argparse
import time
import random
import math
import sys
from itertools import permutations
from collections import deque

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path) as f:
        instance = json.load(f)

    start_time = time.time()
    time_limit = args.time_limit

    num_tasks = instance['num_tasks']
    num_stations = instance['num_stations']
    pt = instance['processing_times']
    prec = instance['precedence_relations']
    fwd = instance['forward_setup_times']
    bwd = instance['backward_setup_times']

    # Build precedence info - convert from 1-based to 0-based
    preds = [[] for _ in range(num_tasks)]
    succs = [[] for _ in range(num_tasks)]
    for pair in prec:
        i, j = pair[0] - 1, pair[1] - 1  # Convert to 0-based
        preds[j].append(i)
        succs[i].append(j)

    # Topological sort
    in_deg = [len(preds[i]) for i in range(num_tasks)]
    queue = deque([i for i in range(num_tasks) if in_deg[i] == 0])
    topo = []
    temp_deg = in_deg[:]
    while queue:
        node = queue.popleft()
        topo.append(node)
        for s in succs[node]:
            temp_deg[s] -= 1
            if temp_deg[s] == 0:
                queue.append(s)

    # If topological sort didn't cover all tasks (shouldn't happen with valid input)
    if len(topo) < num_tasks:
        for t in range(num_tasks):
            if t not in topo:
                topo.append(t)

    def station_time(seq):
        if not seq:
            return 0
        total = sum(pt[t] for t in seq)
        for i in range(len(seq) - 1):
            total += fwd[seq[i]][seq[i + 1]]
        total += bwd[seq[-1]][seq[0]]
        return total

    def sequence_station(tasks):
        """Find best ordering of tasks at a station respecting precedence."""
        n = len(tasks)
        if n == 0:
            return []
        if n == 1:
            return tasks[:]

        task_set = set(tasks)
        local_preds = {t: [] for t in tasks}
        local_succs = {t: [] for t in tasks}
        for i, j in prec:
            i0, j0 = i - 1, j - 1
            if i0 in task_set and j0 in task_set:
                local_preds[j0].append(i0)
                local_succs[i0].append(j0)

        if n <= 8:
            best_cost = float('inf')
            best_order = None
            for perm in permutations(tasks):
                pos = {t: idx for idx, t in enumerate(perm)}
                valid = True
                for i, j in prec:
                    i0, j0 = i - 1, j - 1
                    if i0 in task_set and j0 in task_set:
                        if pos[i0] > pos[j0]:
                            valid = False
                            break
                if not valid:
                    continue
                cost = 0
                for k in range(n - 1):
                    cost += fwd[perm[k]][perm[k + 1]]
                cost += bwd[perm[-1]][perm[0]]
                if cost < best_cost:
                    best_cost = cost
                    best_order = list(perm)
            return best_order if best_order else tasks[:]
        else:
            # Greedy with precedence constraints
            in_degree = {t: len(local_preds[t]) for t in tasks}
            best_cost = float('inf')
            best_order = None

            for start in tasks:
                if in_degree[start] > 0:
                    continue
                order = [start]
                in_deg_copy = dict(in_degree)
                for s in local_succs[start]:
                    in_deg_copy[s] -= 1
                current = start
                visited = {start}

                while len(order) < n:
                    avail = [t for t in tasks if t not in visited and in_deg_copy[t] == 0]
                    if not avail:
                        break
                    next_t = min(avail, key=lambda t: fwd[current][t])
                    order.append(next_t)
                    visited.add(next_t)
                    for s in local_succs[next_t]:
                        in_deg_copy[s] -= 1
                    current = next_t

                if len(order) == n:
                    cost = 0
                    for k in range(n - 1):
                        cost += fwd[order[k]][order[k + 1]]
                    cost += bwd[order[-1]][order[0]]
                    if cost < best_cost:
                        best_cost = cost
                        best_order = order[:]

            # Or-opt improvement
            if best_order and n > 2:
                improved = True
                max_iter = 50
                iter_count = 0
                while improved and iter_count < max_iter:
                    improved = False
                    iter_count += 1
                    for i in range(n):
                        for j in range(i + 2, min(i + 5, n)):
                            for k in range(j + 1, min(j + 4, n + 1)):
                                seg = best_order[i:j]
                                new_order = best_order[:i] + best_order[j:k] + seg + best_order[k:]
                                pos = {t: idx for idx, t in enumerate(new_order)}
                                valid = True
                                for a, b in prec:
                                    a0, b0 = a - 1, b - 1
                                    if a0 in task_set and b0 in task_set:
                                        if pos[a0] > pos[b0]:
                                            valid = False
                                            break
                                if not valid:
                                    continue
                                old_cost = sum(fwd[best_order[x]][best_order[x + 1]] for x in range(n - 1)) + bwd[best_order[-1]][best_order[0]]
                                new_cost = sum(fwd[new_order[x]][new_order[x + 1]] for x in range(n - 1)) + bwd[new_order[-1]][new_order[0]]
                                if new_cost < old_cost:
                                    best_order = new_order
                                    improved = True

            return best_order if best_order else tasks[:]

    def evaluate_assignment(assignment):
        """Evaluate a complete assignment, returning (cycle_time, sequences)."""
        stations = [[] for _ in range(num_stations)]
        for task in range(num_tasks):
            stations[assignment[task]].append(task)

        for s in range(num_stations):
            if not stations[s]:
                return float('inf'), None

        for i, j in prec:
            i0, j0 = i - 1, j - 1
            if assignment[i0] > assignment[j0]:
                return float('inf'), None

        sequences = {}
        max_time = 0
        for s in range(num_stations):
            seq = sequence_station(stations[s])
            sequences[s] = seq
            t = station_time(seq)
            if t > max_time:
                max_time = t

        return max_time, sequences

    def get_feasible_range(task, assignment):
        """Get feasible station range for a task."""
        min_s = 0
        max_s = num_stations - 1
        for p in preds[task]:
            if p in assignment:
                min_s = max(min_s, assignment[p])
        for s in succs[task]:
            if s in assignment:
                max_s = min(max_s, assignment[s])
        return min_s, max_s

    def construct_greedy(variant=0):
        """Construct initial solution using greedy approach."""
        assignment = {}
        station_loads = [0] * num_stations

        if variant == 0:
            # Process in topological order, assign to least loaded feasible station
            for task in topo:
                min_s, max_s = get_feasible_range(task, assignment)
                if min_s > max_s:
                    min_s = max_s = 0
                best_s = min_s
                best_load = float('inf')
                for s in range(min_s, max_s + 1):
                    if station_loads[s] < best_load:
                        best_load = station_loads[s]
                        best_s = s
                assignment[task] = best_s
                station_loads[best_s] += pt[task]

        elif variant == 1:
            # Process in topological order, assign to minimize max load
            for task in topo:
                min_s, max_s = get_feasible_range(task, assignment)
                if min_s > max_s:
                    min_s = max_s = 0
                best_s = min_s
                best_max = float('inf')
                for s in range(min_s, max_s + 1):
                    new_max = max(station_loads[s] + pt[task], max(station_loads) if station_loads else 0)
                    if new_max < best_max:
                        best_max = new_max
                        best_s = s
                assignment[task] = best_s
                station_loads[best_s] += pt[task]

        elif variant == 2:
            # Assign considering setup times - spread tasks more evenly
            target_load = sum(pt) / num_stations
            for task in topo:
                min_s, max_s = get_feasible_range(task, assignment)
                if min_s > max_s:
                    min_s = max_s = 0
                best_s = min_s
                best_score = float('inf')
                for s in range(min_s, max_s + 1):
                    score = abs(station_loads[s] + pt[task] - target_load)
                    if score < best_score:
                        best_score = score
                        best_s = s
                assignment[task] = best_s
                station_loads[best_s] += pt[task]

        elif variant == 3:
            # Assign to earliest possible station (pack early)
            for task in topo:
                min_s, max_s = get_feasible_range(task, assignment)
                if min_s > max_s:
                    min_s = max_s = 0
                assignment[task] = min_s
                station_loads[min_s] += pt[task]

        elif variant == 4:
            # Assign to latest possible station (pack late)
            for task in topo:
                min_s, max_s = get_feasible_range(task, assignment)
                if min_s > max_s:
                    min_s = max_s = 0
                assignment[task] = max_s
                station_loads[max_s] += pt[task]

        # Ensure all stations have at least one task
        for s in range(num_stations):
            if not any(assignment[t] == s for t in range(num_tasks)):
                for task in reversed(topo):
                    old_s = assignment[task]
                    if old_s == s:
                        continue
                    min_s, max_s = get_feasible_range(task, assignment)
                    if min_s <= s <= max_s:
                        count_old = sum(1 for t in range(num_tasks) if assignment[t] == old_s)
                        if count_old > 1:
                            assignment[task] = s
                            break

        return assignment

    def format_solution(assignment, sequences, obj_val):
        sol = {
            "objective_value": float(obj_val),
            "assignment": {str(t + 1): int(assignment[t]) + 1 for t in range(num_tasks)},
            "station_sequences": {str(s + 1): [int(t) + 1 for t in sequences[s]] for s in range(num_stations)},
            "cycle_time": float(obj_val)
        }
        return sol

    # Initial solutions from multiple constructive heuristics
    best_cost = float('inf')
    best_assignment = None
    best_seqs = None

    for variant in range(5):
        if time.time() - start_time > time_limit * 0.9:
            break
        assignment = construct_greedy(variant)
        cost, seqs = evaluate_assignment(assignment)
        if cost < best_cost and cost < float('inf'):
            best_cost = cost
            best_assignment = dict(assignment)
            best_seqs = seqs
            if logger:
                sol_dict = format_solution(best_assignment, best_seqs, best_cost)
                logger.log_solution(best_cost, sol_dict)

    # Also try random assignments
    for trial in range(20):
        if time.time() - start_time > time_limit * 0.8:
            break
        assignment = {}
        for task in topo:
            min_s, max_s = get_feasible_range(task, assignment)
            if min_s <= max_s:
                assignment[task] = random.randint(min_s, max_s)
            else:
                assignment[task] = 0

        # Fix empty stations
        valid = True
        for s in range(num_stations):
            if not any(assignment.get(t, -1) == s for t in range(num_tasks)):
                placed = False
                for task in reversed(topo):
                    old_s = assignment[task]
                    min_s2, max_s2 = get_feasible_range(task, assignment)
                    if min_s2 <= s <= max_s2 and old_s != s:
                        count_old = sum(1 for t in range(num_tasks) if assignment[t] == old_s)
                        if count_old > 1:
                            assignment[task] = s
                            placed = True
                            break
                if not placed:
                    valid = False
                    break

        if valid:
            cost, seqs = evaluate_assignment(assignment)
            if cost < best_cost and cost < float('inf'):
                best_cost = cost
                best_assignment = dict(assignment)
                best_seqs = seqs
                if logger:
                    sol_dict = format_solution(best_assignment, best_seqs, best_cost)
                    logger.log_solution(best_cost, sol_dict)

    if best_assignment is None:
        # Fallback: simple round-robin
        assignment = {}
        for idx, task in enumerate(topo):
            assignment[task] = idx % num_stations
        best_cost, best_seqs = evaluate_assignment(assignment)
        best_assignment = assignment
        if best_cost == float('inf'):
            # Last resort
            best_cost = 0
            best_seqs = {}
            for s in range(num_stations):
                best_seqs[s] = []

    # Simulated Annealing improvement
    sa_start = time.time()
    sa_budget = time_limit - (sa_start - start_time) - 0.5

    if sa_budget > 1 and best_cost < float('inf'):
        current_assignment = dict(best_assignment)
        current_cost = best_cost

        T = max(best_cost * 0.05, 1.0)
        cooling = 0.99995
        no_improve_count = 0

        while time.time() - start_time < time_limit - 0.5:
            # Choose move type
            move_type = random.random()

            old_assignment = dict(current_assignment)

            if move_type < 0.6:
                # Move a single task
                task = random.randint(0, num_tasks - 1)
                old_station = current_assignment[task]
                min_s, max_s = get_feasible_range(task, current_assignment)
                options = [s for s in range(min_s, max_s + 1) if s != old_station]
                if not options:
                    continue
                if sum(1 for t in range(num_tasks) if current_assignment[t] == old_station) <= 1:
                    continue
                new_station = random.choice(options)
                current_assignment[task] = new_station

            elif move_type < 0.8:
                # Swap two tasks
                t1 = random.randint(0, num_tasks - 1)
                t2 = random.randint(0, num_tasks - 1)
                if t1 == t2:
                    continue
                s1, s2 = current_assignment[t1], current_assignment[t2]
                if s1 == s2:
                    continue
                current_assignment[t1] = s2
                current_assignment[t2] = s1

            else:
                # Move task to adjacent station (more local)
                task = random.randint(0, num_tasks - 1)
                old_station = current_assignment[task]
                direction = random.choice([-1, 1])
                new_station = old_station + direction
                if new_station < 0 or new_station >= num_stations:
                    continue
                min_s, max_s = get_feasible_range(task, current_assignment)
                if not (min_s <= new_station <= max_s):
                    continue
                if sum(1 for t in range(num_tasks) if current_assignment[t] == old_station) <= 1:
                    continue
                current_assignment[task] = new_station

            new_cost, new_seqs = evaluate_assignment(current_assignment)

            if new_cost < float('inf'):
                delta = new_cost - current_cost
                if delta <= 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_assignment = dict(current_assignment)
                        best_seqs = new_seqs
                        no_improve_count = 0
                        if logger:
                            sol_dict = format_solution(best_assignment, best_seqs, best_cost)
                            logger.log_solution(best_cost, sol_dict)
                    else:
                        no_improve_count += 1
                else:
                    current_assignment = old_assignment
            else:
                current_assignment = old_assignment

            T *= cooling

            if no_improve_count > 5000:
                T = max(best_cost * 0.03, 0.5)
                no_improve_count = 0

    # Final local search: try improving sequencing of each station
    if time.time() - start_time < time_limit - 0.2:
        for s in range(num_stations):
            tasks_at_s = [t for t in range(num_tasks) if best_assignment[t] == s]
            if len(tasks_at_s) > 1:
                new_seq = sequence_station(tasks_at_s)
                best_seqs[s] = new_seq

        # Recompute objective
        max_time = 0
        for s in range(num_stations):
            t = station_time(best_seqs[s])
            max_time = max(max_time, t)
        best_cost = max_time

    # Write solution
    solution = format_solution(best_assignment, best_seqs, best_cost)

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

    if logger:
        logger.log_solution(best_cost, solution)

if __name__ == '__main__':
    main()
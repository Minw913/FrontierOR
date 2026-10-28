import argparse
import json
import time
import random
import math
from solution_logger import SolutionLogger

# ---------- Helper functions ----------

def station_time(seq, processing_times, forward_setup, backward_setup):
    if not seq:
        return 0
    t = sum(processing_times[x] for x in seq)
    for i in range(len(seq) - 1):
        t += forward_setup[seq[i]][seq[i+1]]
    t += backward_setup[seq[-1]][seq[0]]
    return t


def optimize_sequence_heuristic(tasks, global_preds, global_succs, processing_times, forward_setup, backward_setup):
    if not tasks:
        return 0, []
    if len(tasks) == 1:
        t = tasks[0]
        return processing_times[t] + backward_setup[t][t], [t]

    task_set = set(tasks)
    pred_in_station = {}
    succ_in_station = {}
    for t in tasks:
        pred_in_station[t] = [p for p in global_preds[t] if p in task_set]
        succ_in_station[t] = [s for s in global_succs[t] if s in task_set]

    # Topological sort (Kahn's algorithm)
    in_degree = {t: len(pred_in_station[t]) for t in tasks}
    queue = [t for t in tasks if in_degree[t] == 0]
    topo = []
    while queue:
        t = queue.pop(0)
        topo.append(t)
        for s in succ_in_station[t]:
            in_degree[s] -= 1
            if in_degree[s] == 0:
                queue.append(s)

    # Insertion heuristic
    seq = []
    for t in topo:
        if not pred_in_station[t]:
            valid_starts = 0
        else:
            last_pred_idx = -1
            for i, s in enumerate(seq):
                if s in pred_in_station[t]:
                    last_pred_idx = i
            valid_starts = last_pred_idx + 1
        best_pos = valid_starts
        best_setup = float('inf')
        for pos in range(valid_starts, len(seq) + 1):
            new_seq = seq[:pos] + [t] + seq[pos:]
            setup = 0
            for i in range(len(new_seq) - 1):
                setup += forward_setup[new_seq[i]][new_seq[i+1]]
            setup += backward_setup[new_seq[-1]][new_seq[0]]
            if setup < best_setup:
                best_setup = setup
                best_pos = pos
        seq = seq[:best_pos] + [t] + seq[best_pos:]

    # Or-opt local search (limited passes)
    passes = 0
    improved = True
    while improved and passes < 3:
        improved = False
        passes += 1
        for idx in range(len(seq)):
            t = seq[idx]
            temp_seq = seq[:idx] + seq[idx+1:]
            if not pred_in_station[t]:
                valid_starts = 0
            else:
                last_pred_idx = -1
                for i, s in enumerate(temp_seq):
                    if s in pred_in_station[t]:
                        last_pred_idx = i
                valid_starts = last_pred_idx + 1
            best_pos = idx
            current_setup = 0
            for i in range(len(seq) - 1):
                current_setup += forward_setup[seq[i]][seq[i+1]]
            current_setup += backward_setup[seq[-1]][seq[0]]
            best_setup = current_setup
            for pos in range(valid_starts, len(temp_seq) + 1):
                new_seq = temp_seq[:pos] + [t] + temp_seq[pos:]
                setup = 0
                for i in range(len(new_seq) - 1):
                    setup += forward_setup[new_seq[i]][new_seq[i+1]]
                setup += backward_setup[new_seq[-1]][new_seq[0]]
                if setup < best_setup - 1e-9:
                    best_setup = setup
                    best_pos = pos
            if best_pos != idx:
                seq = temp_seq[:best_pos] + [t] + temp_seq[best_pos:]
                improved = True
                break  # restart loop on modified sequence

    proc_sum = sum(processing_times[t] for t in tasks)
    setup = 0
    for i in range(len(seq) - 1):
        setup += forward_setup[seq[i]][seq[i+1]]
    setup += backward_setup[seq[-1]][seq[0]]
    return proc_sum + setup, seq


def optimize_sequence_exact(tasks, global_preds, global_succs, processing_times, forward_setup, backward_setup):
    if not tasks:
        return 0, []
    if len(tasks) == 1:
        t = tasks[0]
        return processing_times[t] + backward_setup[t][t], [t]

    task_set = set(tasks)
    pred_in_station = {}
    for t in tasks:
        pred_in_station[t] = [p for p in global_preds[t] if p in task_set]

    task_list = list(tasks)
    k = len(task_list)
    task_to_idx = {t: i for i, t in enumerate(task_list)}

    pred_mask = [0] * k
    for i, t in enumerate(task_list):
        mask = 0
        for p in pred_in_station[t]:
            mask |= (1 << task_to_idx[p])
        pred_mask[i] = mask

    proc_sum = sum(processing_times[t] for t in task_list)
    best_setup = float('inf')
    best_seq = None
    full_mask = (1 << k) - 1

    for start_idx in range(k):
        dp = [[float('inf')] * k for _ in range(1 << k)]
        parent = [[-1] * k for _ in range(1 << k)]
        dp[1 << start_idx][start_idx] = 0

        for mask in range(1 << k):
            if not (mask & (1 << start_idx)):
                continue
            for last in range(k):
                if not (mask & (1 << last)):
                    continue
                cost = dp[mask][last]
                if cost == float('inf'):
                    continue
                for nxt in range(k):
                    if mask & (1 << nxt):
                        continue
                    if (pred_mask[nxt] & mask) != pred_mask[nxt]:
                        continue
                    new_mask = mask | (1 << nxt)
                    new_cost = cost + forward_setup[task_list[last]][task_list[nxt]]
                    if new_cost < dp[new_mask][nxt]:
                        dp[new_mask][nxt] = new_cost
                        parent[new_mask][nxt] = last

        for last in range(k):
            if dp[full_mask][last] == float('inf'):
                continue
            total_setup = dp[full_mask][last] + backward_setup[task_list[last]][task_list[start_idx]]
            if total_setup < best_setup:
                best_setup = total_setup
                seq = []
                mask = full_mask
                curr = last
                while curr != -1:
                    seq.append(task_list[curr])
                    prev = parent[mask][curr]
                    mask ^= (1 << curr)
                    curr = prev
                seq.reverse()
                best_seq = seq

    if best_seq is None:
        return optimize_sequence_heuristic(tasks, global_preds, global_succs, processing_times, forward_setup, backward_setup)
    return proc_sum + best_setup, best_seq


def random_topo_order(preds, succs, num_tasks):
    in_degree = [len(p) for p in preds]
    queue = [i for i in range(num_tasks) if in_degree[i] == 0]
    random.shuffle(queue)
    topo = []
    while queue:
        t = queue.pop()
        topo.append(t)
        for s in succs[t]:
            in_degree[s] -= 1
            if in_degree[s] == 0:
                queue.append(s)
        random.shuffle(queue)
    return topo


def dp_partition(topo, proc, num_stations):
    N = len(topo)
    S = num_stations
    proc_sum = [0] * (N + 1)
    for i in range(N):
        proc_sum[i+1] = proc_sum[i] + proc[topo[i]]
    INF = float('inf')
    dp = [[INF] * (S + 1) for _ in range(N + 1)]
    dp[0][0] = 0
    for i in range(1, N + 1):
        for s in range(1, min(i, S) + 1):
            for j in range(s - 1, i):
                load = proc_sum[i] - proc_sum[j]
                val = max(dp[j][s-1], load)
                if val < dp[i][s]:
                    dp[i][s] = val
    cuts = []
    i = N
    s = S
    while s > 0:
        for j in range(s - 1, i):
            load = proc_sum[i] - proc_sum[j]
            if dp[j][s-1] != INF and max(dp[j][s-1], load) == dp[i][s]:
                cuts.append((j, i))
                i = j
                s -= 1
                break
    cuts.reverse()
    assignment = [0] * N
    for s_idx, (start, end) in enumerate(cuts):
        for idx in range(start, end):
            task = topo[idx]
            assignment[task] = s_idx
    return assignment


def greedy_partition(topo, proc, num_stations):
    N = len(topo)
    assignment = [0] * N
    total_proc = sum(proc)
    target = total_proc / num_stations
    current_load = 0
    current_station = 0
    for i, t in enumerate(topo):
        if (current_load > 0 and current_load + proc[t] > target and
            current_station < num_stations - 1 and
            (N - i) > (num_stations - current_station - 1)):
            current_station += 1
            current_load = 0
        assignment[t] = current_station
        current_load += proc[t]
    return assignment


def build_solution_dict(assignment, seqs, obj, num_tasks, num_stations):
    sol = {
        "objective_value": float(obj),
        "assignment": {},
        "station_sequences": {},
        "cycle_time": float(obj)
    }
    for t in range(num_tasks):
        sol["assignment"][str(t + 1)] = int(assignment[t] + 1)
    for s in range(num_stations):
        sol["station_sequences"][str(s + 1)] = [int(t + 1) for t in seqs[s]]
    return sol


# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    random.seed(0)
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path) as f:
        instance = json.load(f)

    num_tasks = instance['num_tasks']
    num_stations = instance['num_stations']
    processing_times = instance['processing_times']
    precedence_relations = instance['precedence_relations']
    forward_setup = instance['forward_setup_times']
    backward_setup = instance['backward_setup_times']

    preds = [[] for _ in range(num_tasks)]
    succs = [[] for _ in range(num_tasks)]
    for i, j in precedence_relations:
        i0 = i - 1
        j0 = j - 1
        preds[j0].append(i0)
        succs[i0].append(j0)

    # Cached heuristic for speed
    heuristic_cache = {}
    def opt_seq_heuristic(tasks):
        key = tuple(sorted(tasks))
        if key in heuristic_cache:
            return heuristic_cache[key]
        res = optimize_sequence_heuristic(tasks, preds, succs, processing_times, forward_setup, backward_setup)
        heuristic_cache[key] = res
        return res

    start_time = time.time()
    best_assignment = None
    best_obj = float('inf')
    best_seqs = None

    # Initial solution construction
    num_initial_tries = 10 if num_tasks <= 200 else 3
    for _ in range(num_initial_tries):
        if time.time() - start_time > args.time_limit * 0.2:
            break
        topo = random_topo_order(preds, succs, num_tasks)
        if num_tasks <= 200:
            assign = dp_partition(topo, processing_times, num_stations)
        else:
            assign = greedy_partition(topo, processing_times, num_stations)

        station_tasks = [[] for _ in range(num_stations)]
        for t in range(num_tasks):
            station_tasks[assign[t]].append(t)
        times = [0] * num_stations
        seqs = [[] for _ in range(num_stations)]
        for s in range(num_stations):
            time_s, seq_s = opt_seq_heuristic(station_tasks[s])
            times[s] = time_s
            seqs[s] = seq_s
        obj = max(times)
        if obj < best_obj:
            best_obj = obj
            best_assignment = assign[:]
            best_seqs = [seq[:] for seq in seqs]
            if logger:
                sol_dict = build_solution_dict(best_assignment, best_seqs, best_obj, num_tasks, num_stations)
                logger.log_solution(best_obj, sol_dict)

    if best_assignment is None:
        # Fallback: all tasks to station 0 in topological order
        in_degree = [len(preds[t]) for t in range(num_tasks)]
        queue = [t for t in range(num_tasks) if in_degree[t] == 0]
        topo = []
        while queue:
            t = queue.pop(0)
            topo.append(t)
            for s in succs[t]:
                in_degree[s] -= 1
                if in_degree[s] == 0:
                    queue.append(s)
        best_assignment = [0] * num_tasks
        best_seqs = [topo]
        best_obj = station_time(topo, processing_times, forward_setup, backward_setup)
        if logger:
            sol_dict = build_solution_dict(best_assignment, best_seqs, best_obj, num_tasks, num_stations)
            logger.log_solution(best_obj, sol_dict)

    # Simulated Annealing on assignment
    current_assignment = best_assignment[:]
    current_station_tasks = [[] for _ in range(num_stations)]
    for t in range(num_tasks):
        current_station_tasks[current_assignment[t]].append(t)
    current_times = [0] * num_stations
    current_seqs = [[] for _ in range(num_stations)]
    for s in range(num_stations):
        current_times[s], current_seqs[s] = opt_seq_heuristic(current_station_tasks[s])
    current_obj = max(current_times)

    if num_stations > 1 and time.time() - start_time < args.time_limit - 1.0:
        T = max(1.0, current_obj * 0.2)
        alpha = 0.9995
        iteration = 0
        last_improvement = 0

        while time.time() - start_time < args.time_limit - 1.0:
            iteration += 1
            move_type = random.random()

            if move_type < 0.6:
                # Move task
                i = random.randint(0, num_tasks - 1)
                s1 = current_assignment[i]
                if len(current_station_tasks[s1]) == 1:
                    continue
                s2 = random.randint(0, num_stations - 1)
                if s2 == s1:
                    continue
                feasible = True
                for p in preds[i]:
                    if current_assignment[p] > s2:
                        feasible = False
                        break
                if feasible:
                    for s in succs[i]:
                        if s2 > current_assignment[s]:
                            feasible = False
                            break
                if not feasible:
                    continue

                new_tasks_s1 = [t for t in current_station_tasks[s1] if t != i]
                new_tasks_s2 = current_station_tasks[s2] + [i]
                new_time_s1, new_seq_s1 = opt_seq_heuristic(new_tasks_s1)
                new_time_s2, new_seq_s2 = opt_seq_heuristic(new_tasks_s2)
                new_obj = max(new_time_s1, new_time_s2)
                for s in range(num_stations):
                    if s != s1 and s != s2 and current_times[s] > new_obj:
                        new_obj = current_times[s]
                delta = new_obj - current_obj
                if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-6)):
                    current_assignment[i] = s2
                    current_station_tasks[s1] = new_tasks_s1
                    current_station_tasks[s2] = new_tasks_s2
                    current_times[s1] = new_time_s1
                    current_times[s2] = new_time_s2
                    current_seqs[s1] = new_seq_s1
                    current_seqs[s2] = new_seq_s2
                    current_obj = new_obj
                    if current_obj < best_obj:
                        best_obj = current_obj
                        best_assignment = current_assignment[:]
                        best_seqs = [seq[:] for seq in current_seqs]
                        last_improvement = iteration
                        if logger:
                            sol_dict = build_solution_dict(best_assignment, best_seqs, best_obj, num_tasks, num_stations)
                            logger.log_solution(best_obj, sol_dict)

            elif move_type < 0.8:
                # Swap tasks
                i = random.randint(0, num_tasks - 1)
                j = random.randint(0, num_tasks - 1)
                if i == j:
                    continue
                s1 = current_assignment[i]
                s2 = current_assignment[j]
                if s1 == s2:
                    continue
                feasible = True
                for p in preds[i]:
                    if p != j and current_assignment[p] > s2:
                        feasible = False
                        break
                if feasible:
                    for s in succs[i]:
                        if s != j and s2 > current_assignment[s]:
                            feasible = False
                            break
                if feasible:
                    for p in preds[j]:
                        if p != i and current_assignment[p] > s1:
                            feasible = False
                            break
                if feasible:
                    for s in succs[j]:
                        if s != i and s1 > current_assignment[s]:
                            feasible = False
                            break
                if not feasible:
                    continue

                new_tasks_s1 = [t for t in current_station_tasks[s1] if t != i] + [j]
                new_tasks_s2 = [t for t in current_station_tasks[s2] if t != j] + [i]
                new_time_s1, new_seq_s1 = opt_seq_heuristic(new_tasks_s1)
                new_time_s2, new_seq_s2 = opt_seq_heuristic(new_tasks_s2)
                new_obj = max(new_time_s1, new_time_s2)
                for s in range(num_stations):
                    if s != s1 and s != s2 and current_times[s] > new_obj:
                        new_obj = current_times[s]
                delta = new_obj - current_obj
                if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-6)):
                    current_assignment[i] = s2
                    current_assignment[j] = s1
                    current_station_tasks[s1] = new_tasks_s1
                    current_station_tasks[s2] = new_tasks_s2
                    current_times[s1] = new_time_s1
                    current_times[s2] = new_time_s2
                    current_seqs[s1] = new_seq_s1
                    current_seqs[s2] = new_seq_s2
                    current_obj = new_obj
                    if current_obj < best_obj:
                        best_obj = current_obj
                        best_assignment = current_assignment[:]
                        best_seqs = [seq[:] for seq in current_seqs]
                        last_improvement = iteration
                        if logger:
                            sol_dict = build_solution_dict(best_assignment, best_seqs, best_obj, num_tasks, num_stations)
                            logger.log_solution(best_obj, sol_dict)

            else:
                # Move from max station to min station
                if num_stations < 2:
                    continue
                max_time = -1
                min_time = float('inf')
                s_max = -1
                s_min = -1
                for s in range(num_stations):
                    if current_times[s] > max_time:
                        max_time = current_times[s]
                        s_max = s
                    if current_times[s] < min_time:
                        min_time = current_times[s]
                        s_min = s
                if s_max == -1 or s_min == -1 or s_max == s_min:
                    continue
                if len(current_station_tasks[s_max]) == 1:
                    continue
                i = random.choice(current_station_tasks[s_max])
                s1 = s_max
                s2 = s_min
                feasible = True
                for p in preds[i]:
                    if current_assignment[p] > s2:
                        feasible = False
                        break
                if feasible:
                    for s in succs[i]:
                        if s2 > current_assignment[s]:
                            feasible = False
                            break
                if not feasible:
                    continue

                new_tasks_s1 = [t for t in current_station_tasks[s1] if t != i]
                new_tasks_s2 = current_station_tasks[s2] + [i]
                new_time_s1, new_seq_s1 = opt_seq_heuristic(new_tasks_s1)
                new_time_s2, new_seq_s2 = opt_seq_heuristic(new_tasks_s2)
                new_obj = max(new_time_s1, new_time_s2)
                for s in range(num_stations):
                    if s != s1 and s != s2 and current_times[s] > new_obj:
                        new_obj = current_times[s]
                delta = new_obj - current_obj
                if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-6)):
                    current_assignment[i] = s2
                    current_station_tasks[s1] = new_tasks_s1
                    current_station_tasks[s2] = new_tasks_s2
                    current_times[s1] = new_time_s1
                    current_times[s2] = new_time_s2
                    current_seqs[s1] = new_seq_s1
                    current_seqs[s2] = new_seq_s2
                    current_obj = new_obj
                    if current_obj < best_obj:
                        best_obj = current_obj
                        best_assignment = current_assignment[:]
                        best_seqs = [seq[:] for seq in current_seqs]
                        last_improvement = iteration
                        if logger:
                            sol_dict = build_solution_dict(best_assignment, best_seqs, best_obj, num_tasks, num_stations)
                            logger.log_solution(best_obj, sol_dict)

            T *= alpha
            if T < 0.01:
                T = 0.01
            if iteration - last_improvement > 5000:
                T = max(1.0, current_obj * 0.1)
                last_improvement = iteration

    # Final exact sequence optimization (if time permits)
    if best_assignment is not None and time.time() - start_time < args.time_limit - 0.5:
        final_seqs = []
        final_times = []
        for s in range(num_stations):
            tasks = [t for t in range(num_tasks) if best_assignment[t] == s]
            if len(tasks) <= 11:
                time_s, seq_s = optimize_sequence_exact(tasks, preds, succs, processing_times, forward_setup, backward_setup)
            else:
                time_s, seq_s = optimize_sequence_heuristic(tasks, preds, succs, processing_times, forward_setup, backward_setup)
            final_seqs.append(seq_s)
            final_times.append(time_s)
        final_obj = max(final_times)
        if final_obj < best_obj:
            best_obj = final_obj
            best_seqs = final_seqs
            if logger:
                sol_dict = build_solution_dict(best_assignment, best_seqs, best_obj, num_tasks, num_stations)
                logger.log_solution(best_obj, sol_dict)

    # Output
    sol_dict = build_solution_dict(best_assignment, best_seqs, best_obj, num_tasks, num_stations)
    with open(args.solution_path, 'w') as f:
        json.dump(sol_dict, f, indent=2)


if __name__ == '__main__':
    main()
import json
import argparse
import time
import sys
import random
from solution_logger import SolutionLogger


def compute_prefix_sums(processing_times, n_jobs, n_machines):
    """Compute prefix sums of processing times for each job."""
    prefix = [[0] * n_jobs for _ in range(n_machines + 1)]
    for j in range(n_jobs):
        for m in range(n_machines):
            prefix[m + 1][j] = prefix[m][j] + processing_times[m][j]
    return prefix


def compute_distance_matrix(processing_times, n_jobs, n_machines, prefix):
    """Compute d(i,j) = minimum gap between start times when j follows i."""
    dist = [[0] * n_jobs for _ in range(n_jobs)]
    for i in range(n_jobs):
        for j in range(n_jobs):
            if i == j:
                dist[i][j] = 0
                continue
            max_val = 0
            for m in range(n_machines):
                val = prefix[m + 1][i] - prefix[m][j]
                if val > max_val:
                    max_val = val
            dist[i][j] = max_val
    return dist


def compute_start_times(sequence, dist):
    """Compute start times for each position in the sequence."""
    n = len(sequence)
    starts = [0] * n
    for idx in range(1, n):
        starts[idx] = starts[idx - 1] + dist[sequence[idx - 1]][sequence[idx]]
    return starts


def compute_makespan_from_starts(starts, sequence, prefix, n_machines):
    """Compute makespan given start times."""
    max_completion = 0
    for idx, job in enumerate(sequence):
        completion = starts[idx] + prefix[n_machines][job]
        if completion > max_completion:
            max_completion = completion
    return max_completion


def compute_makespan(sequence, dist, prefix, n_machines):
    """Compute makespan for a given sequence."""
    starts = compute_start_times(sequence, dist)
    return compute_makespan_from_starts(starts, sequence, prefix, n_machines)


def check_due_dates(sequence, dist, prefix, n_machines, due_dates):
    """Check if all due dates are satisfied."""
    current_time = 0
    for idx, job in enumerate(sequence):
        if idx > 0:
            current_time += dist[sequence[idx - 1]][job]
        completion = current_time + prefix[n_machines][job]
        if completion > due_dates[job]:
            return False
    return True


def evaluate_sequence(sequence, dist, prefix, n_machines, due_dates):
    """Evaluate a sequence: return (makespan, feasible)."""
    starts = compute_start_times(sequence, dist)
    feasible = True
    max_completion = 0
    for idx, job in enumerate(sequence):
        completion = starts[idx] + prefix[n_machines][job]
        if completion > due_dates[job]:
            feasible = False
            break
        if completion > max_completion:
            max_completion = completion
    return max_completion, feasible


def nearest_neighbor(dist, n_jobs, prefix, n_machines, due_dates, criterion='distance'):
    """Nearest neighbor construction heuristic."""
    best_solution = None
    best_makespan = float('inf')

    for start in range(n_jobs):
        unvisited = set(range(n_jobs)) - {start}
        sequence = [start]
        current = start
        current_time = 0
        feasible = True

        while unvisited:
            best_next = None
            best_score = float('inf')

            for j in unvisited:
                d = dist[current][j]
                new_time = current_time + d
                completion = new_time + prefix[n_machines][j]
                if completion <= due_dates[j]:
                    if criterion == 'distance':
                        score = d
                    elif criterion == 'slack':
                        score = due_dates[j] - completion
                    elif criterion == 'combined':
                        score = d + 0.1 * (due_dates[j] - completion)
                    else:
                        score = d
                    if score < best_score:
                        best_score = score
                        best_next = j

            if best_next is None:
                feasible = False
                break

            current_time += dist[current][best_next]
            sequence.append(best_next)
            unvisited.remove(best_next)
            current = best_next

        if feasible:
            makespan = compute_makespan(sequence, dist, prefix, n_machines)
            if makespan < best_makespan:
                best_makespan = makespan
                best_solution = sequence[:]

    return best_solution, best_makespan


def insertion_heuristic(dist, n_jobs, prefix, n_machines, due_dates, time_limit, start_time):
    """Cheapest insertion heuristic."""
    best_solution = None
    best_makespan = float('inf')

    for start in range(n_jobs):
        if time.time() - start_time > time_limit * 0.3:
            break
        sequence = [start]
        unvisited = set(range(n_jobs)) - {start}

        while unvisited:
            best_cost = float('inf')
            best_pos = -1
            best_job = -1

            for job in unvisited:
                for pos in range(len(sequence) + 1):
                    new_seq = sequence[:pos] + [job] + sequence[pos:]
                    makespan, feasible = evaluate_sequence(new_seq, dist, prefix, n_machines, due_dates)
                    if feasible and makespan < best_cost:
                        best_cost = makespan
                        best_pos = pos
                        best_job = job

            if best_job == -1:
                break

            sequence = sequence[:best_pos] + [best_job] + sequence[best_pos:]
            unvisited.remove(best_job)

        if not unvisited:
            makespan = compute_makespan(sequence, dist, prefix, n_machines)
            if makespan < best_makespan:
                best_makespan = makespan
                best_solution = sequence[:]

    return best_solution, best_makespan


def local_search(sequence, dist, prefix, n_machines, due_dates, time_limit, start_time):
    """Combined local search with swap, insert, and 2-opt moves."""
    n = len(sequence)
    best_seq = sequence[:]
    best_makespan = compute_makespan(best_seq, dist, prefix, n_machines)
    
    improved = True
    while improved:
        if time.time() - start_time > time_limit - 0.5:
            break
        improved = False

        # Swap moves
        for i in range(n - 1):
            for j in range(i + 1, n):
                if time.time() - start_time > time_limit - 0.5:
                    return best_seq, best_makespan
                best_seq[i], best_seq[j] = best_seq[j], best_seq[i]
                makespan, feasible = evaluate_sequence(best_seq, dist, prefix, n_machines, due_dates)
                if feasible and makespan < best_makespan:
                    best_makespan = makespan
                    improved = True
                else:
                    best_seq[i], best_seq[j] = best_seq[j], best_seq[i]

        # Insert moves
        for i in range(n):
            for j in range(n):
                if i == j or abs(i - j) == 1:
                    continue
                if time.time() - start_time > time_limit - 0.5:
                    return best_seq, best_makespan
                job = best_seq.pop(i)
                best_seq.insert(j, job)
                makespan, feasible = evaluate_sequence(best_seq, dist, prefix, n_machines, due_dates)
                if feasible and makespan < best_makespan:
                    best_makespan = makespan
                    improved = True
                else:
                    best_seq.pop(j)
                    best_seq.insert(i, job)

        # 2-opt moves
        for i in range(n - 2):
            for j in range(i + 2, n):
                if time.time() - start_time > time_limit - 0.5:
                    return best_seq, best_makespan
                best_seq[i + 1:j + 1] = best_seq[i + 1:j + 1][::-1]
                makespan, feasible = evaluate_sequence(best_seq, dist, prefix, n_machines, due_dates)
                if feasible and makespan < best_makespan:
                    best_makespan = makespan
                    improved = True
                else:
                    best_seq[i + 1:j + 1] = best_seq[i + 1:j + 1][::-1]

    return best_seq, best_makespan


def fast_local_search(sequence, dist, prefix, n_machines, due_dates, time_limit, start_time):
    """Faster local search focusing on insert moves only."""
    n = len(sequence)
    best_seq = sequence[:]
    best_makespan = compute_makespan(best_seq, dist, prefix, n_machines)
    
    improved = True
    while improved:
        if time.time() - start_time > time_limit - 0.5:
            break
        improved = False

        for i in range(n):
            if time.time() - start_time > time_limit - 0.5:
                return best_seq, best_makespan
            job = best_seq[i]
            for j in range(n):
                if i == j:
                    continue
                new_seq = best_seq[:i] + best_seq[i + 1:]
                new_seq.insert(j if j < i else j - 1, job)
                makespan, feasible = evaluate_sequence(new_seq, dist, prefix, n_machines, due_dates)
                if feasible and makespan < best_makespan:
                    best_seq = new_seq
                    best_makespan = makespan
                    improved = True
                    break
            if improved:
                break

    return best_seq, best_makespan


def solve_with_gurobi(n_jobs, n_machines, due_dates, dist, prefix, time_limit):
    """Solve using Gurobi MIP formulation."""
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError:
        return None, float('inf')

    try:
        model = gp.Model("no_wait_flow_shop")
        model.setParam('Seed', 0)
        model.setParam('MIPGap', 1e-4)
        model.setParam('NumericFocus', 0)
        model.setParam('Threads', 1)
        model.setParam('TimeLimit', max(time_limit, 1))
        model.setParam('OutputFlag', 0)

        max_due = max(due_dates)
        
        # x[i,j] = 1 if job j immediately follows job i
        x = {}
        for i in range(n_jobs):
            for j in range(n_jobs):
                if i != j:
                    x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

        # s[j] = start time of job j on machine 1
        s = {}
        for j in range(n_jobs):
            s[j] = model.addVar(lb=0, ub=max_due, vtype=GRB.CONTINUOUS, name=f"s_{j}")

        # C_max
        C_max = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="C_max")

        # MTZ variables
        u = {}
        for j in range(n_jobs):
            u[j] = model.addVar(lb=0, ub=n_jobs - 1, vtype=GRB.CONTINUOUS, name=f"u_{j}")

        model.update()
        model.setObjective(C_max, GRB.MINIMIZE)

        # Flow constraints
        for i in range(n_jobs):
            model.addConstr(gp.quicksum(x[i, j] for j in range(n_jobs) if j != i) <= 1)
        for j in range(n_jobs):
            model.addConstr(gp.quicksum(x[i, j] for i in range(n_jobs) if i != j) <= 1)
        model.addConstr(gp.quicksum(x[i, j] for i in range(n_jobs) for j in range(n_jobs) if i != j) == n_jobs - 1)

        # Timing constraints
        M_big = max_due + 1
        for i in range(n_jobs):
            for j in range(n_jobs):
                if i != j:
                    model.addConstr(s[j] >= s[i] + dist[i][j] - M_big * (1 - x[i, j]))

        # Due date constraints
        for j in range(n_jobs):
            model.addConstr(s[j] + prefix[n_machines][j] <= due_dates[j])

        # Makespan
        for j in range(n_jobs):
            model.addConstr(C_max >= s[j] + prefix[n_machines][j])

        # Subtour elimination (MTZ)
        for i in range(n_jobs):
            for j in range(n_jobs):
                if i != j:
                    model.addConstr(u[j] >= u[i] + 1 - n_jobs * (1 - x[i, j]))

        model.optimize()

        if model.SolCount > 0:
            succ = [-1] * n_jobs
            has_pred = [False] * n_jobs
            for i in range(n_jobs):
                for j in range(n_jobs):
                    if i != j and x[i, j].X > 0.5:
                        succ[i] = j
                        has_pred[j] = True

            start = -1
            for j in range(n_jobs):
                if not has_pred[j]:
                    start = j
                    break

            if start == -1:
                return None, float('inf')

            sequence = [start]
            current = start
            while succ[current] != -1:
                current = succ[current]
                sequence.append(current)

            if len(sequence) == n_jobs:
                makespan, feasible = evaluate_sequence(sequence, dist, prefix, n_machines, due_dates)
                if feasible:
                    return sequence, makespan

        return None, float('inf')

    except Exception:
        return None, float('inf')


def greedy_due_date(dist, n_jobs, prefix, n_machines, due_dates):
    """Greedy heuristic prioritizing jobs with tight due dates."""
    best_solution = None
    best_makespan = float('inf')

    for start in range(n_jobs):
        unvisited = set(range(n_jobs)) - {start}
        sequence = [start]
        current = start
        current_time = 0
        feasible = True

        while unvisited:
            candidates = []
            for j in unvisited:
                d = dist[current][j]
                new_time = current_time + d
                completion = new_time + prefix[n_machines][j]
                slack = due_dates[j] - completion
                if slack >= 0:
                    candidates.append((slack, d, j))

            if not candidates:
                feasible = False
                break

            candidates.sort()
            _, d, best_next = candidates[0]
            current_time += d
            sequence.append(best_next)
            unvisited.remove(best_next)
            current = best_next

        if feasible:
            makespan = compute_makespan(sequence, dist, prefix, n_machines)
            if makespan < best_makespan:
                best_makespan = makespan
                best_solution = sequence[:]

    return best_solution, best_makespan


def solve(instance_path, solution_path, time_limit, log_path=None):
    start_time = time.time()

    with open(instance_path, 'r') as f:
        instance = json.load(f)

    n_jobs = instance['n_jobs']
    n_machines = instance['n_machines']
    processing_times = instance['processing_times']
    due_dates = instance['due_dates']

    logger = SolutionLogger(log_path, sense="minimize") if log_path else None

    prefix = compute_prefix_sums(processing_times, n_jobs, n_machines)
    dist = compute_distance_matrix(processing_times, n_jobs, n_machines, prefix)

    best_sequence = None
    best_makespan = float('inf')

    def log_if_improved(seq, mk):
        nonlocal best_sequence, best_makespan
        if mk < best_makespan:
            best_sequence = seq[:]
            best_makespan = mk
            sol = {
                "objective_value": float(best_makespan),
                "sequence": [j + 1 for j in best_sequence]
            }
            if logger:
                logger.log_solution(best_makespan, sol)

    # Phase 1: Construction heuristics
    for criterion in ['distance', 'slack', 'combined']:
        if time.time() - start_time > time_limit * 0.2:
            break
        seq, mk = nearest_neighbor(dist, n_jobs, prefix, n_machines, due_dates, criterion)
        if seq is not None:
            log_if_improved(seq, mk)

    # Greedy due date
    if time.time() - start_time < time_limit * 0.3:
        seq, mk = greedy_due_date(dist, n_jobs, prefix, n_machines, due_dates)
        if seq is not None:
            log_if_improved(seq, mk)

    # Insertion heuristic for small instances
    if n_jobs <= 25 and time.time() - start_time < time_limit * 0.4:
        seq, mk = insertion_heuristic(dist, n_jobs, prefix, n_machines, due_dates, time_limit * 0.4, start_time)
        if seq is not None:
            log_if_improved(seq, mk)

    # Phase 2: Exact method for small instances
    if n_jobs <= 20 and time.time() - start_time < time_limit - 2:
        remaining = time_limit - (time.time() - start_time) - 1
        if remaining > 1:
            seq, mk = solve_with_gurobi(n_jobs, n_machines, due_dates, dist, prefix, remaining)
            if seq is not None:
                log_if_improved(seq, mk)

    # Phase 3: Local search
    if best_sequence is not None and time.time() - start_time < time_limit - 1:
        if n_jobs <= 60:
            seq, mk = local_search(best_sequence, dist, prefix, n_machines, due_dates, time_limit, start_time)
            log_if_improved(seq, mk)
        else:
            seq, mk = fast_local_search(best_sequence, dist, prefix, n_machines, due_dates, time_limit, start_time)
            log_if_improved(seq, mk)

    # Phase 4: Random restarts for larger instances
    if n_jobs > 20 and time.time() - start_time < time_limit - 2:
        random.seed(42)
        attempts = 0
        while time.time() - start_time < time_limit - 1.5 and attempts < 100:
            attempts += 1
            seq = list(range(n_jobs))
            random.shuffle(seq)
            if check_due_dates(seq, dist, prefix, n_machines, due_dates):
                mk = compute_makespan(seq, dist, prefix, n_machines)
                log_if_improved(seq, mk)
                if time.time() - start_time < time_limit - 1:
                    if n_jobs <= 40:
                        improved_seq, improved_mk = local_search(seq, dist, prefix, n_machines, due_dates, time_limit, start_time)
                    else:
                        improved_seq, improved_mk = fast_local_search(seq, dist, prefix, n_machines, due_dates, time_limit, start_time)
                    log_if_improved(improved_seq, improved_mk)

    # Fallback for very small instances
    if best_sequence is None and n_jobs <= 10:
        from itertools import permutations
        for perm in permutations(range(n_jobs)):
            perm = list(perm)
            mk, feasible = evaluate_sequence(perm, dist, prefix, n_machines, due_dates)
            if feasible:
                log_if_improved(perm, mk)

    # Final fallback
    if best_sequence is None:
        # Try to find any feasible sequence
        seq = list(range(n_jobs))
        seq.sort(key=lambda j: due_dates[j])
        if check_due_dates(seq, dist, prefix, n_machines, due_dates):
            best_sequence = seq
            best_makespan = compute_makespan(seq, dist, prefix, n_machines)
        else:
            # Last resort
            best_sequence = list(range(n_jobs))
            best_makespan = compute_makespan(best_sequence, dist, prefix, n_machines)

    solution = {
        "objective_value": float(best_makespan),
        "sequence": [j + 1 for j in best_sequence]
    }

    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="No-Wait Flow Shop Scheduler")
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)


if __name__ == '__main__':
    main()
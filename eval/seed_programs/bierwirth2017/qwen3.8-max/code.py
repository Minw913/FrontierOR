import json
import argparse
import time
import random
import math
from copy import deepcopy
from solution_logger import SolutionLogger


def load_instance(path):
    with open(path, 'r') as f:
        return json.load(f)


def compute_schedule_from_sequences(instance, machine_sequences):
    """
    Given the order of jobs on each machine, compute start times.
    machine_sequences: dict mapping machine_id -> list of (job_id, op_index)
    """
    num_jobs = instance['num_jobs']
    num_machines = instance['num_machines']
    jobs = instance['jobs']

    op_info = {}
    for job in jobs:
        jid = job['job_id']
        op_info[jid] = [(op['machine'], op['processing_time']) for op in job['operations']]

    # Build predecessor relationships
    machine_pred = {}
    for m in range(num_machines):
        seq = machine_sequences.get(m, [])
        for i, (jid, oidx) in enumerate(seq):
            if i > 0:
                machine_pred[(jid, oidx)] = seq[i - 1]
            else:
                machine_pred[(jid, oidx)] = None

    # Compute start times using iterative relaxation
    start = [[-1.0] * len(op_info[j]) for j in range(num_jobs)]

    for _ in range(num_jobs * num_machines + 10):
        changed = False
        for j in range(num_jobs):
            for k in range(len(op_info[j])):
                if start[j][k] >= 0:
                    continue
                earliest = 0.0
                ready = True

                if k == 0:
                    earliest = max(earliest, float(jobs[j]['release_date']))

                if k > 0:
                    if start[j][k - 1] < 0:
                        ready = False
                    else:
                        earliest = max(earliest, start[j][k - 1] + op_info[j][k - 1][1])

                pred = machine_pred.get((j, k))
                if pred is not None:
                    pj, pk = pred
                    if start[pj][pk] < 0:
                        ready = False
                    else:
                        earliest = max(earliest, start[pj][pk] + op_info[pj][pk][1])

                if ready:
                    start[j][k] = earliest
                    changed = True

        if not changed:
            break

    # Verify all assigned
    for j in range(num_jobs):
        for k in range(len(op_info[j])):
            if start[j][k] < 0:
                raise ValueError("Cycle detected or incomplete schedule")

    return start


def evaluate_schedule(instance, start):
    """Compute total weighted tardiness from start times."""
    jobs = instance['jobs']
    total = 0.0
    for job in jobs:
        jid = job['job_id']
        ops = job['operations']
        last_k = len(ops) - 1
        completion = start[jid][last_k] + ops[last_k]['processing_time']
        tardiness = max(0.0, completion - job['due_date'])
        total += job['weight'] * tardiness
    return total


def build_solution_dict(instance, start):
    """Build the solution dictionary from start times."""
    jobs = instance['jobs']
    total_weighted_tardiness = 0.0
    schedule = []

    for job in jobs:
        jid = job['job_id']
        ops = job['operations']
        last_k = len(ops) - 1
        completion = start[jid][last_k] + ops[last_k]['processing_time']
        tardiness = max(0.0, completion - job['due_date'])
        total_weighted_tardiness += job['weight'] * tardiness

        op_list = []
        for k, op in enumerate(ops):
            op_list.append({
                'machine': op['machine'],
                'start_time': float(start[jid][k])
            })

        schedule.append({
            'job_id': jid,
            'completion_time': float(completion),
            'tardiness': float(tardiness),
            'operations': op_list
        })

    return {
        'objective_value': float(total_weighted_tardiness),
        'schedule': schedule
    }


def dispatch_heuristic(instance, rule='EDD', randomize=False):
    """Dispatch jobs to machines using a priority rule."""
    num_jobs = instance['num_jobs']
    num_machines = instance['num_machines']
    jobs = instance['jobs']

    machine_sequences = {m: [] for m in range(num_machines)}
    machine_end_time = {m: 0.0 for m in range(num_machines)}
    completed_ops = [0] * num_jobs
    job_ready = [float(jobs[j]['release_date']) for j in range(num_jobs)]

    total_ops = sum(len(job['operations']) for job in jobs)
    scheduled_ops = 0

    while scheduled_ops < total_ops:
        candidates = []
        for j in range(num_jobs):
            if completed_ops[j] >= len(jobs[j]['operations']):
                continue
            k = completed_ops[j]
            m = jobs[j]['operations'][k]['machine']
            earliest = max(job_ready[j], machine_end_time[m])
            candidates.append((earliest, j, k, m))

        if not candidates:
            break

        min_time = min(c[0] for c in candidates)

        if rule == 'EDD':
            candidates.sort(key=lambda x: (x[0], jobs[x[1]]['due_date']))
        elif rule == 'SPT':
            candidates.sort(key=lambda x: (x[0], jobs[x[1]]['operations'][x[2]]['processing_time']))
        elif rule == 'WSPT':
            candidates.sort(key=lambda x: (x[0], jobs[x[1]]['operations'][x[2]]['processing_time'] / max(jobs[x[1]]['weight'], 1e-6)))
        elif rule == 'ATC':
            avg_pt = sum(op['processing_time'] for job in jobs for op in job['operations']) / total_ops
            candidates.sort(key=lambda x: (x[0], -atc_priority(instance, x[1], x[2], x[0], avg_pt)))
        elif rule == 'MDD':
            candidates.sort(key=lambda x: (x[0], max(jobs[x[1]]['due_date'], x[0] + sum(o['processing_time'] for o in jobs[x[1]]['operations'][x[2]:]))))
        elif rule == 'RANDOM':
            random.shuffle(candidates)
            candidates.sort(key=lambda x: x[0])
        else:
            candidates.sort(key=lambda x: x[0])

        _, j, k, m = candidates[0]
        start_time = max(job_ready[j], machine_end_time[m])

        machine_sequences[m].append((j, k))
        machine_end_time[m] = start_time + jobs[j]['operations'][k]['processing_time']
        job_ready[j] = machine_end_time[m]
        completed_ops[j] += 1
        scheduled_ops += 1

    return machine_sequences


def atc_priority(instance, job_id, op_idx, current_time, avg_pt):
    """Compute ATC priority."""
    job = instance['jobs'][job_id]
    op = job['operations'][op_idx]
    remaining = sum(o['processing_time'] for o in job['operations'][op_idx:])
    slack = job['due_date'] - current_time - remaining
    k2 = 0.5
    if avg_pt <= 0:
        avg_pt = 1.0
    atc = (job['weight'] / max(op['processing_time'], 1e-6)) * math.exp(-max(slack, 0) / (k2 * avg_pt * len(job['operations'])))
    return atc


def local_search_adjacent(instance, machine_sequences, max_iterations=2000, time_budget=30):
    """Local search with adjacent swaps on machines."""
    start_time = time.time()

    try:
        current_start = compute_schedule_from_sequences(instance, machine_sequences)
        current_obj = evaluate_schedule(instance, current_start)
    except ValueError:
        return machine_sequences, None, float('inf')

    best_sequences = deepcopy(machine_sequences)
    best_obj = current_obj
    best_start = current_start

    no_improve_count = 0
    max_no_improve = 50

    for iteration in range(max_iterations):
        if time.time() - start_time > time_budget:
            break

        improved = False
        # Iterate over machines in random order
        machine_order = list(range(instance['num_machines']))
        random.shuffle(machine_order)

        for m in machine_order:
            seq = machine_sequences[m]
            if len(seq) < 2:
                continue

            for i in range(len(seq) - 1):
                # Swap adjacent
                seq[i], seq[i + 1] = seq[i + 1], seq[i]

                try:
                    new_start = compute_schedule_from_sequences(instance, machine_sequences)
                    new_obj = evaluate_schedule(instance, new_start)

                    if new_obj < current_obj - 1e-9:
                        current_obj = new_obj
                        current_start = new_start
                        improved = True
                        no_improve_count = 0

                        if current_obj < best_obj - 1e-9:
                            best_obj = current_obj
                            best_start = deepcopy(current_start)
                            best_sequences = deepcopy(machine_sequences)
                    else:
                        seq[i], seq[i + 1] = seq[i + 1], seq[i]
                except ValueError:
                    seq[i], seq[i + 1] = seq[i + 1], seq[i]

            if time.time() - start_time > time_budget:
                break

        if not improved:
            no_improve_count += 1
            if no_improve_count >= max_no_improve:
                break

    return best_sequences, best_start, best_obj


def simulated_annealing(instance, initial_sequences, initial_obj, time_budget, logger=None):
    """Simulated annealing with adjacent swap and insertion neighborhoods."""
    start_time = time.time()
    random.seed(42)

    best_sequences = deepcopy(initial_sequences)
    best_obj = initial_obj

    current_sequences = deepcopy(initial_sequences)
    current_obj = initial_obj

    # Estimate initial temperature
    if initial_obj > 0:
        T_start = initial_obj * 0.05
    else:
        T_start = 10.0
    T_end = 0.001

    # Estimate number of iterations
    elapsed = time.time() - start_time
    # Rough estimate: each iteration takes about 0.001-0.01 seconds
    est_iter_time = 0.005
    max_iters = int((time_budget - elapsed) / est_iter_time)
    max_iters = max(max_iters, 100)

    cooling = (T_end / T_start) ** (1.0 / max(max_iters, 1))
    T = T_start
    iteration = 0

    while time.time() - start_time < time_budget and iteration < max_iters:
        iteration += 1

        # Choose neighborhood
        if random.random() < 0.7:
            # Adjacent swap
            m = random.randint(0, instance['num_machines'] - 1)
            seq = current_sequences[m]
            if len(seq) < 2:
                T *= cooling
                continue
            i = random.randint(0, len(seq) - 2)
            seq[i], seq[i + 1] = seq[i + 1], seq[i]

            try:
                new_start = compute_schedule_from_sequences(instance, current_sequences)
                new_obj = evaluate_schedule(instance, new_start)
                delta = new_obj - current_obj

                if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                    current_obj = new_obj
                    if current_obj < best_obj - 1e-9:
                        best_obj = current_obj
                        best_sequences = deepcopy(current_sequences)
                        if logger:
                            sol_dict = build_solution_dict(instance, new_start)
                            logger.log_solution(best_obj, sol_dict)
                else:
                    seq[i], seq[i + 1] = seq[i + 1], seq[i]
            except ValueError:
                seq[i], seq[i + 1] = seq[i + 1], seq[i]
        else:
            # Insertion move: remove an operation and insert at another position
            m = random.randint(0, instance['num_machines'] - 1)
            seq = current_sequences[m]
            if len(seq) < 2:
                T *= cooling
                continue
            i = random.randint(0, len(seq) - 1)
            item = seq.pop(i)
            j = random.randint(0, len(seq))
            seq.insert(j, item)

            try:
                new_start = compute_schedule_from_sequences(instance, current_sequences)
                new_obj = evaluate_schedule(instance, new_start)
                delta = new_obj - current_obj

                if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                    current_obj = new_obj
                    if current_obj < best_obj - 1e-9:
                        best_obj = current_obj
                        best_sequences = deepcopy(current_sequences)
                        if logger:
                            sol_dict = build_solution_dict(instance, new_start)
                            logger.log_solution(best_obj, sol_dict)
                else:
                    seq.pop(j)
                    seq.insert(i, item)
            except ValueError:
                seq.pop(j)
                seq.insert(i, item)

        T *= cooling

    return best_sequences, best_obj


def solve_with_gurobi(instance, time_limit):
    """Solve using Gurobi MIP for small instances."""
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError:
        return None

    num_jobs = instance['num_jobs']
    num_machines = instance['num_machines']
    jobs = instance['jobs']

    # Count binary variables
    num_binary = num_machines * (num_jobs * (num_jobs - 1)) // 2
    if num_binary > 3000:
        return None

    try:
        model = gp.Model("job_shop_tardiness")
        model.setParam('Seed', 0)
        model.setParam('MIPGap', 1e-4)
        model.setParam('NumericFocus', 0)
        model.setParam('Threads', 1)
        model.setParam('TimeLimit', time_limit)
        model.setParam('OutputFlag', 0)

        # Big-M
        total_proc = sum(op['processing_time'] for job in jobs for op in job['operations'])
        max_release = max(job['release_date'] for job in jobs)
        M = total_proc + max_release + 1

        # Start time variables
        s = {}
        for job in jobs:
            j = job['job_id']
            for k in range(len(job['operations'])):
                s[j, k] = model.addVar(lb=0, ub=M, name=f"s_{j}_{k}")

        # Tardiness variables
        T = {}
        for job in jobs:
            j = job['job_id']
            T[j] = model.addVar(lb=0, name=f"T_{j}")

        # Machine operations
        machine_ops = {m: [] for m in range(num_machines)}
        for job in jobs:
            j = job['job_id']
            for k, op in enumerate(job['operations']):
                machine_ops[op['machine']].append((j, k))

        # Binary variables for ordering
        y = {}
        for m in range(num_machines):
            ops = machine_ops[m]
            for i in range(len(ops)):
                for j_idx in range(i + 1, len(ops)):
                    y[ops[i], ops[j_idx]] = model.addVar(
                        vtype=GRB.BINARY, name=f"y_{ops[i][0]}_{ops[i][1]}_{ops[j_idx][0]}_{ops[j_idx][1]}")

        model.update()

        # Objective
        obj = gp.quicksum(jobs[j]['weight'] * T[j] for j in range(num_jobs))
        model.setObjective(obj, GRB.MINIMIZE)

        # Constraints
        for job in jobs:
            j = job['job_id']
            ops = job['operations']

            # Release date
            model.addConstr(s[j, 0] >= job['release_date'], name=f"release_{j}")

            # Precedence
            for k in range(len(ops) - 1):
                model.addConstr(s[j, k + 1] >= s[j, k] + ops[k]['processing_time'],
                                name=f"prec_{j}_{k}")

            # Tardiness definition
            last_k = len(ops) - 1
            model.addConstr(T[j] >= s[j, last_k] + ops[last_k]['processing_time'] - job['due_date'],
                            name=f"tard_{j}")

        # Disjunctive constraints
        for m in range(num_machines):
            ops = machine_ops[m]
            for i in range(len(ops)):
                for j_idx in range(i + 1, len(ops)):
                    j1, k1 = ops[i]
                    j2, k2 = ops[j_idx]
                    p1 = jobs[j1]['operations'][k1]['processing_time']
                    p2 = jobs[j2]['operations'][k2]['processing_time']
                    var = y[ops[i], ops[j_idx]]

                    model.addConstr(s[j2, k2] >= s[j1, k1] + p1 - M * (1 - var),
                                    name=f"disj1_{j1}_{k1}_{j2}_{k2}")
                    model.addConstr(s[j1, k1] >= s[j2, k2] + p2 - M * var,
                                    name=f"disj2_{j1}_{k1}_{j2}_{k2}")

        model.optimize()

        if model.SolCount > 0:
            start = [[0.0] * len(jobs[j]['operations']) for j in range(num_jobs)]
            for job in jobs:
                j = job['job_id']
                for k in range(len(job['operations'])):
                    start[j][k] = s[j, k].X
            return start

        return None
    except Exception:
        return None


def extract_sequences_from_start(instance, start):
    """Extract machine sequences from start times."""
    num_machines = instance['num_machines']
    jobs = instance['jobs']

    machine_ops = {m: [] for m in range(num_machines)}
    for job in jobs:
        jid = job['job_id']
        for k, op in enumerate(job['operations']):
            machine_ops[op['machine']].append((start[jid][k], jid, k))

    machine_sequences = {}
    for m in range(num_machines):
        machine_ops[m].sort(key=lambda x: x[0])
        machine_sequences[m] = [(jid, k) for (_, jid, k) in machine_ops[m]]

    return machine_sequences


def main():
    parser = argparse.ArgumentParser(description='Job Shop Scheduling - Weighted Tardiness')
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    instance = load_instance(args.instance_path)
    global_start = time.time()

    num_jobs = instance['num_jobs']
    num_machines = instance['num_machines']
    total_ops = sum(len(job['operations']) for job in instance['jobs'])

    best_solution = None
    best_obj = float('inf')

    # Phase 1: Try Gurobi for small instances
    remaining = args.time_limit - (time.time() - global_start)
    if total_ops <= 150 and remaining > 10:
        gurobi_time = min(remaining * 0.5, remaining - 5)
        gurobi_time = max(gurobi_time, 5)
        gurobi_start_times = solve_with_gurobi(instance, gurobi_time)
        if gurobi_start_times is not None:
            obj = evaluate_schedule(instance, gurobi_start_times)
            sol_dict = build_solution_dict(instance, gurobi_start_times)
            if obj < best_obj:
                best_obj = obj
                best_solution = sol_dict
                if logger:
                    logger.log_solution(best_obj, best_solution)

    # Phase 2: Dispatch heuristics
    remaining = args.time_limit - (time.time() - global_start)
    if remaining > 0.5:
        rules = ['EDD', 'SPT', 'WSPT', 'ATC', 'MDD']
        best_dispatch_seq = None
        best_dispatch_obj = float('inf')

        for rule in rules:
            if time.time() - global_start >= args.time_limit - 1:
                break
            try:
                seq = dispatch_heuristic(instance, rule)
                start = compute_schedule_from_sequences(instance, seq)
                obj = evaluate_schedule(instance, start)

                if obj < best_obj:
                    best_obj = obj
                    best_solution = build_solution_dict(instance, start)
                    if logger:
                        logger.log_solution(best_obj, best_solution)

                if obj < best_dispatch_obj:
                    best_dispatch_obj = obj
                    best_dispatch_seq = deepcopy(seq)
            except Exception:
                continue

        # Also try randomized dispatch
        for trial in range(5):
            if time.time() - global_start >= args.time_limit - 1:
                break
            try:
                seq = dispatch_heuristic(instance, 'RANDOM')
                start = compute_schedule_from_sequences(instance, seq)
                obj = evaluate_schedule(instance, start)
                if obj < best_dispatch_obj:
                    best_dispatch_obj = obj
                    best_dispatch_seq = deepcopy(seq)
                if obj < best_obj:
                    best_obj = obj
                    best_solution = build_solution_dict(instance, start)
                    if logger:
                        logger.log_solution(best_obj, best_solution)
            except Exception:
                continue

    # Phase 3: Local search on best dispatch solution
    remaining = args.time_limit - (time.time() - global_start)
    if remaining > 2 and best_dispatch_seq is not None:
        ls_budget = min(remaining * 0.4, 60)
        improved_seq, improved_start, improved_obj = local_search_adjacent(
            instance, best_dispatch_seq, max_iterations=5000, time_budget=ls_budget
        )
        if improved_obj < best_obj:
            best_obj = improved_obj
            best_solution = build_solution_dict(instance, improved_start)
            if logger:
                logger.log_solution(best_obj, best_solution)
            best_dispatch_seq = improved_seq

    # Phase 4: Simulated annealing
    remaining = args.time_limit - (time.time() - global_start)
    if remaining > 3 and best_dispatch_seq is not None:
        sa_budget = remaining - 1
        sa_sequences, sa_obj = simulated_annealing(
            instance, best_dispatch_seq, best_dispatch_obj, sa_budget, logger
        )
        if sa_obj < best_obj:
            best_obj = sa_obj
            sa_start = compute_schedule_from_sequences(instance, sa_sequences)
            best_solution = build_solution_dict(instance, sa_start)
            if logger:
                logger.log_solution(best_obj, best_solution)

    # Phase 5: If we have a solution from Gurobi, try local search on it too
    if best_solution is not None:
        remaining = args.time_limit - (time.time() - global_start)
        if remaining > 2:
            # Reconstruct sequences from best solution and try more local search
            try:
                current_start = [[op['start_time'] for op in entry['operations']]
                                 for entry in best_solution['schedule']]
                # Map back to job_id indexed
                start_by_job = {}
                for entry in best_solution['schedule']:
                    start_by_job[entry['job_id']] = [op['start_time'] for op in entry['operations']]
                start_array = [[0.0] * len(instance['jobs'][j]['operations']) for j in range(num_jobs)]
                for j in range(num_jobs):
                    jid = instance['jobs'][j]['job_id']
                    for k in range(len(instance['jobs'][j]['operations'])):
                        start_array[j][k] = start_by_job[jid][k]

                seq_from_best = extract_sequences_from_start(instance, start_array)
                ls_budget = min(remaining - 1, 30)
                improved_seq2, improved_start2, improved_obj2 = local_search_adjacent(
                    instance, seq_from_best, max_iterations=3000, time_budget=ls_budget
                )
                if improved_obj2 < best_obj:
                    best_obj = improved_obj2
                    best_solution = build_solution_dict(instance, improved_start2)
                    if logger:
                        logger.log_solution(best_obj, best_solution)
            except Exception:
                pass

    # Fallback: ensure we have a solution
    if best_solution is None:
        try:
            seq = dispatch_heuristic(instance, 'EDD')
            start = compute_schedule_from_sequences(instance, seq)
            best_solution = build_solution_dict(instance, start)
            if logger:
                logger.log_solution(best_solution['objective_value'], best_solution)
        except Exception:
            # Last resort: sequential scheduling
            start = [[0.0] * len(instance['jobs'][j]['operations']) for j in range(num_jobs)]
            for j in range(num_jobs):
                job = instance['jobs'][j]
                current_time = float(job['release_date'])
                for k, op in enumerate(job['operations']):
                    start[j][k] = current_time
                    current_time += op['processing_time']
            best_solution = build_solution_dict(instance, start)
            if logger:
                logger.log_solution(best_solution['objective_value'], best_solution)

    # Write final solution
    with open(args.solution_path, 'w') as f:
        json.dump(best_solution, f, indent=2)


if __name__ == '__main__':
    main()
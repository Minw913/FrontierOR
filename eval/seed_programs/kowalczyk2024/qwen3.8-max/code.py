#!/usr/bin/env python3
"""
Solver for Parallel Machine Scheduling with Total Weighted Tardiness.
Problem: P_m || sum w_j T_j

Approach:
1. Multiple dispatching rules for initial solutions
2. Best-improvement local search with insertion moves
3. Simulated annealing for diversification
4. Gurobi MIP for small instances
"""

import json
import argparse
import time
import random
import math
import sys


def main():
    parser = argparse.ArgumentParser(description='Parallel Machine Scheduling Solver')
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    global_start = time.time()
    random.seed(42)

    # Read instance
    with open(args.instance_path) as f:
        data = json.load(f)

    n = data['n']
    m = data['m']
    T = data['T']
    jobs = data['jobs']
    jobs_dict = {j['id']: j for j in jobs}
    job_ids = [j['id'] for j in jobs]

    if n == 0:
        solution = {"objective_value": 0.0, "schedule": {}}
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
        return

    def evaluate(sequences):
        """Compute total weighted tardiness and schedule from machine sequences."""
        total_wt = 0
        schedule = {}
        for seq in sequences:
            t = 0
            for jid in seq:
                job = jobs_dict[jid]
                completion = t + job['p']
                tardiness = max(0, completion - job['d'])
                total_wt += job['w'] * tardiness
                schedule[jid] = t
                t = completion
        return total_wt, schedule

    def evaluate_machine_cost(seq):
        """Compute weighted tardiness for a single machine sequence."""
        total_wt = 0
        t = 0
        for jid in seq:
            job = jobs_dict[jid]
            completion = t + job['p']
            tardiness = max(0, completion - job['d'])
            total_wt += job['w'] * tardiness
            t = completion
        return total_wt

    # === Dispatching Rules ===

    def dispatch_atc(k=4.5):
        """Apparent Tardiness Cost dispatching rule."""
        avg_p = sum(j['p'] for j in jobs) / max(n, 1)
        remaining = list(jobs)
        machine_times = [0.0] * m
        sequences = [[] for _ in range(m)]

        while remaining:
            min_machine = min(range(m), key=lambda i: machine_times[i])
            t = machine_times[min_machine]

            best_idx = -1
            best_priority = -1.0
            for idx, job in enumerate(remaining):
                slack = max(0.0, job['d'] - t - job['p'])
                priority = (job['w'] / max(job['p'], 1)) * math.exp(-slack / (k * avg_p + 1e-10))
                if priority > best_priority:
                    best_priority = priority
                    best_idx = idx

            job = remaining.pop(best_idx)
            sequences[min_machine].append(job['id'])
            machine_times[min_machine] += job['p']

        return sequences

    def dispatch_wspt():
        """Weighted Shortest Processing Time rule."""
        sorted_jobs = sorted(jobs, key=lambda j: j['w'] / max(j['p'], 1), reverse=True)
        machine_times = [0.0] * m
        sequences = [[] for _ in range(m)]
        for job in sorted_jobs:
            best_machine = min(range(m), key=lambda i: machine_times[i])
            sequences[best_machine].append(job['id'])
            machine_times[best_machine] += job['p']
        return sequences

    def dispatch_edd():
        """Earliest Due Date rule."""
        sorted_jobs = sorted(jobs, key=lambda j: (j['d'], -j['w']))
        machine_times = [0.0] * m
        sequences = [[] for _ in range(m)]
        for job in sorted_jobs:
            best_machine = min(range(m), key=lambda i: machine_times[i])
            sequences[best_machine].append(job['id'])
            machine_times[best_machine] += job['p']
        return sequences

    def dispatch_wslack():
        """Weighted slack time rule."""
        remaining = list(jobs)
        machine_times = [0.0] * m
        sequences = [[] for _ in range(m)]

        while remaining:
            min_machine = min(range(m), key=lambda i: machine_times[i])
            t = machine_times[min_machine]

            best_idx = -1
            best_priority = -float('inf')
            for idx, job in enumerate(remaining):
                slack = job['d'] - t - job['p']
                if slack <= 0:
                    priority = job['w'] * 10000 + job['w'] / max(job['p'], 1)
                else:
                    priority = job['w'] / (slack + 1)
                if priority > best_priority:
                    best_priority = priority
                    best_idx = idx

            job = remaining.pop(best_idx)
            sequences[min_machine].append(job['id'])
            machine_times[min_machine] += job['p']

        return sequences

    def dispatch_mdd():
        """Modified Due Date rule."""
        sorted_jobs = sorted(jobs, key=lambda j: (j['d'] - j['p'], -j['w']))
        machine_times = [0.0] * m
        sequences = [[] for _ in range(m)]
        for job in sorted_jobs:
            best_machine = min(range(m), key=lambda i: machine_times[i])
            sequences[best_machine].append(job['id'])
            machine_times[best_machine] += job['p']
        return sequences

    # === Local Search ===

    def local_search(sequences, current_obj, end_time):
        """Best-improvement local search with insertion moves."""
        best_sequences = [seq[:] for seq in sequences]
        best_obj = current_obj

        improved = True
        while improved and time.time() < end_time:
            improved = False

            for src_m in range(m):
                if time.time() >= end_time:
                    break
                src_len = len(best_sequences[src_m])
                for src_pos in range(src_len):
                    if time.time() >= end_time:
                        break
                    jid = best_sequences[src_m][src_pos]

                    # Cost of source machine without this job
                    seq_without = best_sequences[src_m][:src_pos] + best_sequences[src_m][src_pos + 1:]
                    cost_src_original = evaluate_machine_cost(best_sequences[src_m])
                    cost_src_without = evaluate_machine_cost(seq_without)

                    best_delta = -1e-10
                    best_dst_m = -1
                    best_dst_pos = -1

                    for dst_m in range(m):
                        if dst_m == src_m:
                            base_seq = seq_without
                            base_cost = cost_src_without
                        else:
                            base_seq = best_sequences[dst_m]
                            base_cost = evaluate_machine_cost(base_seq)

                        for dst_pos in range(len(base_seq) + 1):
                            new_seq = base_seq[:dst_pos] + [jid] + base_seq[dst_pos:]
                            new_cost = evaluate_machine_cost(new_seq)

                            if dst_m == src_m:
                                delta = new_cost - cost_src_original
                            else:
                                delta = (cost_src_without - cost_src_original) + (new_cost - base_cost)

                            if delta < best_delta:
                                best_delta = delta
                                best_dst_m = dst_m
                                best_dst_pos = dst_pos

                    if best_dst_m >= 0:
                        best_sequences[src_m].pop(src_pos)
                        best_sequences[best_dst_m].insert(best_dst_pos, jid)
                        best_obj += best_delta
                        improved = True
                        break

        _, best_schedule = evaluate(best_sequences)
        return best_sequences, best_obj, best_schedule

    # === Simulated Annealing ===

    def simulated_annealing(sequences, current_obj, end_time):
        """Simulated annealing for diversification."""
        current_sequences = [seq[:] for seq in sequences]
        current_obj_val = current_obj

        best_sequences = [seq[:] for seq in current_sequences]
        best_obj = current_obj_val
        _, best_schedule = evaluate(best_sequences)

        if logger:
            logger.log_solution(best_obj, best_schedule)

        # Estimate initial temperature
        deltas = []
        for _ in range(min(200, n * m * 2)):
            test_seqs = [seq[:] for seq in current_sequences]
            src_m = random.randint(0, m - 1)
            if not test_seqs[src_m]:
                continue
            src_pos = random.randint(0, len(test_seqs[src_m]) - 1)
            jid = test_seqs[src_m].pop(src_pos)
            dst_m = random.randint(0, m - 1)
            dst_pos = random.randint(0, len(test_seqs[dst_m]))
            test_seqs[dst_m].insert(dst_pos, jid)
            new_obj_val, _ = evaluate(test_seqs)
            deltas.append(abs(new_obj_val - current_obj_val))

        if deltas and sum(deltas) > 0:
            T_init = (sum(deltas) / len(deltas)) * 3.0
        else:
            T_init = max(1.0, current_obj_val * 0.1)

        T_temp = max(T_init, 0.1)
        cooling = 0.9997

        no_improve_count = 0

        while time.time() < end_time - 0.5:
            # Generate neighbor
            neighbor_sequences = [seq[:] for seq in current_sequences]

            move_type = random.random()

            if move_type < 0.55:
                # Insertion move
                src_m = random.randint(0, m - 1)
                if not neighbor_sequences[src_m]:
                    continue
                src_pos = random.randint(0, len(neighbor_sequences[src_m]) - 1)
                jid = neighbor_sequences[src_m].pop(src_pos)
                dst_m = random.randint(0, m - 1)
                dst_pos = random.randint(0, len(neighbor_sequences[dst_m]))
                neighbor_sequences[dst_m].insert(dst_pos, jid)
            elif move_type < 0.85:
                # Swap move
                m1 = random.randint(0, m - 1)
                m2 = random.randint(0, m - 1)
                if not neighbor_sequences[m1] or not neighbor_sequences[m2]:
                    continue
                p1 = random.randint(0, len(neighbor_sequences[m1]) - 1)
                p2 = random.randint(0, len(neighbor_sequences[m2]) - 1)
                if m1 == m2 and p1 == p2:
                    continue
                neighbor_sequences[m1][p1], neighbor_sequences[m2][p2] = \
                    neighbor_sequences[m2][p2], neighbor_sequences[m1][p1]
            else:
                # Or-opt: move a block of 2 consecutive jobs
                src_m = random.randint(0, m - 1)
                if len(neighbor_sequences[src_m]) < 2:
                    continue
                src_pos = random.randint(0, len(neighbor_sequences[src_m]) - 2)
                block = neighbor_sequences[src_m][src_pos:src_pos + 2]
                neighbor_sequences[src_m] = neighbor_sequences[src_m][:src_pos] + \
                    neighbor_sequences[src_m][src_pos + 2:]
                dst_m = random.randint(0, m - 1)
                dst_pos = random.randint(0, len(neighbor_sequences[dst_m]))
                for k, jid in enumerate(block):
                    neighbor_sequences[dst_m].insert(dst_pos + k, jid)

            neighbor_obj, neighbor_schedule = evaluate(neighbor_sequences)
            delta = neighbor_obj - current_obj_val

            accept = False
            if delta < 0:
                accept = True
            elif T_temp > 0.001:
                try:
                    if random.random() < math.exp(-delta / T_temp):
                        accept = True
                except OverflowError:
                    pass

            if accept:
                current_sequences = neighbor_sequences
                current_obj_val = neighbor_obj

                if current_obj_val < best_obj:
                    best_obj = current_obj_val
                    best_sequences = [seq[:] for seq in current_sequences]
                    best_schedule = neighbor_schedule
                    no_improve_count = 0
                    if logger:
                        logger.log_solution(best_obj, best_schedule)
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1

            T_temp *= cooling

            # Reheat if stuck for too long
            if no_improve_count > 5000:
                T_temp = max(T_init * 0.5, 1.0)
                no_improve_count = 0

        return best_sequences, best_obj, best_schedule

    # === Gurobi MIP ===

    def try_gurobi(gurobi_time_limit):
        """Try to solve with Gurobi MIP for small instances."""
        try:
            import gurobipy as gp
            from gurobipy import GRB
        except ImportError:
            return None, None

        if n * m * T > 25000 or n > 30:
            return None, None

        try:
            model = gp.Model("scheduling")
            model.Params.Seed = 0
            model.Params.MIPGap = 1e-4
            model.Params.NumericFocus = 0
            model.Params.Threads = 1
            model.Params.TimeLimit = gurobi_time_limit
            model.Params.OutputFlag = 0

            # x[j, i, t] = 1 if job j starts at time t on machine i
            x = {}
            for job in jobs:
                jid = job['id']
                p_j = job['p']
                for i in range(m):
                    for t in range(0, max(0, T - p_j) + 1):
                        x[jid, i, t] = model.addVar(vtype=GRB.BINARY, name=f"x_{jid}_{i}_{t}")

            # Each job is scheduled exactly once
            for job in jobs:
                jid = job['id']
                p_j = job['p']
                model.addConstr(
                    gp.quicksum(x[jid, i, t] for i in range(m)
                                for t in range(0, max(0, T - p_j) + 1)) == 1,
                    name=f"assign_{jid}"
                )

            # No overlap on each machine: at most one job active at any time
            for i in range(m):
                for t in range(T):
                    expr = gp.quicksum(
                        x[job['id'], i, s]
                        for job in jobs
                        for s in range(max(0, t - job['p'] + 1), min(t + 1, max(0, T - job['p']) + 1))
                    )
                    model.addConstr(expr <= 1, name=f"cap_{i}_{t}")

            # Tardiness variables and constraints
            tard = {}
            for job in jobs:
                jid = job['id']
                p_j = job['p']
                d_j = job['d']
                tard[jid] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"tard_{jid}")

                model.addConstr(
                    tard[jid] >= gp.quicksum(
                        (t + p_j - d_j) * x[jid, i, t]
                        for i in range(m)
                        for t in range(0, max(0, T - p_j) + 1)
                    ),
                    name=f"tard_def_{jid}"
                )

            # Objective: minimize total weighted tardiness
            model.setObjective(
                gp.quicksum(job['w'] * tard[job['id']] for job in jobs),
                GRB.MINIMIZE
            )

            model.optimize()

            if model.SolCount > 0:
                # Extract solution
                machine_jobs = [[] for _ in range(m)]
                for job in jobs:
                    jid = job['id']
                    p_j = job['p']
                    found = False
                    for i in range(m):
                        if found:
                            break
                        for t in range(0, max(0, T - p_j) + 1):
                            if x[jid, i, t].X > 0.5:
                                machine_jobs[i].append((t, jid))
                                found = True
                                break

                # Sort by start time and extract sequences
                sequences = []
                for i in range(m):
                    machine_jobs[i].sort()
                    sequences.append([jid for _, jid in machine_jobs[i]])

                obj_val, schedule = evaluate(sequences)
                return obj_val, schedule

            return None, None

        except Exception:
            return None, None

    # === Main Algorithm ===

    best_obj = float('inf')
    best_sequences = None
    best_schedule = None

    # Try Gurobi for small instances
    elapsed = time.time() - global_start
    remaining = args.time_limit - elapsed

    if n <= 25 and m <= 6 and n * m * T <= 25000 and remaining > 10:
        gurobi_time = min(remaining * 0.5, 90)
        result = try_gurobi(gurobi_time)
        if result[0] is not None:
            best_obj, best_schedule = result
            # Reconstruct sequences from schedule for further improvement
            if logger:
                logger.log_solution(best_obj, best_schedule)
            if best_obj <= 0:
                solution = {
                    "objective_value": float(best_obj),
                    "schedule": {str(k): int(v) for k, v in best_schedule.items()}
                }
                with open(args.solution_path, 'w') as f:
                    json.dump(solution, f, indent=2)
                return

    # Generate initial solutions using dispatching rules
    initial_solutions = []

    for k_val in [2.0, 3.0, 4.5, 6.0, 8.0, 12.0]:
        seqs = dispatch_atc(k_val)
        obj, sched = evaluate(seqs)
        initial_solutions.append((obj, seqs, sched))

    seqs = dispatch_wspt()
    obj, sched = evaluate(seqs)
    initial_solutions.append((obj, seqs, sched))

    seqs = dispatch_edd()
    obj, sched = evaluate(seqs)
    initial_solutions.append((obj, seqs, sched))

    seqs = dispatch_wslack()
    obj, sched = evaluate(seqs)
    initial_solutions.append((obj, seqs, sched))

    seqs = dispatch_mdd()
    obj, sched = evaluate(seqs)
    initial_solutions.append((obj, seqs, sched))

    # Find best initial solution
    for obj, seqs, sched in initial_solutions:
        if obj < best_obj:
            best_obj = obj
            best_sequences = [seq[:] for seq in seqs]
            best_schedule = sched

    if logger:
        logger.log_solution(best_obj, best_schedule)

    # Local search on best initial solution
    elapsed = time.time() - global_start
    remaining = args.time_limit - elapsed

    if remaining > 2:
        ls_end = global_start + args.time_limit - max(remaining * 0.45, 2)
        best_sequences, best_obj, best_schedule = local_search(
            best_sequences, best_obj, ls_end
        )
        if logger:
            logger.log_solution(best_obj, best_schedule)

    # Try local search on 2nd and 3rd best initial solutions
    initial_solutions.sort(key=lambda x: x[0])
    for rank in range(1, min(4, len(initial_solutions))):
        elapsed = time.time() - global_start
        remaining = args.time_limit - elapsed
        if remaining > 3:
            obj, seqs, sched = initial_solutions[rank]
            ls_end = time.time() + remaining * 0.15
            new_seqs, new_obj, new_sched = local_search(seqs, obj, ls_end)
            if new_obj < best_obj:
                best_obj = new_obj
                best_sequences = new_seqs
                best_schedule = new_sched
                if logger:
                    logger.log_solution(best_obj, best_schedule)

    # Simulated annealing from best solution
    elapsed = time.time() - global_start
    remaining = args.time_limit - elapsed

    if remaining > 2:
        sa_end = global_start + args.time_limit - 1.0
        sa_sequences, sa_obj, sa_schedule = simulated_annealing(
            best_sequences, best_obj, sa_end
        )
        if sa_obj < best_obj:
            best_obj = sa_obj
            best_sequences = sa_sequences
            best_schedule = sa_schedule

    # Final local search polish
    elapsed = time.time() - global_start
    remaining = args.time_limit - elapsed
    if remaining > 2:
        ls_end = global_start + args.time_limit - 0.5
        best_sequences, best_obj, best_schedule = local_search(
            best_sequences, best_obj, ls_end
        )
        if logger:
            logger.log_solution(best_obj, best_schedule)

    # Write final solution
    solution = {
        "objective_value": float(best_obj),
        "schedule": {str(k): int(v) for k, v in best_schedule.items()}
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == '__main__':
    main()
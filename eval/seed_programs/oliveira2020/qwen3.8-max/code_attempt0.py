#!/usr/bin/env python3
"""
Solver for Parallel Machine Scheduling to Minimize Total Weighted Tardiness
P_m || sum w_j T_j

Approach:
1. Multiple constructive heuristics (EDD, WSPT, ATC with various parameters)
2. Local search with insertion neighborhood
3. Iterated Local Search for further improvement
4. Gurobi MIP for small instances (time-indexed formulation)
"""
import json
import argparse
import time
import math
import random
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    wall_start = time.time()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    jobs_data = instance['jobs']
    n = instance['parameters']['n']
    m = instance['parameters']['m']
    time_horizon = instance['time_horizon']

    # Internal arrays (0-indexed)
    p = [jobs_data[j]['processing_time'] for j in range(n)]
    w = [jobs_data[j]['weight'] for j in range(n)]
    d = [jobs_data[j]['due_date'] for j in range(n)]
    job_ids = [jobs_data[j]['id'] for j in range(n)]

    def elapsed():
        return time.time() - wall_start

    def remaining():
        return args.time_limit - elapsed()

    def evaluate_sequences(sequences):
        """Compute total weighted tardiness from machine sequences."""
        total = 0.0
        for seq in sequences:
            t = 0
            for j in seq:
                t += p[j]
                if t > d[j]:
                    total += w[j] * (t - d[j])
        return total

    def get_start_times(sequences):
        """Get start time for each job."""
        starts = [0] * n
        for seq in sequences:
            t = 0
            for j in seq:
                starts[j] = t
                t += p[j]
        return starts

    def build_solution(sequences):
        """Build solution dict in required output format."""
        starts = get_start_times(sequences)
        total_obj = 0.0
        schedule = []

        for machine_id in range(m):
            prev_id = 0
            for j in sequences[machine_id]:
                completion = starts[j] + p[j]
                tardiness = max(0, completion - d[j])
                wt = float(w[j] * tardiness)
                total_obj += wt
                schedule.append({
                    'machine_id': machine_id,
                    'prev_job': prev_id,
                    'job': job_ids[j],
                    'start_time': starts[j],
                    'completion_time': completion,
                    'weighted_tardiness': wt
                })
                prev_id = job_ids[j]

        schedule.sort(key=lambda x: x['job'])
        return {
            'objective_value': total_obj,
            'schedule': schedule
        }

    # === Constructive Heuristics ===

    def dispatch_rule(rule_func):
        """Schedule jobs using a priority dispatching rule."""
        sequences = [[] for _ in range(m)]
        machine_time = [0] * m
        unscheduled = set(range(n))

        while unscheduled:
            # Pick machine with earliest availability
            best_machine = min(range(m), key=lambda k: machine_time[k])
            current_time = machine_time[best_machine]

            # Pick best job according to rule
            best_job = min(unscheduled, key=lambda j: rule_func(j, current_time))

            unscheduled.remove(best_job)
            sequences[best_machine].append(best_job)
            machine_time[best_machine] += p[best_job]

        return sequences

    p_avg = sum(p) / max(n, 1)

    def edd_rule(j, t):
        return d[j]

    def wspt_rule(j, t):
        return -w[j] / p[j]

    def spt_rule(j, t):
        return p[j]

    def atc_rule_factory(k_param):
        def rule(j, t):
            slack = max(0, d[j] - t - p[j])
            return -(w[j] / p[j]) * math.exp(-slack / (k_param * p_avg + 1e-10))
        return rule

    def cr_rule(j, t):
        """Critical ratio rule."""
        remaining = d[j] - t
        return remaining / p[j]

    def mdd_rule(j, t):
        """Modified due date rule."""
        return max(d[j], t + p[j])

    # Generate initial solutions
    best_sequences = None
    best_obj = float('inf')

    rules = [
        edd_rule,
        wspt_rule,
        spt_rule,
        cr_rule,
        mdd_rule,
        atc_rule_factory(0.5),
        atc_rule_factory(1.0),
        atc_rule_factory(2.0),
        atc_rule_factory(3.0),
        atc_rule_factory(5.0),
    ]

    for rule in rules:
        if remaining() < 1:
            break
        seq = dispatch_rule(rule)
        obj = evaluate_sequences(seq)
        if obj < best_obj:
            best_obj = obj
            best_sequences = [s[:] for s in seq]

    # Log initial best
    if best_sequences is not None:
        sol = build_solution(best_sequences)
        if logger:
            logger.log_solution(sol['objective_value'], sol)

    # === Local Search: Insertion Neighborhood ===

    def find_job_position(sequences, j):
        """Find which machine and position job j is in."""
        for k in range(m):
            for pos, job in enumerate(sequences[k]):
                if job == j:
                    return k, pos
        return -1, -1

    def local_search_first_improvement(sequences, max_time):
        """First-improvement insertion local search."""
        current = [s[:] for s in sequences]
        current_obj = evaluate_sequences(current)
        
        improved = True
        while improved and remaining() > max_time:
            improved = False
            job_order = list(range(n))
            random.shuffle(job_order)
            
            for j in job_order:
                if remaining() < max_time + 0.1:
                    break
                
                src_m, src_pos = find_job_position(current, j)
                
                for tgt_m in range(m):
                    if remaining() < max_time + 0.05:
                        break
                    
                    tgt_seq = current[tgt_m]
                    if tgt_m == src_m:
                        positions = [pos for pos in range(len(tgt_seq) + 1) if pos != src_pos and pos != src_pos + 1]
                    else:
                        positions = range(len(tgt_seq) + 1)
                    
                    for pos in positions:
                        # Build new sequences
                        new_seqs = [s[:] for s in current]
                        new_seqs[src_m].remove(j)
                        new_seqs[tgt_m].insert(pos, j)
                        
                        new_obj = evaluate_sequences(new_seqs)
                        
                        if new_obj < current_obj - 1e-10:
                            current = new_seqs
                            current_obj = new_obj
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        
        return current, current_obj

    def local_search_best_improvement(sequences, max_time):
        """Best-improvement insertion local search."""
        current = [s[:] for s in sequences]
        current_obj = evaluate_sequences(current)
        
        improved = True
        while improved and remaining() > max_time:
            improved = False
            best_move = None
            best_move_obj = current_obj
            
            for j in range(n):
                if remaining() < max_time + 0.5:
                    break
                
                src_m, src_pos = find_job_position(current, j)
                
                for tgt_m in range(m):
                    tgt_seq = current[tgt_m]
                    
                    if tgt_m == src_m:
                        # Skip positions that don't change anything
                        positions = [pos for pos in range(len(tgt_seq) + 1) 
                                   if pos != src_pos and pos != src_pos + 1]
                    else:
                        positions = range(len(tgt_seq) + 1)
                    
                    for pos in positions:
                        new_seqs = [s[:] for s in current]
                        new_seqs[src_m].remove(j)
                        new_seqs[tgt_m].insert(pos, j)
                        
                        new_obj = evaluate_sequences(new_seqs)
                        
                        if new_obj < best_move_obj - 1e-10:
                            best_move_obj = new_obj
                            best_move = (j, src_m, tgt_m, pos)
            
            if best_move is not None:
                j, src_m, tgt_m, pos = best_move
                current[src_m].remove(j)
                current[tgt_m].insert(pos, j)
                current_obj = best_move_obj
                improved = True
        
        return current, current_obj

    # Choose local search strategy based on instance size
    if n <= 60:
        ls_func = local_search_best_improvement
    else:
        ls_func = local_search_first_improvement

    # Run local search on best initial solution
    if remaining() > 2:
        result_seqs, result_obj = ls_func(best_sequences, 1.0)
        if result_obj < best_obj:
            best_obj = result_obj
            best_sequences = [s[:] for s in result_seqs]
            sol = build_solution(best_sequences)
            if logger:
                logger.log_solution(sol['objective_value'], sol)

    # === Iterated Local Search ===
    random.seed(42)

    def perturb(sequences, strength):
        """Random perturbation by moving random jobs."""
        new_seq = [s[:] for s in sequences]
        for _ in range(strength):
            j = random.randint(0, n - 1)
            src_m, _ = find_job_position(new_seq, j)
            new_seq[src_m].remove(j)
            tgt_m = random.randint(0, m - 1)
            pos = random.randint(0, len(new_seq[tgt_m]))
            new_seq[tgt_m].insert(pos, j)
        return new_seq

    ils_strength = max(1, min(5, n // 20))
    
    while remaining() > 3:
        perturbed = perturb(best_sequences, ils_strength)
        
        # Use faster local search for ILS iterations
        time_for_ls = min(remaining() - 1.5, 3.0)
        result_seqs, result_obj = local_search_first_improvement(perturbed, 0.5)
        
        if result_obj < best_obj:
            best_obj = result_obj
            best_sequences = [s[:] for s in result_seqs]
            sol = build_solution(best_sequences)
            if logger:
                logger.log_solution(sol['objective_value'], sol)
        elif remaining() > 5:
            # Occasionally do best-improvement on current best
            result_seqs2, result_obj2 = local_search_best_improvement(best_sequences, 1.0)
            if result_obj2 < best_obj:
                best_obj = result_obj2
                best_sequences = [s[:] for s in result_seqs2]
                sol = build_solution(best_sequences)
                if logger:
                    logger.log_solution(sol['objective_value'], sol)

    # === Try Gurobi for small instances ===
    if remaining() > 5 and n <= 15 and time_horizon <= 300:
        try:
            import gurobipy as gp
            from gurobipy import GRB

            model = gp.Model("scheduling")
            model.setParam('Seed', 0)
            model.setParam('MIPGap', 1e-4)
            model.setParam('NumericFocus', 0)
            model.setParam('Threads', 1)
            model.setParam('TimeLimit', max(1, remaining() - 2))
            model.setParam('OutputFlag', 0)

            T = time_horizon

            # Decision variables: x[j,t,k] = 1 if job j starts at time t on machine k
            x = {}
            for j in range(n):
                for t in range(T - p[j] + 1):
                    for k in range(m):
                        x[j, t, k] = model.addVar(vtype=GRB.BINARY, name=f"x_{j}_{t}_{k}")

            # Tardiness variables
            Tard = {}
            for j in range(n):
                Tard[j] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"Tard_{j}")

            # Each job scheduled exactly once
            for j in range(n):
                model.addConstr(
                    gp.quicksum(x[j, t, k] for t in range(T - p[j] + 1) for k in range(m)) == 1,
                    name=f"assign_{j}"
                )

            # No overlap on each machine at each time point
            for k in range(m):
                for t in range(T):
                    overlapping = []
                    for j in range(n):
                        for s in range(max(0, t - p[j] + 1), min(t + 1, T - p[j] + 1)):
                            overlapping.append(x[j, s, k])
                    if overlapping:
                        model.addConstr(gp.quicksum(overlapping) <= 1, name=f"cap_{k}_{t}")

            # Tardiness constraints: T_j >= C_j - d_j
            for j in range(n):
                completion_expr = gp.quicksum(
                    (t + p[j]) * x[j, t, k]
                    for t in range(T - p[j] + 1)
                    for k in range(m)
                )
                model.addConstr(Tard[j] >= completion_expr - d[j], name=f"tard_{j}")

            # Objective
            model.setObjective(gp.quicksum(w[j] * Tard[j] for j in range(n)), GRB.MINIMIZE)

            model.optimize()

            if model.SolCount > 0:
                gurobi_obj = model.ObjVal
                if gurobi_obj < best_obj - 1e-6:
                    # Extract solution
                    sequences = [[] for _ in range(m)]
                    for j in range(n):
                        for t in range(T - p[j] + 1):
                            for k in range(m):
                                if x[j, t, k].X > 0.5:
                                    sequences[k].append((t, j))
                                    break
                            else:
                                continue
                            break

                    for k in range(m):
                        sequences[k].sort()
                        sequences[k] = [j for _, j in sequences[k]]

                    best_obj = gurobi_obj
                    best_sequences = sequences
                    sol = build_solution(best_sequences)
                    if logger:
                        logger.log_solution(sol['objective_value'], sol)
        except Exception as e:
            pass

    # === Write final solution ===
    sol = build_solution(best_sequences)
    with open(args.solution_path, 'w') as f:
        json.dump(sol, f, indent=2)


if __name__ == '__main__':
    main()
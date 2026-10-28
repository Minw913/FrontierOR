#!/usr/bin/env python3
import json
import argparse
import time
import sys
import random
from solution_logger import SolutionLogger

def read_instance(path):
    with open(path) as f:
        return json.load(f)

def get_processing_times(instance):
    """Get p[j][m] matrix for all jobs and machines."""
    n = instance['num_jobs']
    m = instance['num_machines']
    machine_type = instance['machine_type']
    
    if machine_type == 'unrelated':
        return instance['jobs']['processing_times']
    elif machine_type == 'uniform':
        base_p = instance['jobs']['base_processing_times']
        speeds = instance['machines']['speeds']
        p = [[base_p[j] / speeds[mi] for mi in range(m)] for j in range(n)]
        return p
    else:  # identical
        pt = instance['jobs']['processing_times']
        return pt

def wspt_order(machine_jobs, p, weights, m_idx):
    """Sort jobs on a machine by WSPT rule (decreasing w_j/p_jm)."""
    return sorted(machine_jobs, key=lambda j: weights[j] / max(p[j][m_idx], 1e-10), reverse=True)

def compute_completion_times(sequences, p):
    """Compute completion times for all jobs given machine sequences."""
    ct = {}
    for m_idx, seq in sequences.items():
        current_time = 0
        for j in seq:
            current_time += p[j][m_idx]
            ct[j] = current_time
    return ct

def compute_wct_objective(sequences, p, weights):
    """Compute total weighted completion time."""
    total = 0.0
    for m_idx, seq in sequences.items():
        current_time = 0
        for j in seq:
            current_time += p[j][m_idx]
            total += weights[j] * current_time
    return total

def compute_tardy_objective(tardy_set, weights):
    """Compute total weight of tardy jobs."""
    return sum(weights[j] for j in tardy_set)

def greedy_wct(instance, p):
    """Greedy WSPT list scheduling for weighted completion time."""
    n = instance['num_jobs']
    m = instance['num_machines']
    weights = instance['jobs']['weights']
    machine_type = instance['machine_type']
    
    # Sort jobs by priority
    if machine_type in ['identical', 'uniform']:
        # WSPT order is the same for all machines
        if machine_type == 'identical':
            priority = [weights[j] / max(p[j][0], 1e-10) for j in range(n)]
        else:
            # For uniform, w_j / p_jm = w_j * s_m / base_p_j, order same across machines
            priority = [weights[j] / max(p[j][0], 1e-10) for j in range(n)]
        job_order = sorted(range(n), key=lambda j: priority[j], reverse=True)
    else:
        # For unrelated, use average processing time for priority
        priority = [weights[j] / max(sum(p[j]) / m, 1e-10) for j in range(n)]
        job_order = sorted(range(n), key=lambda j: priority[j], reverse=True)
    
    # Assign jobs to machines greedily
    machine_loads = [0.0] * m
    assignment = [[] for _ in range(m)]
    
    for j in job_order:
        best_m = 0
        best_cost = float('inf')
        for mi in range(m):
            # Cost of adding job j to machine mi
            new_completion = machine_loads[mi] + p[j][mi]
            cost = weights[j] * new_completion
            if cost < best_cost:
                best_cost = cost
                best_m = mi
        assignment[best_m].append(j)
        machine_loads[best_m] += p[j][best_m]
    
    # Sequence each machine by WSPT
    sequences = {}
    for mi in range(m):
        sequences[mi] = wspt_order(assignment[mi], p, weights, mi)
    
    return sequences

def greedy_tardy(instance, p):
    """Greedy heuristic for weighted tardy jobs."""
    n = instance['num_jobs']
    m = instance['num_machines']
    weights = instance['jobs']['weights']
    due_dates = instance['jobs']['due_dates']
    
    # Sort by EDD, then by weight (higher weight first for ties)
    job_order = sorted(range(n), key=lambda j: (due_dates[j], -weights[j]))
    
    # Try to schedule jobs on time using list scheduling
    machine_times = [0.0] * m
    assignment = [[] for _ in range(m)]
    on_time = set()
    tardy = set()
    
    for j in job_order:
        best_m = -1
        best_ct = float('inf')
        for mi in range(m):
            ct = machine_times[mi] + p[j][mi]
            if ct <= due_dates[j] and ct < best_ct:
                best_ct = ct
                best_m = mi
        if best_m >= 0:
            assignment[best_m].append(j)
            machine_times[best_m] = best_ct
            on_time.add(j)
        else:
            tardy.add(j)
    
    # Sequence each machine by EDD
    sequences = {}
    for mi in range(m):
        sequences[mi] = sorted(assignment[mi], key=lambda j: due_dates[j])
    
    return sequences, on_time, tardy

def local_search_wct(instance, p, sequences, start_time, time_limit):
    """Local search improvement for weighted completion time."""
    n = instance['num_jobs']
    m = instance['num_machines']
    weights = instance['jobs']['weights']
    
    best_sequences = {mi: list(seq) for mi, seq in sequences.items()}
    best_obj = compute_wct_objective(best_sequences, p, weights)
    
    improved = True
    while improved and (time.time() - start_time) < time_limit * 0.8:
        improved = False
        
        # Try moving each job to each other machine
        for mi in range(m):
            for pos_i, j in enumerate(list(best_sequences[mi])):
                for mj in range(m):
                    if mi == mj:
                        continue
                    # Move j from mi to mj
                    new_seq_mi = [x for x in best_sequences[mi] if x != j]
                    new_seq_mj = best_sequences[mj] + [j]
                    
                    # Re-sequence both machines by WSPT
                    new_seq_mi = wspt_order(new_seq_mi, p, weights, mi)
                    new_seq_mj = wspt_order(new_seq_mj, p, weights, mj)
                    
                    # Compute new objective (only affected machines)
                    old_ct_mi = 0
                    old_contrib_mi = 0
                    for jj in best_sequences[mi]:
                        old_ct_mi += p[jj][mi]
                        old_contrib_mi += weights[jj] * old_ct_mi
                    
                    old_ct_mj = 0
                    old_contrib_mj = 0
                    for jj in best_sequences[mj]:
                        old_ct_mj += p[jj][mj]
                        old_contrib_mj += weights[jj] * old_ct_mj
                    
                    new_ct_mi = 0
                    new_contrib_mi = 0
                    for jj in new_seq_mi:
                        new_ct_mi += p[jj][mi]
                        new_contrib_mi += weights[jj] * new_ct_mi
                    
                    new_ct_mj = 0
                    new_contrib_mj = 0
                    for jj in new_seq_mj:
                        new_ct_mj += p[jj][mj]
                        new_contrib_mj += weights[jj] * new_ct_mj
                    
                    delta = (new_contrib_mi + new_contrib_mj) - (old_contrib_mi + old_contrib_mj)
                    
                    if delta < -1e-9:
                        best_sequences[mi] = new_seq_mi
                        best_sequences[mj] = new_seq_mj
                        best_obj += delta
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        # Try swapping jobs between machines
        if not improved:
            for mi in range(m):
                for mj in range(mi + 1, m):
                    for j1 in best_sequences[mi]:
                        for j2 in best_sequences[mj]:
                            new_seq_mi = [j2 if x == j1 else x for x in best_sequences[mi]]
                            new_seq_mj = [j1 if x == j2 else x for x in best_sequences[mj]]
                            
                            new_seq_mi = wspt_order(new_seq_mi, p, weights, mi)
                            new_seq_mj = wspt_order(new_seq_mj, p, weights, mj)
                            
                            old_ct = 0
                            old_contrib = 0
                            for jj in best_sequences[mi]:
                                old_ct += p[jj][mi]
                                old_contrib += weights[jj] * old_ct
                            for jj in best_sequences[mj]:
                                old_ct += p[jj][mj]
                                old_contrib += weights[jj] * old_ct
                            
                            # Recompute old contributions separately
                            old_contrib_mi = 0
                            ct_tmp = 0
                            for jj in best_sequences[mi]:
                                ct_tmp += p[jj][mi]
                                old_contrib_mi += weights[jj] * ct_tmp
                            old_contrib_mj = 0
                            ct_tmp = 0
                            for jj in best_sequences[mj]:
                                ct_tmp += p[jj][mj]
                                old_contrib_mj += weights[jj] * ct_tmp
                            
                            new_contrib_mi = 0
                            ct_tmp = 0
                            for jj in new_seq_mi:
                                ct_tmp += p[jj][mi]
                                new_contrib_mi += weights[jj] * ct_tmp
                            new_contrib_mj = 0
                            ct_tmp = 0
                            for jj in new_seq_mj:
                                ct_tmp += p[jj][mj]
                                new_contrib_mj += weights[jj] * ct_tmp
                            
                            delta = (new_contrib_mi + new_contrib_mj) - (old_contrib_mi + old_contrib_mj)
                            
                            if delta < -1e-9:
                                best_sequences[mi] = new_seq_mi
                                best_sequences[mj] = new_seq_mj
                                best_obj += delta
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
    
    return best_sequences, best_obj

def local_search_tardy(instance, p, sequences, on_time, tardy, start_time, time_limit):
    """Local search improvement for weighted tardy jobs."""
    n = instance['num_jobs']
    m = instance['num_machines']
    weights = instance['jobs']['weights']
    due_dates = instance['jobs']['due_dates']
    
    best_sequences = {mi: list(seq) for mi, seq in sequences.items()}
    best_on_time = set(on_time)
    best_tardy = set(tardy)
    best_obj = compute_tardy_objective(best_tardy, weights)
    
    improved = True
    iterations = 0
    max_iterations = 1000
    
    while improved and (time.time() - start_time) < time_limit * 0.8 and iterations < max_iterations:
        improved = False
        iterations += 1
        
        # Try to move a tardy job on-time
        for j in list(best_tardy):
            for mi in range(m):
                # Try inserting j at each position in machine mi's schedule
                seq = best_sequences[mi]
                for pos in range(len(seq) + 1):
                    new_seq = seq[:pos] + [j] + seq[pos:]
                    # Check if all jobs in new_seq meet due dates
                    ct = 0
                    feasible = True
                    for jj in new_seq:
                        ct += p[jj][mi]
                        if ct > due_dates[jj]:
                            feasible = False
                            break
                    if feasible:
                        best_sequences[mi] = new_seq
                        best_tardy.discard(j)
                        best_on_time.add(j)
                        best_obj -= weights[j]
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        if improved:
            continue
        
        # Try swapping: remove a low-weight on-time job, add a high-weight tardy job
        for j_tardy in list(best_tardy):
            for mi in range(m):
                for pos_remove, j_remove in enumerate(best_sequences[mi]):
                    if weights[j_tardy] <= weights[j_remove]:
                        continue
                    # Remove j_remove, try to insert j_tardy
                    new_seq = [x for x in best_sequences[mi] if x != j_remove]
                    for pos in range(len(new_seq) + 1):
                        test_seq = new_seq[:pos] + [j_tardy] + new_seq[pos:]
                        ct = 0
                        feasible = True
                        for jj in test_seq:
                            ct += p[jj][mi]
                            if ct > due_dates[jj]:
                                feasible = False
                                break
                        if feasible:
                            best_sequences[mi] = test_seq
                            best_tardy.discard(j_tardy)
                            best_on_time.add(j_tardy)
                            best_tardy.add(j_remove)
                            best_on_time.discard(j_remove)
                            best_obj += weights[j_remove] - weights[j_tardy]
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        
        if improved:
            continue
        
        # Try moving an on-time job to a different machine to free up space
        for mi in range(m):
            for j_move in list(best_sequences[mi]):
                for mj in range(m):
                    if mi == mj:
                        continue
                    # Try moving j_move to mj
                    new_seq_mj = best_sequences[mj] + [j_move]
                    # Re-sequence by EDD
                    new_seq_mj = sorted(new_seq_mj, key=lambda x: due_dates[x])
                    ct = 0
                    feasible = True
                    for jj in new_seq_mj:
                        ct += p[jj][mj]
                        if ct > due_dates[jj]:
                            feasible = False
                            break
                    if feasible:
                        new_seq_mi = [x for x in best_sequences[mi] if x != j_move]
                        # Check remaining jobs on mi still feasible
                        ct = 0
                        feasible_mi = True
                        for jj in new_seq_mi:
                            ct += p[jj][mi]
                            if ct > due_dates[jj]:
                                feasible_mi = False
                                break
                        if feasible_mi:
                            best_sequences[mi] = new_seq_mi
                            best_sequences[mj] = new_seq_mj
                            # Now try to add tardy jobs to the freed space
                            for j_t in list(best_tardy):
                                for mk in range(m):
                                    seq_k = best_sequences[mk]
                                    for pos in range(len(seq_k) + 1):
                                        test_seq = seq_k[:pos] + [j_t] + seq_k[pos:]
                                        ct2 = 0
                                        feas = True
                                        for jj in test_seq:
                                            ct2 += p[jj][mk]
                                            if ct2 > due_dates[jj]:
                                                feas = False
                                                break
                                        if feas:
                                            best_sequences[mk] = test_seq
                                            best_tardy.discard(j_t)
                                            best_on_time.add(j_t)
                                            best_obj -= weights[j_t]
                                            improved = True
                                            break
                                    if improved:
                                        break
                                if improved:
                                    break
                            if improved:
                                break
                if improved:
                    break
            if improved:
                break
    
    return best_sequences, best_on_time, best_tardy, best_obj

def solve_gurobi_wct(instance, p, time_remaining, warm_sequences=None):
    """Solve weighted completion time with Gurobi MIP."""
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError:
        return None, None
    
    n = instance['num_jobs']
    m = instance['num_machines']
    weights = instance['jobs']['weights']
    
    # Limit problem size for Gurobi
    if n * n * m > 50000:
        return None, None
    
    try:
        model = gp.Model("wct")
        model.setParam('Seed', 0)
        model.setParam('MIPGap', 1e-4)
        model.setParam('NumericFocus', 0)
        model.setParam('Threads', 1)
        model.setParam('TimeLimit', max(time_remaining - 2, 1))
        model.setParam('OutputFlag', 0)
        
        max_p = max(p[j][mi] for j in range(n) for mi in range(m))
        M = n * max_p + 1
        
        # Variables: x[j, mi, k] = 1 if job j is in position k on machine mi
        x = {}
        for j in range(n):
            for mi in range(m):
                for k in range(n):
                    x[j, mi, k] = model.addVar(vtype=GRB.BINARY, name=f"x_{j}_{mi}_{k}")
        
        # Completion time variables
        C = {}
        for j in range(n):
            C[j] = model.addVar(lb=0, ub=GRB.INFINITY, name=f"C_{j}")
        
        model.update()
        
        # Constraint 1: Each job in exactly one position
        for j in range(n):
            model.addConstr(
                gp.quicksum(x[j, mi, k] for mi in range(m) for k in range(n)) == 1,
                name=f"assign_{j}"
            )
        
        # Constraint 2: At most one job per position per machine
        for mi in range(m):
            for k in range(n):
                model.addConstr(
                    gp.quicksum(x[j, mi, k] for j in range(n)) <= 1,
                    name=f"pos_{mi}_{k}"
                )
        
        # Constraint 3: No gaps
        for mi in range(m):
            for k in range(n - 1):
                model.addConstr(
                    gp.quicksum(x[j, mi, k] for j in range(n)) >= 
                    gp.quicksum(x[j, mi, k + 1] for j in range(n)),
                    name=f"nogap_{mi}_{k}"
                )
        
        # Constraint 4: Completion time
        for j in range(n):
            for mi in range(m):
                for k in range(n):
                    # If job j is in position k on machine mi, C[j] >= sum of p for positions 0..k
                    rhs = gp.quicksum(x[j2, mi, l] * p[j2][mi] 
                                     for l in range(k + 1) for j2 in range(n))
                    model.addConstr(
                        C[j] >= rhs - M * (1 - x[j, mi, k]),
                        name=f"ct_{j}_{mi}_{k}"
                    )
        
        # Objective
        model.setObjective(gp.quicksum(weights[j] * C[j] for j in range(n)), GRB.MINIMIZE)
        
        # Warm start
        if warm_sequences is not None:
            for mi in range(m):
                for k, j in enumerate(warm_sequences.get(mi, [])):
                    x[j, mi, k].Start = 1
            # Set non-assigned positions to 0
            for j in range(n):
                for mi in range(m):
                    for k in range(n):
                        if x[j, mi, k].Start is None:
                            x[j, mi, k].Start = 0
        
        model.optimize()
        
        if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL]:
            if model.SolCount > 0:
                # Extract solution
                sequences = {mi: [] for mi in range(m)}
                for j in range(n):
                    for mi in range(m):
                        for k in range(n):
                            if x[j, mi, k].X > 0.5:
                                sequences[mi].append((k, j))
                                break
                
                for mi in range(m):
                    sequences[mi] = [j for k, j in sorted(sequences[mi])]
                
                obj = model.ObjVal
                return sequences, obj
        
        return None, None
    except Exception as e:
        return None, None

def solve_gurobi_tardy(instance, p, time_remaining, warm_sequences=None, warm_on_time=None, warm_tardy=None):
    """Solve weighted tardy jobs with Gurobi MIP."""
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError:
        return None, None, None, None
    
    n = instance['num_jobs']
    m = instance['num_machines']
    weights = instance['jobs']['weights']
    due_dates = instance['jobs']['due_dates']
    
    # Limit problem size
    if n * n * m > 50000:
        return None, None, None, None
    
    try:
        model = gp.Model("tardy")
        model.setParam('Seed', 0)
        model.setParam('MIPGap', 1e-4)
        model.setParam('NumericFocus', 0)
        model.setParam('Threads', 1)
        model.setParam('TimeLimit', max(time_remaining - 2, 1))
        model.setParam('OutputFlag', 0)
        
        max_p = max(p[j][mi] for j in range(n) for mi in range(m))
        M_big = n * max_p + max(due_dates) + 1
        
        # Variables
        x = {}
        for j in range(n):
            for mi in range(m):
                for k in range(n):
                    x[j, mi, k] = model.addVar(vtype=GRB.BINARY, name=f"x_{j}_{mi}_{k}")
        
        u = {}
        for j in range(n):
            u[j] = model.addVar(vtype=GRB.BINARY, name=f"u_{j}")
        
        C = {}
        for j in range(n):
            C[j] = model.addVar(lb=0, ub=GRB.INFINITY, name=f"C_{j}")
        
        model.update()
        
        # Constraint: each job either on-time or tardy
        for j in range(n):
            model.addConstr(
                gp.quicksum(x[j, mi, k] for mi in range(m) for k in range(n)) + u[j] == 1,
                name=f"assign_{j}"
            )
        
        # At most one job per position per machine
        for mi in range(m):
            for k in range(n):
                model.addConstr(
                    gp.quicksum(x[j, mi, k] for j in range(n)) <= 1,
                    name=f"pos_{mi}_{k}"
                )
        
        # No gaps
        for mi in range(m):
            for k in range(n - 1):
                model.addConstr(
                    gp.quicksum(x[j, mi, k] for j in range(n)) >= 
                    gp.quicksum(x[j, mi, k + 1] for j in range(n)),
                    name=f"nogap_{mi}_{k}"
                )
        
        # Completion time constraints
        for j in range(n):
            for mi in range(m):
                for k in range(n):
                    rhs = gp.quicksum(x[j2, mi, l] * p[j2][mi] 
                                     for l in range(k + 1) for j2 in range(n))
                    model.addConstr(
                        C[j] >= rhs - M_big * (1 - x[j, mi, k]),
                        name=f"ct_{j}_{mi}_{k}"
                    )
        
        # Due date constraints
        for j in range(n):
            model.addConstr(
                C[j] <= due_dates[j] + M_big * u[j],
                name=f"due_{j}"
            )
        
        # Objective: minimize weighted tardy jobs
        model.setObjective(gp.quicksum(weights[j] * u[j] for j in range(n)), GRB.MINIMIZE)
        
        # Warm start
        if warm_sequences is not None:
            for mi in range(m):
                for k, j in enumerate(warm_sequences.get(mi, [])):
                    x[j, mi, k].Start = 1
            if warm_tardy is not None:
                for j in warm_tardy:
                    u[j].Start = 1
                for j in range(n):
                    if j not in warm_tardy:
                        u[j].Start = 0
            # Set remaining x to 0
            for j in range(n):
                for mi in range(m):
                    for k in range(n):
                        if x[j, mi, k].Start is None:
                            x[j, mi, k].Start = 0
        
        model.optimize()
        
        if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL]:
            if model.SolCount > 0:
                sequences = {mi: [] for mi in range(m)}
                on_time = set()
                tardy_set = set()
                
                for j in range(n):
                    if u[j].X > 0.5:
                        tardy_set.add(j)
                    else:
                        on_time.add(j)
                        for mi in range(m):
                            for k in range(n):
                                if x[j, mi, k].X > 0.5:
                                    sequences[mi].append((k, j))
                                    break
                
                for mi in range(m):
                    sequences[mi] = [j for k, j in sorted(sequences[mi])]
                
                obj = model.ObjVal
                return sequences, on_time, tardy_set, obj
        
        return None, None, None, None
    except Exception as e:
        return None, None, None, None

def build_wct_solution(instance, sequences, p):
    """Build solution dict for weighted completion time."""
    n = instance['num_jobs']
    m = instance['num_machines']
    weights = instance['jobs']['weights']
    
    completion_times = {}
    schedule = {}
    
    for mi in range(m):
        schedule[str(mi)] = list(sequences.get(mi, []))
        ct = 0
        for j in sequences.get(mi, []):
            ct += p[j][mi]
            completion_times[str(j)] = ct
    
    # Ensure all jobs have completion times
    for j in range(n):
        if str(j) not in completion_times:
            completion_times[str(j)] = 0
    
    obj = sum(weights[j] * completion_times[str(j)] for j in range(n))
    
    solution = {
        "objective_value": obj,
        "schedule": schedule,
        "completion_times": completion_times,
        "tardy_jobs": [],
        "on_time_jobs": list(range(n))
    }
    return solution

def build_tardy_solution(instance, sequences, on_time, tardy_set, p):
    """Build solution dict for weighted tardy jobs."""
    n = instance['num_jobs']
    m = instance['num_machines']
    weights = instance['jobs']['weights']
    
    completion_times = {}
    schedule = {}
    
    for mi in range(m):
        schedule[str(mi)] = list(sequences.get(mi, []))
        ct = 0
        for j in sequences.get(mi, []):
            ct += p[j][mi]
            completion_times[str(j)] = ct
    
    # Tardy jobs get completion time 0 (or we can omit them)
    for j in tardy_set:
        if str(j) not in completion_times:
            completion_times[str(j)] = 0
    
    obj = sum(weights[j] for j in tardy_set)
    
    solution = {
        "objective_value": obj,
        "schedule": schedule,
        "completion_times": completion_times,
        "tardy_jobs": sorted(list(tardy_set)),
        "on_time_jobs": sorted(list(on_time))
    }
    return solution

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    start_time = time.time()
    instance = read_instance(args.instance_path)
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    n = instance['num_jobs']
    m = instance['num_machines']
    weights = instance['jobs']['weights']
    problem_type = instance['problem_type']
    
    p = get_processing_times(instance)
    
    if problem_type == 'weighted_completion_time':
        # Greedy solution
        sequences = greedy_wct(instance, p)
        obj = compute_wct_objective(sequences, p, weights)
        solution = build_wct_solution(instance, sequences, p)
        
        if logger:
            logger.log_solution(obj, solution)
        
        # Local search
        elapsed = time.time() - start_time
        if elapsed < args.time_limit * 0.7:
            sequences, obj = local_search_wct(instance, p, sequences, start_time, args.time_limit)
            solution = build_wct_solution(instance, sequences, p)
            if logger:
                logger.log_solution(obj, solution)
        
        # Try Gurobi
        elapsed = time.time() - start_time
        remaining = args.time_limit - elapsed
        if remaining > 5 and n <= 80:
            gurobi_seq, gurobi_obj = solve_gurobi_wct(instance, p, remaining, sequences)
            if gurobi_seq is not None and gurobi_obj is not None:
                if gurobi_obj < obj - 1e-9:
                    sequences = gurobi_seq
                    obj = gurobi_obj
                    solution = build_wct_solution(instance, sequences, p)
                    if logger:
                        logger.log_solution(obj, solution)
        
        # Additional local search after Gurobi
        elapsed = time.time() - start_time
        if elapsed < args.time_limit * 0.9:
            sequences2, obj2 = local_search_wct(instance, p, sequences, start_time, args.time_limit)
            if obj2 < obj - 1e-9:
                sequences = sequences2
                obj = obj2
                solution = build_wct_solution(instance, sequences, p)
                if logger:
                    logger.log_solution(obj, solution)
    
    else:  # weighted_tardy_jobs
        # Greedy solution
        sequences, on_time, tardy_set = greedy_tardy(instance, p)
        obj = compute_tardy_objective(tardy_set, weights)
        solution = build_tardy_solution(instance, sequences, on_time, tardy_set, p)
        
        if logger:
            logger.log_solution(obj, solution)
        
        # Local search
        elapsed = time.time() - start_time
        if elapsed < args.time_limit * 0.7:
            sequences, on_time, tardy_set, obj = local_search_tardy(
                instance, p, sequences, on_time, tardy_set, start_time, args.time_limit)
            solution = build_tardy_solution(instance, sequences, on_time, tardy_set, p)
            if logger:
                logger.log_solution(obj, solution)
        
        # Try Gurobi
        elapsed = time.time() - start_time
        remaining = args.time_limit - elapsed
        if remaining > 5 and n <= 80:
            g_seq, g_on_time, g_tardy, g_obj = solve_gurobi_tardy(
                instance, p, remaining, sequences, on_time, tardy_set)
            if g_seq is not None and g_obj is not None:
                if g_obj < obj - 1e-9:
                    sequences = g_seq
                    on_time = g_on_time
                    tardy_set = g_tardy
                    obj = g_obj
                    solution = build_tardy_solution(instance, sequences, on_time, tardy_set, p)
                    if logger:
                        logger.log_solution(obj, solution)
        
        # Additional local search
        elapsed = time.time() - start_time
        if elapsed < args.time_limit * 0.9:
            seq2, ot2, t2, obj2 = local_search_tardy(
                instance, p, sequences, on_time, tardy_set, start_time, args.time_limit)
            if obj2 < obj - 1e-9:
                sequences, on_time, tardy_set, obj = seq2, ot2, t2, obj2
                solution = build_tardy_solution(instance, sequences, on_time, tardy_set, p)
                if logger:
                    logger.log_solution(obj, solution)
    
    # Write final solution
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == '__main__':
    main()
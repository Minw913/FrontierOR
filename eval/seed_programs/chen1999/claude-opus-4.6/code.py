import argparse
import json
import time
import sys
from itertools import permutations

def solve_instance(instance, time_limit, logger=None):
    start_time = time.time()
    
    problem_type = instance['problem_type']
    machine_type = instance['machine_type']
    num_jobs = instance['num_jobs']
    num_machines = instance['num_machines']
    weights = instance['jobs']['weights']
    processing_times = instance['jobs']['processing_times']
    due_dates = instance['jobs']['due_dates']
    
    if problem_type == 'weighted_completion_time':
        return solve_wct(instance, time_limit, start_time, logger)
    else:
        return solve_wtj(instance, time_limit, start_time, logger)


def solve_wct(instance, time_limit, start_time, logger):
    num_jobs = instance['num_jobs']
    num_machines = instance['num_machines']
    weights = instance['jobs']['weights']
    processing_times = instance['jobs']['processing_times']
    
    # For weighted completion time on parallel machines:
    # On each machine, optimal order is by WSPT rule (w_j/p_ij descending)
    # For identical machines, this is a well-known problem.
    
    # Try using Gurobi for small-medium instances, heuristic for large
    if num_jobs <= 200 and num_machines <= 20:
        return solve_wct_gurobi(instance, time_limit, start_time, logger)
    else:
        return solve_wct_heuristic(instance, time_limit, start_time, logger)


def solve_wct_heuristic(instance, time_limit, start_time, logger):
    num_jobs = instance['num_jobs']
    num_machines = instance['num_machines']
    weights = instance['jobs']['weights']
    processing_times = instance['jobs']['processing_times']
    
    # Greedy assignment: for each job, assign to the machine that minimizes
    # the increase in total weighted completion time.
    # On each machine, jobs are ordered by WSPT (w/p decreasing).
    
    # Start with a greedy approach
    # For each machine, maintain list of assigned jobs
    machine_jobs = [[] for _ in range(num_machines)]
    
    # Sort jobs by weight/min_processing_time ratio (descending) for greedy
    job_order = sorted(range(num_jobs), key=lambda j: -weights[j] / max(min(processing_times[j]), 1))
    
    best_obj = float('inf')
    best_schedule = None
    
    # Greedy: assign each job to the machine that gives minimum cost increase
    for j in job_order:
        best_machine = -1
        best_cost_increase = float('inf')
        
        for m in range(num_machines):
            # If we add job j to machine m, recompute the optimal schedule cost for that machine
            # With WSPT ordering, cost = sum over jobs ordered by w/p desc: w_i * C_i
            # where C_i = sum of p_k for all k processed before or at i
            
            # Calculate cost increase
            test_jobs = machine_jobs[m] + [j]
            # Sort by WSPT: w/p descending
            test_jobs_sorted = sorted(test_jobs, key=lambda x: -weights[x] / max(processing_times[x][m], 0.001))
            
            new_cost = 0
            cumulative_time = 0
            for jj in test_jobs_sorted:
                cumulative_time += processing_times[jj][m]
                new_cost += weights[jj] * cumulative_time
            
            # Current cost of machine m
            old_jobs_sorted = sorted(machine_jobs[m], key=lambda x: -weights[x] / max(processing_times[x][m], 0.001))
            old_cost = 0
            cumulative_time = 0
            for jj in old_jobs_sorted:
                cumulative_time += processing_times[jj][m]
                old_cost += weights[jj] * cumulative_time
            
            cost_increase = new_cost - old_cost
            if cost_increase < best_cost_increase:
                best_cost_increase = cost_increase
                best_machine = m
        
        machine_jobs[best_machine].append(j)
    
    # Compute final schedule
    schedule = {}
    completion_times = {}
    total_obj = 0
    
    for m in range(num_machines):
        jobs_sorted = sorted(machine_jobs[m], key=lambda x: -weights[x] / max(processing_times[x][m], 0.001))
        schedule[str(m)] = jobs_sorted
        cumulative_time = 0
        for j in jobs_sorted:
            cumulative_time += processing_times[j][m]
            completion_times[str(j)] = cumulative_time
            total_obj += weights[j] * cumulative_time
    
    best_obj = total_obj
    best_schedule = (dict(schedule), dict(completion_times))
    
    if logger:
        logger.log(best_obj)
    
    # Local search: try swapping jobs between machines and moving jobs
    improved = True
    iteration = 0
    while improved and time.time() - start_time < time_limit - 1:
        improved = False
        iteration += 1
        
        # Try moving each job to a different machine
        for j in range(num_jobs):
            if time.time() - start_time >= time_limit - 1:
                break
            
            # Find current machine
            current_machine = -1
            for m in range(num_machines):
                if j in machine_jobs[m]:
                    current_machine = m
                    break
            
            # Try moving to each other machine
            for m in range(num_machines):
                if m == current_machine:
                    continue
                
                # Compute cost without j on current machine
                old_jobs_cm = [x for x in machine_jobs[current_machine] if x != j]
                old_sorted_cm = sorted(old_jobs_cm, key=lambda x: -weights[x] / max(processing_times[x][current_machine], 0.001))
                cost_cm_without = 0
                ct = 0
                for jj in old_sorted_cm:
                    ct += processing_times[jj][current_machine]
                    cost_cm_without += weights[jj] * ct
                
                # Current cost of current machine
                old_sorted_cm_full = sorted(machine_jobs[current_machine], key=lambda x: -weights[x] / max(processing_times[x][current_machine], 0.001))
                cost_cm_with = 0
                ct = 0
                for jj in old_sorted_cm_full:
                    ct += processing_times[jj][current_machine]
                    cost_cm_with += weights[jj] * ct
                
                # Cost of target machine with j
                new_jobs_m = machine_jobs[m] + [j]
                new_sorted_m = sorted(new_jobs_m, key=lambda x: -weights[x] / max(processing_times[x][m], 0.001))
                cost_m_with = 0
                ct = 0
                for jj in new_sorted_m:
                    ct += processing_times[jj][m]
                    cost_m_with += weights[jj] * ct
                
                # Current cost of target machine
                old_sorted_m = sorted(machine_jobs[m], key=lambda x: -weights[x] / max(processing_times[x][m], 0.001))
                cost_m_without = 0
                ct = 0
                for jj in old_sorted_m:
                    ct += processing_times[jj][m]
                    cost_m_without += weights[jj] * ct
                
                delta = (cost_cm_without + cost_m_with) - (cost_cm_with + cost_m_without)
                if delta < -0.5:
                    machine_jobs[current_machine].remove(j)
                    machine_jobs[m].append(j)
                    best_obj += delta
                    improved = True
                    if logger:
                        logger.log(best_obj)
                    break
        
        # Try swapping pairs of jobs between different machines
        if not improved or time.time() - start_time >= time_limit - 1:
            break
            
    # Recompute final solution
    schedule = {}
    completion_times = {}
    total_obj = 0
    
    for m in range(num_machines):
        jobs_sorted = sorted(machine_jobs[m], key=lambda x: -weights[x] / max(processing_times[x][m], 0.001))
        schedule[str(m)] = jobs_sorted
        ct = 0
        for j in jobs_sorted:
            ct += processing_times[j][m]
            completion_times[str(j)] = ct
            total_obj += weights[j] * ct
    
    return {
        "objective_value": total_obj,
        "schedule": schedule,
        "completion_times": completion_times,
        "tardy_jobs": [],
        "on_time_jobs": []
    }


def solve_wct_gurobi(instance, time_limit, start_time, logger):
    import gurobipy as gp
    from gurobipy import GRB
    
    num_jobs = instance['num_jobs']
    num_machines = instance['num_machines']
    weights = instance['jobs']['weights']
    processing_times = instance['jobs']['processing_times']
    
    # First, get a good heuristic solution
    heuristic_sol = solve_wct_heuristic(instance, min(time_limit * 0.3, 10), start_time, logger)
    
    remaining_time = time_limit - (time.time() - start_time) - 1
    if remaining_time < 2:
        return heuristic_sol
    
    # MIP formulation using assignment + position-based or Smith's rule
    # For the WCT problem with unrelated machines, we use:
    # x[j,m] = 1 if job j assigned to machine m
    # On each machine, jobs are ordered by WSPT rule (this is optimal for single machine WCT)
    # So the objective contribution depends on which jobs are assigned together.
    
    # Smith's rule decomposition:
    # Total WCT = sum over machines: sum over pairs (i,j) on same machine where i precedes j: w_j * p_i
    # Plus sum_j w_j * p_j (for each job's own processing time contribution)
    
    # Actually for WSPT ordering on a single machine:
    # WCT = sum_j w_j * p_j + sum_{i<j in WSPT order} w_j * p_i
    # This is complex to linearize directly.
    
    # Alternative: use a simpler assignment-based formulation
    # with position-based variables for small instances
    
    if num_jobs <= 30:
        return solve_wct_position_gurobi(instance, time_limit, start_time, logger, heuristic_sol)
    
    # For larger instances, use the heuristic + local search
    return heuristic_sol


def solve_wct_position_gurobi(instance, time_limit, start_time, logger, heuristic_sol):
    import gurobipy as gp
    from gurobipy import GRB
    
    num_jobs = instance['num_jobs']
    num_machines = instance['num_machines']
    weights = instance['jobs']['weights']
    processing_times = instance['jobs']['processing_times']
    
    remaining_time = time_limit - (time.time() - start_time) - 1
    if remaining_time < 2:
        return heuristic_sol
    
    # Position-based formulation:
    # x[j,m,k] = 1 if job j is assigned to position k on machine m
    # Positions 0..num_jobs-1 on each machine
    
    try:
        model = gp.Model("WCT")
        model.Params.TimeLimit = remaining_time
        model.Params.OutputFlag = 0
        model.Params.Threads = 1
        
        # Variables
        x = {}
        for j in range(num_jobs):
            for m in range(num_machines):
                for k in range(num_jobs):
                    x[j, m, k] = model.addVar(vtype=GRB.BINARY, name=f"x_{j}_{m}_{k}")
        
        # Each job assigned to exactly one (machine, position)
        for j in range(num_jobs):
            model.addConstr(
                gp.quicksum(x[j, m, k] for m in range(num_machines) for k in range(num_jobs)) == 1
            )
        
        # Each (machine, position) has at most one job
        for m in range(num_machines):
            for k in range(num_jobs):
                model.addConstr(
                    gp.quicksum(x[j, m, k] for j in range(num_jobs)) <= 1
                )
        
        # No gaps: if position k is used, position k-1 must be used
        for m in range(num_machines):
            for k in range(1, num_jobs):
                model.addConstr(
                    gp.quicksum(x[j, m, k] for j in range(num_jobs)) <=
                    gp.quicksum(x[j, m, k-1] for j in range(num_jobs))
                )
        
        # Objective: minimize total weighted completion time
        # If job j is at position k on machine m, its completion time is
        # sum of processing times of jobs at positions 0..k on machine m.
        # 
        # Completion time contribution: w_j * C_j
        # C_j = sum_{k'=0}^{k} p_{j', m} where j' is at position k'
        # = sum_{k'=0}^{k} sum_{j'} p_{j',m} * x[j',m,k']
        #
        # This is quadratic. We can linearize:
        # w_j * C_j = w_j * sum_{k'<=k} sum_{j'} p_{j',m} * x[j',m,k'] * x[j,m,k]
        #
        # Alternative: use the fact that the contribution of job j' at position k' to
        # the total WCT equals p_{j',m} * sum of weights of jobs at positions k' to num_jobs-1 on machine m.
        # 
        # Total WCT = sum_m sum_{k'} (processing time at position k') * (sum of weights at positions k' to end)
        # = sum_m sum_{k'} sum_j p_{j,m} * x[j,m,k'] * sum_{k''>=k'} sum_{j'} w_{j'} * x[j',m,k'']
        # Still quadratic.
        #
        # Standard linearization: define y[j,j',m] = 1 if both j and j' are on machine m with j' at same or later position
        # That's O(n^2 * m) variables...
        
        # Better approach: use the positional weight formulation.
        # If job j is at position k (0-indexed) on machine m with K total jobs on m,
        # then its processing time contributes to its own and all later jobs' completion times.
        # Specifically, p_{j,m} is added to the completion time of (K - k) jobs.
        # But we don't know K in advance.
        #
        # Actually, the total weighted completion time can be written as:
        # sum_j w_j * C_j = sum_m sum_{k=0}^{K_m-1} sum_{k'=0}^{k} p_{sigma_m(k'), m} * w_{sigma_m(k)}
        # = sum_m sum_{k'=0}^{K_m-1} p_{sigma_m(k'), m} * sum_{k=k'}^{K_m-1} w_{sigma_m(k)}
        # = sum_m sum_{k'=0}^{K_m-1} p_{sigma_m(k'), m} * W_remaining(k')
        #
        # With position variables:
        # = sum_m sum_{k'} sum_j p_{j,m} * x[j,m,k'] * sum_{k>=k'} sum_{j'} w_{j'} * x[j',m,k]
        # This is still quadratic.
        
        # Let's use a different approach. Define:
        # For each position k on machine m, the "positional multiplier" for a job at position k
        # depends on how many jobs come after it. But this is variable.
        # 
        # Actually, in the WCT problem with WSPT on each machine, we can note:
        # Total WCT = sum_j w_j * p_j(m_j) + sum of w_j * p_i(m) for all pairs (i,j) on same machine 
        # where i precedes j.
        # 
        # Using WSPT ordering, the second term is determined once we know the assignment.
        # But even computing it requires knowing which pairs end up together.
        
        # Let me use a simpler formulation: for small instances, the position-based approach
        # can work with a coefficient trick.
        
        # Total WCT = sum_m sum_k (sum_{j} x[j,m,k] * w_j) * (sum_{k'<=k} sum_{j'} x[j',m,k'] * p_{j',m})
        # 
        # Alternative known linearization for identical machines (all p same across machines):
        # The contribution of job j at position k on machine m is:
        # w_j * (sum_{k'=0}^{k} p_{j',m} for j' at positions 0..k)
        
        # For general case, let's use the "agreement" property:
        # On any given machine, the optimal order is WSPT.
        # So the problem reduces to: assign jobs to machines, then order by WSPT.
        
        # Let's formulate as pure assignment:
        # y[j,m] = 1 if job j assigned to machine m
        # Then total WCT for machine m = computed from the WSPT ordering of assigned jobs.
        # This is nonlinear but we can handle it with lazy constraints or callbacks.
        
        # Actually for small n, let's just use pairwise variables.
        # z[i,j,m] = 1 if both i and j are assigned to machine m (i != j)
        # Then WCT = sum_j w_j * p_{j,m_j} + sum_{i!=j on same machine, i before j} w_j * p_{i,m}
        # With WSPT, i before j iff w_i/p_{i,m} >= w_j/p_{j,m}
        
        # This still requires knowing which machine they share to determine ordering...
        # Let me think differently.
        
        # For this problem size, let's just try a different formulation.
        # Let y[j,m] = 1 if job j assigned to machine m.
        # For a set S of jobs on machine m with WSPT ordering, the WCT is:
        # sum_j in S: w_j * p_{j,m} + sum_{i,j in S, i before j in WSPT} w_j * p_{i,m}
        
        # The pair (i,j) with i before j in WSPT on machine m means w_i * p_{j,m} >= w_j * p_{i,m}
        # (or equivalently w_i/p_{i,m} >= w_j/p_{j,m}).
        
        # For unrelated machines, the WSPT ratio depends on the machine!
        # So whether i precedes j depends on which machine they share.
        
        # This makes it complex. Let me use a cleaner formulation.
        
        model.remove(model.getVars())
        model.remove(model.getConstrs())
        model.update()
        
        # Reformulate with assignment + completion time variables
        y = {}
        for j in range(num_jobs):
            for m in range(num_machines):
                y[j, m] = model.addVar(vtype=GRB.BINARY, name=f"y_{j}_{m}")
        
        C = {}
        for j in range(num_jobs):
            C[j] = model.addVar(lb=0, name=f"C_{j}")
        
        # Each job assigned to exactly one machine
        for j in range(num_jobs):
            model.addConstr(gp.quicksum(y[j, m] for m in range(num_machines)) == 1)
        
        # For pairwise interactions on same machine:
        # If i and j on same machine m, and i before j in WSPT on m, then C_j >= C_i + p_{j,m}
        # But C variables are continuous and we need to link them to assignments.
        
        # This gets complex. Let me use the position-based formulation properly.
        # For num_jobs <= 30, we have at most 30 positions per machine.
        
        model.remove(model.getVars())
        model.remove(model.getConstrs())
        model.update()
        
        # Position-based: x[j,m,k] = 1 if job j is at position k on machine m
        x = {}
        for j in range(num_jobs):
            for m in range(num_machines):
                for k in range(num_jobs):
                    x[j, m, k] = model.addVar(vtype=GRB.BINARY, name=f"x_{j}_{m}_{k}")
        
        # Each job exactly one position on one machine
        for j in range(num_jobs):
            model.addConstr(
                gp.quicksum(x[j, m, k] for m in range(num_machines) for k in range(num_jobs)) == 1
            )
        
        # Each (machine, position) at most one job
        for m in range(num_machines):
            for k in range(num_jobs):
                model.addConstr(
                    gp.quicksum(x[j, m, k] for j in range(num_jobs)) <= 1
                )
        
        # No gaps
        for m in range(num_machines):
            for k in range(1, num_jobs):
                model.addConstr(
                    gp.quicksum(x[j, m, k] for j in range(num_jobs)) <=
                    gp.quicksum(x[j, m, k-1] for j in range(num_jobs))
                )
        
        # Objective using the identity:
        # Total WCT = sum_m sum_{k} sum_j x[j,m,k] * w_j * (sum_{k'<=k} sum_{j'} x[j',m,k'] * p_{j',m})
        # Linearize with auxiliary variables:
        # Let q[j,j',m,k,k'] represent the product x[j,m,k] * x[j',m,k'] for k' <= k
        # This is too many variables.
        
        # Better: note that total WCT = sum_m sum_{k} sum_{k'<=k} 
        #   (sum_j w_j * x[j,m,k]) * (sum_{j'} p_{j',m} * x[j',m,k'])
        
        # Define for each (m,k): W[m,k] = sum_j w_j * x[j,m,k] and P[m,k] = sum_j p_{j,m} * x[j,m,k]
        # These are linear. Then WCT = sum_m sum_k W[m,k] * (sum_{k'<=k} P[m,k'])
        # = sum_m sum_k W[m,k] * CumP[m,k] where CumP[m,k] = sum_{k'<=k} P[m,k']
        # W[m,k] * CumP[m,k] is still a product of two linear expressions, which is quadratic.
        
        # We need to linearize. Since at most one job is at each position:
        # W[m,k] = sum_j w_j * x[j,m,k], and this equals w_j if job j is at position k, 0 otherwise.
        # Similarly for P[m,k].
        
        # So the product W[m,k] * CumP[m,k] = (sum_j w_j * x[j,m,k]) * CumP[m,k]
        # = sum_j w_j * x[j,m,k] * CumP[m,k]
        
        # We can introduce variables z[j,m,k] = x[j,m,k] * CumP[m,k]
        # Since x is binary: z[j,m,k] >= CumP[m,k] - M*(1 - x[j,m,k]) and z >= 0
        # And z[j,m,k] <= CumP[m,k] and z[j,m,k] <= M * x[j,m,k]
        
        # But CumP[m,k] is also a variable (linear in x).
        # This introduces O(n^2 * m) auxiliary variables and constraints.
        
        # For n=30, m=5, that's 4500 auxiliary variables. Should be fine.
        
        # Define CumP[m,k]
        CumP = {}
        for m in range(num_machines):
            for k in range(num_jobs):
                CumP[m, k] = model.addVar(lb=0, name=f"CumP_{m}_{k}")
        
        # CumP[m,k] = sum_{k'=0}^{k} sum_j p_{j,m} * x[j,m,k']
        for m in range(num_machines):
            for k in range(num_jobs):
                if k == 0:
                    model.addConstr(
                        CumP[m, k] == gp.quicksum(processing_times[j][m] * x[j, m, 0] for j in range(num_jobs))
                    )
                else:
                    model.addConstr(
                        CumP[m, k] == CumP[m, k-1] + gp.quicksum(processing_times[j][m] * x[j, m, k] for j in range(num_jobs))
                    )
        
        # Big M for CumP
        max_total_p = max(sum(max(processing_times[j][m] for j in range(num_jobs)) for j2 in range(num_jobs)) for m in range(num_machines))
        # Actually a tighter bound:
        M_val = sum(max(processing_times[j][m] for m in range(num_machines)) for j in range(num_jobs))
        
        # z[j,m,k] = x[j,m,k] * CumP[m,k]
        z = {}
        for j in range(num_jobs):
            for m in range(num_machines):
                for k in range(num_jobs):
                    z[j, m, k] = model.addVar(lb=0, name=f"z_{j}_{m}_{k}")
        
        for j in range(num_jobs):
            for m in range(num_machines):
                for k in range(num_jobs):
                    model.addConstr(z[j, m, k] <= M_val * x[j, m, k])
                    model.addConstr(z[j, m, k] <= CumP[m, k])
                    model.addConstr(z[j, m, k] >= CumP[m, k] - M_val * (1 - x[j, m, k]))
        
        # Objective
        model.setObjective(
            gp.quicksum(weights[j] * z[j, m, k] for j in range(num_jobs) for m in range(num_machines) for k in range(num_jobs)),
            GRB.MINIMIZE
        )
        
        # Set initial solution from heuristic
        for j in range(num_jobs):
            for m in range(num_machines):
                for k in range(num_jobs):
                    x[j, m, k].Start = 0
        
        for m_str, job_list in heuristic_sol['schedule'].items():
            m = int(m_str)
            for k, j in enumerate(job_list):
                x[j, m, k].Start = 1
        
        model.update()
        
        # Add callback for logging
        best_found = [heuristic_sol['objective_value']]
        best_sol_data = [heuristic_sol]
        
        def callback(model, where):
            if where == GRB.Callback.MIPSOL:
                obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if obj < best_found[0] - 0.5:
                    best_found[0] = obj
                    if logger:
                        logger.log(obj)
        
        model.optimize(callback)
        
        if model.SolCount > 0:
            # Extract solution
            schedule = {}
            completion_times = {}
            total_obj = 0
            
            for m in range(num_machines):
                job_list = []
                for k in range(num_jobs):
                    for j in range(num_jobs):
                        if x[j, m, k].X > 0.5:
                            job_list.append(j)
                            break
                schedule[str(m)] = job_list
                ct = 0
                for j in job_list:
                    ct += processing_times[j][m]
                    completion_times[str(j)] = ct
                    total_obj += weights[j] * ct
            
            if total_obj < best_sol_data[0]['objective_value']:
                return {
                    "objective_value": total_obj,
                    "schedule": schedule,
                    "completion_times": completion_times,
                    "tardy_jobs": [],
                    "on_time_jobs": []
                }
        
        return best_sol_data[0]
        
    except Exception as e:
        return heuristic_sol


def solve_wtj(instance, time_limit, start_time, logger):
    num_jobs = instance['num_jobs']
    num_machines = instance['num_machines']
    weights = instance['jobs']['weights']
    processing_times = instance['jobs']['processing_times']
    due_dates = instance['jobs']['due_dates']
    
    if num_jobs <= 100 and num_machines <= 10:
        return solve_wtj_gurobi(instance, time_limit, start_time, logger)
    else:
        return solve_wtj_heuristic(instance, time_limit, start_time, logger)


def solve_wtj_heuristic(instance, time_limit, start_time, logger):
    num_jobs = instance['num_jobs']
    num_machines = instance['num_machines']
    weights = instance['jobs']['weights']
    processing_times = instance['jobs']['processing_times']
    due_dates = instance['jobs']['due_dates']
    
    # For weighted tardy jobs: minimize sum of weights of tardy jobs
    # We want to maximize the total weight of on-time jobs.
    
    # Greedy: try to schedule jobs with highest weight first, assign to machine
    # where they can finish by their due date with minimum impact.
    
    machine_load = [0] * num_machines  # Current completion time on each machine
    machine_jobs = [[] for _ in range(num_machines)]
    on_time = []
    tardy = []
    
    # Sort jobs by due date (EDD within machines is often good for feasibility)
    # But we also want to prioritize high-weight jobs.
    # Use a greedy approach: sort by due date, try to fit each job.
    
    # Actually, for single machine weighted tardy jobs, the approach is:
    # Consider jobs in EDD order, and use a "rejection" scheme where we drop the
    # lowest weight job if adding a new job makes it infeasible.
    
    # For multiple machines, let's use a greedy assignment.
    
    # Sort by due date
    job_order = sorted(range(num_jobs), key=lambda j: due_dates[j])
    
    # For each job in EDD order, try to assign to the machine where it can finish on time
    # with the earliest completion time. If no machine works, mark as tardy.
    for j in job_order:
        best_machine = -1
        best_finish = float('inf')
        
        for m in range(num_machines):
            finish = machine_load[m] + processing_times[j][m]
            if finish <= due_dates[j] and finish < best_finish:
                best_finish = finish
                best_machine = m
        
        if best_machine >= 0:
            machine_jobs[best_machine].append(j)
            machine_load[best_machine] += processing_times[j][best_machine]
            on_time.append(j)
        else:
            tardy.append(j)
    
    tardy_weight = sum(weights[j] for j in tardy)
    
    if logger:
        logger.log(tardy_weight)
    
    best_obj = tardy_weight
    best_machine_jobs = [list(mj) for mj in machine_jobs]
    best_on_time = list(on_time)
    best_tardy = list(tardy)
    
    # Improvement: try swapping tardy jobs with on-time jobs if beneficial
    improved = True
    while improved and time.time() - start_time < time_limit - 1:
        improved = False
        
        # Try replacing a low-weight on-time job with a high-weight tardy job
        for tj in sorted(tardy, key=lambda j: -weights[j]):
            if time.time() - start_time >= time_limit - 1:
                break
            
            for m in range(num_machines):
                # Try adding tj to machine m, removing some on-time jobs to make it fit
                # Find the best single swap
                for idx, oj in enumerate(best_machine_jobs[m]):
                    if weights[tj] > weights[oj]:
                        # Try replacing oj with tj
                        new_jobs = [x for x in best_machine_jobs[m] if x != oj]
                        # Insert tj and check feasibility (need to reorder by EDD)
                        new_jobs.append(tj)
                        new_jobs.sort(key=lambda j: due_dates[j])
                        
                        feasible = True
                        ct = 0
                        for jj in new_jobs:
                            ct += processing_times[jj][m]
                            if ct > due_dates[jj]:
                                feasible = False
                                break
                        
                        if feasible:
                            delta = weights[oj] - weights[tj]  # Change in tardy weight
                            if delta < 0:
                                best_machine_jobs[m] = new_jobs
                                best_tardy.remove(tj)
                                best_tardy.append(oj)
                                best_on_time.remove(oj)
                                best_on_time.append(tj)
                                best_obj += delta
                                tardy = list(best_tardy)
                                improved = True
                                if logger:
                                    logger.log(best_obj)
                                break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            # Try adding tardy jobs directly to machines
            for tj in sorted(best_tardy, key=lambda j: -weights[j]):
                if time.time() - start_time >= time_limit - 1:
                    break
                
                for m in range(num_machines):
                    new_jobs = best_machine_jobs[m] + [tj]
                    new_jobs.sort(key=lambda j: due_dates[j])
                    
                    feasible = True
                    ct = 0
                    for jj in new_jobs:
                        ct += processing_times[jj][m]
                        if ct > due_dates[jj]:
                            feasible = False
                            break
                    
                    if feasible:
                        best_machine_jobs[m] = new_jobs
                        best_tardy.remove(tj)
                        best_on_time.append(tj)
                        best_obj -= weights[tj]
                        tardy = list(best_tardy)
                        improved = True
                        if logger:
                            logger.log(best_obj)
                        break
                if improved:
                    break
    
    # Build solution
    schedule = {}
    completion_times = {}
    
    for m in range(num_machines):
        jobs_sorted = sorted(best_machine_jobs[m], key=lambda j: due_dates[j])
        schedule[str(m)] = jobs_sorted
        ct = 0
        for j in jobs_sorted:
            ct += processing_times[j][m]
            completion_times[str(j)] = ct
    
    return {
        "objective_value": best_obj,
        "schedule": schedule,
        "completion_times": completion_times,
        "tardy_jobs": best_tardy,
        "on_time_jobs": best_on_time
    }


def solve_wtj_gurobi(instance, time_limit, start_time, logger):
    import gurobipy as gp
    from gurobipy import GRB
    
    num_jobs = instance['num_jobs']
    num_machines = instance['num_machines']
    weights = instance['jobs']['weights']
    processing_times = instance['jobs']['processing_times']
    due_dates = instance['jobs']['due_dates']
    
    # Get heuristic solution first
    heuristic_sol = solve_wtj_heuristic(instance, min(time_limit * 0.2, 5), start_time, logger)
    
    remaining_time = time_limit - (time.time() - start_time) - 1
    if remaining_time < 2:
        return heuristic_sol
    
    try:
        model = gp.Model("WTJ")
        model.Params.TimeLimit = remaining_time
        model.Params.OutputFlag = 0
        model.Params.Threads = 1
        
        # Variables:
        # y[j,m] = 1 if job j is assigned to machine m (on time)
        # u[j] = 1 if job j is tardy
        # For ordering on each machine, we use position-based or time-indexed
        
        # For the tardy jobs problem, we need to determine:
        # 1. Which jobs are on time
        # 2. Assignment to machines
        # 3. Order on each machine (must respect due dates)
        
        # Formulation with ordering constraints:
        # y[j,m] = 1 if job j assigned to machine m (on time)
        # u[j] = 1 if job j is tardy
        # C[j] = completion time of job j (if on time)
        
        y = {}
        for j in range(num_jobs):
            for m in range(num_machines):
                y[j, m] = model.addVar(vtype=GRB.BINARY, name=f"y_{j}_{m}")
        
        u = {}
        for j in range(num_jobs):
            u[j] = model.addVar(vtype=GRB.BINARY, name=f"u_{j}")
        
        # Each job either on one machine or tardy
        for j in range(num_jobs):
            model.addConstr(
                gp.quicksum(y[j, m] for m in range(num_machines)) + u[j] == 1
            )
        
        # Objective: minimize weighted tardy
        model.setObjective(
            gp.quicksum(weights[j] * u[j] for j in range(num_jobs)),
            GRB.MINIMIZE
        )
        
        # For ordering: use position-based variables
        # x[j,m,k] = 1 if job j is at position k on machine m
        # Only create for feasible assignments (where p[j][m] <= d[j])
        
        # Precompute feasibility: job j can only be on machine m if p[j][m] <= d[j]
        feasible = {}
        for j in range(num_jobs):
            for m in range(num_machines):
                feasible[j, m] = processing_times[j][m] <= due_dates[j]
        
        # Force infeasible assignments to 0
        for j in range(num_jobs):
            for m in range(num_machines):
                if not feasible[j, m]:
                    model.addConstr(y[j, m] == 0)
        
        # Position-based approach for small instances
        if num_jobs <= 50:
            x = {}
            for j in range(num_jobs):
                for m in range(num_machines):
                    if not feasible[j, m]:
                        continue
                    for k in range(num_jobs):
                        x[j, m, k] = model.addVar(vtype=GRB.BINARY, name=f"x_{j}_{m}_{k}")
            
            # Link y and x
            for j in range(num_jobs):
                for m in range(num_machines):
                    if feasible[j, m]:
                        model.addConstr(
                            y[j, m] == gp.quicksum(x[j, m, k] for k in range(num_jobs) if (j, m, k) in x)
                        )
            
            # Each position on each machine has at most one job
            for m in range(num_machines):
                for k in range(num_jobs):
                    expr = gp.quicksum(x[j, m, k] for j in range(num_jobs) if (j, m, k) in x)
                    model.addConstr(expr <= 1)
            
            # No gaps
            for m in range(num_machines):
                for k in range(1, num_jobs):
                    model.addConstr(
                        gp.quicksum(x[j, m, k] for j in range(num_jobs) if (j, m, k) in x) <=
                        gp.quicksum(x[j, m, k-1] for j in range(num_jobs) if (j, m, k-1) in x)
                    )
            
            # Completion time constraints: for job j at position k on machine m,
            # C_j = sum of processing times of jobs at positions 0..k
            # C_j <= d_j
            
            # CumP[m,k] = cumulative processing time through position k
            CumP = {}
            for m in range(num_machines):
                for k in range(num_jobs):
                    CumP[m, k] = model.addVar(lb=0, name=f"CumP_{m}_{k}")
            
            for m in range(num_machines):
                for k in range(num_jobs):
                    if k == 0:
                        model.addConstr(
                            CumP[m, k] == gp.quicksum(
                                processing_times[j][m] * x[j, m, 0]
                                for j in range(num_jobs) if (j, m, 0) in x
                            )
                        )
                    else:
                        model.addConstr(
                            CumP[m, k] == CumP[m, k-1] + gp.quicksum(
                                processing_times[j][m] * x[j, m, k]
                                for j in range(num_jobs) if (j, m, k) in x
                            )
                        )
            
            # Due date constraint: if job j at position k on machine m, then CumP[m,k] <= d_j
            M_val = sum(max(processing_times[j][m] for m in range(num_machines)) for j in range(num_jobs))
            
            for j in range(num_jobs):
                for m in range(num_machines):
                    if not feasible[j, m]:
                        continue
                    for k in range(num_jobs):
                        if (j, m, k) in x:
                            model.addConstr(
                                CumP[m, k] <= due_dates[j] + M_val * (1 - x[j, m, k])
                            )
        else:
            # For larger instances, use time-indexed or simpler constraints
            # Use completion time variables with big-M disjunctive constraints
            C = {}
            for j in range(num_jobs):
                C[j] = model.addVar(lb=0, name=f"C_{j}")
            
            # If job j is on time: C[j] <= d[j]
            M_val = sum(max(processing_times[j][m] for m in range(num_machines)) for j in range(num_jobs))
            for j in range(num_jobs):
                model.addConstr(C[j] <= due_dates[j] + M_val * u[j])
            
            # Disjunctive constraints: for jobs on same machine, one must complete before the other starts
            # s[i,j,m] = 1 if both i and j on machine m and i before j
            s = {}
            for i in range(num_jobs):
                for j in range(i+1, num_jobs):
                    for m in range(num_machines):
                        if feasible[i, m] and feasible[j, m]:
                            s[i, j, m] = model.addVar(vtype=GRB.BINARY, name=f"s_{i}_{j}_{m}")
            
            # C[j] >= C[i] + p[j][m] - M*(1 - s[i,j,m])
            # C[i] >= C[j] + p[i][m] - M*(1 - (y[i,m] + y[j,m] - 1 - s[i,j,m]))
            # This gets very complex and large. Let me use a simpler approach.
            
            # Actually, for the tardy jobs problem, the key insight is:
            # On each machine, on-time jobs should be ordered by EDD (Moore's algorithm).
            # So we just need the assignment and subset selection.
            
            # Constraint: the jobs assigned to machine m must be feasibly schedulable
            # in EDD order, meaning each job's cumulative processing time <= its due date.
            
            # This is hard to express as a linear constraint without ordering variables.
            # We could add them as lazy constraints via callback.
            
            # For now, let me use a different approach:
            # Aggregate constraint: for each machine m and each job j on m,
            # the total processing time of jobs on m with due date <= d_j must be <= d_j.
            
            # This is necessary but not sufficient for feasibility under EDD.
            # Actually, this IS sufficient if jobs are ordered by EDD!
            # Because in EDD order, the cumulative time through job j equals
            # the sum of processing times of all jobs with d <= d_j (or at least the ones before j).
            # If we process in EDD order and the cumulative time of all jobs with d_i <= d_j is <= d_j,
            # then job j finishes on time.
            
            # So: for each machine m and each job j assigned to m,
            # sum_{i: d_i <= d_j, i assigned to m} p_i_m <= d_j
            
            # Let's sort jobs by due date
            sorted_jobs = sorted(range(num_jobs), key=lambda j: due_dates[j])
            
            for m in range(num_machines):
                cumsum_expr = None
                for j in sorted_jobs:
                    if not feasible[j, m]:
                        continue
                    if cumsum_expr is None:
                        cumsum_expr = processing_times[j][m] * y[j, m]
                    else:
                        cumsum_expr = cumsum_expr + processing_times[j][m] * y[j, m]
                    
                    # For job j, all jobs with d_i <= d_j that are on machine m must fit
                    # But we need: sum_{i: d_i <= d_j, i on m} p_i_m <= d_j + M*(1-y[j,m])
                    # i.e., if j is on m, the cumulative time <= d_j
                    
                    # cumsum_expr includes all jobs with d <= d_j on machine m (since sorted by due date)
                    # But wait, cumsum_expr accumulates over all j in sorted order, not just for machine m.
                    # Let me redo this.
                    pass
            
            # Redo properly
            for m in range(num_machines):
                # For each job j, define the "load" constraint:
                # if j is on machine m, then sum of p[i][m] * y[i,m] for all i with d[i] <= d[j] must be <= d[j]
                # This gives: sum_{i: d[i] <= d[j]} p[i][m] * y[i,m] <= d[j] + M*(1 - y[j,m])
                
                # Precompute groups by due date
                for j in range(num_jobs):
                    if not feasible[j, m]:
                        continue
                    # Jobs with due date <= d[j]
                    load = gp.quicksum(
                        processing_times[i][m] * y[i, m]
                        for i in range(num_jobs)
                        if feasible[i, m] and due_dates[i] <= due_dates[j]
                    )
                    model.addConstr(load <= due_dates[j] + M_val * (1 - y[j, m]))
        
        # Set initial solution
        for j in range(num_jobs):
            u[j].Start = 1 if j in heuristic_sol['tardy_jobs'] else 0
            for m in range(num_machines):
                y[j, m].Start = 0
        
        for m_str, job_list in heuristic_sol['schedule'].items():
            m = int(m_str)
            for j in job_list:
                y[j, m].Start = 1
        
        best_found = [heuristic_sol['objective_value']]
        
        def callback(model, where):
            if where == GRB.Callback.MIPSOL:
                obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if obj < best_found[0] - 0.5:
                    best_found[0] = obj
                    if logger:
                        logger.log(obj)
        
        model.optimize(callback)
        
        if model.SolCount > 0:
            # Extract solution
            tardy_jobs = []
            on_time_jobs = []
            machine_jobs = [[] for _ in range(num_machines)]
            
            for j in range(num_jobs):
                if u[j].X > 0.5:
                    tardy_jobs.append(j)
                else:
                    on_time_jobs.append(j)
                    for m in range(num_machines):
                        if y[j, m].X > 0.5:
                            machine_jobs[m].append(j)
                            break
            
            # Order jobs on each machine by EDD
            schedule = {}
            completion_times = {}
            
            for m in range(num_machines):
                jobs_sorted = sorted(machine_jobs[m], key=lambda j: due_dates[j])
                schedule[str(m)] = jobs_sorted
                ct = 0
                for j in jobs_sorted:
                    ct += processing_times[j][m]
                    completion_times[str(j)] = ct
            
            obj_val = sum(weights[j] for j in tardy_jobs)
            
            # Verify feasibility
            feasible_sol = True
            for j in on_time_jobs:
                if str(j) in completion_times and completion_times[str(j)] > due_dates[j]:
                    feasible_sol = False
                    break
            
            if feasible_sol and obj_val <= heuristic_sol['objective_value']:
                return {
                    "objective_value": obj_val,
                    "schedule": schedule,
                    "completion_times": completion_times,
                    "tardy_jobs": tardy_jobs,
                    "on_time_jobs": on_time_jobs
                }
        
        return heuristic_sol
        
    except Exception as e:
        return heuristic_sol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()
    
    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    with open(args.instance_path) as f:
        instance = json.load(f)
    
    solution = solve_instance(instance, args.time_limit, logger)
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == '__main__':
    main()
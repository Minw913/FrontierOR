import argparse
import json
import time
import random
from collections import defaultdict

def solve(instance_path, solution_path, time_limit, log_path):
    from solution_logger import SolutionLogger
    
    start_time = time.time()
    
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    num_jobs = instance['num_jobs']
    num_machines = instance['num_machines']
    jobs_data = instance['jobs']
    
    # Parse jobs
    jobs = []
    for job_info in jobs_data:
        ops = []
        for op_info in job_info['operations']:
            eligible = op_info['eligible_machines']
            ptimes = op_info['processing_times']
            # ptimes is dict with string keys
            machine_times = {}
            for m in eligible:
                machine_times[m] = ptimes[str(m)]
            ops.append({
                'eligible_machines': eligible,
                'machine_times': machine_times
            })
        jobs.append(ops)
    
    num_ops_per_job = [len(j) for j in jobs]
    
    # Compute a good upper bound for makespan
    # Sum of minimum processing times across all operations
    total_min = sum(min(op['machine_times'].values()) for job in jobs for op in job)
    # Max over jobs of sum of min processing times
    max_job_length = max(sum(min(op['machine_times'].values()) for op in job) for job in jobs)
    # Upper bound: sum of max processing times
    total_max = sum(max(op['machine_times'].values()) for job in jobs for op in job)
    
    # --- Greedy heuristic to get initial solution ---
    def greedy_solution():
        """List scheduling heuristic"""
        # For each operation, pick the machine with minimum processing time
        assignment = {}
        proc_times = {}
        for j, job in enumerate(jobs):
            for o, op in enumerate(job):
                best_m = min(op['machine_times'], key=lambda m: op['machine_times'][m])
                assignment[(j, o)] = best_m
                proc_times[(j, o)] = op['machine_times'][best_m]
        
        # Schedule using list scheduling
        # Priority: jobs with longest remaining processing time first
        job_next_op = [0] * num_jobs
        job_avail = [0] * num_jobs  # earliest time next op of job can start
        machine_avail = [0] * num_machines  # earliest time machine is free
        
        schedule = {}
        
        # Repeatedly schedule the next available operation
        while True:
            # Find schedulable operations
            candidates = []
            for j in range(num_jobs):
                if job_next_op[j] < num_ops_per_job[j]:
                    o = job_next_op[j]
                    m = assignment[(j, o)]
                    earliest = max(job_avail[j], machine_avail[m])
                    remaining = sum(proc_times[(j, oo)] for oo in range(o, num_ops_per_job[j]))
                    candidates.append((remaining, earliest, j, o, m))
            
            if not candidates:
                break
            
            # Sort by earliest start, then by longest remaining (descending)
            candidates.sort(key=lambda x: (x[1], -x[0]))
            
            # Schedule the first candidate
            _, earliest, j, o, m = candidates[0]
            p = proc_times[(j, o)]
            # Non-preemptive scheduling in greedy
            periods = list(range(earliest + 1, earliest + p + 1))  # 1-based
            schedule[(j, o)] = periods
            job_avail[j] = earliest + p
            machine_avail[m] = earliest + p
            job_next_op[j] += 1
        
        makespan = max(max(schedule[(j, num_ops_per_job[j]-1)]) for j in range(num_jobs))
        return assignment, schedule, makespan
    
    def greedy_solution_v2():
        """More sophisticated greedy with preemption awareness"""
        # Try all combinations? No, just pick min processing time machine
        assignment = {}
        proc_times = {}
        for j, job in enumerate(jobs):
            for o, op in enumerate(job):
                best_m = min(op['machine_times'], key=lambda m: op['machine_times'][m])
                assignment[(j, o)] = best_m
                proc_times[(j, o)] = op['machine_times'][best_m]
        
        # Use time-slot based scheduling with preemption
        T = total_max + 1
        machine_schedule = [[] for _ in range(num_machines)]  # list of (time, job, op) for each machine
        machine_busy = [[False] * (T + 1) for _ in range(num_machines)]
        
        job_next_op = [0] * num_jobs
        job_avail = [0] * num_jobs
        op_remaining = {}
        for j, job in enumerate(jobs):
            for o, op in enumerate(job):
                op_remaining[(j, o)] = proc_times[(j, o)]
        
        schedule = defaultdict(list)
        
        for t in range(T):
            # Find operations that can be scheduled at time t
            candidates = []
            for j in range(num_jobs):
                if job_next_op[j] < num_ops_per_job[j]:
                    o = job_next_op[j]
                    if job_avail[j] <= t and op_remaining[(j, o)] > 0:
                        m = assignment[(j, o)]
                        if not machine_busy[m][t]:
                            remaining_job = sum(proc_times[(j, oo)] for oo in range(o, num_ops_per_job[j]))
                            candidates.append((remaining_job, j, o, m))
            
            # Sort by longest remaining job time (descending) - LPT
            candidates.sort(key=lambda x: -x[0])
            
            machines_used = set()
            for _, j, o, m in candidates:
                if m not in machines_used:
                    machines_used.add(m)
                    machine_busy[m][t] = True
                    schedule[(j, o)].append(t + 1)  # 1-based
                    op_remaining[(j, o)] -= 1
                    if op_remaining[(j, o)] == 0:
                        job_avail[j] = t + 1
                        job_next_op[j] += 1
        
        # Check feasibility
        for j in range(num_jobs):
            if job_next_op[j] < num_ops_per_job[j]:
                return None, None, float('inf')
        
        makespan = max(max(schedule[(j, num_ops_per_job[j]-1)]) for j in range(num_jobs))
        return assignment, schedule, makespan
    
    # Try simple greedy first
    best_assignment, best_schedule, best_makespan = greedy_solution()
    if logger:
        logger.log(best_makespan)
    
    # Try v2 greedy
    a2, s2, m2 = greedy_solution_v2()
    if m2 < best_makespan:
        best_assignment, best_schedule, best_makespan = a2, s2, m2
        if logger:
            logger.log(best_makespan)
    
    # --- Try CP/scheduling approach using Gurobi MIP ---
    # For small instances, try exact MIP formulation
    # For larger instances, use heuristic approaches
    
    total_ops = sum(num_ops_per_job)
    
    def try_mip():
        nonlocal best_assignment, best_schedule, best_makespan
        
        import gurobipy as gp
        from gurobipy import GRB
        
        remaining_time = time_limit - (time.time() - start_time) - 2
        if remaining_time < 5:
            return
        
        # Time-indexed formulation with preemption
        # But time-indexed can be huge. Let's try a position-based or start-time formulation.
        
        # For moderate instances, use a compact formulation:
        # Variables:
        # x[j,o,m] = 1 if operation (j,o) is assigned to machine m
        # s[j,o] = start time of operation (j,o) (earliest time it's processed)
        # c[j,o] = completion time of operation (j,o) (latest time + 1)
        # C_max = makespan
        
        # With preemption, the completion time can be much later than start + processing time.
        # The key constraint is: on each machine, at most one operation at any time.
        # With preemption this is tricky without time-indexed variables.
        
        # Alternative: use a non-preemptive model as lower bound and heuristic guide
        # Then use a time-indexed model if small enough
        
        # Let's estimate if time-indexed is feasible
        T_ub = best_makespan  # use best known makespan as upper bound
        
        if total_ops * T_ub > 50000 or total_ops > 100:
            # Too large for time-indexed, use compact non-preemptive formulation
            try_compact_mip(remaining_time)
            return
        
        # Time-indexed formulation
        try_time_indexed_mip(T_ub, remaining_time)
    
    def try_compact_mip(remaining_time):
        """Non-preemptive compact MIP formulation"""
        nonlocal best_assignment, best_schedule, best_makespan
        
        import gurobipy as gp
        from gurobipy import GRB
        
        model = gp.Model("fjsp_compact")
        model.setParam('TimeLimit', remaining_time)
        model.setParam('Threads', 1)
        model.setParam('OutputFlag', 0)
        
        # Variables
        # x[j,o,m] = 1 if op (j,o) assigned to machine m
        x = {}
        for j, job in enumerate(jobs):
            for o, op in enumerate(job):
                for m in op['eligible_machines']:
                    x[j, o, m] = model.addVar(vtype=GRB.BINARY, name=f"x_{j}_{o}_{m}")
        
        # s[j,o] = start time
        # p_eff[j,o] = effective processing time
        s = {}
        p_eff = {}
        for j, job in enumerate(jobs):
            for o, op in enumerate(job):
                s[j, o] = model.addVar(lb=0, ub=total_max, vtype=GRB.CONTINUOUS, name=f"s_{j}_{o}")
                p_eff[j, o] = model.addVar(lb=0, ub=total_max, vtype=GRB.CONTINUOUS, name=f"p_{j}_{o}")
        
        C_max = model.addVar(lb=0, ub=total_max, vtype=GRB.CONTINUOUS, name="Cmax")
        
        model.setObjective(C_max, GRB.MINIMIZE)
        
        # Constraints
        # Each operation assigned to exactly one machine
        for j, job in enumerate(jobs):
            for o, op in enumerate(job):
                model.addConstr(
                    gp.quicksum(x[j, o, m] for m in op['eligible_machines']) == 1
                )
        
        # Processing time depends on machine
        for j, job in enumerate(jobs):
            for o, op in enumerate(job):
                model.addConstr(
                    p_eff[j, o] == gp.quicksum(
                        op['machine_times'][m] * x[j, o, m] for m in op['eligible_machines']
                    )
                )
        
        # Job precedence: s[j,o+1] >= s[j,o] + p_eff[j,o]
        for j, job in enumerate(jobs):
            for o in range(len(job) - 1):
                model.addConstr(s[j, o+1] >= s[j, o] + p_eff[j, o])
        
        # Makespan
        for j, job in enumerate(jobs):
            o = len(job) - 1
            model.addConstr(C_max >= s[j, o] + p_eff[j, o])
        
        # Machine disjunctive constraints (non-preemptive)
        # For operations that might share a machine
        M_big = total_max + 1
        
        # Collect all operations per machine
        ops_per_machine = defaultdict(list)
        for j, job in enumerate(jobs):
            for o, op in enumerate(job):
                for m in op['eligible_machines']:
                    ops_per_machine[m].append((j, o))
        
        y = {}
        for m in range(num_machines):
            ops = ops_per_machine[m]
            for i in range(len(ops)):
                for k in range(i+1, len(ops)):
                    j1, o1 = ops[i]
                    j2, o2 = ops[k]
                    y[j1, o1, j2, o2, m] = model.addVar(vtype=GRB.BINARY, name=f"y_{j1}_{o1}_{j2}_{o2}_{m}")
                    
                    # If both assigned to m, one must precede the other
                    # s[j1,o1] + p_eff[j1,o1] <= s[j2,o2] + M*(1-y) + M*(1-x[j1,o1,m]) + M*(1-x[j2,o2,m])
                    # s[j2,o2] + p_eff[j2,o2] <= s[j1,o1] + M*y + M*(1-x[j1,o1,m]) + M*(1-x[j2,o2,m])
                    model.addConstr(
                        s[j1, o1] + p_eff[j1, o1] <= s[j2, o2] + M_big * (1 - y[j1, o1, j2, o2, m]) + M_big * (2 - x[j1, o1, m] - x[j2, o2, m])
                    )
                    model.addConstr(
                        s[j2, o2] + p_eff[j2, o2] <= s[j1, o1] + M_big * y[j1, o1, j2, o2, m] + M_big * (2 - x[j1, o1, m] - x[j2, o2, m])
                    )
        
        # Set initial solution
        for j, job in enumerate(jobs):
            for o, op in enumerate(job):
                m_assigned = best_assignment[(j, o)]
                for m in op['eligible_machines']:
                    x[j, o, m].Start = 1.0 if m == m_assigned else 0.0
                periods = best_schedule[(j, o)]
                s[j, o].Start = min(periods) - 1  # 0-based start
                p_eff[j, o].Start = len(periods)
        C_max.Start = best_makespan
        
        model.addConstr(C_max <= best_makespan)  # Use as cutoff
        
        # Callback for incumbent solutions
        def callback(model, where):
            nonlocal best_assignment, best_schedule, best_makespan
            if where == GRB.Callback.MIPSOL:
                obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if obj < best_makespan:
                    # Extract solution
                    new_assignment = {}
                    new_schedule = {}
                    for j, job in enumerate(jobs):
                        for o, op in enumerate(job):
                            for m in op['eligible_machines']:
                                if model.cbGetSolution(x[j, o, m]) > 0.5:
                                    new_assignment[(j, o)] = m
                                    break
                            start = round(model.cbGetSolution(s[j, o]))
                            pt = round(model.cbGetSolution(p_eff[j, o]))
                            # Non-preemptive: consecutive periods
                            new_schedule[(j, o)] = list(range(start + 1, start + pt + 1))
                    
                    best_assignment = new_assignment
                    best_schedule = new_schedule
                    best_makespan = obj
                    if logger:
                        logger.log(obj)
        
        model.optimize(callback)
        
        if model.status in [GRB.OPTIMAL, GRB.SUBOPTIMAL] or model.SolCount > 0:
            obj = model.ObjVal
            if obj < best_makespan:
                new_assignment = {}
                new_schedule = {}
                for j, job in enumerate(jobs):
                    for o, op in enumerate(job):
                        for m in op['eligible_machines']:
                            if x[j, o, m].X > 0.5:
                                new_assignment[(j, o)] = m
                                break
                        start_val = round(s[j, o].X)
                        pt_val = round(p_eff[j, o].X)
                        new_schedule[(j, o)] = list(range(start_val + 1, start_val + pt_val + 1))
                
                best_assignment = new_assignment
                best_schedule = new_schedule
                best_makespan = obj
                if logger:
                    logger.log(obj)
    
    def try_time_indexed_mip(T_ub, remaining_time):
        """Time-indexed MIP formulation with preemption"""
        nonlocal best_assignment, best_schedule, best_makespan
        
        import gurobipy as gp
        from gurobipy import GRB
        
        T = int(T_ub)
        
        model = gp.Model("fjsp_time_indexed")
        model.setParam('TimeLimit', remaining_time)
        model.setParam('Threads', 1)
        model.setParam('OutputFlag', 0)
        
        # y[j,o,m,t] = 1 if operation (j,o) is processed on machine m at time period t (1-based)
        # x[j,o,m] = 1 if operation (j,o) is assigned to machine m
        
        x = {}
        y = {}
        
        for j, job in enumerate(jobs):
            for o, op in enumerate(job):
                for m in op['eligible_machines']:
                    x[j, o, m] = model.addVar(vtype=GRB.BINARY, name=f"x_{j}_{o}_{m}")
                    for t in range(1, T + 1):
                        y[j, o, m, t] = model.addVar(vtype=GRB.BINARY, name=f"y_{j}_{o}_{m}_{t}")
        
        C_max = model.addVar(lb=0, ub=T, vtype=GRB.INTEGER, name="Cmax")
        model.setObjective(C_max, GRB.MINIMIZE)
        
        # Each operation assigned to exactly one machine
        for j, job in enumerate(jobs):
            for o, op in enumerate(job):
                model.addConstr(
                    gp.quicksum(x[j, o, m] for m in op['eligible_machines']) == 1
                )
        
        # Processing time constraint: total periods on assigned machine = processing time
        for j, job in enumerate(jobs):
            for o, op in enumerate(job):
                for m in op['eligible_machines']:
                    model.addConstr(
                        gp.quicksum(y[j, o, m, t] for t in range(1, T + 1)) == op['machine_times'][m] * x[j, o, m]
                    )
        
        # Operation can only be processed on its assigned machine
        for j, job in enumerate(jobs):
            for o, op in enumerate(job):
                for m in op['eligible_machines']:
                    for t in range(1, T + 1):
                        model.addConstr(y[j, o, m, t] <= x[j, o, m])
        
        # Machine capacity: at most one operation per machine per time period
        ops_per_machine = defaultdict(list)
        for j, job in enumerate(jobs):
            for o, op in enumerate(job):
                for m in op['eligible_machines']:
                    ops_per_machine[m].append((j, o))
        
        for m in range(num_machines):
            ops = ops_per_machine[m]
            if len(ops) > 1:
                for t in range(1, T + 1):
                    model.addConstr(
                        gp.quicksum(y[j, o, m, t] for j, o in ops) <= 1
                    )
        
        # Job precedence: for each pair of consecutive operations in a job,
        # all processing of (j,o) at time t' must have t' < any time t where (j,o+1) is processed
        # With preemption: sum of y[j,o,m,t'] for t' >= t must be 0 if y[j,o+1,m',t] = 1
        # Equivalent: completion of (j,o) <= start of (j,o+1)
        # Let c[j,o] = last period where (j,o) is processed
        # Let s[j,o+1] = first period where (j,o+1) is processed
        # We need c[j,o] < s[j,o+1], i.e., c[j,o] + 1 <= s[j,o+1]
        
        # But tracking c and s with binary variables...
        # Alternative: cumulative constraint
        # For each time t, sum of y[j,o,m,t'] for t' >= t over all m >= remaining processing is guaranteed
        # Actually simpler: the cumulative processed amount of (j,o) by time t must equal processing time
        # before (j,o+1) can be processed at time t.
        
        # For job precedence with preemption:
        # For operation (j,o) and (j,o+1):
        # For each time t: sum_{m', t'>=t} y[j,o,m',t'] + sum_{m'', t''<=t} y[j,o+1,m'',t''] <= max(p[j,o], p[j,o+1])
        # This isn't quite right either.
        
        # Correct formulation:
        # cumulative_processed(j,o,t) = sum_{m, t'<=t} y[j,o,m,t']
        # For (j,o+1) to be processed at time t: cumulative_processed(j,o,t-1) >= processing_time(j,o)
        # Since processing_time depends on machine:
        # sum_{m, t'<t} y[j,o,m,t'] >= sum_m (p[j,o,m] * x[j,o,m]) * (any y[j,o+1,m',t] for any m')
        
        # This is complex. Let's use a simpler formulation:
        # sum_{m, t'<t} y[j,o,m,t'] >= sum_{m'} y[j,o+1,m',t] * sum_m(p[j,o,m] * x[j,o,m])
        # This is nonlinear. Linearize:
        
        # For each t, sum_{m'} y[j,o+1,m',t] <= sum_{m, t'<t} y[j,o,m,t'] / p_min
        # Not exact.
        
        # Better: define z[j,o+1,t] = sum_{m'} y[j,o+1,m',t] (at most 1 since assigned to one machine)
        # Then: for each t, sum_{m,t'<t} y[j,o,m,t'] >= z[j,o+1,t] * p[j,o,m] for the assigned machine
        # Since p depends on machine, we can write:
        # sum_{m,t'<=t-1} y[j,o,m,t'] >= sum_{m'} y[j,o+1,m',t] * (any positive amount implies all processing done)
        
        # Actually wait. The constraint is simpler than I'm making it:
        # All processing of (j,o) must complete before any processing of (j,o+1) starts.
        # With preemption, (j,o) might be processed at times {1,3,5} and (j,o+1) at {6,8}.
        # The constraint is: for all t where (j,o+1) is processed and all t' where (j,o) is processed: t' < t
        # Equivalently: max time of (j,o) < min time of (j,o+1)
        
        # Formulation:
        # For each j, o, for each time t:
        # sum_{m', t''>=t} y[j,o,m',t''] + sum_{m'', t'''<=t} y[j,o+1,m'',t'''] <= max_possible
        # Hmm, let me think differently.
        
        # For each j and consecutive operations o, o+1:
        # For each time t:
        #   If (j,o+1) is processed at time t, then (j,o) has completed all processing by time t-1
        #   sum_{m} y[j,o+1,m,t] <= (sum_{m',t'<=t-1} y[j,o,m',t']) / p_min_for_op_jo
        # This doesn't work well.
        
        # Standard approach: for each pair of consecutive ops and each time t:
        # sum_{m', t'>=t} y[j,o,m',t'] + sum_{m'', t''<=t} y[j,o+1,m'',t''] <= max(sum_{m'} y[j,o,m',*], 1)
        
        # Simplest correct formulation:
        # For each time t: (j,o) remaining processing from t onward + (j,o+1) cumulative up to t <= total_p[j,o]
        # But total_p depends on machine assignment.
        
        # Let's use: for all t, for consecutive o, o+1:
        # sum_{m,t'>=t} y[j,o,m,t'] + sum_{m,t''<=t} y[j,o+1,m,t''] <= sum_m p[j,o,m]*x[j,o,m]
        # This says: remaining work on (j,o) from time t + work done on (j,o+1) up to time t <= total work on (j,o)
        # But the total work on (j,o) = sum_m p[j,o,m]*x[j,o,m] and also = sum_{m,t'} y[j,o,m,t']
        # So: sum_{m,t'>=t} y[j,o,m,t'] + sum_{m,t''<=t} y[j,o+1,m,t''] <= sum_{m,t'} y[j,o,m,t']
        # Which simplifies to: sum_{m,t''<=t} y[j,o+1,m,t''] <= sum_{m,t'<=t-1} y[j,o,m,t']
        # i.e., cumulative processing of (j,o+1) up to time t <= cumulative processing of (j,o) up to time t-1
        
        # That's correct and elegant! Let me verify:
        # If (j,o+1) processes at time t, then cumulative(j,o+1,t) >= 1, so cumulative(j,o,t-1) >= 1
        # If (j,o+1) finishes at time T, cumulative(j,o+1,T) = p[j,o+1], so cumulative(j,o,T-1) >= p[j,o+1]
        # Wait, that's not right. We need cumulative(j,o) to be at max (p[j,o]) before (j,o+1) starts.
        
        # Actually the constraint sum_{m,t''<=t} y[j,o+1,m,t''] <= sum_{m,t'<=t-1} y[j,o,m,t']
        # means: at any time t, the amount of (j,o+1) done so far <= amount of (j,o) done before t.
        # But (j,o) total is p[j,o] and (j,o+1) total is p[j,o+1]. These can be different.
        # If p[j,o] = 2 and p[j,o+1] = 5, then cumulative(j,o+1) can be at most 2 at any point before (j,o) finishes.
        # That's wrong - it would prevent (j,o+1) from processing 5 units since max of LHS is 5 but RHS maxes at 2.
        
        # Wait, as t -> T, LHS approaches p[j,o+1] = 5, RHS approaches p[j,o] = 2. Constraint violated.
        # So this formulation is WRONG for the general case.
        
        # Correct formulation for strict precedence (no overlap):
        # For each t: sum_{m} y[j,o+1,m,t] <= floor(sum_{m,t'<t} y[j,o,m,t'] / p[j,o,assigned_machine])
        # Or more directly:
        # For each t: if any y[j,o+1,m,t] = 1, then sum_{m,t'<t} y[j,o,m,t'] = p[j,o]
        # Which means: y[j,o+1,m,t] <= 1 - (p[j,o] - sum_{m',t'<t} y[j,o,m',t']) / p[j,o]
        # = sum_{m',t'<t} y[j,o,m',t'] / p[j,o]
        
        # This is still complex with variable p[j,o]. Let me just use:
        # For each t and each machine m'' eligible for (j,o+1):
        # y[j,o+1,m'',t] * sum_m(p[j,o,m]*x[j,o,m]) <= sum_{m',t'<t} y[j,o,m',t']
        # This is bilinear. Need to linearize.
        
        # Alternative simpler approach: since processing time depends on machine, introduce
        # completion time c[j,o] and start time s[j,o+1]:
        # c[j,o] is the last time period where (j,o) is processed
        # s[j,o+1] is the first time period where (j,o+1) is processed  
        # Constraint: c[j,o] < s[j,o+1]
        
        # c[j,o] = max{t : exists m with y[j,o,m,t]=1}
        # s[j,o+1] = min{t : exists m with y[j,o+1,m,t]=1}
        
        # Model c[j,o] >= t * y[j,o,m,t] for all m,t
        # Model s[j,o+1] <= t + T*(1 - y[j,o+1,m,t]) for all m,t
        # Constraint: c[j,o] < s[j,o+1], i.e., c[j,o] + 1 <= s[j,o+1]
        
        c_var = {}
        s_var = {}
        for j, job in enumerate(jobs):
            for o, op in enumerate(job):
                c_var[j, o] = model.addVar(lb=1, ub=T, vtype=GRB.INTEGER, name=f"c_{j}_{o}")
                s_var[j, o] = model.addVar(lb=1, ub=T, vtype=GRB.INTEGER, name=f"s_{j}_{o}")
        
        for j, job in enumerate(jobs):
            for o, op in enumerate(job):
                for m in op['eligible_machines']:
                    for t in range(1, T + 1):
                        model.addConstr(c_var[j, o] >= t * y[j, o, m, t])
                        model.addConstr(s_var[j, o] <= t + T * (1 - y[j, o, m, t]))
        
        # Precedence
        for j, job in enumerate(jobs):
            for o in range(len(job) - 1):
                model.addConstr(c_var[j, o] + 1 <= s_var[j, o + 1])
        
        # Makespan
        for j, job in enumerate(jobs):
            o = len(job) - 1
            model.addConstr(C_max >= c_var[j, o] + 1)  # completion is end of last period
        
        # Set cutoff
        model.addConstr(C_max <= best_makespan)
        
        # Set initial solution
        for j, job in enumerate(jobs):
            for o, op in enumerate(job):
                m_assigned = best_assignment[(j, o)]
                for m in op['eligible_machines']:
                    x[j, o, m].Start = 1.0 if m == m_assigned else 0.0
                    for t in range(1, T + 1):
                        y[j, o, m, t].Start = 1.0 if (m == m_assigned and t in best_schedule[(j, o)]) else 0.0
                periods = best_schedule[(j, o)]
                c_var[j, o].Start = max(periods)
                s_var[j, o].Start = min(periods)
        C_max.Start = best_makespan
        
        def callback(model, where):
            nonlocal best_assignment, best_schedule, best_makespan
            if where == GRB.Callback.MIPSOL:
                obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if obj < best_makespan:
                    new_assignment = {}
                    new_schedule = {}
                    for j, job in enumerate(jobs):
                        for o, op in enumerate(job):
                            for m in op['eligible_machines']:
                                if model.cbGetSolution(x[j, o, m]) > 0.5:
                                    new_assignment[(j, o)] = m
                                    break
                            periods = []
                            assigned_m = new_assignment[(j, o)]
                            for t in range(1, T + 1):
                                if model.cbGetSolution(y[j, o, assigned_m, t]) > 0.5:
                                    periods.append(t)
                            new_schedule[(j, o)] = periods
                    
                    best_assignment = new_assignment
                    best_schedule = new_schedule
                    best_makespan = obj
                    if logger:
                        logger.log(obj)
        
        model.optimize(callback)
        
        if model.SolCount > 0:
            obj = model.ObjVal
            if obj < best_makespan:
                new_assignment = {}
                new_schedule = {}
                for j, job in enumerate(jobs):
                    for o, op in enumerate(job):
                        for m in op['eligible_machines']:
                            if x[j, o, m].X > 0.5:
                                new_assignment[(j, o)] = m
                                break
                        periods = []
                        assigned_m = new_assignment[(j, o)]
                        for t in range(1, T + 1):
                            if y[j, o, assigned_m, t].X > 0.5:
                                periods.append(t)
                        new_schedule[(j, o)] = periods
                
                best_assignment = new_assignment
                best_schedule = new_schedule
                best_makespan = obj
                if logger:
                    logger.log(obj)
    
    # Decide which approach to use based on problem size
    # First, let's try improving with local search / genetic algorithm approach
    
    def schedule_with_assignment(assignment_dict):
        """Given machine assignments, schedule operations using list scheduling with preemption.
        Returns schedule dict and makespan, or None if infeasible."""
        proc_times = {}
        for j, job in enumerate(jobs):
            for o, op in enumerate(job):
                m = assignment_dict[(j, o)]
                proc_times[(j, o)] = op['machine_times'][m]
        
        # Compute critical path length for priority
        # cp[j,o] = proc_times[j,o] + cp[j,o+1] if exists
        cp = {}
        for j, job in enumerate(jobs):
            for o in range(len(job) - 1, -1, -1):
                if o == len(job) - 1:
                    cp[(j, o)] = proc_times[(j, o)]
                else:
                    cp[(j, o)] = proc_times[(j, o)] + cp[(j, o + 1)]
        
        T = sum(max(op['machine_times'].values()) for job in jobs for op in job) + 1
        
        # Time-slot greedy scheduling
        machine_busy = [set() for _ in range(num_machines)]
        schedule = {}
        job_next_op = [0] * num_jobs
        job_avail = [0] * num_jobs  # 0-based: earliest time period (1-based) op can start
        op_remaining = {}
        for key in proc_times:
            op_remaining[key] = proc_times[key]
        
        all_done = False
        t = 1  # 1-based time
        while not all_done and t <= T:
            # Find operations that can be scheduled at time t
            candidates = []
            for j in range(num_jobs):
                if job_next_op[j] < num_ops_per_job[j]:
                    o = job_next_op[j]
                    if job_avail[j] < t:  # available before or at time t
                        m = assignment_dict[(j, o)]
                        if t not in machine_busy[m]:
                            candidates.append((j, o, m, cp[(j, o)]))
            
            # Sort by critical path (longest first)
            candidates.sort(key=lambda x: -x[3])
            
            machines_used_this_t = set()
            for j, o, m, _ in candidates:
                if m not in machines_used_this_t:
                    machines_used_this_t.add(m)
                    machine_busy[m].add(t)
                    if (j, o) not in schedule:
                        schedule[(j, o)] = []
                    schedule[(j, o)].append(t)
                    op_remaining[(j, o)] -= 1
                    if op_remaining[(j, o)] == 0:
                        job_avail[j] = t  # next op can start at t+1 (1-based), so avail = t
                        job_next_op[j] += 1
            
            # Check if all done
            all_done = all(job_next_op[j] >= num_ops_per_job[j] for j in range(num_jobs))
            t += 1
        
        if not all_done:
            return None, float('inf')
        
        makespan = max(max(schedule[(j, num_ops_per_job[j]-1)]) for j in range(num_jobs))
        return schedule, makespan
    
    def random_assignment():
        """Generate a random machine assignment"""
        assignment = {}
        for j, job in enumerate(jobs):
            for o, op in enumerate(job):
                m = random.choice(op['eligible_machines'])
                assignment[(j, o)] = m
        return assignment
    
    def min_pt_assignment():
        """Assign each op to its fastest machine"""
        assignment = {}
        for j, job in enumerate(jobs):
            for o, op in enumerate(job):
                best_m = min(op['machine_times'], key=lambda m: op['machine_times'][m])
                assignment[(j, o)] = best_m
        return assignment
    
    def balanced_assignment():
        """Try to balance load across machines"""
        machine_load = [0] * num_machines
        assignment = {}
        # Process operations in order of decreasing min processing time
        all_ops = []
        for j, job in enumerate(jobs):
            for o, op in enumerate(job):
                min_pt = min(op['machine_times'].values())
                all_ops.append((min_pt, j, o))
        all_ops.sort(reverse=True)
        
        for _, j, o in all_ops:
            op = jobs[j][o]
            # Pick machine that minimizes max load
            best_m = None
            best_load = float('inf')
            for m in op['eligible_machines']:
                new_load = machine_load[m] + op['machine_times'][m]
                if new_load < best_load:
                    best_load = new_load
                    best_m = m
            assignment[(j, o)] = best_m
            machine_load[best_m] += op['machine_times'][best_m]
        return assignment
    
    # Try different assignment strategies
    for assign_func in [min_pt_assignment, balanced_assignment]:
        if time.time() - start_time > time_limit - 2:
            break
        a = assign_func()
        s, mk = schedule_with_assignment(a)
        if mk < best_makespan:
            best_assignment = a
            best_schedule = s
            best_makespan = mk
            if logger:
                logger.log(mk)
    
    # Local search: try changing machine assignments
    def local_search():
        nonlocal best_assignment, best_schedule, best_makespan
        
        current_assignment = dict(best_assignment)
        current_schedule = dict(best_schedule)
        current_makespan = best_makespan
        
        improved = True
        iterations = 0
        while improved and time.time() - start_time < time_limit - 5:
            improved = False
            iterations += 1
            
            # Try changing one operation's machine assignment
            ops_list = [(j, o) for j in range(num_jobs) for o in range(num_ops_per_job[j])]
            random.shuffle(ops_list)
            
            for j, o in ops_list:
                if time.time() - start_time > time_limit - 5:
                    break
                op = jobs[j][o]
                current_m = current_assignment[(j, o)]
                
                for m in op['eligible_machines']:
                    if m == current_m:
                        continue
                    
                    # Try this new assignment
                    trial = dict(current_assignment)
                    trial[(j, o)] = m
                    s, mk = schedule_with_assignment(trial)
                    
                    if mk < current_makespan:
                        current_assignment = trial
                        current_schedule = s
                        current_makespan = mk
                        improved = True
                        
                        if mk < best_makespan:
                            best_assignment = dict(current_assignment)
                            best_schedule = dict(current_schedule)
                            best_makespan = mk
                            if logger:
                                logger.log(mk)
                        break
    
    # Run local search
    if time.time() - start_time < time_limit - 10:
        local_search()
    
    # Random restarts with local search
    def random_restart_search():
        nonlocal best_assignment, best_schedule, best_makespan
        
        while time.time() - start_time < time_limit - 5:
            # Generate random assignment
            a = random_assignment()
            s, mk = schedule_with_assignment(a)
            
            if mk == float('inf'):
                continue
            
            # Local search from this point
            current_assignment = a
            current_schedule = s
            current_makespan = mk
            
            improved = True
            while improved and time.time() - start_time < time_limit - 5:
                improved = False
                ops_list = [(j, o) for j in range(num_jobs) for o in range(num_ops_per_job[j])]
                random.shuffle(ops_list)
                
                for j, o in ops_list:
                    if time.time() - start_time > time_limit - 5:
                        break
                    op = jobs[j][o]
                    current_m = current_assignment[(j, o)]
                    
                    for m in op['eligible_machines']:
                        if m == current_m:
                            continue
                        trial = dict(current_assignment)
                        trial[(j, o)] = m
                        s2, mk2 = schedule_with_assignment(trial)
                        
                        if mk2 < current_makespan:
                            current_assignment = trial
                            current_schedule = s2
                            current_makespan = mk2
                            improved = True
                            
                            if mk2 < best_makespan:
                                best_assignment = dict(current_assignment)
                                best_schedule = dict(current_schedule)
                                best_makespan = mk2
                                if logger:
                                    logger.log(mk2)
                            break
    
    # Try MIP if instance is small enough
    if total_ops <= 50 and num_machines <= 10:
        if time.time() - start_time < time_limit - 10:
            try:
                try_mip()
            except Exception:
                pass
    
    # More random restarts if time allows
    if time.time() - start_time < time_limit - 5:
        random_restart_search()
    
    # Convert solution to output format
    assignment_out = {}
    schedule_out = {}
    
    for j in range(num_jobs):
        for o in range(num_ops_per_job[j]):
            key = f"({j},{o})"
            assignment_out[key] = best_assignment[(j, o)]
            schedule_out[key] = sorted(best_schedule[(j, o)])
    
    solution = {
        "objective_value": float(best_makespan),
        "assignment": assignment_out,
        "schedule": schedule_out
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
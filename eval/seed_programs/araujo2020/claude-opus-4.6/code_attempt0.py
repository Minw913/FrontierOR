import argparse
import json
import time
import sys
from collections import defaultdict, deque

def solve(instance_path, solution_path, time_limit, log_path):
    from solution_logger import SolutionLogger
    
    start_time = time.time()
    
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    projects = instance['projects']
    resources = instance['resources']
    jobs_data = instance['jobs']
    
    renewable_resources = resources.get('renewable', [])
    nonrenewable_resources = resources.get('nonrenewable', [])
    
    num_renewable = len(renewable_resources)
    num_nonrenewable = len(nonrenewable_resources)
    
    renewable_caps = [r['capacity'] for r in renewable_resources]
    nonrenewable_caps = [r['capacity'] for r in nonrenewable_resources]
    
    # Build data structures
    job_by_id = {}
    for j in jobs_data:
        job_by_id[j['job_id']] = j
    
    project_by_id = {}
    for p in projects:
        project_by_id[p['project_id']] = p
    
    # Build project -> jobs mapping
    project_jobs = defaultdict(list)
    for j in jobs_data:
        project_jobs[j['project_id']].append(j['job_id'])
    
    # Build precedence graph
    successors = defaultdict(list)
    predecessors = defaultdict(list)
    for j in jobs_data:
        for s in j['successors']:
            successors[j['job_id']].append(s)
            predecessors[s].append(j['job_id'])
    
    all_job_ids = [j['job_id'] for j in jobs_data]
    
    # Compute critical path duration for each project using shortest duration modes
    # CPM forward pass: earliest start times
    # For CPM, use minimum duration mode for each job
    min_duration = {}
    for j in jobs_data:
        min_dur = min(m['duration'] for m in j['modes'])
        min_duration[j['job_id']] = min_dur
    
    # Topological sort of all jobs
    in_degree = defaultdict(int)
    for jid in all_job_ids:
        in_degree[jid] = len(predecessors[jid])
    
    topo_order = []
    queue = deque()
    for jid in all_job_ids:
        if in_degree[jid] == 0:
            queue.append(jid)
    while queue:
        jid = queue.popleft()
        topo_order.append(jid)
        for s in successors[jid]:
            in_degree[s] -= 1
            if in_degree[s] == 0:
                queue.append(s)
    
    # Earliest start times (CPM, ignoring resources, using min durations, respecting release dates)
    est = {}
    for jid in topo_order:
        j = job_by_id[jid]
        proj = project_by_id[j['project_id']]
        release = proj['release_date']
        es = release
        for p in predecessors[jid]:
            es = max(es, est[p] + min_duration[p])
        est[jid] = es
    
    # Critical path duration for each project
    cp_duration = {}
    for p in projects:
        sink = p['artificial_sink_job_id']
        source = p['artificial_source_job_id']
        cp_duration[p['project_id']] = est[sink] - p['release_date']
    
    # Compute longest path from each job to project sink using min durations (backward)
    # This is the "longest path to sink" used for latest start computation
    reverse_topo = list(reversed(topo_order))
    longest_to_sink = {}
    
    # For each project, identify sink
    job_project = {}
    for j in jobs_data:
        job_project[j['job_id']] = j['project_id']
    
    project_sink = {}
    for p in projects:
        project_sink[p['project_id']] = p['artificial_sink_job_id']
    
    for jid in reverse_topo:
        pid = job_project[jid]
        if jid == project_sink[pid]:
            longest_to_sink[jid] = 0
        else:
            max_path = 0
            for s in successors[jid]:
                if s in longest_to_sink:
                    max_path = max(max_path, min_duration[s] + longest_to_sink[s])
            # Include self duration? No - longest path FROM job to sink includes the job's own duration?
            # Actually looking at the problem description: "minus the total duration of the longest path from that job to the project's artificial end job"
            # This path includes the job itself? Let me re-read...
            # "the total duration of the longest path from that job to the project's artificial end job computed using the shortest-duration modes for all jobs along that path"
            # A path from job to sink: job -> ... -> sink. The total duration = sum of durations of jobs on the path.
            # So longest_to_sink[jid] should include min_duration[jid] itself.
            longest_to_sink[jid] = min_duration[jid] + max_path
    
    # For sink jobs, the path from sink to sink has duration = min_duration[sink] = 0
    # So longest_to_sink[sink] = 0 is correct.
    
    # Get upper bound from a simple heuristic first (serial SGS)
    # We need an initial feasible solution to compute the upper bound and latest start times
    
    def compute_serial_sgs(priority_list, mode_choices):
        """Serial Schedule Generation Scheme"""
        # priority_list: list of job_ids in scheduling order
        # mode_choices: dict job_id -> mode_id
        
        scheduled_start = {}
        scheduled_mode = {}
        
        # Resource usage tracking
        # For renewable: usage per period
        # We'll use a dict for sparse representation
        renewable_usage = defaultdict(lambda: [0] * num_renewable)
        nonrenew_used = [0] * num_nonrenewable
        
        for jid in priority_list:
            j = job_by_id[jid]
            mid = mode_choices[jid]
            mode = None
            for m in j['modes']:
                if m['mode_id'] == mid:
                    mode = m
                    break
            
            dur = mode['duration']
            renew_cons = mode['renewable_consumption']
            nonrenew_cons = mode['nonrenewable_consumption']
            
            proj = project_by_id[j['project_id']]
            release = proj['release_date']
            
            # Earliest start from precedence
            es = release
            for p in predecessors[jid]:
                if p in scheduled_start:
                    p_mode = None
                    for m in job_by_id[p]['modes']:
                        if m['mode_id'] == scheduled_mode[p]:
                            p_mode = m
                            break
                    es = max(es, scheduled_start[p] + p_mode['duration'])
            
            # Check nonrenewable feasibility
            nr_ok = True
            for r in range(num_nonrenewable):
                if nonrenew_used[r] + nonrenew_cons[r] > nonrenewable_caps[r]:
                    nr_ok = False
                    break
            
            if not nr_ok:
                return None  # Infeasible
            
            # Find earliest feasible start respecting renewable resources
            t = es
            while True:
                feasible = True
                if dur > 0:
                    for period in range(t, t + dur):
                        for r in range(num_renewable):
                            if renewable_usage[period][r] + renew_cons[r] > renewable_caps[r]:
                                feasible = False
                                break
                        if not feasible:
                            break
                if feasible:
                    break
                t += 1
                if t > 10000:  # Safety limit
                    return None
            
            scheduled_start[jid] = t
            scheduled_mode[jid] = mid
            
            # Update resource usage
            if dur > 0:
                for period in range(t, t + dur):
                    for r in range(num_renewable):
                        renewable_usage[period][r] += renew_cons[r]
            for r in range(num_nonrenewable):
                nonrenew_used[r] += nonrenew_cons[r]
        
        return scheduled_start, scheduled_mode
    
    def compute_objective(scheduled_start, scheduled_mode):
        total_delay = 0
        makespan = 0
        for p in projects:
            sink = p['artificial_sink_job_id']
            completion = scheduled_start[sink]
            delay = completion - (p['release_date'] + cp_duration[p['project_id']])
            total_delay += delay
            makespan = max(makespan, completion)
        return total_delay, makespan
    
    def combined_objective(total_delay, makespan):
        return total_delay + 0.001 * makespan
    
    # Default mode choices: pick mode with minimum duration (tie-break by min resource usage)
    default_mode = {}
    for j in jobs_data:
        best_mode = None
        best_dur = float('inf')
        for m in j['modes']:
            if m['duration'] < best_dur:
                best_dur = m['duration']
                best_mode = m['mode_id']
        default_mode[j['job_id']] = best_mode
    
    # Priority: earliest start time (topological order as tie-break)
    topo_index = {jid: i for i, jid in enumerate(topo_order)}
    priority_list = sorted(all_job_ids, key=lambda jid: (est[jid], topo_index[jid]))
    
    result = compute_serial_sgs(priority_list, default_mode)
    
    if result is None:
        # Try different modes
        # Pick modes that minimize nonrenewable consumption
        for j in jobs_data:
            best_mode = None
            best_score = float('inf')
            for m in j['modes']:
                score = sum(m['nonrenewable_consumption']) + m['duration']
                if score < best_score:
                    best_score = score
                    best_mode = m['mode_id']
            default_mode[j['job_id']] = best_mode
        result = compute_serial_sgs(priority_list, default_mode)
    
    best_solution = None
    best_obj = float('inf')
    best_delay = float('inf')
    best_makespan = float('inf')
    
    if result is not None:
        sched_start, sched_mode = result
        delay, makespan = compute_objective(sched_start, sched_mode)
        obj = combined_objective(delay, makespan)
        best_solution = (sched_start, sched_mode)
        best_obj = obj
        best_delay = delay
        best_makespan = makespan
        if logger:
            logger.log(obj)
    
    # Try multiple priority rules for initial solutions
    def try_priority_and_modes(plist, modes):
        nonlocal best_solution, best_obj, best_delay, best_makespan
        res = compute_serial_sgs(plist, modes)
        if res is not None:
            ss, sm = res
            d, mk = compute_objective(ss, sm)
            o = combined_objective(d, mk)
            if o < best_obj:
                best_solution = (ss, sm)
                best_obj = o
                best_delay = d
                best_makespan = mk
                if logger:
                    logger.log(o)
                return True
        return False
    
    # Latest finish time priority (Most Total Successors, etc.)
    # Compute LFT for each job using CPM backward pass
    # But first let's try different priority rules
    
    # Count total successors (transitive)
    total_succ_count = {}
    for jid in reverse_topo:
        count = 0
        for s in successors[jid]:
            count += 1 + total_succ_count.get(s, 0)
        total_succ_count[jid] = count
    
    # Priority by most total successors (descending)
    plist_mts = sorted(all_job_ids, key=lambda jid: (-total_succ_count[jid], est[jid], topo_index[jid]))
    # Make sure topological order is maintained
    # Actually serial SGS requires topological feasibility - predecessors must be scheduled before successors
    # So we need to ensure the priority list respects precedence
    
    def make_feasible_list(priority_key_func):
        """Generate a feasible scheduling order that respects precedence, using priority_key_func for tie-breaking"""
        scheduled_set = set()
        ready = set()
        in_deg = {}
        for jid in all_job_ids:
            in_deg[jid] = len(predecessors[jid])
            if in_deg[jid] == 0:
                ready.add(jid)
        
        order = []
        while ready:
            # Pick from ready set based on priority
            best = min(ready, key=priority_key_func)
            ready.remove(best)
            order.append(best)
            scheduled_set.add(best)
            for s in successors[best]:
                in_deg[s] -= 1
                if in_deg[s] == 0:
                    ready.add(s)
        return order
    
    # Try several priority rules
    priority_rules = [
        lambda jid: (est[jid], topo_index[jid]),  # Earliest start
        lambda jid: (-total_succ_count[jid], est[jid]),  # Most total successors
        lambda jid: (est[jid] + min_duration[jid], topo_index[jid]),  # Earliest finish
        lambda jid: (-longest_to_sink[jid], est[jid]),  # Longest path to sink (critical first)
        lambda jid: (min_duration[jid], est[jid]),  # Shortest processing time
    ]
    
    # Also try different mode selection strategies
    def get_modes_min_duration():
        modes = {}
        for j in jobs_data:
            best = min(j['modes'], key=lambda m: m['duration'])
            modes[j['job_id']] = best['mode_id']
        return modes
    
    def get_modes_min_nonrenew():
        modes = {}
        for j in jobs_data:
            best = min(j['modes'], key=lambda m: (sum(m['nonrenewable_consumption']), m['duration']))
            modes[j['job_id']] = best['mode_id']
        return modes
    
    def get_modes_min_renewable():
        modes = {}
        for j in jobs_data:
            best = min(j['modes'], key=lambda m: (sum(m['renewable_consumption']), m['duration']))
            modes[j['job_id']] = best['mode_id']
        return modes
    
    def get_modes_balanced():
        modes = {}
        for j in jobs_data:
            best = min(j['modes'], key=lambda m: (m['duration'] + sum(m['renewable_consumption']) * 0.1 + sum(m['nonrenewable_consumption']) * 0.1))
            modes[j['job_id']] = best['mode_id']
        return modes
    
    mode_strategies = [get_modes_min_duration, get_modes_min_nonrenew, get_modes_min_renewable, get_modes_balanced]
    
    for pr in priority_rules:
        if time.time() - start_time > time_limit * 0.1:
            break
        plist = make_feasible_list(pr)
        for ms in mode_strategies:
            if time.time() - start_time > time_limit * 0.1:
                break
            modes = ms()
            try_priority_and_modes(plist, modes)
    
    if best_solution is None:
        # Fallback: try all mode combinations for small instances
        # Just output something
        output = {
            "objective_value": 0.0,
            "schedule": [],
            "makespan": 0
        }
        with open(solution_path, 'w') as f:
            json.dump(output, f, indent=2)
        return
    
    # Now try to improve with local search
    # We'll use a simple approach: for each job, try different modes and see if we can improve
    
    def rebuild_schedule_with_modes(mode_choices, priority_key_func=None):
        if priority_key_func is None:
            priority_key_func = lambda jid: (est[jid], topo_index[jid])
        plist = make_feasible_list(priority_key_func)
        return compute_serial_sgs(plist, mode_choices)
    
    # Local search: try swapping modes for individual jobs
    def local_search_modes(current_modes, max_iter=1000):
        nonlocal best_solution, best_obj, best_delay, best_makespan
        
        current = dict(current_modes)
        cur_result = rebuild_schedule_with_modes(current)
        if cur_result is None:
            return current
        
        cur_start, cur_mode = cur_result
        cur_delay, cur_mk = compute_objective(cur_start, cur_mode)
        cur_obj = combined_objective(cur_delay, cur_mk)
        
        improved = True
        iteration = 0
        while improved and iteration < max_iter:
            if time.time() - start_time > time_limit * 0.7:
                break
            improved = False
            iteration += 1
            
            for j in jobs_data:
                if time.time() - start_time > time_limit * 0.7:
                    break
                if len(j['modes']) <= 1:
                    continue
                
                jid = j['job_id']
                original_mode = current[jid]
                
                for m in j['modes']:
                    if m['mode_id'] == original_mode:
                        continue
                    
                    current[jid] = m['mode_id']
                    res = rebuild_schedule_with_modes(current)
                    if res is not None:
                        ss, sm = res
                        d, mk = compute_objective(ss, sm)
                        o = combined_objective(d, mk)
                        if o < cur_obj:
                            cur_obj = o
                            cur_delay = d
                            cur_mk = mk
                            improved = True
                            if o < best_obj:
                                best_solution = (ss, sm)
                                best_obj = o
                                best_delay = d
                                best_makespan = mk
                                if logger:
                                    logger.log(o)
                            break  # Accept first improvement
                    
                if not improved or current[jid] == original_mode:
                    current[jid] = original_mode
        
        return current
    
    # Run local search on best modes found so far
    best_modes = dict(best_solution[1]) if best_solution else get_modes_min_duration()
    
    if time.time() - start_time < time_limit * 0.3:
        local_search_modes(best_modes)
    
    # Now try a more sophisticated approach: forward-backward improvement
    # Try different priority orderings based on the current best solution
    
    def forward_backward_improvement(max_iterations=50):
        nonlocal best_solution, best_obj, best_delay, best_makespan
        
        if best_solution is None:
            return
        
        cur_start, cur_mode = best_solution
        
        for iteration in range(max_iterations):
            if time.time() - start_time > time_limit * 0.85:
                break
            
            # Create priority based on current start times (with some perturbation)
            import random
            random.seed(iteration * 42)
            
            if iteration == 0:
                # Try latest start time priority (backward)
                pk = lambda jid: (-cur_start.get(jid, 0), -topo_index[jid])
            elif iteration % 3 == 0:
                # Random perturbation of current start times
                perturbation = {jid: cur_start.get(jid, 0) + random.randint(-5, 5) for jid in all_job_ids}
                pk = lambda jid: (perturbation[jid], topo_index[jid])
            elif iteration % 3 == 1:
                # Priority by slack (difference between actual and earliest start)
                pk = lambda jid: (cur_start.get(jid, 0) - est[jid], topo_index[jid])
            else:
                # Random priority
                random_prio = {jid: random.random() for jid in all_job_ids}
                pk = lambda jid: (random_prio[jid],)
            
            plist = make_feasible_list(pk)
            modes = dict(cur_mode)
            
            res = compute_serial_sgs(plist, modes)
            if res is not None:
                ss, sm = res
                d, mk = compute_objective(ss, sm)
                o = combined_objective(d, mk)
                if o < best_obj:
                    best_solution = (ss, sm)
                    best_obj = o
                    best_delay = d
                    best_makespan = mk
                    cur_start, cur_mode = ss, sm
                    if logger:
                        logger.log(o)
    
    forward_backward_improvement()
    
    # Try Gurobi MIP if instance is small enough
    num_jobs = len(jobs_data)
    
    # Compute horizon and time windows for MIP
    if best_solution is not None:
        sched_start, sched_mode = best_solution
    
    # Determine if MIP is feasible given instance size
    # For MIP, we need to compute proper time windows
    
    # Upper bound on each project's completion time from best solution
    project_ub = {}
    if best_solution is not None:
        for p in projects:
            sink = p['artificial_sink_job_id']
            project_ub[p['project_id']] = sched_start[sink]
    
    # Max allowable total project delay
    max_total_delay = 0
    for p in projects:
        pid = p['project_id']
        if pid in project_ub:
            max_total_delay += project_ub[pid] - (p['release_date'] + cp_duration[pid])
    
    # Horizon
    horizon = 0
    for p in projects:
        pid = p['project_id']
        horizon = max(horizon, p['release_date'] + cp_duration[pid] + max_total_delay)
    
    # Latest start time for each job in each mode
    def compute_lst(jid, mode_dur):
        pid = job_project[jid]
        p = project_by_id[pid]
        # lst = release + cp_duration - longest_to_sink[jid] + max_total_delay
        # But longest_to_sink includes the job's own duration
        # lst(j,m) = release + cp_dur - longest_path_from_j_to_sink + max_total_delay
        lst = p['release_date'] + cp_duration[pid] - longest_to_sink[jid] + max_total_delay
        return max(est[jid], min(lst, horizon - mode_dur))
    
    # Estimate total number of binary variables for MIP
    total_binary_vars = 0
    for j in jobs_data:
        for m in j['modes']:
            lst = compute_lst(j['job_id'], m['duration'])
            window_size = max(0, lst - est[j['job_id']] + 1)
            total_binary_vars += window_size
    
    use_mip = total_binary_vars < 50000 and num_jobs < 200 and time.time() - start_time < time_limit * 0.4
    
    if use_mip:
        try:
            import gurobipy as gp
            from gurobipy import GRB
            
            remaining_time = time_limit - (time.time() - start_time) - 2
            if remaining_time < 5:
                use_mip = False
            
            if use_mip:
                model = gp.Model("RCPSP_multi")
                model.Params.TimeLimit = remaining_time
                model.Params.Threads = 1
                model.Params.MIPFocus = 1  # Focus on finding feasible solutions
                model.Params.OutputFlag = 0
                
                # Variables: x[j,m,t] = 1 if job j starts in mode m at time t
                x = {}
                job_mode_times = {}
                
                for j in jobs_data:
                    jid = j['job_id']
                    job_mode_times[jid] = []
                    for m in j['modes']:
                        mid = m['mode_id']
                        dur = m['duration']
                        lst = compute_lst(jid, dur)
                        es = est[jid]
                        for t in range(es, lst + 1):
                            if t + dur <= horizon + 1:
                                x[jid, mid, t] = model.addVar(vtype=GRB.BINARY, name=f"x_{jid}_{mid}_{t}")
                                job_mode_times[jid].append((mid, t))
                
                model.update()
                
                # Each job assigned exactly once
                for j in jobs_data:
                    jid = j['job_id']
                    model.addConstr(
                        gp.quicksum(x[jid, mid, t] for mid, t in job_mode_times[jid]) == 1,
                        name=f"assign_{jid}"
                    )
                
                # Precedence constraints
                # For each precedence j -> s: start[j] + dur[j] <= start[s]
                for j in jobs_data:
                    jid = j['job_id']
                    for sid in j['successors']:
                        # sum over j's (m,t): (t + dur_m) * x[j,m,t] <= sum over s's (m',t'): t' * x[s,m',t']
                        # Linearize: for each combination... too many
                        # Better: use auxiliary variables for start time and mode
                        pass
                
                # Actually, let's use a different formulation with start time variables
                # start[j] = sum_t,m t * x[j,m,t]
                # dur[j] = sum_t,m dur_m * x[j,m,t]
                # For precedence: start[j] + dur[j] <= start[s]
                
                start_var = {}
                finish_var = {}
                for j in jobs_data:
                    jid = j['job_id']
                    start_var[jid] = gp.quicksum(t * x[jid, mid, t] for mid, t in job_mode_times[jid])
                    
                    dur_map = {}
                    for m in j['modes']:
                        dur_map[m['mode_id']] = m['duration']
                    
                    finish_var[jid] = gp.quicksum((t + dur_map[mid]) * x[jid, mid, t] for mid, t in job_mode_times[jid])
                
                # Precedence
                for j in jobs_data:
                    jid = j['job_id']
                    for sid in j['successors']:
                        model.addConstr(finish_var[jid] <= start_var[sid], name=f"prec_{jid}_{sid}")
                
                # Renewable resource constraints
                # For each period t and resource r:
                # sum of jobs processing at time t in their chosen mode <= capacity
                # Job j in mode m started at time s processes during [s, s+dur-1]
                # So x[j,m,s] contributes to periods s, s+1, ..., s+dur-1
                
                # Build period -> list of (jid, mid, s) that are active
                period_activities = defaultdict(list)
                for j in jobs_data:
                    jid = j['job_id']
                    for m in j['modes']:
                        mid = m['mode_id']
                        dur = m['duration']
                        if dur == 0:
                            continue
                        renew = m['renewable_consumption']
                        for (mid2, t) in job_mode_times[jid]:
                            if mid2 == mid:
                                for period in range(t, t + dur):
                                    period_activities[period].append((jid, mid, t, renew))
                
                for period in period_activities:
                    for r in range(num_renewable):
                        terms = [(renew[r], x[jid, mid, t]) for jid, mid, t, renew in period_activities[period] if renew[r] > 0]
                        if terms:
                            model.addConstr(
                                gp.quicksum(coef * var for coef, var in terms) <= renewable_caps[r],
                                name=f"renew_{r}_{period}"
                            )
                
                # Nonrenewable resource constraints
                for r in range(num_nonrenewable):
                    terms = []
                    for j in jobs_data:
                        jid = j['job_id']
                        nr_map = {}
                        for m in j['modes']:
                            nr_map[m['mode_id']] = m['nonrenewable_consumption'][r]
                        for mid, t in job_mode_times[jid]:
                            if nr_map[mid] > 0:
                                terms.append((nr_map[mid], x[jid, mid, t]))
                    if terms:
                        model.addConstr(
                            gp.quicksum(coef * var for coef, var in terms) <= nonrenewable_caps[r],
                            name=f"nonrenew_{r}"
                        )
                
                # Objective: minimize total delay + epsilon * makespan
                # Total delay = sum over projects of (completion[sink] - (release + cp_dur))
                # Makespan = max over projects of completion[sink]
                
                makespan_var = model.addVar(vtype=GRB.INTEGER, name="makespan", lb=0, ub=horizon)
                
                total_delay_expr = gp.LinExpr()
                for p in projects:
                    pid = p['project_id']
                    sink = p['artificial_sink_job_id']
                    # completion of sink = start of sink (since dur = 0)
                    total_delay_expr += start_var[sink] - (p['release_date'] + cp_duration[pid])
                    model.addConstr(makespan_var >= start_var[sink], name=f"makespan_{pid}")
                
                model.setObjective(total_delay_expr + 0.001 * makespan_var, GRB.MINIMIZE)
                
                # Warm start with best solution
                if best_solution is not None:
                    bs_start, bs_mode = best_solution
                    for j in jobs_data:
                        jid = j['job_id']
                        for mid, t in job_mode_times[jid]:
                            if (jid, mid, t) in x:
                                if mid == bs_mode[jid] and t == bs_start[jid]:
                                    x[jid, mid, t].Start = 1.0
                                else:
                                    x[jid, mid, t].Start = 0.0
                
                # Callback to log solutions
                def callback(model, where):
                    nonlocal best_solution, best_obj, best_delay, best_makespan
                    if where == GRB.Callback.MIPSOL:
                        obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                        if obj_val < best_obj:
                            # Extract solution
                            new_start = {}
                            new_mode = {}
                            for j in jobs_data:
                                jid = j['job_id']
                                for mid, t in job_mode_times[jid]:
                                    if (jid, mid, t) in x:
                                        val = model.cbGetSolution(x[jid, mid, t])
                                        if val > 0.5:
                                            new_start[jid] = t
                                            new_mode[jid] = mid
                                            break
                            
                            if len(new_start) == len(jobs_data):
                                d, mk = compute_objective(new_start, new_mode)
                                o = combined_objective(d, mk)
                                if o < best_obj:
                                    best_solution = (new_start, new_mode)
                                    best_obj = o
                                    best_delay = d
                                    best_makespan = mk
                                    if logger:
                                        logger.log(o)
                
                model.optimize(callback)
                
                if model.SolCount > 0:
                    new_start = {}
                    new_mode = {}
                    for j in jobs_data:
                        jid = j['job_id']
                        for mid, t in job_mode_times[jid]:
                            if (jid, mid, t) in x:
                                if x[jid, mid, t].X > 0.5:
                                    new_start[jid] = t
                                    new_mode[jid] = mid
                                    break
                    
                    if len(new_start) == len(jobs_data):
                        d, mk = compute_objective(new_start, new_mode)
                        o = combined_objective(d, mk)
                        if o < best_obj:
                            best_solution = (new_start, new_mode)
                            best_obj = o
                            best_delay = d
                            best_makespan = mk
                            if logger:
                                logger.log(o)
        
        except Exception as e:
            pass  # Fall back to heuristic solution
    
    # If still have time, do more local search
    remaining = time_limit - (time.time() - start_time)
    if remaining > 2 and best_solution is not None:
        import random
        random.seed(12345)
        
        cur_start, cur_mode = best_solution
        cur_delay, cur_mk = compute_objective(cur_start, cur_mode)
        cur_obj = combined_objective(cur_delay, cur_mk)
        
        # Enhanced local search: try swapping modes and rebuilding with different priorities
        iteration = 0
        while time.time() - start_time < time_limit - 1:
            iteration += 1
            random.seed(iteration * 7 + 31)
            
            # Perturbation: randomly change some modes
            new_modes = dict(cur_mode)
            num_changes = random.randint(1, max(1, num_jobs // 5))
            real_jobs = [j['job_id'] for j in jobs_data if len(job_by_id[j['job_id']]['modes']) > 1]
            
            if real_jobs:
                for _ in range(min(num_changes, len(real_jobs))):
                    jid = random.choice(real_jobs)
                    j = job_by_id[jid]
                    m = random.choice(j['modes'])
                    new_modes[jid] = m['mode_id']
            
            # Random priority perturbation
            perturbation = {jid: cur_start.get(jid, est[jid]) + random.gauss(0, 3) for jid in all_job_ids}
            pk = lambda jid: (perturbation[jid], topo_index[jid])
            plist = make_feasible_list(pk)
            
            res = compute_serial_sgs(plist, new_modes)
            if res is not None:
                ss, sm = res
                d, mk = compute_objective(ss, sm)
                o = combined_objective(d, mk)
                
                # Accept improvement or sometimes accept worse (simulated annealing-like)
                if o < cur_obj:
                    cur_start, cur_mode = ss, sm
                    cur_delay, cur_mk = d, mk
                    cur_obj = o
                    
                    if o < best_obj:
                        best_solution = (ss, sm)
                        best_obj = o
                        best_delay = d
                        best_makespan = mk
                        if logger:
                            logger.log(o)
                elif random.random() < 0.05:  # Small chance to accept worse
                    cur_start, cur_mode = ss, sm
                    cur_delay, cur_mk = d, mk
                    cur_obj = o
    
    # Output solution
    sched_start, sched_mode = best_solution
    
    schedule = []
    for j in jobs_data:
        jid = j['job_id']
        schedule.append({
            "job_id": jid,
            "mode_id": sched_mode[jid],
            "start_time": sched_start[jid]
        })
    
    output = {
        "objective_value": best_obj,
        "schedule": schedule,
        "makespan": best_makespan
    }
    
    with open(solution_path, 'w') as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()
    
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)
import argparse
import json
import time
import itertools
from collections import defaultdict

def solve(instance_path, solution_path, time_limit, log_path):
    from solution_logger import SolutionLogger
    
    with open(instance_path, 'r') as f:
        data = json.load(f)
    
    n = data['n']
    m = data['m']
    num_scenarios = data['num_scenarios']
    facility_caps = data['facility_capacity_limits']
    cap_req = data['capacity_requirements']  # [m][n]
    release_times = data['release_times']
    scenarios = data['scenarios']
    
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    start_time = time.time()
    
    def time_remaining():
        return time_limit - (time.time() - start_time)
    
    # For small instances, try to use Gurobi MIP
    # For larger instances, use heuristic approaches
    
    def schedule_facility_scenario(jobs_on_facility, facility, scenario_idx):
        """
        Schedule a set of jobs on a single facility for a given scenario.
        Jobs must respect release times and resource capacity.
        Returns dict: job -> start_time, and the makespan for this facility.
        Uses a greedy/list scheduling approach.
        """
        if not jobs_on_facility:
            return {}, 0
        
        proc_times = scenarios[scenario_idx]['processing_times']
        cap = facility_caps[facility]
        
        # Sort jobs by release time, then by processing time (longest first for better packing)
        job_info = []
        for j in jobs_on_facility:
            pt = proc_times[facility][j]
            cr = cap_req[facility][j]
            rt = release_times[j]
            job_info.append((j, rt, pt, cr))
        
        # Sort by release time, then longest processing time first
        job_info.sort(key=lambda x: (x[1], -x[2]))
        
        # Event-based scheduling
        # Track resource usage over time using events
        # For each job, find earliest start time >= release_time where resource is available
        
        scheduled = {}  # job -> (start, finish)
        # intervals: list of (start, finish, resource) for scheduled jobs
        intervals = []
        
        for j, rt, pt, cr in job_info:
            if pt == 0:
                scheduled[j] = (rt, rt)
                continue
            
            # Find earliest start >= rt where adding cr doesn't exceed cap for duration pt
            # Collect all critical time points
            time_points = set()
            time_points.add(rt)
            for (s, f, r) in intervals:
                if f > rt:
                    time_points.add(max(s, rt))
                    time_points.add(f)
            
            time_points = sorted(time_points)
            
            best_start = None
            
            for candidate_start in time_points:
                if candidate_start < rt:
                    continue
                candidate_end = candidate_start + pt
                
                # Check resource feasibility: at every point in [candidate_start, candidate_end),
                # total resource + cr <= cap
                feasible = True
                
                # Check at all sub-intervals created by existing intervals
                check_points = set()
                check_points.add(candidate_start)
                for (s, f, r) in intervals:
                    if s > candidate_start and s < candidate_end:
                        check_points.add(s)
                    if f > candidate_start and f < candidate_end:
                        check_points.add(f)
                
                check_points = sorted(check_points)
                
                for cp in check_points:
                    total = cr
                    for (s, f, r) in intervals:
                        if s <= cp < f:
                            total += r
                    if total > cap:
                        feasible = False
                        break
                
                if feasible:
                    best_start = candidate_start
                    break
            
            if best_start is None:
                # Try after all existing intervals end
                all_ends = [f for (s, f, r) in intervals]
                if all_ends:
                    candidate_start = max(max(all_ends), rt)
                else:
                    candidate_start = rt
                
                # Keep trying later times
                while True:
                    candidate_end = candidate_start + pt
                    feasible = True
                    check_points = set()
                    check_points.add(candidate_start)
                    for (s, f, r) in intervals:
                        if s > candidate_start and s < candidate_end:
                            check_points.add(s)
                        if f > candidate_start and f < candidate_end:
                            check_points.add(f)
                    check_points = sorted(check_points)
                    
                    for cp in check_points:
                        total = cr
                        for (s, f, r) in intervals:
                            if s <= cp < f:
                                total += r
                        if total > cap:
                            feasible = False
                            break
                    
                    if feasible:
                        best_start = candidate_start
                        break
                    else:
                        # Move to next event point after candidate_start
                        next_points = []
                        for (s, f, r) in intervals:
                            if f > candidate_start:
                                next_points.append(f)
                        if next_points:
                            candidate_start = min(next_points)
                        else:
                            candidate_start = candidate_start + 1
            
            scheduled[j] = (best_start, best_start + pt)
            intervals.append((best_start, best_start + pt, cr))
        
        makespan = max(f for (s, f) in scheduled.values()) if scheduled else 0
        return scheduled, makespan
    
    def schedule_facility_scenario_better(jobs_on_facility, facility, scenario_idx):
        """
        More careful scheduling using event-based resource tracking.
        """
        if not jobs_on_facility:
            return {}, 0
        
        proc_times = scenarios[scenario_idx]['processing_times']
        cap = facility_caps[facility]
        
        job_info = []
        for j in jobs_on_facility:
            pt = proc_times[facility][j]
            cr = cap_req[facility][j]
            rt = release_times[j]
            job_info.append((j, rt, pt, cr))
        
        # Sort by release time, then longest processing time first
        job_info.sort(key=lambda x: (x[1], -x[2]))
        
        scheduled = {}
        # Use a more efficient resource profile
        # resource_profile: sorted list of (time, resource_change)
        # At any time t, resource usage = sum of changes for all times <= t
        events = []  # list of (time, delta)
        
        def get_resource_at(t):
            total = 0
            for (et, ed) in events:
                if et <= t:
                    total += ed
                else:
                    break
            return total
        
        def check_feasibility(start, end, cr):
            """Check if adding cr resource from start to end is feasible."""
            # Get all event times in [start, end)
            critical_times = [start]
            for (et, ed) in events:
                if start < et < end:
                    critical_times.append(et)
            
            for t in critical_times:
                usage = cr
                for (et, ed_val) in events:
                    if et <= t:
                        usage += ed_val if (et, ed_val) in _cumulative_cache(t) else 0
                # Actually let me just compute properly
                pass
            
            # Simpler: compute resource usage at each critical point
            for t in critical_times:
                total = cr
                # Sum all active intervals at time t
                for i in range(0, len(events), 2):
                    if i + 1 < len(events):
                        s_ev = events[i]
                        e_ev = events[i+1]
                    # This is getting complicated; just use intervals directly
                pass
            return True
        
        # Fall back to interval-based approach but more efficient
        intervals = []  # (start, finish, resource)
        
        for j, rt, pt, cr in job_info:
            if pt == 0:
                scheduled[j] = (rt, rt)
                continue
            
            if cr > cap:
                # Infeasible, but still schedule it (shouldn't happen with valid data)
                scheduled[j] = (rt, rt + pt)
                intervals.append((rt, rt + pt, cr))
                continue
            
            # Collect candidate start times
            candidates = set()
            candidates.add(rt)
            for (s, f, r) in intervals:
                if f >= rt:
                    candidates.add(f)
                if s >= rt:
                    candidates.add(s)
            
            candidates = sorted(candidates)
            
            found = False
            for cs in candidates:
                if cs < rt:
                    continue
                ce = cs + pt
                
                # Check feasibility
                check_pts = set()
                check_pts.add(cs)
                for (s, f, r) in intervals:
                    if cs < s < ce:
                        check_pts.add(s)
                    if cs < f < ce:
                        check_pts.add(f)
                
                feasible = True
                for cp in sorted(check_pts):
                    total = cr
                    for (s, f, r) in intervals:
                        if s <= cp < f:
                            total += r
                    if total > cap:
                        feasible = False
                        break
                
                if feasible:
                    scheduled[j] = (cs, ce)
                    intervals.append((cs, ce, cr))
                    found = True
                    break
            
            if not found:
                # Schedule after everything
                max_end = max((f for s, f, r in intervals), default=rt)
                cs = max(max_end, rt)
                scheduled[j] = (cs, cs + pt)
                intervals.append((cs, cs + pt, cr))
        
        makespan = max(f for (s, f) in scheduled.values()) if scheduled else 0
        return scheduled, makespan
    
    def evaluate_assignment(assignment):
        """Given assignment[j] = facility, compute expected makespan."""
        total_expected = 0.0
        all_schedules = {}
        
        for s_idx in range(num_scenarios):
            prob = scenarios[s_idx]['probability']
            
            # Group jobs by facility
            facility_jobs = defaultdict(list)
            for j in range(n):
                facility_jobs[assignment[j]].append(j)
            
            scenario_makespan = 0
            scenario_schedule = {}
            
            for fac in range(m):
                jobs = facility_jobs[fac]
                if not jobs:
                    continue
                sched, ms = schedule_facility_scenario_better(jobs, fac, s_idx)
                for j, (st, ft) in sched.items():
                    scenario_schedule[j] = (fac, st, ft)
                scenario_makespan = max(scenario_makespan, ms)
            
            total_expected += prob * scenario_makespan
            all_schedules[s_idx] = scenario_schedule
        
        return total_expected, all_schedules
    
    # Strategy: For small n*m, try enumeration or MIP; otherwise use heuristic
    
    best_obj = float('inf')
    best_assignment = None
    best_schedules = None
    
    def update_best(obj, assignment, schedules):
        nonlocal best_obj, best_assignment, best_schedules
        if obj < best_obj:
            best_obj = obj
            best_assignment = dict(assignment)
            best_schedules = schedules
            if logger:
                logger.log(obj)
    
    if n <= 12 and m <= 4:
        # Try to use Gurobi for assignment + scheduling
        # But the scheduling part is complex with resource constraints
        # Let's try a smarter enumeration / local search
        
        # First, get a greedy initial assignment
        # Assign each job to the facility where its average processing time is smallest
        assignment = {}
        for j in range(n):
            best_fac = 0
            best_avg = float('inf')
            for fac in range(m):
                avg_pt = sum(scenarios[s]['processing_times'][fac][j] * scenarios[s]['probability'] 
                           for s in range(num_scenarios))
                if avg_pt < best_avg:
                    best_avg = avg_pt
                    best_fac = fac
            assignment[j] = best_fac
        
        obj, scheds = evaluate_assignment(assignment)
        update_best(obj, assignment, scheds)
        
        # Try capacity-aware greedy
        assignment2 = {}
        for j in range(n):
            best_fac = 0
            best_score = float('inf')
            for fac in range(m):
                avg_pt = sum(scenarios[s]['processing_times'][fac][j] * scenarios[s]['probability'] 
                           for s in range(num_scenarios))
                # Penalize if resource consumption is high relative to capacity
                resource_ratio = cap_req[fac][j] / facility_caps[fac] if facility_caps[fac] > 0 else float('inf')
                score = avg_pt * (1 + resource_ratio)
                if score < best_score:
                    best_score = score
                    best_fac = fac
            assignment2[j] = best_fac
        
        obj2, scheds2 = evaluate_assignment(assignment2)
        update_best(obj2, assignment2, scheds2)
        
        # If very small, enumerate all possible assignments
        if m ** n <= 100000 and time_remaining() > 2:
            count = 0
            for combo in itertools.product(range(m), repeat=n):
                if time_remaining() < 1:
                    break
                count += 1
                a = {j: combo[j] for j in range(n)}
                obj, scheds = evaluate_assignment(a)
                update_best(obj, a, scheds)
        else:
            # Local search: try swapping each job to each facility
            improved = True
            while improved and time_remaining() > 2:
                improved = False
                for j in range(n):
                    current_fac = best_assignment[j]
                    for fac in range(m):
                        if fac == current_fac:
                            continue
                        if time_remaining() < 1:
                            break
                        test_assignment = dict(best_assignment)
                        test_assignment[j] = fac
                        obj, scheds = evaluate_assignment(test_assignment)
                        if obj < best_obj:
                            update_best(obj, test_assignment, scheds)
                            improved = True
                            break
                    if improved:
                        break
            
            # Try pairwise swaps
            improved = True
            while improved and time_remaining() > 2:
                improved = False
                for j1 in range(n):
                    for j2 in range(j1+1, n):
                        if best_assignment[j1] == best_assignment[j2]:
                            continue
                        if time_remaining() < 1:
                            break
                        test_assignment = dict(best_assignment)
                        test_assignment[j1], test_assignment[j2] = test_assignment[j2], test_assignment[j1]
                        obj, scheds = evaluate_assignment(test_assignment)
                        if obj < best_obj:
                            update_best(obj, test_assignment, scheds)
                            improved = True
                            break
                    if improved:
                        break
    
    else:
        # For larger instances, use heuristic approach
        
        # Greedy assignment: minimize expected processing time
        assignment = {}
        for j in range(n):
            best_fac = 0
            best_avg = float('inf')
            for fac in range(m):
                avg_pt = sum(scenarios[s]['processing_times'][fac][j] * scenarios[s]['probability'] 
                           for s in range(num_scenarios))
                if avg_pt < best_avg:
                    best_avg = avg_pt
                    best_fac = fac
            assignment[j] = best_fac
        
        obj, scheds = evaluate_assignment(assignment)
        update_best(obj, assignment, scheds)
        
        # Load-balanced greedy
        facility_load = [0.0] * m
        assignment3 = {}
        # Sort jobs by max processing time (descending) for better load balancing
        jobs_sorted = sorted(range(n), key=lambda j: max(
            sum(scenarios[s]['processing_times'][fac][j] * scenarios[s]['probability'] 
                for s in range(num_scenarios)) for fac in range(m)), reverse=True)
        
        for j in jobs_sorted:
            best_fac = 0
            best_score = float('inf')
            for fac in range(m):
                avg_pt = sum(scenarios[s]['processing_times'][fac][j] * scenarios[s]['probability'] 
                           for s in range(num_scenarios))
                # Score = expected completion considering current load
                score = max(facility_load[fac], release_times[j]) + avg_pt
                if score < best_score:
                    best_score = score
                    best_fac = fac
            assignment3[j] = best_fac
            facility_load[best_fac] = best_score
        
        obj3, scheds3 = evaluate_assignment(assignment3)
        update_best(obj3, assignment3, scheds3)
        
        # Local search
        improved = True
        iterations = 0
        while improved and time_remaining() > 2:
            improved = False
            iterations += 1
            for j in range(n):
                current_fac = best_assignment[j]
                for fac in range(m):
                    if fac == current_fac:
                        continue
                    if time_remaining() < 1:
                        break
                    test_assignment = dict(best_assignment)
                    test_assignment[j] = fac
                    obj, scheds = evaluate_assignment(test_assignment)
                    if obj < best_obj:
                        update_best(obj, test_assignment, scheds)
                        improved = True
                        break
                if improved:
                    break
                if time_remaining() < 1:
                    break
    
    # Now try to use Gurobi MIP for the assignment part if time allows
    if time_remaining() > 5 and n <= 20 and m <= 5:
        try:
            import gurobipy as gp
            from gurobipy import GRB
            
            model = gp.Model("assignment")
            model.setParam('OutputFlag', 0)
            model.setParam('TimeLimit', max(1, time_remaining() - 2))
            model.setParam('Threads', 1)
            
            # Binary assignment variables
            x = {}
            for j in range(n):
                for fac in range(m):
                    x[j, fac] = model.addVar(vtype=GRB.BINARY, name=f"x_{j}_{fac}")
            
            # Each job assigned to exactly one facility
            for j in range(n):
                model.addConstr(gp.quicksum(x[j, fac] for fac in range(m)) == 1)
            
            # Makespan variables per scenario
            C = {}
            for s_idx in range(num_scenarios):
                C[s_idx] = model.addVar(lb=0, name=f"C_{s_idx}")
            
            # For scheduling, we need a lower bound on makespan
            # Lower bound: for each facility, the makespan >= release_time[j] + processing_time[fac][j][s] for each assigned job
            # Also, resource-based lower bound
            
            # Simple lower bound: makespan >= max over assigned jobs of (release + proc)
            for s_idx in range(num_scenarios):
                proc = scenarios[s_idx]['processing_times']
                for j in range(n):
                    for fac in range(m):
                        model.addConstr(
                            C[s_idx] >= (release_times[j] + proc[fac][j]) * x[j, fac]
                        )
                
                # Resource-based lower bound for each facility
                for fac in range(m):
                    cap = facility_caps[fac]
                    if cap > 0:
                        # Sum of resource*time / capacity is a lower bound on the time span needed
                        # But we need to account for release times
                        # Simple: total_work / capacity
                        model.addConstr(
                            C[s_idx] >= gp.quicksum(
                                (cap_req[fac][j] * proc[fac][j] / cap) * x[j, fac]
                                for j in range(n)
                            )
                        )
            
            # Objective: minimize expected makespan
            model.setObjective(
                gp.quicksum(scenarios[s_idx]['probability'] * C[s_idx] for s_idx in range(num_scenarios)),
                GRB.MINIMIZE
            )
            
            # Warm start with best known assignment
            if best_assignment is not None:
                for j in range(n):
                    for fac in range(m):
                        x[j, fac].Start = 1.0 if best_assignment[j] == fac else 0.0
            
            model.optimize()
            
            if model.SolCount > 0:
                # Extract assignment
                mip_assignment = {}
                for j in range(n):
                    for fac in range(m):
                        if x[j, fac].X > 0.5:
                            mip_assignment[j] = fac
                            break
                
                # Evaluate with actual scheduling
                obj, scheds = evaluate_assignment(mip_assignment)
                update_best(obj, mip_assignment, scheds)
                
                # Local search on MIP solution
                improved = True
                while improved and time_remaining() > 1:
                    improved = False
                    for j in range(n):
                        current_fac = best_assignment[j]
                        for fac in range(m):
                            if fac == current_fac:
                                continue
                            if time_remaining() < 0.5:
                                break
                            test_assignment = dict(best_assignment)
                            test_assignment[j] = fac
                            obj, scheds = evaluate_assignment(test_assignment)
                            if obj < best_obj:
                                update_best(obj, test_assignment, scheds)
                                improved = True
                                break
                        if improved:
                            break
                        if time_remaining() < 0.5:
                            break
        except Exception:
            pass
    
    # Try different scheduling orderings to potentially improve
    if time_remaining() > 2 and best_assignment is not None:
        # Try scheduling with different priority rules
        def schedule_facility_scenario_spt(jobs_on_facility, facility, scenario_idx):
            """Schedule using shortest processing time first."""
            if not jobs_on_facility:
                return {}, 0
            
            proc_times = scenarios[scenario_idx]['processing_times']
            cap = facility_caps[facility]
            
            job_info = []
            for j in jobs_on_facility:
                pt = proc_times[facility][j]
                cr = cap_req[facility][j]
                rt = release_times[j]
                job_info.append((j, rt, pt, cr))
            
            # Sort by processing time (shortest first)
            job_info.sort(key=lambda x: (x[2], x[1]))
            
            scheduled = {}
            intervals = []
            
            for j, rt, pt, cr in job_info:
                if pt == 0:
                    scheduled[j] = (rt, rt)
                    continue
                
                candidates = set()
                candidates.add(rt)
                for (s, f, r) in intervals:
                    if f >= rt:
                        candidates.add(f)
                    if s >= rt:
                        candidates.add(s)
                
                candidates = sorted(candidates)
                
                found = False
                for cs in candidates:
                    if cs < rt:
                        continue
                    ce = cs + pt
                    
                    check_pts = set()
                    check_pts.add(cs)
                    for (s, f, r) in intervals:
                        if cs < s < ce:
                            check_pts.add(s)
                        if cs < f < ce:
                            check_pts.add(f)
                    
                    feasible = True
                    for cp in sorted(check_pts):
                        total = cr
                        for (s, f, r) in intervals:
                            if s <= cp < f:
                                total += r
                        if total > cap:
                            feasible = False
                            break
                    
                    if feasible:
                        scheduled[j] = (cs, ce)
                        intervals.append((cs, ce, cr))
                        found = True
                        break
                
                if not found:
                    max_end = max((f for s, f, r in intervals), default=rt)
                    cs = max(max_end, rt)
                    scheduled[j] = (cs, cs + pt)
                    intervals.append((cs, cs + pt, cr))
            
            makespan = max(f for (s, f) in scheduled.values()) if scheduled else 0
            return scheduled, makespan
        
        def schedule_facility_scenario_lpt(jobs_on_facility, facility, scenario_idx):
            """Schedule using longest processing time first."""
            if not jobs_on_facility:
                return {}, 0
            
            proc_times = scenarios[scenario_idx]['processing_times']
            cap = facility_caps[facility]
            
            job_info = []
            for j in jobs_on_facility:
                pt = proc_times[facility][j]
                cr = cap_req[facility][j]
                rt = release_times[j]
                job_info.append((j, rt, pt, cr))
            
            # Sort by release time, then longest processing time first
            job_info.sort(key=lambda x: (x[1], -x[2]))
            
            scheduled = {}
            intervals = []
            
            for j, rt, pt, cr in job_info:
                if pt == 0:
                    scheduled[j] = (rt, rt)
                    continue
                
                candidates = set()
                candidates.add(rt)
                for (s, f, r) in intervals:
                    if f >= rt:
                        candidates.add(f)
                    if s >= rt:
                        candidates.add(s)
                
                candidates = sorted(candidates)
                
                found = False
                for cs in candidates:
                    if cs < rt:
                        continue
                    ce = cs + pt
                    
                    check_pts = set()
                    check_pts.add(cs)
                    for (s, f, r) in intervals:
                        if cs < s < ce:
                            check_pts.add(s)
                        if cs < f < ce:
                            check_pts.add(f)
                    
                    feasible = True
                    for cp in sorted(check_pts):
                        total = cr
                        for (s, f, r) in intervals:
                            if s <= cp < f:
                                total += r
                        if total > cap:
                            feasible = False
                            break
                    
                    if feasible:
                        scheduled[j] = (cs, ce)
                        intervals.append((cs, ce, cr))
                        found = True
                        break
                
                if not found:
                    max_end = max((f for s, f, r in intervals), default=rt)
                    cs = max(max_end, rt)
                    scheduled[j] = (cs, cs + pt)
                    intervals.append((cs, cs + pt, cr))
            
            makespan = max(f for (s, f) in scheduled.values()) if scheduled else 0
            return scheduled, makespan
        
        # Try different scheduling strategies for current best assignment
        for sched_func in [schedule_facility_scenario_spt, schedule_facility_scenario_lpt]:
            if time_remaining() < 1:
                break
            
            total_expected = 0.0
            all_schedules = {}
            
            facility_jobs = defaultdict(list)
            for j in range(n):
                facility_jobs[best_assignment[j]].append(j)
            
            for s_idx in range(num_scenarios):
                prob = scenarios[s_idx]['probability']
                scenario_makespan = 0
                scenario_schedule = {}
                
                for fac in range(m):
                    jobs = facility_jobs[fac]
                    if not jobs:
                        continue
                    sched, ms = sched_func(jobs, fac, s_idx)
                    for j, (st, ft) in sched.items():
                        scenario_schedule[j] = (fac, st, ft)
                    scenario_makespan = max(scenario_makespan, ms)
                
                total_expected += prob * scenario_makespan
                all_schedules[s_idx] = scenario_schedule
            
            if total_expected < best_obj:
                update_best(total_expected, best_assignment, all_schedules)
    
    # Construct output
    output = {
        "objective_value": best_obj,
        "assignment": {},
        "schedule": {}
    }
    
    for j in range(n):
        output["assignment"][str(j)] = best_assignment[j]
    
    for s_idx in range(num_scenarios):
        scenario_key = str(s_idx)
        output["schedule"][scenario_key] = {}
        for j in range(n):
            if j in best_schedules[s_idx]:
                fac, st, ft = best_schedules[s_idx][j]
            else:
                fac = best_assignment[j]
                st = release_times[j]
                ft = release_times[j] + scenarios[s_idx]['processing_times'][fac][j]
            output["schedule"][scenario_key][str(j)] = {
                "facility": fac,
                "start": st,
                "finish": ft
            }
    
    with open(solution_path, 'w') as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)
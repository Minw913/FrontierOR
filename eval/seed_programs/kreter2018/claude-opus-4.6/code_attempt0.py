import argparse
import json
import time
import sys
from collections import defaultdict, deque

def solve(instance_path, solution_path, time_limit, log_path):
    from solution_logger import SolutionLogger
    
    start_time = time.time()
    
    with open(instance_path, 'r') as f:
        data = json.load(f)
    
    num_real = data['num_real_activities']
    num_total = data['num_total_activities']
    num_resources = data['num_resource_types']
    deadline = data['max_project_duration']
    activities = data['activities']
    precedences = data['precedence_relations']
    resource_costs = data['resource_costs']
    
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    # Build activity info
    act_info = {}
    for act in activities:
        act_info[act['id']] = {
            'duration': act['duration'],
            'resource_requirements': act['resource_requirements']
        }
    
    act_ids = sorted(act_info.keys())
    
    # Build adjacency lists
    successors = defaultdict(list)
    predecessors = defaultdict(list)
    for pred, succ in precedences:
        successors[pred].append(succ)
        predecessors[succ].append(pred)
    
    # Compute earliest start times (forward pass)
    earliest = {}
    # Topological sort
    in_degree = defaultdict(int)
    for a in act_ids:
        in_degree[a] = 0
    for pred, succ in precedences:
        in_degree[succ] += 1
    
    queue = deque()
    for a in act_ids:
        if in_degree[a] == 0:
            queue.append(a)
    
    topo_order = []
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for s in successors[node]:
            in_degree[s] -= 1
            if in_degree[s] == 0:
                queue.append(s)
    
    for a in topo_order:
        es = 0
        for p in predecessors[a]:
            es = max(es, earliest[p] + act_info[p]['duration'])
        earliest[a] = es
    
    # Compute latest start times (backward pass)
    latest = {}
    for a in reversed(topo_order):
        ls = deadline - act_info[a]['duration']
        for s in successors[a]:
            ls = min(ls, latest[s] - act_info[a]['duration'])
        latest[a] = ls
    
    # Find fictitious start and end activities (duration 0, all resource reqs 0)
    # The project start must begin at time 0
    # Identify start activity: no predecessors
    start_act = None
    end_act = None
    for a in act_ids:
        if len(predecessors[a]) == 0:
            start_act = a
        if len(successors[a]) == 0:
            end_act = a
    
    # Use Gurobi MIP
    import gurobipy as gp
    from gurobipy import GRB
    
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time < 1:
        remaining_time = 1
    
    model = gp.Model("RCPSP_cost")
    model.setParam('TimeLimit', remaining_time - 1)
    model.setParam('Threads', 1)
    model.setParam('MIPFocus', 1)  # Focus on finding feasible solutions
    
    # Decision variables: start time for each activity
    s = {}
    for a in act_ids:
        if a == start_act:
            s[a] = model.addVar(lb=0, ub=0, vtype=GRB.INTEGER, name=f"s_{a}")
        else:
            s[a] = model.addVar(lb=earliest[a], ub=latest[a], vtype=GRB.INTEGER, name=f"s_{a}")
    
    # Resource provision variables
    R = {}
    for k in range(num_resources):
        # Upper bound: sum of all requirements for resource k
        max_req = sum(act_info[a]['resource_requirements'][k] for a in act_ids)
        R[k] = model.addVar(lb=0, ub=max_req, vtype=GRB.INTEGER, name=f"R_{k}")
    
    # Objective: minimize total resource cost
    model.setObjective(gp.quicksum(resource_costs[k] * R[k] for k in range(num_resources)), GRB.MINIMIZE)
    
    # Precedence constraints
    for pred, succ in precedences:
        model.addConstr(s[succ] >= s[pred] + act_info[pred]['duration'])
    
    # For resource constraints, we use time-indexed binary variables
    # x[a,t] = 1 if activity a starts at time t
    # But this can be large. Let's check the size.
    
    # Calculate total number of binary variables needed
    total_bins = sum(latest[a] - earliest[a] + 1 for a in act_ids if act_info[a]['duration'] > 0)
    
    # Filter to real activities (non-zero duration or resource usage)
    real_acts = [a for a in act_ids if act_info[a]['duration'] > 0]
    
    # Check if time-indexed formulation is tractable
    # If too many variables, use a different approach
    if total_bins <= 50000 and deadline <= 500:
        # Time-indexed formulation
        x = {}
        for a in real_acts:
            for t in range(earliest[a], latest[a] + 1):
                x[a, t] = model.addVar(vtype=GRB.BINARY, name=f"x_{a}_{t}")
        
        # Link x to s
        for a in real_acts:
            model.addConstr(gp.quicksum(x[a, t] for t in range(earliest[a], latest[a] + 1)) == 1)
            model.addConstr(s[a] == gp.quicksum(t * x[a, t] for t in range(earliest[a], latest[a] + 1)))
        
        # Resource constraints for each time period and resource
        for k in range(num_resources):
            # For each time period, compute which activities could be active
            # Activity a is active in period t if s[a] <= t < s[a] + duration[a]
            # i.e., t - duration[a] + 1 <= s[a] <= t
            # i.e., x[a, tau] for tau in [max(earliest[a], t - dur + 1), min(latest[a], t)]
            
            # Find the range of periods where any activity could be active
            min_period = min(earliest[a] for a in real_acts)
            max_period = max(latest[a] + act_info[a]['duration'] - 1 for a in real_acts)
            
            for t in range(min_period, max_period + 1):
                terms = []
                for a in real_acts:
                    req = act_info[a]['resource_requirements'][k]
                    if req == 0:
                        continue
                    dur = act_info[a]['duration']
                    tau_min = max(earliest[a], t - dur + 1)
                    tau_max = min(latest[a], t)
                    if tau_min > tau_max:
                        continue
                    for tau in range(tau_min, tau_max + 1):
                        terms.append(req * x[a, tau])
                
                if terms:
                    model.addConstr(gp.quicksum(terms) <= R[k])
        
        model.update()
        
        # Add callback to log solutions
        best_obj = [float('inf')]
        best_solution = [None]
        
        def callback(model, where):
            if where == GRB.Callback.MIPSOL:
                obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if obj < best_obj[0] - 0.5:
                    best_obj[0] = obj
                    # Extract solution
                    sol_s = {}
                    for a in act_ids:
                        sol_s[a] = round(model.cbGetSolution(s[a]))
                    sol_R = {}
                    for k_idx in range(num_resources):
                        sol_R[k_idx] = round(model.cbGetSolution(R[k_idx]))
                    best_solution[0] = (sol_s, sol_R, obj)
                    if logger:
                        logger.log(obj)
        
        model.optimize(callback)
        
        if model.SolCount > 0:
            sol_s = {}
            for a in act_ids:
                sol_s[a] = round(s[a].X)
            sol_R = {}
            for k in range(num_resources):
                sol_R[k] = round(R[k].X)
            obj_val = model.ObjVal
        elif best_solution[0] is not None:
            sol_s, sol_R, obj_val = best_solution[0]
        else:
            # Fallback: use earliest start times
            sol_s = {a: earliest[a] for a in act_ids}
            sol_R = {}
            for k in range(num_resources):
                max_usage = 0
                for t in range(deadline):
                    usage = 0
                    for a in real_acts:
                        if sol_s[a] <= t < sol_s[a] + act_info[a]['duration']:
                            usage += act_info[a]['resource_requirements'][k]
                    max_usage = max(max_usage, usage)
                sol_R[k] = max_usage
            obj_val = sum(resource_costs[k] * sol_R[k] for k in range(num_resources))
            if logger:
                logger.log(obj_val)
    else:
        # For large instances, use a heuristic/iterative approach
        # First try with Gurobi but with discrete-event formulation
        # Use on/off constraints with big-M or indicator constraints
        
        # Actually, let's try a simpler approach: 
        # Use the time-indexed formulation but with aggregated periods
        # Or use a list-scheduling heuristic
        
        # Heuristic: schedule activities to minimize peak resource usage
        # Try serial SGS with different priority rules
        
        def compute_resource_usage(schedule, real_acts, act_info, num_resources, deadline):
            usage = [[0] * deadline for _ in range(num_resources)]
            for a in real_acts:
                st = schedule[a]
                dur = act_info[a]['duration']
                for k in range(num_resources):
                    req = act_info[a]['resource_requirements'][k]
                    if req > 0:
                        for t in range(st, st + dur):
                            if t < deadline:
                                usage[k][t] += req
            peak = [max(usage[k]) if usage[k] else 0 for k in range(num_resources)]
            return peak, usage
        
        def compute_cost(peak, resource_costs):
            return sum(resource_costs[k] * peak[k] for k in range(len(resource_costs)))
        
        # Try multiple priority rules and keep the best
        best_obj_val = float('inf')
        best_schedule = None
        best_peak = None
        
        # Priority rules: by latest finish time, earliest start, most resource usage, etc.
        priority_rules = []
        
        # LFT (Latest Finish Time)
        lft = {a: latest[a] + act_info[a]['duration'] for a in act_ids}
        priority_rules.append(('LFT', sorted(real_acts, key=lambda a: lft[a])))
        
        # EST (Earliest Start Time)
        priority_rules.append(('EST', sorted(real_acts, key=lambda a: earliest[a])))
        
        # Most Total Resource Usage
        priority_rules.append(('MRU', sorted(real_acts, key=lambda a: -sum(
            act_info[a]['resource_requirements'][k] * act_info[a]['duration'] for k in range(num_resources)))))
        
        # Greatest Resource Demand
        priority_rules.append(('GRD', sorted(real_acts, key=lambda a: -sum(
            resource_costs[k] * act_info[a]['resource_requirements'][k] for k in range(num_resources)))))
        
        for rule_name, priority_list in priority_rules:
            if time.time() - start_time > time_limit - 2:
                break
            
            # Serial SGS: schedule activities one by one in priority order
            # respecting precedences and trying to minimize peak usage
            schedule = {}
            # Set fictitious activities
            for a in act_ids:
                if a not in real_acts:
                    schedule[a] = earliest[a]
            
            current_usage = [[0] * deadline for _ in range(num_resources)]
            current_peak = [0] * num_resources
            
            scheduled = set(a for a in act_ids if a not in real_acts)
            
            for a in priority_list:
                # Find earliest feasible start time
                es = earliest[a]
                for p in predecessors[a]:
                    if p in schedule:
                        es = max(es, schedule[p] + act_info[p]['duration'])
                
                # Try to find the start time that minimizes cost increase
                best_t = es
                best_cost_increase = float('inf')
                dur = act_info[a]['duration']
                
                for t in range(es, latest[a] + 1):
                    if t + dur > deadline:
                        break
                    # Compute cost increase
                    cost_increase = 0
                    for k in range(num_resources):
                        req = act_info[a]['resource_requirements'][k]
                        if req == 0:
                            continue
                        new_peak_k = current_peak[k]
                        for tt in range(t, t + dur):
                            new_peak_k = max(new_peak_k, current_usage[k][tt] + req)
                        cost_increase += resource_costs[k] * (new_peak_k - current_peak[k])
                    
                    if cost_increase < best_cost_increase:
                        best_cost_increase = cost_increase
                        best_t = t
                    
                    if cost_increase == 0:
                        break  # Can't do better than no increase
                
                schedule[a] = best_t
                dur = act_info[a]['duration']
                for k in range(num_resources):
                    req = act_info[a]['resource_requirements'][k]
                    if req > 0:
                        for tt in range(best_t, best_t + dur):
                            current_usage[k][tt] += req
                            current_peak[k] = max(current_peak[k], current_usage[k][tt])
                scheduled.add(a)
            
            cost = compute_cost(current_peak, resource_costs)
            if cost < best_obj_val:
                best_obj_val = cost
                best_schedule = dict(schedule)
                best_peak = list(current_peak)
                if logger:
                    logger.log(cost)
        
        # Local search: try to improve by shifting activities
        if best_schedule is not None:
            improved = True
            iteration = 0
            while improved and time.time() - start_time < time_limit - 2:
                improved = False
                iteration += 1
                
                # Recompute usage from scratch
                usage_arr = [[0] * deadline for _ in range(num_resources)]
                for a in real_acts:
                    st = best_schedule[a]
                    dur = act_info[a]['duration']
                    for k in range(num_resources):
                        req = act_info[a]['resource_requirements'][k]
                        if req > 0:
                            for t in range(st, st + dur):
                                usage_arr[k][t] += req
                
                peak_arr = [max(usage_arr[k]) if any(usage_arr[k]) else 0 for k in range(num_resources)]
                current_cost = sum(resource_costs[k] * peak_arr[k] for k in range(num_resources))
                
                for a in real_acts:
                    if time.time() - start_time > time_limit - 2:
                        break
                    
                    old_st = best_schedule[a]
                    dur = act_info[a]['duration']
                    
                    # Remove activity from usage
                    for k in range(num_resources):
                        req = act_info[a]['resource_requirements'][k]
                        if req > 0:
                            for t in range(old_st, old_st + dur):
                                usage_arr[k][t] -= req
                    
                    # Compute feasible window
                    es = earliest[a]
                    for p in predecessors[a]:
                        es = max(es, best_schedule[p] + act_info[p]['duration'])
                    ls = latest[a]
                    for sc in successors[a]:
                        if sc in best_schedule:
                            ls = min(ls, best_schedule[sc] - dur)
                    
                    best_t = old_st
                    best_local_cost = current_cost
                    
                    for t in range(es, ls + 1):
                        if t + dur > deadline:
                            break
                        # Compute new peak and cost
                        new_peak = list(peak_arr)
                        cost = 0
                        valid = True
                        for k in range(num_resources):
                            req = act_info[a]['resource_requirements'][k]
                            if req > 0:
                                pk = 0
                                for tt in range(deadline):
                                    val = usage_arr[k][tt]
                                    if t <= tt < t + dur:
                                        val += req
                                    pk = max(pk, val)
                                new_peak[k] = pk
                            else:
                                # Peak might have decreased if we removed from old position
                                pk = max(usage_arr[k])
                                new_peak[k] = pk
                            cost += resource_costs[k] * new_peak[k]
                        
                        if cost < best_local_cost:
                            best_local_cost = cost
                            best_t = t
                    
                    # Place activity at best_t
                    best_schedule[a] = best_t
                    for k in range(num_resources):
                        req = act_info[a]['resource_requirements'][k]
                        if req > 0:
                            for t in range(best_t, best_t + dur):
                                usage_arr[k][t] += req
                    
                    # Recompute peaks
                    peak_arr = [max(usage_arr[k]) if any(usage_arr[k]) else 0 for k in range(num_resources)]
                    new_cost = sum(resource_costs[k] * peak_arr[k] for k in range(num_resources))
                    
                    if new_cost < current_cost:
                        current_cost = new_cost
                        improved = True
                        if current_cost < best_obj_val:
                            best_obj_val = current_cost
                            best_peak = list(peak_arr)
                            if logger:
                                logger.log(current_cost)
        
        sol_s = best_schedule
        sol_R = {k: best_peak[k] for k in range(num_resources)}
        obj_val = best_obj_val
    
    # Write solution
    solution = {
        "objective_value": float(obj_val),
        "start_times": {str(a): int(sol_s[a]) for a in act_ids},
        "resource_amounts": {str(k): int(sol_R[k]) for k in range(num_resources)}
    }
    
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)
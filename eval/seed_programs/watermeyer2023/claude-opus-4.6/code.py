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
    num_nodes = data['num_nodes']
    num_prr = data['num_partially_renewable_resources']
    deadline = data['deadline']
    proc_times = data['processing_times']
    temporal_constraints = data['temporal_constraints']
    resources = data['resources']
    
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    n = num_nodes  # total activities: 0..num_real+1
    end_node = num_real + 1
    
    # Build adjacency for temporal constraints (for longest path / constraint propagation)
    # Each constraint: start[to] >= start[from] + delta
    adj = defaultdict(list)  # adj[from] = [(to, delta)]
    radj = defaultdict(list)  # radj[to] = [(from, delta)]
    
    for tc in temporal_constraints:
        fr, to, delta = tc['from'], tc['to'], tc['delta']
        adj[fr].append((to, delta))
        radj[to].append((fr, delta))
    
    # Compute earliest start times using longest path from node 0
    def compute_earliest():
        est = [-10**9] * n
        est[0] = 0
        # Bellman-Ford style or topological (but might have negative edges, use SPFA)
        in_queue = [False] * n
        queue = deque([0])
        in_queue[0] = True
        count = [0] * n
        
        while queue:
            u = queue.popleft()
            in_queue[u] = False
            for v, d in adj[u]:
                if est[u] + d > est[v]:
                    est[v] = est[u] + d
                    if not in_queue[v]:
                        in_queue[v] = True
                        queue.append(v)
                        count[v] += 1
                        if count[v] > n:
                            return None  # negative cycle
        return est
    
    # Compute latest start times using reverse graph from end_node
    def compute_latest(makespan):
        lst = [10**9] * n
        lst[end_node] = makespan
        # Reverse: for constraint start[to] >= start[from] + delta
        # => start[from] <= start[to] - delta
        in_queue = [False] * n
        queue = deque([end_node])
        in_queue[end_node] = True
        count = [0] * n
        
        while queue:
            v = queue.popleft()
            in_queue[v] = False
            for u, d in radj[v]:
                if lst[v] - d < lst[u]:
                    lst[u] = lst[v] - d
                    if not in_queue[u]:
                        in_queue[u] = True
                        queue.append(u)
                        count[u] += 1
                        if count[u] > n:
                            return None
        return lst
    
    est = compute_earliest()
    if est is None:
        # Infeasible
        return
    
    # Ensure est[0] = 0
    est[0] = max(est[0], 0)
    
    # Lower bound on makespan
    lb_makespan = max(est[end_node], 0)
    
    lst = compute_latest(deadline)
    if lst is None:
        return
    
    # Precompute resource usage function
    # For activity i with start time s_i, execution interval is (s_i, s_i + p_i]
    # i.e., time periods s_i+1, s_i+2, ..., s_i+p_i
    # Resource usage = number of resource's designated time periods in that interval
    # Resource consumption = usage * demand[i]
    
    # Precompute for each resource and activity, a function mapping start_time -> usage
    # Since start times range from 0 to deadline, we can precompute these
    
    # For efficiency, precompute for each resource, the set of time periods as a sorted list
    # Then for activity i starting at s, count periods in (s, s+p_i] ∩ resource_periods
    
    # Use binary search for counting
    import bisect
    
    resource_periods_sorted = []
    for r in resources:
        periods = sorted(r['time_periods'])
        resource_periods_sorted.append(periods)
    
    def compute_usage(res_idx, activity, start):
        p = proc_times[activity]
        if p == 0:
            return 0
        periods = resource_periods_sorted[res_idx]
        # Count periods in (start, start+p] = [start+1, start+p]
        lo = bisect.bisect_left(periods, start + 1)
        hi = bisect.bisect_right(periods, start + p)
        return hi - lo
    
    def compute_consumption(res_idx, activity, start):
        demand = resources[res_idx]['demands'][activity]
        if demand == 0:
            return 0
        return demand * compute_usage(res_idx, activity, start)
    
    # Try using Gurobi MIP formulation
    # Variables: s_i integer start time for each activity
    # Constraints: temporal constraints, resource constraints
    # For resource constraints with partially renewable resources, we need to model
    # the consumption which depends on start times in a nonlinear way.
    
    # One approach: use binary variables for start times (time-indexed formulation)
    # x[i,t] = 1 if activity i starts at time t
    # Then consumption for resource r from activity i = sum_t x[i,t] * consumption(r,i,t)
    
    # This can be large but manageable for small instances
    
    # Check if instance is small enough for time-indexed MIP
    total_vars = 0
    time_windows = {}
    for i in range(n):
        e = max(est[i], 0)
        l = min(lst[i], deadline)
        if proc_times[i] > 0:
            l = min(l, deadline - proc_times[i])
        l = min(l, deadline)
        e = max(e, 0)
        if i == 0:
            e = 0
            l = 0
        time_windows[i] = (e, l)
        if l >= e:
            total_vars += (l - e + 1)
    
    elapsed = time.time() - start_time
    remaining = time_limit - elapsed
    
    if remaining < 2:
        # Just use earliest start times as fallback
        result_start = list(est)
        result_start[0] = 0
        obj = result_start[end_node]
        if logger:
            logger.log(obj)
        solution = {
            "objective_value": obj,
            "start_times": result_start
        }
        with open(solution_path, 'w') as f:
            json.dump(solution, f)
        return
    
    # If total_vars is manageable, use time-indexed MIP
    # Otherwise, use a CP/heuristic approach
    
    use_mip = total_vars <= 500000
    
    if use_mip:
        import gurobipy as gp
        from gurobipy import GRB
        
        model = gp.Model("RCPSP_PRR")
        model.setParam('OutputFlag', 0)
        model.setParam('TimeLimit', max(remaining - 2, 1))
        model.setParam('Threads', 1)
        
        # Binary variables x[i,t]
        x = {}
        for i in range(n):
            e, l = time_windows[i]
            for t in range(e, l + 1):
                x[i, t] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{t}")
        
        # Each activity starts exactly once
        for i in range(n):
            e, l = time_windows[i]
            model.addConstr(
                gp.quicksum(x[i, t] for t in range(e, l + 1)) == 1,
                name=f"assign_{i}"
            )
        
        # Start time expression for each activity
        s = {}
        for i in range(n):
            e, l = time_windows[i]
            s[i] = gp.quicksum(t * x[i, t] for t in range(e, l + 1))
        
        # Temporal constraints: s[to] >= s[from] + delta
        for tc in temporal_constraints:
            fr, to, delta = tc['from'], tc['to'], tc['delta']
            model.addConstr(s[to] >= s[fr] + delta, name=f"tc_{fr}_{to}")
        
        # Resource constraints
        for r_idx, r in enumerate(resources):
            cap = r['capacity']
            demands = r['demands']
            
            # Total consumption = sum_i sum_t x[i,t] * consumption(r, i, t)
            terms = []
            for i in range(n):
                if demands[i] == 0 or proc_times[i] == 0:
                    continue
                e, l = time_windows[i]
                for t in range(e, l + 1):
                    cons = compute_consumption(r_idx, i, t)
                    if cons > 0:
                        terms.append((cons, x[i, t]))
            
            if terms:
                model.addConstr(
                    gp.quicksum(c * v for c, v in terms) <= cap,
                    name=f"res_{r_idx}"
                )
        
        # Objective: minimize s[end_node]
        model.setObjective(s[end_node], GRB.MINIMIZE)
        
        # Add initial solution using earliest start times (if feasible)
        # We'll try to provide a warm start
        # First check if EST is feasible for resources
        def check_resource_feasibility(starts):
            for r_idx, r in enumerate(resources):
                total = 0
                for i in range(n):
                    total += compute_consumption(r_idx, i, starts[i])
                if total > r['capacity']:
                    return False
            return True
        
        def check_temporal_feasibility(starts):
            for tc in temporal_constraints:
                fr, to, delta = tc['from'], tc['to'], tc['delta']
                if starts[to] < starts[fr] + delta:
                    return False
            return True
        
        # Try heuristic: schedule using forward pass with resource awareness
        # Simple serial SGS
        def serial_sgs(priority_list, target_makespan=None):
            """Serial Schedule Generation Scheme"""
            starts_sgs = [0] * n
            scheduled = [False] * n
            
            for act in priority_list:
                e, l = time_windows[act]
                if act == 0:
                    starts_sgs[0] = 0
                    scheduled[0] = True
                    continue
                
                # Earliest possible considering predecessors
                earliest = e
                for u, d in radj[act]:
                    if scheduled[u]:
                        earliest = max(earliest, starts_sgs[u] + d)
                
                # Find earliest feasible start
                best_t = None
                for t in range(earliest, l + 1):
                    starts_sgs[act] = t
                    # Check resource feasibility incrementally
                    feasible = True
                    for r_idx, r in enumerate(resources):
                        if r['demands'][act] == 0 or proc_times[act] == 0:
                            continue
                        cons_act = compute_consumption(r_idx, act, t)
                        if cons_act == 0:
                            continue
                        # Compute total so far
                        total = cons_act
                        for j in range(n):
                            if j != act and scheduled[j]:
                                total += compute_consumption(r_idx, j, starts_sgs[j])
                        if total > r['capacity']:
                            feasible = False
                            break
                    if feasible:
                        best_t = t
                        break
                
                if best_t is None:
                    return None
                starts_sgs[act] = best_t
                scheduled[act] = True
            
            return starts_sgs
        
        # Build topological order based on temporal constraints
        def topological_sort():
            in_degree = defaultdict(int)
            for i in range(n):
                if i not in in_degree:
                    in_degree[i] = 0
            for tc in temporal_constraints:
                if tc['delta'] >= 0:  # Only positive arcs define precedence
                    in_degree[tc['to']] += 1
            
            queue = deque()
            for i in range(n):
                if in_degree[i] == 0:
                    queue.append(i)
            
            order = []
            while queue:
                u = queue.popleft()
                order.append(u)
                for v, d in adj[u]:
                    if d >= 0:
                        in_degree[v] -= 1
                        if in_degree[v] == 0:
                            queue.append(v)
            
            # Add any remaining nodes
            remaining_nodes = set(range(n)) - set(order)
            order.extend(sorted(remaining_nodes, key=lambda x: est[x]))
            
            return order
        
        topo = topological_sort()
        
        heuristic_sol = serial_sgs(topo)
        
        if heuristic_sol is not None and check_temporal_feasibility(heuristic_sol) and check_resource_feasibility(heuristic_sol):
            # Set as warm start
            for i in range(n):
                t = heuristic_sol[i]
                if (i, t) in x:
                    x[i, t].Start = 1.0
            
            if logger:
                logger.log(heuristic_sol[end_node])
        
        # Callback to log incumbent solutions
        best_obj = [float('inf')]
        
        def callback(model, where):
            if where == GRB.Callback.MIPSOL:
                obj_val = int(round(model.cbGet(GRB.Callback.MIPSOL_OBJ)))
                if obj_val < best_obj[0]:
                    best_obj[0] = obj_val
                    if logger:
                        logger.log(obj_val)
        
        model.optimize(callback)
        
        if model.SolCount > 0:
            result_start = [0] * n
            for i in range(n):
                e, l = time_windows[i]
                for t in range(e, l + 1):
                    if x[i, t].X > 0.5:
                        result_start[i] = t
                        break
            
            obj = result_start[end_node]
            if logger:
                logger.log(obj)
            
            solution = {
                "objective_value": obj,
                "start_times": result_start
            }
        elif heuristic_sol is not None:
            solution = {
                "objective_value": heuristic_sol[end_node],
                "start_times": heuristic_sol
            }
        else:
            # Fallback
            solution = {
                "objective_value": est[end_node],
                "start_times": list(est)
            }
        
        with open(solution_path, 'w') as f:
            json.dump(solution, f)
    
    else:
        # Large instance: use heuristic approach
        # Serial SGS with priority rules + local search
        
        def check_resource_feasibility(starts):
            for r_idx, r in enumerate(resources):
                total = 0
                for i in range(n):
                    total += compute_consumption(r_idx, i, starts[i])
                if total > r['capacity']:
                    return False
            return True
        
        def check_temporal_feasibility(starts):
            for tc in temporal_constraints:
                fr, to, delta = tc['from'], tc['to'], tc['delta']
                if starts[to] < starts[fr] + delta:
                    return False
            return True
        
        # Forward pass to get earliest starts respecting temporal constraints
        def forward_pass(starts):
            changed = True
            iterations = 0
            while changed and iterations < n * 10:
                changed = False
                iterations += 1
                for tc in temporal_constraints:
                    fr, to, delta = tc['from'], tc['to'], tc['delta']
                    if starts[to] < starts[fr] + delta:
                        starts[to] = starts[fr] + delta
                        changed = True
            return starts
        
        # Build topological order
        def topological_sort_est():
            order = list(range(n))
            order.sort(key=lambda x: (est[x], x))
            return order
        
        def serial_sgs(priority_list):
            starts_sgs = [0] * n
            for i in range(n):
                starts_sgs[i] = max(est[i], 0)
            scheduled = [False] * n
            
            for act in priority_list:
                e, l = time_windows[act]
                if act == 0:
                    starts_sgs[0] = 0
                    scheduled[0] = True
                    continue
                
                earliest = e
                for u, d in radj[act]:
                    if scheduled[u]:
                        earliest = max(earliest, starts_sgs[u] + d)
                
                best_t = None
                for t in range(max(earliest, e), l + 1):
                    starts_sgs[act] = t
                    feasible = True
                    for r_idx, r in enumerate(resources):
                        if r['demands'][act] == 0 or proc_times[act] == 0:
                            continue
                        cons_act = compute_consumption(r_idx, act, t)
                        if cons_act == 0:
                            continue
                        total = cons_act
                        for j in range(n):
                            if j != act and scheduled[j]:
                                total += compute_consumption(r_idx, j, starts_sgs[j])
                        if total > r['capacity']:
                            feasible = False
                            break
                    if feasible:
                        best_t = t
                        break
                
                if best_t is None:
                    return None
                starts_sgs[act] = best_t
                scheduled[act] = True
            
            return starts_sgs
        
        topo = topological_sort_est()
        best_sol = serial_sgs(topo)
        
        if best_sol is not None:
            best_obj = best_sol[end_node]
            if logger:
                logger.log(best_obj)
        else:
            best_sol = list(est)
            best_sol[0] = 0
            best_obj = best_sol[end_node]
        
        # Try different priority rules
        import random
        random.seed(42)
        
        elapsed = time.time() - start_time
        while time.time() - start_time < time_limit - 1:
            # Random perturbation of priority
            prio = list(range(n))
            random.shuffle(prio)
            # Sort by est + random noise
            prio.sort(key=lambda x: (est[x] + random.randint(0, 3), x))
            
            sol = serial_sgs(prio)
            if sol is not None and sol[end_node] < best_obj:
                if check_temporal_feasibility(sol):
                    best_sol = sol
                    best_obj = sol[end_node]
                    if logger:
                        logger.log(best_obj)
        
        solution = {
            "objective_value": best_obj,
            "start_times": best_sol
        }
        
        with open(solution_path, 'w') as f:
            json.dump(solution, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)


if __name__ == '__main__':
    main()
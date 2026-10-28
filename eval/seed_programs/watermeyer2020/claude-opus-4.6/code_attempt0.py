import argparse
import json
import time
import sys
from collections import defaultdict

def solve_instance(instance, time_limit, logger=None):
    start_time = time.time()
    
    n = instance["n"]
    num_activities = instance["num_activities"]
    num_resources = instance["num_resources"]
    d_bar = instance["d_bar"]
    activities = instance["activities"]
    temporal_constraints = instance["temporal_constraints"]
    resources = instance["resources"]
    
    # Parse activity data
    proc_times = {}
    res_demands = {}
    for aid_str, adata in activities.items():
        aid = int(aid_str)
        proc_times[aid] = adata["processing_time"]
        res_demands[aid] = {}
        for rid_str, dem in adata["resource_demands"].items():
            res_demands[aid][rid_str] = dem
    
    # Parse resource data
    res_capacity = {}
    res_periods = {}
    for rid_str, rdata in resources.items():
        res_capacity[rid_str] = rdata["capacity"]
        res_periods[rid_str] = set(rdata["periods"])
    
    # Build temporal constraint graph
    # s[j] - s[i] >= delta
    constraints = []
    for tc in temporal_constraints:
        i = tc["i"]
        j = tc["j"]
        delta = tc["delta"]
        constraints.append((i, j, delta))
    
    # Compute earliest start times using longest path (Bellman-Ford style)
    # s[j] >= s[i] + delta for each constraint (i, j, delta)
    # s[0] = 0
    
    all_acts = sorted(proc_times.keys())
    project_start = 0
    project_end = n + 1
    
    est = {a: 0 for a in all_acts}
    est[project_start] = 0
    
    # Bellman-Ford for longest paths from source (activity 0)
    changed = True
    iterations = 0
    while changed and iterations < num_activities + 1:
        changed = False
        for (i, j, delta) in constraints:
            if est[i] + delta > est[j]:
                est[j] = est[i] + delta
                changed = True
        iterations += 1
    
    # Compute latest start times using reverse Bellman-Ford
    # s[j] - s[i] >= delta => s[i] <= s[j] - delta
    lst = {a: d_bar for a in all_acts}
    lst[project_start] = 0
    lst[project_end] = d_bar
    
    changed = True
    iterations = 0
    while changed and iterations < num_activities + 1:
        changed = False
        for (i, j, delta) in constraints:
            if lst[j] - delta < lst[i]:
                lst[i] = lst[j] - delta
                changed = True
        iterations += 1
    
    # Build adjacency lists for forward and backward propagation
    fwd_adj = defaultdict(list)  # i -> [(j, delta)]
    bwd_adj = defaultdict(list)  # j -> [(i, delta)]
    for (i, j, delta) in constraints:
        fwd_adj[i].append((j, delta))
        bwd_adj[j].append((i, delta))
    
    def compute_overlap(activity_id, start_t, resource_id):
        """Compute overlap count: number of periods in resource's period set
        that fall in (start_t, start_t + proc_time]"""
        pt = proc_times[activity_id]
        if pt == 0:
            return 0
        periods_set = res_periods[resource_id]
        count = 0
        for p in range(start_t + 1, start_t + pt + 1):
            if p in periods_set:
                count += 1
        return count
    
    # Precompute overlap counts for each activity, resource, and possible start time
    # overlap_table[aid][rid][st] = overlap count
    overlap_table = {}
    for aid in all_acts:
        pt = proc_times[aid]
        if pt == 0:
            continue
        overlap_table[aid] = {}
        for rid_str in resources:
            demand = res_demands.get(aid, {}).get(rid_str, 0)
            if demand == 0:
                continue
            periods_set = res_periods[rid_str]
            # Convert to sorted list for efficient computation
            sorted_periods = sorted(periods_set)
            # Precompute prefix sums for period membership in [1..d_bar]
            # period_present[t] = 1 if t in periods_set
            max_time = d_bar + max(proc_times.values()) + 1
            prefix = [0] * (max_time + 2)
            for p in sorted_periods:
                if 1 <= p <= max_time:
                    prefix[p] = 1
            for t in range(1, max_time + 1):
                prefix[t] += prefix[t-1]
            overlap_table[aid][rid_str] = (prefix, demand)
    
    def get_overlap_fast(aid, rid_str, st):
        if aid not in overlap_table or rid_str not in overlap_table[aid]:
            return 0
        prefix, demand = overlap_table[aid][rid_str]
        pt = proc_times[aid]
        # periods in (st, st+pt] = [st+1, st+pt]
        lo = st + 1
        hi = st + pt
        if hi < 1 or lo > len(prefix) - 1:
            return 0
        lo = max(lo, 1)
        hi = min(hi, len(prefix) - 1)
        count = prefix[hi] - prefix[lo - 1]
        return count * demand
    
    def get_resource_consumption(start_times):
        """Compute total resource consumption for each resource"""
        consumption = {}
        for rid_str in resources:
            total = 0
            for aid in all_acts:
                if proc_times[aid] == 0:
                    continue
                demand = res_demands.get(aid, {}).get(rid_str, 0)
                if demand == 0:
                    continue
                total += get_overlap_fast(aid, rid_str, start_times[aid])
                if total > res_capacity[rid_str]:
                    break
            consumption[rid_str] = total
        return consumption
    
    def check_feasibility(start_times):
        """Check if start_times satisfy all constraints"""
        # Check temporal constraints
        for (i, j, delta) in constraints:
            if start_times[j] - start_times[i] < delta:
                return False
        # Check resource constraints
        for rid_str in resources:
            total = 0
            for aid in all_acts:
                if proc_times[aid] == 0:
                    continue
                demand = res_demands.get(aid, {}).get(rid_str, 0)
                if demand == 0:
                    continue
                total += get_overlap_fast(aid, rid_str, start_times[aid])
            if total > res_capacity[rid_str]:
                return False
        return True
    
    def propagate_est(est_local):
        """Forward propagate earliest start times"""
        changed = True
        while changed:
            changed = False
            for (i, j, delta) in constraints:
                val = est_local[i] + delta
                if val > est_local[j]:
                    est_local[j] = val
                    changed = True
        return est_local
    
    def propagate_lst(lst_local, makespan):
        """Backward propagate latest start times given makespan"""
        lst_l = {a: makespan for a in all_acts}
        lst_l[project_start] = 0
        lst_l[project_end] = makespan
        # Also set upper bounds from est
        changed = True
        while changed:
            changed = False
            for (i, j, delta) in constraints:
                val = lst_l[j] - delta
                if val < lst_l[i]:
                    lst_l[i] = val
                    changed = True
        return lst_l
    
    # Try to use Gurobi MIP
    best_solution = None
    best_obj = d_bar
    
    # First try a simple greedy schedule based on EST
    greedy_times = dict(est)
    greedy_times[project_start] = 0
    if check_feasibility(greedy_times):
        best_solution = dict(greedy_times)
        best_obj = greedy_times[project_end]
        if logger:
            logger.log(best_obj)
    
    # Try MIP with Gurobi
    try:
        import gurobipy as gp
        from gurobipy import GRB
        
        elapsed = time.time() - start_time
        remaining = time_limit - elapsed - 2  # reserve 2 seconds
        
        if remaining > 1:
            model = gp.Model("RCPSP_PR")
            model.setParam("TimeLimit", remaining)
            model.setParam("Threads", 1)
            model.setParam("OutputFlag", 0)
            
            # Decision variables: start time for each activity
            s = {}
            for aid in all_acts:
                lb = max(0, est[aid])
                ub = min(d_bar, lst[aid])
                if ub < lb:
                    ub = d_bar
                s[aid] = model.addVar(vtype=GRB.INTEGER, lb=lb, ub=ub, name=f"s_{aid}")
            
            model.update()
            
            # Fix project start
            model.addConstr(s[project_start] == 0)
            
            # Objective: minimize project end start time
            model.setObjective(s[project_end], GRB.MINIMIZE)
            
            # Temporal constraints
            for (i, j, delta) in constraints:
                model.addConstr(s[j] - s[i] >= delta)
            
            # Resource constraints using binary variables for time assignment
            # For each activity with nonzero processing time, we need to determine
            # which time slot it starts at.
            
            # Check problem size to decide approach
            real_acts = [a for a in all_acts if proc_times[a] > 0]
            
            # For partially renewable resources, the constraint is:
            # sum over all activities a: demand[a][r] * |{t in periods[r] : t in (s[a], s[a]+p[a]]}| <= capacity[r]
            
            # We need binary variables x[a][t] = 1 if activity a starts at time t
            # Then overlap = sum_{t in periods[r] intersect (st, st+p[a]]} 1
            # = sum_{tau} x[a][tau] * |{t in periods[r] : tau < t <= tau + p[a]}|
            
            # Precompute for each activity, resource: overlap_coeff[a][r][tau] = overlap count if a starts at tau
            
            # Create binary start time variables
            x = {}
            total_binary = 0
            
            # Determine if we need binary variables at all
            has_resource_constraints = False
            for rid_str in resources:
                cap = res_capacity[rid_str]
                periods = res_periods[rid_str]
                if len(periods) == 0:
                    continue
                # Check if any activity has demand on this resource
                for aid in real_acts:
                    if res_demands.get(aid, {}).get(rid_str, 0) > 0:
                        has_resource_constraints = True
                        break
                if has_resource_constraints:
                    break
            
            if has_resource_constraints:
                # Create binary variables for start times
                for aid in real_acts:
                    lb_a = max(0, est[aid])
                    ub_a = min(d_bar, lst[aid])
                    if ub_a < lb_a:
                        ub_a = d_bar
                    total_binary += (ub_a - lb_a + 1)
                
                # If too many binary variables, try a tighter formulation or heuristic
                max_binary = 200000
                use_mip = total_binary <= max_binary
                
                if use_mip:
                    for aid in real_acts:
                        lb_a = max(0, est[aid])
                        ub_a = min(d_bar, lst[aid])
                        if ub_a < lb_a:
                            ub_a = d_bar
                        x[aid] = {}
                        for t in range(lb_a, ub_a + 1):
                            x[aid][t] = model.addVar(vtype=GRB.BINARY, name=f"x_{aid}_{t}")
                    
                    model.update()
                    
                    # Link s[a] = sum_t t * x[a][t]
                    for aid in real_acts:
                        model.addConstr(
                            s[aid] == gp.quicksum(t * x[aid][t] for t in x[aid]),
                            name=f"link_{aid}"
                        )
                        # Exactly one start time
                        model.addConstr(
                            gp.quicksum(x[aid][t] for t in x[aid]) == 1,
                            name=f"one_{aid}"
                        )
                    
                    # Resource constraints
                    for rid_str in resources:
                        cap = res_capacity[rid_str]
                        periods_set = res_periods[rid_str]
                        if len(periods_set) == 0:
                            continue
                        
                        # Build constraint: sum_a sum_t x[a][t] * overlap(a, r, t) * demand(a,r) <= cap
                        terms = []
                        for aid in real_acts:
                            demand = res_demands.get(aid, {}).get(rid_str, 0)
                            if demand == 0:
                                continue
                            for t in x[aid]:
                                ov = get_overlap_fast(aid, rid_str, t)
                                if ov > 0:
                                    terms.append((ov, x[aid][t]))
                        
                        if terms:
                            model.addConstr(
                                gp.quicksum(coeff * var for coeff, var in terms) <= cap,
                                name=f"res_{rid_str}"
                            )
                    
                    # Warm start with best solution
                    if best_solution:
                        for aid in all_acts:
                            s[aid].Start = best_solution[aid]
                        for aid in real_acts:
                            for t in x[aid]:
                                if t == best_solution[aid]:
                                    x[aid][t].Start = 1
                                else:
                                    x[aid][t].Start = 0
                    
                    # Callback for logging
                    incumbent_vals = [best_obj if best_solution else d_bar + 1]
                    
                    def callback(model, where):
                        if where == GRB.Callback.MIPSOL:
                            obj_val = int(round(model.cbGetSolution(s[project_end])))
                            if obj_val < incumbent_vals[0]:
                                incumbent_vals[0] = obj_val
                                if logger:
                                    logger.log(obj_val)
                    
                    model.optimize(callback)
                    
                    if model.SolCount > 0:
                        sol_times = {}
                        for aid in all_acts:
                            sol_times[aid] = int(round(s[aid].X))
                        obj_val = sol_times[project_end]
                        if obj_val < best_obj:
                            best_obj = obj_val
                            best_solution = sol_times
                            if logger:
                                logger.log(best_obj)
                else:
                    # Too many binary variables - use a different approach
                    # Try constraint programming style with iterative solving
                    # or a priority-rule based heuristic with local search
                    pass
            else:
                # No resource constraints - just solve the temporal constraint problem
                model.optimize()
                if model.SolCount > 0:
                    sol_times = {}
                    for aid in all_acts:
                        sol_times[aid] = int(round(s[aid].X))
                    obj_val = sol_times[project_end]
                    if obj_val < best_obj:
                        best_obj = obj_val
                        best_solution = sol_times
                        if logger:
                            logger.log(best_obj)
    
    except ImportError:
        pass
    except Exception as e:
        pass
    
    # If no MIP solution, try heuristic approaches
    if best_solution is None:
        # Use EST as fallback
        best_solution = dict(est)
        best_solution[project_start] = 0
        best_obj = best_solution[project_end]
        if logger:
            logger.log(best_obj)
    
    # Local search improvement
    elapsed = time.time() - start_time
    remaining = time_limit - elapsed - 1
    
    if remaining > 0.5 and best_solution is not None:
        current = dict(best_solution)
        current_obj = best_obj
        
        # Try shifting activities earlier
        real_acts_sorted = sorted([a for a in all_acts if proc_times[a] > 0], 
                                   key=lambda a: current[a])
        
        improved = True
        iter_count = 0
        while improved and time.time() - start_time < time_limit - 0.5:
            improved = False
            iter_count += 1
            
            for aid in real_acts_sorted:
                if time.time() - start_time >= time_limit - 0.5:
                    break
                    
                # Try to shift activity earlier
                current_st = current[aid]
                
                # Find minimum feasible start time based on temporal constraints
                min_st = est[aid]
                for (i, j, delta) in constraints:
                    if j == aid:
                        min_st = max(min_st, current[i] + delta)
                    if i == aid:
                        # s[j] - s[aid] >= delta => s[aid] <= s[j] - delta
                        max_from_this = current[j] - delta
                        # This gives an upper bound, not a lower bound
                
                if min_st >= current_st:
                    continue
                
                # Try each time from min_st to current_st - 1
                for t in range(min_st, current_st):
                    old_st = current[aid]
                    current[aid] = t
                    
                    # Check temporal feasibility
                    feasible = True
                    for (i, j, delta) in constraints:
                        if i == aid:
                            if current[j] - current[aid] < delta:
                                feasible = False
                                break
                        if j == aid:
                            if current[aid] - current[i] < delta:
                                feasible = False
                                break
                    
                    if not feasible:
                        current[aid] = old_st
                        continue
                    
                    # Check resource feasibility
                    res_ok = True
                    for rid_str in resources:
                        demand = res_demands.get(aid, {}).get(rid_str, 0)
                        if demand == 0:
                            continue
                        # Recompute total for this resource
                        total = 0
                        for a2 in all_acts:
                            if proc_times[a2] == 0:
                                continue
                            d2 = res_demands.get(a2, {}).get(rid_str, 0)
                            if d2 == 0:
                                continue
                            total += get_overlap_fast(a2, rid_str, current[a2])
                        if total > res_capacity[rid_str]:
                            res_ok = False
                            break
                    
                    if res_ok:
                        improved = True
                        break
                    else:
                        current[aid] = old_st
            
            # Update project end
            # Recompute project end from temporal constraints
            new_end = 0
            for (i, j, delta) in constraints:
                if j == project_end:
                    new_end = max(new_end, current[i] + delta)
            current[project_end] = new_end
            
            if current[project_end] < best_obj:
                best_obj = current[project_end]
                best_solution = dict(current)
                if logger:
                    logger.log(best_obj)
    
    return best_solution, best_obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, default=300)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()
    
    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    with open(args.instance_path, "r") as f:
        instance = json.load(f)
    
    solution, obj_val = solve_instance(instance, args.time_limit, logger)
    
    # Format output
    output = {
        "objective_value": obj_val,
        "start_times": {str(k): v for k, v in solution.items()}
    }
    
    with open(args.solution_path, "w") as f:
        json.dump(output, f)


if __name__ == "__main__":
    main()
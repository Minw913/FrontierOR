import json
import argparse
import time
import random
from solution_logger import SolutionLogger


def ssgs_schedule(jobs, facility, proc_times, cap_req, release_times, capacity_limit, priority_key):
    """Serial Schedule Generation Scheme with given priority."""
    if not jobs:
        return {}
    
    priority_list = sorted(jobs, key=priority_key)
    starts = {}
    scheduled = []  # list of (start, end, resource)
    
    for j in priority_list:
        p = proc_times[facility][j]
        c = cap_req[facility][j]
        r = release_times[j]
        
        if p == 0:
            starts[j] = r
            continue
        
        # Collect candidate start times
        candidates = {r}
        for (s, e, res) in scheduled:
            if e >= r:
                candidates.add(e)
        candidates = sorted(candidates)
        
        best_start = None
        for t in candidates:
            if t < r:
                continue
            feasible = True
            for tau in range(t, t + p):
                usage = c
                for (s, e, res) in scheduled:
                    if s <= tau < e:
                        usage += res
                if usage > capacity_limit:
                    feasible = False
                    break
            if feasible:
                best_start = t
                break
        
        if best_start is None:
            best_start = max((e for (s, e, res) in scheduled), default=r)
            # Verify and extend if needed
            while True:
                feasible = True
                for tau in range(best_start, best_start + p):
                    usage = c
                    for (s, e, res) in scheduled:
                        if s <= tau < e:
                            usage += res
                    if usage > capacity_limit:
                        feasible = False
                        break
                if feasible:
                    break
                best_start += 1
        
        starts[j] = best_start
        scheduled.append((best_start, best_start + p, c))
    
    return starts


def schedule_facility_best(jobs_on_f, f, proc, cap_req, release_times, capacity_limit):
    """Try multiple priority rules and return the best schedule."""
    if not jobs_on_f:
        return {}, 0
    
    best_starts = None
    best_ms = float('inf')
    
    priority_keys = [
        lambda j: (release_times[j], -proc[f][j]),
        lambda j: (-proc[f][j], release_times[j]),
        lambda j: (-cap_req[f][j], release_times[j]),
        lambda j: (release_times[j], -cap_req[f][j]),
        lambda j: (-proc[f][j] * cap_req[f][j], release_times[j]),
    ]
    
    for pk in priority_keys:
        starts = ssgs_schedule(jobs_on_f, f, proc, cap_req, release_times, capacity_limit, pk)
        if starts:
            ms = max(starts[j] + proc[f][j] for j in jobs_on_f)
        else:
            ms = 0
        if ms < best_ms:
            best_ms = ms
            best_starts = starts
    
    return best_starts, best_ms


def compute_expected_makespan(assignment, n, m, num_scenarios, scenarios, cap_req, release_times, capacity):
    """Compute expected makespan and schedules for a given assignment."""
    total = 0.0
    all_schedules = {}
    
    for sc_idx in range(num_scenarios):
        proc = scenarios[sc_idx]['processing_times']
        prob = scenarios[sc_idx]['probability']
        scenario_schedule = {}
        scenario_makespan = 0
        
        for f in range(m):
            jobs_on_f = [j for j in range(n) if assignment[j] == f]
            if not jobs_on_f:
                continue
            
            starts, ms = schedule_facility_best(jobs_on_f, f, proc, cap_req, release_times, capacity[f])
            
            for j in jobs_on_f:
                scenario_schedule[j] = {
                    'facility': f,
                    'start': starts[j],
                    'finish': starts[j] + proc[f][j]
                }
            scenario_makespan = max(scenario_makespan, ms)
        
        total += prob * scenario_makespan
        all_schedules[sc_idx] = scenario_schedule
    
    return total, all_schedules


def heuristic_assignment(n, m, num_scenarios, scenarios, cap_req, release_times, capacity):
    """Generate heuristic assignments and return the best."""
    feasible_facilities = {}
    for j in range(n):
        feasible_facilities[j] = [f for f in range(m) if cap_req[f][j] <= capacity[f]]
        if not feasible_facilities[j]:
            feasible_facilities[j] = [0]  # fallback
    
    strategies = []
    
    # Strategy 1: Min expected processing time
    a1 = {}
    for j in range(n):
        best_f, best_val = feasible_facilities[j][0], float('inf')
        for f in feasible_facilities[j]:
            ep = sum(scenarios[sc]['probability'] * scenarios[sc]['processing_times'][f][j] for sc in range(num_scenarios))
            if ep < best_val:
                best_val, best_f = ep, f
        a1[j] = best_f
    strategies.append(a1)
    
    # Strategy 2: Min resource consumption
    a2 = {}
    for j in range(n):
        best_f, best_val = feasible_facilities[j][0], float('inf')
        for f in feasible_facilities[j]:
            if cap_req[f][j] < best_val:
                best_val, best_f = cap_req[f][j], f
        a2[j] = best_f
    strategies.append(a2)
    
    # Strategy 3: Round-robin
    a3 = {j: j % m for j in range(n)}
    strategies.append(a3)
    
    # Strategy 4: Balance load (min expected_p * cap_req / capacity)
    a4 = {}
    for j in range(n):
        best_f, best_val = feasible_facilities[j][0], float('inf')
        for f in feasible_facilities[j]:
            ep = sum(scenarios[sc]['probability'] * scenarios[sc]['processing_times'][f][j] for sc in range(num_scenarios))
            val = ep * cap_req[f][j] / max(capacity[f], 1)
            if val < best_val:
                best_val, best_f = val, f
        a4[j] = best_f
    strategies.append(a4)
    
    # Strategy 5: Min expected processing time * resource
    a5 = {}
    for j in range(n):
        best_f, best_val = feasible_facilities[j][0], float('inf')
        for f in feasible_facilities[j]:
            ep = sum(scenarios[sc]['probability'] * scenarios[sc]['processing_times'][f][j] for sc in range(num_scenarios))
            val = ep * cap_req[f][j]
            if val < best_val:
                best_val, best_f = val, f
        a5[j] = best_f
    strategies.append(a5)
    
    # Random strategies
    random.seed(42)
    for _ in range(30):
        ar = {j: random.choice(feasible_facilities[j]) for j in range(n)}
        strategies.append(ar)
    
    best_assignment, best_obj, best_schedules = None, float('inf'), None
    
    for assignment in strategies:
        obj, schedules = compute_expected_makespan(assignment, n, m, num_scenarios, scenarios, cap_req, release_times, capacity)
        if obj < best_obj:
            best_obj = obj
            best_assignment = assignment.copy()
            best_schedules = schedules
    
    return best_assignment, best_obj, best_schedules


def local_search(assignment, n, m, num_scenarios, scenarios, cap_req, release_times, capacity, time_limit, start_time):
    """Improve assignment using local search (move and swap neighborhoods)."""
    current_assignment = assignment.copy()
    current_obj, current_schedules = compute_expected_makespan(
        current_assignment, n, m, num_scenarios, scenarios, cap_req, release_times, capacity)
    
    improved = True
    iteration = 0
    max_iterations = 100
    
    while improved and iteration < max_iterations and time.time() - start_time < time_limit * 0.4:
        improved = False
        iteration += 1
        
        # Move neighborhood
        for j in range(n):
            if time.time() - start_time >= time_limit * 0.4:
                break
            current_f = current_assignment[j]
            for f in range(m):
                if f == current_f:
                    continue
                if cap_req[f][j] > capacity[f]:
                    continue
                current_assignment[j] = f
                obj, schedules = compute_expected_makespan(
                    current_assignment, n, m, num_scenarios, scenarios, cap_req, release_times, capacity)
                if obj < current_obj - 1e-10:
                    current_obj = obj
                    current_schedules = schedules
                    improved = True
                else:
                    current_assignment[j] = current_f
        
        # Swap neighborhood
        if time.time() - start_time < time_limit * 0.35:
            for j1 in range(n):
                if time.time() - start_time >= time_limit * 0.35:
                    break
                for j2 in range(j1 + 1, n):
                    f1 = current_assignment[j1]
                    f2 = current_assignment[j2]
                    if f1 == f2:
                        continue
                    if cap_req[f2][j1] > capacity[f2] or cap_req[f1][j2] > capacity[f1]:
                        continue
                    current_assignment[j1] = f2
                    current_assignment[j2] = f1
                    obj, schedules = compute_expected_makespan(
                        current_assignment, n, m, num_scenarios, scenarios, cap_req, release_times, capacity)
                    if obj < current_obj - 1e-10:
                        current_obj = obj
                        current_schedules = schedules
                        improved = True
                    else:
                        current_assignment[j1] = f1
                        current_assignment[j2] = f2
    
    return current_assignment, current_obj, current_schedules


def build_solution_dict(assignment, obj, schedules, n, num_scenarios):
    """Build the solution dictionary in the required output format."""
    solution = {
        "objective_value": float(obj),
        "assignment": {str(j): int(assignment[j]) for j in range(n)},
        "schedule": {}
    }
    
    for sc_idx in range(num_scenarios):
        solution["schedule"][str(sc_idx)] = {}
        for j in range(n):
            entry = schedules[sc_idx][j]
            solution["schedule"][str(sc_idx)][str(j)] = {
                "facility": int(entry['facility']),
                "start": int(entry['start']),
                "finish": int(entry['finish'])
            }
    
    return solution


def solve_mip(n, m, num_scenarios, scenarios, cap_req, release_times, capacity, time_limit, warm_assignment=None):
    """Solve using time-indexed MIP formulation with Gurobi."""
    try:
        from gurobipy import Model, GRB, quicksum
    except ImportError:
        return None
    
    # Compute T_max
    T_max = max(release_times) if release_times else 0
    for sc_idx in range(num_scenarios):
        proc = scenarios[sc_idx]['processing_times']
        for f in range(m):
            total_p = sum(proc[f][j] for j in range(n))
            T_max = max(T_max, (max(release_times) if release_times else 0) + total_p)
    
    # Add some buffer
    T_max = min(T_max, (max(release_times) if release_times else 0) + sum(
        max(scenarios[sc]['processing_times'][f][j] for sc in range(num_scenarios) for f in range(m))
        for j in range(n)
    ))
    
    # Check problem size
    num_y_vars = 0
    for j in range(n):
        for f in range(m):
            if cap_req[f][j] <= capacity[f]:
                num_y_vars += num_scenarios * max(0, T_max - release_times[j] + 1)
    
    if num_y_vars > 300000:
        return None
    
    model = Model()
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', max(1, time_limit))
    model.setParam('OutputFlag', 0)
    
    # Feasible (j, f) pairs
    feasible = set()
    for j in range(n):
        for f in range(m):
            if cap_req[f][j] <= capacity[f]:
                feasible.add((j, f))
    
    # Variables
    x = {}
    for (j, f) in feasible:
        x[j, f] = model.addVar(vtype=GRB.BINARY, name=f"x_{j}_{f}")
    
    y = {}
    for (j, f) in feasible:
        for sc_idx in range(num_scenarios):
            for t in range(release_times[j], T_max + 1):
                y[j, f, sc_idx, t] = model.addVar(vtype=GRB.BINARY, name=f"y_{j}_{f}_{sc_idx}_{t}")
    
    M_var = {}
    for sc_idx in range(num_scenarios):
        M_var[sc_idx] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"M_{sc_idx}")
    
    model.update()
    
    # Objective
    obj = quicksum(scenarios[sc_idx]['probability'] * M_var[sc_idx] for sc_idx in range(num_scenarios))
    model.setObjective(obj, GRB.MINIMIZE)
    
    # Constraint 1: Each job assigned to exactly one facility
    for j in range(n):
        model.addConstr(
            quicksum(x[j, f] for f in range(m) if (j, f) in feasible) == 1,
            name=f"assign_{j}"
        )
    
    # Constraint 2: Each job starts exactly once per scenario
    for j in range(n):
        for sc_idx in range(num_scenarios):
            model.addConstr(
                quicksum(y[j, f, sc_idx, t]
                         for f in range(m) if (j, f) in feasible
                         for t in range(release_times[j], T_max + 1)) == 1,
                name=f"start_{j}_{sc_idx}"
            )
    
    # Constraint 3: Linking
    for (j, f) in feasible:
        for sc_idx in range(num_scenarios):
            model.addConstr(
                quicksum(y[j, f, sc_idx, t] for t in range(release_times[j], T_max + 1)) <= x[j, f],
                name=f"link_{j}_{f}_{sc_idx}"
            )
    
    # Constraint 4: Capacity
    for f in range(m):
        for sc_idx in range(num_scenarios):
            proc = scenarios[sc_idx]['processing_times']
            for t in range(T_max + 1):
                terms = []
                for j in range(n):
                    if (j, f) not in feasible:
                        continue
                    p = proc[f][j]
                    if p == 0:
                        continue
                    tau_lo = max(release_times[j], t - p + 1)
                    tau_hi = t
                    if tau_lo <= tau_hi:
                        terms.append(cap_req[f][j] * quicksum(
                            y[j, f, sc_idx, tau] for tau in range(tau_lo, tau_hi + 1)))
                if terms:
                    model.addConstr(quicksum(terms) <= capacity[f], name=f"cap_{f}_{sc_idx}_{t}")
    
    # Constraint 5: Makespan
    for sc_idx in range(num_scenarios):
        proc = scenarios[sc_idx]['processing_times']
        for j in range(n):
            finish_expr = quicksum(
                (t + proc[f][j]) * y[j, f, sc_idx, t]
                for f in range(m) if (j, f) in feasible
                for t in range(release_times[j], T_max + 1)
            )
            model.addConstr(M_var[sc_idx] >= finish_expr, name=f"mk_{j}_{sc_idx}")
    
    # Valid inequalities: M[sc] >= r[j] + min_f p[f,j,sc]
    for sc_idx in range(num_scenarios):
        proc = scenarios[sc_idx]['processing_times']
        for j in range(n):
            min_p = min(proc[f][j] for f in range(m) if (j, f) in feasible)
            model.addConstr(M_var[sc_idx] >= release_times[j] + min_p)
    
    # Warm start
    if warm_assignment is not None:
        for (j, f) in feasible:
            x[j, f].Start = 1.0 if warm_assignment[j] == f else 0.0
    
    model.optimize()
    
    if model.SolCount > 0:
        try:
            # Extract assignment
            assignment = {}
            for j in range(n):
                for f in range(m):
                    if (j, f) in feasible and x[j, f].X > 0.5:
                        assignment[j] = f
                        break
                if j not in assignment:
                    # Fallback
                    for f in range(m):
                        if (j, f) in feasible:
                            assignment[j] = f
                            break
            
            mip_obj = model.ObjVal
            
            # Recompute schedules with the MIP assignment (to ensure consistency)
            _, schedules = compute_expected_makespan(assignment, n, m, num_scenarios, scenarios, cap_req, release_times, capacity)
            
            # Also compute actual expected makespan from schedules
            actual_obj = 0.0
            for sc_idx in range(num_scenarios):
                sc_makespan = 0
                for j in range(n):
                    sc_makespan = max(sc_makespan, schedules[sc_idx][j]['finish'])
                actual_obj += scenarios[sc_idx]['probability'] * sc_makespan
            
            return actual_obj, assignment, schedules
        except Exception:
            return None
    
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    total_start = time.time()
    
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    n = instance['n']
    m = instance['m']
    num_scenarios = instance['num_scenarios']
    capacity = instance['facility_capacity_limits']
    cap_req = instance['capacity_requirements']
    release_times = instance['release_times']
    scenarios = instance['scenarios']
    
    # Handle trivial case
    if n == 0:
        solution = {
            "objective_value": 0.0,
            "assignment": {},
            "schedule": {str(sc): {} for sc in range(num_scenarios)}
        }
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f)
        if logger:
            logger.log_solution(0.0, solution)
        return
    
    # Phase 1: Heuristic
    best_assignment, best_obj, best_schedules = heuristic_assignment(
        n, m, num_scenarios, scenarios, cap_req, release_times, capacity)
    
    solution_dict = build_solution_dict(best_assignment, best_obj, best_schedules, n, num_scenarios)
    if logger:
        logger.log_solution(best_obj, solution_dict)
    
    # Phase 2: Local search
    elapsed = time.time() - total_start
    if elapsed < args.time_limit * 0.4:
        improved_assignment, improved_obj, improved_schedules = local_search(
            best_assignment, n, m, num_scenarios, scenarios, cap_req, release_times, capacity,
            args.time_limit, total_start
        )
        if improved_obj < best_obj - 1e-10:
            best_assignment = improved_assignment
            best_obj = improved_obj
            best_schedules = improved_schedules
            solution_dict = build_solution_dict(best_assignment, best_obj, best_schedules, n, num_scenarios)
            if logger:
                logger.log_solution(best_obj, solution_dict)
    
    # Phase 3: MIP
    elapsed = time.time() - total_start
    remaining = args.time_limit - elapsed
    if remaining > 20:
        mip_result = solve_mip(
            n, m, num_scenarios, scenarios, cap_req, release_times, capacity,
            remaining - 5, best_assignment
        )
        if mip_result is not None:
            mip_obj, mip_assignment, mip_schedules = mip_result
            if mip_obj < best_obj - 1e-10:
                best_assignment = mip_assignment
                best_obj = mip_obj
                best_schedules = mip_schedules
                solution_dict = build_solution_dict(best_assignment, best_obj, best_schedules, n, num_scenarios)
                if logger:
                    logger.log_solution(best_obj, solution_dict)
    
    # Write final solution
    solution_dict = build_solution_dict(best_assignment, best_obj, best_schedules, n, num_scenarios)
    with open(args.solution_path, 'w') as f:
        json.dump(solution_dict, f, indent=2)


if __name__ == '__main__':
    main()
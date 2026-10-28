import argparse
import json
import time
import math
import numpy as np
from itertools import product as iter_product

def solve(instance_path, solution_path, time_limit, log_path):
    from solution_logger import SolutionLogger
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    
    params = instance['parameters']
    T = params['T_planning_horizon_minutes']
    D = params['D_num_train_services']
    NT = params['NT_num_demand_periods']
    theta = params['theta_period_length_minutes']
    h_min = params['h_min_minutes']
    h_max = params['h_max_minutes']
    C = params['C_train_capacity_passengers']
    
    demand_periods = instance['demand_periods']
    lambdas = [dp['lambda_pax_per_min'] for dp in sorted(demand_periods, key=lambda x: x['period'])]
    
    start_time = time.time()
    
    # Period start times
    period_starts = [i * theta for i in range(NT)]
    period_ends = [(i + 1) * theta for i in range(NT)]
    
    # Cumulative arrivals function: given time t, compute cumulative passengers from time 0 to t
    def cumulative_arrivals(t):
        """Compute cumulative passenger arrivals from time 0 to time t."""
        if t <= 0:
            return 0.0
        total = 0.0
        for i in range(NT):
            ps = period_starts[i]
            pe = period_ends[i]
            if t <= ps:
                break
            elapsed = min(t, pe) - ps
            total += elapsed * lambdas[i]
        return total
    
    # Arrivals between t1 and t2
    def arrivals_between(t1, t2):
        return cumulative_arrivals(t2) - cumulative_arrivals(t1)
    
    # Compute objective given departure times
    def compute_objective(deps):
        """Compute total waiting time given sorted departure times."""
        n = len(deps)
        if n == 0:
            # No trains: all passengers wait
            arr_total = arrivals_between(0, T)
            return T * (0 + arr_total / 2.0)
        
        # Build intervals: [0, deps[0]], [deps[0], deps[1]], ..., [deps[-1], T]
        boundaries = [0.0] + list(deps) + [T]
        
        waiting = 0.0  # passengers waiting at start of interval
        total_wt = 0.0
        
        for i in range(len(boundaries) - 1):
            t_start = boundaries[i]
            t_end = boundaries[i + 1]
            h = t_end - t_start
            if h < -1e-9:
                return float('inf')
            if h < 1e-12:
                continue
            new_arr = arrivals_between(t_start, t_end)
            # Platform occupancy estimate: waiting + new_arr/2
            total_wt += h * (waiting + new_arr / 2.0)
            
            if i < len(boundaries) - 2:
                # A train departs at t_end (boundaries[i+1])
                # Passengers who can board = min(C, waiting + new_arr)
                total_on_platform = waiting + new_arr
                boarding = min(C, total_on_platform)
                waiting = total_on_platform - boarding
            else:
                # Final interval, no train departs at T
                pass
        
        return total_wt
    
    best_obj = float('inf')
    best_deps = None
    
    def update_best(deps):
        nonlocal best_obj, best_deps
        obj = compute_objective(deps)
        if obj < best_obj:
            best_obj = obj
            best_deps = list(deps)
            if logger:
                logger.log(best_obj)
        return obj
    
    # ---- Heuristic 1: Equal spacing ----
    if D > 0:
        total_span = T
        spacing = total_span / (D + 1)
        deps_equal = [spacing * (i + 1) for i in range(D)]
        # Clamp to valid range
        deps_equal = [max(0, min(T, d)) for d in deps_equal]
        update_best(deps_equal)
    
    # ---- Heuristic 2: Spacing proportional to demand ----
    if D > 0:
        # Place trains so that equal amounts of passengers are between consecutive trains
        total_demand = cumulative_arrivals(T)
        if total_demand > 0:
            target_per_interval = total_demand / (D + 1)
            deps_demand = []
            for k in range(1, D + 1):
                target = target_per_interval * k
                # Binary search for time
                lo, hi = 0.0, T
                for _ in range(100):
                    mid = (lo + hi) / 2
                    if cumulative_arrivals(mid) < target:
                        lo = mid
                    else:
                        hi = mid
                deps_demand.append((lo + hi) / 2)
            
            # Enforce headway constraints
            deps_demand = enforce_headway_constraints(deps_demand, h_min, h_max, T)
            if deps_demand is not None:
                update_best(deps_demand)
    
    # ---- Heuristic 3: Square root of demand proportional ----
    if D > 0 and total_demand > 0:
        # Weight by sqrt of lambda for each period
        sqrt_weights = [math.sqrt(max(l, 0.001)) for l in lambdas]
        total_sw = sum(sw * theta for sw in sqrt_weights)
        if total_sw > 0:
            deps_sqrt = []
            target_per_interval = total_sw / (D + 1)
            for k in range(1, D + 1):
                target = target_per_interval * k
                cum = 0.0
                t = 0.0
                for i in range(NT):
                    add = sqrt_weights[i] * theta
                    if cum + add >= target:
                        remaining = target - cum
                        t = period_starts[i] + remaining / sqrt_weights[i]
                        break
                    cum += add
                    t = period_ends[i]
                deps_sqrt.append(min(t, T))
            
            deps_sqrt = enforce_headway_constraints(deps_sqrt, h_min, h_max, T)
            if deps_sqrt is not None:
                update_best(deps_sqrt)
    
    # ---- Local search / continuous optimization ----
    if best_deps is not None and D > 0:
        deps = list(best_deps)
        deps = local_search(deps, h_min, h_max, T, C, NT, theta, lambdas, period_starts, period_ends,
                           cumulative_arrivals, arrivals_between, compute_objective, update_best, 
                           start_time, time_limit)
    
    # ---- Try Gurobi MIP for small instances ----
    if D > 0 and D * NT <= 5000 and time.time() - start_time < time_limit * 0.5:
        try:
            gurobi_deps = solve_with_gurobi(T, D, NT, theta, h_min, h_max, C, lambdas, period_starts, period_ends,
                                            cumulative_arrivals, arrivals_between, compute_objective,
                                            start_time, time_limit, best_deps, best_obj)
            if gurobi_deps is not None:
                update_best(gurobi_deps)
        except Exception as e:
            pass
    
    # ---- More local search with the best solution ----
    if best_deps is not None and D > 0 and time.time() - start_time < time_limit - 2:
        deps = list(best_deps)
        deps = local_search_fine(deps, h_min, h_max, T, C, NT, theta, lambdas, period_starts, period_ends,
                                cumulative_arrivals, arrivals_between, compute_objective, update_best,
                                start_time, time_limit)
    
    # Prepare output
    if best_deps is None:
        best_deps = []
        best_obj = compute_objective([])
    
    solution = {
        "objective_value": best_obj,
        "departure_schedule_minutes": {}
    }
    for i, dep in enumerate(best_deps):
        solution["departure_schedule_minutes"][str(i + 1)] = round(dep, 6)
    
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


def enforce_headway_constraints(deps, h_min, h_max, T):
    """Try to adjust departure times to satisfy headway constraints."""
    D = len(deps)
    if D <= 1:
        return deps
    
    deps = sorted(deps)
    # Clamp to [0, T]
    deps = [max(0.0, min(T, d)) for d in deps]
    
    # Forward pass: ensure minimum headway
    for i in range(1, D):
        if deps[i] < deps[i-1] + h_min:
            deps[i] = deps[i-1] + h_min
    
    # Check if last departure exceeds T
    if deps[-1] > T:
        # Backward pass
        deps[-1] = T
        for i in range(D-2, -1, -1):
            if deps[i] > deps[i+1] - h_min:
                deps[i] = deps[i+1] - h_min
        if deps[0] < 0:
            return None  # Infeasible
    
    # Check maximum headway
    for i in range(1, D):
        if deps[i] - deps[i-1] > h_max:
            # Try to fix by adjusting
            deps[i] = deps[i-1] + h_max
    
    # Re-enforce min headway forward
    for i in range(1, D):
        if deps[i] < deps[i-1] + h_min:
            deps[i] = deps[i-1] + h_min
    
    if deps[-1] > T:
        return None
    
    # Verify
    for i in range(1, D):
        hw = deps[i] - deps[i-1]
        if hw < h_min - 1e-6 or hw > h_max + 1e-6:
            return None
    
    return deps


def local_search(deps, h_min, h_max, T, C, NT, theta, lambdas, period_starts, period_ends,
                 cumulative_arrivals, arrivals_between, compute_objective, update_best,
                 start_time, time_limit):
    """Local search by perturbing individual departure times."""
    D = len(deps)
    if D == 0:
        return deps
    
    current_obj = compute_objective(deps)
    improved = True
    iteration = 0
    
    while improved and time.time() - start_time < time_limit - 5:
        improved = False
        iteration += 1
        
        for i in range(D):
            if time.time() - start_time > time_limit - 5:
                break
            
            # Determine bounds for deps[i]
            if i == 0:
                lb = 0.0
            else:
                lb = deps[i-1] + h_min
            
            if i == D - 1:
                ub = T
            else:
                ub = deps[i+1] - h_min
            
            # Also enforce max headway
            if i > 0:
                lb = max(lb, deps[i-1] + h_min)
                ub_from_prev = deps[i-1] + h_max
                ub = min(ub, ub_from_prev)
            if i < D - 1:
                lb_from_next = deps[i+1] - h_max
                lb = max(lb, lb_from_next)
            
            if lb > ub + 1e-9:
                continue
            
            lb = max(lb, 0.0)
            ub = min(ub, T)
            
            # Golden section search or grid search
            best_t = deps[i]
            best_local_obj = current_obj
            
            # Try a grid of points
            n_points = 50
            for j in range(n_points + 1):
                t = lb + (ub - lb) * j / n_points
                old_val = deps[i]
                deps[i] = t
                obj = compute_objective(deps)
                if obj < best_local_obj - 1e-9:
                    best_local_obj = obj
                    best_t = t
                deps[i] = old_val
            
            if best_local_obj < current_obj - 1e-9:
                deps[i] = best_t
                current_obj = best_local_obj
                update_best(deps)
                improved = True
    
    return deps


def local_search_fine(deps, h_min, h_max, T, C, NT, theta, lambdas, period_starts, period_ends,
                     cumulative_arrivals, arrivals_between, compute_objective, update_best,
                     start_time, time_limit):
    """Fine-grained local search."""
    D = len(deps)
    if D == 0:
        return deps
    
    current_obj = compute_objective(deps)
    
    # Multiple rounds with increasing precision
    for n_points in [100, 200, 500]:
        if time.time() - start_time > time_limit - 2:
            break
        improved = True
        while improved and time.time() - start_time < time_limit - 2:
            improved = False
            for i in range(D):
                if time.time() - start_time > time_limit - 2:
                    break
                
                if i == 0:
                    lb = 0.0
                else:
                    lb = deps[i-1] + h_min
                
                if i == D - 1:
                    ub = T
                else:
                    ub = deps[i+1] - h_min
                
                if i > 0:
                    ub = min(ub, deps[i-1] + h_max)
                if i < D - 1:
                    lb = max(lb, deps[i+1] - h_max)
                
                lb = max(lb, 0.0)
                ub = min(ub, T)
                
                if lb > ub + 1e-9:
                    continue
                
                best_t = deps[i]
                best_local_obj = current_obj
                
                for j in range(n_points + 1):
                    t = lb + (ub - lb) * j / n_points
                    old_val = deps[i]
                    deps[i] = t
                    obj = compute_objective(deps)
                    if obj < best_local_obj - 1e-9:
                        best_local_obj = obj
                        best_t = t
                    deps[i] = old_val
                
                if best_local_obj < current_obj - 1e-9:
                    deps[i] = best_t
                    current_obj = best_local_obj
                    update_best(deps)
                    improved = True
    
    # Ternary search refinement
    for _ in range(5):
        if time.time() - start_time > time_limit - 1:
            break
        any_improved = False
        for i in range(D):
            if time.time() - start_time > time_limit - 1:
                break
            
            if i == 0:
                lb = 0.0
            else:
                lb = deps[i-1] + h_min
            
            if i == D - 1:
                ub = T
            else:
                ub = deps[i+1] - h_min
            
            if i > 0:
                ub = min(ub, deps[i-1] + h_max)
            if i < D - 1:
                lb = max(lb, deps[i+1] - h_max)
            
            lb = max(lb, 0.0)
            ub = min(ub, T)
            
            if lb > ub + 1e-9:
                continue
            
            # Ternary search (works if objective is unimodal in this variable - not guaranteed, but worth trying)
            lo, hi = lb, ub
            for _ in range(100):
                if hi - lo < 1e-6:
                    break
                m1 = lo + (hi - lo) / 3
                m2 = hi - (hi - lo) / 3
                
                old_val = deps[i]
                deps[i] = m1
                obj1 = compute_objective(deps)
                deps[i] = m2
                obj2 = compute_objective(deps)
                deps[i] = old_val
                
                if obj1 < obj2:
                    hi = m2
                else:
                    lo = m1
            
            best_t = (lo + hi) / 2
            old_val = deps[i]
            deps[i] = best_t
            obj = compute_objective(deps)
            if obj < current_obj - 1e-9:
                current_obj = obj
                update_best(deps)
                any_improved = True
            else:
                deps[i] = old_val
        
        if not any_improved:
            break
    
    return deps


def solve_with_gurobi(T, D, NT, theta, h_min, h_max, C, lambdas, period_starts, period_ends,
                      cumulative_arrivals, arrivals_between, compute_objective,
                      start_time, time_limit, warm_start_deps, warm_start_obj):
    """Solve using Gurobi MIP with piecewise linear approximation."""
    import gurobipy as gp
    from gurobipy import GRB
    
    remaining_time = time_limit - (time.time() - start_time) - 5
    if remaining_time < 5:
        return None
    
    m = gp.Model("train_scheduling")
    m.Params.TimeLimit = remaining_time
    m.Params.OutputFlag = 0
    m.Params.Threads = 1
    
    # Decision variables: departure times
    t = m.addVars(D, lb=0, ub=T, name="t")
    
    # Headway variables
    # h[0] = t[0] - 0 (initial)
    # h[i] = t[i] - t[i-1] for i=1,...,D-1 (interior)
    # h[D] = T - t[D-1] (final)
    h = m.addVars(D + 1, lb=0, name="h")
    
    # Ordering and headway constraints
    m.addConstr(h[0] == t[0])
    for i in range(1, D):
        m.addConstr(h[i] == t[i] - t[i - 1])
        m.addConstr(h[i] >= h_min)
        m.addConstr(h[i] <= h_max)
    m.addConstr(h[D] == T - t[D - 1])
    
    # For the objective, we need to linearize h[i] * (waiting[i] + arrivals[i]/2)
    # This is complex because waiting depends on capacity constraints
    
    # Let's use a simpler approach: discretize time and use assignment variables
    # Actually, for moderate D and NT, let's try a direct nonlinear approach with PWL
    
    # Cumulative arrivals at time t is piecewise linear - good!
    # A(t) = sum over periods before t of lambda_i * theta + lambda_k * (t - start_k)
    # where k is the period containing t
    
    # Create piecewise linear cumulative arrival variables
    # A[i] = cumulative arrivals at departure i
    A = m.addVars(D, lb=0, name="A")
    
    # A is piecewise linear in t[i]
    # Breakpoints at period boundaries
    for i in range(D):
        breakpoints = [0.0]
        values = [0.0]
        for p in range(NT):
            bp = period_ends[p]
            val = cumulative_arrivals(bp)
            breakpoints.append(bp)
            values.append(val)
        # Add general constraint for PWL
        m.addGenConstrPWL(t[i], A[i], breakpoints, values)
    
    # Also need A at time 0 and time T
    A_0 = 0.0
    A_T = cumulative_arrivals(T)
    
    # New arrivals in each interval
    # arr[0] = A[0] - A_0 = A[0]
    # arr[i] = A[i] - A[i-1] for i=1,...,D-1
    # arr[D] = A_T - A[D-1]
    arr = m.addVars(D + 1, lb=0, name="arr")
    m.addConstr(arr[0] == A[0])
    for i in range(1, D):
        m.addConstr(arr[i] == A[i] - A[i - 1])
    m.addConstr(arr[D] == A_T - A[D - 1])
    
    # Boarding and waiting
    board = m.addVars(D, lb=0, ub=C, name="board")
    wait_before = m.addVars(D + 1, lb=0, name="wait_before")  # waiting at start of interval i
    
    # wait_before[0] = 0 (no one waiting at time 0)
    m.addConstr(wait_before[0] == 0)
    
    # After each train i departs (end of interval i):
    # people on platform = wait_before[i] + arr[i]
    # board[i] = min(C, wait_before[i] + arr[i])
    # wait_before[i+1] = wait_before[i] + arr[i] - board[i]
    
    for i in range(D):
        # platform[i] = wait_before[i] + arr[i]
        platform_i = m.addVar(lb=0, name=f"platform_{i}")
        m.addConstr(platform_i == wait_before[i] + arr[i])
        m.addConstr(board[i] <= platform_i)
        m.addConstr(board[i] <= C)
        m.addConstr(wait_before[i + 1] == platform_i - board[i])
    
    # Objective: sum over intervals of h[i] * (wait_before[i] + arr[i]/2)
    # This is bilinear: h[i] * wait_before[i] + h[i] * arr[i] / 2
    
    # For interval 0: h[0] * (0 + arr[0]/2) = h[0] * arr[0] / 2
    # = t[0] * A[0] / 2   -- this is bilinear in t[0] and A[0], but A[0] is PWL in t[0]
    # Actually A[0] is a function of t[0] only, so h[0]*arr[0]/2 = t[0]*A(t[0])/2
    # which is a function of t[0] only.
    
    # This is getting complex. Let me try a different approach:
    # Approximate the objective using the warm start solution and solve a sequence of LPs.
    
    m.remove(m.getVars())
    m.remove(m.getConstrs())
    m.remove(m.getGenConstrs())
    del m
    
    # Alternative: Use scipy.optimize
    return scipy_optimize(T, D, NT, theta, h_min, h_max, C, lambdas, period_starts, period_ends,
                         cumulative_arrivals, arrivals_between, compute_objective,
                         start_time, time_limit, warm_start_deps)


def scipy_optimize(T, D, NT, theta, h_min, h_max, C, lambdas, period_starts, period_ends,
                  cumulative_arrivals, arrivals_between, compute_objective,
                  start_time, time_limit, warm_start_deps):
    """Use scipy for continuous optimization."""
    try:
        from scipy.optimize import minimize
    except ImportError:
        return None
    
    if D == 0:
        return None
    
    def objective(x):
        deps = sorted(x)
        return compute_objective(deps)
    
    # Bounds
    bounds = [(0, T)] * D
    
    x0 = np.array(warm_start_deps)
    
    # Constraints: h_min <= x[i+1] - x[i] <= h_max for interior
    constraints = []
    for i in range(D - 1):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i+1] - x[i] - h_min})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: h_max - (x[i+1] - x[i])})
    
    remaining = time_limit - (time.time() - start_time) - 5
    if remaining < 2:
        return None
    
    try:
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints,
                         options={'maxiter': 1000, 'ftol': 1e-10})
        if result.success or result.fun < compute_objective(warm_start_deps):
            deps = sorted(result.x.tolist())
            # Verify feasibility
            feasible = True
            for i in range(D - 1):
                hw = deps[i+1] - deps[i]
                if hw < h_min - 1e-6 or hw > h_max + 1e-6:
                    feasible = False
                    break
            if feasible:
                return deps
    except:
        pass
    
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, default=300)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)


if __name__ == '__main__':
    main()
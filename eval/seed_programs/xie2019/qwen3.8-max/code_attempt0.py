import json
import argparse
import time
import random
import itertools
import sys
from collections import defaultdict


def generate_all_profiles(instance):
    """Generate QC profiles for all vessels."""
    T = instance['time_steps_per_shift']
    total_time_steps = instance['total_time_steps']
    time_horizon = instance['time_horizon_hours']
    
    big_params = instance['qc_profiles']['big']
    small_params = instance['qc_profiles']['small']
    
    # Pre-enumerate and sample profiles for big and small types
    def enumerate_and_sample(params, num_profiles, seed=0):
        qc_range = list(range(params['min_qc'], params['max_qc'] + 1))
        all_sequences = []
        for h in range(params['min_handling_shifts'], params['max_handling_shifts'] + 1):
            for seq in itertools.product(qc_range, repeat=h):
                all_sequences.append((h, seq))
        
        rng = random.Random(seed)
        if len(all_sequences) <= num_profiles:
            sampled = all_sequences
        else:
            sampled = rng.sample(all_sequences, num_profiles)
        return sampled
    
    big_sampled = enumerate_and_sample(big_params, instance['num_big_profiles'], seed=0)
    small_sampled = enumerate_and_sample(small_params, instance['num_small_profiles'], seed=0)
    
    vessel_info = []
    
    for v_idx, vessel in enumerate(instance['vessels']):
        D = int(round(vessel['duration_hours'] * total_time_steps / time_horizon))
        if D <= 0:
            D = 1
        h = (D + T - 1) // T  # ceiling division
        s = h * T - D  # within-shift start position
        
        # Determine vessel type and get appropriate sampled profiles
        if small_params['min_handling_shifts'] <= h <= small_params['max_handling_shifts']:
            sampled_pool = small_sampled
        else:
            sampled_pool = big_sampled
        
        # Filter profiles matching this vessel's required number of shifts
        matching = [(hh, seq) for (hh, seq) in sampled_pool if hh == h]
        
        # If no matching profiles from the preferred type, try the other
        if not matching:
            if small_params['min_handling_shifts'] <= h <= small_params['max_handling_shifts']:
                matching = [(hh, seq) for (hh, seq) in big_sampled if hh == h]
            else:
                matching = [(hh, seq) for (hh, seq) in small_sampled if hh == h]
        
        # If still no match, generate directly
        if not matching:
            params = big_params if h > small_params['max_handling_shifts'] else small_params
            qc_range = list(range(params['min_qc'], params['max_qc'] + 1))
            num_p = instance['num_big_profiles'] if h > small_params['max_handling_shifts'] else instance['num_small_profiles']
            all_seqs_h = list(itertools.product(qc_range, repeat=h))
            rng = random.Random(0)
            if len(all_seqs_h) <= num_p:
                matching = [(h, seq) for seq in all_seqs_h]
            else:
                sampled_seqs = rng.sample(all_seqs_h, num_p)
                matching = [(h, seq) for seq in sampled_seqs]
        
        # Create profile objects (expand to time-step level)
        profiles = []
        for (hh, seq) in matching:
            qc_per_step = []
            for shift_idx in range(hh):
                n_steps = T - s if shift_idx == 0 else T
                qc_per_step.extend([seq[shift_idx]] * n_steps)
            assert len(qc_per_step) == D
            profiles.append(qc_per_step)
        
        # Determine feasible (berth, start_time, profile) combinations
        feasible = []
        earliest = vessel['a']
        latest = vessel['b']
        
        for b_idx, berth in enumerate(instance['berths']):
            lo = max(earliest, berth['a_k'])
            hi = min(latest, berth['b_k'] - D + 1)
            
            if lo > hi:
                continue
            
            # Find first valid start time with correct within-shift position
            remainder = s - (lo % T)
            if remainder < 0:
                remainder += T
            first_t = lo + remainder
            
            if first_t > hi:
                continue
            
            for t in range(first_t, hi + 1, T):
                for p_idx in range(len(profiles)):
                    feasible.append((b_idx, t, p_idx))
        
        vessel_info.append({
            'id': v_idx,
            'duration': D,
            'within_shift': s,
            'num_shifts': h,
            'profiles': profiles,
            'feasible': feasible,
            'k_bar': vessel['k_bar'],
            't_bar': vessel['t_bar'],
            'a': vessel['a'],
            'b': vessel['b']
        })
    
    return vessel_info


def compute_objective(instance, assignment, vessel_info):
    """Compute total objective value."""
    c1 = instance['c1']
    c2 = instance['c2']
    obj = 0
    for v in range(instance['num_vessels']):
        b, t, p = assignment[v]
        vi = vessel_info[v]
        obj += c1 * abs(b - vi['k_bar']) + c2 * abs(t - vi['t_bar'])
    return obj


def get_solution_dict(instance, assignment, vessel_info, obj_val):
    """Create solution dictionary."""
    sol_vessels = []
    for v in range(instance['num_vessels']):
        b, t, p = assignment[v]
        vi = vessel_info[v]
        sol_vessels.append({
            'id': v,
            'berth': b,
            'start_time': t,
            'end_time': t + vi['duration'] - 1,
            'profile': p
        })
    return {'objective_value': obj_val, 'vessels': sol_vessels}


def solve_greedy(instance, vessel_info, time_limit, logger):
    """Greedy construction + local search."""
    num_vessels = instance['num_vessels']
    total_qc = instance['total_qc']
    total_time_steps = instance['total_time_steps']
    c1 = instance['c1']
    c2 = instance['c2']
    
    start_time = time.time()
    
    # Sort vessels by number of feasible assignments (most constrained first)
    order = sorted(range(num_vessels), key=lambda v: len(vessel_info[v]['feasible']))
    
    assignment = {}
    berth_intervals = defaultdict(list)  # b -> list of (start, end, vessel)
    qc_usage = [0] * total_time_steps
    
    def is_feasible(v, b, t, p):
        vi = vessel_info[v]
        D = vi['duration']
        profile = vi['profiles'][p]
        
        # Check berth non-overlap
        for (s, e, ov) in berth_intervals[b]:
            if ov != v and t <= e and t + D - 1 >= s:
                return False
        
        # Check QC capacity
        for tau in range(t, min(t + D, total_time_steps)):
            if qc_usage[tau] + profile[tau - t] > total_qc:
                return False
        
        return True
    
    def do_assign(v, b, t, p):
        vi = vessel_info[v]
        D = vi['duration']
        profile = vi['profiles'][p]
        assignment[v] = (b, t, p)
        berth_intervals[b].append((t, t + D - 1, v))
        for tau in range(t, min(t + D, total_time_steps)):
            qc_usage[tau] += profile[tau - t]
    
    def do_unassign(v):
        if v not in assignment:
            return
        b, t, p = assignment[v]
        vi = vessel_info[v]
        D = vi['duration']
        profile = vi['profiles'][p]
        berth_intervals[b] = [(s, e, ov) for (s, e, ov) in berth_intervals[b] if ov != v]
        for tau in range(t, min(t + D, total_time_steps)):
            qc_usage[tau] -= profile[tau - t]
        del assignment[v]
    
    # Greedy assignment
    for v in order:
        vi = vessel_info[v]
        best_cost = float('inf')
        best_assign = None
        
        # Sort feasible by cost for this vessel
        sorted_feasible = sorted(vi['feasible'],
                                  key=lambda x: c1 * abs(x[0] - vi['k_bar']) + c2 * abs(x[1] - vi['t_bar']))
        
        for (b, t, p) in sorted_feasible:
            if is_feasible(v, b, t, p):
                cost = c1 * abs(b - vi['k_bar']) + c2 * abs(t - vi['t_bar'])
                best_assign = (b, t, p)
                break
        
        if best_assign:
            do_assign(v, *best_assign)
    
    if len(assignment) < num_vessels:
        # Try to assign remaining vessels with any feasible option
        for v in range(num_vessels):
            if v not in assignment:
                vi = vessel_info[v]
                for (b, t, p) in vi['feasible']:
                    if is_feasible(v, b, t, p):
                        do_assign(v, b, t, p)
                        break
    
    if len(assignment) < num_vessels:
        return None
    
    best_obj = compute_objective(instance, assignment, vessel_info)
    
    if logger:
        logger.log_solution(best_obj, get_solution_dict(instance, assignment, vessel_info, best_obj))
    
    # Local search - iterative improvement
    end_time = start_time + time_limit
    iteration = 0
    
    while time.time() < end_time - 0.5:
        improved = False
        iteration += 1
        
        # Randomize vessel order each iteration for diversity
        if iteration > 1:
            random.shuffle(order)
        
        for v in order:
            if time.time() >= end_time - 0.5:
                break
            
            vi = vessel_info[v]
            current_b, current_t, current_p = assignment[v]
            current_cost = c1 * abs(current_b - vi['k_bar']) + c2 * abs(current_t - vi['t_bar'])
            
            do_unassign(v)
            
            best_new_cost = current_cost
            best_new_assign = (current_b, current_t, current_p)
            
            for (b, t, p) in vi['feasible']:
                if is_feasible(v, b, t, p):
                    cost = c1 * abs(b - vi['k_bar']) + c2 * abs(t - vi['t_bar'])
                    if cost < best_new_cost:
                        best_new_cost = cost
                        best_new_assign = (b, t, p)
            
            do_assign(v, *best_new_assign)
            
            if best_new_cost < current_cost:
                improved = True
        
        if improved:
            new_obj = compute_objective(instance, assignment, vessel_info)
            if new_obj < best_obj:
                best_obj = new_obj
                if logger:
                    logger.log_solution(best_obj, get_solution_dict(instance, assignment, vessel_info, best_obj))
        else:
            break
    
    return best_obj, dict(assignment)


def solve_with_gurobi(instance, vessel_info, time_limit, logger, warm_start=None):
    """Solve using Gurobi MIP."""
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError:
        return None
    
    num_vessels = instance['num_vessels']
    num_berths = instance['num_berths']
    total_qc = instance['total_qc']
    total_time_steps = instance['total_time_steps']
    c1 = instance['c1']
    c2 = instance['c2']
    
    total_vars = sum(len(vi['feasible']) for vi in vessel_info)
    if total_vars > 200000:
        return None
    
    model = gp.Model("bap")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = max(1, time_limit)
    model.Params.OutputFlag = 0
    
    # Variables
    x = {}
    for v in range(num_vessels):
        vi = vessel_info[v]
        for (b, t, p) in vi['feasible']:
            x[v, b, t, p] = model.addVar(vtype=GRB.BINARY, name=f"x_{v}_{b}_{t}_{p}")
    
    model.update()
    
    # Objective
    obj = gp.LinExpr()
    for v in range(num_vessels):
        vi = vessel_info[v]
        for (b, t, p) in vi['feasible']:
            cost = c1 * abs(b - vi['k_bar']) + c2 * abs(t - vi['t_bar'])
            obj += cost * x[v, b, t, p]
    model.setObjective(obj, GRB.MINIMIZE)
    
    # Assignment constraints
    for v in range(num_vessels):
        vi = vessel_info[v]
        if vi['feasible']:
            model.addConstr(
                gp.quicksum(x[v, b, t, p] for (b, t, p) in vi['feasible']) == 1,
                name=f"assign_{v}"
            )
        else:
            return None  # Infeasible
    
    # Berth non-overlap constraints
    berth_time_vars = defaultdict(list)
    for v in range(num_vessels):
        vi = vessel_info[v]
        D = vi['duration']
        for (b, t, p) in vi['feasible']:
            for tau in range(t, min(t + D, total_time_steps)):
                berth_time_vars[(b, tau)].append(x[v, b, t, p])
    
    for key, var_list in berth_time_vars.items():
        if len(var_list) > 1:
            model.addConstr(gp.quicksum(var_list) <= 1, name=f"berth_{key[0]}_{key[1]}")
    
    # QC capacity constraints
    time_qc_vars = defaultdict(list)
    for v in range(num_vessels):
        vi = vessel_info[v]
        D = vi['duration']
        for (b, t, p) in vi['feasible']:
            profile = vi['profiles'][p]
            for tau in range(t, min(t + D, total_time_steps)):
                qc_count = profile[tau - t]
                if qc_count > 0:
                    time_qc_vars[tau].append((qc_count, x[v, b, t, p]))
    
    for tau, var_list in time_qc_vars.items():
        if var_list:
            model.addConstr(
                gp.quicksum(qc * var for (qc, var) in var_list) <= total_qc,
                name=f"qc_{tau}"
            )
    
    # Warm start
    if warm_start is not None:
        for v in range(num_vessels):
            if v in warm_start:
                b, t, p = warm_start[v]
                if (v, b, t, p) in x:
                    x[v, b, t, p].Start = 1.0
    
    # Callback for logging
    best_cb_obj = [float('inf')]
    
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj_val < best_cb_obj[0]:
                best_cb_obj[0] = obj_val
                sol_vessels = []
                for v in range(num_vessels):
                    vi = vessel_info[v]
                    for (b, t, p) in vi['feasible']:
                        if model.cbGetSolution(x[v, b, t, p]) > 0.5:
                            sol_vessels.append({
                                'id': v,
                                'berth': b,
                                'start_time': t,
                                'end_time': t + vi['duration'] - 1,
                                'profile': p
                            })
                            break
                if logger and len(sol_vessels) == num_vessels:
                    logger.log_solution(obj_val, {'objective_value': obj_val, 'vessels': sol_vessels})
    
    model.optimize(callback)
    
    if model.SolCount > 0:
        obj_val = model.ObjVal
        sol_vessels = []
        assignment = {}
        for v in range(num_vessels):
            vi = vessel_info[v]
            found = False
            for (b, t, p) in vi['feasible']:
                if x[v, b, t, p].X > 0.5:
                    sol_vessels.append({
                        'id': v,
                        'berth': b,
                        'start_time': t,
                        'end_time': t + vi['duration'] - 1,
                        'profile': p
                    })
                    assignment[v] = (b, t, p)
                    found = True
                    break
            if not found:
                return None
        
        if logger:
            logger.log_solution(obj_val, {'objective_value': obj_val, 'vessels': sol_vessels})
        
        return obj_val, assignment
    
    return None


def main():
    parser = argparse.ArgumentParser(description="Berth Allocation and QC Scheduling")
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    start_time = time.time()
    
    with open(args.instance_path) as f:
        instance = json.load(f)
    
    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    # Generate profiles and feasible assignments
    vessel_info = generate_all_profiles(instance)
    
    num_vessels = instance['num_vessels']
    best_obj = float('inf')
    best_assignment = None
    
    # Phase 1: Greedy heuristic for initial solution
    greedy_time = min(args.time_limit * 0.3, 30)
    greedy_result = solve_greedy(instance, vessel_info, greedy_time, logger)
    
    if greedy_result is not None:
        best_obj, best_assignment = greedy_result
    
    # Phase 2: Try MIP if time allows and problem is not too large
    elapsed = time.time() - start_time
    remaining = args.time_limit - elapsed
    
    total_vars = sum(len(vi['feasible']) for vi in vessel_info)
    
    if remaining > 5 and total_vars <= 150000:
        mip_time = remaining - 2
        mip_result = solve_with_gurobi(instance, vessel_info, mip_time, logger, 
                                        warm_start=best_assignment)
        
        if mip_result is not None:
            mip_obj, mip_assignment = mip_result
            if mip_obj < best_obj:
                best_obj = mip_obj
                best_assignment = mip_assignment
    
    # Phase 3: Additional local search if time remains
    elapsed = time.time() - start_time
    remaining = args.time_limit - elapsed
    
    if remaining > 2 and best_assignment is not None:
        # Try additional local search from the best solution found
        extra_result = solve_greedy(instance, vessel_info, remaining - 1, logger)
        if extra_result is not None and extra_result[0] < best_obj:
            best_obj, best_assignment = extra_result
    
    # Write solution
    if best_assignment is not None and len(best_assignment) == num_vessels:
        solution = get_solution_dict(instance, best_assignment, vessel_info, best_obj)
    else:
        # Fallback: try to construct any feasible solution
        solution = {'objective_value': None, 'vessels': []}
        for v in range(num_vessels):
            vi = vessel_info[v]
            if vi['feasible']:
                b, t, p = vi['feasible'][0]
                solution['vessels'].append({
                    'id': v,
                    'berth': b,
                    'start_time': t,
                    'end_time': t + vi['duration'] - 1,
                    'profile': p
                })
            else:
                solution['vessels'].append({
                    'id': v,
                    'berth': 0,
                    'start_time': 0,
                    'end_time': 0,
                    'profile': 0
                })
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == '__main__':
    main()
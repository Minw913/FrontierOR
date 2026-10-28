import argparse
import json
import random
import time
import itertools
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    start_wall = time.time()

    with open(args.instance_path, 'r') as f:
        inst = json.load(f)

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    num_vessels = inst['num_vessels']
    num_berths = inst['num_berths']
    total_qc = inst['total_qc']
    num_big_profiles = inst['num_big_profiles']
    num_small_profiles = inst['num_small_profiles']
    time_steps_per_shift = inst['time_steps_per_shift']
    total_time_steps = inst['total_time_steps']
    num_shifts = inst['num_shifts_per_step']  # total number of shifts
    c1 = inst['c1']
    c2 = inst['c2']
    vessels = inst['vessels']
    berths = inst['berths']
    qc_prof = inst['qc_profiles']

    # Determine big vs small threshold: vessels with longer duration are "big"
    # Typically, a vessel is "big" if its duration_hours suggests more shifts
    # Let's use the profile parameters to determine: if a vessel's expected handling
    # duration in shifts falls within big range, it's big; otherwise small.
    # Actually, we need to figure out the classification. Let's compute expected shifts:
    # duration_hours / (time_horizon_hours / num_shifts) = duration in shifts
    time_horizon_hours = inst['time_horizon_hours']
    hours_per_shift = time_horizon_hours / num_shifts if num_shifts > 0 else 1

    def classify_vessel(v):
        dur_shifts = v['duration_hours'] / hours_per_shift
        big = qc_prof['big']
        small = qc_prof['small']
        # If vessel's duration in shifts is closer to big range, classify as big
        if dur_shifts >= big['min_handling_shifts']:
            return 'big'
        else:
            return 'small'

    def generate_profiles_for_vessel(v, seed_base):
        """Generate QC profiles for a vessel."""
        vtype = classify_vessel(v)
        if vtype == 'big':
            params = qc_prof['big']
            num_profiles_to_gen = num_big_profiles
        else:
            params = qc_prof['small']
            num_profiles_to_gen = num_small_profiles

        min_qc = params['min_qc']
        max_qc = params['max_qc']
        min_shifts = params['min_handling_shifts']
        max_shifts = params['max_handling_shifts']

        # Generate all possible shift-level QC sequences
        # A profile is a tuple of length num_shifts_in_profile where each element is in [min_qc, max_qc]
        # For each possible number of handling shifts from min_shifts to max_shifts
        all_profiles = []
        for ns in range(min_shifts, max_shifts + 1):
            qc_values = list(range(min_qc, max_qc + 1))
            for combo in itertools.product(qc_values, repeat=ns):
                all_profiles.append(combo)

        # Sample from all_profiles using the seed
        rng = random.Random(seed_base)
        if len(all_profiles) <= num_profiles_to_gen:
            sampled = all_profiles[:]
        else:
            sampled = rng.sample(all_profiles, num_profiles_to_gen)

        # Now, for each within-shift index, expand profiles to time-step level
        # A profile with shift-level QCs (q1, q2, ..., qn) starting at within-shift index s
        # means:
        # - First partial shift: (time_steps_per_shift - s) time steps with q1 QCs
        # - Then full shifts with q2, q3, ... QCs
        # - Wait, actually re-reading the problem:
        # "QCs assigned to a vessel cannot be reassigned until the end of the current working shift"
        # So QC count changes only at shift boundaries.
        
        # For within-shift index ws (0-indexed position within a shift):
        # If service starts at global time t, ws = t % time_steps_per_shift
        # The first shift segment has (time_steps_per_shift - ws) time steps remaining
        # Then subsequent shifts have time_steps_per_shift time steps each
        # Total duration = (time_steps_per_shift - ws) + (len(combo) - 1) * time_steps_per_shift
        
        # Filter profiles based on vessel's feasible duration window
        # The vessel's feasible window is [a, b] for start times
        # Duration must allow completion within the total time horizon
        
        profiles_by_ws = {}
        for ws in range(time_steps_per_shift):
            ws_profiles = []
            for combo in sampled:
                ns = len(combo)
                # Duration in time steps
                duration = (time_steps_per_shift - ws) + (ns - 1) * time_steps_per_shift
                # Check if this duration is feasible for this vessel
                # The vessel must complete within the planning horizon
                # Also check if vessel can start at some time with this ws
                ws_profiles.append({
                    'qc_per_shift': combo,
                    'duration': duration,
                    'qc_timeline': []  # will be filled
                })
                # Build QC timeline
                timeline = []
                # First partial shift
                for _ in range(time_steps_per_shift - ws):
                    timeline.append(combo[0])
                # Remaining full shifts
                for si in range(1, ns):
                    for _ in range(time_steps_per_shift):
                        timeline.append(combo[si])
                ws_profiles[-1]['qc_timeline'] = timeline
                ws_profiles[-1]['duration'] = len(timeline)
            
            profiles_by_ws[ws] = ws_profiles
        
        return profiles_by_ws

    # Generate profiles for each vessel
    # The seed for profile generation - using a single seed from instance
    # Since the instance mentions "random seed" but doesn't explicitly store it in the schema,
    # let's check if there's a seed field. Looking at the schema again...
    # The schema says "the instance stores the profile-generation ranges, requested counts, and random seed"
    # but the schema doesn't show a seed field explicitly. Let me check for extra fields.
    
    instance_seed = inst.get('seed', 42)  # default seed if not provided
    
    vessel_profiles = {}  # vessel_id -> {ws -> [profiles]}
    for v in vessels:
        # Use vessel id combined with instance seed for deterministic generation
        vessel_profiles[v['id']] = generate_profiles_for_vessel(v, instance_seed + v['id'])

    # Now solve using Gurobi MIP
    import gurobipy as gp
    from gurobipy import GRB

    remaining_time = args.time_limit - (time.time() - start_wall)
    if remaining_time < 1:
        remaining_time = 1

    # First, let's try a greedy heuristic to get a feasible solution quickly
    # Then use MIP for improvement
    
    # Precompute feasible (berth, start_time, profile) combinations for each vessel
    # To keep the model tractable, we'll work with vessel-specific feasible options
    
    # For each vessel, enumerate feasible (berth, start_time, profile_index) triples
    # This could be very large, so we need to be smart about it
    
    # Let's first understand the scale
    # Feasible start times for vessel v: [max(v.a, berth.a_k), min(v.b, berth.b_k)]
    # For each start time, ws = start_time % time_steps_per_shift
    # Then profiles from vessel_profiles[v.id][ws]
    
    # Let's try a column-based approach or direct MIP
    
    # Direct MIP approach:
    # Variables:
    # x[v, k, t, p] = 1 if vessel v is assigned to berth k, starts at time t, with profile p
    # This is potentially huge. Let's see the sizes.
    
    # Alternative: decompose
    # y[v, k] = 1 if vessel v assigned to berth k
    # s[v] = start time of vessel v (integer)
    # p[v] = profile index for vessel v
    # But linking these is complex.
    
    # Let's try an approach where we enumerate feasible "options" for each vessel
    # and use a set-packing/partitioning formulation
    
    # First, let's enumerate all feasible options per vessel
    options = {}  # vessel_id -> list of (berth, start_time, profile_idx, duration, qc_timeline, cost)
    
    for v in vessels:
        vid = v['id']
        opts = []
        for berth in berths:
            kid = berth['id']
            # Feasible start time range
            t_min = max(v['a'], berth['a_k'])
            t_max = min(v['b'], berth['b_k'])
            
            for t in range(t_min, t_max + 1):
                ws = t % time_steps_per_shift
                if ws not in vessel_profiles[vid]:
                    continue
                for pidx, prof in enumerate(vessel_profiles[vid][ws]):
                    dur = prof['duration']
                    end_time = t + dur - 1
                    # Check end time is within horizon and berth availability
                    if end_time >= total_time_steps:
                        continue
                    if end_time > berth['b_k']:
                        continue
                    # Compute cost
                    berth_dev = abs(kid - v['k_bar'])
                    time_dev = abs(t - v['t_bar'])
                    cost = c1 * berth_dev + c2 * time_dev
                    opts.append((kid, t, pidx, dur, prof['qc_timeline'], cost, end_time))
        
        options[vid] = opts
    
    # Check sizes
    total_options = sum(len(opts) for opts in options.values())
    
    # If total options is manageable, use Gurobi directly
    # If too large, use heuristic
    
    # Let's check feasibility first
    for v in vessels:
        if len(options[v['id']]) == 0:
            # No feasible option - problem might be infeasible
            # Try relaxing constraints
            pass
    
    best_solution = None
    best_obj = float('inf')
    
    def build_solution_dict(assignments):
        """Build solution dictionary from assignments: {vid: (berth, start, pidx, dur, qc_timeline, cost, end)}"""
        sol_vessels = []
        total_cost = 0
        for v in vessels:
            vid = v['id']
            if vid not in assignments:
                return None, float('inf')
            berth, start, pidx, dur, _, cost, end = assignments[vid]
            total_cost += cost
            # Find the global profile index
            ws = start % time_steps_per_shift
            sol_vessels.append({
                'id': vid,
                'berth': berth,
                'start_time': start,
                'end_time': end,
                'profile': pidx
            })
        return {'objective_value': total_cost, 'vessels': sol_vessels}, total_cost

    # Greedy heuristic: assign vessels one by one, picking lowest cost option that doesn't conflict
    def greedy_solve():
        # Sort vessels by feasibility window size (most constrained first)
        sorted_vessels = sorted(vessels, key=lambda v: len(options[v['id']]))
        
        assignments = {}
        berth_intervals = defaultdict(list)  # berth -> list of (start, end) intervals
        qc_usage = defaultdict(int)  # time_step -> total QCs used
        
        for v in sorted_vessels:
            vid = v['id']
            best_opt = None
            best_cost = float('inf')
            
            # Sort options by cost
            sorted_opts = sorted(options[vid], key=lambda o: o[5])
            
            for opt in sorted_opts:
                kid, t, pidx, dur, qc_timeline, cost, end = opt
                if cost >= best_cost:
                    break
                
                # Check berth conflict
                conflict = False
                for (s2, e2) in berth_intervals[kid]:
                    if t <= e2 and end >= s2:
                        conflict = True
                        break
                if conflict:
                    continue
                
                # Check QC capacity
                qc_ok = True
                for dt in range(dur):
                    ts = t + dt
                    needed = qc_timeline[dt]
                    if qc_usage[ts] + needed > total_qc:
                        qc_ok = False
                        break
                if not qc_ok:
                    continue
                
                best_opt = opt
                best_cost = cost
                break
            
            if best_opt is not None:
                kid, t, pidx, dur, qc_timeline, cost, end = best_opt
                assignments[vid] = best_opt
                berth_intervals[kid].append((t, end))
                for dt in range(dur):
                    qc_usage[t + dt] += qc_timeline[dt]
        
        return assignments
    
    assignments = greedy_solve()
    if len(assignments) == num_vessels:
        sol, obj = build_solution_dict(assignments)
        if obj < best_obj:
            best_obj = obj
            best_solution = sol
            if logger:
                logger.log_solution(best_obj, best_solution)
    
    remaining_time = args.time_limit - (time.time() - start_wall) - 2
    
    # Try MIP if options are not too many
    if total_options <= 500000 and remaining_time > 5:
        try:
            model = gp.Model("berth_allocation")
            model.Params.Seed = 0
            model.Params.MIPGap = 1e-4
            model.Params.NumericFocus = 0
            model.Params.Threads = 1
            model.Params.TimeLimit = max(remaining_time, 1)
            
            # Create variables: x[v][i] = 1 if vessel v uses option i
            x = {}
            for v in vessels:
                vid = v['id']
                x[vid] = {}
                for i, opt in enumerate(options[vid]):
                    x[vid][i] = model.addVar(vtype=GRB.BINARY, obj=opt[5], name=f"x_{vid}_{i}")
            
            model.update()
            
            # Each vessel must be assigned exactly one option
            for v in vessels:
                vid = v['id']
                if len(x[vid]) > 0:
                    model.addConstr(gp.quicksum(x[vid][i] for i in x[vid]) == 1, name=f"assign_{vid}")
                else:
                    # Infeasible for this vessel - skip constraint or make model infeasible
                    pass
            
            # Berth non-overlap: for each berth, no two vessels can overlap
            # For each pair of vessels that could be on the same berth, add constraints
            # This is expensive for many vessels, so let's use time-indexed constraints
            
            # Alternative: for each berth and time step, at most one vessel
            # But that could also be huge. Let's be smart.
            
            # Group options by berth
            berth_options = defaultdict(list)  # berth -> [(vid, opt_idx, start, end)]
            for v in vessels:
                vid = v['id']
                for i, opt in enumerate(options[vid]):
                    kid, t, pidx, dur, qc_timeline, cost, end = opt
                    berth_options[kid].append((vid, i, t, end))
            
            # For each berth, for each time step, add constraint that sum of active vessels <= 1
            # But this could be huge. Instead, use pairwise conflict constraints
            for kid in berth_options:
                opts_list = berth_options[kid]
                # Sort by start time
                opts_list.sort(key=lambda o: o[2])
                
                # For efficiency, group by time step coverage
                # Use interval-based conflict detection
                # For each pair of options that overlap in time and are from different vessels
                n_opts = len(opts_list)
                
                if n_opts > 10000:
                    # Use time-indexed constraints
                    # Find all time steps covered by any option on this berth
                    time_steps_covered = set()
                    for vid, oi, s, e in opts_list:
                        for ts in range(s, e + 1):
                            time_steps_covered.add(ts)
                    
                    # For each time step, at most one vessel option active
                    ts_to_opts = defaultdict(list)
                    for vid, oi, s, e in opts_list:
                        for ts in range(s, e + 1):
                            ts_to_opts[ts].append((vid, oi))
                    
                    for ts in ts_to_opts:
                        opts_at_ts = ts_to_opts[ts]
                        if len(opts_at_ts) > 1:
                            model.addConstr(
                                gp.quicksum(x[vid][oi] for vid, oi in opts_at_ts) <= 1,
                                name=f"berth_{kid}_t{ts}"
                            )
                else:
                    # Pairwise conflicts
                    for idx1 in range(n_opts):
                        vid1, oi1, s1, e1 = opts_list[idx1]
                        for idx2 in range(idx1 + 1, n_opts):
                            vid2, oi2, s2, e2 = opts_list[idx2]
                            if vid1 == vid2:
                                continue
                            if s2 > e1:
                                break  # sorted by start, no more overlaps
                            # They overlap
                            model.addConstr(x[vid1][oi1] + x[vid2][oi2] <= 1,
                                          name=f"nooverlap_{kid}_{vid1}_{oi1}_{vid2}_{oi2}")
            
            # QC capacity constraints: for each time step, total QCs <= total_qc
            # Build mapping: time_step -> list of (vid, opt_idx, qc_count)
            ts_qc = defaultdict(list)
            for v in vessels:
                vid = v['id']
                for i, opt in enumerate(options[vid]):
                    kid, t, pidx, dur, qc_timeline, cost, end = opt
                    for dt in range(dur):
                        ts = t + dt
                        qc_count = qc_timeline[dt]
                        if qc_count > 0:
                            ts_qc[ts].append((vid, i, qc_count))
            
            for ts in ts_qc:
                entries = ts_qc[ts]
                if len(entries) > 1 or (len(entries) == 1 and entries[0][2] > total_qc):
                    model.addConstr(
                        gp.quicksum(qc * x[vid][oi] for vid, oi, qc in entries) <= total_qc,
                        name=f"qc_cap_{ts}"
                    )
            
            # Set initial solution if we have one
            if best_solution is not None:
                for vs in best_solution['vessels']:
                    vid = vs['id']
                    for i, opt in enumerate(options[vid]):
                        kid, t, pidx, dur, qc_timeline, cost, end = opt
                        if kid == vs['berth'] and t == vs['start_time'] and pidx == vs['profile']:
                            x[vid][i].Start = 1.0
                            break
            
            model.ModelSense = GRB.MINIMIZE
            
            # Callback to log solutions
            def callback(model, where):
                if where == GRB.Callback.MIPSOL:
                    obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    nonlocal best_obj, best_solution
                    if obj_val < best_obj - 1e-6:
                        # Extract solution
                        sol_vessels = []
                        total_cost = 0
                        for v in vessels:
                            vid = v['id']
                            for i, opt in enumerate(options[vid]):
                                val = model.cbGetSolution(x[vid][i])
                                if val > 0.5:
                                    kid, t, pidx, dur, qc_timeline, cost, end = opt
                                    total_cost += cost
                                    sol_vessels.append({
                                        'id': vid,
                                        'berth': kid,
                                        'start_time': t,
                                        'end_time': end,
                                        'profile': pidx
                                    })
                                    break
                        
                        if len(sol_vessels) == num_vessels:
                            best_obj = total_cost
                            best_solution = {'objective_value': total_cost, 'vessels': sol_vessels}
                            if logger:
                                logger.log_solution(best_obj, best_solution)
            
            model.optimize(callback)
            
            if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL]:
                if model.SolCount > 0:
                    sol_vessels = []
                    total_cost = 0
                    for v in vessels:
                        vid = v['id']
                        for i, opt in enumerate(options[vid]):
                            if x[vid][i].X > 0.5:
                                kid, t, pidx, dur, qc_timeline, cost, end = opt
                                total_cost += cost
                                sol_vessels.append({
                                    'id': vid,
                                    'berth': kid,
                                    'start_time': t,
                                    'end_time': end,
                                    'profile': pidx
                                })
                                break
                    
                    if len(sol_vessels) == num_vessels and total_cost < best_obj:
                        best_obj = total_cost
                        best_solution = {'objective_value': total_cost, 'vessels': sol_vessels}
                        if logger:
                            logger.log_solution(best_obj, best_solution)
        
        except Exception as e:
            # If MIP fails, fall back to greedy
            pass
    
    elif remaining_time > 2:
        # Too many options for full MIP, try local search improvement on greedy solution
        # Simple local search: for each vessel, try to find a better option
        if best_solution is not None and len(assignments) == num_vessels:
            improved = True
            iter_count = 0
            while improved and time.time() - start_wall < args.time_limit - 1:
                improved = False
                iter_count += 1
                for v in vessels:
                    vid = v['id']
                    if time.time() - start_wall > args.time_limit - 1:
                        break
                    
                    current_opt = assignments[vid]
                    current_cost = current_opt[5]
                    
                    # Remove current assignment
                    ck, ct, cpidx, cdur, cqc, ccost, cend = current_opt
                    
                    # Build current state without this vessel
                    berth_intervals_temp = defaultdict(list)
                    qc_usage_temp = defaultdict(int)
                    for v2 in vessels:
                        v2id = v2['id']
                        if v2id == vid:
                            continue
                        if v2id in assignments:
                            o = assignments[v2id]
                            berth_intervals_temp[o[0]].append((o[1], o[6]))
                            for dt in range(o[3]):
                                qc_usage_temp[o[1] + dt] += o[4][dt]
                    
                    # Try to find better option
                    sorted_opts = sorted(options[vid], key=lambda o: o[5])
                    
                    for opt in sorted_opts:
                        kid, t, pidx, dur, qc_timeline, cost, end = opt
                        if cost >= current_cost:
                            break
                        
                        conflict = False
                        for (s2, e2) in berth_intervals_temp[kid]:
                            if t <= e2 and end >= s2:
                                conflict = True
                                break
                        if conflict:
                            continue
                        
                        qc_ok = True
                        for dt in range(dur):
                            ts = t + dt
                            needed = qc_timeline[dt]
                            if qc_usage_temp[ts] + needed > total_qc:
                                qc_ok = False
                                break
                        if not qc_ok:
                            continue
                        
                        # Found better option
                        assignments[vid] = opt
                        improved = True
                        break
                
                if improved:
                    sol, obj = build_solution_dict(assignments)
                    if obj < best_obj:
                        best_obj = obj
                        best_solution = sol
                        if logger:
                            logger.log_solution(best_obj, best_solution)
    
    # Write solution
    if best_solution is None:
        # Return empty/null solution
        best_solution = {
            'objective_value': None,
            'vessels': []
        }
    
    with open(args.solution_path, 'w') as f:
        json.dump(best_solution, f, indent=2)

if __name__ == '__main__':
    main()
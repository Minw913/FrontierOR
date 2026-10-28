#!/usr/bin/env python3
import json
import argparse
import time
from collections import defaultdict, deque
import sys

def hopcroft_karp(adj, n):
    """Find maximum matching in bipartite graph using Hopcroft-Karp algorithm."""
    pair_left = [-1] * n
    pair_right = [-1] * n
    dist = [0] * n
    
    def bfs():
        queue = deque()
        for u in range(n):
            if pair_left[u] == -1:
                dist[u] = 0
                queue.append(u)
            else:
                dist[u] = float('inf')
        found = False
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if pair_right[v] == -1:
                    found = True
                elif dist[pair_right[v]] == float('inf'):
                    dist[pair_right[v]] = dist[u] + 1
                    queue.append(pair_right[v])
        return found
    
    def dfs(u):
        for v in adj[u]:
            if pair_right[v] == -1 or (dist[pair_right[v]] == dist[u] + 1 and dfs(pair_right[v])):
                pair_left[u] = v
                pair_right[v] = u
                return True
        dist[u] = float('inf')
        return False
    
    sys.setrecursionlimit(max(10000, n * 3))
    matching = 0
    while bfs():
        for u in range(n):
            if pair_left[u] == -1:
                if dfs(u):
                    matching += 1
    return matching, pair_left


def solve_vehicle_scheduling(trips, travel_times, num_trips):
    """Find minimum path cover using maximum bipartite matching."""
    adj = [[] for _ in range(num_trips)]
    for i in range(num_trips):
        for j in range(i + 1, num_trips):
            if trips[i]['end_time'] <= trips[j]['start_time']:
                tt = travel_times[trips[i]['end_location']][trips[j]['start_location']]
                if trips[i]['end_time'] + tt <= trips[j]['start_time']:
                    adj[i].append(j)
    
    matching_size, pair_left = hopcroft_karp(adj, num_trips)
    
    has_pred = [False] * num_trips
    for i in range(num_trips):
        if pair_left[i] != -1:
            has_pred[pair_left[i]] = True
    
    blocks = []
    for i in range(num_trips):
        if not has_pred[i]:
            block = [i]
            current = i
            while pair_left[current] != -1:
                current = pair_left[current]
                block.append(current)
            blocks.append(block)
    
    return blocks


def get_break_points(block, trips, relief_locations):
    """Get valid break positions in a block.
    Position k means break between trip block[k-1] and block[k].
    Position 0 = before first trip, position len(block) = after last trip.
    """
    break_points = {0, len(block)}
    for k in range(1, len(block)):
        prev_trip = block[k - 1]
        next_trip = block[k]
        if (trips[prev_trip]['end_location'] in relief_locations or
                trips[next_trip]['start_location'] in relief_locations):
            break_points.add(k)
    return sorted(break_points)


def generate_pieces(blocks, trips, relief_locations):
    """Generate all valid pieces from vehicle blocks."""
    all_pieces = []
    for block_idx, block in enumerate(blocks):
        bp = get_break_points(block, trips, relief_locations)
        for i in range(len(bp)):
            for j in range(i + 1, len(bp)):
                start_idx = bp[i]
                end_idx = bp[j] - 1
                piece_trips = block[start_idx:end_idx + 1]
                if not piece_trips:
                    continue
                start_t = trips[piece_trips[0]]['start_time']
                end_t = trips[piece_trips[-1]]['end_time']
                duration = end_t - start_t
                all_pieces.append({
                    'block_idx': block_idx,
                    'trips': piece_trips,
                    'trip_set': set(piece_trips),
                    'start_time': start_t,
                    'end_time': end_t,
                    'duration': duration,
                    'start_pos': start_idx,
                    'end_pos': end_idx
                })
    return all_pieces


def generate_duties(all_pieces, duty_types, crew_params, num_trips, max_duties=100000):
    """Generate feasible duties from pieces."""
    sign_on = crew_params['sign_on_time_depot_minutes']
    sign_off = crew_params['sign_off_time_depot_minutes']
    
    duties = []
    
    # Generate tripper duties (single piece)
    tripper_type = None
    for name, dt in duty_types.items():
        if dt['num_pieces'] == 1:
            tripper_type = (name, dt)
            break
    
    if tripper_type:
        tname, tdt = tripper_type
        for piece in all_pieces:
            if piece['duration'] < tdt['piece_length_min']:
                continue
            if piece['duration'] > tdt['piece_length_max']:
                continue
            duty_start = piece['start_time'] - sign_on
            duty_end = piece['end_time'] + sign_off
            if tdt.get('start_time_min') is not None and duty_start < tdt['start_time_min']:
                continue
            if tdt.get('start_time_max') is not None and duty_start > tdt['start_time_max']:
                continue
            if tdt.get('end_time_max') is not None and duty_end > tdt['end_time_max']:
                continue
            if tdt.get('duty_length_max') is not None:
                if (duty_end - duty_start) > tdt['duty_length_max']:
                    continue
            if tdt.get('work_time_max') is not None:
                if piece['duration'] > tdt['work_time_max']:
                    continue
            duties.append({
                'type': tname,
                'trips': piece['trips'],
                'trip_set': piece['trip_set'],
                'pieces': [piece]
            })
    
    # Generate 2-piece duties
    two_piece_types = [(name, dt) for name, dt in duty_types.items() if dt['num_pieces'] == 2]
    
    if two_piece_types and len(all_pieces) > 0:
        # Sort pieces by start time for efficiency
        sorted_pieces = sorted(all_pieces, key=lambda p: p['start_time'])
        
        for tname, tdt in two_piece_types:
            if len(duties) >= max_duties:
                break
            pl_min = tdt['piece_length_min']
            pl_max = tdt['piece_length_max']
            br_min = tdt.get('break_length_min', 0) or 0
            dl_max = tdt.get('duty_length_max')
            wt_max = tdt.get('work_time_max')
            st_min = tdt.get('start_time_min')
            st_max = tdt.get('start_time_max')
            et_max = tdt.get('end_time_max')
            
            for i, p1 in enumerate(sorted_pieces):
                if len(duties) >= max_duties:
                    break
                if p1['duration'] < pl_min or p1['duration'] > pl_max:
                    continue
                duty_start = p1['start_time'] - sign_on
                if st_min is not None and duty_start < st_min:
                    continue
                if st_max is not None and duty_start > st_max:
                    continue
                
                for j in range(i + 1, len(sorted_pieces)):
                    if len(duties) >= max_duties:
                        break
                    p2 = sorted_pieces[j]
                    if p2['start_time'] - sign_on > (st_max if st_max is not None else float('inf')):
                        break
                    if p2['start_time'] <= p1['end_time']:
                        continue
                    if p2['duration'] < pl_min or p2['duration'] > pl_max:
                        continue
                    
                    # Check break
                    break_time = p2['start_time'] - p1['end_time']
                    if break_time < br_min:
                        continue
                    
                    # Check overlap
                    if p1['trip_set'] & p2['trip_set']:
                        continue
                    
                    # Check duty span
                    duty_end = p2['end_time'] + sign_off
                    if dl_max is not None:
                        if (duty_end - duty_start) > dl_max:
                            continue
                    
                    # Check work time
                    if wt_max is not None:
                        if (p1['duration'] + p2['duration']) > wt_max:
                            continue
                    
                    # Check end time
                    if et_max is not None and duty_end > et_max:
                        continue
                    
                    duties.append({
                        'type': tname,
                        'trips': p1['trips'] + p2['trips'],
                        'trip_set': p1['trip_set'] | p2['trip_set'],
                        'pieces': [p1, p2]
                    })
    
    return duties


def greedy_crew_scheduling(duties, num_trips):
    """Greedy set cover for crew scheduling."""
    uncovered = set(range(num_trips))
    selected = []
    
    # Sort duties by number of trips covered (descending), then by type preference
    sorted_duties = sorted(duties, key=lambda d: (-len(d['trip_set']), d['type']))
    
    for duty in sorted_duties:
        if not uncovered:
            break
        if duty['trip_set'] & uncovered:
            # Check if this duty only covers uncovered trips
            if duty['trip_set'] <= uncovered:
                selected.append(duty)
                uncovered -= duty['trip_set']
    
    # Handle remaining uncovered trips with individual duties
    if uncovered:
        # Create fallback duties for uncovered trips
        for t in uncovered:
            selected.append({
                'type': 'tripper',
                'trips': [t],
                'trip_set': {t},
                'pieces': []
            })
    
    return selected


def solve_with_gurobi(duties, num_trips, time_remaining):
    """Solve set partitioning with Gurobi."""
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError:
        return None
    
    try:
        model = gp.Model("crew_scheduling")
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1
        model.Params.TimeLimit = max(1, time_remaining)
        model.Params.OutputFlag = 0
        
        n_duties = len(duties)
        x = model.addVars(n_duties, vtype=GRB.BINARY, name="x")
        
        # Build coverage: for each trip, which duties cover it
        trip_to_duties = defaultdict(list)
        for d_idx, duty in enumerate(duties):
            for t in duty['trip_set']:
                trip_to_duties[t].append(d_idx)
        
        # Check feasibility
        for t in range(num_trips):
            if not trip_to_duties[t]:
                return None
        
        # Set partitioning constraints
        for t in range(num_trips):
            model.addConstr(
                gp.quicksum(x[d_idx] for d_idx in trip_to_duties[t]) == 1,
                name=f"cover_{t}"
            )
        
        # Objective: minimize number of duties
        model.setObjective(gp.quicksum(x[d_idx] for d_idx in range(n_duties)), GRB.MINIMIZE)
        
        model.optimize()
        
        if model.SolCount > 0:
            selected = [d_idx for d_idx in range(n_duties) if x[d_idx].X > 0.5]
            return selected
        
        return None
    except Exception:
        return None


def build_solution(blocks, selected_duties, num_trips, depots, trips):
    """Build the final solution dictionary."""
    depot_name = depots[0]['name']
    
    # Vehicle schedule: map depot name to list of trip chains
    vehicle_schedule = {depot_name: [list(block) for block in blocks]}
    
    # Crew schedule
    crew_schedule = []
    for duty in selected_duties:
        crew_schedule.append({
            'depot': depot_name,
            'type': duty['type'],
            'trips': duty['trips']
        })
    
    num_vehicles = len(blocks)
    num_duties = len(crew_schedule)
    objective_value = num_vehicles + num_duties
    
    solution = {
        'objective_value': objective_value,
        'vehicle_schedule': vehicle_schedule,
        'crew_schedule': crew_schedule
    }
    
    return solution


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    start_time = time.time()
    time_limit = args.time_limit
    
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    trips = instance['trips']
    num_trips = instance['num_trips']
    depots = instance['depots']
    num_depots = instance['num_depots']
    travel_times = instance['travel_times']
    relief_locations = set(instance['relief_locations'])
    duty_types = instance['duty_types']
    crew_params = instance['crew_parameters']
    
    if num_trips == 0:
        solution = {
            'objective_value': 0,
            'vehicle_schedule': {depots[0]['name']: []},
            'crew_schedule': []
        }
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f)
        if logger:
            logger.log_solution(0, solution)
        return
    
    # Step 1: Vehicle scheduling
    blocks = solve_vehicle_scheduling(trips, travel_times, num_trips)
    num_vehicles = len(blocks)
    
    elapsed = time.time() - start_time
    remaining = time_limit - elapsed
    
    if remaining <= 0:
        # Fallback: one duty per trip
        crew_schedule = []
        for t in range(num_trips):
            crew_schedule.append({
                'depot': depots[0]['name'],
                'type': list(duty_types.keys())[0],
                'trips': [t]
            })
        solution = {
            'objective_value': num_vehicles + num_trips,
            'vehicle_schedule': {depots[0]['name']: [list(b) for b in blocks]},
            'crew_schedule': crew_schedule
        }
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f)
        if logger:
            logger.log_solution(solution['objective_value'], solution)
        return
    
    # Step 2: Generate pieces
    all_pieces = generate_pieces(blocks, trips, relief_locations)
    
    # Step 3: Generate duties
    max_duties = 80000
    duties = generate_duties(all_pieces, duty_types, crew_params, num_trips, max_duties)
    
    # Ensure we have fallback duties for every trip
    covered_trips = set()
    for d in duties:
        covered_trips |= d['trip_set']
    
    # Add individual trip duties as fallback if needed
    for t in range(num_trips):
        if t not in covered_trips:
            # Try to find any duty type that can cover this single trip
            trip_duration = trips[t]['end_time'] - trips[t]['start_time']
            added = False
            for tname, tdt in duty_types.items():
                if tdt['num_pieces'] == 1:
                    if tdt['piece_length_min'] <= trip_duration <= tdt['piece_length_max']:
                        duties.append({
                            'type': tname,
                            'trips': [t],
                            'trip_set': {t},
                            'pieces': []
                        })
                        added = True
                        break
            if not added:
                # Use first available type as fallback
                first_type = list(duty_types.keys())[0]
                duties.append({
                    'type': first_type,
                    'trips': [t],
                    'trip_set': {t},
                    'pieces': []
                })
    
    elapsed = time.time() - start_time
    remaining = time_limit - elapsed
    
    # Step 4: Solve crew scheduling
    selected_indices = None
    
    if remaining > 5 and len(duties) <= 80000:
        selected_indices = solve_with_gurobi(duties, num_trips, remaining - 2)
    
    if selected_indices is not None:
        selected_duties = [duties[i] for i in selected_indices]
    else:
        selected_duties = greedy_crew_scheduling(duties, num_trips)
    
    # Verify all trips are covered
    covered = set()
    for d in selected_duties:
        covered |= d['trip_set']
    
    # If some trips are not covered, add fallback
    uncovered = set(range(num_trips)) - covered
    for t in uncovered:
        first_type = list(duty_types.keys())[0]
        selected_duties.append({
            'type': first_type,
            'trips': [t],
            'trip_set': {t},
            'pieces': []
        })
    
    # Step 5: Build and output solution
    solution = build_solution(blocks, selected_duties, num_trips, depots, trips)
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)
    
    if logger:
        logger.log_solution(solution['objective_value'], solution)
    
    # Try to improve if time remains
    elapsed = time.time() - start_time
    remaining = time_limit - elapsed
    
    if remaining > 10:
        # Try alternative vehicle scheduling with slightly more vehicles
        # to potentially reduce crew duties
        improved_solution = try_improve(blocks, trips, travel_times, num_trips,
                                         relief_locations, duty_types, crew_params,
                                         depots, remaining - 2, solution['objective_value'])
        if improved_solution is not None and improved_solution['objective_value'] < solution['objective_value']:
            solution = improved_solution
            with open(args.solution_path, 'w') as f:
                json.dump(solution, f)
            if logger:
                logger.log_solution(solution['objective_value'], solution)


def try_improve(blocks, trips, travel_times, num_trips, relief_locations,
                duty_types, crew_params, depots, time_remaining, current_obj):
    """Try to improve by using different vehicle block structures."""
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError:
        return None
    
    try:
        # Try a MIP for integrated vehicle scheduling
        # that minimizes vehicles + estimated crew cost
        model = gp.Model("vsp_mip")
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1
        model.Params.TimeLimit = max(1, min(time_remaining, 60))
        model.Params.OutputFlag = 0
        
        n = num_trips
        
        # Build compatibility
        compat = []
        for i in range(n):
            for j in range(i + 1, n):
                if trips[i]['end_time'] <= trips[j]['start_time']:
                    tt = travel_times[trips[i]['end_location']][trips[j]['start_location']]
                    if trips[i]['end_time'] + tt <= trips[j]['start_time']:
                        compat.append((i, j))
        
        if not compat:
            return None
        
        # Variables
        x = {}
        for (i, j) in compat:
            x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
        
        # Each trip has at most one successor
        for i in range(n):
            succs = [(i, j) for (ii, j) in compat if ii == i]
            if succs:
                model.addConstr(gp.quicksum(x[e] for e in succs) <= 1)
        
        # Each trip has at most one predecessor
        for j in range(n):
            preds = [(i, j) for (i, jj) in compat if jj == j]
            if preds:
                model.addConstr(gp.quicksum(x[e] for e in preds) <= 1)
        
        # Objective: maximize connections (= minimize vehicles)
        model.setObjective(gp.quicksum(x[e] for e in compat), GRB.MAXIMIZE)
        
        model.optimize()
        
        if model.SolCount > 0 and model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
            # Extract matching
            pair_left = [-1] * n
            for (i, j) in compat:
                if (i, j) in x and x[i, j].X > 0.5:
                    pair_left[i] = j
            
            # Extract blocks
            has_pred = [False] * n
            for i in range(n):
                if pair_left[i] != -1:
                    has_pred[pair_left[i]] = True
            
            new_blocks = []
            for i in range(n):
                if not has_pred[i]:
                    block = [i]
                    current = i
                    while pair_left[current] != -1:
                        current = pair_left[current]
                        block.append(current)
                    new_blocks.append(block)
            
            # Now solve crew scheduling for new blocks
            all_pieces = generate_pieces(new_blocks, trips, relief_locations)
            duties = generate_duties(all_pieces, duty_types, crew_params, num_trips, 50000)
            
            # Add fallback
            covered_trips = set()
            for d in duties:
                covered_trips |= d['trip_set']
            for t in range(n):
                if t not in covered_trips:
                    first_type = list(duty_types.keys())[0]
                    duties.append({
                        'type': first_type,
                        'trips': [t],
                        'trip_set': {t},
                        'pieces': []
                    })
            
            remaining_t = time_remaining - (time.time() - time.time())  # approximate
            selected_indices = solve_with_gurobi(duties, n, max(1, time_remaining / 2))
            
            if selected_indices is not None:
                selected_duties = [duties[i] for i in selected_indices]
            else:
                selected_duties = greedy_crew_scheduling(duties, n)
            
            # Verify coverage
            covered = set()
            for d in selected_duties:
                covered |= d['trip_set']
            for t in set(range(n)) - covered:
                first_type = list(duty_types.keys())[0]
                selected_duties.append({
                    'type': first_type,
                    'trips': [t],
                    'trip_set': {t},
                    'pieces': []
                })
            
            solution = build_solution(new_blocks, selected_duties, n, depots, trips)
            if solution['objective_value'] < current_obj:
                return solution
        
        return None
    except Exception:
        return None


if __name__ == '__main__':
    main()
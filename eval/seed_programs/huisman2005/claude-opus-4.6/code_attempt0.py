import argparse
import json
import time
import sys
from itertools import combinations
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    start_time = time.time()

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    num_trips = data['num_trips']
    num_depots = data['num_depots']
    depots = data['depots']
    trips = data['trips']
    travel_times = data['travel_times']
    duty_types = data['duty_types']
    crew_params = data['crew_parameters']
    vehicle_params = data['vehicle_parameters']
    relief_locations = data['relief_locations']

    sign_on = crew_params['sign_on_time_depot_minutes']
    sign_off = crew_params['sign_off_time_depot_minutes']
    extra_time_non_depot = crew_params['extra_time_non_depot_relief_minutes']
    extra_includes_deadhead = crew_params['extra_time_includes_deadhead_to_depot']
    max_pieces = crew_params['max_pieces_per_duty']
    continuous_attendance = crew_params['continuous_attendance']
    changeovers_allowed = crew_params['changeovers_allowed']
    fict_cost = vehicle_params['fictitious_cost_per_minute_empty_outside_depot']

    depot_names = [d['name'] for d in depots]

    # Sort trips by start_time
    trips_sorted = sorted(trips, key=lambda t: t['start_time'])
    trip_by_id = {t['trip_id']: t for t in trips_sorted}

    def get_travel_time(loc1, loc2):
        if loc1 == loc2:
            return 0
        return travel_times.get(loc1, {}).get(loc2, 9999)

    # For each depot, determine which trip pairs can be consecutive on the same vehicle
    # Two trips i, j (i before j) are compatible if:
    #   trip_i.end_time + travel_time(trip_i.end_location, trip_j.start_location) <= trip_j.start_time

    def is_short_arc(trip_i, trip_j, depot_name):
        """Check if vehicle stays outside depot between trips i and j"""
        dh_time = get_travel_time(trip_i['end_location'], trip_j['start_location'])
        if trip_i['end_time'] + dh_time > trip_j['start_time']:
            return False  # not compatible at all
        # Check if vehicle can return to depot and come back
        time_to_depot = get_travel_time(trip_i['end_location'], depot_name)
        time_from_depot = get_travel_time(depot_name, trip_j['start_location'])
        gap = trip_j['start_time'] - trip_i['end_time']
        if gap >= time_to_depot + time_from_depot:
            return False  # long arc - vehicle returns to depot
        return True  # short arc - vehicle stays outside

    def are_compatible(trip_i, trip_j):
        """Check if trip_j can follow trip_i on the same vehicle"""
        dh_time = get_travel_time(trip_i['end_location'], trip_j['start_location'])
        return trip_i['end_time'] + dh_time <= trip_j['start_time']

    # Build a greedy vehicle schedule for each depot configuration
    # Then build crew schedules on top

    # Strategy: Use a greedy approach to build vehicle chains, then greedily assign crew duties

    def build_vehicle_schedule_greedy(depot_name, assigned_trips):
        """Build vehicle chains for a given depot using greedy approach"""
        if not assigned_trips:
            return []
        
        sorted_trips = sorted(assigned_trips, key=lambda t: t['start_time'])
        vehicles = []  # Each vehicle is a list of trips
        
        for trip in sorted_trips:
            best_vehicle = None
            best_idle = float('inf')
            
            for i, vehicle in enumerate(vehicles):
                last_trip = vehicle[-1]
                if are_compatible(last_trip, trip):
                    idle = trip['start_time'] - last_trip['end_time']
                    if idle < best_idle:
                        best_idle = idle
                        best_vehicle = i
            
            if best_vehicle is not None:
                vehicles[best_vehicle].append(trip)
            else:
                vehicles.append([trip])
        
        return vehicles

    def compute_piece_duration(task_trips):
        """Compute duration of a piece of work (sequence of consecutive tasks on one vehicle)"""
        if not task_trips:
            return 0
        first = task_trips[0]
        last = task_trips[-1]
        return last['end_time'] - first['start_time']

    def check_duty_feasibility(pieces, duty_type, depot_name):
        """Check if a duty with given pieces is feasible for a duty type"""
        dt = duty_type
        num_pieces_required = dt['num_pieces']
        
        if len(pieces) != num_pieces_required:
            return False
        
        # Check piece lengths
        for piece in pieces:
            if not piece:
                return False
            duration = piece[-1]['end_time'] - piece[0]['start_time']
            if duration < dt['piece_length_min'] or duration > dt['piece_length_max']:
                return False
        
        # Compute duty start and end times with sign-on/sign-off
        first_piece = pieces[0]
        last_piece = pieces[-1]
        
        first_trip = first_piece[0]
        last_trip = last_piece[-1]
        
        # Sign-on time
        first_loc = first_trip['start_location']
        if first_loc == depot_name:
            duty_start = first_trip['start_time'] - sign_on
        elif first_loc in relief_locations:
            if extra_includes_deadhead:
                duty_start = first_trip['start_time'] - extra_time_non_depot
            else:
                dh = get_travel_time(depot_name, first_loc)
                duty_start = first_trip['start_time'] - extra_time_non_depot - dh
        else:
            duty_start = first_trip['start_time'] - sign_on
        
        # Sign-off time
        last_loc = last_trip['end_location']
        if last_loc == depot_name:
            duty_end = last_trip['end_time'] + sign_off
        elif last_loc in relief_locations:
            if extra_includes_deadhead:
                duty_end = last_trip['end_time'] + extra_time_non_depot
            else:
                dh = get_travel_time(last_loc, depot_name)
                duty_end = last_trip['end_time'] + extra_time_non_depot + dh
        else:
            duty_end = last_trip['end_time'] + sign_off
        
        # Check start/end time constraints
        if dt.get('start_time_min') is not None and duty_start < dt['start_time_min']:
            return False
        if dt.get('start_time_max') is not None and duty_start > dt['start_time_max']:
            return False
        if dt.get('end_time_max') is not None and duty_end > dt['end_time_max']:
            return False
        
        # Check duty length
        duty_length = duty_end - duty_start
        if dt.get('duty_length_max') is not None and duty_length > dt['duty_length_max']:
            return False
        
        # Check work time
        work_time = sum(p[-1]['end_time'] - p[0]['start_time'] for p in pieces)
        # Add sign-on/sign-off and deadhead times
        total_work = work_time + (first_trip['start_time'] - duty_start) + (duty_end - last_trip['end_time'])
        if dt.get('work_time_max') is not None and total_work > dt['work_time_max']:
            return False
        
        # Check break between pieces
        if len(pieces) == 2:
            break_start = pieces[0][-1]['end_time']
            break_end = pieces[1][0]['start_time']
            break_duration = break_end - break_start
            if dt.get('break_length_min') is not None and break_duration < dt['break_length_min']:
                return False
        
        return True

    def find_duty_type(pieces, depot_name):
        """Find a valid duty type for given pieces"""
        for dt_name, dt in duty_types.items():
            if check_duty_feasibility(pieces, dt, depot_name):
                return dt_name
        return None

    def build_crew_schedule_for_depot(depot_name, vehicle_chains):
        """Build crew duties for vehicles assigned to a depot"""
        duties = []
        uncovered_trips = set()
        
        # For each vehicle chain, try to split into duties
        # Collect all "pieces of work" - segments between relief points
        
        all_pieces = []  # (vehicle_idx, piece_trips)
        
        for v_idx, chain in enumerate(vehicle_chains):
            if not chain:
                continue
            
            # Split chain into pieces at relief points
            # A piece is a maximal sequence of consecutive tasks bounded by relief points
            # For simplicity, treat each trip as a task, and breaks can happen at trip boundaries
            # where both the end location of the previous trip and start location of next trip are relief locations
            
            # For now, try to form pieces from consecutive trips
            # Relief can happen at any trip boundary where end_loc or start_loc is a relief location
            
            # Find possible break points (indices where a break can occur)
            break_points = []
            for i in range(len(chain) - 1):
                end_loc = chain[i]['end_location']
                start_loc = chain[i+1]['start_location']
                # A break can occur here if both locations allow relief
                if end_loc in relief_locations or end_loc == depot_name:
                    if start_loc in relief_locations or start_loc == depot_name:
                        break_points.append(i)
            
            # Try to form duties from the chain
            # Each duty has 1 or 2 pieces
            # Try to cover all trips in the chain with minimal duties
            
            all_pieces.append((v_idx, chain, break_points))
        
        # Simple approach: for each vehicle chain, greedily form duties
        for v_idx, chain, break_points in all_pieces:
            remaining = list(range(len(chain)))
            
            while remaining:
                start_idx = remaining[0]
                best_duty = None
                best_end = start_idx
                
                # Try single piece (tripper) duty
                for end_idx in range(start_idx, min(start_idx + len(chain), len(chain))):
                    if end_idx not in remaining:
                        break
                    piece = chain[start_idx:end_idx+1]
                    dt_name = find_duty_type([piece], depot_name)
                    if dt_name is not None:
                        best_duty = ([piece], dt_name)
                        best_end = end_idx
                
                # Try two-piece duty with break at a break point
                for bp in break_points:
                    if bp < start_idx:
                        continue
                    # Piece 1: start_idx to bp
                    # Piece 2: bp+1 to some end
                    if bp < start_idx or bp >= len(chain) - 1:
                        continue
                    piece1 = chain[start_idx:bp+1]
                    all_in_remaining_1 = all(i in remaining for i in range(start_idx, bp+1))
                    if not all_in_remaining_1:
                        continue
                    
                    for end_idx in range(bp+1, len(chain)):
                        all_in_remaining_2 = all(i in remaining for i in range(bp+1, end_idx+1))
                        if not all_in_remaining_2:
                            break
                        piece2 = chain[bp+1:end_idx+1]
                        dt_name = find_duty_type([piece1, piece2], depot_name)
                        if dt_name is not None:
                            if end_idx > best_end or best_duty is None:
                                best_duty = ([piece1, piece2], dt_name)
                                best_end = end_idx
                
                if best_duty is not None:
                    pieces, dt_name = best_duty
                    trip_ids = []
                    for piece in pieces:
                        for t in piece:
                            trip_ids.append(t['trip_id'])
                    duties.append({
                        'depot': depot_name,
                        'type': dt_name,
                        'trips': trip_ids
                    })
                    for i in range(start_idx, best_end + 1):
                        if i in remaining:
                            remaining.remove(i)
                else:
                    # Fallback: assign single trip as tripper if possible, otherwise use any type
                    trip = chain[start_idx]
                    trip_ids = [trip['trip_id']]
                    # Try tripper
                    dt_name = find_duty_type([[trip]], depot_name)
                    if dt_name is None:
                        # Use first available duty type that might work
                        dt_name = 'tripper'  # fallback
                    duties.append({
                        'depot': depot_name,
                        'type': dt_name,
                        'trips': trip_ids
                    })
                    remaining.remove(start_idx)
        
        return duties

    # More sophisticated crew scheduling with changeovers
    def build_crew_schedule_advanced(depot_name, vehicle_chains):
        """Build crew duties allowing changeovers between vehicles"""
        duties = []
        
        # Collect all individual trips with their vehicle assignment
        trip_vehicle_map = {}
        for v_idx, chain in enumerate(vehicle_chains):
            for t in chain:
                trip_vehicle_map[t['trip_id']] = v_idx
        
        # Build pieces of work for each vehicle
        # A piece of work is a sequence of consecutive trips on one vehicle
        # We need to find good break points within each vehicle chain
        
        # First, identify all possible relief points in each chain
        vehicle_pieces = {}  # v_idx -> list of possible pieces (each piece is a list of trips)
        
        for v_idx, chain in enumerate(vehicle_chains):
            if not chain:
                continue
            
            # Find all valid break points
            break_points = set()
            break_points.add(-1)  # before first trip
            break_points.add(len(chain) - 1)  # after last trip
            
            for i in range(len(chain) - 1):
                end_loc = chain[i]['end_location']
                start_loc = chain[i+1]['start_location']
                if (end_loc in relief_locations or end_loc in depot_names):
                    if (start_loc in relief_locations or start_loc in depot_names):
                        break_points.add(i)
            
            vehicle_pieces[v_idx] = (chain, sorted(break_points))
        
        # For each vehicle, generate candidate pieces
        candidate_pieces = []  # (v_idx, start_trip_idx, end_trip_idx, trips, duration)
        
        for v_idx, (chain, bps) in vehicle_pieces.items():
            for i in range(len(bps)):
                for j in range(i + 1, len(bps)):
                    bp_start = bps[i]
                    bp_end = bps[j]
                    piece_trips = chain[bp_start + 1:bp_end + 1]
                    if not piece_trips:
                        continue
                    duration = piece_trips[-1]['end_time'] - piece_trips[0]['start_time']
                    candidate_pieces.append((v_idx, bp_start + 1, bp_end, piece_trips, duration))
        
        # Now we need to cover all trips using duties
        # Each duty consists of 1 or 2 pieces
        # With changeovers allowed, pieces in a duty can be from different vehicles
        
        # Simple greedy: for each vehicle, split into pieces that fit tripper or normal duty types
        # Then try to combine pieces from different vehicles into 2-piece duties
        
        # First pass: try to cover as many trips as possible with 2-piece normal duties
        # This is complex, so let's use a simpler approach
        
        # Greedy approach per vehicle: split each chain into pieces, then form duties
        covered = set()
        pending_pieces = []  # pieces waiting to be paired
        
        for v_idx, (chain, bps) in vehicle_pieces.items():
            # Generate maximal pieces that fit within piece_length_max of some duty type
            max_piece_len = max(dt['piece_length_max'] for dt in duty_types.values())
            min_piece_len = min(dt['piece_length_min'] for dt in duty_types.values())
            
            # Greedily form pieces
            pieces_for_vehicle = []
            current_start = 0
            
            while current_start < len(chain):
                # Find the longest piece starting from current_start that fits within max_piece_len
                best_end = current_start
                
                for bp in bps:
                    if bp < current_start:
                        continue
                    piece = chain[current_start:bp + 1]
                    if not piece:
                        continue
                    duration = piece[-1]['end_time'] - piece[0]['start_time']
                    if duration <= max_piece_len:
                        best_end = bp
                    else:
                        break
                
                piece = chain[current_start:best_end + 1]
                if piece:
                    pieces_for_vehicle.append(piece)
                    current_start = best_end + 1
                else:
                    # Single trip piece
                    pieces_for_vehicle.append([chain[current_start]])
                    current_start += 1
            
            # Try to form duties from these pieces
            i = 0
            while i < len(pieces_for_vehicle):
                piece1 = pieces_for_vehicle[i]
                
                # Try as single-piece (tripper) duty
                dt_name = find_duty_type([piece1], depot_name)
                if dt_name is not None:
                    trip_ids = [t['trip_id'] for t in piece1]
                    duties.append({
                        'depot': depot_name,
                        'type': dt_name,
                        'trips': trip_ids
                    })
                    for t in piece1:
                        covered.add(t['trip_id'])
                    i += 1
                    continue
                
                # Try to pair with next piece as 2-piece duty
                if i + 1 < len(pieces_for_vehicle):
                    piece2 = pieces_for_vehicle[i + 1]
                    dt_name = find_duty_type([piece1, piece2], depot_name)
                    if dt_name is not None:
                        trip_ids = [t['trip_id'] for t in piece1] + [t['trip_id'] for t in piece2]
                        duties.append({
                            'depot': depot_name,
                            'type': dt_name,
                            'trips': trip_ids
                        })
                        for t in piece1:
                            covered.add(t['trip_id'])
                        for t in piece2:
                            covered.add(t['trip_id'])
                        i += 2
                        continue
                
                # Add to pending for cross-vehicle pairing
                pending_pieces.append(piece1)
                for t in piece1:
                    covered.add(t['trip_id'])
                i += 1
        
        # Handle pending pieces - try to pair them into 2-piece duties
        pending_pieces.sort(key=lambda p: p[0]['start_time'])
        used_pending = [False] * len(pending_pieces)
        
        for i in range(len(pending_pieces)):
            if used_pending[i]:
                continue
            piece1 = pending_pieces[i]
            
            # Try to pair with another pending piece
            paired = False
            for j in range(i + 1, len(pending_pieces)):
                if used_pending[j]:
                    continue
                piece2 = pending_pieces[j]
                dt_name = find_duty_type([piece1, piece2], depot_name)
                if dt_name is not None:
                    trip_ids = [t['trip_id'] for t in piece1] + [t['trip_id'] for t in piece2]
                    duties.append({
                        'depot': depot_name,
                        'type': dt_name,
                        'trips': trip_ids
                    })
                    used_pending[i] = True
                    used_pending[j] = True
                    paired = True
                    break
            
            if not paired:
                # Single piece duty - use tripper or fallback
                dt_name = find_duty_type([piece1], depot_name)
                if dt_name is None:
                    dt_name = 'tripper'
                trip_ids = [t['trip_id'] for t in piece1]
                duties.append({
                    'depot': depot_name,
                    'type': dt_name,
                    'trips': trip_ids
                })
                used_pending[i] = True
        
        # Check for uncovered trips
        all_trip_ids = set()
        for chain in vehicle_chains:
            for t in chain:
                all_trip_ids.add(t['trip_id'])
        
        for tid in all_trip_ids:
            if tid not in covered:
                trip = trip_by_id[tid]
                dt_name = find_duty_type([[trip]], depot_name)
                if dt_name is None:
                    dt_name = 'tripper'
                duties.append({
                    'depot': depot_name,
                    'type': dt_name,
                    'trips': [tid]
                })
        
        return duties

    # Main solving logic
    # Try different depot assignments to minimize vehicles + duties
    
    elapsed = time.time() - start_time
    remaining_time = args.time_limit - elapsed
    
    if num_depots == 1:
        # Single depot - assign all trips to it
        depot_name = depot_names[0]
        all_trips = [trip_by_id[t['trip_id']] for t in trips_sorted]
        vehicle_chains = build_vehicle_schedule_greedy(depot_name, all_trips)
        
        num_vehicles = len(vehicle_chains)
        
        # Build crew schedule
        crew_duties = build_crew_schedule_advanced(depot_name, vehicle_chains)
        num_duties = len(crew_duties)
        
        vehicle_schedule = {depot_name: [[t['trip_id'] for t in chain] for chain in vehicle_chains]}
        
        objective = num_vehicles + num_duties
        
        if logger:
            logger.log(objective)
        
        solution = {
            'objective_value': objective,
            'vehicle_schedule': vehicle_schedule,
            'crew_schedule': crew_duties
        }
    else:
        # Multiple depots - try to assign trips to depots intelligently
        # Simple approach: assign each trip to the nearest depot
        
        best_solution = None
        best_obj = float('inf')
        
        # Strategy 1: Assign all to each depot individually
        for depot_name in depot_names:
            all_trips_list = [trip_by_id[t['trip_id']] for t in trips_sorted]
            vehicle_chains = build_vehicle_schedule_greedy(depot_name, all_trips_list)
            crew_duties = build_crew_schedule_advanced(depot_name, vehicle_chains)
            
            num_vehicles = len(vehicle_chains)
            num_duties = len(crew_duties)
            obj = num_vehicles + num_duties
            
            if obj < best_obj:
                best_obj = obj
                vehicle_schedule = {dn: [] for dn in depot_names}
                vehicle_schedule[depot_name] = [[t['trip_id'] for t in chain] for chain in vehicle_chains]
                best_solution = {
                    'objective_value': obj,
                    'vehicle_schedule': vehicle_schedule,
                    'crew_schedule': crew_duties
                }
                if logger:
                    logger.log(obj)
        
        # Strategy 2: Assign trips to nearest depot
        depot_trip_assignment = defaultdict(list)
        for trip in trips_sorted:
            best_depot = None
            best_dist = float('inf')
            for d in depots:
                d_name = d['name']
                # Consider travel time from depot to trip start and from trip end to depot
                t_to = get_travel_time(d_name, trip['start_location'])
                t_from = get_travel_time(trip['end_location'], d_name)
                dist = t_to + t_from
                if dist < best_dist:
                    best_dist = dist
                    best_depot = d_name
            depot_trip_assignment[best_depot].append(trip)
        
        total_vehicles = 0
        total_duties = 0
        vehicle_schedule = {dn: [] for dn in depot_names}
        all_crew_duties = []
        
        for depot_name in depot_names:
            assigned = depot_trip_assignment.get(depot_name, [])
            if not assigned:
                continue
            vehicle_chains = build_vehicle_schedule_greedy(depot_name, assigned)
            crew_duties = build_crew_schedule_advanced(depot_name, vehicle_chains)
            
            vehicle_schedule[depot_name] = [[t['trip_id'] for t in chain] for chain in vehicle_chains]
            all_crew_duties.extend(crew_duties)
            total_vehicles += len(vehicle_chains)
            total_duties += len(crew_duties)
        
        obj = total_vehicles + total_duties
        if obj < best_obj:
            best_obj = obj
            best_solution = {
                'objective_value': obj,
                'vehicle_schedule': vehicle_schedule,
                'crew_schedule': all_crew_duties
            }
            if logger:
                logger.log(obj)
        
        solution = best_solution
    
    # Now try to improve the solution with more sophisticated crew scheduling
    # Try to reduce the number of duties by better pairing pieces
    
    elapsed = time.time() - start_time
    remaining_time = args.time_limit - elapsed
    
    if remaining_time > 5:
        # Try to improve crew schedule by re-splitting vehicle chains
        improved = try_improve_crew_schedule(solution, data, depot_names, trip_by_id, 
                                              duty_types, relief_locations, depot_names,
                                              sign_on, sign_off, extra_time_non_depot,
                                              extra_includes_deadhead, travel_times,
                                              get_travel_time, find_duty_type,
                                              args.time_limit - (time.time() - start_time))
        if improved is not None and improved['objective_value'] < solution['objective_value']:
            solution = improved
            if logger:
                logger.log(solution['objective_value'])
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


def try_improve_crew_schedule(solution, data, depot_names, trip_by_id, duty_types, 
                               relief_locations, all_depot_names, sign_on, sign_off,
                               extra_time_non_depot, extra_includes_deadhead, travel_times,
                               get_travel_time, find_duty_type, time_remaining):
    """Try to improve crew schedule by better piece formation and pairing"""
    
    start = time.time()
    
    vehicle_schedule = solution['vehicle_schedule']
    best_solution = dict(solution)
    
    for depot_name in depot_names:
        chains_ids = vehicle_schedule.get(depot_name, [])
        if not chains_ids:
            continue
        
        # Reconstruct chains as trip objects
        chains = []
        for chain_ids in chains_ids:
            chain = [trip_by_id[tid] for tid in chain_ids]
            chains.append(chain)
        
        # Try different piece splitting strategies
        all_pieces = []
        
        for v_idx, chain in enumerate(chains):
            # Find all possible break points
            break_indices = set()
            break_indices.add(-1)
            break_indices.add(len(chain) - 1)
            
            for i in range(len(chain) - 1):
                end_loc = chain[i]['end_location']
                start_loc = chain[i+1]['start_location']
                if (end_loc in relief_locations or end_loc in all_depot_names):
                    if (start_loc in relief_locations or start_loc in all_depot_names):
                        break_indices.add(i)
            
            break_indices = sorted(break_indices)
            
            # Generate all valid pieces
            pieces = []
            for i in range(len(break_indices)):
                for j in range(i + 1, len(break_indices)):
                    bp_start = break_indices[i]
                    bp_end = break_indices[j]
                    piece = chain[bp_start + 1:bp_end + 1]
                    if not piece:
                        continue
                    duration = piece[-1]['end_time'] - piece[0]['start_time']
                    # Check if this piece duration fits any duty type
                    max_len = max(dt['piece_length_max'] for dt in duty_types.values())
                    if duration <= max_len:
                        pieces.append({
                            'v_idx': v_idx,
                            'start': bp_start + 1,
                            'end': bp_end,
                            'trips': piece,
                            'duration': duration,
                            'trip_ids': set(t['trip_id'] for t in piece)
                        })
            
            all_pieces.extend(pieces)
        
        if time.time() - start > time_remaining * 0.8:
            break
        
        # Use a greedy set cover approach to cover all trips with minimum duties
        all_trip_ids = set()
        for chain_ids in chains_ids:
            all_trip_ids.update(chain_ids)
        
        # Generate candidate duties (single piece and two piece)
        candidate_duties = []
        
        # Single piece duties
        for p in all_pieces:
            dt_name = find_duty_type([p['trips']], depot_name)
            if dt_name is not None:
                candidate_duties.append({
                    'pieces': [p],
                    'type': dt_name,
                    'trip_ids': p['trip_ids'],
                    'trips': [t['trip_id'] for t in p['trips']]
                })
        
        # Two piece duties (limit combinations for performance)
        if len(all_pieces) < 500:
            pieces_by_end = sorted(all_pieces, key=lambda p: p['trips'][-1]['end_time'])
            
            for i, p1 in enumerate(pieces_by_end):
                if time.time() - start > time_remaining * 0.5:
                    break
                for j in range(i + 1, len(pieces_by_end)):
                    p2 = pieces_by_end[j]
                    
                    # Pieces must not overlap in trips
                    if p1['trip_ids'] & p2['trip_ids']:
                        continue
                    
                    # p1 must end before p2 starts
                    if p1['trips'][-1]['end_time'] >= p2['trips'][0]['start_time']:
                        continue
                    
                    dt_name = find_duty_type([p1['trips'], p2['trips']], depot_name)
                    if dt_name is not None:
                        combined_ids = p1['trip_ids'] | p2['trip_ids']
                        candidate_duties.append({
                            'pieces': [p1, p2],
                            'type': dt_name,
                            'trip_ids': combined_ids,
                            'trips': [t['trip_id'] for t in p1['trips']] + [t['trip_id'] for t in p2['trips']]
                        })
        
        if not candidate_duties:
            continue
        
        # Greedy set cover
        covered = set()
        selected_duties = []
        
        # Sort candidates by coverage (more trips covered = better), then by fewer pieces
        candidate_duties.sort(key=lambda d: (-len(d['trip_ids']), len(d['pieces'])))
        
        while covered != all_trip_ids:
            best_cand = None
            best_new = 0
            
            for cand in candidate_duties:
                new_covered = len(cand['trip_ids'] - covered)
                if new_covered > best_new:
                    best_new = new_covered
                    best_cand = cand
                    if new_covered == len(cand['trip_ids']):
                        break  # This candidate covers only new trips
            
            if best_cand is None or best_new == 0:
                # Cover remaining trips individually
                for tid in all_trip_ids - covered:
                    trip = trip_by_id[tid]
                    dt_name = find_duty_type([[trip]], depot_name)
                    if dt_name is None:
                        dt_name = 'tripper'
                    selected_duties.append({
                        'depot': depot_name,
                        'type': dt_name,
                        'trips': [tid]
                    })
                    covered.add(tid)
                break
            
            selected_duties.append({
                'depot': depot_name,
                'type': best_cand['type'],
                'trips': best_cand['trips']
            })
            covered.update(best_cand['trip_ids'])
            
            # Remove candidates that overlap with newly covered trips
            candidate_duties = [c for c in candidate_duties 
                              if not (c['trip_ids'] & best_cand['trip_ids'])]
            
            if time.time() - start > time_remaining * 0.8:
                # Cover remaining
                for tid in all_trip_ids - covered:
                    trip = trip_by_id[tid]
                    dt_name = find_duty_type([[trip]], depot_name)
                    if dt_name is None:
                        dt_name = 'tripper'
                    selected_duties.append({
                        'depot': depot_name,
                        'type': dt_name,
                        'trips': [tid]
                    })
                    covered.add(tid)
                break
        
        # Compare with current solution
        current_depot_duties = [d for d in solution['crew_schedule'] if d['depot'] == depot_name]
        
        if len(selected_duties) < len(current_depot_duties):
            # Replace crew schedule for this depot
            new_crew = [d for d in best_solution['crew_schedule'] if d['depot'] != depot_name]
            new_crew.extend(selected_duties)
            
            num_vehicles = sum(len(v) for v in vehicle_schedule.values())
            num_duties = len(new_crew)
            new_obj = num_vehicles + num_duties
            
            if new_obj < best_solution['objective_value']:
                best_solution = {
                    'objective_value': new_obj,
                    'vehicle_schedule': vehicle_schedule,
                    'crew_schedule': new_crew
                }
    
    if best_solution['objective_value'] < solution['objective_value']:
        return best_solution
    return None


if __name__ == '__main__':
    main()
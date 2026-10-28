import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
from collections import defaultdict
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    start_time = time.time()
    num_trips = instance['num_trips']
    depots = instance['depots']
    depot_names = [d['name'] for d in depots]
    num_depots = len(depots)
    trips = instance['trips']
    travel_times = instance['travel_times']
    duty_types = instance['duty_types']
    crew_params = instance['crew_parameters']
    vehicle_params = instance['vehicle_parameters']
    relief_locations = set(instance['relief_locations'])
    locations = instance['locations']

    # Map depot id to name
    depot_id_to_name = {d['depot_id']: d['name'] for d in depots}
    depot_name_to_id = {d['name']: d['depot_id'] for d in depots}

    # Precompute compatible connections for vehicles
    # For each pair of trips i < j, can j follow i directly (with deadhead)
    can_follow = [[False] * num_trips for _ in range(num_trips)]
    travel_time_matrix = {}
    all_locs = list(locations.keys())
    for o in all_locs:
        for d in all_locs:
            travel_time_matrix[(o, d)] = travel_times.get(o, {}).get(d, 9999)

    for i in range(num_trips):
        for j in range(i + 1, num_trips):
            if trips[i]['end_time'] + travel_time_matrix[(trips[i]['end_location'], trips[j]['start_location'])] <= trips[j]['start_time']:
                can_follow[i][j] = True

    # Also can start from depot or end to depot - but modeled via source/sink

    # Simple construction heuristic: assign trips to depots greedily, then group into vehicles and duties
    # Since changeovers allowed and continuous attendance false in many cases, we can have loose coupling

    # Phase 1: assign each trip to the closest depot (by deadhead time)
    trip_to_depot = [-1] * num_trips
    depot_for_trip = {}
    for t_idx, trip in enumerate(trips):
        best_depot = 0
        best_time = 9999
        s_loc = trip['start_location']
        e_loc = trip['end_location']
        for d_idx, dname in enumerate(depot_names):
            t1 = travel_time_matrix[(dname, s_loc)]
            t2 = travel_time_matrix[(e_loc, dname)]
            total = t1 + t2
            if total < best_time:
                best_time = total
                best_depot = d_idx
        trip_to_depot[t_idx] = best_depot
        depot_for_trip[t_idx] = depot_names[best_depot]

    # Phase 2: build vehicle schedules per depot - greedy chaining with long arc preference
    vehicle_schedules = {dname: [] for dname in depot_names}
    for d_idx, dname in enumerate(depot_names):
        # Get trips for this depot
        my_trips = [i for i in range(num_trips) if trip_to_depot[i] == d_idx]
        my_trips.sort(key=lambda x: trips[x]['start_time'])
        used = [False] * num_trips
        chains = []
        for start_idx in my_trips:
            if used[start_idx]:
                continue
            chain = [start_idx]
            used[start_idx] = True
            current = start_idx
            while True:
                next_trip = None
                best_end = 999999
                for j in range(current + 1, num_trips):
                    if not used[j] and trip_to_depot[j] == d_idx and can_follow[current][j]:
                        if trips[j]['end_time'] < best_end:
                            best_end = trips[j]['end_time']
                            next_trip = j
                if next_trip is None:
                    break
                chain.append(next_trip)
                used[next_trip] = True
                current = next_trip
            chains.append(chain)
        vehicle_schedules[dname] = chains

    num_vehicles = sum(len(chains) for chains in vehicle_schedules.values())

    # Phase 3: build crew duties - for simplicity, one duty per vehicle chain if possible, else split
    # But to minimize total, we try to cover as many trips as possible with fewer duties using max piece length
    crew_duties = []
    # Simple approach: treat each vehicle chain as one piece, assign to tripper or normal duties
    duty_type_list = list(duty_types.keys())
    # Prefer 'tripper' for short, 'day' for long
    for dname in depot_names:
        for chain in vehicle_schedules[dname]:
            if not chain:
                continue
            # Compute approximate duration
            first_start = trips[chain[0]]['start_time']
            last_end = trips[chain[-1]]['end_time']
            duration = last_end - first_start
            if duration <= 240 and duty_types.get('tripper'):
                # Use tripper
                crew_duties.append({
                    "depot": dname,
                    "type": "tripper",
                    "trips": chain[:]
                })
            else:
                # Use day duty
                crew_duties.append({
                    "depot": dname,
                    "type": "day",
                    "trips": chain[:]
                })

    num_crews = len(crew_duties)
    objective = num_vehicles + num_crews

    # Log initial solution
    if logger:
        logger.log(objective)

    # Try to improve with a very simple local search: merge duties when possible
    # This is very basic but respects time limit
    improvement = True
    iterations = 0
    max_iter = 50
    while improvement and iterations < max_iter and (time.time() - start_time) < args.time_limit - 2:
        improvement = False
        iterations += 1
        # Try to find two duties from same depot that can be merged into one with two pieces
        for i in range(len(crew_duties)):
            for j in range(i+1, len(crew_duties)):
                if crew_duties[i]['depot'] != crew_duties[j]['depot']:
                    continue
                if crew_duties[i]['type'] != 'tripper' or crew_duties[j]['type'] != 'tripper':
                    continue
                # Check if combining them respects piece limits for 'day' or 'split'
                d1_trips = crew_duties[i]['trips']
                d2_trips = crew_duties[j]['trips']
                if not d1_trips or not d2_trips:
                    continue
                # Assume we put them as two pieces with a break in between
                piece1_end = trips[d1_trips[-1]]['end_time']
                piece2_start = trips[d2_trips[0]]['start_time']
                break_time = piece2_start - piece1_end
                if break_time < 30:  # assume min break
                    continue
                # Check max piece length - simplistic
                p1_dur = trips[d1_trips[-1]]['end_time'] - trips[d1_trips[0]]['start_time']
                p2_dur = trips[d2_trips[-1]]['end_time'] - trips[d2_trips[0]]['start_time']
                if p1_dur > 240 or p2_dur > 240:
                    continue
                # Merge into one day duty
                combined_trips = d1_trips + d2_trips
                new_duty = {
                    "depot": crew_duties[i]['depot'],
                    "type": "day",
                    "trips": combined_trips
                }
                # Replace two duties with one
                new_crews = [new_duty] + [crew_duties[k] for k in range(len(crew_duties)) if k != i and k != j]
                new_obj = num_vehicles + len(new_crews)
                if new_obj < objective:
                    crew_duties = new_crews
                    objective = new_obj
                    improvement = True
                    if logger:
                        logger.log(objective)
                    break
            if improvement:
                break

    # Final solution construction - ensure all trips are covered (they are by construction)
    # But we must have exact cover, our heuristic does
    final_vehicle_schedule = {}
    for dname, chains in vehicle_schedules.items():
        final_vehicle_schedule[dname] = [[int(t) for t in chain] for chain in chains]

    final_crew_schedule = []
    for duty in crew_duties:
        final_crew_schedule.append({
            "depot": duty["depot"],
            "type": duty["type"],
            "trips": [int(t) for t in duty["trips"]]
        })

    # Ensure all trips are covered at least once in crew (our construction does via vehicle)
    covered = set()
    for duty in final_crew_schedule:
        for t in duty['trips']:
            covered.add(t)
    assert len(covered) == num_trips, "Not all trips covered by crew"

    solution = {
        "objective_value": int(objective),
        "vehicle_schedule": final_vehicle_schedule,
        "crew_schedule": final_crew_schedule
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

    print(f"Final objective: {objective}", file=sys.stderr)

if __name__ == "__main__":
    main()
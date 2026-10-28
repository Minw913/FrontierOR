import argparse
import json
import math
from collections import defaultdict


def solve_container_terminal(instance_data):
    # Extract instance data
    S = instance_data["S"]
    n = instance_data["n"]
    m = instance_data["m"]
    p = instance_data["p"]
    sigma_w = instance_data["sigma_w"]
    sigma_l = instance_data["sigma_l"]
    
    seaside_containers = instance_data["seaside_containers"]
    seaside_containers.sort(key=lambda x: x["id"])
    
    landside_containers = instance_data["landside_containers"]
    landside_containers.sort(key=lambda x: x["deadline"])
    
    # Calculate time horizon
    max_deadline = max([c["deadline"] for c in landside_containers]) if landside_containers else 0
    time_horizon = max_deadline + p * 2 + 2 * max(S, sigma_l, sigma_w)
    
    # Preprocessing - find which containers need to be processed
    landside_to_process = list(range(len(landside_containers)))[:]

    # Determine what landside containers must be served
    # All deadlines <= max makespan, plus at least one from next deadline group
    # For this heuristic solution we'll try to process all
    
    # Heuristic approach to schedule cranes
    # We will simulate moving cranes while respecting all constraints
    
    # Initialize crane positions
    cpos_w = {0: sigma_w}
    cpos_l = {0: sigma_l}
    
    # Results to build up
    seaside_events = []
    landside_events = []
    
    # Track crane operations
    crane_w_operations = {}  # tracks ongoing lift/drop operations
    crane_l_operations = {}
    
    # Assign landside containers to be handled first
    available_land_crane_time = 0
    landside_slots = [True] * (S + 2)
    for lc in landside_containers:
        landside_slots[lc["source_slot"]] = False  # occupied
    
    landside_container_handled_times = defaultdict(list)
    
    # Handle landside containers
    for i, lc in enumerate(landside_containers):
        src_slot = lc["source_slot"]
        
        # Find when we can start lifting this landside container
        lift_start_time = max(
            available_land_crane_time,
            lc["earliest_finish_time"] - p,  # start lifting early enough to finish by earliest
        )
        
        # Move landside crane to get to the source slot
        current_pos = cpos_l[max(cpos_l.keys())]
        pos_at_lift = current_pos
        
        # Estimate required movements
        dist = abs(pos_at_lift - src_slot)
        arrival_time = lift_start_time - dist
        
        # Need to ensure this doesn't overlap with any ongoing operations
        actual_lift_time = max(arrival_time + dist, available_land_crane_time)
        
        # Check for crane conflicts if needed (in a simple simulation)
        while actual_lift_time < lift_start_time - dist and cpos_l.get(actual_lift_time - 1) == src_slot:
            actual_lift_time += 1
            
        # Simulate movement to source slot
        t = max(cpos_l.keys())
        for move_time in range(t + 1, actual_lift_time + 1):
            if move_time < actual_lift_time:
                # Moving
                from_pos = cpos_l[move_time - 1]
                to_pos = min(from_pos + 1, src_slot) if from_pos < src_slot else max(from_pos - 1, src_slot)
                cpos_l[move_time] = to_pos
            else:
                cpos_l[move_time] = src_slot
        
        # Perform lift operation lasting 'p' units
        lift_time = actual_lift_time
        for t in range(lift_time, lift_time + p):
            cpos_l[t] = src_slot  # Stay put during lifting
            crane_l_operations[t] = ("lift", lc["id"])
        
        # Now move to landside handover point (slot S+1)
        landside_handover = S + 1
        start_drop_move = lift_time + p
        pos = src_slot
        t = start_drop_move
        while pos != landside_handover:
            pos = pos + 1 if pos < landside_handover else pos - 1
            if t in cpos_l:
                continue
            cpos_l[t] = pos
            t += 1
            
        drop_time = t - 1 + (landside_handover - pos if landside_handover > pos else pos - landside_handover)
        
        # Start drop at landside handover
        drop_start = drop_time
        for dt in range(drop_start, drop_start + p):
            cpos_l[dt] = landside_handover
            crane_l_operations[dt] = ("drop", lc["id"])
            
        # Verify deadlines and update availability
        finish_time = drop_start + p
        drop_time = drop_start  # beginning drop
        if finish_time > lc["deadline"]:
            # This indicates the instance may be infeasible given our approach
            print(f"WARNING: Landside container {lc['id']} finishes at {finish_time} but deadline is {lc['deadline']}")
        
        available_land_crane_time = finish_time
        landside_events.append({
            "container_id": lc["id"],
            "source_slot": src_slot,
            "lift_time": lift_time,
            "drop_time": drop_time
        })
    
    # Now handle seaside containers
    seaside_last_drop_time = 0
    available_sea_crane_time = 0
    
    for sc in seaside_containers:
        id_sc = sc["id"]
        target = sc["target_slot"]
        
        # Seaside crane starts at slot 0 to pick up container
        # Current position is at slot 0 initially for pickup
        current_pos = cpos_w[max(cpos_w.keys())] if len(cpos_w) > 0 else sigma_w
        
        # Since containers arrive in sequence, we can model that we can only
        # lift a container after finishing with previous one
        lift_start_time = max(available_sea_crane_time, seaside_last_drop_time)
        
        # Move to slot 0 if not already there (or after previous operation)
        t = max(cpos_w.keys()) if cpos_w else 0
        pos = current_pos
        
        while pos != 0:
            pos = pos - 1 if pos > 0 else pos  
            if t + 1 > len(cpos_w):
                cpos_w[t + 1] = pos
            else:
                t += 1
                cpos_w[t] = pos
        
        actual_pickup_start = max(t, available_sea_crane_time)
        
        # Lift the container (stay at position 0)
        for lt in range(actual_pickup_start, actual_pickup_start + p):
            cpos_w[lt] = 0  # stays at slot 0 during lift
            if lt not in crane_w_operations:
                crane_w_operations[lt] = ("lift", id_sc)
        
        # Now start moving to target slot (or maybe an intermediate point)
        pickup_end_time = actual_pickup_start + p

        # Move to target slot
        pos_at_pickup = 0
        move_start_time = pickup_end_time
        
        current_time = move_start_time 
        pos = 0
        
        while pos != target:
            direction = 1 if pos < target else -1
            new_pos = pos + direction
            # Check if cranes conflict
            land_pos_t = cpos_l.get(current_time, cpos_l[max(cpos_l.keys())])
            if new_pos >= land_pos_t:  # seaside crane must stay left of landside
                # Wait instead of moving
                cpos_w[current_time] = pos
                current_time += 1
                continue
            pos = new_pos
            cpos_w[current_time] = pos  
            current_time += 1
        
        # At target slot now, perform the drop
        drop_time = current_time
        for dt in range(drop_time, drop_time + p):
            cpos_w[dt] = target
            crane_w_operations[dt] = ("drop", id_sc)
        
        available_sea_crane_time = drop_time + p
        seaside_last_drop_time = drop_time
        
        # Record the event
        event = {
            "container_id": id_sc,
            "target_slot": target,
            "seaside_drop_time": drop_time,
            "seaside_drop_slot": target
            # No cooperation since we're delivering directly
        }
        seaside_events.append(event)
    
    # Extend crane position timeline to the max time observed
    max_time = 0
    if cpos_w:
        max_time = max(max_time, max(cpos_w.keys()))
    if cpos_l:
        max_time = max(max_time, max(cpos_l.keys()))
        
    # Ensure both crane timelines go up to the max observed
    if max_time > 0:
        last_w = cpos_w.get(max_time, cpos_w[max(cpos_w.keys())])
        last_l = cpos_l.get(max_time, cpos_l[max(cpos_l.keys())])
        for t in range(max_time+1):
            if t not in cpos_w:
                cpos_w[t] = last_w
            if t not in cpos_l:
                cpos_l[t] = last_l
                
    # Calculate makespan
    if not seaside_events:
        makespan = 0
    else:
        # Makespan is when ALL seaside containers are placed
        makespan = 0
        for evt in seaside_events:
            # Container is placed fully after its drop op ends
            completion = evt["seaside_drop_time"] + p
            makespan = max(makespan, completion)
    
    # Build solution object
    solution = {
        "objective_value": makespan,
        "crane_w_positions": {str(t): pos for t, pos in cpos_w.items()},
        "crane_l_positions": {str(t): pos for t, pos in cpos_l.items()},
        "seaside_events": seaside_events,
        "landside_events": landside_events
    }
    
    return solution


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    # Import logger if logging path provided
    logger = None
    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize")

    # Load instance data
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    # Solve
    solution = solve_container_terminal(instance)

    # Log the solution if requested
    if logger:
        logger.log(solution["objective_value"])

    # Write solution to file
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()
import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from collections import defaultdict
import time
from solution_logger import SolutionLogger

def solve_transit_scheduling(instance_path, solution_path, time_limit, log_path):
    # Read instance data
    with open(instance_path, 'r') as f:
        data = json.load(f)
    
    # Initialize logger
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    # Extract data
    terminals = data['topology']['terminals']
    depot = data['topology']['depot']
    directions = {d['pattern_id']: d for d in data['directions']}
    time_windows = data['time_windows']
    potential_trips = {t['id']: t for t in data['potential_trips']}
    
    # Build mapping from pattern_id to trips
    pattern_to_trips = defaultdict(list)
    for trip_id, trip in potential_trips.items():
        pattern_to_trips[trip['pattern_id']].append(trip_id)
    
    # Extract stopping times and pull in/out times
    stopping_times = {}
    pull_in_out_times = {}
    for term in terminals:
        stopping_times[term] = {entry['time_window_index']: entry for entry in data['stopping_times'][term]}
        pull_in_out_times[term] = {entry['time_window_index']: entry for entry in data['pull_in_out_times'][term]}
    
    # Parameters
    alpha = data['objective_function']['alpha']
    penalty_func_type = data['penalty_function']['type']
    
    # Create model
    model = gp.Model("transit_scheduling")
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', time_limit)
    
    # Decision variables
    # Timetable selection variables
    x = {}  # x[trip_id] = 1 if trip is selected
    for trip_id in potential_trips:
        x[trip_id] = model.addVar(vtype=GRB.BINARY, name=f"x_{trip_id}")
    
    # Timetable arc variables (for consecutive trips in same direction)
    y_tt = {}  # y_tt[(i,j)] = 1 if trip i is followed by trip j in timetable
    for pattern_id in pattern_to_trips:
        trips_in_pattern = pattern_to_trips[pattern_id]
        for i in trips_in_pattern:
            for j in trips_in_pattern:
                trip_i = potential_trips[i]
                trip_j = potential_trips[j]
                
                # Check if trip j can follow trip i in the same direction
                # Trip j must start after trip i finishes + minimum headway
                min_headway = data['time_windows'][trip_i['time_window_index']]['min_headway_seconds'] / 60.0  # convert to minutes
                if trip_j['main_stop_arrival_time_minutes'] >= trip_i['main_stop_arrival_time_minutes'] + min_headway:
                    y_tt[(i, j)] = model.addVar(vtype=GRB.BINARY, name=f"y_tt_{i}_{j}")
    
    # Vehicle scheduling flow variables
    y_vs = {}  # y_vs[(i,j)] = flow from node i to node j in vehicle scheduling network
    
    # Define special nodes for vehicle scheduling
    depot_out = "VS_O_minus"
    depot_in = "VS_O_plus"
    
    # Add arcs from depot to start of trips
    for trip_id in potential_trips:
        trip = potential_trips[trip_id]
        pull_out_time = pull_in_out_times[trip['start_terminal']][trip['time_window_index']]['pull_out_time_minutes']
        y_vs[(depot_out, trip_id)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"y_vs_depot_to_{trip_id}")
    
    # Add arcs from end of trips to depot
    for trip_id in potential_trips:
        trip = potential_trips[trip_id]
        pull_in_time = pull_in_out_times[trip['end_terminal']][trip['time_window_index']]['pull_in_time_minutes']
        y_vs[(trip_id, depot_in)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"y_vs_{trip_id}_to_depot")
    
    # Add in-line compatibility arcs (same terminal)
    for trip_i_id in potential_trips:
        trip_i = potential_trips[trip_i_id]
        for trip_j_id in potential_trips:
            trip_j = potential_trips[trip_j_id]
            
            # Check if trips end and start at same terminal
            if trip_i['end_terminal'] == trip_j['start_terminal']:
                # Check if timing allows
                min_stopping_time = stopping_times[trip_i['end_terminal']][trip_i['time_window_index']]['min_stopping_time_minutes']
                max_stopping_time = stopping_times[trip_i['end_terminal']][trip_i['time_window_index']]['max_stopping_time_minutes']
                
                gap = trip_j['departure_time_minutes'] - trip_i['arrival_time_minutes']
                if min_stopping_time <= gap <= max_stopping_time:
                    y_vs[(trip_i_id, trip_j_id)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"y_vs_{trip_i_id}_{trip_j_id}")
    
    # Add out-line compatibility arcs (different terminals)
    for trip_i_id in potential_trips:
        trip_i = potential_trips[trip_i_id]
        for trip_j_id in potential_trips:
            trip_j = potential_trips[trip_j_id]
            
            # Check if trips end and start at different terminals
            if trip_i['end_terminal'] != trip_j['start_terminal']:
                # Check if timing allows
                pull_in_time = pull_in_out_times[trip_i['end_terminal']][trip_i['time_window_index']]['pull_in_time_minutes']
                min_depot_time = 0  # Assuming minimum depot stopping time is 0 as per description
                pull_out_time = pull_in_out_times[trip_j['start_terminal']][trip_j['time_window_index']]['pull_out_time_minutes']
                
                min_total_time = pull_in_time + min_depot_time + pull_out_time
                gap = trip_j['departure_time_minutes'] - trip_i['arrival_time_minutes']
                
                if gap >= min_total_time:
                    y_vs[(trip_i_id, trip_j_id)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"y_vs_{trip_i_id}_{trip_j_id}")
    
    # Flow conservation constraints for vehicle scheduling
    for trip_id in potential_trips:
        # Flow into trip end = flow out of trip start = selected trips
        model.addConstr(
            gp.quicksum(y_vs[(prev_id, trip_id)] for prev_id in potential_trips if (prev_id, trip_id) in y_vs) +
            y_vs[(depot_out, trip_id)] == x[trip_id],
            name=f"flow_in_{trip_id}"
        )
        
        model.addConstr(
            gp.quicksum(y_vs[(trip_id, next_id)] for next_id in potential_trips if (trip_id, next_id) in y_vs) +
            y_vs[(trip_id, depot_in)] == x[trip_id],
            name=f"flow_out_{trip_id}"
        )
    
    # Timetable constraints
    for pattern_id in pattern_to_trips:
        trips_in_pattern = pattern_to_trips[pattern_id]
        
        # Each selected trip in pattern has at most one predecessor and successor
        for trip_id in trips_in_pattern:
            # At most one predecessor
            model.addConstr(
                gp.quicksum(y_tt[(prev_id, trip_id)] for prev_id in trips_in_pattern 
                           if (prev_id, trip_id) in y_tt) <= x[trip_id],
                name=f"pred_{trip_id}"
            )
            
            # At most one successor
            model.addConstr(
                gp.quicksum(y_tt[(trip_id, next_id)] for next_id in trips_in_pattern 
                           if (trip_id, next_id) in y_tt) <= x[trip_id],
                name=f"succ_{trip_id}"
            )
        
        # Chain formation constraints (flow conservation for timetable)
        for trip_id in trips_in_pattern:
            model.addConstr(
                gp.quicksum(y_tt[(prev_id, trip_id)] for prev_id in trips_in_pattern 
                           if (prev_id, trip_id) in y_tt) ==
                gp.quicksum(y_tt[(trip_id, next_id)] for next_id in trips_in_pattern 
                           if (trip_id, next_id) in y_tt),
                name=f"chain_flow_{trip_id}"
            )
    
    # Headway constraints
    for (i, j) in y_tt:
        trip_i = potential_trips[i]
        trip_j = potential_trips[j]
        
        # Ensure headway constraints are satisfied
        min_headway = data['time_windows'][trip_i['time_window_index']]['min_headway_seconds'] / 60.0  # convert to minutes
        max_headway = data['time_windows'][trip_i['time_window_index']]['max_headway_seconds'] / 60.0  # convert to minutes
        
        # Only if y_tt[i,j] = 1, enforce headway constraint
        model.addGenConstrIndicator(y_tt[(i, j)], 1,
                                   trip_j['main_stop_arrival_time_minutes'] - trip_i['main_stop_arrival_time_minutes'] >= min_headway,
                                   name=f"headway_min_{i}_{j}")
        model.addGenConstrIndicator(y_tt[(i, j)], 1,
                                   trip_j['main_stop_arrival_time_minutes'] - trip_i['main_stop_arrival_time_minutes'] <= max_headway,
                                   name=f"headway_max_{i}_{j}")
    
    # First and last trip constraints
    initial_trip_windows = data.get('initial_trip_time_windows', {})
    final_trip_windows = data.get('final_trip_time_windows', {})
    
    for pattern_id, window_idx in initial_trip_windows.items():
        trips_in_pattern_and_window = [
            trip_id for trip_id in pattern_to_trips[pattern_id] 
            if potential_trips[trip_id]['time_window_index'] == window_idx
        ]
        
        # Exactly one first trip from allowed candidates
        model.addConstr(
            gp.quicksum(x[trip_id] - gp.quicksum(y_tt[(prev_id, trip_id)] for prev_id in pattern_to_trips[pattern_id] 
                                               if (prev_id, trip_id) in y_tt) 
                       for trip_id in trips_in_pattern_and_window) == 1,
            name=f"first_trip_{pattern_id}"
        )
    
    for pattern_id, window_idx in final_trip_windows.items():
        trips_in_pattern_and_window = [
            trip_id for trip_id in pattern_to_trips[pattern_id] 
            if potential_trips[trip_id]['time_window_index'] == window_idx
        ]
        
        # Exactly one last trip from allowed candidates
        model.addConstr(
            gp.quicksum(x[trip_id] - gp.quicksum(y_tt[(trip_id, next_id)] for next_id in pattern_to_trips[pattern_id] 
                                               if (trip_id, next_id) in y_tt) 
                       for trip_id in trips_in_pattern_and_window) == 1,
            name=f"last_trip_{pattern_id}"
        )
    
    # Objective function
    # Timetable cost: sum of quadratic headway deviation penalties
    tt_cost_expr = gp.QuadExpr()
    for (i, j) in y_tt:
        trip_i = potential_trips[i]
        trip_j = potential_trips[j]
        window_idx = trip_i['time_window_index']
        ideal_headway = data['time_windows'][window_idx]['ideal_headway_seconds'] / 60.0  # convert to minutes
        actual_headway = trip_j['main_stop_arrival_time_minutes'] - trip_i['main_stop_arrival_time_minutes']
        deviation = actual_headway - ideal_headway
        
        # Quadratic penalty: (deviation)^2
        tt_cost_expr += y_tt[(i, j)] * (deviation * deviation)
    
    # Vehicle scheduling cost
    vs_cost_expr = gp.LinExpr()
    
    # Component 1: Per-vehicle deployment cost
    # This is equivalent to the flow on the depot circulation arc (VS_O_plus --> VS_O_minus)
    # We'll add this variable and link it to the flow
    num_vehicles_var = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="num_vehicles")
    model.addConstr(num_vehicles_var == y_vs[(depot_in, depot_out)] if (depot_in, depot_out) in y_vs else 0)
    
    # For now, we'll just count the number of vehicles needed based on flow
    # Deploy cost is implicitly handled by the flow conservation
    deploy_cost_coeff = 1000000  # Large per-vehicle deployment cost
    vs_cost_expr += deploy_cost_coeff * num_vehicles_var
    
    # Component 2: Extra waiting time beyond minimum stopping time
    for trip_i_id in potential_trips:
        trip_i = potential_trips[trip_i_id]
        for trip_j_id in potential_trips:
            trip_j = potential_trips[trip_j_id]
            
            if trip_i['end_terminal'] == trip_j['start_terminal']:
                # In-line compatibility
                min_stopping_time = stopping_times[trip_i['end_terminal']][trip_i['time_window_index']]['min_stopping_time_minutes']
                gap = trip_j['departure_time_minutes'] - trip_i['arrival_time_minutes']
                extra_time = gap - min_stopping_time
                
                if (trip_i_id, trip_j_id) in y_vs and extra_time > 0:
                    vs_cost_expr += y_vs[(trip_i_id, trip_j_id)] * extra_time
    
    # Component 3: Pull-in and pull-out travel times
    # Add pull-out for first trips
    for trip_id in potential_trips:
        trip = potential_trips[trip_id]
        pull_out_time = pull_in_out_times[trip['start_terminal']][trip['time_window_index']]['pull_out_time_minutes']
        vs_cost_expr += y_vs[(depot_out, trip_id)] * pull_out_time
    
    # Add pull-in for last trips
    for trip_id in potential_trips:
        trip = potential_trips[trip_id]
        pull_in_time = pull_in_out_times[trip['end_terminal']][trip['time_window_index']]['pull_in_time_minutes']
        vs_cost_expr += y_vs[(trip_id, depot_in)] * pull_in_time
    
    # Add pull-in + pull-out for out-line connections
    for trip_i_id in potential_trips:
        trip_i = potential_trips[trip_i_id]
        for trip_j_id in potential_trips:
            trip_j = potential_trips[trip_j_id]
            
            if trip_i['end_terminal'] != trip_j['start_terminal']:
                pull_in_time = pull_in_out_times[trip_i['end_terminal']][trip_i['time_window_index']]['pull_in_time_minutes']
                pull_out_time = pull_in_out_times[trip_j['start_terminal']][trip_j['time_window_index']]['pull_out_time_minutes']
                
                if (trip_i_id, trip_j_id) in y_vs:
                    vs_cost_expr += y_vs[(trip_i_id, trip_j_id)] * (pull_in_time + pull_out_time)
    
    # Total cost
    total_cost = alpha * vs_cost_expr + tt_cost_expr
    model.setObjective(total_cost, GRB.MINIMIZE)
    
    # Solve the model
    start_time = time.time()
    model.optimize()
    end_time = time.time()
    
    # Process results
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        # Extract selected trips
        selected_trips = []
        for trip_id in x:
            if x[trip_id].x > 0.9:
                selected_trips.append(trip_id)
        
        # Extract timetable arcs for each pattern
        tt_arcs_used = {}
        for pattern_id in pattern_to_trips:
            arcs = []
            for (i, j) in y_tt:
                if potential_trips[i]['pattern_id'] == pattern_id and y_tt[(i, j)].x > 0.9:
                    arcs.append([str(i), str(j)])
            
            # Add source and sink nodes to complete the chain
            # Find first and last trips in the pattern
            trips_in_pattern = pattern_to_trips[pattern_id]
            first_trips = []
            last_trips = []
            
            for trip_id in trips_in_pattern:
                if x[trip_id].x > 0.9:
                    has_pred = any(y_tt[(prev_id, trip_id)].x > 0.9 for prev_id in trips_in_pattern if (prev_id, trip_id) in y_tt)
                    has_succ = any(y_tt[(trip_id, next_id)].x > 0.9 for next_id in trips_in_pattern if (trip_id, next_id) in y_tt)
                    
                    if not has_pred:
                        first_trips.append(trip_id)
                    if not has_succ:
                        last_trips.append(trip_id)
            
            # Add source to first trip and last trip to sink
            for first_trip in first_trips:
                arcs.insert(0, [f"S_{pattern_id}", str(first_trip)])
            for last_trip in last_trips:
                arcs.append([str(last_trip), f"T_{pattern_id}"])
                
            tt_arcs_used[pattern_id] = arcs
        
        # Extract vehicle scheduling flows
        vs_flows = {}
        for (i, j) in y_vs:
            if y_vs[(i, j)].x > 0.001:  # Only include non-zero flows
                vs_flows[f"{i}-->{j}"] = y_vs[(i, j)].x
        
        # Add circulation arc for number of vehicles
        vs_flows[f"{depot_in}-->{depot_out}"] = num_vehicles_var.x if 'num_vehicles_var' in locals() else 0
        
        # Calculate number of vehicles (flow on circulation arc)
        num_vehicles = int(round(y_vs[(depot_in, depot_out)].x)) if (depot_in, depot_out) in y_vs else 0
        
        # Prepare solution dict
        solution_dict = {
            "objective_value": model.objVal,
            "selected_trips": sorted(selected_trips),
            "num_vehicles": num_vehicles,
            "tt_arcs_used": tt_arcs_used,
            "vs_flows": vs_flows
        }
        
        # Log solution if logger is available
        if logger:
            logger.log_solution(model.objVal, solution_dict)
        
        # Write solution to file
        with open(solution_path, 'w') as f:
            json.dump(solution_dict, f)
        
        print(f"Solution written to {solution_path}")
        print(f"Objective value: {model.objVal}")
        print(f"Number of selected trips: {len(selected_trips)}")
        print(f"Number of vehicles: {num_vehicles}")
    else:
        print(f"Optimization ended with status: {model.status}")
        print("No feasible solution found.")

def main():
    parser = argparse.ArgumentParser(description='Solve transit scheduling problem')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the instance JSON file')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to save the solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds')
    parser.add_argument('--log_path', type=str, default=None, help='Path to log intermediate solutions')
    
    args = parser.parse_args()
    
    solve_transit_scheduling(args.instance_path, args.solution_path, args.time_limit, args.log_path)

if __name__ == "__main__":
    main()
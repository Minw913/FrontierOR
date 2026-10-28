import gurobipy as gp
from gurobipy import GRB
import json

def load_instance(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)

def solve(instance):
    model = gp.Model("Integrated_Timetabling_Vehicle_Scheduling")
    
    # Extract topology
    terminals = instance['topology']['terminals']
    depot = instance['topology']['depot']
    
    # Extract time windows
    time_windows = instance['time_windows']
    
    # Extract patterns
    patterns = instance['patterns']
    pattern_names = list(patterns.keys())
    
    # Extract trips
    trips = instance['trips']
    trip_ids = list(trips.keys())
    
    # Convert times to seconds for headway constraints
    for trip_id, trip in trips.items():
        trip['departure_time_seconds'] = trip['departure_time_minutes'] * 60
        trip['arrival_time_seconds'] = trip['arrival_time_minutes'] * 60
        trip['main_stop_arrival_time_seconds'] = trip['main_stop_arrival_time_minutes'] * 60
    
    # Create binary variables for trip selection
    x = model.addVars(trip_ids, vtype=GRB.BINARY, name="x")
    
    # Build TT networks for each pattern
    y = {}
    TT_arcs = {p: [] for p in pattern_names}
    TT_arc_costs = {}
    
    for p in pattern_names:
        pattern_trips = [trip_id for trip_id in trip_ids if trips[trip_id]['pattern'] == p]
        source = f"source_{p}"
        sink = f"sink_{p}"
        
        # Source to initial trips
        initial_tw = patterns[p]['initial_trip_time_windows']
        for trip_id in pattern_trips:
            if trips[trip_id]['time_window_index'] in initial_tw:
                arc = (source, trip_id)
                TT_arcs[p].append(arc)
                y[arc] = model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f"y_{p}_{source}_{trip_id}")
        
        # Final trips to sink
        final_tw = patterns[p]['final_trip_time_windows']
        for trip_id in pattern_trips:
            if trips[trip_id]['time_window_index'] in final_tw:
                arc = (trip_id, sink)
                TT_arcs[p].append(arc)
                y[arc] = model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f"y_{p}_{trip_id}_{sink}")
        
        # Between trips
        pattern_trips_sorted = sorted(pattern_trips, key=lambda tid: trips[tid]['main_stop_arrival_time_seconds'])
        for i in range(len(pattern_trips_sorted)):
            trip_i = pattern_trips_sorted[i]
            t_i = trips[trip_i]
            for j in range(i+1, len(pattern_trips_sorted)):
                trip_j = pattern_trips_sorted[j]
                t_j = trips[trip_j]
                headway = t_j['main_stop_arrival_time_seconds'] - t_i['main_stop_arrival_time_seconds']
                tw_i = t_i['time_window_index']
                min_headway = time_windows[tw_i]['min_headway_seconds']
                max_headway = time_windows[tw_i]['max_headway_seconds']
                ideal_headway = time_windows[tw_i]['ideal_headway_seconds']
                
                if min_headway <= headway <= max_headway:
                    arc = (trip_i, trip_j)
                    TT_arcs[p].append(arc)
                    y[arc] = model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f"y_{p}_{trip_i}_{trip_j}")
                    TT_arc_costs[arc] = (headway - ideal_headway) ** 2
    
    # Add TT constraints
    for p in pattern_names:
        source = f"source_{p}"
        sink = f"sink_{p}"
        pattern_trips = [trip_id for trip_id in trip_ids if trips[trip_id]['pattern'] == p]
        
        # Source constraint
        model.addConstr(gp.quicksum(y[arc] for arc in TT_arcs[p] if arc[0] == source) == 1, f"TT_source_{p}")
        
        # Sink constraint
        model.addConstr(gp.quicksum(y[arc] for arc in TT_arcs[p] if arc[1] == sink) == 1, f"TT_sink_{p}")
        
        # Flow conservation for trips
        for trip_id in pattern_trips:
            in_arcs = [arc for arc in TT_arcs[p] if arc[1] == trip_id]
            out_arcs = [arc for arc in TT_arcs[p] if arc[0] == trip_id]
            model.addConstr(gp.quicksum(y[arc] for arc in in_arcs) == x[trip_id], f"TT_in_{trip_id}")
            model.addConstr(gp.quicksum(y[arc] for arc in out_arcs) == x[trip_id], f"TT_out_{trip_id}")
    
    # Precompute VS compatible arcs
    in_line_arcs = []
    out_line_arcs = []
    stopping_times = instance['stopping_times']
    pull_in_out_times = instance['pull_in_out_times']
    
    for i in trip_ids:
        t_i = trips[i]
        end_term_i = t_i['end_terminal']
        arr_i = t_i['arrival_time_seconds']
        tw_i = t_i['time_window_index']
        
        for j in trip_ids:
            if i == j:
                continue
            t_j = trips[j]
            start_term_j = t_j['start_terminal']
            dep_j = t_j['departure_time_seconds']
            tw_j = t_j['time_window_index']
            gap = dep_j - arr_i
            
            if gap < 0:
                continue
                
            if end_term_i == start_term_j:
                # In-line connection
                stop_info = next(st for st in stopping_times[end_term_i] if st['time_window_index'] == tw_i)
                min_stop = stop_info['min_stopping_time_minutes'] * 60
                max_stop = stop_info['max_stopping_time_minutes'] * 60
                if min_stop <= gap <= max_stop:
                    in_line_arcs.append((i, j))
            else:
                # Out-line connection
                pull_in_i = next(pt for pt in pull_in_out_times[end_term_i] if pt['time_window_index'] == tw_i)['pull_in_time_minutes'] * 60
                pull_out_j = next(pt for pt in pull_in_out_times[start_term_j] if pt['time_window_index'] == tw_j)['pull_out_time_minutes'] * 60
                min_depot_stop = 0
                if depot in stopping_times:
                    depot_stop = next((st for st in stopping_times[depot] if st['time_window_index'] == tw_i), None)
                    if depot_stop:
                        min_depot_stop = depot_stop['min_stopping_time_minutes'] * 60
                if gap >= pull_in_i + min_depot_stop + pull_out_j:
                    out_line_arcs.append((i, j))
    
    # Build VS network
    f = {}
    vs_arcs = []
    
    # Pull-out arcs
    for i in trip_ids:
        arc = ('VS_O_minus', (i, 'start'))
        vs_arcs.append(arc)
        f[arc] = model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"f_pull_out_{i}")
    
    # Trip operation arcs
    for i in trip_ids:
        arc = ((i, 'start'), (i, 'end'))
        vs_arcs.append(arc)
        f[arc] = model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"f_trip_{i}")
    
    # In-line connection arcs
    for (i, j) in in_line_arcs:
        arc = ((i, 'end'), (j, 'start'))
        vs_arcs.append(arc)
        f[arc] = model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"f_in_line_{i}_{j}")
    
    # Out-line connection arcs
    for (i, j) in out_line_arcs:
        arc = ((i, 'end'), (j, 'start'))
        vs_arcs.append(arc)
        f[arc] = model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"f_out_line_{i}_{j}")
    
    # Pull-in arcs
    for i in trip_ids:
        arc = ((i, 'end'), 'VS_O_plus')
        vs_arcs.append(arc)
        f[arc] = model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"f_pull_in_{i}")
    
    # Circulation arc
    circ_arc = ('VS_O_plus', 'VS_O_minus')
    vs_arcs.append(circ_arc)
    f[circ_arc] = model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="f_circ")
    
    # VS flow conservation
    # Depot nodes
    model.addConstr(
        gp.quicksum(f[arc] for arc in vs_arcs if arc[0] == 'VS_O_minus') == f[circ_arc],
        "VS_O_minus_conservation"
    )
    model.addConstr(
        gp.quicksum(f[arc] for arc in vs_arcs if arc[1] == 'VS_O_plus') == f[circ_arc],
        "VS_O_plus_conservation"
    )
    
    # Trip start nodes
    for i in trip_ids:
        node = (i, 'start')
        inflow = gp.quicksum(
            f[arc] for arc in vs_arcs if arc[1] == node
        )
        outflow = gp.quicksum(
            f[arc] for arc in vs_arcs if arc[0] == node
        )
        model.addConstr(inflow == outflow, f"VS_start_{i}_conservation")
    
    # Trip end nodes
    for i in trip_ids:
        node = (i, 'end')
        inflow = gp.quicksum(
            f[arc] for arc in vs_arcs if arc[1] == node
        )
        outflow = gp.quicksum(
            f[arc] for arc in vs_arcs if arc[0] == node
        )
        model.addConstr(inflow == outflow, f"VS_end_{i}_conservation")
    
    # Trip operation constraint
    for i in trip_ids:
        arc = ((i, 'start'), (i, 'end'))
        model.addConstr(f[arc] == x[i], f"VS_trip_operation_{i}")
    
    # Set objective
    alpha = instance['alpha']
    per_vehicle_cost = instance['per_vehicle_deployment_cost']
    
    # Timetable cost
    timetable_cost = gp.quicksum(
        TT_arc_costs[arc] * y[arc] 
        for p in pattern_names 
        for arc in TT_arcs[p] 
        if arc in TT_arc_costs
    )
    
    # Vehicle scheduling cost
    vs_cost_components = []
    
    # Circulation cost
    vs_cost_components.append(per_vehicle_cost * f[circ_arc])
    
    # Pull-out costs
    for i in trip_ids:
        arc = ('VS_O_minus', (i, 'start'))
        pull_out_time = next(
            pt for pt in pull_in_out_times[trips[i]['start_terminal']] 
            if pt['time_window_index'] == trips[i]['time_window_index']
        )['pull_out_time_minutes']
        vs_cost_components.append(pull_out_time * f[arc])
    
    # Pull-in costs
    for i in trip_ids:
        arc = ((i, 'end'), 'VS_O_plus')
        pull_in_time = next(
            pt for pt in pull_in_out_times[trips[i]['end_terminal']] 
            if pt['time_window_index'] == trips[i]['time_window_index']
        )['pull_in_time_minutes']
        vs_cost_components.append(pull_in_time * f[arc])
    
    # In-line connection costs (extra waiting time in minutes)
    for (i, j) in in_line_arcs:
        arc = ((i, 'end'), (j, 'start'))
        t_i = trips[i]
        t_j = trips[j]
        gap_seconds = t_j['departure_time_seconds'] - t_i['arrival_time_seconds']
        min_stop_seconds = next(
            st for st in stopping_times[t_i['end_terminal']] 
            if st['time_window_index'] == t_i['time_window_index']
        )['min_stopping_time_minutes'] * 60
        extra_waiting_minutes = (gap_seconds - min_stop_seconds) / 60.0
        vs_cost_components.append(extra_waiting_minutes * f[arc])
    
    # Out-line connection costs
    for (i, j) in out_line_arcs:
        arc = ((i, 'end'), (j, 'start'))
        t_i = trips[i]
        t_j = trips[j]
        pull_in_time_i = next(
            pt for pt in pull_in_out_times[t_i['end_terminal']] 
            if pt['time_window_index'] == t_i['time_window_index']
        )['pull_in_time_minutes']
        pull_out_time_j = next(
            pt for pt in pull_in_out_times[t_j['start_terminal']] 
            if pt['time_window_index'] == t_j['time_window_index']
        )['pull_out_time_minutes']
        vs_cost_components.append((pull_in_time_i + pull_out_time_j) * f[arc])
    
    vs_cost = gp.quicksum(vs_cost_components)
    total_cost = alpha * vs_cost + timetable_cost
    model.setObjective(total_cost, GRB.MINIMIZE)
    
    # Solve
    model.optimize()
    
    # Extract solution
    solution = {
        "selected_trips": [trip_id for trip_id in trip_ids if x[trip_id].X > 0.5],
        "timetable_chains": {},
        "vehicle_flows": {}
    }
    
    # Timetable chains
    for p in pattern_names:
        chain = []
        source = f"source_{p}"
        sink = f"sink_{p}"
        current = source
        while current != sink:
            for arc in TT_arcs[p]:
                if arc[0] == current and y[arc].X > 0.5:
                    if arc[1] != sink:
                        chain.append(arc[1])
                    current = arc[1]
                    break
        solution["timetable_chains"][p] = chain
    
    # Vehicle flows
    solution["vehicle_flows"] = {
        "num_vehicles": f[circ_arc].X,
        "pull_out": {i: f[('VS_O_minus', (i, 'start'))].X for i in trip_ids},
        "trip_operation": {i: f[((i, 'start'), (i, 'end'))].X for i in trip_ids},
        "in_line": {(i,j): f[((i, 'end'), (j, 'start'))].X for (i,j) in in_line_arcs},
        "out_line": {(i,j): f[((i, 'end'), (j, 'start'))].X for (i,j) in out_line_arcs},
        "pull_in": {i: f[((i, 'end'), 'VS_O_plus')].X for i in trip_ids}
    }
    
    return solution

if __name__ == "__main__":
    import sys
    instance_file = sys.argv[1]
    instance = load_instance(instance_file)
    solution = solve(instance)
    with open("solution.json", "w") as f:
        json.dump(solution, f, indent=2)
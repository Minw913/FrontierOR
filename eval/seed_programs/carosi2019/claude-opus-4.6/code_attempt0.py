import argparse
import json
import time
import sys
from collections import defaultdict

def solve(instance_path, solution_path, time_limit, log_path):
    from solution_logger import SolutionLogger
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    start_time = time.time()
    
    with open(instance_path, 'r') as f:
        data = json.load(f)
    
    # Parse instance data
    terminals = data['topology']['terminals']
    depot = data['topology']['depot']
    directions_info = data['directions']
    time_windows = data['time_windows']
    stopping_times = data['stopping_times']
    pull_in_out_times = data['pull_in_out_times']
    initial_tw = data['initial_trip_time_windows']
    final_tw = data['final_trip_time_windows']
    alpha = data['objective_function']['alpha']
    potential_trips = data['potential_trips']
    
    # Per-vehicle cost
    vs_cost_components = data['objective_function']['vs_cost_components']
    # Find deployment cost - it's mentioned as "large per-vehicle deployment cost"
    # Look for it in the data
    deployment_cost = data['objective_function'].get('deployment_cost', 0)
    if deployment_cost == 0:
        # Try to find it elsewhere
        for comp in vs_cost_components:
            pass
        # Check if there's a specific field
        if 'deployment_cost' in data:
            deployment_cost = data['deployment_cost']
        elif 'vehicle_cost' in data:
            deployment_cost = data['vehicle_cost']
        else:
            # Search in objective_function for any numeric field
            for k, v in data['objective_function'].items():
                if isinstance(v, (int, float)) and k not in ['alpha']:
                    if k == 'deployment_cost' or k == 'vehicle_cost' or 'cost' in k.lower():
                        deployment_cost = v
            if deployment_cost == 0:
                deployment_cost = data['objective_function'].get('deployment_cost', 
                                  data['objective_function'].get('vehicle_deployment_cost',
                                  data.get('deployment_cost', 1000)))
    
    # Stopping times by terminal and time window
    stop_times = {}
    for term in terminals:
        key = term
        if key not in stopping_times:
            # Try other key formats
            for k in stopping_times:
                if k == term or k.lower() == term.lower():
                    key = k
                    break
        if key in stopping_times:
            for entry in stopping_times[key]:
                tw_idx = entry['time_window_index']
                stop_times[(term, tw_idx)] = {
                    'min': entry['min_stopping_time_minutes'],
                    'max': entry['max_stopping_time_minutes']
                }
    
    # Depot stopping time
    depot_min_stop = data.get('depot_min_stopping_time_minutes', 0)
    if 'depot_stopping_time' in data:
        depot_min_stop = data['depot_stopping_time'].get('min_stopping_time_minutes', 0)
    # Check stopping_times for depot
    if depot in stopping_times:
        for entry in stopping_times[depot]:
            tw_idx = entry['time_window_index']
            stop_times[(depot, tw_idx)] = {
                'min': entry['min_stopping_time_minutes'],
                'max': entry.get('max_stopping_time_minutes', 99999)
            }
    
    # Pull in/out times by terminal and time window
    pio_times = {}
    for term in terminals:
        key = term
        if key not in pull_in_out_times:
            for k in pull_in_out_times:
                if k == term or k.lower() == term.lower():
                    key = k
                    break
        if key in pull_in_out_times:
            for entry in pull_in_out_times[key]:
                tw_idx = entry['time_window_index']
                pio_times[(term, tw_idx)] = {
                    'pull_out': entry['pull_out_time_minutes'],
                    'pull_in': entry['pull_in_time_minutes']
                }
    
    # Group trips by pattern
    pattern_ids = list(set(t['pattern_id'] for t in potential_trips))
    trips_by_pattern = defaultdict(list)
    trip_by_id = {}
    for t in potential_trips:
        trips_by_pattern[t['pattern_id']].append(t)
        trip_by_id[t['id']] = t
    
    # Sort trips by main_stop_arrival_time within each pattern
    for pid in trips_by_pattern:
        trips_by_pattern[pid].sort(key=lambda t: t['main_stop_arrival_time_minutes'])
    
    # Time window info
    tw_info = {}
    for tw in time_windows:
        tw_info[tw['index']] = tw
    
    # Build timetable arcs for each pattern
    # For each pattern, we need a chain from source to sink
    # source -> first admissible trips, last admissible trips -> sink
    # consecutive trips must have feasible headways
    
    import gurobipy as gp
    from gurobipy import GRB
    
    model = gp.Model("transit_opt")
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    remaining_time = max(10, time_limit - (time.time() - start_time))
    model.setParam('TimeLimit', remaining_time)
    
    # Decision variables
    # x[trip_id] = 1 if trip is selected
    x = {}
    for t in potential_trips:
        x[t['id']] = model.addVar(vtype=GRB.BINARY, name=f"x_{t['id']}")
    
    # Timetable arc variables for each pattern
    # tt_arc[(pid, i, j)] = 1 if trip j immediately follows trip i in pattern pid's chain
    # Also source/sink arcs
    tt_arc = {}
    tt_cost = {}  # quadratic headway deviation cost for each arc
    
    # For each pattern, build feasible consecutive trip pairs
    tt_arcs_by_pattern = defaultdict(list)
    tt_arcs_from = defaultdict(list)  # arcs from a trip in a pattern
    tt_arcs_to = defaultdict(list)    # arcs to a trip in a pattern
    
    for pid in pattern_ids:
        trips = trips_by_pattern[pid]
        init_tw_idx = initial_tw[pid]
        final_tw_idx = final_tw[pid]
        
        # Source arcs: source -> trip (for initial trips)
        for t in trips:
            if t['time_window_index'] == init_tw_idx:
                arc_key = (pid, 'source', t['id'])
                tt_arc[arc_key] = model.addVar(vtype=GRB.BINARY, name=f"tt_{pid}_s_{t['id']}")
                tt_cost[arc_key] = 0.0
                tt_arcs_by_pattern[pid].append(arc_key)
                tt_arcs_to[(pid, t['id'])].append(arc_key)
        
        # Sink arcs: trip -> sink (for final trips)
        for t in trips:
            if t['time_window_index'] == final_tw_idx:
                arc_key = (pid, t['id'], 'sink')
                tt_arc[arc_key] = model.addVar(vtype=GRB.BINARY, name=f"tt_{pid}_{t['id']}_k")
                tt_cost[arc_key] = 0.0
                tt_arcs_by_pattern[pid].append(arc_key)
                tt_arcs_from[(pid, t['id'])].append(arc_key)
        
        # Inter-trip arcs
        for i_idx, ti in enumerate(trips):
            tw_i = tw_info[ti['time_window_index']]
            min_hw = tw_i['min_headway_seconds']
            max_hw = tw_i['max_headway_seconds']
            ideal_hw = tw_i['ideal_headway_seconds']
            
            for j_idx in range(i_idx + 1, len(trips)):
                tj = trips[j_idx]
                # Headway = difference in main_stop_arrival_time (in seconds)
                headway_minutes = tj['main_stop_arrival_time_minutes'] - ti['main_stop_arrival_time_minutes']
                headway_seconds = headway_minutes * 60
                
                if headway_seconds < min_hw:
                    continue
                if headway_seconds > max_hw:
                    break  # trips are sorted, no need to check further
                
                # Feasible pair
                deviation = headway_seconds - ideal_hw
                penalty = deviation * deviation
                
                arc_key = (pid, ti['id'], tj['id'])
                tt_arc[arc_key] = model.addVar(vtype=GRB.BINARY, name=f"tt_{pid}_{ti['id']}_{tj['id']}")
                tt_cost[arc_key] = penalty
                tt_arcs_by_pattern[pid].append(arc_key)
                tt_arcs_from[(pid, ti['id'])].append(arc_key)
                tt_arcs_to[(pid, tj['id'])].append(arc_key)
    
    # Flow conservation for timetable chains
    for pid in pattern_ids:
        trips = trips_by_pattern[pid]
        
        # Source must send exactly 1 unit of flow
        source_arcs = [tt_arc[ak] for ak in tt_arcs_by_pattern[pid] if ak[1] == 'source']
        if source_arcs:
            model.addConstr(gp.quicksum(source_arcs) == 1, name=f"tt_source_{pid}")
        
        # Sink must receive exactly 1 unit of flow
        sink_arcs = [tt_arc[ak] for ak in tt_arcs_by_pattern[pid] if ak[2] == 'sink']
        if sink_arcs:
            model.addConstr(gp.quicksum(sink_arcs) == 1, name=f"tt_sink_{pid}")
        
        # For each trip node: flow in = flow out = x[trip]
        for t in trips:
            in_arcs = tt_arcs_to.get((pid, t['id']), [])
            out_arcs = tt_arcs_from.get((pid, t['id']), [])
            
            if in_arcs:
                model.addConstr(gp.quicksum(tt_arc[ak] for ak in in_arcs) == x[t['id']],
                               name=f"tt_in_{pid}_{t['id']}")
            else:
                model.addConstr(x[t['id']] == 0, name=f"tt_noin_{pid}_{t['id']}")
            
            if out_arcs:
                model.addConstr(gp.quicksum(tt_arc[ak] for ak in out_arcs) == x[t['id']],
                               name=f"tt_out_{pid}_{t['id']}")
            else:
                # If no outgoing arcs, this trip can only be selected if it has a sink arc
                # Already handled by the in_arcs constraint
                model.addConstr(x[t['id']] == 0, name=f"tt_noout_{pid}_{t['id']}")
    
    # Vehicle scheduling network
    # Nodes: depot_minus (O-), depot_plus (O+), and for each trip: (trip_id, 'start'), (trip_id, 'end')
    # Arcs:
    # 1. Pull-out: O- -> (trip, 'start') for each trip
    # 2. Trip operation: (trip, 'start') -> (trip, 'end') for each trip
    # 3. Pull-in: (trip, 'end') -> O+ for each trip
    # 4. Deadhead/layover: (trip_i, 'end') -> (trip_j, 'start') for compatible pairs
    # 5. Circulation: O+ -> O- (with deployment cost)
    
    # VS arc flow variables
    vs_flow = {}
    vs_cost_val = {}
    
    # Number of vehicles
    num_vehicles_var = model.addVar(vtype=GRB.INTEGER, lb=0, name="num_vehicles")
    
    # Build compatible trip pairs for vehicle scheduling
    all_trips = potential_trips[:]
    all_trips.sort(key=lambda t: t['departure_time_minutes'])
    
    # For each trip, compute pull-out and pull-in costs
    def get_pull_out_time(terminal, tw_idx):
        key = (terminal, tw_idx)
        if key in pio_times:
            return pio_times[key]['pull_out']
        # Try any time window for this terminal
        for (t, twi), v in pio_times.items():
            if t == terminal:
                return v['pull_out']
        return 0
    
    def get_pull_in_time(terminal, tw_idx):
        key = (terminal, tw_idx)
        if key in pio_times:
            return pio_times[key]['pull_in']
        for (t, twi), v in pio_times.items():
            if t == terminal:
                return v['pull_in']
        return 0
    
    def get_min_stop_time(terminal, tw_idx):
        key = (terminal, tw_idx)
        if key in stop_times:
            return stop_times[key]['min']
        for (t, twi), v in stop_times.items():
            if t == terminal:
                return v['min']
        return 0
    
    def get_max_stop_time(terminal, tw_idx):
        key = (terminal, tw_idx)
        if key in stop_times:
            return stop_times[key]['max']
        for (t, twi), v in stop_times.items():
            if t == terminal:
                return v['max']
        return 99999
    
    def get_depot_min_stop(tw_idx):
        key = (depot, tw_idx)
        if key in stop_times:
            return stop_times[key]['min']
        return depot_min_stop
    
    # Pull-out arcs: depot -> trip start
    for t in potential_trips:
        tid = t['id']
        po_time = get_pull_out_time(t['start_terminal'], t['time_window_index'])
        arc_name = f"VS_O_minus-->{tid}_start"
        vs_flow[('VS_O_minus', f"({tid}, 'start')")] = model.addVar(
            vtype=GRB.BINARY, lb=0, name=arc_name)
        vs_cost_val[('VS_O_minus', f"({tid}, 'start')")] = po_time
    
    # Pull-in arcs: trip end -> depot
    for t in potential_trips:
        tid = t['id']
        pi_time = get_pull_in_time(t['end_terminal'], t['time_window_index'])
        arc_name = f"{tid}_end-->VS_O_plus"
        vs_flow[(f"({tid}, 'end')", 'VS_O_plus')] = model.addVar(
            vtype=GRB.BINARY, lb=0, name=arc_name)
        vs_cost_val[(f"({tid}, 'end')", 'VS_O_plus')] = pi_time
    
    # Trip operation arcs: trip start -> trip end (cost = 0, flow = x[trip])
    for t in potential_trips:
        tid = t['id']
        vs_flow[(f"({tid}, 'start')", f"({tid}, 'end')")] = x[tid]  # linked to selection
    
    # Deadhead/layover arcs between compatible trips
    vs_deadhead = {}
    
    for i_idx, ti in enumerate(all_trips):
        ti_end_time = ti['arrival_time_minutes']
        ti_end_terminal = ti['end_terminal']
        ti_tw = ti['time_window_index']
        
        for j_idx, tj in enumerate(all_trips):
            if ti['id'] == tj['id']:
                continue
            
            tj_start_time = tj['departure_time_minutes']
            tj_start_terminal = tj['start_terminal']
            tj_tw = tj['time_window_index']
            
            gap = tj_start_time - ti_end_time
            
            if gap < 0:
                continue
            
            cost = 0.0
            compatible = False
            
            if ti_end_terminal == tj_start_terminal:
                # In-line compatibility
                min_stop = get_min_stop_time(ti_end_terminal, ti_tw)
                max_stop = get_max_stop_time(ti_end_terminal, ti_tw)
                if min_stop <= gap <= max_stop:
                    compatible = True
                    extra_wait = gap - min_stop
                    cost = extra_wait  # extra waiting time at terminal
            else:
                # Out-line compatibility
                pi_time = get_pull_in_time(ti_end_terminal, ti_tw)
                dep_min_stop = get_depot_min_stop(ti_tw)
                po_time = get_pull_out_time(tj_start_terminal, tj_tw)
                min_gap = pi_time + dep_min_stop + po_time
                if gap >= min_gap:
                    compatible = True
                    cost = pi_time + po_time  # pull-in + pull-out travel times
            
            if compatible:
                arc_key = (f"({ti['id']}, 'end')", f"({tj['id']}, 'start')")
                vs_deadhead[(ti['id'], tj['id'])] = model.addVar(
                    vtype=GRB.BINARY, lb=0, 
                    name=f"vs_{ti['id']}_{tj['id']}")
                vs_flow[arc_key] = vs_deadhead[(ti['id'], tj['id'])]
                vs_cost_val[arc_key] = cost
    
    # Circulation arc: O+ -> O- with deployment cost
    vs_flow[('VS_O_plus', 'VS_O_minus')] = num_vehicles_var
    vs_cost_val[('VS_O_plus', 'VS_O_minus')] = deployment_cost
    
    # Flow conservation at each node
    # At depot_minus (O-): inflow from O+ = outflow to trip starts
    model.addConstr(
        num_vehicles_var == gp.quicksum(
            vs_flow[('VS_O_minus', f"({t['id']}, 'start')")] 
            for t in potential_trips),
        name="depot_minus_flow")
    
    # At depot_plus (O+): inflow from trip ends = outflow to O-
    model.addConstr(
        gp.quicksum(
            vs_flow[(f"({t['id']}, 'end')", 'VS_O_plus')] 
            for t in potential_trips) == num_vehicles_var,
        name="depot_plus_flow")
    
    # At trip start nodes: pull-out + deadhead in = trip operation out (= x[trip])
    for t in potential_trips:
        tid = t['id']
        start_node = f"({tid}, 'start')"
        
        inflow_vars = [vs_flow[('VS_O_minus', start_node)]]
        # Add deadhead arcs coming in
        for ti in potential_trips:
            if (ti['id'], tid) in vs_deadhead:
                inflow_vars.append(vs_deadhead[(ti['id'], tid)])
        
        model.addConstr(gp.quicksum(inflow_vars) == x[tid],
                       name=f"vs_start_{tid}")
    
    # At trip end nodes: trip operation in = pull-in + deadhead out
    for t in potential_trips:
        tid = t['id']
        end_node = f"({tid}, 'end')"
        
        outflow_vars = [vs_flow[(end_node, 'VS_O_plus')]]
        # Add deadhead arcs going out
        for tj in potential_trips:
            if (tid, tj['id']) in vs_deadhead:
                outflow_vars.append(vs_deadhead[(tid, tj['id'])])
        
        model.addConstr(x[tid] == gp.quicksum(outflow_vars),
                       name=f"vs_end_{tid}")
    
    # Objective function
    # TT cost: sum of quadratic penalties for consecutive trips
    tt_cost_expr = gp.quicksum(tt_cost[ak] * tt_arc[ak] for ak in tt_arc if tt_cost[ak] != 0)
    
    # VS cost: deployment cost * num_vehicles + terminal waiting + deadhead travel
    vs_cost_expr = deployment_cost * num_vehicles_var
    
    # Pull-out costs for first trips of each vehicle
    for t in potential_trips:
        tid = t['id']
        po_time = get_pull_out_time(t['start_terminal'], t['time_window_index'])
        if po_time > 0:
            vs_cost_expr += po_time * vs_flow[('VS_O_minus', f"({tid}, 'start')")]
    
    # Pull-in costs for last trips of each vehicle
    for t in potential_trips:
        tid = t['id']
        pi_time = get_pull_in_time(t['end_terminal'], t['time_window_index'])
        if pi_time > 0:
            vs_cost_expr += pi_time * vs_flow[(f"({tid}, 'end')", 'VS_O_plus')]
    
    # Deadhead costs (in-line extra waiting + out-line pull-in/pull-out)
    for (ti_id, tj_id), var in vs_deadhead.items():
        ti = trip_by_id[ti_id]
        tj = trip_by_id[tj_id]
        gap = tj['departure_time_minutes'] - ti['arrival_time_minutes']
        
        if ti['end_terminal'] == tj['start_terminal']:
            # In-line: extra waiting
            min_stop = get_min_stop_time(ti['end_terminal'], ti['time_window_index'])
            cost = gap - min_stop
        else:
            # Out-line: pull-in + pull-out
            pi_time = get_pull_in_time(ti['end_terminal'], ti['time_window_index'])
            po_time = get_pull_out_time(tj['start_terminal'], tj['time_window_index'])
            cost = pi_time + po_time
        
        if cost > 0:
            vs_cost_expr += cost * var
    
    # Total objective
    model.setObjective(alpha * vs_cost_expr + tt_cost_expr, GRB.MINIMIZE)
    
    # Callback for logging
    best_obj = [float('inf')]
    
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj_val < best_obj[0] - 1e-6:
                best_obj[0] = obj_val
                if logger:
                    logger.log(obj_val)
    
    model.optimize(callback)
    
    # Extract solution
    if model.SolCount > 0:
        obj_val = model.ObjVal
        
        selected_trips = []
        for t in potential_trips:
            if x[t['id']].X > 0.5:
                selected_trips.append(t['id'])
        
        n_vehicles = int(round(num_vehicles_var.X))
        
        # Build tt_arcs_used
        tt_arcs_used = {}
        for pid in pattern_ids:
            arcs_list = []
            for ak in tt_arcs_by_pattern[pid]:
                if tt_arc[ak].X > 0.5:
                    from_node = str(ak[1])
                    to_node = str(ak[2])
                    arcs_list.append([from_node, to_node])
            tt_arcs_used[pid] = arcs_list
        
        # Build vs_flows
        vs_flows_dict = {}
        
        # Circulation arc
        circ_key = "VS_O_plus-->VS_O_minus"
        vs_flows_dict[circ_key] = n_vehicles
        
        # Pull-out arcs
        for t in potential_trips:
            tid = t['id']
            flow_key = ('VS_O_minus', f"({tid}, 'start')")
            val = vs_flow[flow_key].X
            if val > 0.5:
                arc_str = f"VS_O_minus-->({tid}, 'start')"
                vs_flows_dict[arc_str] = 1
        
        # Pull-in arcs
        for t in potential_trips:
            tid = t['id']
            flow_key = (f"({tid}, 'end')", 'VS_O_plus')
            val = vs_flow[flow_key].X
            if val > 0.5:
                arc_str = f"({tid}, 'end')-->VS_O_plus"
                vs_flows_dict[arc_str] = 1
        
        # Trip operation arcs
        for t in potential_trips:
            tid = t['id']
            if x[tid].X > 0.5:
                arc_str = f"({tid}, 'start')-->({tid}, 'end')"
                vs_flows_dict[arc_str] = 1
        
        # Deadhead arcs
        for (ti_id, tj_id), var in vs_deadhead.items():
            if var.X > 0.5:
                arc_str = f"({ti_id}, 'end')-->({tj_id}, 'start')"
                vs_flows_dict[arc_str] = 1
        
        solution = {
            "objective_value": obj_val,
            "selected_trips": selected_trips,
            "num_vehicles": n_vehicles,
            "tt_arcs_used": tt_arcs_used,
            "vs_flows": vs_flows_dict
        }
        
        if logger:
            logger.log_solution(obj_val, solution)
        
        with open(solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
    else:
        # No feasible solution found
        solution = {
            "objective_value": float('inf'),
            "selected_trips": [],
            "num_vehicles": 0,
            "tt_arcs_used": {},
            "vs_flows": {}
        }
        with open(solution_path, 'w') as f:
            json.dump(solution, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)
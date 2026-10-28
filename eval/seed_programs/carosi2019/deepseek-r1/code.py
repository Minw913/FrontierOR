import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def load_instance(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)

def build_solution(model, x, y, f, patterns, TT_arcs, in_line_arcs, out_line_arcs, trip_ids, circ_arc):
    solution_dict = {}
    
    selected_trips = [trip_id for trip_id in trip_ids if x[trip_id].X > 0.5]
    solution_dict["selected_trips"] = selected_trips
    
    tt_arcs_used = {}
    for p in patterns:
        chain_arcs = []
        source = f"source_{p}"
        sink = f"sink_{p}"
        for arc in TT_arcs[p]:
            if y[arc].X > 0.5:
                chain_arcs.append([str(arc[0]), str(arc[1])])
        tt_arcs_used[p] = chain_arcs
    solution_dict["tt_arcs_used"] = tt_arcs_used
    
    vs_flows = {}
    vs_flows["VS_O_plus-->VS_O_minus"] = f[circ_arc].X
    
    for trip_id in trip_ids:
        arc_start = (trip_id, 'start')
        arc_end = (trip_id, 'end')
        arc_key = f"({trip_id}, 'start')-->({trip_id}, 'end')"
        vs_flows[arc_key] = f[(arc_start, arc_end)].X
    
    for trip_id in trip_ids:
        arc_pull_out = ('VS_O_minus', (trip_id, 'start'))
        arc_key = f"VS_O_minus-->({trip_id}, 'start')"
        vs_flows[arc_key] = f[arc_pull_out].X
    
    for trip_id in trip_ids:
        arc_pull_in = ((trip_id, 'end'), 'VS_O_plus')
        arc_key = f"({trip_id}, 'end')-->VS_O_plus"
        vs_flows[arc_key] = f[arc_pull_in].X
    
    for (i, j) in in_line_arcs:
        from_node = (i, 'end')
        to_node = (j, 'start')
        arc_key = f"({i}, 'end')-->({j}, 'start')"
        arc = (from_node, to_node)
        vs_flows[arc_key] = f[arc].X
    
    for (i, j) in out_line_arcs:
        from_node = (i, 'end')
        to_node = (j, 'start')
        arc_key = f"({i}, 'end')-->({j}, 'start')"
        arc = (from_node, to_node)
        vs_flows[arc_key] = f[arc].X
    
    solution_dict["vs_flows"] = vs_flows
    solution_dict["num_vehicles"] = f[circ_arc].X
    
    return solution_dict

def solve_instance(instance, time_limit, logger=None):
    model = gp.Model("Integrated_Timetabling_Vehicle_Scheduling")
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', time_limit)
    
    terminals = instance['topology']['terminals']
    depot = instance['topology']['depot']
    time_windows = instance['time_windows']
    directions = instance['directions']
    patterns = {d['pattern_id']: d for d in directions}
    trips_list = instance['potential_trips']
    trips = {trip['id']: trip for trip in trips_list}
    trip_ids = list(trips.keys())
    stopping_times = instance['stopping_times']
    pull_in_out_times = instance['pull_in_out_times']
    initial_trip_time_windows = instance['initial_trip_time_windows']
    final_trip_time_windows = instance['final_trip_time_windows']
    alpha = instance['objective_function']['alpha']
    per_vehicle_cost = instance.get('per_vehicle_deployment_cost', 1000)
    
    for trip in trips.values():
        trip['departure_time_seconds'] = trip['departure_time_minutes'] * 60
        trip['arrival_time_seconds'] = trip['arrival_time_minutes'] * 60
        trip['main_stop_arrival_time_seconds'] = trip['main_stop_arrival_time_minutes'] * 60
    
    x = model.addVars(trip_ids, vtype=GRB.BINARY, name="x")
    
    y = {}
    TT_arcs = {p: [] for p in patterns}
    TT_arc_costs = {}
    
    for p in patterns:
        pattern_trips = [trip_id for trip_id in trip_ids if trips[trip_id]['pattern_id'] == p]
        source = f"source_{p}"
        sink = f"sink_{p}"
        
        initial_tw = initial_trip_time_windows[p]
        for trip_id in pattern_trips:
            if trips[trip_id]['time_window_index'] == initial_tw:
                arc = (source, trip_id)
                TT_arcs[p].append(arc)
                y[arc] = model.addVar(vtype=GRB.BINARY, name=f"y_{p}_{source}_{trip_id}")
        
        final_tw = final_trip_time_windows[p]
        for trip_id in pattern_trips:
            if trips[trip_id]['time_window_index'] == final_tw:
                arc = (trip_id, sink)
                TT_arcs[p].append(arc)
                y[arc] = model.addVar(vtype=GRB.BINARY, name=f"y_{p}_{trip_id}_{sink}")
        
        pattern_trips_sorted = sorted(pattern_trips, key=lambda tid: trips[tid]['main_stop_arrival_time_seconds'])
        for i in range(len(pattern_trips_sorted)):
            trip_i = pattern_trips_sorted[i]
            t_i = trips[trip_i]
            for j in range(i+1, len(pattern_trips_sorted)):
                trip_j = pattern_trips_sorted[j]
                t_j = trips[trip_j]
                headway = t_j['main_stop_arrival_time_seconds'] - t_i['main_stop_arrival_time_seconds']
                tw_i = t_i['time_window_index']
                min_headway = next(tw for tw in time_windows if tw['index'] == tw_i)['min_headway_seconds']
                max_headway = next(tw for tw in time_windows if tw['index'] == tw_i)['max_headway_seconds']
                ideal_headway = next(tw for tw in time_windows if tw['index'] == tw_i)['ideal_headway_seconds']
                
                if min_headway <= headway <= max_headway:
                    arc = (trip_i, trip_j)
                    TT_arcs[p].append(arc)
                    y[arc] = model.addVar(vtype=GRB.BINARY, name=f"y_{p}_{trip_i}_{trip_j}")
                    TT_arc_costs[arc] = (headway - ideal_headway) ** 2
    
    for p in patterns:
        source = f"source_{p}"
        sink = f"sink_{p}"
        pattern_trips = [trip_id for trip_id in trip_ids if trips[trip_id]['pattern_id'] == p]
        
        model.addConstr(gp.quicksum(y[arc] for arc in TT_arcs[p] if arc[0] == source) == 1, f"TT_source_{p}")
        model.addConstr(gp.quicksum(y[arc] for arc in TT_arcs[p] if arc[1] == sink) == 1, f"TT_sink_{p}")
        
        for trip_id in pattern_trips:
            in_arcs = [arc for arc in TT_arcs[p] if arc[1] == trip_id]
            out_arcs = [arc for arc in TT_arcs[p] if arc[0] == trip_id]
            model.addConstr(gp.quicksum(y[arc] for arc in in_arcs) == x[trip_id], f"TT_in_{trip_id}")
            model.addConstr(gp.quicksum(y[arc] for arc in out_arcs) == x[trip_id], f"TT_out_{trip_id}")
    
    in_line_arcs = []
    out_line_arcs = []
    
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
                stop_info_list = stopping_times[end_term_i]
                stop_info = next((st for st in stop_info_list if st['time_window_index'] == tw_i), None)
                if stop_info is None:
                    continue
                min_stop = stop_info['min_stopping_time_minutes'] * 60
                max_stop = stop_info['max_stopping_time_minutes'] * 60
                if min_stop <= gap <= max_stop:
                    in_line_arcs.append((i, j))
            else:
                pull_in_i_list = pull_in_out_times[end_term_i]
                pull_in_i_entry = next((pt for pt in pull_in_i_list if pt['time_window_index'] == tw_i), None)
                if pull_in_i_entry is None:
                    continue
                pull_in_i = pull_in_i_entry['pull_in_time_minutes'] * 60
                
                pull_out_j_list = pull_in_out_times[start_term_j]
                pull_out_j_entry = next((pt for pt in pull_out_j_list if pt['time_window_index'] == tw_j), None)
                if pull_out_j_entry is None:
                    continue
                pull_out_j = pull_out_j_entry['pull_out_time_minutes'] * 60
                
                min_depot_stop = 0
                if depot in stopping_times:
                    depot_stop_list = stopping_times[depot]
                    depot_stop = next((st for st in depot_stop_list if st['time_window_index'] == tw_i), None)
                    if depot_stop:
                        min_depot_stop = depot_stop['min_stopping_time_minutes'] * 60
                
                if gap >= pull_in_i + min_depot_stop + pull_out_j:
                    out_line_arcs.append((i, j))
    
    vs_arcs = []
    f_vars = {}
    
    for i in trip_ids:
        arc_out = ('VS_O_minus', (i, 'start'))
        vs_arcs.append(arc_out)
        f_vars[arc_out] = model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"f_pull_out_{i}")
    
    for i in trip_ids:
        arc_start = (i, 'start')
        arc_end = (i, 'end')
        arc_op = (arc_start, arc_end)
        vs_arcs.append(arc_op)
        f_vars[arc_op] = model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"f_trip_{i}")
        model.addConstr(f_vars[arc_op] == x[i], f"VS_trip_op_{i}")
    
    for (i, j) in in_line_arcs:
        from_node = (i, 'end')
        to_node = (j, 'start')
        arc = (from_node, to_node)
        vs_arcs.append(arc)
        f_vars[arc] = model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"f_in_line_{i}_{j}")
    
    for (i, j) in out_line_arcs:
        from_node = (i, 'end')
        to_node = (j, 'start')
        arc = (from_node, to_node)
        vs_arcs.append(arc)
        f_vars[arc] = model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"f_out_line_{i}_{j}")
    
    for i in trip_ids:
        arc_in = ((i, 'end'), 'VS_O_plus')
        vs_arcs.append(arc_in)
        f_vars[arc_in] = model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"f_pull_in_{i}")
    
    circ_arc = ('VS_O_plus', 'VS_O_minus')
    vs_arcs.append(circ_arc)
    f_vars[circ_arc] = model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.INTEGER, name="f_circ")

    model.addConstr(
        gp.quicksum(f_vars[arc] for arc in vs_arcs if arc[0] == 'VS_O_minus') == f_vars[circ_arc],
        "VS_O_minus_flow"
    )
    model.addConstr(
        gp.quicksum(f_vars[arc] for arc in vs_arcs if arc[1] == 'VS_O_plus') == f_vars[circ_arc],
        "VS_O_plus_flow"
    )
    
    for trip_id in trip_ids:
        node_start = (trip_id, 'start')
        in_flow = gp.quicksum(f_vars[arc] for arc in vs_arcs if arc[1] == node_start)
        out_flow = gp.quicksum(f_vars[arc] for arc in vs_arcs if arc[0] == node_start)
        model.addConstr(in_flow == out_flow, f"VS_conserve_start_{trip_id}")
    
    for trip_id in trip_ids:
        node_end = (trip_id, 'end')
        in_flow = gp.quicksum(f_vars[arc] for arc in vs_arcs if arc[1] == node_end)
        out_flow = gp.quicksum(f_vars[arc] for arc in vs_arcs if arc[0] == node_end)
        model.addConstr(in_flow == out_flow, f"VS_conserve_end_{trip_id}")
    
    timetable_cost_expr = gp.quicksum(
        TT_arc_costs[arc] * y[arc] 
        for arc in TT_arc_costs.keys()
    )
    
    vs_cost_components = []
    
    vs_cost_components.append(per_vehicle_cost * f_vars[circ_arc])
    
    for i in trip_ids:
        arc = ('VS_O_minus', (i, 'start'))
        start_term_i = trips[i]['start_terminal']
        tw_i = trips[i]['time_window_index']
        pull_out_entry = next(
            pt for pt in pull_in_out_times[start_term_i] 
            if pt['time_window_index'] == tw_i
        )
        pull_out_time = pull_out_entry['pull_out_time_minutes']
        vs_cost_components.append(pull_out_time * f_vars[arc])
    
    for i in trip_ids:
        arc = ((i, 'end'), 'VS_O_plus')
        end_term_i = trips[i]['end_terminal']
        tw_i = trips[i]['time_window_index']
        pull_in_entry = next(
            pt for pt in pull_in_out_times[end_term_i] 
            if pt['time_window_index'] == tw_i
        )
        pull_in_time = pull_in_entry['pull_in_time_minutes']
        vs_cost_components.append(pull_in_time * f_vars[arc])
    
    for (i, j) in in_line_arcs:
        arc = ((i, 'end'), (j, 'start'))
        gap = trips[j]['departure_time_seconds'] - trips[i]['arrival_time_seconds']
        tw_i = trips[i]['time_window_index']
        end_term_i = trips[i]['end_terminal']
        stop_info = next(
            st for st in stopping_times[end_term_i] 
            if st['time_window_index'] == tw_i
        )
        min_stop_seconds = stop_info['min_stopping_time_minutes'] * 60
        extra_waiting_minutes = (gap - min_stop_seconds) / 60.0
        vs_cost_components.append(extra_waiting_minutes * f_vars[arc])
    
    for (i, j) in out_line_arcs:
        arc = ((i, 'end'), (j, 'start'))
        tw_i = trips[i]['time_window_index']
        end_term_i = trips[i]['end_terminal']
        tw_j = trips[j]['time_window_index']
        start_term_j = trips[j]['start_terminal']
        pull_in_i_entry = next(pt for pt in pull_in_out_times[end_term_i] if pt['time_window_index'] == tw_i)
        pull_in_i = pull_in_i_entry['pull_in_time_minutes']
        pull_out_j_entry = next(pt for pt in pull_in_out_times[start_term_j] if pt['time_window_index'] == tw_j)
        pull_out_j = pull_out_j_entry['pull_out_time_minutes']
        vs_cost_components.append((pull_in_i + pull_out_j) * f_vars[arc])
    
    vs_cost_expr = gp.quicksum(vs_cost_components)
    total_cost = alpha * vs_cost_expr + timetable_cost_expr
    model.setObjective(total_cost, GRB.MINIMIZE)
    
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj)
    
    model.optimize(callback if logger else None)
    
    if model.SolCount > 0:
        solution_dict = build_solution(
            model, x, y, f_vars, list(patterns.keys()), TT_arcs, 
            in_line_arcs, out_line_arcs, trip_ids, circ_arc
        )
        obj_value = model.ObjVal
    else:
        solution_dict = {
            "selected_trips": [],
            "tt_arcs_used": {},
            "vs_flows": {},
            "num_vehicles": 0
        }
        obj_value = 0.0
    
    solution_dict["objective_value"] = obj_value
    
    if logger:
        logger.log_solution(obj_value, solution_dict)
    
    return solution_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    instance = load_instance(args.instance_path)
    solution = solve_instance(instance, args.time_limit, logger)
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()
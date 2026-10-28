import argparse
import json
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger
import time

def read_instance(instance_path):
    with open(instance_path, 'r') as file:
        return json.load(file)

def solve_instance(instance):
    directions = instance['directions']
    time_windows = instance['time_windows']
    stopping_times = instance['stopping_times']
    pull_in_out_times = instance['pull_in_out_times']
    potential_trips = instance['potential_trips']
    alpha = instance['objective_function']['alpha']
    per_vehicle_cost = 1000  
    
    model = Model("BusScheduling")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    
    trip_selection = {}
    for trip in potential_trips:
        trip_selection[trip['id']] = model.addVar(vtype=GRB.BINARY, name=f"trip_{trip['id']}")
    
    vehicle_arcs = {}
    depot_node_minus = 'VS_O_minus'
    depot_node_plus = 'VS_O_plus'
    for trip in potential_trips:
        start_node = f"VS_{trip['id']}_start"
        end_node = f"VS_{trip['id']}_end"
        vehicle_arcs[f'{depot_node_minus}-->{start_node}'] = model.addVar(vtype=GRB.BINARY, name=f'{depot_node_minus}-->{start_node}')
        vehicle_arcs[f'{end_node}-->{depot_node_plus}'] = model.addVar(vtype=GRB.BINARY, name=f'{end_node}-->{depot_node_plus}')
        for next_trip in potential_trips:
            if trip != next_trip:
                next_start_node = f"VS_{next_trip['id']}_start"
                arc_name = f'{end_node}-->{next_start_node}'
                vehicle_arcs[arc_name] = model.addVar(vtype=GRB.BINARY, name=arc_name)
    vehicle_arcs[f'{depot_node_plus}-->{depot_node_minus}'] = model.addVar(vtype=GRB.INTEGER, name=f'{depot_node_plus}-->{depot_node_minus}')
    
    model.update()
    
    timetable_cost = quicksum(((trip_selection[trip['id']] * trip_selection[next_trip['id']] * 
                               ((next_trip['main_stop_arrival_time_minutes'] - trip['main_stop_arrival_time_minutes']) * 60 - 
                                time_windows[trip['time_window_index']]['ideal_headway_seconds']) ** 2)
                              for direction in directions 
                              for trip, next_trip in 
                              zip(sorted([t for t in potential_trips if t['pattern_id'] == direction['pattern_id']], key=lambda x: x['departure_time_minutes'])[:-1], 
                                  sorted([t for t in potential_trips if t['pattern_id'] == direction['pattern_id']], key=lambda x: x['departure_time_minutes'])[1:])))
    vehicle_scheduling_cost = per_vehicle_cost * vehicle_arcs[f'{depot_node_plus}-->{depot_node_minus}'] 
    model.setObjective(alpha * vehicle_scheduling_cost + timetable_cost, GRB.MINIMIZE)
    
    for trip in potential_trips:
        inflow = quicksum(vehicle_arcs.get(arc, model.addVar(vtype=GRB.CONTINUOUS, name=arc, lb=0, ub=0)) for arc in vehicle_arcs if f'-->{trip["id"]}_start' in arc)
        outflow = quicksum(vehicle_arcs.get(arc, model.addVar(vtype=GRB.CONTINUOUS, name=arc, lb=0, ub=0)) for arc in vehicle_arcs if f'{trip["id"]}_end-->' in arc)
        model.addConstr(inflow == trip_selection[trip['id']], name=f'cover_trip_{trip["id"]}')
        model.addConstr(outflow == trip_selection[trip['id']], name=f'outflow_trip_{trip["id"]}')
    
    model.addConstr(quicksum(vehicle_arcs.get(arc, model.addVar(vtype=GRB.CONTINUOUS, name=arc, lb=0, ub=0)) for arc in vehicle_arcs if f'{depot_node_minus}-->' in arc) == vehicle_arcs[f'{depot_node_plus}-->{depot_node_minus}'], name='depot_outflow')
    model.addConstr(quicksum(vehicle_arcs.get(arc, model.addVar(vtype=GRB.CONTINUOUS, name=arc, lb=0, ub=0)) for arc in vehicle_arcs if f'-->{depot_node_plus}' in arc) == vehicle_arcs[f'{depot_node_plus}-->{depot_node_minus}'], name='depot_inflow')
    
    for direction in directions:
        pattern_id = direction['pattern_id']
        trips_in_direction = sorted([trip for trip in potential_trips if trip['pattern_id'] == pattern_id], key=lambda x: x['departure_time_minutes'])
        for i in range(len(trips_in_direction) - 1):
            trip = trips_in_direction[i]
            next_trip = trips_in_direction[i+1]
            time_window_index = trip['time_window_index']
            min_headway = time_windows[time_window_index]['min_headway_seconds']
            max_headway = time_windows[time_window_index]['max_headway_seconds']
            headway = (next_trip['main_stop_arrival_time_minutes'] - trip['main_stop_arrival_time_minutes']) * 60
            model.addConstr((trip_selection[trip['id']] + trip_selection[next_trip['id']] - 1) * min_headway <= headway, name=f'headway_min_{trip["id"]}_{next_trip["id"]}')
            model.addConstr((trip_selection[trip['id']] + trip_selection[next_trip['id']] - 1) * headway <= max_headway, name=f'headway_max_{trip["id"]}_{next_trip["id"]}')
    
    model.update()
    model.optimize()
    
    solution = {}
    solution['objective_value'] = model.ObjVal if model.status == GRB.OPTIMAL else float('inf')
    solution['selected_trips'] = [trip['id'] for trip in potential_trips if trip_selection[trip['id']].X > 0.5]
    solution['num_vehicles'] = int(vehicle_arcs[f'{depot_node_plus}-->{depot_node_minus}'].X)
    solution['tt_arcs_used'] = {}
    for direction in directions:
        pattern_id = direction['pattern_id']
        trips_in_direction = sorted([trip for trip in potential_trips if trip['pattern_id'] == pattern_id and trip_selection[trip['id']].X > 0.5], key=lambda x: x['departure_time_minutes'])
        solution['tt_arcs_used'][pattern_id] = [[str(trip['id'])] for trip in trips_in_direction]
    solution['vs_flows'] = {arc: var.X for arc, var in vehicle_arcs.items()}
    
    return solution

def main():
    parser = argparse.ArgumentParser(description='Solve bus scheduling problem')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the instance JSON file')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to write the solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds')
    parser.add_argument('--log_path', type=str, help='Path to log intermediate solutions')
    args = parser.parse_args()
    
    instance = read_instance(args.instance_path)
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    model = Model("BusScheduling")
    model.Params.TimeLimit = args.time_limit
    
    start_time = time.time()
    solution = solve_instance(instance)
    end_time = time.time()
    
    if logger:
        logger.log_solution(solution['objective_value'], solution)
    
    with open(args.solution_path, 'w') as file:
        json.dump(solution, file, indent=4)

if __name__ == "__main__":
    main()
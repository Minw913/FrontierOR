import argparse
import json
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger
import time

def read_instance(instance_path):
    with open(instance_path, 'r') as file:
        return json.load(file)

def solve_instance(instance):
    # Extract relevant data from the instance
    directions = instance['directions']
    time_windows = instance['time_windows']
    stopping_times = instance['stopping_times']
    pull_in_out_times = instance['pull_in_out_times']
    potential_trips = instance['potential_trips']
    alpha = instance['objective_function']['alpha']
    per_vehicle_cost = 1000  # Assuming a large per-vehicle deployment cost
    
    # Create a Gurobi model
    model = Model("BusScheduling")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    
    # Decision variables
    trip_selection = {}
    for trip in potential_trips:
        trip_selection[trip['id']] = model.addVar(vtype=GRB.BINARY, name=f"trip_{trip['id']}")
    
    vehicle_arcs = {}
    for direction in directions:
        pattern_id = direction['pattern_id']
        trips_in_direction = [trip for trip in potential_trips if trip['pattern_id'] == pattern_id]
        trips_in_direction.sort(key=lambda x: x['departure_time_minutes'])
        for i in range(len(trips_in_direction) - 1):
            trip1 = trips_in_direction[i]
            trip2 = trips_in_direction[i+1]
            headway = trip2['main_stop_arrival_time_minutes'] - trip1['main_stop_arrival_time_minutes']
            time_window_index = trip1['time_window_index']
            ideal_headway = time_windows[time_window_index]['ideal_headway_seconds'] / 60
            min_headway = time_windows[time_window_index]['min_headway_seconds'] / 60
            max_headway = time_windows[time_window_index]['max_headway_seconds'] / 60
            if min_headway <= headway <= max_headway:
                arc_name = f"vs_{trip1['id']}-->{trip2['id']}"
                vehicle_arcs[arc_name] = model.addVar(vtype=GRB.BINARY, name=arc_name)
    
    # Additional vehicle scheduling arcs and variables
    depot_node = 'VS_O'
    vehicle_arcs[f'{depot_node}_plus-->{depot_node}_minus'] = model.addVar(vtype=GRB.INTEGER, name=f'{depot_node}_plus-->{depot_node}_minus')
    for trip in potential_trips:
        start_node = f"VS_{trip['id']}_start"
        end_node = f"VS_{trip['id']}_end"
        vehicle_arcs[f'{depot_node}_minus-->{start_node}'] = model.addVar(vtype=GRB.BINARY, name=f'{depot_node}_minus-->{start_node}')
        vehicle_arcs[f'{end_node}-->{depot_node}_plus'] = model.addVar(vtype=GRB.BINARY, name=f'{end_node}-->{depot_node}_plus')
        for next_trip in potential_trips:
            if trip['end_terminal'] == next_trip['start_terminal']:
                next_start_node = f"VS_{next_trip['id']}_start"
                arc_name = f'{end_node}-->{next_start_node}'
                vehicle_arcs[arc_name] = model.addVar(vtype=GRB.BINARY, name=arc_name)
            else:
                pull_out_time = pull_in_out_times[next_trip['start_terminal']][next_trip['time_window_index']]['pull_out_time_minutes']
                pull_in_time = pull_in_out_times[trip['end_terminal']][trip['time_window_index']]['pull_in_time_minutes']
                min_depot_stopping_time = 0  # Assuming minimum depot stopping time is 0 for simplicity
                if trip['arrival_time_minutes'] + pull_in_time + min_depot_stopping_time + pull_out_time <= next_trip['departure_time_minutes']:
                    next_start_node = f"VS_{next_trip['id']}_start"
                    arc_name = f'{end_node}-->{next_start_node}'
                    vehicle_arcs[arc_name] = model.addVar(vtype=GRB.BINARY, name=arc_name)
    
    model.update()
    
    # Objective function
    timetable_cost = quicksum((trip_selection[trip['id']] * trip_selection[next_trip['id']] * 
                               ((next_trip['main_stop_arrival_time_minutes'] - trip['main_stop_arrival_time_minutes']) - 
                                time_windows[trip['time_window_index']]['ideal_headway_seconds'] / 60) ** 2)
                              for direction in directions for trip, next_trip in 
                              zip([t for t in potential_trips if t['pattern_id'] == direction['pattern_id']], 
                                  [t for t in potential_trips if t['pattern_id'] == direction['pattern_id']][1:]))
    vehicle_scheduling_cost = per_vehicle_cost * vehicle_arcs[f'{depot_node}_plus-->{depot_node}_minus'] + \
                              quicksum(vehicle_arcs[arc] * (pull_in_out_times[trip['end_terminal']][trip['time_window_index']]['pull_in_time_minutes'] + 
                                                            pull_in_out_times[next_trip['start_terminal']][next_trip['time_window_index']]['pull_out_time_minutes'])
                                       for arc, var in vehicle_arcs.items() if '-->' in arc and 'VS_' in arc)
    model.setObjective(alpha * vehicle_scheduling_cost + timetable_cost, GRB.MINIMIZE)
    
    # Constraints
    for trip in potential_trips:
        model.addConstr(quicksum(vehicle_arcs[arc] for arc in vehicle_arcs if f'-->{trip["id"]}' in arc or f'{trip["id"]}-->' in arc) == trip_selection[trip['id']])
    
    model.addConstr(quicksum(vehicle_arcs[f'{depot_node}_minus-->{trip["id"]}_start'] for trip in potential_trips) == vehicle_arcs[f'{depot_node}_plus-->{depot_node}_minus'])
    model.addConstr(quicksum(vehicle_arcs[f'{trip["id"]}_end-->{depot_node}_plus'] for trip in potential_trips) == vehicle_arcs[f'{depot_node}_plus-->{depot_node}_minus'])
    
    for direction in directions:
        pattern_id = direction['pattern_id']
        admissible_first_trips = [trip for trip in potential_trips if trip['pattern_id'] == pattern_id and trip['time_window_index'] == instance['initial_trip_time_windows'][pattern_id]]
        admissible_last_trips = [trip for trip in potential_trips if trip['pattern_id'] == pattern_id and trip['time_window_index'] == instance['final_trip_time_windows'][pattern_id]]
        model.addConstr(quicksum(trip_selection[trip['id']] for trip in admissible_first_trips) == 1)
        model.addConstr(quicksum(trip_selection[trip['id']] for trip in admissible_last_trips) == 1)
        trips_in_direction = sorted([trip for trip in potential_trips if trip['pattern_id'] == pattern_id], key=lambda x: x['departure_time_minutes'])
        for i in range(len(trips_in_direction) - 1):
            trip = trips_in_direction[i]
            next_trip = trips_in_direction[i+1]
            model.addConstr(trip_selection[next_trip['id']] <= trip_selection[trip['id']])
    
    model.update()
    model.optimize()
    
    # Construct solution dictionary
    solution = {}
    solution['objective_value'] = model.ObjVal
    solution['selected_trips'] = [trip['id'] for trip in potential_trips if trip_selection[trip['id']].X > 0.5]
    solution['num_vehicles'] = int(vehicle_arcs[f'{depot_node}_plus-->{depot_node}_minus'].X)
    solution['tt_arcs_used'] = {}
    for direction in directions:
        pattern_id = direction['pattern_id']
        trips_in_direction = sorted([trip for trip in potential_trips if trip['pattern_id'] == pattern_id and trip_selection[trip['id']].X > 0.5], key=lambda x: x['departure_time_minutes'])
        solution['tt_arcs_used'][pattern_id] = [[str(trips_in_direction[i]['id']), str(trips_in_direction[i+1]['id'])] for i in range(len(trips_in_direction) - 1)]
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
    # ... (rest of the model setup and optimization)
    
    start_time = time.time()
    solution = solve_instance(instance)
    end_time = time.time()
    
    if logger:
        logger.log_solution(solution['objective_value'], solution)
    
    with open(args.solution_path, 'w') as file:
        json.dump(solution, file, indent=4)

if __name__ == "__main__":
    main()
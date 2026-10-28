import argparse
import json
from math import inf
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    depots = instance['depots']
    trips = instance['trips']
    travel_times = instance['travel_times']
    relief_locations = set(instance['relief_locations'])
    duty_types = instance['duty_types']
    crew_parameters = instance['crew_parameters']
    vehicle_parameters = instance['vehicle_parameters']

    trips_sorted = sorted(trips, key=lambda x: x['start_time'])
    depot_id_to_name = {depot['depot_id']: depot['name'] for depot in depots}
    depot_name_to_id = {depot['name']: depot['depot_id'] for depot in depots}
    depots_by_id = {depot['depot_id']: depot for depot in depots}

    vehicles = []
    for trip in trips_sorted:
        assigned = False
        best_vehicle_idx = None
        best_depot_id = None
        for idx, veh in enumerate(vehicles):
            depot_id = veh['depot_id']
            current_loc = veh['current_location']
            avail_time = veh['available_time']
            depot_name = depot_id_to_name[depot_id]
            if current_loc == depot_name:
                tt = travel_times[depot_name][trip['start_location']]
                depart_time = trip['start_time'] - tt
                if avail_time <= depart_time:
                    best_vehicle_idx = idx
                    best_depot_id = depot_id
                    assigned = True
                    break
            else:
                tt_to_depot = travel_times[current_loc][depot_name]
                tt_depot_to_start = travel_times[depot_name][trip['start_location']]
                total_tt = tt_to_depot + tt_depot_to_start
                gap = trip['start_time'] - avail_time
                if gap >= total_tt:
                    if avail_time + tt_to_depot <= trip['start_time'] - tt_depot_to_start:
                        best_vehicle_idx = idx
                        best_depot_id = depot_id
                        assigned = True
                        break
                else:
                    tt_direct = travel_times[current_loc][trip['start_location']]
                    if avail_time + tt_direct <= trip['start_time']:
                        best_vehicle_idx = idx
                        best_depot_id = depot_id
                        assigned = True
                        break
        if assigned:
            veh = vehicles[best_vehicle_idx]
            depot_id = best_depot_id
            depot_name = depot_id_to_name[depot_id]
            if veh['current_location'] == depot_name:
                new_chain = [trip['trip_id']]
                veh['current_chain'] = new_chain
            else:
                gap = trip['start_time'] - veh['available_time']
                tt_to_depot = travel_times[veh['current_location']][depot_name]
                tt_depot_to_start = travel_times[depot_name][trip['start_location']]
                if gap >= tt_to_depot + tt_depot_to_start:
                    veh['all_chains'].append(veh['current_chain'])
                    new_chain = [trip['trip_id']]
                    veh['current_chain'] = new_chain
                else:
                    veh['current_chain'].append(trip['trip_id'])
            veh['current_location'] = trip['end_location']
            veh['available_time'] = trip['end_time']
        else:
            best_depot_id = None
            min_tt = inf
            for depot in depots:
                depot_name = depot['name']
                tt = travel_times[depot_name][trip['start_location']]
                if tt < min_tt:
                    min_tt = tt
                    best_depot_id = depot['depot_id']
            new_vehicle = {
                'depot_id': best_depot_id,
                'current_location': trip['end_location'],
                'available_time': trip['end_time'],
                'current_chain': [trip['trip_id']],
                'all_chains': []
            }
            vehicles.append(new_vehicle)

    for veh in vehicles:
        if veh['current_chain']:
            veh['all_chains'].append(veh['current_chain'])
            veh['current_chain'] = None

    vehicle_schedule = {}
    for depot in depots:
        depot_name = depot['name']
        chains = []
        for veh in vehicles:
            if veh['depot_id'] == depot['depot_id']:
                chains.extend(veh['all_chains'])
        vehicle_schedule[depot_name] = chains

    def split_chain_into_pieces(chain_trips, depot_name, trips_dict, travel_times, relief_locations_set):
        if not chain_trips:
            return []
        tasks = []
        prev_trip_end_location = None
        prev_trip_end_time = None
        for i, trip_id in enumerate(chain_trips):
            trip = trips_dict[trip_id]
            if i == 0:
                deadhead_start_time = trip['start_time'] - travel_times[depot_name][trip['start_location']]
                deadhead_end_time = trip['start_time']
                tasks.append({
                    'type': 'deadhead',
                    'start_time': deadhead_start_time,
                    'end_time': deadhead_end_time,
                    'start_location': depot_name,
                    'end_location': trip['start_location'],
                    'trip_id': None
                })
            else:
                deadhead_start_time = prev_trip_end_time
                deadhead_end_time = prev_trip_end_time + travel_times[prev_trip_end_location][trip['start_location']]
                tasks.append({
                    'type': 'deadhead',
                    'start_time': deadhead_start_time,
                    'end_time': deadhead_end_time,
                    'start_location': prev_trip_end_location,
                    'end_location': trip['start_location'],
                    'trip_id': None
                })
            tasks.append({
                'type': 'trip',
                'start_time': trip['start_time'],
                'end_time': trip['end_time'],
                'start_location': trip['start_location'],
                'end_location': trip['end_location'],
                'trip_id': trip_id
            })
            prev_trip_end_location = trip['end_location']
            prev_trip_end_time = trip['end_time']
        last_trip = trips_dict[chain_trips[-1]]
        tasks.append({
            'type': 'deadhead',
            'start_time': last_trip['end_time'],
            'end_time': last_trip['end_time'] + travel_times[last_trip['end_location']][depot_name],
            'start_location': last_trip['end_location'],
            'end_location': depot_name,
            'trip_id': None
        })
        pieces = []
        current_piece = []
        for task in tasks:
            current_piece.append(task)
            if task['end_location'] in relief_locations_set:
                pieces.append(current_piece)
                current_piece = []
        if current_piece:
            pieces.append(current_piece)
        result = []
        for piece_tasks in pieces:
            if not piece_tasks:
                continue
            start_time = piece_tasks[0]['start_time']
            end_time = piece_tasks[-1]['end_time']
            start_location = piece_tasks[0]['start_location']
            end_location = piece_tasks[-1]['end_location']
            trip_ids = [t['trip_id'] for t in piece_tasks if t['type'] == 'trip']
            result.append({
                'start_time': start_time,
                'end_time': end_time,
                'start_location': start_location,
                'end_location': end_location,
                'trip_ids': trip_ids
            })
        return result

    trips_dict = {trip['trip_id']: trip for trip in trips}
    all_pieces_by_depot = {}
    for depot in depots:
        depot_name = depot['name']
        chains = vehicle_schedule.get(depot_name, [])
        pieces = []
        for chain in chains:
            pieces_chain = split_chain_into_pieces(chain, depot_name, trips_dict, travel_times, relief_locations)
            pieces.extend(pieces_chain)
        all_pieces_by_depot[depot_name] = pieces

    def satisfies_one_piece(piece, dt, depot_name, crew_params, travel_times):
        work_duration = piece['end_time'] - piece['start_time']
        if work_duration < dt['piece_length_min'] or work_duration > dt['piece_length_max']:
            return False
        start_loc = piece['start_location']
        end_loc = piece['end_location']
        if start_loc == depot_name:
            duty_start = piece['start_time'] - crew_params['sign_on_time_depot_minutes']
        else:
            extra_time = crew_params['extra_time_non_depot_relief_minutes']
            if not crew_params['extra_time_includes_deadhead_to_depot']:
                extra_time += travel_times[start_loc][depot_name]
            duty_start = piece['start_time'] - extra_time
        if end_loc == depot_name:
            duty_end = piece['end_time'] + crew_params['sign_off_time_depot_minutes']
        else:
            extra_time = crew_params['extra_time_non_depot_relief_minutes']
            if not crew_params['extra_time_includes_deadhead_to_depot']:
                extra_time += travel_times[end_loc][depot_name]
            duty_end = piece['end_time'] + extra_time
        duty_span = duty_end - duty_start
        if dt['duty_length_max'] is not None and duty_span > dt['duty_length_max']:
            return False
        if dt['work_time_max'] is not None and work_duration > dt['work_time_max']:
            return False
        if dt['start_time_min'] is not None and duty_start < dt['start_time_min']:
            return False
        if dt['start_time_max'] is not None and duty_start > dt['start_time_max']:
            return False
        if dt['end_time_max'] is not None and duty_end > dt['end_time_max']:
            return False
        return True

    def satisfies_two_piece(piece1, piece2, dt, depot_name, crew_params, travel_times):
        if piece1['end_time'] > piece2['start_time']:
            return False
        break_duration = piece2['start_time'] - piece1['end_time']
        if break_duration < dt['break_length_min']:
            return False
        work_duration1 = piece1['end_time'] - piece1['start_time']
        work_duration2 = piece2['end_time'] - piece2['start_time']
        if (work_duration1 < dt['piece_length_min'] or work_duration1 > dt['piece_length_max'] or
            work_duration2 < dt['piece_length_min'] or work_duration2 > dt['piece_length_max']):
            return False
        start_loc1 = piece1['start_location']
        end_loc2 = piece2['end_location']
        if start_loc1 == depot_name:
            duty_start = piece1['start_time'] - crew_params['sign_on_time_depot_minutes']
        else:
            extra_time = crew_params['extra_time_non_depot_relief_minutes']
            if not crew_params['extra_time_includes_deadhead_to_depot']:
                extra_time += travel_times[start_loc1][depot_name]
            duty_start = piece1['start_time'] - extra_time
        if end_loc2 == depot_name:
            duty_end = piece2['end_time'] + crew_params['sign_off_time_depot_minutes']
        else:
            extra_time = crew_params['extra_time_non_depot_relief_minutes']
            if not crew_params['extra_time_includes_deadhead_to_depot']:
                extra_time += travel_times[end_loc2][depot_name]
            duty_end = piece2['end_time'] + extra_time
        duty_span = duty_end - duty_start
        if dt['duty_length_max'] is not None and duty_span > dt['duty_length_max']:
            return False
        total_work_time = work_duration1 + work_duration2
        if dt['work_time_max'] is not None and total_work_time > dt['work_time_max']:
            return False
        if dt['start_time_min'] is not None and duty_start < dt['start_time_min']:
            return False
        if dt['start_time_max'] is not None and duty_start > dt['start_time_max']:
            return False
        if dt['end_time_max'] is not None and duty_end > dt['end_time_max']:
            return False
        return True

    crew_schedule = []
    duty_type_name_to_data = {name: data for name, data in duty_types.items()}
    two_piece_duty_types = [dt for dt in duty_types.values() if dt['num_pieces'] == 2]
    min_break = min(dt['break_length_min'] for dt in two_piece_duty_types) if two_piece_duty_types else 0
    one_piece_duty_types = [dt for dt in duty_types.values() if dt['num_pieces'] == 1]

    for depot_name, pieces in all_pieces_by_depot.items():
        if not pieces:
            continue
        unassigned_pieces = sorted(pieces, key=lambda x: x['end_time'])
        assigned = [False] * len(unassigned_pieces)
        duties_depot = []
        for i in range(len(unassigned_pieces)):
            if assigned[i]:
                continue
            piece_i = unassigned_pieces[i]
            for dt_name, dt in duty_type_name_to_data.items():
                if dt['num_pieces'] == 1:
                    if satisfies_one_piece(piece_i, dt, depot_name, crew_parameters, travel_times):
                        duty = {
                            'depot': depot_name,
                            'type': dt_name,
                            'trips': piece_i['trip_ids']
                        }
                        duties_depot.append(duty)
                        assigned[i] = True
                        break
            if assigned[i]:
                continue
            for j in range(i+1, len(unassigned_pieces)):
                if assigned[j]:
                    continue
                piece_j = unassigned_pieces[j]
                if piece_j['start_time'] < piece_i['end_time'] + min_break:
                    continue
                for dt_name, dt in duty_type_name_to_data.items():
                    if dt['num_pieces'] == 2:
                        if satisfies_two_piece(piece_i, piece_j, dt, depot_name, crew_parameters, travel_times):
                            duty = {
                                'depot': depot_name,
                                'type': dt_name,
                                'trips': piece_i['trip_ids'] + piece_j['trip_ids']
                            }
                            duties_depot.append(duty)
                            assigned[i] = True
                            assigned[j] = True
                            break
                if assigned[i]:
                    break
        for i in range(len(unassigned_pieces)):
            if not assigned[i]:
                piece_i = unassigned_pieces[i]
                for dt_name, dt in duty_type_name_to_data.items():
                    if dt['num_pieces'] == 1:
                        if satisfies_one_piece(piece_i, dt, depot_name, crew_parameters, travel_times):
                            duty = {
                                'depot': depot_name,
                                'type': dt_name,
                                'trips': piece_i['trip_ids']
                            }
                            duties_depot.append(duty)
                            assigned[i] = True
                            break
                if not assigned[i] and one_piece_duty_types:
                    dt = one_piece_duty_types[0]
                    for dt_name, dt_data in duty_type_name_to_data.items():
                        if dt_data == dt:
                            duty = {
                                'depot': depot_name,
                                'type': dt_name,
                                'trips': piece_i['trip_ids']
                            }
                            duties_depot.append(duty)
                            assigned[i] = True
                            break
        crew_schedule.extend(duties_depot)

    total_vehicles = len(vehicles)
    total_duties = len(crew_schedule)
    objective_value = total_vehicles + total_duties

    if logger:
        logger.log(objective_value)

    solution = {
        'objective_value': objective_value,
        'vehicle_schedule': vehicle_schedule,
        'crew_schedule': crew_schedule
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == '__main__':
    main()
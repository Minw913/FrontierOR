import json
import argparse
import heapq
from collections import defaultdict
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

    # Precompute task details and station sets
    task_dict = {task['task_id']: task for task in instance['tasks']}
    for task in instance['tasks']:
        task['route_stations'] = get_task_route_stations(task, instance)
    
    # Precompute duty details
    duties = {}
    for duty in instance['active_duties'] + instance['reserve_duties']:
        duties[duty['duty_id']] = duty
        duty['start_station'] = duty.get('current_station', duty.get('standby_station'))
        duty['start_time_minutes'] = instance['disruption']['rescheduling_time_minutes']
        duty['original_tasks_set'] = set(duty['tasks'])
        duty['original_task_sequence'] = duty['tasks']
        duty['original_task_pairs'] = set()
        if len(duty['tasks']) >= 2:
            for i in range(len(duty['tasks']) - 1):
                duty['original_task_pairs'].add((duty['tasks'][i], duty['tasks'][i+1]))
    
    # Initialize assignment and cancellation
    assigned_tasks = defaultdict(list)
    canceled_tasks = set(instance['disruption']['canceled_tasks'])
    all_tasks = [task['task_id'] for task in instance['tasks'] if task['task_id'] not in canceled_tasks]
    
    # Greedy task assignment
    for task_id in all_tasks:
        task = task_dict[task_id]
        best_duty = None
        best_cost = float('inf')
        
        for duty_id, duty in duties.items():
            if task_id in duty['original_tasks_set']:
                cost = 0
            else:
                cost = instance['parameters']['cost_task_reassigned']
            
            if task_id in assigned_tasks[duty_id]:
                continue
            
            if task['rolling_stock_type'] not in duty['rolling_stock_licenses']:
                continue
            
            if not all(station in duty['route_knowledge'] for station in task['route_stations']):
                continue
            
            if cost < best_cost:
                best_cost = cost
                best_duty = duty_id
        
        if best_duty:
            assigned_tasks[best_duty].append(task_id)
        else:
            canceled_tasks.add(task_id)
    
    # Build paths for each duty
    solution = {
        "released_problem_variant": "relief_point_completion_v1",
        "objective_value": 0,
        "canceled_tasks": list(canceled_tasks),
        "duty_assignments": {}
    }
    
    for duty_id, task_ids in assigned_tasks.items():
        duty = duties[duty_id]
        path_events, cost, uses_taxi, release_at_relief_point, return_station, return_time = build_duty_path(
            duty, task_ids, task_dict, instance, canceled_tasks
        )
        covered_tasks = [event[1] for event in path_events if event[0] == 'operate']
        
        new_tasks_set = set(task_ids)
        reassigned_count = len(new_tasks_set - duty['original_tasks_set'])
        new_transfers = count_new_transfers(duty, covered_tasks)
        duty_cost = 0
        if covered_tasks != duty['tasks'] or uses_taxi or release_at_relief_point:
            duty_cost = instance['parameters']['cost_duty_changed']
            duty_cost += instance['parameters']['cost_task_reassigned'] * reassigned_count
            duty_cost += instance['parameters']['cost_new_transfer'] * new_transfers
            if uses_taxi:
                duty_cost += instance['parameters']['cost_taxi']
        
        solution['duty_assignments'][duty_id] = {
            "completion_index": 0,
            "cost": duty_cost,
            "covered_tasks": covered_tasks,
            "uses_taxi": uses_taxi,
            "release_at_relief_point": release_at_relief_point,
            "return_station": return_station,
            "return_time_minutes": return_time
        }
        solution['objective_value'] += duty_cost
    
    # Add cancellation costs
    for task_id in canceled_tasks:
        task = task_dict.get(task_id)
        if task:
            if task['task_type'] == 'A_to_B':
                solution['objective_value'] += instance['parameters']['cost_cancel_A_to_B']
            else:
                solution['objective_value'] += instance['parameters']['cost_cancel_A_to_A']
    
    # Log and save solution
    if logger:
        logger.log_solution(solution['objective_value'], solution)
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

def get_task_route_stations(task, instance):
    line = next(line for line in instance['train_lines'] if line['line_id'] == task['line_id'])
    stations = line['stations']
    dep_idx = stations.index(task['departure_station'])
    arr_idx = stations.index(task['arrival_station'])
    if dep_idx <= arr_idx:
        return stations[dep_idx:arr_idx+1]
    else:
        return stations[arr_idx:dep_idx+1][::-1]

def build_duty_path(duty, task_ids, task_dict, instance, canceled_tasks):
    tasks = sorted([task_dict[task_id] for task_id in task_ids], key=lambda t: t['departure_time_minutes'])
    path_events = []
    current_station = duty['start_station']
    current_time = duty['start_time_minutes']
    uses_taxi = False
    
    # Build graph for deadheading and taxi connections
    graph = defaultdict(list)
    for task in instance['tasks']:
        if task['task_id'] in canceled_tasks:
            continue
        graph[task['departure_station']].append((
            task['arrival_station'], 
            task['departure_time_minutes'], 
            task['arrival_time_minutes'], 
            'deadhead', 
            task['task_id']
        ))
    
    for taxi in instance['network']['taxi_connections']:
        if taxi['bidirectional']:
            graph[taxi['from_station']].append((
                taxi['to_station'],
                current_time,  # Can be used at any time
                current_time + taxi['travel_time_minutes'],
                'taxi',
                None
            ))
            graph[taxi['to_station']].append((
                taxi['from_station'],
                current_time,
                current_time + taxi['travel_time_minutes'],
                'taxi',
                None
            ))
        else:
            graph[taxi['from_station']].append((
                taxi['to_station'],
                current_time,
                current_time + taxi['travel_time_minutes'],
                'taxi',
                None
            ))
    
    # Process each assigned task
    for task in tasks:
        # Move to task departure station
        if current_station != task['departure_station']:
            success, events, new_time, new_station = find_path(
                current_station, current_time, 
                task['departure_station'], task['departure_time_minutes'],
                graph
            )
            if not success:
                return [], float('inf'), False, False, None, None
            path_events.extend(events)
            current_station = new_station
            current_time = new_time
            if any(event[0] == 'taxi' for event in events):
                uses_taxi = True
        
        # Operate the task
        if current_time > task['departure_time_minutes']:
            return [], float('inf'), False, False, None, None
        path_events.append(('operate', task['task_id'], task['departure_station'], task['departure_time_minutes'], task['arrival_station'], task['arrival_time_minutes']))
        current_station = task['arrival_station']
        current_time = task['arrival_time_minutes']
    
    # Return to crew base or relief point
    max_end_time = duty['end_time_minutes'] + instance['parameters']['max_overtime_minutes']
    success, events, return_time, return_station = find_return_path(
        current_station, current_time, 
        duty['crew_base'], 
        max_end_time,
        instance,
        graph
    )
    release_at_relief_point = not success
    if not success:
        # Find nearest relief point
        for station in instance['network']['stations']:
            if station['is_relief_point']:
                success, events, return_time, return_station = find_path(
                    current_station, current_time, 
                    station['id'], max_end_time,
                    graph
                )
                if success:
                    break
        if not success:
            return [], float('inf'), False, False, None, None
    
    path_events.extend(events)
    if any(event[0] == 'taxi' for event in events):
        uses_taxi = True
    
    return path_events, 0, uses_taxi, release_at_relief_point, return_station, return_time

def find_path(start_station, start_time, end_station, deadline, graph):
    pq = [(start_time, start_station)]
    dist = {station: float('inf') for station in graph}
    dist[start_station] = start_time
    prev = {}
    
    while pq:
        current_time, current_station = heapq.heappop(pq)
        if current_station == end_station:
            break
        if current_time > dist[current_station]:
            continue
        for neighbor, dep_time, arr_time, conn_type, task_id in graph[current_station]:
            if dep_time < current_time:
                continue
            new_time = arr_time
            if new_time < dist[neighbor]:
                dist[neighbor] = new_time
                prev[neighbor] = (current_station, conn_type, task_id, dep_time, arr_time)
                heapq.heappush(pq, (new_time, neighbor))
    
    if dist.get(end_station, float('inf')) > deadline:
        return False, [], None, None
    
    # Reconstruct path
    path = []
    current = end_station
    while current != start_station:
        station, conn_type, task_id, dep_time, arr_time = prev[current]
        path.append((conn_type, task_id, station, dep_time, current, arr_time))
        current = station
    path.reverse()
    return True, path, dist[end_station], end_station

def find_return_path(current_station, current_time, crew_base, max_end_time, instance, graph):
    # Try to reach crew base first
    success, events, return_time, return_station = find_path(
        current_station, current_time, crew_base, max_end_time, graph
    )
    if success:
        return True, events, return_time, return_station
    
    # If not possible, find any relief point
    for station in instance['network']['stations']:
        if station['is_relief_point']:
            success, events, return_time, return_station = find_path(
                current_station, current_time, station['id'], max_end_time, graph
            )
            if success:
                return False, events, return_time, return_station
    return False, [], None, None

def count_new_transfers(duty, covered_tasks):
    if len(covered_tasks) < 2:
        return 0
    new_pairs = set()
    for i in range(len(covered_tasks) - 1):
        new_pairs.add((covered_tasks[i], covered_tasks[i+1]))
    return len(new_pairs - duty['original_task_pairs'])

if __name__ == "__main__":
    main()
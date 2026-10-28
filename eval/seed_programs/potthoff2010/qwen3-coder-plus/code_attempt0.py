import argparse
import json
import math
from collections import defaultdict
import datetime


def parse_time(time_str):
    """Convert HH:MM string to minutes from midnight."""
    h, m = map(int, time_str.split(':'))
    return h * 60 + m


def time_to_str(minutes):
    """Convert minutes from midnight to HH:MM string."""
    h = minutes // 60
    m = minutes % 60
    return f"{h:02d}:{m:02d}"


def dijkstra(graph, start, end=None):
    """Dijkstra's algorithm to find shortest paths from start node."""
    dist = {node: float('inf') for node in graph}
    prev = {node: None for node in graph}
    dist[start] = 0
    unvisited = set(graph.keys())
    
    while unvisited:
        u = min(unvisited, key=lambda x: dist[x])
        if dist[u] == float('inf'):
            break
        unvisited.remove(u)
        
        if end is not None and u == end:
            break
            
        for v, weight in graph[u].items():
            alt = dist[u] + weight
            if alt < dist[v]:
                dist[v] = alt
                prev[v] = u
    
    return dist, prev


class CrewSchedulingSolver:
    def __init__(self, instance_data):
        self.instance = instance_data
        self.network = instance_data['network']
        self.tasks = {t['task_id']: t for t in instance_data['tasks']}
        self.active_duties = {d['duty_id']: d for d in instance_data['active_duties']}
        self.reserve_duties = {d['duty_id']: d for d in instance_data['reserve_duties']}
        self.parameters = instance_data['parameters']
        self.disruption = instance_data['disruption']
        self.all_duties = {**self.active_duties, **self.reserve_duties}
        
        # Build station index
        self.station_id_to_idx = {s['id']: i for i, s in enumerate(self.network['stations'])}
        self.stations_by_id = {s['id']: s for s in self.network['stations']}
        
        # Create graph for route finding
        self.build_graph()
        
        # Precompute task compatibility
        self.precompute_compatibility()
    
    def build_graph(self):
        """Build a graph representation of the network for pathfinding."""
        self.graph = defaultdict(dict)
        
        # Add route segments
        for seg in self.network['route_segments']:
            if seg['bidirectional']:
                self.graph[seg['from_station']][seg['to_station']] = seg['travel_time_minutes']
                self.graph[seg['to_station']][seg['from_station']] = seg['travel_time_minutes']
            else:
                self.graph[seg['from_station']][seg['to_station']] = seg['travel_time_minutes']
        
        # Add taxi connections
        for taxi in self.network['taxi_connections']:
            if taxi['bidirectional']:
                self.graph[taxi['from_station']][taxi['to_station']] = taxi['travel_time_minutes']
                self.graph[taxi['to_station']][taxi['from_station']] = taxi['travel_time_minutes']
            else:
                self.graph[taxi['from_station']][taxi['to_station']] = taxi['travel_time_minutes']
    
    def precompute_compatibility(self):
        """Precompute which tasks can be handled by which duties."""
        self.task_duty_compatibility = defaultdict(list)
        
        for duty_id, duty in self.all_duties.items():
            for task_id in self.tasks:
                task = self.tasks[task_id]
                
                # Check rolling stock license
                if task['rolling_stock_type'] not in duty['rolling_stock_licenses']:
                    continue
                
                # Check route knowledge
                valid_route = True
                for station_id in [task['departure_station'], task['arrival_station']]:
                    if station_id not in duty['route_knowledge']:
                        valid_route = False
                        break
                        
                if valid_route:
                    self.task_duty_compatibility[task_id].append(duty_id)
    
    def find_feasible_paths(self, duty, start_time, start_station, max_end_time):
        """Find all feasible task sequences for a duty starting from a given state."""
        # We'll use a simplified approach to generate candidate completions
        
        # Get tasks that can be assigned to this duty
        eligible_tasks = []
        for task_id in self.tasks:
            if duty['duty_id'] in self.task_duty_compatibility[task_id]:
                task = self.tasks[task_id]
                # Only consider tasks that haven't been canceled and start after rescheduling time
                if task_id not in self.disruption['canceled_tasks'] and \
                   task['departure_time_minutes'] >= self.disruption['rescheduling_time_minutes']:
                    eligible_tasks.append(task)
        
        # Sort tasks by departure time
        eligible_tasks.sort(key=lambda t: t['departure_time_minutes'])
        
        # Generate feasible completions using a greedy approach with backtracking
        completions = []
        
        def backtrack(current_path, current_time, current_station, used_tasks):
            # Add current path to completions
            completions.append({
                'path': current_path[:],
                'end_time': current_time,
                'end_station': current_station
            })
            
            # Try adding next tasks
            for task in eligible_tasks:
                if task['task_id'] in used_tasks:
                    continue
                
                # Check if we can reach the departure station in time
                if current_station != task['departure_station']:
                    # Find travel time
                    dist, _ = dijkstra(self.graph, current_station, task['departure_station'])
                    travel_time = dist.get(task['departure_station'], float('inf'))
                    
                    if travel_time == float('inf'):
                        continue  # No path exists
                    
                    arrival_time = current_time + travel_time
                else:
                    arrival_time = current_time
                
                # Check if we can make it to the task on time
                if arrival_time > task['departure_time_minutes']:
                    continue
                
                # Check if duty would exceed max end time
                task_end_time = task['arrival_time_minutes']
                if task_end_time > max_end_time:
                    continue
                
                # Add the task
                used_tasks.add(task['task_id'])
                current_path.append(task['task_id'])
                
                # Recurse
                backtrack(current_path, 
                         task_end_time, 
                         task['arrival_station'], 
                         used_tasks)
                
                # Backtrack
                current_path.pop()
                used_tasks.remove(task['task_id'])
        
        # Start the search
        backtrack([], start_time, start_station, set())
        
        # Add the option of doing nothing
        completions.append({
            'path': [],
            'end_time': start_time,
            'end_station': start_station
        })
        
        return completions
    
    def calculate_completion_cost(self, duty, completion_path):
        """Calculate the cost of a specific completion."""
        if not completion_path:
            return 0.0  # No change cost if no tasks added
        
        cost = self.parameters['cost_duty_changed']
        
        # Count reassigned tasks
        original_task_set = set(duty['tasks'])
        for task_id in completion_path:
            if task_id not in original_task_set:
                cost += self.parameters['cost_task_reassigned']
        
        # Count new transfers
        all_tasks_in_duty = duty['tasks'] + completion_path
        for i in range(len(all_tasks_in_duty) - 1):
            if all_tasks_in_duty[i] not in duty['tasks'] or all_tasks_in_duty[i+1] not in duty['tasks']:
                # At least one is new, so this is a new transfer
                cost += self.parameters['cost_new_transfer']
        
        return cost
    
    def solve(self):
        """Main solving method."""
        # Determine which tasks need to be covered
        remaining_tasks = set()
        for task_id in self.tasks:
            if task_id not in self.disruption['canceled_tasks'] and \
               self.tasks[task_id]['departure_time_minutes'] >= self.disruption['rescheduling_time_minutes']:
                remaining_tasks.add(task_id)
        
        # For each duty, find feasible completions
        duty_completions = {}
        for duty_id, duty in self.all_duties.items():
            # Determine start time and station
            if duty['duty_type'] == 'active':
                # Active duty - driver is currently at their current station
                start_station = duty['current_station']  # Assuming this field exists in the actual data
                # Since this field is not in schema, we use arrival of last pre-disruption task if exists
                if duty['pre_disruption_tasks']:
                    last_task_id = duty['pre_disruption_tasks'][-1]
                    last_task = self.tasks[last_task_id]
                    start_station = last_task['arrival_station']
                    start_time = last_task['arrival_time_minutes']
                else:
                    start_time = self.disruption['rescheduling_time_minutes']
                    start_station = duty['crew_base']  # Assume they start at base if no pre-tasks
            else:  # reserve duty
                start_station = duty['standby_station']
                start_time = self.disruption['rescheduling_time_minutes']
            
            max_end_time = duty['end_time_minutes'] + self.parameters['max_overtime_minutes']
            
            # Get feasible completions
            completions = self.find_feasible_paths(duty, start_time, start_station, max_end_time)
            duty_completions[duty_id] = completions
        
        # Greedy assignment approach
        # Initialize assignment
        task_assignment = {}  # Maps task_id to duty_id that covers it
        duty_assignments = {}  # Maps duty_id to selected completion index
        canceled_tasks = set(remaining_tasks)  # Initially all remaining tasks are candidates for cancellation
        
        # Simple greedy assignment: try to assign tasks to duties that can cover them with lowest marginal cost
        unassigned_tasks = list(remaining_tasks)
        
        while unassigned_tasks:
            best_assignment = None
            best_cost_increase = float('inf')
            
            for task_id in unassigned_tasks:
                for duty_id in self.task_duty_compatibility[task_id]:
                    duty = self.all_duties[duty_id]
                    
                    # Find the best completion for this duty that includes this task
                    best_completion_for_task = None
                    best_completion_cost = float('inf')
                    
                    for i, completion in enumerate(duty_completions[duty_id]):
                        if task_id in completion['path']:
                            # Calculate cost of this completion
                            cost = self.calculate_completion_cost(duty, completion['path'])
                            
                            if cost < best_completion_cost:
                                best_completion_cost = cost
                                best_completion_for_task = i
                    
                    if best_completion_for_task is not None and best_completion_cost < best_cost_increase:
                        best_assignment = (task_id, duty_id, best_completion_for_task, best_completion_cost)
                        best_cost_increase = best_completion_cost
            
            if best_assignment is None:
                # Could not assign any more tasks, break
                break
            
            # Assign the task
            task_id, duty_id, completion_idx, cost = best_assignment
            task_assignment[task_id] = duty_id
            duty_assignments[duty_id] = completion_idx
            canceled_tasks.discard(task_id)
            
            # Remove assigned task from unassigned list
            unassigned_tasks.remove(task_id)
        
        # For duties not in duty_assignments, assign the empty completion (index 0)
        for duty_id in self.all_duties:
            if duty_id not in duty_assignments:
                # Find the empty completion (should be index 0)
                for i, completion in enumerate(duty_completions[duty_id]):
                    if len(completion['path']) == 0:
                        duty_assignments[duty_id] = i
                        break
        
        # Calculate final objective value
        total_cost = 0.0
        
        # Add completion costs
        for duty_id, completion_idx in duty_assignments.items():
            completion = duty_completions[duty_id][completion_idx]
            duty = self.all_duties[duty_id]
            cost = self.calculate_completion_cost(duty, completion['path'])
            total_cost += cost
        
        # Add cancellation costs
        for task_id in canceled_tasks:
            task_type = self.tasks[task_id]['task_type']
            if task_type == 'A_to_B':
                total_cost += self.parameters['cost_cancel_A_to_B']
            else:
                total_cost += self.parameters['cost_cancel_A_to_A']
        
        # Construct solution
        solution = {
            "released_problem_variant": "relief_point_completion_v1",
            "objective_value": total_cost,
            "canceled_tasks": list(canceled_tasks),
            "duty_assignments": {}
        }
        
        for duty_id, completion_idx in duty_assignments.items():
            completion = duty_completions[duty_id][completion_idx]
            duty = self.all_duties[duty_id]
            
            # Calculate cost again for the output
            cost = self.calculate_completion_cost(duty, completion['path'])
            
            # Determine if uses taxi (simplified check)
            uses_taxi = False
            if completion['path']:
                # Check if there are gaps that require taxi
                prev_arrival_station = duty['crew_base'] if duty['duty_type'] == 'reserve' else duty.get('current_station', duty['crew_base'])
                if duty['pre_disruption_tasks']:
                    last_task = self.tasks[duty['pre_disruption_tasks'][-1]]
                    prev_arrival_station = last_task['arrival_station']
                
                for task_id in completion['path']:
                    task = self.tasks[task_id]
                    if prev_arrival_station != task['departure_station']:
                        # Check if there's a train connection
                        dist, _ = dijkstra(self.graph, prev_arrival_station, task['departure_station'])
                        if dist.get(task['departure_station']) == float('inf'):
                            # This shouldn't happen if our path was feasible, but just in case
                            pass
                        else:
                            # If travel time is significantly longer than direct taxi, might indicate taxi
                            # For now, assume no taxi unless specifically needed
                            pass
                    prev_arrival_station = task['arrival_station']
            
            # Check if needs release at relief point
            release_at_relief_point = False
            if completion['end_station'] != duty['crew_base']:
                # Check if return to base is possible within horizon
                dist, _ = dijkstra(self.graph, completion['end_station'], duty['crew_base'])
                if dist.get(duty['crew_base']) == float('inf'):
                    release_at_relief_point = True
                else:
                    return_time = completion['end_time'] + dist[duty['crew_base']]
                    if return_time > duty['end_time_minutes'] + self.parameters['max_overtime_minutes']:
                        release_at_relief_point = True
            
            solution["duty_assignments"][duty_id] = {
                "completion_index": completion_idx,
                "cost": cost,
                "covered_tasks": completion['path'],
                "uses_taxi": uses_taxi,
                "release_at_relief_point": release_at_relief_point,
                "return_station": completion['end_station'],
                "return_time_minutes": completion['end_time']
            }
        
        return solution


def main():
    parser = argparse.ArgumentParser(description='Solve crew scheduling problem.')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the instance JSON file')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to save the solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds')
    parser.add_argument('--log_path', type=str, default=None, help='Path to log intermediate solutions')
    
    args = parser.parse_args()
    
    # Import logger if log path provided
    logger = None
    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize")
    
    # Load instance data
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)
    
    # Create solver
    solver = CrewSchedulingSolver(instance_data)
    
    # Solve
    solution = solver.solve()
    
    # Log final solution if logger is available
    if logger:
        logger.log_solution(solution['objective_value'], solution)
    
    # Write solution
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == '__main__':
    main()
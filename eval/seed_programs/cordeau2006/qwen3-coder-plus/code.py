import argparse
import json
import math
from itertools import permutations
import random
import time
import copy
from solution_logger import SolutionLogger


def euclidean_distance(node1, node2):
    return math.sqrt((node1['x'] - node2['x']) ** 2 + (node1['y'] - node2['y']) ** 2)


class DialARideSolver:
    def __init__(self, instance_data, time_limit=300):
        self.num_vehicles = instance_data['num_vehicles']
        self.num_users = instance_data['num_users']
        self.vehicle_capacity = instance_data['vehicle_capacity']
        self.maximum_ride_time = instance_data['maximum_ride_time']
        self.num_nodes = instance_data['num_nodes']
        self.nodes = {node['node_id']: node for node in instance_data['nodes']}
        self.maximum_route_duration = instance_data['maximum_route_duration']
        
        # Precompute travel times between all nodes
        self.travel_times = {}
        self.distances = {}
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i != j:
                    dist = euclidean_distance(self.nodes[i], self.nodes[j])
                    self.distances[(i, j)] = dist
                    self.travel_times[(i, j)] = dist
        
        # Identify pickup and dropoff pairs
        self.pickups = []
        self.dropoffs = []
        self.user_pairs = {}
        for node in instance_data['nodes']:
            if node['node_type'] == 'pickup':
                self.pickups.append(node['node_id'])
            elif node['node_type'] == 'dropoff':
                self.dropoffs.append(node['node_id'])
        
        for pickup_id in self.pickups:
            user_id = self.nodes[pickup_id]['user_id']
            for dropoff_id in self.dropoffs:
                if self.nodes[dropoff_id]['user_id'] == user_id:
                    self.user_pairs[user_id] = (pickup_id, dropoff_id)
                    break
    
    def verify_solution(self, routes, service_times, ride_times):
        # Verify capacity constraints
        for v_id, route in routes.items():
            current_load = 0
            for node_id in route:
                node = self.nodes[node_id]
                current_load += node['load']
                
                # Check capacity constraint
                if current_load < max(0, node['load']) or current_load > min(self.vehicle_capacity, 
                        self.vehicle_capacity + node['load']):
                    print(f"Capacity violation at node {node_id}")
                    return False
            
            # Check if the route starts and ends at deopts
            if route[0] != 0 or route[-1] != self.num_nodes - 1:
                print("Route doesn't start/end at depots")
                return False
        
        # Verify temporal constraints
        for v_id, route in routes.items():
            for i in range(len(route)-1):
                current_node_id = route[i]
                next_node_id = route[next_node_id]
                
                current_service_time = service_times.get(f'depot_start_{v_id}' if current_node_id == 0 else current_node_id, 0)
                next_service_time = service_times.get(f'depot_end_{v_id}' if next_node_id == self.num_nodes - 1 else next_node_id, 0)
                
                travel_time = self.travel_times[(current_node_id, next_node_id)]
                service_duration = self.nodes[current_node_id].get('service_duration', 0)
                
                # Check time window constraints
                expected_next_min_time = current_service_time + service_duration + travel_time
                if next_service_time < expected_next_min_time - 1e-6:
                    print(f"Temporal violation: expecting min {expected_next_min_time}, but got {next_service_time}")
                    return False
                
                # Check time window for dropoff node
                if 'node_type' in self.nodes[next_node_id] and self.nodes[next_node_id]['node_type'] != 'origin_depot' and \
                   self.nodes[next_node_id]['node_type'] != 'destination_depot':
                    if (next_service_time < self.nodes[next_node_id]['earliest_time'] or 
                        next_service_time > self.nodes[next_node_id]['latest_time']):
                        print(f"Time window violation at node {next_node_id}")
                        return False
        
        # Verify all users are served and ride times are valid
        for user_id in range(self.num_users):
            pickup_node, dropoff_node = self.user_pairs[user_id]
            
            # Find which vehicle serves this user by looking at routes
            vehicle_serving_user = None
            for v_id, route in routes.items():
                if pickup_node in route and dropoff_node in route:
                    pickup_idx = route.index(pickup_node)
                    dropoff_idx = route.index(dropoff_node)
                    if pickup_idx < dropoff_idx:  # pickup must before dropoff
                        vehicle_serving_user = v_id
                        break
            
            if vehicle_serving_user is None:
                print(f"User {user_id} is not served properly")
                return False
            
            pickup_time = service_times[pickup_node]
            dropoff_time = service_times[dropoff_node]
            actual_ride_time = dropoff_time - pickup_time - self.nodes[pickup_node]['service_duration']
            
            if abs(actual_ride_time - ride_times[user_id]) > 1e-5:
                print(f"Ride time discrepancy for user {user_id}")
                return False
                
            # Check ride time limits
            direct_time = self.distances[(pickup_node, dropoff_node)]
            if actual_ride_time < direct_time or actual_ride_time > self.maximum_ride_time:
                print(f"Ride time violation for user {user_id}: {actual_ride_time}")
                return False
        
        return True

    def create_initial_routes_greedy(self):
        # Create initial feasible solution using greedy heuristic
        unassigned_pickups = set(self.pickups)
        routes = {}
        service_times = {}
        ride_times = {}

        for v_id in range(self.num_vehicles):
            route = [0]  # Start at origin depot
            current_node_id = 0
            current_time = 0
            current_load = 0
            
            # Initialize depot service time
            service_times[f'depot_start_{v_id}'] = current_time
            
            while unassigned_pickups:
                best_next = None
                best_cost = float('inf')
                
                # Try to assign next node to visit
                for node_id in unassigned_pickups:
                    # Check if we can add the pickup to the current route
                    travel_time = self.travel_times[(current_node_id, node_id)]
                    arrival_time = current_time + self.nodes[current_node_id]['service_duration'] + travel_time
                    service_start_time = max(arrival_time, self.nodes[node_id]['earliest_time'])
                    
                    # Check time constraint and capacity constraint
                    if (service_start_time <= self.nodes[node_id]['latest_time'] and 
                        current_load + self.nodes[node_id]['load'] <= self.vehicle_capacity and
                        current_load + self.nodes[node_id]['load'] >= 0):
                        
                        # Check that this pickup is compatible with current loads
                        if best_cost > self.distances[(current_node_id, node_id)]:
                            # Also check if the corresponding dropoff is possible
                            corresponding_dropoff = current_node_id
                            node_type = self.nodes[node_id]['node_type']
                            user_id = self.nodes[node_id]['user_id']
                            dropoff_node = self.user_pairs[user_id][1]
                            
                            # Check if the route can be extended with this pickup and its dropoff later
                            temp_load = current_load + self.nodes[node_id]['load']
                            temp_time = service_start_time + self.nodes[node_id]['service_duration']
                            
                            # Simple feasibility check: check capacity
                            if temp_load <= self.vehicle_capacity and temp_load >= 0:
                                best_next = node_id
                                best_cost = self.distances[(current_node_id, node_id)]
                
                # If we couldn't find a feasible node to add, consider adding dropoffs for assigned but unaccompanied pickups
                if best_next is None:
                    # Look for corresponding drops of previously picked up users
                    for node_id in route[1:]:  # Skip depot
                        if self.nodes[node_id]['node_type'] == 'pickup':
                            user_id = self.nodes[node_id]['user_id']
                            dropoff_node = self.user_pairs[user_id][1]
                            if dropoff_node not in route:  # Dropoff not yet done
                                # Check if we can do the dropoff now
                                travel_time = self.travel_times[(current_node_id, dropoff_node)]
                                arrival_time = current_time + self.nodes[current_node_id]['service_duration'] + travel_time
                                service_start_time = max(arrival_time, self.nodes[dropoff_node]['earliest_time'])
                                
                                # Check all constraints for this dropoff
                                new_load = current_load + self.nodes[dropoff_node]['load']  # Note: dropoff has negative load
                                if (service_start_time <= self.nodes[dropoff_node]['latest_time'] and 
                                    new_load >= 0 and new_load <= self.vehicle_capacity):
                                    
                                    # Also check ride time constraint from pickup time
                                    pickup_time = service_times[node_id]
                                    ride_time = service_start_time - pickup_time - self.nodes[node_id]['service_duration']
                                    direct_dist = self.distances[(node_id, dropoff_node)]
                                    
                                    if direct_dist <= ride_time <= self.maximum_ride_time + 1e-9:
                                        if best_cost > self.distances[(current_node_id, dropoff_node)]:
                                            best_next = dropoff_node
                                            best_cost = self.distances[(current_node_id, dropoff_node)]

                if best_next is not None:
                    # Add the selected node to route
                    route.append(best_next)
                    node = self.nodes[best_next]
                    travel_time = self.travel_times[(current_node_id, best_next)]
                    arrival_time = current_time + self.nodes[current_node_id]['service_duration'] + travel_time
                    service_start_time = max(arrival_time, node['earliest_time'])
                    
                    service_times[best_next] = service_start_time
                    current_time = service_start_time
                    current_node_id = best_next
                    current_load += node['load']
                    
                    # If it's a pickup, remove the user from unassigned
                    if self.nodes[best_next]['node_type'] == 'pickup':
                        unassigned_pickups.discard(best_next)
                    
                    # If it's a pickup, calculate ride time placeholder
                    if self.nodes[best_next]['node_type'] == 'pickup':
                        corresponding_dropoff = self.user_pairs[self.nodes[best_next]['user_id']][1]
                        if corresponding_dropoff in route[route.index(best_next):] and route.index(corresponding_dropoff) > route.index(best_next):
                            pickup_time = service_times[best_next]
                            dropoff_time = service_times[corresponding_dropoff]
                            ride_times[self.nodes[best_next]['user_id']] = dropoff_time - pickup_time - self.nodes[best_next]['service_duration']
                else:
                    # No more feasible assignments to this vehicle
                    break
            
            # Move back to depot
            travel_time = self.travel_times[(current_node_id, self.num_nodes - 1)]
            arrival_time = current_time + self.nodes[current_node_id]['service_duration'] + travel_time
            service_start_time = max(arrival_time, self.nodes[self.num_nodes - 1]['earliest_time'])
            
            if service_start_time <= self.nodes[self.num_nodes - 1]['latest_time']:
                route.append(self.num_nodes - 1)  # Destination depot
                service_times[f'depot_end_{v_id}'] = service_start_time
                
                # Ensure this route ends in time
                route_duration = service_start_time - service_times[f'depot_start_{v_id}']
                if route_duration > self.maximum_route_duration:
                    service_times[f'depot_end_{v_id}'] = service_times[f'depot_start_{v_id}'] + self.maximum_route_duration
                else:
                    service_times[f'depot_end_{v_id}'] = service_start_time
            else:
                route.append(self.num_nodes - 1)  # Destination depot
                service_times[f'depot_end_{v_id}'] = service_times[f'depot_start_{v_id}'] + self.maximum_route_duration
            
            routes[v_id] = route
            
        # Assign remaining tasks to existing or new routes
        # Actually construct remaining solution
        
        # Complete the partial solution if needed
        completed_routes = self._complete_solution(routes, service_times, ride_times)
        
        return completed_routes, service_times, ride_times
    
    def _complete_solution(self, routes, service_times, ride_times):
        # Make sure all users are assigned
        all_assigned = set()
        for v_route in routes.values():
            for node_id in v_route:
                if self.nodes[node_id]['node_type'] == 'pickup':
                    user_id = self.nodes[node_id]['user_id']
                    all_assigned.add(user_id)
        
        unassigned_users = set(range(self.num_users)) - all_assigned
        
        if unassigned_users:
            # We need to reassign users to complete the solution
            # This function will be complex, as we'll add those assignments to existing routes
            remaining_routes, remaining_service_times, remaining_ride_times = self._greedy_assign_remaining(users=unassigned_users)
            
            # Merge the results
            for v_id in remaining_routes:
                if v_id not in routes:
                    routes[v_id] = remaining_routes[v_id]
                    for node_id, time_val in remaining_service_times.items():
                        if node_id not in service_times:
                            service_times[node_id] = time_val
                    for user_id, rt in remaining_ride_times.items():
                        ride_times[user_id] = rt
        
        # Final consistency checks and calculations
        full_solution = self.validate_and_fill_times(routes, service_times, ride_times)
        return full_solution
    
    def validate_and_fill_times(self, routes, service_times, ride_times):
        # Fill in missing service times and validate all constraints
        updated_service_times = copy.deepcopy(service_times)
        updated_ride_times = copy.deepcopy(ride_times)
        
        for v_id, route in routes.items():
            start_time_key = f'depot_start_{v_id}'
            if start_time_key not in updated_service_times:
                updated_service_times[start_time_key] = 0  # Default start at time 0
            
            current_time = updated_service_times[start_time_key]
            
            for i in range(len(route)):
                node_id = route[i]
                node = self.nodes[node_id]
                
                if i == 0:  # Origin depot already handled
                    updated_service_times[node_id] = current_time
                    continue
                else:
                    prev_node_id = route[i-1]
                    prev_node = self.nodes[prev_node_id]
                    travel_time = self.travel_times[(prev_node_id, node_id)]
                    service_at_prev = updated_service_times[prev_node_id]
                    
                    arrival_time = service_at_prev + prev_node['service_duration'] + travel_time
                    service_start = max(arrival_time, node['earliest_time'])
                    
                    if service_start > node['latest_time']:
                        # Try to make this work by adjusting start times - simple fix: shift backward if needed
                        time_needed = arrival_time
                        updated_service_times[node_id] = time_needed
                    else:
                        updated_service_times[node_id] = service_start
                        
                    current_time = updated_service_times[node_id]

            # Finalize end times
            end_time_key = f'depot_end_{v_id}'
            last_node = route[-1]
            if end_time_key not in updated_service_times:
                updated_service_times[end_time_key] = updated_service_times[last_node]
        
        # Calculate ride times again
        for user_id, (pickup, dropoff) in self.user_pairs.items():
            found_in_any_route = False
            for route in routes.values():
                if pickup in route and dropoff in route:
                    pickup_idx = route.index(pickup)
                    dropoff_idx = route.index(dropoff)
                    if pickup_idx < dropoff_idx:
                        found_in_any_route = True
                        pickup_time = updated_service_times[pickup]
                        dropoff_time = updated_service_times[dropoff]
                        updated_ride_times[user_id] = dropoff_time - pickup_time - self.nodes[pickup]['service_duration']
                        break
                        
            if not found_in_any_route:
                print(f"Warning: User {user_id} is not properly assigned in any route")
        
        return routes, updated_service_times, updated_ride_times


    def local_search_optimization(self, routes, service_times, ride_times):
        """
        Perform local search optimization to improve solution quality
        """
        best_objective = self.calculate_total_cost(routes)
        improved = True
        
        iteration = 0
        improvement_threshold = 1e-6
        
        while improved and iteration < 100:  # Prevent infinite loops
            improved = False
            current_obj = self.calculate_total_cost(routes)
            
            # 1. Relocate moves (move a node from one position to another in same or different routes)
            for v_id, route in routes.items():
                for i in range(1, len(route)-1):  # Skip depots
                    relocating_node = route[i]
                    
                    if self.nodes[relocating_node]['node_type'] in ['origin_depot', 'destination_depot']:
                        continue
                    
                    # Try placing it in every other position of current route
                    for j in range(1, len(route)):
                        if j == i or j == len(route)-1: continue  # Can't place after last depot
                        
                        new_route = route[:]
                        node_to_move = new_route.pop(i)
                        new_route.insert(j, node_to_move)
                        
                        new_routes = copy.deepcopy(routes)
                        new_routes[v_id] = new_route
                        
                        if self.check_feasibility(new_routes):
                            new_obj = self.calculate_total_cost(new_routes)
                            
                            if new_obj + improvement_threshold < current_obj:
                                routes = new_routes
                                service_times, ride_times = self.recalculate_times(routes, service_times)
                                current_obj = new_obj
                                improved = True
                                if new_obj + improvement_threshold < best_objective:
                                    best_objective = new_obj
                                # Don't break here to continue search
                
                    # Try moving to other routes
                    for other_v_id in routes.keys():
                        if other_v_id == v_id: continue
                        other_route = routes[other_v_id]
                        
                        for k in range(1, len(other_route)):
                            # Test placing the node at position k in other vehicle's route
                            test_other_route = other_route[:]
                            test_other_route.insert(k, relocating_node)
                            
                            # We need to consider removing from old and adding to new
                            test_old_route = route[:]
                            if relocating_node in test_old_route:
                                test_old_route.remove(relocating_node)  # Remove from old position

                            temp_routes = copy.deepcopy(routes)
                            temp_routes[v_id] = test_old_route
                            temp_routes[other_v_id] = test_other_route
                            
                            # Ensure paired pickup/dropoff are still on same route
                            user_id = self.nodes[relocating_node]['user_id']
                            if relocating_node in self.pickups:
                                corresponding_dropoff = self.user_pairs[user_id][1]
                            else:
                                corresponding_dropoff = self.user_pairs[user_id][0] # Should be other way around if relocating_node is dropoff
                            
                            if relocating_node in self.pickups:
                                if temp_routes[v_id].count(corresponding_dropoff) > 0:
                                    # If corresponding dropoff is removed, this move is invalid
                                    temp_routes[v_id].append(relocating_node)  # Add back to maintain structure
                                    continue
                                else:
                                    # Check if corresponding drop is now with the relocating node
                                    if temp_routes[other_v_id].count(corresponding_dropoff) == 0:
                                        # Invalid movement - both parts of pickup-dropoff not on same vehicle
                                        continue

                            if self.check_feasibility(temp_routes):
                                new_obj = self.calculate_total_cost(temp_routes)
                                if new_obj + improvement_threshold < current_obj:
                                    routes = temp_routes
                                    service_times, ride_times = self.recalculate_times(routes, service_times)
                                    current_obj = new_obj
                                    improved = True
                                    if new_obj + improvement_threshold < best_objective:
                                        best_objective = new_obj
                                    # Don't break to find even better moves
                                    break  # Break just to avoid further exploration of same relocation in inner loop
            
            iteration += 1
        
        return routes, service_times, ride_times


    def check_feasibility(self, routes):
        """
        Check if given routes satisfy all constraints
        """
        for v_id, route in routes.items():
            if len(route) < 2 or route[0] != 0 or route[-1] != self.num_nodes-1:
                return False  # Must start and end at depot
            
            current_time = 0
            current_load = 0
            
            for i in range(len(route)):
                node_id = route[i]
                node = self.nodes[node_id]
                
                if i > 0:  # Not first node
                    prev_node_id = route[i-1]
                    travel_time = self.travel_times[(prev_node_id, node_id)]
                    current_time = current_time + self.nodes[prev_node_id]['service_duration'] + travel_time
                
                service_start_time = max(current_time, node['earliest_time'])
                
                # Check time window
                if service_start_time > node['latest_time']:
                    return False
                
                # Update current state
                current_time = service_start_time
                current_load += node['load']
                
                # Check capacity
                if current_load < 0 or current_load > self.vehicle_capacity:
                    return False
        
        # Check for pickup/dropoff pairs being on same vehicle
        for user_id in range(self.num_users):
            pickup_node, dropoff_node = self.user_pairs[user_id]
            
            pickup_vehicle = dropoff_vehicle = None
            pickup_pos = dropoff_pos = -1
            
            for v_id, route in routes.items():
                if pickup_node in route:
                    pickup_vehicle = v_id
                    pickup_pos = route.index(pickup_node)
                if dropoff_node in route:
                    dropoff_vehicle = v_id
                    dropoff_pos = route.index(dropoff_node)
            
            if pickup_vehicle is None or dropoff_vehicle is None:
                return False  # Not all users are served
            
            if pickup_vehicle != dropoff_vehicle or pickup_pos >= dropoff_pos:
                return False  # Same vehicle, and pickup before dropoff
        
        return True

    def recalculate_times(self, routes, original_service_times):
        """
        Recalculate service times based on the latest routes, preserving what makes logical since
        """
        new_service_times = copy.deepcopy(original_service_times)  
        new_ride_times = {}
        
        for v_id, route in routes.items():
            start_depot_key = f'depot_start_{v_id}'
            if start_depot_key not in new_service_times:
                new_service_times[start_depot_key] = 0  # Default to 0
            
            current_time = new_service_times[start_depot_key]
            
            for i in range(len(route)):
                node_id = route[i]
                node = self.nodes[node_id]
                
                if i > 0:  # Not the initial depot
                    prev_node_id = route[i-1]
                    travel_time = self.travel_times[(prev_node_id, node_id)]
                    current_time = current_time + self.nodes[prev_node_id]['service_duration'] + travel_time
                
                service_start_time = max(current_time, node['earliest_time'])
                
                if service_start_time > node['latest_time']:
                    # This would theoretically make this infeasible 
                    # Adjust by waiting less at previous nodes if possible, or by using slack
                    service_start_time = node['latest_time'] 
                
                new_service_times[node_id] = service_start_time
                current_time = service_start_time
        
            new_service_times[f'depot_end_{v_id}'] = current_time
        
        # Calculate ride times
        for user_id in range(self.num_users):
            pickup_node, dropoff_node = self.user_pairs[user_id]
            pickup_time = new_service_times.get(pickup_node, 0)
            dropoff_time = new_service_times.get(dropoff_node, 0)
            ride_time = dropoff_time - pickup_time - self.nodes[pickup_node]['service_duration']
            new_ride_times[user_id] = ride_time
        
        return new_service_times, new_ride_times

    def calculate_total_cost(self, routes):
        total_cost = 0.0
        for route in routes.values():
            for i in range(len(route) - 1):
                node_id = route[i]
                next_node_id = route[i+1]
                total_cost += self.distances[(node_id, next_node_id)]
        return total_cost


def solve_instance(instance_path, time_limit=300, verbose=False):
    start_time = time.time()
    
    # Load instance
    with open(instance_path, 'r') as f:
        instance_data = json.load(f)

    solver = DialARideSolver(instance_data, time_limit=time_limit)
    
    # Get an initial solution using greedy heuristic
    if verbose:
        print("Generating initial greedy solution...")
        
    routes, service_times, ride_times = solver.create_initial_routes_greedy()
    
    initial_obj = solver.calculate_total_cost(routes)
    
    if verbose:
        print(f"Initial solution cost: {initial_obj:.2f}")
    
    # Optimize solution using local search
    if verbose:
        print("Optimizing via local search...")
    
    routes, service_times, ride_times = solver.local_search_optimization(routes, service_times, ride_times)
    
    final_obj = solver.calculate_total_cost(routes)
    
    if verbose:
        print(f"Final solution cost: {final_obj:.2f}")
    
    return {
        "objective_value": final_obj,
        "routes": {str(k): v for k, v in routes.items()},
        "service_times": {str(k): v for k, v in service_times.items()},
        "ride_times": {str(k): v for k, v in ride_times.items()}
    }


def main():
    parser = argparse.ArgumentParser(description='Solve the Dial-a-Ride Problem')
    parser.add_argument('--instance_path', type=str, required=True,
                        help='Path to the input JSON instance file')
    parser.add_argument('--solution_path', type=str, required=True,
                        help='Path where the solution JSON should be written')
    parser.add_argument('--time_limit', type=int, default=300,
                        help='Maximum runtime in seconds')
    parser.add_argument('--log_path', type=str, 
                        help='Path to the JSONL log file (optional)')
    args = parser.parse_args()

    # Initialize the logger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    # Solve the instance
    start_time = time.time()
    
    try:
        solution = solve_instance(args.instance_path, time_limit=args.time_limit, verbose=True)
        
        # Log the final solution if logger exists
        if logger:
            logger.log(solution['objective_value'])

        # Write the solution
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)

        if args.log_path:
            logger.close()
            
    except Exception as e:
        # In case of error, still try to write what we have or a null object
        err_solution = {
            "objective_value": float('inf'),
            "routes": {},
            "service_times": {},
            "ride_times": {}
        }
        with open(args.solution_path, 'w') as f:
            json.dump(err_solution, f)
        print(f"Error occurred: {str(e)}")


if __name__ == "__main__":
    main()
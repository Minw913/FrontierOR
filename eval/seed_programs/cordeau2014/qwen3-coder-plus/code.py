import argparse
import json
import math
import time
import random
from collections import defaultdict
import heapq
import itertools


def compute_travel_time(departure_time, distance, max_speeds_by_period):
    """
    Compute the travel time for a given arc considering the time-dependent travel speeds.
    
    Args:
        departure_time: The time when the vehicle starts traversing the arc.
        distance: Total travel distance of the arc.
        max_speeds_by_period: List of speeds for each period in effect for the arc.
        
    Returns:
        The total time needed to cross the arc.
    """
    # Simplified model with hardcoded assumption about periods
    speeds = max_speeds_by_period  # List[H] with each element = actual speed in that period
    time_periods = len(speeds)
    # Assume periods start at time 0 and extend until time_horizon_T is known
    # The period_boundaries parameter is passed separately, so we ignore time_horizons here initially
    
    period_start_times = [None] * (time_periods + 1)  # Placeholders to handle actual values via param

    # This is incorrect approach. Let me create a proper implementation based on the formula provided.
    # When a vehicle starts during period h, it travels distance/speed_in_that_period until it
    # either crosses the destination or enters a new period.
    return compute_travel_time_correctly(departure_time, distance, max_speeds_by_period)


def compute_travel_time_correctly(departure_time, distance, speeds_per_period, period_boundaries):
    """
    Actually compute the travel time correctly using time-dependent speed changes.
    
    Args:
        departure_time: Time vehicle starts traversing arc
        distance: Length of arc
        speeds_per_period: List of speeds per period for this arc
        period_boundaries: Array with start/end times of periods
    
    Returns:
        Time to traverse arc
    """
    period_idx = 0
    total_periods = len(speeds_per_period)
    
    # Find current period at departure
    for i in range(len(period_boundaries) - 1):
        if period_boundaries[i] <= departure_time < period_boundaries[i+1]:
            period_idx = i
            break
    
    current_time = departure_time
    remaining_distance = distance
    
    while remaining_distance > 0:
        # Current period details
        start_time = period_boundaries[period_idx]
        end_time = period_boundaries[period_idx + 1]
        current_speed = speeds_per_period[period_idx]
        
        # If speed is 0 or negative, we can't continue
        if current_speed <= 0:
            return float('inf')
        
        # Time needed to cover remaining distance at current speed
        time_needed_for_remaining = remaining_distance / current_speed
        
        # Tentative arrival time if we continue in current speed
        tentative_end_time = current_time + time_needed_for_remaining
        
        if tentative_end_time <= end_time:
            # Finish in current period
            return tentative_end_time - departure_time
        else:
            # Finish current period's portion and move to next
            time_in_current_period = end_time - current_time
            distance_covered = current_speed * time_in_current_period
            
            remaining_distance -= distance_covered
            current_time = end_time
            
            if period_idx < total_periods - 1:
                period_idx += 1
            else:
                # Exceeded time horizon
                return float('inf')
                
    return current_time - departure_time


def compute_total_travel_time(tour, distances, speeds_cube, period_boundaries):
    """
    Calculate the total travel time for a given tour, considering time-dependent speeds.
    
    Args:
        tour: List of vertices in the tour (starting and ending at depot).
        distances: Distance matrix.
        speeds_cube: [i][j][h] = speed for (i,j) in period h
        period_boundaries: Period boundaries array
        
    Returns:
        Total travel time for the tour, or infinity if it's invalid.
    """
    n = len(tour)
    if n < 2:
        return 0.0

    current_time = 0.0

    for i in range(n - 1):
        from_node = tour[i]
        to_node = tour[i + 1]
        
        # Get the speed over each period for this arc
        speeds_for_arc = speeds_cube[from_node][to_node]
        
        travel_time = compute_travel_time_correctly(current_time, 
                                                   distances[from_node][to_node],
                                                   speeds_for_arc, 
                                                   period_boundaries)
        
        if travel_time == float('inf'):
            return float('inf')
            
        current_time += travel_time

    return current_time  # Total duration from start at depot to end at depot


class TDSPTSolver:
    def __init__(self, instance_data):
        self.instance = instance_data
        self.n_customers = instance_data['parameters']['n_customers']
        self.total_vertices = instance_data['parameters']['total_vertices']
        self.H = instance_data['parameters']['H']
        self.b_h = instance_data['parameters']['b_h']
        self.time_horizon_T = instance_data['parameters']['time_horizon_T']
        self.period_boundaries = instance_data['parameters']['period_boundaries']
        
        self.vertices = instance_data['vertices']
        self.depot_idx = instance_data['depot']['index']  # Should be 0
        
        self.distances = instance_data['arcs']['distances']
        self.max_speeds = instance_data['arcs']['max_speeds_u_ij']
        self.speeds_cube = instance_data['arcs']['speeds_v_ijh']
        
        self.all_nodes = set(range(self.total_vertices))
        
        # Validate that depot is at index 0
        assert self.depot_idx == 0, f"Expected depot at index 0, but got {self.depot_idx}"

    def construct_greedy_solution(self):
        """
        Construct an initial solution using nearest neighbor heuristic adapted for early arrivals.
        """
        unvisited = set(range(1, self.total_vertices))  # All except depot
        current_node = 0  # Start at depot
        tour = [current_node]
        current_time = 0.0

        while unvisited:
            next_node = min(unvisited, 
                           key=lambda node: compute_travel_time_correctly(
                               current_time,
                               self.distances[current_node][node],
                               self.speeds_cube[current_node][node],
                               self.period_boundaries))
            
            travel_time = compute_travel_time_correctly(current_time,
                                                      self.distances[current_node][next_node],
                                                      self.speeds_cube[current_node][next_node],
                                                      self.period_boundaries)
            current_time += travel_time
            current_node = next_node
            tour.append(current_node)
            unvisited.remove(current_node)

        # Return to depot
        travel_time = compute_travel_time_correctly(current_time,
                                                  self.distances[current_node][0],
                                                  self.speeds_cube[current_node][0],
                                                  self.period_boundaries)
        current_time += travel_time
        tour.append(0)

        total_time = compute_total_travel_time(tour, self.distances, self.speeds_cube, 
                                             self.period_boundaries)
        return tour, total_time

    def two_opt_swap(self, tour, i, k):
        """
        Perform 2-opt swap on tour from position i to k.
        """
        new_tour = tour[:i] + tour[i:k+1][::-1] + tour[k+1:]
        return new_tour

    def local_search_2opt(self, initial_tour, time_limit):
        """
        Perform 2-opt local search until no improvement or timeout.
        """
        start_time = time.time()
        best_tour = initial_tour[:]
        best_obj = compute_total_travel_time(best_tour, self.distances, self.speeds_cube, 
                                           self.period_boundaries)

        improved = True
        while improved and (time.time() - start_time < time_limit):
            improved = False
            tour_len = len(best_tour) - 1  # Exclude return-to-depot
            
            for i in range(1, tour_len - 1):  # First position after depot
                if time.time() - start_time >= time_limit:
                    break
                for k in range(i + 1, tour_len):  # Last position
                    if time.time() - start_time >= time_limit:
                        break
                    
                    new_tour = self.two_opt_swap(best_tour, i, k)
                    new_obj = compute_total_travel_time(new_tour, self.distances, self.speeds_cube,
                                                       self.period_boundaries)
                    
                    if new_obj < best_obj:
                        best_tour = new_tour
                        best_obj = new_obj
                        improved = True
                        break
                if improved:
                    break

        return best_tour, best_obj

    def solve_with_local_search(self, time_limit=300):
        """
        Solve using initial solution then refine with local search.
        """
        start_time = time.time()
        
        # Construct initial solution greedily
        current_tour, current_obj = self.construct_greedy_solution()
        
        time_left = time_limit - (time.time() - start_time)
        if time_left <= 0:
            return current_tour, current_obj
        
        # Perform local search with 2-opt moves
        best_tour, best_obj = self.local_search_2opt(current_tour, time_left)
        
        return best_tour, best_obj


def solve_tdsp(instance_data, args, logger=None):
    """
    Main solving function for time-dependent traveling salesman problem.
    """
    solver = TDSPTSolver(instance_data)
    
    # Get the best tour and its objective value
    tour, obj_val = solver.solve_with_local_search(time_limit=args.time_limit)
    
    # Log best objective if available
    if logger and obj_val < float('inf'):
        logger.log(obj_val)
    
    # Return the solution in the required format
    solution = {
        "objective_value": obj_val,
        "tour": tour
    }
    
    return solution


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    # Load the instance data
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)

    # Initialize logger if path provided
    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize")
    else:
        logger = None
        
    # Solve the instance
    start_time = time.time()
    try:
        solution = solve_tdsp(instance_data, args, logger)
        solve_time = time.time() - start_time
        
        # Write the solution
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f)
        
        print(f"Solved in {solve_time:.2f}s")
        
    except Exception as e:
        print(f"Error solving instance: {str(e)}")
        raise


if __name__ == '__main__':
    main()